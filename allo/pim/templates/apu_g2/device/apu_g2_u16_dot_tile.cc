// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <device/vector_core/gsi_libgal.h>

#include <g2_64vl_types.h>
namespace gsi::g2_64vl {
void copy(const L1Vectors &, const MmbVectors &);
void copy(const MmbVectors &, const L1Vectors &);
void copy(uint64_t, const MmbVectors &);
void mul(const MmbVectors_seg0 &, const MmbVectors_seg0 &,
         const MmbVectors_seg1 &);
void sum(const MmbVectors_seg0 &, uint32_t, const MmbVectors_seg1 &);
void shift_left(const MmbVectors &, uint32_t);
void add(const L1Vectors &, const MmbVectors &, const MmbVectors &);
void add(const MmbVectors_seg0 &, const MmbVectors_seg1 &, const MmbVectors &);
void seu_barrier(void);
} // namespace gsi::g2_64vl

#include <gsi_arc_cache.h>
#include <gsi_perf.h>

#include "apu_g2_params.h"
#include "gsi_library.h"

namespace {

using namespace gsi::g2_64vl;

constexpr uint32_t kVectors = 4;
constexpr uint32_t kValueBits = 16;
constexpr uint32_t kByteBits = 8;
constexpr uint32_t kProductScratchRow = 384;
constexpr uint32_t kDotScratchRow = 448;
constexpr uint32_t kScaledAccumulatorRow = 512;

struct DotTileViews {
  L1Vectors left_low;
  L1Vectors left_high;
  L1Vectors right_low;
  L1Vectors right_high;
  L1Vectors accumulator;
  L1Vectors product_scratch;
  L1Vectors dot;
  L1Vectors scaled_accumulator;
  L1Vectors out;

  MmbVectors_seg0 operand0;
  MmbVectors_seg0 operand1;
  MmbVectors_seg0 sum_source;
  MmbVectors_seg1 product;
  MmbVectors_seg1 sum_destination;
  MmbVectors sum_low16;

  explicit DotTileViews(const ApuG2DotTileParams &p)
      : left_low(kByteBits, gsi::G2_TYPES::UINT, kValueBits, kVectors,
                 p.left_l1_row),
        left_high(kByteBits, gsi::G2_TYPES::UINT, kValueBits, kVectors,
                  p.left_l1_row + kByteBits),
        right_low(kByteBits, gsi::G2_TYPES::UINT, kValueBits, kVectors,
                  p.right_l1_row),
        right_high(kByteBits, gsi::G2_TYPES::UINT, kValueBits, kVectors,
                   p.right_l1_row + kByteBits),
        accumulator(kValueBits, gsi::G2_TYPES::UINT, kValueBits, kVectors,
                    p.accumulator_l1_row),
        product_scratch(kValueBits, gsi::G2_TYPES::UINT, kValueBits, kVectors,
                        kProductScratchRow),
        dot(kValueBits, gsi::G2_TYPES::UINT, kValueBits, kVectors,
            kDotScratchRow),
        scaled_accumulator(kValueBits, gsi::G2_TYPES::UINT, kValueBits,
                           kVectors, kScaledAccumulatorRow),
        out(kValueBits, gsi::G2_TYPES::UINT, kValueBits, kVectors,
            p.out_l1_row),
        operand0(kByteBits, gsi::G2_TYPES::UINT, kVectors, 0),
        operand1(kByteBits, gsi::G2_TYPES::UINT, kVectors, 8),
        sum_source(kValueBits, gsi::G2_TYPES::UINT, kVectors, 0),
        product(kValueBits, gsi::G2_TYPES::UINT, kVectors, 24),
        sum_destination(kValueBits + p.log_block_size, gsi::G2_TYPES::UINT,
                        kVectors, 24),
        sum_low16(kValueBits, gsi::G2_TYPES::UINT, kVectors, 24) {}
};

void multiply_reduce(const DotTileViews &v, const L1Vectors &left_byte,
                     const L1Vectors &right_byte, uint32_t log_block_size) {
  copy(left_byte, v.operand0);
  copy(right_byte, v.operand1);
  mul(v.operand0, v.operand1, v.product);
  copy(v.product, v.product_scratch);
  copy(v.product_scratch, v.sum_source);
  sum(v.sum_source, log_block_size, v.sum_destination);
}

void dot_tile(const DotTileViews &v, uint32_t log_block_size,
              const L1Vectors &destination) {
  multiply_reduce(v, v.left_low, v.right_low, log_block_size);
  copy(v.sum_low16, destination);

  multiply_reduce(v, v.left_low, v.right_high, log_block_size);
  shift_left(v.sum_low16, kByteBits);
  add(destination, v.sum_low16, v.sum_low16);
  copy(v.sum_low16, destination);

  multiply_reduce(v, v.left_high, v.right_low, log_block_size);
  shift_left(v.sum_low16, kByteBits);
  add(destination, v.sum_low16, v.sum_low16);
  copy(v.sum_low16, destination);
}

void multiply_scalar_u16(const DotTileViews &v, const L1Vectors &value,
                         uint32_t scalar, const L1Vectors &destination) {
  const L1Vectors value_low(kByteBits, gsi::G2_TYPES::UINT, kValueBits,
                            kVectors, value.l1_start_row);
  const L1Vectors value_high(kByteBits, gsi::G2_TYPES::UINT, kValueBits,
                             kVectors, value.l1_start_row + kByteBits);
  const uint32_t scalar_low = scalar & 0xffu;
  const uint32_t scalar_high = (scalar >> kByteBits) & 0xffu;

  copy(value_low, v.operand0);
  copy(scalar_low, v.operand1);
  mul(v.operand0, v.operand1, v.product);
  copy(v.product, destination);

  copy(value_low, v.operand0);
  copy(scalar_high, v.operand1);
  mul(v.operand0, v.operand1, v.product);
  shift_left(v.product, kByteBits);
  add(destination, v.product, v.product);
  copy(v.product, destination);

  copy(value_high, v.operand0);
  copy(scalar_low, v.operand1);
  mul(v.operand0, v.operand1, v.product);
  shift_left(v.product, kByteBits);
  add(destination, v.product, v.product);
  copy(v.product, destination);
}

void epilogue(const DotTileViews &v, uint32_t alpha, uint32_t beta) {
  multiply_scalar_u16(v, v.dot, alpha, v.out);
  multiply_scalar_u16(v, v.accumulator, beta, v.scaled_accumulator);
  copy(v.out, v.sum_source);
  copy(v.scaled_accumulator, v.product);
  add(v.sum_source, v.product, v.product);
  copy(v.product, v.out);
}

void pipeline(const DotTileViews &v, const ApuG2DotTileParams &p) {
  if (p.enable_epilogue != 0) {
    dot_tile(v, p.log_block_size, v.dot);
    epilogue(v, p.alpha, p.beta);
  } else {
    dot_tile(v, p.log_block_size, v.out);
  }
}

template <typename Body> uint64_t time_fragment(Body body) {
  gsi_perf_start();
  body();
  seu_barrier();
  return gsi_perf_end();
}

} // namespace

GSI_LIBRARY_ENTRY_POINT(tenon_apu_g2_u16_dot_tile, uint64_t params_l5) {
  const auto *p = reinterpret_cast<const ApuG2DotTileParams *>(
      static_cast<uintptr_t>(params_l5));
  if (p == nullptr || p->timings_l5 == 0 || p->repetitions == 0 ||
      p->log_block_size > 8)
    return GSI_STATUS_GENERAL_ERROR;

  const DotTileViews v(*p);
  ApuG2DotTileTimings timings{};
  timings.pipeline_ticks = time_fragment([&]() {
    for (uint32_t i = 0; i < p->repetitions; ++i)
      pipeline(v, *p);
  });
  timings.final_pipeline_ticks = time_fragment([&]() { pipeline(v, *p); });

  auto *result = reinterpret_cast<ApuG2DotTileTimings *>(
      static_cast<uintptr_t>(p->timings_l5));
  *result = timings;
  gsi_arc_dcache_flush_mlines(result, sizeof(*result));
  return GSI_STATUS_SUCCESS;
}
