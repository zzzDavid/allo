// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <device/vector_core/gsi_libgal.h>
#include <g2_64vl_types.h>
namespace gsi::g2_64vl {
void copy(const L1Vectors &, const MmbVectors &);
void copy(const L1Vector &, const MmbVectors &);
void copy(const MmbVectors &, const L1Vectors &);
void mul(const MmbVectors_seg0 &, const MmbVectors_seg0 &,
         const MmbVectors_seg1 &);
void sum(const MmbVectors_seg0 &, uint32_t, const MmbVectors_seg1 &);
void shift_left(const MmbVectors &, uint32_t);
void add(const L1Vectors &, const MmbVectors &, const MmbVectors &);
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
constexpr uint32_t kProductScratchRow = 576;

struct GemvViews {
  L1Vectors matrix_low;
  L1Vectors matrix_high;
  L1Vector x_low;
  L1Vector x_high;
  L1Vectors accumulator;
  L1Vectors product_scratch;
  L1Vectors tmp;
  L1Vectors out;

  MmbVectors_seg0 operand0;
  MmbVectors_seg0 operand1;
  MmbVectors_seg0 sum_source;
  MmbVectors_seg1 product;
  MmbVectors_seg1 sum_destination;
  MmbVectors sum_low16;

  explicit GemvViews(const ApuG2GemvParams &p)
      : matrix_low(kByteBits, gsi::G2_TYPES::UINT, kValueBits, kVectors,
                   p.matrix_l1_row),
        matrix_high(kByteBits, gsi::G2_TYPES::UINT, kValueBits, kVectors,
                    p.matrix_l1_row + kByteBits),
        x_low(kByteBits, gsi::G2_TYPES::UINT, p.x_l1_row),
        x_high(kByteBits, gsi::G2_TYPES::UINT, p.x_l1_row + kByteBits),
        accumulator(kValueBits, gsi::G2_TYPES::UINT, kValueBits, kVectors,
                    p.accumulator_l1_row),
        product_scratch(kValueBits, gsi::G2_TYPES::UINT, kValueBits, kVectors,
                        kProductScratchRow),
        tmp(kValueBits, gsi::G2_TYPES::UINT, kValueBits, kVectors,
            p.tmp_l1_row),
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

void multiply_reduce(const GemvViews &v, const L1Vectors &matrix_byte,
                     const L1Vector &x_byte, uint32_t log_block_size) {
  copy(matrix_byte, v.operand0);
  copy(x_byte, v.operand1);
  mul(v.operand0, v.operand1, v.product);
  copy(v.product, v.product_scratch);
  copy(v.product_scratch, v.sum_source);
  sum(v.sum_source, log_block_size, v.sum_destination);
}

void gemv_product(const GemvViews &v, uint32_t log_block_size) {
  // low16(a*x) = a0*x0 + ((a0*x1 + a1*x0) << 8), modulo 2^16.
  multiply_reduce(v, v.matrix_low, v.x_low, log_block_size);
  copy(v.sum_low16, v.tmp);

  multiply_reduce(v, v.matrix_low, v.x_high, log_block_size);
  shift_left(v.sum_low16, kByteBits);
  add(v.tmp, v.sum_low16, v.sum_low16);
  copy(v.sum_low16, v.tmp);

  multiply_reduce(v, v.matrix_high, v.x_low, log_block_size);
  shift_left(v.sum_low16, kByteBits);
  add(v.tmp, v.sum_low16, v.sum_low16);
  copy(v.sum_low16, v.tmp);
}

void full_pipeline(const GemvViews &v, uint32_t log_block_size) {
  gemv_product(v, log_block_size);
  // Preserve ordinary Allo reduction-update semantics: out += matrix @ x.
  add(v.accumulator, v.sum_low16, v.sum_low16);
  copy(v.sum_low16, v.out);
}

template <typename Body> uint64_t time_fragment(Body body) {
  gsi_perf_start();
  body();
  seu_barrier();
  return gsi_perf_end();
}

} // namespace

GSI_LIBRARY_ENTRY_POINT(tenon_apu_g2_u16_gemv, uint64_t params_l5) {
  const auto *p = reinterpret_cast<const ApuG2GemvParams *>(
      static_cast<uintptr_t>(params_l5));
  if (p == nullptr || p->timings_l5 == 0 || p->repetitions == 0 ||
      p->log_block_size > 8)
    return GSI_STATUS_GENERAL_ERROR;

  const GemvViews v(*p);
  ApuG2GemvTimings timings{};
  timings.product_ticks = time_fragment([&]() {
    for (uint32_t i = 0; i < p->repetitions; ++i)
      gemv_product(v, p->log_block_size);
  });
  timings.pipeline_ticks = time_fragment([&]() {
    for (uint32_t i = 0; i < p->repetitions; ++i)
      full_pipeline(v, p->log_block_size);
  });
  timings.final_pipeline_ticks =
      time_fragment([&]() { full_pipeline(v, p->log_block_size); });

  auto *result = reinterpret_cast<ApuG2GemvTimings *>(
      static_cast<uintptr_t>(p->timings_l5));
  *result = timings;
  gsi_arc_dcache_flush_mlines(result, sizeof(*result));
  return GSI_STATUS_SUCCESS;
}
