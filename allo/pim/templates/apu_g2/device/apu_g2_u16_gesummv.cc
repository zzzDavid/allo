// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <device/vector_core/gsi_libgal.h>

// The installed SDK omits the hardware-only header transitively included by
// g2_64vl.h.  Use the packaged descriptors and declare the exported ABI used
// by this direct-VL64 task.
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
void add(const MmbVectors_seg0 &, const MmbVectors_seg1 &, const MmbVectors &);
void copy_from_odd_to_even_vectors(const MmbVectors &, const MmbVectors &);
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
constexpr uint32_t kScaledScratchRow = 704;

struct GesummvViews {
  L1Vectors matrices_low;
  L1Vectors matrices_high;
  L1Vector x_low;
  L1Vector x_high;
  L1Vectors scalars_low;
  L1Vectors scalars_high;
  L1Vectors product_scratch;
  L1Vectors tmp;
  L1Vectors scaled;
  L1Vectors y;

  MmbVectors_seg0 operand0;
  MmbVectors_seg0 operand1;
  MmbVectors_seg0 sum_source;
  MmbVectors_seg1 product;
  MmbVectors_seg1 sum_destination;
  MmbVectors sum_low16;

  explicit GesummvViews(const ApuG2GesummvParams &p)
      : matrices_low(kByteBits, gsi::G2_TYPES::UINT, kValueBits, kVectors,
                     p.matrices_l1_row),
        matrices_high(kByteBits, gsi::G2_TYPES::UINT, kValueBits, kVectors,
                      p.matrices_l1_row + kByteBits),
        x_low(kByteBits, gsi::G2_TYPES::UINT, p.x_l1_row),
        x_high(kByteBits, gsi::G2_TYPES::UINT, p.x_l1_row + kByteBits),
        scalars_low(kByteBits, gsi::G2_TYPES::UINT, kValueBits, kVectors,
                    p.scalars_l1_row),
        scalars_high(kByteBits, gsi::G2_TYPES::UINT, kValueBits, kVectors,
                     p.scalars_l1_row + kByteBits),
        product_scratch(kValueBits, gsi::G2_TYPES::UINT, kValueBits, kVectors,
                        kProductScratchRow),
        tmp(kValueBits, gsi::G2_TYPES::UINT, kValueBits, kVectors,
            p.tmp_l1_row),
        scaled(kValueBits, gsi::G2_TYPES::UINT, kValueBits, kVectors,
               kScaledScratchRow),
        y(kValueBits, gsi::G2_TYPES::UINT, kValueBits, kVectors, p.y_l1_row),
        operand0(kByteBits, gsi::G2_TYPES::UINT, kVectors, 0),
        operand1(kByteBits, gsi::G2_TYPES::UINT, kVectors, 8),
        sum_source(kValueBits, gsi::G2_TYPES::UINT, kVectors, 0),
        product(kValueBits, gsi::G2_TYPES::UINT, kVectors, 24),
        sum_destination(kValueBits + p.log_block_size, gsi::G2_TYPES::UINT,
                        kVectors, 24),
        sum_low16(kValueBits, gsi::G2_TYPES::UINT, kVectors, 24) {}
};

void multiply_reduce(const GesummvViews &v, const L1Vectors &matrix_byte,
                     const L1Vector &x_byte, uint32_t log_block_size) {
  copy(matrix_byte, v.operand0);
  copy(x_byte, v.operand1);
  mul(v.operand0, v.operand1, v.product);

  // SUM requires its source in MMB segment 0.  This L1 bounce is the
  // hardware-proven, segment-safe route from the multiply result.
  copy(v.product, v.product_scratch);
  copy(v.product_scratch, v.sum_source);
  sum(v.sum_source, log_block_size, v.sum_destination);
}

void gemv_pair(const GesummvViews &v, uint32_t log_block_size) {
  // A and B occupy MMB sets 0 and 1, so all three multiply/reduction calls
  // compute the two GEMVs concurrently.  For unsigned 16-bit values:
  // low(a*b) = a0*b0 + ((a0*b1 + a1*b0) << 8) modulo 2^16.
  multiply_reduce(v, v.matrices_low, v.x_low, log_block_size);
  copy(v.sum_low16, v.tmp);

  multiply_reduce(v, v.matrices_low, v.x_high, log_block_size);
  shift_left(v.sum_low16, kByteBits);
  add(v.tmp, v.sum_low16, v.sum_low16);
  copy(v.sum_low16, v.tmp);

  multiply_reduce(v, v.matrices_high, v.x_low, log_block_size);
  shift_left(v.sum_low16, kByteBits);
  add(v.tmp, v.sum_low16, v.sum_low16);
  copy(v.sum_low16, v.tmp);
}

void scale_pair(const GesummvViews &v) {
  const L1Vectors tmp_low(kByteBits, gsi::G2_TYPES::UINT, kValueBits, kVectors,
                          v.tmp.l1_start_row);
  const L1Vectors tmp_high(kByteBits, gsi::G2_TYPES::UINT, kValueBits, kVectors,
                           v.tmp.l1_start_row + kByteBits);

  copy(tmp_low, v.operand0);
  copy(v.scalars_low, v.operand1);
  mul(v.operand0, v.operand1, v.product);
  copy(v.product, v.scaled);

  copy(tmp_low, v.operand0);
  copy(v.scalars_high, v.operand1);
  mul(v.operand0, v.operand1, v.product);
  shift_left(v.product, kByteBits);
  add(v.scaled, v.product, v.product);
  copy(v.product, v.scaled);

  copy(tmp_high, v.operand0);
  copy(v.scalars_low, v.operand1);
  mul(v.operand0, v.operand1, v.product);
  shift_left(v.product, kByteBits);
  add(v.scaled, v.product, v.product);
  copy(v.product, v.scaled);
}

void combine_pair(const GesummvViews &v) {
  copy(v.scaled, v.sum_source);
  // This primitive copies set 1 -> set 0 (and set 3 -> set 2).
  copy_from_odd_to_even_vectors(v.sum_source, v.product);
  add(v.sum_source, v.product, v.product);
  copy(v.product, v.y);
}

void scale_and_combine(const GesummvViews &v) {
  scale_pair(v);
  combine_pair(v);
}

void full_pipeline(const GesummvViews &v, uint32_t log_block_size) {
  gemv_pair(v, log_block_size);
  scale_and_combine(v);
}

template <typename Body> uint64_t time_fragment(Body body) {
  gsi_perf_start();
  body();
  seu_barrier();
  return gsi_perf_end();
}

} // namespace

GSI_LIBRARY_ENTRY_POINT(tenon_apu_g2_u16_gesummv, uint64_t params_l5) {
  const auto *p = reinterpret_cast<const ApuG2GesummvParams *>(
      static_cast<uintptr_t>(params_l5));
  if (p == nullptr || p->timings_l5 == 0 || p->repetitions == 0 ||
      p->log_block_size > 8)
    return GSI_STATUS_GENERAL_ERROR;

  const GesummvViews v(*p);
  ApuG2GesummvTimings timings{};

  timings.gemv_ticks = time_fragment([&]() {
    for (uint32_t i = 0; i < p->repetitions; ++i)
      gemv_pair(v, p->log_block_size);
  });

  gemv_pair(v, p->log_block_size);
  seu_barrier();
  timings.scale_combine_ticks = time_fragment([&]() {
    for (uint32_t i = 0; i < p->repetitions; ++i)
      scale_and_combine(v);
  });

  timings.pipeline_ticks = time_fragment([&]() {
    for (uint32_t i = 0; i < p->repetitions; ++i)
      full_pipeline(v, p->log_block_size);
  });

  // Leave independently computed tmp/y packs for host readback.
  timings.final_pipeline_ticks =
      time_fragment([&]() { full_pipeline(v, p->log_block_size); });

  auto *result = reinterpret_cast<ApuG2GesummvTimings *>(
      static_cast<uintptr_t>(p->timings_l5));
  *result = timings;
  gsi_arc_dcache_flush_mlines(result, sizeof(*result));
  return GSI_STATUS_SUCCESS;
}
