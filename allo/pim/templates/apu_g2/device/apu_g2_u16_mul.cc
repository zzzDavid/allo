// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <device/vector_core/gsi_libgal.h>

#include <g2_64vl_types.h>
namespace gsi::g2_64vl {
void copy(const L1Vectors &, const MmbVectors &);
void copy(const MmbVectors &, const L1Vectors &);
void mul(const MmbVectors_seg0 &, const MmbVectors_seg0 &,
         const MmbVectors_seg1 &);
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

constexpr uint32_t kBits = 16;
constexpr uint32_t kByteBits = 8;
constexpr uint32_t kVectors = 4;

struct MulViews {
  L1Vectors lhs_low;
  L1Vectors lhs_high;
  L1Vectors rhs_low;
  L1Vectors rhs_high;
  L1Vectors out;
  MmbVectors_seg0 operand0;
  MmbVectors_seg0 operand1;
  MmbVectors_seg1 product;

  explicit MulViews(const ApuG2MulParams &p)
      : lhs_low(kByteBits, gsi::G2_TYPES::UINT, kBits, kVectors,
                p.lhs_l1_row),
        lhs_high(kByteBits, gsi::G2_TYPES::UINT, kBits, kVectors,
                 p.lhs_l1_row + kByteBits),
        rhs_low(kByteBits, gsi::G2_TYPES::UINT, kBits, kVectors,
                p.rhs_l1_row),
        rhs_high(kByteBits, gsi::G2_TYPES::UINT, kBits, kVectors,
                 p.rhs_l1_row + kByteBits),
        out(kBits, gsi::G2_TYPES::UINT, kBits, kVectors, p.out_l1_row),
        operand0(kByteBits, gsi::G2_TYPES::UINT, kVectors, 0),
        operand1(kByteBits, gsi::G2_TYPES::UINT, kVectors, 8),
        product(kBits, gsi::G2_TYPES::UINT, kVectors, 24) {}
};

inline void multiply_byte_pair(const MulViews &v, const L1Vectors &lhs_byte,
                               const L1Vectors &rhs_byte) {
  copy(lhs_byte, v.operand0);
  copy(rhs_byte, v.operand1);
  mul(v.operand0, v.operand1, v.product);
}

inline void mul_loaded(const MulViews &v) {
  // lo*lo
  multiply_byte_pair(v, v.lhs_low, v.rhs_low);
  copy(v.product, v.out);

  // (lo*hi) << 8
  multiply_byte_pair(v, v.lhs_low, v.rhs_high);
  shift_left(v.product, kByteBits);
  add(v.out, v.product, v.product);
  copy(v.product, v.out);

  // (hi*lo) << 8
  multiply_byte_pair(v, v.lhs_high, v.rhs_low);
  shift_left(v.product, kByteBits);
  add(v.out, v.product, v.product);
  copy(v.product, v.out);
}

template <typename Body> uint64_t time_fragment(Body body) {
  gsi_perf_start();
  body();
  seu_barrier();
  return gsi_perf_end();
}

} // namespace

GSI_LIBRARY_ENTRY_POINT(tenon_apu_g2_u16_mul, uint64_t params_l5) {
  const auto *p = reinterpret_cast<const ApuG2MulParams *>(
      static_cast<uintptr_t>(params_l5));
  if (p == nullptr || p->timings_l5 == 0 || p->repetitions == 0)
    return GSI_STATUS_GENERAL_ERROR;

  const MulViews v(*p);
  ApuG2MulTimings timings{};

  timings.pipeline_ticks = time_fragment([&]() {
    for (uint32_t i = 0; i < p->repetitions; ++i)
      mul_loaded(v);
  });

  timings.final_pipeline_ticks = time_fragment([&]() { mul_loaded(v); });

  auto *result = reinterpret_cast<ApuG2MulTimings *>(
      static_cast<uintptr_t>(p->timings_l5));
  *result = timings;
  gsi_arc_dcache_flush_mlines(result, sizeof(*result));
  return GSI_STATUS_SUCCESS;
}
