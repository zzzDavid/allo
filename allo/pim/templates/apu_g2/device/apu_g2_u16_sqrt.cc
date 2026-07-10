// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <device/vector_core/gsi_libgal.h>

#include <g2_64vl_types.h>
namespace gsi::g2_64vl {
void copy(uint64_t, const MmbVectors &);
void copy(const L1Vectors &, const MmbVectors &);
void copy(const L1Vectors &, const MmbVectors &, const MmbMarkers &);
void copy(const MmbVectors &, const L1Vectors &);
void lt(const MmbVectors_seg0 &, const MmbVectors_seg1 &, const MmbMarkers &);
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
constexpr uint32_t kCandidateRow = 448;
constexpr uint32_t kSquareRow = 640;

struct SqrtViews {
  L1Vectors src;
  L1Vectors out;
  L1Vectors candidate;
  L1Vectors candidate_low;
  L1Vectors candidate_high;
  L1Vectors square;

  MmbVectors_seg0 operand0;
  MmbVectors_seg0 operand1;
  MmbVectors_seg0 compare_lhs;
  MmbVectors_seg1 work;
  MmbMarkers_seg0 predicate;

  explicit SqrtViews(const ApuG2SqrtParams &p)
      : src(kBits, gsi::G2_TYPES::UINT, kBits, kVectors, p.src_l1_row),
        out(kBits, gsi::G2_TYPES::UINT, kBits, kVectors, p.out_l1_row),
        candidate(kBits, gsi::G2_TYPES::UINT, kBits, kVectors, kCandidateRow),
        candidate_low(kByteBits, gsi::G2_TYPES::UINT, kBits, kVectors,
                      kCandidateRow),
        candidate_high(kByteBits, gsi::G2_TYPES::UINT, kBits, kVectors,
                       kCandidateRow + kByteBits),
        square(kBits, gsi::G2_TYPES::UINT, kBits, kVectors, kSquareRow),
        operand0(kByteBits, gsi::G2_TYPES::UINT, kVectors, 0),
        operand1(kByteBits, gsi::G2_TYPES::UINT, kVectors, 8),
        compare_lhs(kBits, gsi::G2_TYPES::UINT, kVectors, 0),
        work(kBits, gsi::G2_TYPES::UINT, kVectors, 24),
        predicate(kVectors, 23) {}
};

inline void fill_l1(const SqrtViews &v, const L1Vectors &dst, uint32_t value) {
  copy(static_cast<uint64_t>(value & 0xffffu), v.work);
  copy(v.work, dst);
}

inline void candidate_with_bit(const SqrtViews &v, uint32_t bit_value) {
  copy(static_cast<uint64_t>(bit_value & 0xffffu), v.work);
  add(v.out, v.work, v.work);
  copy(v.work, v.candidate);
}

inline void multiply_byte_pair(const SqrtViews &v, const L1Vectors &lhs_byte,
                               const L1Vectors &rhs_byte) {
  copy(lhs_byte, v.operand0);
  copy(rhs_byte, v.operand1);
  mul(v.operand0, v.operand1, v.work);
}

inline void square_candidate(const SqrtViews &v) {
  // candidate is at most 255, so this byte-product expansion computes the
  // exact 16-bit square.  The high*high term is zero for all valid candidates,
  // but the generic modular-u16 multiply expansion is retained for safety.
  multiply_byte_pair(v, v.candidate_low, v.candidate_low);
  copy(v.work, v.square);

  multiply_byte_pair(v, v.candidate_low, v.candidate_high);
  shift_left(v.work, kByteBits);
  add(v.square, v.work, v.work);
  copy(v.work, v.square);

  multiply_byte_pair(v, v.candidate_high, v.candidate_low);
  shift_left(v.work, kByteBits);
  add(v.square, v.work, v.work);
  copy(v.work, v.square);
}

inline void keep_candidate_if_valid(const SqrtViews &v) {
  // If src < candidate^2, keep the old root.  Otherwise accept candidate.
  copy(v.src, v.compare_lhs);
  copy(v.square, v.work);
  lt(v.compare_lhs, v.work, v.predicate);
  copy(v.candidate, v.work);
  copy(v.out, v.work, v.predicate);
  copy(v.work, v.out);
}

inline void sqrt_loaded(const SqrtViews &v) {
  fill_l1(v, v.out, 0);
  for (int bit = 7; bit >= 0; --bit) {
    candidate_with_bit(v, 1u << static_cast<uint32_t>(bit));
    square_candidate(v);
    keep_candidate_if_valid(v);
  }
}

template <typename Body> uint64_t time_fragment(Body body) {
  gsi_perf_start();
  body();
  seu_barrier();
  return gsi_perf_end();
}

} // namespace

GSI_LIBRARY_ENTRY_POINT(tenon_apu_g2_u16_sqrt, uint64_t params_l5) {
  const auto *p = reinterpret_cast<const ApuG2SqrtParams *>(
      static_cast<uintptr_t>(params_l5));
  if (p == nullptr || p->timings_l5 == 0 || p->repetitions == 0)
    return GSI_STATUS_GENERAL_ERROR;

  const SqrtViews v(*p);
  ApuG2SqrtTimings timings{};

  timings.pipeline_ticks = time_fragment([&]() {
    for (uint32_t i = 0; i < p->repetitions; ++i)
      sqrt_loaded(v);
  });

  timings.final_pipeline_ticks = time_fragment([&]() { sqrt_loaded(v); });

  auto *result = reinterpret_cast<ApuG2SqrtTimings *>(
      static_cast<uintptr_t>(p->timings_l5));
  *result = timings;
  gsi_arc_dcache_flush_mlines(result, sizeof(*result));
  return GSI_STATUS_SUCCESS;
}
