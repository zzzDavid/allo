// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <device/vector_core/gsi_libgal.h>

#include <g2_64vl_types.h>
namespace gsi::g2_64vl {
void copy(const L1Vectors &, const MmbVectors &);
void copy(const L1Vectors &, const MmbVectors &, const MmbMarkers &);
void copy(const MmbVectors &, const L1Vectors &);
void lt(const MmbVectors_seg0 &, const MmbVectors_seg1 &, const MmbMarkers &);
void seu_barrier(void);
} // namespace gsi::g2_64vl

#include <gsi_arc_cache.h>
#include <gsi_perf.h>

#include "apu_g2_params.h"
#include "gsi_library.h"

namespace {

using namespace gsi::g2_64vl;

constexpr uint32_t kBits = 16;
constexpr uint32_t kVectors = 4;

struct SelectViews {
  L1Vectors lhs;
  L1Vectors rhs;
  L1Vectors true_value;
  L1Vectors false_value;
  L1Vectors out;
  MmbVectors_seg0 lhs_mmb;
  MmbVectors_seg1 rhs_out_mmb;
  MmbMarkers_seg0 predicate;

  explicit SelectViews(const ApuG2SelectParams &p)
      : lhs(kBits, gsi::G2_TYPES::UINT, kBits, kVectors, p.lhs_l1_row),
        rhs(kBits, gsi::G2_TYPES::UINT, kBits, kVectors, p.rhs_l1_row),
        true_value(kBits, gsi::G2_TYPES::UINT, kBits, kVectors,
                   p.true_l1_row),
        false_value(kBits, gsi::G2_TYPES::UINT, kBits, kVectors,
                    p.false_l1_row),
        out(kBits, gsi::G2_TYPES::UINT, kBits, kVectors, p.out_l1_row),
        lhs_mmb(kBits, gsi::G2_TYPES::UINT, kVectors, 0),
        rhs_out_mmb(kBits, gsi::G2_TYPES::UINT, kVectors, 24),
        predicate(kVectors, 23) {}
};

inline void select_loaded(const SelectViews &v) {
  copy(v.lhs, v.lhs_mmb);
  copy(v.rhs, v.rhs_out_mmb);
  lt(v.lhs_mmb, v.rhs_out_mmb, v.predicate);
  copy(v.false_value, v.rhs_out_mmb);
  copy(v.true_value, v.rhs_out_mmb, v.predicate);
  copy(v.rhs_out_mmb, v.out);
}

template <typename Body> uint64_t time_fragment(Body body) {
  gsi_perf_start();
  body();
  seu_barrier();
  return gsi_perf_end();
}

} // namespace

GSI_LIBRARY_ENTRY_POINT(tenon_apu_g2_u16_select_lt, uint64_t params_l5) {
  const auto *p = reinterpret_cast<const ApuG2SelectParams *>(
      static_cast<uintptr_t>(params_l5));
  if (p == nullptr || p->timings_l5 == 0 || p->repetitions == 0)
    return GSI_STATUS_GENERAL_ERROR;

  const SelectViews v(*p);
  ApuG2SelectTimings timings{};

  timings.pipeline_ticks = time_fragment([&]() {
    for (uint32_t i = 0; i < p->repetitions; ++i)
      select_loaded(v);
  });

  timings.final_pipeline_ticks = time_fragment([&]() { select_loaded(v); });

  auto *result = reinterpret_cast<ApuG2SelectTimings *>(
      static_cast<uintptr_t>(p->timings_l5));
  *result = timings;
  gsi_arc_dcache_flush_mlines(result, sizeof(*result));
  return GSI_STATUS_SUCCESS;
}
