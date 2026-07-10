// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <device/vector_core/gsi_libgal.h>

#include <g2_64vl_types.h>
namespace gsi::g2_64vl {
void copy(const L1Vectors &, const MmbVectors &);
void copy(const MmbVectors &, const L1Vectors &);
void shift_right(const MmbVectors &, uint32_t);
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

struct ShiftRightViews {
  L1Vectors src;
  L1Vectors out;
  MmbVectors src_out_mmb;

  explicit ShiftRightViews(const ApuG2ShiftRightParams &p)
      : src(kBits, gsi::G2_TYPES::UINT, kBits, kVectors, p.src_l1_row),
        out(kBits, gsi::G2_TYPES::UINT, kBits, kVectors, p.out_l1_row),
        src_out_mmb(kBits, gsi::G2_TYPES::UINT, kVectors, 24) {}
};

inline void shift_right_loaded(const ShiftRightViews &v, uint32_t shift) {
  copy(v.src, v.src_out_mmb);
  shift_right(v.src_out_mmb, shift);
  copy(v.src_out_mmb, v.out);
}

template <typename Body> uint64_t time_fragment(Body body) {
  gsi_perf_start();
  body();
  seu_barrier();
  return gsi_perf_end();
}

} // namespace

GSI_LIBRARY_ENTRY_POINT(tenon_apu_g2_u16_shift_right, uint64_t params_l5) {
  const auto *p = reinterpret_cast<const ApuG2ShiftRightParams *>(
      static_cast<uintptr_t>(params_l5));
  if (p == nullptr || p->timings_l5 == 0 || p->repetitions == 0 ||
      p->shift > 15)
    return GSI_STATUS_GENERAL_ERROR;

  const ShiftRightViews v(*p);
  ApuG2ShiftRightTimings timings{};

  timings.pipeline_ticks = time_fragment([&]() {
    for (uint32_t i = 0; i < p->repetitions; ++i)
      shift_right_loaded(v, p->shift);
  });

  timings.final_pipeline_ticks =
      time_fragment([&]() { shift_right_loaded(v, p->shift); });

  auto *result = reinterpret_cast<ApuG2ShiftRightTimings *>(
      static_cast<uintptr_t>(p->timings_l5));
  *result = timings;
  gsi_arc_dcache_flush_mlines(result, sizeof(*result));
  return GSI_STATUS_SUCCESS;
}
