// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <device/vector_core/gsi_libgal.h>

#include <g2_64vl_types.h>
namespace gsi::g2_64vl {
void copy(uint64_t, const MmbVectors &);
void copy(const MmbVectors &, const L1Vectors &);
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

struct FillViews {
  L1Vectors out;
  MmbVectors out_mmb;

  explicit FillViews(const ApuG2FillParams &p)
      : out(kBits, gsi::G2_TYPES::UINT, kBits, kVectors, p.out_l1_row),
        out_mmb(kBits, gsi::G2_TYPES::UINT, kVectors, 24) {}
};

inline void fill_loaded(const FillViews &v, uint32_t value) {
  copy(static_cast<uint64_t>(value & 0xffffu), v.out_mmb);
  copy(v.out_mmb, v.out);
}

template <typename Body> uint64_t time_fragment(Body body) {
  gsi_perf_start();
  body();
  seu_barrier();
  return gsi_perf_end();
}

} // namespace

GSI_LIBRARY_ENTRY_POINT(tenon_apu_g2_u16_fill, uint64_t params_l5) {
  const auto *p = reinterpret_cast<const ApuG2FillParams *>(
      static_cast<uintptr_t>(params_l5));
  if (p == nullptr || p->timings_l5 == 0 || p->repetitions == 0 ||
      p->value > 0xffffu)
    return GSI_STATUS_GENERAL_ERROR;

  const FillViews v(*p);
  ApuG2FillTimings timings{};

  timings.pipeline_ticks = time_fragment([&]() {
    for (uint32_t i = 0; i < p->repetitions; ++i)
      fill_loaded(v, p->value);
  });

  timings.final_pipeline_ticks = time_fragment([&]() { fill_loaded(v, p->value); });

  auto *result = reinterpret_cast<ApuG2FillTimings *>(
      static_cast<uintptr_t>(p->timings_l5));
  *result = timings;
  gsi_arc_dcache_flush_mlines(result, sizeof(*result));
  return GSI_STATUS_SUCCESS;
}
