// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <device/vector_core/gsi_libgal.h>

#include <g2_64vl_types.h>
namespace gsi::g2_64vl {
void copy(const L1Vectors &, const MmbVectors &);
void copy(const MmbVectors &, const L1Vectors &);
void sum(const MmbVectors_seg0 &, uint32_t, const MmbVectors_seg1 &);
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

struct BlockSumViews {
  L1Vectors src;
  L1Vectors out;
  MmbVectors_seg0 sum_source;
  MmbVectors_seg1 sum_destination;
  MmbVectors sum_low16;

  explicit BlockSumViews(const ApuG2BlockSumParams &p)
      : src(kBits, gsi::G2_TYPES::UINT, kBits, kVectors, p.src_l1_row),
        out(kBits, gsi::G2_TYPES::UINT, kBits, kVectors, p.out_l1_row),
        sum_source(kBits, gsi::G2_TYPES::UINT, kVectors, 0),
        sum_destination(kBits + p.log_block_size, gsi::G2_TYPES::UINT,
                        kVectors, 24),
        sum_low16(kBits, gsi::G2_TYPES::UINT, kVectors, 24) {}
};

inline void block_sum_loaded(const BlockSumViews &v, uint32_t log_block_size) {
  copy(v.src, v.sum_source);
  sum(v.sum_source, log_block_size, v.sum_destination);
  copy(v.sum_low16, v.out);
}

template <typename Body> uint64_t time_fragment(Body body) {
  gsi_perf_start();
  body();
  seu_barrier();
  return gsi_perf_end();
}

} // namespace

GSI_LIBRARY_ENTRY_POINT(tenon_apu_g2_u16_block_sum, uint64_t params_l5) {
  const auto *p = reinterpret_cast<const ApuG2BlockSumParams *>(
      static_cast<uintptr_t>(params_l5));
  if (p == nullptr || p->timings_l5 == 0 || p->repetitions == 0 ||
      p->log_block_size > 8)
    return GSI_STATUS_GENERAL_ERROR;

  const BlockSumViews v(*p);
  ApuG2BlockSumTimings timings{};

  timings.pipeline_ticks = time_fragment([&]() {
    for (uint32_t i = 0; i < p->repetitions; ++i)
      block_sum_loaded(v, p->log_block_size);
  });

  timings.final_pipeline_ticks =
      time_fragment([&]() { block_sum_loaded(v, p->log_block_size); });

  auto *result = reinterpret_cast<ApuG2BlockSumTimings *>(
      static_cast<uintptr_t>(p->timings_l5));
  *result = timings;
  gsi_arc_dcache_flush_mlines(result, sizeof(*result));
  return GSI_STATUS_SUCCESS;
}
