// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <device/vector_core/gsi_libgal.h>

// gsi-gtml-g2 0.0.1 does not package g2_64vl_apl_defs.h, which the public
// umbrella header includes for ARC builds.  The descriptor definitions and
// exported library ABI are installed, so declare only the primitives used by
// this task.  This block can go away when the vendor package is complete.
#include <g2_64vl_types.h>
namespace gsi::g2_64vl {
void copy(const L1Vectors &, const MmbVectors &);
void copy(const MmbVectors &, const L1Vectors &);
void add(const MmbVectors_seg0 &, const MmbVectors_seg1 &, const MmbVectors &);
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

struct AddViews {
  L1Vectors lhs;
  L1Vectors rhs;
  L1Vectors out;
  MmbVectors_seg0 lhs_mmb;
  MmbVectors_seg1 rhs_mmb;
  MmbVectors out_mmb;

  explicit AddViews(const ApuG2AddParams &p)
      : lhs(kBits, gsi::G2_TYPES::UINT, kBits, kVectors, p.lhs_l1_row),
        rhs(kBits, gsi::G2_TYPES::UINT, kBits, kVectors, p.rhs_l1_row),
        out(kBits, gsi::G2_TYPES::UINT, kBits, kVectors, p.out_l1_row),
        lhs_mmb(kBits, gsi::G2_TYPES::UINT, kVectors, 0),
        rhs_mmb(kBits, gsi::G2_TYPES::UINT, kVectors, 24),
        out_mmb(kBits, gsi::G2_TYPES::UINT, kVectors, 24) {}
};

inline void load_inputs(const AddViews &v) {
  copy(v.lhs, v.lhs_mmb);
  copy(v.rhs, v.rhs_mmb);
}

inline void add_loaded(const AddViews &v) {
  // A 16-bit UINT destination intentionally discards carry, implementing
  // the modulo-2^16 semantics of NumPy/C uint16 addition.
  add(v.lhs_mmb, v.rhs_mmb, v.out_mmb);
}

inline void store_output(const AddViews &v) { copy(v.out_mmb, v.out); }

template <typename Body> uint64_t time_fragment(Body body) {
  gsi_perf_start();
  body();
  seu_barrier();
  return gsi_perf_end();
}

} // namespace

GSI_LIBRARY_ENTRY_POINT(tenon_apu_g2_u16_add, uint64_t params_l5) {
  const auto *p = reinterpret_cast<const ApuG2AddParams *>(
      static_cast<uintptr_t>(params_l5));
  if (p == nullptr || p->timings_l5 == 0 || p->repetitions == 0)
    return GSI_STATUS_GENERAL_ERROR;

  const AddViews v(*p);
  ApuG2AddTimings timings{};

  timings.input_copy_ticks = time_fragment([&]() {
    for (uint32_t i = 0; i < p->repetitions; ++i)
      load_inputs(v);
  });

  // Initialize both operands before timing ADD in isolation.  The result
  // aliases segment-1 storage, so repeated ADDs form a dependency chain;
  // that is intentional and avoids counting reloads in the ADD phase.
  load_inputs(v);
  seu_barrier();
  timings.add_ticks = time_fragment([&]() {
    for (uint32_t i = 0; i < p->repetitions; ++i)
      add_loaded(v);
  });

  timings.output_copy_ticks = time_fragment([&]() {
    for (uint32_t i = 0; i < p->repetitions; ++i)
      store_output(v);
  });

  // This is the primary measured kernel: every repetition reloads both
  // operands, performs one coalesced VL64 ADD, and writes one vector pack.
  timings.pipeline_ticks = time_fragment([&]() {
    for (uint32_t i = 0; i < p->repetitions; ++i) {
      load_inputs(v);
      add_loaded(v);
      store_output(v);
    }
  });

  // Leave a result whose correctness is independent of any phase-timing
  // dependency chain, and include the required task-boundary barrier.
  timings.final_pipeline_ticks = time_fragment([&]() {
    load_inputs(v);
    add_loaded(v);
    store_output(v);
  });

  auto *result = reinterpret_cast<ApuG2AddTimings *>(
      static_cast<uintptr_t>(p->timings_l5));
  *result = timings;
  gsi_arc_dcache_flush_mlines(result, sizeof(*result));
  return GSI_STATUS_SUCCESS;
}
