// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <device/vector_core/gsi_libgal.h>

#include <g2_64vl_types.h>
namespace gsi::g2_64vl {
void copy(const L1Vectors &, const MmbVectors &);
void copy(const MmbVectors &, const L1Vectors &);
void add(const MmbVectors_seg0 &, const MmbVectors_seg1 &, const MmbVectors &);
void mul(const MmbVectors_seg0 &, const MmbVectors_seg0 &,
         const MmbVectors_seg1 &);
void sum(const MmbVectors_seg0 &, uint32_t, const MmbVectors_seg1 &);
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
constexpr uint32_t kSegmentBits = 24;

gsi::G2_TYPES value_type(uint32_t is_signed) {
  return is_signed != 0 ? gsi::G2_TYPES::INT : gsi::G2_TYPES::UINT;
}

bool legal(const ApuG2ComposedDotParams &p) {
  if (p.timings_l5 == 0 || p.repetitions == 0 || p.lhs_bits < 2 ||
      p.rhs_bits < 2 || p.lhs_bits > kSegmentBits ||
      p.rhs_bits > kSegmentBits || p.dot_bits > kSegmentBits ||
      p.auxiliary_bits == 0 || p.auxiliary_bits > kSegmentBits ||
      p.out_bits == 0 || p.out_bits > kSegmentBits || p.lhs_signed > 1 ||
      p.rhs_signed > 1 || p.auxiliary_signed > 1 || p.out_signed > 1 ||
      p.log_reduction == 0 || p.log_reduction > 8 ||
      p.lhs_signed != p.rhs_signed || p.lhs_signed != p.out_signed ||
      p.lhs_bits + p.rhs_bits > 23 ||
      p.dot_bits != p.lhs_bits + p.rhs_bits + p.log_reduction)
    return false;

  switch (p.epilogue) {
  case APUG2_COMPOSED_DOT_IDENTITY:
    return p.out_bits == p.dot_bits;
  case APUG2_COMPOSED_DOT_PAIR_AFFINE:
    return p.auxiliary_signed == p.lhs_signed &&
           p.dot_bits + p.auxiliary_bits <= kSegmentBits &&
           p.out_bits == p.dot_bits + p.auxiliary_bits + 1;
  case APUG2_COMPOSED_DOT_ACCUMULATE:
    return p.auxiliary_signed == p.lhs_signed &&
           p.auxiliary_bits == p.dot_bits && p.out_bits == p.dot_bits + 1;
  default:
    return false;
  }
}

struct ComposedDotViews {
  L1Vectors lhs;
  L1Vectors rhs;
  L1Vectors auxiliary;
  L1Vectors out;
  L1Vectors product_scratch;
  L1Vectors dot_scratch;
  L1Vectors scaled_scratch;

  MmbVectors_seg0 lhs_mmb;
  MmbVectors_seg0 rhs_mmb;
  MmbVectors_seg1 product;
  MmbVectors_seg0 sum_source;
  MmbVectors_seg1 dot_result;

  MmbVectors_seg0 dot_source;
  MmbVectors_seg0 auxiliary_source;
  MmbVectors_seg1 scaled_or_accumulator;
  MmbVectors_seg0 scaled_source;
  MmbVectors_seg1 odd_to_even;
  MmbVectors epilogue_result;

  explicit ComposedDotViews(const ApuG2ComposedDotParams &p)
      : lhs(p.lhs_bits, value_type(p.lhs_signed), p.lhs_bits, kVectors,
            p.lhs_l1_row),
        rhs(p.rhs_bits, value_type(p.rhs_signed), p.rhs_bits, kVectors,
            p.rhs_l1_row),
        auxiliary(p.auxiliary_bits, value_type(p.auxiliary_signed),
                  p.auxiliary_bits, kVectors, p.auxiliary_l1_row),
        out(p.out_bits, value_type(p.out_signed), p.out_bits, kVectors,
            p.out_l1_row),
        product_scratch(p.lhs_bits + p.rhs_bits, value_type(p.lhs_signed),
                        p.lhs_bits + p.rhs_bits, kVectors, p.scratch_l1_row),
        dot_scratch(p.dot_bits, value_type(p.out_signed), p.dot_bits, kVectors,
                    p.scratch_l1_row),
        scaled_scratch(p.epilogue == APUG2_COMPOSED_DOT_PAIR_AFFINE
                           ? p.dot_bits + p.auxiliary_bits
                           : p.dot_bits,
                       value_type(p.out_signed),
                       p.epilogue == APUG2_COMPOSED_DOT_PAIR_AFFINE
                           ? p.dot_bits + p.auxiliary_bits
                           : p.dot_bits,
                       kVectors, p.scratch_l1_row),
        lhs_mmb(p.lhs_bits, value_type(p.lhs_signed), kVectors, 0),
        rhs_mmb(p.rhs_bits, value_type(p.rhs_signed), kVectors, p.lhs_bits),
        product(p.lhs_bits + p.rhs_bits, value_type(p.lhs_signed), kVectors,
                24),
        sum_source(p.lhs_bits + p.rhs_bits, value_type(p.lhs_signed), kVectors,
                   0),
        dot_result(p.dot_bits, value_type(p.out_signed), kVectors, 24),
        dot_source(p.dot_bits, value_type(p.out_signed), kVectors, 0),
        auxiliary_source(
            p.auxiliary_bits, value_type(p.auxiliary_signed), kVectors,
            p.epilogue == APUG2_COMPOSED_DOT_PAIR_AFFINE ? p.dot_bits : 0),
        scaled_or_accumulator(p.epilogue == APUG2_COMPOSED_DOT_PAIR_AFFINE
                                  ? p.dot_bits + p.auxiliary_bits
                                  : p.auxiliary_bits,
                              value_type(p.auxiliary_signed), kVectors, 24),
        scaled_source(p.epilogue == APUG2_COMPOSED_DOT_PAIR_AFFINE
                          ? p.dot_bits + p.auxiliary_bits
                          : p.dot_bits,
                      value_type(p.out_signed), kVectors, 0),
        odd_to_even(p.epilogue == APUG2_COMPOSED_DOT_PAIR_AFFINE
                        ? p.dot_bits + p.auxiliary_bits
                        : p.dot_bits,
                    value_type(p.out_signed), kVectors, 24),
        epilogue_result(p.out_bits, value_type(p.out_signed), kVectors, 24) {}
};

void reduce_dots(const ComposedDotViews &v, const ApuG2ComposedDotParams &p) {
  copy(v.lhs, v.lhs_mmb);
  copy(v.rhs, v.rhs_mmb);
  mul(v.lhs_mmb, v.rhs_mmb, v.product);
  copy(v.product, v.product_scratch);
  copy(v.product_scratch, v.sum_source);
  sum(v.sum_source, p.log_reduction, v.dot_result);
}

void identity_epilogue(const ComposedDotViews &v) { copy(v.dot_result, v.out); }

void pair_affine_epilogue(const ComposedDotViews &v) {
  copy(v.dot_result, v.dot_scratch);
  copy(v.dot_scratch, v.dot_source);
  copy(v.auxiliary, v.auxiliary_source);
  mul(v.dot_source, v.auxiliary_source, v.scaled_or_accumulator);
  copy(v.scaled_or_accumulator, v.scaled_scratch);
  copy(v.scaled_scratch, v.scaled_source);
  copy_from_odd_to_even_vectors(v.scaled_source, v.odd_to_even);
  add(v.scaled_source, v.odd_to_even, v.epilogue_result);
  copy(v.epilogue_result, v.out);
}

void accumulate_epilogue(const ComposedDotViews &v) {
  copy(v.dot_result, v.dot_scratch);
  copy(v.dot_scratch, v.dot_source);
  copy(v.auxiliary, v.scaled_or_accumulator);
  add(v.dot_source, v.scaled_or_accumulator, v.epilogue_result);
  copy(v.epilogue_result, v.out);
}

void pipeline(const ComposedDotViews &v, const ApuG2ComposedDotParams &p) {
  reduce_dots(v, p);
  switch (p.epilogue) {
  case APUG2_COMPOSED_DOT_IDENTITY:
    identity_epilogue(v);
    return;
  case APUG2_COMPOSED_DOT_PAIR_AFFINE:
    pair_affine_epilogue(v);
    return;
  case APUG2_COMPOSED_DOT_ACCUMULATE:
    accumulate_epilogue(v);
    return;
  default:
    return;
  }
}

template <typename Body>
void measure(const ApuG2ComposedDotParams &p, Body body,
             ApuG2ComposedDotTimings &timings) {
  gsi_perf_start();
  for (uint32_t i = 0; i < p.repetitions; ++i)
    body();
  seu_barrier();
  timings.pipeline_ticks = gsi_perf_end();

  gsi_perf_start();
  body();
  seu_barrier();
  timings.final_pipeline_ticks = gsi_perf_end();
}

} // namespace

GSI_LIBRARY_ENTRY_POINT(tenon_apu_g2_composed_dot, uint64_t params_l5) {
  const auto *p = reinterpret_cast<const ApuG2ComposedDotParams *>(
      static_cast<uintptr_t>(params_l5));
  if (p == nullptr || !legal(*p))
    return GSI_STATUS_GENERAL_ERROR;

  const ComposedDotViews views(*p);
  ApuG2ComposedDotTimings timings{};
  measure(*p, [&]() { pipeline(views, *p); }, timings);
  auto *result = reinterpret_cast<ApuG2ComposedDotTimings *>(
      static_cast<uintptr_t>(p->timings_l5));
  *result = timings;
  gsi_arc_dcache_flush_mlines(result, sizeof(*result));
  return GSI_STATUS_SUCCESS;
}
