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
void div(const MmbVectors_seg0 &, const MmbVectors_seg1 &, const L1Vectors &);
void sum(const MmbVectors_seg0 &, uint32_t, const MmbVectors_seg1 &);
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

bool common_legal(const ApuG2TypedParams &p) {
  return p.timings_l5 != 0 && p.repetitions != 0 && p.lhs_bits >= 1 &&
         p.lhs_bits <= kSegmentBits && p.out_bits >= 1 &&
         p.out_bits <= kSegmentBits && p.lhs_signed <= 1 && p.rhs_signed <= 1 &&
         p.out_signed <= 1;
}

bool operation_legal(const ApuG2TypedParams &p) {
  if (!common_legal(p))
    return false;
  switch (p.operation) {
  case APUG2_TYPED_ADD:
    return p.rhs_bits == p.lhs_bits && p.out_bits == p.lhs_bits &&
           p.rhs_signed == p.lhs_signed && p.out_signed == p.lhs_signed;
  case APUG2_TYPED_MUL:
    return p.lhs_bits >= 2 && p.rhs_bits >= 2 &&
           p.lhs_bits + p.rhs_bits == p.out_bits &&
           p.out_bits <= kSegmentBits && p.rhs_signed == p.lhs_signed &&
           p.out_signed == p.lhs_signed;
  case APUG2_TYPED_DIV:
    return p.lhs_signed == 0 && p.rhs_signed == 0 && p.out_signed == 0 &&
           p.rhs_bits == p.lhs_bits && p.out_bits == p.lhs_bits;
  case APUG2_TYPED_BLOCK_SUM:
    return p.lhs_signed == 0 && p.out_signed == 0 && p.log_reduction >= 1 &&
           p.log_reduction <= 12 && p.out_bits == p.lhs_bits + p.log_reduction;
  case APUG2_TYPED_DOT:
    return p.lhs_bits >= 2 && p.rhs_bits >= 2 &&
           p.lhs_bits + p.rhs_bits <= 23 && p.log_reduction >= 1 &&
           p.log_reduction <= 12 &&
           p.out_bits == p.lhs_bits + p.rhs_bits + p.log_reduction &&
           p.out_bits <= kSegmentBits && p.rhs_signed == p.lhs_signed &&
           p.out_signed == p.lhs_signed;
  default:
    return false;
  }
}

struct AddViews {
  L1Vectors lhs;
  L1Vectors rhs;
  L1Vectors out;
  MmbVectors_seg0 lhs_mmb;
  MmbVectors_seg1 rhs_mmb;
  MmbVectors result;

  explicit AddViews(const ApuG2TypedParams &p)
      : lhs(p.lhs_bits, value_type(p.lhs_signed), p.lhs_bits, kVectors,
            p.lhs_l1_row),
        rhs(p.rhs_bits, value_type(p.rhs_signed), p.rhs_bits, kVectors,
            p.rhs_l1_row),
        out(p.out_bits, value_type(p.out_signed), p.out_bits, kVectors,
            p.out_l1_row),
        lhs_mmb(p.lhs_bits, value_type(p.lhs_signed), kVectors, 0),
        rhs_mmb(p.rhs_bits, value_type(p.rhs_signed), kVectors, 24),
        result(p.out_bits, value_type(p.out_signed), kVectors, 24) {}
};

void pipeline(const AddViews &v) {
  copy(v.lhs, v.lhs_mmb);
  copy(v.rhs, v.rhs_mmb);
  add(v.lhs_mmb, v.rhs_mmb, v.result);
  copy(v.result, v.out);
}

struct MulViews {
  L1Vectors lhs;
  L1Vectors rhs;
  L1Vectors out;
  MmbVectors_seg0 lhs_mmb;
  MmbVectors_seg0 rhs_mmb;
  MmbVectors_seg1 product;

  explicit MulViews(const ApuG2TypedParams &p)
      : lhs(p.lhs_bits, value_type(p.lhs_signed), p.lhs_bits, kVectors,
            p.lhs_l1_row),
        rhs(p.rhs_bits, value_type(p.rhs_signed), p.rhs_bits, kVectors,
            p.rhs_l1_row),
        out(p.out_bits, value_type(p.out_signed), p.out_bits, kVectors,
            p.out_l1_row),
        lhs_mmb(p.lhs_bits, value_type(p.lhs_signed), kVectors, 0),
        rhs_mmb(p.rhs_bits, value_type(p.rhs_signed), kVectors, p.lhs_bits),
        product(p.out_bits, value_type(p.out_signed), kVectors, 24) {}
};

void pipeline(const MulViews &v) {
  copy(v.lhs, v.lhs_mmb);
  copy(v.rhs, v.rhs_mmb);
  mul(v.lhs_mmb, v.rhs_mmb, v.product);
  copy(v.product, v.out);
}

struct DivViews {
  L1Vectors lhs;
  L1Vectors rhs;
  L1Vectors out;
  MmbVectors_seg0 lhs_mmb;
  MmbVectors_seg1 rhs_mmb;

  explicit DivViews(const ApuG2TypedParams &p)
      : lhs(p.lhs_bits, gsi::G2_TYPES::UINT, p.lhs_bits, kVectors,
            p.lhs_l1_row),
        rhs(p.rhs_bits, gsi::G2_TYPES::UINT, p.rhs_bits, kVectors,
            p.rhs_l1_row),
        out(p.out_bits, gsi::G2_TYPES::UINT, p.out_bits, kVectors,
            p.out_l1_row),
        lhs_mmb(p.lhs_bits, gsi::G2_TYPES::UINT, kVectors, 0),
        rhs_mmb(p.rhs_bits, gsi::G2_TYPES::UINT, kVectors, 24) {}
};

void pipeline(const DivViews &v) {
  copy(v.lhs, v.lhs_mmb);
  copy(v.rhs, v.rhs_mmb);
  div(v.lhs_mmb, v.rhs_mmb, v.out);
}

struct SumViews {
  L1Vectors source;
  L1Vectors out;
  MmbVectors_seg0 source_mmb;
  MmbVectors_seg1 result;

  explicit SumViews(const ApuG2TypedParams &p)
      : source(p.lhs_bits, value_type(p.lhs_signed), p.lhs_bits, kVectors,
               p.lhs_l1_row),
        out(p.out_bits, value_type(p.out_signed), p.out_bits, kVectors,
            p.out_l1_row),
        source_mmb(p.lhs_bits, value_type(p.lhs_signed), kVectors, 0),
        result(p.out_bits, value_type(p.out_signed), kVectors, 24) {}
};

void pipeline(const SumViews &v, uint32_t log_reduction) {
  copy(v.source, v.source_mmb);
  sum(v.source_mmb, log_reduction, v.result);
  copy(v.result, v.out);
}

struct DotViews {
  L1Vectors lhs;
  L1Vectors rhs;
  L1Vectors product_scratch;
  L1Vectors out;
  MmbVectors_seg0 lhs_mmb;
  MmbVectors_seg0 rhs_mmb;
  MmbVectors_seg1 product;
  MmbVectors_seg0 sum_source;
  MmbVectors_seg1 sum_result;

  explicit DotViews(const ApuG2TypedParams &p)
      : lhs(p.lhs_bits, value_type(p.lhs_signed), p.lhs_bits, kVectors,
            p.lhs_l1_row),
        rhs(p.rhs_bits, value_type(p.rhs_signed), p.rhs_bits, kVectors,
            p.rhs_l1_row),
        product_scratch(p.lhs_bits + p.rhs_bits, value_type(p.out_signed),
                        p.lhs_bits + p.rhs_bits, kVectors, p.scratch_l1_row),
        out(p.out_bits, value_type(p.out_signed), p.out_bits, kVectors,
            p.out_l1_row),
        lhs_mmb(p.lhs_bits, value_type(p.lhs_signed), kVectors, 0),
        rhs_mmb(p.rhs_bits, value_type(p.rhs_signed), kVectors, p.lhs_bits),
        product(p.lhs_bits + p.rhs_bits, value_type(p.out_signed), kVectors,
                24),
        sum_source(p.lhs_bits + p.rhs_bits, value_type(p.out_signed), kVectors,
                   0),
        sum_result(p.out_bits, value_type(p.out_signed), kVectors, 24) {}
};

void pipeline(const DotViews &v, uint32_t log_reduction) {
  copy(v.lhs, v.lhs_mmb);
  copy(v.rhs, v.rhs_mmb);
  mul(v.lhs_mmb, v.rhs_mmb, v.product);
  copy(v.product, v.product_scratch);
  copy(v.product_scratch, v.sum_source);
  sum(v.sum_source, log_reduction, v.sum_result);
  copy(v.sum_result, v.out);
}

template <typename Body>
void measure(const ApuG2TypedParams &p, Body body, ApuG2TypedTimings &timings) {
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

gsi_status_t run(const ApuG2TypedParams &p, ApuG2TypedTimings &timings) {
  switch (p.operation) {
  case APUG2_TYPED_ADD: {
    const AddViews views(p);
    measure(p, [&]() { pipeline(views); }, timings);
    return GSI_STATUS_SUCCESS;
  }
  case APUG2_TYPED_MUL: {
    const MulViews views(p);
    measure(p, [&]() { pipeline(views); }, timings);
    return GSI_STATUS_SUCCESS;
  }
  case APUG2_TYPED_DIV: {
    const DivViews views(p);
    measure(p, [&]() { pipeline(views); }, timings);
    return GSI_STATUS_SUCCESS;
  }
  case APUG2_TYPED_BLOCK_SUM: {
    const SumViews views(p);
    measure(p, [&]() { pipeline(views, p.log_reduction); }, timings);
    return GSI_STATUS_SUCCESS;
  }
  case APUG2_TYPED_DOT: {
    const DotViews views(p);
    measure(p, [&]() { pipeline(views, p.log_reduction); }, timings);
    return GSI_STATUS_SUCCESS;
  }
  default:
    return GSI_STATUS_GENERAL_ERROR;
  }
}

} // namespace

GSI_LIBRARY_ENTRY_POINT(tenon_apu_g2_typed, uint64_t params_l5) {
  const auto *p = reinterpret_cast<const ApuG2TypedParams *>(
      static_cast<uintptr_t>(params_l5));
  if (p == nullptr || !operation_legal(*p))
    return GSI_STATUS_GENERAL_ERROR;

  ApuG2TypedTimings timings{};
  const gsi_status_t status = run(*p, timings);
  if (status != GSI_STATUS_SUCCESS)
    return status;
  auto *result = reinterpret_cast<ApuG2TypedTimings *>(
      static_cast<uintptr_t>(p->timings_l5));
  *result = timings;
  gsi_arc_dcache_flush_mlines(result, sizeof(*result));
  return GSI_STATUS_SUCCESS;
}
