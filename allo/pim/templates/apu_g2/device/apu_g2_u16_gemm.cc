// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <device/vector_core/gsi_libgal.h>

#include <g2_64vl_types.h>
namespace gsi::g2_64vl {
void copy(const L1Vectors &, const MmbVectors &);
void copy(const MmbVectors &, const L1Vectors &);
void copy(uint64_t, const MmbVectors &);
void mul(const MmbVectors_seg0 &, const MmbVectors_seg0 &,
         const MmbVectors_seg1 &);
void shift_left(const MmbVectors &, uint32_t);
void add(const L1Vectors &, const MmbVectors &, const MmbVectors &);
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
constexpr uint32_t kByteBits = 8;

struct ScalarMulViews {
  L1Vectors value_low;
  L1Vectors value_high;
  L1Vectors accumulator;
  L1Vectors product_scratch;
  MmbVectors_seg0 operand0;
  MmbVectors_seg0 operand1;
  MmbVectors_seg0 accumulator_mmb;
  MmbVectors_seg1 product;

  ScalarMulViews(uint32_t value_row, uint32_t accumulator_row,
                 uint32_t product_row)
      : value_low(kByteBits, gsi::G2_TYPES::UINT, kBits, 1, value_row),
        value_high(kByteBits, gsi::G2_TYPES::UINT, kBits, 1,
                   value_row + kByteBits),
        accumulator(kBits, gsi::G2_TYPES::UINT, kBits, 1, accumulator_row),
        product_scratch(kBits, gsi::G2_TYPES::UINT, kBits, 1, product_row),
        operand0(kByteBits, gsi::G2_TYPES::UINT, 1, 0),
        operand1(kByteBits, gsi::G2_TYPES::UINT, 1, 8),
        accumulator_mmb(kBits, gsi::G2_TYPES::UINT, 1, 0),
        product(kBits, gsi::G2_TYPES::UINT, 1, 24) {}
};

void multiply_byte(const ScalarMulViews &v, const L1Vectors &value,
                   uint32_t scalar) {
  copy(value, v.operand0);
  copy(scalar, v.operand1);
  mul(v.operand0, v.operand1, v.product);
}

void multiply_accumulate(const ScalarMulViews &v, uint32_t scalar) {
  const uint32_t scalar_low = scalar & 0xffu;
  const uint32_t scalar_high = (scalar >> kByteBits) & 0xffu;

  multiply_byte(v, v.value_low, scalar_low);
  copy(v.product, v.product_scratch);

  multiply_byte(v, v.value_low, scalar_high);
  shift_left(v.product, kByteBits);
  add(v.product_scratch, v.product, v.product);
  copy(v.product, v.product_scratch);

  multiply_byte(v, v.value_high, scalar_low);
  shift_left(v.product, kByteBits);
  add(v.product_scratch, v.product, v.product);

  copy(v.accumulator, v.accumulator_mmb);
  add(v.accumulator_mmb, v.product, v.product);
  copy(v.product, v.accumulator);
}

} // namespace

GSI_LIBRARY_ENTRY_POINT(tenon_apu_g2_u16_gemm_chunk, uint64_t params_l5) {
  const auto *p = reinterpret_cast<const ApuG2U16GemmParams *>(
      static_cast<uintptr_t>(params_l5));
  if (p == nullptr || p->timings_l5 == 0 || p->num_columns == 0 ||
      p->num_columns > APUG2_U16_GEMM_BMAX || p->reduction_count == 0 ||
      p->reduction_count > APUG2_U16_GEMM_CHUNK)
    return GSI_STATUS_GENERAL_ERROR;

  gsi_perf_start();
  for (uint32_t column = 0; column < p->num_columns; ++column) {
    const uint16_t *scalars =
        p->scalars + column * APUG2_U16_GEMM_CHUNK;
    const uint32_t accumulator_row =
        p->accumulator_l1_row + column * kBits;
    for (uint32_t reduction = 0; reduction < p->reduction_count; ++reduction) {
      const uint32_t weight_row = p->weight_l1_row + reduction * kBits;
      const ScalarMulViews views(weight_row, accumulator_row,
                                 p->product_l1_row);
      multiply_accumulate(views, scalars[reduction]);
    }
  }
  seu_barrier();
  const ApuG2U16GemmTimings timings{gsi_perf_end()};

  auto *result = reinterpret_cast<ApuG2U16GemmTimings *>(
      static_cast<uintptr_t>(p->timings_l5));
  *result = timings;
  gsi_arc_dcache_flush_mlines(result, sizeof(*result));
  return GSI_STATUS_SUCCESS;
}
