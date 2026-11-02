// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Card index passed as the first DeviceConfig field by every host program;
// the build defines it from TENON_APU_G2_DEVICE (apu-NN -> NN).
#ifndef TENON_APU_G2_DEVICE_INDEX
#define TENON_APU_G2_DEVICE_INDEX 0
#endif

#pragma pack(push, 1)
struct ApuG2AddTimings {
  uint64_t input_copy_ticks;
  uint64_t add_ticks;
  uint64_t output_copy_ticks;
  uint64_t pipeline_ticks;
  uint64_t final_pipeline_ticks;
};

struct ApuG2AddParams {
  uint32_t lhs_l1_row;
  uint32_t rhs_l1_row;
  uint32_t out_l1_row;
  uint32_t repetitions;
  uint64_t timings_l5;
};

enum ApuG2TypedOperation : uint32_t {
  APUG2_TYPED_ADD = 1,
  APUG2_TYPED_MUL = 2,
  APUG2_TYPED_DIV = 3,
  APUG2_TYPED_BLOCK_SUM = 4,
  APUG2_TYPED_DOT = 5,
};

struct ApuG2TypedTimings {
  uint64_t pipeline_ticks;
  uint64_t final_pipeline_ticks;
};

struct ApuG2TypedParams {
  uint32_t operation;
  uint32_t lhs_l1_row;
  uint32_t rhs_l1_row;
  uint32_t out_l1_row;
  uint32_t scratch_l1_row;
  uint32_t lhs_bits;
  uint32_t rhs_bits;
  uint32_t out_bits;
  uint32_t lhs_signed;
  uint32_t rhs_signed;
  uint32_t out_signed;
  uint32_t log_reduction;
  uint32_t repetitions;
  uint64_t timings_l5;
};

enum ApuG2ComposedDotEpilogue : uint32_t {
  APUG2_COMPOSED_DOT_IDENTITY = 0,
  APUG2_COMPOSED_DOT_PAIR_AFFINE = 1,
  APUG2_COMPOSED_DOT_ACCUMULATE = 2,
};

struct ApuG2ComposedDotTimings {
  uint64_t pipeline_ticks;
  uint64_t final_pipeline_ticks;
};

struct ApuG2ComposedDotParams {
  uint32_t lhs_l1_row;
  uint32_t rhs_l1_row;
  uint32_t auxiliary_l1_row;
  uint32_t out_l1_row;
  uint32_t scratch_l1_row;
  uint32_t lhs_bits;
  uint32_t rhs_bits;
  uint32_t dot_bits;
  uint32_t auxiliary_bits;
  uint32_t out_bits;
  uint32_t lhs_signed;
  uint32_t rhs_signed;
  uint32_t auxiliary_signed;
  uint32_t out_signed;
  uint32_t log_reduction;
  uint32_t epilogue;
  uint32_t repetitions;
  uint64_t timings_l5;
};
#pragma pack(pop)

static_assert(sizeof(ApuG2AddTimings) == 40);
static_assert(sizeof(ApuG2AddParams) == 24);
static_assert(sizeof(ApuG2TypedTimings) == 16);
static_assert(sizeof(ApuG2TypedParams) == 60);
static_assert(sizeof(ApuG2ComposedDotTimings) == 16);
static_assert(sizeof(ApuG2ComposedDotParams) == 76);
