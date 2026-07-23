// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

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

struct ApuG2GesummvTimings {
  uint64_t gemv_ticks;
  uint64_t scale_combine_ticks;
  uint64_t pipeline_ticks;
  uint64_t final_pipeline_ticks;
};

struct ApuG2GemvTimings {
  uint64_t product_ticks;
  uint64_t pipeline_ticks;
  uint64_t final_pipeline_ticks;
};

struct ApuG2GemvParams {
  uint32_t matrix_l1_row;
  uint32_t x_l1_row;
  uint32_t accumulator_l1_row;
  uint32_t tmp_l1_row;
  uint32_t out_l1_row;
  uint32_t log_block_size;
  uint32_t repetitions;
  uint64_t timings_l5;
};

struct ApuG2AtaxTimings {
  uint64_t pipeline_ticks;
  uint64_t final_pipeline_ticks;
};

struct ApuG2AtaxParams {
  uint32_t repetitions;
  uint64_t timings_l5;
};

struct ApuG2DotTileTimings {
  uint64_t pipeline_ticks;
  uint64_t final_pipeline_ticks;
};

struct ApuG2SelectTimings {
  uint64_t pipeline_ticks;
  uint64_t final_pipeline_ticks;
};

struct ApuG2MinMaxTimings {
  uint64_t pipeline_ticks;
  uint64_t final_pipeline_ticks;
};

struct ApuG2DivTimings {
  uint64_t pipeline_ticks;
  uint64_t final_pipeline_ticks;
};

struct ApuG2MulTimings {
  uint64_t pipeline_ticks;
  uint64_t final_pipeline_ticks;
};

struct ApuG2FillTimings {
  uint64_t pipeline_ticks;
  uint64_t final_pipeline_ticks;
};

struct ApuG2BlockSumTimings {
  uint64_t pipeline_ticks;
  uint64_t final_pipeline_ticks;
};

struct ApuG2ShiftRightTimings {
  uint64_t pipeline_ticks;
  uint64_t final_pipeline_ticks;
};

struct ApuG2SqrtTimings {
  uint64_t pipeline_ticks;
  uint64_t final_pipeline_ticks;
};

struct ApuG2SubTimings {
  uint64_t pipeline_ticks;
  uint64_t final_pipeline_ticks;
};

struct ApuG2DotTileParams {
  uint32_t left_l1_row;
  uint32_t right_l1_row;
  uint32_t accumulator_l1_row;
  uint32_t out_l1_row;
  uint32_t log_block_size;
  uint32_t repetitions;
  uint32_t alpha;
  uint32_t beta;
  uint32_t enable_epilogue;
  uint64_t timings_l5;
};

struct ApuG2SelectParams {
  uint32_t lhs_l1_row;
  uint32_t rhs_l1_row;
  uint32_t true_l1_row;
  uint32_t false_l1_row;
  uint32_t out_l1_row;
  uint32_t repetitions;
  uint64_t timings_l5;
};

struct ApuG2MinMaxParams {
  uint32_t lhs_l1_row;
  uint32_t rhs_l1_row;
  uint32_t min_l1_row;
  uint32_t max_l1_row;
  uint32_t repetitions;
  uint64_t timings_l5;
};

struct ApuG2DivParams {
  uint32_t lhs_l1_row;
  uint32_t rhs_l1_row;
  uint32_t out_l1_row;
  uint32_t repetitions;
  uint64_t timings_l5;
};

struct ApuG2MulParams {
  uint32_t lhs_l1_row;
  uint32_t rhs_l1_row;
  uint32_t out_l1_row;
  uint32_t repetitions;
  uint64_t timings_l5;
};

struct ApuG2FillParams {
  uint32_t out_l1_row;
  uint32_t value;
  uint32_t repetitions;
  uint64_t timings_l5;
};

struct ApuG2BlockSumParams {
  uint32_t src_l1_row;
  uint32_t out_l1_row;
  uint32_t log_block_size;
  uint32_t repetitions;
  uint64_t timings_l5;
};

struct ApuG2ShiftRightParams {
  uint32_t src_l1_row;
  uint32_t out_l1_row;
  uint32_t shift;
  uint32_t repetitions;
  uint64_t timings_l5;
};

struct ApuG2SqrtParams {
  uint32_t src_l1_row;
  uint32_t out_l1_row;
  uint32_t repetitions;
  uint64_t timings_l5;
};

struct ApuG2SubParams {
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

constexpr uint32_t APUG2_U16_GEMM_CHUNK = 128;
constexpr uint32_t APUG2_U16_GEMM_BMAX = 31;

struct ApuG2U16GemmTimings {
  uint64_t pipeline_ticks;
};

struct ApuG2U16GemmParams {
  uint32_t accumulator_l1_row;
  uint32_t weight_l1_row;
  uint32_t product_l1_row;
  uint32_t num_columns;
  uint32_t reduction_count;
  uint64_t timings_l5;
  uint16_t scalars[APUG2_U16_GEMM_BMAX * APUG2_U16_GEMM_CHUNK];
};

struct ApuG2GesummvParams {
  uint32_t matrices_l1_row;
  uint32_t x_l1_row;
  uint32_t scalars_l1_row;
  uint32_t tmp_l1_row;
  uint32_t y_l1_row;
  uint32_t log_block_size;
  uint32_t repetitions;
  uint64_t timings_l5;
};
#pragma pack(pop)

static_assert(sizeof(ApuG2AddTimings) == 40);
static_assert(sizeof(ApuG2AddParams) == 24);
static_assert(sizeof(ApuG2GesummvTimings) == 32);
static_assert(sizeof(ApuG2GesummvParams) == 36);
static_assert(sizeof(ApuG2GemvTimings) == 24);
static_assert(sizeof(ApuG2GemvParams) == 36);
static_assert(sizeof(ApuG2AtaxTimings) == 16);
static_assert(sizeof(ApuG2AtaxParams) == 12);
static_assert(sizeof(ApuG2DotTileTimings) == 16);
static_assert(sizeof(ApuG2DotTileParams) == 44);
static_assert(sizeof(ApuG2SelectTimings) == 16);
static_assert(sizeof(ApuG2SelectParams) == 32);
static_assert(sizeof(ApuG2MinMaxTimings) == 16);
static_assert(sizeof(ApuG2MinMaxParams) == 28);
static_assert(sizeof(ApuG2DivTimings) == 16);
static_assert(sizeof(ApuG2DivParams) == 24);
static_assert(sizeof(ApuG2MulTimings) == 16);
static_assert(sizeof(ApuG2MulParams) == 24);
static_assert(sizeof(ApuG2FillTimings) == 16);
static_assert(sizeof(ApuG2FillParams) == 20);
static_assert(sizeof(ApuG2BlockSumTimings) == 16);
static_assert(sizeof(ApuG2BlockSumParams) == 24);
static_assert(sizeof(ApuG2ShiftRightTimings) == 16);
static_assert(sizeof(ApuG2ShiftRightParams) == 24);
static_assert(sizeof(ApuG2SqrtTimings) == 16);
static_assert(sizeof(ApuG2SqrtParams) == 20);
static_assert(sizeof(ApuG2SubTimings) == 16);
static_assert(sizeof(ApuG2SubParams) == 24);
static_assert(sizeof(ApuG2TypedTimings) == 16);
static_assert(sizeof(ApuG2TypedParams) == 60);
static_assert(sizeof(ApuG2ComposedDotTimings) == 16);
static_assert(sizeof(ApuG2ComposedDotParams) == 76);
static_assert(sizeof(ApuG2U16GemmTimings) == 8);
static_assert(sizeof(ApuG2U16GemmParams) == 28 + 2 * APUG2_U16_GEMM_BMAX * APUG2_U16_GEMM_CHUNK);
