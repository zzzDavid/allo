// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include <device/vector_core/gsi_libgal.h>
#include <g2_64vl_types.h>
namespace gsi::g2_64vl {
void copy(const L1Vectors &, const MmbVectors &);
void copy(const L1Vector &, const MmbVectors &);
void copy(const MmbVectors &, const L1Vectors &);
void copy(const MmbVectors &, const L1Vectors &, const L1Markers &);
void mul(const MmbVectors_seg0 &, const MmbVectors_seg0 &,
         const MmbVectors_seg1 &);
void sum(const MmbVectors_seg0 &, uint32_t, const MmbVectors_seg1 &);
void shift_left(const MmbVectors &, uint32_t);
void add(const L1Vectors &, const MmbVectors &, const MmbVectors &);
void add(const MmbVectors_seg0 &, const MmbVectors_seg1 &, const MmbVectors &);
void copy_from_odd_to_even_vectors(const MmbVectors &, const MmbVectors &);
void copy_from_vector_2_to_0(const MmbVectors &, const MmbVectors &);
void reset(const L1Vectors &);
void seu_barrier(void);
} // namespace gsi::g2_64vl

#include <g2_gtml.h>
#include <gsi_arc_cache.h>
#include <gsi_perf.h>

#include "apu_g2_params.h"
#include "gsi_library.h"

namespace {
using namespace gsi::g2_64vl;

constexpr uint32_t kVectors = 4;
constexpr uint32_t kValueBits = 16;
constexpr uint32_t kByteBits = 8;
constexpr uint32_t kChunkLog = 5;
constexpr uint32_t kChunks = 4;
constexpr uint32_t kBlocks = 65536u >> kChunkLog;

constexpr uint32_t kStage1MatricesRow = 64;
constexpr uint32_t kStage1XRow = 320;
constexpr uint32_t kStage2MatrixRow = 384;
constexpr uint32_t kYRow = 448;
constexpr uint32_t kPartialRow = 512;
constexpr uint32_t kStage1AccRow = 576;
constexpr uint32_t kBroadcastRow = 640;
constexpr uint32_t kProductScratchRow = 704;
constexpr uint32_t kStage2PartialRow = 768;
constexpr uint32_t kOutRow = 832;
constexpr uint32_t kBlockMarkersRow = 896;

struct AtaxViews {
  L1Vectors partial;
  L1Vectors stage1_acc;
  L1Vectors broadcast;
  L1Vectors product_scratch;
  L1Vectors stage2_partial;
  L1Vectors y;
  L1Vectors out;
  L1Markers block_first;

  MmbVectors_seg0 operand0;
  MmbVectors_seg0 operand1;
  MmbVectors_seg0 sum_source;
  MmbVectors_seg1 product;
  MmbVectors_seg1 sum_destination;
  MmbVectors sum_low16;

  AtaxViews()
      : partial(kValueBits, gsi::G2_TYPES::UINT, kValueBits, kVectors,
                kPartialRow),
        stage1_acc(kValueBits, gsi::G2_TYPES::UINT, kValueBits, kVectors,
                   kStage1AccRow),
        broadcast(kValueBits, gsi::G2_TYPES::UINT, kValueBits, kVectors,
                  kBroadcastRow),
        product_scratch(kValueBits, gsi::G2_TYPES::UINT, kValueBits, kVectors,
                        kProductScratchRow),
        stage2_partial(kValueBits, gsi::G2_TYPES::UINT, kValueBits, kVectors,
                       kStage2PartialRow),
        y(kValueBits, gsi::G2_TYPES::UINT, kValueBits, kVectors, kYRow),
        out(kValueBits, gsi::G2_TYPES::UINT, kValueBits, kVectors, kOutRow),
        block_first(kVectors, kValueBits, kBlockMarkersRow),
        operand0(kByteBits, gsi::G2_TYPES::UINT, kVectors, 0),
        operand1(kByteBits, gsi::G2_TYPES::UINT, kVectors, 8),
        sum_source(kValueBits, gsi::G2_TYPES::UINT, kVectors, 0),
        product(kValueBits, gsi::G2_TYPES::UINT, kVectors, 24),
        sum_destination(kValueBits + kChunkLog, gsi::G2_TYPES::UINT, kVectors,
                        24),
        sum_low16(kValueBits, gsi::G2_TYPES::UINT, kVectors, 24) {}

  L1Vectors stage1_matrix_low(uint32_t chunk) const {
    return L1Vectors(kByteBits, gsi::G2_TYPES::UINT, kValueBits, kVectors,
                     kStage1MatricesRow + chunk * 64);
  }
  L1Vectors stage1_matrix_high(uint32_t chunk) const {
    return L1Vectors(kByteBits, gsi::G2_TYPES::UINT, kValueBits, kVectors,
                     kStage1MatricesRow + chunk * 64 + kByteBits);
  }
  L1Vector stage1_x_low(uint32_t chunk) const {
    return L1Vector(kByteBits, gsi::G2_TYPES::UINT,
                    kStage1XRow + chunk * kValueBits);
  }
  L1Vector stage1_x_high(uint32_t chunk) const {
    return L1Vector(kByteBits, gsi::G2_TYPES::UINT,
                    kStage1XRow + chunk * kValueBits + kByteBits);
  }
  L1Vectors stage2_matrix_low() const {
    return L1Vectors(kByteBits, gsi::G2_TYPES::UINT, kValueBits, kVectors,
                     kStage2MatrixRow);
  }
  L1Vectors stage2_matrix_high() const {
    return L1Vectors(kByteBits, gsi::G2_TYPES::UINT, kValueBits, kVectors,
                     kStage2MatrixRow + kByteBits);
  }
  L1Vectors broadcast_low() const {
    return L1Vectors(kByteBits, gsi::G2_TYPES::UINT, kValueBits, kVectors,
                     kBroadcastRow);
  }
  L1Vectors broadcast_high() const {
    return L1Vectors(kByteBits, gsi::G2_TYPES::UINT, kValueBits, kVectors,
                     kBroadcastRow + kByteBits);
  }
};

void multiply_reduce(const AtaxViews &v, const L1Vectors &matrix_byte,
                     const L1Vector &vector_byte) {
  copy(matrix_byte, v.operand0);
  copy(vector_byte, v.operand1);
  mul(v.operand0, v.operand1, v.product);
  copy(v.product, v.product_scratch);
  copy(v.product_scratch, v.sum_source);
  sum(v.sum_source, kChunkLog, v.sum_destination);
}

void multiply_reduce(const AtaxViews &v, const L1Vectors &matrix_byte,
                     const L1Vectors &vector_byte) {
  copy(matrix_byte, v.operand0);
  copy(vector_byte, v.operand1);
  mul(v.operand0, v.operand1, v.product);
  copy(v.product, v.product_scratch);
  copy(v.product_scratch, v.sum_source);
  sum(v.sum_source, kChunkLog, v.sum_destination);
}

template <typename MatrixLow, typename MatrixHigh, typename VectorLow,
          typename VectorHigh>
void product32(const AtaxViews &v, const MatrixLow &matrix_low,
               const MatrixHigh &matrix_high, const VectorLow &vector_low,
               const VectorHigh &vector_high) {
  multiply_reduce(v, matrix_low, vector_low);
  copy(v.sum_low16, v.partial);

  multiply_reduce(v, matrix_low, vector_high);
  shift_left(v.sum_low16, kByteBits);
  add(v.partial, v.sum_low16, v.sum_low16);
  copy(v.sum_low16, v.partial);

  multiply_reduce(v, matrix_high, vector_low);
  shift_left(v.sum_low16, kByteBits);
  add(v.partial, v.sum_low16, v.sum_low16);
  copy(v.sum_low16, v.partial);
}

void stage1(const AtaxViews &v) {
  for (uint32_t chunk = 0; chunk < kChunks; ++chunk) {
    product32(v, v.stage1_matrix_low(chunk), v.stage1_matrix_high(chunk),
              v.stage1_x_low(chunk), v.stage1_x_high(chunk));
    if (chunk == 0) {
      reset(v.stage1_acc);
    } else {
      add(v.stage1_acc, v.sum_low16, v.sum_low16);
      reset(v.stage1_acc);
    }
    // SUM documents only block-first outputs.  The marker removes every
    // internal reduction lane before resident compaction.
    copy(v.sum_low16, v.stage1_acc, v.block_first);
  }
}

bool rebroadcast_resident_tmp() {
  auto &gtml = gsi::gtml::G2Gtml::getInstance();
  for (uint32_t stream = 0; stream < kVectors; ++stream) {
    const gsi::L1Container src(kStage1AccRow + stream * kValueBits, kValueBits);
    for (uint32_t log = 10; log >= 6; --log)
      if (gtml.squeeze_rows_inplace(src, log) != 0)
        return false;
    const gsi::L1Container dst(kBroadcastRow + stream * kValueBits, kValueBits);
    if (gtml.spread_blk(src, kChunkLog, 0, kChunkLog, kBlocks, dst, true) != 0)
      return false;
  }
  return true;
}

void stage2(const AtaxViews &v) {
  product32(v, v.stage2_matrix_low(), v.stage2_matrix_high(), v.broadcast_low(),
            v.broadcast_high());

  // Reduce the four 32-element chunk sums: (set0+set1)+(set2+set3).
  copy(v.sum_low16, v.stage2_partial);
  copy(v.stage2_partial, v.sum_source);
  copy_from_odd_to_even_vectors(v.sum_source, v.product);
  add(v.sum_source, v.product, v.product);
  copy_from_vector_2_to_0(v.product, v.sum_source);
  add(v.sum_source, v.product, v.product);

  // Canonical Allo stage-N semantics accumulate into the caller's y.
  add(v.y, v.product, v.product);
  copy(v.product, v.out);
}

bool full_pipeline(const AtaxViews &v) {
  stage1(v);
  if (!rebroadcast_resident_tmp())
    return false;
  stage2(v);
  return true;
}

template <typename Body> uint64_t time_fragment(Body body) {
  gsi_perf_start();
  body();
  seu_barrier();
  return gsi_perf_end();
}
} // namespace

GSI_LIBRARY_ENTRY_POINT(tenon_apu_g2_u16_atax, uint64_t params_l5) {
  const auto *p = reinterpret_cast<const ApuG2AtaxParams *>(
      static_cast<uintptr_t>(params_l5));
  if (p == nullptr || p->timings_l5 == 0 || p->repetitions == 0)
    return GSI_STATUS_GENERAL_ERROR;
  const AtaxViews v;
  ApuG2AtaxTimings timings{};
  bool success = true;
  timings.pipeline_ticks = time_fragment([&]() {
    for (uint32_t i = 0; i < p->repetitions; ++i)
      success &= full_pipeline(v);
  });
  timings.final_pipeline_ticks =
      time_fragment([&]() { success &= full_pipeline(v); });
  if (!success)
    return GSI_STATUS_GENERAL_ERROR;
  auto *result = reinterpret_cast<ApuG2AtaxTimings *>(
      static_cast<uintptr_t>(p->timings_l5));
  *result = timings;
  gsi_arc_dcache_flush_mlines(result, sizeof(*result));
  return GSI_STATUS_SUCCESS;
}
