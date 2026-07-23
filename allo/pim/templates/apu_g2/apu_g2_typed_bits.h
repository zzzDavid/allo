// Copyright Allo authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// VectorPackRef exposes a host container value, whose unused high bits are
// not part of the declared logical scalar.  Canonicalize before converting to
// a native NumPy storage type.  This is especially important for signed INT15
// readback, where carrier bit 15 has been observed set independently of the
// 15-bit payload.
constexpr int64_t apu_g2_canonical_scalar(int64_t carrier, uint32_t bits,
                                          bool is_signed) {
  const uint64_t mask = (UINT64_C(1) << bits) - 1;
  const uint64_t payload = static_cast<uint64_t>(carrier) & mask;
  if (is_signed && (payload & (UINT64_C(1) << (bits - 1))) != 0)
    return static_cast<int64_t>(payload | ~mask);
  return static_cast<int64_t>(payload);
}

static_assert(apu_g2_canonical_scalar(INT64_C(0x8580), 15, true) == 1408);
static_assert(apu_g2_canonical_scalar(INT64_C(0x7fff), 15, true) == -1);
static_assert(apu_g2_canonical_scalar(INT64_C(0x4000), 15, true) == -16384);
static_assert(apu_g2_canonical_scalar(INT64_C(0x1ffff), 16, false) == 65535);
