# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-card uint16 modular multiplication primitive gate for APUg2."""

import numpy as np
import pytest

from allo.pim.apu_g2_mul_runtime import run_apu_g2_u16_mul


@pytest.mark.apu_g2_device
def test_mul_apu_g2_hardware(apu_g2_device_gate):
    rng = np.random.default_rng(20260708)
    shape = (4, 65536)
    lhs = rng.integers(0, 65536, size=shape, dtype=np.uint16)
    rhs = rng.integers(0, 65536, size=shape, dtype=np.uint16)

    lhs[0, :16] = np.array(
        [0, 1, 2, 3, 255, 256, 257, 1024, 4096, 32767, 32768, 65535, 123, 60000, 7, 99],
        dtype=np.uint16,
    )
    rhs[0, :16] = np.array(
        [0, 1, 2, 65535, 255, 256, 257, 64, 16, 3, 2, 65535, 456, 9, 60000, 100],
        dtype=np.uint16,
    )
    expected = (lhs.astype(np.uint32) * rhs.astype(np.uint32)).astype(np.uint16)

    import conftest

    with conftest.apu_g2_board_lock():
        result = run_apu_g2_u16_mul(lhs, rhs, repetitions=64)

    np.testing.assert_array_equal(result.extra["outputs"]["out"], expected)
    assert result.backend == "apu_v2"
    assert result.cycles > 0
    assert result.extra["total_ticks"] > 0
    assert result.extra["final_pipeline_ticks"] > 0
    assert result.extra["per_call_ticks"]["pipeline"] > 0
    assert result.extra["vectorization_certificate"]["scalar_tensor_updates"] == 0
    assert result.extra["recipe"]["metadata"]["operation"] == "mul"
    assert result.extra["recipe"]["metadata"]["omitted_term"] == "hi_hi_is_zero_mod_2_16"

    device_source = result.extra["sources"]["device/apu_g2_u16_mul.cc"]
    assert "mul(v.operand0, v.operand1, v.product);" in device_source
    assert "shift_left(v.product, kByteBits);" in device_source
