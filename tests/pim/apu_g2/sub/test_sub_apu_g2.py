# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-card uint16 subtraction primitive gate for APUg2."""

import numpy as np
import pytest

from allo.pim.apu_g2_sub_runtime import run_apu_g2_u16_sub


@pytest.mark.apu_g2_device
def test_sub_apu_g2_hardware(apu_g2_device_gate):
    rng = np.random.default_rng(20260708)
    shape = (4, 65536)
    lhs = rng.integers(0, 65536, size=shape, dtype=np.uint16)
    rhs = rng.integers(0, 65536, size=shape, dtype=np.uint16)

    lhs[0, :12] = np.array(
        [0, 1, 2, 42, 65535, 65534, 32768, 1234, 60000, 7, 99, 100],
        dtype=np.uint16,
    )
    rhs[0, :12] = np.array(
        [0, 1, 3, 7, 1, 65535, 32768, 4321, 9, 60000, 100, 99],
        dtype=np.uint16,
    )
    expected = np.subtract(lhs, rhs, dtype=np.uint16)

    import conftest

    with conftest.apu_g2_board_lock():
        result = run_apu_g2_u16_sub(lhs, rhs, repetitions=128)

    np.testing.assert_array_equal(result.extra["outputs"]["out"], expected)
    assert result.backend == "apu_v2"
    assert result.cycles > 0
    assert result.extra["total_ticks"] > 0
    assert result.extra["final_pipeline_ticks"] > 0
    assert result.extra["per_call_ticks"]["pipeline"] > 0
    assert result.extra["vectorization_certificate"]["scalar_tensor_updates"] == 0
    assert result.extra["recipe"]["metadata"]["operation"] == "sub"

    device_source = result.extra["sources"]["device/apu_g2_u16_sub.cc"]
    assert "sub(v.lhs_mmb, v.rhs_out_mmb);" in device_source
