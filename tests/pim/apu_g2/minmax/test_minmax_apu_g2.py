# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-card uint16 min/max primitive gate for APUg2."""

import numpy as np
import pytest

from allo.pim.apu_g2_minmax_runtime import run_apu_g2_u16_minmax


@pytest.mark.apu_g2_device
def test_minmax_apu_g2_hardware(apu_g2_device_gate):
    rng = np.random.default_rng(20260708)
    shape = (4, 65536)
    lhs = rng.integers(0, 65536, size=shape, dtype=np.uint16)
    rhs = rng.integers(0, 65536, size=shape, dtype=np.uint16)

    lhs[0, :10] = np.array(
        [0, 1, 1, 42, 65535, 65534, 32768, 1234, 60000, 7],
        dtype=np.uint16,
    )
    rhs[0, :10] = np.array(
        [0, 0, 2, 42, 65534, 65535, 32768, 4321, 9, 60000],
        dtype=np.uint16,
    )
    expected_min = np.minimum(lhs, rhs)
    expected_max = np.maximum(lhs, rhs)

    import conftest

    with conftest.apu_g2_board_lock():
        result = run_apu_g2_u16_minmax(lhs, rhs, repetitions=128)

    np.testing.assert_array_equal(result.extra["outputs"]["min"], expected_min)
    np.testing.assert_array_equal(result.extra["outputs"]["max"], expected_max)
    assert result.backend == "apu_v2"
    assert result.cycles > 0
    assert result.extra["total_ticks"] > 0
    assert result.extra["final_pipeline_ticks"] > 0
    assert result.extra["per_call_ticks"]["pipeline"] > 0
    assert result.extra["vectorization_certificate"]["scalar_tensor_updates"] == 0
    assert result.extra["recipe"]["metadata"]["operation"] == "minmax"

    device_source = result.extra["sources"]["device/apu_g2_u16_minmax.cc"]
    assert "min_mmb(v.rhs_out_mmb, v.lhs_mmb);" in device_source
    assert "max_mmb(v.rhs_out_mmb, v.lhs_mmb);" in device_source
