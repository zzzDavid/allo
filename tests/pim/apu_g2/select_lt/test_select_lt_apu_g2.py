# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-card uint16 compare/select primitive gate for APUg2."""

import numpy as np
import pytest

from allo.pim.apu_g2_select_runtime import run_apu_g2_u16_select_lt


@pytest.mark.apu_g2_device
def test_select_lt_apu_g2_hardware(apu_g2_device_gate):
    rng = np.random.default_rng(20260708)
    shape = (4, 65536)
    lhs = rng.integers(0, 65536, size=shape, dtype=np.uint16)
    rhs = rng.integers(0, 65536, size=shape, dtype=np.uint16)
    true_value = rng.integers(0, 65536, size=shape, dtype=np.uint16)
    false_value = rng.integers(0, 65536, size=shape, dtype=np.uint16)

    lhs[0, :8] = np.array([0, 1, 1, 42, 65535, 65534, 32768, 1234], dtype=np.uint16)
    rhs[0, :8] = np.array([0, 0, 2, 42, 65534, 65535, 32768, 4321], dtype=np.uint16)
    true_value[0, :8] = np.array(
        [101, 102, 103, 104, 105, 106, 107, 108], dtype=np.uint16
    )
    false_value[0, :8] = np.array(
        [201, 202, 203, 204, 205, 206, 207, 208], dtype=np.uint16
    )
    expected = np.where(lhs < rhs, true_value, false_value).astype(np.uint16)

    import conftest

    with conftest.apu_g2_board_lock():
        result = run_apu_g2_u16_select_lt(
            lhs, rhs, true_value, false_value, repetitions=128
        )

    np.testing.assert_array_equal(result.extra["outputs"]["out"], expected)
    assert result.backend == "apu_v2"
    assert result.cycles > 0
    assert result.extra["total_ticks"] > 0
    assert result.extra["final_pipeline_ticks"] > 0
    assert result.extra["per_call_ticks"]["pipeline"] > 0
    assert result.extra["vectorization_certificate"]["scalar_tensor_updates"] == 0
    assert result.extra["recipe"]["metadata"]["operation"] == "select_lt"

    device_source = result.extra["sources"]["device/apu_g2_u16_select.cc"]
    assert "lt(v.lhs_mmb, v.rhs_out_mmb, v.predicate);" in device_source
    assert "copy(v.true_value, v.rhs_out_mmb, v.predicate);" in device_source
