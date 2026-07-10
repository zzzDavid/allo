# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-card uint16 unsigned division primitive gate for APUg2."""

import numpy as np
import pytest

from allo.pim.apu_g2_div_runtime import run_apu_g2_u16_div


@pytest.mark.apu_g2_device
def test_div_apu_g2_hardware(apu_g2_device_gate):
    rng = np.random.default_rng(20260708)
    shape = (4, 65536)
    lhs = rng.integers(0, 65536, size=shape, dtype=np.uint16)
    rhs = rng.integers(1, 65536, size=shape, dtype=np.uint16)

    lhs[0, :12] = np.array(
        [0, 1, 2, 42, 65535, 65534, 32768, 1234, 60000, 7, 99, 100],
        dtype=np.uint16,
    )
    rhs[0, :12] = np.array(
        [1, 1, 2, 7, 1, 65535, 32768, 4321, 9, 60000, 10, 3],
        dtype=np.uint16,
    )
    expected = np.floor_divide(lhs, rhs).astype(np.uint16)

    zero_rhs = rhs.copy()
    zero_rhs[0, 0] = 0
    with pytest.raises(ValueError, match="nonzero"):
        run_apu_g2_u16_div(lhs, zero_rhs, repetitions=1)

    import conftest

    with conftest.apu_g2_board_lock():
        result = run_apu_g2_u16_div(lhs, rhs, repetitions=4)

    np.testing.assert_array_equal(result.extra["outputs"]["out"], expected)
    assert result.backend == "apu_v2"
    assert result.cycles > 0
    assert result.extra["total_ticks"] > 0
    assert result.extra["final_pipeline_ticks"] > 0
    assert result.extra["per_call_ticks"]["pipeline"] > 0
    assert result.extra["vectorization_certificate"]["scalar_tensor_updates"] == 0
    assert result.extra["recipe"]["metadata"]["operation"] == "div"

    device_source = result.extra["sources"]["device/apu_g2_u16_div.cc"]
    assert "div(v.lhs_mmb, v.rhs_mmb, v.out);" in device_source
