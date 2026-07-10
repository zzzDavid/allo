# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-card uint16 floor-square-root macro gate for APUg2."""

import numpy as np
import pytest

from allo.pim.apu_g2_sqrt_runtime import (
    run_apu_g2_u16_sqrt,
    sqrt_u16_reference,
)


@pytest.mark.apu_g2_device
def test_sqrt_apu_g2_hardware(apu_g2_device_gate):
    rng = np.random.default_rng(20260708)
    src = rng.integers(0, 65536, size=(4, 65536), dtype=np.uint16)
    src[0, :24] = np.array(
        [
            0,
            1,
            2,
            3,
            4,
            8,
            15,
            16,
            17,
            24,
            25,
            26,
            255,
            256,
            257,
            1023,
            1024,
            1025,
            32767,
            32768,
            65024,
            65025,
            65026,
            65535,
        ],
        dtype=np.uint16,
    )
    expected = sqrt_u16_reference(src)

    import conftest

    with conftest.apu_g2_board_lock():
        result = run_apu_g2_u16_sqrt(src, repetitions=8)

    np.testing.assert_array_equal(result.extra["outputs"]["out"], expected)
    assert result.backend == "apu_v2"
    assert result.cycles > 0
    assert result.extra["total_ticks"] > 0
    assert result.extra["final_pipeline_ticks"] > 0
    assert result.extra["per_call_ticks"]["pipeline"] > 0
    assert result.extra["vectorization_certificate"]["scalar_tensor_updates"] == 0
    assert result.extra["vectorization_certificate"]["macro"] == (
        "binary_restoring_floor_sqrt_u16"
    )

    device_source = result.extra["sources"]["device/apu_g2_u16_sqrt.cc"]
    assert "GSI_LIBRARY_ENTRY_POINT(tenon_apu_g2_u16_sqrt" in device_source
    assert "lt(v.compare_lhs, v.work, v.predicate);" in device_source
    assert "mul(v.operand0, v.operand1, v.work);" in device_source
    assert "simulator" not in device_source.lower()
