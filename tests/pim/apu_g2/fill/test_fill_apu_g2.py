# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-card uint16 scalar-fill primitive gate for APUg2."""

import numpy as np
import pytest

from allo.pim.apu_g2_fill_runtime import run_apu_g2_u16_fill


@pytest.mark.apu_g2_device
def test_fill_apu_g2_hardware(apu_g2_device_gate):
    value = 0xA55A
    expected = np.full((4, 65536), value, dtype=np.uint16)

    with pytest.raises(ValueError, match="fit uint16"):
        run_apu_g2_u16_fill(65536, repetitions=1)
    with pytest.raises(TypeError, match="integer"):
        run_apu_g2_u16_fill(True, repetitions=1)

    import conftest

    with conftest.apu_g2_board_lock():
        result = run_apu_g2_u16_fill(value, repetitions=128)

    np.testing.assert_array_equal(result.extra["outputs"]["out"], expected)
    assert result.backend == "apu_v2"
    assert result.cycles > 0
    assert result.extra["total_ticks"] > 0
    assert result.extra["final_pipeline_ticks"] > 0
    assert result.extra["per_call_ticks"]["pipeline"] > 0
    assert result.extra["vectorization_certificate"]["scalar_tensor_updates"] == 0
    assert result.extra["recipe"]["metadata"]["operation"] == "fill"
    assert result.extra["recipe"]["metadata"]["value"] == value

    device_source = result.extra["sources"]["device/apu_g2_u16_fill.cc"]
    assert "copy(static_cast<uint64_t>(value & 0xffffu), v.out_mmb);" in device_source
    assert "copy(v.out_mmb, v.out);" in device_source
