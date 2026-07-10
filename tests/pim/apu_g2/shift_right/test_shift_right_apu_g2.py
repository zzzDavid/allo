# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-card uint16 logical right-shift primitive gate for APUg2."""

import numpy as np
import pytest

from allo.pim.apu_g2_shift_right_runtime import run_apu_g2_u16_shift_right


@pytest.mark.apu_g2_device
def test_shift_right_apu_g2_hardware(apu_g2_device_gate):
    rng = np.random.default_rng(20260708)
    shape = (4, 65536)
    src = rng.integers(0, 65536, size=shape, dtype=np.uint16)
    src[0, :16] = np.array(
        [0, 1, 2, 3, 7, 8, 15, 16, 255, 256, 1023, 1024, 32768, 65534, 65535, 12345],
        dtype=np.uint16,
    )
    shift = 3
    expected = (src.astype(np.uint32) >> np.uint32(shift)).astype(np.uint16)

    with pytest.raises(ValueError, match=r"\[0, 15\]"):
        run_apu_g2_u16_shift_right(src, 16, repetitions=1)

    import conftest

    with conftest.apu_g2_board_lock():
        result = run_apu_g2_u16_shift_right(src, shift, repetitions=128)

    np.testing.assert_array_equal(result.extra["outputs"]["out"], expected)
    assert result.backend == "apu_v2"
    assert result.cycles > 0
    assert result.extra["total_ticks"] > 0
    assert result.extra["final_pipeline_ticks"] > 0
    assert result.extra["per_call_ticks"]["pipeline"] > 0
    assert result.extra["vectorization_certificate"]["scalar_tensor_updates"] == 0
    assert result.extra["recipe"]["metadata"]["operation"] == "shift_right"
    assert result.extra["recipe"]["metadata"]["shift"] == shift

    device_source = result.extra["sources"]["device/apu_g2_u16_shift_right.cc"]
    assert "shift_right(v.src_out_mmb, shift);" in device_source
