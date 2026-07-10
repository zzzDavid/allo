# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-card uint16 block-sum primitive gate for APUg2."""

import numpy as np
import pytest

from allo.pim.apu_g2_block_sum_runtime import run_apu_g2_u16_block_sum


def _expected_block_sums(src, log_block_size):
    block_size = 1 << log_block_size
    blocks_per_group = 4096 // block_size
    grouped = src.reshape(4, 16, 4096)
    sums = grouped.reshape(4, 16, blocks_per_group, block_size).astype(np.uint64).sum(
        axis=3,
        dtype=np.uint64,
    )
    return (sums & np.uint64(0xFFFF)).astype(np.uint16)


@pytest.mark.apu_g2_device
def test_block_sum_apu_g2_hardware(apu_g2_device_gate):
    rng = np.random.default_rng(20260708)
    shape = (4, 65536)
    src = rng.integers(0, 65536, size=shape, dtype=np.uint16)
    src[0, :512] = np.arange(512, dtype=np.uint16)
    src[1, :512] = np.uint16(65535)
    log_block_size = 8
    expected = _expected_block_sums(src, log_block_size)

    with pytest.raises(ValueError, match=r"\[0, 8\]"):
        run_apu_g2_u16_block_sum(src, log_block_size=9, repetitions=1)

    import conftest

    with conftest.apu_g2_board_lock():
        result = run_apu_g2_u16_block_sum(
            src,
            log_block_size=log_block_size,
            repetitions=8,
        )

    np.testing.assert_array_equal(result.extra["outputs"]["block_sums"], expected)
    assert result.backend == "apu_v2"
    assert result.cycles > 0
    assert result.extra["total_ticks"] > 0
    assert result.extra["final_pipeline_ticks"] > 0
    assert result.extra["per_call_ticks"]["pipeline"] > 0
    assert result.extra["vectorization_certificate"]["scalar_tensor_updates"] == 0
    assert result.extra["recipe"]["metadata"]["operation"] == "block_sum"
    assert result.extra["recipe"]["metadata"]["log_block_size"] == log_block_size

    device_source = result.extra["sources"]["device/apu_g2_u16_block_sum.cc"]
    assert "sum(v.sum_source, log_block_size, v.sum_destination);" in device_source
    assert "copy(v.sum_low16, v.out);" in device_source
