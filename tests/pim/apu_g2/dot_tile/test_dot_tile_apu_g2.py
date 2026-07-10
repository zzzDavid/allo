# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-card gate for arbitrary four-set uint16 dot-product tiles."""

import numpy as np
import pytest

from allo.pim.apu_g2_dot_tile_runtime import run_apu_g2_u16_dot_tile
from allo.pim.apu_g2_layout import APUG2DotTilePlan


@pytest.mark.apu_g2_device
def test_dot_tile_k180_all_sets_hardware(apu_g2_device_gate):
    output_extent = 1000
    reduction_extent = 180
    rows = np.arange(output_extent, dtype=np.uint64)[:, None]
    reductions = np.arange(reduction_extent, dtype=np.uint64)[None, :]
    left = ((rows * 40507 + reductions * 7919 + 65521) & 0xFFFF).astype(np.uint16)
    right = ((rows * 32749 + reductions * 65521 + 43219) & 0xFFFF).astype(np.uint16)
    left[0, :6] = [0, 1, 65535, 65535, 32768, 60000]
    right[0, :6] = [65535, 65535, 1, 65535, 32768, 60000]
    expected = (
        np.sum(
            left.astype(np.uint64) * right.astype(np.uint64),
            axis=1,
            dtype=np.uint64,
        )
        & np.uint64(0xFFFF)
    ).astype(np.uint16)

    plan = APUG2DotTilePlan(output_extent, reduction_extent)
    assert {plan.physical_coordinate(row)[1] for row in range(output_extent)} == {
        0,
        1,
        2,
        3,
    }

    import conftest

    with conftest.apu_g2_board_lock():
        result = run_apu_g2_u16_dot_tile(left, right, repetitions=8)

    np.testing.assert_array_equal(result.extra["outputs"]["out"], expected)
    assert result.backend == "apu_v2"
    assert result.cycles > 0
    assert result.extra["repetitions"] == 8
    assert result.extra["total_ticks"] > 0
    assert result.extra["final_pipeline_ticks"] > 0
    assert result.extra["layout"]["padded_reduction_extent"] == 256
    assert result.extra["layout"]["active_sets"] == [0, 1, 2, 3]

    source = result.extra["sources"]["device/apu_g2_u16_dot_tile.cc"]
    assert "L1Vectors right_low" in source
    assert source.count("mul(") >= 1
    assert source.count("sum(") >= 1
    assert "sum(v.sum_source, log_block_size" in source
    assert "seu_barrier();" in source
    assert "simulator" not in source.lower()


@pytest.mark.apu_g2_device
def test_dot_tile_fused_epilogue_k70_hardware(apu_g2_device_gate):
    # The integer-ratio policy preserves canonical alpha:beta = 1.5:1.2 as 5:4.
    alpha, beta = 5, 4
    output_extent = 1000
    reduction_extent = 70
    rng = np.random.default_rng(20260709)
    left = rng.integers(
        0, 65536, size=(output_extent, reduction_extent), dtype=np.uint16
    )
    right = rng.integers(
        0, 65536, size=(output_extent, reduction_extent), dtype=np.uint16
    )
    accumulator = rng.integers(1, 65536, size=output_extent, dtype=np.uint16)
    dot = (
        np.sum(
            left.astype(np.uint64) * right.astype(np.uint64),
            axis=1,
            dtype=np.uint64,
        )
        & np.uint64(0xFFFF)
    ).astype(np.uint16)
    expected = (
        (
            np.uint64(alpha) * dot.astype(np.uint64)
            + np.uint64(beta) * accumulator.astype(np.uint64)
        )
        & np.uint64(0xFFFF)
    ).astype(np.uint16)

    import conftest

    with conftest.apu_g2_board_lock():
        result = run_apu_g2_u16_dot_tile(
            left,
            right,
            accumulator=accumulator,
            alpha=alpha,
            beta=beta,
            repetitions=8,
        )

    np.testing.assert_array_equal(result.extra["outputs"]["out"], expected)
    assert result.cycles > 0
    assert result.extra["layout"]["padded_reduction_extent"] == 128
    assert result.extra["layout"]["active_sets"] == [0, 1, 2, 3]
    assert result.extra["epilogue"] == {
        "enabled": True,
        "alpha": 5,
        "beta": 4,
        "inventory": {"mul": 6, "shift_left": 4, "add": 5},
    }
    source = result.extra["sources"]["device/apu_g2_u16_dot_tile.cc"]
    assert "multiply_scalar_u16" in source
    assert "epilogue(v, p.alpha, p.beta);" in source
