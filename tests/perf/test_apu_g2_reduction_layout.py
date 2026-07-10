# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure-Python gates for APUg2 padded reduction layouts and host packing."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from allo.pim.apu_g2_layout import (
    APUG2DotTilePlan,
    APUG2DotTiling,
    APUG2_MMB_SETS,
    APUG2_U16_SHAPE,
    APUG2_VECTOR_LANES,
    APUG2ReductionPlan,
)
from allo.spmw_linear_layout import LinearLayout


def test_reduction_plan_pads_ninety_and_maps_sets_groups_and_slots():
    plan = APUG2ReductionPlan(90, 90, stream_extent=2)

    assert plan.output_extent == 90
    assert plan.reduction_extent == 90
    assert plan.padded_output_extent == 128
    assert plan.padded_reduction_extent == 128
    assert plan.stream_extent == plan.padded_stream_extent == 2
    assert plan.log_block_size == 7
    assert plan.active_block_count == 90
    assert isinstance(plan.layout, LinearLayout)
    assert plan.layout.out_dims == ("l1_column", "mmb_set")
    assert plan.layout.out_sizes == (APUG2_VECTOR_LANES, APUG2_MMB_SETS)
    assert plan.layout.size_of("row") == 128
    assert plan.layout.size_of("reduction") == 128
    assert plan.layout.size_of("stream") == 2

    assert plan.physical_coordinate(0, 0) == (0, 0)
    assert plan.physical_coordinate(3, 0) == (12288, 0)
    assert plan.physical_coordinate(15, 0) == (61440, 0)
    assert plan.physical_coordinate(16, 0) == (128, 0)
    assert plan.physical_coordinate(63, 0) == (61824, 0)
    assert plan.physical_coordinate(64, 0) == (512, 0)
    assert plan.physical_coordinate(89, 89, stream=1) == (37593, 1)
    assert plan.layout.apply(stream=1, row=127, reduction=127) == (62463, 1)

    groups = set()
    sets = set()
    physical = set()
    for stream in range(plan.stream_extent):
        for row in range(plan.output_extent):
            for reduction in range(plan.reduction_extent):
                column, mmb_set = plan.physical_coordinate(
                    row, reduction, stream=stream
                )
                groups.add(column // 4096)
                sets.add(mmb_set)
                physical.add((mmb_set, column))
    assert groups == set(range(16))
    assert sets == {0, 1}
    assert len(physical) == 2 * 90 * 90
    assert plan.layout.image_size(varying_inputs=("stream", "row", "reduction")) == (
        2 * 128 * 128
    )


def test_reduction_plan_is_immutable_and_checks_physical_capacity():
    plan = APUG2ReductionPlan(np.int64(90), np.int32(90))

    with pytest.raises(FrozenInstanceError):
        plan.output_extent = 91
    with pytest.raises(TypeError):
        plan.layout.bases["row"] = ()
    with pytest.raises(TypeError):
        plan.layout.bases["row"][0] = (0, 0)

    for output_extent, reduction_extent in ((0, 1), (1, 0), (-1, 2)):
        with pytest.raises(ValueError, match="positive"):
            APUG2ReductionPlan(output_extent, reduction_extent)
    for output_extent, reduction_extent in ((True, 1), (1, 1.5), ("90", 90)):
        with pytest.raises(TypeError, match="integer"):
            APUG2ReductionPlan(output_extent, reduction_extent)

    for stream_extent in (0, 5):
        with pytest.raises(ValueError, match="stream_extent"):
            APUG2ReductionPlan(90, 90, stream_extent=stream_extent)
    with pytest.raises(TypeError, match="stream_extent.*integer"):
        APUG2ReductionPlan(90, 90, stream_extent=1.5)

    # Both limits fit exactly: 16 groups * 4096 columns.
    APUG2ReductionPlan(16, 4096)
    APUG2ReductionPlan(65536, 1)
    with pytest.raises(ValueError, match="does not fit"):
        APUG2ReductionPlan(17, 4096)
    with pytest.raises(ValueError, match="does not fit"):
        APUG2ReductionPlan(65537, 1)


def test_matrix_pack_splits_parallel_streams_and_zero_fills_padding():
    plan = APUG2ReductionPlan(90, 90, stream_extent=2)
    streams = np.arange(2, dtype=np.uint16)[:, None, None]
    rows = np.arange(90, dtype=np.uint16)[:, None]
    reductions = np.arange(90, dtype=np.uint16)[None, :]
    matrices = (
        (streams << np.uint16(15))
        | (rows[None, :, :] << np.uint16(8))
        | reductions[None, :, :]
    ).astype(np.uint16)

    low, high = plan.pack_matrices(matrices)

    assert low.shape == high.shape == APUG2_U16_SHAPE
    assert low.dtype == high.dtype == np.uint8
    assert low.flags.c_contiguous and high.flags.c_contiguous
    used = np.zeros(APUG2_U16_SHAPE, dtype=np.bool_)
    for stream in range(2):
        for row in range(90):
            for reduction in range(90):
                column, mmb_set = plan.physical_coordinate(
                    row, reduction, stream=stream
                )
                used[mmb_set, column] = True
                assert low[mmb_set, column] == reduction
                assert high[mmb_set, column] == row + stream * 128
    assert np.all(low[~used] == 0)
    assert np.all(high[~used] == 0)

    # The singular helper places one GEMV matrix in a selected stream only.
    one_low, one_high = plan.pack_matrix(matrices[1], stream=1)
    assert np.all(one_low[0] == 0) and np.all(one_high[0] == 0)
    np.testing.assert_array_equal(one_low[1], low[1])
    np.testing.assert_array_equal(one_high[1], high[1])


def test_broadcast_pack_populates_each_active_block_once_and_leaves_padding_zero():
    plan = APUG2ReductionPlan(90, 90)
    vector = (
        np.uint16(0x8100) + np.arange(plan.reduction_extent, dtype=np.uint16)
    ).astype(np.uint16)

    low, high = plan.pack_broadcast_vector(vector)

    assert low.shape == high.shape == (APUG2_VECTOR_LANES,)
    assert low.dtype == high.dtype == np.uint8
    active_bases = {
        plan.physical_coordinate(row, 0)[0] for row in range(plan.output_extent)
    }
    assert len(active_bases) == plan.active_block_count
    used = np.zeros(APUG2_VECTOR_LANES, dtype=np.bool_)
    for base in active_bases:
        stop = base + plan.reduction_extent
        used[base:stop] = True
        np.testing.assert_array_equal(low[base:stop], np.arange(90, dtype=np.uint8))
        assert np.all(high[base:stop] == np.uint8(0x81))
        assert np.all(low[stop : base + plan.padded_reduction_extent] == 0)
        assert np.all(high[stop : base + plan.padded_reduction_extent] == 0)
    assert np.all(low[~used] == 0)
    assert np.all(high[~used] == 0)


def test_unpack_reduction_gathers_only_block_first_values():
    plan = APUG2ReductionPlan(90, 90, stream_extent=2)
    packed = np.full(APUG2_U16_SHAPE, 0xDEAD, dtype=np.uint16)
    expected = np.stack(
        [
            (
                np.uint16(0xE000 + stream * 0x1000)
                + np.arange(plan.output_extent, dtype=np.uint16)
            ).astype(np.uint16)
            for stream in range(plan.stream_extent)
        ]
    )
    for stream in range(plan.stream_extent):
        for row, value in enumerate(expected[stream]):
            column, mmb_set = plan.physical_coordinate(row, 0, stream=stream)
            packed[mmb_set, column] = value

    output = plan.unpack_reductions(packed)

    assert output.dtype == np.uint16
    assert output.shape == (plan.stream_extent, plan.output_extent)
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(plan.unpack_reduction(packed, stream=1), expected[1])


def test_reduction_packers_reject_wrong_dtypes_shapes_and_coordinates():
    plan = APUG2ReductionPlan(90, 90, stream_extent=2)
    matrix = np.zeros((90, 90), dtype=np.uint16)
    vector = np.zeros(90, dtype=np.uint16)
    packed = np.zeros(APUG2_U16_SHAPE, dtype=np.uint16)

    with pytest.raises(TypeError, match="matrix.*uint16"):
        plan.pack_matrix(matrix.astype(np.int16))
    with pytest.raises(ValueError, match="matrix.*shape"):
        plan.pack_matrix(np.zeros((90, 89), dtype=np.uint16))
    with pytest.raises(TypeError, match="matrices.*uint16"):
        plan.pack_matrices(np.zeros((2, 90, 90), dtype=np.int16))
    with pytest.raises(ValueError, match="matrices.*shape"):
        plan.pack_matrices(np.zeros((1, 90, 90), dtype=np.uint16))
    with pytest.raises(TypeError, match="vector.*uint16"):
        plan.pack_broadcast_vector(vector.astype(np.int16))
    with pytest.raises(ValueError, match="vector.*shape"):
        plan.pack_broadcast_vector(np.zeros(89, dtype=np.uint16))
    with pytest.raises(TypeError, match="packed reduction.*uint16"):
        plan.unpack_reduction(packed.astype(np.int16))
    with pytest.raises(ValueError, match="packed reduction.*shape"):
        plan.unpack_reduction(np.zeros((4, 65535), dtype=np.uint16))

    with pytest.raises(IndexError, match="row"):
        plan.physical_coordinate(90, 0)
    with pytest.raises(IndexError, match="reduction"):
        plan.physical_coordinate(0, 90)
    with pytest.raises(TypeError, match="row"):
        plan.physical_coordinate(True, 0)
    with pytest.raises(IndexError, match="stream"):
        plan.physical_coordinate(0, 0, stream=2)
    with pytest.raises(TypeError, match="stream"):
        plan.pack_matrix(matrix, stream=True)


def test_dot_tile_spends_all_four_sets_on_independent_outputs():
    plan = APUG2DotTilePlan(1000, 180)

    assert plan.padded_output_extent == 1024
    assert plan.padded_reduction_extent == 256
    assert plan.output_capacity == 1024
    assert plan.log_block_size == 8
    assert plan.layout.size_of("row") == 1024
    assert plan.layout.size_of("reduction") == 256
    assert plan.physical_coordinate(0, 0) == (0, 0)
    assert plan.physical_coordinate(1, 0) == (0, 1)
    assert plan.physical_coordinate(3, 0) == (0, 3)
    assert plan.physical_coordinate(4, 0) == (4096, 0)
    assert plan.physical_coordinate(63, 0) == (61440, 3)
    assert plan.physical_coordinate(64, 0) == (256, 0)
    assert plan.physical_coordinate(999, 179) == (40883, 3)

    physical = {
        plan.physical_coordinate(row, reduction)
        for row in range(plan.output_extent)
        for reduction in range(plan.reduction_extent)
    }
    assert len(physical) == 1000 * 180
    assert {mmb_set for _column, mmb_set in physical} == {0, 1, 2, 3}


def test_dot_tile_packs_two_arbitrary_operands_and_unpacks_results():
    plan = APUG2DotTilePlan(70, 50)
    left = (
        np.arange(70, dtype=np.uint16)[:, None] * np.uint16(257)
        + np.arange(50, dtype=np.uint16)[None, :]
    ).astype(np.uint16)
    right = (
        np.arange(70, dtype=np.uint16)[:, None] * np.uint16(509)
        + np.arange(50, dtype=np.uint16)[None, :] * np.uint16(3)
    ).astype(np.uint16)

    left_low, left_high, right_low, right_high = plan.pack_operands(left, right)
    for row in range(70):
        for reduction in range(50):
            column, mmb_set = plan.physical_coordinate(row, reduction)
            assert (
                int(left_low[mmb_set, column]) | int(left_high[mmb_set, column]) << 8
            ) == int(left[row, reduction])
            assert (
                int(right_low[mmb_set, column]) | int(right_high[mmb_set, column]) << 8
            ) == int(right[row, reduction])

    packed = np.full(APUG2_U16_SHAPE, 0xDEAD, dtype=np.uint16)
    expected = np.arange(70, dtype=np.uint16) * np.uint16(997)
    for row, value in enumerate(expected):
        column, mmb_set = plan.physical_coordinate(row)
        packed[mmb_set, column] = value
    np.testing.assert_array_equal(plan.unpack_reduction(packed), expected)


def test_dot_tiling_covers_rank_two_outputs_without_exceeding_vl64_capacity():
    # Canonical SMALL GEMM is A[P,Q] @ B[Q,R]: P*R=60*70 outputs, K=Q=80.
    tiling = APUG2DotTiling(60 * 70, 80)

    assert tiling.padded_reduction_extent == 128
    assert tiling.tile_capacity == 2048
    assert tiling.tile_count == 3
    assert [tiling.tile_bounds(tile) for tile in range(3)] == [
        (0, 2048),
        (2048, 4096),
        (4096, 4200),
    ]
    assert [tiling.tile_plan(tile).output_extent for tile in range(3)] == [
        2048,
        2048,
        104,
    ]


def test_dot_tile_chunks_wide_reductions_and_rejects_oversized_physical_tiles():
    with pytest.raises(ValueError, match="at most 256"):
        APUG2DotTilePlan(1, 257)
    with pytest.raises(ValueError, match="exceeds capacity"):
        APUG2DotTilePlan(1025, 180)

    tiling = APUG2DotTiling(2050, 600)
    assert tiling.tile_capacity == 1024
    assert tiling.tile_count == 3
    assert tiling.reduction_tile_count == 3
    assert tiling.task_count == 9
    assert tiling.reduction_bounds(0) == (0, 256)
    assert tiling.reduction_bounds(1) == (256, 512)
    assert tiling.reduction_bounds(2) == (512, 600)
    assert tiling.tile_plan(2, 2).output_extent == 2
    assert tiling.tile_plan(2, 2).reduction_extent == 88
