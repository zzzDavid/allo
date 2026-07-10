# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Executable cost gates for Gemini-II uint16 direct-VL64 kernels."""

import pytest

from allo.perf import CostEvent, ExecutionGraph
from allo.pim.apu_g2_pipeline import (
    NORMALIZED_DUAL_STREAM_REDUCTION_CALIBRATION,
)
from allo.pim.costs.apu_g2 import (
    APUG2_ADD_U16_TICKS,
    APUG2_COPY_L1_TO_MMB_SEG0_U16_TICKS,
    APUG2_COPY_L1_TO_MMB_SEG1_U16_TICKS,
    APUG2_COPY_MMB_TO_L1_U16_TICKS,
    APUG2_GEMV_ADD_CALLS,
    APUG2_GEMV_BARRIER_CALLS,
    APUG2_GEMV_FULL_PIPELINE_TICKS,
    APUG2_GEMV_MEASURED_TICKS_PER_PIPELINE,
    APUG2_GEMV_MUL_CALLS,
    APUG2_GEMV_REDUCE_CALLS,
    APUG2_GEMV_SHIFT_CALLS,
    APUG2_GESUMMV_ADD_CALLS,
    APUG2_GESUMMV_ADD_U16_TICKS,
    APUG2_GESUMMV_BARRIER_CALLS,
    APUG2_GESUMMV_FULL_PIPELINE_TICKS,
    APUG2_GESUMMV_GROUP_REDUCE_U16_TO_U23_TICKS,
    APUG2_GESUMMV_MEASURED_TICKS_PER_PIPELINE,
    APUG2_GESUMMV_MUL_CALLS,
    APUG2_GESUMMV_MUL_U8_TO_U16_TICKS,
    APUG2_GESUMMV_REDUCE_CALLS,
    APUG2_GESUMMV_SEU_BARRIER_TICKS,
    APUG2_GESUMMV_SHIFT_CALLS,
    APUG2_GESUMMV_SHIFT_LEFT_U16_TICKS,
    APUG2_SEU_BARRIER_TICKS,
    apu_g2_cost,
    build_apu_g2_add_graph,
    build_apu_g2_gemv_graph,
    build_apu_g2_gesummv_graph,
    estimate_apu_g2_add,
    estimate_apu_g2_gemv,
    estimate_apu_g2_gesummv,
    estimate_apu_g2_u16_gemm_wall_us,
)
from allo.pim.targets import build_apu_g2_target


def test_transport_model_tracks_measured_large_gemm_and_gemv_wall():
    gemm = estimate_apu_g2_u16_gemm_wall_us(1000, 1200, 1100)
    gemv = estimate_apu_g2_u16_gemm_wall_us(1900, 2100, 1, batch_columns=1)

    assert gemm["hardware_tasks"] == 360
    assert gemm["output_readbacks"] == 36
    assert gemm["h2d_us"] == pytest.approx(1_264_712.441, abs=0.001)
    assert gemm["host_task_us"] == pytest.approx(2_839_283.568, abs=0.001)
    assert gemm["d2h_us"] == pytest.approx(54_754.593, abs=0.001)
    assert gemm["wall_us"] == pytest.approx(4_158_750.602, abs=0.001)
    assert gemv["hardware_tasks"] == 17
    assert gemv["output_readbacks"] == 1
    assert gemv["index_bytes"] == 8_192
    assert gemv["h2d_calls"] == 19
    assert gemv["h2d_bytes"] == (
        gemv["index_bytes"] + gemv["weight_bytes"] + gemv["accumulator_bytes"]
    )
    assert gemv["h2d_us"] == pytest.approx(69_388.553, abs=0.001)
    assert gemv["host_task_us"] == pytest.approx(16_293.35, abs=0.001)
    assert gemv["d2h_us"] == pytest.approx(864.646, abs=0.001)
    assert gemv["wall_us"] == pytest.approx(86_546.549, abs=0.001)


def test_transport_h2d_regime_depends_on_column_batches_not_gemv_shape():
    one_column = estimate_apu_g2_u16_gemm_wall_us(1900, 2100, 1)
    two_columns = estimate_apu_g2_u16_gemm_wall_us(1900, 2100, 2)
    two_batches = estimate_apu_g2_u16_gemm_wall_us(1900, 2100, 32)

    assert one_column["steady_batch_bytes"] == 0
    assert two_columns["steady_batch_bytes"] == 0
    assert two_batches["steady_batch_bytes"] > 0
    assert one_column["cold_batch_bytes"] == (
        one_column["index_bytes"]
        + one_column["weight_bytes"]
        + one_column["accumulator_bytes"]
    )
    assert two_columns["cold_batch_bytes"] == (
        two_columns["index_bytes"]
        + two_columns["weight_bytes"]
        + two_columns["accumulator_bytes"]
    )


def test_apu_g2_cost_expands_one_add_event_into_the_vl64_pipeline():
    target = build_apu_g2_target()
    graph = build_apu_g2_add_graph(target, apu_g2_cost, vl64_calls=1)

    assert graph.metadata["target"] == "apu_v2"
    assert graph.metadata["dtype"] == "uint16"
    assert graph.metadata["shape"] == (4, 65536)
    assert graph.metadata["coalesced_groups"] == 16
    assert [activity.label for activity in graph.activities] == [
        "copy_l1_to_mmb_seg0_u16",
        "copy_l1_to_mmb_seg1_u16",
        "add_u16",
        "copy_mmb_to_l1_u16",
        "seu_barrier",
    ]
    assert sum(activity.label == "add_u16" for activity in graph.activities) == 1
    assert all(
        activity.id.startswith("apu_g2:add_u16:cost:") for activity in graph.activities
    )
    assert [activity.latency_cycles for activity in graph.activities] == [
        APUG2_COPY_L1_TO_MMB_SEG0_U16_TICKS,
        APUG2_COPY_L1_TO_MMB_SEG1_U16_TICKS,
        APUG2_ADD_U16_TICKS,
        APUG2_COPY_MMB_TO_L1_U16_TICKS,
        APUG2_SEU_BARRIER_TICKS,
    ]

    estimate = estimate_apu_g2_add(target, apu_g2_cost)
    assert estimate.cycles == sum(
        activity.latency_cycles for activity in graph.activities
    )
    assert estimate.cycles > 0


@pytest.mark.parametrize("vl64_calls", [0, 2, 16, True, 1.0])
def test_apu_g2_cost_rejects_anything_except_one_integer_vl64_call(vl64_calls):
    target = build_apu_g2_target()

    with pytest.raises(ValueError, match="exactly one VL64|requires vl64_calls=1"):
        build_apu_g2_add_graph(target, apu_g2_cost, vl64_calls=vl64_calls)


def test_apu_g2_cost_resources_are_core_wide_not_replicated_per_pe():
    target = build_apu_g2_target()
    graph = build_apu_g2_add_graph(target, apu_g2_cost)
    add = next(activity for activity in graph.activities if activity.label == "add_u16")
    paths = {occupancy.handle.path for occupancy in add.occupancy}

    assert "apu_v2/core/vector_engine/ADD_U16" in paths
    assert "apu_v2/core/vector_engine" in paths
    assert "apu_v2/core/mmb" in paths
    assert not any("/pe" in path for path in paths)


def test_apu_g2_gesummv_has_exact_proven_vl64_inventory_and_calibration():
    target = build_apu_g2_target()
    graph = build_apu_g2_gesummv_graph(
        target,
        apu_g2_cost,
        output_extent=90,
        reduction_extent=90,
    )

    labels = [activity.label for activity in graph.activities]
    assert labels.count("dual_stream_mul_u8_to_u16") == APUG2_GESUMMV_MUL_CALLS
    assert (
        labels.count("dual_stream_group_reduce_add_u16_to_u23")
        == APUG2_GESUMMV_REDUCE_CALLS
    )
    assert labels.count("dual_stream_shift_left_u16") == APUG2_GESUMMV_SHIFT_CALLS
    assert labels.count("dual_stream_add_u16") == APUG2_GESUMMV_ADD_CALLS
    assert labels.count("seu_barrier") == APUG2_GESUMMV_BARRIER_CALLS
    assert len(labels) == 19

    assert graph.metadata["vl64_compute_calls"] == 18
    assert graph.metadata["vl64_barrier_calls"] == 1
    assert graph.metadata["calibration_ticks"] == 8_001
    assert graph.metadata["measured_ticks_per_pipeline"] == pytest.approx(8_001.375)
    assert graph.metadata["calibration_repetitions"] == 8
    assert (
        graph.metadata["calibration_basis"]
        == "real_card_full_pipeline_normalized_attribution"
    )
    estimate = estimate_apu_g2_gesummv(
        target,
        apu_g2_cost,
        output_extent=90,
        reduction_extent=90,
    )
    assert estimate.cycles == APUG2_GESUMMV_FULL_PIPELINE_TICKS
    assert estimate.cycles == (
        APUG2_GESUMMV_MUL_CALLS * APUG2_GESUMMV_MUL_U8_TO_U16_TICKS
        + APUG2_GESUMMV_REDUCE_CALLS * APUG2_GESUMMV_GROUP_REDUCE_U16_TO_U23_TICKS
        + APUG2_GESUMMV_SHIFT_CALLS * APUG2_GESUMMV_SHIFT_LEFT_U16_TICKS
        + APUG2_GESUMMV_ADD_CALLS * APUG2_GESUMMV_ADD_U16_TICKS
        + APUG2_GESUMMV_BARRIER_CALLS * APUG2_GESUMMV_SEU_BARRIER_TICKS
    )


def test_apu_g2_gesummv_n90_records_padding_without_multiplying_calls():
    target = build_apu_g2_target()
    graph = build_apu_g2_gesummv_graph(
        target,
        output_extent=90,
        reduction_extent=90,
    )

    assert graph.metadata["shape"] == (90, 90)
    assert graph.metadata["padded_output_extent"] == 128
    assert graph.metadata["padded_reduction_extent"] == 128
    assert graph.metadata["log_block_size"] == 7
    assert graph.metadata["coalesced_groups"] == 16
    assert graph.metadata["matrix_streams"] == 2
    # Calls are not unrolled over 90 rows, 16 groups, or the A/B streams.
    assert len(graph.activities) == 19
    assert all(
        activity.metadata["coalesced_spmw_axis"] == "group"
        for activity in graph.activities
    )
    assert all(
        activity.metadata["attribution"] == "normalized_full_pipeline_not_isolated"
        for activity in graph.activities
    )
    assert all("kernel" not in activity.metadata for activity in graph.activities)
    assert all(
        not any("/pe" in occupancy.handle.path for occupancy in activity.occupancy)
        for activity in graph.activities
    )


@pytest.mark.parametrize(
    "output_extent,reduction_extent,exception",
    [
        (0, 90, ValueError),
        (90, 0, ValueError),
        (True, 90, TypeError),
        (90, 1.5, TypeError),
        (1025, 90, ValueError),
        (1, 257, ValueError),
    ],
)
def test_apu_g2_gesummv_reuses_reduction_layout_validation(
    output_extent, reduction_extent, exception
):
    target = build_apu_g2_target()

    with pytest.raises(exception):
        build_apu_g2_gesummv_graph(
            target,
            output_extent=output_extent,
            reduction_extent=reduction_extent,
        )


def test_apu_g2_accumulated_gemv_inventory_and_real_card_calibration():
    target = build_apu_g2_target()
    graph = build_apu_g2_gemv_graph(target, output_extent=120, reduction_extent=120)
    labels = [activity.label for activity in graph.activities]

    assert labels.count("single_stream_mul_u8_to_u16") == APUG2_GEMV_MUL_CALLS == 3
    assert (
        labels.count("single_stream_group_reduce_add_u16_to_u23")
        == APUG2_GEMV_REDUCE_CALLS
        == 3
    )
    assert labels.count("single_stream_shift_left_u16") == APUG2_GEMV_SHIFT_CALLS == 2
    assert labels.count("single_stream_add_u16") == APUG2_GEMV_ADD_CALLS == 3
    assert labels.count("seu_barrier") == APUG2_GEMV_BARRIER_CALLS == 1
    assert len(labels) == 12
    assert graph.metadata["vl64_compute_calls"] == 11
    assert graph.metadata["padded_reduction_extent"] == 128
    assert graph.metadata["log_block_size"] == 7
    assert graph.metadata["matrix_streams"] == 1
    assert graph.metadata["calibration_ticks"] == 6_893
    assert graph.metadata["measured_ticks_per_pipeline"] == pytest.approx(
        APUG2_GEMV_MEASURED_TICKS_PER_PIPELINE
    )
    assert graph.metadata["calibration_repetitions"] == 8
    assert all(
        activity.metadata["attribution"] == "normalized_full_pipeline_not_isolated"
        for activity in graph.activities
    )
    estimate = estimate_apu_g2_gemv(target, output_extent=120, reduction_extent=120)
    assert estimate.cycles == APUG2_GEMV_FULL_PIPELINE_TICKS == 6_893


def test_normalized_vl64_cost_is_identity_invariant():
    target = build_apu_g2_target()
    bound = apu_g2_cost.bind(target)

    def evaluate(event_id, diagnostic):
        graph = ExecutionGraph(name=diagnostic)
        event = CostEvent.create(
            event_id,
            target.op("MUL_U8_TO_U16"),
            metrics={
                "vl64_calls": 1,
                "pipeline_calibration": (NORMALIZED_DUAL_STREAM_REDUCTION_CALIBRATION),
                "coalesced_matrix_streams": 2,
            },
            attributes={"diagnostic": diagnostic},
        )
        bound.emit(graph, event)
        return bound.evaluate(graph)

    original = evaluate("arbitrary:first", "first_name")
    renamed = evaluate("unrelated:second", "renamed_isomorphic_event")

    assert original.cycles == renamed.cycles == APUG2_GESUMMV_MUL_U8_TO_U16_TICKS
