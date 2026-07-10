# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Executable-cost lowering for immutable APU v1 vector plans."""

from dataclasses import replace

import allo
import pytest
from allo.ir.types import float16, int32, uint16

from allo.pim.apu_v1_layout import PlanOperation, Transfer
from allo.pim.apu_v1_vector_codegen import SpatialResidentReductionEmission
from allo.pim.apu_v1_vector_cost import (
    build_apu_v1_plan_graph,
    estimate_apu_v1_plan,
    estimate_apu_v1_realization,
    materialized_apu_v1_operation_inventory,
    rank_apu_v1_plans,
)
from allo.pim.apu_v1_vector_program import _realize
from allo.pim.apu_v1_vectorize import generate_apu_v1_vectorization_candidates
from allo.pim.costs.apu_v1 import apu_v1_cost
from allo.pim.targets import build_apu_v1_target


def micro_1k(
    left: float16[1024, 64],
    right: float16[64, 1024],
    result: float16[1024, 1024],
):
    for row, column in allo.grid(1024, 1024):
        for depth in allo.reduction(64):
            result[row, column] += left[row, depth] * right[depth, column]


def binary_micro(left: int32[16, 32], right: int32[32, 8], result: int32[16, 8]):
    for row, column in allo.grid(16, 8):
        for depth in allo.reduction(32):
            result[row, column] += left[row, depth] & right[depth, column]


def streamed_micro(
    left: uint16[256, 128],
    right: uint16[128, 1024],
    result: uint16[256, 1024],
):
    for row, column in allo.grid(256, 1024):
        for depth in allo.reduction(128):
            result[row, column] += left[row, depth] * right[depth, column]


def spatial_gemv(
    left: uint16[1900, 2100],
    right: uint16[2100, 1],
    result: uint16[1900, 1],
):
    for row, column in allo.grid(1900, 1):
        for depth in allo.reduction(2100):
            result[row, column] += left[row, depth] * right[depth, column]


def large_spatial_gemv(
    left: uint16[70000, 2100],
    right: uint16[2100, 1],
    result: uint16[70000, 1],
):
    for row, column in allo.grid(70000, 1):
        for depth in allo.reduction(2100):
            result[row, column] += left[row, depth] * right[depth, column]


def renamed_large_spatial_gemv(
    matrix: uint16[70000, 2100],
    vector: uint16[2100, 1],
    destination: uint16[70000, 1],
):
    for output_row, singleton_column in allo.grid(70000, 1):
        for reduction_index in allo.reduction(2100):
            destination[output_row, singleton_column] += (
                matrix[output_row, reduction_index]
                * vector[reduction_index, singleton_column]
            )


@pytest.fixture(scope="module")
def plans():
    module = allo.customize(micro_1k, enable_tensor=False).module
    return tuple(
        candidate.plan for candidate in generate_apu_v1_vectorization_candidates(module)
    )


def test_plan_graph_uses_structural_operations_moves_and_vector_counts(plans):
    target = build_apu_v1_target()
    bound = apu_v1_cost.bind(target)
    optimized = plans[-1]
    graph = build_apu_v1_plan_graph(optimized, target, bound)

    assert graph.activities
    assert graph.metadata["vector_facts"]["vector_calls"] > 0
    primitives = {activity.primitive for activity in graph.activities}
    assert any(name.endswith("/MUL") for name in primitives)
    assert any(name.endswith("/DMA_L4_TO_L1_32K") for name in primitives)
    assert any(name.endswith("/LOAD_L1_TO_VR16") for name in primitives)
    assert any(name.endswith("/LOOKUP_16") for name in primitives)
    assert any(name.endswith("/DUPLICATE_SUBGROUP_16") for name in primitives)
    assert any(name.endswith("/DMA_L1_TO_L4_32K") for name in primitives)
    assert all(activity.latency_cycles > 0 for activity in graph.activities)


def test_full_micro_shape_ranks_accumulator_blocks_by_rhs_reuse(plans):
    target = build_apu_v1_target()
    ranked = rank_apu_v1_plans(plans, target, apu_v1_cost)

    cycles = [item.cycles for item in ranked]
    assert all(cycle > 0 for cycle in cycles)
    assert len(set(cycles)) == 10
    assert [item.plan.name for item in ranked] == [
        "temporal_dma_coalescing_broadcast_friendly_acc8",
        "temporal_dma_coalescing_broadcast_friendly_acc4",
        "temporal_dma_coalescing_broadcast_friendly_acc2",
        "temporal_dma_coalescing_broadcast_friendly_acc8_l4",
        "temporal_dma_coalescing_broadcast_friendly_acc4_l4",
        "temporal_dma_coalescing_broadcast_friendly",
        "temporal_dma_coalescing_broadcast_friendly_acc2_l4",
        "temporal_dma_coalescing",
        "baseline_spatial_reduction",
        "temporal_svp",
    ]


def test_transfer_traffic_is_derived_from_route_layout_relations(plans):
    target = build_apu_v1_target()
    plan = next(
        item
        for item in plans
        if item.name == "temporal_dma_coalescing_broadcast_friendly"
    )
    result = estimate_apu_v1_plan(plan, target, apu_v1_cost)
    routes = {
        (route["value"], step["kind"]): step
        for route in result.graph.metadata["transfer_routes"]
        for step in route["steps"]
    }

    lhs_dma = routes[("left", "dma_l4_l3")]
    lhs_lookup = routes[("left", "lookup")]
    rhs_dma = routes[("right", "dma_l4_l1_32k")]
    rhs_duplicate = routes[("right", "duplicate_subgroup")]

    assert lhs_dma["call_count"] == 1
    assert lhs_dma["source_elements_per_call"] == 1024 * 64
    assert lhs_lookup["call_count"] == 32 * 64
    assert lhs_lookup["source_elements_per_call"] == 32
    assert lhs_lookup["destination_elements_per_call"] == 32768
    assert lhs_lookup["expansion_factor"] == 1024
    assert rhs_dma["call_count"] == 8
    assert rhs_dma["resident_reuse_factor"] == 32
    assert rhs_duplicate["call_count"] == 32 * 64
    # Four physical 1K subgroups are resident in parallel; one GVML call
    # expands them to four 8K groups.
    assert rhs_duplicate["source_elements_per_call"] == 4 * 1024
    assert rhs_duplicate["destination_elements_per_call"] == 32768
    assert rhs_duplicate["expansion_factor"] == 8

    # Repeats are summarized in cost activities, not unrolled 2,048 times.
    assert len(result.graph.activities) < 20


def test_transfer_cost_does_not_consume_planner_broadcast_metadata(plans):
    target = build_apu_v1_target()
    optimized = plans[-1]
    original = estimate_apu_v1_plan(optimized, target, apu_v1_cost).cycles
    poisoned = replace(
        optimized,
        metadata={
            **dict(optimized.metadata),
            "broadcast_geometry": {"left": {"table_size": 1}},
            "transfer_facts": {"input_transfer_calls": 1},
        },
    )

    assert estimate_apu_v1_plan(poisoned, target, apu_v1_cost).cycles == original


def test_lookup_cost_prices_l4_source_from_the_explicit_route(plans):
    target = build_apu_v1_target()
    l3 = next(item for item in plans if item.name.endswith("acc8"))
    l4 = next(item for item in plans if item.name.endswith("acc8_l4"))

    l3_result = estimate_apu_v1_plan(l3, target, apu_v1_cost)
    l4_result = estimate_apu_v1_plan(l4, target, apu_v1_cost)
    l4_lookup = next(
        step
        for route in l4_result.graph.metadata["transfer_routes"]
        for step in route["steps"]
        if step["kind"] == "lookup"
    )

    assert l4_lookup["source_storage"] == "l4"
    assert l4_result.cycles > l3_result.cycles


def test_streamed_resident_route_prices_each_accumulator_block_replay():
    module = allo.customize(streamed_micro, enable_tensor=False).module
    plan = generate_apu_v1_vectorization_candidates(module)[-1].plan
    result = estimate_apu_v1_plan(plan, build_apu_v1_target(), apu_v1_cost)
    routes = {
        (route["value"], step["kind"]): step
        for route in result.graph.metadata["transfer_routes"]
        for step in route["steps"]
    }

    rhs_dma = routes[("right", "dma_l4_l1_32k")]
    assert plan.accumulator_block == 8
    assert rhs_dma["call_count"] == 16
    assert rhs_dma["resident_reuse_factor"] == 1
    assert rhs_dma["streaming_replay_factor"] == 1
    rhs_duplicate = routes[("right", "duplicate_subgroup")]
    assert rhs_duplicate["call_count"] == 128


def test_spatial_gemv_prices_matrix_tiles_and_resident_vector_bank():
    module = allo.customize(spatial_gemv, enable_tensor=False).module
    plans = generate_apu_v1_vectorization_candidates(module)
    plan = next(
        candidate.plan
        for candidate in plans
        if candidate.name == "spatial_gemv_group_reduction"
    )
    result = estimate_apu_v1_plan(plan, build_apu_v1_target(), apu_v1_cost)
    routes = {
        (route["value"], step["kind"]): step
        for route in result.graph.metadata["transfer_routes"]
        for step in route["steps"]
    }

    assert plan.metadata["tile_sizes"] == {
        "row": 128,
        "column": 1,
        "depth": 256,
    }
    assert routes[("left", "dma_l4_l1_32k")]["call_count"] == 135
    assert routes[("right", "dma_l4_l1_32k")]["call_count"] == 9
    assert routes[("right", "dma_l4_l1_32k")]["resident_reuse_factor"] == 15
    assert result.cycles > 0
    temporal = next(
        candidate.plan
        for candidate in plans
        if candidate.name == "temporal_dma_coalescing"
    )
    assert (
        result.cycles
        < estimate_apu_v1_plan(temporal, build_apu_v1_target(), apu_v1_cost).cycles
    )


def test_materialized_spatial_gemv_inventory_uses_exact_ragged_tile_counts():
    def realize(function, *, strip_dispatch_markers=False):
        module = allo.customize(function, enable_tensor=False).module
        candidate = generate_apu_v1_vectorization_candidates(module)[-1]
        plan = candidate.plan
        if strip_dispatch_markers:
            transfers = tuple(
                replace(
                    transfer,
                    route=tuple(
                        replace(step, parameters={}) for step in transfer.route
                    ),
                )
                for transfer in plan.transfers
            )
            plan = replace(
                plan,
                name="renamed_physical_plan",
                transfers=transfers,
                metadata={
                    **dict(plan.metadata),
                    "diagnostic_only_marker": "ignored",
                    "problem_shape": {"M": 1, "N": 1, "K": 1},
                    "reduction_tiles": 1,
                },
            )
        realization, error = _realize(candidate.analysis, plan)
        assert realization is not None, error
        return realization

    original = realize(large_spatial_gemv)
    stripped = realize(large_spatial_gemv, strip_dispatch_markers=True)
    renamed = realize(renamed_large_spatial_gemv)
    emission = original.physical_emission

    assert "gemv_spatial_reduction" not in original.plan.metadata
    assert isinstance(emission, SpatialResidentReductionEmission)
    assert original.plan.output_batching.work_tile_counts == (256, 256, 35)
    assert original.output_batches == 3
    assert original.reduction_steps == 2304
    assert emission.matrix_tiles == 547
    assert emission.reduction_tiles == 9
    assert emission.operation_call_count == 4923
    assert original.output_batches * original.reduction_steps == 6912

    expected = (
        ("RESET_16", 547),
        ("MUL_U16", 4923),
        ("RESET_16", 4923),
        ("GROUP_REDUCE_U16", 4923),
        ("ADD_U16", 4923),
    )
    inventories = tuple(
        materialized_apu_v1_operation_inventory(realization)
        for realization in (original, stripped, renamed)
    )
    assert all(
        tuple((operation.opcode, operation.count) for operation in inventory)
        == expected
        for inventory in inventories
    )
    assert stripped.physical_emission == original.physical_emission
    assert (
        "for (uint32_t output_tile = 0; output_tile < 547;" in stripped.device_source()
    )
    original_source = original.device_source()
    assert stripped.device_source() == original_source.replace(
        "spatial_gemv_group_reduction_vector",
        "renamed_physical_plan_vector",
    )

    target = build_apu_v1_target()
    estimates = tuple(
        estimate_apu_v1_realization(realization, target, apu_v1_cost)
        for realization in (original, stripped, renamed)
    )
    assert all(
        estimate.operation_inventory == inventory
        for estimate, inventory in zip(estimates, inventories)
    )
    assert len({estimate.cycles for estimate in estimates}) == 1


def test_estimate_retains_plan_graph_and_bound_cost_fingerprint(plans):
    target = build_apu_v1_target()
    result = estimate_apu_v1_plan(plans[2], target, apu_v1_cost.bind(target))

    assert result.plan is plans[2]
    assert result.cycles == result.estimate.cycles
    assert result.graph.metadata["cost_fingerprint"]
    assert (
        result.estimate.model_fingerprint == result.graph.metadata["cost_fingerprint"]
    )


def test_unknown_operation_fails_closed(plans):
    plan = replace(plans[0], operations=(PlanOperation("UNPRICED", 1),))
    with pytest.raises(ValueError, match="unknown/unpriced opcode"):
        estimate_apu_v1_plan(plan, build_apu_v1_target(), apu_v1_cost)


def test_unpriced_transfer_direction_fails_closed(plans):
    plan = replace(plans[0], transfers=(Transfer("left", "vr_to_vr"),))
    with pytest.raises(ValueError, match="explicit layout route"):
        estimate_apu_v1_plan(plan, build_apu_v1_target(), apu_v1_cost)


def test_cost_binding_and_target_mismatch_fail_closed(plans):
    target = build_apu_v1_target()
    other = build_apu_v1_target()
    bound = apu_v1_cost.bind(target)
    with pytest.raises(ValueError, match="another target"):
        estimate_apu_v1_plan(plans[0], other, bound)
    with pytest.raises(TypeError, match="CostSpec"):
        estimate_apu_v1_plan(plans[0], target, object())


def test_binary_product_uses_priced_and_handle():
    module = allo.customize(binary_micro, enable_tensor=False).module
    plan = generate_apu_v1_vectorization_candidates(module)[0].plan
    result = estimate_apu_v1_plan(plan, build_apu_v1_target(), apu_v1_cost)

    assert result.cycles > 0
    assert any(
        activity.primitive.endswith("/AND_16") for activity in result.graph.activities
    )
