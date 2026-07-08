# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Executable-cost lowering for immutable APU v1 vector plans."""

from dataclasses import replace

import allo
import pytest
from allo.ir.types import float16, int32

from allo.pim.apu_v1_layout import PlanOperation, Transfer
from allo.pim.apu_v1_vector_cost import (
    build_apu_v1_plan_graph,
    estimate_apu_v1_plan,
    rank_apu_v1_plans,
)
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


def test_full_micro_shape_has_four_positive_costs_and_broadcast_plan_wins(plans):
    target = build_apu_v1_target()
    ranked = rank_apu_v1_plans(plans, target, apu_v1_cost)

    cycles = [item.cycles for item in ranked]
    assert all(cycle > 0 for cycle in cycles)
    assert len(set(cycles)) == 4
    assert [item.plan.name for item in ranked] == [
        "temporal_dma_coalescing_broadcast_friendly",
        "temporal_dma_coalescing",
        "baseline_spatial_reduction",
        "temporal_svp",
    ]


def test_transfer_traffic_is_derived_from_route_layout_relations(plans):
    target = build_apu_v1_target()
    result = estimate_apu_v1_plan(plans[-1], target, apu_v1_cost)
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
