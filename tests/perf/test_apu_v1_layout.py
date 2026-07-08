# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for immutable APU v1 layout and plan values."""

import dataclasses
import json

import pytest

from allo.pim.apu_v1_layout import (
    APUReduction,
    APUV1Plan,
    AffineTiledLayout,
    IterationLayout,
    PlanOperation,
    ReductionStrategy,
    TemporalAxis,
    TemporalStrategy,
    Transfer,
    TransferLayout,
    TransferRouteStep,
    ReuseWindow,
    UnsupportedLayoutError,
    VRAllocation,
    Validity,
    ValueLayout,
    derive_layout_metrics,
)
from allo.spmw_linear_layout import LinearLayout


def _group_layout():
    return LinearLayout(
        bases={
            "group": [((8 << bit),) for bit in range(2)],
            "lane": [((1 << bit),) for bit in range(3)],
        },
        out_dims=("vr_lane",),
        out_sizes=(32,),
    )


def test_linear_layout_metrics_derive_image_fiber_occupancy_and_contiguity():
    layout = LinearLayout(
        bases={
            "element": [((1 << bit),) for bit in range(2)],
            "replica": [(0,)],
        },
        out_dims=("vr_lane",),
        out_sizes=(8,),
    )

    metrics = derive_layout_metrics(layout)

    assert metrics.domain_size == 8
    assert metrics.image_size == 4
    assert metrics.fiber_size == 2
    assert metrics.replication_factor == 2
    assert metrics.occupancy == 0.5
    assert metrics.unique_elements == 4
    assert metrics.address_span == 4
    assert metrics.low_bit_contiguous is True


def test_non_power_of_two_affine_tile_has_explicit_padding_and_coordinates():
    layout = AffineTiledLayout.packed({"M": 3, "N": 5}, max_extent=64)

    assert layout.padded_extents == {"M": 4, "N": 8}
    assert layout.apply(M=2, N=4) == (20,)
    assert layout.coordinate(M=2, N=4) == {"vr_lane": 20}
    assert layout.lane_for(M=2, N=4) == 20
    with pytest.raises(ValueError, match="logical extent"):
        layout.apply(M=3, N=0)

    metrics = derive_layout_metrics(layout)
    assert metrics.domain_size == 15
    assert metrics.padded_domain_size == 32
    assert metrics.image_size == 15
    assert metrics.padding_elements == 17
    assert metrics.occupancy == 15 / 32
    assert metrics.address_span == 21
    assert metrics.low_bit_contiguous is False


def test_large_f2_metrics_use_rank_without_enumerating_domain():
    layout = LinearLayout(
        bases={
            "M": [((1 << bit),) for bit in range(10)],
            "N": [(1 << (10 + bit),) for bit in range(5)] + [(0,)] * 5,
            "K": [(0,)] * 6,
        },
        out_dims=("vr_lane",),
        out_sizes=(32768,),
    )

    metrics = derive_layout_metrics(layout)

    assert metrics.domain_size == 1024 * 1024 * 64
    assert metrics.image_size == 32768
    assert metrics.fiber_size == 2048
    assert metrics.occupancy == 1.0
    assert metrics.address_span is None
    assert metrics.low_bit_contiguous is None


def test_value_layout_inherits_iteration_mapping_and_tracks_replica_axes():
    iteration = IterationLayout(("group", "lane"), _group_layout())
    value = ValueLayout(
        "weights",
        ("lane",),
        replica_axes=("group",),
        validity=Validity(("lane",), (5,), (8,), padding_value=0),
    )

    metrics = value.metrics(iteration)

    assert metrics.logical_elements == 5
    assert metrics.physical_elements == 32
    assert metrics.explicit_replication_factor == 4
    assert metrics.padding_elements == 3
    assert value.lane_for(iteration, group=3, lane=7) == 31


def test_plan_retains_temporal_reduction_transfer_operations_and_vr_assignment():
    iteration = IterationLayout(
        ("group", "lane"),
        _group_layout(),
        temporal_axes=(TemporalAxis("group", 4, carried_values=("acc",)),),
    )
    values = (
        ValueLayout("x", ("lane",), replica_axes=("group",)),
        ValueLayout("acc", ("group",)),
    )
    plan = APUV1Plan(
        "gemv_plan",
        iteration,
        values,
        (
            Transfer("x", "l4_to_vr", coalesced=True, broadcast=True),
            Transfer("acc", "vr_to_l4", coalesced=True),
        ),
        TemporalStrategy((TemporalAxis("group", 4),), barrier="batch"),
        APUReduction("lane", "spatial", group_size=8),
        operations=(PlanOperation("mul_f16", 32, loop_roles=("lane",)),),
        vr_allocations=(
            VRAllocation("input_vr", "GVML_VR16_0", value="x", live_range=(0, 2)),
            VRAllocation("acc_vr", 1, value="acc", live_range=(1, 3)),
        ),
        metadata={"dtype": "f16", "logical_extent": 32},
    )

    assert plan.reduction_strategy.kind == "group_tree"
    assert plan.temporal_extent == 4
    assert plan.value_metrics("x").explicit_replication_factor == 4
    assert plan.vr_allocations[0].concrete_name == "GVML_VR16_0"
    assert plan.manifest()["metadata"]["dtype"] == "f16"
    json.dumps(plan.manifest())
    with pytest.raises(TypeError):
        plan.metadata["dtype"] = "i16"
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.name = "changed"


def test_transfer_route_derives_expansion_calls_and_cross_batch_residency():
    compact = TransferLayout(
        "l3",
        ("m", "k"),
        AffineTiledLayout.packed({"m": 32, "k": 64}, max_extent=None),
        role="lookup_tables",
    )
    expanded_layout = LinearLayout(
        {
            "m": [((1024 << bit),) for bit in range(5)],
            "n": [((1 << bit),) for bit in range(10)],
            "k": [(0,)] * 6,
        },
        ("vr_lane",),
        (32768,),
    )
    expanded = TransferLayout(
        "compute_vr",
        ("m", "k"),
        expanded_layout,
        replica_axes=("n",),
        role="expanded_compute",
    )
    step = TransferRouteStep(
        "lookup",
        compact,
        expanded,
        temporal_axis="k",
        executed_at=(
            ReuseWindow("m", 32, 1024),
            ReuseWindow("k", 1, 64),
        ),
        resident_across=(ReuseWindow("m", 32, 1024),),
        parameters={"table_size": 32},
    )
    transfer = Transfer("x", "in", route=(step,))
    metrics = step.metrics()

    assert transfer.source_layout is compact
    assert transfer.transit_layout is None
    assert transfer.destination_layout is expanded
    assert transfer.broadcast and not transfer.coalesced
    assert step.replication_axes_added == ("n",)
    assert metrics.source_elements_per_call == 32
    assert metrics.destination_elements_per_call == 32768
    assert metrics.call_count == 32 * 64
    assert metrics.expansion_factor == 1024
    assert metrics.resident_reuse_factor == 32
    with pytest.raises(TypeError):
        step.parameters["table_size"] = 64


def test_reduction_aliases_are_canonical():
    assert ReductionStrategy("K", "spatial").kind == "group_tree"
    assert ReductionStrategy("K", "temporal").kind == "temporal_accumulate"


def test_vr_allocation_rejects_overlapping_concrete_registers():
    iteration = IterationLayout(("group", "lane"), _group_layout())
    values = (ValueLayout("x", ("lane",), replica_axes=("group",)),)
    with pytest.raises(ValueError, match="overlapping live ranges"):
        APUV1Plan(
            "bad_alloc",
            iteration,
            values,
            (Transfer("x", "load"),),
            "none",
            None,
            vr_allocations=(
                VRAllocation("a", 0, live_range=(0, 2)),
                VRAllocation("b", 0, live_range=(1, 3)),
            ),
        )


def test_fail_closed_for_unsupported_or_too_large_masked_layouts():
    with pytest.raises(UnsupportedLayoutError, match="unsupported value layout"):
        ValueLayout("x", ("i",), layout=object())
    large_masked = AffineTiledLayout.packed({"M": 2047, "N": 2047}, max_extent=None)
    metrics = derive_layout_metrics(large_masked)
    assert metrics.domain_size == 2047 * 2047
    assert metrics.padded_domain_size == 2048 * 2048
    assert metrics.image_size == metrics.domain_size
    assert metrics.padding_elements == metrics.padded_domain_size - metrics.domain_size
    with pytest.raises(ValueError, match="broadcast"):
        Transfer("x", "vr_to_l4", broadcast=True)


def test_wrappers_snapshot_mutable_linear_layout_input():
    original = _group_layout()
    iteration = IterationLayout(("group", "lane"), original)
    original.bases["lane"][0] = (0,)

    assert iteration.lane_for(group=0, lane=1) == 1
    with pytest.raises(TypeError):
        iteration.layout.bases["lane"] = ((0,),)
