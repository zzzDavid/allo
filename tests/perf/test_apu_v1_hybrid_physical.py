# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent-L4 lowering and aggregate costs for hybrid APU v1 programs."""

from dataclasses import replace
from functools import lru_cache

import allo
import pytest

from allo.backend.c import emit_c_from_mlir
from allo.pim.apu_v1_hybrid import (
    APUv1RegionExecutable,
    APUv1PhysicalHybridManifest,
    build_apu_v1_hybrid_execution_graph,
    discover_apu_v1_hybrid_manifest,
    lower_apu_v1_hybrid_manifest,
    partition_apu_v1_region,
)
from allo.pim.costs import apu_v1_cost
from allo.pim.targets import build_apu_v1_target

from test_apu_v1_hybrid_manifest import _program


@lru_cache(maxsize=None)
def _logical(case, *, cost=None):
    program = _program(case)
    phase = program.phases[0]
    schedule = allo.customize(
        phase.kernel,
        enable_tensor=False,
        instantiate=list(phase.instantiate),
    )
    artifact = emit_c_from_mlir(schedule.module, schedule.top_func_name)
    target = build_apu_v1_target()
    return (
        discover_apu_v1_hybrid_manifest(
            phase,
            schedule,
            artifact,
            target,
            cost=apu_v1_cost.bind(target) if cost else None,
            program_name=program.name,
        ),
        target,
    )


def _executables(manifest, *, omit=None):
    return {
        region.id: APUv1RegionExecutable(
            region.id,
            functional=lambda _buffers, _phase: None,
            partition=(
                partition_apu_v1_region(region) if region.kind == "vector" else None
            ),
        )
        for logical_phase in manifest.phases
        for region in logical_phase.regions
        if region.id != omit
    }


def _partitions(manifest):
    return {
        region.id: partition_apu_v1_region(region)
        for phase in manifest.phases
        for region in phase.regions
        if region.kind == "vector"
    }


@pytest.mark.parametrize(
    "case,kinds",
    [
        (
            "gemm",
            ["convert", "convert", "vector", "convert", "barrier", "arc_scalar"],
        ),
        (
            "2mm",
            [
                "convert",
                "convert",
                "vector",
                "barrier",
                "convert",
                "vector",
                "convert",
                "barrier",
                "arc_scalar",
            ],
        ),
        (
            "3mm",
            [
                "convert",
                "convert",
                "vector",
                "convert",
                "convert",
                "vector",
                "barrier",
                "vector",
                "convert",
            ],
        ),
    ],
)
def test_canonical_hybrid_lowering_uses_persistent_l4_and_explicit_phases(case, kinds):
    logical, _target = _logical(case)
    physical = lower_apu_v1_hybrid_manifest(logical, _executables(logical))

    assert [phase.kind for phase in physical.phases] == kinds
    assert physical.manifest()["host_intermediate_round_trips"] == 0
    assert physical.persistent_l4_bytes > 0
    assert all(
        allocation.intermediate
        for allocation in physical.allocations
        if "__f16__" in allocation.name
    )
    assert not any(
        allocation.host_input or allocation.host_output
        for allocation in physical.allocations
        if allocation.intermediate
    )
    assert all(
        phase.apucs == (0, 1, 2, 3)
        for phase in physical.phases
        if phase.kind in {"vector", "convert"}
    )
    assert all(
        phase.apucs == (0,) for phase in physical.phases if phase.kind == "arc_scalar"
    )


def test_two_mm_physical_dependencies_follow_conversions_and_barriers():
    logical, _target = _logical("2mm")
    physical = lower_apu_v1_hybrid_manifest(logical, _executables(logical))
    phases = {phase.name: phase for phase in physical.phases}

    assert phases["region0"].dependencies == ("convert0", "convert1")
    assert phases["barrier0"].dependencies == ("region0",)
    assert "barrier0" in phases["convert2"].dependencies
    assert {"barrier0", "convert2"} <= set(phases["region1"].dependencies)
    assert phases["convert3"].dependencies == ("region1",)
    assert phases["barrier1"].dependencies == ("convert3",)
    assert "barrier1" in phases["region2"].dependencies

    # The f16 vector intermediate remains in L4 through mm1 -> mm2.  Only the
    # vector-to-scalar edge converts out_ABC back to canonical f32.
    region0_output = dict(phases["region0"].bindings)["out_AB"]
    region1_input = dict(phases["region1"].bindings)["out_AB"]
    assert region0_output == region1_input
    assert region0_output.endswith("__f16__resident")
    assert phases["convert3"].writes == ("out_ABC",)


def test_missing_region_executable_fails_closed():
    logical, _target = _logical("3mm")
    with pytest.raises(ValueError, match="no executable artifact for region 'region1'"):
        lower_apu_v1_hybrid_manifest(logical, _executables(logical, omit="region1"))


def test_physical_manifest_rejects_broken_dependencies_and_output_liveness():
    logical, _target = _logical("2mm")
    physical = lower_apu_v1_hybrid_manifest(logical, _executables(logical))

    broken_dependency = list(physical.phases)
    broken_dependency[2] = replace(broken_dependency[2], dependencies=("not_emitted",))
    with pytest.raises(ValueError, match="unsatisfied dependencies"):
        APUv1PhysicalHybridManifest(
            physical.name,
            logical,
            physical.allocations,
            tuple(broken_dependency),
        )

    no_output_write = tuple(
        replace(phase, writes=()) if phase.name == "region2" else phase
        for phase in physical.phases
    )
    with pytest.raises(ValueError, match="host outputs uninitialized"):
        APUv1PhysicalHybridManifest(
            physical.name,
            logical,
            physical.allocations,
            no_output_write,
        )


def test_vector_region_requires_declared_physical_partition():
    logical, target = _logical("2mm")
    executables = dict(_executables(logical))
    executables["region0"] = replace(executables["region0"], partition=None)
    with pytest.raises(ValueError, match="explicit APUC partition"):
        lower_apu_v1_hybrid_manifest(logical, executables)

    bound = apu_v1_cost.bind(target)
    with pytest.raises(ValueError, match="requires a physical partition"):
        build_apu_v1_hybrid_execution_graph(logical, target, bound, partitions={})


@pytest.mark.parametrize("case", ["gemm", "2mm", "3mm"])
def test_selected_vector_plans_have_serialized_partition_contracts(case):
    logical, _target = _logical(case)
    partitions = _partitions(logical)
    physical = lower_apu_v1_hybrid_manifest(logical, _executables(logical))
    vector_phases = [phase for phase in physical.phases if phase.kind == "vector"]
    assert vector_phases
    for phase in vector_phases:
        partition = phase.executable.partition
        assert partition is not None
        assert partition.region_id == phase.name
        assert partition.estimate_scope == "full_domain"
        assert partition.apucs == phase.apucs == (0, 1, 2, 3)
        assert sum(shard.extent for shard in partition.shards) == partition.extent
        assert tuple((shard.start, shard.stop) for shard in partition.shards) == tuple(
            (
                partition.extent * apuc // 4,
                partition.extent * (apuc + 1) // 4,
            )
            for apuc in range(4)
        )
        assert phase.manifest()["partition"] == partition.manifest()
        logical_region = next(
            region
            for logical_phase in logical.phases
            for region in logical_phase.regions
            if region.id == phase.name
        )
        output_value = next(
            operand.value
            for operand in logical_region.operands
            if operand.role == "output"
        )
        assert output_value in partition.sharded_values
        assert output_value not in partition.replicated_values
        assert logical_region.physical_plan_verified is True
        assert logical_region.manifest()["physical_plan_verified"] is True
        lhs_value = next(
            operand.value
            for operand in logical_region.operands
            if operand.role == "lhs"
        )
        rhs_value = next(
            operand.value
            for operand in logical_region.operands
            if operand.role == "rhs"
        )
        assert lhs_value in partition.sharded_values
        assert rhs_value in partition.replicated_values


@pytest.mark.parametrize("case", ["gemm", "2mm", "3mm"])
def test_program_estimate_composes_vector_conversion_and_scalar_regions(case):
    logical, target = _logical(case)
    bound = apu_v1_cost.bind(target)
    graph = build_apu_v1_hybrid_execution_graph(
        logical, target, bound, partitions=_partitions(logical)
    )
    estimate = bound.evaluate(graph)

    assert graph.metadata["hybrid"] is True
    assert estimate.cycles > 0
    assert any(
        activity.metadata.get("kind") == "precision_conversion"
        for activity in graph.activities
    )
    assert any(
        activity.metadata.get("hybrid_region", "").startswith("hybrid:region")
        for activity in graph.activities
    )
    expected_scalar = case in {"gemm", "2mm"}
    assert (
        any(
            activity.metadata.get("kind") == "arc_scalar"
            for activity in graph.activities
        )
        is expected_scalar
    )
    assert any(activity.label == "hybrid_barrier" for activity in graph.activities)
    assert estimate.model_fingerprint == graph.metadata["cost_fingerprint"]


def test_vector_cost_graph_is_sharded_over_four_concrete_apucs():
    logical, target = _logical("3mm")
    bound = apu_v1_cost.bind(target)
    partitions = _partitions(logical)
    graph = build_apu_v1_hybrid_execution_graph(
        logical, target, bound, partitions=partitions
    )
    estimate = bound.evaluate(graph)

    assert graph.metadata["region_partitions"] == {
        region_id: partition.manifest() for region_id, partition in partitions.items()
    }
    per_apuc = set()
    for activity in graph.activities:
        region = activity.metadata.get("hybrid_region", "")
        if not region.startswith("hybrid:region0:apuc"):
            continue
        for use in activity.occupancy:
            for axis, coordinate in use.handle.coordinates:
                if axis == "apuc":
                    per_apuc.add(coordinate)
    assert per_apuc == {0, 1, 2, 3}

    # mm1 and mm2 have no data edge in 3mm. Their graph roots depend only on
    # their own input conversions; four-APUC resource occupancy, rather than a
    # fabricated dependency, determines launch serialization.
    region1_roots = [
        activity
        for activity in graph.activities
        if activity.metadata.get("hybrid_region", "").startswith("hybrid:region1:apuc")
        and not any("hybrid:region1:apuc" in dep for dep in activity.depends_on)
    ]
    assert region1_roots
    assert not any(
        "hybrid:region0" in dependency
        for activity in region1_roots
        for dependency in activity.depends_on
    )
    assert estimate.cycles > 0


def test_uneven_partition_scales_repeats_but_preserves_fixed_calls():
    logical, target = _logical("3mm")
    partitions = _partitions(logical)
    partition = partitions["region1"]  # mm2 has 50 output rows.
    assert [shard.extent for shard in partition.shards] == [12, 13, 12, 13]

    bound = apu_v1_cost.bind(target)
    graph = build_apu_v1_hybrid_execution_graph(
        logical, target, bound, partitions=partitions
    )
    by_apuc = {apuc: [] for apuc in range(4)}
    for activity in graph.activities:
        region = activity.metadata.get("hybrid_region", "")
        for apuc in range(4):
            if region.startswith(f"hybrid:region1:apuc{apuc}"):
                by_apuc[apuc].append(activity)

    repeated_mul = {
        apuc: next(item for item in items if item.label == "gvml_mul_f16")
        for apuc, items in by_apuc.items()
    }
    assert repeated_mul[1].latency_cycles > repeated_mul[0].latency_cycles
    assert repeated_mul[3].latency_cycles == repeated_mul[1].latency_cycles

    fixed_load = {
        apuc: next(
            item
            for item in items
            if item.label == "gvml_load_16" and item.metadata["cost_count"] == 1
        )
        for apuc, items in by_apuc.items()
    }
    assert {item.latency_cycles for item in fixed_load.values()} == {29}


def test_public_compile_default_device_installs_physical_hybrid_runner():
    """backend=None selects the physical runner, never the scalar baseline."""

    target = build_apu_v1_target()
    compiled = allo.compile(_program("gemm"), target, apu_v1_cost)
    assert compiled.compiled.hybrid_callable.backend == "device"
    assert compiled.compiled.hybrid_callable.device_runner is not None
    assert compiled.compiled.hybrid_callable.device_runner.__name__ == (
        "run_apu_v1_hybrid"
    )
