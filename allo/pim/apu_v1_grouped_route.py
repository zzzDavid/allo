# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The APU v1 grouped matcher route of a dataflow region on the shared
candidate contract (release gate G9, ISSUES X-28).

A ``@allo.work`` scope whose body is one FP16 ``acc += x[..] * y[..]``
reduction compiles on APU v1 to the grouped GVML program (one full-VR
``gvml_mul_f16`` and one ``gvml_add_subgrps_f16_grp`` tree over groups of
``group_size`` lanes, four APUCs in SPMD, ``batches`` 32K-lane batches per
APUC).  This module states what that program does in the common candidate
contract (`allo.spmw_candidate.CandidateEffects`) from the work body's
retained facts (`Contraction`: loop labels and trips, access coordinates,
the accumulator initialization, the sinks) and from the placement (the VR
each operand occupies, the group layout), and derives the runtime's packing
and ABI binding from the same facts:

* `apu_v1_grouped_candidate_effects` is the effects deriver the enumerator
  attaches to every grouped candidate and the contract re-derives against
  (`EFFECTS_DERIVERS["apu_v1"]`); `APUv1Ctx.check_realization` compares the
  recorded emission with the declaration at materialization.
* `grouped_device_packing` is what the grouped runtime can execute on the
  board: a dense FP16 GEMM (two rank-2 operands sharing the reduction
  extent, the result covering the whole output) packed one dot product per
  group, rows split over the four APUCs; the batch count the device program
  loops over is derived here, so the emitted loop and the runtime's packing
  agree.  Anything else is ``virtual_only``: the estimate is available, a
  device run is refused with the reason.
* `grouped_runtime_binding` binds the work body's buffers to the caller's
  parameters by retained identity (`trace.source_value_refs`), never by
  spelling, so a region whose work parameters are named differently from its
  arguments executes correctly.

Restrictions kept explicit: the route regroups the reduction (a
``reduction_order="strict"`` FP16 program is refused before enumeration,
`APU_V1_GROUPED_ROUTE_REGROUPS`), it zeroes its accumulator (a carried
destination is refused, `carried_accumulator_message`), and it realizes one
MAC contraction per work item (other operations on this route are refused
with `GROUPED_ROUTE_OPERATIONS`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

APUCS = 4
LANE_BYTES = 2
GROUPED_MODE = "grouped_f16"
LAUNCH_KIND = "gvml_grouped_mac"
# Scratch the emitted body uses besides the operand registers (see
# `APUv1Ctx.iter_vr_aliases`, `emit_l4_to_vr`, `emit_group_reduce_f16`).
SCRATCH_REGISTERS = ("mac_tmp", "reduce_tmp")
L1_STAGING_SLOTS = 4  # GVML_VM_0..2 operand staging, GVML_VM_3 reduce scratch

GROUPED_ROUTE_OPERATIONS = (
    "the APU v1 grouped matcher route realizes one FP16 MAC contraction per "
    "work item (gvml_mul_f16 and a gvml_add_subgrps_f16_grp tree); it declares "
    "no program for the operations {ops} of {function!r}"
)


@dataclass(frozen=True)
class GroupedGeometry:
    """The VR partition of one grouped placement."""

    lanes: int
    group_size: int
    groups_per_vr: int


@dataclass(frozen=True)
class GroupedPacking:
    """What the grouped runtime executes for one work body.

    ``kind`` is ``"gemm"`` (packed one dot product per group, the rows of
    the first operand split over the APUCs, ``batches`` 32K-lane batches per
    APUC) or ``"virtual_only"`` (``reason`` names the restriction; only the
    estimate is available).
    """

    kind: str
    reduction: int
    batches: int
    reason: str | None = None
    rows: int | None = None
    columns: int | None = None
    rows_per_apuc: int | None = None
    lhs: str | None = None
    rhs: str | None = None
    output: str | None = None


def _product(values) -> int:
    total = 1
    for value in values:
        total *= int(value or 0)
    return total


def grouped_geometry(target, placement) -> GroupedGeometry:
    """Group size and group count of a grouped placement from its carried
    layout (the ``group`` / ``lane_in_group`` bases), the same derivation
    the emission walk applies (`derive_layout_properties`)."""
    from ..spmw_autoschedule import derive_layout_properties

    properties = derive_layout_properties(target, placement)
    lanes = int(target.vr0.axes["lane"])
    group_size = int(properties.get("group_size") or 0)
    groups = int(properties.get("groups_per_vr") or 0)
    if group_size <= 0 or groups <= 0 or group_size * groups != lanes:
        raise ValueError(
            "APU v1 grouped placement does not partition one VR into groups: "
            f"group_size {group_size} x groups_per_vr {groups} != {lanes} lanes"
        )
    return GroupedGeometry(lanes, group_size, groups)


def contraction_facts(matches):
    """The `Contraction` facts of a grouped work item's matches, or None
    when any match carries none (a trace assembled outside the shared
    front stages: the legacy handle-only route)."""
    facts = [match.extra.get("semantic_facts") for match in matches]
    if not matches or any(fact is None for fact in facts):
        return None
    return facts


def require_grouped_operations(matches, facts) -> None:
    """The route's one operation: a MAC contraction with facts."""
    ops = sorted({match.target_op_name for match in matches})
    if ops != ["MAC"] or any(not hasattr(fact, "reduction_vars") for fact in facts):
        raise NotImplementedError(
            GROUPED_ROUTE_OPERATIONS.format(ops=ops, function=matches[0].func_name)
        )


def reduction_extent(fact) -> int:
    return _product(fact.trips.get(var) for var in fact.reduction_vars)


def result_instances(fact) -> int:
    return max(1, _product(fact.trips.get(var) for var in fact.result_vars))


def _result_buffers(fact):
    """``(memref, access)`` of the work body's external results: the
    accumulator when it is a buffer, every external sink of a local one."""
    out = []
    if not fact.accumulator.local and fact.accumulator.memref_name is not None:
        out.append((fact.accumulator.memref_name, fact.accumulator))
    for sink in fact.external_sinks:
        if sink.memref_name is not None:
            out.append((sink.memref_name, sink))
    return out


def grouped_device_packing(fact, geometry: GroupedGeometry) -> GroupedPacking:
    """The grouped runtime's packing of one work body, from its facts.

    The runtime (`spmw_apu_v1._apu_v1_prepare_io`) packs a dense FP16 GEMM
    ``x[M, K] * y[K, N]`` one dot product per ``group_size``-lane group,
    the rows of ``x`` split over the four APUCs, ``ceil(ceil(M / 4) * N /
    groups_per_vr)`` batches per APUC, and gathers the group heads into the
    ``[M, N]`` result.  That is the whole result of the kernel group, so the
    sink must be the ``[M, N]`` buffer.  Everything else (a GEMV, a
    non-FP16 operand, a partial result) has no device packing.
    """
    reduction = reduction_extent(fact)
    inputs = tuple(fact.inputs)
    results = _result_buffers(fact)
    if len(inputs) != 2 or not results:
        return GroupedPacking(
            "virtual_only",
            reduction,
            1,
            reason="the grouped runtime packs a two-operand contraction with an "
            f"external result; got {len(inputs)} operand(s) and "
            f"{len(results)} result(s)",
        )
    x, y = inputs
    (output, result) = results[0]
    x_shape, y_shape = tuple(x.shape or ()), tuple(y.shape or ())
    result_shape = tuple(result.shape or ())
    elements = {getattr(access.element, "text", None) for access in (x, y, result)}
    if (
        len(x_shape) == 2
        and len(y_shape) == 2
        and x_shape[1] == y_shape[0] == reduction
        and result_shape == (x_shape[0], y_shape[1])
        and elements == {"f16"}
    ):
        rows, columns = int(x_shape[0]), int(y_shape[1])
        rows_per_apuc = math.ceil(rows / APUCS)
        batches = max(1, math.ceil(rows_per_apuc * columns / geometry.groups_per_vr))
        return GroupedPacking(
            "gemm",
            reduction,
            batches,
            rows=rows,
            columns=columns,
            rows_per_apuc=rows_per_apuc,
            lhs=x.memref_name,
            rhs=y.memref_name,
            output=output,
        )
    return GroupedPacking(
        "virtual_only",
        reduction,
        1,
        reason=(
            "the grouped runtime packs a dense FP16 GEMM (two rank-2 operands "
            "sharing the reduction extent, the result the whole [rows, columns] "
            f"buffer) one dot product per group; got {x.memref_name}{list(x_shape)}"
            f" x {y.memref_name}{list(y_shape)} -> {output}{list(result_shape)} "
            f"with element types {sorted(str(e) for e in elements)}"
        ),
    )


def _storage_name(handle) -> str | None:
    name = getattr(handle, "name", None)
    return str(name) if name else None


def apu_v1_grouped_candidate_effects(target, matches, placement):
    """`CandidateEffects` of one grouped APU v1 placement for one work item,
    derived from the shared facts and the placement; None for a vector
    plan or program record (they declare their own effects) and for a
    trace without facts.

    Coverage: the contraction's operands and external results.  Value
    layouts: each operand in the VR the placement binds it to, packed
    ``group_lanes:<group_size>`` (the reduction's elements in the first
    lanes of one group, the rest zero); the result in the accumulator's VR
    at the group heads (``group_head:<group_size>``); axes are the source
    loop labels the accesses range over.  Transfers: every operand packed
    by the host into L4 and staged through L1 into its VR; the result from
    the accumulator VR through L1 to L4 and gathered by the host.  One
    launch of kind `LAUNCH_KIND` advancing every match of the work item over
    its result instances and the whole reduction with the source's
    accumulator initialization; ``covers`` records the mode, the group
    geometry, the APUC count, the batch count the device program loops
    over and the runtime packing kind.  Allocations: the two scratch VRs
    and the L1 staging slots.
    """
    from ..spmw_candidate import (
        CandidateCoverage,
        CandidateEffects,
        CandidateLaunch,
        CandidateTransfer,
        CandidateValueLayout,
        value_axes,
    )

    extra = placement.extra or {}
    if extra.get("payload") is not None:
        return None
    facts = contraction_facts(matches)
    if facts is None:
        return None
    require_grouped_operations(matches, facts)
    geometry = grouped_geometry(target, placement)
    labels: dict = {}
    for fact in facts:
        labels.update(fact.loop_labels)
    first = facts[0]
    reduction = reduction_extent(first)
    instances = result_instances(first)
    for fact in facts[1:]:
        if reduction_extent(fact) != reduction or result_instances(fact) != instances:
            raise ValueError(
                "APU v1 grouped route: the matches of one work item differ in "
                "reduction extent or result instances"
            )
    if reduction > geometry.group_size:
        raise ValueError(
            f"APU v1 reduction extent {reduction} exceeds the "
            f"{geometry.group_size}-lane group of the placement"
        )
    init = first.accumulator_init
    packing = grouped_device_packing(first, geometry)
    batches = max(1, int(extra.get("n_out_tiles", 1) or 1))
    group = geometry.group_size
    inputs: list = []
    layouts: dict = {}
    transfers: dict = {}
    for fact in facts:
        for access in fact.inputs:
            memref = access.memref_name
            if memref is None or access.local or memref in inputs:
                continue
            storage = _storage_name(placement.placements.get(memref))
            if storage is None:
                raise ValueError(
                    f"APU v1 grouped placement binds operand {memref!r} to no register"
                )
            inputs.append(memref)
            layouts[memref] = CandidateValueLayout(
                memref, value_axes(access, labels), storage, f"group_lanes:{group}"
            )
            transfers[memref] = CandidateTransfer(
                memref, "in", ("host_pack", "l4", "vmrs", storage)
            )
    accumulator_storage = _storage_name(
        placement.placements.get(first.accumulator.memref_name)
    )
    if accumulator_storage is None:
        raise ValueError(
            "APU v1 grouped placement binds the accumulator "
            f"{first.accumulator.memref_name!r} to no register"
        )
    outputs: list = []
    for fact in facts:
        for memref, access in _result_buffers(fact):
            if memref in outputs:
                continue
            outputs.append(memref)
            carried = init == "carried"
            if carried and memref not in inputs:
                inputs.append(memref)
            layouts[memref] = CandidateValueLayout(
                memref,
                value_axes(access, labels),
                accumulator_storage,
                f"group_head:{group}",
            )
            transfers[memref] = CandidateTransfer(
                memref,
                "inout" if carried else "out",
                (
                    (
                        "host_pack",
                        "l4",
                        "vmrs",
                        accumulator_storage,
                        "l4",
                        "host_gather",
                    )
                    if carried
                    else (accumulator_storage, "vmrs", "l4", "host_gather")
                ),
            )
    launch = CandidateLaunch(
        index=0,
        kind=LAUNCH_KIND,
        operations=tuple(range(len(matches))),
        instances=(0, instances),
        reduction=(0, reduction),
        accumulator=init,
        covers=(
            GROUPED_MODE,
            geometry.group_size,
            geometry.groups_per_vr,
            APUCS,
            batches,
            packing.kind,
        ),
    )
    lane_bytes = geometry.lanes * LANE_BYTES
    allocations = tuple((name, lane_bytes, "vr") for name in SCRATCH_REGISTERS) + (
        ("l1_staging", L1_STAGING_SLOTS * lane_bytes, "vmrs"),
    )
    return CandidateEffects(
        coverage=CandidateCoverage(
            matches[0].func_name, ("MAC",), tuple(inputs), tuple(outputs)
        ),
        value_layouts=tuple(layouts.values()),
        transfers=tuple(transfers.values()),
        launches=(launch,),
        abi_inputs=tuple(inputs),
        abi_outputs=tuple(outputs),
        device_scope=(
            f"{APUCS} APUCs in SPMD, one {geometry.group_size}-lane group per "
            f"output dot product ({geometry.groups_per_vr} per VR), {batches} "
            f"32K-lane batch(es) per APUC; device packing: {packing.kind}"
        ),
        allocations=allocations,
    )


def declared_launch(effects):
    """``(group_size, groups_per_vr, apucs, batches, packing kind)`` of a
    grouped declaration's launch, or None for another kind of launch."""
    if effects is None or len(effects.launches) != 1:
        return None
    launch = effects.launches[0]
    if launch.kind != LAUNCH_KIND or len(launch.covers) != 6:
        return None
    _mode, group_size, groups, apucs, batches, kind = launch.covers
    return int(group_size), int(groups), int(apucs), int(batches), str(kind)


def grouped_runtime_binding(trace, parameter_names) -> dict:
    """``{work-body memref: caller parameter}`` by retained identity: the
    trace's ``source_value_refs`` (parameter ordinal -> value) and the
    operands' and sinks' ``value_ref``.  Never by spelling."""
    refs = getattr(trace, "source_value_refs", None) or {}
    names = tuple(parameter_names or ())
    by_ref = {}
    for ordinal, ref in refs.items():
        if 0 <= int(ordinal) < len(names):
            by_ref.setdefault(ref, names[int(ordinal)])
    out: dict = {}
    for match in getattr(trace, "matches", ()):
        for operand in match.operands:
            ref = getattr(operand, "value_ref", None)
            if operand.memref_name and ref in by_ref:
                out.setdefault(operand.memref_name, by_ref[ref])
        sinks = match.extra.get("output_sinks", ())
        sink_refs = match.extra.get("output_sink_refs", ())
        for (memref, _text), (ref, local) in zip(sinks, sink_refs):
            if not local and ref in by_ref:
                out.setdefault(memref, by_ref[ref])
    return out


def _derive_apu_v1_effects(view, matches, placement):
    """Late-binding entry of the effects registry (`EFFECTS_DERIVERS`): the
    grouped route's derivation; None for the vector plans and program
    records, which declare their own effects."""
    return apu_v1_grouped_candidate_effects(view, matches, placement)


def _register_effects_deriver():
    from ..spmw_candidate import EFFECTS_DERIVERS

    EFFECTS_DERIVERS["apu_v1"] = _derive_apu_v1_effects


_register_effects_deriver()


__all__ = [
    "APUCS",
    "GROUPED_MODE",
    "GROUPED_ROUTE_OPERATIONS",
    "GroupedGeometry",
    "GroupedPacking",
    "LAUNCH_KIND",
    "apu_v1_grouped_candidate_effects",
    "contraction_facts",
    "declared_launch",
    "grouped_device_packing",
    "grouped_geometry",
    "grouped_runtime_binding",
    "reduction_extent",
    "require_grouped_operations",
    "result_instances",
]
