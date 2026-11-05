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
* `prove_dense_gemm_template` proves, from the retained access relations
  of every work item of the kernel group, that the region is the dense FP16
  GEMM the grouped runtime packs (operands read at ``[r, k]`` and ``[k, c]``
  with the reduction in lane order, the result written at ``[r, c]``, each
  work item one rectangular block, the blocks a disjoint cover of the whole
  output, a zero accumulator), or names the first condition the source
  fails; `grouped_device_packing` turns the proven template into the
  packing (one dot product per group, rows split over the four APUCs) and
  the batch count the device program loops over, so the emitted loop and
  the runtime's packing agree.  Anything else is ``virtual_only``: the
  estimate is available, an executable backend is refused before
  scheduling (`require_grouped_device_packing`) with the reason.
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
class WorkRectangle:
    """The block of the output one work item writes: rows ``[row0, row0 +
    rows)`` and columns ``[col0, col0 + columns)``."""

    work_id: tuple
    row0: int
    rows: int
    col0: int
    columns: int

    @property
    def area(self) -> int:
        return self.rows * self.columns


@dataclass(frozen=True)
class GemmTemplate:
    """The dense FP16 GEMM the grouped runtime executes, proven from the
    retained access relations of every work item of the kernel group:
    ``output[r, c] = sum_k lhs[r, k] * rhs[k, c]`` with the reduction in lane
    order, a zero accumulator, and the work items' output blocks a disjoint
    cover of the whole ``[rows, columns]`` buffer."""

    lhs: str
    rhs: str
    output: str
    rows: int
    columns: int
    reduction: int
    blocks: tuple


@dataclass(frozen=True)
class GroupedPacking:
    """What the grouped runtime executes for one kernel group.

    ``kind`` is ``"gemm"`` (the proven `GemmTemplate`, packed one dot
    product per group, the rows split over the APUCs, ``batches`` 32K-lane
    batches per APUC) or ``"virtual_only"`` (``reason`` names the first
    condition of the template the source fails; only the estimate is
    available and an executable backend is refused before scheduling).
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
    template: GemmTemplate | None = None


def _product(values) -> int:
    total = 1
    for value in values:
        total *= int(value or 0)
    return total


def grouped_geometry(target, placement) -> GroupedGeometry:
    """Group size and group count of a grouped placement from its carried
    layout (the ``group`` / ``lane_in_group`` bases), the same derivation
    the emission walk applies (`derive_layout_properties`).  Cardinalities
    only; `grouped_map_problem` states whether the map is the one the
    executable realizes."""
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


def grouped_map_problem(layout, geometry: GroupedGeometry) -> str | None:
    """Why the placement's selected map is not the map the grouped
    executable realizes, or None when it is (ISSUES X-31).

    The GVML group tree (``gvml_add_subgrps_f16_grp`` with ``GVML_P2_<group
    size>``) and the host packing address contiguous groups: physical lane
    = group x group_size + lane_in_group, the reduction's elements in the
    first lanes of a group.  A carried `LinearLayout` must therefore send
    ``lane_in_group`` bit b to physical lane bit b and ``group`` bit b to
    physical lane bit b + log2(group_size), with no other output dimension
    moved by those bases.  An interleaved or non-injective map (the same
    cardinalities, other physical lanes) is not realized by the emitted
    program and is refused per candidate.  A placement without a linear
    layout (a hand-built handle-only placement) is taken as the contiguous
    map (its group size comes from ``extra``)."""
    from ..spmw_linear_layout import LinearLayout

    if not isinstance(layout, LinearLayout):
        return None
    if "vr_lane" not in layout.out_dims:
        return "the layout has no vr_lane output dimension"
    column = layout.out_dims.index("vr_lane")
    lane_bits = (geometry.group_size - 1).bit_length()
    expected = {
        "lane_in_group": [1 << bit for bit in range(lane_bits)],
        "group": [
            geometry.group_size << bit
            for bit in range((geometry.groups_per_vr - 1).bit_length())
        ],
    }
    for name, wanted in expected.items():
        vectors = list(layout.bases.get(name, ()))
        physical = [int(vector[column]) for vector in vectors]
        if physical != wanted:
            return (
                f"the executable addresses contiguous groups (physical lane = group x "
                f"{geometry.group_size} + lane_in_group); the placement's map sends the "
                f"{name} bits to physical lanes {physical}, not {wanted}"
            )
        for vector in vectors:
            if any(int(v) for index, v in enumerate(vector) if index != column):
                return (
                    f"the placement's {name} bases move output dimensions other than "
                    f"vr_lane ({vector}); the executable's groups are lanes of one VR"
                )
    return None


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


def _affine(coord):
    """``(coefficients by loop variable, constant)`` of one retained
    coordinate (`spmw_match_engine.affine_form`), None when it is not
    affine in loop variables."""
    try:
        terms, constant = coord
        coefficients = {}
        for var, coefficient in terms:
            coefficients[str(var)] = coefficients.get(str(var), 0) + int(coefficient)
        return {v: c for v, c in coefficients.items() if c != 0}, int(constant)
    except (TypeError, ValueError):
        return None


def _affine_text(coord, labels) -> str:
    form = _affine(coord)
    if form is None:
        return "<non-affine>"
    coefficients, constant = form
    parts = []
    for var, coefficient in coefficients.items():
        label = labels.get(var, var)
        parts.append(label if coefficient == 1 else f"{coefficient} * {label}")
    if constant or not parts:
        parts.append(str(constant))
    return " + ".join(parts).replace("+ -", "- ")


def group_contractions(matches):
    """``[(work id, Contraction)]`` of every work item of the matches'
    kernel group, as the ranking stamped them before enumeration
    (``extra["spmw_group_contractions"]``, `_stamp_group_replication`);
    None when the stamp or any fact is missing."""
    if not matches:
        return None
    group = matches[0].extra.get("spmw_group_contractions")
    if not group or any(fact is None for _work_id, fact in group):
        return None
    return list(group)


def group_contractions_of_trace(trace) -> dict:
    """``{group id: [(work id, Contraction)]}`` of a trace's kernel groups
    (the pre-scheduling view, before the ranking stamps them)."""
    from ..spmw_autoschedule import _matcher_work_scope

    groups: dict = {}
    for match in getattr(trace, "matches", ()):
        scope = _matcher_work_scope(match)
        groups.setdefault(scope.group_id, []).append(
            (tuple(scope.work_id), match.extra.get("semantic_facts"))
        )
    return groups


def prove_dense_gemm_template(group):
    """Prove the dense GEMM template over the kernel group's work items, or
    name the first condition the source fails.

    Per work item (the `Contraction` facts of its one MAC): two rank-2
    operands and one rank-2 external result of FP16; a multiply-add with a
    zero accumulator; one reduction loop ``k`` and ``range(N)`` loops; the
    first operand read at ``[r, k]`` and the second at ``[k, c]`` with the
    reduction coordinate exactly ``k`` (lane order); the result written at
    ``[r, c]`` with the same ``r`` and ``c`` as the operands; ``r`` and
    ``c`` each a result loop plus a constant (or a constant), every result
    loop used, so the work item writes one rectangular block once.  Across
    the group: the same buffers and shapes, the blocks pairwise disjoint
    and together the whole ``[rows, columns]`` output (an untouched element
    would be overwritten by the device's gather).  Returns ``(template,
    None)`` or ``(None, reason)``.
    """
    if not group:
        return None, "the kernel group has no work items with retained facts"
    blocks = []
    identity = None
    for work_id, fact in group:
        if fact is None or not hasattr(fact, "reduction_vars"):
            return None, f"work item {tuple(work_id)} carries no contraction facts"
        labels = dict(fact.loop_labels)
        name = fact.function
        where = f"work item {tuple(work_id)} ({name!r})"
        inputs = tuple(fact.inputs)
        results = _result_buffers(fact)
        if len(inputs) != 2 or len(results) != 1:
            return None, (
                f"{where}: the device packs a two-operand contraction with one "
                f"external result; the body has {len(inputs)} operand(s) and "
                f"{len(results)} result(s)"
            )
        x, y = inputs
        output, result = results[0]
        if fact.accumulator_init != "zero":
            return (
                None,
                f"{where}: the accumulator is {fact.accumulator_init!r}, not zero",
            )
        if (fact.product_kind, fact.accumulate_kind) != ("mul", "add"):
            return None, (
                f"{where}: the device multiplies and adds; the body is "
                f"{fact.product_kind} / {fact.accumulate_kind}"
            )
        elements = {getattr(a.element, "text", None) for a in (x, y, result)}
        if elements != {"f16"}:
            return None, (
                f"{where}: the device packs FP16 operands and results; the body has "
                f"{sorted(str(e) for e in elements)}"
            )
        if any(not loop.normalized for loop in fact.loops):
            return None, f"{where}: a loop is not range(N) (lower bound 0, step 1)"
        if len(fact.reduction_vars) != 1:
            return None, (
                f"{where}: one reduction loop is packed in lane order; the body has "
                f"{len(fact.reduction_vars)}"
            )
        k = fact.reduction_vars[0]
        x_shape, y_shape = tuple(x.shape or ()), tuple(y.shape or ())
        out_shape = tuple(result.shape or ())
        x_coords, y_coords, out_coords = (
            tuple(x.coords or ()),
            tuple(y.coords or ()),
            tuple(result.coords or ()),
        )
        if (
            len(x_shape) != 2
            or len(y_shape) != 2
            or len(out_shape) != 2
            or len(x_coords) != 2
            or len(y_coords) != 2
            or len(out_coords) != 2
        ):
            return None, (
                f"{where}: the device packs rank-2 operands and a rank-2 result; got "
                f"{x.memref_name}{list(x_shape)}, {y.memref_name}{list(y_shape)} -> "
                f"{output}{list(out_shape)}"
            )
        k_form = ({k: 1}, 0)
        if _affine(x_coords[1]) != k_form:
            return None, (
                f"{where}: operand {x.memref_name!r} is read at "
                f"[{_affine_text(x_coords[0], labels)}, {_affine_text(x_coords[1], labels)}]"
                f" along the reduction; the device packs {x.memref_name}[r, "
                f"{labels.get(k, k)}] (the reduction in lane order)"
            )
        if _affine(y_coords[0]) != k_form:
            return None, (
                f"{where}: operand {y.memref_name!r} is read at "
                f"[{_affine_text(y_coords[0], labels)}, {_affine_text(y_coords[1], labels)}]"
                f" along the reduction; the device packs {y.memref_name}["
                f"{labels.get(k, k)}, c] (the reduction in lane order)"
            )
        row_form, col_form = _affine(x_coords[0]), _affine(y_coords[1])
        if row_form is None or col_form is None:
            return None, f"{where}: an operand coordinate is not affine in the loops"
        if _affine(out_coords[0]) != row_form or _affine(out_coords[1]) != col_form:
            return None, (
                f"{where}: the result {output!r} is written at "
                f"[{_affine_text(out_coords[0], labels)}, {_affine_text(out_coords[1], labels)}]"
                f" while the operands are read at row {_affine_text(x_coords[0], labels)} "
                f"of {x.memref_name!r} and column {_affine_text(y_coords[1], labels)} of "
                f"{y.memref_name!r}; the device computes {output}[r, c] = "
                f"sum {x.memref_name}[r, k] * {y.memref_name}[k, c]"
            )
        extents = []
        used = []
        for axis, (coefficients, constant) in (("row", row_form), ("column", col_form)):
            if len(coefficients) > 1 or any(c != 1 for c in coefficients.values()):
                text = _affine_text(
                    x_coords[0] if axis == "row" else y_coords[1], labels
                )
                return None, (
                    f"{where}: the {axis} coordinate {text} is not a loop plus a "
                    f"constant; the device writes one contiguous block of rows and "
                    "columns per work item"
                )
            var = next(iter(coefficients), None)
            if var is not None and var in used:
                return (
                    None,
                    f"{where}: loop {labels.get(var, var)} indexes both the row and the column",
                )
            used.append(var)
            extents.append(
                (constant, 1 if var is None else int(fact.trips.get(var) or 0))
            )
        unused = [v for v in fact.result_vars if v not in used]
        if unused:
            return None, (
                f"{where}: loop {labels.get(unused[0], unused[0])} does not index the "
                f"result {output!r} (its iterations rewrite the same elements); the "
                "device writes each output once"
            )
        reduction = int(fact.trips.get(k) or 0)
        if x_shape[1] != reduction or y_shape[0] != reduction:
            return None, (
                f"{where}: the reduction loop has {reduction} iterations; "
                f"{x.memref_name} has {x_shape[1]} columns and {y.memref_name} "
                f"{y_shape[0]} rows (the device packs the whole reduction)"
            )
        if out_shape != (x_shape[0], y_shape[1]):
            return None, (
                f"{where}: the result {output}{list(out_shape)} is not [{x_shape[0]}, "
                f"{y_shape[1]}] of {x.memref_name}[{x_shape[0]}, {reduction}] x "
                f"{y.memref_name}[{reduction}, {y_shape[1]}]"
            )
        (row0, rows), (col0, columns) = extents
        if (
            row0 < 0
            or col0 < 0
            or row0 + rows > out_shape[0]
            or col0 + columns > out_shape[1]
        ):
            return None, f"{where}: writes outside {output}{list(out_shape)}"
        this = (x.memref_name, y.memref_name, output, x_shape, y_shape, out_shape)
        if identity is None:
            identity = this
        elif this != identity:
            return None, (
                f"{where}: operates on {this[:3]} {list(x_shape)} x {list(y_shape)}, "
                f"another work item on {identity[:3]}; one GEMM per kernel group"
            )
        blocks.append(WorkRectangle(tuple(work_id), row0, rows, col0, columns))
    lhs, rhs, output, x_shape, _y_shape, out_shape = identity
    total_rows, total_columns = out_shape
    for index, block in enumerate(blocks):
        for other in blocks[index + 1 :]:
            if (
                block.row0 < other.row0 + other.rows
                and other.row0 < block.row0 + block.rows
                and block.col0 < other.col0 + other.columns
                and other.col0 < block.col0 + block.columns
            ):
                return None, (
                    f"work items {block.work_id} and {other.work_id} both write "
                    f"rows {block.row0}..{block.row0 + block.rows - 1} x columns "
                    f"{block.col0}..{block.col0 + block.columns - 1} of {output!r} "
                    "(overlapping blocks); the device writes each output once"
                )
    covered = sum(block.area for block in blocks)
    if covered != total_rows * total_columns:
        written = ", ".join(
            f"rows {b.row0}..{b.row0 + b.rows - 1} x columns "
            f"{b.col0}..{b.col0 + b.columns - 1}"
            for b in blocks
        )
        return None, (
            f"the work items write {covered} of the {total_rows * total_columns} "
            f"elements of {output}{list(out_shape)} ({written}); the device computes "
            "and gathers the whole output, so an untouched element would be "
            "overwritten"
        )
    reduction = int(x_shape[1])
    return (
        GemmTemplate(
            lhs, rhs, output, total_rows, total_columns, reduction, tuple(blocks)
        ),
        None,
    )


def grouped_device_packing(group, geometry: GroupedGeometry) -> GroupedPacking:
    """The grouped runtime's packing of one kernel group, from the proven
    dense GEMM template (`prove_dense_gemm_template`) of its work items.

    The runtime (`spmw_apu_v1._apu_v1_prepare_grouped_io`) packs
    ``lhs[M, K] * rhs[K, N]`` one dot product per ``group_size``-lane group
    (the reduction in lane order), the rows of ``lhs`` split over the four
    APUCs, ``ceil(ceil(M / 4) * N / groups_per_vr)`` batches per APUC, and
    gathers the group heads into the ``[M, N]`` result.  That is exactly the
    template's semantics (each element once, from a zero accumulator);
    anything the proof refuses is ``virtual_only`` with the reason.
    """
    reduction = 0
    if group:
        facts = [fact for _work_id, fact in group if fact is not None]
        if facts and hasattr(facts[0], "reduction_vars"):
            reduction = reduction_extent(facts[0])
    template, reason = prove_dense_gemm_template(group)
    if template is None:
        return GroupedPacking("virtual_only", reduction, 1, reason=reason)
    rows_per_apuc = math.ceil(template.rows / APUCS)
    batches = max(
        1, math.ceil(rows_per_apuc * template.columns / geometry.groups_per_vr)
    )
    return GroupedPacking(
        "gemm",
        template.reduction,
        batches,
        rows=template.rows,
        columns=template.columns,
        rows_per_apuc=rows_per_apuc,
        lhs=template.lhs,
        rhs=template.rhs,
        output=template.output,
        template=template,
    )


EXECUTABLE_BACKENDS = frozenset({None, "device", "functional"})


def require_grouped_device_packing(target, trace, backend) -> None:
    """Before scheduling: a dataflow region on APU v1 compiled for an
    executable backend must be the dense GEMM template the grouped runtime
    packs; otherwise compilation stops here with the condition the source
    fails (``backend="virtual"`` keeps the estimate, declared
    ``virtual_only``).  The functional backend has no model of the grouped
    program and is refused as such.  Plain-function programs (the vector
    plans) and traces without facts are not this route's."""
    if getattr(target, "name", None) != "apu_v1" or backend not in EXECUTABLE_BACKENDS:
        return
    from ..spmw_autoschedule import _matcher_work_scope

    matches = list(getattr(trace, "matches", ()))
    if not matches or not any(_matcher_work_scope(m).group_shape for m in matches):
        return
    if any(m.extra.get("semantic_facts") is None for m in matches):
        return
    if any(m.extra.get("program_facts") is not None for m in matches) and any(
        getattr(m.extra.get("program_facts"), "frontend", None) is not None
        for m in matches
    ):
        return
    if backend == "functional":
        raise NotImplementedError(
            "APU v1 grouped matcher program of a dataflow region: no host "
            "functional model exists; compile with backend='virtual' for the "
            "estimate or backend=None for the board"
        )
    for _group_id, group in group_contractions_of_trace(trace).items():
        ops = sorted({fact.target_op_name for _w, fact in group if fact is not None})
        if ops != ["MAC"]:
            # Refused by the enumerator with `GROUPED_ROUTE_OPERATIONS`.
            continue
        template, reason = prove_dense_gemm_template(group)
        if template is None:
            raise NotImplementedError(
                "APU v1 grouped matcher program: the device packs a dense FP16 GEMM "
                "(output[r, c] = sum_k lhs[r, k] * rhs[k, c] over the whole output, "
                f"zero accumulator) and this region is not one: {reason}; compile "
                "with backend='virtual' for the estimate only"
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
    # A map the executable does not realize invalidates every operand's and
    # result's packing: the contract reports it per candidate before
    # costing (`_derivation_problems`), the realization check refuses it at
    # materialization.
    map_problem = grouped_map_problem(getattr(placement, "layout", None), geometry)
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
    packing = grouped_device_packing(group_contractions(matches), geometry)
    # The batch count the device program must loop over: the proven
    # packing's for an executable GEMM (never the placement's own field, so
    # a placement whose body loops over fewer batches than the packing
    # needs declares the true count and is refused at materialization,
    # ISSUES X-30); the placement's tile count for an estimate-only program.
    if packing.kind == "gemm":
        batches = int(packing.batches)
    else:
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
                memref,
                value_axes(access, labels),
                storage,
                f"invalid:{map_problem}" if map_problem else f"group_lanes:{group}",
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
                f"invalid:{map_problem}" if map_problem else f"group_head:{group}",
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
    "GemmTemplate",
    "GroupedPacking",
    "LAUNCH_KIND",
    "WorkRectangle",
    "apu_v1_grouped_candidate_effects",
    "contraction_facts",
    "declared_launch",
    "group_contractions",
    "group_contractions_of_trace",
    "grouped_device_packing",
    "grouped_geometry",
    "grouped_map_problem",
    "grouped_runtime_binding",
    "prove_dense_gemm_template",
    "reduction_extent",
    "require_grouped_device_packing",
    "require_grouped_operations",
    "result_instances",
]
