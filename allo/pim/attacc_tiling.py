# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AttAcc tiling domain, command inventory and selected-program record.

An AttAcc kernel instance is one contraction family over three logical axes,
in this order: ``rows`` (heads x batch, the axis whose values select their own
vector), ``reduction`` and ``outputs``.  A GEMV is ``y[r, n] = sum_k
W[r, n, k] * x[r, k]``; attention scoring is the fused ``QK^T -> softmax ->
PV`` of one decode step with ``reduction = d_head`` and ``outputs = L``.
Axes are derived from the matcher's structural roles
(``match.extra["axis_roles"]``), never from loop positions.

A tiling assigns every axis a triple ``(a, b, s)``: ``a`` channels, ``b`` PIM
units (banks) per channel and a serial extent ``s`` with ``a * b * s >=
extent``.  The candidate space is the 3-D space of the DCC compiler (Yang et
al., "DCC", 2025; ``generate_configs_3D``), re-implemented here from the
target geometry; no DCC code is used.  It has two parts:

* every assignment whose channel factors and unit factors exhaust the device
  under integer division (``channels // a0 // a1 // a2 == 1`` and the same for
  units) with at least one full MAC column of reduction per unit
  (``s_reduction >= lanes``), keeping the first tiling per distinct serial
  triple, plus every assignment of AttAcc's fixed form (all channels on rows,
  one bank group's banks on the reduction, the bank groups on the outputs);
* when there are fewer rows than channels, AttAcc's two fixed mappings that
  spread the spare channels over the outputs or over the reduction.

The order of the list is the enumeration order (first-minimum ties of the
reference selector are resolved in it).

Legality comes from the target: factor products never exceed the declared
channels and units (by construction), every tiling covers its extents (ceil
padding), and a tiling is dropped when one row's working set (vector, matrix
tile and result, rounded to 32-byte columns) does not fit a bank.  A kernel
with no legal tiling raises :class:`AttAccDomainError`.

Command counts follow the AttAcc trace format (``PIM_WR_GB``, ``PIM_MAC_AB``,
``PIM_MV_SB``, ``PIM_MV_GB``, ``PIM_SFM``, ``ST``); they equal, command for
command, the trace of the DCC generator at the checkout recorded in
:mod:`allo.pim.costs.attacc` for all 668,575 candidates of the paper's 500
instances (``tests/pim/test_attacc_virtual.py`` pins sampled counts).
Before counting, a non-fixed tiling has its channel and unit factors rounded
up to powers of two, as the trace address mapping does; the rounded fan-out
may exceed the device for some candidates (it is the reference model's
convention and is what the counts describe).
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Literal


class AttAccDomainError(ValueError):
    """The kernel is outside the AttAcc tiling domain."""


# --------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class AttAccGeometry:
    """Target facts the tiling and the counts read (from the target tree)."""

    devices: int
    channels: int  # per device
    units: int  # PIM units (banks) per channel
    banks_per_group: int  # banks sharing one bank-group MV_SB path
    lanes: int  # FP16 MAC lanes per 32-byte column (n_mac)
    rows: int
    columns: int
    column_bytes: int
    element_bytes: int

    @property
    def bank_groups(self) -> int:
        return self.units // self.banks_per_group

    @property
    def bank_bytes(self) -> int:
        return self.rows * self.columns * self.column_bytes


def _unit_extent(target, name) -> int:
    extent = 1
    for factor in target.unit(name).mapping:
        extent *= int(factor)
    return extent


def attacc_geometry(target) -> AttAccGeometry:
    """Read the AttAcc geometry from a :func:`build_attacc_target` tree."""
    banks = target.banks.geometry
    units = _unit_extent(target, "pim")
    if int(banks["banks"]) != units:
        raise ValueError(
            f"AttAcc target declares {banks['banks']} banks per channel but "
            f"{units} PIM units"
        )
    groups = (
        int(banks["pseudochannels"]) * int(banks["ranks"]) * int(banks["bank_groups"])
    )
    if units % groups:
        raise ValueError(f"{units} banks do not divide into {groups} bank groups")
    width = int(banks["width"])
    element_bits = int(banks["element_bits"])
    return AttAccGeometry(
        devices=_unit_extent(target, "stack"),
        channels=_unit_extent(target, "channel"),
        units=units,
        banks_per_group=units // groups,
        lanes=width // element_bits,
        rows=int(banks["rows"]),
        columns=int(banks["cols"]),
        column_bytes=width // 8,
        element_bytes=element_bits // 8,
    )


# --------------------------------------------------------------------- #
# Kernel record
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class AttAccKernel:
    """One AttAcc kernel instance: family and logical extents."""

    kind: Literal["gemv", "attention"]
    rows: int
    reduction: int
    outputs: int

    @property
    def dims(self) -> tuple[int, int, int]:
        return (self.rows, self.reduction, self.outputs)

    def manifest(self) -> dict:
        return {
            "kind": self.kind,
            "rows": self.rows,
            "reduction": self.reduction,
            "outputs": self.outputs,
        }


_ATTENTION_SOFTMAX = ("SFM_EXP", "SFM_SUM", "SFM_DIV")


def _element_type(memref_type) -> str | None:
    text = str(memref_type or "")
    if not text.startswith("memref<"):
        return None
    return text[len("memref<") :].rstrip(">").split(",")[0].split("x")[-1].strip()


def _contraction_extents(match) -> tuple[int, int, int]:
    roles = match.extra.get("axis_roles") or {}
    reason = roles.get("unresolved", "no axis roles were stamped")
    if not roles or reason is not None:
        raise AttAccDomainError(
            f"AttAcc: the MAC in {match.func_name} has no structural axis roles "
            f"({reason})"
        )
    if roles.get("same_buffer"):
        raise AttAccDomainError(
            f"AttAcc: the MAC in {match.func_name} reads one buffer twice; the "
            "AttAcc MAC multiplies a bank-resident matrix by a GEMV-buffer vector"
        )
    from ..spmw_match_engine import output_map_problem

    problem = output_map_problem(match)
    if problem is not None:
        raise AttAccDomainError(
            f"AttAcc: the MAC in {match.func_name} does not write each result at "
            f"the row it reads: {problem}"
        )
    trips = roles["trips"]

    def extent(axes):
        value = 1
        for var in axes:
            value *= int(trips[var])
        return value

    init = match.extra.get("accumulator_init")
    if init != "zero":
        raise AttAccDomainError(
            f"AttAcc: the MAC accumulator in {match.func_name} is {init!r}; the "
            "AttAcc command stream starts every output from zero "
            "(`acc = 0; for k: acc += ...; y[...] = acc`)"
        )
    for operand in match.operands:
        if operand.memref_name is None:
            continue
        element = _element_type(getattr(operand, "memref_type", None))
        if element != "f16":
            raise AttAccDomainError(
                f"AttAcc: operand {operand.memref_name!r} of the MAC in "
                f"{match.func_name} has element type {element!r}; the AttAcc "
                "datapath and its command counts are FP16"
            )
    return extent(roles["batch"]), extent(roles["reduction"]), extent(roles["output"])


def _loop_extent(match) -> int:
    trips = (match.extra.get("axis_roles") or {}).get("trips") or {}
    if trips:
        values = trips.values()
    else:
        from ..spmw_tripcount import _parse_loop_bound

        values = []
        for _var, lower, upper, step in match.enclosing_loops:
            lo, hi = _parse_loop_bound(lower), _parse_loop_bound(upper)
            if lo is None or hi is None:
                return -1
            values.append(-(-(hi - lo) // int(step)))
    out = 1
    for value in values:
        out *= int(value)
    return out


def attacc_kernel_from_matches(matches) -> AttAccKernel:
    """Classify the matches of one work item as an AttAcc kernel.

    A GEMV is exactly one ``MAC``.  Attention scoring is exactly two ``MAC``
    (score, then context) and one each of ``SFM_EXP``, ``SFM_SUM`` and
    ``SFM_DIV`` over the score rows, where the context reduces over the score
    outputs and produces the score reduction extent.  Anything else raises.
    """
    from ..spmw_autoschedule import _matcher_work_scope

    matches = list(matches)
    if not matches:
        raise AttAccDomainError("AttAcc: empty kernel")
    shape = tuple(_matcher_work_scope(matches[0]).group_shape or (1,))
    if math.prod(shape) != 1:
        raise AttAccDomainError(
            f"AttAcc: work grid {list(shape)} has {math.prod(shape)} items; the "
            "attacc_tiling knob decomposes ONE work item over a device's "
            "channels and units, so write the kernel with @allo.work(mapping=[1])"
        )
    names = [match.target_op_name for match in matches]
    unknown = sorted(set(names) - {"MAC", *_ATTENTION_SOFTMAX})
    if unknown:
        raise AttAccDomainError(f"AttAcc: no lowering for {unknown} in this kernel")
    macs = [match for match in matches if match.target_op_name == "MAC"]
    softmax = [match for match in matches if match.target_op_name != "MAC"]
    if len(macs) == 1 and not softmax:
        rows, reduction, outputs = _contraction_extents(macs[0])
        return AttAccKernel("gemv", rows, reduction, outputs)
    counts = sorted(match.target_op_name for match in softmax)
    if len(macs) == 2 and counts == sorted(_ATTENTION_SOFTMAX):
        score, context = (_contraction_extents(match) for match in macs)
        rows, d_head, length = score
        if context != (rows, length, d_head):
            raise AttAccDomainError(
                f"AttAcc: score {score} and context {context} (rows, reduction, "
                "outputs) do not form QK^T then PV over the same heads"
            )
        for match in softmax:
            if _loop_extent(match) != rows * length:
                raise AttAccDomainError(
                    f"AttAcc: {match.target_op_name} spans {_loop_extent(match)} "
                    f"elements, not the {rows}x{length} score rows"
                )
        _prove_attention_dataflow(matches)
        return AttAccKernel("attention", rows, d_head, length)
    raise AttAccDomainError(
        f"AttAcc: kernel matches {names}; supported are one MAC (GEMV) or "
        "MAC, SFM_EXP, SFM_SUM, SFM_DIV, MAC (attention scoring)"
    )


def _coord_labels(access, labels) -> tuple | None:
    """The loop labels an access's coordinates range over, one per
    coordinate, when every coordinate is one loop variable with unit
    coefficient and no offset (``s[h, t]``); None otherwise."""
    out = []
    for coord in getattr(access, "coords", None) or ():
        if coord is None:
            return None
        atoms, const = coord
        if int(const) != 0 or len(atoms) != 1 or int(atoms[0][1]) != 1:
            return None
        label = labels.get(atoms[0][0])
        if label is None:
            return None
        out.append(label)
    return tuple(out)


def _prove_attention_dataflow(matches) -> None:
    """Prove, from the shared semantic facts, that the five matches are one
    attention kernel the fused AttAcc inventory realizes: the score MAC's
    sink feeds the exponential, the sum and the division, which feed the
    context MAC's vector (retained value identities, not names); every
    softmax access reads and writes the score element ``[row, length]`` of
    its own iteration; the sum's accumulator is initialized under the row
    loop and the division reads it inside the same row iteration (per-row
    normalization); and the context MAC reduces over the length axis.
    Programs with another dataflow (a softmax over a different buffer, a
    normalization across rows) are refused with the difference named."""
    facts = [match.extra.get("semantic_facts") for match in matches]
    program = matches[0].extra.get("program_facts") if matches else None
    if any(fact is None for fact in facts) or program is None:
        raise AttAccDomainError(
            "AttAcc attention needs the shared semantic facts of its work body "
            "(a trace assembled outside the shared front stages is not lowered)"
        )
    by_op = {}
    for match, fact in zip(matches, facts):
        by_op.setdefault(match.target_op_name, []).append(fact)
    (score, context) = by_op["MAC"]
    (exp,) = by_op["SFM_EXP"]
    (total,) = by_op["SFM_SUM"]
    (div,) = by_op["SFM_DIV"]
    labels = {}
    for fact in facts:
        labels.update(fact.loop_labels)

    def fail(what):
        raise AttAccDomainError(f"AttAcc attention: {what}")

    # Axes: the score's result loops are (rows..., length); the length is the
    # axis the context MAC reduces over.
    score_result = tuple(labels[v] for v in score.result_vars)
    context_reduction = tuple(labels[v] for v in context.reduction_vars)
    if len(context_reduction) != 1 or context_reduction[0] not in score_result:
        fail(
            f"the context MAC reduces over {context_reduction}, not over an axis "
            f"of the score result {score_result}"
        )
    length = context_reduction[0]
    row_labels = tuple(label for label in score_result if label != length)
    if not row_labels or score_result[-1] != length:
        fail(
            f"the score result axes are {score_result}; expected (rows..., "
            f"length={length})"
        )
    # Dataflow by retained identity.
    sinks = [sink for sink in score.sinks if sink.value_ref is not None]
    if len(sinks) != 1:
        fail(f"the score MAC writes {len(sinks)} buffers, not one score buffer")
    scores_ref = sinks[0].value_ref
    if _coord_labels(sinks[0], labels) != score_result:
        fail("the score buffer is not indexed by the score result axes")
    (exp_in,) = exp.operands
    if exp_in.value_ref != scores_ref or exp.result.value_ref != scores_ref:
        fail(
            f"the exponential reads {exp_in.memref_name!r} and writes "
            f"{exp.result.memref_name!r}, not the score buffer "
            f"{sinks[0].memref_name!r} the first MAC writes"
        )
    (sum_in,) = total.inputs
    if sum_in.value_ref != scores_ref:
        fail(f"the softmax sum reads {sum_in.memref_name!r}, not the score buffer")
    total_ref = total.accumulator.value_ref
    div_refs = {operand.value_ref for operand in div.operands}
    if div_refs != {scores_ref, total_ref} or div.result.value_ref != scores_ref:
        fail(
            "the division does not divide the exponentiated scores by the "
            "softmax sum in place"
        )
    if context.vector is None or context.vector.value_ref != scores_ref:
        fail("the context MAC's vector is not the normalized score buffer")
    # Access relations: every softmax access is the iteration's own element.
    for name, accesses in (
        ("SFM_EXP", (exp_in, exp.result)),
        ("SFM_SUM", (sum_in,)),
        (
            "SFM_DIV",
            (*[o for o in div.operands if o.value_ref == scores_ref], div.result),
        ),
    ):
        for access in accesses:
            if _coord_labels(access, labels) != score_result:
                fail(
                    f"{name} accesses the score buffer over "
                    f"{_coord_labels(access, labels)}, not [{', '.join(score_result)}]"
                )
    # Per-row normalization: the sum's accumulator starts under the row
    # loop(s) and the division reads it inside the same row iteration.
    row_vars = tuple(v for v in total.loop_vars if labels[v] in row_labels)
    if tuple(labels[v] for v in total.loop_vars) != row_labels + (length,):
        fail(
            f"the softmax sum iterates {[labels[v] for v in total.loop_vars]}, "
            f"not the rows {row_labels} then the length {length}"
        )
    inits = [
        effect
        for effect in program.effects
        if effect.category == "accumulator_init"
        and effect.destination.value_ref == total_ref
    ]
    if not inits or any(
        tuple(loop.var for loop in effect.loops) != row_vars for effect in inits
    ):
        fail(
            "the softmax sum is not re-initialized per row: its initializer sits "
            f"under loops {[[loop.var for loop in e.loops] for e in inits]}, the "
            f"rows are {list(row_vars)} (a normalization across rows is not the "
            "AttAcc softmax)"
        )
    div_rows = tuple(v for v in div.loop_vars if labels[v] in row_labels)
    if div_rows != row_vars:
        fail(
            "the division runs in another row loop than the sum; the row's total "
            "is consumed outside its iteration"
        )


# --------------------------------------------------------------------- #
# Tiling and its space
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class AttAccTiling:
    """``((a, b, s) per axis)`` for the axes (rows, reduction, outputs)."""

    dims: tuple[tuple[int, int, int], ...]

    @classmethod
    def coerce(cls, value) -> "AttAccTiling":
        if isinstance(value, AttAccTiling):
            return value
        dims = tuple(tuple(int(item) for item in axis) for axis in value)
        if len(dims) != 3 or any(len(axis) != 3 for axis in dims):
            raise ValueError(
                f"AttAcc tiling must be three (a, b, s) triples, got {value!r}"
            )
        return cls(dims)

    @property
    def channels(self) -> int:
        return math.prod(axis[0] for axis in self.dims)

    @property
    def units(self) -> int:
        return math.prod(axis[1] for axis in self.dims)

    def manifest(self) -> tuple[tuple[int, int, int], ...]:
        return self.dims

    def dcc_tiling(self) -> list[list[int]]:
        """The tiling in the list form the AttAcc/DCC trace generators take."""
        return [list(axis) for axis in self.dims]

    def __str__(self) -> str:
        return str(self.dcc_tiling()).replace(" ", "")


def _ceil(a: int, b: int) -> int:
    return -(-a // b)


def _round_up(nbytes: int, granule: int) -> int:
    return _ceil(nbytes, granule) * granule


def _fixed_form(dims, geometry) -> bool:
    """AttAcc's fixed per-channel bank split: one bank group's banks on the
    reduction, the channel's bank groups on the outputs."""
    return dims[1][1] == geometry.banks_per_group and dims[2][1] == geometry.bank_groups


def _row_footprint(tiling: AttAccTiling, kind: str, geometry, rows: int) -> int:
    """Bank bytes of ``rows`` rows: vector, matrix tile and result."""
    s_red, s_out = tiling.dims[1][2], tiling.dims[2][2]
    result = s_out if kind == "gemv" else s_red
    size = geometry.element_bytes
    granule = geometry.column_bytes
    return (
        _round_up(rows * s_red * size, granule)
        + _round_up(rows * s_red * s_out * size, granule)
        + _round_up(rows * result * size, granule)
    )


def tiling_violations(
    kernel: AttAccKernel, tiling: AttAccTiling, geometry
) -> list[str]:
    """Why ``tiling`` is not a legal AttAcc tiling of ``kernel`` (empty if legal)."""
    out = []
    for name, (a, b, s), extent in zip(
        ("rows", "reduction", "outputs"), tiling.dims, kernel.dims
    ):
        if min(a, b, s) < 1:
            out.append(f"{name}: factors must be positive, got {(a, b, s)}")
        elif a * b * s < extent:
            out.append(f"{name}: {a}x{b}x{s} does not cover extent {extent}")
    if tiling.channels > geometry.channels:
        out.append(f"channel product {tiling.channels} exceeds {geometry.channels}")
    if tiling.units > geometry.units:
        out.append(f"unit product {tiling.units} exceeds {geometry.units}")
    if (
        not out
        and _row_footprint(tiling, kernel.kind, geometry, 1) > geometry.bank_bytes
    ):
        out.append(
            f"one row needs {_row_footprint(tiling, kernel.kind, geometry, 1)} "
            f"bytes per bank; a bank holds {geometry.bank_bytes}"
        )
    return out


@functools.lru_cache(maxsize=256)
def _space(kernel: AttAccKernel, geometry: AttAccGeometry) -> tuple[AttAccTiling, ...]:
    extents = kernel.dims
    out: list[tuple[tuple[int, int, int], ...]] = []
    seen_serial: set = set()

    def recurse(axis, channels_left, units_left, chosen):
        if axis == len(extents):
            serial = tuple(item[2] for item in chosen)
            if chosen[0][0] == geometry.channels and _fixed_form(chosen, geometry):
                out.append(tuple(chosen))
                seen_serial.add(serial)
            elif (
                channels_left == 1
                and units_left == 1
                and chosen[1][2] >= geometry.lanes
                and serial not in seen_serial
            ):
                out.append(tuple(chosen))
                seen_serial.add(serial)
            return
        extent = extents[axis]
        for a in range(1, min(channels_left, extent) + 1):
            for b in range(1, min(units_left, extent) + 1):
                if a * b > extent:
                    break
                chosen.append((a, b, _ceil(extent, a * b)))
                recurse(axis + 1, channels_left // a, units_left // b, chosen)
                chosen.pop()

    recurse(0, geometry.channels, geometry.units, [])

    rows, reduction, outputs = extents
    if rows < geometry.channels:
        on_rows = min(rows, geometry.channels)
        spare = geometry.channels // on_rows
        bpg, groups = geometry.banks_per_group, geometry.bank_groups
        row_axis = (on_rows, 1, _ceil(rows, on_rows))
        for fixed in (
            (
                row_axis,
                (1, bpg, _ceil(reduction, bpg)),
                (spare, groups, _ceil(outputs, groups * spare)),
            ),
            (
                row_axis,
                (spare, bpg, _ceil(reduction, bpg * spare)),
                (1, groups, _ceil(outputs, groups)),
            ),
        ):
            if fixed not in out:
                out.append(fixed)

    legal = []
    for dims in out:
        tiling = AttAccTiling(dims)
        if not tiling_violations(kernel, tiling, geometry):
            legal.append(tiling)
    return tuple(legal)


def attacc_tiling_space(
    kernel: AttAccKernel, geometry: AttAccGeometry
) -> list[AttAccTiling]:
    """Every legal tiling of ``kernel`` on ``geometry``, in enumeration order."""
    space = _space(kernel, geometry)
    if not space:
        raise AttAccDomainError(
            f"AttAcc: no legal tiling for {kernel.kind} rows={kernel.rows} "
            f"reduction={kernel.reduction} outputs={kernel.outputs} on "
            f"{geometry.channels} channels x {geometry.units} units "
            f"({geometry.bank_bytes} bytes per bank)"
        )
    return list(space)


# --------------------------------------------------------------------- #
# Command inventory
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class AttAccCounts:
    """Commands of one kernel instance under one tiling (whole device).

    ``iterations`` is the number of row blocks streamed through the banks
    (the smallest count whose per-bank working set fits); ``rows_per_iteration``
    the rows of one block.  ``channels_used`` and ``units_used`` are the
    power-of-two rounded fan-outs the trace addresses.  ``serial_reduction``
    and ``serial_outputs`` are the per-unit serial extents.
    """

    wr_gb: int
    mac_ab: int
    mv_sb: int
    mv_gb: int
    sfm: int
    st: int
    iterations: int
    rows_per_iteration: int
    channels_used: int
    units_used: int
    serial_reduction: int
    serial_outputs: int

    def commands(self) -> tuple[tuple[str, int], ...]:
        """``(trace opcode, count)`` pairs, zero counts omitted."""
        pairs = (
            ("PIM_WR_GB", self.wr_gb),
            ("PIM_MAC_AB", self.mac_ab),
            ("PIM_MV_SB", self.mv_sb),
            ("PIM_MV_GB", self.mv_gb),
            ("PIM_SFM", self.sfm),
            ("ST", self.st),
        )
        return tuple((name, count) for name, count in pairs if count)

    def manifest(self) -> dict:
        return {
            **dict(self.commands()),
            "iterations": self.iterations,
            "rows_per_iteration": self.rows_per_iteration,
            "channels_used": self.channels_used,
            "units_used": self.units_used,
            "serial_reduction": self.serial_reduction,
            "serial_outputs": self.serial_outputs,
        }


def _pow2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def _iterations(tiling: AttAccTiling, kind: str, geometry) -> tuple[int, int]:
    s_rows = tiling.dims[0][2]
    n = 1
    while n <= s_rows:
        rows = _ceil(s_rows, n)
        if _row_footprint(tiling, kind, geometry, rows) <= geometry.bank_bytes:
            break
        n += 1
    return n, _ceil(s_rows, n)


def attacc_command_counts(
    kernel: AttAccKernel, tiling: AttAccTiling, geometry: AttAccGeometry
) -> AttAccCounts:
    """Command inventory of ``kernel`` under ``tiling`` (see the module doc)."""
    tiling = AttAccTiling.coerce(tiling)
    n, it = _iterations(tiling, kernel.kind, geometry)
    dims = [(tiling.dims[0][0], tiling.dims[0][1], it), tiling.dims[1], tiling.dims[2]]
    fixed = (
        dims[0][1] == 1
        and dims[1][0] == 1
        and dims[2][0] == 1
        and _fixed_form(dims, geometry)
    )
    if not fixed:
        dims = [(_pow2(a), _pow2(b), s) for a, b, s in dims]
    ch = math.prod(axis[0] for axis in dims)
    ba = math.prod(axis[1] for axis in dims)
    s1, s2 = dims[1][2], dims[2][2]
    lanes, bpg = geometry.lanes, geometry.banks_per_group
    if kernel.kind == "gemv":
        wr_gb = _ceil(it * s1, lanes) * ba * ch
        mac_ab = s2 * _ceil(s1, lanes) * ch * it
        mv_sb = _ceil(s2, lanes) * _ceil(ba, bpg) * ch * it
        mv_gb = sfm = 0
        st = (_ceil(it * s1, lanes) + _ceil(it * s2, lanes)) * ba * ch
    else:
        wr_gb = _ceil(it * s1, lanes) * ba * ch
        mac_ab = it * ch * (s2 * _ceil(s1, lanes) + s1 * _ceil(s2, lanes))
        mv_sb = it * _ceil(s1, lanes) * _ceil(ba, bpg) * ch
        mv_gb = _ceil(it * s1, lanes) * ba * ch
        # One softmax per row-unit slot on every channel of the device.
        sfm = dims[0][1] * geometry.channels
        st = 2 * _ceil(it * s1, lanes) * ba * ch
    return AttAccCounts(
        wr_gb=wr_gb * n,
        mac_ab=mac_ab * n,
        mv_sb=mv_sb * n,
        mv_gb=mv_gb * n,
        sfm=sfm * n,
        st=st * n,
        iterations=n,
        rows_per_iteration=it,
        channels_used=ch,
        units_used=ba,
        serial_reduction=s1,
        serial_outputs=s2,
    )


# --------------------------------------------------------------------- #
# Selected-program record
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class AttAccProgram:
    """The emitted AttAcc program of one kernel: kernel, tiling, commands."""

    kernel: AttAccKernel
    tiling: AttAccTiling
    counts: AttAccCounts
    geometry: AttAccGeometry

    def dcc_tiling(self) -> list[list[int]]:
        return self.tiling.dcc_tiling()

    def command_lines(self) -> list[str]:
        return [f"{name} count={count}" for name, count in self.counts.commands()]

    def manifest(self) -> dict:
        return {
            "kernel": self.kernel.manifest(),
            "tiling": self.dcc_tiling(),
            "counts": self.counts.manifest(),
        }


def attacc_program(kernel, tiling, geometry) -> AttAccProgram:
    """Validate ``tiling`` for ``kernel`` and build its program record."""
    tiling = AttAccTiling.coerce(tiling)
    bad = tiling_violations(kernel, tiling, geometry)
    if bad:
        raise AttAccDomainError(f"AttAcc: illegal tiling {tiling}: {'; '.join(bad)}")
    return AttAccProgram(
        kernel, tiling, attacc_command_counts(kernel, tiling, geometry), geometry
    )


def rank_attacc_tilings(
    kernel, geometry, cost_fn=None
) -> list[tuple[AttAccTiling, float]]:
    """Every legal tiling with its closed-form cycles, cheapest first.

    Ties keep enumeration order (the reference selector's first minimum).
    ``cost_fn(program) -> cycles`` defaults to
    :func:`allo.pim.costs.attacc.attacc_kernel_cycles`, the function the
    ``attacc_cost`` rules call.  Used by the selector-timing script; the
    compiler ranks through ``allo.compile`` (integer cycles, then
    ``spmw_autoschedule.TIE_BREAK_RULE``).
    """
    if cost_fn is None:
        from .costs.attacc import attacc_kernel_cycles as cost_fn
    scored = [
        (
            tiling,
            cost_fn(
                AttAccProgram(
                    kernel,
                    tiling,
                    attacc_command_counts(kernel, tiling, geometry),
                    geometry,
                )
            ),
        )
        for tiling in attacc_tiling_space(kernel, geometry)
    ]
    return sorted(scored, key=lambda item: item[1])


__all__ = [
    "AttAccCounts",
    "AttAccDomainError",
    "AttAccGeometry",
    "AttAccKernel",
    "AttAccProgram",
    "AttAccTiling",
    "attacc_command_counts",
    "attacc_geometry",
    "attacc_kernel_from_matches",
    "attacc_program",
    "attacc_tiling_space",
    "rank_attacc_tilings",
    "tiling_violations",
]
