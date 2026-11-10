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
    # The bank-resident (matrix) operand of a GEMV kernel, by memref name:
    # the orientation the kernel realizes (`attacc_orientations`).  None
    # for attention (its roles follow from index containment) and for
    # records built before orientations were explicit.
    matrix: str | None = None

    @property
    def dims(self) -> tuple[int, int, int]:
        return (self.rows, self.reduction, self.outputs)

    def manifest(self) -> dict:
        out = {
            "kind": self.kind,
            "rows": self.rows,
            "reduction": self.reduction,
            "outputs": self.outputs,
        }
        if self.matrix is not None:
            out["matrix"] = self.matrix
        return out


_ATTENTION_SOFTMAX = ("SFM_EXP", "SFM_SUM", "SFM_DIV")


def _element_type(memref_type) -> str | None:
    text = str(memref_type or "")
    if not text.startswith("memref<"):
        return None
    return text[len("memref<") :].rstrip(">").split(",")[0].split("x")[-1].strip()


def attacc_orientations(matches) -> list[dict]:
    """The storage orientations of one work item's kernel: which input the
    AttAcc datapath keeps in the banks (the matrix) and which it streams
    through the GEMV buffer (the vector).

    The shared role analysis names the matrix by index containment
    (``K[h, s, d]`` over ``q[h, d]``, ``W[i, k]`` over ``x[k]``); that is the
    one orientation.  When neither input's index set contains the other's
    (a GEMM: each input reads the reduction loop and one result loop), the
    analysis stamps ``matrix_by == "pattern_order"`` and the naming is a
    convention, not a storage decision (release ledger X-41): both
    orientations are legal realizations of the same contraction (``C`` or
    ``C^T`` as a batch of GEMVs), so both are candidates, priced and ranked
    with their tilings; the candidate set does not depend on which input
    is written first.  Each entry: ``matrix``/``vector`` memref names,
    their operand roles, and ``by`` (``roles`` or ``alternative``)."""
    macs = [match for match in matches if match.target_op_name == "MAC"]
    if len(macs) != 1:
        return [{"by": "roles"}]
    match = macs[0]
    roles = match.extra.get("axis_roles") or {}
    by_role = {op.role: op for op in match.operands}
    matrix_role, vector_role = roles.get("matrix"), roles.get("vector")
    if matrix_role not in by_role or vector_role not in by_role:
        return [{"by": "roles"}]
    first = {
        "matrix": by_role[matrix_role].memref_name,
        "vector": by_role[vector_role].memref_name,
        "matrix_role": matrix_role,
        "vector_role": vector_role,
        "by": "roles",
    }
    if roles.get("matrix_by") != "pattern_order" or roles.get("same_buffer"):
        return [first]
    second = {
        "matrix": by_role[vector_role].memref_name,
        "vector": by_role[matrix_role].memref_name,
        "matrix_role": vector_role,
        "vector_role": matrix_role,
        "by": "alternative",
    }
    return [first, second]


def _oriented_extents(match, roles, orientation) -> tuple[int, int, int]:
    """``(rows, reduction, outputs)`` under an explicit orientation: the
    result loops that index the vector operand are the GEMV rows (a vector
    per iteration), the other result loops the outputs (the matrix rows),
    from the operands' resolved coordinates."""
    from ..spmw_match_engine import operand_index_terms

    trips = roles["trips"]
    loop_vars = [loop[0] for loop in match.enclosing_loops]
    by_role = {op.role: op for op in match.operands}
    vector = by_role[orientation["vector_role"]]
    vector_vars, _opaque = operand_index_terms(vector, set(loop_vars))
    result_vars = [
        var
        for var in loop_vars
        if var in set(roles.get("output", ())) | set(roles.get("batch", ()))
    ]
    rows = outputs = reduction = 1
    for var in result_vars:
        if var in vector_vars:
            rows *= int(trips[var])
        else:
            outputs *= int(trips[var])
    for var in roles.get("reduction", ()):
        reduction *= int(trips[var])
    return rows, reduction, outputs


def _contraction_extents(match, orientation=None) -> tuple[int, int, int]:
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
    if orientation is not None and orientation.get("vector_role") is not None:
        return _oriented_extents(match, roles, orientation)
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


def attacc_kernel_from_matches(matches, orientation=None) -> AttAccKernel:
    """Classify the matches of one work item as an AttAcc kernel.

    A GEMV is exactly one ``MAC``.  Attention scoring is exactly two ``MAC``
    (score, then context) and one each of ``SFM_EXP``, ``SFM_SUM`` and
    ``SFM_DIV`` over the score rows, where the context reduces over the score
    outputs and produces the score reduction extent.  Anything else raises.
    ``orientation`` (one entry of `attacc_orientations`) fixes which GEMV
    input is the bank-resident matrix; the default is the orientation the
    shared roles name.
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
        if orientation is None or orientation.get("vector_role") is None:
            orientation = attacc_orientations(matches)[0]
        rows, reduction, outputs = _contraction_extents(macs[0], orientation)
        return AttAccKernel("gemv", rows, reduction, outputs, orientation.get("matrix"))
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


def _own_element_vars(access, loops) -> tuple | None:
    """The loop variable indexing each dimension of ``access`` when the access
    is the iteration's own element of the whole buffer: every coordinate is
    one loop variable of ``loops`` with unit coefficient and no offset, the
    variables are distinct and each loop's trip count equals the dimension
    extent.  None otherwise.  Loop identity is the retained variable, never
    a source spelling."""
    shape = getattr(access, "shape", None)
    coords = getattr(access, "coords", None) or ()
    if not shape or len(coords) != len(shape):
        return None
    trips = {loop.var: loop.trip for loop in loops}
    out = []
    for coord, extent in zip(coords, shape):
        if coord is None:
            return None
        atoms, const = coord
        if int(const) != 0 or len(atoms) != 1 or int(atoms[0][1]) != 1:
            return None
        var = atoms[0][0]
        if var not in trips or trips[var] != int(extent) or var in out:
            return None
        out.append(var)
    return tuple(out)


def _prove_attention_dataflow(matches) -> None:
    """Prove, from the shared semantic facts, that the five matches are one
    attention kernel the fused AttAcc inventory realizes.  The proof is
    structural: buffers are followed by retained value identity, axes by
    the dimensions of the score buffer, loops by their retained variables.

    * the score MAC writes one local score buffer ``S`` with the iteration's
      own element (one loop per dimension); the dimension of ``S`` the
      context MAC's vector access indexes with the context's reduction loop
      is the length axis, the others are the row axes; the extents agree
      with the kernel's (rows, d_head, length);
    * the exponential reads and writes ``S``, the sum reads ``S``, the
      division writes ``S`` as ``S / total`` (numerator ``S``, denominator
      the sum's accumulator; the reverse form is refused), and the context
      MAC's vector is ``S``: each as the iteration's own element;
    * per-row normalization: the sum's accumulator initializer (the retained
      ``accumulator_init`` store effect) sits under exactly the loops that
      index the row dimensions in the sum, and the division indexes the row
      dimensions with those same loops (the row's total is consumed inside
      its row iteration).

    Any other dataflow is an `AttAccDomainError` naming the difference.
    """
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

    def fail(what):
        raise AttAccDomainError(f"AttAcc attention: {what}")

    # The score buffer and its dimensions.
    sinks = [sink for sink in score.sinks if sink.value_ref is not None]
    if len(sinks) != 1 or not sinks[0].local:
        fail(f"the score MAC writes {len(sinks)} buffers, not one local score buffer")
    scores = sinks[0]
    scores_ref = scores.value_ref
    score_vars = _own_element_vars(scores, score.loops)
    if score_vars is None or set(score_vars) != set(score.result_vars):
        fail(
            "the score buffer is not written as the iteration's own element over the score MAC's result loops"
        )
    shape = tuple(int(d) for d in scores.shape)
    # The length axis: the dimension the context MAC reduces over.
    if context.vector is None or context.vector.value_ref != scores_ref:
        fail("the context MAC's vector is not the score buffer the first MAC writes")
    context_vars = _own_element_vars(context.vector, context.loops)
    if context_vars is None:
        fail(
            "the context MAC does not read the score buffer as its iteration's own element"
        )
    length_dims = [
        dim for dim, var in enumerate(context_vars) if var in context.reduction_vars
    ]
    if len(length_dims) != 1 or any(
        var not in context.result_vars
        for dim, var in enumerate(context_vars)
        if dim != length_dims[0]
    ):
        fail(
            "the context MAC must reduce over exactly one dimension of the score "
            "buffer and keep the others as result axes"
        )
    length_dim = length_dims[0]
    row_dims = tuple(dim for dim in range(len(shape)) if dim != length_dim)
    if score_vars[length_dim] in score.reduction_vars:
        fail("the length axis is a reduction axis of the score MAC")
    rows = math.prod(shape[dim] for dim in row_dims)
    extents = _contraction_extents_from_facts(score)
    if extents is not None and (rows, shape[length_dim]) != (extents[0], extents[2]):
        fail(
            f"the score buffer {shape} does not match the score MAC's rows and "
            f"length {extents[0]} x {extents[2]}"
        )

    # Dataflow and access relations of the softmax, by retained identity.
    def own(name, access):
        if access.value_ref != scores_ref:
            fail(
                f"{name} uses {access.memref_name!r}, not the score buffer "
                f"{scores.memref_name!r} the first MAC writes"
            )
        variables = _own_element_vars(access, by_fact[name].loops)
        if variables is None:
            fail(
                f"{name} does not access the score buffer as its iteration's own element"
            )
        return variables

    by_fact = {"SFM_EXP": exp, "SFM_SUM": total, "SFM_DIV": div}
    (exp_in,) = exp.operands
    own("SFM_EXP", exp_in)
    own("SFM_EXP", exp.result)
    (sum_in,) = total.inputs
    sum_vars = own("SFM_SUM", sum_in)
    total_ref = total.accumulator.value_ref
    if not total.accumulator.local or total.accumulator.coords not in ((), None):
        fail("the softmax sum does not accumulate into a local scalar")
    if len(div.operands) != 2:
        fail("the division does not have two operands")
    numerator, denominator = div.operands
    if numerator.value_ref != scores_ref or denominator.value_ref != total_ref:
        fail(
            "the division is not (exponentiated score) / (softmax sum): the "
            f"operands are {numerator.memref_name!r} / {denominator.memref_name!r}"
        )
    div_vars = own("SFM_DIV", numerator)
    if own("SFM_DIV", div.result) != div_vars:
        fail("the division does not write the element it reads")

    # Per-row normalization: initializer scope and consumer scope by loop
    # identity (the retained variables), rows by buffer dimension.
    sum_rows = tuple(sum_vars[dim] for dim in row_dims)
    inits = [
        effect
        for effect in program.effects
        if effect.category == "accumulator_init"
        and effect.destination.value_ref == total_ref
    ]
    if not inits or any(
        tuple(loop.var for loop in effect.loops) != sum_rows for effect in inits
    ):
        fail(
            "the softmax sum is not re-initialized per row: its initializer sits "
            f"under loops {[[loop.var for loop in e.loops] for e in inits]}, the "
            f"row loops are {list(sum_rows)} (a normalization across rows is not "
            "the AttAcc softmax)"
        )
    if tuple(div_vars[dim] for dim in row_dims) != sum_rows:
        fail(
            "the division runs in another row loop than the sum; the row's total "
            "is consumed outside its iteration"
        )


def _contraction_extents_from_facts(fact) -> tuple | None:
    """``(rows, reduction, outputs)`` from the match's structural axis roles,
    None when the roles are unresolved (the caller's extents already failed
    in that case)."""
    try:
        return _contraction_extents(fact.match)
    except AttAccDomainError:
        return None


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
    "attacc_orientations",
    "attacc_program",
    "attacc_tiling_space",
    "rank_attacc_tilings",
    "tiling_violations",
]
