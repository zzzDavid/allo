# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Samsung HBM-PIM lowering and PIMSimulator runtime.

The codegen context lowers the selected placement to a per-pass CRF program
(the words PIM blocks execute) plus a program record of the facts the
conductor needs (reduction tiles, the bank fiber of each tile, GRF_A fill,
rows per block, operand bindings of every CRF word).  The runtime derives the
data placement from that record and the simulator geometry, uploads the
emitted CRF verbatim, and reads back the drained results.  The conductor
(``pim_driver --op GENERIC``, ``tests/TenonConductor.inc``) never regenerates
microcode.  Contract: ``docs/release/design/samsung-execution.md``.
"""

from __future__ import annotations

import math
from collections import Counter
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .spmw_match import MatchedOp
from .spmw_target import MemoryRef, Register
from .spmw_codegen import (
    PIMCmd,
    CodegenContext,
    HostTrigger,
    _bank_fiber_class,
    _bucket_by_work_id,
    _matcher_command_manifest,
    _parse_loop_bound,
    RunResult,
    RUN_KIND_SIMULATOR,
    SimulatorUnavailable,
    Compiled,
)
from .spmw_simenv import pimsim_root as _pimsim_root
from .spmw_simenv import samsung_unavailable_reason as _samsung_unavailable_reason


# FP16 is the only element type of the Samsung PIM datapath (PIMBlock).
_FP16_BITS = 16
_FIBER_OF_CLASS = {"EVEN_BANK": 0, "ODD_BANK": 1}
SAMSUNG_ELEMENTWISE_OPS = frozenset({"ADD", "MUL", "MAD", "RELU"})


class SamsungUnsupported(NotImplementedError):
    """A workload outside the documented Samsung lowering domain."""


# --------------------------------------------------------------------- #
# Target facts
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class SamsungFacts:
    """Per-channel facts of the Samsung target that the lowering relies on."""

    channels: int  # declared pseudo-channels
    pim_blocks: int  # PIM units per pseudo-channel
    banks: int  # banks per pseudo-channel
    banks_per_pim: int  # bank fibers owned by one PIM unit
    grf_a_entries: int
    grf_b_entries: int
    lanes: int  # FP16 lanes per GRF entry
    crf_words: int


def _unit_extent(target, name):
    for unit in target._walk():
        if unit.name == name:
            return math.prod(unit.mapping) if unit.mapping else None
    return None


def samsung_target_facts(target) -> SamsungFacts:
    """Read the lowering's geometry facts off the target tree.

    Raises ValueError naming the missing declaration instead of assuming a
    value (release audit S-09).
    """
    grf_a = getattr(target, "grf_a", None)
    grf_b = getattr(target, "grf_b", None)
    banks = getattr(target, "banks", None)
    crf = getattr(target, "crf", None)
    channels = _unit_extent(target, "pseudo_channel")
    pim_blocks = _unit_extent(target, "pim")
    missing = [
        name
        for name, value in (
            ("register grf_a", grf_a),
            ("register grf_b", grf_b),
            ("memory banks", banks),
            ("memory crf", crf),
            ("unit pseudo_channel", channels),
            ("unit pim", pim_blocks),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            f"target {getattr(target, 'name', None)!r} lacks the Samsung "
            f"declarations {missing}; the lowering derives its geometry from them"
        )
    n_banks = getattr(banks, "banks", None)
    crf_words = (getattr(crf, "geometry", {}) or {}).get("entries")
    if not n_banks or n_banks % pim_blocks:
        raise ValueError(
            f"Samsung banks ({n_banks}) must divide evenly over {pim_blocks} PIM units"
        )
    if not crf_words:
        raise ValueError("Samsung memory 'crf' must declare its word count (entries)")
    if grf_a.width % _FP16_BITS or grf_a.width != grf_b.width:
        raise ValueError(
            "Samsung GRF_A/GRF_B must share a width that holds whole FP16 lanes"
        )
    return SamsungFacts(
        channels=int(channels),
        pim_blocks=int(pim_blocks),
        banks=int(n_banks),
        banks_per_pim=int(n_banks // pim_blocks),
        grf_a_entries=int(grf_a.slots),
        grf_b_entries=int(grf_b.slots),
        lanes=int(grf_a.width // _FP16_BITS),
        crf_words=int(crf_words),
    )


# --------------------------------------------------------------------- #
# Structural operand classification (release audit S-05)
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class SamsungContraction:
    """One work item's contraction, classified by indexing structure.

    The matrix is the input indexed by the output row and the reduction axis;
    the vector is indexed only by the reduction axis (and, for GEMM, by the
    inner result loop that selects a column).  Source operand order does not
    enter.
    """

    matrix_memref: str
    vector_memref: str
    acc_memref: str
    matrix_role: str
    vector_role: str
    reduction_var: str
    row_var: str | None  # None: one output row per work item, no row loop
    column_var: str | None
    K: int  # reduction trip count
    rows_per_item: int
    N: int  # result columns (1 for GEMV)
    matrix_shape: tuple
    vector_shape: tuple
    matrix_k_axis: int  # 1: A[i, k]; 0: A[k, i]
    vector_k_axis: int  # GEMV 0; GEMM: position of k in the vector's indices
    row_offset: int  # first output row of this work item
    output_memref: str
    # "zero" (literal 0 before the reduction) or "carried" (the output's
    # incoming value is the initial accumulator; only the ordered program
    # loads it).
    accumulator_init: str = "zero"

    def structure(self):
        """Everything but the row offset: identical across work items."""
        return (
            self.matrix_memref,
            self.vector_memref,
            self.acc_memref,
            self.matrix_role,
            self.vector_role,
            self.K,
            self.rows_per_item,
            self.N,
            self.matrix_shape,
            self.vector_shape,
            self.matrix_k_axis,
            self.vector_k_axis,
            self.output_memref,
            self.accumulator_init,
        )


_MEMREF_TYPE = re.compile(r"memref<([^>]*)>")


def _memref_dims(type_text):
    """``memref<4096x256xf16>`` -> ((4096, 256), "f16")."""
    m = _MEMREF_TYPE.search(type_text or "")
    if not m:
        return None, None
    parts = m.group(1).split("x")
    try:
        return tuple(int(p) for p in parts[:-1]), parts[-1].split(",")[0].strip()
    except ValueError:
        return None, None


def _split_top_level(text):
    parts, depth, cur = [], 0, ""
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


def _affine_offset(text, var):
    """Offset ``c`` when ``text`` is ``var`` plus a constant ``c``.

    The matcher spells store indices either functionally (``add(c, var)``)
    or infix (``var + c``); both orders are accepted."""
    text = re.sub(r"\s+", "", text)
    if text == var:
        return 0
    v, c = re.escape(var), r"(-?\d+)"
    for pattern in (
        rf"add\({c},{v}\)",
        rf"add\({v},{c}\)",
        rf"{v}\+{c}",
        rf"{c}\+{v}",
    ):
        m = re.fullmatch(pattern, text)
        if m:
            return int(m.group(1))
    return None


def _loop_trip(loop):
    _var, lower, upper, step = loop
    lo, hi = _parse_loop_bound(str(lower)), _parse_loop_bound(str(upper))
    if lo is None or hi is None or lo != 0 or int(step) != 1:
        return None
    return hi


_AFFINE_MAP = re.compile(r"affine_map<\(([^)]*)\)(?:\[[^\]]*\])?\s*->\s*\((.*)\)>")


def _coordinates(operand):
    """Accessed coordinates of an operand, as expressions over loop variables.

    ``memref.load`` operands carry the coordinates directly (``indices``);
    ``affine.load`` operands carry map operands plus the map, whose results
    are the coordinates (``(d0, d1) -> (d0 + 32, d1)`` over ``[%i, %k]`` reads
    ``[%i + 32, %k]``).  Returns None when the map cannot be read."""
    if operand.index_map is None:
        return [str(i) for i in operand.indices]
    m = _AFFINE_MAP.fullmatch(operand.index_map.strip())
    if not m or "[" in operand.index_map:
        return None
    dims = [d.strip() for d in m.group(1).split(",") if d.strip()]
    if len(dims) != len(operand.indices):
        return None
    names = dict(zip(dims, operand.indices))
    results = _split_top_level(m.group(2)) if m.group(2).strip() else []
    return [
        re.sub(r"\bd\d+\b", lambda t: names.get(t.group(0), t.group(0)), r)
        for r in results
    ]


def samsung_contraction(match: MatchedOp) -> SamsungContraction:
    """Classify one MAC match; raise SamsungUnsupported with the reason."""

    def reject(reason):
        raise SamsungUnsupported(
            f"Samsung REDUCE lowering ({match.func_name}): {reason}"
        )

    if match.target_op_name != "MAC":
        reject(f"expected a MAC match, got {match.target_op_name}")
    accumulator_init = match.extra.get("accumulator_init")
    if accumulator_init not in ("zero", "carried"):
        reject(
            "the accumulator must be initialized to literal 0 before the reduction "
            "or carry the output's incoming value "
            f"(accumulator_init={accumulator_init!r})"
        )
    loops = list(match.enclosing_loops)
    if len(loops) not in (1, 2, 3):
        reject(
            "supported nests are [k] (one row per work item), [row, k] (GEMV) and "
            f"[row, column, k] (GEMM); got {len(loops)} enclosing loops"
        )
    trips = [_loop_trip(loop) for loop in loops]
    if any(t is None or t <= 0 for t in trips):
        reject("every enclosing loop needs a static [0, n) range with step 1")
    loop_vars = [loop[0] for loop in loops]
    k = loop_vars[-1]
    roles = match.extra.get("axis_roles") or {}
    if tuple(roles.get("reduction") or ()) != (k,):
        reject(
            "the reduction must be exactly the innermost loop "
            f"(matcher reduction axes {roles.get('reduction')!r})"
        )
    accs = [op for op in match.operands if op.is_loop_carried]
    inputs = [op for op in match.operands if not op.is_loop_carried]
    if len(accs) != 1 or len(inputs) != 2:
        reject("expected one loop-carried accumulator and two inputs")
    if inputs[0].memref_name == inputs[1].memref_name:
        reject(
            f"both inputs read {inputs[0].memref_name!r}; there is no separate "
            "bank-resident matrix and broadcast vector"
        )
    coords = {}
    for op in inputs:
        c = _coordinates(op)
        if c is None:
            reject(f"cannot read the access map {op.index_map!r} of {op.memref_name!r}")
        if [_affine_offset(x, k) for x in c].count(0) != 1:
            reject(
                f"input {op.memref_name!r} must read the reduction loop {k} unshifted "
                f"in exactly one coordinate (coordinates {c})"
            )
        coords[op.role] = c
    free = {
        op.role: [x for x in coords[op.role] if _affine_offset(x, k) != 0]
        for op in inputs
    }
    row_var = loop_vars[0] if len(loops) > 1 else None
    column_var = loop_vars[1] if len(loops) == 3 else None
    if column_var is None:
        vectors = [op for op in inputs if not free[op.role]]
        matrices = [op for op in inputs if len(free[op.role]) == 1]
    else:
        vectors = [op for op in inputs if free[op.role] == [column_var]]
        matrices = [
            op
            for op in inputs
            if len(free[op.role]) == 1 and free[op.role] != [column_var]
        ]
    if len(vectors) != 1 or len(matrices) != 1 or vectors[0] is matrices[0]:
        reject(
            "cannot tell the matrix from the vector: the matrix must be indexed by "
            "the output row and the reduction loop, the vector only by the reduction "
            "loop"
            + (" and the column loop" if column_var else "")
            + f" (coordinates: {[(op.memref_name, coords[op.role]) for op in inputs]})"
        )
    matrix, vector = matrices[0], vectors[0]
    from .spmw_match_engine import _render_form, output_map_problem

    # Output map (C-23): the conductor writes each result at the matrix row
    # it reads, so the store coordinates must be exactly the operands' row
    # coordinates as the matcher resolved them (`store_coords`, operand
    # ``coords``); index text is never trusted.
    problem = output_map_problem(match)
    if problem is not None:
        reject(f"the output map is not the identity the lowering realizes: {problem}")
    (matrix_row_position,) = [
        i for i, x in enumerate(coords[matrix.role]) if _affine_offset(x, k) != 0
    ]
    # A scalar accumulator reaches the output through its single copy, whose
    # resolved coordinates the coverage analysis stamps as
    # ``output_sink_coords``; ``y[i] += ...`` stores the output directly.
    store_coords = match.extra.get("store_coords")
    sink_coords = match.extra.get("output_sink_coords") or []
    if not store_coords and len(sink_coords) == 1:
        store_coords = sink_coords[0]
    if (
        matrix.coords is None
        or matrix.coords[matrix_row_position] is None
        or not store_coords
        or any(c is None for c in store_coords)
    ):
        reject("the matrix row or store coordinates were not resolved by the matcher")
    matrix_row_form = matrix.coords[matrix_row_position]
    m_dims, m_elt = _memref_dims(matrix.memref_type)
    v_dims, v_elt = _memref_dims(vector.memref_type)
    _a_dims, a_elt = _memref_dims(accs[0].memref_type)
    if not m_dims or len(m_dims) != 2 or not v_dims:
        reject("operand memref types with static extents are required")
    if {m_elt, v_elt, a_elt} != {"f16"}:
        reject(
            f"the PIM datapath is FP16 only; got matrix {m_elt}, vector {v_elt}, "
            f"accumulator {a_elt}"
        )
    matrix_k_axis = [_affine_offset(x, k) for x in coords[matrix.role]].index(0)
    vector_k_axis = [_affine_offset(x, k) for x in coords[vector.role]].index(0)
    K = trips[-1]
    if K > m_dims[matrix_k_axis] or K > v_dims[vector_k_axis]:
        reject(f"reduction loop {K} exceeds the operand extents")
    sinks = match.extra.get("output_sinks") or []
    if len(sinks) != 1:
        reject(f"expected one store of the accumulator, got {sinks!r}")
    output_memref, _sink_text = sinks[0]
    if len(store_coords) != (1 if column_var is None else 2):
        reject(
            f"store index [{', '.join(_render_form(c) for c in store_coords)}] "
            "does not match the loop nest"
        )
    row_form = store_coords[0]
    atoms = dict(row_form[0])
    if row_var is None:
        row_offset = row_form[1] if not atoms else None
    else:
        row_offset = row_form[1] if atoms == {row_var: 1} else None
    if row_offset is None or row_offset < 0:
        reject(
            f"store row index {_render_form(row_form)} is not the row loop "
            f"({row_var or 'none'}) plus a non-negative constant"
        )
    if matrix_row_form != row_form:
        reject(
            f"the matrix row coordinate {_render_form(matrix_row_form)} and the "
            f"store row {_render_form(row_form)} differ"
        )
    if column_var is not None and store_coords[1] != (((column_var, 1),), 0):
        reject(
            f"store column index {_render_form(store_coords[1])} is not the column "
            f"loop {column_var}"
        )
    return SamsungContraction(
        matrix_memref=matrix.memref_name,
        vector_memref=vector.memref_name,
        acc_memref=accs[0].memref_name,
        matrix_role=matrix.role,
        vector_role=vector.role,
        reduction_var=k,
        row_var=row_var,
        column_var=column_var,
        K=K,
        rows_per_item=trips[0] if row_var is not None else 1,
        N=trips[1] if column_var is not None else 1,
        matrix_shape=tuple(m_dims),
        vector_shape=tuple(v_dims),
        matrix_k_axis=matrix_k_axis,
        vector_k_axis=vector_k_axis,
        row_offset=row_offset,
        output_memref=output_memref,
        accumulator_init=accumulator_init,
    )


def samsung_contraction_roles(matches) -> SamsungContraction:
    """Classify every MAC of one work item group; all must agree structurally."""
    macs = [m for m in matches if m.target_op_name == "MAC"]
    if not macs:
        raise SamsungUnsupported("Samsung REDUCE lowering: no MAC in the work item")
    by_item = {}
    for m in macs:
        by_item.setdefault(tuple(m.work_id), []).append(m)
    if any(len(v) != 1 for v in by_item.values()):
        raise SamsungUnsupported(
            "Samsung REDUCE lowering supports one contraction per work item"
        )
    classified = [samsung_contraction(m) for m in macs]
    first = classified[0]
    if any(c.structure() != first.structure() for c in classified[1:]):
        raise SamsungUnsupported(
            "Samsung REDUCE lowering: work items differ in contraction structure"
        )
    return first


@dataclass(frozen=True)
class SamsungElementwise:
    """One elementwise op over a 1-D FP16 extent, proven elementwise."""

    op: str
    input_memrefs: tuple  # in fn-signature role order (x, y[, z])
    input_roles: tuple
    output_memref: str
    n: int
    # Buffers the source copies the result into element for element
    # (accepted copies, C-24); the runtime writes the result to each.
    sink_memrefs: tuple = ()


def samsung_elementwise(matches) -> SamsungElementwise:
    """Classify an elementwise trace; raise SamsungUnsupported with the reason.

    Domain: one work item whose single loop walks every element of 1-D FP16
    operands of one extent, every input and the store indexed by that loop
    variable.  Partitioned forms index through opaque expressions the matcher
    does not retain, so elementwise-ness could not be proven; they are
    rejected rather than assumed.
    """
    if len(matches) != 1:
        raise SamsungUnsupported(
            f"Samsung elementwise lowering needs one work item whose loop walks the "
            f"whole extent (got {len(matches)} matches); a partitioned work grid "
            "indexes elements through expressions the lowering cannot prove "
            "elementwise"
        )
    (m,) = matches
    op = m.target_op_name
    if op not in SAMSUNG_ELEMENTWISE_OPS:
        raise SamsungUnsupported(f"Samsung elementwise lowering: unsupported op {op}")

    def reject(reason):
        raise SamsungUnsupported(f"Samsung elementwise {op} ({m.func_name}): {reason}")

    if len(m.enclosing_loops) != 1:
        reject("expected exactly one enclosing loop over the elements")
    n = _loop_trip(m.enclosing_loops[0])
    var = m.enclosing_loops[0][0]
    if n is None:
        reject("the element loop needs a static [0, n) range with step 1")
    from .spmw_match_engine import _render_form, coordinate_forms, output_map_problem

    element = coordinate_forms(var)  # the resolved coordinate ``var``
    inputs = [o for o in m.operands if not o.is_loop_carried]
    if any(o.is_loop_carried for o in m.operands):
        reject("an elementwise op cannot carry an accumulator")
    for o in inputs:
        dims, elt = _memref_dims(o.memref_type)
        # Resolved coordinates (C-23/C-24), never index text: the input is
        # read at exactly the element the loop variable names.
        if o.coords is None or tuple(o.coords) != element:
            shown = (
                [_render_form(c) if c else "?" for c in o.coords] if o.coords else "?"
            )
            reject(
                f"input {o.memref_name!r} is read at coordinate {shown}, not at the "
                f"element loop {var}; the lowering walks one element per iteration "
                "(a partitioned work grid, a shift or a broadcast is not elementwise)"
            )
        if dims != (n,) or elt != "f16":
            reject(
                f"input {o.memref_name!r} is a {elt} buffer of shape {dims} but the "
                f"element loop walks {n} elements; the lowering needs one work item "
                "whose loop walks the whole 1-D FP16 extent (a partitioned work grid "
                "is not supported)"
            )
    store_coords = m.extra.get("store_coords")
    if (
        m.result_memref_name is None
        or store_coords is None
        or tuple(store_coords) != element
    ):
        shown = (
            [_render_form(c) if c else "?" for c in store_coords]
            if store_coords
            else "?"
        )
        reject(
            f"the result is stored at coordinate {shown}, not at the element loop "
            f"index {var}; the lowering writes element i of the output from element "
            "i of the inputs, so a reversed, shifted or broadcast destination is "
            "not realized"
        )
    problem = output_map_problem(m)
    if problem is not None:
        reject(f"the output map is not the identity the lowering realizes: {problem}")
    out_dims, out_elt = _memref_dims(m.extra.get("result_memref_type"))
    if out_dims != (n,) or out_elt not in ("f16", "f32"):
        # f32: the FP16 result is widened exactly on readback (the source's
        # ``y_f32[i] = a[i] * b[i]`` multiplies in FP16 and widens the result).
        reject(
            f"the output {m.result_memref_name!r} is a {out_elt} buffer of shape "
            f"{out_dims} but the element loop writes {n} FP16 results; the output "
            "must be a 1-D FP16 (or FP32, widened on readback) buffer of that extent"
        )
    if len({o.memref_name for o in inputs}) != len(inputs):
        reject("one buffer bound to several inputs is not supported")
    # Copies the matcher accepted are element-for-element copies of the result
    # (`_copy_contract_problem`); the runtime writes the result to each sink.
    sinks = tuple(memref for memref, _text in m.extra.get("output_sinks", ()))
    return SamsungElementwise(
        op=op,
        input_memrefs=tuple(o.memref_name for o in inputs),
        input_roles=tuple(o.role for o in inputs),
        output_memref=m.result_memref_name,
        n=n,
        sink_memrefs=sinks,
    )


# --------------------------------------------------------------------- #
# The common candidate contract of a Samsung placement (gate G9)
# --------------------------------------------------------------------- #


def _storage_name(handle) -> str | None:
    """The declared storage a placement handle lives in (``grf_a``,
    ``grf_b``, ``banks``)."""
    home = getattr(handle, "home_handle", handle)
    if isinstance(home, MemoryRef):
        return getattr(home.memory, "name", None)
    return getattr(home, "name", None)


def _axes_of(access, labels) -> tuple:
    """Source loop labels the access coordinates range over (the shared
    `spmw_candidate.value_axes`)."""
    from .spmw_candidate import value_axes

    return value_axes(access, labels)


def _elementwise_register_plan(target):
    """Which register each elementwise input is loaded into and whether the
    MAD addend's fiber is the opposite one (the emitter's `_register_moves`
    logic over the target's declared moves): ``(dest registers by fiber,
    addend_swapped)``."""
    ctx = SamsungCtx(target)
    loads, drains = ctx._register_moves()
    swapped = None
    for fiber, (_st, dest) in sorted(drains.items()):
        other = target.grf_b if dest is target.grf_a else target.grf_a
        _name, o_fiber = loads.get(other.name, (None, None))
        if o_fiber is not None:
            swapped = (o_fiber != fiber) if swapped is None else swapped
    return bool(swapped)


def _output_row_packing(target, matches, placement) -> str:
    """``row_slots:movement`` when the placement carries a movement layout
    (the result rows occupy the slots that layout assigns, identity or a
    permutation the group-level owner map decides), ``row_slots:identity``
    when it carries none (rows in row order).  The realization check
    confirms the program's row map was built from that layout."""
    del target, matches
    from .spmw_layout_problem import split_movement_layout

    try:
        _backend, movement = split_movement_layout(getattr(placement, "layout", None))
    except Exception:  # pylint: disable=broad-exception-caught
        movement = None
    return "row_slots:movement" if movement is not None else "row_slots:identity"


def samsung_candidate_effects(target, matches, placement):
    """`CandidateEffects` of one Samsung placement for one work item, derived
    from the shared semantic facts of its matches (`Contraction`/`Epilogue`:
    loop labels, trips, access coordinates, accumulator initialization,
    sinks) and from the placement (handles and levers).

    Backing storage is where the conductor keeps a value between launches
    (the DRAM ``banks`` for every matrix, every elementwise operand and
    every result; ``grf_a`` for a host-broadcast vector); transient register
    use is declared as allocations (the accumulator's ``grf_b``, the tile
    registers of the elementwise program); packing names the bank image
    (``row_slots:identity``/``row_slots:movement`` for results of the REDUCE
    programs, ``all_bank`` / ``all_bank:swapped_fibers`` for elementwise
    operands); transfers are the conductor's scatter, broadcast and gather
    steps.  `SamsungCtx.check_realization` compares all of these with the
    emitted program.  None when the matches carry no facts (a trace
    assembled outside the shared front stages)."""
    from .spmw_candidate import (
        CandidateCoverage,
        CandidateEffects,
        CandidateLaunch,
        CandidateTransfer,
        CandidateValueLayout,
    )

    facts = [m.extra.get("semantic_facts") for m in matches]
    if not matches or any(f is None for f in facts):
        return None
    extra = placement.extra or {}
    residency = extra.get("grf_residency", {}) or {}
    ops = {m.target_op_name for m in matches}
    contraction = ops == {"MAC"}
    temporal = extra.get("reduction_program") == "temporal"
    labels = {}
    for fact in facts:
        labels.update(fact.loop_labels)
    inputs, outputs, layouts, transfers, allocations = [], [], {}, {}, []
    instances = reduction = None
    init = None
    vector_memref = None
    for match, fact in zip(matches, facts):
        if hasattr(fact, "reduction_vars"):
            trips = fact.trips
            extent = 1
            for var in fact.result_vars:
                extent *= int(trips.get(var) or 0)
            red = 1
            for var in fact.reduction_vars:
                red *= int(trips.get(var) or 0)
            instances = extent if instances is None else instances
            reduction = red if reduction is None else reduction
            init = fact.accumulator_init
            accesses = [(a.memref_name, a) for a in fact.inputs]
            if fact.vector is not None:
                vector_memref = fact.vector.memref_name
            result_accesses = []
            if not fact.accumulator.local:
                result_accesses.append((fact.accumulator.memref_name, fact.accumulator))
            for sink in fact.external_sinks:
                result_accesses.append((sink.memref_name, sink))
        else:
            extent = 1
            for loop in fact.loops:
                extent *= int(loop.trip or 0)
            instances = extent if instances is None else instances
            # An operand reading the element it writes with a proven zero
            # initializer (``accumulator_init == "zero"``) has dead incoming
            # contents: not an input (the shared coverage rule).
            accesses = [
                (a.memref_name, a)
                for a in fact.operands
                if not a.local
                and not (
                    a.memref_name == fact.result.memref_name
                    and match.extra.get("accumulator_init") == "zero"
                )
            ]
            result_accesses = (
                [] if fact.result.local else [(fact.result.memref_name, fact.result)]
            )
            for memref, _text in match.extra.get("output_sinks", ()):
                if memref != fact.result.memref_name:
                    result_accesses.append((memref, fact.result))
        for position, (memref, access) in enumerate(accesses):
            if memref is None or memref in inputs:
                continue
            inputs.append(memref)
            axes = _axes_of(access, labels)
            if contraction:
                handle = placement.placements.get(memref)
                storage = _storage_name(handle)
                if storage == "grf_a" and (
                    residency.get(memref, "crf") != "host" and not temporal
                ):
                    # CRF fill: the vector is scattered to the banks and loaded
                    # into GRF_A per reduction tile (LD_A).
                    storage = "banks"
                packing = None
                if contraction and memref == getattr(fact.matrix, "memref_name", None) and not temporal:
                    # The matrix's row packing in the banks is the selected
                    # half -> bank map (vendor: both rows in the tile's fiber).
                    try:
                        packing = samsung_matrix_packing(placement)
                    except ValueError as error:
                        packing = f"invalid:{error}"
                layouts[(memref, "value")] = CandidateValueLayout(
                    memref, axes, storage, packing
                )
                transfers[memref] = CandidateTransfer(
                    memref,
                    "in",
                    (
                        ("host_broadcast", "grf_a")
                        if storage == "grf_a"
                        else ("host_scatter", "banks")
                    ),
                )
            else:
                # Every elementwise operand is a flat all-bank image; the MAD
                # addend is stored with the fibers exchanged when its register
                # loads from the opposite fiber.
                packing = "all_bank"
                if match.target_op_name == "MAD" and position == 2:
                    if _elementwise_register_plan(target):
                        packing = "all_bank:swapped_fibers"
                layouts[(memref, "in")] = CandidateValueLayout(
                    memref, axes, "banks", packing, image="in"
                )
                transfers[memref] = CandidateTransfer(
                    memref, "in", ("host_scatter", "banks")
                )
        for memref, access in result_accesses:
            if memref is None:
                continue
            if memref not in outputs:
                outputs.append(memref)
            axes = _axes_of(access, labels)
            carried = contraction and init == "carried"
            if carried and memref not in inputs:
                # The destination's incoming contents seed the reduction.
                inputs.append(memref)
            packing = (
                _output_row_packing(target, matches, placement)
                if contraction
                else "all_bank"
            )
            # An elementwise operand that is also written (a copy sink of the
            # result) has a distinct output image: its input image keeps its
            # own packing (the MAD addend's swapped fibers).
            image = "out" if (not contraction and memref in inputs) else "value"
            layouts[(memref, image)] = CandidateValueLayout(
                memref, axes, "banks", packing, image=image
            )
            transfers[memref] = CandidateTransfer(
                memref,
                "inout" if carried or memref in inputs else "out",
                (
                    ("host_scatter", "banks", "host_gather")
                    if (carried or memref in inputs)
                    else ("banks", "host_gather")
                ),
            )
    if contraction:
        kind = "TEMPORAL_REDUCE" if temporal else "REDUCE"
        allocations = [("accumulator", 0, "grf_b"), ("vector_tile", 0, "grf_a")]
    else:
        kind = "ELTWISE"
        allocations = [("tile", 0, "grf_a")]
        if ops == {"MAD"}:
            allocations.append(("addend_tile", 0, "grf_b"))
    launch = CandidateLaunch(
        index=0,
        kind=kind,
        operations=tuple(range(len(matches))),
        instances=(0, int(instances or 0)),
        reduction=(0, int(reduction or 0)) if contraction else None,
        accumulator=init if contraction else None,
        covers=(placement.mode,),
    )
    del vector_memref
    return CandidateEffects(
        coverage=CandidateCoverage(
            matches[0].func_name, tuple(sorted(ops)), tuple(inputs), tuple(outputs)
        ),
        value_layouts=tuple(layouts.values()),
        transfers=tuple(transfers.values()),
        launches=(launch,),
        abi_inputs=tuple(inputs),
        abi_outputs=tuple(outputs),
        device_scope="one CRF per kernel group, every PIM unit of a pseudo-channel in lockstep",
        allocations=tuple(allocations),
    )


def _derive_samsung_effects(view, matches, placement):
    """Late-binding entry of the effects registry: resolves
    `samsung_candidate_effects` at call time, so the enumerator, the
    movement-stage re-declaration and a supplied-placement rebinding all
    use the same (possibly instrumented) derivation."""
    return samsung_candidate_effects(view, matches, placement)


def _register_effects_deriver():
    from .spmw_candidate import EFFECTS_DERIVERS

    EFFECTS_DERIVERS["samsung_hbm_pim"] = _derive_samsung_effects


_register_effects_deriver()


# --------------------------------------------------------------------- #
# Codegen context
# --------------------------------------------------------------------- #


@dataclass
class SamsungProgram:
    """One lowered kernel: CRF word range plus what the conductor needs."""

    kind: str  # "REDUCE" | "ELTWISE"
    group_id: int
    start: int
    end: int
    # (pc relative to start, binding kind, bank fiber, operand, column offset)
    binds: tuple
    facts: SamsungFacts
    crf_issue: str
    work_items: int
    contraction: SamsungContraction | None = None
    rows: int = 0  # logical output rows covered by the work items
    k_tiles: int = 0
    tile_fiber: tuple = ()
    rows_per_block: int = 0
    vector_from_bank: bool = False
    weight_resident: bool = True
    elementwise: SamsungElementwise | None = None
    swap_parity: tuple = ()
    lever_record: dict = field(default_factory=dict)
    # Slot of each output row (row -> slot) when the selected owner map is not
    # the row-order packing; None means rows occupy slots in row order.
    row_slots: tuple | None = None
    row_map: str = "identity"
    # Issue order of a tile's MAC groups: "tile_major" (the conductor's
    # default, one barrier per group) or "row_pair" (group g and g + G/2 of
    # one tile, its two physical rows, under one barrier).
    issue_order: str = "tile_major"
    # The fiber of each of a tile's two physical rows, from the selected
    # half -> bank map: (0, 0) "same_bank", (0, 1) "split_fibers".
    half_fibers: tuple = (0, 0)
    tile_rows: str = "same_bank"


ISSUE_ORDERS = ("tile_major", "row_pair")
TILE_ROW_PLACEMENTS = ("same_bank", "split_fibers")


def samsung_half_fibers(placement) -> tuple:
    """The bank fiber of each of a tile's two physical rows, read from the
    placement's selected ``half_bank_map`` (a ``LinearLayout`` over
    ``{bank, half}`` into ``(bank,)``): ``(0, 0)`` keeps both rows in the
    tile's fiber, ``(0, 1)`` stages the second row in the odd fiber.  A
    placement without a map is the vendor layout ``(0, 0)``.  The map is
    the production source of this choice: the packing the candidate
    declares, the program the lowering emits and the plan the conductor
    executes all derive from it, and the realization check compares them.
    """

    extra = getattr(placement, "extra", None) or {}
    layout = extra.get("half_bank_map")
    if layout is None:
        return (0, 0)
    if tuple(layout.out_dims) != ("bank",) or tuple(layout.out_sizes) != (2,):
        raise ValueError(
            f"half_bank_map must map into the two fibers (bank,) of size 2, got "
            f"{tuple(layout.out_dims)} {tuple(layout.out_sizes)}"
        )
    if sorted(layout.bases) != ["bank", "half"] or layout.size_of("half") != 2:
        raise ValueError("half_bank_map takes the inputs {bank, half} with two halves")
    bank_index = layout.out_dims.index("bank")
    # The lowering realizes only the half -> fiber choice: a tile's base
    # fiber is the bank input itself (the bank basis must be the identity;
    # a map that permutes or collapses the bank input describes something
    # the lowering does not do and is refused).
    for bank in range(layout.size_of("bank")):
        if int(layout.apply(bank=bank, half=0)[bank_index]) != bank:
            raise ValueError(
                "half_bank_map must keep the bank input as the tile's base fiber "
                f"(bank {bank}, half 0 maps to fiber "
                f"{int(layout.apply(bank=bank, half=0)[bank_index])})"
            )
    halves = []
    for half in (0, 1):
        try:
            physical = layout.apply(bank=0, half=half)
        except Exception as error:
            raise ValueError(f"half_bank_map cannot place half {half}: {error}")
        bank = int(physical[bank_index])
        if bank not in (0, 1):
            raise ValueError(f"half_bank_map places half {half} in bank {bank}, not a fiber")
        halves.append(bank)
    return tuple(halves)


def samsung_matrix_packing(placement) -> str:
    """The matrix packing a candidate declares, named from its half -> bank
    map: ``half_bank:0,0`` (both rows in the tile's fiber) or
    ``half_bank:0,1`` (second row in the odd fiber)."""

    h0, h1 = samsung_half_fibers(placement)
    return f"half_bank:{h0},{h1}"


def _infeasible(message):
    from .pim.schedule_search import InfeasibleSchedule

    return InfeasibleSchedule(f"Samsung: {message}")


class SamsungCtx(CodegenContext):
    """Codegen context for Samsung HBM-PIM.

    Translates Tenon `Register` / `MemoryRef` handles into PIMSimulator
    `PIMOpdType` names and lowers each kernel to a per-pass CRF program
    (`emit_program`).  Every instruction is produced by the target's own
    op/move `emit` callbacks; this class decides which ones, in which order,
    and the loop counts, from the placement and the target facts.
    """

    _REG_TO_OPD = {"grf_a": "GRF_A", "grf_b": "GRF_B"}

    @staticmethod
    def realized_effects(trace):
        """Destination refs the Samsung lowerings write: each match's own
        destination and every accepted copy sink (the REDUCE programs write
        the contraction's sink, the elementwise program writes its result and
        every sink element for element).  `require_effects_realized` compares
        this with the source's external effects."""
        realized = set()
        for match in trace.matches:
            if match.result_value_ref is not None:
                realized.add(match.result_value_ref)
            for ref, _local in match.extra.get("output_sink_refs", ()):
                if ref is not None:
                    realized.add(ref)
        return realized

    def __init__(self, target):
        super().__init__(target)
        # Host-move bookkeeping: the host-scope moves' `emit` closures append
        # `(hook, device_handle)` records here. The conductor performs the
        # transfers; these records carry cost and operand-role information.
        self.host_moves_emitted: list[tuple[str, object]] = []
        self.programs: list[SamsungProgram] = []
        # id(layout) -> kernel group of the program it was lowered into
        # (`emit_program`), read by `check_realization`.
        self._group_of_layout: dict = {}

    # ------------------------------------------------------------------ #
    # Host-transfer hooks (no compute PIMCmd)
    # ------------------------------------------------------------------ #

    def drain(self, dst=None, src0=None):
        """GRF->bank writeback as a NOP hold; the conductor issues the WRITE
        column commands (ISA 1.0 forbids MOV bank<-GRF, PIMCmd.cpp:73-97).
        The hold covers every GRF entry, one write each."""
        entries = getattr(src0, "slots", None) or self.target.grf_b.slots
        self.cmds.append(PIMCmd(type_="NOP", loopCounter_=int(entries) - 1))

    def program_crf(self, crf_handle):
        """Record the CRF upload (no compute PIMCmd)."""
        self.host_moves_emitted.append(("program_crf", crf_handle))

    def host_broadcast(self, handle):
        """Record a host->device broadcast (GRF_A/GRF_B/SRF)."""
        self.host_moves_emitted.append(("broadcast", handle))

    def host_scatter(self, handle):
        """Record a host->device per-bank scatter."""
        self.host_moves_emitted.append(("scatter", handle))

    def host_gather(self, handle):
        """Record a device->host gather."""
        self.host_moves_emitted.append(("gather", handle))

    # ------------------------------------------------------------------ #
    # Operand encoding
    # ------------------------------------------------------------------ #

    def _opd(self, handle):
        """Return (opd_name, idx) for a Tenon handle, or ("A_OUT", 0)."""
        if handle is None:
            return ("A_OUT", 0)
        if isinstance(handle, Register):
            opd = self._REG_TO_OPD.get(handle.name)
            if opd is None:
                raise NotImplementedError(
                    f"SamsungCtx: register {handle.name!r} has no PIMOpdType "
                    "mapping. Only grf_a/grf_b are supported."
                )
            return (opd, 0)
        if isinstance(handle, MemoryRef):
            cls = _bank_fiber_class(handle.idx)
            if cls is None:
                raise NotImplementedError(
                    f"SamsungCtx: memref index {handle.idx!r} is not the canonical "
                    "stride*pid + r fiber form."
                )
            return (cls, 0)
        raise NotImplementedError(
            f"SamsungCtx: unknown handle type {type(handle).__name__}."
        )

    def cmd(self, name, dst=None, src0=None, src1=None, src2=None, is_relu=0):
        dst_name, dst_idx = self._opd(dst)
        src0_name, src0_idx = self._opd(src0)
        src1_name, src1_idx = self._opd(src1)
        src2_name, _src2_idx = self._opd(src2)
        # Auto mode: an arithmetic op that streams a bank operand walks the
        # GRF entries by column address (PIMRank::readOpd).  FILL is always
        # auto in hardware and carries no flag.
        is_auto = int(
            name in ("MAC", "ADD", "MUL", "MAD")
            and any(isinstance(s, MemoryRef) for s in (src0, src1, src2))
        )
        self.cmds.append(
            PIMCmd(
                type_=name,
                dst_=dst_name,
                dstIdx_=dst_idx,
                src0_=src0_name,
                src0Idx_=src0_idx,
                src1_=src1_name,
                src1Idx_=src1_idx,
                src2_=src2_name,
                isAuto_=is_auto,
                isRelu_=is_relu,
            )
        )

    # Read-only roles emit LD only; read-write roles emit LD + ST.
    _ROLE_IS_READ_ONLY = {"x": True, "y": True, "acc": False, "dst": False}

    def resolve_moves(self, role, src_handle=None, dst_handle=None):
        if isinstance(dst_handle, MemoryRef):
            return (None, None)
        if isinstance(dst_handle, Register):
            if dst_handle.name == "grf_a":
                ld, st = "LD_A", "ST_A"
            elif dst_handle.name == "grf_b":
                ld, st = "LD_B", "ST_B"
            else:
                return (None, None)
            if self._ROLE_IS_READ_ONLY.get(role, False):
                return (ld, None)
            if role == "acc":
                return (None, st)
            return (ld, st)
        return (None, None)

    def resolve_spill_moves(self, tier, home_handle, *, n_entries=1):
        if tier != "bank_row":
            return super().resolve_spill_moves(tier, home_handle, n_entries=n_entries)
        if isinstance(home_handle, Register) and home_handle.name == "grf_b":
            return ("LD_B", "ST_B")
        return ("LD_A", "ST_A")

    # ------------------------------------------------------------------ #
    # Program lowering
    # ------------------------------------------------------------------ #

    def _emit_move(self, name):
        move = self.target.move(name)
        if move.emit is None:
            raise NotImplementedError(f"Samsung move {name!r} has no emit callback")
        move.emit(self)

    def _emit_jump(self, loop_counter, loop_offset):
        """JUMP back `loop_offset - 1` words, taken `loop_counter` times."""
        if loop_counter <= 0:
            return
        self._emit_move("JUMP")
        self.cmds[-1].loopCounter_ = int(loop_counter)
        self.cmds[-1].loopOffset_ = int(loop_offset)

    def emit_program(
        self, trace, layouts, *, schedule=None, host_moves=(), module=None
    ):
        from .spmw_autoschedule import (
            _bucket_for_autoschedule,
            _matcher_placement_decision,
            _matcher_search_scope,
            _matcher_work_scope,
            derive_layout_properties,
        )
        from .spmw_match import MatchTrace

        layout_by_scope = {
            scope: layout
            for (scope, _matches), layout in zip(
                _bucket_for_autoschedule(trace), layouts
            )
        }
        groups: dict = {}
        for work_id, matches in _bucket_by_work_id(trace):
            group_id = _matcher_work_scope(matches[0]).group_id
            groups.setdefault(group_id, []).append((work_id, matches))

        for group_id, buckets in groups.items():
            group_layouts = [
                layout_by_scope[_matcher_search_scope(matches[0])]
                for _work_id, matches in buckets
            ]
            for layout in group_layouts:
                layout.extra = derive_layout_properties(self.target, layout)
                self._group_of_layout[id(layout)] = group_id
            issues = {
                (layout.extra or {}).get("crf_issue", "per_workid")
                for layout in group_layouts
            }
            if not issues <= {"shared", "per_workid"} or len(issues) != 1:
                raise _infeasible(
                    f"matcher group {group_id} has CRF issue decisions {sorted(issues)}"
                )
            (issue,) = issues
            # The PIM blocks of a pseudo-channel execute one CRF in lockstep, so
            # every work item of the kernel runs the same program.
            keys = {
                _matcher_placement_decision(
                    MatchTrace(trace.target_name, trace.module_name, matches=matches),
                    layout,
                )
                for (_w, matches), layout in zip(buckets, group_layouts)
            }
            if len(keys) != 1:
                raise _infeasible(
                    f"matcher group {group_id} assigns distinct placements to work "
                    "items that share one lockstep CRF"
                )
            if issue == "per_workid" and len(buckets) > 1:
                raise _infeasible(
                    f"crf_issue=per_workid concatenates {len(buckets)} per-work-item "
                    "bodies into one CRF, but the 8 PIM blocks of a pseudo-channel "
                    "execute that CRF in lockstep (PIMRank::doPIM), so every block "
                    "would run every body; only a one-work-item kernel can issue "
                    "its CRF per work item"
                )
            ops = {m.target_op_name for _w, matches in buckets for m in matches}
            if ops == {"MAC"} and _is_temporal(group_layouts[0]):
                program = self._emit_temporal(
                    group_id, buckets, group_layouts[0], issue
                )
            elif ops == {"MAC"}:
                program = self._emit_reduce(group_id, buckets, group_layouts[0], issue)
            elif len(ops) == 1 and ops <= SAMSUNG_ELEMENTWISE_OPS:
                program = self._emit_eltwise(group_id, buckets, group_layouts[0], issue)
            else:
                raise SamsungUnsupported(
                    f"Samsung lowering: kernel group {group_id} mixes ops {sorted(ops)}; "
                    "a kernel is one contraction or one elementwise op"
                )
            self.programs.append(program)
            if issue == "shared":
                for work_id, matches in buckets:
                    self.host_schedule.append(
                        HostTrigger(
                            work_id,
                            sum(1 for m in matches if m.target_op_name == "MAC"),
                        )
                    )

    def _check_lever_realization(self, layout, program) -> None:
        """The emitted program's issue order must be the lever the candidate
        declared, and its tile-row placement (the fiber of each physical
        row) must be the matrix packing the candidate's shared contract
        declares from its half -> bank map."""
        from .spmw_candidate import candidate_effects

        if program.kind != "REDUCE":
            return
        extra = layout.extra or {}
        declared_order = extra.get("issue_order", "tile_major")
        if declared_order != getattr(program, "issue_order", "tile_major"):
            raise RuntimeError(
                "Samsung realization issues the MAC groups as "
                f"{program.issue_order!r} but the candidate declared {declared_order!r}"
            )
        effects = candidate_effects(layout)
        rep = program.contraction
        if effects is None or rep is None:
            return
        declared = None
        for value_layout in effects.value_layouts:
            if value_layout.value == rep.matrix_memref and value_layout.storage == "banks":
                declared = value_layout.packing
                break
        realized = "half_bank:{},{}".format(*(program.half_fibers or (0, 0)))
        if declared != realized:
            raise RuntimeError(
                f"Samsung realization stages the matrix rows as {realized!r}; the "
                f"candidate declared packing {declared!r}"
            )

    def check_realization(self, layouts) -> None:
        """The lowered CRF programs against the selected candidates' declared
        effects, per kernel group: the program kind; for a contraction the
        reduction extent, the result instances per work item, the
        accumulator initialization, the vector's residency, the result's row
        packing and the carried destination's ingress; for an elementwise
        program every operand's bank residency and fiber packing (the MAD
        addend's swap parity); for both the conductor's transfers (scatter,
        broadcast, gather) and the transient registers the program uses."""
        from .spmw_candidate import candidate_effects

        layouts = list(layouts) if isinstance(layouts, (list, tuple)) else [layouts]
        by_group = {}
        for program in self.programs:
            by_group.setdefault(program.group_id, program)
        for layout in layouts:
            effects = candidate_effects(layout)
            if effects is None:
                continue
            program = by_group.get(self._group_of_layout.get(id(layout)))
            if program is None:
                raise RuntimeError(
                    f"Samsung realization lowered no program for the candidate "
                    f"{layout.mode!r} that declares effects"
                )
            self._check_lever_realization(layout, program)
            if len(effects.launches) != 1 or effects.launches[0].kind != program.kind:
                raise RuntimeError(
                    f"Samsung realization is a {program.kind} program; the candidate "
                    f"declares launches {[l.kind for l in effects.launches]}"
                )
            (launch,) = effects.launches
            images = {}
            for v in effects.value_layouts:
                images.setdefault(v.value, {})[v.image] = v

            def layout_of(value, side):
                """The layout of ``value``'s ``side`` (``in``/``out``) image:
                the side's own image when declared, else the single image."""
                candidates = images.get(value, {})
                return candidates.get(side) or candidates.get("value")

            transfers_by_value = {t.value: t for t in effects.transfers}
            allocated = {storage for _name, _size, storage in effects.allocations}
            expected_transfers = {}
            if program.contraction is not None:
                rep = program.contraction
                declared = (
                    tuple(launch.instances),
                    tuple(launch.reduction or ()),
                    launch.accumulator,
                )
                realized = (
                    (0, rep.rows_per_item * rep.N),
                    (0, rep.K),
                    rep.accumulator_init,
                )
                if declared != realized:
                    raise RuntimeError(
                        "Samsung realization launch differs from the candidate "
                        f"contract: realized (instances, reduction, accumulator) "
                        f"{realized}, declared {declared}"
                    )
                matrix = layout_of(rep.matrix_memref, "in")
                if matrix is None or matrix.storage != "banks":
                    raise RuntimeError(
                        f"Samsung realization keeps the matrix {rep.matrix_memref!r} "
                        f"in the banks; the candidate declared {matrix}"
                    )
                vector = layout_of(rep.vector_memref, "in")
                vector_banks = bool(program.vector_from_bank)
                vector_storage = "banks" if vector_banks else "grf_a"
                vector_transfer = transfers_by_value.get(rep.vector_memref)
                if vector is None or vector.storage != vector_storage:
                    raise RuntimeError(
                        "Samsung realization loads the vector from "
                        f"{'the banks' if vector_banks else 'a GRF_A broadcast'} "
                        f"({vector_storage}); the candidate declared {vector}"
                    )
                if vector_transfer is None or (
                    tuple(vector_transfer.route)[-1] != vector_storage
                ):
                    raise RuntimeError(
                        f"Samsung realization moves the vector into {vector_storage}; "
                        f"the candidate's transfer ends at "
                        f"{vector_transfer and vector_transfer.route}"
                    )
                output = layout_of(rep.output_memref, "out")
                from .spmw_layout_problem import split_movement_layout

                _backend, movement = split_movement_layout(
                    getattr(layout, "layout", None)
                )
                # The program's row map is derived from the placement's
                # movement layout (`_row_slots`); without one the rows are in
                # row order.
                row_packing = (
                    "row_slots:movement"
                    if movement is not None
                    else "row_slots:identity"
                )
                if movement is None and program.row_map != "identity":
                    raise RuntimeError(
                        f"Samsung realization permutes the rows of {rep.output_memref!r} "
                        "without a movement layout on the placement"
                    )
                if (
                    output is None
                    or output.storage != "banks"
                    or output.packing != (row_packing)
                ):
                    raise RuntimeError(
                        f"Samsung realization writes {rep.output_memref!r} to the banks "
                        f"with {row_packing} (row map {program.row_map}); the candidate "
                        f"declared {output}"
                    )
                carried = rep.accumulator_init == "carried"
                if carried and rep.output_memref not in effects.coverage.inputs:
                    raise RuntimeError(
                        f"Samsung realization uploads the carried destination "
                        f"{rep.output_memref!r} before the reduction; the candidate "
                        "does not declare it as an input"
                    )
                expected_transfers[rep.matrix_memref] = (
                    "in",
                    ("host_scatter", "banks"),
                )
                expected_transfers[rep.vector_memref] = (
                    "in",
                    (
                        ("host_scatter", "banks")
                        if vector_banks
                        else ("host_broadcast", "grf_a")
                    ),
                )
                expected_transfers[rep.output_memref] = (
                    ("inout", ("host_scatter", "banks", "host_gather"))
                    if carried
                    else ("out", ("banks", "host_gather"))
                )
                expected_allocations = {"grf_b", "grf_a"}
            else:
                el = program.elementwise
                if tuple(launch.instances) != (0, el.n):
                    raise RuntimeError(
                        f"Samsung elementwise realization covers {el.n} elements; "
                        f"the candidate declared instances {launch.instances}"
                    )
                for position, memref in enumerate(el.input_memrefs):
                    value = layout_of(memref, "in")
                    swapped = bool(
                        program.swap_parity[position]
                        if position < len(program.swap_parity)
                        else False
                    )
                    packing = "all_bank:swapped_fibers" if swapped else "all_bank"
                    if (
                        value is None
                        or value.storage != "banks"
                        or (value.packing != packing)
                    ):
                        raise RuntimeError(
                            f"Samsung elementwise realization streams {memref!r} from "
                            f"the banks as {packing}; the candidate declared {value}"
                        )
                    expected_transfers[memref] = ("in", ("host_scatter", "banks"))
                # The op's own result is external only when the program
                # declares it (a local intermediate copied to a sink is not).
                written = [
                    memref
                    for memref in (el.output_memref, *el.sink_memrefs)
                    if memref in effects.coverage.outputs
                ]
                if not written:
                    raise RuntimeError(
                        "Samsung elementwise realization writes "
                        f"{[el.output_memref, *el.sink_memrefs]}; the candidate "
                        f"declares outputs {list(effects.coverage.outputs)}"
                    )
                for memref in written:
                    value = layout_of(memref, "out")
                    if (
                        value is None
                        or value.storage != "banks"
                        or (value.packing != "all_bank")
                    ):
                        raise RuntimeError(
                            f"Samsung elementwise realization writes {memref!r} to the "
                            f"banks as all_bank; the candidate declared {value}"
                        )
                    expected_transfers[memref] = (
                        ("inout", ("host_scatter", "banks", "host_gather"))
                        if memref in el.input_memrefs
                        else ("out", ("banks", "host_gather"))
                    )
                expected_allocations = (
                    {"grf_a", "grf_b"} if el.op == "MAD" else {"grf_a"}
                )
            declared_transfers = {
                value: (t.direction, tuple(t.route))
                for value, t in transfers_by_value.items()
            }
            if declared_transfers != expected_transfers:
                raise RuntimeError(
                    "Samsung realization moves "
                    f"{sorted(expected_transfers.items())}; the candidate declared "
                    f"{sorted(declared_transfers.items())}"
                )
            if allocated != expected_allocations:
                raise RuntimeError(
                    f"Samsung realization uses registers {sorted(expected_allocations)}; "
                    f"the candidate declares {sorted(allocated)}"
                )

    def _check_unrealized_levers(self, layout):
        extra = layout.extra or {}
        tile = extra.get("tile")
        if tile is not None and not getattr(tile, "is_identity", True):
            raise _infeasible(
                "the Samsung lowering does not realize a non-identity tile"
            )
        if extra.get("double_buffer"):
            raise _infeasible("the Samsung lowering has no double-buffer pair")

    def _fiber_handles(self, layout, base_handle):
        """Bank handles of the placement's fibers, ordered by fiber index."""
        fibers = list((layout.extra or {}).get("fibers") or [base_handle])
        indexed = []
        for handle in fibers:
            if not isinstance(handle, MemoryRef):
                raise _infeasible(f"fiber handle {handle!r} is not a bank")
            cls = _bank_fiber_class(handle.idx)
            if cls not in _FIBER_OF_CLASS:
                raise _infeasible(
                    f"bank fiber class {cls!r} has no PIMSimulator operand encoding"
                )
            indexed.append((_FIBER_OF_CLASS[cls], handle))
        if len({f for f, _h in indexed}) != len(indexed):
            raise _infeasible("a placement names the same bank fiber twice")
        return indexed

    @staticmethod
    def _covered_contraction(buckets):
        """The group's contraction and its row count, after checking that the
        work items' store ranges partition the matrix rows (S-10)."""
        contractions = [samsung_contraction_roles(matches) for _w, matches in buckets]
        rep = contractions[0]
        if any(c.structure() != rep.structure() for c in contractions):
            raise SamsungUnsupported(
                "Samsung REDUCE lowering: work items differ in contraction structure"
            )
        # Row coverage (S-10): the work items' store ranges must partition the
        # matrix rows; the conductor computes and gathers every row.
        rows_total = rep.matrix_shape[1 - rep.matrix_k_axis]
        spans = sorted((c.row_offset, c.rows_per_item) for c in contractions)
        cursor = 0
        for offset, count in spans:
            if offset != cursor:
                raise ValueError(
                    f"Samsung REDUCE: the {len(spans)} work items store output rows "
                    f"{spans[0][0]}..{spans[-1][0] + spans[-1][1] - 1} with a gap or "
                    f"overlap at row {cursor} (next item starts at {offset}); the "
                    "work grid must partition the output rows exactly"
                )
            cursor += count
        if cursor != rows_total:
            raise ValueError(
                f"Samsung REDUCE: the work items cover {cursor} output rows but the "
                f"matrix {rep.matrix_memref!r} has {rows_total}; the device would "
                "overwrite rows the program leaves untouched. Make the work grid "
                "(mapping x rows per item) cover every row"
            )
        return rep, rows_total

    def _check_fixed_owner_map(self, buckets, layout):
        """The elementwise placement is fixed: reject a carried owner map
        other than the program's work-grid map."""
        from .pim.layout_problem import TargetHierarchyError
        from .spmw_layout_problem import (
            LayoutDerivationError,
            derive_layout_problem,
            split_movement_layout,
        )

        _backend, movement = split_movement_layout(getattr(layout, "layout", None))
        if movement is None:
            return
        try:
            derived = derive_layout_problem(
                self.target, [m for _w, ms in buckets for m in ms]
            )
            matrix = derived.movement_matrix(movement)
        except (LayoutDerivationError, TargetHierarchyError) as error:
            raise _infeasible(f"the carried movement layout cannot be placed: {error}")
        program = derived.program_layout()
        if derived.owner_flag(matrix) != derived.owner_flag(program):
            raise _infeasible(
                "the elementwise placement is fixed (flat channel and bank order); "
                "owner maps other than the work grid's are not realized"
            )

    def _row_slots(self, buckets, layout, rows_total):
        """Row-to-slot table of the owner map the placement carries.

        Returns ``(None, "identity")`` when the placement carries no movement
        layout or carries the program's own work-grid map.  Otherwise rows
        that the map assigns to one owner unit (``pseudo_channel`` x ``pim``,
        flattened with the outer level most significant) occupy that unit's
        contiguous slot range ``[unit * R, (unit + 1) * R)`` in ascending row
        order, ``R`` rows per unit.  The program map therefore reproduces the
        row-order packing exactly.  Raises InfeasibleSchedule with the reason
        when the map is not realizable (``row_map_unrealizable_reason``).
        """
        from .pim.layout_problem import TargetHierarchyError
        from .spmw_autoschedule import _matcher_work_scope
        from .spmw_layout_problem import (
            LayoutDerivationError,
            derive_layout_problem,
            row_map_unrealizable_reason,
            split_movement_layout,
        )

        _backend, movement = split_movement_layout(getattr(layout, "layout", None))
        if movement is None:
            return None, "identity"
        matches = [m for _w, ms in buckets for m in ms]
        try:
            derived = derive_layout_problem(self.target, matches)
            matrix = derived.movement_matrix(movement)
        except (LayoutDerivationError, TargetHierarchyError) as error:
            raise _infeasible(f"the carried movement layout cannot be placed: {error}")
        program = derived.program_layout()
        if matrix == program:
            return None, "identity"
        if derived.owner_flag(matrix) != derived.owner_flag(program):
            reason = row_map_unrealizable_reason(derived, matrix)
            if reason is not None:
                raise _infeasible(f"owner map not realizable: {reason}")
        inverse = matrix.inverse()
        n_work = len(derived.work_levels)
        units: dict[int, list[int]] = {}
        for _work_id, ms in buckets:
            c = samsung_contraction_roles(ms)
            mac = next(m for m in ms if m.target_op_name == "MAC")
            work_id = tuple(_matcher_work_scope(mac).work_id)
            loops = [loop[0] for loop in mac.enclosing_loops]
            row_loop = loops.index(c.row_var) if c.row_var is not None else None
            for i in range(c.rows_per_item):
                values = {d: work_id[d] for d in range(n_work)}
                if row_loop is not None:
                    values[n_work + row_loop] = i
                unit = derived.owner_unit(inverse, values)
                units.setdefault(unit, []).append(c.row_offset + i)
        per_unit = {len(rows) for rows in units.values()}
        if len(per_unit) != 1 or sorted(units) != list(range(len(units))):
            raise _infeasible(
                "owner map not realizable: its units own unequal row counts "
                f"({sorted(per_unit)})"
            )
        (R,) = per_unit
        slots = [0] * rows_total
        for unit, rows in units.items():
            for rank, row in enumerate(sorted(rows)):
                slots[row] = unit * R + rank
        if sorted(slots) != list(range(rows_total)):
            raise _infeasible("owner map does not place every row in one slot")
        if slots == list(range(rows_total)):
            return None, "identity"
        moved = sum(1 for row, slot in enumerate(slots) if row != slot)
        return (
            tuple(slots),
            f"permuted ({moved} of {rows_total} rows moved, {R} per unit)",
        )

    def _emit_reduce(self, group_id, buckets, layout, issue):
        from .spmw_reduction_policy import may_regroup, no_strict_program_error

        target = self.target
        facts = samsung_target_facts(target)
        rep, rows_total = self._covered_contraction(buckets)
        matches = [m for _w, ms in buckets for m in ms]
        if not may_regroup(matches):
            # Reached only with a forced layout: the enumerator prunes REDUCE
            # candidates under a strict policy.
            raise no_strict_program_error(
                target.name,
                matches,
                f"the selected REDUCE placement {layout.mode!r} sums per-lane FP16 "
                "partial sums in FP32; the ordered (temporal) placement keeps the "
                "source order",
            )
        if rep.accumulator_init != "zero":
            raise _infeasible(
                "the REDUCE program zeroes GRF_B; only the ordered (temporal) "
                "program carries the accumulator's initial value"
            )

        self._check_unrealized_levers(layout)
        row_slots, row_map = self._row_slots(buckets, layout, rows_total)
        handles = layout.placements
        matrix_h = handles.get(rep.matrix_memref)
        vector_h = handles.get(rep.vector_memref)
        acc_h = handles.get(rep.acc_memref)
        if acc_h is not target.grf_b:
            raise _infeasible("the MAC accumulator must live in GRF_B")
        if isinstance(matrix_h, Register):
            raise _infeasible(
                "grf_staged places the matrix in a GRF; a non-auto MAC reads fixed GRF "
                "entries (src0Idx/src1Idx), so its JUMP loop would repeat one product "
                "instead of walking the reduction, and a per-entry program needs more "
                f"than the {facts.crf_words}-word CRF"
            )
        if not isinstance(matrix_h, MemoryRef):
            raise _infeasible("the matrix must be bank-resident")
        if vector_h is not target.grf_a:
            raise _infeasible("the broadcast vector must be placed in GRF_A")
        if facts.grf_b_entries != facts.grf_a_entries:
            raise _infeasible("auto-mode MAC needs equal GRF_A and GRF_B depths")

        fibers = self._fiber_handles(layout, matrix_h)
        E, L, G = facts.grf_a_entries, facts.lanes, facts.grf_b_entries
        T = math.ceil(rep.K / (E * L))
        residency = (layout.extra or {}).get("grf_residency", {}) or {}
        vector_from_bank = residency.get(rep.vector_memref, "crf") != "host"
        mac = target.op("MAC")
        extra = layout.extra or {}
        issue_order = extra.get("issue_order", "tile_major")
        try:
            half_fibers = samsung_half_fibers(layout)
        except ValueError as error:
            raise _infeasible(str(error))
        tile_rows = "split_fibers" if half_fibers[1] != half_fibers[0] else "same_bank"
        declared_rows = extra.get("tile_rows")
        if declared_rows is not None and declared_rows != tile_rows:
            raise _infeasible(
                f"the candidate labels its tile rows {declared_rows!r} but its "
                f"half -> bank map places them {half_fibers} ({tile_rows})"
            )
        if issue_order not in ISSUE_ORDERS:
            raise _infeasible(f"unknown issue_order lever {issue_order!r}")
        if issue_order == "row_pair" or tile_rows == "split_fibers":
            # A tile's G*E columns must be exactly two physical rows of
            # cols_per_row burst columns (the pair, or the split, is the
            # tile's two rows), and the GRF_B entry of a group follows the
            # row parity.
            if G % 2 or G * E != 2 * SAMSUNG_DEFAULT_GEOMETRY["cols_per_row"]:
                raise _infeasible(
                    "row_pair/split_fibers need a tile of exactly two physical rows "
                    f"(G*E = {G * E}, cols_per_row = {SAMSUNG_DEFAULT_GEOMETRY['cols_per_row']})"
                )
        if issue_order == "row_pair" and tile_rows == "split_fibers":
            # Measured on the conductor (VALIDATION round 36): with the two
            # rows in two banks under one barrier, the memory controller may
            # reorder the two banks' column commands, and the PIM's lockstep
            # CRF executes whatever instruction is current with a fixed bank
            # operand (EVEN_BANK/ODD_BANK), so a command for one bank can
            # run the other bank's MAC and the results are wrong.  The ISA
            # has no bank-agnostic operand; two banks need a barrier between
            # them, which is the tile_major split_fibers candidate.
            raise _infeasible(
                "row_pair with split_fibers is unsound under the lockstep CRF: two "
                "banks' column commands inside one barrier group may be reordered "
                "against the fixed-bank MAC instructions (results differ; measured)"
            )
        if tile_rows == "split_fibers":
            if vector_from_bank:
                raise _infeasible(
                    "split_fibers has no CRF-filled GRF_A variant (the odd fiber holds "
                    "the tiles' second rows, not the vector)"
                )
            if [f for f, _h in fibers] != [0, 1]:
                raise _infeasible(
                    "split_fibers needs the even fiber (first rows) and the odd fiber "
                    "(second rows) of each PIM unit"
                )
            tile_fiber = tuple(0 for _ in range(T))
        else:
            tile_fiber = tuple(fibers[t % len(fibers)][0] for t in range(T))

        start = len(self.cmds)
        binds = []
        if tile_rows == "split_fibers":
            # One loop body of two auto MACs: the even fiber serves the first
            # row of a tile (groups 0..G/2-1), the odd fiber its second row;
            # the conductor issues one barrier per group.
            (even_fiber, even_h), (odd_fiber, odd_h) = fibers
            binds.append((len(self.cmds) - start, "matrix", even_fiber, -1, 0))
            mac.emit(vector_h, even_h, acc_h, self)
            binds.append((len(self.cmds) - start, "matrix", odd_fiber, -1, 0))
            mac.emit(vector_h, odd_h, acc_h, self)
            self._emit_jump(T * (G // 2) - 1, 3)
        else:
            for fiber_index, (fiber, handle) in enumerate(fibers):
                tiles = [t for t in range(T) if t % len(fibers) == fiber_index]
                if not tiles:
                    continue
                if vector_from_bank:
                    # One JUMP register: loops cannot nest, so each tile gets its
                    # own FILL + MAC row loop.
                    for _tile in tiles:
                        binds.append((len(self.cmds) - start, "vector", 0, -1, 0))
                        self._emit_move("LD_A")
                        binds.append((len(self.cmds) - start, "matrix", fiber, -1, 0))
                        mac.emit(vector_h, handle, acc_h, self)
                        self._emit_jump(G - 1, 2)
                else:
                    binds.append((len(self.cmds) - start, "matrix", fiber, -1, 0))
                    mac.emit(vector_h, handle, acc_h, self)
                    self._emit_jump(G * len(tiles) - 1, 2)
        binds.append((len(self.cmds) - start, "drain", 1, -1, 0))
        self._emit_move("ST_B")
        self.cmds.append(PIMCmd(type_="EXIT"))
        words = len(self.cmds) - start
        if words > facts.crf_words:
            raise _infeasible(
                f"the per-pass program needs {words} CRF words; the CRF holds "
                f"{facts.crf_words}"
            )
        return SamsungProgram(
            kind="REDUCE",
            group_id=group_id,
            start=start,
            end=len(self.cmds),
            binds=tuple(binds),
            facts=facts,
            crf_issue=issue,
            work_items=len(buckets),
            contraction=rep,
            rows=rows_total,
            k_tiles=T,
            tile_fiber=tile_fiber,
            rows_per_block=G,
            vector_from_bank=vector_from_bank,
            weight_resident=bool(extra.get("stage_resident", False)),
            lever_record={
                "mode": layout.mode,
                "fibers": len(fibers),
                "grf_residency": dict(residency),
                "crf_issue": issue,
                "stage_resident": bool(extra.get("stage_resident", False)),
                "residency": extra.get("residency"),
                "row_map": row_map,
                "issue_order": issue_order,
                "tile_rows": tile_rows,
            },
            row_slots=row_slots,
            row_map=row_map,
            issue_order=issue_order,
            half_fibers=tuple(half_fibers),
            tile_rows=tile_rows,
        )

    def _emit_temporal(self, group_id, buckets, layout, issue):
        """Lower the ordered program (conductor ``TEMPORAL_REDUCE``).

        FP16 lane l of PIM unit b owns one output row; the per-pass CRF is
        ``MOV GRF_B <- EVEN_BANK`` (initial accumulator, bank row 0),
        ``MAC GRF_B <- GRF_A * EVEN_BANK`` (not auto: one column command, one
        reduction element), ``JUMP K-1``, ``NOP 0`` (drain the one GRF_B
        entry) and ``EXIT``.  The conductor broadcasts element k of the vector
        into GRF_A before MAC k, in ascending k, so every multiply and add
        rounds to FP16 in source order and the host does no arithmetic.
        """
        target = self.target
        facts = samsung_target_facts(target)
        rep, rows_total = self._covered_contraction(buckets)
        self._check_unrealized_levers(layout)
        row_slots, row_map = self._row_slots(buckets, layout, rows_total)
        extra = layout.extra or {}
        handles = layout.placements
        matrix_h = handles.get(rep.matrix_memref)
        vector_h = handles.get(rep.vector_memref)
        acc_h = handles.get(rep.acc_memref)
        if acc_h is not target.grf_b:
            raise _infeasible("the ordered program accumulates in GRF_B")
        if vector_h is not target.grf_a:
            raise _infeasible("the ordered program broadcasts the vector into GRF_A")
        if (
            not isinstance(matrix_h, MemoryRef)
            or _bank_fiber_class(matrix_h.idx) != "EVEN_BANK"
        ):
            raise _infeasible("the ordered program reads the matrix from the even bank")
        residency = extra.get("grf_residency", {}) or {}
        if residency.get(rep.vector_memref) != "host":
            raise _infeasible(
                "the ordered program receives each reduction element by host "
                "broadcast (grf_residency=host)"
            )
        if extra.get("stage_resident"):
            raise _infeasible("the ordered program stages the matrix for every column")
        if not 1 <= rep.K <= _TEMPORAL_MAX_DEPTH:
            raise _infeasible(
                f"the ordered program supports 1 <= K <= {_TEMPORAL_MAX_DEPTH} "
                f"(conductor contract, TemporalReduce.inc); K = {rep.K}"
            )

        start = len(self.cmds)
        binds = [(0, "initial", 0, -1, 0)]
        self.cmd("MOV", dst=acc_h, src0=matrix_h)
        binds.append((len(self.cmds) - start, "matrix", 0, -1, 0))
        target.op("MAC").emit(vector_h, matrix_h, acc_h, self)
        # One column command per reduction element: no auto-mode GRF walk.
        self.cmds[-1].isAuto_ = 0
        self._emit_move("JUMP")
        self.cmds[-1].loopCounter_ = rep.K - 1
        self.cmds[-1].loopOffset_ = 2
        binds.append((len(self.cmds) - start, "drain", 1, -1, 0))
        self.cmds.append(PIMCmd(type_="NOP", loopCounter_=0))
        self.cmds.append(PIMCmd(type_="EXIT"))
        return SamsungProgram(
            kind="TEMPORAL_REDUCE",
            group_id=group_id,
            start=start,
            end=len(self.cmds),
            binds=tuple(binds),
            facts=facts,
            crf_issue=issue,
            work_items=len(buckets),
            contraction=rep,
            rows=rows_total,
            k_tiles=rep.K,
            rows_per_block=facts.lanes,
            vector_from_bank=False,
            weight_resident=False,
            lever_record={
                "mode": layout.mode,
                "reduction_program": "temporal",
                "grf_residency": dict(residency),
                "crf_issue": issue,
                "stage_resident": False,
                "residency": extra.get("residency"),
                "accumulator_init": rep.accumulator_init,
                "row_map": row_map,
            },
            row_slots=row_slots,
            row_map=row_map,
        )

    def _register_moves(self):
        """Structural load/drain moves of a PIM unit: which register each
        loads from which bank fiber, and which register drains to which."""
        loads, drains = {}, {}
        for unit in self.target._walk():
            if unit.name != "pim":
                continue
            for move in unit.moves.values():
                src, dst = move.src, move.dst
                if isinstance(src, MemoryRef) and isinstance(dst, Register):
                    cls = _bank_fiber_class(src.idx)
                    if cls in _FIBER_OF_CLASS:
                        loads[dst.name] = (move.name, _FIBER_OF_CLASS[cls])
                elif isinstance(src, Register) and isinstance(dst, MemoryRef):
                    cls = _bank_fiber_class(dst.idx)
                    if cls in _FIBER_OF_CLASS:
                        drains[_FIBER_OF_CLASS[cls]] = (move.name, src)
        return loads, drains

    def _emit_eltwise(self, group_id, buckets, layout, issue):
        target = self.target
        facts = samsung_target_facts(target)
        all_matches = [m for _w, matches in buckets for m in matches]
        el = samsung_elementwise(all_matches)
        self._check_unrealized_levers(layout)
        self._check_fixed_owner_map(buckets, layout)
        handles = layout.placements
        anchor = handles.get(el.input_memrefs[1]) if len(el.input_memrefs) > 1 else None
        anchor = anchor if isinstance(anchor, MemoryRef) else None
        fibers = self._fiber_handles(layout, anchor)
        if sorted(f for f, _h in fibers) != [0, 1]:
            raise _infeasible(
                "elementwise programs stream both bank fibers of every PIM unit "
                "(the flat all-bank operand layout)"
            )
        loads, drains = self._register_moves()
        op = target.op(el.op)
        E = facts.grf_a_entries
        start = len(self.cmds)
        binds = []
        swap = [False] * len(el.input_memrefs)
        for fiber, bank in fibers:
            if fiber not in drains:
                raise _infeasible(f"no move drains a register to bank fiber {fiber}")
            st_name, dest = drains[fiber]
            ld_name, ld_fiber = loads.get(dest.name, (None, None))
            if ld_name is None or ld_fiber != fiber:
                raise _infeasible(
                    f"register {dest.name} drains to fiber {fiber} but is not loaded from it"
                )
            pc = lambda: len(self.cmds) - start  # noqa: E731
            if el.op == "RELU":
                # The target's RELU is a non-auto MOV with the relu flag; one
                # word per GRF entry covers a tile.
                for entry in range(E):
                    binds.append((pc(), "input", fiber, 0, entry))
                    op.emit(bank, dest, self)
                    self.cmds[-1].dstIdx_ = entry
            else:
                binds.append((pc(), "input", fiber, 0, 0))
                self._emit_move(ld_name)
                other = None
                if el.op == "MAD":
                    other = target.grf_b if dest is target.grf_a else target.grf_a
                    o_name, o_fiber = loads.get(other.name, (None, None))
                    if o_name is None:
                        raise _infeasible(f"no move loads register {other.name}")
                    # The addend reaches the other register through its own
                    # load move, i.e. from the other fiber: store it with the
                    # fibers exchanged so that read returns this half's elements.
                    swap[2] = o_fiber != fiber
                    binds.append((pc(), "input", o_fiber, 2, 0))
                    self._emit_move(o_name)
                binds.append((pc(), "input", fiber, 1, 0))
                if el.op == "MAD":
                    op.emit(dest, bank, other, dest, self)
                else:
                    op.emit(dest, bank, dest, self)
            binds.append((pc(), "drain", fiber, -1, 0))
            self._emit_move(st_name)
        self.cmds.append(PIMCmd(type_="EXIT"))
        words = len(self.cmds) - start
        if words > facts.crf_words:
            raise _infeasible(
                f"the elementwise program needs {words} CRF words; the CRF holds "
                f"{facts.crf_words}"
            )
        return SamsungProgram(
            kind="ELTWISE",
            group_id=group_id,
            start=start,
            end=len(self.cmds),
            binds=tuple(binds),
            facts=facts,
            crf_issue=issue,
            work_items=len(buckets),
            elementwise=el,
            swap_parity=tuple(swap),
            lever_record={"mode": layout.mode, "crf_issue": issue},
        )

    def runtime_route(self, commands) -> tuple[str, tuple]:
        if any(program.kind == "TEMPORAL_REDUCE" for program in self.programs):
            route = "GENERIC_TEMPORAL_REDUCE"
        else:
            route = _samsung_runtime_route(commands)
        return (route, (_matcher_command_manifest(commands),))


# Conductor contract of the ordered program (TemporalReduce.inc): at most
# 4096 reduction elements per pass.
_TEMPORAL_MAX_DEPTH = 4096


def _is_temporal(layout) -> bool:
    return (getattr(layout, "extra", None) or {}).get("reduction_program") == "temporal"


def samsung_vendor_gemv_layout(target, trace) -> list:
    """Placements, one per work-item bucket, whose emitted program is the
    vendor GEMV CRF of PIMSimulator/PIMLibrary.

    That is ``dual_fiber`` (reduction tiles alternate between the even and
    odd bank of each PIM unit), host-broadcast GRF_A fill, one shared CRF,
    matrix staged once.  Pass it as ``allo.compile(..., layout=...)`` to pin
    the vendor-equivalent program (the paper's Samsung comparison rows); the
    default compile instead selects by cost.  Raises SamsungUnsupported when
    the trace is not a single-kernel contraction.
    """
    from .spmw_autoschedule import _bucket_for_autoschedule, _samsung_enumerate
    from .spmw_knobs import reset_active_liveness, set_active_liveness

    buckets = _bucket_for_autoschedule(trace)
    if not buckets or len({_scope[0] for _scope, _m in buckets}) != 1:
        raise SamsungUnsupported("the vendor GEMV layout needs a single-kernel trace")
    token = set_active_liveness(None)
    try:
        candidates = _samsung_enumerate(target, buckets[0][1])
    finally:
        reset_active_liveness(token)
    for candidate in candidates:
        extra = candidate.extra
        fills = set((extra.get("grf_residency") or {}).values())
        if (
            candidate.mode.split("+")[0] == "dual_fiber"
            and fills == {"host"}
            and extra.get("crf_issue") == "shared"
            and not extra.get("stage_resident")
        ):
            return [candidate] * len(buckets)
    pruned = {
        reason
        for candidate in candidates
        for _mode, reason in candidate.extra.get("reduction_policy_pruned", ())
    }
    raise SamsungUnsupported(
        "the trace has no dual_fiber + host-fill + shared-CRF candidate"
        + (f" (pruned: {'; '.join(sorted(pruned))})" if pruned else "")
    )


def _samsung_runtime_route(commands) -> str:
    op_types = {c.type_ for c in commands if isinstance(c, PIMCmd)}
    if "MAC" in op_types:
        return "GENERIC_REDUCE"
    return "GENERIC_ELTWISE"


def _write_samsung_cmds(path: Path, cmds) -> None:
    """Serialise PIMCmds to the line format `pim_driver --crf` reads."""
    Path(path).write_text(_samsung_cmds_text(cmds))


def _samsung_cmds_text(cmds) -> str:
    lines = []
    for c in cmds:
        parts = [c.type_]
        for field_name, key in (
            ("dst_", "dst"),
            ("src0_", "src0"),
            ("src1_", "src1"),
            ("src2_", "src2"),
        ):
            value = getattr(c, field_name)
            if value != "A_OUT":
                parts.append(f"{key}={value}")
        for field_name, key in (
            ("loopCounter_", "loop_counter"),
            ("loopOffset_", "loop_offset"),
            ("isAuto_", "is_auto"),
            ("dstIdx_", "dst_idx"),
            ("src0Idx_", "src0_idx"),
            ("src1Idx_", "src1_idx"),
            ("isRelu_", "is_relu"),
        ):
            value = getattr(c, field_name)
            if value:
                parts.append(f"{key}={value}")
        lines.append(" ".join(parts))
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------- #


_GEOMETRY_CACHE: dict = {}


def samsung_simulator_geometry(root: Path) -> dict:
    """Ask the conductor for the simulated geometry (`pim_driver --describe`)."""
    driver = Path(root) / "pim_driver"
    stat = driver.stat()
    key = (str(driver), stat.st_mtime_ns, stat.st_size)
    if key in _GEOMETRY_CACHE:
        return dict(_GEOMETRY_CACHE[key])
    proc = subprocess.run(
        [str(driver), "--describe"],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    m = re.search(r"PIM_GEOMETRY (.*)", proc.stdout)
    if proc.returncode != 0 or not m:
        raise RuntimeError(
            f"{driver} does not report its geometry (`--describe`); rebuild it from "
            "experiments/simulators/setup/PIMSimulator (see its README). Output: "
            + (proc.stdout + proc.stderr)[-400:]
        )
    geometry = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", m.group(1))}
    _GEOMETRY_CACHE[key] = geometry
    return dict(geometry)


def _check_geometry(facts: SamsungFacts, geometry: dict) -> None:
    expected = {
        "pim_blocks": facts.pim_blocks,
        "grf_a": facts.grf_a_entries,
        "grf_b": facts.grf_b_entries,
        "lanes": facts.lanes,
        "crf_words": facts.crf_words,
        "banks": facts.banks,
    }
    mismatched = {
        key: (value, geometry.get(key))
        for key, value in expected.items()
        if geometry.get(key) != value
    }
    if mismatched:
        raise RuntimeError(
            "PIMSimulator geometry disagrees with the Samsung target "
            f"(target, simulator): {mismatched}"
        )


def _samsung_outbin_error(out_path, what, need, have, driver_log):
    tail = (driver_log or "")[-800:]
    return RuntimeError(
        f"Samsung pim_driver {what} readback: {out_path} "
        + ("is absent" if have is None else "is too short")
        + f"; expected {need} bytes, got {0 if have is None else have} bytes. "
        f"Driver output tail:\n{tail}"
    )


def _samsung_read_bursts(out_path, nbursts, what, driver_log=""):
    """Read `nbursts` 16-lane FP16 bursts; an all-zero blob is a valid result.
    Raises RuntimeError when the file is absent or short."""
    import numpy as np

    p = Path(out_path)
    need = int(nbursts) * 16 * np.dtype(np.float16).itemsize
    if not p.exists():
        raise _samsung_outbin_error(p, what, need, None, driver_log)
    have = p.stat().st_size
    if have < need:
        raise _samsung_outbin_error(p, what, need, have, driver_log)
    return np.fromfile(str(p), dtype=np.float16)[: int(nbursts) * 16].reshape(-1, 16)


def _plan_header(kind, facts, geometry):
    return [
        "tenon_samsung_plan v1",
        f"kind {kind}",
        f"channels {geometry['channels']}",
        f"pim_blocks {facts.pim_blocks}",
        f"grf_a {facts.grf_a_entries}",
        f"grf_b {facts.grf_b_entries}",
        f"lanes {facts.lanes}",
        f"crf_words {facts.crf_words}",
    ]


def _plan_binds(program):
    return [
        f"bind {pc} {kind} {bank} {operand} {offset}"
        for pc, kind, bank, operand, offset in program.binds
    ]


def _row_map_digest(values) -> str:
    """FNV-1a 64 over the table values; the conductor reports the same digest."""
    digest = 0xCBF29CE484222325
    for value in values:
        digest ^= int(value)
        digest = (digest * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{digest:016x}"


# --------------------------------------------------------------------- #
# Executable access stream of a lowered REDUCE program
# --------------------------------------------------------------------- #

# Samsung HBM-PIM geometry the conductor reports (`pim_driver --describe`);
# used when the simulator is not installed.  cols_per_row is NUM_COLS / BL.
SAMSUNG_DEFAULT_GEOMETRY = {
    "channels": 64,
    "ranks": 1,
    "pim_blocks": 8,
    "banks": 16,
    "bank_groups": 4,
    "cols_per_row": 32,
    "rows": 16384,
}


@dataclass(frozen=True)
class SamsungAccess:
    """One normalized DRAM transaction of channel 0 (the other channels
    receive the same address): bank within the PIM unit's pair for the
    all-bank PIM commands, physical row after the column-to-row carry of
    ``PIMAddrManager::addrGenSafe``, burst column."""

    write: bool
    bank: int
    row: int
    column: int


@dataclass(frozen=True)
class SamsungAccessRound:
    """Transactions issued together and closed by one barrier on every
    channel, in the conductor's order.  ``kind`` is MAC (a matrix tile's
    auto-MAC group, E columns), FILL (a CRF-issued GRF_A fill from the even
    bank), HOST_FILL (the host's GRF_A broadcast, no bank transaction) or
    DRAIN (the GRF_B writeback, G columns to the odd bank)."""

    index: int
    column_index: int  # vector n
    pass_index: int  # output tile j
    kind: str
    fiber: int | None
    tile: int | None
    group: int | None
    accesses: tuple

    @property
    def rows(self) -> tuple:
        return tuple(sorted({a.row for a in self.accesses}))


def _normalized(cols_per_row: int, column: int) -> tuple:
    return divmod(int(column), int(cols_per_row))


@dataclass(frozen=True)
class SamsungAccessStream:
    """The executable pass-phase stream of one lowered REDUCE program.

    ``geometry_source`` says where the simulated geometry came from: the
    installed conductor (``pim_driver --describe``) or the explicitly
    requested offline default; an offline stream is a model of the
    conductor's addressing, not the installed simulator's stream.
    """

    rounds: tuple
    columns: int
    geometry: dict
    geometry_source: str

    def __iter__(self):
        return iter(self.rounds)

    def __len__(self):
        return len(self.rounds)


GEOMETRY_SOURCE_INSTALLED = "installed pim_driver --describe"
GEOMETRY_SOURCE_DEFAULT = "SAMSUNG_DEFAULT_GEOMETRY (offline model, no installed simulator queried)"


def samsung_reduce_access_stream(
    program: SamsungProgram, geometry: dict, columns: int | None = None, weight_resident=None
) -> list:
    """The transaction stream the conductor issues for ``program`` during its
    passes, derived from the lowered program (``binds``, ``tile_fiber``,
    ``k_tiles``, ``rows_per_block``, ``vector_from_bank``) and the simulated
    geometry, in the order ``TenonConductor.inc`` ``run_pass`` executes the
    CRF: one round per barrier.  ``columns`` is the program's result column
    count (``contraction.N``, the complete program); passing another count
    is an error, so a stream is never a silent subset or repetition of the
    program.

    Columns follow the conductor: a matrix tile occupies ``G*E`` consecutive
    columns of its fiber at ``(pass * tiles_on_fiber + local) * G*E``; the
    vectors sit after the matrix in the even bank and the drained results
    after the matrix in the odd bank; ``addrGenSafe`` carries column
    overflow into the row (``cols_per_row`` burst columns per row), so a
    tile of 64 columns spans two physical rows.  Matrix staging, CRF
    programming, mode changes and readback are not part of the stream.
    """

    facts = program.facts
    if program.kind != "REDUCE":
        raise ValueError(f"access stream of a {program.kind} program is not modelled")
    program_columns = int(program.contraction.N) if program.contraction is not None else 1
    if columns is None:
        columns = program_columns
    elif isinstance(columns, bool) or not isinstance(columns, int) or columns <= 0:
        raise ValueError(f"columns must be a positive integer, got {columns!r}")
    elif int(columns) != program_columns:
        raise ValueError(
            f"the program computes {program_columns} result column(s); a stream of "
            f"{columns} is not the program's stream (pass columns=None)"
        )
    for key in ("channels", "cols_per_row"):
        if key not in geometry:
            raise ValueError(f"geometry lacks {key!r}: {sorted(geometry)}")
    G, E, T = program.rows_per_block, facts.grf_a_entries, program.k_tiles
    C, P = int(geometry["channels"]), int(geometry.get("pim_blocks", facts.pim_blocks))
    cols_per_row = int(geometry["cols_per_row"])
    J = math.ceil(program.rows / (C * P * G))
    tile_cols = G * E
    fiber_tiles = {
        f: [t for t in range(T) if program.tile_fiber[t] == f] for f in (0, 1)
    }
    local = {t: fiber_tiles[program.tile_fiber[t]].index(t) for t in range(T)}
    vector_base = J * len(fiber_tiles[0]) * tile_cols
    result_base = J * len(fiber_tiles[1]) * tile_cols
    if tuple(getattr(program, "half_fibers", (0, 0)) or (0, 0)) == (0, 1):
        result_base = J * len(fiber_tiles[0]) * tile_cols
    N = int(columns)
    rounds = []

    def add(kind, n, j, fiber, tile, group, write, bank, first_col, count):
        accesses = []
        for k in range(count):
            row, col = _normalized(cols_per_row, first_col + k)
            accesses.append(SamsungAccess(write, bank, row, col))
        rounds.append(
            SamsungAccessRound(
                len(rounds), n, j, kind, fiber, tile, group, tuple(accesses)
            )
        )

    row_pair = getattr(program, "issue_order", "tile_major") == "row_pair"
    halves = tuple(getattr(program, "half_fibers", (0, 0)) or (0, 0))
    split = halves[1] != halves[0]
    if split:
        # The odd fiber serves the second row of every even-fiber tile.
        fiber_tiles = {0: list(fiber_tiles[0]), 1: list(fiber_tiles[0])}
    for n in range(N):
        for j in range(J):
            groups = {0: 0, 1: 0}
            pair_issued = {}
            filled = None
            binds = list(program.binds)
            matrix_binds = [b for b in binds if b[1] == "matrix"]
            if split:
                # One loop body of (even MAC, odd MAC) repeated T*G/2 times.
                executions = []
                for _ in range(len(fiber_tiles[0]) * (G // 2)):
                    executions.extend(int(b[2]) for b in matrix_binds)
                sequence = [("matrix", f) for f in executions]
                sequence.append(("drain", 1))
            else:
                sequence = []
                for _pc, kind, fiber, _operand, _offset in binds:
                    if kind == "matrix":
                        f = int(fiber)
                        count = G if program.vector_from_bank else G * len(fiber_tiles[f])
                        sequence.extend([("matrix", f)] * count)
                    elif kind == "vector":
                        sequence.append(("vector", _pc))
                    elif kind == "drain":
                        sequence.append(("drain", 1))
            for kind, fiber in sequence:
                if kind == "matrix":
                    f = int(fiber)
                    tiles = fiber_tiles[f]
                    e = groups[f]
                    groups[f] += 1
                    if split:
                        loc = e // (G // 2)
                        grp = e % (G // 2) + (G // 2 if f else 0)
                    elif row_pair:
                        loc, q = divmod(e, G)
                        grp = q // 2 if q % 2 == 0 else q // 2 + G // 2
                    else:
                        loc, grp = divmod(e, G)
                    tile = tiles[loc]
                    first_group_of_tile = (grp == 0) if not split else (f == 0 and grp == 0)
                    if not program.vector_from_bank and first_group_of_tile:
                        rounds.append(
                            SamsungAccessRound(
                                len(rounds), n, j, "HOST_FILL", None, tile, 0, ()
                            )
                        )
                    col = (j * len(tiles) + loc) * tile_cols + grp * E
                    add("MAC", n, j, f, tile, grp, False, f, col, E)
                    if row_pair:
                        key = (loc, grp % (G // 2))
                        pair_issued[key] = pair_issued.get(key, 0) + 1
                        if pair_issued[key] < 2:
                            # the first half of a pair: no barrier yet; merge
                            # into one round with the second half
                            first = rounds.pop()
                            pending = first
                            continue
                        if "pending" in locals() and pending is not None:
                            second = rounds.pop()
                            rounds.append(
                                SamsungAccessRound(
                                    len(rounds), n, j, "MAC", None, tile, grp,
                                    tuple(pending.accesses) + tuple(second.accesses),
                                )
                            )
                            pending = None
                elif kind == "vector":
                    # The fill provides the tile of the next MAC group on the
                    # fiber the next matrix bind names.
                    pc = fiber
                    nxt = next(
                        (int(b[2]) for b in binds if b[1] == "matrix" and b[0] > pc),
                        None,
                    )
                    if nxt is None:
                        raise ValueError("a vector fill without a following MAC group")
                    tile = fiber_tiles[nxt][groups[nxt] // G]
                    col = vector_base + (n * T + tile) * E
                    add("FILL", n, j, 0, tile, None, False, 0, col, E)
                    filled = tile
                elif kind == "drain":
                    col = result_base + (n * J + j) * G
                    add("DRAIN", n, j, 1, None, None, True, 1, col, G)
            del filled
    return rounds


def samsung_access_stream(
    compiled, columns: int | None = None, geometry: dict | None = None
) -> SamsungAccessStream:
    """The executable access stream of a compiled single-kernel Samsung
    REDUCE program (an ``allo.compile`` result or its backend artifact).

    With ``geometry=None`` the installed conductor is asked for its geometry
    and any discovery or configuration error propagates (an installed
    simulator that cannot describe itself yields no stream).  Offline
    modelling passes ``geometry=SAMSUNG_DEFAULT_GEOMETRY`` (or another
    explicit geometry) and the result is labelled as such.
    """

    artifact = getattr(compiled, "compiled", compiled)
    program = _program_of(artifact)
    if geometry is None:
        geometry = samsung_simulator_geometry(_pimsim_root())
        source = GEOMETRY_SOURCE_INSTALLED
    elif geometry is SAMSUNG_DEFAULT_GEOMETRY or geometry == SAMSUNG_DEFAULT_GEOMETRY:
        source = GEOMETRY_SOURCE_DEFAULT
    else:
        source = "explicit geometry"
    rounds = samsung_reduce_access_stream(program, geometry, columns)
    resolved = int(program.contraction.N) if program.contraction is not None else 1
    return SamsungAccessStream(tuple(rounds), resolved, dict(geometry), source)


def samsung_access_stream_summary(rounds) -> dict:
    """Counts of the modelled stream: rounds by kind, the rows each bank
    touches and the address changes between successive MAC rounds of one
    bank.  These are properties of the derived address sequence, not
    measured row activations or simulator timing; what the simulator does
    with the sequence is only known from running it."""

    label = getattr(rounds, "geometry_source", None)
    rounds = list(rounds)
    by_kind = Counter(r.kind for r in rounds)
    rows = {}
    transitions = {}
    last = {}
    for r in rounds:
        for a in r.accesses:
            rows.setdefault(a.bank, set()).add(a.row)
        if r.kind != "MAC" or not r.accesses:
            continue
        bank, row = r.accesses[0].bank, r.accesses[0].row
        if bank in last and last[bank] != row:
            transitions[bank] = transitions.get(bank, 0) + 1
        last[bank] = row
    return {
        "rounds": len(rounds),
        "rounds_by_kind": dict(sorted(by_kind.items())),
        "transactions": sum(len(r.accesses) for r in rounds),
        "rows_by_bank": {b: sorted(v) for b, v in sorted(rows.items())},
        # Address changes between successive MAC groups of one bank in the
        # modelled stream; not measured row activations of the simulator.
        "mac_same_bank_row_transitions": dict(sorted(transitions.items())),
        "geometry_source": label,
    }


def samsung_access_trace_text(rounds, kinds=("MAC", "FILL")) -> str:
    """The stream as the layout-trace harness consumes it (``round bank row
    col``, one line per read transaction of channel 0, which the harness
    replicates over the 64 channels and closes with a barrier per round),
    restricted to the read rounds of ``kinds``: a replay of the pass-phase
    bank reads without PIM execution, writes, staging or mode changes."""

    lines = []
    index = 0
    for r in rounds:
        if r.kind not in kinds or not r.accesses:
            continue
        for a in r.accesses:
            lines.append(f"{index} {a.bank} {a.row} {a.column}")
        index += 1
    return "\n".join(lines) + ("\n" if lines else "")


def samsung_conductor_trace_text(rounds) -> str:
    """The stream in the conductor's own trace format
    (``TENON_CONDUCTOR_ADDRESS_TRACE``: ``PASS n j``, ``<tag> 0 bank row
    col``, ``BARRIER``), for byte comparison with a captured dump."""

    lines = []
    current = None
    for r in rounds:
        key = (r.column_index, r.pass_index)
        if key != current:
            lines.append(f"PASS {r.column_index} {r.pass_index}")
            current = key
        if r.kind == "HOST_FILL":
            lines.append(f"Whost_fill_grf_a x{r.accesses and len(r.accesses) or 8}")
        else:
            tag = {"MAC": "MAC", "FILL": "FILL", "DRAIN": "WDRAIN"}[r.kind]
            for a in r.accesses:
                lines.append(f"{tag} 0 {a.bank} {a.row} {a.column}")
        lines.append("BARRIER")
    return "\n".join(lines) + ("\n" if lines else "")


def samsung_slot_position(
    slot: int, geometry: dict, facts, rows_per_block: int
) -> dict:
    """Physical position of one REDUCE slot: pass, channel, PIM block, GRF_B entry."""
    C, P, G = geometry["channels"], facts.pim_blocks, rows_per_block
    return {
        "pass": slot // (C * P * G),
        "channel": slot // (P * G) % C,
        "block": slot // G % P,
        "entry": slot % G,
    }


def samsung_reduce_plan(
    program: SamsungProgram,
    geometry: dict,
    columns: int,
    weight_resident: bool | None = None,
) -> dict:
    """Physical plan of a REDUCE program on the simulated geometry.

    Slot s is ((pass * channels + channel) * blocks + block) * rows_per_block
    + entry.  Without a row map, row r occupies slot r (row-order packing,
    the vendor placement).  When the selected owner map places rows
    elsewhere (``program.row_slots``), the plan carries ``row_slots``: the
    slot of every row of the zero-padded matrix (padding rows keep their own
    slots), a permutation the conductor applies when it stages the matrix and
    when it drains the results, which it returns in row order.
    """
    facts = program.facts
    C, P, G = geometry["channels"], facts.pim_blocks, program.rows_per_block
    E, L = facts.grf_a_entries, facts.lanes
    passes = math.ceil(program.rows / (C * P * G))
    resident = program.weight_resident if weight_resident is None else weight_resident
    lines = _plan_header("REDUCE", facts, geometry) + [
        f"out_tiles {passes}",
        f"k_tiles {program.k_tiles}",
        "tile_fiber " + " ".join(str(f) for f in program.tile_fiber),
        f"rows_per_block {G}",
        f"vector_from_bank {int(program.vector_from_bank)}",
        f"columns {columns}",
        f"weight_resident {int(bool(resident))}",
        f"issue_order {int(program.issue_order == 'row_pair')}",
        "half_fiber " + " ".join(str(f) for f in (program.half_fibers or (0, 0))),
    ]
    slots = passes * C * P * G
    table = None
    if program.row_slots is not None:
        table = list(program.row_slots) + list(range(len(program.row_slots), slots))
        lines.append("row_slots " + " ".join(str(slot) for slot in table))
    lines += _plan_binds(program)
    return {
        "text": "\n".join(lines) + "\n",
        "passes": passes,
        "slots": slots,
        "padded_k": program.k_tiles * E * L,
        "weight_resident": bool(resident),
        "row_map": program.row_map,
        "row_map_digest": _row_map_digest(table) if table is not None else None,
    }


def samsung_eltwise_plan(program: SamsungProgram, geometry: dict) -> dict:
    facts = program.facts
    C = geometry["channels"]
    tile_elements = C * facts.banks * facts.grf_a_entries * facts.lanes
    tiles = math.ceil(program.elementwise.n / tile_elements)
    bursts = tiles * tile_elements // facts.lanes
    row_bursts = C * facts.banks * geometry["cols_per_row"]
    rows_per_operand = math.ceil(bursts / row_bursts)
    n_inputs = len(program.elementwise.input_memrefs)
    if (n_inputs + 1) * rows_per_operand > geometry["rows"]:
        raise ValueError(
            f"Samsung elementwise operands of {program.elementwise.n} elements need "
            f"{(n_inputs + 1) * rows_per_operand} DRAM rows; a bank has {geometry['rows']}"
        )
    input_rows = [k * rows_per_operand for k in range(n_inputs)]
    lines = (
        _plan_header("ELTWISE", facts, geometry)
        + [
            f"tiles {tiles}",
            f"inputs {n_inputs}",
            "input_rows " + " ".join(str(r) for r in input_rows),
            "input_swap_parity " + " ".join(str(int(s)) for s in program.swap_parity),
            f"output_row {n_inputs * rows_per_operand}",
        ]
        + _plan_binds(program)
    )
    return {
        "text": "\n".join(lines) + "\n",
        "tiles": tiles,
        "padded_n": tiles * tile_elements,
    }


def _invoke_conductor(root, argv, what):
    try:
        proc = subprocess.run(
            argv, capture_output=True, cwd=str(root), timeout=1800, check=False
        )
    except (subprocess.SubprocessError, OSError) as exc:
        raise RuntimeError(
            f"Samsung pim_driver {what} invocation failed: {exc}"
        ) from exc
    combined = proc.stdout.decode("utf-8", "replace") + proc.stderr.decode(
        "utf-8", "replace"
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Samsung pim_driver {what} failed (rc={proc.returncode}): "
            + combined[-1200:]
        )
    m = re.search(
        r"PIM_CYCLES total=(\d+) preload=(\d+) exec=(\d+) readback=(\d+)", combined
    )
    if not m:
        raise RuntimeError(
            f"Samsung pim_driver {what} reported no cycles: {combined[-600:]}"
        )
    receipt = re.search(r"TENON_CONDUCTOR (.*)", combined)
    conductor = (
        {
            k: (int(v) if v.isdigit() else v)
            for k, v in re.findall(r"(\w+)=(\S+)", receipt.group(1))
        }
        if receipt
        else {}
    )
    row_map = re.search(r"TENON_ROW_MAP (.*)", combined)
    if row_map:
        conductor.update(
            {
                f"row_map_{k}": (int(v) if v.isdigit() and k != "digest" else v)
                for k, v in re.findall(r"(\w+)=(\S+)", row_map.group(1))
            }
        )
    phases = {
        "preload": int(m.group(2)),
        "exec": int(m.group(3)),
        "readback": int(m.group(4)),
    }
    return int(m.group(1)), phases, conductor, combined


_REDUCE_TIMING_SCOPE = (
    "PIMSimulator memory-clock cycles of the executed padded program on the "
    "simulated system: matrix (and, for CRF-filled GRF_A, vector) preload, "
    "CRF upload, all passes with their GRF_A fills, and readback of the "
    "drained partial sums. Excludes host padding, the host 16-lane tree "
    "reduction and row un-permutation."
)
_ELTWISE_TIMING_SCOPE = (
    "PIMSimulator memory-clock cycles of the executed padded program: operand "
    "preload, CRF upload, one pass per tile, and output readback. Excludes "
    "host padding."
)


def _samsung_run_reduce(
    compiled, program, crf_words, root, geometry, matrix, vectors, weight_resident=None
):
    """Execute one REDUCE program for `vectors` (N, K) and return the result
    rows (N, M), cycles, phases, plan and conductor receipt."""
    import numpy as np

    facts = program.facts
    c = program.contraction
    M, K = program.rows, c.K
    N = int(vectors.shape[0])
    plan = samsung_reduce_plan(program, geometry, N, weight_resident)
    slots, Kp = plan["slots"], plan["padded_k"]
    weight = np.zeros((slots, Kp), dtype=np.float16)
    weight[:M, :K] = matrix
    vec = np.zeros((N, Kp), dtype=np.float16)
    vec[:, :K] = vectors
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        a_path, b_path, out_path = td / "A.npy", td / "B.npy", td / "out.bin"
        crf_path, plan_path = td / "crf.txt", td / "plan.txt"
        np.save(a_path, weight)
        np.save(b_path, vec)
        crf_text = _samsung_cmds_text(crf_words)
        crf_path.write_text(crf_text)
        plan_path.write_text(plan["text"])
        argv = [
            str(Path(root) / "pim_driver"),
            "--op",
            "GENERIC",
            "--op-kind",
            "REDUCE",
            "--plan",
            str(plan_path),
            "--crf",
            str(crf_path),
            "--inputs",
            f"{a_path},{b_path}",
            "--out",
            str(out_path),
        ]
        cycles, phases, conductor, log = _invoke_conductor(root, argv, "REDUCE")
        bursts = _samsung_read_bursts(out_path, N * slots, "REDUCE", log)
    if plan["row_map_digest"] is not None and (
        conductor.get("row_map_map") != "permuted"
        or conductor.get("row_map_digest") != plan["row_map_digest"]
    ):
        raise RuntimeError(
            "Samsung pim_driver did not confirm the plan's row map (expected "
            f"digest {plan['row_map_digest']}, receipt {conductor}); rebuild "
            "pim_driver from experiments/simulators/setup/PIMSimulator"
        )
    # Host reduction of the 16 lane partial sums (the regrouping the paper
    # permits).  The conductor returns the drained slots in row order (the
    # plan's row_slots permutation, identity without one).
    sums = bursts.astype(np.float32).sum(axis=1).reshape(N, slots)[:, :M]
    issued = slots * Kp
    slot_of = program.row_slots or range(M)
    row_map = {
        "map": program.row_map,
        "digest": plan["row_map_digest"],
        "placement": {
            row: samsung_slot_position(
                slot_of[row], geometry, facts, program.rows_per_block
            )
            for row in sorted({0, 1, 2, 31, 32, M // 2, M - 1} & set(range(M)))
        },
    }
    return {
        "row_map": row_map,
        "rows": sums,
        "cycles": cycles,
        "phases": phases,
        "plan": plan,
        "crf": crf_text,
        "conductor": conductor,
        "log": log,
        "logical_shape": {"rows": M, "reduction": K, "columns": N},
        "padded_shape": {"rows": slots, "reduction": Kp, "columns": N},
        "padding_fraction": 1.0 - (M * K) / issued,
    }


_TEMPORAL_TIMING_SCOPE = (
    "PIMSimulator memory-clock cycles of the executed ordered program on the "
    "simulated system: for every row tile and result column, the weight and "
    "initial-accumulator preload (K + 1 bursts per PIM unit), CRF upload, K "
    "host broadcasts of one vector element each followed by one MAC, the "
    "drain, and readback. The outputs are the drained FP16 accumulators; the "
    "host performs no arithmetic. Excludes host padding."
)


def _samsung_run_temporal(program, crf_words, root, geometry, matrix, vectors, initial):
    """Execute the ordered program: ``matrix`` (M, K), ``vectors`` (N, K),
    ``initial`` (M, N); return the FP16 results (M, N) and the run record."""
    import numpy as np

    facts = program.facts
    M, K = matrix.shape
    N = int(vectors.shape[0])
    tile_rows = geometry["channels"] * facts.pim_blocks * facts.lanes
    padded_rows = math.ceil(M / tile_rows) * tile_rows
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        a_path, b_path, i_path = td / "A.npy", td / "B.npy", td / "init.npy"
        out_path, crf_path = td / "out.bin", td / "crf.txt"
        np.save(a_path, np.ascontiguousarray(matrix, dtype=np.float16))
        np.save(b_path, np.ascontiguousarray(vectors.T, dtype=np.float16))
        np.save(i_path, np.ascontiguousarray(initial, dtype=np.float16))
        crf_text = _samsung_cmds_text(crf_words)
        crf_path.write_text(crf_text)
        argv = [
            str(Path(root) / "pim_driver"),
            "--op",
            "GENERIC",
            "--op-kind",
            "TEMPORAL_REDUCE",
            "--cmds",
            str(crf_path),
            "--inputs",
            f"{a_path},{b_path},{i_path}",
            "--out-rows",
            str(M),
            "--reduce-k",
            str(K),
            "--out-cols",
            str(N),
            "--temporal-audit",
            str(td / "audit"),
            "--out",
            str(out_path),
        ]
        cycles, phases, _conductor, log = _invoke_conductor(
            root, argv, "TEMPORAL_REDUCE"
        )
        bursts = _samsung_read_bursts(
            out_path, (M * N + 15) // 16, "TEMPORAL_REDUCE", log
        )
    receipt = re.search(r"PIM_TEMPORAL_REDUCE (.*)", log)
    if not receipt or "ordered_k=1" not in receipt.group(1):
        raise RuntimeError(
            "Samsung pim_driver TEMPORAL_REDUCE did not confirm ordered execution: "
            + log[-600:]
        )
    conductor = {
        k: (int(v) if v.isdigit() else v)
        for k, v in re.findall(r"(\w+)=(\S+)", receipt.group(1))
    }
    return {
        "rows": bursts.reshape(-1)[: M * N].reshape(M, N),
        "cycles": cycles,
        "phases": phases,
        "crf": crf_text,
        "conductor": conductor,
        "log": log,
        "logical_shape": {"rows": M, "reduction": K, "columns": N},
        "padded_shape": {"rows": padded_rows, "reduction": K, "columns": N},
        "padding_fraction": 1.0 - M / padded_rows,
    }


def _program_of(compiled) -> SamsungProgram:
    ctx = getattr(compiled, "layout_ctx", None)
    programs = list(getattr(ctx, "programs", ()) or ())
    if not programs:
        raise RuntimeError(
            "compiled Samsung artifact carries no lowered program record; compile "
            "it with allo.compile / compile_for_target"
        )
    if len(programs) != 1:
        raise NotImplementedError(
            f"this Samsung entry point runs one kernel; the artifact has "
            f"{len(programs)} kernels (run a multi-kernel artifact with run(), "
            "which follows its host program's launch schedule)"
        )
    return programs[0]


def _check_host_moves(compiled, program, inputs) -> None:
    """Recorded host moves must name the operands the conductor moves that way."""
    moves = getattr(compiled, "host_moves", None)
    if not moves:
        return
    supplied = {name for name, value in inputs.items() if value is not None}
    for rhm in moves:
        verb = getattr(rhm.verb, "name", str(rhm.verb))
        role = rhm.buffer_role
        if verb == "gather" or role is None:
            continue
        if role not in supplied:
            raise ValueError(
                f"Samsung host move ({verb}) binds buffer {role!r}, but no input "
                f"named {role!r} was supplied (supplied: {sorted(supplied)})"
            )
        if program.kind not in ("REDUCE", "TEMPORAL_REDUCE"):
            continue
        value = inputs[role]
        is_matrix = value is inputs.get("A")
        is_vector = value is inputs.get("B")
        if verb == "scatter" and not (
            is_matrix or (is_vector and program.vector_from_bank)
        ):
            raise ValueError(
                f"Samsung host move scatter binds {role!r}, which the selected program "
                "does not place in the banks"
            )
        if verb == "broadcast" and not (is_vector and not program.vector_from_bank):
            raise ValueError(
                f"Samsung host move broadcast binds {role!r}, but the selected program "
                "broadcasts only the vector, and only with host GRF_A residency"
            )


def _run_samsung(compiled: "Compiled", **inputs) -> RunResult:
    """Run a compiled Samsung HBM-PIM artifact on PIMSimulator.

    The emitted CRF words of the kernel are uploaded verbatim; the plan
    derived from the selected placement and the simulated geometry places the
    operands and drives the passes (see ``samsung_reduce_plan``).  Operands
    arrive under the structural runtime roles bound by
    ``source_backend_binding``: ``A`` (bank-resident matrix) and ``B``
    (broadcast vector, or the ``[K, N]`` second operand) for a contraction;
    ``IN0..IN2`` for an elementwise op.

    ``RunResult.cycles`` is the simulated cycle count of the *padded* program
    (``extra["timing_scope"]``); ``extra`` also records the logical and padded
    shapes, the padding fraction of issued work, the plan, the CRF and the
    conductor receipt.  Missing operands raise ValueError; a conductor
    rejection or an absent/short readback raises RuntimeError.

    An artifact with several kernels runs its host program's launch schedule,
    one conductor invocation per launch (``_run_samsung_launches``); its
    operands arrive under the workload's parameter names.
    """
    ctx_programs = list(
        getattr(getattr(compiled, "layout_ctx", None), "programs", ()) or ()
    )
    if len(ctx_programs) > 1:
        return _run_samsung_launches(compiled, inputs)
    reason = _samsung_unavailable_reason()
    if reason is not None:
        raise SimulatorUnavailable("samsung_hbm_pim", reason)
    program = _program_of(compiled)
    root = _pimsim_root()
    geometry = samsung_simulator_geometry(root)
    _check_geometry(program.facts, geometry)
    _check_host_moves(compiled, program, inputs)
    return _run_samsung_program(compiled, program, root, geometry, inputs)


def _run_samsung_program(compiled, program, root, geometry, inputs) -> RunResult:
    """Execute one lowered program (one ``pim_driver`` invocation) on the
    runtime-role operands ``inputs`` (``A``/``B``/``C`` or ``IN0..IN2``)."""
    import numpy as np

    crf_words = list(compiled.cmds[program.start : program.end])
    supplied = sorted(k for k, v in inputs.items() if v is not None)

    if program.kind == "TEMPORAL_REDUCE":
        return _run_samsung_temporal(program, crf_words, root, geometry, inputs)

    if program.kind == "REDUCE":
        c = program.contraction
        A, B = inputs.get("A"), inputs.get("B")
        if A is None or B is None:
            raise ValueError(
                "Samsung REDUCE needs the bank-resident matrix (role 'A') and the "
                f"vector (role 'B'); got operands {supplied}"
            )
        A = np.asarray(A, dtype=np.float16)
        B = np.asarray(B, dtype=np.float16)
        if A.shape != c.matrix_shape or B.shape != c.vector_shape:
            raise ValueError(
                f"Samsung REDUCE operands have shapes A{A.shape}, B{B.shape}; the "
                f"compiled program reads A{c.matrix_shape}, B{c.vector_shape}"
            )
        matrix = (A if c.matrix_k_axis == 1 else A.T)[: program.rows, : c.K]
        if c.column_var is None:
            vectors = B[: c.K].reshape(1, c.K)
        elif c.vector_k_axis == 0:
            vectors = B[: c.K, : c.N].T
        else:
            vectors = B[: c.N, : c.K]
        res = _samsung_run_reduce(
            compiled, program, crf_words, root, geometry, matrix, vectors
        )
        rows = res["rows"]
        outputs = {"y": rows[0]} if c.column_var is None else {"out": rows.T}
        extra = {
            "kernel": "GENERIC_REDUCE",
            "returncode": 0,
            "outputs": outputs,
            "logical_shape": res["logical_shape"],
            "padded_shape": res["padded_shape"],
            "padding_fraction": res["padding_fraction"],
            "timing_scope": _REDUCE_TIMING_SCOPE,
            "cycle_phases": res["phases"],
            "plan": res["plan"]["text"],
            "crf": res["crf"],
            "conductor": res["conductor"],
            "row_map": res["row_map"],
            "simulator_geometry": geometry,
            "levers": dict(program.lever_record),
            "cycles_source": "PIMSimulator, Tenon conductor executing the uploaded CRF",
            "correctness_source": "PIMSimulator, Tenon conductor executing the uploaded CRF",
        }
        return RunResult(
            kind=RUN_KIND_SIMULATOR,
            cycles=res["cycles"],
            stdout=res["log"],
            backend="samsung_hbm_pim",
            extra=extra,
        )

    el = program.elementwise
    arrays = []
    for index in range(len(el.input_memrefs)):
        value = inputs.get(f"IN{index}")
        if value is None:
            raise ValueError(
                f"Samsung elementwise {el.op} needs input role 'IN{index}'; got "
                f"operands {supplied}"
            )
        value = np.asarray(value, dtype=np.float16).reshape(-1)
        if value.size != el.n:
            raise ValueError(
                f"Samsung elementwise input IN{index} has {value.size} elements; the "
                f"program reads {el.n}"
            )
        arrays.append(value)
    plan = samsung_eltwise_plan(program, geometry)
    padded = plan["padded_n"]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        paths = []
        for index, value in enumerate(arrays):
            buf = np.zeros(padded, dtype=np.float16)
            buf[: el.n] = value
            path = td / f"in{index}.npy"
            np.save(path, buf)
            paths.append(str(path))
        crf_text = _samsung_cmds_text(crf_words)
        (td / "crf.txt").write_text(crf_text)
        (td / "plan.txt").write_text(plan["text"])
        out_path = td / "out.bin"
        argv = [
            str(Path(root) / "pim_driver"),
            "--op",
            "GENERIC",
            "--op-kind",
            "ELTWISE",
            "--plan",
            str(td / "plan.txt"),
            "--crf",
            str(td / "crf.txt"),
            "--inputs",
            ",".join(paths),
            "--out",
            str(out_path),
        ]
        cycles, phases, conductor, log = _invoke_conductor(root, argv, "ELTWISE")
        flat = _samsung_read_bursts(out_path, padded // 16, "ELTWISE", log).reshape(-1)
    result = flat[: el.n].astype(np.float32)
    # Outputs are keyed by the destinations' own memref names (the producer's
    # and each accepted copy's), so no source name can collide with a fixed
    # key (C-25); the binding maps each name to its caller parameter.
    outputs = {el.output_memref: result}
    for sink in el.sink_memrefs:
        outputs[sink] = result.copy()
    extra = {
        "kernel": "GENERIC_ELTWISE",
        "returncode": 0,
        "outputs": outputs,
        "copies": list(el.sink_memrefs),
        "logical_shape": {"elements": el.n},
        "padded_shape": {"elements": padded},
        "padding_fraction": 1.0 - el.n / padded,
        "timing_scope": _ELTWISE_TIMING_SCOPE,
        "cycle_phases": phases,
        "plan": plan["text"],
        "crf": crf_text,
        "conductor": conductor,
        "simulator_geometry": geometry,
        "levers": dict(program.lever_record),
        "cycles_source": "PIMSimulator, Tenon conductor executing the uploaded CRF",
        "correctness_source": "PIMSimulator, Tenon conductor executing the uploaded CRF",
    }
    return RunResult(
        kind=RUN_KIND_SIMULATOR,
        cycles=cycles,
        stdout=log,
        backend="samsung_hbm_pim",
        extra=extra,
    )


def _contraction_operands(program, inputs):
    """The (M, K) matrix and (N, K) vectors of a contraction from roles A, B."""
    import numpy as np

    c = program.contraction
    A, B = inputs.get("A"), inputs.get("B")
    if A is None or B is None:
        supplied = sorted(k for k, v in inputs.items() if v is not None)
        raise ValueError(
            "Samsung contractions need the bank-resident matrix (role 'A') and the "
            f"vector (role 'B'); got operands {supplied}"
        )
    A = np.asarray(A, dtype=np.float16)
    B = np.asarray(B, dtype=np.float16)
    if A.shape != c.matrix_shape or B.shape != c.vector_shape:
        raise ValueError(
            f"Samsung contraction operands have shapes A{A.shape}, B{B.shape}; the "
            f"compiled program reads A{c.matrix_shape}, B{c.vector_shape}"
        )
    matrix = (A if c.matrix_k_axis == 1 else A.T)[: program.rows, : c.K]
    if c.column_var is None:
        vectors = B[: c.K].reshape(1, c.K)
    elif c.vector_k_axis == 0:
        vectors = B[: c.K, : c.N].T
    else:
        vectors = B[: c.N, : c.K]
    return matrix, vectors


def _run_samsung_temporal(program, crf_words, root, geometry, inputs) -> RunResult:
    import numpy as np

    c = program.contraction
    matrix, vectors = _contraction_operands(program, inputs)
    M, N = program.rows, int(vectors.shape[0])
    if c.accumulator_init == "carried":
        init = inputs.get("C")
        if init is None:
            raise ValueError(
                "the ordered Samsung program carries the output's incoming value "
                "as its initial accumulator; the runtime role 'C' (the output "
                "array) was not supplied"
            )
        init = np.asarray(init, dtype=np.float16)
        initial = init.reshape(M, 1) if c.column_var is None else init.reshape(M, N)
    else:
        initial = np.zeros((M, N), dtype=np.float16)
    slots = None
    if program.row_slots is not None:
        # The ordered conductor gives physical lane slot s to input row s; the
        # owner map is realized by staging row r at slot row_slots[r] and
        # reading it back from there.
        slots = np.asarray(program.row_slots, dtype=np.int64)
        staged, staged_init = np.empty_like(matrix), np.empty_like(initial)
        staged[slots], staged_init[slots] = matrix, initial
        matrix, initial = staged, staged_init
    res = _samsung_run_temporal(
        program, crf_words, root, geometry, matrix, vectors, initial
    )
    rows = res["rows"] if slots is None else res["rows"][slots]
    tile_rows = geometry["channels"] * program.facts.pim_blocks * program.facts.lanes

    def lane(slot):
        return {
            "tile": slot // tile_rows,
            "channel": slot % tile_rows // 16 // program.facts.pim_blocks,
            "block": slot % tile_rows // 16 % program.facts.pim_blocks,
            "lane": slot % 16,
        }

    row_map = {
        "map": program.row_map,
        "digest": None if slots is None else _row_map_digest(program.row_slots),
        "placement": {
            row: lane(row if slots is None else int(slots[row]))
            for row in sorted({0, 1, 2, 31, 32, M // 2, M - 1} & set(range(M)))
        },
        "applied_by": "host staging (TEMPORAL_REDUCE inputs in lane-slot order)",
    }
    outputs = {"y": rows[:, 0]} if c.column_var is None else {"out": rows}
    extra = {
        "kernel": "GENERIC_TEMPORAL_REDUCE",
        "returncode": 0,
        "outputs": outputs,
        "logical_shape": res["logical_shape"],
        "padded_shape": res["padded_shape"],
        "padding_fraction": res["padding_fraction"],
        "timing_scope": _TEMPORAL_TIMING_SCOPE,
        "cycle_phases": res["phases"],
        "crf": res["crf"],
        "conductor": res["conductor"],
        "row_map": row_map,
        "simulator_geometry": geometry,
        "levers": dict(program.lever_record),
        "accumulation": "source-order FP16 (one GRF_B entry per output row, "
        "K issued in ascending order, no host arithmetic)",
        "cycles_source": "PIMSimulator, TEMPORAL_REDUCE conductor executing the uploaded CRF",
        "correctness_source": "PIMSimulator, TEMPORAL_REDUCE conductor executing the uploaded CRF",
    }
    return RunResult(
        kind=RUN_KIND_SIMULATOR,
        cycles=res["cycles"],
        stdout=res["log"],
        backend="samsung_hbm_pim",
        extra=extra,
    )


# --------------------------------------------------------------------- #
# Multi-kernel artifacts: one conductor invocation per launch
# --------------------------------------------------------------------- #

_LAUNCHES_TIMING_SCOPE = (
    "Sum over the host program's launches, run back to back without overlap, "
    "of each launch's pim_driver invocation (PIMSimulator memory-clock cycles; "
    "each launch states its own scope in extra['launches'][i]['timing_scope']: "
    "staging of the operands the host program scattered for it, CRF upload, "
    "execution with its host GRF_A broadcasts, readback of its result), plus "
    "the cost-graph latency of host steps no invocation performs "
    "(extra['host_steps'] entries realized_by 'modelled'). Not timed: host "
    "arithmetic between launches (the 16-lane sum of a REDUCE result, the "
    "rounding of a gathered buffer to its FP16 host copy) and host padding."
)

_CONTRACTION_KINDS = ("REDUCE", "TEMPORAL_REDUCE")


def _launch_buffers(compiled, program, names) -> dict:
    """Workload parameters one lowered kernel reads and writes, by runtime role.

    Binds through the trace's retained source identities: an operand whose
    ``value_ref`` is a parameter's ``source_value_refs`` entry is that
    parameter.  A contraction stores through a local accumulator, so its
    output is the parameter bound to an argument of the kernel's own
    functions that none of its operands read (``_build_ir_value_refs`` names
    argument ``i`` of function ``f`` ``(f, 0, i)`` and values inside ``f``
    ``(f, 1|2, ...)``).  Returns ``{"A", "B"[, "C"], "out"}`` for a
    contraction, ``{"IN0".., "out"}`` for an elementwise kernel.
    """
    from .spmw_autoschedule import _matcher_work_scope

    trace = compiled.trace
    refs = getattr(trace, "source_value_refs", None) or {}
    name_of = {
        ref: names[int(o)] for o, ref in refs.items() if 0 <= int(o) < len(names)
    }
    matches = [
        m for m in trace.matches if _matcher_work_scope(m).group_id == program.group_id
    ]
    kernel = (
        matches[0].func_name.rsplit("_", 1)[0] if matches else str(program.group_id)
    )

    def bind(what, value_refs):
        found = {name_of.get(ref) for ref in value_refs}
        if len(found) != 1 or None in found:
            raise ValueError(
                f"Samsung multi-kernel execution: the {what} of kernel {kernel!r} "
                f"binds to {sorted(str(n) for n in found)}, not to one workload "
                "parameter"
            )
        return found.pop()

    def operand_refs(role):
        return [op.value_ref for m in matches for op in m.operands if op.role == role]

    if program.kind in _CONTRACTION_KINDS:
        c = program.contraction
        roles = {
            "A": bind("matrix", operand_refs(c.matrix_role)),
            "B": bind("vector", operand_refs(c.vector_role)),
        }
    else:
        roles = {
            f"IN{index}": bind(f"input {role!r}", operand_refs(role))
            for index, role in enumerate(program.elementwise.input_roles)
        }
    results = {name_of.get(m.result_value_ref) for m in matches} - {None}
    if len(results) == 1:
        roles["out"] = results.pop()
    else:
        read = {op.value_ref for m in matches for op in m.operands}
        functions = {
            ref.path[0]
            for m in matches
            for ref in (m.result_value_ref, *(op.value_ref for op in m.operands))
            if ref is not None and len(ref.path) > 1 and ref.path[1] != 0
        }
        roles["out"] = bind(
            "output",
            [
                ref
                for ref in name_of
                if len(ref.path) == 3
                and ref.path[1] == 0
                and ref.path[0] in functions
                and ref not in read
            ]
            or [None],
        )
    if (
        program.kind in _CONTRACTION_KINDS
        and program.contraction.accumulator_init == "carried"
    ):
        roles["C"] = roles["out"]
    return roles


def _host_step_activity_cycles(compiled, schedule) -> dict:
    """Cost-graph latency of each host step (``host_index -> cycles``), or
    an empty map when the artifact retains no execution graph."""
    from .spmw_plan import HostStep

    graph = getattr(compiled, "execution_graph", None)
    if graph is None:
        return {}
    counts = {"ingress": 0, "egress": 0}
    cycles = {}
    for step in schedule.steps:
        if not isinstance(step, HostStep):
            continue
        resolved = compiled.host_moves[step.host_index]
        direction = "egress" if resolved.verb.name == "gather" else "ingress"
        prefix = f"host:{direction}:{counts[direction]}:"
        counts[direction] += 1
        cycles[step.host_index] = sum(
            int(a.latency_cycles) for a in graph.activities if a.id.startswith(prefix)
        )
    return cycles


def samsung_launch_plan(compiled) -> dict:
    """Bind a multi-kernel artifact's launch schedule to conductor invocations.

    Pure (no simulator).  Walks the host program in order:

    * ``scatter(b)`` / ``broadcast(b)`` snapshots the host copy of ``b`` and
      stages it for the next launch that reads ``b``.  The selected program
      decides the verb: the matrix (and the initial accumulator of a carried
      ordered program, and every elementwise input) is bank-resident, so it
      needs a scatter; the vector needs a broadcast under host GRF_A fill and
      a scatter under CRF fill.  The staged copy is placed by that launch's
      invocation (preload phase for a scatter, execution phase for a
      broadcast).
    * A launch consumes one staging of each operand and runs its kernel's
      program in its own ``pim_driver`` invocation; its result becomes the
      device value of its output buffer.
    * ``gather(b)`` takes the device value of ``b`` (the readback of the
      launch that produced it) into the host copy.

    The conductor keeps nothing between invocations: banks, GRF and CRF
    start empty in every run.  A launch operand that is not staged for it is
    therefore rejected, naming the residency the program assumed (a result
    left on the device for the next kernel, a staging reused by a second
    launch, a buffer declared resident before the program), as is a staging
    that would read a host copy older than a device result.  Host steps no
    invocation performs (a staging no launch reads, a repeated gather) are
    priced from the retained execution graph.

    Returns ``events`` (the execution order), ``launches`` (one record per
    launch), ``host_steps`` (one per host transfer, with ``realized_by``),
    and ``external_inputs`` (buffers staged from the caller's arrays).
    """
    from .spmw_plan import HostStep, LaunchStep

    programs = list(
        getattr(getattr(compiled, "layout_ctx", None), "programs", ()) or ()
    )
    schedule = getattr(compiled, "launch_schedule", None)
    if schedule is None or not any(isinstance(s, LaunchStep) for s in schedule.steps):
        raise NotImplementedError(
            f"this Samsung artifact has {len(programs)} kernels and no launch "
            "schedule; multi-kernel execution follows the host program's "
            "launches: compile with host_moves=<an @allo.host_program that "
            "launches every kernel and declares its transfers>"
        )
    names = tuple(name for name, _shape in schedule.parameters)
    by_group = {program.group_id: program for program in programs}
    activity_cycles = _host_step_activity_cycles(compiled, schedule)

    events, launches, host_steps = [], [], {}
    staged = {}  # buffer -> host_index staged and not yet read by a launch
    consumed = {}  # buffer -> (host_index, launch) of its last staging read
    produced = {}  # buffer -> launch whose device value is current
    gathered = {}  # buffer -> launch whose value the host copy holds
    external = []

    def model(host_index, reason):
        record = host_steps[host_index]
        record["realized_by"] = "modelled"
        record["reason"] = reason
        record["modelled_cycles"] = activity_cycles.get(host_index)

    def residency_error(index, kernel, role_text, buffer, verb):
        where = f"launch {index} ({kernel!r}) reads {buffer!r} as its {role_text}"
        keep = (
            "The Samsung conductor runs one program per pim_driver invocation "
            "and keeps no bank, GRF or CRF contents between invocations"
        )
        if buffer in produced and gathered.get(buffer) != produced[buffer]:
            source = launches[produced[buffer]]
            return ValueError(
                f"{where}, which launch {source['launch']} ({source['kernel']!r}) "
                f"left on the device: the host program keeps {buffer!r} "
                "device-resident across the two launches (no gather and "
                f"{verb} between them). {keep}, and the result drains to the "
                f"host; declare gather({buffer}) and {verb}({buffer}) between "
                "the launches"
            )
        if buffer in consumed:
            host_index, reader = consumed[buffer]
            return ValueError(
                f"{where}, but host step {host_index} staged it for launch "
                f"{reader} and no host step stages it again: the program keeps "
                f"{buffer!r} device-resident across launches. {keep}; "
                f"{verb} {buffer!r} again before launch {index}"
            )
        if buffer in (getattr(schedule, "resident", ()) or ()):
            return ValueError(
                f"{where}; the host program declares {buffer!r} resident before "
                f"it runs. {keep}, so it cannot honour that residency; {verb} "
                f"{buffer!r} before launch {index}"
            )
        return ValueError(
            f"{where}, but no host step stages it before the launch; {verb} "
            f"{buffer!r} before launch {index}"
        )

    for step in schedule.steps:
        if isinstance(step, HostStep):
            index = step.host_index
            resolved = compiled.host_moves[index]
            verb = resolved.verb.name
            buffer = resolved.buffer_role
            host_steps[index] = {
                "host_index": index,
                "verb": verb,
                "buffer": buffer,
                "realized_by": None,
            }
            if not isinstance(buffer, str) or buffer not in names:
                raise ValueError(
                    f"Samsung multi-kernel execution moves whole workload buffers "
                    f"{list(names)}; host step {index} ({verb}) moves {buffer!r}"
                )
            if verb == "gather":
                if buffer not in produced:
                    raise ValueError(
                        f"host step {index} gathers {buffer!r}, which no earlier "
                        "launch produced on the device"
                    )
                if gathered.get(buffer) == produced[buffer]:
                    model(index, f"the host already holds this value of {buffer!r}")
                else:
                    gathered[buffer] = produced[buffer]
                    host_steps[index]["realized_by"] = {
                        "launch": produced[buffer],
                        "phase": "readback",
                    }
                    launches[produced[buffer]]["gathered_by"].append(index)
                events.append(("gather", index, buffer))
            elif verb in ("scatter", "broadcast"):
                if buffer in produced and gathered.get(buffer) != produced[buffer]:
                    raise ValueError(
                        f"host step {index} ({verb}) stages the host copy of "
                        f"{buffer!r}, but launch {produced[buffer]} produced "
                        f"{buffer!r} on the device and the host program has not "
                        "gathered it before this step"
                    )
                if buffer in staged:
                    model(
                        staged[buffer],
                        f"host step {index} stages {buffer!r} again before any "
                        "launch reads this staging",
                    )
                if buffer not in gathered and buffer not in external:
                    external.append(buffer)
                staged[buffer] = index
                events.append(("stage", index, buffer))
            else:
                raise ValueError(
                    f"Samsung multi-kernel execution: host step {index} has verb "
                    f"{verb!r}; the conductor stages by scatter/broadcast and "
                    "reads back by gather"
                )
        elif isinstance(step, LaunchStep):
            program = by_group.get(step.group_id)
            if program is None:
                raise ValueError(
                    f"launch of {step.kernel!r} names kernel group {step.group_id}, "
                    "which has no lowered Samsung program"
                )
            index = len(launches)
            roles = _launch_buffers(compiled, program, names)
            contraction = program.kind in _CONTRACTION_KINDS
            operands = {}
            for role, buffer in roles.items():
                if role == "out":
                    continue
                if contraction and role == "B" and not program.vector_from_bank:
                    need, phase, text = "broadcast", "exec", "vector (host GRF_A fill)"
                elif contraction and role == "B":
                    need, phase, text = "scatter", "preload", "vector (CRF GRF_A fill)"
                elif contraction and role == "A":
                    need, phase, text = "scatter", "preload", "bank-resident matrix"
                elif role == "C":
                    need, phase, text = "scatter", "preload", "initial accumulator"
                else:
                    need, phase, text = "scatter", "preload", f"bank operand {role}"
                host_index = staged.pop(buffer, None)
                if host_index is None:
                    raise residency_error(index, step.kernel, text, buffer, need)
                verb = host_steps[host_index]["verb"]
                if verb != need:
                    raise ValueError(
                        f"launch {index} ({step.kernel!r}) reads {buffer!r} as its "
                        f"{text}, which its selected program takes by {need}; "
                        f"host step {host_index} declares {verb}"
                    )
                host_steps[host_index]["realized_by"] = {
                    "launch": index,
                    "phase": phase,
                }
                consumed[buffer] = (host_index, index)
                operands[role] = (buffer, host_index)
            produced[roles["out"]] = index
            launches.append(
                {
                    "launch": index,
                    "kernel": step.kernel,
                    "group_id": step.group_id,
                    "ordinal": step.ordinal,
                    "program": program.kind,
                    "operands": operands,
                    "output": roles["out"],
                    "gathered_by": [],
                }
            )
            events.append(("launch", index))
        else:
            raise TypeError(f"unknown launch schedule step {step!r}")
    for buffer, host_index in staged.items():
        model(host_index, f"no launch reads this staging of {buffer!r}")
    return {
        "events": events,
        "launches": launches,
        "host_steps": [host_steps[i] for i in sorted(host_steps)],
        "external_inputs": external,
        "parameters": dict(schedule.parameters),
    }


def _run_samsung_launches(compiled, inputs) -> RunResult:
    """Execute a multi-kernel artifact: ``samsung_launch_plan``'s launches, one
    ``pim_driver`` invocation each, threading gathered results into later
    stagings.  ``inputs`` are named by workload parameter."""
    import numpy as np

    plan = samsung_launch_plan(compiled)
    missing = [b for b in plan["external_inputs"] if inputs.get(b) is None]
    if missing:
        raise ValueError(
            f"Samsung multi-kernel execution stages {missing} from the caller; "
            f"supplied {sorted(k for k, v in inputs.items() if v is not None)}"
        )
    modelled = [s for s in plan["host_steps"] if s["realized_by"] == "modelled"]
    unpriced = [s["host_index"] for s in modelled if s["modelled_cycles"] is None]
    if unpriced:
        raise RuntimeError(
            f"host steps {unpriced} are performed by no invocation and the "
            "artifact retains no execution graph to price them"
        )
    reason = _samsung_unavailable_reason()
    if reason is not None:
        raise SimulatorUnavailable("samsung_hbm_pim", reason)
    root = _pimsim_root()
    geometry = samsung_simulator_geometry(root)
    by_group = {p.group_id: p for p in compiled.layout_ctx.programs}
    for program in by_group.values():
        _check_geometry(program.facts, geometry)

    host, snapshots, device, outputs = {}, {}, {}, {}
    records, logs = [], []
    for event in plan["events"]:
        if event[0] == "stage":
            _kind, host_index, buffer = event
            value = host[buffer] if buffer in host else inputs[buffer]
            snapshots[host_index] = np.array(value, dtype=np.float16, copy=True)
        elif event[0] == "launch":
            launch = plan["launches"][event[1]]
            program = by_group[launch["group_id"]]
            role_inputs = {
                role: snapshots[host_index]
                for role, (_buffer, host_index) in launch["operands"].items()
            }
            res = _run_samsung_program(compiled, program, root, geometry, role_inputs)
            (value,) = res.extra["outputs"].values()
            device[launch["output"]] = np.asarray(value)
            extra = res.extra
            record = {
                key: launch[key]
                for key in (
                    "launch",
                    "kernel",
                    "group_id",
                    "ordinal",
                    "program",
                    "output",
                    "gathered_by",
                )
            }
            record.update(
                operands={
                    role: buffer for role, (buffer, _h) in launch["operands"].items()
                },
                staged_by={
                    role: h for role, (_buffer, h) in launch["operands"].items()
                },
                cycles=res.cycles,
                **{
                    key: extra[key]
                    for key in (
                        "timing_scope",
                        "cycle_phases",
                        "conductor",
                        "crf",
                        "plan",
                        "levers",
                        "logical_shape",
                        "padded_shape",
                        "padding_fraction",
                        "row_map",
                        "accumulation",
                    )
                    if key in extra
                },
            )
            record["kernel_route"] = extra.get("kernel")
            records.append(record)
            logs.append(
                f"--- launch {launch['launch']}: {launch['kernel']} ---\n{res.stdout}"
            )
        else:
            _kind, _host_index, buffer = event
            destination = inputs.get(buffer)
            dtype = (
                destination.dtype if isinstance(destination, np.ndarray) else np.float16
            )
            value = device[buffer]
            shape = plan["parameters"].get(buffer)
            if shape is not None and value.size == math.prod(shape):
                value = value.reshape(shape)
            host[buffer] = value.astype(dtype)
            outputs[buffer] = host[buffer]
    launch_cycles = sum(r["cycles"] for r in records)
    modelled_cycles = sum(s["modelled_cycles"] for s in modelled)
    return RunResult(
        kind=RUN_KIND_SIMULATOR,
        cycles=launch_cycles + modelled_cycles,
        stdout="\n".join(logs),
        backend="samsung_hbm_pim",
        extra={
            "kernel": "SAMSUNG_LAUNCH_SCHEDULE",
            "returncode": 0,
            "outputs": outputs,
            "launches": records,
            "host_steps": plan["host_steps"],
            "timing_scope": _LAUNCHES_TIMING_SCOPE,
            "cycle_phases": {
                "launches": launch_cycles,
                "modelled_host_steps": modelled_cycles,
            },
            "simulator_geometry": geometry,
            "cycles_source": "PIMSimulator, one Tenon conductor invocation per launch, "
            "plus the cost graph for host steps no invocation performs",
            "correctness_source": "PIMSimulator, one Tenon conductor invocation per launch",
        },
    )


def _layout_stage_resident(layout) -> bool:
    """The chosen placement's `stage_resident` flag (False when absent)."""
    if isinstance(layout, list):
        layout = layout[0] if layout else None
    if layout is None:
        return False
    return bool(getattr(layout, "extra", {}).get("stage_resident", False))


def _run_samsung_batched(
    compiled: "Compiled", W, X, compare_native: bool = True
) -> RunResult:
    """Run the compiled GEMV for B input vectors X[B, K] against one matrix W.

    The same emitted CRF runs B column passes; the selected placement's
    `stage_resident` decides whether the matrix is staged once (resident) or
    re-staged with a CRF upload for every vector (restage).  With
    `compare_native`, the restage sequencing is also run as the comparator.
    Outputs (`Y`, shape [B, M]) are read back and returned.
    """
    import numpy as np

    reason = _samsung_unavailable_reason()
    if reason is not None:
        raise SimulatorUnavailable("samsung_hbm_pim", reason)
    program = _program_of(compiled)
    if program.kind != "REDUCE" or program.contraction.column_var is not None:
        raise NotImplementedError("run_batched runs a compiled Samsung GEMV")
    c = program.contraction
    root = _pimsim_root()
    geometry = samsung_simulator_geometry(root)
    _check_geometry(program.facts, geometry)
    crf_words = list(compiled.cmds[program.start : program.end])
    W = np.asarray(W, dtype=np.float16)
    X = np.asarray(X, dtype=np.float16)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if W.shape != c.matrix_shape or X.shape[1:] != c.vector_shape:
        raise ValueError(
            f"run_batched operands W{W.shape}, X{X.shape} do not match the compiled "
            f"GEMV (W{c.matrix_shape}, X[B, {c.vector_shape[0]}])"
        )
    matrix = (W if c.matrix_k_axis == 1 else W.T)[: program.rows, : c.K]
    resident = program.weight_resident
    tenon = _samsung_run_reduce(
        compiled, program, crf_words, root, geometry, matrix, X[:, : c.K], resident
    )
    extra = {
        "kernel": "GENERIC_REDUCE",
        "batch": int(X.shape[0]),
        "stage_resident": resident,
        "tenon_total": tenon["cycles"],
        "tenon_phases": tenon["phases"],
        "outputs": {"Y": tenon["rows"]},
        "logical_shape": tenon["logical_shape"],
        "padded_shape": tenon["padded_shape"],
        "padding_fraction": tenon["padding_fraction"],
        "timing_scope": _REDUCE_TIMING_SCOPE,
        "plan": tenon["plan"]["text"],
        "crf": tenon["crf"],
        "conductor": tenon["conductor"],
    }
    stdout = tenon["log"]
    if compare_native:
        native = _samsung_run_reduce(
            compiled, program, crf_words, root, geometry, matrix, X[:, : c.K], False
        )
        extra["native_total"] = native["cycles"]
        extra["native_phases"] = native["phases"]
        stdout += "\n--- restage comparator ---\n" + native["log"]
    return RunResult(
        kind=RUN_KIND_SIMULATOR,
        cycles=tenon["cycles"],
        stdout=stdout,
        backend="samsung_hbm_pim",
        extra=extra,
    )
