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


def _is_opaque(coord, loop_vars):
    """An SSA value the matcher retains only by name (not a loop variable)."""
    return re.fullmatch(r"%\w+", coord.strip()) is not None and coord.strip() not in loop_vars


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
    (matrix_row,) = free[matrix.role]
    # The matrix row coordinate: the row loop plus a constant (checked against
    # the store below), a constant, or an SSA value the matcher retains only
    # by name (then the store row is trusted; see the design note).
    if _is_opaque(matrix_row, loop_vars):
        matrix_row_offset = None
    elif row_var is not None and _affine_offset(matrix_row, row_var) is not None:
        matrix_row_offset = _affine_offset(matrix_row, row_var)
    elif row_var is None and re.fullmatch(r"\d+", matrix_row.strip()):
        matrix_row_offset = int(matrix_row)
    else:
        reject(
            f"the matrix row coordinate {matrix_row!r} is not the output row "
            f"({row_var or 'a constant'}) plus a constant"
        )
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
    output_memref, sink_text = sinks[0]
    dims = _split_top_level(sink_text)
    if len(dims) != (1 if column_var is None else 2):
        reject(f"store index {sink_text!r} does not match the loop nest")
    if row_var is None:
        row_offset = int(dims[0]) if re.fullmatch(r"\d+", dims[0].strip()) else None
    else:
        row_offset = _affine_offset(dims[0], row_var)
    if row_offset is None or row_offset < 0:
        reject(
            f"store row index {dims[0]!r} is not the row loop "
            f"({row_var or 'none'}) plus a constant"
        )
    if matrix_row_offset is not None and matrix_row_offset != row_offset:
        reject(
            f"the matrix row coordinate {matrix_row!r} and the store row "
            f"{dims[0]!r} differ"
        )
    if column_var is not None and dims[1].strip() != column_var:
        reject(f"store column index {dims[1]!r} is not the column loop {column_var}")
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
    identity = (None, "affine_map<(d0) -> (d0)>")
    inputs = [o for o in m.operands if not o.is_loop_carried]
    if any(o.is_loop_carried for o in m.operands):
        reject("an elementwise op cannot carry an accumulator")
    for o in inputs:
        dims, elt = _memref_dims(o.memref_type)
        if o.indices != [var] or o.index_map not in identity:
            reject(
                f"input {o.memref_name!r} is not indexed by the element loop {var} "
                f"(index {o.indices}); a partitioned work grid indexes elements "
                "through expressions the lowering cannot prove elementwise, so write "
                "one work item whose loop walks the whole extent"
            )
        if dims != (n,) or elt != "f16":
            reject(f"input {o.memref_name!r} must be a 1-D FP16 buffer of {n} elements")
    if list(m.extra.get("store_indices") or []) != [var] or m.result_memref_name is None:
        reject(f"the result must be stored at the element loop index {var}")
    if len({o.memref_name for o in inputs}) != len(inputs):
        reject("one buffer bound to several inputs is not supported")
    return SamsungElementwise(
        op=op,
        input_memrefs=tuple(o.memref_name for o in inputs),
        input_roles=tuple(o.role for o in inputs),
        output_memref=m.result_memref_name,
        n=n,
    )


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

    def __init__(self, target):
        super().__init__(target)
        # Host-move bookkeeping: the host-scope moves' `emit` closures append
        # `(hook, device_handle)` records here. The conductor performs the
        # transfers; these records carry cost and operand-role information.
        self.host_moves_emitted: list[tuple[str, object]] = []
        self.programs: list[SamsungProgram] = []

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

    def emit_program(self, trace, layouts, *, schedule=None, host_moves=(), module=None):
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
            for (scope, _matches), layout in zip(_bucket_for_autoschedule(trace), layouts)
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
                            work_id, sum(1 for m in matches if m.target_op_name == "MAC")
                        )
                    )

    def _check_unrealized_levers(self, layout):
        extra = layout.extra or {}
        tile = extra.get("tile")
        if tile is not None and not getattr(tile, "is_identity", True):
            raise _infeasible("the Samsung lowering does not realize a non-identity tile")
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
        return tuple(slots), f"permuted ({moved} of {rows_total} rows moved, {R} per unit)"

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
        tile_fiber = tuple(fibers[t % len(fibers)][0] for t in range(T))
        residency = (layout.extra or {}).get("grf_residency", {}) or {}
        vector_from_bank = residency.get(rep.vector_memref, "crf") != "host"
        mac = target.op("MAC")

        start = len(self.cmds)
        binds = []
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
        extra = layout.extra or {}
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
            },
            row_slots=row_slots,
            row_map=row_map,
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


def samsung_slot_position(slot: int, geometry: dict, facts, rows_per_block: int) -> dict:
    """Physical position of one REDUCE slot: pass, channel, PIM block, GRF_B entry."""
    C, P, G = geometry["channels"], facts.pim_blocks, rows_per_block
    return {
        "pass": slot // (C * P * G),
        "channel": slot // (P * G) % C,
        "block": slot // G % P,
        "entry": slot % G,
    }


def samsung_reduce_plan(program: SamsungProgram, geometry: dict, columns: int,
                        weight_resident: bool | None = None) -> dict:
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
    lines = _plan_header("ELTWISE", facts, geometry) + [
        f"tiles {tiles}",
        f"inputs {n_inputs}",
        "input_rows " + " ".join(str(r) for r in input_rows),
        "input_swap_parity " + " ".join(str(int(s)) for s in program.swap_parity),
        f"output_row {n_inputs * rows_per_operand}",
    ] + _plan_binds(program)
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
        raise RuntimeError(f"Samsung pim_driver {what} invocation failed: {exc}") from exc
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
        raise RuntimeError(f"Samsung pim_driver {what} reported no cycles: {combined[-600:]}")
    receipt = re.search(r"TENON_CONDUCTOR (.*)", combined)
    conductor = (
        {k: (int(v) if v.isdigit() else v) for k, v in re.findall(r"(\w+)=(\S+)", receipt.group(1))}
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


def _samsung_run_reduce(compiled, program, crf_words, root, geometry, matrix, vectors,
                        weight_resident=None):
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
            "--op", "GENERIC",
            "--op-kind", "REDUCE",
            "--plan", str(plan_path),
            "--crf", str(crf_path),
            "--inputs", f"{a_path},{b_path}",
            "--out", str(out_path),
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
            row: samsung_slot_position(slot_of[row], geometry, facts, program.rows_per_block)
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
            "--op", "GENERIC",
            "--op-kind", "TEMPORAL_REDUCE",
            "--cmds", str(crf_path),
            "--inputs", f"{a_path},{b_path},{i_path}",
            "--out-rows", str(M),
            "--reduce-k", str(K),
            "--out-cols", str(N),
            "--temporal-audit", str(td / "audit"),
            "--out", str(out_path),
        ]
        cycles, phases, _conductor, log = _invoke_conductor(root, argv, "TEMPORAL_REDUCE")
        bursts = _samsung_read_bursts(out_path, (M * N + 15) // 16, "TEMPORAL_REDUCE", log)
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
            f"Samsung execution runs one kernel per artifact; this one has "
            f"{len(programs)} kernels (multi-kernel Samsung programs are compile-only)"
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
        if verb == "scatter" and not (is_matrix or (is_vector and program.vector_from_bank)):
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
    """
    import numpy as np

    reason = _samsung_unavailable_reason()
    if reason is not None:
        raise SimulatorUnavailable("samsung_hbm_pim", reason)
    program = _program_of(compiled)
    root = _pimsim_root()
    geometry = samsung_simulator_geometry(root)
    _check_geometry(program.facts, geometry)
    _check_host_moves(compiled, program, inputs)
    crf_words = list(compiled.cmds[program.start:program.end])
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
        res = _samsung_run_reduce(compiled, program, crf_words, root, geometry, matrix, vectors)
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
            "--op", "GENERIC",
            "--op-kind", "ELTWISE",
            "--plan", str(td / "plan.txt"),
            "--crf", str(td / "crf.txt"),
            "--inputs", ",".join(paths),
            "--out", str(out_path),
        ]
        cycles, phases, conductor, log = _invoke_conductor(root, argv, "ELTWISE")
        flat = _samsung_read_bursts(out_path, padded // 16, "ELTWISE", log).reshape(-1)
    extra = {
        "kernel": "GENERIC_ELTWISE",
        "returncode": 0,
        "outputs": {"out": flat[: el.n].astype(np.float32)},
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
        initial = (init.reshape(M, 1) if c.column_var is None else init.reshape(M, N))
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
    res = _samsung_run_temporal(program, crf_words, root, geometry, matrix, vectors, initial)
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


def _layout_stage_resident(layout) -> bool:
    """The chosen placement's `stage_resident` flag (False when absent)."""
    if isinstance(layout, list):
        layout = layout[0] if layout else None
    if layout is None:
        return False
    return bool(getattr(layout, "extra", {}).get("stage_resident", False))


def _run_samsung_batched(compiled: "Compiled", W, X, compare_native: bool = True) -> RunResult:
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
    crf_words = list(compiled.cmds[program.start:program.end])
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
