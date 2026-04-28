# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SPMW codegen — generic walker over target.move / target.op declarations.

The walker dispatches each `MatchedOp` to its target op's `emit`
callback with a backend-specific `ctx`, per report 16. Each backend
supplies its own `CodegenContext` subclass; this module supplies a
Samsung HBM-PIM ctx (`SamsungCtx`) and a placeholder layout map for
the report-16 GEMV shape.

Several pieces are intentionally still stubbed and raise structured
`NotImplementedError`s when the walker hits them — see the BLOCKER
comments below and report 16 for the design.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .spmw_match import MatchTrace, MatchedOp
from .spmw_target import MemoryRef, Register, SymExpr, UnitId


# --------------------------------------------------------------------- #
# Target-neutral instruction record
# --------------------------------------------------------------------- #


@dataclass
class PIMCmd:
    """Python mirror of DRAMSim::PIMCmd from PIMSimulator/src/PIMCmd.h.

    Used as the codegen output unit — a static record that can be
    compared field-by-field against the canonical Samsung kernel from
    PIMCmdGen.h without booting the simulator. Field names mirror the
    C++ struct's trailing-underscore convention.
    """

    type_: str  # opcode name, e.g. "MAC", "MOV", "JUMP", "NOP", "EXIT"
    dst_: str = "A_OUT"  # PIMOpdType name
    src0_: str = "A_OUT"
    src1_: str = "A_OUT"
    src2_: str = "A_OUT"
    loopCounter_: int = 0
    loopOffset_: int = 0
    isAuto_: int = 0
    dstIdx_: int = 0
    src0Idx_: int = 0
    src1Idx_: int = 0
    isRelu_: int = 0


# --------------------------------------------------------------------- #
# Generic walker
# --------------------------------------------------------------------- #


class CodegenContext:
    """Backend-specific instruction builder.

    Each backend supplies its own subclass exposing the methods its
    `emit` lambdas need (Samsung: `ctx.cmd(name, dst, src0, src1)`).
    """

    def __init__(self, target):
        self.target = target
        self.cmds: list[PIMCmd] = []

    def cmd(self, name: str, **fields):  # pragma: no cover - abstract
        raise NotImplementedError(
            f"{type(self).__name__}.cmd is not implemented; "
            "each backend must subclass CodegenContext and define cmd()."
        )


# --------------------------------------------------------------------- #
# Samsung HBM-PIM backend
# --------------------------------------------------------------------- #


def _bank_parity(idx) -> str | None:
    """Classify a `MemoryRef.idx` SymExpr as the even or odd bank pattern.

    Returns "EVEN_BANK" for `2*X`, "ODD_BANK" for `2*X+1`, None otherwise.
    Codegen for any other indexing form is the autoscheduler's problem
    (Blocker 4).
    """
    if isinstance(idx, SymExpr) and idx.op == "mul":
        a, b = idx.args
        if (a == 2 and isinstance(b, UnitId)) or (b == 2 and isinstance(a, UnitId)):
            return "EVEN_BANK"
    if isinstance(idx, SymExpr) and idx.op == "add":
        a, b = idx.args
        # `2*X + 1` — match either argument order.
        for x, y in ((a, b), (b, a)):
            if y == 1 and isinstance(x, SymExpr) and x.op == "mul":
                m, n = x.args
                if (m == 2 and isinstance(n, UnitId)) or (n == 2 and isinstance(m, UnitId)):
                    return "ODD_BANK"
    return None


class SamsungCtx(CodegenContext):
    """Codegen ctx for Samsung HBM-PIM.

    Translates Tenon `Register` / `MemoryRef` handles into Samsung's
    `PIMOpdType` enum names (as strings), infers the is_auto column-
    strobe flag from operand shapes, and accumulates the result as a
    `list[PIMCmd]`.

    Recognized opcodes: "MAC", "MUL", "ADD", "MAD", "MOV", "FILL",
    "NOP", "JUMP", "EXIT" — i.e. the `PIMCmdType` enum from PIMCmd.h.
    """

    _REG_TO_OPD = {"grf_a": "GRF_A", "grf_b": "GRF_B"}

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
            parity = _bank_parity(handle.idx)
            if parity is None:
                raise NotImplementedError(
                    f"SamsungCtx: memref index {handle.idx!r} is not the "
                    "canonical 2*pid / 2*pid+1 even/odd bank form. Real "
                    "layout decisions are the autoscheduler's job (Blocker 4)."
                )
            return (parity, 0)
        raise NotImplementedError(
            f"SamsungCtx: unknown handle type {type(handle).__name__}."
        )

    def cmd(self, name, dst=None, src0=None, src1=None):
        dst_name, dst_idx = self._opd(dst)
        src0_name, src0_idx = self._opd(src0)
        src1_name, src1_idx = self._opd(src1)
        # Samsung's "is_auto" column-strobe modifier: set when src1 is
        # bank-shaped (the bank stream is read in burst mode while the
        # GRF source stays put). Inferred per report 16 §"is_auto".
        is_auto = 1 if isinstance(src1, MemoryRef) else 0
        self.cmds.append(
            PIMCmd(
                type_=name,
                dst_=dst_name,
                dstIdx_=dst_idx,
                src0_=src0_name,
                src0Idx_=src0_idx,
                src1_=src1_name,
                src1Idx_=src1_idx,
                isAuto_=is_auto,
            )
        )


# --------------------------------------------------------------------- #
# Layout / scheduling
# --------------------------------------------------------------------- #


def _resolve_layout(match: MatchedOp, target) -> dict[str, Any]:
    """Map workload-side operand bindings to target handles.

    Real implementation is the autoscheduler (Blocker 4). For now this
    is a hard-coded placeholder that knows the report-16 GEMV shape.
    """
    if target.name != "samsung_hbm_pim":
        raise NotImplementedError(
            f"BLOCKER 4: no layout for target {target.name!r}. The Samsung "
            "placeholder is the only supported mapping."
        )
    # GEMV placeholder: matrix → grf_a, vector → even_bank, accumulator → grf_b.
    # See HANDOFF.md Blocker 4 — this is *not* Samsung's real GEMV mapping,
    # it is the placeholder that produces canonical bytes for the static
    # comparison test. The real autoscheduler will replace it.
    layout = {
        "local_W": target.grf_a,
        "local_x": target.banks[2 * UnitId(level=1, unit=None)],  # even_bank pattern
        "acc": target.grf_b,
    }
    bindings = {}
    for opb in match.operands:
        if opb.memref_name not in layout:
            raise NotImplementedError(
                f"BLOCKER 4: operand memref {opb.memref_name!r} has no "
                "placeholder layout entry. The placeholder only knows the "
                "report-16 GEMV memref names."
            )
        bindings[opb.role] = layout[opb.memref_name]
    return bindings


# Samsung HBM-PIM GRF lane count — each MAC consumes this many fp16
# elements per K-iteration, so the inner-K loop trip count is
# `(K // _SAMSUNG_LANE_BURST) - 1` JUMPs after the first MAC. The real
# value is layout-dependent (autoscheduler territory); we hard-code 8
# here to match the report-16 target spec's `lanes=8`.
_SAMSUNG_LANE_BURST = 8


def _parse_loop_bound(text: str) -> int | None:
    """Extract a constant integer from an affine-map-text upper-bound
    string. Returns None if it can't be parsed.

    Real shape examples: `"1024"`, `"() -> (1024)"`, `"affine_map<() -> (1024)>"`.
    """
    try:
        return int(text.strip())
    except ValueError:
        pass
    import re
    nums = re.findall(r"\b(\d+)\b", text)
    if len(nums) == 1:
        return int(nums[0])
    return None


def _emit_inner_loop_jump(
    match: MatchedOp, ctx: CodegenContext, n_emitted: int
) -> None:
    """Append a Samsung-style JUMP that folds the innermost reduction
    loop.

    Loop counter is `(inner_ub // burst) - 1` (the first MAC counts as
    iteration 0; JUMP loops back the rest). Loop offset is the number of
    instructions that comprise one iteration body — i.e. `n_emitted` for
    the compute ops the match emitted, plus 1 for the JUMP itself.
    """
    if not match.enclosing_loops:
        return
    inner = match.enclosing_loops[-1]
    ub = _parse_loop_bound(inner[2])
    if ub is None:
        return
    loop_counter = ub // _SAMSUNG_LANE_BURST - 1
    if loop_counter <= 0:
        return
    ctx.cmds.append(
        PIMCmd(
            type_="JUMP",
            loopCounter_=loop_counter,
            loopOffset_=n_emitted + 1,
        )
    )


def _walk_and_emit(target, trace: MatchTrace, ctx: CodegenContext) -> None:
    """Walk every MAC/MUL/... match in `trace` and dispatch to the
    corresponding target op's `emit`.

    Moves (LD/ST) are intentionally not walked here — naive blanket
    emission is wrong, since moves are preloads/storebacks that belong
    around the compute sites. That is Blocker 5 (move scheduling).

    The innermost reduction loop on each match is folded into a JUMP
    instruction (Blocker 6) rather than emitted as N unrolled MACs.
    """
    for match in trace.matches:
        op_obj = target.op(match.target_op_name)
        emit = getattr(op_obj, "emit", None)
        if emit is None:
            raise NotImplementedError(
                f"target op {op_obj.name!r} has no `emit` callback."
            )
        bindings = _resolve_layout(match, target)
        before = len(ctx.cmds)
        if op_obj.accumulates:
            emit(bindings["x"], bindings["y"], bindings["acc"], ctx)
        else:
            emit(bindings["x"], bindings["y"], bindings["dst"], ctx)
        n_emitted = len(ctx.cmds) - before
        _emit_inner_loop_jump(match, ctx, n_emitted)


# --------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------- #


class Compiled:
    """Compiled artifact: the emitted PIMCmd stream plus a `run` hook
    (still gated on Blocker 7 — PIMSimulator's driver).
    """

    def __init__(self, target, trace: MatchTrace, cmds: list[PIMCmd]):
        self.target = target
        self.trace = trace
        self.cmds = cmds

    def run(self, **inputs):  # pragma: no cover - blocked
        raise NotImplementedError(
            "BLOCKER 7: PIMSimulator's `pim_driver` only accepts fixed "
            "kernels (--op GEMV / ADD / MUL / RELU). Running an arbitrary "
            "emitted PIMCmd stream requires a new driver entry point."
        )


_BACKEND_CTX = {
    "samsung_hbm_pim": SamsungCtx,
}


def compile_for_target(target: Any, trace: MatchTrace) -> Compiled:
    """Lower a (target, trace) pair to a runnable backend artifact by
    walking the target's declarations.
    """
    target_name = getattr(target, "name", None)
    if trace.target_name != target_name:
        raise ValueError(
            f"trace.target_name {trace.target_name!r} != target.name {target_name!r}"
        )

    ctx_cls = _BACKEND_CTX.get(target_name)
    if ctx_cls is None:
        raise NotImplementedError(
            f"no codegen ctx for target {target_name!r}; supported: "
            f"{sorted(_BACKEND_CTX)}"
        )
    ctx = ctx_cls(target)
    _walk_and_emit(target, trace, ctx)
    return Compiled(target, trace, ctx.cmds)
