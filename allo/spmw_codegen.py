# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SPMW codegen — generic walker over target.move / target.op declarations.

The previous version of this module (commit `dcc97b7`) detected a
"GEMV-shaped" `MatchTrace` and dispatched to PIMSimulator's hard-coded
GEMV kernel. That was never the design; report 16 says codegen should
walk the target's declared moves/ops and call each one's `emit` callback
with a backend-specific `ctx`. This file is the start of that walker.

It is intentionally incomplete — see the BLOCKER comments below and the
report associated with this commit. End-to-end execution is **not**
attempted; running this raises with a structured blocker message.
"""

from __future__ import annotations

from typing import Any

from .spmw_match import MatchTrace


# --------------------------------------------------------------------- #
# Generic walker (skeleton)
# --------------------------------------------------------------------- #


class CodegenContext:
    """Backend-specific instruction builder.

    Each backend supplies its own subclass exposing the methods its
    `emit` lambdas need (Samsung: `ctx.cmd(name, dst, src0, src1)`).
    No subclass exists yet — see BLOCKER 3.
    """

    def __init__(self, target):
        self.target = target

    def cmd(self, name: str, **fields):  # pragma: no cover - abstract
        raise NotImplementedError(
            f"{type(self).__name__}.cmd is not implemented; "
            "each backend must subclass CodegenContext and define cmd()."
        )


def _resolve_layout(match, target) -> dict[str, Any]:
    """Map the matcher's workload-side bindings to target handles.

    The `MatchedOp.operands[i].memref_name` is a *workload* memref like
    `local_W`. The `emit` callback expects a *target* handle (a Register
    or MemoryRef). Translating one to the other is the autoscheduler's
    job — see BLOCKER 4.
    """
    raise NotImplementedError(
        "BLOCKER 4: layout / bank-and-register allocation is not implemented. "
        "match.operands carry workload memref names; emit lambdas need target "
        "handles (Register / MemoryRef). Need an autoscheduler pass that maps "
        "workload tensors to target memories/registers, threaded into emit's "
        "operand slots."
    )


def _walk_and_emit(target, trace: MatchTrace, ctx: CodegenContext) -> None:
    """Walk every match in `trace` and dispatch to the corresponding
    target op's `emit`. Walk every declared move and call its `emit`.
    """

    # 1. Walk declared moves on every unit.
    #    Moves' src/dst are bound at spec time, so emit takes only ctx.
    for unit in target._walk():
        for mv in unit.moves.values():
            emit = getattr(mv, "emit", None)
            if emit is None:
                raise NotImplementedError(
                    f"BLOCKER 1+2: move {mv.name!r} on unit "
                    f"{unit.name!r} has no `emit` attribute. The "
                    "Move/Op data model in spmw_target.py and the "
                    "allo.move()/allo.op() factory functions do not yet "
                    "accept an `emit=` kwarg. (See report 16 for the API.)"
                )
            emit(ctx)

    # 2. Walk matched op sites and dispatch via name.
    for match in trace.matches:
        op_obj = target.op(match.target_op_name)
        emit = getattr(op_obj, "emit", None)
        if emit is None:
            raise NotImplementedError(
                f"BLOCKER 1+2: op {op_obj.name!r} has no `emit` attribute. "
                "Same as the move case above."
            )
        # Each operand of the match needs to be turned into a target handle
        # before being passed to emit.
        bindings = _resolve_layout(match, target)  # raises BLOCKER 4
        if op_obj.accumulates:
            emit(bindings["x"], bindings["y"], bindings["acc"], ctx)
        else:
            emit(bindings["x"], bindings["y"], bindings["dst"], ctx)


# --------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------- #


class Compiled:
    """Placeholder for what compile_for_target should return — once
    BLOCKERs 1-6 are addressed, this should hold the emitted instruction
    stream and a runnable interface for it.
    """

    def __init__(self, target, trace: MatchTrace):
        self.target = target
        self.trace = trace

    def run(self, **inputs):  # pragma: no cover - blocked
        raise NotImplementedError(
            "BLOCKER 6: PIMSimulator's `pim_driver` only accepts fixed "
            "kernels (--op GEMV / ADD / MUL / RELU). Running an arbitrary "
            "emitted PIMCmd stream requires a different driver entry point "
            "into the simulator. The previous Q3 implementation worked "
            "around this by recognising the GEMV shape and calling the "
            "built-in GEMV kernel; that path was removed."
        )


def compile_for_target(target: Any, trace: MatchTrace) -> Compiled:
    """Lower a (target, trace) pair to a runnable backend artifact by
    walking the target's declarations.

    Currently raises a `NotImplementedError` chain that surfaces the
    blockers preventing end-to-end execution; see module docstring.
    """
    target_name = getattr(target, "name", None)
    if trace.target_name != target_name:
        raise ValueError(
            f"trace.target_name {trace.target_name!r} != target.name {target_name!r}"
        )

    ctx = CodegenContext(target)  # BLOCKER 3: no real backend ctx exists

    # Try the walk; this WILL raise — BLOCKER messages are how we
    # surface the gaps.
    _walk_and_emit(target, trace, ctx)

    return Compiled(target, trace)
