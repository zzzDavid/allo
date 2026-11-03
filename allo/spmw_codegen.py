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

import functools
import importlib
import inspect
import sys
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

from .spmw_autoschedule import (
    Placement,
    _matcher_search_scope,
    _matcher_work_scope,
    autoschedule,
    derive_layout_properties,
)
from .spmw_match import MatchTrace, MatchedOp
from .spmw_simenv import (
    aim_unavailable_reason as _aim_unavailable_reason,
    apu_v1_unavailable_reason as _apu_v1_unavailable_reason,
    samsung_unavailable_reason as _samsung_unavailable_reason,
    upmem_unavailable_reason as _upmem_unavailable_reason,
)
from .spmw_target import SymExpr, UnitId


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


@dataclass
class ResolvedHostMove:
    """A `host_xfer.*` record resolved against the target (spec 001 D5).

    `verb` is the `_Verb` sentinel; `move` is the declared `Move` it resolved
    to (by handle identity, not string ==); `device_handle` is the target's
    `Memory`/`Register` the transfer touches; `buffer_role` is the workload
    buffer's parameter name (`A`/`B`/`x`/`out`), recovered from the record's
    non-handle arg when it carries one (else `None`). These carry cost +
    operand-role info; the driver still performs the transfer (Q2 option a).
    """

    verb: object
    move: object
    device_handle: object
    buffer_role: "str | None" = None


@dataclass
class HostTrigger:
    """Lever 3 (SPEC-025 §5.4): one host-issued CRF fire for a work-id.

    In shared-CRF mode the CRF body is uploaded once (`programCrf`) and
    the host fires it per work-id rather than re-uploading the body.
    A trigger is NOT a `PIMCmd` -- it never flows through `programCrf`'s
    `<=4`-burst CRF upload / `validationCheck`. It rides a side-list
    (`Compiled.host_schedule`) the faithful run path reads. `tile_count`
    is the per-work-id output-tile fire count derived from the loop bound
    the body's JUMP folds, never a shape literal.
    """

    work_id: int
    tile_count: int


_PIM_CMD_FIELDS = (
    "type_",
    "dst_",
    "src0_",
    "src1_",
    "src2_",
    "loopCounter_",
    "loopOffset_",
    "isAuto_",
    "dstIdx_",
    "src0Idx_",
    "src1Idx_",
    "isRelu_",
)


def _clone_matcher_command(command):
    if isinstance(command, PIMCmd):
        return PIMCmd(**{name: getattr(command, name) for name in _PIM_CMD_FIELDS})
    if isinstance(command, str):
        return str(command)
    raise TypeError(
        f"matcher command stream contains unsupported {type(command).__name__!r}"
    )


def _matcher_command_manifest(commands) -> tuple:
    manifest = []
    for command in commands:
        if isinstance(command, PIMCmd):
            manifest.append(
                ("pim_cmd", tuple(getattr(command, name) for name in _PIM_CMD_FIELDS))
            )
        elif isinstance(command, str):
            manifest.append(("source_line", command))
        else:
            raise TypeError(
                "matcher command stream contains unsupported "
                f"{type(command).__name__!r}"
            )
    return tuple(manifest)


def _matcher_handle_path(handle) -> str:
    from .perf.cost import handle_path

    return handle_path(handle)


def _matcher_resolved_host_move_manifest(moves) -> tuple:
    manifest = []
    for resolved in moves:
        verb = getattr(resolved.verb, "name", None)
        if not isinstance(verb, str) or not verb:
            raise TypeError("resolved matcher host transfer has no typed verb")
        manifest.append(
            (
                verb,
                _matcher_handle_path(resolved.move),
                _matcher_handle_path(resolved.device_handle),
                resolved.buffer_role,
            )
        )
    return tuple(manifest)


@dataclass(frozen=True)
class MatcherCodegenArtifact:
    """Frozen command/source stream and runtime contract for one candidate."""

    target_name: str
    commands: tuple
    host_schedule: tuple
    host_preloads: tuple
    resolved_host_moves: tuple
    runtime_abi: tuple
    runtime_segments: tuple
    context: Any = field(compare=False, repr=False)

    def instantiate_commands(self) -> list:
        return [_clone_matcher_command(command) for command in self.commands]

    def instantiate_host_schedule(self) -> list[HostTrigger]:
        return [
            HostTrigger(trigger.work_id, trigger.tile_count)
            for trigger in self.host_schedule
        ]


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
        # Lever 2 (SPEC-024 §5): GRF preloads whose residency is "host"
        # are hoisted off the CRF stream onto the native broadcast; we
        # record (move_name, phase) here instead of emitting a CRF MOV so
        # the move is absent from `cmds` (the faithful run path then
        # excludes it). Default-empty for every backend that never sets
        # host residency.
        self.host_preloads: list[tuple[str, str]] = []
        # Lever 3 (SPEC-025 §5.3): in shared-CRF mode the CRF body is
        # emitted once into `cmds` and the per-work-id fires are recorded
        # here as HostTrigger records (not PIMCmds). Default-empty for the
        # per-work-id path and every non-Samsung backend.
        self.host_schedule: list["HostTrigger"] = []

    def cmd(self, name: str, **fields):  # pragma: no cover - abstract
        raise NotImplementedError(
            f"{type(self).__name__}.cmd is not implemented; "
            "each backend must subclass CodegenContext and define cmd()."
        )

    # ------------------------------------------------------------------ #
    # Move-scheduling hooks (spec 009).
    # ------------------------------------------------------------------ #

    def resolve_moves(self, role: str, src_handle, dst_handle):
        """Return ``(load_move_name, store_move_name)`` for ``role``.

        ``load_move_name`` is the name of a Move on the target whose dst
        is the placement handle and whose src is the operand's home
        memory (None if no preload is needed). ``store_move_name`` is
        the symmetric storeback (None for read-only operands).

        Default raises so a new ctx subclass cannot silently accept the
        call -- each backend must override; see spec 009 §B.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.resolve_moves must be overridden; "
            "see spec 009 §B for the per-backend table."
        )

    def resolve_spill_moves(self, tier: str, home_handle, *, n_entries: int = 1):
        """Return ``(load_move_name, store_move_name)`` for a value spilled
        to ``tier`` (SPEC-022 D1).

        ``home_handle`` is the register the value lives in while live;
        ``tier`` is the backend spill tier name
        (``bank_row`` | ``mram`` | ``wram`` | ``l1`` | ``l2``). The move
        scheduler prepends the load (tier -> home) at the work-id window
        open and appends the store (home -> tier) at the window close, in
        addition to the home register's own LD/ST. The returned names are
        the SAME ``(load, store)`` move-name pair shape ``resolve_moves``
        returns, so they reuse the existing ``_emit_move`` machinery.

        Default raises: a backend the allocator may spill onto MUST
        override. A backend whose ``tier`` genuinely cannot spill leaves
        this default (or re-raises), and that NotImplementedError
        propagates as a hard compile error -- never a silent
        register-resident emission.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.resolve_spill_moves({tier!r}) is not "
            "implemented; this backend cannot spill to that tier."
        )

    def after_match(self, match, n_emitted: int) -> None:
        """Post-match hook. Default no-op; Samsung overrides to emit the
        inner-K JUMP that folds the reduction loop (spec 009 §E rule 4).
        """
        return None

    # ------------------------------------------------------------------ #
    # Program-lowering hooks (matcher-path-shared-infra D1).
    # ------------------------------------------------------------------ #

    def emit_program(
        self, trace, layouts, *, schedule=None, host_moves=(), module=None
    ) -> None:
        """Lower the matched program into this context. Default: per-bucket walk.

        Contract for overrides:

        1. Sub-trace validity. ``rank_matcher_placements`` calls this on one
           kernel's sub-trace with ``schedule=None, host_moves=(),
           module=None``. An override must lower or reject such a sub-trace
           without the whole program, the schedule or the MLIR module, so a
           whole-program emitter needs a device-only path for feasibility.
        2. Rejection type. A layout the backend cannot realize raises
           ``pim.schedule_search.InfeasibleSchedule`` (ranking falls through to
           the next candidate). An op the backend cannot lower at all raises
           ``NotImplementedError`` (a compile error, not a ranking outcome).
        3. Output. Write whatever the backend's ``_run_<backend>`` consumes,
           ``self.cmds`` or additional ctx state, deterministically for a
           given input.
        4. Schedule honoring. When ``schedule`` is not None, issue launches
           and host transfers in ``schedule.steps`` order.
        """
        _walk_buckets(self.target, trace, self, layouts)

    def runtime_route(self, commands) -> tuple[str, tuple]:
        """(route name, runtime segments) recorded in the matcher runtime ABI."""
        return "unsupported-exact-matcher-runtime", ()


# --------------------------------------------------------------------- #
# Samsung HBM-PIM backend
# --------------------------------------------------------------------- #


def _eval_sym(
    expr, env: dict | None = None, *, representative: bool = False
) -> int | None:
    """Evaluate a SymExpr to a Python int, substituting UnitIds via ``env``.

    ``env`` maps ``(level, unit_name_or_None)`` (or a bare level) to a
    concrete int. An unbound UnitId raises ``ValueError`` naming its level and
    unit, unless the caller explicitly asks for a ``representative`` value, in
    which case it evaluates as unit 0. Returns ``None`` if the expression
    contains a non-numeric atom the evaluator can't reduce.
    """
    env = env or {}
    if isinstance(expr, int):
        return expr
    if isinstance(expr, UnitId):
        # Look up by (level, unit_name) first, then by level alone.
        unit_name = expr.unit.name if expr.unit is not None else None
        if (expr.level, unit_name) in env:
            return env[(expr.level, unit_name)]
        if expr.level in env:
            return env[expr.level]
        if representative:
            return 0
        raise ValueError(
            f"unbound UnitId at level {expr.level} (unit {unit_name!r}); bind "
            "it from the placement or request representative=True"
        )
    if isinstance(expr, SymExpr):
        vals = [_eval_sym(a, env, representative=representative) for a in expr.args]
        if any(v is None for v in vals):
            return None
        if expr.op == "add":
            return sum(vals)
        if expr.op == "sub":
            return vals[0] - vals[1]
        if expr.op == "mul":
            r = 1
            for v in vals:
                r *= v
            return r
        if expr.op == "floordiv":
            return vals[0] // vals[1]
        if expr.op == "mod":
            return vals[0] % vals[1]
        return None
    return None


def _parse_fiber_idx(idx):
    """Decompose a bank `MemoryRef.idx` of the form `stride*pid + r` into
    `(stride, r)`, or `None` if it is not that affine-in-one-UnitId form.

    `stride*pid` (mul) -> `(stride, 0)`; `stride*pid + r` (add) -> `(stride, r)`.
    The `stride` is the bank coordinate multiplier on the swizzle column (the
    banks-per-pim ratio the enumerator bound via `bank_stride * pid`), read off
    the index itself -- the idx is self-describing, so no separate stride
    argument is needed and nothing is pasted.
    """
    if isinstance(idx, SymExpr) and idx.op == "mul":
        a, b = idx.args
        if isinstance(a, int) and isinstance(b, UnitId):
            return (a, 0)
        if isinstance(b, int) and isinstance(a, UnitId):
            return (b, 0)
    if isinstance(idx, SymExpr) and idx.op == "add":
        a, b = idx.args
        for base, rem in ((a, b), (b, a)):
            if isinstance(rem, int) and isinstance(base, SymExpr) and base.op == "mul":
                m, n = base.args
                if isinstance(m, int) and isinstance(n, UnitId):
                    return (m, rem)
                if isinstance(n, int) and isinstance(m, UnitId):
                    return (n, rem)
    return None


def _bank_fiber_class(idx, stride: int | None = None) -> str | None:
    """Classify a bank `MemoryRef.idx` (`stride*pid + r`) to its per-fiber
    bank-class name -- the `range(stride)` generalization of the deleted
    two-class `_bank_parity` (SPEC-022 D3).

    `stride` may be supplied (e.g. from the carried `LinearLayout`'s
    `size_of(fiber_axis)`); when omitted it is the index's own coefficient.
    For `stride == 2` (the Samsung hardware fact: 16 banks / 8 pim units) the
    fiber remainder maps to exactly `"EVEN_BANK"` (r=0) / `"ODD_BANK"` (r=1),
    BYTE-IDENTICAL to `_bank_parity` on the Samsung path. For `stride > 2`
    (a no-sim wide-fiber target) it returns a per-fiber `"BANK_<r>"` name --
    PIMSimulator's `PIMOpdType` enum has only the two Samsung names, so a
    third class can only be emitted on the cost-only / virtual path.

    Returns `None` for an index that is not the `stride*pid + r` fiber form
    (the same "not the autoscheduler's canonical form" signal the old
    `_bank_parity` returned, so the `_opd` hard-error is preserved).
    """
    parsed = _parse_fiber_idx(idx)
    if parsed is None:
        return None
    coeff, r = parsed
    stride = coeff if stride is None else stride
    if r < 0 or r >= stride:
        return None
    if stride == 2:
        return "EVEN_BANK" if r == 0 else "ODD_BANK"
    return f"BANK_{r}"


# --------------------------------------------------------------------- #
# SK-Hynix AiM (GDDR6 PIM) backend
# --------------------------------------------------------------------- #


# --------------------------------------------------------------------- #
# GSI APU v1 (Gemini 1) backend
# --------------------------------------------------------------------- #


# --------------------------------------------------------------------- #
# GSI APU v2 (Gemini 2, G2) backend
# --------------------------------------------------------------------- #


# --------------------------------------------------------------------- #
# Placement / scheduling
# --------------------------------------------------------------------- #


def _resolve_layout(match: MatchedOp, layout: Placement) -> dict[str, Any]:
    """Translate a `MatchedOp`'s workload-side operand bindings to
    target handles via `layout.placements`.

    The autoscheduler (`spmw_autoschedule.autoschedule`) constructs the
    `Placement`; this function is a thin lookup that turns role → memref →
    handle into role → handle so `emit` can be called.

    """
    bindings: dict[str, Any] = {}
    for opb in match.operands:
        handle = layout.placements.get(opb.memref_name)
        if handle is None:
            raise NotImplementedError(
                f"layout has no placement for memref {opb.memref_name!r}; "
                f"available: {sorted(layout.placements)}"
            )
        bindings[opb.role] = handle
    # Inject the store target as role "dst" (and "acc" stays from operands).
    # result_memref_name is the @allo.work store's `to` memref. Non-accumulating
    # ops (MUL/ADD/RELU) write a register dst; the placement maps the result
    # memref to that register the same way it maps inputs. Inert for MAC, whose
    # accumulate emit reads `acc`, not `dst`.
    if match.result_memref_name is not None and "dst" not in bindings:
        dst_handle = layout.placements.get(match.result_memref_name)
        if dst_handle is not None:
            bindings["dst"] = dst_handle
    return bindings


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


def _bucket_by_work_id(trace: MatchTrace):
    """Group matches by their retained structural group and work ids.

    Returns a list of ``(work_id, [matches])`` pairs preserving
    first-seen order. ``group_id`` keeps distinct ``@allo.work`` kernels
    separate when they reuse the same numeric ``work_id``; symbol spelling is
    deliberately irrelevant.
    """
    buckets: dict = {}
    order: list = []
    for m in trace.matches:
        scope = _matcher_work_scope(m)
        key = (scope.group_id, scope.work_id)
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(m)
    return [(key[1], buckets[key]) for key in order]


def _emit_move(target, name, ctx) -> None:
    """Look up a named Move on ``target`` and invoke its ``emit`` callback
    with ``ctx``."""
    mv = target.move(name)
    if mv.emit is None:
        raise NotImplementedError(
            f"Move {name!r} on target {target.name!r} has no emit callback."
        )
    mv.emit(ctx)


def _schedule_moves(
    target,
    ctx: CodegenContext,
    layout: Placement,
    role_to_memref: dict[str, str],
    phase: str,
) -> None:
    """Emit the preload (``phase == 'pre'``) or storeback
    (``phase == 'post'``) move set for one work-id.

    Walks every role present in the layout's placements; asks the ctx
    which move name to emit; dedups by move name so broadcast operands
    that route through the same Move only emit it once per phase.
    """
    # Lever 2 (SPEC-024 §5): a move name is host-resident only if EVERY
    # role routing to it is host-resident; if any role on the name is crf
    # the CRF MOV must still be emitted (it materialises that crf role).
    # Default-missing residency == "crf", so non-Samsung backends and
    # every existing placement keep emitting exactly as before.
    residency = getattr(layout, "extra", {}).get("grf_residency", {})

    # First pass: resolve each role to its move name and whether it is
    # crf-resident on that name.
    name_crf: dict[str, bool] = {}
    name_seen_order: list[str] = []
    for role, memref in role_to_memref.items():
        handle = layout.placements.get(memref)
        if handle is None:
            continue
        ld_name, st_name = ctx.resolve_moves(role, src_handle=None, dst_handle=handle)
        chosen = ld_name if phase == "pre" else st_name
        if chosen is None:
            continue
        is_crf = residency.get(memref, "crf") != "host"
        if chosen not in name_crf:
            name_crf[chosen] = is_crf
            name_seen_order.append(chosen)
        else:
            name_crf[chosen] = name_crf[chosen] or is_crf

    for chosen in name_seen_order:
        if name_crf[chosen]:
            _emit_move(target, chosen, ctx)
        elif phase == "pre":
            # Host-resident preload: the native HAB broadcast fills GRF_A
            # (--op GEMV scaffolding); Tenon emits no CRF MOV so the move
            # is absent from compiled.cmds and the faithful run path
            # excludes it with zero new logic. Recorded on a ctx side-list
            # for run-path/audit visibility.
            if hasattr(ctx, "host_preloads"):
                ctx.host_preloads.append((chosen, phase))


def _walk_and_emit(
    target,
    trace: MatchTrace,
    ctx: CodegenContext,
    layouts: list[Placement],
    *,
    schedule=None,
    host_moves=(),
    module=None,
) -> None:
    """Lower ``trace`` into ``ctx`` through the backend's ``emit_program`` hook."""
    ctx.emit_program(
        trace, layouts, schedule=schedule, host_moves=host_moves, module=module
    )


def _walk_buckets(
    target,
    trace: MatchTrace,
    ctx: CodegenContext,
    layouts: list[Placement],
) -> None:
    """Walk every match in ``trace``, bucketed by work-id, and dispatch
    each match to its target op's ``emit`` callback.

    ``layouts`` is aligned with ``_bucket_for_autoschedule(trace)``. For each
    work-id bucket, we look up the layout by its retained matcher search scope
    and compute the `role -> memref` map from THIS bucket's matches (the real
    bug fix: role -> memref is per-kernel, not per-trace).

    Move scheduling (preloads before the first match of a work-id and
    storebacks after the last) is delegated to ``_schedule_moves``; the
    per-backend ``resolve_moves`` hook controls which Move names get
    emitted. Backend-specific post-match work (e.g. Samsung's inner-K
    JUMP) flows through ``ctx.after_match``.
    """
    from .spmw_autoschedule import _bucket_for_autoschedule, _trace_memrefs_by_role

    layout_by_scope: dict[tuple, Placement] = {
        search_scope: layout
        for (search_scope, _matches), layout in zip(
            _bucket_for_autoschedule(trace), layouts
        )
    }

    def _emit_one_bucket(matches: list[MatchedOp]) -> None:
        layout = layout_by_scope[_matcher_search_scope(matches[0])]
        # Make backend decisions derived from the carried F2 layout visible to
        # operation dispatch and backend contexts. This deliberately
        # overwrites stale duplicated fields on hand-authored placements.
        layout.extra = derive_layout_properties(target, layout)
        # Backend contexts use this coordinate to materialise symbolic target
        # handles (for AiM: channel mask and 4*bank_group+bank index).
        ctx._active_work_id = _matcher_work_scope(matches[0]).work_id
        if _is_backend_ctx(ctx, "apu_v1"):
            ctx.group_size = int(layout.extra.get("group_size", 32768))
            ctx.vector_batches = max(1, int(layout.extra.get("n_out_tiles", 1)))

        # Per-bucket role -> memref must come from THIS bucket's matches,
        # not the full trace -- that was the root bug for multi-kernel
        # workloads (e.g. MLP layer1/layer2 each binding `x` to a
        # different weight memref).
        try:
            role_to_memref = _trace_memrefs_by_role(matches)
        except NotImplementedError:
            # Non-uniform traces aren't supported by the move scheduler;
            # bypass move emission rather than failing the whole compile.
            role_to_memref = {}

        _schedule_moves(target, ctx, layout, role_to_memref, phase="pre")
        for match in matches:
            op_obj = target.op(
                getattr(layout, "extra", {}).get("operation_name", match.target_op_name)
            )
            emit = getattr(op_obj, "emit", None)
            if emit is None:
                raise NotImplementedError(
                    f"target op {op_obj.name!r} has no `emit` callback."
                )
            bindings = _resolve_layout(match, layout)
            # Expose the active placement + bindings so a dual-fiber
            # Samsung ctx can re-emit the odd-fiber MAC in after_match.
            # Additive ctx state; ignored by every other backend.
            ctx._active_placement = layout
            ctx._active_bindings = bindings
            before = len(ctx.cmds)
            if op_obj.accumulates:
                emit(bindings["x"], bindings["y"], bindings["acc"], ctx)
            elif len(inspect.signature(op_obj.fn).parameters) == 1:
                # Unary op (e.g. RELU): single source + store dst.
                emit(bindings["x"], bindings["dst"], ctx)
            else:
                emit(bindings["x"], bindings["y"], bindings["dst"], ctx)
            n_emitted = len(ctx.cmds) - before
            ctx.after_match(match, n_emitted)
        _schedule_moves(target, ctx, layout, role_to_memref, phase="post")

    if _is_backend_ctx(ctx, "apu_v1") and any(
        _matcher_work_scope(match).coalesced_axes for match in trace.matches
    ):
        buckets = [
            (_matcher_work_scope(matches[0]).work_id, matches)
            for _name, matches in _bucket_for_autoschedule(trace)
        ]
    else:
        buckets = _bucket_by_work_id(trace)

    if _is_backend_ctx(ctx, "samsung_hbm_pim"):
        # Resolve CRF issue independently for every retained matcher group.
        # A prior implementation inspected the first placement carrying a
        # `crf_issue` field and silently imposed that choice on the complete
        # program. Mixed shared/per-work-id programs now materialize exactly,
        # while an internally mixed group fails as an infeasible schedule.
        per_kernel: dict[int, list] = {}
        kernel_order: list[int] = []
        for work_id, matches in buckets:
            group_id = _matcher_work_scope(matches[0]).group_id
            if group_id not in per_kernel:
                per_kernel[group_id] = []
                kernel_order.append(group_id)
            per_kernel[group_id].append((work_id, matches))

        expected = target.work_grid()[1]
        for group_id in kernel_order:
            k_buckets = per_kernel[group_id]
            group_layouts = [
                layout_by_scope[_matcher_search_scope(matches[0])]
                for _work_id, matches in k_buckets
            ]
            issues = tuple(
                getattr(layout, "extra", {}).get("crf_issue", "per_workid")
                for layout in group_layouts
            )
            if any(issue not in ("shared", "per_workid") for issue in issues):
                from .pim.schedule_search import InfeasibleSchedule

                raise InfeasibleSchedule(
                    f"matcher group {group_id} has an invalid CRF issue decision"
                )
            if len(set(issues)) != 1:
                from .pim.schedule_search import InfeasibleSchedule

                raise InfeasibleSchedule(
                    f"matcher group {group_id} mixes shared and per-work-id CRF issue"
                )
            if issues[0] == "per_workid":
                for _work_id, matches in k_buckets:
                    _emit_one_bucket(matches)
                continue

            from .spmw_autoschedule import _matcher_placement_decision

            representative_trace = MatchTrace(
                target_name=trace.target_name,
                module_name=trace.module_name,
                matches=k_buckets[0][1],
            )
            representative = _matcher_placement_decision(
                representative_trace,
                group_layouts[0],
            )
            for layout, (_work_id, matches) in zip(group_layouts[1:], k_buckets[1:]):
                candidate_trace = MatchTrace(
                    target_name=trace.target_name,
                    module_name=trace.module_name,
                    matches=matches,
                )
                if (
                    _matcher_placement_decision(candidate_trace, layout)
                    != representative
                ):
                    from .pim.schedule_search import InfeasibleSchedule

                    raise InfeasibleSchedule(
                        f"matcher group {group_id} shares CRF across distinct bodies"
                    )

            _emit_one_bucket(k_buckets[0][1])
            for work_id, matches in k_buckets:
                tile_count = sum(
                    1 for match in matches if match.target_op_name == "MAC"
                )
                ctx.host_schedule.append(HostTrigger(work_id, tile_count))
            # SPEC-025 §5.3 agreement guard: a single full-grid GEMV kernel's
            # work-id axis must equal the unit-tree fanout product, else the
            # trigger schedule would be mispriced. Only a single-kernel
            # workload spans the full grid; a multi-kernel layer
            # (mapping=[1], etc.) legitimately walks fewer work-ids, so the
            # guard applies only when this is the sole kernel.
            if (
                len(kernel_order) == 1
                and len(k_buckets) > 1
                and len(k_buckets) != expected
            ):
                # Gap-1: advisory, not fatal. The trigger schedule above used
                # the observed bucket count as authored; warn on mismatch so a
                # mispriced grid surfaces without blocking the compile.
                warnings.warn(
                    "samsung shared-CRF: single-kernel trace has "
                    f"{len(k_buckets)} work-id buckets but target geometry "
                    f"yields {expected} (unit-tree fanout product); the "
                    "shared-CRF trigger schedule may be mispriced. Proceeding "
                    "with the authored work-id count.",
                    stacklevel=2,
                )
        return

    for _work_id, matches in buckets:
        _emit_one_bucket(matches)


# --------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------- #


# What a `RunResult` evidences. See the `RunResult` docstring.
RUN_KIND_DEVICE = "device"
RUN_KIND_SIMULATOR = "simulator"
RUN_KIND_SIMULATOR_TIMING_ONLY = "simulator-timing-only"
RUN_KIND_FUNCTIONAL = "functional"
RUN_KIND_VIRTUAL = "virtual"
RUN_KIND_UNSPECIFIED = "unspecified"
RUN_KINDS = frozenset(
    {
        RUN_KIND_DEVICE,
        RUN_KIND_SIMULATOR,
        RUN_KIND_SIMULATOR_TIMING_ONLY,
        RUN_KIND_FUNCTIONAL,
        RUN_KIND_VIRTUAL,
    }
)
# Kinds whose run executed the program and read its numerical outputs back.
RUN_KINDS_WITH_OUTPUTS = frozenset(
    {RUN_KIND_DEVICE, RUN_KIND_SIMULATOR, RUN_KIND_FUNCTIONAL}
)


@dataclass
class RunResult:
    """Outcome of running a compiled artifact.

    `kind` states what the result evidences; every backend runner must set
    it to one of:

    * ``"device"``: physical hardware executed the program and its outputs
      were read back.
    * ``"simulator"``: a functional simulator executed the program and its
      outputs were read back.
    * ``"simulator-timing-only"``: a timing simulator ran the command stream;
      no numerical outputs exist.
    * ``"functional"``: a host-side functional execution of the lowered
      program; outputs exist, `cycles` is not a measurement.
    * ``"virtual"``: analytical cost evaluation only; nothing executed and
      caller arrays are not written.

    Numerical outputs live only in ``extra["outputs"]`` (a mapping from
    output role or argument name to array). For ``device``, ``simulator``
    and ``functional`` results it holds every output the run produced (an
    all-zero array is a valid output); a runner that cannot read an output
    back raises instead of returning. ``virtual`` and
    ``simulator-timing-only`` results carry no outputs.

    `cycles` is the backend's cycle figure, or `None` when it has none. A
    simulator or device that cannot run raises `SimulatorUnavailable`.
    `stdout` is the captured output of the sub-process / API call.
    `backend` is the target (or route) name.

    `kind` defaults to ``"unspecified"`` so the dataclass stays
    constructible; `validate()` rejects such a result, and
    `CompiledCallable.run_backend` validates every backend result.
    """

    cycles: int | None
    stdout: str
    backend: str
    extra: dict = field(default_factory=dict)
    kind: str = RUN_KIND_UNSPECIFIED

    def __post_init__(self):
        if self.kind != RUN_KIND_UNSPECIFIED and self.kind not in RUN_KINDS:
            raise ValueError(
                f"RunResult.kind must be one of {sorted(RUN_KINDS)}, got {self.kind!r}"
            )

    @property
    def outputs(self) -> dict:
        """``extra["outputs"]``, or an empty dict when there is none."""
        return dict((self.extra or {}).get("outputs") or {})

    @property
    def executed(self) -> bool:
        """True when the run executed the program and read outputs back."""
        return self.kind in RUN_KINDS_WITH_OUTPUTS

    def validate(self) -> "RunResult":
        """Check the kind contract; return ``self``. Raise RuntimeError on a
        result with no kind or a non-executing result carrying outputs."""
        if self.kind not in RUN_KINDS:
            raise RuntimeError(
                f"backend {self.backend!r} returned a RunResult without a kind; "
                f"runners must set kind to one of {sorted(RUN_KINDS)}"
            )
        if not self.executed and self.outputs:
            raise RuntimeError(
                f"backend {self.backend!r} returned a {self.kind!r} RunResult "
                f"carrying outputs {sorted(self.outputs)}; only "
                f"{sorted(RUN_KINDS_WITH_OUTPUTS)} results may carry outputs"
            )
        return self


class SimulatorUnavailable(RuntimeError):
    """The target's simulator or device is not usable in this environment."""

    def __init__(self, target_name: str, reason: str):
        super().__init__(f"{target_name}: simulator unavailable: {reason}")
        self.target_name = target_name
        self.reason = reason


def simulator_unavailable_reason(target_name: str) -> str | None:
    """None when `run()` can execute for this target on this host."""
    if target_name == "samsung_hbm_pim":
        return _samsung_unavailable_reason()
    if target_name == "aim":
        return _aim_unavailable_reason()
    if target_name == "apu_v1":
        return _apu_v1_unavailable_reason()
    if target_name == "upmem":
        return _upmem_unavailable_reason()
    return f"no simulator runner for target {target_name!r}"


# --------------------------------------------------------------------- #
# Per-backend run hooks
# --------------------------------------------------------------------- #


def _run_virtual(compiled: "Compiled", **inputs) -> RunResult:
    """Evaluate the already-lowered cost program; nothing executes.

    The result has ``kind="virtual"`` and no outputs: caller arrays are left
    untouched.
    """
    if compiled.cost is None or compiled.execution_graph is None:
        raise RuntimeError("virtual execution requires an executable CostSpec")
    # A backend context may refuse inputs the selected program could not take
    # on an executing route (an initial output a plan resets): checked, never
    # written.
    check = getattr(getattr(compiled, "layout_ctx", None), "virtual_run_check", None)
    if callable(check):
        check(inputs)
    estimate = compiled.cost.evaluate(compiled.execution_graph)
    extra = {
        "critical_path": list(estimate.critical_path),
        "utilization": dict(estimate.utilization),
        "bottlenecks": list(estimate.bottlenecks),
        "model_fingerprint": estimate.model_fingerprint,
        "cost_model": compiled.cost.spec.name,
        "priced_target": compiled.target.name,
    }
    # A backend context may label the estimate (validity domain of its cost
    # model, timing scope); it never changes the estimate itself.
    hook = getattr(getattr(compiled, "layout_ctx", None), "virtual_run_extra", None)
    if callable(hook):
        extra.update(hook())
    return RunResult(
        kind=RUN_KIND_VIRTUAL,
        cycles=estimate.cycles,
        stdout=(
            f"virtual backend: executable cost spec {compiled.cost.spec.name!r}; "
            "analytical estimate only, nothing executed, outputs not written"
        ),
        backend="virtual",
        extra=extra,
    )


@dataclass(frozen=True)
class SourceBackendBinding:
    """Workload-parameter names bound to backend runtime kwargs.

    ``inputs`` maps a backend kwarg (e.g. ``"A"``) to the source parameter
    name that supplies it. ``output`` names the source parameter receiving
    the single backend output, or None.
    """

    inputs: "dict[str, str]" = field(default_factory=dict)
    output: "str | None" = None
    # Every externally visible destination the backend writes, keyed by the
    # runtime's output name (``"out"`` for the producer, a sink's memref name
    # for a copy), mapped to the source parameter receiving it.
    outputs: "dict[str, str]" = field(default_factory=dict)


# Samsung runtime kwargs. A contraction binds by structure (release audit
# S-05): the matrix (indexed by the output row and the reduction axis) is the
# bank-resident `A`, the vector (indexed only by the reduction axis) is `B`,
# whatever the source operand order. Elementwise inputs bind in signature
# order of the matched op (x, y, z) to `IN0..IN2`.
_SAMSUNG_MAC_ROLE_KWARGS = (("matrix", "A"), ("vector", "B"))
_SAMSUNG_ELTWISE_ROLE_KWARGS = (("x", "IN0"), ("y", "IN1"), ("z", "IN2"))


def source_backend_binding(target, trace, parameter_names) -> SourceBackendBinding:
    """Bind workload parameters to the Samsung runtime roles structurally.

    Uses the trace's retained source identities (`trace.source_value_refs`,
    keyed by parameter ordinal) and the operands' `value_ref`. Returns an
    empty binding for other targets, multi-kernel traces, traces outside the
    Samsung lowering domain, or when the matches disagree.
    """
    empty = SourceBackendBinding()
    if getattr(target, "name", None) != "samsung_hbm_pim":
        return empty
    source_value_refs = getattr(trace, "source_value_refs", None) or {}
    if not source_value_refs or not trace.matches:
        return empty
    if len({_matcher_work_scope(m).group_id for m in trace.matches}) != 1:
        return empty

    from .spmw_samsung import (
        SAMSUNG_ELEMENTWISE_OPS,
        SamsungUnsupported,
        samsung_contraction,
    )

    ops = {m.target_op_name for m in trace.matches}
    role_kwargs = []  # (kwarg, set of operand roles per match)
    if ops == {"MAC"}:
        try:
            classified = [(m, samsung_contraction(m)) for m in trace.matches]
        except SamsungUnsupported:
            return empty
        for kind, kwarg in _SAMSUNG_MAC_ROLE_KWARGS:
            refs = set()
            for match, c in classified:
                role = c.matrix_role if kind == "matrix" else c.vector_role
                refs.update(op.value_ref for op in match.operands if op.role == role)
            role_kwargs.append((kwarg, refs))
    elif len(ops) == 1 and ops <= SAMSUNG_ELEMENTWISE_OPS:
        for role, kwarg in _SAMSUNG_ELTWISE_ROLE_KWARGS:
            refs = {
                op.value_ref
                for match in trace.matches
                for op in match.operands
                if op.role == role
            }
            if refs:
                role_kwargs.append((kwarg, refs))
    else:
        return empty

    parameter_names = tuple(parameter_names)
    ordinal_by_ref = {}
    for ordinal, ref in source_value_refs.items():
        if 0 <= int(ordinal) < len(parameter_names):
            ordinal_by_ref.setdefault(ref, int(ordinal))

    inputs = {}
    for kwarg, refs in role_kwargs:
        if len(refs) != 1:
            return empty
        ordinal = ordinal_by_ref.get(next(iter(refs)))
        if ordinal is None:
            return empty
        inputs[kwarg] = parameter_names[ordinal]

    operand_refs = {
        operand.value_ref
        for match in trace.matches
        for operand in match.operands
        if operand.value_ref is not None
    }
    unreferenced = [
        parameter_names[int(ordinal)]
        for ordinal, ref in sorted(source_value_refs.items())
        if 0 <= int(ordinal) < len(parameter_names) and ref not in operand_refs
    ]
    # Every external destination: the producers' own buffers and the sinks
    # of accepted copies (`external_effects`), by retained identity.
    # Keys are the destinations' own memref names (distinct within a
    # function), never a fixed word a source buffer could also be called.
    outputs = {}
    producer_output = None
    stored = {m.result_value_ref for m in trace.matches}
    if len(stored) == 1 and next(iter(stored)) in ordinal_by_ref:
        producer_output = parameter_names[ordinal_by_ref[next(iter(stored))]]
        names = {m.result_memref_name for m in trace.matches if m.result_memref_name}
        if len(names) == 1:
            outputs[next(iter(names))] = producer_output
    for match in trace.matches:
        for (memref, _text), (ref, local) in zip(
            match.extra.get("output_sinks", ()), match.extra.get("output_sink_refs", ())
        ):
            if not local and ref in ordinal_by_ref:
                outputs[memref] = parameter_names[ordinal_by_ref[ref]]
    output = unreferenced[0] if len(unreferenced) == 1 else None
    if output is None:
        # In place (``a[i] = a[i] + b[i]``): the stored parameter is also an
        # operand, so only the matches' store destination names the output.
        output = producer_output
    if output is None and len(outputs) == 1:
        output = next(iter(outputs.values()))
    return SourceBackendBinding(inputs=inputs, output=output, outputs=outputs)


class Compiled:
    """Compiled artifact: the emitted backend instruction stream plus a
    `run` hook that dispatches to the per-backend simulator/hardware.
    """

    def __init__(
        self,
        target,
        trace: MatchTrace,
        cmds: list,
        layout: Placement,
        ctx: "CodegenContext | None" = None,
        backend: "str | None" = None,
        host_moves: "list[ResolvedHostMove] | None" = None,
        execution_graph=None,
        cost=None,
        host_schedule=None,
        runtime_abi=(),
        runtime_segments=(),
        matcher_codegen_artifact=None,
        placement_ranking=None,
        launch_schedule=None,
    ):
        self.target = target
        self.trace = trace
        self.cmds = cmds
        self.layout = layout
        # Backend project emitters may need the realized layout/codegen state
        # (for APU v1: used VR aliases and group size).
        self.layout_ctx = ctx
        # spec 001 D5: the resolved `host_xfer.*` moves (verb, declared Move,
        # device handle, buffer role). Empty by default -> today's inferred-role
        # path in `_run_samsung`. When non-empty, the run path asserts the
        # operand->role binding against these explicit moves.
        self.host_moves: list[ResolvedHostMove] = list(host_moves or [])
        # Canonical candidate plan consumed by both virtual execution and
        # diagnostics. Targets not yet ported to the resource model leave it
        # unset and remain on their existing path during migration.
        self.execution_graph = execution_graph
        self.cost = cost
        self.backend = backend
        self.runtime_abi = tuple(runtime_abi)
        self.runtime_segments = tuple(runtime_segments)
        self.matcher_codegen_artifact = matcher_codegen_artifact
        self.placement_ranking = placement_ranking
        self.launch_schedule = launch_schedule
        # Lever 3 (SPEC-025 §5.4): shared-CRF host trigger schedule (one
        # HostTrigger per work-id), parallel to `cmds`. Empty for the
        # per-work-id path and every non-Samsung backend.
        self.host_schedule: list[HostTrigger] = list(
            getattr(ctx, "host_schedule", []) or []
            if host_schedule is None
            else host_schedule
        )

    def run(self, **inputs) -> RunResult:
        """Run this compiled artifact on its target backend.

        Returns a `RunResult` with cycle count (when the simulator
        provides one), the simulator's stdout, and the backend name.
        Raises `SimulatorUnavailable` when the simulator/hardware cannot
        run here, and `NotImplementedError` for a target with no run hook.
        """
        target_name = getattr(self.target, "name", None)
        # design 04 §2.1: a `backend="virtual"` Compiled dispatches to the
        # sim-free virtual runner regardless of target.name.
        try:
            runner = backend_runner(
                "virtual" if self.backend == "virtual" else target_name
            )
        except KeyError:
            runner = None
        if runner is None:
            raise NotImplementedError(
                f"no run hook for target {target_name!r}; "
                f"supported: {_runnable_backends()}"
            )
        return runner(self, **inputs)

    def run_batched(self, W, X, compare_native: bool = True) -> RunResult:
        """Run a batched GEMV (SPEC-026): B input vectors X[B,K] against
        one weight W. Samsung-only. The chosen placement's
        `stage_resident` flag (set by argmin) selects preload-once vs
        re-preload-per-vector sequencing; codegen materialises it.

        `compare_native=True` also runs the native rebaseline comparator
        so the caller can assert the gate-A strict beat. B is read off
        `X.shape[0]`, never a literal.
        """
        target_name = getattr(self.target, "name", None)
        if target_name != "samsung_hbm_pim":
            raise NotImplementedError(
                f"run_batched is Samsung-only; got target {target_name!r}"
            )
        from .spmw_samsung import _run_samsung_batched

        return _run_samsung_batched(self, W, X, compare_native=compare_native)


# Backend contexts and runners live in per-backend modules, imported only
# when a target of that backend is compiled or run.
_BACKENDS: dict[str, tuple[str, str, str | None]] = {
    "samsung_hbm_pim": ("allo.spmw_samsung", "SamsungCtx", "_run_samsung"),
    "aim": ("allo.spmw_aim", "AimCtx", "_run_aim"),
    "apu_v1": ("allo.spmw_apu_v1", "APUv1Ctx", "_run_apu_v1"),
    "apu_v2": ("allo.spmw_apu_g2", "APUG2SourceCtx", "_run_apu_g2"),
    "upmem": ("allo.spmw_upmem", "UpmemCtx", "_run_upmem"),
    # Analytical-only: the runner raises (no device, no in-package simulator).
    "attacc": ("allo.spmw_attacc", "AttAccCtx", "_run_attacc"),
}


def _backend_entry(target_name: str) -> tuple[str, str, str | None]:
    try:
        return _BACKENDS[target_name]
    except KeyError:
        raise KeyError(
            f"unknown backend target {target_name!r}; known: {sorted(_BACKENDS)}"
        ) from None


@functools.cache
def backend_context_class(target_name: str) -> type[CodegenContext]:
    module_name, class_name, _runner = _backend_entry(target_name)
    return getattr(importlib.import_module(module_name), class_name)


@functools.cache
def backend_runner(target_name: str) -> Callable | None:
    if target_name == "virtual":
        return _run_virtual
    module_name, _class_name, runner_name = _backend_entry(target_name)
    if runner_name is None:
        return None
    return getattr(importlib.import_module(module_name), runner_name)


def _runnable_backends() -> list[str]:
    return sorted(
        [name for name, entry in _BACKENDS.items() if entry[2] is not None]
        + ["virtual"]
    )


def _is_backend_ctx(ctx, target_name: str) -> bool:
    # A ctx of a backend implies its module is already imported; checking
    # sys.modules keeps the walker from importing unrelated backends.
    module_name, class_name, _runner = _BACKENDS[target_name]
    module = sys.modules.get(module_name)
    return module is not None and isinstance(ctx, getattr(module, class_name))


_MOVED: dict[str, str] = {
    "SamsungCtx": "allo.spmw_samsung",
    "_write_samsung_cmds": "allo.spmw_samsung",
    "_samsung_runtime_route": "allo.spmw_samsung",
    "_run_samsung": "allo.spmw_samsung",
    "_layout_stage_resident": "allo.spmw_samsung",
    "_run_samsung_batched": "allo.spmw_samsung",
    "AimCtx": "allo.spmw_aim",
    "_aim_runtime_segments": "allo.spmw_aim",
    "_run_aim": "allo.spmw_aim",
    "APUv1Ctx": "allo.spmw_apu_v1",
    "_apu_v1_kernel_src": "allo.spmw_apu_v1",
    "_parse_apu_v1_prof_print": "allo.spmw_apu_v1",
    "_apu_v1_output_specs": "allo.spmw_apu_v1",
    "_apu_v1_prepare_io": "allo.spmw_apu_v1",
    "_run_apu_v1": "allo.spmw_apu_v1",
}


def __getattr__(name):
    # Forwarding for out-of-tree callers only; code in allo/ imports from the
    # owning module.
    module_name = _MOVED.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module_name), name)


# Backends whose lowering decomposes ONE work item over the whole device
# itself, so `@allo.work(mapping=[1])` is an executable launch:
#   samsung_hbm_pim: elementwise ops are one item walking a 1-D extent
#     (spmw_samsung.samsung_elementwise); the REDUCE program accepts any item
#     set whose stores partition the rows (`_covered_contraction`), one item
#     covering every row included.
#   upmem: the DPU x tasklet split is the backend's decision domain
#     (spmw_upmem knobs), not the workload grid.
#   aim: the AiM lowering maps a single item's rows onto channels
#     (tests/pim/test_aim_axis_roles.py `gemv_single`).
#   attacc: the `attacc_tiling` knob maps a single item over one device's
#     channels and units (allo/pim/attacc_tiling.py).
# apu_v1 and apu_v2 are not listed: a dataflow region's scalar mapping on
# apu_v1 is a coalesced VR partition axis, accepted by its own branch below;
# a plain-function contraction on either APU is a single item only through a
# candidate that declares ``extra["whole_device_launch"]`` (the vector plans
# and the composed-dot launch plans decompose the one item over the device).
_SINGLE_ITEM_LAUNCH_BACKENDS = frozenset({"samsung_hbm_pim", "upmem", "aim", "attacc"})
# Backends that map a 1-D grid of n items (n <= the OUTERMOST extent) onto
# disjoint groups of outermost units:
#   upmem: n DPUs are allocated and loaded (`num_dpus == mapping[0]`,
#     tests/pim/test_gemv_all_targets.py `upmem_gemv`).
#   aim: each item is a replica on its own channel group
#     (aim_lowering._replica_channel_layout, which raises when the channels
#     do not divide among the replicas).
# Samsung is not listed: the 8 PIM blocks of a pseudo-channel run its CRF in
# lockstep and the conductor drives every channel.
_OUTER_SUBSET_LAUNCH_BACKENDS = frozenset({"upmem", "aim"})


def _check_work_grid(target, trace, *, auto_fill=False, layouts=None):
    """Check that every kernel's declared work grid maps onto the target's
    unit tree in a form the backend executes; raise ValueError otherwise.
    Returns the work-item count of a single-kernel trace (the derived
    fanout product for a multi-kernel trace).

    Accepted forms, per `@allo.work` kernel (grid = its `mapping`):

    ======================  ===================================================
    full fanout             grid == unit-tree fanout (Samsung [16, 8])
    outer prefix            grid == a prefix of the fanout ([32] of AiM
                            [32, 4, 4]); each item addresses an outer unit
                            and the selected op spans its descendants
    single item             grid == [1] on a backend in
                            `_SINGLE_ITEM_LAUNCH_BACKENDS` (or a virtual-only
                            target, which executes nothing)
    outer-level subset      grid == [n], n <= the outermost extent, on a
                            backend in `_OUTER_SUBSET_LAUNCH_BACKENDS` (UPMEM
                            [8] launches 8 of 64 DPUs; AiM [4] places 4
                            replicas on 8-channel groups)
    APU v1 coalesced axis   the scalar mapping partitions the VR lanes; the
                            backend supplies the four APUC tasks
    ======================  ===================================================

    Anything else (e.g. [4, 4] on Samsung's 16 x 8 tree, whose 16 items
    match no unit level) raises: no unit mapping exists for it, and the
    cost program's one-stream-per-unit graph would misprice it.  Nothing
    is auto-filled; `auto_fill` is accepted for compatibility and ignored.
    ``layouts`` are the selected placements: a single item is also accepted
    when every selected placement declares ``extra["whole_device_launch"]``
    (a backend candidate plan that decomposes the one item over the device).
    Backend partition checks that need the workload (Samsung row coverage,
    `SamsungCtx._covered_contraction`) still apply on top of this one.
    """
    import math

    from .spmw_autoschedule import _bucket_for_autoschedule

    del auto_fill
    factors, derived = target.work_grid()
    factors = tuple(int(f) for f in factors)
    buckets = _bucket_for_autoschedule(trace)
    if target.name == "apu_v1" and any(
        _matcher_work_scope(match).coalesced_axes for match in trace.matches
    ):
        return derived
    groups: dict[int, list] = {}
    for _scope, matches in buckets:
        groups.setdefault(_matcher_work_scope(matches[0]).group_id, []).append(matches)
    prefixes = [factors[:n] for n in range(1, len(factors) + 1)]
    whole_device = bool(layouts) and all(
        (getattr(layout, "extra", None) or {}).get("whole_device_launch")
        for layout in layouts
    )
    single_ok = (
        target.name in _SINGLE_ITEM_LAUNCH_BACKENDS
        or target.name not in _BACKENDS
        or whole_device
    )
    problems = []
    for group_id, group in groups.items():
        observed = len(group)
        shape = tuple(
            int(f) for f in (_matcher_work_scope(group[0][0]).group_shape or ())
        )
        if shape and math.prod(shape) != observed:
            shape = ()  # partially retained scope: judge by count alone
        grid = tuple(f for f in shape if f != 1) if shape else None
        if grid is not None:
            if grid in prefixes:
                continue
        elif observed in {math.prod(p) for p in prefixes}:
            continue
        if (
            target.name in _OUTER_SUBSET_LAUNCH_BACKENDS
            and factors
            and (grid is None or len(grid) <= 1)
            and observed <= factors[0]
        ):
            continue
        if observed == 1:
            if single_ok:
                continue
            problems.append(
                f"kernel group {group_id} declares one work item, but backend "
                f"{target.name!r} does not decompose a single-item launch over "
                f"the device; declare @allo.work(mapping={list(factors)})"
            )
            continue
        problems.append(
            f"kernel group {group_id} declares the work grid "
            f"{list(shape) if shape else observed} ({observed} items), which is "
            f"neither the unit-tree fanout {list(factors)} nor an outer prefix "
            f"of it ({[list(p) for p in prefixes[:-1]]})"
            + (" nor a single item" if single_ok else "")
            + "; no unit mapping exists for it"
        )
    if problems:
        raise ValueError(
            f"work grid does not partition target {target.name!r}: "
            + "; ".join(problems)
        )
    if len(groups) == 1:
        return len(buckets)
    return derived


def _resolve_host_moves(target, records):
    """Resolve recorded `host_xfer.*` calls against `target` (spec 001 D5).

    Returns `list[ResolvedHostMove]`. Each `HostMoveRecord` is dispatched via
    `BackendHandle.resolve` (verb identity + device-handle identity); the
    buffer role is the record's single non-handle arg (its parameter name when
    the workload passed a named buffer, else `None`). An empty / None `records`
    yields `[]` -> today's inferred-role run path.
    """
    if not records:
        return []
    from .spmw_target import BackendHandle, HandleToken

    try:
        from .spmw_target import _VerbCallOrToken

        _token_types = (HandleToken, _VerbCallOrToken)
    except ImportError:  # pragma: no cover - _VerbCallOrToken always present
        _token_types = (HandleToken,)

    bh = BackendHandle(target)
    resolved = []
    for rec in records:
        move = bh.resolve(rec)
        # Device handle = the resolved move's device endpoint (Memory/Register).
        device_handle = bh.__getattr__(  # by declared name -> identical object
            next(a for a in rec.args if isinstance(a, _token_types)).name
        )
        # Buffer role = the workload buffer arg (the non-token). Recover its
        # name when it is a string label; otherwise leave None (the run path
        # falls back to operand-shape inference for the actual array).
        buffers = [a for a in rec.args if not isinstance(a, _token_types)]
        role = None
        if buffers:
            b = buffers[0]
            role = b if isinstance(b, str) else getattr(b, "name", None)
        resolved.append(
            ResolvedHostMove(
                verb=rec.verb,
                move=move,
                device_handle=device_handle,
                buffer_role=role,
            )
        )
    return resolved


def _stamp_host_moves(stored_layout, resolved):
    """Stamp resolved host moves onto placement `extra["host_moves"]` (D5).

    Free-form provenance only; the cost arithmetic does not read the value to
    compute cycles, so numbers are unchanged. Handles a single Placement or a
    list of them.
    """
    layouts = stored_layout if isinstance(stored_layout, list) else [stored_layout]
    for pl in layouts:
        extra = getattr(pl, "extra", None)
        if isinstance(extra, dict):
            extra["host_moves"] = resolved


def _matcher_trace_runtime_abi(trace: MatchTrace) -> tuple:
    matches = []
    for match in trace.matches:
        scope = _matcher_work_scope(match)
        operands = tuple(
            (
                operand.role,
                operand.memref_name,
                tuple(map(str, operand.indices)),
                operand.memref_type,
                bool(operand.is_loop_carried),
                None if operand.value_ref is None else operand.value_ref.manifest(),
            )
            for operand in match.operands
        )
        matches.append(
            (
                scope.group_id,
                tuple(scope.work_id),
                match.target_op_name,
                tuple(
                    (str(var), str(lower), str(upper), int(step))
                    for var, lower, upper, step in match.enclosing_loops
                ),
                operands,
                match.result_memref_name,
                (
                    None
                    if match.result_value_ref is None
                    else match.result_value_ref.manifest()
                ),
            )
        )
    return tuple(matches)


def _materialize_matcher_codegen(
    target,
    trace: MatchTrace,
    layouts,
    *,
    host_moves=(),
) -> MatcherCodegenArtifact:
    """Emit and freeze the exact executable side of one scored candidate."""

    target_name = getattr(target, "name", None)
    if target_name not in _BACKENDS:
        raise NotImplementedError(f"no matcher codegen context for {target_name!r}")
    ctx_cls = backend_context_class(target_name)

    ctx = ctx_cls(target)
    _walk_and_emit(target, trace, ctx, layouts)
    # The realized program against the candidate's declared effects, at the
    # materialization boundary: a mismatch is a feasibility rejection the
    # ranking records, so another candidate can be selected.
    check_realization = getattr(ctx, "check_realization", None)
    if check_realization is not None:
        from .pim.schedule_search import InfeasibleSchedule

        try:
            check_realization(layouts)
        except RuntimeError as error:
            raise InfeasibleSchedule(
                f"realization differs from the candidate contract: {error}"
            ) from error
    commands = tuple(_clone_matcher_command(command) for command in ctx.cmds)
    host_schedule = tuple(
        HostTrigger(trigger.work_id, trigger.tile_count)
        for trigger in getattr(ctx, "host_schedule", ())
    )
    host_preloads = tuple(getattr(ctx, "host_preloads", ()))
    resolved_host_moves = tuple(host_moves)
    try:
        resolved_host_moves_manifest = _matcher_resolved_host_move_manifest(
            resolved_host_moves
        )
    except (AttributeError, TypeError, ValueError):
        resolved_host_moves_manifest = (
            ("unavailable", tuple(type(item).__name__ for item in resolved_host_moves)),
        )
    trace_abi = _matcher_trace_runtime_abi(trace)

    route, runtime_segments = ctx.runtime_route(commands)

    runtime_abi = (
        "matcher-runtime-abi-v1",
        target_name,
        route,
        trace_abi,
        resolved_host_moves_manifest,
    )
    return MatcherCodegenArtifact(
        target_name=target_name,
        commands=commands,
        host_schedule=host_schedule,
        host_preloads=host_preloads,
        resolved_host_moves=resolved_host_moves,
        runtime_abi=runtime_abi,
        runtime_segments=runtime_segments,
        context=ctx,
    )


def _single_launch_orchestration(name, trace, launch_schedule, host_moves, what):
    """Shared capability rule of the APU plan backends: the plan owns its
    launches and transfers, so an explicit host program is accepted only
    when it is the plan's own orchestration (one launch of every kernel
    group, no host transfer steps) and no explicit host moves are given.
    Anything else is refused here, before selection, instead of being
    dropped by the graph and the runtime."""

    from .spmw_plan import HostStep, LaunchStep

    if host_moves:
        raise NotImplementedError(
            f"{name}: explicit host transfers (host_moves) are not implemented for "
            f"{what}, which derive their own transfers; pass no host moves"
        )
    if launch_schedule is None:
        return
    from .spmw_autoschedule import _matcher_work_scope

    host_steps = [s for s in launch_schedule.steps if isinstance(s, HostStep)]
    launches = {}
    order = []
    for step in launch_schedule.steps:
        if isinstance(step, LaunchStep):
            launches[step.group_id] = launches.get(step.group_id, 0) + 1
            order.append(step.group_id)
    trace_order = []
    for match in trace.matches:
        group_id = _matcher_work_scope(match).group_id
        if group_id not in trace_order:
            trace_order.append(group_id)
    if order != trace_order:
        raise NotImplementedError(
            f"{name}: the host program launches kernel groups in the order {order}; "
            f"{what} execute the program's kernel groups once each in program "
            f"order {trace_order}, and another order or coverage is not implemented"
        )
    if host_steps or any(count != 1 for count in launches.values()):
        raise NotImplementedError(
            f"{name}: the host program requests "
            f"{sorted(launches.values())} launch(es) per kernel group and "
            f"{len(host_steps)} host transfer step(s); {what} execute one launch "
            "of the whole problem per kernel group and own their transfers. "
            "Repeated launches, interleaved transfers and launch ordering are not "
            "implemented for this backend; express them as separate compilations "
            "or drop the host program."
        )


def _select_named_candidates(view, trace, request):
    """Without a cost program, ``layout=request`` must designate exactly one
    legal candidate of every kernel; those candidates are the program."""

    from .spmw_autoschedule import (
        _ENUMERATORS,
        _bucket_for_autoschedule,
        derive_layout_properties,
        select_candidates,
    )
    from .spmw_legality import check_placement

    enumerator = _ENUMERATORS.get(getattr(view, "name", None))
    if enumerator is None:
        raise NotImplementedError(
            f"no autoscheduler enumerator registered for target "
            f"{getattr(view, 'name', None)!r}"
        )
    layouts = []
    for _scope, matches in _bucket_for_autoschedule(trace):
        candidates = select_candidates(enumerator(view, matches), request, matches)
        chosen = [
            candidate
            for candidate in candidates
            if not candidate.extra.get("candidate_excluded")
            and check_placement(view, matches, candidate).legal
        ]
        if len(chosen) != 1:
            names = sorted(candidate.mode for candidate in chosen)
            raise ValueError(
                f"layout={request!r} designates {len(chosen)} legal candidates of "
                f"kernel {matches[0].func_name!r} ({names}); pass a cost program to "
                "rank them or name one candidate"
            )
        (placement,) = chosen
        placement.extra = derive_layout_properties(view, placement)
        layouts.append(placement)
    return layouts


def carried_accumulator_message(target_name: str, match) -> str:
    """Why a backend that zeroes its accumulator cannot lower ``y[i] += ...``."""
    return (
        f"{target_name}: {match.target_op_name} in {match.func_name!r} "
        f"accumulates into {match.result_memref_name!r}, a caller-provided "
        "buffer the kernel never initializes (the `y[i] += ...` form, "
        "accumulator_init='carried'). This device zeroes its accumulator "
        "before each reduction, so the caller's initial value would be "
        "lost. Initialize a local accumulator with literal 0 before the "
        "reduction loop and store it to the output "
        "(`acc = 0; for k: acc += ...; y[i] = acc`); lowering the carried "
        "form with a host epilogue add is not implemented."
    )


def compile_for_target(
    target: Any,
    trace: MatchTrace,
    layout: Placement | list[Placement] | None = None,
    backend: "str | None" = None,
    host_moves: "list | None" = None,
    buffer_metrics: "dict | None" = None,
    cost=None,
    module=None,
    launch_schedule=None,
) -> Compiled:
    """Lower a (target, trace) pair to a runnable backend artifact by
    walking the target's declarations.

    If ``layout`` is None, the autoscheduler picks per-kernel
    placements. If ``layout`` is a single `Placement`, it is broadcast
    across every kernel (back-compat for single-layer tests). If
    ``layout`` is a list, its length must equal the number of
    `@allo.work` kernels in the trace.  Any other ``layout`` (a candidate
    name, a candidate family name or a backend plan object such as an
    ``APUV1Plan``) selects among the autoscheduler's candidates
    (`spmw_autoschedule.select_candidates`); the ranking runs over the
    selected subset.

    ``backend="virtual"`` executes the supplied CostSpec without a simulator.

    ``host_moves`` (spec 001 D5): an optional list of recorded ``host_xfer.*``
    calls (``HostMoveRecord``) from the workload's region body. ``None`` (the
    default) preserves today's inferred-role path. When supplied they are
    resolved against the target (verb + device-handle identity) and stored on
    ``Compiled.host_moves`` so the run path asserts the operand->role binding.

    ``module`` is the customized MLIR module, forwarded to the backend's
    ``emit_program`` hook. It is None when the caller has no module.

    ``launch_schedule`` (a ``spmw_plan.LaunchSchedule``) orders host transfers
    and kernel launches for codegen and the execution graph. Its host steps
    must index ``host_moves`` one to one. None keeps the legacy order.
    """
    target_name = getattr(target, "name", None)
    resolved_host_moves = _resolve_host_moves(target, host_moves)
    if launch_schedule is not None:
        from .spmw_plan import HostStep

        n_host_steps = sum(isinstance(step, HostStep) for step in launch_schedule.steps)
        if n_host_steps != len(resolved_host_moves):
            raise ValueError(
                f"launch schedule has {n_host_steps} host steps but "
                f"{len(resolved_host_moves)} host moves were resolved"
            )
    if trace.target_name != target_name:
        raise ValueError(
            f"trace.target_name {trace.target_name!r} != target.name {target_name!r}"
        )

    # Release gate G2: no dropped store and no silently changed accumulator
    # initial value, whichever route produced the trace.
    from .spmw_match_engine import require_match_coverage

    checked_trace = require_match_coverage(trace, target=target, module=module)
    # Samsung is not listed: its REDUCE candidates are pruned for a carried
    # accumulator and the ordered TEMPORAL program loads the initial value.
    # APU v1's grouped program of a dataflow region refuses a carried
    # accumulator in its enumerator; its vector plans carry the output.
    if target_name == "aim":
        carried = [
            match
            for match in checked_trace.matches
            if match.extra.get("accumulator_init") == "carried"
        ]
        if carried:
            first = carried[0]
            raise NotImplementedError(carried_accumulator_message(target_name, first))

    from .spmw_pipeline import matcher_view

    # Legality reads the target's declarations through its matcher view (the
    # target itself for every backend but Gemini-II).
    view = matcher_view(target)
    explicit_layout = layout is not None and (
        isinstance(layout, Placement)
        or (
            isinstance(layout, (list, tuple))
            and all(isinstance(item, Placement) for item in layout)
        )
    )
    candidate_filter = None if layout is None or explicit_layout else layout

    # Placement legality (release audit C-01): a caller-supplied layout must
    # respect the target's access scopes, operand patterns and moves.
    if explicit_layout:
        from .spmw_legality import check_layouts

        check_layouts(view, trace, layout).raise_if_illegal("user-supplied layout")

    # A virtual-only target needs no code generator. Its executable cost spec
    # lowers the trace directly to the retained handle-based graph.
    if backend == "virtual" and target_name not in _BACKENDS:
        if hasattr(target, "work_grid"):
            _check_work_grid(target, trace)
        if layout is None:
            stored_layout = Placement(placements={})
        elif isinstance(layout, Placement):
            stored_layout = layout
        else:
            layouts = list(layout)
            stored_layout = layouts[0] if len(layouts) == 1 else layouts
        execution_graph = None
        if cost is not None:
            from .spmw_plan import build_execution_graph

            execution_graph = build_execution_graph(
                target,
                trace,
                stored_layout,
                cost,
                host_moves=resolved_host_moves,
                buffer_metrics=buffer_metrics,
                launch_schedule=launch_schedule,
            )
        return Compiled(
            target,
            trace,
            [],
            stored_layout,
            ctx=None,
            backend=backend,
            host_moves=resolved_host_moves,
            execution_graph=execution_graph,
            cost=cost,
            launch_schedule=launch_schedule,
        )

    if target_name not in _BACKENDS:
        raise NotImplementedError(
            f"no codegen ctx for target {target_name!r}; supported: "
            f"{sorted(_BACKENDS)}"
        )
    ctx_cls = backend_context_class(target_name)

    # Orchestration capability (gate G9): a backend states which supplied
    # host programs and transfers it implements; anything else is refused
    # before selection instead of being dropped by the graph or the runtime.
    validate_orchestration = getattr(ctx_cls, "validate_orchestration", None)
    if validate_orchestration is not None:
        validate_orchestration(trace, launch_schedule, resolved_host_moves)

    # Release gate G2 (C-24): every externally visible write of the source
    # (matched stores and accepted copies into function arguments) must be
    # realized by this backend, or compilation stops here.
    from .spmw_match_engine import require_effects_realized

    declare = getattr(ctx_cls, "realized_effects", None)
    require_effects_realized(
        target, checked_trace, declare(checked_trace) if declare is not None else None
    )

    from .spmw_autoschedule import _bucket_for_autoschedule

    buckets = _bucket_for_autoschedule(trace)
    n_groups = len(buckets) or 1

    if not explicit_layout and cost is None:
        # No cost program: a request that names exactly one candidate per
        # kernel is materialized without ranking; anything else needs the
        # ranking (and therefore a cost program).
        if candidate_filter is None:
            raise ValueError(
                f"target {target_name!r}: candidates are selected by cost ranking; "
                "pass an executable cost program (allo.pim.costs) or choose a "
                "candidate with layout=<name>"
            )
        layouts = _select_named_candidates(view, trace, candidate_filter)
    elif not explicit_layout:
        layouts = autoschedule(
            target,
            trace,
            cost=cost,
            host_moves=resolved_host_moves,
            buffer_metrics=buffer_metrics,
            launch_schedule=launch_schedule,
            candidate_filter=candidate_filter,
        )
    elif isinstance(layout, Placement):
        # Single layout: replicate across every kernel. Equivalent to
        # the old behaviour for single-layer traces.
        layouts = [layout] * n_groups
    else:
        layouts = list(layout)

    if len(layouts) != len(buckets):
        raise ValueError(
            f"layout list has {len(layouts)} entries but trace has "
            f"{len(buckets)} @allo.work kernels"
        )
    if explicit_layout:
        # A supplied placement is a decision, not an executable: it is
        # rebound to the candidate of the current program with the same
        # decision after legality and the current reduction-order policy
        # (`spmw_candidate.rebind_supplied_layouts`); a backend plan from
        # another compilation is never executed as such.
        from .spmw_candidate import rebind_supplied_layouts

        layouts = rebind_supplied_layouts(view, trace, layouts)
    # The selected candidates' declared effects against the program.
    from .spmw_candidate import require_candidate_effects

    for (_scope, bucket_matches), placement in zip(buckets, layouts):
        require_candidate_effects(bucket_matches, placement, "selected placement", view)
    if not explicit_layout:
        # The autoscheduler filtered its candidates; re-check what survived
        # the cross-kernel reconciliation.
        from .spmw_legality import check_layouts

        check_layouts(view, trace, layouts).raise_if_illegal("autoscheduled layout")

    # The declared work grid must map onto the unit tree (release audit C-04).
    # Checked after selection so backend partition diagnostics raised while
    # materializing candidates (Samsung row coverage) report first.
    if hasattr(target, "work_grid"):
        _check_work_grid(target, trace, layouts=layouts)

    ctx = ctx_cls(target)
    # The backend the artifact will run on: a context may refuse a backend
    # its selected program cannot serve at materialization time.
    ctx.backend = backend
    _walk_and_emit(
        target,
        trace,
        ctx,
        layouts,
        schedule=launch_schedule,
        host_moves=resolved_host_moves,
        module=module,
    )
    # The realized program against the selected candidate's declared effects.
    check_realization = getattr(ctx, "check_realization", None)
    if check_realization is not None:
        check_realization(layouts)

    # `Compiled.layout` historically held a single Placement; preserve
    # that for back-compat when there's only one kernel.
    stored_layout = layouts[0] if len(layouts) == 1 else layouts
    # spec 001 D5: stamp the resolved host moves onto the placement(s)' `extra`
    # so the `host_staging` cost compose can ATTRIBUTE its (unchanged) M,K-derived
    # cost to the explicit `host_xfer.*` moves (provenance cross-check). Additive:
    # a placement without this key prices exactly as before.
    if resolved_host_moves:
        _stamp_host_moves(stored_layout, resolved_host_moves)
    execution_graph = None
    if cost is not None:
        from .spmw_plan import build_execution_graph

        execution_graph = build_execution_graph(
            target,
            trace,
            stored_layout,
            cost,
            host_moves=resolved_host_moves,
            buffer_metrics=buffer_metrics,
            launch_schedule=launch_schedule,
        )
    placement_ranking = getattr(layouts, "placement_ranking", None)
    if placement_ranking is not None and execution_graph is not None:
        # The selected program's whole-program cycles are those of the graph
        # retained for `estimate()` (docs/release/design/selection.md).
        from dataclasses import replace

        from .spmw_plan import _bind_cost

        placement_ranking = replace(
            placement_ranking,
            program_cycles=int(
                _bind_cost(cost, target).evaluate(execution_graph).cycles
            ),
        )
    return Compiled(
        target,
        trace,
        ctx.cmds,
        stored_layout,
        ctx=ctx,
        backend=backend,
        host_moves=resolved_host_moves,
        execution_graph=execution_graph,
        cost=cost,
        placement_ranking=placement_ranking,
        launch_schedule=launch_schedule,
    )
