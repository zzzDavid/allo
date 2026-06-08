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

import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .spmw_autoschedule import Placement, autoschedule
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

    def after_match(self, match, n_emitted: int) -> None:
        """Post-match hook. Default no-op; Samsung overrides to emit the
        inner-K JUMP that folds the reduction loop (spec 009 §E rule 4).
        """
        return None


# --------------------------------------------------------------------- #
# Samsung HBM-PIM backend
# --------------------------------------------------------------------- #


def _eval_sym(expr, env: dict | None = None) -> int | None:
    """Evaluate a SymExpr to a Python int, substituting UnitIds via ``env``.

    ``env`` maps ``(level, unit_name_or_None)`` to a concrete int. UnitIds
    not found in ``env`` default to 0 — this lets backends that emit one
    representative trace per work-item render symbolic bank indices like
    ``8*bg + bank`` as a concrete integer (here, 0). Returns ``None`` if
    the expression contains a non-numeric atom the evaluator can't reduce.
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
        # Default: representative lane 0 (autoscheduler emits one trace
        # per work-id; symbolic bg/bank placeholders collapse to bank 0).
        return 0
    if isinstance(expr, SymExpr):
        vals = [_eval_sym(a, env) for a in expr.args]
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

    # Read-only roles emit LD only; read-write roles emit LD + ST.
    # `acc` starts at 0 in the GEMV workload IR so its LD half is a
    # zero-init, not a real memory move -- see spec 009 §E rule 3.
    # TODO(task-015): revisit when accumulators carry across calls.
    _ROLE_IS_READ_ONLY = {"x": True, "y": True, "acc": False, "dst": False}

    def resolve_moves(self, role, src_handle=None, dst_handle=None):
        # dst_handle is the placement-side handle. For Samsung that's
        # either grf_a / grf_b (LD/ST target) or a bank-shaped MemoryRef
        # (in-place; no LD/ST needed).
        if isinstance(dst_handle, MemoryRef):
            return (None, None)
        if isinstance(dst_handle, Register):
            if dst_handle.name == "grf_a":
                ld, st = "LD_A", "ST_A"
            elif dst_handle.name == "grf_b":
                ld, st = "LD_B", "ST_B"
            else:
                return (None, None)
            read_only = self._ROLE_IS_READ_ONLY.get(role, False)
            if read_only:
                return (ld, None)
            # acc carries no prior value in the GEMV workload IR; emit
            # only the storeback for now (see spec 009 §E rule 3).
            if role == "acc":
                return (None, st)
            return (ld, st)
        return (None, None)

    # Active placement + resolved bindings for the current match, set by
    # `_walk_and_emit` before each compute emit. Default None keeps the
    # single-fiber path (no dual-fiber state) unchanged.
    _active_placement = None
    _active_bindings = None

    def after_match(self, match, n_emitted):
        # Inner-K JUMP -- the loop counter is computed off the
        # just-emitted MAC body, so the call must stay inside the
        # work-id window between compute and storeback.
        placement = self._active_placement
        # Lever 3 appends a `+crf_shared`/`+crf_per_workid` token to `mode`
        # for audit legibility (SPEC-025 §3.1); the lever-1 dual-fiber
        # signal is the base mode token, so split on the `+` joiner rather
        # than matching the whole (possibly-suffixed) string.
        base_mode = getattr(placement, "mode", "").split("+", 1)[0]
        if (
            placement is not None
            and base_mode == "dual_fiber"
            and match.target_op_name == "MAC"
        ):
            self._emit_dual_fiber_jumps(match, n_emitted)
            return
        _emit_inner_loop_jump(match, self, n_emitted)

    def _emit_dual_fiber_jumps(self, match, n_emitted):
        """Materialise the alternating (JUMP even, MAC odd, JUMP odd, ...)
        tail for a dual-fiber MAC.

        The walker already emitted the canonical MAC against the EVEN
        fiber (`placements[y]`). Here we close fiber 0's inner loop with
        its split JUMP, then for each later fiber emit one MAC (same
        dst/src0, src1 = that fiber's bank handle so `_bank_parity` stamps
        the parity) followed by its own split JUMP. Trip counts come from
        `inner_ub // lanes` split across the fibers — no shape literal.
        """
        if not match.enclosing_loops:
            return
        ub = _parse_loop_bound(match.enclosing_loops[-1][2])
        if ub is None:
            return
        folded = ub // _samsung_lane_burst(self)
        fibers = self._active_placement.extra.get("fibers", [])
        n_fibers = len(fibers)
        if n_fibers <= 1:
            _emit_inner_loop_jump(match, self, n_emitted)
            return
        split = _fiber_fold_split(folded, n_fibers)

        bindings = self._active_bindings or {}
        mac_dst = bindings.get("acc")
        mac_src0 = bindings.get("x")

        def emit_jump(trips):
            # First MAC of the fiber is iteration 0; JUMP loops the rest.
            loop_counter = trips - 1
            if loop_counter <= 0:
                return
            self.cmds.append(
                PIMCmd(
                    type_="JUMP",
                    loopCounter_=loop_counter,
                    loopOffset_=n_emitted + 1,
                )
            )

        # Fiber 0: the EVEN MAC the walker already emitted; just its JUMP.
        emit_jump(split[0])
        # Fibers 1..n-1: MAC against the fiber's bank handle, then JUMP.
        for i in range(1, n_fibers):
            self.cmd("MAC", dst=mac_dst, src0=mac_src0, src1=fibers[i])
            emit_jump(split[i])


# --------------------------------------------------------------------- #
# SK-Hynix AiM (GDDR6 PIM) backend
# --------------------------------------------------------------------- #


class AimCtx(CodegenContext):
    """Codegen ctx for AiM. Emits one text-format ISR line per `cmd(...)`.

    Output is a list of strings on `self.cmds` (NOT `PIMCmd` records —
    AiM's ramulator2 consumes a plain text trace). Each line is a
    space-separated, positional sequence of integer fields in the order
    ramulator2 expects per opcode (see `_ISR_FIELDS`). A side-channel
    `_human_lines` mirror keeps the older `key=value` annotation that
    `_operand_text` produces — useful for unit tests that inspect the
    rendered trace by substring.
    """

    # Per-opcode field ordering (positional, matches
    # ``Ramulator::AiMISRInfo::opcode_str_to_aim_ISR`` in request.h).
    # Each tuple is the legal field sequence; values are pulled from
    # the field map built by `_field_map(...)`.
    _ISR_FIELDS = {
        "WR_SBK":     ("gpr_addr_0", "channel_mask", "bank_index", "row_addr"),
        "WR_ABK":     ("gpr_addr_0", "channel_mask", "row_addr"),
        "WR_GB":      ("opsize", "gpr_addr_0", "channel_mask"),
        "WR_BIAS":    ("gpr_addr_0", "channel_mask"),
        "WR_AFLUT":   ("opsize",),
        "RD_MAC":     ("gpr_addr_0", "channel_mask"),
        "RD_AF":      ("gpr_addr_0", "channel_mask"),
        "RD_SBK":     ("gpr_addr_0", "channel_mask", "bank_index", "row_addr"),
        "COPY_BKGB":  ("opsize", "channel_mask", "bank_index", "row_addr"),
        "COPY_GBBK":  ("opsize", "channel_mask", "bank_index", "row_addr"),
        "MAC_SBK":    ("opsize", "channel_mask", "bank_index", "row_addr"),
        "MAC_ABK":    ("opsize", "channel_mask", "row_addr"),
        "AF":         ("channel_mask",),
        "EWMUL":      ("opsize", "channel_mask", "row_addr"),
        "EWADD":      ("opsize", "gpr_addr_0", "gpr_addr_1"),
        "SYNC":       (),
        "EOC":        (),
    }

    def __init__(self, target):
        super().__init__(target)
        # Override the parent's PIMCmd list with a plain text-line list —
        # AiM's ramulator2 frontend consumes a text trace.
        self.cmds: list[str] = []
        # Parallel list of the human-readable key=value annotation per
        # emitted line; populated by `cmd()` so tests can introspect.
        self._human_lines: list[str] = []

    def _operand_text(self, handle, role: str) -> str:
        """Render a Tenon handle as the AiM trace field set it implies.

        This is the human-readable `key=value` annotation; positional
        emission flows through `_handle_to_fields`. Tests that previously
        asserted on this format inspect `self._human_lines`.
        """
        if handle is None:
            return ""
        if isinstance(handle, Register):
            if handle.name == "gpr":
                prefix = "gpr_in" if role.startswith("src") else "gpr_out"
                return f"{prefix}=0"
            if handle.name == "bias":
                return "bias=0"
            raise NotImplementedError(
                f"AimCtx: register {handle.name!r} has no operand mapping."
            )
        if isinstance(handle, MemoryRef):
            mem = handle.memory
            if mem.name == "banks":
                # `handle.idx` may be a symbolic SymExpr (e.g. autoscheduler
                # produces `8*bg + bank` with placeholder UnitIds); collapse
                # to a concrete int via `_eval_sym` so ramulator2 accepts it.
                bank_int = _eval_sym(handle.idx)
                if bank_int is None:
                    raise NotImplementedError(
                        f"AimCtx: bank index {handle.idx!r} cannot be reduced "
                        "to an integer; supply concrete UnitId values."
                    )
                return f"bank={bank_int} row=0"
            if mem.name == "gb":
                return "gb=1"
            raise NotImplementedError(
                f"AimCtx: memory {mem.name!r} has no operand mapping."
            )
        # Memory (whole-memory broadcast operand, e.g. WR_ABK src0=banks).
        from .spmw_target import Memory
        if isinstance(handle, Memory):
            if handle.name == "banks":
                return "bank=ALL row=0"
            if handle.name == "gb":
                return "gb=1"
            raise NotImplementedError(
                f"AimCtx: whole-memory {handle.name!r} has no operand mapping."
            )
        raise NotImplementedError(
            f"AimCtx: unknown handle type {type(handle).__name__}."
        )

    def _handle_to_fields(self, handle, role: str, fields: dict) -> None:
        """Update `fields` with the integer values implied by `handle`.

        Keys written: `gpr_addr_0`, `gpr_addr_1`, `bank_index`, `row_addr`,
        and `channel_mask` for whole-memory broadcasts. Unknown handles
        are tolerated — the caller's ISR-field selector picks only the
        fields the opcode declares as legal.
        """
        if handle is None:
            return
        from .spmw_target import Memory
        if isinstance(handle, Register):
            if handle.name in ("gpr", "bias"):
                # gpr_addr_1 hosts the second GPR for ISRs that need two
                # (EWADD); the first src GPR routes to gpr_addr_0.
                if role == "src1" and "gpr_addr_0" in fields:
                    fields["gpr_addr_1"] = 0
                else:
                    fields.setdefault("gpr_addr_0", 0)
            return
        if isinstance(handle, MemoryRef):
            mem = handle.memory
            if mem.name == "banks":
                bank_int = _eval_sym(handle.idx)
                if bank_int is None:
                    raise NotImplementedError(
                        f"AimCtx: bank index {handle.idx!r} cannot be reduced "
                        "to an integer; supply concrete UnitId values."
                    )
                fields["bank_index"] = bank_int
                fields.setdefault("row_addr", 0)
            elif mem.name == "gb":
                fields.setdefault("channel_mask", 1)
            return
        if isinstance(handle, Memory):
            if handle.name == "banks":
                # WR_ABK / MAC_ABK target every bank in a channel; the
                # parser still wants a row_addr field.
                fields.setdefault("row_addr", 0)
            elif handle.name == "gb":
                fields.setdefault("channel_mask", 1)
            return

    def cmd(self, name: str, dst=None, src0=None, src1=None, **fields):
        """Emit one ISR line in ramulator2's positional format.

        `fields` may carry overrides (e.g. `row_addr=8`, `opsize=2`);
        unspecified fields default to 0 (or 1 for `channel_mask`).
        Unknown opcodes fall back to the legacy key=value form so
        callers using non-canonical opcode names still produce a
        parseable diagnostic line.
        """
        # Build the integer field map from operand handles + overrides.
        field_map: dict = {}
        self._handle_to_fields(src0, "src0", field_map)
        self._handle_to_fields(src1, "src1", field_map)
        self._handle_to_fields(dst, "dst", field_map)
        # Caller overrides (e.g. `row_addr=...`, `opsize=...`).
        for k, v in fields.items():
            field_map[k] = v

        # Human-readable mirror — keeps the older key=value annotation so
        # tests can introspect handle-derived field labels.
        parts = [f"AiM {name}", "ch=0"]
        if src0 is not None:
            parts.append(self._operand_text(src0, "src0"))
        if src1 is not None:
            parts.append(self._operand_text(src1, "src1"))
        if dst is not None:
            parts.append(self._operand_text(dst, "dst"))
        for k, v in fields.items():
            parts.append(f"{k}={v}")
        human = "  ".join(p for p in parts if p)
        self._human_lines.append(human)

        # Positional output — ramulator2 expects space-separated ints in
        # the order declared by `_ISR_FIELDS[opcode]`. Defaults:
        # gpr/opsize/bank/row -> 0; channel_mask -> 1 (channel 0 enabled).
        order = self._ISR_FIELDS.get(name)
        if order is None:
            # Unknown opcode -- preserve diagnostic mode by emitting the
            # key=value annotation. ramulator2 will reject it, which is
            # the correct behaviour (caller asked for an opcode the spec
            # doesn't know about).
            self.cmds.append(human)
            return
        defaults = {
            "gpr_addr_0": 0,
            "gpr_addr_1": 0,
            "opsize": 1,
            "bank_index": 0,
            "row_addr": 0,
            "channel_mask": 1,
        }
        tokens = [f"AiM {name}"]
        for fld in order:
            tokens.append(str(field_map.get(fld, defaults[fld])))
        self.cmds.append(" ".join(tokens))

    def append(self, line: str):
        """Low-level escape hatch — append a raw trace line."""
        self.cmds.append(line)

    def emit_eoc(self):
        """End-of-command marker — every AiM trace ends with EOC."""
        self.cmds.append("AiM EOC")
        self._human_lines.append("AiM EOC")

    def resolve_moves(self, role, src_handle=None, dst_handle=None):
        from .spmw_target import Memory
        if dst_handle is None:
            return (None, None)
        if isinstance(dst_handle, MemoryRef):
            mem = dst_handle.memory
            if mem.name == "banks":
                # per-bank: WR_SBK in, RD_SBK out
                return ("WR_SBK", "RD_SBK")
            if mem.name == "gb":
                return ("WR_GB", None)
            return (None, None)
        if isinstance(dst_handle, Memory):
            if dst_handle.name == "banks":
                # all-bank broadcast write; no readback path
                return ("WR_ABK", None)
            if dst_handle.name == "gb":
                return ("WR_GB", None)
            return (None, None)
        if isinstance(dst_handle, Register):
            # gpr is the MAC accumulator target -- read-back the accumulated
            # value via RD_MAC at storeback time for the `acc` role.
            if dst_handle.name == "gpr":
                if role == "acc":
                    return (None, "RD_MAC")
                return (None, None)
            # TODO(task-017): per-bank bias handle collapse -- `bias`
            # registers are declared per bank but the flat handle map
            # only retains one. WR_BIAS is per-bank, so without the
            # rework we can only safely emit one per work-id.
            if dst_handle.name == "bias":
                return ("WR_BIAS", None)
            return (None, None)
        return (None, None)

    def after_match(self, match, n_emitted):
        """Fold the inner K reduction into the just-emitted MAC ISR's
        ``opsize`` field. ramulator2 prices ``MAC_SBK opsize=N`` by
        issuing N column-address requests (see SPEC-019 §3.1 for the
        simulator citation). Tactical no-ops:

        - non-MAC matches: nothing to fold;
        - ``n_emitted != 1``: compound emits (e.g. a future MAC_ABK +
          RD_MAC pair) need a wider fold -- until then leave opsize=1;
        - empty ``enclosing_loops`` (synthetic traces from
          ``test_run.py`` / ``test_target_aim.py``): preserve
          pre-SPEC-019 behaviour;
        - inner-loop ub not reducible to a positive constant > 1: same
          fall-back, matching the existing Samsung JUMP path.
        """
        if match.target_op_name != "MAC":
            return
        if n_emitted != 1:
            return
        if not match.enclosing_loops:
            return
        inner = match.enclosing_loops[-1]
        k = _parse_loop_bound(inner[2])
        if k is None or k <= 1:
            return
        # Rewrite the last emitted positional trace line. Expected shape:
        # ``AiM MAC_SBK <opsize> <channel_mask> <bank_index> <row_addr>``
        # (field order from ``_ISR_FIELDS["MAC_SBK"]``).
        last = self.cmds[-1]
        parts = last.split(" ")
        if len(parts) < 3 or parts[0:2] != ["AiM", "MAC_SBK"]:
            return
        parts[2] = str(k)
        self.cmds[-1] = " ".join(parts)
        # Mirror the key=value annotation so ``_human_lines`` stays
        # consistent with the positional trace for introspection.
        import re
        human_last = self._human_lines[-1]
        if "opsize=" in human_last:
            self._human_lines[-1] = re.sub(
                r"opsize=\d+", f"opsize={k}", human_last)
        else:
            self._human_lines[-1] = f"{human_last}  opsize={k}"


# --------------------------------------------------------------------- #
# UPMEM (DPU, DRAM-PIM) backend
# --------------------------------------------------------------------- #


class UPMEMCtx(CodegenContext):
    """Codegen ctx for UPMEM DPU tasklets.

    Output is a list of C source lines on `self.cmds`. The coder
    assembles them into a `task.c` with the standard DPU includes and
    `main()` wrapper. Handle-to-C-name translation is table-driven so
    workload-side memref names flow through unchanged.
    """

    def __init__(self, target):
        super().__init__(target)
        # Override the parent's PIMCmd list — UPMEM emits raw C text.
        self.cmds: list[str] = []
        # Map from handle id -> C variable name. Populated lazily the
        # first time a handle is referenced.
        self._name_table: dict[int, str] = {}
        # SPEC-019: channel for the inner-K bound, set by _walk_and_emit
        # immediately before invoking the MAC emit lambda. The fixture's
        # MAC emit reads this to materialise an explicit C `for` loop so
        # uPIMulator prices the actual K MACs, not a single statement.
        self.pending_k_bound: int | None = None

    # WRAM/MRAM memory names map onto the envelope's fixed buffer
    # parameters of `tenon_kernel(T *bufferB, T *bufferA, ...)`. Both
    # `wram` and `mram` lower to `bufferA` because the SDK outer loop in
    # `main_kernel1` stages MRAM into `cache_A`/`cache_B` and passes them
    # in as `bufferA`/`bufferB`; the emitted body sees the WRAM cache.
    # Picking `bufferA` for both inputs and accumulator is a deliberate
    # cycle-only simplification (see SPEC-003 §7 deferred work — real
    # role-aware naming needs the envelope to grow per workload shape).
    _ENVELOPE_MEM_NAME = {
        "wram": "bufferA",
        "mram": "bufferA",
    }

    def handle_c_name(self, handle) -> str:
        """Return the C identifier this handle lowers to.

        `Register`s use their declared name (the C compiler manages real
        register assignment). `MemoryRef`s render as
        `<env_name>[<idx>]`. Whole `Memory`s render as `<env_name>`,
        where `<env_name>` is the envelope-fixed buffer parameter name
        for known memories (wram/mram -> bufferA) and a `<mem.name>_buf`
        fallback otherwise.
        """
        from .spmw_target import Memory
        key = id(handle)
        cached = self._name_table.get(key)
        if cached is not None:
            return cached
        if isinstance(handle, Register):
            name = handle.name or f"reg_{key}"
            self._name_table[key] = name
            return name
        if isinstance(handle, MemoryRef):
            mem = handle.memory
            env_name = self._ENVELOPE_MEM_NAME.get(mem.name, f"{mem.name}_buf")
            name = f"{env_name}[{handle.idx!r}]"
            self._name_table[key] = name
            return name
        if isinstance(handle, Memory):
            env_name = self._ENVELOPE_MEM_NAME.get(handle.name, f"{handle.name}_buf")
            self._name_table[key] = env_name
            return env_name
        raise NotImplementedError(
            f"UPMEMCtx: unknown handle type {type(handle).__name__}."
        )

    def emit_c_line(self, line: str) -> None:
        """Append one C source line."""
        self.cmds.append(line)

    def cmd(self, name: str, **fields) -> None:
        """Compatibility shim — emits a comment line so the unified
        walker can call `ctx.cmd(...)` if it needs to. Real emits go
        through `emit_c_line`."""
        rendered = " ".join(f"{k}={v!r}" for k, v in fields.items())
        self.emit_c_line(f"/* {name} {rendered} */")

    def append(self, line: str) -> None:
        """Low-level escape hatch."""
        self.cmds.append(line)

    def resolve_moves(self, role, src_handle=None, dst_handle=None):
        from .spmw_target import Memory
        if dst_handle is None:
            return (None, None)
        # Whole-MRAM placements drive bulk MRAM↔WRAM copies; WRAM↔GPR is
        # handled by the C compiler so wram/gprs placements are no-ops.
        if isinstance(dst_handle, Memory):
            if dst_handle.name == "mram":
                return ("LD_MRAM", "ST_MRAM")
            return (None, None)
        if isinstance(dst_handle, MemoryRef):
            mem = dst_handle.memory
            if mem.name == "mram":
                return ("LD_MRAM", "ST_MRAM")
            return (None, None)
        if isinstance(dst_handle, Register):
            return (None, None)
        return (None, None)

    def emit_mac_kreduce(self, acc, x, y, k_bound) -> None:
        """Emit a K-reduction MAC body (SPEC-019 §4.3).

        When `k_bound` is a positive int and both x/y render as
        subscripted memrefs, emits:
            for (unsigned k = 0; k < <k_bound>; ++k) {
                <acc> += <x>[k] * <y>[k];
            }
        Otherwise falls back to the un-looped form `<acc> += <x> * <y>;`
        — preserves pre-SPEC-019 behaviour for synthetic test traces
        (empty enclosing_loops) and for register-operand MACs where
        per-K subscripting is not meaningful.
        """
        acc_c = self.handle_c_name(acc)
        x_c = self.handle_c_name(x)
        y_c = self.handle_c_name(y)

        # `handle_c_name` for a MemoryRef returns `<env_name>[<idx>]`;
        # for K-reduction we want the per-K subscript inside the loop,
        # so we strip the rendered `[<idx>]` suffix and append `[k]`.
        # Register / whole-Memory operands have no `[...]` suffix and
        # are not k-indexable.
        def _kify(c_name: str):
            if c_name.endswith("]"):
                head = c_name.rsplit("[", 1)[0]
                return f"{head}[k]"
            return None

        x_k = _kify(x_c)
        y_k = _kify(y_c)
        if k_bound is None or k_bound <= 1 or x_k is None or y_k is None:
            # Fallback identical to the pre-SPEC-019 emit (one line). The
            # spec's §4.5 invariant requires synthetic UPMEM traces with
            # empty enclosing_loops to keep producing a single statement
            # (test_target_upmem.py::test_upmem_ctx_emits_c and
            # test_run.py's _upmem_mac_trace both assert this). The
            # spec's helper-code sketch also emitted a `/* k bound
            # unparseable */` debug comment in this branch; we drop it
            # to preserve the structural one-line invariant — direct
            # ctx.emit calls (no walker) cannot be distinguished from a
            # walker call where the bound failed to parse without a
            # separate channel.
            self.emit_c_line(f"{acc_c} += {x_c} * {y_c};")
            return
        self.emit_c_line(f"for (unsigned k = 0; k < {k_bound}; ++k) {{")
        self.emit_c_line(f"    {acc_c} += {x_k} * {y_k};")
        self.emit_c_line("}")

    def get_kernel_src(self) -> str:
        """Return the assembled DPU kernel C source.

        Emits a full PrIM-shaped DPU envelope (DPU_INPUT_ARGUMENTS,
        kernels[] dispatch table, BARRIER_INIT, MRAM<->WRAM staging in
        main_kernel1) and inlines `self.cmds` as the body of
        `tenon_kernel(T *bufferB, T *bufferA, unsigned int l_size)`. The
        envelope is fixed and shared across all VA-shape / MAC-shape
        emitted kernels; a richer envelope selector is left to a
        follow-up task (see SPEC-003 §6).
        """
        # Each cmd is expected to be a valid C statement (terminated `;`
        # or a `{...}` block). UPMEMCtx.emit_c_line / .cmd already
        # produce that.
        body_lines = []
        for line in self.cmds:
            if not isinstance(line, str):
                continue
            body_lines.append("    " + line)
        body = "\n".join(body_lines) if body_lines else "    /* empty body */"
        return (
            "#include <stdint.h>\n"
            "#include <stdio.h>\n"
            "#include <defs.h>\n"
            "#include <mram.h>\n"
            "#include <alloc.h>\n"
            "#include <perfcounter.h>\n"
            "#include <barrier.h>\n"
            "\n"
            '#include "../support/common.h"\n'
            "\n"
            "__host dpu_arguments_t DPU_INPUT_ARGUMENTS;\n"
            "\n"
            "void __attribute__ ((noinline))\n"
            "tenon_kernel(T *bufferB, T *bufferA, unsigned int l_size) {\n"
            "    /* === BEGIN tenon-emitted body === */\n"
            f"{body}\n"
            "    /* === END tenon-emitted body === */\n"
            "}\n"
            "\n"
            "BARRIER_INIT(my_barrier, NR_TASKLETS);\n"
            "\n"
            "extern int main_kernel1(void);\n"
            "int (*kernels[nr_kernels])(void) = {main_kernel1};\n"
            "\n"
            "int main(void) {\n"
            "    return kernels[DPU_INPUT_ARGUMENTS.kernel]();\n"
            "}\n"
            "\n"
            "int main_kernel1(void) {\n"
            "    unsigned int tasklet_id = me();\n"
            "    if (tasklet_id == 0) { mem_reset(); }\n"
            "    barrier_wait(&my_barrier);\n"
            "\n"
            "    uint32_t input_size_dpu_bytes = DPU_INPUT_ARGUMENTS.size;\n"
            "    uint32_t input_size_dpu_bytes_transfer = DPU_INPUT_ARGUMENTS.transfer_size;\n"
            "    uint32_t base_tasklet = tasklet_id << BLOCK_SIZE_LOG2;\n"
            "    uint32_t mram_base_addr_A = (uint32_t)DPU_MRAM_HEAP_POINTER;\n"
            "    uint32_t mram_base_addr_B = (uint32_t)(DPU_MRAM_HEAP_POINTER + input_size_dpu_bytes_transfer);\n"
            "\n"
            "    T *cache_A = (T *) mem_alloc(BLOCK_SIZE);\n"
            "    T *cache_B = (T *) mem_alloc(BLOCK_SIZE);\n"
            "\n"
            "    for (unsigned int byte_index = base_tasklet;\n"
            "         byte_index < input_size_dpu_bytes;\n"
            "         byte_index += BLOCK_SIZE * NR_TASKLETS) {\n"
            "        uint32_t l_size_bytes = (byte_index + BLOCK_SIZE >= input_size_dpu_bytes)\n"
            "            ? (input_size_dpu_bytes - byte_index) : BLOCK_SIZE;\n"
            "\n"
            "        mram_read((__mram_ptr void const*)(mram_base_addr_A + byte_index), cache_A, l_size_bytes);\n"
            "        mram_read((__mram_ptr void const*)(mram_base_addr_B + byte_index), cache_B, l_size_bytes);\n"
            "\n"
            "        tenon_kernel(cache_B, cache_A, l_size_bytes >> DIV);\n"
            "\n"
            "        mram_write(cache_B, (__mram_ptr void*)(mram_base_addr_B + byte_index), l_size_bytes);\n"
            "    }\n"
            "    return 0;\n"
            "}\n"
        )


# --------------------------------------------------------------------- #
# GSI APU v1 (Gemini 1) backend
# --------------------------------------------------------------------- #


class APUv1Ctx(CodegenContext):
    """Codegen ctx for GSI APU v1.

    Output is a list of C-source lines on `self.cmds` (each is a single
    GVML call). Handle-to-C translation is by canonical name: L4 maps to
    one of `inp_L4ptr` / `wgt_L4ptr` / `out_L4ptr` keyed by the operand
    role; L1 maps to `GVML_VM_<idx>`; VRs use either an autoscheduler-
    bound C alias (see `bind_handle`) or fall back to the register's
    declared name. MAC has no fused opcode -- `emit_mac_lookup` expands
    it into `gvml_lookup_16(..., mac_lut_ptr, 256)` + `gvml_add_s16` so
    backend emit lambdas stay one-liners.
    """

    _L4_PTR_BY_ROLE = {
        "x": "inp_L4ptr",
        "y": "wgt_L4ptr",
        "acc": "out_L4ptr",
        "dst": "out_L4ptr",
    }

    def __init__(self, target):
        super().__init__(target)
        # Override the parent's PIMCmd list -- APU v1 emits plain C text.
        self.cmds: list[str] = []
        # Reverse-handle map populated by the autoscheduler before
        # walk_and_emit runs. Maps id(handle) -> C symbol. Also accepts
        # string keys (e.g. "mac_tmp") for scratch slots that aren't
        # backed by a Tenon handle.
        self._handle_names: dict = {}

    def bind_handle(self, handle, c_name: str) -> None:
        """Teach the ctx that `handle` lowers to the C identifier
        `c_name`. Called by the autoscheduler (or the test harness) once
        per placement -- e.g. `bind_handle(target.vrs, "inp_vr0")`.

        `handle` may be a Tenon handle (`Register`/`MemoryRef`/`Memory`)
        or a string label used to name scratch slots like `"mac_tmp"`.
        """
        key = handle if isinstance(handle, str) else id(handle)
        self._handle_names[key] = c_name

    def _name(self, handle, role: str = "") -> str:
        from .spmw_target import Memory
        cached = self._handle_names.get(id(handle))
        if cached is not None:
            return cached
        if isinstance(handle, Register):
            return handle.name or "vr_unknown"
        if isinstance(handle, MemoryRef):
            mem = handle.memory
            if mem.name == "l4":
                return self._L4_PTR_BY_ROLE.get(role, "inp_L4ptr")
            if mem.name == "l1":
                return f"GVML_VM_{handle.idx!r}"
            return f"{mem.name}_ref"
        if isinstance(handle, Memory):
            if handle.name == "l4":
                return self._L4_PTR_BY_ROLE.get(role, "inp_L4ptr")
            if handle.name == "l1":
                return "GVML_VM_0"
            return f"{handle.name}_ref"
        raise NotImplementedError(
            f"APUv1Ctx: unknown handle type {type(handle).__name__}."
        )

    def cmd(self, name: str, dst=None, src0=None, src1=None, **kwargs):
        """Emit one GVML call as a C line."""
        operands = []
        if dst is not None:
            operands.append(self._name(dst, "dst"))
        if src0 is not None:
            operands.append(self._name(src0, "src0"))
        if src1 is not None:
            operands.append(self._name(src1, "src1"))
        for k, v in kwargs.items():
            operands.append(f"/* {k}= */ {v!r}")
        self.cmds.append(name + "(" + ", ".join(operands) + ");")

    def emit_mac_lookup(self, acc, x, y) -> None:
        """Expand MAC into the GSI sv-lookup pattern:

            gvml_lookup_16(<tmp>, <x>, mac_lut_ptr, 256);
            gvml_add_s16(<acc>, <acc>, <tmp>);

        Semantics: `<tmp>[i] = mac_lut_ptr[<x>[i]]`. The autoscheduler
        must pre-pack the byte-pair index into `x` before this MAC fires
        (today's matcher passes the same VR for `x` and `y`, so binary
        MAC must encode both operands into the `x` VR upstream of this
        call -- out of scope for SPEC-018; a TODO for SPEC-018b).

        `mac_lut_ptr` resolves to a 256-entry `uint16_t` popcount table
        declared in the build harness's emitted `device.c`. The LUT lives
        inline in the cmd struct (`data->mac_lut`); see SPEC-018 §3.1-§3.3.
        Length is fixed at 256.

        The autoscheduler may reserve a scratch VR alias via
        `bind_handle("mac_tmp", "<alias>")`; otherwise the canonical name
        `mac_tmp_vr` is used. `y` is accepted for signature symmetry with
        `emit_mac_mul_add` but is not referenced -- the byte pair is
        packed into `x` upstream.
        """
        tmp_name = self._handle_names.get("mac_tmp", "mac_tmp_vr")
        acc_n = self._name(acc, "acc")
        x_n = self._name(x, "x")
        self.cmds.append(
            f"gvml_lookup_16({tmp_name}, {x_n}, mac_lut_ptr, 256);"
        )
        self.cmds.append(f"gvml_add_s16({acc_n}, {acc_n}, {tmp_name});")

    def emit_mac_mul_add(self, acc, x, y) -> None:
        """Expand MAC into the raw SV-mode pattern (no lookup table):

            gvml_mul_u16(<tmp>, <x>, <y>);
            gvml_add_s16(<acc>, <acc>, <tmp>);

        Selected by the walker when `placement.mode == "sv"` (per
        SPEC-009 §2). Cost-modelled at 18 cyc/MAC vs 8 cyc/MAC for
        the lookup expansion, so autoschedule prefers `emit_mac_lookup`
        unless the user overrides.
        """
        tmp_name = self._handle_names.get("mac_tmp", "mac_tmp_vr")
        acc_n = self._name(acc, "acc")
        x_n = self._name(x, "x")
        y_n = self._name(y, "y")
        self.cmds.append(f"gvml_mul_u16({tmp_name}, {x_n}, {y_n});")
        self.cmds.append(f"gvml_add_s16({acc_n}, {acc_n}, {tmp_name});")

    def append(self, line: str) -> None:
        """Low-level escape hatch -- append a raw C source line."""
        self.cmds.append(line)

    def iter_vr_aliases(self) -> list[tuple[str, str]]:
        """Return [(c_name, gvml_vr_enum), ...] in bind order.

        Each entry becomes one line `enum gvml_vr16 <c_name> = <enum>;`
        in the emitted device.c. Until the regalloc spec rebinds VRs
        per live-range, the autoscheduler's bind list may be empty; in
        that case the build harness substitutes a canonical default
        (vrs + mac_tmp_vr) so the emitted body still compiles.
        """
        out: list[tuple[str, str]] = []
        for i, (_key, c_name) in enumerate(self._handle_names.items()):
            out.append((c_name, f"GVML_VR16_{i}"))
        return out

    def iter_l4_roles(self) -> list[str]:
        """Return the L4 pointer C names referenced by `self.cmds`, in
        canonical role-table order (inp, wgt, out). Used by the build
        harness to emit one `gal_mem_handle_to_apu_ptr` decl per
        actual reference (avoiding unused-variable warnings)."""
        names: set[str] = set()
        for line in self.cmds:
            for ptr in self._L4_PTR_BY_ROLE.values():
                if ptr in line:
                    names.add(ptr)
        role_order = ["inp_L4ptr", "wgt_L4ptr", "out_L4ptr"]
        return [n for n in role_order if n in names]

    def resolve_moves(self, role, src_handle=None, dst_handle=None):
        from .spmw_target import Memory
        if dst_handle is None:
            return (None, None)
        # Chained L4↔VR moves cover the two-stage DMA + LD/ST path; VR
        # placements are no-ops (operands already live in VRs).
        if isinstance(dst_handle, Memory):
            if dst_handle.name == "l4":
                return ("LD_L4_TO_VR", "ST_VR_TO_L4")
            if dst_handle.name == "l1":
                return ("LD_VR", "ST_VR")
            return (None, None)
        if isinstance(dst_handle, MemoryRef):
            mem = dst_handle.memory
            if mem.name == "l4":
                return ("LD_L4_TO_VR", "ST_VR_TO_L4")
            if mem.name == "l1":
                return ("LD_VR", "ST_VR")
            return (None, None)
        if isinstance(dst_handle, Register):
            return (None, None)
        return (None, None)


# --------------------------------------------------------------------- #
# GSI APU v2 (Gemini 2, G2) backend
# --------------------------------------------------------------------- #


class APUv2Ctx(CodegenContext):
    """Codegen ctx for GSI APU v2.

    Output is a list of C++ source lines on `self.cmds`. Each entry is
    one GTML call (e.g. `g.add(out, lhs, rhs);`). `get_program_src()`
    wraps them with the standard GTML singleton boilerplate to produce
    a compilable `.cc` file.
    """

    def __init__(self, target):
        super().__init__(target)
        # Override the parent's PIMCmd list -- APU v2 emits C++ text.
        self.cmds: list[str] = []
        self._handle_names: dict[int, str] = {}
        self._tmp_counter = 0

    def bind_handle(self, handle, cpp_name: str) -> None:
        """Teach the ctx a handle's C++ container variable name.

        The autoscheduler populates this before walk_and_emit.
        """
        self._handle_names[id(handle)] = cpp_name

    def _name(self, handle) -> str:
        cached = self._handle_names.get(id(handle))
        if cached is not None:
            return cached
        if isinstance(handle, MemoryRef):
            mem = handle.memory
            return f"{mem.name}_ref_{handle.idx!r}"
        return f"opd_{id(handle)}"

    def cmd(self, name: str, dst=None, src0=None, src1=None, **kwargs):
        """Emit one GTML call as a C++ line.

        `name` is the fully-qualified function name (e.g. `"g.matmul"`
        or `"copy_to_l1"`). Operand order: dst first (out-parameter
        convention in GTML), then src0, src1.
        """
        parts = []
        if dst is not None:
            parts.append(self._name(dst))
        if src0 is not None:
            parts.append(self._name(src0))
        if src1 is not None:
            parts.append(self._name(src1))
        for k, v in kwargs.items():
            parts.append(f"/* {k}= */ {v!r}")
        self.cmds.append(f"{name}({', '.join(parts)});")

    def emit_mac_matmul(self, acc, x, y) -> None:
        """Expand MAC into the GTML matmul + add pattern.

        For a length-K dot product per work-item, this is one
        `g.matmul` (rank-1 x rank-1 -> scalar) followed by an add into
        the running acc. The canonical declarative form emits the
        per-iter expansion; a fully-vectorised matmul would replace
        this with a single outer-granularity `g.matmul`.
        """
        acc_n = self._name(acc)
        x_n = self._name(x)
        y_n = self._name(y)
        tmp = f"mac_tmp_{self._tmp_counter}"
        self._tmp_counter += 1
        self.cmds.append(f"L1Container {tmp} = g.alloc_vector();")
        self.cmds.append(f"g.matmul({tmp}, {x_n}, {y_n});")
        self.cmds.append(f"g.add({acc_n}, {acc_n}, {tmp});")

    def append(self, line: str) -> None:
        """Low-level escape hatch -- append a raw C++ source line."""
        self.cmds.append(line)

    def resolve_moves(self, role, src_handle=None, dst_handle=None):
        # l1_sim ignores moves at functional level; the cost model
        # already prices COPY_TO_L1 / COPY_FROM_L1 for autoschedule.
        return (None, None)

    def get_program_src(self) -> str:
        """Return the assembled G2 program source.

        Wraps `self.cmds` with the standard GTML singleton accessor
        plus `main()`. The coder may replace this wrapper later.
        """
        header = (
            "#include \"gtml.h\"\n\n"
            "int main() {\n"
            "    G2Gtml &g = G2Gtml::instance();\n"
        )
        body = "\n".join("    " + line for line in self.cmds)
        footer = "\n    return 0;\n}\n"
        return header + body + footer


# --------------------------------------------------------------------- #
# Placement / scheduling
# --------------------------------------------------------------------- #


def _resolve_layout(match: MatchedOp, layout: Placement) -> dict[str, Any]:
    """Translate a `MatchedOp`'s workload-side operand bindings to
    target handles via `layout.placements`.

    The autoscheduler (`spmw_autoschedule.autoschedule`) constructs the
    `Placement`; this function is a thin lookup that turns role → memref →
    handle into role → handle so `emit` can be called.

    Per spec 015 §7.4, if the regalloc wrapped a placement in a
    `Spilled(home, tier)`, codegen unwraps it to `home` (the move
    scheduler emits the spill LD/ST round-trip around the work-id
    bucket -- a deferred extension; today's tests do not exercise the
    spill code path).
    """
    # Imported lazily because spmw_regalloc imports Placement from
    # spmw_autoschedule which is imported here at module load -- the
    # cycle is broken by deferring the regalloc import until first use.
    from .spmw_regalloc import Spilled

    bindings: dict[str, Any] = {}
    for opb in match.operands:
        handle = layout.placements.get(opb.memref_name)
        if handle is None:
            raise NotImplementedError(
                f"layout has no placement for memref {opb.memref_name!r}; "
                f"available: {sorted(layout.placements)}"
            )
        if isinstance(handle, Spilled):
            handle = handle.home_handle
        bindings[opb.role] = handle
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


def _samsung_lane_burst(ctx) -> int:
    """Samsung GRF lane count read from the target geometry.

    Each MAC consumes this many fp16 elements per K-iteration, so the
    inner-K loop folds `K` MACs into `K // lanes`. Sourced from
    `target.grf_a.lanes` (the report-16 target spec, `lanes=8`); no
    shape literal.
    """
    return ctx.target.grf_a.lanes


def _fiber_fold_split(folded: int, n_fibers: int) -> list[int]:
    """Round-robin split of `folded` burst-tiles across `n_fibers` banks.

    Fiber `i` gets `ceil` for the first `folded % n_fibers` fibers and
    `floor` after, so the busier fiber stays bounded (matches the cost
    model's ceil `per_fiber`). For folded=128, n=2 -> [64, 64]; for an
    odd folded=127, n=2 -> [64, 63].
    """
    return [(folded + (n_fibers - 1 - i)) // n_fibers for i in range(n_fibers)]


def _emit_inner_loop_jump(
    match: MatchedOp, ctx: CodegenContext, n_emitted: int
) -> None:
    """Append a Samsung-style JUMP that folds the innermost reduction
    loop.

    Loop counter is `(inner_ub // lanes) - 1` (the first MAC counts as
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
    loop_counter = ub // _samsung_lane_burst(ctx) - 1
    if loop_counter <= 0:
        return
    ctx.cmds.append(
        PIMCmd(
            type_="JUMP",
            loopCounter_=loop_counter,
            loopOffset_=n_emitted + 1,
        )
    )


def _is_samsung_preload_mov(c: PIMCmd) -> bool:
    """A GRF<-bank load MOV that opens a work-id (the `LD_A`/`LD_B`
    preload). Used to delimit per-layer GEMV groups in the run path."""
    if c.type_ not in ("MOV", "FILL"):
        return False
    grf_dst = c.dst_ in ("GRF_A", "GRF_B")
    bank_src = any(s in ("EVEN_BANK", "ODD_BANK") for s in (c.src0_, c.src1_))
    return grf_dst and bank_src


def _is_samsung_storeback_mov(c: PIMCmd) -> bool:
    """A bank<-GRF store MOV that closes a work-id (the `ST_A`/`ST_B`
    storeback). `acc -> grf_b` is never host-eligible (lever 2), so this
    MOV is always present and is the residency-robust work-id delimiter."""
    if c.type_ not in ("MOV", "FILL"):
        return False
    bank_dst = c.dst_ in ("EVEN_BANK", "ODD_BANK")
    grf_src = any(s in ("GRF_A", "GRF_B") for s in (c.src0_, c.src1_))
    return bank_dst and grf_src


def _split_samsung_layers(cmds: list[PIMCmd]) -> list[list[PIMCmd]]:
    """Group a flat Samsung GEMV cmd stream into per-layer (work-id)
    chunks, closed by the storeback MOV that ends each work-id.

    A layer body is `[LD MOV?, (MAC, JUMP)+, ST MOV]`; under lever 1 the
    `(MAC, JUMP)` part repeats once per bank fiber (so we must NOT split
    at JUMP), and under lever 2 the opening LD MOV is *absent* for
    host-resident preloads -- so the preload MOV is no longer a reliable
    boundary. The storeback MOV (`acc -> grf_b -> bank`, never
    host-eligible) always closes a work-id, so we split *after* it.
    Trailing setup/terminator (NOP/EXIT) with no MAC attaches to the
    last group; groups with no MAC are dropped.
    """
    groups: list[list[PIMCmd]] = []
    cur: list[PIMCmd] = []
    for c in cmds:
        cur.append(c)
        if _is_samsung_storeback_mov(c) and any(
            g.type_ == "MAC" for g in cur
        ):
            # This storeback closes the current work-id's body.
            groups.append(cur)
            cur = []
    if cur:
        if groups and not any(g.type_ == "MAC" for g in cur):
            groups[-1].extend(cur)
        else:
            groups.append(cur)
    return [g for g in groups if any(c.type_ == "MAC" for c in g)]


def _bucket_by_work_id(trace: MatchTrace):
    """Group matches in trace order by ``(func_name, work_id)``.

    Returns a list of ``(work_id, [matches])`` pairs preserving
    first-seen order. We key on `(func_name, work_id)` rather than
    `work_id` alone because multi-kernel workloads (e.g. an MLP with
    two `@allo.work` layers) can reuse the same numeric `work_id`
    across kernels; without the func_name in the key, layer1 and
    layer2 matches would collapse into the same bucket and the walker
    would apply the wrong placement to one of them.
    """
    buckets: dict = {}
    order: list = []
    for m in trace.matches:
        key = (m.func_name, m.work_id)
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
    # Unwrap Spilled() here: per spec 015 §7.4 the simple-path codegen
    # treats Spilled(home, tier) as if it were `home` (the spill
    # round-trip is a deferred extension).
    from .spmw_regalloc import Spilled

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
        if isinstance(handle, Spilled):
            handle = handle.home_handle
        ld_name, st_name = ctx.resolve_moves(
            role, src_handle=None, dst_handle=handle
        )
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
) -> None:
    """Walk every match in ``trace``, bucketed by work-id, and dispatch
    each match to its target op's ``emit`` callback.

    ``layouts`` is aligned with ``_bucket_for_autoschedule(trace)`` —
    one `Placement` per `@allo.work` kernel. For each work-id bucket,
    we look up the layout by the bucket's `func_name` and compute the
    `role -> memref` map from THIS bucket's matches (the real bug fix:
    role -> memref is per-kernel, not per-trace).

    Move scheduling (preloads before the first match of a work-id and
    storebacks after the last) is delegated to ``_schedule_moves``; the
    per-backend ``resolve_moves`` hook controls which Move names get
    emitted. Backend-specific post-match work (e.g. Samsung's inner-K
    JUMP) flows through ``ctx.after_match``.
    """
    from .spmw_autoschedule import _bucket_for_autoschedule, _trace_memrefs_by_role

    layout_by_func: dict[str, Placement] = {}
    for (func_name, _), layout in zip(_bucket_for_autoschedule(trace), layouts):
        layout_by_func[func_name] = layout

    def _emit_one_bucket(matches: list[MatchedOp]) -> None:
        # All matches in one work-id bucket share a func_name (work_id
        # is parsed from func_name in the matcher). Take the first.
        func_name = matches[0].func_name
        layout = layout_by_func[func_name]

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
            op_obj = target.op(match.target_op_name)
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
            # SPEC-019: UPMEM MAC needs the inner-K bound on the ctx so
            # the fixture emit lambda can materialise an explicit C loop
            # (uPIMulator prices per DPU instruction; without the loop
            # the body is two statements). Gated on isinstance to keep
            # other backends inert.
            if isinstance(ctx, UPMEMCtx) and match.target_op_name == "MAC":
                inner = (
                    match.enclosing_loops[-1] if match.enclosing_loops else None
                )
                ctx.pending_k_bound = (
                    _parse_loop_bound(inner[2]) if inner is not None else None
                )
            elif hasattr(ctx, "pending_k_bound"):
                ctx.pending_k_bound = None
            # APU v1 MAC dispatch: `placement.mode == "sv"` overrides the
            # fixture's `emit_mac_lookup` lambda and emits raw MUL+ADD
            # (SPEC-009 §2). Other backends ignore `mode`.
            if (
                isinstance(ctx, APUv1Ctx)
                and match.target_op_name == "MAC"
                and getattr(layout, "mode", "") == "sv"
            ):
                ctx.emit_mac_mul_add(
                    acc=bindings["acc"], x=bindings["x"], y=bindings["y"],
                )
            elif op_obj.accumulates:
                emit(bindings["x"], bindings["y"], bindings["acc"], ctx)
            else:
                emit(bindings["x"], bindings["y"], bindings["dst"], ctx)
            n_emitted = len(ctx.cmds) - before
            ctx.after_match(match, n_emitted)
        _schedule_moves(target, ctx, layout, role_to_memref, phase="post")

    buckets = _bucket_by_work_id(trace)

    # Lever 3 (SPEC-025 §5): the CRF-issue mode is a decided property of the
    # layout (`extra["crf_issue"]`, set by argmin). Codegen only
    # materialises it -- it never chooses on shape/count/stream length.
    # Default-missing key == "per_workid", so every pre-lever-3 placement
    # and every non-Samsung backend takes the replicated path unchanged.
    crf_issue = "per_workid"
    for layout in layouts:
        issue = getattr(layout, "extra", {}).get("crf_issue")
        if issue is not None:
            crf_issue = issue
            break

    if crf_issue == "shared":
        # Shared CRF: per kernel, emit ONE representative CRF body (walked
        # through the normal emit path -- generated from the Placement, not
        # a golden table, SPEC-025 §5.5) and record one host trigger per
        # work-id bucket. A single-kernel GEMV thus emits one body + N
        # triggers; a multi-kernel workload (MLP) emits one body per layer
        # (each layer's CRF is its own shared program).
        from .spmw_cost_models import _samsung_workid_count

        def _base_kernel(m) -> str:
            # The grid replicates ONE logical `@allo.work` kernel across
            # work-ids by suffixing the func_name with the work_id coords
            # (e.g. `gemv` -> `gemv_0_0`..`gemv_15_7`). Strip that suffix so
            # all grid replicas of one kernel share a base name and collapse
            # to one shared CRF body, while genuinely-distinct kernels (an
            # MLP's `mlp_layer1` vs `mlp_layer2`) stay separate. The number
            # of suffixed coords is len(work_id).
            fn = m.func_name
            wid = m.work_id or ()
            for coord in reversed(wid):
                tail = f"_{coord}"
                if fn.endswith(tail):
                    fn = fn[: -len(tail)]
            return fn

        # Group buckets by base kernel, preserving first-seen order. Grid
        # replicas of one kernel share a body; distinct kernels do not.
        per_kernel: dict[str, list] = {}
        kernel_order: list[str] = []
        for work_id, matches in buckets:
            fn = _base_kernel(matches[0])
            if fn not in per_kernel:
                per_kernel[fn] = []
                kernel_order.append(fn)
            per_kernel[fn].append((work_id, matches))

        expected = _samsung_workid_count(target)
        for fn in kernel_order:
            k_buckets = per_kernel[fn]
            # One shared body for this kernel (the first work-id bucket).
            _emit_one_bucket(k_buckets[0][1])
            for work_id, matches in k_buckets:
                # tile_count = the per-work-id output-tile fire count (the
                # MAC sites the host fires for this work-id), from the
                # bucket's matches, not a shape literal.
                tile_count = sum(
                    1 for m in matches if m.target_op_name == "MAC"
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
                raise ValueError(
                    "samsung shared-CRF: single-kernel trace has "
                    f"{len(k_buckets)} work-id buckets but target geometry "
                    f"yields {expected} (unit-tree fanout product); the "
                    "shared-CRF trigger schedule would be mispriced."
                )
        return

    for _work_id, matches in buckets:
        _emit_one_bucket(matches)


# --------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------- #


@dataclass
class RunResult:
    """Outcome of running a `Compiled` artifact on a simulator/hardware.

    `cycles` is `None` for backends that don't report cycle counts
    (APU v2 / l1_sim is functional-only) or when the simulator was
    unavailable. `stdout` is the captured stdout from the
    sub-process / API call (or a short "simulator unavailable" note).
    `backend` is the target name -- a convenience for callers that
    don't want to dig through `compiled.target.name`.
    """

    cycles: int | None
    stdout: str
    backend: str
    extra: dict = field(default_factory=dict)


# Per-backend simulator location overrides — tests / CI may patch these
# environment variables before importing this module. The defaults
# match the layout on this server (see `experiments/simulators/`).
_DEFAULT_PIMSIM_ROOT = (
    Path(__file__).resolve().parents[2]
    / "simulators"
    / "PIMSimulator"
)
_DEFAULT_AIM_ROOT = (
    Path(__file__).resolve().parents[2]
    / "simulators"
    / "aim_simulator"
)
_DEFAULT_UPIM_ROOT = (
    Path(__file__).resolve().parents[2]
    / "simulators"
    / "uPIMulator"
    / "golang"
    / "uPIMulator"
)


def _pimsim_root() -> Path:
    return Path(os.environ.get("PIMSIMULATOR_ROOT", str(_DEFAULT_PIMSIM_ROOT)))


def _aim_root() -> Path:
    return Path(os.environ.get("AIM_SIMULATOR_ROOT", str(_DEFAULT_AIM_ROOT)))


def _upim_root() -> Path:
    return Path(os.environ.get("UPIMULATOR_ROOT", str(_DEFAULT_UPIM_ROOT)))


def _docker_available() -> bool:
    return shutil.which("docker") is not None


def _docker_image_exists(name: str) -> bool:
    if not _docker_available():
        return False
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", name],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


_DEFAULT_APU_V1_TOOLCHAIN_BASE = (
    "/usr/local/gsi-apu/13.7.1/ubuntu_20_04/"
    "arc_gnu_2021.09-release_elf32_le_linux_no_sdata"
)
_DEFAULT_APU_V1_TEMPLATE_DIR = (
    "/home/nz264/shared/accelerator-hub/gsi-apu/example-gvml"
)
_DEFAULT_APU_V1_PCI_NODE = "/sys/bus/pci/devices/0000:41:00.0"


def _apu_v1_unavailable_reason() -> str | None:
    """Return a short human-readable reason the APU v1 path can't run,
    or None if all preconditions (ARC toolchain dir, example-gvml
    template dir, GSI PCI sysfs node) are present.

    Overridable via env vars TENON_APU_V1_TOOLCHAIN_BASE and
    TENON_APU_V1_TEMPLATE_DIR.
    """
    arc_base = Path(
        os.environ.get(
            "TENON_APU_V1_TOOLCHAIN_BASE", _DEFAULT_APU_V1_TOOLCHAIN_BASE
        )
    )
    if not arc_base.exists():
        return f"simulator unavailable: ARC toolchain missing at {arc_base}"
    template = Path(
        os.environ.get(
            "TENON_APU_V1_TEMPLATE_DIR", _DEFAULT_APU_V1_TEMPLATE_DIR
        )
    )
    if not template.exists():
        return (
            f"simulator unavailable: example-gvml template missing at {template}"
        )
    pci = Path(_DEFAULT_APU_V1_PCI_NODE)
    if not pci.exists():
        return f"simulator unavailable: GSI device not present at {pci}"
    # GVML SDK headers: a partial install (eltwise present, logical
    # absent) makes the build fail mid-way; check both canaries up-front
    # so the run path returns a clean RunResult(cycles=None, ...) skip
    # instead of raising RuntimeError from `make`.
    from .spmw_apu_v1_build import _gvml_sdk_available, _gvml_include_root
    if not _gvml_sdk_available():
        return (
            f"simulator unavailable: GVML SDK headers missing under "
            f"{_gvml_include_root()}"
        )
    return None


# --------------------------------------------------------------------- #
# Per-backend run hooks
# --------------------------------------------------------------------- #


def _write_samsung_cmds(path: Path, cmds: list[PIMCmd]) -> None:
    """Serialise a list of PIMCmd to the line-delimited format pim_driver
    accepts via `--cmds`. See `_run_samsung` docstring for the grammar.
    """
    with open(path, "w") as f:
        for c in cmds:
            parts = [c.type_]
            # Only emit fields that differ from PIMCmd's default ctor so
            # the trace stays human-greppable.
            if c.dst_ != "A_OUT":
                parts.append(f"dst={c.dst_}")
            if c.src0_ != "A_OUT":
                parts.append(f"src0={c.src0_}")
            if c.src1_ != "A_OUT":
                parts.append(f"src1={c.src1_}")
            if c.src2_ != "A_OUT":
                parts.append(f"src2={c.src2_}")
            if c.loopCounter_:
                parts.append(f"loop_counter={c.loopCounter_}")
            if c.loopOffset_:
                parts.append(f"loop_offset={c.loopOffset_}")
            if c.isAuto_:
                parts.append(f"is_auto={c.isAuto_}")
            if c.dstIdx_:
                parts.append(f"dst_idx={c.dstIdx_}")
            if c.src0Idx_:
                parts.append(f"src0_idx={c.src0Idx_}")
            if c.src1Idx_:
                parts.append(f"src1_idx={c.src1Idx_}")
            if c.isRelu_:
                parts.append(f"is_relu={c.isRelu_}")
            f.write(" ".join(parts) + "\n")


def _run_samsung_one(
    driver: Path,
    root: Path,
    cmd_subset: list[PIMCmd],
    layer_input: dict,
    np_mod,
    layer_idx: int = 0,
) -> tuple[int, str]:
    """SPEC-020: invoke pim_driver for a single GEMV layer.

    Used by the multi-layer branch of `_run_samsung` (one call per MLP
    layer). Returns `(cycles, combined_stdout)`. The single-layer
    back-compat path in `_run_samsung` does not go through here -- it
    keeps the inlined logic so callers that pass a single-MAC stream
    see identical behaviour to pre-SPEC-020.
    """
    W = layer_input.get("W")
    if W is None:
        W = layer_input.get("weight")
    x = layer_input.get("x")
    if x is None:
        x = layer_input.get("in")
    if W is None or x is None:
        raise ValueError(
            f"Samsung layer {layer_idx}: each entry in layers=[...] "
            "must provide W and x (or weight/in) arrays"
        )
    W = np_mod.asarray(W, dtype=np_mod.float16)
    x = np_mod.asarray(x, dtype=np_mod.float16)
    # SPEC-020 §2.5: see comment in `_run_samsung` GEMV branch.
    if x.ndim == 1:
        x = x.reshape(1, -1)

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        out_path = td_path / "out.bin"
        w_path = td_path / "W.npy"
        x_path = td_path / "x.npy"
        cmds_path = td_path / "cmds.txt"
        np_mod.save(w_path, W)
        np_mod.save(x_path, x)
        _write_samsung_cmds(cmds_path, cmd_subset)

        argv = [
            str(driver),
            "--op", "GEMV",
            "--out", str(out_path),
            "--weight", str(w_path),
            "--in", str(x_path),
            "--output-dim", str(W.shape[0]),
            "--input-dim", str(W.shape[1]),
            "--cmds", str(cmds_path),
            # SPEC-021 task 025: faithful run path -- issued PIM transactions
            # track the emitted cmd stream, so a better/worse stream costs
            # fewer/more cycles under the same accounting for native and Tenon.
            "--faithful",
        ]

        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                cwd=str(root),
                timeout=600,
                check=False,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            raise RuntimeError(
                f"Samsung pim_driver invocation failed (layer {layer_idx}): {exc}"
            ) from exc

        stdout = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")
        combined = stdout + ("\n" + stderr if stderr else "")
        m = re.search(r"PIM_CYCLES total=(\d+)", combined)
        if not m:
            raise RuntimeError(
                f"Samsung pim_driver returned but stdout missing "
                f"'PIM_CYCLES total=...' line (layer {layer_idx}); "
                f"tail: {combined[-400:]}"
            )
        return int(m.group(1)), combined


def _run_samsung(compiled: "Compiled", **inputs) -> RunResult:
    """Run a compiled Samsung HBM-PIM artifact via `pim_driver`.

    Wire protocol (SPEC-005): the emitted PIMCmd stream is serialised
    to a line-delimited `cmds.txt` file in the temp dir and passed via
    ``pim_driver --cmds <path>``. The C++ side uses that as the CRF
    microcode (uploaded by `programCrf`) instead of regenerating it via
    `PIMCmdGen::getPIMCmds`. The legacy ``--op <kernel>`` flag is still
    passed so the driver knows which data-path scaffolding (eltwise vs
    GEMV) and which numpy inputs to wire up; the kernel choice no longer
    determines the CRF program (placement does).

    Line-delimited cmd format:
        MAC dst=GRF_B src0=GRF_A src1=EVEN_BANK is_auto=1
        JUMP loop_counter=7 loop_offset=2
        NOP loop_counter=7
        EXIT
    """
    root = _pimsim_root()
    driver = root / "pim_driver"
    if not driver.exists():
        return RunResult(
            cycles=None,
            stdout=f"simulator unavailable: pim_driver not found at {driver}",
            backend="samsung_hbm_pim",
        )

    # The `--op` flag still picks the data-path scaffolding (eltwise vs
    # GEMV) and which numpy inputs to wire up, but no longer determines
    # the CRF microcode -- that comes from `compiled.cmds` via `--cmds`.
    # The choice of which scaffolding to use is inferred from MAC-vs-eltwise
    # opcodes in the emitted stream.
    op_types = {c.type_ for c in compiled.cmds if isinstance(c, PIMCmd)}
    if "MAC" in op_types:
        kernel = "GEMV"
    elif "MUL" in op_types:
        kernel = "MUL"
    elif "ADD" in op_types:
        kernel = "ADD"
    else:
        kernel = "RELU"

    # numpy is a hard Tenon dependency; "no numpy" is a setup bug, not
    # an env skip -- let the ImportError propagate.
    import numpy as np

    # SPEC-020: ISA-valid filter applied up-front so layer-splitting sees
    # the same cmd stream the driver eventually runs.
    def _crf_valid(c: PIMCmd) -> bool:
        # SamsungCtx emits per-role storeback "MOV ODD_BANK <- GRF_B" /
        # "MOV EVEN_BANK <- GRF_A" entries (the `ST_A`/`ST_B` moves).
        # The C++ PIMCmd::validationCheck rejects those as "Invalid in
        # ISA 1.0" because bank stores are issued by the DRAM controller
        # via addTransactionAll, not by CRF MOV. Drop them here so the
        # cmd stream that reaches programCrf is ISA-valid. This is a
        # run-side workaround; the proper fix is in SamsungCtx (followup
        # SPEC-005 §8 / needs-arch-MMM).
        if c.type_ in ("MOV", "FILL"):
            bank_dst = c.dst_ in ("EVEN_BANK", "ODD_BANK")
            grf_src = any(
                s in ("GRF_A", "GRF_B")
                for s in (c.src0_, c.src1_, c.src2_)
            )
            if bank_dst and grf_src:
                return False
        return True

    pim_cmds_raw = [c for c in compiled.cmds if isinstance(c, PIMCmd)]

    # SPEC-020: detect multi-layer GEMV streams. `pim_driver` accepts one
    # GEMV per invocation, so a cmd stream covering >1 layer (e.g. an MLP)
    # must be split into per-layer groups dispatched as separate driver
    # calls, cycles summed. A layer boundary is the storeback MOV that
    # closes each work-id; this stays correct under lever 1 (one layer's
    # body holds several MAC+JUMP pairs -- splitting at JUMP would cut a
    # dual-fiber layer in two) and under lever 2 (host residency omits the
    # opening preload MOV, so the storeback is the residency-robust
    # delimiter). We split on the RAW stream so the storeback MOV is still
    # present, then drop the ISA-invalid storebacks per group.
    raw_groups = _split_samsung_layers(pim_cmds_raw)
    groups = [[c for c in g if _crf_valid(c)] for g in raw_groups]
    pim_cmds_all = [c for c in pim_cmds_raw if _crf_valid(c)]
    multi_layer = kernel == "GEMV" and len(groups) > 1

    if multi_layer:
        layers = inputs.get("layers")
        if layers is None:
            # Back-compat escape: `compiled.run()` with no kwargs (e.g.
            # `test_run_returns_runresult_for_all_backends`) must keep
            # returning cycles=None rather than raise.
            return RunResult(
                cycles=None,
                stdout=(
                    "Samsung: multi-layer cmd stream needs layers=[...] kwarg "
                    f"({len(groups)} GEMV layers detected)"
                ),
                backend="samsung_hbm_pim",
            )
        if len(layers) != len(groups):
            raise ValueError(
                f"Samsung: {len(groups)} MAC groups in cmd stream but "
                f"{len(layers)} layers= entries"
            )
        total_cycles = 0
        combined_parts: list[str] = []
        for i, (grp, layer_in) in enumerate(zip(groups, layers)):
            cyc, out = _run_samsung_one(
                driver, root, grp, layer_in, np, layer_idx=i
            )
            total_cycles += cyc
            combined_parts.append(out)
        return RunResult(
            cycles=total_cycles,
            stdout="\n--- next layer ---\n".join(combined_parts),
            backend="samsung_hbm_pim",
            extra={"kernel": kernel, "n_layers": len(groups)},
        )

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        out_path = td_path / "out.bin"
        argv = [str(driver), "--op", kernel, "--out", str(out_path)]

        if kernel == "GEMV":
            W = inputs.get("W")
            if W is None:
                W = inputs.get("weight")
            x = inputs.get("x")
            if x is None:
                x = inputs.get("in")
            if W is None or x is None:
                # SPEC-001 §6 escape: tests/spmw/test_run.py's
                # `test_run_returns_runresult_for_all_backends` calls
                # compiled.run() with no kwargs and expects cycles=None;
                # keep this branch returning cycles=None for backward
                # compat. Flagged for PR review.
                return RunResult(
                    cycles=None,
                    stdout="GEMV needs W and x kwargs",
                    backend="samsung_hbm_pim",
                )
            W = np.asarray(W, dtype=np.float16)
            x = np.asarray(x, dtype=np.float16)
            # SPEC-020 §2.5: a 1D x of shape (K,) makes Burst.h::loadFp16
            # emit bShape=[ceil(K/16)], which executeGemv reads as
            # num_batch=ceil(K/16) -- i.e. the GEMV runs ceil(K/16) times.
            # The canonical Samsung fixture (data/gemv/gen_gemv.py:34)
            # stores x as (1, K) so num_batch=1. Match that contract.
            if x.ndim == 1:
                x = x.reshape(1, -1)
            w_path = td_path / "W.npy"
            x_path = td_path / "x.npy"
            np.save(w_path, W)
            np.save(x_path, x)
            argv += [
                "--weight", str(w_path),
                "--in", str(x_path),
                "--output-dim", str(W.shape[0]),
                "--input-dim", str(W.shape[1]),
            ]
        elif kernel in ("ADD", "MUL"):
            a = inputs.get("a")
            if a is None:
                a = inputs.get("in0")
            b = inputs.get("b")
            if b is None:
                b = inputs.get("in1")
            if a is None or b is None:
                raise ValueError(
                    f"Samsung {kernel} needs a/b (or in0/in1) kwargs"
                )
            a = np.asarray(a, dtype=np.float16).reshape(-1)
            b = np.asarray(b, dtype=np.float16).reshape(-1)
            a_path = td_path / "a.npy"
            b_path = td_path / "b.npy"
            np.save(a_path, a)
            np.save(b_path, b)
            argv += [
                "--in0", str(a_path),
                "--in1", str(b_path),
                "--n", str(a.size),
            ]
        else:  # RELU
            a = inputs.get("a")
            if a is None:
                a = inputs.get("in0")
            if a is None:
                raise ValueError(
                    "Samsung RELU needs a (or in0) kwarg"
                )
            a = np.asarray(a, dtype=np.float16).reshape(-1)
            a_path = td_path / "a.npy"
            np.save(a_path, a)
            argv += ["--in0", str(a_path), "--n", str(a.size)]

        # SPEC-005: serialise compiled.cmds to a line-delimited file and
        # pass --cmds so the driver uses our CRF microcode instead of
        # PIMCmdGen's canonical one. Layout/placement changes show up in
        # cycles only because this file flows into programCrf().
        # The ISA-valid filter was applied up-front into `pim_cmds_all`.
        if pim_cmds_all:
            cmds_path = td_path / "cmds.txt"
            _write_samsung_cmds(cmds_path, pim_cmds_all)
            argv += ["--cmds", str(cmds_path)]
            # SPEC-021 task 025: GEMV gets the faithful run path so issued
            # PIM transactions track the emitted stream (a redundant stream
            # costs more, a folded one costs less). Eltwise has no faithful
            # variant yet, so it stays on executeEltwiseWithCmds.
            if kernel == "GEMV":
                argv += ["--faithful"]

        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                cwd=str(root),  # pim_driver resolves ini/ paths relative to cwd
                timeout=600,
                check=False,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            raise RuntimeError(
                f"Samsung pim_driver invocation failed: {exc}"
            ) from exc

        stdout = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")
        combined = stdout + ("\n" + stderr if stderr else "")
        match = re.search(r"PIM_CYCLES total=(\d+)", combined)
        if not match:
            raise RuntimeError(
                "Samsung pim_driver returned but stdout missing "
                "'PIM_CYCLES total=...' line; tail: " + combined[-400:]
            )
        cycles = int(match.group(1))
        return RunResult(
            cycles=cycles,
            stdout=combined,
            backend="samsung_hbm_pim",
            extra={"kernel": kernel, "returncode": proc.returncode},
        )


def _layout_weight_resident(layout) -> bool:
    """Read the chosen placement's `weight_resident` flag (SPEC-026 §4.1).

    `Compiled.layout` is a single Placement (one kernel) or a list. The
    batched run path is single-GEMV; default-absent key -> False (the
    pre-026 non-resident shape). Codegen materialises this decision; it
    does not re-decide (I5).
    """
    if isinstance(layout, list):
        layout = layout[0] if layout else None
    if layout is None:
        return False
    return bool(getattr(layout, "extra", {}).get("weight_resident", False))


def _samsung_batched_invoke(
    driver: Path,
    root: Path,
    cmd_subset: list,
    W,
    X,
    batch: int,
    native_rebaseline: bool,
    np_mod,
) -> tuple[int, dict, str]:
    """One batched pim_driver --batch invocation (SPEC-026 §4.2).

    Returns `(total_cycles, phase_dict, combined_stdout)`. `phase_dict`
    has preload/exec/readback so the caller can assert the per-phase
    split against the cost model. `native_rebaseline=True` selects the
    re-preload-per-vector comparator loop.
    """
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        out_path = td_path / "out.bin"
        w_path = td_path / "W.npy"
        x_path = td_path / "X.npy"
        cmds_path = td_path / "cmds.txt"
        np_mod.save(w_path, W)
        np_mod.save(x_path, X)
        _write_samsung_cmds(cmds_path, cmd_subset)

        argv = [
            str(driver),
            "--op", "GEMV",
            "--out", str(out_path),
            "--weight", str(w_path),
            "--in", str(x_path),
            "--output-dim", str(W.shape[0]),
            "--input-dim", str(W.shape[1]),
            "--cmds", str(cmds_path),
            "--faithful",
            "--batch", str(batch),
        ]
        if native_rebaseline:
            argv.append("--native-rebaseline")

        try:
            proc = subprocess.run(
                argv, capture_output=True, cwd=str(root),
                timeout=600, check=False,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            raise RuntimeError(
                f"Samsung batched pim_driver invocation failed: {exc}"
            ) from exc

        stdout = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")
        combined = stdout + ("\n" + stderr if stderr else "")
        m = re.search(
            r"PIM_CYCLES total=(\d+)\s+preload=(\d+)\s+exec=(\d+)\s+readback=(\d+)",
            combined,
        )
        if not m:
            raise RuntimeError(
                "Samsung batched pim_driver returned but stdout missing "
                "'PIM_CYCLES total=...' line; tail: " + combined[-400:]
            )
        phases = {
            "preload": int(m.group(2)),
            "exec": int(m.group(3)),
            "readback": int(m.group(4)),
        }
        return int(m.group(1)), phases, combined


def _run_samsung_batched(
    compiled: "Compiled",
    W,
    X,
    compare_native: bool = True,
) -> RunResult:
    """Batched-GEMV run path (SPEC-026 §4.2/§4.3).

    Emits the Tenon stream (preload once + B*(exec+readback), selected by
    the chosen placement's `weight_resident` flag) AND, when
    `compare_native` is set, the native rebaseline (B*(preload+exec+
    readback)) under the SAME faithful instrument, cmd stream, W, and
    X(B,K). Only the preload-loop placement differs (I2). B = X.shape[0]
    is read off the operand shape, never a literal (I3).

    `RunResult.cycles` is the Tenon total; `extra` carries the native
    total + both per-phase splits so the gate-A comparison and the
    cost-model assertion can be made by the caller.
    """
    import numpy as np

    root = _pimsim_root()
    driver = root / "pim_driver"
    if not driver.exists():
        return RunResult(
            cycles=None,
            stdout=f"simulator unavailable: pim_driver not found at {driver}",
            backend="samsung_hbm_pim",
        )

    W = np.asarray(W, dtype=np.float16)
    X = np.asarray(X, dtype=np.float16)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    B = int(X.shape[0])  # I3: B is operand geometry (X[B,K] leading dim).

    pim_cmds = [c for c in compiled.cmds if isinstance(c, PIMCmd)]
    # Same ISA-valid filter the single-vector run path applies.
    def _crf_valid(c: PIMCmd) -> bool:
        if c.type_ in ("MOV", "FILL"):
            bank_dst = c.dst_ in ("EVEN_BANK", "ODD_BANK")
            grf_src = any(
                s in ("GRF_A", "GRF_B") for s in (c.src0_, c.src1_, c.src2_)
            )
            if bank_dst and grf_src:
                return False
        return True

    pim_cmds = [c for c in pim_cmds if _crf_valid(c)]

    resident = _layout_weight_resident(compiled.layout)
    # Tenon: resident mode preloads once when B>1; the driver reads the
    # resident vs native loop from --native-rebaseline (absent => resident).
    # If the chosen placement is NOT weight_resident, Tenon's own run is the
    # native (re-preload-per-vector) sequencing -- codegen materialises the
    # decision argmin made, it does not override it.
    tenon_total, tenon_phases, tenon_out = _samsung_batched_invoke(
        driver, root, pim_cmds, W, X, B,
        native_rebaseline=not resident, np_mod=np,
    )

    extra = {
        "kernel": "GEMV",
        "batch": B,
        "weight_resident": resident,
        "tenon_total": tenon_total,
        "tenon_phases": tenon_phases,
    }
    combined = tenon_out
    if compare_native:
        native_total, native_phases, native_out = _samsung_batched_invoke(
            driver, root, pim_cmds, W, X, B,
            native_rebaseline=True, np_mod=np,
        )
        extra["native_total"] = native_total
        extra["native_phases"] = native_phases
        combined = (
            tenon_out
            + "\n--- native rebaseline ---\n"
            + native_out
        )

    return RunResult(
        cycles=tenon_total,
        stdout=combined,
        backend="samsung_hbm_pim",
        extra=extra,
    )


def _run_aim(compiled: "Compiled", **inputs) -> RunResult:
    """Run a compiled AiM artifact through ramulator2 in Docker.

    `compiled.cmds` is a list of text trace lines; we serialise them
    plus a trailing EOC marker, then invoke the ramulator2 binary
    inside the `aim-simulator-build` Docker image. The simulator's
    YAML stdout carries per-channel `memory_system_cycles`; we parse
    out the max across channels as the headline cycle count.
    """
    root = _aim_root()
    yaml_cfg = root / "test" / "example.yaml"
    if not _docker_image_exists("aim-simulator-build") or not yaml_cfg.exists():
        return RunResult(
            cycles=None,
            stdout="simulator unavailable: aim-simulator-build docker image / config missing",
            backend="aim",
        )

    with tempfile.NamedTemporaryFile(
        mode="w",
        delete=False,
        suffix=".trace",
        dir=str(root / "test"),
    ) as tf:
        for line in compiled.cmds:
            tf.write(str(line) + "\n")
        if not any("EOC" in str(l) for l in compiled.cmds):
            tf.write("AiM EOC\n")
        trace_path = Path(tf.name)
    trace_rel = trace_path.relative_to(root)
    cfg_rel = yaml_cfg.relative_to(root)

    try:
        proc = subprocess.run(
            [
                "docker", "run", "--rm",
                "-v", f"{root}:/work",
                "aim-simulator-build",
                "bash", "-c",
                f"cd /work && ./build/ramulator2 -f {cfg_rel} -t {trace_rel}",
            ],
            capture_output=True,
            timeout=600,
            check=False,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        trace_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"AiM ramulator2 invocation failed: {exc}"
        ) from exc
    finally:
        # Keep trace on disk only for the duration of the run; cleanup.
        trace_path.unlink(missing_ok=True)

    stdout = proc.stdout.decode("utf-8", errors="replace")
    stderr = proc.stderr.decode("utf-8", errors="replace")
    combined = stdout + ("\n" + stderr if stderr else "")
    # ramulator2 emits per-channel `memory_system_cycles: <N>` in YAML;
    # take the max across channels as the headline number.
    cycle_vals = [
        int(m) for m in re.findall(r"memory_system_cycles:\s*(\d+)", combined)
    ]
    if not cycle_vals:
        raise RuntimeError(
            "AiM ramulator2 returned but stdout missing "
            "'memory_system_cycles: ...' line; tail: " + combined[-400:]
        )
    cycles = max(cycle_vals)
    return RunResult(
        cycles=cycles,
        stdout=combined,
        backend="aim",
        extra={"returncode": proc.returncode},
    )


def _run_upmem(compiled: "Compiled", **inputs) -> RunResult:
    """Run a compiled UPMEM artifact through the Go uPIMulator.

    The emitted DPU source from `UPMEMCtx.get_kernel_src()` is written
    into the persistent TENON benchmark slot at
    `benchmark/TENON/dpu/task.c` (registered in uPIMulator's CMake +
    assembler maps; see SPEC-003 §3). uPIMulator is then invoked with
    `--benchmark TENON`, which triggers a CMake re-build of just that
    `task.c` before the simulation. Cycle counts therefore come from
    the emitted kernel, not a PrIM proxy.
    """
    root = _upim_root()
    binary = root / "build" / "uPIMulator"
    if not binary.exists():
        return RunResult(
            cycles=None,
            stdout=f"simulator unavailable: {binary} not found",
            backend="upmem",
        )

    # Render the full DPU envelope around `cmds` via UPMEMCtx; fall back
    # to a bare join if no ctx was preserved (legacy call shape).
    ctx = getattr(compiled, "_ctx", None)
    if ctx is not None and hasattr(ctx, "get_kernel_src"):
        kernel_src = ctx.get_kernel_src()
    else:
        kernel_src = "\n".join(
            c for c in compiled.cmds if isinstance(c, str)
        )

    if not kernel_src.strip():
        raise RuntimeError(
            "UPMEM: _run_upmem received empty kernel source; "
            "compile_for_target produced no commands"
        )

    slot_dir = root / "benchmark" / "TENON" / "dpu"
    if not slot_dir.exists():
        # uPIMulator binary is present but the TENON slot has not been
        # provisioned in this checkout. Skip cleanly rather than raise
        # (matches the `_sim_unavailable` branch in test_e2e_mlp_upmem).
        return RunResult(
            cycles=None,
            stdout=(
                f"simulator unavailable: TENON benchmark slot not "
                f"provisioned at {slot_dir}; rebuild uPIMulator with "
                f"the TENON benchmark registered."
            ),
            backend="upmem",
        )
    task_c = slot_dir / "task.c"
    task_c.write_text(kernel_src, encoding="utf-8")

    with tempfile.TemporaryDirectory() as td:
        bin_dir = Path(td) / "bin"
        bin_dir.mkdir()
        try:
            proc = subprocess.run(
                [
                    str(binary),
                    "--root_dirpath", str(root),
                    "--bin_dirpath", str(bin_dir),
                    "--benchmark", "TENON",
                    "--num_channels", "1",
                    "--num_dpus_per_rank", "1",
                    "--num_tasklets", "1",
                    "--data_prep_params", "1024",
                ],
                capture_output=True,
                timeout=600,
                check=False,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            raise RuntimeError(
                f"UPMEM uPIMulator invocation failed: {exc}"
            ) from exc

        stdout = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")
        combined = stdout + ("\n" + stderr if stderr else "")
        # uPIMulator prints "cycle: <N>" in log.txt; also tail it if present.
        log_path = bin_dir / "log.txt"
        if log_path.exists():
            combined += "\n" + log_path.read_text(errors="replace")
        cycle_match = re.search(r"cycle[s]?\s*[:=]\s*(\d+)", combined, re.IGNORECASE)
        if cycle_match is None:
            raise RuntimeError(
                "UPMEM uPIMulator returned but stdout/log missing "
                "'cycle: ...' line; tail: " + combined[-400:]
            )
        cycles = int(cycle_match.group(1))
        return RunResult(
            cycles=cycles,
            stdout=combined,
            backend="upmem",
            extra={
                "kernel_src": kernel_src,
                "benchmark": "TENON",
                "returncode": proc.returncode,
            },
        )


def _apu_v1_kernel_src(compiled: "Compiled") -> str:
    return "\n".join(str(c) for c in compiled.cmds if isinstance(c, str))


def _parse_apu_v1_prof_print(text: str) -> int | None:
    """Pull the `crun` integer from the `total` PROF_PRINT line. APU v1
    hardware emits fields with a colon separator, e.g.
    `ARCT[0]: ***  total - hits:1 seu:374 crun:170227 iall:37027 ...`.
    Accept `=` as well for forward compatibility. Returns None if no match.
    """
    # Prefer the explicit `total` line; fall back to the first `crun`
    # we find anywhere if PROF_PRINT names diverge in future kernels.
    m = re.search(r"\btotal\b[^\n]*?\bcrun\s*[:=]\s*(\d+)", text)
    if m:
        return int(m.group(1))
    m = re.search(r"\bcrun\s*[:=]\s*(\d+)", text)
    if m:
        return int(m.group(1))
    return None


def _apu_v1_output_specs(compiled: "Compiled", inputs: dict) -> dict:
    """Derive `output_specs: {role: (shape, dtype)}` from the trace.

    For each `MatchedOp` whose loop-carried `acc` operand defines an
    output memref (`result_memref_name`), we take the shape and dtype
    from a same-named input if present, else fall back to the largest
    input's shape and uint16 dtype. APU v1 today is uint16-only
    (`_32K` constant) so the dtype fallback is safe.
    """
    import numpy as np

    out: dict = {}
    trace = getattr(compiled, "trace", None)
    if trace is None:
        return out

    # Determine a fallback shape from the largest input.
    fallback_shape: tuple = ()
    fallback_size = -1
    for arr in inputs.values():
        sz = getattr(arr, "size", 0)
        if sz > fallback_size:
            fallback_size = sz
            fallback_shape = tuple(getattr(arr, "shape", ()))

    for m in trace.matches:
        name = m.result_memref_name
        if not name or name in out:
            continue
        # Prefer shape/dtype from an input that happens to share the name.
        if name in inputs and hasattr(inputs[name], "shape"):
            out[name] = (tuple(inputs[name].shape), inputs[name].dtype)
        else:
            # APU v1 output: derive from the first input as a 1D vector
            # of the same element count (the GEMV `acc` is M-shaped, but
            # the declarative trace doesn't pin that; the user can
            # override via `output_specs` once exposed).
            out[name] = (fallback_shape or (1,), np.dtype("uint16"))
    return out


def _apu_v1_prepare_io(compiled: "Compiled", inputs: dict):
    """Validate + normalize `inputs` for the APU v1 build harness.

    The user may pass workload-side memref names (e.g. ``local_W``) or
    the shorter role names (``x``, ``y``). We pass the dict through
    untouched today, since the build harness keys struct fields by the
    actual `inputs` keys.
    """
    import numpy as np

    normalized: dict = {}
    for name, arr in inputs.items():
        if not isinstance(arr, np.ndarray):
            raise ValueError(
                f"APU v1 run: input {name!r} must be a numpy.ndarray; "
                f"got {type(arr).__name__}"
            )
        # APU v1 is uint16-native today; we view int16/float16 buffers
        # as their raw bytes (host.c writes them straight into L4) so
        # we don't impose a dtype cast here.
        normalized[name] = arr
    output_specs = _apu_v1_output_specs(compiled, normalized)
    return normalized, output_specs


def _run_apu_v1(compiled: "Compiled", **inputs) -> RunResult:
    """Run a compiled APU v1 artifact against real GSI hardware.

    Materializes a build directory via `gen_apu_v1_low_mode_project`,
    runs `make` with the ARC GNU toolchain, executes the resulting
    binary against the Gemini PCI device, and parses cycle counts from
    its PROF_PRINT stdout. Returns `RunResult(cycles=None, ...)` with
    a "simulator unavailable" stdout when an environment gate
    (toolchain, template dir, PCI sysfs node, GVML SDK headers) is
    missing. Build failures past the gate raise `RuntimeError` so tests
    cannot silently PASS on a cycles=None RunResult; a missing
    PROF_PRINT 'total crun=' line, however, yields `cycles=None` rather
    than raising -- the device program ran (or tried to) but produced
    no profile output, which is observable through the returned
    RunResult.stdout. See spec 017 and SPEC-001 §3.5.
    """
    kernel_src = _apu_v1_kernel_src(compiled)

    skip_reason = _apu_v1_unavailable_reason()
    if skip_reason:
        return RunResult(
            cycles=None,
            stdout=skip_reason,
            backend="apu_v1",
            extra={"kernel_src": kernel_src},
        )

    from .spmw_apu_v1_build import (
        _assert_gvml_sdk_present,
        gen_apu_v1_low_mode_project,
    )

    # Build-harness probe: a missing SDK here means the toolchain gate
    # passed but headers are absent; fail fast with a readable error
    # rather than letting `make` emit a 200-line stderr blob below.
    _assert_gvml_sdk_present()

    tmpdir = tempfile.mkdtemp(prefix="tenon-apu-v1-")
    try:
        # prepare_io is build-harness Python; ValueError here is a Tenon
        # bug or user-input contract violation, not an env skip.
        inputs_np, output_specs = _apu_v1_prepare_io(compiled, inputs)

        # Write each input to <tmpdir>/in_<role>.bin so host.c can fread it.
        input_bin_paths: dict[str, str] = {}
        for role, arr in inputs_np.items():
            p = Path(tmpdir) / f"in_{role}.bin"
            arr.tofile(str(p))
            input_bin_paths[role] = str(p)

        output_bin_paths: dict[str, str] = {
            role: str(Path(tmpdir) / f"out_{role}.bin")
            for role in output_specs
        }

        # Project emission. The toolchain/template gate above already
        # ensured the template dir exists; if gen_apu_v1_low_mode_project
        # still raises FileNotFoundError, that's a Tenon bug, not an
        # env skip -- let it propagate.
        project_dir = gen_apu_v1_low_mode_project(
            dst_dir=Path(tmpdir) / "project",
            compiled=compiled,
            inputs=inputs_np,
            output_specs=output_specs,
            lab_name="tenon-kernel",
        )

        # Build. Past the toolchain/SDK gate, every failure below is a
        # build- or runtime-bug, not an environment skip -- raise so
        # the test does not silently PASS on a cycles=None RunResult.
        try:
            mk = subprocess.run(
                ["make"],
                cwd=str(project_dir),
                capture_output=True,
                timeout=600,
                check=False,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            raise RuntimeError(
                f"APU v1 make invocation failed: {exc}\n"
                f"--- project_dir: {project_dir}"
            ) from exc
        if mk.returncode != 0:
            raise RuntimeError(
                "APU v1 make failed:\n"
                + mk.stderr.decode("utf-8", errors="replace")
                + "\n--- stdout ---\n"
                + mk.stdout.decode("utf-8", errors="replace")
                + f"\n--- project_dir: {project_dir}"
            )

        bin_path = project_dir / "build" / "debug" / "tenon-kernel"
        if not bin_path.exists():
            raise RuntimeError(
                f"APU v1 build succeeded but binary not found at {bin_path} "
                f"(project_dir: {project_dir})"
            )

        # Build argv: inputs first then outputs, in the same role order
        # the build harness wrote into struct.h (sorted alpha).
        sorted_in = sorted(input_bin_paths.keys())
        sorted_out = sorted(output_bin_paths.keys())
        argv = [str(bin_path)]
        argv += [input_bin_paths[r] for r in sorted_in]
        argv += [output_bin_paths[r] for r in sorted_out]

        try:
            proc = subprocess.run(
                argv,
                cwd=str(project_dir),
                capture_output=True,
                timeout=300,
                check=False,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            raise RuntimeError(
                f"APU v1 binary invocation failed: {exc}\n"
                f"--- project_dir: {project_dir}"
            ) from exc

        stdout_text = proc.stdout.decode("utf-8", errors="replace")
        stderr_text = proc.stderr.decode("utf-8", errors="replace")
        binary_output = stdout_text + ("\n" + stderr_text if stderr_text else "")

        # PROF_PRINT lines go to the device system log (the ledag
        # channel), not to the binary's stdout. Drain that channel via
        # `ledag-ssh flo` and concatenate the printable bytes so the
        # `crun` parse below has something to match against. The ARC
        # binary may need a moment to flush its PROF_END(total) entry
        # to the device log after returncode is delivered to us, hence
        # the brief sleep. Skill: see "Pattern B: scripted capture".
        ledag_text = ""
        if shutil.which("ledag-ssh") is not None:
            time.sleep(0.5)
            try:
                ledag_proc = subprocess.run(
                    ["ledag-ssh", "-o", "localhost"],
                    input=b"flo\nquit\n",
                    capture_output=True,
                    timeout=30,
                    check=False,
                )
                ledag_raw = ledag_proc.stdout or b""
                # `| strings` equivalent: keep printable ASCII plus tab
                # / newline / CR; the ledag wire format is otherwise
                # binary-framed and decodes to mojibake.
                ledag_text = "".join(
                    chr(b)
                    for b in ledag_raw
                    if 32 <= b < 127 or b in (9, 10, 13)
                )
            except (subprocess.SubprocessError, OSError):
                # ledag-ssh available but failed (timeout, device busy,
                # auth) -- treat the same as missing-on-PATH: degrade
                # to cycles=None rather than crash the run path.
                ledag_text = ""

        combined = binary_output + ("\n" + ledag_text if ledag_text else "")
        cycles = _parse_apu_v1_prof_print(combined)
        # A missing PROF_PRINT 'total crun=' line is *not* a build-harness
        # bug: it means the device program ran (or attempted to) but
        # produced no PROF_PRINT output -- typically because the GSI PCI
        # device is absent, gated off, or returned an error before our
        # PROF_END(total), or because `ledag-ssh` is not available on
        # this host. Surface this as cycles=None so callers (and the
        # cross-backend `test_run_returns_runresult_for_all_backends`
        # contract) still receive a RunResult; build failures above
        # already raise RuntimeError before we reach here.

        # Read outputs back.
        import numpy as np
        outputs: dict = {}
        for role, (shape, dtype) in output_specs.items():
            p = Path(output_bin_paths[role])
            if p.exists():
                outputs[role] = np.fromfile(str(p), dtype=dtype).reshape(shape)

        return RunResult(
            cycles=cycles,
            stdout=combined,
            backend="apu_v1",
            extra={
                "outputs": outputs,
                "kernel_src": kernel_src,
                "project_dir": str(project_dir),
                "returncode": proc.returncode,
            },
        )
    finally:
        if os.environ.get("TENON_APU_V1_KEEP_TMP") != "1":
            shutil.rmtree(tmpdir, ignore_errors=True)


def _run_apu_v2(compiled: "Compiled", **inputs) -> RunResult:
    """Run a compiled APU v2 artifact through the `gsi-g2-l1sim` image.

    APU v2 / l1_sim is functional-only (no cycle counts; see report 13
    §1, `caps["perf_is_placeholder"] = True`). We pass the emitted
    C++ source through to the container so the user can inspect it,
    but always return `cycles=None`.
    """
    cpp_src = "\n".join(str(c) for c in compiled.cmds if isinstance(c, str))
    if not _docker_image_exists("gsi-g2-l1sim"):
        return RunResult(
            cycles=None,
            stdout="simulator unavailable: gsi-g2-l1sim docker image missing",
            backend="apu_v2",
            extra={"program_src": cpp_src},
        )
    # l1_sim does not report cycles; we don't bother compiling here,
    # since the GTML build pipeline needs a real project tree. The
    # functional-correctness build flow is spec 014's territory.
    return RunResult(
        cycles=None,
        stdout="apu_v2 l1_sim is functional-only; cycles=None by design",
        backend="apu_v2",
        extra={"program_src": cpp_src},
    )


_BACKEND_RUN = {
    "samsung_hbm_pim": _run_samsung,
    "aim": _run_aim,
    "upmem": _run_upmem,
    "apu_v1": _run_apu_v1,
    "apu_v2": _run_apu_v2,
}


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
    ):
        self.target = target
        self.trace = trace
        self.cmds = cmds
        self.layout = layout
        # `_ctx` is currently only consumed by `_run_upmem`, which needs
        # `UPMEMCtx.get_kernel_src()` to render the full DPU envelope
        # around `cmds`. Other backends ignore the field.
        self._ctx = ctx
        # Lever 3 (SPEC-025 §5.4): shared-CRF host trigger schedule (one
        # HostTrigger per work-id), parallel to `cmds`. Empty for the
        # per-work-id path and every non-Samsung backend.
        self.host_schedule: list[HostTrigger] = list(
            getattr(ctx, "host_schedule", []) or []
        )

    def run(self, **inputs) -> RunResult:
        """Run this compiled artifact on its target backend.

        Returns a `RunResult` with cycle count (when the simulator
        provides one), the simulator's stdout, and the backend name.
        If the simulator/hardware is unavailable, the call still
        succeeds and returns `cycles=None` with a "simulator
        unavailable" stdout — `run()` never raises for missing tooling.
        """
        target_name = getattr(self.target, "name", None)
        runner = _BACKEND_RUN.get(target_name)
        if runner is None:
            return RunResult(
                cycles=None,
                stdout=f"no run hook for target {target_name!r}; "
                f"supported: {sorted(_BACKEND_RUN)}",
                backend=str(target_name),
            )
        return runner(self, **inputs)

    def run_batched(self, W, X, compare_native: bool = True) -> RunResult:
        """Run a batched GEMV (SPEC-026): B input vectors X[B,K] against
        one weight W. Samsung-only. The chosen placement's
        `weight_resident` flag (set by argmin) selects preload-once vs
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
        return _run_samsung_batched(self, W, X, compare_native=compare_native)


_BACKEND_CTX = {
    "samsung_hbm_pim": SamsungCtx,
    "aim": AimCtx,
    "upmem": UPMEMCtx,
    "apu_v1": APUv1Ctx,
    "apu_v2": APUv2Ctx,
}


def compile_for_target(
    target: Any,
    trace: MatchTrace,
    layout: Placement | list[Placement] | None = None,
) -> Compiled:
    """Lower a (target, trace) pair to a runnable backend artifact by
    walking the target's declarations.

    If ``layout`` is None, the autoscheduler picks per-kernel
    placements. If ``layout`` is a single `Placement`, it is broadcast
    across every kernel (back-compat for single-layer tests). If
    ``layout`` is a list, its length must equal the number of
    `@allo.work` kernels in the trace.
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

    from .spmw_autoschedule import _bucket_for_autoschedule
    buckets = _bucket_for_autoschedule(trace)
    n_groups = len(buckets) or 1

    if layout is None:
        layouts = autoschedule(target, trace)
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

    ctx = ctx_cls(target)
    _walk_and_emit(target, trace, ctx, layouts)

    # `Compiled.layout` historically held a single Placement; preserve
    # that for back-compat when there's only one kernel.
    stored_layout = layouts[0] if len(layouts) == 1 else layouts
    return Compiled(target, trace, ctx.cmds, stored_layout, ctx=ctx)
