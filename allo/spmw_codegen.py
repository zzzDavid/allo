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

    def after_match(self, match, n_emitted):
        # Inner-K JUMP -- the loop counter is computed off the
        # just-emitted MAC body, so the call must stay inside the
        # work-id window between compute and storeback.
        _emit_inner_loop_jump(match, self, n_emitted)


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

    def handle_c_name(self, handle) -> str:
        """Return the C identifier this handle lowers to.

        `Register`s use their declared name (the C compiler manages real
        register assignment). `MemoryRef`s render as `<mem.name>_buf[<idx>]`.
        Whole `Memory`s render as `<mem.name>_buf`.
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
            name = f"{mem.name}_buf[{handle.idx!r}]"
            self._name_table[key] = name
            return name
        if isinstance(handle, Memory):
            name = f"{handle.name}_buf"
            self._name_table[key] = name
            return name
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

    def get_kernel_src(self) -> str:
        """Return the assembled DPU kernel C source.

        Wraps `self.cmds` with the standard DPU runtime includes, the
        tasklet entry point, and a tasklet-id local. The coder may
        override the wrapper later for custom kernel shapes.
        """
        header = (
            "#include <defs.h>\n"
            "#include <mram.h>\n"
            "#include <stdint.h>\n"
            "\n"
            "int main(void) {\n"
            "    uint32_t tasklet_id = me();\n"
        )
        body = "\n".join("    " + line for line in self.cmds)
        footer = "\n    return 0;\n}\n"
        return header + body + footer


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
    it into `gvml_lookup_16` + `gvml_add_s16` so backend emit lambdas
    stay one-liners.
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

            gvml_lookup_16(<tmp>, <x>, <y>);
            gvml_add_s16(<acc>, <acc>, <tmp>);

        The autoscheduler may reserve a scratch VR alias via
        `bind_handle("mac_tmp", "<alias>")`; otherwise the canonical
        name `mac_tmp_vr` is used.
        """
        tmp_name = self._handle_names.get("mac_tmp", "mac_tmp_vr")
        acc_n = self._name(acc, "acc")
        x_n = self._name(x, "x")
        y_n = self._name(y, "y")
        self.cmds.append(f"gvml_lookup_16({tmp_name}, {x_n}, {y_n});")
        self.cmds.append(f"gvml_add_s16({acc_n}, {acc_n}, {tmp_name});")

    def append(self, line: str) -> None:
        """Low-level escape hatch -- append a raw C source line."""
        self.cmds.append(line)

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

    emitted_names: set[str] = set()
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
        if chosen in emitted_names:
            continue
        emitted_names.add(chosen)
        _emit_move(target, chosen, ctx)


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

    for _work_id, matches in _bucket_by_work_id(trace):
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
            before = len(ctx.cmds)
            if op_obj.accumulates:
                emit(bindings["x"], bindings["y"], bindings["acc"], ctx)
            else:
                emit(bindings["x"], bindings["y"], bindings["dst"], ctx)
            n_emitted = len(ctx.cmds) - before
            ctx.after_match(match, n_emitted)
        _schedule_moves(target, ctx, layout, role_to_memref, phase="post")


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


def _gvml_available() -> bool:
    try:
        import gvml  # noqa: F401
        return True
    except ImportError:
        return False


# --------------------------------------------------------------------- #
# Per-backend run hooks
# --------------------------------------------------------------------- #


def _run_samsung(compiled: "Compiled", **inputs) -> RunResult:
    """Run a compiled Samsung HBM-PIM artifact via `pim_driver`.

    `pim_driver` is a fixed-kernel CLI added to PIMSimulator for this
    project (ADD / MUL / RELU / GEMV). We infer the kernel from the
    emitted PIMCmd stream: presence of "MAC" -> GEMV, of "MUL" -> MUL,
    of "ADD" -> ADD, of a single unary -> RELU. We pass user inputs
    through .npy files and parse the `PIM_CYCLES total=` line from
    stdout.
    """
    root = _pimsim_root()
    driver = root / "pim_driver"
    if not driver.exists():
        return RunResult(
            cycles=None,
            stdout=f"simulator unavailable: pim_driver not found at {driver}",
            backend="samsung_hbm_pim",
        )

    # Detect kernel from emitted cmd types.
    op_types = {c.type_ for c in compiled.cmds if isinstance(c, PIMCmd)}
    if "MAC" in op_types:
        kernel = "GEMV"
    elif "MUL" in op_types:
        kernel = "MUL"
    elif "ADD" in op_types:
        kernel = "ADD"
    else:
        kernel = "RELU"

    try:
        import numpy as np
    except ImportError:
        return RunResult(
            cycles=None,
            stdout="numpy unavailable",
            backend="samsung_hbm_pim",
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
                return RunResult(
                    cycles=None,
                    stdout="GEMV needs W and x kwargs",
                    backend="samsung_hbm_pim",
                )
            W = np.asarray(W, dtype=np.float16)
            x = np.asarray(x, dtype=np.float16)
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
                return RunResult(
                    cycles=None,
                    stdout=f"{kernel} needs a/b (or in0/in1) kwargs",
                    backend="samsung_hbm_pim",
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
                return RunResult(
                    cycles=None,
                    stdout="RELU needs a (or in0) kwarg",
                    backend="samsung_hbm_pim",
                )
            a = np.asarray(a, dtype=np.float16).reshape(-1)
            a_path = td_path / "a.npy"
            np.save(a_path, a)
            argv += ["--in0", str(a_path), "--n", str(a.size)]

        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                cwd=str(root),  # pim_driver resolves ini/ paths relative to cwd
                timeout=600,
                check=False,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            return RunResult(
                cycles=None,
                stdout=f"pim_driver invocation failed: {exc}",
                backend="samsung_hbm_pim",
            )

        stdout = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")
        combined = stdout + ("\n" + stderr if stderr else "")
        cycles = None
        match = re.search(r"PIM_CYCLES total=(\d+)", combined)
        if match:
            cycles = int(match.group(1))
        return RunResult(
            cycles=cycles,
            stdout=combined,
            backend="samsung_hbm_pim",
            extra={"kernel": kernel, "returncode": proc.returncode},
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
        return RunResult(
            cycles=None,
            stdout=f"ramulator2 invocation failed: {exc}",
            backend="aim",
        )
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
    cycles = max(cycle_vals) if cycle_vals else None
    return RunResult(
        cycles=cycles,
        stdout=combined,
        backend="aim",
        extra={"returncode": proc.returncode},
    )


def _run_upmem(compiled: "Compiled", **inputs) -> RunResult:
    """Run a compiled UPMEM artifact through the Go uPIMulator.

    The current uPIMulator front-end only accepts pre-registered PrIM
    benchmark names (`--benchmark VA`, `GEMV`, ...). Running an
    arbitrary emitted kernel would require registering it as a new
    benchmark in `src/assembler/`. For now we run the closest existing
    PrIM kernel as a cycle proxy and return its count; the emitted C
    source is preserved in `extra["kernel_src"]` for inspection.
    """
    root = _upim_root()
    binary = root / "build" / "uPIMulator"
    if not binary.exists():
        return RunResult(
            cycles=None,
            stdout=f"simulator unavailable: {binary} not found",
            backend="upmem",
        )

    # Pull the emitted C kernel out of the ctx-style command list.
    if isinstance(compiled.cmds, list) and compiled.cmds and isinstance(
        compiled.cmds[0], str
    ):
        kernel_src = "\n".join(compiled.cmds)
    else:
        kernel_src = ""

    # Map emitted ops to a PrIM proxy: a MAC-shaped body -> GEMV, else VA.
    benchmark = "VA"
    if any("+=" in line and "*" in line for line in compiled.cmds if isinstance(line, str)):
        benchmark = "GEMV"

    with tempfile.TemporaryDirectory() as td:
        bin_dir = Path(td) / "bin"
        bin_dir.mkdir()
        try:
            proc = subprocess.run(
                [
                    str(binary),
                    "--root_dirpath", str(root),
                    "--bin_dirpath", str(bin_dir),
                    "--benchmark", benchmark,
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
            return RunResult(
                cycles=None,
                stdout=f"uPIMulator invocation failed: {exc}",
                backend="upmem",
                extra={"kernel_src": kernel_src},
            )

        stdout = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")
        combined = stdout + ("\n" + stderr if stderr else "")
        # uPIMulator prints "cycle: <N>" in log.txt; also tail it if present.
        log_path = bin_dir / "log.txt"
        if log_path.exists():
            combined += "\n" + log_path.read_text(errors="replace")
        cycle_match = re.search(r"cycle[s]?\s*[:=]\s*(\d+)", combined, re.IGNORECASE)
        cycles = int(cycle_match.group(1)) if cycle_match else None
        return RunResult(
            cycles=cycles,
            stdout=combined,
            backend="upmem",
            extra={
                "kernel_src": kernel_src,
                "benchmark": benchmark,
                "returncode": proc.returncode,
            },
        )


def _run_apu_v1(compiled: "Compiled", **inputs) -> RunResult:
    """Run a compiled APU v1 artifact against real GSI hardware.

    The walker accumulates one C-source GVML call per `cmd(...)`; on
    real hardware we'd hand that off to `gen_apu_v1_low_mode_project`,
    build with the ARC toolchain, and run the resulting binary. On
    this machine that path requires `gvml` to be importable; without
    it we return an "unavailable" result rather than fail.
    """
    if not _gvml_available():
        return RunResult(
            cycles=None,
            stdout="simulator unavailable: `gvml` Python module not importable",
            backend="apu_v1",
            extra={"kernel_src": "\n".join(
                str(c) for c in compiled.cmds
                if isinstance(c, str)
            )},
        )

    # Real-hardware path: build the project via apu_v1_codegen and run
    # the resulting binary. Kept gated until a workload-level harness
    # arrives — return None cycles for now, with the emitted source.
    return RunResult(
        cycles=None,
        stdout="apu_v1 build harness not wired (gvml is importable but "
        "the workload-level build helper is owned by spec 014).",
        backend="apu_v1",
        extra={"kernel_src": "\n".join(
            str(c) for c in compiled.cmds if isinstance(c, str)
        )},
    )


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
    ):
        self.target = target
        self.trace = trace
        self.cmds = cmds
        self.layout = layout

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
    return Compiled(target, trace, ctx.cmds, stored_layout)
