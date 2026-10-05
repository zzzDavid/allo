# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SK hynix AiM codegen context and ramulator2 runtime (moved from spmw_codegen)."""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from .spmw_target import MemoryRef, Register
from .spmw_codegen import (
    CodegenContext,
    _eval_sym,
    _parse_loop_bound,
    RunResult,
    SimulatorUnavailable,
    Compiled,
)
from .spmw_simenv import aim_root as _aim_root
from .spmw_simenv import aim_unavailable_reason as _aim_unavailable_reason


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
        "WR_SBK": ("gpr_addr_0", "channel_mask", "bank_index", "row_addr"),
        "WR_ABK": ("gpr_addr_0", "channel_mask", "row_addr"),
        "WR_GB": ("opsize", "gpr_addr_0", "channel_mask"),
        "WR_BIAS": ("gpr_addr_0", "channel_mask"),
        "WR_AFLUT": ("opsize",),
        "RD_MAC": ("gpr_addr_0", "channel_mask"),
        "RD_AF": ("gpr_addr_0", "channel_mask"),
        "RD_SBK": ("gpr_addr_0", "channel_mask", "bank_index", "row_addr"),
        "COPY_BKGB": ("opsize", "channel_mask", "bank_index", "row_addr"),
        "COPY_GBBK": ("opsize", "channel_mask", "bank_index", "row_addr"),
        "MAC_SBK": ("opsize", "channel_mask", "bank_index", "row_addr"),
        "MAC_ABK": ("opsize", "channel_mask", "row_addr"),
        "AF": ("channel_mask",),
        "EWMUL": ("opsize", "channel_mask", "row_addr"),
        "EWADD": ("opsize", "gpr_addr_0", "gpr_addr_1"),
        "SYNC": (),
        "EOC": (),
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
            if handle.name in ("mac_reg", "af_reg"):
                return f"{handle.name}=0"
            raise NotImplementedError(
                f"AimCtx: register {handle.name!r} has no operand mapping."
            )
        if isinstance(handle, MemoryRef):
            mem = handle.memory
            if mem.name == "banks":
                # `handle.idx` may be a symbolic SymExpr (e.g. autoscheduler
                # produces `8*bg + bank` with placeholder UnitIds); collapse
                # to a concrete int via `_eval_sym` so ramulator2 accepts it.
                env = {
                    level: value
                    for level, value in enumerate(getattr(self, "_active_work_id", ()))
                }
                bank_int = _eval_sym(handle.idx, env)
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
            if handle.name == "gpr":
                prefix = "gpr_in" if role.startswith("src") else "gpr_out"
                return f"{prefix}=0"
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
            if handle.name in ("gpr", "mac_reg", "af_reg"):
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
                env = {
                    level: value
                    for level, value in enumerate(getattr(self, "_active_work_id", ()))
                }
                bank_int = _eval_sym(handle.idx, env)
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
            elif handle.name == "gpr":
                if role == "src1" and "gpr_addr_0" in fields:
                    fields["gpr_addr_1"] = 0
                else:
                    fields.setdefault("gpr_addr_0", 0)
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
        work_id = getattr(self, "_active_work_id", ())
        channel_id = int(work_id[0]) if work_id else 0
        field_map: dict = {"channel_mask": 1 << channel_id}
        self._handle_to_fields(src0, "src0", field_map)
        self._handle_to_fields(src1, "src1", field_map)
        self._handle_to_fields(dst, "dst", field_map)
        # Caller overrides (e.g. `row_addr=...`, `opsize=...`).
        for k, v in fields.items():
            field_map[k] = v

        # Human-readable mirror — keeps the older key=value annotation so
        # tests can introspect handle-derived field labels.
        parts = [f"AiM {name}", f"ch={channel_id}"]
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

    @staticmethod
    def _memref_shape(type_text: str | None) -> tuple[int, ...]:
        """Parse the static dimensions retained on an operand binding."""
        if not type_text or not type_text.startswith("memref<"):
            return ()
        body = type_text[len("memref<") :].split(">", 1)[0]
        dims = []
        for token in body.split("x")[:-1]:
            try:
                dims.append(int(token))
            except ValueError:
                return ()
        return tuple(dims)

    def gemv_shape(self, match) -> tuple[int, int, int]:
        """Recover ``(outputs, reduction, batches)`` from one MAC match.

        The matcher retains the loop bounds and source memref shapes.  The
        innermost loop is the reduction; an input-only loop is the independent
        GEMV batch; the remaining axis of an operand containing the reduction
        is the logical output extent.  This covers GEMV, transposed GEMV and
        the batched-GEMV decomposition of GEMM without benchmark metadata.
        """
        if not match.enclosing_loops:
            raise ValueError("AiM MAC match has no reduction loop")
        reduction_var, _lb, reduction_ub, _step = match.enclosing_loops[-1]
        reduction = _parse_loop_bound(reduction_ub)
        if reduction is None or reduction <= 0:
            raise ValueError(f"AiM reduction extent is not static: {reduction_ub}")
        batch_var = match.extra.get("batch_loop_var")
        batches = int(match.extra.get("batch_dim", 1) or 1)

        output_candidates: list[int] = []
        for operand in match.operands:
            if operand.is_loop_carried:
                continue
            shape = self._memref_shape(getattr(operand, "memref_type", None))
            if len(shape) != len(operand.indices):
                continue
            if reduction_var not in operand.indices:
                continue
            for extent, index in zip(shape, operand.indices):
                if index == reduction_var or index == batch_var:
                    continue
                output_candidates.append(extent)
        if not output_candidates:
            # Compatibility for old/synthetic matches without retained types:
            # use the per-work-item output loop and structural channel grid.
            outer = _parse_loop_bound(match.enclosing_loops[0][2])
            channels = int(
                getattr(self.target.unit("channel"), "axes", {}).get("channel", 32)
            )
            if outer is None:
                raise ValueError("AiM cannot recover output extent from match")
            outputs = outer * channels
        else:
            outputs = output_candidates[0]
        return outputs, reduction, batches

    def emit_gemv(self, match) -> None:
        """Emit a complete, native ABK GEMV trace segment from a MAC match."""
        outputs, reduction, batches = self.gemv_shape(match)
        groups = (outputs + 15) // 16
        columns = (reduction + 15) // 16
        all_channels = (1 << 32) - 1
        # Keep batched GEMM compact. The runner profiles this exact native
        # GEMV segment once and multiplies by the statically proven repeat
        # count, matching the PolyBench AiM summed-leg methodology.
        self.append(f"# TENON_GEMV {outputs} {reduction} repeat={batches}")
        self.append("W CFR 0 1")
        self.append(f"AiM WR_GB 2 2 {all_channels}")
        for group in range(groups):
            self.append(f"AiM WR_ABK 4 1 {group}")
        for group in range(groups):
            self.append(f"AiM MAC_ABK {columns} {all_channels} {group}")
        self.append(f"AiM RD_MAC 8 {all_channels}")

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
            if dst_handle.name == "mac_reg":
                return (None, "RD_MAC")
            if dst_handle.name == "af_reg":
                return (None, "RD_AF")
            if dst_handle.name == "gpr":
                return (None, None)
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
        operation_name = getattr(
            getattr(self, "_active_placement", None), "extra", {}
        ).get("operation_name", match.target_op_name)
        if operation_name not in ("MAC", "MAC_ABK"):
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
        if (
            len(parts) < 3
            or parts[0] != "AiM"
            or parts[1]
            not in (
                "MAC_SBK",
                "MAC_ABK",
            )
        ):
            return
        # ``opsize`` counts 256-bit DRAM columns, not scalar loop iterations.
        # One column supplies 16 BF16 lanes in each targeted bank. MAC_ABK's
        # banks hold independent output rows; they do not partition K. Keep
        # this conversion identical to the AiM cost program's `_columns`.
        lanes = 256 // int(getattr(self.target.banks, "width", 16))
        columns = (k + lanes - 1) // lanes
        parts[2] = str(columns)
        self.cmds[-1] = " ".join(parts)
        # Mirror the key=value annotation so ``_human_lines`` stays
        # consistent with the positional trace for introspection.
        import re

        human_last = self._human_lines[-1]
        if "opsize=" in human_last:
            self._human_lines[-1] = re.sub(
                r"opsize=\d+", f"opsize={columns}", human_last
            )
        else:
            self._human_lines[-1] = f"{human_last}  opsize={columns}"


def _aim_runtime_segments(commands) -> tuple[tuple[str, int, tuple[str, ...]], ...]:
    """Parse the exact pre-terminated AiM streams consumed by Ramulator2."""

    segments: list[tuple[str, int, tuple[str, ...]]] = []
    current_label = "program"
    current_repeat = 1
    current: list[str] = []

    def finish_segment() -> None:
        nonlocal current
        if not current:
            return
        eoc_positions = [
            index for index, line in enumerate(current) if line.strip() == "AiM EOC"
        ]
        if eoc_positions != [len(current) - 1]:
            raise ValueError(
                "AiM runtime segment must contain exactly one trailing EOC"
            )
        segments.append((current_label, current_repeat, tuple(current)))
        current = []

    for raw in map(str, commands):
        if raw.startswith("# TENON_GEMV "):
            finish_segment()
            current_label = raw[2:]
            repeat_match = re.search(r"repeat=(\d+)", raw)
            current_repeat = int(repeat_match.group(1)) if repeat_match else 1
            current = []
        else:
            current.append(raw)
    finish_segment()
    if not segments:
        raise ValueError("AiM runtime requires at least one terminated segment")
    return tuple(segments)


def _run_aim(compiled: "Compiled", **inputs) -> RunResult:
    """Run a compiled AiM artifact through ramulator2 in Docker.

    `compiled.cmds` is a list of already-terminated text trace lines. We
    serialize those exact candidate-owned segments and invoke the ramulator2
    binary inside the `aim-simulator-build` Docker image. The simulator's
    YAML stdout carries per-channel `memory_system_cycles`; we parse
    out the max across channels as the headline cycle count.
    """
    reason = _aim_unavailable_reason()
    if reason is not None:
        raise SimulatorUnavailable("aim", reason)
    root = _aim_root()
    yaml_cfg = root / "test" / "example.yaml"

    cfg_rel = yaml_cfg.relative_to(root)

    def run_segment(lines: list[str]) -> tuple[int, str, int]:
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".trace", dir=str(root / "test")
        ) as tf:
            for line in lines:
                tf.write(str(line) + "\n")
            trace_path = Path(tf.name)
        trace_rel = trace_path.relative_to(root)
        try:
            proc = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "-v",
                    f"{root}:/work",
                    "aim-simulator-build",
                    "bash",
                    "-c",
                    f"cd /work && ./build/ramulator2 -f {cfg_rel} -t {trace_rel}",
                ],
                capture_output=True,
                timeout=600,
                check=False,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            raise RuntimeError(f"AiM ramulator2 invocation failed: {exc}") from exc
        finally:
            trace_path.unlink(missing_ok=True)
        stdout = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")
        combined = stdout + ("\n" + stderr if stderr else "")
        values = [
            int(m) for m in re.findall(r"memory_system_cycles:\s*(\d+)", combined)
        ]
        if not values:
            raise RuntimeError(
                "AiM ramulator2 returned but stdout missing "
                "'memory_system_cycles: ...' line; tail: " + combined[-400:]
            )
        return max(values), combined, proc.returncode

    # Compact traces carry independently dispatched GEMV templates. Profile
    # every native segment once, then apply its compile-time repeat count. All
    # base values are direct Ramulator2 measurements from this invocation.
    segments = tuple(getattr(compiled, "runtime_segments", ()))
    if not segments:
        segments = _aim_runtime_segments(compiled.cmds)

    total_cycles = 0
    logs: list[str] = []
    returncodes: list[int] = []
    cache: dict[tuple[str, ...], tuple[int, str, int]] = {}
    for label, repeat, lines in segments:
        key = tuple(lines)
        if key not in cache:
            cache[key] = run_segment(lines)
        raw_cycles, log, returncode = cache[key]
        total_cycles += raw_cycles * repeat
        returncodes.append(returncode)
        logs.append(
            f"=== {label}; repeat={repeat}; raw_cycles={raw_cycles}; "
            f"composed_cycles={raw_cycles * repeat} ===\n{log}"
        )
    cycles = total_cycles
    combined = "\n".join(logs)
    return RunResult(
        cycles=cycles,
        stdout=combined,
        backend="aim",
        extra={
            "returncode": max(returncodes, default=0),
            "segments": len(segments),
            "unique_segments": len(cache),
        },
    )
