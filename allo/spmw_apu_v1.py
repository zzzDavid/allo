# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""GSI APU v1 codegen context and device runtime (moved from spmw_codegen)."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from .spmw_target import MemoryRef, Register
from .spmw_codegen import (
    CodegenContext,
    RunResult,
    SimulatorUnavailable,
    Compiled,
)
from .spmw_simenv import apu_v1_unavailable_reason as _apu_v1_unavailable_reason


class APUv1Ctx(CodegenContext):
    """Codegen ctx for GSI APU v1.

    Output is a list of C-source lines containing real GVML calls. Logical
    reductions use an F2 layout-derived group size and lower to native FP16
    multiply plus ``gvml_add_subgrps_f16_grp``.
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
        self._used_vr_names: list[str] = []
        self.group_size: int = 32768
        self.vector_batches: int = 1

    def _enabled(self) -> bool:
        """Emit one SPMD body; the host launches it on every APUC."""
        work_id = tuple(getattr(self, "_active_work_id", ()))
        return not work_id or all(int(value) == 0 for value in work_id)

    def _append(self, line: str) -> None:
        if self._enabled():
            self.cmds.append(line)

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

        if isinstance(handle, str):
            return self._handle_names.get(handle, f"{handle}_vr")
        cached = self._handle_names.get(id(handle))
        if cached is not None:
            return cached
        if isinstance(handle, Register):
            name = handle.name or "vr_unknown"
            if name.startswith("vr") and name[2:].isdigit():
                if name not in self._used_vr_names:
                    self._used_vr_names.append(name)
            return name
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
        self._append(name + "(" + ", ".join(operands) + ");")

    @staticmethod
    def _group_enum(group_size: int) -> str:
        if group_size <= 0 or group_size > 32768 or group_size & (group_size - 1):
            raise ValueError(f"invalid GVML group size {group_size}")
        suffix = f"{group_size // 1024}K" if group_size >= 1024 else str(group_size)
        return f"GVML_P2_{suffix}"

    def emit_l4_to_vr(self, role: str, vr, vm_index: int) -> None:
        pointer = self._L4_PTR_BY_ROLE[role]
        if self.vector_batches > 1:
            pointer = f"({pointer} + batch * 32768)"
        vr_name = self._name(vr, role)
        self._append(f"direct_dma_l4_to_l1_32k(GVML_VM_{vm_index}, {pointer});")
        self._append(f"gvml_load_16({vr_name}, GVML_VM_{vm_index});")

    def emit_vr_to_l4(self, role: str, vr, vm_index: int) -> None:
        pointer = self._L4_PTR_BY_ROLE[role]
        if self.vector_batches > 1:
            pointer = f"({pointer} + batch * 32768)"
        vr_name = self._name(vr, role)
        self._append(f"gvml_store_16(GVML_VM_{vm_index}, {vr_name});")
        self._append(f"direct_dma_l1_to_l4_32k({pointer}, GVML_VM_{vm_index});")

    def emit_group_reduce_f16(self, dst, src) -> None:
        dst_name = self._name(dst, "acc")
        src_name = self._name(src, "x")
        group = self._group_enum(self.group_size)
        self._append(
            "gvml_add_subgrps_f16_grp("
            f"{dst_name}, {src_name}, {group}, GVML_P2_1, 0, "
            "GVML_VM_3, reduce_tmp_vr);"
        )

    def emit_grouped_f16_mac(self, acc, x, y) -> None:
        self._append(
            f"gvml_mul_f16(mac_tmp_vr, {self._name(x, 'x')}, " f"{self._name(y, 'y')});"
        )
        self.emit_group_reduce_f16(acc, "mac_tmp")

    def append(self, line: str) -> None:
        """Low-level escape hatch -- append a raw C source line."""
        self._append(line)

    def iter_vr_aliases(self) -> list[tuple[str, str]]:
        """Return [(c_name, gvml_vr_enum), ...] in bind order.

        Each entry becomes one line `enum gvml_vr16 <c_name> = <enum>;`
        in the emitted device.c. Until the regalloc spec rebinds VRs
        per live-range, the autoscheduler's bind list may be empty; in
        that case the build harness substitutes a canonical default
        (vrs + mac_tmp_vr) so the emitted body still compiles.
        """
        out: list[tuple[str, str]] = []
        used_indices = set()
        for name in self._used_vr_names:
            index = int(name[2:])
            used_indices.add(index)
            out.append((name, f"GVML_VR16_{index}"))
        for temporary in ("mac_tmp_vr", "reduce_tmp_vr"):
            index = next(i for i in range(15) if i not in used_indices)
            used_indices.add(index)
            out.append((temporary, f"GVML_VR16_{index}"))
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
        if dst_handle is None:
            return (None, None)
        if isinstance(dst_handle, Register):
            role_name = {"x": "X", "y": "Y", "acc": "ACC", "dst": "ACC"}.get(role)
            if role_name is None:
                return (None, None)
            store = "ST_ACC_VR_TO_L4" if role_name == "ACC" else None
            return (f"LD_{role_name}_L4_TO_VR", store)
        return (None, None)

    def resolve_spill_moves(self, tier, home_handle, *, n_entries=1):
        return super().resolve_spill_moves(tier, home_handle, n_entries=n_entries)


def _apu_v1_kernel_src(compiled: "Compiled") -> str:
    return "\n".join(str(c) for c in compiled.cmds if isinstance(c, str))


def _parse_apu_v1_prof_print(text: str) -> int | None:
    """Return the four-APUC parallel makespan from ``total`` PROF_PRINTs.

    Hardware emits fields with a colon separator, e.g.
    `ARCT[0]: ***  total - hits:1 seu:374 crun:170227 iall:37027 ...`.
    ``ledag flo`` can include older buffered records, so only the final four
    totals belong to the batch just launched. Those APUCs run concurrently;
    their cost is the maximum CRUN, not their sum. Accept `=` as well for
    forward compatibility. Returns None if no match.
    """
    totals = re.findall(r"\btotal\b[^\n]*?\bcrun\s*[:=]\s*(\d+)", text)
    if totals:
        return max(int(value) for value in totals[-4:])
    counters = re.findall(r"\bcrun\s*[:=]\s*(\d+)", text)
    if counters:
        return max(int(value) for value in counters[-4:])
    return None


def _apu_v1_output_specs(compiled: "Compiled", inputs: dict) -> dict:
    """Derive `output_specs: {role: (shape, dtype)}` from the trace.

    This is the fallback for non-grouped operations. Grouped contractions use
    `_apu_v1_prepare_io`, which knows the logical output shape and FP16 ABI.
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
        if name and name.startswith("local_"):
            name = name[len("local_") :]
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
    """Normalize inputs and realize the grouped-contraction callable ABI.

    Dense contractions are packed one output dot product per GVML group, split
    over four APUCs, and streamed in 32K-lane batches. The returned metadata
    drives group-head gathering after execution.
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

    layout = getattr(compiled, "layout", None)
    if isinstance(layout, list):
        layout = layout[0] if layout else None
    extra = getattr(layout, "extra", {}) or {}
    group_size = int(extra.get("group_size", 0) or 0)
    batches = max(1, int(extra.get("n_out_tiles", 1)))
    if group_size and len(normalized) >= 2:
        trace_output_names = []
        operand_names = []
        for match in compiled.trace.matches:
            name = match.result_memref_name
            if name:
                name = name[len("local_") :] if name.startswith("local_") else name
                if name not in trace_output_names:
                    trace_output_names.append(name)
            for operand in match.operands:
                if operand.role not in ("x", "y") or not operand.memref_name:
                    continue
                operand_name = operand.memref_name
                if operand_name.startswith("local_"):
                    operand_name = operand_name[len("local_") :]
                if operand_name not in operand_names:
                    operand_names.append(operand_name)
        # Reduction matchers name the loop-carried scalar ``acc`` as the
        # result.  At the callable boundary, the real output is the source
        # argument not bound to semantic x/y (C in GEMM, y in GEMV).
        abi_output_names = [name for name in normalized if name not in operand_names]
        operands = [
            (name, arr)
            for name, arr in normalized.items()
            if name in operand_names and arr.ndim > 0
        ]
        matrices = [(name, arr) for name, arr in operands if arr.ndim == 2]
        if len(matrices) >= 2:
            # Grouped GEMM: one group is one output dot product.  Rows are
            # partitioned over four APUCs; each core may stream several VRs.
            (lhs_name, lhs), (rhs_name, rhs) = matrices[:2]
            if lhs.shape[1] == rhs.shape[0]:
                if lhs.dtype != np.float16 or rhs.dtype != np.float16:
                    raise ValueError(
                        "APU v1 grouped contractions require float16 inputs; "
                        f"got {lhs.dtype} and {rhs.dtype}"
                    )
                rows, reduction = lhs.shape
                cols = rhs.shape[1]
                if reduction > group_size:
                    raise ValueError("APU v1 group is smaller than GEMM reduction")
                groups_per_vr = 32768 // group_size
                rows_per_core = (rows + 3) // 4
                required = (rows_per_core * cols + groups_per_vr - 1) // groups_per_vr
                batches = max(batches, required)
                packed_lhs = np.zeros((4, batches, 32768), dtype=lhs.dtype)
                packed_rhs = np.zeros((4, batches, 32768), dtype=rhs.dtype)
                for core in range(4):
                    row0 = core * rows_per_core
                    pairs = [
                        (row, col)
                        for row in range(row0, min(rows, row0 + rows_per_core))
                        for col in range(cols)
                    ]
                    for output_index, (row, col) in enumerate(pairs):
                        batch, group = divmod(output_index, groups_per_vr)
                        base = group * group_size
                        packed_lhs[core, batch, base : base + reduction] = lhs[row, :]
                        packed_rhs[core, batch, base : base + reduction] = rhs[:, col]
                normalized = {
                    lhs_name: packed_lhs.reshape(-1),
                    rhs_name: packed_rhs.reshape(-1),
                }
                output_name = (
                    abi_output_names[0]
                    if abi_output_names
                    else trace_output_names[0] if trace_output_names else "output"
                )
                output_specs = {output_name: ((4, batches, 32768), np.dtype(lhs.dtype))}
                return (
                    normalized,
                    output_specs,
                    {
                        "kind": "gemm",
                        "output": output_name,
                        "logical_shape": (rows, cols),
                        "rows_per_core": rows_per_core,
                        "group_size": group_size,
                        "groups_per_vr": groups_per_vr,
                    },
                )

        raise NotImplementedError(
            "APU v1 physical execution currently supports a dense FP16 matrix "
            "contraction with lhs.shape[1] == rhs.shape[0]"
        )

    output_specs = _apu_v1_output_specs(compiled, normalized)
    return normalized, output_specs, None


def _run_apu_v1(compiled: "Compiled", **inputs) -> RunResult:
    """Run a compiled APU v1 artifact against real GSI hardware.

    Materializes a build directory via `gen_apu_v1_low_mode_project`,
    runs `make` with the ARC GNU toolchain, executes the resulting
    binary against the Gemini PCI device, and parses cycle counts from
    its PROF_PRINT stdout. Raises `SimulatorUnavailable` when an
    environment gate (toolchain, template dir, PCI sysfs node, GVML SDK
    headers) is missing. Build failures past the gate raise
    `RuntimeError` so tests cannot silently PASS; a missing
    PROF_PRINT 'total crun=' line, however, yields `cycles=None` rather
    than raising -- the device program ran (or tried to) but produced
    no profile output, which is observable through the returned
    RunResult.stdout. See spec 017 and SPEC-001 §3.5.
    """
    kernel_src = _apu_v1_kernel_src(compiled)

    skip_reason = _apu_v1_unavailable_reason()
    if skip_reason:
        raise SimulatorUnavailable("apu_v1", skip_reason)

    from .spmw_apu_v1_build import (
        _apu_v1_build_config,
        _assert_gvml_sdk_present,
        gen_apu_v1_low_mode_project,
    )

    # Build-harness probe: a missing SDK here means the toolchain gate
    # passed but headers are absent; fail fast with a readable error
    # rather than letting `make` emit a 200-line stderr blob below.
    _assert_gvml_sdk_present()
    build_config = _apu_v1_build_config()

    tmpdir = tempfile.mkdtemp(prefix="tenon-apu-v1-")
    try:
        # prepare_io is build-harness Python; ValueError here is a Tenon
        # bug or user-input contract violation, not an env skip.
        inputs_np, output_specs, packing = _apu_v1_prepare_io(compiled, inputs)

        # Write each input to <tmpdir>/in_<role>.bin so host.c can fread it.
        input_bin_paths: dict[str, str] = {}
        for role, arr in inputs_np.items():
            p = Path(tmpdir) / f"in_{role}.bin"
            arr.tofile(str(p))
            input_bin_paths[role] = str(p)

        output_bin_paths: dict[str, str] = {
            role: str(Path(tmpdir) / f"out_{role}.bin") for role in output_specs
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
                build_config.make_command,
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

        bin_path = build_config.binary_path(project_dir, "tenon-kernel")
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
        if proc.returncode != 0:
            raise RuntimeError(
                f"APU v1 device program failed with exit code {proc.returncode}:\n"
                f"{binary_output[-2000:]}\n--- project_dir: {project_dir}"
            )

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
                    chr(b) for b in ledag_raw if 32 <= b < 127 or b in (9, 10, 13)
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

        if packing and packing["kind"] == "gemm":
            packed = outputs[packing["output"]]
            rows, cols = packing["logical_shape"]
            rows_per_core = packing["rows_per_core"]
            group_size = packing["group_size"]
            groups_per_vr = packing["groups_per_vr"]
            logical = np.zeros((rows, cols), dtype=packed.dtype)
            for core in range(4):
                row0 = core * rows_per_core
                pairs = [
                    (row, col)
                    for row in range(row0, min(rows, row0 + rows_per_core))
                    for col in range(cols)
                ]
                for output_index, (row, col) in enumerate(pairs):
                    batch, group = divmod(output_index, groups_per_vr)
                    logical[row, col] = packed[core, batch, group * group_size]
            outputs[packing["output"]] = logical

        return RunResult(
            cycles=cycles,
            stdout=combined,
            backend="apu_v1",
            extra={
                "outputs": outputs,
                "kernel_src": kernel_src,
                "build_mode": build_config.mode,
                "project_dir": str(project_dir),
                "returncode": proc.returncode,
            },
        )
    finally:
        if os.environ.get("TENON_APU_V1_KEEP_TMP") != "1":
            shutil.rmtree(tmpdir, ignore_errors=True)
