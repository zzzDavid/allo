# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""MLIR-to-scalar-ARC baseline programs for GSI APU v1.

GVML remains the preferred group-parallel path.  This module handles complete
programs whose control flow or scalar operations cannot yet be vectorized: the
same retained Allo MLIR is emitted as C11, compiled by the ARC toolchain, and
executed against L4-backed arrays on APUC 0.  Other APUCs are intentionally not
launched until a phase proves that its outer frontier is independent and has a
device-wide barrier plan.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, replace
import fcntl
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import time
from types import MappingProxyType

import numpy as np

from ..backend.c import emit_c_from_mlir
from ..customize import customize
from ..perf import CostEvent
from ..perf.graph import ExecutionGraph
from ..spmw_apu_v1_build import _apu_v1_build_config
from ..spmw_codegen import RunResult
from .upmem_analysis import analyze_upmem_mlir
from .apu_v1_hybrid import discover_apu_v1_hybrid_manifest


_C_TYPES = {
    "i1": ("bool", np.dtype(np.bool_)),
    "ui1": ("bool", np.dtype(np.bool_)),
    "i8": ("int8_t", np.dtype(np.int8)),
    "ui8": ("uint8_t", np.dtype(np.uint8)),
    "i16": ("int16_t", np.dtype(np.int16)),
    "ui16": ("uint16_t", np.dtype(np.uint16)),
    "i32": ("int32_t", np.dtype(np.int32)),
    "ui32": ("uint32_t", np.dtype(np.uint32)),
    "i64": ("int64_t", np.dtype(np.int64)),
    "ui64": ("uint64_t", np.dtype(np.uint64)),
    "index": ("int64_t", np.dtype(np.int64)),
    "f32": ("float", np.dtype(np.float32)),
    "f64": ("double", np.dtype(np.float64)),
}

_LOCAL_ARRAY = re.compile(
    r"^(?P<indent>[ \t]+)"
    r"(?P<ctype>bool|(?:u?int(?:8|16|32|64)_t)|float|double)\s+"
    r"(?P<name>[A-Za-z_]\w*)"
    r"(?P<dims>(?:\[\d+\])+);(?P<comment>[^\n]*)$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class APUv1Phase:
    """One complete Allo function for APU v1.

    ``vectorize=False`` lowers the function to scalar ARC C on APUC 0;
    ``vectorize="required"`` requires one native GVML recipe to prove it.
    ``argument_bounds`` declares inclusive integer ranges of source arguments
    (consumed by the native recipes).  An ``APUv1Program`` has exactly one
    phase; precision policies, vector layouts and multi-phase dataflow fields
    were removed with the hybrid runtime (release audit V1-11).
    """

    kernel: object
    name: str | None = None
    instantiate: tuple = ()
    result_names: tuple[str, ...] = ()
    vectorize: str | bool = False
    argument_bounds: object = ()

    def __post_init__(self):
        if not callable(self.kernel):
            raise TypeError("APUv1Phase kernel must be callable")
        object.__setattr__(self, "instantiate", tuple(self.instantiate))
        object.__setattr__(self, "result_names", tuple(self.result_names))
        vectorize = "required" if self.vectorize is True else self.vectorize
        if vectorize not in {False, "required"}:
            raise ValueError("APUv1Phase.vectorize must be False or 'required'")
        argument_bounds = dict(self.argument_bounds)
        source_parameters = set(inspect.signature(self.kernel).parameters)
        unknown_bounds = set(argument_bounds) - source_parameters
        if unknown_bounds:
            raise ValueError(
                f"argument_bounds names are not kernel arguments: {sorted(unknown_bounds)}"
            )
        for name, interval in argument_bounds.items():
            if (
                not isinstance(interval, (tuple, list))
                or len(interval) != 2
                or any(not isinstance(value, int) for value in interval)
                or interval[0] > interval[1]
            ):
                raise ValueError(
                    f"argument_bounds[{name!r}] must be an inclusive integer pair"
                )
        object.__setattr__(self, "vectorize", vectorize)
        object.__setattr__(
            self,
            "argument_bounds",
            MappingProxyType(
                {name: tuple(interval) for name, interval in argument_bounds.items()}
            ),
        )

    @property
    def phase_name(self):
        return self.name or getattr(self.kernel, "__name__", "phase")


class APUv1Program:
    """A one-phase APU v1 program (``phases`` is a one-element tuple)."""

    def __init__(self, phases, *, name: str | None = None):
        self.phases = tuple(phases)
        if not self.phases or not all(
            isinstance(phase, APUv1Phase) for phase in self.phases
        ):
            raise ValueError("APUv1Program requires one APUv1Phase value")
        if len(self.phases) != 1:
            raise ValueError(
                "APUv1Program supports exactly one phase; multi-phase hybrid "
                "programs were removed (compile each phase separately)"
            )
        self.name = name or self.phases[0].phase_name

    def build(self):
        return self


@dataclass(frozen=True)
class _Argument:
    name: str
    dtype: str
    shape: tuple[int, ...]
    mode: str
    source: str

    @property
    def numpy_dtype(self):
        try:
            return _C_TYPES[self.dtype][1]
        except KeyError as exc:
            raise TypeError(f"APU v1 scalar ABI does not support {self.dtype}") from exc

    @property
    def ctype(self):
        return _C_TYPES[self.dtype][0]

    @property
    def nbytes(self):
        return math.prod(self.shape) * self.numpy_dtype.itemsize


# The device link has no libm, so ``sqrt``/``sqrtf`` calls in the portable C
# are redirected to these helpers.  ``allo_sqrt`` follows IEEE special cases
# (+-0, +inf, NaN; negative -> NaN), scales subnormals into the normal range,
# starts from the exponent-halved power of two (within a factor of two of the
# root) and runs Newton's iteration until it stops decreasing: after the first
# step the iterate is above the root and decreases monotonically, and
# quadratic convergence reaches double precision in at most six steps, so the
# loop bound of ten is never the stopping reason for finite inputs.  The
# result is within one ulp of the correctly rounded root on the whole double
# range; ``allo_sqrtf`` rounds that to float.
APU_V1_SQRT_HELPERS = """
static double allo_sqrt(double value) {
    union { double f; uint64_t u; } bits;
    if (value != value || value == 0.0) return value;
    if (value < 0.0) { bits.u = UINT64_C(0x7ff8000000000000); return bits.f; }
    bits.f = value;
    if ((bits.u >> 52) == 0x7ffu) return value;
    double scale = 1.0;
    if ((bits.u >> 52) == 0u) {
        value *= 18014398509481984.0; /* 2^54 */
        scale = 7.450580596923828125e-9; /* 2^-27 */
        bits.f = value;
    }
    int64_t exponent = (int64_t)(bits.u >> 52) - 1023;
    int64_t half = exponent >= 0 ? exponent / 2 : -((1 - exponent) / 2);
    bits.u = (uint64_t)(half + 1023) << 52;
    double root = bits.f;
    root = 0.5 * (root + value / root);
    for (int step = 0; step < 10; ++step) {
        double next = 0.5 * (root + value / root);
        if (!(next < root)) break;
        root = next;
    }
    return root * scale;
}
static float allo_sqrtf(float value) {
    return (float)allo_sqrt((double)value);
}
"""


def _replace_local_arrays(source: str) -> tuple[str, int]:
    """Move emitted local arrays from the small ARC stack into L4 scratch."""
    total = 0

    def replace(match):
        nonlocal total
        ctype = match.group("ctype")
        name = match.group("name")
        dims = tuple(int(value) for value in re.findall(r"\d+", match.group("dims")))
        itemsize = np.dtype(
            {
                "bool": np.bool_,
                "int8_t": np.int8,
                "uint8_t": np.uint8,
                "int16_t": np.int16,
                "uint16_t": np.uint16,
                "int32_t": np.int32,
                "uint32_t": np.uint32,
                "int64_t": np.int64,
                "uint64_t": np.uint64,
                "float": np.float32,
                "double": np.float64,
            }[ctype]
        ).itemsize
        nbytes = math.prod(dims) * itemsize
        total += (nbytes + 7) & ~7
        indent = match.group("indent")
        if len(dims) == 1:
            declaration = f"{ctype} *{name}"
        else:
            tail = "".join(f"[{extent}]" for extent in dims[1:])
            declaration = f"{ctype} (*{name}){tail}"
        return (
            f"{indent}{declaration} = ({ctype} (*)"
            f"{''.join(f'[{extent}]' for extent in dims[1:]) if len(dims) > 1 else ''})"
            f"allo_l4_alloc({nbytes});{match.group('comment')}"
            if len(dims) > 1
            else f"{indent}{declaration} = ({ctype} *)allo_l4_alloc({nbytes});"
            f"{match.group('comment')}"
        )

    transformed = _LOCAL_ARRAY.sub(replace, source)
    transformed = re.sub(r"\bsqrtf\s*\(", "allo_sqrtf(", transformed)
    transformed = re.sub(r"\bsqrt\s*\(", "allo_sqrt(", transformed)
    allocator = (
        """
static uint8_t *allo_l4_cursor;
static void *allo_l4_alloc(size_t bytes) {
    uintptr_t cursor = ((uintptr_t)allo_l4_cursor + 7u) & ~(uintptr_t)7u;
    allo_l4_cursor = (uint8_t *)(cursor + bytes);
    return (void *)cursor;
}
"""
        + APU_V1_SQRT_HELPERS
    )
    first_function = transformed.find("void ")
    if first_function < 0:
        raise RuntimeError("portable C source contains no function")
    transformed = (
        transformed[:first_function] + allocator + transformed[first_function:]
    )
    return transformed, max(8, total)


def _pointer_cast(argument: _Argument, pointer: str) -> str:
    if not argument.shape:
        raise TypeError("scalar source arguments are not yet supported by APU v1 ABI")
    if len(argument.shape) == 1:
        return f"({argument.ctype} *){pointer}"
    tail = "".join(f"[{extent}]" for extent in argument.shape[1:])
    return f"({argument.ctype} (*){tail}){pointer}"


def _emit_device_source(top_name, source, arguments, outputs, scratch_role):
    source_arguments = [
        argument for argument in arguments if argument.source == "argument"
    ]
    result_arguments = [
        argument for argument in arguments if argument.source == "result"
    ]
    declarations = []
    call_arguments = []
    for argument in source_arguments + result_arguments:
        role = argument.name
        declarations.append(
            f"    uint8_t *raw_{role} = (uint8_t *)gal_mem_handle_to_apu_ptr("
            f"data->mem_hndl_{role});"
        )
        call_arguments.append(_pointer_cast(argument, f"raw_{role}"))
    declarations.append(
        f"    allo_l4_cursor = (uint8_t *)gal_mem_handle_to_apu_ptr("
        f"data->mem_hndl_{scratch_role});"
    )
    copies = []
    for argument in outputs:
        if argument.source != "argument":
            continue
        alias = f"result_{argument.name}"
        declarations.append(
            f"    uint8_t *raw_{alias} = (uint8_t *)gal_mem_handle_to_apu_ptr("
            f"data->mem_hndl_{alias});"
        )
        copies.append(
            f"    for (size_t i = 0; i < {argument.nbytes}; ++i) "
            f"raw_{alias}[i] = raw_{argument.name}[i];"
        )
    body = "\n".join(declarations)
    copy_body = "\n".join(copies)
    return (
        "#include <gsi/libsys.h>\n#include <gsi/libgal.h>\n"
        '#include "struct.h"\n#include <gsi_device_profiling.h>\n\n'
        "PROF_VAR(total);\n"
        + source
        + "\nstatic int my_kernel(struct program_data *data) {\n"
        "    arc_counters_init();\n    PROF_INIT(total);\n"
        + body
        + "\n    PROF_START(total);\n    "
        + top_name
        + "("
        + ", ".join(call_arguments)
        + ");\n"
        + copy_body
        + "\n    PROF_END(total);\n    PROF_PRINT(total);\n    return 0;\n}\n\n"
        "GAL_TASK_ENTRY_POINT(apu_kernel_task, in, out) {\n"
        "    struct program_cmd *cmd = (struct program_cmd *)in;\n"
        "    return my_kernel(&cmd->data);\n}\n"
    )


def _generate_project(
    dst,
    device_source,
    inputs,
    output_specs,
    lab_name,
    *,
    host_source=None,
    struct_source=None,
    initial_outputs=(),
):
    from ..spmw_apu_v1_build import (
        _emit_host_c,
        _emit_makefile,
        _emit_struct_h,
        stage_apu_v1_template,
    )

    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    stage_apu_v1_template(dst)
    in_roles = sorted(inputs)
    out_roles = sorted(output_specs)
    input_sizes = {name: value.nbytes for name, value in inputs.items()}
    output_sizes = {
        name: math.prod(shape) * np.dtype(dtype).itemsize
        for name, (shape, dtype) in output_specs.items()
    }
    if host_source is None:
        host = _emit_host_c(
            in_roles,
            out_roles,
            input_sizes,
            output_sizes,
            initial_output_roles=tuple(sorted(initial_outputs)),
        )
        # The explicit scalar baseline is deliberately single-core until a
        # phase carries a proven partition plus cross-APUC barrier plan.
        host = host.replace("enum { NUM_APUC = 4 };", "enum { NUM_APUC = 1 };")
    else:
        host = str(host_source)
    (dst / "Makefile").write_text(_emit_makefile(lab_name))
    (dst / "struct.h").write_text(
        _emit_struct_h(in_roles, out_roles)
        if struct_source is None
        else str(struct_source)
    )
    (dst / "device.c").write_text(device_source)
    (dst / "host.c").write_text(host)
    return dst


_APU_V1_LOCK_ENV = "TENON_APU_V1_LOCK"
_board_lock_guard = threading.RLock()
_board_lock_state = {"depth": 0, "fd": None, "path": None}


def apu_v1_lock_path() -> Path:
    """Board lock file: ``$TENON_APU_V1_LOCK`` or one per PCI device node."""

    override = os.environ.get(_APU_V1_LOCK_ENV)
    if override:
        return Path(override)
    from ..spmw_simenv import apu_v1_pci_node

    node = apu_v1_pci_node()
    identity = "unknown" if node is None else re.sub(r"[^\w.:-]", "_", node.name)
    # A fixed directory (not $TMPDIR) so every user and process agrees.
    return Path("/tmp") / f"tenon-apu-v1-{identity}.lock"


def _open_lock_file(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    except PermissionError:
        # Another user created it; flock works on a read-only descriptor.
        return os.open(path, os.O_RDONLY)
    with contextlib.suppress(OSError):
        os.fchmod(fd, 0o666)
    return fd


@contextlib.contextmanager
def apu_v1_board_lock():
    """Exclusive advisory lock serializing every APU v1 board launch.

    Reentrant within one process (the pytest fixture and the runtime both
    take it); other threads of the process wait, other processes block on
    ``flock``.
    """

    with _board_lock_guard:
        state = _board_lock_state
        if state["depth"] == 0:
            path = apu_v1_lock_path()
            fd = _open_lock_file(path)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
            except BaseException:
                os.close(fd)
                raise
            state.update(fd=fd, path=path)
        state["depth"] += 1
        try:
            yield state["path"]
        finally:
            state["depth"] -= 1
            if state["depth"] == 0:
                fd = state["fd"]
                state.update(fd=None, path=None)
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    os.close(fd)


def _ledag_available_reason() -> str | None:
    if shutil.which("ledag-ssh") is None:
        return "ledag-ssh is not on PATH, so the device profile cannot be read"
    return None


# The documented interactive ``scp d2h`` command copies the firmware's ten
# numbered rotation slots directly to host storage. No ``flo`` download is
# made. The vendor CLI still writes a small session log in /tmp; its
# PyInstaller extraction and the retained copies are independently located.
_LEDAG_SCRATCH = Path("/tmp")
_LEDAG_LOG_REQUIRED_BYTES = 64 * 1024  # diagnostic session: 1,983 bytes; reserve
_LEDAG_EXTRACTION_REQUIRED_BYTES = 16 * 1024 * 1024  # measured bundle: 11.5 MB
_LEDAG_COPY_REQUIRED_BYTES = 64 * 1024 * 1024  # measured ten-slot snapshot: 34 MB
_LEDAG_CARD_LOG_ROOT = "/run/media/mmcblk0p2/logs"
_LEDAG_COPY_TIMEOUT_SECONDS = 300
_LEDAG_LOG_SLOTS = range(10)  # vendor firmware rotation format, not active slot
_LEDAG_TIMESTAMP = re.compile(
    r"TIME STAMP:\s*(\d{2})/(\d{2})/(\d{2})\s+"
    r"(\d{2}):(\d{2}):(\d{2})\.(\d{1,9})\s+UTC"
)
_LEDAG_NON_TEXT = bytes(
    b for b in range(256) if not (32 <= b < 127 or b in (9, 10, 13))
)


def _ledag_extraction_dir() -> Path:
    configured = os.environ.get("TMPDIR")
    return (Path(configured) if configured else Path(tempfile.gettempdir())).resolve()


def ledag_profile_dir() -> Path:
    """Explicit retained profile storage; never accumulate raw logs in /tmp."""
    configured = os.environ.get("TENON_APU_V1_PROFILE_DIR")
    if not configured:
        raise ValueError(
            "set TENON_APU_V1_PROFILE_DIR to a provisioned retained-evidence "
            "directory writable by the vendor daemon before reading board profiles"
        )
    return Path(configured).resolve()


def _ledag_setting(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        result = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _same_filesystem(first: Path, second: Path) -> bool:
    return first.stat().st_dev == second.stat().st_dev


def _mib(value: int) -> str:
    return f"{value / 1024 / 1024:.3f} MiB"


def ledag_scratch_headroom_problem() -> str | None:
    """Check actual session-log, extraction and direct-copy filesystems.

    Requirements sharing a filesystem are added, not checked independently.
    These are per-read reserves, not a campaign budget or a disk reservation.
    A transfer can still fail (e.g. concurrent consumption); it never yields
    timing on failure. Provision larger retained-evidence budgets separately.
    """
    try:
        profile_root = ledag_profile_dir()
        copy_bytes = _ledag_setting(
            "TENON_APU_V1_PROFILE_COPY_BYTES", _LEDAG_COPY_REQUIRED_BYTES
        )
        _ledag_setting(
            "TENON_APU_V1_PROFILE_TIMEOUT_SECONDS", _LEDAG_COPY_TIMEOUT_SECONDS
        )
    except ValueError as error:
        return str(error)
    needs = [
        (_LEDAG_SCRATCH, _LEDAG_LOG_REQUIRED_BYTES, "vendor session log"),
        (
            _ledag_extraction_dir(),
            _LEDAG_EXTRACTION_REQUIRED_BYTES,
            "bundle extraction (TMPDIR)",
        ),
        (profile_root, copy_bytes, "direct-copy firmware logs"),
    ]
    grouped = []
    try:
        for directory, required, what in needs:
            if not directory.is_dir():
                return f"profile storage directory does not exist: {directory}"
            for group in grouped:
                if _same_filesystem(directory, group[0]):
                    group[1] += required
                    group[2] += f", {what} at {directory}"
                    break
            else:
                grouped.append([directory, required, f"{what} at {directory}"])
        for directory, required, what in grouped:
            free = shutil.disk_usage(directory).free
            if free < required:
                return (
                    f"{directory} has {_mib(free)} free; a ledag-ssh read needs "
                    f"{_mib(required)} there for {what}; provision these actual "
                    "directories (TMPDIR and TENON_APU_V1_PROFILE_DIR are configurable)"
                )
    except OSError as error:
        return f"cannot check profile storage: {error}"
    return None


def _ledag_owned_session_logs(session_id: int) -> dict:
    """Identify live CLI logs by kernel session ID, not a before/after glob.

    PyInstaller forks: the log's PID can be the child, not Popen.pid. Only a
    live member of our new process session proves ownership. Unknown files
    are left alone, including simultaneous tools run by this same user.
    """
    owned = {}
    for path in _LEDAG_SCRATCH.glob("ledag_ssh_*_log.txt"):
        match = re.search(r"_(\d+)_log\.txt$", path.name)
        if match is None:
            continue
        try:
            pid = int(match.group(1))
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                continue
            if os.getsid(pid) == session_id:
                owned[path] = (info.st_dev, info.st_ino)
        except (OSError, ValueError):
            continue
    return owned


def _ledag_copy_session_logs(owned: dict, evidence: Path) -> None:
    """Retain then unlink only identified files, checking identity again."""
    for path, identity in owned.items():
        try:
            info = path.lstat()
            if (info.st_dev, info.st_ino) != identity or not stat.S_ISREG(info.st_mode):
                continue
            shutil.copyfile(path, evidence / path.name)
            # Retention must succeed before reclaiming the residual /tmp file.
            if (path.lstat().st_dev, path.lstat().st_ino) == identity:
                path.unlink()
        except OSError:
            # Keep the original if it cannot be retained; the next guard still
            # accounts for any resulting loss of /tmp headroom.
            continue


def _ledag_copy_logs(evidence: Path) -> dict:
    commands = (
        "".join(
            f"scp d2h {_LEDAG_CARD_LOG_ROOT}/{slot} {evidence / str(slot)}\n"
            for slot in _LEDAG_LOG_SLOTS
        )
        + "quit\n"
    )
    (evidence / "commands.txt").write_text(commands)
    owned = {}
    stdout, stderr = b"", b""
    process = None
    timeout = _ledag_setting(
        "TENON_APU_V1_PROFILE_TIMEOUT_SECONDS", _LEDAG_COPY_TIMEOUT_SECONDS
    )
    deadline = time.monotonic() + timeout
    try:
        process = subprocess.Popen(
            ["ledag-ssh", "-o", "localhost"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(os.environ, TMPDIR=str(_ledag_extraction_dir())),
            cwd=evidence,
            start_new_session=True,
        )
        pending_input = commands.encode()
        while True:
            owned.update(_ledag_owned_session_logs(process.pid))
            try:
                stdout, stderr = process.communicate(input=pending_input, timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                pending_input = None
                if time.monotonic() >= deadline:
                    # Stop the CLI and PyInstaller child together. Preserve
                    # partial destinations; this read cannot yield timing.
                    os.killpg(process.pid, signal.SIGKILL)
                    stdout, stderr = process.communicate()
                    raise RuntimeError(
                        f"ledag-ssh profile copy timed out after {timeout} seconds"
                    )
        if process.returncode:
            raise RuntimeError(f"ledag-ssh exited with {process.returncode}")
        if stdout.count(b"Download Success") != len(_LEDAG_LOG_SLOTS):
            raise RuntimeError("ledag-ssh did not confirm every numbered log copy")
        transcript = stdout.decode(errors="replace")
        cli_version = re.search(r"^Version:\s*(\S+)", transcript, re.MULTILINE)
        firmware_versions = sorted(
            set(re.findall(r"'Version':\s*'([^']+)'", transcript))
        )
        return {
            "returncode": process.returncode,
            "pid": process.pid,
            "ledag_ssh_version": cli_version.group(1) if cli_version else None,
            "firmware_versions": firmware_versions,
            "timeout_seconds": timeout,
        }
    finally:
        if process is not None and process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        # Preserve the partial transcript even on timeout or failed transfer.
        (evidence / "stdout.txt").write_bytes(stdout)
        (evidence / "stderr.txt").write_bytes(stderr)
        _ledag_copy_session_logs(owned, evidence)


def _ledag_order_logs(evidence: Path) -> tuple[str, list]:
    logs = []
    for slot in _LEDAG_LOG_SLOTS:
        path = evidence / str(slot)
        raw = path.read_bytes()  # missing copies fail; stdout alone is insufficient
        text = raw.translate(None, _LEDAG_NON_TEXT).decode("ascii")
        stamps = [
            (
                int(y),
                int(mo),
                int(day),
                int(h),
                int(mi),
                int(s),
                int(frac.ljust(9, "0")),
            )
            for mo, day, y, h, mi, s, frac in _LEDAG_TIMESTAMP.findall(text)
        ]
        if not stamps or any(a > b for a, b in zip(stamps, stamps[1:])):
            raise RuntimeError(
                f"firmware log slot {slot} has missing or nonmonotonic timestamps"
            )
        logs.append(
            (
                stamps[0],
                stamps[-1],
                slot,
                text,
                {
                    "slot": slot,
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "first_timestamp": stamps[0],
                    "last_timestamp": stamps[-1],
                },
            )
        )
    logs.sort(key=lambda item: item[0])
    if any(a[1] >= b[0] for a, b in zip(logs, logs[1:])):
        raise RuntimeError(
            "firmware rotation slots overlap or have ambiguous chronology"
        )
    return "\n".join(item[3] for item in logs), [item[4] for item in logs]


def _ledag_drain() -> str:
    """Read all firmware rotation slots with supported direct-copy transport.

    Raw files, commands, CLI output and the timestamp/hash manifest survive
    failures. The launch window's outer lock remains held across both reads;
    taking it here too also serializes independent diagnostic reads.
    """
    with apu_v1_board_lock():
        problem = ledag_scratch_headroom_problem()
        if problem is not None:
            raise RuntimeError(f"ledag-ssh cannot read the device log: {problem}")
        # The vendor command parser does not document escaping/quoting. Refuse
        # paths it could tokenize as multiple arguments before starting it.
        root = ledag_profile_dir()
        if re.search(r"[^A-Za-z0-9_./-]", str(root)):
            raise RuntimeError(
                "TENON_APU_V1_PROFILE_DIR must use only letters, digits, _, ., / or -"
            )
        evidence = Path(tempfile.mkdtemp(prefix="tenon-profile-", dir=root))
        started = time.monotonic()
        manifest = {
            "transport": "ledag-ssh scp d2h",
            "evidence": str(evidence),
            "extraction": str(_ledag_extraction_dir()),
            "board_lock": str(apu_v1_lock_path()),
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "status": "incomplete",
        }
        try:
            manifest.update(_ledag_copy_logs(evidence))
            text, files = _ledag_order_logs(evidence)
            manifest.update(status="copied", files=files)
            return f"Tenon profile copy evidence: {evidence}\n" + text
        except Exception as error:
            manifest["error"] = f"{type(error).__name__}: {error}"
            raise RuntimeError(
                f"ledag-ssh profile read failed; evidence {evidence}: {error}"
            ) from error
        finally:
            manifest["elapsed_seconds"] = time.monotonic() - started
            (evidence / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )


_TOTAL_RECORD = re.compile(r"\btotal\b[^\n]*?\bcrun\s*[:=]\s*(\d+)[^\n]*")
_HITS_FIELD = re.compile(r"\bhits\s*[:=]\s*(\d+)")


def fresh_profile_records(before: str, after: str) -> list:
    """The ``total`` PROF_PRINT records of ``after`` that are not in ``before``.

    ``before`` is the device log read immediately before the launch and
    ``after`` the log read after it.  The log is append-only (older records
    may age out at the front), so the fresh records are those following the
    last record of ``before`` within ``after``.  The anchor is the text of the
    last three ``before`` records; if it cannot be found in ``after`` the log
    rotated past the launch window and the measurement is not attributable.
    """

    after_records = list(_TOTAL_RECORD.finditer(after))
    before_records = list(_TOTAL_RECORD.finditer(before))
    if not before_records:
        return after_records
    anchor = before[
        before_records[max(0, len(before_records) - 3)]
        .start() : before_records[-1]
        .end()
    ]
    position = after.rfind(anchor)
    if position < 0:
        raise RuntimeError(
            "APU v1 profile freshness check failed: the device log read before the "
            "launch is not a prefix of the log read after it, so this launch's "
            "PROF_PRINT records cannot be told from older ones"
        )
    boundary = position + len(anchor)
    return [record for record in after_records if record.start() >= boundary]


def parse_fresh_profile(
    before: str, after: str, expected_records: int
) -> tuple[int, ...]:
    """Return the CRUN of exactly ``expected_records`` fresh ``total`` records.

    Anything but exactly the expected number of records appended by the
    launch, each a single measurement (``hits`` 1) with a positive CRUN, is a
    stale, missing, or ambiguous profile and raises.
    """

    records = fresh_profile_records(before, after)
    if len(records) != expected_records:
        raise RuntimeError(
            f"APU v1 profile freshness check failed: expected {expected_records} "
            f"fresh 'total' PROF_PRINT record(s) after the launch, found "
            f"{len(records)}; the cycle count would be stale or missing"
        )
    cycles = []
    for record in records:
        hits = _HITS_FIELD.search(record.group(0))
        crun = int(record.group(1))
        if (hits is not None and int(hits.group(1)) != 1) or crun <= 0:
            raise RuntimeError(
                f"APU v1 profile record is not one fresh measurement: {record.group(0)!r}"
            )
        cycles.append(crun)
    return tuple(cycles)


_COUNTER_RE = re.compile(
    r"(?<![A-Za-z0-9_])(hits|seu|crun|iall|icm|dcm)\s*[:=]\s*([0-9]+)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)


def profile_counters(profile: str) -> dict:
    """The counters (``hits``, ``seu``, ``crun``, ``iall``, ``icm``, ``dcm``)
    of the last ``total`` PROF_PRINT record in ``profile``; empty when the
    record carries none."""
    records = [match.group(0) for match in _TOTAL_RECORD.finditer(profile)] or [profile]
    return {key.lower(): int(value) for key, value in _COUNTER_RE.findall(records[-1])}


class APUV1LaunchFailure(RuntimeError):
    """A board launch that produced no attributable measurement.

    ``diagnostics`` is the structured evidence of the failed launch, gathered
    before raising (ISSUES V1-21): the argv and working directory, the
    binary's identity, every attempt's exit status and process output, the
    device log read before the launch and every read after it, the number of
    fresh ``total`` records found against the number expected, the board lock
    and the time.  The runtimes add the generated source and the launch's
    roles; a measurement driver serializes it beside the launch.  No cycle
    count is ever derived from such a launch.
    """

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = dict(diagnostics)


def _binary_identity(argv):
    path = Path(argv[0]) if argv else None
    if path is None or not path.is_file():
        return {
            "binary_path": None if path is None else str(path),
            "binary_sha256": None,
            "binary_bytes": None,
        }
    return {
        "binary_path": str(path),
        "binary_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "binary_bytes": path.stat().st_size,
    }


def launch_on_board(argv, cwd, *, expected_records, timeout, attempts=3):
    """Run one built APU v1 binary under the board lock with a fresh profile.

    Returns ``(run text, profile text, crun tuple)``.  The device log is read
    before every attempt and after the run; :func:`parse_fresh_profile`
    proves that exactly the expected PROF_PRINT records were appended by this
    launch.  Every way the launch can fail to yield an attributable
    measurement raises :class:`APUV1LaunchFailure` carrying the launch's
    diagnostics (argv, binary identity, every attempt's exit status and
    output including a timed-out process's partial output, every device-log
    read before and after, the record counts, and an explicit ``reason``:
    ``device_run_failed``, ``timeout``, ``device_log_read_failed``,
    ``log_rotated``, ``missing_or_stale_profile``, ``malformed_profile_record``
    or ``unexpected``), so that nothing about the failure is lost when the
    caller discards its build directory.  No cycle count is produced for a
    failed launch (ISSUES V1-21).
    """

    argv_text = [str(a) for a in argv]
    diagnostics = {
        "argv": argv_text,
        "cwd": str(cwd),
        **_binary_identity(argv_text),
        "board_lock": str(apu_v1_lock_path()),
        "expected_records": int(expected_records),
        "attempts": [],
        # every read before an attempt (one per retry) and the last one
        "device_log_before_reads": [],
        "device_log_before": None,
        "device_log_reads_after": [],
        "fresh_records_found": None,
        "reason": None,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    def fail(reason, message, cause=None):
        diagnostics["reason"] = reason
        failure = APUV1LaunchFailure(message, diagnostics)
        if cause is not None:
            raise failure from cause
        raise failure

    def decode(data):
        if data is None:
            return ""
        return data.decode(errors="replace") if isinstance(data, bytes) else str(data)

    process = None
    text = ""
    before = ""
    with apu_v1_board_lock():
        try:
            for attempt in range(attempts):
                try:
                    before = _ledag_drain()
                except Exception as error:  # the vendor logging tool failed
                    fail(
                        "device_log_read_failed",
                        f"APU v1 device log could not be read before the launch: {error}",
                        error,
                    )
                diagnostics["device_log_before_reads"].append(before)
                diagnostics["device_log_before"] = before
                try:
                    process = subprocess.run(
                        argv, cwd=cwd, capture_output=True, timeout=timeout, check=False
                    )
                except subprocess.TimeoutExpired as error:
                    partial = decode(error.stdout) + decode(error.stderr)
                    diagnostics["attempts"].append(
                        {
                            "returncode": None,
                            "timed_out_after_seconds": timeout,
                            "process_output": partial,
                        }
                    )
                    fail(
                        "timeout",
                        f"APU v1 device run timed out after {timeout} s:\n{partial[-6000:]}",
                        error,
                    )
                text = decode(process.stdout) + decode(process.stderr)
                diagnostics["attempts"].append(
                    {"returncode": process.returncode, "process_output": text}
                )
                if process.returncode == 0 or "no valid context" not in text:
                    break
                time.sleep(attempt + 1)
            if process is None or process.returncode:
                code = None if process is None else process.returncode
                fail(
                    "device_run_failed",
                    f"APU v1 device run failed ({code}):\n{text[-6000:]}",
                )
            profile = ""
            for _ in range(4):
                time.sleep(0.75)
                try:
                    profile = _ledag_drain()
                except Exception as error:
                    fail(
                        "device_log_read_failed",
                        f"APU v1 device log could not be read after the launch: {error}",
                        error,
                    )
                diagnostics["device_log_reads_after"].append(profile)
                try:
                    found = len(fresh_profile_records(before, profile))
                except RuntimeError as error:  # the log rotated past the launch
                    diagnostics["fresh_records_found"] = "unattributable (log rotated)"
                    fail("log_rotated", str(error), error)
                diagnostics["fresh_records_found"] = found
                if found >= expected_records:
                    break
            try:
                cycles = parse_fresh_profile(before, profile, expected_records)
            except RuntimeError as error:
                reason = (
                    "malformed_profile_record"
                    if "not one fresh measurement" in str(error)
                    else "missing_or_stale_profile"
                )
                fail(reason, str(error), error)
        except APUV1LaunchFailure:
            raise
        except Exception as error:  # anything else inside the launch window
            fail(
                "unexpected",
                f"APU v1 launch failed: {type(error).__name__}: {error}",
                error,
            )
    return text, profile, cycles


def read_output_image(path, role: str, dtype, shape) -> np.ndarray:
    """Read one device output file, naming the role on a missing/short image."""

    path = Path(path)
    dtype = np.dtype(dtype)
    expected = math.prod(shape) * dtype.itemsize
    if not path.is_file():
        raise RuntimeError(
            f"APU v1 output {role!r}: the device run wrote no file {path} "
            f"(expected {expected} bytes for {tuple(shape)} {dtype})"
        )
    actual = path.stat().st_size
    if actual != expected:
        raise RuntimeError(
            f"APU v1 output {role!r}: {path} holds {actual} bytes, expected "
            f"{expected} bytes for {tuple(shape)} {dtype}"
        )
    return np.fromfile(path, dtype=dtype).reshape(shape)


_APU_V1_PROGRAM_BACKENDS = (None, "virtual", "functional")


def customize_apu_v1_phase(phase):
    """The retained MLIR schedule of one phase (``enable_tensor=False``,
    the phase's ``instantiate`` list)."""
    kwargs = {"enable_tensor": False}
    if phase.instantiate:
        kwargs["instantiate"] = list(phase.instantiate)
    return customize(phase.kernel, **kwargs)


class CompiledAPUv1Program:
    """One single-phase APUv1Program lowered to scalar ARC C or a native recipe.

    ``backend`` selects what :meth:`run` does:

    * ``None``: build and launch on the APU v1 board (``kind="device"``).
    * ``"virtual"``: evaluate the retained cost graph; the device is neither
      probed nor launched and no array is written (``kind="virtual"``).
    * ``"functional"``: compile the retained MLIR-derived portable C for the
      host and execute it (``kind="functional"``, ``cycles=None``). The scalar
      ARC device program is that same C, so this executes the lowered
      program. Native-vector programs run a GVML recipe that has no host
      execution path, so they reject ``"functional"`` at compile time.
    """

    def __init__(self, program, target, cost=None, backend=None, *, schedule=None):
        if backend not in _APU_V1_PROGRAM_BACKENDS:
            raise ValueError(
                "APUv1Program supports backend=None (device), 'virtual', or "
                f"'functional'; got {backend!r}"
            )
        self.program = program
        self.target = target
        self.cost = cost
        self.backend = backend
        self._host_module = None
        if backend == "virtual" and cost is None:
            raise ValueError(
                "APUv1Program backend='virtual' evaluates the retained cost graph "
                "and needs a cost spec (e.g. allo.pim.costs.apu_v1_cost)"
            )
        if len(program.phases) != 1:
            raise RuntimeError(
                "multi-phase APUv1Program requires the hybrid orchestration backend"
            )
        phase = program.phases[0]
        if schedule is None:
            # The shared front stages normally customize the phase once
            # (`apu_v1_record_frontend.normalize_apu_v1_program`) and hand
            # the retained schedule here, so the lowering and the semantic
            # facts come from the same IR.
            schedule = customize_apu_v1_phase(phase)
        self.schedule = schedule
        self.artifact = emit_c_from_mlir(
            self.schedule.module,
            self.schedule.top_func_name,
            wrap_wide_integers=True,
        )
        source_names = tuple(inspect.signature(phase.kernel).parameters)
        source_abi = [
            arg for arg in self.artifact.arguments if arg.source == "argument"
        ]
        result_abi = [arg for arg in self.artifact.arguments if arg.source == "result"]
        if len(source_names) != len(source_abi):
            raise ValueError("APU v1 source/MLIR ABI argument count mismatch")
        if len(phase.result_names) != len(result_abi):
            raise ValueError("APU v1 result_names do not match returned MLIR memrefs")
        self.arguments = tuple(
            _Argument(name, arg.dtype, tuple(arg.shape), arg.mode, arg.source)
            for name, arg in zip(source_names, source_abi)
        ) + tuple(
            _Argument(name, arg.dtype, tuple(arg.shape), arg.mode, arg.source)
            for name, arg in zip(phase.result_names, result_abi)
        )
        self.source_arguments = tuple(
            argument for argument in self.arguments if argument.source == "argument"
        )
        self.outputs = tuple(
            argument
            for argument in self.arguments
            if argument.source == "result" or argument.mode in {"out", "both"}
        )
        from .apu_v1_native_vector import (
            discover_apu_v1_native_vector_lowering,
        )

        # A required-vector phase either proves one native recipe or raises
        # NoNativeVectorRouteError naming each recipe's mismatch.
        self.native_vector_lowering = discover_apu_v1_native_vector_lowering(
            phase,
            self.artifact,
            self.arguments,
            self.schedule.top_func_name,
            target,
        )
        # Native recipes carry their own complete structural proof and source
        # lowering.  Discover the ordinary scalar ABI manifest without asking
        # the contraction-only hybrid discoverer to prove the same phase.
        manifest_phase = (
            replace(phase, vectorize=False)
            if self.native_vector_lowering is not None
            else phase
        )
        self.hybrid_manifest = discover_apu_v1_hybrid_manifest(
            manifest_phase,
            self.schedule,
            self.artifact,
            target,
            cost=cost,
            program_name=program.name,
        )
        if self.hybrid_manifest.has_vector_regions:
            raise NotImplementedError(
                "APUv1Program vector regions (hybrid scalar/vector execution) were "
                "removed; compile contractions as a plain callable through allo.compile "
                "to use the APU v1 vector-program route"
            )
        transformed, scalar_scratch_bytes = _replace_local_arrays(
            self.artifact.c_source
        )
        self.scratch_role = "allo_scratch"
        if self.native_vector_lowering is None:
            self.scratch_bytes = scalar_scratch_bytes
            self.device_source = _emit_device_source(
                self.schedule.top_func_name,
                transformed,
                self.arguments,
                self.outputs,
                self.scratch_role,
            )
        else:
            self.scratch_bytes = self.native_vector_lowering.scratch_bytes
            self.device_source = self.native_vector_lowering.emit_device_source()
        self.execution_graph = ExecutionGraph(program.name)
        self._graph_fingerprint = None
        if cost is not None:
            self.execution_graph = self.execution_graph_for(cost)
        self.check_backend(backend)

    @property
    def route(self) -> str:
        """``"scalar_arc_c"`` or the proven native recipe's route."""
        if self.native_vector_lowering is None:
            return "scalar_arc_c"
        return self.native_vector_lowering.route

    def check_backend(self, backend) -> None:
        """Refuse a backend this lowering cannot serve: anything but the
        device, the virtual runner or the host functional C, and
        ``"functional"`` for a native recipe (no host execution path)."""
        if backend not in (None, "device", "virtual", "functional"):
            raise ValueError(
                "APUv1Program supports backend=None (device), 'virtual', or "
                f"'functional'; got {backend!r}"
            )
        if backend == "functional" and self.native_vector_lowering is not None:
            raise NotImplementedError(
                f"APUv1Program backend='functional' executes the scalar ARC C on "
                f"the host; {self.program.name!r} lowers to the native vector route "
                f"{self.native_vector_lowering.route!r}, whose GVML program has no "
                "host execution path. Use backend=None (device) or 'virtual'"
            )

    def execution_graph_for(self, cost):
        """The retained cost graph of this lowering under ``cost``: the
        native recipe's structural graph, or the scalar C instruction count
        of the retained MLIR priced as one ``SCALAR_C`` event.  Cached per
        cost fingerprint and kept as ``execution_graph``."""
        fingerprint = getattr(cost, "fingerprint", None)
        if (
            self._graph_fingerprint is not None
            and self._graph_fingerprint == (id(cost), fingerprint)
            and self.execution_graph is not None
        ):
            return self.execution_graph
        target = self.target
        graph = ExecutionGraph(self.program.name)
        if self.native_vector_lowering is not None and hasattr(
            self.native_vector_lowering, "build_execution_graph"
        ):
            graph = self.native_vector_lowering.build_execution_graph(target, cost)
        else:
            fallback = max(
                (extent for argument in self.arguments for extent in argument.shape),
                default=1,
            )
            summary = analyze_upmem_mlir(
                self.artifact.source_mlir, dynamic_trip_count=fallback
            )
            event = CostEvent.create(
                "scalar_c",
                target.op("SCALAR_C"),
                metrics={"instructions": summary.total_instruction_count},
                attributes={"source": "retained_mlir"},
            )
            cost.emit(graph, event)
        if "cost_validity" not in graph.metadata:
            from .costs.apu_v1 import apu_v1_cost_validity

            validity = apu_v1_cost_validity(
                self.route, graph.metadata.get("operation_inventory"), cost
            )
            graph.metadata["cost_validity"] = validity
            graph.metadata["estimate_validity"] = validity["cost_validity"]
        graph.metadata.setdefault("launch_count", 1)
        graph.metadata.setdefault("plan", self.route)
        self.execution_graph = graph
        self.cost = cost
        self._graph_fingerprint = (id(cost), fingerprint)
        return graph

    def _checked_sources(self, inputs):
        normalized = {}
        for argument in self.source_arguments:
            value = inputs[argument.name]
            if not isinstance(value, np.ndarray):
                raise TypeError(f"{argument.name} must be a NumPy array")
            if value.shape != argument.shape or value.dtype != argument.numpy_dtype:
                raise TypeError(
                    f"{argument.name} must be {argument.numpy_dtype}{argument.shape}, "
                    f"got {value.dtype}{value.shape}"
                )
            normalized[argument.name] = np.ascontiguousarray(value)
        return normalized

    def _run_virtual(self, inputs):
        self._checked_sources(inputs)
        estimate = self.cost.evaluate(self.execution_graph)
        spec = getattr(self.cost, "spec", self.cost)
        return RunResult(
            estimate.cycles,
            "virtual APUv1Program cost evaluation; device not launched, "
            "nothing executed, outputs not written",
            "virtual",
            kind="virtual",
            extra={
                "critical_path": list(estimate.critical_path),
                "utilization": dict(estimate.utilization),
                "bottlenecks": list(estimate.bottlenecks),
                "model_fingerprint": estimate.model_fingerprint,
                "cost_model": getattr(spec, "name", None),
                "priced_target": self.target.name,
                "cost_validity": self.execution_graph.metadata.get("cost_validity"),
            },
        )

    def _run_functional(self, inputs):
        # Private copies keep the caller's arrays untouched until the outputs
        # are copied back, as on the device path.
        sources = {
            name: np.array(value, order="C", copy=True)
            for name, value in self._checked_sources(inputs).items()
        }
        if self._host_module is None:
            self._host_module = self.artifact.compile()
        returned = self._host_module(
            *(sources[argument.name] for argument in self.source_arguments)
        )
        results = [
            argument for argument in self.arguments if argument.source == "result"
        ]
        if len(results) == 1:
            returned = (returned,)
        returned = dict(zip((argument.name for argument in results), returned or ()))
        outputs = {}
        for argument in self.outputs:
            value = (
                returned[argument.name]
                if argument.source == "result"
                else sources[argument.name]
            )
            value = np.asarray(value, dtype=argument.numpy_dtype).reshape(
                argument.shape
            )
            outputs[argument.name] = value
            if argument.name in inputs:
                np.copyto(inputs[argument.name], value, casting="same_kind")
        return RunResult(
            None,
            "functional APUv1Program: retained MLIR-derived portable C executed "
            "on the host; cycles are not measured",
            "apu_v1-functional",
            kind="functional",
            extra={
                "outputs": outputs,
                "scalar_arc": True,
                "host_library": self._host_module.library_path,
            },
        )

    def run(self, **inputs):
        from ..spmw_codegen import SimulatorUnavailable
        from ..spmw_simenv import (
            apu_v1_unavailable_reason as _apu_v1_unavailable_reason,
        )

        if self.backend == "virtual":
            return self._run_virtual(inputs)
        if self.backend == "functional":
            self.check_backend("functional")
            return self._run_functional(inputs)
        if self.backend not in (None, "device"):
            raise ValueError(
                "APUv1Program supports backend=None (device), 'virtual', or "
                f"'functional'; got {self.backend!r}"
            )
        reason = _apu_v1_unavailable_reason() or _ledag_available_reason()
        if reason:
            raise SimulatorUnavailable("apu_v1", reason)
        normalized = self._checked_sources(inputs)
        normalized[self.scratch_role] = np.zeros(self.scratch_bytes, dtype=np.uint8)
        if self.native_vector_lowering is not None:
            normalized = self.native_vector_lowering.pack_inputs(normalized)
        output_specs = {}
        output_roles = {}
        for argument in self.outputs:
            role = (
                argument.name
                if argument.source == "result"
                else f"result_{argument.name}"
            )
            output_specs[role] = (argument.shape, argument.numpy_dtype)
            output_roles[argument.name] = role

        root = tempfile.mkdtemp(prefix="tenon-apu-v1-scalar-")
        try:
            build_config = _apu_v1_build_config()
            project = _generate_project(
                Path(root) / "project",
                self.device_source,
                normalized,
                output_specs,
                "tenon-scalar",
            )
            input_paths = {}
            for role, value in normalized.items():
                path = Path(root) / f"in_{role}.bin"
                value.tofile(path)
                input_paths[role] = path
            output_paths = {
                role: Path(root) / f"out_{role}.bin" for role in output_specs
            }
            build = subprocess.run(
                build_config.make_command,
                cwd=project,
                capture_output=True,
                timeout=600,
                check=False,
            )
            if build.returncode:
                raise RuntimeError(
                    "APU v1 scalar build failed:\n"
                    + build.stderr.decode(errors="replace")[-6000:]
                )
            binary = build_config.binary_path(project, "tenon-scalar")
            argv = [str(binary)]
            argv.extend(str(input_paths[name]) for name in sorted(input_paths))
            argv.extend(str(output_paths[name]) for name in sorted(output_paths))
            binary_sha256 = hashlib.sha256(Path(binary).read_bytes()).hexdigest()
            try:
                text, profile, cycles = launch_on_board(
                    argv, project, expected_records=1, timeout=900
                )
            except APUV1LaunchFailure as failure:
                # The build directory is removed below; the failure keeps
                # what it would otherwise lose (ISSUES V1-21).
                failure.diagnostics.update(
                    {
                        "device_source": self.device_source,
                        "device_source_sha256": hashlib.sha256(
                            self.device_source.encode()
                        ).hexdigest(),
                        "native_vector_route": (
                            None
                            if self.native_vector_lowering is None
                            else self.native_vector_lowering.route
                        ),
                        "build_mode": build_config.mode,
                        "launch_roles": {
                            "inputs": sorted(input_paths),
                            "outputs": sorted(output_paths),
                        },
                    }
                )
                raise
            log = text + "\n" + profile
            counters = profile_counters(profile)
            outputs = {}
            for argument in self.outputs:
                role = output_roles[argument.name]
                outputs[argument.name] = read_output_image(
                    output_paths[role], role, argument.numpy_dtype, argument.shape
                )
            for argument in self.outputs:
                if argument.name in inputs:
                    np.copyto(
                        inputs[argument.name],
                        outputs[argument.name],
                        casting="same_kind",
                    )
            return RunResult(
                cycles[0],
                log,
                "apu_v1",
                kind="device",
                extra={
                    "outputs": outputs,
                    "scalar_arc": self.native_vector_lowering is None,
                    "native_vector_route": (
                        None
                        if self.native_vector_lowering is None
                        else self.native_vector_lowering.route
                    ),
                    "build_mode": build_config.mode,
                    "device_source": self.device_source,
                    "board_lock": str(apu_v1_lock_path()),
                    "profile_records": 1,
                    # Launch provenance: the binary that ran, the fresh
                    # profiler record's counters and the argument roles.
                    "binary_sha256": binary_sha256,
                    "binary_bytes": Path(binary).stat().st_size,
                    "profile_counters": counters,
                    "profile_text": profile,
                    "launch_roles": {
                        "inputs": sorted(input_paths),
                        "outputs": sorted(output_paths),
                    },
                    "project_dir": (
                        str(project)
                        if os.environ.get("TENON_APU_V1_KEEP_TMP") == "1"
                        else None
                    ),
                },
            )
        finally:
            if os.environ.get("TENON_APU_V1_KEEP_TMP") != "1":
                shutil.rmtree(root, ignore_errors=True)


__all__ = [
    "APUV1LaunchFailure",
    "apu_v1_board_lock",
    "apu_v1_lock_path",
    "launch_on_board",
    "parse_fresh_profile",
    "read_output_image",
    "APUv1Phase",
    "APUv1Program",
    "CompiledAPUv1Program",
    "customize_apu_v1_phase",
]
