# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hardware-only Gemini-II runtime for Tenon's first uint16 VL64 kernel.

The project template is intentionally self-contained: it cross-compiles one
ARC task library, builds one GDML host executable, stages a four-vector pack,
and validates every hardware result against modulo-2^16 addition.  There is no
simulator fallback, so an unavailable SDK or card is reported as a finite
error rather than silently changing the execution backend.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from types import CodeType, FunctionType, ModuleType
from collections.abc import Mapping, Sequence

import numpy as np

from ..spmw_codegen import RunResult


_SHAPE = (4, 65536)
_TEMPLATE = Path(__file__).resolve().parent / "templates" / "apu_g2"
_KEEP_ENV = "TENON_APU_G2_KEEP_TMP"
_BUILD_JOBS_ENV = "TENON_APU_G2_BUILD_JOBS"

_REQUIRED_SDK_PATHS = (
    Path("/opt/gsi/include/64vl/g2_64vl/g2_64vl_types.h"),
    Path("/opt/gsi/g2/vector_core/lib/libg2_64vl.a"),
    Path("/opt/gsi/g2/vector_core/bin/g2_md_update.py"),
    Path("/opt/gsi/lib/libg2_device_core.a"),
)

_TICK_FIELDS = {
    "input_copy": "device_input_copy_ticks",
    "add": "device_add_ticks",
    "output_copy": "device_output_copy_ticks",
    "pipeline": "device_pipeline_ticks",
    "final_pipeline": "device_final_pipeline_ticks",
}
_HOST_FIELDS = ("h2d_us", "host_task_us", "d2h_us", "end_to_end_us")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _code_constant_manifest(value):
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is bytes:
        return {"bytes": value.hex()}
    if type(value) is float:
        return {"float": value.hex()}
    if type(value) is complex:
        return {"complex": [value.real.hex(), value.imag.hex()]}
    if type(value) is tuple:
        return {"tuple": [_code_constant_manifest(item) for item in value]}
    if type(value) is frozenset:
        items = [_code_constant_manifest(item) for item in value]
        return {"frozenset": sorted(items, key=_canonical_json)}
    if value is Ellipsis:
        return {"ellipsis": True}
    if isinstance(value, CodeType):
        return {"code": _code_manifest(value)}
    raise TypeError(f"unsupported executor code constant {type(value).__name__!r}")


def _code_manifest(code: CodeType):
    return {
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "nlocals": code.co_nlocals,
        "stacksize": code.co_stacksize,
        "flags": code.co_flags,
        "code": code.co_code.hex(),
        "constants": [_code_constant_manifest(value) for value in code.co_consts],
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
        "exceptiontable": getattr(code, "co_exceptiontable", b"").hex(),
    }


def _callable_manifest(executor: FunctionType) -> dict[str, str]:
    if not isinstance(executor, FunctionType):
        raise TypeError("APUg2 runtime executor must be a Python function")
    return {
        "module": executor.__module__,
        "qualname": executor.__qualname__,
        "code_sha256": _sha256_bytes(
            _canonical_json(_code_manifest(executor.__code__)).encode("ascii")
        ),
    }


def _module_source_hashes(
    executor: FunctionType,
    dependency_modules: Sequence[ModuleType | str],
) -> tuple[tuple[str, str], ...]:
    names = {__name__, executor.__module__}
    names.update(
        module if isinstance(module, str) else module.__name__
        for module in dependency_modules
    )
    hashes = []
    for name in sorted(names):
        module = importlib.import_module(name)
        source = inspect.getsourcefile(module)
        if source is None:
            raise RuntimeError(f"APUg2 runtime module {name!r} has no source file")
        path = Path(source).resolve()
        if not path.is_file():
            raise RuntimeError(f"APUg2 runtime source is missing: {path}")
        hashes.append((name, _sha256_file(path)))
    return tuple(hashes)


def _promotion_platform_fingerprint():
    """Return no promotion identity until trusted board attestation exists.

    Caller-authored JSON and hashes of caller-selected software files cannot
    prove the identity or current state of the board, driver, or loaded
    firmware. They therefore must not authorize schedule activation.
    """

    return None


def _project_snapshot() -> tuple[tuple[str, bytes], ...]:
    sources, reported_hashes = _source_snapshot()
    if not isinstance(sources, Mapping) or not isinstance(reported_hashes, Mapping):
        raise RuntimeError("APUg2 source snapshot must contain two mappings")
    files = []
    for relative, source in sorted(sources.items()):
        if not isinstance(relative, str) or not relative:
            raise RuntimeError("APUg2 source snapshot contains an invalid path")
        data = source.encode("utf-8") if isinstance(source, str) else source
        if isinstance(data, bytearray):
            data = bytes(data)
        if not isinstance(data, bytes):
            raise RuntimeError(f"APUg2 source {relative!r} is not bytes or text")
        digest = _sha256_bytes(data)
        if reported_hashes.get(relative) != digest:
            raise RuntimeError(f"APUg2 source snapshot hash is stale for {relative!r}")
        files.append((relative, data))
    if set(reported_hashes) != {relative for relative, _data in files}:
        raise RuntimeError("APUg2 source snapshot hash inventory is inconsistent")
    return tuple(files)


def _project_modes(
    files: Sequence[tuple[str, bytes]],
) -> tuple[tuple[str, int], ...]:
    modes = []
    for relative, _data in files:
        path = _TEMPLATE / relative
        if not path.is_file():
            raise RuntimeError(f"APUg2 project source is missing: {path}")
        modes.append((relative, path.stat().st_mode & 0o777))
    return tuple(modes)


@dataclass(frozen=True)
class APUG2RuntimeArtifact:
    """Complete immutable project and launch contract used by one executor."""

    project_files: tuple[tuple[str, bytes], ...]
    project_modes: tuple[tuple[str, int], ...]
    module_source_hashes: tuple[tuple[str, str], ...]
    dependency_modules: tuple[str, ...]
    executor_manifest_json: str
    build_manifest_json: str
    abi_manifest_json: str
    source_hashes: tuple[tuple[str, str], ...]
    source_fingerprint: str
    platform_fingerprint: object | None

    @property
    def executor_identity(self) -> tuple[str, str, str]:
        manifest = json.loads(self.executor_manifest_json)
        return (
            manifest["module"],
            manifest["qualname"],
            manifest["code_sha256"],
        )

    def write_project(self, destination: Path) -> Path:
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        root = destination.resolve()
        for relative, data in self.project_files:
            path = (destination / relative).resolve()
            if root not in path.parents:
                raise RuntimeError(f"unsafe APUg2 project path {relative!r}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            path.chmod(dict(self.project_modes)[relative])
        return destination

    @contextmanager
    def temporary_project(self):
        root = Path(tempfile.mkdtemp(prefix="tenon-apu-g2-frozen-"))
        try:
            yield self.write_project(root / "project")
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def current_platform_fingerprint(self):
        return _promotion_platform_fingerprint()

    def current_source_hashes(
        self, executor: FunctionType
    ) -> tuple[tuple[str, str], ...]:
        current_files = _project_snapshot()
        current_modes = _project_modes(current_files)
        dependencies = tuple(
            importlib.import_module(name) for name in self.dependency_modules
        )
        module_hashes = _module_source_hashes(executor, dependencies)
        executor_json = _canonical_json(_callable_manifest(executor))
        inventory = [
            (f"project/{relative}", _sha256_bytes(data))
            for relative, data in current_files
        ]
        inventory.extend((f"runtime/{name}", digest) for name, digest in module_hashes)
        inventory.extend(
            (
                (
                    "contract/executor.json",
                    _sha256_bytes(executor_json.encode("ascii")),
                ),
                (
                    "contract/build.json",
                    _sha256_bytes(self.build_manifest_json.encode("ascii")),
                ),
                (
                    "contract/abi.json",
                    _sha256_bytes(self.abi_manifest_json.encode("ascii")),
                ),
                (
                    "contract/modes.json",
                    _sha256_bytes(_canonical_json(current_modes).encode("ascii")),
                ),
            )
        )
        return tuple(sorted(inventory))

    def current_source_fingerprint(self, executor: FunctionType) -> str:
        return _sha256_bytes(
            _canonical_json(self.current_source_hashes(executor)).encode("ascii")
        )

    def promotion_source_fingerprint(self, executor: FunctionType) -> str | None:
        try:
            self.assert_current(executor)
        except (OSError, RuntimeError, TypeError, ValueError):
            return None
        return self.source_fingerprint

    def assert_current(self, executor: FunctionType) -> None:
        if self.current_source_fingerprint(executor) != self.source_fingerprint:
            raise RuntimeError("APUg2 runtime source changed after materialization")
        if self.platform_fingerprint != self.current_platform_fingerprint():
            raise RuntimeError("APUg2 platform contract changed after materialization")


def freeze_apu_g2_runtime_artifact(
    executor: FunctionType,
    *,
    build_manifest: Mapping[str, object],
    abi_manifest: Mapping[str, object],
    dependency_modules: Sequence[ModuleType | str] = (),
) -> APUG2RuntimeArtifact:
    """Freeze every source and contract component consumed by ``executor``."""

    files = _project_snapshot()
    modes = _project_modes(files)
    dependency_names = tuple(
        sorted(
            module if isinstance(module, str) else module.__name__
            for module in dependency_modules
        )
    )
    module_hashes = _module_source_hashes(executor, dependency_names)
    executor_json = _canonical_json(_callable_manifest(executor))
    build_json = _canonical_json(dict(build_manifest))
    abi_json = _canonical_json(dict(abi_manifest))
    modes_json = _canonical_json(modes)
    inventory = [
        (f"project/{relative}", _sha256_bytes(data)) for relative, data in files
    ]
    inventory.extend((f"runtime/{name}", digest) for name, digest in module_hashes)
    inventory.extend(
        (
            ("contract/executor.json", _sha256_bytes(executor_json.encode("ascii"))),
            ("contract/build.json", _sha256_bytes(build_json.encode("ascii"))),
            ("contract/abi.json", _sha256_bytes(abi_json.encode("ascii"))),
            ("contract/modes.json", _sha256_bytes(modes_json.encode("ascii"))),
        )
    )
    ordered = tuple(sorted(inventory))
    fingerprint = _sha256_bytes(_canonical_json(ordered).encode("ascii"))
    return APUG2RuntimeArtifact(
        project_files=files,
        project_modes=modes,
        module_source_hashes=module_hashes,
        dependency_modules=dependency_names,
        executor_manifest_json=executor_json,
        build_manifest_json=build_json,
        abi_manifest_json=abi_json,
        source_hashes=ordered,
        source_fingerprint=fingerprint,
        platform_fingerprint=_promotion_platform_fingerprint(),
    )


def _validate_operand(name: str, value) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != np.dtype(np.uint16):
        raise TypeError(f"{name} must have dtype uint16, got {value.dtype}")
    if value.shape != _SHAPE:
        raise ValueError(f"{name} must have shape {_SHAPE}, got {value.shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return value


def _validate_repetitions(repetitions) -> int:
    if isinstance(repetitions, (bool, np.bool_)) or not isinstance(
        repetitions, (int, np.integer)
    ):
        raise TypeError("repetitions must be an integer")
    repetitions = int(repetitions)
    if repetitions <= 0 or repetitions > np.iinfo(np.uint32).max:
        raise ValueError("repetitions must fit a nonzero uint32")
    return repetitions


def _source_snapshot() -> tuple[dict[str, str], dict[str, str]]:
    sources = {}
    hashes = {}
    for path in sorted(item for item in _TEMPLATE.rglob("*") if item.is_file()):
        relative = str(path.relative_to(_TEMPLATE))
        data = path.read_bytes()
        sources[relative] = data.decode("utf-8")
        hashes[relative] = hashlib.sha256(data).hexdigest()
    return sources, hashes


def _require_hardware_stack() -> str:
    if not _TEMPLATE.is_dir():
        raise RuntimeError(f"APUg2 project template is missing: {_TEMPLATE}")
    missing = [str(path) for path in _REQUIRED_SDK_PATHS if not path.is_file()]
    if missing:
        raise RuntimeError("APUg2 SDK is incomplete; missing: " + ", ".join(missing))
    tool = shutil.which("gsi_tool")
    if tool is None:
        raise RuntimeError("gsi_tool is unavailable; cannot verify real-card ownership")
    status = subprocess.run(
        [tool, "info", "apu-00", "-v"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    text = status.stdout + status.stderr
    if status.returncode:
        raise RuntimeError(
            f"cannot inspect apu-00 ({status.returncode}):\n{text[-3000:]}"
        )
    match = re.search(r"^Status\s*:\s*(.+?)\s*$", text, re.MULTILINE)
    if match is None or match.group(1) != "Available":
        observed = "unknown" if match is None else match.group(1)
        raise RuntimeError(
            f"apu-00 is not available (status={observed}); another process may own it"
        )
    core = re.search(r"^Core 0 is (.+?)\s*$", text, re.MULTILINE)
    if core is not None and core.group(1) != "idle":
        raise RuntimeError(f"apu-00 core 0 is not idle ({core.group(1)})")
    return text


def _run_command(command, *, cwd: Path, timeout: int, label: str):
    process = subprocess.run(
        [str(part) for part in command],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if process.returncode:
        output = process.stdout + process.stderr
        raise RuntimeError(
            f"APUg2 {label} failed ({process.returncode}):\n{output[-10000:]}"
        )
    return process


def _parse_metrics(text: str) -> tuple[dict[str, int], dict[str, float]]:
    ticks = {}
    for name, field in _TICK_FIELDS.items():
        matches = re.findall(rf"^{re.escape(field)}=(\d+)\s*$", text, re.MULTILINE)
        if len(matches) != 1:
            raise RuntimeError(f"hardware output has {len(matches)} values for {field}")
        ticks[name] = int(matches[0])
        if ticks[name] <= 0:
            raise RuntimeError(f"hardware returned a nonpositive {field}")

    host = {}
    for field in _HOST_FIELDS:
        matches = re.findall(
            rf"^{re.escape(field)}=([0-9]+(?:\.[0-9]+)?)\s*$",
            text,
            re.MULTILINE,
        )
        if len(matches) != 1:
            raise RuntimeError(f"hardware output has {len(matches)} values for {field}")
        host[field.removesuffix("_us")] = float(matches[0])
    if "PASS checked=262144" not in text:
        raise RuntimeError("hardware host did not report its all-lane correctness gate")
    return ticks, host


def _u16_add_build_manifest():
    return {
        "schema": "apu-g2-cmake-build-v1",
        "device_configure": [
            "cmake",
            "-S",
            "{project}/device",
            "-B",
            "{project}/build/device",
        ],
        "device_compile": [
            "cmake",
            "--build",
            "{project}/build/device",
            "-j",
            "{jobs}",
        ],
        "host_configure": [
            "cmake",
            "-S",
            "{project}",
            "-B",
            "{project}/build/host",
            "-DGSI_TARGET=DEVICE",
            "-DBOARD_DEVICE_LIB={device_library}",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        "host_compile": [
            "cmake",
            "--build",
            "{project}/build/host",
            "-j",
            "{jobs}",
        ],
        "jobs_policy": "TENON_APU_G2_BUILD_JOBS_or_min_cpu_count_8",
        "device_library": "build/device/bin/tenon_apu_g2_tasks.update.bin",
        "host_binary": "build/host/tenon_apu_g2_u16_add",
    }


def _u16_add_abi_manifest():
    return {
        "schema": "apu-g2-u16-add-abi-v1",
        "inputs": [
            ["lhs", "uint16", list(_SHAPE), "c_contiguous"],
            ["rhs", "uint16", list(_SHAPE), "c_contiguous"],
        ],
        "output": ["out", "uint16", list(_SHAPE), "c_contiguous"],
        "argv": ["lhs_path", "rhs_path", "output_path", "repetitions"],
        "arithmetic": "modulo_2^16",
    }


def freeze_apu_g2_u16_add_runtime_artifact() -> APUG2RuntimeArtifact:
    return freeze_apu_g2_runtime_artifact(
        run_apu_g2_u16_add,
        build_manifest=_u16_add_build_manifest(),
        abi_manifest=_u16_add_abi_manifest(),
    )


def run_apu_g2_u16_add(
    lhs, rhs, *, repetitions=256, runtime_artifact=None
) -> RunResult:
    """Run one four-vector uint16 add on Gemini-II core 0.

    ``lhs`` and ``rhs`` must be C-contiguous uint16 arrays of shape
    ``(4, 65536)``.  The four rows map to the four MMB sets, while consecutive
    4096-element portions of each row map to the 16 L1 groups.  The returned
    cycle count is the rounded device ticks per complete repeated pipeline.
    """

    lhs = _validate_operand("lhs", lhs)
    rhs = _validate_operand("rhs", rhs)
    repetitions = _validate_repetitions(repetitions)
    if runtime_artifact is None:
        runtime_artifact = freeze_apu_g2_u16_add_runtime_artifact()
    if not isinstance(runtime_artifact, APUG2RuntimeArtifact):
        raise TypeError("runtime_artifact must be an APUG2RuntimeArtifact")
    runtime_artifact.assert_current(run_apu_g2_u16_add)
    card_info = _require_hardware_stack()
    sources = {
        relative: data.decode("utf-8")
        for relative, data in runtime_artifact.project_files
    }
    source_hashes = dict(runtime_artifact.source_hashes)

    keep = os.environ.get(_KEEP_ENV) == "1"
    root = Path(tempfile.mkdtemp(prefix="tenon-apu-g2-"))
    project = root / "project"
    commands = []
    try:
        runtime_artifact.write_project(project)
        lhs_path = root / "lhs.bin"
        rhs_path = root / "rhs.bin"
        output_path = root / "out.bin"
        lhs.tofile(lhs_path)
        rhs.tofile(rhs_path)

        jobs_text = os.environ.get(_BUILD_JOBS_ENV)
        jobs = int(jobs_text) if jobs_text is not None else min(os.cpu_count() or 1, 8)
        if jobs <= 0:
            raise ValueError(f"{_BUILD_JOBS_ENV} must be positive")

        device_build = project / "build" / "device"
        host_build = project / "build" / "host"
        device_configure = ["cmake", "-S", project / "device", "-B", device_build]
        device_compile = ["cmake", "--build", device_build, "-j", str(jobs)]
        commands.extend((device_configure, device_compile))
        _run_command(
            device_configure, cwd=project, timeout=120, label="device configure"
        )
        _run_command(device_compile, cwd=project, timeout=600, label="device build")

        device_library = device_build / "bin" / "tenon_apu_g2_tasks.update.bin"
        if not device_library.is_file():
            raise RuntimeError(f"ARC build did not produce {device_library}")

        host_configure = [
            "cmake",
            "-S",
            project,
            "-B",
            host_build,
            "-DGSI_TARGET=DEVICE",
            f"-DBOARD_DEVICE_LIB={device_library}",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        host_compile = ["cmake", "--build", host_build, "-j", str(jobs)]
        commands.extend((host_configure, host_compile))
        _run_command(host_configure, cwd=project, timeout=120, label="host configure")
        _run_command(host_compile, cwd=project, timeout=600, label="host build")

        host_binary = host_build / "tenon_apu_g2_u16_add"
        if not host_binary.is_file():
            raise RuntimeError(f"host build did not produce {host_binary}")
        hardware_command = [
            host_binary,
            lhs_path,
            rhs_path,
            output_path,
            str(repetitions),
        ]
        commands.append(hardware_command)
        process = _run_command(
            hardware_command, cwd=project, timeout=120, label="hardware execution"
        )
        text = process.stdout + process.stderr
        ticks, host_timings = _parse_metrics(text)

        output = np.fromfile(output_path, dtype=np.uint16)
        if output.size != np.prod(_SHAPE):
            raise RuntimeError(
                f"hardware output has {output.size} elements, expected {np.prod(_SHAPE)}"
            )
        output = output.reshape(_SHAPE)
        expected = np.add(lhs, rhs, dtype=np.uint16)
        if not np.array_equal(output, expected):
            mismatch = np.argwhere(output != expected)[0]
            index = tuple(int(value) for value in mismatch)
            raise RuntimeError(
                f"APUg2 result differs from NumPy at {index}: "
                f"hardware={int(output[index])} numpy={int(expected[index])}"
            )

        per_call = {
            name: value / repetitions
            for name, value in ticks.items()
            if name != "final_pipeline"
        }
        per_call["final_pipeline"] = float(ticks["final_pipeline"])
        cycles = int(round(per_call["pipeline"]))
        timing_text = "\n".join(
            f"{name}_us={value:.3f}" for name, value in host_timings.items()
        )
        project_metadata = {
            "template_path": str(_TEMPLATE),
            "generated_project_path": str(project) if keep else None,
            "device_library": str(device_library) if keep else device_library.name,
            "source_sha256": source_hashes,
            "source_fingerprint": runtime_artifact.source_fingerprint,
            "commands": [[str(part) for part in command] for command in commands],
            "card_info": card_info,
            "promotion_platform_fingerprint": (
                runtime_artifact.current_platform_fingerprint()
            ),
            "temporary_project_kept": keep,
            "keep_environment_variable": _KEEP_ENV,
        }
        return RunResult(
            cycles,
            text,
            "apu_v2",
            extra={
                "outputs": {"out": output},
                "raw_ticks": ticks,
                "phase_ticks": {
                    name: ticks[name] for name in ("input_copy", "add", "output_copy")
                },
                "total_ticks": ticks["pipeline"],
                "final_pipeline_ticks": ticks["final_pipeline"],
                "per_call_ticks": per_call,
                "repetitions": repetitions,
                "host_timings_us": host_timings,
                "host_timing_text": timing_text,
                "sources": sources,
                "project": project_metadata,
            },
        )
    finally:
        if not keep:
            shutil.rmtree(root, ignore_errors=True)


__all__ = [
    "APUG2RuntimeArtifact",
    "freeze_apu_g2_runtime_artifact",
    "freeze_apu_g2_u16_add_runtime_artifact",
    "run_apu_g2_u16_add",
]
