# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-device runtime for a realized APU v1 vector plan.

The vector lowering deliberately stops at a reusable GVML function fragment.
This module supplies the GDL task/host harness, translates layout-defined NumPy
images into per-APUC L4 buffers, and gathers the returned VR images.  It is kept
separate from planning and costing so those abstractions remain executable on a
machine without the proprietary SDK.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from types import CodeType, FunctionType

import numpy as np

from ..spmw_codegen import RunResult
from .apu_v1_program import _generate_project, _ledag_log
from .apu_v1_vector_codegen import PointerOffsetRef, PointerRef, VR_LANES


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


def _callable_manifest(function: FunctionType) -> tuple[str, str, str]:
    if not isinstance(function, FunctionType):
        raise TypeError("APU v1 runtime dependencies must be Python functions")
    return (
        function.__module__,
        function.__qualname__,
        _sha256_bytes(
            _canonical_json(_code_manifest(function.__code__)).encode("ascii")
        ),
    )


def _promotion_platform_fingerprint():
    """Return no promotion identity until trusted board attestation exists.

    Caller-authored JSON and hashes of caller-selected software files cannot
    prove the identity or current state of the board, driver, or loaded
    firmware. They therefore must not authorize schedule activation.
    """

    return None


def _module_source_hashes() -> tuple[tuple[str, str], ...]:
    names = (
        "allo.pim.apu_v1_layout",
        "allo.pim.apu_v1_program",
        "allo.pim.apu_v1_vector_codegen",
        "allo.pim.apu_v1_vector_runtime",
        "allo.spmw_apu_v1_build",
        "allo.spmw_codegen",
        "allo.spmw_linear_layout",
    )
    hashes = []
    for name in names:
        module = importlib.import_module(name)
        source = inspect.getsourcefile(module)
        if source is None:
            raise RuntimeError(f"APU v1 runtime module {name!r} has no source")
        path = Path(source).resolve()
        if not path.is_file():
            raise RuntimeError(f"APU v1 runtime source is missing: {path}")
        hashes.append((name, _sha256_file(path)))
    return tuple(hashes)


def _implementation_manifest(realization) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        sorted(
            _callable_manifest(function)
            for function in (
                run_apu_v1_vector,
                _full_device_source,
                _generate_project,
                _generate_runtime_project,
                _pointer_values,
                _parse_vector_profile,
                type(realization).device_source,
                type(realization.abi).pack,
                type(realization.abi).transfer_input_images,
                type(realization.abi).gather,
            )
        )
    )


def _template_inventory():
    from ..spmw_apu_v1_build import _COPY_FILES, _template_dir

    template = _template_dir()
    paths = [item for item in (template / "Common").rglob("*") if item.is_file()]
    paths.extend(template / name for name in _COPY_FILES)
    hashes = []
    modes = []
    for path in sorted(paths):
        if not path.is_file():
            raise FileNotFoundError(f"APU v1 template source is missing: {path}")
        relative = str(path.relative_to(template))
        hashes.append((relative, _sha256_file(path)))
        modes.append((relative, path.stat().st_mode & 0o777))
    return tuple(hashes), tuple(modes)


def _zero_arrays(realization):
    return {
        value.name: np.zeros(value.shape, dtype=value.dtype)
        for value in realization.values
    }


def _runtime_io(realization):
    arrays = _zero_arrays(realization)
    output_names = tuple(
        value.name for value in realization.values if value.intent in {"out", "inout"}
    )
    packed = realization.abi.pack(arrays, values=output_names)
    input_images = realization.abi.transfer_input_images(arrays)
    output_specs = {}
    output_batches = {}
    for value in realization.values:
        if value.intent in {"out", "inout"}:
            batches = int(packed[value.name].data.shape[0])
            output_batches[value.name] = batches
            output_specs[value.name] = ((batches, VR_LANES), np.uint16)
    role_bytes = {
        **{name: value.nbytes for name, value in input_images.items()},
        **{
            name: math.prod(shape) * np.dtype(dtype).itemsize
            for name, (shape, dtype) in output_specs.items()
        },
    }
    return input_images, output_specs, output_batches, role_bytes


def _abi_manifest(realization, input_images, output_specs, output_batches, role_bytes):
    return {
        "schema": "apu-v1-vector-runtime-abi-v1",
        "inputs": [
            [name, str(value.dtype), int(value.nbytes)]
            for name, value in sorted(input_images.items())
        ],
        "outputs": [
            [
                name,
                list(shape),
                np.dtype(dtype).str,
                int(output_batches[name]),
                int(role_bytes[name]),
            ]
            for name, (shape, dtype) in sorted(output_specs.items())
        ],
        "pointers": [list(pointer) for pointer in _pointer_values(realization)],
        "argv": {
            "inputs": sorted(input_images),
            "outputs": sorted(output_specs),
        },
        "single_apuc": True,
        "vector_lanes": VR_LANES,
    }


def _build_manifest(lab_name: str):
    return {
        "schema": "apu-v1-make-build-v1",
        "build": ["make"],
        "build_timeout_seconds": 600,
        "binary": f"build/debug/{lab_name}",
        "execution_timeout_seconds": 900,
        "execution_attempts": 3,
        "retry_marker": "no valid context",
        "retry_backoff_seconds": [1, 2],
    }


def _snapshot_directory(root: Path):
    paths = sorted(item for item in root.rglob("*") if item.is_file())
    files = tuple((str(path.relative_to(root)), path.read_bytes()) for path in paths)
    modes = tuple(
        (str(path.relative_to(root)), path.stat().st_mode & 0o777) for path in paths
    )
    return files, modes


def _generate_runtime_project(realization, lab_name: str):
    input_images, output_specs, output_batches, role_bytes = _runtime_io(realization)
    root = Path(tempfile.mkdtemp(prefix="tenon-apu-v1-freeze-"))
    try:
        project = _generate_project(
            root / "project",
            _full_device_source(realization, role_bytes),
            input_images,
            output_specs,
            lab_name,
        )
        files, modes = _snapshot_directory(project)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return (
        files,
        modes,
        _abi_manifest(
            realization,
            input_images,
            output_specs,
            output_batches,
            role_bytes,
        ),
    )


def _runtime_source_hashes(
    files,
    modes,
    module_hashes,
    implementation,
    build_json,
    abi_json,
):
    inventory = [
        (f"project/{relative}", _sha256_bytes(data)) for relative, data in files
    ]
    inventory.extend((f"runtime/{name}", digest) for name, digest in module_hashes)
    inventory.extend(
        (
            (
                "contract/executor.json",
                _sha256_bytes(_canonical_json(implementation).encode("ascii")),
            ),
            ("contract/build.json", _sha256_bytes(build_json.encode("ascii"))),
            ("contract/abi.json", _sha256_bytes(abi_json.encode("ascii"))),
            (
                "contract/modes.json",
                _sha256_bytes(_canonical_json(modes).encode("ascii")),
            ),
        )
    )
    return tuple(sorted(inventory))


@dataclass(frozen=True)
class APUV1RuntimeArtifact:
    project_files: tuple[tuple[str, bytes], ...]
    project_modes: tuple[tuple[str, int], ...]
    template_source_hashes: tuple[tuple[str, str], ...]
    template_modes: tuple[tuple[str, int], ...]
    module_source_hashes: tuple[tuple[str, str], ...]
    implementation_manifest: tuple[tuple[str, str, str], ...]
    build_manifest_json: str
    abi_manifest_json: str
    source_hashes: tuple[tuple[str, str], ...]
    source_fingerprint: str
    platform_fingerprint: object | None
    lab_name: str

    def write_project(self, destination: Path) -> Path:
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        root = destination.resolve()
        for relative, data in self.project_files:
            path = (destination / relative).resolve()
            if root not in path.parents:
                raise RuntimeError(f"unsafe APU v1 project path {relative!r}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            path.chmod(dict(self.project_modes)[relative])
        return destination

    def current_platform_fingerprint(self):
        return _promotion_platform_fingerprint()

    def current_source_hashes(self, realization) -> tuple[tuple[str, str], ...]:
        files, modes, abi = _generate_runtime_project(realization, self.lab_name)
        return _runtime_source_hashes(
            files,
            modes,
            _module_source_hashes(),
            _implementation_manifest(realization),
            _canonical_json(_build_manifest(self.lab_name)),
            _canonical_json(abi),
        )

    def current_source_fingerprint(self, realization) -> str:
        return _sha256_bytes(
            _canonical_json(self.current_source_hashes(realization)).encode("ascii")
        )

    def promotion_source_fingerprint(self, realization) -> str | None:
        try:
            self.assert_current(realization)
        except (OSError, RuntimeError, TypeError, ValueError):
            return None
        return self.source_fingerprint

    def assert_current(self, realization) -> None:
        template_hashes, template_modes = _template_inventory()
        if template_hashes != self.template_source_hashes:
            raise RuntimeError("APU v1 runtime template changed after materialization")
        if template_modes != self.template_modes:
            raise RuntimeError(
                "APU v1 runtime template modes changed after materialization"
            )
        if _module_source_hashes() != self.module_source_hashes:
            raise RuntimeError("APU v1 runtime driver changed after materialization")
        if _implementation_manifest(realization) != self.implementation_manifest:
            raise RuntimeError(
                "APU v1 executor implementation changed after materialization"
            )
        if self.current_source_fingerprint(realization) != self.source_fingerprint:
            raise RuntimeError("APU v1 runtime project changed after materialization")
        if self.platform_fingerprint != self.current_platform_fingerprint():
            raise RuntimeError("APU v1 platform contract changed after materialization")


def freeze_apu_v1_runtime_artifact(
    realization, *, lab_name: str = "tenon-vector"
) -> APUV1RuntimeArtifact:
    files, modes, abi = _generate_runtime_project(realization, lab_name)
    template_hashes, template_modes = _template_inventory()
    module_hashes = _module_source_hashes()
    implementation = _implementation_manifest(realization)
    build_json = _canonical_json(_build_manifest(lab_name))
    abi_json = _canonical_json(abi)
    source_hashes = _runtime_source_hashes(
        files,
        modes,
        module_hashes,
        implementation,
        build_json,
        abi_json,
    )
    return APUV1RuntimeArtifact(
        project_files=files,
        project_modes=modes,
        template_source_hashes=template_hashes,
        template_modes=template_modes,
        module_source_hashes=module_hashes,
        implementation_manifest=implementation,
        build_manifest_json=build_json,
        abi_manifest_json=abi_json,
        source_hashes=source_hashes,
        source_fingerprint=_sha256_bytes(
            _canonical_json(source_hashes).encode("ascii")
        ),
        platform_fingerprint=_promotion_platform_fingerprint(),
        lab_name=lab_name,
    )


def _pointer_values(realization):
    return tuple(
        sorted(
            {
                (argument.name, argument.level)
                for instruction in realization.instructions
                for argument in instruction.args
                if isinstance(argument, (PointerRef, PointerOffsetRef))
            }
        )
    )


def _parse_vector_profile(text: str) -> int | None:
    """Return the newest device ``total`` CRUN counter."""

    totals = re.findall(r"\btotal\b[^\n]*?\bcrun\s*[:=]\s*(\d+)", text)
    return int(totals[-1]) if totals else None


def _full_device_source(realization, role_bytes: dict[str, int]) -> str:
    pointers = _pointer_values(realization)
    function = re.sub(r"\W", "_", str(getattr(realization.plan, "name", "apu_vector")))
    declarations = []
    arguments = []
    l3_copies = []
    for name, level in pointers:
        if level == "L4":
            declarations.append(
                f"    uint16_t *{name}_L4ptr = (uint16_t *)"
                f"gal_mem_handle_to_apu_ptr(data->mem_hndl_{name});"
            )
        else:
            nbytes = int(role_bytes[name])
            if nbytes > (1 << 20):
                raise ValueError(
                    f"APU v1 L3 image {name!r} needs {nbytes} bytes; per-APUC L3 is 1 MiB"
                )
            declarations.append(
                f"    uint16_t *{name}_L3ptr = (uint16_t *)"
                f"gal_fast_malloc_cache_aligned({nbytes}, true);"
            )
            declarations.append(
                f"    uint16_t *{name}_L4_source = (uint16_t *)"
                f"gal_mem_handle_to_apu_ptr(data->mem_hndl_{name});"
            )
            l3_copies.append(
                f"    allo_l4_to_l3({name}_L3ptr, {name}_L4_source, {nbytes});"
            )
        arguments.append(f"{name}_{level}ptr")

    return (
        "#include <gsi/libsys/assert.h>\n"
        "#include <gsi/libsys.h>\n"
        "#include <gsi/libgal.h>\n"
        "#include <gsi/gal-fast-funcs.h>\n"
        '#include "struct.h"\n'
        "#include <gsi_device_profiling.h>\n\n"
        + realization.device_source()
        + "\nPROF_VAR(total);\n\n"
        "static void allo_l4_to_l3(void *destination, void *source, uint32_t bytes) {\n"
        "    uint8_t *dst = (uint8_t *)destination;\n"
        "    uint8_t *src = (uint8_t *)source;\n"
        "    gal_fast_l2dma_async_memcpy_init(GAL_L2DMA_APC_ID_0);\n"
        "    for (uint32_t offset = 0; offset < bytes; offset += 512)\n"
        "        gal_fast_l2dma_mem_to_mem_512(dst + offset, src + offset, "
        "GAL_L2DMA_APC_ID_0);\n"
        "    gal_fast_l2dma_async_memcpy_end(GAL_L2DMA_APC_ID_0);\n"
        "}\n\n"
        "static int my_kernel(struct program_data *data) {\n"
        "    arc_counters_init();\n"
        "    PROF_INIT(total);\n"
        + "\n".join(declarations)
        + "\n"
        + "\n".join(l3_copies)
        + "\n    PROF_START(total);\n"
        f"    int ret = {function}_vector({', '.join(arguments)});\n"
        "    PROF_END(total);\n"
        "    PROF_PRINT(total);\n"
        "    return ret;\n"
        "}\n\n"
        "GAL_TASK_ENTRY_POINT(apu_kernel_task, in, out) {\n"
        "    struct program_cmd *cmd = (struct program_cmd *)in;\n"
        "    return my_kernel(&cmd->data);\n"
        "}\n"
    )


def run_apu_v1_vector(compiled, arrays, *, lab_name: str = "tenon-vector"):
    """Build and execute one vector realization on the installed APU v1 board.

    This reference mapping intentionally launches APUC 0, matching the
    MICRO'25 layout study.  Explicit transfer routes compact lookup tables,
    retain RHS banks, and materialize compute replicas on device; ``vr_batch``
    handles outputs larger than one VR.  An ``apuc`` layout dimension is the
    independent multicore extension and does not change this ABI.
    """

    from ..spmw_codegen import _apu_v1_unavailable_reason

    realization = compiled.realization
    if realization is None:
        raise RuntimeError(compiled.realization_error or "plan has no GVML realization")
    runtime_artifact = getattr(realization, "runtime_artifact", None)
    if runtime_artifact is not None:
        if not isinstance(runtime_artifact, APUV1RuntimeArtifact):
            raise TypeError("realization runtime_artifact has the wrong type")
        if runtime_artifact.lab_name != lab_name:
            raise RuntimeError(
                "APU v1 launch name differs from the frozen materialization"
            )
        runtime_artifact.assert_current(realization)
    reason = _apu_v1_unavailable_reason()
    if reason:
        return RunResult(None, reason, "apu_v1")
    if runtime_artifact is None:
        raise RuntimeError(
            "APU v1 device execution has no complete frozen runtime artifact"
        )

    if realization.compute_tiles_per_output > 1:
        for value in realization.values:
            if value.intent in {"out", "inout"} and np.any(arrays[value.name] != 0):
                raise ValueError(
                    "spatial APU group-reduction scatter currently requires a "
                    "zero-initialized output; nonzero inout accumulation has no "
                    "layout-safe group-head ingress"
                )

    output_names = tuple(
        value.name for value in realization.values if value.intent in {"out", "inout"}
    )
    packed = realization.abi.pack(arrays, values=output_names)
    routed_inputs = realization.abi.transfer_input_images(arrays)
    input_images = {}
    output_specs = {}
    output_batches = {}
    for value in realization.values:
        if value.intent == "in":
            input_images[value.name] = routed_inputs[value.name]
        elif value.intent in {"out", "inout"}:
            image = packed[value.name]
            # The current contraction plans initialize the carried output to
            # zero.  Keeping it as the output buffer avoids an unnecessary
            # host round trip and matches the MICRO kernels.
            batches = int(image.data.shape[0])
            output_batches[value.name] = batches
            output_specs[value.name] = ((batches, VR_LANES), np.uint16)

    referenced = {name for name, _level in _pointer_values(realization)}
    absent = referenced - set(input_images) - set(output_specs)
    if absent:
        raise ValueError(
            f"vector device source references unbound roles: {sorted(absent)}"
        )
    role_bytes = {
        **{name: value.nbytes for name, value in input_images.items()},
        **{
            name: math.prod(shape) * np.dtype(dtype).itemsize
            for name, (shape, dtype) in output_specs.items()
        },
    }
    current_abi = _abi_manifest(
        realization,
        input_images,
        output_specs,
        output_batches,
        role_bytes,
    )
    if _canonical_json(current_abi) != runtime_artifact.abi_manifest_json:
        raise RuntimeError("APU v1 launch inputs do not match the frozen ABI")

    root = tempfile.mkdtemp(prefix="tenon-apu-v1-vector-")
    try:
        project = runtime_artifact.write_project(Path(root) / "project")
        input_paths = {}
        for role, value in input_images.items():
            path = Path(root) / f"in_{role}.bin"
            value.tofile(path)
            input_paths[role] = path
        output_paths = {role: Path(root) / f"out_{role}.bin" for role in output_specs}
        build = subprocess.run(
            ["make"], cwd=project, capture_output=True, timeout=600, check=False
        )
        if build.returncode:
            raise RuntimeError(
                "APU v1 vector build failed:\n"
                + build.stderr.decode(errors="replace")[-8000:]
            )
        binary = project / "build" / "debug" / lab_name
        argv = [str(binary)]
        argv.extend(str(input_paths[name]) for name in sorted(input_paths))
        argv.extend(str(output_paths[name]) for name in sorted(output_paths))
        process = None
        text = ""
        for attempt in range(3):
            process = subprocess.run(
                argv, cwd=project, capture_output=True, timeout=900, check=False
            )
            text = process.stdout.decode(errors="replace") + process.stderr.decode(
                errors="replace"
            )
            if process.returncode == 0 or "no valid context" not in text:
                break
            time.sleep(attempt + 1)
        if process is None or process.returncode:
            code = None if process is None else process.returncode
            raise RuntimeError(
                f"APU v1 vector device run failed ({code}):\n{text[-6000:]}"
            )

        raw_outputs = {}
        packed_outputs = dict(packed)
        for role, batches in output_batches.items():
            raw = np.fromfile(output_paths[role], dtype=np.uint16).reshape(
                batches, VR_LANES
            )
            raw_outputs[role] = raw
            image = packed_outputs[role]
            data = image.data.copy()
            data[:, 0, :] = raw
            packed_outputs[role] = data
        outputs = realization.abi.gather(packed_outputs, outputs=tuple(output_specs))
        for name, value in outputs.items():
            if name in arrays:
                np.copyto(arrays[name], value, casting="same_kind")
        log = text + "\n" + _ledag_log()
        return RunResult(
            _parse_vector_profile(log),
            log,
            "apu_v1",
            extra={
                "outputs": outputs,
                "raw_outputs": raw_outputs,
                "selected_plan": realization.plan.name,
                "device_source": dict(runtime_artifact.project_files)[
                    "device.c"
                ].decode("utf-8"),
                "source_sha256": dict(runtime_artifact.source_hashes),
                "source_fingerprint": runtime_artifact.source_fingerprint,
                "promotion_platform_fingerprint": (
                    runtime_artifact.current_platform_fingerprint()
                ),
                "single_apuc": True,
            },
        )
    finally:
        if os.environ.get("TENON_APU_V1_KEEP_TMP") != "1":
            shutil.rmtree(root, ignore_errors=True)


__all__ = [
    "APUV1RuntimeArtifact",
    "freeze_apu_v1_runtime_artifact",
    "run_apu_v1_vector",
]
