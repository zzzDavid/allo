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

from dataclasses import dataclass
import inspect
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from types import MappingProxyType

import numpy as np

from ..backend.c import emit_c_from_mlir
from ..customize import customize
from ..perf import CostEvent
from ..perf.graph import ExecutionGraph
from ..spmw_codegen import RunResult
from .upmem_analysis import analyze_upmem_mlir
from .apu_v1_hybrid import (
    APUv1PrecisionPolicy,
    build_apu_v1_hybrid_execution_graph,
    build_apu_v1_region_executables,
    compile_apu_v1_hybrid,
    discover_apu_v1_hybrid_manifest,
)


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
    """One complete Allo function lowered to scalar ARC C."""

    kernel: object
    name: str | None = None
    instantiate: tuple = ()
    result_names: tuple[str, ...] = ()
    vectorize: str | bool = False
    precision_policy: APUv1PrecisionPolicy | None = None
    vector_layout: str | None = None
    produces: tuple[str, ...] = ()
    bindings: Mapping[str, str] | tuple[tuple[str, str], ...] = ()
    dependencies: tuple[str, ...] = ()
    zero_initialize: tuple[str, ...] = ()

    def __post_init__(self):
        if not callable(self.kernel):
            raise TypeError("APUv1Phase kernel must be callable")
        object.__setattr__(self, "instantiate", tuple(self.instantiate))
        object.__setattr__(self, "result_names", tuple(self.result_names))
        vectorize = "required" if self.vectorize is True else self.vectorize
        if vectorize not in {False, "required"}:
            raise ValueError("APUv1Phase.vectorize must be False or 'required'")
        if self.precision_policy is not None and not isinstance(
            self.precision_policy, APUv1PrecisionPolicy
        ):
            raise TypeError("precision_policy must be an APUv1PrecisionPolicy")
        if self.vector_layout is not None and not isinstance(self.vector_layout, str):
            raise TypeError("vector_layout must be a vector-plan name")
        produces = tuple(self.produces)
        dependencies = tuple(self.dependencies)
        zero_initialize = tuple(self.zero_initialize)
        if len(produces) != len(set(produces)):
            raise ValueError("APUv1Phase.produces values must be unique")
        if not set(zero_initialize) <= set(produces):
            raise ValueError("zero_initialize values must also appear in produces")
        bindings = dict(self.bindings)
        if any(not key or not value for key, value in bindings.items()):
            raise ValueError("APUv1Phase bindings require non-empty names")
        object.__setattr__(self, "vectorize", vectorize)
        object.__setattr__(self, "produces", produces)
        object.__setattr__(self, "bindings", MappingProxyType(bindings))
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(self, "zero_initialize", zero_initialize)

    @property
    def phase_name(self):
        return self.name or getattr(self.kernel, "__name__", "phase")


class APUv1Program:
    """An ordered logical APU program; physical execution may be hybrid."""

    def __init__(self, phases, *, name: str | None = None):
        self.phases = tuple(phases)
        if not self.phases or not all(
            isinstance(phase, APUv1Phase) for phase in self.phases
        ):
            raise ValueError("APUv1Program requires one or more APUv1Phase values")
        names = tuple(phase.phase_name for phase in self.phases)
        if len(names) != len(set(names)):
            raise ValueError("APUv1Program phase names must be unique")
        seen = set()
        produced = set()
        for phase in self.phases:
            unknown = set(phase.dependencies) - seen
            if unknown:
                raise ValueError(
                    f"phase {phase.phase_name!r} has unknown/forward dependencies "
                    f"{sorted(unknown)}"
                )
            overlap = produced & set(phase.produces)
            if overlap:
                raise ValueError(
                    f"logical buffers have multiple producing phases: {sorted(overlap)}"
                )
            seen.add(phase.phase_name)
            produced.update(phase.produces)
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
    transformed = re.sub(r"\bsqrt\s*\(", "allo_sqrtf(", transformed)
    allocator = """
static uint8_t *allo_l4_cursor;
static void *allo_l4_alloc(size_t bytes) {
    uintptr_t cursor = ((uintptr_t)allo_l4_cursor + 7u) & ~(uintptr_t)7u;
    allo_l4_cursor = (uint8_t *)(cursor + bytes);
    return (void *)cursor;
}
static float allo_sqrtf(float value) {
    if (value <= 0.0f) return 0.0f;
    float estimate = value > 1.0f ? value : 1.0f;
    for (int i = 0; i < 16; ++i)
        estimate = 0.5f * (estimate + value / estimate);
    return estimate;
}
"""
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
):
    from ..spmw_apu_v1_build import (
        _COPY_FILES,
        _emit_host_c,
        _emit_makefile,
        _emit_struct_h,
        _template_dir,
    )

    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    template = _template_dir()
    shutil.copytree(template / "Common", dst / "Common")
    for name in _COPY_FILES:
        shutil.copy2(template / name, dst / name)
    in_roles = sorted(inputs)
    out_roles = sorted(output_specs)
    input_sizes = {name: value.nbytes for name, value in inputs.items()}
    output_sizes = {
        name: math.prod(shape) * np.dtype(dtype).itemsize
        for name, (shape, dtype) in output_specs.items()
    }
    if host_source is None:
        host = _emit_host_c(in_roles, out_roles, input_sizes, output_sizes)
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


def _ledag_log():
    if shutil.which("ledag-ssh") is None:
        return ""
    time.sleep(0.5)
    process = subprocess.run(
        ["ledag-ssh", "-o", "localhost"],
        input=b"flo\nquit\n",
        capture_output=True,
        timeout=30,
        check=False,
    )
    return "".join(
        chr(byte)
        for byte in process.stdout or b""
        if 32 <= byte < 127 or byte in (9, 10, 13)
    )


def _parse_scalar_profile(text):
    """The scalar baseline launches one APUC, so the newest total is the result."""
    totals = re.findall(r"\btotal\b[^\n]*?\bcrun\s*[:=]\s*(\d+)", text)
    return int(totals[-1]) if totals else None


class CompiledAPUv1Program:
    def __init__(self, program, target, cost=None, backend=None):
        self.program = program
        self.target = target
        self.cost = cost
        if len(program.phases) != 1:
            raise RuntimeError(
                "multi-phase APUv1Program requires the hybrid orchestration backend"
            )
        phase = program.phases[0]
        kwargs = {"enable_tensor": False}
        if phase.instantiate:
            kwargs["instantiate"] = list(phase.instantiate)
        self.schedule = customize(phase.kernel, **kwargs)
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
        self.hybrid_manifest = discover_apu_v1_hybrid_manifest(
            phase,
            self.schedule,
            self.artifact,
            target,
            cost=cost,
            program_name=program.name,
        )
        self.hybrid_callable = None
        if self.hybrid_manifest.has_vector_regions:
            executables = build_apu_v1_region_executables(
                self.hybrid_manifest, self.schedule
            )
            device_runner = None
            if backend is None:
                from .apu_v1_hybrid_runtime import run_apu_v1_hybrid

                device_runner = run_apu_v1_hybrid
            self.hybrid_callable = compile_apu_v1_hybrid(
                self.hybrid_manifest,
                executables,
                target,
                backend="device" if backend is None else backend,
                device_runner=device_runner,
            )
        transformed, self.scratch_bytes = _replace_local_arrays(self.artifact.c_source)
        self.scratch_role = "allo_scratch"
        self.device_source = _emit_device_source(
            self.schedule.top_func_name,
            transformed,
            self.arguments,
            self.outputs,
            self.scratch_role,
        )
        self.execution_graph = ExecutionGraph(program.name)
        if cost is not None:
            if self.hybrid_manifest.has_vector_regions:
                self.execution_graph = build_apu_v1_hybrid_execution_graph(
                    self.hybrid_manifest,
                    target,
                    cost,
                    partitions={
                        region_id: executable.partition
                        for region_id, executable in executables.items()
                        if executable.partition is not None
                    },
                )
            else:
                fallback = max(
                    (
                        extent
                        for argument in self.arguments
                        for extent in argument.shape
                    ),
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
                cost.emit(self.execution_graph, event)

    def run(self, **inputs):
        from ..spmw_codegen import _apu_v1_unavailable_reason

        reason = _apu_v1_unavailable_reason()
        if reason:
            return RunResult(None, reason, "apu_v1")
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
        normalized[self.scratch_role] = np.zeros(self.scratch_bytes, dtype=np.uint8)
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
                ["make"], cwd=project, capture_output=True, timeout=600, check=False
            )
            if build.returncode:
                raise RuntimeError(
                    "APU v1 scalar build failed:\n"
                    + build.stderr.decode(errors="replace")[-6000:]
                )
            binary = project / "build" / "debug" / "tenon-scalar"
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
            if process.returncode:
                raise RuntimeError(
                    f"APU v1 scalar device run failed ({process.returncode}):\n{text[-4000:]}"
                )
            log = text + "\n" + _ledag_log()
            outputs = {}
            for argument in self.outputs:
                role = output_roles[argument.name]
                value = np.fromfile(
                    output_paths[role], dtype=argument.numpy_dtype
                ).reshape(argument.shape)
                outputs[argument.name] = value
                if argument.name in inputs:
                    np.copyto(inputs[argument.name], value, casting="same_kind")
            return RunResult(
                _parse_scalar_profile(log),
                log,
                "apu_v1",
                extra={
                    "outputs": outputs,
                    "scalar_arc": True,
                    "device_source": self.device_source,
                },
            )
        finally:
            if os.environ.get("TENON_APU_V1_KEEP_TMP") != "1":
                shutil.rmtree(root, ignore_errors=True)


class APUv1ProgramCallable:
    def __init__(self, compiled):
        self.compiled = compiled
        self.program = compiled.program
        self.target = compiled.target
        self.cost = compiled.cost
        self.arguments = compiled.source_arguments
        self.signature = inspect.Signature(
            inspect.Parameter(arg.name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for arg in self.arguments
        )
        self.__signature__ = self.signature
        self.last_result = None

    @property
    def execution_graph(self):
        return self.compiled.execution_graph

    @property
    def hybrid_manifest(self):
        return self.compiled.hybrid_manifest

    def __call__(self, *args, **kwargs):
        bound = self.signature.bind(*args, **kwargs)
        if self.compiled.hybrid_callable is None:
            result = self.compiled.run(**bound.arguments)
        else:
            hybrid = self.compiled.hybrid_callable
            forwarded = {
                name: value
                for name, value in bound.arguments.items()
                if name in hybrid.input_names
            }
            result = hybrid(**forwarded)
            for name, value in result.extra.get("outputs", {}).items():
                if name in bound.arguments:
                    np.copyto(bound.arguments[name], value, casting="same_kind")
        self.last_result = result
        return result

    run = __call__

    def run_backend(self, **inputs):
        return self(**inputs)

    def estimate(self):
        if self.cost is None:
            raise RuntimeError("compiled APU v1 program has no cost spec")
        return self.cost.evaluate(self.execution_graph)


def compile_apu_v1_program(program, target, *, cost=None, backend=None):
    if target.name != "apu_v1":
        raise ValueError("APUv1Program requires the apu_v1 target")
    return APUv1ProgramCallable(
        CompiledAPUv1Program(program, target, cost=cost, backend=backend)
    )


__all__ = [
    "APUv1PrecisionPolicy",
    "APUv1Phase",
    "APUv1Program",
    "APUv1ProgramCallable",
    "CompiledAPUv1Program",
    "compile_apu_v1_program",
]
