# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-card runtime for exact-bitwidth Gemini-II VL64 programs."""

from __future__ import annotations

import math
import os
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np

from ..spmw_codegen import RunResult
from .apu_g2_runtime import (
    _BUILD_JOBS_ENV,
    _HOST_FIELDS,
    _KEEP_ENV,
    _TEMPLATE,
    _require_hardware_stack,
    _run_command,
    _source_snapshot,
)
from .apu_g2_typed_program import (
    APUG2TypedOperation,
    APUG2TypedProgram,
    APUG2_TYPED_CAPACITY,
    APUG2_TYPED_LANES_PER_VECTOR,
    APUG2_TYPED_PHYSICAL_VECTORS,
    build_apu_g2_typed_recipe,
)


_OPERATION_ABI = {
    APUG2TypedOperation.ADD: 1,
    APUG2TypedOperation.MUL: 2,
    APUG2TypedOperation.DIV: 3,
    APUG2TypedOperation.BLOCK_SUM: 4,
    APUG2TypedOperation.DOT: 5,
}


def _pack_operand(array: np.ndarray, dtype: np.dtype) -> np.ndarray:
    physical = np.zeros(APUG2_TYPED_CAPACITY, dtype=dtype)
    flattened = np.asarray(array).reshape(-1)
    physical[: flattened.size] = flattened
    return physical.reshape(
        APUG2_TYPED_PHYSICAL_VECTORS,
        APUG2_TYPED_LANES_PER_VECTOR,
    )


def _logical_output(program: APUG2TypedProgram, physical: np.ndarray) -> np.ndarray:
    flattened = physical.reshape(-1)
    if program.operation in {
        APUG2TypedOperation.BLOCK_SUM,
        APUG2TypedOperation.DOT,
    }:
        count = math.prod(program.output_shape)
        values = flattened[
            : count * program.reduction_extent : program.reduction_extent
        ].copy()
        return values.reshape(program.output_shape)
    return flattened[: math.prod(program.shape)].copy().reshape(program.shape)


def _canonicalize(values: np.ndarray, scalar_type) -> np.ndarray:
    """Apply the declared logical integer width inside its NumPy container."""

    wide = np.asarray(values, dtype=np.int64)
    modulus = 1 << scalar_type.bits
    payload = np.mod(wide, modulus)
    if scalar_type.signed:
        sign = 1 << (scalar_type.bits - 1)
        payload = np.where(payload >= sign, payload - modulus, payload)
    return payload.astype(scalar_type.numpy_dtype)


def _expected_output(program: APUG2TypedProgram, operands) -> np.ndarray:
    if program.operation is APUG2TypedOperation.ADD:
        values = operands[0].astype(np.int64) + operands[1].astype(np.int64)
    elif program.operation is APUG2TypedOperation.MUL:
        values = operands[0].astype(np.int64) * operands[1].astype(np.int64)
    elif program.operation is APUG2TypedOperation.DIV:
        values = np.floor_divide(
            operands[0].astype(np.uint64),
            operands[1].astype(np.uint64),
        )
    elif program.operation is APUG2TypedOperation.BLOCK_SUM:
        flattened = operands[0].reshape(-1)
        values = (
            flattened.reshape(-1, program.reduction_extent)
            .astype(np.uint64)
            .sum(axis=1, dtype=np.uint64)
        )
    elif program.operation is APUG2TypedOperation.DOT:
        values = (operands[0].astype(np.int64) * operands[1].astype(np.int64)).sum(
            axis=1, dtype=np.int64
        )
    else:
        raise AssertionError(f"unhandled APUg2 typed operation {program.operation}")
    return _canonicalize(values, program.output_type)


def _parse_metrics(
    text: str, device_library: Path
) -> tuple[dict[str, int], dict[str, float]]:
    if not isinstance(device_library, Path) or not device_library.is_absolute():
        raise TypeError("typed device library attestation requires an absolute Path")

    def unique_text(field: str) -> str:
        matches = re.findall(
            rf"^{re.escape(field)}=([^\r\n]+)\s*$",
            text,
            re.MULTILINE,
        )
        if len(matches) != 1:
            raise RuntimeError(f"hardware output has {len(matches)} values for {field}")
        return matches[0]

    exact_attestations = {
        "target": "hardware",
        "device_library": str(device_library),
        "task_status": "0",
        "timed_scope": "direct_vl64_pipeline",
        "completion_barrier_included": "1",
        "independent_final_correctness_call": "1",
    }
    for field, expected in exact_attestations.items():
        actual = unique_text(field)
        if actual != expected:
            raise RuntimeError(
                f"hardware output {field}={actual!r}, expected {expected!r}"
            )

    ticks = {}
    for name, field in (
        ("pipeline", "device_pipeline_ticks"),
        ("final_pipeline", "device_final_pipeline_ticks"),
    ):
        matches = re.findall(rf"^{field}=(\d+)\s*$", text, re.MULTILINE)
        if len(matches) != 1:
            raise RuntimeError(f"hardware output has {len(matches)} values for {field}")
        ticks[name] = int(matches[0])
        if ticks[name] <= 0:
            raise RuntimeError(f"hardware returned nonpositive {field}")
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
    physical_outputs = re.findall(
        r"^PASS physical_outputs=(\d+)\s*$", text, re.MULTILINE
    )
    if physical_outputs != [str(APUG2_TYPED_CAPACITY)]:
        raise RuntimeError(
            "typed host did not uniquely attest its full physical output"
        )
    return ticks, host


def run_apu_g2_typed(program: APUG2TypedProgram, *operands) -> RunResult:
    """Build and run one legality-checked typed program on ``apu-00``.

    The repeated device timing is the primary throughput sample.  The output
    is produced by a separate final call and independently checked against a
    NumPy oracle.  Both compact logical output and the complete four-vector
    carrier are retained in the result evidence.
    """

    if not isinstance(program, APUG2TypedProgram):
        raise TypeError("program must be an APUG2TypedProgram")
    if len(operands) != len(program.input_types):
        raise TypeError(
            f"{program.operation} requires {len(program.input_types)} operands"
        )
    validated = []
    for index, (operand, shape, scalar_type) in enumerate(
        zip(operands, program.input_shapes, program.input_types)
    ):
        if not isinstance(operand, np.ndarray):
            raise TypeError(f"operand {index} must be a NumPy array")
        if operand.dtype != scalar_type.numpy_dtype:
            raise TypeError(
                f"operand {index} must have dtype {scalar_type.numpy_dtype.name}"
            )
        if operand.shape != shape:
            raise ValueError(f"operand {index} must have shape {shape}")
        if not operand.flags.c_contiguous:
            raise ValueError(f"operand {index} must be C-contiguous")
        if np.any(operand < scalar_type.minimum) or np.any(
            operand > scalar_type.maximum
        ):
            raise ValueError(
                f"operand {index} contains values outside {scalar_type.name}"
            )
        validated.append(operand)
    operands = tuple(validated)
    if program.operation is APUG2TypedOperation.DIV and np.any(operands[1] == 0):
        raise ValueError("APUg2 division requires nonzero divisors")

    card_info = _require_hardware_stack()
    sources, source_hashes = _source_snapshot()
    recipe = build_apu_g2_typed_recipe(program)
    physical_inputs = tuple(
        _pack_operand(operand, scalar_type.numpy_dtype)
        for operand, scalar_type in zip(operands, program.input_types)
    )

    keep = os.environ.get(_KEEP_ENV) == "1"
    root = Path(tempfile.mkdtemp(prefix="tenon-apu-g2-typed-"))
    project = root / "project"
    commands = []
    try:
        shutil.copytree(_TEMPLATE, project)
        input_paths = []
        for index, physical in enumerate(physical_inputs):
            path = root / f"input-{index}.bin"
            physical.tofile(path)
            input_paths.append(path)
        output_path = root / "out.bin"

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
            device_configure,
            cwd=project,
            timeout=120,
            label="typed device configure",
        )
        _run_command(
            device_compile,
            cwd=project,
            timeout=600,
            label="typed device build",
        )
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
        _run_command(
            host_configure,
            cwd=project,
            timeout=120,
            label="typed host configure",
        )
        _run_command(
            host_compile,
            cwd=project,
            timeout=600,
            label="typed host build",
        )
        host_binary = host_build / "tenon_apu_g2_typed"
        if not host_binary.is_file():
            raise RuntimeError(f"host build did not produce {host_binary}")

        lhs_type = program.input_types[0]
        rhs_type = program.input_types[1] if len(program.input_types) == 2 else None
        hardware_command = [
            host_binary,
            str(_OPERATION_ABI[program.operation]),
            input_paths[0],
            input_paths[1] if rhs_type is not None else "-",
            output_path,
            str(lhs_type.bits),
            str(int(lhs_type.signed)),
            str(0 if rhs_type is None else rhs_type.bits),
            str(0 if rhs_type is None else int(rhs_type.signed)),
            str(program.output_type.bits),
            str(int(program.output_type.signed)),
            str(program.log_reduction),
            str(program.repetitions),
        ]
        commands.append(hardware_command)
        process = _run_command(
            hardware_command,
            cwd=project,
            timeout=180,
            label="typed hardware execution",
        )
        text = process.stdout + process.stderr
        ticks, host_timings = _parse_metrics(text, device_library)

        physical_output = np.fromfile(
            output_path, dtype=program.output_type.numpy_dtype
        )
        if physical_output.size != APUG2_TYPED_CAPACITY:
            raise RuntimeError(
                "typed host returned "
                f"{physical_output.size} physical values, "
                f"expected {APUG2_TYPED_CAPACITY}"
            )
        physical_output = physical_output.reshape(
            APUG2_TYPED_PHYSICAL_VECTORS,
            APUG2_TYPED_LANES_PER_VECTOR,
        )
        logical_output = _logical_output(program, physical_output)
        expected = _expected_output(program, operands)
        if not np.array_equal(logical_output, expected):
            mismatch = np.argwhere(logical_output != expected)[0]
            index = tuple(int(item) for item in mismatch)
            raise RuntimeError(
                f"APUg2 typed {program.operation} differs from NumPy at "
                f"{index}: hardware={int(logical_output[index])}, "
                f"numpy={int(expected[index])}"
            )

        per_call = {
            "pipeline": ticks["pipeline"] / program.repetitions,
            "final_pipeline": float(ticks["final_pipeline"]),
        }
        cycles = int(round(per_call["pipeline"]))
        return RunResult(
            cycles,
            text,
            "apu_v2",
            extra={
                "outputs": {
                    "out": logical_output,
                    "physical_out": physical_output,
                },
                "oracle": expected,
                "raw_ticks": ticks,
                "total_ticks": ticks["pipeline"],
                "final_pipeline_ticks": ticks["final_pipeline"],
                "per_call_ticks": per_call,
                "repetitions": program.repetitions,
                "host_timings_us": host_timings,
                "program": program.manifest(),
                "recipe": recipe.manifest(),
                "recipe_fingerprint": recipe.structural_fingerprint,
                "vectorization_certificate": recipe.certificate.manifest(),
                "sources": sources,
                "project": {
                    "template_path": str(_TEMPLATE),
                    "generated_project_path": str(project) if keep else None,
                    "device_library": (
                        str(device_library) if keep else device_library.name
                    ),
                    "source_sha256": source_hashes,
                    "commands": [
                        [str(part) for part in command] for command in commands
                    ],
                    "card_info": card_info,
                    "temporary_project_kept": keep,
                    "keep_environment_variable": _KEEP_ENV,
                },
            },
        )
    finally:
        if not keep:
            shutil.rmtree(root, ignore_errors=True)


__all__ = ["run_apu_g2_typed"]
