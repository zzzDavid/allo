# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-task, resident-intermediate APUg2 uint16 ATAX runtime."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np

from ..spmw_codegen import RunResult
from .apu_g2_layout import APUG2_U16_SHAPE
from .apu_g2_runtime import (
    _BUILD_JOBS_ENV,
    _HOST_FIELDS,
    _KEEP_ENV,
    _TEMPLATE,
    _require_hardware_stack,
    _run_command,
    _source_snapshot,
    _validate_repetitions,
)


_CHUNK = 32
_CHUNKS = 4
_LANES = APUG2_U16_SHAPE[1]
_TICK_FIELDS = {
    "pipeline": "device_pipeline_ticks",
    "final_pipeline": "device_final_pipeline_ticks",
}


def _matrix(value, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != np.dtype(np.uint16):
        raise TypeError(f"{name} must have dtype uint16, got {value.dtype}")
    if value.ndim != 2 or 0 in value.shape:
        raise ValueError(f"{name} must be a nonempty matrix, got {value.shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return value


def _vector(value, name: str, extent: int) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != np.dtype(np.uint16):
        raise TypeError(f"{name} must have dtype uint16, got {value.dtype}")
    if value.shape != (extent,):
        raise ValueError(f"{name} must have shape {(extent,)}, got {value.shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return value


def _pack_stage1(matrix: np.ndarray) -> np.ndarray:
    rows, reduction = matrix.shape
    packed = np.zeros((_CHUNKS, *APUG2_U16_SHAPE), dtype=np.uint16)
    for row in range(rows):
        stream, local_row = divmod(row, _CHUNK)
        for depth in range(reduction):
            chunk, local_depth = divmod(depth, _CHUNK)
            packed[chunk, stream, local_row * _CHUNK + local_depth] = matrix[row, depth]
    return packed


def _pack_stage1_vector(vector: np.ndarray) -> np.ndarray:
    packed = np.zeros((_CHUNKS, _LANES), dtype=np.uint16)
    for chunk in range(_CHUNKS):
        values = np.zeros(_CHUNK, dtype=np.uint16)
        begin = chunk * _CHUNK
        end = min(begin + _CHUNK, vector.size)
        values[: end - begin] = vector[begin:end]
        packed[chunk] = np.tile(values, _LANES // _CHUNK)
    return packed


def _stage2_coordinate(output: int) -> int:
    group, slot = output % 16, output // 16
    return group * 4096 + slot * _CHUNK


def _pack_stage2_matrix(matrix: np.ndarray) -> np.ndarray:
    rows, outputs = matrix.shape
    packed = np.zeros(APUG2_U16_SHAPE, dtype=np.uint16)
    for output in range(outputs):
        base = _stage2_coordinate(output)
        for depth in range(rows):
            stream, local_depth = divmod(depth, _CHUNK)
            packed[stream, base + local_depth] = matrix[depth, output]
    return packed


def _pack_output(vector: np.ndarray) -> np.ndarray:
    packed = np.zeros(APUG2_U16_SHAPE, dtype=np.uint16)
    for output, value in enumerate(vector):
        packed[0, _stage2_coordinate(output)] = value
    return packed


def _unpack_tmp(packed: np.ndarray, extent: int) -> np.ndarray:
    result = np.empty(extent, dtype=np.uint16)
    for row in range(extent):
        stream, local_row = divmod(row, _CHUNK)
        result[row] = packed[stream, local_row]
    return result


def _unpack_output(packed: np.ndarray, extent: int) -> np.ndarray:
    return np.asarray(
        [packed[0, _stage2_coordinate(output)] for output in range(extent)],
        dtype=np.uint16,
    )


def _reference(matrix, vector, initial):
    tmp = (
        (matrix.astype(np.uint64) @ vector.astype(np.uint64)) & np.uint64(0xFFFF)
    ).astype(np.uint16)
    out = (
        initial.astype(np.uint64) + matrix.astype(np.uint64).T @ tmp.astype(np.uint64)
    ) & np.uint64(0xFFFF)
    return tmp, out.astype(np.uint16)


def _parse_metrics(text: str):
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
    if "PASS atax_resident_outputs=524288" not in text:
        raise RuntimeError("hardware host did not report its ATAX readback gate")
    return ticks, host


def run_apu_g2_u16_atax(matrix, vector, output, *, repetitions=4) -> RunResult:
    """Execute ``tmp=A@vector; output+=A.T@tmp`` modulo 2^16 in one task."""

    matrix = _matrix(matrix, "matrix")
    rows, columns = matrix.shape
    if rows > 128 or columns > 128:
        raise ValueError("resident APUg2 ATAX currently supports extents up to 128")
    vector = _vector(vector, "vector", columns)
    output = _vector(output, "output", columns)
    repetitions = _validate_repetitions(repetitions)

    stage1_matrices = _pack_stage1(matrix)
    stage1_x = _pack_stage1_vector(vector)
    stage2_matrix = _pack_stage2_matrix(matrix)
    y = _pack_output(output)
    expected_tmp, expected_out = _reference(matrix, vector, output)

    card_info = _require_hardware_stack()
    sources, source_hashes = _source_snapshot()
    keep = os.environ.get(_KEEP_ENV) == "1"
    root = Path(tempfile.mkdtemp(prefix="tenon-apu-g2-atax-"))
    project = root / "project"
    commands = []
    try:
        shutil.copytree(_TEMPLATE, project)
        paths = {
            "stage1_matrices": root / "stage1_matrices.bin",
            "stage1_x": root / "stage1_x.bin",
            "stage2_matrix": root / "stage2_matrix.bin",
            "y": root / "y.bin",
            "tmp": root / "resident_tmp.bin",
            "out": root / "out.bin",
        }
        stage1_matrices.tofile(paths["stage1_matrices"])
        stage1_x.tofile(paths["stage1_x"])
        stage2_matrix.tofile(paths["stage2_matrix"])
        y.tofile(paths["y"])

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
        host_binary = host_build / "tenon_apu_g2_u16_atax"
        if not host_binary.is_file():
            raise RuntimeError(f"host build did not produce {host_binary}")

        hardware_command = [
            host_binary,
            paths["stage1_matrices"],
            paths["stage1_x"],
            paths["stage2_matrix"],
            paths["y"],
            paths["tmp"],
            paths["out"],
            str(repetitions),
        ]
        commands.append(hardware_command)
        process = _run_command(
            hardware_command, cwd=project, timeout=240, label="hardware execution"
        )
        text = process.stdout + process.stderr
        ticks, host_timings = _parse_metrics(text)

        expected_size = int(np.prod(APUG2_U16_SHAPE))
        packed_tmp = np.fromfile(paths["tmp"], dtype=np.uint16)
        packed_out = np.fromfile(paths["out"], dtype=np.uint16)
        if packed_tmp.size != expected_size or packed_out.size != expected_size:
            raise RuntimeError("hardware returned a malformed ATAX output pack")
        observed_tmp = _unpack_tmp(packed_tmp.reshape(APUG2_U16_SHAPE), rows)
        observed_out = _unpack_output(packed_out.reshape(APUG2_U16_SHAPE), columns)
        for name, observed, expected in (
            ("tmp", observed_tmp, expected_tmp),
            ("out", observed_out, expected_out),
        ):
            if not np.array_equal(observed, expected):
                index = int(np.flatnonzero(observed != expected)[0])
                raise RuntimeError(
                    f"APUg2 ATAX {name} differs from NumPy at {index}: "
                    f"hardware={int(observed[index])} numpy={int(expected[index])}"
                )
        output[...] = observed_out

        per_call = {
            "pipeline": ticks["pipeline"] / repetitions,
            "final_pipeline": float(ticks["final_pipeline"]),
        }
        return RunResult(
            int(round(per_call["pipeline"])),
            text,
            "apu_v2",
            extra={
                "outputs": {"tmp": observed_tmp, "out": observed_out},
                "raw_ticks": ticks,
                "total_ticks": ticks["pipeline"],
                "final_pipeline_ticks": ticks["final_pipeline"],
                "per_call_ticks": per_call,
                "repetitions": repetitions,
                "host_timings_us": host_timings,
                "resident_intermediate": True,
                "hardware_tasks": 1,
                "sources": sources,
                "project": {
                    "template_path": str(_TEMPLATE),
                    "generated_project_path": str(project) if keep else None,
                    "device_library": (
                        str(device_library) if keep else device_library.name
                    ),
                    "host_binary": str(host_binary) if keep else host_binary.name,
                    "source_sha256": source_hashes,
                    "commands": [
                        [str(part) for part in command] for command in commands
                    ],
                    "card_info": card_info,
                    "temporary_project_kept": keep,
                    "keep_environment_variable": _KEEP_ENV,
                },
                "layout": {
                    "rows": rows,
                    "columns": columns,
                    "chunk_size": _CHUNK,
                    "stage1_streams": _CHUNKS,
                    "stage2_resident_rebroadcast": True,
                },
            },
        )
    finally:
        if not keep:
            shutil.rmtree(root, ignore_errors=True)


__all__ = ["run_apu_g2_u16_atax"]
