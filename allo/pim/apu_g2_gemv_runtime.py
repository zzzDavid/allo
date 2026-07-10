# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hardware-only direct-VL64 accumulated uint16 GEMV runtime."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np

from ..spmw_codegen import RunResult
from .apu_g2_layout import APUG2ReductionPlan, APUG2_U16_SHAPE
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


_TICK_FIELDS = {
    "product": "device_product_ticks",
    "pipeline": "device_pipeline_ticks",
    "final_pipeline": "device_final_pipeline_ticks",
}


def _matrix(value, name: str):
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != np.dtype(np.uint16):
        raise TypeError(f"{name} must have dtype uint16, got {value.dtype}")
    if value.ndim != 2 or 0 in value.shape:
        raise ValueError(f"{name} must be a nonempty matrix, got {value.shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return value


def _vector(value, name: str, extent: int):
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != np.dtype(np.uint16):
        raise TypeError(f"{name} must have dtype uint16, got {value.dtype}")
    if value.shape != (extent,):
        raise ValueError(f"{name} must have shape {(extent,)}, got {value.shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return value


def _merge_bytes(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(
        low.astype(np.uint16) | (high.astype(np.uint16) << np.uint16(8))
    )


def _pack_accumulator(plan: APUG2ReductionPlan, accumulator) -> np.ndarray:
    packed = np.zeros(APUG2_U16_SHAPE, dtype=np.uint16)
    for row, value in enumerate(accumulator):
        column, mmb_set = plan.physical_coordinate(row, 0)
        packed[mmb_set, column] = value
    return packed


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
    if "PASS gemv_physical_outputs=524288" not in text:
        raise RuntimeError("hardware host did not report its GEMV readback gate")
    return ticks, host


def _reference(matrix, vector, accumulator):
    product = (
        (matrix.astype(np.uint64) @ vector.astype(np.uint64)) & np.uint64(0xFFFF)
    ).astype(np.uint16)
    out = (
        (product.astype(np.uint64) + accumulator.astype(np.uint64)) & np.uint64(0xFFFF)
    ).astype(np.uint16)
    return product, out


def run_apu_g2_u16_gemv(
    matrix,
    vector,
    accumulator,
    *,
    repetitions=8,
) -> RunResult:
    """Execute ``accumulator + matrix @ vector`` modulo 2^16 on core 0."""

    matrix = _matrix(matrix, "matrix")
    rows, reduction = matrix.shape
    vector = _vector(vector, "vector", reduction)
    accumulator = _vector(accumulator, "accumulator", rows)
    repetitions = _validate_repetitions(repetitions)
    plan = APUG2ReductionPlan(rows, reduction, stream_extent=1)
    if plan.log_block_size > 8:
        raise ValueError(
            "direct uint16 VL64 SUM supports padded reduction extents up to 256"
        )

    matrix_low, matrix_high = plan.pack_matrix(matrix)
    matrix_physical = _merge_bytes(matrix_low, matrix_high)
    x_low, x_high = plan.pack_broadcast_vector(vector)
    x_physical = _merge_bytes(x_low, x_high)
    accumulator_physical = _pack_accumulator(plan, accumulator)

    card_info = _require_hardware_stack()
    sources, source_hashes = _source_snapshot()
    keep = os.environ.get(_KEEP_ENV) == "1"
    root = Path(tempfile.mkdtemp(prefix="tenon-apu-g2-gemv-"))
    project = root / "project"
    commands = []
    try:
        shutil.copytree(_TEMPLATE, project)
        matrix_path = root / "matrix.bin"
        x_path = root / "x.bin"
        accumulator_path = root / "accumulator.bin"
        tmp_path = root / "tmp.bin"
        out_path = root / "out.bin"
        matrix_physical.tofile(matrix_path)
        x_physical.tofile(x_path)
        accumulator_physical.tofile(accumulator_path)

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
        host_binary = host_build / "tenon_apu_g2_u16_gemv"
        if not host_binary.is_file():
            raise RuntimeError(f"host build did not produce {host_binary}")

        hardware_command = [
            host_binary,
            matrix_path,
            x_path,
            accumulator_path,
            tmp_path,
            out_path,
            str(plan.log_block_size),
            str(repetitions),
        ]
        commands.append(hardware_command)
        process = _run_command(
            hardware_command, cwd=project, timeout=180, label="hardware execution"
        )
        text = process.stdout + process.stderr
        ticks, host_timings = _parse_metrics(text)

        expected_size = int(np.prod(APUG2_U16_SHAPE))
        packed_tmp = np.fromfile(tmp_path, dtype=np.uint16)
        packed_out = np.fromfile(out_path, dtype=np.uint16)
        if packed_tmp.size != expected_size or packed_out.size != expected_size:
            raise RuntimeError("hardware returned a malformed GEMV output pack")
        packed_tmp = packed_tmp.reshape(APUG2_U16_SHAPE)
        packed_out = packed_out.reshape(APUG2_U16_SHAPE)
        observed_tmp = plan.unpack_reduction(packed_tmp)
        observed_out = plan.unpack_reduction(packed_out)
        expected_tmp, expected_out = _reference(matrix, vector, accumulator)
        for name, observed, expected in (
            ("tmp", observed_tmp, expected_tmp),
            ("out", observed_out, expected_out),
        ):
            if not np.array_equal(observed, expected):
                index = int(np.flatnonzero(observed != expected)[0])
                raise RuntimeError(
                    f"APUg2 GEMV {name} differs from NumPy at {index}: "
                    f"hardware={int(observed[index])} numpy={int(expected[index])}"
                )

        per_call = {
            name: value / repetitions
            for name, value in ticks.items()
            if name != "final_pipeline"
        }
        per_call["final_pipeline"] = float(ticks["final_pipeline"])
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
                    "output_extent": rows,
                    "reduction_extent": reduction,
                    "padded_reduction_extent": plan.padded_reduction_extent,
                    "log_block_size": plan.log_block_size,
                    "streams": 1,
                },
            },
        )
    finally:
        if not keep:
            shutil.rmtree(root, ignore_errors=True)


__all__ = ["run_apu_g2_u16_gemv"]
