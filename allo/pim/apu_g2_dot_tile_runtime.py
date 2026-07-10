# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hardware-only four-set uint16 dot-product tiles for Gemini-II."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np

from ..spmw_codegen import RunResult
from .apu_g2_gesummv_runtime import _merge_bytes, _sha256, _u16_scalar
from .apu_g2_layout import APUG2DotTilePlan, APUG2_U16_SHAPE
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


def _operand(value, name: str, shape=None) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != np.dtype(np.uint16):
        raise TypeError(f"{name} must have dtype uint16, got {value.dtype}")
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty rank-two array")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return value


def _parse_metrics(
    text: str, epilogue_enabled: bool
) -> tuple[dict[str, int], dict[str, float]]:
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
    if "PASS physical_outputs=262144" not in text:
        raise RuntimeError("hardware host did not report its physical readback gate")
    expected_epilogue = int(epilogue_enabled)
    if f"epilogue_enabled={expected_epilogue}" not in text:
        raise RuntimeError("hardware host reported the wrong epilogue mode")
    return ticks, host


def run_apu_g2_u16_dot_tile(
    left,
    right,
    *,
    accumulator=None,
    alpha=1,
    beta=0,
    repetitions=8,
) -> RunResult:
    """Evaluate independent row-wise dot products modulo 2^16 on APUg2.

    Supplying ``accumulator`` or non-identity coefficients enables the fused
    device epilogue ``alpha * dot + beta * accumulator``.  Calls using only
    ``left``, ``right`` and ``repetitions`` retain the raw-dot behavior.
    """

    left = _operand(left, "left")
    right = _operand(right, "right", left.shape)
    alpha = _u16_scalar(alpha, "alpha")
    beta = _u16_scalar(beta, "beta")
    repetitions = _validate_repetitions(repetitions)
    plan = APUG2DotTilePlan(left.shape[0], left.shape[1])
    epilogue_enabled = accumulator is not None or alpha != 1 or beta != 0
    if accumulator is None:
        accumulator = np.zeros(left.shape[0], dtype=np.uint16)
    else:
        if not isinstance(accumulator, np.ndarray):
            raise TypeError("accumulator must be a NumPy array")
        if accumulator.dtype != np.dtype(np.uint16):
            raise TypeError(
                "accumulator must have dtype uint16, " f"got {accumulator.dtype}"
            )
        if accumulator.shape != (left.shape[0],):
            raise ValueError(
                f"accumulator must have shape {(left.shape[0],)}, "
                f"got {accumulator.shape}"
            )
        if not accumulator.flags.c_contiguous:
            raise ValueError("accumulator must be C-contiguous")

    left_low, left_high, right_low, right_high = plan.pack_operands(left, right)
    left_physical = _merge_bytes(left_low, left_high)
    right_physical = _merge_bytes(right_low, right_high)
    accumulator_physical = np.zeros(APUG2_U16_SHAPE, dtype=np.uint16)
    for row, value in enumerate(accumulator):
        column, mmb_set = plan.physical_coordinate(row)
        accumulator_physical[mmb_set, column] = value

    card_info = _require_hardware_stack()
    sources, source_hashes = _source_snapshot()
    keep = os.environ.get(_KEEP_ENV) == "1"
    root = Path(tempfile.mkdtemp(prefix="tenon-apu-g2-dot-tile-"))
    project = root / "project"
    commands = []
    try:
        shutil.copytree(_TEMPLATE, project)
        left_path = root / "left.bin"
        right_path = root / "right.bin"
        accumulator_path = root / "accumulator.bin"
        output_path = root / "out.bin"
        left_physical.tofile(left_path)
        right_physical.tofile(right_path)
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

        host_binary = host_build / "tenon_apu_g2_u16_dot_tile"
        if not host_binary.is_file():
            raise RuntimeError(f"host build did not produce {host_binary}")
        hardware_command = [
            host_binary,
            left_path,
            right_path,
            accumulator_path,
            output_path,
            str(plan.log_block_size),
            str(alpha),
            str(beta),
            str(int(epilogue_enabled)),
            str(repetitions),
        ]
        commands.append(hardware_command)
        process = _run_command(
            hardware_command, cwd=project, timeout=180, label="hardware execution"
        )
        text = process.stdout + process.stderr
        ticks, host_timings = _parse_metrics(text, epilogue_enabled)

        packed = np.fromfile(output_path, dtype=np.uint16)
        expected_size = int(np.prod(APUG2_U16_SHAPE))
        if packed.size != expected_size:
            raise RuntimeError(
                f"hardware output has {packed.size} values, expected {expected_size}"
            )
        packed = packed.reshape(APUG2_U16_SHAPE)
        observed = plan.unpack_reduction(packed)
        dot = (
            np.sum(
                left.astype(np.uint64) * right.astype(np.uint64),
                axis=1,
                dtype=np.uint64,
            )
            & np.uint64(0xFFFF)
        ).astype(np.uint16)
        if epilogue_enabled:
            expected = (
                (
                    np.uint64(alpha) * dot.astype(np.uint64)
                    + np.uint64(beta) * accumulator.astype(np.uint64)
                )
                & np.uint64(0xFFFF)
            ).astype(np.uint16)
        else:
            expected = dot
        if not np.array_equal(observed, expected):
            index = int(np.flatnonzero(observed != expected)[0])
            raise RuntimeError(
                f"APUg2 dot result differs from NumPy at {index}: "
                f"hardware={int(observed[index])} numpy={int(expected[index])}"
            )

        per_call = {
            "pipeline": ticks["pipeline"] / repetitions,
            "final_pipeline": float(ticks["final_pipeline"]),
        }
        cycles = int(round(per_call["pipeline"]))
        return RunResult(
            cycles,
            text,
            "apu_v2",
            extra={
                "outputs": {"out": observed},
                "raw_ticks": ticks,
                "total_ticks": ticks["pipeline"],
                "final_pipeline_ticks": ticks["final_pipeline"],
                "per_call_ticks": per_call,
                "repetitions": repetitions,
                "epilogue": {
                    "enabled": epilogue_enabled,
                    "alpha": alpha,
                    "beta": beta,
                    "inventory": {
                        "mul": 6 if epilogue_enabled else 0,
                        "shift_left": 4 if epilogue_enabled else 0,
                        "add": 5 if epilogue_enabled else 0,
                    },
                },
                "host_timings_us": host_timings,
                "sources": sources,
                "layout": {
                    "output_extent": plan.output_extent,
                    "reduction_extent": plan.reduction_extent,
                    "padded_output_extent": plan.padded_output_extent,
                    "padded_reduction_extent": plan.padded_reduction_extent,
                    "output_capacity": plan.output_capacity,
                    "log_block_size": plan.log_block_size,
                    "active_sets": sorted(
                        {
                            plan.physical_coordinate(row)[1]
                            for row in range(plan.output_extent)
                        }
                    ),
                },
                "project": {
                    "template_path": str(_TEMPLATE),
                    "generated_project_path": str(project) if keep else None,
                    "device_library": (
                        str(device_library) if keep else device_library.name
                    ),
                    "host_binary": str(host_binary) if keep else host_binary.name,
                    "artifact_sha256": {
                        "device_library": _sha256(device_library),
                        "host_binary": _sha256(host_binary),
                    },
                    "source_sha256": source_hashes,
                    "runtime_sha256": _sha256(Path(__file__)),
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


__all__ = ["run_apu_g2_u16_dot_tile"]
