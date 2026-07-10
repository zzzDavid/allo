# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hardware-only direct-VL64 uint16 GESUMMV runtime for Gemini-II."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np

from ..spmw_codegen import RunResult
from .apu_g2_layout import APUG2ReductionPlan, APUG2_U16_SHAPE, APUG2_VECTOR_LANES
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
    "gemv": "device_gemv_ticks",
    "scale_combine": "device_scale_combine_ticks",
    "pipeline": "device_pipeline_ticks",
    "final_pipeline": "device_final_pipeline_ticks",
}


def _logical_matrix(value, name: str, shape: tuple[int, int] | None = None):
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != np.dtype(np.uint16):
        raise TypeError(f"{name} must have dtype uint16, got {value.dtype}")
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty matrix, got {value.shape}")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return value


def _logical_vector(value, name: str, extent: int):
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != np.dtype(np.uint16):
        raise TypeError(f"{name} must have dtype uint16, got {value.dtype}")
    if value.shape != (extent,):
        raise ValueError(f"{name} must have shape {(extent,)}, got {value.shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return value


def _u16_scalar(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if not 0 <= value <= np.iinfo(np.uint16).max:
        raise ValueError(f"{name} must fit uint16")
    return value


def _merge_bytes(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(
        low.astype(np.uint16) | (high.astype(np.uint16) << np.uint16(8))
    )


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
    if "PASS physical_outputs=524288" not in text:
        raise RuntimeError("hardware host did not report its physical readback gate")
    return ticks, host


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reference(A, B, x, alpha: int, beta: int):
    a64 = A.astype(np.uint64)
    b64 = B.astype(np.uint64)
    x64 = x.astype(np.uint64)
    mask = np.uint64(0xFFFF)
    tmp_a = ((a64 @ x64) & mask).astype(np.uint16)
    tmp_b = ((b64 @ x64) & mask).astype(np.uint16)
    out = (
        (
            np.uint64(alpha) * tmp_a.astype(np.uint64)
            + np.uint64(beta) * tmp_b.astype(np.uint64)
        )
        & mask
    ).astype(np.uint16)
    return tmp_a, tmp_b, out


def run_apu_g2_u16_gesummv(
    A,
    B,
    x,
    *,
    alpha=5,
    beta=4,
    repetitions=64,
) -> RunResult:
    """Run ``alpha*(A@x) + beta*(B@x)`` modulo 2^16 on APUg2.

    Logical operands are packed through :class:`APUG2ReductionPlan`; callers
    never observe the padded 65,536-lane carrier.
    """

    A = _logical_matrix(A, "A")
    B = _logical_matrix(B, "B", A.shape)
    output_extent, reduction_extent = A.shape
    x = _logical_vector(x, "x", reduction_extent)
    alpha = _u16_scalar(alpha, "alpha")
    beta = _u16_scalar(beta, "beta")
    repetitions = _validate_repetitions(repetitions)

    plan = APUG2ReductionPlan(output_extent, reduction_extent, stream_extent=2)
    if plan.log_block_size > 8:
        raise ValueError(
            "direct uint16 VL64 SUM supports padded reduction extents up to 256"
        )

    matrices_low, matrices_high = plan.pack_matrices(np.stack((A, B), axis=0))
    matrices = _merge_bytes(matrices_low, matrices_high)
    x_low, x_high = plan.pack_broadcast_vector(x)
    x_physical = _merge_bytes(x_low, x_high)
    scalars = np.zeros(APUG2_U16_SHAPE, dtype=np.uint16)
    scalars[0].fill(alpha)
    scalars[1].fill(beta)

    card_info = _require_hardware_stack()
    sources, source_hashes = _source_snapshot()
    keep = os.environ.get(_KEEP_ENV) == "1"
    root = Path(tempfile.mkdtemp(prefix="tenon-apu-g2-gesummv-"))
    project = root / "project"
    commands = []
    try:
        shutil.copytree(_TEMPLATE, project)
        matrices_path = root / "matrices.bin"
        x_path = root / "x.bin"
        scalars_path = root / "scalars.bin"
        tmp_path = root / "tmp.bin"
        out_path = root / "out.bin"
        matrices.tofile(matrices_path)
        x_physical.tofile(x_path)
        scalars.tofile(scalars_path)

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

        host_binary = host_build / "tenon_apu_g2_u16_gesummv"
        if not host_binary.is_file():
            raise RuntimeError(f"host build did not produce {host_binary}")
        hardware_command = [
            host_binary,
            matrices_path,
            x_path,
            scalars_path,
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

        packed_tmp = np.fromfile(tmp_path, dtype=np.uint16)
        packed_out = np.fromfile(out_path, dtype=np.uint16)
        expected_size = int(np.prod(APUG2_U16_SHAPE))
        if packed_tmp.size != expected_size or packed_out.size != expected_size:
            raise RuntimeError("hardware returned a malformed physical output pack")
        packed_tmp = packed_tmp.reshape(APUG2_U16_SHAPE)
        packed_out = packed_out.reshape(APUG2_U16_SHAPE)
        tmp_a, tmp_b = plan.unpack_reductions(packed_tmp)
        observed_out = plan.unpack_reduction(packed_out, stream=0)
        expected_a, expected_b, expected_out = _reference(A, B, x, alpha, beta)
        for name, observed, expected in (
            ("tmp_a", tmp_a, expected_a),
            ("tmp_b", tmp_b, expected_b),
            ("out", observed_out, expected_out),
        ):
            if not np.array_equal(observed, expected):
                index = int(np.flatnonzero(observed != expected)[0])
                raise RuntimeError(
                    f"APUg2 {name} differs from NumPy at {index}: "
                    f"hardware={int(observed[index])} numpy={int(expected[index])}"
                )

        per_call = {
            name: value / repetitions
            for name, value in ticks.items()
            if name != "final_pipeline"
        }
        per_call["final_pipeline"] = float(ticks["final_pipeline"])
        cycles = int(round(per_call["pipeline"]))
        project_metadata = {
            "template_path": str(_TEMPLATE),
            "generated_project_path": str(project) if keep else None,
            "device_library": str(device_library) if keep else device_library.name,
            "host_binary": str(host_binary) if keep else host_binary.name,
            "artifact_sha256": {
                "device_library": _sha256(device_library),
                "host_binary": _sha256(host_binary),
            },
            "source_sha256": source_hashes,
            "runtime_sha256": _sha256(Path(__file__)),
            "commands": [[str(part) for part in command] for command in commands],
            "card_info": card_info,
            "temporary_project_kept": keep,
            "keep_environment_variable": _KEEP_ENV,
        }
        return RunResult(
            cycles,
            text,
            "apu_v2",
            extra={
                "outputs": {"tmp_a": tmp_a, "tmp_b": tmp_b, "out": observed_out},
                "raw_ticks": ticks,
                "phase_ticks": {
                    "gemv": ticks["gemv"],
                    "scale_combine": ticks["scale_combine"],
                },
                "total_ticks": ticks["pipeline"],
                "final_pipeline_ticks": ticks["final_pipeline"],
                "per_call_ticks": per_call,
                "repetitions": repetitions,
                "host_timings_us": host_timings,
                "sources": sources,
                "project": project_metadata,
                "layout": {
                    "output_extent": plan.output_extent,
                    "reduction_extent": plan.reduction_extent,
                    "padded_reduction_extent": plan.padded_reduction_extent,
                    "log_block_size": plan.log_block_size,
                    "streams": 2,
                    "physical_vector_lanes": APUG2_VECTOR_LANES,
                },
            },
        )
    finally:
        if not keep:
            shutil.rmtree(root, ignore_errors=True)


__all__ = ["run_apu_g2_u16_gesummv"]
