# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent column-batched uint16 GEMM runtime for Gemini-II."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np

from ..spmw_codegen import RunResult
from .apu_g2_gesummv_runtime import _sha256, _u16_scalar
from .apu_g2_runtime import (
    APUG2RuntimeArtifact,
    _BUILD_JOBS_ENV,
    _HOST_FIELDS,
    _KEEP_ENV,
    _require_hardware_stack,
    _run_command,
    freeze_apu_g2_runtime_artifact,
)


APUG2_U16_GEMM_CHUNK = 128
APUG2_U16_GEMM_BMAX = 31


def _build_manifest():
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
            "--target",
            "tenon_apu_g2_u16_gemm",
            "-j",
            "{jobs}",
        ],
        "jobs_policy": "TENON_APU_G2_BUILD_JOBS_or_min_cpu_count_8",
        "device_library": "build/device/bin/tenon_apu_g2_tasks.update.bin",
        "host_binary": "build/host/tenon_apu_g2_u16_gemm",
    }


def _abi_manifest():
    return {
        "schema": "apu-g2-u16-gemm-abi-v1",
        "inputs": [
            ["lhs", "uint16", ["M", "K"], "c_contiguous"],
            ["rhs", "uint16", ["K", "N"], "c_contiguous"],
            ["accumulator", "uint16", ["M", "N"], "c_contiguous"],
        ],
        "output": ["out", "uint16", ["M", "N"], "c_contiguous"],
        "argv": [
            "lhs_path",
            "rhs_path",
            "accumulator_path",
            "output_path",
            "M",
            "K",
            "N",
            "alpha_u16",
            "beta_u16",
            "batch_columns",
        ],
        "batch_columns": [1, APUG2_U16_GEMM_BMAX],
        "reduction_tile": APUG2_U16_GEMM_CHUNK,
        "arithmetic": "modulo_2^16",
    }


def freeze_apu_g2_u16_gemm_runtime_artifact() -> APUG2RuntimeArtifact:
    """Freeze the complete project, driver, build, ABI, and executor contract."""

    return freeze_apu_g2_runtime_artifact(
        run_apu_g2_u16_gemm,
        build_manifest=_build_manifest(),
        abi_manifest=_abi_manifest(),
        dependency_modules=("allo.pim.apu_g2_gesummv_runtime",),
    )


def _matrix(value, name, shape=None):
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != np.dtype(np.uint16):
        raise TypeError(f"{name} must have dtype uint16, got {value.dtype}")
    if value.ndim != 2 or not all(value.shape):
        raise ValueError(f"{name} must be a nonempty rank-two array")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return value


def _batch_columns(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("batch_columns must be an integer")
    value = int(value)
    if not 1 <= value <= APUG2_U16_GEMM_BMAX:
        raise ValueError(f"batch_columns must be in [1, {APUG2_U16_GEMM_BMAX}]")
    return value


def _parse_metrics(text):
    integers = {}
    for name in ("device_pipeline_ticks", "hardware_tasks", "weight_uploads"):
        matches = re.findall(rf"^{name}=(\d+)\s*$", text, re.MULTILINE)
        if len(matches) != 1:
            raise RuntimeError(f"hardware output has {len(matches)} values for {name}")
        integers[name] = int(matches[0])
        if integers[name] <= 0:
            raise RuntimeError(f"hardware returned a nonpositive {name}")

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
    if "PASS logical_outputs=" not in text:
        raise RuntimeError("hardware host did not report its logical output gate")
    return integers, host


def run_apu_g2_u16_gemm(
    lhs,
    rhs,
    accumulator,
    *,
    alpha=1,
    beta=1,
    batch_columns=APUG2_U16_GEMM_BMAX,
    runtime_artifact=None,
):
    """Run ``alpha * lhs @ rhs + beta * accumulator`` modulo 2^16."""

    lhs = _matrix(lhs, "lhs")
    rhs = _matrix(rhs, "rhs")
    if lhs.shape[1] != rhs.shape[0]:
        raise ValueError(f"GEMM reduction mismatch: lhs {lhs.shape}, rhs {rhs.shape}")
    accumulator = _matrix(accumulator, "accumulator", (lhs.shape[0], rhs.shape[1]))
    if lhs.shape[0] > 65_536:
        raise ValueError("GEMM output rows exceed one VL64 vector")
    alpha = _u16_scalar(alpha, "alpha")
    beta = _u16_scalar(beta, "beta")
    batch_columns = _batch_columns(batch_columns)

    if runtime_artifact is None:
        runtime_artifact = freeze_apu_g2_u16_gemm_runtime_artifact()
    if not isinstance(runtime_artifact, APUG2RuntimeArtifact):
        raise TypeError("runtime_artifact must be an APUG2RuntimeArtifact")
    runtime_artifact.assert_current(run_apu_g2_u16_gemm)
    card_info = _require_hardware_stack()
    sources = {
        relative: data.decode("utf-8")
        for relative, data in runtime_artifact.project_files
    }
    source_hashes = dict(runtime_artifact.source_hashes)
    keep = os.environ.get(_KEEP_ENV) == "1"
    root = Path(tempfile.mkdtemp(prefix="tenon-apu-g2-u16-gemm-"))
    project = root / "project"
    commands = []
    try:
        runtime_artifact.write_project(project)
        lhs_path = root / "lhs.bin"
        rhs_path = root / "rhs.bin"
        accumulator_path = root / "accumulator.bin"
        output_path = root / "out.bin"
        lhs.tofile(lhs_path)
        rhs.tofile(rhs_path)
        accumulator.tofile(accumulator_path)

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
        _run_command(device_compile, cwd=project, timeout=600, label="device compile")

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
        host_compile = [
            "cmake",
            "--build",
            host_build,
            "--target",
            "tenon_apu_g2_u16_gemm",
            "-j",
            str(jobs),
        ]
        commands.extend((host_configure, host_compile))
        _run_command(host_configure, cwd=project, timeout=120, label="host configure")
        _run_command(host_compile, cwd=project, timeout=600, label="host compile")

        host_binary = host_build / "tenon_apu_g2_u16_gemm"
        hardware_command = [
            host_binary,
            lhs_path,
            rhs_path,
            accumulator_path,
            output_path,
            str(lhs.shape[0]),
            str(lhs.shape[1]),
            str(rhs.shape[1]),
            str(alpha),
            str(beta),
            str(batch_columns),
        ]
        commands.append(hardware_command)
        process = _run_command(
            hardware_command,
            cwd=project,
            timeout=3600,
            label="hardware uint16 GEMM",
        )
        text = process.stdout + process.stderr
        counters, host_timings = _parse_metrics(text)

        observed = np.fromfile(output_path, dtype=np.uint16)
        expected_shape = (lhs.shape[0], rhs.shape[1])
        if observed.size != int(np.prod(expected_shape)):
            raise RuntimeError("hardware returned a malformed GEMM output")
        observed = observed.reshape(expected_shape)
        expected = (
            np.uint64(alpha) * (lhs.astype(np.uint64) @ rhs.astype(np.uint64))
            + np.uint64(beta) * accumulator.astype(np.uint64)
        ) & np.uint64(0xFFFF)
        expected = expected.astype(np.uint16)
        if not np.array_equal(observed, expected):
            index = tuple(int(i) for i in np.argwhere(observed != expected)[0])
            raise RuntimeError(
                "APUg2 uint16 GEMM differs from NumPy at "
                f"{index}: hardware={int(observed[index])} "
                f"numpy={int(expected[index])}"
            )

        return RunResult(
            counters["device_pipeline_ticks"],
            text,
            "apu_v2",
            extra={
                "outputs": {"out": observed},
                "host_timings_us": host_timings,
                "hardware_tasks": counters["hardware_tasks"],
                "weight_uploads": counters["weight_uploads"],
                "schedule": {
                    "kind": "column_batched_u16_gemm",
                    "batch_columns": batch_columns,
                    "reduction_tile": APUG2_U16_GEMM_CHUNK,
                    "resident_accumulator": True,
                    "contiguous_readback": True,
                },
                "sources": sources,
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
                    "source_fingerprint": runtime_artifact.source_fingerprint,
                    "runtime_sha256": _sha256(Path(__file__)),
                    "commands": [
                        [str(part) for part in command] for command in commands
                    ],
                    "card_info": card_info,
                    "promotion_platform_fingerprint": (
                        runtime_artifact.current_platform_fingerprint()
                    ),
                    "temporary_project_kept": keep,
                    "keep_environment_variable": _KEEP_ENV,
                },
            },
        )
    finally:
        if not keep:
            shutil.rmtree(root, ignore_errors=True)


__all__ = [
    "APUG2_U16_GEMM_BMAX",
    "APUG2_U16_GEMM_CHUNK",
    "freeze_apu_g2_u16_gemm_runtime_artifact",
    "run_apu_g2_u16_gemm",
]
