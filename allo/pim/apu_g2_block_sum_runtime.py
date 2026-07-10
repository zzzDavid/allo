# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hardware-only uint16 block-sum primitive for Gemini-II."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np

from ..spmw_codegen import RunResult
from .apu_g2_recipe import build_apu_g2_u16_block_sum_recipe
from .apu_g2_runtime import (
    _BUILD_JOBS_ENV,
    _HOST_FIELDS,
    _KEEP_ENV,
    _SHAPE,
    _TEMPLATE,
    _require_hardware_stack,
    _run_command,
    _source_snapshot,
    _validate_operand,
    _validate_repetitions,
)


def _validate_log_block_size(log_block_size) -> int:
    if isinstance(log_block_size, (bool, np.bool_)) or not isinstance(
        log_block_size, (int, np.integer)
    ):
        raise TypeError("log_block_size must be an integer")
    log_block_size = int(log_block_size)
    if not 0 <= log_block_size <= 8:
        raise ValueError("log_block_size must be in [0, 8]")
    return log_block_size


def _expected_block_sums(src: np.ndarray, log_block_size: int) -> np.ndarray:
    block_size = 1 << log_block_size
    blocks_per_group = 4096 // block_size
    grouped = src.reshape(4, 16, 4096)
    sums = (
        grouped.reshape(4, 16, blocks_per_group, block_size)
        .astype(np.uint64)
        .sum(
            axis=3,
            dtype=np.uint64,
        )
    )
    return (sums & np.uint64(0xFFFF)).astype(np.uint16)


def _observed_block_sums(output: np.ndarray, log_block_size: int) -> np.ndarray:
    block_size = 1 << log_block_size
    grouped = output.reshape(4, 16, 4096)
    return grouped[:, :, ::block_size].copy()


def _parse_metrics(
    text: str, log_block_size: int
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
    checked = int(np.prod(_SHAPE)) // (1 << log_block_size)
    if f"PASS checked={checked}" not in text:
        raise RuntimeError("hardware host did not report its block-sum gate")
    return ticks, host


def run_apu_g2_u16_block_sum(src, *, log_block_size=8, repetitions=8) -> RunResult:
    """Run uint16 block sums on the real APUg2 card.

    The returned ``block_sums`` array has shape ``(4, 16, 4096/block_size)``.
    It contains the meaningful hardware result lanes: the first lane of every
    block, truncated to uint16.
    """

    src = _validate_operand("src", src)
    log_block_size = _validate_log_block_size(log_block_size)
    repetitions = _validate_repetitions(repetitions)
    card_info = _require_hardware_stack()
    sources, source_hashes = _source_snapshot()
    recipe = build_apu_g2_u16_block_sum_recipe(log_block_size)

    keep = os.environ.get(_KEEP_ENV) == "1"
    root = Path(tempfile.mkdtemp(prefix="tenon-apu-g2-block-sum-"))
    project = root / "project"
    commands = []
    try:
        shutil.copytree(_TEMPLATE, project)
        src_path = root / "src.bin"
        output_path = root / "out.bin"
        src.tofile(src_path)

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

        host_binary = host_build / "tenon_apu_g2_u16_block_sum"
        if not host_binary.is_file():
            raise RuntimeError(f"host build did not produce {host_binary}")
        hardware_command = [
            host_binary,
            src_path,
            output_path,
            str(log_block_size),
            str(repetitions),
        ]
        commands.append(hardware_command)
        process = _run_command(
            hardware_command, cwd=project, timeout=120, label="hardware execution"
        )
        text = process.stdout + process.stderr
        ticks, host_timings = _parse_metrics(text, log_block_size)

        output = np.fromfile(output_path, dtype=np.uint16)
        if output.size != np.prod(_SHAPE):
            raise RuntimeError(
                f"hardware output has {output.size} elements, expected {np.prod(_SHAPE)}"
            )
        output = output.reshape(_SHAPE)
        observed = _observed_block_sums(output, log_block_size)
        expected = _expected_block_sums(src, log_block_size)
        if not np.array_equal(observed, expected):
            mismatch = np.argwhere(observed != expected)[0]
            index = tuple(int(item) for item in mismatch)
            raise RuntimeError(
                f"APUg2 block sum differs from NumPy at {index}: "
                f"hardware={int(observed[index])} numpy={int(expected[index])}"
            )

        per_call = {"pipeline": ticks["pipeline"] / repetitions}
        per_call["final_pipeline"] = float(ticks["final_pipeline"])
        cycles = int(round(per_call["pipeline"]))
        timing_text = "\n".join(
            f"{name}_us={item:.3f}" for name, item in host_timings.items()
        )
        project_metadata = {
            "template_path": str(_TEMPLATE),
            "generated_project_path": str(project) if keep else None,
            "device_library": str(device_library) if keep else device_library.name,
            "source_sha256": source_hashes,
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
                "outputs": {"out": output, "block_sums": observed},
                "raw_ticks": ticks,
                "total_ticks": ticks["pipeline"],
                "final_pipeline_ticks": ticks["final_pipeline"],
                "per_call_ticks": per_call,
                "repetitions": repetitions,
                "log_block_size": log_block_size,
                "host_timings_us": host_timings,
                "host_timing_text": timing_text,
                "vectorization_certificate": recipe.certificate.manifest(),
                "recipe": recipe.manifest(),
                "sources": sources,
                "project": project_metadata,
            },
        )
    finally:
        if not keep:
            shutil.rmtree(root, ignore_errors=True)


__all__ = ["run_apu_g2_u16_block_sum"]
