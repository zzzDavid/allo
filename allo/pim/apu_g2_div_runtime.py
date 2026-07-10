# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hardware-only uint16 unsigned division primitive for Gemini-II."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np

from ..spmw_codegen import RunResult
from .apu_g2_recipe import build_apu_g2_u16_div_recipe
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


def _parse_metrics(text: str) -> tuple[dict[str, int], dict[str, float]]:
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
    if f"PASS checked={np.prod(_SHAPE)}" not in text:
        raise RuntimeError("hardware host did not report its all-lane div gate")
    return ticks, host


def run_apu_g2_u16_div(lhs, rhs, *, repetitions=4) -> RunResult:
    """Run unsigned ``lhs // rhs`` on the real APUg2 card.

    Division by zero is rejected before any hardware work is issued.
    """

    lhs = _validate_operand("lhs", lhs)
    rhs = _validate_operand("rhs", rhs)
    if np.any(rhs == 0):
        raise ValueError("APUg2 uint16 division requires all divisors to be nonzero")
    repetitions = _validate_repetitions(repetitions)
    card_info = _require_hardware_stack()
    sources, source_hashes = _source_snapshot()
    recipe = build_apu_g2_u16_div_recipe()

    keep = os.environ.get(_KEEP_ENV) == "1"
    root = Path(tempfile.mkdtemp(prefix="tenon-apu-g2-div-"))
    project = root / "project"
    commands = []
    try:
        shutil.copytree(_TEMPLATE, project)
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

        host_binary = host_build / "tenon_apu_g2_u16_div"
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
        expected = np.floor_divide(lhs, rhs).astype(np.uint16)
        if not np.array_equal(output, expected):
            mismatch = np.argwhere(output != expected)[0]
            index = tuple(int(value) for value in mismatch)
            raise RuntimeError(
                f"APUg2 div result differs from NumPy at {index}: "
                f"hardware={int(output[index])} numpy={int(expected[index])}"
            )

        per_call = {"pipeline": ticks["pipeline"] / repetitions}
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
                "outputs": {"out": output},
                "raw_ticks": ticks,
                "total_ticks": ticks["pipeline"],
                "final_pipeline_ticks": ticks["final_pipeline"],
                "per_call_ticks": per_call,
                "repetitions": repetitions,
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


__all__ = ["run_apu_g2_u16_div"]
