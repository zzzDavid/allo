# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-card runtime for LinearLayout-aware composed Gemini-II dots."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np

from ..spmw_codegen import RunResult
from .apu_g2_composed_contraction import APUG2ComposedContractionProgram
from .apu_g2_composed_layout import (
    APUG2_COMPOSED_EPILOGUE_ABI,
    APUG2ComposedHostFiles,
    build_apu_g2_composed_host_command,
    expected_apu_g2_composed_output,
    gather_apu_g2_composed_output,
    pack_apu_g2_composed_auxiliary,
    pack_apu_g2_composed_operand,
)
from .apu_g2_composed_program import build_apu_g2_composed_recipe
from .apu_g2_layout import APUG2_U16_SHAPE
from .apu_g2_runtime import (
    _BUILD_JOBS_ENV,
    _HOST_FIELDS,
    _KEEP_ENV,
    _TEMPLATE,
    _require_hardware_stack,
    _run_command,
    _source_snapshot,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_text(text: str, field: str) -> str:
    matches = re.findall(
        rf"^{re.escape(field)}=([^\r\n]+)\s*$",
        text,
        re.MULTILINE,
    )
    if len(matches) != 1:
        raise RuntimeError(f"composed host returned {len(matches)} values for {field}")
    return matches[0]


def _parse_metrics(
    text: str,
    program: APUG2ComposedContractionProgram,
    device_library: Path,
) -> tuple[dict[str, int], dict[str, float]]:
    """Parse and authenticate the complete generic host attestation."""

    if not device_library.is_absolute():
        raise TypeError("device-library attestation requires an absolute path")
    lhs_type, rhs_type = program.input_types
    auxiliary_type = program.auxiliary_type
    exact = {
        "target": "hardware",
        "device_library": str(device_library),
        "task_status": "0",
        "timed_scope": "composed_dot_vl64_pipeline",
        "completion_barrier_included": "1",
        "independent_final_correctness_call": "1",
        "layout_fingerprint": program.layout_manifest()["fingerprint"],
        "epilogue_mode": str(APUG2_COMPOSED_EPILOGUE_ABI[program.epilogue.mode]),
        "lhs_bits": str(lhs_type.bits),
        "lhs_signed": str(int(lhs_type.signed)),
        "rhs_bits": str(rhs_type.bits),
        "rhs_signed": str(int(rhs_type.signed)),
        "dot_bits": str(program.dot_type.bits),
        "auxiliary_bits": str(1 if auxiliary_type is None else auxiliary_type.bits),
        "auxiliary_signed": str(
            0 if auxiliary_type is None else int(auxiliary_type.signed)
        ),
        "out_bits": str(program.output_type.bits),
        "out_signed": str(int(program.output_type.signed)),
        "log_reduction": str(program.reduction_extent.bit_length() - 1),
        "repetitions": str(program.repetitions),
    }
    for field, expected in exact.items():
        actual = _unique_text(text, field)
        if actual != expected:
            raise RuntimeError(
                f"composed host {field}={actual!r}, expected {expected!r}"
            )
    if _unique_text(text, "PASS physical_outputs") != str(
        int(np.prod(APUG2_U16_SHAPE))
    ):
        raise RuntimeError("composed host did not attest its full output carrier")

    ticks = {}
    for name, field in (
        ("pipeline", "device_pipeline_ticks"),
        ("final_pipeline", "device_final_pipeline_ticks"),
    ):
        value = _unique_text(text, field)
        if not value.isdecimal() or int(value) <= 0:
            raise RuntimeError(f"composed host returned invalid {field}")
        ticks[name] = int(value)
    host = {}
    for field in _HOST_FIELDS:
        value = _unique_text(text, field)
        try:
            parsed = float(value)
        except ValueError as error:
            raise RuntimeError(f"composed host returned invalid {field}") from error
        if not math.isfinite(parsed) or parsed < 0:
            raise RuntimeError(f"composed host returned invalid {field}")
        host[field.removesuffix("_us")] = parsed
    return ticks, host


def run_apu_g2_composed(
    program: APUG2ComposedContractionProgram,
    lhs,
    rhs,
    *,
    accumulator=None,
) -> RunResult:
    """Build, execute, and independently verify one composed-dot program."""

    if not isinstance(program, APUG2ComposedContractionProgram):
        raise TypeError("program must be APUG2ComposedContractionProgram")
    physical_lhs = pack_apu_g2_composed_operand(program, lhs, 0)
    physical_rhs = pack_apu_g2_composed_operand(program, rhs, 1)
    physical_auxiliary = pack_apu_g2_composed_auxiliary(program, accumulator)
    oracle = expected_apu_g2_composed_output(
        program,
        lhs,
        rhs,
        accumulator=accumulator,
    )
    recipe = build_apu_g2_composed_recipe(program)
    card_info = _require_hardware_stack()
    sources, source_hashes = _source_snapshot()
    keep = os.environ.get(_KEEP_ENV) == "1"
    root = Path(tempfile.mkdtemp(prefix="tenon-apu-g2-composed-"))
    project = root / "project"
    commands = []
    try:
        shutil.copytree(_TEMPLATE, project)
        lhs_path = root / "lhs.bin"
        rhs_path = root / "rhs.bin"
        auxiliary_path = None if physical_auxiliary is None else root / "auxiliary.bin"
        output_path = root / "out.bin"
        physical_lhs.tofile(lhs_path)
        physical_rhs.tofile(rhs_path)
        if physical_auxiliary is not None:
            physical_auxiliary.tofile(auxiliary_path)

        jobs_text = os.environ.get(_BUILD_JOBS_ENV)
        jobs = int(jobs_text) if jobs_text is not None else min(os.cpu_count() or 1, 8)
        if jobs <= 0:
            raise ValueError(f"{_BUILD_JOBS_ENV} must be positive")

        device_build = project / "build" / "device"
        host_build = project / "build" / "host"
        device_configure = [
            "cmake",
            "-S",
            project / "device",
            "-B",
            device_build,
        ]
        device_compile = [
            "cmake",
            "--build",
            device_build,
            "-j",
            str(jobs),
        ]
        commands.extend((device_configure, device_compile))
        _run_command(
            device_configure,
            cwd=project,
            timeout=120,
            label="composed device configure",
        )
        _run_command(
            device_compile,
            cwd=project,
            timeout=600,
            label="composed device build",
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
        host_compile = [
            "cmake",
            "--build",
            host_build,
            "--target",
            "tenon_apu_g2_composed_dot",
            "-j",
            str(jobs),
        ]
        commands.extend((host_configure, host_compile))
        _run_command(
            host_configure,
            cwd=project,
            timeout=120,
            label="composed host configure",
        )
        _run_command(
            host_compile,
            cwd=project,
            timeout=600,
            label="composed host build",
        )
        host_binary = host_build / "tenon_apu_g2_composed_dot"
        if not host_binary.is_file():
            raise RuntimeError(f"host build did not produce {host_binary}")

        files = APUG2ComposedHostFiles(
            lhs_path,
            rhs_path,
            auxiliary_path,
            output_path,
        )
        hardware_command = build_apu_g2_composed_host_command(
            program,
            host_binary,
            files,
        )
        commands.append(hardware_command)
        process = _run_command(
            hardware_command,
            cwd=project,
            timeout=180,
            label="composed hardware execution",
        )
        text = process.stdout + process.stderr
        ticks, host_timings = _parse_metrics(text, program, device_library)

        physical_output = np.fromfile(
            output_path,
            dtype=program.output_type.numpy_dtype,
        )
        expected_elements = int(np.prod(APUG2_U16_SHAPE))
        if physical_output.size != expected_elements:
            raise RuntimeError(
                "composed host returned "
                f"{physical_output.size} values, expected {expected_elements}"
            )
        physical_output = physical_output.reshape(APUG2_U16_SHAPE)
        logical_output = gather_apu_g2_composed_output(program, physical_output)
        if not np.array_equal(logical_output, oracle):
            mismatch = np.argwhere(logical_output != oracle)[0]
            index = tuple(int(item) for item in mismatch)
            raise RuntimeError(
                "APUg2 composed result differs from NumPy at "
                f"{index}: hardware={int(logical_output[index])}, "
                f"numpy={int(oracle[index])}"
            )

        per_call = {
            "pipeline": ticks["pipeline"] / program.repetitions,
            "final_pipeline": float(ticks["final_pipeline"]),
        }
        return RunResult(
            int(round(per_call["pipeline"])),
            text,
            "apu_v2",
            extra={
                "outputs": {
                    "out": logical_output,
                    "physical_out": physical_output,
                },
                "oracle": oracle,
                "raw_ticks": ticks,
                "total_ticks": ticks["pipeline"],
                "final_pipeline_ticks": ticks["final_pipeline"],
                "per_call_ticks": per_call,
                "repetitions": program.repetitions,
                "host_timings_us": host_timings,
                "program": program.manifest(),
                "program_fingerprint": program.structural_fingerprint,
                "layout": program.layout_manifest(),
                "layout_fingerprint": program.layout_manifest()["fingerprint"],
                "recipe": recipe.manifest(),
                "recipe_fingerprint": recipe.structural_fingerprint,
                "operation_inventory": dict(recipe.inventory),
                "barrier_inventory": {
                    "SEU_BARRIER": recipe.inventory.get("SEU_BARRIER", 0)
                },
                "sources": sources,
                "project": {
                    "template_path": str(_TEMPLATE),
                    "generated_project_path": str(project) if keep else None,
                    "device_library": (
                        str(device_library) if keep else device_library.name
                    ),
                    "host_binary": (str(host_binary) if keep else host_binary.name),
                    "artifact_sha256": {
                        "device_library": _sha256(device_library),
                        "host_binary": _sha256(host_binary),
                    },
                    "source_sha256": source_hashes,
                    "runtime_sha256": _sha256(Path(__file__)),
                    "python_implementation_sha256": {
                        name: _sha256(Path(__file__).with_name(filename))
                        for name, filename in (
                            (
                                "contraction",
                                "apu_g2_composed_contraction.py",
                            ),
                            ("layout", "apu_g2_composed_layout.py"),
                            ("program", "apu_g2_composed_program.py"),
                            ("runtime", "apu_g2_composed_runtime.py"),
                        )
                    },
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


__all__ = ["run_apu_g2_composed"]
