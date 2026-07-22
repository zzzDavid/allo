#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compile and measure the typed Tenon equivalents of 14 CENT AiM cases.

The campaign deliberately invokes the timing simulator itself instead of the
normal Tenon runtime wrapper.  That makes the exact trace, immutable container
image, command line, raw process streams, and exit status explicit evidence.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
from typing import Any

try:
    from .evidence import (
        COMPARISON_FIELDS,
        EXPECTED_CASE_IDS,
        EXPECTED_SIMULATOR_COMMIT,
        EvidenceError,
        canonical_json_bytes,
        command_shape_signature,
        comparison_row,
        comparison_totals,
        jsonable,
        opcode_active_channel_coverage,
        physical_work_comparison,
        parse_simulator_stdout,
        read_json,
        reconcile_request_counts,
        render_comparison_markdown,
        render_reuse_calibration_markdown,
        sha256_bytes,
        sha256_file,
        validate_frozen_vendor_trace,
        validate_trace_text,
        verify_checksums,
        write_checksums,
        write_json,
        write_text,
    )
except ImportError:  # Support direct execution from this directory.
    from evidence import (  # type: ignore
        COMPARISON_FIELDS,
        EXPECTED_CASE_IDS,
        EXPECTED_SIMULATOR_COMMIT,
        EvidenceError,
        canonical_json_bytes,
        command_shape_signature,
        comparison_row,
        comparison_totals,
        jsonable,
        opcode_active_channel_coverage,
        physical_work_comparison,
        parse_simulator_stdout,
        read_json,
        reconcile_request_counts,
        render_comparison_markdown,
        render_reuse_calibration_markdown,
        sha256_bytes,
        sha256_file,
        validate_frozen_vendor_trace,
        validate_trace_text,
        verify_checksums,
        write_checksums,
        write_json,
        write_text,
    )

try:
    from .vendor_contract import CENT_SOURCE_COMMIT, VENDOR_TRACE_CONTRACTS
except ImportError:  # Support direct execution from this directory.
    from vendor_contract import (  # type: ignore
        CENT_SOURCE_COMMIT,
        VENDOR_TRACE_CONTRACTS,
    )


DEFAULT_IMAGE = "aim-simulator-build:latest"
DEFAULT_CONFIG_RELATIVE = Path("test/example.yaml")
_IMMUTABLE_IMAGE_RE = re.compile(r"sha256:[0-9a-f]{64}")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_capture(
    argv: list[str],
    *,
    cwd: Path | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise EvidenceError(
            f"command failed to execute: {shlex.join(argv)}: {error}"
        ) from error


def _checked_output(argv: list[str], *, cwd: Path | None = None) -> str:
    completed = _run_capture(argv, cwd=cwd)
    if completed.returncode:
        raise EvidenceError(
            f"command exited {completed.returncode}: {shlex.join(argv)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed.stdout.strip()


def _git_state(root: Path) -> dict[str, Any]:
    commit = _checked_output(["git", "rev-parse", "HEAD"], cwd=root)
    status = _checked_output(["git", "status", "--short"], cwd=root)
    diff = _run_capture(["git", "diff", "--binary", "HEAD"], cwd=root)
    if diff.returncode:
        raise EvidenceError(f"cannot capture git diff for {root}")
    return {
        "root": str(root),
        "commit": commit,
        "status_short": status.splitlines() if status else [],
        "tracked_diff_sha256": sha256_bytes(diff.stdout.encode("utf-8")),
        "tracked_diff_bytes": len(diff.stdout.encode("utf-8")),
    }


def _ensure_new_output(root: Path) -> None:
    if root.exists() and any(root.iterdir()):
        raise EvidenceError(
            f"output directory must be absent or empty to prevent mixed evidence: {root}"
        )
    root.mkdir(parents=True, exist_ok=True)


def _source_provenance(tenon_root: Path, output_root: Path) -> dict[str, Any]:
    import allo

    imported_allo = Path(allo.__file__).resolve()
    if tenon_root.resolve() not in imported_allo.parents:
        raise EvidenceError(
            "the imported allo package is not from --tenon-root: "
            f"{imported_allo} is outside {tenon_root}"
        )
    source_roots = (tenon_root / "allo", tenon_root / "benchmarks/cent_aim")
    relative_sources = tuple(
        sorted(
            {
                Path("benchmarks/__init__.py"),
                *(
                    source.relative_to(tenon_root)
                    for source_root in source_roots
                    for source in source_root.rglob("*.py")
                    if "__pycache__" not in source.parts
                ),
            }
        )
    )
    if not relative_sources:
        raise EvidenceError("no Tenon Python sources found for provenance snapshot")
    files: dict[str, str] = {}
    snapshot = output_root / "provenance/compiler-source"
    for relative in relative_sources:
        source = tenon_root / relative
        if not source.is_file():
            raise EvidenceError(f"required compiler source is missing: {source}")
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        files[relative.as_posix()] = sha256_file(source)
    tree_material = "".join(
        f"{path}\0{digest}\n" for path, digest in sorted(files.items())
    )
    provenance = {
        "schema": "tenon-compiler-provenance-v1",
        "git": _git_state(tenon_root),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "imported_allo": str(imported_allo),
        },
        "source_files": files,
        "source_tree_sha256": sha256_bytes(tree_material.encode("utf-8")),
        "source_snapshot": "provenance/compiler-source",
        "campaign_argv": list(sys.argv),
        "campaign_command": shlex.join([sys.executable, *sys.argv]),
    }
    write_json(output_root / "provenance/compiler.json", provenance)
    return provenance


def _recheck_source_provenance(
    tenon_root: Path, compiler_provenance: dict[str, Any]
) -> None:
    for relative, expected_hash in compiler_provenance["source_files"].items():
        if sha256_file(tenon_root / relative) != expected_hash:
            raise EvidenceError(f"compiler source changed during campaign: {relative}")


def _copy_common_evidence(output_root: Path, case_dir: Path) -> None:
    """Make each eventual per-kernel archive independently auditable."""
    shutil.copytree(
        output_root / "provenance",
        case_dir / "common-evidence",
    )


def _resolve_image(image: str, output_root: Path) -> tuple[str, dict[str, Any]]:
    command = ["docker", "image", "inspect", image]
    completed = _run_capture(command)
    write_text(
        output_root / "provenance/docker-image-inspect.stdout.json",
        completed.stdout,
    )
    write_text(
        output_root / "provenance/docker-image-inspect.stderr.log",
        completed.stderr,
    )
    write_text(
        output_root / "provenance/docker-image-inspect.exit-code.txt",
        f"{completed.returncode}\n",
    )
    write_json(output_root / "provenance/docker-image-inspect.command.json", command)
    if completed.returncode:
        raise EvidenceError(f"docker cannot inspect image {image!r}")
    try:
        inspected = json.loads(completed.stdout)
        record = inspected[0]
        image_id = str(record["Id"])
    except (json.JSONDecodeError, IndexError, KeyError, TypeError) as error:
        raise EvidenceError(
            "docker image inspection did not return an image Id"
        ) from error
    if not _IMMUTABLE_IMAGE_RE.fullmatch(image_id):
        raise EvidenceError(f"docker returned a non-immutable image id: {image_id!r}")
    return image_id, record


def _simulator_provenance(
    simulator_root: Path,
    config_relative: Path,
    image_name: str,
    output_root: Path,
) -> dict[str, Any]:
    if config_relative.is_absolute() or ".." in config_relative.parts:
        raise EvidenceError("--config-relative must stay inside --simulator-root")
    state = _git_state(simulator_root)
    if state["commit"] != EXPECTED_SIMULATOR_COMMIT:
        raise EvidenceError(
            "installed simulator commit mismatch: expected "
            f"{EXPECTED_SIMULATOR_COMMIT}, got {state['commit']}"
        )
    binary = simulator_root / "build/ramulator2"
    library = simulator_root / "libramulator.so"
    config = simulator_root / config_relative
    for required in (binary, library, config):
        if not required.is_file():
            raise EvidenceError(f"required simulator artifact is missing: {required}")
    image_id, image_record = _resolve_image(image_name, output_root)
    copied_config = output_root / "provenance/simulator-config.yaml"
    copied_config.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config, copied_config)
    provenance = {
        "schema": "tenon-aim-simulator-provenance-v1",
        "git": state,
        "required_commit": EXPECTED_SIMULATOR_COMMIT,
        "binary_sha256": sha256_file(binary),
        "library_sha256": sha256_file(library),
        "config_relative": config_relative.as_posix(),
        "config_sha256": sha256_file(config),
        "docker_image_requested": image_name,
        "docker_image_immutable_id": image_id,
        "docker_image_repo_digests": image_record.get("RepoDigests") or [],
        "docker_image_created": image_record.get("Created"),
        "clock_GHz": 2.0,
        "cycle_metric": "global memory_system_cycles",
    }
    write_json(output_root / "provenance/simulator.json", provenance)
    return provenance


def _recheck_simulator_provenance(
    simulator_root: Path, simulator_provenance: dict[str, Any]
) -> None:
    current = {
        "binary_sha256": sha256_file(simulator_root / "build/ramulator2"),
        "library_sha256": sha256_file(simulator_root / "libramulator.so"),
        "config_sha256": sha256_file(
            simulator_root / simulator_provenance["config_relative"]
        ),
    }
    for field, observed in current.items():
        if observed != simulator_provenance[field]:
            raise EvidenceError(f"simulator {field} changed during campaign")


def _case_id(case: Any) -> str:
    if isinstance(case, dict):
        value = case.get("case_id")
    else:
        value = getattr(case, "case_id", None)
    if not isinstance(value, str) or not value:
        raise EvidenceError("typed workload case lacks a non-empty case_id")
    return value


def _case_sequence_length(case: Any, spec: Any) -> int:
    for owner in (case, spec):
        for name in ("L", "sequence_length"):
            value = (
                owner.get(name)
                if isinstance(owner, dict)
                else getattr(owner, name, None)
            )
            if value is not None:
                return int(value)
    raise EvidenceError(f"cannot derive sequence length for {_case_id(case)}")


def _load_workload_descriptors() -> list[tuple[Any, Any]]:
    try:
        from benchmarks.cent_aim.workloads import iter_case_programs
    except ImportError as error:
        raise EvidenceError("cannot import benchmarks.cent_aim.workloads") from error
    descriptors = []
    for item in iter_case_programs():
        try:
            case, spec, _program = item
        except (TypeError, ValueError) as error:
            raise EvidenceError(
                "iter_case_programs() must yield (case, spec, program) triples"
            ) from error
        descriptors.append((case, spec))
    ids = tuple(_case_id(case) for case, _spec in descriptors)
    if ids != EXPECTED_CASE_IDS:
        raise EvidenceError(
            "typed workload suite does not match the frozen 14-case order: "
            f"expected {EXPECTED_CASE_IDS}, got {ids}"
        )
    return descriptors


def _compile_case_twice(
    *,
    case: Any,
    spec: Any,
    case_dir: Path,
    vendor_evidence: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    import allo
    from allo.pim.targets import build_aim_target
    from benchmarks.cent_aim.workloads import build_case, semantic_manifest

    case_id = _case_id(case)
    write_json(case_dir / "case-definition.json", jsonable(case))
    write_json(case_dir / "decode-spec.json", jsonable(spec))
    write_json(
        case_dir / "compile-command.json",
        {
            "program_builder": "benchmarks.cent_aim.workloads.build_case",
            "program_builder_arguments": {"case_id": case_id, "spec": jsonable(spec)},
            "compiler": "allo.compile",
            "target_builder": "allo.pim.targets.build_aim_target",
            "independent_build_compile_repeats": 2,
        },
    )
    traces: list[str] = []
    source_manifests: list[bytes] = []
    compiled_manifests: list[bytes] = []
    semantic_manifests: list[bytes] = []
    final_compiled = None
    for repeat in (1, 2):
        program = build_case(case_id, spec=spec)
        compiled = allo.compile(program, build_aim_target())
        trace = str(compiled.trace)
        trace_name = "input.trace" if repeat == 1 else "input.repeat.trace"
        write_text(case_dir / trace_name, trace)
        source = canonical_json_bytes(program.manifest())
        materialization = canonical_json_bytes(compiled.manifest)
        semantics = canonical_json_bytes(semantic_manifest(case_id, spec, compiled))
        write_text(case_dir / f"source.run-{repeat}.json", source.decode("utf-8"))
        write_text(
            case_dir / f"compiled.run-{repeat}.json",
            materialization.decode("utf-8"),
        )
        write_text(
            case_dir / f"semantic.run-{repeat}.json",
            semantics.decode("utf-8"),
        )
        traces.append(trace)
        source_manifests.append(source)
        compiled_manifests.append(materialization)
        semantic_manifests.append(semantics)
        final_compiled = compiled
    if traces[0].encode("utf-8") != traces[1].encode("utf-8"):
        raise EvidenceError(
            f"{case_id}: independent compilations emitted different traces"
        )
    if source_manifests[0] != source_manifests[1]:
        raise EvidenceError(f"{case_id}: independent source manifests differ")
    if compiled_manifests[0] != compiled_manifests[1]:
        raise EvidenceError(f"{case_id}: independent compiled manifests differ")
    if semantic_manifests[0] != semantic_manifests[1]:
        raise EvidenceError(f"{case_id}: independent semantic manifests differ")
    trace_metadata = validate_trace_text(traces[0], label=case_id)
    try:
        physical_contract = VENDOR_TRACE_CONTRACTS[case_id]
    except KeyError as error:
        raise EvidenceError(f"{case_id}: no frozen vendor work contract") from error
    expected_opcode_counts = dict(sorted(physical_contract.opcode_counts.items()))
    try:
        vendor_shape_signature = vendor_evidence["physical_work_contract"][
            "command_shape_signature"
        ]
    except (KeyError, TypeError) as error:
        raise EvidenceError(
            f"{case_id}: vendor evidence lacks a command-shape signature"
        ) from error
    compiled_manifest = json.loads(compiled_manifests[0])
    declared_trace = compiled_manifest.get("trace", {})
    if declared_trace.get("sha256") != trace_metadata["sha256"]:
        raise EvidenceError(f"{case_id}: compiled manifest trace hash is stale")
    if declared_trace.get("eoc_count") != 1:
        raise EvidenceError(f"{case_id}: compiled manifest does not declare one EOC")
    vendor_wr_abk_coverage = vendor_evidence["physical_work_contract"][
        "wr_abk_channel_coverage"
    ]
    semantic_manifest_value = json.loads(semantic_manifests[0])
    physical_comparison = physical_work_comparison(
        cent_source_commit=CENT_SOURCE_COMMIT,
        vendor_trace_sha256=physical_contract.trace_sha256,
        vendor_opcode_counts=expected_opcode_counts,
        vendor_command_shape_signature=vendor_shape_signature,
        vendor_wr_abk_channel_coverage=vendor_wr_abk_coverage,
        tenon_trace_text=traces[0],
        target_channels=int(compiled_manifest["geometry"]["channels"]),
        replica_count=int(getattr(spec, "replicas")),
        channels_per_replica=int(getattr(spec, "channels_per_replica")),
        logical_work_coverage=semantic_manifest_value["logical_work_coverage"],
    )
    write_json(
        case_dir / "trace-metadata.json",
        {
            **trace_metadata,
            "compile_repeats": 2,
            "byte_identical": True,
            "repeat_sha256": sha256_bytes(traces[1].encode("utf-8")),
        },
    )
    write_json(
        case_dir / "semantic-work-invariant.json",
        {
            "schema": "tenon-cent-aim-semantic-evidence-v1",
            "case_id": case_id,
            "timing_model": "SK hynix AiM Ramulator2",
            "timing_only": True,
            "numeric_inputs_consumed": False,
            "numeric_outputs_produced": False,
            "numeric_correctness_claim": "not_available_from_timing_simulator",
            "physical_work_contract": physical_comparison,
            "semantic_manifest": semantic_manifest_value,
            "independent_manifest_repeats": 2,
            "manifests_byte_identical": True,
        },
    )
    write_text(
        case_dir / "data-output-status.txt",
        (
            "The SK hynix AiM Ramulator2 model is timing-only. It consumes the "
            "retained command/address trace, but no numerical tensor payload, and "
            "produces no numerical output file. This campaign therefore makes no "
            "simulator-backed numerical-correctness claim. The separately retained "
            "typed source, compiled lowering, and semantic/work-invariant manifests "
            "define the measured work; the measured datum is global "
            "memory_system_cycles in each raw stdout log.\n"
        ),
    )
    assert final_compiled is not None
    return final_compiled, trace_metadata


def _simulator_command(
    *,
    simulator_root: Path,
    evidence_dir: Path,
    image_id: str,
    config_relative: Path,
) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        "LD_LIBRARY_PATH=/work",
        "-v",
        f"{simulator_root}:/work:ro",
        "-v",
        f"{evidence_dir}:/evidence:ro",
        image_id,
        "/work/build/ramulator2",
        "-f",
        f"/work/{config_relative.as_posix()}",
        "-t",
        "/evidence/input.trace",
    ]


def _run_simulator_twice(
    *,
    evidence_dir: Path,
    trace_metadata: dict[str, Any],
    simulator_root: Path,
    config_relative: Path,
    simulator_provenance: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    sim_dir = evidence_dir / "simulator"
    sim_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(simulator_root / config_relative, sim_dir / "config.yaml")
    command = _simulator_command(
        simulator_root=simulator_root,
        evidence_dir=evidence_dir,
        image_id=simulator_provenance["docker_image_immutable_id"],
        config_relative=config_relative,
    )
    write_json(sim_dir / "command.argv.json", command)
    write_text(sim_dir / "command.txt", shlex.join(command) + "\n")
    run_metrics = []
    for repeat in (1, 2):
        completed = _run_capture(command, timeout=timeout)
        stdout_path = sim_dir / f"run-{repeat}.stdout.raw.log"
        stderr_path = sim_dir / f"run-{repeat}.stderr.raw.log"
        write_text(stdout_path, completed.stdout)
        write_text(stderr_path, completed.stderr)
        write_text(sim_dir / f"run-{repeat}.exit-code.txt", f"{completed.returncode}\n")
        if completed.returncode:
            raise EvidenceError(
                f"simulation repeat {repeat} exited {completed.returncode} for {evidence_dir.name}"
            )
        parsed = parse_simulator_stdout(completed.stdout)
        reconcile_request_counts(
            trace_metadata["opcode_counts"], parsed["total_request_counts"]
        )
        parsed.update(
            {
                "repeat": repeat,
                "exit_code": completed.returncode,
                "trace_sha256": trace_metadata["sha256"],
                "stdout_sha256": sha256_file(stdout_path),
                "stderr_sha256": sha256_file(stderr_path),
            }
        )
        write_json(sim_dir / f"run-{repeat}.metrics.json", parsed)
        run_metrics.append(parsed)
    comparable = [
        {
            "memory_system_cycles": item["memory_system_cycles"],
            "eoc_requests": item["eoc_requests"],
            "total_request_counts": item["total_request_counts"],
        }
        for item in run_metrics
    ]
    if comparable[0] != comparable[1]:
        raise EvidenceError(f"simulator repeats differ for {evidence_dir.name}")
    summary = {
        "schema": "tenon-cent-aim-simulator-summary-v1",
        "simulator_commit": simulator_provenance["required_commit"],
        "simulator_binary_sha256": simulator_provenance["binary_sha256"],
        "simulator_library_sha256": simulator_provenance["library_sha256"],
        "config_sha256": simulator_provenance["config_sha256"],
        "docker_image_immutable_id": simulator_provenance["docker_image_immutable_id"],
        "trace_sha256": trace_metadata["sha256"],
        "trace_opcode_counts": trace_metadata["opcode_counts"],
        "repeat_count": 2,
        "cycles_deterministic": True,
        "memory_system_cycles": run_metrics[0]["memory_system_cycles"],
        "latency_us_at_2GHz": run_metrics[0]["latency_us_at_2GHz"],
        "runs": run_metrics,
    }
    write_json(sim_dir / "summary.json", summary)
    return summary


def _docker_image_from_command(command: list[Any]) -> str:
    try:
        binary_index = command.index("/work/build/ramulator2")
    except ValueError as error:
        raise EvidenceError(
            "vendor docker command lacks ramulator2 executable"
        ) from error
    if binary_index == 0:
        raise EvidenceError("vendor docker command lacks image argument")
    return str(command[binary_index - 1])


def _load_vendor_evidence(
    *,
    vendor_root: Path,
    case_id: str,
    simulator_provenance: dict[str, Any],
) -> dict[str, Any]:
    case_dir = vendor_root / case_id
    verify_checksums(case_dir, require_complete=True)
    try:
        physical_contract = VENDOR_TRACE_CONTRACTS[case_id]
    except KeyError as error:
        raise EvidenceError(f"{case_id}: no frozen vendor work contract") from error
    expected_opcode_counts = dict(sorted(physical_contract.opcode_counts.items()))
    case, vendor_trace = validate_frozen_vendor_trace(
        case_dir,
        case_id=case_id,
        expected_trace_sha256=physical_contract.trace_sha256,
        expected_opcode_counts=expected_opcode_counts,
        expected_source_commit=CENT_SOURCE_COMMIT,
    )
    sim_dir = case_dir / "installed-0f28a07b"
    summary_path = sim_dir / "summary.json"
    summary = read_json(summary_path)
    if summary.get("simulator_commit") != EXPECTED_SIMULATOR_COMMIT:
        raise EvidenceError(f"{case_id}: vendor result is not from simulator 0f28a07b")
    equality_fields = {
        "simulator_binary_sha256": "binary_sha256",
        "simulator_library_sha256": "library_sha256",
        "config_sha256": "config_sha256",
    }
    for vendor_field, current_field in equality_fields.items():
        if summary.get(vendor_field) != simulator_provenance.get(current_field):
            raise EvidenceError(
                f"{case_id}: vendor and Tenon {vendor_field} differ; comparison is not fair"
            )
    command = read_json(sim_dir / "command.argv.json")
    if (
        _docker_image_from_command(command)
        != simulator_provenance["docker_image_immutable_id"]
    ):
        raise EvidenceError(f"{case_id}: vendor and Tenon immutable images differ")
    cycles = []
    raw_runs = []
    for repeat in (1, 2):
        exit_code = int(
            (sim_dir / f"run-{repeat}.exit-code.txt")
            .read_text(encoding="utf-8")
            .strip()
        )
        if exit_code:
            raise EvidenceError(f"{case_id}: vendor repeat {repeat} exited {exit_code}")
        stdout_path = sim_dir / f"run-{repeat}.stdout.raw.log"
        stderr_path = sim_dir / f"run-{repeat}.stderr.raw.log"
        parsed = parse_simulator_stdout(stdout_path.read_text(encoding="utf-8"))
        reconcile_request_counts(
            vendor_trace["opcode_counts"], parsed["total_request_counts"]
        )
        metrics = read_json(sim_dir / f"run-{repeat}.metrics.json")
        if metrics.get("memory_system_cycles") != parsed["memory_system_cycles"]:
            raise EvidenceError(
                f"{case_id}: vendor parsed metrics disagree with raw stdout"
            )
        cycles.append(parsed["memory_system_cycles"])
        raw_runs.append(
            {
                "repeat": repeat,
                "exit_code": exit_code,
                "stdout_sha256": sha256_file(stdout_path),
                "stderr_sha256": sha256_file(stderr_path),
                "metrics_sha256": sha256_file(sim_dir / f"run-{repeat}.metrics.json"),
                "memory_system_cycles": parsed["memory_system_cycles"],
                "eoc_requests": parsed["eoc_requests"],
            }
        )
    if cycles != [summary.get("memory_system_cycles")] * 2:
        raise EvidenceError(f"{case_id}: vendor repeats do not support summary cycles")
    trace_path = case_dir / "input.trace"
    if vendor_trace["sha256"] != summary.get("trace_sha256"):
        raise EvidenceError(f"{case_id}: vendor summary trace hash is stale")
    vendor_trace_text = trace_path.read_text(encoding="utf-8")
    vendor_shape_signature = command_shape_signature(vendor_trace_text)
    vendor_wr_abk_coverage = opcode_active_channel_coverage(
        vendor_trace_text,
        "AiM_WR_ABK",
        target_channels=32,
    )
    return {
        "schema": "tenon-cent-vendor-evidence-join-v1",
        "case_id": case_id,
        "vendor_case": case,
        "vendor_root_recorded": str(vendor_root),
        "vendor_case_directory": case_id,
        "case_checksum_manifest_sha256": sha256_file(case_dir / "SHA256SUMS"),
        "case_manifest_sha256": sha256_file(case_dir / "case.json"),
        "trace_sha256": sha256_file(trace_path),
        "physical_work_contract": {
            "cent_source_commit": CENT_SOURCE_COMMIT,
            "vendor_trace_sha256": physical_contract.trace_sha256,
            "opcode_counts": expected_opcode_counts,
            "command_shape_signature": vendor_shape_signature,
            "wr_abk_channel_coverage": vendor_wr_abk_coverage,
        },
        "simulator_summary_sha256": sha256_file(summary_path),
        "simulator_commit": summary["simulator_commit"],
        "simulator_binary_sha256": summary["simulator_binary_sha256"],
        "simulator_library_sha256": summary["simulator_library_sha256"],
        "config_sha256": summary["config_sha256"],
        "docker_image_immutable_id": _docker_image_from_command(command),
        "repeat_count": 2,
        "cycles_deterministic": True,
        "memory_system_cycles": cycles[0],
        "raw_runs": raw_runs,
    }


def _write_comparison(output_root: Path, rows: list[dict[str, Any]]) -> None:
    document = {
        "schema": "tenon-cent-aim-comparison-v1",
        "metric": "global memory_system_cycles",
        "lower_is_better": True,
        "simulator_commit": EXPECTED_SIMULATOR_COMMIT,
        "totals": comparison_totals(rows),
        "rows": rows,
    }
    write_json(output_root / "comparison.json", document)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=COMPARISON_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    write_text(output_root / "comparison.csv", stream.getvalue())
    write_text(output_root / "COMPARISON.md", render_comparison_markdown(rows))


def _write_case_comparison(case_dir: Path, row: dict[str, Any]) -> None:
    local_row = {
        **row,
        "tenon_evidence": "simulator/summary.json",
        "vendor_evidence": "vendor-evidence.json",
    }
    write_json(
        case_dir / "comparison.json",
        {
            "schema": "tenon-cent-aim-case-comparison-v1",
            "row": local_row,
        },
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=COMPARISON_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerow(local_row)
    write_text(case_dir / "comparison.csv", stream.getvalue())
    write_text(case_dir / "RESULT.md", render_comparison_markdown([local_row]))
    write_text(
        case_dir / "VERIFY.md",
        (
            "Run from this directory with the sibling frozen vendor root:\n\n"
            "```bash\n"
            "PYTHONDONTWRITEBYTECODE=1 "
            "PYTHONPATH=common-evidence/compiler-source python -B -m "
            "benchmarks.cent_aim.verify_case . --vendor-root /path/to/skhynix/vendor\n"
            "```\n"
        ),
    )


def _reuse_candidates(target: Any) -> tuple[int, ...]:
    capacity = int(target.mac_reg.slots)
    if capacity <= 0:
        raise EvidenceError("target.mac_reg.slots must be positive")
    candidates = []
    candidate = 1
    while candidate <= capacity:
        candidates.append(candidate)
        candidate *= 2
    if not candidates or candidates[-1] != capacity:
        candidates.append(capacity)
    return tuple(candidates)


def _spec_dimension(spec: Any) -> int:
    value = spec.get("D") if isinstance(spec, dict) else getattr(spec, "D", None)
    if value is None:
        raise EvidenceError("reuse calibration requires DecodeSpec.D")
    return int(value)


def _compile_calibration_twice(
    *,
    candidate: int,
    candidates: tuple[int, ...],
    dimension: int,
    replicas: int,
    channels_per_replica: int,
    output: Path,
) -> dict[str, Any]:
    import allo
    from allo.pim.aim_program import AimContraction, AimProgram
    from allo.pim.targets import build_aim_target

    traces = []
    for repeat in (1, 2):
        program = AimProgram(
            [
                AimContraction(
                    outputs=dimension,
                    reduction=dimension,
                    replicas=replicas,
                    channels=tuple(range(replicas * channels_per_replica)),
                    channels_per_replica=channels_per_replica,
                    reuse_group_size=candidate,
                    reuse_group_candidates=candidates,
                    name="strict_square_projection",
                )
            ],
            name=f"reuse_group_{candidate}",
        )
        compiled = allo.compile(program, build_aim_target())
        trace = str(compiled.trace)
        traces.append(trace)
        trace_name = "input.trace" if repeat == 1 else "input.repeat.trace"
        write_text(output / trace_name, trace)
        write_json(output / f"source.run-{repeat}.json", program.manifest())
        write_json(output / f"compiled.run-{repeat}.json", compiled.manifest)
    if traces[0].encode("utf-8") != traces[1].encode("utf-8"):
        raise EvidenceError(
            f"reuse calibration candidate {candidate} is nondeterministic"
        )
    metadata = validate_trace_text(traces[0], label=f"reuse-group-{candidate}")
    write_json(
        output / "trace-metadata.json",
        {**metadata, "compile_repeats": 2, "byte_identical": True},
    )
    write_json(
        output / "calibration-work.json",
        {
            "schema": "tenon-aim-reuse-calibration-work-v1",
            "operation": "dense_contraction",
            "outputs": dimension,
            "reduction": dimension,
            "batches": 1,
            "replicas": replicas,
            "channels_per_replica": channels_per_replica,
            "reuse_group_candidate": candidate,
            "legal_candidates": list(candidates),
            "candidate_limit_source": "target.mac_reg.slots",
        },
    )
    return metadata


def _run_reuse_calibration(
    *,
    output_root: Path,
    spec: Any,
    simulator_root: Path,
    config_relative: Path,
    simulator_provenance: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    from allo.pim.targets import build_aim_target

    calibration_root = output_root / "calibration/reuse-groups"
    target = build_aim_target()
    candidates = _reuse_candidates(target)
    dimension = _spec_dimension(spec)
    replicas = int(spec.replicas)
    channels_per_replica = int(spec.channels_per_replica)
    rows = []
    for candidate in candidates:
        candidate_dir = calibration_root / f"candidate-{candidate}"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        metadata = _compile_calibration_twice(
            candidate=candidate,
            candidates=candidates,
            dimension=dimension,
            replicas=replicas,
            channels_per_replica=channels_per_replica,
            output=candidate_dir,
        )
        summary = _run_simulator_twice(
            evidence_dir=candidate_dir,
            trace_metadata=metadata,
            simulator_root=simulator_root,
            config_relative=config_relative,
            simulator_provenance=simulator_provenance,
            timeout=timeout,
        )
        row = {
            "reuse_group_size": candidate,
            "memory_system_cycles": summary["memory_system_cycles"],
            "trace_commands": metadata["nonempty_lines"],
            "trace_sha256": metadata["sha256"],
            "evidence": f"calibration/reuse-groups/candidate-{candidate}/simulator/summary.json",
        }
        rows.append(row)
        write_checksums(candidate_dir)
    best = min(
        rows, key=lambda row: (row["memory_system_cycles"], row["reuse_group_size"])
    )
    report = {
        "schema": "tenon-aim-reuse-calibration-v1",
        "work": {
            "outputs": dimension,
            "reduction": dimension,
            "replicas": replicas,
            "channels_per_replica": channels_per_replica,
            "dimension_source": "DecodeSpec.D",
        },
        "legal_candidate_source": "powers of two through target.mac_reg.slots",
        "legal_candidates": list(candidates),
        "simulator_commit": EXPECTED_SIMULATOR_COMMIT,
        "compile_repeats_per_candidate": 2,
        "simulator_repeats_per_candidate": 2,
        "rows": rows,
        "measured_best_legal_candidate": best["reuse_group_size"],
        "measured_best_cycles": best["memory_system_cycles"],
        "activation_policy": "advisory evidence; campaign does not mutate compiler policy",
    }
    write_json(calibration_root / "results.json", report)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    write_text(calibration_root / "results.csv", stream.getvalue())
    write_text(
        calibration_root / "RESULTS.md",
        render_reuse_calibration_markdown(rows, dimension),
    )
    write_checksums(calibration_root)
    return report


def run_campaign(args: argparse.Namespace) -> Path:
    output_root = args.output_root.resolve()
    tenon_root = args.tenon_root.resolve()
    simulator_root = args.simulator_root.resolve()
    vendor_root = args.vendor_root.resolve()
    config_relative = Path(args.config_relative)
    _ensure_new_output(output_root)
    started = _now_utc()
    compiler_provenance = _source_provenance(tenon_root, output_root)
    simulator_provenance = _simulator_provenance(
        simulator_root,
        config_relative,
        args.docker_image,
        output_root,
    )
    descriptors = _load_workload_descriptors()
    rows = []
    for index, (case, spec) in enumerate(descriptors, start=1):
        case_id = _case_id(case)
        print(f"[{index}/{len(descriptors)}] compile and measure {case_id}", flush=True)
        case_dir = output_root / "cases" / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        _copy_common_evidence(output_root, case_dir)
        write_json(
            case_dir / "campaign-provenance.json",
            {
                "compiler": compiler_provenance,
                "simulator": simulator_provenance,
            },
        )
        vendor = _load_vendor_evidence(
            vendor_root=vendor_root,
            case_id=case_id,
            simulator_provenance=simulator_provenance,
        )
        write_json(case_dir / "vendor-evidence.json", vendor)
        _compiled, trace_metadata = _compile_case_twice(
            case=case,
            spec=spec,
            case_dir=case_dir,
            vendor_evidence=vendor,
        )
        sim_summary = _run_simulator_twice(
            evidence_dir=case_dir,
            trace_metadata=trace_metadata,
            simulator_root=simulator_root,
            config_relative=config_relative,
            simulator_provenance=simulator_provenance,
            timeout=args.timeout,
        )
        vendor_case = vendor["vendor_case"]
        row = comparison_row(
            case_id=case_id,
            kernel=str(vendor_case["kernel"]),
            sequence_length=_case_sequence_length(case, spec),
            vendor_cycles=vendor["memory_system_cycles"],
            tenon_cycles=sim_summary["memory_system_cycles"],
            trace_sha256=trace_metadata["sha256"],
        )
        rows.append(row)
        _write_case_comparison(case_dir, row)
        write_checksums(case_dir)
    _write_comparison(output_root, rows)
    calibration = None
    if not args.skip_reuse_calibration:
        print("[calibration] measure target-legal contraction reuse groups", flush=True)
        calibration = _run_reuse_calibration(
            output_root=output_root,
            spec=descriptors[0][1],
            simulator_root=simulator_root,
            config_relative=config_relative,
            simulator_provenance=simulator_provenance,
            timeout=args.timeout,
        )
    _recheck_source_provenance(tenon_root, compiler_provenance)
    _recheck_simulator_provenance(simulator_root, simulator_provenance)
    campaign = {
        "schema": "tenon-cent-aim-campaign-v1",
        "started_utc": started,
        "completed_utc": _now_utc(),
        "case_ids": list(EXPECTED_CASE_IDS),
        "case_count": len(rows),
        "compile_repeats_per_case": 2,
        "simulator_repeats_per_case": 2,
        "simulator_commit": EXPECTED_SIMULATOR_COMMIT,
        "cycle_metric": "global memory_system_cycles",
        "lower_is_better": True,
        "modeled_clock_GHz": 2.0,
        "timing_only": True,
        "numeric_correctness_claim": "not_available_from_timing_simulator",
        "comparison_totals": comparison_totals(rows),
        "compiler_source_tree_sha256": compiler_provenance["source_tree_sha256"],
        "simulator_binary_sha256": simulator_provenance["binary_sha256"],
        "simulator_library_sha256": simulator_provenance["library_sha256"],
        "simulator_config_sha256": simulator_provenance["config_sha256"],
        "docker_image_immutable_id": simulator_provenance["docker_image_immutable_id"],
        "reuse_calibration": (
            None
            if calibration is None
            else {
                "results": "calibration/reuse-groups/results.json",
                "measured_best_legal_candidate": calibration[
                    "measured_best_legal_candidate"
                ],
            }
        ),
        "arguments": {
            "tenon_root": str(tenon_root),
            "simulator_root": str(simulator_root),
            "vendor_root": str(vendor_root),
            "output_root": str(output_root),
            "config_relative": config_relative.as_posix(),
            "docker_image_requested": args.docker_image,
            "timeout_seconds": args.timeout,
            "skip_reuse_calibration": args.skip_reuse_calibration,
        },
    }
    write_json(output_root / "campaign.json", campaign)
    write_checksums(output_root)
    print(
        f"wrote {len(rows)} evidence-backed comparisons to {output_root}",
        flush=True,
    )
    return output_root


def _parser() -> argparse.ArgumentParser:
    repository_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tenon-root",
        type=Path,
        default=repository_default,
        help="Allo/Tenon repository root (defaults to the repository containing this script)",
    )
    parser.add_argument(
        "--simulator-root",
        type=Path,
        required=True,
        help="installed AiM simulator checkout at required commit 0f28a07b",
    )
    parser.add_argument(
        "--vendor-root",
        type=Path,
        required=True,
        help="frozen vendor directory containing one checksummed directory per case",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--docker-image", default=DEFAULT_IMAGE)
    parser.add_argument(
        "--config-relative",
        default=DEFAULT_CONFIG_RELATIVE.as_posix(),
        help="simulator config path relative to --simulator-root",
    )
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--skip-reuse-calibration",
        action="store_true",
        help="omit the evidence-backed generic reuse-group calibration phase",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")
    try:
        run_campaign(args)
    except EvidenceError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
