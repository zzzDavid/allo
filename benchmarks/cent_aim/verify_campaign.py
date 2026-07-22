#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify every performance cell in a Tenon-versus-CENT AiM campaign."""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

try:
    from .evidence import (
        COMPARISON_FIELDS,
        EXPECTED_CASE_IDS,
        EXPECTED_SIMULATOR_COMMIT,
        EvidenceError,
        command_shape_signature,
        compiler_logical_work_coverage,
        opcode_active_channel_coverage,
        physical_work_comparison,
        comparison_row,
        comparison_totals,
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
    )
except ImportError:  # Support direct execution from this directory.
    from evidence import (  # type: ignore
        COMPARISON_FIELDS,
        EXPECTED_CASE_IDS,
        EXPECTED_SIMULATOR_COMMIT,
        EvidenceError,
        command_shape_signature,
        compiler_logical_work_coverage,
        opcode_active_channel_coverage,
        physical_work_comparison,
        comparison_row,
        comparison_totals,
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
    )

try:
    from .vendor_contract import CENT_SOURCE_COMMIT, VENDOR_TRACE_CONTRACTS
except ImportError:  # Support direct execution from this directory.
    from vendor_contract import (  # type: ignore
        CENT_SOURCE_COMMIT,
        VENDOR_TRACE_CONTRACTS,
    )


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise EvidenceError(f"cannot read text evidence {path}: {error}") from error


def _read_exit_code(path: Path) -> int:
    try:
        return int(_read_text(path).strip())
    except ValueError as error:
        raise EvidenceError(f"invalid exit code evidence {path}") from error


def _source_tree_digest(files: dict[str, str]) -> str:
    material = "".join(f"{path}\0{digest}\n" for path, digest in sorted(files.items()))
    return sha256_bytes(material.encode("utf-8"))


def _verify_provenance(
    root: Path,
    *,
    tenon_root: Path | None,
    simulator_root: Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    compiler = read_json(root / "provenance/compiler.json")
    simulator = read_json(root / "provenance/simulator.json")
    if compiler.get("schema") != "tenon-compiler-provenance-v1":
        raise EvidenceError("unknown compiler provenance schema")
    source_files = compiler.get("source_files")
    if not isinstance(source_files, dict) or not source_files:
        raise EvidenceError("compiler provenance has no source files")
    for relative, expected_hash in source_files.items():
        snapshot = root / "provenance/compiler-source" / relative
        if sha256_file(snapshot) != expected_hash:
            raise EvidenceError(f"compiler source snapshot is stale: {relative}")
        if (
            tenon_root is not None
            and sha256_file(tenon_root / relative) != expected_hash
        ):
            raise EvidenceError(
                f"current Tenon source differs from campaign: {relative}"
            )
    if _source_tree_digest(source_files) != compiler.get("source_tree_sha256"):
        raise EvidenceError("compiler source-tree digest is stale")
    if simulator.get("schema") != "tenon-aim-simulator-provenance-v1":
        raise EvidenceError("unknown simulator provenance schema")
    if simulator.get("required_commit") != EXPECTED_SIMULATOR_COMMIT:
        raise EvidenceError("campaign simulator commit is not the required revision")
    image_id = str(simulator.get("docker_image_immutable_id", ""))
    if not image_id.startswith("sha256:") or len(image_id) != 71:
        raise EvidenceError("campaign lacks a content-addressed Docker image id")
    if sha256_file(root / "provenance/simulator-config.yaml") != simulator.get(
        "config_sha256"
    ):
        raise EvidenceError("retained simulator config hash is stale")
    inspect_exit = _read_exit_code(
        root / "provenance/docker-image-inspect.exit-code.txt"
    )
    if inspect_exit:
        raise EvidenceError("retained docker image inspection failed")
    inspected = read_json(root / "provenance/docker-image-inspect.stdout.json")
    try:
        inspected_id = inspected[0]["Id"]
    except (IndexError, KeyError, TypeError) as error:
        raise EvidenceError("retained docker inspection lacks image Id") from error
    if inspected_id != image_id:
        raise EvidenceError(
            "docker inspection and simulator provenance image ids differ"
        )
    inspect_command = read_json(root / "provenance/docker-image-inspect.command.json")
    if inspect_command != [
        "docker",
        "image",
        "inspect",
        simulator.get("docker_image_requested"),
    ]:
        raise EvidenceError("retained docker inspection command is inconsistent")
    if simulator_root is not None:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=simulator_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if (
            completed.returncode
            or completed.stdout.strip() != EXPECTED_SIMULATOR_COMMIT
        ):
            raise EvidenceError("current simulator checkout is not commit 0f28a07b")
        checks = {
            simulator_root / "build/ramulator2": simulator["binary_sha256"],
            simulator_root / "libramulator.so": simulator["library_sha256"],
            simulator_root / simulator["config_relative"]: simulator["config_sha256"],
        }
        for path, expected in checks.items():
            if sha256_file(path) != expected:
                raise EvidenceError(f"current simulator artifact differs: {path}")
    return compiler, simulator


def _verify_command(command: list[Any], simulator: dict[str, Any]) -> None:
    if len(command) != 20 or command[:7] != [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--user",
    ]:
        raise EvidenceError("simulator command isolation prefix differs")
    user = str(command[7]).split(":")
    if len(user) != 2 or not all(part.isdigit() for part in user):
        raise EvidenceError("simulator command has an invalid numeric user")
    if command[8:11] != ["-e", "LD_LIBRARY_PATH=/work", "-v"]:
        raise EvidenceError("simulator command library environment differs")
    if command[12] != "-v":
        raise EvidenceError("simulator command evidence mount flag differs")
    if not str(command[11]).startswith("/") or not str(command[11]).endswith(
        ":/work:ro"
    ):
        raise EvidenceError("simulator root is not mounted read-only at /work")
    if not str(command[13]).startswith("/") or not str(command[13]).endswith(
        ":/evidence:ro"
    ):
        raise EvidenceError("trace evidence is not mounted read-only at /evidence")
    if command[14] != simulator["docker_image_immutable_id"]:
        raise EvidenceError("simulator command does not use recorded immutable image")
    if command[15:] != [
        "/work/build/ramulator2",
        "-f",
        f"/work/{simulator['config_relative']}",
        "-t",
        "/evidence/input.trace",
    ]:
        raise EvidenceError(
            "simulator executable arguments differ from campaign contract"
        )


def _verify_simulator_runs(
    evidence_dir: Path,
    trace: dict[str, Any],
    simulator: dict[str, Any],
) -> dict[str, Any]:
    sim_dir = evidence_dir / "simulator"
    if sha256_file(sim_dir / "config.yaml") != simulator["config_sha256"]:
        raise EvidenceError(f"{evidence_dir.name}: simulator config differs")
    command = read_json(sim_dir / "command.argv.json")
    _verify_command(command, simulator)
    if _read_text(sim_dir / "command.txt") != shlex.join(command) + "\n":
        raise EvidenceError(f"{evidence_dir.name}: command text and argv differ")
    parsed_runs = []
    for repeat in (1, 2):
        if _read_exit_code(sim_dir / f"run-{repeat}.exit-code.txt") != 0:
            raise EvidenceError(
                f"{evidence_dir.name}: simulator repeat {repeat} failed"
            )
        stdout_path = sim_dir / f"run-{repeat}.stdout.raw.log"
        stderr_path = sim_dir / f"run-{repeat}.stderr.raw.log"
        parsed = parse_simulator_stdout(_read_text(stdout_path))
        reconcile_request_counts(trace["opcode_counts"], parsed["total_request_counts"])
        metrics = read_json(sim_dir / f"run-{repeat}.metrics.json")
        expected_fields = {
            **parsed,
            "repeat": repeat,
            "exit_code": 0,
            "trace_sha256": trace["sha256"],
            "stdout_sha256": sha256_file(stdout_path),
            "stderr_sha256": sha256_file(stderr_path),
        }
        if metrics != expected_fields:
            raise EvidenceError(
                f"{evidence_dir.name}: repeat {repeat} metrics do not derive from raw logs"
            )
        parsed_runs.append(metrics)
    comparable = [
        (
            run["memory_system_cycles"],
            run["eoc_requests"],
            run["total_request_counts"],
        )
        for run in parsed_runs
    ]
    if comparable[0] != comparable[1]:
        raise EvidenceError(
            f"{evidence_dir.name}: simulator repeats are not deterministic"
        )
    summary = read_json(sim_dir / "summary.json")
    expected_summary = {
        "schema": "tenon-cent-aim-simulator-summary-v1",
        "simulator_commit": simulator["required_commit"],
        "simulator_binary_sha256": simulator["binary_sha256"],
        "simulator_library_sha256": simulator["library_sha256"],
        "config_sha256": simulator["config_sha256"],
        "docker_image_immutable_id": simulator["docker_image_immutable_id"],
        "trace_sha256": trace["sha256"],
        "trace_opcode_counts": trace["opcode_counts"],
        "repeat_count": 2,
        "cycles_deterministic": True,
        "memory_system_cycles": parsed_runs[0]["memory_system_cycles"],
        "latency_us_at_2GHz": parsed_runs[0]["latency_us_at_2GHz"],
        "runs": parsed_runs,
    }
    if summary != expected_summary:
        raise EvidenceError(f"{evidence_dir.name}: simulator summary is not derivable")
    return summary


def _verify_compile(case_dir: Path) -> dict[str, Any]:
    first_trace = _read_text(case_dir / "input.trace")
    second_trace = _read_text(case_dir / "input.repeat.trace")
    if first_trace.encode("utf-8") != second_trace.encode("utf-8"):
        raise EvidenceError(f"{case_dir.name}: compiler trace repeats differ")
    trace = validate_trace_text(first_trace, label=case_dir.name)
    try:
        physical_contract = VENDOR_TRACE_CONTRACTS[case_dir.name]
    except KeyError as error:
        raise EvidenceError(
            f"{case_dir.name}: no frozen vendor work contract"
        ) from error
    expected_opcode_counts = dict(sorted(physical_contract.opcode_counts.items()))
    metadata = read_json(case_dir / "trace-metadata.json")
    expected_metadata = {
        **trace,
        "compile_repeats": 2,
        "byte_identical": True,
        "repeat_sha256": trace["sha256"],
    }
    if metadata != expected_metadata:
        raise EvidenceError(f"{case_dir.name}: trace metadata is stale")
    for kind in ("source", "compiled", "semantic"):
        first = (case_dir / f"{kind}.run-1.json").read_bytes()
        second = (case_dir / f"{kind}.run-2.json").read_bytes()
        if first != second:
            raise EvidenceError(f"{case_dir.name}: {kind} manifest repeats differ")
        read_json(case_dir / f"{kind}.run-1.json")
    compiled = read_json(case_dir / "compiled.run-1.json")
    source = read_json(case_dir / "source.run-1.json")
    semantic_run = read_json(case_dir / "semantic.run-1.json")
    case_definition = read_json(case_dir / "case-definition.json")
    decode_spec = read_json(case_dir / "decode-spec.json")
    if case_definition.get("case_id") != case_dir.name:
        raise EvidenceError(f"{case_dir.name}: case-definition identity differs")
    if compiled.get("source") != source:
        raise EvidenceError(f"{case_dir.name}: compiled/source manifests differ")
    if semantic_run.get("case") != case_definition:
        raise EvidenceError(f"{case_dir.name}: semantic case definition differs")
    if semantic_run.get("decode_spec") != decode_spec:
        raise EvidenceError(f"{case_dir.name}: semantic decode spec differs")
    logical_coverage = compiler_logical_work_coverage(compiled, first_trace)
    if not logical_coverage["complete"]:
        raise EvidenceError(
            f"{case_dir.name}: typed logical work/coverage is incomplete: "
            + "; ".join(logical_coverage["errors"])
        )
    if semantic_run.get("compiled_trace") != {
        "opcode_counts": trace["opcode_counts"],
        "eoc_count": 1,
        "command_count": trace["nonempty_lines"],
        "sha256": trace["sha256"],
        "command_shape_signature": command_shape_signature(first_trace),
    }:
        raise EvidenceError(f"{case_dir.name}: semantic compiled-trace binding differs")
    if semantic_run.get("logical_work_coverage") != logical_coverage:
        raise EvidenceError(f"{case_dir.name}: semantic logical-work coverage is stale")
    if compiled.get("trace", {}).get("sha256") != trace["sha256"]:
        raise EvidenceError(f"{case_dir.name}: compiled manifest trace hash differs")
    if compiled.get("trace", {}).get("eoc_count") != 1:
        raise EvidenceError(f"{case_dir.name}: compiled manifest lacks one EOC")
    vendor_join = read_json(case_dir / "vendor-evidence.json")
    try:
        vendor_physical = vendor_join["physical_work_contract"]
        physical_comparison = physical_work_comparison(
            cent_source_commit=CENT_SOURCE_COMMIT,
            vendor_trace_sha256=physical_contract.trace_sha256,
            vendor_opcode_counts=expected_opcode_counts,
            vendor_command_shape_signature=vendor_physical["command_shape_signature"],
            vendor_wr_abk_channel_coverage=vendor_physical["wr_abk_channel_coverage"],
            tenon_trace_text=first_trace,
            target_channels=int(compiled["geometry"]["channels"]),
            replica_count=int(decode_spec["replicas"]),
            channels_per_replica=int(decode_spec["channels_per_replica"]),
            logical_work_coverage=logical_coverage,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise EvidenceError(
            f"{case_dir.name}: physical-work evidence is incomplete"
        ) from error
    semantic = read_json(case_dir / "semantic-work-invariant.json")
    if semantic != {
        "schema": "tenon-cent-aim-semantic-evidence-v1",
        "case_id": case_dir.name,
        "timing_model": "SK hynix AiM Ramulator2",
        "timing_only": True,
        "numeric_inputs_consumed": False,
        "numeric_outputs_produced": False,
        "numeric_correctness_claim": "not_available_from_timing_simulator",
        "physical_work_contract": physical_comparison,
        "semantic_manifest": read_json(case_dir / "semantic.run-1.json"),
        "independent_manifest_repeats": 2,
        "manifests_byte_identical": True,
    }:
        raise EvidenceError(f"{case_dir.name}: semantic evidence is inconsistent")
    if "timing-only" not in _read_text(case_dir / "data-output-status.txt"):
        raise EvidenceError(f"{case_dir.name}: timing-only limitation is not disclosed")
    command = read_json(case_dir / "compile-command.json")
    if command.get("independent_build_compile_repeats") != 2:
        raise EvidenceError(
            f"{case_dir.name}: compile command does not require two runs"
        )
    if command.get("program_builder_arguments", {}).get("case_id") != case_dir.name:
        raise EvidenceError(f"{case_dir.name}: compile command names another case")
    if command.get("program_builder_arguments", {}).get("spec") != decode_spec:
        raise EvidenceError(f"{case_dir.name}: compile command spec differs")
    return trace


def _verify_vendor(
    case_dir: Path, vendor_root: Path, simulator: dict[str, Any]
) -> dict[str, Any]:
    join = read_json(case_dir / "vendor-evidence.json")
    case_id = case_dir.name
    vendor_case_dir = vendor_root / case_id
    verify_checksums(vendor_case_dir, require_complete=True)
    try:
        physical_contract = VENDOR_TRACE_CONTRACTS[case_id]
    except KeyError as error:
        raise EvidenceError(f"{case_id}: no frozen vendor work contract") from error
    expected_opcode_counts = dict(sorted(physical_contract.opcode_counts.items()))
    vendor_case, vendor_trace = validate_frozen_vendor_trace(
        vendor_case_dir,
        case_id=case_id,
        expected_trace_sha256=physical_contract.trace_sha256,
        expected_opcode_counts=expected_opcode_counts,
        expected_source_commit=CENT_SOURCE_COMMIT,
    )
    if join.get("case_checksum_manifest_sha256") != sha256_file(
        vendor_case_dir / "SHA256SUMS"
    ):
        raise EvidenceError(f"{case_id}: vendor checksum-manifest join is stale")
    if join.get("case_manifest_sha256") != sha256_file(vendor_case_dir / "case.json"):
        raise EvidenceError(f"{case_id}: vendor case-manifest join is stale")
    if join.get("vendor_case") != vendor_case:
        raise EvidenceError(f"{case_id}: joined vendor case differs")
    trace_path = vendor_case_dir / "input.trace"
    if join.get("trace_sha256") != sha256_file(trace_path):
        raise EvidenceError(f"{case_id}: vendor trace join is stale")
    vendor_trace_text = _read_text(trace_path)
    if join.get("physical_work_contract") != {
        "cent_source_commit": CENT_SOURCE_COMMIT,
        "vendor_trace_sha256": physical_contract.trace_sha256,
        "opcode_counts": expected_opcode_counts,
        "command_shape_signature": command_shape_signature(vendor_trace_text),
        "wr_abk_channel_coverage": opcode_active_channel_coverage(
            vendor_trace_text,
            "AiM_WR_ABK",
            target_channels=32,
        ),
    }:
        raise EvidenceError(f"{case_id}: joined physical-work contract is stale")
    sim_dir = vendor_case_dir / "installed-0f28a07b"
    summary_path = sim_dir / "summary.json"
    if join.get("simulator_summary_sha256") != sha256_file(summary_path):
        raise EvidenceError(f"{case_id}: vendor simulator-summary join is stale")
    summary = read_json(summary_path)
    if summary.get("trace_sha256") != vendor_trace["sha256"]:
        raise EvidenceError(f"{case_id}: vendor summary trace hash is stale")
    cross_fields = {
        "simulator_commit": "required_commit",
        "simulator_binary_sha256": "binary_sha256",
        "simulator_library_sha256": "library_sha256",
        "config_sha256": "config_sha256",
    }
    for joined_field, current_field in cross_fields.items():
        expected = simulator[current_field]
        if join.get(joined_field) != expected or summary.get(joined_field) != expected:
            raise EvidenceError(
                f"{case_id}: vendor {joined_field} is not comparison-identical"
            )
    command = read_json(sim_dir / "command.argv.json")
    _verify_command(command, simulator)
    try:
        binary_index = command.index("/work/build/ramulator2")
        image_id = command[binary_index - 1]
    except (ValueError, IndexError) as error:
        raise EvidenceError(f"{case_id}: malformed vendor simulator command") from error
    if (
        image_id != simulator["docker_image_immutable_id"]
        or join.get("docker_image_immutable_id") != image_id
    ):
        raise EvidenceError(f"{case_id}: vendor and Tenon immutable images differ")
    raw_runs = []
    cycles = []
    for repeat in (1, 2):
        if _read_exit_code(sim_dir / f"run-{repeat}.exit-code.txt") != 0:
            raise EvidenceError(f"{case_id}: vendor simulator repeat {repeat} failed")
        stdout_path = sim_dir / f"run-{repeat}.stdout.raw.log"
        stderr_path = sim_dir / f"run-{repeat}.stderr.raw.log"
        parsed = parse_simulator_stdout(_read_text(stdout_path))
        reconcile_request_counts(
            vendor_trace["opcode_counts"], parsed["total_request_counts"]
        )
        metrics_path = sim_dir / f"run-{repeat}.metrics.json"
        metrics = read_json(metrics_path)
        if metrics.get("memory_system_cycles") != parsed["memory_system_cycles"]:
            raise EvidenceError(f"{case_id}: vendor metrics disagree with raw stdout")
        raw_runs.append(
            {
                "repeat": repeat,
                "exit_code": 0,
                "stdout_sha256": sha256_file(stdout_path),
                "stderr_sha256": sha256_file(stderr_path),
                "metrics_sha256": sha256_file(metrics_path),
                "memory_system_cycles": parsed["memory_system_cycles"],
                "eoc_requests": parsed["eoc_requests"],
            }
        )
        cycles.append(parsed["memory_system_cycles"])
    if join.get("raw_runs") != raw_runs:
        raise EvidenceError(f"{case_id}: joined vendor raw-run hashes are stale")
    if cycles != [summary.get("memory_system_cycles")] * 2:
        raise EvidenceError(f"{case_id}: vendor cycle repeats do not support summary")
    if join.get("memory_system_cycles") != cycles[0]:
        raise EvidenceError(f"{case_id}: joined vendor cycles are stale")
    return join


def _canonical_csv(rows: list[dict[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=COMPARISON_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _verify_common_evidence(root: Path, case_dir: Path) -> None:
    source = root / "provenance"
    copied = case_dir / "common-evidence"
    source_files = {
        path.relative_to(source): sha256_file(path)
        for path in source.rglob("*")
        if path.is_file()
    }
    copied_files = {
        path.relative_to(copied): sha256_file(path)
        for path in copied.rglob("*")
        if path.is_file()
    }
    if copied_files != source_files:
        raise EvidenceError(f"{case_dir.name}: standalone common evidence differs")


def _compiled_reuse_window(compiled: dict[str, Any], candidate: int) -> int:
    """Read the target's mapper-visible reuse window from a compiled manifest."""
    try:
        capacity = int(compiled["geometry"]["reuse_window"])
    except (KeyError, TypeError, ValueError) as error:
        raise EvidenceError(
            f"reuse candidate {candidate}: target reuse-window manifest is incomplete"
        ) from error
    if capacity <= 0:
        raise EvidenceError(
            f"reuse candidate {candidate}: target reuse window must be positive"
        )
    return capacity


def _verify_reuse_calibration(root: Path, simulator: dict[str, Any]) -> dict[str, Any]:
    calibration_root = root / "calibration/reuse-groups"
    verify_checksums(calibration_root, require_complete=True)
    report = read_json(calibration_root / "results.json")
    candidates = report.get("legal_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise EvidenceError("reuse calibration has no legal candidates")
    if candidates != sorted(set(int(candidate) for candidate in candidates)):
        raise EvidenceError("reuse calibration candidates are not unique and sorted")
    rows = []
    observed_capacity = None
    reference_work = None
    for candidate in candidates:
        candidate_dir = calibration_root / f"candidate-{candidate}"
        verify_checksums(candidate_dir, require_complete=True)
        first_trace = _read_text(candidate_dir / "input.trace")
        second_trace = _read_text(candidate_dir / "input.repeat.trace")
        if first_trace.encode("utf-8") != second_trace.encode("utf-8"):
            raise EvidenceError(f"reuse candidate {candidate}: compile repeats differ")
        trace = validate_trace_text(first_trace, label=f"reuse-candidate-{candidate}")
        metadata = read_json(candidate_dir / "trace-metadata.json")
        if metadata != {**trace, "compile_repeats": 2, "byte_identical": True}:
            raise EvidenceError(f"reuse candidate {candidate}: trace metadata is stale")
        for kind in ("source", "compiled"):
            if (candidate_dir / f"{kind}.run-1.json").read_bytes() != (
                candidate_dir / f"{kind}.run-2.json"
            ).read_bytes():
                raise EvidenceError(
                    f"reuse candidate {candidate}: {kind} repeats differ"
                )
        source = read_json(candidate_dir / "source.run-1.json")
        compiled = read_json(candidate_dir / "compiled.run-1.json")
        try:
            source_operation = source["operations"][0]
        except (KeyError, IndexError, TypeError) as error:
            raise EvidenceError(
                f"reuse candidate {candidate}: source manifest is incomplete"
            ) from error
        capacity = _compiled_reuse_window(compiled, candidate)
        if source_operation.get("reuse_group_size") != candidate:
            raise EvidenceError(
                f"reuse candidate {candidate}: source manifest selects another group"
            )
        if source_operation.get("reuse_group_candidates") != candidates:
            raise EvidenceError(
                f"reuse candidate {candidate}: source manifest legal set differs"
            )
        if observed_capacity is None:
            observed_capacity = capacity
        elif observed_capacity != capacity:
            raise EvidenceError(
                "reuse calibration mixes target reuse-window capacities"
            )
        work = read_json(candidate_dir / "calibration-work.json")
        if work.get("reuse_group_candidate") != candidate:
            raise EvidenceError(f"reuse candidate {candidate}: work manifest differs")
        if work.get("legal_candidates") != candidates:
            raise EvidenceError(f"reuse candidate {candidate}: legal set differs")
        invariant_work = {
            key: value for key, value in work.items() if key != "reuse_group_candidate"
        }
        if reference_work is None:
            reference_work = invariant_work
        elif reference_work != invariant_work:
            raise EvidenceError("reuse calibration candidates measure different work")
        summary = _verify_simulator_runs(candidate_dir, trace, simulator)
        rows.append(
            {
                "reuse_group_size": candidate,
                "memory_system_cycles": summary["memory_system_cycles"],
                "trace_commands": trace["nonempty_lines"],
                "trace_sha256": trace["sha256"],
                "evidence": (
                    f"calibration/reuse-groups/candidate-{candidate}/"
                    "simulator/summary.json"
                ),
            }
        )
    derived_candidates = []
    value = 1
    while value <= observed_capacity:
        derived_candidates.append(value)
        value *= 2
    if derived_candidates[-1] != observed_capacity:
        derived_candidates.append(observed_capacity)
    if candidates != derived_candidates:
        raise EvidenceError(
            "reuse calibration candidates were not derived from reuse-window capacity"
        )
    best = min(
        rows, key=lambda row: (row["memory_system_cycles"], row["reuse_group_size"])
    )
    expected_report = {
        "schema": "tenon-aim-reuse-calibration-v1",
        "work": {
            "outputs": reference_work["outputs"],
            "reduction": reference_work["reduction"],
            "replicas": reference_work["replicas"],
            "channels_per_replica": reference_work["channels_per_replica"],
            "dimension_source": "DecodeSpec.D",
        },
        "legal_candidate_source": "powers of two through target.mac_reg.slots",
        "legal_candidates": candidates,
        "simulator_commit": EXPECTED_SIMULATOR_COMMIT,
        "compile_repeats_per_candidate": 2,
        "simulator_repeats_per_candidate": 2,
        "rows": rows,
        "measured_best_legal_candidate": best["reuse_group_size"],
        "measured_best_cycles": best["memory_system_cycles"],
        "activation_policy": "advisory evidence; campaign does not mutate compiler policy",
    }
    if report != expected_report:
        raise EvidenceError(
            "reuse calibration report is not derivable from raw evidence"
        )
    with (calibration_root / "results.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        csv_rows = list(csv.DictReader(handle))
    expected_csv = [{field: str(value) for field, value in row.items()} for row in rows]
    if csv_rows != expected_csv:
        raise EvidenceError("reuse calibration CSV differs from raw evidence")
    if _read_text(calibration_root / "RESULTS.md") != render_reuse_calibration_markdown(
        rows, reference_work["outputs"]
    ):
        raise EvidenceError("reuse calibration Markdown differs from raw evidence")
    return report


def verify_campaign(
    root: Path,
    vendor_root: Path,
    *,
    tenon_root: Path | None = None,
    simulator_root: Path | None = None,
) -> dict[str, int]:
    root = root.resolve()
    vendor_root = vendor_root.resolve()
    verify_checksums(root, require_complete=True)
    compiler, simulator = _verify_provenance(
        root,
        tenon_root=None if tenon_root is None else tenon_root.resolve(),
        simulator_root=(None if simulator_root is None else simulator_root.resolve()),
    )
    campaign = read_json(root / "campaign.json")
    expected_campaign_fields = {
        "schema": "tenon-cent-aim-campaign-v1",
        "case_count": len(EXPECTED_CASE_IDS),
        "simulator_commit": EXPECTED_SIMULATOR_COMMIT,
        "cycle_metric": "global memory_system_cycles",
        "lower_is_better": True,
        "modeled_clock_GHz": 2.0,
    }
    for field, expected in expected_campaign_fields.items():
        if campaign.get(field) != expected:
            raise EvidenceError(f"campaign {field} is stale")
    if campaign.get("case_ids") != list(EXPECTED_CASE_IDS):
        raise EvidenceError("campaign does not contain the frozen 14-case suite")
    if (
        campaign.get("compile_repeats_per_case") != 2
        or campaign.get("simulator_repeats_per_case") != 2
    ):
        raise EvidenceError(
            "campaign does not require exactly two compile/simulator runs"
        )
    if (
        campaign.get("timing_only") is not True
        or campaign.get("numeric_correctness_claim")
        != "not_available_from_timing_simulator"
    ):
        raise EvidenceError("campaign overstates timing-simulator correctness")
    if campaign.get("compiler_source_tree_sha256") != compiler["source_tree_sha256"]:
        raise EvidenceError("campaign compiler hash differs from provenance")
    hash_fields = {
        "simulator_binary_sha256": "binary_sha256",
        "simulator_library_sha256": "library_sha256",
        "simulator_config_sha256": "config_sha256",
        "docker_image_immutable_id": "docker_image_immutable_id",
    }
    for campaign_field, simulator_field in hash_fields.items():
        if campaign.get(campaign_field) != simulator[simulator_field]:
            raise EvidenceError(f"campaign {campaign_field} differs from provenance")
    expected_rows = []
    for case_id in EXPECTED_CASE_IDS:
        case_dir = root / "cases" / case_id
        verify_checksums(case_dir, require_complete=True)
        _verify_common_evidence(root, case_dir)
        if read_json(case_dir / "campaign-provenance.json") != {
            "compiler": compiler,
            "simulator": simulator,
        }:
            raise EvidenceError(f"{case_id}: copied campaign provenance differs")
        trace = _verify_compile(case_dir)
        sim_summary = _verify_simulator_runs(case_dir, trace, simulator)
        vendor = _verify_vendor(case_dir, vendor_root, simulator)
        vendor_case = vendor["vendor_case"]
        row = comparison_row(
            case_id=case_id,
            kernel=str(vendor_case["kernel"]),
            sequence_length=int(vendor_case["sequence_length"]),
            vendor_cycles=vendor["memory_system_cycles"],
            tenon_cycles=sim_summary["memory_system_cycles"],
            trace_sha256=trace["sha256"],
        )
        expected_rows.append(row)
        local_row = {
            **row,
            "tenon_evidence": "simulator/summary.json",
            "vendor_evidence": "vendor-evidence.json",
        }
        if read_json(case_dir / "comparison.json") != {
            "schema": "tenon-cent-aim-case-comparison-v1",
            "row": local_row,
        }:
            raise EvidenceError(f"{case_id}: standalone comparison JSON differs")
        if _read_text(case_dir / "comparison.csv") != _canonical_csv([local_row]):
            raise EvidenceError(f"{case_id}: standalone comparison CSV differs")
        if _read_text(case_dir / "RESULT.md") != render_comparison_markdown(
            [local_row]
        ):
            raise EvidenceError(f"{case_id}: standalone result Markdown differs")
        print(f"PASS {case_id}")
    comparison = read_json(root / "comparison.json")
    if comparison != {
        "schema": "tenon-cent-aim-comparison-v1",
        "metric": "global memory_system_cycles",
        "lower_is_better": True,
        "simulator_commit": EXPECTED_SIMULATOR_COMMIT,
        "totals": comparison_totals(expected_rows),
        "rows": expected_rows,
    }:
        raise EvidenceError("comparison JSON is not derivable from retained evidence")
    if _read_text(root / "comparison.csv") != _canonical_csv(expected_rows):
        raise EvidenceError("comparison CSV is not derivable from retained evidence")
    if _read_text(root / "COMPARISON.md") != render_comparison_markdown(expected_rows):
        raise EvidenceError(
            "comparison Markdown is not derivable from retained evidence"
        )
    totals = comparison_totals(expected_rows)
    if campaign.get("comparison_totals") != totals:
        raise EvidenceError("campaign totals differ from the comparison table")
    calibration_ref = campaign.get("reuse_calibration")
    if calibration_ref is not None:
        calibration = _verify_reuse_calibration(root, simulator)
        if calibration_ref != {
            "results": "calibration/reuse-groups/results.json",
            "measured_best_legal_candidate": calibration[
                "measured_best_legal_candidate"
            ],
        }:
            raise EvidenceError("campaign reuse-calibration reference is stale")
        print("PASS reuse-group calibration")
    print(
        "PASS campaign: "
        f"{totals['case_count']} rows, {totals['wins']} wins, "
        f"{totals['ties']} ties, {totals['losses']} losses; "
        "checksums/logs/traces/repeats/tables reconciled"
    )
    return totals


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument(
        "--tenon-root",
        type=Path,
        help="optionally require current compiler sources to match the snapshot",
    )
    parser.add_argument(
        "--simulator-root",
        type=Path,
        help="optionally require current simulator binary/config to match",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        verify_campaign(
            args.bundle,
            args.vendor_root,
            tenon_root=args.tenon_root,
            simulator_root=args.simulator_root,
        )
    except EvidenceError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
