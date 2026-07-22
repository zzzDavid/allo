# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the CENT/AiM evidence campaign and verifier."""

from pathlib import Path
import shlex
from types import SimpleNamespace

import pytest

from benchmarks.cent_aim import verify_campaign as verify_campaign_module

from benchmarks.cent_aim.evidence import (
    EvidenceError,
    command_shape_signature,
    command_shape_signature_delta,
    compiler_logical_work_coverage,
    comparison_row,
    comparison_totals,
    parse_simulator_stdout,
    opcode_active_channel_coverage,
    physical_work_comparison,
    reconcile_request_counts,
    render_comparison_markdown,
    sha256_bytes,
    sha256_file,
    validate_trace_text,
    verify_checksums,
    write_checksums,
    write_json,
    write_text,
)
from benchmarks.cent_aim.run_campaign import _reuse_candidates, _write_case_comparison
from benchmarks.cent_aim.verify_campaign import (
    _compiled_reuse_window,
    _verify_compile,
    _verify_simulator_runs,
)


def _stdout(cycles=123):
    return (
        "Statistics:\n"
        "  total_num_AiM_ISR_EOC_requests: 1\n"
        f"  memory_system_cycles: {cycles}\n"
    )


def test_trace_and_raw_simulator_evidence_fail_closed():
    trace = validate_trace_text("AiM WR_BIAS 0 1\nAiM EOC\n")
    assert trace["eoc_count"] == 1
    assert trace["opcode_counts"] == {"AiM_EOC": 1, "AiM_WR_BIAS": 1}

    with pytest.raises(EvidenceError, match="exactly one trailing"):
        validate_trace_text("AiM EOC\nAiM WR_BIAS 0 1\n")
    with pytest.raises(EvidenceError, match="exactly one trailing"):
        validate_trace_text("AiM EOC\nAiM EOC\n")
    with pytest.raises(EvidenceError, match="exactly one global"):
        parse_simulator_stdout(_stdout() + "memory_system_cycles: 124\n")

    parsed = parse_simulator_stdout(_stdout())
    reconcile_request_counts({"AiM_EOC": 1}, parsed["total_request_counts"])
    with pytest.raises(EvidenceError, match="does not match trace"):
        reconcile_request_counts({"AiM_EOC": 2}, parsed["total_request_counts"])


def test_command_shape_signature_tracks_opsize_and_mask_but_not_addresses():
    vendor = "AiM MAC_ABK 2 0x1 7\n" "AiM RD_MAC 0 0x1\n" "AiM EOC\n"
    address_only_change = "AiM RD_MAC 99 0x1\n" "AiM MAC_ABK 2 0x1 1234\n" "AiM EOC\n"
    legal_shape_change = "AiM MAC_ABK 1 0x3 1234\n" "AiM RD_MAC 99 0x1\n" "AiM EOC\n"
    vendor_signature = command_shape_signature(vendor)
    assert command_shape_signature(address_only_change) == vendor_signature
    delta = command_shape_signature_delta(
        vendor_signature, command_shape_signature(legal_shape_change)
    )
    assert delta["equal"] is False
    assert delta["equality_required"] is False
    assert {record["op_size"] for record in delta["records"]} >= {1, 2}
    assert {record["active_mask_fanout"] for record in delta["records"]} >= {
        1,
        2,
    }


def test_logical_coverage_is_derived_from_trace_capacity_and_spans():
    trace_text = (
        "AiM WR_GB 1 0 0x1\n"
        "AiM WR_BIAS 0 0x1\n"
        "AiM MAC_ABK 1 0x1 0\n"
        "AiM RD_MAC 0 0x1\n"
        "AiM EOC\n"
    )
    source_operation = {
        "kind": "contraction",
        "name": "tiny",
        "outputs": 16,
        "reduction": 16,
        "batches": 1,
        "replicas": 1,
        "input_source": "gb",
        "batch_mapping": "flattened",
    }
    lowered_operation = {
        "index": 0,
        "kind": "contraction",
        "name": "tiny",
        "command_span": [0, 4],
        "command_count": 4,
        "logical_scalar_macs": 256,
        "physical_mac_slots": 256,
        "physical_padded_scalar_macs": 256,
        "unique_logical_gb_payload_elements": 16,
        "physical_wr_gb_elements": 16,
        "physical_gb_transfer_elements": 16,
        "gb_replication_and_reload_factor": 1.0,
    }
    compiled = {
        "source": {"operations": [source_operation]},
        "operations": [lowered_operation],
        "geometry": {"channels": 32, "banks": 16, "bank_groups": 4, "lanes": 16},
        "trace": {"body_command_count": 4},
    }
    coverage = compiler_logical_work_coverage(compiled, trace_text)
    assert coverage["complete"] is True
    assert coverage["contractions"][0]["trace_derived_physical_mac_slots"] == 256

    incomplete = {
        **compiled,
        "source": {
            "operations": [{**source_operation, "outputs": 32}],
        },
    }
    bad = compiler_logical_work_coverage(incomplete, trace_text)
    assert bad["complete"] is False
    assert any("does not cover" in error for error in bad["errors"])


def test_elementwise_coverage_reconciles_linear_layout_capacity():
    trace_text = "AiM EWMUL 1 0x1 9\nAiM EOC\n"
    compiled = {
        "source": {
            "operations": [
                {
                    "kind": "elementwise",
                    "name": "mul",
                    "operation": "mul",
                    "elements": 64,
                    "replicas": 1,
                }
            ]
        },
        "operations": [
            {
                "index": 0,
                "kind": "elementwise",
                "name": "mul",
                "command_span": [0, 1],
                "command_count": 1,
                "logical_elements": 64,
                "physical_element_slots": 64,
            }
        ],
        "geometry": {"channels": 32, "banks": 16, "bank_groups": 4, "lanes": 16},
        "trace": {"body_command_count": 1},
    }
    coverage = compiler_logical_work_coverage(compiled, trace_text)
    assert coverage["complete"] is True
    assert coverage["elementwise"][0]["trace_derived_physical_element_slots"] == 64


def test_checksums_reject_tampering_and_unmanifested_files(tmp_path):
    write_text(tmp_path / "a.txt", "original\n")
    write_checksums(tmp_path)
    verify_checksums(tmp_path)

    write_text(tmp_path / "a.txt", "tampered\n")
    with pytest.raises(EvidenceError, match="checksum mismatch"):
        verify_checksums(tmp_path)
    write_text(tmp_path / "a.txt", "original\n")
    write_text(tmp_path / "extra.txt", "not in the manifest\n")
    with pytest.raises(EvidenceError, match="unmanifested"):
        verify_checksums(tmp_path)


def test_comparison_direction_and_markdown_are_derived():
    rows = [
        comparison_row(
            case_id="win",
            kernel="projection",
            sequence_length=128,
            vendor_cycles=120,
            tenon_cycles=100,
            trace_sha256="1" * 64,
        ),
        comparison_row(
            case_id="tie",
            kernel="elementwise",
            sequence_length=128,
            vendor_cycles=100,
            tenon_cycles=100,
            trace_sha256="2" * 64,
        ),
        comparison_row(
            case_id="loss",
            kernel="attention",
            sequence_length=512,
            vendor_cycles=100,
            tenon_cycles=125,
            trace_sha256="3" * 64,
        ),
    ]
    assert comparison_totals(rows) == {
        "case_count": 3,
        "wins": 1,
        "ties": 1,
        "losses": 1,
        "matches_or_beats": 2,
    }
    markdown = render_comparison_markdown(rows)
    assert "2/3 cases" in markdown
    assert "| `win`" in markdown and "WIN" in markdown
    assert "timing-only" in markdown


def _write_compiler_evidence(case_dir: Path):
    trace_text = "AiM EOC\n"
    trace = validate_trace_text(trace_text, label=case_dir.name)
    case_definition = {"case_id": case_dir.name}
    decode_spec = {"L": 1, "replicas": 1, "channels_per_replica": 32}
    source = {"schema": "synthetic-source", "operations": []}
    write_text(case_dir / "input.trace", trace_text)
    write_text(case_dir / "input.repeat.trace", trace_text)
    write_json(case_dir / "case-definition.json", case_definition)
    write_json(case_dir / "decode-spec.json", decode_spec)
    write_json(case_dir / "source.run-1.json", source)
    write_json(case_dir / "source.run-2.json", source)
    compiled = {
        "source": source,
        "operations": [],
        "geometry": {"channels": 32, "banks": 16, "bank_groups": 4, "lanes": 16},
        "trace": {
            "sha256": trace["sha256"],
            "eoc_count": 1,
            "body_command_count": 0,
        },
    }
    write_json(case_dir / "compiled.run-1.json", compiled)
    write_json(case_dir / "compiled.run-2.json", compiled)
    logical_coverage = compiler_logical_work_coverage(compiled, trace_text)
    signature = command_shape_signature(trace_text)
    semantics = {
        "case": case_definition,
        "decode_spec": decode_spec,
        "compiled_trace": {
            "opcode_counts": trace["opcode_counts"],
            "eoc_count": 1,
            "command_count": trace["nonempty_lines"],
            "sha256": trace["sha256"],
            "command_shape_signature": signature,
        },
        "logical_work_coverage": logical_coverage,
    }
    write_json(case_dir / "semantic.run-1.json", semantics)
    write_json(case_dir / "semantic.run-2.json", semantics)
    write_json(
        case_dir / "trace-metadata.json",
        {
            **trace,
            "compile_repeats": 2,
            "byte_identical": True,
            "repeat_sha256": trace["sha256"],
        },
    )
    vendor_wr_coverage = opcode_active_channel_coverage(
        trace_text, "AiM_WR_ABK", target_channels=32
    )
    physical = physical_work_comparison(
        cent_source_commit="synthetic-cent",
        vendor_trace_sha256="d" * 64,
        vendor_opcode_counts={"AiM_EOC": 1},
        vendor_command_shape_signature=signature,
        vendor_wr_abk_channel_coverage=vendor_wr_coverage,
        tenon_trace_text=trace_text,
        target_channels=32,
        replica_count=1,
        channels_per_replica=32,
        logical_work_coverage=logical_coverage,
    )
    write_json(
        case_dir / "semantic-work-invariant.json",
        {
            "schema": "tenon-cent-aim-semantic-evidence-v1",
            "case_id": case_dir.name,
            "timing_model": "SK hynix AiM Ramulator2",
            "timing_only": True,
            "numeric_inputs_consumed": False,
            "numeric_outputs_produced": False,
            "numeric_correctness_claim": "not_available_from_timing_simulator",
            "physical_work_contract": physical,
            "semantic_manifest": semantics,
            "independent_manifest_repeats": 2,
            "manifests_byte_identical": True,
        },
    )
    write_text(case_dir / "data-output-status.txt", "timing-only model\n")
    write_json(
        case_dir / "compile-command.json",
        {
            "program_builder_arguments": {
                "case_id": case_dir.name,
                "spec": decode_spec,
            },
            "independent_build_compile_repeats": 2,
        },
    )
    write_json(
        case_dir / "vendor-evidence.json",
        {
            "physical_work_contract": {
                "command_shape_signature": signature,
                "wr_abk_channel_coverage": vendor_wr_coverage,
            }
        },
    )
    return trace


def _write_simulator_evidence(case_dir: Path, trace, simulator, cycles=123):
    sim_dir = case_dir / "simulator"
    write_text(sim_dir / "config.yaml", "config\n")
    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--user",
        "1:1",
        "-e",
        "LD_LIBRARY_PATH=/work",
        "-v",
        "/sim:/work:ro",
        "-v",
        f"{case_dir}:/evidence:ro",
        simulator["docker_image_immutable_id"],
        "/work/build/ramulator2",
        "-f",
        f"/work/{simulator['config_relative']}",
        "-t",
        "/evidence/input.trace",
    ]
    write_json(sim_dir / "command.argv.json", command)
    write_text(sim_dir / "command.txt", shlex.join(command) + "\n")
    runs = []
    for repeat in (1, 2):
        stdout_path = sim_dir / f"run-{repeat}.stdout.raw.log"
        stderr_path = sim_dir / f"run-{repeat}.stderr.raw.log"
        write_text(stdout_path, _stdout(cycles))
        write_text(stderr_path, "")
        write_text(sim_dir / f"run-{repeat}.exit-code.txt", "0\n")
        parsed = parse_simulator_stdout(_stdout(cycles))
        metrics = {
            **parsed,
            "repeat": repeat,
            "exit_code": 0,
            "trace_sha256": trace["sha256"],
            "stdout_sha256": sha256_file(stdout_path),
            "stderr_sha256": sha256_file(stderr_path),
        }
        write_json(sim_dir / f"run-{repeat}.metrics.json", metrics)
        runs.append(metrics)
    write_json(
        sim_dir / "summary.json",
        {
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
            "memory_system_cycles": cycles,
            "latency_us_at_2GHz": cycles * 0.5 / 1000.0,
            "runs": runs,
        },
    )


def test_compile_and_simulator_verifier_rederive_every_field(tmp_path, monkeypatch):
    monkeypatch.setattr(
        verify_campaign_module,
        "VENDOR_TRACE_CONTRACTS",
        {
            "synthetic_case": SimpleNamespace(
                trace_sha256="d" * 64,
                opcode_counts={"AiM_EOC": 1},
            )
        },
    )
    monkeypatch.setattr(verify_campaign_module, "CENT_SOURCE_COMMIT", "synthetic-cent")
    case_dir = tmp_path / "synthetic_case"
    case_dir.mkdir()
    trace = _write_compiler_evidence(case_dir)
    simulator = {
        "required_commit": "0f28a07bdb83e42b9305ad3d45410ebd3aa2c091",
        "binary_sha256": "a" * 64,
        "library_sha256": "b" * 64,
        "config_relative": "test/example.yaml",
        "config_sha256": sha256_bytes(b"config\n"),
        "docker_image_immutable_id": "sha256:" + "c" * 64,
    }
    _write_simulator_evidence(case_dir, trace, simulator)

    assert _verify_compile(case_dir) == trace
    summary = _verify_simulator_runs(case_dir, trace, simulator)
    assert summary["memory_system_cycles"] == 123

    metrics = case_dir / "simulator/run-2.metrics.json"
    value = metrics.read_text(encoding="utf-8").replace(
        '"memory_system_cycles": 123', '"memory_system_cycles": 999'
    )
    write_text(metrics, value)
    with pytest.raises(EvidenceError, match="do not derive from raw logs"):
        _verify_simulator_runs(case_dir, trace, simulator)


def test_reuse_candidates_are_derived_from_target_capacity():
    class MacReg:
        slots = 16

    class Target:
        mac_reg = MacReg()

    assert _reuse_candidates(Target()) == (1, 2, 4, 8, 16)

    Target.mac_reg.slots = 12
    assert _reuse_candidates(Target()) == (1, 2, 4, 8, 12)


def test_calibration_verifier_reads_compiled_reuse_window():
    assert _compiled_reuse_window({"geometry": {"reuse_window": 32}}, 1) == 32
    with pytest.raises(EvidenceError, match="reuse-window manifest is incomplete"):
        _compiled_reuse_window({"geometry": {"accumulator_slots": 32}}, 1)


def test_standalone_verifier_disables_bytecode_writes(tmp_path):
    row = comparison_row(
        case_id="synthetic",
        kernel="projection",
        sequence_length=1,
        vendor_cycles=10,
        tenon_cycles=10,
        trace_sha256="1" * 64,
    )
    _write_case_comparison(tmp_path, row)
    instructions = (tmp_path / "VERIFY.md").read_text(encoding="utf-8")
    assert "PYTHONDONTWRITEBYTECODE=1" in instructions
    assert "python -B -m benchmarks.cent_aim.verify_case" in instructions
