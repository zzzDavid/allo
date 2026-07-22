# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for strict uPIMulator calibration evidence."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import pytest

from allo.pim.upmem_calibration import (
    REPORT_SCHEMA,
    UPMEMCounterError,
    UPMEMMetadataError,
    UPMEMRankingError,
    build_calibration_report,
    build_candidate_row,
    compute_ranking_metrics,
    load_candidate_row,
    parse_metadata_json,
    parse_provenance_metadata,
    parse_result_metadata,
    parse_upimulator_log,
    spearman_rank_correlation,
)


def _log(
    *,
    logic_cycle=100,
    run=50,
    etc=10,
    dma=15,
    backpressure=25,
):
    # Deliberately shuffled among unrelated counters, as in archived log.txt.
    return f"""
RowBuffer[0_0_0]_read_bytes: 4096
Logic[0_0_0]_backpressure: {backpressure}
ThreadScheduler[0_0_0]_breakdown_dma: {dma}
Logic[0_0_0]_active_tasklets_12: 70
ThreadScheduler[0_0_0]_breakdown_run: {run}
Logic[0_0_0]_logic_cycle: {logic_cycle}
ThreadScheduler[0_0_0]_breakdown_etc: {etc}
"""


def _result(*, cycles=100, compiler="Tenon", workload="gemv"):
    return {
        "schema": "upimulator-kernel-result-v1",
        "compiler": compiler,
        "workload": workload,
        "status": "pass",
        "compile_exit_code": 0,
        "simulation_exit_code": 0,
        "simulator": "uPIMulator Go",
        "num_dpus": 1,
        "num_tasklets": 12,
        "logic_frequency_mhz": 350,
        "executions": 1,
        "logic_cycles": cycles,
        "predicted_kernel_ms": cycles / 350000,
        "oracle_validation": {"status": "pass", "scope": "complete output"},
    }


def _provenance(*, compiler="Tenon", workload="gemv"):
    return {
        "compiler": compiler,
        "compiler_commit": "a" * 40,
        "workload": workload,
        "num_dpus": 1,
        "num_tasklets": 12,
        "num_executions": 1,
        "fixture_manifest": "fixture/manifest.json",
    }


def _row(
    candidate,
    measured,
    predicted,
    *,
    candidate_set="gemv",
    run=None,
):
    run = measured - 50 if run is None else run
    return build_candidate_row(
        candidate_set=candidate_set,
        candidate_id=candidate,
        log_text=_log(
            logic_cycle=measured,
            run=run,
            etc=10,
            dma=15,
            backpressure=25,
        ),
        result_metadata=_result(cycles=measured, workload=candidate_set),
        provenance_metadata=_provenance(workload=candidate_set),
        predicted_logic_cycle=predicted,
        cost_features={"mram_bytes": 4096, "layout": "linear"},
        prediction_fields={"model": "upmem-cost-v2", "compute_cycles": predicted - 20},
    )


def test_archived_counter_keys_parse_and_verify_exact_decomposition():
    counters = parse_upimulator_log(_log())

    assert counters.logic_cycle == 100
    assert counters.logic_cycles == 100
    assert counters.breakdown_run == 50
    assert counters.breakdown_etc == 10
    assert counters.breakdown_dma == 15
    assert counters.backpressure == 25
    assert counters.component_sum == counters.logic_cycle
    assert counters.manifest()["decomposition_verified"] is True


@pytest.mark.parametrize(
    "broken,match",
    (
        (
            _log().replace("ThreadScheduler[0_0_0]_breakdown_dma: 15\n", ""),
            "missing.*breakdown_dma",
        ),
        (
            _log() + "Logic[0_0_0]_logic_cycle: 100\n",
            "duplicate.*logic_cycle",
        ),
        (
            _log(logic_cycle=101),
            "inconsistent.*decomposition",
        ),
        (
            _log().replace(
                "Logic[0_0_0]_backpressure: 25", "Logic[0_0_0]_backpressure: nope"
            ),
            "unsigned decimal",
        ),
        (
            _log() + "Logic[0_0_1]_logic_cycle: 7\n",
            "unexpected DPU",
        ),
    ),
)
def test_counter_parser_fails_closed_on_missing_duplicate_malformed_or_mixed_dpu(
    broken,
    match,
):
    with pytest.raises(UPMEMCounterError, match=match):
        parse_upimulator_log(broken)


def test_metadata_json_rejects_duplicate_keys_and_result_counter_mismatch():
    with pytest.raises(UPMEMMetadataError, match="duplicate JSON object key 'status'"):
        parse_metadata_json('{"status": "pass", "status": "fail"}')

    counters = parse_upimulator_log(_log())
    with pytest.raises(UPMEMMetadataError, match="does not match the uPIMulator log"):
        parse_result_metadata(json.dumps(_result(cycles=101)), counters=counters)

    inconsistent_ms = _result()
    inconsistent_ms["predicted_kernel_ms"] = 123.0
    with pytest.raises(UPMEMMetadataError, match="predicted_kernel_ms is inconsistent"):
        parse_result_metadata(json.dumps(inconsistent_ms), counters=counters)


def test_result_and_provenance_must_describe_the_same_execution():
    result = parse_result_metadata(json.dumps(_result()))

    unsplit = _provenance()
    unsplit["subrun"] = None
    assert (
        parse_provenance_metadata(json.dumps(unsplit), result_metadata=result)[
            "workload"
        ]
        == "gemv"
    )

    with pytest.raises(UPMEMMetadataError, match="compiler fields disagree"):
        parse_provenance_metadata(
            json.dumps(_provenance(compiler="Other")),
            result_metadata=result,
        )
    with pytest.raises(UPMEMMetadataError, match="workload does not match"):
        parse_provenance_metadata(
            json.dumps(_provenance(workload="atax")),
            result_metadata=result,
        )


def test_normalized_candidate_row_retains_cost_prediction_and_provenance_fields():
    row = _row("linear-layout-tile-16", 100, 90)
    manifest = row.manifest()

    assert manifest["schema"] == "upmem-calibration-candidate-v1"
    assert manifest["candidate_set"] == "gemv"
    assert manifest["candidate_id"] == "linear-layout-tile-16"
    assert manifest["logic_cycle"] == 100
    assert manifest["predicted_logic_cycle"] == 90.0
    assert manifest["cost_features"] == {
        "layout": "linear",
        "mram_bytes": 4096,
    }
    assert manifest["prediction_fields"]["model"] == "upmem-cost-v2"
    assert manifest["provenance_metadata"]["compiler_commit"] == "a" * 40
    assert row.compiler_revision == "a" * 40
    with pytest.raises(TypeError):
        row.cost_features["mram_bytes"] = 1
    with pytest.raises(FrozenInstanceError):
        row.candidate_id = "forged"


def test_prediction_can_be_supplied_as_a_named_prediction_field():
    row = build_candidate_row(
        candidate_set="gemv",
        candidate_id="candidate",
        log_text=_log(),
        result_metadata=_result(),
        provenance_metadata=_provenance(),
        prediction_fields={"predicted_cycles": 105},
    )

    assert row.predicted_logic_cycle == 105.0
    with pytest.raises(UPMEMMetadataError, match="disagrees"):
        build_candidate_row(
            candidate_set="gemv",
            candidate_id="bad-candidate",
            log_text=_log(),
            result_metadata=_result(),
            provenance_metadata=_provenance(),
            predicted_logic_cycle=99,
            prediction_fields={"predicted_cycles": 105},
        )


def test_spearman_uses_average_ranks_and_is_undefined_for_constant_ranking():
    assert spearman_rank_correlation((1, 1, 2, 3), (1, 2, 2, 3)) == pytest.approx(5 / 6)
    assert spearman_rank_correlation((1, 2, 3), (3, 2, 1)) == pytest.approx(-1.0)
    assert spearman_rank_correlation((1, 1), (2, 3)) is None


def test_metrics_report_conservative_tie_aware_top1_regret_bias_and_apes():
    tied = (
        _row("a", 100, 10),
        _row("b", 200, 10),
        _row("c", 300, 20),
    )
    tied_metrics = compute_ranking_metrics(tied)

    assert tied_metrics.predicted_top_candidate_ids == ("a", "b")
    assert tied_metrics.measured_top_candidate_ids == ("a",)
    assert tied_metrics.top_1_regret_cycles == 100
    assert tied_metrics.top_1_regret_percent == 100

    errors = (
        _row("a", 100, 110),
        _row("b", 200, 160),
    )
    metrics = compute_ranking_metrics(errors)
    assert metrics.signed_bias_cycles == pytest.approx(-15.0)
    assert metrics.signed_relative_bias_percent == pytest.approx(-5.0)
    assert metrics.median_absolute_percentage_error == pytest.approx(15.0)
    assert metrics.p90_absolute_percentage_error == pytest.approx(19.0)
    assert metrics.max_absolute_percentage_error == pytest.approx(20.0)


def test_candidate_sets_render_deterministic_json_and_markdown():
    rows = (
        _row("tile-32", 120, 100, candidate_set="gemv"),
        _row("tile-16", 100, 110, candidate_set="gemv"),
        _row("tasklets-12", 80, 90, candidate_set="va"),
        _row("tasklets-8", 90, 100, candidate_set="va"),
    )
    forward = build_calibration_report(rows, report_metadata={"model": "upmem-v2"})
    backward = build_calibration_report(
        reversed(rows), report_metadata={"model": "upmem-v2"}
    )

    assert forward.manifest() == backward.manifest()
    assert forward.to_json() == backward.to_json()
    assert len(forward.report_fingerprint) == 64
    parsed = json.loads(forward.to_json())
    assert parsed["schema"] == REPORT_SCHEMA
    assert [item["candidate_set"] for item in parsed["candidate_sets"]] == [
        "gemv",
        "va",
    ]
    assert parsed["aggregate_metrics"]["candidate_count"] == 4
    markdown = forward.to_markdown()
    assert "# UPMEM calibration report" in markdown
    assert "| gemv | 2 |" in markdown
    assert "Run | Etc | DMA | Backpressure" in markdown


def test_reports_reject_missing_predictions_and_duplicate_candidates():
    missing_prediction = build_candidate_row(
        candidate_set="gemv",
        candidate_id="missing",
        log_text=_log(),
        result_metadata=_result(),
        provenance_metadata=_provenance(),
    )
    with pytest.raises(UPMEMRankingError, match="missing predicted_logic_cycle"):
        build_calibration_report((missing_prediction,))

    duplicate = _row("same", 100, 100)
    with pytest.raises(UPMEMRankingError, match="unique"):
        build_calibration_report((duplicate, duplicate))


def test_real_archived_atim_log_is_parseable_read_only_when_available():
    archive = Path.home() / "shared" / "tenon-artifacts" / "upmem"
    log_path = archive / "atim" / "runs" / "va" / "bin" / "log.txt"
    result_path = archive / "atim" / "runs" / "va" / "result.json"
    provenance_path = archive / "atim" / "programs" / "va" / "provenance.json"
    if not all(path.is_file() for path in (log_path, result_path, provenance_path)):
        pytest.skip("read-only archived UPMEM evidence is not installed")

    row = load_candidate_row(
        log_path=log_path,
        result_path=result_path,
        provenance_path=provenance_path,
        candidate_set="va",
        candidate_id="atim-archive",
        predicted_logic_cycle=131166,
    )

    assert row.measured_logic_cycle == 131166
    assert row.counters.component_sum == row.measured_logic_cycle
    assert row.result_metadata["status"] == "pass"
    assert row.artifact_paths["counter_log"] == str(log_path)
