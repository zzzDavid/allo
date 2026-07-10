"""Focused tests for structural candidate-sweep ranking reports."""

import importlib.util
import json
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "report_candidate_ranking_calibration.py"
SPEC = importlib.util.spec_from_file_location("candidate_ranking_calibration", SCRIPT)
REPORTER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = REPORTER
SPEC.loader.exec_module(REPORTER)


def _platform_contract(backend, fingerprint="a" * 64):
    return {
        "schema": "tenon-promotion-platform-v1",
        "target": "apu_v1" if backend == "apu_v1" else "apu_v2",
        "hardware_family": "gemini-i" if backend == "apu_v1" else "gemini-ii",
        "fingerprint": fingerprint,
    }


def _apu_v1_report(shape, measured, predicted, suffix):
    candidates = []
    for index, (measured_value, predicted_value) in enumerate(zip(measured, predicted)):
        candidates.append(
            {
                "candidate_identity_fingerprint": f"candidate-{suffix}-{index}",
                "measurement_fingerprint": f"measurement-{suffix}-{index}",
                "model_prediction": {"composed_cycles": predicted_value},
                "correlated_cycle_samples": [
                    {"composed_cycles": measured_value - 1},
                    {"composed_cycles": measured_value + 1},
                ],
            }
        )
    return {
        "schema": REPORTER.APUV1_MEASURER.SCHEMA,
        "logical_shape": {"M": shape, "K": 8, "N": 8},
        "provenance": {
            "target_fingerprint": "target-v1",
            "cost_fingerprint": "cost-v1",
            "platform_contract": _platform_contract("apu_v1"),
        },
        "domain": {
            "derived_row_tiles": [shape],
            "selected_row_tiles": [shape],
            "selected_physical_plan_fingerprints": None,
            "row_tile_domain_complete": True,
            "physical_plan_domain_complete": True,
            "full_domain_candidate_count": len(candidates),
            "selected_candidate_count": len(candidates),
            "materialized_candidate_count": len(candidates),
            "rejected_candidate_count": 0,
            "measured_candidate_count": len(candidates),
            "candidate_domain_complete": True,
        },
        "candidates": candidates,
        "rejections": [],
    }


def _apu_g2_report(shape, measured, predicted, suffix):
    candidates = []
    for index, (measured_value, predicted_value) in enumerate(zip(measured, predicted)):
        candidates.append(
            {
                "candidate_index": index,
                "identity": {"candidate_fingerprint": f"candidate-{suffix}-{index}"},
                "schedule": {"batch_columns": index + 1},
                "model_prediction": {"wall_us": predicted_value},
                "samples": [
                    {"host_us": {"wall_us": measured_value - 1}},
                    {"host_us": {"wall_us": measured_value + 1}},
                ],
            }
        )
    return {
        "schema_version": REPORTER.APUG2_MEASURER.SCHEMA_VERSION,
        "record_kind": REPORTER.APUG2_MEASURER.RECORD_KIND,
        "structural_case": {"M": shape, "K": 8, "N": 8, "alpha": 1, "beta": 1},
        "provenance": {
            "source_fingerprint": "source-v1",
            "runtime_driver_sha256": "runtime-v1",
            "model_fingerprint": "model-v1",
            "platform_contract": _platform_contract("apu_g2"),
        },
        "candidate_domain": {
            "full_batch_columns": list(REPORTER.APUG2_MEASURER.DEFAULT_BATCH_COLUMNS),
            "selected_batch_columns": list(
                REPORTER.APUG2_MEASURER.DEFAULT_BATCH_COLUMNS
            ),
            "full_candidate_count": len(REPORTER.APUG2_MEASURER.DEFAULT_BATCH_COLUMNS),
            "selected_candidate_count": len(
                REPORTER.APUG2_MEASURER.DEFAULT_BATCH_COLUMNS
            ),
            "candidate_domain_complete": True,
        },
        "measurement_fingerprint": f"report-{suffix}",
        "candidates": candidates,
        "rejections": [],
    }


@pytest.mark.parametrize(
    "factory,validator",
    (
        (_apu_v1_report, "APUV1_MEASURER"),
        (_apu_g2_report, "APUG2_MEASURER"),
    ),
)
def test_leave_one_shape_ranking_is_deterministic(factory, validator, monkeypatch):
    monkeypatch.setattr(
        getattr(REPORTER, validator),
        "validate_measurement_report",
        lambda _report: None,
    )
    candidate_count = (
        2
        if factory is _apu_v1_report
        else len(REPORTER.APUG2_MEASURER.DEFAULT_BATCH_COLUMNS)
    )
    measured = tuple(range(10, 10 + candidate_count))
    first = factory(16, measured, measured, "a")
    second = factory(32, measured, tuple(reversed(measured)), "b")

    forward = REPORTER.build_candidate_ranking_report((first, second))
    backward = REPORTER.build_candidate_ranking_report((second, first))

    assert forward == backward
    assert forward["report_kind"] == "candidate_ranking_evaluation"
    assert forward["partition_strategy"] == "frozen_model_shape_stratified_evaluation"
    assert forward["promotion_eligible"] is False
    assert forward["candidate_coverage"] == (
        "adapter_verified_complete_candidate_domains"
    )
    assert forward["evidence_strength"] == (
        "complete_domains_with_consistent_platform_contract"
    )
    assert forward["model_training_provenance"] == "not_provided"
    assert forward["platform_contract"] == _platform_contract(
        "apu_v1" if factory is _apu_v1_report else "apu_g2"
    )
    metrics = forward["evaluation"]["aggregate_metrics"]
    assert metrics["observed_ranking_problem_count"] == 2
    assert metrics["observed_candidate_count"] == 2 * candidate_count
    assert metrics["kendall_tau_b"] == pytest.approx(0.0)
    assert metrics["spearman_rank_correlation"] == pytest.approx(0.0)
    assert metrics["partition_novel_shape_coverage"] == 1.0
    assert len(forward["report_fingerprint"]) == 64
    rendered = REPORTER.report_to_json(forward)
    assert json.loads(rendered) == forward.manifest()
    assert "extrapolat" not in rendered.lower()
    assert "heldout" not in rendered.lower()


def test_mixed_backends_and_single_shape_fail_closed(monkeypatch):
    monkeypatch.setattr(
        REPORTER.APUV1_MEASURER, "validate_measurement_report", lambda _report: None
    )
    monkeypatch.setattr(
        REPORTER.APUG2_MEASURER, "validate_measurement_report", lambda _report: None
    )
    apu_v1 = _apu_v1_report(16, (10, 20), (10, 20), "a")
    values = tuple(range(10, 41))
    apu_g2 = _apu_g2_report(16, values, values, "b")

    with pytest.raises(REPORTER.CandidateRankingReportError, match="mix"):
        REPORTER.build_candidate_ranking_report((apu_v1, apu_g2))
    with pytest.raises(
        REPORTER.CandidateRankingReportError, match="two measurement reports"
    ):
        REPORTER.build_candidate_ranking_report((apu_v1,))


def test_incomplete_or_rejected_domains_cannot_be_cherry_picked(monkeypatch):
    monkeypatch.setattr(
        REPORTER.APUV1_MEASURER, "validate_measurement_report", lambda _report: None
    )
    first = _apu_v1_report(16, (10, 20), (10, 20), "a")
    second = _apu_v1_report(32, (10, 20), (10, 20), "b")

    first["domain"]["candidate_domain_complete"] = False
    with pytest.raises(REPORTER.CandidateRankingReportError, match="complete"):
        REPORTER.build_candidate_ranking_report((first, second))

    first = _apu_v1_report(16, (10, 20), (10, 20), "a")
    first["rejections"].append({"reason": "hidden"})
    with pytest.raises(REPORTER.CandidateRankingReportError, match="zero"):
        REPORTER.build_candidate_ranking_report((first, second))

    monkeypatch.setattr(
        REPORTER.APUG2_MEASURER, "validate_measurement_report", lambda _report: None
    )
    values = tuple(range(10, 41))
    first_g2 = _apu_g2_report(16, values, values, "c")
    second_g2 = _apu_g2_report(32, values, values, "d")
    first_g2["candidates"].pop()
    with pytest.raises(REPORTER.CandidateRankingReportError, match="exact"):
        REPORTER.build_candidate_ranking_report((first_g2, second_g2))


def test_unverifiable_or_mixed_platforms_are_refused(monkeypatch):
    monkeypatch.setattr(
        REPORTER.APUV1_MEASURER, "validate_measurement_report", lambda _report: None
    )
    first = _apu_v1_report(16, (10, 20), (10, 20), "a")
    second = _apu_v1_report(32, (10, 20), (10, 20), "b")

    first["provenance"]["platform_contract"] = None
    with pytest.raises(REPORTER.CandidateRankingReportError, match="unverifiable"):
        REPORTER.build_candidate_ranking_report((first, second))

    first = _apu_v1_report(16, (10, 20), (10, 20), "a")
    second["provenance"]["platform_contract"] = _platform_contract("apu_v1", "b" * 64)
    with pytest.raises(REPORTER.CandidateRankingReportError, match="mixed"):
        REPORTER.build_candidate_ranking_report((first, second))


def test_serialization_rejects_forged_mappings_and_tampered_instances(monkeypatch):
    monkeypatch.setattr(
        REPORTER.APUV1_MEASURER, "validate_measurement_report", lambda _report: None
    )
    first = _apu_v1_report(16, (10, 20), (10, 20), "a")
    second = _apu_v1_report(32, (10, 20), (20, 10), "b")
    report = REPORTER.build_candidate_ranking_report((first, second))

    with pytest.raises(FrozenInstanceError):
        report._components = None
    with pytest.raises(AttributeError):
        object.__setattr__(report, "to_json", lambda: "forged")

    forged = report.manifest()
    forged["promotion_eligible"] = True
    forged["evidence_strength"] = "trust-me"
    forged["evaluation"]["aggregate_metrics"]["observed_set_top_1_regret"] = 0
    forged_body = dict(forged)
    forged_body.pop("report_fingerprint")
    forged["report_fingerprint"] = REPORTER._fingerprint(
        "candidate-ranking-evaluation-v2",
        forged_body,
    )
    with pytest.raises(
        REPORTER.CandidateRankingReportError,
        match="immutable builder-produced",
    ):
        REPORTER.report_to_json(forged)

    original = report._components
    tampered_fingerprints = tuple(
        sorted(("0" * 64, *original.source_report_fingerprints[1:]))
    )
    tampered = REPORTER._CandidateRankingComponents(
        backend=original.backend,
        platform_contract_json=original.platform_contract_json,
        source_report_fingerprints=tampered_fingerprints,
        evaluation=original.evaluation,
    )
    object.__setattr__(report, "_components", tampered)
    with pytest.raises(
        REPORTER.CandidateRankingReportError,
        match="frozen source records",
    ):
        REPORTER.report_to_json(report)
