"""Focused tests for structural calibration and held-out ranking reports."""

import json
import math
from dataclasses import FrozenInstanceError

import pytest

from allo.pim.calibration_report import (
    CalibrationDataError,
    CalibrationObjectiveDomain,
    CalibrationSample,
    ObjectiveDomainMismatch,
    PartitionLeakageError,
    build_calibration_report,
    build_explicit_calibration_report,
    build_frozen_model_leave_one_family_out_report,
    build_frozen_model_leave_one_shape_out_report,
    build_leave_one_family_out_report,
    build_leave_one_shape_out_report,
    explicit_partition,
    kendall_tau_b,
    leave_one_family_out_partitions,
    leave_one_shape_out_partitions,
    spearman_rank_correlation,
)


DOMAIN = CalibrationObjectiveDomain(
    metric="latency",
    unit="cycles",
    target="target-fingerprint",
    target_revision="revision-fingerprint",
    model_fingerprint="model-fingerprint",
    fidelity="measured-and-modeled",
    scope="complete-executable-graph",
)


def _sample(
    family,
    shape,
    candidate,
    measured,
    predicted,
    *,
    sample_fingerprint=None,
    domain=DOMAIN,
):
    return CalibrationSample(
        sample_fingerprint=sample_fingerprint or f"sample:{family}:{shape}:{candidate}",
        family_fingerprint=family,
        shape_fingerprint=shape,
        candidate_fingerprint=candidate,
        objective_domain=domain,
        measured=measured,
        predicted=predicted,
    )


def _problem(family, shape, measured, predicted):
    assert len(measured) == len(predicted)
    return tuple(
        _sample(
            family,
            shape,
            f"candidate-{index}",
            measured_value,
            predicted_value,
        )
        for index, (measured_value, predicted_value) in enumerate(
            zip(measured, predicted)
        )
    )


def test_rank_metrics_cover_perfect_reversed_and_tied_orderings():
    assert kendall_tau_b((1, 2, 3), (1, 2, 3)) == pytest.approx(1.0)
    assert spearman_rank_correlation((1, 2, 3), (1, 2, 3)) == pytest.approx(1.0)
    assert kendall_tau_b((1, 2, 3), (3, 2, 1)) == pytest.approx(-1.0)
    assert spearman_rank_correlation((1, 2, 3), (3, 2, 1)) == pytest.approx(-1.0)

    measured = (1, 1, 2, 3)
    predicted = (1, 2, 2, 3)
    assert kendall_tau_b(measured, predicted) == pytest.approx(0.8)
    assert spearman_rank_correlation(measured, predicted) == pytest.approx(5 / 6)


def test_per_holdout_and_aggregate_rank_metrics_and_top_1_regret():
    train = _problem("family-train", "shape-train", (4, 8), (5, 7))
    perfect = explicit_partition(
        train,
        _problem("family-perfect", "shape-perfect", (1, 2, 3), (1, 2, 3)),
        partition_fingerprint="perfect",
    )
    reversed_partition = explicit_partition(
        train,
        _problem("family-reversed", "shape-reversed", (1, 2, 3), (3, 2, 1)),
        partition_fingerprint="reversed",
    )

    report = build_calibration_report(
        (reversed_partition, perfect),
        partition_strategy="explicit",
    )
    by_partition = {
        holdout.partition_fingerprint: holdout.metrics for holdout in report.holdouts
    }

    assert by_partition["perfect"].kendall_tau_b == pytest.approx(1.0)
    assert by_partition["perfect"].spearman_rank_correlation == pytest.approx(1.0)
    assert by_partition["perfect"].top_1_regret == pytest.approx(0.0)
    assert by_partition["reversed"].kendall_tau_b == pytest.approx(-1.0)
    assert by_partition["reversed"].spearman_rank_correlation == pytest.approx(-1.0)
    assert by_partition["reversed"].top_1_regret == pytest.approx(2.0)
    assert report.aggregate_metrics.kendall_tau_b == pytest.approx(0.0)
    assert report.aggregate_metrics.spearman_rank_correlation == pytest.approx(0.0)
    assert report.aggregate_metrics.top_1_regret == pytest.approx(1.0)


def test_bias_and_absolute_percentage_errors_use_heldout_candidates():
    report = build_explicit_calibration_report(
        _problem("family-train", "shape-train", (3, 6), (3, 6)),
        _problem("family-holdout", "shape-holdout", (100, 200), (110, 160)),
    )
    metrics = report.aggregate_metrics

    assert metrics.signed_relative_bias == pytest.approx(-0.05)
    assert metrics.median_absolute_percentage_error == pytest.approx(15.0)
    assert metrics.p90_absolute_percentage_error == pytest.approx(19.0)
    assert metrics.top_1_regret == pytest.approx(0.0)


def test_constant_rankings_are_explicitly_undefined_not_nonfinite():
    report = build_explicit_calibration_report(
        _problem("family-train", "shape-train", (3, 6), (3, 6)),
        _problem("family-holdout", "shape-holdout", (1, 1), (2, 2)),
    )
    metrics = report.aggregate_metrics

    assert metrics.kendall_tau_b is None
    assert metrics.spearman_rank_correlation is None
    assert metrics.rank_correlation_problem_count == 0
    assert json.loads(report.to_json())["aggregate_metrics"]["kendall_tau_b"] is None


def test_partition_novel_shape_coverage_is_relative_to_reference_samples():
    train = _problem("family-train", "shape-shared", (5, 8), (6, 7))
    holdout = (
        *_problem("family-holdout", "shape-shared", (2, 4), (2, 5)),
        *_problem(
            "family-holdout",
            "shape-unseen",
            (1, 3, 6),
            (1, 4, 5),
        ),
    )

    report = build_explicit_calibration_report(train, holdout)
    holdout_report = report.holdouts[0]

    assert holdout_report.metrics.partition_novel_shape_problem_count == 1
    assert holdout_report.metrics.partition_novel_shape_coverage == pytest.approx(0.5)
    assert report.aggregate_metrics.partition_novel_shape_coverage == pytest.approx(0.5)
    assert {
        problem.key.shape_fingerprint: problem.partition_novel_shape
        for problem in holdout_report.problems
    } == {"shape-shared": False, "shape-unseen": True}


def test_family_stratified_evaluation_is_deterministic_and_exploratory():
    samples = (
        *_problem("structural-family-a", "shape-shared", (2, 4), (2, 5)),
        *_problem("structural-family-b", "shape-shared", (3, 6), (4, 6)),
        *_problem("structural-family-c", "shape-unseen", (1, 8), (1, 7)),
    )

    forward = build_frozen_model_leave_one_family_out_report(samples)
    backward = build_frozen_model_leave_one_family_out_report(reversed(samples))

    assert forward.manifest() == backward.manifest()
    assert forward.to_json() == backward.to_json()
    assert forward.report_fingerprint == backward.report_fingerprint
    assert len(forward.report_fingerprint) == 64
    assert forward.partition_strategy == "frozen_model_family_stratified_evaluation"
    assert len(forward.holdouts) == 3
    assert forward.aggregate_metrics.partition_novel_shape_coverage == pytest.approx(
        1 / 3
    )
    manifest = forward.manifest()
    assert manifest["report_kind"] == "exploratory_candidate_set_evaluation"
    assert manifest["identity_evidence"] == "caller_supplied_ids_unverified"
    assert manifest["candidate_coverage"] == "observed_candidate_sets_only"
    assert manifest["model_training_provenance"] == "not_provided"
    assert manifest["promotion_eligible"] is False

    for holdout in forward.holdouts:
        train_families = {
            sample.family_fingerprint for sample in holdout.partition.train
        }
        heldout_families = {
            sample.family_fingerprint for sample in holdout.partition.holdout
        }
        assert len(heldout_families) == 1
        assert train_families.isdisjoint(heldout_families)


def test_shape_stratified_evaluation_allows_same_family_without_shape_leakage():
    samples = (
        *_problem("dense-contraction", "shape-a", (2, 4), (2, 5)),
        *_problem("dense-contraction", "shape-b", (3, 6), (4, 6)),
        *_problem("dense-contraction", "shape-c", (1, 8), (1, 7)),
    )

    forward = build_frozen_model_leave_one_shape_out_report(samples)
    backward = build_frozen_model_leave_one_shape_out_report(reversed(samples))

    assert forward.manifest() == backward.manifest()
    assert forward.partition_strategy == "frozen_model_shape_stratified_evaluation"
    assert len(forward.holdouts) == 3
    assert forward.aggregate_metrics.partition_novel_shape_coverage == 1.0
    for holdout in forward.holdouts:
        assert holdout.partition.separation_axis == "shape"
        train_shapes = {sample.shape_fingerprint for sample in holdout.partition.train}
        heldout_shapes = {
            sample.shape_fingerprint for sample in holdout.partition.holdout
        }
        assert train_shapes.isdisjoint(heldout_shapes)
        assert {sample.family_fingerprint for sample in holdout.partition.train} == {
            "dense-contraction"
        }

    partitions = leave_one_shape_out_partitions(samples)
    assert tuple(partition.partition_fingerprint for partition in partitions) == (
        "shape:shape-a",
        "shape:shape-b",
        "shape:shape-c",
    )


def test_caller_relabeling_never_becomes_structural_evidence():
    first = build_explicit_calibration_report(
        _problem("family-reference", "shape-reference", (2, 4), (2, 4)),
        _problem("workload-name", "shape-tag", (1, 3), (1, 2)),
    )
    relabeled = build_explicit_calibration_report(
        _problem("renamed-reference", "renamed-shape", (2, 4), (2, 4)),
        _problem("arbitrary-tag", "opaque-label", (1, 3), (1, 2)),
    )

    assert first.aggregate_metrics == relabeled.aggregate_metrics
    assert first.report_fingerprint != relabeled.report_fingerprint
    for report in (first, relabeled):
        manifest = report.manifest()
        assert manifest["identity_evidence"] == "caller_supplied_ids_unverified"
        assert manifest["promotion_eligible"] is False
        sample = manifest["partition_evaluations"][0]["partition"][
            "partition_evaluation_samples"
        ][0]
        assert "caller_supplied_family_id" in sample
        assert "caller_supplied_shape_id" in sample
        assert "caller_supplied_candidate_id" in sample


def test_candidate_omission_can_change_metrics_but_not_evidence_strength():
    reference = _problem("reference", "reference-shape", (2, 4), (2, 4))
    complete_observed_set = _problem(
        "evaluation",
        "evaluation-shape",
        (1, 2, 100),
        (1, 100, 2),
    )
    full = build_explicit_calibration_report(reference, complete_observed_set)
    omitted = build_explicit_calibration_report(reference, complete_observed_set[:2])

    assert full.aggregate_metrics.top_1_regret == omitted.aggregate_metrics.top_1_regret
    assert (
        full.aggregate_metrics.kendall_tau_b != omitted.aggregate_metrics.kendall_tau_b
    )
    for report in (full, omitted):
        manifest = report.manifest()
        assert manifest["candidate_coverage"] == "observed_candidate_sets_only"
        assert manifest["evidence_strength"] == (
            "exploratory_unverified_identity_and_candidate_coverage"
        )
        assert manifest["promotion_eligible"] is False


def test_predicted_ties_report_all_tops_and_use_worst_case_regret():
    report = build_explicit_calibration_report(
        _problem("family-train", "shape-train", (3, 6), (3, 6)),
        _problem("family-holdout", "shape-holdout", (1, 3), (2, 2)),
    )
    problem = report.holdouts[0].problems[0]

    assert problem.predicted_top_candidate_fingerprints == (
        "candidate-0",
        "candidate-1",
    )
    assert problem.metrics.top_1_regret == pytest.approx(2.0)


def test_records_and_partitions_are_frozen_snapshots():
    train = list(_problem("family-train", "shape-train", (1, 2), (1, 2)))
    holdout = list(_problem("family-holdout", "shape-holdout", (2, 3), (2, 3)))
    partition = explicit_partition(train, holdout)
    train.clear()
    holdout.clear()

    assert len(partition.train) == 2
    assert len(partition.holdout) == 2
    with pytest.raises(FrozenInstanceError):
        partition.train[0].measured = 0.5
    with pytest.raises(FrozenInstanceError):
        partition.partition_fingerprint = "changed"


def test_sample_and_family_leakage_are_rejected():
    train = _problem("family-shared", "shape-train", (1, 2), (1, 2))
    same_family_holdout = _problem("family-shared", "shape-holdout", (3, 4), (3, 4))
    with pytest.raises(PartitionLeakageError, match="family leakage"):
        explicit_partition(train, same_family_holdout)

    same_shape_train = _problem("family-a", "shape-shared", (1, 2), (1, 2))
    same_shape_holdout = _problem("family-b", "shape-shared", (3, 4), (3, 4))
    with pytest.raises(PartitionLeakageError, match="shape leakage"):
        explicit_partition(
            same_shape_train,
            same_shape_holdout,
            separation_axis="shape",
        )

    leaked_sample = _sample(
        "family-holdout",
        "shape-holdout",
        "candidate-0",
        3,
        3,
        sample_fingerprint=train[0].sample_fingerprint,
    )
    holdout = (
        leaked_sample,
        _sample(
            "family-holdout",
            "shape-holdout",
            "candidate-1",
            4,
            4,
        ),
    )
    with pytest.raises(PartitionLeakageError, match="sample leakage"):
        explicit_partition(train, holdout)


def test_duplicate_candidates_and_single_candidate_problems_are_rejected():
    train = _problem("family-train", "shape-train", (1, 2), (1, 2))
    duplicate_candidates = (
        _sample(
            "family-holdout",
            "shape-holdout",
            "same-candidate",
            3,
            3,
            sample_fingerprint="sample-one",
        ),
        _sample(
            "family-holdout",
            "shape-holdout",
            "same-candidate",
            4,
            4,
            sample_fingerprint="sample-two",
        ),
    )
    with pytest.raises(CalibrationDataError, match="duplicate candidate identity"):
        explicit_partition(train, duplicate_candidates)

    one_candidate = (_sample("family-holdout", "shape-holdout", "only", 3, 3),)
    with pytest.raises(CalibrationDataError, match="at least two candidates"):
        explicit_partition(train, one_candidate)


def test_metric_and_objective_domain_mismatches_are_rejected():
    train = _problem("family-train", "shape-train", (1, 2), (1, 2))
    different_domain = CalibrationObjectiveDomain(
        metric="latency",
        unit="microseconds",
        target=DOMAIN.target,
        target_revision=DOMAIN.target_revision,
        model_fingerprint=DOMAIN.model_fingerprint,
        fidelity=DOMAIN.fidelity,
        scope=DOMAIN.scope,
        objective="maximize",
    )
    holdout = tuple(
        _sample(
            "family-holdout",
            "shape-holdout",
            f"candidate-{index}",
            index + 1,
            index + 1,
            domain=different_domain,
        )
        for index in range(2)
    )

    with pytest.raises(ObjectiveDomainMismatch):
        explicit_partition(train, holdout)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("measured", 0, "positive"),
        ("measured", -1, "positive"),
        ("measured", math.nan, "finite"),
        ("predicted", math.inf, "finite"),
        ("predicted", True, "real number"),
        ("predicted", "1", "real number"),
    ],
)
def test_malformed_or_nonfinite_values_fail_closed(field, value, message):
    arguments = {"measured": 1.0, "predicted": 1.0}
    arguments[field] = value

    with pytest.raises(CalibrationDataError, match=message):
        _sample(
            "family",
            "shape",
            "candidate",
            arguments["measured"],
            arguments["predicted"],
        )


def test_leave_one_family_out_requires_multiple_complete_families():
    one_family = _problem("family-only", "shape", (1, 2), (1, 2))
    with pytest.raises(CalibrationDataError, match="at least two"):
        leave_one_family_out_partitions(one_family)

    partitions = leave_one_family_out_partitions(
        (
            *one_family,
            *_problem("family-second", "shape", (2, 3), (2, 3)),
        )
    )
    assert [partition.partition_fingerprint for partition in partitions] == [
        "family:family-only",
        "family:family-second",
    ]
