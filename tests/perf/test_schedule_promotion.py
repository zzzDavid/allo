"""Focused tests for fail-closed schedule-promotion evidence."""

import json
from dataclasses import FrozenInstanceError, dataclass, replace

import pytest

from allo.pim.schedule_promotion import (
    CorrectnessEvidence,
    ExactCyclePolicy,
    MetricMeasurements,
    MetricPromotionCell,
    NoisyHardwarePolicy,
    ObjectiveMetricBridge,
    PromotionEvidence,
    ScheduleEvidence,
    SchedulePromotionGate,
    SemanticScope,
    WarmupEvidence,
)
from allo.pim.schedule_search import (
    DecisionDomain,
    OpaqueScheduleIncumbent,
    ScheduleCandidate,
    ScheduleObjectiveDomain,
    guarded_schedule_activation,
    grid_search,
)


SEARCH_DOMAIN = ScheduleObjectiveDomain(
    metric="device_ticks",
    target="structural_test_target",
    target_revision="revision-1",
    model_fingerprint=("cost-model", 1),
    fidelity="analytical",
    scope="whole_program",
    unit="device_ticks",
    direction="minimize",
)
HARDWARE_POLICY = NoisyHardwarePolicy(
    warmup_samples=1,
    min_samples=5,
    median_ratio=1.0,
    max_run_ratio=1.05,
)
WARMUP = WarmupEvidence(1, True)
CORRECTNESS = CorrectnessEvidence.exact_pass(("oracle", "bit-exact-u16"))
FULL_SCOPE = SemanticScope(
    domain=("semantic-contract", 1),
    guarantees=(("output", 0), ("epilogue", "beta")),
    complete=True,
)
PLATFORM = ("platform", "test-card", "sdk-1", "firmware-1")
_DEFAULT = object()


@dataclass(frozen=True)
class _TestMaterialization:
    plan: str
    promotion_materialization_fingerprint: tuple[str, str]
    promotion_platform_fingerprint: tuple[str, str, str, str] = PLATFORM


@dataclass(frozen=True)
class _NoPlatformMaterialization:
    plan: str
    promotion_materialization_fingerprint: tuple[str, str]


def _metric_domain(
    *,
    metric="device_ticks",
    target_revision="hardware-revision-1",
    target="structural_test_target",
    scope="whole_program",
    unit="device_ticks",
):
    return ScheduleObjectiveDomain(
        metric=metric,
        target=target,
        target_revision=target_revision,
        model_fingerprint=("measurement-campaign", 7),
        fidelity="hardware",
        scope=scope,
        unit=unit,
        direction="minimize",
    )


def _search_result():
    costs = {"incumbent": 100, "challenger": 80}
    return grid_search(
        (DecisionDomain("plan", tuple(costs)),),
        build=lambda decisions: decisions["plan"],
        materialize=lambda plan: _TestMaterialization(
            plan,
            ("materialization", plan),
        ),
        score=lambda artifact: costs[artifact.plan],
        objective=lambda cycles: cycles,
        objective_domain=SEARCH_DOMAIN,
        incumbent={"plan": "incumbent"},
    )


def _objective_bridge(
    measurement_domain,
    *,
    search_domain=SEARCH_DOMAIN,
    search_unit=None,
    measurement_unit=None,
):
    return ObjectiveMetricBridge(
        search_objective_domain=search_domain,
        measurement_objective_domain=measurement_domain,
        search_unit=search_domain.unit if search_unit is None else search_unit,
        measurement_unit=(
            measurement_domain.unit if measurement_unit is None else measurement_unit
        ),
        search_direction="minimize",
        measurement_direction="minimize",
        relation="identity_metric",
        provenance_fingerprint=("calibrated-objective-bridge", 1),
    )


def _candidate(plan, *, incumbent=False, index=0):
    return ScheduleCandidate(
        decisions={"plan": plan},
        payload=plan,
        materialized=_TestMaterialization(plan, ("materialization", plan)),
        score=100,
        objective=100,
        objective_domain=SEARCH_DOMAIN,
        is_incumbent=incumbent,
        enumeration_index=index,
    )


def _artifact_fingerprint(schedule):
    return schedule.materialized.promotion_materialization_fingerprint


def _promotion_evidence(
    recommended,
    incumbent,
    *,
    recommended_correctness=CORRECTNESS,
    incumbent_correctness=CORRECTNESS,
    recommended_scope=FULL_SCOPE,
    incumbent_scope=FULL_SCOPE,
    recommended_scored=_DEFAULT,
    recommended_emitted=_DEFAULT,
    incumbent_scored=_DEFAULT,
    incumbent_emitted=_DEFAULT,
    recommended_domain=_DEFAULT,
    incumbent_domain=_DEFAULT,
    recommended_samples=(99, 100, 99, 100, 99),
    incumbent_samples=(100, 100, 100, 100, 100),
    recommended_warmup=_DEFAULT,
    incumbent_warmup=_DEFAULT,
    recommended_platform=PLATFORM,
    incumbent_platform=PLATFORM,
    recommended_measurement_platform=_DEFAULT,
    incumbent_measurement_platform=_DEFAULT,
    objective_bridge=_DEFAULT,
    policy=HARDWARE_POLICY,
):
    recommended_artifact = _artifact_fingerprint(recommended)
    incumbent_artifact = _artifact_fingerprint(incumbent)
    if recommended_scored is _DEFAULT:
        recommended_scored = recommended_artifact
    if recommended_emitted is _DEFAULT:
        recommended_emitted = recommended_artifact
    if incumbent_scored is _DEFAULT:
        incumbent_scored = incumbent_artifact
    if incumbent_emitted is _DEFAULT:
        incumbent_emitted = incumbent_artifact
    if recommended_domain is _DEFAULT:
        recommended_domain = _metric_domain()
    if incumbent_domain is _DEFAULT:
        incumbent_domain = recommended_domain
    if recommended_warmup is _DEFAULT:
        recommended_warmup = None if isinstance(policy, ExactCyclePolicy) else WARMUP
    if incumbent_warmup is _DEFAULT:
        incumbent_warmup = None if isinstance(policy, ExactCyclePolicy) else WARMUP
    if recommended_measurement_platform is _DEFAULT:
        recommended_measurement_platform = recommended_platform
    if incumbent_measurement_platform is _DEFAULT:
        incumbent_measurement_platform = incumbent_platform
    if objective_bridge is _DEFAULT:
        bridge_domain = (
            recommended_domain
            if isinstance(recommended_domain, ScheduleObjectiveDomain)
            else incumbent_domain
        )
        objective_bridge = _objective_bridge(bridge_domain)

    return PromotionEvidence(
        recommended=ScheduleEvidence.from_schedule(
            recommended,
            correctness=recommended_correctness,
            semantic_scope=recommended_scope,
            scored_fingerprint=recommended_scored,
            emitted_fingerprint=recommended_emitted,
            platform_fingerprint=recommended_platform,
        ),
        incumbent=ScheduleEvidence.from_schedule(
            incumbent,
            correctness=incumbent_correctness,
            semantic_scope=incumbent_scope,
            scored_fingerprint=incumbent_scored,
            emitted_fingerprint=incumbent_emitted,
            platform_fingerprint=incumbent_platform,
        ),
        metrics=(
            MetricPromotionCell(
                recommended=MetricMeasurements(
                    recommended_domain,
                    recommended_samples,
                    recommended_warmup,
                    recommended_measurement_platform,
                ),
                incumbent=MetricMeasurements(
                    incumbent_domain,
                    incumbent_samples,
                    incumbent_warmup,
                    incumbent_measurement_platform,
                ),
                policy=policy,
                objective_bridge=objective_bridge,
            ),
        ),
    )


def test_accepted_challenger_activates_through_guarded_search_gate():
    result = _search_result()
    evidence = _promotion_evidence(result.best, result.best_incumbent)

    activation = guarded_schedule_activation(
        result,
        promotion_gate=SchedulePromotionGate(evidence),
    )

    assert activation.promoted is True
    assert activation.active is result.best
    assert activation.fallback_reason is None
    assert evidence.manifest() == evidence.manifest()
    json.dumps(evidence.manifest(), sort_keys=True)


@pytest.mark.parametrize(
    ("correctness", "reason"),
    [
        (
            CorrectnessEvidence(False, True, ("oracle", "bit-exact-u16")),
            "recommended_correctness_failed",
        ),
        (
            CorrectnessEvidence(True, False, ("oracle", "bit-exact-u16")),
            "recommended_correctness_not_exact",
        ),
        (
            CorrectnessEvidence.exact_pass(("oracle", "different")),
            "correctness_oracle_mismatch",
        ),
    ],
)
def test_correctness_mismatch_fails_closed(correctness, reason):
    result = _search_result()
    gate = SchedulePromotionGate(
        _promotion_evidence(
            result.best,
            result.best_incumbent,
            recommended_correctness=correctness,
        )
    )

    decision = gate(result.best, result.best_incumbent)

    assert decision.eligible is False
    assert decision.reason.startswith(reason)


def test_semantically_weaker_scope_fails_closed():
    result = _search_result()
    weaker = SemanticScope(
        domain=FULL_SCOPE.domain,
        guarantees=(("output", 0),),
        omissions=(("epilogue", "beta"),),
        complete=False,
    )
    gate = SchedulePromotionGate(
        _promotion_evidence(
            result.best,
            result.best_incumbent,
            recommended_scope=weaker,
        )
    )

    decision = gate(result.best, result.best_incumbent)

    assert decision.eligible is False
    assert decision.reason.startswith("semantic_scope_weaker")
    assert "complete coverage was lost" in decision.reason


def test_scored_and_emitted_artifact_mismatch_fails_closed():
    result = _search_result()
    gate = SchedulePromotionGate(
        _promotion_evidence(
            result.best,
            result.best_incumbent,
            recommended_emitted=("materialization", "different"),
        )
    )

    decision = gate(result.best, result.best_incumbent)

    assert decision.eligible is False
    assert decision.reason.startswith(
        "recommended_materialization_fingerprint_mismatch"
    )


def test_equal_but_stale_materialization_fingerprints_fail_closed():
    result = _search_result()
    stale = ("materialization", "different")
    gate = SchedulePromotionGate(
        _promotion_evidence(
            result.best,
            result.best_incumbent,
            recommended_scored=stale,
            recommended_emitted=stale,
        )
    )

    decision = gate(result.best, result.best_incumbent)

    assert decision.eligible is False
    assert decision.reason.startswith("recommended_materialization_fingerprint_stale")


def test_evidence_cannot_replay_across_equal_decisions_with_other_artifact():
    result = _search_result()
    gate = SchedulePromotionGate(
        _promotion_evidence(result.best, result.best_incumbent)
    )
    other_artifact = replace(
        result.best,
        materialized=_TestMaterialization(
            "challenger",
            ("materialization", "other-program"),
        ),
    )

    decision = gate(other_artifact, result.best_incumbent)

    assert decision.eligible is False
    assert decision.reason.startswith("recommended_materialization_fingerprint_stale")


def test_missing_current_materialization_fingerprint_fails_closed():
    result = _search_result()
    gate = SchedulePromotionGate(
        _promotion_evidence(result.best, result.best_incumbent)
    )
    missing = replace(result.best, materialized="unfingerprinted-artifact")

    decision = gate(missing, result.best_incumbent)

    assert decision.eligible is False
    assert decision.reason.startswith(
        "recommended_materialization_fingerprint_unavailable"
    )


def test_missing_or_stale_current_platform_fingerprint_fails_closed():
    result = _search_result()
    gate = SchedulePromotionGate(
        _promotion_evidence(result.best, result.best_incumbent)
    )
    missing = replace(
        result.best,
        materialized=_NoPlatformMaterialization(
            "challenger",
            ("materialization", "challenger"),
        ),
    )
    stale = replace(
        result.best,
        materialized=_TestMaterialization(
            "challenger",
            ("materialization", "challenger"),
            ("platform", "test-card", "sdk-2", "firmware-1"),
        ),
    )

    assert gate(missing, result.best_incumbent).reason.startswith(
        "recommended_platform_fingerprint_unavailable"
    )
    assert gate(stale, result.best_incumbent).reason.startswith(
        "recommended_platform_fingerprint_stale"
    )


def test_materialization_and_measurement_platform_mismatches_fail_closed():
    result = _search_result()
    other = ("platform", "test-card", "sdk-2", "firmware-1")
    materialization_gate = SchedulePromotionGate(
        _promotion_evidence(
            result.best,
            result.best_incumbent,
            recommended_platform=other,
        )
    )
    measurement_gate = SchedulePromotionGate(
        _promotion_evidence(
            result.best,
            result.best_incumbent,
            recommended_measurement_platform=other,
        )
    )

    assert materialization_gate(result.best, result.best_incumbent).reason.startswith(
        "materialization_platform_mismatch"
    )
    assert measurement_gate(result.best, result.best_incumbent).reason.startswith(
        "recommended_measurement_platform_mismatch"
    )


def test_stale_recommended_decision_identity_fails_closed():
    result = _search_result()
    gate = SchedulePromotionGate(
        _promotion_evidence(result.best, result.best_incumbent)
    )
    stale_recommendation = _candidate("new-challenger")

    decision = gate(stale_recommendation, result.best_incumbent)

    assert decision.eligible is False
    assert decision.reason.startswith("recommended_identity_mismatch")
    assert result.best.decisions != stale_recommendation.decisions


@pytest.mark.parametrize(
    ("recommended_domain", "reason"),
    [
        (
            _metric_domain(target_revision="hardware-revision-2"),
            "objective_domain_mismatch",
        ),
        ("device_ticks", "recommended_objective_domain_untyped"),
    ],
)
def test_each_metric_requires_an_exact_typed_objective_domain(
    recommended_domain,
    reason,
):
    result = _search_result()
    gate = SchedulePromotionGate(
        _promotion_evidence(
            result.best,
            result.best_incumbent,
            recommended_domain=recommended_domain,
            incumbent_domain=_metric_domain(),
        )
    )

    decision = gate(result.best, result.best_incumbent)

    assert decision.eligible is False
    assert decision.reason.startswith(reason)


@pytest.mark.parametrize(
    ("domain", "reason"),
    [
        (_metric_domain(target="different-target"), "measurement_target_mismatch"),
        (_metric_domain(scope="region"), "measurement_scope_mismatch"),
    ],
)
def test_measurements_bind_the_searched_target_and_scope(domain, reason):
    result = _search_result()
    gate = SchedulePromotionGate(
        _promotion_evidence(
            result.best,
            result.best_incumbent,
            recommended_domain=domain,
            incumbent_domain=domain,
        )
    )

    decision = gate(result.best, result.best_incumbent)

    assert decision.eligible is False
    assert decision.reason.startswith(reason)


@pytest.mark.parametrize(
    ("bridge", "reason"),
    [
        (
            _objective_bridge(
                _metric_domain(),
                search_domain=ScheduleObjectiveDomain(
                    metric="device_ticks",
                    target="structural_test_target",
                    target_revision="other-search-revision",
                    model_fingerprint=("other-cost-model", 1),
                    fidelity="analytical",
                    scope="whole_program",
                    unit="device_ticks",
                    direction="minimize",
                ),
            ),
            "search_objective_bridge_mismatch",
        ),
        (
            _objective_bridge(
                _metric_domain(target_revision="other-hardware-revision")
            ),
            "measurement_objective_bridge_mismatch",
        ),
    ],
)
def test_metric_evidence_requires_exact_objective_bridge(bridge, reason):
    result = _search_result()
    gate = SchedulePromotionGate(
        _promotion_evidence(
            result.best,
            result.best_incumbent,
            objective_bridge=bridge,
        )
    )

    decision = gate(result.best, result.best_incumbent)

    assert decision.eligible is False
    assert decision.reason.startswith(reason)


def test_unrelated_energy_metric_cannot_construct_objective_bridge():
    energy_domain = _metric_domain(metric="energy_joules", unit="joules")
    with pytest.raises(ValueError, match="identical.*metrics"):
        _objective_bridge(energy_domain)


def test_objective_bridge_units_must_match_both_typed_domains():
    with pytest.raises(ValueError, match="measurement unit"):
        _objective_bridge(_metric_domain(), measurement_unit="joules")
    cross_unit = _metric_domain(unit="joules")
    with pytest.raises(ValueError, match="identical units"):
        _objective_bridge(cross_unit)


@pytest.mark.parametrize(
    "policy",
    (NoisyHardwarePolicy(1, 5, 1.0, 1.05),),
)
def test_hardware_policy_accepts_only_bounded_noise(policy):
    assert policy.median_ratio <= 1.0
    assert policy.max_run_ratio <= 1.05


@pytest.mark.parametrize(
    "arguments",
    (
        (1, 4, 1.01, 1.05),
        (1, 5, 1.001, 1.05),
        (1, 5, 1.01, None),
        (1, 5, 1.01, 1.051),
    ),
)
def test_hardware_policy_rejects_unsafe_thresholds(arguments):
    with pytest.raises(ValueError):
        NoisyHardwarePolicy(*arguments)


@pytest.mark.parametrize(
    ("warmup", "samples", "reason"),
    [
        (None, (99, 99, 99, 99, 99), "recommended_warmup_missing"),
        (WARMUP, (), "recommended_samples_missing"),
        (WARMUP, (99, 99, 99, 99), "recommended_samples_insufficient"),
    ],
)
def test_hardware_policy_requires_explicit_warmup_and_samples(
    warmup,
    samples,
    reason,
):
    result = _search_result()
    gate = SchedulePromotionGate(
        _promotion_evidence(
            result.best,
            result.best_incumbent,
            recommended_warmup=warmup,
            recommended_samples=samples,
        )
    )

    decision = gate(result.best, result.best_incumbent)

    assert decision.eligible is False
    assert decision.reason.startswith(reason)


def test_hardware_median_regression_fails_its_cell():
    result = _search_result()
    gate = SchedulePromotionGate(
        _promotion_evidence(
            result.best,
            result.best_incumbent,
            recommended_samples=(102, 102, 102, 102, 102),
        )
    )

    decision = gate(result.best, result.best_incumbent)

    assert decision.eligible is False
    assert decision.reason.startswith("median_regression")
    assert "1.02x incumbent median 100" in decision.reason


def test_hardware_tail_regression_fails_after_median_passes():
    result = _search_result()
    gate = SchedulePromotionGate(
        _promotion_evidence(
            result.best,
            result.best_incumbent,
            recommended_samples=(100, 100, 100, 100, 106),
        )
    )

    decision = gate(result.best, result.best_incumbent)

    assert decision.eligible is False
    assert decision.reason.startswith("maximum_run_regression")
    assert "1.06x incumbent median 100" in decision.reason


def test_exact_cycle_policy_requires_integral_non_regressing_cycles():
    result = _search_result()
    policy = ExactCyclePolicy()
    accepted_gate = SchedulePromotionGate(
        _promotion_evidence(
            result.best,
            result.best_incumbent,
            recommended_samples=(99,),
            incumbent_samples=(100,),
            policy=policy,
        )
    )
    regressing_gate = SchedulePromotionGate(
        _promotion_evidence(
            result.best,
            result.best_incumbent,
            recommended_samples=(101,),
            incumbent_samples=(100,),
            policy=policy,
        )
    )
    non_integral_gate = SchedulePromotionGate(
        _promotion_evidence(
            result.best,
            result.best_incumbent,
            recommended_samples=(99.0,),
            incumbent_samples=(100,),
            policy=policy,
        )
    )

    assert accepted_gate(result.best, result.best_incumbent).eligible is True
    assert regressing_gate(result.best, result.best_incumbent).reason.startswith(
        "median_regression"
    )
    assert non_integral_gate(result.best, result.best_incumbent).reason.startswith(
        "recommended_exact_cycles_non_integral"
    )


def test_opaque_incumbent_identity_and_fingerprint_are_bound_exactly():
    result = grid_search(
        (DecisionDomain("plan", ("challenger",)),),
        build=lambda decisions: decisions["plan"],
        materialize=lambda plan: _TestMaterialization(
            plan,
            ("materialization", plan),
        ),
        score=lambda _artifact: 80,
        objective=lambda cycles: cycles,
        objective_domain=SEARCH_DOMAIN,
    )
    incumbent = OpaqueScheduleIncumbent(
        decisions={"legacy_route": ("physical", 3)},
        payload="legacy",
        materialized=_TestMaterialization(
            "legacy",
            ("materialization", "legacy"),
        ),
        fingerprint=("sha256", "a" * 64),
    )
    evidence = _promotion_evidence(result.best, incumbent)
    gate = SchedulePromotionGate(evidence)

    activation = guarded_schedule_activation(
        result,
        incumbent=incumbent,
        promotion_gate=gate,
    )

    assert activation.promoted is False
    assert activation.active is incumbent
    assert activation.fallback_reason.startswith(
        "opaque_incumbent_promotion_unsupported"
    )
    changed_fingerprint = replace(
        incumbent,
        fingerprint=("sha256", "b" * 64),
    )
    rejected = gate(result.best, changed_fingerprint)
    assert rejected.eligible is False
    assert rejected.reason.startswith("incumbent_identity_mismatch")

    changed_materialization = replace(
        incumbent,
        materialized=_TestMaterialization(
            "legacy",
            ("materialization", "mutated"),
        ),
    )
    rejected = gate(result.best, changed_materialization)
    assert rejected.eligible is False
    assert rejected.reason.startswith("incumbent_materialization_fingerprint_stale")


def test_evidence_records_are_frozen_and_measurements_must_be_positive_finite():
    with pytest.raises(FrozenInstanceError):
        HARDWARE_POLICY.min_samples = 6
    for invalid in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and positive"):
            MetricMeasurements(_metric_domain(), (invalid,), WARMUP, PLATFORM)
