"""Focused gates for the workload-neutral finite-domain search core."""

from dataclasses import FrozenInstanceError, dataclass

import pytest

from allo.pim.schedule_search import (
    AssignmentProblem,
    DecisionDomain,
    InfeasibleIncumbent,
    InfeasibleSchedule,
    InvalidDecisionValue,
    InvalidObjectiveValue,
    InvalidProducedAssignment,
    LegalityConstraint,
    MissingScheduleIncumbent,
    NoFeasibleSchedule,
    NoFeasibleScheduleInPrefix,
    NonFiniteObjective,
    OpaqueScheduleIncumbent,
    ObjectiveDomainMismatch,
    PromotionDecision,
    ProducedAssignments,
    ScheduleCandidate,
    ScheduleObjectiveDomain,
    guarded_schedule_activation,
    grid_search,
)


@dataclass(frozen=True)
class Estimate:
    cycles: float
    label: str


@dataclass(frozen=True)
class FrozenChoice:
    shape: tuple[int, ...]
    modes: frozenset[str]


@dataclass(frozen=True, order=True)
class FrozenObjective:
    cycles: float
    tie_break: tuple[int, ...]


def test_cartesian_and_dependent_domains_preserve_enumeration_order():
    result = grid_search(
        (
            DecisionDomain("family", ("small", "large")),
            DecisionDomain(
                "tile",
                lambda decisions: (1, 2) if decisions["family"] == "small" else (2, 4),
            ),
            DecisionDomain("unroll", (1, 2)),
        ),
        build=lambda decisions: tuple(decisions.items()),
        materialize=lambda payload: {"plan": payload},
        score=lambda realized: Estimate(1, str(realized["plan"])),
        objective=lambda estimate: estimate.cycles,
        objective_domain="cycles@test-model",
    )

    assert [tuple(candidate.decisions.items()) for candidate in result.ranked] == [
        (("family", "small"), ("tile", 1), ("unroll", 1)),
        (("family", "small"), ("tile", 1), ("unroll", 2)),
        (("family", "small"), ("tile", 2), ("unroll", 1)),
        (("family", "small"), ("tile", 2), ("unroll", 2)),
        (("family", "large"), ("tile", 2), ("unroll", 1)),
        (("family", "large"), ("tile", 2), ("unroll", 2)),
        (("family", "large"), ("tile", 4), ("unroll", 1)),
        (("family", "large"), ("tile", 4), ("unroll", 2)),
    ]
    assert result.stats.complete_assignments_considered == 8
    assert result.stats.termination == "exhausted"


def test_external_assignment_producer_uses_the_same_lifecycle_and_optimum():
    events = []

    def producer(problem):
        assert isinstance(problem, AssignmentProblem)
        assert tuple(domain.name for domain in problem.domains) == ("tile",)
        return ProducedAssignments(
            ({"tile": 4}, {"tile": 2}, {"tile": 1}),
            termination="exhausted",
        )

    result = grid_search(
        (DecisionDomain("tile", (1, 2, 4)),),
        build=lambda decisions: events.append(("build", decisions["tile"]))
        or decisions["tile"],
        materialize=lambda payload: events.append(("materialize", payload))
        or f"artifact:{payload}",
        score=lambda artifact: events.append(("score", artifact))
        or Estimate(int(artifact.removeprefix("artifact:")), artifact),
        objective=lambda estimate: estimate.cycles,
        objective_domain="cycles@test-model",
        incumbent={"tile": 2},
        assignment_producer=producer,
    )

    assert result.best.payload == 1
    assert result.best_incumbent.payload == 2
    assert result.stats.complete_assignments_considered == 2
    assert events[0:3] == [
        ("build", 2),
        ("materialize", 2),
        ("score", "artifact:2"),
    ]
    for tile in (4, 1):
        assert events.index(("materialize", tile)) < events.index(
            ("score", f"artifact:{tile}")
        )


def test_external_assignment_producer_is_validated_and_reports_truncation():
    common = dict(
        domains=(DecisionDomain("choice", (0, 1)),),
        build=lambda decisions: decisions["choice"],
        materialize=lambda payload: payload,
        score=lambda payload: Estimate(payload + 1, str(payload)),
        objective=lambda estimate: estimate.cycles,
        objective_domain="cycles@test-model",
    )
    truncated = grid_search(
        **common,
        assignment_producer=lambda _problem: ProducedAssignments(
            ({"choice": 0},), termination="truncated"
        ),
    )
    assert truncated.stats.truncated

    with pytest.raises(InvalidProducedAssignment, match="outside dependent domain"):
        grid_search(
            **common,
            assignment_producer=lambda _problem: ProducedAssignments(({"choice": 7},)),
        )
    with pytest.raises(InvalidProducedAssignment, match="duplicates"):
        grid_search(
            **common,
            assignment_producer=lambda _problem: ProducedAssignments(
                ({"choice": 0}, {"choice": 0})
            ),
        )


def test_producer_exhaustion_is_proven_and_coverage_uses_the_finite_problem():
    domains = (
        DecisionDomain("family", ("small", "large")),
        DecisionDomain(
            "tile",
            lambda decisions: ((1, 2) if decisions["family"] == "small" else (2, 4)),
        ),
    )
    common = dict(
        domains=domains,
        constraints=(
            LegalityConstraint(
                "supported_tile",
                lambda decisions: decisions.get("tile") != 4,
                scope="partial",
            ),
        ),
        build=lambda decisions: tuple(decisions.items()),
        materialize=lambda payload: payload,
        score=lambda payload: Estimate(1, str(payload)),
        objective=lambda estimate: estimate.cycles,
        objective_domain="cycles@test-model",
    )
    subset = ({"family": "small", "tile": 1},)

    with pytest.raises(InvalidProducedAssignment, match="claimed exhausted.*omitted 2"):
        grid_search(
            **common,
            assignment_producer=lambda _problem: ProducedAssignments(
                subset,
                termination="exhausted",
            ),
        )

    truncated = grid_search(
        **common,
        assignment_producer=lambda _problem: ProducedAssignments(
            subset,
            termination="truncated",
        ),
    )
    assert truncated.stats.complete_assignments_covered == 1
    assert truncated.stats.complete_assignments_total == 3
    assert truncated.stats.candidate_coverage == pytest.approx(1 / 3)
    assert truncated.report()["candidate_coverage"] == pytest.approx(1 / 3)


def test_named_partial_and_full_constraints_prune_at_the_earliest_scope():
    dependent_domain_calls = []
    built = []

    def choices(decisions):
        dependent_domain_calls.append(decisions["outer"])
        return (0, 1)

    result = grid_search(
        (
            DecisionDomain("outer", (1, 2)),
            DecisionDomain("inner", choices),
        ),
        constraints=(
            LegalityConstraint(
                "outer_capacity",
                lambda decisions: decisions.get("outer") != 2,
                scope="partial",
            ),
            LegalityConstraint(
                "even_inner",
                lambda decisions: decisions["inner"] == 0,
            ),
        ),
        build=lambda decisions: built.append(tuple(decisions.items()))
        or dict(decisions),
        materialize=lambda payload: payload,
        score=lambda _realized: Estimate(1, "legal"),
        objective=lambda estimate: estimate.cycles,
        objective_domain="cycles@test-model",
    )

    assert dependent_domain_calls == [1]
    assert built == [(("outer", 1), ("inner", 0))]
    assert result.stats.pruned_branches == 1
    assert [(rejection.stage, rejection.name) for rejection in result.rejections] == [
        ("full_constraint", "even_inner"),
        ("partial_constraint", "outer_capacity"),
    ]


def test_expected_stage_rejections_never_score_unrealized_plans():
    events = []

    def build(decisions):
        name = decisions["case"]
        events.append(f"build:{name}")
        if name == "build":
            raise InfeasibleSchedule("cannot build")
        return name

    def materialize(payload):
        events.append(f"materialize:{payload}")
        if payload == "materialize":
            raise InfeasibleSchedule("cannot realize")
        return f"realized:{payload}"

    def score(realized):
        name = realized.removeprefix("realized:")
        events.append(f"score:{name}")
        if name == "score":
            raise InfeasibleSchedule("cannot score")
        return Estimate(1, name)

    result = grid_search(
        (DecisionDomain("case", ("build", "materialize", "score", "ok")),),
        build=build,
        materialize=materialize,
        score=score,
        objective=lambda estimate: estimate.cycles,
        objective_domain="cycles@test-model",
    )

    assert [candidate.payload for candidate in result.ranked] == ["ok"]
    assert [rejection.stage for rejection in result.rejections] == [
        "build",
        "materialize",
        "score",
    ]
    assert "score:materialize" not in events
    assert events.index("materialize:ok") < events.index("score:ok")


def test_object_scores_use_a_separate_objective_with_stable_ties():
    cycles = {"first": 5, "second": 5, "fast": 2}
    result = grid_search(
        (DecisionDomain("plan", tuple(cycles)),),
        build=lambda decisions: decisions["plan"],
        materialize=lambda payload: f"realized:{payload}",
        score=lambda realized: Estimate(
            cycles[realized.removeprefix("realized:")], realized
        ),
        objective=lambda estimate: estimate.cycles,
        objective_domain="cycles@test-model",
    )

    assert [candidate.payload for candidate in result.ranked] == [
        "fast",
        "first",
        "second",
    ]
    assert all(isinstance(candidate.score, Estimate) for candidate in result.ranked)
    assert [candidate.enumeration_index for candidate in result.ranked] == [2, 0, 1]


def test_equal_challengers_use_structural_order_not_producer_order():
    def search(order):
        return grid_search(
            (DecisionDomain("choice", (3, 1, 2)),),
            build=lambda decisions: decisions["choice"],
            materialize=lambda payload: payload,
            score=lambda payload: Estimate(5, str(payload)),
            objective=lambda estimate: estimate.cycles,
            objective_domain="cycles@test-model",
            assignment_producer=lambda _problem: ProducedAssignments(
                tuple({"choice": choice} for choice in order),
                termination="exhausted",
            ),
        )

    forward = search((3, 1, 2))
    reversed_result = search((2, 1, 3))

    assert [candidate.payload for candidate in forward.ranked] == [1, 2, 3]
    assert [candidate.payload for candidate in reversed_result.ranked] == [1, 2, 3]


def test_explicit_incumbent_is_an_ordinary_immutable_candidate():
    costs = {"baseline": 10, "model": 7, "slower": 12}
    result = grid_search(
        (DecisionDomain("plan", tuple(costs)),),
        build=lambda decisions: decisions["plan"],
        materialize=lambda payload: payload,
        score=lambda payload: Estimate(costs[payload], payload),
        objective=lambda estimate: estimate.cycles,
        objective_domain="cycles@test-model",
        incumbent={"plan": "baseline"},
    )

    incumbent = result.best_incumbent
    assert result.best.payload == "model"
    assert isinstance(incumbent, ScheduleCandidate)
    assert incumbent in result.ranked
    assert incumbent.payload == "baseline"
    assert result.best.objective <= incumbent.objective
    assert sum(candidate.is_incumbent for candidate in result.ranked) == 1
    with pytest.raises(TypeError):
        incumbent.decisions["plan"] = "changed"
    with pytest.raises(FrozenInstanceError):
        incumbent.payload = "changed"


def test_equal_objective_prefers_incumbent_not_first_in_domain():
    result = grid_search(
        (DecisionDomain("plan", ("first", "baseline", "last")),),
        build=lambda decisions: decisions["plan"],
        materialize=lambda payload: payload,
        score=lambda payload: Estimate(5, payload),
        objective=lambda estimate: estimate.cycles,
        objective_domain="cycles@test-model",
        incumbent={"plan": "baseline"},
    )

    assert [candidate.payload for candidate in result.ranked] == [
        "baseline",
        "first",
        "last",
    ]
    assert result.best is result.best_incumbent


def test_guarded_activation_keeps_incumbent_and_records_fallback_reason():
    costs = {"baseline": 10, "challenger": 7}
    result = grid_search(
        (DecisionDomain("plan", tuple(costs)),),
        build=lambda decisions: decisions["plan"],
        materialize=lambda payload: f"artifact:{payload}",
        score=lambda artifact: Estimate(
            costs[artifact.removeprefix("artifact:")], artifact
        ),
        objective=lambda estimate: estimate.cycles,
        objective_domain="cycles@test-model",
        incumbent={"plan": "baseline"},
    )

    activation = guarded_schedule_activation(result)

    assert activation.recommended.payload == "challenger"
    assert activation.active is result.best_incumbent
    assert activation.active_materialized == "artifact:baseline"
    assert activation.fallback_reason == (
        "shadow_only: promotion evidence was not requested"
    )
    assert activation.promoted is False


def test_promotion_gate_can_accept_or_reject_a_challenger():
    costs = {"baseline": 10, "challenger": 7}
    result = grid_search(
        (DecisionDomain("plan", tuple(costs)),),
        build=lambda decisions: decisions["plan"],
        materialize=lambda payload: payload,
        score=lambda payload: Estimate(costs[payload], payload),
        objective=lambda estimate: estimate.cycles,
        objective_domain="cycles@test-model",
        incumbent={"plan": "baseline"},
    )

    rejected = guarded_schedule_activation(
        result,
        promotion_gate=lambda recommended, incumbent: PromotionDecision(
            False,
            f"hardware evidence missing for {recommended.payload} over "
            f"{incumbent.payload}",
        ),
    )
    accepted = guarded_schedule_activation(
        result,
        promotion_gate=lambda _recommended, _incumbent: PromotionDecision(True),
    )

    assert rejected.active is result.best_incumbent
    assert rejected.fallback_reason.startswith("hardware evidence missing")
    assert accepted.active is result.best
    assert accepted.fallback_reason is None
    assert accepted.promoted is True


def test_opaque_incumbent_can_fall_back_outside_generated_domain():
    result = grid_search(
        (DecisionDomain("plan", ("generated",)),),
        build=lambda decisions: decisions["plan"],
        materialize=lambda payload: f"artifact:{payload}",
        score=lambda payload: Estimate(1, payload),
        objective=lambda estimate: estimate.cycles,
        objective_domain="cycles@test-model",
    )
    incumbent = OpaqueScheduleIncumbent(
        decisions={"legacy_route": "pinned"},
        payload="legacy",
        materialized="artifact:legacy",
        fingerprint=("sha256", "0" * 64),
    )

    activation = guarded_schedule_activation(result, incumbent=incumbent)

    assert activation.recommended is result.best
    assert activation.incumbent is incumbent
    assert activation.active is incumbent
    assert activation.active_materialized == "artifact:legacy"
    assert activation.fallback_reason.startswith("shadow_only")


def test_guarded_activation_fails_closed_without_any_incumbent():
    result = grid_search(
        (DecisionDomain("plan", ("only",)),),
        build=lambda decisions: decisions["plan"],
        materialize=lambda payload: payload,
        score=lambda payload: Estimate(1, payload),
        objective=lambda estimate: estimate.cycles,
        objective_domain="cycles@test-model",
    )

    with pytest.raises(MissingScheduleIncumbent):
        guarded_schedule_activation(result)


def test_incumbent_is_evaluated_first_outside_cap_and_duplicate_is_skipped():
    built = []
    result = grid_search(
        (DecisionDomain("plan", ("first", "second", "baseline")),),
        build=lambda decisions: built.append(decisions["plan"]) or decisions["plan"],
        materialize=lambda payload: payload,
        score=lambda payload: Estimate(1, payload),
        objective=lambda estimate: estimate.cycles,
        objective_domain="cycles@test-model",
        incumbent={"plan": "baseline"},
        max_complete_assignments=1,
    )

    assert built == ["baseline", "first"]
    assert result.stats.complete_assignments_considered == 1
    assert result.stats.incumbent_evaluated
    assert result.stats.termination == "truncated"
    assert [candidate.payload for candidate in result.ranked] == [
        "baseline",
        "first",
    ]


def test_incumbent_requires_exact_names_and_dependent_domain_membership():
    domains = (
        DecisionDomain("outer", (1,)),
        DecisionDomain("inner", lambda decisions: (decisions["outer"],)),
    )

    with pytest.raises(InfeasibleIncumbent) as missing:
        grid_search(
            domains,
            build=lambda decisions: decisions,
            materialize=lambda payload: payload,
            score=lambda payload: Estimate(1, str(payload)),
            objective=lambda estimate: estimate.cycles,
            objective_domain="cycles@test-model",
            incumbent={"outer": 1},
        )
    assert missing.value.diagnostics[-1].stage == "incumbent"

    with pytest.raises(InfeasibleIncumbent) as absent:
        grid_search(
            domains,
            build=lambda decisions: decisions,
            materialize=lambda payload: payload,
            score=lambda payload: Estimate(1, str(payload)),
            objective=lambda estimate: estimate.cycles,
            objective_domain="cycles@test-model",
            incumbent={"outer": 1, "inner": 2},
        )
    assert absent.value.diagnostics[-1].stage == "domain"
    assert absent.value.diagnostics[-1].name == "inner"


def test_illegal_or_unrealizable_incumbent_is_a_dedicated_hard_failure():
    with pytest.raises(InfeasibleIncumbent) as illegal:
        grid_search(
            (DecisionDomain("plan", ("baseline", "other")),),
            constraints=(
                LegalityConstraint(
                    "not_baseline",
                    lambda decisions: decisions["plan"] != "baseline",
                ),
            ),
            build=lambda decisions: decisions["plan"],
            materialize=lambda payload: payload,
            score=lambda payload: Estimate(1, payload),
            objective=lambda estimate: estimate.cycles,
            objective_domain="cycles@test-model",
            incumbent={"plan": "baseline"},
        )
    assert illegal.value.diagnostics[-1].stage == "full_constraint"

    with pytest.raises(InfeasibleIncumbent) as unrealizable:
        grid_search(
            (DecisionDomain("plan", ("other", "baseline")),),
            build=lambda decisions: decisions["plan"],
            materialize=lambda payload: (_ for _ in ()).throw(
                InfeasibleSchedule("cannot realize incumbent")
            ),
            score=lambda payload: Estimate(1, payload),
            objective=lambda estimate: estimate.cycles,
            objective_domain="cycles@test-model",
            incumbent={"plan": "baseline"},
            max_complete_assignments=1,
        )
    assert unrealizable.value.diagnostics[-1].stage == "materialize"
    assert "cannot realize incumbent" in unrealizable.value.diagnostics[-1].reason


def test_mixed_objective_domains_are_rejected_before_ranking():
    with pytest.raises(ObjectiveDomainMismatch) as caught:
        grid_search(
            (DecisionDomain("plan", ("first", "second")),),
            build=lambda decisions: decisions["plan"],
            materialize=lambda payload: payload,
            score=lambda payload: Estimate(1, payload),
            objective=lambda estimate: estimate.cycles,
            objective_domain=lambda estimate: f"cycles@{estimate.label}",
        )

    assert caught.value.expected == "cycles@first"
    assert caught.value.actual == "cycles@second"
    assert caught.value.decisions["plan"] == "second"


def test_typed_objective_domain_carries_explicit_comparability_provenance():
    domain = ScheduleObjectiveDomain.fingerprinted_target(
        metric="cycles",
        target="test_target",
        model_fingerprint=("model", 3),
        fidelity="analytical",
        scope="whole_program",
        unit="cycles",
        direction="minimize",
    )
    result = grid_search(
        (DecisionDomain("plan", ("only",)),),
        build=lambda decisions: decisions["plan"],
        materialize=lambda payload: payload,
        score=lambda payload: Estimate(1, payload),
        objective=lambda estimate: estimate.cycles,
        objective_domain=domain,
    )

    assert result.best.objective_domain is domain
    report = result.report()
    assert report["termination"] == "exhausted"
    assert report["elapsed_seconds"] >= 0
    assert report["candidate_coverage"] == 1.0
    assert report["candidate_coverage_unit"] == (
        "fraction_of_constraint_legal_complete_assignments"
    )
    assert report["complete_assignments_covered"] == 1
    assert report["complete_assignments_total"] == 1
    assert report["recommended_decisions"] == {"plan": "only"}
    assert report["objective_domain"] == domain.manifest()
    assert report["objective_unit"] == "cycles"
    assert report["objective_direction"] == "minimize"
    assert domain.target_revision == "fingerprinted:('model', 3)"
    assert domain.manifest() == {
        "metric": "cycles",
        "target": "test_target",
        "target_revision": "fingerprinted:('model', 3)",
        "model_fingerprint": ("model", 3),
        "fidelity": "analytical",
        "scope": "whole_program",
        "unit": "cycles",
        "direction": "minimize",
    }

    with pytest.raises(InvalidDecisionValue):
        ScheduleObjectiveDomain(
            "cycles",
            "test_target",
            "revision",
            ["mutable"],
            "analytical",
            "region",
            "cycles",
            "minimize",
        )
    with pytest.raises(ValueError, match="direction"):
        ScheduleObjectiveDomain(
            "cycles",
            "test_target",
            "revision",
            ("model", 1),
            "analytical",
            "region",
            "cycles",
            "maximize",
        )


def test_exhaustive_no_feasible_schedule_is_a_hard_failure_with_diagnostics():
    def reject(decisions):
        raise InfeasibleSchedule(f"rejected {decisions['choice']}")

    with pytest.raises(NoFeasibleSchedule, match="no feasible schedule") as caught:
        grid_search(
            (DecisionDomain("choice", ("a", "b")),),
            build=reject,
            materialize=lambda payload: payload,
            score=lambda payload: payload,
            objective=lambda value: value,
            objective_domain="cycles@test-model",
        )

    assert type(caught.value) is NoFeasibleSchedule
    assert caught.value.stats.termination == "exhausted"
    assert caught.value.stats.complete_assignments_considered == 2
    assert [rejection.reason for rejection in caught.value.rejections] == [
        "rejected a",
        "rejected b",
    ]


def test_exact_complete_assignment_bound_is_exhausted_but_larger_space_truncates():
    def search(values):
        return grid_search(
            (DecisionDomain("choice", values),),
            build=lambda decisions: decisions["choice"],
            materialize=lambda payload: payload,
            score=lambda payload: Estimate(1, str(payload)),
            objective=lambda estimate: estimate.cycles,
            objective_domain="cycles@test-model",
            max_complete_assignments=2,
        )

    exact = search((0, 1))
    truncated = search((0, 1, 2))

    assert exact.stats.complete_assignments_considered == 2
    assert exact.stats.termination == "exhausted"
    assert exact.stats.exhaustive
    assert truncated.stats.complete_assignments_considered == 2
    assert truncated.stats.termination == "truncated"
    assert truncated.stats.truncated


def test_truncated_no_feasible_prefix_has_a_distinct_error():
    with pytest.raises(NoFeasibleScheduleInPrefix) as caught:
        grid_search(
            (DecisionDomain("choice", (0, 1)),),
            build=lambda decisions: (_ for _ in ()).throw(
                InfeasibleSchedule(f"reject {decisions['choice']}")
            ),
            materialize=lambda payload: payload,
            score=lambda payload: payload,
            objective=lambda value: value,
            objective_domain="cycles@test-model",
            max_complete_assignments=1,
        )

    assert caught.value.stats.termination == "truncated"
    assert caught.value.stats.complete_assignments_considered == 1
    assert len(caught.value.rejections) == 1


def test_frozen_recursive_decisions_are_retained_without_mutable_aliases():
    source = [FrozenChoice((2, 4), frozenset({"a", "b"}))]
    domain = DecisionDomain("choice", source)
    source.clear()

    result = grid_search(
        (domain,),
        build=lambda decisions: decisions["choice"],
        materialize=lambda payload: payload,
        score=lambda payload: Estimate(1, str(payload)),
        objective=lambda estimate: estimate.cycles,
        objective_domain="cycles@test-model",
    )

    retained = result.best.decisions["choice"]
    assert retained == FrozenChoice((2, 4), frozenset({"a", "b"}))
    with pytest.raises(FrozenInstanceError):
        retained.shape = (8,)


def test_nested_mutable_and_mutable_hashable_decisions_are_rejected():
    shared_list = []
    with pytest.raises(InvalidDecisionValue, match="mutable"):
        DecisionDomain("bad", ((shared_list, shared_list),))

    class MutableHashable:
        __hash__ = object.__hash__

    with pytest.raises(InvalidDecisionValue, match="identity-based"):
        DecisionDomain("bad", (MutableHashable(),))

    class MutableTuple(tuple):
        pass

    tuple_subclass = MutableTuple((1, 2))
    tuple_subclass.alias = shared_list
    with pytest.raises(InvalidDecisionValue, match="identity-based"):
        DecisionDomain("bad", (tuple_subclass,))

    dependent = DecisionDomain("bad", lambda _decisions: ([1, 2],))
    with pytest.raises(InvalidDecisionValue, match="mutable"):
        grid_search(
            (dependent,),
            build=lambda decisions: decisions,
            materialize=lambda payload: payload,
            score=lambda payload: Estimate(1, str(payload)),
            objective=lambda estimate: estimate.cycles,
            objective_domain="cycles@test-model",
        )


def test_duplicate_values_domain_names_and_constraint_names_are_rejected():
    with pytest.raises(ValueError, match="duplicate values"):
        DecisionDomain("choice", (1, 1))

    with pytest.raises(ValueError, match="domain names must be unique"):
        grid_search(
            (DecisionDomain("same", (1,)), DecisionDomain("same", (2,))),
            build=lambda decisions: decisions,
            materialize=lambda payload: payload,
            score=lambda payload: Estimate(1, str(payload)),
            objective=lambda estimate: estimate.cycles,
            objective_domain="cycles@test-model",
        )

    with pytest.raises(ValueError, match="constraint names must be unique"):
        grid_search(
            (DecisionDomain("choice", (1,)),),
            constraints=(
                LegalityConstraint("same", lambda _decisions: True),
                LegalityConstraint(
                    "same",
                    lambda _decisions: True,
                    scope="partial",
                ),
            ),
            build=lambda decisions: decisions,
            materialize=lambda payload: payload,
            score=lambda payload: Estimate(1, str(payload)),
            objective=lambda estimate: estimate.cycles,
            objective_domain="cycles@test-model",
        )


def test_dependent_domain_duplicate_values_are_rejected():
    with pytest.raises(ValueError, match="duplicate values"):
        grid_search(
            (DecisionDomain("choice", lambda _decisions: (1, 1)),),
            build=lambda decisions: decisions,
            materialize=lambda payload: payload,
            score=lambda payload: Estimate(1, str(payload)),
            objective=lambda estimate: estimate.cycles,
            objective_domain="cycles@test-model",
        )


def test_non_finite_real_objective_is_rejected():
    with pytest.raises(NonFiniteObjective, match="finite"):
        grid_search(
            (DecisionDomain("choice", (1,)),),
            build=lambda decisions: decisions,
            materialize=lambda payload: payload,
            score=lambda payload: Estimate(1, str(payload)),
            objective=lambda _estimate: float("nan"),
            objective_domain="cycles@test-model",
        )


def test_mutable_arbitrary_and_nested_non_finite_objectives_are_rejected():
    class Comparable:
        def __lt__(self, _other):
            return False

    common = dict(
        domains=(DecisionDomain("choice", (1,)),),
        build=lambda decisions: decisions,
        materialize=lambda payload: payload,
        score=lambda _payload: Estimate(1, "only"),
        objective_domain="cycles@test-model",
    )

    for objective_value in ([1.0], Comparable()):
        with pytest.raises(InvalidObjectiveValue, match="unsupported mutable"):
            grid_search(
                **common,
                objective=lambda _estimate, value=objective_value: value,
            )

    with pytest.raises(NonFiniteObjective, match=r"objective\[1\]\[0\].*finite"):
        grid_search(
            **common,
            objective=lambda _estimate: (1.0, (float("nan"),)),
        )


def test_objective_is_snapshotted_and_revalidated_before_reporting():
    shared = FrozenObjective(7.0, (2,))
    result = grid_search(
        (DecisionDomain("choice", (1,)),),
        build=lambda decisions: decisions,
        materialize=lambda payload: payload,
        score=lambda _payload: Estimate(1, "only"),
        objective=lambda _estimate: shared,
        objective_domain="cycles@test-model",
    )

    object.__setattr__(shared, "cycles", 1.0)
    assert result.best.objective == FrozenObjective(7.0, (2,))
    assert result.report()["recommended_objective"] == FrozenObjective(7.0, (2,))

    object.__setattr__(result.best.objective, "cycles", float("nan"))
    with pytest.raises(NonFiniteObjective, match="finite"):
        result.report()
