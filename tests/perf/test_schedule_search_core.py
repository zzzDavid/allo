"""Focused gates for the workload-neutral finite-domain search core."""

from dataclasses import dataclass

import pytest

from allo.pim.schedule_search import (
    DecisionDomain,
    InfeasibleSchedule,
    NoFeasibleSchedule,
    grid_search,
)


@dataclass(frozen=True)
class Estimate:
    cycles: float
    label: str


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
