# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic-search adapter gates for APU v1 vector plans."""

from dataclasses import dataclass, replace
import json
from types import SimpleNamespace

import allo
import pytest
from allo.ir.types import uint16
from allo.pim import apu_v1_vector_program as apu_program
from allo.pim.apu_v1_vector_cost import (
    estimate_apu_v1_plan,
    materialized_apu_v1_operation_inventory,
    rank_apu_v1_plans,
)
from allo.pim.apu_v1_vector_program import (
    APUV1PlanDecision,
    APUv1VectorCallable,
    search_apu_v1_vector_plans,
)
from allo.pim.apu_v1_vectorize import generate_apu_v1_vectorization_candidates
from allo.pim.costs import apu_v1_cost
from allo.pim.schedule_search import NoFeasibleSchedule
from allo.pim.targets import build_apu_v1_target


def representative_gemm(
    left: uint16[32, 16],
    right: uint16[16, 8],
    result: uint16[32, 8],
):
    for row, column in allo.grid(32, 8):
        for depth in allo.reduction(16):
            result[row, column] += left[row, depth] * right[depth, column]


def representative_gemv(
    left: uint16[17, 19],
    right: uint16[19, 1],
    result: uint16[17, 1],
):
    for row, column in allo.grid(17, 1):
        for depth in allo.reduction(19):
            result[row, column] += left[row, depth] * right[depth, column]


def renamed_gemm(
    matrix_lhs: uint16[32, 16],
    matrix_rhs: uint16[16, 8],
    destination: uint16[32, 8],
):
    for output_i, output_j in allo.grid(32, 8):
        for reduction_k in allo.reduction(16):
            destination[output_i, output_j] += (
                matrix_lhs[output_i, reduction_k] * matrix_rhs[reduction_k, output_j]
            )


@dataclass(frozen=True)
class _Plan:
    name: str


@dataclass(frozen=True)
class _Realization:
    plan: object


def _estimate(plan, cycles, *, fingerprint="apu-v1-test-model"):
    return SimpleNamespace(
        plan=plan,
        cycles=int(cycles),
        graph=SimpleNamespace(metadata={"target": "apu_v1"}),
        estimate=SimpleNamespace(model_fingerprint=fingerprint),
    )


def test_plan_decision_allowlists_physical_state_not_diagnostic_metadata():
    schedule = allo.customize(representative_gemm, enable_tensor=False)
    plan = generate_apu_v1_vectorization_candidates(schedule.module)[2].plan
    diagnostic_metadata = dict(plan.metadata)
    diagnostic_metadata.update(
        {
            "function": "renamed_diagnostic_function",
            "analysis": "renamed-diagnostic-analysis",
            "debug_label": "arbitrary non-physical note",
        }
    )
    renamed = replace(
        plan,
        name="renamed_physical_plan",
        metadata=diagnostic_metadata,
    )
    layout_changed = replace(
        plan,
        value_layouts=(
            replace(plan.value_layouts[0], storage="l4"),
            *plan.value_layouts[1:],
        ),
    )
    transfer = plan.transfers[0]
    route_step = transfer.route[0]
    route_changed = replace(
        plan,
        transfers=(
            replace(
                transfer,
                route=(
                    replace(
                        route_step,
                        executed_at=(
                            *route_step.executed_at[:-1],
                            replace(route_step.executed_at[-1], tile_extent=2),
                        ),
                    ),
                ),
            ),
            *plan.transfers[1:],
        ),
    )
    operation_changed = replace(
        plan,
        operations=(
            replace(plan.operations[0], count=plan.operations[0].count + 1),
            *plan.operations[1:],
        ),
    )

    decision = apu_program._apu_v1_plan_decision(plan)
    assert apu_program._apu_v1_plan_decision(renamed) == decision
    assert apu_program._apu_v1_plan_decision(layout_changed) != decision
    assert apu_program._apu_v1_plan_decision(route_changed) != decision
    assert apu_program._apu_v1_plan_decision(operation_changed) != decision


def test_estimate_lookup_uses_physical_identity_not_plan_name(monkeypatch):
    schedule = allo.customize(representative_gemm, enable_tensor=False)
    candidate = generate_apu_v1_vectorization_candidates(schedule.module)[2]
    plan = candidate.plan
    renamed = replace(
        plan,
        name="renamed_physical_plan",
        metadata={**plan.metadata, "debug_label": "renamed diagnostic"},
    )
    same_name_but_changed = replace(
        plan,
        operations=(
            replace(plan.operations[0], count=plan.operations[0].count + 1),
            *plan.operations[1:],
        ),
    )
    ranked = _estimate(plan, 7)
    fallback = _estimate(same_name_but_changed, 11)
    monkeypatch.setattr(
        apu_program,
        "rank_apu_v1_plans",
        lambda *_args: (ranked,),
    )
    monkeypatch.setattr(
        apu_program,
        "estimate_apu_v1_plan",
        lambda selected, *_args: (
            fallback
            if selected is same_name_but_changed
            else pytest.fail("physical alias missed its retained estimate")
        ),
    )
    monkeypatch.setattr(
        apu_program,
        "_realize",
        lambda _analysis, selected: (_Realization(selected), None),
    )

    compiled = APUv1VectorCallable(
        representative_gemm,
        SimpleNamespace(name="apu_v1"),
        SimpleNamespace(module=object()),
        (candidate,),
        cost=object(),
        layout=renamed,
        backend="functional",
    )

    assert compiled.selected_plan is renamed
    assert compiled.selected_estimate is ranked
    assert compiled.estimate(renamed) is ranked.estimate
    assert compiled.estimate(same_name_but_changed) is fallback.estimate


def test_unrealizable_plan_is_rejected_before_scoring(monkeypatch):
    plans = (_Plan("unrealizable"), _Plan("realizable"))
    events = []

    def realize(_analysis, plan):
        events.append(("realize", plan))
        if plan is plans[0]:
            return None, "cannot lower plan"
        return _Realization(plan), None

    def estimate(realization, _target, _cost):
        plan = realization.plan
        events.append(("score", plan))
        return _estimate(plan, 7)

    monkeypatch.setattr(apu_program, "_realize", realize)
    monkeypatch.setattr(apu_program, "estimate_apu_v1_realization", estimate)
    result = search_apu_v1_vector_plans(object(), plans, object(), object())

    assert result.best.payload is plans[1]
    assert tuple(result.best.decisions) == ("plan",)
    assert isinstance(result.best.decisions["plan"], APUV1PlanDecision)
    assert result.stats.materialize_attempts == 2
    assert result.stats.score_attempts == 1
    assert [(item.stage, item.reason) for item in result.rejections] == [
        ("materialize", "cannot lower plan")
    ]
    assert ("score", plans[0]) not in events
    assert events.index(("realize", plans[1])) < events.index(("score", plans[1]))


def test_callable_retains_only_estimates_for_realized_candidates(monkeypatch):
    plans = (_Plan("unrealizable"), _Plan("realizable"))
    analysis = object()
    candidates = tuple(
        SimpleNamespace(name=plan.name, plan=plan, analysis=analysis) for plan in plans
    )
    scored = []

    def realize(_analysis, plan):
        if plan is plans[0]:
            return None, "cannot lower plan"
        return _Realization(plan), None

    def estimate(realization, _target, _cost):
        plan = realization.plan
        scored.append(plan)
        return _estimate(plan, 7)

    monkeypatch.setattr(apu_program, "_realize", realize)
    monkeypatch.setattr(apu_program, "estimate_apu_v1_realization", estimate)
    monkeypatch.setattr(
        apu_program,
        "rank_apu_v1_plans",
        lambda feasible, _target, _cost: tuple(
            _estimate(plan, index + 1) for index, plan in enumerate(feasible)
        ),
    )

    def workload(value):
        return value

    compiled = APUv1VectorCallable(
        workload,
        SimpleNamespace(name="apu_v1"),
        SimpleNamespace(module=object()),
        candidates,
        cost=object(),
        backend="functional",
    )

    assert scored == [plans[1]]
    assert [estimate.plan for estimate in compiled.candidate_estimates] == [plans[1]]
    assert compiled.schedule_search_result.rejections[0].stage == "materialize"


def test_objective_domain_is_comparable_and_ties_prefer_incumbent(monkeypatch):
    plans = (_Plan("first"), _Plan("incumbent"), _Plan("last"))
    monkeypatch.setattr(
        apu_program,
        "_realize",
        lambda _analysis, plan: (_Realization(plan), None),
    )
    monkeypatch.setattr(
        apu_program,
        "estimate_apu_v1_realization",
        lambda realization, _target, _cost: _estimate(realization.plan, 11),
    )

    first = search_apu_v1_vector_plans(
        SimpleNamespace(function="first_workload"),
        plans,
        object(),
        object(),
        incumbent_plan=plans[1],
    )
    second = search_apu_v1_vector_plans(
        SimpleNamespace(function="different_workload"),
        tuple(reversed(plans)),
        object(),
        object(),
    )

    assert first.best.payload is plans[1]
    assert first.best.is_incumbent
    assert [candidate.payload for candidate in first.ranked] == [
        plans[1],
        plans[0],
        plans[2],
    ]
    assert first.best.objective == 11
    assert isinstance(first.best.objective, int)
    assert first.best.objective_domain.metric == "cycles"
    assert first.best.objective_domain.target == "apu_v1"
    assert first.best.objective_domain.model_fingerprint == "apu-v1-test-model"
    assert first.best.objective_domain.fidelity == "analytical"
    assert first.best.objective_domain.scope == "region"
    assert first.best.objective_domain.target_revision.startswith("fingerprinted:")
    assert second.best.objective_domain == first.best.objective_domain
    assert "workload" not in repr(first.best.objective_domain)


def test_incumbent_and_materialization_identity_fail_closed(monkeypatch):
    plan = _Plan("candidate")
    copied_plan = _Plan("candidate")

    with pytest.raises(ValueError, match="searched plan objects"):
        search_apu_v1_vector_plans(
            object(),
            (plan,),
            object(),
            object(),
            incumbent_plan=copied_plan,
        )

    monkeypatch.setattr(
        apu_program,
        "_realize",
        lambda _analysis, _plan: (_Realization(copied_plan), None),
    )
    monkeypatch.setattr(
        apu_program,
        "estimate_apu_v1_realization",
        lambda *_args: pytest.fail("mismatched realization was scored"),
    )

    with pytest.raises(NoFeasibleSchedule):
        search_apu_v1_vector_plans(object(), (plan,), object(), object())


@pytest.mark.parametrize("workload", [representative_gemm, representative_gemv])
def test_search_selection_matches_legacy_rank_then_realize(workload):
    schedule = allo.customize(workload, enable_tensor=False)
    candidates = generate_apu_v1_vectorization_candidates(schedule.module)
    plans = tuple(candidate.plan for candidate in candidates)
    target = build_apu_v1_target()
    ranked = rank_apu_v1_plans(plans, target, apu_v1_cost)
    legacy_selected = next(
        estimate.plan
        for estimate in ranked
        if apu_program._realize(candidates[0].analysis, estimate.plan)[0] is not None
    )

    compiled = APUv1VectorCallable(
        workload,
        target,
        schedule,
        candidates,
        cost=apu_v1_cost,
        backend="virtual",
    )

    assert compiled.selected_plan is legacy_selected
    assert compiled.schedule_search_result is not None
    assert compiled.schedule_activation.recommended is (
        compiled.schedule_search_result.best
    )
    assert compiled.schedule_activation.active is compiled.schedule_search_result.best
    assert compiled.fallback_reason is None
    assert compiled.realization is compiled.schedule_search_result.best.materialized
    assert len(compiled.realization.promotion_materialization_fingerprint) == 64
    assert (
        compiled.realization.promotion_materialization_fingerprint
        == compiled.realization.promotion_materialization_fingerprint
    )
    assert compiled.selected_estimate is compiled.schedule_search_result.best.score
    assert compiled.candidate_estimates == tuple(
        candidate.score for candidate in compiled.schedule_search_result.ranked
    )
    assert len(compiled.candidate_estimates) == (
        compiled.schedule_search_result.stats.feasible_candidates
    )
    assert all(json.dumps(plan.manifest()) for plan in compiled.plans)


def test_scored_materialization_inventory_is_name_independent():
    target = build_apu_v1_target()
    first_schedule = allo.customize(representative_gemm, enable_tensor=False)
    second_schedule = allo.customize(renamed_gemm, enable_tensor=False)
    first_candidates = generate_apu_v1_vectorization_candidates(first_schedule.module)
    second_candidates = generate_apu_v1_vectorization_candidates(second_schedule.module)

    first = search_apu_v1_vector_plans(
        first_candidates[0].analysis,
        tuple(candidate.plan for candidate in first_candidates),
        target,
        apu_v1_cost,
    )
    second = search_apu_v1_vector_plans(
        second_candidates[0].analysis,
        tuple(candidate.plan for candidate in second_candidates),
        target,
        apu_v1_cost,
    )

    first_by_decision = {
        candidate.decisions["plan"]: candidate for candidate in first.ranked
    }
    second_by_decision = {
        candidate.decisions["plan"]: candidate for candidate in second.ranked
    }
    assert first_by_decision.keys() == second_by_decision.keys()
    assert len(first_by_decision) == first.stats.feasible_candidates
    corrected_inventory = []
    for decision, first_candidate in first_by_decision.items():
        second_candidate = second_by_decision[decision]
        first_inventory = materialized_apu_v1_operation_inventory(
            first_candidate.materialized
        )
        second_inventory = materialized_apu_v1_operation_inventory(
            second_candidate.materialized
        )
        assert first_candidate.score.operation_inventory == first_inventory
        assert second_candidate.score.operation_inventory == second_inventory
        assert first_candidate.score.graph.metadata["operation_inventory"] == tuple(
            operation.canonical_manifest for operation in first_inventory
        )
        assert tuple(
            activity.metadata["operation"]
            for activity in first_candidate.score.graph.activities
            if "operation" in activity.metadata
        ) == tuple(operation.opcode for operation in first_inventory)
        assert tuple(
            activity.metadata["operation"]
            for activity in second_candidate.score.graph.activities
            if "operation" in activity.metadata
        ) == tuple(operation.opcode for operation in second_inventory)
        assert first_inventory == second_inventory
        assert first_candidate.objective == second_candidate.objective
        if len(first_inventory) > len(first_candidate.payload.operations):
            corrected_inventory.append((first_candidate, first_inventory))

    assert corrected_inventory
    corrected_candidate, inventory = corrected_inventory[0]
    assert tuple(operation.opcode for operation in inventory) == (
        "MUL_U16",
        "RESET_16",
        "GROUP_REDUCE_U16",
        "ADD_U16",
    )
    assert (
        corrected_candidate.score.cycles
        > estimate_apu_v1_plan(
            corrected_candidate.payload,
            target,
            apu_v1_cost,
        ).cycles
    )


def test_explicit_layout_pins_plan_and_bypasses_search(monkeypatch):
    plans = (_Plan("first"), _Plan("pinned"))
    analysis = object()
    candidates = tuple(
        SimpleNamespace(name=plan.name, plan=plan, analysis=analysis) for plan in plans
    )
    estimates = tuple(_estimate(plan, index + 1) for index, plan in enumerate(plans))
    monkeypatch.setattr(
        apu_program,
        "search_apu_v1_vector_plans",
        lambda *_args, **_kwargs: pytest.fail("explicit layout invoked search"),
    )
    monkeypatch.setattr(
        apu_program,
        "rank_apu_v1_plans",
        lambda *_args, **_kwargs: estimates,
    )
    monkeypatch.setattr(
        apu_program,
        "_realize",
        lambda _analysis, plan: (_Realization(plan), None),
    )

    def workload(value):
        return value

    compiled = APUv1VectorCallable(
        workload,
        SimpleNamespace(name="apu_v1"),
        SimpleNamespace(module=object()),
        candidates,
        cost=object(),
        layout="pinned",
        backend="functional",
    )

    assert compiled.schedule_search_result is None
    assert compiled.schedule_activation is None
    assert compiled.selected_plan is plans[1]
    assert compiled.realization.plan is plans[1]
    assert compiled.candidate_estimates == estimates


def test_corrected_shadow_ranking_retains_preinventory_incumbent(monkeypatch):
    plans = (_Plan("legacy"), _Plan("corrected_challenger"))
    analysis = object()
    candidates = tuple(
        SimpleNamespace(name=plan.name, plan=plan, analysis=analysis) for plan in plans
    )

    monkeypatch.setattr(
        apu_program,
        "_realize",
        lambda _analysis, plan: (_Realization(plan), None),
    )
    monkeypatch.setattr(
        apu_program,
        "rank_apu_v1_plans",
        lambda _plans, _target, _cost: (
            _estimate(plans[0], 2),
            _estimate(plans[1], 3),
        ),
    )
    monkeypatch.setattr(
        apu_program,
        "estimate_apu_v1_realization",
        lambda realization, _target, _cost: _estimate(
            realization.plan,
            1 if realization.plan is plans[1] else 10,
        ),
    )

    def workload(value):
        return value

    compiled = APUv1VectorCallable(
        workload,
        SimpleNamespace(name="apu_v1"),
        SimpleNamespace(module=object()),
        candidates,
        cost=object(),
        backend="functional",
    )

    assert compiled.schedule_search_result.best.payload is plans[1]
    assert compiled.schedule_search_result.best_incumbent.payload is plans[0]
    assert compiled.schedule_activation.recommended.payload is plans[1]
    assert compiled.schedule_activation.active.payload is plans[0]
    assert compiled.selected_plan is plans[0]
    assert compiled.fallback_reason.startswith("shadow_only:")
