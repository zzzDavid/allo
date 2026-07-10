"""Focused gates for standalone APUv1 outer profile-row search."""

import hashlib
from dataclasses import replace
from types import SimpleNamespace

import allo
import pytest
from allo.ir.types import uint16
from allo.pim import apu_v1_profile_search as profile_search
from allo.pim.apu_v1_profile_search import (
    APUV1ProfileCompositionScore,
    UnsupportedAPUV1ProfileSearchError,
    derive_apu_v1_profile_plan_decisions,
    derive_apu_v1_profile_row_tiles,
    search_apu_v1_profile_rows,
)
from allo.pim.apu_v1_vector_program import APUv1VectorCallable
from allo.pim.apu_v1_vectorize import (
    ContractionAnalysis,
    LogicalAxis,
    ValueAccess,
)
from allo.pim.costs import apu_v1_cost
from allo.pim.schedule_search import (
    InfeasibleSchedule,
    MissingScheduleIncumbent,
    OpaqueScheduleIncumbent,
)
from allo.pim.targets import build_apu_v1_target


def _analysis(
    rows=256,
    columns=1024,
    reduction=2200,
    *,
    names=("profile_kernel", "row", "column", "depth", "left", "right", "output"),
):
    function, row, column, depth, left, right, output = names
    axes = (
        LogicalAxis(row, rows, 0, rows, 1, False, "%row"),
        LogicalAxis(column, columns, 0, columns, 1, False, "%column"),
        LogicalAxis(depth, reduction, 0, reduction, 1, True, "%depth"),
    )
    return ContractionAnalysis(
        function,
        axes,
        (row, column),
        (row, column),
        depth,
        ValueAccess(left, (row, depth), "read", "ui16", (rows, reduction)),
        ValueAccess(right, (depth, column), "read", "ui16", (reduction, columns)),
        ValueAccess(output, (row, column), "read", "ui16", (rows, columns)),
        ValueAccess(output, (row, column), "write", "ui16", (rows, columns)),
        "arith.muli",
        "arith.addi",
        "ui16",
    )


def _decision(analysis, target, row_tile, index=2):
    return derive_apu_v1_profile_plan_decisions(analysis, target, row_tile=row_tile)[
        index
    ]


def _fake_search(monkeypatch, analysis, target, incumbent_row, incumbent_plan):
    events = []

    def materialize(problem, decision):
        events.append(("materialize", decision.row_tile, decision.plan))
        return SimpleNamespace(
            decision=decision,
            row_waves=(problem.logical_shape.rows + decision.row_tile - 1)
            // decision.row_tile,
            column_repetitions=problem.column_repetitions,
            objective_domain=problem.objective_domain,
        )

    def score(problem, materialized):
        events.append(
            ("score", materialized.decision.row_tile, materialized.decision.plan)
        )
        shard = 1000 - materialized.decision.row_tile
        return APUV1ProfileCompositionScore(
            shard,
            materialized.row_waves,
            materialized.column_repetitions,
            shard * materialized.row_waves * materialized.column_repetitions,
            problem.objective_domain,
        )

    monkeypatch.setattr(profile_search, "_materialize", materialize)
    monkeypatch.setattr(profile_search, "_score", score)
    result = search_apu_v1_profile_rows(
        analysis,
        target,
        apu_v1_cost,
        incumbent_row_tile=incumbent_row,
        incumbent_plan_decision=incumbent_plan,
    )
    return result, events


def test_parity_row_domain_and_singleton_full_m_bypass():
    target = build_apu_v1_target()

    assert derive_apu_v1_profile_row_tiles(_analysis(), target) == (
        32,
        64,
        128,
        256,
    )
    assert derive_apu_v1_profile_row_tiles(_analysis(rows=128), target) == (
        32,
        64,
        128,
    )
    assert derive_apu_v1_profile_row_tiles(
        _analysis(rows=256, columns=512), target
    ) == (64, 128, 256)
    assert derive_apu_v1_profile_row_tiles(
        _analysis(rows=2000, columns=64), target
    ) == (512, 1024, 2000)
    assert derive_apu_v1_profile_row_tiles(_analysis(rows=1900, columns=1), target) == (
        1900,
    )
    assert derive_apu_v1_profile_row_tiles(_analysis(rows=77, reduction=1), target) == (
        77,
    )


def test_plan_domain_depends_on_outer_row_tile_not_inner_output_tile():
    target = build_apu_v1_target()
    analysis = _analysis()
    domains = {
        row: derive_apu_v1_profile_plan_decisions(analysis, target, row_tile=row)
        for row in (32, 64, 128, 256)
    }

    assert [len(domains[row]) for row in domains] == [4, 6, 8, 10]
    assert all(decision.output_tile_shape == (32, 1024) for decision in domains[64][1:])
    assert domains[64][1].output_tile_shape == domains[128][1].output_tile_shape
    assert domains[64][1].physical_fingerprint != domains[128][1].physical_fingerprint


def test_materialization_precedes_score_and_exact_incumbent_is_retained(monkeypatch):
    target = build_apu_v1_target()
    analysis = _analysis(rows=128, reduction=64)
    incumbent = _decision(analysis, target, 32)
    result, events = _fake_search(monkeypatch, analysis, target, 32, incumbent)

    assert len(events) == result.stats.materialize_attempts * 2
    assert all(
        events[index][0] == "materialize" and events[index + 1][0] == "score"
        for index in range(0, len(events), 2)
    )
    assert result.best_incumbent.is_incumbent
    assert result.best_incumbent.decisions["row_tile"] == 32
    assert result.best.objective_domain.scope == "whole_profile_composition"
    activation = result.activation
    assert activation.active is result.best_incumbent
    assert activation.promoted is False
    if result.best is not result.best_incumbent:
        assert activation.fallback_reason.startswith("shadow_only:")


def test_missing_incumbent_fails_closed_and_opaque_incumbent_stays_active(monkeypatch):
    target = build_apu_v1_target()
    analysis = _analysis(rows=64, reduction=32)
    with pytest.raises(MissingScheduleIncumbent):
        search_apu_v1_profile_rows(analysis, target, apu_v1_cost)

    incumbent = OpaqueScheduleIncumbent(
        decisions={"row_tile": 64, "plan": ("unmapped", 1)},
        payload=("frozen-artifact", 1),
        materialized=("frozen-source", 1),
        fingerprint=("opaque-profile", 1),
    )
    plan = _decision(analysis, target, 32)
    result, _events = _fake_search(monkeypatch, analysis, target, 32, plan)
    monkeypatch.setattr(
        profile_search,
        "_materialize",
        lambda problem, decision: SimpleNamespace(
            decision=decision,
            row_waves=1,
            column_repetitions=1,
            objective_domain=problem.objective_domain,
        ),
    )
    monkeypatch.setattr(
        profile_search,
        "_score",
        lambda problem, materialized: APUV1ProfileCompositionScore(
            1, 1, 1, 1, problem.objective_domain
        ),
    )
    opaque_result = search_apu_v1_profile_rows(
        analysis,
        target,
        apu_v1_cost,
        opaque_incumbent=incumbent,
    )

    assert result.best_incumbent.decisions["row_tile"] == 32
    assert opaque_result.best_incumbent is incumbent
    assert opaque_result.activation.active is incumbent
    assert opaque_result.activation.promoted is False


def test_exact_sources_and_costs_are_sensitive_to_profile_m(monkeypatch):
    target = build_apu_v1_target()
    emitted_source_fingerprints = []
    structural_source_fingerprints = []
    plan_fingerprints = []
    shard_cycles = []
    for rows in (32, 64, 128, 256):
        problem = profile_search._ProfileProblem(
            _analysis(rows=rows), target, apu_v1_cost, None
        )
        entry = problem.entries(rows)[2]
        decision = profile_search.APUV1ProfileRowDecision(rows, entry.decision)
        materialized = profile_search._materialize(problem, decision)
        score = profile_search._score(problem, materialized)
        emitted_source_fingerprints.append(materialized.emitted_source_fingerprint)
        structural_source_fingerprints.append(
            materialized.structural_source_fingerprint
        )
        plan_fingerprints.append(materialized.plan_fingerprint)
        shard_cycles.append(score.shard_cycles)
        assert materialized.shard_shape.manifest() == {
            "M": rows,
            "N": 1024,
            "K": 2200,
        }
        assert materialized.row_waves == materialized.column_repetitions == 1
        assert len(materialized.promotion_materialization_fingerprint) == 64
        assert len(materialized.runtime_source_fingerprint) == 64
        assert dict(materialized.realization.runtime_artifact.source_hashes)[
            "project/host.c"
        ]

    assert len(set(emitted_source_fingerprints)) == 4
    assert len(set(structural_source_fingerprints)) == 4
    assert len(set(plan_fingerprints)) == 4
    assert len(set(shard_cycles)) == 4

    baseline = profile_search._ProfileProblem(
        _analysis(rows=32), target, apu_v1_cost, None
    )
    monkeypatch.setattr(
        profile_search,
        "_bound_cost",
        lambda selected_target, _cost: SimpleNamespace(
            target=selected_target,
            fingerprint="changed-profile-cost-fingerprint",
        ),
    )
    changed_target = build_apu_v1_target()
    changed = profile_search._ProfileProblem(
        _analysis(rows=32), changed_target, apu_v1_cost, None
    )
    assert baseline.cost.fingerprint != changed.cost.fingerprint
    assert baseline.objective_domain != changed.objective_domain


def test_complete_runtime_mutation_is_rejected_before_profile_score(monkeypatch):
    from allo.pim import apu_v1_vector_runtime as runtime

    target = build_apu_v1_target()
    problem = profile_search._ProfileProblem(
        _analysis(rows=32, columns=16, reduction=16),
        target,
        apu_v1_cost,
        None,
    )
    entry = problem.entries(32)[2]
    materialized = profile_search._materialize(
        problem,
        profile_search.APUV1ProfileRowDecision(32, entry.decision),
    )
    artifact = materialized.realization.runtime_artifact
    profile_search._score(problem, materialized)
    original_inventory = runtime._template_inventory

    def mutated_inventory():
        hashes, modes = original_inventory()
        hashes = dict(hashes)
        path = next(iter(hashes))
        hashes[path] = "f" * 64
        return tuple(sorted(hashes.items())), modes

    monkeypatch.setattr(runtime, "_template_inventory", mutated_inventory)

    assert materialized.promotion_materialization_fingerprint is None
    with pytest.raises(RuntimeError, match="template changed"):
        artifact.assert_current(materialized.realization)


def test_underfilled_partial_collapses_but_row_wave_objective_remains_distinct():
    target = build_apu_v1_target()
    problem = profile_search._ProfileProblem(
        _analysis(rows=160), target, apu_v1_cost, None
    )
    first_entry = problem.entries(64)[2]
    second_entry = next(
        entry for entry in problem.entries(128) if entry.family == first_entry.family
    )
    first = profile_search._materialize(
        problem,
        profile_search.APUV1ProfileRowDecision(64, first_entry.decision),
    )
    second = profile_search._materialize(
        problem,
        profile_search.APUV1ProfileRowDecision(128, second_entry.decision),
    )
    first_score = profile_search._score(problem, first)
    second_score = profile_search._score(problem, second)

    assert first.final_partial_shape.rows == second.final_partial_shape.rows == 32
    assert (
        first.final_plan_emitted_source_fingerprint
        == second.final_plan_emitted_source_fingerprint
    )
    assert (first.row_waves, second.row_waves) == (3, 2)
    assert first_score.final_shard_cycles is not None
    assert second_score.final_shard_cycles is not None
    assert first_score.composed_cycles == (
        first_score.shard_cycles * 2 + first_score.final_shard_cycles
    )
    assert second_score.composed_cycles == (
        second_score.shard_cycles + second_score.final_shard_cycles
    )
    assert (
        first.promotion_materialization_fingerprint
        != second.promotion_materialization_fingerprint
    )


def test_unmaterializable_partial_and_incomparable_columns_fail_closed():
    target = build_apu_v1_target()
    problem = profile_search._ProfileProblem(
        _analysis(rows=160), target, apu_v1_cost, None
    )
    blocked = next(
        entry
        for entry in problem.entries(128)
        if entry.decision.accumulator_block == 4
        and all("dma_l4_l3" not in route for route in entry.decision.transfer_routes)
    )
    with pytest.raises(InfeasibleSchedule, match="partial row wave"):
        profile_search._materialize(
            problem,
            profile_search.APUV1ProfileRowDecision(128, blocked.decision),
        )
    with pytest.raises(UnsupportedAPUV1ProfileSearchError, match="exactly divide"):
        derive_apu_v1_profile_row_tiles(
            _analysis(rows=64, columns=100), target, column_extent=64
        )


def test_rename_invariance_excludes_workload_operand_loop_and_plan_names():
    first_target = build_apu_v1_target()
    second_target = build_apu_v1_target()
    first_problem = profile_search._ProfileProblem(
        _analysis(rows=64, columns=64, reduction=64),
        first_target,
        apu_v1_cost,
        None,
    )
    second_problem = profile_search._ProfileProblem(
        _analysis(
            rows=64,
            columns=64,
            reduction=64,
            names=("renamed", "i", "j", "k", "matrix_a", "matrix_b", "destination"),
        ),
        second_target,
        apu_v1_cost,
        None,
    )
    first_entry = first_problem.entries(64)[2]
    second_entry = second_problem.entries(64)[2]
    first = profile_search._materialize(
        first_problem,
        profile_search.APUV1ProfileRowDecision(64, first_entry.decision),
    )
    second = profile_search._materialize(
        second_problem,
        profile_search.APUV1ProfileRowDecision(64, second_entry.decision),
    )

    assert first_entry.decision == second_entry.decision
    assert first.semantic_fingerprint == second.semantic_fingerprint
    assert first.structural_source_fingerprint == second.structural_source_fingerprint
    assert first.emitted_source_fingerprint != second.emitted_source_fingerprint
    assert (
        first.plan_emitted_source_fingerprint != second.plan_emitted_source_fingerprint
    )
    assert (
        first.promotion_materialization_fingerprint
        != second.promotion_materialization_fingerprint
    )
    assert (
        first.emitted_source_fingerprint
        == hashlib.sha256(first.realization.device_source().encode("utf-8")).hexdigest()
    )
    with pytest.raises(ValueError, match="does not bind the realization"):
        replace(
            first,
            realization=SimpleNamespace(
                device_source=lambda: first.realization.device_source() + "\n"
            ),
        )
    identity_text = repr((first.decision, first.semantic_fingerprint))
    assert all(
        name not in identity_text
        for name in ("profile_kernel", "renamed", "matrix_a", "destination")
    )


def representative_gemm(
    left: uint16[32, 16],
    right: uint16[16, 8],
    output: uint16[32, 8],
):
    for row, column in allo.grid(32, 8):
        for depth in allo.reduction(16):
            output[row, column] += left[row, depth] * right[depth, column]


def test_standalone_search_does_not_change_public_apuv1_compile(monkeypatch):
    before = allo.compile(
        representative_gemm,
        build_apu_v1_target(),
        apu_v1_cost,
        backend="virtual",
    )
    target = build_apu_v1_target()
    analysis = _analysis(rows=32, columns=16, reduction=16)
    incumbent = _decision(analysis, target, 32)
    _fake_search(monkeypatch, analysis, target, 32, incumbent)
    after = allo.compile(
        representative_gemm,
        build_apu_v1_target(),
        apu_v1_cost,
        backend="virtual",
    )

    assert isinstance(before, APUv1VectorCallable)
    assert isinstance(after, APUv1VectorCallable)
    assert before.selected_plan.name == after.selected_plan.name
    assert (
        before.realization.promotion_materialization_fingerprint
        == after.realization.promotion_materialization_fingerprint
    )
