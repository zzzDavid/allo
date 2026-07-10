# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent APUg2 GEMM schedule-search migration gates."""

import hashlib
import inspect
import json
from dataclasses import FrozenInstanceError, replace

import allo
import numpy as np
import pytest
from allo.ir.types import uint16
from allo.perf import cost as cost_spec
from allo.perf import rule
from allo.pim import apu_g2_vector_program as apu_program
from allo.pim.costs import apu_g2 as apu_costs
from allo.pim.apu_g2_contraction import plan_apu_g2_rank_n_contraction
from allo.pim.apu_g2_vectorize import UnsupportedAPUG2ContractionError
from allo.pim.apu_g2_vector_program import (
    APUG2ColumnBatchedGemmCallable,
    APUG2PersistentGemmMaterialization,
    APUG2PersistentGemmSchedule,
    search_apu_g2_persistent_gemm_schedules,
)
from allo.pim.costs.apu_g2 import (
    APUG2_GEMM_MAX_BATCH_COLUMNS,
    APUG2_GEMM_MAX_REDUCTION_TILE,
    apu_g2_cost,
    estimate_apu_g2_u16_gemm_wall_us,
)
from allo.pim.schedule_promotion import (
    CorrectnessEvidence,
    MetricMeasurements,
    MetricPromotionCell,
    NoisyHardwarePolicy,
    ObjectiveMetricBridge,
    PromotionEvidence,
    SchedulePromotionGate,
    ScheduleEvidence,
    SemanticScope,
    WarmupEvidence,
)
from allo.pim.schedule_search import (
    InfeasibleIncumbent,
    InfeasibleSchedule,
    ScheduleObjectiveDomain,
    guarded_schedule_activation,
)
from allo.pim.targets import build_apu_g2_target
from allo.spmw_codegen import RunResult


def representative_gemm(A: uint16[60, 80], B: uint16[80, 70], output: uint16[60, 70]):
    for row, column in allo.grid(60, 70):
        for depth in allo.reduction(80):
            output[row, column] += A[row, depth] * B[depth, column]


def differently_named_gemm(
    lhs: uint16[60, 80], rhs: uint16[80, 70], result: uint16[60, 70]
):
    for row, column in allo.grid(60, 70):
        for depth in allo.reduction(80):
            result[row, column] += lhs[row, depth] * rhs[depth, column]


def output_aliased_gemm(matrix: uint16[8, 8], state: uint16[8, 8]):
    for row, column in allo.grid(8, 8):
        for depth in allo.reduction(8):
            state[row, column] += matrix[row, depth] * state[depth, column]


def _plan(workload=representative_gemm):
    module = allo.customize(workload, enable_tensor=False).module
    return plan_apu_g2_rank_n_contraction(module)


def _bound_cost():
    target = build_apu_g2_target()
    return target, apu_g2_cost.bind(target)


def _promotion_evidence(result):
    platform = ("platform", "apu_v2", "hardware-campaign-1")
    metric_domain = ScheduleObjectiveDomain(
        metric="wall_us",
        target="apu_v2",
        target_revision="hardware-campaign-1",
        model_fingerprint=("repeated-board-runs", 1),
        fidelity="hardware",
        scope="whole_program_transport",
        unit="microseconds",
        direction="minimize",
    )
    correctness = CorrectnessEvidence.exact_pass(("modular-u16-oracle", 1))
    semantic_scope = SemanticScope(
        ("rank-two-u16-contraction", 1),
        (("complete-output", True), ("epilogue", "preserved")),
        complete=True,
    )

    def schedule_evidence(candidate):
        fingerprint = candidate.materialized.promotion_materialization_fingerprint
        return ScheduleEvidence.from_schedule(
            candidate,
            correctness=correctness,
            semantic_scope=semantic_scope,
            scored_fingerprint=fingerprint,
            emitted_fingerprint=fingerprint,
            platform_fingerprint=platform,
        )

    warmup = WarmupEvidence(1, True)
    return PromotionEvidence(
        schedule_evidence(result.best),
        schedule_evidence(result.best_incumbent),
        (
            MetricPromotionCell(
                MetricMeasurements(metric_domain, (99,) * 7, warmup, platform),
                MetricMeasurements(metric_domain, (100,) * 7, warmup, platform),
                NoisyHardwarePolicy(1, 7, 1.0, 1.05),
                ObjectiveMetricBridge(
                    result.best.objective_domain,
                    metric_domain,
                    result.best.objective_domain.unit,
                    metric_domain.unit,
                    result.best.objective_domain.direction,
                    metric_domain.direction,
                    "identity_metric",
                    ("apu-v2-wall-time-calibration", 1),
                ),
            ),
        ),
    )


def test_materialization_precedes_wall_score_and_retains_explicit_incumbent(
    monkeypatch, tmp_path
):
    plan = _plan()
    target, cost = _bound_cost()
    events = []
    original_materialize = apu_program._materialize_apu_g2_persistent_gemm_schedule
    original_score = cost.score_materialization

    def materialize(candidate_plan, candidate_target, schedule):
        events.append("materialize")
        return original_materialize(candidate_plan, candidate_target, schedule)

    def score(materialized):
        events.append("score")
        return original_score(materialized)

    monkeypatch.setattr(
        apu_program,
        "_materialize_apu_g2_persistent_gemm_schedule",
        materialize,
    )
    monkeypatch.setattr(cost, "score_materialization", score)

    result = search_apu_g2_persistent_gemm_schedules(plan, target, cost)

    assert len(events) == 62
    assert events[::2] == ["materialize"] * 31
    assert events[1::2] == ["score"] * 31
    assert result.best_incumbent is not None
    assert dict(result.best_incumbent.decisions) == {
        "batch_columns": 31,
        "reduction_tile": 128,
    }
    assert result.stats.incumbent_evaluated is True
    assert result.stats.complete_assignments_considered == 30
    assert result.stats.materialize_attempts == result.stats.score_attempts == 31
    assert len(result.ranked) == 31
    assert isinstance(result.best.payload, APUG2PersistentGemmSchedule)
    assert isinstance(result.best.materialized, APUG2PersistentGemmMaterialization)
    assert len(result.best.materialized.source_fingerprint) == 64
    source_hashes = dict(result.best.materialized.source_hashes)
    assert source_hashes["project/device/apu_g2_u16_gemm.cc"]
    assert source_hashes["project/host_u16_gemm.cc"]
    assert source_hashes["contract/build.json"]
    assert source_hashes["contract/abi.json"]
    assert source_hashes["contract/executor.json"]
    assert any(path.startswith("runtime/") for path in source_hashes)
    artifact = result.best.materialized.runtime_artifact
    written = artifact.write_project(tmp_path / "project")
    for relative, mode in artifact.project_modes:
        assert (written / relative).stat().st_mode & 0o777 == mode
    assert (
        len({candidate.materialized.fingerprint for candidate in result.ranked}) == 31
    )
    assert all(
        candidate.materialized.promotion_materialization_fingerprint
        == candidate.materialized.fingerprint
        for candidate in result.ranked
    )
    with pytest.raises(FrozenInstanceError):
        result.best.payload.batch_columns = 1
    with pytest.raises(FrozenInstanceError):
        result.best.materialized.rows = 1


def test_materialization_rejection_never_reaches_wall_score(monkeypatch):
    target, cost = _bound_cost()
    monkeypatch.setattr(
        apu_program,
        "_matrixized_gemm_spec",
        lambda _plan: {
            "row_axes": ("row",),
            "column_axes": ("column",),
            "rows": 65_537,
            "columns": 1,
            "reduction": 1,
        },
    )
    monkeypatch.setattr(
        cost,
        "score_materialization",
        lambda _materialized: pytest.fail("rejected geometry was scored"),
    )

    with pytest.raises(InfeasibleIncumbent) as caught:
        search_apu_g2_persistent_gemm_schedules(object(), target, cost)

    assert caught.value.diagnostics[-1].stage == "materialize"
    assert "physical VL64/L1 capacity" in caught.value.diagnostics[-1].reason


def test_persistent_runtime_rejects_output_operand_alias_recurrence():
    target, cost = _bound_cost()

    with pytest.raises(InfeasibleIncumbent, match="canonical"):
        search_apu_g2_persistent_gemm_schedules(
            _plan(output_aliased_gemm), target, cost
        )
    with pytest.raises(
        UnsupportedAPUG2ContractionError,
        match="not to alias output storage",
    ):
        allo.compile(
            output_aliased_gemm,
            build_apu_g2_target(),
            apu_g2_cost,
            backend="virtual",
        )


def test_persistent_search_rejects_graph_only_cost_specs():
    target = build_apu_g2_target()

    @cost_spec(target="apu_v2")
    def graph_only_cost(candidate_target):
        @rule(candidate_target.op("ADD_U16"))
        def add(_event, context):
            context.step(cycles=1)

    with pytest.raises(TypeError, match="materialization cost scorer"):
        search_apu_g2_persistent_gemm_schedules(
            _plan(), target, graph_only_cost.bind(target)
        )


def test_all_runtime_batch_widths_materialize_but_reduction_tile_stays_pinned(
    monkeypatch,
):
    plan = _plan()
    target, _cost = _bound_cost()
    monkeypatch.setattr(
        apu_costs,
        "estimate_apu_g2_u16_gemm_wall_us",
        lambda *_args, **_kwargs: pytest.fail("materialization computed wall score"),
    )

    materialized = apu_program._materialize_apu_g2_persistent_gemm_schedule(
        plan, target, APUG2PersistentGemmSchedule(31, 128)
    )
    assert (materialized.rows, materialized.reduction, materialized.columns) == (
        60,
        80,
        70,
    )
    for batch_columns in range(1, 32):
        candidate = apu_program._materialize_apu_g2_persistent_gemm_schedule(
            plan, target, APUG2PersistentGemmSchedule(batch_columns, 128)
        )
        assert candidate.schedule.batch_columns == batch_columns
    with pytest.raises(InfeasibleSchedule, match="reduction tile is fixed"):
        apu_program._materialize_apu_g2_persistent_gemm_schedule(
            plan, target, APUG2PersistentGemmSchedule(31, 64)
        )


def test_persistent_materialization_refreshes_runtime_source_hashes(monkeypatch):
    from allo.pim import apu_g2_runtime

    plan = _plan()
    target, _cost = _bound_cost()
    original_snapshot = apu_g2_runtime._source_snapshot
    calls = 0

    def changing_snapshot():
        nonlocal calls
        calls += 1
        sources, hashes = original_snapshot()
        if calls > 1:
            sources = dict(sources)
            hashes = dict(hashes)
            sources["host_u16_gemm.cc"] += "\n// exact source mutation\n"
            hashes["host_u16_gemm.cc"] = hashlib.sha256(
                sources["host_u16_gemm.cc"].encode("utf-8")
            ).hexdigest()
        return sources, hashes

    monkeypatch.setattr(apu_g2_runtime, "_source_snapshot", changing_snapshot)
    schedule = APUG2PersistentGemmSchedule(31, 128)
    first = apu_program._materialize_apu_g2_persistent_gemm_schedule(
        plan, target, schedule
    )
    second = apu_program._materialize_apu_g2_persistent_gemm_schedule(
        plan, target, schedule
    )

    assert calls == 2
    assert first.source_fingerprint != second.source_fingerprint
    assert first.promotion_materialization_fingerprint != (
        second.promotion_materialization_fingerprint
    )


def test_stale_runtime_source_is_rejected_before_candidate_score(monkeypatch):
    from allo.pim import apu_g2_runtime

    plan = _plan()
    target, cost = _bound_cost()
    materialized = apu_program._materialize_apu_g2_persistent_gemm_schedule(
        plan, target, APUG2PersistentGemmSchedule(31, 128)
    )
    fingerprint = materialized.promotion_materialization_fingerprint
    original_snapshot = apu_g2_runtime._source_snapshot

    def mutated_snapshot():
        sources, hashes = original_snapshot()
        sources = dict(sources)
        hashes = dict(hashes)
        sources["host_u16_gemm.cc"] += "\n// post-score mutation\n"
        hashes["host_u16_gemm.cc"] = hashlib.sha256(
            sources["host_u16_gemm.cc"].encode("utf-8")
        ).hexdigest()
        return sources, hashes

    monkeypatch.setattr(apu_g2_runtime, "_source_snapshot", mutated_snapshot)

    assert fingerprint is not None
    assert materialized.promotion_materialization_fingerprint is None
    assert materialized.current_source_fingerprint != materialized.source_fingerprint
    with pytest.raises(InfeasibleSchedule, match="source changed"):
        apu_program._score_apu_g2_persistent_gemm_materialization(materialized, cost)


def test_stale_runtime_source_retains_incumbent_before_activation(monkeypatch):
    from allo.pim import apu_g2_runtime

    original_estimator = apu_costs.estimate_apu_g2_u16_gemm_wall_us

    def favor_small_batch(*args, **kwargs):
        estimate = dict(original_estimator(*args, **kwargs))
        estimate["wall_us"] = float(kwargs["batch_columns"])
        return estimate

    monkeypatch.setattr(
        apu_costs,
        "estimate_apu_g2_u16_gemm_wall_us",
        favor_small_batch,
    )
    target, cost = _bound_cost()
    searched = search_apu_g2_persistent_gemm_schedules(_plan(), target, cost)
    evidence = _promotion_evidence(searched)
    original_snapshot = apu_g2_runtime._source_snapshot

    def mutated_snapshot():
        sources, hashes = original_snapshot()
        sources = dict(sources)
        hashes = dict(hashes)
        sources["host_u16_gemm.cc"] += "\n// mutation after evidence\n"
        hashes["host_u16_gemm.cc"] = hashlib.sha256(
            sources["host_u16_gemm.cc"].encode("utf-8")
        ).hexdigest()
        return sources, hashes

    monkeypatch.setattr(apu_g2_runtime, "_source_snapshot", mutated_snapshot)
    activation = guarded_schedule_activation(
        searched,
        promotion_gate=SchedulePromotionGate(evidence),
    )

    assert activation.promoted is False
    assert activation.active is searched.best_incumbent
    assert activation.fallback_reason.startswith(
        "recommended_materialization_fingerprint_unavailable"
    )


def test_caller_supplied_g2_files_cannot_attest_hardware_platform(
    monkeypatch, tmp_path
):
    from allo.pim import apu_g2_u16_gemm_runtime

    components = {}
    for category in ("sdk", "toolchain", "firmware"):
        path = tmp_path / f"{category}.contract"
        path.write_bytes(f"{category}-revision-1".encode("ascii"))
        components[category] = {
            category: {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        }
    manifest = {
        "schema": "tenon-promotion-platform-v1",
        "target": "apu_v2",
        "hardware_family": "gemini-ii",
        **components,
    }
    contract = tmp_path / "platform.json"
    contract.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv("TENON_APU_G2_PLATFORM_CONTRACT", str(contract))

    target, _cost = _bound_cost()
    materialized = apu_program._materialize_apu_g2_persistent_gemm_schedule(
        _plan(), target, APUG2PersistentGemmSchedule(31, 128)
    )
    artifact = materialized.runtime_artifact
    assert artifact.platform_fingerprint is None
    assert artifact.current_platform_fingerprint() is None
    assert materialized.promotion_platform_fingerprint is None
    assert materialized.promotion_materialization_fingerprint is not None

    (tmp_path / "firmware.contract").write_bytes(b"firmware-revision-2")
    assert materialized.promotion_platform_fingerprint is None
    artifact.assert_current(apu_g2_u16_gemm_runtime.run_apu_g2_u16_gemm)


def test_objective_domain_is_structural_and_workload_name_independent():
    first_target, first_cost = _bound_cost()
    second_target, second_cost = _bound_cost()

    first = search_apu_g2_persistent_gemm_schedules(
        _plan(representative_gemm), first_target, first_cost
    )
    second = search_apu_g2_persistent_gemm_schedules(
        _plan(differently_named_gemm), second_target, second_cost
    )

    assert tuple(
        inspect.signature(search_apu_g2_persistent_gemm_schedules).parameters
    ) == (
        "plan",
        "target",
        "cost",
    )
    assert first.best.objective_domain.metric == "wall_us"
    assert first.best.objective_domain.target == "apu_v2"
    assert first.best.objective_domain.model_fingerprint == first_cost.fingerprint
    assert first.best.objective_domain.fidelity == "calibrated"
    assert first.best.objective_domain.scope == "whole_program_transport"
    assert first.best.objective_domain.target_revision.startswith("fingerprinted:")
    assert second.best.objective_domain == first.best.objective_domain
    assert [
        candidate.materialized.semantic_fingerprint for candidate in first.ranked
    ] == [candidate.materialized.semantic_fingerprint for candidate in second.ranked]
    assert [
        candidate.materialized.promotion_materialization_fingerprint
        for candidate in first.ranked
    ] == [
        candidate.materialized.promotion_materialization_fingerprint
        for candidate in second.ranked
    ]
    assert "workload" not in repr(first.best.objective_domain)
    assert np.isfinite(first.best.objective)
    assert first.best.objective == first.best.score["wall_us"]


def test_persistent_materialization_fingerprint_binds_retained_epilogue():
    plan = _plan()
    scaled = replace(
        plan,
        analysis=replace(
            plan.analysis,
            product_coefficient=5,
            accumulator_coefficient=4,
        ),
    )
    target, _cost = _bound_cost()
    schedule = APUG2PersistentGemmSchedule(31, 128)
    unit = apu_program._materialize_apu_g2_persistent_gemm_schedule(
        plan,
        target,
        schedule,
    )
    nonunit = apu_program._materialize_apu_g2_persistent_gemm_schedule(
        scaled,
        target,
        schedule,
    )

    assert unit.epilogue == (1, 1)
    assert nonunit.epilogue == (5, 4)
    assert unit.semantic_fingerprint != nonunit.semantic_fingerprint
    assert (
        unit.promotion_materialization_fingerprint
        != nonunit.promotion_materialization_fingerprint
    )


def test_virtual_compile_preserves_schedule_and_device_tick_estimate():
    compiled = allo.compile(
        representative_gemm,
        build_apu_g2_target(),
        apu_g2_cost,
        backend="virtual",
    )

    assert isinstance(compiled, APUG2ColumnBatchedGemmCallable)
    assert compiled.estimate().cycles == 9_806 + 9_806 + 9_911
    assert len(compiled.schedule_search_result.ranked) == 31
    assert compiled.schedule_search_result.best_incumbent.is_incumbent
    assert compiled.schedule_activation.active is (
        compiled.schedule_search_result.best_incumbent
    )
    assert compiled.fallback_reason == (
        None
        if compiled.schedule_search_result.best
        is compiled.schedule_search_result.best_incumbent
        else "shadow_only: promotion evidence was not requested"
    )
    assert compiled.selected_transport_schedule == APUG2PersistentGemmSchedule(31, 128)
    assert (
        compiled.selected_transport_estimate
        is compiled.schedule_search_result.best_incumbent.score
    )
    transport = compiled.execution_graph.metadata["transport_schedule"]
    assert (transport["batch_columns"], transport["reduction_tile"]) == (31, 128)
    assert transport["host_wall_estimate"] == compiled.selected_transport_estimate

    A = np.zeros((60, 80), dtype=np.uint16)
    B = np.zeros((80, 70), dtype=np.uint16)
    output = np.zeros((60, 70), dtype=np.uint16)
    result = compiled(A, B, output)
    assert result.cycles == compiled.estimate().cycles
    assert (
        result.extra["schedule"]["batch_columns"],
        result.extra["schedule"]["reduction_tile"],
    ) == (
        31,
        128,
    )


def test_promotion_fails_closed_without_current_platform_fingerprint(
    monkeypatch,
):
    original_estimator = apu_costs.estimate_apu_g2_u16_gemm_wall_us

    def favor_small_batch(*args, **kwargs):
        estimate = dict(original_estimator(*args, **kwargs))
        estimate["wall_us"] = float(kwargs["batch_columns"])
        return estimate

    monkeypatch.setattr(
        apu_costs,
        "estimate_apu_g2_u16_gemm_wall_us",
        favor_small_batch,
    )
    target, cost = _bound_cost()
    searched = search_apu_g2_persistent_gemm_schedules(_plan(), target, cost)
    assert dict(searched.best.decisions)["batch_columns"] == 1

    compiled = allo.compile(
        representative_gemm,
        build_apu_g2_target(),
        apu_g2_cost,
        backend="virtual",
        promotion_evidence=_promotion_evidence(searched),
    )

    assert compiled.schedule_activation.promoted is False
    assert (
        compiled.schedule_activation.active
        is compiled.schedule_search_result.best_incumbent
    )
    assert compiled.fallback_reason.startswith(
        "recommended_platform_fingerprint_unavailable"
    )
    assert compiled.selected_transport_schedule.batch_columns == 31
    assert (
        compiled.schedule_activation.recommended.materialized.promotion_materialization_fingerprint
        == searched.best.materialized.promotion_materialization_fingerprint
    )


def test_device_runtime_receives_selected_batch_columns_without_hardware(
    monkeypatch,
):
    from allo.pim import apu_g2_u16_gemm_runtime as runtime

    compiled = allo.compile(
        representative_gemm,
        build_apu_g2_target(),
        apu_g2_cost,
        backend="device",
    )
    observed = {}

    def fake_runtime(
        lhs,
        rhs,
        accumulator,
        *,
        alpha,
        beta,
        batch_columns,
        runtime_artifact,
    ):
        observed.update(
            alpha=alpha,
            beta=beta,
            batch_columns=batch_columns,
            source_fingerprint=runtime_artifact.source_fingerprint,
        )
        return RunResult(
            cycles=123,
            stdout="mock persistent GEMM",
            backend="apu_v2",
            extra={"outputs": {"out": accumulator.copy()}},
        )

    monkeypatch.setattr(runtime, "run_apu_g2_u16_gemm", fake_runtime)
    A = np.zeros((60, 80), dtype=np.uint16)
    B = np.zeros((80, 70), dtype=np.uint16)
    output = np.zeros((60, 70), dtype=np.uint16)

    result = compiled(A, B, output)

    assert result.cycles == 123
    assert observed == {
        "alpha": 1,
        "beta": 1,
        "batch_columns": 31,
        "source_fingerprint": (
            compiled.selected_transport_materialization.source_fingerprint
        ),
    }


def test_wall_estimator_rejects_uncalibrated_reduction_tiles():
    with pytest.raises(ValueError, match="calibrated runtime tile"):
        estimate_apu_g2_u16_gemm_wall_us(
            60,
            80,
            70,
            reduction_tile=APUG2_GEMM_MAX_REDUCTION_TILE + 1,
        )


def test_cost_fingerprint_tracks_explicit_wall_model_provenance(monkeypatch):
    provenance = apu_g2_cost.fingerprint_data
    assert provenance["max_batch_columns"] == APUG2_GEMM_MAX_BATCH_COLUMNS
    assert provenance["max_reduction_tile"] == APUG2_GEMM_MAX_REDUCTION_TILE
    assert provenance["sample_revision"]

    first_target = build_apu_g2_target()
    first = apu_g2_cost.bind(first_target).fingerprint
    monkeypatch.setitem(
        provenance,
        "scalar_mac_us",
        provenance["scalar_mac_us"] + 0.01,
    )
    second_target = build_apu_g2_target()
    second = apu_g2_cost.bind(second_target).fingerprint

    assert second != first
