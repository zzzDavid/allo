"""Fail-closed APUg2 direct-reduction/dot-tile shadow search."""

import hashlib
from dataclasses import FrozenInstanceError, replace

import allo
import numpy as np
import pytest
from allo.ir.types import uint16
from allo.pim import apu_g2_route_search as route_search
from allo.pim.apu_g2_contraction import plan_apu_g2_rank_n_contraction
from allo.pim.apu_g2_route_search import (
    DIRECT_REDUCTION,
    DOT_TILE,
    UnsupportedAPUG2RouteSearchError,
    search_apu_g2_single_contraction_routes,
)
from allo.pim.apu_g2_vector_program import APUG2GemvCallable
from allo.pim.costs.apu_g2 import apu_g2_cost
from allo.pim.schedule_search import (
    InfeasibleSchedule,
    ScheduleObjectiveDomain,
    guarded_schedule_activation,
)
from allo.pim.targets import build_apu_g2_target


def representative_gemv(
    matrix: uint16[120, 120],
    vector: uint16[120],
    output: uint16[120],
):
    for row in allo.grid(120):
        for reduction in allo.reduction(120):
            output[row] += matrix[row, reduction] * vector[reduction]


def renamed_gemv(
    left: uint16[120, 120],
    right: uint16[120],
    result: uint16[120],
):
    for outer in allo.grid(120):
        for inner in allo.reduction(120):
            result[outer] += left[outer, inner] * right[inner]


def transposed_rhs_gemv(
    vector: uint16[120],
    matrix: uint16[120, 96],
    output: uint16[96],
):
    for row in allo.grid(96):
        for reduction in allo.reduction(120):
            output[row] += vector[reduction] * matrix[reduction, row]


def nonunit_gemv(
    matrix: uint16[120, 120],
    vector: uint16[120],
    output: uint16[120],
):
    for row in allo.grid(120):
        output[row] = output[row] * 2
        for reduction in allo.reduction(120):
            output[row] += matrix[row, reduction] * vector[reduction]


def output_aliased_gemv(matrix: uint16[120, 120], state: uint16[120]):
    for row in allo.grid(120):
        for reduction in allo.reduction(120):
            state[row] += matrix[row, reduction] * state[reduction]


def oversized_gemv(
    matrix: uint16[40, 300],
    vector: uint16[300],
    output: uint16[40],
):
    for row in allo.grid(40):
        for reduction in allo.reduction(300):
            output[row] += matrix[row, reduction] * vector[reduction]


def rank_two_contraction(
    left: uint16[4, 5],
    right: uint16[5, 3],
    output: uint16[4, 3],
):
    for row, column in allo.grid(4, 3):
        for reduction in allo.reduction(5):
            output[row, column] += left[row, reduction] * right[reduction, column]


def _plan(workload=representative_gemv):
    module = allo.customize(workload, enable_tensor=False).module
    return plan_apu_g2_rank_n_contraction(module)


def _bound_cost():
    target = build_apu_g2_target()
    return target, apu_g2_cost.bind(target)


def _by_route(result):
    return {candidate.decisions["route"]: candidate for candidate in result.ranked}


def test_two_frozen_routes_materialize_before_score_with_direct_incumbent(
    monkeypatch,
):
    target, cost = _bound_cost()
    events = []
    original_materialize = route_search._materialize_apu_g2_single_contraction_route
    original_score = route_search._score_apu_g2_single_contraction_route

    def materialize(*args):
        events.append(("materialize", args[3].route))
        return original_materialize(*args)

    def score(materialized, candidate_cost):
        events.append(("score", materialized.route))
        return original_score(materialized, candidate_cost)

    monkeypatch.setattr(
        route_search,
        "_materialize_apu_g2_single_contraction_route",
        materialize,
    )
    monkeypatch.setattr(
        route_search,
        "_score_apu_g2_single_contraction_route",
        score,
    )

    result = search_apu_g2_single_contraction_routes(_plan(), target, cost)
    candidates = _by_route(result)

    assert events == [
        ("materialize", DIRECT_REDUCTION),
        ("score", DIRECT_REDUCTION),
        ("materialize", DOT_TILE),
        ("score", DOT_TILE),
    ]
    assert set(candidates) == {DIRECT_REDUCTION, DOT_TILE}
    assert result.stats.materialize_attempts == result.stats.score_attempts == 2
    assert result.best_incumbent is candidates[DIRECT_REDUCTION]
    assert result.best_incumbent.is_incumbent is True
    assert candidates[DOT_TILE].is_incumbent is False
    direct_identity = candidates[DIRECT_REDUCTION].materialized.executor_identity
    dot_identity = candidates[DOT_TILE].materialized.executor_identity
    assert direct_identity[:2] == (
        "allo.pim.apu_g2_gemv_runtime",
        "run_apu_g2_u16_gemv",
    )
    assert dot_identity[:2] == (
        "allo.pim.apu_g2_dot_tile_runtime",
        "run_apu_g2_u16_dot_tile",
    )
    assert len(direct_identity[2]) == len(dot_identity[2]) == 64
    for candidate in candidates.values():
        source_hashes = dict(candidate.materialized.source_hashes)
        assert source_hashes["project/device/apu_g2_u16_gemv.cc"]
        assert source_hashes["project/device/apu_g2_u16_dot_tile.cc"]
        assert source_hashes["project/host_gemv.cc"]
        assert source_hashes["project/host_dot_tile.cc"]
        assert source_hashes["contract/executor.json"]
        assert any(path.startswith("runtime/") for path in source_hashes)
    with pytest.raises(FrozenInstanceError):
        candidates[DIRECT_REDUCTION].materialized.route = DOT_TILE


def test_semantic_and_materialization_fingerprints_ignore_all_retained_names():
    first_target, first_cost = _bound_cost()
    second_target, second_cost = _bound_cost()
    first = _by_route(
        search_apu_g2_single_contraction_routes(
            _plan(representative_gemv), first_target, first_cost
        )
    )
    second = _by_route(
        search_apu_g2_single_contraction_routes(
            _plan(renamed_gemv), second_target, second_cost
        )
    )

    for route in (DIRECT_REDUCTION, DOT_TILE):
        assert (
            first[route].materialized.semantic_fingerprint
            == second[route].materialized.semantic_fingerprint
        )
        assert (
            first[route].materialized.promotion_materialization_fingerprint
            == second[route].materialized.promotion_materialization_fingerprint
        )


@pytest.mark.parametrize(
    "workload,match",
    [
        (nonunit_gemv, "unit epilogue"),
        (output_aliased_gemv, "non-aliasing output"),
        (oversized_gemv, "direct-reduction executor"),
        (rank_two_contraction, "one-output-axis comparison domain"),
    ],
)
def test_noncomparable_contractions_are_rejected_before_search(workload, match):
    target, cost = _bound_cost()

    with pytest.raises(UnsupportedAPUG2RouteSearchError, match=match):
        search_apu_g2_single_contraction_routes(_plan(workload), target, cost)


def test_complete_project_mutation_changes_every_route_materialization(
    monkeypatch,
):
    from allo.pim import apu_g2_runtime

    target, cost = _bound_cost()
    baseline = _by_route(search_apu_g2_single_contraction_routes(_plan(), target, cost))
    original_snapshot = apu_g2_runtime._source_snapshot

    def mutated_snapshot():
        sources, hashes = original_snapshot()
        sources = dict(sources)
        hashes = dict(hashes)
        sources["host_gemv.cc"] += "\n// route-search provenance mutation\n"
        hashes["host_gemv.cc"] = hashlib.sha256(
            sources["host_gemv.cc"].encode("utf-8")
        ).hexdigest()
        return sources, hashes

    monkeypatch.setattr(
        apu_g2_runtime,
        "_source_snapshot",
        mutated_snapshot,
    )
    changed = _by_route(search_apu_g2_single_contraction_routes(_plan(), target, cost))

    assert (
        baseline[DIRECT_REDUCTION].materialized.source_fingerprint
        != changed[DIRECT_REDUCTION].materialized.source_fingerprint
    )
    assert (
        baseline[DIRECT_REDUCTION].materialized.promotion_materialization_fingerprint
        != changed[DIRECT_REDUCTION].materialized.promotion_materialization_fingerprint
    )
    assert (
        baseline[DOT_TILE].materialized.source_fingerprint
        != changed[DOT_TILE].materialized.source_fingerprint
    )


def test_stale_project_and_substituted_callable_fail_before_route_score(
    monkeypatch,
):
    from allo.pim import apu_g2_runtime

    target, cost = _bound_cost()
    direct = _by_route(search_apu_g2_single_contraction_routes(_plan(), target, cost))[
        DIRECT_REDUCTION
    ].materialized
    original_snapshot = apu_g2_runtime._source_snapshot

    substituted = replace(direct, executor=route_search._digest)
    assert substituted.promotion_materialization_fingerprint is None
    with pytest.raises(InfeasibleSchedule, match="source changed"):
        route_search._score_apu_g2_single_contraction_route(substituted, cost)

    def mutated_snapshot():
        sources, hashes = original_snapshot()
        sources = dict(sources)
        hashes = dict(hashes)
        sources["device/apu_g2_u16_gemv.cc"] += "\n// stale route source\n"
        hashes["device/apu_g2_u16_gemv.cc"] = hashlib.sha256(
            sources["device/apu_g2_u16_gemv.cc"].encode("utf-8")
        ).hexdigest()
        return sources, hashes

    monkeypatch.setattr(apu_g2_runtime, "_source_snapshot", mutated_snapshot)
    assert direct.promotion_materialization_fingerprint is None
    with pytest.raises(InfeasibleSchedule, match="source changed"):
        route_search._score_apu_g2_single_contraction_route(direct, cost)


def test_routes_share_one_typed_repeated_device_compute_domain():
    target, cost = _bound_cost()
    result = search_apu_g2_single_contraction_routes(_plan(), target, cost)
    domains = {candidate.objective_domain for candidate in result.ranked}

    assert len(domains) == 1
    domain = next(iter(domains))
    assert isinstance(domain, ScheduleObjectiveDomain)
    assert domain.metric == "device_pipeline_ticks"
    assert domain.fidelity == "real_card_calibrated_model"
    assert domain.scope == "device_compute_repeated_throughput"
    assert all(
        candidate.objective == candidate.score.device_pipeline_ticks
        for candidate in result.ranked
    )
    assert all(
        candidate.materialized.target_name == "apu_v2"
        and candidate.materialized.cost_fingerprint == cost.fingerprint
        and candidate.materialized.target_revision == domain.target_revision
        for candidate in result.ranked
    )


def test_guarded_activation_retains_direct_incumbent_when_shadow_prefers_dot(
    monkeypatch,
):
    original_score = route_search._score_apu_g2_single_contraction_route

    def favor_dot(materialized, cost):
        score = original_score(materialized, cost)
        return replace(
            score,
            device_pipeline_ticks=(1.0 if materialized.route == DOT_TILE else 2.0),
        )

    monkeypatch.setattr(
        route_search,
        "_score_apu_g2_single_contraction_route",
        favor_dot,
    )
    target, cost = _bound_cost()
    result = search_apu_g2_single_contraction_routes(_plan(), target, cost)
    activation = guarded_schedule_activation(result)

    assert result.best.decisions["route"] == DOT_TILE
    assert activation.recommended is result.best
    assert activation.active is result.best_incumbent
    assert activation.active.decisions["route"] == DIRECT_REDUCTION
    assert activation.promoted is False
    assert activation.fallback_reason.startswith("shadow_only:")


def test_score_rejects_transport_or_wall_time_materialization():
    target, cost = _bound_cost()
    result = search_apu_g2_single_contraction_routes(_plan(), target, cost)
    direct = _by_route(result)[DIRECT_REDUCTION].materialized
    direct.execution_graph.metadata["transport_schedule"] = {
        "host_wall_estimate": {"wall_us": 1.0}
    }

    with pytest.raises(InfeasibleSchedule, match="wall-time"):
        route_search._score_apu_g2_single_contraction_route(direct, cost)


def test_score_does_not_dispatch_on_diagnostic_program_names():
    target, cost = _bound_cost()
    result = search_apu_g2_single_contraction_routes(_plan(), target, cost)
    direct = _by_route(result)[DIRECT_REDUCTION].materialized
    baseline = route_search._score_apu_g2_single_contraction_route(direct, cost)

    direct.execution_graph.metadata["program"] = (
        "user_persistent_streaming_chain_diagnostic"
    )
    renamed = route_search._score_apu_g2_single_contraction_route(direct, cost)

    assert renamed == baseline


def test_standalone_search_does_not_change_public_compile_dispatch():
    target, cost = _bound_cost()
    before = allo.compile(
        representative_gemv,
        build_apu_g2_target(),
        apu_g2_cost,
        backend="virtual",
    )
    search_apu_g2_single_contraction_routes(_plan(), target, cost)
    after = allo.compile(
        representative_gemv,
        build_apu_g2_target(),
        apu_g2_cost,
        backend="virtual",
    )

    assert isinstance(before, APUG2GemvCallable)
    assert isinstance(after, APUG2GemvCallable)
    assert not hasattr(after, "schedule_search_result")


@pytest.mark.parametrize("workload", [representative_gemv, transposed_rhs_gemv])
def test_materialized_executor_contracts_are_modularly_equivalent(
    monkeypatch,
    workload,
):
    from allo.pim import apu_g2_dot_tile_runtime, apu_g2_gemv_runtime

    def modular_dot(left, right, accumulator, alpha=1, beta=1):
        dot = np.sum(
            left.astype(np.uint64) * right.astype(np.uint64),
            axis=1,
            dtype=np.uint64,
        )
        return (
            (np.uint64(alpha) * dot + np.uint64(beta) * accumulator.astype(np.uint64))
            & np.uint64(0xFFFF)
        ).astype(np.uint16)

    def software_direct(matrix, vector, accumulator, *, repetitions=8):
        assert repetitions == 8
        right = np.broadcast_to(vector, matrix.shape)
        return modular_dot(matrix, right, accumulator)

    def software_dot_tile(
        left,
        right,
        *,
        accumulator=None,
        alpha=1,
        beta=0,
        repetitions=8,
    ):
        assert repetitions == 8
        return modular_dot(left, right, accumulator, alpha, beta)

    monkeypatch.setattr(
        apu_g2_gemv_runtime,
        "run_apu_g2_u16_gemv",
        software_direct,
    )
    monkeypatch.setattr(
        apu_g2_dot_tile_runtime,
        "run_apu_g2_u16_dot_tile",
        software_dot_tile,
    )
    target, cost = _bound_cost()
    candidates = _by_route(
        search_apu_g2_single_contraction_routes(_plan(workload), target, cost)
    )
    rng = np.random.default_rng(7)
    facts = candidates[DIRECT_REDUCTION].materialized.facts
    operands = tuple(
        rng.integers(0, 65536, shape, dtype=np.uint16) for shape in facts.operand_shapes
    )
    accumulator = rng.integers(
        0,
        65536,
        facts.output_shape,
        dtype=np.uint16,
    )

    direct = candidates[DIRECT_REDUCTION].materialized.execute(
        *operands,
        accumulator,
        repetitions=8,
    )
    tiled = candidates[DOT_TILE].materialized.execute(
        *operands,
        accumulator,
        repetitions=8,
    )

    np.testing.assert_array_equal(direct, tiled)
    assert facts.matrix_transposed is (workload is transposed_rhs_gemv)
