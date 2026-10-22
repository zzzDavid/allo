# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ranked matcher placement selection and bounded grid search (spec 003)."""

import time

import numpy as np
import pytest

import allo
from allo import spmw_match_engine
from allo.compiler import CompiledCallable, _stamp_matcher_work_scopes
from allo.customize import customize
from allo.dataflow import region as _df_region
from allo.ir.types import float16 as fp16
from allo.pim.costs import samsung_cost
from allo.pim.schedule_search import DecisionDomain, InfeasibleSchedule, grid_search
from allo.pim.targets import build_samsung_target
from allo.spmw_autoschedule import (
    _bucket_decision_map,
    _bucket_for_autoschedule,
    _matcher_placement_decision,
    _matcher_work_scope,
    _per_bucket_argmin_index,
    _samsung_enumerate,
    autoschedule,
)
from allo.spmw_codegen import compile_for_target, source_backend_binding
from allo.spmw_knobs import reset_active_liveness, set_active_liveness
from allo.spmw_liveness import trace_liveness
from allo.spmw_match import MatchTrace

M, K, ROWS = 128, 16, 1


@_df_region()
def small_gemv(W: fp16[M, K], x: fp16[K], y: fp16[M]):
    @allo.work(mapping=[16, 8], args=[W, x, y])
    def gemv(local_W: fp16[M, K], local_x: fp16[K], local_y: fp16[M]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * ROWS
        for i in range(ROWS):
            acc: fp16 = 0
            for k in range(K):
                acc += local_W[row0 + i, k] * local_x[k]
            local_y[row0 + i] = acc


@_df_region()
def small_mvt(A: fp16[M, M], y1: fp16[M], y2: fp16[M], x1: fp16[M], x2: fp16[M]):
    @allo.work(mapping=[16, 8], args=[A, y1, x1])
    def mv_a(local_A: fp16[M, M], local_y1: fp16[M], local_x1: fp16[M]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * ROWS
        for i in range(ROWS):
            acc: fp16 = 0
            for j in range(M):
                acc += local_A[row0 + i, j] * local_y1[j]
            local_x1[row0 + i] = acc

    @allo.work(mapping=[16, 8], args=[A, y2, x2])
    def mv_b(local_A: fp16[M, M], local_y2: fp16[M], local_x2: fp16[M]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * ROWS
        for i in range(ROWS):
            acc: fp16 = 0
            for k in range(M):
                acc += local_A[k, row0 + i] * local_y2[k]
            local_x2[row0 + i] = acc


# A 2x2 grid keeps the plain-`get_name()` reference (quadratic in module size)
# cheap for the naming-equality check.
@_df_region()
def tiny_gemv(W: fp16[4, K], x: fp16[K], y: fp16[4]):
    @allo.work(mapping=[2, 2], args=[W, x, y])
    def gemv(local_W: fp16[4, K], local_x: fp16[K], local_y: fp16[4]):
        pid, uid = allo.get_wid()
        row0 = pid * 2 + uid
        acc: fp16 = 0
        for k in range(K):
            acc += local_W[row0, k] * local_x[k]
        local_y[row0] = acc


# The weight parameter is literally named `B`, which is also the Samsung
# runtime kwarg for the broadcast vector.
@_df_region()
def colliding_gemv(B: fp16[M, K], x: fp16[K], y: fp16[M]):
    @allo.work(mapping=[16, 8], args=[B, x, y])
    def gemv(local_B: fp16[M, K], local_x: fp16[K], local_y: fp16[M]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * ROWS
        for i in range(ROWS):
            acc: fp16 = 0
            for k in range(K):
                acc += local_B[row0 + i, k] * local_x[k]
            local_y[row0 + i] = acc


def _match(region, target):
    schedule = customize(region, enable_tensor=False)
    trace = spmw_match_engine.match_workload(target, schedule.module)
    _stamp_matcher_work_scopes(target, schedule.module, trace)
    return schedule, trace


@pytest.fixture(scope="module")
def gemv_case():
    target = build_samsung_target()
    schedule, trace = _match(small_gemv, target)
    return target, schedule, trace


@pytest.fixture(scope="module")
def mvt_case():
    target = build_samsung_target()
    _schedule, trace = _match(small_mvt, target)
    return target, trace


def _kernel_choice_set_sizes(target, trace):
    """Independent recount of |K_k|: keys every bucket of a kernel enumerates,
    under the same whole-trace liveness `autoschedule` activates."""
    token = set_active_liveness(trace_liveness(trace))
    try:
        kernels = {}
        for _scope, matches in _bucket_for_autoschedule(trace):
            group_id = _matcher_work_scope(matches[0]).group_id
            _t, _c, key_map = _bucket_decision_map(
                target, trace, _samsung_enumerate, matches
            )
            keys = kernels.get(group_id)
            kernels[group_id] = (
                list(key_map) if keys is None else [k for k in keys if k in key_map]
            )
    finally:
        reset_active_liveness(token)
    return [len(keys) for keys in kernels.values()]


def test_asmstate_names_equal_plain_get_name(monkeypatch):
    target = build_samsung_target()
    schedule, trace = _match(tiny_gemv, target)
    # `get_name(False)` resolves to the no-AsmState overload, i.e. the plain
    # `get_name()` the matcher used before the per-function AsmState.
    monkeypatch.setattr(spmw_match_engine, "AsmState", lambda _func: False)
    reference = spmw_match_engine.match_workload(target, schedule.module)
    _stamp_matcher_work_scopes(target, schedule.module, reference)

    assert len(trace.matches) == len(reference.matches) == 4
    for new, old in zip(trace.matches, reference.matches):
        assert new.func_name == old.func_name
        assert new.work_id == old.work_id
        assert [
            (o.role, o.memref_name, o.value_ref, o.indices) for o in new.operands
        ] == [(o.role, o.memref_name, o.value_ref, o.indices) for o in old.operands]
        assert new.result_memref_name == old.result_memref_name
        assert new.enclosing_loops == old.enclosing_loops
        assert new.op_range == old.op_range
        assert new.extra == old.extra


def test_ranked_uses_one_decision_per_kernel(gemv_case):
    target, _schedule, trace = gemv_case
    placements = autoschedule(target, trace, samsung_cost)
    buckets = _bucket_for_autoschedule(trace)

    keys = {
        _matcher_placement_decision(
            MatchTrace(trace.target_name, trace.module_name, matches=matches),
            placement,
        )
        for (_scope, matches), placement in zip(buckets, placements)
    }
    ranking = placements.placement_ranking
    assert len(keys) == 1
    assert len(ranking.kernels) == 1
    assert ranking.kernels[0].bucket_count == 128
    assert ranking.kernels[0].fallback is None
    assert not hasattr(placements, "schedule_search_result")
    assert not hasattr(placements, "active_materialization")


@pytest.mark.parametrize("case", ["gemv_case", "mvt_case"])
def test_ranked_evaluation_count_is_bounded(case, request, monkeypatch):
    target, trace = request.getfixturevalue(case)[0], request.getfixturevalue(case)[-1]
    expected = _kernel_choice_set_sizes(target, trace)
    assert len(expected) == (1 if case == "gemv_case" else 2)

    from allo import spmw_plan

    original = spmw_plan.build_execution_graph
    calls = []

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(spmw_plan, "build_execution_graph", counting)
    placements = autoschedule(target, trace, samsung_cost)

    assert len(calls) == sum(expected)
    assert [len(k.scores) for k in placements.placement_ranking.kernels] == expected


def test_ranked_is_no_worse_than_uniform_per_bucket_argmin(gemv_case):
    """Spec 003 test 4 expected ranked == per-bucket argmin on a uniform kernel.
    On this region every bucket's own argmin is the same dual_fiber key, but
    scored over the whole kernel that key costs more than the ranked bank_row
    key, so only the kernel-level dominance holds."""
    target, _schedule, trace = gemv_case
    bound = samsung_cost.bind(target)
    kernel = autoschedule(target, trace, bound).placement_ranking.kernels[0]
    cycles_by_position = {position: cycles for position, cycles, _ in kernel.scores}

    argmin_keys = set()
    token = set_active_liveness(trace_liveness(trace))
    try:
        for _scope, matches in _bucket_for_autoschedule(trace):
            bucket_trace, candidates, key_map = _bucket_decision_map(
                target, trace, _samsung_enumerate, matches
            )
            argmin = _per_bucket_argmin_index(target, bucket_trace, candidates, bound)
            argmin_keys.add(
                _matcher_placement_decision(bucket_trace, candidates[argmin])
            )
            ordered_keys = list(key_map)
    finally:
        reset_active_liveness(token)

    assert len(argmin_keys) == 1
    argmin_position = ordered_keys.index(next(iter(argmin_keys)))
    chosen_cycles = cycles_by_position[kernel.chosen_position]
    assert chosen_cycles == min(cycles_by_position.values())
    assert chosen_cycles <= cycles_by_position[argmin_position]


def test_ranked_falls_through_infeasible_codegen(gemv_case, monkeypatch):
    target, _schedule, trace = gemv_case
    from allo import spmw_codegen

    original = spmw_codegen._materialize_matcher_codegen
    calls = []

    def first_infeasible(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise InfeasibleSchedule("forced top-rank rejection")
        return original(*args, **kwargs)

    monkeypatch.setattr(
        spmw_codegen, "_materialize_matcher_codegen", first_infeasible
    )
    kernel = autoschedule(target, trace, samsung_cost).placement_ranking.kernels[0]

    first_position = kernel.scores[0][0]
    second_position = kernel.scores[1][0]
    assert kernel.rejections == ((first_position, "forced top-rank rejection"),)
    assert kernel.chosen_position == second_position


def test_grid_search_coverage_is_closed_form_without_constraints():
    domains = tuple(DecisionDomain(f"d{index}", (0, 1)) for index in range(40))
    started = time.perf_counter()
    result = grid_search(
        domains,
        build=lambda decisions: dict(decisions),
        materialize=lambda payload: payload,
        score=lambda realized: sum(realized.values()),
        objective=lambda estimate: estimate,
        objective_domain="closed_form_coverage",
        max_complete_assignments=4,
    )
    assert time.perf_counter() - started < 1.0
    assert result.stats.complete_assignments_total == 2**40
    assert result.stats.complete_assignments_covered == 4
    assert result.stats.truncated


def test_source_binding_aliases_samsung_roles(gemv_case):
    target, _schedule, trace = gemv_case
    binding = source_backend_binding(target, trace, ("W", "x", "y"))
    assert dict(binding.inputs) == {"A": "W", "B": "x"}
    assert binding.output == "y"

    collide_target = build_samsung_target()
    collide_schedule, collide_trace = _match(colliding_gemv, collide_target)
    compiled = compile_for_target(collide_target, collide_trace, cost=samsung_cost)
    callable_ = CompiledCallable(
        colliding_gemv,
        collide_target,
        collide_schedule,
        collide_trace,
        compiled,
    )
    W = np.zeros((M, K), dtype=np.float16)
    x = np.zeros(K, dtype=np.float16)
    y = np.zeros(M, dtype=np.float16)
    with pytest.raises(ValueError, match="collides with the Samsung runtime role"):
        callable_(W, x, y)
