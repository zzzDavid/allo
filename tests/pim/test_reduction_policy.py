# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reduction-order policy (``allo.compile(..., reduction_order=...)``).

Compile-only: the policy is stamped on accumulating matches, regrouping
candidates are pruned before ranking under ``"strict"`` on a floating-point
accumulator, integer accumulators keep their splits, and backends without an
order-preserving program fail naming the policy.  The simulator check of the
Samsung ordered program is ``samsung_hbm_pim/test_samsung_reduction_order.py``.
Design: ``docs/release/design/reduction-policy.md``.
"""

from __future__ import annotations

import numpy as np
import pytest

import allo
from allo.customize import customize
from allo.dataflow import region as _df_region
from allo.ir.types import bfloat16, float16 as fp16, int32
from allo.pim.costs import aim_cost, samsung_cost
from allo.pim.targets import build_aim_target, build_samsung_target, build_upmem_target
from allo.spmw_match_engine import match_workload
from allo.spmw_reduction_policy import (
    ReductionOrderError,
    ReductionPolicy,
    normalize_reduction_order,
    reduction_policy,
)
from allo.spmw_samsung import samsung_vendor_gemv_layout

M, K, ROWS = 2048, 64, 16


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


ELT_N = 131072


def flat_add(a: fp16[ELT_N], b: fp16[ELT_N], c: fp16[ELT_N]):
    for i in range(ELT_N):
        c[i] = a[i] + b[i]


UPMEM_M, UPMEM_K, UPMEM_ROWS = 768, 128, 96


@_df_region()
def upmem_gemv(W: int32[UPMEM_M, UPMEM_K], x: int32[UPMEM_K], y: int32[UPMEM_M]):
    @allo.work(mapping=[8], args=[W, x, y])
    def mv(
        local_W: int32[UPMEM_M, UPMEM_K],
        local_x: int32[UPMEM_K],
        local_y: int32[UPMEM_M],
    ):
        (dpu,) = allo.get_wid()
        for i in range(UPMEM_ROWS):
            local_y[dpu * UPMEM_ROWS + i] = 0
            for j in range(UPMEM_K):
                local_y[dpu * UPMEM_ROWS + i] += (
                    local_W[dpu * UPMEM_ROWS + i, j] * local_x[j]
                )


AIM_M = AIM_K = 256


@_df_region()
def aim_gemv(W: bfloat16[AIM_M, AIM_K], x: bfloat16[AIM_K], y: bfloat16[AIM_M]):
    @allo.work(mapping=[32], args=[W, x, y])
    def mv(
        local_W: bfloat16[AIM_M, AIM_K],
        local_x: bfloat16[AIM_K],
        local_y: bfloat16[AIM_M],
    ):
        (channel,) = allo.get_wid()
        row0 = channel * 8
        for i in range(8):
            acc: bfloat16 = 0
            for j in range(AIM_K):
                acc += local_W[row0 + i, j] * local_x[j]
            local_y[row0 + i] = acc


def _modes(compiled):
    (kernel,) = compiled.placement_ranking.kernels
    return {mode.split("+")[0] for _position, _cycles, mode in kernel.scores}


@pytest.fixture(scope="module")
def samsung_strict():
    return allo.compile(
        small_gemv, build_samsung_target(), samsung_cost, backend="virtual"
    )


@pytest.fixture(scope="module")
def samsung_reassociate():
    return allo.compile(
        small_gemv,
        build_samsung_target(),
        samsung_cost,
        backend="virtual",
        reduction_order="reassociate",
    )


def test_policy_semantics():
    assert normalize_reduction_order(None) == "strict"
    assert normalize_reduction_order("reassociate") == "reassociate"
    with pytest.raises(ValueError, match="reduction_order must be one of"):
        normalize_reduction_order("relaxed")
    assert not ReductionPolicy("strict", "f16").may_regroup
    assert not ReductionPolicy("strict", "bf16").may_regroup
    # An unreadable accumulator type is treated as floating point.
    assert not ReductionPolicy("strict", None).may_regroup
    # Integer addition wraps modulo 2^width: any grouping is exact.
    for integer in ("i32", "i64", "ui16", "si8", "index"):
        assert ReductionPolicy("strict", integer).may_regroup
    assert ReductionPolicy("reassociate", "f32").may_regroup
    with pytest.raises(ValueError, match="reduction_order must be one of"):
        allo.compile(
            small_gemv, build_samsung_target(), samsung_cost, reduction_order="fast"
        )


def test_match_workload_stamps_accumulating_matches():
    target = build_samsung_target()
    module = customize(small_gemv, enable_tensor=False).module
    strict = match_workload(target, module)
    assert strict.matches
    for match in strict.matches:
        assert match.extra["reduction_policy"] == ReductionPolicy("strict", "f16")
    relaxed = match_workload(target, module, reduction_order="reassociate")
    assert {m.extra["reduction_policy"].order for m in relaxed.matches} == {
        "reassociate"
    }

    elementwise = match_workload(
        target, customize(flat_add, enable_tensor=False).module
    )
    assert elementwise.matches
    assert all("reduction_policy" not in m.extra for m in elementwise.matches)
    assert all(reduction_policy(m) is None for m in elementwise.matches)


def test_samsung_strict_prunes_regrouping_placements(samsung_strict):
    assert _modes(samsung_strict) == {"temporal"}
    pruned = dict(samsung_strict.reduction_policy_pruned)
    assert set(pruned) == {"bank_row", "grf_staged", "dual_fiber"}
    assert all("lane partial sums in FP32" in reason for reason in pruned.values())
    assert "2 bank fibers" in pruned["dual_fiber"]
    # Pruned before ranking: no regrouping key was scored or rejected (the
    # only rejections are per-work-item CRF bodies of the ordered program).
    (kernel,) = samsung_strict.placement_ranking.kernels
    assert all("per_workid" in reason for _position, reason in kernel.rejections)
    assert dict(kernel.policy_pruned) == pruned

    words = [
        (w.type_, w.dst_, w.src0_, w.src1_, w.isAuto_, w.loopCounter_, w.loopOffset_)
        for w in samsung_strict.compiled.cmds
    ]
    assert words == [
        ("MOV", "GRF_B", "EVEN_BANK", "A_OUT", 0, 0, 0),
        ("MAC", "GRF_B", "GRF_A", "EVEN_BANK", 0, 0, 0),
        ("JUMP", "A_OUT", "A_OUT", "A_OUT", 0, K - 1, 2),
        ("NOP", "A_OUT", "A_OUT", "A_OUT", 0, 0, 0),
        ("EXIT", "A_OUT", "A_OUT", "A_OUT", 0, 0, 0),
    ]
    (program,) = samsung_strict.compiled.layout_ctx.programs
    assert program.kind == "TEMPORAL_REDUCE"
    assert program.lever_record["grf_residency"] == {"local_x": "host"}
    assert (
        samsung_strict.compiled.layout_ctx.runtime_route(samsung_strict.compiled.cmds)[
            0
        ]
        == "GENERIC_TEMPORAL_REDUCE"
    )


def test_samsung_reassociate_ranks_both_families(samsung_reassociate):
    assert _modes(samsung_reassociate) == {
        "bank_row",
        "grf_staged",
        "dual_fiber",
        "temporal",
    }
    assert samsung_reassociate.reduction_policy_pruned == ()
    (kernel,) = samsung_reassociate.placement_ranking.kernels
    chosen = dict((p, m) for p, _c, m in kernel.scores)[kernel.chosen_position]
    assert not chosen.startswith("temporal")
    scores = {m.split("+")[0]: c for _p, c, m in kernel.scores}
    # The ordered program issues one reduction element per step.
    assert scores["temporal"] > 10 * scores["bank_row"]


def test_samsung_forced_regrouping_layout_fails_under_strict(samsung_reassociate):
    target = samsung_reassociate.target
    vendor = samsung_vendor_gemv_layout(target, samsung_reassociate.trace)
    with pytest.raises(ReductionOrderError, match="reduction_order='strict'"):
        allo.compile(small_gemv, target, samsung_cost, layout=vendor)
    # Under strict the vendor layout is not even enumerated.
    with pytest.raises(NotImplementedError, match="pruned: reduction_order='strict'"):
        samsung_vendor_gemv_layout(
            target,
            allo.compile(small_gemv, target, samsung_cost, backend="virtual").trace,
        )


def test_result_records_the_policy(samsung_strict, samsung_reassociate):
    operands = (
        np.zeros((M, K), np.float16),
        np.zeros(K, np.float16),
        np.zeros(M, np.float16),
    )
    strict = samsung_strict(*operands)
    assert strict.kind == "virtual"
    assert strict.extra["reduction_order"] == "strict"
    assert strict.extra["reduction_policy"] == [
        {"accumulator_type": "f16", "may_regroup": False}
    ]
    relaxed = samsung_reassociate(*operands)
    assert relaxed.extra["reduction_order"] == "reassociate"
    assert relaxed.extra["reduction_policy"] == [
        {"accumulator_type": "f16", "may_regroup": True}
    ]


def test_strict_permits_splitting_an_integer_reduction():
    from allo.compiler import _stamp_matcher_work_scopes
    from allo.spmw_autoschedule import _bucket_for_autoschedule, _upmem_enumerate

    target = build_upmem_target()
    schedule = customize(upmem_gemv, enable_tensor=False)
    trace = match_workload(target, schedule.module)
    _stamp_matcher_work_scopes(target, schedule.module, trace)
    policies = {
        m.extra["reduction_policy"]
        for m in trace.matches
        if "reduction_policy" in m.extra
    }
    assert policies == {ReductionPolicy("strict", "i32")}
    assert all(policy.may_regroup for policy in policies)
    # The 12-tasklet split of every row's reduction survives the strict policy.
    candidates = _upmem_enumerate(target, _bucket_for_autoschedule(trace)[0][1])
    assert candidates
    assert {candidate.extra["tasklets"] for candidate in candidates} == {12}


def test_aim_float_reduction_needs_permission():
    target = build_aim_target()
    with pytest.raises(ReductionOrderError, match="adder tree") as error:
        allo.compile(aim_gemv, target, aim_cost, backend="virtual")
    assert "reduction_order='reassociate'" in str(error.value)
    compiled = allo.compile(
        aim_gemv, target, aim_cost, backend="virtual", reduction_order="reassociate"
    )
    assert compiled.reduction_order == "reassociate"
