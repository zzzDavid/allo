# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Declared work grid versus the target unit tree (release audit C-04)."""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest

import allo
from allo.dataflow import region as _df_region
from allo.ir.types import float16 as fp16
from allo.pim.costs import samsung_cost
from allo.pim.targets import build_samsung_target
from allo.spmw_autoschedule import MatcherWorkScope
from allo.spmw_codegen import (
    _OUTER_SUBSET_LAUNCH_BACKENDS,
    _SINGLE_ITEM_LAUNCH_BACKENDS,
    _check_work_grid,
)
from allo.spmw_match import MatchedOp, MatchTrace

M, K = 128, 16


@_df_region()
def gemv_4x4(W: fp16[M, K], x: fp16[K], y: fp16[M]):
    @allo.work(mapping=[4, 4], args=[W, x, y])
    def gemv(local_W: fp16[M, K], local_x: fp16[K], local_y: fp16[M]):
        pid, uid = allo.get_wid()
        row0 = (pid * 4 + uid) * 8
        for i in range(8):
            acc: fp16 = 0
            for k in range(K):
                acc += local_W[row0 + i, k] * local_x[k]
            local_y[row0 + i] = acc


@_df_region()
def gemv_16x8(W: fp16[M, K], x: fp16[K], y: fp16[M]):
    @allo.work(mapping=[16, 8], args=[W, x, y])
    def gemv(local_W: fp16[M, K], local_x: fp16[K], local_y: fp16[M]):
        pid, uid = allo.get_wid()
        row0 = pid * 8 + uid
        acc: fp16 = 0
        for k in range(K):
            acc += local_W[row0, k] * local_x[k]
        local_y[row0] = acc


N = 4096


@_df_region()
def flat_add(a: fp16[N], b: fp16[N], c: fp16[N]):
    @allo.work(mapping=[1], args=[a, b, c])
    def kernel(la: fp16[N], lb: fp16[N], lc: fp16[N]):
        for i in range(N):
            lc[i] = la[i] + lb[i]


def test_samsung_4x4_grid_is_an_error():
    with pytest.raises(ValueError, match=r"work grid \[4, 4\] \(16 items\)"):
        allo.compile(gemv_4x4, build_samsung_target(), samsung_cost, backend="virtual")


def test_samsung_full_grid_compiles():
    compiled = allo.compile(
        gemv_16x8, build_samsung_target(), samsung_cost, backend="virtual"
    )
    assert compiled.placement_ranking.kernels[0].bucket_count == 128


def test_samsung_single_item_elementwise_compiles():
    compiled = allo.compile(
        flat_add, build_samsung_target(), samsung_cost, backend="virtual"
    )
    assert compiled.placement_ranking.kernels[0].bucket_count == 1


def _trace(target_name, shape, group_id=1):
    matches = []
    for index, work_id in enumerate(itertools.product(*(range(e) for e in shape))):
        match = MatchedOp(
            target_op_name="MAC",
            func_name=f"k{group_id}_{index}",
            work_id=work_id,
            enclosing_loops=[],
            operands=[],
            result_memref_name=None,
            op_range=("a", "b"),
        )
        match.extra["spmw_work_scope"] = MatcherWorkScope(
            group_id=group_id, work_id=work_id, group_shape=tuple(shape)
        )
        matches.append(match)
    return MatchTrace(target_name, "grid", matches)


def _target(name, factors):
    product = 1
    for factor in factors:
        product *= factor
    return SimpleNamespace(name=name, work_grid=lambda: (list(factors), product))


@pytest.mark.parametrize(
    "name, factors, shape, ok",
    [
        ("samsung_hbm_pim", [16, 8], [16, 8], True),  # full fanout
        ("samsung_hbm_pim", [16, 8], [16], True),  # outer prefix
        ("samsung_hbm_pim", [16, 8], [1], True),  # declared single item
        ("samsung_hbm_pim", [16, 8], [4, 4], False),
        ("samsung_hbm_pim", [16, 8], [16, 4], False),
        ("aim", [32, 4, 4], [32], True),
        ("aim", [32, 4, 4], [32, 4], True),
        ("aim", [32, 4, 4], [8, 4], False),
        ("aim", [32, 4, 4], [4], True),  # 4 replicas on channel groups
        ("aim", [32, 4, 4], [64], False),
        ("upmem", [64, 24], [1], True),
        ("upmem", [64, 24], [8], True),  # 8 of the 64 DPUs
        ("upmem", [64, 24], [128], False),
        ("upmem", [64, 24], [2, 4], False),
        ("samsung_hbm_pim", [16, 8], [8], False),  # no subset launch
        ("apu_v1", [4], [1], False),  # single item only through the coalesced axis
        ("apu_v1", [4], [4], True),
    ],
)
def test_work_grid_rule_table(name, factors, shape, ok):
    target, trace = _target(name, factors), _trace(name, shape)
    if ok:
        assert _check_work_grid(target, trace) == len(trace.matches)
    else:
        with pytest.raises(ValueError, match="work grid"):
            _check_work_grid(target, trace)


def test_single_item_backends_are_declared():
    assert _SINGLE_ITEM_LAUNCH_BACKENDS == {"samsung_hbm_pim", "upmem", "aim", "attacc"}
    assert _OUTER_SUBSET_LAUNCH_BACKENDS == {"upmem", "aim"}


def test_virtual_only_target_accepts_single_item():
    target, trace = _target("virtual_only", [8]), _trace("virtual_only", [1])
    assert _check_work_grid(target, trace) == 1
    with pytest.raises(ValueError, match="work grid"):
        _check_work_grid(target, _trace("virtual_only", [3]))


def test_multi_kernel_grids_are_checked_per_kernel():
    target = _target("samsung_hbm_pim", [16, 8])
    good, bad = _trace("samsung_hbm_pim", [1], 1), _trace("samsung_hbm_pim", [4, 4], 2)
    trace = MatchTrace("samsung_hbm_pim", "two", good.matches + bad.matches)
    with pytest.raises(ValueError, match="kernel group 2"):
        _check_work_grid(target, trace)
