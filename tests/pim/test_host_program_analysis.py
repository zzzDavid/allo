# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-program residency analysis (the host-xcel scheduling pass) unit tests.

Backend-agnostic, NO simulator: exercises `allo.spmw_host_program.analyze` and the
cost-from-schedule seam directly. The load-bearing assertion (acceptance #5 of the
host-programming factoring) is that the RUN path and the COST path derive weight
residency from the SAME analysis -- the residency win (`P + B*(E+R)` vs
`B*(P+E+R)`) emerges from where `scatter(W)` sits in the program, never a hardcoded
`weight_resident` flag.
"""

from __future__ import annotations

import sys

import allo
from allo.dataflow import region as _df_region
from allo.ir.types import float32 as fp16
from allo.spmw_host_program import (
    analyze, resolve_shapes, schedule_residency, schedule_cost,
)

M, K, B = 4096, 1024, 8
_banks = allo.host_xfer.banks
_grf_a = allo.host_xfer.grf_a

# Calibrated per-phase costs @4096x1024 (design_doc/results/samsung-batched-gemv.md).
_P, _E, _R = 11368, 3702, 181


@_df_region()
def _gemv_top(W: fp16[M, K], X: fp16[B, K], Y: fp16[B, M]):
    @allo.work(mapping=[16, 8], args=[W, X, Y])
    def gemv(lW: fp16[M, K], lX: fp16[B, K], lY: fp16[B, M]):
        pass


@allo.host_program(_gemv_top)
def _amortized(W, X, Y):
    allo.host_xfer.scatter(W, _banks)            # preload ONCE -> resident
    for b in range(B):
        allo.host_xfer.broadcast(X[b], _grf_a)
        allo.launch("gemv", W, X[b], Y[b])
        allo.host_xfer.gather(Y[b], _banks)


@allo.host_program(_gemv_top)
def _native(W, X, Y):
    for b in range(B):
        allo.host_xfer.scatter(W, _banks)        # re-preload EVERY vector
        allo.host_xfer.broadcast(X[b], _grf_a)
        allo.launch("gemv", W, X[b], Y[b])
        allo.host_xfer.gather(Y[b], _banks)


def _sched(hp):
    mod = sys.modules[__name__]
    return analyze(hp, resolve_shapes(hp, vars(mod)))


def test_hoisted_scatter_coalesces_into_one_group():
    """Scatter(W) hoisted out of the batch loop -> a single coalesced group of B
    launches reusing the resident weight."""
    s = _sched(_amortized)
    assert s.n_groups == 1 and s.n_launches == B
    assert s.groups[0].is_batched and len(s.groups[0].launches) == B
    assert s.groups[0].weight == "W" and s.groups[0].weight_shape == (M, K)
    assert s.external_inputs[0] == "W"           # weight + B distinct X[b] rows
    assert s.final == "Y[%d]" % (B - 1)


def test_scatter_in_loop_is_not_coalesced():
    """Scatter(W) inside the loop re-stages every vector -> B singleton groups."""
    s = _sched(_native)
    assert s.n_groups == B and s.n_launches == B
    assert all(not g.is_batched for g in s.groups)


def test_residency_is_read_from_program_structure():
    """`schedule_residency` -> the cost model's (stage_resident, batch) levers,
    flipped purely by scatter placement -- not a hardcoded flag."""
    assert schedule_residency(_sched(_amortized)) == (True, B)
    assert schedule_residency(_sched(_native)) == (False, 1)


def test_cost_and_run_share_one_analysis():
    """The crossover falls out of the SAME schedule the executor coalesces on:
    one preload per group + exec/readback per launch."""
    preload = lambda m, k: _P
    er = lambda m, k, n: _E + _R
    amort = schedule_cost(_sched(_amortized), preload, er)
    native = schedule_cost(_sched(_native), preload, er)
    assert amort == _P + B * (_E + _R)           # 42432
    assert native == B * (_P + _E + _R)          # 122008
    assert amort < native                        # the beat, from structure


def test_batch_one_is_parity():
    """B=1 reduces both schedules to a single preload+launch (no residency win)."""

    @allo.host_program(_gemv_top)
    def one(W, X, Y):
        allo.host_xfer.scatter(W, _banks)
        allo.host_xfer.broadcast(X[0], _grf_a)
        allo.launch("gemv", W, X[0], Y[0])
        allo.host_xfer.gather(Y[0], _banks)

    s = _sched(one)
    assert s.n_groups == 1 and s.n_launches == 1
    assert schedule_residency(s) == (False, 1)   # one launch: nothing to amortize
    preload = lambda m, k: _P
    er = lambda m, k, n: _E + _R
    assert schedule_cost(s, preload, er) == _P + _E + _R   # 15251
