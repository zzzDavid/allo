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


# --------------------------------------------------------------------- #
# Launch order into the execution graph (spec 001 D3)
# --------------------------------------------------------------------- #

from types import SimpleNamespace

import pytest

from allo.ir.types import float16
from allo.pim.costs import samsung_cost
from allo.pim.targets import build_samsung_target
from allo.spmw_plan import launch_schedule
from allo.spmw_target import LaunchRecord

# 16 work items (the outer prefix of Samsung's 16 x 8 grid), 4 rows each.
_LM, _LK, _LR = 64, 64, 4
_hx = allo.host_xfer


@_df_region()
def _two_kernels(
    W: float16[_LM, _LK],
    V: float16[_LM, _LK],
    x: float16[_LK],
    y: float16[_LM],
    z: float16[_LM],
):
    @allo.work(mapping=[16], args=[W, x, y])
    def mv1(lW: float16[_LM, _LK], lx: float16[_LK], ly: float16[_LM]):
        (c,) = allo.get_wid()
        for i in range(_LR):
            acc: float16 = 0
            for k in range(_LK):
                acc += lW[c * _LR + i, k] * lx[k]
            ly[c * _LR + i] = acc

    @allo.work(mapping=[16], args=[V, y, z])
    def mv2(lV: float16[_LM, _LK], ly: float16[_LK], lz: float16[_LM]):
        (c,) = allo.get_wid()
        for i in range(_LR):
            acc: float16 = 0
            for k in range(_LK):
                acc += lV[c * _LR + i, k] * ly[k]
            lz[c * _LR + i] = acc


@allo.host_program(_two_kernels)
def _chained(W, V, x, y, z):
    _hx.scatter(W, _hx.banks)
    _hx.scatter(V, _hx.banks)
    _hx.broadcast(x, _hx.grf_a)
    allo.launch("mv1", W, x, y)
    _hx.gather(y, _hx.banks)
    _hx.broadcast(y, _hx.grf_a)
    allo.launch("mv2", V, y, z)
    _hx.gather(z, _hx.banks)


@allo.host_program(_two_kernels)
def _legacy_order(W, V, x, y, z):
    _hx.scatter(W, _hx.banks)
    _hx.scatter(V, _hx.banks)
    _hx.broadcast(x, _hx.grf_a)
    allo.launch("mv1", W, x, y)
    allo.launch("mv2", V, y, z)
    _hx.gather(y, _hx.banks)
    _hx.gather(z, _hx.banks)


def _virtual(host_moves):
    return allo.compile(
        _two_kernels,
        build_samsung_target(),
        samsung_cost,
        backend="virtual",
        host_moves=host_moves,
    )


def _graph_rows(compiled):
    return [
        (a.id, a.depends_on, a.latency_cycles)
        for a in compiled.execution_graph.activities
    ]


def test_launch_schedule_orders_host_transfers_between_kernels():
    """gather(y)/broadcast(y) between the launches sit after kernel 1's
    terminals and before kernel 2's first activity."""
    compiled = _virtual(_chained)
    assert compiled.compiled.launch_schedule is not None
    acts = list(compiled.execution_graph.activities)
    by_id = {a.id: a for a in acts}
    group_of = lambda a: a.metadata.get("group_id")
    g1, g2 = dict.fromkeys(group_of(a) for a in acts if group_of(a) is not None)

    gather_y = next(a for a in acts if a.id.startswith("host:egress:0:"))
    bcast_y = next(a for a in acts if a.id.startswith("host:ingress:3:"))
    g1_terminals = {
        a.id for a in acts if group_of(a) == g1 and a.metadata["phase"] == "post"
    }
    assert set(gather_y.depends_on) == g1_terminals and len(g1_terminals) == 16
    assert bcast_y.depends_on == (gather_y.id,)

    g2_first = [
        a
        for a in acts
        if group_of(a) == g2
        and not any(group_of(by_id[d]) == g2 for d in a.depends_on)
    ]
    assert g2_first and all(a.depends_on == (bcast_y.id,) for a in g2_first)


def test_legacy_order_schedule_matches_unscheduled_graph():
    """A schedule equal to the legacy order (ingress, launches in group order,
    gathers) reproduces the launch_schedule=None graph and cycles."""
    scheduled = _virtual(_legacy_order)
    legacy = _virtual(list(_legacy_order.moves))
    assert scheduled.compiled.launch_schedule is not None
    assert legacy.compiled.launch_schedule is None
    assert _graph_rows(scheduled) == _graph_rows(legacy)
    assert scheduled.estimate().cycles == legacy.estimate().cycles


def test_launch_schedule_binding_errors():
    trace = _virtual(list(_legacy_order.moves)).trace
    unknown = SimpleNamespace(
        steps=[LaunchRecord("mv1", ()), LaunchRecord("mv3", ()), LaunchRecord("mv2", ())]
    )
    with pytest.raises(ValueError, match="no matched implementation"):
        launch_schedule(unknown, trace)
    unlaunched = SimpleNamespace(steps=[LaunchRecord("mv1", ())])
    with pytest.raises(ValueError, match="never launched"):
        launch_schedule(unlaunched, trace)


# --------------------------------------------------------------------- #
# Samsung executes the launch schedule: one conductor run per launch
# (docs/release/design/samsung-execution.md section 9)
# --------------------------------------------------------------------- #

import numpy as np

from allo import spmw_samsung


@allo.host_program(_two_kernels)
def _relaunch_without_restage(W, V, x, y, z):
    _hx.scatter(W, _hx.banks)
    _hx.scatter(V, _hx.banks)
    _hx.broadcast(x, _hx.grf_a)
    allo.launch("mv1", W, x, y)
    _hx.gather(y, _hx.banks)
    _hx.broadcast(y, _hx.grf_a)
    allo.launch("mv2", V, y, z)
    _hx.gather(z, _hx.banks)
    _hx.broadcast(x, _hx.grf_a)
    allo.launch("mv1", W, x, y)  # W was staged for the first mv1 only
    _hx.gather(y, _hx.banks)


def _samsung(host_program, **kwargs):
    return allo.compile(
        _two_kernels,
        build_samsung_target(),
        samsung_cost,
        host_moves=host_program,
        **kwargs,
    )


def _f32(a):
    return np.asarray(a, dtype=np.float32)


def test_samsung_launch_plan_threads_gathered_results():
    """Each launch reads the stagings declared before it; the gathered y is
    the value broadcast to mv2; every host step is performed by an
    invocation (nothing modelled)."""
    plan = spmw_samsung.samsung_launch_plan(_samsung(_chained).compiled)
    assert [
        (
            launch["kernel"],
            {role: buffer for role, (buffer, _h) in launch["operands"].items()},
            launch["output"],
            launch["gathered_by"],
        )
        for launch in plan["launches"]
    ] == [
        ("mv1", {"A": "W", "B": "x"}, "y", [3]),
        ("mv2", {"A": "V", "B": "y"}, "z", [5]),
    ]
    assert [(s["verb"], s["buffer"], s["realized_by"]) for s in plan["host_steps"]] == [
        ("scatter", "W", {"launch": 0, "phase": "preload"}),
        ("scatter", "V", {"launch": 1, "phase": "preload"}),
        ("broadcast", "x", {"launch": 0, "phase": "exec"}),
        ("gather", "y", {"launch": 0, "phase": "readback"}),
        ("broadcast", "y", {"launch": 1, "phase": "exec"}),
        ("gather", "z", {"launch": 1, "phase": "readback"}),
    ]
    assert plan["external_inputs"] == ["W", "V", "x"]
    events = plan["events"]
    assert (
        events.index(("launch", 0))
        < events.index(("gather", 3, "y"))
        < events.index(("stage", 4, "y"))
        < events.index(("launch", 1))
    )


def test_samsung_legacy_order_rejects_device_resident_intermediate():
    """_legacy_order hands y from mv1 to mv2 without a gather/broadcast; the
    conductor keeps nothing across invocations, so the run is rejected
    (before any simulator call) naming that residency."""
    compiled = _samsung(_legacy_order)
    match = r"keeps 'y' device-resident across the two launches"
    with pytest.raises(ValueError, match=match):
        spmw_samsung.samsung_launch_plan(compiled.compiled)
    z = np.zeros(_LM, dtype=np.float16)
    with pytest.raises(ValueError, match=match):
        compiled(
            np.ones((_LM, _LK), np.float16),
            np.ones((_LM, _LK), np.float16),
            np.ones(_LK, np.float16),
            np.zeros(_LM, np.float16),
            z,
        )
    assert not z.any()


def test_samsung_reused_staging_is_rejected():
    """A staging read by one launch is not kept for a later launch."""
    with pytest.raises(ValueError, match=r"host step 0 staged it for launch 0"):
        spmw_samsung.samsung_launch_plan(_samsung(_relaunch_without_restage).compiled)


@pytest.mark.parametrize(
    "order, program",
    [("reassociate", "REDUCE"), ("strict", "TEMPORAL_REDUCE")],
)
def test_samsung_runs_chained_launches(samsung_sim_gate, record_property, order, program):
    """z = V @ (W @ x) through two conductor invocations on fresh random
    inputs.  ``strict`` (the default) selects the ordered program (FP16
    accumulation in source order); ``reassociate`` the lane-parallel REDUCE
    (FP16 lane partials, FP32 host lane sum).  Both are compared with the
    FP32 chain at FP16 tolerance; mv2 reads the FP16 y the host gathered."""
    compiled = _samsung(_chained, reduction_order=order)
    seed = int(np.random.SeedSequence().entropy % 2**32)
    rng = np.random.default_rng(seed)
    W = rng.uniform(-0.5, 0.5, (_LM, _LK)).astype(np.float16)
    V = rng.uniform(-0.5, 0.5, (_LM, _LK)).astype(np.float16)
    x = rng.uniform(-1.0, 1.0, _LK).astype(np.float16)
    y = np.zeros(_LM, dtype=np.float16)
    z = np.zeros(_LM, dtype=np.float16)

    result = compiled(W, V, x, y, z)

    assert result.kind == "simulator" and result.backend == "samsung_hbm_pim"
    launches = result.extra["launches"]
    assert [(l["kernel"], l["program"]) for l in launches] == [
        ("mv1", program),
        ("mv2", program),
    ]
    for launch in launches:
        assert launch["cycles"] == sum(launch["cycle_phases"].values())
        assert launch["timing_scope"]
    assert result.extra["cycle_phases"] == {
        "launches": sum(l["cycles"] for l in launches),
        "modelled_host_steps": 0,
    }
    assert result.cycles == sum(l["cycles"] for l in launches)
    assert set(result.extra["outputs"]) == {"y", "z"}
    record_property("samsung_two_kernel_cycles", [l["cycles"] for l in launches])

    y_ref = _f32(W) @ _f32(x)
    z_ref = _f32(V) @ y_ref
    tol_y = 1e-2 * np.abs(y_ref).max()
    tol_z = 1e-2 * np.abs(z_ref).max()
    assert np.abs(_f32(y) - y_ref).max() <= tol_y, f"seed {seed}"
    assert np.abs(_f32(z) - z_ref).max() <= tol_z, f"seed {seed}"
    # mv2 consumed the gathered FP16 y, not a host-side recomputation.
    assert np.abs(_f32(z) - _f32(V) @ _f32(y)).max() <= tol_z, f"seed {seed}"
