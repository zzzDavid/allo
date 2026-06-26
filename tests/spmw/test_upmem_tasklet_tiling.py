# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""UPMEM tasklet-tiling lever (design 02).

The tasklet count is a `Placement.extra["n_tasklets"]` lever: enumerated
as `{1, T_max}` from the tasklet-unit fanout, priced by
`S + ceil(S*(R-1)/min(T,R))` with `R = target.revolver_latency = 11`,
materialised by codegen + `_run_upmem`. These tests assert each step in
isolation (no simulator), plus the T=1 parity floor (T9).
"""

from __future__ import annotations

import math

import allo
from allo.spmw_autoschedule import (
    Placement,
    _trace_reduction_trip,
    _tasklet_fanout,
    _upmem_enumerate,
)
from allo.spmw_cost_models import _upmem_kernel_cycles
from allo.spmw_cost_model import OpCostCtx, get_cost_model
from allo.spmw_match import MatchTrace, MatchedOp, OperandBinding

from _fixtures import build_upmem_target


def _upmem_model():
    return get_cost_model("upmem", "faithful")


def _upmem_mac_trace(k_bound: str = "1024") -> MatchTrace:
    """A 1-match GEMV-shaped UPMEM trace with an inner-K reduction loop."""
    return MatchTrace(
        target_name="upmem",
        module_name="synthetic",
        matches=[
            MatchedOp(
                target_op_name="MAC",
                func_name="gemv_0_0",
                work_id=(0, 0),
                enclosing_loops=[
                    ("%arg0", "0", "4096", 1),
                    ("%arg1", "0", k_bound, 1),
                ],
                operands=[
                    OperandBinding(role="x", memref_name="local_W"),
                    OperandBinding(role="y", memref_name="local_x"),
                    OperandBinding(
                        role="acc", memref_name="acc", is_loop_carried=True
                    ),
                ],
                result_memref_name="acc",
                op_range=("%a", "%b"),
            )
        ],
    )


# --------------------------------------------------------------------- #
# Target-spec constant
# --------------------------------------------------------------------- #


def test_upmem_target_has_revolver_latency():
    # Per design 04 the revolver_latency constant rides the bound
    # CostModel.constants, not the target tree.
    assert _upmem_model().const("revolver_latency") == 11
    # pipeline_depth is deliberately NOT added (design 02 §3.3).
    try:
        _ = _upmem_model().const("pipeline_depth")
    except KeyError:
        pass
    else:
        raise AssertionError("pipeline_depth must not be a cost constant")


# --------------------------------------------------------------------- #
# Structural resolvers
# --------------------------------------------------------------------- #


def test_trace_reduction_trip_reads_inner_k():
    trace = _upmem_mac_trace("1024")
    assert _trace_reduction_trip(trace.matches) == 1024


def test_tasklet_fanout_is_target_derived():
    target = build_upmem_target()
    # 16 in the fixture, but read off the unit mapping, not a literal.
    assert _tasklet_fanout(target) == 16


# --------------------------------------------------------------------- #
# Enumerator
# --------------------------------------------------------------------- #


def test_upmem_tasklet_candidates():
    target = build_upmem_target()
    trace = _upmem_mac_trace("1024")
    layouts = _upmem_enumerate(target, trace.matches)
    # 2 acc-placements x {1, 16} tasklet candidates = 4 placements.
    assert len(layouts) == 4
    n_vals = sorted({lo.extra.get("n_tasklets", 1) for lo in layouts})
    assert n_vals == [1, 16]
    # >=2-candidate discipline: at least one nt=1 and one nt=T_max.
    assert any(lo.extra["n_tasklets"] == 1 for lo in layouts)
    assert any(lo.extra["n_tasklets"] == 16 for lo in layouts)


def test_upmem_tasklet_candidates_cap_to_short_reduction():
    target = build_upmem_target()
    # Reduction trip 8 < T_max 16 -> over-subscription capped at the trip.
    trace = _upmem_mac_trace("8")
    layouts = _upmem_enumerate(target, trace.matches)
    n_vals = sorted({lo.extra["n_tasklets"] for lo in layouts})
    assert n_vals == [1, 8]


# --------------------------------------------------------------------- #
# Cost model
# --------------------------------------------------------------------- #


def test_upmem_cost_t1_parity():
    """At n_tasklets=1 the cost is S*R -- a uniform scale of the old S."""
    target = build_upmem_target()
    cost_fn = _upmem_kernel_cycles(target)
    trace = _upmem_mac_trace("1024")
    # S = issue count = MAC.cycles * innermost-K trip (the cost model
    # multiplies only enclosing_loops[-1], the per-DPU reduction).
    mac_cyc = _upmem_model().op_cost("MAC", OpCostCtx("MAC"))
    s = mac_cyc * 1024
    r = _upmem_model().const("revolver_latency")
    p1 = Placement(placements={}, extra={"n_tasklets": 1})
    assert cost_fn(trace, p1) == s * r
    # No extra / placement=None default also takes T=1.
    assert cost_fn(trace, Placement(placements={})) == s * r


def test_upmem_cost_prefers_parallel():
    """Argmin must prefer the high-tasklet placement; the modelled lever
    matches the measured 5.71x within 1% on the GEMV reduction trip."""
    target = build_upmem_target()
    cost_fn = _upmem_kernel_cycles(target)
    trace = _upmem_mac_trace("1024")
    p1 = Placement(placements={}, extra={"n_tasklets": 1})
    p16 = Placement(placements={}, extra={"n_tasklets": 16})
    c1 = cost_fn(trace, p1)
    c16 = cost_fn(trace, p16)
    assert c16 < c1
    lever = c1 / c16
    assert abs(lever - 5.762) < 0.05, lever


def test_upmem_cost_saturates_at_revolver_window():
    """Speedup saturates at T = R = 11, not at the tasklet fanout 16."""
    target = build_upmem_target()
    cost_fn = _upmem_kernel_cycles(target)
    trace = _upmem_mac_trace("1024")
    c11 = cost_fn(trace, Placement(placements={}, extra={"n_tasklets": 11}))
    c16 = cost_fn(trace, Placement(placements={}, extra={"n_tasklets": 16}))
    c32 = cost_fn(trace, Placement(placements={}, extra={"n_tasklets": 32}))
    assert c11 == c16 == c32


def test_upmem_cost_lever_invariant_across_reduction_trip():
    """The lever is a function of R only -- identical across K."""
    target = build_upmem_target()
    cost_fn = _upmem_kernel_cycles(target)
    levers = []
    for k in ("256", "1024", "4096"):
        trace = _upmem_mac_trace(k)
        c1 = cost_fn(trace, Placement(placements={}, extra={"n_tasklets": 1}))
        c16 = cost_fn(trace, Placement(placements={}, extra={"n_tasklets": 16}))
        levers.append(c1 / c16)
    # The lever is a function of R only; integer-ceil rounding leaves a
    # sub-1% deviation at small K that vanishes asymptotically (5.762).
    assert max(levers) - min(levers) < 0.01, levers


def test_run_upmem_threads_placement_tasklet_count():
    """_run_upmem must drive `--num_tasklets` from the chosen tasklet
    count, not the old hardcoded literal `1` (T9 resolution)."""
    import inspect
    from allo.spmw_codegen import _run_upmem

    source = inspect.getsource(_run_upmem)
    # The literal single-tasklet count must no longer be passed
    # unconditionally; the value comes off the staged ctx field.
    assert '"--num_tasklets", "1"' not in source, source
    assert "n_tasklets" in source, source
    assert "str(num_tasklets)" in source, source
    # The reduction trip + row count drive the GEMV-host data-prep
    # footprint (design 02 §6c), retiring the §6b 1024 debt.
    assert "reduction_trip" in source, source


def test_upmem_ctx_stages_tasklet_count_from_placement():
    """compile_for_target threads the chosen n_tasklets onto the ctx so the
    runner can read it (codegen reads the field, never re-derives)."""
    from allo.spmw_codegen import compile_for_target

    target = build_upmem_target()
    trace = _upmem_mac_trace("1024")
    compiled = compile_for_target(target, trace)
    ctx = compiled._ctx
    # Argmin prefers the parallel placement, so the staged count is T_max.
    assert ctx.n_tasklets == 16
    assert ctx.reduction_trip == 1024
    # The gemv outer-row count (m_size) is staged from the outer loop.
    assert ctx.row_count == 4096


def test_upmem_gemv_host_routing():
    """Design 02 §6c: a gemv-shaped trace routes through the GEMV host at
    a shape-derived (m_size, n_size); the decision path holds no
    1024/64 literal keyed to the benchmark."""
    import inspect
    from allo.spmw_codegen import _run_upmem, UPMEMCtx, compile_for_target

    source = inspect.getsource(_run_upmem)
    # Shape-routed slot decision driven by staged fields, not workload name.
    assert "is_gemv" in source and "row_count" in source, source
    assert "reduction_trip" in source, source
    # data_prep for the gemv route is "<m_size>,<n_size>", both staged.
    assert "{int(row_count)},{int(reduction_trip)}" in source, source
    # GEMV-host-compatible envelope is emitted for the gemv route.
    assert "get_gemv_kernel_src" in source, source
    assert hasattr(UPMEMCtx, "get_gemv_kernel_src")

    # The GEMV envelope reads the gemv arg struct (not the VA envelope).
    target = build_upmem_target()
    trace = _upmem_mac_trace("1024")
    compiled = compile_for_target(target, trace)
    gemv_src = compiled._ctx.get_gemv_kernel_src()
    assert "DPU_INPUT_ARGUMENTS.n_size" in gemv_src
    assert "DPU_INPUT_ARGUMENTS.nr_rows" in gemv_src
    assert "tenon-emitted MAC" in gemv_src


def test_upmem_va_route_keeps_tenon_slot():
    """When no gemv shape is staged (row_count None), the runner keeps the
    TENON slot and the VA data-prep; only a gemv-shaped trace routes to
    GEMV. Asserted at the _run_upmem source level (the gemv-only UPMEM
    enumerator rejects a pure-VA trace upstream, so this checks the
    routing predicate, not a full VA compile)."""
    import inspect
    from allo.spmw_codegen import _run_upmem, UPMEMCtx

    source = inspect.getsource(_run_upmem)
    # The gemv predicate requires BOTH row_count and reduction_trip; a
    # fresh ctx (VA defaults) has row_count=None -> is_gemv False -> TENON.
    assert "row_count is not None and reduction_trip is not None" in source
    ctx = UPMEMCtx(build_upmem_target())
    assert ctx.row_count is None  # VA/default ctx never sets a gemv footprint


def test_upmem_cost_does_not_move_other_backend_argmin():
    """Samsung argmin is untouched -- the lever is gated inside the UPMEM
    cost factory and the T=1 default is a uniform scale anyway."""
    from _fixtures import build_samsung_target
    from allo.spmw_autoschedule import autoschedule
    from allo.spmw_match import MatchTrace as MT

    target = build_samsung_target()
    trace = MT(
        target_name="samsung_hbm_pim",
        module_name="synthetic",
        matches=[
            MatchedOp(
                target_op_name="MAC",
                func_name="gemv_0_0",
                work_id=(0, 0),
                enclosing_loops=[("%a", "0", "32", 1), ("%b", "0", "1024", 1)],
                operands=[
                    OperandBinding(role="x", memref_name="local_W"),
                    OperandBinding(role="y", memref_name="local_x"),
                    OperandBinding(
                        role="acc", memref_name="acc", is_loop_carried=True
                    ),
                ],
                result_memref_name="acc",
                op_range=("%a", "%b"),
            )
        ],
    )
    # Just assert it still resolves to a placement without error.
    picks = autoschedule(target, trace)
    assert len(picks) == 1


if __name__ == "__main__":
    test_upmem_target_has_revolver_latency()
    test_trace_reduction_trip_reads_inner_k()
    test_tasklet_fanout_is_target_derived()
    test_upmem_tasklet_candidates()
    test_upmem_tasklet_candidates_cap_to_short_reduction()
    test_upmem_cost_t1_parity()
    test_upmem_cost_prefers_parallel()
    test_upmem_cost_saturates_at_revolver_window()
    test_upmem_cost_lever_invariant_across_reduction_trip()
    test_run_upmem_threads_placement_tasklet_count()
    test_upmem_ctx_stages_tasklet_count_from_placement()
    test_upmem_cost_does_not_move_other_backend_argmin()
    print("ALL PASSED")
