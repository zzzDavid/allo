# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""CostModel swap-test (design 04 §1.4 / §5.3).

The decoupling property: a *second* cost flavor prices the *same*
structural target with a *different* estimate, and the device tree is not
edited to do it. This is the acid test that the per-op/per-move numbers
live on the CostModel, not on the target.
"""
from __future__ import annotations

import allo
from allo.spmw_cost_model import (
    ComposeCtx,
    MoveCostCtx,
    OpCostCtx,
    get_cost_model,
)
from allo.spmw_match import MatchedOp, MatchTrace, OperandBinding

from _fixtures import build_samsung_target


def _samsung_gemv_trace() -> MatchTrace:
    return MatchTrace(
        target_name="samsung_hbm_pim",
        module_name="synthetic",
        matches=[
            MatchedOp(
                target_op_name="MAC",
                func_name="gemv_0_0",
                work_id=(0, 0),
                enclosing_loops=[
                    ("%i", "0", "32", 1),
                    ("%k", "0", "1024", 1),
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


def test_two_flavors_bound_to_same_target():
    f = get_cost_model("samsung_hbm_pim", "faithful")
    o = get_cost_model("samsung_hbm_pim", "optimistic")
    assert f.target_name == o.target_name == "samsung_hbm_pim"
    assert f.name != o.name


def test_swap_changes_estimate_with_zero_device_edit():
    target = build_samsung_target()  # ONE device tree, never edited
    trace = _samsung_gemv_trace()
    layout = allo.Placement(placements={})

    f = get_cost_model("samsung_hbm_pim", "faithful")
    o = get_cost_model("samsung_hbm_pim", "optimistic")
    cyc_faithful = f.compose(ComposeCtx(target, trace, layout)).cycles
    cyc_optimistic = o.compose(ComposeCtx(target, trace, layout)).cycles

    # Different cost table -> different estimate; same target object.
    assert cyc_faithful != cyc_optimistic
    # Optimistic MAC (2 vs 4 cyc) prices the exec phase strictly cheaper.
    assert cyc_optimistic < cyc_faithful


def test_target_tree_carries_no_cost_after_swap():
    """The swap is possible precisely because the target carries no cost
    number -- assert that invariant directly."""
    t = build_samsung_target()
    for u in t._walk():
        for mv in u.moves.values():
            assert mv.cycles is None
        for op in u.ops.values():
            assert op.cycles is None


def test_per_op_cost_is_a_callable_not_a_scalar():
    """Layer A entries are CALLABLES of a context, not scalars -- the seam
    a per-op analytical model (e.g. APU v1 MICRO-2025) plugs into."""
    f = get_cost_model("samsung_hbm_pim", "faithful")
    mac = f.op_costs["MAC"]
    assert callable(mac.fn)
    # The full OpCostCtx is accepted even though the v1 entry ignores it.
    assert mac.fn(OpCostCtx("MAC", iters=99, lane_width=8)) == 4
    # Task-017: the preload fan-out width re-homed off the deleted
    # PRELOAD_FAN move onto the `host_staging` concern as STAGE_BCAST=369.
    hs = get_cost_model("samsung_hbm_pim", "faithful", concern="host_staging")
    fan = hs.move_costs["STAGE_BCAST"]
    assert fan.fn(MoveCostCtx("STAGE_BCAST")) == 369


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("STATIC PASSED")
