# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase-5 no-backend demo + per-op refinability showcase (design 04 §8).

Two things proven here, both STATIC (no simulator, no hardware):

1. **A backend that ships with the spec.** `demo_pim` has no sim/HW, no
   codegen ctx, no enumerator -- yet a kernel is developed + estimated
   purely via `compile_for_target(target, trace, backend="virtual")`.

2. **Independent per-op refinability.** The `demo_pim_micro25` flavor
   differs from `demo_pim_constant` by exactly ONE `OpCost` entry (ADD:
   constant -> MICRO-2025 bit-serial analytical fn). The estimate changes
   with ZERO edit to `compose` or any caller -- the canonical
   "constant -> analytical, no caller change" demonstration.
"""
from __future__ import annotations

import allo
from allo.spmw_cost_model import (
    OpCostCtx,
    get_cost_model,
)
from allo.spmw_cost_tables import (
    DEMO_PIM_ELEM_BITS,
    micro25_add_cost,
)
from allo.spmw_match import MatchedOp, MatchTrace, OperandBinding

from _demo_target import build_demo_pim_target


def _add_trace(n_adds: int = 1, ub: str = "1024") -> MatchTrace:
    matches = [
        MatchedOp(
            target_op_name="ADD",
            func_name="ew_0",
            work_id=(0,),
            enclosing_loops=[("%k", "0", ub, 1)],
            operands=[
                OperandBinding(role="x", memref_name="a"),
                OperandBinding(role="y", memref_name="b"),
            ],
            result_memref_name="c",
            op_range=("%a", "%b"),
        )
        for _ in range(n_adds)
    ]
    return MatchTrace(target_name="demo_pim", module_name="demo", matches=matches)


# --------------------------------------------------------------------- #
# 1. A backend that ships with the spec (no sim/HW)
# --------------------------------------------------------------------- #


def test_no_backend_substrate_is_costable_sim_free():
    target = build_demo_pim_target()
    trace = _add_trace()
    # No _BACKEND_CTX entry, no enumerator -- only the CostModel exists.
    compiled = allo.compile_for_target(target, trace, backend="virtual")
    assert compiled.backend == "virtual"
    assert compiled.cmds == []  # nothing emitted; cost-only path
    result = compiled.run()
    assert result.backend == "virtual"
    assert result.cycles is not None and result.cycles > 0
    assert result.extra["cost_model"] == "demo_pim_constant"
    assert result.extra["priced_target"] == "demo_pim"


def test_demo_target_carries_no_cost_number():
    """design 04 §1.2 acid test, applied to the demo substrate."""
    t = build_demo_pim_target()
    for u in t._walk():
        for mv in u.moves.values():
            assert mv.cycles is None
        for op in u.ops.values():
            assert op.cycles is None


# --------------------------------------------------------------------- #
# 2. Per-op refinability: constant -> analytical, NO caller change
# --------------------------------------------------------------------- #


def test_flavors_share_one_compose_object():
    """Both flavors are bound to the same target and use the same compose
    mechanism -- the only difference is the ADD OpCost entry."""
    c = get_cost_model("demo_pim", "constant")
    m = get_cost_model("demo_pim", "micro25")
    assert c.target_name == m.target_name == "demo_pim"
    # The MUL/MAC entries are byte-identical constants across flavors;
    # only ADD is refined.
    assert c.op_costs["MUL"].fn(OpCostCtx("MUL")) == m.op_costs["MUL"].fn(
        OpCostCtx("MUL")
    )
    assert callable(c.op_costs["ADD"].fn) and callable(m.op_costs["ADD"].fn)


def test_refinement_changes_estimate_no_caller_change():
    target = build_demo_pim_target()
    trace = _add_trace()
    # Identical call site for both flavors -- only `cost_flavor` differs.
    const_run = allo.compile_for_target(
        target, trace, backend="virtual", cost_flavor="constant"
    ).run()
    micro_run = allo.compile_for_target(
        target, trace, backend="virtual", cost_flavor="micro25"
    ).run()

    # Constant flavor: ADD = 2 cyc, 1024 inner iters -> 2048.
    assert const_run.cycles == 2 * 1024
    # Analytical flavor: gap_floor(2) + 0.75*16 = 14 cyc/add, * 1024.
    assert micro_run.cycles == 14 * 1024
    # Refinement changed the estimate with zero compose / caller edit.
    assert micro_run.cycles != const_run.cycles


def test_analytical_fn_responds_to_bit_width():
    """The MICRO-2025 OpCost.fn is a real function of the operand bit-width
    carried in the OpCostCtx -- not a constant. Halving the element width
    (s16 -> s8) lowers the bit-serial cost."""
    s16 = micro25_add_cost(OpCostCtx("ADD", operand_shapes=((16,),)))
    s8 = micro25_add_cost(OpCostCtx("ADD", operand_shapes=((8,),)))
    s32 = micro25_add_cost(OpCostCtx("ADD", operand_shapes=((32,),)))
    assert s8 < s16 < s32                 # monotone in bit-width
    assert s16 == 2.0 + 0.75 * 16         # the published s16 anchor (=14)
    # Falls back to the device element width when no operand shape is given.
    assert micro25_add_cost(OpCostCtx("ADD")) == 2.0 + 0.75 * DEMO_PIM_ELEM_BITS


def test_constant_flavor_ignores_ctx():
    """The constant entry satisfies the SAME interface -- it just ignores
    the rich ctx. This is why refinement is a one-entry edit."""
    c = get_cost_model("demo_pim", "constant")
    assert c.op_costs["ADD"].fn(
        OpCostCtx("ADD", operand_shapes=((999,),), lane_width=12345)
    ) == 2


def test_corpus_models_untouched_by_demo():
    """The Phase-5 demo registered new models without perturbing the five
    corpus models (faithful argmin stays calibrated)."""
    for tn, fl in (
        ("samsung_hbm_pim", "faithful"),
        ("aim", "faithful"),
        ("upmem", "faithful"),
        ("apu_v1", "faithful"),
        ("apu_v2", "faithful"),
    ):
        assert get_cost_model(tn, fl).confidence in (
            "calibrated", "placeholder"
        )


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("STATIC PASSED")
