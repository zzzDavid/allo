# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lever-1 (SPEC-023) dual-fiber bank placement evidence.

Three positive-evidence checks the anti-hardcoding audit (task 100) cites:

  (a) the Samsung enumerator emits a `dual_fiber` candidate whose
      `extra["fibers"]` are EVEN then ODD by `_bank_fiber_class` — the
      layout-algebra receipt (even/odd indices fall out of the swizzle
      column, not a pasted 2*pid / 2*pid+1);
  (b) the cost model ranks `dual_fiber` < `bank_row` at K=1024, and the
      ranking *flips* when JUMP cycles are bumped high — proving the win
      comes from the modelled cost, not from enumeration order;
  (c) the per-fiber trip-count split follows the SPEC-023 §5.2 formula
      for two shapes including a non-power-of-two K.
"""

from __future__ import annotations

import allo
from allo.spmw_autoschedule import _samsung_enumerate
from allo.spmw_codegen import _bank_fiber_class, _fiber_fold_split
from allo.spmw_match import MatchTrace, MatchedOp, OperandBinding

from _fixtures import build_samsung_target


def _mac_trace(k: int) -> MatchTrace:
    """A 1-match GEMV-shaped trace with inner-K bound `k`."""
    return MatchTrace(
        target_name="samsung_hbm_pim",
        module_name="synthetic",
        matches=[
            MatchedOp(
                target_op_name="MAC",
                func_name="gemv_0_0",
                work_id=(0, 0),
                enclosing_loops=[
                    ("%arg0", "0", "32", 1),
                    ("%arg1", "0", str(k), 1),
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


def _by_mode(candidates):
    """Index candidates by the base `mode` token (lever 3 / SPEC-025
    appends a `+crf_*` suffix). Keep the per_workid CRF-issue variant so
    these lever-1 receipts compare body costs under a uniform x n_workids
    scale (the dual-fiber-vs-bank-row ranking is preserved)."""
    out = {}
    for c in candidates:
        base = c.mode.split("+", 1)[0]
        if c.extra.get("crf_issue", "per_workid") != "per_workid":
            continue
        out.setdefault(base, c)
    return out


# --------------------------------------------------------------------- #
# (a) Layout-algebra receipt: even/odd fibers out of the swizzle column
# --------------------------------------------------------------------- #


def test_enumerator_emits_dual_fiber_even_then_odd():
    target = build_samsung_target()
    candidates = _samsung_enumerate(target, _mac_trace(1024).matches)
    by_mode = _by_mode(candidates)

    # bank_row / grf_staged retained unchanged; dual_fiber added.
    assert {"bank_row", "grf_staged", "dual_fiber"} <= set(by_mode), by_mode

    dual = by_mode["dual_fiber"]
    fibers = dual.extra["fibers"]
    # Fiber count comes from the layout's tile-axis size (== 2), not a
    # literal pasted into the enumerator.
    assert dual.extra["n_fibers"] == len(fibers) == 2

    # The receipt: each fiber's idx is the swizzled-layout output at that
    # tile value. _bank_fiber_class classifies fiber 0 as EVEN (2*pid) and
    # fiber 1 as ODD (2*pid + 1).
    assert _bank_fiber_class(fibers[0].idx) == "EVEN_BANK"
    assert _bank_fiber_class(fibers[-1].idx) == "ODD_BANK"

    # The default y placement is the EVEN fiber, so any consumer that
    # ignores `extra` degrades to bank_row.
    assert dual.placements["local_x"] is fibers[0]


# --------------------------------------------------------------------- #
# (b) Cost drives the choice: dual_fiber < bank_row, flips on JUMP bump
# --------------------------------------------------------------------- #


def test_cost_ranks_dual_fiber_below_bank_row():
    target = build_samsung_target()
    trace = _mac_trace(1024)
    cost_fn = allo.get_cost("kernel_cycles", target)
    by_mode = _by_mode(_samsung_enumerate(target, trace.matches))

    bank_row_cost = cost_fn(trace, by_mode["bank_row"])
    dual_cost = cost_fn(trace, by_mode["dual_fiber"])
    assert dual_cost < bank_row_cost, (dual_cost, bank_row_cost)


def test_cost_ranking_flips_when_jump_is_expensive():
    """Make JUMP dominate: dual_fiber pays one extra JUMP per fiber, so a
    huge JUMP cost erases the ~2x MAC saving and bank_row wins again. The
    flip proves the ranking is *computed*, not hand-picked."""
    # Fresh target so get_cost's per-target cache doesn't return a stale
    # closure built against the original JUMP cycles.
    target = build_samsung_target()
    folded = 1024 // target.grf_a.lanes  # = 128
    # Pick a JUMP cost large enough that the extra fiber's JUMP outweighs
    # the MAC cycles it saves: saving ~= (folded/2)*mac_cyc, extra cost =
    # 1 * jump_cyc. Setting jump_cyc above the saving guarantees the flip.
    # Per design 04 the JUMP/MAC costs live on the bound CostModel, not the
    # target tree; perturb the model's move_costs (and restore) to make the
    # JUMP dominant.
    from allo.spmw_cost_model import MoveCost, OpCostCtx, get_cost_model

    model = get_cost_model("samsung_hbm_pim", "faithful")
    mac_cyc = model.op_cost("MAC", OpCostCtx("MAC"))
    saved_jump = model.move_costs["JUMP"]
    model.move_costs["JUMP"] = MoveCost(lambda c, _v=folded * mac_cyc: _v)
    try:
        trace = _mac_trace(1024)
        cost_fn = allo.get_cost("kernel_cycles", target)
        by_mode = _by_mode(_samsung_enumerate(target, trace.matches))

        bank_row_cost = cost_fn(trace, by_mode["bank_row"])
        dual_cost = cost_fn(trace, by_mode["dual_fiber"])
        assert dual_cost > bank_row_cost, (dual_cost, bank_row_cost)
    finally:
        model.move_costs["JUMP"] = saved_jump


# --------------------------------------------------------------------- #
# (c) Per-fiber trip-count split — the SPEC-023 §5.2 formula
# --------------------------------------------------------------------- #


def test_fiber_fold_split_even_power_of_two():
    # K=1024, lanes=8 -> folded=128, n=2 -> [64, 64]; JUMP counter 63 each.
    folded, n = 1024 // 8, 2
    split = _fiber_fold_split(folded, n)
    assert split == [64, 64]
    assert [t - 1 for t in split] == [63, 63]  # the '63' emerges, not written


def test_fiber_fold_split_non_power_of_two_k():
    # K=768, lanes=8 -> folded=96, n=2 -> [48, 48]; JUMP counter 47 each.
    folded, n = 768 // 8, 2
    split = _fiber_fold_split(folded, n)
    assert split == [48, 48]
    assert [t - 1 for t in split] == [47, 47]


def test_fiber_fold_split_odd_fold_ceils_busier_fiber():
    # An odd burst-fold count: folded=127, n=2 -> [64, 63] (ceil bounds
    # the busier fiber, matching the cost model's ceil per_fiber).
    split = _fiber_fold_split(127, 2)
    assert split == [64, 63]
    assert sum(split) == 127


if __name__ == "__main__":
    test_enumerator_emits_dual_fiber_even_then_odd()
    test_cost_ranks_dual_fiber_below_bank_row()
    test_cost_ranking_flips_when_jump_is_expensive()
    test_fiber_fold_split_even_power_of_two()
    test_fiber_fold_split_non_power_of_two_k()
    test_fiber_fold_split_odd_fold_ceils_busier_fiber()
    print("ALL PASSED")
