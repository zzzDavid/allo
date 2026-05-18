# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the SPMW register / buffer allocator (spec 015, Task 016)."""

from __future__ import annotations

import warnings

import allo
from allo.spmw_autoschedule import (
    Placement,
    _aim_enumerate,
    _apu_v1_enumerate,
    _apu_v2_enumerate,
    _samsung_enumerate,
    _upmem_enumerate,
    autoschedule,
)
from allo.spmw_match import MatchTrace, MatchedOp, OperandBinding
from allo.spmw_regalloc import (
    AllocResult,
    CostVector,
    LiveRange,
    Spilled,
    allocate,
    extract_live_ranges,
)
from allo.spmw_target import MemoryRef, Register

from _fixtures import (
    build_aim_target,
    build_apu_v1_target,
    build_apu_v2_target,
    build_samsung_target,
    build_upmem_target,
)


# --------------------------------------------------------------------- #
# Synthetic traces
# --------------------------------------------------------------------- #


def _mac_match(idx: int, w_name: str = "local_W") -> MatchedOp:
    return MatchedOp(
        target_op_name="MAC",
        func_name=f"gemv_{idx}",
        work_id=(idx,),
        enclosing_loops=[("%arg0", "0", "1024", 1)],
        operands=[
            OperandBinding(role="x", memref_name=w_name),
            OperandBinding(role="y", memref_name="local_x"),
            OperandBinding(
                role="acc", memref_name="acc", is_loop_carried=True
            ),
        ],
        result_memref_name="acc",
        op_range=("%a", "%b"),
    )


def _gemv_trace() -> MatchTrace:
    return MatchTrace(
        target_name="samsung_hbm_pim",
        module_name="synthetic",
        matches=[_mac_match(0)],
    )


# --------------------------------------------------------------------- #
# Live-range extraction
# --------------------------------------------------------------------- #


def test_extract_live_ranges_single_kernel():
    """GEMV: three live ranges (local_W, local_x, acc), sorted by first
    appearance and memref name.
    """
    lrs = extract_live_ranges(_gemv_trace().matches)
    names = [lr.memref_name for lr in lrs]
    assert set(names) == {"local_W", "local_x", "acc"}
    assert len(lrs) == 3


def test_extract_live_ranges_loop_carried_acc_extends():
    """A reduction's `acc` lives to the end of the group (last_idx ==
    len(matches) - 1) even when its operand only appears in the first
    match.
    """
    matches = [_mac_match(0), _mac_match(1), _mac_match(2)]
    lrs = extract_live_ranges(matches)
    acc = [lr for lr in lrs if lr.memref_name == "acc"][0]
    assert acc.last_idx == 2
    assert acc.is_loop_carried


# --------------------------------------------------------------------- #
# Capacity / no-spill cases (one per backend)
# --------------------------------------------------------------------- #


def test_samsung_no_spill_under_capacity():
    target = build_samsung_target()
    matches = _gemv_trace().matches
    cands = _samsung_enumerate(target, matches)
    result = allocate(target, matches, cands[0], all_candidates=cands)
    assert result.spilled == [], result.spilled
    assert all(
        not isinstance(h, Spilled) for h in result.placement.placements.values()
    )


def test_aim_no_spill_under_capacity():
    target = build_aim_target()
    matches = [
        MatchedOp(
            target_op_name="MAC",
            func_name="aim_0",
            work_id=(0,),
            enclosing_loops=[("%arg0", "0", "16", 1)],
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
    ]
    cands = _aim_enumerate(target, matches)
    result = allocate(target, matches, cands[0], all_candidates=cands)
    assert result.spilled == []


def test_upmem_no_spill_small_workload():
    """Two live ranges fit easily in WRAM + GPR. Spec 015 success
    criterion (b): allocator places small values without spilling.
    """
    target = build_upmem_target()
    matches = [
        MatchedOp(
            target_op_name="MAC",
            func_name="upmem_0",
            work_id=(0,),
            enclosing_loops=[("%arg0", "0", "16", 1)],
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
    ]
    cands = _upmem_enumerate(target, matches)
    result = allocate(target, matches, cands[0], all_candidates=cands)
    assert result.spilled == []


def test_apu_v1_no_spill_three_lrs():
    target = build_apu_v1_target()
    matches = _gemv_trace().matches
    cands = _apu_v1_enumerate(target, matches)
    result = allocate(target, matches, cands[0], all_candidates=cands)
    assert result.spilled == []


def test_apu_v2_runs_with_warning():
    """Placeholder cost factory fires a one-time RuntimeWarning; the
    allocator returns a placement without crashing.
    """
    target = build_apu_v2_target()
    matches = _gemv_trace().matches
    cands = _apu_v2_enumerate(target, matches)
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        result = allocate(target, matches, cands[0], all_candidates=cands)
    assert isinstance(result, AllocResult)
    assert "acc" in result.placement.placements


# --------------------------------------------------------------------- #
# Capacity overflow -> spill
# --------------------------------------------------------------------- #


def _many_grf_a_matches(n: int) -> list[MatchedOp]:
    """Synthesize one match referencing `n` distinct memrefs, each as a
    loop-carried `acc`-role operand so that all `n` live ranges have
    overlapping (full-group) lifetimes. With `n > 8`, the Samsung GRF_A
    capacity is exceeded.
    """
    operands = [
        OperandBinding(
            role=f"acc{i}",
            memref_name=f"v{i}",
            is_loop_carried=True,
        )
        for i in range(n)
    ]
    return [
        MatchedOp(
            target_op_name="MUL",
            func_name="overflow",
            work_id=(0,),
            enclosing_loops=[("%arg0", "0", "1024", 1)],
            operands=operands,
            result_memref_name="v0",
            op_range=("%a", "%b"),
        )
    ]


def test_samsung_grf_a_overflow_spills_to_bank():
    """Spec 015 success criterion (a) + spec §10 test 4: when more
    live values are bound to grf_a than its 8-entry capacity, the
    allocator spills the overflow to a bank row.

    We force every live range onto `grf_a` by handing the allocator a
    single candidate placement that pins `vN -> grf_a` for all
    memrefs; with 9 overlapping LRs and slots[grf_a]=8, exactly one
    must be Spilled.
    """
    target = build_samsung_target()
    matches = _many_grf_a_matches(9)
    # Build a candidate that pins every memref to grf_a.
    cand = Placement(
        placements={f"v{i}": target.grf_a for i in range(9)}
    )
    result = allocate(target, matches, cand, all_candidates=[cand])
    n_spilled = sum(
        1 for h in result.placement.placements.values()
        if isinstance(h, Spilled)
    )
    assert n_spilled == 1, (
        f"expected exactly 1 spill (8 slots, 9 live ranges); got "
        f"{n_spilled}: {result.placement.placements}"
    )
    # The cost of the spill must come from the Samsung register_spill
    # factory; assert total_cost > 0.
    assert result.total_cost > 0
    # And the spilled tier must be bank_row.
    for h in result.placement.placements.values():
        if isinstance(h, Spilled):
            assert h.tier == "bank_row"


def test_register_spill_factory_each_backend():
    """`get_cost("register_spill", target)` returns a callable for each
    of the five backends; calling with `n_entries=3` yields a positive
    int.
    """
    for builder in (
        build_samsung_target,
        build_aim_target,
        build_upmem_target,
        build_apu_v1_target,
    ):
        target = builder()
        spill_cb = allo.get_cost("register_spill", target)
        # Pick a register handle that exists on this target.
        if target.name == "samsung_hbm_pim":
            reg = target.grf_a
        elif target.name == "aim":
            reg = target.gpr
        elif target.name == "upmem":
            reg = target.gprs
        else:
            reg = target.vrs
        cost = spill_cb(reg, n_entries=3)
        assert isinstance(cost, int) and cost > 0, (
            f"{target.name}: spill_cb returned {cost!r}"
        )

    # APU v2: warns once, returns 6 (= 2 * 3).
    target = build_apu_v2_target()
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        spill_cb = allo.get_cost("register_spill", target)
    cost = spill_cb(None, n_entries=3)
    assert cost == 6


# --------------------------------------------------------------------- #
# Autoschedule integration: cycle counts unchanged for canonical GEMV.
# --------------------------------------------------------------------- #


def test_autoschedule_with_regalloc_picks_canonical_samsung():
    """Existing test `test_autoschedule_picks_is_auto_layout` invariant
    holds with regalloc enabled: the picked layout is is_auto and no
    live range spilled.
    """
    target = build_samsung_target()
    trace = MatchTrace(
        target_name="samsung_hbm_pim",
        module_name="synthetic",
        matches=[_mac_match(0)],
    )
    layouts = autoschedule(target, trace)
    assert len(layouts) == 1
    assert isinstance(layouts[0].placements["local_x"], MemoryRef)
    # No Spilled wrappers in the final placement.
    for h in layouts[0].placements.values():
        assert not isinstance(h, Spilled), h


def test_costvector_gap_orders_by_constraint():
    """Smoke check: CostVector.gap() returns cheapest_alt - cheapest,
    which the greedy uses to pick "most constrained first".
    """
    cv = CostVector(entries={"a": 10, "b": 100, "c": 1000})
    cheapest, _ = cv.cheapest()
    assert cheapest == "a"
    assert cv.gap() == 90  # 100 - 10


def test_kill_switch_disables_regalloc(monkeypatch):
    """`SPMW_DISABLE_REGALLOC=1` skips the allocator; autoschedule
    behaves like spec 012a (kernel_cycles argmin only).
    """
    monkeypatch.setenv("SPMW_DISABLE_REGALLOC", "1")
    target = build_samsung_target()
    trace = MatchTrace(
        target_name="samsung_hbm_pim",
        module_name="synthetic",
        matches=[_mac_match(0)],
    )
    layouts = autoschedule(target, trace)
    assert len(layouts) == 1
    # Layout still valid; just no allocator second-stage.
    assert isinstance(layouts[0].placements["local_x"], MemoryRef)


if __name__ == "__main__":
    test_extract_live_ranges_single_kernel()
    test_extract_live_ranges_loop_carried_acc_extends()
    test_samsung_no_spill_under_capacity()
    test_aim_no_spill_under_capacity()
    test_upmem_no_spill_small_workload()
    test_apu_v1_no_spill_three_lrs()
    test_apu_v2_runs_with_warning()
    test_samsung_grf_a_overflow_spills_to_bank()
    test_register_spill_factory_each_backend()
    test_autoschedule_with_regalloc_picks_canonical_samsung()
    test_costvector_gap_orders_by_constraint()
    print("ALL PASSED")
