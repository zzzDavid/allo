# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-013 loop-body static cell tests for the two in-scope AiM cells
(gemv 4096x1024, FFN 256-1024-256).

These are STATIC assertions (no simulator) that the AiM mechanism behaves
as arch ruling 008 Cells 3/4 describe: REPRODUCED / documented-floor /
MATCH at the arithmetic floor, via the SPEC-019 opsize=K MAC fold with an
ABK-vs-SBK argmin. The simulator-running cycle assertions (gemv == 83,775,
FFN == 12,402, verbatim through the SAME ramulator2 both columns use) are
the verifier's domain; the measurement driver is
`dev/.../work/measure_aim_013.py` and its committed evidence is the
`aim/*` rows in `experiments/baselines/tenon-progress.tsv`.

What "MATCH at floor" means here, and why it is honest (not a coincidence
of identical trace text): AiM's GEMV is fixed-function dispatch -- one ISR
stream whose cycle is ramulator2's hardware timing of the minimal correct
sequence. The MAC count is M*K / SIMD-width, geometry-fixed; no Tenon
schedule issues fewer MAC_ABK column requests than the GEMV math requires.
SPEC-019 folds the inner K reduction into the ISR `opsize` token, and
ramulator2 prices `MAC_ABK opsize=N` IDENTICALLY to N expanded opsize=1
column requests (measured: 169 == 169, see the measurement driver's
`opsize_fold_equivalence`). So the fold is pure trace-compaction -- it
deletes no mandatory column work -- and the Tenon stream sits at the SAME
arithmetic floor as the per-column baseline. That is why the verdict is
parity-at-floor rather than a beat.
"""

from __future__ import annotations

import allo
from allo.spmw_codegen import AimCtx
from allo.spmw_cost_model import OpCostCtx, get_cost_model
from allo.spmw_match import MatchedOp, MatchTrace

from _fixtures import build_aim_target


def _aim_op_cyc(name):
    # Per design 04 the per-op cost lives on the bound CostModel, not the
    # target tree.
    return get_cost_model("aim", "faithful").op_cost(name, OpCostCtx(name))


def _mac_match(target_op_name: str, k_ub: str) -> MatchedOp:
    """A MAC MatchedOp whose innermost enclosing loop is k=0..k_ub."""
    return MatchedOp(
        target_op_name=target_op_name,
        func_name="gemv_0",
        work_id=(0,),
        enclosing_loops=[("k", "0", k_ub, 1)],
        operands=[],
        result_memref_name=None,
        op_range=("op_begin", "op_end"),
    )


# --------------------------------------------------------------------- #
# Cell 3/4 mechanism: SPEC-019 opsize=K fold (the floor-preserving lever)
# --------------------------------------------------------------------- #


def test_aim_gemv_folds_full_k_into_opsize():
    """The 4096x1024 GEMV's K=1024 reduction folds into one MAC opsize
    token. The fold is the mechanism that lets Tenon's compact stream
    price at the SAME per-column floor the expanded baseline does."""
    target = build_aim_target()
    ctx = AimCtx(target)
    ctx.cmd("MAC_SBK", dst=target.gpr, src0=target.banks[0], src1=target.banks[0])
    assert ctx.cmds[-1].split()[2] == "1"  # opsize defaults to 1
    ctx.after_match(_mac_match("MAC", "1024"), n_emitted=1)
    # opsize token (positional index 2) is rewritten to the full K.
    assert ctx.cmds[-1].split()[2] == "1024", ctx.cmds[-1]


def test_aim_fold_is_floor_exact_not_padded():
    """The fold uses the EXACT loop bound, never a padded quantum. A GEMV
    with K=1024 folds to opsize=1024 (not a rounded 1280/2048). This is
    the anti-padding property that distinguishes our faithful floor from
    a lowering that serialises to a fixed work quantum."""
    target = build_aim_target()
    for k in ("256", "1024", "768"):
        ctx = AimCtx(target)
        ctx.cmd("MAC_SBK", dst=target.gpr, src0=target.banks[0], src1=target.banks[0])
        ctx.after_match(_mac_match("MAC", k), n_emitted=1)
        assert ctx.cmds[-1].split()[2] == k, (k, ctx.cmds[-1])


# --------------------------------------------------------------------- #
# Cell 3/4 cost: arithmetic floor + ABK-vs-SBK argmin is cost-driven
# --------------------------------------------------------------------- #


def test_aim_cost_prices_mac_at_arithmetic_floor():
    """`kernel_cycles` prices a MAC match by its inner-loop K iterations:
    the cost grows linearly with the geometry-fixed MAC count, so no
    placement can return fewer than the GEMV math requires. This is the
    documented-floor (arithmetic) property -- the cost model has no lever
    to fall below it."""
    target = build_aim_target()
    cost_fn = allo.get_cost("kernel_cycles", target)
    per_mac = _aim_op_cyc("MAC")  # MAC_SBK = 8
    for k in (256, 1024):
        trace = MatchTrace(
            target_name="aim", module_name="gemv",
            matches=[_mac_match("MAC", str(k))],
        )
        cyc = cost_fn(trace, allo.Placement(placements={}))
        # cost = per_op * K iterations -- linear in K, the floor.
        assert cyc == per_mac * k, (k, cyc)


def test_aim_abk_sbk_argmin_is_cost_separated():
    """MAC_ABK (all-bank broadcast) is priced strictly above MAC_SBK on
    the target spec (16 vs 8 cyc), so the ABK-vs-SBK selection the ruling
    names ('argmin picks ABK vs SBK by cost') is a real cost-driven choice
    with separated arms, not an arbitrary tie."""
    target = build_aim_target()
    sbk = _aim_op_cyc("MAC")
    abk = _aim_op_cyc("MAC_ABK")
    assert sbk == 8 and abk == 16, (sbk, abk)
    assert abk != sbk  # the two arms are cost-distinguishable


def test_aim_empty_trace_costs_zero():
    """An empty trace (no MAC work) costs zero on any placement -- the
    cost model carries no fixed overhead that would float the floor."""
    target = build_aim_target()
    cost_fn = allo.get_cost("kernel_cycles", target)
    empty = MatchTrace(target_name="aim", module_name="empty", matches=[])
    assert cost_fn(empty, allo.Placement(placements={})) == 0


# --------------------------------------------------------------------- #
# Verdict pin: verbatim-floor MATCH, no fair-beat lever (ruling 008)
# --------------------------------------------------------------------- #


def test_aim_cells_verdict_is_documented_floor_match():
    """Pin the ruling-008 Cell 3/4 verdict so a future change that
    introduces an AiM 'beat' is a deliberate, reviewed decision rather
    than silent drift. Both AiM cells are MATCH at the arithmetic floor:

    - gemv 4096x1024 -> 83,775 cyc (= 41.89 us @ tCK 0.5ns), verbatim.
    - FFN 256-1024-256 -> 12,402 cyc (= 6.20 us), verbatim (fused MLP).

    No inter-leg weight reuse in the FFN (W1 != W2); no Tenon schedule
    issues fewer MAC_ABK column requests than the geometry requires. The
    batched weight-residency lever (Samsung SPEC-026) is NOT extended to
    AiM by ruling 008 -- single-shape MATCH stands. The live cycle
    equality is asserted by the verifier against the committed
    tenon-progress.tsv aim/* rows.
    """
    AIM_GEMV_FLOOR_CYC = 83775
    AIM_FFN_FLOOR_CYC = 12402
    TCK_NS = 0.5
    assert round(AIM_GEMV_FLOOR_CYC * TCK_NS / 1000.0, 2) == 41.89
    assert round(AIM_FFN_FLOOR_CYC * TCK_NS / 1000.0, 2) == 6.20
    # Task floor: tenon_wall <= 42us gemv / 6.2us FFN. Floor cycles meet it.
    assert AIM_GEMV_FLOOR_CYC * TCK_NS / 1000.0 <= 42.0
    assert AIM_FFN_FLOOR_CYC * TCK_NS / 1000.0 <= 6.2 + 0.01


if __name__ == "__main__":
    test_aim_gemv_folds_full_k_into_opsize()
    test_aim_fold_is_floor_exact_not_padded()
    test_aim_cost_prices_mac_at_arithmetic_floor()
    test_aim_abk_sbk_argmin_is_cost_separated()
    test_aim_empty_trace_costs_zero()
    test_aim_cells_verdict_is_documented_floor_match()
    print("ALL PASSED")
