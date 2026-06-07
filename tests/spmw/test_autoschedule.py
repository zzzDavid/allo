# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Autoscheduler + cost-model tests.

Exercises:
- The Samsung enumerator produces a sensible candidate set under MAC's
  `dst=grf_b` constraint.
- `kernel_cycles` ranks is_auto-enabled layouts ahead of the unrolled
  alternatives by ~8x.
- `compile_for_target` with no explicit layout autoschedules a layout
  whose emitted bytes match the canonical Samsung GEMV inner loop.
"""

from __future__ import annotations

import allo
from allo.dataflow import region as _df_region
from allo.ir.types import float32 as fp16
from allo.spmw_autoschedule import (
    _apu_v1_enumerate,
    _apu_v2_enumerate,
    _samsung_enumerate,
    autoschedule,
)
from allo.spmw_codegen import PIMCmd
from allo.spmw_match import MatchTrace, MatchedOp, OperandBinding
from allo.spmw_target import MemoryRef, Register, UnitId

from _fixtures import build_apu_v1_target, build_apu_v2_target, build_samsung_target


M, K = 4096, 1024
ROWS = M // (16 * 8)


@_df_region()
def gemv_top(W: fp16[M, K], x: fp16[K], y: fp16[M]):
    @allo.work(mapping=[16, 8], args=[W, x, y])
    def gemv(local_W: fp16[M, K], local_x: fp16[K], local_y: fp16[M]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * ROWS
        for i in range(ROWS):
            acc: fp16 = 0
            for k in range(K):
                acc += local_W[row0 + i, k] * local_x[k]
            local_y[row0 + i] = acc


def _synthetic_mac_trace() -> MatchTrace:
    """A 1-match GEMV-shaped trace, used by tests that don't need to
    spin up the full MLIR pipeline (~3 minutes).
    """
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
                    ("%arg1", "0", "1024", 1),
                ],
                operands=[
                    OperandBinding(role="x", memref_name="local_W"),
                    OperandBinding(role="y", memref_name="local_x"),
                    OperandBinding(role="acc", memref_name="acc", is_loop_carried=True),
                ],
                result_memref_name="acc",
                op_range=("%a", "%b"),
            )
        ],
    )


# --------------------------------------------------------------------- #
# Enumerator
# --------------------------------------------------------------------- #


def test_samsung_enumerator_respects_mac_dst_grf_b():
    target = build_samsung_target()
    trace = _synthetic_mac_trace()

    candidates = _samsung_enumerate(target, trace.matches)
    assert candidates, "expected at least one candidate layout"

    # Every candidate must place `acc` on grf_b (MAC's dst constraint).
    for layout in candidates:
        acc_handle = layout.placements["acc"]
        assert isinstance(acc_handle, Register) and acc_handle.name == "grf_b"

    # The bank-row candidate must place x and y on distinct resources
    # (grf_a vs a bank handle). SPEC-009 §1's second candidate
    # deliberately stages y on grf_a too, so the "distinct" invariant
    # only applies to the bank-row layout.
    bank_row = [c for c in candidates if c.mode == "bank_row"]
    assert bank_row, "samsung enumerator must produce a bank_row candidate"
    assert (
        bank_row[0].placements["local_W"] is not bank_row[0].placements["local_x"]
    )


# --------------------------------------------------------------------- #
# Cost model
# --------------------------------------------------------------------- #


def test_kernel_cycles_prefers_is_auto():
    """A layout with y=bank (is_auto=1) must cost ~8x less than a
    layout with y=grf (is_auto=0). The factor is the GRF lane burst
    size — every 8 K-iterations fold into one MAC.
    """
    target = build_samsung_target()
    cost_fn = allo.get_cost("kernel_cycles", target)
    trace = _synthetic_mac_trace()

    pid = UnitId(level=1, unit=None)
    fast = allo.Placement(
        placements={
            "local_W": target.grf_a,
            "local_x": target.banks[2 * pid],
            "acc": target.grf_b,
        }
    )
    slow = allo.Placement(
        placements={
            "local_W": target.banks[2 * pid],
            "local_x": target.grf_a,
            "acc": target.grf_b,
        }
    )

    fast_cost = cost_fn(trace, fast)
    slow_cost = cost_fn(trace, slow)
    assert fast_cost < slow_cost
    # K=1024, lane burst = 8 — folded form is 128 MACs (× 4 cyc) + 1 JUMP cyc;
    # unrolled form is 1024 MACs × 4 cyc.
    assert fast_cost == 128 * 4 + 1
    assert slow_cost == 1024 * 4


# --------------------------------------------------------------------- #
# End-to-end: autoschedule -> compile -> canonical bytes
# --------------------------------------------------------------------- #


def _canonical_mac() -> PIMCmd:
    return PIMCmd(
        type_="MAC",
        dst_="GRF_B",
        src0_="GRF_A",
        src1_="EVEN_BANK",
        isAuto_=1,
    )


def _canonical_jump() -> PIMCmd:
    return PIMCmd(type_="JUMP", loopCounter_=K // 8 - 1, loopOffset_=2)


def test_autoschedule_picks_is_auto_layout():
    """`autoschedule` on the synthetic GEMV trace must return a layout
    with y bound to a bank — otherwise the picked layout isn't
    is_auto-enabled and would cost ~8x more.
    """
    target = build_samsung_target()
    trace = _synthetic_mac_trace()
    layouts = autoschedule(target, trace)
    assert len(layouts) == 1
    assert isinstance(layouts[0].placements["local_x"], MemoryRef), layouts[0].placements


def test_compile_with_autoschedule_emits_canonical_bytes():
    """Driving `compile_for_target` with no explicit layout must produce
    the lever-1 dual-fiber Samsung GEMV inner loop — argmin now picks the
    dual-fiber placement (both bank halves busy), so each work-id emits a
    MAC against EVEN_BANK *and* a MAC against ODD_BANK, each followed by a
    per-fiber split JUMP. The bank parity falls out of the fiber handle's
    idx via SamsungCtx's `_bank_parity`.

    The per-fiber JUMP trip count is the K reduction split across the two
    fibers: `K // lanes // n_fibers - 1` (= 63 for K=1024, lanes=8,
    n_fibers=2) — derived, not written.
    """
    target = build_samsung_target()
    schedule = allo.customize(gemv_top, enable_tensor=False)
    trace = allo.match_workload(target, schedule.module)

    compiled = allo.compile_for_target(target, trace)

    lanes, n_fibers = 8, 2
    split_jump = PIMCmd(
        type_="JUMP", loopCounter_=K // lanes // n_fibers - 1, loopOffset_=2
    )
    even_mac_jump = 0
    odd_mac_jump = 0
    for i in range(len(compiled.cmds) - 1):
        cur, nxt = compiled.cmds[i], compiled.cmds[i + 1]
        if cur.type_ == "MAC" and nxt == split_jump:
            assert cur.dst_ == "GRF_B"
            assert cur.src0_ == "GRF_A"
            assert cur.isAuto_ == 1
            if cur.src1_ == "EVEN_BANK":
                even_mac_jump += 1
            elif cur.src1_ == "ODD_BANK":
                odd_mac_jump += 1
            else:
                raise AssertionError(cur.src1_)
    assert even_mac_jump == 16 * 8, even_mac_jump
    assert odd_mac_jump == 16 * 8, odd_mac_jump


# --------------------------------------------------------------------- #
# SPEC-009: argmin sees >=2 candidates per backend
# --------------------------------------------------------------------- #


def _apu_synthetic_mac_trace(target_name: str) -> MatchTrace:
    """Per-backend 1-match MAC trace with the local_W/local_x/acc role
    triple the enumerators expect. Independent of the Samsung-shaped
    fixture so role names line up with what `_trace_memrefs_by_role`
    reports."""
    return MatchTrace(
        target_name=target_name,
        module_name="synthetic",
        matches=[
            MatchedOp(
                target_op_name="MAC",
                func_name="kern_0_0",
                work_id=(0, 0),
                enclosing_loops=[
                    ("%arg0", "0", "16", 1),
                    ("%arg1", "0", "1024", 1),
                ],
                operands=[
                    OperandBinding(role="x", memref_name="local_W"),
                    OperandBinding(role="y", memref_name="local_x"),
                    OperandBinding(role="acc", memref_name="acc", is_loop_carried=True),
                ],
                result_memref_name="acc",
                op_range=("%a", "%b"),
            )
        ],
    )


def test_samsung_enumerator_returns_two_candidates():
    """SPEC-009 §1: bank-row + GRF-staged candidates so argmin has a
    real choice; the cost model's `isinstance(y_handle, MemoryRef)`
    branch produces a ~8x gap between them."""
    target = build_samsung_target()
    candidates = _samsung_enumerate(target, _synthetic_mac_trace().matches)
    assert len(candidates) >= 2, len(candidates)
    modes = {c.mode for c in candidates}
    assert {"bank_row", "grf_staged"} <= modes, modes


def test_samsung_argmin_picks_dual_fiber():
    """SPEC-023 lever 1: the dual-fiber candidate (is_auto=1, both bank
    halves busy) prices ~2x cheaper than bank_row (single fiber) and far
    cheaper than grf_staged (K MACs unrolled), so argmin now returns the
    dual-fiber placement. `y` still lands on a bank `MemoryRef` (the EVEN
    fiber), and `extra["fibers"]` carries both bank fibers EVEN then ODD.
    The winner moved from bank_row -> dual_fiber purely via the cost
    model's `n_fibers` read (SPEC-023 §6)."""
    target = build_samsung_target()
    trace = _synthetic_mac_trace()
    layouts = autoschedule(target, trace)
    assert len(layouts) == 1
    chosen = layouts[0]
    assert chosen.mode == "dual_fiber", chosen.mode
    y_handle = chosen.placements["local_x"]
    assert isinstance(y_handle, MemoryRef), (
        f"dual_fiber y is the EVEN bank MemoryRef; got {y_handle!r}"
    )
    from allo.spmw_codegen import _bank_parity

    fibers = chosen.extra["fibers"]
    assert _bank_parity(fibers[0].idx) == "EVEN_BANK"
    assert _bank_parity(fibers[-1].idx) == "ODD_BANK"


def test_apu_v1_enumerator_returns_two_candidates():
    """SPEC-009 §2: sv + sv_lookup candidates differ only in
    `placement.mode`; cost model branches on mode to charge the
    MUL+ADD (18 cyc) vs lookup+ADD (8 cyc) expansion."""
    target = build_apu_v1_target()
    trace = _apu_synthetic_mac_trace("apu_v1")
    candidates = _apu_v1_enumerate(target, trace.matches)
    assert len(candidates) >= 2, len(candidates)
    modes = {c.mode for c in candidates}
    assert {"sv", "sv_lookup"} <= modes, modes


def test_apu_v1_argmin_picks_sv_lookup():
    """SPEC-009 §2: argmin must select the sv_lookup placement (8 cyc
    per MAC) over the sv placement (18 cyc per MAC)."""
    target = build_apu_v1_target()
    trace = _apu_synthetic_mac_trace("apu_v1")
    layouts = autoschedule(target, trace)
    assert len(layouts) == 1
    assert layouts[0].mode == "sv_lookup", layouts[0].mode


def test_apu_v2_enumerator_returns_two_candidates():
    """SPEC-009 §3: canonical + reversed L1-row bindings; cost is a
    placeholder so the candidates tie and argmin picks lex-first by
    enumerator index."""
    target = build_apu_v2_target()
    trace = _apu_synthetic_mac_trace("apu_v2")
    candidates = _apu_v2_enumerate(target, trace.matches)
    assert len(candidates) >= 2, len(candidates)
    modes = {c.mode for c in candidates}
    assert {"l1_row_canonical", "l1_row_reversed"} <= modes, modes


def test_apu_v2_argmin_picks_canonical():
    """Placeholder cost ties both candidates; argmin's stable
    enumerator-index tie-break must select the canonical binding."""
    import warnings as _warnings

    target = build_apu_v2_target()
    trace = _apu_synthetic_mac_trace("apu_v2")
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", RuntimeWarning)
        layouts = autoschedule(target, trace)
    assert len(layouts) == 1
    assert layouts[0].mode == "l1_row_canonical", layouts[0].mode


if __name__ == "__main__":
    test_samsung_enumerator_respects_mac_dst_grf_b()
    test_kernel_cycles_prefers_is_auto()
    test_autoschedule_picks_is_auto_layout()
    test_compile_with_autoschedule_emits_canonical_bytes()
    test_samsung_enumerator_returns_two_candidates()
    test_samsung_argmin_picks_dual_fiber()
    test_apu_v1_enumerator_returns_two_candidates()
    test_apu_v1_argmin_picks_sv_lookup()
    test_apu_v2_enumerator_returns_two_candidates()
    test_apu_v2_argmin_picks_canonical()
    print("ALL PASSED")
