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
from allo.spmw_autoschedule import _samsung_enumerate, autoschedule
from allo.spmw_codegen import PIMCmd
from allo.spmw_match import MatchTrace, MatchedOp, OperandBinding
from allo.spmw_target import MemoryRef, Register, UnitId

from _fixtures import build_samsung_target


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

    # x and y must be distinct hardware resources.
    for layout in candidates:
        assert layout.placements["local_W"] is not layout.placements["local_x"]


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
    """Driving `compile_for_target` with no explicit layout must still
    produce the canonical Samsung GEMV inner loop — autoschedule's
    argmin should pick the (x=grf_a, y=even_bank) layout (or its
    odd-bank equivalent), and either yields canonical bytes after the
    bank-parity classifier in SamsungCtx.

    After spec 009 the cmds stream is bracketed with LD/ST MOV
    instructions per work-id; the canonical (MAC, JUMP) subsequence
    still appears once per work-id.
    """
    target = build_samsung_target()
    schedule = allo.customize(gemv_top, enable_tensor=False)
    trace = allo.match_workload(target, schedule.module)

    compiled = allo.compile_for_target(target, trace)

    canonical_jump = _canonical_jump()
    mac_jump_count = 0
    for i in range(len(compiled.cmds) - 1):
        cur, nxt = compiled.cmds[i], compiled.cmds[i + 1]
        if cur.type_ == "MAC" and nxt == canonical_jump:
            assert cur.dst_ == "GRF_B"
            assert cur.src0_ == "GRF_A"
            assert cur.src1_ in ("EVEN_BANK", "ODD_BANK"), cur.src1_
            assert cur.isAuto_ == 1
            mac_jump_count += 1
    assert mac_jump_count == 16 * 8


if __name__ == "__main__":
    test_samsung_enumerator_respects_mac_dst_grf_b()
    test_kernel_cycles_prefers_is_auto()
    test_autoschedule_picks_is_auto_layout()
    test_compile_with_autoschedule_emits_canonical_bytes()
    print("ALL PASSED")
