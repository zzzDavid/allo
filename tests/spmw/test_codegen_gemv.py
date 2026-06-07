# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static-comparison test for the SPMW codegen walker.

Drives `compile_for_target` against the report-16 GEMV workload + Samsung
HBM-PIM target and compares the emitted `PIMCmd` records against the
canonical Samsung GEMV inner-loop op from
`PIMSimulator/src/tests/PIMCmdGen.h::GemvPIMKernel`. No simulator boot.
"""

from __future__ import annotations

import allo
from allo.dataflow import region as _df_region
from allo.ir.types import float32 as fp16
from allo.spmw_codegen import PIMCmd

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


def _canonical_mac() -> PIMCmd:
    """Mirror of PIMCmdGen.h:119 —
    `PIMCmd(MAC, GRF_B, GRF_A, EVEN_BANK, is_auto=1)`.
    """
    return PIMCmd(
        type_="MAC",
        dst_="GRF_B",
        src0_="GRF_A",
        src1_="EVEN_BANK",
        isAuto_=1,
    )


def _canonical_mac_odd() -> PIMCmd:
    """The dual-fiber ODD-bank MAC the lever-1 placement adds — same
    GRF_B<-GRF_A MAC but `src1=ODD_BANK` (2*pid+1)."""
    return PIMCmd(
        type_="MAC",
        dst_="GRF_B",
        src0_="GRF_A",
        src1_="ODD_BANK",
        isAuto_=1,
    )


def _split_jump_k() -> PIMCmd:
    """Per-fiber inner-K JUMP under lever 1: the K reduction is split
    across the two bank halves, so each fiber's JUMP loops
    `(K // lanes) // n_fibers - 1` times. With lanes=8, n_fibers=2,
    K=1024 this is `1024 // 8 // 2 - 1 = 63` — the `63` *emerges* from
    the geometry, it is not written into codegen.
    """
    lanes, n_fibers = 8, 2
    return PIMCmd(
        type_="JUMP",
        loopCounter_=K // lanes // n_fibers - 1,
        loopOffset_=2,
    )


def test_compile_emits_dual_fiber_mac_jump_per_match():
    """Lever 1: argmin now picks the dual-fiber placement, so each of the
    128 MAC matches produces an alternating
    (MAC EVEN, JUMP n_even, MAC ODD, JUMP n_odd) quad keeping both bank
    halves busy. The per-fiber JUMP trip count is the K reduction split
    across the two fibers (`K // lanes // n_fibers - 1`), and the ODD MAC
    targets `ODD_BANK` (2*pid+1) — the parity falls out of the fiber
    handle's idx, not a literal in codegen.
    """
    target = build_samsung_target()
    schedule = allo.customize(gemv_top, enable_tensor=False)
    trace = allo.match_workload(target, schedule.module)

    compiled = allo.compile_for_target(target, trace)

    assert compiled.target is target
    expected_pairs = 16 * 8  # 16 pseudo-channels × 8 PIM units

    mac_even = _canonical_mac()
    mac_odd = _canonical_mac_odd()
    split_jump = _split_jump_k()

    quad_count = 0
    cmds = compiled.cmds
    for i in range(len(cmds) - 3):
        if (
            cmds[i] == mac_even
            and cmds[i + 1] == split_jump
            and cmds[i + 2] == mac_odd
            and cmds[i + 3] == split_jump
        ):
            quad_count += 1
    assert quad_count == expected_pairs, (
        f"expected {expected_pairs} dual-fiber (MAC EVEN, JUMP, MAC ODD, "
        f"JUMP) quads, got {quad_count}; cmds[0:6]={cmds[:6]!r}"
    )

    # At least one MOV must appear before the first MAC and at least
    # one MOV after the last MAC -- the preload/storeback wrappers.
    mac_positions = [i for i, c in enumerate(compiled.cmds) if c.type_ == "MAC"]
    assert mac_positions, "no MAC emitted"
    head = compiled.cmds[: mac_positions[0]]
    tail = compiled.cmds[mac_positions[-1] + 1 :]
    assert any(c.type_ == "MOV" for c in head), (
        f"expected a MOV preload before the first MAC; head={head!r}"
    )
    assert any(c.type_ == "MOV" for c in tail), (
        f"expected a MOV storeback after the last MAC; tail={tail!r}"
    )


if __name__ == "__main__":
    test_compile_emits_dual_fiber_mac_jump_per_match()
    print("ALL PASSED")
