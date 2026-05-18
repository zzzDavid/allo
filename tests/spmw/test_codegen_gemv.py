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


def _canonical_jump_k() -> PIMCmd:
    """The Samsung GEMV inner-K JUMP: `JUMP(K // 8 - 1, 2)`.

    Mirrors PIMCmdGen.h:121 with `num_jump_to_be_taken_even_bank` set to
    the placeholder formula `(K // GRF_LANES) - 1` used by the walker.
    """
    return PIMCmd(
        type_="JUMP",
        loopCounter_=K // 8 - 1,
        loopOffset_=2,
    )


def test_compile_emits_canonical_mac_jump_pair_per_match():
    """Each of the 128 MAC matches must produce a (MAC, JUMP) pair
    matching the canonical Samsung GEMV inner loop, now bracketed by
    preload (MOV LD_*) and storeback (MOV ST_*) moves per spec 009.

    The canonical (MAC, JUMP) subsequence still appears 128 times in
    the cmds stream; this test asserts the subsequence rather than a
    strict total length so the new LD/ST records don't desynchronize it.
    """
    target = build_samsung_target()
    schedule = allo.customize(gemv_top, enable_tensor=False)
    trace = allo.match_workload(target, schedule.module)

    compiled = allo.compile_for_target(target, trace)

    assert compiled.target is target
    expected_pairs = 16 * 8  # 16 pseudo-channels × 8 PIM units

    canonical_mac = _canonical_mac()
    canonical_jump = _canonical_jump_k()

    mac_jump_count = 0
    for i in range(len(compiled.cmds) - 1):
        if compiled.cmds[i] == canonical_mac and compiled.cmds[i + 1] == canonical_jump:
            mac_jump_count += 1
    assert mac_jump_count == expected_pairs, (
        f"expected {expected_pairs} canonical (MAC, JUMP) pairs, "
        f"got {mac_jump_count}; cmds[0:6]={compiled.cmds[:6]!r}"
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
    test_compile_emits_canonical_mac_jump_pair_per_match()
    print("ALL PASSED")
