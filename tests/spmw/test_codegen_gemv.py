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
    """Each of the 128 MAC matches must lower to a (MAC, JUMP) pair
    matching the canonical Samsung GEMV inner loop.

    The walker emits one (MAC + JUMP) pair per match site (128 pairs =
    256 PIMCmds). The pair is the canonical Samsung GEMV inner-K
    sequence from PIMCmdGen.h:GemvPIMKernel — MAC followed by a
    column-strobe JUMP back over the MAC + JUMP body (loop offset 2).
    """
    target = build_samsung_target()
    schedule = allo.customize(gemv_top, enable_tensor=False)
    trace = allo.match_workload(target, schedule.module)

    compiled = allo.compile_for_target(target, trace)

    assert compiled.target is target
    # 16 pseudo-channels × 8 PIM units = 128 work-items, one (MAC, JUMP) each.
    expected_pairs = 16 * 8
    assert len(compiled.cmds) == 2 * expected_pairs, (
        f"expected {2 * expected_pairs} PIMCmds, got {len(compiled.cmds)}"
    )

    canonical_mac = _canonical_mac()
    canonical_jump = _canonical_jump_k()
    for i in range(0, len(compiled.cmds), 2):
        mac, jmp = compiled.cmds[i], compiled.cmds[i + 1]
        assert mac == canonical_mac, f"cmds[{i}] = {mac!r}, expected {canonical_mac!r}"
        assert jmp == canonical_jump, f"cmds[{i+1}] = {jmp!r}, expected {canonical_jump!r}"


if __name__ == "__main__":
    test_compile_emits_canonical_mac_jump_pair_per_match()
    print("ALL PASSED")
