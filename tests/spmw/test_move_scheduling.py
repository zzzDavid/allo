# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Move scheduling (spec 009) -- preload/storeback emission tests.

Each test drives a synthetic trace through `compile_for_target` (no
simulator boot) and asserts the LD/ST move pattern emitted around the
compute body.
"""

from __future__ import annotations

import warnings

import allo
from allo.spmw_codegen import PIMCmd, compile_for_target
from allo.spmw_match import MatchTrace, MatchedOp, OperandBinding
from allo.spmw_target import UnitId

from _fixtures import (
    build_aim_target,
    build_apu_v1_target,
    build_apu_v2_target,
    build_samsung_target,
    build_upmem_target,
)


def _samsung_synthetic_trace(work_ids: list[tuple[int, ...]]) -> MatchTrace:
    """Build a Samsung-shaped MAC trace with one match per ``work_id``."""
    matches = []
    for wid in work_ids:
        matches.append(
            MatchedOp(
                target_op_name="MAC",
                func_name=f"gemv_{'_'.join(map(str, wid))}",
                work_id=wid,
                enclosing_loops=[
                    ("%arg0", "0", "32", 1),
                    ("%arg1", "0", "1024", 1),
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
        )
    return MatchTrace(
        target_name="samsung_hbm_pim", module_name="synthetic", matches=matches
    )


def _aim_synthetic_trace(work_ids: list[tuple[int, ...]]) -> MatchTrace:
    """Build an AiM-shaped MAC trace."""
    matches = []
    for wid in work_ids:
        matches.append(
            MatchedOp(
                target_op_name="MAC",
                func_name=f"gemv_{'_'.join(map(str, wid))}",
                work_id=wid,
                enclosing_loops=[
                    ("%arg0", "0", "16", 1),
                    ("%arg1", "0", "1024", 1),
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
        )
    return MatchTrace(
        target_name="aim", module_name="synthetic", matches=matches
    )


def _apu_v1_synthetic_trace() -> MatchTrace:
    return MatchTrace(
        target_name="apu_v1",
        module_name="synthetic",
        matches=[
            MatchedOp(
                target_op_name="MAC",
                func_name="gemv_0",
                work_id=(0,),
                enclosing_loops=[
                    ("%arg0", "0", "16", 1),
                    ("%arg1", "0", "1024", 1),
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


# --------------------------------------------------------------------- #
# Samsung
# --------------------------------------------------------------------- #


def test_samsung_gemv_emits_ld_mac_jump_st_per_workid():
    """One work-id => LD_A preload, MAC + JUMP body, ST_B storeback.

    With autoschedule's preferred placement (x=grf_a, y=even_bank,
    acc=grf_b), grf_a's preload is LD_A (read-only x) and grf_b's
    storeback is ST_B (acc; no LD half since acc starts at 0).
    """
    target = build_samsung_target()
    trace = _samsung_synthetic_trace([(0, 0)])
    compiled = compile_for_target(target, trace)

    cmds = compiled.cmds
    types = [c.type_ for c in cmds]
    # Exactly: MOV (LD_A), MAC, JUMP, MOV (ST_B).
    assert types == ["MOV", "MAC", "JUMP", "MOV"], types

    # MOV LD_A -- dst is GRF_A, src0 is the EVEN_BANK that local_W lives on.
    assert cmds[0].dst_ == "GRF_A", cmds[0]
    # MOV ST_B -- src0 is GRF_B, dst is the ODD_BANK that acc spills to.
    assert cmds[-1].src0_ == "GRF_B", cmds[-1]


def test_two_workids_emit_two_preload_blocks():
    """Two distinct work_ids => two LD/ST blocks, not one shared."""
    target = build_samsung_target()
    trace = _samsung_synthetic_trace([(0, 0), (0, 1)])
    compiled = compile_for_target(target, trace)

    # Two work-ids × (LD_A, MAC, JUMP, ST_B) = 8 PIMCmds.
    types = [c.type_ for c in compiled.cmds]
    assert types == [
        "MOV", "MAC", "JUMP", "MOV",
        "MOV", "MAC", "JUMP", "MOV",
    ], types
    # Two preload MOVs (load) and two storeback MOVs (store).
    movs = [c for c in compiled.cmds if c.type_ == "MOV"]
    assert len(movs) == 4


def test_samsung_jump_stays_paired_with_mac():
    """The JUMP record must sit immediately after its MAC inside the
    work-id window, not after the storeback. Spec 009 §E rule 4."""
    target = build_samsung_target()
    trace = _samsung_synthetic_trace([(0, 0)])
    compiled = compile_for_target(target, trace)

    mac_idx = [i for i, c in enumerate(compiled.cmds) if c.type_ == "MAC"]
    jump_idx = [i for i, c in enumerate(compiled.cmds) if c.type_ == "JUMP"]
    assert len(mac_idx) == 1 and len(jump_idx) == 1
    # JUMP follows MAC directly.
    assert jump_idx[0] == mac_idx[0] + 1


# --------------------------------------------------------------------- #
# AiM
# --------------------------------------------------------------------- #


def test_aim_no_pimcmd_record_in_cmds():
    """AiM compiled.cmds must be all-string -- no PIMCmd JUMP leakage
    from the Samsung-specific inner-loop folder."""
    target = build_aim_target()
    trace = _aim_synthetic_trace([(0, 0, 0, 0)])
    compiled = compile_for_target(target, trace)
    assert all(isinstance(c, str) for c in compiled.cmds), compiled.cmds


def test_aim_per_bank_emits_wr_sbk_and_rd_mac():
    """The per-bank MAC placement preloads each operand bank via WR_SBK
    and reads the accumulator back via RD_MAC at the end."""
    target = build_aim_target()
    trace = _aim_synthetic_trace([(0, 0, 0, 0)])
    compiled = compile_for_target(target, trace)

    joined = "\n".join(compiled.cmds)
    assert "WR_SBK" in joined, joined
    assert "MAC_SBK" in joined, joined
    assert "RD_MAC" in joined, joined
    # The WR_SBK preload precedes MAC_SBK; RD_MAC follows it.
    wr_pos = next(i for i, c in enumerate(compiled.cmds) if "WR_SBK" in c)
    mac_pos = next(i for i, c in enumerate(compiled.cmds) if "MAC_SBK" in c)
    rd_pos = next(i for i, c in enumerate(compiled.cmds) if "RD_MAC" in c)
    assert wr_pos < mac_pos < rd_pos


def test_broadcast_operand_emits_one_preload():
    """An AiM placement that routes both x and y through `WR_GB` (or any
    shared broadcast move) emits the move exactly once per work-id."""
    target = build_aim_target()
    trace = _aim_synthetic_trace([(0, 0, 0, 0)])

    # Force both x and y onto `gb` (the broadcast Memory). Even though
    # the canonical enumerator wouldn't pick this, the dedup invariant
    # must hold for any layout the autoscheduler eventually returns.
    layout = allo.Placement(placements={
        "local_W": target.gb,
        "local_x": target.gb,
        "acc": target.gpr,
    })
    compiled = compile_for_target(target, trace, layout=layout)

    wr_gb_count = sum(1 for c in compiled.cmds if c.startswith("AiM WR_GB"))
    assert wr_gb_count == 1, compiled.cmds


# --------------------------------------------------------------------- #
# APU v1
# --------------------------------------------------------------------- #


def test_apu_v1_two_stage_l4_preload():
    """APU v1 with an l4-source placement emits both stages of the
    chained `LD_L4_TO_VR` move (DMA L4->L1 then gvml_load_16)."""
    target = build_apu_v1_target()
    trace = _apu_v1_synthetic_trace()
    layout = allo.Placement(placements={
        "local_W": target.l4,
        "local_x": target.l4,
        "acc": target.l4,
    })
    compiled = compile_for_target(target, trace, layout=layout)

    joined = "\n".join(compiled.cmds)
    assert "direct_dma_l4_to_l1_32k" in joined, joined
    assert "gvml_load_16" in joined, joined
    # The DMA stage precedes the VR load.
    dma_pos = next(
        i for i, c in enumerate(compiled.cmds)
        if "direct_dma_l4_to_l1_32k" in c
    )
    ld_pos = next(
        i for i, c in enumerate(compiled.cmds) if "gvml_load_16" in c
    )
    assert dma_pos < ld_pos


# --------------------------------------------------------------------- #
# APU v2
# --------------------------------------------------------------------- #


def test_apu_v2_emits_no_moves():
    """APU v2's resolve_moves returns (None, None) for everything --
    compiled.cmds contains only the g.matmul / g.add expansion."""
    target = build_apu_v2_target()
    trace = MatchTrace(
        target_name="apu_v2",
        module_name="synthetic",
        matches=[
            MatchedOp(
                target_op_name="MAC",
                func_name="gemv_0",
                work_id=(0,),
                enclosing_loops=[
                    ("%arg0", "0", "16", 1),
                    ("%arg1", "0", "4096", 1),
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        compiled = compile_for_target(target, trace)
    # No copy_to_l1 / copy_from_l1 lines -- moves are functional no-ops.
    joined = "\n".join(compiled.cmds)
    assert "copy_to_l1" not in joined, joined
    assert "copy_from_l1" not in joined, joined
    # The matmul + add body is present.
    assert "g.matmul(" in joined
    assert "g.add(" in joined


if __name__ == "__main__":
    test_samsung_gemv_emits_ld_mac_jump_st_per_workid()
    test_two_workids_emit_two_preload_blocks()
    test_samsung_jump_stays_paired_with_mac()
    test_aim_no_pimcmd_record_in_cmds()
    test_aim_per_bank_emits_wr_sbk_and_rd_mac()
    test_broadcast_operand_emits_one_preload()
    test_apu_v1_two_stage_l4_preload()
    test_apu_v2_emits_no_moves()
    print("ALL PASSED")
