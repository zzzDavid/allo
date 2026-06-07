# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static assertions for the cycle-constant lift (task 013).

Per-op and per-move cycle numbers are declared on each target's spec
(Move.cycles / Op.cycles) and read by `spmw_cost_models` from there.
This test pins the values to their published sources so a future
mechanical-refactor doesn't silently drift them.
"""
from __future__ import annotations

from _fixtures import (
    build_aim_target,
    build_apu_v1_target,
    build_samsung_target,
    build_upmem_target,
)


def test_samsung_cycles_pinned():
    t = build_samsung_target()
    # spec 015 §6.1: tCCDL=4, RL=20, WL=8, BL=4
    # load = tCCDL+RL+BL//2 = 26; store = tCCDL+WL+BL//2 = 14
    assert t.move("LD_A").cycles == 26
    assert t.move("LD_B").cycles == 26
    assert t.move("ST_A").cycles == 14
    assert t.move("ST_B").cycles == 14
    assert t.move("JUMP").cycles == 1     # _SAMSUNG_JUMP_CYCLES
    assert t.op("MAC").cycles == 4         # _SAMSUNG_TCCDL
    assert t.op("MUL").cycles == 4
    assert t.grf_a.lanes == 8              # _SAMSUNG_LANE_BURST


def test_aim_cycles_pinned():
    t = build_aim_target()
    # JSSC 2023 §IV
    assert t.op("MUL").cycles == 4         # _AIM_EWMUL_CYCLES
    assert t.op("ADD").cycles == 4         # _AIM_EWADD_CYCLES
    assert t.op("MAC").cycles == 8         # _AIM_MAC_SBK_CYCLES
    assert t.op("MAC_ABK").cycles == 16    # _AIM_MAC_ABK_CYCLES
    assert t.op("AF").cycles == 6          # _AIM_AF_CYCLES
    assert t.move("RD_SBK").cycles == 24   # tCCDL+RD+BURST
    assert t.move("ST_SBK").cycles == 20   # tCCDL+WR+BURST


def test_upmem_cycles_pinned():
    t = build_upmem_target()
    # uPIMulator / HPCA 2024 Table 2
    assert t.move("LD_MRAM").cycles == 1000
    assert t.move("ST_MRAM").cycles == 1000
    assert t.move("LD_WRAM").cycles == 1
    assert t.move("ST_WRAM").cycles == 1
    assert t.op("MUL").cycles == 1          # _UPMEM_GPR_OP_CYCLES
    assert t.op("ADD").cycles == 1
    assert t.op("MAC").cycles == 2          # _UPMEM_MAC_CYCLES


def test_apu_v1_cycles_pinned():
    t = build_apu_v1_target()
    # report 12 §4.2 + pim-apu-v1 skill
    assert t.move("DMA_L4_L1").cycles == 140
    assert t.move("DMA_L1_L4").cycles == 140
    assert t.move("LD_VR").cycles == 5
    assert t.move("ST_VR").cycles == 5
    assert t.op("ADD").cycles == 2          # _APU_V1_ADD_CYCLES
    assert t.op("MUL").cycles == 16         # _APU_V1_MUL_CYCLES
    assert t.op("MAC").cycles == 8          # lookup + add = 6 + 2


def test_cost_models_have_no_local_cycle_constants():
    """The five constant blocks listed in task 013 must not reappear."""
    import pathlib

    here = pathlib.Path(__file__).resolve().parent.parent.parent
    src = (here / "allo" / "spmw_cost_models.py").read_text()
    for tok in (
        "_SAMSUNG_TCCDL",
        "_SAMSUNG_JUMP_CYCLES",
        "_SAMSUNG_LANE_BURST",
        "_AIM_MAC_SBK_CYCLES",
        "_AIM_OP_CYCLES",
        "_UPMEM_MRAM_READ_CYCLES_PER_64B",
        "_UPMEM_OP_CYCLES",
        "_APU_V1_DMA_L4_L1_CYCLES",
        "_APU_V1_MUL_CYCLES",
        "_APU_V1_OP_CYCLES",
    ):
        assert tok not in src, f"{tok!r} reappeared in spmw_cost_models.py"
