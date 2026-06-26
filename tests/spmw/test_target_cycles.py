# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static assertions for the cycle-constant location (tasks 013 + design 04).

Per design 04 the per-op / per-move cycle numbers no longer live on the
target tree (Move.cycles / Op.cycles are structure-only, now None) -- they
migrated to the bound `CostModel` (`spmw_cost_tables`). This test pins the
values to their published sources *at their new home* so a future
mechanical-refactor doesn't silently drift them, and asserts the target
tree carries no cost number (the design 04 §1.2 acid test).
"""
from __future__ import annotations

from allo.spmw_cost_model import MoveCostCtx, OpCostCtx, get_cost_model
from _fixtures import (
    build_aim_target,
    build_apu_v1_target,
    build_samsung_target,
    build_upmem_target,
)


def _op(model, name):
    return model.op_cost(name, OpCostCtx(name))


def _mv(model, name):
    return model.move_cost(name, MoveCostCtx(name))


def test_samsung_cycles_pinned():
    m = get_cost_model("samsung_hbm_pim", "faithful")
    # spec 015 §6.1: tCCDL=4, RL=20, WL=8, BL=4
    # load = tCCDL+RL+BL//2 = 26; store = tCCDL+WL+BL//2 = 14
    assert _mv(m, "LD_A") == 26
    assert _mv(m, "LD_B") == 26
    assert _mv(m, "ST_A") == 14
    assert _mv(m, "ST_B") == 14
    assert _mv(m, "JUMP") == 1
    assert _op(m, "MAC") == 4               # tCCDL
    assert _op(m, "MUL") == 4
    # grf_a.lanes is GEOMETRY -- it stays on the target tree.
    assert build_samsung_target().grf_a.lanes == 8


def test_aim_cycles_pinned():
    m = get_cost_model("aim", "faithful")
    # JSSC 2023 §IV
    assert _op(m, "MUL") == 4               # EWMUL
    assert _op(m, "ADD") == 4               # EWADD
    assert _op(m, "MAC") == 8               # MAC_SBK
    assert _op(m, "MAC_ABK") == 16
    assert _op(m, "AF") == 6
    assert _mv(m, "RD_SBK") == 24           # tCCDL+RD+BURST
    assert _mv(m, "ST_SBK") == 20           # tCCDL+WR+BURST


def test_upmem_cycles_pinned():
    m = get_cost_model("upmem", "faithful")
    # uPIMulator / HPCA 2024 Table 2
    assert _mv(m, "LD_MRAM") == 1000
    assert _mv(m, "ST_MRAM") == 1000
    assert _mv(m, "LD_WRAM") == 1
    assert _mv(m, "ST_WRAM") == 1
    assert _op(m, "MUL") == 1
    assert _op(m, "ADD") == 1
    assert _op(m, "MAC") == 2
    # revolver_latency rides CostModel.constants (design 04 §1.2).
    assert m.const("revolver_latency") == 11


def test_apu_v1_cycles_pinned():
    m = get_cost_model("apu_v1", "faithful")
    # report 12 §4.2 + pim-apu-v1 skill
    assert _mv(m, "DMA_L4_L1") == 140
    assert _mv(m, "DMA_L1_L4") == 140
    assert _mv(m, "LD_VR") == 5
    assert _mv(m, "ST_VR") == 5
    assert _op(m, "ADD") == 2
    assert _op(m, "MUL") == 16
    assert _op(m, "MAC") == 8               # lookup + add = 6 + 2


def test_target_tree_carries_no_cost_number():
    """The design 04 §1.2 acid test: a target tree, after the decoupling,
    contains no number that is a cost. Geometry (lanes, rows, mapping)
    stays; Move.cycles / Op.cycles / Unit.constants are empty."""
    for build in (
        build_samsung_target,
        build_aim_target,
        build_upmem_target,
        build_apu_v1_target,
    ):
        t = build()
        for u in t._walk():
            for mv in u.moves.values():
                assert mv.cycles is None, (t.name, mv.name)
            for o in u.ops.values():
                assert o.cycles is None, (t.name, o.name)
            assert not u.constants, (t.name, u.name, u.constants)


def test_cost_models_have_no_local_cycle_constants():
    """The five constant blocks listed in task 013 must not reappear in
    the factory module (the numbers live in spmw_cost_tables now)."""
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
