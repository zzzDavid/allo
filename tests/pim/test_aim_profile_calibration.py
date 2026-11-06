# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The AiM contraction profile's command-class charges (release round 50,
M-19), each class fixed by a controlled single-factor measurement on the
timing simulator (``tenon-release-evidence/r50/controlled_variations.json``):

* WR_GB: one burst (N_CCD) per op-size column plus the command latency
  (measured 2n + 3 after a register read, n = 1..64);
* WR_BIAS: one burst each (measured 2, runs of 1..16);
* MAC line after a fresh activation: N_RCD_RD_MAC then N_CCD per further
  column (measured 55 + 2(c-1) + the mode change, c = 1..64);
* consecutive MAC at a new row (the previous command a MAC): tRTP + tRP +
  tRCD + N_CCD per further column (measured 100 + 2(c-1), 2..9 MACs);
* MAC at a new row after register commands: tRP + tRCD + N_CCD per column
  (measured 104 for c = 8 per extra reuse group);
* MAC at the open row: N_CCD per column (measured 2c);
* RD_MAC: its read latency (3) each, one more for the first of a run
  (measured 36 = 32 + 4 then 3 per extra read, runs of 1..32);
* the controller's mode change (TMOD, N_MODCH) once per transition between
  register and array commands, from the controller's initial array mode
  (the same stream on 1, 2, 4 or 8 channels measures the same cycles).

These tests pin the structure of the charges, not any workload's outcome.
"""

import pytest

import allo
from allo.pim import aim_lowering
from allo.pim.costs import aim as aim_costs
from allo.pim.costs import aim_cost
from allo.pim.targets import build_aim_target

from benchmarks.cent_aim import token_partition as tp


@pytest.fixture(scope="module")
def target():
    return build_aim_target()


def _profiles(target, built):
    geometry = aim_lowering._target_geometry(target)
    out = {}
    for name, placement in zip(("attention_qk", "attention_sv"), tp._placements(built)):
        operation = aim_costs._contraction_record(placement.extra)
        profile = aim_lowering.aim_contraction_profile(
            operation, geometry, aim_costs._sbk_banks(placement.extra, geometry)
        )
        out[name] = (
            placement,
            profile,
            aim_costs.contraction_profile_charges(profile),
            aim_costs._contraction_profile_cycles(placement.extra, geometry),
        )
    return out


def test_constants_are_the_simulator_timings():
    assert (
        aim_costs.N_RTP == 12
        and aim_costs.N_RP == 32
        and aim_costs.N_RCD_RD_MAC == 56
        and aim_costs.N_CCD == 2
        and aim_costs.N_MODCH == 32
    )
    assert aim_costs.CMD_RD_REG == 3  # nCLREG + nBL
    assert aim_costs.CMD_WR_GB == 3  # nCWLGB + nBL
    assert aim_costs.BATCH_MAPPING_ROW_ACTIVATE_CYCLES == 100
    assert aim_costs.GROUP_ROW_SWITCH_CYCLES == 88
    assert aim_costs.MODE_SWITCH_CYCLES == 32
    assert aim_lowering.REGISTER_MODE_OPCODES == {"WR_GB", "WR_BIAS", "RD_MAC", "RD_AF"}


def test_single_group_contraction_charges(target):
    """QK flattened, one reuse group (WR_GB 8, WR_BIAS, MAC_ABK 8 columns,
    RD_MAC): the standalone stream the simulator runs in 190 cycles."""
    attention, program = tp.build_attention(1, 128)
    built = tp.compile_program(attention, program, target, budget=tp.BUDGET)
    placement, profile, charges, total = _profiles(target, built)["attention_qk"]
    assert (
        tp._knobs(placement.mode)["batch_mapping"] == "flattened"
        and tp._knobs(placement.mode)["reuse_group"] == "1"
    )
    assert profile["counts"] == {"WR_GB": 1, "WR_BIAS": 1, "MAC_ABK": 1, "RD_MAC": 1}
    assert (
        profile["wr_gb_columns"] == 8
        and profile["mode_switches"] == 3
        and profile["register_read_runs"] == 1
    )
    assert (
        profile["first_activations"] == 1
        and profile["first_columns"] == 8
        and profile["row_switches"] == 0
    )
    assert charges == {
        "mac_continuing": 0,
        "mac_first_activations": 56 + 2 * 7,
        "mac_run_switches": 0,
        "mac_group_switches": 0,
        "mode_switches": 3 * 32,
        "wr_gb": 3 + 2 * 8,
        "wr_bias": 2,
        "register_reads": 3 + 1,
        "af": 0,
    }
    assert total == sum(charges.values()) == 191


def test_channel_batched_split_contraction_charges(target):
    """SV channel-batched, four-way split, 16-slot reuse: one WR_GB of two
    columns, eight WR_BIAS, eight MAC lines of two columns at eight
    different rows (one fresh activation, seven in-run row switches), eight
    RD_MAC; the mode changes once into register mode, once to the MACs and
    once back."""
    attention, program = tp.build_attention(1, 128)
    ranking = tp.compile_program(
        attention, program, target, budget=tp.BUDGET, virtual=True
    )
    variants = tp.budget_variants(target, ranking)
    built = tp.compile_variant(
        attention,
        program,
        target,
        variants,
        variants.per_partition[4],
        budget=tp.BUDGET,
    )
    placement, profile, charges, total = _profiles(target, built)["attention_sv"]
    assert (
        tp._knobs(placement.mode)["batch_mapping"] == "channels"
        and tp.partitions_of(placement.mode) == 4
    )
    assert profile["counts"] == {"WR_GB": 1, "WR_BIAS": 8, "MAC_ABK": 8, "RD_MAC": 8}
    assert (
        profile["first_activations"],
        profile["run_switches"],
        profile["group_switches"],
    ) == (1, 7, 0)
    assert (
        profile["first_columns"] == 2
        and profile["run_switch_columns"] == 14
        and profile["continuing_columns"] == 0
    )
    assert (
        profile["mode_switches"] == 3
        and profile["register_read_runs"] == 1
        and profile["wr_gb_columns"] == 2
    )
    assert charges["mac_first_activations"] == 56 + 2 and charges[
        "mac_run_switches"
    ] == 7 * (100 + 2)
    assert (
        charges["wr_gb"] == 3 + 4
        and charges["wr_bias"] == 16
        and charges["register_reads"] == 8 * 3 + 1
    )
    assert total == 916


def test_reuse_groups_pay_two_mode_changes_each(target):
    """The historical control's SV (channel-batched, one MAC per reuse
    group, eight groups): every group after the first pays WR_BIAS, a mode
    change, a row switch after register commands (tRP + tRCD + its
    columns), a mode change back and its RD_MAC run; the QK row-packed with
    one reuse group per MAC walks the open row (column pitch only) except
    where a new WR_GB scope starts a new row."""
    attention, program = tp.cent_repaired_program(1, 512)
    ranking = tp.compile_program(
        attention, program, target, budget=tp.BUDGET, virtual=True
    )
    layouts, _description = tp.cent_repaired_layouts(target, ranking, 1)
    built = tp.compile_program(
        attention, program, target, layouts=layouts, budget=tp.BUDGET
    )
    profiles = _profiles(target, built)
    _placement, sv, sv_charges, sv_total = profiles["attention_sv"]
    assert (
        sv["reuse_groups"] == 8
        and sv["mode_switches"] == 1 + 2 * 8
        and sv["register_read_runs"] == 8
    )
    assert (sv["first_activations"], sv["run_switches"], sv["group_switches"]) == (
        1,
        0,
        7,
    )
    assert sv_charges["mac_group_switches"] == 7 * 88 + 2 * sv["group_switch_columns"]
    assert (
        sv_charges["mode_switches"] == 17 * 32
        and sv_charges["register_reads"] == 8 * 3 + 8
    )
    _placement, qk, qk_charges, qk_total = profiles["attention_qk"]
    assert (
        qk["reuse_groups"] == 32
        and qk["mode_switches"] == 1 + 2 * 32
        and qk["counts"]["WR_GB"] == 4
    )
    assert (
        qk["continuing_columns"] == 224
        and qk["group_switches"] == 3
        and qk["run_switches"] == 0
    )
    assert (
        qk_charges["mac_continuing"] == 2 * 224
        and qk_charges["wr_gb"] == 4 * 3 + 2 * qk["wr_gb_columns"]
    )
    assert sv_total == sum(sv_charges.values()) and qk_total == sum(qk_charges.values())


def test_register_commands_are_not_charged_a_mode_change_each(target):
    """Splitting a channel-batched SV into eight reuse groups adds the
    per-group structure at its measured cost, not N_MODCH per register
    command: the difference between the one-group (16-slot) and the
    eight-group (reuse 1) programs is the seven extra groups' two mode
    changes and RD_MAC run start, plus their row switches after register
    commands (tRP + tRCD + all columns) in place of the in-run switches
    (tRTP + tRP + tRCD + the further columns); the WR_BIAS per accumulator
    is the same in both."""
    attention, program = tp.cent_repaired_program(1, 512)
    ranking = tp.compile_program(
        attention, program, target, budget=tp.BUDGET, virtual=True
    )
    _index, scores = tp.kernel_choices(target, ranking)["attention_sv"]
    by_reuse = {}
    for _cycles, mode, placement in scores:
        knobs = tp._knobs(mode)
        if (
            knobs["bank_scope"] == "abk"
            and knobs["batch_mapping"] == "channels"
            and knobs["replica_partitions"] == "8"
            and tp.partitions_of(mode) == 1
        ):
            by_reuse[int(knobs["reuse_group"])] = placement
    geometry = aim_lowering._target_geometry(target)
    one = aim_costs._contraction_profile_cycles(by_reuse[1].extra, geometry)
    sixteen = aim_costs._contraction_profile_cycles(by_reuse[16].extra, geometry)
    assert one - sixteen == 7 * (2 * 32 + 1) + 7 * (88 + 2 * 32) - 7 * (100 + 2 * 31)
