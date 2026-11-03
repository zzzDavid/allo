# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AiM realizes minimum-movement tie layouts as output-row placements.

A tie of the movement stage maps output-row bits to the owner level
``channel``.  The AiM lowering turns it into a row -> output-frontier slot
table (the rows of owner unit ``u`` occupy that unit's slots in ascending
order); the slot's channel and bank follow the mapping's storage contract.
MAC, GB and readout commands address slots, not rows, so the trace of a GB-fed
contraction is unchanged; per-row host transfers are re-addressed and the
host-side result mapping (row -> slot -> channel, bank) is recorded in the
lowering manifest and ``RunResult.extra["row_map"]``.  The program's own map is
the identity (no ``row_map`` entry, identical trace).  A reduction split is
rejected unless it is the ``reduction_partitions`` split, which is routed to
the candidates carrying that ``P``.  Contract: ``docs/release/design/
aim-token-partition.md`` (owner-map row placement) and
``docs/release/design/layout-search.md`` section 8.
"""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

import allo
from allo.dataflow import region as _df_region
from allo.ir.types import bfloat16
from allo.pim.costs import aim_cost
from allo.pim.targets import build_aim_target
from allo.spmw_aim import _frontier_position
from allo.spmw_layout_problem import (
    MovementSelectionPolicy,
    movement_selection_policy,
    split_movement_layout,
)

TARGET = build_aim_target()
REASSOCIATE = {"reduction_order": "reassociate"}
CYCLIC = "rows cyclic over channel"


# 2048 x 64 over 32 channels: M > 32 * K / 2, so splitting k never pays and
# the row-only owner maps are the minimum.
M, K, CHANNELS = 2048, 64, 32
ROWS = M // CHANNELS
# 64 x 2048: every channel bit from k is the minimum (the P = 32 split).
SM, SK = 64, 2048
SROWS = SM // CHANNELS


@_df_region()
def GEMV(W: bfloat16[M, K], x: bfloat16[K], y: bfloat16[M]):
    @allo.work(mapping=[CHANNELS], args=[W, x, y])
    def mv(local_W: bfloat16[M, K], local_x: bfloat16[K], local_y: bfloat16[M]):
        (channel,) = allo.get_wid()
        row0 = channel * ROWS
        for i in range(ROWS):
            acc: bfloat16 = 0
            for j in range(K):
                acc += local_W[row0 + i, j] * local_x[j]
            local_y[row0 + i] = acc


@_df_region()
def SKINNY(W: bfloat16[SM, SK], x: bfloat16[SK], y: bfloat16[SM]):
    @allo.work(mapping=[CHANNELS], args=[W, x, y])
    def mv(local_W: bfloat16[SM, SK], local_x: bfloat16[SK], local_y: bfloat16[SM]):
        (channel,) = allo.get_wid()
        row0 = channel * SROWS
        for i in range(SROWS):
            acc: bfloat16 = 0
            for j in range(SK):
                acc += local_W[row0 + i, j] * local_x[j]
            local_y[row0 + i] = acc


@pytest.fixture(scope="module")
def default():
    return allo.compile(GEMV, TARGET, aim_cost, **REASSOCIATE)


def _ties(compiled):
    return {
        (
            tie.description.split(":")[0]
            if tie.description.startswith("rows ")
            else "program"
        ): tie
        for tie in compiled.placement_ranking.kernels[0].movement.ties
    }


def _with_tie(placement, tie):
    clone = copy.copy(placement)
    clone.extra = dict(placement.extra)
    clone.layout = split_movement_layout(placement.layout)[0].product(tie.layout)
    return clone


def _forced(compiled, tie, workload=GEMV):
    placements = [_with_tie(p, tie) for p in compiled.compiled.layout]
    return allo.compile(workload, TARGET, aim_cost, layout=placements, **REASSOCIATE)


def _contraction(compiled):
    return next(
        entry
        for entry in compiled.compiled.layout_ctx.lowering_manifest
        if entry["kind"] == "contraction"
    )


# --------------------------------------------------------------------- #
# Static: the movement stage and the emitted placement
# --------------------------------------------------------------------- #


def test_default_compile_reports_minimal_realizable_ties(default):
    kernel = default.placement_ranking.kernels[0]
    movement = kernel.movement
    assert movement.status == "analyzed" and movement.optimality_proven
    assert movement.program_cost == movement.min_cost
    assert movement.chosen_label == "minimal"
    assert movement.tie_count == movement.realizable_tie_count == 2
    assert set(_ties(default)) == {"program", CYCLIC}
    assert not kernel.movement_rejections


def test_identity_map_reproduces_the_trace_and_manifest(default):
    with movement_selection_policy(MovementSelectionPolicy(enabled=False)):
        today = allo.compile(GEMV, TARGET, aim_cost, **REASSOCIATE)
    identity = _forced(default, _ties(default)["program"])
    assert list(identity.compiled.cmds) == list(today.compiled.cmds)
    assert "row_map" not in _contraction(identity)
    assert _contraction(identity) == _contraction(today)


def test_pinned_aim_gemv_trace_is_unchanged():
    """The all-targets GEMV (256 x 256): its minimum ties split k and are
    rejected; the selected program's trace equals the one without the
    movement stage (the CENT traces are pinned in test_cent_aim_workloads)."""
    path = Path(__file__).resolve().parent / "test_gemv_all_targets.py"
    spec = importlib.util.spec_from_file_location("aim_owner_maps_all_targets", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    compiled = allo.compile(module.aim_gemv, TARGET, aim_cost, **REASSOCIATE)
    with movement_selection_policy(MovementSelectionPolicy(enabled=False)):
        today = allo.compile(module.aim_gemv, TARGET, aim_cost, **REASSOCIATE)
    assert list(compiled.compiled.cmds) == list(today.compiled.cmds)
    movement = compiled.placement_ranking.kernels[0].movement
    assert movement.realizable_tie_count == 0
    for tie in movement.ties:
        assert "splits the reduction" in tie.reason
        assert "not that split" in tie.reason


def test_cyclic_tie_places_rows_and_maps_results_back(default):
    identity = _forced(default, _ties(default)["program"])
    cyclic = _forced(default, _ties(default)[CYCLIC])
    # Slots, not rows, are addressed by the GB-fed all-bank trace.
    assert list(cyclic.compiled.cmds) == list(identity.compiled.cmds)
    entry = _contraction(cyclic)
    row_map = entry["row_map"]
    slots = np.asarray(row_map["row_slots"])
    assert sorted(slots.tolist()) == list(range(M))
    assert row_map["map"] == "permuted" and row_map["rows_per_unit"] == ROWS
    # channel <- the low five row bits: unit u holds rows u, u + 32, ... at
    # slots [64 u, 64 u + 64) in ascending order.
    for unit in range(CHANNELS):
        rows = np.arange(unit, M, CHANNELS)
        assert slots[rows].tolist() == list(range(unit * ROWS, (unit + 1) * ROWS))
    for row, position in row_map["placement"].items():
        assert position == {
            "slot": int(slots[row]),
            **_frontier_position(entry, int(slots[row])),
        }
    # Host-side result mapping: slot s computes the row stored there, the host
    # reads row r from slot[r].
    rng = np.random.default_rng(0)
    W = rng.standard_normal((M, K))
    x = rng.standard_normal(K)
    row_of_slot = np.argsort(slots)
    physical = W[row_of_slot] @ x
    np.testing.assert_allclose(physical[slots], W @ x)


def test_reduction_partitions_split_is_routed_to_its_candidates():
    from allo.pim.schedule_search import InfeasibleSchedule
    from allo.spmw_autoschedule import _aim_layout_candidates

    compiled = allo.compile(SKINNY, TARGET, aim_cost, **REASSOCIATE)
    movement = compiled.placement_ranking.kernels[0].movement
    routed = [
        tie for tie in movement.ties if "reduction_partitions=32" in tie.description
    ]
    assert len(routed) == 1 and routed[0].realizable
    assert movement.chosen_label == "minimal"
    # Other members of the k-split class (high k bits) are rejected.
    assert all(
        "not that split" in tie.reason for tie in movement.ties if not tie.realizable
    )
    entry = _contraction(compiled)
    assert entry["reduction_partition"]["partitions"] == 32
    assert entry["row_map"] == {"map": "reduction_partitions", "partitions": 32}
    # The split on a placement that lowers P = 1 is refused with the reason.
    _sbk, abk = _aim_layout_candidates(TARGET, compiled.compiled.trace.matches)
    with pytest.raises(InfeasibleSchedule, match="reduction_partitions=32 split"):
        allo.compile(
            SKINNY, TARGET, aim_cost, layout=_with_tie(abk, routed[0]), **REASSOCIATE
        )


# --------------------------------------------------------------------- #
# Simulator: two realizable ties on ramulator2 (timing only)
# --------------------------------------------------------------------- #


def test_two_ties_simulate_with_equal_cycles(default, aim_sim_gate, record_property):
    """Expected before running: equal cycles.  The ties issue the same
    trace (all-bank MACs, masked WR_GB and RD_MAC address slots, not rows);
    only which matrix row each (channel, bank) holds and where the host
    reads each result differ.  Measured 2026-11-02: 507 and 507."""
    del aim_sim_gate
    ties = _ties(default)
    operands = (
        np.zeros((M, K), np.float32),
        np.zeros(K, np.float32),
        np.zeros(M, np.float32),
    )
    cycles = {}
    for name in ("program", CYCLIC):
        compiled = _forced(default, ties[name])
        result = compiled(*operands)
        assert result.kind == "simulator-timing-only"
        if name == CYCLIC:
            (record,) = result.extra["row_map"]
            assert record["digest"] == _contraction(compiled)["row_map"]["digest"]
        else:
            assert "row_map" not in result.extra
        cycles[name] = result.cycles
        record_property(name, result.cycles)
    assert cycles["program"] == cycles[CYCLIC] > 0
