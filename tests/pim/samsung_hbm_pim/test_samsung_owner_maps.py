# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Samsung realizes minimum-movement ties as row placements.

A tie of the movement stage maps output-row bits to the owner levels
``pseudo_channel`` and ``pim``.  The Samsung lowering turns it into a
row-to-slot table (rows of one owner unit share that unit's slot range);
the REDUCE plan hands the table to the conductor (``row_slots``), the
ordered program stages its inputs in that order.  The program's own
work-grid map is the identity table, so the vendor-equivalent program is
unchanged.  Ties that split a row's reduction across units are rejected.
Contract: ``docs/release/design/samsung-execution.md`` section 8 and
``docs/release/design/layout-search.md`` section 8.
"""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

import allo
from allo import spmw_samsung
from allo.pim.costs import samsung_cost
from allo.pim.targets import build_samsung_target
from allo.spmw_layout_problem import split_movement_layout

M, K = 4096, 256
TARGET = build_samsung_target()


def _workload():
    path = Path(__file__).resolve().parent / "gemv" / "workload.py"
    spec = importlib.util.spec_from_file_location("samsung_owner_map_workload", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.gemv_m4096_k256


@pytest.fixture(scope="module")
def strict():
    """The default compile (reduction_order strict: the ordered program)."""
    return allo.compile(_workload(), TARGET, samsung_cost)


@pytest.fixture(scope="module")
def reassociate():
    return allo.compile(
        _workload(), TARGET, samsung_cost, reduction_order="reassociate"
    )


def _ties_by_name(compiled):
    return {
        (
            tie.description.split(":")[0]
            if tie.description.startswith("rows ")
            else "program"
        ): tie
        for tie in compiled.placement_ranking.kernels[0].movement.ties
    }


def _with_tie(placements, tie):
    """The vendor placements with the tie's movement layout: an edited
    decision supplied without a declaration (the vendor declaration names
    the identity row packing; the boundary derives the edited decision's
    own declaration and would refuse a stale one)."""
    out = []
    for placement in placements:
        clone = copy.copy(placement)
        clone.extra = {k: v for k, v in placement.extra.items() if k != "effects"}
        clone.layout = split_movement_layout(placement.layout)[0].product(tie.layout)
        out.append(clone)
    return out


def _forced(placements, tie, **kwargs):
    return allo.compile(
        _workload(),
        TARGET,
        samsung_cost,
        layout=_with_tie(placements, tie),
        **kwargs,
    )


def _program(compiled):
    (program,) = compiled.compiled.layout_ctx.programs
    return program


CYCLIC = "rows cyclic over pseudo_channel, cyclic over pim"


# --------------------------------------------------------------------- #
# Static: the movement stage and the emitted row table
# --------------------------------------------------------------------- #


def test_default_compile_crosses_realizable_ties_and_realizes_the_chosen(strict):
    kernel = strict.placement_ranking.kernels[0]
    movement = kernel.movement
    assert movement.status == "analyzed" and movement.optimality_proven
    assert movement.chosen_label == "minimal"
    assert movement.tie_count > 1 and movement.ties_crossed == movement.tie_count
    assert all(tie.realizable for tie in movement.ties)
    assert len({repr(tie.layout) for tie in movement.ties}) == movement.tie_count
    assert {tie.cost for tie in movement.ties} == {movement.min_cost}
    assert not kernel.movement_rejections
    chosen = strict.compiled.layout[0]
    assert chosen.extra["movement"] == "minimal"
    carried = split_movement_layout(chosen.layout)[1]
    assert carried in [tie.layout for tie in movement.ties]
    # Every tie reached the cost ranking: ties x decision keys were scored.
    assert len(kernel.scores) % movement.tie_count == 0
    assert len(kernel.scores) > movement.tie_count


def test_program_tie_is_the_identity_row_map_and_others_permute(strict):
    ties = _ties_by_name(strict)
    assert set(ties) >= {"program", CYCLIC}
    chosen = strict.compiled.layout
    identity = _program(_forced(chosen, ties["program"]))
    assert identity.row_slots is None and identity.row_map == "identity"
    cyclic = _program(_forced(chosen, ties[CYCLIC]))
    slots = np.asarray(cyclic.row_slots)
    assert sorted(slots.tolist()) == list(range(M))
    # pseudo_channel <- row bits 0..3 and pim <- the next three: the 32 rows of
    # one unit are one residue class mod 128, in ascending order.
    units = slots // 32
    for row in range(128):
        members = np.arange(row, M, 128)
        assert len(set(units[members].tolist())) == 1
        assert (np.diff(slots[members]) == 1).all()
    assert cyclic.row_map.startswith("permuted")
    assert cyclic.lever_record["row_map"] == cyclic.row_map


def test_reduce_plan_carries_the_row_table_only_when_rows_move(reassociate, strict):
    target = TARGET
    vendor = spmw_samsung.samsung_vendor_gemv_layout(target, reassociate.trace)
    geometry = {"channels": 64}
    ties = _ties_by_name(strict)
    plain = _program(
        allo.compile(
            _workload(),
            target,
            samsung_cost,
            layout=vendor,
            reduction_order="reassociate",
        )
    )
    program = _program(_forced(vendor, ties["program"], reduction_order="reassociate"))
    cyclic = _program(_forced(vendor, ties[CYCLIC], reduction_order="reassociate"))
    plain_plan = spmw_samsung.samsung_reduce_plan(plain, geometry, 1)
    assert (
        spmw_samsung.samsung_reduce_plan(program, geometry, 1)["text"]
        == plain_plan["text"]
    )
    assert (
        "row_slots" not in plain_plan["text"] and plain_plan["row_map_digest"] is None
    )
    plan = spmw_samsung.samsung_reduce_plan(cyclic, geometry, 1)
    (line,) = [l for l in plan["text"].splitlines() if l.startswith("row_slots ")]
    table = [int(v) for v in line.split()[1:]]
    assert table == list(cyclic.row_slots) and len(table) == plan["slots"] == M
    assert plan["row_map_digest"] == spmw_samsung._row_map_digest(table)
    # Same CRF words: the row map changes data placement only.
    assert plain.binds == cyclic.binds and plain.tile_fiber == cyclic.tile_fiber


def test_reduction_splitting_ties_are_rejected_with_the_reason(reassociate):
    kernel = reassociate.placement_ranking.kernels[0]
    movement = kernel.movement
    assert movement.min_cost < movement.program_cost
    assert movement.tie_count >= 1 and not any(t.realizable for t in movement.ties)
    for tie in movement.ties:
        assert "splits the reduction" in tie.reason
        assert "belongs to the reduction policy" in tie.reason
    assert kernel.movement_rejections
    assert movement.chosen_label == "not-minimal"
    # Forcing such a tie fails at emission with the same reason.
    target = TARGET
    vendor = spmw_samsung.samsung_vendor_gemv_layout(target, reassociate.trace)
    with pytest.raises(Exception, match="splits the reduction"):
        _program(_forced(vendor, movement.ties[0], reduction_order="reassociate"))


def test_tied_layouts_have_equal_predicted_cycles(strict, reassociate):
    ties = _ties_by_name(strict)
    vendor = spmw_samsung.samsung_vendor_gemv_layout(TARGET, reassociate.trace)
    predicted = {
        name: _forced(vendor, tie, reduction_order="reassociate").estimate().cycles
        for name, tie in ties.items()
    }
    assert len(set(predicted.values())) == 1, predicted


# --------------------------------------------------------------------- #
# Simulator: the placement is executed
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def gemv_inputs():
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((M, K)) * 0.1).astype(np.float16)
    x = (rng.standard_normal(K) * 0.1).astype(np.float16)
    return W, x, W.astype(np.float32) @ x.astype(np.float32)


def test_vendor_program_runs_each_row_map(
    samsung_sim_gate, strict, reassociate, gemv_inputs
):
    W, x, ref = gemv_inputs
    vendor = spmw_samsung.samsung_vendor_gemv_layout(TARGET, reassociate.trace)
    ties = _ties_by_name(strict)
    runs = {}
    for name in ("program", CYCLIC):
        compiled = _forced(vendor, ties[name], reduction_order="reassociate")
        y = np.zeros(M, dtype=np.float16)
        result = compiled(W, x, y)
        np.testing.assert_allclose(y.astype(np.float32), ref, rtol=1e-2, atol=1e-2)
        runs[name] = result
    identity, cyclic = runs["program"], runs[CYCLIC]
    assert identity.cycles == cyclic.cycles == 4435
    assert identity.extra["conductor"]["row_map_map"] == "identity"
    assert identity.extra["conductor"]["row_map_rows_moved"] == 0
    receipt = cyclic.extra["conductor"]
    assert receipt["row_map_map"] == "permuted"
    assert receipt["row_map_rows_moved"] > 0
    assert receipt["row_map_digest"] == cyclic.extra["row_map"]["digest"]
    assert (
        "row_slots" in cyclic.extra["plan"]
        and "row_slots" not in identity.extra["plan"]
    )
    # Row 1: next GRF_B entry of block 0 versus another pseudo-channel group.
    assert identity.extra["row_map"]["placement"][1] == {
        "pass": 0,
        "channel": 0,
        "block": 0,
        "entry": 1,
    }
    assert cyclic.extra["row_map"]["placement"][1]["channel"] != 0
    for key in ("passes", "mac_groups_even", "mac_groups_odd", "column_commands"):
        assert identity.extra["conductor"][key] == receipt[key]


def test_default_ordered_program_runs_each_tie(samsung_sim_gate, strict, gemv_inputs):
    W, x, ref = gemv_inputs
    ties = _ties_by_name(strict)
    runs = {}
    for name in ("program", CYCLIC):
        compiled = _forced(strict.compiled.layout, ties[name])
        y = np.zeros(M, dtype=np.float16)
        result = compiled(W, x, y)
        np.testing.assert_allclose(y.astype(np.float32), ref, rtol=1e-2, atol=1e-2)
        runs[name] = (result, y.copy())
    (identity, y0), (cyclic, y1) = runs["program"], runs[CYCLIC]
    assert identity.cycles == cyclic.cycles
    # Source-order FP16 per row: the placement does not change any rounding.
    assert np.array_equal(y0, y1)
    assert identity.extra["row_map"]["map"] == "identity"
    assert cyclic.extra["row_map"]["map"].startswith("permuted")
    assert (
        identity.extra["row_map"]["placement"] != cyclic.extra["row_map"]["placement"]
    )
