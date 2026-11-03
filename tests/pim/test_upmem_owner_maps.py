# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""UPMEM realizes minimum-movement tie layouts as row-to-DPU assignments.

A tie of the movement stage maps output-row bits to the owner level ``dpu``.
The UPMEM lowering turns it into the rows each DPU holds (ascending, the
plan's row count per DPU): the pack ABI builds DPU ``d``'s MRAM image from
those rows, the gather writes its output back to them, and the task image
(``task.c``) is unchanged.  The program's own work-grid map is the identity
assignment, so its fixtures are byte-identical to a compile without the
movement stage.  Ties that split a row's reduction across DPUs are rejected.
Contract: ``docs/release/design/upmem-domain.md`` (owner-map row placement)
and ``docs/release/design/layout-search.md`` section 8.
"""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

import allo
from allo.dataflow import region as _df_region
from allo.ir.types import int32
from allo.pim.costs import upmem_cost
from allo.pim.targets import build_upmem_target
from allo.spmw_layout_problem import (
    MovementSelectionPolicy,
    movement_selection_policy,
    split_movement_layout,
)
from allo.spmw_upmem import pack_expected, pack_inputs

M, K, DPUS = 1024, 128, 8
ROWS = M // DPUS  # 128: unmasked, and M > DPUS * K / 2 keeps rows minimal
TARGET = build_upmem_target()
CYCLIC = "rows cyclic over dpu"


@_df_region()
def gemv(W: int32[M, K], x: int32[K], y: int32[M]):
    @allo.work(mapping=[DPUS], args=[W, x, y])
    def mv(local_W: int32[M, K], local_x: int32[K], local_y: int32[M]):
        (dpu,) = allo.get_wid()
        for i in range(ROWS):
            local_y[dpu * ROWS + i] = 0
            for j in range(K):
                local_y[dpu * ROWS + i] += local_W[dpu * ROWS + i, j] * local_x[j]


def _all_targets():
    path = Path(__file__).resolve().parent / "test_gemv_all_targets.py"
    spec = importlib.util.spec_from_file_location("upmem_owner_maps_all_targets", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def default():
    return allo.compile(gemv, TARGET, upmem_cost)


def _ties(compiled):
    return {
        (
            tie.description.split(":")[0]
            if tie.description.startswith("rows ")
            else "program"
        ): tie
        for tie in compiled.placement_ranking.kernels[0].movement.ties
    }


def _forced(compiled, tie, workload=gemv):
    placements = []
    for placement in compiled.compiled.layout:
        clone = copy.copy(placement)
        clone.extra = dict(placement.extra)
        clone.layout = split_movement_layout(placement.layout)[0].product(tie.layout)
        placements.append(clone)
    return allo.compile(workload, TARGET, upmem_cost, layout=placements)


def _group(compiled):
    (program,) = compiled.compiled.layout_ctx.groups.values()
    return program


def _fixtures(compiled, *arrays):
    bundle = compiled.compiled.layout_ctx.prepare(list(arrays))
    return [(f.manifest_bytes(), f.files) for f in bundle.fixtures]


def _inputs(m=M, k=K, equal_rows=False):
    rng = np.random.default_rng(0)
    W = rng.integers(-8, 8, size=(m, k), dtype=np.int32)
    if equal_rows:
        W = np.tile(W[:1], (m, 1))
    x = rng.integers(-8, 8, size=k, dtype=np.int32)
    return W, x, np.zeros(m, dtype=np.int32)


# --------------------------------------------------------------------- #
# Static: the movement stage and the emitted assignment
# --------------------------------------------------------------------- #


def test_default_compile_reports_minimal_realizable_ties(default):
    kernel = default.placement_ranking.kernels[0]
    movement = kernel.movement
    assert movement.status == "analyzed" and movement.optimality_proven
    assert movement.program_cost == movement.min_cost
    assert movement.chosen_label == "minimal"
    # The program (block) map and the cyclic row distribution.
    assert movement.tie_count == movement.realizable_tie_count == 2
    assert set(_ties(default)) == {"program", CYCLIC}
    assert not kernel.movement_rejections
    assert len(kernel.scores) % movement.tie_count == 0
    carried = split_movement_layout(default.compiled.layout[0].layout)[1]
    assert carried in [tie.layout for tie in movement.ties]


def test_identity_map_is_byte_identical_to_the_work_grid_assignment(default):
    with movement_selection_policy(MovementSelectionPolicy(enabled=False)):
        today = allo.compile(gemv, TARGET, upmem_cost)
    identity = _forced(default, _ties(default)["program"])
    group = _group(identity)
    assert group.dpu_rows is None and group.row_map is None
    assert list(identity.compiled.cmds) == list(today.compiled.cmds)
    assert group.device_source == _group(today).device_source
    W, x, y = _inputs()
    assert _fixtures(identity, W, x, y) == _fixtures(today, W, x, y)


def test_pinned_upmem_gemv_default_is_unchanged():
    """The all-targets GEMV (96 masked rows per DPU): its minimum tie leaves
    two DPUs without rows and is rejected; the program is selected and its
    fixture bytes equal a compile without the movement stage."""
    workload = _all_targets().upmem_gemv
    compiled = allo.compile(workload, TARGET, upmem_cost)
    with movement_selection_policy(MovementSelectionPolicy(enabled=False)):
        today = allo.compile(workload, TARGET, upmem_cost)
    kernel = compiled.placement_ranking.kernels[0]
    assert kernel.movement.realizable_tie_count == 0
    assert kernel.movement.chosen_label == "not-minimal"
    assert all("unequal row counts" in r for _p, r in kernel.movement_rejections)
    assert list(compiled.compiled.cmds) == list(today.compiled.cmds)
    W, x, y = _inputs(768, 128)
    assert _fixtures(compiled, W, x, y) == _fixtures(today, W, x, y)


def test_cyclic_tie_packs_each_dpu_from_its_rows(default):
    identity = _forced(default, _ties(default)["program"])
    cyclic = _forced(default, _ties(default)[CYCLIC])
    group = _group(cyclic)
    # dpu <- the low three row bits: DPU d holds rows d, d + 8, ... ascending.
    assert group.dpu_rows == tuple(tuple(range(d, M, DPUS)) for d in range(DPUS))
    assert group.row_map["map"] == "permuted"
    # A row stays only when its cyclic DPU equals its block DPU.
    assert group.row_map["rows_moved"] == M - DPUS * (ROWS // DPUS)
    assert group.row_map["placement"][1] == {"dpu": 1, "local_row": 0}
    # Same task image; the command records the map's digest.
    assert group.device_source == _group(identity).device_source
    assert cyclic.compiled.cmds[0].endswith(f"row_map={group.row_map['digest']}")
    W, x, y = _inputs()
    bundle = cyclic.compiled.layout_ctx.prepare([W, x, y])
    (fixture,) = bundle.fixtures
    files = dict(fixture.files)
    for dpu, record in enumerate(fixture.manifest["executions"][0]["dpus"]):
        rows = list(range(dpu, M, DPUS))
        heap = pack_inputs(group.plan, group.kernel, {"matrix": W[rows], "vector": x})
        assert files[record["mram_heap_input"]["file"]] == heap
        _offset, expected = pack_expected(
            group.plan,
            group.kernel,
            {"matrix": W[rows], "vector": x, "out": (W @ x)[rows]},
        )
        assert files[record["mram_heap_output"]["file"]] == expected
    assert _fixtures(cyclic, W, x, y) != _fixtures(identity, W, x, y)


def test_reduction_split_is_rejected_with_the_upmem_reason():
    """64 DPUs x 12 rows: the minimum ties take owner bits from k."""
    from allo.pim.schedule_search import InfeasibleSchedule

    spec = importlib.util.spec_from_file_location(
        "upmem_owner_maps_layout_selection",
        Path(__file__).resolve().parent / "test_layout_selection.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    compiled = allo.compile(module.upmem_gemv, TARGET, upmem_cost)
    movement = compiled.placement_ranking.kernels[0].movement
    (split, *_rest) = [tie for tie in movement.ties if not tie.realizable]
    assert "splits the reduction" in split.reason
    assert "UPMEM accumulates in int32" in split.reason
    assert "do not combine partial sums across DPUs" in split.reason
    with pytest.raises((InfeasibleSchedule, RuntimeError), match="reduction"):
        _forced(compiled, split, module.upmem_gemv)


# --------------------------------------------------------------------- #
# Simulator: two realizable ties executed on uPIMulator
# --------------------------------------------------------------------- #


def test_two_ties_execute_with_identical_outputs(default, record_property):
    """Expected before running: the per-DPU work is identical (128 rows x 128
    columns, same task image, same MRAM packing), so cycles are equal when
    every DPU sees the same values; with random values they may differ by
    the DPU multiply's value-dependent cost, since the DPUs hold other
    rows.  Measured 2026-11-02: random 595,371 (program) vs 593,932
    (cyclic); equal rows 604,829 for both."""
    import conftest

    conftest._simulator_gate("upmem")
    ties = _ties(default)
    W, x, _y = _inputs()
    Wq, _xq, _yq = _inputs(equal_rows=True)
    cycles = {}
    for name in ("program", CYCLIC):
        compiled = _forced(default, ties[name])
        for label, matrix in (("random", W), ("equal_rows", Wq)):
            y = np.zeros(M, dtype=np.int32)
            result = compiled(matrix, x, y)
            np.testing.assert_array_equal(y, matrix @ x)
            assert result.backend == "upmem" and result.kind == "simulator"
            (record,) = result.extra["upmem"]
            if name == CYCLIC:
                assert record["row_map"]["digest"] == _group(compiled).row_map["digest"]
            else:
                assert "row_map" not in record
            cycles[(name, label)] = result.cycles
            record_property(f"{name}:{label}", result.cycles)
    assert cycles[("program", "equal_rows")] == cycles[(CYCLIC, "equal_rows")]
