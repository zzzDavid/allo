# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Samsung executable-cost integration tests."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from allo.perf import cost
from allo.pim.costs import samsung_cost
from allo.pim.targets import build_samsung_target
from allo.spmw_autoschedule import Placement, autoschedule
from allo import spmw_samsung
from allo.spmw_codegen import Compiled, PIMCmd
from allo.spmw_match import MatchTrace, MatchedOp, OperandBinding
from allo.spmw_plan import build_execution_graph


def _mac(work_id, index=0):
    return MatchedOp(
        target_op_name="MAC",
        func_name="gemv",
        work_id=work_id,
        enclosing_loops=[("k", "0", "128", 1)],
        operands=[],
        result_memref_name=None,
        op_range=(f"begin{index}", f"end{index}"),
    )


def _bound_mac(work_id=(0, 0)):
    # A structurally complete GEMV match: y[i] = sum_k W[i, k] * x[k].
    match = _mac(work_id)
    match.enclosing_loops = [("i", "0", "8", 1), ("k", "0", "128", 1)]
    from allo.spmw_match_engine import coordinate_forms

    match.operands = [
        OperandBinding(
            "x", "x", ["k"], memref_type="memref<128xf16>", coords=coordinate_forms("k")
        ),
        OperandBinding(
            "y",
            "W",
            ["i", "k"],
            memref_type="memref<8x128xf16>",
            coords=coordinate_forms("i", "k"),
        ),
        OperandBinding(
            "acc", "acc", is_loop_carried=True, memref_type="memref<f16>", coords=()
        ),
    ]
    match.result_memref_name = "acc"
    match.extra = {
        "accumulator_init": "zero",
        "axis_roles": {"reduction": ("k",), "output": ("i",)},
        "output_sinks": [("y", "i")],
        "store_coords": (),
        "output_sink_coords": [coordinate_forms("i")],
    }
    # The REDUCE placements regroup the FP16 reduction: permit it explicitly.
    from allo.spmw_reduction_policy import stamp_reduction_policy

    stamp_reduction_policy([match], "reassociate")
    return match


def test_virtual_graph_includes_shape_aware_host_transfers():
    target = build_samsung_target()
    trace = MatchTrace(target.name, "host_boundary", [_bound_mac((0, 0))])
    bound = samsung_cost.bind(target)
    scatter = SimpleNamespace(
        verb=SimpleNamespace(name="scatter"),
        move=target.move("SCATTER_BANKS"),
        buffer_role="W",
    )
    gather = SimpleNamespace(
        verb=SimpleNamespace(name="gather"),
        move=target.move("GATHER_BANKS"),
        buffer_role="out",
    )

    graph = build_execution_graph(
        target,
        trace,
        # The MAC accumulates in GRF_B; the plan reads its lane count there.
        Placement(placements={"acc": target.grf_b}, extra={"n_fibers": 1}),
        bound,
        host_moves=[scatter, gather],
        buffer_metrics={
            "W": {"shape": (16,), "elements": 16, "bytes": 64},
            "out": {"shape": (8,), "elements": 8, "bytes": 32},
        },
    )
    estimate = bound.evaluate(graph)

    # The accumulator lives in GRF_B, so the plan materializes its load and
    # drain around the MAC; the host moves bracket the device program.  Since
    # round 37 the MAC event also carries this work item's share of the first
    # matrix staging (``mac_stage``), which precedes the MAC as the conductor
    # stages before it executes (round 38); this placement declares no GRF_A
    # residency, which the model prices as a bank fill (as the LD_A rule
    # does), so the vector staging of that fill (``mac_vector_stage``, round
    # 132) follows the matrix staging.
    assert [activity.primitive for activity in graph.activities] == [
        "samsung_hbm_pim/host/SCATTER_BANKS",
        "samsung_hbm_pim/hbm_pim/pseudo_channel/pim/LD_B",
        "samsung_hbm_pim/hbm_pim/pseudo_channel/pim/MAC",
        "samsung_hbm_pim/hbm_pim/pseudo_channel/pim/MAC",
        "samsung_hbm_pim/hbm_pim/pseudo_channel/pim/MAC",
        "samsung_hbm_pim/hbm_pim/pseudo_channel/pim/ST_B",
        "samsung_hbm_pim/host/GATHER_BANKS",
    ]
    assert [activity.label for activity in graph.activities][2:5] == [
        "mac_stage",
        "mac_vector_stage",
        "mac",
    ]
    stage, vector, mac = graph.activities[2:5]
    assert stage.id in vector.depends_on and vector.id in mac.depends_on
    # Host moves are priced from their buffer bytes (64 B scatter -> 4 cycles,
    # 32 B gather -> 3 cycles) and serialize with the device program.
    device_only = bound.evaluate(
        build_execution_graph(
            target,
            trace,
            Placement(placements={"acc": target.grf_b}, extra={"n_fibers": 1}),
            bound,
        )
    )
    assert estimate.cycles == 4 + device_only.cycles + 3


def test_autoscheduler_and_virtual_backend_share_cost_program(monkeypatch):
    target = build_samsung_target()
    bound = samsung_cost.bind(target)
    trace = MatchTrace(target.name, "synthetic_gemv", [_bound_mac()])
    monkeypatch.setenv("SPMW_DISABLE_REGALLOC", "1")

    chosen = autoschedule(target, trace, cost=bound)[0]
    graph = build_execution_graph(target, trace, chosen, bound)
    direct = bound.evaluate(graph)
    compiled = Compiled(
        target,
        trace,
        [],
        chosen,
        backend="virtual",
        execution_graph=graph,
        cost=bound,
    )
    result = compiled.run()

    # Both bank fibers share the channel bus (costs/samsung.py MAC rule), so
    # the reduction tiles stay on one fiber; since round 37 the estimate
    # includes the first matrix staging from the conductor's schedule, and
    # the placement that stages every tile's second row in the odd fiber
    # (`split_rows`, no exposed row change in its staging) is the cheapest
    # one, as the conductor measures on the paper GEMVs and a held-out shape.
    assert chosen.mode.startswith("split_rows")
    assert chosen.extra.get("tile_rows") == "split_fibers"
    assert chosen.extra.get("issue_order") == "tile_major"
    assert result.cycles == direct.cycles
    assert result.extra["critical_path"] == list(direct.critical_path)
    assert result.extra["model_fingerprint"] == bound.fingerprint


def _gemv_program(rows, reduction):
    from allo.spmw_samsung import (
        SamsungContraction,
        SamsungProgram,
        samsung_target_facts,
    )

    facts = samsung_target_facts(build_samsung_target())
    tiles = -(-reduction // (facts.grf_a_entries * facts.lanes))
    contraction = SamsungContraction(
        matrix_memref="W",
        vector_memref="x",
        acc_memref="acc",
        matrix_role="x",
        vector_role="y",
        reduction_var="k",
        row_var="i",
        column_var=None,
        K=reduction,
        rows_per_item=rows,
        N=1,
        matrix_shape=(rows, reduction),
        vector_shape=(reduction,),
        matrix_k_axis=1,
        vector_k_axis=0,
        row_offset=0,
        output_memref="y",
    )
    return SamsungProgram(
        kind="REDUCE",
        group_id=0,
        start=0,
        end=0,
        binds=((0, "matrix", 0, -1, 0), (2, "drain", 1, -1, 0)),
        facts=facts,
        crf_issue="shared",
        work_items=1,
        contraction=contraction,
        rows=rows,
        k_tiles=tiles,
        tile_fiber=(0,) * tiles,
        rows_per_block=facts.grf_b_entries,
    )


GEOMETRY = {
    "channels": 64,
    "pim_blocks": 8,
    "grf_a": 8,
    "grf_b": 8,
    "lanes": 16,
    "crf_words": 32,
    "banks": 16,
    "cols_per_row": 32,
    "rows": 16384,
}


def test_reduce_plan_pads_rows_to_whole_passes_and_k_to_tiles():
    plan = spmw_samsung.samsung_reduce_plan(_gemv_program(4096, 256), GEOMETRY, 1)
    assert (plan["passes"], plan["slots"], plan["padded_k"]) == (1, 4096, 256)
    plan = spmw_samsung.samsung_reduce_plan(_gemv_program(5000, 100), GEOMETRY, 1)
    assert (plan["passes"], plan["slots"], plan["padded_k"]) == (2, 8192, 128)
    assert "out_tiles 2" in plan["text"] and "k_tiles 1" in plan["text"]


def test_reduce_run_zero_pads_into_physical_slots(monkeypatch, tmp_path):
    captured = {}

    def fake_conductor(root, argv, what):
        a_path, b_path = argv[argv.index("--inputs") + 1].split(",")
        captured["weight_shape"] = np.load(a_path).shape
        captured["vector_shape"] = np.load(b_path).shape
        captured["plan"] = Path(argv[argv.index("--plan") + 1]).read_text()
        out = Path(argv[argv.index("--out") + 1])
        np.zeros(4096 * 16, dtype=np.float16).tofile(out)
        return 17, {"preload": 8, "exec": 6, "readback": 3}, {}, "log"

    monkeypatch.setattr(spmw_samsung, "_invoke_conductor", fake_conductor)
    program = _gemv_program(1024, 1408)
    result = spmw_samsung._samsung_run_reduce(
        program,
        [PIMCmd(type_="EXIT")],
        tmp_path,
        GEOMETRY,
        np.ones((1024, 1408), dtype=np.float16),
        np.ones((1, 1408), dtype=np.float16),
    )

    assert result["cycles"] == 17
    assert captured["weight_shape"] == (4096, 1408)
    assert captured["vector_shape"] == (1, 1408)
    assert "k_tiles 11" in captured["plan"]
    assert result["padding_fraction"] == pytest.approx(1 - 1024 / 4096)
    assert result["rows"].shape == (1, 1024)


def test_staging_shares_partition_the_conductors_staging():
    """Round 38 (reviewer): a work item is charged its own rows' share of the
    first staging, the shares of a kernel's work items partition the
    conductor's staging exactly, equal work gives equal cost, rows scale it,
    and the staging precedes the MAC; the same scope prices the ordered
    program."""
    from allo.perf import CostEvent, ExecutionGraph
    from allo.pim.costs.samsung import (
        STAGE_ROW_CHANGE_CYCLES,
        STAGE_WRITE_CYCLES,
        samsung_staging_cycles,
        samsung_staging_schedule,
        samsung_staging_share,
    )

    target = build_samsung_target()
    bound = samsung_cost.bind(target)

    def staged(rows, k, candidate=None):
        graph = ExecutionGraph()
        event = CostEvent.create(
            "mac",
            target.op("MAC"),
            work_id=(0, 0),
            metrics={
                "loop_extents": (rows, k),
                "reduction_extent": k,
                "lanes": 16,
                "n_fibers": 1,
                # a host-filled GRF_A: the matrix staging alone (the vector
                # staging of a bank fill is `test_vector_staging_shares_...`)
                "candidate": {"grf_residency": {"x": "host"}, **(candidate or {})},
            },
        )
        bound.emit(graph, event)
        labels = [a.label for a in graph.activities]
        assert labels[:2] == ["mac_stage", "mac"], labels
        assert graph.activities[0].id in graph.activities[1].depends_on
        return graph.activities[0].latency_cycles

    # Equal padded bursts (1,024), equal cost; rows scale a bus-bound share
    # linearly; a share too small to fill its bank's row occupancy (8 rows,
    # 64 writes) sits on that floor rather than scaling below it.
    assert staged(64, 256) == staged(32, 512) == staged(128, 128)
    assert staged(1024, 128) == 8 * staged(128, 128)
    G, E = int(target.grf_b.slots), int(target.grf_a.slots)
    small = samsung_staging_share(8, 128, G=G, E=E, lanes=16)
    assert (
        staged(8, 128)
        == small["latency"]
        == max(small["bus_cycles"], small["bank_cycles"])
    )
    assert small["bank_cycles"] > small["bus_cycles"] and small["writes"] == 64
    # The paper GEMV (8192 x 256) as its 128 work items of 64 rows: the shares
    # sum to the conductor's staging (64 channels x 2,048 writes per channel,
    # 32 exposed row changes per channel for the vendor rows, none when the
    # second rows sit in the odd fiber), i.e. to the whole-pass schedule.
    for split in (False, True):
        share = samsung_staging_share(64, 256, G=G, E=E, lanes=16, split_rows=split)
        total_writes = 128 * share["writes"]
        total_exposed = 128 * share["exposed_row_changes"]
        schedule = samsung_staging_schedule(
            2, 2, 1, (0, 1) if split else (0, 0), G, E, 32, 8
        )
        whole = samsung_staging_cycles(schedule)
        assert total_writes == 64 * whole["writes"] == 64 * 2048
        assert (
            total_exposed
            == 64 * whole["exposed_row_changes"]
            == (0 if split else 64 * 32)
        )
        assert (
            share["bus_cycles"]
            == share["writes"] * STAGE_WRITE_CYCLES
            + share["exposed_row_changes"] * STAGE_ROW_CHANGE_CYCLES
        )
    # The split rows remove the exposed changes: a cheaper share, the
    # ranking the conductor confirms (VALIDATION rounds 37-38).
    assert staged(64, 256, {"tile_rows": "split_fibers"}) < staged(64, 256)
    # The ordered program carries the same scope (its own preload pattern).
    temporal = staged(64, 256, {"reduction_program": "temporal"})
    assert (
        temporal
        == samsung_staging_share(64, 256, G=G, E=E, lanes=16, temporal=True)["latency"]
        > 0
    )


def test_vector_staging_shares_partition_the_conductors_vector_staging():
    """Round 132 (S-12): a bank-filled GRF_A (``grf_residency`` ``crf``) is
    charged its blocks' share of the conductor's vector staging
    (``stage_vector``: every PIM block receives the vector's k_tiles x E
    bursts per column before the passes), the shares of a kernel's work
    items of a channel together write what the conductor writes per channel,
    a host-filled GRF_A has no such
    step, columns scale it, and it sits between the matrix staging and the
    MAC in the priced graph."""
    from allo.perf import CostEvent, ExecutionGraph
    from allo.pim.costs.samsung import (
        STAGE_ROW_VISIT_CYCLES,
        STAGE_WRITE_CYCLES,
        samsung_vector_staging_share,
    )

    target = build_samsung_target()
    bound = samsung_cost.bind(target)
    E = int(target.grf_a.slots)

    def emitted(rows, k, residency, extents=None):
        graph = ExecutionGraph()
        event = CostEvent.create(
            "mac",
            target.op("MAC"),
            work_id=(0, 0),
            metrics={
                "loop_extents": extents or (rows, k),
                "reduction_extent": k,
                "lanes": 16,
                "n_fibers": 1,
                "candidate": {"grf_residency": {"x": residency}},
            },
        )
        bound.emit(graph, event)
        return graph

    host = emitted(32, 256, "host")
    assert [a.label for a in host.activities][:2] == ["mac_stage", "mac"]
    crf = emitted(32, 256, "crf")
    labels = [a.label for a in crf.activities]
    assert labels[:3] == ["mac_stage", "mac_vector_stage", "mac"], labels
    stage, vector, mac = crf.activities[:3]
    assert stage.id in vector.depends_on and vector.id in mac.depends_on
    share = samsung_vector_staging_share(256, E=E, lanes=16)
    assert vector.latency_cycles == share["latency"] > 0
    # The replication is per PIM unit, once whatever the pass count: the
    # conductor writes k_tiles (2) x E bursts into each of its 8 blocks per
    # channel (64 x 128 writes on its 64-channel system; design 6.2 measures
    # the same preload increase, 93 cycles, at 4096 x 256 and 8192 x 256),
    # and the eight work items of a declared channel together write that.
    assert share["writes"] == 2 * E == 16
    assert 8 * share["writes"] == 8 * 2 * E
    assert share["exposed_row_changes"] == 0
    assert share["bus_cycles"] == share["writes"] * STAGE_WRITE_CYCLES
    assert share["bank_cycles"] == STAGE_ROW_VISIT_CYCLES + 2 * E * STAGE_WRITE_CYCLES
    assert emitted(64, 256, "crf").activities[1].latency_cycles == vector.latency_cycles
    # Padded tiles cost the same; the bank's row visits grow once the staged
    # columns exceed one row (32 columns = 4 tiles).
    assert share == samsung_vector_staging_share(129, E=E, lanes=16)
    wide = samsung_vector_staging_share(1024, E=E, lanes=16)
    assert wide["writes"] == 8 * E
    assert (
        wide["bank_cycles"] == 2 * STAGE_ROW_VISIT_CYCLES + 8 * E * STAGE_WRITE_CYCLES
    )
    # Every result column stages its own tiles (the conductor stages the
    # vector of every column, resident or restaged).
    two = emitted(32, 256, "crf", extents=(32, 2, 256))
    assert two.activities[1].label == "mac_vector_stage"
    assert (
        two.activities[1].latency_cycles
        == samsung_vector_staging_share(256, E=E, lanes=16, columns=2)["latency"]
        > vector.latency_cycles
    )
    # The bank fill now carries the staging the simulator charges it, so a
    # bank-filled program of the same rows costs more than the host-filled
    # one (design 6.2: CRF fill 4,560 / 4,589 against 4,435 / 4,389 cycles).
    total = lambda graph: sum(a.latency_cycles for a in graph.activities)
    assert total(crf) > total(host)


def test_staging_share_follows_the_geometry_columns_per_row():
    """Round 92 (P-23 W0613 triage): ``samsung_staging_share`` accepted
    ``cols_per_row`` and ignored it, hard-coding the two physical rows a
    64-column run spans at the simulator's 32 burst columns per row.  The
    rows per run now derive from the geometry: the default is unchanged,
    sixteen columns per row makes a run span four rows (three exposed
    changes per run, four row visits), sixty-four removes the exposed change,
    and the split map's half runs follow the same rule."""
    from allo.pim.costs.samsung import (
        STAGE_ROW_CHANGE_CYCLES,
        STAGE_ROW_VISIT_CYCLES,
        STAGE_WRITE_CYCLES,
        samsung_staging_share,
    )

    G, E, lanes = 8, 8, 16  # run = G*E = 64 writes; 256-wide reduction = 2 tiles
    default = samsung_staging_share(64, 256, G=G, E=E, lanes=lanes)
    explicit = samsung_staging_share(64, 256, G=G, E=E, lanes=lanes, cols_per_row=32)
    assert default == explicit
    groups, tiles = 8, 2
    assert default["exposed_row_changes"] == groups * tiles  # two rows per run
    assert default["bank_cycles"] == tiles * (
        2 * STAGE_ROW_VISIT_CYCLES + 64 * STAGE_WRITE_CYCLES
    )
    narrow = samsung_staging_share(64, 256, G=G, E=E, lanes=lanes, cols_per_row=16)
    assert narrow["writes"] == default["writes"]
    assert narrow["exposed_row_changes"] == groups * tiles * 3
    assert narrow["bank_cycles"] == tiles * (
        4 * STAGE_ROW_VISIT_CYCLES + 64 * STAGE_WRITE_CYCLES
    )
    assert narrow["bus_cycles"] == (
        narrow["writes"] * STAGE_WRITE_CYCLES
        + narrow["exposed_row_changes"] * STAGE_ROW_CHANGE_CYCLES
    )
    wide = samsung_staging_share(64, 256, G=G, E=E, lanes=lanes, cols_per_row=64)
    assert wide["exposed_row_changes"] == 0
    assert wide["bank_cycles"] == tiles * (
        STAGE_ROW_VISIT_CYCLES + 64 * STAGE_WRITE_CYCLES
    )
    split_default = samsung_staging_share(
        64, 256, G=G, E=E, lanes=lanes, split_rows=True
    )
    split_narrow = samsung_staging_share(
        64, 256, G=G, E=E, lanes=lanes, split_rows=True, cols_per_row=16
    )
    assert split_default["exposed_row_changes"] == 0
    assert (
        split_narrow["exposed_row_changes"] == groups * tiles * 2
    )  # 32 writes span 2 rows per bank
    assert split_narrow["bank_cycles"] == tiles * (
        2 * STAGE_ROW_VISIT_CYCLES + 32 * STAGE_WRITE_CYCLES
    )
    with pytest.raises(ValueError, match="cols_per_row"):
        samsung_staging_share(64, 256, G=G, E=E, lanes=lanes, cols_per_row=0)
