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
    match.operands = [
        OperandBinding("x", "x", ["k"], memref_type="memref<128xf16>"),
        OperandBinding("y", "W", ["i", "k"], memref_type="memref<8x128xf16>"),
        OperandBinding("acc", "acc", is_loop_carried=True, memref_type="memref<f16>"),
    ]
    match.result_memref_name = "acc"
    match.extra = {
        "accumulator_init": "zero",
        "axis_roles": {"reduction": ("k",), "output": ("i",)},
        "output_sinks": [("y", "i")],
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
    # drain around the MAC; the host moves bracket the device program.
    assert [activity.primitive for activity in graph.activities] == [
        "samsung_hbm_pim/host/SCATTER_BANKS",
        "samsung_hbm_pim/hbm_pim/pseudo_channel/pim/LD_B",
        "samsung_hbm_pim/hbm_pim/pseudo_channel/pim/MAC",
        "samsung_hbm_pim/hbm_pim/pseudo_channel/pim/ST_B",
        "samsung_hbm_pim/host/GATHER_BANKS",
    ]
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
    # the selected placement keeps the reduction on one fiber.
    assert chosen.mode.startswith("bank_row")
    assert chosen.extra.get("n_fibers", 1) == 1
    assert result.cycles == direct.cycles
    assert result.extra["critical_path"] == list(direct.critical_path)
    assert result.extra["model_fingerprint"] == bound.fingerprint


def _gemv_program(rows, reduction):
    from allo.spmw_samsung import SamsungContraction, SamsungProgram, samsung_target_facts

    facts = samsung_target_facts(build_samsung_target())
    tiles = -(-reduction // (facts.grf_a_entries * facts.lanes))
    contraction = SamsungContraction(
        matrix_memref="W", vector_memref="x", acc_memref="acc", matrix_role="x",
        vector_role="y", reduction_var="k", row_var="i", column_var=None,
        K=reduction, rows_per_item=rows, N=1, matrix_shape=(rows, reduction),
        vector_shape=(reduction,), matrix_k_axis=1, vector_k_axis=0, row_offset=0,
        output_memref="y",
    )
    return SamsungProgram(
        kind="REDUCE", group_id=0, start=0, end=0,
        binds=((0, "matrix", 0, -1, 0), (2, "drain", 1, -1, 0)),
        facts=facts, crf_issue="shared", work_items=1, contraction=contraction,
        rows=rows, k_tiles=tiles, tile_fiber=(0,) * tiles,
        rows_per_block=facts.grf_b_entries,
    )


GEOMETRY = {"channels": 64, "pim_blocks": 8, "grf_a": 8, "grf_b": 8, "lanes": 16,
            "crf_words": 32, "banks": 16, "cols_per_row": 32, "rows": 16384}


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
        None, program, [PIMCmd(type_="EXIT")], tmp_path, GEOMETRY,
        np.ones((1024, 1408), dtype=np.float16), np.ones((1, 1408), dtype=np.float16),
    )

    assert result["cycles"] == 17
    assert captured["weight_shape"] == (4096, 1408)
    assert captured["vector_shape"] == (1, 1408)
    assert "k_tiles 11" in captured["plan"]
    assert result["padding_fraction"] == pytest.approx(1 - 1024 / 4096)
    assert result["rows"].shape == (1, 1024)
