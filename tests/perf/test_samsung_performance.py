# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Samsung executable-cost integration tests."""

from types import SimpleNamespace

import numpy as np

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
    match = _mac(work_id)
    match.operands = [
        OperandBinding("x", "x"),
        OperandBinding("y", "W"),
        OperandBinding("acc", "acc", is_loop_carried=True),
    ]
    match.result_memref_name = "acc"
    return match


def test_virtual_graph_includes_shape_aware_host_transfers():
    target = build_samsung_target()
    trace = MatchTrace(target.name, "host_boundary", [_mac((0, 0))])
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
        Placement(placements={}, extra={"n_fibers": 1}),
        bound,
        host_moves=[scatter, gather],
        buffer_metrics={
            "W": {"shape": (16,), "elements": 16, "bytes": 64},
            "out": {"shape": (8,), "elements": 8, "bytes": 32},
        },
    )
    estimate = bound.evaluate(graph)

    assert [activity.primitive for activity in graph.activities] == [
        "samsung_hbm_pim/host/SCATTER_BANKS",
        "samsung_hbm_pim/hbm_pim/pseudo_channel/pim/MAC",
        "samsung_hbm_pim/host/GATHER_BANKS",
    ]
    assert estimate.cycles == 4 + 65 + 3


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

    assert chosen.extra["n_fibers"] == 2
    assert result.cycles == direct.cycles
    assert result.extra["critical_path"] == list(direct.critical_path)
    assert result.extra["model_fingerprint"] == bound.fingerprint


def test_reduce_row_tiling_uses_per_tile_output_extent():
    assert spmw_samsung._samsung_reduce_row_tiling(4096) == (4096, 4096, 1)
    assert spmw_samsung._samsung_reduce_row_tiling(5000) == (8192, 4096, 2)
    assert spmw_samsung._samsung_reduce_row_tiling(8192) == (8192, 4096, 2)

    with np.testing.assert_raises_regex(ValueError, "must be positive"):
        spmw_samsung._samsung_reduce_row_tiling(0)


def test_batched_invoke_zero_pads_to_physical_fabric(monkeypatch, tmp_path):
    captured = {}

    def fake_run(argv, **_kwargs):
        weight = np.load(argv[argv.index("--weight") + 1])
        input_batch = np.load(argv[argv.index("--in") + 1])
        captured["weight_shape"] = weight.shape
        captured["input_shape"] = input_batch.shape
        captured["output_dim"] = argv[argv.index("--output-dim") + 1]
        captured["input_dim"] = argv[argv.index("--input-dim") + 1]
        return SimpleNamespace(
            returncode=0,
            stdout=b"PIM_CYCLES total=17 preload=8 exec=6 readback=3\n",
            stderr=b"",
        )

    monkeypatch.setattr(spmw_samsung.subprocess, "run", fake_run)
    cycles, phases, _stdout = spmw_samsung._samsung_batched_invoke(
        tmp_path / "pim_driver",
        tmp_path,
        [PIMCmd(type_="NOP", loopCounter_=7)],
        np.ones((1024, 1408), dtype=np.float16),
        np.ones((1, 1408), dtype=np.float16),
        1,
        False,
        np,
    )

    assert cycles == 17
    assert phases == {"preload": 8, "exec": 6, "readback": 3}
    assert captured == {
        "weight_shape": (4096, 1536),
        "input_shape": (1, 1536),
        "output_dim": "4096",
        "input_dim": "1536",
    }
