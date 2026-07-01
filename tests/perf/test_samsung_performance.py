# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Samsung executable-cost integration tests."""

from types import SimpleNamespace

from allo.perf import cost, rule
from allo.pim.costs import samsung_cost
from allo.pim.targets import build_samsung_target
from allo.spmw_autoschedule import Placement, autoschedule
from allo.spmw_codegen import Compiled
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


def _evaluate(target, trace, placement, cost_spec=samsung_cost):
    bound = cost_spec.bind(target)
    graph = build_execution_graph(target, trace, placement, bound)
    return graph, bound.evaluate(graph)


def test_target_contains_structure_and_no_cost_model():
    target = build_samsung_target()

    assert target.work_grid() == ([16, 8], 128)
    assert target.unit("pseudo_channel").mapping == [16]
    assert target.unit("pseudo_channel").axes == {"channel": 16}
    assert target.unit("pim").mapping == [8]
    assert target.unit("pim").axes == {"pim": 8}
    assert not hasattr(target, "resource_topology")
    assert not hasattr(target, "timing_library")
    for operation_name in ("MAC", "ADD", "MUL", "MAD", "RELU"):
        operation = target.op(operation_name)
        assert not hasattr(operation, "timing_model")
        assert not hasattr(operation, "resources")


def test_parallelism_comes_from_target_unit_instances():
    target = build_samsung_target()
    trace = MatchTrace(
        target_name=target.name,
        module_name="synthetic_gemv",
        matches=[_mac((0, 0), 0), _mac((0, 1), 1), _mac((1, 0), 2)],
    )
    _graph, estimate = _evaluate(
        target, trace, Placement(placements={}, extra={"n_fibers": 1})
    )
    spans = list(estimate.spans.values())

    assert estimate.cycles == 69
    assert [span.start_cycle for span in spans] == [0, 4, 0]
    assert [span.end_cycle for span in spans] == [65, 69, 65]


def test_cost_source_can_branch_on_candidate_placement():
    target = build_samsung_target()
    trace = MatchTrace(target.name, "synthetic_gemv", [_mac((0, 0))])
    _one_graph, one = _evaluate(
        target, trace, Placement(placements={}, extra={"n_fibers": 1})
    )
    _two_graph, two = _evaluate(
        target, trace, Placement(placements={}, extra={"n_fibers": 2})
    )

    assert one.cycles == 65
    assert two.cycles == 34


def test_full_spatial_grid_overlaps_by_channel_and_pim_axes():
    target = build_samsung_target()
    matches = []
    placements = []
    for channel in range(16):
        for pim in range(8):
            match = _mac((channel, pim), channel * 8 + pim)
            match.func_name = f"gemv_{channel}_{pim}"
            matches.append(match)
            placements.append(Placement(placements={}, extra={"n_fibers": 1}))
    trace = MatchTrace(target.name, "full_grid", matches)

    _graph, estimate = _evaluate(target, trace, placements)

    assert estimate.cycles == 65 + 7 * 4


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


def test_agents_calibrate_by_editing_cost_program_code():
    target = build_samsung_target()

    @cost(target="samsung_hbm_pim")
    def slower_mac(spec):
        channel = spec.unit("pseudo_channel")
        pim = spec.unit("pim")
        operation = spec.op("MAC")

        @rule(operation)
        def mac(event, ctx):
            cycles = event.reduction_extent + 1
            ctx.step(
                cycles=cycles,
                occupy=[
                    ctx.use(operation, cycles=8),
                    ctx.use(pim, cycles=cycles),
                    ctx.use(channel, cycles=8),
                ],
            )

    trace = MatchTrace(target.name, "source_edit", [_mac((0, 0))])
    _graph, estimate = _evaluate(
        target,
        trace,
        Placement(placements={}, extra={"n_fibers": 1}),
        cost_spec=slower_mac,
    )

    assert estimate.cycles == 129
