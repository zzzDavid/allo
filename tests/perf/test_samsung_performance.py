# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from allo.perf import ParameterValue
from allo.pim.performance import SAMSUNG_BASE_PROFILE, virtual_target
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


def test_target_declares_instanced_resources_and_cycle_models():
    target = build_samsung_target()

    command = target.resource("command_bus")
    alu = target.resource("alu")

    assert target.has_performance_model
    assert command.instances == 16
    assert command.pipelined
    assert alu.instances == 16 * 8
    assert "samsung.mac" in target.timing_library.models


def test_samsung_parallelism_comes_from_resource_instances():
    target = build_samsung_target()
    trace = MatchTrace(
        target_name=target.name,
        module_name="synthetic_gemv",
        matches=[_mac((0, 0), 0), _mac((0, 1), 1), _mac((1, 0), 2)],
    )
    graph = build_execution_graph(
        target, trace, Placement(placements={}, extra={"n_fibers": 1})
    )

    estimate = virtual_target(target).evaluate(graph)
    spans = list(estimate.spans.values())

    # Each reduction costs ceil(128/8)*4 + one JUMP = 65 cycles. PIMs on
    # different channels start together; two PIMs on one channel use distinct
    # ALUs but share a pipelined command bus and therefore issue 4 cycles apart.
    assert estimate.cycles == 69
    assert [span.start_cycle for span in spans] == [0, 4, 0]
    assert [span.end_cycle for span in spans] == [65, 69, 65]


def test_fiber_schedule_changes_analytical_operation_duration():
    target = build_samsung_target()
    trace = MatchTrace(
        target_name=target.name,
        module_name="synthetic_gemv",
        matches=[_mac((0, 0))],
    )
    one = build_execution_graph(
        target, trace, Placement(placements={}, extra={"n_fibers": 1})
    )
    two = build_execution_graph(
        target, trace, Placement(placements={}, extra={"n_fibers": 2})
    )

    assert virtual_target(target).evaluate(one).cycles == 65
    assert virtual_target(target).evaluate(two).cycles == 34


def test_replicated_function_suffixes_form_one_parallel_kernel():
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

    graph = build_execution_graph(target, trace, placements)
    estimate = virtual_target(target).evaluate(graph)

    # Eight PIMs per channel issue four cycles apart; all 16 channels overlap.
    assert estimate.cycles == 65 + 7 * 4


def test_autoscheduler_and_virtual_backend_share_candidate_graph(monkeypatch):
    target = build_samsung_target()
    trace = MatchTrace(
        target_name=target.name,
        module_name="synthetic_gemv",
        matches=[_bound_mac()],
    )
    monkeypatch.setenv("SPMW_DISABLE_REGALLOC", "1")

    chosen = autoschedule(target, trace)[0]
    graph = build_execution_graph(target, trace, chosen)
    direct = virtual_target(target).evaluate(graph)
    compiled = Compiled(
        target, trace, [], chosen, backend="virtual", execution_graph=graph
    )
    result = compiled.run()

    assert chosen.extra["n_fibers"] == 2
    assert result.cycles == direct.cycles == 100
    assert result.extra["critical_path"] == list(direct.critical_path)
    assert result.extra["model_fingerprint"] == direct.model_fingerprint


def test_compiled_virtual_run_uses_explicit_cost_profile():
    target = build_samsung_target()
    trace = MatchTrace(target.name, "profile_swap", [_mac((0, 0))])
    placement = Placement(placements={}, extra={"n_fibers": 1})
    graph = build_execution_graph(target, trace, placement)
    profile = SAMSUNG_BASE_PROFILE.overlay(
        "samsung_hbm_pim/slower-mac",
        {"mac_cycles": ParameterValue(8, provenance="measured")},
    )
    performance_model = virtual_target(target, profile)
    compiled = Compiled(
        target,
        trace,
        [],
        placement,
        backend="virtual",
        execution_graph=graph,
        performance_model=performance_model,
    )

    result = compiled.run()

    assert result.cycles == 129
    assert result.extra["model_fingerprint"] == profile.fingerprint()
