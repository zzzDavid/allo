# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Samsung executable-cost integration tests."""

import warnings
from types import SimpleNamespace

import numpy as np

from allo.perf import cost, rule
from allo.pim.costs import samsung_cost
from allo.pim.targets import build_samsung_target
from allo.spmw_autoschedule import (
    MatcherWorkScope,
    Placement,
    _samsung_enumerate,
    autoschedule,
)
from allo import spmw_codegen
from allo.spmw_codegen import Compiled, PIMCmd, SamsungCtx
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


def _scoped_bound_mac(func_name, group_id, work_id, group_shape):
    match = _bound_mac(work_id)
    match.func_name = func_name
    match.extra["spmw_work_scope"] = MatcherWorkScope(
        group_id=group_id,
        work_id=work_id,
        group_shape=group_shape,
    )
    return match


def _shared_bank_row(target, match):
    return next(
        candidate
        for candidate in _samsung_enumerate(target, [match])
        if candidate.mode == "bank_row+crf_shared"
        and candidate.extra.get("grf_residency", {}).get("x") == "crf"
        and candidate.extra.get("stage_resident") is False
    )


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


def test_shared_crf_codegen_is_invariant_to_digit_ending_symbol_renames():
    target = build_samsung_target()

    def compile_names(names, parsed_work_ids):
        matches = [
            _scoped_bound_mac(names[0], 0, (0, 0), (1, 2)),
            _scoped_bound_mac(names[1], 0, (0, 1), (1, 2)),
            _scoped_bound_mac(names[2], 1, (0, 0), (1, 2)),
            _scoped_bound_mac(names[3], 1, (0, 1), (1, 2)),
        ]
        for match, parsed_work_id in zip(matches, parsed_work_ids):
            match.work_id = parsed_work_id
        trace = MatchTrace(target.name, "shared_crf_rename", matches)
        layout = _shared_bank_row(target, matches[0])
        return spmw_codegen.compile_for_target(target, trace, layout=layout)

    original = compile_names(
        ("gemv2_0_0", "gemv2_0_1", "tail7_0_0", "tail7_0_1"),
        ((0, 0), (0, 1), (0, 0), (0, 1)),
    )
    renamed = compile_names(
        ("phase_2024", "phase_2025", "answer42", "answer43"),
        ((2024,), (2025,), (42,), (43,)),
    )

    assert renamed.cmds == original.cmds
    assert renamed.host_schedule == original.host_schedule
    assert len(original.host_schedule) == 4


def test_each_matcher_group_materializes_its_own_crf_issue_decision():
    target = build_samsung_target()
    first = _scoped_bound_mac("first", 0, (0, 0), (1, 1))
    second = _scoped_bound_mac("second", 1, (0, 0), (1, 1))
    trace = MatchTrace(target.name, "mixed_crf_issue", [first, second])
    first_candidates = _samsung_enumerate(target, [first])
    second_candidates = _samsung_enumerate(target, [second])
    shared = next(
        candidate
        for candidate in first_candidates
        if candidate.mode == "bank_row+crf_shared"
        and candidate.extra.get("stage_resident") is False
    )
    per_workid = next(
        candidate
        for candidate in second_candidates
        if candidate.mode == "bank_row+crf_per_workid"
        and candidate.extra.get("stage_resident") is False
    )

    compiled = spmw_codegen.compile_for_target(
        target,
        trace,
        layout=[shared, per_workid],
    )

    assert compiled.host_schedule == [spmw_codegen.HostTrigger((0, 0), 1)]


def test_work_grid_validation_is_invariant_to_digit_ending_symbol_renames():
    target = build_samsung_target()

    def validate(names, parsed_work_ids):
        matches = [
            _scoped_bound_mac(names[0], 0, (0, 0), (1, 2)),
            _scoped_bound_mac(names[1], 0, (0, 1), (1, 2)),
        ]
        for match, parsed_work_id in zip(matches, parsed_work_ids):
            match.work_id = parsed_work_id
        trace = MatchTrace(
            target.name,
            "work_grid_rename",
            matches,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = spmw_codegen._check_work_grid(target, trace, auto_fill=False)
        return result, tuple(str(item.message) for item in caught)

    original = validate(("gemv2_0_0", "gemv2_0_1"), ((0, 0), (0, 1)))
    renamed = validate(("phase_2024", "answer42"), ((2024,), (42,)))

    assert renamed == original
    assert original[0] == 2
    assert len(original[1]) == 1


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


def test_autoscheduler_scores_with_final_host_and_buffer_context(monkeypatch):
    target = build_samsung_target()
    trace = MatchTrace(target.name, "host_context", [_bound_mac((0, 0))])
    host_moves = (object(),)
    buffer_metrics = {"W": {"bytes": 128}}
    observed = []

    class FakeCost:
        fingerprint = "host-context-test"

        def evaluate(self, graph):
            return type("Estimate", (), {"cycles": graph[1]})()

    def fake_graph(
        _target,
        _trace,
        placement,
        _cost,
        *,
        host_moves=(),
        buffer_metrics=None,
    ):
        observed.append((host_moves, buffer_metrics))
        return placement, int(placement.extra.get("n_fibers", 1))

    monkeypatch.setattr("allo.spmw_plan.build_execution_graph", fake_graph)

    autoschedule(
        target,
        trace,
        FakeCost(),
        host_moves=host_moves,
        buffer_metrics=buffer_metrics,
    )

    assert observed
    assert all(item == (host_moves, buffer_metrics) for item in observed)


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


def test_samsung_drain_emits_required_eight_cycle_hold():
    ctx = SamsungCtx(build_samsung_target())

    ctx.drain()

    assert ctx.cmds == [PIMCmd(type_="NOP", loopCounter_=7)]


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

    monkeypatch.setattr(spmw_codegen.subprocess, "run", fake_run)
    cycles, phases, _stdout = spmw_codegen._samsung_batched_invoke(
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


def test_batched_runner_does_not_program_host_fill(monkeypatch, tmp_path):
    captured = {}

    def fake_invoke(_driver, _root, commands, *_args, **_kwargs):
        captured["types"] = [command.type_ for command in commands]
        return 11, {"preload": 5, "exec": 4, "readback": 2}, "raw"

    monkeypatch.setattr(spmw_codegen, "_pimsim_root", lambda: tmp_path)
    (tmp_path / "pim_driver").touch()
    monkeypatch.setattr(spmw_codegen, "_samsung_batched_invoke", fake_invoke)
    target = build_samsung_target()
    trace = MatchTrace(target.name, "batched", [])
    compiled = Compiled(
        target,
        trace,
        [
            PIMCmd(type_="FILL", dst_="GRF_A", src0_="EVEN_BANK"),
            PIMCmd(
                type_="MAC",
                dst_="GRF_B",
                src0_="GRF_A",
                src1_="EVEN_BANK",
                isAuto_=1,
            ),
            PIMCmd(type_="NOP", loopCounter_=7),
        ],
        Placement(placements={}),
    )

    result = spmw_codegen._run_samsung_batched(
        compiled,
        np.ones((128, 256), dtype=np.float16),
        np.ones((1, 256), dtype=np.float16),
        compare_native=False,
    )

    assert result.cycles == 11
    assert captured["types"] == ["MAC", "NOP"]
