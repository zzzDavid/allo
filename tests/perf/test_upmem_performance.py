# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""UPMEM structural-target and executable-cost integration tests."""

from allo.perf import CostEvent, ExecutionGraph
from allo.pim.costs import upmem_cost
from allo.pim.targets import build_upmem_target


def _evaluate_event(target, primitive, **metrics):
    graph = ExecutionGraph("synthetic_upmem_event")
    bound = upmem_cost.bind(target)
    metrics.setdefault("candidate", {"tasklet_fanout": 11})
    bound.emit(graph, CostEvent.create("event", primitive, metrics=metrics))
    return bound.evaluate(graph)


def test_target_matches_one_rank_upmem_topology_without_embedded_cost():
    target = build_upmem_target()

    assert target.work_grid() == ([64, 24], 1536)
    assert target.unit("rank").axes == {}
    assert target.unit("dpu").axes == {"dpu": 64}
    assert target.unit("tasklet").axes == {"tasklet": 24}
    assert target.mram.geometry["size_bytes"] == 64 * 1024 * 1024
    assert target.wram.geometry["size_bytes"] == 64 * 1024
    assert target.iram.geometry["size_bytes"] == 24 * 1024
    assert target.atomic.geometry["size_bytes"] == 256
    assert (target.gprs.slots, target.gprs.width) == (24, 32)
    assert target.move("SCATTER_MRAM").verb.name == "scatter"
    assert target.move("BCAST_MRAM").verb.name == "broadcast"
    assert target.move("GATHER_MRAM").verb.name == "gather"
    assert {
        "ADD",
        "SUB",
        "MUL",
        "DIV",
        "MAC",
        "SQRT",
        "CMP",
        "SELECT",
        "MIN",
        "MAX",
        "BRANCH",
    }.issubset({op.name for unit in target._walk() for op in unit.ops.values()})
    assert not hasattr(target, "timing_library")


def test_revolver_cost_saturates_at_eleven_tasklets():
    target = build_upmem_target()
    one = _evaluate_event(
        target,
        target.op("MAC"),
        iterations=32,
        candidate={"tasklet_fanout": 1},
    )
    eleven = _evaluate_event(
        target,
        target.op("MAC"),
        iterations=32,
        candidate={"tasklet_fanout": 11},
    )
    twenty_four = _evaluate_event(
        target,
        target.op("MAC"),
        iterations=32,
        candidate={"tasklet_fanout": 24},
    )

    assert one.cycles == 32 * 33 * 11 + 13
    assert eleven.cycles == 32 * 33 + 13
    assert twenty_four.cycles == eleven.cycles


def test_general_operations_distinguish_native_integer_and_software_float():
    target = build_upmem_target()

    integer_add = _evaluate_event(
        target, target.op("ADD"), iterations=8, numeric_kind="integer"
    )
    floating_add = _evaluate_event(
        target, target.op("ADD"), iterations=8, numeric_kind="float32"
    )
    integer_div = _evaluate_event(
        target, target.op("DIV"), iterations=2, numeric_kind="integer"
    )
    floating_div = _evaluate_event(
        target, target.op("DIV"), iterations=2, numeric_kind="float32"
    )

    assert integer_add.cycles == 8 + 13
    assert floating_add.cycles == 8 * 96 + 13
    assert integer_div.cycles == 2 * 48 + 13
    assert floating_div.cycles == 2 * 256 + 13


def test_summarized_instruction_count_overrides_analytical_defaults():
    target = build_upmem_target()

    estimate = _evaluate_event(
        target,
        target.op("SQRT"),
        iterations=10_000,
        numeric_kind="float32",
        instruction_count=37,
    )
    wram = _evaluate_event(
        target,
        target.move("LD_WRAM"),
        iterations=10_000,
        instruction_count=19,
    )

    assert estimate.cycles == 37 + 13
    assert wram.cycles == 19 + 13


def test_compare_select_minmax_and_branch_are_explicit_costed_primitives():
    target = build_upmem_target()

    integer_cmp = _evaluate_event(target, target.op("CMP"), iterations=4)
    floating_cmp = _evaluate_event(
        target, target.op("CMP"), iterations=4, is_float=True
    )
    integer_min = _evaluate_event(target, target.op("MIN"), iterations=4)
    floating_max = _evaluate_event(
        target, target.op("MAX"), iterations=4, element_type="f32"
    )
    select = _evaluate_event(target, target.op("SELECT"), iterations=4)
    branch = _evaluate_event(target, target.op("BRANCH"), iterations=4)

    assert integer_cmp.cycles == 4 + 13
    assert floating_cmp.cycles == 4 * 24 + 13
    assert integer_min.cycles == 4 * 3 + 13
    assert floating_max.cycles == 4 * 26 + 13
    assert select.cycles == 4 * 2 + 13
    assert branch.cycles == 4 + 13
