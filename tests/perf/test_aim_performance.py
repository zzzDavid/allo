# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SK hynix AiM target/cost/programming-abstraction integration tests."""

from allo.pim.costs import aim_cost
from allo.pim.targets import build_aim_target
from allo.spmw_autoschedule import Placement, autoschedule
from allo.spmw_codegen import AimCtx
from allo.spmw_match import MatchTrace, MatchedOp, OperandBinding
from allo.spmw_plan import build_execution_graph


def _mac(work_id=(0, 0, 0), index=0, *, bound=False):
    match = MatchedOp(
        target_op_name="MAC",
        func_name="gemv",
        work_id=work_id,
        enclosing_loops=[("k", "0", "256", 1)],
        operands=[],
        result_memref_name=None,
        op_range=(f"begin{index}", f"end{index}"),
    )
    if bound:
        match.operands = [
            OperandBinding("x", "W"),
            OperandBinding("y", "x"),
            OperandBinding("acc", "out", is_loop_carried=True),
        ]
        match.result_memref_name = "out"
    return match


def _evaluate(target, matches, placements):
    trace = MatchTrace(target.name, "synthetic_aim", matches)
    bound = aim_cost.bind(target)
    graph = build_execution_graph(target, trace, placements, bound)
    return graph, bound.evaluate(graph)


def test_target_is_structural_and_matches_simulator_topology():
    target = build_aim_target()

    assert target.work_grid() == ([32, 4, 4], 512)
    assert target.unit("channel").axes == {"channel": 32}
    assert target.unit("bank_group").axes == {"bank_group": 4}
    assert target.unit("bank").axes == {"bank": 4}
    assert target.banks.geometry == {
        "banks": 16,
        "bank_groups": 4,
        "rows": 16384,
        "cols": 1024,
        "width": 16,
    }
    assert target.gb.geometry == {"entries": 64, "width": 256}
    assert target.gpr.geometry["size_bytes"] == 512 * 1024
    assert target.af_lut.geometry == {
        "banks": 16,
        "rows": 1,
        "cols": 1024,
        "width": 16,
    }
    assert target.op("MAC").owner is target.unit("bank")
    assert target.op("MAC_ABK").owner is target.unit("channel")
    assert target.op("MUL").owner is target.unit("bank_group")
    assert target.op("MAC_ABK").matchable is False
    assert not hasattr(target, "timing_library")


def test_cost_distinguishes_single_bank_and_all_bank_mac():
    target = build_aim_target()
    match = _mac()
    single = Placement(placements={}, extra={"operation_name": "MAC"})
    all_bank = Placement(placements={}, extra={"operation_name": "MAC_ABK"})

    _single_graph, single_estimate = _evaluate(target, [match], single)
    _all_graph, all_estimate = _evaluate(target, [match], all_bank)

    # 256 scalars are 16 SBK columns, but one 16-bank ABK column.
    assert single_estimate.cycles == 87
    assert all_estimate.cycles == 57


def test_parallelism_follows_channel_and_bank_instances():
    target = build_aim_target()
    matches = [_mac((0, 0, 0), 0), _mac((0, 0, 1), 1), _mac((1, 0, 0), 2)]
    placement = Placement(placements={}, extra={"operation_name": "MAC"})

    _graph, estimate = _evaluate(target, matches, placement)
    spans = list(estimate.spans.values())

    assert estimate.cycles == 119
    assert [span.start_cycle for span in spans] == [0, 32, 0]
    assert [span.end_cycle for span in spans] == [87, 119, 87]


def test_autoscheduler_selects_channel_scoped_mac_abk():
    target = build_aim_target()
    trace = MatchTrace(target.name, "synthetic_aim", [_mac(bound=True)])
    bound = aim_cost.bind(target)

    chosen = autoschedule(target, trace, cost=bound)[0]

    assert chosen.mode == "all_bank"
    assert chosen.extra["operation_name"] == "MAC_ABK"
    assert chosen.placements["W"] is target.banks
    assert chosen.placements["x"] is target.gb
    assert chosen.placements["out"] is target.mac_reg


def test_codegen_materializes_channel_mask_bank_index_and_column_count():
    target = build_aim_target()

    abk_ctx = AimCtx(target)
    abk_ctx._active_work_id = (3,)
    abk_ctx._active_placement = Placement(
        placements={}, extra={"operation_name": "MAC_ABK"}
    )
    target.op("MAC_ABK").emit(target.banks, target.gb, target.mac_reg, abk_ctx)
    abk_ctx.after_match(_mac((3,)), 1)
    assert abk_ctx.cmds == ["AiM MAC_ABK 1 8 0"]

    sbk_ctx = AimCtx(target)
    sbk_ctx._active_work_id = (3, 2, 1)
    sbk_ctx._active_placement = Placement(
        placements={}, extra={"operation_name": "MAC"}
    )
    operation = target.op("MAC")
    operation.emit(operation.src[0], target.gb, target.mac_reg, sbk_ctx)
    sbk_ctx.after_match(_mac((3, 2, 1)), 1)
    assert sbk_ctx.cmds == ["AiM MAC_SBK 16 8 9 0"]
