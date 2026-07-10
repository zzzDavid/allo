# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SK hynix AiM target/cost/programming-abstraction integration tests."""

import warnings

import pytest

from allo.pim.costs import aim_cost
from allo.pim.targets import build_aim_target
from allo.spmw_autoschedule import (
    MatcherWorkScope,
    Placement,
    _aim_enumerate,
    _aim_layout_candidates,
    autoschedule,
)
from allo.spmw_codegen import AimCtx, _aim_runtime_segments, compile_for_target
from allo.spmw_linear_layout import LinearLayout
from allo.spmw_match import MatchTrace, MatchedOp, OperandBinding
from allo.spmw_match_engine import batch_dim
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


def _scoped_codegen_mac(func_name, work_id):
    match = _mac(work_id, bound=True)
    match.func_name = func_name
    match.enclosing_loops = [("i", "0", "1", 1), ("k", "0", "16", 1)]
    match.operands[0].indices = ["i", "k"]
    match.operands[0].memref_type = "memref<32x16xbf16>"
    match.operands[1].indices = ["k"]
    match.operands[1].memref_type = "memref<16xbf16>"
    match.extra["spmw_work_scope"] = MatcherWorkScope(
        group_id=0,
        work_id=work_id,
        group_shape=(2,),
    )
    return match


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
    assert isinstance(chosen.layout, LinearLayout)
    assert (
        chosen.layout.image_size(varying_inputs=("lane_bank",), output_dims=("bank",))
        == 16
    )
    assert (
        chosen.layout.conflict_count(bank_dims=("bank",), varying_inputs=("lane_bank",))
        == 0
    )


def test_linear_layout_is_load_bearing_for_aim_bank_scope():
    target = build_aim_target()
    single, all_bank = _aim_layout_candidates(target, [_mac(bound=True)])

    assert (
        single.layout.image_size(varying_inputs=("lane_bank",), output_dims=("bank",))
        == 1
    )
    assert single.extra == {
        "operation_name": "MAC",
        "bank_fanout": 1,
        "bank_conflicts": 15,
    }
    assert (
        all_bank.layout.image_size(varying_inputs=("lane_bank",), output_dims=("bank",))
        == 16
    )
    assert all_bank.extra == {
        "operation_name": "MAC_ABK",
        "bank_fanout": 16,
        "bank_conflicts": 0,
    }

    # Duplicated metadata cannot override the F2 map: the execution planner
    # re-derives fanout and primitive selection from Placement.layout.
    all_bank.extra.update(operation_name="MAC", bank_fanout=1)
    graph, _estimate = _evaluate(target, [_mac(bound=True)], all_bank)
    mac = next(
        activity for activity in graph.activities if "MAC_ABK" in activity.primitive
    )
    assert mac.latency_cycles == 57


def test_active_aim_enumerator_excludes_unrealized_single_bank_layout():
    target = build_aim_target()

    candidates = _aim_enumerate(target, [_mac(bound=True)])

    assert len(candidates) == 1
    assert candidates[0].mode == "all_bank"
    assert candidates[0].extra["operation_name"] == "MAC_ABK"


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


def test_whole_program_codegen_is_invariant_to_digit_ending_symbol_renames():
    target = build_aim_target()

    def compile_names(names, parsed_work_ids):
        matches = [
            _scoped_codegen_mac(names[0], (0,)),
            _scoped_codegen_mac(names[1], (1,)),
        ]
        for match, parsed_work_id in zip(matches, parsed_work_ids):
            match.work_id = parsed_work_id
        trace = MatchTrace(
            target.name,
            "aim_codegen_rename",
            matches,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return compile_for_target(target, trace, layout=Placement(placements={}))

    original = compile_names(("gemv2_0", "gemv2_1"), ((0,), (1,)))
    renamed = compile_names(("phase_2024", "answer42"), ((2024,), (42,)))

    assert renamed.cmds == original.cmds
    assert renamed.cmds.count("# TENON_GEMV 32 16 repeat=1") == 1
    assert renamed.cmds[-1] == "AiM EOC"


def test_aim_runtime_requires_codegen_owned_trailing_eoc():
    with pytest.raises(ValueError, match="exactly one trailing EOC"):
        _aim_runtime_segments(
            (
                "# TENON_GEMV 16 16 repeat=1",
                "AiM MAC_ABK 1 4294967295 0",
            )
        )

    assert _aim_runtime_segments(
        (
            "# TENON_GEMV 16 16 repeat=1",
            "AiM MAC_ABK 1 4294967295 0",
            "AiM EOC",
        )
    ) == (
        (
            "TENON_GEMV 16 16 repeat=1",
            1,
            ("AiM MAC_ABK 1 4294967295 0", "AiM EOC"),
        ),
    )


def test_codegen_materializes_complete_batched_gemv_from_mlir_shapes():
    target = build_aim_target()
    match = MatchedOp(
        target_op_name="MAC",
        func_name="gemm_0",
        work_id=(0,),
        enclosing_loops=[
            ("i", "0", "32", 1),
            ("j", "0", "1100", 1),
            ("k", "0", "1200", 1),
        ],
        operands=[
            OperandBinding(
                "x", "A", ["row", "k"], memref_type="memref<1000x1200xbf16>"
            ),
            OperandBinding("y", "B", ["k", "j"], memref_type="memref<1200x1100xbf16>"),
            OperandBinding("acc", "acc", [], is_loop_carried=True),
        ],
        result_memref_name="acc",
        op_range=("begin", "end"),
    )
    batch_var, batches = batch_dim(match)
    match.extra.update(batch_loop_var=batch_var, batch_dim=batches)

    ctx = AimCtx(target)
    ctx.emit_gemv(match)

    assert ctx.cmds[0] == "# TENON_GEMV 1000 1200 repeat=1100"
    assert ctx.cmds.count("AiM WR_ABK 4 1 0") == 1
    assert "AiM MAC_ABK 75 4294967295 62" in ctx.cmds
    assert ctx.cmds[-1] == "AiM RD_MAC 8 4294967295"
    assert len(ctx.cmds) == 130
