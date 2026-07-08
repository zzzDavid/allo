# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""APU v1 target, group layout, codegen, and calibrated-cost tests."""

from allo.perf import CostEvent, ExecutionGraph
from allo.pim.apu_v1_program import _parse_scalar_profile
from allo.pim.costs import apu_v1_cost
from allo.pim.costs.apu_v1 import (
    GROUPED_BATCH_CONTROL_CRUN,
    GROUPED_KERNEL_STARTUP_CRUN,
)
from allo.pim.targets import build_apu_v1_target
from allo.spmw_autoschedule import _apu_v1_enumerate, derive_layout_properties
from allo.spmw_codegen import APUv1Ctx, _parse_apu_v1_prof_print
from allo.spmw_match import MatchedOp, OperandBinding


def _mac(rows=60, reduction=80, groups=256):
    return MatchedOp(
        target_op_name="MAC",
        func_name="gemm",
        work_id=(0,),
        enclosing_loops=[
            ("i", "0", str(rows), 1),
            ("k", "0", str(reduction), 1),
        ],
        operands=[
            OperandBinding("x", "A"),
            OperandBinding("y", "B"),
            OperandBinding("acc", "out", is_loop_carried=True),
        ],
        result_memref_name="out",
        op_range=("begin", "end"),
        extra={
            "spmw_group_count": groups,
            "coalesced_spmw_axis": "group",
        },
    )


def _estimate(target, primitive, **metrics):
    graph = ExecutionGraph("apu_v1_synthetic")
    bound = apu_v1_cost.bind(target)
    bound.emit(graph, CostEvent.create("event", primitive, metrics=metrics))
    return bound.evaluate(graph)


def test_target_describes_gvml_resources_without_timing():
    target = build_apu_v1_target()

    assert target.work_grid() == ([4], 4)
    assert target.unit("apuc").axes == {"apuc": 4}
    assert target.l4.geometry["size_bytes"] == 14 * 2**30
    assert target.l4.capacity == 4
    assert target.vr0.axes == {"lane": 32768}
    assert (target.vr0.lanes, target.vr0.width, target.vr0.slots) == (32768, 16, 1)
    assert (target.vmrs.lanes, target.vmrs.width, target.vmrs.slots) == (
        32768,
        16,
        48,
    )
    assert target.index_vr.slots == 1
    assert (target.markers.width, target.markers.slots) == (1, 8)
    assert {target.unit(name).capacity for name in ("arc", "seu", "dma")} == {1}
    assert not hasattr(target, "timing_library")


def test_group_layout_maps_contiguous_reductions_to_vr_lanes():
    target = build_apu_v1_target()
    placement = _apu_v1_enumerate(target, [_mac()])[0]
    properties = derive_layout_properties(target, placement)

    assert placement.mode == "grouped_f16"
    assert properties["group_size"] == 128
    assert properties["groups_per_vr"] == 256
    assert placement.layout.apply(group=0, lane_in_group=1) == (1,)
    assert placement.layout.apply(group=1, lane_in_group=0) == (128,)
    assert (
        placement.layout.image_size(
            varying_inputs=("group", "lane_in_group"), output_dims=("vr_lane",)
        )
        == 32768
    )


def test_scalar_mapping_eight_selects_eight_4k_groups():
    target = build_apu_v1_target()
    placement = _apu_v1_enumerate(target, [_mac(groups=8)])[0]
    properties = derive_layout_properties(target, placement)

    assert properties["groups_per_vr"] == 8
    assert properties["group_size"] == 4096
    assert placement.layout.apply(group=1, lane_in_group=0) == (4096,)


def test_real_device_f16_costs_and_group_reduction_are_composed():
    target = build_apu_v1_target()
    candidate = {"group_size": 128, "groups_per_vr": 256}

    add = _estimate(target, target.op("ADD"), iterations=32768)
    mac = _estimate(
        target,
        target.op("MAC"),
        iterations=60 * 80,
        reduction_extent=80,
        candidate=candidate,
    )
    two_batches = _estimate(
        target,
        target.op("MAC"),
        iterations=300 * 80,
        reduction_extent=80,
        candidate=candidate,
    )

    assert add.cycles == 264
    assert mac.cycles == (
        GROUPED_KERNEL_STARTUP_CRUN + GROUPED_BATCH_CONTROL_CRUN + 254 + 5308
    )
    assert two_batches.cycles == (
        GROUPED_KERNEL_STARTUP_CRUN + 2 * (GROUPED_BATCH_CONTROL_CRUN + 254 + 5308)
    )


def test_micro_vector_and_memory_primitives_are_structural_and_costed():
    target = build_apu_v1_target()

    assert target.l3.geometry["size_bytes"] == 1 << 20
    assert target.l2.geometry["size_bytes"] == 64 << 10
    assert target.op("XOR_16").matchable is False
    assert target.op("LOOKUP_16").matchable is False

    xor = _estimate(target, target.op("XOR_16"), count=10)
    lookup = _estimate(target, target.op("LOOKUP_16"), table_size=32)
    l4_l3 = _estimate(target, target.move("DMA_L4_TO_L3"), bytes=1024)

    assert xor.cycles == 120
    assert lookup.cycles == 858
    assert l4_l3.cycles == 41359


def test_codegen_uses_native_f16_group_operations():
    target = build_apu_v1_target()
    ctx = APUv1Ctx(target)
    ctx.group_size = 128

    ctx.emit_l4_to_vr("x", target.vr0, 0)
    ctx.emit_l4_to_vr("y", target.vr1, 1)
    ctx.emit_grouped_f16_mac(target.vr2, target.vr0, target.vr1)
    ctx.emit_vr_to_l4("acc", target.vr2, 2)

    source = "\n".join(ctx.cmds)
    assert "gvml_mul_f16(mac_tmp_vr, vr0, vr1);" in source
    assert "GVML_P2_128, GVML_P2_1" in source
    assert "gvml_lookup_16" not in source
    assert ctx.iter_vr_aliases() == [
        ("vr0", "GVML_VR16_0"),
        ("vr1", "GVML_VR16_1"),
        ("vr2", "GVML_VR16_2"),
        ("mac_tmp_vr", "GVML_VR16_3"),
        ("reduce_tmp_vr", "GVML_VR16_4"),
    ]


def test_profile_parser_uses_latest_parallel_batch_makespan():
    text = "\n".join(
        [
            "ARCT[0]: total - crun:3609445",
            "ARCT[0]: total - crun:628100",
            "ARCT[1]: total - crun:628182",
            "ARCT[2]: total - crun:625945",
            "ARCT[3]: total - crun:626624",
        ]
    )
    assert _parse_apu_v1_prof_print(text) == 628182


def test_scalar_profile_parser_uses_newest_total():
    text = "\n".join(
        [
            "ARCT[0]: total - crun:466141607",
            "ARCT[0]: total - crun:11598092",
        ]
    )
    assert _parse_scalar_profile(text) == 11598092
