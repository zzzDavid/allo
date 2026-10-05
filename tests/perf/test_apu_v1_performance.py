# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""APU v1 target, group layout, codegen, and calibrated-cost tests."""

from allo.pim.targets import build_apu_v1_target
from allo.spmw_apu_v1 import APUv1Ctx, _parse_apu_v1_prof_print


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
