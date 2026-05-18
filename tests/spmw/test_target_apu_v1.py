# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for the GSI APU v1 (Gemini 1) target spec.

Per spec 003 §F.5: target build, ctx MAC -> lookup+add expansion, and
the kernel_cycles cost factory returning a callable for an empty trace.

Also pins the SV-vs-SV-lookup cost ordering called out in task 006:
the MAC (sv-lookup) opcode must price lower than a raw MUL (sv mode)
under the same loop nest, so autoschedule's argmin prefers MAC.
"""

from __future__ import annotations

import allo
from allo.spmw_autoschedule import autoschedule
from allo.spmw_codegen import APUv1Ctx
from allo.spmw_match import MatchTrace, MatchedOp, OperandBinding
from allo.spmw_target import Memory, MemoryRef, Register

from _fixtures import build_apu_v1_target


def _synthetic_mac_trace(target_op_name: str = "MAC") -> MatchTrace:
    """Single-match GEMV-shaped trace for APU v1.

    Outer loop covers the work-item tile rows (`M_tile=16`); the
    innermost K loop is the bit-serial-folded dimension.
    """
    return MatchTrace(
        target_name="apu_v1",
        module_name="synthetic",
        matches=[
            MatchedOp(
                target_op_name=target_op_name,
                func_name="gemv_0",
                work_id=(0,),
                enclosing_loops=[
                    ("%arg0", "0", "16", 1),
                    ("%arg1", "0", "1024", 1),
                ],
                operands=[
                    OperandBinding(role="x", memref_name="local_W"),
                    OperandBinding(role="y", memref_name="local_x"),
                    OperandBinding(
                        role="acc", memref_name="acc", is_loop_carried=True
                    ),
                ],
                result_memref_name="acc",
                op_range=("%a", "%b"),
            )
        ],
    )


# --------------------------------------------------------------------- #
# Target build
# --------------------------------------------------------------------- #


def test_apu_v1_target_builds():
    target = build_apu_v1_target()
    assert target.name == "apu_v1"
    # device -> one APUC-group child (the @unit named `apuc`).
    assert len(target.root.children) == 1
    # Memories/registers resolve via the flat handle map.
    assert isinstance(target.l4, Memory)
    assert isinstance(target.l1, Memory)
    assert isinstance(target.vrs, Register)
    assert target.l4.geometry["size_bytes"] == 14 * 2**30
    assert target.l1.geometry["size_bytes"] == 32768
    assert target.l1.geometry["slots"] == 32
    assert target.vrs.lanes == 16
    assert target.vrs.width == 32768

    # Four moves declared on the apuc unit per spec §B.
    for mv in ("DMA_L4_L1", "DMA_L1_L4", "LD_VR", "ST_VR"):
        assert target.move(mv) is not None
    # Three compute ops declared.
    for op_name in ("ADD", "MUL", "MAC"):
        assert target.op(op_name) is not None


# --------------------------------------------------------------------- #
# Codegen ctx: MAC -> sv-lookup expansion
# --------------------------------------------------------------------- #


def test_apu_v1_ctx_emit_mac_expands_to_lookup_plus_add():
    """The MAC op's emit must produce a `gvml_lookup_16(...)` line
    followed by a `gvml_add_s16(...)` line via APUv1Ctx -- this is the
    GSI sv-lookup pattern from the MICRO '25 hardware validation."""
    target = build_apu_v1_target()
    ctx = APUv1Ctx(target)
    mac = target.op("MAC")
    # All three roles share the VR file handle per the canonical
    # placement; the ctx differentiates by role when needed.
    vrs = target.vrs
    mac.emit(vrs, vrs, vrs, ctx)

    assert len(ctx.cmds) == 2, ctx.cmds
    assert ctx.cmds[0].startswith("gvml_lookup_16("), ctx.cmds[0]
    assert ctx.cmds[1].startswith("gvml_add_s16("), ctx.cmds[1]
    # The expansion must reference a scratch register name for the
    # lookup temp; the default is `mac_tmp_vr` until the regalloc spec
    # rebinds it.
    assert "mac_tmp_vr" in ctx.cmds[0]
    assert "mac_tmp_vr" in ctx.cmds[1]


def test_apu_v1_ctx_emit_mac_honors_bind_handle():
    """`bind_handle("mac_tmp", "<alias>")` must thread through into the
    sv-lookup expansion so the autoscheduler can name the scratch VR
    once and have all MAC emits pick it up."""
    target = build_apu_v1_target()
    ctx = APUv1Ctx(target)
    ctx.bind_handle("mac_tmp", "inp_vr_tmp")
    target.op("MAC").emit(target.vrs, target.vrs, target.vrs, ctx)
    assert "inp_vr_tmp" in ctx.cmds[0]
    assert "inp_vr_tmp" in ctx.cmds[1]


# --------------------------------------------------------------------- #
# Cost factory
# --------------------------------------------------------------------- #


def test_apu_v1_cost_factory_returns_callable():
    target = build_apu_v1_target()
    cost_fn = allo.get_cost("kernel_cycles", target)
    assert callable(cost_fn)
    empty_trace = MatchTrace(
        target_name="apu_v1", module_name="empty", matches=[]
    )
    assert cost_fn(empty_trace, allo.Placement(placements={})) == 0


def test_apu_v1_sv_lookup_beats_sv_mode():
    """For a matmul-shaped trace, the sv-lookup placement (MAC opcode,
    priced as `gvml_lookup_16 + gvml_add_s16` = 8 cycles per outer
    iter) must cost less than an SV-mode equivalent (raw `gvml_mul_u16`
    = 16 cycles per outer iter). This is the cost-model expression of
    the 19.7x speedup reported in the MICRO '25 hardware run."""
    target = build_apu_v1_target()
    cost_fn = allo.get_cost("kernel_cycles", target)
    layout = allo.Placement(placements={
        "local_W": target.vrs,
        "local_x": target.vrs,
        "acc": target.vrs,
    })

    sv_lookup_cost = cost_fn(_synthetic_mac_trace("MAC"), layout)
    sv_cost = cost_fn(_synthetic_mac_trace("MUL"), layout)
    assert sv_lookup_cost < sv_cost, (sv_lookup_cost, sv_cost)
    # Outer loop = 16, inner K loop folded into SIMD lanes.
    # MAC: (lookup 6 + add 2) * 16 = 128. MUL: 16 * 16 = 256.
    assert sv_lookup_cost == 16 * 8
    assert sv_cost == 16 * 16


def test_apu_v1_autoschedule_picks_canonical_vr_placement():
    """The enumerator returns a single canonical VR placement; verify
    `autoschedule` returns it intact and all three roles bind to the
    VR file handle."""
    target = build_apu_v1_target()
    trace = _synthetic_mac_trace()
    layouts = autoschedule(target, trace)
    assert len(layouts) == 1
    layout = layouts[0]
    assert layout.placements["local_W"] is target.vrs
    assert layout.placements["local_x"] is target.vrs
    assert layout.placements["acc"] is target.vrs


if __name__ == "__main__":
    test_apu_v1_target_builds()
    test_apu_v1_ctx_emit_mac_expands_to_lookup_plus_add()
    test_apu_v1_ctx_emit_mac_honors_bind_handle()
    test_apu_v1_cost_factory_returns_callable()
    test_apu_v1_sv_lookup_beats_sv_mode()
    test_apu_v1_autoschedule_picks_canonical_vr_placement()
    print("ALL PASSED")
