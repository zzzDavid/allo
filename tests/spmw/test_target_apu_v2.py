# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for the GSI APU v2 (Gemini 2, G2, GTML) target spec.

Per spec 003 §F.5: target build, ctx MAC -> matmul+add expansion, the
GTML program-source wrapper, and the placeholder cost factory's
one-shot warning.
"""

from __future__ import annotations

import warnings

import pytest

import allo
from allo.spmw_autoschedule import autoschedule
from allo.spmw_codegen import APUv2Ctx, compile_for_target
from allo.spmw_match import MatchTrace, MatchedOp, OperandBinding
from allo.spmw_target import Memory

from _fixtures import build_apu_v2_target


def _synthetic_mac_trace() -> MatchTrace:
    """Single-match GEMV-shaped trace for APU v2."""
    return MatchTrace(
        target_name="apu_v2",
        module_name="synthetic",
        matches=[
            MatchedOp(
                target_op_name="MAC",
                func_name="gemv_0",
                work_id=(0,),
                enclosing_loops=[
                    ("%arg0", "0", "16", 1),
                    ("%arg1", "0", "4096", 1),
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


def test_apu_v2_target_builds():
    target = build_apu_v2_target()
    assert target.name == "apu_v2"
    # device -> chip child.
    assert len(target.root.children) == 1
    # L5 host DRAM and L1 bitline grid resolve via the flat handle map.
    assert isinstance(target.l5, Memory)
    assert isinstance(target.l1, Memory)
    assert target.l5.geometry["size_bytes"] == 2 ** 40
    assert target.l1.geometry["rows"] == 3072
    assert target.l1.geometry["cols"] == 65536
    assert target.l1.geometry["width"] == 1

    # Two moves declared on row_group per spec §B.
    for mv in ("COPY_TO_L1", "COPY_FROM_L1"):
        assert target.move(mv) is not None
    # Four+ compute ops declared (ADD, MUL, MAC, MATMUL, RMS_NORM,
    # SOFTMAX, AF).
    for op_name in ("ADD", "MUL", "MAC", "MATMUL", "RMS_NORM",
                    "SOFTMAX", "AF"):
        assert target.op(op_name) is not None


# --------------------------------------------------------------------- #
# Codegen ctx: MAC -> matmul + add expansion
# --------------------------------------------------------------------- #


def test_apu_v2_ctx_emit_mac_expands_to_matmul_plus_add():
    """The MAC op's emit must produce a `g.matmul(...)` line followed
    by a `g.add(...)` line via APUv2Ctx -- the GTML expansion since
    GTML has no fused MAC primitive."""
    target = build_apu_v2_target()
    ctx = APUv2Ctx(target)
    mac = target.op("MAC")
    l1 = target.l1
    # Stub handles: distinct L1 row offsets for x / y / acc.
    mac.emit(l1[0], l1[1], l1[2], ctx)

    # alloc_vector + matmul + add = 3 lines.
    assert len(ctx.cmds) == 3, ctx.cmds
    assert "g.alloc_vector" in ctx.cmds[0], ctx.cmds[0]
    assert ctx.cmds[1].startswith("g.matmul("), ctx.cmds[1]
    assert ctx.cmds[2].startswith("g.add("), ctx.cmds[2]


def test_apu_v2_ctx_cmd_emits_cpp_lines():
    """ADD / MUL emit one `g.add(...)` / `g.mul(...)` line each via
    `ctx.cmd`. Confirms the C++-text accumulation contract."""
    target = build_apu_v2_target()
    ctx = APUv2Ctx(target)
    target.op("ADD").emit(target.l1[0], target.l1[1], target.l1[2], ctx)
    assert len(ctx.cmds) == 1
    assert ctx.cmds[0].startswith("g.add("), ctx.cmds[0]
    assert ctx.cmds[0].endswith(");"), ctx.cmds[0]


def test_apu_v2_get_program_src_wraps():
    """`get_program_src()` wraps `self.cmds` with the standard GTML
    singleton boilerplate (`#include "gtml.h"` + `G2Gtml::instance()`)."""
    target = build_apu_v2_target()
    ctx = APUv2Ctx(target)
    target.op("ADD").emit(target.l1[0], target.l1[1], target.l1[2], ctx)
    src = ctx.get_program_src()
    assert "#include \"gtml.h\"" in src
    assert "G2Gtml::instance()" in src
    # Body lines flow into main().
    assert "g.add(" in src


# --------------------------------------------------------------------- #
# Cost factory
# --------------------------------------------------------------------- #


def test_apu_v2_cost_factory_warns_once():
    """The cost factory must emit a `RuntimeWarning` mentioning
    "placeholder" on first build and stay silent on the second build
    (idempotent guard). Both calls must return a working callable."""
    # Reset the module-level guard so we can assert the warning fires.
    import allo.spmw_cost_models as cm
    cm._APU_V2_WARNED = False

    target = build_apu_v2_target()

    with pytest.warns(RuntimeWarning, match="placeholder"):
        cost_fn = allo.get_cost("kernel_cycles", target)
    assert callable(cost_fn)

    # Second call: no warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cost_fn2 = allo.get_cost("kernel_cycles", target)
    assert callable(cost_fn2)

    # Placeholder cost: 1 cycle per match.
    trace = _synthetic_mac_trace()
    assert cost_fn(trace, allo.Placement(placements={})) == 1


# --------------------------------------------------------------------- #
# Autoscheduler + end-to-end codegen
# --------------------------------------------------------------------- #


def test_apu_v2_autoschedule_returns_l1_placement():
    """The enumerator returns a single canonical L1 placement; verify
    `autoschedule` picks it intact and all three roles bind to L1
    refs."""
    target = build_apu_v2_target()
    trace = _synthetic_mac_trace()
    # Suppress the placeholder warning so the test output stays clean.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        layouts = autoschedule(target, trace)
    assert len(layouts) == 1
    layout = layouts[0]
    for memref_name in ("local_W", "local_x", "acc"):
        placed = layout.placements[memref_name]
        # Each operand placed on the L1 memory.
        assert placed.memory is target.l1


def test_apu_v2_compile_emits_cpp_strings():
    """End-to-end: compile a synthetic MAC trace and assert `ctx.cmds`
    is a non-empty list with C++ GTML snippet strings.

    After spec 009 the JUMP-folding code only runs on Samsung (via the
    `after_match` hook), so APU v2's cmds list is all-string."""
    target = build_apu_v2_target()
    trace = _synthetic_mac_trace()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        compiled = compile_for_target(target, trace)
    assert isinstance(compiled.cmds, list)
    assert len(compiled.cmds) > 0
    # All entries are C++ snippet strings -- no PIMCmd JUMP leakage.
    assert all(isinstance(c, str) for c in compiled.cmds), compiled.cmds
    joined = "\n".join(compiled.cmds)
    # MAC expansion: alloc_vector + matmul + add.
    assert "g.matmul(" in joined
    assert "g.add(" in joined


if __name__ == "__main__":
    test_apu_v2_target_builds()
    test_apu_v2_ctx_emit_mac_expands_to_matmul_plus_add()
    test_apu_v2_ctx_cmd_emits_cpp_lines()
    test_apu_v2_get_program_src_wraps()
    test_apu_v2_cost_factory_warns_once()
    test_apu_v2_autoschedule_returns_l1_placement()
    test_apu_v2_compile_emits_cpp_strings()
    print("ALL PASSED")
