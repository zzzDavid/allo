# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for the SK-Hynix AiM (GDDR6 PIM) target spec.

Three per spec 003 §F.5: target build, ctx text-line emission, and the
kernel_cycles cost factory returning a callable that handles an empty
trace.
"""

from __future__ import annotations

import allo
from allo.spmw_codegen import AimCtx
from allo.spmw_match import MatchTrace
from allo.spmw_target import Memory, MemoryRef, Register

from _fixtures import build_aim_target


def test_aim_target_builds():
    target = build_aim_target()
    assert target.name == "aim"
    # device -> one channel-group child (the @unit named `channel`).
    assert len(target.root.children) == 1
    # Bank-level memories/registers are exposed via the target's flat
    # handle map; check that they resolve and have the right types.
    assert isinstance(target.banks, Memory)
    assert isinstance(target.gb, Memory)
    assert isinstance(target.gpr, Register)
    assert target.banks.geometry["banks"] == 32
    assert target.banks.geometry["rows"] == 16384
    assert target.gb.geometry["entries"] == 512


def test_aim_ctx_emits_text():
    target = build_aim_target()
    ctx = AimCtx(target)
    # Build operand handles directly from the target — same shapes the
    # emit lambdas would supply.
    gpr_handle = target.gpr
    bank_handle = target.banks[3]
    ctx.cmd("MAC_SBK", dst=gpr_handle, src0=bank_handle, src1=bank_handle)

    # Positional ramulator2 trace: `AiM MAC_SBK <opsize> <channel_mask>
    # <bank_index> <row_addr>`. With bank=3 supplied, bank_index must be 3.
    assert len(ctx.cmds) == 1
    line = ctx.cmds[0]
    assert line.startswith("AiM MAC_SBK"), line
    tokens = line.split()
    assert tokens == ["AiM", "MAC_SBK", "1", "1", "3", "0"], tokens

    # Human-readable mirror retains the older key=value annotation
    # so workflow-level inspection / diffing still works.
    assert len(ctx._human_lines) == 1
    human = ctx._human_lines[0]
    assert "gpr_out=0" in human, human
    assert "bank=3" in human, human


def test_aim_cost_factory_returns_callable():
    target = build_aim_target()
    cost_fn = allo.get_cost("kernel_cycles", target)
    assert callable(cost_fn)
    empty_trace = MatchTrace(target_name="aim", module_name="empty", matches=[])
    # An empty trace has no work; the cost is zero regardless of layout.
    assert cost_fn(empty_trace, allo.Placement(placements={})) == 0


if __name__ == "__main__":
    test_aim_target_builds()
    test_aim_ctx_emits_text()
    test_aim_cost_factory_returns_callable()
    print("ALL PASSED")
