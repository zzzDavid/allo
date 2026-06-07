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
from allo.spmw_match import MatchedOp, MatchTrace
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


def _make_mac_match(k_ub: str) -> MatchedOp:
    """Synthetic MAC MatchedOp with a single affine.for k=0..k_ub loop."""
    return MatchedOp(
        target_op_name="MAC",
        func_name="gemv_0",
        work_id=(0,),
        enclosing_loops=[("k", "0", k_ub, 1)],
        operands=[],
        result_memref_name=None,
        op_range=("op_begin", "op_end"),
    )


def test_aim_after_match_folds_k_into_opsize():
    """SPEC-019 §3.5: after_match must rewrite the MAC_SBK opsize token
    from 1 -> K when the inner reduction loop has a constant bound > 1.
    """
    target = build_aim_target()
    ctx = AimCtx(target)
    ctx.cmd("MAC_SBK", dst=target.gpr, src0=target.banks[0], src1=target.banks[0])
    # Before after_match: opsize defaults to 1.
    assert ctx.cmds[-1].split() == ["AiM", "MAC_SBK", "1", "1", "0", "0"]

    ctx.after_match(_make_mac_match("128"), n_emitted=1)
    # Positional trace: opsize token (index 2) rewritten to 128.
    assert ctx.cmds[-1].split() == ["AiM", "MAC_SBK", "128", "1", "0", "0"]
    # Human-readable mirror picks up the appended opsize annotation.
    assert "opsize=128" in ctx._human_lines[-1]


def test_aim_after_match_parses_affine_map_bound():
    """`_parse_loop_bound` accepts `affine_map<() -> (N)>`-shaped strings;
    after_match must thread them through to the opsize token.
    """
    target = build_aim_target()
    ctx = AimCtx(target)
    ctx.cmd("MAC_SBK", dst=target.gpr, src0=target.banks[0], src1=target.banks[0])
    ctx.after_match(_make_mac_match("affine_map<() -> (256)>"), n_emitted=1)
    assert ctx.cmds[-1].split()[2] == "256"


def test_aim_after_match_is_noop_for_non_mac():
    """A non-MAC match must not touch the last-emitted line."""
    target = build_aim_target()
    ctx = AimCtx(target)
    ctx.cmd("MAC_SBK", dst=target.gpr, src0=target.banks[0], src1=target.banks[0])
    before = ctx.cmds[-1]
    fake = _make_mac_match("128")
    fake.target_op_name = "ADD"
    ctx.after_match(fake, n_emitted=1)
    assert ctx.cmds[-1] == before


def test_aim_after_match_is_noop_for_empty_loops():
    """Synthetic traces (no enclosing loops) preserve pre-SPEC-019 behaviour."""
    target = build_aim_target()
    ctx = AimCtx(target)
    ctx.cmd("MAC_SBK", dst=target.gpr, src0=target.banks[0], src1=target.banks[0])
    before = ctx.cmds[-1]
    empty = _make_mac_match("128")
    empty.enclosing_loops = []
    ctx.after_match(empty, n_emitted=1)
    assert ctx.cmds[-1] == before


def test_aim_after_match_is_noop_for_unparseable_bound():
    """Symbolic / multi-int bounds fall back to opsize=1; SPEC-019 §2."""
    target = build_aim_target()
    ctx = AimCtx(target)
    ctx.cmd("MAC_SBK", dst=target.gpr, src0=target.banks[0], src1=target.banks[0])
    before = ctx.cmds[-1]
    ctx.after_match(_make_mac_match("symbolic_K"), n_emitted=1)
    assert ctx.cmds[-1] == before


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
    test_aim_after_match_folds_k_into_opsize()
    test_aim_after_match_parses_affine_map_bound()
    test_aim_after_match_is_noop_for_non_mac()
    test_aim_after_match_is_noop_for_empty_loops()
    test_aim_after_match_is_noop_for_unparseable_bound()
    test_aim_cost_factory_returns_callable()
    print("ALL PASSED")
