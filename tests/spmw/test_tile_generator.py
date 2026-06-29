# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SPEC-023 D2: capacity-bounded tile/fold generator.

`spmw_tiling.tile_candidates` emits a small set of LEGAL re-tilings of the
matched nest, derived from `MatchedOp.enclosing_loops` bounds + operand indices
+ target-tree capacity (`_build_capacity` / `_estimate_bytes`) -- NO `allo/ir/`
edit and NO tile literal. The identity tiling is always first (byte-identical);
a non-identity retile appears ONLY when a legal capacity-fitting one exists.
The choice rides a typed `tile` `Knob` on the landed `spmw_knobs.py`.

Regression anchor: today's corpus carries no dtype, so no capacity bound is
derivable -> identity-only -> the tile knob is a 1x fan -> byte-identical. The
sim-confirmed "retile beats the user's nest" win is verifier task 008.
"""

from __future__ import annotations

import allo
from allo.spmw_tiling import tile_candidates, TilePlan
from allo.spmw_autoschedule import _samsung_enumerate
from allo.spmw_knobs import registered_knobs, _tile_candidates, KnobCtx
from allo.spmw_match import MatchedOp, OperandBinding

from _fixtures import build_samsung_target


def _gemv_mac(dtype_bits=None, k=1024) -> MatchedOp:
    extra = {"dtype_bits": dtype_bits} if dtype_bits else {}
    return MatchedOp(
        target_op_name="MAC", func_name="gemv_0_0", work_id=(0, 0),
        enclosing_loops=[("%m", "0", "32", 1), ("%k", "0", str(k), 1)],
        operands=[
            OperandBinding(role="x", memref_name="local_W"),
            OperandBinding(role="y", memref_name="local_x"),
            OperandBinding(role="acc", memref_name="acc", is_loop_carried=True),
        ],
        result_memref_name="acc", op_range=("%a", "%b"), extra=extra)


def _tile_demo_target():
    """A target with `grf_a` lanes + a small bounded scratchpad, so a large
    reduction's fp16 working set overflows it and the generator must retile."""
    @allo.target("tiledemo")
    def device():
        @allo.unit(mapping=[1])
        def pe():
            allo.reg(8, 256, name="grf_a")
            allo.mem(size_bytes=2048, name="gb")  # 2048-byte bounded scratchpad
    return device


def _reduction_mac(k, dtype_bits):
    return MatchedOp(
        target_op_name="MAC", func_name="g", work_id=(0,),
        enclosing_loops=[("%k", "0", str(k), 1)],
        operands=[
            OperandBinding(role="x", memref_name="W"),
            OperandBinding(role="y", memref_name="x"),
            OperandBinding(role="acc", memref_name="acc", is_loop_carried=True),
        ],
        result_memref_name="acc", op_range=("%a", "%b"),
        extra={"dtype_bits": dtype_bits})


# --------------------------------------------------------------------- #
# Identity-only on the no-capacity-bound corpus (byte-identical)
# --------------------------------------------------------------------- #


def test_identity_only_when_no_capacity_bound():
    """No dtype -> no derivable footprint -> no capacity bound -> the generator
    emits the identity tiling ONLY (byte-identical). Identity is first."""
    target = build_samsung_target()
    plans = tile_candidates(target, [_gemv_mac(dtype_bits=None, k=1024)])
    assert len(plans) == 1
    assert plans[0].is_identity and plans[0].tile_size == 1024


def test_identity_always_first():
    target = build_samsung_target()
    plans = tile_candidates(target, [_gemv_mac(dtype_bits=None)])
    assert plans and plans[0].is_identity


def test_no_tileable_nest_returns_empty():
    """A trace with no reducing (MAC-with-inner-loop) match -> no tile axis ->
    empty (the caller keeps the un-tiled candidate)."""
    add = MatchedOp(
        target_op_name="ADD", func_name="f", work_id=(0,),
        enclosing_loops=[("%i", "0", "16", 1)],
        operands=[OperandBinding(role="x", memref_name="a")],
        result_memref_name="a", op_range=("%a", "%b"))
    assert tile_candidates(build_samsung_target(), [add]) == []


# --------------------------------------------------------------------- #
# >=2 candidates when capacity binds; derived, no literal
# --------------------------------------------------------------------- #


def test_capacity_bound_retile_is_legal_and_derived():
    """When the fp16 working set overflows the 2048-byte scratchpad, the
    generator emits identity + a capacity-fitting retile. The tile size is a
    DERIVED divisor of the bound, lane-aligned, capacity-fitting -- never a
    literal."""
    target = _tile_demo_target()
    bound = 4096
    plans = tile_candidates(target, [_reduction_mac(bound, dtype_bits=16)])
    assert len(plans) == 2, [(p.tile_size, p.is_identity) for p in plans]
    identity, retile = plans
    assert identity.is_identity and identity.tile_size == bound
    assert not retile.is_identity
    # Derived properties (not a literal): strictly smaller, a divisor of the
    # bound, lane-aligned (lanes=8), and the fp16 working set fits 2048 bytes.
    assert retile.tile_size < bound
    assert bound % retile.tile_size == 0
    assert retile.tile_size % 8 == 0          # grf_a.lanes alignment
    assert retile.tile_size * 2 <= 2048       # fp16 (2 B) fits the scratchpad


def test_retile_tracks_geometry_not_literal():
    """Halving the scratchpad halves the legal tile -- proving the size is read
    off target geometry, not a pasted constant."""
    @allo.target("tiledemo_small")
    def device():
        @allo.unit(mapping=[1])
        def pe():
            allo.reg(8, 256, name="grf_a")
            allo.mem(size_bytes=1024, name="gb")  # half the cap
    plans = tile_candidates(device, [_reduction_mac(4096, dtype_bits=16)])
    assert len(plans) == 2
    # 1024 bytes / 2 B/elem = 512 elems, lane-aligned divisor of 4096.
    assert plans[1].tile_size == 512


# --------------------------------------------------------------------- #
# The tile knob: registered, 1x fan byte-identical on the corpus
# --------------------------------------------------------------------- #


def test_tile_knob_registered():
    for tname in ("samsung_hbm_pim", "upmem", "apu_v1"):
        assert "tile" in {k.name for k in registered_knobs(tname)}


def test_tile_knob_one_x_fan_on_corpus():
    """On the no-dtype corpus the tile knob returns the identity singleton (1x
    fan) and writes no `extra` -> the candidate set is byte-identical."""
    target = build_samsung_target()
    cands = _samsung_enumerate(target, [_gemv_mac(dtype_bits=None)])
    # No candidate carries a `tile` extra (identity writes nothing).
    assert all("tile" not in c.extra for c in cands), [c.extra for c in cands]


def test_tile_knob_candidates_via_ctx():
    target = _tile_demo_target()
    ctx = KnobCtx(target=target, matches=[_reduction_mac(4096, 16)])
    plans = _tile_candidates(ctx)
    assert plans is not None and len(plans) == 2
    assert plans[0].is_identity and not plans[1].is_identity


if __name__ == "__main__":
    test_identity_only_when_no_capacity_bound()
    test_identity_always_first()
    test_no_tileable_nest_returns_empty()
    test_capacity_bound_retile_is_legal_and_derived()
    test_retile_tracks_geometry_not_literal()
    test_tile_knob_registered()
    test_tile_knob_one_x_fan_on_corpus()
    test_tile_knob_candidates_via_ctx()
    print("ALL PASSED")
