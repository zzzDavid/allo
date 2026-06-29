# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SPEC-023 D3: double-buffer depth as an overlap-ranked Knob.

Depth 1 (serialized DMA then compute) vs depth 2 (ping-pong). Depth-2 emits the
DMA as a LOOP-ENCODED `Phase(Resource.DMA, latency=per_move, ii=per_move,
count=n)` so under `combine(overlap=True)` / `_max_across_with_fill_drain` the
steady-state DMA hides behind compute (only the fill is serial); depth-1 emits
the COLLAPSED `Phase(DMA, latency=total, ii=0, count=1)` that cannot hide.
Decision in argmin (the overlap flavor), mechanism in codegen.

Regression: under the default `faithful` flavor depth-1 and depth-2 tie
(collapsed both) -> argmin keeps depth-1 -> byte-identical. The candidate set
is [1] on backends with no DMA-hide DOF (a 1x fan), [1, 2] only where an
overlap compose exists (APU v1). The sim-confirmed depth-2-beats-depth-1 win is
verifier task 008; this discharges the cost cycle's deferred overlap sim-flip.
"""

from __future__ import annotations

import allo
from allo.spmw_knobs import (
    _double_buffer_candidates,
    _double_buffer_emit,
    _has_dma_hide_dof,
    KnobCtx,
    registered_knobs,
)
from allo.spmw_cost_model import (
    get_cost_model,
    knob_cost,
    ComposeCtx,
    combine,
    Resource,
)
from allo.spmw_match import MatchTrace, MatchedOp, OperandBinding

from _fixtures import build_apu_v1_target, build_samsung_target, build_upmem_target


def _apu_gemv() -> MatchTrace:
    return MatchTrace(
        target_name="apu_v1", module_name="g", matches=[
            MatchedOp(
                target_op_name="MAC", func_name="gemv_0", work_id=(0,),
                enclosing_loops=[("%m", "0", "4096", 1), ("%k", "0", "1024", 1)],
                operands=[
                    OperandBinding(role="x", memref_name="W"),
                    OperandBinding(role="y", memref_name="x"),
                    OperandBinding(role="acc", memref_name="acc",
                                   is_loop_carried=True),
                ],
                result_memref_name="acc", op_range=("%a", "%b"))])


def _apu_base(target):
    return allo.Placement(
        placements={"W": target.vrs, "x": target.vrs, "acc": target.vrs},
        mode="sv_lookup", extra={"vr_dma": "inter"})


# --------------------------------------------------------------------- #
# Candidate set: [1,2] only where a DMA-hide DOF exists
# --------------------------------------------------------------------- #


def test_depth_candidates_gated_on_dma_hide_dof():
    apu = build_apu_v1_target()
    samsung = build_samsung_target()
    upmem = build_upmem_target()
    assert _has_dma_hide_dof(apu)            # APU v1 has an overlap compose
    assert not _has_dma_hide_dof(samsung)    # Samsung bank-row fold is depth-1
    assert not _has_dma_hide_dof(upmem)      # no overlap compose (today)
    assert _double_buffer_candidates(KnobCtx(target=apu)) == [1, 2]
    assert _double_buffer_candidates(KnobCtx(target=samsung)) == [1]
    assert _double_buffer_candidates(KnobCtx(target=upmem)) == [1]


def test_double_buffer_knob_registered_everywhere():
    for tname in ("samsung_hbm_pim", "upmem", "apu_v1", "apu_v2"):
        assert "double_buffer" in {k.name for k in registered_knobs(tname)}


def test_depth_emit_byte_identical_for_depth_1():
    apu = build_apu_v1_target()
    base = _apu_base(apu)
    d1 = _double_buffer_emit(1, base, KnobCtx(target=apu))
    # depth-1 writes NO double_buffer key -> byte-identical to today.
    assert "double_buffer" not in d1.extra
    d2 = _double_buffer_emit(2, base, KnobCtx(target=apu))
    assert d2.extra["double_buffer"] is True


# --------------------------------------------------------------------- #
# The overlap fold: depth-2 hides DMA behind compute; faithful ties
# --------------------------------------------------------------------- #


def _score(model, target, trace, layout, overlap):
    return combine(list(model.compose(ComposeCtx(target, trace, layout)).phases),
                   overlap=overlap)


def test_depth_2_beats_depth_1_under_overlap():
    """Under the overlap flavor, depth-2's loop-encoded DMA hides behind
    compute (steady state absorbed) -> strictly cheaper than depth-1's
    collapsed DMA. This is the §A1 overlap-fold payoff."""
    apu = build_apu_v1_target()
    trace = _apu_gemv()
    base = _apu_base(apu)
    ov = get_cost_model("apu_v1", "overlap")
    d1 = _double_buffer_emit(1, base, KnobCtx(target=apu))
    d2 = _double_buffer_emit(2, base, KnobCtx(target=apu))
    c1 = _score(ov, apu, trace, d1, overlap=True)
    c2 = _score(ov, apu, trace, d2, overlap=True)
    assert c2 < c1, (c2, c1)


def test_faithful_default_ties_depth_1_and_2():
    """Under the default faithful flavor both depths emit the collapsed DMA
    phase -> equal cost -> argmin keeps depth-1 -> byte-identical default."""
    apu = build_apu_v1_target()
    trace = _apu_gemv()
    base = _apu_base(apu)
    fa = get_cost_model("apu_v1", "faithful")
    d1 = _double_buffer_emit(1, base, KnobCtx(target=apu))
    d2 = _double_buffer_emit(2, base, KnobCtx(target=apu))
    c1 = _score(fa, apu, trace, d1, overlap=False)
    c2 = _score(fa, apu, trace, d2, overlap=False)
    assert c1 == c2, (c1, c2)


# --------------------------------------------------------------------- #
# The registered knob_cost: DMA phases on Resource.DMA, right encoding
# --------------------------------------------------------------------- #


def test_double_buffer_knob_cost_registered_and_shaped():
    """`register_knob_cost("apu_v1", "double_buffer", ...)` returns DMA phases:
    depth-2 LOOP-ENCODED (ii == latency, count == n_moves -> fill hides),
    depth-1 COLLAPSED (ii == 0, count == 1 -> fill is the whole DMA)."""
    apu = build_apu_v1_target()
    trace = _apu_gemv()
    base = _apu_base(apu)
    ctx2 = ComposeCtx(apu, trace, _double_buffer_emit(2, base, KnobCtx(target=apu)))
    ctx1 = ComposeCtx(apu, trace, _double_buffer_emit(1, base, KnobCtx(target=apu)))
    ph2 = knob_cost("apu_v1", "double_buffer", 2, ctx2)
    ph1 = knob_cost("apu_v1", "double_buffer", 1, ctx1)
    assert ph2 and ph2[0].resource is Resource.DMA
    assert ph2[0].count > 1 and ph2[0].ii == ph2[0].latency   # loop-encoded
    assert ph1 and ph1[0].resource is Resource.DMA
    assert ph1[0].count == 1 and ph1[0].ii == 0               # collapsed
    # Same total DMA work both ways (the encoding differs, not the magnitude).
    from allo.spmw_cost_model import phase_cycles
    assert phase_cycles(ph2[0]) == phase_cycles(ph1[0])


if __name__ == "__main__":
    test_depth_candidates_gated_on_dma_hide_dof()
    test_double_buffer_knob_registered_everywhere()
    test_depth_emit_byte_identical_for_depth_1()
    test_depth_2_beats_depth_1_under_overlap()
    test_faithful_default_ties_depth_1_and_2()
    test_double_buffer_knob_cost_registered_and_shaped()
    print("ALL PASSED")
