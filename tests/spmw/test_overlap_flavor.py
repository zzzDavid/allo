# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Design 07 D1 (task 005): the `overlap` flavor's fold rule.

Overlap is a PROPERTY OF THE Phase TIMELINE, not a parallel hand-coded
flavor: the ONLY per-flavor difference vs faithful is the fold rule bound in
`_COMBINER_FOR_FLAVOR` (`combine(overlap=True)` = max-across independent
resources + DMA fill; `overlap=False` = serial sum). The backend overlap
compose differs from faithful ONLY in the DMA phase encoding it picks for a
double-buffered schedule.

All STATIC (no simulator). What is proven:

  * `combine(overlap=True)` = max(COMPUTE,DMA) + DMA-fill + HOST + LOCALITY
    (D1.2); the faithful sum fold is unchanged.
  * The collapsed DMA phase does NOT hide (serialized); the loop-encoded DMA
    phase hides its steady state (double-buffered).
  * The D1.4 argmin flip: APU v1 double-buffered (b) scores BELOW serialized
    (a) under overlap, EQUAL under faithful.
  * faithful APU v1 numbers are byte-identical to before (the overlap arm is a
    separate flavor; faithful folds the same collapsed phases by sum).
"""
from __future__ import annotations

import allo
from allo.spmw_cost_model import (
    ComposeCtx,
    Phase,
    Resource,
    combine,
    evaluate,
    get_cost_model,
    phases_as_dict,
)
from allo.spmw_match import MatchTrace, MatchedOp, OperandBinding
from allo.spmw_autoschedule import Placement

from _fixtures import build_apu_v1_target, build_samsung_target


# --------------------------------------------------------------------- #
# combine(overlap=True) unit semantics (design 07 §D1.2)
# --------------------------------------------------------------------- #


def test_overlap_max_across_compute_dma():
    # COMPUTE dominates; a collapsed DMA (fill = its whole latency) does NOT
    # hide -> max(C, D) + fill = max(100,30)+30 = 130 (== serial sum here only
    # because the single-shot DMA fill is its whole cost).
    phases = [
        Phase(Resource.COMPUTE, 100, 0, 1, "exec"),
        Phase(Resource.DMA, 30, 0, 1, "vr_dma"),
    ]
    assert combine(phases, overlap=True) == 100 + 30  # collapsed: no hide
    assert combine(phases, overlap=False) == 130


def test_overlap_loop_encoded_dma_hides_steady_state():
    # Loop-encoded DMA: latency=per_move=10, ii=10, count=5 -> phase_cycles=50.
    # cyc_DMA=50 hides behind cyc_COMPUTE=100; only the fill (one per_move=10)
    # is charged: max(100,50)+10 = 110 < serial sum 150.
    phases = [
        Phase(Resource.COMPUTE, 100, 0, 1, "exec"),
        Phase(Resource.DMA, 10, 10, 5, "vr_dma"),
    ]
    assert combine(phases, overlap=True) == 100 + 10
    assert combine(phases, overlap=False) == 100 + 50


def test_overlap_dma_dominates_when_bigger():
    # If DMA exceeds compute, the max picks DMA; fill still added on top.
    phases = [
        Phase(Resource.COMPUTE, 40, 0, 1, "exec"),
        Phase(Resource.DMA, 10, 10, 20, "vr_dma"),  # cyc_DMA = 200
    ]
    # max(40, 200) + fill(10) = 210.
    assert combine(phases, overlap=True) == 200 + 10


def test_overlap_host_and_locality_stay_serial():
    # HOST staging + LOCALITY stall are NOT overlappable (D1.2): added on top.
    phases = [
        Phase(Resource.COMPUTE, 100, 0, 1, "exec"),
        Phase(Resource.DMA, 10, 10, 5, "vr_dma"),    # hides -> fill 10
        Phase(Resource.HOST, 70, 0, 1, "stage"),     # serial
        Phase(Resource.LOCALITY, 13, 0, 1, "locality"),  # serial stall
    ]
    # max(100,50) + 10 + 70 + 13 = 193.
    assert combine(phases, overlap=True) == 100 + 10 + 70 + 13


def test_overlap_no_dma_equals_compute():
    phases = [Phase(Resource.COMPUTE, 256, 0, 1, "exec")]
    assert combine(phases, overlap=True) == 256
    assert combine(phases, overlap=False) == 256


# --------------------------------------------------------------------- #
# APU v1 D1.4 argmin flip (the worked example)
# --------------------------------------------------------------------- #


def _ffn_trace() -> MatchTrace:
    """Two-stage FFN -> n_weight_tiles=1, n_boundaries=1; intra vr_dma gives
    n_moves=2 (so the loop-encoded fill < the collapsed total -> a strict
    overlap hide win)."""
    return MatchTrace(
        target_name="apu_v1",
        module_name="synthetic_ffn",
        matches=[
            MatchedOp(
                target_op_name="MAC", func_name="ffn_0_0", work_id=(0, 0),
                enclosing_loops=[("%i1", "0", "256", 1), ("%k1", "0", "64", 1)],
                operands=[
                    OperandBinding(role="x", memref_name="local_W1"),
                    OperandBinding(role="y", memref_name="local_x"),
                    OperandBinding(role="acc", memref_name="local_h",
                                   is_loop_carried=True),
                ],
                result_memref_name="local_h", op_range=("%a", "%b"),
            ),
            MatchedOp(
                target_op_name="MAC", func_name="ffn_0_0", work_id=(0, 0),
                enclosing_loops=[("%i2", "0", "64", 1), ("%k2", "0", "256", 1)],
                operands=[
                    OperandBinding(role="x", memref_name="local_W2"),
                    OperandBinding(role="y", memref_name="local_h"),
                    OperandBinding(role="acc", memref_name="local_y",
                                   is_loop_carried=True),
                ],
                result_memref_name="local_y", op_range=("%c", "%d"),
            ),
        ],
    )


def _serialized(vr_dma: str = "intra") -> Placement:
    return Placement(mode="sv_lookup", extra={"vr_dma": vr_dma})


def _double_buffered(vr_dma: str = "intra") -> Placement:
    return Placement(mode="sv_lookup", extra={"vr_dma": vr_dma, "double_buffer": True})


def test_apu_v1_overlap_flip_double_buffer_below_serialized():
    """D1.4: with the SAME vr_dma schedule (same n_moves, same move_total),
    the double-buffered candidate hides its DMA under overlap and scores
    strictly below the serialized one; faithful scores them EQUAL."""
    target = build_apu_v1_target()
    trace = _ffn_trace()

    f_serial = evaluate(target, trace, _serialized(), flavor="faithful").cycles
    f_double = evaluate(target, trace, _double_buffered(), flavor="faithful").cycles
    o_serial = evaluate(target, trace, _serialized(), flavor="overlap").cycles
    o_double = evaluate(target, trace, _double_buffered(), flavor="overlap").cycles

    # faithful cannot see the hiding: the two candidates tie.
    assert f_serial == f_double, (f_serial, f_double)
    # overlap flips: double-buffered wins (DMA steady state hidden).
    assert o_double < o_serial, (o_double, o_serial)
    # serialized overlap == faithful (collapsed DMA does not hide).
    assert o_serial == f_serial, (o_serial, f_serial)


def test_apu_v1_overlap_flip_quantities():
    """Pin the exact numbers behind the flip (so the flip is not an
    accident): exec_ops=2560, move_total(intra)=290, per_move=145."""
    target = build_apu_v1_target()
    trace = _ffn_trace()
    # exec: sv_lookup MAC=8; iters 256 + 64 = 320 -> 8*320 = 2560.
    EXEC = 8 * (256 + 64)
    PER_MOVE = 140 + 5
    N_MOVES = 2                       # intra: n_wt(1) + n_bnd(1)*n_out(1)
    MOVE_TOTAL = N_MOVES * PER_MOVE   # 290

    f = evaluate(target, trace, _serialized(), flavor="faithful").cycles
    assert f == EXEC + MOVE_TOTAL                       # 2850
    o_serial = evaluate(target, trace, _serialized(), flavor="overlap").cycles
    assert o_serial == max(EXEC, MOVE_TOTAL) + MOVE_TOTAL  # 2850 (no hide)
    o_double = evaluate(target, trace, _double_buffered(), flavor="overlap").cycles
    assert o_double == max(EXEC, MOVE_TOTAL) + PER_MOVE    # 2705 (hide steady)


def test_apu_v1_overlap_dma_phase_loop_encoded_when_double_buffered():
    """The overlap compose emits a LOOP-ENCODED DMA phase under the
    double_buffer knob (fill = per_move), and a COLLAPSED one otherwise."""
    target = build_apu_v1_target()
    trace = _ffn_trace()
    model = get_cost_model("apu_v1", "overlap")

    res_db = model.compose(ComposeCtx(target, trace, _double_buffered()))
    dma_db = [p for p in res_db.phases if p.resource is Resource.DMA][0]
    assert dma_db.count == 2 and dma_db.ii == 145 and dma_db.latency == 145

    res_s = model.compose(ComposeCtx(target, trace, _serialized()))
    dma_s = [p for p in res_s.phases if p.resource is Resource.DMA][0]
    assert dma_s.count == 1 and dma_s.ii == 0 and dma_s.latency == 290

    # Both encodings carry the SAME phase_cycles (= move_total) -> faithful
    # fold is identical regardless of encoding.
    from allo.spmw_cost_model import phase_cycles
    assert phase_cycles(dma_db) == phase_cycles(dma_s) == 290


# --------------------------------------------------------------------- #
# faithful is untouched by the new overlap flavor (the anchor)
# --------------------------------------------------------------------- #


def test_apu_v1_faithful_unchanged_by_overlap_flavor():
    """Registering the overlap flavor does not perturb the faithful APU v1
    estimate: faithful folds the collapsed phases by sum, as before."""
    target = build_apu_v1_target()
    trace = _ffn_trace()
    model = get_cost_model("apu_v1", "faithful")
    for layout in (_serialized("intra"), _serialized("inter")):
        res = model.compose(ComposeCtx(target, trace, layout))
        assert combine(list(res.phases), overlap=False) == res.cycles


def test_overlap_flavor_registered_and_bound():
    from allo.spmw_cost_models import _COMBINER_FOR_FLAVOR

    assert _COMBINER_FOR_FLAVOR["overlap"] is True
    assert _COMBINER_FOR_FLAVOR["faithful"] is False
    m = get_cost_model("apu_v1", "overlap")
    assert m.flavor == "overlap"
    assert m.confidence == "coarse"   # overlap fold is not sim-calibrated (D1.3)
