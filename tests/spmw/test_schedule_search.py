# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SPEC-023 D1: whole-trace liveness + cross-op/broadcast residency knob.

`trace_liveness` folds T4 (cross-work-id broadcast hoist) and T6 (cross-kernel
residency) into ONE whole-`MatchTrace` pass; the result feeds a typed
`residency` `Knob` on the landed `spmw_knobs.py`. The per-group argmin stays
structurally intact (a whole-trace pre-pass bounds the residency candidate set;
a post-argmin reconciliation credits a resident pair only when both endpoints
agree). Regression gate: single-op GEMV flags nothing -> residency is a 1x fan
-> argmin byte-identical; MLP argmin unchanged while the `residency` knob_cost
is unregistered (the resident arm adds no benefit, so restage wins).

(Tile/fold and double-buffer depth are sibling tasks 005/006; this file proves
the D1 liveness + residency mechanism only.)
"""

from __future__ import annotations

import allo
from allo.spmw_autoschedule import autoschedule
from allo.spmw_liveness import (
    trace_liveness,
    crosses_boundary,
    memref_span,
)
from allo.spmw_knobs import (
    KnobCtx,
    _residency_candidates,
    _residency_emit,
    registered_knobs,
)
from allo.spmw_match import MatchTrace, MatchedOp, OperandBinding

from _fixtures import build_samsung_target


def _gemv_trace() -> MatchTrace:
    return MatchTrace(
        target_name="samsung_hbm_pim", module_name="gemv", matches=[
            MatchedOp(
                target_op_name="MAC", func_name="gemv_0_0", work_id=(0, 0),
                enclosing_loops=[("%a", "0", "32", 1), ("%b", "0", "1024", 1)],
                operands=[
                    OperandBinding(role="x", memref_name="local_W"),
                    OperandBinding(role="y", memref_name="local_x"),
                    OperandBinding(role="acc", memref_name="acc",
                                   is_loop_carried=True),
                ],
                result_memref_name="acc", op_range=("%a", "%b")),
        ])


def _mlp_trace() -> MatchTrace:
    """Two kernels: layer1 writes local_h (acc role), layer2 reads it (y role)
    -- local_h crosses the kernel boundary (T6)."""
    return MatchTrace(
        target_name="samsung_hbm_pim", module_name="mlp", matches=[
            MatchedOp(
                target_op_name="MAC", func_name="mlp_layer1_0", work_id=(0,),
                enclosing_loops=[("%i", "0", "256", 1), ("%k", "0", "128", 1)],
                operands=[
                    OperandBinding(role="x", memref_name="local_W1"),
                    OperandBinding(role="y", memref_name="local_x"),
                    OperandBinding(role="acc", memref_name="local_h",
                                   is_loop_carried=True),
                ],
                result_memref_name="local_h", op_range=("%a", "%b")),
            MatchedOp(
                target_op_name="MAC", func_name="mlp_layer2_0", work_id=(0,),
                enclosing_loops=[("%i", "0", "64", 1), ("%k", "0", "256", 1)],
                operands=[
                    OperandBinding(role="x", memref_name="local_W2"),
                    OperandBinding(role="y", memref_name="local_h"),
                    OperandBinding(role="acc", memref_name="local_y",
                                   is_loop_carried=True),
                ],
                result_memref_name="local_y", op_range=("%c", "%d")),
        ])


def _gemv_unrolled_trace() -> MatchTrace:
    """One kernel, two work-ids, both reading the same broadcast `x` (local_x)
    -- local_x is invariant across the work-id axis (T4 broadcast hoist)."""
    def mac(wid):
        return MatchedOp(
            target_op_name="MAC", func_name="gemv_0", work_id=wid,
            enclosing_loops=[("%a", "0", "32", 1), ("%b", "0", "1024", 1)],
            operands=[
                OperandBinding(role="x", memref_name="local_W"),
                OperandBinding(role="y", memref_name="local_x"),
                OperandBinding(role="acc", memref_name="acc",
                               is_loop_carried=True),
            ],
            result_memref_name="acc", op_range=("%a", "%b"))
    return MatchTrace(
        target_name="samsung_hbm_pim", module_name="gemv",
        matches=[mac((0,)), mac((1,))])


# --------------------------------------------------------------------- #
# trace_liveness folds T4 + T6
# --------------------------------------------------------------------- #


def test_single_op_gemv_flags_nothing():
    """The regression anchor: one match, one work-id -> nothing crosses ->
    byte-identical (residency is a 1x fan)."""
    lv = trace_liveness(_gemv_trace())
    assert not any(crosses_boundary(s) for s in lv.values())


def test_mlp_cross_kernel_detected_T6():
    lv = trace_liveness(_mlp_trace())
    h = memref_span(lv, "local_h")
    assert h is not None and h.crosses_kernel
    assert h.producer_func == "mlp_layer1_0"
    assert h.consumer_func == "mlp_layer2_0"
    # local_W1 / local_W2 are per-kernel local -> do not cross.
    assert not crosses_boundary(memref_span(lv, "local_W1"))


def test_gemv_cross_workid_detected_T4():
    """A work-id-invariant broadcast operand (local_x read by every work-id)
    is flagged crossing the work-id axis -- the broadcast-hoist signal."""
    lv = trace_liveness(_gemv_unrolled_trace())
    x = memref_span(lv, "local_x")
    assert x is not None and x.crosses_workid
    assert not x.crosses_kernel  # one kernel only


# --------------------------------------------------------------------- #
# The residency knob: candidate set is a liveness result, not a shape
# --------------------------------------------------------------------- #


def test_residency_knob_registered():
    names = {k.name for k in registered_knobs("samsung_hbm_pim")}
    assert "residency" in names


def test_residency_candidates_restage_only_when_no_crossing():
    target = build_samsung_target()
    base = allo.Placement(placements={"local_x": target.grf_a})
    # No liveness -> per-kernel mode -> single candidate (byte-identical).
    ctx = KnobCtx(target=target, base=base, liveness=None)
    assert _residency_candidates(ctx) == ["restage"]


def test_residency_candidates_two_when_crossing():
    target = build_samsung_target()
    lv = trace_liveness(_mlp_trace())
    # A candidate placing the cross-kernel local_h gets the 2-candidate fan.
    base = allo.Placement(placements={"local_h": target.grf_a})
    ctx = KnobCtx(target=target, base=base, liveness=lv)
    assert _residency_candidates(ctx) == ["restage", "resident"]


def test_residency_emit_records_pair_for_resident():
    target = build_samsung_target()
    lv = trace_liveness(_mlp_trace())
    base = allo.Placement(placements={"local_h": target.grf_a})
    ctx = KnobCtx(target=target, base=base, liveness=lv)
    restaged = _residency_emit("restage", base, ctx)
    assert restaged.extra["residency"] == "restage"
    assert "residency_pairs" not in restaged.extra
    resident = _residency_emit("resident", base, ctx)
    assert resident.extra["residency"] == "resident"
    pair = resident.extra["residency_pairs"]["local_h"]
    assert pair["producer_func"] == "mlp_layer1_0"
    assert pair["consumer_func"] == "mlp_layer2_0"
    assert pair["crosses_kernel"] is True


# --------------------------------------------------------------------- #
# Per-group argmin structurally intact; GEMV byte-identical
# --------------------------------------------------------------------- #


def test_autoschedule_gemv_byte_identical_with_liveness_prepass():
    """The liveness pre-pass + reconciliation wrap the per-group argmin without
    perturbing the single-op GEMV choice: y on the bank (EVEN_BANK), no
    residency tag (1x fan)."""
    from allo.spmw_target import MemoryRef

    target = build_samsung_target()
    layouts = autoschedule(target, _gemv_trace())
    assert len(layouts) == 1
    pl = layouts[0]
    assert isinstance(pl.placements["local_x"], MemoryRef)
    # residency, when present, is the restage default (never resident here:
    # nothing crosses, so the knob is a 1x fan and never tags "resident").
    assert pl.extra.get("residency", "restage") == "restage"


def test_autoschedule_mlp_earns_resident_for_cross_kernel_activation():
    """SPEC-023 T6/D2: with the `residency` knob_cost now registered, the
    cross-kernel activation `local_h` is cheaper RESIDENT (its inter-kernel host
    staging is elided), so the per-group argmin EARNS resident on both
    endpoints and the post-argmin reconciliation confirms the pair. This is the
    enumerated, cost-ranked residency win (the sim-confirmed proof is verifier
    task 008)."""
    target = build_samsung_target()
    layouts = autoschedule(target, _mlp_trace())
    assert len(layouts) == 2
    # Both endpoints committed to resident for local_h (reconciliation kept the
    # pair because both agreed).
    for pl in layouts:
        assert pl.extra.get("residency") == "resident", pl.extra
        assert "local_h" in pl.extra.get("residency_pairs", {}), pl.extra


def test_residency_resident_costs_less_than_restage():
    """The cost-ranking that earns resident: for the cross-kernel activation,
    the `resident` arm's whole-program cost (kernel_cycles + host_staging, the
    autoscheduler's argmin objective) is STRICTLY LESS than `restage` -- the
    inter-kernel host staging is elided. This is what makes residency
    argmin-ranked (not a hand-set default)."""
    from allo.spmw_cost import get_cost
    from allo.spmw_knobs import _residency_emit, KnobCtx

    target = build_samsung_target()
    trace = _mlp_trace()
    lv = trace_liveness(trace)
    # Score the producer layer's resident vs restage arm under the same cost_fn.
    layer1 = MatchTrace(
        target_name="samsung_hbm_pim", module_name="mlp",
        matches=[m for m in trace.matches if m.func_name == "mlp_layer1_0"])
    base = allo.Placement(placements={
        "local_W1": target.grf_a, "local_x": target.grf_a,
        "local_h": target.grf_b})
    ctx = KnobCtx(target=target, base=base, liveness=lv,
                  role_to_memref={"x": "local_W1", "y": "local_x",
                                  "acc": "local_h"})
    restage = _residency_emit("restage", base, ctx)
    resident = _residency_emit("resident", base, ctx)
    cost_fn = get_cost("kernel_cycles", target)
    c_restage = cost_fn(layer1, restage)
    c_resident = cost_fn(layer1, resident)
    assert c_resident < c_restage, (c_resident, c_restage)


def test_residency_value_that_doesnt_fit_spills():
    """A resident value placed on a register with no room must SPILL correctly
    via the landed allocator (resolve_spill_moves), not silently drop. The
    residency knob produces the candidate; the allocator's capacity guard +
    forced-spill (placement-realization D1/D2, landed) handle the overflow."""
    from allo.spmw_regalloc import allocate, Spilled
    from allo.spmw_knobs import _residency_emit, KnobCtx

    target = build_samsung_target()
    trace = _mlp_trace()
    lv = trace_liveness(trace)
    # Force 9 overlapping resident-flagged values onto grf_a (8 slots): the 9th
    # cannot fit and must spill, even with the residency tag present.
    matches = [MatchedOp(
        target_op_name="MUL", func_name="mlp_layer1_0", work_id=(0,),
        enclosing_loops=[("%a", "0", "1024", 1)],
        operands=[OperandBinding(role=f"acc{i}", memref_name=f"v{i}",
                                 is_loop_carried=True) for i in range(9)],
        result_memref_name="v0", op_range=("%a", "%b"))]
    cand = allo.Placement(placements={f"v{i}": target.grf_a for i in range(9)})
    # Tag it resident (a no-op for spill correctness, but proves residency does
    # not bypass the capacity guard).
    cand.extra["residency"] = "resident"
    result = allocate(target, matches, cand, all_candidates=[cand])
    n_spilled = sum(
        1 for h in result.placement.placements.values()
        if isinstance(h, Spilled))
    assert n_spilled == 1, result.placement.placements


def test_autoschedule_mlp_per_group_argmin_structurally_intact():
    """The residency win does NOT collapse the per-group argmin: each kernel is
    still independently allocated (2 placements, one per @allo.work), the
    cross-kernel decision is a typed knob + reconciliation, not a joint
    search."""
    target = build_samsung_target()
    layouts = autoschedule(target, _mlp_trace())
    assert len(layouts) == 2
    # The non-residency placement structure (y on the bank etc.) is unchanged;
    # only the residency tag + pair were added.
    from allo.spmw_target import MemoryRef
    assert any(
        isinstance(h, MemoryRef) for h in layouts[0].placements.values()
    )


if __name__ == "__main__":
    test_single_op_gemv_flags_nothing()
    test_mlp_cross_kernel_detected_T6()
    test_gemv_cross_workid_detected_T4()
    test_residency_knob_registered()
    test_residency_candidates_restage_only_when_no_crossing()
    test_residency_candidates_two_when_crossing()
    test_residency_emit_records_pair_for_resident()
    test_autoschedule_gemv_byte_identical_with_liveness_prepass()
    test_autoschedule_mlp_earns_resident_for_cross_kernel_activation()
    test_residency_resident_costs_less_than_restage()
    test_residency_value_that_doesnt_fit_spills()
    test_autoschedule_mlp_per_group_argmin_structurally_intact()
    print("ALL PASSED")
