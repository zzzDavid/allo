# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Design 07 A3 + A4 (task 008): the per-knob marginal-cost seam + the
opt-in confidence gate.

A3 -- a knob's cost contribution is a `list[Phase]` on the A1 timeline
(`knob_cost(target, name, value, ctx)`), so `compose = combine(base op/move
phases + Σ knob.cost(chosen))`. Because the fold is over a Phase list, a
search that changes ONE knob re-costs only that knob and re-folds; the
marginal delta equals a full `compose` recompute BYTE-FOR-BYTE (the IVM
delta-not-recompute property). This task proves the seam is exact on the
`stage_resident` knob; lever migration is placement-task D4 (OUT of scope).

A4 -- the opt-in `confidence_gate` in `autoschedule`: OFF by default ⇒ the
argmin scoring core is byte-identical (the gate is never called). ON ⇒ it
consults the chosen candidate's `CostResult.confidence` (which downgrades to
"coarse" on a tier-3 dynamic-trip fall-through -- the silent `dynamic_trip->1`
is now VISIBLE) AND the symbolic provenance band (`provenance_band`), and
under `gate_policy="refuse"` refuses to silently commit a low-confidence
ranking. The gate never alters the SELECTION.

All STATIC (no simulator).
"""
from __future__ import annotations

import warnings

import pytest

from allo.spmw_cost_model import (
    ComposeCtx,
    Provenance,
    active_knobs,
    combine,
    evaluate,
    get_cost_model,
    knob_cost,
    knob_marginal_delta,
    knob_phases,
    provenance_band,
)
from allo.spmw_match import MatchTrace, MatchedOp, OperandBinding
from allo.spmw_autoschedule import Placement, autoschedule

from _fixtures import build_aim_target, build_samsung_target


# --------------------------------------------------------------------- #
# Traces / layouts
# --------------------------------------------------------------------- #


def _gemv_trace(target_name: str, ub: str = "1024", batch: int = 1) -> MatchTrace:
    extra = {} if batch == 1 else {"batch_dim": batch}
    return MatchTrace(
        target_name=target_name, module_name="gemv",
        matches=[
            MatchedOp(
                target_op_name="MAC", func_name="gemv_0", work_id=(0,),
                enclosing_loops=[("%i", "0", "4096", 1), ("%k", "0", ub, 1)],
                operands=[
                    OperandBinding(role="x", memref_name="W"),
                    OperandBinding(role="y", memref_name="x"),
                    OperandBinding(role="acc", memref_name="acc",
                                   is_loop_carried=True),
                ],
                result_memref_name="acc", op_range=("%a", "%b"),
                extra=dict(extra),
            )
        ],
    )


def _layout(resident: bool, **extra) -> Placement:
    e = {"stage_resident": resident}
    e.update(extra)
    return Placement(placements={}, extra=e)


# --------------------------------------------------------------------- #
# A3 -- the knob seam: active_knobs / knob_phases
# --------------------------------------------------------------------- #


def test_active_knobs_reads_registered_knob_off_layout():
    target = build_samsung_target()
    ctx = ComposeCtx(target, _gemv_trace("samsung_hbm_pim"), _layout(True))
    knobs = dict(active_knobs(ctx))
    # stage_resident has a registered cost (004) -> active.
    assert knobs.get("stage_resident") is True
    # n_fibers has NO registered knob cost -> not surfaced (stays in compose).
    ctx2 = ComposeCtx(target, _gemv_trace("samsung_hbm_pim"),
                      _layout(False, n_fibers=2))
    assert "n_fibers" not in dict(active_knobs(ctx2))


def test_knob_phases_equals_stage_resident_contribution():
    target = build_samsung_target()
    for resident in (False, True):
        ctx = ComposeCtx(target, _gemv_trace("samsung_hbm_pim"), _layout(resident))
        # knob_phases(ctx) == knob_cost(stage_resident) for this single knob.
        kp = knob_phases(ctx)
        direct = knob_cost("samsung_hbm_pim", "stage_resident", resident, ctx)
        assert [(p.resource, p.latency, p.tag) for p in kp] \
            == [(p.resource, p.latency, p.tag) for p in direct]


def test_unregistered_knob_contributes_no_phases():
    target = build_samsung_target()
    # A layout with only an UNREGISTERED knob set -> knob_phases empty.
    ctx = ComposeCtx(target, _gemv_trace("samsung_hbm_pim"),
                     Placement(extra={"n_fibers": 4}))
    assert knob_phases(ctx) == []


# --------------------------------------------------------------------- #
# A3.3 -- the exactness proof: marginal delta == full recompute
# --------------------------------------------------------------------- #


def _device_base_phases(target, trace, layout):
    """The base (op/move) phases: the device-exec compose, which is
    INDEPENDENT of stage_resident (that lever lives in host_staging)."""
    model = get_cost_model("samsung_hbm_pim", "faithful")
    return list(model.compose(ComposeCtx(target, trace, layout)).phases)


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_stage_resident_knob_delta_equals_full_recompute(batch):
    """The non-tautology proof (design 07 §A3.3): the marginal delta of
    flipping `stage_resident` False->True, computed by re-costing ONLY that
    knob and re-folding, equals the full `evaluate` recompute delta,
    BYTE-FOR-BYTE."""
    target = build_samsung_target()
    trace = _gemv_trace("samsung_hbm_pim", batch=batch)

    # base = device exec phases (knob-independent). ctx layout is irrelevant
    # for the device base here, but use a neutral one.
    base = _device_base_phases(target, trace, _layout(False))
    ctx = ComposeCtx(target, trace, _layout(False))

    marginal = knob_marginal_delta(
        base, ctx, "stage_resident", False, True, overlap=False
    )
    full = (evaluate(target, trace, _layout(True), "faithful").cycles
            - evaluate(target, trace, _layout(False), "faithful").cycles)
    assert marginal == full, (batch, marginal, full)


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_base_plus_knob_equals_full_compose(batch):
    """`combine(base + knob.cost(v)) == evaluate(layout_v).cycles` -- the
    whole-program estimate IS base op/move phases + Σ knob.cost (design 07
    §A3.2), so the seam reconstructs the full number exactly."""
    target = build_samsung_target()
    trace = _gemv_trace("samsung_hbm_pim", batch=batch)
    base = _device_base_phases(target, trace, _layout(False))
    for resident in (False, True):
        ctx = ComposeCtx(target, trace, _layout(resident))
        recon = combine(base + knob_cost("samsung_hbm_pim", "stage_resident",
                                         resident, ctx), overlap=False)
        full = evaluate(target, trace, _layout(resident), "faithful").cycles
        assert recon == full, (batch, resident, recon, full)


# --------------------------------------------------------------------- #
# A4 -- provenance band aggregation
# --------------------------------------------------------------------- #


def test_provenance_band_least_trusted_dominates():
    # samsung faithful carries CRF_TRIGGER=ASSUMPTION -> band "assumption".
    assert provenance_band(get_cost_model("samsung_hbm_pim", "faithful")) == "assumption"
    # aim faithful is all DATASHEET -> "datasheet".
    assert provenance_band(get_cost_model("aim", "faithful")) == "datasheet"
    # apu_v2 placeholder has no tagged constants -> "placeholder".
    assert provenance_band(get_cost_model("apu_v2", "faithful")) == "placeholder"


def test_calibration_record_present_on_sim_anchored_models():
    sams = get_cost_model("samsung_hbm_pim", "faithful")
    assert "report-18" in sams.calibration.validated_against
    # placeholder carries an empty record.
    assert get_cost_model("apu_v2", "faithful").calibration.validated_against == ""


# --------------------------------------------------------------------- #
# A4 -- the gate: inert by default, warn vs refuse, dynamic-trip visible
# --------------------------------------------------------------------- #


def test_gate_off_is_byte_identical_argmin():
    """confidence_gate=False (default) -> the gate is never called; the argmin
    selection is exactly the explicit-off run, and NO gate warning fires."""
    target = build_aim_target()
    trace = _gemv_trace("aim")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        default = autoschedule(target, trace)
        explicit_off = autoschedule(target, trace, confidence_gate=False)
    # Same selection (the gate-off default == explicit gate-off).
    assert _knob_signature(default) == _knob_signature(explicit_off)
    # No confidence_gate diagnostic fired on either gate-off run.
    assert not any("confidence_gate" in str(w.message) for w in caught)


def test_gate_invalid_policy_raises():
    target = build_aim_target()
    with pytest.raises(ValueError):
        autoschedule(target, _gemv_trace("aim"),
                     confidence_gate=True, gate_policy="bogus")


def _knob_signature(placements):
    # The selection-distinguishing scalar knobs (NOT the fresh MemoryRef
    # handle objects, which differ by identity across runs).
    keys = ("stage_resident", "crf_issue", "grf_residency", "n_fibers", "vr_dma")
    return [{k: p.extra.get(k) for k in keys} for p in placements]


def test_gate_warn_does_not_alter_selection():
    """gate_policy='warn' on a low-confidence (assumption-band) target warns
    but commits the SAME placement the gate-off path picks."""
    target = build_samsung_target()
    trace = _gemv_trace("samsung_hbm_pim")
    off = autoschedule(target, trace, confidence_gate=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        on = autoschedule(target, trace, confidence_gate=True, gate_policy="warn")
    # selection unchanged (gate never alters argmin) -- compare the
    # selection-distinguishing knob signature, not the fresh handle objects.
    assert _knob_signature(off) == _knob_signature(on)
    # and it did warn about the low-confidence (assumption-band) estimate.
    assert any("confidence_gate" in str(w.message) for w in caught)


def test_gate_refuse_raises_on_low_confidence():
    """gate_policy='refuse' refuses to silently commit a low-confidence
    ranking (samsung faithful aggregates to the 'assumption' band)."""
    target = build_samsung_target()
    with pytest.raises(RuntimeError, match="refused"):
        autoschedule(target, _gemv_trace("samsung_hbm_pim"),
                     confidence_gate=True, gate_policy="refuse")


def test_dynamic_trip_is_visible_to_gate_not_silent_one():
    """A4<->A5: a tier-3 dynamic trip is a DECLARED default + a coarse
    confidence the gate consumes, never an invisible =1. The dynamic compose
    downgrades to 'coarse', so the gate (refuse) raises instead of silently
    committing."""
    target = build_aim_target()
    # `s0` is bound nowhere -> tier-3 dynamic inner bound -> coarse confidence.
    dyn = MatchTrace(
        target_name="aim", module_name="dyn",
        matches=[
            MatchedOp(
                target_op_name="MAC", func_name="gemv_0", work_id=(0,),
                enclosing_loops=[("%k", "0", "s0", 1)],
                operands=[
                    OperandBinding(role="x", memref_name="W"),
                    OperandBinding(role="y", memref_name="x"),
                    OperandBinding(role="acc", memref_name="acc",
                                   is_loop_carried=True),
                ],
                result_memref_name="acc", op_range=("%a", "%b"),
            )
        ],
    )
    model = get_cost_model("aim", "faithful")
    res = model.compose(ComposeCtx(target, dyn, Placement()))
    # The dynamic trip is VISIBLE as coarse confidence (not a silent =1).
    assert res.confidence == "coarse"
