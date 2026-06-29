# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Design 07 keystone reshape (task 004): the phase-timeline carrier + the
closed vocabulary + the provenance/calibration fields + the knob seam.

All STATIC (no simulator): these assert the *shape* of the reshape and the
byte-identical faithful fold. The simulator-anchored rank-preservation gate
(15251 / B*=2 / 3.93x) is the verifier's domain (task 009); here we assert:

  A1  -- Phase / phase_cycles / combine collapse rules; CostResult.phases is
         a list[Phase]; every faithful compose folds to its old `.cycles`.
  A2  -- enriched OpCostCtx/MoveCostCtx fields default-absent; AccessDescr
         identity is conflict-free (zero penalty) by construction.
  A4  -- OpCost/MoveCost carry a Provenance tag (default ASSUMPTION); the
         sim-anchored Samsung set is MEASURED; CostModel carries a
         CalibrationRecord.
  A5  -- an unknown op is a hard KeyError (the silent per_op=4 / GPR /
         add_cyc fallbacks are gone).
  A3  -- the stage_resident knob's cost(value, ctx) -> list[Phase] equals
         the host_staging compose's phases for that value (the seam is exact).
"""
from __future__ import annotations

import pytest

from allo.spmw_cost_model import (
    AccessDescr,
    CalibrationRecord,
    ComposeCtx,
    CostResult,
    MoveCost,
    MoveCostCtx,
    OpCost,
    OpCostCtx,
    Phase,
    Provenance,
    Resource,
    combine,
    get_cost_model,
    knob_cost,
    phase_cycles,
    phases_as_dict,
)
from allo.spmw_match import MatchTrace, MatchedOp, OperandBinding
from allo.spmw_autoschedule import Placement

from _fixtures import build_aim_target, build_samsung_target, build_upmem_target


# --------------------------------------------------------------------- #
# Trace + layout helpers
# --------------------------------------------------------------------- #


def _mac(ub: str, target_name: str = "aim") -> MatchedOp:
    return MatchedOp(
        target_op_name="MAC",
        func_name="gemv_0",
        work_id=(0,),
        enclosing_loops=[("%k", "0", ub, 1)],
        operands=[
            OperandBinding(role="x", memref_name="W"),
            OperandBinding(role="y", memref_name="x"),
            OperandBinding(role="acc", memref_name="acc", is_loop_carried=True),
        ],
        result_memref_name="acc",
        op_range=("%a", "%b"),
    )


def _trace(target_name: str, op_name: str = "MAC", ub: str = "1024") -> MatchTrace:
    m = _mac(ub, target_name)
    m.target_op_name = op_name
    return MatchTrace(target_name=target_name, module_name="t", matches=[m])


def _layout(**extra) -> Placement:
    return Placement(placements={}, extra=dict(extra))


# --------------------------------------------------------------------- #
# A1 -- Phase / phase_cycles / combine
# --------------------------------------------------------------------- #


def test_phase_cycles_collapsed_encoding():
    # latency=total, ii=0, count=1 -> total.
    assert phase_cycles(Phase(Resource.COMPUTE, latency=42, ii=0, count=1)) == 42


def test_phase_cycles_loop_encoding():
    # latency=per_op, ii=per_op, count=iters -> per_op*iters.
    assert phase_cycles(Phase(Resource.COMPUTE, latency=4, ii=4, count=1024)) == 4 * 1024


def test_phase_cycles_zero_count():
    assert phase_cycles(Phase(Resource.HOST, latency=99, ii=3, count=0)) == 0


def test_combine_faithful_sums_within_and_across():
    phases = [
        Phase(Resource.COMPUTE, 100, 0, 1),
        Phase(Resource.COMPUTE, 50, 0, 1),     # sum within COMPUTE -> 150
        Phase(Resource.HOST, 30, 0, 1),        # + HOST 30
        Phase(Resource.DMA, 20, 0, 1),         # + DMA 20
    ]
    assert combine(phases, overlap=False) == 200


def test_combine_overlap_fold_is_implemented():
    # design 07 D1 (task 005) filled the overlap fold: a lone COMPUTE phase
    # folds to itself under both rules (no DMA to hide). The detailed
    # max-across + fill/drain semantics live in test_overlap_flavor.py.
    assert combine([Phase(Resource.COMPUTE, 7, 0, 1)], overlap=True) == 7
    assert combine([Phase(Resource.COMPUTE, 7, 0, 1)], overlap=False) == 7


def test_costresult_phases_is_list():
    r = CostResult(cycles=5, phases=[Phase(Resource.COMPUTE, 5, 0, 1, "exec")])
    assert isinstance(r.phases, list)
    assert all(isinstance(p, Phase) for p in r.phases)


def test_phases_as_dict_groups_by_tag():
    phases = [
        Phase(Resource.COMPUTE, 10, 0, 1, "exec"),
        Phase(Resource.HOST, 3, 0, 1, "readback"),
        Phase(Resource.HOST, 7, 0, 1, "readback"),  # same tag -> summed
    ]
    d = phases_as_dict(phases)
    assert d == {"exec": 10, "readback": 10}


# --------------------------------------------------------------------- #
# A1 byte-identical -- every faithful compose folds to its `.cycles`
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "build, target_name",
    [
        (build_aim_target, "aim"),
        (build_upmem_target, "upmem"),
        (build_samsung_target, "samsung_hbm_pim"),
    ],
)
def test_faithful_compose_fold_equals_cycles(build, target_name):
    """The combiner over a compose's phases reproduces its `.cycles` exactly
    (design 07 §A1.6 collapsed-encoding proof, device-only arm)."""
    target = build()
    model = get_cost_model(target_name, "faithful")
    res = model.compose(ComposeCtx(target, _trace(target_name), _layout()))
    assert combine(list(res.phases), overlap=False) == res.cycles


def test_apu_v1_split_dma_phase_preserves_total():
    """APU v1 SPLITS the vr_dma move cost onto Resource.DMA (design 07
    §A1.5); under the faithful sum-across fold the total is unchanged."""
    from _fixtures import build_apu_v1_target

    target = build_apu_v1_target()
    model = get_cost_model("apu_v1", "faithful")
    layout = Placement(placements={}, mode="sv_lookup", extra={"vr_dma": "inter"})
    res = model.compose(ComposeCtx(target, _trace("apu_v1"), layout))
    resources = {p.resource for p in res.phases}
    assert Resource.DMA in resources       # the vr_dma phase is split out
    assert Resource.COMPUTE in resources
    assert combine(list(res.phases), overlap=False) == res.cycles


# --------------------------------------------------------------------- #
# A2 -- enriched ctx fields + AccessDescr identity
# --------------------------------------------------------------------- #


def test_op_cost_ctx_new_fields_default_absent():
    c = OpCostCtx("MAC")
    assert c.placement == {}
    assert c.access is None
    assert c.live_set is None
    assert c.dtype_bits is None


def test_move_cost_ctx_new_fields_default_absent():
    c = MoveCostCtx("LD_A")
    assert c.placement == {}
    assert c.access is None
    assert c.live_set is None


def test_access_descr_identity_is_conflict_free():
    a = AccessDescr.identity()
    assert a.conflict_free is True
    assert a.conflict_count == 0
    assert a.stride == 1
    assert a.tier == "register"


def test_access_descr_conflict_invariant():
    # conflict_free=True with a nonzero count is a hard error.
    with pytest.raises(ValueError):
        AccessDescr(tier="near_bank", conflict_free=True, conflict_count=3)
    # conflict_free=False with a count type-checks.
    AccessDescr(tier="near_bank", conflict_free=False, conflict_count=3)


# --------------------------------------------------------------------- #
# A4 -- provenance + calibration
# --------------------------------------------------------------------- #


def test_op_move_cost_default_provenance_is_assumption():
    assert OpCost(lambda c: 1).provenance is Provenance.ASSUMPTION
    assert MoveCost(lambda c: 1).provenance is Provenance.ASSUMPTION


def test_samsung_sim_anchored_set_is_measured():
    model = get_cost_model("samsung_hbm_pim", "faithful")
    assert model.op_costs["MAC"].provenance is Provenance.MEASURED
    hs = get_cost_model("samsung_hbm_pim", "faithful", concern="host_staging")
    assert hs.move_costs["STAGE_BCAST"].provenance is Provenance.MEASURED
    assert hs.move_costs["GATHER_RD"].provenance is Provenance.MEASURED
    # CRF_TRIGGER is a genuine guess -> ASSUMPTION (the honest worst case).
    assert model.move_costs["CRF_TRIGGER"].provenance is Provenance.ASSUMPTION


def test_calibration_record_on_sim_anchored_model():
    model = get_cost_model("samsung_hbm_pim", "faithful")
    assert isinstance(model.calibration, CalibrationRecord)
    assert "report-18" in model.calibration.validated_against
    assert model.calibration.residual_error == 0.0
    assert model.calibration.shape_coverage == ((4096, 1024),)


def test_placeholder_model_has_empty_calibration():
    model = get_cost_model("apu_v2", "faithful")
    assert model.calibration.validated_against == ""


# --------------------------------------------------------------------- #
# A5 -- closed vocabulary: unknown op is a hard error
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("target_name", ["aim", "upmem"])
def test_unknown_op_is_hard_error(target_name):
    from _fixtures import build_aim_target, build_upmem_target

    build = {"aim": build_aim_target, "upmem": build_upmem_target}[target_name]
    target = build()
    model = get_cost_model(target_name, "faithful")
    # NOT_AN_OP is in no backend's op_costs: A5 raises instead of the old
    # silent per_op=4 / GPR-baseline default.
    bad = _trace(target_name, op_name="NOT_AN_OP")
    with pytest.raises(KeyError):
        model.compose(ComposeCtx(target, bad, _layout()))


def test_apu_v1_unknown_op_is_hard_error():
    from _fixtures import build_apu_v1_target

    target = build_apu_v1_target()
    model = get_cost_model("apu_v1", "faithful")
    bad = _trace("apu_v1", op_name="NOT_AN_OP")
    layout = Placement(placements={}, mode="sv_lookup", extra={})
    with pytest.raises(KeyError):
        model.compose(ComposeCtx(target, bad, layout))


# --------------------------------------------------------------------- #
# A3 -- the knob seam is exact (the cost side of the co-owned Knob)
# --------------------------------------------------------------------- #


def test_stage_resident_knob_cost_equals_host_staging_phases():
    """The `stage_resident` knob's cost(value, ctx) -> list[Phase] reproduces
    the Samsung host_staging compose's phases for that value, byte-for-byte
    (design 07 §A3.2/§A3.3 -- the seam is exact, not approximate)."""
    target = build_samsung_target()
    hs = get_cost_model("samsung_hbm_pim", "faithful", concern="host_staging")
    trace = _gemv_trace()
    for resident in (False, True):
        layout = _layout(stage_resident=resident)
        direct = hs.compose(ComposeCtx(target, trace, layout))
        ctx = ComposeCtx(target, trace, layout)
        knob_phases = knob_cost("samsung_hbm_pim", "stage_resident", resident, ctx)
        # Folded total matches the host_staging compose's total.
        assert combine(knob_phases, overlap=False) == direct.cycles
        # And the per-tag breakdown matches phase-for-phase.
        assert phases_as_dict(knob_phases) == phases_as_dict(list(direct.phases))


def test_unregistered_knob_contributes_nothing():
    target = build_samsung_target()
    ctx = ComposeCtx(target, _gemv_trace(), _layout())
    assert knob_cost("samsung_hbm_pim", "no_such_knob", 0, ctx) == []


# --------------------------------------------------------------------- #
# A Samsung GEMV trace fixture (M/K resolvable -> P/E/R all nonzero).
# --------------------------------------------------------------------- #


def _gemv_trace() -> MatchTrace:
    m = MatchedOp(
        target_op_name="MAC",
        func_name="gemv_0",
        work_id=(0,),
        enclosing_loops=[("%i", "0", "4096", 1), ("%k", "0", "1024", 1)],
        operands=[
            OperandBinding(role="x", memref_name="W"),
            OperandBinding(role="y", memref_name="x"),
            OperandBinding(role="acc", memref_name="acc", is_loop_carried=True),
        ],
        result_memref_name="acc",
        op_range=("%a", "%b"),
    )
    return MatchTrace(
        target_name="samsung_hbm_pim", module_name="gemv", matches=[m]
    )
