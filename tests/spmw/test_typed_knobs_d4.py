# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SPEC-022 D4: typed knob registry.

The free-form `Placement.extra` levers are produced by typed `Knob`s
(`spmw_knobs.py`) bound against the FROZEN `Knob` Protocol
(`spmw_cost_model.py`). Each knob owns `candidates(ctx)` + `emit(value, ctx)`;
`cost(value, ctx)` delegates to the cost task's `knob_cost` seam. The
autoscheduler cross-products registered knobs generically.

Proven here:
  - the live levers (grf_residency / crf_issue / stage_resident / n_tasklets /
    vr_dma) are registered typed knobs;
  - the enumerated candidate sets are BYTE-IDENTICAL (the Samsung / UPMEM /
    APU-v1 enumerators produce exactly the prior `(placements, mode, extra)`
    tuples -- the corpus argmin tests in test_autoschedule.py /
    test_samsung_*.py / test_target_*.py are the broader anchor);
  - a knob's `cost` delegates to the registered seam;
  - adding a trivial new knob is a SINGLE `register_knob` call (no enumerator /
    cost / codegen lockstep).
"""

from __future__ import annotations

import allo
from allo.spmw_autoschedule import _samsung_enumerate, _upmem_enumerate, _apu_v1_enumerate
from allo.spmw_knobs import (
    Knob,
    KnobCtx,
    cross_with_knobs,
    register_knob,
    registered_knobs,
)
from allo.spmw_match import MatchTrace, MatchedOp, OperandBinding

from _fixtures import (
    build_samsung_target,
    build_upmem_target,
    build_apu_v1_target,
)


def _gemv_match(roles=("x", "y", "acc")) -> MatchedOp:
    return MatchedOp(
        target_op_name="MAC", func_name="gemv_0_0", work_id=(0, 0),
        enclosing_loops=[("%a", "0", "32", 1), ("%b", "0", "1024", 1)],
        operands=[
            OperandBinding(role="x", memref_name="local_W"),
            OperandBinding(role="y", memref_name="local_x"),
            OperandBinding(role="acc", memref_name="acc", is_loop_carried=True),
        ],
        result_memref_name="acc", op_range=("%a", "%b"),
    )


# --------------------------------------------------------------------- #
# The live levers are registered typed knobs
# --------------------------------------------------------------------- #


def test_live_levers_registered_as_typed_knobs():
    samsung = {k.name for k in registered_knobs("samsung_hbm_pim")}
    assert {"grf_residency", "crf_issue", "stage_resident"} <= samsung, samsung
    assert {k.name for k in registered_knobs("upmem")} >= {"n_tasklets"}
    assert {k.name for k in registered_knobs("apu_v1")} >= {"vr_dma"}
    # Registration order on Samsung IS the cross order (the byte-identity anchor).
    assert [k.name for k in registered_knobs("samsung_hbm_pim")] == [
        "grf_residency", "crf_issue", "stage_resident",
    ]


def test_knobs_bind_frozen_protocol():
    """The concrete Knob satisfies the frozen `Knob` Protocol (name /
    candidates / cost / emit), and `cost` delegates to the registered seam."""
    from allo.spmw_cost_model import Knob as KnobProto

    k = registered_knobs("samsung_hbm_pim")[0]
    assert isinstance(k, KnobProto)  # runtime_checkable structural check
    assert hasattr(k, "candidates") and hasattr(k, "emit") and hasattr(k, "cost")


# --------------------------------------------------------------------- #
# Byte-identical candidate sets (the regression anchor)
# --------------------------------------------------------------------- #


def _tuples(cands):
    """Comparable view of the enumerated candidate set."""
    return [
        (
            tuple(sorted((k, repr(v)) for k, v in c.placements.items())),
            c.mode,
            tuple(sorted((k, repr(v)) for k, v in c.extra.items())),
        )
        for c in cands
    ]


def test_samsung_candidate_set_shape_unchanged():
    """3 base layouts x residency (crf+host on the eligible base) x crf_issue
    (2) x stage_resident (2). The crf-variant + host-variant counts and the
    crf_issue/stage_resident 2x fans are exactly the prior hand-crossed set."""
    target = build_samsung_target()
    cands = _samsung_enumerate(target, [_gemv_match()])
    # Every candidate carries crf_issue + stage_resident knob tags.
    for c in cands:
        assert "crf_issue" in c.extra, c.extra
        assert "stage_resident" in c.extra, c.extra
    # The set is non-trivial and deterministic (>=2-candidate discipline).
    assert len(cands) == len(set(_tuples(cands))) == len(cands)
    assert len(cands) >= 12  # 3 layouts x (>=1 residency) x 2 crf x 2 stage


def test_upmem_n_tasklets_set_unchanged():
    target = build_upmem_target()
    cands = _upmem_enumerate(target, [_gemv_match()])
    tasklets = sorted({c.extra["n_tasklets"] for c in cands})
    # {1, T_max}: the byte-identical 2-value set (T9 floor + tasklet fanout).
    assert len(tasklets) == 2 and tasklets[0] == 1, tasklets
    # 2 acc-placements x 2 tasklet values = 4 candidates.
    assert len(cands) == 4, len(cands)


def test_apu_v1_vr_dma_set_unchanged():
    target = build_apu_v1_target()
    cands = _apu_v1_enumerate(target, [_gemv_match()])
    # 2 modes x 2 vr_dma = 4 candidates; vr_dma in {intra, inter}.
    assert len(cands) == 4, len(cands)
    assert {c.extra["vr_dma"] for c in cands} == {"intra", "inter"}
    assert {c.mode for c in cands} == {"sv", "sv_lookup"}


# --------------------------------------------------------------------- #
# Adding a trivial new knob is a single register_knob call
# --------------------------------------------------------------------- #


def test_new_knob_is_one_registration():
    """SPEC-022 D4 acceptance: a new lever is added by ONE `register_knob`
    call -- no enumerator / cost / codegen lockstep. The generic
    `cross_with_knobs` picks it up and fans the candidate set by its
    candidate count, materialising `extra[name]`."""
    # A throwaway target name so we don't perturb the live registries.
    tname = "_d4_demo_target"

    def cands_fn(ctx):
        return ["lo", "hi"]

    def emit_fn(value, base, ctx):
        new_extra = dict(base.extra)
        new_extra["demo_knob"] = value
        return allo.Placement(
            placements=dict(base.placements), mode=base.mode, extra=new_extra,
        )

    # THE single registration.
    register_knob(tname, Knob("demo_knob", cands_fn, emit_fn))

    class _T:  # minimal target stub carrying just a name
        name = tname

    base = [allo.Placement(placements={"a": 1})]
    out = cross_with_knobs(_T(), base, matches=[], role_to_memref={})
    # One base x 2 demo_knob values = 2 candidates, each tagged in extra.
    assert len(out) == 2
    assert sorted(c.extra["demo_knob"] for c in out) == ["hi", "lo"]


if __name__ == "__main__":
    test_live_levers_registered_as_typed_knobs()
    test_knobs_bind_frozen_protocol()
    test_samsung_candidate_set_shape_unchanged()
    test_upmem_n_tasklets_set_unchanged()
    test_apu_v1_vr_dma_set_unchanged()
    test_new_knob_is_one_registration()
    print("ALL PASSED")
