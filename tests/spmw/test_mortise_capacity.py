# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mortise capacity-lever build acid tests (design 06; report-26).

All STATIC (no simulator, no hardware): the Mortise substrate is priced
purely via `compile_for_target(target, trace, backend="virtual")`.

What is proven here (the design-06 Phase-1 receipts):

1. **Structure-only target (acid test).** Every `Move`/`Op` on the Mortise
   tree carries no `cycles=` -- `mv.cycles is None`, `op.cycles is None`
   (design 04 §1.2 / design 06 §1). The only numbers on the tree are
   GEOMETRY, incl. the new lever const `resident_cap_elems`.

2. **The CostModel swap (the swap test).** Two cost models (faithful,
   unlimited), SAME target, SAME trace, ZERO device-tree edit -- only
   `cost_flavor` changes -- and the estimate changes (mirror of
   `test_phase5_demo.py`).

3. **The capacity lever reductions (design 06 §2.2).** phi=1+resident
   -> P+B*R; phi=1+baseline -> B*(P+R); phi=0.5+resident adds 0.5*P_var.

4. **The ablation (report-26 §3).** `cost_flavor="unlimited"` forces phi=1,
   so the resident-arm whole-program is C-INDEPENDENT (every capacity arm
   collapses to the report-18 curve); under faithful it spreads.

5. **Argmin-selected, not hand-set (design 06 §3.3).** The autoscheduler
   argmin over `stage_resident in {False, True}` picks resident at B>=2.

6. **Provenance (design 06 §2.3).** Every Mortise cost `note=` carries one
   of [sim-anchored]/[assumption]/[structural]/[datasheet].

7. **The anchor = numerical identity (report-26 §5).** Mortise faithful at
   C>=T_w + resident == Samsung faithful, byte-identical (P+B*(E+R)).

8. **Corpus untouched.** The five corpus models keep their flavors/argmin.
"""
from __future__ import annotations

import os

import pytest

import allo
from allo.spmw_cost_model import (
    ComposeCtx,
    MoveCostCtx,
    evaluate,
    get_cost_model,
)
from allo.spmw_cost_tables import (
    _samsung_preload_cycles,
    _samsung_readback_cycles,
    _samsung_mk,
)
from allo.spmw_match import MatchTrace, MatchedOp, OperandBinding
from allo.spmw_autoschedule import Placement, autoschedule, _mortise_enumerate, \
    _bucket_for_autoschedule

from _mortise_target import build_mortise_target, _T_W_FULL
from _fixtures import build_samsung_target


M, K = 4096, 1024


# --------------------------------------------------------------------- #
# Trace + layout helpers
# --------------------------------------------------------------------- #


def _trace(B: int, target_name: str = "mortise") -> MatchTrace:
    return MatchTrace(
        target_name=target_name,
        module_name="mortise_test",
        matches=[
            MatchedOp(
                target_op_name="MAC",
                func_name="gemv_0_0",
                work_id=(0, 0),
                enclosing_loops=[
                    ("%b", "0", str(B), 1),
                    ("%i", "0", "32", 1),
                    ("%k", "0", str(K), 1),
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
                extra={"batch_dim": B, "batch_loop_var": "%b"},
            )
        ],
    )


def _resident(flag: bool) -> Placement:
    return Placement(placements={}, extra={"stage_resident": flag})


def _hs(flavor: str = "faithful"):
    return get_cost_model("mortise", flavor, concern="host_staging")


# --------------------------------------------------------------------- #
# 1. Structure-only target (acid test)
# --------------------------------------------------------------------- #


def test_mortise_target_carries_no_cost_number():
    """design 04 §1.2 acid test for the Mortise tree: no node carries a
    cycle number; the only tree numbers are geometry (incl. the lever)."""
    t = build_mortise_target()
    for u in t._walk():
        for mv in u.moves.values():
            assert mv.cycles is None, (mv.name, mv.cycles)
        for op in u.ops.values():
            assert op.cycles is None, (op.name, op.cycles)
    # The lever IS on the tree -- as geometry, not cycles.
    assert t.resident_cap_elems == _T_W_FULL
    assert t.lanes_const == 16 and t.elem_bits_const == 16


def test_mortise_capacity_is_constructor_swept_not_source_edit():
    """The sweep seam (design 06 §3.1): C is a constructor kwarg, so the
    same source produces different capacities with zero edit."""
    full = build_mortise_target()
    half = build_mortise_target(resident_cap_elems=_T_W_FULL // 2)
    assert full.resident_cap_elems == _T_W_FULL
    assert half.resident_cap_elems == _T_W_FULL // 2


# --------------------------------------------------------------------- #
# 2. The CostModel swap (the swap test) -- mirror test_phase5_demo.py
# --------------------------------------------------------------------- #


def test_swap_two_models_same_target_zero_device_edit():
    """Two cost models, SAME target object, SAME trace, ZERO device-tree
    edit -- only cost_flavor changes -- and the estimate changes."""
    t = build_mortise_target(resident_cap_elems=_T_W_FULL // 2)
    trace = _trace(8)
    layout = _resident(True)
    faithful = allo.compile_for_target(
        t, trace, layout=layout, backend="virtual", cost_flavor="faithful"
    ).run().cycles
    unlimited = allo.compile_for_target(
        t, trace, layout=layout, backend="virtual", cost_flavor="unlimited"
    ).run().cycles
    assert faithful != unlimited, (faithful, unlimited)
    # faithful pays the eviction shortfall; unlimited (phi=1) does not.
    assert faithful > unlimited


def test_default_flavor_resolves_for_mortise():
    """A bare virtual compile (default cost_flavor='faithful') resolves --
    Mortise ships a usable default."""
    t = build_mortise_target()
    r = allo.compile_for_target(t, _trace(2), backend="virtual").run()
    assert r.cycles is not None and r.cycles > 0
    assert r.extra["cost_model"] == "mortise_faithful"
    assert r.extra["priced_target"] == "mortise"


# --------------------------------------------------------------------- #
# 3. The capacity-lever reductions (design 06 §2.2)
# --------------------------------------------------------------------- #


def test_reduction_phi1_resident_is_preload_once_plus_readback():
    """phi=1 (C>=T_w) + resident: host_staging = P + B*R."""
    t = build_mortise_target()                       # C = T_w -> phi=1
    hs = _hs()
    P = _samsung_preload_cycles(hs, M, K)
    R = _samsung_readback_cycles(hs, M)
    for B in (1, 2, 8):
        h = hs.compose(ComposeCtx(t, _trace(B), _resident(True)))
        assert h.cycles == P + B * R, (B, h.cycles, P, R)
        assert h.phases["evict_per_call"] == 0


def test_reduction_phi1_baseline_is_repreload_every_vector():
    """phi=1 + non-resident: host_staging = B*(P+R) (the comparator)."""
    t = build_mortise_target()
    hs = _hs()
    P = _samsung_preload_cycles(hs, M, K)
    R = _samsung_readback_cycles(hs, M)
    for B in (1, 2, 8):
        h = hs.compose(ComposeCtx(t, _trace(B), _resident(False)))
        assert h.cycles == B * (P + R), (B, h.cycles)


def test_reduction_phi_half_resident_adds_half_pvar_per_call():
    """phi=0.5 + resident: per-call adds 0.5*P_var (design 06 §2.2)."""
    t = build_mortise_target(resident_cap_elems=_T_W_FULL // 2)   # phi=0.5
    hs = _hs()
    P = _samsung_preload_cycles(hs, M, K)
    R = _samsung_readback_cycles(hs, M)
    crf = hs.move_cost("STAGE_CRF", MoveCostCtx("STAGE_CRF"))
    P_var = P - crf
    evict = round(0.5 * P_var)
    for B in (2, 8):
        h = hs.compose(ComposeCtx(t, _trace(B), _resident(True)))
        # P once + B*evict shortfall + B*R readback.
        assert h.cycles == P + B * evict + B * R, (B, h.cycles)
        assert h.phases["evict_per_call"] == B * evict


def test_capacity_monotone_collapse():
    """The resident-arm host_staging cost rises monotonically as C falls
    (more eviction always costs more) -- the structural collapse."""
    hs = _hs()
    prev = None
    for phi in (1.0, 0.75, 0.5, 0.25, 0.0):
        t = build_mortise_target(resident_cap_elems=int(phi * _T_W_FULL))
        h = hs.compose(ComposeCtx(t, _trace(8), _resident(True))).cycles
        if prev is not None:
            assert h >= prev, (phi, h, prev)
        prev = h


# --------------------------------------------------------------------- #
# 4. The ablation (report-26 §3): unlimited forces phi=1 for every C
# --------------------------------------------------------------------- #


def test_ablation_unlimited_flattens_all_capacities():
    """With the feature off (unlimited, phi=1), the resident-arm
    whole-program is C-INDEPENDENT: every capacity arm collapses to the
    report-18 curve. Under faithful it spreads. (report-26 §3.)"""
    layout = _resident(True)
    trace = _trace(8)
    unlimited_cycles = set()
    faithful_cycles = set()
    for phi in (1.0, 0.5, 0.1):
        t = build_mortise_target(resident_cap_elems=int(phi * _T_W_FULL))
        unlimited_cycles.add(
            evaluate(t, trace, layout, flavor="unlimited").cycles
        )
        faithful_cycles.add(
            evaluate(t, trace, layout, flavor="faithful").cycles
        )
    # Feature off: the spread vanishes (one value across all C).
    assert len(unlimited_cycles) == 1, unlimited_cycles
    # Feature on: the spread is real (C-attributable).
    assert len(faithful_cycles) == 3, faithful_cycles


def test_ablation_unlimited_equals_faithful_at_full_capacity():
    """At C=T_w (phi=1 already) the ablation is a no-op: faithful ==
    unlimited -- the only difference is the C-dependent term, which is
    already zero at full capacity."""
    t = build_mortise_target()                       # phi=1
    layout = _resident(True)
    for B in (2, 8):
        f = evaluate(t, _trace(B), layout, flavor="faithful").cycles
        u = evaluate(t, _trace(B), layout, flavor="unlimited").cycles
        assert f == u, (B, f, u)


# --------------------------------------------------------------------- #
# 5. Argmin-selected, not hand-set (design 06 §3.3)
# --------------------------------------------------------------------- #


def test_enumerator_emits_both_residency_candidates():
    t = build_mortise_target()
    _fn, matches = _bucket_for_autoschedule(_trace(2))[0]
    cands = _mortise_enumerate(t, matches)
    flags = {c.extra.get("stage_resident") for c in cands}
    assert flags == {False, True}, flags
    assert len(cands) >= 2


def test_argmin_picks_resident_at_b_ge_2():
    """The autoscheduler argmin (NOT a hand-set flag) picks the resident arm
    at B>=2 and the baseline at B=1 (the tie). Mortise has no regalloc
    capacity table, so the argmin runs with the documented kill-switch; the
    residency flag rides `extra`, so disabling regalloc cannot change which
    arm wins."""
    prev = os.environ.get("SPMW_DISABLE_REGALLOC")
    os.environ["SPMW_DISABLE_REGALLOC"] = "1"
    try:
        t = build_mortise_target()
        picks = {
            B: bool(autoschedule(t, _trace(B))[0].extra.get("stage_resident"))
            for B in (1, 2, 8)
        }
    finally:
        if prev is None:
            os.environ.pop("SPMW_DISABLE_REGALLOC", None)
        else:
            os.environ["SPMW_DISABLE_REGALLOC"] = prev
    assert picks[1] is False, picks      # B=1 tie -> baseline
    assert picks[2] is True, picks       # B>=2 -> resident earned
    assert picks[8] is True, picks


# --------------------------------------------------------------------- #
# 6. Provenance (design 06 §2.3): every note carries a tag
# --------------------------------------------------------------------- #


_PROVENANCE_TAGS = ("[sim-anchored]", "[assumption]", "[structural]", "[datasheet]")


@pytest.mark.parametrize("flavor", ["faithful", "unlimited", "optimistic"])
@pytest.mark.parametrize("concern", ["kernel_cycles", "host_staging"])
def test_every_mortise_cost_constant_is_provenance_tagged(flavor, concern):
    m = get_cost_model("mortise", flavor, concern=concern)
    entries = list(m.op_costs.values()) + list(m.move_costs.values())
    # host_staging has no op_costs; kernel_cycles has both -- either way every
    # populated entry must carry a tag.
    for entry in entries:
        assert any(tag in entry.note for tag in _PROVENANCE_TAGS), (
            flavor, concern, entry.note
        )


# --------------------------------------------------------------------- #
# 7. The anchor = numerical identity (report-26 §5)
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("B", [2, 4])
def test_anchor_mortise_faithful_equals_samsung_faithful_at_full_capacity(B):
    """Mortise faithful at C>=T_w + resident == Samsung faithful for the same
    shape/B, byte-identical (both = P + B*(E+R), report-18). This makes the
    report-26 §5 credibility bridge a NUMERICAL identity, not an analogy."""
    tm = build_mortise_target()                      # C = T_w -> phi=1
    ts = build_samsung_target()
    layout = _resident(True)
    m = evaluate(tm, _trace(B), layout, flavor="faithful").cycles
    s = evaluate(ts, _trace(B, "samsung_hbm_pim"), layout, flavor="faithful").cycles
    assert m == s, (B, m, s)


def test_mortise_recovers_report18_shape():
    """Sanity: the Mortise tree recovers the report-18 shape (M=4096,
    K=1024) and the report-18 preload/readback anchors (P=11368, R=181),
    so the capacity algebra rides the calibrated constants."""
    t = build_mortise_target()
    assert _samsung_mk(t, _trace(2)) == (M, K)
    hs = _hs()
    assert _samsung_preload_cycles(hs, M, K) == 11368
    assert _samsung_readback_cycles(hs, M) == 181


# --------------------------------------------------------------------- #
# 8. Corpus untouched
# --------------------------------------------------------------------- #


def test_corpus_models_untouched_by_mortise():
    for tn, fl in (
        ("samsung_hbm_pim", "faithful"),
        ("aim", "faithful"),
        ("upmem", "faithful"),
        ("apu_v1", "faithful"),
        ("apu_v2", "faithful"),
        ("demo_pim", "constant"),
    ):
        assert get_cost_model(tn, fl).confidence in (
            "calibrated", "placeholder", "coarse"
        )


if __name__ == "__main__":
    import sys

    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            # crude parametrize expansion for __main__ runs
            import inspect
            sig = inspect.signature(fn)
            try:
                if "B" in sig.parameters:
                    for B in (2, 4):
                        fn(B)
                elif "flavor" in sig.parameters:
                    for flv in ("faithful", "unlimited", "optimistic"):
                        for cn in ("kernel_cycles", "host_staging"):
                            fn(flv, cn)
                else:
                    fn()
            except Exception as e:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {e}")
    print("STATIC PASSED" if not failures else f"{failures} FAILURES")
    sys.exit(1 if failures else 0)
