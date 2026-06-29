# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Design 07 D2 (task 006): the layout-derived bank-locality cost term.

A `Resource.LOCALITY` phase whose penalty is

    penalty = AccessDescr.conflict_count * ROW_BUFFER_MISS[target]

where `conflict_count` is layout-derived (the F2 conflict predicate
`LinearLayout.conflict_count`, the counting companion to
`describes_conflict_free`, design 07 §A2.4) and `ROW_BUFFER_MISS` is ONE
provenance-tagged per-backend row-buffer constant (read off the bound model,
never pasted in the compose body; SAFARI PIM line owns the quantity).

All STATIC (no simulator). What is proven:

  * `conflict_count(...) == 0` IFF `describes_conflict_free(...)` -- the
    invariant `AccessDescr` and the D2 penalty rely on.
  * Conflict-free layouts (everything `optimal_swizzle` emits) -> penalty 0
    -> NO LOCALITY phase -> the faithful corpus number is byte-identical
    (design 07 §D2.2).
  * A deliberately parity-colliding swizzle -> conflict_count > 0 -> a
    LOCALITY phase keyed on conflict_count * ROW_BUFFER_MISS -> the term
    SEPARATES a swizzle pair the faithful fold scores equal (the D2 flip).
  * The row-buffer constant is a provenance-tagged MoveCost (not a pasted
    magnitude).
"""
from __future__ import annotations

from allo.spmw_cost_model import (
    AccessDescr,
    ComposeCtx,
    MoveCostCtx,
    Provenance,
    Resource,
    combine,
    get_cost_model,
    phases_as_dict,
)
from allo.spmw_cost_tables import (
    _access_descr_from_layout,
    _locality_phase,
)
from allo.spmw_linear_layout import LinearLayout
from allo.spmw_match import MatchTrace, MatchedOp, OperandBinding
from allo.spmw_autoschedule import Placement

from _fixtures import build_aim_target, build_samsung_target, build_upmem_target


# --------------------------------------------------------------------- #
# Swizzle pair: conflict-free (good) vs parity-colliding (bad)
# --------------------------------------------------------------------- #


def _conflict_free_layout() -> LinearLayout:
    """Every nonzero x-vector reaches a nonzero bank bit -> 0 conflicts."""
    return LinearLayout(
        bases={"x": [(1,), (2,)]}, out_dims=("bank",), out_sizes=(4,)
    )


def _colliding_layout() -> LinearLayout:
    """The 2nd x basis bit contributes 0 to the bank -> the vector that
    activates only it maps to bank 0 (a collision). conflict_count == 1."""
    return LinearLayout(
        bases={"x": [(1,), (0,)]}, out_dims=("bank",), out_sizes=(4,)
    )


def test_conflict_count_matches_describes_conflict_free():
    good, bad = _conflict_free_layout(), _colliding_layout()
    for ll in (good, bad):
        cc = ll.conflict_count(bank_dims=("bank",), varying_inputs=("x",))
        cf = ll.describes_conflict_free(bank_dims=("bank",), varying_inputs=("x",))
        # invariant: conflict_count == 0  iff  describes_conflict_free
        assert (cc == 0) == cf, (ll, cc, cf)
    assert good.conflict_count(bank_dims=("bank",), varying_inputs=("x",)) == 0
    assert bad.conflict_count(bank_dims=("bank",), varying_inputs=("x",)) == 1


def test_optimal_swizzle_is_conflict_free_zero_count():
    """The swizzle the Samsung enumerator emits is conflict-free -> 0 count
    -> provably inert D2 penalty (design 07 §D2.2)."""
    base = LinearLayout.identity({"grf": 8, "bank": 16, "tile": 2},
                                 out_dims=("grf", "bank"))
    sw = LinearLayout.optimal_swizzle(
        base, vec_dims=("grf",), bank_dims=("bank",), segment_dims=("tile",)
    )
    assert sw.conflict_count(bank_dims=("bank",), varying_inputs=("bank",)) == 0


# --------------------------------------------------------------------- #
# AccessDescr derivation from the layout
# --------------------------------------------------------------------- #


def _layout_with(ll: LinearLayout) -> Placement:
    return Placement(extra={
        "bank_layout": ll,
        "bank_dims": ("bank",),
        "varying_inputs": ("x",),
    })


def test_access_descr_identity_when_no_layout():
    a = _access_descr_from_layout(Placement(extra={}))
    assert a.conflict_free is True and a.conflict_count == 0


def test_access_descr_conflict_free_layout_is_identity_like():
    a = _access_descr_from_layout(_layout_with(_conflict_free_layout()))
    assert a.conflict_free is True and a.conflict_count == 0


def test_access_descr_colliding_layout_counts():
    a = _access_descr_from_layout(_layout_with(_colliding_layout()))
    assert a.conflict_free is False and a.conflict_count == 1


def test_access_descr_explicit_access_passthrough():
    explicit = AccessDescr(tier="dram", conflict_free=False, conflict_count=3)
    a = _access_descr_from_layout(Placement(extra={"access": explicit}))
    assert a is explicit


# --------------------------------------------------------------------- #
# The locality Phase + the conflict-free-is-zero anchor
# --------------------------------------------------------------------- #


def test_locality_phase_none_when_conflict_free():
    model = get_cost_model("samsung_hbm_pim", "faithful")
    assert _locality_phase(model, _layout_with(_conflict_free_layout())) is None
    assert _locality_phase(model, Placement(extra={})) is None


def test_locality_phase_penalty_keyed_on_count_times_constant():
    model = get_cost_model("samsung_hbm_pim", "faithful")
    miss = model.move_cost("ROW_BUFFER_MISS", MoveCostCtx("ROW_BUFFER_MISS"))
    ph = _locality_phase(model, _layout_with(_colliding_layout()))
    assert ph is not None
    assert ph.resource is Resource.LOCALITY
    assert ph.tag == "locality"
    # conflict_count(1) * ROW_BUFFER_MISS -- the penalty traces to the
    # layout-derived count x the provenance-tagged constant, not a literal.
    assert ph.latency == 1 * miss


def test_row_buffer_miss_is_provenance_tagged():
    for target_name in ("samsung_hbm_pim", "aim", "upmem"):
        model = get_cost_model(target_name, "faithful")
        entry = model.move_costs["ROW_BUFFER_MISS"]
        assert entry.provenance in (Provenance.DATASHEET, Provenance.ASSUMPTION)
        assert entry.note  # carries a citation string


# --------------------------------------------------------------------- #
# The D2 argmin flip: the term separates a swizzle pair faithful ties
# --------------------------------------------------------------------- #


def _gemv_trace(target_name: str) -> MatchTrace:
    return MatchTrace(
        target_name=target_name, module_name="gemv",
        matches=[
            MatchedOp(
                target_op_name="MAC", func_name="gemv_0", work_id=(0,),
                enclosing_loops=[("%k", "0", "1024", 1)],
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


def test_d2_swizzle_pair_flip_samsung():
    """Two candidates identical except the bank swizzle: faithful (no
    locality phase) scores them EQUAL; the D2 locality term separates them
    (the colliding swizzle pays conflict_count * ROW_BUFFER_MISS more)."""
    target = build_samsung_target()
    model = get_cost_model("samsung_hbm_pim", "faithful")
    trace = _gemv_trace("samsung_hbm_pim")

    good = model.compose(ComposeCtx(target, trace, _layout_with(_conflict_free_layout())))
    bad = model.compose(ComposeCtx(target, trace, _layout_with(_colliding_layout())))

    # The conflict-free candidate emits NO locality phase (penalty 0).
    assert "locality" not in phases_as_dict(good.phases)
    # The colliding candidate emits a nonzero locality phase.
    assert phases_as_dict(bad.phases)["locality"] > 0

    # Under the faithful fold the good candidate is strictly cheaper -- the
    # term that finally makes a swizzle choice cost-relevant (D2.2).
    g = combine(list(good.phases), overlap=False)
    b = combine(list(bad.phases), overlap=False)
    assert b > g, (b, g)
    # The difference is EXACTLY the layout-derived penalty.
    miss = model.move_cost("ROW_BUFFER_MISS", MoveCostCtx("ROW_BUFFER_MISS"))
    assert b - g == 1 * miss


def test_d2_conflict_free_corpus_is_byte_identical():
    """A bare placement (no bank_layout) -> no locality phase -> the faithful
    estimate is identical to the pre-D2 number for every backend (the
    conflict-free-is-zero anchor, design 07 §D2.2)."""
    for build, name in (
        (build_samsung_target, "samsung_hbm_pim"),
        (build_aim_target, "aim"),
        (build_upmem_target, "upmem"),
    ):
        target = build()
        model = get_cost_model(name, "faithful")
        res = model.compose(ComposeCtx(target, _gemv_trace(name), Placement(extra={})))
        # No LOCALITY phase, and the fold reproduces .cycles exactly.
        assert all(p.resource is not Resource.LOCALITY for p in res.phases)
        assert combine(list(res.phases), overlap=False) == res.cycles
