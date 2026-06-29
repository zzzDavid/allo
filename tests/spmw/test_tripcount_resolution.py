# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""D3 trip-count resolution (design 04 §3).

`resolve_trip_count(match, loop_idx, *, shapes, mapping_env)` resolves a
`MatchedOp`'s enclosing-loop bound in three escalating tiers: literal /
affine-over-shapes+mapping / declared dynamic fallback (None). The
string-level core is `resolve_bound_text`. These are STATIC assertions
(no simulator). Tier-3 (None) is what makes `compose` apply a DECLARED
default and downgrade `confidence="coarse"` instead of the silent =1.
"""
from __future__ import annotations

from allo.spmw_cost_model import ComposeCtx, get_cost_model, phases_as_dict
from allo.spmw_match import MatchedOp, MatchTrace, OperandBinding
from allo.spmw_tripcount import (
    _parse_loop_bound,
    resolve_bound_text,
    resolve_trip_count,
)


def _mac(ub: str, outer: str | None = None) -> MatchedOp:
    loops = []
    if outer is not None:
        loops.append(("%i", "0", outer, 1))
    loops.append(("%k", "0", ub, 1))
    return MatchedOp(
        target_op_name="MAC",
        func_name="gemv_0",
        work_id=(0,),
        enclosing_loops=loops,
        operands=[
            OperandBinding(role="x", memref_name="W"),
            OperandBinding(role="y", memref_name="x"),
            OperandBinding(role="acc", memref_name="acc", is_loop_carried=True),
        ],
        result_memref_name="acc",
        op_range=("%a", "%b"),
    )


# --------------------------------------------------------------------- #
# resolve_trip_count (MatchedOp-facing, the design-04 §3.2 signature)
# --------------------------------------------------------------------- #


def test_match_innermost_literal():
    assert resolve_trip_count(_mac("1024")) == 1024


def test_match_loop_idx_selects_outer():
    m = _mac("1024", outer="32")
    assert resolve_trip_count(m, 0) == 32
    assert resolve_trip_count(m, -1) == 1024


def test_match_no_loops_returns_none():
    m = _mac("1024")
    m.enclosing_loops = []
    assert resolve_trip_count(m) is None


def test_match_out_of_range_idx_returns_none():
    assert resolve_trip_count(_mac("1024"), 5) is None


def test_match_affine_over_shapes():
    m = _mac("M * K floordiv 16")
    assert resolve_trip_count(m, shapes={"M": 4096, "K": 1024}) == (
        4096 * 1024 // 16
    )


# --------------------------------------------------------------------- #
# Tier 1 -- literal (resolve_bound_text)
# --------------------------------------------------------------------- #


def test_tier1_bare_literal():
    assert resolve_bound_text("1024") == 1024


def test_tier1_affine_map_wrapped_literal():
    assert resolve_bound_text("() -> (1024)") == 1024


def test_tier1_sole_number_helper():
    assert resolve_bound_text("d0 ceildiv 1") is None  # ceildiv -> tier3
    assert _parse_loop_bound("foo 256 bar") == 256


# --------------------------------------------------------------------- #
# Tier 2 -- affine over operand shapes + mapping params
# --------------------------------------------------------------------- #


def test_tier2_product_over_shapes():
    assert resolve_bound_text(
        "M * K floordiv 16", shapes={"M": 4096, "K": 1024}
    ) == 4096 * 1024 // 16


def test_tier2_mapping_param():
    assert resolve_bound_text("512 * c0", mapping_env={"c0": 8}) == 4096


def test_tier2_affine_map_form():
    assert resolve_bound_text(
        "(d0) -> (d0 * 512)", mapping_env={"d0": 4}
    ) == 2048


def test_tier2_mod_normalised():
    assert resolve_bound_text("K mod 100", shapes={"K": 1024}) == 24


def test_tier2_mapping_wins_on_name_clash():
    assert resolve_bound_text(
        "n", shapes={"n": 1}, mapping_env={"n": 9}
    ) == 9


# --------------------------------------------------------------------- #
# Tier 3 -- genuinely dynamic -> None (the honesty seam)
# --------------------------------------------------------------------- #


def test_tier3_unbound_symbol_returns_none():
    assert resolve_bound_text("s0 * 4", shapes={"M": 4096}) is None


def test_tier3_no_env_nonliteral_returns_none():
    assert resolve_bound_text("M * K") is None


def test_tier3_ceildiv_not_supported_returns_none():
    assert resolve_bound_text("M ceildiv 16", shapes={"M": 4096}) is None


def test_tier3_rejects_unsafe_alphabet():
    assert resolve_bound_text("M.__class__", shapes={"M": 4096}) is None


# --------------------------------------------------------------------- #
# Tier 3 in compose: declared default + dynamic_assumed marker + coarse
# (design 04 §3.2 -- never the silent =1)
# --------------------------------------------------------------------- #


def _dynamic_trace(target_name: str) -> MatchTrace:
    # A data-dependent inner bound the resolver cannot bind -> tier 3.
    return MatchTrace(
        target_name=target_name,
        module_name="dyn",
        matches=[_mac("s0")],  # `s0` bound nowhere
    )


def _resolvable_trace(target_name: str) -> MatchTrace:
    return MatchTrace(
        target_name=target_name,
        module_name="ok",
        matches=[_mac("1024")],
    )


def _layout():
    from allo.spmw_autoschedule import Placement

    return Placement(placements={})


def test_compose_dynamic_marks_coarse_aim():
    from _fixtures import build_aim_target

    target = build_aim_target()
    model = get_cost_model("aim", "faithful")
    res = model.compose(ComposeCtx(target, _dynamic_trace("aim"), _layout()))
    assert res.confidence == "coarse"
    # design 07 §A1.2: `dynamic_assumed` is now a tagged (zero-cost) Phase in
    # the list carrier, a visible breakdown flag (was the dict `=1` flag).
    assert "dynamic_assumed" in phases_as_dict(res.phases)
    # declared default = 1 iter -> MAC_SBK (8) * 1.
    assert res.cycles == 8


def test_compose_resolvable_stays_calibrated_aim():
    from _fixtures import build_aim_target

    target = build_aim_target()
    model = get_cost_model("aim", "faithful")
    res = model.compose(ComposeCtx(target, _resolvable_trace("aim"), _layout()))
    assert res.confidence == "calibrated"
    assert "dynamic_assumed" not in phases_as_dict(res.phases)
    assert res.cycles == 8 * 1024


def test_compose_dynamic_marks_coarse_upmem():
    from _fixtures import build_upmem_target

    target = build_upmem_target()
    model = get_cost_model("upmem", "faithful")
    res = model.compose(ComposeCtx(target, _dynamic_trace("upmem"), _layout()))
    assert res.confidence == "coarse"
    assert "dynamic_assumed" in phases_as_dict(res.phases)


def test_compose_dynamic_marks_coarse_samsung():
    from _fixtures import build_samsung_target

    target = build_samsung_target()
    model = get_cost_model("samsung_hbm_pim", "faithful")
    res = model.compose(
        ComposeCtx(target, _dynamic_trace("samsung_hbm_pim"), _layout())
    )
    assert res.confidence == "coarse"
    assert "dynamic_assumed" in phases_as_dict(res.phases)


def test_declared_default_overridable():
    """A model can declare a per-op fallback other than 1; compose uses it."""
    from _fixtures import build_aim_target

    target = build_aim_target()
    model = get_cost_model("aim", "faithful")
    saved = dict(model.dynamic_trip_defaults)
    model.dynamic_trip_defaults["MAC"] = 7
    try:
        res = model.compose(
            ComposeCtx(target, _dynamic_trace("aim"), _layout())
        )
        # MAC_SBK (8) * declared 7.
        assert res.cycles == 8 * 7
        assert res.confidence == "coarse"
    finally:
        model.dynamic_trip_defaults.clear()
        model.dynamic_trip_defaults.update(saved)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("STATIC PASSED")
