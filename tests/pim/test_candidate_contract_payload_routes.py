# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The shared candidate contract on the payload routes (round 140, ISSUES
X-45): the APU v1 vector plans and the Gemini-II source plans declare their
effects from their executable payloads, and the contract compares a
declaration with that derivation before costing and selection.  An edited
declaration (packing, storage, transfer route, allocation) of one candidate
rejects that candidate and lets a faithful alternative be selected; edited
declarations of every candidate fail the compilation; a supplied placement
with an edited declaration is refused; the independent realization checks
refuse an edited declaration as well; a required target with neither a
deriver nor a payload is not accepted on its declaration's word."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import allo
from allo.ir.types import float16, int4, int15
from allo.pim import apu_g2_source_route as g2_route
from allo.pim import apu_v1_vector_program as v1_route
from allo.pim.costs import apu_g2_cost, apu_v1_cost
from allo.pim.targets import build_apu_g2_target, build_apu_v1_target
from allo.spmw_autoschedule import Placement, _bucket_for_autoschedule
from allo.spmw_candidate import (
    candidate_effects,
    derive_candidate_effects,
    validate_candidate_effects,
)
from allo.spmw_pipeline import matcher_view


def hazel(fern: float16[3, 8], reed: float16[8, 2], leaf: float16[3, 2]):
    for row, column in allo.grid(3, 2):
        for depth in range(8):
            leaf[row, column] += fern[row, depth] * reed[depth, column]


def rowan(bark: int4[3, 8], moss: int4[8, 2], cone: int15[3, 2]):
    for row, column in allo.grid(3, 2):
        acc: int15 = 0
        for depth in range(8):
            acc += bark[row, depth] * moss[depth, column]
        cone[row, column] = acc


ROUTES = {
    "apu_v1_vector": (
        v1_route,
        "apu_v1_vector_candidates",
        hazel,
        build_apu_v1_target,
        apu_v1_cost,
        "functional",
    ),
    "apu_g2_source": (
        g2_route,
        "apu_g2_source_candidates",
        rowan,
        build_apu_g2_target,
        apu_g2_cost,
        "virtual",
    ),
}


def _edit(effects, field):
    """One edited field of a declaration: a packing, a storage, a transfer
    route or an allocation the executable plan does not have."""
    if field == "packing":
        return replace(
            effects,
            value_layouts=tuple(
                replace(v, packing="edited_packing") for v in effects.value_layouts
            ),
        )
    if field == "storage":
        return replace(
            effects,
            value_layouts=tuple(
                replace(v, storage="edited_storage") for v in effects.value_layouts
            ),
        )
    if field == "transfer_route":
        return replace(
            effects,
            transfers=tuple(
                replace(t, route=tuple(t.route) + ("edited_hop",))
                for t in effects.transfers
            ),
        )
    if field == "allocation":
        return replace(
            effects,
            allocations=tuple(effects.allocations) + (("edited_scratch", 64, "l4"),),
        )
    raise KeyError(field)


def _compile(route, layout=None):
    _module, _symbol, fn, build_target, cost, backend = ROUTES[route]
    kwargs = {"layout": layout} if layout is not None else {}
    return allo.compile(fn, build_target(), cost, backend=backend, **kwargs)


def _single_placement(compiled):
    layout = compiled.compiled.layout
    (placement,) = layout if isinstance(layout, (list, tuple)) else [layout]
    return placement


def _corrupting(module, symbol, field, which):
    """Wrap the route's enumerator so that the candidates named by ``which``
    (a set of plan names, or ``"all"``) declare an edited ``field``; the
    payloads, the executable plans, are untouched."""
    original = getattr(module, symbol)

    def corrupted(view, matches):
        placements = original(view, matches)
        for placement in placements:
            if which == "all" or placement.mode in which:
                placement.extra["effects"] = _edit(placement.extra["effects"], field)
        return placements

    return corrupted


@pytest.fixture(scope="module")
def stock():
    return {route: _compile(route) for route in ROUTES}


@pytest.mark.parametrize("route", sorted(ROUTES))
def test_payload_routes_declare_their_payloads_derivation(stock, route):
    module, symbol, _fn, build_target, _cost, _backend = ROUTES[route]
    compiled = stock[route]
    placement = _single_placement(compiled)
    target = build_target()
    view = matcher_view(target)
    ((_scope, matches),) = _bucket_for_autoschedule(compiled.trace)
    declared = candidate_effects(placement)
    derived = derive_candidate_effects(view, matches, placement)
    assert declared is not None and derived == declared
    assert derived == placement.extra["payload"].effects()
    assert validate_candidate_effects(matches, placement, view) == []
    # every enumerated candidate declares its payload's derivation
    for candidate in getattr(module, symbol)(view, matches):
        assert candidate_effects(candidate) == candidate.extra["payload"].effects()


@pytest.mark.parametrize(
    "field", ["packing", "storage", "transfer_route", "allocation"]
)
@pytest.mark.parametrize("route", sorted(ROUTES))
def test_one_edited_candidate_is_rejected_and_a_faithful_alternative_selected(
    monkeypatch, stock, route, field
):
    module, symbol, _fn, build_target, _cost, _backend = ROUTES[route]
    stock_mode = _single_placement(stock[route]).mode
    monkeypatch.setattr(
        module, symbol, _corrupting(module, symbol, field, {stock_mode})
    )
    compiled = _compile(route)
    placement = _single_placement(compiled)
    assert placement.mode != stock_mode
    assert candidate_effects(placement) == placement.extra["payload"].effects()
    view = matcher_view(build_target())
    ((_scope, matches),) = _bucket_for_autoschedule(compiled.trace)
    assert validate_candidate_effects(matches, placement, view) == []
    compiled.compiled.layout_ctx.check_realization([placement])


@pytest.mark.parametrize(
    "field", ["packing", "storage", "transfer_route", "allocation"]
)
@pytest.mark.parametrize("route", sorted(ROUTES))
def test_every_candidate_edited_fails_the_compilation(monkeypatch, route, field):
    module, symbol, _fn, _build_target, _cost, _backend = ROUTES[route]
    monkeypatch.setattr(module, symbol, _corrupting(module, symbol, field, "all"))
    with pytest.raises(Exception, match="violates the candidate contract"):
        _compile(route)


@pytest.mark.parametrize("route", sorted(ROUTES))
def test_supplied_placement_with_an_edited_declaration_is_refused(stock, route):
    placement = _single_placement(stock[route])
    tampered = Placement(
        placements=dict(placement.placements),
        mode=placement.mode,
        extra={
            **placement.extra,
            "effects": _edit(candidate_effects(placement), "packing"),
        },
        layout=getattr(placement, "layout", None),
    )
    with pytest.raises(ValueError, match="violates the candidate contract"):
        _compile(route, layout=[tampered])
    # the same placement without its declaration compiles to the same program
    handle_only = Placement(
        placements=dict(placement.placements),
        mode=placement.mode,
        extra={k: v for k, v in placement.extra.items() if k != "effects"},
        layout=getattr(placement, "layout", None),
    )
    again = _compile(route, layout=[handle_only])
    assert again.compiled.cmds == stock[route].compiled.cmds
    assert candidate_effects(_single_placement(again)) == candidate_effects(placement)


@pytest.mark.parametrize(
    "field", ["packing", "storage", "transfer_route", "allocation"]
)
@pytest.mark.parametrize("route", sorted(ROUTES))
def test_realization_check_refuses_an_edited_declaration(stock, route, field):
    compiled = stock[route]
    placement = _single_placement(compiled)
    tampered = Placement(
        placements=dict(placement.placements),
        mode=placement.mode,
        extra={
            **placement.extra,
            "effects": _edit(candidate_effects(placement), field),
        },
        layout=getattr(placement, "layout", None),
    )
    with pytest.raises(RuntimeError, match="differs? from the candidate contract"):
        compiled.compiled.layout_ctx.check_realization([tampered])
    compiled.compiled.layout_ctx.check_realization([placement])


@pytest.mark.parametrize("route", sorted(ROUTES))
def test_a_declaration_without_any_derivation_is_not_accepted(stock, route):
    """A placement of a required target that carries a declaration but no
    payload (and whose target has no deriver for it) is a contract problem,
    not an accepted declaration."""
    _module, _symbol, _fn, build_target, _cost, _backend = ROUTES[route]
    compiled = stock[route]
    placement = _single_placement(compiled)
    view = matcher_view(build_target())
    ((_scope, matches),) = _bucket_for_autoschedule(compiled.trace)
    bare = Placement(
        placements=dict(placement.placements),
        mode=placement.mode,
        extra={k: v for k, v in placement.extra.items() if k != "payload"},
        layout=getattr(placement, "layout", None),
    )
    problems = validate_candidate_effects(matches, bare, view)
    assert problems, "a declaration with no derivation was accepted"
    if route == "apu_g2_source":
        # Gemini-II registers no deriver: without the payload nothing can
        # check the declaration, and that is a problem in itself.
        assert any(
            "no effects derivation is available" in p for p in problems
        ), problems
    else:
        # APU v1's registered deriver is the grouped route's: a vector
        # placement without its payload is not a grouped placement and the
        # derivation rejects it.
        assert any("effects derivation rejects" in p for p in problems), problems


def test_apu_v1_vector_neighbor_computes_exactly_with_a_faithful_declaration(
    monkeypatch, stock
):
    """The route whose stock plan declares an edited packing is rejected and
    the faithful alternative executes the neighbouring contraction exactly on
    the functional backend."""
    module, symbol, fn, build_target, cost, backend = ROUTES["apu_v1_vector"]
    stock_mode = _single_placement(stock["apu_v1_vector"]).mode
    monkeypatch.setattr(
        module, symbol, _corrupting(module, symbol, "packing", {stock_mode})
    )
    compiled = _compile("apu_v1_vector")
    a = (np.arange(24, dtype=np.float16).reshape(3, 8) % 4).astype(np.float16)
    b = (np.arange(16, dtype=np.float16).reshape(8, 2) % 3).astype(np.float16)
    c = np.full((3, 2), 2, dtype=np.float16)
    expected = (a.astype(np.float32) @ b.astype(np.float32) + 2).astype(np.float16)
    compiled(a, b, c)
    np.testing.assert_array_equal(c, expected)
