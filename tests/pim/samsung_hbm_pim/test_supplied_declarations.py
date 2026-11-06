# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A caller-supplied placement's declared effects at the public boundary.

A supplied declaration is a claim about the program: it is normalized onto
the work item it is supplied for (function and value bindings by operand
role), validated in full, and must be the current candidate's declaration
for the same decision; otherwise ``allo.compile`` refuses it (reviewer
round 38 reproducer: an edited image, buffer identity, launch domain,
allocation or axis set; round 39: a second, contradicting layout of one
value and image or transfer of one value and direction, in either order).  Legitimate reuse is preserved: one body's
declaration serves every body of a lockstep kernel, a bare decision (no
declaration) is rebound, and another program's plan supplies its decision.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

import allo
from allo.dataflow import region as _df_region
from allo.ir.types import float16
from allo.pim.costs import samsung_cost
from allo.pim.targets import build_samsung_target
from allo.spmw_autoschedule import (
    Placement,
    _bucket_for_autoschedule,
    _samsung_enumerate,
)
from allo.spmw_candidate import validate_candidate_effects
from allo.spmw_knobs import reset_active_liveness, set_active_liveness

M, K, N = 23, 81, 2
REASSOCIATE = dict(reduction_order="reassociate", backend="virtual")


def contraction(left: float16[M, K], right: float16[K, N], result: float16[M, N]):
    for row, col in allo.grid(M, N):
        total: float16 = 0
        for depth in allo.reduction(K):
            total += left[row, depth] * right[depth, col]
        result[row, col] = total


@_df_region()
def lockstep_gemv(W: float16[256, 100], x: float16[100], y: float16[256]):
    @allo.work(mapping=[16], args=[W, x, y])
    def kernel(lW: float16[256, 100], lx: float16[100], ly: float16[256]):
        pid = allo.get_wid()
        row0 = pid * 16
        for i in range(16):
            acc: float16 = 0
            for k in range(100):
                acc += lW[row0 + i, k] * lx[k]
            ly[row0 + i] = acc


def _enumerate(fn):
    target = build_samsung_target()
    compiled = allo.compile(fn, target, samsung_cost, **REASSOCIATE)
    buckets = _bucket_for_autoschedule(compiled.trace)
    token = set_active_liveness(None)
    try:
        candidates = _samsung_enumerate(target, buckets[0][1])
    finally:
        reset_active_liveness(token)
    return target, buckets, candidates


def _layouts(compiled):
    """The compiled placements as a list (one work item stores the placement)."""
    stored = compiled.compiled.layout
    return list(stored) if isinstance(stored, (list, tuple)) else [stored]


def _split_rows(candidates):
    return next(
        c
        for c in candidates
        if c.mode.split("+")[0] == "split_rows"
        and set(c.extra["grf_residency"].values()) == {"host"}
        and c.extra.get("crf_issue") == "shared"
        and not c.extra.get("stage_resident")
    )


def _edit(effects, what):
    matrix = next(l for l in effects.value_layouts if l.value == "left")
    if what == "image":
        layouts = tuple(
            replace(l, image="out") if l is matrix else l for l in effects.value_layouts
        )
        return replace(effects, value_layouts=layouts)
    if what == "buffer_identity":
        layouts = tuple(
            replace(l, value="right") if l is matrix else l
            for l in effects.value_layouts
        )
        return replace(effects, value_layouts=layouts)
    if what == "launch_domain":
        return replace(
            effects,
            launches=tuple(replace(l, instances=(0, 1)) for l in effects.launches),
        )
    if what == "allocation":
        return replace(effects, allocations=(("unrealized", 1 << 20, "banks"),))
    if what == "axes":
        layouts = tuple(
            replace(l, axes=()) if l is matrix else l for l in effects.value_layouts
        )
        return replace(effects, value_layouts=layouts)
    if what.startswith("duplicate_layout_"):
        # A second layout of the same value and image, contradicting the
        # first, placed before or after it: never a quieter alternative.
        twin = replace(matrix, packing="unrealized_packing")
        layouts = (
            (twin,) + effects.value_layouts
            if what.endswith("first")
            else effects.value_layouts + (twin,)
        )
        return replace(effects, value_layouts=layouts)
    if what.startswith("duplicate_transfer_"):
        first = next(t for t in effects.transfers if t.value == "left")
        twin = replace(first, route=("host_copy",))
        transfers = (
            (twin,) + effects.transfers
            if what.endswith("first")
            else effects.transfers + (twin,)
        )
        return replace(effects, transfers=transfers)
    raise AssertionError(what)


@pytest.mark.parametrize(
    "what",
    [
        "image",
        "buffer_identity",
        "launch_domain",
        "allocation",
        "axes",
        "duplicate_layout_first",
        "duplicate_layout_last",
        "duplicate_transfer_first",
        "duplicate_transfer_last",
    ],
)
def test_an_edited_declaration_is_refused_at_both_boundaries(what):
    target, buckets, candidates = _enumerate(contraction)
    chosen = _split_rows(candidates)
    original = chosen.extra["effects"]
    chosen.extra["effects"] = _edit(original, what)
    try:
        problems = validate_candidate_effects(buckets[0][1], chosen, target)
        assert problems, what
        if what.startswith("duplicate_"):
            kind = "layouts" if "layout" in what else "transfers"
            assert any(f"two {kind}" in problem for problem in problems), problems
        with pytest.raises(
            ValueError, match="violates the candidate contract"
        ) as error:
            allo.compile(
                contraction,
                target,
                samsung_cost,
                layout=[chosen] * len(buckets),
                **REASSOCIATE,
            )
    finally:
        chosen.extra["effects"] = original
    # The refusal names the problem the validator found, not a generic
    # substitution; the unedited candidate compiles.
    assert any(
        problem.split(";")[0][:40] in str(error.value) for problem in problems
    ), (problems, str(error.value))
    compiled = allo.compile(
        contraction, target, samsung_cost, layout=[chosen] * len(buckets), **REASSOCIATE
    )
    assert _layouts(compiled)[0].extra["effects"] == original


def test_a_bare_decision_and_a_foreign_plan_supply_their_decision():
    """Reuse without a declaration: the placement stripped of its effects is
    rebound to the current candidate and compiles; a declaration scoped to a
    function outside the program (another program's plan) supplies its
    decision and its declaration is re-derived, recorded in the placement."""
    target, buckets, candidates = _enumerate(contraction)
    chosen = _split_rows(candidates)
    bare = Placement(
        placements=dict(chosen.placements),
        mode=chosen.mode,
        extra={k: v for k, v in chosen.extra.items() if k != "effects"},
        layout=chosen.layout,
    )
    compiled = allo.compile(
        contraction, target, samsung_cost, layout=[bare] * len(buckets), **REASSOCIATE
    )
    assert _layouts(compiled)[0].extra["effects"] == chosen.extra["effects"]
    foreign_effects = replace(
        chosen.extra["effects"],
        coverage=replace(chosen.extra["effects"].coverage, function="another_program"),
        launches=tuple(
            replace(l, instances=(0, 7)) for l in chosen.extra["effects"].launches
        ),
    )
    foreign = Placement(
        placements=dict(chosen.placements),
        mode=chosen.mode,
        extra={**chosen.extra, "effects": foreign_effects},
        layout=chosen.layout,
    )
    compiled = allo.compile(
        contraction,
        target,
        samsung_cost,
        layout=[foreign] * len(buckets),
        **REASSOCIATE,
    )
    rebound = _layouts(compiled)[0]
    assert rebound.extra["effects"] == chosen.extra["effects"]
    assert "another_program" in rebound.extra["candidate_selection"]


def test_one_bodys_declaration_serves_every_body_of_a_lockstep_kernel():
    """The first body's candidate supplied for all 16 work items: each is
    normalized onto its own body (the function is the body's, the values
    bind by role) and compiles with that body's declaration."""
    target, buckets, candidates = _enumerate(lockstep_gemv)
    assert len(buckets) == 16
    chosen = _split_rows(candidates)
    assert chosen.extra["effects"].coverage.function == buckets[0][1][0].func_name
    compiled = allo.compile(
        lockstep_gemv,
        target,
        samsung_cost,
        layout=[chosen] * len(buckets),
        **REASSOCIATE,
    )
    functions = [p.extra["effects"].coverage.function for p in _layouts(compiled)]
    assert functions == [matches[0].func_name for _scope, matches in buckets]
    assert len(set(functions)) == 16
    # A declaration edited for one body is refused for every body it is
    # supplied to, including the body it was declared for.
    original = chosen.extra["effects"]
    chosen.extra["effects"] = replace(original, allocations=())
    try:
        with pytest.raises(ValueError, match="declares allocations"):
            allo.compile(
                lockstep_gemv,
                target,
                samsung_cost,
                layout=[chosen] * len(buckets),
                **REASSOCIATE,
            )
    finally:
        chosen.extra["effects"] = original
