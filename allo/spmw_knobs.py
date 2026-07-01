# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SPMW typed schedule-knob registry.

Knobs enumerate and materialize schedule choices. Their performance is not
defined here: the executable CostSpec observes the resulting Placement while
scoring each candidate.

Ownership split (design 07 §A3 / SPEC-022 D4):
  - `candidates(ctx)` -- the per-lever fan-out, OWNED here (lifted out of the
    hand-crossed enumerator code).
  - `emit(value, ctx)` -- the per-lever materializer, OWNED here. It writes the
    chosen value onto the candidate's `extra[knob.name]` (the compatibility
    carrier the cost task's `active_knobs`/`knob_phases` read side consumes,
    and the codegen `extra`-read branches consume). The typed knob owns
    PRODUCTION of the value set and its placement into `extra`; the existing
    `extra`-read consumers are unchanged.
The autoscheduler cross-products the registered knobs generically
(`cross_with_knobs`): adding a lever is a single `register_knob(...)` call, not
lockstep edits to enumerator + cost-compose + codegen (the lockstep that
produced the SPEC-022 drift).
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from typing import Any, Callable


# Whole-trace liveness for the current `autoschedule` run (SPEC-023 D1).
# `autoschedule` sets this once (before the per-group loop) so the per-group
# enumerators' `cross_with_knobs` -- which builds `KnobCtx` -- can thread the
# liveness into each ctx WITHOUT changing the landed `enumerator(target,
# matches)` signature. Default `None` (unset) == today's per-kernel behaviour,
# so any caller outside an `autoschedule` run is byte-identical.
_active_liveness: "contextvars.ContextVar[Any]" = contextvars.ContextVar(
    "spmw_active_liveness", default=None
)


# The SPEC-023 schedule-search knobs (D1 cross-op residency, D2 tile/fold, D3
# double-buffer depth). `SPMW_DISABLE_SCHEDULE_SEARCH=1` skips exactly these in
# `cross_with_knobs`, leaving the placement-realization levers -> the
# group-local baseline schedule (the verifier's baseline-vs-search comparison).
_SCHEDULE_SEARCH_KNOBS = frozenset({"residency", "tile", "double_buffer"})


def set_active_liveness(liveness):
    """Set the whole-trace liveness for the current run; returns the
    contextvars Token so the caller can reset it. Called by `autoschedule`."""
    return _active_liveness.set(liveness)


def reset_active_liveness(token) -> None:
    """Restore the prior liveness (paired with `set_active_liveness`)."""
    _active_liveness.reset(token)


# --------------------------------------------------------------------- #
# Knob ctx -- the bundle a knob's candidates()/emit() read
# --------------------------------------------------------------------- #


@dataclass
class KnobCtx:
    """Context handed to `Knob.candidates`/`Knob.emit` at enumeration time.

    Carries the target tree, the matched ops (shape source), the role->memref
    map, and -- for `emit` -- the base `Placement` the knob value is applied
    to. This is the enumeration-side ctx; the cost side keeps its own
    `ComposeCtx` (frozen, unchanged).
    """

    target: Any
    matches: list = field(default_factory=list)
    role_to_memref: dict = field(default_factory=dict)
    base: Any = None  # the Placement being materialised (emit only)
    # Whole-trace liveness result (SPEC-023 D1), threaded in by `autoschedule`
    # so the cross-kernel/cross-work-id `residency` knob can read it. Additive,
    # default `None` == today's per-kernel-local behaviour (no residency DOF).
    # `KnobCtx` is the placement task's type; this is the additive `liveness`
    # field the schedule-search task escalated to add (SPEC-023 Answer 4 #1).
    liveness: Any = None


# --------------------------------------------------------------------- #
# The concrete typed Knob
# --------------------------------------------------------------------- #


@dataclass
class Knob:
    """A concrete typed schedule knob.

    `candidates_fn(ctx) -> list[value]` is the lever's fan-out; `emit_fn(value,
    base, ctx) -> Placement` materialises the value onto a copy of `base`
    (writing `extra[name] = value` on the resulting placement).
    """

    name: str
    candidates_fn: Callable[[KnobCtx], list]
    emit_fn: Callable[[Any, Any, KnobCtx], Any]

    def candidates(self, ctx: KnobCtx) -> list:
        return list(self.candidates_fn(ctx))

    def emit(self, value, ctx: KnobCtx):
        return self.emit_fn(value, ctx.base, ctx)


# --------------------------------------------------------------------- #
# Registry (mirrors register_knob_cost; keyed (target_name, knob_name))
# --------------------------------------------------------------------- #


# Ordered per target so the generic cross-product applies knobs in a stable,
# byte-identical order. `dict` preserves insertion order, which IS the cross
# order (matches the hand-crossed enumerator's lever sequence).
_knob_registry: dict[str, "dict[str, Knob]"] = {}


def register_knob(target_name: str, knob: Knob) -> Knob:
    """Register a typed knob for `target_name` (SPEC-022 D4 write-side).

    Idempotent re-registration of the same knob object is allowed; a
    different knob under an existing (target, name) is an error. Registration
    ORDER is the cross-product order, so register levers in the sequence the
    hand-crossed enumerator applied them (the byte-identity anchor)."""
    by_name = _knob_registry.setdefault(target_name, {})
    existing = by_name.get(knob.name)
    if existing is not None and existing is not knob:
        raise ValueError(f"duplicate knob {knob.name!r} for target {target_name!r}")
    by_name[knob.name] = knob
    return knob


def registered_knobs(target_name: str) -> "list[Knob]":
    """The knobs registered for `target_name`, in registration (= cross) order."""
    return list(_knob_registry.get(target_name, {}).values())


# --------------------------------------------------------------------- #
# The six live levers, lifted onto typed knobs (SPEC-022 D4 migration).
# Each knob REUSES the existing enumerator helper for its emit body, so the
# materialised `(placements, mode, extra)` tuples are byte-identical to the
# hand-crossed enumerator; the knob is the typed owner + the registry the
# single-registration extension point.
# --------------------------------------------------------------------- #


def _grf_residency_candidates(ctx: KnobCtx) -> list:
    # crf always; host ONLY when a role is host-eligible (SPEC-024 §3) -- the
    # exact conditional the hand-crossed residency loop applied.
    from .spmw_autoschedule import _samsung_host_eligible_memrefs

    host = _samsung_host_eligible_memrefs(ctx.target, ctx.base, ctx.role_to_memref)
    return ["crf", "host"] if host else ["crf"]


def _grf_residency_emit(value, base, ctx: KnobCtx):
    from .spmw_autoschedule import _samsung_host_eligible_memrefs, _with_residency

    host = _samsung_host_eligible_memrefs(ctx.target, base, ctx.role_to_memref)
    out = base
    for mref in host:
        out = _with_residency(out, mref, value)
    return out


def _crf_issue_candidates(ctx: KnobCtx) -> list:
    return ["shared", "per_workid"]


def _crf_issue_emit(value, base, ctx: KnobCtx):
    # _with_crf_modes builds both variants; pick the one matching `value` so
    # the mode-token suffix + extra["crf_issue"] are byte-identical.
    from .spmw_autoschedule import _with_crf_modes

    for variant in _with_crf_modes(base):
        if variant.extra.get("crf_issue") == value:
            return variant
    raise AssertionError(f"crf_issue: no variant for {value!r}")


def _stage_resident_candidates(ctx: KnobCtx) -> list:
    return [False, True]


def _stage_resident_emit(value, base, ctx: KnobCtx):
    from .spmw_autoschedule import _with_stage_resident

    return _with_stage_resident(base, value)


def _n_tasklets_candidates(ctx: KnobCtx) -> list:
    from .spmw_autoschedule import _upmem_tasklet_candidates

    return _upmem_tasklet_candidates(ctx.target, ctx.matches)


def _n_tasklets_emit(value, base, ctx: KnobCtx):
    from .spmw_autoschedule import Placement

    new_extra = dict(base.extra)
    new_extra["n_tasklets"] = value
    return Placement(
        placements=dict(base.placements),
        mode=base.mode,
        extra=new_extra,
        layout=getattr(base, "layout", None),
    )


def _vr_dma_candidates(ctx: KnobCtx) -> list:
    return ["intra", "inter"]


def _vr_dma_emit(value, base, ctx: KnobCtx):
    from .spmw_autoschedule import Placement

    new_extra = dict(base.extra)
    new_extra["vr_dma"] = value
    return Placement(
        placements=dict(base.placements),
        mode=base.mode,
        extra=new_extra,
        layout=getattr(base, "layout", None),
    )


def _residency_crossing_memrefs(ctx: KnobCtx) -> list:
    """The candidate's memrefs whose whole-trace liveness crosses a kernel or
    work-id boundary (SPEC-023 D1). Empty when liveness is absent (per-kernel
    mode) or nothing crosses -- the byte-identical no-residency case."""
    from .spmw_liveness import memref_span, crosses_boundary

    liveness = getattr(ctx, "liveness", None)
    if not liveness:
        return []
    base = ctx.base
    out = []
    for mref in getattr(base, "placements", {}):
        span = memref_span(liveness, mref)
        if span is not None and crosses_boundary(span):
            out.append((mref, span))
    return out


def _residency_candidates(ctx: KnobCtx) -> list:
    # >=2-candidate discipline ONLY when liveness says residency is possible:
    # `["restage", "resident"]` when some memref crosses a boundary, else the
    # single-candidate `["restage"]` (today's behaviour, 1x fan, byte-identical).
    return ["restage", "resident"] if _residency_crossing_memrefs(ctx) else ["restage"]


def _residency_emit(value, base, ctx: KnobCtx):
    from .spmw_autoschedule import Placement

    new_extra = dict(base.extra)
    new_extra["residency"] = value
    crossing = _residency_crossing_memrefs(ctx)
    if crossing:
        # Record the crossing memrefs (with their LiveSpan endpoints) on BOTH
        # arms (SPEC-023 T6/D2) so the `residency` knob_cost knows the
        # inter-kernel staging magnitude: `restage` PAYS that staging, `resident`
        # ELIDES it. A non-crossing value (no entries) records nothing -> its
        # restage knob_cost is empty -> byte-identical to today.
        info = {
            mref: {
                "producer_func": span.producer_func,
                "consumer_func": span.consumer_func,
                "crosses_kernel": span.crosses_kernel,
                "crosses_workid": span.crosses_workid,
                "handle": base.placements.get(mref),
            }
            for mref, span in crossing
        }
        new_extra["residency_crossing"] = info
        if value == "resident":
            # `residency_pairs` is the resident-only carrier codegen's
            # move-elision branch + the post-argmin reconciliation read; the
            # reconciliation pops a mref from here when an endpoint disagrees.
            new_extra["residency_pairs"] = dict(info)
    return Placement(
        placements=dict(base.placements),
        mode=base.mode,
        extra=new_extra,
        layout=getattr(base, "layout", None),
    )


def _tile_candidates(ctx: KnobCtx) -> list:
    # Legal capacity-bounded re-tilings of the matched nest (SPEC-023 D2), from
    # the tile generator. The identity tiling is always first (byte-identical);
    # a non-identity retile appears ONLY when a legal capacity-fitting one
    # exists. >=2-candidate discipline gated on capacity, never a tile literal.
    from .spmw_tiling import tile_candidates

    plans = tile_candidates(ctx.target, ctx.matches)
    return plans if plans else None  # None -> knob is inert (no tileable nest)


def _tile_emit(value, base, ctx: KnobCtx):
    from .spmw_autoschedule import Placement

    new_extra = dict(base.extra)
    # Store the chosen TilePlan (the materialiser reads tile_size off it; the
    # identity tiling is the default and is byte-identical). Only record a
    # NON-identity tile so the identity candidate's extra is unchanged.
    if value is not None and not getattr(value, "is_identity", True):
        new_extra["tile"] = value
    return Placement(
        placements=dict(base.placements),
        mode=base.mode,
        extra=new_extra,
        layout=getattr(base, "layout", None),
    )


def _double_buffer_candidates(ctx: KnobCtx) -> list:
    # Double buffering needs a structural buffer pair before it is a legal
    # candidate. No current target declares that pair, so this knob is inert.
    # Once that structure lands, CostSpec can price both placements directly.
    del ctx
    return [1]


def _double_buffer_emit(value, base, ctx: KnobCtx):
    from .spmw_autoschedule import Placement

    new_extra = dict(base.extra)
    # Record the double-buffer flag ONLY for depth 2 so depth-1's extra is
    # byte-identical to today. The overlap compose reads `extra["double_buffer"]`
    # to pick the loop-encoded (hiding) vs collapsed (serial) DMA phase.
    if int(value) >= 2:
        new_extra["double_buffer"] = True
    return Placement(
        placements=dict(base.placements),
        mode=base.mode,
        extra=new_extra,
        layout=getattr(base, "layout", None),
    )


def register_default_knobs():
    """Register the six live levers as typed knobs, in the per-target cross
    ORDER the hand-crossed enumerator applied them (the byte-identity anchor).

    Samsung/Mortise: grf_residency -> crf_issue -> stage_resident (the
    `fibers` lever is baked into the base candidates, not a cross). UPMEM:
    n_tasklets. APU v1's vr_dma is crossed WITH `mode` inside its enumerator
    (a paired 4-candidate fan), so it is registered as one knob whose
    candidates carry the (mode, vr_dma) pair -- kept in the enumerator to
    preserve the exact 4-tuple set. Idempotent (safe to call at import).
    """
    for tname in ("samsung_hbm_pim", "mortise", "mortise_wide"):
        register_knob(
            tname, Knob("grf_residency", _grf_residency_candidates, _grf_residency_emit)
        )
        register_knob(tname, Knob("crf_issue", _crf_issue_candidates, _crf_issue_emit))
        register_knob(
            tname,
            Knob("stage_resident", _stage_resident_candidates, _stage_resident_emit),
        )
    register_knob("upmem", Knob("n_tasklets", _n_tasklets_candidates, _n_tasklets_emit))
    register_knob("apu_v1", Knob("vr_dma", _vr_dma_candidates, _vr_dma_emit))

    # Cross-kernel / cross-work-id residency (SPEC-023 D1/T6). A 1x fan
    # (byte-identical) for any value that does NOT cross a boundary; the
    # candidate set only grows when whole-trace liveness flags residency as
    # possible. Its `cost` delegates to the registered `knob_cost(target,
    # "residency", ...)` (task 004); the resident arm is credited the elided
    # inter-kernel staging so the argmin earns it.
    for tname in (
        "samsung_hbm_pim",
        "mortise",
        "mortise_wide",
        "upmem",
        "apu_v1",
        "apu_v2",
    ):
        register_knob(tname, Knob("residency", _residency_candidates, _residency_emit))

    # Capacity-bounded tile/fold (SPEC-023 D2). Registered LAST on every
    # backend so it is a 1x fan (byte-identical) whenever the generator finds
    # no legal capacity-fitting retile -- the candidate set only grows when a
    # legal one exists. Its `cost` delegates to `knob_cost(target, "tile",
    # ...)`; the identity tiling is the default and writes no `extra`.
    for tname in (
        "samsung_hbm_pim",
        "mortise",
        "mortise_wide",
        "upmem",
        "apu_v1",
        "apu_v2",
    ):
        register_knob(tname, Knob("tile", _tile_candidates, _tile_emit))

    # Double-buffer depth (SPEC-023 D3). A 1x fan ([1], byte-identical) on
    # every backend with no overlap-fold DMA-hide DOF; [1, 2] where one exists
    # (APU v1 today). The depth-2 arm sets `extra["double_buffer"]`, which the
    # overlap compose reads to emit the loop-encoded (hiding) DMA phase; under
    # the default faithful flavor depth-1 and depth-2 tie (collapsed phase
    # both), so the argmin default is byte-identical.
    for tname in (
        "samsung_hbm_pim",
        "mortise",
        "mortise_wide",
        "upmem",
        "apu_v1",
        "apu_v2",
    ):
        register_knob(
            tname, Knob("double_buffer", _double_buffer_candidates, _double_buffer_emit)
        )


def cross_with_knobs(
    target, base_candidates: list, matches: list, role_to_memref: dict
) -> list:
    """Cross `base_candidates` with every registered knob for `target`,
    generically (SPEC-022 D4).

    For each knob in registration order, replace each in-flight candidate with
    one materialised candidate per `knob.candidates(ctx)` value. A knob whose
    `candidates` returns a single value contributes a 1x fan (no set growth,
    just the materialised `extra` tag). Byte-identical to the prior
    hand-crossed loop when knobs are registered in the same order with the
    same per-knob candidate sets.
    """
    target_name = getattr(target, "name", None)
    # Thread the run's whole-trace liveness (SPEC-023 D1) into every KnobCtx so
    # the residency knob can read it. `None` outside an autoschedule run ->
    # byte-identical to today.
    liveness = _active_liveness.get()
    # Baseline-vs-search toggle (SPEC-023 wiring, task 007): with
    # `SPMW_DISABLE_SCHEDULE_SEARCH=1` the three SPEC-023 schedule-search knobs
    # (residency / tile / double_buffer) are SKIPPED, leaving only the
    # placement-realization levers -- i.e. the group-local baseline schedule.
    # Default off (unset) -> the full search, byte-identical to today. The
    # verifier (008/009) compiles the SAME workload twice (this flag set vs
    # unset) and compares real-sim cycles; this is the single choke point.
    import os

    _search_off = os.environ.get("SPMW_DISABLE_SCHEDULE_SEARCH") == "1"
    current = list(base_candidates)
    for knob in registered_knobs(target_name):
        if _search_off and knob.name in _SCHEDULE_SEARCH_KNOBS:
            continue
        nxt: list = []
        for base in current:
            ctx = KnobCtx(
                target=target,
                matches=matches,
                role_to_memref=role_to_memref,
                base=base,
                liveness=liveness,
            )
            values = knob.candidates(ctx)
            if not values:
                # A knob that produces no value for this candidate leaves it
                # unchanged (e.g. residency with no host-eligible role).
                nxt.append(base)
                continue
            for v in values:
                nxt.append(knob.emit(v, ctx))
        current = nxt
    return current


# Register the six live levers at import (idempotent). Importing this module
# -- which `_samsung_enumerate` / `_upmem_enumerate` do via `cross_with_knobs`
# -- installs the typed knob registry.
register_default_knobs()
