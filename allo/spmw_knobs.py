# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SPMW typed knob registry (SPEC-022 D4).

The write-side of the co-owned knob seam. The `Knob` Protocol is FROZEN in
`spmw_cost_model.py` (`name` / `candidates(ctx)` / `cost(value, ctx)` /
`emit(value, ctx)`); this module supplies the concrete `Knob` implementations
and the `register_knob` registry, binding against the frozen
`register_knob_cost` / `knob_cost` seam there. It does NOT redefine the
Protocol or the cost side.

Ownership split (design 07 §A3 / SPEC-022 D4):
  - `candidates(ctx)` -- the per-lever fan-out, OWNED here (lifted out of the
    hand-crossed enumerator code).
  - `emit(value, ctx)` -- the per-lever materializer, OWNED here. It writes the
    chosen value onto the candidate's `extra[knob.name]` (the compatibility
    carrier the cost task's `active_knobs`/`knob_phases` read side consumes,
    and the codegen `extra`-read branches consume). The typed knob owns
    PRODUCTION of the value set and its placement into `extra`; the existing
    `extra`-read consumers are unchanged.
  - `cost(value, ctx)` -- delegates to `knob_cost(target, name, value, ctx)`
    (the cost task's registered seam). Exactly one cost definition exists.

The autoscheduler cross-products the registered knobs generically
(`cross_with_knobs`): adding a lever is a single `register_knob(...)` call, not
lockstep edits to enumerator + cost-compose + codegen (the lockstep that
produced the SPEC-022 drift).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


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


# --------------------------------------------------------------------- #
# The concrete typed Knob
# --------------------------------------------------------------------- #


@dataclass
class Knob:
    """A concrete typed schedule knob (binds the frozen `Knob` Protocol).

    `candidates_fn(ctx) -> list[value]` is the lever's fan-out; `emit_fn(value,
    base, ctx) -> Placement` materialises the value onto a copy of `base`
    (writing `extra[name] = value` as the compatibility carrier). `cost(...)`
    delegates to the cost task's registered `knob_cost` seam -- the knob never
    re-implements cost.
    """

    name: str
    candidates_fn: Callable[[KnobCtx], list]
    emit_fn: Callable[[Any, Any, KnobCtx], Any]

    def candidates(self, ctx: KnobCtx) -> list:
        return list(self.candidates_fn(ctx))

    def emit(self, value, ctx: KnobCtx):
        return self.emit_fn(value, ctx.base, ctx)

    def cost(self, value, ctx):
        # Delegate to the cost task's frozen seam: ONE cost definition.
        from .spmw_cost_model import knob_cost

        target_name = getattr(getattr(ctx, "target", None), "name", None)
        return knob_cost(target_name, self.name, value, ctx)


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
        raise ValueError(
            f"duplicate knob {knob.name!r} for target {target_name!r}"
        )
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
        placements=dict(base.placements), mode=base.mode, extra=new_extra,
        layout=getattr(base, "layout", None),
    )


def _vr_dma_candidates(ctx: KnobCtx) -> list:
    return ["intra", "inter"]


def _vr_dma_emit(value, base, ctx: KnobCtx):
    from .spmw_autoschedule import Placement

    new_extra = dict(base.extra)
    new_extra["vr_dma"] = value
    return Placement(
        placements=dict(base.placements), mode=base.mode, extra=new_extra,
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
        register_knob(tname, Knob("grf_residency",
                                  _grf_residency_candidates, _grf_residency_emit))
        register_knob(tname, Knob("crf_issue",
                                  _crf_issue_candidates, _crf_issue_emit))
        register_knob(tname, Knob("stage_resident",
                                  _stage_resident_candidates, _stage_resident_emit))
    register_knob("upmem", Knob("n_tasklets",
                                _n_tasklets_candidates, _n_tasklets_emit))
    register_knob("apu_v1", Knob("vr_dma", _vr_dma_candidates, _vr_dma_emit))


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
    current = list(base_candidates)
    for knob in registered_knobs(target_name):
        nxt: list = []
        for base in current:
            ctx = KnobCtx(
                target=target, matches=matches,
                role_to_memref=role_to_memref, base=base,
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
