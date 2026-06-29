# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decoupled `CostModel` mechanism (design 04 §1).

This module owns the *mechanism* of the two-layer cost spec; the
*numbers* live in `spmw_cost_tables.py`. The split is the same discipline
as target-structure vs target-cost, one level down: a researcher refining
a per-op cost against board profiling edits only the tables, never this.

Two layers, both swappable, both independently per-op refinable:

  * Layer A -- `OpCost` / `MoveCost`: per *named* op/move, a CALLABLE that
    returns a cycle count given the op's local context (`OpCostCtx` /
    `MoveCostCtx`). The trivial entry is "constant N cycles"
    (`OpCost(lambda ctx: 4)`); the APU-v1-MICRO-2025 entry is an
    analytical function of operand width and lane geometry. Both satisfy
    the same interface, so a refinement is a one-entry edit.

  * Layer B -- `compose`: folds the per-op/per-move costs over a match
    trace + a placement into a whole-program `CostResult`. This is the
    per-target phase algebra (Samsung preload->exec->readback, UPMEM
    revolver, AiM sum, APU v1 outer-loop + DMA).

A `CostModel` is bound to a structural target by `(target_name, flavor,
concern)`. The autoschedule-facing surface (`@cost("kernel_cycles")` in
`spmw_cost.py`) is preserved: its factory looks up the bound model and
returns a `(trace, layout) -> int` closure calling `result.cycles`, so
the argmin path is byte-identical to the pre-decoupling numbers.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable


# --------------------------------------------------------------------- #
# A1 -- the phase timeline (design 07 §A1)
# --------------------------------------------------------------------- #
#
# `Phase` is a modulo-scheduling reservation-table-shaped object (Rau
# HPL-94-115; Lam PLDI'88). `ii` is the initiation interval. The combiner's
# sum-within / max-across fold is ALCOP's load-compute max-over-stages +
# roofline + uiCA resource-overlap (report 27 §1.1). This carrier replaces
# the free-form `phases` dict; the faithful fold is byte-identical (§A1.6).


class Resource(enum.Enum):
    COMPUTE = "compute"    # the device exec lane (MAC/MUL/ADD/AF on PIM)
    DMA = "dma"            # on-device data movement (L4<->L1, MRAM<->WRAM, bank<->GRF)
    HOST = "host"          # host<->device staging (the host_staging concern)
    LOCALITY = "locality"  # D2 row-buffer/bank-conflict penalty (a COMPUTE-side
    #                        stall, tagged separately so it is greppable + zeroable)


@dataclass(frozen=True)
class Phase:
    """One reservation on one resource (design 07 §A1.1).

    Steady-state cycles:
      cyc(phase) = latency + ii*(count-1)   when count >= 1
      cyc(phase) = 0                        when count == 0

    `tag` is the *display* label that preserves the legacy phase-name
    breakdown ("exec", "stage_resident", "readback", "dynamic_assumed",
    "evict_per_call") without re-introducing a free-form dict as the
    structural carrier. The fold reads resource/latency/ii/count; the
    breakdown surfacing reads `tag`.
    """

    resource: Resource
    latency: int          # pipeline fill/drain (depth); cycles before steady state
    ii: int               # initiation interval (steady-state per-iteration throughput)
    count: int            # number of iterations
    tag: str = ""         # free-form provenance label for breakdown


def phase_cycles(phase: Phase) -> int:
    """Steady-state cycles for one `Phase` (design 07 §A1.1).

    Module-level pure function (NOT a method) so a knob's `cost(...)` and the
    combiner share one definition.
    """
    if phase.count <= 0:
        return 0
    return phase.latency + phase.ii * (phase.count - 1)


def phases_as_dict(phases: "list[Phase]") -> dict:
    """Back-compat dict view of a `Phase` list (design 07 §A1.2).

    Groups by `tag`, summing `phase_cycles`. This is the ONLY place the old
    dict shape survives -- it is read-only and derived, feeding the
    `RunResult.extra["phases"]` surfacing path + the report harness.
    """
    out: dict = {}
    for ph in phases:
        out[ph.tag] = out.get(ph.tag, 0) + phase_cycles(ph)
    return out


def combine(phases: "list[Phase]", *, overlap: bool) -> int:
    """Resource-aware whole-program fold (design 07 §A1.3).

    Sum within each resource class (serial issue on one resource). Then:
      * `overlap=False` (faithful): SUM across resources too -- everything
        serial. Byte-identical to the old per-phase + whole-program sum by
        construction (§A1.6).
      * `overlap=True`: MAX across independent resources + fill/drain of the
        subordinate (hidden) resources. The overlap fold is a *new flavor*,
        never the default; its semantics are D1's (task 005), not A1's.
    """
    per_resource: dict[Resource, int] = {}
    for ph in phases:
        per_resource[ph.resource] = per_resource.get(ph.resource, 0) + phase_cycles(ph)
    if not overlap:
        return sum(per_resource.values())
    return _max_across_with_fill_drain(per_resource, phases)


def _max_across_with_fill_drain(per_resource: dict, phases: "list[Phase]") -> int:
    """Overlap fold (design 07 §D1.2): max-across the two pipelining resources
    (COMPUTE dominant, DMA subordinate) + the subordinate's fill, plus the
    genuinely-serial resources (HOST staging, LOCALITY stall) added on top.

        total = max(cyc_COMPUTE, cyc_DMA) + dma_fill   (DMA hides behind compute)
                + cyc_HOST                               (host stages serially)
                + cyc_LOCALITY                           (row-buffer stall, serial)

    The DMA "fill" is the first iteration's pipeline-fill latency that cannot
    hide behind compute (`fill = Σ latency over DMA phases`). For a
    loop-encoded DMA phase `Phase(DMA, latency=per_move, ii=per_move,
    count=n)` the fill is one `per_move` and the steady-state
    `per_move*(n-1)` is absorbed into compute (it hides) -- a double-buffered
    schedule. For a collapsed DMA phase `Phase(DMA, latency=total, ii=0,
    count=1)` the fill IS the whole DMA (`latency=total`), so a single-shot /
    serialized DMA does not hide -- conservatively `max(cyc_C, total) + total
    >= cyc_C + total`, i.e. no overlap win. This is the D1.4 flip lever: the
    backend's overlap compose picks the encoding by the double-buffer knob.

    Honest boundary (D1.3, T21): this over-counts partial overlap (a DMA that
    hides only half is priced fully hidden) and ignores bank-arbitration
    contention -- a structural max-over-resources fold, NOT a pipeline
    simulator. The verifier's bar is rank-correctness of the flip, not cycle
    accuracy.
    """
    cyc_compute = per_resource.get(Resource.COMPUTE, 0)
    cyc_dma = per_resource.get(Resource.DMA, 0)
    cyc_host = per_resource.get(Resource.HOST, 0)
    cyc_locality = per_resource.get(Resource.LOCALITY, 0)
    dma_fill = sum(ph.latency for ph in phases if ph.resource is Resource.DMA)
    return max(cyc_compute, cyc_dma) + dma_fill + cyc_host + cyc_locality


# --------------------------------------------------------------------- #
# A2 -- placed-trace access descriptor (design 07 §A2.3)
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class AccessDescr:
    """Layout-derived access pattern for one operand at one access site.

    CO-OWNED with `autosched-placement-realization`: consumed by (a) the D2
    locality term here and (b) the allocator's non-zero base access cost
    there. ONE notion of access pattern, two consumers. Population of the
    layout-derived fields is D2's job (task 006); A2/task 004 defines the
    type + the conflict-free `identity()` default.
    """

    tier: str                      # "register" | "near_bank" | "scratchpad" | "dram"
    stride: int = 1                # element stride between consecutive lane accesses
    bank_dims: tuple = ()          # out-dim names that index banks (from the layout)
    n_banks: int = 1               # bank fan-out the access touches
    conflict_free: bool = True     # describes_conflict_free over (bank_dims, varying)
    conflict_count: int = 0        # # of varying-input vectors that collide on a bank
    row_hits: int | None = None    # resolved open-row reuse count, if derivable (else None)

    def __post_init__(self):
        # Invariant (design 07 §A2.3): conflict_free=True <=> conflict_count=0.
        if self.conflict_free and self.conflict_count != 0:
            raise ValueError(
                "AccessDescr: conflict_free=True requires conflict_count==0 "
                f"(got {self.conflict_count})"
            )

    @classmethod
    def identity(cls, tier: str = "register") -> "AccessDescr":
        """The back-compat conflict-free default: stride 1, conflict-free,
        count 0 -- yields a zero D2 locality penalty by construction."""
        return cls(tier=tier)


# --------------------------------------------------------------------- #
# A4 -- provenance + calibration record (design 07 §A4)
# --------------------------------------------------------------------- #


class Provenance(enum.Enum):
    MEASURED = "measured"      # anchored to a sim/HW run (a calibration record names it)
    DATASHEET = "datasheet"    # from a published spec/paper (tCCDL, MICRO'25 Table 5)
    ASSUMPTION = "assumption"  # an engineering guess, not yet validated


@dataclass(frozen=True)
class CalibrationRecord:
    """Per-(target, flavor) calibration data (design 07 §A4.2). Read by the
    confidence-gate + the user/optimizer; NOT consulted by the argmin core."""

    validated_against: str = ""           # which sim/HW run
    residual_error: float | None = None   # |est - measured| / measured, if known
    shape_coverage: tuple = ()            # the shapes this (target,flavor) was checked at


# The symbolic confidence band, ordered worst -> best. The band is derived
# from WHICH provenance tags participated (report 28 §A4.3 -- a function of
# the per-constant provenance, NOT a learned posterior variance): the
# least-trusted participating tag dominates.
_PROVENANCE_TO_BAND = {
    Provenance.ASSUMPTION: "assumption",
    Provenance.DATASHEET: "datasheet",
    Provenance.MEASURED: "calibrated",
}
# worst-first precedence for the dominating (least-trusted) tag.
_BAND_PRECEDENCE = (
    Provenance.ASSUMPTION, Provenance.DATASHEET, Provenance.MEASURED,
)


def provenance_band(model: "CostModel") -> str:
    """The symbolic confidence band for `model`, derived from its constants'
    provenance tags (design 07 §A4.3 / report 28 §A4.3). The LEAST-trusted
    participating tag dominates: any `ASSUMPTION` -> "assumption"; else any
    `DATASHEET` -> "datasheet"; all `MEASURED` -> "calibrated"; an empty model
    (no tagged constants, e.g. the placeholder) -> "placeholder".

    This is the SYMBOLIC band the A4.3 gate consults -- not a learned variance.
    It does NOT feed the argmin scoring core (objective-only invariant)."""
    tags = [c.provenance for c in model.op_costs.values()]
    tags += [c.provenance for c in model.move_costs.values()]
    if not tags:
        return "placeholder"
    for p in _BAND_PRECEDENCE:           # worst-first: first hit dominates
        if p in tags:
            return _PROVENANCE_TO_BAND[p]
    return "calibrated"


# --------------------------------------------------------------------- #
# A3 -- the knob cost seam (design 07 §A3.1)
# --------------------------------------------------------------------- #


@runtime_checkable
class Knob(Protocol):
    """The CO-OWNED typed knob. `candidates()`/`emit()` are owned by
    `autosched-placement-realization` D4; ONLY `cost(...)` is owned here.

    A knob's cost contribution is exactly a set of `Phase`s on the A1
    timeline, so the knob seam and the timeline are the same currency."""

    name: str

    def candidates(self, ctx) -> list:            # OWNED by placement-realization D4
        ...

    def cost(self, value, ctx: "ComposeCtx") -> "list[Phase]":  # OWNED HERE
        """The chosen value's contribution to the A1 phase timeline."""
        ...

    def emit(self, value, ctx):                   # OWNED by placement-realization D4
        ...


# keyed by (target_name, knob_name) -> cost(value, ctx) -> list[Phase]
_knob_cost_registry: dict[tuple[str, str], Callable] = {}


def register_knob_cost(target_name: str, knob_name: str, fn: Callable) -> Callable:
    """Register a knob's `cost(value, ctx) -> list[Phase]` contribution
    (design 07 §A3.1). Idempotent re-registration of the same callable is
    allowed; a different callable under an existing key is an error."""
    key = (target_name, knob_name)
    existing = _knob_cost_registry.get(key)
    if existing is not None and existing is not fn:
        raise ValueError(f"duplicate knob cost for {key!r}")
    _knob_cost_registry[key] = fn
    return fn


def knob_cost(target_name: str, knob_name: str, value, ctx) -> "list[Phase]":
    """The chosen `value`'s phase contribution, or `[]` if no knob cost is
    registered for `(target_name, knob_name)` (design 07 §A3.2). A missing
    knob contributes nothing -- the faithful base timeline is unchanged."""
    fn = _knob_cost_registry.get((target_name, knob_name))
    if fn is None:
        return []
    return fn(value, ctx)


# The schedule knobs a candidate carries on `layout.extra` (design 07 §A3.2).
# Only those with a REGISTERED `cost(...)` contribute phases here; the rest
# stay as `layout.extra` reads inside compose until placement-task D4 migrates
# them. This list is the read side only -- the typed registry (candidates/emit)
# is the placement task's.
_KNOB_NAMES = (
    "stage_resident", "grf_residency", "crf_issue",
    "n_tasklets", "vr_dma", "n_fibers", "double_buffer",
)


def active_knobs(ctx) -> "list[tuple[str, object]]":
    """The (knob_name, chosen_value) pairs this candidate carries on
    `ctx.layout.extra` for which a knob cost is registered (design 07 §A3.2).
    A knob with no registered `cost(...)` is skipped (its effect stays folded
    inside the base compose until D4)."""
    target_name = getattr(getattr(ctx, "target", None), "name", None)
    extra = getattr(getattr(ctx, "layout", None), "extra", {}) or {}
    out: list[tuple[str, object]] = []
    for name in _KNOB_NAMES:
        if name in extra and (target_name, name) in _knob_cost_registry:
            out.append((name, extra[name]))
    return out


def knob_phases(ctx) -> "list[Phase]":
    """`Σ knob.cost(chosen)` over the candidate's active knobs (design 07
    §A3.2) -- the additive knob-cost term `compose` concatenates onto the base
    op/move phases. Empty when no registered knob is active."""
    target_name = getattr(getattr(ctx, "target", None), "name", None)
    phases: list[Phase] = []
    for name, value in active_knobs(ctx):
        phases += knob_cost(target_name, name, value, ctx)
    return phases


def knob_marginal_delta(
    base_phases: "list[Phase]", ctx, knob_name: str, v0, v1, *, overlap: bool
) -> int:
    """The EXACT marginal cost of moving knob `knob_name` from `v0` to `v1`
    (design 07 §A3.3): re-cost ONLY that knob and re-fold; do not re-walk the
    trace. Because `compose = combine(base + Σ knob.cost)` and `combine` is a
    fold over a Phase list, this delta equals a full `compose` recompute of
    the changed candidate, byte-for-byte (the IVM delta-not-recompute
    property; the non-tautology test in task 008 asserts the equality)."""
    target_name = getattr(getattr(ctx, "target", None), "name", None)
    base = list(base_phases)
    c0 = combine(base + knob_cost(target_name, knob_name, v0, ctx), overlap=overlap)
    c1 = combine(base + knob_cost(target_name, knob_name, v1, ctx), overlap=overlap)
    return c1 - c0


# --------------------------------------------------------------------- #
# Per-op / per-move cost context (the refinement seam, design 04 §1.3)
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class OpCostCtx:
    """Immutable local context handed to an `OpCost.fn`.

    A per-op analytical model has what it needs WITHOUT reaching into the
    trace (which would re-couple it to composition). v1 entries ignore
    everything but return a constant; the full context is still passed so
    upgrading an op from constant to analytical is a one-entry edit.

    The A2 fields (`placement`/`access`/`live_set`/`dtype_bits`) default to
    empty/None so every existing bare `OpCostCtx(name)` construction is
    unchanged and the faithful number is unaffected (the constant `OpCost.fn`
    ignores them). Population is opt-in per compose: D2 fills `access`
    (task 006), D3 fills `dtype_bits` (task 007).
    """

    op_name: str
    iters: int = 1                       # resolved inner trip count (D3)
    operand_shapes: tuple = ()           # ((M,K),...) where known
    lane_width: int | None = None        # device SIMD/fold width (geometry)
    mode: str = ""                       # placement mode passthrough
    extra: dict = field(default_factory=dict)  # placement.extra passthrough
    # NEW (A2):
    placement: dict = field(default_factory=dict)  # {role -> Handle} for THIS op
    access: "AccessDescr | None" = None             # layout-derived access pattern (D2)
    live_set: int | None = None          # register pressure here (placement task reads)
    dtype_bits: int | None = None        # D3: operand element bit-width


@dataclass(frozen=True)
class MoveCostCtx:
    """Immutable local context handed to a `MoveCost.fn`.

    The data-movement analogue of `OpCostCtx`: burst length, src/dst tier,
    element count for a per-move analytical model. v1 entries return a
    constant (or a fan-out width) and ignore the context.
    """

    move_name: str
    iters: int = 1
    operand_shapes: tuple = ()
    elem_count: int | None = None        # elements moved, where known
    src_tier: str | None = None
    dst_tier: str | None = None
    extra: dict = field(default_factory=dict)
    # NEW (A2):
    placement: dict = field(default_factory=dict)
    access: "AccessDescr | None" = None
    live_set: int | None = None


# --------------------------------------------------------------------- #
# Layer A entries (design 04 §1.1)
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class OpCost:
    """Per-op cost entry. `fn` returns cycles for ONE issued op given its
    local `OpCostCtx`. A constant-cost op is `OpCost(lambda ctx: 4)`.
    `note` is free-form provenance (citation), surfaced in breakdowns.
    `provenance` is the A4 STRUCTURED tag (design 07 §A4.1); default
    `ASSUMPTION` is the honest worst case -- forgetting to tag widens the
    confidence band, never silently narrows it.
    """

    fn: Callable[["OpCostCtx"], "int | float"]
    note: str = ""
    provenance: Provenance = Provenance.ASSUMPTION


@dataclass(frozen=True)
class MoveCost:
    """Per-move cost entry; the data-movement analogue of `OpCost`.

    Some entries (the Samsung host_staging STAGE_BCAST / GATHER_FAN) carry a
    *fan-out width*, not a cycle count -- `compose` reads them as model
    parameters (design 04 §1.2 edge case). The interface is the same: a
    callable of `MoveCostCtx`. `provenance` is the A4 structured tag.
    """

    fn: Callable[["MoveCostCtx"], "int | float"]
    note: str = ""
    provenance: Provenance = Provenance.ASSUMPTION


# --------------------------------------------------------------------- #
# Layer B context + result (design 04 §1.1)
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class ComposeCtx:
    """Everything a `compose` fn folds: the structural target, the match
    trace, and the chosen placement. `compose` reads geometry off
    `target` (mapping fan-outs, register lanes) and costs off the bound
    `CostModel` (never `target.move(...).cycles`).
    """

    target: Any
    trace: Any
    layout: Any


@dataclass(frozen=True)
class CostResult:
    """What a `compose` fn returns.

    `cycles` is the whole-program estimate (the int the argmin expects).
    `phases` is the A1 phase timeline: a `list[Phase]` (design 07 §A1.2),
    the structural carrier the combiner folds. The legacy dict view is
    derived on demand via `phases_as_dict` at the surfacing boundary
    (`evaluate`); composes always emit a list.
    `confidence` is a static annotation the model author sets:
    "calibrated" | "coarse" | "placeholder".
    """

    cycles: int
    phases: "list[Phase]" = field(default_factory=list)
    confidence: str = "calibrated"


# --------------------------------------------------------------------- #
# The CostModel (design 04 §1.1)
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class CostModel:
    """A bound, two-layer cost model for one `(target, flavor, concern)`.

    Layer A is `op_costs` / `move_costs` (dicts of callables). Layer B is
    `compose`. `constants` carries flat scalars (UPMEM revolver_latency)
    that migrated off the target tree.

    The `compose` fn calls `op_cost` / `move_cost` / `const` on this
    model instead of `target.op(n).cycles` / `target.move(n).cycles` /
    `target.<name>`.
    """

    name: str                          # e.g. "samsung_faithful"
    target_name: str                   # which structural target it prices
    op_costs: dict                     # {Op.name: OpCost}
    move_costs: dict                   # {Move.name: MoveCost}
    constants: dict                    # {name: scalar}
    compose: Callable[["ComposeCtx"], "CostResult"]
    flavor: str = "faithful"
    concern: str = "kernel_cycles"
    confidence: str = "calibrated"
    # design 04 §3.2 tier-3: the DECLARED per-op fallback trip count when
    # `resolve_trip_count` returns None (a genuinely dynamic / data-dependent
    # bound). v1 default is 1, but compose stamps a `"dynamic_assumed"` phase
    # marker and downgrades confidence to "coarse" so the estimate is never
    # silently wrong -- it is *visibly* coarse. A model author overrides per
    # op via the dict; the `"*"` key sets the catch-all default.
    dynamic_trip_defaults: dict = field(default_factory=dict)
    # A4 (design 07 §A4.2): per-(target, flavor) calibration data, read by
    # the confidence-gate + the user/optimizer; NOT consulted by argmin.
    calibration: CalibrationRecord = field(default_factory=CalibrationRecord)

    def dynamic_trip_default(self, op_name: str) -> "int | str":
        """Declared fallback trip count for `op_name` (design 04 §3.2).

        Returns the per-op override if set, else the model's `"*"` default,
        else 1. May return the sentinel `"unbounded"` for a model that wants
        to flag a data-dependent loop it refuses to bound.
        """
        if op_name in self.dynamic_trip_defaults:
            return self.dynamic_trip_defaults[op_name]
        return self.dynamic_trip_defaults.get("*", 1)

    def op_cost(self, name: str, ctx: "OpCostCtx | None" = None):
        """Cycles for one issued op `name` under `ctx`."""
        entry = self.op_costs.get(name)
        if entry is None:
            raise KeyError(
                f"CostModel {self.name!r} has no op cost for {name!r}"
            )
        if ctx is None:
            ctx = OpCostCtx(op_name=name)
        return entry.fn(ctx)

    def move_cost(self, name: str, ctx: "MoveCostCtx | None" = None):
        """Cycles (or fan-out width) for move `name` under `ctx`."""
        entry = self.move_costs.get(name)
        if entry is None:
            raise KeyError(
                f"CostModel {self.name!r} has no move cost for {name!r}"
            )
        if ctx is None:
            ctx = MoveCostCtx(move_name=name)
        return entry.fn(ctx)

    def has_op_cost(self, name: str) -> bool:
        return name in self.op_costs

    def const(self, name: str):
        try:
            return self.constants[name]
        except KeyError as e:
            raise KeyError(
                f"CostModel {self.name!r} has no constant {name!r}"
            ) from e


# --------------------------------------------------------------------- #
# Registry (design 04 §1.5)
# --------------------------------------------------------------------- #


# keyed by (target_name, flavor, concern)
_model_registry: dict[tuple[str, str, str], CostModel] = {}


def register_cost_model(model: CostModel) -> CostModel:
    """Register `model` under `(target_name, flavor, concern)`.

    Idempotent re-registration of the *same* model object is allowed (the
    tables module is imported once, but a re-import under a reloaded
    package should not raise); a *different* model under an existing key
    is an error.
    """
    key = (model.target_name, model.flavor, model.concern)
    existing = _model_registry.get(key)
    if existing is not None and existing is not model:
        raise ValueError(
            f"duplicate CostModel for {key!r}: {existing.name!r} vs "
            f"{model.name!r}"
        )
    _model_registry[key] = model
    return model


def get_cost_model(
    target_name: str, flavor: str = "faithful", concern: str = "kernel_cycles"
) -> CostModel:
    """Look up the CostModel bound to `(target_name, flavor, concern)`."""
    _ensure_tables_loaded()
    key = (target_name, flavor, concern)
    model = _model_registry.get(key)
    if model is None:
        raise KeyError(
            f"no CostModel registered for {key!r}; registered: "
            f"{sorted(_model_registry)}"
        )
    return model


def _ensure_tables_loaded() -> None:
    """Import the tables module so its CostModels self-register.

    Lazy + idempotent: avoids an import cycle (tables import this module)
    while still letting `get_cost_model` work before anything else has
    touched the tables.
    """
    if _model_registry:
        return
    from . import spmw_cost_tables  # noqa: F401  -- registers all models


# --------------------------------------------------------------------- #
# Sim-free evaluation entry point (design 04 §2.3)
# --------------------------------------------------------------------- #


def evaluate(target, trace, layout, flavor: str = "faithful") -> CostResult:
    """Sim-free whole-program estimate for `(target, trace, layout)`.

    This is the seam the virtual backend's `_run_virtual` delegates to.
    It imports nothing from the simulator paths, calls no subprocess and
    no Docker -- its only inputs are the bound `CostModel` and the
    in-memory trace, so the no-sim guarantee holds by construction.

    Whole-program = `kernel_cycles + host_staging` (design 05 §5, task-017):
    the device-exec body plus the host<->device staging, folded by the
    flavor-bound combiner (default `faithful` = serial sum, design 07 §A1.3).
    Targets with no `host_staging` model registered return `kernel_cycles`
    unchanged.

    This is the SURFACING BOUNDARY: the device + host composes each emit a
    `list[Phase]`; `evaluate` concatenates them, folds via `combine`, and
    returns a `CostResult` whose `.phases` is the DERIVED dict view
    (`phases_as_dict` + the `preload`/`exec` aliases). The dict view is what
    `RunResult.extra["phases"]` + the report harness read (design 07 §A1.2:
    the shim is the only place the dict shape survives). The `compose`
    results above keep the structural `list[Phase]` carrier.
    """
    name = getattr(target, "name", None)
    model = get_cost_model(name, flavor)
    overlap = _flavor_overlap(name, flavor)
    device = model.compose(ComposeCtx(target, trace, layout))
    try:
        hs_model = get_cost_model(name, flavor, concern="host_staging")
    except KeyError:
        return CostResult(
            cycles=combine(list(device.phases), overlap=overlap),
            phases=_surface_phases(list(device.phases)),
            confidence=device.confidence,
        )
    host = hs_model.compose(ComposeCtx(target, trace, layout))
    merged = list(device.phases) + list(host.phases)
    confidence = (
        "coarse"
        if "coarse" in (device.confidence, host.confidence)
        else device.confidence
    )
    return CostResult(
        cycles=combine(merged, overlap=overlap),
        phases=_surface_phases(merged),
        confidence=confidence,
    )


def _surface_phases(phases: "list[Phase]") -> dict:
    """Derived dict view for the surfacing boundary (design 07 §A1.2).

    `phases_as_dict` grouped by tag, plus the legacy `preload` alias
    (= `stage_resident` + `stage_per_call`) the report harness + the virtual
    backend test read. Read-only, derived; the only place the dict survives.
    """
    out = phases_as_dict(phases)
    out["preload"] = out.get("stage_resident", 0) + out.get("stage_per_call", 0)
    return out


def _flavor_overlap(target_name, flavor: str) -> bool:
    """Whether `flavor` binds the overlap fold (design 07 §A1.3). Imported
    lazily from the tables' `_COMBINER_FOR_FLAVOR` to avoid the import cycle;
    every flavor not in the table defaults to the serial sum (`False`)."""
    from .spmw_cost_models import _COMBINER_FOR_FLAVOR
    return _COMBINER_FOR_FLAVOR.get(flavor, False)
