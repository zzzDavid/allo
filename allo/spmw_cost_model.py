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

from dataclasses import dataclass, field
from typing import Any, Callable


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
    """

    op_name: str
    iters: int = 1                       # resolved inner trip count (D3)
    operand_shapes: tuple = ()           # ((M,K),...) where known
    lane_width: int | None = None        # device SIMD/fold width (geometry)
    mode: str = ""                       # placement mode passthrough
    extra: dict = field(default_factory=dict)  # placement.extra passthrough


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


# --------------------------------------------------------------------- #
# Layer A entries (design 04 §1.1)
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class OpCost:
    """Per-op cost entry. `fn` returns cycles for ONE issued op given its
    local `OpCostCtx`. A constant-cost op is `OpCost(lambda ctx: 4)`.
    `note` is free-form provenance (citation), surfaced in breakdowns.
    """

    fn: Callable[["OpCostCtx"], "int | float"]
    note: str = ""


@dataclass(frozen=True)
class MoveCost:
    """Per-move cost entry; the data-movement analogue of `OpCost`.

    Some entries (the Samsung host_staging STAGE_BCAST / GATHER_FAN) carry a
    *fan-out width*, not a cycle count -- `compose` reads them as model
    parameters (design 04 §1.2 edge case). The interface is the same: a
    callable of `MoveCostCtx`.
    """

    fn: Callable[["MoveCostCtx"], "int | float"]
    note: str = ""


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
    `phases` is the per-phase breakdown (feeds `RunResult.extra`).
    `confidence` is a static annotation the model author sets:
    "calibrated" | "coarse" | "placeholder".
    """

    cycles: int
    phases: dict = field(default_factory=dict)
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
    the device-exec body plus the host<->device staging, summed (the default
    combiner). Targets with no `host_staging` model registered return
    `kernel_cycles` unchanged. The merged `phases` carry both the device
    `exec` and the host `stage_resident`/`stage_per_call`/`readback`
    breakdown, plus a `preload` alias (= resident + per_call) for callers
    reading the legacy phase name.
    """
    name = getattr(target, "name", None)
    model = get_cost_model(name, flavor)
    device = model.compose(ComposeCtx(target, trace, layout))
    try:
        hs_model = get_cost_model(name, flavor, concern="host_staging")
    except KeyError:
        return device
    host = hs_model.compose(ComposeCtx(target, trace, layout))
    phases = dict(device.phases)
    phases.update(host.phases)
    phases["preload"] = host.phases.get("stage_resident", 0) + host.phases.get(
        "stage_per_call", 0
    )
    confidence = (
        "coarse"
        if "coarse" in (device.confidence, host.confidence)
        else device.confidence
    )
    return CostResult(
        cycles=device.cycles + host.cycles, phases=phases, confidence=confidence
    )
