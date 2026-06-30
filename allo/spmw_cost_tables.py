# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The faithful `CostModel`s -- the numbers, not the mechanism (design 04
§1.4, §5).

This is the "Energy-Reference-Table" file: one `CostModel` per
`(target, flavor)` for the `kernel_cycles` concern. The per-op / per-move
numbers and the per-target `compose` bodies are lifted *verbatim* from
the pre-decoupling `spmw_cost_models._xxx_kernel_cycles` bodies, so the
faithful flavor's argmin is byte-identical (report 18 invariants: 15251,
B*=2, 3.93x). A researcher refining a per-op cost against board profiling
edits ONLY this file; the mechanism in `spmw_cost_model.py` never moves.

The trivial v1 entry is a constant `OpCost(lambda ctx: N)` -- the full
`OpCostCtx` is still passed so the APU-v1-MICRO-2025 refinement is a
one-entry edit (design 04 §1.3, T-APU-v1).
"""

from __future__ import annotations

import math
from dataclasses import replace

from .spmw_cost_model import (
    AccessDescr,
    CalibrationRecord,
    CostModel,
    MoveCostCtx,
    OpCost,
    OpCostCtx,
    MoveCost,
    CostResult,
    Phase,
    Provenance,
    Resource,
    register_cost_model,
    register_knob_cost,
)
from .spmw_linear_layout import LinearLayout
from .spmw_match import MatchTrace
from .spmw_target import MemoryRef, Register
from .spmw_tripcount import resolve_bound_text, resolve_trip_count


def _unwrap(handle):
    """Strip a `Spilled` wrapper, returning the home handle (design parity
    with the pre-decoupling `_unwrap`)."""
    try:
        from .spmw_regalloc import Spilled
    except ImportError:
        return handle
    if isinstance(handle, Spilled):
        return handle.home_handle
    return handle


# --------------------------------------------------------------------- #
# Shared trip-count helper: resolve the inner bound for a match site.
# Tier-1/2 (literal / affine-over-shapes+mapping); tier-3 returns None and
# the caller applies its declared fallback (design 04 §3).
# --------------------------------------------------------------------- #


def _mapping_env(target) -> dict[str, int]:
    """Unit-tree mapping fan-outs as an affine env. Names are the unit
    names; values are the product of that unit's mapping list. Used to
    resolve mapping-dependent bounds (design 04 §3.2)."""
    env: dict[str, int] = {}
    for u in target._walk():
        prod = 1
        for f in u.mapping:
            prod *= f
        env[u.name] = prod
    return env


def _resolve_inner(target, match) -> int | None:
    """Innermost (reduction) trip count for `match`, or None if dynamic.

    None is the tier-3 signal; each compose applies the model's DECLARED
    `dynamic_trip_default` and downgrades confidence (design 04 §3.2),
    never the silent `=1`.
    """
    return resolve_trip_count(match, -1, mapping_env=_mapping_env(target))


def _resolve_iters(model, target, match) -> tuple[int, bool]:
    """Inner trip count + a `dynamic` flag (design 04 §3.2 tier-3).

    Tier 1/2 -> (resolved_count, False). Tier 3 (None) -> the model's
    DECLARED `dynamic_trip_default(op_name)` with `dynamic=True`, so the
    caller can mark the estimate coarse instead of silently assuming 1.
    `"unbounded"` is coerced to the v1 default of 1 (still flagged dynamic).
    """
    inner_ub = _resolve_inner(target, match)
    if inner_ub is not None:
        return inner_ub, False
    default = model.dynamic_trip_default(match.target_op_name)
    if not isinstance(default, int):
        default = 1
    return default, True


def _confidence(model, dynamic: bool) -> str:
    """`"coarse"` when any tier-3 fallback fired, else the model's declared
    confidence (design 04 §3.2 -- never silently wrong, *visibly* coarse)."""
    return "coarse" if dynamic else model.confidence


# --------------------------------------------------------------------- #
# A1 phase-carrier helpers (design 07 §A1.4): every faithful compose uses
# the COLLAPSED encoding (latency=total, ii=0, count=1) -> phase_cycles ==
# total, so the serial-sum fold is byte-identical to the old per-phase /
# whole-program sum (§A1.6 proof). The exec/dma/host helpers keep the call
# sites terse; the `dynamic_assumed` marker is a ZERO-COST tagged phase
# (a flag in the breakdown, never folded into cycles).
# --------------------------------------------------------------------- #


def _exec_phase(total: int, tag: str = "exec") -> Phase:
    return Phase(Resource.COMPUTE, latency=int(total), ii=0, count=1, tag=tag)


def _host_phase(total: int, tag: str) -> Phase:
    return Phase(Resource.HOST, latency=int(total), ii=0, count=1, tag=tag)


def _dma_phase(total: int, tag: str = "vr_dma") -> Phase:
    return Phase(Resource.DMA, latency=int(total), ii=0, count=1, tag=tag)


def _dynamic_marker() -> Phase:
    """Zero-cost COMPUTE phase tagged `dynamic_assumed` -- the visible tier-3
    flag (design 04 §3.2). `latency=0` so it contributes nothing to the
    fold; the tag surfaces it in the breakdown (the old dict `=1` flag)."""
    return Phase(Resource.COMPUTE, latency=0, ii=0, count=1, tag="dynamic_assumed")


# --------------------------------------------------------------------- #
# D2 locality term (design 07 §D2): a Resource.LOCALITY phase whose penalty
# is layout-derived (AccessDescr.conflict_count) x ONE provenance-tagged
# per-backend row-buffer constant. Conflict-free layouts (everything
# optimal_swizzle emits) -> conflict_count == 0 -> penalty 0 -> NO LOCALITY
# phase emitted -> the faithful corpus number is byte-identical by
# construction (design 07 §D2.2). The term only moves a number when a worse
# (parity-colliding) swizzle is enumerated.
# --------------------------------------------------------------------- #


def _access_descr_from_layout(layout) -> AccessDescr:
    """Derive an `AccessDescr` for the locality term from `layout` (design 07
    §A2.3 / §D2). A future placement task may stamp a fully-formed
    `AccessDescr` on `layout.extra["access"]`; otherwise we derive
    `conflict_count` from a promoted `LinearLayout` carried on
    `layout.extra["bank_layout"]` together with its `bank_dims` /
    `varying_inputs` (the layout-derived F2 conflict predicate,
    `spmw_linear_layout.conflict_count`, design 07 §A2.4). Absent any layout
    to derive from, return the conflict-free identity (penalty 0) -- the
    back-compat default that keeps the corpus byte-identical.
    """
    extra = getattr(layout, "extra", {}) or {}
    access = extra.get("access")
    if isinstance(access, AccessDescr):
        return access
    ll = extra.get("bank_layout")
    bank_dims = extra.get("bank_dims")
    varying_inputs = extra.get("varying_inputs")
    if isinstance(ll, LinearLayout) and bank_dims and varying_inputs:
        n = ll.conflict_count(
            bank_dims=tuple(bank_dims), varying_inputs=tuple(varying_inputs)
        )
        if n == 0:
            return AccessDescr.identity(tier="near_bank")
        return AccessDescr(
            tier="near_bank",
            bank_dims=tuple(bank_dims),
            conflict_free=False,
            conflict_count=n,
        )
    return AccessDescr.identity()


def _locality_phase(model, layout):
    """The D2 locality `Phase` (design 07 §D2.1), or `None` when the access is
    conflict-free (penalty 0 -> no phase, so the fold is untouched).

        penalty = access.conflict_count * ROW_BUFFER_MISS

    `ROW_BUFFER_MISS` is the model's provenance-tagged per-backend row-buffer
    constant (read off the bound model, NEVER pasted in the compose body). A
    model with no `ROW_BUFFER_MISS` entry contributes no locality phase --
    the term is opt-in per backend.
    """
    access = _access_descr_from_layout(layout)
    if access.conflict_count <= 0:
        return None
    if "ROW_BUFFER_MISS" not in model.move_costs:
        return None
    miss = model.move_cost("ROW_BUFFER_MISS", MoveCostCtx("ROW_BUFFER_MISS"))
    penalty = access.conflict_count * miss
    if penalty <= 0:
        return None
    return Phase(Resource.LOCALITY, latency=int(penalty), ii=0, count=1,
                 tag="locality")


# ===================================================================== #
# Samsung HBM-PIM (faithful)  -- design 04 §1.2/§1.5, SPEC-026
# ===================================================================== #


def _samsung_workid_count(target) -> int:
    # Delegate to target.work_grid() so the canonical full-grid work-id count
    # (== unit-tree fanout product) is computed in exactly one place. The
    # `f != 1` filter in work_grid only drops no-op [1] factors, so the product
    # is identical to the historical all-factor multiply.
    return target.work_grid()[1]


def _trace_batch_dim(trace: MatchTrace) -> int:
    return max((m.extra.get("batch_dim", 1) for m in trace.matches), default=1)


def _samsung_output_rows(target, match, batch_var: str | None) -> int:
    loops = match.enclosing_loops
    if len(loops) < 2:
        return 1
    rows = 1
    menv = _mapping_env(target)
    for (var, _lb, ub, _step) in loops[:-1]:  # drop innermost (reduction)
        if batch_var is not None and var == batch_var:
            continue
        b = resolve_bound_text(ub, mapping_env=menv)
        if b is not None:
            rows *= b
    return rows


def _samsung_mk(target, trace: MatchTrace) -> tuple[int, int]:
    n_workids = _samsung_workid_count(target)
    menv = _mapping_env(target)
    M = 0
    K = 0
    for match in trace.matches:
        if not match.enclosing_loops:
            continue
        k = resolve_trip_count(match, -1, mapping_env=menv)
        if k is not None:
            K = max(K, k)
        batch_var = match.extra.get("batch_loop_var")
        rows = _samsung_output_rows(target, match, batch_var)
        M = max(M, rows * n_workids)
    return M, K


def _samsung_preload_cycles(hs_model, M: int, K: int) -> int:
    """Preload (broadcast/scatter) staging cost (design 05 §5).

    Re-homed off the old `PRELOAD_*` self-moves onto the `host_staging`
    CostModel's `STAGE_*` carriers (369/1/2 VERBATIM, task-017). `hs_model`
    is the `host_staging`-concern CostModel. The arithmetic is unchanged:
    `(M*K // fan) * per_group + crf_upload`.
    """
    if M <= 0 or K <= 0:
        return 0
    write_width = hs_model.move_cost("STAGE_BCAST", MoveCostCtx("STAGE_BCAST"))
    write_cyc = hs_model.move_cost("STAGE_SCATTER", MoveCostCtx("STAGE_SCATTER"))
    crf_cyc = hs_model.move_cost("STAGE_CRF", MoveCostCtx("STAGE_CRF"))
    return (M * K // write_width) * write_cyc + crf_cyc


def _samsung_readback_cycles(hs_model, M: int) -> int:
    """Readback (gather) staging cost (design 05 §5). Re-homed off
    `READBACK_*` onto the `host_staging` `GATHER_*` carriers (4096/181
    VERBATIM)."""
    if M <= 0:
        return 0
    read_width = hs_model.move_cost("GATHER_FAN", MoveCostCtx("GATHER_FAN"))
    read_cyc = hs_model.move_cost("GATHER_RD", MoveCostCtx("GATHER_RD"))
    return ((M + read_width - 1) // read_width) * read_cyc


def _samsung_host_staging_compose_with(hs_model, ctx):
    """`host_staging`-concern compose (design 05 §5, task-017),
    parameterised by the bound `host_staging` model so the faithful +
    optimistic flavors reuse one phase algebra.

    Prices the host<->device staging for a Samsung GEMV: preload
    (broadcast/scatter the weight) + readback (gather the output). The
    structural resident flag (`layout.extra["stage_resident"]`, stamped by
    the 013 hoist via the enumerator — bridge option (b)) selects
    preload-once (resident) vs preload-B (per-call). Readback is always paid
    per batch vector.

    Returns a per-phase `CostResult.phases` breakdown so the whole-program
    combiner can later become `max` for async overlap (design 05 §5,
    open-Q4) without touching this function; the default combiner is `sum`.
    Whole-program = kernel_cycles + host_staging.
    """
    target = ctx.target
    trace = ctx.trace
    layout = ctx.layout
    M, K = _samsung_mk(target, trace)
    B = _trace_batch_dim(trace)
    preload_cyc = _samsung_preload_cycles(hs_model, M, K)
    readback_cyc = _samsung_readback_cycles(hs_model, M)
    resident = bool(getattr(layout, "extra", {}).get("stage_resident", False))
    if resident:
        stage_resident = preload_cyc          # preload paid ONCE
        stage_per_call = 0
    else:
        stage_resident = 0
        stage_per_call = B * preload_cyc       # preload paid per vector
    readback_total = B * readback_cyc
    # Cross-op residency (SPEC-023 T6/D2): when a cross-kernel activation is
    # kept RESIDENT on-device (`extra["residency"] == "resident"`), the
    # inter-kernel host round-trip it would otherwise pay (gather the producer
    # output to host + re-scatter it for the consumer) is ELIDED. The credit is
    # one host gather (`readback_cyc`, the SAME helper) per resident crossing
    # activation -- a NEGATIVE host-staging phase, the staging the resident
    # schedule avoids. `restage` (or key absent) elides nothing -> byte-
    # identical. The `residency` knob_cost (register_knob_cost, §A3) pins this
    # shape exactly; this is where it folds into the argmin (mirroring how
    # `stage_resident` folds through this same compose, not the dormant
    # knob_phases seam).
    resident_credit = 0
    if str(getattr(layout, "extra", {}).get("residency", "restage")) == "resident":
        n_resident = len(getattr(layout, "extra", {}).get("residency_pairs", {}) or {})
        resident_credit = n_resident * readback_cyc
    cycles = stage_resident + stage_per_call + readback_total - resident_credit
    phases = [
        _host_phase(stage_resident, "stage_resident"),
        _host_phase(stage_per_call, "stage_per_call"),
        _host_phase(readback_total, "readback"),
    ]
    # Append the elision phase ONLY when there is a credit, so a non-resident
    # placement's phase list is byte-identical to today (the regression anchor).
    if resident_credit:
        phases.append(_host_phase(-resident_credit, "residency_elision"))
    return CostResult(
        cycles=cycles,
        phases=phases,
        confidence=hs_model.confidence,
    )


def _samsung_host_staging_compose(ctx):
    return _samsung_host_staging_compose_with(SAMSUNG_HOST_STAGING, ctx)


def _samsung_optimistic_host_staging_compose(ctx):
    return _samsung_host_staging_compose_with(SAMSUNG_OPTIMISTIC_HOST_STAGING, ctx)


def _samsung_compose(ctx):
    # Single phase-algebra implementation, parameterised by the bound model
    # (see `_samsung_compose_with`) so the faithful + optimistic flavors
    # never diverge.
    return _samsung_compose_with(SAMSUNG_FAITHFUL, ctx)


SAMSUNG_FAITHFUL = CostModel(
    name="samsung_faithful",
    target_name="samsung_hbm_pim",
    op_costs={
        # tCCDL = 4 (column-strobe period). MUL/MAC both 4 cyc. The exec
        # tCCDL=4 is the report-18 15251-anchored datasheet period (A4).
        "MAC": OpCost(lambda c: 4, note="tCCDL, Samsung ISCA'21 §4.1",
                      provenance=Provenance.MEASURED),
        "MUL": OpCost(lambda c: 4, note="tCCDL", provenance=Provenance.MEASURED),
    },
    move_costs={
        # spec 015 §6.1: load = tCCDL+RL+BL//2 = 26; store = tCCDL+WL+BL//2 = 14
        "LD_A": MoveCost(lambda c: 26, note="tCCDL+RL+BL//2",
                         provenance=Provenance.DATASHEET),
        "LD_B": MoveCost(lambda c: 26, note="tCCDL+RL+BL//2",
                         provenance=Provenance.DATASHEET),
        "ST_A": MoveCost(lambda c: 14, note="tCCDL+WL+BL//2",
                         provenance=Provenance.DATASHEET),
        "ST_B": MoveCost(lambda c: 14, note="tCCDL+WL+BL//2",
                         provenance=Provenance.DATASHEET),
        "JUMP": MoveCost(lambda c: 1, note="1-cyc control op",
                         provenance=Provenance.DATASHEET),
        # CRF_TRIGGER is the lever-3 (SPEC-025 shared-CRF) per-work-id fire
        # latency, consumed in the DEVICE-exec phase (`trigger_cyc`). It
        # stays in kernel_cycles (lever 3 is retained, task-017). The old
        # PRELOAD_*/READBACK_* preload/readback constants re-homed into the
        # host_staging concern below (SAMSUNG_HOST_STAGING). A genuine guess
        # (not yet sim-anchored) -> ASSUMPTION.
        "CRF_TRIGGER": MoveCost(lambda c: 2, note="host per-tile fire latency",
                                provenance=Provenance.ASSUMPTION),
        # D2 (design 07 §D2.1): per-bank row-buffer miss penalty. HBM2 row
        # cycle = tRCD + tRP (activate + precharge) per row-buffer conflict;
        # the SAFARI PIM line owns the row-buffer penalty QUANTITY (report 27
        # §1.2 -- cite, do not claim). tRCD+tRP ~ 14+14 = 28 cyc at the
        # PIMSimulator clock. Read off the model by the D2 locality term;
        # never pasted in the compose body. DATASHEET (JEDEC HBM2 timing).
        "ROW_BUFFER_MISS": MoveCost(lambda c: 28,
                                    note="HBM2 tRCD+tRP row-buffer miss; "
                                         "SAFARI PIM line (report 27 §1.2)",
                                    provenance=Provenance.DATASHEET),
    },
    constants={},
    compose=_samsung_compose,
    calibration=CalibrationRecord(
        validated_against="report-18 PIMSimulator GEMV M=4096 K=1024",
        residual_error=0.0,
        shape_coverage=((4096, 1024),),
    ),
)


# host_staging concern (design 05 §5, task-017): the report-18 preload /
# readback calibration anchors re-homed VERBATIM off the deleted
# PRELOAD_*/READBACK_* self-moves. NEW concern on the same
# (target_name, flavor, concern) registry — not a fresh extraction.
SAMSUNG_HOST_STAGING = register_cost_model(CostModel(
    name="samsung_host_staging",
    target_name="samsung_hbm_pim",
    flavor="faithful",
    concern="host_staging",
    op_costs={},
    move_costs={
        "STAGE_BCAST": MoveCost(lambda c: 369, note="HAB preload fan-out width",
                                provenance=Provenance.MEASURED),
        "STAGE_SCATTER": MoveCost(lambda c: 1, note="per-group column-strobe",
                                  provenance=Provenance.MEASURED),
        "STAGE_CRF": MoveCost(lambda c: 2, note="programCrf upload",
                              provenance=Provenance.ASSUMPTION),
        "GATHER_FAN": MoveCost(lambda c: 4096, note="readback tile width",
                               provenance=Provenance.MEASURED),
        "GATHER_RD": MoveCost(lambda c: 181, note="per-tile readResult",
                              provenance=Provenance.MEASURED),
    },
    constants={},
    compose=_samsung_host_staging_compose,
    calibration=CalibrationRecord(
        validated_against="report-18 PIMSimulator GEMV M=4096 K=1024",
        residual_error=0.0,
        shape_coverage=((4096, 1024),),
    ),
))


# --------------------------------------------------------------------- #
# A3 (design 07 §A3): the one-knob `cost(value, ctx) -> list[Phase]` demo.
# The `stage_resident` knob's cost contribution IS the Samsung host_staging
# preload/readback branch re-expressed on the A1 timeline -- preload paid
# ONCE (resident) vs per-vector (non-resident). This proves the knob seam
# and the phase timeline are the SAME currency: a knob's cost is a
# `list[Phase]` the combiner folds exactly like the base phases (so a
# marginal per-knob delta is exact, not approximate -- the A3.3 basis the
# task 008 non-tautology test checks). The other five levers stay as
# `layout.extra` reads inside compose until placement-task D4 migrates them.
# --------------------------------------------------------------------- #


def _samsung_stage_resident_knob_cost(value, ctx) -> "list[Phase]":
    """`stage_resident` knob's phase contribution (design 07 §A3.2).

    `value` is the chosen resident bool; the HOST phases returned are EXACTLY
    the Samsung host_staging compose's phases for that resident choice -- so
    `knob_cost(...)` folds to the same number as the host_staging compose.
    The seam is exact by construction (it reuses the one phase algebra)."""
    from .spmw_autoschedule import Placement

    base_extra = dict(getattr(ctx.layout, "extra", {}) or {})
    base_extra["stage_resident"] = bool(value)
    knob_layout = Placement(
        placements=getattr(ctx.layout, "placements", {}),
        extra=base_extra,
        mode=getattr(ctx.layout, "mode", ""),
    )
    from .spmw_cost_model import ComposeCtx
    res = _samsung_host_staging_compose(
        ComposeCtx(ctx.target, ctx.trace, knob_layout)
    )
    return list(res.phases)


register_knob_cost(
    "samsung_hbm_pim", "stage_resident", _samsung_stage_resident_knob_cost
)


def _samsung_residency_knob_cost(value, ctx) -> "list[Phase]":
    """`residency` knob's phase contribution (SPEC-023 T6/D2).

    A cross-kernel value (MLP inter-layer activation `h`) either RE-STAGES
    through the host between layers (`"restage"`: gather the producer output to
    host + re-scatter to the consumer -- a `Resource.HOST` staging phase) or
    STAYS RESIDENT on-device (`"resident"`: that HOST round-trip is elided ->
    zero phases). So `restage` is strictly costlier and argmin earns
    `resident`; the saving is the inter-kernel host staging the resident value
    avoids -- expressed as a positive HOST phase on restage, none on resident
    (no negative phase needed; the seam carries it as landed).

    Exact-by-construction: the HOST phase reuses the SAME host_staging gather
    helper (`_samsung_readback_cycles`) the host_staging compose uses, sized by
    the crossing activation's element count (the producer output rows). A value
    that does NOT cross a boundary carries no `residency_crossing` entry, so
    `restage` returns `[]` -- byte-identical to today (no residency knob_cost
    effect on the per-kernel-local corpus).
    """
    extra = getattr(ctx.layout, "extra", {}) or {}
    crossing = extra.get("residency_crossing", {})
    if not crossing or value == "resident":
        return []  # resident elides the staging; non-crossing pays nothing
    # restage: one host gather+scatter round-trip per crossing activation.
    hs_model = SAMSUNG_HOST_STAGING
    M, _K = _samsung_mk(ctx.target, ctx.trace)
    if M <= 0:
        return []
    per_activation = _samsung_readback_cycles(hs_model, M)
    total = per_activation * len(crossing)
    if total <= 0:
        return []
    return [_host_phase(int(total), "residency_restage")]


register_knob_cost(
    "samsung_hbm_pim", "residency", _samsung_residency_knob_cost
)


def _upmem_residency_knob_cost(value, ctx) -> "list[Phase]":
    """`residency` knob's phase contribution for UPMEM (SPEC-023 T6/D2).

    Mirrors the Samsung shape, sized by the UPMEM inter-kernel MRAM transfer:
    `restage` pays an `LD_MRAM + ST_MRAM` round-trip per crossing activation (a
    positive DMA phase the resident schedule avoids); `resident` (or no
    crossing) contributes nothing. The argmin folds this through the UPMEM
    `host_staging` compose (where the credit also lands), so the resident arm
    is strictly cheaper and is earned. Exact-by-construction (same MRAM move
    costs)."""
    extra = getattr(ctx.layout, "extra", {}) or {}
    crossing = extra.get("residency_crossing", {})
    if not crossing or value == "resident":
        return []
    ld = UPMEM_FAITHFUL.move_cost("LD_MRAM", MoveCostCtx("LD_MRAM"))
    st = UPMEM_FAITHFUL.move_cost("ST_MRAM", MoveCostCtx("ST_MRAM"))
    total = (ld + st) * len(crossing)
    return [_dma_phase(int(total), "residency_restage")] if total > 0 else []


register_knob_cost("upmem", "residency", _upmem_residency_knob_cost)


def _tile_knob_cost(value, ctx) -> "list[Phase]":
    """`tile` knob's phase contribution (SPEC-023 D2).

    The identity tiling (or key absent) contributes NOTHING -> byte-identical.
    The win of a capacity-fitting non-identity retile is realized through the
    allocator's capacity gate (a smaller per-tile working set FITS a bounded
    tier the full nest would overflow, so the retiled candidate avoids the
    spill `total_cost` the identity pays) -- a STRUCTURAL win on the landed
    regalloc, not a separate phase here. So this knob_cost is empty by design;
    it exists so `(target, "tile")` is a registered seam (the read side gates
    on registration). A backend whose retile win needs an explicit phase
    (a fold-overhead model) registers it here as a follow-up -- this task
    pins the empty/structural shape and the generator; the sim-confirmed
    "retile beats the user's nest" is verifier task 008.
    """
    return []


# Register the tile knob_cost for every target that can carry a tile knob, so
# the cost read-side `active_knobs` recognises `(target, "tile")`. Empty
# contribution: the retile win flows through regalloc's capacity gate.
for _tname in ("samsung_hbm_pim", "mortise", "mortise_wide",
               "upmem", "apu_v1", "apu_v2"):
    register_knob_cost(_tname, "tile", _tile_knob_cost)


# ===================================================================== #
# SK-Hynix AiM (faithful)  -- JSSC 2023 §IV
# ===================================================================== #


def _aim_compose(ctx):
    target = ctx.target
    trace = ctx.trace
    model = AIM_FAITHFUL
    total = 0
    dynamic = False
    for match in trace.matches:
        # A5 (design 07 §A5): closed vocabulary -- an op not in op_costs is a
        # hard KeyError (via op_cost), not a silent per_op=4 default.
        per_op = model.op_cost(
            match.target_op_name, OpCostCtx(match.target_op_name)
        )
        iters, dyn = _resolve_iters(model, target, match)
        dynamic = dynamic or dyn
        total += per_op * iters
    phases = [_exec_phase(total)]
    loc = _locality_phase(model, ctx.layout)   # D2: 0 for conflict-free corpus
    if loc is not None:
        phases.append(loc)
        total += loc.latency
    if dynamic:
        phases.append(_dynamic_marker())
    return CostResult(cycles=total, phases=phases,
                      confidence=_confidence(model, dynamic))


AIM_FAITHFUL = CostModel(
    name="aim_faithful",
    target_name="aim",
    op_costs={
        # JSSC 2023 §IV: EWMUL=EWADD=4; MAC_SBK=8; MAC_ABK=16; AF=6.
        "MUL": OpCost(lambda c: 4, note="EWMUL", provenance=Provenance.DATASHEET),
        "ADD": OpCost(lambda c: 4, note="EWADD", provenance=Provenance.DATASHEET),
        "MAC": OpCost(lambda c: 8, note="MAC_SBK = 1 burst x 16 lanes",
                      provenance=Provenance.DATASHEET),
        "MAC_ABK": OpCost(lambda c: 16, note="all-bank broadcast = 2x",
                          provenance=Provenance.DATASHEET),
        "AF": OpCost(lambda c: 6, note="GELU/SIGMOID", provenance=Provenance.DATASHEET),
    },
    move_costs={
        # spill model: RD_SBK = tCCDL+RD+BURST = 24; ST_SBK = tCCDL+WR+BURST = 20
        "RD_SBK": MoveCost(lambda c: 24, note="tCCDL+RD+BURST",
                           provenance=Provenance.DATASHEET),
        "ST_SBK": MoveCost(lambda c: 20, note="tCCDL+WR+BURST",
                           provenance=Provenance.DATASHEET),
        # D2 (design 07 §D2.1): GDDR6 row-buffer miss penalty per bank
        # conflict (tRCD+tRP). JSSC 2023 §IV timing; SAFARI PIM line owns the
        # penalty quantity (report 27 §1.2 -- cite, do not claim). Read off
        # the model by the D2 locality term, never pasted in the compose.
        "ROW_BUFFER_MISS": MoveCost(lambda c: 22,
                                    note="GDDR6 tRCD+tRP row-buffer miss; "
                                         "AiM JSSC 2023 §IV / SAFARI PIM line",
                                    provenance=Provenance.DATASHEET),
    },
    constants={},
    compose=_aim_compose,
    calibration=CalibrationRecord(
        validated_against="ramulator2 AiM GEMV shape sweep K=512/1024/2048",
        shape_coverage=((512,), (1024,), (2048,)),
    ),
)


# ===================================================================== #
# UPMEM DPU (faithful)  -- uPIMulator / HPCA 2024 Table 2, research-021
# ===================================================================== #


def _upmem_compose(ctx):
    target = ctx.target
    trace = ctx.trace
    layout = ctx.layout
    model = UPMEM_FAITHFUL
    s = 0
    dynamic = False
    for match in trace.matches:
        # A5 (design 07 §A5): closed vocabulary -- unknown op = hard KeyError,
        # not the silent GPR (ADD) baseline default.
        per_op = model.op_cost(
            match.target_op_name, OpCostCtx(match.target_op_name)
        )
        iters, dyn = _resolve_iters(model, target, match)
        dynamic = dynamic or dyn
        s += per_op * iters
    r = model.const("revolver_latency")
    t = layout.extra.get("n_tasklets", 1)
    # The revolver fold stays INSIDE compose (design 07 §A1.7); the combiner
    # never sees r/t. Emitted as one collapsed COMPUTE phase.
    cycles = s + math.ceil(s * (r - 1) / min(t, r))
    phases = [_exec_phase(cycles)]
    # D2 (design 07 §D2): locality penalty is a SEPARATE LOCALITY phase, not
    # folded into the revolver-pipelined COMPUTE body. 0 for conflict-free.
    loc = _locality_phase(model, layout)
    if loc is not None:
        phases.append(loc)
        cycles += loc.latency
    if dynamic:
        phases.append(_dynamic_marker())
    return CostResult(cycles=cycles, phases=phases,
                      confidence=_confidence(model, dynamic))


UPMEM_FAITHFUL = CostModel(
    name="upmem_faithful",
    target_name="upmem",
    op_costs={
        # DPU GPR ops issue at 1 cyc; MAC = mul+add = 2 cyc (no fused MAC).
        "MUL": OpCost(lambda c: 1, note="GPR op", provenance=Provenance.DATASHEET),
        "ADD": OpCost(lambda c: 1, note="GPR op", provenance=Provenance.DATASHEET),
        "MAC": OpCost(lambda c: 2, note="mul+add, no fused MAC",
                      provenance=Provenance.DATASHEET),
    },
    move_costs={
        # HPCA 2024 Table 2: MRAM = 1000 cyc/64B burst; WRAM = 1 cyc.
        "LD_MRAM": MoveCost(lambda c: 1000, note="MRAM read per 64B burst",
                            provenance=Provenance.DATASHEET),
        "ST_MRAM": MoveCost(lambda c: 1000, note="MRAM write per 64B burst",
                            provenance=Provenance.DATASHEET),
        "LD_WRAM": MoveCost(lambda c: 1, note="WRAM->GPR fused",
                            provenance=Provenance.DATASHEET),
        "ST_WRAM": MoveCost(lambda c: 1, note="GPR->WRAM fused",
                            provenance=Provenance.DATASHEET),
        # D2 (design 07 §D2.1): DRAM-bank row-buffer miss per conflict. The
        # MRAM row activate/precharge dominates a row miss; HPCA 2024 §IV
        # MRAM-row timing, SAFARI PIM line owns the quantity (report 27 §1.2).
        # Read off the model by the D2 locality term, never pasted.
        "ROW_BUFFER_MISS": MoveCost(lambda c: 33,
                                    note="MRAM-row activate+precharge miss; "
                                         "HPCA 2024 §IV / SAFARI PIM line",
                                    provenance=Provenance.ASSUMPTION),
    },
    constants={
        # revolver scheduling window (uPIMulator src/main.go:115). NOT the
        # 14-stage pipeline depth (research-021). MEASURED off the
        # uPIMulator source (see calibration record); constants carry no
        # structured provenance field -- the calibration record is its anchor.
        "revolver_latency": 11,
    },
    compose=_upmem_compose,
    calibration=CalibrationRecord(
        validated_against="uPIMulator GEMV; revolver_latency from src/main.go:115",
        shape_coverage=((1024,), (2048,)),
    ),
)


def _upmem_host_staging_compose(ctx):
    """UPMEM `host_staging`-concern compose (design 05 §5/§7, task 008).

    OPT-IN: returns 0 unless the layout carries an explicit host-staging
    signal (`layout.extra["host_stage"]`). This keeps every existing UPMEM
    `kernel_cycles` estimate byte-identical when summed through the
    whole-program seam (the device GEMV traces carry no staging signal, so
    host_staging = 0).

    When present, `host_stage` is a list of `(collective, fan_degree)` pairs
    — the fan degree is the PLAIN INT host-side DPU partition fan
    (`prod(mapping)` of the `over=` unit level, e.g. 2560 / 2552), NEVER a
    `LinearLayout` (design 05 §1 invariant). A scatter/broadcast costs
    `fan * STAGE_XFER` (per-DPU prepare+push); a gather costs
    `fan * GATHER_XFER` (per-DPU copy-from). The non-pow2 fan type-checks and
    prices with no F2 constraint — the generality proof (T-NONPOW2).
    """
    layout = ctx.layout
    stages = list(getattr(layout, "extra", {}).get("host_stage", []) or [])
    # Cross-op residency (SPEC-023 T6/D2, UPMEM): a cross-kernel activation kept
    # RESIDENT on-device elides the inter-kernel MRAM round-trip it would
    # otherwise pay (the producer ST_MRAM of `local_h` + the consumer LD_MRAM).
    # `restage` (or key absent) pays it; `resident` elides it -> a negative
    # host-staging credit, so the argmin (kernel_cycles + host_staging) earns
    # resident. A value that does not cross carries no `residency_crossing`
    # entry -> zero credit -> byte-identical. Exact-by-construction: the credit
    # is the SAME MRAM move costs (LD_MRAM + ST_MRAM) the UPMEM kernel pays for
    # an inter-kernel activation transfer, per crossing value.
    resident_credit = 0
    rextra = getattr(layout, "extra", {}) or {}
    if str(rextra.get("residency", "restage")) == "resident":
        n_resident = len(rextra.get("residency_crossing", {}) or {})
        if n_resident:
            ld = UPMEM_FAITHFUL.move_cost("LD_MRAM", MoveCostCtx("LD_MRAM"))
            st = UPMEM_FAITHFUL.move_cost("ST_MRAM", MoveCostCtx("ST_MRAM"))
            resident_credit = n_resident * (ld + st)
    if not stages and not resident_credit:
        return CostResult(cycles=0, phases=[], confidence="calibrated")
    xfer = UPMEM_HOST_STAGING.move_cost("STAGE_XFER", MoveCostCtx("STAGE_XFER"))
    gxfer = UPMEM_HOST_STAGING.move_cost("GATHER_XFER", MoveCostCtx("GATHER_XFER"))
    scatter_bcast = 0
    gather = 0
    for collective, fan in stages:
        fan = int(fan)  # PLAIN INT — never routed through LinearLayout
        if collective in ("scatter", "broadcast"):
            scatter_bcast += fan * xfer
        elif collective == "gather":
            gather += fan * gxfer
        else:
            raise ValueError(
                f"upmem host_staging: unknown collective {collective!r}"
            )
    cycles = scatter_bcast + gather - resident_credit
    phases = [
        _host_phase(scatter_bcast, "stage_scatter_bcast"),
        _host_phase(gather, "gather"),
    ]
    if resident_credit:
        phases.append(_host_phase(-resident_credit, "residency_elision"))
    return CostResult(
        cycles=cycles,
        phases=phases,
        confidence="calibrated",
    )


# UPMEM host_staging concern (design 05 §7, task 008 generality proof). The
# per-DPU transfer constants (HPCA 2024 §IV CPU<->DPU bandwidth class) price
# a host-driven scatter/gather/broadcast over the integer DPU fan-out.
UPMEM_HOST_STAGING = register_cost_model(CostModel(
    name="upmem_host_staging",
    target_name="upmem",
    flavor="faithful",
    concern="host_staging",
    op_costs={},
    move_costs={
        "STAGE_XFER": MoveCost(lambda c: 1000, note="per-DPU prepare+push (MRAM-class)",
                               provenance=Provenance.DATASHEET),
        "GATHER_XFER": MoveCost(lambda c: 1000, note="per-DPU copy-from (MRAM-class)",
                                provenance=Provenance.DATASHEET),
    },
    constants={},
    compose=_upmem_host_staging_compose,
))


# ===================================================================== #
# GSI APU v1 (faithful)  -- design 01, report 12 §4.2, MICRO 2025
# ===================================================================== #


def _apu_v1_vr_tiling(target, trace: MatchTrace) -> tuple[int, int, int]:
    lane_width = target.vrs.width
    if lane_width is None or lane_width <= 0:
        lane_width = 1
    menv = _mapping_env(target)
    n_out = 1
    weight_elems = 0
    n_macs = 0
    for match in trace.matches:
        if getattr(match, "target_op_name", None) == "MAC":
            n_macs += 1
        loops = match.enclosing_loops or []
        if not loops:
            continue
        k = resolve_trip_count(match, -1, mapping_env=menv) or 1
        rows = 1
        for (_, _, ub_text, _) in loops[:-1]:
            ub = resolve_bound_text(ub_text, mapping_env=menv)
            if ub is not None:
                rows *= ub
        n_out = max(n_out, rows)
        weight_elems += rows * k
    n_out_tiles = max(1, math.ceil(n_out / lane_width))
    n_weight_tiles = max(1, math.ceil(weight_elems / lane_width))
    n_boundaries = max(0, n_macs - 1)
    return n_out_tiles, n_weight_tiles, n_boundaries


def _apu_v1_move_breakdown(model, target, trace: MatchTrace, layout):
    """(n_moves, per_move) for the chosen `vr_dma` schedule, or `(0, 0)` when
    no DMA is requested. Split out (design 07 §D1.4) so the overlap arm can
    build a LOOP-ENCODED DMA phase (`Phase(DMA, latency=per_move,
    ii=per_move, count=n_moves)`) whose fill is one `per_move` and whose
    steady state hides behind compute; the faithful arm just multiplies
    `n_moves * per_move` (collapsed, fully charged)."""
    vr_dma = getattr(layout, "extra", {}).get("vr_dma")
    if vr_dma is None:
        return 0, 0
    n_out_tiles, n_weight_tiles, n_boundaries = _apu_v1_vr_tiling(target, trace)
    if vr_dma == "intra":
        n_moves = n_weight_tiles + n_boundaries * n_out_tiles
    elif vr_dma == "inter":
        n_moves = n_weight_tiles
    else:
        raise ValueError(
            f"apu_v1 move cost: unknown vr_dma {vr_dma!r}; "
            "expected 'intra' or 'inter'"
        )
    per_move = (
        model.move_cost("DMA_L4_L1", MoveCostCtx("DMA_L4_L1"))
        + model.move_cost("LD_VR", MoveCostCtx("LD_VR"))
    )
    return n_moves, per_move


def _apu_v1_move_cycles(model, target, trace: MatchTrace, layout) -> int:
    n_moves, per_move = _apu_v1_move_breakdown(model, target, trace, layout)
    return n_moves * per_move


_APU_V1_ALLOWED_MODES = {"", "sv", "sv_lookup"}


# D3 (design 07 §D3): bit-serial width. On a bit-serial SRAM compute machine
# (GSI APU) an op is computed one bit-plane at a time, so its cycle cost scales
# with operand bit-width -- and a MULTIPLY's per-bit work is far steeper than
# an ADD's (MICRO'25 Table 5: mul_u16=201 vs add_u16=12, ~16x; mul_f16=77).
# The lambdas mirror that TABLE'S SHAPE (the `micro25_add_cost` affine template
# `gap_floor + seu_per_bit * bits`), not its absolute magnitudes -- the claim
# is the THREADING of width into the fused objective, not the APU table itself
# (report 27 §1.3 -- cite Stripes/BitFusion + the APU table, claim only the
# threading). Coefficients are anchored so width=16 reproduces today's
# constant (ADD=2, MUL=16, MAC=8), and the MUL slope is 8x the ADD slope (the
# table's mul>>add shape).
APU_V1_BASE_BITS = 16  # gvml_*_16: the width today's constants are quoted at.


def _apu_width_op(base_cyc, seu_per_bit, *, gap_floor=0.0):
    """Build a width-aware APU OpCost.fn (design 07 §D3.2).

    `dtype_bits is None` -> return `base_cyc` VERBATIM (the back-compat anchor:
    a width-absent call is byte-identical to today's constant). Otherwise scale
    affinely with the operand bit-width: `gap_floor + seu_per_bit * dtype_bits`,
    rounded. Coefficients are chosen by the caller so the value at
    `APU_V1_BASE_BITS` equals `base_cyc` (the anchor) and is monotonic in bits.
    """

    def fn(ctx):
        if getattr(ctx, "dtype_bits", None) is None:
            return base_cyc
        return int(round(gap_floor + seu_per_bit * ctx.dtype_bits))

    return fn


def _apu_v1_compose(ctx):
    # Faithful arm: collapsed DMA phase, byte-identical sum fold.
    return _apu_v1_compose_with(APU_V1_FAITHFUL, ctx, overlap=False)


def _apu_v1_overlap_compose(ctx):
    # Overlap arm (design 07 §D1.4): same phase algebra + numbers, but the DMA
    # phase is LOOP-ENCODED when the schedule is double-buffered so it can hide
    # behind compute under the overlap fold. Bound to the overlap flavor.
    return _apu_v1_compose_with(APU_V1_OVERLAP, ctx, overlap=True)


def _apu_v1_compose_with(model, ctx, *, overlap: bool):
    """APU v1 compose parameterised by model + an `overlap` flag (design 07
    §D1.4). `exec_ops` and the DMA `n_moves * per_move` total are IDENTICAL
    across both arms; the ONLY difference is the DMA phase ENCODING:

      * faithful (`overlap=False`): one COLLAPSED `Phase(DMA, latency=move_total,
        ii=0, count=1)` -> folds to `exec_ops + move_total` (byte-identical).
      * overlap (`overlap=True`) + double-buffered: one LOOP-ENCODED
        `Phase(DMA, latency=per_move, ii=per_move, count=n_moves)` -> same
        `phase_cycles = move_total`, but fill = `per_move` so the steady state
        hides behind compute under the max-across fold (D1.2). A serialized
        (non-double-buffered) overlap candidate keeps the collapsed phase, so
        its DMA cannot hide -- the D1.4 flip: double-buffered (b) scores below
        serialized (a) under overlap, equal under faithful.
    """
    target = ctx.target
    trace = ctx.trace
    layout = ctx.layout
    menv = _mapping_env(target)

    mode = getattr(layout, "mode", "")
    if mode not in _APU_V1_ALLOWED_MODES:
        raise ValueError(
            f"apu_v1 kernel_cycles: unknown layout.mode {mode!r}; "
            f"allowed: {sorted(_APU_V1_ALLOWED_MODES)}"
        )
    sv_raw = mode == "sv"
    exec_ops = 0
    dynamic = False
    for match in trace.matches:
        # D3 (design 07 §D3.1): thread the operand element bit-width to the
        # per-op OpCostCtx. The matcher stamps it on `match.extra["dtype_bits"]`
        # (the existing extension seam, like batch_dim); absent -> None ->
        # the width-aware lambda returns today's constant (byte-identical).
        bits = match.extra.get("dtype_bits")
        if sv_raw and match.target_op_name == "MAC":
            # SV (raw MUL+ADD) MAC: width-aware MUL + ADD, each fed the dtype.
            per_op = (
                model.op_cost("MUL", OpCostCtx("MUL", dtype_bits=bits))
                + model.op_cost("ADD", OpCostCtx("ADD", dtype_bits=bits))
            )
        else:
            # A5 (design 07 §A5): closed vocabulary -- unknown op = hard
            # KeyError (via op_cost), not the silent ADD-cycle default.
            per_op = model.op_cost(
                match.target_op_name,
                OpCostCtx(match.target_op_name, dtype_bits=bits),
            )
        # Outer loops only -- the innermost loop is subsumed by the 32K-lane
        # SIMD width, so it is NOT multiplied here (hence no _resolve_inner).
        iters = 1
        if match.enclosing_loops:
            for (_, _, ub_text, _) in match.enclosing_loops[:-1]:
                ub = resolve_bound_text(ub_text, mapping_env=menv)
                if ub is not None:
                    iters *= ub
                else:
                    # Tier-3: an outer bound we can't resolve. Apply the
                    # declared default and flag the estimate coarse rather
                    # than silently dropping the loop (design 04 §3.2).
                    d = model.dynamic_trip_default(match.target_op_name)
                    iters *= d if isinstance(d, int) else 1
                    dynamic = True
        exec_ops += per_op * iters
    # A1 (design 07 §A1.5/§A1.7): the vr_dma move cost lives on Resource.DMA.
    # The move arithmetic stays inside the breakdown helper; the combiner sees
    # only the resolved Phase. faithful: total = exec_ops + move_total.
    n_moves, per_move = _apu_v1_move_breakdown(model, target, trace, layout)
    move_total = n_moves * per_move
    total = exec_ops + move_total
    phases = [_exec_phase(exec_ops)]
    if move_total:
        double_buffered = bool(getattr(layout, "extra", {}).get("double_buffer", False))
        if overlap and double_buffered and n_moves > 0:
            # Loop-encoded: fill = per_move, steady state hides (D1.2/D1.4).
            phases.append(
                Phase(Resource.DMA, latency=per_move, ii=per_move,
                      count=n_moves, tag="vr_dma")
            )
        else:
            # Collapsed: fill = move_total (no hiding). Both the faithful arm
            # and a serialized overlap candidate take this branch.
            phases.append(_dma_phase(move_total, "vr_dma"))
    if dynamic:
        phases.append(_dynamic_marker())
    return CostResult(cycles=total, phases=phases,
                      confidence=_confidence(model, dynamic))


APU_V1_FAITHFUL = CostModel(
    name="apu_v1_faithful",
    target_name="apu_v1",
    op_costs={
        # 32K bit-serial lanes per VR: gvml_add_s16=2; gvml_mul_u16=16;
        # MAC (SV-lookup = gvml_lookup_16(6) + gvml_add_s16(2)) = 8. MICRO'25
        # Table 5 gvml per-op datasheet (A4). D3 (design 07 §D3): width-aware.
        # Anchored at 16 bits to today's constant; width-absent -> the constant
        # (byte-identical, design 07 §D3.2). seu_per_bit ratio MUL:ADD = 8:1
        # mirrors the table's mul>>add per-bit shape (mul_u16=201 vs add_u16=12).
        "ADD": OpCost(_apu_width_op(2, seu_per_bit=0.125),     # 16 bits -> 2
                      note="gvml_add_s16; width-aware (MICRO'25 Table5 add)",
                      provenance=Provenance.DATASHEET),
        "MUL": OpCost(_apu_width_op(16, seu_per_bit=1.0),      # 16 bits -> 16
                      note="gvml_mul_u16; width-aware (MICRO'25 Table5 mul)",
                      provenance=Provenance.DATASHEET),
        "MAC": OpCost(_apu_width_op(8, seu_per_bit=0.5),       # 16 bits -> 8
                      note="gvml_lookup_16 + gvml_add_s16; width-aware",
                      provenance=Provenance.DATASHEET),
    },
    move_costs={
        # report 12 §4.2: DMA L4<->L1 = 140 cyc/32K burst; LD/ST_VR = 5 cyc.
        # The report-12 moves are sim/HW-anchored (MEASURED).
        "DMA_L4_L1": MoveCost(lambda c: 140, note="L4<->L1 32K burst",
                              provenance=Provenance.MEASURED),
        "DMA_L1_L4": MoveCost(lambda c: 140, note="L4<->L1 32K burst",
                              provenance=Provenance.MEASURED),
        "LD_VR": MoveCost(lambda c: 5, note="gvml_load_16",
                          provenance=Provenance.MEASURED),
        "ST_VR": MoveCost(lambda c: 5, note="gvml_store_16",
                          provenance=Provenance.MEASURED),
    },
    constants={},
    compose=_apu_v1_compose,
    calibration=CalibrationRecord(
        validated_against="report-12 §4.2 GSI APU v1 (Gemini 1) SV / SV-lookup",
    ),
)


# APU v1 OVERLAP flavor (design 07 §D1): SAME structural target, SAME per-op /
# per-move numbers, SAME compose algebra -- the ONLY difference is the fold
# rule (`combine(overlap=True)`, bound via `_COMBINER_FOR_FLAVOR`) plus the
# DMA phase encoding the overlap compose picks for a double-buffered schedule.
# This is the report-27/28 keystone: overlap is a PROPERTY OF THE TIMELINE,
# not a parallel hand-coded flavor. `confidence="coarse"` -- the overlap fold
# over-counts partial overlap (D1.3 / T21), so it is honestly not calibrated.
APU_V1_OVERLAP = replace(
    APU_V1_FAITHFUL,
    name="apu_v1_overlap",
    flavor="overlap",
    compose=_apu_v1_overlap_compose,
    confidence="coarse",
    calibration=CalibrationRecord(),   # overlap fold is not sim-calibrated (D1.3)
)


def _apu_v1_double_buffer_knob_cost(value, ctx) -> "list[Phase]":
    """`double_buffer` knob's phase contribution (SPEC-023 D3).

    The knob's MARGINAL effect is the DMA phase ENCODING the overlap fold sees
    (the compute COMPUTE phase is the compose's, not re-emitted here, so no
    double-count):
      * depth 2 -> LOOP-ENCODED `Phase(DMA, latency=per_move, ii=per_move,
        count=n_moves)`: fill = per_move, steady state `per_move*(n-1)` hides
        behind compute under `combine(overlap=True)` / `_max_across_with_fill_drain`.
      * depth 1 (or no DMA) -> COLLAPSED `Phase(DMA, latency=move_total, ii=0,
        count=1)`: fill IS the whole DMA, so it does not hide (byte-identical).
    The magnitudes come from `_apu_v1_move_breakdown` (the SAME breakdown the
    overlap compose uses -> exact by construction). Empty when there is no DMA.
    """
    target = ctx.target
    layout = ctx.layout
    n_moves, per_move = _apu_v1_move_breakdown(APU_V1_OVERLAP, target, ctx.trace, layout)
    if not n_moves or per_move <= 0:
        return []
    if int(value) >= 2:
        return [Phase(Resource.DMA, latency=per_move, ii=per_move,
                      count=n_moves, tag="double_buffer_dma")]
    return [Phase(Resource.DMA, latency=n_moves * per_move, ii=0,
                  count=1, tag="double_buffer_dma")]


register_knob_cost(
    "apu_v1", "double_buffer", _apu_v1_double_buffer_knob_cost
)


# ===================================================================== #
# GSI APU v2 (placeholder)  -- l1_sim is functional-only (SPEC-011)
# ===================================================================== #


def _apu_v2_compose(ctx):
    # Non-comparative: l1_sim declares perf_is_placeholder. Return a
    # well-defined-but-not-predictive count so argmin is deterministic.
    n = len(ctx.trace.matches)
    return CostResult(cycles=n, phases=[_exec_phase(n)], confidence="placeholder")


APU_V2_PLACEHOLDER = CostModel(
    name="apu_v2_placeholder",
    target_name="apu_v2",
    op_costs={},
    move_costs={},
    constants={},
    compose=_apu_v2_compose,
    confidence="placeholder",
)


# ===================================================================== #
# Samsung optimistic (swap-test flavor) -- design 04 §1.4
# ===================================================================== #
#
# A second Samsung flavor that recalibrates per-op/per-move costs WITHOUT
# touching the structural target (the device tree is shared, unchanged).
# It proves the decoupling property the swap-test (task 008) gates on: a
# different cost table yields a different estimate with zero device edit.
# `compose` is reused verbatim (it reads costs off `SAMSUNG_OPTIMISTIC`
# via the module-level binding pattern below).


def _samsung_optimistic_compose(ctx):
    # Identical phase algebra to the faithful model; only the bound model
    # (hence the numbers) differs. Re-bind by swapping the closed-over
    # model reference through a thin wrapper.
    return _samsung_compose_with(SAMSUNG_OPTIMISTIC, ctx)


def _samsung_compose_with(model, ctx):
    """Samsung compose parameterised by an explicit `model` (so a second
    flavor reuses the exact phase algebra). Mirrors `_samsung_compose`."""
    target = ctx.target
    trace = ctx.trace
    layout = ctx.layout
    mac_cyc = model.op_cost("MAC", OpCostCtx("MAC"))
    jump_cyc = model.move_cost("JUMP", MoveCostCtx("JUMP"))
    lane_burst = target.grf_a.lanes
    n_workids = _samsung_workid_count(target)
    trigger_cyc = model.move_cost("CRF_TRIGGER", MoveCostCtx("CRF_TRIGGER"))
    grf_a = target.grf_a
    grf_b = target.grf_b

    def _preload_name(handle):
        if isinstance(handle, Register):
            if handle is grf_a:
                return "LD_A"
            if handle is grf_b:
                return "LD_B"
        return None

    body_cyc = 0
    dynamic = False
    residency = layout.extra.get("grf_residency", {})
    for match in trace.matches:
        handles = {
            opb.role: _unwrap(layout.placements.get(opb.memref_name))
            for opb in match.operands
        }
        host_only: dict[str, bool] = {}
        for opb in match.operands:
            if opb.role == "acc":
                continue
            load_name = _preload_name(handles.get(opb.role))
            if load_name is None:
                continue
            is_host = residency.get(opb.memref_name, "crf") == "host"
            if load_name not in host_only:
                host_only[load_name] = is_host
            else:
                host_only[load_name] = host_only[load_name] and is_host
        for load_name, dropped in host_only.items():
            if dropped:
                body_cyc -= model.move_cost(load_name, MoveCostCtx(load_name))
        inner_ub = _resolve_inner(target, match)
        if inner_ub is None:
            # Tier-3: genuinely dynamic inner bound. Apply the model's
            # DECLARED default (v1 = 1 MAC, the unfolded form) and flag the
            # whole estimate coarse -- never the silent =1 (design 04 §3.2).
            default, _ = _resolve_iters(model, target, match)
            body_cyc += default * mac_cyc
            dynamic = True
            continue
        y_handle = handles.get("y")
        is_auto = isinstance(y_handle, MemoryRef)
        if is_auto:
            folded = inner_ub // lane_burst
            n_fibers = layout.extra.get("n_fibers", 1)
            per_fiber = (folded + n_fibers - 1) // n_fibers
            body_cyc += per_fiber * mac_cyc + n_fibers * jump_cyc
        else:
            body_cyc += inner_ub * mac_cyc

    crf_issue = layout.extra.get("crf_issue", "per_workid")
    if crf_issue == "shared":
        exec_cyc = body_cyc + trigger_cyc * n_workids
    elif crf_issue == "per_workid":
        exec_cyc = body_cyc * n_workids
    else:
        raise ValueError(
            f"samsung kernel_cycles: unknown crf_issue {crf_issue!r}"
        )

    # kernel_cycles is now the DEVICE-exec body ONLY (task-017): preload /
    # readback re-homed into the host_staging concern, composed at the
    # whole-program seam (`_kernel_cycles_factory`:
    # whole-program = kernel_cycles + host_staging). The `weight_resident`
    # batching branch is deleted — preload-once vs preload-B is now the
    # host_staging compose's job (it reads `layout.extra["stage_resident"]`,
    # bridge option (b)). Exec is paid per batch vector.
    B = _trace_batch_dim(trace)
    cycles = B * exec_cyc
    # Collapsed COMPUTE phase carries the FULL device-exec total (B*exec_cyc)
    # so the serial-sum fold reproduces `cycles` byte-identically (design 07
    # §A1.6). The combiner sums phase_cycles; there is no separate B multiply.
    phases = [_exec_phase(cycles)]
    # D2 (design 07 §D2): layout-derived bank-locality penalty. None (no
    # phase) for the conflict-free corpus -> faithful number byte-identical.
    loc = _locality_phase(model, layout)
    if loc is not None:
        phases.append(loc)
        cycles += loc.latency
    if dynamic:
        phases.append(_dynamic_marker())
    return CostResult(
        cycles=cycles, phases=phases, confidence=_confidence(model, dynamic)
    )


SAMSUNG_OPTIMISTIC = CostModel(
    name="samsung_optimistic",
    target_name="samsung_hbm_pim",
    op_costs={
        # Optimistic re-calibration: MAC at half tCCDL (a faster column
        # strobe assumption). Same interface, different number.
        "MAC": OpCost(lambda c: 2, note="optimistic: tCCDL/2"),
        "MUL": OpCost(lambda c: 2, note="optimistic"),
    },
    move_costs={
        "LD_A": MoveCost(lambda c: 13),
        "LD_B": MoveCost(lambda c: 13),
        "ST_A": MoveCost(lambda c: 7),
        "ST_B": MoveCost(lambda c: 7),
        "JUMP": MoveCost(lambda c: 1),
        "CRF_TRIGGER": MoveCost(lambda c: 1),
        # PRELOAD_*/READBACK_* re-homed into the host_staging concern
        # (task-017); SAMSUNG_OPTIMISTIC_HOST_STAGING carries the optimistic
        # preload/readback re-calibration below.
    },
    constants={},
    compose=_samsung_optimistic_compose,
    flavor="optimistic",
    confidence="coarse",
)


# Optimistic host_staging flavor (so the virtual backend's whole-program
# estimate for cost_flavor="optimistic" stays consistent with the seam).
SAMSUNG_OPTIMISTIC_HOST_STAGING = register_cost_model(CostModel(
    name="samsung_optimistic_host_staging",
    target_name="samsung_hbm_pim",
    flavor="optimistic",
    concern="host_staging",
    op_costs={},
    move_costs={
        "STAGE_BCAST": MoveCost(lambda c: 738, note="optimistic: 2x fan-out"),
        "STAGE_SCATTER": MoveCost(lambda c: 1),
        "STAGE_CRF": MoveCost(lambda c: 2),
        "GATHER_FAN": MoveCost(lambda c: 4096),
        "GATHER_RD": MoveCost(lambda c: 90),
    },
    constants={},
    compose=_samsung_optimistic_host_staging_compose,
    confidence="coarse",
))


# ===================================================================== #
# Phase-5 no-backend demo substrate (design 04 §8) -- "a backend that
# ships with the spec." A bit-serial PIM substrate ("demo_pim") that has
# NO simulator and NO hardware: it is developable + costable purely via
# `compile_for_target(target, trace, backend="virtual")`. The structural
# target lives in `tests/spmw/_demo_target.py`; its CostModels live here.
#
# Two flavors share ONE compose, differing only by a single OpCost entry
# -- this is the canonical refinability showcase (design 04 §0.1/§1.3):
#   * demo_pim_constant : ADD is a constant `OpCost(lambda c: 2)`.
#   * demo_pim_micro25  : ADD is the MICRO-2025 per-op ANALYTICAL model
#     (a function of operand bit-width + the bit-serial lane geometry).
# Swapping one entry changes the estimate with NO edit to `compose` or any
# caller -- "an OpCost entry goes from constant to analytical, no caller
# change."
# ===================================================================== #

DEMO_PIM_LANES = 16384            # bit-serial lanes per demo VR (geometry)
DEMO_PIM_ELEM_BITS = 16           # u16 elements


def micro25_add_cost(ctx: OpCostCtx):
    """MICRO-2025-style per-op ANALYTICAL cost for a bit-serial s16 ADD.

    On a bit-serial substrate an add is computed one bit-plane at a time,
    so the SEU micro-op count scales with operand bit-width (carry-
    prediction trims the per-bit work to ~0.75 SEU/bit -- the APU
    Microbenchmarking Report's 12 SEU for s16 vs the naive 16). The
    measured steady-state cost also carries a fixed inter-fragment gap
    floor that dominates short calls (the report's 14.03 cyc/op for ADD
    despite 12 SEU). So:

        cyc = gap_floor + seu_per_bit * bit_width

    with `bit_width` read from `ctx` (operand element bits, falling back to
    the device lane element width). This is the SAME interface the constant
    entry satisfies -- `OpCost.fn(ctx) -> cycles` -- which is the whole
    point: refinement is a one-entry edit.
    """
    bits = None
    if ctx.operand_shapes:
        # operand_shapes carries (element_bits,) for this demo's ADD.
        first = ctx.operand_shapes[0]
        if isinstance(first, (tuple, list)) and first:
            bits = first[0]
        elif isinstance(first, int):
            bits = first
    if bits is None:
        bits = DEMO_PIM_ELEM_BITS
    gap_floor = 2.0          # inter-fragment minimum gap (report §"gap")
    seu_per_bit = 0.75       # carry-predicted SEU micro-ops per bit-plane
    return gap_floor + seu_per_bit * bits


def _demo_pim_compose(model):
    """Build a `compose` bound to `model`. The compose feeds each OpCost a
    RICH `OpCostCtx` (lane geometry + operand element bits) so an analytical
    entry has what it needs -- the constant entry simply ignores it. One
    compose, two flavors (design 04 §8)."""

    def compose(ctx):
        target = ctx.target
        trace = ctx.trace
        lane_width = getattr(target, "lanes_const", DEMO_PIM_LANES)
        elem_bits = getattr(target, "elem_bits_const", DEMO_PIM_ELEM_BITS)
        total = 0.0
        dynamic = False
        for match in trace.matches:
            name = match.target_op_name
            opctx = OpCostCtx(
                op_name=name,
                operand_shapes=((elem_bits,),),
                lane_width=lane_width,
            )
            if model.has_op_cost(name):
                per_op = model.op_cost(name, opctx)
            else:
                # Demo substrate: an unmodeled op falls back to the ADD entry
                # (the demo's intentional "everything reduces to a bit-serial
                # add" stance). This is NOT the A5 silent-default antipattern
                # -- it is an explicit, documented per-substrate policy on a
                # no-hardware demo, kept for the refinability showcase.
                per_op = model.op_cost("ADD", opctx)
            iters, dyn = _resolve_iters(model, target, match)
            dynamic = dynamic or dyn
            total += per_op * iters
        cycles = int(round(total))
        phases = [_exec_phase(cycles)]
        if dynamic:
            phases.append(_dynamic_marker())
        return CostResult(
            cycles=cycles, phases=phases,
            confidence=_confidence(model, dynamic),
        )

    return compose


DEMO_PIM_CONSTANT = CostModel(
    name="demo_pim_constant",
    target_name="demo_pim",
    op_costs={
        "ADD": OpCost(lambda c: 2, note="v1 placeholder constant"),
        "MUL": OpCost(lambda c: 16, note="v1 placeholder constant"),
        "MAC": OpCost(lambda c: 18, note="v1 placeholder constant"),
    },
    move_costs={},
    constants={},
    compose=None,          # set below (needs the model reference)
    flavor="constant",
    confidence="placeholder",
)
DEMO_PIM_CONSTANT = replace(
    DEMO_PIM_CONSTANT, compose=_demo_pim_compose(DEMO_PIM_CONSTANT)
)
# The substrate's default (`faithful`) flavor IS the constant placeholder,
# so a bare `compile_for_target(target, trace, backend="virtual")` (default
# cost_flavor="faithful") resolves -- the demo ships a usable default.
DEMO_PIM_DEFAULT = replace(DEMO_PIM_CONSTANT, flavor="faithful")


DEMO_PIM_MICRO25 = CostModel(
    name="demo_pim_micro25",
    target_name="demo_pim",
    op_costs={
        # The ONE refined entry: ADD is now an analytical function of the
        # bit-serial geometry, not a constant. MUL/MAC stay constant -- a
        # researcher refines ops one at a time.
        "ADD": OpCost(micro25_add_cost, note="MICRO-2025 bit-serial s16 ADD"),
        "MUL": OpCost(lambda c: 16, note="v1 placeholder constant"),
        "MAC": OpCost(lambda c: 18, note="v1 placeholder constant"),
    },
    move_costs={},
    constants={},
    compose=None,
    flavor="micro25",
    confidence="calibrated",
)
DEMO_PIM_MICRO25 = replace(
    DEMO_PIM_MICRO25, compose=_demo_pim_compose(DEMO_PIM_MICRO25)
)


# ===================================================================== #
# Mortise hypothetical PIM substrate (design 06; report-26) -- a
# capacity-aware near-bank-SIMD what-if grafted onto the Samsung exec
# algebra. The structural target lives in `tests/spmw/_mortise_target.py`;
# its CostModels live here. Three flavors on `(target_name="mortise")`:
#
#   * faithful    -- the baseline finding (the report-26 §2.1 capacity-aware
#                    host_staging compose: `P + B*(R + (1-phi)*P_var)`
#                    resident, `B*(P+R)` baseline; `phi=min(1,C/T_w)`).
#   * unlimited   -- the ABLATION (report-26 §3): hard-wires `phi=1`, so the
#                    `(1-phi)*P_var` capacity term vanishes for every `C` and
#                    every capacity arm collapses to the report-18 curve.
#   * optimistic  -- the swept sensitivity variant (report-26 §4): same
#                    compose, the §4 0.5x-2x band on the staging numbers.
#
# The capacity lever lives ENTIRELY in the host_staging compose
# (`B*(1-phi)*P_var`); `kernel_cycles` is the Samsung device-exec phase
# verbatim (design 06 §0, the one load-bearing decision). This makes the
# report-26 §5 anchor a NUMERICAL identity: at `phi=1` (C >= T_w) +
# resident, Mortise's whole-program is byte-identical to the
# Samsung-validated resident schedule (both = `P + B*(E+R)`).
#
# `unlimited` is carried as a CLOSURE-BAKED flag (design 06 §2.4
# alternative: `phi=1` set inside the bound compose), so NO field is added
# to the `CostModel` dataclass -- zero-FPGA-blast, spmw-local.
# ===================================================================== #


def _mortise_host_staging_compose_with(hs_model, ctx, *, unlimited: bool):
    """`host_staging`-concern compose for Mortise (design 06 §2.2).

    `_samsung_host_staging_compose_with` plus the report-26 §2.1 capacity
    term. The capacity `C` is read off the target tree const
    `resident_cap_elems` (a GEOMETRY constant, not a cost number); the
    *cycles* of re-streaming the evicted shortfall live here, in the cost
    model.

        phi = min(1, C / T_w)         (unlimited -> phi == 1)
        evict_per_call = (1 - phi) * P_var       P_var = P - crf_cyc

    resident=True : P paid ONCE + B * evict_per_call  (the lever earns this)
    resident=False: B * P                              (re-preload baseline)
    readback is always paid per batch vector.

    Reductions that must hold (report-26 §2.2):
      * phi=1, resident   -> P + B*R         (whole-program = report-18)
      * phi=1, non-resident -> B*(P+R)       (the re-preload comparator)
      * phi=0.5, resident -> per-call adds 0.5*P_var
    """
    target = ctx.target
    trace = ctx.trace
    layout = ctx.layout
    M, K = _samsung_mk(target, trace)
    B = _trace_batch_dim(trace)
    T_w = M * K                                  # weight-tile size (elements)

    # --- the capacity lever (the one new term over Samsung) ---
    C = getattr(target, "resident_cap_elems", T_w)
    if unlimited:
        phi = 1.0                                # ablation: capacity ignored
    else:
        phi = 1.0 if T_w <= 0 else min(1.0, C / T_w)

    preload_cyc = _samsung_preload_cycles(hs_model, M, K)       # = P
    readback_cyc = _samsung_readback_cycles(hs_model, M)        # = R per vector
    crf_cyc = hs_model.move_cost("STAGE_CRF", MoveCostCtx("STAGE_CRF"))
    P_var = preload_cyc - crf_cyc                # data-proportional part (§2.1)
    evict_per_call = int(round((1.0 - phi) * P_var))

    resident = bool(getattr(layout, "extra", {}).get("stage_resident", False))
    if resident:
        stage_resident = preload_cyc            # P paid ONCE
        stage_per_call = B * evict_per_call      # only the evicted shortfall/vec
    else:
        stage_resident = 0
        stage_per_call = B * preload_cyc         # re-preload-every-vector
    readback_total = B * readback_cyc
    cycles = stage_resident + stage_per_call + readback_total
    # The evicted-shortfall quantity (B*evict_per_call) IS `stage_per_call` in
    # the resident arm (design 06 §2.2); under the list carrier it is no
    # longer a redundant surfacing key -- the sweep reads it off the
    # `stage_per_call` HOST phase (which folds, unlike the old non-folded
    # `evict_per_call` alias).
    return CostResult(
        cycles=cycles,
        phases=[
            _host_phase(stage_resident, "stage_resident"),
            _host_phase(stage_per_call, "stage_per_call"),
            _host_phase(readback_total, "readback"),
        ],
        confidence=hs_model.confidence,
    )


def _mortise_host_staging_compose(ctx):
    return _mortise_host_staging_compose_with(
        MORTISE_HOST_STAGING, ctx, unlimited=False
    )


def _mortise_unlimited_host_staging_compose(ctx):
    return _mortise_host_staging_compose_with(
        MORTISE_UNLIMITED_HOST_STAGING, ctx, unlimited=True
    )


def _mortise_optimistic_host_staging_compose(ctx):
    return _mortise_host_staging_compose_with(
        MORTISE_OPTIMISTIC_HOST_STAGING, ctx, unlimited=False
    )


def _mortise_compose(ctx):
    # Device-exec phase = Samsung's exec verbatim (design 06 §2.1): Mortise
    # grafts onto the Samsung near-bank-SIMD substrate, so exec IS Samsung's.
    return _samsung_compose_with(MORTISE_FAITHFUL, ctx)


def _mortise_optimistic_compose(ctx):
    return _samsung_compose_with(MORTISE_OPTIMISTIC, ctx)


# --- kernel_cycles concern (shared exec algebra; Samsung-cloned numbers) ---
# Every constant carries a provenance tag (design 06 §2.3 table); none is a
# free fit param. Tags: [sim-anchored] / [assumption] / [structural].
MORTISE_FAITHFUL = CostModel(
    name="mortise_faithful",
    target_name="mortise",
    op_costs={
        "MAC": OpCost(lambda c: 4, note="[sim-anchored] tCCDL column-strobe; "
                      "Samsung-analog, report-18 E folded MAC=(K//8)*4"),
        "MUL": OpCost(lambda c: 4, note="[sim-anchored] tCCDL, Samsung-analog"),
    },
    move_costs={
        "LD_A": MoveCost(lambda c: 26, note="[sim-anchored] tCCDL+RL+BL//2"),
        "LD_B": MoveCost(lambda c: 26, note="[sim-anchored] tCCDL+RL+BL//2"),
        "ST_A": MoveCost(lambda c: 14, note="[sim-anchored] tCCDL+WL+BL//2"),
        "ST_B": MoveCost(lambda c: 14, note="[sim-anchored] tCCDL+WL+BL//2"),
        "JUMP": MoveCost(lambda c: 1, note="[structural] 1-cyc control op"),
        "CRF_TRIGGER": MoveCost(lambda c: 2,
            note="[assumption] host per-tile fire latency; Samsung-analog"),
    },
    constants={},
    compose=_mortise_compose,
    flavor="faithful",
)


# host_staging concern (faithful): the report-18 calibration anchors,
# re-homed VERBATIM from SAMSUNG_HOST_STAGING (design 06 §2.3). Every
# constant is provenance-tagged; `C` (the swept lever) lives on the tree,
# not here.
MORTISE_HOST_STAGING = register_cost_model(CostModel(
    name="mortise_host_staging",
    target_name="mortise",
    flavor="faithful",
    concern="host_staging",
    op_costs={},
    move_costs={
        "STAGE_BCAST": MoveCost(lambda c: 369, note="[sim-anchored] HAB "
            "preload fan-out width; report-18 §1 P=11368 on PIMSimulator; "
            "Mortise inherits Samsung HAB bcast rate. Uncertainty: Mortise "
            "bcast width may differ -> swept 0.5x-2x in optimistic flavor."),
        "STAGE_SCATTER": MoveCost(lambda c: 1, note="[sim-anchored] per-group "
            "column-strobe; report-18 P decomposition."),
        "STAGE_CRF": MoveCost(lambda c: 2, note="[assumption] programCrf "
            "upload; Samsung STAGE_CRF analog, capped 4-burst per report-18 "
            "§1; swept 1-8 in optimistic flavor."),
        "GATHER_FAN": MoveCost(lambda c: 4096, note="[sim-anchored] readback "
            "tile width; report-18 R=181 @ M=4096."),
        "GATHER_RD": MoveCost(lambda c: 181, note="[sim-anchored] per-tile "
            "readResult; report-18 §1 GATHER_RD=181."),
    },
    constants={},
    compose=_mortise_host_staging_compose,
))


# --- ablation flavor: unlimited (phi forced to 1; report-26 §3) ---
# Same kernel numbers, same staging numbers; the ONLY difference is the
# closure-baked `unlimited=True` in its host_staging compose, which kills
# the capacity term for every `C`.
MORTISE_UNLIMITED = replace(
    MORTISE_FAITHFUL, name="mortise_unlimited", flavor="unlimited",
    compose=lambda ctx: _samsung_compose_with(MORTISE_UNLIMITED, ctx),
)
MORTISE_UNLIMITED_HOST_STAGING = register_cost_model(CostModel(
    name="mortise_unlimited_host_staging",
    target_name="mortise",
    flavor="unlimited",
    concern="host_staging",
    op_costs={},
    move_costs=MORTISE_HOST_STAGING.move_costs,     # SAME faithful numbers
    constants={},
    compose=_mortise_unlimited_host_staging_compose,
))


# --- swept sensitivity flavor: optimistic (report-26 §4 band) ---
# Same compose; the §4 0.5x-2x band on the staging / exec numbers. Capacity
# term retained (this flavor still prices the lever; only the constants move).
MORTISE_OPTIMISTIC = CostModel(
    name="mortise_optimistic",
    target_name="mortise",
    op_costs={
        "MAC": OpCost(lambda c: 2, note="[sim-anchored] optimistic: tCCDL/2 "
                      "(report-26 §4 0.5x band on E)"),
        "MUL": OpCost(lambda c: 2, note="[sim-anchored] optimistic: tCCDL/2"),
    },
    move_costs={
        "LD_A": MoveCost(lambda c: 13, note="[sim-anchored] optimistic 0.5x"),
        "LD_B": MoveCost(lambda c: 13, note="[sim-anchored] optimistic 0.5x"),
        "ST_A": MoveCost(lambda c: 7, note="[sim-anchored] optimistic 0.5x"),
        "ST_B": MoveCost(lambda c: 7, note="[sim-anchored] optimistic 0.5x"),
        "JUMP": MoveCost(lambda c: 1, note="[structural] 1-cyc control op"),
        "CRF_TRIGGER": MoveCost(lambda c: 1, note="[assumption] optimistic"),
    },
    constants={},
    compose=_mortise_optimistic_compose,
    flavor="optimistic",
    confidence="coarse",
)
MORTISE_OPTIMISTIC_HOST_STAGING = register_cost_model(CostModel(
    name="mortise_optimistic_host_staging",
    target_name="mortise",
    flavor="optimistic",
    concern="host_staging",
    op_costs={},
    move_costs={
        "STAGE_BCAST": MoveCost(lambda c: 738, note="[sim-anchored] optimistic: "
            "2x fan-out (report-26 §4 2x band on P_var)"),
        "STAGE_SCATTER": MoveCost(lambda c: 1, note="[sim-anchored]"),
        "STAGE_CRF": MoveCost(lambda c: 4, note="[assumption] optimistic: "
            "4-burst crf (report-26 §4 crf 1-8 band)"),
        "GATHER_FAN": MoveCost(lambda c: 4096, note="[sim-anchored]"),
        "GATHER_RD": MoveCost(lambda c: 90, note="[sim-anchored] optimistic 0.5x R"),
    },
    constants={},
    compose=_mortise_optimistic_host_staging_compose,
    confidence="coarse",
))


# --- mortise_wide (SPEC-022 D3 proof: banks_per_pim==4) ---
# The wide-fiber proof target is Samsung-shaped, so its kernel_cycles model is
# Mortise's verbatim (same exec algebra, same numbers) -- only the bank fanout
# differs (a layout property the cost model reads via `n_fibers`, not a new
# number). Aliased off MORTISE_FAITHFUL so the wide-fiber candidate is rankable
# (the cost model can SCORE it -- the spec's "ranks the wide-fiber candidate"),
# proving a banks_per_pim>2 layout is no longer un-costable.
MORTISE_WIDE_FAITHFUL = replace(
    MORTISE_FAITHFUL, name="mortise_wide_faithful", target_name="mortise_wide",
    compose=lambda ctx: _samsung_compose_with(MORTISE_WIDE_FAITHFUL, ctx),
)


# --------------------------------------------------------------------- #
# Register all faithful models + the swap-test flavor + the Phase-5 demo
# at import (design 04 §1.5, §8).
# --------------------------------------------------------------------- #

for _m in (
    SAMSUNG_FAITHFUL,
    AIM_FAITHFUL,
    UPMEM_FAITHFUL,
    APU_V1_FAITHFUL,
    APU_V1_OVERLAP,
    APU_V2_PLACEHOLDER,
    SAMSUNG_OPTIMISTIC,
    DEMO_PIM_DEFAULT,
    DEMO_PIM_CONSTANT,
    DEMO_PIM_MICRO25,
    MORTISE_FAITHFUL,
    MORTISE_UNLIMITED,
    MORTISE_OPTIMISTIC,
    MORTISE_WIDE_FAITHFUL,
):
    register_cost_model(_m)
