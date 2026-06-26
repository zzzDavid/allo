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
    CostModel,
    MoveCostCtx,
    OpCost,
    OpCostCtx,
    MoveCost,
    CostResult,
    register_cost_model,
)
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


# ===================================================================== #
# Samsung HBM-PIM (faithful)  -- design 04 §1.2/§1.5, SPEC-026
# ===================================================================== #


def _samsung_workid_count(target) -> int:
    n = 1
    for u in target._walk():
        for f in u.mapping:
            n *= f
    return n


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


def _samsung_preload_cycles(model, M: int, K: int) -> int:
    if M <= 0 or K <= 0:
        return 0
    write_width = model.move_cost("PRELOAD_FAN", MoveCostCtx("PRELOAD_FAN"))
    write_cyc = model.move_cost("PRELOAD_WR", MoveCostCtx("PRELOAD_WR"))
    crf_cyc = model.move_cost("PRELOAD_CRF", MoveCostCtx("PRELOAD_CRF"))
    return (M * K // write_width) * write_cyc + crf_cyc


def _samsung_readback_cycles(model, M: int) -> int:
    if M <= 0:
        return 0
    read_width = model.move_cost("READBACK_FAN", MoveCostCtx("READBACK_FAN"))
    read_cyc = model.move_cost("READBACK_RD", MoveCostCtx("READBACK_RD"))
    return ((M + read_width - 1) // read_width) * read_cyc


def _samsung_compose(ctx):
    # Single phase-algebra implementation, parameterised by the bound model
    # (see `_samsung_compose_with`) so the faithful + optimistic flavors
    # never diverge.
    return _samsung_compose_with(SAMSUNG_FAITHFUL, ctx)


SAMSUNG_FAITHFUL = CostModel(
    name="samsung_faithful",
    target_name="samsung_hbm_pim",
    op_costs={
        # tCCDL = 4 (column-strobe period). MUL/MAC both 4 cyc.
        "MAC": OpCost(lambda c: 4, note="tCCDL, Samsung ISCA'21 §4.1"),
        "MUL": OpCost(lambda c: 4, note="tCCDL"),
    },
    move_costs={
        # spec 015 §6.1: load = tCCDL+RL+BL//2 = 26; store = tCCDL+WL+BL//2 = 14
        "LD_A": MoveCost(lambda c: 26, note="tCCDL+RL+BL//2"),
        "LD_B": MoveCost(lambda c: 26, note="tCCDL+RL+BL//2"),
        "ST_A": MoveCost(lambda c: 14, note="tCCDL+WL+BL//2"),
        "ST_B": MoveCost(lambda c: 14, note="tCCDL+WL+BL//2"),
        "JUMP": MoveCost(lambda c: 1, note="1-cyc control op"),
        "CRF_TRIGGER": MoveCost(lambda c: 2, note="host per-tile fire latency"),
        # SPEC-026 §3.2/§3.3 calibration anchors (report 18 §3). PRELOAD_FAN
        # / READBACK_FAN are fan-out WIDTHS, not cycle counts -- they ride
        # MoveCost so a pessimistic flavor can recalibrate them without
        # editing the device tree (design 04 §1.2 edge case).
        "PRELOAD_FAN": MoveCost(lambda c: 369, note="HAB preload fan-out width"),
        "PRELOAD_WR": MoveCost(lambda c: 1, note="per-group column-strobe"),
        "PRELOAD_CRF": MoveCost(lambda c: 2, note="programCrf upload"),
        "READBACK_FAN": MoveCost(lambda c: 4096, note="readback tile width"),
        "READBACK_RD": MoveCost(lambda c: 181, note="per-tile readResult"),
    },
    constants={},
    compose=_samsung_compose,
)


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
        if model.has_op_cost(match.target_op_name):
            per_op = model.op_cost(
                match.target_op_name, OpCostCtx(match.target_op_name)
            )
        else:
            per_op = 4
        if per_op is None:
            per_op = 4
        iters, dyn = _resolve_iters(model, target, match)
        dynamic = dynamic or dyn
        total += per_op * iters
    phases = {"exec": total}
    if dynamic:
        phases["dynamic_assumed"] = 1
    return CostResult(cycles=total, phases=phases,
                      confidence=_confidence(model, dynamic))


AIM_FAITHFUL = CostModel(
    name="aim_faithful",
    target_name="aim",
    op_costs={
        # JSSC 2023 §IV: EWMUL=EWADD=4; MAC_SBK=8; MAC_ABK=16; AF=6.
        "MUL": OpCost(lambda c: 4, note="EWMUL"),
        "ADD": OpCost(lambda c: 4, note="EWADD"),
        "MAC": OpCost(lambda c: 8, note="MAC_SBK = 1 burst x 16 lanes"),
        "MAC_ABK": OpCost(lambda c: 16, note="all-bank broadcast = 2x"),
        "AF": OpCost(lambda c: 6, note="GELU/SIGMOID"),
    },
    move_costs={
        # spill model: RD_SBK = tCCDL+RD+BURST = 24; ST_SBK = tCCDL+WR+BURST = 20
        "RD_SBK": MoveCost(lambda c: 24, note="tCCDL+RD+BURST"),
        "ST_SBK": MoveCost(lambda c: 20, note="tCCDL+WR+BURST"),
    },
    constants={},
    compose=_aim_compose,
)


# ===================================================================== #
# UPMEM DPU (faithful)  -- uPIMulator / HPCA 2024 Table 2, research-021
# ===================================================================== #


def _upmem_compose(ctx):
    target = ctx.target
    trace = ctx.trace
    layout = ctx.layout
    model = UPMEM_FAITHFUL
    gpr_cyc = model.op_cost("ADD", OpCostCtx("ADD"))  # 1 cyc GPR baseline
    s = 0
    dynamic = False
    for match in trace.matches:
        if model.has_op_cost(match.target_op_name):
            per_op = model.op_cost(
                match.target_op_name, OpCostCtx(match.target_op_name)
            )
        else:
            per_op = gpr_cyc
        if per_op is None:
            per_op = gpr_cyc
        iters, dyn = _resolve_iters(model, target, match)
        dynamic = dynamic or dyn
        s += per_op * iters
    r = model.const("revolver_latency")
    t = layout.extra.get("n_tasklets", 1)
    cycles = s + math.ceil(s * (r - 1) / min(t, r))
    phases = {"exec": cycles}
    if dynamic:
        phases["dynamic_assumed"] = 1
    return CostResult(cycles=cycles, phases=phases,
                      confidence=_confidence(model, dynamic))


UPMEM_FAITHFUL = CostModel(
    name="upmem_faithful",
    target_name="upmem",
    op_costs={
        # DPU GPR ops issue at 1 cyc; MAC = mul+add = 2 cyc (no fused MAC).
        "MUL": OpCost(lambda c: 1, note="GPR op"),
        "ADD": OpCost(lambda c: 1, note="GPR op"),
        "MAC": OpCost(lambda c: 2, note="mul+add, no fused MAC"),
    },
    move_costs={
        # HPCA 2024 Table 2: MRAM = 1000 cyc/64B burst; WRAM = 1 cyc.
        "LD_MRAM": MoveCost(lambda c: 1000, note="MRAM read per 64B burst"),
        "ST_MRAM": MoveCost(lambda c: 1000, note="MRAM write per 64B burst"),
        "LD_WRAM": MoveCost(lambda c: 1, note="WRAM->GPR fused"),
        "ST_WRAM": MoveCost(lambda c: 1, note="GPR->WRAM fused"),
    },
    constants={
        # revolver scheduling window (uPIMulator src/main.go:115). NOT the
        # 14-stage pipeline depth (research-021).
        "revolver_latency": 11,
    },
    compose=_upmem_compose,
)


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


def _apu_v1_move_cycles(model, target, trace: MatchTrace, layout) -> int:
    vr_dma = getattr(layout, "extra", {}).get("vr_dma")
    if vr_dma is None:
        return 0
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
    return n_moves * per_move


_APU_V1_ALLOWED_MODES = {"", "sv", "sv_lookup"}


def _apu_v1_compose(ctx):
    target = ctx.target
    trace = ctx.trace
    layout = ctx.layout
    model = APU_V1_FAITHFUL
    menv = _mapping_env(target)
    add_cyc = model.op_cost("ADD", OpCostCtx("ADD"))
    mul_cyc = model.op_cost("MUL", OpCostCtx("MUL"))
    sv_mac_cyc = mul_cyc + add_cyc  # raw MUL+ADD

    mode = getattr(layout, "mode", "")
    if mode not in _APU_V1_ALLOWED_MODES:
        raise ValueError(
            f"apu_v1 kernel_cycles: unknown layout.mode {mode!r}; "
            f"allowed: {sorted(_APU_V1_ALLOWED_MODES)}"
        )
    sv_raw = mode == "sv"
    total = 0
    dynamic = False
    for match in trace.matches:
        if match.target_op_name == "MAC" and sv_raw:
            per_op = sv_mac_cyc
        else:
            if model.has_op_cost(match.target_op_name):
                per_op = model.op_cost(
                    match.target_op_name, OpCostCtx(match.target_op_name)
                )
            else:
                per_op = add_cyc
            if per_op is None:
                per_op = add_cyc
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
        total += per_op * iters
    total += _apu_v1_move_cycles(model, target, trace, layout)
    phases = {"exec": total}
    if dynamic:
        phases["dynamic_assumed"] = 1
    return CostResult(cycles=total, phases=phases,
                      confidence=_confidence(model, dynamic))


APU_V1_FAITHFUL = CostModel(
    name="apu_v1_faithful",
    target_name="apu_v1",
    op_costs={
        # 32K bit-serial lanes per VR: gvml_add_s16=2; gvml_mul_u16=16;
        # MAC (SV-lookup = gvml_lookup_16(6) + gvml_add_s16(2)) = 8.
        "ADD": OpCost(lambda c: 2, note="gvml_add_s16"),
        "MUL": OpCost(lambda c: 16, note="gvml_mul_u16"),
        "MAC": OpCost(lambda c: 8, note="gvml_lookup_16 + gvml_add_s16"),
    },
    move_costs={
        # report 12 §4.2: DMA L4<->L1 = 140 cyc/32K burst; LD/ST_VR = 5 cyc.
        "DMA_L4_L1": MoveCost(lambda c: 140, note="L4<->L1 32K burst"),
        "DMA_L1_L4": MoveCost(lambda c: 140, note="L4<->L1 32K burst"),
        "LD_VR": MoveCost(lambda c: 5, note="gvml_load_16"),
        "ST_VR": MoveCost(lambda c: 5, note="gvml_store_16"),
    },
    constants={},
    compose=_apu_v1_compose,
)


# ===================================================================== #
# GSI APU v2 (placeholder)  -- l1_sim is functional-only (SPEC-011)
# ===================================================================== #


def _apu_v2_compose(ctx):
    # Non-comparative: l1_sim declares perf_is_placeholder. Return a
    # well-defined-but-not-predictive count so argmin is deterministic.
    n = len(ctx.trace.matches)
    return CostResult(cycles=n, phases={"exec": n}, confidence="placeholder")


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

    M, K = _samsung_mk(target, trace)
    preload_cyc = _samsung_preload_cycles(model, M, K)
    readback_cyc = _samsung_readback_cycles(model, M)
    B = _trace_batch_dim(trace)
    weight_resident = layout.extra.get("weight_resident", False)
    if weight_resident:
        cycles = preload_cyc + B * (exec_cyc + readback_cyc)
    else:
        cycles = B * (preload_cyc + exec_cyc + readback_cyc)
    phases = {
        "preload": preload_cyc,
        "exec": exec_cyc,
        "readback": readback_cyc,
    }
    if dynamic:
        phases["dynamic_assumed"] = 1
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
        "PRELOAD_FAN": MoveCost(lambda c: 738, note="optimistic: 2x fan-out"),
        "PRELOAD_WR": MoveCost(lambda c: 1),
        "PRELOAD_CRF": MoveCost(lambda c: 2),
        "READBACK_FAN": MoveCost(lambda c: 4096),
        "READBACK_RD": MoveCost(lambda c: 90),
    },
    constants={},
    compose=_samsung_optimistic_compose,
    flavor="optimistic",
    confidence="coarse",
)


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
                per_op = model.op_cost("ADD", opctx)
            iters, dyn = _resolve_iters(model, target, match)
            dynamic = dynamic or dyn
            total += per_op * iters
        cycles = int(round(total))
        phases = {"exec": cycles}
        if dynamic:
            phases["dynamic_assumed"] = 1
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


# --------------------------------------------------------------------- #
# Register all faithful models + the swap-test flavor + the Phase-5 demo
# at import (design 04 §1.5, §8).
# --------------------------------------------------------------------- #

for _m in (
    SAMSUNG_FAITHFUL,
    AIM_FAITHFUL,
    UPMEM_FAITHFUL,
    APU_V1_FAITHFUL,
    APU_V2_PLACEHOLDER,
    SAMSUNG_OPTIMISTIC,
    DEMO_PIM_DEFAULT,
    DEMO_PIM_CONSTANT,
    DEMO_PIM_MICRO25,
):
    register_cost_model(_m)
