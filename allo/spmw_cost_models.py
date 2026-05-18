# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Concrete `@allo.cost(...)` factories.

These run at import time to register cost callbacks with `spmw_cost`.
Importing this module is what makes `allo.get_cost("kernel_cycles", ...)`
return a working callback.
"""

from __future__ import annotations

import re
import warnings

from .spmw_cost import cost
from .spmw_match import MatchTrace
from .spmw_target import MemoryRef


def _unwrap(handle):
    """Strip a `Spilled` wrapper, returning the home handle. Cost models
    score the post-allocator placement: a spilled value still executes
    against its `home_handle`, so unwrapping is what gives the right
    is_auto / handle-type classification.
    """
    # Imported lazily to avoid a circular import (spmw_regalloc imports
    # `get_cost` from this module's neighbour).
    try:
        from .spmw_regalloc import Spilled
    except ImportError:
        return handle
    if isinstance(handle, Spilled):
        return handle.home_handle
    return handle


def _parse_loop_bound(text: str) -> int | None:
    """Best-effort integer extraction from an affine-map upper-bound
    string (`"1024"`, `"() -> (1024)"`, ...). Returns None on failure.
    """
    try:
        return int(text.strip())
    except ValueError:
        pass
    nums = re.findall(r"\b(\d+)\b", text)
    if len(nums) == 1:
        return int(nums[0])
    return None


# Samsung HBM2 timings (used as a coarse cost model only — not for
# cycle-accurate simulation). MAC takes one tCCDL column command; JUMP
# is treated as a 1-cycle control op; the inner-K reduction folds into
# `(K // GRF_LANES)` MACs when src1 is bank-shaped (is_auto = 1) and
# stays unrolled at K MACs otherwise.
_SAMSUNG_TCCDL = 4
_SAMSUNG_JUMP_CYCLES = 1
_SAMSUNG_LANE_BURST = 8


# AiM timing (from JSSC 2023 §IV, ramulator2 baseline). Coarse model —
# exact per-ISR timings live in the ramulator2 YAML; here we only need
# a monotonic objective for the autoscheduler's argmin.
_AIM_MAC_SBK_CYCLES = 8     # one MAC_SBK = 1 burst, 16 lanes
_AIM_MAC_ABK_CYCLES = 16    # all-bank broadcast: 2x latency
_AIM_EWMUL_CYCLES = 4
_AIM_EWADD_CYCLES = 4
_AIM_AF_CYCLES = 6          # GELU/SIGMOID: 6 cycles per 16 lanes

_AIM_OP_CYCLES = {
    "MUL": _AIM_EWMUL_CYCLES,
    "ADD": _AIM_EWADD_CYCLES,
    "MAC": _AIM_MAC_SBK_CYCLES,
    "MAC_ABK": _AIM_MAC_ABK_CYCLES,
    "AF": _AIM_AF_CYCLES,
}


# UPMEM DPU latency model (uPIMulator / HPCA 2024 Table 2 + Figure 8).
# MRAM read/write is the dominant cost; WRAM and GPR access are 1 cycle.
# Per-op cycles below price the compute kernel only; bulk-load moves
# are scheduled separately (Blocker 5).
_UPMEM_MRAM_READ_CYCLES_PER_64B = 1000
_UPMEM_MRAM_WRITE_CYCLES_PER_64B = 1000
_UPMEM_WRAM_ACCESS_CYCLES = 1
_UPMEM_GPR_OP_CYCLES = 1     # int add/mul: 1 cycle issue
_UPMEM_MAC_CYCLES = 2        # mul + add (no fused MAC on DPU)

_UPMEM_OP_CYCLES = {
    "MUL": _UPMEM_GPR_OP_CYCLES,
    "ADD": _UPMEM_GPR_OP_CYCLES,
    "MAC": _UPMEM_MAC_CYCLES,
}


# GSI APU v1 cycle estimates (from report 12 §4.2 + the pim-apu-v1 skill
# latency table). The 32K-lane bit-serial SIMD runs every VR op in
# lockstep across all lanes, so one GVML call covers the innermost
# K-dimension when the work-item tile fits within 32K lanes -- the
# innermost loop is folded into the SIMD width, not multiplied in.
_APU_V1_DMA_L4_L1_CYCLES = 140
_APU_V1_DMA_L1_L4_CYCLES = 140
_APU_V1_LD_VR_CYCLES = 4
_APU_V1_ST_VR_CYCLES = 4
_APU_V1_ADD_CYCLES = 2
_APU_V1_MUL_CYCLES = 16    # gvml_mul_u16 over 32K lanes
_APU_V1_LOOKUP_CYCLES = 6  # gvml_lookup_16 (sv-lookup body)
_APU_V1_MAC_CYCLES = _APU_V1_LOOKUP_CYCLES + _APU_V1_ADD_CYCLES

_APU_V1_OP_CYCLES = {
    "ADD": _APU_V1_ADD_CYCLES,
    "MUL": _APU_V1_MUL_CYCLES,
    "MAC": _APU_V1_MAC_CYCLES,
}


@cost("kernel_cycles")
def _kernel_cycles_factory(target):
    """Build a kernel-cycle estimator specialised to `target`.

    The callback takes `(trace, layout)` and returns a coarse cycle
    count for the compute portion of the trace under `layout`. It
    deliberately does not boot the simulator — the goal is a cheap,
    monotonic objective for the autoscheduler's argmin search.
    """
    target_name = getattr(target, "name", None)
    if target_name == "samsung_hbm_pim":
        return _samsung_kernel_cycles(target)
    if target_name == "aim":
        return _aim_kernel_cycles(target)
    if target_name == "upmem":
        return _upmem_kernel_cycles(target)
    if target_name == "apu_v1":
        return _apu_v1_kernel_cycles(target)
    if target_name == "apu_v2":
        return _apu_v2_kernel_cycles(target)
    raise NotImplementedError(
        f"kernel_cycles cost: no model for target {target_name!r}"
    )


def _samsung_kernel_cycles(target):
    def cost_fn(trace: MatchTrace, layout) -> int:
        total = 0
        for match in trace.matches:
            handles = {
                opb.role: _unwrap(layout.placements.get(opb.memref_name))
                for opb in match.operands
            }

            inner_ub = None
            if match.enclosing_loops:
                inner_ub = _parse_loop_bound(match.enclosing_loops[-1][2])
            if inner_ub is None:
                # Without an inner loop we can't bound the work; assume 1 op.
                total += _SAMSUNG_TCCDL
                continue

            # is_auto is set when src1 is bank-shaped — the bank read
            # column-strobes through one inner-loop iteration per cycle,
            # collapsing K MACs into (K // lanes) MACs + 1 JUMP.
            y_handle = handles.get("y")
            is_auto = isinstance(y_handle, MemoryRef)
            if is_auto:
                folded = inner_ub // _SAMSUNG_LANE_BURST
                total += folded * _SAMSUNG_TCCDL + _SAMSUNG_JUMP_CYCLES
            else:
                total += inner_ub * _SAMSUNG_TCCDL

        return total

    return cost_fn


def _aim_kernel_cycles(target):
    def cost_fn(trace: MatchTrace, layout) -> int:
        total = 0
        for match in trace.matches:
            per_op = _AIM_OP_CYCLES.get(match.target_op_name, 4)
            inner_ub = None
            if match.enclosing_loops:
                inner_ub = _parse_loop_bound(match.enclosing_loops[-1][2])
            iters = inner_ub if inner_ub is not None else 1
            total += per_op * iters
        return total

    return cost_fn


def _apu_v1_kernel_cycles(target):
    """Cycle estimator for the APU v1 (Gemini 1) compute kernel.

    The bit-serial SIMD width (32K lanes per VR) folds one work-item
    dimension into a single GVML call -- the innermost loop is therefore
    NOT multiplied into the per-op cost. Outer loops (everything but
    the last enclosing loop) contribute multiplicatively. MAC is priced
    as `gvml_lookup_16 + gvml_add_s16` (the SV-lookup expansion), which
    is the placement preferred over a raw MUL+ADD (SV mode) by argmin.
    """
    def cost_fn(trace: MatchTrace, layout) -> int:
        total = 0
        for match in trace.matches:
            per_op = _APU_V1_OP_CYCLES.get(
                match.target_op_name, _APU_V1_ADD_CYCLES
            )
            # Outer loops only -- the innermost loop is subsumed by the
            # 32K-lane SIMD width.
            iters = 1
            if match.enclosing_loops:
                for (_, _, ub_text, _) in match.enclosing_loops[:-1]:
                    ub = _parse_loop_bound(ub_text)
                    if ub is not None:
                        iters *= ub
            total += per_op * iters
        return total

    return cost_fn


def _upmem_kernel_cycles(target):
    """Cycle estimator for the UPMEM DPU compute kernel.

    Per-op cost comes from `_UPMEM_OP_CYCLES`; MAC is priced as 2 cycles
    because the DPU has no fused MAC. Bulk MRAM<->WRAM transfers are
    scheduled by a later pass and are not counted here.
    """
    def cost_fn(trace: MatchTrace, layout) -> int:
        total = 0
        for match in trace.matches:
            per_op = _UPMEM_OP_CYCLES.get(
                match.target_op_name, _UPMEM_GPR_OP_CYCLES
            )
            inner_ub = None
            if match.enclosing_loops:
                inner_ub = _parse_loop_bound(match.enclosing_loops[-1][2])
            iters = inner_ub if inner_ub is not None else 1
            total += per_op * iters
        return total

    return cost_fn


# GSI APU v2 (Gemini 2). The only available simulator on this server is
# `l1_sim`, which declares `perf_is_placeholder = True` (skill file
# §Limitations). The cost model therefore returns a *structural*
# constant -- enough to keep autoschedule's argmin deterministic, but
# not performance-predictive. A one-time warning fires at factory-build
# time (NOT inside cost_fn, to avoid log-flood during search).
_APU_V2_WARNED = False


def _apu_v2_kernel_cycles(target):
    """Placeholder cost for APU v2.

    Returns `len(trace.matches)` -- 1 cycle per match. The autoscheduler
    can still call this without crashing; argmin between candidate
    placements stays deterministic but is not performance-predictive.
    """
    global _APU_V2_WARNED
    if not _APU_V2_WARNED:
        warnings.warn(
            "apu_v2 kernel_cycles cost is a structural placeholder: "
            "l1_sim declares perf_is_placeholder = True. Cost is "
            "constant per match (1 cycle/op); autoscheduler argmin "
            "is order-dependent.",
            RuntimeWarning,
            stacklevel=2,
        )
        _APU_V2_WARNED = True

    def cost_fn(trace: MatchTrace, layout) -> int:
        return len(trace.matches)

    return cost_fn


# --------------------------------------------------------------------- #
# Register-spill cost factories (spec 015)
# --------------------------------------------------------------------- #
#
# Each factory runs once per target (via `get_cost`'s cache) and returns
# a closure `spill(reg, n_entries=1)` that returns the cycles charged
# for spilling a single live range to its declared spill tier over
# `n_entries` uses. The allocator builds a `Spilled(home, tier)`
# placement when no register-file tier has room; codegen reads the
# `tier` field to decide which LD/ST moves to inject.


@cost("register_spill")
def _register_spill_factory(target):
    name = getattr(target, "name", None)
    if name == "samsung_hbm_pim":
        return _samsung_register_spill(target)
    if name == "aim":
        return _aim_register_spill(target)
    if name == "upmem":
        return _upmem_register_spill(target)
    if name == "apu_v1":
        return _apu_v1_register_spill(target)
    if name == "apu_v2":
        return _apu_v2_register_spill(target)
    raise NotImplementedError(
        f"register_spill: no factory for target {name!r}"
    )


def _samsung_register_spill(target):
    """Samsung HBM-PIM: GRF_A/B <-> bank-row round-trip.

    Numbers from spec 015 §6.1: tCCDL=4, RL=20, WL=8, BL=4.
    load = tCCDL + RL + BL//2 = 26 cycles; store = tCCDL + WL + BL//2 = 14.
    """
    tCCDL, RL, WL, BL = 4, 20, 8, 4
    load_cyc = tCCDL + RL + BL // 2     # 26
    store_cyc = tCCDL + WL + BL // 2    # 14
    for mv in ("LD_A", "LD_B"):
        target.move(mv).cycles = load_cyc
    for mv in ("ST_A", "ST_B"):
        target.move(mv).cycles = store_cyc

    def spill(reg, n_entries=1):
        side = "A" if reg is target.grf_a else "B"
        return (
            target.move(f"ST_{side}").cycles
            + target.move(f"LD_{side}").cycles
        ) * n_entries

    return spill


def _aim_register_spill(target):
    """AiM: GPR <-> bank-row round-trip.

    Numbers from JSSC 2023 §IV: tCCDL=4, RD=16, WR=12, burst=4.
    load = 24, store = 20.

    AiM's existing fixture already declares RD_SBK (bank-row read) but
    not the symmetric write-back; the cost model uses the existing
    RD_SBK for loads and falls back to a fixed `store_cyc` constant for
    the writeback (the allocator only needs a monotonic value; no
    bank-row writeback opcode is emitted today).
    """
    tCCDL = 4
    RD, WR, BURST = 16, 12, 4
    load_cyc = tCCDL + RD + BURST   # 24
    store_cyc = tCCDL + WR + BURST  # 20
    # AiM's fixture declares RD_SBK (load from bank to gpr); use it for
    # the load side and keep the store side as a constant (no bank-row
    # writeback move is currently declared on AiM).
    try:
        target.move("RD_SBK").cycles = load_cyc
    except KeyError:
        pass

    def spill(reg, n_entries=1):
        return (load_cyc + store_cyc) * n_entries

    return spill


def _upmem_register_spill(target):
    """UPMEM: WRAM <-> MRAM round-trip dominates spill cost.

    Per-burst (64 B) from uPIMulator HPCA 2024 Table 2. The cost vector
    asks for whichever tier matches via the `tier` kwarg (spec 015 §6.3).
    """
    wram_load = 1
    wram_store = 1
    mram_burst = 1000
    for mv_name, cyc in (
        ("LD_WRAM", wram_load),
        ("ST_WRAM", wram_store),
        ("LD_MRAM", mram_burst),
        ("ST_MRAM", mram_burst),
    ):
        try:
            target.move(mv_name).cycles = cyc
        except KeyError:
            pass

    def spill(reg, n_entries=1, tier="mram"):
        if tier == "wram":
            return (wram_store + wram_load) * n_entries
        if tier == "mram":
            return (mram_burst + mram_burst) * n_entries
        raise ValueError(f"upmem spill tier {tier!r}")

    return spill


def _apu_v1_register_spill(target):
    """APU v1: VR <-> L1 round-trip.

    Same SRAM fabric as the VRs, so spill cost is roughly 1x register
    access. Numbers from spec 015 §6.4 (~5 cycles each direction).
    """
    load_cyc, store_cyc = 5, 5
    for mv_name, cyc in (("LD_VR", load_cyc), ("ST_VR", store_cyc)):
        try:
            target.move(mv_name).cycles = cyc
        except KeyError:
            pass

    def spill(reg, n_entries=1):
        return (store_cyc + load_cyc) * n_entries

    return spill


# Module-level flag so the placeholder warning fires once per process,
# matching the kernel_cycles APU v2 convention above.
_APU_V2_SPILL_WARNED = False


def _apu_v2_register_spill(target):
    """APU v2: L1 row <-> L2 row; l1_sim placeholder.

    l1_sim declares `perf_is_placeholder = True`; the allocator's
    argmin will still run deterministically but its cost is not
    performance-predictive. Constant `2 * n_entries`.
    """
    global _APU_V2_SPILL_WARNED
    if not _APU_V2_SPILL_WARNED:
        warnings.warn(
            "apu_v2 register_spill cost is a structural placeholder: "
            "l1_sim declares perf_is_placeholder = True.",
            RuntimeWarning,
            stacklevel=2,
        )
        _APU_V2_SPILL_WARNED = True

    def spill(reg, n_entries=1):
        return 2 * n_entries

    return spill
