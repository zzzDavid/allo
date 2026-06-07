# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Concrete `@allo.cost(...)` factories.

These run at import time to register cost callbacks with `spmw_cost`.
Importing this module is what makes `allo.get_cost("kernel_cycles", ...)`
return a working callback.
"""

from __future__ import annotations

import re

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


# Per-op / per-move cycle constants are NOT born here -- they live on
# each target's spec (Move.cycles / Op.cycles, set in the fixture). The
# cost factories below read those fields via `target.move(name).cycles`
# and `target.op(name).cycles`. See tests/spmw/_fixtures.py for the
# Samsung / AiM / UPMEM / APU v1 / APU v2 numbers and their citations.


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
    """Samsung HBM-PIM kernel-cycle estimator.

    Per-op cycles come from `target.op("MAC").cycles` (tCCDL, 4 cyc).
    JUMP is `target.move("JUMP").cycles`. The inner-K fold groups MACs
    by GRF lane count (`target.grf_a.lanes`) when src1 is bank-shaped.
    """
    mac_cyc = target.op("MAC").cycles
    jump_cyc = target.move("JUMP").cycles
    lane_burst = target.grf_a.lanes

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
                total += mac_cyc
                continue

            # is_auto is set when src1 is bank-shaped — the bank read
            # column-strobes through one inner-loop iteration per cycle,
            # collapsing K MACs into (K // lanes) MACs + 1 JUMP.
            y_handle = handles.get("y")
            is_auto = isinstance(y_handle, MemoryRef)
            if is_auto:
                folded = inner_ub // lane_burst
                # Dual-fiber placements run their bank halves concurrently:
                # the modelled MAC time is the busier fiber's share, not the
                # sum. `n_fibers` comes from the layout's segment-axis size
                # (filled by the enumerator from the swizzle), defaulting to
                # 1 so bank_row/grf_staged keep today's cost byte-identical.
                n_fibers = layout.extra.get("n_fibers", 1)
                # Ceil-split so an odd fold still bounds the busier fiber;
                # one JUMP folds each fiber's own inner loop.
                per_fiber = (folded + n_fibers - 1) // n_fibers
                total += per_fiber * mac_cyc + n_fibers * jump_cyc
            else:
                total += inner_ub * mac_cyc

        return total

    return cost_fn


def _aim_kernel_cycles(target):
    def cost_fn(trace: MatchTrace, layout) -> int:
        total = 0
        for match in trace.matches:
            try:
                per_op = target.op(match.target_op_name).cycles
            except KeyError:
                per_op = 4
            if per_op is None:
                per_op = 4
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
    the last enclosing loop) contribute multiplicatively.

    MAC pricing branches on `layout.mode`:
      * `"sv"`        -> raw MUL+ADD per MAC
      * `"sv_lookup"` -> gvml_lookup_16 + gvml_add_s16 (= MAC.cycles)
      * `""` (back-compat) -> SV-lookup expansion.
    Unknown modes raise. Argmin picks SV-lookup (cheaper).
    """
    _ALLOWED_MODES = {"", "sv", "sv_lookup"}
    add_cyc = target.op("ADD").cycles
    mul_cyc = target.op("MUL").cycles
    mac_cyc = target.op("MAC").cycles
    sv_mac_cyc = mul_cyc + add_cyc  # raw MUL+ADD

    def cost_fn(trace: MatchTrace, layout) -> int:
        mode = getattr(layout, "mode", "")
        if mode not in _ALLOWED_MODES:
            raise ValueError(
                f"apu_v1 kernel_cycles: unknown layout.mode {mode!r}; "
                f"allowed: {sorted(_ALLOWED_MODES)}"
            )
        sv_raw = mode == "sv"
        total = 0
        for match in trace.matches:
            if match.target_op_name == "MAC" and sv_raw:
                per_op = sv_mac_cyc
            else:
                try:
                    per_op = target.op(match.target_op_name).cycles
                except KeyError:
                    per_op = add_cyc
                if per_op is None:
                    per_op = add_cyc
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

    Per-op cost comes from `target.op(name).cycles`; MAC is priced as
    2 cycles because the DPU has no fused MAC. Bulk MRAM<->WRAM
    transfers are scheduled by a later pass and are not counted here.
    """
    gpr_cyc = target.op("ADD").cycles  # 1 cyc GPR baseline

    def cost_fn(trace: MatchTrace, layout) -> int:
        total = 0
        for match in trace.matches:
            try:
                per_op = target.op(match.target_op_name).cycles
            except KeyError:
                per_op = gpr_cyc
            if per_op is None:
                per_op = gpr_cyc
            inner_ub = None
            if match.enclosing_loops:
                inner_ub = _parse_loop_bound(match.enclosing_loops[-1][2])
            iters = inner_ub if inner_ub is not None else 1
            total += per_op * iters
        return total

    return cost_fn


# GSI APU v2 (Gemini 2). The only available simulator on this server is
# `l1_sim`, which declares `perf_is_placeholder = True` (skill file
# §Limitations). APU v2 is therefore positioned as a functional-only
# target; cost-based scheduling is out of scope until GTML grows cycle
# counters. See SPEC-011 for the decision rationale and the option-a
# upgrade slot.


def _apu_v2_kernel_cycles(target):
    """Functional-only cost stub for APU v2.

    APU v2 ships as a functional-correctness target: the only available
    simulator (`l1_sim`) declares `perf_is_placeholder = True` and does
    not report cycles. The autoscheduler still runs argmin on APU v2
    candidates (so the autoschedule path stays uniform across backends),
    but the ranking is a deterministic enumerator-order tie-break, not
    a performance prediction. This stub returns `len(trace.matches)` so
    argmin is well-defined; it is intentionally non-comparative across
    placements. See SPEC-011 for the rationale and the upgrade slot.
    """

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

    Numbers from spec 015 §6.1, declared on the target spec (LD_A/LD_B
    cycles 26 from tCCDL+RL+BL//2; ST_A/ST_B cycles 14 from tCCDL+WL+BL//2).
    """

    def spill(reg, n_entries=1):
        side = "A" if reg is target.grf_a else "B"
        return (
            target.move(f"ST_{side}").cycles
            + target.move(f"LD_{side}").cycles
        ) * n_entries

    return spill


def _aim_register_spill(target):
    """AiM: GPR <-> bank-row round-trip.

    Numbers from JSSC 2023 §IV, declared on the target spec: RD_SBK = 24
    = tCCDL+RD+BURST; ST_SBK = 20 = tCCDL+WR+BURST. AiM emits no
    bank-row writeback opcode today; ST_SBK is a synthetic move on the
    fixture so the allocator's monotonic cost vector still has a number.
    """

    def spill(reg, n_entries=1):
        load_cyc = target.move("RD_SBK").cycles
        store_cyc = target.move("ST_SBK").cycles
        return (load_cyc + store_cyc) * n_entries

    return spill


def _upmem_register_spill(target):
    """UPMEM: WRAM <-> MRAM round-trip dominates spill cost.

    Per-burst (64 B) from uPIMulator HPCA 2024 Table 2, declared on the
    target spec. The cost vector asks for whichever tier matches via
    the `tier` kwarg (spec 015 §6.3).
    """

    def spill(reg, n_entries=1, tier="mram"):
        if tier == "wram":
            return (
                target.move("ST_WRAM").cycles + target.move("LD_WRAM").cycles
            ) * n_entries
        if tier == "mram":
            return (
                target.move("ST_MRAM").cycles + target.move("LD_MRAM").cycles
            ) * n_entries
        raise ValueError(f"upmem spill tier {tier!r}")

    return spill


def _apu_v1_register_spill(target):
    """APU v1: VR <-> L1 round-trip.

    Same SRAM fabric as the VRs, so spill cost is roughly 1x register
    access. Numbers from spec 015 §6.4 (~5 cycles each direction),
    declared on the target spec (LD_VR / ST_VR).
    """

    def spill(reg, n_entries=1):
        return (
            target.move("ST_VR").cycles + target.move("LD_VR").cycles
        ) * n_entries

    return spill


# Module-level flag so the placeholder warning fires once per process,
# matching the kernel_cycles APU v2 convention above.
def _apu_v2_register_spill(target):
    """Functional-only spill stub for APU v2.

    Mirrors `_apu_v2_kernel_cycles`: APU v2 is a functional-correctness
    target (`l1_sim` declares `perf_is_placeholder = True`), so this
    factory returns a non-comparative constant (`2 * n_entries`) and
    does not emit a runtime warning. The allocator's argmin still runs
    deterministically; the value is not performance-predictive. See
    SPEC-011 for the rationale and the option-a upgrade slot when GTML
    timing lands.
    """

    def spill(reg, n_entries=1):
        return 2 * n_entries

    return spill
