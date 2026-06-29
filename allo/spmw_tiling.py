# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capacity-bounded tile/fold generator for the SPMW autoscheduler (SPEC-023 D2).

`tile_candidates` emits a small set of LEGAL re-tilings of the nest the user
already wrote -- derived purely from the matched trace
(`MatchedOp.enclosing_loops` bounds + operand `indices`) and the target tree's
capacity, with NO `allo/ir/` edit and NO tile literal. This is the same
additive-trace discipline `_trace_reduction_trip` / `batch_dim` already prove:
the loop bounds give the iteration space, the operand indices give which axis
each operand strides on, and the target geometry (register lanes / scratchpad
bytes via the landed `_build_capacity` / `_estimate_bytes`) gives the capacity
that bounds the tile.

The ALWAYS-present first candidate is the IDENTITY tiling (tile_size = full
bound) == today's nest. Additional candidates are capacity-fitting divisors of
the bound, register-lane-aligned, STRICTLY smaller than the bound, and only
when a legal one exists AND it fits a bounded tier -- otherwise the
identity-only singleton (byte-identical, the regression default). The candidate
set is a function of `bound` and `target` geometry, never a literal.

This generalizes Mortise's single capacity constant `C` (`resident_cap_elems`)
into a swept, argmin-ranked tile -- the choice rides a typed `tile` `Knob`
(`spmw_knobs.py`); the sim-confirmed "retiling beats the user's nest" win is
the verifier's.

We deliberately emit a SMALL set of legal re-tilings of the user's nest, not
invented loop nests, and check legality structurally (a reduction axis can be
strip-mined/folded but not reordered across a carried dependence) -- NOT a full
affine-dependence analysis. PLuTo is the general polyhedral scheduler we do not
build; Timeloop/CoSA the capacity-bounded mapspace we bound the set against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TilePlan:
    """One legal re-tiling of the matched nest.

    `tiled_axis` is the enclosing-loop iter-var name being strip-mined;
    `tile_size` the per-tile trip (== `full_bound` for the identity tiling).
    `full_bound` is the user's original bound (so codegen knows the fold ratio
    and the cost knob the per-tile work). `is_identity` marks the byte-identical
    first candidate.
    """

    tiled_axis: str
    tile_size: int
    full_bound: int
    is_identity: bool = False


def _loop_bound(text) -> int | None:
    """Parse a single constant trip count out of an affine-bound string
    (mirrors `spmw_regalloc._loop_bound` / `_parse_loop_bound`)."""
    try:
        return int(str(text).strip())
    except (TypeError, ValueError):
        pass
    import re
    nums = re.findall(r"\b(\d+)\b", str(text))
    return int(nums[0]) if len(nums) == 1 else None


def _reducing_match(matches: list) -> Any:
    """The accumulating (reduction) match whose innermost loop is the tileable
    axis -- the same `MAC`-with-inner-loop the lane-burst fold and
    `_trace_reduction_trip` key on. None when no reducing match carries an
    inner loop."""
    for m in matches:
        if m.target_op_name == "MAC" and m.enclosing_loops:
            return m
    return None


def _lane_burst(target) -> int | None:
    """The register SIMD lane count the inner reduction loop folds into
    (`target.grf_a.lanes` for the near-bank-SIMD targets), or None. A legal
    tile must be a MULTIPLE of this (the fold granularity), so the lane burst
    is both the alignment and the smallest meaningful tile -- read off the
    tree, never a literal."""
    reg = getattr(target, "grf_a", None)
    lanes = getattr(reg, "lanes", None) if reg is not None else None
    return int(lanes) if lanes else None


def _capacity_tile_cap(target, match, opb) -> int | None:
    """The largest tile (in elements) whose working set fits a bounded tier,
    derived from the landed `_build_capacity` + `_estimate_bytes` (ONE capacity
    notion, shared with the allocator). Returns None when no byte cap is
    derivable (today's corpus carries no dtype -> unbounded -> identity-only,
    byte-identical). The element cap is the bounded-tier byte cap divided by the
    element width implied by the operand's footprint."""
    from .spmw_regalloc import _build_capacity, _estimate_bytes

    nbytes = _estimate_bytes(match, opb)
    if nbytes is None:
        return None  # no derivable footprint -> no capacity bound -> identity
    cap = _build_capacity(target)
    if not cap.bytes_cap:
        return None
    tier_bytes = min(cap.bytes_cap.values())
    # element width from the footprint / iteration count.
    n_elems = 1
    for loop in match.enclosing_loops or ():
        ub = _loop_bound(loop[2]) if len(loop) >= 3 else None
        if ub is None:
            return None
        n_elems *= ub
    if n_elems <= 0:
        return None
    elem_bytes = max(1, nbytes // n_elems)
    return max(1, tier_bytes // elem_bytes)


def tile_candidates(target, matches: list) -> "list[TilePlan]":
    """Legal capacity-bounded re-tilings of the matched nest (SPEC-023 D2).

    Returns the identity tiling first (byte-identical), then at most one
    capacity-fitting register-lane-aligned divisor of the inner reduction bound
    that is STRICTLY smaller than the bound -- only when one is both legal and
    fits a bounded tier. Otherwise the identity-only singleton. No tile literal:
    every number comes from the loop bound or the target geometry.
    """
    match = _reducing_match(matches)
    if match is None:
        return []  # no tileable nest -> caller keeps the un-tiled candidate

    inner = match.enclosing_loops[-1]
    axis = inner[0]
    bound = _loop_bound(inner[2]) if len(inner) >= 3 else None
    if bound is None or bound <= 1:
        return [TilePlan(axis, bound or 1, bound or 1, is_identity=True)]

    identity = TilePlan(axis, bound, bound, is_identity=True)
    plans = [identity]

    lanes = _lane_burst(target)
    if not lanes or lanes >= bound:
        return plans  # no lane granularity / already at floor -> identity only

    # Capacity element cap: the tile must fit a bounded tier. Use the reducing
    # match's accumulator/contraction operand footprint as the working set.
    cap_elems = None
    for opb in match.operands:
        c = _capacity_tile_cap(target, match, opb)
        if c is not None:
            cap_elems = c if cap_elems is None else min(cap_elems, c)
    if cap_elems is None:
        # No derivable capacity bound (today's corpus) -> identity only,
        # byte-identical. A workload that DOES carry a footprint gets the retile.
        return plans

    # The largest lane-aligned divisor of `bound` that is <= cap_elems and
    # strictly < bound. Search divisors derived from bound/lanes (no literal).
    max_tile = min(bound, cap_elems)
    # Round down to a lane multiple.
    cand = (max_tile // lanes) * lanes
    while cand >= lanes:
        if cand < bound and bound % cand == 0:
            plans.append(TilePlan(axis, cand, bound, is_identity=False))
            break
        cand -= lanes
    return plans
