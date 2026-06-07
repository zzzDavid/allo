# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SPMW register / buffer allocator (spec 015, Task 016).

Runs after the per-backend candidate enumerator and before codegen
consumes a `Placement`. Turns "candidate placement (op-binding only)"
into "concrete placement (with spill decisions)" by:

  1. Extracting `LiveRange`s from a single autoschedule group's
     `list[MatchedOp]` -- one live range per distinct memref the group
     touches.
  2. Building a `CostVector` per live range -- the per-candidate access
     cost plus declared spill alternatives.
  3. Running a greedy (Chow-Hennessy cost-gap ordering) solver: most-
     constrained live range first, cheapest choice that still fits;
     fall back to a Spilled() if no register tier has room.

The PBQP-shaped solver (cost vectors per node, edge cost matrices,
branch-and-bound) is a documented future slot at `_solve_pbqp` -- the
data structures here (`LiveRange`, `CostVector`, `CapacityTable`) are
shared with that future solver.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from .spmw_autoschedule import Placement
from .spmw_cost import get_cost
from .spmw_match import MatchedOp
from .spmw_target import MemoryRef, Register


__all__ = [
    "LiveRange",
    "Spilled",
    "CostVector",
    "CapacityTable",
    "AllocResult",
    "allocate",
    "extract_live_ranges",
]


# --------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class LiveRange:
    """One value that must be placed on the target.

    See spec 015 §1.1.
    """

    memref_name: str
    role: str
    first_idx: int
    last_idx: int
    is_loop_carried: bool
    size_bytes: int | None = None


@dataclass(frozen=True)
class Spilled:
    """Wraps a placement handle to indicate the allocator spilled this
    live range to `tier`. Codegen unwraps `home_handle` and inserts the
    spill LD/ST round-trip on the move boundary.

    See spec 015 §1.2.
    """

    home_handle: Any
    tier: str


@dataclass
class CostVector:
    """Per-live-range cost of each candidate placement choice.

    `entries`: ordered dict {choice -> cycles}. A `choice` is either a
    raw target handle (no spill) or a `Spilled(home_handle, tier)`.
    """

    entries: dict[Any, int]

    def cheapest(self) -> tuple[Any, int]:
        """(choice, cycles) for the lowest-cost option."""
        return min(self.entries.items(), key=lambda kv: kv[1])

    def gap(self) -> int:
        """`cheapest_alt - cheapest`. 0 if only one option."""
        if len(self.entries) < 2:
            return 0
        sorted_costs = sorted(self.entries.values())
        return sorted_costs[1] - sorted_costs[0]


@dataclass
class CapacityTable:
    """Declared per-backend resource limits.

    See spec 015 §1.4. All three maps are handle-keyed via a stable
    handle-id (`_handle_key`).
    """

    slots: dict[Any, int] = field(default_factory=dict)
    bytes_cap: dict[Any, int] = field(default_factory=dict)
    unlimited: set[Any] = field(default_factory=set)


@dataclass
class AllocResult:
    placement: Placement
    total_cost: int
    spilled: list[LiveRange]


# --------------------------------------------------------------------- #
# Handle keying
# --------------------------------------------------------------------- #


def _handle_key(h: Any) -> Any:
    """Stable identity key for register / memory handles.

    Two `MemoryRef(banks, 2*pid)` constructed at different sites are
    distinct Python objects but reference the same hardware bank in the
    trace -- so we key on `(id(memory), repr(idx))`. Registers and other
    handles fall through to Python `id(...)`.
    """
    if isinstance(h, MemoryRef):
        return ("memref", id(h.memory), repr(h.idx))
    if isinstance(h, Spilled):
        return ("spilled", id(h.home_handle), h.tier)
    return ("obj", id(h))


# --------------------------------------------------------------------- #
# Live-range extraction
# --------------------------------------------------------------------- #


def _estimate_bytes(match: MatchedOp, opb) -> int | None:
    """Best-effort byte estimate from the match's enclosing loops.

    Today returns None (slot-count fallback) -- spec 015 §1.1 says the
    allocator treats memory-tier placements as slot-counted when
    `size_bytes` is unknown, which keeps existing traces unblocked.
    Future workloads with byte-budget pressure can tighten this.
    """
    return None


def extract_live_ranges(matches: list[MatchedOp]) -> list[LiveRange]:
    """One `LiveRange` per distinct `(memref_name, role)` in `matches`.

    Lifetime spans `[first match index, last match index]`. Reductions
    (`is_loop_carried`) extend to `len(matches) - 1`. Output is sorted
    by `(first_idx, memref_name)` for determinism.
    """
    by_key: dict[tuple[str, str], LiveRange] = {}
    for idx, match in enumerate(matches):
        for opb in match.operands:
            if opb.memref_name is None:
                continue
            key = (opb.memref_name, opb.role)
            existing = by_key.get(key)
            if existing is None:
                by_key[key] = LiveRange(
                    memref_name=opb.memref_name,
                    role=opb.role,
                    first_idx=idx,
                    last_idx=idx,
                    is_loop_carried=opb.is_loop_carried,
                    size_bytes=_estimate_bytes(match, opb),
                )
            else:
                by_key[key] = replace(
                    existing,
                    last_idx=max(existing.last_idx, idx),
                    is_loop_carried=existing.is_loop_carried
                    or opb.is_loop_carried,
                )
    n = len(matches)
    out: list[LiveRange] = []
    for lr in by_key.values():
        if lr.is_loop_carried and n > 0:
            lr = replace(lr, last_idx=n - 1)
        out.append(lr)
    out.sort(key=lambda lr: (lr.first_idx, lr.memref_name))
    return out


# --------------------------------------------------------------------- #
# Capacity tables per backend
# --------------------------------------------------------------------- #


def _samsung_capacity(target) -> CapacityTable:
    return CapacityTable(
        slots={
            _handle_key(target.grf_a): 8,
            _handle_key(target.grf_b): 8,
        },
        bytes_cap={},
        unlimited={("bank", id(target.banks))},
    )


def _aim_capacity(target) -> CapacityTable:
    return CapacityTable(
        slots={_handle_key(target.gpr): 31},
        bytes_cap={_handle_key(target.gb): 1024},
        unlimited={("bank", id(target.banks))},
    )


def _upmem_capacity(target) -> CapacityTable:
    return CapacityTable(
        slots={_handle_key(target.gprs): 24},
        bytes_cap={_handle_key(target.wram): 65536},
        unlimited={("mem", id(target.mram))},
    )


def _apu_v1_capacity(target) -> CapacityTable:
    return CapacityTable(
        slots={_handle_key(target.vrs): 16},
        bytes_cap={_handle_key(target.l1): 32768},
        unlimited={("mem", id(target.l4))},
    )


def _apu_v2_capacity(target) -> CapacityTable:
    return CapacityTable(
        slots={},
        bytes_cap={_handle_key(target.l1): 3072 * 65536},
        unlimited={("mem", id(target.l5))},
    )


_CAPACITY_BUILDERS: dict[str, Callable[[Any], CapacityTable]] = {
    "samsung_hbm_pim": _samsung_capacity,
    "aim": _aim_capacity,
    "upmem": _upmem_capacity,
    "apu_v1": _apu_v1_capacity,
    "apu_v2": _apu_v2_capacity,
}


_CAPACITY_CACHE: dict[int, CapacityTable] = {}


def _get_capacity(target) -> CapacityTable:
    key = id(target)
    cached = _CAPACITY_CACHE.get(key)
    if cached is not None:
        return cached
    name = getattr(target, "name", None)
    builder = _CAPACITY_BUILDERS.get(name)
    if builder is None:
        raise NotImplementedError(
            f"regalloc: no capacity table for target {name!r}"
        )
    cap = builder(target)
    _CAPACITY_CACHE[key] = cap
    return cap


# --------------------------------------------------------------------- #
# Spill-tier dispatch
# --------------------------------------------------------------------- #


def _samsung_spill_tier(target) -> tuple[str, Any]:
    return ("bank_row", target.banks)


def _aim_spill_tier(target) -> tuple[str, Any]:
    return ("bank_row", target.banks)


def _upmem_spill_tier(target) -> tuple[str, Any]:
    return ("mram", target.mram)


def _apu_v1_spill_tier(target) -> tuple[str, Any]:
    return ("l1", target.l1)


def _apu_v2_spill_tier(target) -> tuple[str, Any]:
    return ("l2", target.l5)


_SPILL_TIER_BUILDERS: dict[str, Callable[[Any], tuple[str, Any]]] = {
    "samsung_hbm_pim": _samsung_spill_tier,
    "aim": _aim_spill_tier,
    "upmem": _upmem_spill_tier,
    "apu_v1": _apu_v1_spill_tier,
    "apu_v2": _apu_v2_spill_tier,
}


# --------------------------------------------------------------------- #
# Cost vector construction
# --------------------------------------------------------------------- #


def _is_register_handle(h: Any) -> bool:
    return isinstance(h, Register)


def _pick_home(candidates: list[Any]) -> Any:
    """The first register-tier candidate the allocator would prefer to
    keep this live range in if there were room. Falls back to the first
    candidate when none are registers.
    """
    for h in candidates:
        if _is_register_handle(h):
            return h
    return candidates[0]


def _base_access_cost(target, lr: LiveRange, handle: Any) -> int:
    """Per-use access cost charged to a non-spilled candidate.

    Today: 0 for register-tier handles (already on the fast path); 0
    also for memory-tier handles (the kernel_cycles cost model already
    prices the per-iteration access). The allocator's contribution is
    the spill round-trip only -- non-spilled handles cost 0 here.
    """
    return 0


def build_cost_vector(
    target,
    lr: LiveRange,
    candidates: list[Any],
    spill_cb: Callable,
) -> CostVector:
    """Build a `CostVector` for one live range.

    `candidates` is the list of distinct handles ever assigned to
    `lr.memref_name` across the enumerator's candidate placements.
    `spill_cb` is the closure returned by `get_cost("register_spill",
    target)`.
    """
    entries: dict[Any, int] = {}
    for h in candidates:
        entries[h] = _base_access_cost(target, lr, h)

    name = getattr(target, "name", None)
    tier_builder = _SPILL_TIER_BUILDERS.get(name)
    if tier_builder is not None and candidates:
        tier_name, _tier_handle = tier_builder(target)
        n_uses = lr.last_idx - lr.first_idx + 1
        home = _pick_home(candidates)
        try:
            spill_cost = spill_cb(home, n_entries=n_uses)
        except TypeError:
            # Older signatures may not accept n_entries kwarg.
            spill_cost = spill_cb(home) * n_uses
        entries[Spilled(home_handle=home, tier=tier_name)] = int(spill_cost)
    return CostVector(entries)


def role_candidates(
    candidates: list[Placement],
    memref_name: str,
) -> list[Any]:
    """Distinct handles ever assigned to `memref_name` across the
    enumerator's candidate placements, preserving first-seen order.
    """
    seen: set = set()
    out: list[Any] = []
    for p in candidates:
        h = p.placements.get(memref_name)
        if h is None:
            continue
        k = _handle_key(h)
        if k in seen:
            continue
        seen.add(k)
        out.append(h)
    return out


# --------------------------------------------------------------------- #
# Greedy solver
# --------------------------------------------------------------------- #


def _intervals_overlap(a: LiveRange, b: LiveRange) -> bool:
    return not (a.last_idx < b.first_idx or b.last_idx < a.first_idx)


def _fits(
    choice: Any,
    lr: LiveRange,
    occupants: dict[Any, list[LiveRange]],
    cap: CapacityTable,
) -> bool:
    """True iff `choice` has room for `lr` given current occupants.

    Spilled choices always fit (the spill tier is unlimited by
    construction). Register-tier handles count overlapping occupants
    against `cap.slots`; memory-tier handles sum overlapping
    occupant byte sizes against `cap.bytes_cap`.

    Unknown `size_bytes` on a memory tier collapses to slot-counted
    behaviour (we never refuse a memory-tier placement for missing size
    info -- spec 015 §4).
    """
    if isinstance(choice, Spilled):
        return True
    key = _handle_key(choice)
    # Unlimited tiers.
    if isinstance(choice, MemoryRef):
        unlimited_key = ("bank", id(choice.memory))
        if unlimited_key in cap.unlimited:
            return True
        mem_key = ("mem", id(choice.memory))
        if mem_key in cap.unlimited:
            return True
    # Register slot accounting.
    if key in cap.slots:
        n_overlap = sum(
            1 for occ in occupants.get(key, []) if _intervals_overlap(occ, lr)
        )
        return n_overlap < cap.slots[key]
    # Byte budget accounting (memory tiers with bytes_cap).
    if isinstance(choice, MemoryRef):
        mem_handle = choice.memory
        mem_key = _handle_key(mem_handle)
        if mem_key in cap.bytes_cap:
            if lr.size_bytes is None:
                # Slot-counted fallback for unknown size on a bounded tier.
                return True
            total = lr.size_bytes
            for occ in occupants.get(mem_key, []):
                if _intervals_overlap(occ, lr):
                    total += occ.size_bytes or 0
            return total <= cap.bytes_cap[mem_key]
    # Default: any tier we don't recognise is treated as unlimited.
    return True


def _record_occupant(
    choice: Any,
    lr: LiveRange,
    occupants: dict[Any, list[LiveRange]],
) -> None:
    """Bookkeep `lr` against `choice`'s capacity key."""
    if isinstance(choice, Spilled):
        return
    if isinstance(choice, MemoryRef):
        mem_key = _handle_key(choice.memory)
        occupants[mem_key].append(lr)
        return
    occupants[_handle_key(choice)].append(lr)


def _solve(
    lrs: list[LiveRange],
    cost_vectors: dict[str, CostVector],
    cap: CapacityTable,
) -> AllocResult:
    """Greedy v1: sort by descending gap; cheapest-fits-first.

    PBQP slots in here -- replace `_solve` (same signature) with a
    branch-and-bound solver later. The data inputs are unchanged.
    """
    placements: dict[str, Any] = {}
    spilled: list[LiveRange] = []
    total = 0
    occupants: dict[Any, list[LiveRange]] = defaultdict(list)

    ordered = sorted(
        lrs,
        key=lambda lr: (-cost_vectors[lr.memref_name].gap(), lr.first_idx, lr.memref_name),
    )

    for lr in ordered:
        cv = cost_vectors[lr.memref_name]
        # Iterate choices cheapest-first; ties broken by insertion order
        # via Python's stable sort.
        ranked = sorted(cv.entries.items(), key=lambda kv: kv[1])
        chosen = None
        chosen_cost = 0
        for choice, cost in ranked:
            if _fits(choice, lr, occupants, cap):
                chosen = choice
                chosen_cost = cost
                break
        if chosen is None:
            raise RuntimeError(
                f"regalloc: no choice fits for {lr.memref_name!r}; "
                f"cost_vector={cv}"
            )
        placements[lr.memref_name] = chosen
        if isinstance(chosen, Spilled):
            spilled.append(lr)
        else:
            _record_occupant(chosen, lr, occupants)
        total += chosen_cost

    return AllocResult(
        placement=Placement(placements=placements),
        total_cost=total,
        spilled=spilled,
    )


# --------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------- #


def allocate(
    target,
    matches: list[MatchedOp],
    candidate: Placement,
    *,
    all_candidates: list[Placement] | None = None,
) -> AllocResult:
    """Run the allocator for one autoschedule group.

    `candidate` is the enumerator-chosen `Placement` whose
    per-memref handles seed the per-live-range candidate set. If
    `all_candidates` is supplied (the full enumerator output for this
    group), the allocator also considers handles assigned to the same
    memref by *other* candidates -- this is what lets it pick a
    different register file under pressure.

    Returns an `AllocResult` whose `placement` may contain `Spilled`
    wrappers for live ranges that did not fit any unspilled tier.
    """
    lrs = extract_live_ranges(matches)
    cap = _get_capacity(target)

    # Per-live-range candidate handles. For the simple case (one
    # enumerator candidate), this is just `[candidate.placements[name]]`.
    pool: list[Placement] = (
        list(all_candidates) if all_candidates is not None else [candidate]
    )

    spill_cb = get_cost("register_spill", target)
    cost_vectors: dict[str, CostVector] = {}
    for lr in lrs:
        cands = role_candidates(pool, lr.memref_name)
        if not cands:
            # No candidate handle for this memref -- preserve whatever
            # the seed placement had (None falls through to codegen's
            # missing-placement error, matching today's behaviour).
            cands = []
            seed = candidate.placements.get(lr.memref_name)
            if seed is not None:
                cands = [seed]
        cost_vectors[lr.memref_name] = build_cost_vector(
            target, lr, cands, spill_cb
        )

    result = _solve(lrs, cost_vectors, cap)
    # Preserve any memrefs the live-range extractor did not see (e.g. a
    # result memref that never appears as an operand). Fall back to the
    # seed candidate's placement for those.
    for name, h in candidate.placements.items():
        if name not in result.placement.placements:
            result.placement.placements[name] = h
    # Preserve the enumerator's `mode` / `extra` on the refined placement
    # (SPEC-009 §0). `_solve` constructs a fresh `Placement` from
    # `placements` only; without this round-trip the codegen-side
    # mode-dispatch (APU v1 sv vs sv_lookup) would lose the signal.
    result.placement.mode = getattr(candidate, "mode", "")
    result.placement.extra = dict(getattr(candidate, "extra", {}))
    return result
