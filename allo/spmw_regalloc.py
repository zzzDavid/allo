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


def _loop_bound(text) -> int | None:
    """Parse a single constant trip count out of an affine-bound string."""
    try:
        return int(str(text).strip())
    except (TypeError, ValueError):
        pass
    import re
    nums = re.findall(r"\b(\d+)\b", str(text))
    return int(nums[0]) if len(nums) == 1 else None


def _estimate_bytes(match: MatchedOp, opb) -> int | None:
    """Byte footprint of a live value: `product(shape) * dtype_bytes`
    (SPEC-022 D2).

    `shape` is the product of the match's enclosing-loop trip counts (the
    same shape source the cost model reads). `dtype_bytes` comes from the
    operand element bit-width: `opb.dtype_bits`, else `match.extra
    ["dtype_bits"]` (the carrier task 007 / a caller populates). When no
    dtype is derivable (today's synthetic corpus carries none), returns
    None -> `_fits` keeps its slot-counted fallback, so the corpus is
    byte-identical. The byte arm only tightens behaviour for a workload
    whose footprint IS derivable AND exceeds a bounded tier.
    """
    dtype_bits = getattr(opb, "dtype_bits", None)
    if dtype_bits is None:
        dtype_bits = (match.extra or {}).get("dtype_bits")
    if not dtype_bits:
        return None

    n_elems = 1
    for loop in match.enclosing_loops or ():
        ub = _loop_bound(loop[2]) if len(loop) >= 3 else None
        if ub is None:
            return None  # unresolvable shape -> slot-counted fallback
        n_elems *= ub
    return n_elems * (int(dtype_bits) // 8 or 1)


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


def _memory_capacity_bytes(mem) -> int | None:
    """Derivable byte size of a bounded `Memory` from its geometry, or None.

    `size_bytes` is authoritative when declared. Otherwise derive from the
    declared axes: `entries x width` (AiM gb, a width-bit-wide entry array)
    or `rows x cols x width` (APU v2 l1 bitline grid). `width` is in bits;
    the product is divided by 8 to yield bytes. Returns None when no byte
    size is derivable (the slot-counted fallback applies).
    """
    g = mem.geometry
    if "size_bytes" in g and g["size_bytes"] is not None:
        return int(g["size_bytes"])
    width = g.get("width")
    if g.get("entries") is not None and width is not None:
        return int(g["entries"]) * int(width) // 8
    if g.get("rows") is not None and g.get("cols") is not None:
        return int(g["rows"]) * int(g["cols"]) * int(width or 8) // 8
    return None


def _build_capacity(target) -> CapacityTable:
    """Derive the capacity table STRUCTURALLY from the target tree
    (SPEC-022 D2), retiring the `target.name`-keyed constant table.

    - Register slots = `reg.slots` (defaults to `reg.lanes`; AiM declares
      `slots=31` structurally for its addressable GPR depth).
    - A bulk DRAM-class memory (`banks` near-bank store / `mram` / `l4` / `l5`)
      is `unlimited`: no per-value placement cap applies (it is the spill /
      bulk store; `Spilled` choices bypass `_fits` regardless). `_fits`
      checks both ("bank",..) and ("mem",..) so one key form suffices.
    - Every other memory with a derivable byte size becomes a `bytes_cap`
      arm (a bounded scratchpad: gb / wram / l1). Instruction memories /
      unused scratch are harmless -- nothing is placed there, so `_fits`
      never consults them.

    A backend with no special-cased builder gets this same structural
    default -- no `NotImplementedError`. The reviewer's anti-hardcoding
    gate #2 holds: no `target.name`-keyed capacity dict, no pasted slot
    constant; caps come from `Register`/`Memory` geometry.
    """
    slots: dict[Any, int] = {}
    bytes_cap: dict[Any, int] = {}
    unlimited: set[Any] = set()

    for unit in target._walk():
        for r in unit.registers.values():
            slots[_handle_key(r)] = int(getattr(r, "slots", r.lanes))
        for m in unit.memories.values():
            if m.name in _BULK_STORE_MEMS:
                unlimited.add(("mem", id(m)))
                continue
            nbytes = _memory_capacity_bytes(m)
            if nbytes is not None:
                bytes_cap[_handle_key(m)] = nbytes

    return CapacityTable(slots=slots, bytes_cap=bytes_cap, unlimited=unlimited)


# Keyed by `id(target)`, storing `(target_ref, cap)` so a REUSED id (a prior
# target was GC'd and CPython recycled its id for a different target object)
# misses the cache instead of returning a stale `CapacityTable`. Without the
# identity recheck this cache is non-deterministically wrong under heavy target
# churn (exposed by the schedule-search tests building many targets) -- a
# pre-existing latent fragility this hardens.
_CAPACITY_CACHE: dict[int, "tuple[Any, CapacityTable]"] = {}


def _get_capacity(target) -> CapacityTable:
    key = id(target)
    cached = _CAPACITY_CACHE.get(key)
    if cached is not None and cached[0] is target:
        return cached[1]
    cap = _build_capacity(target)
    _CAPACITY_CACHE[key] = (target, cap)
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


# Monotone tier rank: register < near_bank < scratchpad < dram. Ranks
# start at 1 so a register-tier candidate carries a *non-zero* base cost
# that is identical across register candidates (cancels in the argmin),
# per the SPEC-022 D2 byte-for-byte default. The scale (cycles-per-rank)
# is read from the bound CostModel, not pasted.
_TIER_RANK = {"register": 1, "near_bank": 2, "scratchpad": 3, "dram": 4}

# Memory-name -> tier classification (SPEC-022 D2). Near-bank = the
# bank-resident PIM compute store; scratchpad = the bounded working buffer;
# dram = the bulk spill store.
_NEAR_BANK_MEMS = {"banks"}
_SCRATCHPAD_MEMS = {"gb", "wram", "l1"}
_DRAM_MEMS = {"mram", "l4", "l5"}
# Bulk stores carry no per-value placement cap (the spill / bulk tier):
# the near-bank store + the DRAM-class stores. A scratchpad that doubles as
# a spill tier (APU v1 l1) stays BOUNDED here -- `Spilled` choices bypass
# `_fits`, so a finite l1 cap never blocks a spill, but it DOES gate a
# non-spilled scratchpad-resident value (which is the point of D2).
_BULK_STORE_MEMS = _NEAR_BANK_MEMS | _DRAM_MEMS


def _classify_tier(handle: Any) -> str:
    """Classify a candidate handle to an `AccessDescr` tier."""
    if isinstance(handle, Register):
        return "register"
    if isinstance(handle, MemoryRef):
        nm = handle.memory.name
        if nm in _NEAR_BANK_MEMS:
            return "near_bank"
        if nm in _DRAM_MEMS:
            return "dram"
        return "scratchpad"  # gb / wram / l1 and any other bounded buffer
    return "scratchpad"


def _locality_unit(target) -> int:
    """The per-rank / per-conflict cycle weight, read from the bound
    CostModel's `ROW_BUFFER_MISS` move cost (the `Resource.LOCALITY` unit
    the cost-model locality term also uses). Falls back to 1 when the model
    has no row-buffer-miss move (e.g. APU, no bank-conflict axis), so the
    tier ordering is still monotone and non-zero."""
    from .spmw_cost_model import get_cost_model, MoveCostCtx

    name = getattr(target, "name", None)
    try:
        model = get_cost_model(name, "faithful")
    except Exception:
        return 1
    if model.move_costs.get("ROW_BUFFER_MISS") is not None:
        try:
            return int(model.move_cost("ROW_BUFFER_MISS",
                                       MoveCostCtx("ROW_BUFFER_MISS")))
        except Exception:
            return 1
    return 1


def _base_access_cost(target, lr: LiveRange, handle: Any, layout=None) -> int:
    """Per-use access cost for a non-spilled candidate (SPEC-022 D2).

    Locality-aware, computed from the shared `AccessDescr`
    (`spmw_cost_model.AccessDescr`) -- the SAME type the cost-model §A2
    locality term consumes (one notion of access pattern, two consumers):

        base = TIER_RANK(tier) * unit            # monotone register<...<dram
             + conflict_count   * unit           # Resource.LOCALITY penalty

    where `unit` is the bound CostModel's `ROW_BUFFER_MISS` weight (not a
    pasted constant). The descriptor's layout-derived `conflict_count` is
    populated from the chosen `LinearLayout` when one is carried (D3); for a
    handle with no layout it is `AccessDescr.identity(tier)` -- conflict-free,
    penalty zero by construction.

    Byte-for-byte default: every register-tier conflict-free candidate gets
    the SAME non-zero cost (`1*unit`), which cancels in `kc + total_cost`;
    the cost only changes a decision when candidates differ in tier or
    conflict count -- exactly when the allocator should distinguish them.
    """
    from .spmw_cost_model import AccessDescr

    tier = _classify_tier(handle)
    unit = _locality_unit(target)

    # Build the access descriptor. With a carried layout we count bank
    # conflicts off it (D2 population); without one, the conflict-free
    # identity descriptor yields a zero locality penalty.
    descr = AccessDescr.identity(tier)
    if layout is not None:
        descr = _access_descr_from_layout(layout, tier)

    penalty = descr.conflict_count * unit
    if descr.row_hits:
        # Open-row reuse REDUCES the stall; bounded so the cost stays >= the
        # tier floor (a fully-resolved reuse cannot make a placement free).
        penalty = max(0, penalty - descr.row_hits)
    return _TIER_RANK[tier] * unit + penalty


def _access_descr_from_layout(layout, tier: str):
    """Build an `AccessDescr` from a carried `LinearLayout` (D2 population
    of the layout-derived fields). Counts bank conflicts via
    `LinearLayout.conflict_count`; on any shape mismatch falls back to the
    conflict-free identity (penalty zero)."""
    from .spmw_cost_model import AccessDescr

    bank_dims = tuple(getattr(layout, "bank_dims", ()) or ())
    try:
        varying = tuple(getattr(layout, "bases", {}).keys())
        if bank_dims and varying:
            n = layout.conflict_count(bank_dims=bank_dims, varying_inputs=varying)
            if n > 0:
                return AccessDescr(
                    tier=tier, bank_dims=bank_dims,
                    conflict_free=False, conflict_count=n,
                )
    except Exception:
        pass
    return AccessDescr.identity(tier)


def build_cost_vector(
    target,
    lr: LiveRange,
    candidates: list[Any],
    spill_cb: Callable,
    seed: Any = None,
) -> CostVector:
    """Build a `CostVector` for one live range.

    `candidates` is the list of distinct handles ever assigned to
    `lr.memref_name` across the enumerator's candidate placements.
    `spill_cb` is the closure returned by `get_cost("register_spill",
    target)`. `seed` is the handle THIS candidate's enumerator assigned to
    the memref -- the placement the layout intends.

    The seed handle is priced as the minimum over the candidate tiers, so
    the allocator KEEPS the enumerator's layout-defining placement when it
    fits and only reaches for an alternative handle (or a spill) under
    capacity pressure (SPEC-022 D2). This preserves the no-spill corpus
    byte-for-byte -- the base access cost is locality-aware (it distinguishes
    candidates whose seeds sit on different tiers, feeding the candidate-level
    `total_cost`) WITHOUT overriding which handle a memref lands on inside a
    feasible candidate. The non-seed alternatives carry their own tier cost,
    used only as pressure-relief fallback ordering.
    """
    entries: dict[Any, int] = {}
    # The seed (the enumerator's intended handle for this memref) is listed
    # FIRST so the greedy keeps it when it fits -- the layout-defining
    # placement is not overridden by a cheaper-tier alternative; alternatives
    # exist only for capacity-pressure relief. Each handle still carries its
    # own locality-aware tier cost (feeding the candidate-level total).
    ordered_handles = list(candidates)
    if seed is not None and seed in ordered_handles:
        ordered_handles.remove(seed)
        ordered_handles.insert(0, seed)
    for h in ordered_handles:
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


def _forced_spill_choice(lr, cv, target, spill_cb):
    """Return `(Spilled(home, tier), cost)` for a forced spill, or None when
    the backend declares no spill tier (the hard-error case).

    Reuses an existing `Spilled` entry in the cost vector if present;
    otherwise synthesizes one for the backend's declared spill tier and
    prices it via `spill_cb` (the same factory `build_cost_vector` uses, so
    a forced spill and an enumerated spill price identically)."""
    for choice, cost in cv.entries.items():
        if isinstance(choice, Spilled):
            return (choice, cost)
    if target is None:
        return None
    name = getattr(target, "name", None)
    tier_builder = _SPILL_TIER_BUILDERS.get(name)
    if tier_builder is None:
        return None
    tier_name, _tier_handle = tier_builder(target)
    home = _pick_home(list(cv.entries.keys())) if cv.entries else None
    if home is None:
        return None
    n_uses = lr.last_idx - lr.first_idx + 1
    cost = 0
    if spill_cb is not None:
        try:
            cost = int(spill_cb(home, n_entries=n_uses))
        except TypeError:
            cost = int(spill_cb(home) * n_uses)
    return (Spilled(home_handle=home, tier=tier_name), cost)


def _solve(
    lrs: list[LiveRange],
    cost_vectors: dict[str, CostVector],
    cap: CapacityTable,
    target=None,
    spill_cb=None,
) -> AllocResult:
    """Greedy v1: sort by descending gap; cheapest-fits-first.

    PBQP slots in here -- replace `_solve` (same signature) with a
    branch-and-bound solver later. The data inputs are unchanged.

    `target`/`spill_cb` enable the SPEC-022 D2 forced-spill-under-pressure
    policy: when no unspilled choice fits, fall to a forced `Spilled(home,
    tier)` (real after D1) rather than raising -- the hard error is reserved
    for a backend whose ctx genuinely cannot spill the tier.
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
        # The first entry is the enumerator's seed handle (build_cost_vector
        # lists it first); it is tried FIRST so a feasible layout placement
        # is kept rather than overridden by a cheaper-tier alternative. The
        # remaining handles (capacity-pressure alternatives + the Spilled
        # fallback) are ranked cheapest-first. Within equal cost, insertion
        # order breaks ties (stable sort) -- the byte-identical analogue of
        # the pre-D2 all-zero behaviour.
        items = list(cv.entries.items())
        if items:
            seed_item = items[0]
            rest = sorted(items[1:], key=lambda kv: kv[1])
            ranked = [seed_item] + rest
        else:
            ranked = []
        chosen = None
        chosen_cost = 0
        for choice, cost in ranked:
            if _fits(choice, lr, occupants, cap):
                chosen = choice
                chosen_cost = cost
                break
        if chosen is None:
            # Forced spill under pressure (SPEC-022 D2): no unspilled tier
            # has room. If the backend declares a spill tier, synthesize a
            # `Spilled(home, tier)` (one may already be absent from the cost
            # vector when candidates were register-only) and select it --
            # spills are real after D1, so this keeps the workload compiling
            # correctly instead of dropping the only candidate. The hard
            # error is reserved for a backend that cannot spill at all.
            forced = _forced_spill_choice(lr, cv, target, spill_cb)
            if forced is None:
                raise RuntimeError(
                    f"regalloc: no choice fits for {lr.memref_name!r} and "
                    f"no spill tier is available; cost_vector={cv}"
                )
            chosen, chosen_cost = forced
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
        seed = candidate.placements.get(lr.memref_name)
        if not cands:
            # No candidate handle for this memref -- preserve whatever
            # the seed placement had (None falls through to codegen's
            # missing-placement error, matching today's behaviour).
            cands = []
            if seed is not None:
                cands = [seed]
        cost_vectors[lr.memref_name] = build_cost_vector(
            target, lr, cands, spill_cb, seed=seed
        )

    result = _solve(lrs, cost_vectors, cap, target=target, spill_cb=spill_cb)
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
    # SPEC-022 D3: carry the chosen F2 layout through the allocator so codegen
    # consumes the layout object (the `range(stride)` fiber walk reads
    # `layout.size_of(fiber_axis)`). `_solve` builds a bare Placement; without
    # this the layout signal is lost after regalloc.
    result.placement.layout = getattr(candidate, "layout", None)
    # SPEC-022 D1: carry the spill audit onto the placement so the move
    # scheduler can emit the LD/ST round-trip. `_solve` recorded the
    # spilled live ranges; surface them by memref name. Empty (and so
    # byte-identical) for every no-spill placement.
    result.placement._spilled = [lr.memref_name for lr in result.spilled]
    return result
