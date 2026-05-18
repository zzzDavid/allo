# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SPMW autoscheduler — picks a target-handle assignment for a trace.

Per report 16 §Autoscheduler, this is where Linear-Layout-driven bank
assignment and PBQP register allocation will eventually live. This file
is the smallest version of that loop: enumerate candidate `Placement`s,
score each via a named cost callback, return the argmin.

The candidate enumerator is per-backend; cost callbacks come from the
`spmw_cost` registry. A new backend plugs in by registering its own
enumerator under the target name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .spmw_cost import get_cost
from .spmw_linear_layout import LinearLayout, materialise_handle
from .spmw_match import MatchTrace, MatchedOp
from .spmw_target import UnitId


@dataclass
class Placement:
    """Per-memref placement on the target.

    `placements` maps a workload memref name (e.g. `"local_W"`,
    `"local_x"`, `"acc"`) to the target handle (`Register`, `MemoryRef`)
    chosen for it. Codegen reads this to translate matcher-side bindings
    to target-side handles when calling `emit`.
    """

    placements: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------- #
# Backend candidate enumerators
# --------------------------------------------------------------------- #


_ENUMERATORS: dict[str, Callable[[Any, list[MatchedOp]], list[Placement]]] = {}


def register_enumerator(target_name: str):
    """Decorator: register a candidate-enumeration function for a target."""

    def decorator(fn):
        _ENUMERATORS[target_name] = fn
        return fn

    return decorator


def _trace_memrefs_by_role(matches_or_trace) -> dict[str, str]:
    """Collect role -> memref_name across the supplied matches.

    Accepts either a `MatchTrace` (legacy single-group caller) or a
    plain `list[MatchedOp]` (the per-group autoscheduler path). Within
    the supplied set, two matches binding the same role to different
    memrefs is still an error — callers must pre-group by `func_name`
    (see `_bucket_for_autoschedule`) before calling this on a
    multi-kernel trace.
    """
    if isinstance(matches_or_trace, MatchTrace):
        matches = matches_or_trace.matches
    else:
        matches = matches_or_trace

    role_to_memref: dict[str, str] = {}
    for match in matches:
        for opb in match.operands:
            existing = role_to_memref.get(opb.role)
            if existing is None:
                role_to_memref[opb.role] = opb.memref_name
            elif existing != opb.memref_name:
                raise NotImplementedError(
                    f"role {opb.role!r} binds to multiple memrefs within "
                    f"the same work-group ({existing!r} vs {opb.memref_name!r}); "
                    "either the matcher is producing a non-uniform group or "
                    "the caller forgot to group by func_name."
                )
    return role_to_memref


def _bucket_for_autoschedule(trace: MatchTrace) -> list[tuple[str, list[MatchedOp]]]:
    """Group `trace.matches` by `func_name`, preserving first-seen order.

    Returns ``[(func_name, [matches]), ...]``. One bucket per
    `@allo.work` kernel; within a bucket, all matches must agree on
    role -> memref (enforced by `_trace_memrefs_by_role`).
    """
    buckets: dict[str, list[MatchedOp]] = {}
    order: list[str] = []
    for m in trace.matches:
        if m.func_name not in buckets:
            buckets[m.func_name] = []
            order.append(m.func_name)
        buckets[m.func_name].append(m)
    return [(name, buckets[name]) for name in order]


@register_enumerator("samsung_hbm_pim")
def _samsung_enumerate(target, matches: list[MatchedOp]) -> list[Placement]:
    """Construct the optimal Samsung layout via LinearLayout algebra.

    Per spec 013 §C: build the identity layout on (grf, bank, tile),
    call `LinearLayout.optimal_swizzle` to drop the bank-bit-0 swizzle
    (Zhou et al. ASPLOS '26 §5.4), then materialise into the canonical
    `target.banks[2*pid]` / grf_a / grf_b handles. MAC's `dst=grf_b`
    constraint pins `acc -> grf_b`.

    Returns exactly ONE Placement — the algebraically optimal layout.
    The cost-fn argmin in `autoschedule` still runs (single-element
    list); keeps the control flow uniform.
    """
    role_to_memref = _trace_memrefs_by_role(matches)
    x_mref = role_to_memref.get("x")
    y_mref = role_to_memref.get("y")
    acc_mref = role_to_memref.get("acc")
    if x_mref is None or y_mref is None or acc_mref is None:
        raise NotImplementedError(
            "samsung enumerator: trace is missing one of x/y/acc roles; "
            f"got {sorted(role_to_memref)}"
        )

    # Build the identity base on (grf, bank, tile) over outputs (grf, bank),
    # then swizzle so the segment dim (tile) contributes to bank bit 0.
    base = LinearLayout.identity(
        {"grf": 8, "bank": 16, "tile": 2},
        out_dims=("grf", "bank"),
    )
    swizzled = LinearLayout.optimal_swizzle(
        base,
        vec_dims=("grf",),
        bank_dims=("bank",),
        segment_dims=("tile",),
    )

    # Bind the bank input dim to `2 * pid` (level-1 pim UnitId): the pim
    # axis has 8 units mapping to 16 banks, so each pim owns the even
    # bank `2*pid` and its odd partner `2*pid + 1`. At fixed tile=0 this
    # yields `idx = 2*pid` (EVEN_BANK); the codegen-side cmd stream
    # toggles tile per K-iteration to fold the swizzle into the loop.
    pid = UnitId(level=1, unit=None)
    y_handle = materialise_handle(
        swizzled,
        target=target,
        out_dim="bank",
        fixed={"grf": 0, "tile": 0},
        symbol_table={"bank": 2 * pid},
    )

    placement = Placement(placements={
        x_mref: target.grf_a,
        y_mref: y_handle,
        acc_mref: target.grf_b,
    })
    return [placement]


@register_enumerator("aim")
def _aim_enumerate(target, matches: list[MatchedOp]) -> list[Placement]:
    """Enumerate candidate layouts for SK-Hynix AiM.

    Per spec 013 §D.1, the AiM A/B choice is "does the layout factor `bank`
    in as an input dim?". The two algebraic constructions are:

      no_bank_layout  = identity({"k": K})       -- per-bank MAC (MAC_SBK)
      all_bank_layout = no_bank ⊗ identity({"bank": NBANKS})  -- broadcast (MAC_ABK)

    Both materialise to the same `target.banks[8*bg + bk]` handle for
    operands that live in DRAM (codegen reads the role-to-bank vs role-to-gb
    selection from the Placement directly). `acc` always lands in `gpr`
    (AiM's MAC ISR writes the per-channel accumulator file).
    """
    role_to_memref = _trace_memrefs_by_role(matches)
    x_mref = role_to_memref.get("x")
    y_mref = role_to_memref.get("y")
    acc_mref = role_to_memref.get("acc")
    if x_mref is None or y_mref is None or acc_mref is None:
        raise NotImplementedError(
            "aim enumerator: trace is missing one of x/y/acc roles; "
            f"got {sorted(role_to_memref)}"
        )

    # Algebraic candidate descriptors (kept for parity with spec 013 §D.1;
    # the concrete handle plumbing below reads from the same target tree
    # either way -- the layout choice is encoded in whether `y` lands in
    # a bank or in `gb`).
    no_bank_layout = LinearLayout.identity({"k": 16}, out_dims=("k",))
    all_bank_layout = no_bank_layout.product(
        LinearLayout.identity({"bank": 16}, out_dims=("bank",))
    )
    del no_bank_layout, all_bank_layout  # algebraic-only; not materialised here

    # AiM tree is `device -> channel -> bg -> bank`. The bank-level unit
    # is where banks / gb / gpr are bound; we read them from the target's
    # flat handle map.
    bg_id = UnitId(level=2, unit=None)
    bk_id = UnitId(level=3, unit=None)
    banks = target.banks
    gb = target.gb
    gpr = target.gpr

    bank_handle = banks[8 * bg_id + bk_id]

    layouts: list[Placement] = []
    # Candidate 1: per-bank MAC (no_bank_layout -- y stays in its bank).
    layouts.append(Placement(placements={
        x_mref: bank_handle,
        y_mref: bank_handle,
        acc_mref: gpr,
    }))
    # Candidate 2: all-bank-broadcast MAC (all_bank_layout -- y rides gb).
    layouts.append(Placement(placements={
        x_mref: bank_handle,
        y_mref: gb,
        acc_mref: gpr,
    }))
    return layouts


@register_enumerator("upmem")
def _upmem_enumerate(target, matches: list[MatchedOp]) -> list[Placement]:
    """Enumerate candidate layouts for UPMEM DPUs.

    Per spec 013 §D.2, the UPMEM A/B choice is whether `acc` rides the
    per-tasklet GPR file or a WRAM cell. Algebraically:

      scalar_layout  = identity({"element": E})         -- acc in WRAM
      tasklet_layout = scalar ⊗ identity({"tasklet": T}) -- acc in GPR

    MRAM is reserved for bulk loads; live operands stay in WRAM/GPR.
    """
    role_to_memref = _trace_memrefs_by_role(matches)
    x_mref = role_to_memref.get("x")
    y_mref = role_to_memref.get("y")
    acc_mref = role_to_memref.get("acc")
    if x_mref is None or y_mref is None or acc_mref is None:
        raise NotImplementedError(
            "upmem enumerator: trace is missing one of x/y/acc roles; "
            f"got {sorted(role_to_memref)}"
        )

    scalar_layout = LinearLayout.identity({"element": 16}, out_dims=("element",))
    tasklet_layout = scalar_layout.product(
        LinearLayout.identity({"tasklet": 16}, out_dims=("tasklet",))
    )
    del scalar_layout, tasklet_layout  # algebraic-only; handles below

    wram = target.wram
    gprs = target.gprs

    # Symbolic WRAM offsets — distinct integer indices keep the three
    # operand placements distinguishable in the layout dict. The C
    # compiler resolves concrete addresses.
    wram_x = wram[0]
    wram_y = wram[1]
    wram_acc = wram[2]

    layouts: list[Placement] = [
        Placement(placements={
            x_mref: wram_x,
            y_mref: wram_y,
            acc_mref: wram_acc,
        }),
        Placement(placements={
            x_mref: wram_x,
            y_mref: wram_y,
            acc_mref: gprs,
        }),
    ]
    return layouts


@register_enumerator("apu_v1")
def _apu_v1_enumerate(target, matches: list[MatchedOp]) -> list[Placement]:
    """Enumerate placements for APU v1.

    Per spec 013 §D.3, this is a one-liner `LinearLayout.identity` over the
    32K-lane bit-serial element axis. All compute operands live in VRs;
    register pressure (16 VRs per APUC) is deferred to the regalloc spec
    (Task 015). MICRO '25 Opt2 (stage-axis lift, 1.25x speedup) needs the
    move scheduler to see the full @allo.work chain and is also deferred.
    """
    role_to_memref = _trace_memrefs_by_role(matches)
    x_mref = role_to_memref.get("x")
    y_mref = role_to_memref.get("y")
    acc_mref = role_to_memref.get("acc")
    if x_mref is None or y_mref is None or acc_mref is None:
        raise NotImplementedError(
            "apu_v1 enumerator: trace is missing one of x/y/acc roles; "
            f"got {sorted(role_to_memref)}"
        )

    _layout = LinearLayout.identity({"element": 32768}, out_dims=("element",))
    del _layout  # algebraic-only; the VR file is the operand store

    vrs = target.vrs
    return [
        Placement(placements={
            x_mref: vrs,
            y_mref: vrs,
            acc_mref: vrs,
        }),
    ]


@register_enumerator("apu_v2")
def _apu_v2_enumerate(target, matches: list[MatchedOp]) -> list[Placement]:
    """Enumerate placements for APU v2.

    Per spec 013 §D.4, the canonical L1-row layout is a one-liner
    `LinearLayout.identity` over the element + group axes. The L1 grid
    is the operand store; l1_sim treats all L1 addresses uniformly so
    there is no swizzle algebra to exercise yet.
    """
    role_to_memref = _trace_memrefs_by_role(matches)
    x_mref = role_to_memref.get("x")
    y_mref = role_to_memref.get("y")
    acc_mref = role_to_memref.get("acc")
    if x_mref is None or y_mref is None or acc_mref is None:
        raise NotImplementedError(
            "apu_v2 enumerator: trace is missing one of x/y/acc roles; "
            f"got {sorted(role_to_memref)}"
        )

    _layout = LinearLayout.identity(
        {"element": 65536, "group": 16}, out_dims=("element", "group")
    )
    del _layout  # algebraic-only; concrete L1-row slots assigned below

    l1 = target.l1
    return [
        Placement(placements={
            x_mref: l1[0],
            y_mref: l1[1],
            acc_mref: l1[2],
        }),
    ]


# --------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------- #


def autoschedule(
    target,
    trace: MatchTrace,
    cost_name: str = "kernel_cycles",
) -> list[Placement]:
    """Pick one `Placement` per `@allo.work` kernel in `trace`.

    Returns a list aligned with `_bucket_for_autoschedule(trace)`:
    ``result[i]`` is the chosen placement for the i-th kernel (in
    trace order). For single-kernel workloads this list has length 1.

    Per spec 015, the allocator runs per candidate (after the
    enumerator, before argmin); the argmin is on
    `kernel_cycles + allocator.total_cost`. Set
    `SPMW_DISABLE_REGALLOC=1` in the environment to bypass the
    allocator (emergency rollback path).
    """
    import os

    target_name = getattr(target, "name", None)
    enumerator = _ENUMERATORS.get(target_name)
    if enumerator is None:
        raise NotImplementedError(
            f"no autoscheduler enumerator registered for target {target_name!r}; "
            f"supported: {sorted(_ENUMERATORS)}"
        )
    cost_fn = get_cost(cost_name, target)
    regalloc_disabled = os.environ.get("SPMW_DISABLE_REGALLOC") == "1"

    placements: list[Placement] = []
    for func_name, matches in _bucket_for_autoschedule(trace):
        candidates = enumerator(target, matches)
        if not candidates:
            raise RuntimeError(
                f"no candidate layouts produced for kernel {func_name!r} "
                f"on target {target_name!r}"
            )
        sub_trace = MatchTrace(
            target_name=trace.target_name,
            module_name=trace.module_name,
            matches=matches,
        )

        if regalloc_disabled:
            # Kill switch: skip the allocator, score by kernel_cycles
            # alone (preserves spec 012a behaviour exactly).
            scored = [
                (cost_fn(sub_trace, layout), idx)
                for idx, layout in enumerate(candidates)
            ]
            scored.sort()
            placements.append(candidates[scored[0][1]])
            continue

        # Per-candidate regalloc; argmin on kernel_cycles + total_cost.
        # Lazy import to avoid an import cycle through spmw_cost_models.
        from .spmw_regalloc import allocate
        # Each scored entry is (combined_cost, enumerator_idx,
        # refined_placement); we keep the placement in the tuple so the
        # sort itself selects the right refined layout.
        scored: list[tuple[int, int, Placement]] = []
        for idx, cand in enumerate(candidates):
            try:
                alloc = allocate(
                    target, matches, cand, all_candidates=candidates,
                )
            except RuntimeError:
                # No allocator-feasible placement for this candidate.
                continue
            kc = cost_fn(sub_trace, alloc.placement)
            scored.append((kc + alloc.total_cost, idx, alloc.placement))
        if not scored:
            raise RuntimeError(
                f"no allocator-feasible placement for {func_name!r} "
                f"on target {target_name!r}"
            )
        scored.sort(key=lambda t: (t[0], t[1]))
        placements.append(scored[0][2])
    return placements
