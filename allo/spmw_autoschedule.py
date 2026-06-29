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
from .spmw_target import Register, UnitId


@dataclass
class Placement:
    """Per-memref placement on the target.

    `placements` maps a workload memref name (e.g. `"local_W"`,
    `"local_x"`, `"acc"`) to the target handle (`Register`, `MemoryRef`)
    chosen for it. Codegen reads this to translate matcher-side bindings
    to target-side handles when calling `emit`.

    `mode` is a free-form label that the cost model and codegen consult
    to disambiguate candidates whose `placements` dict is identical or
    whose op-expansion differs (e.g. APU v1 SV vs SV-lookup). Default
    `""` preserves existing behaviour for cost models that don't read
    it. `extra` is a free-form per-candidate scratch dict.
    """

    placements: dict[str, Any] = field(default_factory=dict)
    mode: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    # Spill audit (SPEC-022 D1): memref names the allocator spilled. Empty
    # for every no-spill placement (byte-identical default); the move
    # scheduler reads it to emit the tier LD/ST round-trip around the
    # spilled value's work-id window. The `Spilled(home, tier)` wrapper in
    # `placements` still unwraps to its home register for the compute emit;
    # this list is what stops the round-trip being silently dropped.
    _spilled: list[str] = field(default_factory=list)
    # Chosen physical layout (SPEC-022 D3): the swizzled `LinearLayout` the
    # enumerator built to materialise the fiber handles. Carried so codegen
    # consumes the F2 layout OBJECT directly (the `range(stride)` fiber walk
    # reads `layout.size_of(fiber_axis)`) instead of pattern-matching a
    # `MemoryRef.idx`. Default `None` == today's behaviour (codegen falls back
    # to the index's own coefficient, byte-identical for Samsung stride 2).
    layout: Any = None


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


def _samsung_host_eligible_memrefs(
    target, placement: "Placement", role_to_memref: dict[str, str]
) -> list[str]:
    """Return the memrefs whose GRF preload may be host-broadcast.

    A role is host-eligible (SPEC-024 §3) iff (a) its placement handle is
    the broadcast GRF register `grf_a` -- the register the target declares
    as the HAB-broadcast input -- AND (b) its home is the broadcast vector,
    not a per-bank-staged operand. For GEMV that is the `x`/vector role:
    `x` is broadcast to every bank, so one host write fills GRF_A for the
    whole fan-out. The weight `y` may also be staged into `grf_a`
    (grf_staged mode) but its home is a per-bank handle, so it is *not*
    broadcast-uniform and stays crf-resident; `acc -> grf_b` is the
    per-work-id accumulator, also excluded. Both tests are over handle
    identity + role home, never a shape.
    """
    grf_a = target.grf_a
    x_mref = role_to_memref.get("x")
    if x_mref is None:
        return []
    handle = placement.placements.get(x_mref)
    if isinstance(handle, Register) and handle is grf_a:
        return [x_mref]
    return []


def _with_residency(base: "Placement", memref: str, mode: str) -> "Placement":
    """Copy `base`, tagging `memref`'s GRF residency in `extra`.

    Residency is *how* a GRF is filled (host broadcast vs CRF MOV), not
    *which* handle holds the value, so `placements` is untouched -- the
    layout algebra (lever 1's fibers) rides alongside unchanged.
    """
    new_extra = dict(base.extra)
    residency = dict(new_extra.get("grf_residency", {}))
    residency[memref] = mode
    new_extra["grf_residency"] = residency
    return Placement(
        placements=dict(base.placements),
        mode=base.mode,
        extra=new_extra,
        layout=getattr(base, "layout", None),
    )


def _join_mode(mode: str, token: str) -> str:
    """Append a human-readable lever token to `mode` for audit dumps.

    The authoritative CRF-issue decision lives in `extra["crf_issue"]`;
    this only keeps `mode` legible (e.g. ``dual_fiber+crf_shared``).
    """
    return f"{mode}+{token}" if mode else token


def _with_crf_modes(base: "Placement") -> list["Placement"]:
    """Return the {shared, per_workid} CRF-issue variants of `base`.

    Lever 3 (SPEC-025 §3): the CRF body is either programmed once and
    fired per work-id by the host (`shared`) or replicated per work-id
    on the CRF stream (`per_workid`). This is orthogonal to where `y`
    lives and to lever-1/lever-2 residency, so it rides
    `extra["crf_issue"]` rather than cross-producting the `mode` string.
    Both variants are materialisable; argmin discards the loser. No shape
    branch -- the enumerator emits both unconditionally.
    """
    variants = []
    for token, issue in (("crf_shared", "shared"),
                         ("crf_per_workid", "per_workid")):
        new_extra = dict(base.extra)
        new_extra["crf_issue"] = issue
        variants.append(
            Placement(
                placements=dict(base.placements),
                mode=_join_mode(base.mode, token),
                extra=new_extra,
                layout=getattr(base, "layout", None),
            )
        )
    return variants


def _with_stage_resident(base: "Placement", resident: bool) -> "Placement":
    """Copy `base`, stamping the structural `extra['stage_resident']` flag.

    Bridge option (b) (design 05 §3 / task-017): the host-staging
    materialisation flag (preload W once and reuse across the batch vs
    re-preload per input vector). It rides `extra`, not `placements` -- the
    bank algebra (lever 1's fibers) is unchanged. The `host_staging`
    CostModel reads this flag to price preload-once vs preload-B; the 013
    residency hoist stamps the same flag from the language-level
    `residency="resident"` collective. (Renamed off the deleted
    `weight_resident` key in task-017; the cost branch that keyed on
    `weight_resident` is gone -- the split now lives in the host_staging
    compose.) At B=1 the two variants tie and the argmin winner is
    undisturbed; the resident one is earned for B>=2.
    """
    new_extra = dict(base.extra)
    new_extra["stage_resident"] = resident
    return Placement(
        placements=dict(base.placements),
        mode=_join_mode(base.mode, "wresident") if resident else base.mode,
        extra=new_extra,
        layout=getattr(base, "layout", None),
    )


@register_enumerator("samsung_hbm_pim")
def _samsung_enumerate(target, matches: list[MatchedOp]) -> list[Placement]:
    """Enumerate Samsung layouts: bank-row `y` vs GRF-staged `y`.

    Per SPEC-009 §1: bank-row `y` (is_auto=1, K-loop folds) is the
    optimal layout via LinearLayout algebra (Zhou et al. ASPLOS '26
    §5.4); GRF-staged `y` (is_auto=0, K MACs unrolled) is the
    alternative the cost model can distinguish via
    `isinstance(y_handle, MemoryRef)`. MAC's `dst=grf_b` constraint
    pins `acc -> grf_b` in both candidates.

    Argmin picks bank-row (~8x cheaper at K=1024).
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
    # then swizzle so the segment dim (tile) contributes to the bank bits.
    # The tile (segment) axis size IS banks-per-pim = bank_out // pim_units,
    # geometry-derived from the tree (SPEC-022 D3): for Samsung 16//8 == 2 (one
    # tile bit -> the even/odd swizzle, byte-identical); a wide target with
    # 16//4 == 4 gets a 2-bit tile -> a 4-fiber `range(stride)` walk. The
    # `range(stride)` generalization is what makes the F2 algebra load-bearing
    # beyond Samsung's factor-2 swizzle.
    bank_out = _bank_out_size(target)
    pim_units = _pim_unit_count(target)
    banks_per_pim = (bank_out // pim_units) if (bank_out and pim_units) else 2
    base = LinearLayout.identity(
        {"grf": 8, "bank": bank_out or 16, "tile": banks_per_pim},
        out_dims=("grf", "bank"),
    )
    swizzled = LinearLayout.optimal_swizzle(
        base,
        vec_dims=("grf",),
        bank_dims=("bank",),
        segment_dims=("tile",),
    )

    # Bind the bank input dim to `stride * pid` (level-1 pim UnitId): the
    # pim axis maps its units onto the bank out-axis, so each pim owns a
    # contiguous run of `stride` banks starting at `stride*pid` (banks-per-pim
    # = the bank out-size over the pim unit count). For Samsung that is
    # `16 // 8 == 2`, so each pim owns the even bank `2*pid` and its odd
    # partner `2*pid + 1`. The stride is geometry-derived (SPEC-022): the
    # `2` is the `bank_out // pim_units` instantiation, not a pasted constant.
    # Materialising the swizzled layout at `fixed={"tile": v}` evaluates the
    # same `tile->bank-bit-0` swizzle column at each fiber value: v=0 ->
    # `stride*pid` (EVEN_BANK), v=1 -> `stride*pid + 1` (ODD_BANK). The
    # `+0`/`+1` fall out of the swizzle column (SPEC-023 §2).
    pid = UnitId(level=1, unit=None)
    bank_stride = _bank_stride_per_pim(target, swizzled)

    # Fiber values come from the layout's segment (tile) axis size, not a
    # literal `2`. `size_of("tile") == 2` here because the swizzle gives
    # the tile axis one basis bit.
    n_fibers = swizzled.size_of("tile")
    fibers = [
        materialise_handle(
            swizzled,
            target=target,
            out_dim="bank",
            fixed={"grf": 0, "tile": v},
            symbol_table={"bank": bank_stride * pid},
        )
        for v in range(n_fibers)
    ]
    y_even = fibers[0]

    # Candidate 1: bank-row `y` (is_auto=1 -> folded K-loop, EVEN fiber).
    # Carries the swizzled F2 layout (SPEC-022 D3) + the fiber axis so codegen
    # reads the bank-fiber stride from `layout.size_of("tile")`, not the index.
    bank_row = Placement(
        placements={
            x_mref: target.grf_a,
            y_mref: y_even,
            acc_mref: target.grf_b,
        },
        mode="bank_row",
        extra={"fiber_axis": "tile"},
        layout=swizzled,
    )
    # Candidate 2: GRF-staged `y` (is_auto=0 -> K MACs unrolled).
    # Both candidates share x->grf_a, acc->grf_b; only y differs.
    grf_staged = Placement(
        placements={
            x_mref: target.grf_a,
            y_mref: target.grf_a,
            acc_mref: target.grf_b,
        },
        mode="grf_staged",
        layout=swizzled,
    )
    # Candidate 3: dual-fiber bank-row `y` (is_auto=1, both bank halves
    # busy). Strict superset of `bank_row`: `placements[y]` is the EVEN
    # fiber so any consumer ignoring `extra` degrades to `bank_row`. The
    # per-fiber handles ride `extra["fibers"]`; codegen materialises the
    # alternating (MAC EVEN, JUMP, MAC ODD, JUMP) stream from them and the
    # cost model prices it ~n_fibers x cheaper (concurrent bank halves).
    dual_fiber = Placement(
        placements={
            x_mref: target.grf_a,
            y_mref: y_even,
            acc_mref: target.grf_b,
        },
        mode="dual_fiber",
        extra={
            "fibers": list(fibers),
            "fiber_axis": "tile",
            "n_fibers": n_fibers,
        },
        layout=swizzled,
    )
    base_candidates = [bank_row, grf_staged, dual_fiber]

    # Lever 2 (SPEC-024): cross the layout candidates with {crf, host}
    # Lever cross-product via the typed knob registry (SPEC-022 D4). The
    # three Samsung levers -- grf_residency (SPEC-024), crf_issue (SPEC-025),
    # stage_resident (SPEC-026) -- are registered knobs; `cross_with_knobs`
    # applies them in registration order (residency -> crf_issue ->
    # stage_resident), byte-identical to the prior hand-crossed loop. Each
    # knob owns its candidate set + materialiser; adding a lever is one
    # `register_knob` call, not a new loop here.
    from .spmw_knobs import cross_with_knobs

    return cross_with_knobs(target, base_candidates, matches, role_to_memref)


@register_enumerator("mortise")
def _mortise_enumerate(target, matches: list[MatchedOp]) -> list[Placement]:
    """Enumerate Mortise layouts (design 06 §3.3).

    Mortise is built Samsung-shaped (`tests/spmw/_mortise_target.py`:
    grf_a/grf_b, nested tiles, the same moves/ops), so its layout
    enumeration IS Samsung's: the same bank-row / grf-staged / dual-fiber
    candidates crossed with residency / crf-issue / `stage_resident`. The
    only Mortise-specific axis is the capacity lever `C`, which is a target
    CONSTANT (`resident_cap_elems`) read by the host_staging compose -- not
    a placement axis, so it does not multiply the candidate set.

    This delegates to `_samsung_enumerate` rather than re-deriving the
    bank algebra: Mortise's tree is Samsung's tree plus a const, so the
    `stage_resident in {False, True}` candidate pair the autoscheduler
    argmin-selects over is produced identically. Argmin picks the resident
    arm at B>=2 (the Mortise faithful host_staging prices it cheaper).
    """
    return _samsung_enumerate(target, matches)


@register_enumerator("mortise_wide")
def _mortise_wide_enumerate(target, matches: list[MatchedOp]) -> list[Placement]:
    """Enumerate the Mortise-WIDE (banks_per_pim==4) layouts (SPEC-022 D3).

    Samsung-shaped, so it delegates to `_samsung_enumerate`. Because the
    swizzle tile axis is now sized by `banks_per_pim` (= bank_out // pim_units),
    a pim fanout of 4 over 16 banks yields a 2-bit tile -> `size_of("tile")==4`
    -> FOUR materialised fiber handles `banks[4*pid + r]` for r in range(4).
    This is the `banks_per_pim > 2` proof input the deleted two-class
    `_bank_parity` matcher could never have classified.
    """
    return _samsung_enumerate(target, matches)


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

    # AiM topology note: the A/B choice here is per-bank MAC (MAC_SBK) vs.
    # all-bank-broadcast MAC (MAC_ABK). Algebraically this is "does the
    # layout factor `bank` into an input dim?" but the runtime distinction
    # is which *physical unit* `y` lives on -- a per-bank operand
    # (`banks[8*bg+bk]`) or the chip-level global buffer (`target.gb`).
    # `gb` is not an algebraic offset from `bank`; it is a discrete
    # hardware unit. LinearLayout cannot model the choice, so we enumerate
    # the two Placements directly. `acc` always lives in the per-channel
    # accumulator file (`target.gpr`) because AiM's MAC ISR writes there.

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


def _trace_reduction_trip(matches: list[MatchedOp]) -> int | None:
    """Innermost enclosing-loop bound of the reducing match (per-DPU K).

    The tasklet lever stripes the inner-K reduction across tasklets, so
    its work quantity is the trip count of the loop the accumulating
    (MAC) match sits in -- the same `enclosing_loops[-1]` bound the cost
    model reads. Returns None when no reducing match carries an inner
    loop, in which case the enumerator falls back to the `nt=1`-only
    candidate (parity with today). Mirror of SPEC-026's structural
    `batch_dim` resolver; rides existing `enclosing_loops` (no
    `allo/ir/` edit, per design 02 §3.1).
    """
    from .spmw_cost_models import _parse_loop_bound

    for match in matches:
        if match.target_op_name != "MAC":
            continue
        if not match.enclosing_loops:
            continue
        bound = _parse_loop_bound(match.enclosing_loops[-1][2])
        if bound is not None:
            return bound
    return None


def _bank_stride_per_pim(target, layout: LinearLayout) -> int:
    """Banks-per-pim stride, derived from layout + target geometry.

    `stride = bank_out_size // pim_unit_count`: the bank out-axis size
    (read off the layout's `bank` axis, == 16 for Samsung) divided by the
    pim-unit fanout (the product of the `pim` unit's `mapping`, == 8). Each
    pim owns `stride` contiguous banks based at `stride*pid`. For Samsung
    this is `16 // 8 == 2` -- the `2` is this instantiation, never a pasted
    constant (SPEC-022 anti-hardcoding gate). Falls back to a stride of 1
    when the geometry is unresolvable (no `pim` unit / no bank axis), which
    degrades to the identity `pid` binding rather than guessing a `2`.
    """
    from math import prod

    bank_out = layout.size_of("bank") if "bank" in layout.bases else None
    pim_units = None
    for u in target._walk():
        if u.name == "pim":
            pim_units = prod(u.mapping) if u.mapping else None
            break
    if not bank_out or not pim_units:
        return 1
    return bank_out // pim_units


def _pim_unit_count(target) -> int | None:
    """Pim-unit fanout (`prod(pim.mapping)`) read from the unit tree, or None
    when there is no `pim` unit. Sibling of `_bank_stride_per_pim`'s pim walk,
    lifted out so the enumerator can size the swizzle tile axis BEFORE the
    layout exists (the tile size == banks-per-pim == bank_out // pim_units)."""
    from math import prod

    for u in target._walk():
        if u.name == "pim":
            return prod(u.mapping) if u.mapping else None
    return None


def _bank_out_size(target) -> int | None:
    """Bank out-axis size (the `banks` count) read from the target's `banks`
    memory geometry, or None when absent. The bank dimension the swizzle maps
    the tile fibers onto; geometry, never a pasted 16."""
    banks = getattr(target, "banks", None)
    if banks is None:
        return None
    n = getattr(banks, "banks", None)
    return int(n) if n else None


def _tasklet_fanout(target) -> int | None:
    """Tasklet-unit fanout (T_max) read from the target unit tree.

    Walks for the unit named `tasklet` and returns the product of its
    `mapping` (= 16 in the fixture's `@allo.unit(mapping=[16])`). This is
    target-derived, never the literal 16. Returns None when absent.
    """
    from math import prod

    for u in target._walk():
        if u.name == "tasklet":
            return prod(u.mapping) if u.mapping else None
    return None


def _upmem_tasklet_candidates(target, matches: list[MatchedOp]) -> list[int]:
    """Derive the enumerated `n_tasklets` candidate set, shape+target only.

    `{1, T_max}` at minimum (>=2-candidate discipline), where `T_max` is
    the tasklet-unit fanout. When the reduction trip is known and smaller
    than `T_max`, cap `T_max` at the trip so a short reduction is not
    over-subscribed. No `16`/`1024`/`5.71` literal: the set is a function
    of the unit fanout and the traced reduction trip only.
    """
    t_max = _tasklet_fanout(target)
    if not t_max or t_max <= 1:
        return [1]  # no tasklet axis -> parity with pre-lever behaviour
    trip = _trace_reduction_trip(matches)
    if trip is not None and trip < t_max:
        t_max = max(1, trip)
    if t_max <= 1:
        return [1]
    return [1, t_max]


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

    # UPMEM topology note: the A/B choice is whether `acc` lives in a
    # WRAM cell or in the per-tasklet GPR file. `wram[0/1/2]` and `gprs`
    # are discrete named storage classes on the DPU hierarchy
    # (mram-vs-wram-vs-gprs), not coordinates on a linear address space,
    # so LinearLayout has no out_dim that names this choice. The two
    # Placements below encode the choice directly. Bulk MRAM loads are
    # the C runtime's job; live operands stay in WRAM/GPR.
    wram = target.wram
    gprs = target.gprs

    # Symbolic WRAM offsets — distinct integer indices keep the three
    # operand placements distinguishable in the layout dict. The C
    # compiler resolves concrete addresses.
    wram_x = wram[0]
    wram_y = wram[1]
    wram_acc = wram[2]

    acc_placements = [
        {x_mref: wram_x, y_mref: wram_y, acc_mref: wram_acc},
        {x_mref: wram_x, y_mref: wram_y, acc_mref: gprs},
    ]

    # Tasklet-tiling lever (design 02 §3.2): cross each acc-placement with the
    # derived `n_tasklets` candidate set via the typed knob registry (SPEC-022
    # D4). The base candidates are the two acc-placements; `cross_with_knobs`
    # applies the registered `n_tasklets` knob (candidates =
    # `_upmem_tasklet_candidates`, shape+target-derived). Byte-identical to the
    # prior hand-crossed loop. Default `n_tasklets=1` == today's behaviour.
    from .spmw_knobs import cross_with_knobs

    base_candidates = [Placement(placements=dict(p)) for p in acc_placements]
    return cross_with_knobs(target, base_candidates, matches, role_to_memref)


@register_enumerator("apu_v1")
def _apu_v1_enumerate(target, matches: list[MatchedOp]) -> list[Placement]:
    """Enumerate placements for APU v1.

    Per spec 013 §D.3 the operand placement is a one-liner
    `LinearLayout.identity` over the 32K-lane bit-serial element axis:
    all compute operands live in `target.vrs`, so there is no operand-
    swizzle DOF.

    The live DOF (design 01 §4) is the **VR-tile / L4-DMA mode**:
    intra-VR re-fetches the contraction operand per output tile,
    inter-VR loads it once and reuses it across output tiles. The two
    cross the {sv, sv_lookup} MAC-expansion choice, giving four
    candidates. The tile counts (`n_out_tiles`, `n_k_tiles`) carried in
    `extra` are `ceil`-arithmetic over the operand shape and
    `target.vrs.*` (computed by `_apu_v1_vr_tiling`), never a benchmark
    literal; codegen materialises the chosen `vr_dma` so host and device
    layout agree by construction.
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

    # APU v1 topology note: 32K-lane bit-serial element axis with a
    # single VR file (16 VRs per APUC). All compute operands live in
    # `target.vrs`, so the enumerator has zero swizzle degrees of
    # freedom -- the two candidates differ only by MAC op-expansion:
    # raw MUL+ADD (SV mode, 18 cyc) vs gvml_lookup_16 + add (SV-lookup,
    # 8 cyc). The cost model branches on `placement.mode`; argmin picks
    # `sv_lookup`. See SPEC-009 §2.
    vrs = target.vrs
    placements = {
        x_mref: vrs,
        y_mref: vrs,
        acc_mref: vrs,
    }

    # VR-tile arithmetic from operand shape + target.vrs (design 01 §4.1).
    # The tile counts are identical for both vr_dma modes (they share the
    # shape); only how the moves are issued (per-tile re-fetch vs reuse)
    # differs, and the cost model prices that difference.
    from .spmw_cost_models import _apu_v1_vr_tiling

    trace = MatchTrace(
        target_name="apu_v1", module_name="<enumerate>", matches=list(matches)
    )
    n_out_tiles, n_weight_tiles, n_boundaries = _apu_v1_vr_tiling(target, trace)

    # vr_dma lever via the typed knob registry (SPEC-022 D4). The base
    # candidates are one per MAC-expansion mode (sv / sv_lookup), each carrying
    # the shared tile-count extra; `cross_with_knobs` applies the registered
    # `vr_dma` knob (candidates = {intra, inter}) as the inner 2x fan. The
    # mode-outer / vr_dma-inner order + the 4-candidate set are byte-identical
    # to the prior hand-crossed `for mode: for vr_dma:` loop.
    from .spmw_knobs import cross_with_knobs

    base_candidates = [
        Placement(
            placements=dict(placements),
            mode=mode,
            extra={
                "n_out_tiles": n_out_tiles,
                "n_weight_tiles": n_weight_tiles,
                "n_stage_boundaries": n_boundaries,
            },
        )
        for mode in ("sv", "sv_lookup")
    ]
    return cross_with_knobs(target, base_candidates, matches, role_to_memref)


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

    # APU v2 topology note: 64K-lane element axis with a 16-row L1
    # group. l1_sim treats all L1 addresses uniformly, so the cost
    # model is a placeholder (constant per match). Two candidates with
    # symbolic l1[0/1/2] bindings keep argmin exercised; under the
    # placeholder cost they tie, and the sort tie-breaks on enumerator
    # index (Candidate 1 wins). See SPEC-009 §3.
    l1 = target.l1
    return [
        Placement(
            placements={
                x_mref: l1[0],
                y_mref: l1[1],
                acc_mref: l1[2],
            },
            mode="l1_row_canonical",
        ),
        Placement(
            placements={
                x_mref: l1[2],
                y_mref: l1[1],
                acc_mref: l1[0],
            },
            mode="l1_row_reversed",
        ),
    ]


# --------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------- #


def autoschedule(
    target,
    trace: MatchTrace,
    cost_name: str = "kernel_cycles",
    confidence_gate: bool = False,
    gate_policy: str = "warn",
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

    `confidence_gate` (design 07 §A4.3) is OPT-IN and defaults OFF: when
    `False` the argmin scoring core is byte-identical to today (commit
    regardless) -- `_check_confidence` is never called. When `True`,
    `_check_confidence` runs AFTER the argmin -- it never alters the argmin
    SELECTION, only whether a low-confidence ranking is silently committed.
    `gate_policy` (only consulted when the gate is on):
      * "warn"   -- warn-and-commit (the default; never blocks a schedule).
      * "refuse" -- refuse-and-error: raise on a placeholder/coarse/assumption
                    ranking instead of silently committing it.
    """
    import os

    if gate_policy not in ("warn", "refuse"):
        raise ValueError(
            f"autoschedule gate_policy must be 'warn' or 'refuse', "
            f"got {gate_policy!r}"
        )

    target_name = getattr(target, "name", None)
    enumerator = _ENUMERATORS.get(target_name)
    if enumerator is None:
        raise NotImplementedError(
            f"no autoscheduler enumerator registered for target {target_name!r}; "
            f"supported: {sorted(_ENUMERATORS)}"
        )
    cost_fn = get_cost(cost_name, target)
    regalloc_disabled = os.environ.get("SPMW_DISABLE_REGALLOC") == "1"

    # Whole-trace liveness pre-pass (SPEC-023 D1): run ONCE before the
    # per-group loop and thread it (via a contextvar `cross_with_knobs` reads)
    # into every group's `KnobCtx`, so the cross-kernel/cross-work-id residency
    # knob can bound its candidate set by the whole-trace analysis. The
    # per-group argmin below stays structurally intact -- liveness only bounds
    # the residency knob's candidate set; the cross-kernel decision is resolved
    # by the post-argmin reconciliation, NOT a joint search. Single-op GEMV
    # flags nothing -> residency returns ["restage"] (1x fan) -> byte-identical.
    from .spmw_liveness import trace_liveness
    from .spmw_knobs import set_active_liveness, reset_active_liveness

    liveness = trace_liveness(trace)
    _liveness_token = set_active_liveness(liveness)
    try:
        placements = _autoschedule_groups(
            target, target_name, trace, enumerator, cost_fn,
            cost_name, confidence_gate, gate_policy, regalloc_disabled,
        )
    finally:
        reset_active_liveness(_liveness_token)
    # Stamp the WORKLOAD-property cross-kernel marker `_xkernel` onto every
    # chosen placement, from liveness, INDEPENDENT of the residency knob
    # (SPEC-023 T6 win-emit, task 008-fix). A multi-op kernel whose activation
    # crosses a kernel boundary must STAGE that activation between kernels --
    # this is true for the group-local baseline too (the residency knob is
    # disabled there, but the staging is still semantically required). Codegen
    # emits the inter-kernel staging round-trip when `_xkernel` is present and
    # the value is NOT resident; the `resident` decision (search only) elides
    # it. Single-op / non-crossing traces get an empty marker -> no staging ->
    # byte-identical floor.
    _stamp_xkernel(trace, placements, liveness)
    # Post-argmin resident-pair reconciliation (SPEC-023 D1): a "resident"
    # choice is only honoured when BOTH endpoints (producer + consumer kernels)
    # selected the matching arm; otherwise fall back to restage. Structurally
    # intact per-group argmins; the cross-kernel constraint is a guard, not a
    # joint optimization.
    _reconcile_resident_pairs(trace, placements, liveness)
    return placements


def _stamp_xkernel(trace, placements, liveness) -> None:
    """Stamp `extra["_xkernel"]` = the cross-kernel memref names each placement
    touches (from whole-trace liveness), on EVERY placement -- baseline and
    search alike. This is the workload-property signal codegen uses to emit the
    inter-kernel activation staging (which `residency=resident` then elides);
    it does not depend on the residency knob, so the group-local baseline
    carries it too. Empty when nothing crosses -> byte-identical."""
    from .spmw_liveness import memref_span, crosses_boundary

    if not liveness:
        return
    bucket_funcs = [fn for fn, _ in _bucket_for_autoschedule(trace)]
    for fn, pl in zip(bucket_funcs, placements):
        xk = []
        for mref in getattr(pl, "placements", {}):
            span = memref_span(liveness, mref)
            if span is not None and span.crosses_kernel:
                xk.append(mref)
        if xk:
            new_extra = dict(getattr(pl, "extra", {}) or {})
            new_extra["_xkernel"] = sorted(xk)
            pl.extra = new_extra


def _autoschedule_groups(
    target, target_name, trace, enumerator, cost_fn,
    cost_name, confidence_gate, gate_policy, regalloc_disabled,
) -> "list[Placement]":
    """The per-group argmin loop (SPEC-023 D1: lifted into a helper so the
    whole-trace liveness pre-pass + post-argmin reconciliation wrap it without
    perturbing the loop body -- it is byte-identical to the prior inline loop).
    """
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
            chosen = candidates[scored[0][1]]
            if confidence_gate:
                # Design 07 §A4.3: the gate is a property of confidence_gate,
                # NOT of the regalloc path. Honour it here too so
                # SPMW_DISABLE_REGALLOC does not silently bypass it. Inert when
                # off (this guard); never alters the SELECTION (`chosen` is
                # already committed below).
                _check_confidence(
                    target, sub_trace, chosen, cost_name, gate_policy
                )
            placements.append(chosen)
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
        if confidence_gate:
            # Design 07 §A4.3: post-argmin gate, OPT-IN. Inert when off; never
            # alters the SELECTION (scored[0] is already committed below). It
            # only decides whether a low-confidence ranking is silently
            # committed (warn) or refused (raise).
            _check_confidence(
                target, sub_trace, scored[0][2], cost_name, gate_policy
            )
        placements.append(scored[0][2])
    return placements


def _reconcile_resident_pairs(trace, placements, liveness) -> None:
    """Post-argmin resident-pair reconciliation (SPEC-023 D1).

    A `residency == "resident"` choice is only HONOURED when both endpoints of
    the value's whole-trace live span agreed on it; otherwise it falls back to
    restage (the resident-pair saving is not credited). This keeps the
    per-group argmins independent -- the cross-kernel decision is expressed as a
    typed knob + this reconciliation guard, NOT a joint optimization.

    - Cross-WORK-ID (T4) residency lives entirely within one kernel (the
      broadcast hoist: preload once, reuse across that kernel's work-ids), so a
      single endpoint suffices -- it is honoured as chosen.
    - Cross-KERNEL (T6) residency couples a producer kernel's output to a
      consumer kernel's input: it is honoured only when the producer placement
      AND the consumer placement both selected `resident` for that memref;
      otherwise both are reverted to restage.

    Mutates `placements` in place (each is a `Placement` whose `extra` carries
    the chosen `residency` / `residency_pairs`). Byte-identical no-op when no
    placement chose resident (the regression-default, since `residency`'s
    `knob_cost` is unregistered so the argmin keeps restage).
    """
    from dataclasses import replace as _dc_replace  # noqa: F401 (kept local)

    if not liveness:
        return

    # `placements` align with `_bucket_for_autoschedule(trace)` order, so zip to
    # recover each placement's func_name WITHOUT stamping it onto `extra` (that
    # would perturb the byte-identical default). The resident-pair check uses
    # this func_name -> placement map.
    bucket_funcs = [fn for fn, _ in _bucket_for_autoschedule(trace)]
    func_to_pl = {}
    pl_func = {}
    for fn, pl in zip(bucket_funcs, placements):
        func_to_pl[fn] = pl
        pl_func[id(pl)] = fn

    def _revert_to_restage(pl, mref):
        new_extra = dict(getattr(pl, "extra", {}))
        new_extra["residency"] = "restage"
        pairs = dict(new_extra.get("residency_pairs", {}))
        pairs.pop(mref, None)
        if pairs:
            new_extra["residency_pairs"] = pairs
        else:
            new_extra.pop("residency_pairs", None)
        pl.extra = new_extra

    for pl in placements:
        extra = getattr(pl, "extra", {}) or {}
        if extra.get("residency") != "resident":
            continue
        pairs = extra.get("residency_pairs", {})
        for mref, info in list(pairs.items()):
            if not info.get("crosses_kernel"):
                continue  # T4: single-endpoint, honoured as chosen
            # T6: require the matching endpoint kernel also chose resident.
            other = info.get("consumer_func")
            if other == pl_func.get(id(pl)):
                other = info.get("producer_func")
            other_pl = func_to_pl.get(other)
            other_ok = (
                other_pl is not None
                and (getattr(other_pl, "extra", {}) or {}).get("residency")
                == "resident"
                and mref in (getattr(other_pl, "extra", {}) or {}).get(
                    "residency_pairs", {}
                )
            )
            if not other_ok:
                _revert_to_restage(pl, mref)


# The confidence bands the gate treats as "not safe to silently commit"
# (design 07 §A4.3): the coarse/placeholder CostResult flags + the
# provenance-derived "assumption" band (the least-trusted symbolic band).
_LOW_CONFIDENCE = ("coarse", "placeholder", "assumption")


def _check_confidence(
    target, sub_trace, placement, cost_name: str, gate_policy: str
) -> None:
    """Confidence gate (design 07 §A4.3). Combines TWO signals, both
    objective-independent:
      1. the chosen candidate's `CostResult.confidence` (which already
         downgrades to "coarse" on a tier-3 dynamic-trip fall-through, so the
         silent `dynamic_trip->1` is now VISIBLE here, A4<->A5); and
      2. the SYMBOLIC provenance band of the bound model
         (`provenance_band` -- "assumption" if any untrusted constant fed the
         estimate, report 28 §A4.3).
    The worse of the two governs. If it is low (coarse/placeholder/assumption),
    the gate does NOT silently commit: `gate_policy="warn"` emits a diagnostic
    and commits anyway; `gate_policy="refuse"` raises. The argmin SELECTION is
    never altered either way."""
    import warnings

    from .spmw_cost_model import evaluate, get_cost_model, provenance_band

    flavor = "faithful"
    try:
        result = evaluate(target, sub_trace, placement, flavor)
        band = provenance_band(get_cost_model(getattr(target, "name", None), flavor))
    except Exception:
        # The gate is advisory under "warn": never let a confidence probe
        # break the schedule. A real cost error would have already surfaced in
        # the argmin scoring core above.
        return
    low = [
        f"{label}={val!r}"
        for label, val in (("confidence", result.confidence), ("provenance_band", band))
        if val in _LOW_CONFIDENCE
    ]
    if not low:
        return
    name = getattr(target, "name", None)
    detail = ", ".join(low)
    msg = (
        f"autoschedule confidence_gate: chosen placement for target {name!r} "
        f"rests on a low-confidence cost estimate ({detail}; "
        f"cost_name={cost_name!r})"
    )
    if gate_policy == "refuse":
        # refuse-and-error: do NOT silently commit a low-confidence ranking.
        raise RuntimeError(
            msg + " -- refused (gate_policy='refuse'). Re-run with a "
            "sim-validated cost model or gate_policy='warn' to commit."
        )
    warnings.warn(msg + " -- committing anyway (gate_policy='warn').",
                  stacklevel=2)
