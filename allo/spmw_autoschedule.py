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
            )
        )
    return variants


def _with_weight_residency(base: "Placement", resident: bool) -> "Placement":
    """Copy `base`, setting `extra['weight_resident']`. placements untouched.

    SPEC-026 §2.2: weight residency is a *materialisation* flag (preload W
    once and reuse across the batch vs re-preload per input vector), the
    same class as `crf_issue in {shared, per_workid}`. It rides `extra`,
    not `placements` -- the bank algebra (lever 1's fibers) is unchanged.
    The cost model (205) earns the resident choice for B>=2; at B=1 the two
    variants tie (I4) and the existing argmin winner is undisturbed.
    """
    new_extra = dict(base.extra)
    new_extra["weight_resident"] = resident
    return Placement(
        placements=dict(base.placements),
        mode=_join_mode(base.mode, "wresident") if resident else base.mode,
        extra=new_extra,
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
    # axis has 8 units mapping to 16 banks (banks-per-pim = the bank
    # out-size over the pim unit count), so each pim owns the even bank
    # `2*pid` and its odd partner `2*pid + 1`. Materialising the swizzled
    # layout at `fixed={"tile": v}` evaluates the same `tile->bank-bit-0`
    # swizzle column at each fiber value: v=0 -> `2*pid` (EVEN_BANK),
    # v=1 -> `2*pid + 1` (ODD_BANK). The `+0`/`+1` fall out of the
    # swizzle column, not a pasted constant (SPEC-023 §2).
    pid = UnitId(level=1, unit=None)

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
            symbol_table={"bank": 2 * pid},
        )
        for v in range(n_fibers)
    ]
    y_even = fibers[0]

    # Candidate 1: bank-row `y` (is_auto=1 -> folded K-loop, EVEN fiber).
    bank_row = Placement(
        placements={
            x_mref: target.grf_a,
            y_mref: y_even,
            acc_mref: target.grf_b,
        },
        mode="bank_row",
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
    )
    base_candidates = [bank_row, grf_staged, dual_fiber]

    # Lever 2 (SPEC-024): cross the layout candidates with {crf, host}
    # GRF residency for the broadcastable role(s). Host residency hoists
    # the preload off the CRF stream onto the native HAB broadcast; crf
    # residency keeps the per-work-id CRF MOV. Both are real Placements;
    # argmin decides -- the enumerator must NOT prune the crf variant.
    # Host-eligibility is COMPUTED from the placement handle (§3), never
    # asserted: only memrefs landing on `grf_a` are broadcast-uniform.
    residency_candidates: list[Placement] = []
    for base in base_candidates:
        host_memrefs = _samsung_host_eligible_memrefs(
            target, base, role_to_memref
        )
        # crf variant == today's behaviour (default-missing key == "crf").
        crf = base
        for mref in host_memrefs:
            crf = _with_residency(crf, mref, "crf")
        residency_candidates.append(crf)
        # host variant: every host-eligible memref moves to host residency.
        # If no role is host-eligible there is no second variant to emit.
        if host_memrefs:
            host = base
            for mref in host_memrefs:
                host = _with_residency(host, mref, "host")
            residency_candidates.append(host)

    # Lever 3 (SPEC-025): cross every candidate with {shared, per_workid}
    # CRF-issue mode. The CRF-issue dimension is orthogonal to the y /
    # even-odd / host dimensions (SPEC-025 §3.1), so it is applied as a
    # final 2x fan-out tagged in `extra["crf_issue"]`. Argmin discards the
    # loser; the enumerator never prunes a variant or branches on shape.
    #
    # SPEC-026 §2.2: weight-residency is the final, outermost 2x tail
    # cross -- orthogonal to lever-1/2/3, exactly as lever 3 is orthogonal
    # to lever 2. Both `weight_resident in {False, True}` variants are
    # emitted UNCONDITIONALLY; the enumerator never reads B or any shape.
    # The cost model (205) earns the resident one for B>=2 and ties them
    # at B=1 (so the pre-026 B=1 winner is undisturbed, I4).
    out: list[Placement] = []
    for cand in residency_candidates:
        for crf_cand in _with_crf_modes(cand):
            out.append(_with_weight_residency(crf_cand, False))
            out.append(_with_weight_residency(crf_cand, True))
    return out


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

    # Tasklet-tiling lever (design 02 §3.2): cross each acc-placement
    # with the derived `n_tasklets` candidate set so argmin ranks the
    # parallelism. Default `n_tasklets=1` == today's behaviour (T9
    # floor). The candidate set is shape+target-derived, not a literal.
    tasklet_candidates = _upmem_tasklet_candidates(target, matches)

    layouts: list[Placement] = []
    for placements in acc_placements:
        for n_tasklets in tasklet_candidates:
            layouts.append(Placement(
                placements=dict(placements),
                extra={"n_tasklets": n_tasklets},
            ))
    return layouts


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

    candidates: list[Placement] = []
    for mode in ("sv", "sv_lookup"):
        for vr_dma in ("intra", "inter"):
            candidates.append(
                Placement(
                    placements=dict(placements),
                    mode=mode,
                    extra={
                        "vr_dma": vr_dma,
                        "n_out_tiles": n_out_tiles,
                        "n_weight_tiles": n_weight_tiles,
                        "n_stage_boundaries": n_boundaries,
                    },
                )
            )
    return candidates


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
