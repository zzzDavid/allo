# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SPMW autoscheduler — picks a target-handle assignment for a trace.

Per report 16 §Autoscheduler, this is where Linear-Layout-driven bank
assignment and PBQP register allocation will eventually live. This file
is the smallest version of that loop: enumerate candidate `Placement`s,
score each via a named cost callback, return the argmin.

The candidate enumerator is per-backend; cost callbacks come from the
the executable :class:`allo.CostSpec`. A new backend plugs in by registering
its own enumerator under the target name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

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

    `mode` is a free-form audit label only. Physical distinctions live in the
    placement handles, layout, and allowlisted structural fields in `extra`;
    schedule identity, costing, and code generation never depend on `mode`.
    `extra` is a free-form per-candidate scratch dict whose diagnostic fields
    are likewise excluded from schedule identity.
    """

    placements: dict[str, Any] = field(default_factory=dict)
    mode: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    # Chosen physical layout (SPEC-022 D3): the swizzled `LinearLayout` the
    # enumerator built to materialise the fiber handles. Carried so codegen
    # consumes the F2 layout OBJECT directly (the `range(stride)` fiber walk
    # reads `layout.size_of(fiber_axis)`) instead of pattern-matching a
    # `MemoryRef.idx`. Default `None` == today's behaviour (codegen falls back
    # to the index's own coefficient, byte-identical for Samsung stride 2).
    layout: Any = None


@dataclass(frozen=True)
class MatcherWorkScope:
    """Structural matcher scope retained independently of symbol spelling."""

    group_id: int
    work_id: tuple[int, ...]
    group_shape: tuple[int, ...] = ()
    coalesced_axes: tuple[int, ...] = ()

    def __post_init__(self):
        if self.group_id < 0:
            raise ValueError("matcher group_id must be non-negative")
        if any(value < 0 for value in self.work_id):
            raise ValueError("matcher work_id coordinates must be non-negative")
        if any(extent <= 0 for extent in self.group_shape):
            raise ValueError("matcher group_shape extents must be positive")
        if self.group_shape and len(self.work_id) != len(self.group_shape):
            raise ValueError("matcher work_id rank must match group_shape rank")
        if any(
            axis < 0 or axis >= len(self.group_shape) for axis in self.coalesced_axes
        ):
            raise ValueError("matcher coalesced axis is outside group_shape")


def _retain_matcher_work_scope(match: MatchedOp, scope: MatcherWorkScope) -> None:
    metadata = dict(match.extra)
    metadata["spmw_work_scope"] = scope
    match.extra = metadata


def _matcher_work_scope(match: MatchedOp) -> MatcherWorkScope:
    scope = match.extra.get("spmw_work_scope")
    if scope is not None:
        if not isinstance(scope, MatcherWorkScope):
            raise TypeError("spmw_work_scope must be a MatcherWorkScope")
        return scope

    return MatcherWorkScope(
        group_id=0,
        work_id=tuple(int(value) for value in match.work_id),
    )


def _matcher_search_scope(match: MatchedOp) -> tuple:
    scope = _matcher_work_scope(match)
    if scope.coalesced_axes:
        return (scope.group_id,)
    return (scope.group_id, scope.work_id)


@dataclass(frozen=True)
class MatcherPlacementDecision:
    """Allowlisted physical matcher choice used by schedule search."""

    handle_paths: tuple[tuple[int, str], ...]
    features: tuple
    layout: tuple | None


_MATCHER_PHYSICAL_EXTRA_FIELDS = (
    "operation_name",
    "bank_fanout",
    "bank_conflicts",
    "fiber_axis",
    "fibers",
    "n_fibers",
    "grf_residency",
    "crf_issue",
    "stage_resident",
    "group_size",
    "groups_per_vr",
    "subgroup_size",
    "n_out_tiles",
    "n_weight_tiles",
    "n_stage_boundaries",
    "residency",
    "residency_crossing",
    "residency_pairs",
    "tile",
    "double_buffer",
    "bank_scope",
    "batch_mapping",
    "reuse_group",
    "replica_partitions",
    "data_layout",
    "tasklets",
    "dma_bytes",
    "chunk",
    "wram_residency",
    "nc",
    "kc",
    "predicate_lowering",
)


def _freeze_matcher_feature(value, operand_roles):
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, dict):
        items = []
        for key, item in value.items():
            normalized_key = (
                ("operand", operand_roles[key])
                if key in operand_roles
                else ("feature", str(key))
            )
            items.append((normalized_key, _freeze_matcher_feature(item, operand_roles)))
        return tuple(sorted(items, key=lambda item: repr(item[0])))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_matcher_feature(item, operand_roles) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(
            sorted(
                (_freeze_matcher_feature(item, operand_roles) for item in value),
                key=repr,
            )
        )
    manifest = getattr(value, "manifest", None)
    if callable(manifest):
        return _freeze_matcher_feature(manifest(), operand_roles)
    try:
        from .perf.cost import handle_path

        return ("handle", handle_path(value))
    except (AttributeError, TypeError, ValueError):
        pass
    raise TypeError(
        f"matcher decision feature has unsupported type {type(value).__name__!r}"
    )


def _freeze_matcher_residency_relations(value, operand_roles):
    if not isinstance(value, dict):
        raise TypeError("matcher residency relations must be a mapping")
    relations = []
    for name, relation in value.items():
        if not isinstance(relation, dict):
            raise TypeError("matcher residency relation must be a mapping")
        role = operand_roles.get(name)
        if role is None:
            continue
        physical_relation = {
            key: relation[key]
            for key in ("crosses_kernel", "crosses_workid", "handle")
            if key in relation
        }
        relations.append(
            (role, _freeze_matcher_feature(physical_relation, operand_roles))
        )
    return tuple(sorted(relations, key=lambda item: item[0]))


def _freeze_matcher_tile(value):
    return (
        int(value.tile_size),
        int(value.full_bound),
        bool(value.is_identity),
    )


def _matcher_physical_features(placement, operand_roles):
    extra = getattr(placement, "extra", {}) or {}
    features = []
    for name in _MATCHER_PHYSICAL_EXTRA_FIELDS:
        if name not in extra:
            continue
        value = extra[name]
        if name in ("residency_crossing", "residency_pairs"):
            frozen = _freeze_matcher_residency_relations(value, operand_roles)
        elif name == "tile":
            frozen = _freeze_matcher_tile(value)
        else:
            frozen = _freeze_matcher_feature(value, operand_roles)
        features.append((name, frozen))
    return tuple(features)


def _matcher_operand_roles(trace, placements=()):
    roles = {}
    for match in trace.matches:
        for operand in match.operands:
            if operand.memref_name is not None and operand.memref_name not in roles:
                roles[operand.memref_name] = len(roles)
        if (
            match.result_memref_name is not None
            and match.result_memref_name not in roles
        ):
            roles[match.result_memref_name] = len(roles)
    for placement in placements:
        for name in placement.placements:
            if name not in roles:
                roles[name] = len(roles)
    return roles


def _matcher_placement_decision(trace, placement, operand_roles=None):
    from .perf.cost import handle_path

    roles = operand_roles or _matcher_operand_roles(trace, (placement,))
    handles = tuple(
        sorted(
            (
                (roles[name], handle_path(handle))
                for name, handle in placement.placements.items()
            ),
            key=lambda item: item[0],
        )
    )
    layout = (
        None
        if placement.layout is None
        else _freeze_matcher_feature(placement.layout, roles)
    )
    return MatcherPlacementDecision(
        handles,
        _matcher_physical_features(placement, roles),
        layout,
    )


class MatcherScheduledPlacements(list):
    """List-compatible result carrying whole-program search evidence."""

    def __init__(
        self,
        placements=(),
        *,
        placement_ranking=None,
    ):
        super().__init__(placements)
        self.placement_ranking = placement_ranking


def _clone_placement_value(value):
    """Clone metadata containers while preserving target-handle identity."""

    if isinstance(value, dict):
        return {key: _clone_placement_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_placement_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_placement_value(item) for item in value)
    if isinstance(value, set):
        return {_clone_placement_value(item) for item in value}
    if isinstance(value, frozenset):
        return frozenset(_clone_placement_value(item) for item in value)
    return value


def _clone_placement(placement: Placement) -> Placement:
    return Placement(
        placements=dict(placement.placements),
        mode=placement.mode,
        extra=_clone_placement_value(placement.extra),
        layout=placement.layout,
    )


def derive_layout_properties(target, placement: Placement) -> dict[str, Any]:
    """Derive backend performance/codegen facts from a carried LinearLayout.

    Explicit ``extra`` fields remain valid for non-linear or hand-authored
    placements. For AiM, layout-derived values take precedence so removing the
    F2 map changes both costing and emitted execution.
    """
    properties = dict(getattr(placement, "extra", {}) or {})
    layout = getattr(placement, "layout", None)
    if not isinstance(layout, LinearLayout):
        return properties
    target_name = getattr(target, "name", None)
    if target_name == "upmem" and "upmem_family" in properties:
        from .spmw_upmem import upmem_kernel_features

        properties.update(
            tasklet_fanout=int(properties["tasklets"]),
            tasklet_mapping=True,
            upmem_features=upmem_kernel_features(properties),
        )
    if (
        target_name == "aim"
        and {
            "lane_bank",
        }.issubset(layout.bases)
        and "bank" in layout.out_dims
    ):
        fanout = layout.image_size(varying_inputs=("lane_bank",), output_dims=("bank",))
        bank_count = int(getattr(target.banks, "banks", 1))
        if fanout not in (1, bank_count):
            raise ValueError(
                f"AiM LinearLayout bank image must be 1 or {bank_count}, got {fanout}"
            )
        properties.update(
            bank_fanout=fanout,
            bank_conflicts=layout.conflict_count(
                bank_dims=("bank",), varying_inputs=("lane_bank",)
            ),
            operation_name="MAC" if fanout == 1 else "MAC_ABK",
        )
    if (
        target_name == "apu_v1"
        and {"group", "lane_in_group"}.issubset(layout.bases)
        and "vr_lane" in layout.out_dims
    ):
        group_size = layout.size_of("lane_in_group")
        groups_per_vr = layout.image_size(
            varying_inputs=("group",), output_dims=("vr_lane",)
        )
        if group_size * groups_per_vr != layout.output_size("vr_lane"):
            raise ValueError("APU v1 group layout must cover one complete VR")
        properties.update(
            group_size=group_size,
            groups_per_vr=groups_per_vr,
            subgroup_size=1,
        )
    return properties


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


def _bucket_for_autoschedule(trace: MatchTrace) -> list[tuple[tuple, list[MatchedOp]]]:
    """Group matches by their retained structural search scope."""
    buckets: dict[tuple, list[MatchedOp]] = {}
    order: list[tuple] = []
    for m in trace.matches:
        scope = _matcher_work_scope(m)
        if scope.coalesced_axes and any(
            scope.work_id[axis] != 0 for axis in scope.coalesced_axes
        ):
            continue
        key = _matcher_search_scope(m)
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(m)
    return [(name, buckets[name]) for name in order]


def _samsung_host_eligible_memrefs(
    target, placement: "Placement", role_to_memref: dict[str, str], matches=None
) -> list[str]:
    """Return the memrefs whose GRF_A fill may be a host broadcast.

    The broadcast operand is identified structurally (release audit S-05):
    for a contraction it is the vector, the input indexed only by the
    reduction axis (``spmw_samsung.samsung_contraction_roles``), never the
    first factor in source order.  It is host-eligible iff the placement
    puts it in ``grf_a``, the register the target's HAB broadcast fills.
    Elementwise programs fill their registers from the banks only.
    """
    del role_to_memref
    from .spmw_samsung import SamsungUnsupported, samsung_contraction_roles

    macs = [m for m in (matches or ()) if m.target_op_name == "MAC"]
    if not macs:
        return []
    try:
        vector = samsung_contraction_roles(macs).vector_memref
    except SamsungUnsupported:
        return []
    handle = placement.placements.get(vector)
    if isinstance(handle, Register) and handle is target.grf_a:
        return [vector]
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
    for token, issue in (("crf_shared", "shared"), ("crf_per_workid", "per_workid")):
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
    """Enumerate Samsung placements for one work item's matches.

    Contractions (MAC) are classified structurally
    (``spmw_samsung.samsung_contraction_roles``): the matrix is the input
    indexed by the output row and the reduction axis, the vector the input
    indexed only by the reduction axis.  Base candidates differ in where the
    matrix lives: ``bank_row`` (all reduction tiles in the even bank fiber),
    ``dual_fiber`` (tiles alternate between the even and odd fiber of each
    PIM unit) and ``grf_staged`` (matrix staged into GRF_A; enumerated so the
    ranking records why it is rejected at materialization) and ``temporal``
    (the ordered TEMPORAL_REDUCE program: one output row per FP16 lane, the
    reduction issued in source order).  The vector is in GRF_A and the
    accumulator in GRF_B in every candidate.  The first three regroup the
    reduction; they are pruned before ranking when a strict reduction-order
    policy covers a floating-point accumulator (``spmw_reduction_policy``) or
    when the accumulator carries an initial value, with ``(mode, reason)``
    recorded in each survivor's ``extra["reduction_policy_pruned"]``.  Elementwise
    ops (ADD/MUL/MAD/RELU, ``spmw_samsung.samsung_elementwise``) have one
    base placement streaming both fibers.  The levers in ``spmw_knobs`` are
    crossed on top; execution semantics of every lever are documented in
    ``docs/release/design/samsung-execution.md``.
    """
    from .spmw_reduction_policy import may_regroup, strict_policies
    from .spmw_samsung import (
        SAMSUNG_ELEMENTWISE_OPS,
        SamsungUnsupported,
        samsung_contraction_roles,
        samsung_elementwise,
        samsung_target_facts,
    )

    facts = samsung_target_facts(target)
    role_to_memref = _trace_memrefs_by_role(matches)
    ops = {m.target_op_name for m in matches}

    # Fiber handles from the swizzled F2 layout (SPEC-022 D3): the segment
    # (tile) axis has banks-per-pim values; materialising it at each value
    # yields the bank `stride*pid + v` of that fiber.
    banks_per_pim = facts.banks_per_pim
    base = LinearLayout.identity(
        {"grf": facts.grf_a_entries, "bank": facts.banks, "tile": banks_per_pim},
        out_dims=("grf", "bank"),
        allow_zero=True,
    )
    swizzled = LinearLayout.optimal_swizzle(
        base,
        vec_dims=("grf",),
        bank_dims=("bank",),
        segment_dims=("tile",),
    )
    pid = UnitId(level=1, unit=None)
    bank_stride = _bank_stride_per_pim(target, swizzled)
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

    if ops == {"MAC"}:
        roles = samsung_contraction_roles(matches)
        matrix, vector, acc = roles.matrix_memref, roles.vector_memref, roles.acc_memref
        bank_row = Placement(
            placements={vector: target.grf_a, matrix: y_even, acc: target.grf_b},
            mode="bank_row",
            extra={"fiber_axis": "tile"},
            layout=swizzled,
        )
        grf_staged = Placement(
            placements={vector: target.grf_a, matrix: target.grf_a, acc: target.grf_b},
            mode="grf_staged",
            layout=swizzled,
        )
        dual_fiber = Placement(
            placements={vector: target.grf_a, matrix: y_even, acc: target.grf_b},
            mode="dual_fiber",
            extra={"fibers": list(fibers), "fiber_axis": "tile", "n_fibers": n_fibers},
            layout=swizzled,
        )
        # Ordered program (TEMPORAL_REDUCE): FP16 lane l of a PIM unit owns
        # output row l of the unit's row block, the reduction walks time in
        # source order in one GRF_B entry, no partial sums exist.  Its layout
        # maps lanes to rows (not to reduction elements), which is also what
        # distinguishes it from bank_row in the decision key.
        temporal = Placement(
            placements={vector: target.grf_a, matrix: y_even, acc: target.grf_b},
            mode="temporal",
            extra={"reduction_program": "temporal"},
            layout=LinearLayout.identity(
                {"lane": facts.lanes, "bank": facts.banks}, out_dims=("lane", "bank")
            ),
        )
        # The REDUCE program accumulates reduction element k in lane k mod
        # `lanes` and the runtime sums the lane partials in FP32 (plus the
        # fiber split for dual_fiber): it regroups the reduction.
        lane_split = (
            f"accumulates reduction element k in FP16 lane k mod {facts.lanes} "
            f"of GRF_B and sums the {facts.lanes} lane partial sums in FP32 on "
            "the host"
        )
        regrouping = {
            "bank_row": lane_split,
            "grf_staged": lane_split,
            "dual_fiber": lane_split
            + f" after splitting the reduction tiles over {n_fibers} bank fibers",
        }
        base_candidates = [bank_row, grf_staged, dual_fiber, temporal]
        pruned = []
        if roles.accumulator_init == "carried":
            # Only the ordered program loads an initial accumulator (its first
            # CRF word moves it from the bank into GRF_B).
            pruned += [
                (c.mode, "the REDUCE program zeroes GRF_B and cannot carry the "
                 "accumulator's initial value")
                for c in base_candidates
                if c.mode in regrouping
            ]
        elif not may_regroup(matches):
            policy = strict_policies(matches)[0]
            pruned += [
                (c.mode, f"{policy.describe()}; the REDUCE program {regrouping[c.mode]}")
                for c in base_candidates
                if c.mode in regrouping
            ]
        dropped = {mode for mode, _reason in pruned}
        base_candidates = [c for c in base_candidates if c.mode not in dropped]
        for candidate in base_candidates:
            candidate.extra["reduction_policy_pruned"] = tuple(pruned)
    elif len(ops) == 1 and ops <= SAMSUNG_ELEMENTWISE_OPS:
        el = samsung_elementwise(matches)
        placements = {el.input_memrefs[0]: target.grf_a}
        if len(el.input_memrefs) > 1:
            placements[el.input_memrefs[1]] = y_even
        if len(el.input_memrefs) > 2:
            placements[el.input_memrefs[2]] = target.grf_b
        placements[el.output_memref] = target.grf_a
        base_candidates = [
            Placement(
                placements=placements,
                mode="all_bank",
                extra={"fibers": list(fibers), "fiber_axis": "tile", "n_fibers": n_fibers},
                layout=swizzled,
            )
        ]
    else:
        raise SamsungUnsupported(
            "samsung enumerator: a kernel must be one contraction (MAC) or one "
            f"elementwise op ({sorted(SAMSUNG_ELEMENTWISE_OPS)}); got {sorted(ops)}"
        )

    from .spmw_knobs import cross_with_knobs

    return cross_with_knobs(target, base_candidates, matches, role_to_memref)


def _aim_layout_candidates(target, matches: list[MatchedOp]) -> list[Placement]:
    """Construct the analyzable SBK and ABK layouts for SK-Hynix AiM.

    Per spec 013 §D.1, the AiM A/B choice is "does the layout factor `bank`
    in as an input dim?". The two algebraic constructions are:

      no_bank_layout  = identity({"k": K})       -- per-bank MAC (MAC_SBK)
      all_bank_layout = no_bank ⊗ identity({"bank": NBANKS})  -- broadcast (MAC_ABK)

    Both materialise to the same ``target.banks[4*bg + bank]`` handle for
    operands that live in DRAM.  The selected placement explicitly names
    either the bank-scoped ``MAC`` or channel-scoped ``MAC_ABK`` primitive;
    both accumulate into the physical MAC register file.
    """
    from .pim.aim_lowering import mac_operands

    mac_matches = [match for match in matches if match.target_op_name == "MAC"]
    # AF also binds an ``x`` role; only MAC roles name the contraction operands.
    role_to_memref = _trace_memrefs_by_role(mac_matches or matches)
    acc_mref = role_to_memref.get("acc")
    if not mac_matches or acc_mref is None:
        raise NotImplementedError(
            "aim enumerator: trace is missing one of x/y/acc roles; "
            f"got {sorted(role_to_memref)}"
        )
    matrix, vector = mac_operands(mac_matches[0])
    x_mref = matrix.memref_name
    y_mref = vector.memref_name
    input_source = "banks" if x_mref == y_mref else "gb"

    # Device grouping nodes are transparent to work coordinates, so the
    # spatial levels are channel=0, bank_group=1, bank=2.
    bg_id = UnitId(level=1, unit=target.unit("bank_group"))
    bk_id = UnitId(level=2, unit=target.unit("bank"))
    banks = target.banks
    gb = target.gb
    mac_reg = target.mac_reg

    reduction = max(1, int(_trace_reduction_trip(matches) or 1))
    reduction_span = 1 << (reduction - 1).bit_length()
    bank_count = int(getattr(banks, "banks", 1))
    if bank_count <= 0 or bank_count & (bank_count - 1):
        raise ValueError("AiM LinearLayout requires a power-of-two bank count")

    # The same logical work-bank coordinate feeds both candidates.  The
    # lane_bank input is either projected to zero (all reduction lanes collide
    # on one bank) or mapped identically (one lane per physical bank).  Thus
    # bank fanout, conflicts, selected primitive, cost width, and generated
    # address scope all derive from the carried F2 map.
    zero = (0,)
    work_bank_basis = [(1 << bit,) for bit in range(bank_count.bit_length() - 1)]
    reduction_basis = [zero for _ in range(reduction_span.bit_length() - 1)]
    lane_zero_basis = [zero for _ in range(bank_count.bit_length() - 1)]
    lane_bank_basis = list(work_bank_basis)
    single_bank_layout = LinearLayout(
        bases={
            "k": reduction_basis,
            "work_bank": work_bank_basis,
            "lane_bank": lane_zero_basis,
        },
        out_dims=("bank",),
        out_sizes=(bank_count,),
    )
    all_bank_layout = LinearLayout(
        bases={
            "k": reduction_basis,
            "work_bank": work_bank_basis,
            "lane_bank": lane_bank_basis,
        },
        out_dims=("bank",),
        out_sizes=(bank_count,),
    )

    def properties(layout):
        fanout = layout.image_size(varying_inputs=("lane_bank",), output_dims=("bank",))
        conflicts = layout.conflict_count(
            bank_dims=("bank",), varying_inputs=("lane_bank",)
        )
        if fanout not in (1, bank_count):
            raise ValueError(
                f"AiM supports bank fanout 1 or {bank_count}, got {fanout}"
            )
        return fanout, conflicts

    single_fanout, single_conflicts = properties(single_bank_layout)
    all_fanout, all_conflicts = properties(all_bank_layout)
    bank_handle = materialise_handle(
        single_bank_layout,
        target=target,
        out_dim="bank",
        fixed={"k": 0, "lane_bank": 0},
        symbol_table={"work_bank": 4 * bg_id + bk_id},
        handle_table={"bank": banks},
    )

    def operand_placements(matrix_handle):
        # A bank-pair MAC reads both inputs from banks; the dict keeps one
        # entry per memref, so the shared memref must map to the bank handle.
        if input_source == "banks":
            return {x_mref: matrix_handle, acc_mref: mac_reg}
        return {x_mref: matrix_handle, y_mref: gb, acc_mref: mac_reg}

    # The contraction shape rides on every base layout, so a placement passed
    # as ``layout=`` is priced by the same lowered-record cost program as an
    # enumerated (knob-crossed) one.
    shape = _aim_contraction_shape(target, matches)
    # Banks one SBK launch visits (AimCtx._sbk_banks order), for the cost.
    sbk_banks = []
    for work_bank in range(single_bank_layout.size_of("work_bank")):
        for lane_bank in range(single_bank_layout.size_of("lane_bank")):
            bank = single_bank_layout.apply(
                k=0, work_bank=work_bank, lane_bank=lane_bank
            )[0]
            if bank not in sbk_banks:
                sbk_banks.append(bank)

    layouts: list[Placement] = []
    # Candidate 1: per-bank MAC (no_bank_layout -- y stays in its bank).
    layouts.append(
        Placement(
            placements=operand_placements(bank_handle),
            mode="single_bank",
            extra={
                "operation_name": "MAC" if single_fanout == 1 else "MAC_ABK",
                "bank_fanout": single_fanout,
                "bank_conflicts": single_conflicts,
                "input_source": input_source,
                "aim_shape": dict(shape),
                "sbk_banks": tuple(sbk_banks),
            },
            layout=single_bank_layout,
        )
    )
    # Candidate 2: all-bank-broadcast MAC (all_bank_layout -- y rides gb).
    layouts.append(
        Placement(
            placements=operand_placements(banks),
            mode="all_bank",
            extra={
                "operation_name": "MAC" if all_fanout == 1 else "MAC_ABK",
                "bank_fanout": all_fanout,
                "bank_conflicts": all_conflicts,
                "input_source": input_source,
                "aim_shape": dict(shape),
            },
            layout=all_bank_layout,
        )
    )
    return layouts


def _aim_contraction_shape(target, matches: list[MatchedOp]) -> dict:
    """Shape of the bucket's first MAC match, as the knobs and cost see it.

    Replication comes from the group stamp of ``rank_matcher_placements``;
    without it, ``matches`` must hold every bucket of the group, whose body
    fingerprints decide it (as codegen does).
    """
    import math

    from .pim.aim_lowering import (
        contraction_shape,
        group_replicated,
        group_replication,
    )

    mac = next(match for match in matches if match.target_op_name == "MAC")
    try:
        replicated, buckets = group_replication(mac)
    except ValueError:
        group_id = _matcher_work_scope(mac).group_id
        group = [m for m in matches if _matcher_work_scope(m).group_id == group_id]
        shape_of = tuple(getattr(mac.extra.get("spmw_work_scope"), "group_shape", ()))
        buckets = math.prod(shape_of) if shape_of else 1
        if len({m.func_name for m in group}) != buckets:
            raise
        replicated = group_replicated(group)
    shape = contraction_shape(
        mac,
        replicated=replicated,
        replicas=buckets if replicated else 1,
    )
    acc = next(op for op in mac.operands if op.role == "acc")
    shape["activation"] = any(
        other.target_op_name == "AF"
        and other.func_name == mac.func_name
        and any(
            op.role == "x" and op.memref_name == acc.memref_name
            for op in other.operands
        )
        for other in matches
    )
    return shape


def _aim_elementwise_base(target, matches: list[MatchedOp], name: str) -> Placement:
    """The single base placement of an AiM MUL (banks) or ADD (gpr) kernel."""
    home = target.banks if name == "MUL" else target.gpr
    placements = {}
    for match in matches:
        for operand in match.operands:
            if operand.memref_name is not None:
                placements.setdefault(operand.memref_name, home)
        if match.result_memref_name is not None:
            placements.setdefault(match.result_memref_name, home)
    return Placement(
        placements=placements,
        mode="elementwise",
        extra={"operation_name": name},
    )


@register_enumerator("aim")
def _aim_enumerate(target, matches: list[MatchedOp]) -> list[Placement]:
    """Return AiM placements crossed with the registered AiM knobs.

    MAC kernels start from the SBK and ABK layouts; ``AimCtx.emit_program``
    realizes either one (a bank fanout of 1 expands each lowered ``MAC_ABK``
    into per-bank ``MAC_SBK`` lines taken from the LinearLayout). MUL and ADD
    kernels have one base placement each.
    """
    from .spmw_knobs import cross_with_knobs

    names = {match.target_op_name for match in matches}
    if "MAC" in names:
        shape = _aim_contraction_shape(target, matches)
        bases = []
        for candidate in _aim_layout_candidates(target, matches):
            candidate.extra["aim_shape"] = dict(shape)
            bases.append(candidate)
        role_to_memref = _trace_memrefs_by_role(
            [match for match in matches if match.target_op_name == "MAC"]
        )
    elif "MUL" in names or "ADD" in names:
        name = "MUL" if "MUL" in names else "ADD"
        bases = [_aim_elementwise_base(target, matches, name)]
        role_to_memref = {}
    else:
        raise NotImplementedError(
            "aim enumerator: kernel has no MAC, MUL or ADD match; "
            f"got {sorted(names)}"
        )
    return cross_with_knobs(target, bases, matches, role_to_memref)


def _trace_reduction_trip(matches: list[MatchedOp]) -> int | None:
    """Innermost enclosing-loop bound of an accumulating MAC match.

    AiM uses this shape-derived extent when constructing the logical reduction
    axis of its bank layout. Returns ``None`` when the trace has no statically
    resolved reducing loop.
    """
    from .spmw_tripcount import _parse_loop_bound

    for match in matches:
        if match.target_op_name != "MAC":
            continue
        if not match.enclosing_loops:
            continue
        bound = _parse_loop_bound(match.enclosing_loops[-1][2])
        if bound is not None:
            return bound
    return None


def _apu_v1_vr_tiling(target, trace: MatchTrace) -> tuple[int, int, int]:
    """Derive APU VR tile counts from target and loop geometry."""
    import math

    from .spmw_tripcount import resolve_bound_text, resolve_trip_count

    lane_width = max(1, int(target.vr0.lanes or 1))
    mapping_env = {}
    for unit_spec in target._walk():
        extent = 1
        for factor in unit_spec.mapping:
            extent *= factor
        mapping_env[unit_spec.name] = extent

    n_out = 1
    weight_elements = 0
    n_macs = 0
    max_reduction = 1
    for match in trace.matches:
        if match.target_op_name == "MAC":
            n_macs += 1
        loops = match.enclosing_loops or []
        if not loops:
            continue
        reduction = resolve_trip_count(match, -1, mapping_env=mapping_env) or 1
        max_reduction = max(max_reduction, int(reduction))
        rows = 1
        for _name, _lower, upper, _step in loops[:-1]:
            bound = resolve_bound_text(upper, mapping_env=mapping_env)
            if bound is not None:
                rows *= bound
        n_out = max(n_out, rows)
        weight_elements += rows * reduction
    scope = _matcher_work_scope(trace.matches[0]) if trace.matches else None
    authored_groups = (
        int(scope.group_shape[0]) if scope is not None and scope.group_shape else 1
    )
    groups_per_vr = authored_groups
    return (
        max(1, math.ceil(n_out / groups_per_vr)),
        max(1, math.ceil(weight_elements / lane_width)),
        max(0, n_macs - 1),
    )


def _bank_stride_per_pim(target, layout: LinearLayout) -> int:
    """Banks-per-pim stride: the layout's `bank` axis size over the `pim`
    unit fanout (16 // 8 == 2 for Samsung).  Each pim owns `stride`
    contiguous banks based at `stride*pid`.  Raises when the geometry is
    not declared instead of guessing a stride (release audit S-09)."""
    from math import prod

    bank_out = layout.size_of("bank") if "bank" in layout.bases else None
    pim_units = None
    for u in target._walk():
        if u.name == "pim":
            pim_units = prod(u.mapping) if u.mapping else None
            break
    if not bank_out or not pim_units or bank_out % pim_units:
        raise ValueError(
            f"cannot derive the banks-per-pim stride: bank axis {bank_out}, "
            f"pim units {pim_units}"
        )
    return bank_out // pim_units


@register_enumerator("upmem")
def _upmem_enumerate(target, matches: list[MatchedOp]) -> list[Placement]:
    """UPMEM physical decisions as base layouts crossed with the seven knobs.

    Each base placement is one data layout of the matched kernel family; the
    knobs fan out only the catalog decisions the family template realizes,
    and the last one attaches the masked 12-of-16 tasklet carrier (spec 003
    U5).  A kernel no catalog decision realizes raises with every decision's
    reason (``spmw_upmem.upmem_base_placements``) instead of yielding no
    candidates.
    """
    from .spmw_knobs import cross_with_knobs
    from .spmw_reduction_policy import may_regroup, no_strict_program_error
    from .spmw_upmem import upmem_base_placements

    bases = upmem_base_placements(target, matches)
    # Every family splits its reduction into tasklet partials (and nc/kc
    # tiles); its accumulators are int32/int64 (UPMEM_ROLE_ELEMENT_TYPES),
    # which a strict policy lets regroup. Classification rejects other types,
    # so this guards the policy should a non-integer family appear.
    if not may_regroup(matches):
        raise no_strict_program_error(
            "upmem",
            matches,
            "UPMEM programs combine tasklet and nc/kc partial sums",
        )
    # UPMEM knobs never read roles; a scaled GEMV binds role "x" to both the
    # matrix (MAC) and the accumulator (SCALE), which the role map rejects.
    return cross_with_knobs(
        target,
        bases,
        matches,
        {},
    )


@register_enumerator("apu_v1")
def _apu_v1_enumerate(target, matches: list[MatchedOp]) -> list[Placement]:
    """Build the group view used by a GVML reduction.

    ``lane_in_group`` is the padded reduction axis and ``group`` enumerates
    independent outputs. Both map contiguously onto the target-declared 32K
    VR lane axis. A single GVML call processes every group concurrently.
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

    reduction = max(1, int(_trace_reduction_trip(matches) or 1))
    vr_lanes = int(target.vr0.axes["lane"])
    scope = _matcher_work_scope(matches[0])
    group_count = int(scope.group_shape[0]) if scope.group_shape else 0
    if (
        group_count <= 0
        or group_count > vr_lanes
        or group_count & (group_count - 1)
        or vr_lanes % group_count
    ):
        raise ValueError(
            "APU v1 scalar mapping must be a power-of-two divisor of "
            f"{vr_lanes}, got {group_count}"
        )
    group_size = vr_lanes // group_count
    if reduction > group_size:
        raise ValueError(
            f"APU v1 reduction extent {reduction} exceeds the {group_size}-lane "
            f"group selected by mapping={group_count}"
        )
    groups_per_vr = group_count
    group_bits = groups_per_vr.bit_length() - 1
    lane_bits = group_size.bit_length() - 1
    layout = LinearLayout(
        bases={
            "group": [((group_size << bit),) for bit in range(group_bits)],
            "lane_in_group": [((1 << bit),) for bit in range(lane_bits)],
        },
        out_dims=("vr_lane",),
        out_sizes=(vr_lanes,),
    )
    placements = {
        x_mref: target.vr0,
        y_mref: target.vr1,
        acc_mref: target.vr2,
    }

    # VR-tile arithmetic from operand shape + target.vrs (design 01 §4.1).
    # The tile counts are identical for both vr_dma modes (they share the
    # shape); only how the moves are issued (per-tile re-fetch vs reuse)
    # differs, and the cost model prices that difference.
    trace = MatchTrace(
        target_name="apu_v1", module_name="<enumerate>", matches=list(matches)
    )
    n_out_tiles, n_weight_tiles, n_boundaries = _apu_v1_vr_tiling(target, trace)

    return [
        Placement(
            placements=placements,
            mode="grouped_f16",
            extra={
                "n_out_tiles": n_out_tiles,
                "n_weight_tiles": n_weight_tiles,
                "n_stage_boundaries": n_boundaries,
            },
            layout=layout,
        )
    ]


# --------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------- #


def autoschedule(
    target,
    trace: MatchTrace,
    cost,
    *,
    host_moves=(),
    buffer_metrics=None,
    movement_policy=None,
) -> list[Placement]:
    """Pick one `Placement` per `@allo.work` kernel in `trace`.

    Returns a list aligned with `_bucket_for_autoschedule(trace)`:
    ``result[i]`` is the chosen placement for the i-th kernel (in
    trace order). For single-kernel workloads this list has length 1.

    Every candidate is lowered through the same executable cost program used
    by the virtual backend. Target declarations never supply timing data.
    Selection ranks one decision per kernel (`rank_matcher_placements`).
    ``movement_policy`` (a ``spmw_layout_problem.MovementSelectionPolicy``)
    configures the layout-construction stage; ``None`` uses the policy of
    ``spmw_layout_problem.movement_selection_policy`` or the default.
    """
    target_name = getattr(target, "name", None)
    enumerator = _ENUMERATORS.get(target_name)
    if enumerator is None:
        raise NotImplementedError(
            f"no autoscheduler enumerator registered for target {target_name!r}; "
            f"supported: {sorted(_ENUMERATORS)}"
        )
    if cost is None:
        raise TypeError("autoschedule() requires an executable CostSpec")

    # The executable cost program is shared by autoscheduling and virtual
    # execution. It interprets each candidate over concrete target handles.
    from .perf import CostSpec

    bound_cost = cost.bind(target) if isinstance(cost, CostSpec) else cost

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

    # Placement legality (release audit C-01, `spmw_legality`): candidates the
    # target description does not permit never reach ranking. Each is
    # recorded as a rejection of its kernel with position None (it never
    # entered the choice set).
    from .spmw_legality import IllegalPlacement, check_placement

    from dataclasses import replace

    from .spmw_legality import LegalityVerdict

    # group id -> reason (without work-item names) -> rejected bucket count
    illegal: dict[int, dict[str, int]] = {}

    def legal_enumerator(target_, matches):
        candidates = enumerator(target_, matches)
        kept, verdicts = [], []
        for candidate in candidates:
            verdict = check_placement(target_, matches, candidate)
            if verdict.legal:
                kept.append(candidate)
                continue
            verdicts.append(verdict)
            anonymous = LegalityVerdict(
                tuple(replace(v, func_name=None) for v in verdict.violations)
            )
            reasons = illegal.setdefault(_matcher_work_scope(matches[0]).group_id, {})
            reason = f"illegal placement {candidate.mode!r}: {anonymous.describe()}"
            reasons[reason] = reasons.get(reason, 0) + 1
        if candidates and not kept:
            raise IllegalPlacement(
                LegalityVerdict(tuple(v for vd in verdicts for v in vd.violations)),
                f"placement: every enumerated candidate for {matches[0].func_name!r}",
            )
        return kept

    liveness = trace_liveness(trace)
    _liveness_token = set_active_liveness(liveness)
    try:
        result = rank_matcher_placements(
            target,
            trace,
            legal_enumerator,
            bound_cost,
            host_moves=host_moves,
            buffer_metrics=buffer_metrics,
            liveness=liveness,
            movement_policy=movement_policy,
        )
    finally:
        reset_active_liveness(_liveness_token)
    ranking = getattr(result, "placement_ranking", None)
    if illegal and ranking is not None:
        result.placement_ranking = replace(
            ranking,
            kernels=tuple(
                replace(
                    kernel,
                    rejections=tuple(kernel.rejections)
                    + tuple(
                        (None, f"{reason} ({count} work item(s))")
                        for reason, count in illegal.get(kernel.group_id, {}).items()
                    ),
                )
                for kernel in ranking.kernels
            ),
        )
    return result


#: Deterministic tie-break among equal-cycle candidates (release audit C-06).
TIE_BREAK_RULE = (
    "cycles, then the structural decision key: the carried layout's exact F2 "
    "map (backend layout x movement layout), then the allowlisted knob and "
    "physical feature values, then the operand handle paths (all rendered as "
    "text, names replaced by operand roles); the candidate position only "
    "separates candidates whose keys are identical"
)


@dataclass(frozen=True)
class KernelRanking:
    """Per-kernel record of the ranked placement selection.

    ``movement`` is the layout-construction record
    (``spmw_layout_problem.KernelMovement``): derived problem, search
    termination, tie classes, the movement of the chosen layout and its label.
    ``cost_ties`` lists every scored candidate at the winner's cycles, in
    tie-break order (``TIE_BREAK_RULE``); its first entry is the winner unless
    the winner's codegen probe rejected the earlier ones.
    ``movement_rejections`` holds ``(position, reason)`` for every movement
    tie crossed with a key that the backend cannot realize; ``rejections``
    keeps the cost-rule and codegen-probe rejections.
    """

    group_id: int
    bucket_count: int
    fallback: str | None
    # One ``(position in the kernel choice set, cycles, mode)`` per key, in
    # rank order. Empty for the per-bucket fallback.
    scores: tuple[tuple[int, int, str], ...]
    chosen_position: int | None
    rejections: tuple[tuple[int, str], ...]
    movement: Any = None
    cost_ties: tuple[tuple[int, int, str], ...] = ()
    tie_break: str = TIE_BREAK_RULE
    movement_rejections: tuple[tuple[int, str], ...] = ()
    # ``(mode, reason)`` of base placements an enumerator pruned before
    # ranking (``extra["reduction_policy_pruned"]`` of the first bucket).
    policy_pruned: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class PlacementRanking:
    """Ranking record of a whole trace.

    ``incomplete`` is True when some kernel's movement search ended at its
    node budget: its ties are the best layouts found, not proven minima.
    """

    kernels: tuple[KernelRanking, ...]
    incomplete: bool = False

    @property
    def movement(self) -> tuple:
        return tuple(kernel.movement for kernel in self.kernels)


def _bucket_decision_map(target, trace, enumerator, matches):
    """Enumerate one bucket; return its sub-trace, candidates, and the
    first-occurrence ``decision key -> candidate`` map in enumeration order."""

    candidates = enumerator(target, matches)
    if not candidates:
        func_name = matches[0].func_name if matches else "<empty>"
        raise RuntimeError(
            f"no candidate layouts produced for kernel {func_name!r} "
            f"on target {getattr(target, 'name', None)!r}"
        )
    bucket_trace = MatchTrace(
        target_name=trace.target_name,
        module_name=trace.module_name,
        matches=matches,
    )
    roles = _matcher_operand_roles(bucket_trace, candidates)
    key_map = {}
    for candidate in candidates:
        key = _matcher_placement_decision(bucket_trace, candidate, roles)
        key_map.setdefault(key, candidate)
    return bucket_trace, candidates, key_map


def _stamp_group_replication(matches: list[MatchedOp]) -> None:
    """Record whether a kernel group's buckets are replicas of one body.

    Enumerators see one bucket; this whole-group fact lets them size
    per-replica problems. Needs every match's retained body fingerprint and
    stamps nothing otherwise.
    """
    fingerprints = [match.extra.get("spmw_body_fingerprint") for match in matches]
    if not fingerprints or None in fingerprints:
        return
    replicated = len(set(fingerprints)) == 1
    for match in matches:
        match.extra["spmw_group_replicated"] = replicated


def _movement_entries(movement, policy):
    """Layout-construction candidates of one kernel from its movement record.

    Each entry is ``(movement layout or None, label, cost, realizable,
    reason, tie index)``.  Analyzed kernels contribute every tie class (up to
    ``policy.max_tie_candidates``) and, when no tie class contains it, the
    program's own layout marked ``not-minimal``.  Unanalyzed kernels keep
    the backend's hand-built layouts with the ``not-analyzed`` reason.
    """
    from .spmw_layout_problem import label_for_cost

    if movement.status == "disabled":
        return [(None, None, None, True, None, None)], 0
    if movement.status != "analyzed":
        return [(None, movement.label, None, True, None, None)], 0
    crossed = movement.ties[: policy.max_tie_candidates]
    entries = [
        (
            tie.layout,
            label_for_cost(movement, tie.cost),
            tie.cost,
            tie.realizable,
            tie.reason,
            tie.index,
        )
        for tie in crossed
    ]
    if not any(tie.realizable for tie in movement.ties):
        entries.append(
            (
                movement.program_layout,
                movement.program_label,
                movement.program_cost,
                True,
                None,
                None,
            )
        )
    # Realizable entries first: they keep the positions of the backend keys.
    entries.sort(key=lambda entry: not entry[3])
    return entries, len(crossed)


def _with_movement(template, layout, label, cost):
    """Copy ``template`` carrying a movement layout and its label.

    The movement layout joins the backend's own ``LinearLayout`` as a
    block-diagonal ``product`` (disjoint ``mm.`` dims), so backend readers of
    their dims see the same map and the decision key includes the movement
    layout.  A template without a ``LinearLayout`` keeps ``layout`` as is.
    """
    placement = _clone_placement(template)
    if label is None:
        return placement
    extra = dict(placement.extra)
    extra["movement"] = label
    if cost is not None:
        extra["movement_cost"] = str(cost)
    if layout is not None and isinstance(placement.layout, LinearLayout):
        placement.layout = placement.layout.product(layout)
    placement.extra = extra
    return placement


def _structural_decision_key(bucket_trace, placement) -> tuple[str, str, str]:
    """Tie-break key of one realized candidate (``TIE_BREAK_RULE``)."""
    decision = _matcher_placement_decision(bucket_trace, placement)
    return (repr(decision.layout), repr(decision.features), repr(decision.handle_paths))


def rank_matcher_placements(
    target,
    trace,
    enumerator,
    bound_cost,
    *,
    host_moves=(),
    buffer_metrics=None,
    liveness=None,
    movement_policy=None,
) -> MatcherScheduledPlacements:
    """Choose one placement decision per `@allo.work` kernel by cost rank.

    The kernel choice set is the decision keys every bucket of the kernel
    enumerates.  Layout construction (``spmw_layout_problem``) then derives
    the kernel's minimum-movement problem, runs the search once and crosses
    every tie class with every key, so movement ties reach the cost ranking;
    the program's own layout is kept, marked ``not-minimal``, when no tie
    class contains it.  A tie the backend cannot realize is rejected before
    emission with the reason.  When the problem cannot be derived the keys
    keep their hand-built layouts, labelled ``not-analyzed: <reason>``.

    Candidates are scored over the kernel's stamped placements and ranked by
    cycles, then by the structural key of ``TIE_BREAK_RULE``; the first
    whose codegen is feasible wins.  Kernels with an empty choice set fall
    back to each bucket's own stable argmin.  Cross-kernel residency is
    reconciled once afterwards.
    """

    from . import spmw_plan
    from .pim.schedule_search import InfeasibleSchedule
    from .spmw_codegen import _materialize_matcher_codegen, _stamp_host_moves
    from .spmw_layout_problem import (
        active_movement_policy,
        analyze_kernel_movement,
        with_chosen,
    )

    policy = active_movement_policy(movement_policy)
    buckets = _bucket_for_autoschedule(trace)
    kernels: dict[int, list[int]] = {}
    for index, (_scope, matches) in enumerate(buckets):
        group_id = _matcher_work_scope(matches[0]).group_id
        kernels.setdefault(group_id, []).append(index)

    chosen: dict[int, Placement] = {}
    records = []

    def realize(kernel_trace, templates):
        placements = [_clone_placement(template) for template in templates]
        for placement in placements:
            placement.extra = derive_layout_properties(target, placement)
        _stamp_xkernel(kernel_trace, placements, liveness)
        if host_moves:
            _stamp_host_moves(placements, host_moves)
        return placements

    for group_id, bucket_indices in kernels.items():
        kernel_trace = MatchTrace(
            target_name=trace.target_name,
            module_name=trace.module_name,
            matches=[
                match for index in bucket_indices for match in buckets[index][1]
            ],
        )
        _stamp_group_replication(kernel_trace.matches)
        infos = [
            _bucket_decision_map(target, trace, enumerator, buckets[index][1])
            for index in bucket_indices
        ]
        choice_set = [
            key
            for key in infos[0][2]
            if all(key in key_map for _t, _c, key_map in infos[1:])
        ]
        movement = analyze_kernel_movement(
            target, kernel_trace.matches, bound_cost, policy
        )

        if not choice_set:
            templates = []
            for bucket_trace, candidates, _key_map in infos:
                best = _per_bucket_argmin_index(
                    target,
                    bucket_trace,
                    candidates,
                    bound_cost,
                    host_moves=host_moves,
                    buffer_metrics=buffer_metrics,
                )
                templates.append(candidates[best])
            if movement.status == "analyzed":
                label = movement.program_label
                cost = movement.program_cost
            else:
                label = None if movement.status == "disabled" else movement.label
                cost = None
            placements = realize(
                kernel_trace,
                [_with_movement(t, None, label, cost) for t in templates],
            )
            for index, placement in zip(bucket_indices, placements):
                chosen[index] = placement
            records.append(
                KernelRanking(
                    group_id=group_id,
                    bucket_count=len(bucket_indices),
                    fallback="per_bucket",
                    scores=(),
                    chosen_position=None,
                    rejections=(),
                    policy_pruned=tuple(
                        infos[0][1][0].extra.get("reduction_policy_pruned", ())
                    ),
                    movement=with_chosen(
                        movement,
                        cost,
                        label,
                        notes=movement.notes
                        + (
                            "per-bucket fallback: the buckets share no decision "
                            "key, so movement layouts were not crossed",
                        ),
                    ),
                )
            )
            continue

        entries, ties_crossed = _movement_entries(movement, policy)
        notes = list(movement.notes)
        if movement.tie_count > ties_crossed and movement.status == "analyzed":
            notes.append(
                f"{movement.tie_count - ties_crossed} of {movement.tie_count} tie "
                f"classes exceed max_tie_candidates={policy.max_tie_candidates} "
                "and were not crossed"
            )
        if movement.status == "analyzed" and not any(
            isinstance(infos[0][2][key].layout, LinearLayout) for key in choice_set
        ):
            notes.append(
                "the backend's placements carry no LinearLayout; the movement "
                "layout is recorded on the ranking only"
            )

        scored = []
        realized = {}
        rejections = []
        movement_rejections = []
        candidate_entry = {}
        candidate_mode = {}
        position = -1
        for entry in entries:
            layout, label, cost, realizable, reason, tie_index = entry
            for key in choice_set:
                position += 1
                candidate_mode[position] = infos[0][2][key].mode
                candidate_entry[position] = entry
                if not realizable:
                    movement_rejections.append(
                        (
                            position,
                            f"movement tie {tie_index} rejected before emission: "
                            f"{reason}",
                        )
                    )
                    continue
                placements = realize(
                    kernel_trace,
                    [
                        _with_movement(key_map[key], layout, label, cost)
                        for _t, _c, key_map in infos
                    ],
                )
                # A cost rule may find a knob combination unrealizable; it is
                # dropped like a codegen-probe rejection (spec 001 D1 rule 2).
                try:
                    graph = spmw_plan.build_execution_graph(
                        target,
                        kernel_trace,
                        placements[0] if len(placements) == 1 else placements,
                        bound_cost,
                        host_moves=host_moves,
                        buffer_metrics=buffer_metrics,
                    )
                    cycles = int(bound_cost.evaluate(graph).cycles)
                except InfeasibleSchedule as error:
                    rejections.append((position, str(error)))
                    continue
                scored.append(
                    (
                        cycles,
                        _structural_decision_key(infos[0][0], placements[0]),
                        position,
                    )
                )
                realized[position] = placements
        scored.sort()

        winner = None
        winner_cycles = None
        for cycles, _key, position in scored:
            try:
                _materialize_matcher_codegen(
                    target,
                    kernel_trace,
                    realized[position],
                    host_moves=host_moves,
                )
            except InfeasibleSchedule as error:
                rejections.append((position, str(error)))
                continue
            winner = position
            winner_cycles = cycles
            break
        if winner is None:
            raise RuntimeError(
                f"no feasible placement for kernel group {group_id} on target "
                f"{getattr(target, 'name', None)!r}; rejections: {rejections}"
            )
        for index, placement in zip(bucket_indices, realized[winner]):
            chosen[index] = placement
        _layout, label, cost, _r, _reason, _tie = candidate_entry[winner]
        records.append(
            KernelRanking(
                group_id=group_id,
                bucket_count=len(bucket_indices),
                fallback=None,
                scores=tuple(
                    (position, cycles, candidate_mode[position])
                    for cycles, _key, position in scored
                ),
                chosen_position=winner,
                rejections=tuple(rejections),
                movement=with_chosen(
                    movement,
                    cost,
                    label,
                    ties_crossed=ties_crossed,
                    notes=tuple(notes),
                ),
                cost_ties=tuple(
                    (position, cycles, candidate_mode[position])
                    for cycles, _key, position in scored
                    if cycles == winner_cycles
                ),
                movement_rejections=tuple(movement_rejections),
                policy_pruned=tuple(
                    infos[0][1][0].extra.get("reduction_policy_pruned", ())
                ),
            )
        )

    placements = [chosen[index] for index in range(len(buckets))]
    _reconcile_resident_pairs(trace, placements, liveness)
    return MatcherScheduledPlacements(
        placements,
        placement_ranking=PlacementRanking(
            tuple(records),
            incomplete=any(
                record.movement is not None and record.movement.incomplete
                for record in records
            ),
        ),
    )


def _stamp_xkernel(trace, placements, liveness) -> None:
    """Stamp `extra["_xkernel"]` = the cross-kernel memref names each placement
    touches (from whole-trace liveness), on EVERY placement -- baseline and
    search alike. This is the workload-property signal codegen uses to emit the
    inter-kernel activation staging (which `residency=resident` then elides);
    it does not depend on the residency knob, so the group-local baseline
    carries it too. Empty when nothing crosses -> byte-identical."""
    from .spmw_liveness import memref_span

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


def _per_bucket_argmin_index(
    target,
    trace,
    candidates,
    bound_cost,
    *,
    host_moves=(),
    buffer_metrics=None,
) -> int:
    """Run the pre-search stable argmin used by public matcher defaults."""

    from .spmw_plan import build_execution_graph

    scored = []
    for index, candidate in enumerate(candidates):
        graph = build_execution_graph(
            target,
            trace,
            candidate,
            bound_cost,
            host_moves=host_moves,
            buffer_metrics=buffer_metrics,
        )
        scored.append((bound_cost.evaluate(graph).cycles, index))
    scored.sort()
    return int(scored[0][1])


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
                and mref
                in (getattr(other_pl, "extra", {}) or {}).get("residency_pairs", {})
            )
            if not other_ok:
                _revert_to_restage(pl, mref)
