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

import hashlib
import json
from dataclasses import dataclass, field, fields, is_dataclass
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
class MatcherPlacementMaterialization:
    """Candidate-owned placement, score graph, and frozen executable."""

    placement: Placement
    execution_graph: Any
    score_graph_fingerprint: str | None = None
    executable: Any = None

    @property
    def promotion_materialization_fingerprint(self) -> str | None:
        executable_fingerprint = getattr(
            self.executable,
            "promotion_materialization_fingerprint",
            None,
        )
        if not executable_fingerprint or not self.score_graph_fingerprint:
            return None
        return _matcher_digest(
            {
                "kind": "matcher-placement-materialization-v1",
                "score_graph": self.score_graph_fingerprint,
                "executable": executable_fingerprint,
            }
        )

    @property
    def promotion_platform_fingerprint(self):
        return getattr(self.executable, "promotion_platform_fingerprint", None)


@dataclass(frozen=True)
class MatcherProgramMaterialization:
    """Candidate-owned placements, score graph, and frozen executable."""

    placements: tuple[Placement, ...]
    execution_graph: Any
    score_graph_fingerprint: str | None = None
    executable: Any = None

    @property
    def promotion_materialization_fingerprint(self) -> str | None:
        executable_fingerprint = getattr(
            self.executable,
            "promotion_materialization_fingerprint",
            None,
        )
        if not executable_fingerprint or not self.score_graph_fingerprint:
            return None
        return _matcher_digest(
            {
                "kind": "matcher-program-materialization-v1",
                "score_graph": self.score_graph_fingerprint,
                "executable": executable_fingerprint,
            }
        )

    @property
    def promotion_platform_fingerprint(self):
        return getattr(self.executable, "promotion_platform_fingerprint", None)


def _matcher_digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _freeze_score_manifest(value: object) -> object:
    if value is None or type(value) in (bool, int, float, str):
        return value
    if type(value) is bytes:
        return {"bytes": value.hex()}
    if isinstance(value, dict):
        return {
            str(key): _freeze_score_manifest(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_freeze_score_manifest(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(
            (_freeze_score_manifest(item) for item in value),
            key=lambda item: json.dumps(item, sort_keys=True),
        )
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                item.name: _freeze_score_manifest(getattr(value, item.name))
                for item in fields(value)
            },
        }
    manifest = getattr(value, "manifest", None)
    if callable(manifest):
        return _freeze_score_manifest(manifest())
    raise TypeError(
        f"score manifest contains unsupported type {type(value).__name__!r}"
    )


def _score_graph_fingerprint(graph: object) -> str | None:
    """Fingerprint the exact graph scored by a matcher candidate.

    Test doubles and legacy graph objects that cannot be represented
    structurally remain searchable but are deliberately non-promotable.
    """

    try:
        activities = getattr(graph, "activities")
        payload = {
            "name": str(getattr(graph, "name", "")),
            "metadata": _freeze_score_manifest(getattr(graph, "metadata", {})),
            "activities": [_freeze_score_manifest(item) for item in activities],
        }
    except (AttributeError, TypeError, ValueError):
        return None
    return _matcher_digest(payload)


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
    """List-compatible result carrying shadow whole-program search evidence."""

    def __init__(
        self,
        placements=(),
        *,
        schedule_search_result=None,
        schedule_activation=None,
        active_materialization=None,
    ):
        super().__init__(placements)
        self.schedule_search_result = schedule_search_result
        self.schedule_activation = schedule_activation
        self.active_materialization = active_materialization
        self.fallback_reason = (
            None if schedule_activation is None else schedule_activation.fallback_reason
        )


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
    role_to_memref = _trace_memrefs_by_role(matches)
    x_mref = role_to_memref.get("x")
    y_mref = role_to_memref.get("y")
    acc_mref = role_to_memref.get("acc")
    if x_mref is None or y_mref is None or acc_mref is None:
        raise NotImplementedError(
            "aim enumerator: trace is missing one of x/y/acc roles; "
            f"got {sorted(role_to_memref)}"
        )

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

    layouts: list[Placement] = []
    # Candidate 1: per-bank MAC (no_bank_layout -- y stays in its bank).
    layouts.append(
        Placement(
            placements={
                x_mref: bank_handle,
                y_mref: gb,
                acc_mref: mac_reg,
            },
            mode="single_bank",
            extra={
                "operation_name": "MAC" if single_fanout == 1 else "MAC_ABK",
                "bank_fanout": single_fanout,
                "bank_conflicts": single_conflicts,
            },
            layout=single_bank_layout,
        )
    )
    # Candidate 2: all-bank-broadcast MAC (all_bank_layout -- y rides gb).
    layouts.append(
        Placement(
            placements={
                x_mref: banks,
                y_mref: gb,
                acc_mref: mac_reg,
            },
            mode="all_bank",
            extra={
                "operation_name": "MAC" if all_fanout == 1 else "MAC_ABK",
                "bank_fanout": all_fanout,
                "bank_conflicts": all_conflicts,
            },
            layout=all_bank_layout,
        )
    )
    return layouts


@register_enumerator("aim")
def _aim_enumerate(target, matches: list[MatchedOp]) -> list[Placement]:
    """Return only AiM layouts the current whole-program emitter realizes.

    ``AimCtx.emit_gemv`` emits one channel-scoped native ``MAC_ABK`` segment.
    The bank-scoped layout remains available through
    :func:`_aim_layout_candidates` for structural and cost tests, but it must
    not compete in autoscheduling until whole-program SBK emission consumes
    the selected bank coordinates.  Keeping a scored SBK candidate here would
    select one program and emit a different one.
    """

    candidates = _aim_layout_candidates(target, matches)
    materializable = [
        candidate
        for candidate in candidates
        if candidate.extra.get("operation_name") == "MAC_ABK"
    ]
    if len(materializable) != 1:
        raise RuntimeError(
            "AiM native whole-program lowering requires exactly one "
            "materializable MAC_ABK layout"
        )
    return materializable


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

    # APU v2 topology note: 64K-lane element axis with a 16-row L1 group.
    # Until retained semantics and codegen prove interchangeable operand rows,
    # expose only the canonical physical binding rather than a cost-identical
    # synthetic challenger.
    l1 = target.l1
    return [
        Placement(
            placements={
                x_mref: l1[0],
                y_mref: l1[1],
                acc_mref: l1[2],
            },
            mode="l1_row_canonical",
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
    promotion_gate=None,
) -> list[Placement]:
    """Pick one `Placement` per `@allo.work` kernel in `trace`.

    Returns a list aligned with `_bucket_for_autoschedule(trace)`:
    ``result[i]`` is the chosen placement for the i-th kernel (in
    trace order). For single-kernel workloads this list has length 1.

    Every candidate is lowered through the same executable cost program used
    by the virtual backend. Target declarations never supply timing data.
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
    from .pim.schedule_promotion import validate_schedule_promotion_gate

    promotion_gate = validate_schedule_promotion_gate(promotion_gate)

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

    liveness = trace_liveness(trace)
    _liveness_token = set_active_liveness(liveness)
    try:
        placements = _autoschedule_groups(
            target,
            target_name,
            trace,
            enumerator,
            bound_cost,
            host_moves,
            buffer_metrics,
            liveness,
            promotion_gate,
        )
    finally:
        reset_active_liveness(_liveness_token)
    # Every activation returns the already-finalized candidate materialization.
    # Workload staging and resident-pair reconciliation happen before scoring
    # and code emission inside the candidate materializer, never afterward.
    return placements


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


def _autoschedule_groups(
    target,
    target_name,
    trace,
    enumerator,
    bound_cost,
    host_moves,
    buffer_metrics,
    liveness,
    promotion_gate,
) -> "list[Placement]":
    """Retain explicit legacy incumbents and rank independent challengers."""
    from .pim.schedule_search import guarded_schedule_activation

    buckets = tuple(_bucket_for_autoschedule(trace))
    candidate_groups = []
    incumbent_indices = []
    legacy_results = []
    for func_name, matches in buckets:
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

        # Freeze the exact public scheduler incumbent before the new shared
        # search ranks anything. This is the former stable `(cycles, index)`
        # argmin, kept as an independent lifecycle input rather than declaring
        # the new recommendation to be its own incumbent.
        incumbent_index = _legacy_autoschedule_group_index(
            target,
            sub_trace,
            candidates,
            bound_cost,
            host_moves=host_moves,
            buffer_metrics=buffer_metrics,
        )

        result = _search_autoschedule_group(
            target,
            sub_trace,
            candidates,
            bound_cost,
            host_moves=host_moves,
            buffer_metrics=buffer_metrics,
            incumbent_index=incumbent_index,
            liveness=liveness if len(buckets) == 1 else None,
        )
        candidate_groups.append(tuple(candidates))
        incumbent_indices.append(incumbent_index)
        legacy_results.append(result)

    if len(candidate_groups) == 1:
        activation = guarded_schedule_activation(
            legacy_results[0],
            promotion_gate=promotion_gate,
        )
        materialized = activation.active.materialized
        return MatcherScheduledPlacements(
            (_clone_placement(materialized.placement),),
            schedule_search_result=legacy_results[0],
            schedule_activation=activation,
            active_materialization=materialized,
        )

    joint_result = _search_autoschedule_program(
        target,
        trace,
        candidate_groups,
        bound_cost,
        liveness=liveness,
        host_moves=host_moves,
        buffer_metrics=buffer_metrics,
        incumbent_indices=incumbent_indices,
    )
    activation = guarded_schedule_activation(
        joint_result,
        promotion_gate=promotion_gate,
    )
    materialized = activation.active.materialized
    return MatcherScheduledPlacements(
        (_clone_placement(item) for item in materialized.placements),
        schedule_search_result=joint_result,
        schedule_activation=activation,
        active_materialization=materialized,
    )


def _legacy_autoschedule_group_index(
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


def _search_autoschedule_group(
    target,
    trace,
    candidates,
    bound_cost,
    *,
    host_moves=(),
    buffer_metrics=None,
    incumbent_index=None,
    liveness=None,
):
    """Run one matcher bucket through the shared schedule-search lifecycle."""

    from .pim.schedule_search import (
        DecisionDomain,
        ScheduleObjectiveDomain,
        grid_search,
    )
    from .spmw_codegen import _materialize_matcher_codegen, _stamp_host_moves
    from .spmw_plan import build_execution_graph

    candidates = tuple(candidates)
    if not candidates:
        raise ValueError("matcher schedule search requires at least one candidate")
    if incumbent_index is not None and not 0 <= int(incumbent_index) < len(candidates):
        raise ValueError("incumbent_index is outside the candidate domain")
    operand_roles = _matcher_operand_roles(trace, candidates)
    choice_to_candidate = {}
    for candidate in candidates:
        choice = _matcher_placement_decision(trace, candidate, operand_roles)
        choice_to_candidate.setdefault(choice, candidate)
    choices = tuple(choice_to_candidate)

    def materialize(template):
        placement = _clone_placement(template)
        placement.extra = derive_layout_properties(target, placement)
        _stamp_xkernel(trace, [placement], liveness)
        _reconcile_resident_pairs(trace, [placement], liveness)
        if host_moves:
            _stamp_host_moves(placement, host_moves)
        executable = _materialize_matcher_codegen(
            target,
            trace,
            [placement],
            host_moves=host_moves,
        )
        graph = build_execution_graph(
            target,
            trace,
            placement,
            bound_cost,
            host_moves=host_moves,
            buffer_metrics=buffer_metrics,
        )
        return MatcherPlacementMaterialization(
            placement,
            graph,
            _score_graph_fingerprint(graph),
            executable,
        )

    return grid_search(
        (DecisionDomain("placement", choices),),
        build=lambda decisions: choice_to_candidate[decisions["placement"]],
        materialize=materialize,
        score=lambda realized: bound_cost.evaluate(realized.execution_graph),
        objective=lambda estimate: int(estimate.cycles),
        objective_domain=ScheduleObjectiveDomain.fingerprinted_target(
            metric="cycles",
            target=getattr(target, "name", ""),
            model_fingerprint=bound_cost.fingerprint,
            fidelity="analytical",
            scope="region",
            unit="cycles",
            direction="minimize",
        ),
        incumbent=(
            None
            if incumbent_index is None
            else {
                "placement": _matcher_placement_decision(
                    trace,
                    candidates[int(incumbent_index)],
                    operand_roles,
                )
            }
        ),
    )


def _search_autoschedule_program(
    target,
    trace,
    candidate_groups,
    bound_cost,
    *,
    liveness=None,
    host_moves=(),
    buffer_metrics=None,
    incumbent_indices=None,
    max_complete_assignments=256,
):
    """Score one immutable placement tuple over the completed trace graph.

    Multi-group winners remain shadow-only: callers activate the explicit
    legacy per-group incumbent until executable equivalence or hardware
    non-regression promotes a joint challenger.
    """

    from .pim.schedule_search import (
        DecisionDomain,
        ScheduleObjectiveDomain,
        grid_search,
    )
    from .spmw_codegen import _materialize_matcher_codegen, _stamp_host_moves
    from .spmw_plan import build_execution_graph

    candidate_groups = tuple(tuple(group) for group in candidate_groups)
    if not candidate_groups or any(not group for group in candidate_groups):
        raise ValueError("matcher program search requires nonempty candidate groups")
    buckets = tuple(_bucket_for_autoschedule(trace))
    if len(buckets) != len(candidate_groups):
        raise ValueError("matcher candidate groups do not align with trace buckets")
    group_traces = tuple(
        MatchTrace(
            target_name=trace.target_name,
            module_name=trace.module_name,
            matches=matches,
        )
        for _function, matches in buckets
    )
    choice_maps = []
    for group_trace, group in zip(group_traces, candidate_groups):
        operand_roles = _matcher_operand_roles(group_trace, group)
        choices = {}
        for candidate in group:
            choice = _matcher_placement_decision(group_trace, candidate, operand_roles)
            choices.setdefault(choice, candidate)
        choice_maps.append(choices)
    choice_maps = tuple(choice_maps)
    domains = tuple(
        DecisionDomain(f"group_{index}_placement", tuple(choices))
        for index, choices in enumerate(choice_maps)
    )
    incumbent = None
    if incumbent_indices is not None:
        incumbent_indices = tuple(int(value) for value in incumbent_indices)
        if len(incumbent_indices) != len(candidate_groups):
            raise ValueError("incumbent indices must align with candidate groups")
        if any(
            not 0 <= value < len(candidate_groups[index])
            for index, value in enumerate(incumbent_indices)
        ):
            raise ValueError("an incumbent index is outside its candidate domain")
        incumbent = {
            f"group_{index}_placement": _matcher_placement_decision(
                group_traces[index],
                candidate_groups[index][value],
                _matcher_operand_roles(group_traces[index], candidate_groups[index]),
            )
            for index, value in enumerate(incumbent_indices)
        }

    def build(decisions):
        return tuple(
            choice_maps[index][decisions[f"group_{index}_placement"]]
            for index in range(len(choice_maps))
        )

    def materialize(templates):
        placements = [_clone_placement(template) for template in templates]
        for placement in placements:
            placement.extra = derive_layout_properties(target, placement)
        _stamp_xkernel(trace, placements, liveness)
        _reconcile_resident_pairs(trace, placements, liveness)
        if host_moves:
            _stamp_host_moves(placements, host_moves)
        executable = _materialize_matcher_codegen(
            target,
            trace,
            placements,
            host_moves=host_moves,
        )
        graph = build_execution_graph(
            target,
            trace,
            placements,
            bound_cost,
            host_moves=host_moves,
            buffer_metrics=buffer_metrics,
        )
        return MatcherProgramMaterialization(
            tuple(placements),
            graph,
            _score_graph_fingerprint(graph),
            executable,
        )

    return grid_search(
        domains,
        build=build,
        materialize=materialize,
        score=lambda realized: bound_cost.evaluate(realized.execution_graph),
        objective=lambda estimate: int(estimate.cycles),
        objective_domain=ScheduleObjectiveDomain.fingerprinted_target(
            metric="cycles",
            target=getattr(target, "name", ""),
            model_fingerprint=bound_cost.fingerprint,
            fidelity="analytical",
            scope="whole_program",
            unit="cycles",
            direction="minimize",
        ),
        incumbent=incumbent,
        max_complete_assignments=max_complete_assignments,
    )


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
