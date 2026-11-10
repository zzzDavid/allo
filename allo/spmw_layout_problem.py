# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimum-movement layout problem of one matcher kernel group (paper IV).

:func:`derive_layout_problem` turns the matched ops of one ``@allo.work``
kernel group into the search input of :mod:`allo.pim.layout_search`;
:func:`analyze_kernel_movement` runs the search and classifies its ties for
the autoscheduler's layout-construction stage
(``spmw_autoschedule.rank_matcher_placements``).

Derivation rules
----------------

Iteration variables.  One variable per work-grid axis
(``MatcherWorkScope.group_shape``, outer first; ``work0, work1, ...``) and
one per enclosing loop of the matched op (outer to inner; ``loop0, ...``).
Names are positional, so renaming source symbols changes nothing.  A loop
must start at 0, step by 1 and have a static bound
(``spmw_tripcount._parse_loop_bound``); its active extent is the trip count.
A variable occupies ``ceil(log2(extent))`` carrier bits; a non-power-of-two
extent makes the domain *masked* (a box of active points).

Owner hierarchy.  ``COMPUTE_LEVELS`` names the compute levels of each
target, outermost first.  The kernel's work grid maps its axes, in order, onto
the owner axes of those levels; the hierarchy is the levels the grid covers
(deeper levels execute inside one work item and are recorded as
``unmodelled_levels``).  A work-grid extent smaller than the level's extent,
or a non-power-of-two level extent (UPMEM's 24 tasklets), becomes an
``active_extents`` subgroup of
:func:`allo.pim.layout_problem.owner_hierarchy_from_target`.  The work-grid
extent itself must be a power of two.  The remaining carrier bits are local.

Accesses.  Every input operand of the matched op is a READ access.  The
loop-carried accumulator is a WRITE_ACCUMULATE access: through its output
sink (``extra["output_sinks"]``) when the accumulator is a scalar, otherwise
through its own indices.  Each coordinate (``index_map`` applied to
``indices``, or the raw index text) must be an affine form
``c + sum a_v v`` over loop variables whose per-work-item constant ``c`` is
affine in the work id (fitted across the group's buckets and checked on
every bucket).  The access then depends on the variables with ``a_v != 0``;
it must do so injectively (mixed radix: sorted by coefficient, every
coefficient exceeds the largest value the smaller terms reach), so two
iterations touch the same element exactly when they agree on those
variables.  The access kernel ``K_j`` is then the coordinate subspace of
the other variables' bits, which is all :mod:`layout_movement` reads.
Non-affine text, non-unit loop steps, negative or non-injective
coefficients, and work dependence that is not affine are rejected with the
reason (:class:`LayoutDerivationError`).

One narrow resolution rule is applied and recorded in ``assumptions``: a
contraction's matrix coordinate that is an SSA value the matcher retains
only by name (``memref.load`` of ``row0 + i``) is taken equal to the single
store row of the accumulator, the rule ``spmw_samsung.samsung_contraction``
already applies.

Element bytes come from the operand's memref element type (``f16``/``bf16``
2, ``f32``/``i32`` 4, ``iN`` ``N/8``); an output sink uses the accumulator's
type.  ``reassociation_allowed`` comes from ``extra["reduction_policy"]``
(``may_regroup``, or ``order == "reassociate"``).  When a match carries no
policy, the paper's reassociating semantics are assumed and the assumption
is recorded.

Movement weights.  ``cost_per_level`` comes from the bound cost program when
it exposes ``movement_cost_per_byte(level_name) -> number`` (on the bound
spec or its ``spec``).  No cost program in the package does today, so every
level weighs 1 per byte and ``movement_weights == "uniform"``.

Search dispatch (:func:`analyze_kernel_movement`): a problem inside the
bounded domain (``policy.search.max_carrier_bits`` bits and
``max_levels`` levels) runs the ported Algorithm 1
(:func:`allo.pim.layout_search.search_min_movement_layout`, masked mode
included).  Larger problems run the exact coordinate-variable search
(:func:`allo.pim.layout_search.search_coordinate_min_movement`), which on a
full carrier is exact over all one-to-one layouts, and on a mask over
coordinate layouts only.

Realization.  APU v1 emits one program per work item, placed on the unit
the work grid names, and does not rewrite the work decomposition.  There a
tie is realizable exactly when its owner flag (``T_s`` per level) equals the
flag of the program layout (work axis ``d``'s bits on owner axis ``d``, loop
bits local).

Targets in :data:`ROW_MAP_TARGETS` place the output rows of a contraction
by the selected owner map: the lowering turns the tie into a row-to-unit
assignment (:func:`owner_rows`; the rows one owner unit holds, in ascending
order).  Samsung HBM-PIM applies it as a row-to-slot table in the
conductor; UPMEM packs DPU ``u``'s MRAM image from the rows unit ``u``
owns and gathers its output back to those rows; AiM places row ``r`` at
output-frontier position ``slot(r)`` (the channel and bank the mapping's
storage contract gives that position), re-addresses per-row host transfers
and records the host-side result mapping.  There a tie is realizable when
it keeps every row's reduction inside one owner unit (the accumulator's
kernel lies in the innermost local space ``T_h``) and the accumulator has
one output-row coordinate (:func:`row_map_unrealizable_reason`).  Samsung
also requires unmasked row variables; UPMEM and AiM require a
one-dimensional work grid and every unit to own the same number of rows
(checked on the active rows, so a masked row loop is accepted when it
divides evenly).  A tie that splits a row's reduction across units is a
reduction split, which belongs to the reduction policy, and is rejected
with that reason; on AiM the one exception is the split the
``reduction_partitions`` knob already lowers (every owner bit from the low
bits of the one reduction loop: element ``t`` on channel group ``t % P``,
:func:`reduction_partition_split`), which is realizable on the candidates
that carry that ``P``.  The program's tie class is also expanded into its
per-level BLOCK/CYCLIC row distributions (:func:`row_distributions`), and
on AiM into the cyclic reduction split when it is minimal, so distinct
realizable owner maps reach the cost ranking as separate ties.
Unrealizable ties are still crossed with the backend candidates and are
rejected before emission with the reason.
"""

from __future__ import annotations

import ast
import contextlib
import contextvars
import itertools
import math
import re
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Any

from .pim.gf2 import GF2Matrix, GF2Subspace
from .pim.layout_movement import AccessMode, LayoutAccess, layout_movement_cost
from .pim.layout_problem import TargetHierarchyError, owner_hierarchy_from_target
from .pim.layout_search import (
    DEFAULT_MAX_VARIABLE_CHOICES,
    SCOPE_ALL_LAYOUTS,
    SCOPE_COORDINATE_LAYOUTS,
    CoordinateAccess,
    CoordinateAssignment,
    CoordinateLayoutProblem,
    CoordinateVariable,
    LayoutSearchDomainError,
    LayoutSearchPolicy,
    LayoutSearchProblem,
    LayoutSearchTermination,
    NoLegalLayoutError,
    coordinate_assignment_cost,
    search_coordinate_min_movement,
    search_min_movement_layout,
)

#: Compute levels per target, outermost first.  Unit names are units of the
#: target's unit tree; ``"reg.axis"`` names a register lane axis.
COMPUTE_LEVELS: dict[str, tuple[str, ...]] = {
    "samsung_hbm_pim": ("pseudo_channel", "pim"),
    "aim": ("channel", "bank_group", "bank"),
    "upmem": ("dpu", "tasklet"),
    "apu_v1": ("vr0.lane",),
}

#: Targets whose contraction lowering places output rows by the selected owner
#: map (``spmw_samsung`` row-to-slot table, ``spmw_upmem`` row-to-DPU images,
#: ``spmw_aim`` row-to-frontier slots); see the module docstring.
ROW_MAP_TARGETS = frozenset({"samsung_hbm_pim", "upmem", "aim"})

#: Row-map targets that need unmasked row variables (Samsung's conductor
#: packs a full slot range per unit); the others check equal row counts on
#: the active rows.
_UNMASKED_ROW_TARGETS = frozenset({"samsung_hbm_pim"})

#: Enumeration cap of :func:`owner_rows` (output rows of one kernel group).
_MAX_OWNER_ROWS = 1 << 22

MOVEMENT_WEIGHTS_UNIFORM = "uniform"
MOVEMENT_WEIGHTS_COST_PROGRAM = "cost-program"

#: Owner-dim prefix of movement layouts carried on ``Placement.layout``; keeps
#: them disjoint from every backend layout's dims in a ``product``.
LAYOUT_DIM_PREFIX = "mm."

_ELEMENT_BYTES = {"f16": 2, "bf16": 2, "f32": 4, "f64": 8, "index": 8}


class LayoutDerivationError(ValueError):
    """The kernel group cannot be stated as a minimum-movement problem."""


# --------------------------------------------------------------------- #
# Derived problem
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class DerivedAccess:
    """One access of the derived problem, with the evidence it came from."""

    name: str
    memref: str
    mode: AccessMode
    variables: tuple[int, ...]
    element_bytes: Fraction
    reassociation_allowed: bool
    coordinates: tuple[str, ...]  # bucket-0 coordinate text, for diagnostics
    # Per coordinate: ((coefficient, variable), ...) sorted by coefficient.
    terms: tuple[tuple[tuple[int, int], ...], ...] = ()
    # Per coordinate: the constant at the grid origin (``c0``).
    offsets: tuple[int, ...] = ()


@dataclass(frozen=True)
class DerivedLayoutProblem:
    # pylint: disable=too-many-instance-attributes
    """A kernel group's minimum-movement problem.

    ``variables`` are in carrier order (work axes, then loops outer to
    inner).  ``level_names`` are the owner levels the work grid covers;
    ``work_levels[d]`` is the level index of work axis ``d``.
    """

    target_name: str
    level_names: tuple[str, ...]
    unmodelled_levels: tuple[str, ...]
    active_extents: tuple[tuple[str, int], ...]
    variables: tuple[CoordinateVariable, ...]
    work_levels: tuple[int, ...]
    accesses: tuple[DerivedAccess, ...]
    cost_per_level: tuple[Fraction, ...]
    movement_weights: str
    assumptions: tuple[str, ...]
    hierarchy: Any = field(repr=False)
    coordinate_problem: CoordinateLayoutProblem = field(repr=False)
    op_name: str | None = None

    @property
    def carrier_bits(self) -> int:
        return self.coordinate_problem.carrier_bits

    @property
    def is_masked(self) -> bool:
        return self.coordinate_problem.is_masked

    def program_assignment(self) -> CoordinateAssignment:
        """The program's own layout: work axis ``d`` on its owner level."""

        levels = []
        for level in range(len(self.level_names)):
            levels.append(
                tuple(
                    (
                        tuple(range(self.variables[v].bits))
                        if v < len(self.work_levels) and self.work_levels[v] == level
                        else ()
                    )
                    for v in range(len(self.variables))
                )
            )
        return CoordinateAssignment(tuple(levels))

    def program_layout(self) -> GF2Matrix:
        return self.program_assignment().matrix(self.coordinate_problem)

    def layout_accesses(self) -> tuple[LayoutAccess, ...]:
        return self.coordinate_problem.layout_accesses()

    def search_problem(self) -> LayoutSearchProblem:
        """The GF(2) problem for the ported search (box mask when masked)."""

        active = self.coordinate_problem.active_points() if self.is_masked else None
        return LayoutSearchProblem(self.hierarchy, self.layout_accesses(), (), active)

    def owner_flag(self, work_to_iteration: GF2Matrix) -> tuple[GF2Subspace, ...]:
        """``T_s`` (iteration directions local through level ``s``) per level."""

        return tuple(
            work_to_iteration.image(self.hierarchy.local_coordinate_space_after(level))
            for level in range(len(self.level_names))
        )

    def linear_layout(self, work_to_iteration: GF2Matrix):
        """The layout as a ``LinearLayout`` over ``mm.``-prefixed dims."""

        from .spmw_linear_layout import LinearLayout

        in_dims = []
        for level in self.hierarchy.levels:
            for axis in level.axes:
                in_dims.append(
                    (LAYOUT_DIM_PREFIX + axis.diagnostic_name, axis.mapped_extent)
                )
        local = self.hierarchy.local_bit_count
        if local:
            in_dims.append((LAYOUT_DIM_PREFIX + "local", 1 << local))
        out_dims = [
            (LAYOUT_DIM_PREFIX + variable.name, 1 << variable.bits)
            for variable in self.variables
            if variable.bits
        ]
        return LinearLayout.from_gf2(work_to_iteration, in_dims, out_dims)

    def accumulator(self) -> DerivedAccess | None:
        """The single WRITE_ACCUMULATE access, or None."""

        found = [a for a in self.accesses if a.mode is AccessMode.WRITE_ACCUMULATE]
        return found[0] if len(found) == 1 else None

    def owner_unit(self, inverse: GF2Matrix, values: dict[int, int]) -> int:
        """Flat owner unit (level 0 most significant) of one iteration point.

        ``inverse`` is the inverse of a work-to-iteration layout; ``values``
        maps variable indices to their values (absent variables are 0).
        """

        problem = self.coordinate_problem
        iteration = 0
        for variable, value in values.items():
            iteration |= int(value) << problem.bit_offset(variable)
        work = inverse.apply(iteration)
        unit, offset = 0, 0
        for bits in problem.level_bits:
            unit = (unit << bits) | ((work >> offset) & ((1 << bits) - 1))
            offset += bits
        return unit

    def movement_matrix(self, layout) -> GF2Matrix:
        """The work-to-iteration matrix of a carried movement layout.

        Raises :class:`LayoutDerivationError` when ``layout`` is not a
        one-to-one layout over this problem's ``mm.`` dims.
        """

        try:
            matrix = layout.to_gf2()
            if self.linear_layout(matrix) != layout:
                raise ValueError("dims differ")
            matrix.inverse()
        except (ValueError, KeyError) as error:
            raise LayoutDerivationError(
                "the carried movement layout is not a one-to-one layout over the "
                f"kernel's derived dims ({type(error).__name__}: {error})"
            ) from error
        return matrix

    def describe_assignment(self, assignment: CoordinateAssignment) -> str:
        parts = []
        for level, name in enumerate(self.level_names):
            owners = [
                f"{self.variables[v].name}[{','.join(str(b) for b in sorted(bits))}]"
                for v, bits in enumerate(assignment.levels[level])
                if bits
            ]
            parts.append(f"{name}<-{'+'.join(owners) or '-'}")
        return "; ".join(parts)


# --------------------------------------------------------------------- #
# Coordinate parsing
# --------------------------------------------------------------------- #

_AFFINE_MAP = re.compile(r"affine_map<\(([^)]*)\)(\[[^\]]*\])?\s*->\s*\((.*)\)>")
_MEMREF_ELEMENT = re.compile(r"memref<(?:[^<>]*x)?([A-Za-z]+\d*)(?:,[^<>]*)?>")


def _split_top_level(text: str) -> list[str]:
    parts, depth, current = [], 0, ""
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if char == "," and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += char
    if current.strip():
        parts.append(current.strip())
    return parts


def _operand_coordinates(operand) -> list[str]:
    """Accessed coordinates as text over SSA names."""

    if operand.index_map is None:
        return [str(index) for index in operand.indices]
    found = _AFFINE_MAP.fullmatch(operand.index_map.strip())
    if not found:
        raise LayoutDerivationError(
            f"{operand.memref_name}: cannot read access map {operand.index_map!r}"
        )
    if found.group(2) and found.group(2).strip("[] "):
        raise LayoutDerivationError(
            f"{operand.memref_name}: access map {operand.index_map!r} has symbols"
        )
    dims = [d.strip() for d in found.group(1).split(",") if d.strip()]
    if len(dims) != len(operand.indices):
        raise LayoutDerivationError(
            f"{operand.memref_name}: access map {operand.index_map!r} has "
            f"{len(dims)} dims for {len(operand.indices)} indices"
        )
    results = _split_top_level(found.group(3))
    out = []
    for result in results:
        text = result
        for dim, index in sorted(zip(dims, operand.indices), key=lambda p: -len(p[0])):
            text = re.sub(rf"\b{re.escape(dim)}\b", f"({index})", text)
        out.append(text)
    return out


def _linear_form(text: str, memref: str) -> tuple[dict[str, int], int]:
    """``text`` as ``({ssa name: coefficient}, constant)``; raises if not affine."""

    if re.search(r"floordiv|ceildiv|\bmod\b|\bselect\b|\bmin\b|\bmax\b", text):
        raise LayoutDerivationError(f"{memref}: coordinate {text!r} is not affine")
    source = re.sub(r"%(\w+)", r"_v_\1", text)
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as error:
        raise LayoutDerivationError(
            f"{memref}: cannot parse coordinate {text!r}"
        ) from error

    def scale(form, factor):
        coefs, const = form
        return {k: v * factor for k, v in coefs.items()}, const * factor

    def plus(a, b, sign=1):
        coefs = dict(a[0])
        for k, v in b[0].items():
            coefs[k] = coefs.get(k, 0) + sign * v
        return {k: v for k, v in coefs.items() if v}, a[1] + sign * b[1]

    def times(a, b):
        if not a[0]:
            return scale(b, a[1])
        if not b[0]:
            return scale(a, b[1])
        raise LayoutDerivationError(f"{memref}: coordinate {text!r} is not affine")

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return {}, int(node.value)
        if isinstance(node, ast.Name) and node.id.startswith("_v_"):
            return {"%" + node.id[3:]: 1}, 0
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return scale(walk(node.operand), -1)
        if isinstance(node, ast.BinOp):
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Add):
                return plus(left, right)
            if isinstance(node.op, ast.Sub):
                return plus(left, right, -1)
            if isinstance(node.op, ast.Mult):
                return times(left, right)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("add", "sub", "mul")
            and len(node.args) == 2
            and not node.keywords
        ):
            left, right = walk(node.args[0]), walk(node.args[1])
            if node.func.id == "add":
                return plus(left, right)
            if node.func.id == "sub":
                return plus(left, right, -1)
            return times(left, right)
        raise LayoutDerivationError(f"{memref}: coordinate {text!r} is not affine")

    return walk(tree)


def _element_bytes(memref_type: str | None, memref: str) -> Fraction:
    found = _MEMREF_ELEMENT.search(memref_type or "")
    if not found:
        raise LayoutDerivationError(
            f"{memref}: element type of {memref_type!r} is not retained"
        )
    name = found.group(1)
    if name in _ELEMENT_BYTES:
        return Fraction(_ELEMENT_BYTES[name])
    integer = re.fullmatch(r"[iu]i?(\d+)", name)
    if integer:
        return Fraction(int(integer.group(1)), 8)
    raise LayoutDerivationError(f"{memref}: unknown element type {name!r}")


def _loop_extent(loop) -> int:
    from .spmw_tripcount import _parse_loop_bound

    var, lower, upper, step = loop
    if int(step) != 1:
        raise LayoutDerivationError(
            f"loop {var} has step {step}; only unit-stride loops are analyzed"
        )
    lo, hi = _parse_loop_bound(str(lower)), _parse_loop_bound(str(upper))
    if lo is None or hi is None:
        raise LayoutDerivationError(
            f"loop {var} has no static bounds ({lower}, {upper})"
        )
    if lo != 0:
        raise LayoutDerivationError(f"loop {var} starts at {lo}, not 0")
    if hi < 1:
        raise LayoutDerivationError(f"loop {var} is empty")
    return hi


# --------------------------------------------------------------------- #
# Derivation
# --------------------------------------------------------------------- #


def _owner_axes(target, level_name):
    """``[(owner key, physical extent)]`` of one compute level."""

    from .spmw_target import Unit

    if "." in level_name:
        reg_name, axis = level_name.split(".", 1)
        register = getattr(target, reg_name, None)
        axes = getattr(register, "axes", None) or {}
        if axis not in axes:
            raise LayoutDerivationError(
                f"target {target.name!r} has no register axis {level_name!r}"
            )
        return [(level_name, int(axes[axis]))]
    try:
        unit = target.unit(level_name)
    except KeyError as error:
        raise LayoutDerivationError(str(error)) from error
    if not isinstance(unit, Unit):
        raise LayoutDerivationError(f"{level_name!r} is not a unit")
    if unit.axes:
        items = list(unit.axes.items())
    else:
        items = [(str(i), extent) for i, extent in enumerate(unit.mapping)]
    return [
        (f"{unit.name}.{name}", int(extent))
        for name, extent in items
        if int(extent) != 1
    ]


def _reassociation(match, assumptions) -> bool:
    policy = match.extra.get("reduction_policy")
    if policy is None:
        assumptions.append(
            "no reduction_policy on the match: the paper's reassociating "
            "semantics are assumed for the accumulator"
        )
        return True
    may_regroup = getattr(policy, "may_regroup", None)
    if may_regroup is not None:
        return bool(may_regroup)
    order = getattr(policy, "order", policy)
    return str(order) == "reassociate"


def _cost_per_level(bound_cost, level_names):
    hook = getattr(bound_cost, "movement_cost_per_byte", None)
    if hook is None:
        hook = getattr(
            getattr(bound_cost, "spec", None), "movement_cost_per_byte", None
        )
    if callable(hook):
        values = [hook(name) for name in level_names]
        if all(value is not None for value in values):
            return (
                tuple(Fraction(value) for value in values),
                MOVEMENT_WEIGHTS_COST_PROGRAM,
            )
    return tuple(Fraction(1) for _ in level_names), MOVEMENT_WEIGHTS_UNIFORM


def derive_layout_problem(target, matches, *, bound_cost=None) -> DerivedLayoutProblem:
    """Derive the minimum-movement problem of one kernel group.

    ``matches`` are the group's matched ops (one per work item).  Raises
    :class:`LayoutDerivationError` (or ``TargetHierarchyError``) with the
    reason when the group falls outside the derivation rules of the module
    docstring.
    """

    from .spmw_autoschedule import _matcher_work_scope

    target_name = getattr(target, "name", None)
    level_table = COMPUTE_LEVELS.get(target_name)
    if level_table is None:
        raise LayoutDerivationError(
            f"no compute levels declared for target {target_name!r}"
        )
    if not matches:
        raise LayoutDerivationError("empty kernel group")

    by_item: dict[tuple, list] = {}
    for match in matches:
        by_item.setdefault(tuple(_matcher_work_scope(match).work_id), []).append(match)
    if any(len(items) != 1 for items in by_item.values()):
        counts = sorted({len(items) for items in by_item.values()})
        raise LayoutDerivationError(
            f"work items hold {counts} matched ops; the layout problem covers "
            "one matched op (one iteration space) per work item"
        )
    first = matches[0]
    scope = _matcher_work_scope(first)
    if scope.coalesced_axes:
        raise LayoutDerivationError(
            "coalesced work axes: only representative buckets are retained"
        )
    group_shape = tuple(int(e) for e in scope.group_shape)
    if not group_shape:
        raise LayoutDerivationError("the work scope retains no group shape")
    if len(by_item) != math.prod(group_shape):
        raise LayoutDerivationError(
            f"{len(by_item)} work items for a {group_shape} work grid"
        )
    ops = {m.target_op_name for m in matches}
    if len(ops) != 1:
        raise LayoutDerivationError(f"work items match different ops {sorted(ops)}")

    # ---- owner hierarchy: the levels the work grid covers ----------------
    level_names, active_extents, work_levels = [], {}, []
    axis = 0
    for level_index, level_name in enumerate(level_table):
        if axis == len(group_shape):
            break
        owner_axes = _owner_axes(target, level_name)
        if axis + len(owner_axes) > len(group_shape):
            raise LayoutDerivationError(
                f"work grid {group_shape} ends inside level {level_name!r}"
            )
        for key, extent in owner_axes:
            grid = group_shape[axis]
            if grid < 2 or grid & (grid - 1):
                raise LayoutDerivationError(
                    f"work axis {axis} has extent {grid}; owner axes need a "
                    "power-of-two extent of at least 2"
                )
            if grid > extent:
                raise LayoutDerivationError(
                    f"work axis {axis} ({grid}) exceeds owner axis {key!r} ({extent})"
                )
            if grid != extent:
                active_extents[key] = grid
            work_levels.append(level_index)
            axis += 1
        level_names.append(level_name)
    if axis != len(group_shape):
        raise LayoutDerivationError(
            f"work grid {group_shape} has more axes than the compute levels "
            f"{level_table} of {target_name!r}"
        )
    unmodelled = tuple(level_table[len(level_names) :])

    # ---- iteration variables ---------------------------------------------
    loops = list(first.enclosing_loops)
    extents = [_loop_extent(loop) for loop in loops]
    for match in matches:
        if [_loop_extent(loop) for loop in match.enclosing_loops] != extents:
            raise LayoutDerivationError("work items differ in their loop nests")
    variables = tuple(
        [CoordinateVariable(f"work{d}", g) for d, g in enumerate(group_shape)]
        + [CoordinateVariable(f"loop{p}", n) for p, n in enumerate(extents)]
    )
    n_work = len(group_shape)

    assumptions: list[str] = []
    reassociation = _reassociation(first, assumptions)

    # ---- per-bucket coordinates --------------------------------------------
    def bucket_accesses(match):
        loop_pos = {loop[0]: p for p, loop in enumerate(match.enclosing_loops)}
        accs = [op for op in match.operands if op.is_loop_carried]
        inputs = [
            op for op in match.operands if not op.is_loop_carried and op.memref_name
        ]
        if len(accs) > 1:
            raise LayoutDerivationError("more than one loop-carried accumulator")
        entries = []  # (memref, mode, element bytes, [coordinate text])
        for op in inputs:
            entries.append(
                [
                    op.memref_name,
                    AccessMode.READ,
                    op.memref_type,
                    _operand_coordinates(op),
                    op,
                ]
            )
        if accs:
            (acc,) = accs
            sinks = match.extra.get("output_sinks") or []
            if acc.indices:
                entries.append(
                    [
                        acc.memref_name,
                        AccessMode.WRITE_ACCUMULATE,
                        acc.memref_type,
                        _operand_coordinates(acc),
                        acc,
                    ]
                )
            elif sinks:
                for memref, text in sinks:
                    entries.append(
                        [
                            memref,
                            AccessMode.WRITE_ACCUMULATE,
                            acc.memref_type,
                            _split_top_level(text),
                            acc,
                        ]
                    )
            else:
                raise LayoutDerivationError(
                    f"scalar accumulator {acc.memref_name!r} has no indexed output sink"
                )
        else:
            store = match.extra.get("store_indices")
            if not store or match.result_memref_name is None:
                raise LayoutDerivationError(
                    "non-accumulating op without a retained store index"
                )
            entries.append(
                [
                    match.result_memref_name,
                    AccessMode.WRITE_ACCUMULATE,
                    None,
                    [str(s) for s in store],
                    None,
                ]
            )

        forms = []
        for memref, mode, mtype, coords, op in entries:
            parsed = [_linear_form(text, memref) for text in coords]
            opaque = [
                (i, name)
                for i, (coefs, _c) in enumerate(parsed)
                for name in coefs
                if name not in loop_pos
            ]
            if opaque:
                parsed = _resolve_opaque(
                    match, op, memref, coords, parsed, opaque, loop_pos, assumptions
                )
            loop_coords = []
            for coefs, const in parsed:
                loop_coords.append(({loop_pos[k]: v for k, v in coefs.items()}, const))
            forms.append((memref, mode, mtype, loop_coords, coords))
        return forms

    per_bucket = {}
    for work_id, (match,) in by_item.items():
        per_bucket[work_id] = bucket_accesses(match)
    zero = tuple(0 for _ in group_shape)
    if zero not in per_bucket:
        raise LayoutDerivationError("the work item at the grid origin is missing")
    base = per_bucket[zero]
    shape_sig = [(m, mode, len(c)) for m, mode, _t, c, _x in base]
    for work_id, forms in per_bucket.items():
        if [(m, mode, len(c)) for m, mode, _t, c, _x in forms] != shape_sig:
            raise LayoutDerivationError("work items differ in their accesses")

    # Fit constant(w) = c0 + sum_d a_d w_d per coordinate; loop coefficients
    # must agree on every bucket.
    accesses = []
    used_names: dict[str, int] = {}
    for index, (memref, mode, mtype, coords0, texts0) in enumerate(base):
        variables_used: set[int] = set()
        coordinate_terms = []
        coordinate_offsets = []
        for c, (loop_coefs, c0) in enumerate(coords0):
            work_coefs = []
            for d in range(n_work):
                unit = tuple(1 if i == d else 0 for i in range(n_work))
                if unit not in per_bucket:
                    raise LayoutDerivationError(f"work item {unit} is missing")
                work_coefs.append(per_bucket[unit][index][3][c][1] - c0)
            for work_id, forms in per_bucket.items():
                coefs, const = forms[index][3][c]
                if coefs != loop_coefs:
                    raise LayoutDerivationError(
                        f"{memref}: coordinate {c} has loop coefficients that "
                        "differ across work items"
                    )
                expected = c0 + sum(a * w for a, w in zip(work_coefs, work_id))
                if const != expected:
                    raise LayoutDerivationError(
                        f"{memref}: coordinate {c} ({forms[index][4][c]!r} at work "
                        f"item {work_id}) is not affine in the work id"
                    )
            terms = [(a, d) for d, a in enumerate(work_coefs) if a]
            terms += [(a, n_work + p) for p, a in loop_coefs.items() if a]
            terms = [(a, v) for a, v in terms if variables[v].extent > 1]
            if any(a < 0 for a, _v in terms):
                raise LayoutDerivationError(
                    f"{memref}: coordinate {texts0[c]!r} has a negative coefficient"
                )
            reach = 0
            for a, v in sorted(terms):
                if a <= reach:
                    raise LayoutDerivationError(
                        f"{memref}: coordinate {texts0[c]!r} is not injective in "
                        f"its variables (coefficient {a} of {variables[v].name} "
                        f"overlaps the {reach + 1} values below it)"
                    )
                reach += a * (variables[v].extent - 1)
            variables_used.update(v for _a, v in terms)
            coordinate_terms.append(tuple(sorted(terms)))
            coordinate_offsets.append(int(c0))
        if mtype is None:
            raise LayoutDerivationError(f"{memref}: element type is not retained")
        name = memref
        if name in used_names:
            used_names[name] += 1
            name = f"{memref}#{used_names[memref]}"
        else:
            used_names[name] = 0
        accesses.append(
            DerivedAccess(
                name=name,
                memref=memref,
                mode=mode,
                variables=tuple(sorted(variables_used)),
                element_bytes=_element_bytes(mtype, memref),
                reassociation_allowed=(
                    True if mode is AccessMode.READ else reassociation
                ),
                coordinates=tuple(texts0),
                terms=tuple(coordinate_terms),
                offsets=tuple(coordinate_offsets),
            )
        )

    cost_per_level, weights = _cost_per_level(bound_cost, level_names)
    level_bits = []
    for level in range(len(level_names)):
        level_bits.append(
            sum(variables[d].bits for d in range(n_work) if work_levels[d] == level)
        )
    coordinate_problem = CoordinateLayoutProblem(
        variables,
        tuple(level_bits),
        tuple(
            CoordinateAccess(
                a.name,
                a.variables,
                a.mode,
                a.element_bytes,
                cost_per_level,
                a.reassociation_allowed,
            )
            for a in accesses
        ),
    )
    owner_bits = sum(level_bits)
    hierarchy = owner_hierarchy_from_target(
        target,
        tuple(level_names),
        local_bits=coordinate_problem.carrier_bits - owner_bits,
        active_extents=active_extents or None,
    )
    if tuple(level.bit_count for level in hierarchy.levels) != tuple(level_bits):
        raise LayoutDerivationError(
            f"owner hierarchy bits {[l.bit_count for l in hierarchy.levels]} differ "
            f"from the work grid's {level_bits}"
        )
    return DerivedLayoutProblem(
        target_name=target_name,
        level_names=tuple(level_names),
        unmodelled_levels=unmodelled,
        active_extents=tuple(sorted(active_extents.items())),
        variables=variables,
        work_levels=tuple(work_levels),
        accesses=tuple(accesses),
        cost_per_level=cost_per_level,
        movement_weights=weights,
        assumptions=tuple(dict.fromkeys(assumptions)),
        hierarchy=hierarchy,
        coordinate_problem=coordinate_problem,
        op_name=next(iter(ops)),
    )


def _resolve_opaque(match, op, memref, coords, parsed, opaque, loop_pos, assumptions):
    """Resolve a contraction matrix row retained only as an SSA name.

    Applies only when: the operand is an input (not the carried accumulator)
    of an accumulation with resolved axis roles and a zero-initialized scalar
    accumulator; the opaque value is one whole coordinate; the operand's other
    coordinates are exactly the reduction loops (so the operand is the one
    that carries the result row, whichever input the roles happen to name
    ``matrix``, release ledger X-41); and the accumulator has a single
    one-coordinate output sink.  The coordinate then equals the sink row,
    proven by the matcher's resolved output map.  Anything else is rejected.
    """

    roles = match.extra.get("axis_roles") or {}
    names = sorted({name for _i, name in opaque})
    reason = f"{memref}: index {names} is an SSA value the matcher does not retain"
    if op is None or op.is_loop_carried:
        raise LayoutDerivationError(reason)
    if (
        roles.get("unresolved") is not None
        or match.extra.get("accumulator_init") != "zero"
    ):
        raise LayoutDerivationError(reason)
    positions = sorted({i for i, _name in opaque})
    if len(positions) != 1:
        raise LayoutDerivationError(reason)
    (position,) = positions
    coefs, const = parsed[position]
    if const != 0 or list(coefs.values()) != [1]:
        raise LayoutDerivationError(reason)
    reduction = set(roles.get("reduction") or ())
    others = [parsed[i] for i in range(len(parsed)) if i != position]
    if sorted(k for c, _k in others for k in c) != sorted(reduction) or any(
        k for _c, k in others
    ):
        raise LayoutDerivationError(reason)
    sinks = match.extra.get("output_sinks") or []
    if len(sinks) != 1 or len(_split_top_level(sinks[0][1])) != 1:
        raise LayoutDerivationError(reason)
    sink_form = _linear_form(sinks[0][1], sinks[0][0])
    if any(name not in loop_pos for name in sink_form[0]):
        raise LayoutDerivationError(reason)
    from .spmw_match_engine import output_map_problem

    problem = output_map_problem(match)
    if problem is not None:
        # The matcher resolved the operand and store coordinates and they do
        # not coincide (C-23): the coordinate is not the sink row.
        raise LayoutDerivationError(f"{reason}; {problem}")
    assumptions.append(
        f"{memref} coordinate {position} ({coords[position]!r}, an SSA value the "
        f"layout derivation reads only by name) equals the store row of "
        f"{sinks[0][0]}: the matcher's resolved coordinates "
        "(spmw_match_engine.output_map_problem) prove the identity output map"
    )
    resolved = list(parsed)
    resolved[position] = sink_form
    return resolved


# --------------------------------------------------------------------- #
# Row-map realization (ROW_MAP_TARGETS)
# --------------------------------------------------------------------- #


def split_movement_layout(layout):
    """``(backend part, movement part or None)`` of a carried layout.

    The movement part is the block of ``mm.``-prefixed dims that the
    selection stage joins to the backend layout with ``product``.
    """

    from .spmw_linear_layout import LinearLayout

    if not isinstance(layout, LinearLayout):
        return layout, None
    keep = [
        i for i, n in enumerate(layout.out_dims) if not n.startswith(LAYOUT_DIM_PREFIX)
    ]
    moved = [
        i for i, n in enumerate(layout.out_dims) if n.startswith(LAYOUT_DIM_PREFIX)
    ]
    if not moved:
        return layout, None

    def part(indices, prefixed):
        return LinearLayout(
            {
                name: [tuple(v[i] for i in indices) for v in vectors]
                for name, vectors in layout.bases.items()
                if name.startswith(LAYOUT_DIM_PREFIX) == prefixed
            },
            tuple(layout.out_dims[i] for i in indices),
            tuple(layout.out_sizes[i] for i in indices),
        )

    return part(keep, False), part(moved, True)


def _split_reduction_bits(derived: DerivedLayoutProblem, matrix, acc) -> list[str]:
    """Owner bits of ``matrix`` taken from variables ``acc`` does not read."""

    problem = derived.coordinate_problem
    innermost = derived.owner_flag(matrix)[-1]
    split = []
    for v, variable in enumerate(derived.variables):
        if v in acc.variables:
            continue
        offset = problem.bit_offset(v)
        bits = [
            b for b in range(variable.bits) if not innermost.contains(1 << (offset + b))
        ]
        if bits:
            split.append(f"{variable.name}[{','.join(map(str, bits))}]")
    return split


def owner_rows(derived: DerivedLayoutProblem, matrix) -> dict[int, list[int]]:
    """Output rows per flat owner unit of a row-only owner map.

    Enumerates the active values of the accumulator's variables (the other
    variables at 0: the map must keep them local, which
    :func:`row_map_unrealizable_reason` checks) and evaluates the
    accumulator's single coordinate ``c0 + sum a_v v`` at each.  Returns
    ``{unit: ascending rows}``; the unit is :meth:`owner_unit`'s flat index
    (outer level most significant).  Raises :class:`LayoutDerivationError`
    when the accumulator has no single coordinate or too many rows.
    """

    acc = derived.accumulator()
    if acc is None or len(acc.terms) != 1 or len(acc.offsets) != 1:
        raise LayoutDerivationError(
            "the accumulator has no single output-row coordinate"
        )
    variables = [v for v in acc.variables if derived.variables[v].extent > 1]
    count = math.prod(derived.variables[v].extent for v in variables)
    if count > _MAX_OWNER_ROWS:
        raise LayoutDerivationError(
            f"{count} output rows exceed the row-map enumeration cap {_MAX_OWNER_ROWS}"
        )
    coefficient = {v: a for a, v in acc.terms[0]}
    inverse = matrix.inverse()
    units: dict[int, list[int]] = {}
    for point in itertools.product(
        *(range(derived.variables[v].extent) for v in variables)
    ):
        values = dict(zip(variables, point))
        row = acc.offsets[0] + sum(coefficient.get(v, 0) * x for v, x in values.items())
        units.setdefault(derived.owner_unit(inverse, values), []).append(row)
    return {unit: sorted(rows) for unit, rows in sorted(units.items())}


def reduction_partition_split(derived: DerivedLayoutProblem, matrix) -> int | None:
    """``P`` when ``matrix`` is the AiM ``reduction_partitions`` split, else None.

    The knob lowers reduction element ``t`` on channel group ``t % P`` and
    keeps every output row in every group (a host merge adds the ``P``
    partial vectors; ``aim_lowering.AimContraction``).  In owner-map terms:
    one owner level, one reduction loop (a variable the accumulator does not
    read), and the units are named by exactly that loop's ``log2 P`` low
    bits.  The flag comparison accepts any relabeling of the groups.
    """

    if derived.target_name != "aim" or len(derived.level_names) != 1:
        return None
    acc = derived.accumulator()
    if derived.op_name != "MAC" or acc is None or len(acc.coordinates) != 1:
        return None
    reductions = [
        v
        for v, variable in enumerate(derived.variables)
        if v not in acc.variables and variable.bits
    ]
    if len(reductions) != 1:
        return None
    (k,) = reductions
    (width,) = derived.coordinate_problem.level_bits
    variable = derived.variables[k]
    if width == 0 or width > variable.bits or variable.extent % (1 << width):
        return None
    if derived.owner_flag(matrix) != derived.owner_flag(
        reduction_partition_assignment(derived, k, width).matrix(
            derived.coordinate_problem
        )
    ):
        return None
    return 1 << width


def reduction_partition_assignment(derived, k, width) -> CoordinateAssignment:
    """The owner map naming the units by the low ``width`` bits of ``k``."""

    level = tuple(
        tuple(range(width)) if v == k else () for v in range(len(derived.variables))
    )
    return CoordinateAssignment((level,))


def _reduction_split_reason(derived, acc, split) -> str:
    owners = "+".join(split)
    head = (
        f"it splits the reduction of each {acc.memref} row across owner units "
        f"(owner bits from {owners}); "
    )
    if derived.target_name == "upmem":
        return head + (
            "UPMEM accumulates in int32, so the reduction policy permits the "
            "regrouping, but the UPMEM templates do not combine partial sums "
            "across DPUs (each DPU writes whole output rows), so this "
            "reduction split is not realized"
        )
    if derived.target_name == "aim":
        return head + (
            "the AiM lowering realizes a reduction split only as the "
            "reduction_partitions split (every owner bit from the low bits of "
            "the one reduction loop: element t on channel group t % P, every "
            "row in every group, host merge), and this tie is not that split"
        )
    return head + (
        "a reduction split needs a partial-sum combine, which belongs to the "
        "reduction policy, not to the row placement"
    )


def row_map_unrealizable_reason(derived: DerivedLayoutProblem, matrix) -> str | None:
    """Why a row-map target cannot realize ``matrix``; None when it can.

    Realizable: the kernel is a contraction (``MAC``) with one accumulator
    of one output-row coordinate, every bit of a variable the accumulator
    does not depend on (the reduction) is local at the innermost owner
    level, so each row's reduction stays in one unit, and every unit owns
    the same number of rows: on Samsung the row variables are not masked;
    on UPMEM and AiM the work grid is one-dimensional (one work item per
    DPU or output-frontier block) and the active rows divide evenly
    (:func:`owner_rows`).  An AiM ``reduction_partitions`` split
    (:func:`reduction_partition_split`) is realizable.
    """

    if derived.target_name not in ROW_MAP_TARGETS:
        return f"{derived.target_name} does not place rows by the owner map"
    acc = derived.accumulator()
    if derived.op_name != "MAC" or acc is None:
        return (
            f"only contractions are placed by the owner map (op {derived.op_name}); "
            "the elementwise placement is fixed"
        )
    if len(acc.coordinates) != 1:
        return (
            f"the accumulator {acc.memref} has {len(acc.coordinates)} coordinates; "
            "the row map places one output-row index"
        )
    if reduction_partition_split(derived, matrix) is not None:
        return None
    split = _split_reduction_bits(derived, matrix, acc)
    if split:
        return _reduction_split_reason(derived, acc, split)
    if derived.target_name in _UNMASKED_ROW_TARGETS:
        masked = [
            derived.variables[v].name
            for v in acc.variables
            if derived.variables[v].masked
        ]
        if masked:
            return (
                f"the row variables {masked} are masked, so units would own unequal "
                "row counts"
            )
        return None
    if len(derived.work_levels) != 1:
        return (
            f"the work grid has {len(derived.work_levels)} axes; the "
            f"{derived.target_name} row map places rows over a one-dimensional "
            "work grid (one work item per unit)"
        )
    expected = 1 << sum(derived.coordinate_problem.level_bits)
    n_work = len(derived.work_levels)
    row_loops = {v for v in acc.variables if v >= n_work}
    batched = [
        a.name
        for a in derived.accesses
        if a.mode is AccessMode.READ
        and set(a.variables) & row_loops
        and not set(acc.variables) <= set(a.variables)
    ]
    if batched:
        return (
            f"the inputs {batched} read the row loops without the whole row index "
            f"(a batched contraction); the {derived.target_name} row map places "
            "the rows of one batch"
        )
    try:
        own = owner_rows(derived, derived.program_layout())
        units = owner_rows(derived, matrix)
    except LayoutDerivationError as error:
        return str(error)
    per_item = len(own.get(0, ()))
    base = own[0][0] if per_item else 0
    if sorted(own) != list(range(expected)) or any(
        rows != list(range(base + u * per_item, base + (u + 1) * per_item))
        for u, rows in own.items()
    ):
        return (
            "the work grid does not partition the output rows into consecutive "
            "per-work-item blocks in work-id order (a replicated or interleaved "
            f"work grid); the {derived.target_name} row map relocates the rows "
            "of such a partition"
        )
    counts = sorted({len(rows) for rows in units.values()})
    if len(units) != expected or len(counts) != 1:
        held = "/".join(map(str, counts))
        return (
            f"its units own unequal row counts ({len(units)} of {expected} units "
            f"own {held} rows, {expected - len(units)} own none); the "
            f"{derived.target_name} lowering gives every work item's unit the "
            "same row count"
        )
    return None


def row_distributions(derived: DerivedLayoutProblem):
    """Per-level BLOCK/CYCLIC distributions of the output-row index.

    Returns ``[(name, CoordinateAssignment)]``, one per choice of BLOCK or
    CYCLIC at each owner level (outermost first).  The row bits are the
    accumulator coordinate's variable bits ordered by significance; a BLOCK
    level takes the most significant remaining bits, a CYCLIC level the
    least significant.  Empty when the accumulator has no single row
    coordinate or too few row bits.
    """

    acc = derived.accumulator()
    if acc is None or len(acc.terms) != 1:
        return []
    bits = [
        (v, b)
        for _coefficient, v in acc.terms[0]
        for b in range(derived.variables[v].bits)
    ]
    level_bits = derived.coordinate_problem.level_bits
    if sum(level_bits) > len(bits):
        return []
    out = []
    for kinds in itertools.product(("block", "cyclic"), repeat=len(level_bits)):
        low, high, levels = 0, len(bits), []
        for width, kind in zip(level_bits, kinds):
            if kind == "block":
                taken, high = bits[high - width : high], high - width
            else:
                taken, low = bits[low : low + width], low + width
            per_variable = [[] for _ in derived.variables]
            for v, b in taken:
                per_variable[v].append(b)
            levels.append(tuple(tuple(sorted(p)) for p in per_variable))
        name = "rows " + ", ".join(
            f"{kind} over {level}" for kind, level in zip(kinds, derived.level_names)
        )
        out.append((name, CoordinateAssignment(tuple(levels))))
    return out


def _reduction_partition_class(derived, min_cost, seen):
    """The AiM ``reduction_partitions`` member of a minimum class, or None."""

    acc = derived.accumulator()
    if derived.target_name != "aim" or acc is None or len(derived.level_names) != 1:
        return None
    reductions = [
        v
        for v, variable in enumerate(derived.variables)
        if v not in acc.variables and variable.bits
    ]
    if len(reductions) != 1:
        return None
    problem = derived.coordinate_problem
    (width,) = problem.level_bits
    assignment = reduction_partition_assignment(derived, reductions[0], width)
    if width > derived.variables[reductions[0]].bits:
        return None
    matrix = assignment.matrix(problem)
    partitions = reduction_partition_split(derived, matrix)
    flag = derived.owner_flag(matrix)
    if partitions is None or flag in seen:
        return None
    cost, profile = coordinate_assignment_cost(problem, assignment)
    if cost is None or cost != min_cost:
        return None
    seen.add(flag)
    return (
        cost,
        profile,
        matrix,
        1,
        f"reduction split t % {partitions} over {derived.level_names[0]} "
        f"(reduction_partitions={partitions}): {derived.describe_assignment(assignment)}",
        False,
    )


def _realization(derived, matrix, program_flag, program, description):
    """``(realizable, reason)`` of one tie (module docstring, Realization)."""

    if derived.owner_flag(matrix) == program_flag:
        return True, None
    partitions = reduction_partition_split(derived, matrix)
    if partitions is not None:
        return True, None
    if derived.target_name in ROW_MAP_TARGETS:
        reason = row_map_unrealizable_reason(derived, matrix)
        if reason is None:
            return True, None
        return False, (
            f"{derived.target_name} places contraction rows by the owner map but "
            f"cannot realize owner map {description}: {reason}"
        )
    return False, (
        f"{derived.target_name} emits one program per work item on the "
        "unit its work id names and does not re-map iterations to owners; "
        f"this tie needs owner map {description}, the program has "
        f"{derived.describe_assignment(program)}"
    )


# --------------------------------------------------------------------- #
# Selection-stage analysis
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class MovementSelectionPolicy:
    """Policy of the layout-construction stage.

    ``search`` is the ported search policy; its ``max_search_nodes``
    (default 20,672, the size of the largest bounded domain) is also the
    node budget of the coordinate search.  ``max_tie_candidates`` caps how
    many tie classes are crossed with the backend candidates (the rest are
    counted, and the cap is reported).  ``enabled=False`` skips the stage.
    """

    enabled: bool = True
    search: LayoutSearchPolicy = field(default_factory=LayoutSearchPolicy)
    max_tie_candidates: int = 64
    max_variable_choices: int = DEFAULT_MAX_VARIABLE_CHOICES


DEFAULT_MOVEMENT_POLICY = MovementSelectionPolicy()

_ACTIVE_POLICY: contextvars.ContextVar = contextvars.ContextVar(
    "spmw_movement_selection_policy", default=None
)


@contextlib.contextmanager
def movement_selection_policy(policy: MovementSelectionPolicy):
    """Use ``policy`` for every ranking inside the ``with`` block."""

    if not isinstance(policy, MovementSelectionPolicy):
        raise TypeError("policy must be a MovementSelectionPolicy")
    token = _ACTIVE_POLICY.set(policy)
    try:
        yield policy
    finally:
        _ACTIVE_POLICY.reset(token)


def active_movement_policy(policy=None) -> MovementSelectionPolicy:
    if policy is not None:
        return policy
    return _ACTIVE_POLICY.get() or DEFAULT_MOVEMENT_POLICY


@dataclass(frozen=True)
class MovementTie:
    """One tie class handed to the cost ranking.

    A class is one locality flag (bounded search) or one movement profile
    (coordinate search); its members have identical movement.
    ``representative`` is the program layout when the class contains it.
    """

    index: int
    cost: Fraction
    profile: tuple
    work_to_iteration: GF2Matrix
    layout: Any  # LinearLayout over mm.-prefixed dims
    multiplicity: int
    realizable: bool
    reason: str | None
    description: str


@dataclass(frozen=True)
class KernelMovement:
    # pylint: disable=too-many-instance-attributes
    """Movement analysis of one kernel group, as recorded on the ranking.

    ``optimality_proven`` holds only when the search finished and its domain
    is every one-to-one layout.  ``incomplete`` marks a node-budget exit.
    ``chosen_cost`` / ``chosen_label`` describe the ranked winner.
    """

    status: str  # "analyzed" | "not-analyzed" | "disabled"
    reason: str | None = None
    implementation: str | None = None
    termination: str | None = None
    optimality_proven: bool = False
    optimality_scope: str | None = None
    incomplete: bool = False
    min_cost: Fraction | None = None
    program_cost: Fraction | None = None
    program_label: str | None = None
    chosen_cost: Fraction | None = None
    chosen_label: str | None = None
    tie_count: int = 0
    tie_layouts: int | None = None
    ties_crossed: int = 0
    visited: int = 0
    pruned: int = 0
    movement_weights: str | None = None
    assumptions: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    ties: tuple[MovementTie, ...] = ()
    program_layout: Any = None
    derived: DerivedLayoutProblem | None = field(default=None, repr=False)

    @property
    def label(self) -> str:
        if self.status == "not-analyzed":
            return f"not-analyzed: {self.reason}"
        return self.status

    @property
    def realizable_tie_count(self) -> int:
        """Tie classes the backend executes (the rest carry a reason)."""
        return sum(1 for tie in self.ties if tie.realizable)


def _label(cost, min_cost, proven, termination, scope):
    if min_cost is None or cost is None or cost > min_cost:
        return "not-minimal"
    if proven:
        return "minimal"
    if termination == LayoutSearchTermination.NODE_BUDGET.value:
        return f"minimal (unproven: {termination})"
    return f"minimal over {scope}"


def analyze_kernel_movement(
    target, matches, bound_cost=None, policy=None
) -> KernelMovement:
    """Derive, search and classify the movement ties of one kernel group."""

    policy = active_movement_policy(policy)
    if not policy.enabled:
        return KernelMovement(status="disabled", reason="movement stage disabled")
    try:
        derived = derive_layout_problem(target, matches, bound_cost=bound_cost)
    except (LayoutDerivationError, TargetHierarchyError) as error:
        return KernelMovement(status="not-analyzed", reason=str(error))
    try:
        return _search_and_classify(derived, policy)
    except (LayoutSearchDomainError, NoLegalLayoutError) as error:
        return KernelMovement(
            status="not-analyzed",
            reason=f"{type(error).__name__}: {error}",
            movement_weights=derived.movement_weights,
            assumptions=derived.assumptions,
            derived=derived,
        )


def _search_and_classify(derived: DerivedLayoutProblem, policy) -> KernelMovement:
    problem = derived.coordinate_problem
    program = derived.program_assignment()
    program_matrix = program.matrix(problem)
    program_flag = derived.owner_flag(program_matrix)
    program_cost, program_profile = coordinate_assignment_cost(problem, program)
    search = policy.search
    bounded = (
        derived.carrier_bits <= search.max_carrier_bits
        and len(derived.level_names) <= search.max_levels
    )
    notes = []
    classes = []  # (cost, profile, matrix, multiplicity, description, is_program)
    if bounded:
        result = search_min_movement_layout(derived.search_problem(), search)
        implementation = result.implementation.value
        termination = result.termination
        proven = result.optimality_proven
        scope = SCOPE_ALL_LAYOUTS
        visited, pruned = result.visited, result.pruned
        min_cost = result.cost
        accesses = derived.layout_accesses()
        active = derived.search_problem().active_points
        by_flag: dict = {}
        for matrix, flag in zip(result.ties, result.tie_flags):
            by_flag.setdefault(flag.spaces[1:], []).append(matrix)
        for spaces, members in by_flag.items():
            is_program = spaces == program_flag
            representative = program_matrix if is_program else members[0]
            terms = layout_movement_cost(
                accesses, derived.hierarchy, representative, active
            ).terms
            profile = tuple(
                (t.access, t.owner_level, t.extra_owner_incidence) for t in terms
            )
            classes.append(
                (
                    result.cost,
                    profile,
                    representative,
                    len(members),
                    (
                        "program layout"
                        if is_program
                        else f"flag dims " f"{[s.rank for s in spaces]}"
                    ),
                    is_program,
                )
            )
        tie_layouts = len(result.ties) * result.tie_multiplicity
    else:
        result = search_coordinate_min_movement(
            problem,
            max_search_nodes=search.max_search_nodes,
            max_variable_choices=policy.max_variable_choices,
        )
        implementation = "coordinate"
        termination = result.termination
        proven = result.optimality_proven
        scope = result.optimality_scope
        visited, pruned = result.visited, result.illegal
        min_cost = result.cost
        if (
            program_cost is not None
            and min_cost is not None
            and program_cost < min_cost
        ):
            notes.append(
                f"the program layout ({program_cost}) beats the search incumbent "
                f"({min_cost}) found before the {termination.value} exit"
            )
        for tie in result.ties:
            is_program = tie.profile == program_profile and program_cost == tie.cost
            assignment = program if is_program else tie.assignment
            classes.append(
                (
                    tie.cost,
                    tie.profile,
                    assignment.matrix(problem),
                    tie.multiplicity,
                    derived.describe_assignment(assignment),
                    is_program,
                )
            )
        tie_layouts = sum(t.multiplicity for t in result.ties)
        if scope == SCOPE_COORDINATE_LAYOUTS:
            notes.append(
                "masked domain: exact over coordinate layouts (owner bits are "
                "iteration bits); non-coordinate layouts are not searched"
            )
        notes.append(
            "tie_layouts counts coordinate owner-bit assignments; non-coordinate "
            "layouts with the same movement exist and are not counted"
        )

    if derived.target_name in ROW_MAP_TARGETS and min_cost is not None:
        # Realizable members of a minimum class that differ in which rows
        # share a unit are distinct placements on a row-map target: cross
        # them separately (multiplicity 1; they stay counted in tie_layouts).
        seen = {derived.owner_flag(c[2]) for c in classes}
        added = 0
        for name, assignment in row_distributions(derived):
            cost, profile = coordinate_assignment_cost(problem, assignment)
            if cost is None or cost != min_cost:
                continue
            matrix = assignment.matrix(problem)
            flag = derived.owner_flag(matrix)
            if flag in seen or row_map_unrealizable_reason(derived, matrix):
                continue
            seen.add(flag)
            added += 1
            classes.append(
                (
                    cost,
                    profile,
                    matrix,
                    1,
                    f"{name}: {derived.describe_assignment(assignment)}",
                    False,
                )
            )
        if added:
            notes.append(
                f"{added} row distribution(s) of a minimum tie class crossed as "
                "separate ties: the backend places rows by the owner map"
            )
        partition = _reduction_partition_class(derived, min_cost, seen)
        if partition is not None:
            classes.append(partition)
            notes.append(
                "the cyclic reduction split of a minimum tie class crossed as a "
                "separate tie: AiM realizes it with the reduction_partitions "
                "candidates of the same P"
            )

    ties = []
    for index, (
        cost,
        profile,
        matrix,
        multiplicity,
        description,
        is_program,
    ) in enumerate(sorted(classes, key=lambda c: (not c[5], repr(c[1]), c[2].rows))):
        realizable, reason = _realization(
            derived, matrix, program_flag, program, description
        )
        ties.append(
            MovementTie(
                index=index,
                cost=cost,
                profile=profile,
                work_to_iteration=matrix,
                layout=derived.linear_layout(matrix),
                multiplicity=multiplicity,
                realizable=realizable,
                reason=reason,
                description=description,
            )
        )
    incomplete = termination is LayoutSearchTermination.NODE_BUDGET
    program_label = _label(program_cost, min_cost, proven, termination.value, scope)
    if program_cost is None:
        program_label = "illegal (splits a strict reduction)"
    return KernelMovement(
        status="analyzed",
        implementation=implementation,
        termination=termination.value,
        optimality_proven=proven,
        optimality_scope=scope,
        incomplete=incomplete,
        min_cost=min_cost,
        program_cost=program_cost,
        program_label=program_label,
        tie_count=len(ties),
        tie_layouts=tie_layouts,
        visited=visited,
        pruned=pruned,
        movement_weights=derived.movement_weights,
        assumptions=derived.assumptions,
        notes=tuple(notes),
        ties=tuple(ties),
        program_layout=derived.linear_layout(program_matrix),
        derived=derived,
    )


def label_for_cost(movement: KernelMovement, cost) -> str:
    """``minimal`` / ``minimal (unproven: ...)`` / ``minimal over ...`` /
    ``not-minimal`` for a layout of movement ``cost``."""

    return _label(
        cost,
        movement.min_cost,
        movement.optimality_proven,
        movement.termination,
        movement.optimality_scope,
    )


def with_chosen(movement: KernelMovement, cost, label, **changes) -> KernelMovement:
    return replace(movement, chosen_cost=cost, chosen_label=label, **changes)


__all__ = [
    "COMPUTE_LEVELS",
    "DEFAULT_MOVEMENT_POLICY",
    "DerivedAccess",
    "DerivedLayoutProblem",
    "KernelMovement",
    "LayoutDerivationError",
    "MovementSelectionPolicy",
    "MovementTie",
    "ROW_MAP_TARGETS",
    "analyze_kernel_movement",
    "derive_layout_problem",
    "label_for_cost",
    "movement_selection_policy",
    "owner_rows",
    "reduction_partition_assignment",
    "reduction_partition_split",
    "row_distributions",
    "row_map_unrealizable_reason",
    "split_movement_layout",
]
