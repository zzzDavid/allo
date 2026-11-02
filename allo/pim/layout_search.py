# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimum-movement layout search (paper Section IV-C, Algorithm 1).

The objective is ``C_move`` of :mod:`allo.pim.layout_movement`.  The cost of
a layout depends only on its locality chain (flag)
``I = T_0 >= T_1 >= ... >= T_h`` with ``T_s = ker Q_s``.  The search builds
the chain one basis vector at a time as nested relative-RREF prefixes,
starting from the directions forced to be local, and completes a full layout
only at the end.

Prefix bound (Eq. pim-minmove-bound).  If ``T_s`` has ``q`` of its
``ell_s`` basis vectors chosen, spanning ``B``, then for every completion::

    r_{j,s} >= dim K_j - min(dim K_j, dim(B intersect K_j) + ell_s - q)

Levels already completed use their exact ``T_s``; levels not yet started use
their forced-local space as ``B``.  Substituting the optimistic ranks into the
cost never overestimates a completion.

Search.  Best-first over ``(lower bound, canonical prefix key)``; a node is
dropped only when its bound *strictly* exceeds the best complete cost, so
every tied layout is retained.  A node budget counts the root, every
evaluated prefix and every materialized layout.  On budget exit the result
carries the best complete layouts seen and the lower bounds of the
unexplored frontier, and does not claim optimality.

Domains.  Three implementations share the prefix tree and the bound:

* the bounded search (at most four carrier bits and three owner levels)
  materializes every layout of a tied flag and is complete over the legal
  domain;
* the masked mode for partial iteration spaces uses exact active-owner
  counts, disables the rank bound and is exact only on exhaustion;
* the scalable flag search (up to 32 bits) stops at flags, reports one
  representative per tied flag and supports budget exits on large carriers.

:func:`search_min_movement_layout` dispatches between them.

Ported from the snapshot (df35093) modules ``canonical_layout_enumeration``,
``paper_partial_basis_search``, ``admissible_layout_search`` and
``scalable_layout_synthesis``; replay, receipt, SHA-256 and exact-type
guards were dropped, the traversal, bound, pruning, budget accounting and
telemetry are unchanged.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from heapq import heappop, heappush
from itertools import combinations, product

from .gf2 import GF2Matrix, GF2Subspace
from .layout_movement import (
    LayoutAccess,
    MovementTerm,
    full_carrier_rank_incidence,
    scale_extra_owner_incidence,
    strict_reduction_kernel,
)
from .layout_problem import LocalSpaceFlag, OwnerHierarchy

# Implemented limits of the bounded search, not tuning knobs.
MAX_BOUNDED_CARRIER_BITS = 4
MAX_BOUNDED_LEVELS = 3
# 1 root + 511 prefix nodes + 20,160 layouts: exactly the largest bounded
# domain (four bits, three one-bit owner levels).
DEFAULT_MAX_SEARCH_NODES = 20_672


class LayoutSearchTermination(str, Enum):
    """How a search ended.  Only the first two prove optimality."""

    EXHAUSTED = "exhausted"
    BOUND_PROVED = "bound-proved"
    NODE_BUDGET = "node-budget"


class LayoutSearchImplementation(str, Enum):
    """Which search produced a result."""

    BOUNDED = "bounded"
    BOUNDED_MASKED = "bounded-masked"
    SCALABLE = "scalable"


class LayoutSearchDomainError(ValueError):
    """The problem is outside every implemented search domain."""


class NoLegalLayoutError(ValueError):
    """The forced-local and strict-reduction constraints admit no layout."""


@dataclass(frozen=True)
class LayoutSearchPolicy:
    """Search policy.

    ``max_carrier_bits`` and ``max_levels`` select the bounded search domain
    (implemented up to 4 bits and 3 levels).  Problems outside it go to the
    scalable flag search when ``allow_scalable`` (full carriers only).
    ``max_search_nodes`` is the node budget (``None``: unbounded).
    ``enable_pruning=False`` turns the bounded search into an exhaustive
    enumeration with the same traversal, for comparisons.
    """

    max_carrier_bits: int = MAX_BOUNDED_CARRIER_BITS
    max_levels: int = MAX_BOUNDED_LEVELS
    max_search_nodes: int | None = DEFAULT_MAX_SEARCH_NODES
    enable_pruning: bool = True
    allow_scalable: bool = True

    def __post_init__(self) -> None:
        for name, limit in (
            ("max_carrier_bits", MAX_BOUNDED_CARRIER_BITS),
            ("max_levels", MAX_BOUNDED_LEVELS),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if not 1 <= value <= limit:
                raise ValueError(
                    f"{name} must be in [1, {limit}] (the implemented bounded "
                    "domain); larger problems use the scalable search"
                )
        budget = self.max_search_nodes
        if budget is not None and (
            isinstance(budget, bool) or not isinstance(budget, int) or budget < 1
        ):
            raise ValueError("max_search_nodes must be a positive integer or None")
        for name in ("enable_pruning", "allow_scalable"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")


@dataclass(frozen=True)
class LayoutSearchProblem:
    """A minimum-movement layout problem.

    ``hierarchy`` fixes the owner levels and local bits; its carrier is the
    iteration space.  ``forced_local`` optionally gives, per owner level
    (outermost first), directions the operation requires to stay inside one
    unit through that level; the spaces must be nested (``F_0 >= F_1 >=``).
    ``active_points`` selects a masked domain (sorted iteration points); the
    full range or ``None`` means every coordinate is active.
    """

    hierarchy: OwnerHierarchy
    accesses: tuple[LayoutAccess, ...]
    forced_local: tuple[GF2Subspace, ...] = ()
    active_points: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.hierarchy, OwnerHierarchy):
            raise TypeError("hierarchy must be an OwnerHierarchy")
        accesses = tuple(self.accesses)
        if not accesses:
            raise ValueError("a layout problem needs at least one access")
        if any(not isinstance(access, LayoutAccess) for access in accesses):
            raise TypeError("accesses must be LayoutAccess values")
        names = tuple(access.name for access in accesses)
        if len(set(names)) != len(names):
            raise ValueError(f"access names must be unique, got {names}")
        carrier = self.hierarchy.carrier_bit_count
        if carrier < 1:
            raise ValueError("the iteration carrier must have at least one bit")
        levels = len(self.hierarchy.levels)
        for access in accesses:
            if access.access_matrix.n_cols != carrier:
                raise ValueError(
                    f"access {access.name!r} reads {access.access_matrix.n_cols} "
                    f"iteration bits; the hierarchy carrier has {carrier}"
                )
            if len(access.cost_per_level) != levels:
                raise ValueError(
                    f"access {access.name!r} has {len(access.cost_per_level)} "
                    f"per-level costs for {levels} owner levels"
                )
        forced = tuple(self.forced_local)
        if forced and len(forced) != levels:
            raise ValueError(
                f"forced_local has {len(forced)} spaces for {levels} owner levels"
            )
        for level, space in enumerate(forced):
            if not isinstance(space, GF2Subspace) or space.dimension != carrier:
                raise ValueError(
                    f"forced_local[{level}] must be a GF2Subspace of dimension "
                    f"{carrier}"
                )
            if level and not space.is_subspace_of(forced[level - 1]):
                raise ValueError("forced_local spaces must be nested (F_s <= F_s-1)")
        active = self.active_points
        if active is not None:
            if isinstance(active, (str, bytes)):
                raise TypeError("active_points must be a sequence of integers")
            active = tuple(active)
            extent = 1 << carrier
            if (
                not active
                or any(
                    isinstance(point, bool)
                    or not isinstance(point, int)
                    or not 0 <= point < extent
                    for point in active
                )
                or active != tuple(sorted(set(active)))
            ):
                raise ValueError(
                    "active_points must be a non-empty strictly increasing "
                    f"subset of range({extent})"
                )
            if len(active) == extent:
                active = None
        object.__setattr__(self, "accesses", accesses)
        object.__setattr__(self, "forced_local", forced)
        object.__setattr__(self, "active_points", active)

    @property
    def carrier_bits(self) -> int:
        return self.hierarchy.carrier_bit_count

    @property
    def is_masked(self) -> bool:
        return self.active_points is not None


@dataclass(frozen=True)
class EvaluatedLayout:
    """One complete layout with its movement terms (``cost=None``: illegal)."""

    work_to_iteration: GF2Matrix
    flag: LocalSpaceFlag
    terms: tuple[MovementTerm, ...]
    cost: Fraction | None

    @property
    def legal(self) -> bool:
        return self.cost is not None


@dataclass(frozen=True)
class LayoutSearchTelemetry:
    # pylint: disable=too-many-instance-attributes
    """Accounting of the search domain.

    Layout counts always partition the domain on a full carrier:
    ``visited + pruned + unresolved == expected`` (bounded) or the same in
    flag units times ``layouts_per_flag`` (scalable).  ``search_nodes_used``
    is what the budget was charged.  ``fallback_incumbent`` marks a budget
    exit whose incumbent was completed outside the budget from the best
    frontier prefix because no complete layout had been reached.
    """

    expected_layouts: int
    expected_flags: int
    layouts_per_flag: int
    expected_prefix_nodes: int | None
    evaluated_prefix_nodes: int
    expanded_prefix_nodes: int
    visited_flags: int
    pruned_frontier_roots: int
    unresolved_frontier_roots: int
    pruned_prefix_descendants: int | None
    visited_layouts: int
    pruned_layouts: int
    unresolved_layouts: int
    illegal_layouts: int
    search_nodes_used: int
    fallback_incumbent: bool


@dataclass(frozen=True)
class LayoutSearchResult:
    # pylint: disable=too-many-instance-attributes
    """Outcome of :func:`search_min_movement_layout`.

    ``ties`` are ``work_to_iteration`` matrices (the paper's ``L``) attaining
    ``cost``.  For the bounded searches they are every minimum layout seen
    (``ties_complete=True``, ``tie_multiplicity=1``); for the scalable search
    they are one canonical representative per tied flag, each standing for
    ``tie_multiplicity`` layouts with the same movement.  ``tie_flags`` lists
    the distinct locality chains of the ties.  With ``optimality_proven``
    False (``NODE_BUDGET``) the ties are the best layouts found and
    ``remaining_lower_bounds`` bound every unexplored layout.
    """

    ties: tuple[GF2Matrix, ...]
    tie_flags: tuple[LocalSpaceFlag, ...]
    cost: Fraction | None
    termination: LayoutSearchTermination
    visited: int
    pruned: int
    remaining_lower_bounds: tuple[Fraction, ...]
    optimality_proven: bool
    implementation: LayoutSearchImplementation
    ties_complete: bool
    tie_multiplicity: int
    telemetry: LayoutSearchTelemetry
    problem: LayoutSearchProblem = field(repr=False)

    def default_in_dims(self) -> tuple[tuple[str, int], ...]:
        """Work-coordinate dims in bit order: owner axes, then local axes."""

        hierarchy = self.problem.hierarchy
        dims = []
        for level in hierarchy.levels:
            for axis in level.axes:
                dims.append(
                    (axis.diagnostic_name or f"owner{axis.ordinal}", axis.mapped_extent)
                )
        for axis in hierarchy.local_axes:
            dims.append((axis.diagnostic_name or f"local{axis.ordinal}", axis.extent))
        return tuple(dims)

    def to_linear_layouts(self, in_dims=None, out_dims=None) -> tuple:
        """Return the ties as :class:`allo.spmw_linear_layout.LinearLayout`.

        ``in_dims`` name the work-coordinate bits in order (default: the
        hierarchy's owner axes then local axes); ``out_dims`` name the
        iteration bits in order (default one dim ``"iteration"``).  Both are
        ``(name, power-of-two size)`` sequences or dicts.
        """

        # pylint: disable=import-outside-toplevel
        from ..spmw_linear_layout import LinearLayout

        if in_dims is None:
            in_dims = self.default_in_dims()
        if out_dims is None:
            out_dims = (("iteration", 1 << self.problem.carrier_bits),)
        return tuple(
            LinearLayout.from_gf2(matrix, in_dims, out_dims) for matrix in self.ties
        )


# --------------------------------------------------------------------------- #
# Counting helpers
# --------------------------------------------------------------------------- #


def gl_count(dimension: int) -> int:
    """``|GL(dimension, 2)|``: the number of invertible layouts."""

    result = 1
    for rank in range(dimension):
        result *= (1 << dimension) - (1 << rank)
    return result


def gaussian_binomial(dimension: int, rank: int) -> int:
    """Number of rank-``rank`` subspaces of ``GF(2)**dimension``."""

    if rank < 0 or rank > dimension:
        return 0
    values = [1]
    for ambient in range(1, dimension + 1):
        current = [1]
        for subspace_rank in range(1, ambient):
            current.append(
                values[subspace_rank]
                + (1 << (ambient - subspace_rank)) * values[subspace_rank - 1]
            )
        current.append(1)
        values = current
    return values[rank]


def layouts_per_flag(hierarchy: OwnerHierarchy) -> int:
    """Layouts sharing one flag: complements times ordered frames."""

    result = gl_count(hierarchy.local_bit_count)
    for level, owner in enumerate(hierarchy.levels):
        local_rank = hierarchy.local_space_dimensions[level + 1]
        result *= (1 << (owner.bit_count * local_rank)) * gl_count(owner.bit_count)
    return result


def _pivot(row: int) -> int:
    return (row & -row).bit_length() - 1


def _relative_extension_count(
    relative_dimension: int, relative_rows: tuple[int, ...], target_rank: int
) -> int:
    """Count RREF completions of ``relative_rows`` to ``target_rank`` rows.

    Dynamic program over eligible pivot columns from high to low: choosing
    pivot ``p`` with ``k`` later pivots leaves ``2**(d - 1 - p - k)`` free
    high entries in its row.
    """

    current_rank = len(relative_rows)
    if target_rank < current_rank or target_rank > relative_dimension:
        return 0
    remaining = target_rank - current_rank
    if remaining == 0:
        return 1
    last_pivot = _pivot(relative_rows[-1]) if relative_rows else -1
    counts = [0] * (remaining + 1)
    counts[0] = 1
    for column in range(relative_dimension - 1, last_pivot, -1):
        if any(row & (1 << column) for row in relative_rows):
            continue
        updated = list(counts)
        for later in range(remaining):
            if counts[later] == 0:
                continue
            free_high = relative_dimension - 1 - column - later
            if free_high < 0:
                continue
            updated[later + 1] += counts[later] * (1 << free_high)
        counts = updated
    return counts[remaining]


def _iter_relative_child_rows(
    relative_dimension: int, relative_rows: tuple[int, ...], target_rank: int
) -> Iterator[tuple[int, ...]]:
    """Yield every extendible one-row extension of an RREF prefix, lazily.

    A new row has a pivot ``p`` after the last pivot, unused by earlier rows,
    and arbitrary bits above ``p``.  It is extendible exactly when enough
    columns above ``p`` stay unused to host the remaining pivots, so only
    those rows are generated (the snapshot scanned every raw row and filtered
    with :func:`_relative_extension_count`; the child set is identical).
    """

    later_pivots = target_rank - len(relative_rows) - 1
    if later_pivots < 0:
        return
    used = 0
    for row in relative_rows:
        used |= row
    last_pivot = _pivot(relative_rows[-1]) if relative_rows else -1
    for pivot in range(last_pivot + 1, relative_dimension):
        if used & (1 << pivot):
            continue
        free = tuple(
            column
            for column in range(pivot + 1, relative_dimension)
            if not used & (1 << column)
        )
        if len(free) < later_pivots:
            break
        taken = tuple(
            column
            for column in range(pivot + 1, relative_dimension)
            if used & (1 << column)
        )
        for taken_bits in range(1 << len(taken)):
            high = 0
            for index, column in enumerate(taken):
                if taken_bits & (1 << index):
                    high |= 1 << column
            for count in range(len(free) - later_pivots + 1):
                for chosen in combinations(free, count):
                    row = (1 << pivot) | high
                    for column in chosen:
                        row |= 1 << column
                    yield relative_rows + (row,)


def _lift_relative_rows(
    complement: GF2Subspace, relative_rows: tuple[int, ...]
) -> GF2Subspace:
    ambient_rows = []
    for row in relative_rows:
        vector = 0
        for index, basis_vector in enumerate(complement.basis):
            if row & (1 << index):
                vector ^= basis_vector
        ambient_rows.append(vector)
    return GF2Subspace(complement.dimension, tuple(ambient_rows))


def _matrix_from_columns(columns: tuple[int, ...], dimension: int) -> GF2Matrix:
    return GF2Matrix(
        dimension,
        dimension,
        tuple(
            sum(
                ((column >> output_bit) & 1) << input_bit
                for input_bit, column in enumerate(columns)
            )
            for output_bit in range(dimension)
        ),
    )


# --------------------------------------------------------------------------- #
# Prepared domain and prefix tree (shared with the scalable search)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Node:
    """A normalized nested relative-RREF prefix.

    ``spaces`` are the completed ``T_0 .. T_level``; ``rows`` the relative
    RREF rows of ``T_{level+1}`` chosen so far, in coordinates of a fixed
    complement of the forced-local space inside ``T_level``.
    """

    level: int
    spaces: tuple[GF2Subspace, ...]
    rows: tuple[int, ...]

    @property
    def key(self) -> tuple[object, ...]:
        return (self.level, tuple(space.basis for space in self.spaces), self.rows)


@dataclass(frozen=True)
class _Bound:
    lower_cost: Fraction
    lower_ranks: tuple[int, ...]


class _Domain:
    """Validated problem plus the forced-local spaces the search uses."""

    def __init__(self, problem: LayoutSearchProblem, *, strict_as_forced: bool):
        self.problem = problem
        self.hierarchy = problem.hierarchy
        self.accesses = problem.accesses
        self.carrier = problem.carrier_bits
        self.level_count = len(self.hierarchy.levels)
        self.dims = self.hierarchy.local_space_dimensions
        self.kernels = tuple(access.kernel for access in self.accesses)
        self.unique_points = tuple(
            1 << (self.carrier - kernel.rank) for kernel in self.kernels
        )
        forced = problem.forced_local or tuple(
            GF2Subspace.zero(self.carrier) for _ in range(self.level_count)
        )
        if strict_as_forced:
            strict = strict_reduction_kernel(self.accesses, self.carrier)
            if strict.rank:
                forced = tuple(space.sum(strict) for space in forced)
                if self.level_count and strict.rank > self.dims[-1]:
                    raise NoLegalLayoutError(
                        "strict (non-reassociable) reductions have a kernel of "
                        f"dimension {strict.rank}, but the innermost owner "
                        f"level keeps only {self.dims[-1]} local bits; every "
                        "layout would split a reduction that may not be "
                        "regrouped"
                    )
        for level, space in enumerate(forced):
            if space.rank > self.dims[level + 1]:
                raise NoLegalLayoutError(
                    f"forced-local space of level {level} has dimension "
                    f"{space.rank} > dim T_{level + 1} = {self.dims[level + 1]}"
                )
        self.forced = forced
        self.layouts_per_flag = layouts_per_flag(self.hierarchy)
        self._complements: dict[tuple[int, GF2Subspace], GF2Subspace] = {}

    # -- prefix tree ------------------------------------------------------- #

    def complement(self, level: int, parent: GF2Subspace) -> GF2Subspace:
        key = (level, parent)
        cached = self._complements.get(key)
        if cached is None:
            cached = self.forced[level].basis_extension(within=parent)
            self._complements[key] = cached
        return cached

    def target_rank(self, level: int) -> int:
        return self.dims[level + 1] - self.forced[level].rank

    def normalize(
        self, level: int, spaces: tuple[GF2Subspace, ...], rows: tuple[int, ...] = ()
    ) -> _Node:
        """Collapse levels whose local space is entirely forced."""

        while level < self.level_count:
            forced = self.forced[level]
            if not forced.is_subspace_of(spaces[-1]):
                raise AssertionError("forced space left its completed parent")
            if self.target_rank(level):
                break
            if rows:
                raise AssertionError("forced-only level retained relative rows")
            spaces = spaces + (forced,)
            level += 1
        return _Node(level, spaces, rows)

    def root(self) -> _Node:
        return self.normalize(0, (GF2Subspace.full(self.carrier),))

    def is_terminal(self, node: _Node) -> bool:
        return node.level == self.level_count

    def iter_children(self, node: _Node) -> Iterator[_Node]:
        """Yield every immediate child exactly once (lazily)."""

        if self.is_terminal(node):
            return
        level = node.level
        complement = self.complement(level, node.spaces[-1])
        target_rank = self.target_rank(level)
        for child_rows in _iter_relative_child_rows(
            complement.rank, node.rows, target_rank
        ):
            if len(child_rows) < target_rank:
                yield self.normalize(level, node.spaces, child_rows)
            else:
                target = self.forced[level].sum(
                    _lift_relative_rows(complement, child_rows)
                )
                yield self.normalize(level + 1, node.spaces + (target,))

    def children(self, node: _Node) -> tuple[_Node, ...]:
        return tuple(sorted(self.iter_children(node), key=lambda item: item.key))

    def partial_basis(self, node: _Node, level: int) -> GF2Subspace:
        if level < node.level:
            return node.spaces[level + 1]
        if level == node.level:
            complement = self.complement(level, node.spaces[-1])
            return self.forced[level].sum(_lift_relative_rows(complement, node.rows))
        return self.forced[level]

    def bound(self, node: _Node) -> _Bound:
        """Literal paper bound; exact at a terminal (complete-flag) node."""

        bases = tuple(
            self.partial_basis(node, level) for level in range(self.level_count)
        )
        ranks = []
        cost = Fraction()
        for access, kernel, unique in zip(
            self.accesses, self.kernels, self.unique_points
        ):
            for level, basis in enumerate(bases):
                reachable = min(
                    kernel.rank,
                    kernel.intersection(basis).rank + self.dims[level + 1] - basis.rank,
                )
                lower_rank = kernel.rank - reachable
                ranks.append(lower_rank)
                if not access.is_strict_reduction:
                    cost += scale_extra_owner_incidence(
                        unique * ((1 << lower_rank) - 1),
                        element_bytes=access.element_bytes,
                        cost_per_byte=access.cost_per_level[level],
                    )
        return _Bound(cost, tuple(ranks))

    def flag(self, node: _Node) -> LocalSpaceFlag:
        return LocalSpaceFlag(self.hierarchy, node.spaces, self.forced)

    # -- counting ---------------------------------------------------------- #

    def deeper_flag_count(self, level: int) -> int:
        count = 1
        for item in range(level, self.level_count):
            forced_rank = self.forced[item].rank
            count *= gaussian_binomial(
                self.dims[item] - forced_rank, self.dims[item + 1] - forced_rank
            )
        return count

    def node_flag_count(self, node: _Node) -> int:
        if self.is_terminal(node):
            return 1
        complement = self.complement(node.level, node.spaces[-1])
        return _relative_extension_count(
            complement.rank, node.rows, self.target_rank(node.level)
        ) * self.deeper_flag_count(node.level + 1)

    def _relative_subtree_size(
        self, relative_dimension: int, rows: tuple[int, ...], target_rank: int, deeper
    ) -> int:
        if len(rows) == target_rank:
            return deeper
        return 1 + sum(
            self._relative_subtree_size(relative_dimension, child, target_rank, deeper)
            for child in _iter_relative_child_rows(
                relative_dimension, rows, target_rank
            )
        )

    def whole_tree_size(self, level: int) -> int:
        """Prefix nodes below a fresh level (same for every parent)."""

        if level == self.level_count:
            return 1
        target_rank = self.target_rank(level)
        deeper = self.whole_tree_size(level + 1)
        if target_rank == 0:
            return deeper
        relative_dimension = self.dims[level] - self.forced[level].rank
        return self._relative_subtree_size(relative_dimension, (), target_rank, deeper)

    def node_subtree_size(self, node: _Node) -> int:
        if self.is_terminal(node):
            return 1
        complement = self.complement(node.level, node.spaces[-1])
        return self._relative_subtree_size(
            complement.rank,
            node.rows,
            self.target_rank(node.level),
            self.whole_tree_size(node.level + 1),
        )

    # -- canonical completion --------------------------------------------- #

    def canonical_terminal(self, node: _Node) -> _Node:
        """Complete a prefix deterministically (lowest eligible unit pivots)."""

        while not self.is_terminal(node):
            level = node.level
            complement = self.complement(level, node.spaces[-1])
            target_rank = self.target_rank(level)
            rows = node.rows
            last = _pivot(rows[-1]) if rows else -1
            for column in range(last + 1, complement.rank):
                if len(rows) == target_rank:
                    break
                if not any(row & (1 << column) for row in rows):
                    rows = rows + (1 << column,)
            if len(rows) != target_rank:
                raise AssertionError("canonical completion ran out of pivots")
            target = self.forced[level].sum(_lift_relative_rows(complement, rows))
            node = self.normalize(level + 1, node.spaces + (target,))
        return node


def canonical_representative(flag_spaces: tuple[GF2Subspace, ...]) -> GF2Matrix:
    """One deterministic layout with the given flag.

    Each owner level takes the canonical complement of ``T_{s+1}`` in ``T_s``
    with its RREF basis as ordered frame; the local bits take the RREF basis
    of ``T_h``.
    """

    columns: tuple[int, ...] = ()
    for outer, local in zip(flag_spaces, flag_spaces[1:]):
        columns += local.basis_extension(within=outer).basis
    columns += flag_spaces[-1].basis
    return _matrix_from_columns(columns, flag_spaces[0].dimension)


# --------------------------------------------------------------------------- #
# Flag completion and per-layout evaluation
# --------------------------------------------------------------------------- #


def _ordered_frames(space: GF2Subspace) -> tuple[tuple[int, ...], ...]:
    """Every ordered basis of ``space`` in lexicographic order."""

    if space.rank == 0:
        return ((),)
    vectors = space.span_vectors(max_results=1 << space.rank)[1:]
    output: list[tuple[int, ...]] = []

    def visit(frame: tuple[int, ...], span: GF2Subspace) -> None:
        if len(frame) == space.rank:
            output.append(frame)
            return
        for vector in vectors:
            if not span.contains(vector):
                visit(
                    frame + (vector,),
                    GF2Subspace(space.dimension, span.basis + (vector,)),
                )

    visit((), GF2Subspace.zero(space.dimension))
    return tuple(output)


def _flag_layouts(
    domain: _Domain, spaces: tuple[GF2Subspace, ...], frame_cache: dict
) -> Iterator[GF2Matrix]:
    """Every layout with flag ``spaces``: complements x frames x local frame."""

    def frames(space: GF2Subspace) -> tuple[tuple[int, ...], ...]:
        cached = frame_cache.get(space)
        if cached is None:
            cached = _ordered_frames(space)
            frame_cache[space] = cached
        return cached

    complement_choices = tuple(
        spaces[level + 1].canonical_complements(
            within=spaces[level],
            max_results=1
            << (spaces[level + 1].rank * (spaces[level].rank - spaces[level + 1].rank)),
        )
        for level in range(domain.level_count)
    )
    terminal_frames = frames(spaces[-1])
    for complements in product(*complement_choices):
        for owner_bases in product(*(frames(space) for space in complements)):
            owner_columns = tuple(vector for basis in owner_bases for vector in basis)
            for terminal_basis in terminal_frames:
                yield _matrix_from_columns(
                    owner_columns + terminal_basis, domain.carrier
                )


def _full_carrier_terms(
    domain: _Domain, spaces: tuple[GF2Subspace, ...]
) -> tuple[MovementTerm, ...]:
    terms = []
    for access, kernel in zip(domain.accesses, domain.kernels):
        for level in range(domain.level_count):
            incidence = full_carrier_rank_incidence(kernel, spaces[level + 1])
            cost = (
                Fraction(0)
                if access.is_strict_reduction
                else scale_extra_owner_incidence(
                    incidence.extra_owner_incidence,
                    element_bytes=access.element_bytes,
                    cost_per_byte=access.cost_per_level[level],
                )
            )
            terms.append(
                MovementTerm(
                    access.name,
                    level,
                    incidence.movement_rank,
                    incidence.extra_owner_incidence,
                    cost,
                )
            )
    return tuple(terms)


class _MaskedEvaluator:
    """Exact active-owner cost; element of every active point precomputed."""

    def __init__(self, domain: _Domain):
        self.domain = domain
        self.points = domain.problem.active_points
        self.elements = tuple(
            tuple(access.access_matrix.apply(point) for point in self.points)
            for access in domain.accesses
        )
        self.owner_masks = tuple(
            (1 << end) - 1 for _, end in domain.hierarchy.owner_bit_ranges
        )

    def evaluate(
        self, work_to_iteration: GF2Matrix
    ) -> tuple[tuple[MovementTerm, ...], Fraction | None]:
        domain = self.domain
        inverse = work_to_iteration.inverse()
        work_points = tuple(inverse.apply(point) for point in self.points)
        terms = []
        legal = True
        for access, elements in zip(domain.accesses, self.elements):
            for level, mask in enumerate(self.owner_masks):
                owners: dict[int, set[int]] = {}
                for element, work in zip(elements, work_points):
                    owners.setdefault(element, set()).add(work & mask)
                extra = sum(len(item) - 1 for item in owners.values())
                if access.is_strict_reduction:
                    legal = legal and extra == 0
                    cost = Fraction(0)
                else:
                    cost = scale_extra_owner_incidence(
                        extra,
                        element_bytes=access.element_bytes,
                        cost_per_byte=access.cost_per_level[level],
                    )
                terms.append(MovementTerm(access.name, level, None, extra, cost))
        total = sum((term.cost for term in terms), Fraction()) if legal else None
        return tuple(terms), total


def _iter_flags(domain: _Domain) -> Iterator[_Node]:
    """Every terminal prefix (complete flag), in canonical DFS order."""

    stack = [domain.root()]
    while stack:
        node = stack.pop()
        if domain.is_terminal(node):
            yield node
            continue
        stack.extend(reversed(domain.children(node)))


def _check_bounded_domain(problem: LayoutSearchProblem, policy: LayoutSearchPolicy):
    bits = problem.carrier_bits
    levels = len(problem.hierarchy.levels)
    if bits > policy.max_carrier_bits or levels > policy.max_levels:
        raise LayoutSearchDomainError(
            f"the bounded search covers at most {policy.max_carrier_bits} "
            f"carrier bits and {policy.max_levels} owner levels; this problem "
            f"has {bits} bits and {levels} levels"
        )


def enumerate_layouts(
    problem: LayoutSearchProblem,
    *,
    max_carrier_bits: int = MAX_BOUNDED_CARRIER_BITS,
) -> Iterator[EvaluatedLayout]:
    """Exhaustively yield every layout of the bounded domain with its cost.

    Layouts are grouped by flag in canonical order; the forced-local
    constraint (and, on a full carrier, strict-reduction kernels) is applied.
    In a masked problem strict reductions are checked per layout and
    illegal layouts are yielded with ``cost=None``.
    """

    if problem.carrier_bits > min(max_carrier_bits, MAX_BOUNDED_CARRIER_BITS):
        raise LayoutSearchDomainError(
            f"exhaustive enumeration is implemented for at most "
            f"{MAX_BOUNDED_CARRIER_BITS} carrier bits"
        )
    domain = _Domain(problem, strict_as_forced=not problem.is_masked)
    masked = _MaskedEvaluator(domain) if problem.is_masked else None
    frame_cache: dict = {}
    for node in _iter_flags(domain):
        flag = domain.flag(node)
        if masked is None:
            full_terms = _full_carrier_terms(domain, node.spaces)
            full_cost = sum((term.cost for term in full_terms), Fraction())
            for matrix in _flag_layouts(domain, node.spaces, frame_cache):
                yield EvaluatedLayout(matrix, flag, full_terms, full_cost)
        else:
            for matrix in _flag_layouts(domain, node.spaces, frame_cache):
                terms, cost = masked.evaluate(matrix)
                yield EvaluatedLayout(matrix, flag, terms, cost)


# --------------------------------------------------------------------------- #
# Bounded full-carrier search (Algorithm 1)
# --------------------------------------------------------------------------- #


def _sorted_ties(ties: list[tuple[_Node, GF2Matrix]]):
    ordered = sorted(ties, key=lambda item: (item[0].key, item[1].rows))
    matrices = tuple(matrix for _, matrix in ordered)
    flags = []
    seen = set()
    for node, _ in ordered:
        if node.key not in seen:
            seen.add(node.key)
            flags.append(node)
    return matrices, flags


def _search_bounded(
    problem: LayoutSearchProblem, policy: LayoutSearchPolicy
) -> LayoutSearchResult:
    domain = _Domain(problem, strict_as_forced=True)
    budget = policy.max_search_nodes
    per_flag = domain.layouts_per_flag
    expected_flags = domain.deeper_flag_count(0)
    expected_layouts = expected_flags * per_flag
    expected_prefix_nodes = domain.whole_tree_size(0)

    evaluated = 0
    visited_layouts = 0
    expanded = 0
    materialized = 0
    best: Fraction | None = None
    ties: list[tuple[_Node, GF2Matrix]] = []
    pruned: list[tuple[_Node, _Bound]] = []
    unresolved: list[tuple[_Node, _Bound]] = []
    frame_cache: dict = {}

    root = domain.root()
    root_bound = domain.bound(root)
    heap: list = []
    if budget is not None and budget < 2:
        unresolved.append((root, root_bound))
    else:
        evaluated += 1
        heappush(heap, (root_bound.lower_cost, root.key, root, root_bound))

    while heap:
        _, _, node, bound = heappop(heap)
        if policy.enable_pruning and best is not None and bound.lower_cost > best:
            pruned.append((node, bound))
            continue
        used = 1 + evaluated + visited_layouts
        if domain.is_terminal(node):
            if budget is not None and used + per_flag > budget:
                unresolved.append((node, bound))
                unresolved.extend((item[2], item[3]) for item in heap)
                heap.clear()
                break
            materialized += 1
            visited_layouts += per_flag
            cost = bound.lower_cost
            if best is None or cost < best:
                best = cost
                ties = []
            if cost == best:
                ties.extend(
                    (node, matrix)
                    for matrix in _flag_layouts(domain, node.spaces, frame_cache)
                )
            continue
        children = domain.children(node)
        if budget is not None and used + len(children) > budget:
            unresolved.append((node, bound))
            unresolved.extend((item[2], item[3]) for item in heap)
            heap.clear()
            break
        expanded += 1
        for child in children:
            child_bound = domain.bound(child)
            evaluated += 1
            heappush(heap, (child_bound.lower_cost, child.key, child, child_bound))

    fallback = False
    if unresolved and not ties:
        fallback = True
        start = min(unresolved, key=lambda item: (item[1].lower_cost, item[0].key))[0]
        terminal = domain.canonical_terminal(start)
        best = domain.bound(terminal).lower_cost
        ties = [(terminal, canonical_representative(terminal.spaces))]

    pruned_layouts = sum(domain.node_flag_count(node) for node, _ in pruned) * per_flag
    unresolved_layouts = (
        sum(domain.node_flag_count(node) for node, _ in unresolved) * per_flag
    )
    if visited_layouts + pruned_layouts + unresolved_layouts != expected_layouts:
        raise AssertionError("search telemetry does not partition the layout domain")
    pruned_descendants = sum(domain.node_subtree_size(node) - 1 for node, _ in pruned)
    telemetry = LayoutSearchTelemetry(
        expected_layouts=expected_layouts,
        expected_flags=expected_flags,
        layouts_per_flag=per_flag,
        expected_prefix_nodes=expected_prefix_nodes,
        evaluated_prefix_nodes=evaluated,
        expanded_prefix_nodes=expanded,
        visited_flags=materialized,
        pruned_frontier_roots=len(pruned),
        unresolved_frontier_roots=len(unresolved),
        pruned_prefix_descendants=pruned_descendants,
        visited_layouts=visited_layouts,
        pruned_layouts=pruned_layouts,
        unresolved_layouts=unresolved_layouts,
        illegal_layouts=0,
        search_nodes_used=1 + evaluated + visited_layouts,
        fallback_incumbent=fallback,
    )
    if unresolved:
        termination = LayoutSearchTermination.NODE_BUDGET
    elif pruned:
        termination = LayoutSearchTermination.BOUND_PROVED
    else:
        termination = LayoutSearchTermination.EXHAUSTED
    matrices, flag_nodes = _sorted_ties(ties)
    return LayoutSearchResult(
        ties=matrices,
        tie_flags=tuple(domain.flag(node) for node in flag_nodes),
        cost=best,
        termination=termination,
        visited=visited_layouts,
        pruned=pruned_layouts,
        remaining_lower_bounds=tuple(
            sorted(bound.lower_cost for _, bound in unresolved)
        ),
        optimality_proven=termination is not LayoutSearchTermination.NODE_BUDGET,
        implementation=LayoutSearchImplementation.BOUNDED,
        ties_complete=True,
        tie_multiplicity=1,
        telemetry=telemetry,
        problem=problem,
    )


# --------------------------------------------------------------------------- #
# Masked search (paper Section VI): exact counts, no rank bound
# --------------------------------------------------------------------------- #


def _search_bounded_masked(
    problem: LayoutSearchProblem, policy: LayoutSearchPolicy
) -> LayoutSearchResult:
    domain = _Domain(problem, strict_as_forced=False)
    evaluator = _MaskedEvaluator(domain)
    budget = policy.max_search_nodes
    per_flag = domain.layouts_per_flag
    flags = tuple(_iter_flags(domain))
    expected_layouts = len(flags) * per_flag

    best: Fraction | None = None
    ties: list[tuple[_Node, GF2Matrix]] = []
    visited_flags = 0
    visited_layouts = 0
    illegal = 0
    frame_cache: dict = {}
    stopped_at = None
    for index, node in enumerate(flags):
        used = 1 + visited_flags + visited_layouts
        if budget is not None and used + 1 + per_flag > budget:
            stopped_at = index
            break
        visited_flags += 1
        for matrix in _flag_layouts(domain, node.spaces, frame_cache):
            visited_layouts += 1
            _, cost = evaluator.evaluate(matrix)
            if cost is None:
                illegal += 1
                continue
            if best is None or cost < best:
                best = cost
                ties = []
            if cost == best:
                ties.append((node, matrix))

    remaining = 0 if stopped_at is None else len(flags) - stopped_at
    fallback = False
    if stopped_at is not None and not ties:
        for node in flags[stopped_at:]:
            for matrix in _flag_layouts(domain, node.spaces, frame_cache):
                _, cost = evaluator.evaluate(matrix)
                if cost is not None:
                    best, ties, fallback = cost, [(node, matrix)], True
                    break
            if fallback:
                break
    if stopped_at is None and best is None:
        raise NoLegalLayoutError(
            "every layout of the masked domain splits a strict reduction"
        )
    telemetry = LayoutSearchTelemetry(
        expected_layouts=expected_layouts,
        expected_flags=len(flags),
        layouts_per_flag=per_flag,
        expected_prefix_nodes=None,
        evaluated_prefix_nodes=0,
        expanded_prefix_nodes=0,
        visited_flags=visited_flags,
        pruned_frontier_roots=0,
        unresolved_frontier_roots=remaining,
        pruned_prefix_descendants=None,
        visited_layouts=visited_layouts,
        pruned_layouts=0,
        unresolved_layouts=remaining * per_flag,
        illegal_layouts=illegal,
        search_nodes_used=1 + visited_flags + visited_layouts,
        fallback_incumbent=fallback,
    )
    termination = (
        LayoutSearchTermination.EXHAUSTED
        if stopped_at is None
        else LayoutSearchTermination.NODE_BUDGET
    )
    matrices, flag_nodes = _sorted_ties(ties)
    return LayoutSearchResult(
        ties=matrices,
        tie_flags=tuple(domain.flag(node) for node in flag_nodes),
        cost=best,
        termination=termination,
        visited=visited_layouts,
        pruned=0,
        # The rank bound is not valid under a mask; zero (costs are
        # non-negative) is the only bound claimed for an unvisited flag.
        remaining_lower_bounds=(Fraction(0),) * remaining,
        optimality_proven=stopped_at is None,
        implementation=LayoutSearchImplementation.BOUNDED_MASKED,
        ties_complete=True,
        tie_multiplicity=1,
        telemetry=telemetry,
        problem=problem,
    )


# --------------------------------------------------------------------------- #
# Scalable flag search (ported from the snapshot scalable_layout_synthesis)
# --------------------------------------------------------------------------- #
#
# The bounded search above materializes every layout of a minimum flag,
# which limits it to four carrier bits.  This search uses
# the same prefix tree and the same literal paper bound but stops at flags:
#
# * children of a prefix are generated lazily, so a node budget stops an
#   expansion without enumerating its (possibly exponential) child set;
# * a terminal prefix is a complete flag; its exact cost is the bound, and it
#   is reported as one tie standing for ``layouts_per_flag`` layouts that all
#   have the same movement (they differ only in complements and basis
#   frames);
# * :func:`complete_scalable_layout_representative` completes a tied flag to
#   one deterministic layout.
#
# Guarantee.  Without a budget, or when the budget is not reached, the search
# is exact: every minimum flag is returned (``EXHAUSTED`` or ``BOUND_PROVED``).
# With a budget the search may stop with ``NODE_BUDGET``: the result then holds
# the best flags seen (or, if none was reached, one canonical completion of
# the best frontier prefix, marked ``fallback_incumbent``) and the lower
# bounds of every unexplored frontier prefix; it makes no optimality claim.
# The budget counts evaluated prefixes, including the root.  Only full
# carriers are supported; masked domains need the bounded masked search.
#
# Ported from the snapshot ``scalable_layout_synthesis`` (SLS2--SLS5): the
# relative-RREF cursor, weighted bound, flag-level best-first loop and
# representative completion.  Dropped: SHA-256 fingerprints and replay
# authentication, the target-constraint callback protocol (SLS6),
# ``coordinate_flags_only`` and the sub-byte ``element_bits`` field (use a
# ``Fraction`` ``element_bytes`` instead).

MAX_SCALABLE_CARRIER_BITS = 32


@dataclass(frozen=True)
class ScalableFlagTie:
    """One minimum flag; ``multiplicity`` layouts share its movement."""

    flag: LocalSpaceFlag
    movement_rank_vector: tuple[int, ...]
    multiplicity: int
    cost: Fraction

    @property
    def structural_key(self) -> tuple[tuple[int, ...], ...]:
        return tuple(space.basis for space in self.flag.spaces[1:])


@dataclass(frozen=True)
class ScalableFlagSearchResult:
    # pylint: disable=too-many-instance-attributes
    """Flag-level search outcome with exact flag accounting.

    ``visited_flags + pruned_flags + unresolved_flags == total_flags``.
    ``movement_rank_vector`` entries are ordered access-major, level-minor.
    """

    termination: LayoutSearchTermination
    best_cost: Fraction | None
    ties: tuple[ScalableFlagTie, ...]
    evaluated_prefixes: int
    expanded_prefixes: int
    visited_flags: int
    pruned_flags: int
    unresolved_flags: int
    total_flags: int
    layouts_per_flag: int
    pruned_frontier_bounds: tuple[Fraction, ...]
    unresolved_frontier_bounds: tuple[Fraction, ...]
    fallback_incumbent: bool
    problem: LayoutSearchProblem = field(repr=False)

    @property
    def optimality_proven(self) -> bool:
        return self.termination is not LayoutSearchTermination.NODE_BUDGET


@dataclass(frozen=True)
class ScalableLayoutRepresentative:
    """One deterministic layout of a tied flag."""

    tie: ScalableFlagTie
    work_to_iteration: GF2Matrix
    iteration_to_work: GF2Matrix
    multiplicity: int


def _validated_domain(problem: LayoutSearchProblem, max_carrier_bits: int) -> _Domain:
    if not isinstance(problem, LayoutSearchProblem):
        raise TypeError("problem must be a LayoutSearchProblem")
    if problem.is_masked:
        raise LayoutSearchDomainError(
            "the scalable search requires a full carrier; masked domains use "
            "the bounded masked search"
        )
    if (
        isinstance(max_carrier_bits, bool)
        or not isinstance(max_carrier_bits, int)
        or not 1 <= max_carrier_bits <= MAX_SCALABLE_CARRIER_BITS
    ):
        raise ValueError(
            f"max_carrier_bits must be in [1, {MAX_SCALABLE_CARRIER_BITS}]"
        )
    if problem.carrier_bits > max_carrier_bits:
        raise LayoutSearchDomainError(
            f"the scalable search covers at most {max_carrier_bits} carrier "
            f"bits; this problem has {problem.carrier_bits}"
        )
    return _Domain(problem, strict_as_forced=True)


def search_scalable_layout_flags(
    problem: LayoutSearchProblem,
    *,
    max_search_nodes: int | None = None,
    max_carrier_bits: int = MAX_SCALABLE_CARRIER_BITS,
) -> ScalableFlagSearchResult:
    """Best-first search over flags with the paper bound and a node budget.

    ``max_search_nodes=None`` runs to exhaustion (exact).  Pruning drops a
    prefix only when its bound strictly exceeds the best flag cost, so every
    minimum flag is kept.  An expansion is atomic: if all children of a
    prefix do not fit in the remaining budget the search stops with that
    prefix and the rest of the heap as the unresolved frontier.
    """

    if max_search_nodes is not None and (
        isinstance(max_search_nodes, bool)
        or not isinstance(max_search_nodes, int)
        or max_search_nodes < 1
    ):
        raise ValueError("max_search_nodes must be a positive integer or None")
    domain = _validated_domain(problem, max_carrier_bits)
    per_flag = domain.layouts_per_flag

    root = domain.root()
    root_bound = domain.bound(root)
    evaluated = {root.key: (root, root_bound)}
    heap = [(root_bound.lower_cost, root.key)]
    pruned = []
    unresolved = []
    expanded = 0
    terminals = 0
    best: Fraction | None = None
    ties: list[ScalableFlagTie] = []

    while heap:
        _, key = heappop(heap)
        node, bound = evaluated[key]
        if best is not None and bound.lower_cost > best:
            pruned.append((node, bound))
            continue
        if domain.is_terminal(node):
            terminals += 1
            tie = ScalableFlagTie(
                domain.flag(node), bound.lower_ranks, per_flag, bound.lower_cost
            )
            if best is None or tie.cost < best:
                best = tie.cost
                ties = [tie]
            elif tie.cost == best:
                ties.append(tie)
            continue
        remaining = (
            None if max_search_nodes is None else max_search_nodes - len(evaluated)
        )
        children = []
        over_budget = False
        for child in domain.iter_children(node):
            children.append(child)
            if remaining is not None and len(children) > remaining:
                over_budget = True
                break
        if over_budget:
            unresolved.append((node, bound))
            unresolved.extend(evaluated[item[1]] for item in heap)
            heap.clear()
            break
        expanded += 1
        for child in children:
            if child.key in evaluated:
                raise AssertionError("prefix generated twice")
            child_bound = domain.bound(child)
            evaluated[child.key] = (child, child_bound)
            heappush(heap, (child_bound.lower_cost, child.key))

    fallback = False
    if unresolved and not ties:
        start = min(unresolved, key=lambda item: (item[1].lower_cost, item[0].key))[0]
        terminal = domain.canonical_terminal(start)
        terminal_bound = domain.bound(terminal)
        best = terminal_bound.lower_cost
        ties = [
            ScalableFlagTie(
                domain.flag(terminal), terminal_bound.lower_ranks, per_flag, best
            )
        ]
        fallback = True

    total_flags = domain.deeper_flag_count(0)
    pruned_flags = sum(domain.node_flag_count(node) for node, _ in pruned)
    unresolved_flags = sum(domain.node_flag_count(node) for node, _ in unresolved)
    visited_flags = total_flags - pruned_flags - unresolved_flags
    if visited_flags != terminals:
        raise AssertionError("flag accounting does not partition the flag domain")
    if unresolved:
        termination = LayoutSearchTermination.NODE_BUDGET
    elif pruned:
        termination = LayoutSearchTermination.BOUND_PROVED
    else:
        termination = LayoutSearchTermination.EXHAUSTED
    return ScalableFlagSearchResult(
        termination=termination,
        best_cost=best,
        ties=tuple(
            sorted(
                ties, key=lambda item: (item.movement_rank_vector, item.structural_key)
            )
        ),
        evaluated_prefixes=len(evaluated),
        expanded_prefixes=expanded,
        visited_flags=visited_flags,
        pruned_flags=pruned_flags,
        unresolved_flags=unresolved_flags,
        total_flags=total_flags,
        layouts_per_flag=per_flag,
        pruned_frontier_bounds=tuple(sorted(bound.lower_cost for _, bound in pruned)),
        unresolved_frontier_bounds=tuple(
            sorted(bound.lower_cost for _, bound in unresolved)
        ),
        fallback_incumbent=fallback,
        problem=problem,
    )


def complete_scalable_layout_representative(
    result: ScalableFlagSearchResult, tie_ordinal: int = 0
) -> ScalableLayoutRepresentative:
    """Complete tie ``tie_ordinal`` to its canonical layout.

    The representative takes, at each owner level, the canonical complement
    of ``T_{s+1}`` in ``T_s`` with its RREF basis, and the RREF basis of
    ``T_h`` for the local bits.  Its movement equals the tie's cost.  A tie
    of a ``NODE_BUDGET`` result is a feasible incumbent, not a proven
    minimum.
    """

    if not isinstance(result, ScalableFlagSearchResult):
        raise TypeError("result must be a ScalableFlagSearchResult")
    if (
        isinstance(tie_ordinal, bool)
        or not isinstance(tie_ordinal, int)
        or not 0 <= tie_ordinal < len(result.ties)
    ):
        raise IndexError(f"tie ordinal {tie_ordinal!r} is out of range")
    tie = result.ties[tie_ordinal]
    work_to_iteration = canonical_representative(tie.flag.spaces)
    hierarchy = result.problem.hierarchy
    images = tuple(
        work_to_iteration.image(space) for space in hierarchy.coordinate_flag().spaces
    )
    if images != tie.flag.spaces:
        raise AssertionError("representative does not reproduce its flag")
    return ScalableLayoutRepresentative(
        tie, work_to_iteration, work_to_iteration.inverse(), tie.multiplicity
    )


def search_scalable_min_movement_layout(
    problem: LayoutSearchProblem, policy: LayoutSearchPolicy | None = None
) -> LayoutSearchResult:
    """Run the scalable flag search and report it as a LayoutSearchResult.

    ``ties`` holds one canonical representative per tied flag
    (``ties_complete=False``, ``tie_multiplicity=layouts_per_flag``).  The
    policy's ``max_search_nodes`` is the prefix budget; ``enable_pruning``
    is ignored (the scalable search always prunes strictly).
    """

    if policy is None:
        policy = LayoutSearchPolicy()
    flags = search_scalable_layout_flags(
        problem, max_search_nodes=policy.max_search_nodes
    )
    representatives = tuple(
        complete_scalable_layout_representative(flags, ordinal)
        for ordinal in range(len(flags.ties))
    )
    per_flag = flags.layouts_per_flag
    telemetry = LayoutSearchTelemetry(
        expected_layouts=flags.total_flags * per_flag,
        expected_flags=flags.total_flags,
        layouts_per_flag=per_flag,
        expected_prefix_nodes=None,
        evaluated_prefix_nodes=flags.evaluated_prefixes,
        expanded_prefix_nodes=flags.expanded_prefixes,
        visited_flags=flags.visited_flags,
        pruned_frontier_roots=len(flags.pruned_frontier_bounds),
        unresolved_frontier_roots=len(flags.unresolved_frontier_bounds),
        pruned_prefix_descendants=None,
        visited_layouts=flags.visited_flags * per_flag,
        pruned_layouts=flags.pruned_flags * per_flag,
        unresolved_layouts=flags.unresolved_flags * per_flag,
        illegal_layouts=0,
        search_nodes_used=flags.evaluated_prefixes,
        fallback_incumbent=flags.fallback_incumbent,
    )
    return LayoutSearchResult(
        ties=tuple(item.work_to_iteration for item in representatives),
        tie_flags=tuple(item.tie.flag for item in representatives),
        cost=flags.best_cost,
        termination=flags.termination,
        visited=telemetry.visited_layouts,
        pruned=telemetry.pruned_layouts,
        remaining_lower_bounds=flags.unresolved_frontier_bounds,
        optimality_proven=flags.optimality_proven,
        implementation=LayoutSearchImplementation.SCALABLE,
        ties_complete=False,
        tie_multiplicity=per_flag,
        telemetry=telemetry,
        problem=problem,
    )


# --------------------------------------------------------------------------- #
# Public entry
# --------------------------------------------------------------------------- #


def search_min_movement_layout(
    problem: LayoutSearchProblem,
    policy: LayoutSearchPolicy | None = None,
) -> LayoutSearchResult:
    """Find every minimum-movement layout of ``problem``.

    Dispatch: a masked problem uses the masked bounded search (it must fit
    the bounded domain); a full-carrier problem within ``max_carrier_bits``
    and ``max_levels`` uses the bounded Algorithm 1; a larger full-carrier
    problem uses the scalable flag search when ``policy.allow_scalable``.
    Raises :class:`LayoutSearchDomainError` otherwise and
    :class:`NoLegalLayoutError` when the constraints admit no layout.
    """

    if policy is None:
        policy = LayoutSearchPolicy()
    if not isinstance(problem, LayoutSearchProblem):
        raise TypeError("problem must be a LayoutSearchProblem")
    if not isinstance(policy, LayoutSearchPolicy):
        raise TypeError("policy must be a LayoutSearchPolicy")
    if problem.is_masked:
        _check_bounded_domain(problem, policy)
        return _search_bounded_masked(problem, policy)
    bits = problem.carrier_bits
    levels = len(problem.hierarchy.levels)
    if bits <= policy.max_carrier_bits and levels <= policy.max_levels:
        return _search_bounded(problem, policy)
    if not policy.allow_scalable:
        _check_bounded_domain(problem, policy)
    return search_scalable_min_movement_layout(problem, policy)


__all__ = [
    "DEFAULT_MAX_SEARCH_NODES",
    "EvaluatedLayout",
    "LayoutSearchDomainError",
    "LayoutSearchImplementation",
    "LayoutSearchPolicy",
    "LayoutSearchProblem",
    "LayoutSearchResult",
    "LayoutSearchTelemetry",
    "LayoutSearchTermination",
    "MAX_BOUNDED_CARRIER_BITS",
    "MAX_BOUNDED_LEVELS",
    "MAX_SCALABLE_CARRIER_BITS",
    "NoLegalLayoutError",
    "ScalableFlagSearchResult",
    "ScalableFlagTie",
    "ScalableLayoutRepresentative",
    "complete_scalable_layout_representative",
    "search_scalable_layout_flags",
    "search_scalable_min_movement_layout",
    "canonical_representative",
    "enumerate_layouts",
    "gaussian_binomial",
    "gl_count",
    "layouts_per_flag",
    "search_min_movement_layout",
]
