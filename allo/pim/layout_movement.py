# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owner-incidence movement algebra for minimum-movement layouts (paper IV-B).

Notation follows Section IV of the paper.  ``I`` is the GF(2) space of
iteration-index bits.  A layout ``L`` is an invertible map from work
coordinates (owner bits of levels ``1..h`` followed by local storage bits)
to iteration bits; this package represents it as ``work_to_iteration``.
``Q_s`` keeps the owner bits through level ``s`` of ``L^-1`` and
``T_s = ker Q_s`` is the local space of level ``s``.  For an access ``j``
with access matrix ``A_j`` (iteration bits to element bits), the iterations
that touch one element differ by vectors of ``K_j = ker A_j``.

Rank formula (Eq. pim-movement-rank)::

    r_{j,s} = rank(Q_s restricted to K_j) = dim K_j - dim(K_j intersect T_s)

``2**r_{j,s}`` units of level ``s`` share every element of ``j``.

Movement cost (Eq. pim-movement-cost)::

    C_move(L) = sum_j sum_s c_{j,s} * B_j * (2**r_{j,s} - 1)

where ``B_j = element_bytes * 2**(n - dim K_j)`` is the number of bytes of
tensor ``j`` touched by the iteration space.

Unit convention.  ``extra_owner_incidence`` is an element count: for a READ
access it is the number of extra copies (``2**r - 1`` per element); for a
WRITE_ACCUMULATE access it is the number of extra partial results that must
be combined (``2**r - 1`` per element).  It is multiplied by
``element_bytes`` to obtain bytes and by ``cost_per_level[s]`` (the target's
cost per byte of moving data across level ``s``, in whatever unit the cost
program uses, for example cycles per byte) to obtain cost.  Costs are exact
``Fraction`` values so ties are exact.

Reduction order.  A WRITE_ACCUMULATE access with
``reassociation_allowed=False`` keeps strict source order: no owner bit may
split its reduction kernel, so a layout is legal only if ``K_j`` is inside
the innermost local space ``T_h`` (equivalently ``r_{j,s} = 0`` for every
level).  Such an access contributes no movement term and instead constrains
the domain before search (see :func:`strict_reduction_kernel`).

Masked domains (paper Section VI).  When only an active subset ``D`` of the
iteration space is valid, the rank formula is not a valid cost or bound.
:func:`exact_active_owner_incidence` counts
``sum_e (|O_{j,s}(e)| - 1)`` with ``O_{j,s}(e) = {Q_s(i) : i in D, A_j i = e}``
directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from fractions import Fraction
from numbers import Rational

from .gf2 import GF2Matrix, GF2Subspace
from .layout_problem import OwnerHierarchy


class AccessMode(str, Enum):
    """How one access contributes owner incidence."""

    READ = "read"
    WRITE_ACCUMULATE = "write-accumulate"


def _exact_nonnegative(value: object, *, name: str) -> Fraction:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a number, got bool")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        value = Fraction(value)
    if not isinstance(value, Rational):
        raise TypeError(f"{name} must be an int, Fraction or float")
    result = Fraction(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative, got {result}")
    return result


@dataclass(frozen=True)
class LayoutAccess:
    """One tensor access of the layout problem.

    ``access_matrix`` maps iteration bits (columns) to element bits (rows).
    ``element_bytes`` is the element size; it may be a ``Fraction`` for
    sub-byte types (an i11 element is ``Fraction(11, 8)``).
    ``cost_per_level[s]`` is the target cost per byte of moving data across
    owner level ``s`` (outermost first); it has one entry per owner level.
    ``reassociation_allowed`` matters only for WRITE_ACCUMULATE: when it is
    ``False`` partial results may not be regrouped, so the reduction may not
    be split across owners.
    """

    name: str
    access_matrix: GF2Matrix
    mode: AccessMode
    element_bytes: Fraction
    cost_per_level: tuple[Fraction, ...]
    reassociation_allowed: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("access name must be a non-empty string")
        if not isinstance(self.access_matrix, GF2Matrix):
            raise TypeError("access_matrix must be a GF2Matrix")
        if not isinstance(self.mode, AccessMode):
            raise TypeError("mode must be an AccessMode")
        element_bytes = _exact_nonnegative(self.element_bytes, name="element_bytes")
        if element_bytes == 0:
            raise ValueError("element_bytes must be positive")
        if isinstance(self.cost_per_level, (str, bytes)):
            raise TypeError("cost_per_level must be a sequence of numbers")
        costs = tuple(
            _exact_nonnegative(cost, name=f"cost_per_level[{level}]")
            for level, cost in enumerate(self.cost_per_level)
        )
        if not isinstance(self.reassociation_allowed, bool):
            raise TypeError("reassociation_allowed must be a bool")
        object.__setattr__(self, "element_bytes", element_bytes)
        object.__setattr__(self, "cost_per_level", costs)

    @property
    def kernel(self) -> GF2Subspace:
        """``K_j = ker A_j``: iteration directions that keep the element."""

        return self.access_matrix.nullspace()

    @property
    def is_strict_reduction(self) -> bool:
        """A WRITE_ACCUMULATE access whose reduction may not be split."""

        return (
            self.mode is AccessMode.WRITE_ACCUMULATE and not self.reassociation_allowed
        )


@dataclass(frozen=True)
class FullCarrierRankIncidence:
    """Rank-formula owner multiplicity of one access at one level."""

    ambient_dimension: int
    kernel_dimension: int
    intersection_dimension: int
    movement_rank: int
    unique_operand_points: int
    extra_owner_multiplicity: int
    extra_owner_incidence: int


@dataclass(frozen=True)
class ActiveOwnerIncidence:
    """Exact active operand fibers and their owner-prefix images.

    ``owners_by_operand_point`` lists ``(element, owners)`` with strictly
    increasing elements and owners; ``extra_owner_incidence`` is
    ``sum(len(owners) - 1)``.
    """

    owners_by_operand_point: tuple[tuple[int, tuple[int, ...]], ...]
    active_iteration_point_count: int
    active_operand_point_count: int
    extra_owner_incidence: int


@dataclass(frozen=True)
class MovementTerm:
    """One ``(access, level)`` term of the movement cost.

    ``movement_rank`` is ``None`` for an exact masked count, where no single
    rank describes the term.
    """

    access: str
    owner_level: int
    movement_rank: int | None
    extra_owner_incidence: int
    cost: Fraction


@dataclass(frozen=True)
class LayoutMovementCost:
    """Movement cost of one layout; ``legal`` is False when a strict
    reduction is split, in which case ``total`` is ``None``."""

    terms: tuple[MovementTerm, ...]
    total: Fraction | None
    legal: bool


def movement_rank(kernel: GF2Subspace, local_space: GF2Subspace) -> int:
    """``r = dim K - dim(K intersect T)``."""

    if kernel.dimension != local_space.dimension:
        raise ValueError("kernel and local space have different ambient dimensions")
    return kernel.rank - kernel.intersection(local_space).rank


def full_carrier_rank_incidence(
    kernel: GF2Subspace,
    local_space: GF2Subspace,
) -> FullCarrierRankIncidence:
    """Return the paper's uniform full-carrier owner multiplicity.

    Valid only when every iteration point is active.  The number of distinct
    elements is ``2**(n - dim K)``; each is shared by ``2**r`` owners, so the
    extra incidence is ``2**(n - dim K) * (2**r - 1)``.
    """

    if not isinstance(kernel, GF2Subspace) or not isinstance(local_space, GF2Subspace):
        raise TypeError("kernel and local_space must be GF2Subspace values")
    if kernel.dimension != local_space.dimension:
        raise ValueError("kernel and local space have different ambient dimensions")
    intersection_dimension = kernel.intersection(local_space).rank
    rank = kernel.rank - intersection_dimension
    unique_operand_points = 1 << (kernel.dimension - kernel.rank)
    multiplicity = (1 << rank) - 1
    return FullCarrierRankIncidence(
        kernel.dimension,
        kernel.rank,
        intersection_dimension,
        rank,
        unique_operand_points,
        multiplicity,
        unique_operand_points * multiplicity,
    )


def _validated_active_points(points: object, carrier_bits: int) -> tuple[int, ...]:
    if isinstance(points, (str, bytes)):
        raise TypeError("active points must be a sequence of integers")
    points = tuple(points)
    extent = 1 << carrier_bits
    if (
        not points
        or any(
            isinstance(point, bool)
            or not isinstance(point, int)
            or not 0 <= point < extent
            for point in points
        )
        or points != tuple(sorted(set(points)))
    ):
        raise ValueError(
            "active points must be a non-empty strictly increasing subset of "
            f"range({extent})"
        )
    return points


def exact_active_owner_incidence(
    access_matrix: GF2Matrix,
    active_iteration_points: tuple[int, ...],
    work_to_iteration: GF2Matrix,
    hierarchy: OwnerHierarchy,
    owner_level: int,
) -> ActiveOwnerIncidence:
    """Count active owners of every accessed element exactly (paper VI).

    For each active iteration ``i`` the element is ``A i`` and the owner is
    the owner-bit prefix through ``owner_level`` of ``L^-1 i``.  The result
    is ``sum_e (|O(e)| - 1)``; three active owners of one element give two
    extra copies, whatever padding would suggest.
    """

    if not isinstance(access_matrix, GF2Matrix) or not isinstance(
        work_to_iteration, GF2Matrix
    ):
        raise TypeError("access_matrix and work_to_iteration must be GF2Matrix")
    if not isinstance(hierarchy, OwnerHierarchy):
        raise TypeError("hierarchy must be an OwnerHierarchy")
    carrier_bits = hierarchy.carrier_bit_count
    if access_matrix.n_cols != carrier_bits:
        raise ValueError("access matrix and owner hierarchy carriers differ")
    if work_to_iteration.n_rows != carrier_bits or work_to_iteration.n_cols != (
        carrier_bits
    ):
        raise ValueError("work layout must be square on the iteration carrier")
    if (
        isinstance(owner_level, bool)
        or not isinstance(owner_level, int)
        or not 0 <= owner_level < len(hierarchy.levels)
    ):
        raise ValueError(f"owner level {owner_level!r} is outside the hierarchy")
    points = _validated_active_points(active_iteration_points, carrier_bits)
    iteration_to_work = work_to_iteration.inverse()

    owner_mask = (1 << hierarchy.owner_bit_ranges[owner_level][1]) - 1
    owner_sets: dict[int, set[int]] = {}
    for point in points:
        element = access_matrix.apply(point)
        owner = iteration_to_work.apply(point) & owner_mask
        owner_sets.setdefault(element, set()).add(owner)
    incidence = tuple(
        (element, tuple(sorted(owners)))
        for element, owners in sorted(owner_sets.items())
    )
    return ActiveOwnerIncidence(
        incidence,
        len(points),
        len(incidence),
        sum(len(owners) - 1 for _, owners in incidence),
    )


def scale_extra_owner_incidence(
    extra_owner_incidence: int,
    *,
    element_bytes: Fraction | int,
    cost_per_byte: Fraction | int,
) -> Fraction:
    """``cost_per_byte * element_bytes * extra_owner_incidence``, exactly."""

    if (
        isinstance(extra_owner_incidence, bool)
        or not isinstance(extra_owner_incidence, int)
        or extra_owner_incidence < 0
    ):
        raise ValueError("extra_owner_incidence must be a non-negative integer")
    element_bytes = _exact_nonnegative(element_bytes, name="element_bytes")
    if element_bytes == 0:
        raise ValueError("element_bytes must be positive")
    cost_per_byte = _exact_nonnegative(cost_per_byte, name="cost_per_byte")
    return cost_per_byte * element_bytes * extra_owner_incidence


def strict_reduction_kernel(
    accesses: tuple[LayoutAccess, ...], carrier_bits: int
) -> GF2Subspace:
    """Span of the kernels of all strict (non-reassociable) reductions.

    A layout keeps every strict reduction unsplit exactly when this span is
    inside the innermost local space ``T_h``.  On a full carrier the search
    adds it to every forced-local space, so illegal layouts never enter the
    search domain.
    """

    basis: tuple[int, ...] = ()
    for access in accesses:
        if access.is_strict_reduction:
            basis += access.kernel.basis
    return GF2Subspace(carrier_bits, basis)


def _validate_layout(
    accesses: tuple[LayoutAccess, ...],
    hierarchy: OwnerHierarchy,
    work_to_iteration: GF2Matrix,
) -> None:
    carrier = hierarchy.carrier_bit_count
    if work_to_iteration.n_rows != carrier or work_to_iteration.n_cols != carrier:
        raise ValueError("work layout must be square on the iteration carrier")
    for access in accesses:
        if access.access_matrix.n_cols != carrier:
            raise ValueError(
                f"access {access.name!r} has {access.access_matrix.n_cols} "
                f"iteration bits; the hierarchy carrier has {carrier}"
            )
        if len(access.cost_per_level) != len(hierarchy.levels):
            raise ValueError(
                f"access {access.name!r} has {len(access.cost_per_level)} "
                f"per-level costs for {len(hierarchy.levels)} owner levels"
            )


def layout_movement_cost(
    accesses: tuple[LayoutAccess, ...],
    hierarchy: OwnerHierarchy,
    work_to_iteration: GF2Matrix,
    active_points: tuple[int, ...] | None = None,
) -> LayoutMovementCost:
    """Evaluate ``C_move`` for one layout.

    With ``active_points=None`` every iteration is active and the rank
    formula is used; otherwise the exact masked count is used.  A strict
    reduction that is split at any level makes the layout illegal.
    """

    accesses = tuple(accesses)
    _validate_layout(accesses, hierarchy, work_to_iteration)
    terms = []
    legal = True
    if active_points is None:
        local_spaces = tuple(
            work_to_iteration.image(hierarchy.local_coordinate_space_after(level))
            for level in range(len(hierarchy.levels))
        )
        for access in accesses:
            kernel = access.kernel
            for level, local_space in enumerate(local_spaces):
                rank_incidence = full_carrier_rank_incidence(kernel, local_space)
                if access.is_strict_reduction:
                    legal = legal and rank_incidence.movement_rank == 0
                    cost = Fraction(0)
                else:
                    cost = scale_extra_owner_incidence(
                        rank_incidence.extra_owner_incidence,
                        element_bytes=access.element_bytes,
                        cost_per_byte=access.cost_per_level[level],
                    )
                terms.append(
                    MovementTerm(
                        access.name,
                        level,
                        rank_incidence.movement_rank,
                        rank_incidence.extra_owner_incidence,
                        cost,
                    )
                )
    else:
        for access in accesses:
            for level in range(len(hierarchy.levels)):
                extra = exact_active_owner_incidence(
                    access.access_matrix,
                    active_points,
                    work_to_iteration,
                    hierarchy,
                    level,
                ).extra_owner_incidence
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
    return LayoutMovementCost(tuple(terms), total, legal)


__all__ = [
    "AccessMode",
    "ActiveOwnerIncidence",
    "FullCarrierRankIncidence",
    "LayoutAccess",
    "LayoutMovementCost",
    "MovementTerm",
    "exact_active_owner_incidence",
    "full_carrier_rank_incidence",
    "layout_movement_cost",
    "movement_rank",
    "scale_extra_owner_incidence",
    "strict_reduction_kernel",
]
