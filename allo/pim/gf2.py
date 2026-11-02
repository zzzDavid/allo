# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small, exact linear algebra over :math:`GF(2)`.

Vectors are non-negative Python integers: bit ``i`` is coordinate ``i`` in an
ambient vector space of dimension ``dimension``.  A :class:`GF2Subspace`
stores the unique reduced-row-echelon basis of its span.  The representation
is therefore immutable, hashable, and independent of input row order.

This module deliberately provides algebra rather than layout policy.  Its
enumerators are intended only for bounded exact layout synthesis and require
an explicit ``max_results`` from every caller.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations


class GF2MatrixShapeError(ValueError):
    """Raised when a GF(2) matrix operation has incompatible dimensions."""


class GF2SingularMatrixError(ValueError):
    """Raised when a GF(2) matrix has no inverse."""


def _require_nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def _require_positive_int(value: object, *, name: str) -> int:
    result = _require_nonnegative_int(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _validate_vector(vector: object, dimension: int) -> int:
    value = _require_nonnegative_int(vector, name="vector")
    if value.bit_length() > dimension:
        raise ValueError(
            f"vector {value} has a set bit outside ambient dimension {dimension}"
        )
    return value


def _canonical_rref(dimension: int, vectors: Iterable[int]) -> tuple[int, ...]:
    """Return the unique RREF row basis, pivoting from low to high bit."""

    try:
        iterator = iter(vectors)
    except TypeError as exc:
        raise TypeError("basis must be an iterable of integer vectors") from exc
    rows = [_validate_vector(vector, dimension) for vector in iterator]
    rows = [row for row in rows if row != 0]

    pivot_row = 0
    for column in range(dimension):
        pivot = next(
            (
                row_index
                for row_index in range(pivot_row, len(rows))
                if rows[row_index] & (1 << column)
            ),
            None,
        )
        if pivot is None:
            continue
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        pivot_value = rows[pivot_row]
        for row_index, row in enumerate(rows):
            if row_index != pivot_row and row & (1 << column):
                rows[row_index] = row ^ pivot_value
        pivot_row += 1
        if pivot_row == len(rows):
            break

    # Rows below pivot_row are zero after eliminating dependent inputs.
    return tuple(rows[:pivot_row])


def _bounded_power_of_two(exponent: int, max_results: int, *, operation: str) -> int:
    """Return ``2**exponent`` after proving it does not exceed the limit."""

    # Avoid constructing a huge integer merely to discover that enumeration
    # is forbidden.
    if exponent >= max_results.bit_length():
        raise ValueError(
            f"{operation} requires 2**{exponent} results, exceeding "
            f"max_results={max_results}"
        )
    count = 1 << exponent
    if count > max_results:
        raise ValueError(
            f"{operation} requires {count} results, exceeding "
            f"max_results={max_results}"
        )
    return count


def _bounded_subspace_count(dimension: int, rank: int | None, max_results: int) -> int:
    """Count requested subspaces, saturating once the caller's limit is passed."""

    # Gaussian binomial recurrence at q=2:
    #   G(n, k) = G(n-1, k) + 2**(n-k) G(n-1, k-1).
    previous = [1]
    cap = max_results + 1
    for n in range(1, dimension + 1):
        current = [1]
        for k in range(1, n):
            value = previous[k] + (1 << (n - k)) * previous[k - 1]
            current.append(min(value, cap))
        current.append(1)
        previous = current
    count = previous[rank] if rank is not None else min(sum(previous), cap)
    if count > max_results:
        rank_text = "all ranks" if rank is None else f"rank {rank}"
        raise ValueError(
            f"subspace enumeration for dimension {dimension}, {rank_text} "
            f"exceeds max_results={max_results}"
        )
    return count


@dataclass(frozen=True, slots=True)
class GF2Matrix:
    """An immutable ``n_rows`` by ``n_cols`` matrix over ``GF(2)``.

    Row ``i`` is an integer bitmask over the input coordinates.  Therefore
    :meth:`apply` maps an ``n_cols``-bit vector to an ``n_rows``-bit vector.
    Matrix multiplication follows function composition: ``a @ b`` applies
    ``b`` first and then ``a``.
    """

    n_rows: int
    n_cols: int
    rows: tuple[int, ...]

    def __post_init__(self) -> None:
        n_rows = _require_nonnegative_int(self.n_rows, name="n_rows")
        n_cols = _require_nonnegative_int(self.n_cols, name="n_cols")
        try:
            rows = tuple(self.rows)
        except TypeError as exc:
            raise TypeError("rows must be an iterable of integer vectors") from exc
        if len(rows) != n_rows:
            raise GF2MatrixShapeError(
                f"matrix declares {n_rows} rows but received {len(rows)}"
            )
        validated_rows = tuple(_validate_vector(row, n_cols) for row in rows)
        object.__setattr__(self, "n_rows", n_rows)
        object.__setattr__(self, "n_cols", n_cols)
        # Preserve an already canonical tuple so repeated validation remains
        # identity-idempotent.  Geometry caches deliberately retain the tuple
        # identity as a constant-time mutation snapshot.
        if type(self.rows) is not tuple or self.rows != validated_rows:
            object.__setattr__(self, "rows", validated_rows)

    @classmethod
    def zero(cls, n_rows: int, n_cols: int) -> GF2Matrix:
        """Return the all-zero matrix with the requested shape."""

        n_rows = _require_nonnegative_int(n_rows, name="n_rows")
        return cls(n_rows, n_cols, (0,) * n_rows)

    @classmethod
    def identity(cls, size: int) -> GF2Matrix:
        """Return the square identity matrix."""

        size = _require_nonnegative_int(size, name="size")
        return cls(size, size, tuple(1 << bit for bit in range(size)))

    @property
    def rank(self) -> int:
        """Matrix rank."""

        return GF2Subspace(self.n_cols, self.rows).rank

    def apply(self, vector: int) -> int:
        """Apply this linear map to one packed input vector."""

        source = _validate_vector(vector, self.n_cols)
        output = 0
        for row_index, row in enumerate(self.rows):
            if (row & source).bit_count() & 1:
                output |= 1 << row_index
        return output

    def transpose(self) -> GF2Matrix:
        """Return the matrix transpose."""

        columns_as_rows = tuple(
            sum(
                ((row >> column) & 1) << row_index
                for row_index, row in enumerate(self.rows)
            )
            for column in range(self.n_cols)
        )
        return GF2Matrix(self.n_cols, self.n_rows, columns_as_rows)

    def multiply(self, other: GF2Matrix) -> GF2Matrix:
        """Return ``self`` composed with ``other``."""

        if not isinstance(other, GF2Matrix):
            raise TypeError(f"expected GF2Matrix, got {type(other).__name__}")
        if self.n_cols != other.n_rows:
            raise GF2MatrixShapeError(
                "cannot multiply shapes "
                f"({self.n_rows}, {self.n_cols}) and "
                f"({other.n_rows}, {other.n_cols})"
            )
        product_rows = []
        for row in self.rows:
            product_row = 0
            for source_row, other_row in enumerate(other.rows):
                if row & (1 << source_row):
                    product_row ^= other_row
            product_rows.append(product_row)
        return GF2Matrix(self.n_rows, other.n_cols, tuple(product_rows))

    def __matmul__(self, other: GF2Matrix) -> GF2Matrix:
        return self.multiply(other)

    def inverse(self) -> GF2Matrix:
        """Return the inverse, rejecting non-square or singular matrices."""

        if self.n_rows != self.n_cols:
            raise GF2MatrixShapeError(
                f"inverse requires a square matrix, got "
                f"({self.n_rows}, {self.n_cols})"
            )
        size = self.n_rows
        augmented = [
            row | (1 << (size + row_index)) for row_index, row in enumerate(self.rows)
        ]
        for column in range(size):
            pivot = next(
                (
                    row_index
                    for row_index in range(column, size)
                    if augmented[row_index] & (1 << column)
                ),
                None,
            )
            if pivot is None:
                raise GF2SingularMatrixError(
                    f"matrix of size {size} is singular over GF(2)"
                )
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
            for row_index in range(size):
                if row_index != column and augmented[row_index] & (1 << column):
                    augmented[row_index] ^= augmented[column]
        inverse_rows = tuple(row >> size for row in augmented)
        return GF2Matrix(size, size, inverse_rows)

    def nullspace(self) -> GF2Subspace:
        """Return ``ker(self)`` as a canonical subspace of the input space."""

        return GF2Subspace(self.n_cols, self.rows).nullspace()

    def image(self, subspace: GF2Subspace | None = None) -> GF2Subspace:
        """Return the image of the full input or of a supplied subspace."""

        domain = GF2Subspace.full(self.n_cols) if subspace is None else subspace
        if not isinstance(domain, GF2Subspace):
            raise TypeError(f"expected GF2Subspace, got {type(domain).__name__}")
        if domain.dimension != self.n_cols:
            raise GF2MatrixShapeError(
                f"map input dimension {self.n_cols} does not match "
                f"subspace dimension {domain.dimension}"
            )
        return GF2Subspace(self.n_rows, tuple(self.apply(row) for row in domain.basis))

    def restricted_rank(self, subspace: GF2Subspace) -> int:
        """Return the rank of this map restricted to ``subspace``."""

        return self.image(subspace).rank

    def manifest(self) -> dict[str, object]:
        """Return a stable JSON-compatible structural manifest."""

        return {
            "kind": "gf2-matrix",
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "rows": list(self.rows),
        }


@dataclass(frozen=True, slots=True)
class GF2Subspace:
    """A canonical subspace of ``GF(2)**dimension``.

    ``basis`` may contain redundant or unordered rows when constructing an
    instance.  It is validated and replaced by its unique RREF basis.
    Coordinate zero is the least-significant bit.
    """

    dimension: int
    basis: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        dimension = _require_nonnegative_int(self.dimension, name="dimension")
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "basis", _canonical_rref(dimension, self.basis))

    @classmethod
    def rref(cls, dimension: int, vectors: Iterable[int]) -> GF2Subspace:
        """Construct the subspace spanned by ``vectors`` in canonical RREF."""

        return cls(dimension=dimension, basis=tuple(vectors))

    @classmethod
    def zero(cls, dimension: int) -> GF2Subspace:
        """Return the zero subspace."""

        return cls(dimension=dimension)

    @classmethod
    def full(cls, dimension: int) -> GF2Subspace:
        """Return the full ambient vector space."""

        dimension = _require_nonnegative_int(dimension, name="dimension")
        return cls(
            dimension=dimension, basis=tuple(1 << bit for bit in range(dimension))
        )

    @property
    def rank(self) -> int:
        """Dimension of this subspace."""

        return len(self.basis)

    def contains(self, vector: int) -> bool:
        """Return whether ``vector`` belongs to this subspace."""

        remainder = _validate_vector(vector, self.dimension)
        for row in self.basis:
            pivot = row & -row
            if remainder & pivot:
                remainder ^= row
        return remainder == 0

    def is_subspace_of(self, other: GF2Subspace) -> bool:
        """Return whether every vector in this subspace belongs to ``other``."""

        self._require_same_ambient(other)
        return all(other.contains(row) for row in self.basis)

    def sum(self, other: GF2Subspace) -> GF2Subspace:
        """Return the span of this subspace and ``other``."""

        self._require_same_ambient(other)
        return GF2Subspace(self.dimension, self.basis + other.basis)

    def is_direct_sum_with(self, other: GF2Subspace) -> bool:
        """Return whether this subspace and ``other`` intersect only at zero."""

        self._require_same_ambient(other)
        return self.intersection(other).rank == 0

    def basis_extension(self, *, within: GF2Subspace | None = None) -> GF2Subspace:
        """Return one deterministic complement that extends this basis.

        The result is a canonical subspace ``C`` such that
        ``self.sum(C) == within`` and the sum is direct.  The full ambient
        space is used when ``within`` is omitted.
        """

        container = GF2Subspace.full(self.dimension) if within is None else within
        self._require_same_ambient(container)
        if not self.is_subspace_of(container):
            raise ValueError("self must be a subspace of within")

        extension_space = self
        extension_rows = []
        for row in container.basis:
            if not extension_space.contains(row):
                extension_rows.append(row)
                extension_space = GF2Subspace(
                    self.dimension, extension_space.basis + (row,)
                )
        result = GF2Subspace(self.dimension, tuple(extension_rows))
        if not self.is_direct_sum_with(result) or self.sum(result) != container:
            raise AssertionError("internal error while extending GF(2) basis")
        return result

    def span_vectors(self, *, max_results: int) -> tuple[int, ...]:
        """Enumerate all vectors in this span in numeric canonical order."""

        limit = _require_positive_int(max_results, name="max_results")
        count = _bounded_power_of_two(self.rank, limit, operation="span enumeration")
        vectors = []
        for coefficients in range(count):
            vector = 0
            for row_index, row in enumerate(self.basis):
                if coefficients & (1 << row_index):
                    vector ^= row
            vectors.append(vector)
        return tuple(sorted(vectors))

    def nullspace(self) -> GF2Subspace:
        """Return the nullspace of the matrix whose rows are this basis.

        Equivalently, this is the orthogonal complement under the standard
        non-degenerate dot product over ``GF(2)``.
        """

        pivot_columns = tuple((row & -row).bit_length() - 1 for row in self.basis)
        pivot_set = frozenset(pivot_columns)
        null_basis = []
        for free_column in range(self.dimension):
            if free_column in pivot_set:
                continue
            vector = 1 << free_column
            for pivot_column, row in zip(pivot_columns, self.basis):
                if row & (1 << free_column):
                    vector |= 1 << pivot_column
            null_basis.append(vector)
        return GF2Subspace(self.dimension, tuple(null_basis))

    def intersection(self, other: GF2Subspace) -> GF2Subspace:
        """Return the exact intersection with ``other``."""

        self._require_same_ambient(other)
        # For a non-degenerate finite-dimensional inner product,
        # U intersection V = (U-perp + V-perp)-perp.
        orthogonal_sum = GF2Subspace(
            self.dimension,
            self.nullspace().basis + other.nullspace().basis,
        )
        return orthogonal_sum.nullspace()

    def canonical_complements(
        self,
        *,
        within: GF2Subspace | None = None,
        max_results: int,
    ) -> tuple[GF2Subspace, ...]:
        """Enumerate every complement of ``self`` inside ``within`` exactly once.

        A returned ``C`` satisfies ``self + C == within`` and
        ``self.intersection(C).rank == 0``.  If ``within`` is omitted, it is
        the full ambient space.

        Given one fixed complement ``F``, every complement is uniquely the
        graph of a linear map ``F -> self``.  Consequently this method emits
        exactly ``2**(self.rank * (within.rank - self.rank))`` canonical RREF
        subspaces.  Enumeration is rejected before allocation when that count
        exceeds the mandatory ``max_results``.
        """

        limit = _require_positive_int(max_results, name="max_results")
        container = GF2Subspace.full(self.dimension) if within is None else within
        self._require_same_ambient(container)
        if not self.is_subspace_of(container):
            raise ValueError("self must be a subspace of within")

        fixed = self.basis_extension(within=container)
        complement_rank = container.rank - self.rank
        if fixed.rank != complement_rank:
            raise AssertionError("internal error while constructing fixed complement")

        map_bits = self.rank * complement_rank
        count = _bounded_power_of_two(
            map_bits, limit, operation="complement enumeration"
        )
        complements = []
        image_mask = (1 << self.rank) - 1
        for encoding in range(count):
            graph_rows = []
            for fixed_index, fixed_row in enumerate(fixed.basis):
                image_encoding = (encoding >> (fixed_index * self.rank)) & image_mask
                image = 0
                for source_index, source_row in enumerate(self.basis):
                    if image_encoding & (1 << source_index):
                        image ^= source_row
                graph_rows.append(fixed_row ^ image)
            complements.append(GF2Subspace(self.dimension, tuple(graph_rows)))
        return tuple(sorted(complements, key=lambda space: space.basis))

    @classmethod
    def enumerate_subspaces(
        cls,
        dimension: int,
        rank: int | None = None,
        *,
        max_results: int,
    ) -> tuple[GF2Subspace, ...]:
        """Enumerate subspaces as unique RREF matrices in structural order.

        ``rank=None`` enumerates every rank.  The exact Gaussian-binomial
        result count is checked against mandatory ``max_results`` before
        constructing any subspace.
        """

        dimension = _require_nonnegative_int(dimension, name="dimension")
        if rank is not None:
            rank = _require_nonnegative_int(rank, name="rank")
            if rank > dimension:
                raise ValueError(f"rank {rank} exceeds ambient dimension {dimension}")
        limit = _require_positive_int(max_results, name="max_results")
        _bounded_subspace_count(dimension, rank, limit)

        ranks = range(dimension + 1) if rank is None else (rank,)
        spaces = []
        columns = tuple(range(dimension))
        for subspace_rank in ranks:
            for pivots in combinations(columns, subspace_rank):
                pivot_set = frozenset(pivots)
                free_positions = tuple(
                    (row_index, column)
                    for row_index, pivot in enumerate(pivots)
                    for column in columns
                    if column > pivot and column not in pivot_set
                )
                for encoding in range(1 << len(free_positions)):
                    rows = [1 << pivot for pivot in pivots]
                    for free_index, (row_index, column) in enumerate(free_positions):
                        if encoding & (1 << free_index):
                            rows[row_index] |= 1 << column
                    spaces.append(cls(dimension, tuple(rows)))
        return tuple(sorted(spaces, key=lambda space: (space.rank, space.basis)))

    def _require_same_ambient(self, other: object) -> None:
        if not isinstance(other, GF2Subspace):
            raise TypeError(f"expected GF2Subspace, got {type(other).__name__}")
        if self.dimension != other.dimension:
            raise ValueError(
                "ambient dimensions differ: " f"{self.dimension} != {other.dimension}"
            )


__all__ = [
    "GF2Matrix",
    "GF2MatrixShapeError",
    "GF2SingularMatrixError",
    "GF2Subspace",
]
