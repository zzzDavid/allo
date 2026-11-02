# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""GF(2) algebra tests, ported from the snapshot (df35093) test_gf2.py and
test_gf2_adversarial.py.  The adversarial half uses literal truth tables and
set images as oracles and imports no layout or backend implementation.
"""

from dataclasses import FrozenInstanceError
from itertools import combinations, permutations

import numpy as np
import pytest

from allo.pim.gf2 import (
    GF2Matrix,
    GF2MatrixShapeError,
    GF2SingularMatrixError,
    GF2Subspace,
)


def _direct_span(rows):
    result = {0}
    for row in rows:
        result |= {vector ^ row for vector in tuple(result)}
    return frozenset(result)


def _direct_subspaces(dimension):
    nonzero_vectors = tuple(range(1, 1 << dimension))
    result = set()
    for selected in range(1 << len(nonzero_vectors)):
        rows = (
            vector
            for index, vector in enumerate(nonzero_vectors)
            if selected & (1 << index)
        )
        result.add(_direct_span(rows))
    return result


def _numpy_reference_rank(dimension, rows):
    matrix = np.asarray(
        [[(row >> column) & 1 for column in range(dimension)] for row in rows],
        dtype=np.uint8,
    )
    if matrix.size == 0:
        return 0
    row_count, column_count = matrix.shape
    pivot_row = 0
    for column in range(column_count):
        candidates = np.flatnonzero(matrix[pivot_row:, column])
        if len(candidates) == 0:
            continue
        selected = pivot_row + int(candidates[0])
        matrix[[pivot_row, selected]] = matrix[[selected, pivot_row]]
        for row in range(row_count):
            if row != pivot_row and matrix[row, column]:
                matrix[row] ^= matrix[pivot_row]
        pivot_row += 1
        if pivot_row == row_count:
            break
    return pivot_row


def _all_matrices(n_rows, n_cols):
    row_mask = (1 << n_cols) - 1
    for encoding in range(1 << (n_rows * n_cols)):
        yield GF2Matrix(
            n_rows,
            n_cols,
            tuple(
                (encoding >> (row_index * n_cols)) & row_mask
                for row_index in range(n_rows)
            ),
        )


def test_matrix_apply_transpose_rank_and_nullspace_match_direct_exhaustion():
    for n_rows in range(4):
        for n_cols in range(4):
            for matrix in _all_matrices(n_rows, n_cols):
                image = {matrix.apply(vector) for vector in range(1 << n_cols)}
                kernel = {
                    vector for vector in range(1 << n_cols) if matrix.apply(vector) == 0
                }
                assert matrix.rank == len(image).bit_length() - 1
                assert (
                    set(matrix.nullspace().span_vectors(max_results=1 << n_cols))
                    == kernel
                )
                assert matrix.transpose().transpose() == matrix


def test_matrix_multiplication_matches_direct_composition_exhaustively():
    for n_rows in range(3):
        for inner in range(3):
            for n_cols in range(3):
                for left in _all_matrices(n_rows, inner):
                    for right in _all_matrices(inner, n_cols):
                        product = left @ right
                        assert product == left.multiply(right)
                        assert product.n_rows == n_rows
                        assert product.n_cols == n_cols
                        assert all(
                            product.apply(vector) == left.apply(right.apply(vector))
                            for vector in range(1 << n_cols)
                        )


def test_matrix_inverse_matches_direct_bijection_exhaustively():
    for size in range(4):
        identity = GF2Matrix.identity(size)
        for matrix in _all_matrices(size, size):
            image = {matrix.apply(vector) for vector in range(1 << size)}
            if len(image) == 1 << size:
                inverse = matrix.inverse()
                assert matrix @ inverse == identity
                assert inverse @ matrix == identity
                assert inverse.inverse() == matrix
            else:
                with pytest.raises(GF2SingularMatrixError, match="singular"):
                    matrix.inverse()


def test_restricted_rank_and_image_match_direct_set_images():
    for n_rows in range(4):
        for n_cols in range(4):
            subspaces = GF2Subspace.enumerate_subspaces(n_cols, max_results=128)
            for matrix in _all_matrices(n_rows, n_cols):
                assert matrix.image().rank == matrix.rank
                for subspace in subspaces:
                    direct_image = {
                        matrix.apply(vector)
                        for vector in subspace.span_vectors(max_results=1 << n_cols)
                    }
                    expected_rank = len(direct_image).bit_length() - 1
                    assert matrix.restricted_rank(subspace) == expected_rank
                    assert (
                        set(
                            matrix.image(subspace).span_vectors(max_results=1 << n_rows)
                        )
                        == direct_image
                    )


def test_matrix_manifest_immutability_and_validation():
    matrix = GF2Matrix(2, 3, [0b101, 0b011])
    assert matrix.rows == (0b101, 0b011)
    assert matrix.manifest() == {
        "kind": "gf2-matrix",
        "n_rows": 2,
        "n_cols": 3,
        "rows": [0b101, 0b011],
    }
    manifest = matrix.manifest()
    manifest["rows"].append(0)
    assert matrix.rows == (0b101, 0b011)
    assert hash(matrix)
    with pytest.raises(FrozenInstanceError):
        matrix.rows = ()
    with pytest.raises(GF2MatrixShapeError, match="declares 2 rows"):
        GF2Matrix(2, 3, (0,))
    with pytest.raises(ValueError, match="outside"):
        GF2Matrix(1, 2, (0b100,))
    with pytest.raises(TypeError, match="integer"):
        GF2Matrix(True, 2, ())
    with pytest.raises(ValueError, match="outside"):
        matrix.apply(0b1000)
    with pytest.raises(GF2MatrixShapeError, match="cannot multiply"):
        matrix @ GF2Matrix.identity(2)
    with pytest.raises(GF2MatrixShapeError, match="square"):
        matrix.inverse()
    with pytest.raises(GF2MatrixShapeError, match="does not match"):
        matrix.restricted_rank(GF2Subspace.full(2))
    with pytest.raises(TypeError, match="expected GF2Matrix"):
        matrix.multiply(object())


@pytest.mark.parametrize("dimension", range(5))
def test_canonical_subspace_enumerator_matches_direct_exhaustion(dimension):
    expected = _direct_subspaces(dimension)
    actual = GF2Subspace.enumerate_subspaces(dimension, max_results=len(expected))

    actual_spans = {
        frozenset(space.span_vectors(max_results=1 << dimension)) for space in actual
    }
    assert actual_spans == expected
    assert len(actual) == len(expected)
    assert tuple((space.rank, space.basis) for space in actual) == tuple(
        sorted((space.rank, space.basis) for space in actual)
    )


def test_rank_rref_and_membership_match_exhaustive_and_numpy_references():
    for dimension in range(5):
        vectors = tuple(range(1 << dimension))
        # Every subset for dimensions through three, and every matrix with at
        # most three rows for dimension four.
        max_rows = len(vectors) if dimension <= 3 else 3
        for row_count in range(max_rows + 1):
            for rows in combinations(vectors, row_count):
                space = GF2Subspace.rref(dimension, rows)
                direct = _direct_span(rows)
                assert space.rank == (len(direct).bit_length() - 1)
                assert space.rank == _numpy_reference_rank(dimension, rows)
                assert {
                    vector for vector in range(1 << dimension) if space.contains(vector)
                } == direct


@pytest.mark.parametrize("dimension", range(5))
def test_nullspace_matches_direct_orthogonality(dimension):
    spaces = GF2Subspace.enumerate_subspaces(dimension, max_results=128)
    for space in spaces:
        expected = {
            vector
            for vector in range(1 << dimension)
            if all((row & vector).bit_count() % 2 == 0 for row in space.basis)
        }
        nullspace = space.nullspace()
        assert set(nullspace.span_vectors(max_results=1 << dimension)) == expected
        assert nullspace.rank == dimension - space.rank
        assert nullspace.nullspace() == space


@pytest.mark.parametrize("dimension", range(5))
def test_intersection_matches_direct_set_intersection(dimension):
    spaces = GF2Subspace.enumerate_subspaces(dimension, max_results=128)
    span_by_space = {
        space: set(space.span_vectors(max_results=1 << dimension)) for space in spaces
    }
    for left in spaces:
        for right in spaces:
            expected = span_by_space[left] & span_by_space[right]
            actual = left.intersection(right)
            assert set(actual.span_vectors(max_results=1 << dimension)) == expected
            assert actual == right.intersection(left)


@pytest.mark.parametrize("dimension", range(5))
def test_sum_direct_sum_and_basis_extension_match_direct_sets(dimension):
    spaces = GF2Subspace.enumerate_subspaces(dimension, max_results=128)
    span_by_space = {
        space: set(space.span_vectors(max_results=1 << dimension)) for space in spaces
    }
    for left in spaces:
        for right in spaces:
            actual_sum = left.sum(right)
            direct_sum_span = {
                left_vector ^ right_vector
                for left_vector in span_by_space[left]
                for right_vector in span_by_space[right]
            }
            assert (
                set(actual_sum.span_vectors(max_results=1 << dimension))
                == direct_sum_span
            )
            assert left.is_direct_sum_with(right) == (
                span_by_space[left] & span_by_space[right] == {0}
            )
            if left.is_subspace_of(right):
                extension = left.basis_extension(within=right)
                assert left.is_direct_sum_with(extension)
                assert left.sum(extension) == right
                assert extension == left.basis_extension(within=right)


@pytest.mark.parametrize("dimension", range(5))
def test_canonical_complements_match_direct_subspace_filter(dimension):
    spaces = GF2Subspace.enumerate_subspaces(dimension, max_results=128)
    for container in spaces:
        for subspace in spaces:
            if not subspace.is_subspace_of(container):
                continue
            expected = {
                candidate
                for candidate in spaces
                if candidate.is_subspace_of(container)
                and subspace.intersection(candidate).rank == 0
                and subspace.rank + candidate.rank == container.rank
            }
            expected_count = 1 << (subspace.rank * (container.rank - subspace.rank))
            actual = subspace.canonical_complements(
                within=container, max_results=expected_count
            )
            assert set(actual) == expected
            assert len(actual) == len(set(actual)) == expected_count
            assert tuple(space.basis for space in actual) == tuple(
                sorted(space.basis for space in actual)
            )


def test_known_rref_and_complement_examples():
    space = GF2Subspace.rref(4, (0b0011, 0b0110, 0b1100))
    assert space.basis == (0b1001, 0b1010, 0b1100)
    assert space.rank == 3

    line = GF2Subspace.rref(3, (0b001,))
    complements = line.canonical_complements(max_results=4)
    assert len(complements) == 4
    assert all(complement.rank == 2 for complement in complements)
    assert all(line.intersection(complement).rank == 0 for complement in complements)


def test_representation_is_immutable_hashable_and_deterministic():
    canonical = GF2Subspace.rref(4, (0b0011, 0b0101, 0b1001))
    reordered = GF2Subspace.rref(4, [0, 0b1001, 0b0101, 0b0011, 0b0011])
    assert canonical == reordered
    assert isinstance(reordered.basis, tuple)
    assert hash(canonical) == hash(reordered)
    with pytest.raises(FrozenInstanceError):
        canonical.dimension = 5
    with pytest.raises(FrozenInstanceError):
        canonical.basis = ()


@pytest.mark.parametrize(
    "constructor,error,match",
    [
        (lambda: GF2Subspace(-1), ValueError, "non-negative"),
        (lambda: GF2Subspace(True), TypeError, "integer"),
        (lambda: GF2Subspace(2, (0b100,)), ValueError, "outside"),
        (lambda: GF2Subspace(2, (-1,)), ValueError, "non-negative"),
        (lambda: GF2Subspace(2, (1.0,)), TypeError, "integer"),
        (
            lambda: GF2Subspace.enumerate_subspaces(3, rank=4, max_results=1),
            ValueError,
            "exceeds ambient",
        ),
        (
            lambda: GF2Subspace.enumerate_subspaces(4, max_results=66),
            ValueError,
            "exceeds max_results",
        ),
        (
            lambda: GF2Subspace.full(4).span_vectors(max_results=15),
            ValueError,
            "exceeding max_results",
        ),
        (
            lambda: GF2Subspace.rref(4, (0b0011, 0b0101)).canonical_complements(
                max_results=15
            ),
            ValueError,
            "exceeding max_results",
        ),
    ],
)
def test_invalid_or_unbounded_requests_fail_closed(constructor, error, match):
    with pytest.raises(error, match=match):
        constructor()


def test_cross_ambient_and_non_containment_fail_closed():
    small = GF2Subspace.full(2)
    large = GF2Subspace.full(3)
    with pytest.raises(ValueError, match="ambient dimensions differ"):
        small.intersection(large)
    with pytest.raises(TypeError, match="expected GF2Subspace"):
        small.is_subspace_of(object())

    line_x = GF2Subspace.rref(2, (0b01,))
    line_y = GF2Subspace.rref(2, (0b10,))
    with pytest.raises(ValueError, match="subspace of within"):
        line_x.canonical_complements(within=line_y, max_results=1)


def test_zero_dimensional_edge_case():
    zero = GF2Subspace.zero(0)
    assert zero == GF2Subspace.full(0)
    assert zero.rank == 0
    assert zero.contains(0)
    assert zero.nullspace() == zero
    assert zero.intersection(zero) == zero
    assert zero.span_vectors(max_results=1) == (0,)
    assert zero.canonical_complements(max_results=1) == (zero,)
    assert GF2Subspace.enumerate_subspaces(0, max_results=1) == (zero,)


# --- adversarial (independent oracles) ---
def _matrix_4x4(encoding):
    return GF2Matrix(
        4,
        4,
        tuple((encoding >> (4 * row_index)) & 0b1111 for row_index in range(4)),
    )


def _literal_span(rows):
    vectors = {0}
    for row in rows:
        vectors |= {vector ^ row for vector in tuple(vectors)}
    return vectors


def _literal_apply(matrix, vector):
    return sum(
        ((row & vector).bit_count() & 1) << row_index
        for row_index, row in enumerate(matrix.rows)
    )


def test_every_4x4_matrix_rank_kernel_and_inverse_match_truth_tables():
    """Cover the audit's inclusive 4-by-4 matrix boundary."""

    identity = GF2Matrix.identity(4)
    for encoding in range(1 << 16):
        matrix = _matrix_4x4(encoding)
        outputs = tuple(_literal_apply(matrix, vector) for vector in range(1 << 4))
        image = set(outputs)
        kernel = {vector for vector, output in enumerate(outputs) if output == 0}

        assert tuple(matrix.apply(vector) for vector in range(1 << 4)) == outputs
        assert matrix.rank == len(image).bit_length() - 1
        assert set(matrix.nullspace().span_vectors(max_results=1 << 4)) == kernel
        if len(image) == 1 << 4:
            inverse = matrix.inverse()
            assert matrix @ inverse == identity
            assert inverse @ matrix == identity


def test_work_iteration_operand_matrix_directions_are_compositionally_exact():
    # W maps packed (owner_0, owner_1, local_0) work bits to iteration bits.
    work_to_iteration = GF2Matrix(
        3,
        3,
        (
            0b101,  # iteration_0 = owner_0 xor local_0
            0b010,  # iteration_1 = owner_1
            0b100,  # iteration_2 = local_0
        ),
    )
    # A maps iteration bits to operand bits.
    iteration_to_operand = GF2Matrix(2, 3, (0b011, 0b100))
    work_to_operand = iteration_to_operand @ work_to_iteration
    iteration_to_work = work_to_iteration.inverse()

    for work in range(1 << 3):
        iteration = work_to_iteration.apply(work)
        assert iteration == (
            (((work >> 0) ^ (work >> 2)) & 1) | (work & 0b010) | (work & 0b100)
        )
        assert iteration_to_work.apply(iteration) == work
        assert work_to_operand.apply(work) == iteration_to_operand.apply(iteration)


def test_rref_is_invariant_under_basis_permutation_and_row_addition():
    for space in GF2Subspace.enumerate_subspaces(4, max_results=67):
        for order in permutations(space.basis):
            assert GF2Subspace.rref(4, order) == space
            for target in range(space.rank if space.rank >= 2 else 0):
                changed = list(order)
                changed[target] ^= order[(target + 1) % space.rank]
                assert GF2Subspace.rref(4, changed) == space


def test_dimension_four_restriction_rank_matches_literal_images():
    subspaces = GF2Subspace.enumerate_subspaces(4, max_results=67)
    # All maps to one bit plus a structurally varied deterministic slice of
    # square maps exercise restriction in the dimension omitted by the base
    # test's rectangular-matrix loops.
    matrices = [GF2Matrix(1, 4, (row,)) for row in range(1 << 4)]
    matrices += [_matrix_4x4(encoding * 257) for encoding in range(256)]

    for matrix in matrices:
        for space in subspaces:
            direct_image = {
                _literal_apply(matrix, vector) for vector in _literal_span(space.basis)
            }
            assert matrix.restricted_rank(space) == (len(direct_image).bit_length() - 1)
            assert (
                set(matrix.image(space).span_vectors(max_results=1 << matrix.n_rows))
                == direct_image
            )


def test_dimension_four_complements_match_literal_set_definition():
    spaces = GF2Subspace.enumerate_subspaces(4, max_results=67)
    spans = {space: frozenset(_literal_span(space.basis)) for space in spaces}
    for container in spaces:
        for subspace in spaces:
            if not spans[subspace] <= spans[container]:
                continue
            expected = {
                candidate
                for candidate in spaces
                if spans[candidate] <= spans[container]
                and spans[candidate] & spans[subspace] == {0}
                and {
                    left ^ right
                    for left in spans[subspace]
                    for right in spans[candidate]
                }
                == spans[container]
            }
            count = 1 << (subspace.rank * (container.rank - subspace.rank))
            actual = subspace.canonical_complements(within=container, max_results=count)
            assert len(actual) == len(set(actual)) == count
            assert set(actual) == expected


def test_enumeration_limits_are_exact_and_reject_boolean_limits():
    assert len(GF2Subspace.enumerate_subspaces(4, max_results=67)) == 67
    with pytest.raises(ValueError, match="exceeds max_results"):
        GF2Subspace.enumerate_subspaces(4, max_results=66)
    with pytest.raises(TypeError, match="integer"):
        GF2Subspace.enumerate_subspaces(0, max_results=True)

    plane = GF2Subspace.rref(4, (0b0001, 0b0010))
    assert len(plane.canonical_complements(max_results=16)) == 16
    with pytest.raises(ValueError, match="exceeding max_results"):
        plane.canonical_complements(max_results=15)


def test_zero_dimension_rectangular_matrix_operations_are_total():
    no_input = GF2Matrix.zero(3, 0)
    no_output = GF2Matrix.zero(0, 3)
    empty = GF2Matrix.identity(0)

    assert no_input.apply(0) == 0
    assert no_input.rank == 0
    assert no_input.nullspace() == GF2Subspace.zero(0)
    assert no_output.apply(0b111) == 0
    assert no_output.image() == GF2Subspace.zero(0)
    assert no_output @ no_input == empty
    assert empty.inverse() == empty
