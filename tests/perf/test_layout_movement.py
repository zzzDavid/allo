# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Movement algebra: the rank formula, exact masked counts, byte scaling.

The raw-set oracles are ported from the snapshot (df35093)
test_prelayout_movement_algebra_oracle.py and use integer parity and Python
sets only.  The GEMV cases reproduce the counts of paper Section IV-A
(Fig. ll-example: lane l = 4m + k stores W[m, k]).
"""

# pylint: disable=missing-function-docstring

from fractions import Fraction
from itertools import product

import pytest

from allo.pim.gf2 import GF2Matrix, GF2Subspace
from allo.pim.layout_movement import (
    AccessMode,
    LayoutAccess,
    exact_active_owner_incidence,
    full_carrier_rank_incidence,
    layout_movement_cost,
    movement_rank,
    scale_extra_owner_incidence,
    strict_reduction_kernel,
)
from allo.pim.layout_problem import (
    LocalAxis,
    OwnerAxis,
    OwnerAxisOrigin,
    OwnerHierarchy,
    OwnerLevel,
)

# --------------------------------------------------------------------------- #
# Raw oracles (no production GF(2) helpers)
# --------------------------------------------------------------------------- #


def _raw_span(generators):
    vectors = {0}
    for generator in generators:
        vectors |= {vector ^ generator for vector in tuple(vectors)}
    return frozenset(vectors)


def _raw_subspaces(dimension):
    nonzero = tuple(range(1, 1 << dimension))
    spaces = {
        _raw_span(
            vector
            for ordinal, vector in enumerate(nonzero)
            if encoding & (1 << ordinal)
        )
        for encoding in range(1 << len(nonzero))
    }
    return tuple(sorted(spaces, key=lambda space: (len(space), tuple(sorted(space)))))


def _raw_coset_label(vector, space):
    return min(vector ^ member for member in space)


def _raw_owner_incidence(dimension, kernel, local_space):
    fibers = {_raw_coset_label(point, kernel) for point in range(1 << dimension)}
    return sum(
        len(
            {
                _raw_coset_label(point, local_space)
                for point in range(1 << dimension)
                if _raw_coset_label(point, kernel) == fiber
            }
        )
        - 1
        for fiber in fibers
    )


def _raw_apply(rows, vector):
    return sum(
        ((row & vector).bit_count() & 1) << output_bit
        for output_bit, row in enumerate(rows)
    )


def _raw_invertible_matrices(dimension):
    extent = 1 << dimension
    output = []
    for rows in product(range(extent), repeat=dimension):
        image = tuple(_raw_apply(rows, point) for point in range(extent))
        if len(set(image)) == extent:
            output.append(
                (rows, {iteration: work for work, iteration in enumerate(image)})
            )
    return tuple(output)


def _raw_active_incidence(access_row, active_points, inverse):
    owners = {}
    for iteration in active_points:
        operand = (access_row & iteration).bit_count() & 1
        owners.setdefault(operand, set()).add(inverse[iteration] & 1)
    ordered = tuple(
        (operand, tuple(sorted(owner_set)))
        for operand, owner_set in sorted(owners.items())
    )
    return ordered, sum(len(owner_set) - 1 for _, owner_set in ordered)


def _hierarchy(*owner_widths, local_bits):
    return OwnerHierarchy(
        tuple(
            OwnerLevel(
                level,
                (OwnerAxis(level, 1 << width, OwnerAxisOrigin.MAPPED_UNIT),),
            )
            for level, width in enumerate(owner_widths)
        ),
        (LocalAxis(0, local_bits),) if local_bits else (),
    )


def _layout(columns):
    """Work-to-iteration matrix whose column ``i`` is the image of work bit i."""

    dimension = len(columns)
    return GF2Matrix(
        dimension,
        dimension,
        tuple(
            sum(((column >> bit) & 1) << index for index, column in enumerate(columns))
            for bit in range(dimension)
        ),
    )


# --------------------------------------------------------------------------- #
# Rank formula and exact masked count against raw sets
# --------------------------------------------------------------------------- #


def test_all_4775_raw_subspace_pairs_match_the_rank_formula():
    expected_space_counts = (1, 2, 5, 16, 67)
    pair_count = 0
    for dimension, expected_count in enumerate(expected_space_counts):
        spaces = _raw_subspaces(dimension)
        assert len(spaces) == expected_count
        for kernel in spaces:
            for local_space in spaces:
                raw_intersection = len(kernel & local_space).bit_length() - 1
                raw_kernel = len(kernel).bit_length() - 1
                raw_rank = raw_kernel - raw_intersection
                raw_unique = 1 << (dimension - raw_kernel)
                raw_extra = raw_unique * ((1 << raw_rank) - 1)
                assert _raw_owner_incidence(dimension, kernel, local_space) == raw_extra

                kernel_space = GF2Subspace(dimension, tuple(kernel))
                local = GF2Subspace(dimension, tuple(local_space))
                production = full_carrier_rank_incidence(kernel_space, local)
                assert production.kernel_dimension == raw_kernel
                assert production.intersection_dimension == raw_intersection
                assert production.movement_rank == raw_rank
                assert movement_rank(kernel_space, local) == raw_rank
                assert production.unique_operand_points == raw_unique
                assert production.extra_owner_incidence == raw_extra
                pair_count += 1
    assert pair_count == 4_775


def test_2740_raw_parity_layout_mask_cases_match_the_exact_active_count():
    expected_gl_counts = {1: 1, 2: 6, 3: 168}
    case_count = 0
    for dimension, expected_layouts in expected_gl_counts.items():
        layouts = _raw_invertible_matrices(dimension)
        assert len(layouts) == expected_layouts
        extent = 1 << dimension
        masks = (tuple(range(extent)), tuple(range(extent - 1)))
        hierarchy = _hierarchy(1, local_bits=dimension - 1)
        for rows, inverse in layouts:
            work_to_iteration = GF2Matrix(dimension, dimension, rows)
            for access_row in range(extent):
                access_matrix = GF2Matrix(1, dimension, (access_row,))
                for active_points in masks:
                    expected_table, expected_extra = _raw_active_incidence(
                        access_row, active_points, inverse
                    )
                    production = exact_active_owner_incidence(
                        access_matrix, active_points, work_to_iteration, hierarchy, 0
                    )
                    assert production.owners_by_operand_point == expected_table
                    assert production.extra_owner_incidence == expected_extra
                    assert production.active_iteration_point_count == len(active_points)
                    case_count += 1
    assert case_count == 2_740


def test_six_bit_full_active_set_agrees_with_the_rank_formula():
    hierarchy = _hierarchy(3, local_bits=3)
    access = GF2Matrix(3, 6, (0b001000, 0b010000, 0b100000))
    direct = exact_active_owner_incidence(
        access, tuple(range(64)), GF2Matrix.identity(6), hierarchy, 0
    )
    rank = full_carrier_rank_incidence(
        access.nullspace(), hierarchy.local_coordinate_space_after(0)
    )
    assert direct.extra_owner_incidence == rank.extra_owner_incidence == 56


# --------------------------------------------------------------------------- #
# Byte scaling and the LayoutAccess record
# --------------------------------------------------------------------------- #


def test_scaling_is_exact_and_supports_sub_byte_elements():
    assert (
        scale_extra_owner_incidence(12, element_bytes=2, cost_per_byte=Fraction(1, 3))
        == 8
    )
    assert scale_extra_owner_incidence(
        8, element_bytes=Fraction(11, 8), cost_per_byte=1
    ) == Fraction(11)
    assert scale_extra_owner_incidence(5, element_bytes=4, cost_per_byte=0) == 0
    with pytest.raises(ValueError):
        scale_extra_owner_incidence(-1, element_bytes=1, cost_per_byte=1)
    with pytest.raises(ValueError):
        scale_extra_owner_incidence(1, element_bytes=0, cost_per_byte=1)
    with pytest.raises(ValueError):
        scale_extra_owner_incidence(1, element_bytes=1, cost_per_byte=-1)


def test_layout_access_normalizes_to_exact_fractions():
    access = LayoutAccess("x", GF2Matrix(1, 2, (1,)), AccessMode.READ, 2, (1, 0.5))
    assert access.element_bytes == Fraction(2)
    assert access.cost_per_level == (Fraction(1), Fraction(1, 2))
    assert access.kernel == GF2Subspace(2, (2,))
    assert not access.is_strict_reduction
    strict = LayoutAccess(
        "y",
        GF2Matrix(1, 2, (1,)),
        AccessMode.WRITE_ACCUMULATE,
        2,
        (1,),
        reassociation_allowed=False,
    )
    assert strict.is_strict_reduction


@pytest.mark.parametrize(
    "kwargs,error",
    (
        ({"name": ""}, ValueError),
        ({"mode": "read"}, TypeError),
        ({"element_bytes": 0}, ValueError),
        ({"cost_per_level": (-1,)}, ValueError),
        ({"cost_per_level": (float("nan"),)}, ValueError),
        ({"cost_per_level": ("1",)}, TypeError),
        ({"reassociation_allowed": 1}, TypeError),
    ),
)
def test_layout_access_rejects_malformed_fields(kwargs, error):
    fields = {
        "name": "x",
        "access_matrix": GF2Matrix(1, 2, (1,)),
        "mode": AccessMode.READ,
        "element_bytes": 1,
        "cost_per_level": (1,),
    }
    fields.update(kwargs)
    with pytest.raises(error):
        LayoutAccess(**fields)


# --------------------------------------------------------------------------- #
# Paper Section IV-A: the 4x4 GEMV on 16 lanes
# --------------------------------------------------------------------------- #

# Iteration bits (k0, k1, m0, m1) = bits 0..3, so iteration = k + 4 m.
A_X = GF2Matrix(2, 4, (0b0001, 0b0010))  # x[k]
A_Y = GF2Matrix(2, 4, (0b0100, 0b1000))  # y[m]
# Work bits: group (owner level 0) = bits 0, 1; inside the group = bits 2, 3.
# Lane l = 4 * group + slot.  "l = 4m + k": group <- m, slot <- k.
PAPER_LAYOUT = _layout((0b0100, 0b1000, 0b0001, 0b0010))
SWAPPED_LAYOUT = _layout((0b0001, 0b0010, 0b0100, 0b1000))


def _gemv_accesses(levels=1, *, y_weight=1, strict=False):
    return (
        LayoutAccess("x", A_X, AccessMode.READ, 1, (1,) * levels),
        LayoutAccess(
            "y",
            A_Y,
            AccessMode.WRITE_ACCUMULATE,
            1,
            (y_weight,) * levels,
            reassociation_allowed=not strict,
        ),
    )


def test_lane_six_stores_w12_needs_x2_and_contributes_to_y1():
    work = 1 | (2 << 2)  # lane 6 = 4 * group 1 + slot 2
    iteration = PAPER_LAYOUT.apply(work)
    assert iteration == 2 + 4 * 1  # W[m=1, k=2]
    assert A_X.apply(iteration) == 2
    assert A_Y.apply(iteration) == 1


def test_paper_example_group_counts_and_replication():
    hierarchy = _hierarchy(2, local_bits=2)
    group_local = PAPER_LAYOUT.image(hierarchy.local_coordinate_space_after(0))
    # Each group touches the span of the varying-bit columns of D_x, D_y.
    assert 1 << A_X.restricted_rank(group_local) == 4  # 4 distinct x per group
    assert 1 << A_Y.restricted_rank(group_local) == 1  # 1 y element per group

    x = full_carrier_rank_incidence(A_X.nullspace(), group_local)
    assert x.movement_rank == 2
    assert x.unique_operand_points * (1 << x.movement_rank) == 16  # placements
    assert x.extra_owner_incidence == 12  # 12 extra copies of x
    y = full_carrier_rank_incidence(A_Y.nullspace(), group_local)
    assert y.extra_owner_incidence == 0  # no partial result crosses groups

    cost = layout_movement_cost(_gemv_accesses(), hierarchy, PAPER_LAYOUT)
    assert [
        (t.access, t.movement_rank, t.extra_owner_incidence) for t in cost.terms
    ] == [
        ("x", 2, 12),
        ("y", 0, 0),
    ]
    assert cost.total == 12 and cost.legal


def test_swapped_assignment_reverses_the_counts():
    hierarchy = _hierarchy(2, local_bits=2)
    cost = layout_movement_cost(_gemv_accesses(), hierarchy, SWAPPED_LAYOUT)
    # One input per group, and every output has a partial in all four groups.
    assert [
        (t.access, t.movement_rank, t.extra_owner_incidence) for t in cost.terms
    ] == [
        ("x", 0, 0),
        ("y", 2, 12),
    ]


def test_within_group_partials_appear_at_the_lane_level():
    hierarchy = _hierarchy(2, 2, local_bits=0)
    cost = layout_movement_cost(_gemv_accesses(2), hierarchy, PAPER_LAYOUT)
    by_term = {(t.access, t.owner_level): t.extra_owner_incidence for t in cost.terms}
    assert by_term == {("x", 0): 12, ("x", 1): 12, ("y", 0): 0, ("y", 1): 12}


def test_masked_count_equals_the_rank_formula_on_the_full_carrier():
    hierarchy = _hierarchy(2, local_bits=2)
    for layout in (PAPER_LAYOUT, SWAPPED_LAYOUT):
        full = layout_movement_cost(_gemv_accesses(), hierarchy, layout)
        masked = layout_movement_cost(
            _gemv_accesses(), hierarchy, layout, active_points=tuple(range(16))
        )
        assert full.total == masked.total
        assert [t.extra_owner_incidence for t in full.terms] == [
            t.extra_owner_incidence for t in masked.terms
        ]


# --------------------------------------------------------------------------- #
# Section VI: masks and strict reductions
# --------------------------------------------------------------------------- #


def test_three_active_units_need_two_extra_copies_not_three():
    hierarchy = _hierarchy(2, local_bits=0)
    broadcast = GF2Matrix(0, 2, ())  # every unit reads the same scalar
    active = (0, 1, 2)  # m < 3, padded to 4
    padded = full_carrier_rank_incidence(
        broadcast.nullspace(), hierarchy.local_coordinate_space_after(0)
    )
    exact = exact_active_owner_incidence(
        broadcast, active, GF2Matrix.identity(2), hierarchy, 0
    )
    assert padded.extra_owner_incidence == 3
    assert exact.owners_by_operand_point == ((0, (0, 1, 2)),)
    assert exact.extra_owner_incidence == 2


def test_strict_reduction_is_illegal_when_an_owner_bit_splits_its_kernel():
    hierarchy = _hierarchy(2, local_bits=2)
    accesses = _gemv_accesses(strict=True)
    assert strict_reduction_kernel(accesses, 4) == A_Y.nullspace()
    paper = layout_movement_cost(accesses, hierarchy, PAPER_LAYOUT)
    assert paper.legal and paper.total == 12  # k stays inside one group
    swapped = layout_movement_cost(accesses, hierarchy, SWAPPED_LAYOUT)
    assert not swapped.legal and swapped.total is None
    masked = layout_movement_cost(
        accesses, hierarchy, SWAPPED_LAYOUT, active_points=(0, 1, 2, 3)
    )
    # Only m = 0 is active, but its four k values still sit in four groups.
    assert not masked.legal
    one_k = layout_movement_cost(
        accesses, hierarchy, SWAPPED_LAYOUT, active_points=(0, 4, 8)
    )
    # With a single active k per output the reduction is never split.
    assert one_k.legal


def test_cost_shape_errors_are_explicit():
    hierarchy = _hierarchy(2, local_bits=2)
    with pytest.raises(ValueError, match="per-level costs"):
        layout_movement_cost(_gemv_accesses(2), hierarchy, PAPER_LAYOUT)
    with pytest.raises(ValueError, match="strictly increasing"):
        layout_movement_cost(
            _gemv_accesses(), hierarchy, PAPER_LAYOUT, active_points=(2, 1)
        )
