# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimum-movement layout search: paper Section IV-C numbers from the package.

* 4 bits admit 20,160 one-to-one layouts; with three one-bit compute levels
  they reduce to 315 chains (flags) of 64 layouts each.
* On the GEMV objective the pruned search visits 576 layouts at two and
  three levels (4,032 at one level); the visited layouts are exactly the
  tie set, and the minimum and complete tie set agree with an independent
  brute-force oracle over GL(4, 2) that uses only integer parity and sets.
* Budget exits return a feasible incumbent, ``optimality_proven=False`` and
  the remaining frontier bounds.

Snapshot (df35093) telemetry witnesses from test_paper_partial_basis_search.py
are re-asserted to show the port keeps the traversal and budget accounting.
"""

# pylint: disable=missing-function-docstring

from fractions import Fraction
from functools import lru_cache
from itertools import product

import pytest

import allo
from allo.pim.gf2 import GF2Matrix, GF2Subspace
from allo.pim.layout_movement import AccessMode, LayoutAccess, layout_movement_cost
from allo.pim.layout_problem import (
    LocalAxis,
    OwnerAxis,
    OwnerAxisOrigin,
    OwnerHierarchy,
    OwnerLevel,
    owner_hierarchy_from_target,
)
from allo.pim.layout_search import (
    DEFAULT_MAX_SEARCH_NODES,
    LayoutSearchDomainError,
    LayoutSearchImplementation,
    LayoutSearchPolicy,
    LayoutSearchProblem,
    LayoutSearchTermination,
    NoLegalLayoutError,
    enumerate_layouts,
    gl_count,
    search_min_movement_layout,
    complete_scalable_layout_representative,
    search_scalable_layout_flags,
)
from allo.spmw_linear_layout import LinearLayout

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _hierarchy(*owner_widths, local_bits):
    return OwnerHierarchy(
        tuple(
            OwnerLevel(
                level, (OwnerAxis(level, 1 << width, OwnerAxisOrigin.MAPPED_UNIT),)
            )
            for level, width in enumerate(owner_widths)
        ),
        (LocalAxis(0, local_bits),) if local_bits else (),
    )


def _gemv_problem(bits, levels):
    """Reviews' GEMV fixture: m on the low bits, k on the high bits.

    x[k] is read (2 bytes, cost 1 per byte); y[m] accumulates distributable
    partials (2 bytes, cost 4 per byte) at every one-bit owner level.
    """

    m_bits = bits // 2
    x = LayoutAccess(
        "x",
        GF2Matrix(bits - m_bits, bits, tuple(1 << b for b in range(m_bits, bits))),
        AccessMode.READ,
        2,
        (1,) * levels,
    )
    y = LayoutAccess(
        "y",
        GF2Matrix(m_bits, bits, tuple(1 << b for b in range(m_bits))),
        AccessMode.WRITE_ACCUMULATE,
        2,
        (4,) * levels,
    )
    return LayoutSearchProblem(
        _hierarchy(*([1] * levels), local_bits=bits - levels), (x, y)
    )


def _read_problem(owner_widths, local_bits, rows, *, rate=1, forced=()):
    dimension = sum(owner_widths) + local_bits
    access = LayoutAccess(
        "a",
        GF2Matrix(len(rows), dimension, rows),
        AccessMode.READ,
        1,
        (rate,) * len(owner_widths),
    )
    return LayoutSearchProblem(
        _hierarchy(*owner_widths, local_bits=local_bits), (access,), forced
    )


# --------------------------------------------------------------------------- #
# Independent raw oracle: every invertible 4x4 matrix, parity and sets only
# --------------------------------------------------------------------------- #


def _parity(rows, vector):
    return sum(((row & vector).bit_count() & 1) << bit for bit, row in enumerate(rows))


@lru_cache(maxsize=None)
def _raw_gl(dimension):
    """Tuples (rows, inverse map) of every invertible matrix, by brute force."""

    extent = 1 << dimension
    output = []
    for rows in product(range(extent), repeat=dimension):
        image = [_parity(rows, work) for work in range(extent)]
        if len(set(image)) == extent:
            inverse = [0] * extent
            for work, iteration in enumerate(image):
                inverse[iteration] = work
            output.append((rows, tuple(inverse)))
    return tuple(output)


def _raw_cost(problem, inverse, points=None):
    """Exact owner-set cost; ``None`` when a strict reduction is split."""

    hierarchy = problem.hierarchy
    points = range(1 << problem.carrier_bits) if points is None else points
    total = Fraction()
    for access in problem.accesses:
        for level, (_, end) in enumerate(hierarchy.owner_bit_ranges):
            owners = {}
            for point in points:
                element = _parity(access.access_matrix.rows, point)
                owners.setdefault(element, set()).add(inverse[point] & ((1 << end) - 1))
            extra = sum(len(item) - 1 for item in owners.values())
            if (
                access.mode is AccessMode.WRITE_ACCUMULATE
                and not access.reassociation_allowed
            ):
                if extra:
                    return None
                continue
            total += access.cost_per_level[level] * access.element_bytes * extra
    return total


def _forced_ok(problem, inverse):
    hierarchy = problem.hierarchy
    for level, space in enumerate(problem.forced_local):
        start = hierarchy.owner_bit_ranges[level][1]
        for vector in space.basis:
            if inverse[vector] & ((1 << start) - 1):
                return False
    return True


def _raw_minimum(problem):
    best, ties = None, set()
    for rows, inverse in _raw_gl(problem.carrier_bits):
        if not _forced_ok(problem, inverse):
            continue
        cost = _raw_cost(problem, inverse)
        if cost is None:
            continue
        if best is None or cost < best:
            best, ties = cost, {rows}
        elif cost == best:
            ties.add(rows)
    return best, ties


# --------------------------------------------------------------------------- #
# Paper numbers: 20,160 / 315 / 576
# --------------------------------------------------------------------------- #


def test_four_bits_admit_20160_one_to_one_layouts():
    assert gl_count(4) == 20_160
    assert len(_raw_gl(4)) == 20_160
    problem = _gemv_problem(4, 3)
    layouts = [item.work_to_iteration.rows for item in enumerate_layouts(problem)]
    assert len(layouts) == len(set(layouts)) == 20_160
    assert set(layouts) == {rows for rows, _ in _raw_gl(4)}


def test_three_one_bit_levels_reduce_20160_layouts_to_315_chains():
    result = search_min_movement_layout(_gemv_problem(4, 3))
    telemetry = result.telemetry
    assert telemetry.expected_flags == 315
    assert telemetry.layouts_per_flag == 64
    assert telemetry.expected_layouts == 315 * 64 == 20_160
    assert telemetry.expected_prefix_nodes == 511
    assert DEFAULT_MAX_SEARCH_NODES == 1 + 511 + 20_160
    flags = {item.flag.spaces for item in enumerate_layouts(_gemv_problem(4, 3))}
    assert len(flags) == 315


@pytest.mark.parametrize("levels,visited", ((1, 4_032), (2, 576), (3, 576)))
def test_pruned_search_visits_576_and_matches_brute_force(levels, visited):
    problem = _gemv_problem(4, levels)
    result = search_min_movement_layout(problem)
    best, ties = _raw_minimum(problem)

    assert result.termination is LayoutSearchTermination.BOUND_PROVED
    assert result.optimality_proven
    assert result.implementation is LayoutSearchImplementation.BOUNDED
    assert result.visited == visited
    # The visited layouts are exactly the tie set.
    assert len(result.ties) == visited
    assert result.cost == best
    assert {matrix.rows for matrix in result.ties} == ties
    assert result.visited + result.pruned == 20_160
    assert result.remaining_lower_bounds == ()
    for matrix in result.ties[:: max(1, len(result.ties) // 16)]:
        assert (
            layout_movement_cost(problem.accesses, problem.hierarchy, matrix).total
            == best
        )

    unpruned = search_min_movement_layout(
        problem, LayoutSearchPolicy(enable_pruning=False, max_search_nodes=None)
    )
    assert unpruned.termination is LayoutSearchTermination.EXHAUSTED
    assert unpruned.visited == 20_160
    assert unpruned.cost == result.cost
    assert unpruned.ties == result.ties


def test_576_is_nine_tied_flags_of_64_layouts_at_three_levels():
    result = search_min_movement_layout(_gemv_problem(4, 3))
    assert len(result.tie_flags) == 9
    assert result.telemetry.visited_flags == 9
    assert len(result.ties) == 9 * 64


# --------------------------------------------------------------------------- #
# Snapshot telemetry witnesses (port fidelity)
# --------------------------------------------------------------------------- #


def test_n3_witness_prunes_before_complete_flag_search():
    result = search_min_movement_layout(_read_problem((1,), 2, (1,)))
    telemetry = result.telemetry
    assert result.cost == 0 and len(result.ties) == 24
    assert result.termination is LayoutSearchTermination.BOUND_PROVED
    assert telemetry.expected_prefix_nodes == 12
    assert telemetry.expected_flags == 7
    assert telemetry.layouts_per_flag == 24
    assert telemetry.evaluated_prefix_nodes == 6
    assert telemetry.expanded_prefix_nodes == 2
    assert telemetry.visited_flags == 1
    assert telemetry.pruned_frontier_roots == 3
    assert telemetry.pruned_prefix_descendants == 6
    assert telemetry.visited_layouts == 24
    assert telemetry.pruned_layouts == 144
    assert telemetry.search_nodes_used == 31


def test_n4_witness_has_47_prefixes_and_preserves_576_ties():
    result = search_min_movement_layout(_read_problem((2,), 2, (1, 2)))
    telemetry = result.telemetry
    assert result.cost == 0 and len(result.ties) == 576
    assert telemetry.expected_prefix_nodes == 47
    assert telemetry.expected_flags == 35
    assert telemetry.evaluated_prefix_nodes == 13
    assert telemetry.expanded_prefix_nodes == 2
    assert telemetry.visited_flags == 1
    assert telemetry.pruned_frontier_roots == 10
    assert telemetry.pruned_prefix_descendants == 34
    assert telemetry.pruned_layouts == 19_584
    assert telemetry.search_nodes_used == 590


def test_zero_weight_equality_exhausts_every_flag():
    result = search_min_movement_layout(_read_problem((1,), 2, (1,), rate=0))
    assert result.cost == 0 and len(result.ties) == 168
    assert result.termination is LayoutSearchTermination.EXHAUSTED
    assert result.telemetry.evaluated_prefix_nodes == 12
    assert result.telemetry.visited_flags == 7
    assert result.pruned == 0


@pytest.mark.parametrize(
    "budget,prefixes,layouts",
    ((1, 0, 0), (2, 1, 0), (5, 1, 0), (6, 5, 0), (30, 6, 0), (31, 6, 24)),
)
def test_atomic_node_budgets_are_conservation_complete(budget, prefixes, layouts):
    problem = _read_problem((1,), 2, (1,))
    result = search_min_movement_layout(
        problem, LayoutSearchPolicy(max_search_nodes=budget)
    )
    telemetry = result.telemetry
    assert telemetry.evaluated_prefix_nodes == prefixes
    assert telemetry.visited_layouts == layouts
    assert telemetry.search_nodes_used <= max(budget, 1)
    assert (
        telemetry.visited_layouts
        + telemetry.pruned_layouts
        + telemetry.unresolved_layouts
        == telemetry.expected_layouts
    )
    if budget < 31:
        assert result.termination is LayoutSearchTermination.NODE_BUDGET
        assert not result.optimality_proven
        assert result.remaining_lower_bounds
        # A feasible incumbent is always returned and its cost is real.
        assert len(result.ties) >= 1
        for matrix in result.ties:
            assert (
                layout_movement_cost(problem.accesses, problem.hierarchy, matrix).total
                == result.cost
            )
        assert telemetry.fallback_incumbent is (layouts == 0)
    else:
        assert result.termination is LayoutSearchTermination.BOUND_PROVED
        assert result.optimality_proven


def test_nested_forced_domain_matches_brute_force():
    problem = _read_problem(
        (1, 1), 1, (1,), forced=(GF2Subspace(3, (4,)), GF2Subspace.zero(3))
    )
    result = search_min_movement_layout(problem)
    best, ties = _raw_minimum(problem)
    assert result.telemetry.expected_flags == 9
    assert result.telemetry.expected_layouts == 72
    assert result.cost == best
    assert {matrix.rows for matrix in result.ties} == ties


# --------------------------------------------------------------------------- #
# Budget exit on the GEMV objective
# --------------------------------------------------------------------------- #


def test_budget_exit_keeps_the_incumbent_and_reports_the_frontier():
    problem = _gemv_problem(4, 3)
    full = search_min_movement_layout(problem)
    # Enough for the prefixes and the first tied flag, not for the rest.
    result = search_min_movement_layout(
        problem, LayoutSearchPolicy(max_search_nodes=300)
    )
    assert result.termination is LayoutSearchTermination.NODE_BUDGET
    assert result.optimality_proven is False
    assert not result.telemetry.fallback_incumbent
    assert result.ties and len(result.ties) < len(full.ties)
    assert set(result.ties) <= set(full.ties)
    assert result.cost == full.cost
    assert result.remaining_lower_bounds
    assert min(result.remaining_lower_bounds) <= result.cost
    assert result.telemetry.unresolved_layouts > 0


def test_budget_exit_before_any_flag_completes_a_fallback_incumbent():
    problem = _gemv_problem(4, 3)
    result = search_min_movement_layout(
        problem, LayoutSearchPolicy(max_search_nodes=40)
    )
    assert result.termination is LayoutSearchTermination.NODE_BUDGET
    assert not result.optimality_proven
    assert result.telemetry.fallback_incumbent
    assert result.telemetry.visited_layouts == 0
    (matrix,) = result.ties
    assert (
        layout_movement_cost(problem.accesses, problem.hierarchy, matrix).total
        == result.cost
    )
    assert result.cost >= search_min_movement_layout(problem).cost
    assert all(bound <= result.cost for bound in result.remaining_lower_bounds[:1])


# --------------------------------------------------------------------------- #
# Paper Section IV-A example through a Tenon target and the LinearLayout bridge
# --------------------------------------------------------------------------- #

A_X = GF2Matrix(2, 4, (0b0001, 0b0010))  # x[k]; iteration = k + 4 m
A_Y = GF2Matrix(2, 4, (0b0100, 0b1000))  # y[m]


def _layout(columns):
    return GF2Matrix(
        4,
        4,
        tuple(
            sum(((column >> bit) & 1) << index for index, column in enumerate(columns))
            for bit in range(4)
        ),
    )


PAPER_LAYOUT = _layout((0b0100, 0b1000, 0b0001, 0b0010))  # l = 4m + k
SWAPPED_LAYOUT = _layout((0b0001, 0b0010, 0b0100, 0b1000))  # groups select k


def _toy_apu():
    @allo.target("toy_apu")
    def toy():
        @allo.unit(mapping={"group": 4})
        def group():
            allo.reg(4, 16, axes={"lane": 4}, name="vr")

    return toy


def _paper_problem(levels=("group",), local_bits=2, *, y_weight=1, strict=False):
    hierarchy = owner_hierarchy_from_target(_toy_apu(), levels, local_bits=local_bits)
    count = len(hierarchy.levels)
    return LayoutSearchProblem(
        hierarchy,
        (
            LayoutAccess("x", A_X, AccessMode.READ, 1, (1,) * count),
            LayoutAccess(
                "y",
                A_Y,
                AccessMode.WRITE_ACCUMULATE,
                1,
                (y_weight,) * count,
                reassociation_allowed=not strict,
            ),
        ),
    )


def test_equal_weights_prefer_the_mixed_layout_over_both_paper_extremes():
    problem = _paper_problem()
    result = search_min_movement_layout(problem)
    # Each group keeps one bit of m and one of k: 2 inputs and 2 partials.
    assert result.cost == 8
    assert PAPER_LAYOUT not in result.ties and SWAPPED_LAYOUT not in result.ties
    assert result.cost == _raw_minimum(problem)[0]


def test_costly_partials_select_the_paper_layout_and_reject_the_swap():
    problem = _paper_problem(y_weight=4)
    result = search_min_movement_layout(problem)
    assert result.cost == 12  # x replicated 4x: 12 extra copies, y local
    assert PAPER_LAYOUT in result.ties
    assert SWAPPED_LAYOUT not in result.ties
    assert {matrix.rows for matrix in result.ties} == _raw_minimum(problem)[1]


def test_strict_reduction_forbids_owner_bits_that_split_its_kernel():
    problem = _paper_problem(strict=True)
    result = search_min_movement_layout(problem)
    best, ties = _raw_minimum(problem)
    assert result.cost == best == 12
    assert {matrix.rows for matrix in result.ties} == ties
    # Only T_1 = span(k) keeps every y[m] reduction inside one group.
    assert len(result.tie_flags) == 1
    assert result.tie_flags[0].spaces[1] == A_Y.nullspace()
    assert result.telemetry.expected_layouts == 576
    assert PAPER_LAYOUT in result.ties and SWAPPED_LAYOUT not in result.ties
    relaxed = search_min_movement_layout(_paper_problem())
    assert relaxed.cost < result.cost


def test_strict_reduction_with_no_room_inside_a_lane_has_no_legal_layout():
    with pytest.raises(NoLegalLayoutError, match="strict"):
        search_min_movement_layout(
            _paper_problem(("group", "vr.lane"), local_bits=0, strict=True)
        )


def test_ties_convert_to_linear_layouts_through_the_bridge():
    result = search_min_movement_layout(_paper_problem(y_weight=4))
    assert result.default_in_dims() == (("group.group", 4), ("local", 4))
    layouts = result.to_linear_layouts(out_dims=(("k", 4), ("m", 4)))
    assert len(layouts) == len(result.ties)
    expected = LinearLayout(
        {"group.group": [(0, 1), (0, 2)], "local": [(1, 0), (2, 0)]},
        ("k", "m"),
        (4, 4),
    )
    assert expected in layouts
    for layout, matrix in zip(layouts, result.ties):
        assert layout.to_gf2() == matrix
        group, slot = 1, 2  # lane 6
        k, m = layout.apply(**{"group.group": group, "local": slot})
        assert matrix.apply(group | (slot << 2)) == k | (m << 2)
    default = result.to_linear_layouts()
    assert default[0].out_dims == ("iteration",) and default[0].out_sizes == (16,)


def test_linear_layout_gf2_bridge_round_trips_and_validates_dims():
    layout = LinearLayout({"a": [(1, 0), (0, 1)], "b": [(3, 1)]}, ("u", "v"), (4, 2))
    matrix = layout.to_gf2()
    assert (matrix.n_rows, matrix.n_cols) == (3, 3)
    assert (
        LinearLayout.from_gf2(matrix, {"a": 4, "b": 2}, (("u", 4), ("v", 2))) == layout
    )
    for value in range(8):
        a, b = value & 3, value >> 2
        u, v = layout.apply(a=a, b=b)
        assert matrix.apply(value) == u | (v << 2)
    with pytest.raises(ValueError, match="in_dims cover"):
        LinearLayout.from_gf2(matrix, {"a": 4}, {"u": 4, "v": 2})
    with pytest.raises(ValueError, match="not a positive power of 2"):
        LinearLayout.from_gf2(matrix, {"a": 3, "b": 2}, {"u": 4, "v": 2})
    with pytest.raises(ValueError, match="unique"):
        LinearLayout.from_gf2(matrix, (("a", 4), ("a", 2)), {"u": 4, "v": 2})


# --------------------------------------------------------------------------- #
# Scalable flag search
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "bits,levels", ((2, 1), (2, 2), (3, 1), (3, 2), (3, 3), (4, 1), (4, 2), (4, 3))
)
def test_scalable_search_agrees_with_the_bounded_search(bits, levels):
    problem = _gemv_problem(bits, levels)
    bounded = search_min_movement_layout(problem)
    flags = search_scalable_layout_flags(problem)
    assert flags.optimality_proven
    assert flags.best_cost == bounded.cost
    assert {tie.flag.spaces for tie in flags.ties} == {
        flag.spaces for flag in bounded.tie_flags
    }
    assert len(flags.ties) * flags.layouts_per_flag == len(bounded.ties)
    assert flags.visited_flags + flags.pruned_flags + flags.unresolved_flags == (
        flags.total_flags
    )
    for ordinal in range(len(flags.ties)):
        representative = complete_scalable_layout_representative(flags, ordinal)
        assert representative.work_to_iteration in bounded.ties
        assert (
            representative.work_to_iteration @ representative.iteration_to_work
            == GF2Matrix.identity(bits)
        )


@pytest.mark.parametrize("bits", (5, 6))
def test_scalable_search_proves_one_level_optima_at_a_256_node_budget(bits):
    problem = _gemv_problem(bits, 1)
    result = search_min_movement_layout(
        problem, LayoutSearchPolicy(max_search_nodes=256)
    )
    assert result.implementation is LayoutSearchImplementation.SCALABLE
    assert result.optimality_proven
    assert not result.ties_complete
    assert result.tie_multiplicity == result.telemetry.layouts_per_flag
    exact = search_scalable_layout_flags(problem)
    assert result.cost == exact.best_cost
    assert len(result.ties) == len(exact.ties)
    for matrix in result.ties:
        assert (
            layout_movement_cost(problem.accesses, problem.hierarchy, matrix).total
            == result.cost
        )


def test_scalable_budget_exit_on_32_bits_returns_a_feasible_incumbent():
    problem = _gemv_problem(32, 3)
    result = search_min_movement_layout(
        problem, LayoutSearchPolicy(max_search_nodes=256)
    )
    assert result.implementation is LayoutSearchImplementation.SCALABLE
    assert result.termination is LayoutSearchTermination.NODE_BUDGET
    assert not result.optimality_proven
    assert result.telemetry.search_nodes_used <= 256
    assert result.remaining_lower_bounds
    assert min(result.remaining_lower_bounds) <= result.cost
    (matrix,) = result.ties
    assert (
        layout_movement_cost(problem.accesses, problem.hierarchy, matrix).total
        == result.cost
    )
    assert result.telemetry.unresolved_layouts > 0


def test_scalable_multilevel_budget_exit_reports_unfinished_search():
    problem = _gemv_problem(6, 2)
    result = search_min_movement_layout(
        problem, LayoutSearchPolicy(max_search_nodes=256)
    )
    assert result.termination is LayoutSearchTermination.NODE_BUDGET
    exact = search_scalable_layout_flags(problem)
    assert exact.optimality_proven
    assert result.cost >= exact.best_cost
    assert min(result.remaining_lower_bounds) <= exact.best_cost


# --------------------------------------------------------------------------- #
# Domains and policy
# --------------------------------------------------------------------------- #


def test_problems_outside_every_domain_are_rejected():
    with pytest.raises(LayoutSearchDomainError, match="bounded search covers"):
        search_min_movement_layout(
            _gemv_problem(5, 1), LayoutSearchPolicy(allow_scalable=False)
        )
    masked = LayoutSearchProblem(
        _gemv_problem(5, 1).hierarchy,
        _gemv_problem(5, 1).accesses,
        active_points=(0, 1),
    )
    with pytest.raises(LayoutSearchDomainError):
        search_min_movement_layout(masked)
    with pytest.raises(LayoutSearchDomainError, match="full carrier"):
        search_scalable_layout_flags(masked)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"max_carrier_bits": 5},
        {"max_carrier_bits": 0},
        {"max_levels": 4},
        {"max_search_nodes": 0},
        {"enable_pruning": 1},
    ),
)
def test_policy_rejects_values_outside_the_implemented_domain(kwargs):
    with pytest.raises((TypeError, ValueError)):
        LayoutSearchPolicy(**kwargs)


def test_problem_validation_is_explicit():
    problem = _gemv_problem(4, 2)
    with pytest.raises(ValueError, match="per-level costs"):
        LayoutSearchProblem(_hierarchy(1, local_bits=3), problem.accesses)
    with pytest.raises(ValueError, match="unique"):
        LayoutSearchProblem(problem.hierarchy, problem.accesses[:1] * 2)
    with pytest.raises(ValueError, match="nested"):
        LayoutSearchProblem(
            problem.hierarchy,
            problem.accesses,
            (GF2Subspace.zero(4), GF2Subspace(4, (1,))),
        )
    full = LayoutSearchProblem(
        problem.hierarchy, problem.accesses, active_points=tuple(range(16))
    )
    assert full.active_points is None and not full.is_masked
