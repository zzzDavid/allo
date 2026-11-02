# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Masked minimum-movement search (paper Sections V-B and VI).

The 121,968-layout oracle is ported from
``reviews/experiments/irregular_audit.py`` (``masked_case``) and
``layout_fixture.py``: twelve GEMV cases (3 and 4 bits, 1 to 3 one-bit owner
levels, rectangular and triangular masks) enumerate every layout of the
bounded domain and check each one with an integer-parity oracle (bijection,
exact active coverage, poisoned padding, modular partial sums per owner,
every owner term, the cost), then require the masked search to return the
same minimum and complete tie set.  The minima and tie counts are pinned to
the snapshot run recorded in
``reviews/experiments/results/2026-09-29-irregular-audit/*/summary.json``.
"""

# pylint: disable=missing-function-docstring,too-many-locals

from fractions import Fraction

import pytest

from allo.pim.gf2 import GF2Matrix
from allo.pim.layout_movement import AccessMode, LayoutAccess, layout_movement_cost
from allo.pim.layout_problem import (
    LocalAxis,
    OwnerAxis,
    OwnerAxisOrigin,
    OwnerHierarchy,
    OwnerLevel,
)
from allo.pim.layout_search import (
    LayoutSearchImplementation,
    LayoutSearchPolicy,
    LayoutSearchProblem,
    LayoutSearchTermination,
    NoLegalLayoutError,
    enumerate_layouts,
    search_min_movement_layout,
)


def _fixture(bits, levels, active=None, *, strict=False):
    """layout_fixture.fixture: m on the low bits; x[k] read, y[m] partials."""

    m_bits = bits // 2
    hierarchy = OwnerHierarchy(
        tuple(
            OwnerLevel(level, (OwnerAxis(level, 2, OwnerAxisOrigin.MAPPED_UNIT),))
            for level in range(levels)
        ),
        (LocalAxis(0, bits - levels),) if bits > levels else (),
    )
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
        reassociation_allowed=not strict,
    )
    return LayoutSearchProblem(hierarchy, (x, y), active_points=active)


def _active(bits, rows, columns, triangular):
    m_bits = bits // 2
    return tuple(
        sorted(
            m | (k << m_bits)
            for m in range(rows)
            for k in range(columns)
            if not triangular or k <= m
        )
    )


# --------------------------------------------------------------------------- #
# Integer-parity oracle (irregular_audit.py)
# --------------------------------------------------------------------------- #


def _parity_map(rows, point):
    return sum(((row & point).bit_count() % 2) << bit for bit, row in enumerate(rows))


def _owner_cost(inverse, active, m_bits, levels):
    terms = {}
    for level in range(levels):
        for operand in (0, 1):
            groups = {}
            for point in active:
                key = point >> m_bits if operand == 0 else point % (1 << m_bits)
                groups.setdefault(key, set()).add(inverse[point] % (1 << (level + 1)))
            terms[operand, level] = sum(len(owners) - 1 for owners in groups.values())
    return terms, sum(v * (2 if op == 0 else 8) for (op, _), v in terms.items())


def _masked_case(bits, levels, rows, columns, triangular):
    m_bits = bits // 2
    active = _active(bits, rows, columns, triangular)
    active_set = set(active)
    problem = _fixture(bits, levels, active)
    expected = [
        sum(
            ((m * 7 + k * 3) % 11 - 5) * ((k * 5) % 7 - 3)
            for k in range(columns)
            if not triangular or k <= m
        )
        % 65536
        for m in range(rows)
    ]
    ordinal = {"x": 0, "y": 1}
    minimum, minimum_keys, layouts, differences = None, [], 0, 0
    seen = set()
    for layout in enumerate_layouts(problem):
        layouts += 1
        matrix = layout.work_to_iteration.rows
        seen.add(matrix)
        mapping = [_parity_map(matrix, work) for work in range(1 << bits)]
        assert sorted(mapping) == list(range(1 << bits))
        inverse = {point: work for work, point in enumerate(mapping)}
        active_work = tuple(w for w, point in enumerate(mapping) if point in active_set)
        assert sorted(mapping[w] for w in active_work) == list(active)
        # Pack through the layout; padded coordinates hold a poison sentinel
        # and must never be consumed by the owner-partial reduction.
        packed = [None] * (1 << bits)
        for point in active:
            m, k = point % (1 << m_bits), point >> m_bits
            packed[inverse[point]] = ((m * 7 + k * 3) % 11 - 5, (k * 5) % 7 - 3)
        partials = {}
        for work in active_work:
            m = mapping[work] % (1 << m_bits)
            owner = work % (1 << levels)
            a, x = packed[work]
            partials[m, owner] = (partials.get((m, owner), 0) + a * x) % 65536
        actual = [
            sum(value for (row, _), value in partials.items() if row == m) % 65536
            for m in range(rows)
        ]
        assert actual == expected
        assert sum(item is None for item in packed) == (1 << bits) - len(active)

        direct_terms, direct = _owner_cost(inverse, active, m_bits, levels)
        assert {
            (ordinal[term.access], term.owner_level): term.extra_owner_incidence
            for term in layout.terms
        } == direct_terms
        assert Fraction(direct) == layout.cost
        _, full = _owner_cost(inverse, range(1 << bits), m_bits, levels)
        differences += full != direct
        if minimum is None or direct < minimum:
            minimum, minimum_keys = direct, [matrix]
        elif direct == minimum:
            minimum_keys.append(matrix)
    assert len(seen) == layouts  # every enumerated layout is distinct

    search = search_min_movement_layout(problem)
    assert search.implementation is LayoutSearchImplementation.BOUNDED_MASKED
    assert search.termination is LayoutSearchTermination.EXHAUSTED
    assert search.optimality_proven
    assert search.cost == minimum
    assert {matrix.rows for matrix in search.ties} == set(minimum_keys)
    assert len(search.ties) == len(minimum_keys)
    assert search.pruned == 0 and search.telemetry.pruned_layouts == 0
    return layouts, minimum, len(minimum_keys), differences


# (bits, levels, rows, columns, triangular) -> (layouts, minimum, ties)
SNAPSHOT_RECORD = {
    (3, 1, 1, 3, False): (168, 0, 24),
    (3, 2, 1, 3, False): (168, 8, 24),
    (3, 3, 1, 3, False): (168, 24, 24),
    (3, 1, 2, 3, False): (168, 6, 24),
    (3, 2, 2, 3, False): (168, 28, 24),
    (3, 3, 2, 3, False): (168, 66, 24),
    (4, 1, 3, 3, False): (20_160, 6, 4_032),
    (4, 2, 3, 3, False): (20_160, 18, 576),
    (4, 3, 3, 3, False): (20_160, 54, 576),
    (4, 1, 3, 3, True): (20_160, 2, 1_344),
    (4, 2, 3, 3, True): (20_160, 8, 192),
    (4, 3, 3, 3, True): (20_160, 22, 64),
}


def test_121968_masked_layouts_match_the_integer_parity_oracle():
    total = 0
    for case, record in SNAPSHOT_RECORD.items():
        layouts, minimum, ties, differences = _masked_case(*case)
        assert (layouts, minimum, ties) == record, case
        # The padded full-carrier objective differs from the exact masked
        # one on every layout of these cases except the 3-bit, m=1, k=3,
        # one-level case, where 72 of 168 layouts agree.
        expected_differences = 96 if case == (3, 1, 2, 3, False) else layouts
        assert differences == expected_differences, case
        total += layouts
    assert total == 121_968


# --------------------------------------------------------------------------- #
# Masked contract
# --------------------------------------------------------------------------- #


def test_three_active_units_need_two_extra_copies():
    hierarchy = OwnerHierarchy(
        (OwnerLevel(0, (OwnerAxis(0, 4, OwnerAxisOrigin.MAPPED_UNIT),)),),
    )
    scalar = LayoutAccess("s", GF2Matrix(0, 2, ()), AccessMode.READ, 1, (1,))
    full = search_min_movement_layout(LayoutSearchProblem(hierarchy, (scalar,)))
    masked = search_min_movement_layout(
        LayoutSearchProblem(hierarchy, (scalar,), active_points=(0, 1, 2))
    )
    assert full.cost == 3  # padding: four units, three extra copies
    assert masked.cost == 2  # three active units, two extra copies
    assert len(masked.ties) == 6 == masked.visited


def test_padding_overstates_cost_and_admits_non_optimal_ties():
    active = _active(4, 3, 3, True)
    masked = search_min_movement_layout(_fixture(4, 3, active))
    padded = search_min_movement_layout(_fixture(4, 3))
    assert (masked.cost, len(masked.ties)) == (22, 64)
    assert (padded.cost, len(padded.ties)) == (88, 576)
    problem = _fixture(4, 3, active)
    real = {
        layout_movement_cost(problem.accesses, problem.hierarchy, matrix, active).total
        for matrix in padded.ties
    }
    # Every masked optimum is a padded tie, but a cost program ranking the
    # padded ties could pick a layout that really costs 32.
    assert set(masked.ties) <= set(padded.ties)
    assert min(real) == 22 and max(real) == 32


def test_masked_search_never_prunes_and_is_exact_only_on_exhaustion():
    active = _active(4, 3, 3, False)
    problem = _fixture(4, 2, active)
    result = search_min_movement_layout(
        problem, LayoutSearchPolicy(max_search_nodes=200)
    )
    assert result.termination is LayoutSearchTermination.NODE_BUDGET
    assert not result.optimality_proven
    assert result.pruned == 0
    # One flag of 192 layouts fits in 200 nodes; zero is the only bound
    # claimed for the other 104 flags because the rank bound is not valid
    # under a mask.
    assert result.telemetry.layouts_per_flag == 192
    assert result.telemetry.visited_flags == 1
    assert result.visited == 192
    assert result.remaining_lower_bounds == (Fraction(0),) * (105 - 1)
    assert result.ties
    for matrix in result.ties:
        assert (
            layout_movement_cost(
                problem.accesses, problem.hierarchy, matrix, active
            ).total
            == result.cost
        )
    exact = search_min_movement_layout(problem)
    assert exact.optimality_proven and exact.cost <= result.cost

    tiny = search_min_movement_layout(problem, LayoutSearchPolicy(max_search_nodes=10))
    assert tiny.telemetry.fallback_incumbent and len(tiny.ties) == 1
    assert tiny.visited == 0 and not tiny.optimality_proven


def test_masked_strict_reduction_is_checked_per_layout():
    # 3x2 mask: every y[m] reduces over k in {0, 1}, so only the k0
    # direction of its kernel span(k0, k1) is active.
    active = _active(4, 3, 2, False)
    problem = _fixture(4, 1, active, strict=True)
    result = search_min_movement_layout(problem)
    legal = [item for item in enumerate_layouts(problem) if item.legal]
    assert result.telemetry.illegal_layouts == 20_160 - len(legal)
    best = min(item.cost for item in legal)
    assert result.cost == best
    assert {m.rows for m in result.ties} == {
        item.work_to_iteration.rows for item in legal if item.cost == best
    }
    for matrix in result.ties:
        assert layout_movement_cost(
            problem.accesses, problem.hierarchy, matrix, active
        ).legal
    # The mask admits layouts that split k1, which the full-carrier rule
    # (kernel inside the innermost local space) would forbid.
    full_rule = sum(
        1
        for item in legal
        if problem.accesses[1].kernel.is_subspace_of(item.flag.spaces[-1])
    )
    assert len(legal) > full_rule


def test_masked_strict_reduction_with_no_legal_layout_raises():
    hierarchy = OwnerHierarchy(
        (OwnerLevel(0, (OwnerAxis(0, 4, OwnerAxisOrigin.MAPPED_UNIT),)),),
    )
    strict = LayoutAccess(
        "y",
        GF2Matrix(0, 2, ()),
        AccessMode.WRITE_ACCUMULATE,
        1,
        (1,),
        reassociation_allowed=False,
    )
    with pytest.raises(NoLegalLayoutError):
        search_min_movement_layout(
            LayoutSearchProblem(hierarchy, (strict,), active_points=(0, 1, 2))
        )
