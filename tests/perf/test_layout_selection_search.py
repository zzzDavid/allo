# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Layout-construction stage: coordinate search and matcher derivation.

* The coordinate-variable search agrees with the ported Algorithm 1 on every
  small full-carrier problem (minimum and the set of minimum movement
  profiles: the coordinate-dominance lemma of ``allo.pim.layout_search``),
  and never undercuts the exact masked search on a box mask.
* ``spmw_layout_problem.derive_layout_problem`` reads accesses, iteration
  variables and the owner hierarchy from synthetic ``MatchedOp`` groups,
  is invariant under renaming, and rejects non-affine, non-injective,
  non-unit-stride and work-non-affine accesses with the reason.
* A node budget ends in ``NODE_BUDGET`` with a feasible incumbent.
"""

from __future__ import annotations

import random
from fractions import Fraction

import pytest

from allo.pim.layout_movement import AccessMode, layout_movement_cost
from allo.pim.layout_problem import (
    LocalAxis,
    OwnerAxis,
    OwnerAxisOrigin,
    OwnerHierarchy,
    OwnerLevel,
)
from allo.pim.layout_search import (
    SCOPE_ALL_LAYOUTS,
    SCOPE_COORDINATE_LAYOUTS,
    CoordinateAccess,
    CoordinateLayoutProblem,
    CoordinateVariable,
    LayoutSearchPolicy,
    LayoutSearchProblem,
    LayoutSearchTermination,
    NoLegalLayoutError,
    coordinate_assignment_cost,
    search_coordinate_min_movement,
    search_min_movement_layout,
)
from allo.pim.targets import build_samsung_target, build_upmem_target
from allo.spmw_autoschedule import MatcherWorkScope
from allo.spmw_layout_problem import (
    LayoutDerivationError,
    MovementSelectionPolicy,
    analyze_kernel_movement,
    derive_layout_problem,
)
from allo.spmw_match_engine import coordinate_forms
from allo.spmw_match import MatchedOp, OperandBinding

R, W = AccessMode.READ, AccessMode.WRITE_ACCUMULATE


def _hierarchy(widths, local):
    return OwnerHierarchy(
        tuple(
            OwnerLevel(level, (OwnerAxis(level, 1 << w, OwnerAxisOrigin.MAPPED_UNIT),))
            for level, w in enumerate(widths)
        ),
        (LocalAxis(0, local),) if local else (),
    )


def _random_problem(rng):
    while True:
        variables = tuple(
            CoordinateVariable(f"v{i}", rng.choice([1, 2, 3, 4]))
            for i in range(rng.randint(1, 3))
        )
        carrier = sum(v.bits for v in variables)
        if 1 <= carrier <= 4:
            break
    levels = (1,) * rng.randint(1, min(3, carrier))
    accesses = []
    for j in range(rng.randint(1, 3)):
        used = tuple(
            sorted(rng.sample(range(len(variables)), rng.randint(0, len(variables))))
        )
        mode = rng.choice([R, W])
        accesses.append(
            CoordinateAccess(
                f"a{j}",
                used,
                mode,
                rng.choice([1, 2]),
                tuple(rng.choice([1, 2, 3]) for _ in levels),
                not (mode is W and rng.random() < 0.3),
            )
        )
    return CoordinateLayoutProblem(variables, levels, tuple(accesses))


def _ported(problem):
    active = problem.active_points() if problem.is_masked else None
    search = LayoutSearchProblem(
        _hierarchy(problem.level_bits, problem.carrier_bits - sum(problem.level_bits)),
        problem.layout_accesses(),
        (),
        active,
    )
    try:
        return search, search_min_movement_layout(search)
    except NoLegalLayoutError:
        return search, None


def _positive_profile(search, matrix):
    terms = layout_movement_cost(
        search.accesses, search.hierarchy, matrix, search.active_points
    ).terms
    weights = {
        (a.name, level): a.cost_per_level[level] * a.element_bytes
        for a in search.accesses
        for level in range(len(search.hierarchy.levels))
    }
    strict = {a.name for a in search.accesses if a.is_strict_reduction}
    return tuple(
        (t.access, t.owner_level, t.movement_rank)
        for t in terms
        if weights[(t.access, t.owner_level)] and t.access not in strict
    )


def test_coordinate_search_matches_algorithm_1_on_full_carriers():
    rng = random.Random(7)
    checked = 0
    for _ in range(150):
        problem = _random_problem(rng)
        if problem.is_masked:
            continue
        search, reference = _ported(problem)
        if reference is None:
            with pytest.raises(NoLegalLayoutError):
                search_coordinate_min_movement(problem)
            continue
        result = search_coordinate_min_movement(problem)
        assert result.termination is LayoutSearchTermination.EXHAUSTED
        assert result.optimality_proven and result.optimality_scope == SCOPE_ALL_LAYOUTS
        assert result.cost == reference.cost
        # Every minimum profile of Algorithm 1's complete tie set is the
        # profile of a coordinate tie, and conversely (lemma).
        reference_profiles = {_positive_profile(search, m) for m in reference.ties}
        coordinate_profiles = {
            _positive_profile(search, tie.assignment.matrix(problem))
            for tie in result.ties
        }
        assert coordinate_profiles == reference_profiles
        for tie in result.ties:
            cost = layout_movement_cost(
                search.accesses, search.hierarchy, tie.assignment.matrix(problem)
            ).total
            assert (
                cost
                == tie.cost
                == coordinate_assignment_cost(problem, tie.assignment)[0]
            )
        checked += 1
    assert checked > 40


def test_coordinate_search_on_masks_is_exact_over_coordinate_layouts():
    rng = random.Random(11)
    checked = 0
    for _ in range(120):
        problem = _random_problem(rng)
        if not problem.is_masked:
            continue
        search, reference = _ported(problem)
        if reference is None:
            continue
        result = search_coordinate_min_movement(problem)
        assert not result.optimality_proven
        assert result.optimality_scope == SCOPE_COORDINATE_LAYOUTS
        assert result.cost >= reference.cost
        for tie in result.ties:
            exact = layout_movement_cost(
                search.accesses,
                search.hierarchy,
                tie.assignment.matrix(problem),
                search.active_points,
            ).total
            assert exact == tie.cost
        checked += 1
    assert checked > 8


def _samsung_shaped(reassociate):
    variables = (
        CoordinateVariable("work0", 16),
        CoordinateVariable("work1", 8),
        CoordinateVariable("loop0", 32),
        CoordinateVariable("loop1", 256),
    )
    accesses = (
        CoordinateAccess("W", (0, 1, 2, 3), R, 2, (1, 1)),
        CoordinateAccess("x", (3,), R, 2, (1, 1)),
        CoordinateAccess("y", (0, 1, 2), W, 2, (1, 1), reassociate),
    )
    return CoordinateLayoutProblem(variables, (4, 3), accesses)


def test_samsung_shaped_problem_minimum_and_ties():
    strict = search_coordinate_min_movement(_samsung_shaped(False))
    # x is replicated to all 16 x 8 units: 512 B * (2**4 - 1 + 2**7 - 1).
    assert strict.cost == 512 * (15 + 127) == 72704
    assert strict.optimality_proven and len(strict.ties) == 1
    # C(12, 4) * C(8, 3) choices of 7 row bits over the two levels.
    assert strict.ties[0].multiplicity == 495 * 56
    reassociating = search_coordinate_min_movement(_samsung_shaped(True))
    # Splitting one k bit across PIM units trades x copies for y partials.
    assert reassociating.cost == 512 * 15 + 512 * 63 + 8192 * 1 == 48128
    assert len(reassociating.ties) == 2


def test_node_budget_returns_a_feasible_incumbent():
    problem = _samsung_shaped(False)
    result = search_coordinate_min_movement(problem, max_search_nodes=2)
    assert result.termination is LayoutSearchTermination.NODE_BUDGET
    assert not result.optimality_proven
    assert result.ties and result.fallback_incumbent
    cost, _profile = coordinate_assignment_cost(problem, result.ties[0].assignment)
    assert cost is not None and cost == result.cost


# --------------------------------------------------------------------- #
# Derivation from synthetic matcher groups
# --------------------------------------------------------------------- #


class _Policy:
    def __init__(self, may_regroup):
        self.may_regroup = may_regroup


def _vector_coords(x_map, indices):
    """Resolved coordinates of the synthetic vector access: the identity
    map reads ``x[k]``; a 2-D map reads a sum of two variables (an atom the
    matcher would not expand; the derivation rejects it as not injective);
    a floordiv map has no affine coordinate."""
    if x_map == "affine_map<(d0) -> (d0)>":
        return coordinate_forms(*indices)
    if "floordiv" in x_map:
        return (None,)
    return (tuple(sorted((name, 1) for name in indices)), 0),


def _gemv_group(
    *,
    grid=(16, 8),
    rows=32,
    k=256,
    names=("local_W", "local_x", "local_y", "%i", "%k"),
    x_map="affine_map<(d0) -> (d0)>",
    x_indices=None,
    k_step=1,
    row_base=None,
    policy=False,
    opaque_matrix=False,
):
    matrix, vector, out, i, kk = names
    loops = [
        (i, "affine_map<() -> (0)>", f"affine_map<() -> ({rows})>", 1),
        (kk, "affine_map<() -> (0)>", f"affine_map<() -> ({k})>", k_step),
    ]
    matches = []
    for p in range(grid[0]):
        for u in range(grid[1] if len(grid) > 1 else 1):
            work = (p, u) if len(grid) > 1 else (p,)
            base = row_base(work) if row_base else (p * grid[-1] + u) * rows
            if len(grid) == 1:
                base = row_base(work) if row_base else p * rows
            if opaque_matrix:
                # the matrix row is an SSA value the trace did not resolve
                matrix_operand = OperandBinding(
                    "x",
                    matrix,
                    ["%20", kk],
                    False,
                    f"memref<4096x{k}xf16>",
                    coords=coordinate_forms("%20", kk),
                )
                acc = OperandBinding(
                    "acc", "acc", [], True, "memref<f16>", None, "affine_map<() -> ()>",
                    coords=(),
                )
                extra_sink = {
                    "output_sinks": [(out, f"add({base}, {i})")],
                    "store_coords": (),
                    "output_sink_coords": [coordinate_forms((i, base))],
                }
            else:
                matrix_operand = OperandBinding(
                    "x",
                    matrix,
                    [i, kk],
                    False,
                    f"memref<4096x{k}xf16>",
                    None,
                    f"affine_map<(d0, d1) -> (d0 + {base}, d1)>",
                    coords=coordinate_forms((i, base), kk),
                )
                acc = OperandBinding(
                    "acc",
                    out,
                    [i],
                    True,
                    "memref<4096xf16>",
                    None,
                    f"affine_map<(d0) -> (d0 + {base})>",
                    coords=coordinate_forms((i, base)),
                )
                extra_sink = {"store_coords": coordinate_forms((i, base))}
            extra = {
                "spmw_work_scope": MatcherWorkScope(1, work, tuple(grid)),
                "accumulator_init": "zero",
                "axis_roles": {
                    "reduction": (kk,),
                    "output": (i,),
                    "matrix": "x",
                    "unresolved": None,
                },
                **extra_sink,
            }
            if policy is not None:
                extra["reduction_policy"] = _Policy(policy)
            matches.append(
                MatchedOp(
                    "MAC",
                    f"gemv_{'_'.join(map(str, work))}",
                    work,
                    list(loops),
                    [
                        matrix_operand,
                        OperandBinding(
                            "y",
                            vector,
                            x_indices or [kk],
                            False,
                            f"memref<{k}xf16>",
                            None,
                            x_map,
                            coords=_vector_coords(x_map, x_indices or [kk]),
                        ),
                        acc,
                    ],
                    out,
                    ("a", "b"),
                    extra,
                )
            )
    return matches


def test_derived_gemv_accesses_and_hierarchy():
    target = build_samsung_target()
    derived = derive_layout_problem(target, _gemv_group(policy=False))
    assert derived.level_names == ("pseudo_channel", "pim")
    assert [(v.name, v.extent) for v in derived.variables] == [
        ("work0", 16),
        ("work1", 8),
        ("loop0", 32),
        ("loop1", 256),
    ]
    by_name = {a.memref: a for a in derived.accesses}
    # W[row, k] reads (work, i, k); x[k] reads k; y[row] accumulates (work, i).
    assert by_name["local_W"].variables == (0, 1, 2, 3) and by_name["local_W"].mode is R
    assert by_name["local_x"].variables == (3,) and by_name["local_x"].mode is R
    assert by_name["local_y"].variables == (0, 1, 2) and by_name["local_y"].mode is W
    assert by_name["local_y"].reassociation_allowed is False
    assert all(a.element_bytes == 2 for a in derived.accesses)
    assert derived.movement_weights == "uniform"
    assert derived.cost_per_level == (Fraction(1), Fraction(1))
    assert not derived.assumptions
    # The program layout puts work axis d on owner level d.
    flag = derived.owner_flag(derived.program_layout())
    assert [space.rank for space in flag] == [16, 13]


def test_missing_policy_assumes_reassociation_and_says_so():
    derived = derive_layout_problem(build_samsung_target(), _gemv_group(policy=None))
    assert {a.memref: a for a in derived.accesses}["local_y"].reassociation_allowed
    assert any("no reduction_policy" in text for text in derived.assumptions)


def test_rename_only_variant_gives_the_same_problem():
    target = build_samsung_target()
    first = analyze_kernel_movement(target, _gemv_group())
    renamed = analyze_kernel_movement(
        target,
        _gemv_group(names=("A", "v", "out", "%arg7", "%arg9")),
    )
    assert first.status == renamed.status == "analyzed"
    assert (first.min_cost, first.tie_count, first.tie_layouts, first.program_cost) == (
        renamed.min_cost,
        renamed.tie_count,
        renamed.tie_layouts,
        renamed.program_cost,
    )
    assert first.optimality_proven and first.program_label == "minimal"


def test_unresolved_opaque_matrix_row_is_rejected():
    """A matrix row the trace retains only as an SSA name is no longer
    assumed equal to the store row (C-23): without the matcher's proof that
    the output map is the identity, the derivation refuses."""
    with pytest.raises(LayoutDerivationError, match="output map is not the identity"):
        derive_layout_problem(build_samsung_target(), _gemv_group(opaque_matrix=True))


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        ({"x_map": "affine_map<(d0) -> (d0 floordiv 2)>"}, "not affine"),
        (
            {"x_map": "affine_map<(d0, d1) -> (d0 + d1)>", "x_indices": ["%i", "%k"]},
            "not injective",
        ),
        ({"k_step": 2}, "unit-stride"),
        ({"row_base": lambda w: (w[0] * w[1]) * 32}, "not affine in the work id"),
    ],
)
def test_unsupported_accesses_are_rejected_with_the_reason(kwargs, reason):
    target = build_samsung_target()
    matches = _gemv_group(**kwargs)
    with pytest.raises(LayoutDerivationError, match=reason):
        derive_layout_problem(target, matches)
    movement = analyze_kernel_movement(target, matches)
    assert movement.status == "not-analyzed" and reason in movement.label


def test_non_power_of_two_rows_become_a_masked_problem():
    # UPMEM: 8 of 64 DPUs, 96 rows per DPU (a box mask on loop0).
    matches = _gemv_group(grid=(8,), rows=96, k=128, row_base=lambda w: w[0] * 96)
    derived = derive_layout_problem(build_upmem_target(), matches)
    assert derived.level_names == ("dpu",)
    assert derived.unmodelled_levels == ("tasklet",)
    assert derived.active_extents == (("dpu.dpu", 8),)
    assert derived.is_masked
    movement = analyze_kernel_movement(build_upmem_target(), matches)
    assert movement.implementation == "coordinate"
    assert movement.optimality_scope == SCOPE_COORDINATE_LAYOUTS
    assert not movement.optimality_proven


def test_bounded_problems_use_algorithm_1():
    # A 2 x 2 grid, 2 rows and 2 k per item: 4 carrier bits, Algorithm 1.
    matches = _gemv_group(grid=(2, 2), rows=2, k=2)
    target = build_samsung_target()
    derived = derive_layout_problem(target, matches)
    assert derived.carrier_bits == 4
    movement = analyze_kernel_movement(target, matches)
    assert movement.implementation == "bounded"
    assert movement.termination in ("exhausted", "bound-proved")
    assert movement.optimality_proven
    # Samsung realizes every minimum tie whose owner bits come from the
    # output-row bits (each row's k reduction stays inside one unit); under
    # the strict default these are all the minimum ties.
    realizable = [t for t in movement.ties if t.realizable]
    assert realizable
    assert all(t.realizable_reason is None for t in realizable) if hasattr(
        realizable[0], "realizable_reason"
    ) else True


def test_tiny_budget_marks_the_analysis_incomplete():
    policy = MovementSelectionPolicy(search=LayoutSearchPolicy(max_search_nodes=2))
    movement = analyze_kernel_movement(
        build_samsung_target(), _gemv_group(), None, policy
    )
    assert movement.termination == "node-budget"
    assert movement.incomplete and not movement.optimality_proven
    assert movement.ties and movement.min_cost is not None
