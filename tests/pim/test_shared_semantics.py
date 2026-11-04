# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared semantic program contract and front stages (release gate G9, slice 1-2).

Compile-only plus the APU v1 functional backend; no board, card or simulator.
Checks that (a) the shared facts are derived from the matcher's structural
analysis, (b) both APU planners consume those facts instead of their own
source analyses, (c) the shared semantic checks run before any APU plan is
selected and reject for the same reason on every target, and (d) a program
the old APU v1 text analysis did not accept (local accumulator copied to the
output) now compiles and executes through the shared path.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import allo
from allo.ir.types import float16, int4, int15, uint16
from allo.pim.apu_g2_source_route import analyze_apu_g2_source
from allo.pim.apu_v1_vectorize import IllegalContractionError
from allo.pim.costs import aim_cost, apu_g2_cost, apu_v1_cost, samsung_cost
from allo.pim.targets import (
    build_aim_target,
    build_apu_g2_target,
    build_apu_v1_target,
    build_apu_v2_target,
    build_samsung_target,
)
from allo.spmw_match import MatchTrace, UnsupportedWorkloadError
from allo.spmw_candidate import CandidateValueLayout
from allo.spmw_reduction_policy import ReductionOrderError
from dataclasses import replace
from allo.spmw_pipeline import matcher_view, normalize_source
from allo.spmw_semantics import ProgramFacts, analyze_program

M, K, N = 8, 16, 4


def gemm(left: float16[M, K], right: float16[K, N], result: float16[M, N]):
    for row, column in allo.grid(M, N):
        for depth in allo.reduction(K):
            result[row, column] += left[row, depth] * right[depth, column]


def gemm_renamed(aa: float16[M, K], bb: float16[K, N], cc: float16[M, N]):
    for p, q in allo.grid(M, N):
        for s in allo.reduction(K):
            cc[p, q] += aa[p, s] * bb[s, q]


def gemm_local_acc(left: float16[M, K], right: float16[K, N], result: float16[M, N]):
    for row, column in allo.grid(M, N):
        acc: float16 = 0.0
        for depth in allo.reduction(K):
            acc += left[row, depth] * right[depth, column]
        result[row, column] = acc


def u16_gemm(left: uint16[M, K], right: uint16[K, N], result: uint16[M, N]):
    for row, column in allo.grid(M, N):
        for depth in allo.reduction(K):
            result[row, column] += left[row, depth] * right[depth, column]


def guarded_rows(L: float16[M, K], R: float16[K, N], C: float16[M, N]):
    for r, col in allo.grid(M, N):
        if r < 4:
            for d in allo.reduction(K):
                C[r, col] += L[r, d] * R[d, col]


def half_rows(L: float16[M, K], R: float16[K, N], C: float16[M, N]):
    for r, col in allo.grid(M // 2, N):
        for d in allo.reduction(K):
            C[r, col] += L[r, d] * R[d, col]


def g2_gemm(A: int4[64, 128], B: int4[128, 31], C: int15[64, 31]):
    for i in range(64):
        for j in range(31):
            acc: int15 = 0
            for k in range(128):
                acc += A[i, k] * B[k, j]
            C[i, j] = acc


# --------------------------------------------------------------------------- #
# (a) The facts come from the matcher's structural analysis
# --------------------------------------------------------------------------- #


def test_facts_are_derived_from_the_match_trace():
    target = build_apu_v1_target()
    program = normalize_source(gemm, target)
    facts = program.program_facts()
    assert isinstance(facts, ProgramFacts)
    assert isinstance(program.trace, MatchTrace) and program.trace.coverage is not None
    assert [p.name for p in facts.parameters] == ["left", "right", "result"]
    assert [p.shape for p in facts.parameters] == [(M, K), (K, N), (M, N)]
    assert {p.element.name for p in facts.parameters} == {"float16"}

    (c,) = facts.contractions
    assert c.match in program.trace.matches
    # Source loop names are retained on the match and used as labels.
    assert c.match.extra["loop_names"] == {loop.var: loop.name for loop in c.loops}
    assert [loop.label for loop in c.loops] == ["row", "column", "depth"]
    assert [loop.trip for loop in c.loops] == [M, N, K]
    labels = c.loop_labels
    assert [labels[v] for v in c.reduction_vars] == ["depth"]
    # The matcher's roles: ``column`` indexes the vector operand ``right``,
    # so it is a batch axis; ``row`` is the shared-vector output axis.
    assert [labels[v] for v in c.output_vars] == ["row"]
    assert [labels[v] for v in c.batch_vars] == ["column"]
    assert [labels[v] for v in c.result_vars] == ["row", "column"]
    assert [a.parameter for a in c.inputs] == ["left", "right"]
    assert [labels[d] for d in c.inputs[0].dims] == ["row", "depth"]
    assert [labels[d] for d in c.inputs[1].dims] == ["depth", "column"]
    assert c.accumulator.parameter == "result" and not c.accumulator.local
    assert c.accumulator_init == "carried" and c.sinks == ()
    assert (c.product_operation, c.accumulate_operation) == (
        "arith.mulf",
        "arith.addf",
    )
    assert c.numeric.text == "f16" and c.reduction_policy.order == "strict"
    assert facts.epilogues == () and facts.other_matches == ()


def test_signedness_and_local_accumulator_facts():
    target = build_apu_v1_target()
    unsigned = normalize_source(u16_gemm, target).program_facts()
    assert {p.element.name for p in unsigned.parameters} == {"uint16"}
    (c,) = unsigned.contractions
    assert c.numeric.source_text == "ui16"
    assert (c.product_operation, c.accumulate_operation) == (
        "arith.muli",
        "arith.addi",
    )

    local = normalize_source(gemm_local_acc, target).program_facts()
    (c,) = local.contractions
    assert c.accumulator.local and c.accumulator.shape == ()
    assert c.accumulator_init == "zero"
    (sink,) = c.sinks
    assert sink.parameter == "result" and not sink.local
    assert [c.loop_labels[d] for d in sink.dims] == ["row", "column"]
    assert [c.loop_labels[v] for v in c.result_vars] == ["row", "column"]


def test_renamed_program_has_the_same_structural_facts():
    target = build_apu_v1_target()
    a = normalize_source(gemm, target).program_facts().manifest()
    b = normalize_source(gemm_renamed, target).program_facts().manifest()
    assert a == b


# --------------------------------------------------------------------------- #
# (b) Both APU planners consume the shared facts
# --------------------------------------------------------------------------- #


def test_apu_v1_planner_consumes_the_shared_facts():
    compiled = allo.compile(gemm, build_apu_v1_target(), apu_v1_cost, backend="virtual")
    facts = compiled.program_facts
    assert isinstance(facts, ProgramFacts)
    assert isinstance(compiled.trace, MatchTrace)
    assert compiled.trace.coverage is not None  # the matcher ran
    (c,) = facts.contractions
    assert compiled.analysis.function == facts.function
    assert compiled.analysis.output_axes == ("row", "column")
    assert compiled.analysis.reduction_axis == "depth"
    assert compiled.analysis.multiply_operation == c.product_operation
    assert compiled.analysis.output_initializer == c.accumulator_init
    renamed = allo.compile(
        gemm_renamed, build_apu_v1_target(), apu_v1_cost, backend="virtual"
    )
    assert renamed.selected_plan.name == compiled.selected_plan.name
    assert renamed.estimate().cycles == compiled.estimate().cycles


def test_apu_g2_planner_consumes_the_shared_facts():
    target = build_apu_g2_target()
    # The hardware target declares a source-idiom view for the matcher.
    assert matcher_view(target) is not target
    assert matcher_view(target).name == "apu_v2"
    program = normalize_source(g2_gemm, target)
    facts = program.program_facts()
    contract = analyze_apu_g2_source(facts)
    compiled = allo.compile(g2_gemm, target, apu_g2_cost)
    assert compiled.contract.manifest() == contract.manifest()
    (stream,) = contract.streams
    assert stream.reduction_extent == 128 and stream.result_extents == (64, 31)
    assert stream.accumulator == "zero" and stream.output == "C"
    # facts from the pattern view are the facts the planner used
    assert (
        analyze_program(matcher_view(target), program.module, compiled.trace).manifest()
        == facts.manifest()
    )


# --------------------------------------------------------------------------- #
# (c) Shared semantic checks run before any APU plan, same reason everywhere
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "build, cost",
    [
        (build_apu_v1_target, apu_v1_cost),
        (build_samsung_target, samsung_cost),
        (build_aim_target, aim_cost),
    ],
    ids=["apu_v1", "samsung", "aim"],
)
def test_guarded_contraction_is_rejected_by_the_shared_stage(build, cost):
    """``if r < 4:`` around the reduction: the old APU v1 text analysis
    planned the whole 8-row problem (ISSUES V1-17); the shared coverage
    check now refuses the guarded store for every target."""
    with pytest.raises(UnsupportedWorkloadError, match="under a guard") as error:
        allo.compile(guarded_rows, build(), cost, backend="virtual")
    assert "matches no target op" in str(error.value)


def test_apu_v1_rejects_partial_output_coverage_explicitly():
    """A loop nest over half the output rows is a legal source program the
    vector plans cannot realize (they transfer whole buffers); it is refused
    at compile time with the restriction named."""
    with pytest.raises(IllegalContractionError, match="must equal the buffer extents"):
        allo.compile(half_rows, build_apu_v1_target(), apu_v1_cost, backend="virtual")


# --------------------------------------------------------------------------- #
# (d) A nearby supported program the old route did not accept
# --------------------------------------------------------------------------- #


def test_local_accumulator_gemm_executes_on_apu_v1_functional_backend():
    compiled = allo.compile(
        gemm_local_acc,
        build_apu_v1_target(),
        apu_v1_cost,
        backend="functional",
    )
    assert compiled.analysis.output_initializer == "zero"
    assert compiled.analysis.output.value == "result"
    rng = np.random.default_rng(7)
    left = (rng.standard_normal((M, K)) * 0.3).astype(np.float16)
    right = (rng.standard_normal((K, N)) * 0.3).astype(np.float16)
    result = np.full((M, N), 5, np.float16)  # overwritten, never accumulated
    run = compiled(left, right, result)
    assert run.kind == "functional"
    reference = np.zeros((M, N), np.float16)
    for d in range(K):
        reference = (reference + (left[:, d : d + 1] * right[d : d + 1, :])).astype(
            np.float16
        )
    np.testing.assert_array_equal(result.view(np.uint16), reference.view(np.uint16))
    # The equivalent carried form selects the same plan family.
    carried = allo.compile(
        gemm, build_apu_v1_target(), apu_v1_cost, backend="functional"
    )
    assert carried.selected_plan.name == compiled.selected_plan.name


# --------------------------------------------------------------------------- #
# (e) Function boundary: the entry ABI and call multiplicity (review round 12)
# --------------------------------------------------------------------------- #


def square_leaf(L: float16[4, 4], R: float16[4, 4], C: float16[4, 4]):
    for r, c in allo.grid(4, 4):
        for d in range(4):
            C[r, c] += L[r, d] * R[d, c]


def square_helper(R: float16[4, 4], L: float16[4, 4], C: float16[4, 4]):
    square_leaf(L, R, C)


def twice(L: float16[4, 4], R: float16[4, 4], C: float16[4, 4]):
    square_leaf(L, R, C)
    square_leaf(L, R, C)


def int_square_leaf(L: int4[4, 4], R: int4[4, 4], C: int15[4, 4]):
    for r, c in allo.grid(4, 4):
        acc: int15 = 0
        for d in range(4):
            acc += L[r, d] * R[d, c]
        C[r, c] = acc


def int_square_helper(R: int4[4, 4], L: int4[4, 4], C: int15[4, 4]):
    int_square_leaf(L, R, C)


@pytest.mark.parametrize(
    "workload, build, cost, calls",
    [
        (square_helper, build_apu_v1_target, apu_v1_cost, "calls 'square_leaf'"),
        (twice, build_apu_v1_target, apu_v1_cost, "calls 'square_leaf' 2 times"),
        (
            int_square_helper,
            build_apu_g2_target,
            apu_g2_cost,
            "calls 'int_square_leaf'",
        ),
        (square_helper, build_samsung_target, samsung_cost, "calls 'square_leaf'"),
    ],
    ids=["apu_v1_reordered", "apu_v1_twice", "apu_g2_reordered", "samsung_reordered"],
)
def test_plain_helper_calls_are_refused_at_the_shared_boundary(
    workload, build, cost, calls
):
    """A helper called with reordered arguments (the reviewer's probe: the
    planned program computed R @ L instead of L @ R) or called twice (one
    contraction planned for two) is refused before any planner, for every
    target, naming the call; the entry ABI is never replaced by a helper's."""
    with pytest.raises(UnsupportedWorkloadError, match=calls) as error:
        allo.compile(workload, build(), cost, backend="virtual")
    assert "Inline the helper" in str(error.value)


def test_both_gemini_ii_builders_compile_the_same_source_program():
    for build in (build_apu_g2_target, build_apu_v2_target):
        compiled = allo.compile(g2_gemm, build(), apu_g2_cost, backend="virtual")
        assert compiled.plan.name == "single/k1"
        assert [p.name for p in compiled.ranking] == ["single/k1", "single/k2"]


# --------------------------------------------------------------------------- #
# (f) Slice 3: both APUs' plans are candidates of the common selection stage
# --------------------------------------------------------------------------- #


def test_apu_v1_plans_are_candidates_of_the_shared_ranking():
    compiled = allo.compile(
        gemm,
        build_apu_v1_target(),
        apu_v1_cost,
        reduction_order="reassociate",
        backend="virtual",
    )
    (kernel,) = compiled.placement_ranking.kernels
    scored = [mode for _position, _cycles, mode in kernel.scores]
    assert set(scored) == {
        "baseline_spatial_reduction",
        "temporal_svp",
        "temporal_dma_coalescing",
        "temporal_dma_coalescing_broadcast_friendly",
    }
    assert scored[0] == compiled.selected_plan.name == kernel.scores[0][2]
    # Every candidate carries its effects through the shared contract.
    for _position, placement in kernel.candidates:
        extra = placement.extra
        assert extra["candidate_plan"] == placement.mode
        effects = extra["effects"]
        assert effects.coverage.outputs == ("result",)
        assert effects.coverage.inputs == ("left", "right", "result")
        assert effects.value_layouts and effects.transfers and effects.launches
        assert effects.abi_outputs == ("result",)
        assert extra["whole_device_launch"]
    # The selected program's graph is the one the estimate reports.
    assert compiled.execution_graph.metadata["plan"] == compiled.selected_plan.name
    assert compiled.estimate().cycles == kernel.scores[0][1]
    # A forced alternative reaches the generated GVML source.
    forced = allo.compile(
        gemm,
        build_apu_v1_target(),
        apu_v1_cost,
        reduction_order="reassociate",
        backend="virtual",
        layout="temporal_svp",
    )
    assert forced.selected_plan.name == "temporal_svp"
    assert forced.plan_selection == "explicit_layout"
    assert forced.device_source() != compiled.device_source()
    assert forced.compiled.cmds != compiled.compiled.cmds


def test_apu_g2_plans_are_candidates_of_the_shared_ranking():
    compiled = allo.compile(g2_gemm, build_apu_g2_target(), apu_g2_cost)
    (kernel,) = compiled.placement_ranking.kernels
    assert [mode for _p, _c, mode in kernel.scores] == ["single/k1", "single/k2"]
    assert compiled.plan.name == "single/k1" and compiled.selection == "cost"
    for _position, placement in kernel.candidates:
        extra = placement.extra
        assert extra["candidate_plan"] == placement.mode
        effects = extra["effects"]
        assert effects.launches and effects.abi_outputs == ("C",)
        assert effects.coverage.inputs == ("A", "B")
    forced = allo.compile(
        g2_gemm, build_apu_g2_target(), apu_g2_cost, layout="single/k2"
    )
    assert forced.plan.name == "single/k2" and forced.selection == "forced"
    # The forced alternative is a different launch program (two chained
    # launches), visible in the runtime route and the command stream.
    assert len(forced.plan.launches) == 2 and len(compiled.plan.launches) == 1
    route = lambda c: c.compiled.layout_ctx.runtime_route(c.compiled.cmds)
    assert route(forced) != route(compiled)
    assert forced.compiled.cmds != compiled.compiled.cmds
    assert forced.estimate().cycles == kernel.scores[1][1]


# --------------------------------------------------------------------------- #
# (g) Candidate boundary: supplied placements, policy and orchestration
#     (review round 13)
# --------------------------------------------------------------------------- #


def gemm_t(left: float16[4, 4], right: float16[4, 4], result: float16[4, 4]):
    for i, j in allo.grid(4, 4):
        for k in range(4):
            result[i, j] += left[k, i] * right[k, j]


def gemm_n(left: float16[4, 4], right: float16[4, 4], result: float16[4, 4]):
    for i, j in allo.grid(4, 4):
        for k in range(4):
            result[i, j] += left[i, k] * right[k, j]


def ig_n(L: int4[4, 4], R: int4[4, 4], C: int15[4, 4]):
    for i, j in allo.grid(4, 4):
        acc: int15 = 0
        for k in range(4):
            acc += L[i, k] * R[k, j]
        C[i, j] = acc


def ig_t(L: int4[4, 4], R: int4[4, 4], C: int15[4, 4]):
    for i, j in allo.grid(4, 4):
        acc: int15 = 0
        for k in range(4):
            acc += L[k, i] * R[k, j]
        C[i, j] = acc


def test_supplied_placement_is_rebound_to_the_current_program_apu_v1():
    """The reviewer's probe: a placement selected for ``L @ R`` supplied to
    the compilation of ``L.T @ R`` is a decision, not an executable; it is
    rebound to the transposed program's own candidate and the functional
    run computes the transposed product."""
    target = build_apu_v1_target()
    first = allo.compile(
        gemm_n,
        target,
        apu_v1_cost,
        backend="functional",
        reduction_order="reassociate",
        layout="baseline_spatial_reduction",
    )
    rebound = allo.compile(
        gemm_t,
        target,
        apu_v1_cost,
        backend="functional",
        reduction_order="reassociate",
        layout=first.compiled.layout,
    )
    assert rebound.analysis.function == "gemm_t"
    assert rebound.analysis.lhs.indices == ("k", "i")
    assert rebound.selected_plan.name == "baseline_spatial_reduction"
    assert rebound.plan_selection == "explicit_layout"
    L = (np.arange(16, dtype=np.float16).reshape(4, 4) % 4).astype(np.float16)
    R = np.eye(4, dtype=np.float16)
    out = np.zeros((4, 4), np.float16)
    rebound(L, R, out)
    np.testing.assert_array_equal(out, L.T @ R)


def test_supplied_placement_is_rebound_to_the_current_program_apu_g2():
    target = build_apu_g2_target()
    first = allo.compile(ig_n, target, apu_g2_cost, backend="virtual")
    rebound = allo.compile(
        ig_t, target, apu_g2_cost, backend="virtual", layout=first.compiled.layout
    )
    fresh = allo.compile(ig_t, target, apu_g2_cost, backend="virtual")
    assert rebound.contract.function == "ig_t"
    assert rebound.contract.manifest() == fresh.contract.manifest()
    assert rebound.plan.name == fresh.plan.name
    L = (np.arange(16, dtype=np.int8).reshape(4, 4) % 4).astype(np.int8)
    R = np.eye(4, dtype=np.int8)
    C = np.zeros((4, 4), np.int16)
    packed = rebound.pack(L=L, R=R, C=C)[0]["lhs"]
    np.testing.assert_array_equal(packed, fresh.pack(L=L, R=R, C=C)[0]["lhs"])


def test_supplied_regrouping_placement_is_refused_under_strict_policy():
    target = build_apu_v1_target()
    regroup = allo.compile(
        gemm_n,
        target,
        apu_v1_cost,
        backend="functional",
        reduction_order="reassociate",
        layout="baseline_spatial_reduction",
    )
    with pytest.raises(ReductionOrderError, match="baseline_spatial_reduction"):
        allo.compile(
            gemm_n,
            target,
            apu_v1_cost,
            backend="functional",
            reduction_order="strict",
            layout=regroup.compiled.layout,
        )
    # A strict-legal plan supplied as a placement is still accepted.
    strict = allo.compile(gemm_n, target, apu_v1_cost, backend="functional")
    again = allo.compile(
        gemm_n,
        target,
        apu_v1_cost,
        backend="functional",
        layout=strict.compiled.layout,
    )
    assert again.selected_plan.name == strict.selected_plan.name
    assert not again.compiled.layout.extra["regroups_reduction"]


@allo.host_program(gemm_n)
def launch_gemm_twice(left, right, result):
    allo.launch("gemm_n", left, right, result)
    allo.launch("gemm_n", left, right, result)


@allo.host_program(gemm_n)
def launch_gemm_once(left, right, result):
    allo.launch("gemm_n", left, right, result)


@allo.host_program(ig_n)
def launch_ig_twice(L, R, C):
    allo.launch("ig_n", L, R, C)
    allo.launch("ig_n", L, R, C)


def test_unsupported_host_programs_are_refused_by_the_apu_backends():
    """A host program that launches the kernel twice is neither priced as
    one launch nor executed once: both APU backends refuse it before
    selection, naming the capability; a single launch and an empty host
    move list remain valid."""
    with pytest.raises(NotImplementedError, match="launch"):
        allo.compile(
            gemm_n,
            build_apu_v1_target(),
            apu_v1_cost,
            backend="functional",
            host_moves=launch_gemm_twice,
        )
    with pytest.raises(NotImplementedError, match="launch"):
        allo.compile(
            ig_n,
            build_apu_g2_target(),
            apu_g2_cost,
            backend="virtual",
            host_moves=launch_ig_twice,
        )
    once = allo.compile(
        gemm_n,
        build_apu_v1_target(),
        apu_v1_cost,
        backend="functional",
        host_moves=launch_gemm_once,
    )
    assert len(once.compiled.launch_schedule.steps) == 1
    empty = allo.compile(
        ig_n, build_apu_g2_target(), apu_g2_cost, backend="virtual", host_moves=[]
    )
    assert empty.plan.name == "single/k1"


def test_candidate_effects_are_validated_against_the_program(monkeypatch):
    """A candidate whose declared coverage does not describe the program is
    rejected by the shared driver, not priced."""
    from allo.pim import apu_v1_vector_program as vector
    from allo.spmw_candidate import CandidateCoverage

    original = vector.apu_v1_vector_candidates

    def wrong_coverage(target, matches):
        placements = original(target, matches)
        from dataclasses import replace

        for placement in placements:
            effects = placement.extra["effects"]
            placement.extra["effects"] = replace(
                effects,
                coverage=CandidateCoverage(
                    effects.coverage.function,
                    effects.coverage.operations,
                    effects.coverage.inputs,
                    ("elsewhere",),
                ),
            )
        return placements

    monkeypatch.setattr(vector, "apu_v1_vector_candidates", wrong_coverage)
    with pytest.raises(ValueError, match="violates the candidate contract"):
        allo.compile(gemm_n, build_apu_v1_target(), apu_v1_cost, backend="virtual")


# --------------------------------------------------------------------------- #
# (h) Launch bindings and the enforced candidate contract (review round 14)
# --------------------------------------------------------------------------- #


@allo.host_program(gemm_n)
def launch_swapped(left, right, result):
    allo.launch("gemm_n", right, left, result)


@allo.host_program(gemm_n)
def launch_without_operands(left, right, result):
    allo.launch("gemm_n")


@allo.host_program(ig_n)
def launch_ig_swapped(L, R, C):
    allo.launch("ig_n", R, L, C)


def test_launch_operands_are_validated_against_the_kernel_abi():
    """The reviewer's probe: ``allo.launch('fold_blocks', B, A, Z)`` was
    accepted and executed ``A @ B``; a launch without operands executed too.
    Operands are now retained on the schedule and checked against the
    kernel's declared buffers, for every target, before selection."""
    with pytest.raises(ValueError, match="rebind its ABI"):
        allo.compile(
            gemm_n,
            build_apu_v1_target(),
            apu_v1_cost,
            backend="functional",
            host_moves=launch_swapped,
        )
    with pytest.raises(ValueError, match="0 operand"):
        allo.compile(
            gemm_n,
            build_apu_v1_target(),
            apu_v1_cost,
            backend="functional",
            host_moves=launch_without_operands,
        )
    with pytest.raises(ValueError, match="rebind its ABI"):
        allo.compile(
            ig_n,
            build_apu_g2_target(),
            apu_g2_cost,
            backend="virtual",
            host_moves=launch_ig_swapped,
        )
    once = allo.compile(
        gemm_n,
        build_apu_v1_target(),
        apu_v1_cost,
        backend="functional",
        host_moves=launch_gemm_once,
    )
    (step,) = once.compiled.launch_schedule.steps
    assert step.operands == ("left", "right", "result")


def _corrupted(original, mutate):
    def enumerate_changed(target, matches):
        placements = original(target, matches)
        for placement in placements:
            placement.extra["effects"] = mutate(placement.extra["effects"])
        return placements

    return enumerate_changed


@pytest.mark.parametrize(
    "mutation, reason",
    [
        (lambda e: replace(e, coverage=replace(e.coverage, inputs=())), "reads"),
        (lambda e: replace(e, transfers=()), "never transferred"),
        (
            lambda e: replace(
                e,
                value_layouts=(
                    CandidateValueLayout(e.abi_outputs[0], ("nonexistent_axis",), "vr"),
                )
                + e.value_layouts[1:],
            ),
            "not a loop of the program",
        ),
        (
            lambda e: replace(
                e,
                value_layouts=tuple(
                    replace(v, storage="not_a_storage") for v in e.value_layouts
                ),
            ),
            "does not declare",
        ),
        (
            lambda e: replace(
                e,
                launches=tuple(
                    replace(l, index=-7, kind="nothing", instances=(0, 1))
                    for l in e.launches
                ),
            ),
            "out of order",
        ),
        (
            lambda e: replace(
                e,
                launches=tuple(replace(l, reduction=(0, 2)) for l in e.launches),
            ),
            "not an ordered partition",
        ),
    ],
    ids=["inputs", "transfers", "axis", "storage", "launch_index", "reduction"],
)
def test_candidate_effects_fault_injection_is_rejected(monkeypatch, mutation, reason):
    """Every field of the candidate contract is checked against the program
    and the target (the reviewer's fault injection): a corrupted candidate
    is rejected before costing on both APU backends."""
    from allo.pim import apu_v1_vector_program as vector
    from allo.pim import apu_g2_source_route as route

    monkeypatch.setattr(
        vector,
        "apu_v1_vector_candidates",
        _corrupted(vector.apu_v1_vector_candidates, mutation),
    )
    with pytest.raises(ValueError, match="violates the candidate contract") as error:
        allo.compile(gemm_n, build_apu_v1_target(), apu_v1_cost, backend="virtual")
    assert reason in str(error.value)
    monkeypatch.setattr(
        route,
        "apu_g2_source_candidates",
        _corrupted(route.apu_g2_source_candidates, mutation),
    )
    with pytest.raises(ValueError, match="violates the candidate contract") as error:
        allo.compile(ig_n, build_apu_g2_target(), apu_g2_cost, backend="virtual")
    assert reason in str(error.value)


def test_cost_graph_must_agree_with_the_declared_launches(monkeypatch):
    """The priced graph of a candidate must describe its declared launches:
    a graph that prices another plan or launch count is refused."""
    from allo.spmw_apu_g2 import APUG2SourceCtx

    original = APUG2SourceCtx.candidate_execution_graph

    def wrong_count(target, trace, layout, bound_cost, **options):
        graph = original(target, trace, layout, bound_cost, **options)
        if graph is not None:
            graph.metadata["launch_count"] = 5
        return graph

    monkeypatch.setattr(
        APUG2SourceCtx,
        "candidate_execution_graph",
        classmethod(lambda cls, *a, **k: wrong_count(*a, **k)),
    )
    with pytest.raises(ValueError, match="prices 5 launch"):
        allo.compile(ig_n, build_apu_g2_target(), apu_g2_cost, backend="virtual")


def test_declared_effects_match_the_realized_programs():
    """The selected candidates' effects are typed and complete on both APUs
    and the realization checks compared them with the artifact."""
    v1 = allo.compile(
        gemm_n,
        build_apu_v1_target(),
        apu_v1_cost,
        backend="virtual",
        reduction_order="reassociate",
    )
    effects = v1.compiled.layout.extra["effects"]
    # ``result`` is carried: its incoming contents are a read (round 20).
    assert effects.coverage.inputs == ("left", "right", "result")
    assert {v.value for v in effects.value_layouts} == {"left", "right", "result"}
    assert all(v.storage == "vr" for v in effects.value_layouts)
    assert {t.direction for t in effects.transfers} == {"in", "inout"}
    (launch,) = effects.launches
    assert launch.operations == (0,) and launch.instances == (0, 16)
    assert launch.reduction == (0, 4) and launch.covers[0] == v1.selected_plan.name
    g2 = allo.compile(ig_n, build_apu_g2_target(), apu_g2_cost, layout="single/k2")
    effects = g2.compiled.layout.extra["effects"]
    assert [l.reduction for l in effects.launches] == [(0, 2), (2, 4)]
    assert [l.accumulator for l in effects.launches] == ["none", "previous"]
    assert all(l.instances == (0, 16) for l in effects.launches)
    assert {v.storage for v in effects.value_layouts} == {"mmb"}


# --------------------------------------------------------------------------- #
# (i) Typed Gemini-II records as frontends of the shared pipeline (slice 4)
# --------------------------------------------------------------------------- #


def test_gemini_ii_records_enter_the_common_pipeline():
    from allo.pim.apu_g2_record_frontend import APUG2RecordProgram
    from allo.pim.apu_g2_typed_program import APUG2ScalarType, APUG2TypedProgram

    I4 = APUG2ScalarType(4, True)
    I15 = APUG2ScalarType(15, True)
    I14 = APUG2ScalarType(14, True)
    I3 = APUG2ScalarType(3, True)
    target = build_apu_g2_target()
    typed = APUG2TypedProgram("dot", (I4, I4), I15, (64, 128), reduction_extent=128)
    compiled = allo.compile(typed, target, apu_g2_cost, backend="virtual")
    assert isinstance(compiled, allo.CompiledCallable)
    # The record's semantics are in the shared contract and its candidate
    # passed legality, the candidate contract, costing and ranking.
    facts = compiled.trace.semantics
    assert facts.frontend is typed and facts.function == typed.name
    (contraction,) = facts.contractions
    assert [contraction.loop_labels[v] for v in contraction.reduction_vars] == ["k"]
    (kernel,) = compiled.placement_ranking.kernels
    assert [mode for _p, _c, mode in kernel.scores] == [typed.name]
    effects = compiled.compiled.layout.extra["effects"]
    assert effects.coverage.inputs == ("lhs", "rhs") and effects.abi_outputs == ("out",)
    (launch,) = effects.launches
    assert launch.instances == (0, 64) and launch.reduction == (0, 128)
    program = compiled.backend_program
    assert isinstance(program, APUG2RecordProgram) and program.record is typed
    assert compiled.estimate().cycles > 0
    assert tuple(inspect.signature(compiled).parameters) == ("lhs", "rhs", "out")
    # A family name selects the record's realization; a foreign plan is refused.
    forced = allo.compile(
        typed, target, apu_g2_cost, backend="virtual", layout="typed_vl64"
    )
    assert forced.backend_program.selection == "forced"
    with pytest.raises(ValueError, match="names no candidate"):
        allo.compile(typed, target, apu_g2_cost, backend="virtual", layout="single/k1")

    composed = allo.APUG2ComposedContractionProgram(
        (I3, I3), I15, (256, 2), 256, allo.APUG2DotEpilogue((1,), I14)
    )
    accumulate = allo.compile(composed, target, apu_g2_cost, backend="virtual")
    facts = accumulate.trace.semantics
    assert [m.target_op_name for m in facts.trace.matches] == ["MAC", "ADD"]
    effects = accumulate.compiled.layout.extra["effects"]
    assert set(effects.coverage.inputs) == {"lhs", "rhs", "accumulator"}
    assert tuple(inspect.signature(accumulate).parameters) == (
        "lhs",
        "rhs",
        "accumulator",
        "out",
    )
    run = accumulate(
        np.zeros((512, 256), np.int8),
        np.zeros((512, 256), np.int8),
        np.zeros((256, 2), np.int16),
    )
    assert run.kind == "virtual" and run.extra["outputs"] == {}


# --------------------------------------------------------------------------- #
# (j) APUv1Program records as frontends of the shared pipeline (slice 4)
# --------------------------------------------------------------------------- #


def scan_scalar(values: allo.ir.types.int32[64], out: allo.ir.types.int32[64]):
    total: allo.ir.types.int32 = 0
    for i in range(64):
        if values[i] > 3:
            total += values[i]
        out[i] = total


def dense_histogram_record(values: uint16[32768]) -> uint16[256]:
    counts: uint16[256] = 0
    for bin_index in range(256):
        for item in allo.reduction(32768):
            if values[item] == bin_index:
                counts[bin_index] += 1
    return counts


def test_apu_v1_program_records_enter_the_common_pipeline():
    """An ``APUv1Program`` (scalar ARC C or a native recipe) is a frontend:
    its effects are stated in the shared contract, its lowering is the one
    candidate validated, priced, ranked and materialized by the common
    pipeline, and the shared artifact runs it."""
    from allo.pim.apu_v1_record_frontend import APUv1RecordProgram

    target = build_apu_v1_target()
    scalar = allo.APUv1Program((allo.APUv1Phase(scan_scalar),), name="scan_scalar")
    compiled = allo.compile(scalar, target, apu_v1_cost, backend="functional")
    assert isinstance(compiled, allo.compiler.CompiledCallable)
    facts = compiled.trace.semantics
    assert facts.frontend is scalar and facts.verbatim
    # The shared front stages ran on the phase's own source: the coverage
    # retained every store (the local accumulator and the output), with its
    # loop nest, guards, reads and carried state.
    assert compiled.trace.coverage is not None
    assert [
        e.destination.parameter or e.destination.memref_name for e in facts.effects
    ] == ["total", "total", "out"]
    total_update, output_store = facts.effects[1], facts.effects[2]
    assert total_update.destination.local and total_update.guards
    assert {a.parameter for a in total_update.reads if not a.local} == {"values"}
    assert output_store.writes_whole_buffer and output_store.instances == 64
    assert output_store.loops[0].trip == 64 and not output_store.guards
    (anchor,) = compiled.trace.matches
    assert anchor.target_op_name == "ARC_SCALAR_PROGRAM"
    assert anchor.extra["whole_function"]
    effects = compiled.compiled.layout.extra["effects"]
    assert effects.coverage.inputs == ("values",) and effects.abi_outputs == ("out",)
    assert {(v.value, v.storage) for v in effects.value_layouts} == {
        ("values", "l4"),
        ("out", "l4"),
    }
    (launch,) = effects.launches
    assert launch.kind == "scalar_arc_c" and launch.whole
    assert launch.instances == (0, 1) and launch.operations == (0, 1, 2)
    assert launch.reduction is None and launch.covers[0] == "scalar_arc_c"
    assert effects.allocations == (("allo_scratch", 8, "l4"),)
    (kernel,) = compiled.placement_ranking.kernels
    assert [mode for _p, _c, mode in kernel.scores] == ["scan_scalar"]
    program = compiled.backend_program
    assert isinstance(program, APUv1RecordProgram) and program.record is scalar
    assert compiled.route == "scalar_arc_c" and compiled.native_vector_lowering is None
    assert "allo_scratch" in compiled.device_source
    assert tuple(inspect.signature(compiled).parameters) == ("values", "out")
    assert compiled.estimate().cycles > 0
    assert compiled.execution_graph.metadata["launch_count"] == 1
    values = np.arange(64, dtype=np.int32) % 7
    out = np.zeros(64, np.int32)
    run = compiled(values, out)
    assert run.kind == "functional" and run.cycles is None
    np.testing.assert_array_equal(
        out, np.cumsum(np.where(values > 3, values, 0)).astype(np.int32)
    )

    phase = allo.APUv1Phase(
        dense_histogram_record, result_names=("counts",), vectorize="required"
    )
    native = allo.APUv1Program((phase,), name="histogram_record")
    virtual = allo.compile(native, target, apu_v1_cost, backend="virtual")
    (anchor,) = virtual.trace.matches
    assert anchor.target_op_name == "NATIVE_VECTOR_PROGRAM"
    facts = virtual.trace.semantics
    # The returned buffer is a parameter; its effects are the declaration's
    # zero fill (whole buffer) and the guarded, carried count over the
    # 32768 items.
    fill, counting = facts.effects
    assert fill.category == "fill" and fill.writes_whole_buffer and fill.term == "0"
    assert counting.destination.parameter == "counts" and counting.external
    assert counting.reads_destination and counting.guards
    assert counting.instances == 256 and counting.reduction_extent == 32768
    assert [p.name for p in facts.parameters if p.returned] == ["counts"]
    effects = virtual.compiled.layout.extra["effects"]
    (launch,) = effects.launches
    assert launch.kind == "native_vector" and launch.whole
    assert launch.covers[0] == virtual.native_vector_lowering.route
    assert effects.coverage.outputs == ("counts",) and effects.abi_inputs == ("values",)
    assert virtual.execution_graph.metadata["estimate_validity"] == "calibrated"
    run = virtual(np.zeros(32768, np.uint16))
    assert run.kind == "virtual" and run.cycles == virtual.estimate().cycles > 0
    assert run.extra["native_vector_route"] == virtual.native_vector_lowering.route
    # The route's own name or family selects it; a foreign name is refused;
    # a native recipe has no host execution path.
    forced = allo.compile(
        native, target, apu_v1_cost, backend="virtual", layout="native_vector"
    )
    assert forced.backend_program.selection == "forced"
    with pytest.raises(ValueError, match="names no candidate"):
        allo.compile(
            native, target, apu_v1_cost, backend="virtual", layout="temporal_svp"
        )
    with pytest.raises(NotImplementedError, match="no host execution path"):
        allo.compile(native, target, apu_v1_cost, backend="functional")
    # The same shared contract checks a record candidate's declared effects.
    from allo.pim import apu_v1_record_frontend as record_frontend

    original = record_frontend.apu_v1_program_candidates

    def partial_launch(view, matches):
        placements = original(view, matches)
        for placement in placements:
            effects = placement.extra["effects"]
            placement.extra["effects"] = replace(
                effects,
                launches=tuple(
                    replace(l, whole=False, instances=(0, 64), operations=(2,))
                    for l in effects.launches
                ),
            )
        return placements

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(record_frontend, "apu_v1_program_candidates", partial_launch)
        with pytest.raises(ValueError, match="executes verbatim"):
            allo.compile(scalar, target, apu_v1_cost, backend="virtual")


# --------------------------------------------------------------------------- #
# (k) The candidate contract's launch chain, accumulator modes and extents
#     of every operation (review round 15, C-30 follow-up)
# --------------------------------------------------------------------------- #


def carry_grid(L: float16[3, 8], R: float16[8, 5], Z: float16[3, 5]):
    for rr, cc in allo.grid(3, 5):
        for kk in range(8):
            Z[rr, cc] += L[rr, kk] * R[kk, cc]


def offset_grid(L: int4[3, 8], R: int4[8, 5], Z: int15[3, 5]):
    for rr, cc in allo.grid(3, 5):
        tmp: int15 = 0
        for kk in range(8):
            tmp += L[rr, kk] * R[kk, cc]
        Z[rr, cc] = tmp


def test_repeated_reduction_chunk_is_rejected_before_selection(monkeypatch):
    """A plan whose launches sum a chunk twice (its effects describing those
    launches accurately) is refused by the contract: the reduction ranges of
    one instance range must be an ordered partition, not a set that tiles."""
    from allo.pim import apu_g2_source_route as route

    original = route.apu_g2_source_candidates

    def duplicated(target, matches):
        candidates = original(target, matches)
        for placement in candidates:
            if placement.mode != "single/k2":
                continue
            payload = placement.extra["payload"]
            first, second = payload.plan.launches
            payload.plan.launches = (first, replace(second, final=False), second)
            placement.extra["effects"] = payload.effects()
            placement.extra["candidate_decision"] = payload.decision()
            placement.extra["launch_count"] = 3
        return candidates

    monkeypatch.setattr(route, "apu_g2_source_candidates", duplicated)
    with pytest.raises(
        (ValueError, RuntimeError), match="ordered partition|cost stage rejected"
    ) as error:
        allo.compile(
            offset_grid, build_apu_g2_target(), apu_g2_cost, layout="single/k2"
        )
    assert "ordered partition" in str(error.value)


def test_declared_accumulator_mode_must_match_the_source_initialization(
    monkeypatch,
):
    """A carried source accumulator cannot be declared as zero-started (nor
    the reverse), and a continuation launch must start from the previous
    launch's partial sum."""
    from allo.pim import apu_v1_vector_program as vector
    from allo.pim import apu_g2_source_route as route

    original_vector = vector.apu_v1_vector_candidates

    def declared_zero(target, matches):
        candidates = original_vector(target, matches)
        for placement in candidates:
            effects = placement.extra["effects"]
            placement.extra["effects"] = replace(
                effects,
                launches=tuple(
                    replace(l, accumulator="zero") for l in effects.launches
                ),
            )
        return candidates

    monkeypatch.setattr(vector, "apu_v1_vector_candidates", declared_zero)
    with pytest.raises(ValueError, match="starts its accumulator as 'zero'") as error:
        allo.compile(
            carry_grid,
            build_apu_v1_target(),
            apu_v1_cost,
            backend="virtual",
            reduction_order="reassociate",
        )
    assert "the source initializes it as 'carried'" in str(error.value)

    original = route.apu_g2_source_candidates

    def continuation_carried(target, matches):
        candidates = original(target, matches)
        for placement in candidates:
            effects = placement.extra["effects"]
            placement.extra["effects"] = replace(
                effects,
                launches=tuple(
                    replace(l, accumulator="carried") if l.index else l
                    for l in effects.launches
                ),
            )
        return candidates

    monkeypatch.setattr(route, "apu_g2_source_candidates", continuation_carried)
    with pytest.raises(ValueError, match="not 'previous'"):
        allo.compile(
            offset_grid, build_apu_g2_target(), apu_g2_cost, layout="single/k2"
        )


def test_elementwise_record_launch_must_cover_every_result_instance(monkeypatch):
    """A typed elementwise record (no contraction) has an instance extent
    too: its loop trips.  A launch declaring one instance for a 64-element
    add is refused, and the realized record is compared with the declared
    effects field by field."""
    from allo.pim import apu_g2_record_frontend as records
    from allo.pim.apu_g2_typed_program import APUG2ScalarType, APUG2TypedProgram

    original = records.apu_g2_record_candidates

    def one_instance(*args):
        candidates = original(*args)
        for placement in candidates:
            effects = placement.extra["effects"]
            placement.extra["effects"] = replace(
                effects,
                launches=tuple(replace(l, instances=(0, 1)) for l in effects.launches),
            )
        return candidates

    monkeypatch.setattr(records, "apu_g2_record_candidates", one_instance)
    i4 = APUG2ScalarType(4, True)
    add = APUG2TypedProgram("add", (i4, i4), i4, (64,))
    with pytest.raises(ValueError, match="not a tiling of \\[0, 64\\)"):
        allo.compile(add, build_apu_g2_target(), apu_g2_cost, backend="virtual")
    monkeypatch.undo()

    # A declaration that passes the contract but is not the record's own
    # effects is caught when the realization is compared with it.
    from allo.spmw_apu_g2 import APUG2SourceCtx

    def swapped_route(*args):
        # A declarable route (the card's L1 is a registered step) that is
        # not the one the record's runtime takes for its output.
        candidates = original(*args)
        for placement in candidates:
            effects = placement.extra["effects"]
            placement.extra["effects"] = replace(
                effects,
                transfers=tuple(
                    replace(t, route=("host_pack", "l1")) if t.value == "out" else t
                    for t in effects.transfers
                ),
            )
        return candidates

    monkeypatch.setattr(records, "apu_g2_record_candidates", swapped_route)
    with pytest.raises(RuntimeError, match="differs from the candidate contract"):
        allo.compile(add, build_apu_g2_target(), apu_g2_cost, backend="virtual")
    assert APUG2SourceCtx.check_realization is not None


# --------------------------------------------------------------------------- #
# (l) Whole-function programs keep their source semantics (review round 16)
# --------------------------------------------------------------------------- #


def keep_unselected(src: allo.ir.types.int32[8], out: allo.ir.types.int32[8]):
    for i in range(8):
        if src[i] > 0:
            out[i] = src[i] + 1


def shift_right(src: allo.ir.types.int32[8], out: allo.ir.types.int32[8]):
    for i in range(7):
        out[i + 1] = src[i]


def accumulate_into(src: allo.ir.types.int32[8], out: allo.ir.types.int32[8]):
    for i in range(8):
        out[i] = out[i] + src[i]


def overwrite(src: allo.ir.types.int32[8], out: allo.ir.types.int32[8]):
    for i in range(8):
        out[i] = src[i] + 1


def _record(fn):
    return allo.APUv1Program((allo.APUv1Phase(fn),))


@pytest.mark.parametrize(
    "kernel, inputs, guarded, carried, whole",
    [
        (keep_unselected, ("src", "out"), True, False, False),
        (shift_right, ("src", "out"), False, False, False),
        (accumulate_into, ("src", "out"), False, True, True),
        (overwrite, ("src",), False, False, True),
    ],
    ids=["guarded", "shifted", "carried", "whole"],
)
def test_output_input_dependency_is_in_the_record_contract(
    kernel, inputs, guarded, carried, whole
):
    """A whole-function program's contract comes from the shared front
    stages run on its own source: a guarded or partial write, or a carried
    read, makes the output's initial contents an input of the program; an
    unconditional whole overwrite does not."""
    compiled = allo.compile(
        _record(kernel), build_apu_v1_target(), apu_v1_cost, backend="functional"
    )
    (effect,) = compiled.trace.semantics.effects
    assert bool(effect.guards) == guarded
    assert effect.reads_destination == carried
    assert effect.writes_whole_buffer == whole
    effects = compiled.compiled.layout.extra["effects"]
    assert effects.coverage.inputs == inputs and effects.abi_inputs == inputs
    assert [(t.value, t.direction) for t in effects.transfers] == [
        ("src", "in"),
        ("out", "inout"),
    ]
    src = np.arange(-4, 4, dtype=np.int32)
    results = []
    for sentinel in (11, 73):
        out = np.full(8, sentinel, np.int32)
        run = compiled(src, out)
        assert run.kind == "functional"
        results.append(out.copy())
    if "out" in inputs:
        assert np.any(results[0] != results[1])
    else:
        np.testing.assert_array_equal(results[0], results[1])
    if kernel is keep_unselected:
        np.testing.assert_array_equal(results[0], np.where(src > 0, src + 1, 11))


def test_record_contract_cannot_hide_a_read_or_a_transfer(monkeypatch):
    """Declaring the guarded output as write-only, or dropping the movement
    the runtime performs, is refused: by the shared validator against the
    retained facts, and by the realization check against the runtime."""
    from allo.pim import apu_v1_record_frontend as record_frontend

    original = record_frontend.apu_v1_program_candidates
    target = build_apu_v1_target()

    def output_only(view, matches):
        placements = original(view, matches)
        for placement in placements:
            effects = placement.extra["effects"]
            placement.extra["effects"] = replace(
                effects,
                coverage=replace(effects.coverage, inputs=("src",)),
                abi_inputs=("src",),
            )
        return placements

    monkeypatch.setattr(record_frontend, "apu_v1_program_candidates", output_only)
    with pytest.raises(ValueError, match="reads \\['src'\\], the program reads"):
        allo.compile(_record(keep_unselected), target, apu_v1_cost, backend="virtual")

    def no_scratch(view, matches):
        placements = original(view, matches)
        for placement in placements:
            effects = placement.extra["effects"]
            placement.extra["effects"] = replace(effects, allocations=())
        return placements

    monkeypatch.setattr(record_frontend, "apu_v1_program_candidates", no_scratch)
    with pytest.raises(RuntimeError, match="realization allocates"):
        allo.compile(_record(keep_unselected), target, apu_v1_cost, backend="virtual")

    def host_only_out(view, matches):
        placements = original(view, matches)
        for placement in placements:
            effects = placement.extra["effects"]
            placement.extra["effects"] = replace(
                effects,
                transfers=tuple(
                    replace(t, route=("host_readback",)) if t.value == "out" else t
                    for t in effects.transfers
                ),
            )
        return placements

    monkeypatch.setattr(record_frontend, "apu_v1_program_candidates", host_only_out)
    with pytest.raises(RuntimeError, match="realization moves"):
        allo.compile(_record(keep_unselected), target, apu_v1_cost, backend="virtual")


# --------------------------------------------------------------------------- #
# (m) Retained effects: indirect loads, fills, unused arguments; whole
#     launches only for verbatim programs (review round 17)
# --------------------------------------------------------------------------- #


def gather(
    src: allo.ir.types.int32[8],
    index: allo.ir.types.int32[8],
    out: allo.ir.types.int32[8],
):
    for i in range(8):
        out[i] = src[index[i]]


def indirect_guard(
    src: allo.ir.types.int32[8],
    index: allo.ir.types.int32[8],
    out: allo.ir.types.int32[8],
):
    for i in range(8):
        if src[index[i]] > 0:
            out[i] = 1


def fill_only(src: allo.ir.types.int32[8]) -> allo.ir.types.int32[8]:
    result: allo.ir.types.int32[8] = 7
    return result


def initialized_return(src: allo.ir.types.int32[8]) -> allo.ir.types.int32[8]:
    result: allo.ir.types.int32[8] = 7
    for i in range(4):
        result[i] = src[i]
    return result


def unused_argument(src: allo.ir.types.int32[8], out: allo.ir.types.int32[8]):
    for i in range(8):
        out[i] = 7


def _compile_record(fn, result_names=()):
    return allo.compile(
        allo.APUv1Program((allo.APUv1Phase(fn, result_names=result_names),)),
        build_apu_v1_target(),
        apu_v1_cost,
        backend="functional",
    )


def test_indirect_loads_are_retained_reads():
    """``src[index[i]]`` reads ``index`` as well as ``src``, in the stored
    value and in a guard; both are inputs, transfers and ABI inputs, and the
    program executes."""
    src = np.arange(-4, 4, dtype=np.int32)
    index = np.arange(7, -1, -1, dtype=np.int32)
    for kernel, expected, inputs in (
        # The unguarded gather overwrites ``out`` whole; the guarded write
        # keeps the untouched elements, so ``out`` is an input too.
        (gather, src[index], ("src", "index")),
        (indirect_guard, np.where(src[index] > 0, 1, 77), ("src", "index", "out")),
    ):
        compiled = _compile_record(kernel)
        (effect,) = compiled.trace.semantics.effects
        roles = {(a.parameter, a.role) for a in effect.reads}
        kind = "guard" if kernel is indirect_guard else "load"
        assert roles == {("src", kind), ("index", kind)}
        effects = compiled.compiled.layout.extra["effects"]
        assert effects.coverage.inputs == inputs
        assert effects.abi_inputs == inputs
        assert {(t.value, t.direction, t.physical) for t in effects.transfers} == {
            ("src", "in", False),
            ("index", "in", False),
            ("out", "inout", False),
        }
        out = np.full(8, 77, np.int32)
        run = compiled(src, index, out)
        assert run.kind == "functional"
        np.testing.assert_array_equal(out, expected)


def test_fill_initializers_are_retained_effects():
    """A declaration's fill is an observable effect of a returned buffer:
    it is retained with its destination, value and whole extent, ordered
    before the stores, and the realized program reproduces it."""
    compiled = _compile_record(fill_only, ("result",))
    (fill,) = compiled.trace.semantics.effects
    assert fill.category == "fill" and fill.whole and fill.constant
    assert fill.term == "7" and fill.destination.parameter == "result"
    assert fill.writes_whole_buffer and fill.instances == 8 and fill.external
    effects = compiled.compiled.layout.extra["effects"]
    assert effects.coverage.inputs == () and effects.abi_outputs == ("result",)
    (launch,) = effects.launches
    assert launch.whole and launch.operations == (0,)
    run = compiled(np.arange(8, dtype=np.int32))
    np.testing.assert_array_equal(run.extra["outputs"]["result"], np.full(8, 7))

    partial = _compile_record(initialized_return, ("result",))
    fill, store = partial.trace.semantics.effects
    assert fill.category == "fill" and fill.writes_whole_buffer
    assert store.category != "fill" and not store.writes_whole_buffer
    assert store.instances == 4 and [a.parameter for a in store.reads] == ["src"]
    effects = partial.compiled.layout.extra["effects"]
    assert effects.coverage.inputs == ("src",)
    assert effects.launches[0].operations == (0, 1)
    src = np.arange(-4, 4, dtype=np.int32)
    run = partial(src)
    np.testing.assert_array_equal(
        run.extra["outputs"]["result"], np.array([-4, -3, -2, -1, 7, 7, 7, 7])
    )


def test_unused_arguments_are_physical_transfers_not_inputs(monkeypatch):
    """The runtime copies every source argument; one the function never
    reads is a physical transfer of a declared parameter, not a semantic
    input, and declaring it as an input is refused."""
    compiled = _compile_record(unused_argument)
    effects = compiled.compiled.layout.extra["effects"]
    assert effects.coverage.inputs == () and effects.abi_inputs == ()
    assert {(t.value, t.direction, t.physical) for t in effects.transfers} == {
        ("src", "in", True),
        ("out", "inout", False),
    }
    out = np.full(8, 77, np.int32)
    compiled(np.arange(8, dtype=np.int32), out)
    np.testing.assert_array_equal(out, np.full(8, 7))

    from allo.pim import apu_v1_record_frontend as record_frontend

    original = record_frontend.apu_v1_program_candidates

    def pretend_input(view, matches):
        placements = original(view, matches)
        for placement in placements:
            effects = placement.extra["effects"]
            placement.extra["effects"] = replace(
                effects,
                coverage=replace(effects.coverage, inputs=("src",)),
                abi_inputs=("src",),
                transfers=tuple(replace(t, physical=False) for t in effects.transfers),
            )
        return placements

    monkeypatch.setattr(record_frontend, "apu_v1_program_candidates", pretend_input)
    with pytest.raises(ValueError, match="reads \\['src'\\], the program reads \\[\\]"):
        allo.compile(
            allo.APUv1Program((allo.APUv1Phase(unused_argument),)),
            build_apu_v1_target(),
            apu_v1_cost,
            backend="virtual",
        )


def test_whole_launches_are_reserved_for_verbatim_programs(monkeypatch):
    """A planned program (a typed Gemini-II add) cannot escape the instance
    and reduction tiling checks by declaring a whole-function launch."""
    from allo.pim import apu_g2_record_frontend as records
    from allo.pim.apu_g2_typed_program import APUG2ScalarType, APUG2TypedProgram
    from allo.spmw_candidate import validate_candidate_effects

    i4 = APUG2ScalarType(4, True)
    add = APUG2TypedProgram("add", (i4, i4), i4, (64,))
    target = build_apu_g2_target()
    compiled = allo.compile(add, target, apu_g2_cost, backend="virtual")
    placement = compiled.compiled.layout
    assert not compiled.trace.semantics.verbatim
    whole = replace(
        placement.extra["effects"],
        launches=tuple(
            replace(l, instances=(0, 1), reduction=None, accumulator=None, whole=True)
            for l in placement.extra["effects"].launches
        ),
    )
    placement.extra["effects"] = whole
    problems = validate_candidate_effects(
        compiled.trace.matches, placement, matcher_view(target)
    )
    assert any("declared whole-function" in p for p in problems)

    original = records.apu_g2_record_candidates

    def whole_launch(*args):
        placements = original(*args)
        for candidate in placements:
            effects = candidate.extra["effects"]
            candidate.extra["effects"] = replace(
                effects,
                launches=tuple(
                    replace(
                        l,
                        instances=(0, 1),
                        reduction=None,
                        accumulator=None,
                        whole=True,
                    )
                    for l in effects.launches
                ),
            )
        return placements

    monkeypatch.setattr(records, "apu_g2_record_candidates", whole_launch)
    with pytest.raises(ValueError, match="declared whole-function"):
        allo.compile(add, target, apu_g2_cost, backend="virtual")


def dependent_fill(
    src: allo.ir.types.int32[8], out: allo.ir.types.int32[8]
) -> allo.ir.types.int32[8]:
    for i in range(8):
        out[i] = src[i] + 2
    seed: allo.ir.types.int32 = out[0]
    result: allo.ir.types.int32[8] = seed
    return result


def test_retained_effects_keep_program_order_across_fills_and_stores():
    """A fill that reads a value an earlier store defined is retained after
    that store (one traversal in site order), so the effect set never places
    a read before its defining write."""
    compiled = _compile_record(dependent_fill, ("result",))
    src = np.arange(8, dtype=np.int32)
    out = np.full(8, 99, np.int32)
    run = compiled(src, out)
    np.testing.assert_array_equal(out, src + 2)
    np.testing.assert_array_equal(run.extra["outputs"]["result"], np.full(8, 2))
    effects = compiled.trace.semantics.effects
    order = [
        (e.destination.parameter or e.destination.memref_name, e.category)
        for e in effects
    ]
    assert order == [("out", "unmatched"), ("seed", "unmatched"), ("result", "fill")]
    assert [a.parameter for a in effects[1].reads] == ["out"]
    # The fill's value is the seed, which the matcher follows to ``out[0]``:
    # the fill depends on the store that defined it.
    assert "out" in {a.parameter for a in effects[2].reads}


# --------------------------------------------------------------------------- #
# (n) The first matcher backend on the shared contract: Samsung HBM-PIM
#     consumes the retained facts and declares its candidates' effects
# --------------------------------------------------------------------------- #

SAMSUNG_M, SAMSUNG_K, SAMSUNG_ROWS = 4096, 256, 32


@allo.dataflow.region()
def samsung_gemv(
    W: float16[SAMSUNG_M, SAMSUNG_K], x: float16[SAMSUNG_K], y: float16[SAMSUNG_M]
):
    @allo.work(mapping=[16, 8], args=[W, x, y])
    def body(
        local_W: float16[SAMSUNG_M, SAMSUNG_K],
        local_x: float16[SAMSUNG_K],
        local_y: float16[SAMSUNG_M],
    ):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * SAMSUNG_ROWS
        for i in range(SAMSUNG_ROWS):
            acc: float16 = 0
            for k in range(SAMSUNG_K):
                acc += local_W[row0 + i, k] * local_x[k]
            local_y[row0 + i] = acc


def test_samsung_candidates_declare_effects_from_the_shared_facts():
    """Every Samsung candidate of a program with shared facts declares its
    effects, derived from the work body's `Contraction` facts (source loop
    labels, trips, access coordinates, accumulator initialization, sinks)
    and the placement (handles, levers); the realized CRF program is
    compared with the selected declaration."""
    from allo.spmw_candidate import EFFECTS_REQUIRED_TARGETS, candidate_effects

    assert "samsung_hbm_pim" in EFFECTS_REQUIRED_TARGETS
    compiled = allo.compile(
        samsung_gemv, build_samsung_target(), samsung_cost, backend="virtual"
    )
    facts = compiled.trace.function_facts["body_0_0"]
    (contraction,) = facts.contractions
    assert [p.name for p in facts.parameters] == ["local_W", "local_x", "local_y"]
    assert contraction.loop_labels == {
        contraction.loop_vars[0]: "i",
        contraction.loop_vars[1]: "k",
    }
    (kernel,) = compiled.placement_ranking.kernels
    candidates = [
        entry[1] if isinstance(entry, tuple) else entry for entry in kernel.candidates
    ]
    assert candidates and all(candidate_effects(c) is not None for c in candidates)
    layout = compiled.compiled.layout
    layout = layout[0] if isinstance(layout, list) else layout
    effects = candidate_effects(layout)
    assert effects.coverage.inputs == ("local_W", "local_x")
    assert effects.coverage.outputs == ("local_y",)
    layouts = {v.value: (v.axes, v.storage) for v in effects.value_layouts}
    assert layouts["local_W"] == (("i", "k"), "banks")
    # The result's backing storage is the banks (its row image, identity
    # row slots); GRF_B is the transient accumulator register.
    assert layouts["local_y"] == (("i",), "banks")
    assert {s for _n, _b, s in effects.allocations} == {"grf_a", "grf_b"}
    assert layouts["local_x"][0] == ("k",) and layouts["local_x"][1] in {
        "grf_a",
        "banks",
    }
    (launch,) = effects.launches
    assert launch.kind in {"REDUCE", "TEMPORAL_REDUCE"}
    assert launch.instances == (0, SAMSUNG_ROWS) and launch.reduction == (0, SAMSUNG_K)
    assert launch.accumulator == "zero" and launch.covers == (layout.mode,)
    assert {t.value: t.direction for t in effects.transfers} == {
        "local_W": "in",
        "local_x": "in",
        "local_y": "out",
    }
    assert compiled.estimate().cycles > 0


@pytest.mark.parametrize(
    "mutation, reason",
    [
        (
            lambda e: replace(e, coverage=replace(e.coverage, inputs=("local_W",))),
            "reads",
        ),
        (
            lambda e: replace(
                e,
                launches=tuple(replace(l, reduction=(0, 128)) for l in e.launches),
            ),
            "not an ordered partition",
        ),
        (
            lambda e: replace(
                e,
                value_layouts=tuple(
                    (
                        replace(v, axes=("nonexistent_axis",))
                        if v.value == "local_W"
                        else v
                    )
                    for v in e.value_layouts
                ),
            ),
            "not a loop of the program",
        ),
        (lambda e: None, "declares no effects"),
    ],
    ids=["inputs", "reduction", "axis", "missing"],
)
def test_corrupted_samsung_declarations_are_refused_before_costing(
    monkeypatch, mutation, reason
):
    from allo import spmw_samsung

    original = spmw_samsung.samsung_candidate_effects

    def corrupted(target, matches, placement):
        effects = original(target, matches, placement)
        return None if effects is None else mutation(effects)

    monkeypatch.setattr(spmw_samsung, "samsung_candidate_effects", corrupted)
    with pytest.raises(ValueError, match="violates the candidate contract") as error:
        allo.compile(
            samsung_gemv, build_samsung_target(), samsung_cost, backend="virtual"
        )
    assert reason in str(error.value)


def test_samsung_selected_program_executes_under_the_declared_contract(
    samsung_sim_gate,
):
    """The selected candidate (declared, validated, realized and checked)
    runs on the Samsung simulator and reproduces the GEMV."""
    compiled = allo.compile(samsung_gemv, build_samsung_target(), samsung_cost)
    layout = compiled.compiled.layout
    layout = layout[0] if isinstance(layout, list) else layout
    assert layout.extra["effects"].launches[0].reduction == (0, SAMSUNG_K)
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((SAMSUNG_M, SAMSUNG_K)) * 0.1).astype(np.float16)
    x = (rng.standard_normal(SAMSUNG_K) * 0.1).astype(np.float16)
    y = np.zeros(SAMSUNG_M, np.float16)
    run = compiled(W, x, y)
    assert run.kind == "simulator" and run.cycles > 0
    np.testing.assert_allclose(
        y.astype(np.float32),
        W.astype(np.float32) @ x.astype(np.float32),
        rtol=1e-2,
        atol=1e-2,
    )


# --------------------------------------------------------------------------- #
# (o) Carried destinations are reads of the program (review round 19)
# --------------------------------------------------------------------------- #

CARRIED_M, CARRIED_K = 2048, 100


@allo.dataflow.region()
def samsung_carried(
    weights: float16[CARRIED_M, CARRIED_K],
    values: float16[CARRIED_K],
    accum: float16[CARRIED_M],
):
    @allo.work(mapping=[16, 8], args=[weights, values, accum])
    def worker(
        matrix: float16[CARRIED_M, CARRIED_K],
        vector: float16[CARRIED_K],
        destination: float16[CARRIED_M],
    ):
        pc, unit = allo.get_wid()
        first = (pc * 8 + unit) * 16
        for row in range(16):
            for depth in range(CARRIED_K):
                destination[first + row] += matrix[first + row, depth] * vector[depth]


def test_carried_destination_is_an_input_on_every_backend():
    """A destination whose incoming contents seed the reduction is a read:
    it is in the coverage inputs, the ABI inputs and moved in on Samsung,
    APU v1 and Gemini-II alike."""
    from allo.spmw_candidate import candidate_effects

    samsung = allo.compile(
        samsung_carried, build_samsung_target(), samsung_cost, backend="virtual"
    )
    facts = samsung.trace.function_facts["worker_0_0"]
    assert facts.contractions[0].accumulator_init == "carried"
    layout = samsung.compiled.layout
    layout = layout[0] if isinstance(layout, list) else layout
    effects = candidate_effects(layout)
    assert effects.coverage.inputs == ("matrix", "vector", "destination")
    assert effects.abi_inputs == effects.coverage.inputs
    (destination,) = [t for t in effects.transfers if t.value == "destination"]
    assert destination.direction == "inout"
    assert destination.route == ("host_scatter", "banks", "host_gather")

    apu = allo.compile(
        gemm_n,
        build_apu_v1_target(),
        apu_v1_cost,
        backend="virtual",
        reduction_order="reassociate",
    )
    effects = apu.compiled.layout.extra["effects"]
    assert effects.coverage.inputs == ("left", "right", "result")
    assert [(t.value, t.direction) for t in effects.transfers] == [
        ("left", "in"),
        ("right", "in"),
        ("result", "inout"),
    ]

    g2 = allo.compile(ig_carried, build_apu_g2_target(), apu_g2_cost, backend="virtual")
    effects = g2.compiled.layout.extra["effects"]
    assert effects.coverage.inputs == ("L", "R", "C")
    (carried,) = [t for t in effects.transfers if t.value == "C"]
    assert carried.direction == "inout" and carried.route[-1] == "device_readback"


def ig_carried(L: int4[4, 4], R: int4[4, 4], C: int15[4, 4]):
    for i, j in allo.grid(4, 4):
        for k in range(4):
            C[i, j] += L[i, k] * R[k, j]


def test_removing_the_carried_ingress_is_refused(monkeypatch):
    """Declaring a carried destination as write-only is refused by the shared
    validator against the facts and, were it to pass, by the realization
    check against the conductor's uploads."""
    from allo import spmw_samsung
    from allo.spmw_candidate import candidate_effects, validate_candidate_effects

    compiled = allo.compile(
        samsung_carried, build_samsung_target(), samsung_cost, backend="virtual"
    )
    layout = compiled.compiled.layout
    layout = layout[0] if isinstance(layout, list) else layout
    effects = candidate_effects(layout)
    matches = [m for m in compiled.trace.matches if m.func_name == "worker_0_0"]
    out_only = replace(
        effects,
        transfers=tuple(
            (
                replace(t, direction="out", route=("banks", "host_gather"))
                if t.value == "destination"
                else t
            )
            for t in effects.transfers
        ),
    )
    layout.extra["effects"] = out_only
    problems = validate_candidate_effects(matches, layout, build_samsung_target())
    assert any("never transferred in" in p for p in problems)
    with pytest.raises(RuntimeError, match="realization moves"):
        compiled.compiled.layout_ctx.check_realization([layout])

    original = spmw_samsung.samsung_candidate_effects

    def no_ingress(target, matches, placement):
        effects = original(target, matches, placement)
        if effects is None:
            return None
        return replace(
            effects,
            coverage=replace(
                effects.coverage,
                inputs=tuple(n for n in effects.coverage.inputs if n != "destination"),
            ),
            abi_inputs=tuple(n for n in effects.abi_inputs if n != "destination"),
            transfers=tuple(
                (
                    replace(t, direction="out", route=("banks", "host_gather"))
                    if t.value == "destination"
                    else t
                )
                for t in effects.transfers
            ),
        )

    monkeypatch.setattr(spmw_samsung, "samsung_candidate_effects", no_ingress)
    with pytest.raises(
        ValueError, match="reads \\['matrix', 'vector'\\], the program reads"
    ):
        allo.compile(
            samsung_carried, build_samsung_target(), samsung_cost, backend="virtual"
        )


# --------------------------------------------------------------------------- #
# (p) Samsung storage, packing and transfers are tied to the emitted program;
#     realization is checked at the materialization boundary
# --------------------------------------------------------------------------- #


def flat_mad_source(
    la: float16[4096], lb: float16[4096], ld: float16[4096], lc: float16[4096]
):
    for i in range(4096):
        lc[i] = la[i] * lb[i] + ld[i]


def test_samsung_elementwise_contract_describes_bank_images_and_fiber_packing():
    from allo.spmw_candidate import candidate_effects

    compiled = allo.compile(
        flat_mad_source, build_samsung_target(), samsung_cost, backend="virtual"
    )
    layout = compiled.compiled.layout
    layout = layout[0] if isinstance(layout, list) else layout
    effects = candidate_effects(layout)
    packing = {v.value: (v.storage, v.packing) for v in effects.value_layouts}
    assert packing == {
        "la": ("banks", "all_bank"),
        "lb": ("banks", "all_bank"),
        "ld": ("banks", "all_bank:swapped_fibers"),
        "lc": ("banks", "all_bank"),
    }
    program = compiled.compiled.layout_ctx.programs[0]
    assert program.kind == "ELTWISE" and program.swap_parity == (False, False, True)
    assert {s for _n, _b, s in effects.allocations} == {"grf_a", "grf_b"}
    assert {t.value: t.route for t in effects.transfers} == {
        "la": ("host_scatter", "banks"),
        "lb": ("host_scatter", "banks"),
        "ld": ("host_scatter", "banks"),
        "lc": ("banks", "host_gather"),
    }


@pytest.mark.parametrize(
    "mutation, reason",
    [
        (
            lambda e: replace(
                e,
                value_layouts=tuple(replace(v, storage="srf") for v in e.value_layouts),
            ),
            "realization",
        ),
        (
            lambda e: replace(
                e,
                transfers=tuple(
                    replace(t, route=("imaginary_route",)) for t in e.transfers
                ),
            ),
            "neither a declared storage nor a declared move",
        ),
        (
            lambda e: replace(
                e,
                value_layouts=tuple(
                    replace(v, packing="all_bank") if v.value == "ld" else v
                    for v in e.value_layouts
                ),
            ),
            "swapped_fibers",
        ),
        (
            lambda e: replace(e, allocations=(("tile", 0, "grf_a"),)),
            "uses registers",
        ),
    ],
    ids=["srf_storage", "imaginary_route", "packing", "registers"],
)
def test_valid_looking_but_wrong_samsung_declarations_are_refused(
    monkeypatch, mutation, reason
):
    """A declared storage that exists on the target but is not the one the
    program uses, a transfer path the target does not declare, a packing the
    emitter does not produce, or missing transient registers are refused:
    by the validator before costing, or by the realization check at
    materialization, which rejects the candidate during the search."""
    from allo import spmw_samsung

    original = spmw_samsung.samsung_candidate_effects

    def corrupted(target, matches, placement):
        effects = original(target, matches, placement)
        return None if effects is None else mutation(effects)

    monkeypatch.setattr(spmw_samsung, "samsung_candidate_effects", corrupted)
    with pytest.raises((ValueError, RuntimeError)) as error:
        allo.compile(
            flat_mad_source, build_samsung_target(), samsung_cost, backend="virtual"
        )
    assert reason in str(error.value)


def test_realization_mismatch_is_rejected_in_the_search_with_a_fallback(monkeypatch):
    """A candidate whose declaration passes the validator but not the
    realization check (the vector declared as a GRF_A broadcast while its
    program loads it from the banks) is rejected at materialization and the
    ranking selects another candidate."""
    from allo import spmw_samsung
    from allo.spmw_candidate import candidate_effects

    baseline = allo.compile(
        samsung_gemv,
        build_samsung_target(),
        samsung_cost,
        backend="virtual",
        reduction_order="reassociate",
    )
    (kernel,) = baseline.placement_ranking.kernels
    assert len(kernel.scores) > 1
    chosen = baseline.compiled.layout
    chosen = chosen[0] if isinstance(chosen, list) else chosen
    victim = chosen.mode  # the candidate the ranking selects when undisturbed
    original = spmw_samsung.samsung_candidate_effects

    def wrong_vector_storage(target, matches, placement):
        # Flip the vector's declared residency (and its transfer to match):
        # both forms are declarable, only one is what the program does.
        effects = original(target, matches, placement)
        if effects is None or placement.mode != victim:
            return effects
        (vector,) = [v for v in effects.value_layouts if v.value == "local_x"]
        flipped = "banks" if vector.storage == "grf_a" else "grf_a"
        route = (
            ("host_scatter", "banks")
            if flipped == "banks"
            else ("host_broadcast", "grf_a")
        )
        return replace(
            effects,
            value_layouts=tuple(
                replace(v, storage=flipped) if v.value == "local_x" else v
                for v in effects.value_layouts
            ),
            transfers=tuple(
                replace(t, route=route) if t.value == "local_x" else t
                for t in effects.transfers
            ),
        )

    monkeypatch.setattr(spmw_samsung, "samsung_candidate_effects", wrong_vector_storage)
    compiled = allo.compile(
        samsung_gemv,
        build_samsung_target(),
        samsung_cost,
        backend="virtual",
        reduction_order="reassociate",
    )
    layout = compiled.compiled.layout
    layout = layout[0] if isinstance(layout, list) else layout
    assert layout.mode != victim
    (kernel,) = compiled.placement_ranking.kernels
    reasons = [reason for _position, reason in kernel.rejections]
    assert any(
        "realization differs from the candidate contract" in str(reason)
        and "loads the vector from" in str(reason)
        for reason in reasons
    ), reasons
    assert candidate_effects(layout) is not None


# --------------------------------------------------------------------------- #
# (q) Distinct images of one buffer; exact vector residency (review round 20)
# --------------------------------------------------------------------------- #

COPYBACK_N = 4113


def update_addend(
    a: float16[COPYBACK_N], b: float16[COPYBACK_N], c: float16[COPYBACK_N]
):
    temp: float16[COPYBACK_N]
    for i in range(COPYBACK_N):
        temp[i] = a[i] * b[i] + c[i]
        c[i] = temp[i]


def test_operand_that_is_also_a_sink_has_distinct_input_and_output_images(
    samsung_sim_gate,
):
    """The MAD addend ``c`` is read with the fibers exchanged and written back
    as the result's copy sink: the contract holds both images, the emitter's
    swap parity matches the input image, and the program executes exactly."""
    from allo.spmw_candidate import candidate_effects

    compiled = allo.compile(update_addend, build_samsung_target(), samsung_cost)
    layout = compiled.compiled.layout
    layout = layout[0] if isinstance(layout, list) else layout
    effects = candidate_effects(layout)
    images = {(v.value, v.image): v.packing for v in effects.value_layouts}
    assert images[("c", "in")] == "all_bank:swapped_fibers"
    assert images[("c", "out")] == "all_bank"
    assert images[("a", "in")] == images[("b", "in")] == "all_bank"
    assert effects.coverage.inputs == ("a", "b", "c")
    assert effects.coverage.outputs == ("c",)
    (transfer,) = [t for t in effects.transfers if t.value == "c"]
    assert transfer.direction == "inout"
    program = compiled.compiled.layout_ctx.programs[0]
    assert program.kind == "ELTWISE" and program.swap_parity == (False, False, True)
    a = (np.arange(COPYBACK_N) % 4 + 1).astype(np.float16)
    b = (np.arange(COPYBACK_N) % 5 - 2).astype(np.float16)
    c = (np.arange(COPYBACK_N) % 3 - 1).astype(np.float16)
    expected = (a * b + c).astype(np.float16)
    run = compiled(a, b, c)
    assert run.kind == "simulator"
    np.testing.assert_array_equal(c, expected)


def test_vector_residency_must_be_the_exact_realized_storage(monkeypatch):
    """Declaring the host-broadcast vector in another existing storage (the
    SRF) or with a transfer that ends elsewhere is refused at realization,
    during the search, for every candidate."""
    from allo import spmw_samsung

    original = spmw_samsung.samsung_candidate_effects

    def vector_in_srf(target, matches, placement):
        effects = original(target, matches, placement)
        if effects is None:
            return None
        return replace(
            effects,
            value_layouts=tuple(
                replace(v, storage="srf") if v.value == "local_x" else v
                for v in effects.value_layouts
            ),
        )

    monkeypatch.setattr(spmw_samsung, "samsung_candidate_effects", vector_in_srf)
    with pytest.raises(RuntimeError, match="no feasible placement") as error:
        allo.compile(
            samsung_gemv, build_samsung_target(), samsung_cost, backend="virtual"
        )
    assert "loads the vector from" in str(error.value)

    def vector_transfer_elsewhere(target, matches, placement):
        effects = original(target, matches, placement)
        if effects is None:
            return None
        return replace(
            effects,
            transfers=tuple(
                (
                    replace(t, route=("host_scatter", "banks"))
                    if t.value == "local_x"
                    else t
                )
                for t in effects.transfers
            ),
        )

    monkeypatch.setattr(
        spmw_samsung, "samsung_candidate_effects", vector_transfer_elsewhere
    )
    with pytest.raises(RuntimeError, match="no feasible placement") as error:
        allo.compile(
            samsung_gemv,
            build_samsung_target(),
            samsung_cost,
            backend="virtual",
            layout="temporal+crf_shared",
        )
    assert "moves the vector into" in str(error.value) or "loads the vector" in str(
        error.value
    )


# --------------------------------------------------------------------------- #
# (r) The second matcher backend on the shared contract: SK hynix AiM
# --------------------------------------------------------------------------- #

AIM_N = 256
AIM_MAPPING = [32]
AIM_ROWS_PER_CHANNEL = AIM_N // AIM_MAPPING[0]


@allo.dataflow.region()
def aim_gemv_region(
    W: allo.ir.types.bfloat16[AIM_N, AIM_N],
    x: allo.ir.types.bfloat16[AIM_N],
    y: allo.ir.types.bfloat16[AIM_N],
):
    @allo.work(mapping=AIM_MAPPING, args=[W, x, y])
    def mv(
        local_W: allo.ir.types.bfloat16[AIM_N, AIM_N],
        local_x: allo.ir.types.bfloat16[AIM_N],
        local_y: allo.ir.types.bfloat16[AIM_N],
    ):
        (channel,) = allo.get_wid()
        row0 = channel * AIM_ROWS_PER_CHANNEL
        for i in range(AIM_ROWS_PER_CHANNEL):
            acc: allo.ir.types.bfloat16 = 0
            for j in range(AIM_N):
                acc += local_W[row0 + i, j] * local_x[j]
            local_y[row0 + i] = acc


def _compile_aim(**kwargs):
    kwargs.setdefault("reduction_order", "reassociate")
    return allo.compile(aim_gemv_region, build_aim_target(), aim_cost, **kwargs)


def test_aim_candidates_declare_effects_from_the_shared_facts():
    """Every AiM candidate of a program with shared facts declares its
    effects from the work body's `Contraction` facts and the placement
    (handles, knobs); the lowered records are compared with the selected
    declaration at materialization."""
    from allo.spmw_candidate import EFFECTS_REQUIRED_TARGETS, candidate_effects

    assert "aim" in EFFECTS_REQUIRED_TARGETS
    compiled = _compile_aim(backend="virtual")
    facts = compiled.trace.function_facts["mv_0"]
    (contraction,) = facts.contractions
    assert contraction.matrix.memref_name == "local_W"
    assert contraction.vector.memref_name == "local_x"
    (kernel,) = compiled.placement_ranking.kernels
    candidates = [e[1] if isinstance(e, tuple) else e for e in kernel.candidates]
    assert candidates and all(candidate_effects(c) is not None for c in candidates)
    assert kernel.rejections == ()
    layout = compiled.compiled.layout
    layout = layout[0] if isinstance(layout, list) else layout
    effects = candidate_effects(layout)
    assert effects.coverage.inputs == ("local_W", "local_x")
    assert effects.coverage.outputs == ("local_y",)
    storages = {v.value: (v.axes, v.storage) for v in effects.value_layouts}
    assert storages["local_W"] == (("i", "j"), "banks")
    assert storages["local_x"] == (("j",), "gb")
    assert storages["local_y"] == (("i",), "gpr")
    (launch,) = effects.launches
    assert launch.kind in {"MAC", "MAC_ABK"}
    assert launch.instances == (0, AIM_ROWS_PER_CHANNEL)
    assert launch.reduction == (0, AIM_N) and launch.accumulator == "zero"
    scope = compiled.compiled.layout_ctx.lowering_manifest[0]["bank_scope"]
    assert launch.covers[1] == scope
    assert launch.kind == ("MAC_ABK" if scope == "abk" else "MAC")
    assert {t.value: t.direction for t in effects.transfers} == {
        "local_W": "in",
        "local_x": "in",
        "local_y": "out",
    }
    assert {s for _n, _b, s in effects.allocations} == {"mac_reg"}
    assert compiled.estimate().cycles > 0


@pytest.mark.parametrize(
    "mutation, reason",
    [
        (
            lambda e: replace(e, coverage=replace(e.coverage, inputs=("local_W",))),
            "reads",
        ),
        (
            lambda e: replace(
                e,
                launches=tuple(replace(l, reduction=(0, 128)) for l in e.launches),
            ),
            "not an ordered partition",
        ),
        (lambda e: None, "declares no effects"),
    ],
    ids=["inputs", "reduction", "missing"],
)
def test_corrupted_aim_declarations_are_refused_before_costing(
    monkeypatch, mutation, reason
):
    from allo import spmw_aim

    original = spmw_aim.aim_candidate_effects

    def corrupted(target, matches, placement):
        effects = original(target, matches, placement)
        return None if effects is None else mutation(effects)

    monkeypatch.setattr(spmw_aim, "aim_candidate_effects", corrupted)
    with pytest.raises(ValueError, match="violates the candidate contract") as error:
        _compile_aim(backend="virtual")
    assert reason in str(error.value)


def test_aim_realization_mismatch_is_rejected_in_the_search_with_a_fallback(
    monkeypatch,
):
    """Declaring the broadcast vector in the banks (a declarable residency the
    lowering does not realize for the GB broadcast candidates) is rejected at
    materialization for those candidates, which the ranking records; the
    selection falls back to a candidate whose declaration holds."""
    from allo import spmw_aim
    from allo.spmw_candidate import candidate_effects

    baseline = _compile_aim(backend="virtual")
    chosen = baseline.compiled.layout
    chosen = chosen[0] if isinstance(chosen, list) else chosen
    victim = chosen.mode
    original = spmw_aim.aim_candidate_effects

    def vector_in_banks(target, matches, placement):
        effects = original(target, matches, placement)
        if effects is None or placement.mode != victim:
            return effects
        return replace(
            effects,
            value_layouts=tuple(
                replace(v, storage="banks") if v.value == "local_x" else v
                for v in effects.value_layouts
            ),
            transfers=tuple(
                (
                    replace(t, route=("host_scatter", "banks"))
                    if t.value == "local_x"
                    else t
                )
                for t in effects.transfers
            ),
        )

    monkeypatch.setattr(spmw_aim, "aim_candidate_effects", vector_in_banks)
    compiled = _compile_aim(backend="virtual")
    layout = compiled.compiled.layout
    layout = layout[0] if isinstance(layout, list) else layout
    assert layout.mode != victim
    (kernel,) = compiled.placement_ranking.kernels
    reasons = [str(reason) for _position, reason in kernel.rejections]
    assert any(
        "realization differs from the candidate contract" in reason
        and "reads the vector" in reason
        for reason in reasons
    ), reasons
    assert candidate_effects(layout) is not None


def test_aim_selected_program_runs_timing_only_under_the_declared_contract(
    aim_sim_gate,
):
    compiled = _compile_aim()
    layout = compiled.compiled.layout
    layout = layout[0] if isinstance(layout, list) else layout
    assert layout.extra["effects"].launches[0].reduction == (0, AIM_N)
    W = np.zeros((AIM_N, AIM_N), np.float32)
    x = np.zeros(AIM_N, np.float32)
    y = np.full(AIM_N, 5.0, np.float32)
    run = compiled(W, x, y)
    assert run.kind == "simulator-timing-only" and run.cycles > 0
    np.testing.assert_array_equal(y, np.full(AIM_N, 5.0, np.float32))
