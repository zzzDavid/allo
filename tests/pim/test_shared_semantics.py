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
import pathlib
import sys

import numpy as np
import pytest

import allo
from allo.ir.types import float16, int3, int4, int12, int15, uint16
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


# --------------------------------------------------------------------------- #
# (s) Materialization under the real orchestration; the AiM batch mapping,
#     result residency and identity-bound host transfers (review round 21)
# --------------------------------------------------------------------------- #


def _aim_gemv_resident_host():
    def host(W, x, y):
        allo.launch("mv", W, x, y)

    return allo.host_program(aim_gemv_region, resident=("W", "x"))(host)


def _chosen(compiled):
    layout = compiled.compiled.layout
    return layout[0] if isinstance(layout, list) else layout


def test_scheduled_realization_failure_is_rejected_before_acceptance(monkeypatch):
    """A complete assignment is accepted only once the whole program
    materializes under the real launch schedule and host moves, the way the
    selected program is emitted (C-33): a realization mismatch that shows only
    under the schedule rejects the candidate in the ranking, which records it,
    and the selection falls back.  The per-kernel probes stay schedule-free
    preliminary filters."""
    from allo import spmw_aim

    host = _aim_gemv_resident_host()
    victim = _chosen(_compile_aim(backend="virtual", host_moves=host)).mode
    original = spmw_aim.AimCtx.check_realization
    counts = {"scheduled": 0, "kernel": 0}

    def faulty(self, layouts):
        scheduled = self._program.schedule is not None
        counts["scheduled" if scheduled else "kernel"] += 1
        original(self, layouts)
        items = layouts if isinstance(layouts, (list, tuple)) else [layouts]
        if scheduled and any(item.mode == victim for item in items):
            raise RuntimeError("scheduled realization differs for the winning mode")

    monkeypatch.setattr(spmw_aim.AimCtx, "check_realization", faulty)
    compiled = _compile_aim(backend="virtual", host_moves=host)
    assert _chosen(compiled).mode != victim
    # The victim was probed under the schedule (and rejected), the fallback
    # under the schedule (accepted) and the final emission ran under it too.
    assert counts["scheduled"] >= 3 and counts["kernel"] >= 2, counts
    joint = compiled.placement_ranking.joint
    assert joint is not None
    assert any(
        "scheduled realization differs" in reason
        for _assignment, reason in joint.probe_rejections
    ), joint.probe_rejections
    (kernel,) = compiled.placement_ranking.kernels
    # The kernel probe did not reject the victim: the filter is preliminary.
    assert all("scheduled" not in str(r) for _p, r in kernel.rejections)


def _aim_fallback_with_reason(monkeypatch, mutate, needle):
    from allo import spmw_aim

    baseline = _compile_aim(backend="virtual")
    victim = _chosen(baseline).mode
    realized = baseline.compiled.layout_ctx.lowering_manifest[0]["batch_mapping"]
    original = spmw_aim.aim_candidate_effects

    def corrupted(target, matches, placement):
        effects = original(target, matches, placement)
        if effects is None or placement.mode != victim:
            return effects
        return mutate(effects, realized)

    monkeypatch.setattr(spmw_aim, "aim_candidate_effects", corrupted)
    compiled = _compile_aim(backend="virtual")
    assert _chosen(compiled).mode != victim
    (kernel,) = compiled.placement_ranking.kernels
    reasons = [str(reason) for _position, reason in kernel.rejections]
    assert any(
        "realization differs from the candidate contract" in reason and needle in reason
        for reason in reasons
    ), reasons
    return compiled


def test_aim_wrong_batch_mapping_declaration_is_rejected_with_a_fallback(
    monkeypatch,
):
    """A valid but wrong batch mapping in the launch's ``covers`` (one the
    lowering can realize, but not the one this candidate's program realizes)
    is a realization mismatch, rejected at materialization."""

    def wrong_mapping(effects, realized):
        wrong = "channels" if realized != "channels" else "flattened"
        return replace(
            effects,
            launches=tuple(
                replace(launch, covers=(*launch.covers[:3], wrong))
                for launch in effects.launches
            ),
        )

    compiled = _aim_fallback_with_reason(monkeypatch, wrong_mapping, "maps batches")
    chosen = _chosen(compiled)
    declared = chosen.extra["effects"].launches[0].covers[3]
    manifest = compiled.compiled.layout_ctx.lowering_manifest[0]
    assert declared in ("auto", "none", manifest["batch_mapping"])
    if declared not in ("auto", "none"):
        assert declared == manifest["batch_mapping"]


def test_aim_wrong_result_residency_declaration_is_rejected_with_a_fallback(
    monkeypatch,
):
    """Declaring the MAC results in the global buffer (a declarable storage
    the readout does not use: results come back through the GPR) is
    rejected at materialization."""

    def results_in_gb(effects, _realized):
        return replace(
            effects,
            value_layouts=tuple(
                replace(v, storage="gb") if v.value == "local_y" else v
                for v in effects.value_layouts
            ),
            transfers=tuple(
                replace(t, route=("gb", "host_gather")) if t.value == "local_y" else t
                for t in effects.transfers
            ),
        )

    _aim_fallback_with_reason(monkeypatch, results_in_gb, "back from gpr")


AIM_D = 128


@allo.dataflow.region()
def aim_pointwise_region(
    A: allo.ir.types.bfloat16[AIM_D],
    B: allo.ir.types.bfloat16[AIM_D],
    C: allo.ir.types.bfloat16[AIM_D],
):
    @allo.work(mapping=[1], args=[A, B, C])
    def multiply(
        left: allo.ir.types.bfloat16[AIM_D],
        right: allo.ir.types.bfloat16[AIM_D],
        dest: allo.ir.types.bfloat16[AIM_D],
    ):
        for i in range(AIM_D):
            dest[i] = left[i] * right[i]


def _compile_aim_pointwise_with_host():
    def host(A, B, C):
        hx = allo.host_xfer
        hx.scatter(A, hx.banks)
        hx.scatter(B, hx.banks)
        allo.launch("multiply", A, B, C)
        hx.gather(C, hx.banks)

    program = allo.host_program(aim_pointwise_region)(host)
    return allo.compile(
        aim_pointwise_region,
        build_aim_target(),
        aim_cost,
        backend="virtual",
        host_moves=program,
    )


def test_aim_host_transfers_carry_retained_buffer_identities():
    """Every host transfer the AiM derivation lowers names the retained
    identity of the buffer it moves, the slice and the group whose layout it
    follows; these bind it to the candidate's declared transfers."""
    from allo.pim.aim_lowering import AimDistributedHostTransfer

    compiled = _compile_aim_pointwise_with_host()
    ctx = compiled.compiled.layout_ctx
    transfers = [
        op
        for op in ctx.lowered_operations
        if isinstance(op, AimDistributedHostTransfer)
    ]
    assert transfers, [type(op).__name__ for op in ctx.lowered_operations]
    program = ctx._program
    refs = {program.key_by_name[name] for name in ("A", "B", "C")}
    moved = set()
    for op in transfers:
        assert op.buffer_refs and len(op.buffer_refs) == len(op.buffer_slices)
        assert op.group_id == program.groups[0].group_id
        moved.update(op.buffer_refs)
    assert moved == refs
    directions = {ref: op.direction for op in transfers for ref in op.buffer_refs}
    assert directions[program.key_by_name["C"]] == "read"
    assert directions[program.key_by_name["A"]] == "write"
    ctx.check_realization(compiled.compiled.layout)


@pytest.mark.parametrize(
    "mutation, needle",
    [
        (
            lambda op, program: replace(
                op, direction="read" if op.direction == "write" else "write"
            ),
            "the candidate declares it",
        ),
        (lambda op, program: replace(op, elements=op.elements * 4), "moves elements"),
        (
            lambda op, program: replace(
                op, buffer_refs=tuple(("memref", "undeclared") for _ in op.buffer_refs)
            ),
            "declares transfers of",
        ),
        (lambda op, program: replace(op, buffer_refs=()), "retained buffer identity"),
    ],
    ids=["direction", "extent", "identity", "unbound"],
)
def test_aim_host_transfer_records_are_checked_by_identity(mutation, needle):
    """A lowered host transfer whose direction, extent or bound buffer differs
    from the candidate's declared transfer is a realization mismatch; the
    binding is through the retained identity, not the record's label."""
    from allo.pim.aim_lowering import AimDistributedHostTransfer

    compiled = _compile_aim_pointwise_with_host()
    ctx = compiled.compiled.layout_ctx
    program = ctx._program
    ctx.lowered_operations = tuple(
        mutation(op, program) if isinstance(op, AimDistributedHostTransfer) else op
        for op in ctx.lowered_operations
    )
    with pytest.raises(RuntimeError, match="AiM realization") as error:
        ctx.check_realization(compiled.compiled.layout)
    assert needle in str(error.value)


def test_aim_host_transfer_labels_are_not_the_binding():
    """Relabelling a record changes nothing the check reads: the transfer is
    still bound to its declared counterpart by identity (no name matching)."""
    from allo.pim.aim_lowering import AimDistributedHostTransfer

    compiled = _compile_aim_pointwise_with_host()
    ctx = compiled.compiled.layout_ctx
    ctx.lowered_operations = tuple(
        (
            replace(op, name="relabelled")
            if isinstance(op, AimDistributedHostTransfer)
            else op
        )
        for op in ctx.lowered_operations
    )
    ctx.check_realization(compiled.compiled.layout)


def test_aim_declared_input_the_host_program_never_supplies_is_rejected():
    """Under a host program every declared input must reach the device: a
    program that launches the kernel without writing an operand (and without
    residency) is refused at materialization, not run on stale banks."""
    from allo.pim.schedule_search import InfeasibleSchedule

    def host(A, B, C):
        hx = allo.host_xfer
        hx.scatter(A, hx.banks)
        allo.launch("multiply", A, B, C)
        hx.gather(C, hx.banks)

    program = allo.host_program(aim_pointwise_region)(host)
    with pytest.raises((RuntimeError, InfeasibleSchedule, ValueError)) as error:
        allo.compile(
            aim_pointwise_region,
            build_aim_target(),
            aim_cost,
            backend="virtual",
            host_moves=program,
        )
    assert "has not brought" in str(error.value) and "right" in str(error.value)


# --------------------------------------------------------------------------- #
# (t) Transfer availability in schedule order; slice-faithful AiM transfers;
#     host records checked against the kernel's physical layout (review
#     round 22)
# --------------------------------------------------------------------------- #


def _compile_aim_pointwise(host, region=None):
    region = aim_pointwise_region if region is None else region
    return allo.compile(
        region,
        build_aim_target(),
        aim_cost,
        backend="virtual",
        host_moves=allo.host_program(region)(host),
    )


def _rejected(host, region=None):
    from allo.pim.schedule_search import InfeasibleSchedule

    with pytest.raises((RuntimeError, InfeasibleSchedule)) as error:
        _compile_aim_pointwise(host, region)
    return str(error.value)


def test_partial_scatter_does_not_make_the_kernel_input_available():
    """The launch reads the whole operand; a scatter of its first half is
    not availability of the rest (shared check, schedule order)."""

    def host(A, B, C):
        hx = allo.host_xfer
        hx.scatter(A[:64], hx.banks)
        hx.scatter(B, hx.banks)
        allo.launch("multiply", A, B, C)
        hx.gather(C, hx.banks)

    message = _rejected(host)
    assert "has not brought" in message and "[(64, 128)]" in message, message


def test_a_scatter_after_the_launch_does_not_serve_that_launch():
    """Availability is in schedule order: the first launch reads B before the
    only scatter of B; a later write cannot serve an earlier launch, and the
    repeated launch after it is fine."""

    def host(A, B, C):
        hx = allo.host_xfer
        hx.scatter(A, hx.banks)
        allo.launch("multiply", A, B, C)
        hx.scatter(B, hx.banks)
        allo.launch("multiply", A, B, C)
        hx.gather(C, hx.banks)

    message = _rejected(host)
    assert "launch 0 of 'multiply'" in message and "'B'" in message, message

    def ordered(A, B, C):
        hx = allo.host_xfer
        hx.scatter(A, hx.banks)
        hx.scatter(B, hx.banks)
        allo.launch("multiply", A, B, C)
        allo.launch("multiply", A, B, C)
        hx.gather(C, hx.banks)

    compiled = _compile_aim_pointwise(ordered)
    from allo.pim.aim_lowering import AimElementwise

    launches = [
        op
        for op in compiled.compiled.layout_ctx.lowered_operations
        if isinstance(op, AimElementwise)
    ]
    assert len(launches) == 2


def _addresses(ctx, entry):
    begin, end = ctx.lowering_manifest[entry]["command_span"]
    return [tuple(line.split()[2:5]) for line in ctx.cmds[begin:end]]


def test_split_scatters_cover_the_vector_once_at_their_own_addresses():
    """Two half scatters move the halves of the kernel's vector: their bursts
    are disjoint, address the layout the kernel reads, and together equal
    the whole-vector scatter; the records carry the slice frame."""
    from allo.pim.aim_lowering import AimDistributedHostTransfer

    def split(A, B, C):
        hx = allo.host_xfer
        hx.scatter(A[:64], hx.banks)
        hx.scatter(A[64:128], hx.banks)
        hx.scatter(B, hx.banks)
        allo.launch("multiply", A, B, C)
        hx.gather(C, hx.banks)

    def whole(A, B, C):
        hx = allo.host_xfer
        hx.scatter(A, hx.banks)
        hx.scatter(B, hx.banks)
        allo.launch("multiply", A, B, C)
        hx.gather(C, hx.banks)

    split_compiled = _compile_aim_pointwise(split)
    split_ctx = split_compiled.compiled.layout_ctx
    whole_ctx = _compile_aim_pointwise(whole).compiled.layout_ctx
    halves = [
        op
        for op in split_ctx.lowered_operations
        if isinstance(op, AimDistributedHostTransfer) and op.direction == "write"
    ][:2]
    assert [(op.first_element, op.elements, op.vector_elements) for op in halves] == [
        (0, 64, AIM_D),
        (64, 64, AIM_D),
    ]
    first, second = _addresses(split_ctx, 0), _addresses(split_ctx, 1)
    assert first and second and not set(first) & set(second)
    # The whole program coalesces A and B into one record (two bank slots);
    # A's bursts are those of its own slot.
    (coalesced,) = [
        op
        for op in whole_ctx.lowered_operations
        if isinstance(op, AimDistributedHostTransfer) and op.direction == "write"
    ]
    assert coalesced.copies == 2 and coalesced.first_element == 0
    slot_of_a = coalesced.bank_offset
    whole_a = [
        address
        for address in _addresses(whole_ctx, 0)
        if int(address[1]) % coalesced.bank_stride == slot_of_a
    ]
    assert len(whole_a) * 2 == len(_addresses(whole_ctx, 0))
    assert sorted(first + second) == sorted(whole_a)
    split_ctx.check_realization(split_compiled.compiled.layout)


AIM_PREFIX_BUFFER = 256
AIM_PREFIX_READ = 128


@allo.dataflow.region()
def aim_prefix_region(
    A: allo.ir.types.bfloat16[AIM_PREFIX_BUFFER],
    B: allo.ir.types.bfloat16[AIM_PREFIX_BUFFER],
    C: allo.ir.types.bfloat16[AIM_PREFIX_BUFFER],
):
    @allo.work(mapping=[1], args=[A, B, C])
    def multiply(
        left: allo.ir.types.bfloat16[AIM_PREFIX_BUFFER],
        right: allo.ir.types.bfloat16[AIM_PREFIX_BUFFER],
        dest: allo.ir.types.bfloat16[AIM_PREFIX_BUFFER],
    ):
        for i in range(AIM_PREFIX_READ):
            dest[i] = left[i] * right[i]


def test_the_slice_frame_is_the_region_the_kernel_reads():
    """A kernel that reads a prefix of its buffers lays out that prefix; a
    scatter of exactly the prefix is the whole vector (the CENT softmax
    form), a scatter of the whole larger buffer is refused rather than
    spread over the layout, and the prefix is what must be available."""
    from allo.pim.aim_lowering import AimDistributedHostTransfer

    def prefix(A, B, C):
        hx = allo.host_xfer
        hx.scatter(A[0:AIM_PREFIX_READ], hx.banks)
        hx.scatter(B[0:AIM_PREFIX_READ], hx.banks)
        allo.launch("multiply", A, B, C)
        hx.gather(C[0:AIM_PREFIX_READ], hx.banks)

    compiled = _compile_aim_pointwise(prefix, aim_prefix_region)
    writes = [
        op
        for op in compiled.compiled.layout_ctx.lowered_operations
        if isinstance(op, AimDistributedHostTransfer)
    ]
    assert {(op.first_element, op.elements, op.vector_elements) for op in writes} == {
        (0, AIM_PREFIX_READ, AIM_PREFIX_READ)
    }

    def whole(A, B, C):
        hx = allo.host_xfer
        hx.scatter(A, hx.banks)
        hx.scatter(B, hx.banks)
        allo.launch("multiply", A, B, C)
        hx.gather(C, hx.banks)

    message = _rejected(whole, aim_prefix_region)
    assert "outside the region" in message, message


@pytest.mark.parametrize(
    "mutation, needle",
    [
        (lambda op: replace(op, row=op.row + 37), "addresses"),
        (lambda op: replace(op, bank_offset=op.bank_offset + 1), "addresses"),
        (
            lambda op: replace(op, channels_per_replica=op.channels_per_replica * 2),
            "addresses",
        ),
        (lambda op: replace(op, first_element=1), "moves elements"),
        (lambda op: replace(op, group_id=100000), "no launched kernel group"),
    ],
    ids=["row", "bank_slot", "channels", "slice_frame", "owner"],
)
def test_aim_host_records_are_checked_against_the_kernel_layout(mutation, needle):
    """A host record whose rows, bank slot, channels, slice frame or owner
    differ from the layout the launched kernel record addresses is a
    realization mismatch."""
    from allo.pim.aim_lowering import AimDistributedHostTransfer

    compiled = _compile_aim_pointwise_with_host()
    ctx = compiled.compiled.layout_ctx
    ctx.lowered_operations = tuple(
        mutation(op) if isinstance(op, AimDistributedHostTransfer) else op
        for op in ctx.lowered_operations
    )
    with pytest.raises(RuntimeError, match="AiM realization") as error:
        ctx.check_realization(compiled.compiled.layout)
    assert needle in str(error.value), str(error.value)


def test_aim_host_records_redirected_before_emission_are_refused(monkeypatch):
    """Records redirected to other rows before command emission (a
    self-consistent but wrong manifest) fail the realization check against
    the kernel record at materialization: no candidate is accepted."""
    from allo import spmw_aim
    from allo.pim.aim_lowering import AimDistributedHostTransfer

    original = spmw_aim._AimProgramDerivation.records

    def redirected(self, *args):
        return [
            (
                (
                    replace(op, row=op.row + 37)
                    if isinstance(op, AimDistributedHostTransfer)
                    else op
                ),
                banks,
            )
            for op, banks in original(self, *args)
        ]

    monkeypatch.setattr(spmw_aim._AimProgramDerivation, "records", redirected)
    with pytest.raises(RuntimeError) as error:
        _compile_aim_pointwise_with_host()
    assert "addresses" in str(error.value), str(error.value)


def test_resident_inputs_and_produced_intermediates_are_available():
    """Residency is explicit availability; a launch's declared outputs are
    available to the launches after it (the GEMV with resident operands and
    the chained pointwise program compile)."""
    compiled = _compile_aim(backend="virtual", host_moves=_aim_gemv_resident_host())
    assert compiled.compiled.layout_ctx.lowered_operations


# --------------------------------------------------------------------------- #
# u. UPMEM and AttAcc declare their candidates' effects from the shared facts
#    (G9: every released matcher backend is on the enforced candidate contract)
# --------------------------------------------------------------------------- #

UPMEM_M, UPMEM_K = 256, 128


def _upmem_gemv_region(dpus=2, *, alpha=None):
    import sys

    sys.path.insert(0, "benchmarks/upmem")
    from workloads import gemv_workload

    return gemv_workload(UPMEM_M, UPMEM_K, dpus, alpha=alpha)


def _compile_upmem(region, **kw):
    from allo.pim.costs import upmem_cost
    from allo.pim.targets import build_upmem_target

    return allo.compile(
        region, build_upmem_target(), upmem_cost, backend="virtual", **kw
    )


def _selected_layouts(compiled):
    layout = compiled.compiled.layout
    return layout if isinstance(layout, list) else [layout]


def _all_rejections(compiled) -> str:
    """Every recorded rejection of the ranking: the per-kernel probes and the
    joint materialization under the real orchestration."""
    ranking = compiled.placement_ranking
    reasons = [str(r) for kernel in ranking.kernels for r in kernel.rejections]
    joint = getattr(ranking, "joint", None)
    if joint is not None:
        reasons.extend(str(r) for _a, r in joint.probe_rejections)
    return " | ".join(reasons)


def test_upmem_candidates_declare_effects_from_the_shared_facts():
    """Every UPMEM candidate of a program with shared facts declares its
    effects from the work body's `Contraction` facts (loop labels, trips,
    accesses, accumulator initialization) and its physical decision: MRAM
    backing storage with the decision's data layout as packing, the vector's
    residency, WRAM as a transient allocation, pack/readback transfers and
    one template launch per work item; the realized plan is compared with
    the selected declaration."""
    from allo.spmw_candidate import EFFECTS_REQUIRED_TARGETS, candidate_effects

    assert {"upmem", "attacc"} <= EFFECTS_REQUIRED_TARGETS
    compiled = _compile_upmem(_upmem_gemv_region(2))
    for kernel in compiled.placement_ranking.kernels:
        candidates = [
            entry[1] if isinstance(entry, tuple) else entry
            for entry in kernel.candidates
        ]
        assert candidates and all(candidate_effects(c) is not None for c in candidates)
    layouts = _selected_layouts(compiled)
    assert len(layouts) == 2  # one work item per DPU
    for layout in layouts:
        effects = candidate_effects(layout)
        assert effects.coverage.inputs == ("lA", "lx")
        assert effects.coverage.outputs == ("ly",)
        images = {(v.value, v.image): v for v in effects.value_layouts}
        matrix = images[("lA", "in")]
        assert matrix.axes == ("i", "j") and matrix.storage == "mram"
        assert matrix.packing == layout.extra["data_layout"]
        vector = images[("lx", "in")]
        assert vector.axes == ("j",) and vector.storage == "mram"
        assert vector.packing == (
            f"{layout.extra['data_layout']}:{layout.extra['wram_residency']}"
        )
        assert images[("ly", "value")].axes == ("i",)
        assert {s for _n, _b, s in effects.allocations} == {"wram"}
        (launch,) = effects.launches
        assert launch.kind == "upmem:matrix_vector"
        assert launch.instances == (0, UPMEM_M // 2)
        assert launch.reduction == (0, UPMEM_K)
        assert launch.accumulator == "zero"
        assert launch.covers[0] == layout.mode
        assert {t.value: (t.direction, t.route) for t in effects.transfers} == {
            "lA": ("in", ("host_pack", "mram")),
            "lx": ("in", ("host_pack", "mram")),
            "ly": ("out", ("mram", "host_readback")),
        }
    assert compiled.estimate().cycles > 0


def test_upmem_scaled_gemv_declares_both_operations_in_one_launch():
    """``y = alpha * (A x)``: the MAC with a local accumulator and the SCALE
    epilogue are one template launch over the same result instances."""
    from allo.spmw_candidate import candidate_effects

    compiled = _compile_upmem(_upmem_gemv_region(1, alpha=3))
    (layout,) = _selected_layouts(compiled)
    effects = candidate_effects(layout)
    assert effects.coverage.operations == ("MAC", "SCALE")
    (launch,) = effects.launches
    assert launch.operations == (0, 1)
    assert launch.instances == (0, UPMEM_M) and launch.reduction == (0, UPMEM_K)


def test_upmem_zero_initialized_partial_is_an_output_not_an_input():
    """``P[d] = 0; for i: P[d] += A[...]`` (the reduction family, matched as
    an accumulating ADD): the partial's incoming contents are dead, so it is
    an output only (the template zero-initializes and the runtime does not
    pack it); a carried partial (no initializer) is ``inout`` and the
    template loads the initial output."""
    import sys

    sys.path.insert(0, "benchmarks/upmem")
    from workloads import reduction_workload
    from allo.spmw_candidate import candidate_effects

    compiled = _compile_upmem(reduction_workload(4096, 2))
    for layout in _selected_layouts(compiled):
        effects = candidate_effects(layout)
        assert effects.coverage.inputs == ("lA",)
        assert effects.coverage.outputs == ("lP",)
        assert {t.value: t.direction for t in effects.transfers} == {
            "lA": "in",
            "lP": "out",
        }
        (launch,) = effects.launches
        assert launch.kind == "upmem:reduction" and launch.instances == (0, 2048)
    import runpy

    generality = runpy.run_path("tests/pim/test_upmem_generality.py")
    carried = _compile_upmem(generality["reduction_region"](4096, carried=True))
    (layout,) = _selected_layouts(carried)
    effects = candidate_effects(layout)
    assert "lP" in effects.coverage.inputs and effects.coverage.outputs == ("lP",)
    (transfer,) = [t for t in effects.transfers if t.value == "lP"]
    assert transfer.direction == "inout"
    assert transfer.route == ("host_pack", "mram", "host_readback")
    assert {(v.value, v.image) for v in effects.value_layouts} >= {
        ("lP", "in"),
        ("lP", "out"),
    }


def test_upmem_indirect_reads_are_inputs():
    """A histogram reads its samples only inside the bin index and the
    guards: the retained store effect records those reads, so the samples
    are a program input with a layout over the sample loop and a packed
    transfer, and the template's ``input`` role is the declared value."""
    import runpy

    from allo.spmw_candidate import candidate_effects

    generality = runpy.run_path("tests/pim/test_upmem_generality.py")
    compiled = _compile_upmem(generality["histogram_region"](1536, 64, 12))
    (layout,) = _selected_layouts(compiled)
    effects = candidate_effects(layout)
    assert effects.coverage.inputs == ("lA",) and effects.coverage.outputs == ("lH",)
    (samples,) = [v for v in effects.value_layouts if v.value == "lA"]
    assert samples.axes == ("i",) and samples.storage == "mram"
    (launch,) = effects.launches
    assert launch.kind == "upmem:histogram"


def test_upmem_host_partition_is_a_declared_whole_function_launch():
    """The k-means update group runs its whole function on the host (the
    paper partition): its candidate declares ``host_function`` with one
    whole-function launch over the function's retained store effects, values
    in ``host_dram`` and host copies; the distances group stays a DPU
    template launch."""
    from benchmarks.upmem.workloads import kmeans_host, kmeans_region
    from allo.spmw_candidate import candidate_effects

    compiled = _compile_upmem(kmeans_region, host_moves=kmeans_host)
    by_kind = {}
    for layout in _selected_layouts(compiled):
        effects = candidate_effects(layout)
        (launch,) = effects.launches
        by_kind[launch.kind] = (effects, launch)
    assert set(by_kind) == {"upmem:kmeans_distances", "upmem_host:kmeans_update"}
    effects, launch = by_kind["upmem_host:kmeans_update"]
    assert effects.host_function and launch.whole
    assert launch.instances == (0, 1) and launch.reduction is None
    assert {v.storage for v in effects.value_layouts} == {"host_dram"}
    assert all("host_dram" in t.route for t in effects.transfers)
    effects, launch = by_kind["upmem:kmeans_distances"]
    assert not effects.host_function and not launch.whole
    assert launch.reduction is not None
    ctx = compiled.compiled.layout_ctx
    assert [s.kind for s in ctx.segments].count("host") == 1


@pytest.mark.parametrize(
    "mutation, reason",
    [
        (
            lambda e: replace(e, coverage=replace(e.coverage, inputs=("lA",))),
            "reads",
        ),
        (
            lambda e: replace(
                e,
                launches=tuple(replace(l, reduction=(0, 64)) for l in e.launches),
            ),
            "not an ordered partition",
        ),
        (
            lambda e: replace(
                e,
                value_layouts=tuple(
                    replace(v, storage="scratchpad") if v.value == "lx" else v
                    for v in e.value_layouts
                ),
            ),
            "does not declare",
        ),
        (lambda e: None, "declares no effects"),
    ],
    ids=["inputs", "reduction", "storage", "missing"],
)
def test_corrupted_upmem_declarations_are_refused_before_costing(
    monkeypatch, mutation, reason
):
    from allo import spmw_upmem

    original = spmw_upmem.upmem_candidate_effects

    def corrupted(target, matches, placement):
        effects = original(target, matches, placement)
        return mutation(effects) if effects is not None else None

    monkeypatch.setattr(spmw_upmem, "upmem_candidate_effects", corrupted)
    with pytest.raises(ValueError) as error:
        _compile_upmem(_upmem_gemv_region(1))
    assert "candidate contract" in str(error.value)
    assert reason in str(error.value), str(error.value)


@pytest.mark.parametrize(
    "mutation, needle",
    [
        (
            lambda e: replace(
                e,
                value_layouts=tuple(
                    (
                        replace(v, packing="fused_replicated:streamed_mram")
                        if v.value == "lx"
                        else v
                    )
                    for v in e.value_layouts
                ),
            ),
            "keeps 'lx'",
        ),
        (
            lambda e: replace(
                e,
                value_layouts=tuple(
                    replace(v, packing="other_layout") if v.value == "lA" else v
                    for v in e.value_layouts
                ),
            ),
            "lays 'lA' out as",
        ),
        (
            lambda e: replace(
                e,
                value_layouts=tuple(
                    replace(v, storage="iram") if v.value == "lx" else v
                    for v in e.value_layouts
                ),
            ),
            "packs 'lx' into the MRAM heap image",
        ),
    ],
    ids=["residency", "data_layout", "storage"],
)
def test_upmem_realization_is_compared_with_the_declaration(
    monkeypatch, mutation, needle
):
    """A declaration the contract accepts but the template does not realize
    (a residency the decision does not take, another data layout than the
    decision's, a declared storage the template does not use) is a
    realization mismatch at materialization: the candidate is rejected with
    the difference named and the selection falls back to a faithful one."""
    from allo import spmw_upmem

    baseline = _compile_upmem(_upmem_gemv_region(1))
    (chosen,) = _selected_layouts(baseline)
    victim = chosen.mode  # the candidate the ranking selects when undisturbed
    original = spmw_upmem.upmem_candidate_effects

    def corrupted(target, matches, placement):
        effects = original(target, matches, placement)
        if effects is None or placement.mode != victim:
            return effects
        return mutation(effects)

    monkeypatch.setattr(spmw_upmem, "upmem_candidate_effects", corrupted)
    compiled = _compile_upmem(_upmem_gemv_region(1))
    # The corrupted winner was rejected at materialization; another won.
    (layout,) = _selected_layouts(compiled)
    assert layout.mode != victim
    rejections = _all_rejections(compiled)
    assert needle in rejections, rejections


def test_upmem_realization_check_refuses_a_foreign_plan(monkeypatch):
    """The final check compares the emitted plan with the declaration: a
    plan whose accumulator chain differs from the kernel's is refused."""
    compiled = _compile_upmem(_upmem_gemv_region(1))
    ctx = compiled.compiled.layout_ctx
    (program,) = ctx.groups.values()
    monkeypatch.setattr(
        program.plan.__class__, "carried_output", property(lambda self: True)
    )
    with pytest.raises(RuntimeError, match="carried_output"):
        ctx.check_realization(compiled.compiled.layout)


ATTACC_ROWS, ATTACC_K, ATTACC_N = 8, 5120, 384


def _compile_attacc(region):
    from allo.pim.costs import attacc_cost
    from allo.pim.targets import build_attacc_target

    return allo.compile(
        region,
        build_attacc_target(),
        attacc_cost,
        backend="virtual",
        reduction_order="reassociate",
    )


def _attacc_regions():
    import runpy

    return runpy.run_path("tests/pim/test_attacc_virtual.py")


def test_attacc_candidates_declare_effects_from_the_shared_facts():
    """Every AttAcc tiling candidate declares its effects from the MAC's
    facts: the matrix in the banks, the vector in the GEMV buffer (through
    the banks and WR_GB), the result in the accumulator, one launch over
    rows x outputs instances and the reduction, started from zero, naming
    the tiling; the emitted command inventory is compared with it."""
    from allo.spmw_candidate import candidate_effects

    regions = _attacc_regions()
    compiled = _compile_attacc(regions["gemv_region"](ATTACC_ROWS, ATTACC_K, ATTACC_N))
    (kernel,) = compiled.placement_ranking.kernels
    candidates = [
        entry[1] if isinstance(entry, tuple) else entry for entry in kernel.candidates
    ]
    assert candidates and all(candidate_effects(c) is not None for c in candidates)
    (layout,) = _selected_layouts(compiled)
    effects = candidate_effects(layout)
    assert effects.coverage.inputs == ("lW", "lx")
    assert effects.coverage.outputs == ("ly",)
    storage = {v.value: (v.axes, v.storage) for v in effects.value_layouts}
    assert storage == {
        "lW": (("h", "n", "k"), "banks"),
        "lx": (("h", "k"), "gemv_buffer"),
        "ly": (("h", "n"), "acc"),
    }
    assert {t.value: t.route for t in effects.transfers} == {
        "lW": ("host_scatter", "banks"),
        "lx": ("host_scatter", "banks", "WR_GB", "gemv_buffer"),
        "ly": ("acc", "host_readback"),
    }
    (launch,) = effects.launches
    assert launch.kind == "attacc_gemv"
    assert launch.instances == (0, ATTACC_ROWS * ATTACC_N)
    assert launch.reduction == (0, ATTACC_K) and launch.accumulator == "zero"
    assert launch.covers == (
        layout.mode,
        str(layout.extra["attacc_tiling"]),
        "MAC",
    )
    assert compiled.estimate().cycles == 7040


def test_attacc_attention_declares_its_five_phases():
    """QK^T, the softmax and PV are five launches in match order; the two
    MAC phases cover rows x length over d_head and rows x d_head over the
    length, both from zero; the score rows are a softmax-buffer allocation."""
    from allo.spmw_candidate import candidate_effects

    regions = _attacc_regions()
    compiled = _compile_attacc(regions["attention_region"](8, 128, 256))
    (layout,) = _selected_layouts(compiled)
    effects = candidate_effects(layout)
    assert effects.coverage.inputs == ("lq", "lk", "lv")
    assert effects.coverage.outputs == ("lo",)
    phases = [(l.covers[-1], l.instances, l.reduction) for l in effects.launches]
    assert phases == [
        ("MAC", (0, 8 * 256), (0, 128)),
        ("SFM_EXP", (0, 8 * 256), None),
        ("SFM_SUM", (0, 8 * 256), None),
        ("SFM_DIV", (0, 8 * 256), None),
        ("MAC", (0, 8 * 128), (0, 256)),
    ]
    assert ("scores", 0, "softmax_buffer") in effects.allocations


@pytest.mark.parametrize(
    "mutation, reason",
    [
        (
            lambda e: replace(e, coverage=replace(e.coverage, outputs=())),
            "writes",
        ),
        (
            lambda e: replace(
                e,
                launches=tuple(replace(l, accumulator="carried") for l in e.launches),
            ),
            "starts its accumulator",
        ),
        (lambda e: None, "declares no effects"),
    ],
    ids=["outputs", "accumulator", "missing"],
)
def test_corrupted_attacc_declarations_are_refused_before_costing(
    monkeypatch, mutation, reason
):
    from allo import spmw_attacc

    original = spmw_attacc.attacc_candidate_effects

    def corrupted(target, matches, placement):
        effects = original(target, matches, placement)
        return mutation(effects) if effects is not None else None

    monkeypatch.setattr(spmw_attacc, "attacc_candidate_effects", corrupted)
    regions = _attacc_regions()
    with pytest.raises(ValueError) as error:
        _compile_attacc(regions["gemv_region"](ATTACC_ROWS, ATTACC_K, ATTACC_N))
    assert "candidate contract" in str(error.value)
    assert reason in str(error.value), str(error.value)


def test_attacc_realization_is_compared_with_the_declaration(monkeypatch):
    """A declaration naming another tiling than the one the program is
    emitted for is a realization mismatch: the candidate is rejected at
    materialization and the selection falls back to a faithful one."""
    from allo import spmw_attacc

    original = spmw_attacc.attacc_candidate_effects
    regions = _attacc_regions()
    reference = _compile_attacc(regions["gemv_region"](ATTACC_ROWS, ATTACC_K, ATTACC_N))
    (best,) = _selected_layouts(reference)
    best_tiling = str(best.extra["attacc_tiling"])

    def corrupted(target, matches, placement):
        effects = original(target, matches, placement)
        if effects is None or str(placement.extra.get("attacc_tiling")) != best_tiling:
            return effects
        return replace(
            effects,
            launches=tuple(
                replace(
                    l, covers=(l.covers[0], "[[1,1,1],[1,1,1],[1,1,1]]", l.covers[2])
                )
                for l in effects.launches
            ),
        )

    monkeypatch.setattr(spmw_attacc, "attacc_candidate_effects", corrupted)
    compiled = _compile_attacc(regions["gemv_region"](ATTACC_ROWS, ATTACC_K, ATTACC_N))
    (layout,) = _selected_layouts(compiled)
    assert str(layout.extra["attacc_tiling"]) != best_tiling
    rejections = _all_rejections(compiled)
    assert "uses tiling" in rejections, rejections
    assert compiled.estimate().cycles >= reference.estimate().cycles


# --------------------------------------------------------------------------- #
# v. Review round 24: AttAcc honours the launch schedule and binds its datapath
#    roles; the shared availability check uses the IR's allocation shapes and
#    converts flat multi-dimensional slices correctly (X-15 to X-18)
# --------------------------------------------------------------------------- #


def _attacc_factory_region():
    from benchmarks.attacc.workloads import gemv_region

    return gemv_region(3, 80, 33)


def _attacc_once(W, x, y):
    allo.launch("gemv", W, x, y)


def _attacc_twice(W, x, y):
    allo.launch("gemv", W, x, y)
    allo.launch("gemv", W, x, y)


def test_attacc_resident_inputs_of_a_deferred_annotation_factory_are_available():
    """A benchmark factory whose annotations name closure dimensions gives the
    schedule no parameter shapes; the availability check takes the full
    allocation shapes from the normalized program's facts, so resident
    inputs are available (X-17)."""
    region = _attacc_factory_region()
    host = allo.host_program(region, resident=("W", "x"))(_attacc_once)
    compiled = _compile_attacc_with(region, host_moves=host)
    assert compiled.estimate().cycles == 320
    assert [s for s in compiled.compiled.launch_schedule.parameters] == [
        ("W", None),
        ("x", None),
        ("y", None),
    ]


def _compile_attacc_with(region, **kw):
    from allo.pim.costs import attacc_cost
    from allo.pim.targets import build_attacc_target

    return allo.compile(
        region,
        build_attacc_target(),
        attacc_cost,
        backend="virtual",
        reduction_order="reassociate",
        **kw,
    )


def test_attacc_materializes_one_inventory_per_launch():
    """The analytical artifact follows the retained schedule: two launches of
    the GEMV are two program records and twice the commands, and the priced
    graph agrees (X-15)."""
    region = _attacc_factory_region()
    once = _compile_attacc_with(
        region, host_moves=allo.host_program(region, resident=("W", "x"))(_attacc_once)
    )
    twice = _compile_attacc_with(
        region,
        host_moves=allo.host_program(region, resident=("W", "x"))(_attacc_twice),
    )
    assert len(once.compiled.layout_ctx.programs) == 1
    assert len(twice.compiled.layout_ctx.programs) == 2
    assert list(twice.compiled.cmds) == list(once.compiled.cmds) * 2
    assert twice.estimate().cycles == 2 * once.estimate().cycles == 640
    manifest = twice.compiled.layout_ctx.virtual_run_extra()
    assert len(manifest["attacc_programs"]) == 2


def test_attacc_swapped_datapath_declaration_is_refused_at_materialization():
    """A declaration that passes the shared validator but swaps the matrix
    and vector homes (and their routes) does not describe the inventory the
    AttAcc model emits: every such candidate is rejected at materialization
    with the role named (X-16)."""
    from allo import spmw_attacc

    original = spmw_attacc.attacc_candidate_effects
    swap = {"banks": "gemv_buffer", "gemv_buffer": "banks"}

    def swapped(target, matches, placement):
        effects = original(target, matches, placement)
        if effects is None:
            return effects
        return replace(
            effects,
            value_layouts=tuple(
                replace(v, storage=swap.get(v.storage, v.storage))
                for v in effects.value_layouts
            ),
            transfers=tuple(
                replace(t, route=tuple(swap.get(s, s) for s in t.route))
                for t in effects.transfers
            ),
        )

    spmw_attacc.attacc_candidate_effects = swapped
    try:
        with pytest.raises(RuntimeError) as error:
            _compile_attacc_with(_attacc_factory_region())
    finally:
        spmw_attacc.attacc_candidate_effects = original
    message = str(error.value)
    assert "keeps 'lW' in 'banks'" in message, message


def test_attacc_attention_phase_order_is_checked(monkeypatch):
    """The attention inventory issues QK^T, exp, sum, div, PV in that order;
    a declaration with the phases reordered is refused."""
    from allo import spmw_attacc

    regions = _attacc_regions()
    original = spmw_attacc.attacc_candidate_effects

    def reordered(target, matches, placement):
        effects = original(target, matches, placement)
        if effects is None:
            return effects
        launches = list(effects.launches)
        launches[0], launches[1] = (
            replace(launches[1], index=0),
            replace(launches[0], index=1),
        )
        return replace(effects, launches=tuple(launches))

    monkeypatch.setattr(spmw_attacc, "attacc_candidate_effects", reordered)
    with pytest.raises(RuntimeError, match="phases"):
        _compile_attacc(regions["attention_region"](8, 128, 256))


def test_flat_slices_of_a_matrix_cover_whole_rows_exactly():
    """Flat ranges over a 3x33x80 matrix: two scatters covering rows 0 and 1
    leave row 2 missing (rejected); three covering every row are complete
    (accepted); a range that is not whole rows is refused (X-18).  Checked
    through the shared stage directly: the AttAcc cost refuses to price host
    ingress, so the compile stops before the stage on that target."""
    from types import SimpleNamespace

    from allo.pim.schedule_search import InfeasibleSchedule
    from allo.spmw_plan import HostStep, LaunchStep, check_transfer_availability

    region = _attacc_factory_region()
    compiled = _compile_attacc_with(
        region, host_moves=allo.host_program(region, resident=("W", "x"))(_attacc_once)
    )
    schedule = compiled.compiled.launch_schedule
    launch = next(s for s in schedule.steps if isinstance(s, LaunchStep))

    def check(slices):
        steps = tuple(HostStep(i) for i in range(len(slices))) + (launch,)
        moves = [
            SimpleNamespace(verb=SimpleNamespace(name="scatter"), buffer_role=s)
            for s in slices
        ]
        check_transfer_availability(
            compiled.trace,
            replace(schedule, steps=steps, resident=frozenset(("x",))),
            moves,
            compiled.compiled.layout,
        )

    with pytest.raises(InfeasibleSchedule, match=r"has not brought \[\(2, 3\)"):
        check(("W[0:2640]", "W[2640:5280]"))
    check(("W[0:2640]", "W[2640:5280]", "W[5280:7920]"))
    check(("W[0:5280]", "W[5280:7920]"))
    with pytest.raises(InfeasibleSchedule, match="whole rows"):
        check(("W[0:2000]", "W[2000:7920]"))


def test_aim_host_rewrite_of_a_produced_value_after_its_launch_is_a_reupload():
    """A host scatter of a kernel output after the kernel ran (CENT's store
    helper after the host side of RoPE) is a re-upload of a produced value:
    the record follows the producer's output slot, is marked ``reupload``
    and binds to the declared output; the same scatter before the launch
    would be an ingress the candidate does not declare."""
    from allo.pim.aim_lowering import AimDistributedHostTransfer

    def rewrite_after(A, B, C):
        hx = allo.host_xfer
        hx.scatter(A, hx.banks)
        hx.scatter(B, hx.banks)
        allo.launch("multiply", A, B, C)
        hx.scatter(C, hx.banks)
        hx.gather(C, hx.banks)

    compiled = _compile_aim_pointwise(rewrite_after)
    ctx = compiled.compiled.layout_ctx
    writes = [
        op
        for op in ctx.lowered_operations
        if isinstance(op, AimDistributedHostTransfer) and op.direction == "write"
    ]
    names = {op.name: op.reupload for op in writes}
    assert names.get("C") is True, names
    assert all(not flag for name, flag in names.items() if name != "C"), names
    ctx.check_realization(compiled.compiled.layout)
    # Relabelled as an ordinary ingress, the same record is refused: C is
    # declared out.
    ctx.lowered_operations = tuple(
        replace(op, reupload=False) if getattr(op, "name", None) == "C" else op
        for op in ctx.lowered_operations
    )
    with pytest.raises(RuntimeError, match="declares it 'out'"):
        ctx.check_realization(compiled.compiled.layout)


# --------------------------------------------------------------------------- #
# w. Review round 25: flat ranges are a distinct region type (X-18 reopened);
#    AttAcc attention proves its dataflow and per-row normalization (X-20)
# --------------------------------------------------------------------------- #


def test_flat_ranges_and_cartesian_boxes_are_distinct_regions():
    """``W[0:4]`` over a (4, 2, 2) buffer is a flat range of its flattening;
    ``W[0:4, 0, 0]`` is the Cartesian box of four scattered elements and is
    not contiguous; a scalar's flat offset is its row-major position."""
    from allo.spmw_plan import FlatRange, flat_range, slice_region

    shape = (4, 2, 2)
    flat = slice_region(shape, ((0, 4),))
    assert flat == FlatRange(0, 4) and flat_range(shape, flat) == (0, 4)
    # Four flat elements are whole row 0; six are not whole rows.
    assert flat.rows(shape) == ((0, 1), (0, 2), (0, 2))
    assert FlatRange(0, 6).rows(shape) is None
    assert FlatRange(0, 8).rows(shape) == ((0, 2), (0, 2), (0, 2))
    box = slice_region(shape, ((0, 4), 0, 0))
    assert box == ((0, 4), (0, 1), (0, 1)) and flat_range(shape, box) is None
    assert flat_range(shape, slice_region(shape, (1, 0, 0))) == (4, 1)
    assert flat_range(shape, slice_region(shape, ((0, 4), None, None))) == (0, 16)


def test_cartesian_scatter_of_scattered_elements_does_not_supply_a_matrix():
    """A GEMV reading ``W[0, :, :]`` is supplied by the flat prefix
    ``W[0:4]`` or the box ``W[0, :, :]``, not by ``W[0:4, 0, 0]`` (four
    scattered elements, one of which lies in the matrix)."""
    from types import SimpleNamespace

    from allo.dataflow import region
    from allo.pim.schedule_search import InfeasibleSchedule
    from allo.spmw_plan import HostStep, LaunchStep, check_transfer_availability

    @region()
    def prefix(W: float16[4, 2, 2], x: float16[4, 2], y: float16[4, 2]):
        @allo.work(mapping=[1], args=[W, x, y])
        def gemv(lW: float16[4, 2, 2], lx: float16[4, 2], ly: float16[4, 2]):
            for h in range(1):
                for n in range(2):
                    acc: float16 = 0
                    for k in range(2):
                        acc += lW[h, n, k] * lx[h, k]
                    ly[h, n] = acc

    compiled = _compile_attacc_with(
        prefix, host_moves=allo.host_program(prefix, resident=("W", "x"))(_attacc_once)
    )
    schedule = compiled.compiled.launch_schedule
    launch = next(s for s in schedule.steps if isinstance(s, LaunchStep))

    def check(role):
        moves = [
            SimpleNamespace(verb=SimpleNamespace(name="scatter"), buffer_role=role)
        ]
        check_transfer_availability(
            compiled.trace,
            replace(schedule, steps=(HostStep(0), launch), resident=frozenset(("x",))),
            moves,
            compiled.compiled.layout,
        )

    with pytest.raises(
        InfeasibleSchedule, match=r"has not brought \[\(0, 1\), \(0, 1\), \(1, 2\)\]"
    ):
        check("W[0:4, 0, 0]")
    check("W[0:4]")
    check("W[0, :, :]")


def _attention_disconnected():
    """Scores in ``s``; the softmax and the context MAC use a separate
    zero-filled ``p`` (the softmax never reads the scores)."""
    from allo.dataflow import region
    from allo.spmw_match import exp

    @region()
    def attention_variant(
        q: float16[2, 16],
        kc: float16[2, 32, 16],
        vc: float16[2, 32, 16],
        o: float16[2, 16],
    ):
        @allo.work(mapping=[1], args=[q, kc, vc, o])
        def attention(
            lq: float16[2, 16],
            lk: float16[2, 32, 16],
            lv: float16[2, 32, 16],
            lo: float16[2, 16],
        ):
            s: float16[2, 32] = 0.0
            p: float16[2, 32] = 0.0
            for h in range(2):
                for t in range(32):
                    acc: float16 = 0.0
                    for d in range(16):
                        acc += lq[h, d] * lk[h, t, d]
                    s[h, t] = acc
            for h in range(2):
                total: float16 = 0.0
                for t in range(32):
                    p[h, t] = exp(p[h, t])
                for t in range(32):
                    total += p[h, t]
                for t in range(32):
                    p[h, t] = p[h, t] / total
            for h in range(2):
                for d in range(16):
                    acc2: float16 = 0.0
                    for t in range(32):
                        acc2 += p[h, t] * lv[h, t, d]
                    lo[h, d] = acc2

    return attention_variant


def _attention_global():
    """The softmax sum runs over both heads (one normalization across rows)."""
    from allo.dataflow import region
    from allo.spmw_match import exp

    @region()
    def attention_variant(
        q: float16[2, 16],
        kc: float16[2, 32, 16],
        vc: float16[2, 32, 16],
        o: float16[2, 16],
    ):
        @allo.work(mapping=[1], args=[q, kc, vc, o])
        def attention(
            lq: float16[2, 16],
            lk: float16[2, 32, 16],
            lv: float16[2, 32, 16],
            lo: float16[2, 16],
        ):
            s: float16[2, 32] = 0.0
            for h in range(2):
                for t in range(32):
                    acc: float16 = 0.0
                    for d in range(16):
                        acc += lq[h, d] * lk[h, t, d]
                    s[h, t] = acc
            for h in range(2):
                for t in range(32):
                    s[h, t] = exp(s[h, t])
            total: float16 = 0.0
            for h in range(2):
                for t in range(32):
                    total += s[h, t]
            for h in range(2):
                for t in range(32):
                    s[h, t] = s[h, t] / total
            for h in range(2):
                for d in range(16):
                    acc2: float16 = 0.0
                    for t in range(32):
                        acc2 += s[h, t] * lv[h, t, d]
                    lo[h, d] = acc2

    return attention_variant


_ATTENTION_VARIANTS = {
    "disconnected": _attention_disconnected,
    "global": _attention_global,
}


@pytest.mark.parametrize(
    "kind, reason",
    [
        ("disconnected", "not the score buffer"),
        ("global", "not re-initialized per row"),
    ],
)
def test_attacc_attention_requires_the_fused_dataflow(kind, reason):
    """The fused attention inventory is selected only when the shared facts
    prove score -> exp -> sum -> div -> context over one score buffer with a
    per-row sum; a softmax over another buffer, or one normalized across
    rows, is refused with the difference named (no source-name checks)."""
    from allo.pim.attacc_tiling import AttAccDomainError

    with pytest.raises(AttAccDomainError, match=reason):
        _compile_attacc_with(_ATTENTION_VARIANTS[kind]())


def test_attacc_ordinary_attention_still_lowers():
    from benchmarks.attacc.workloads import attention_region

    compiled = _compile_attacc_with(attention_region(2, 16, 32))
    assert compiled.estimate().cycles == 465


# --------------------------------------------------------------------------- #
# x. Review round 26: the attention proof is structural (X-20 reopened): the
#    division's operand order is checked, equivalent loop renamings pass
# --------------------------------------------------------------------------- #


def _attention_source(*, divide="s[h, t] / total", context_loop="t", row_loop="h"):
    """The paper's attention region with one of its loops or its division
    rewritten; the semantics are unchanged by a renaming and changed by the
    reversed division."""
    from allo.dataflow import region
    from allo.spmw_match import exp

    r = row_loop
    z = context_loop
    src = f"""
@region()
def attention_variant(q: float16[2, 16], kc: float16[2, 32, 16], vc: float16[2, 32, 16], o: float16[2, 16]):
    @allo.work(mapping=[1], args=[q, kc, vc, o])
    def attention(lq: float16[2, 16], lk: float16[2, 32, 16], lv: float16[2, 32, 16], lo: float16[2, 16]):
        s: float16[2, 32] = 0.0
        for h in range(2):
            for t in range(32):
                acc: float16 = 0.0
                for d in range(16):
                    acc += lq[h, d] * lk[h, t, d]
                s[h, t] = acc
        for {r} in range(2):
            total: float16 = 0.0
            for t in range(32):
                s[{r}, t] = exp(s[{r}, t])
            for t in range(32):
                total += s[{r}, t]
            for t in range(32):
                s[{r}, t] = {divide.replace("s[h, t]", f"s[{r}, t]")}
        for h in range(2):
            for d in range(16):
                acc2: float16 = 0.0
                for {z} in range(32):
                    acc2 += s[h, {z}] * lv[h, {z}, d]
                lo[h, d] = acc2
"""
    # The frontend reads the kernel's source with ``inspect``: the variant is
    # a real module file.
    import hashlib
    import importlib.util
    import tempfile

    del region, exp
    header = (
        "import allo\nfrom allo.dataflow import region\n"
        "from allo.ir.types import float16\nfrom allo.spmw_match import exp\n"
    )
    digest = hashlib.sha256(src.encode()).hexdigest()[:12]
    path = pathlib.Path(tempfile.gettempdir()) / f"attention_variant_{digest}.py"
    path.write_text(header + src)
    name = f"attention_variant_{digest}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.attention_variant


def _attacc_digest(compiled):
    import hashlib

    return hashlib.sha256(repr(list(compiled.compiled.cmds)).encode()).hexdigest()


def test_attacc_attention_rejects_the_reversed_division():
    """``total / s[h, t]`` is not the softmax: the facts retain the operand
    order and the proof checks numerator and denominator roles."""
    from allo.pim.attacc_tiling import AttAccDomainError

    with pytest.raises(AttAccDomainError, match="operands are 'total' / 's'"):
        _compile_attacc_with(_attention_source(divide="total / s[h, t]"))


@pytest.mark.parametrize(
    "rename",
    [
        {"context_loop": "z"},
        {"row_loop": "row"},
        {"context_loop": "z", "row_loop": "row"},
    ],
    ids=["context_loop", "softmax_row_loop", "both"],
)
def test_attacc_attention_accepts_alpha_renamed_loops(rename):
    """Renaming a loop (the context reduction, or the softmax row loop that
    is scoped independently of the MAC's row loop) does not change the
    program: the same inventory and the same cycles are selected.  Axis
    correspondence is by score-buffer dimension, loop scope by retained
    loop identity, never by source spelling."""
    reference = _compile_attacc_with(_attention_source())
    renamed = _compile_attacc_with(_attention_source(**rename))
    assert renamed.estimate().cycles == reference.estimate().cycles == 465
    assert _attacc_digest(renamed) == _attacc_digest(reference)


# --------------------------------------------------------------------------- #
# (q) Gemini-II launch recipes are bound to the contract and the partition
#     (review round 53: an internally inconsistent recipe passed costing,
#     selection and materialization until public packing failed)
# --------------------------------------------------------------------------- #


def transposed_dot_g2(a: int3[12, 32], b: int3[32, 7], z: int12[7, 12]):
    for m, n in allo.grid(12, 7):
        acc: int12 = 0
        for k in allo.reduction(32):
            acc += a[m, k] * b[k, n]
        z[n, m] = acc


def _damage_single_k1(monkeypatch, mutate):
    """Replace ``single/k1``'s launches through the enumerator (the fault the
    reviewer injected): ``mutate(launch) -> launch``."""
    from allo.pim import apu_g2_source_route as route

    original = route.enumerate_apu_g2_launch_plans

    def damaged(contract, **kwargs):
        plans, rejected = original(contract, **kwargs)
        for plan in plans:
            if plan.name == "single/k1":
                plan.launches = tuple(mutate(launch) for launch in plan.launches)
        return plans, rejected

    monkeypatch.setattr(route, "enumerate_apu_g2_launch_plans", damaged)


def _halve_recipe(launch):
    """A legal K = 16 recipe with its 10-bit output under a launch that still
    declares the reduction range (0, 32)."""
    return replace(
        launch,
        program=replace(
            launch.program,
            reduction_extent=launch.program.reduction_extent // 2,
            output_type=replace(launch.program.output_type, bits=launch.program.output_type.bits - 1),
        ),
    )


def test_transposed_gemini_ii_program_is_supported_unmodified():
    compiled = allo.compile(transposed_dot_g2, build_apu_g2_target(), apu_g2_cost, backend="virtual")
    assert compiled.plan.name == "single/k1" and compiled.estimate().cycles == 892
    (launch,) = compiled.plan.launches
    assert launch.program.reduction_extent == 32 and tuple(launch.reduction) == (0, 32)
    assert launch.final and launch.program.output_type.bits == 11
    compiled.pack(a=np.ones((12, 32), np.int8), b=np.ones((32, 7), np.int8), z=np.zeros((7, 12), np.int16))


def test_damaged_cheapest_recipe_is_rejected_and_the_intact_alternative_selected(monkeypatch):
    """The reviewer's probe: ``single/k1``'s recipe halved (K 16, 10-bit
    output) under an unchanged declared reduction (0, 32).  It is the
    cheapest estimate (700 against 1,494) and must be refused at
    materialization as a feasibility rejection, so the shared selection
    takes the intact ``single/k2``, whose artifact packs."""
    _damage_single_k1(monkeypatch, _halve_recipe)
    compiled = allo.compile(transposed_dot_g2, build_apu_g2_target(), apu_g2_cost, backend="virtual")
    assert compiled.plan.name == "single/k2" and compiled.estimate().cycles == 1494
    assert [launch.program.reduction_extent for launch in compiled.plan.launches] == [16, 16]
    assert [tuple(launch.reduction) for launch in compiled.plan.launches] == [(0, 16), (16, 32)]
    (kernel,) = compiled.placement_ranking.kernels
    assert [mode for _p, _c, mode in kernel.scores] == ["single/k1", "single/k2"]
    reasons = [reason for _position, reason in kernel.rejections]
    assert any("recipe the contract does not imply" in reason and "reduction_extent realized 16 vs derived 32" in reason for reason in reasons), reasons
    assert any("output_type" in reason for reason in reasons)
    compiled.pack(a=np.ones((12, 32), np.int8), b=np.ones((32, 7), np.int8), z=np.zeros((7, 12), np.int16))


def test_forcing_the_damaged_recipe_fails_at_compilation(monkeypatch):
    _damage_single_k1(monkeypatch, _halve_recipe)
    with pytest.raises(Exception, match="recipe the contract does not imply"):
        allo.compile(transposed_dot_g2, build_apu_g2_target(), apu_g2_cost, backend="virtual", layout="single/k1")


@pytest.mark.parametrize(
    "name, mutate, expected",
    [
        ("output geometry", lambda l: replace(l, program=replace(l.program, output_shape=(84,))), ("output_shape realized (84,) vs derived (7, 12)",)),
        ("final routing", lambda l: replace(l, final=False), ("final realized False vs derived True",)),
        # A first chunk declared chained is already refused by the shared
        # candidate contract (accumulator initialization), before realization.
        ("accumulator mode", lambda l: replace(l, accumulator="previous"), ("accumulator realized 'previous' vs derived 'none'", "starts its accumulator as 'previous'")),
    ],
)
def test_other_recipe_inconsistencies_are_rejected(monkeypatch, name, mutate, expected):
    """A flattened output geometry the contract does not imply, a final
    launch declared partial, or a first chunk declared chained are refused
    (the recipe record itself refuses an output width inconsistent with its
    own reduction extent, so that case cannot be constructed)."""
    _damage_single_k1(monkeypatch, mutate)
    compiled = allo.compile(transposed_dot_g2, build_apu_g2_target(), apu_g2_cost, backend="virtual")
    assert compiled.plan.name == "single/k2"
    (kernel,) = compiled.placement_ranking.kernels
    assert any(any(text in reason for text in expected) for _p, reason in kernel.rejections), (name, kernel.rejections)


def nearby_pair_g2(a: int3[5, 16], b: int3[5, 16], x: int3[16], z: int15[5]):
    for i in range(5):
        t: int12 = 0
        u: int12 = 0
        for k in range(16):
            t += a[i, k] * x[k]
            u += b[i, k] * x[k]
        z[i] = 3 * t + 2 * u


_PAIR_ARGS = dict(a=np.ones((5, 16), np.int8), b=np.full((5, 16), 2, np.int8), x=np.ones(16, np.int8), z=np.zeros(5, np.int16))


def _recipe_result(compiled):
    """The launch recipe's epilogue applied to the operands the public
    packing interleaves (the device's arithmetic, on the host)."""
    from allo.pim import apu_g2_source_route as route

    (launch,) = compiled.plan.launches
    dots = [route.apu_g2_stream_dots(s, _PAIR_ARGS) for s in compiled.contract.streams]
    lhs, rhs, _acc = route._launch_operands(compiled.contract, launch, dots, {})
    individual = np.sum(lhs.astype(np.int64) * rhs.astype(np.int64), axis=1)
    coefficients = np.array(launch.program.epilogue.coefficients, dtype=np.int64)
    return (individual.reshape(-1, 2) * coefficients).sum(axis=1)


def _reorder_streams(monkeypatch, swap_coefficients):
    from allo.pim import apu_g2_source_route as route
    from allo.pim.apu_g2_composed_contraction import APUG2DotEpilogue

    original = route.enumerate_apu_g2_launch_plans

    def reversed_streams(contract, **kwargs):
        plans, rejected = original(contract, **kwargs)
        for plan in plans:
            launches = []
            for launch in plan.launches:
                program = launch.program
                if swap_coefficients:
                    program = replace(program, epilogue=APUG2DotEpilogue(coefficients=tuple(reversed(program.epilogue.coefficients))))
                launches.append(replace(launch, streams=tuple(reversed(launch.streams)), program=program))
            plan.launches = tuple(launches)
        return plans, rejected

    monkeypatch.setattr(route, "enumerate_apu_g2_launch_plans", reversed_streams)


def test_pair_affine_coefficients_follow_the_stream_identities():
    """The ordinary pair ``z = 3 (a x) + 2 (b x)``: one fused launch with
    the streams in contract order and coefficients (3, 2); the recipe's
    epilogue on the packed operands equals the source result."""
    compiled = allo.compile(nearby_pair_g2, build_apu_g2_target(), apu_g2_cost, backend="virtual")
    compiled.pack(**_PAIR_ARGS)
    (launch,) = compiled.plan.launches
    assert compiled.plan.name == "fused/k1" and tuple(launch.streams) == (0, 1)
    assert launch.program.epilogue.coefficients == (3, 2)
    expected = 3 * (_PAIR_ARGS["a"].astype(np.int64) @ _PAIR_ARGS["x"]) + 2 * (_PAIR_ARGS["b"].astype(np.int64) @ _PAIR_ARGS["x"])
    np.testing.assert_array_equal(_recipe_result(compiled), expected)
    assert expected.tolist() == [112] * 5


def test_reversed_streams_with_unchanged_coefficients_are_rejected(monkeypatch):
    """The reviewer's probe: the launch's streams reversed through the
    enumerator while the recipe keeps coefficients (3, 2) would compute
    3 (b x) + 2 (a x) = 128 instead of 112.  The re-derived recipe binds
    the coefficients to the stream identities ((2, 3) in that order), so
    the only legal plan is refused at materialization and the compilation
    fails instead of packing a wrong program."""
    _reorder_streams(monkeypatch, swap_coefficients=False)
    with pytest.raises(Exception, match="recipe the contract does not imply.*epilogue"):
        allo.compile(nearby_pair_g2, build_apu_g2_target(), apu_g2_cost, backend="virtual")


def test_reversed_streams_with_matching_coefficients_are_accepted(monkeypatch):
    """A launch that places stream 1 first with the coefficients in that
    order ((2, 3)) is a correct realization: accepted, packs, and the
    recipe on the packed operands equals the source result."""
    _reorder_streams(monkeypatch, swap_coefficients=True)
    compiled = allo.compile(nearby_pair_g2, build_apu_g2_target(), apu_g2_cost, backend="virtual")
    compiled.pack(**_PAIR_ARGS)
    (launch,) = compiled.plan.launches
    assert tuple(launch.streams) == (1, 0) and launch.program.epilogue.coefficients == (2, 3)
    np.testing.assert_array_equal(_recipe_result(compiled), [112] * 5)


# --------------------------------------------------------------------------- #
# (r) A Gemini-II repetitions update keeps the validated plan (review round 57:
#     the setter re-enumerated plans and took a replacement by name without
#     the shared checks)
# --------------------------------------------------------------------------- #


def _reps_probe(a: int3[3, 32], b: int3[32, 5], z: int12[5, 3]):
    for m, n in allo.grid(3, 5):
        acc: int12 = 0
        for k in allo.reduction(32):
            acc += a[m, k] * b[k, n]
        z[n, m] = acc


_REPS_ARGS = dict(a=np.ones((3, 32), np.int8), b=np.ones((32, 5), np.int8), z=np.zeros((5, 3), np.int16))


def _partition(plan):
    return [(tuple(l.streams), tuple(l.coordinates), tuple(l.reduction), l.accumulator, l.final) for l in plan.launches]


def test_repetitions_update_does_not_consult_the_enumerator(monkeypatch):
    """The reviewer's probe: a corrupted enumeration (K halved, narrower
    output) during a repetitions update.  The update derives the launches
    from the validated plan's own partitions, so the corruption cannot
    reach the artifact: the recipe keeps K = 32, only the repetition count
    changes, the artifact packs, and the enumerator is not called."""
    from allo.pim import apu_g2_source_route as route

    compiled = allo.compile(_reps_probe, build_apu_g2_target(), apu_g2_cost, backend="virtual")
    before = _partition(compiled.plan); estimate = compiled.estimate().cycles
    calls = []

    def broken(contract, **kwargs):
        calls.append(1)
        raise AssertionError("the enumerator must not be consulted by a repetitions update")

    monkeypatch.setattr(route, "enumerate_apu_g2_launch_plans", broken)
    compiled.repetitions = 17
    assert not calls and compiled.repetitions == 17
    assert _partition(compiled.plan) == before
    (launch,) = compiled.plan.launches
    assert launch.program.reduction_extent == 32 and launch.program.repetitions == 17
    # The per-call estimate is independent of the profiling repetitions; the
    # plan is re-priced and the graph re-attached.
    assert compiled.estimate().cycles == estimate == compiled.plan.estimate_cycles
    assert compiled.execution_graph is compiled.plan.execution_graph
    compiled.pack(**_REPS_ARGS)
    assert all(l["program"]["repetitions"] == 17 for l in compiled.manifest()["selected"]["launches"])


@pytest.mark.parametrize("name, mutate", [
    ("halved reduction", lambda p: replace(p, reduction_extent=p.reduction_extent // 2, output_type=replace(p.output_type, bits=p.output_type.bits - 1))),
    ("flattened output geometry", lambda p: replace(p, output_shape=(15,))),
])
def test_repetitions_update_rejects_a_drifted_recipe_and_keeps_the_valid_plan(monkeypatch, name, mutate):
    """If the derivation itself returned a recipe differing from the
    validated launch in anything but repetitions, the update is refused and
    the previous plan, graph and repetition count stay installed."""
    from allo.pim import apu_g2_source_route as route

    compiled = allo.compile(_reps_probe, build_apu_g2_target(), apu_g2_cost, backend="virtual")
    before = _partition(compiled.plan); graph = compiled.execution_graph; estimate = compiled.estimate().cycles
    original = route.launch_program_for

    def drifted(contract, group, coordinates, reduction, *, repetitions):
        program, accumulator, final = original(contract, group, coordinates, reduction, repetitions=repetitions)
        return mutate(program), accumulator, final

    monkeypatch.setattr(route, "launch_program_for", drifted)
    with pytest.raises(ValueError, match="beyond its repetition count"):
        compiled.repetitions = 17
    assert compiled.repetitions == 256 and _partition(compiled.plan) == before
    assert compiled.execution_graph is graph and compiled.estimate().cycles == estimate
    assert compiled.plan.launches[0].program.repetitions == 256
    compiled.pack(**_REPS_ARGS)


def test_repetitions_update_rejects_reordered_pair_streams(monkeypatch):
    """A pair-affine plan whose derivation returned the coefficients in the
    wrong order during the update is refused (the validated launch keeps
    (3, 2)); the artifact's recipe still computes the source result."""
    from allo.pim import apu_g2_source_route as route
    from allo.pim.apu_g2_composed_contraction import APUG2DotEpilogue

    compiled = allo.compile(nearby_pair_g2, build_apu_g2_target(), apu_g2_cost, backend="virtual")
    original = route.launch_program_for

    def swapped(contract, group, coordinates, reduction, *, repetitions):
        program, accumulator, final = original(contract, group, coordinates, reduction, repetitions=repetitions)
        return replace(program, epilogue=APUG2DotEpilogue(coefficients=tuple(reversed(program.epilogue.coefficients)))), accumulator, final

    monkeypatch.setattr(route, "launch_program_for", swapped)
    with pytest.raises(ValueError, match="epilogue"):
        compiled.repetitions = 17
    (launch,) = compiled.plan.launches
    assert launch.program.epilogue.coefficients == (3, 2) and compiled.repetitions == 256
    compiled.pack(**_PAIR_ARGS)
    np.testing.assert_array_equal(_recipe_result(compiled), [112] * 5)


def test_repetitions_update_of_a_forced_nondefault_plan_is_valid():
    """A valid update of the forced two-launch plan: both launches keep
    their partition, each recipe carries the new count, the plan is
    re-priced, packing and provenance agree."""
    compiled = allo.compile(_reps_probe, build_apu_g2_target(), apu_g2_cost, backend="virtual", layout="single/k2")
    assert compiled.plan.name == "single/k2" and len(compiled.plan.launches) == 2 and compiled.selection == "forced"
    before = _partition(compiled.plan); estimate = compiled.estimate().cycles
    compiled.repetitions = 17
    assert compiled.plan.name == "single/k2" and _partition(compiled.plan) == before
    assert [l.program.repetitions for l in compiled.plan.launches] == [17, 17]
    assert [l.program.reduction_extent for l in compiled.plan.launches] == [16, 16]
    assert compiled.estimate().cycles == estimate == compiled.plan.estimate_cycles
    assert compiled.execution_graph is compiled.plan.execution_graph
    compiled.pack(**_REPS_ARGS)
    manifest = compiled.manifest()
    assert manifest["selected"]["name"] == "single/k2"
    assert all(l["program"]["repetitions"] == 17 for l in manifest["selected"]["launches"])


def test_repetitions_update_runs_the_shared_graph_check_before_installing(monkeypatch):
    """The reviewer's probe: a priced graph claiming one launch more than
    the forced two-launch plan declares.  The shared graph consistency check
    runs on the attached artifact's layout before anything is installed:
    the update is refused and the previous plan object, both retained graph
    references, the repetition count and the estimate stay as they were;
    on a valid update the check is invoked and passes."""
    from allo import spmw_candidate
    from allo.pim import apu_g2_source_route as route

    compiled = allo.compile(_reps_probe, build_apu_g2_target(), apu_g2_cost, backend="virtual", layout="single/k2")
    plan, graph, estimate = compiled.plan, compiled.execution_graph, compiled.estimate().cycles
    assert graph is compiled.plan.execution_graph and graph.metadata["launch_count"] == 2
    calls = []
    original_check = spmw_candidate.check_graph_consistency
    original_price = route.price_apu_g2_launch_plan

    def spy(placement, priced):
        calls.append(priced.metadata.get("launch_count"))
        return original_check(placement, priced)

    def inconsistent_price(*args, **kwargs):
        priced = original_price(*args, **kwargs)
        priced.metadata["launch_count"] += 1
        return priced

    monkeypatch.setattr(spmw_candidate, "check_graph_consistency", spy)
    monkeypatch.setattr(route, "price_apu_g2_launch_plan", inconsistent_price)
    with pytest.raises(ValueError, match="prices 3 launch"):
        compiled.repetitions = 19
    assert calls == [3]
    assert compiled.plan is plan and compiled.execution_graph is graph and compiled.plan.execution_graph is graph
    assert compiled.repetitions == 256 and compiled.estimate().cycles == estimate
    assert [l.program.repetitions for l in compiled.plan.launches] == [256, 256]
    compiled.pack(**_REPS_ARGS)
    # A valid update runs the same check and installs the consistent graph.
    monkeypatch.setattr(route, "price_apu_g2_launch_plan", original_price)
    compiled.repetitions = 19
    assert calls == [3, 2]
    assert compiled.repetitions == 19 and compiled.plan is not plan
    assert compiled.execution_graph is compiled.plan.execution_graph and compiled.execution_graph.metadata["launch_count"] == 2
    assert [l.program.repetitions for l in compiled.plan.launches] == [19, 19]


# --------------------------------------------------------------------------- #
# (u) The APU v1 grouped route of a dataflow region on the contract (review
# round 60, ISSUES X-28): candidates declare their effects from the shared
# facts, the emission is checked against them, the runtime binds by identity
# --------------------------------------------------------------------------- #


@allo.dataflow.region()
def grouped_gemv_region(a: float16[4, 8], b: float16[8], c: float16[4]):
    """The reviewer's round 60 source: four work items, one FP16 row dot
    product each, a zero local accumulator; the work parameters are named
    differently from the region's arguments."""

    @allo.work(mapping=[4], args=[a, b, c])
    def reduce_row(mat: float16[4, 8], vec: float16[8], dst: float16[4]):
        row = allo.get_wid()
        acc: float16 = 0
        for k in allo.reduction(8):
            acc += mat[row, k] * vec[k]
        dst[row] = acc


@allo.dataflow.region()
def grouped_gemv_region_renamed(lhs_in: float16[4, 8], rhs_in: float16[8], out: float16[4]):
    @allo.work(mapping=[4], args=[lhs_in, rhs_in, out])
    def one_row(A: float16[4, 8], x: float16[8], y: float16[4]):
        r = allo.get_wid()
        total: float16 = 0
        for j in allo.reduction(8):
            total += A[r, j] * x[j]
        y[r] = total


G_M, G_K, G_N = 16, 32, 8


@allo.dataflow.region()
def grouped_gemm_region(p: float16[G_M, G_K], q: float16[G_K, G_N], r_out: float16[G_M, G_N]):
    """A nearby supported program: each of the four work items owns a block of
    four rows of a dense FP16 GEMM (the runtime's device packing)."""

    @allo.work(mapping=[4], args=[p, q, r_out])
    def row_block(A: float16[G_M, G_K], B: float16[G_K, G_N], C: float16[G_M, G_N]):
        core = allo.get_wid()
        for i in allo.grid(G_M // 4):
            for j in allo.grid(G_N):
                acc: float16 = 0
                for k in allo.reduction(G_K):
                    acc += A[core * (G_M // 4) + i, k] * B[k, j]
                C[core * (G_M // 4) + i, j] = acc


@allo.dataflow.region()
def grouped_gemv_carried(a: float16[4, 8], b: float16[8], c: float16[4]):
    @allo.work(mapping=[4], args=[a, b, c])
    def reduce_row(mat: float16[4, 8], vec: float16[8], dst: float16[4]):
        row = allo.get_wid()
        for k in allo.reduction(8):
            dst[row] += mat[row, k] * vec[k]


@allo.dataflow.region()
def grouped_add_region(a: float16[4, 8], b: float16[4, 8], c: float16[4, 8]):
    @allo.work(mapping=[4], args=[a, b, c])
    def add_row(x: float16[4, 8], y: float16[4, 8], z: float16[4, 8]):
        row = allo.get_wid()
        for k in allo.grid(8):
            z[row, k] = x[row, k] + y[row, k]


def _compile_grouped(region, **kwargs):
    kwargs.setdefault("reduction_order", "reassociate")
    kwargs.setdefault("backend", "virtual")
    return allo.compile(region, build_apu_v1_target(), apu_v1_cost, **kwargs)


def test_apu_v1_grouped_candidates_declare_effects_from_the_shared_facts():
    """The reviewer's source compiles through public ``allo.compile``; every
    selected grouped placement declares coverage, layouts, transfers, the
    launch and the ABI derived from the work body's facts and the group
    layout, and the emitted body was checked against them."""
    from allo.spmw_candidate import EFFECTS_DERIVERS, EFFECTS_REQUIRED_TARGETS, candidate_effects

    assert "apu_v1" in EFFECTS_REQUIRED_TARGETS
    compiled = _compile_grouped(grouped_gemv_region)
    assert "apu_v1" in EFFECTS_DERIVERS  # registered by the route module
    assert compiled.estimate().cycles > 0
    inner = compiled.compiled
    assert len(inner.layout) == 4 and all(l.mode == "grouped_f16" for l in inner.layout)
    for placement in inner.layout:
        effects = candidate_effects(placement)
        assert effects is not None
        assert effects.coverage.operations == ("MAC",)
        assert effects.coverage.inputs == ("mat", "vec") and effects.coverage.outputs == ("dst",)
        assert effects.abi_inputs == ("mat", "vec") and effects.abi_outputs == ("dst",)
        layouts = {v.value: v for v in effects.value_layouts}
        assert (layouts["mat"].storage, layouts["mat"].axes, layouts["mat"].packing) == ("vr0", ("k",), "group_lanes:8192")
        assert (layouts["vec"].storage, layouts["vec"].axes) == ("vr1", ("k",))
        assert (layouts["dst"].storage, layouts["dst"].packing) == ("vr2", "group_head:8192")
        routes = {(t.value, t.direction): tuple(t.route) for t in effects.transfers}
        assert routes[("mat", "in")] == ("host_pack", "l4", "vmrs", "vr0")
        assert routes[("dst", "out")] == ("vr2", "vmrs", "l4", "host_gather")
        (launch,) = effects.launches
        assert launch.kind == "gvml_grouped_mac" and launch.operations == (0,)
        assert launch.instances == (0, 1) and launch.reduction == (0, 8)
        assert launch.accumulator == "zero"
        assert launch.covers == ("grouped_f16", 8192, 4, 4, 1, "virtual_only")
        assert {a[2] for a in effects.allocations} == {"vr", "vmrs"}
    # The realization the check compared: one SPMD body (work item 0) with the
    # declared loads, one multiply and one 8192-lane group sum, one store.
    realized = inner.layout_ctx.realized
    assert [e[1:] for e in realized] == [
        ("load", "x", "vr0", 0),
        ("load", "y", "vr1", 1),
        ("load", "acc", "vr2", 2),
        ("mul", "vr0", "vr1"),
        ("group_reduce", "vr2", "mac_tmp_vr", 8192),
        ("store", "acc", "vr2", 2),
    ]
    assert all(e[0] == (0,) for e in realized)
    assert "GVML_P2_8K" in "\n".join(inner.cmds)


def test_apu_v1_grouped_renamed_program_gets_the_same_decision():
    """A program differing only in spelling (region arguments and work
    parameters) gets the same placement, declaration (up to names) and
    estimate."""
    from allo.spmw_candidate import candidate_effects

    first = _compile_grouped(grouped_gemv_region)
    renamed = _compile_grouped(grouped_gemv_region_renamed)
    assert renamed.estimate().cycles == first.estimate().cycles
    a = candidate_effects(first.compiled.layout[0]).manifest()
    b = candidate_effects(renamed.compiled.layout[0]).manifest()
    assert a["launches"] == b["launches"]
    # Axes are the source's loop labels (``k`` there, ``j`` here): compared
    # by count; storage and packing are the decision.
    assert [(v["storage"], v["packing"], len(v["axes"])) for v in a["value_layouts"]] == [
        (v["storage"], v["packing"], len(v["axes"])) for v in b["value_layouts"]
    ]
    assert b["coverage"]["inputs"] == ["A", "x"] and b["coverage"]["outputs"] == ["y"]
    assert renamed.compiled.cmds == first.compiled.cmds


def test_apu_v1_grouped_gemm_declares_the_runtime_packing_and_batches():
    """A dense FP16 GEMM region has a device packing: the declaration's batch
    count is the packing's (rows split over four APUCs, one dot product per
    group), and the emitted body loops over exactly that many batches."""
    from allo.spmw_candidate import candidate_effects

    compiled = _compile_grouped(grouped_gemm_region)
    inner = compiled.compiled
    placement = inner.layout[0]
    effects = candidate_effects(placement)
    (launch,) = effects.launches
    # 16 rows over 4 APUCs = 4 rows x 8 columns = 32 dot products per APUC,
    # 4 groups of 8192 lanes per VR -> 8 batches.
    assert launch.covers == ("grouped_f16", 8192, 4, 4, 8, "gemm")
    assert launch.instances == (0, 32) and launch.reduction == (0, 32)
    assert placement.extra["n_out_tiles"] == 8 and inner.layout_ctx.vector_batches == 8
    layouts = {v.value: v for v in effects.value_layouts}
    assert layouts["A"].axes == ("i", "k") and layouts["B"].axes == ("k", "j")
    assert layouts["C"].axes == ("i", "j") and layouts["C"].packing == "group_head:8192"


def test_apu_v1_grouped_runtime_binds_the_callers_arrays_by_identity():
    """The grouped runtime packs the caller's arrays for the declared program:
    the work body's buffers are bound to the region's parameters by retained
    identity (``A`` is ``p``, ``B`` is ``q``, ``C`` is ``r_out``), each dot
    product occupies one group, the batch count is the declared one; a
    program without a device packing (the GEMV) is refused with the reason."""
    from allo.spmw_apu_v1 import _apu_v1_prepare_io

    compiled = _compile_grouped(grouped_gemm_region)
    inner = compiled.compiled
    assert inner.source_parameters == ("p", "q", "r_out")
    rng = np.random.default_rng(3)
    A = rng.standard_normal((G_M, G_K)).astype(np.float16)
    B = rng.standard_normal((G_K, G_N)).astype(np.float16)
    C = np.zeros((G_M, G_N), np.float16)
    normalized, specs, packing = _apu_v1_prepare_io(inner, {"p": A, "q": B, "r_out": C})
    assert sorted(normalized) == ["p", "q"]
    assert specs == {"r_out": ((4, 8, 32768), np.dtype(np.float16))}
    assert packing["logical_shape"] == (G_M, G_N) and packing["rows_per_core"] == 4
    lhs = normalized["p"].reshape(4, 8, 32768)
    rhs = normalized["q"].reshape(4, 8, 32768)
    # core 1, batch 0, group 2: output (row 4, column 2)
    assert np.array_equal(lhs[1, 0, 2 * 8192 : 2 * 8192 + G_K], A[4])
    assert np.array_equal(rhs[1, 0, 2 * 8192 : 2 * 8192 + G_K], B[:, 2])
    assert not lhs[1, 0, 2 * 8192 + G_K : 3 * 8192].any()
    with pytest.raises(ValueError, match="no argument for the work body's 'A'"):
        _apu_v1_prepare_io(inner, {"A": A, "B": B, "C": C})
    gemv = _compile_grouped(grouped_gemv_region).compiled
    with pytest.raises(NotImplementedError, match="device execution is not implemented") as error:
        _apu_v1_prepare_io(gemv, {"a": np.zeros((4, 8), np.float16), "b": np.zeros(8, np.float16), "c": np.zeros(4, np.float16)})
    assert "rank-2 operands and a rank-2 result; got mat[4, 8], vec[8] -> dst[4]" in str(error.value)


def test_apu_v1_grouped_restrictions_are_explicit():
    """Strict FP16 reduction (the route regroups), a carried destination (the
    route zeroes its accumulator) and a non-MAC work body are refused by
    name, before enumeration reaches the contract."""
    with pytest.raises(ReductionOrderError, match="gvml_add_subgrps"):
        _compile_grouped(grouped_gemv_region, reduction_order="strict")
    with pytest.raises(NotImplementedError, match="accumulator_init='carried'"):
        _compile_grouped(grouped_gemv_carried)
    with pytest.raises(NotImplementedError, match="declares no program for the operations \\['ADD'\\]"):
        _compile_grouped(grouped_add_region)


def test_apu_v1_grouped_emission_is_checked_against_the_declaration(monkeypatch):
    """Fault injection into the emission: a group-sum tree over half the
    declared group makes the emitted body differ from the declaration, and
    materialization refuses every candidate (the shared feasibility
    rejection) instead of installing it."""
    from allo.spmw_apu_v1 import APUv1Ctx

    original = APUv1Ctx.emit_group_reduce_f16

    def halved(self, dst, src):
        saved = self.group_size
        self.group_size = saved // 2
        try:
            original(self, dst, src)
        finally:
            self.group_size = saved

    monkeypatch.setattr(APUv1Ctx, "emit_group_reduce_f16", halved)
    with pytest.raises(Exception, match="reduces groups of \\[4096\\] lanes; the candidate declares 8192"):
        _compile_grouped(grouped_gemv_region)


def test_apu_v1_grouped_supplied_declaration_must_be_the_derivation():
    """A supplied placement whose declaration was edited (the matrix in
    another register) is refused by the contract before costing."""
    from allo.spmw_autoschedule import Placement
    from allo.spmw_candidate import candidate_effects

    compiled = _compile_grouped(grouped_gemv_region)
    placement = compiled.compiled.layout[0]
    effects = candidate_effects(placement)
    edited = replace(
        effects,
        value_layouts=tuple(
            replace(v, storage="vr5") if v.value == "mat" else v for v in effects.value_layouts
        ),
    )
    tampered = Placement(
        placements=dict(placement.placements),
        mode=placement.mode,
        extra={**placement.extra, "effects": edited},
        layout=placement.layout,
    )
    with pytest.raises(ValueError, match="declares 'mat' in storage 'vr5'"):
        _compile_grouped(grouped_gemv_region, layout=[tampered] * 4)
    # The same placement supplied without its declaration is re-declared
    # from the facts and compiles to the same program.
    handle_only = Placement(
        placements=dict(placement.placements),
        mode=placement.mode,
        extra={k: v for k, v in placement.extra.items() if k != "effects"},
        layout=placement.layout,
    )
    again = _compile_grouped(grouped_gemv_region, layout=[handle_only] * 4)
    assert again.compiled.cmds == compiled.compiled.cmds
    assert candidate_effects(again.compiled.layout[0]).manifest() == effects.manifest()


# The dense GEMM template is proven from the retained access relations of
# every work item of the kernel group (review round 61, ISSUES X-29): the
# same-shaped sources below differ from the template in one access or in the
# written domain, compile for the estimate only and are refused before
# scheduling when an executable is requested.


@allo.dataflow.region()
def grouped_gemm_reversed_k(p: float16[G_M, G_K], q: float16[G_K, G_N], r_out: float16[G_M, G_N]):
    @allo.work(mapping=[4], args=[p, q, r_out])
    def row_block(A: float16[G_M, G_K], B: float16[G_K, G_N], C: float16[G_M, G_N]):
        core = allo.get_wid()
        for i in allo.grid(G_M // 4):
            for j in allo.grid(G_N):
                acc: float16 = 0
                for k in allo.reduction(G_K):
                    acc += A[core * (G_M // 4) + i, k] * B[G_K - 1 - k, j]
                C[core * (G_M // 4) + i, j] = acc


@allo.dataflow.region()
def grouped_gemm_partial_rows(p: float16[G_M, G_K], q: float16[G_K, G_N], r_out: float16[G_M, G_N]):
    @allo.work(mapping=[4], args=[p, q, r_out])
    def row_block(A: float16[G_M, G_K], B: float16[G_K, G_N], C: float16[G_M, G_N]):
        core = allo.get_wid()
        for i in allo.grid(G_M // 8):
            for j in allo.grid(G_N):
                acc: float16 = 0
                for k in allo.reduction(G_K):
                    acc += A[core * (G_M // 4) + i, k] * B[k, j]
                C[core * (G_M // 4) + i, j] = acc


@allo.dataflow.region()
def grouped_gemm_partial_columns(p: float16[G_M, G_K], q: float16[G_K, G_N], r_out: float16[G_M, G_N]):
    @allo.work(mapping=[4], args=[p, q, r_out])
    def row_block(A: float16[G_M, G_K], B: float16[G_K, G_N], C: float16[G_M, G_N]):
        core = allo.get_wid()
        for i in allo.grid(G_M // 4):
            for j in allo.grid(G_N // 2):
                acc: float16 = 0
                for k in allo.reduction(G_K):
                    acc += A[core * (G_M // 4) + i, k] * B[k, j]
                C[core * (G_M // 4) + i, j] = acc


@allo.dataflow.region()
def grouped_gemm_repeated_row(p: float16[G_M, G_K], q: float16[G_K, G_N], r_out: float16[G_M, G_N]):
    @allo.work(mapping=[4], args=[p, q, r_out])
    def row_block(A: float16[G_M, G_K], B: float16[G_K, G_N], C: float16[G_M, G_N]):
        core = allo.get_wid()
        for i in allo.grid(G_M // 4):
            for j in allo.grid(G_N):
                acc: float16 = 0
                for k in allo.reduction(G_K):
                    acc += A[0, k] * B[k, j]
                C[core * (G_M // 4) + i, j] = acc


@allo.dataflow.region()
def grouped_gemm_permuted_rows(p: float16[G_M, G_K], q: float16[G_K, G_N], r_out: float16[G_M, G_N]):
    @allo.work(mapping=[4], args=[p, q, r_out])
    def row_block(A: float16[G_M, G_K], B: float16[G_K, G_N], C: float16[G_M, G_N]):
        core = allo.get_wid()
        for i in allo.grid(G_M // 4):
            for j in allo.grid(G_N):
                acc: float16 = 0
                for k in allo.reduction(G_K):
                    acc += A[G_M - 1 - (core * (G_M // 4) + i), k] * B[k, j]
                C[core * (G_M // 4) + i, j] = acc


@allo.dataflow.region()
def grouped_gemm_overlapping(p: float16[G_M, G_K], q: float16[G_K, G_N], r_out: float16[G_M, G_N]):
    """Every work item writes the same four rows (the work id is unused)."""

    @allo.work(mapping=[4], args=[p, q, r_out])
    def row_block(A: float16[G_M, G_K], B: float16[G_K, G_N], C: float16[G_M, G_N]):
        for i in allo.grid(G_M // 4):
            for j in allo.grid(G_N):
                acc: float16 = 0
                for k in allo.reduction(G_K):
                    acc += A[i, k] * B[k, j]
                C[i, j] = acc


@allo.dataflow.region()
def grouped_gemm_column_blocks(p: float16[G_M, G_K], q: float16[G_K, G_N], r_out: float16[G_M, G_N]):
    """A nearby supported partition: the work items own column blocks."""

    @allo.work(mapping=[4], args=[p, q, r_out])
    def column_block(A: float16[G_M, G_K], B: float16[G_K, G_N], C: float16[G_M, G_N]):
        core = allo.get_wid()
        for i in allo.grid(G_M):
            for j in allo.grid(G_N // 4):
                acc: float16 = 0
                for k in allo.reduction(G_K):
                    acc += A[i, k] * B[k, core * (G_N // 4) + j]
                C[i, core * (G_N // 4) + j] = acc


_NOT_THE_TEMPLATE = [
    (grouped_gemm_reversed_k, "operand 'B' is read at \\[-1 \\* k \\+ 31, j\\] along the reduction", 32),
    (grouped_gemm_partial_rows, "write 64 of the 128 elements of C\\[16, 8\\]", 16),
    (grouped_gemm_partial_columns, "write 64 of the 128 elements of C\\[16, 8\\]", 16),
    (grouped_gemm_repeated_row, "written at \\[i, j\\] while the operands are read at row 0 of 'A'", 32),
    (grouped_gemm_permuted_rows, "read at row -1 \\* i \\+ 15 of 'A'", 32),
    (grouped_gemm_overlapping, "work items \\(0,\\) and \\(1,\\) both write rows 0..3 x columns 0..7", 32),
]


@pytest.mark.parametrize(
    "region, reason, instances",
    _NOT_THE_TEMPLATE,
    ids=["reversed_k", "partial_rows", "partial_columns", "repeated_row", "permuted_rows", "overlapping"],
)
def test_apu_v1_grouped_non_template_regions_are_refused_for_execution(region, reason, instances):
    """Same buffers, same shapes, the same emitted body as the dense GEMM,
    different semantics: an executable backend is refused before scheduling
    with the failing condition; ``backend='virtual'`` keeps the estimate
    with the candidate declared ``virtual_only`` and the reason, and the
    runtime refuses to pack it."""
    from allo.spmw_apu_v1 import _apu_v1_prepare_io
    from allo.spmw_candidate import candidate_effects

    with pytest.raises(NotImplementedError, match=reason) as error:
        _compile_grouped(region, backend=None)
    assert "the device packs a dense FP16 GEMM" in str(error.value)
    with pytest.raises(NotImplementedError, match="no host functional model"):
        _compile_grouped(region, backend="functional")
    compiled = _compile_grouped(region, backend="virtual")
    placement = compiled.compiled.layout[0]
    effects = candidate_effects(placement)
    (launch,) = effects.launches
    assert launch.covers[-1] == "virtual_only" and launch.instances == (0, instances)
    assert placement.extra["device_packing"] == "virtual_only"
    import re

    assert re.search(reason, placement.extra["device_packing_reason"])
    arrays = {
        "p": np.zeros((G_M, G_K), np.float16),
        "q": np.zeros((G_K, G_N), np.float16),
        "r_out": np.zeros((G_M, G_N), np.float16),
    }
    with pytest.raises(NotImplementedError, match=reason):
        _apu_v1_prepare_io(compiled.compiled, arrays)


def test_apu_v1_grouped_template_is_proven_over_the_whole_kernel_group():
    """The proof reads every work item's facts: the canonical row-block
    partition and a column-block partition are the template (the blocks a
    disjoint cover of the output); the packing then implements the source
    (the products the emitted multiply and group sum imply equal A @ B)."""
    from allo.pim.apu_v1_grouped_route import (
        group_contractions,
        grouped_device_packing,
        grouped_geometry,
    )
    from allo.spmw_apu_v1 import _apu_v1_prepare_io
    from allo.spmw_candidate import candidate_effects

    rng = np.random.default_rng(5)
    A = rng.integers(-3, 4, size=(G_M, G_K)).astype(np.float16)
    B = rng.integers(-3, 4, size=(G_K, G_N)).astype(np.float16)
    for region, blocks in (
        (grouped_gemm_region, [(0, 4, 0, 8), (4, 4, 0, 8), (8, 4, 0, 8), (12, 4, 0, 8)]),
        (grouped_gemm_column_blocks, [(0, 16, 0, 2), (0, 16, 2, 2), (0, 16, 4, 2), (0, 16, 6, 2)]),
    ):
        compiled = _compile_grouped(region, backend=None)  # the board route
        inner = compiled.compiled
        placement = inner.layout[0]
        group = group_contractions(inner.trace.matches)
        assert [w for w, _f in group] == [(0,), (1,), (2,), (3,)]
        packing = grouped_device_packing(group, grouped_geometry(inner.target, placement))
        assert packing.kind == "gemm"
        assert [(b.row0, b.rows, b.col0, b.columns) for b in packing.template.blocks] == blocks
        assert (packing.template.lhs, packing.template.rhs, packing.template.output) == ("A", "B", "C")
        (launch,) = candidate_effects(placement).launches
        assert launch.covers[-1] == "gemm"
        assert launch.instances == (0, blocks[0][1] * blocks[0][3])
        normalized, specs, _packing = _apu_v1_prepare_io(inner, {"p": A, "q": B, "r_out": np.zeros((G_M, G_N), np.float16)})
        batches = specs["r_out"][0][1]
        x = normalized["p"].reshape(4, batches, 4, 8192).astype(np.float32)
        y = normalized["q"].reshape(x.shape).astype(np.float32)
        implied = (x * y).sum(axis=-1).reshape(4, -1)[:, : batches * 4]
        rows_per_core = 4
        gathered = implied.reshape(4, rows_per_core, G_N).reshape(G_M, G_N)
        np.testing.assert_array_equal(gathered, A.astype(np.float32) @ B.astype(np.float32))
    # The named-candidate path without a cost program stamps the group too.
    named = allo.compile(
        grouped_gemm_region, build_apu_v1_target(), None, layout="grouped_f16", backend="virtual", reduction_order="reassociate"
    )
    assert candidate_effects(named.compiled.layout[0]).launches[0].covers[-1] == "gemm"


# The batch count the device program loops over is validated against the
# proven packing at the shared boundary (review round 63, ISSUES X-30): a
# grouped candidate whose body loops over fewer batches than the packing
# needs is a feasibility rejection, never the selected program.


def _one_batch_clones(compiled, keep_effects=False):
    """The compiled region's grouped placements with the device loop cut to
    one batch; without their declarations (re-derived by the shared stage)
    unless ``keep_effects``."""
    from allo.spmw_autoschedule import Placement

    clones = []
    for original in compiled.compiled.layout:
        extra = {k: v for k, v in original.extra.items() if keep_effects or k != "effects"}
        extra["n_out_tiles"] = 1
        clones.append(
            Placement(placements=dict(original.placements), mode=original.mode, layout=original.layout, extra=extra)
        )
    return clones


def test_apu_v1_grouped_supplied_placement_with_too_few_batches_is_refused():
    """The dense 16 x 32 x 8 region needs eight batches per APUC.  Supplied
    placements with a one-batch device loop and no declaration are
    re-declared from the facts with the proven eight batches and refused at
    materialization: the body loops over one batch, the packing needs eight.
    Supplied with their original declaration edited to one batch they are
    refused by the contract before costing."""
    from allo.spmw_candidate import candidate_effects

    base = _compile_grouped(grouped_gemm_region, backend=None)
    assert base.compiled.layout[0].extra["n_out_tiles"] == 8
    with pytest.raises(Exception, match="loops over 1 batch\\(es\\); the candidate declares 8"):
        _compile_grouped(grouped_gemm_region, backend=None, layout=_one_batch_clones(base))
    edited = _one_batch_clones(base, keep_effects=True)
    for placement in edited:
        effects = candidate_effects(placement)
        (launch,) = effects.launches
        placement.extra["effects"] = replace(
            effects, launches=(replace(launch, covers=launch.covers[:4] + (1,) + launch.covers[5:]),)
        )
    with pytest.raises(ValueError, match="violates the candidate contract"):
        _compile_grouped(grouped_gemm_region, backend=None, layout=edited)


def test_apu_v1_grouped_search_rejects_an_under_batched_candidate(monkeypatch):
    """Fault injection into normal search: the enumerator offers a cheaper
    one-batch clone of every intact candidate, declared by the production
    deriver.  The clone's declaration carries the proven eight batches, its
    body loops over one, so materialization records a per-candidate
    feasibility rejection and the intact candidate is selected."""
    from allo import spmw_autoschedule as sa
    from allo.pim.apu_v1_grouped_route import apu_v1_grouped_candidate_effects
    from allo.spmw_autoschedule import Placement
    from allo.spmw_candidate import candidate_effects

    original = sa._ENUMERATORS["apu_v1"]

    def injected(target, matches):
        intact = original(target, matches)
        clones = []
        for placement in intact:
            clone = Placement(
                placements=dict(placement.placements),
                mode=placement.mode,
                layout=placement.layout,
                extra={**placement.extra, "n_out_tiles": 1},
            )
            clone.extra["effects"] = apu_v1_grouped_candidate_effects(target, matches, clone)
            clones.append(clone)
        return clones + intact

    monkeypatch.setitem(sa._ENUMERATORS, "apu_v1", injected)
    compiled = _compile_grouped(grouped_gemm_region, backend=None)
    assert compiled.compiled.layout[0].extra["n_out_tiles"] == 8
    assert compiled.compiled.layout_ctx.vector_batches == 8
    assert candidate_effects(compiled.compiled.layout[0]).launches[0].covers[4] == 8
    assert compiled.estimate().cycles == _compile_grouped(grouped_gemm_region, backend=None).estimate().cycles
    rejections = [reason for kernel in compiled.placement_ranking.kernels for _position, reason in kernel.rejections]
    assert any("loops over 1 batch(es); the candidate declares 8" in reason for reason in rejections), rejections


# The placement's selected map must be the map the grouped executable
# realizes (review round 65, ISSUES X-31): the GVML group tree and the host
# packing address contiguous groups, so an interleaved or collapsed map with
# the same group cardinalities is refused per candidate.


def _grouped_map_variants():
    from allo.spmw_linear_layout import LinearLayout

    interleaved = LinearLayout(
        bases={"group": [(1,), (2,)], "lane_in_group": [(1 << b,) for b in range(2, 15)]},
        out_dims=("vr_lane",),
        out_sizes=(32768,),
    )
    collapsed = LinearLayout(
        bases={"group": [(8192,), (16384,)], "lane_in_group": [(1,)] * 13},
        out_dims=("vr_lane",),
        out_sizes=(32768,),
    )
    return {"interleaved": interleaved, "collapsed": collapsed}


def _with_map(compiled, layout, keep_effects=False):
    from allo.spmw_autoschedule import Placement

    return [
        Placement(
            placements=dict(p.placements),
            mode=p.mode,
            layout=layout,
            extra={k: v for k, v in p.extra.items() if keep_effects or k != "effects"},
        )
        for p in compiled.compiled.layout
    ]


@pytest.mark.parametrize("variant", ["interleaved", "collapsed"])
def test_apu_v1_grouped_non_contiguous_map_is_refused_when_supplied(variant):
    """Supplied placements carrying an interleaved (group bits in the low
    physical lanes) or a collapsed (thirteen lane bits on one physical bit)
    map with the canonical cardinalities are refused by the contract before
    costing, with the executable's contiguous map named; the canonical map
    (with its movement annotations) still compiles."""
    from allo.pim.apu_v1_grouped_route import grouped_geometry, grouped_map_problem

    base = _compile_grouped(grouped_gemm_region, backend=None)
    canonical = base.compiled.layout[0]
    assert grouped_map_problem(canonical.layout, grouped_geometry(base.compiled.target, canonical)) is None
    layout = _grouped_map_variants()[variant]
    with pytest.raises(ValueError, match="the placement's selected map is invalid") as error:
        _compile_grouped(grouped_gemm_region, backend=None, layout=_with_map(base, layout))
    assert "contiguous groups" in str(error.value)
    # the same map with the canonical declaration copied onto it: the
    # derivation disagrees with the declaration
    with pytest.raises(ValueError, match="violates the candidate contract"):
        _compile_grouped(grouped_gemm_region, backend=None, layout=_with_map(base, layout, keep_effects=True))
    again = _compile_grouped(grouped_gemm_region, backend=None, layout=_with_map(base, canonical.layout))
    assert again.compiled.cmds == base.compiled.cmds


def test_apu_v1_grouped_search_rejects_non_contiguous_maps(monkeypatch):
    """Fault injection into normal search: the enumerator offers the
    interleaved and collapsed clones of every intact candidate, each
    declared by the production deriver.  Their packings are declared
    invalid, the contract filters them per candidate and the intact
    (canonical) candidate is selected and emitted."""
    from allo import spmw_autoschedule as sa
    from allo.pim.apu_v1_grouped_route import apu_v1_grouped_candidate_effects, grouped_map_problem, grouped_geometry
    from allo.spmw_autoschedule import Placement
    from allo.spmw_candidate import candidate_effects

    original = sa._ENUMERATORS["apu_v1"]
    variants = _grouped_map_variants()

    def injected(target, matches):
        intact = original(target, matches)
        clones = []
        for placement in intact:
            for layout in variants.values():
                clone = Placement(
                    placements=dict(placement.placements),
                    mode=placement.mode,
                    layout=layout,
                    extra={k: v for k, v in placement.extra.items() if k != "effects"},
                )
                clone.extra["effects"] = apu_v1_grouped_candidate_effects(target, matches, clone)
                assert all(v.packing.startswith("invalid:") for v in clone.extra["effects"].value_layouts)
                clones.append(clone)
        return clones + intact

    monkeypatch.setitem(sa._ENUMERATORS, "apu_v1", injected)
    compiled = _compile_grouped(grouped_gemm_region, backend=None)
    selected = compiled.compiled.layout[0]
    assert grouped_map_problem(selected.layout, grouped_geometry(compiled.compiled.target, selected)) is None
    assert [v[0] for v in selected.layout.bases["lane_in_group"]][:3] == [1, 2, 4]
    assert all(v.packing.startswith(("group_lanes", "group_head")) for v in candidate_effects(selected).value_layouts)
    reference = _compile_grouped(grouped_gemm_region, backend=None)
    assert compiled.compiled.cmds == reference.compiled.cmds
    assert compiled.estimate().cycles == reference.estimate().cycles
