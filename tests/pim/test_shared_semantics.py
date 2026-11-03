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
    build_samsung_target,
)
from allo.spmw_match import MatchTrace, UnsupportedWorkloadError
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
    assert compiled.compiled.contract.manifest() == contract.manifest()
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
