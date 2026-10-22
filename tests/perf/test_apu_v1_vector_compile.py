# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect

import allo
import numpy as np
from allo.ir.types import float16
from allo.pim.apu_v1_vector_program import APUv1VectorCallable
from allo.pim.costs import apu_v1_cost
from allo.pim.targets import build_apu_v1_target


def ordinary_contraction(
    left: float16[4, 8], right: float16[8, 6], result: float16[4, 6]
):
    for row, column in allo.grid(4, 6):
        for depth in allo.reduction(8):
            result[row, column] += left[row, depth] * right[depth, column]


def test_public_compile_generates_ranks_and_executes_four_apu_plans():
    target = build_apu_v1_target()
    compiled = allo.compile(
        ordinary_contraction,
        target,
        apu_v1_cost,
        backend="functional",
    )

    assert isinstance(compiled, APUv1VectorCallable)
    assert inspect.signature(compiled) == inspect.signature(ordinary_contraction)
    assert [candidate.name for candidate in compiled.candidates] == [
        "baseline_spatial_reduction",
        "temporal_svp",
        "temporal_dma_coalescing",
        "temporal_dma_coalescing_broadcast_friendly",
    ]
    assert {plan.name for plan in compiled.ranked_plans} == {
        candidate.name for candidate in compiled.candidates
    }
    assert compiled.selected_plan is compiled.ranked_plans[0]
    assert compiled.estimate().cycles > 0
    assert compiled.execution_graph.metadata["plan"] == compiled.selected_plan.name
    assert compiled.realization is not None, compiled.realization_error
    assert "gvml_" in compiled.device_source()

    rng = np.random.default_rng(0)
    left = rng.normal(size=(4, 8)).astype(np.float16)
    right = rng.normal(size=(8, 6)).astype(np.float16)
    result = np.zeros((4, 6), dtype=np.float16)
    reference = left @ right
    run = compiled(left, right, result)

    np.testing.assert_allclose(result, reference, rtol=2e-3, atol=2e-3)
    np.testing.assert_array_equal(run.extra["outputs"]["result"], result)
    assert run.cycles == compiled.estimate().cycles
    assert run.extra["plan"] == compiled.selected_plan.name


def test_public_compile_accepts_candidate_name_and_plan_object():
    target = build_apu_v1_target()
    named = allo.compile(
        ordinary_contraction,
        target,
        apu_v1_cost,
        backend="virtual",
        layout="baseline_spatial_reduction",
    )
    assert named.selected_plan.name == "baseline_spatial_reduction"
    assert named.estimate("temporal_svp").cycles > 0

    explicit = allo.compile(
        ordinary_contraction,
        target,
        apu_v1_cost,
        backend="functional",
        layout=named.plans[1],
    )
    assert explicit.selected_plan.name == "temporal_svp"
