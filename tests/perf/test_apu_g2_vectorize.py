# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""MLIR recognition and public compilation for APUg2 uint16 GEMV."""

import inspect

import allo
import numpy as np
import pytest
from allo.ir.types import float16, uint16

from allo.pim.apu_g2_vector_program import APUG2GemvCallable
from allo.pim.apu_g2_vectorize import (
    UnsupportedAPUG2ContractionError,
    plan_apu_g2_gemv,
)
from allo.pim.costs.apu_g2 import apu_g2_cost
from allo.pim.targets import build_apu_g2_target


def ordinary_gemv(A: uint16[5, 7], x: uint16[7], result: uint16[5]):
    for row in allo.grid(5):
        for depth in allo.reduction(7):
            result[row] += A[row, depth] * x[depth]


def transposed_gemv(A: uint16[7, 5], x: uint16[7], result: uint16[5]):
    for row in allo.grid(5):
        for depth in allo.reduction(7):
            result[row] += A[depth, row] * x[depth]


def floating_gemv(A: float16[5, 7], x: float16[7], result: float16[5]):
    for row in allo.grid(5):
        for depth in allo.reduction(7):
            result[row] += A[row, depth] * x[depth]


def matrix_multiply(A: uint16[3, 7], B: uint16[7, 5], result: uint16[3, 5]):
    for row, column in allo.grid(3, 5):
        for depth in allo.reduction(7):
            result[row, column] += A[row, depth] * B[depth, column]


def _module(function):
    return allo.customize(function, enable_tensor=False).module


def test_apu_g2_gemv_plan_is_derived_from_mlir_roles_and_layout():
    plan = plan_apu_g2_gemv(_module(ordinary_gemv))

    assert plan.matrix.value == "A"
    assert plan.vector.value == "x"
    assert plan.output.value == "result"
    assert (plan.output_extent, plan.reduction_extent) == (5, 7)
    assert plan.matrix_transposed is False
    assert plan.layout.padded_output_extent == 8
    assert plan.layout.padded_reduction_extent == 8
    assert plan.log_block_size == 3
    assert plan.layout.stream_extent == 1


def test_apu_g2_gemv_plan_recognizes_transposed_matrix_access():
    plan = plan_apu_g2_gemv(_module(transposed_gemv))
    assert plan.matrix.indices == ("depth", "row")
    assert plan.matrix_transposed is True
    assert (plan.output_extent, plan.reduction_extent) == (5, 7)


def test_apu_g2_gemv_plan_rejects_other_types_and_contraction_ranks():
    with pytest.raises(UnsupportedAPUG2ContractionError, match="uint16"):
        plan_apu_g2_gemv(_module(floating_gemv))
    with pytest.raises(UnsupportedAPUG2ContractionError, match="one output axis"):
        plan_apu_g2_gemv(_module(matrix_multiply))


@pytest.mark.parametrize(
    "workload,transposed", [(ordinary_gemv, False), (transposed_gemv, True)]
)
def test_public_compile_routes_ordinary_allo_gemv_to_apu_g2(workload, transposed):
    compiled = allo.compile(
        workload,
        build_apu_g2_target(),
        apu_g2_cost,
        backend="virtual",
    )

    assert isinstance(compiled, APUG2GemvCallable)
    assert inspect.signature(compiled) == inspect.signature(workload)
    assert compiled.plan.matrix_transposed is transposed
    assert compiled.execution_graph.metadata["program"] == "gemv_u16"
    assert compiled.execution_graph.metadata["vl64_calls"] == 11
    assert compiled.execution_graph.metadata["work_grid"] == (16,)
    assert compiled.estimate().cycles > 0

    matrix_shape = (7, 5) if transposed else (5, 7)
    A = np.arange(np.prod(matrix_shape), dtype=np.uint16).reshape(matrix_shape)
    x = np.arange(7, dtype=np.uint16)
    result = np.full(5, 19, dtype=np.uint16)
    run = compiled(A, x, result)
    assert run.backend == "virtual"
    assert run.extra["outputs"] == {}
    assert np.all(result == 19)
