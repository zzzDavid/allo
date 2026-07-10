# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""MLIR planning and calibrated cost checks for fused APUg2 ATAX."""

import inspect

import allo
import numpy as np
from allo.ir.types import uint16

from allo.pim.apu_g2_vector_program import APUG2AtaxCallable
from allo.pim.apu_g2_vectorize import plan_apu_g2_atax
from allo.pim.costs.apu_g2 import (
    APUG2_ATAX_FULL_PIPELINE_TICKS,
    APUG2_ATAX_MEASURED_TICKS_PER_PIPELINE,
    apu_g2_cost,
    build_apu_g2_atax_graph,
)
from allo.pim.targets import build_apu_g2_target


def stage_m(A: uint16[5, 7], x: uint16[7], tmp: uint16[5]):
    for row in allo.grid(5):
        for depth in allo.reduction(7):
            tmp[row] += A[row, depth] * x[depth]


def stage_n(A: uint16[5, 7], tmp: uint16[5], y: uint16[7]):
    for column in allo.grid(7):
        for depth in allo.reduction(5):
            y[column] += A[depth, column] * tmp[depth]


def ordinary_atax(A: uint16[5, 7], x: uint16[7], y: uint16[7]):
    tmp: uint16[5] = 0
    stage_m(A, x, tmp)
    stage_n(A, tmp, y)


def test_apu_g2_atax_plan_proves_linked_direct_and_transposed_gemvs():
    module = allo.customize(ordinary_atax, enable_tensor=False).module
    plan = plan_apu_g2_atax(module)

    assert plan.matrix.value == "A"
    assert plan.vector.value == "x"
    assert plan.intermediate.value == "tmp"
    assert plan.output.value == "y"
    assert (plan.row_extent, plan.column_extent) == (5, 7)
    assert plan.stage_m.matrix_transposed is False
    assert plan.stage_n.matrix_transposed is True


def test_public_compile_routes_two_contractions_to_one_resident_atax_task():
    target = build_apu_g2_target()
    compiled = allo.compile(ordinary_atax, target, apu_g2_cost, backend="virtual")

    assert isinstance(compiled, APUG2AtaxCallable)
    assert inspect.signature(compiled) == inspect.signature(ordinary_atax)
    assert compiled.estimate().cycles == APUG2_ATAX_FULL_PIPELINE_TICKS
    assert compiled.execution_graph.metadata["program"] == "atax_u16"
    assert compiled.execution_graph.metadata["hardware_tasks"] == 1
    assert compiled.execution_graph.metadata["resident_intermediate"] is True
    assert compiled.execution_graph.metadata["work_grid"] == (16,)
    assert compiled.execution_graph.metadata["vl64_calls"] == 80

    A = np.arange(35, dtype=np.uint16).reshape(5, 7)
    x = np.arange(7, dtype=np.uint16)
    y = np.full(7, 19, dtype=np.uint16)
    run = compiled(A, x, y)
    assert run.backend == "virtual"
    assert run.extra["hardware_tasks"] == 1
    assert np.all(y == 19)


def test_apu_g2_atax_cost_is_real_card_calibrated_and_core_wide():
    target = build_apu_g2_target()
    graph = build_apu_g2_atax_graph(
        target, apu_g2_cost, row_extent=116, column_extent=124
    )
    result = apu_g2_cost.bind(target).evaluate(graph)

    assert result.cycles == APUG2_ATAX_FULL_PIPELINE_TICKS
    assert APUG2_ATAX_MEASURED_TICKS_PER_PIPELINE == 76118.25
    assert graph.metadata["vl64_compute_calls"] == 56
    assert graph.metadata["resident_transform_calls"] == 24
    assert len(graph.activities) == 81
