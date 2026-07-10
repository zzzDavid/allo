# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structural gates for APUg2 uint16 correlation lowering."""

import math

import allo
import numpy as np

from allo.ir.types import uint16
from allo.pim.apu_g2_correlation_runtime import correlation_u16_reference
from allo.pim.apu_g2_vector_program import APUG2CorrelationCallable
from allo.pim.costs.apu_g2 import apu_g2_cost
from allo.pim.targets import build_apu_g2_target


M = 80
N = 100
SQRT_N = math.isqrt(N)


def ordinary_correlation(
    data_mean: uint16[N, M],
    data_stddev: uint16[N, M],
    data_for_center: uint16[N, M],
    corr: uint16[M, M],
):
    sample_count: uint16 = N
    sqrt_sample_count: uint16 = SQRT_N
    one: uint16 = 1
    mean: uint16[M] = 0
    std_centered: uint16[N, M] = 0
    variance: uint16[M] = 0
    stddev: uint16[M] = 0
    centered: uint16[N, M] = 0
    denominator: uint16[M] = 0
    normalized: uint16[N, M] = 0

    for row, column in allo.grid(N, M):
        mean[column] += data_mean[row, column]

    for column in allo.grid(M):
        mean[column] = mean[column] / sample_count

    for row, column in allo.grid(N, M):
        std_centered[row, column] = data_stddev[row, column] - mean[column]

    for column in allo.grid(M):
        for row in allo.reduction(N):
            variance[column] += (
                std_centered[row, column] * std_centered[row, column]
            )
        variance[column] = variance[column] / sample_count

    for column in allo.grid(M):
        stddev[column] = allo.sqrt(variance[column])
        if stddev[column] == 0:
            stddev[column] = one
        denominator[column] = sqrt_sample_count * stddev[column]

    for row, column in allo.grid(N, M):
        centered[row, column] = data_for_center[row, column] - mean[column]
        normalized[row, column] = centered[row, column] / denominator[column]

    for row, column in allo.grid(M, M):
        corr[row, column] = 0
        for depth in allo.reduction(N):
            corr[row, column] += (
                normalized[depth, row] * normalized[depth, column]
            )
        if row == column:
            corr[row, column] = 1


def test_correlation_u16_reference_uses_integer_sqrt_and_diagonal_select():
    rows = np.arange(N, dtype=np.uint64)[:, None]
    cols = np.arange(M, dtype=np.uint64)[None, :]
    data = ((rows * 251 + cols * 509 + 17) & 0xFFFF).astype(np.uint16)

    reference = correlation_u16_reference(data, data.copy(), data.copy())

    assert reference["mean"].shape == (M,)
    assert reference["stddev"].shape == (M,)
    assert reference["normalized"].shape == (N, M)
    assert reference["correlation"].shape == (M, M)
    np.testing.assert_array_equal(np.diag(reference["correlation"]), np.ones(M))
    assert np.all(reference["stddev"] >= 1)
    assert int(reference["sqrt_n"]) == 10


def test_public_compile_routes_correlation_to_structural_apu_g2_chain():
    target = build_apu_g2_target()
    compiled = allo.compile(
        ordinary_correlation,
        target,
        apu_g2_cost,
        backend="virtual",
    )

    assert isinstance(compiled, APUG2CorrelationCallable)
    assert compiled.backend == "virtual"
    assert compiled.execution_graph.metadata["program"] == "transport_aware_correlation_u16"
    assert compiled.execution_graph.metadata["work_grid"] == (16,)
    assert compiled.execution_graph.metadata["hardware_tasks"] == 17
    assert compiled.execution_graph.metadata["primitive_programs"] == {
        "dot_tile": 1,
        "column_batched_gemm": 2,
        "fill": 3,
        "div": 3,
        "sub": 2,
        "sqrt": 1,
        "minmax": 1,
        "mul": 1,
        "select_lt": 1,
    }
    assert compiled.execution_graph.metadata["tiling"]["corr_hardware_tasks"] == 3
    assert compiled.execution_graph.metadata["calibrated_device_ticks"] is False
    assert compiled.estimate().cycles > 0

    data = np.zeros((N, M), dtype=np.uint16)
    corr = np.full((M, M), 17, dtype=np.uint16)
    run = compiled(data, data.copy(), data.copy(), corr)
    assert run.backend == "virtual"
    assert run.cycles == compiled.estimate().cycles
    assert run.extra["outputs"] == {}
    assert run.extra["hardware_tasks"] == 17
    np.testing.assert_array_equal(corr, np.full((M, M), 17, dtype=np.uint16))
