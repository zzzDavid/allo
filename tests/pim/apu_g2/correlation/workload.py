# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench correlation with explicit modular-uint16 semantics."""

import math

import allo
from allo.ir.types import uint16
from lib.shapes import shape


_SHAPE = shape("correlation")
M, N = _SHAPE["M"], _SHAPE["N"]
SQRT_N = math.isqrt(N)


def kernel_correlation(
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


def build():
    return kernel_correlation


STAGES = [(M, N), (M, N), (M, N), (M * M, N)]
