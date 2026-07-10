# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench covariance with explicit modular-uint16 semantics."""

import allo
from allo.ir.types import uint16
from lib.shapes import shape


_SHAPE = shape("covariance")
M, N = _SHAPE["M"], _SHAPE["N"]


def kernel_covariance(data: uint16[N, M], mean: uint16[M], cov: uint16[M, M]):
    sample_count: uint16 = N
    covariance_scale: uint16 = N - 1

    for column in allo.grid(M):
        mean[column] = 0

    for row, column in allo.grid(N, M):
        mean[column] += data[row, column]

    for column in allo.grid(M):
        mean[column] = mean[column] / sample_count

    for row, column in allo.grid(N, M):
        data[row, column] = data[row, column] - mean[column]

    for row, column in allo.grid(M, M):
        cov[row, column] = 0
        for depth in allo.reduction(N):
            cov[row, column] += data[depth, row] * data[depth, column]
        cov[row, column] = cov[row, column] / covariance_scale


def build():
    return kernel_covariance


STAGES = [(M, N), (M * M, N)]
