# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical PolyBench ATAX as one ordinary Allo uint16 workload."""

import allo
from allo.ir.types import uint16
from lib.shapes import shape


M = shape("atax")["M"]
N = shape("atax")["N"]


def stage_m(A: uint16[M, N], x: uint16[N], tmp: uint16[M]):
    for row in allo.grid(M):
        for depth in allo.reduction(N):
            tmp[row] += A[row, depth] * x[depth]


def stage_n(A: uint16[M, N], tmp: uint16[M], y: uint16[N]):
    for column in allo.grid(N):
        for depth in allo.reduction(M):
            y[column] += A[depth, column] * tmp[depth]


def kernel_atax(A: uint16[M, N], x: uint16[N], y: uint16[N]):
    tmp: uint16[M] = 0
    stage_m(A, x, tmp)
    stage_n(A, tmp, y)


def build():
    return kernel_atax
