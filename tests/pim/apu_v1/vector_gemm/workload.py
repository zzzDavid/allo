# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-VR MICRO-shaped FP16 contraction for the vector-planning milestone."""

import allo
from allo.ir.types import float16

M = 1024
N = 1024
K = 64


def vector_gemm(left: float16[M, K], right: float16[K, N], result: float16[M, N]):
    for row, column in allo.grid(M, N):
        for depth in allo.reduction(K):
            result[row, column] += left[row, depth] * right[depth, column]


def build():
    return vector_gemm
