# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A logical reduction wider than one proven APUg2 SUM block."""

import allo
from allo.ir.types import uint16


M = 64
K = 300


def wide_gemv(A: uint16[M, K], x: uint16[K], y: uint16[M]):
    for row in allo.grid(M):
        for depth in allo.reduction(K):
            y[row] += A[row, depth] * x[depth]


def build():
    return wide_gemv
