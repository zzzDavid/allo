# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Target-neutral packed XNOR/popcount contraction for APU v1."""

import allo
from allo.ir.types import int16


M = 32
N = 1024
K = 8


def vector_binary(left: int16[M, K], right: int16[K, N], result: int16[M, N]):
    for row, column in allo.grid(M, N):
        for depth in allo.reduction(K):
            result[row, column] += allo.popcount(
                ~(left[row, depth] ^ right[depth, column])
            )


def build():
    return vector_binary
