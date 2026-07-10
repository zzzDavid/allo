# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Full-output uint16 SYR2K specialization for APUg2 rank-N dot tiles."""

import allo
from allo.ir.types import uint16
from allo.pim.apu_g2_numeric import specialize_integer_ratio
from lib.shapes import shape


_SHAPE = shape("syr2k")
M, N = _SHAPE["M"], _SHAPE["N"]
_COEFFICIENTS = specialize_integer_ratio({"alpha": 1.5, "beta": 1.2}).encoded
SYR2K_ALPHA = _COEFFICIENTS["alpha"]
SYR2K_BETA = _COEFFICIENTS["beta"]


def stage_ab(A: uint16[N, M], B: uint16[N, M], C: uint16[N, N]):
    for row, column in allo.grid(N, N):
        C[row, column] = C[row, column] * SYR2K_BETA
        for depth in allo.reduction(M):
            C[row, column] += (
                SYR2K_ALPHA * A[row, depth] * B[column, depth]
            )


def stage_ba(A: uint16[N, M], B: uint16[N, M], C: uint16[N, N]):
    for row, column in allo.grid(N, N):
        for depth in allo.reduction(M):
            C[row, column] += (
                SYR2K_ALPHA * B[row, depth] * A[column, depth]
            )


def kernel_syr2k(A: uint16[N, M], B: uint16[N, M], C: uint16[N, N]):
    stage_ab(A, B, C)
    stage_ba(A, B, C)


def build():
    return kernel_syr2k


STAGES = [(N * N, M), (N * N, M)]
