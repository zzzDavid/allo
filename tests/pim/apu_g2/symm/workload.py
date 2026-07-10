# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench SYMM as a lower-triangular symmetric uint16 dot-tile workload."""

import allo
from allo.ir.types import uint16
from allo.pim.apu_g2_numeric import specialize_integer_ratio
from lib.shapes import shape


_SHAPE = shape("symm")
M, N = _SHAPE["M"], _SHAPE["N"]
_COEFFICIENTS = specialize_integer_ratio({"alpha": 1.5, "beta": 1.2}).encoded
SYMM_ALPHA = _COEFFICIENTS["alpha"]
SYMM_BETA = _COEFFICIENTS["beta"]


def kernel_symm(A: uint16[M, M], B: uint16[M, N], C: uint16[M, N]):
    for row, column in allo.grid(M, N):
        C[row, column] = C[row, column] * SYMM_BETA
        for depth in allo.reduction(M):
            if depth <= row:
                C[row, column] += (
                    SYMM_ALPHA * A[row, depth] * B[depth, column]
                )
            else:
                C[row, column] += (
                    SYMM_ALPHA * A[depth, row] * B[depth, column]
                )


def build():
    return kernel_symm


STAGES = [(M * N, M)]
