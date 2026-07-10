# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench TRMM as an upper-triangular uint16 dot-tile workload."""

import allo
from allo.ir.types import uint16
from allo.pim.apu_g2_numeric import specialize_integer_ratio
from lib.shapes import shape


_SHAPE = shape("trmm")
M, N = _SHAPE["M"], _SHAPE["N"]
_COEFFICIENTS = specialize_integer_ratio({"alpha": 1.5}).encoded
TRMM_ALPHA = _COEFFICIENTS["alpha"]


def kernel_trmm(A: uint16[M, M], B: uint16[M, N]):
    for row, column in allo.grid(M, N):
        for depth in allo.reduction(M):
            if depth > row:
                B[row, column] += A[depth, row] * B[depth, column]
        B[row, column] = B[row, column] * TRMM_ALPHA


def build():
    return kernel_trmm


STAGES = [(M * N, M)]
