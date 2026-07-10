# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Full-output uint16 SYRK specialization for APUg2 rank-N dot tiles."""

import allo
from allo.ir.types import uint16
from allo.pim.apu_g2_numeric import specialize_integer_ratio
from lib.shapes import shape


_SHAPE = shape("syrk")
M, N = _SHAPE["M"], _SHAPE["N"]
_COEFFICIENTS = specialize_integer_ratio({"alpha": 1.5, "beta": 1.2}).encoded
SYRK_ALPHA = _COEFFICIENTS["alpha"]
SYRK_BETA = _COEFFICIENTS["beta"]


def kernel_syrk(A: uint16[N, M], C: uint16[N, N]):
    for row, column in allo.grid(N, N):
        C[row, column] = C[row, column] * SYRK_BETA
        for depth in allo.reduction(M):
            C[row, column] += (
                SYRK_ALPHA * A[row, depth] * A[column, depth]
            )


def build():
    return kernel_syrk


STAGES = [(N * N, M)]
