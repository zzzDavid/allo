# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench Gemver as a structured uint16 VL64 chain for APUg2."""

import allo
from allo.ir.types import uint16
from allo.pim.apu_g2_numeric import specialize_integer_ratio
from lib.shapes import shape


_SHAPE = shape("gemver")
N = _SHAPE["N"]
_COEFFICIENTS = specialize_integer_ratio({"alpha": 0.1, "beta": 0.1}).encoded
GEMVER_ALPHA = _COEFFICIENTS["alpha"]
GEMVER_BETA = _COEFFICIENTS["beta"]


def kernel_gemver(
    A: uint16[N, N],
    u1: uint16[N],
    u2: uint16[N],
    v1: uint16[N],
    v2: uint16[N],
    x: uint16[N],
    y: uint16[N],
    w: uint16[N],
    z: uint16[N],
):
    for row, column in allo.grid(N, N):
        A[row, column] = A[row, column] + u1[row] * v1[column] + u2[row] * v2[column]

    for row, depth in allo.grid(N, N):
        x[row] = x[row] + A[depth, row] * y[depth]

    for row in allo.grid(N):
        x[row] = x[row] + z[row]

    for row, depth in allo.grid(N, N):
        w[row] = w[row] + A[row, depth] * x[depth]


def build():
    return kernel_gemver


STAGES = [(N * N, 2), (N, N), (N, 1), (N, N)]
