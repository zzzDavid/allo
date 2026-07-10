# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical PolyBench 2mm as chained APUg2 uint16 contractions."""

import allo
from allo.ir.types import uint16
from allo.pim.apu_g2_numeric import specialize_integer_ratio
from lib.shapes import shape


_SHAPE = shape("2mm")
P, Q, R, S = _SHAPE["P"], _SHAPE["Q"], _SHAPE["R"], _SHAPE["S"]
_COEFFICIENTS = specialize_integer_ratio({"alpha": 0.1, "beta": 0.5}).encoded
TWO_MM_ALPHA = _COEFFICIENTS["alpha"]
TWO_MM_BETA = _COEFFICIENTS["beta"]


def mm1(A: uint16[P, Q], B: uint16[Q, R], AB: uint16[P, R]):
    for row, column in allo.grid(P, R):
        for depth in allo.reduction(Q):
            AB[row, column] += A[row, depth] * B[depth, column]


def mm2(AB: uint16[P, R], C: uint16[R, S], D: uint16[P, S]):
    for row, column in allo.grid(P, S):
        D[row, column] = D[row, column] * TWO_MM_ALPHA
        for depth in allo.reduction(R):
            D[row, column] += (
                TWO_MM_BETA * AB[row, depth] * C[depth, column]
            )


def kernel_2mm(A: uint16[P, Q], B: uint16[Q, R], C: uint16[R, S], D: uint16[P, S]):
    AB: uint16[P, R] = 0
    mm1(A, B, AB)
    mm2(AB, C, D)


def build():
    return kernel_2mm


STAGES = [(P * R, Q), (P * S, R)]
