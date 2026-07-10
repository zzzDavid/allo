# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ordinary Allo uint16 SMALL GEMM for APUg2 rank-N contraction lowering."""

import allo
from allo.ir.types import uint16
from lib.shapes import shape


_SHAPE = shape("gemm")
P, Q, R = _SHAPE["P"], _SHAPE["Q"], _SHAPE["R"]


def kernel_gemm(A: uint16[P, Q], B: uint16[Q, R], C: uint16[P, R]):
    for row, column in allo.grid(P, R):
        for depth in allo.reduction(Q):
            C[row, column] += A[row, depth] * B[depth, column]


def build():
    return kernel_gemm


STAGES = [(P * R, Q)]
