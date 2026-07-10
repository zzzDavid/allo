# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integer-specialized PolyBench Doitgen as a batched contraction."""

import allo
from allo.ir.types import uint16
from lib.shapes import shape


Q = shape("doitgen")["Q"]
R = shape("doitgen")["R"]
P = shape("doitgen")["P"]
S = shape("doitgen")["S"]


def kernel_doitgen(
    A: uint16[R, Q, S],
    x: uint16[S, P],
    output: uint16[R, Q, P],
):
    for r, q, p in allo.grid(R, Q, P):
        for s in allo.reduction(S):
            output[r, q, p] += A[r, q, s] * x[s, p]


def build():
    return kernel_doitgen
