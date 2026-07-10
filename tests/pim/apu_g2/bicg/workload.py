# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical PolyBench BiCG as one ordinary Allo uint16 workload."""

import allo
from allo.ir.types import uint16
from lib.shapes import shape


M = shape("bicg")["M"]
N = shape("bicg")["N"]


def stage_q(A: uint16[M, N], p: uint16[N], q: uint16[M]):
    for row in allo.grid(M):
        for depth in allo.reduction(N):
            q[row] += A[row, depth] * p[depth]


def stage_s(A: uint16[M, N], r: uint16[M], s: uint16[N]):
    for column in allo.grid(N):
        for depth in allo.reduction(M):
            s[column] += A[depth, column] * r[depth]


def kernel_bicg(
    A: uint16[M, N],
    p: uint16[N],
    r: uint16[M],
    q: uint16[M],
    s: uint16[N],
):
    stage_q(A, p, q)
    stage_s(A, r, s)


def build():
    return kernel_bicg
