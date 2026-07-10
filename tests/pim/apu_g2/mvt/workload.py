# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ordinary Allo uint16 contractions for the two PolyBench MVT stages."""

import allo
from allo.ir.types import uint16
from lib.shapes import shape


N = shape("mvt")["N"]


def stage_a(A: uint16[N, N], y1: uint16[N], x1: uint16[N]):
    for row in allo.grid(N):
        for depth in allo.reduction(N):
            x1[row] += A[row, depth] * y1[depth]


def stage_b(A: uint16[N, N], y2: uint16[N], x2: uint16[N]):
    for row in allo.grid(N):
        for depth in allo.reduction(N):
            x2[row] += A[depth, row] * y2[depth]


def kernel_mvt(
    A: uint16[N, N],
    y1: uint16[N],
    y2: uint16[N],
    x1: uint16[N],
    x2: uint16[N],
):
    stage_a(A, y1, x1)
    stage_b(A, y2, x2)


def build():
    return kernel_mvt


STAGES = [(N, N), (N, N)]
