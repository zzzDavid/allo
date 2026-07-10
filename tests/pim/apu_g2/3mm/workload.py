# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical PolyBench 3mm as an APUg2 uint16 contraction DAG."""

import allo
from allo.ir.types import uint16
from lib.shapes import shape


_SHAPE = shape("3mm")
P, Q, R, NT, S = (
    _SHAPE["P"],
    _SHAPE["Q"],
    _SHAPE["R"],
    _SHAPE["T"],
    _SHAPE["S"],
)


def mm1(A: uint16[P, Q], B: uint16[Q, R], AB: uint16[P, R]):
    for row, column in allo.grid(P, R):
        for depth in allo.reduction(Q):
            AB[row, column] += A[row, depth] * B[depth, column]


def mm2(C: uint16[R, S], D: uint16[S, NT], CD: uint16[R, NT]):
    for row, column in allo.grid(R, NT):
        for depth in allo.reduction(S):
            CD[row, column] += C[row, depth] * D[depth, column]


def mm3(AB: uint16[P, R], CD: uint16[R, NT], output: uint16[P, NT]):
    for row, column in allo.grid(P, NT):
        for depth in allo.reduction(R):
            output[row, column] += AB[row, depth] * CD[depth, column]


def kernel_3mm(
    A: uint16[P, Q],
    B: uint16[Q, R],
    C: uint16[R, S],
    D: uint16[S, NT],
    output: uint16[P, NT],
):
    AB: uint16[P, R] = 0
    CD: uint16[R, NT] = 0
    mm1(A, B, AB)
    mm2(C, D, CD)
    mm3(AB, CD, output)


def build():
    return kernel_3mm


STAGES = [(P * R, Q), (R * NT, S), (P * NT, R)]
