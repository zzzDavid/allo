# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench 3mm: G = (A@B) @ (C@D). Shared workload spec.

Single-output stage decomposition: three chained GEMM stages (AB = A@B,
CD = C@D, G = AB@CD). Dims from lib.shapes (psize 'three_mm' SMALL:
P=40,Q=60,R=50,T=70,S=80). Canonical shapes: A[P,Q] B[Q,R] -> AB[P,R];
C[R,S] D[S,T] -> CD[R,T]; AB[P,R] CD[R,T] -> G[P,T].
"""

from __future__ import annotations

import allo
from allo.ir.types import float32 as fp16
from allo.dataflow import region as _df_region

from lib.shapes import shape

_S = shape("3mm")
P, Q, R, NT, S = _S["P"], _S["Q"], _S["R"], _S["T"], _S["S"]


@_df_region()
def _three_mm_top(
    A: fp16[P, Q], B: fp16[Q, R], C: fp16[R, S], D: fp16[S, NT],
    AB: fp16[P, R], CD: fp16[R, NT], G: fp16[P, NT],
):
    @allo.work(mapping=[1], args=[A, B, AB])
    def mm1(local_A: fp16[P, Q], local_B: fp16[Q, R], local_AB: fp16[P, R]):
        for i in range(P):
            for j in range(R):
                acc: fp16 = 0
                for k in range(Q):
                    acc += local_A[i, k] * local_B[k, j]
                local_AB[i, j] = acc

    @allo.work(mapping=[1], args=[C, D, CD])
    def mm2(local_C: fp16[R, S], local_D: fp16[S, NT], local_CD: fp16[R, NT]):
        for i in range(R):
            for j in range(NT):
                acc: fp16 = 0
                for k in range(S):
                    acc += local_C[i, k] * local_D[k, j]
                local_CD[i, j] = acc

    @allo.work(mapping=[1], args=[AB, CD, G])
    def mm3(local_AB: fp16[P, R], local_CD: fp16[R, NT], local_G: fp16[P, NT]):
        for i in range(P):
            for j in range(NT):
                acc: fp16 = 0
                for k in range(R):
                    acc += local_AB[i, k] * local_CD[k, j]
                local_G[i, j] = acc


def build():
    return _three_mm_top


# (rows, reduction) per MAC stage -- the GEMV-harness geometry per Samsung
# layer / the UPMEM GEMV slot. Cycles-measurement shapes (zeros), not numerics.
STAGES = [(P, Q), (R, S), (P, R)]
