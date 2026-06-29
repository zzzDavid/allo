# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench syr2k: symmetric rank-2k, lower-triangular (j<=i). Shared spec.

C[i,j] += alpha*A[i,k]*B[j,k] + alpha*B[i,k]*A[j,k] (k over M). Single-output
stage decomposition: two MAC-reduction products (AB^T and BA^T) summed
host-side, with the triangular `j<=i` mask and alpha/beta scale applied against
the numpy reference. Dims from lib.shapes (psize syr2k SMALL: A,B are [N,M] with
N=80, M=60; C is [N,N]).
"""

from __future__ import annotations

import allo
from allo.ir.types import float32 as fp16
from allo.dataflow import region as _df_region

from lib.shapes import shape

_S = shape("syr2k")
M, N = _S["M"], _S["N"]


@_df_region()
def _syr2k_top(A: fp16[N, M], B: fp16[N, M], P1: fp16[N, N], P2: fp16[N, N]):
    @allo.work(mapping=[1], args=[A, B, P1])
    def prod_ab(local_A: fp16[N, M], local_B: fp16[N, M], local_P1: fp16[N, N]):
        for i in range(N):
            for j in range(N):
                acc: fp16 = 0
                for k in range(M):
                    acc += local_A[i, k] * local_B[j, k]
                local_P1[i, j] = acc

    @allo.work(mapping=[1], args=[B, A, P2])
    def prod_ba(local_B: fp16[N, M], local_A: fp16[N, M], local_P2: fp16[N, N]):
        for i in range(N):
            for j in range(N):
                acc: fp16 = 0
                for k in range(M):
                    acc += local_B[i, k] * local_A[j, k]
                local_P2[i, j] = acc


def build():
    return _syr2k_top


# (rows, reduction) per MAC stage -- the GEMV-harness geometry per Samsung
# layer / the UPMEM GEMV slot. Cycles-measurement shapes (zeros), not numerics.
STAGES = [(N, M), (N, M)]
