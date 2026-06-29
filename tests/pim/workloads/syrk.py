# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench syrk: C = alpha*A@A^T + beta*C, lower-triangular (j<=i). Shared spec.

The contraction is the matmul C[i,j] = sum_k A[i,k]*A[j,k] (k over M); the
triangular `j<=i` mask and the alpha/beta scale are applied host-side against
the numpy reference. The workload computes the full symmetric product (the
MAC-reduction the matcher lowers). Dims from lib.shapes (psize syrk SMALL:
A is [N,M] with N=80, M=60; C is [N,N]).
"""

from __future__ import annotations

import allo
from allo.ir.types import float32 as fp16
from allo.dataflow import region as _df_region

from lib.shapes import shape

_S = shape("syrk")
M, N = _S["M"], _S["N"]


@_df_region()
def _syrk_top(A: fp16[N, M], C: fp16[N, N]):
    @allo.work(mapping=[1], args=[A, C])
    def syrk_k(local_A: fp16[N, M], local_C: fp16[N, N]):
        for i in range(N):
            for j in range(N):
                acc: fp16 = 0
                for k in range(M):
                    acc += local_A[i, k] * local_A[j, k]
                local_C[i, j] = acc


def build():
    return _syrk_top


# (rows, reduction) per MAC stage -- the GEMV-harness geometry per Samsung
# layer / the UPMEM GEMV slot. Cycles-measurement shapes (zeros), not numerics.
STAGES = [(N, M)]
