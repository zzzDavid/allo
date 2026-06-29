# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench symm (Tier-2): C = alpha*B*A_sym + beta*C, A symmetric. Shared spec.

The MAC-reduction core the matcher lowers is the full product
`summ[i,j] = sum_k B[k,j]*A[i,k]` (k over M). The PolyBench triangular access
(`k<i` accumulation + the diagonal `A[i,i]` term), the alpha/beta scale, and the
running C update are applied host-side against `symm_np` -- the SAME decomposition
syrk uses (bare MAC product as the workload; triangular mask host-side). No
matcher change is needed: the bare contraction nest matches MAC on every backend.
Dims from lib.shapes (psize symm SMALL: A is [M,M], B/C are [M,N]; M=60, N=80).
"""

from __future__ import annotations

import allo
from allo.ir.types import float32 as fp16
from allo.dataflow import region as _df_region

from lib.shapes import shape

_S = shape("symm")
M, N = _S["M"], _S["N"]


@_df_region()
def _symm_top(A: fp16[M, M], B: fp16[M, N], S: fp16[M, N]):
    @allo.work(mapping=[1], args=[A, B, S])
    def symm_k(local_A: fp16[M, M], local_B: fp16[M, N], local_S: fp16[M, N]):
        for i in range(M):
            for j in range(N):
                acc: fp16 = 0
                for k in range(M):
                    acc += local_B[k, j] * local_A[i, k]
                local_S[i, j] = acc


def build():
    return _symm_top


# (rows, reduction) per MAC stage -- the GEMV-harness geometry per Samsung
# layer / the UPMEM GEMV slot. Cycles-measurement shapes (zeros), not numerics.
STAGES = [(M, M)]
