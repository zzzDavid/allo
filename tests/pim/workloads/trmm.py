# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench trmm (Tier-2): B = alpha * triangular(A^T) @ B (in-place). Shared spec.

The MAC-reduction core the matcher lowers is the full product
`Bout[i,j] = sum_k A[k,i]*B[k,j]` (k over M -- an A^T@B contraction). The
PolyBench triangular access (`k>i` accumulation), the `+B[i,j]` self-term, and
the alpha scale are applied host-side against `trmm_np` -- the SAME decomposition
syrk uses (bare MAC product as the workload; triangular mask host-side). No
matcher change is needed: the bare contraction nest matches MAC on every backend.
Dims from lib.shapes (psize trmm SMALL: A is [M,M], B is [M,N]; M=60, N=80).
"""

from __future__ import annotations

import allo
from allo.ir.types import float32 as fp16
from allo.dataflow import region as _df_region

from lib.shapes import shape

_S = shape("trmm")
M, N = _S["M"], _S["N"]


@_df_region()
def _trmm_top(A: fp16[M, M], B: fp16[M, N], Bout: fp16[M, N]):
    @allo.work(mapping=[1], args=[A, B, Bout])
    def trmm_k(local_A: fp16[M, M], local_B: fp16[M, N], local_Bout: fp16[M, N]):
        for i in range(M):
            for j in range(N):
                acc: fp16 = 0
                for k in range(M):
                    acc += local_A[k, i] * local_B[k, j]
                local_Bout[i, j] = acc


def build():
    return _trmm_top


# (rows, reduction) per MAC stage -- the GEMV-harness geometry per Samsung
# layer / the UPMEM GEMV slot. Cycles-measurement shapes (zeros), not numerics.
STAGES = [(M, M)]
