# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench 2mm: D = alpha*(A@B)@C + beta*D. Shared workload spec.

Single-output stage decomposition: two chained GEMM stages (AB = A@B, then
ABC = AB@C). alpha/beta applied host-side against the numpy reference. Dims from
lib.shapes (psize 'two_mm' SMALL: P=40,Q=70,R=50,S=80).
"""

from __future__ import annotations

import allo
from allo.ir.types import float32 as fp16
from allo.dataflow import region as _df_region

from lib.shapes import shape

_S = shape("2mm")
P, Q, R, S = _S["P"], _S["Q"], _S["R"], _S["S"]


@_df_region()
def _two_mm_top(
    A: fp16[P, Q], B: fp16[Q, R], C: fp16[R, S], AB: fp16[P, R], D: fp16[P, S]
):
    @allo.work(mapping=[1], args=[A, B, AB])
    def mm1(local_A: fp16[P, Q], local_B: fp16[Q, R], local_AB: fp16[P, R]):
        for i in range(P):
            for j in range(R):
                acc: fp16 = 0
                for k in range(Q):
                    acc += local_A[i, k] * local_B[k, j]
                local_AB[i, j] = acc

    @allo.work(mapping=[1], args=[AB, C, D])
    def mm2(local_AB: fp16[P, R], local_C: fp16[R, S], local_D: fp16[P, S]):
        for i in range(P):
            for j in range(S):
                acc: fp16 = 0
                for k in range(R):
                    acc += local_AB[i, k] * local_C[k, j]
                local_D[i, j] = acc


def build():
    return _two_mm_top


# (rows, reduction) per MAC stage -- the GEMV-harness geometry per Samsung
# layer / the UPMEM GEMV slot. Cycles-measurement shapes (zeros), not numerics.
STAGES = [(P, Q), (P, R)]
