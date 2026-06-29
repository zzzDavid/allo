# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench gesummv: y = alpha*(A@x) + beta*(B@x). Shared workload spec.

Single-output stage decomposition: two GEMV stages (tmp = A@x, y = B@x). The
alpha/beta axpy combine is applied host-side against the numpy reference (the
two contractions are what the matcher lowers). Dims from lib.shapes (N=90).
"""

from __future__ import annotations

import allo
from allo.ir.types import float32 as fp16
from allo.dataflow import region as _df_region

from lib.shapes import shape

N = shape("gesummv")["N"]


@_df_region()
def _gesummv_top(A: fp16[N, N], B: fp16[N, N], x: fp16[N], tmp: fp16[N], y: fp16[N]):
    @allo.work(mapping=[1], args=[A, x, tmp])
    def gv_a(local_A: fp16[N, N], local_x: fp16[N], local_tmp: fp16[N]):
        for i in range(N):
            acc: fp16 = 0
            for j in range(N):
                acc += local_A[i, j] * local_x[j]
            local_tmp[i] = acc

    @allo.work(mapping=[1], args=[B, x, y])
    def gv_b(local_B: fp16[N, N], local_x: fp16[N], local_y: fp16[N]):
        for i in range(N):
            acc: fp16 = 0
            for j in range(N):
                acc += local_B[i, j] * local_x[j]
            local_y[i] = acc


def build():
    return _gesummv_top


# (rows, reduction) per MAC stage -- the GEMV-harness geometry per Samsung
# layer / the UPMEM GEMV slot. Cycles-measurement shapes (zeros), not numerics.
STAGES = [(N, N), (N, N)]
