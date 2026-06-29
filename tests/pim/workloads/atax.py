# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench atax: y = A^T (A x). Shared workload spec (multi-stage, D2).

Single-output stage decomposition (the validated reference decomposition):
stage1 `tmp = A @ x` (GEMV, rows M, reduction N); stage2 `y = A^T @ tmp` (GEMV,
rows N, reduction M). Two distinct `@allo.work` funcs -> two buckets the
pipeline lowers today (no matcher change). Dims from lib.shapes (M=116, N=124).
"""

from __future__ import annotations

import allo
from allo.ir.types import float32 as fp16
from allo.dataflow import region as _df_region

from lib.shapes import shape

_S = shape("atax")
M, N = _S["M"], _S["N"]


@_df_region()
def _atax_top(A: fp16[M, N], x: fp16[N], tmp: fp16[M], y: fp16[N]):
    @allo.work(mapping=[1], args=[A, x, tmp])
    def ax(local_A: fp16[M, N], local_x: fp16[N], local_tmp: fp16[M]):
        for i in range(M):
            acc: fp16 = 0
            for j in range(N):
                acc += local_A[i, j] * local_x[j]
            local_tmp[i] = acc

    @allo.work(mapping=[1], args=[A, tmp, y])
    def atx(local_A: fp16[M, N], local_tmp: fp16[M], local_y: fp16[N]):
        for j in range(N):
            acc: fp16 = 0
            for i in range(M):
                acc += local_A[i, j] * local_tmp[i]
            local_y[j] = acc


def build():
    return _atax_top


# (rows, reduction) per MAC stage -- the GEMV-harness geometry.
STAGES = [(M, N), (N, M)]
