# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench bicg: s = A^T r; q = A p. Shared workload spec (multi-stage, D2).

Single-output stage decomposition: stage1 `q = A @ p` (GEMV, rows M, reduction
N); stage2 `s = A^T @ r` (GEMV, rows N, reduction M). Two distinct `@allo.work`
funcs -> two buckets the pipeline lowers today. A is [M,N]; p is [N], r is [M].
Dims from lib.shapes (M=116, N=124).
"""

from __future__ import annotations

import allo
from allo.ir.types import float32 as fp16
from allo.dataflow import region as _df_region

from lib.shapes import shape

_S = shape("bicg")
M, N = _S["M"], _S["N"]


@_df_region()
def _bicg_top(A: fp16[M, N], p: fp16[N], r: fp16[M], q: fp16[M], s: fp16[N]):
    @allo.work(mapping=[1], args=[A, p, q])
    def ap(local_A: fp16[M, N], local_p: fp16[N], local_q: fp16[M]):
        for i in range(M):
            acc: fp16 = 0
            for j in range(N):
                acc += local_A[i, j] * local_p[j]
            local_q[i] = acc

    @allo.work(mapping=[1], args=[A, r, s])
    def atr(local_A: fp16[M, N], local_r: fp16[M], local_s: fp16[N]):
        for j in range(N):
            acc: fp16 = 0
            for i in range(M):
                acc += local_A[i, j] * local_r[i]
            local_s[j] = acc


def build():
    return _bicg_top


# (rows, reduction) per MAC stage.
STAGES = [(M, N), (N, M)]
