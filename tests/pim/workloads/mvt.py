# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench mvt: x1 += A y1; x2 += A^T y2. Shared workload spec (multi-stage, D2).

Single-output stage decomposition (the shipped stageA/stageB): stage1
`x1 = A @ y1` (GEMV), stage2 `x2 = A^T @ y2` (GEMV). Two distinct `@allo.work`
funcs -> two buckets. (The canonical `+=` onto x1/x2 is folded host-side against
the numpy reference; the contraction is what the matcher lowers.) A is [N,N].
Dims from lib.shapes (N=120).
"""

from __future__ import annotations

import allo
from allo.ir.types import float32 as fp16
from allo.dataflow import region as _df_region

from lib.shapes import shape

N = shape("mvt")["N"]


@_df_region()
def _mvt_top(A: fp16[N, N], y1: fp16[N], y2: fp16[N], x1: fp16[N], x2: fp16[N]):
    @allo.work(mapping=[1], args=[A, y1, x1])
    def stage_a(local_A: fp16[N, N], local_y1: fp16[N], local_x1: fp16[N]):
        for i in range(N):
            acc: fp16 = 0
            for j in range(N):
                acc += local_A[i, j] * local_y1[j]
            local_x1[i] = acc

    @allo.work(mapping=[1], args=[A, y2, x2])
    def stage_b(local_A: fp16[N, N], local_y2: fp16[N], local_x2: fp16[N]):
        for i in range(N):
            acc: fp16 = 0
            for j in range(N):
                acc += local_A[j, i] * local_y2[j]
            local_x2[i] = acc


def build():
    return _mvt_top


# (rows, reduction) per MAC stage.
STAGES = [(N, N), (N, N)]
