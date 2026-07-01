# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench atax on SK hynix GDDR6-AiM -- SPMW slice form (spec 05).

y = A^T (A x). Two stages: tmp = A@x (slice the M output rows), y = A^T@tmp
(slice the N output rows, transposed access local_A[j, row0+i]). Each stage's
body slices its output-row dimension over mapping=[32] via get_wid(). The SPMW
region retains the tmp dependency between stages.
"""

from __future__ import annotations

import allo
from allo.ir.types import float32 as fp16
from allo.dataflow import region as _df_region

from lib.shapes import shape

_S = shape("atax")
M, N = _S["M"], _S["N"]

_MAPPING = [32]  # one work item per channel; MAC_ABK spans its 16 banks
_NPE = _MAPPING[0]
ROWS_M = -(-M // _NPE)  # stage1 slice (M rows)
ROWS_N = -(-N // _NPE)  # stage2 slice (N rows)


@_df_region()
def _atax_top(A: fp16[M, N], x: fp16[N], tmp: fp16[M], y: fp16[N]):
    @allo.work(mapping=_MAPPING, args=[A, x, tmp])
    def ax(local_A: fp16[M, N], local_x: fp16[N], local_tmp: fp16[M]):
        (channel,) = allo.get_wid()
        row0 = channel * ROWS_M
        for i in range(ROWS_M):
            acc: fp16 = 0
            for j in range(N):
                acc += local_A[row0 + i, j] * local_x[j]
            local_tmp[row0 + i] = acc

    @allo.work(mapping=_MAPPING, args=[A, tmp, y])
    def atx(local_A: fp16[M, N], local_tmp: fp16[M], local_y: fp16[N]):
        (channel,) = allo.get_wid()
        row0 = channel * ROWS_N
        for i in range(ROWS_N):
            acc: fp16 = 0
            for k in range(M):
                acc += local_A[k, row0 + i] * local_tmp[k]  # A^T access
            local_y[row0 + i] = acc


def build():
    return _atax_top


# (output rows, reduction extent) for result reporting.
STAGES = [(M, N), (N, M)]
