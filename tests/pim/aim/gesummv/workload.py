# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench gesummv on SK hynix GDDR6-AiM -- SPMW slice form (spec 05).

tmp = A@x ; y = B@x ; out = alpha*tmp + beta*y. Two INDEPENDENT GEMVs over the
same x. Each body slices its N output rows over mapping=[32] via get_wid(); the
alpha/beta axpy combine is outside the two contraction kernels.
"""

from __future__ import annotations

import allo
from allo.ir.types import float32 as fp16
from allo.dataflow import region as _df_region

from lib.shapes import shape

N = shape("gesummv")["N"]

_MAPPING = [32]  # one work item per channel; MAC_ABK spans its 16 banks
_NPE = _MAPPING[0]
ROWS = -(-N // _NPE)


@_df_region()
def _gesummv_top(A: fp16[N, N], B: fp16[N, N], x: fp16[N], tmp: fp16[N], y: fp16[N]):
    @allo.work(mapping=_MAPPING, args=[A, x, tmp])
    def gv_a(local_A: fp16[N, N], local_x: fp16[N], local_tmp: fp16[N]):
        (channel,) = allo.get_wid()
        row0 = channel * ROWS
        for i in range(ROWS):
            acc: fp16 = 0
            for j in range(N):
                acc += local_A[row0 + i, j] * local_x[j]
            local_tmp[row0 + i] = acc

    @allo.work(mapping=_MAPPING, args=[B, x, y])
    def gv_b(local_B: fp16[N, N], local_x: fp16[N], local_y: fp16[N]):
        (channel,) = allo.get_wid()
        row0 = channel * ROWS
        for i in range(ROWS):
            acc: fp16 = 0
            for j in range(N):
                acc += local_B[row0 + i, j] * local_x[j]
            local_y[row0 + i] = acc


def build():
    return _gesummv_top


# (output rows, reduction extent) for result reporting.
STAGES = [(N, N), (N, N)]
