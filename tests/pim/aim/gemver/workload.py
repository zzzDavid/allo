# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench gemver on SK hynix GDDR6-AiM -- SPMW slice form (spec 05).

4-phase: rank-1 updates A += u1*v1, A += u2*v2 (ELTWISE outer products, NOT a
K-reduction), then x = beta*A^T@y, w = alpha*A@x (GEMV). All 4 bodies slice their
row dimension over mapping=[32] via get_wid(). This deliberately exercises both
AiM elementwise operations and MAC reductions. The trace shows the 32-channel
grid per stage; ramulator2 provides cycles but no functional output arrays.
"""

from __future__ import annotations

import allo
from allo.ir.types import float32 as fp16
from allo.dataflow import region as _df_region

from lib.shapes import shape

N = shape("gemver")["N"]

_MAPPING = [32]  # one work item per channel; MAC_ABK spans its 16 banks
_NPE = _MAPPING[0]
ROWS = -(-N // _NPE)


@_df_region()
def _gemver_top(
    A: fp16[N, N],
    u1: fp16[N],
    v1: fp16[N],
    u2: fp16[N],
    v2: fp16[N],
    y: fp16[N],
    x: fp16[N],
    w: fp16[N],
):
    @allo.work(mapping=_MAPPING, args=[A, u1, v1])
    def rank1_a(lA: fp16[N, N], lu1: fp16[N], lv1: fp16[N]):
        (channel,) = allo.get_wid()
        row0 = channel * ROWS
        for i in range(ROWS):
            for j in range(N):
                lA[row0 + i, j] = lA[row0 + i, j] + lu1[row0 + i] * lv1[j]

    @allo.work(mapping=_MAPPING, args=[A, u2, v2])
    def rank1_b(lA: fp16[N, N], lu2: fp16[N], lv2: fp16[N]):
        (channel,) = allo.get_wid()
        row0 = channel * ROWS
        for i in range(ROWS):
            for j in range(N):
                lA[row0 + i, j] = lA[row0 + i, j] + lu2[row0 + i] * lv2[j]

    @allo.work(mapping=_MAPPING, args=[A, y, x])
    def aty(lA: fp16[N, N], ly: fp16[N], lx: fp16[N]):
        (channel,) = allo.get_wid()
        row0 = channel * ROWS
        for i in range(ROWS):
            acc: fp16 = 0
            for k in range(N):
                acc += lA[k, row0 + i] * ly[k]  # A^T access
            lx[row0 + i] = acc

    @allo.work(mapping=_MAPPING, args=[A, x, w])
    def ax(lA: fp16[N, N], lx: fp16[N], lw: fp16[N]):
        (channel,) = allo.get_wid()
        row0 = channel * ROWS
        for i in range(ROWS):
            acc: fp16 = 0
            for j in range(N):
                acc += lA[row0 + i, j] * lx[j]
            lw[row0 + i] = acc


def build():
    return _gemver_top


# (output rows, reduction extent) for result reporting.
STAGES = [(N, N), (N, N), (N, N), (N, N)]
