# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench bicg on Samsung HBM-PIM -- SPMW slice form (spec 05).

q = A@p ; s = A^T@r. Two INDEPENDENT GEMVs (both read external p, r). stage1
slices M output rows; stage2 slices N output rows (transposed access). Each body
slices its output-row dim over mapping=[16,8] via get_wid(). Co-located leaf-dir.
"""

from __future__ import annotations

import allo
from allo.ir.types import float32 as fp16
from allo.dataflow import region as _df_region

from lib.shapes import shape

_S = shape("bicg")
M, N = _S["M"], _S["N"]

_MAPPING = [16, 8]
_NPE = _MAPPING[0] * _MAPPING[1]
ROWS_M = -(-M // _NPE)
ROWS_N = -(-N // _NPE)


@_df_region()
def _bicg_top(A: fp16[M, N], p: fp16[N], r: fp16[M], q: fp16[M], s: fp16[N]):
    @allo.work(mapping=_MAPPING, args=[A, p, q])
    def aq(local_A: fp16[M, N], local_p: fp16[N], local_q: fp16[M]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * ROWS_M
        for i in range(ROWS_M):
            acc: fp16 = 0
            for j in range(N):
                acc += local_A[row0 + i, j] * local_p[j]
            local_q[row0 + i] = acc

    @allo.work(mapping=_MAPPING, args=[A, r, s])
    def ats(local_A: fp16[M, N], local_r: fp16[M], local_s: fp16[N]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * ROWS_N
        for i in range(ROWS_N):
            acc: fp16 = 0
            for k in range(M):
                acc += local_A[k, row0 + i] * local_r[k]   # A^T access
            local_s[row0 + i] = acc


def build():
    return _bicg_top


STAGES = [(M, N), (N, M)]
