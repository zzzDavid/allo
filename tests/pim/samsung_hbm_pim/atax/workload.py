# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench atax on Samsung HBM-PIM -- SPMW slice form (spec 05).

y = A^T (A x). Two stages: tmp = A@x (slice the M output rows), y = A^T@tmp
(slice the N output rows, transposed access local_A[j, row0+i]). Each stage's
body slices its own output-row dim over mapping=[16,8] via get_wid(). The cell's
chain helper threads stage1's real readback into stage2. Co-located leaf-dir.
"""

from __future__ import annotations

import allo
from allo.ir.types import float32 as fp16
from allo.dataflow import region as _df_region

from lib.shapes import shape
from lib import host_staging

_S = shape("atax")
M, N = _S["M"], _S["N"]

_MAPPING = [16, 8]
_NPE = _MAPPING[0] * _MAPPING[1]
ROWS_M = -(-M // _NPE)  # stage1 slice (M rows)
ROWS_N = -(-N // _NPE)  # stage2 slice (N rows)


@_df_region()
def _atax_top(A: fp16[M, N], x: fp16[N], tmp: fp16[M], y: fp16[N]):
    @allo.work(mapping=_MAPPING, args=[A, x, tmp])
    def ax(local_A: fp16[M, N], local_x: fp16[N], local_tmp: fp16[M]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * ROWS_M
        for i in range(ROWS_M):
            acc: fp16 = 0
            for j in range(N):
                acc += local_A[row0 + i, j] * local_x[j]
            local_tmp[row0 + i] = acc

    @allo.work(mapping=_MAPPING, args=[A, tmp, y])
    def atx(local_A: fp16[M, N], local_tmp: fp16[M], local_y: fp16[N]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * ROWS_N
        for i in range(ROWS_N):
            acc: fp16 = 0
            for k in range(M):
                acc += local_A[k, row0 + i] * local_tmp[k]  # A^T access
            local_y[row0 + i] = acc


def build():
    return _atax_top


# Host data movement (spec backend-host-transfer-dispatch.md), per stage:
#   stage1 tmp = A @ x      -- scatter A, broadcast x; tmp stays DEVICE-RESIDENT
#                              (threaded into stage2), so NO gather (per-cell
#                              override of the GEMM-family triple).
#   stage2 y   = A^T @ tmp  -- scatter A (transposed access), broadcast tmp;
#                              gather y (the final readback).
with allo.record_host_moves() as _hm:
    host_staging.stage_in(weight="A", vec="x")  # stage1: tmp on-device
    host_staging.stage_in(weight="A", vec="tmp")  # stage2 inputs
    host_staging.gather_out(out="y")  # stage2 readback
HOST_MOVES = list(_hm)


STAGES = [(M, N), (N, M)]
