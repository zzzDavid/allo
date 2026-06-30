# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench mvt on Samsung HBM-PIM -- SPMW slice form (spec 05).

x1 = A@y1 ; x2 = A^T@y2. Two INDEPENDENT GEMVs over the N output dim (stage2
transposed). Each body slices its N output rows over mapping=[16,8] via
get_wid(). The host `x += both` fold is on the ref. Co-located leaf-dir.
"""

from __future__ import annotations

import allo
from allo.ir.types import float32 as fp16
from allo.dataflow import region as _df_region

from lib.shapes import shape
from lib import host_staging

N = shape("mvt")["N"]

_MAPPING = [16, 8]
_NPE = _MAPPING[0] * _MAPPING[1]
ROWS = -(-N // _NPE)


@_df_region()
def _mvt_top(A: fp16[N, N], y1: fp16[N], y2: fp16[N], x1: fp16[N], x2: fp16[N]):
    @allo.work(mapping=_MAPPING, args=[A, y1, x1])
    def mv_a(local_A: fp16[N, N], local_y1: fp16[N], local_x1: fp16[N]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * ROWS
        for i in range(ROWS):
            acc: fp16 = 0
            for j in range(N):
                acc += local_A[row0 + i, j] * local_y1[j]
            local_x1[row0 + i] = acc

    @allo.work(mapping=_MAPPING, args=[A, y2, x2])
    def mv_b(local_A: fp16[N, N], local_y2: fp16[N], local_x2: fp16[N]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * ROWS
        for i in range(ROWS):
            acc: fp16 = 0
            for k in range(N):
                acc += local_A[k, row0 + i] * local_y2[k]  # A^T access
            local_x2[row0 + i] = acc


def build():
    return _mvt_top


# Host data movement (spec backend-host-transfer-dispatch.md): two INDEPENDENT
# GEMVs over the same A; both outputs (x1, x2) are gathered back for the host
# `x += x1 + x2` fold.
with allo.record_host_moves() as _hm:
    host_staging.stage_gemm(weight="A", vec="y1", out="x1")  # x1 = A @ y1
    host_staging.stage_gemm(weight="A", vec="y2", out="x2")  # x2 = A^T @ y2
HOST_MOVES = list(_hm)


STAGES = [(N, N), (N, N)]
