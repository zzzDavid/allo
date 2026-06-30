# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench syr2k on Samsung HBM-PIM -- SPMW slice form (Tier-2, task 006).

Symmetric rank-2k, lower-triangular (j<=i): C[i,j] += alpha*A[i,k]*B[j,k] +
alpha*B[i,k]*A[j,k] (k over M). Decomposed into two MAC-reduction products
(P1 = A@B^T, P2 = B@A^T) summed host-side, with the triangular `j<=i` mask +
alpha/beta scale on the numpy ref. Each stage's `@allo.work` body describes ONE
PE's output-row slice indexed by `get_wid()`; `mapping=[16,8]` declares the
128-PE Samsung grid, ROWS = ceil(N / prod(mapping)). Co-located leaf-dir workload.

Multi-stage with no `samsung_reduce_chain` recipe (the host sum of P1+P2 is not a
device-threaded chain) -> the cell records an HONEST CYCLES-ONLY via one stage-0
REDUCE pass (real cycles, numerics unchecked, NEVER a fabricated PASS).
"""

from __future__ import annotations

import allo
from allo.ir.types import float32 as fp16
from allo.dataflow import region as _df_region

from lib.shapes import shape
from lib import host_staging

_S = shape("syr2k")
M, N = _S["M"], _S["N"]

_MAPPING = [16, 8]
_NPE = _MAPPING[0] * _MAPPING[1]
ROWS = -(-N // _NPE)  # ceil(N / 128); both stages out N rows


@_df_region()
def _syr2k_top(A: fp16[N, M], B: fp16[N, M], P1: fp16[N, N], P2: fp16[N, N]):
    @allo.work(mapping=_MAPPING, args=[A, B, P1])
    def prod_ab(local_A: fp16[N, M], local_B: fp16[N, M], local_P1: fp16[N, N]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * ROWS
        for i in range(ROWS):
            for j in range(N):
                acc: fp16 = 0
                for k in range(M):
                    acc += local_A[row0 + i, k] * local_B[j, k]  # A @ B^T
                local_P1[row0 + i, j] = acc

    @allo.work(mapping=_MAPPING, args=[B, A, P2])
    def prod_ba(local_B: fp16[N, M], local_A: fp16[N, M], local_P2: fp16[N, N]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * ROWS
        for i in range(ROWS):
            for j in range(N):
                acc: fp16 = 0
                for k in range(M):
                    acc += local_B[row0 + i, k] * local_A[j, k]  # B @ A^T
                local_P2[row0 + i, j] = acc


def build():
    return _syr2k_top


# Host data movement (spec backend-host-transfer-dispatch.md): two MAC products
# whose readbacks (P1, P2) are summed HOST-SIDE, so both are gathered (neither is
# a device-resident chain input). Stage1 scatters A / broadcasts B; stage2
# scatters B / broadcasts A.
with allo.record_host_moves() as _hm:
    host_staging.stage_gemm(weight="A", vec="B", out="P1")  # P1 = A @ B^T
    host_staging.stage_gemm(weight="B", vec="A", out="P2")  # P2 = B @ A^T
HOST_MOVES = list(_hm)


STAGES = [(N, M), (N, M)]
