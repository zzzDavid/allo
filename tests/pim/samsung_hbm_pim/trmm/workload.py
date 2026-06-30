# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench trmm on Samsung HBM-PIM -- SPMW slice form (Tier-2, task 006).

B = alpha * triangular(A^T) @ B (in-place). The MAC-reduction core the matcher
lowers is the bare product Bout[i,j] = sum_k A[k,i]*B[k,j] (k over M -- an A^T@B
contraction). The PolyBench triangular `k>i` accumulation, the `+B[i,j]`
self-term, and the alpha scale are a host pass on the numpy ref. The `@allo.work`
body describes ONE PE's output-row slice indexed by `get_wid()`; `mapping=[16,8]`
declares the 128-PE Samsung grid, ROWS = ceil(M / prod(mapping)). Co-located
leaf-dir workload.

No `_samsung_gemm_operands` reference entry -> the cell records an HONEST
CYCLES-ONLY via one stage-0 REDUCE pass (real cycles, numerics unchecked, NEVER a
fabricated PASS).
"""

from __future__ import annotations

import allo
from allo.ir.types import float32 as fp16
from allo.dataflow import region as _df_region

from lib.shapes import shape
from lib import host_staging

_S = shape("trmm")
M, N = _S["M"], _S["N"]

_MAPPING = [16, 8]
_NPE = _MAPPING[0] * _MAPPING[1]
ROWS = -(-M // _NPE)  # ceil(M / 128)


@_df_region()
def _trmm_top(A: fp16[M, M], B: fp16[M, N], Bout: fp16[M, N]):
    @allo.work(mapping=_MAPPING, args=[A, B, Bout])
    def trmm_k(local_A: fp16[M, M], local_B: fp16[M, N], local_Bout: fp16[M, N]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * ROWS  # this PE's first output row (i)
        for i in range(ROWS):  # SLICE loop -- per-PE rows
            for j in range(N):  # output columns
                acc: fp16 = 0
                for k in range(M):  # contraction (K), A^T access
                    acc += local_A[k, row0 + i] * local_B[k, j]
                local_Bout[row0 + i, j] = acc


def build():
    return _trmm_top


# Host data movement (spec backend-host-transfer-dispatch.md): scatter the
# weight A into bank DRAM, broadcast B into GRF_A, gather Bout back.
with allo.record_host_moves() as _hm:
    host_staging.stage_gemm(weight="A", vec="B", out="Bout")
HOST_MOVES = list(_hm)


STAGES = [(M, M)]
