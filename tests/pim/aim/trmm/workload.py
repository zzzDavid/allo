# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench trmm on SK hynix GDDR6-AiM -- SPMW slice form (Tier-2, task 006).

B = alpha * triangular(A^T) @ B (in-place). The MAC-reduction core the matcher
lowers is the bare product Bout[i,j] = sum_k A[k,i]*B[k,j] (k over M -- an A^T@B
contraction). The PolyBench triangular `k>i` accumulation, the `+B[i,j]`
self-term, and the alpha scale are a host pass on the numpy ref. The `@allo.work`
body describes ONE PE's output-row slice indexed by `get_wid()`; `mapping=[32]`
declares the 32-channel AiM grid, ROWS = ceil(M / prod(mapping)). Ramulator2
reports timing without functional output arrays, so the cell is intentionally
CYCLES-ONLY.
"""

from __future__ import annotations

import allo
from allo.ir.types import float32 as fp16
from allo.dataflow import region as _df_region

from lib.shapes import shape

_S = shape("trmm")
M, N = _S["M"], _S["N"]

_MAPPING = [32]  # one work item per channel; MAC_ABK spans its 16 banks
_NPE = _MAPPING[0]
ROWS = -(-M // _NPE)  # ceil(M / 32)


@_df_region()
def _trmm_top(A: fp16[M, M], B: fp16[M, N], Bout: fp16[M, N]):
    @allo.work(mapping=_MAPPING, args=[A, B, Bout])
    def trmm_k(local_A: fp16[M, M], local_B: fp16[M, N], local_Bout: fp16[M, N]):
        (channel,) = allo.get_wid()
        row0 = channel * ROWS  # this PE's first output row (i)
        for i in range(ROWS):  # SLICE loop -- per-PE rows
            for j in range(N):  # output columns
                acc: fp16 = 0
                for k in range(M):  # contraction (K), A^T access
                    acc += local_A[k, row0 + i] * local_B[k, j]
                local_Bout[row0 + i, j] = acc


def build():
    return _trmm_top


# (output rows, reduction extent) for result reporting.
STAGES = [(M, M)]
