# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench symm on SK hynix GDDR6-AiM -- SPMW slice form (Tier-2, task 006).

C = alpha*B*A_sym + beta*C, A symmetric. The MAC-reduction core the matcher
lowers is the bare product summ[i,j] = sum_k A[i,k]*B[k,j] (k over M). The
triangular access (`k<i` + diagonal term), the alpha/beta scale, and the running
C update are a host pass on the numpy ref. The `@allo.work` body describes ONE
PE's output-row slice indexed by `get_wid()`; `mapping=[32]` declares the
32-channel AiM grid, ROWS = ceil(M / prod(mapping)). Ramulator2 reports timing
without functional output arrays, so the cell is intentionally CYCLES-ONLY.
"""

from __future__ import annotations

import allo
from allo.ir.types import bfloat16 as fp16
from allo.dataflow import region as _df_region

from lib.shapes import shape

_S = shape("symm")
M, N = _S["M"], _S["N"]

_MAPPING = [32]  # one work item per channel; MAC_ABK spans its 16 banks
_NPE = _MAPPING[0]
ROWS = -(-M // _NPE)  # ceil(M / 32)


@_df_region()
def _symm_top(A: fp16[M, M], B: fp16[M, N], S: fp16[M, N]):
    @allo.work(mapping=_MAPPING, args=[A, B, S])
    def symm_k(local_A: fp16[M, M], local_B: fp16[M, N], local_S: fp16[M, N]):
        (channel,) = allo.get_wid()
        row0 = channel * ROWS  # this PE's first output row (i)
        for i in range(ROWS):  # SLICE loop -- per-PE rows
            for j in range(N):  # output columns
                acc: fp16 = 0
                for k in range(M):  # contraction (K)
                    acc += local_A[row0 + i, k] * local_B[k, j]
                local_S[row0 + i, j] = acc


def build():
    return _symm_top


# (output rows, reduction extent) for result reporting.
STAGES = [(M, M)]
