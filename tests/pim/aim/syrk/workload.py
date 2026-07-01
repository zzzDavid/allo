# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench syrk on SK hynix GDDR6-AiM -- SPMW slice form (Tier-2, task 006).

C = alpha*A@A^T + beta*C, lower-triangular (j<=i). The MAC-reduction core the
matcher lowers is the bare symmetric product C[i,j] = sum_k A[i,k]*A[j,k] (k over
M); the triangular `j<=i` mask + alpha/beta scale are a host post-pass on the
numpy ref (AiM has no divide). The `@allo.work` body describes ONE PE's
output-row slice indexed by `get_wid()`; `mapping=[32]` declares the 32-channel
AiM grid, ROWS = ceil(N / prod(mapping)). Ramulator2 reports timing without
functional output arrays, so the cell is intentionally CYCLES-ONLY.
"""

from __future__ import annotations

import allo
from allo.ir.types import float32 as fp16
from allo.dataflow import region as _df_region

from lib.shapes import shape

_S = shape("syrk")
M, N = _S["M"], _S["N"]

_MAPPING = [32]  # one work item per channel; MAC_ABK spans its 16 banks
_NPE = _MAPPING[0]
ROWS = -(-N // _NPE)  # ceil(N / 32)


@_df_region()
def _syrk_top(A: fp16[N, M], C: fp16[N, N]):
    @allo.work(mapping=_MAPPING, args=[A, C])
    def syrk_k(local_A: fp16[N, M], local_C: fp16[N, N]):
        (channel,) = allo.get_wid()
        row0 = channel * ROWS  # this PE's first output row
        for i in range(ROWS):  # SLICE loop -- per-PE rows
            for j in range(N):  # output columns
                acc: fp16 = 0
                for k in range(M):  # contraction (K)
                    acc += local_A[row0 + i, k] * local_A[j, k]
                local_C[row0 + i, j] = acc


def build():
    return _syrk_top


# (output rows, reduction extent) for result reporting.
STAGES = [(N, M)]
