# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench syrk on Samsung HBM-PIM -- SPMW slice form (Tier-2, task 006).

C = alpha*A@A^T + beta*C, lower-triangular (j<=i). The MAC-reduction core the
matcher lowers is the bare symmetric product C[i,j] = sum_k A[i,k]*A[j,k] (k over
M); the triangular `j<=i` mask + alpha/beta scale are a host post-pass on the
numpy ref (Samsung has no divide). The `@allo.work` body describes ONE PE's
output-row slice indexed by `get_wid()`; `mapping=[16,8]` declares the 128-PE
Samsung grid, ROWS = ceil(N / prod(mapping)). Co-located leaf-dir workload.

This kernel has no `_samsung_gemm_operands` reference entry (the contraction is
A@A^T, not a clean A@B at the cell's seeded shape), so the cell records an HONEST
CYCLES-ONLY via one stage-0 REDUCE pass -- a real cycle count, numerics unchecked,
NEVER a fabricated PASS.
"""

from __future__ import annotations

import allo
from allo.ir.types import float32 as fp16
from allo.dataflow import region as _df_region

from lib.shapes import shape
from lib import host_staging

_S = shape("syrk")
M, N = _S["M"], _S["N"]

_MAPPING = [16, 8]
_NPE = _MAPPING[0] * _MAPPING[1]
ROWS = -(-N // _NPE)  # ceil(N / 128)


@_df_region()
def _syrk_top(A: fp16[N, M], C: fp16[N, N]):
    @allo.work(mapping=_MAPPING, args=[A, C])
    def syrk_k(local_A: fp16[N, M], local_C: fp16[N, N]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * ROWS  # this PE's first output row
        for i in range(ROWS):  # SLICE loop -- per-PE rows
            for j in range(N):  # output columns
                acc: fp16 = 0
                for k in range(M):  # contraction (K)
                    acc += local_A[row0 + i, k] * local_A[j, k]
                local_C[row0 + i, j] = acc


def build():
    return _syrk_top


# Host data movement (spec backend-host-transfer-dispatch.md): A is staged as
# BOTH the scattered weight and the broadcast input (the contraction is A@A^T);
# C is gathered back. ``allo.compile`` consumes HOST_MOVES through the suite's
# public compile helper.
with allo.record_host_moves() as _hm:
    host_staging.stage_gemm(weight="A", vec="A", out="C")
HOST_MOVES = list(_hm)


STAGES = [(N, M)]
