# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench gramschmidt on SK hynix GDDR6-AiM -- SPMW slice form (Tier-2, task 006).

Modified Gram-Schmidt QR. Its dominant MAC-reduction core is the Gram-matrix
contraction G[p,q] = sum_i A[i,p]*A[i,q] (i over M) -- the same dot-product
reduction the projections/column-norms use. The `@allo.work` body describes ONE
PE's output-row slice indexed by `get_wid()`; `mapping=[32]` declares the
32-channel AiM grid, ROWS = ceil(N / prod(mapping)). The column-by-column
normalize + subtract orthogonalization (loop-carried) is a host pass on the numpy
ref. Ramulator2 reports timing for the emitted AiM trace without functional
output arrays, so the cell is intentionally CYCLES-ONLY.
"""

from __future__ import annotations

import allo
from allo.ir.types import bfloat16 as fp16
from allo.dataflow import region as _df_region

from lib.shapes import shape

_S = shape("gramschmidt")
M, N = _S["M"], _S["N"]

_MAPPING = [32]  # one work item per channel; MAC_ABK spans its 16 banks
_NPE = _MAPPING[0]
ROWS = -(-N // _NPE)  # ceil(N / 32)


@_df_region()
def _gramschmidt_top(A: fp16[M, N], G: fp16[N, N]):
    @allo.work(mapping=_MAPPING, args=[A, G])
    def gramschmidt_k(local_A: fp16[M, N], local_G: fp16[N, N]):
        (channel,) = allo.get_wid()
        row0 = channel * ROWS  # this PE's first output row (p)
        for i in range(ROWS):  # SLICE loop -- per-PE rows
            for q in range(N):  # output columns
                acc: fp16 = 0
                for k in range(M):  # contraction over rows (i over M)
                    acc += local_A[k, row0 + i] * local_A[k, q]  # A^T A
                local_G[row0 + i, q] = acc


def build():
    return _gramschmidt_top


# (output rows, reduction extent) for result reporting.
STAGES = [(N, M)]
