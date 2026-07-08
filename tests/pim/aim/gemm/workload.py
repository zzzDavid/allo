# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench gemm on SK hynix GDDR6-AiM -- SPMW slice form (spec 05).

C = A @ B (single MAC-reduction nest). The `@allo.work` body describes ONE PE's
output-row slice indexed by `get_wid()`; `mapping=[32]` declares the 32-channel
AiM grid. The slice size ROWS = ceil(P / prod(mapping)) is parameterized by
`prod(mapping)` (a module constant), so the partition flows mapping -> trace ->
codegen (the matcher sees prod(mapping) MAC buckets, each with a per-PE slice
loop bound == ROWS). alpha/beta scaling is a host post-pass on the numpy ref.

Co-located in the AiM leaf dir (workloads are target-specific now).
"""

from __future__ import annotations

import allo
from allo.ir.types import bfloat16 as fp16
from allo.dataflow import region as _df_region

from lib.shapes import shape

_S = shape("gemm")
P, Q, R = _S["P"], _S["Q"], _S["R"]  # M=P, K=Q, N=R

# The AiM channel grid; bank fan-out is expressed by the selected MAC_ABK op.
_MAPPING = [32]  # one work item per channel; MAC_ABK spans its 16 banks
_NPE = _MAPPING[0]
# Per-PE output-row slice. P is padded to a multiple of _NPE by the harness
# (zeros), so ROWS divides evenly; the readback slices the real P back.
ROWS = -(-P // _NPE)  # ceil(P / 32)


@_df_region()
def _gemm_top(A: fp16[P, Q], B: fp16[Q, R], C: fp16[P, R]):
    @allo.work(mapping=_MAPPING, args=[A, B, C])
    def gemm_k(local_A: fp16[P, Q], local_B: fp16[Q, R], local_C: fp16[P, R]):
        (channel,) = allo.get_wid()
        row0 = channel * ROWS  # this PE's first output row
        for i in range(ROWS):  # SLICE loop -- per-PE rows
            for j in range(R):  # output columns (N)
                acc: fp16 = 0
                for k in range(Q):  # contraction (K)
                    acc += local_A[row0 + i, k] * local_B[k, j]
                local_C[row0 + i, j] = acc


def build():
    return _gemm_top


# (output rows, reduction extent) for result reporting.
STAGES = [(P, Q)]
