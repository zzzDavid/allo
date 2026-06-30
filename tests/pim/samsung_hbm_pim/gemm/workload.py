# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench gemm on Samsung HBM-PIM -- SPMW slice form (spec 05).

C = A @ B (single MAC-reduction nest). The `@allo.work` body describes ONE PE's
output-row slice indexed by `get_wid()`; `mapping=[16,8]` declares the 128-PE
Samsung grid. The slice size ROWS = ceil(P / prod(mapping)) is parameterized by
`prod(mapping)` (a module constant), so the partition flows mapping -> trace ->
codegen (the matcher sees prod(mapping) MAC buckets, each with a per-PE slice
loop bound == ROWS). alpha/beta scaling is a host post-pass on the numpy ref.

Co-located in the Samsung leaf dir (workloads are target-specific now).
"""

from __future__ import annotations

import allo
from allo.ir.types import float32 as fp16
from allo.dataflow import region as _df_region

from lib.shapes import shape
from lib import host_staging

_S = shape("gemm")
P, Q, R = _S["P"], _S["Q"], _S["R"]  # M=P, K=Q, N=R

# The Samsung PE grid; prod = 128. Authored once; other backends pass their own.
_MAPPING = [16, 8]
_NPE = _MAPPING[0] * _MAPPING[1]
# Per-PE output-row slice. P is padded to a multiple of _NPE by the harness
# (zeros), so ROWS divides evenly; the readback slices the real P back.
ROWS = -(-P // _NPE)  # ceil(P / 128)


@_df_region()
def _gemm_top(A: fp16[P, Q], B: fp16[Q, R], C: fp16[P, R]):
    @allo.work(mapping=_MAPPING, args=[A, B, C])
    def gemm_k(local_A: fp16[P, Q], local_B: fp16[Q, R], local_C: fp16[P, R]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * ROWS  # this PE's first output row
        for i in range(ROWS):  # SLICE loop -- per-PE rows
            for j in range(R):  # output columns (N)
                acc: fp16 = 0
                for k in range(Q):  # contraction (K)
                    acc += local_A[row0 + i, k] * local_B[k, j]
                local_C[row0 + i, j] = acc


def build():
    return _gemm_top


# Host data movement, expressed VISIBLY via `allo.host_xfer` (spec
# backend-host-transfer-dispatch.md): scatter the weight A into bank DRAM,
# broadcast B into GRF_A, gather the result C back. Recorded at import;
# ``allo.compile`` consumes HOST_MOVES so operand-role binding is driven by
# these moves rather than inferred. Buffer operands name the @allo.work
# params (A=weight, B=input, C=output).
with allo.record_host_moves() as _hm:
    host_staging.stage_gemm(weight="A", vec="B", out="C")
HOST_MOVES = list(_hm)


# (rows, reduction) per MAC stage -- the GEMV-harness geometry. Single stage.
STAGES = [(P, Q)]
