# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench gemm: C = A @ B (single MAC-reduction nest). Shared workload spec.

Dimensions from lib.shapes (psize.json SMALL: P=60,Q=80,R=70). The alpha/beta
scaling is applied host-side against the numpy reference; the workload is the
pure contraction the matcher lowers directly.
"""

from __future__ import annotations

import allo
from allo.ir.types import float32 as fp16
from allo.dataflow import region as _df_region

from lib.shapes import shape

_S = shape("gemm")
P, Q, R = _S["P"], _S["Q"], _S["R"]


@_df_region()
def _gemm_top(A: fp16[P, Q], B: fp16[Q, R], C: fp16[P, R]):
    @allo.work(mapping=[1], args=[A, B, C])
    def gemm_k(local_A: fp16[P, Q], local_B: fp16[Q, R], local_C: fp16[P, R]):
        for i in range(P):
            for j in range(R):
                acc: fp16 = 0
                for k in range(Q):
                    acc += local_A[i, k] * local_B[k, j]
                local_C[i, j] = acc


def build():
    return _gemm_top


# (rows, reduction) per MAC stage -- the GEMV-harness geometry per Samsung
# layer / the UPMEM GEMV slot. Cycles-measurement shapes (zeros), not numerics.
STAGES = [(P, Q)]
