# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""GEMM contraction used to validate the scalar ``mapping=8`` group axis."""

import allo
from allo.dataflow import region as _df_region
from allo.ir.types import float32 as fp16

from lib.shapes import shape

_S = shape("gemm")
P, Q, R = _S["P"], _S["Q"], _S["R"]
MAPPING = 8
ROWS = -(-P // 4)


@_df_region()
def _gemm_grouped(A: fp16[P, Q], B: fp16[Q, R], C: fp16[P, R]):
    @allo.work(mapping=MAPPING, args=[A, B, C])
    def gemm_k(local_A: fp16[P, Q], local_B: fp16[Q, R], local_C: fp16[P, R]):
        for i in range(ROWS):
            for j in range(R):
                acc: fp16 = 0
                for k in range(Q):
                    acc += local_A[i, k] * local_B[k, j]
                local_C[i, j] = acc


def build():
    return _gemm_grouped
