# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench gramschmidt on Samsung HBM-PIM -- SPMW slice form (Tier-2, task 006).

Modified Gram-Schmidt QR. Its dominant MAC-reduction core is the Gram-matrix
contraction G[p,q] = sum_i A[i,p]*A[i,q] (i over M) -- the same dot-product
reduction the projections/column-norms use. The `@allo.work` body describes ONE
PE's output-row slice indexed by `get_wid()`; `mapping=[16,8]` declares the
128-PE Samsung grid, ROWS = ceil(N / prod(mapping)). The column-by-column
normalize + subtract orthogonalization (loop-carried) is a host pass on the numpy
ref. Co-located leaf-dir workload.

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

_S = shape("gramschmidt")
M, N = _S["M"], _S["N"]

_MAPPING = [16, 8]
_NPE = _MAPPING[0] * _MAPPING[1]
ROWS = -(-N // _NPE)  # ceil(N / 128)


@_df_region()
def _gramschmidt_top(A: fp16[M, N], G: fp16[N, N]):
    @allo.work(mapping=_MAPPING, args=[A, G])
    def gramschmidt_k(local_A: fp16[M, N], local_G: fp16[N, N]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * ROWS  # this PE's first output row (p)
        for i in range(ROWS):  # SLICE loop -- per-PE rows
            for q in range(N):  # output columns
                acc: fp16 = 0
                for k in range(M):  # contraction over rows (i over M)
                    acc += local_A[k, row0 + i] * local_A[k, q]  # A^T A
                local_G[row0 + i, q] = acc


def build():
    return _gramschmidt_top


# Host data movement (spec backend-host-transfer-dispatch.md): the Gram core is
# A^T @ A, so A is staged as BOTH the scattered weight and the broadcast input;
# G is gathered. Recorded at import and consumed by ``allo.compile``.
with allo.record_host_moves() as _hm:
    host_staging.stage_gemm(weight="A", vec="A", out="G")
HOST_MOVES = list(_hm)


STAGES = [(N, M)]
