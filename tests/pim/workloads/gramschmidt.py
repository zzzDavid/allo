# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench gramschmidt (Tier-2): modified Gram-Schmidt QR. Shared spec.

Modified Gram-Schmidt is the orthogonalization member of the MAC-reduction
paradigm (Phase-0 scope call: IN SCOPE, "same MAC-reduction paradigm with a
triangular/orthogonalization access pattern"). Its dominant MAC-reduction core
is the Gram-matrix / projection contraction `G[p,q] = sum_i A[i,p]*A[i,q]`
(k over M) -- the `sum_i Q[i,k]*A[i,j]` projections and the `sum_i A[i,k]^2`
column norms are the same dot-product reduction. The `@allo.work` is that bare
contraction (matches MAC on every backend, no matcher change); the column-by-
column normalize + subtract orthogonalization (loop-carried across columns) is
applied host-side against `gramschmidt_np`. Dims from lib.shapes (psize SMALL:
A is [M,N]; M=60, N=80).
"""

from __future__ import annotations

import allo
from allo.ir.types import float32 as fp16
from allo.dataflow import region as _df_region

from lib.shapes import shape

_S = shape("gramschmidt")
M, N = _S["M"], _S["N"]


@_df_region()
def _gramschmidt_top(A: fp16[M, N], G: fp16[N, N]):
    @allo.work(mapping=[1], args=[A, G])
    def gramschmidt_k(local_A: fp16[M, N], local_G: fp16[N, N]):
        for p in range(N):
            for q in range(N):
                acc: fp16 = 0
                for i in range(M):
                    acc += local_A[i, p] * local_A[i, q]
                local_G[p, q] = acc


def build():
    return _gramschmidt_top


# (rows, reduction) per MAC stage -- the projection/Gram contraction geometry
# (N output rows, M reduction). Cycles-measurement shapes (zeros), not numerics.
STAGES = [(N, M)]
