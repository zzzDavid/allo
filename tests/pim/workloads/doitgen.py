# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench doitgen: out[r,q,p] = sum_s A[r,q,s]*x[s,p]. Shared workload spec.

A batched GEMV over (r,q): for each (r,q) the inner contraction over s is a
matvec against x. Single MAC-reduction nest (the matcher lowers it). Dims from
lib.shapes (psize doitgen SMALL: R=25,Q=20,P=30,S=30; x is [S,P]). Written to a
separate `out` so the workload is single-output (canonical doitgen writes back
into A; the host applies that against the numpy reference).
"""

from __future__ import annotations

import allo
from allo.ir.types import float32 as fp16
from allo.dataflow import region as _df_region

from lib.shapes import shape

_S = shape("doitgen")
R, Q, P, S = _S["R"], _S["Q"], _S["P"], _S["S"]


@_df_region()
def _doitgen_top(A: fp16[R, Q, S], x: fp16[S, P], out: fp16[R, Q, P]):
    @allo.work(mapping=[1], args=[A, x, out])
    def doitgen_k(local_A: fp16[R, Q, S], local_x: fp16[S, P], local_out: fp16[R, Q, P]):
        for r in range(R):
            for q in range(Q):
                for p in range(P):
                    acc: fp16 = 0
                    for s in range(S):
                        acc += local_A[r, q, s] * local_x[s, p]
                    local_out[r, q, p] = acc


def build():
    return _doitgen_top


# (rows, reduction) per MAC stage -- the GEMV-harness geometry per Samsung
# layer / the UPMEM GEMV slot. Cycles-measurement shapes (zeros), not numerics.
STAGES = [(R * Q, S)]
