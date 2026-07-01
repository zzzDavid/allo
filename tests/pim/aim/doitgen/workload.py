# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench doitgen on SK hynix GDDR6-AiM -- SPMW slice form (spec 05).

out[r,q,p] = sum_s A[r,q,s]*x[s,p] -- a batched GEMV over (r,q), i.e. a GEMM of
the flattened (R*Q, S) batch against x[S,P]. The body slices the flattened
batch-row dim (R*Q) over mapping=[32] via get_wid(); the harness reshapes
A[R,Q,S]->[R*Q,S]. Co-located leaf-dir workload.
"""

from __future__ import annotations

import allo
from allo.ir.types import float32 as fp16
from allo.dataflow import region as _df_region

from lib.shapes import shape

_S = shape("doitgen")
R, Q, P, S = _S["R"], _S["Q"], _S["P"], _S["S"]
_RQ = R * Q  # flattened batch rows (the M dim)

_MAPPING = [32]  # one work item per channel; MAC_ABK spans its 16 banks
_NPE = _MAPPING[0]
ROWS = -(-_RQ // _NPE)  # ceil(R*Q / 32)


@_df_region()
def _doitgen_top(A: fp16[_RQ, S], x: fp16[S, P], out: fp16[_RQ, P]):
    @allo.work(mapping=_MAPPING, args=[A, x, out])
    def doitgen_k(local_A: fp16[_RQ, S], local_x: fp16[S, P], local_out: fp16[_RQ, P]):
        (channel,) = allo.get_wid()
        row0 = channel * ROWS
        for i in range(ROWS):  # SLICE over flattened (r,q) rows
            for p in range(P):  # output cols (N)
                acc: fp16 = 0
                for s in range(S):  # contraction (K)
                    acc += local_A[row0 + i, s] * local_x[s, p]
                local_out[row0 + i, p] = acc


def build():
    return _doitgen_top


# (output rows, reduction extent) for result reporting.
STAGES = [(_RQ, S)]
