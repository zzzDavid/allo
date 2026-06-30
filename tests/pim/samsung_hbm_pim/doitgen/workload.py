# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench doitgen on Samsung HBM-PIM -- SPMW slice form (spec 05).

out[r,q,p] = sum_s A[r,q,s]*x[s,p] -- a batched GEMV over (r,q), i.e. a GEMM of
the flattened (R*Q, S) batch against x[S,P]. The body slices the flattened
batch-row dim (R*Q) over mapping=[16,8] via get_wid(); the harness reshapes
A[R,Q,S]->[R*Q,S]. Co-located leaf-dir workload.
"""

from __future__ import annotations

import allo
from allo.ir.types import float32 as fp16
from allo.dataflow import region as _df_region

from lib.shapes import shape
from lib import host_staging

_S = shape("doitgen")
R, Q, P, S = _S["R"], _S["Q"], _S["P"], _S["S"]
_RQ = R * Q  # flattened batch rows (the M dim)

_MAPPING = [16, 8]
_NPE = _MAPPING[0] * _MAPPING[1]
ROWS = -(-_RQ // _NPE)  # ceil(R*Q / 128)


@_df_region()
def _doitgen_top(A: fp16[_RQ, S], x: fp16[S, P], out: fp16[_RQ, P]):
    @allo.work(mapping=_MAPPING, args=[A, x, out])
    def doitgen_k(local_A: fp16[_RQ, S], local_x: fp16[S, P], local_out: fp16[_RQ, P]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * ROWS
        for i in range(ROWS):  # SLICE over flattened (r,q) rows
            for p in range(P):  # output cols (N)
                acc: fp16 = 0
                for s in range(S):  # contraction (K)
                    acc += local_A[row0 + i, s] * local_x[s, p]
                local_out[row0 + i, p] = acc


def build():
    return _doitgen_top


# Host data movement (spec backend-host-transfer-dispatch.md): scatter the
# flattened batch weight A into bank DRAM, broadcast x into GRF_A, gather out.
# Recorded at import and consumed by the suite's ``allo.compile`` helper.
with allo.record_host_moves() as _hm:
    host_staging.stage_gemm(weight="A", vec="x", out="out")
HOST_MOVES = list(_hm)


STAGES = [(_RQ, S)]
