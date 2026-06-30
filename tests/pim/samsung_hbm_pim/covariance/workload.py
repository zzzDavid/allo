# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench covariance on Samsung HBM-PIM -- SPMW slice form (spec 05).

cov_raw[i,j] = sum_k cdata[k,i]*cdata[k,j] -- the BARE GEMM contraction on
pre-centered data. The body slices the output-row dim M over mapping=[16,8] via
get_wid(); the host mean/center/normalize/symmetrize is applied to the numpy ref
(Samsung has no divide). Co-located leaf-dir workload.
"""

from __future__ import annotations

import allo
from allo.ir.types import float32 as fp16
from allo.dataflow import region as _df_region

from lib.shapes import shape

_S = shape("covariance")
M, N = _S["M"], _S["N"]

_MAPPING = [16, 8]
_NPE = _MAPPING[0] * _MAPPING[1]
ROWS = -(-M // _NPE)                          # ceil(M / 128)


@_df_region()
def _covariance_top(cdata: fp16[N, M], cov_raw: fp16[M, M]):
    @allo.work(mapping=_MAPPING, args=[cdata, cov_raw])
    def cov_k(local_cdata: fp16[N, M], local_cov: fp16[M, M]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * ROWS
        for i in range(ROWS):
            for j in range(M):
                acc: fp16 = 0
                for k in range(N):
                    acc += local_cdata[k, row0 + i] * local_cdata[k, j]
                local_cov[row0 + i, j] = acc


def build():
    return _covariance_top


STAGES = [(M, N)]
