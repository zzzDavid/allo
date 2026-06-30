# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench covariance: cov = (centered data)^T @ (centered data) / (N-1).

The @allo.work is the BARE GEMM contraction on PRE-CENTERED data:
`cov_raw[i, j] = sum_k cdata[k, i] * cdata[k, j]` (single MAC-reduction nest, the
matcher lowers it directly -- mirrors gemm/two_mm). The host pre-pass (column
mean, broadcast-subtract centering) and post-pass (the `/(N-1)` scale and the
symmetric `cov[j,i]=cov[i,j]` copy) run host-side against `covariance_np`:
Samsung has no divide and no cross-lane reduction beyond the GRF MAC accumulator,
so mean/center/normalize stay on the host (kernels-spec 03 sec 2.1). Dims from
lib.shapes (psize covariance SMALL: M=80, N=100; data is [N, M]).
"""

from __future__ import annotations

import allo
from allo.ir.types import float32 as fp16
from allo.dataflow import region as _df_region

from lib.shapes import shape

_S = shape("covariance")
M, N = _S["M"], _S["N"]


@_df_region()
def _covariance_top(cdata: fp16[N, M], cov_raw: fp16[M, M]):
    @allo.work(mapping=[1], args=[cdata, cov_raw])
    def cov_k(local_cdata: fp16[N, M], local_cov: fp16[M, M]):
        for i in range(M):
            for j in range(M):
                acc: fp16 = 0
                for k in range(N):
                    acc += local_cdata[k, i] * local_cdata[k, j]
                local_cov[i, j] = acc


def build():
    return _covariance_top


# (rows, reduction) per MAC stage -- the GEMV-harness geometry (cf. doitgen's
# (R*Q, S)). One MAC group: M output rows, N reduction. Cycles-measurement
# shapes (zeros); the host mean/centering/normalize is applied vs the ref.
STAGES = [(M, N)]
