# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench covariance PIM core on SK hynix GDDR6-AiM.

The AiM-resident portion computes the column means and the uncentered Gram
matrix.  Mean-centering, ``X^T X - N*mu*mu^T``, and the ``1/(N-1)`` scaling
remain explicit host work because AiM has no broadcast-subtract or divide
instruction.  The two contractions below are therefore the complete, honest
device portion of covariance rather than a claim that normalization runs on
AiM.
"""

from __future__ import annotations

import allo
from allo.dataflow import region as _df_region
from allo.ir.types import bfloat16 as bf16

from lib.shapes import shape

_S = shape("covariance")
M, N = _S["M"], _S["N"]  # M variables, N observations
_MAPPING = [32]
ROWS = -(-M // _MAPPING[0])


@_df_region()
def _covariance_top(
    data: bf16[N, M],
    ones: bf16[N],
    mean_sum: bf16[M],
    gram: bf16[M, M],
):
    @allo.work(mapping=_MAPPING, args=[data, ones, mean_sum])
    def mean_k(local_data: bf16[N, M], local_ones: bf16[N], local_mean: bf16[M]):
        (channel,) = allo.get_wid()
        row0 = channel * ROWS
        for i in range(ROWS):
            acc: bf16 = 0
            for k in range(N):
                acc += local_data[k, row0 + i] * local_ones[k]
            local_mean[row0 + i] = acc

    @allo.work(mapping=_MAPPING, args=[data, gram])
    def gram_k(local_data: bf16[N, M], local_gram: bf16[M, M]):
        (channel,) = allo.get_wid()
        row0 = channel * ROWS
        for i in range(ROWS):
            for j in range(M):
                acc: bf16 = 0
                for k in range(N):
                    acc += local_data[k, row0 + i] * local_data[k, j]
                local_gram[row0 + i, j] = acc


def build():
    return _covariance_top


STAGES = [(M, N), (M, N)]
HOST_POSTPASS = "mean-center + N*mu*mu^T correction + divide by N-1"
