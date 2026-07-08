# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench correlation PIM core on SK hynix GDDR6-AiM.

AiM computes column sums and the uncentered Gram matrix.  Mean-centering,
variance/sqrt, and pairwise standard-deviation normalization are retained as
host postpasses because those operations are absent from the AiM ISA.
"""

from __future__ import annotations

import allo
from allo.dataflow import region as _df_region
from allo.ir.types import bfloat16 as bf16

from lib.shapes import shape

_S = shape("correlation")
M, N = _S["M"], _S["N"]
_MAPPING = [32]
ROWS = -(-M // _MAPPING[0])


@_df_region()
def _correlation_top(
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
    return _correlation_top


STAGES = [(M, N), (M, N)]
HOST_POSTPASS = "mean-center + variance/sqrt + pairwise normalize"
