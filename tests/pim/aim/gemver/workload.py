# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench gemver's honest SK hynix GDDR6-AiM device portion.

AiM executes both matrix-vector contractions, ``x = A^T y`` and ``w = A x``.
The two rank-1 updates and alpha/beta/z combines remain on the host: pretending
that the one-dimensional EWMUL/EWADD instructions implement an N-by-N outer
product was semantically incorrect.
"""

from __future__ import annotations

import allo
from allo.dataflow import region as _df_region
from allo.ir.types import bfloat16 as bf16

from lib.shapes import shape

N = shape("gemver")["N"]
_MAPPING = [32]
ROWS = -(-N // _MAPPING[0])


@_df_region()
def _gemver_device(A: bf16[N, N], y: bf16[N], x: bf16[N], w: bf16[N]):
    @allo.work(mapping=_MAPPING, args=[A, y, x])
    def aty(local_A: bf16[N, N], local_y: bf16[N], local_x: bf16[N]):
        (channel,) = allo.get_wid()
        row0 = channel * ROWS
        for i in range(ROWS):
            acc: bf16 = 0
            for k in range(N):
                acc += local_A[k, row0 + i] * local_y[k]
            local_x[row0 + i] = acc

    @allo.work(mapping=_MAPPING, args=[A, x, w])
    def ax(local_A: bf16[N, N], local_x: bf16[N], local_w: bf16[N]):
        (channel,) = allo.get_wid()
        row0 = channel * ROWS
        for i in range(ROWS):
            acc: bf16 = 0
            for k in range(N):
                acc += local_A[row0 + i, k] * local_x[k]
            local_w[row0 + i] = acc


def build():
    return _gemver_device


STAGES = [(N, N), (N, N)]
HOST_POSTPASS = "rank-1 updates + beta/z and alpha scaling"
