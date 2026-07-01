# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench 2mm on SK hynix GDDR6-AiM -- SPMW slice form (spec 05).

AB = A@B ; D = AB@C. Each stage slices its P output rows over mapping=[32]
via get_wid(). The SPMW region retains the two-stage dependency; AiM's trace
simulator reports timing without functional readback.
"""

from __future__ import annotations

import allo
from allo.ir.types import float32 as fp16
from allo.dataflow import region as _df_region

from lib.shapes import shape

_S = shape("2mm")
P, Q, R, S = _S["P"], _S["Q"], _S["R"], _S["S"]

_MAPPING = [32]  # one work item per channel; MAC_ABK spans its 16 banks
_NPE = _MAPPING[0]
ROWS = -(-P // _NPE)  # both stages output P rows


@_df_region()
def _two_mm_top(
    A: fp16[P, Q], B: fp16[Q, R], C: fp16[R, S], AB: fp16[P, R], D: fp16[P, S]
):
    @allo.work(mapping=_MAPPING, args=[A, B, AB])
    def mm1(local_A: fp16[P, Q], local_B: fp16[Q, R], local_AB: fp16[P, R]):
        (channel,) = allo.get_wid()
        row0 = channel * ROWS
        for i in range(ROWS):
            for j in range(R):
                acc: fp16 = 0
                for k in range(Q):
                    acc += local_A[row0 + i, k] * local_B[k, j]
                local_AB[row0 + i, j] = acc

    @allo.work(mapping=_MAPPING, args=[AB, C, D])
    def mm2(local_AB: fp16[P, R], local_C: fp16[R, S], local_D: fp16[P, S]):
        (channel,) = allo.get_wid()
        row0 = channel * ROWS
        for i in range(ROWS):
            for j in range(S):
                acc: fp16 = 0
                for k in range(R):
                    acc += local_AB[row0 + i, k] * local_C[k, j]
                local_D[row0 + i, j] = acc


def build():
    return _two_mm_top


# (output rows, reduction extent) for result reporting.
STAGES = [(P, Q), (P, R)]
