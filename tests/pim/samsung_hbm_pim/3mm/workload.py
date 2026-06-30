# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench 3mm on Samsung HBM-PIM -- SPMW slice form (spec 05).

AB = A@B ; CD = C@D ; G = AB@CD. A GEMM chain that threads TWO prior readbacks.
Each stage slices its output-row dim (P, R, P) over mapping=[16,8] via get_wid().
The cell's chain helper threads AB + CD into G. Co-located leaf-dir workload.
"""

from __future__ import annotations

import allo
from allo.ir.types import float32 as fp16
from allo.dataflow import region as _df_region

from lib.shapes import shape

_S = shape("3mm")
P, Q, R, NT, S = _S["P"], _S["Q"], _S["R"], _S["T"], _S["S"]

_MAPPING = [16, 8]
_NPE = _MAPPING[0] * _MAPPING[1]
ROWS_P = -(-P // _NPE)                           # mm1 (AB), mm3 (G) output P rows
ROWS_R = -(-R // _NPE)                           # mm2 (CD) output R rows


@_df_region()
def _three_mm_top(
    A: fp16[P, Q], B: fp16[Q, R], C: fp16[R, S], D: fp16[S, NT],
    AB: fp16[P, R], CD: fp16[R, NT], G: fp16[P, NT],
):
    @allo.work(mapping=_MAPPING, args=[A, B, AB])
    def mm1(local_A: fp16[P, Q], local_B: fp16[Q, R], local_AB: fp16[P, R]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * ROWS_P
        for i in range(ROWS_P):
            for j in range(R):
                acc: fp16 = 0
                for k in range(Q):
                    acc += local_A[row0 + i, k] * local_B[k, j]
                local_AB[row0 + i, j] = acc

    @allo.work(mapping=_MAPPING, args=[C, D, CD])
    def mm2(local_C: fp16[R, S], local_D: fp16[S, NT], local_CD: fp16[R, NT]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * ROWS_R
        for i in range(ROWS_R):
            for j in range(NT):
                acc: fp16 = 0
                for k in range(S):
                    acc += local_C[row0 + i, k] * local_D[k, j]
                local_CD[row0 + i, j] = acc

    @allo.work(mapping=_MAPPING, args=[AB, CD, G])
    def mm3(local_AB: fp16[P, R], local_CD: fp16[R, NT], local_G: fp16[P, NT]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * ROWS_P
        for i in range(ROWS_P):
            for j in range(NT):
                acc: fp16 = 0
                for k in range(R):
                    acc += local_AB[row0 + i, k] * local_CD[k, j]
                local_G[row0 + i, j] = acc


def build():
    return _three_mm_top


STAGES = [(P, Q), (R, S), (P, R)]
