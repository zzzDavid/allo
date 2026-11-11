# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import allo
from allo.dataflow import region as _df_region
from allo.ir.types import float16 as fp16
M0, K0, ROWS0 = 4096, 256, 32

@_df_region()
def gemv_m4096_k256(W: fp16[M0, K0], x: fp16[K0], y: fp16[M0]):
    @allo.work(mapping=[16, 8], args=[W, x, y])
    def gemv(local_W: fp16[M0, K0], local_x: fp16[K0], local_y: fp16[M0]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * ROWS0
        for i in range(ROWS0):
            acc: fp16 = 0
            for k in range(K0):
                acc += local_W[row0 + i, k] * local_x[k]
            local_y[row0 + i] = acc
