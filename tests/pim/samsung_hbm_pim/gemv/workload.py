"""Static Tenon GEMV workloads used by the Samsung HBM-PIM campaign.

The three logical shapes are the shapes in Samsung PIMLibrary's compiler
examples ``gemv.cpp``, ``multi_input_tile_gemv.cpp``, and
``multi_output_tile_gemv.cpp``.  The 16 x 8 work mapping is the Samsung
target's 128-work-item fabric mapping.
"""

from __future__ import annotations

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


M1, K1, ROWS1 = 4096, 512, 32


@_df_region()
def gemv_m4096_k512(W: fp16[M1, K1], x: fp16[K1], y: fp16[M1]):
    @allo.work(mapping=[16, 8], args=[W, x, y])
    def gemv(local_W: fp16[M1, K1], local_x: fp16[K1], local_y: fp16[M1]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * ROWS1
        for i in range(ROWS1):
            acc: fp16 = 0
            for k in range(K1):
                acc += local_W[row0 + i, k] * local_x[k]
            local_y[row0 + i] = acc


M2, K2, ROWS2 = 8192, 256, 64


@_df_region()
def gemv_m8192_k256(W: fp16[M2, K2], x: fp16[K2], y: fp16[M2]):
    @allo.work(mapping=[16, 8], args=[W, x, y])
    def gemv(local_W: fp16[M2, K2], local_x: fp16[K2], local_y: fp16[M2]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * ROWS2
        for i in range(ROWS2):
            acc: fp16 = 0
            for k in range(K2):
                acc += local_W[row0 + i, k] * local_x[k]
            local_y[row0 + i] = acc


CASES = {
    "gemv_m4096_k256": (gemv_m4096_k256, M0, K0),
    "gemv_m4096_k512": (gemv_m4096_k512, M1, K1),
    "gemv_m8192_k256": (gemv_m8192_k256, M2, K2),
}
