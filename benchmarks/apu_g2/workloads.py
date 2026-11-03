# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The five Gemini-II evaluation kernels as ordinary ``@allo`` source.

Each function is the semantic graph of one artifact case
(``tenon-artifacts/apuv2-composed-v2/graph-specs``) written as plain loops.
``allo.compile(fn, build_apu_g2_target(), apu_g2_cost)`` derives the composed
program, host packing, and launch plan from the code; nothing in the compiler
refers to these names or shapes.  Each kernel's source block is delimited by
``# --- <name> ---`` comments so its lines can be counted.
"""

from allo.ir.types import int2, int3, int4, int11, int14, int15

# --- gemm ---
GEMM_M, GEMM_K, GEMM_N = 64, 128, 31


def gemm(A: int4[GEMM_M, GEMM_K], B: int4[GEMM_K, GEMM_N], C: int15[GEMM_M, GEMM_N]):
    for i in range(GEMM_M):
        for j in range(GEMM_N):
            acc: int15 = 0
            for k in range(GEMM_K):
                acc += A[i, k] * B[k, j]
            C[i, j] = acc


# --- batched_gemm ---
BG_B, BG_M, BG_K, BG_N = 4, 32, 128, 16


def batched_gemm(
    A: int4[BG_B, BG_M, BG_K], B: int4[BG_B, BG_K, BG_N], C: int15[BG_B, BG_M, BG_N]
):
    for b in range(BG_B):
        for i in range(BG_M):
            for j in range(BG_N):
                acc: int15 = 0
                for k in range(BG_K):
                    acc += A[b, i, k] * B[b, k, j]
                C[b, i, j] = acc


# --- gesummv ---
GS_M, GS_K = 1024, 128
ALPHA, BETA = 3, 2


def gesummv(A: int2[GS_M, GS_K], B: int2[GS_M, GS_K], x: int2[GS_K], y: int15[GS_M]):
    for i in range(GS_M):
        tmp: int11 = 0
        acc: int11 = 0
        for k in range(GS_K):
            tmp += A[i, k] * x[k]
            acc += B[i, k] * x[k]
        y[i] = ALPHA * tmp + BETA * acc


# --- bicg ---
BICG_N = 256


def bicg(
    A: int3[BICG_N, BICG_N],
    p: int3[BICG_N],
    r: int3[BICG_N],
    q: int14[BICG_N],
    s: int14[BICG_N],
):
    for i in range(BICG_N):
        s[i] = 0
    for i in range(BICG_N):
        q[i] = 0
        for j in range(BICG_N):
            q[i] += A[i, j] * p[j]
            s[j] += r[i] * A[i, j]


# --- mvt ---
MVT_N = 256


def mvt(
    A: int3[MVT_N, MVT_N],
    y1: int3[MVT_N],
    y2: int3[MVT_N],
    x1: int15[MVT_N],
    x2: int15[MVT_N],
):
    for i in range(MVT_N):
        for j in range(MVT_N):
            x1[i] += A[i, j] * y1[j]
    for i in range(MVT_N):
        for j in range(MVT_N):
            x2[i] += A[j, i] * y2[j]


# --- end ---

PAPER_KERNELS = {
    "gemm": gemm,
    "batched_gemm": batched_gemm,
    "gesummv": gesummv,
    "bicg": bicg,
    "mvt": mvt,
}
