# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench gemver (4-phase). Shared workload spec; exercises the multi-term
reduction matcher (SPEC-004 D3).

Phases authored as distinct `@allo.work` stages:
  1a/1b. rank-1 update  A[i,j] += u1[i]*v1[j]  then  A[i,j] += u2[i]*v2[j]
       -- authored as TWO single-`mul` accumulate funcs (distinct buckets).
          The single-statement two-`mul` form `A += u1*v1 + u2*v2` IS recognized
          by the additive add-chain flatten (SPEC-004 D3) -- it emits 2 MAC
          matches sharing A (guarded by tests/spmw/test_match_multi_output.py) --
          but those two matches reuse the MAC pattern's literal roles (x/y/acc)
          within ONE bucket, which the `_trace_memrefs_by_role` one-role-one-
          memref invariant rejects (codegen keys on the literal roles). So for
          a workload that LOWERS today, the two summands are authored as two
          buckets; the flatten stays a guarded matcher capability. (Folding the
          single-statement form past the matcher needs an autoschedule decision-
          path change -- out of scope, flagged for the architect.)
  2. x = beta * A^T @ y     (GEMV; the beta scale folds host-side vs the ref)
  3. w = alpha * A @ x      (GEMV)
The `x += z` reduction-free add is applied host-side against the numpy
reference -- it carries no contraction/cycles. Dims from lib.shapes (N=120).
"""

from __future__ import annotations

import allo
from allo.ir.types import float32 as fp16
from allo.dataflow import region as _df_region

from lib.shapes import shape

N = shape("gemver")["N"]


@_df_region()
def _gemver_top(
    A: fp16[N, N], u1: fp16[N], v1: fp16[N], u2: fp16[N], v2: fp16[N],
    y: fp16[N], x: fp16[N], w: fp16[N],
):
    @allo.work(mapping=[1], args=[A, u1, v1])
    def rank1_a(lA: fp16[N, N], lu1: fp16[N], lv1: fp16[N]):
        for i in range(N):
            for j in range(N):
                lA[i, j] = lA[i, j] + lu1[i] * lv1[j]

    @allo.work(mapping=[1], args=[A, u2, v2])
    def rank1_b(lA: fp16[N, N], lu2: fp16[N], lv2: fp16[N]):
        for i in range(N):
            for j in range(N):
                lA[i, j] = lA[i, j] + lu2[i] * lv2[j]

    @allo.work(mapping=[1], args=[A, y, x])
    def aty(lA: fp16[N, N], ly: fp16[N], lx: fp16[N]):
        for i in range(N):
            acc: fp16 = 0
            for j in range(N):
                acc += lA[j, i] * ly[j]
            lx[i] = acc

    @allo.work(mapping=[1], args=[A, x, w])
    def ax(lA: fp16[N, N], lx: fp16[N], lw: fp16[N]):
        for i in range(N):
            acc: fp16 = 0
            for j in range(N):
                acc += lA[i, j] * lx[j]
            lw[i] = acc


def build():
    return _gemver_top


# (rows, reduction) per MAC stage -- FOUR Samsung MAC groups: rank1_a, rank1_b
# (the two single-`mul` rank-1 accumulates), then the two GEMVs. One harness
# layer per group (zeros; cycles-measurement geometry).
STAGES = [(N, N), (N, N), (N, N), (N, N)]
