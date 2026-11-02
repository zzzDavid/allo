# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The AttAcc GPT-13B decode kernels as ordinary ``@allo`` sources.

Kernel instances are per HBM3-PIM device: GPT-13B's 40 heads are split over
the five devices (8 heads each at batch 1, 32 head instances at batch 4), as
in the paper's study (``paper/latex/sections/5-evaluation.tex``,
"Cost Modeling and LLM Decode on AttAcc").  Data is FP16.

* ``gemv_region(H, K, N)``: ``y[h, n] = sum_k W[h, n, k] * x[h, k]``.  QKV
  generation is ``K = 5120, N = 384`` (q, k, v of one 128-wide head), the
  output projection ``K = 128, N = 5120``.
* ``attention_region(H, D, L)``: one decode step of attention scoring,
  ``s = q K^T``, ``p = softmax(s)``, ``o = p V`` over an L-token KV cache.
  The softmax is written without max subtraction: the max-stabilized form
  needs an accumulator initialized to ``-inf``, which the matcher does not
  accept yet (only literal-zero initializers), so it is not offered here.
  The ``1/sqrt(D)`` score scale is assumed folded into ``q`` by the caller.

``GPT13B_INSTANCES`` is the paper's 500-instance list (4 GEMV + 2 x 248
attention shapes).  ``paper_selection.csv`` next to this file holds the
tiling the paper's selector chose for each instance (an evaluation oracle;
nothing in ``allo`` reads it).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import allo
from allo.dataflow import region as _df_region
from allo.ir.types import float16
from allo.spmw_match import exp

PAPER_SELECTION = Path(__file__).with_name("paper_selection.csv")


@dataclass(frozen=True)
class AttAccInstance:
    id: str
    family: str  # "gemv" or "attention"
    rows: int  # heads x batch on one device
    reduction: int
    outputs: int

    def region(self):
        if self.family == "gemv":
            return gemv_region(self.rows, self.reduction, self.outputs)
        return attention_region(self.rows, self.reduction, self.outputs)


def _instances():
    out = []
    for heads in (8, 32):
        out.append(AttAccInstance(f"qkv_gemv_nh{heads}_in5120_out384", "gemv", heads, 5120, 384))
        out.append(AttAccInstance(f"proj_gemv_nh{heads}_in128_out5120", "gemv", heads, 128, 5120))
    for heads in (8, 32):
        for length in range(144, 4096 + 1, 16):
            out.append(
                AttAccInstance(
                    f"score_attention_nh{heads}_in128_out{length}",
                    "attention",
                    heads,
                    128,
                    length,
                )
            )
    return tuple(out)


GPT13B_INSTANCES = _instances()


def paper_selection() -> dict[str, list[list[int]]]:
    """Instance id -> the tiling the paper's selector chose."""
    with PAPER_SELECTION.open() as handle:
        return {row["id"]: json.loads(row["tiling"]) for row in csv.DictReader(handle)}


# Factories rebind their shape arguments to ``_aa_*`` names: Allo resolves
# kernel names from caller frames at customize time, so a caller local such
# as ``H`` would otherwise shadow them.


def gemv_region(H: int, K: int, N: int):
    """Per-head GEMV ``y[h, n] = sum_k W[h, n, k] * x[h, k]`` (FP16)."""
    _aa_h, _aa_k, _aa_n = H, K, N

    @_df_region()
    def attacc_gemv(
        W: float16[_aa_h, _aa_n, _aa_k], x: float16[_aa_h, _aa_k], y: float16[_aa_h, _aa_n]
    ):
        @allo.work(mapping=[1], args=[W, x, y])
        def gemv(
            lW: float16[_aa_h, _aa_n, _aa_k],
            lx: float16[_aa_h, _aa_k],
            ly: float16[_aa_h, _aa_n],
        ):
            for h in range(_aa_h):
                for n in range(_aa_n):
                    acc: float16 = 0
                    for k in range(_aa_k):
                        acc += lW[h, n, k] * lx[h, k]
                    ly[h, n] = acc

    return attacc_gemv


def attention_region(H: int, D: int, L: int):
    """One decode step of attention scoring over an L-token cache (FP16)."""
    _aa_h, _aa_d, _aa_l = H, D, L

    @_df_region()
    def attacc_attention(
        q: float16[_aa_h, _aa_d],
        kc: float16[_aa_h, _aa_l, _aa_d],
        vc: float16[_aa_h, _aa_l, _aa_d],
        o: float16[_aa_h, _aa_d],
    ):
        @allo.work(mapping=[1], args=[q, kc, vc, o])
        def attention(
            lq: float16[_aa_h, _aa_d],
            lk: float16[_aa_h, _aa_l, _aa_d],
            lv: float16[_aa_h, _aa_l, _aa_d],
            lo: float16[_aa_h, _aa_d],
        ):
            s: float16[_aa_h, _aa_l] = 0.0
            for h in range(_aa_h):
                for t in range(_aa_l):
                    acc: float16 = 0.0
                    for d in range(_aa_d):
                        acc += lq[h, d] * lk[h, t, d]
                    s[h, t] = acc
            for h in range(_aa_h):
                total: float16 = 0.0
                for t in range(_aa_l):
                    s[h, t] = exp(s[h, t])
                for t in range(_aa_l):
                    total += s[h, t]
                for t in range(_aa_l):
                    s[h, t] = s[h, t] / total
            for h in range(_aa_h):
                for d in range(_aa_d):
                    acc2: float16 = 0.0
                    for t in range(_aa_l):
                        acc2 += s[h, t] * lv[h, t, d]
                    lo[h, d] = acc2

    return attacc_attention
