"""End-to-end self-attention on real UPMEM uPIMulator — multi-stage DPU codegen.

Two DPU kernels chained through host-side softmax:
  stage 1 (DSLATTN_QKT): scores[s] = sum_d Q[d] * K[s,d]
  host        : probs_q = round(softmax(scores / sqrt(D)) * 2^15)
  stage 2 (DSLATTN_AV ): out[d]    = sum_s (probs_q[s] * V[s,d]) >> 15

Each stage's DPU task.c, support/common.h, CMakeLists, and the Go Assemblable
are emitted by `pimdsl.runtime.upmem_attention.emit_attention_benchmarks()`.
The Go assemblables read Python-written input files from $UPM/dsl_data/ —
this is how the host-computed probs flow into stage 2.

Verification oracle = same integer arithmetic in numpy (`numpy_reference`).
A successful run means every DPU-computed int32 element is bit-identical to
the numpy oracle — no rounding slack allowed.
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

import numpy as np

from allo.pim.runtime.upmem_attention import (
    emit_attention_benchmarks, run_attention, numpy_reference)
from allo.pim.runtime.upmem_codegen import rebuild_upimulator


def test_upmem_attention_e2e():
    S, D = 64, 16
    # keep values small so int32 sums don't overflow
    np.random.seed(11)
    Q = np.random.randint(-4, 5, size=(D,), dtype=np.int32)
    K = np.random.randint(-4, 5, size=(S, D), dtype=np.int32)
    V = np.random.randint(-1024, 1024, size=(S, D), dtype=np.int32)

    emit_attention_benchmarks(max_s=S, max_d=D)
    if not shutil.which("docker"):
        print("skip: docker not available")
        return
    rebuild_upimulator()

    out_dpu, scores_dpu, probs_q = run_attention(Q, K, V, num_tasklets=16)
    out_ref, scores_ref, probs_ref = numpy_reference(Q, K, V)

    print(f"S={S} D={D}")
    print(f"  stage1 scores: DPU first 5 = {scores_dpu[:5]}")
    print(f"               ref first 5 = {scores_ref[:5]}")
    print(f"  probs_q first 5 = {probs_q[:5]}  (scale=2^15)")
    print(f"  stage2 out   : DPU first 5 = {out_dpu[:5]}")
    print(f"               ref first 5 = {out_ref[:5]}")
    score_eq = (scores_dpu == scores_ref).all()
    prob_eq = (probs_q == probs_ref).all()
    out_eq = (out_dpu == out_ref).all()
    print(f"  scores match: {score_eq}   probs match: {prob_eq}   out match: {out_eq}")
    assert score_eq, f"stage1 scores diverge: n_diff={np.sum(scores_dpu != scores_ref)}"
    assert prob_eq, "host-computed probs differ between run and ref (shouldn't happen)"
    assert out_eq, f"stage2 out diverges: n_diff={np.sum(out_dpu != out_ref)}"
    print("ok  test_upmem_attention_e2e")


if __name__ == "__main__":
    test_upmem_attention_e2e()
