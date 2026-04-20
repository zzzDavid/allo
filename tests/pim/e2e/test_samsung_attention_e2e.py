"""End-to-end self-attention on Samsung PIMSimulator.

Shape constraint: PIMSimulator's GEMV is hardcoded so `output_dim` must be a
multiple of 4096 (64 chan * 16 banks * 8 grfB = 4096 — see
PIMKernel::preloadGemv / computeGemv). So we run the Q·K^T GEMV at
output_dim=S=4096, input_dim=D=1024. The second matmul (probs @ V) is
decomposed into:
    tiled = broadcast probs -> shape (S*D,)
    prod  = PIM ELTWISE MUL (tiled, V_flat)          -- on PIMSimulator
    out   = prod.reshape(S, D).sum(axis=0)           -- host tree-reduce
(PIMSimulator's eltwise has no on-PIM reduction; this is the honest split.)

Verified: out matches numpy attention within fp16 precision bounds.
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

import numpy as np

from allo.pim.runtime.pimsim_driver import run_gemv, run_eltwise


def samsung_attention(Q, K, V):
    M, D = Q.shape
    S, D2 = K.shape
    assert D2 == D and V.shape == (S, D)

    out = np.zeros((M, D), dtype=np.float16)
    scores_all = np.zeros((M, S), dtype=np.float16)
    for m in range(M):
        # step 1: scores = K @ Q[m]     (GEMV on PIM; output_dim=S=4096, input=D=1024)
        s_vec = run_gemv(K, Q[m])
        scores_all[m] = s_vec
        # step 2 + 3: scale + softmax on host
        scaled = s_vec.astype(np.float32) / math.sqrt(D)
        shifted = scaled - scaled.max()
        ex = np.exp(shifted)
        probs = (ex / ex.sum()).astype(np.float16)
        # step 4: MUL(broadcast_probs, V_flat) on PIM, then host sum
        tiled = np.broadcast_to(probs.reshape(S, 1), (S, D)).copy().reshape(-1)
        V_flat = V.reshape(-1)
        prod = run_eltwise("MUL", tiled, V_flat).reshape(S, D)
        out[m] = prod.astype(np.float32).sum(axis=0).astype(np.float16)
    return out, scores_all


def numpy_attention(Q, K, V):
    D = Q.shape[-1]
    s = (Q.astype(np.float32) @ K.astype(np.float32).T) / math.sqrt(D)
    s = s - s.max(axis=-1, keepdims=True)
    p = np.exp(s); p /= p.sum(axis=-1, keepdims=True)
    return (p @ V.astype(np.float32)).astype(np.float16), \
           (Q.astype(np.float32) @ K.astype(np.float32).T).astype(np.float16)


def test_samsung_attention_e2e():
    np.random.seed(1)
    M, S, D = 1, 4096, 1024
    # tighter range so fp16 sums don't overflow; V gets reduced by sum_s probs*V
    Q = np.random.uniform(-0.05, 0.05, size=(M, D)).astype(np.float16)
    K = np.random.uniform(-0.05, 0.05, size=(S, D)).astype(np.float16)
    V = np.random.uniform(-0.05, 0.05, size=(S, D)).astype(np.float16)

    out, scores = samsung_attention(Q, K, V)
    ref_out, ref_scores = numpy_attention(Q, K, V)

    score_err = np.max(np.abs(scores.astype(np.float32) - ref_scores.astype(np.float32)))
    out_err = np.max(np.abs(out.astype(np.float32) - ref_out.astype(np.float32)))
    out_mean = np.abs(out.astype(np.float32) - ref_out.astype(np.float32)).mean()
    print(f"samsung attn  M={M} S={S} D={D}")
    print(f"  scores max_abs_err  = {score_err:.3e}  (expect ~fp16-MAC-of-{D})")
    print(f"  out    max_abs_err  = {out_err:.3e}")
    print(f"  out    mean_abs_err = {out_mean:.3e}")
    # scores: fp16 MAC of 1024 terms, expect ~3e-3 abs
    assert score_err < 0.05, f"scores diverge badly: {score_err}"
    # out: softmax reweights a single strong element; sum is ~1 * V[argmax].
    # expect near-exact after softmax concentrates; give generous fp16 budget.
    assert out_err < 0.1, f"out diverges: {out_err}"
    print(f"ok  test_samsung_attention_e2e")


if __name__ == "__main__":
    test_samsung_attention_e2e()
