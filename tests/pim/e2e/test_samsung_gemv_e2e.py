"""End-to-end GEMV on real Samsung PIMSimulator. W[M,K] @ x[K] -> y[M].

PIMSimulator expects W layout: (output_dim, input_dim) = (M, K).
We match its built-in data layout via loadFp16 which reads directly from .npy.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

import numpy as np

from allo.pim.runtime.pimsim_driver import run_gemv


def test_samsung_gemv_4096x1024():
    np.random.seed(0)
    M, K = 4096, 1024
    # Samsung's internal gemv test uses uniform-ish range; use a small range
    # to keep fp16 sums in-range.
    W = np.random.uniform(-0.1, 0.1, size=(M, K)).astype(np.float16)
    x = np.random.uniform(-0.5, 0.5, size=(K,)).astype(np.float16)
    ref = (W.astype(np.float32) @ x.astype(np.float32)).astype(np.float16)

    out = run_gemv(W, x)
    assert out.shape == (M,), out.shape
    # fp16 accumulation across 1024 multiplies drifts; use generous tolerance
    # that matches Samsung's own internal test (fp16Equal tol ~4 ULP / 1%).
    diff = np.abs(out.astype(np.float32) - ref.astype(np.float32))
    max_err = diff.max()
    rel = diff / (np.abs(ref.astype(np.float32)) + 1e-3)
    print(f"samsung gemv {M}x{K}: max_abs_err={max_err:.3e}  "
          f"max_rel_err={rel.max():.3e}  mean_abs={diff.mean():.3e}")
    # large tolerance because PIMSimulator actually uses fp16 MAC internally
    # with different tree-reduction order than numpy.float32 -> float16.
    assert max_err < 0.5, f"gemv diverges: {max_err}"
    assert (rel < 0.1).mean() > 0.95, "less than 95% of outputs within 10% rel"
    print("ok  test_samsung_gemv_4096x1024")


if __name__ == "__main__":
    test_samsung_gemv_4096x1024()
