"""End-to-end vector-add on real Samsung PIMSimulator.

1. Generate fp16 inputs.
2. Invoke the compiled pim_driver (C++, links PIMSimulator).
3. Read output fp16 blob.
4. Assert np.allclose against numpy a+b.

PIMSimulator has a functional model (EltwisePIMKernel actually computes ADD),
so this is a true end-to-end correctness check.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

import numpy as np

from allo.pim.runtime.pimsim_driver import run_eltwise


def test_samsung_vadd():
    np.random.seed(3)
    N = 131072   # PIMSimulator eltwise minimum tile
    a = np.random.uniform(-1, 1, size=N).astype(np.float16)
    b = np.random.uniform(-1, 1, size=N).astype(np.float16)
    ref = (a.astype(np.float32) + b.astype(np.float32)).astype(np.float16)
    out = run_eltwise("ADD", a, b)
    err = np.max(np.abs(out.astype(np.float32) - ref.astype(np.float32)))
    print(f"samsung vadd: N={N} max_abs_err={err:.3e}")
    assert out.shape == (N,)
    assert err <= 1e-3, f"output diverges: max_abs_err={err:.3e}"
    print("ok  test_samsung_vadd")


def test_samsung_vmul():
    np.random.seed(4)
    N = 131072
    a = np.random.uniform(-1, 1, size=N).astype(np.float16)
    b = np.random.uniform(-1, 1, size=N).astype(np.float16)
    ref = (a.astype(np.float32) * b.astype(np.float32)).astype(np.float16)
    out = run_eltwise("MUL", a, b)
    err = np.max(np.abs(out.astype(np.float32) - ref.astype(np.float32)))
    print(f"samsung vmul: N={N} max_abs_err={err:.3e}")
    assert err <= 1e-3
    print("ok  test_samsung_vmul")


if __name__ == "__main__":
    test_samsung_vadd()
    test_samsung_vmul()
