# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os

import numpy as np
import pytest

from allo.ir.types import uint16


def _load_example(name):
    path = os.path.join(
        os.path.dirname(__file__), os.pardir, "examples", "polybench", f"{name}.py"
    )
    spec = importlib.util.spec_from_file_location(f"polybench_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mvt_instantiates_with_uint16_and_wraps_modulo_2_16():
    mvt = _load_example("mvt")
    N = 8
    mod = mvt.mvt(uint16, N).build()

    rng = np.random.default_rng(0)
    A = rng.integers(0, 1 << 16, size=(N, N), dtype=np.uint16)
    y1 = rng.integers(0, 1 << 16, size=(N,), dtype=np.uint16)
    y2 = rng.integers(0, 1 << 16, size=(N,), dtype=np.uint16)
    x1 = rng.integers(0, 1 << 16, size=(N,), dtype=np.uint16)
    x2 = rng.integers(0, 1 << 16, size=(N,), dtype=np.uint16)
    x1_out = np.zeros(N, dtype=np.uint16)
    x2_out = np.zeros(N, dtype=np.uint16)

    mod(A, A.copy(), y1, y2, x1.copy(), x2.copy(), x1_out, x2_out)

    wide = A.astype(np.int64)
    x1_ref = (x1.astype(np.int64) + wide @ y1.astype(np.int64)) % (1 << 16)
    x2_ref = (x2.astype(np.int64) + wide.T @ y2.astype(np.int64)) % (1 << 16)
    np.testing.assert_array_equal(x1_out, x1_ref.astype(np.uint16))
    np.testing.assert_array_equal(x2_out, x2_ref.astype(np.uint16))


if __name__ == "__main__":
    pytest.main([__file__])
