# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Samsung readback contract: zero-valued outputs are results, missing ones fail."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import allo
from allo.pim.costs import samsung_cost
from allo.pim.targets import build_samsung_target
from allo.spmw_samsung import (
    _samsung_read_generic_outbin,
    _samsung_read_reduce_outbin,
)


def _paper_gemv():
    path = Path(__file__).resolve().parent / "workload.py"
    spec = importlib.util.spec_from_file_location("samsung_readback_workload", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CASES["gemv_m4096_k256"]


def test_reduce_readback_returns_all_zero_result(tmp_path):
    M, N, m_pad = 8, 1, 16
    blob = tmp_path / "out.bin"
    np.zeros(m_pad * N * 16, dtype=np.float16).tofile(blob)
    outputs = _samsung_read_reduce_outbin(blob, M, N, m_pad)
    assert set(outputs) == {"y"}
    np.testing.assert_array_equal(outputs["y"], np.zeros(M, dtype=np.float32))


def test_reduce_readback_rejects_absent_and_truncated_blob(tmp_path):
    blob = tmp_path / "out.bin"
    with pytest.raises(RuntimeError, match=r"is absent; expected 512 bytes, got 0"):
        _samsung_read_reduce_outbin(blob, 8, 1, 16, "driver said hello")
    np.zeros(10, dtype=np.float16).tofile(blob)
    with pytest.raises(RuntimeError, match=r"too short; expected 512 bytes, got 20") as err:
        _samsung_read_reduce_outbin(blob, 8, 1, 16, "driver said hello")
    assert "driver said hello" in str(err.value)


def test_generic_readback_returns_all_zero_and_rejects_short_blob(tmp_path):
    blob = tmp_path / "out.bin"
    np.zeros(32, dtype=np.float16).tofile(blob)
    np.testing.assert_array_equal(
        _samsung_read_generic_outbin(blob, 32)["out"], np.zeros(32, np.float32)
    )
    with pytest.raises(RuntimeError, match=r"too short; expected 128 bytes, got 64"):
        _samsung_read_generic_outbin(blob, 64)
    with pytest.raises(RuntimeError, match="is absent"):
        _samsung_read_generic_outbin(tmp_path / "missing.bin", 32)


def test_paper_gemv_zero_input_writes_zero_output(samsung_sim_gate):
    workload, M, K = _paper_gemv()
    compiled = allo.compile(workload, build_samsung_target(), samsung_cost)

    W = (np.random.default_rng(0).standard_normal((M, K)) * 0.1).astype(np.float16)
    x = np.zeros(K, dtype=np.float16)
    y = np.ones(M, dtype=np.float16)
    result = compiled(W, x, y)

    assert result.kind == "simulator"
    assert result.cycles > 0
    assert set(result.outputs) == {"y"}
    np.testing.assert_array_equal(y, np.zeros(M, dtype=np.float16))
