# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-device gate for target-neutral XNOR/popcount vector lowering."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import allo
from allo.pim.costs import apu_v1_cost
from allo.pim.targets import build_apu_v1_target


_SPEC = importlib.util.spec_from_file_location(
    "apu_v1_vector_binary_workload", Path(__file__).with_name("workload.py")
)
_WORKLOAD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_WORKLOAD)
K, M, N = _WORKLOAD.K, _WORKLOAD.M, _WORKLOAD.N
vector_binary = _WORKLOAD.vector_binary


def _reference(left, right):
    lhs = left.view(np.uint16)[:, None, :]
    rhs = np.ascontiguousarray(right.T).view(np.uint16)[None, :, :]
    xnor = np.ascontiguousarray(np.bitwise_not(np.bitwise_xor(lhs, rhs)))
    return (
        np.unpackbits(xnor.view(np.uint8), axis=-1)
        .reshape(M, N, K, 16)
        .sum(axis=(2, 3))
        .astype(np.int16)
    )


@pytest.mark.apu_v1_device
def test_vector_binary_apu_v1(apu_v1_device_gate):
    compiled = allo.compile(vector_binary, build_apu_v1_target(), apu_v1_cost)
    assert compiled.analysis.multiply_operation == "allo.xnor_popcount"
    assert compiled.analysis.packed_word_bits == 16
    source = compiled.device_source()
    for api in ("gvml_xor_16", "gvml_not_16", "gvml_popcount_16", "gvml_add_s16"):
        assert api in source
    assert "gvml_sl_imm_16" not in source
    assert "gvml_sub_s16" not in source

    rng = np.random.default_rng(31)
    left = rng.integers(-32768, 32767, (M, K), dtype=np.int16)
    right = rng.integers(-32768, 32767, (K, N), dtype=np.int16)
    result = np.zeros((M, N), dtype=np.int16)

    import conftest

    with conftest.board_lock():
        run = compiled(left, right, result)

    np.testing.assert_array_equal(result, _reference(left, right))
    assert run.cycles is not None and run.cycles > 0
