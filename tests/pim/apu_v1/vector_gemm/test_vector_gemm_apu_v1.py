# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-device gate for MLIR -> LinearLayout -> GVML vector compilation."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import allo
from allo.pim.costs import apu_v1_cost
from allo.pim.targets import build_apu_v1_target

_SPEC = importlib.util.spec_from_file_location(
    "apu_v1_vector_gemm_workload", Path(__file__).with_name("workload.py")
)
_WORKLOAD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_WORKLOAD)
K, M, N = _WORKLOAD.K, _WORKLOAD.M, _WORKLOAD.N
vector_gemm = _WORKLOAD.vector_gemm


@pytest.mark.apu_v1_device
def test_vector_gemm_apu_v1(apu_v1_device_gate):
    target = build_apu_v1_target()
    compiled = allo.compile(vector_gemm, target, apu_v1_cost)

    assert compiled.selected_plan.name == ("temporal_dma_coalescing_broadcast_friendly")
    assert compiled.selected_plan.iteration_layout.layout.out_sizes[0] == 32768
    assert compiled.selected_plan.metadata["output_tiles"] == 32
    assert compiled.realization.output_batches == 32
    source = compiled.device_source()
    assert source.count("direct_dma_l4_to_l1_32k") == 9  # 8 resident RHS + C
    assert "gvml_lookup_16" in source
    assert "gvml_duplicate_subgrp_16_grp_sgidx" in source
    assert "gvml_mul_f16" in source

    rng = np.random.default_rng(25)
    left = (rng.standard_normal((M, K)) * 0.05).astype(np.float16)
    right = (rng.standard_normal((K, N)) * 0.05).astype(np.float16)
    result = np.zeros((M, N), dtype=np.float16)
    expected = left.astype(np.float32) @ right.astype(np.float32)

    import conftest

    with conftest.board_lock():
        run = compiled(left, right, result)

    np.testing.assert_allclose(result, expected, rtol=5e-2, atol=5e-2)
    assert run.cycles is not None and run.cycles > 0
