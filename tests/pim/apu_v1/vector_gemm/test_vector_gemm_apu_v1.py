# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""APU v1 layout ablation: coalesced DMA versus broadcast-friendly plan.

Both plans come from the same ordinary Allo contraction and are selected with
explicit ``layout=``, the convention the paper producer uses. Their board
cycles are the ``vector_gemm_*`` rows of ``PAPER_GOLDEN`` (median of 3).
"""

import importlib.util
import statistics
import sys
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


def _paper_golden_rows():
    path = Path(__file__).resolve().parents[2] / "test_paper_golden.py"
    name = "vector_gemm_paper_golden"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_GOLDEN = _paper_golden_rows()
PLANS = {
    "temporal_dma_coalescing": "vector_gemm_temporal_dma_coalescing",
    "temporal_dma_coalescing_broadcast_friendly": (
        "vector_gemm_temporal_dma_coalescing_broadcast_friendly"
    ),
}


def _assert_broadcast_friendly_structure(compiled):
    assert compiled.selected_plan.iteration_layout.layout.out_sizes[0] == 32768
    assert compiled.selected_plan.metadata["output_tiles"] == 32
    assert compiled.realization.output_batches == 32
    source = compiled.device_source()
    assert source.count("direct_dma_l4_to_l1_32k") == 9  # 8 resident RHS + C
    assert "gvml_lookup_16" in source
    assert "gvml_duplicate_subgrp_16_grp_sgidx" in source
    assert "gvml_mul_f16" in source


@pytest.mark.apu_v1_device
@pytest.mark.parametrize("plan", sorted(PLANS))
def test_vector_gemm_apu_v1(plan, apu_v1_device_gate):
    row = _GOLDEN.golden_row(PLANS[plan])
    compiled = allo.compile(vector_gemm, build_apu_v1_target(), apu_v1_cost, layout=plan)
    assert compiled.selected_plan.name == plan
    if plan == "temporal_dma_coalescing_broadcast_friendly":
        _assert_broadcast_friendly_structure(compiled)

    rng = np.random.default_rng(25)
    left = (rng.standard_normal((M, K)) * 0.05).astype(np.float16)
    right = (rng.standard_normal((K, N)) * 0.05).astype(np.float16)
    expected = left.astype(np.float32) @ right.astype(np.float32)

    import conftest

    cycles = []
    with conftest.board_lock():
        for _ in range(3):
            result = np.zeros((M, N), dtype=np.float16)
            run = compiled(left, right, result)
            assert result.size == 1_048_576
            np.testing.assert_allclose(result, expected, rtol=5e-2, atol=5e-2)
            cycles.append(run.cycles)

    median = statistics.median(cycles)
    assert _GOLDEN._within(median, row), (row, cycles)
