# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paper Samsung HBM-PIM GEMV shapes through public ``allo.compile``."""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pytest

import allo
from allo.pim.costs import samsung_cost
from allo.pim.targets import build_samsung_target


def _load_workload():
    path = Path(__file__).resolve().parent / "workload.py"
    spec = importlib.util.spec_from_file_location("samsung_paper_gemv_workload", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _paper_cycles():
    path = Path(__file__).resolve().parents[2] / "test_paper_golden.py"
    name = "samsung_gemv_paper_golden"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return {
        row.case_id: row.expected
        for row in module.PAPER_GOLDEN
        if row.backend == "samsung_hbm_pim"
    }


CASES = _load_workload().CASES
PAPER_CYCLES = _paper_cycles()
COMPILE_BUDGET_SECONDS = 120.0


@pytest.mark.parametrize("case", sorted(CASES))
def test_gemv_samsung_hbm_pim(case, samsung_sim_gate):
    workload, M, K = CASES[case]

    started = time.perf_counter()
    compiled = allo.compile(workload, build_samsung_target(), samsung_cost)
    assert time.perf_counter() - started < COMPILE_BUDGET_SECONDS

    kernels = compiled.placement_ranking.kernels
    assert len(kernels) == 1
    assert kernels[0].bucket_count == 128
    assert kernels[0].fallback is None
    # 24 enumerated placements x the restage/resident residency knob that the
    # whole-trace liveness enables inside autoschedule.
    assert len(kernels[0].scores) == 48

    rng = np.random.default_rng(0)
    W = (rng.standard_normal((M, K)) * 0.1).astype(np.float16)
    x = (rng.standard_normal(K) * 0.1).astype(np.float16)
    y = np.zeros(M, dtype=np.float16)
    result = compiled(W, x, y)

    assert result.cycles == PAPER_CYCLES[case]
    np.testing.assert_allclose(
        y.astype(np.float32),
        W.astype(np.float32) @ x.astype(np.float32),
        rtol=1e-2,
        atol=1e-2,
    )
