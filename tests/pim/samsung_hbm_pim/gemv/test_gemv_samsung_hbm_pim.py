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
from allo.spmw_samsung import samsung_vendor_gemv_layout


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
def test_gemv_samsung_hbm_pim(case, samsung_sim_gate, record_property):
    """Vendor-equivalent placement pinned with `layout=` reproduces the paper
    cycles; the cost-selected default is correct and its cycles recorded."""
    workload, M, K = CASES[case]
    target = build_samsung_target()

    started = time.perf_counter()
    # The paper rows permit regrouping partial sums (5-evaluation.tex); the
    # strict default selects the ordered program (test_samsung_reduction_order).
    compiled = allo.compile(
        workload, target, samsung_cost, reduction_order="reassociate"
    )
    assert time.perf_counter() - started < COMPILE_BUDGET_SECONDS

    kernels = compiled.placement_ranking.kernels
    assert len(kernels) == 1
    assert kernels[0].bucket_count == 128
    assert kernels[0].fallback is None
    # 24 REDUCE placements plus 2 ordered ones, x the restage/resident
    # residency knob that the whole-trace liveness enables inside autoschedule.
    assert len(kernels[0].scores) == 52
    vendor = allo.compile(
        workload,
        target,
        samsung_cost,
        layout=samsung_vendor_gemv_layout(target, compiled.trace),
        reduction_order="reassociate",
    )

    rng = np.random.default_rng(0)
    W = (rng.standard_normal((M, K)) * 0.1).astype(np.float16)
    x = (rng.standard_normal(K) * 0.1).astype(np.float16)
    cycles = {}
    for name, artifact in (("vendor", vendor), ("default", compiled)):
        y = np.zeros(M, dtype=np.float16)
        result = artifact(W, x, y)
        np.testing.assert_allclose(
            y.astype(np.float32),
            W.astype(np.float32) @ x.astype(np.float32),
            rtol=1e-2,
            atol=1e-2,
        )
        cycles[name] = result.cycles
    record_property("vendor_cycles", cycles["vendor"])
    record_property("default_cycles", cycles["default"])
    print(f"SAMSUNG_GEMV {case} vendor={cycles['vendor']} default={cycles['default']}")

    assert cycles["vendor"] == PAPER_CYCLES[case]
    assert cycles["default"] > 0
