# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-card full-output SYRK gate through APUg2 rank-N dot tiles."""

import allo
import numpy as np
import pytest

from lib import cell
from lib.cost import bind_cost
from lib.targets import build_target


def _matmul_u16(lhs, rhs):
    return (lhs.astype(np.uint64) @ rhs.astype(np.uint64) & np.uint64(0xFFFF)).astype(
        np.uint16
    )


@pytest.mark.apu_g2_device
def test_syrk_apu_g2_full_output_rank_n_hardware(request, apu_g2_device_gate):
    workload = cell.load_workload(request.path.parent, "syrk")
    target = build_target("apu_v2")
    compiled = allo.compile(workload.build(), target, bind_cost(target))
    assert compiled.epilogue == (workload.SYRK_ALPHA, workload.SYRK_BETA)

    n, m = workload.N, workload.M
    rows = np.arange(n, dtype=np.uint64)[:, None]
    depths = np.arange(m, dtype=np.uint64)[None, :]
    columns = np.arange(n, dtype=np.uint64)[None, :]
    A = ((rows * 251 + depths * 17 + 3) & 0xFFFF).astype(np.uint16)
    C = ((rows * 13 + columns * 43 + 5) & 0xFFFF).astype(np.uint16)
    original = C.copy()

    dot = _matmul_u16(A, A.T.copy())
    expected = (
        np.uint64(workload.SYRK_ALPHA) * dot.astype(np.uint64)
        + np.uint64(workload.SYRK_BETA) * original.astype(np.uint64)
    ) & np.uint64(0xFFFF)
    expected = expected.astype(np.uint16)

    import conftest

    with conftest.apu_g2_board_lock():
        run = compiled(A, C)

    np.testing.assert_array_equal(C, expected)
    np.testing.assert_array_equal(run.extra["outputs"]["C"], expected)
    assert run.backend == "apu_v2"
    assert run.extra["hardware_tasks"] == compiled.execution_graph.metadata[
        "hardware_tasks"
    ]
    assert run.extra["schedule"]["resident_accumulator"] is True
    assert run.extra["epilogue"] == {
        "alpha": workload.SYRK_ALPHA,
        "beta": workload.SYRK_BETA,
    }
    assert run.extra["vectorization_certificate"]["scalar_tensor_updates"] == 0
    assert compiled.execution_graph.metadata["program"] == "column_batched_gemm_u16"
    predicted = compiled.estimate().cycles
    assert abs(run.cycles - predicted) / predicted <= 0.10
