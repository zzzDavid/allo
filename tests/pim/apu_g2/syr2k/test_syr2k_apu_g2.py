# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-card full-output SYR2K gate through chained APUg2 dot tiles."""

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
def test_syr2k_apu_g2_full_output_chain_hardware(request, apu_g2_device_gate):
    workload = cell.load_workload(request.path.parent, "syr2k")
    target = build_target("apu_v2")
    compiled = allo.compile(workload.build(), target, bind_cost(target))
    assert compiled.stage_epilogues == (
        (workload.SYR2K_ALPHA, workload.SYR2K_BETA),
        (workload.SYR2K_ALPHA, 1),
    )

    n, m = workload.N, workload.M
    rows = np.arange(n, dtype=np.uint64)[:, None]
    depths = np.arange(m, dtype=np.uint64)[None, :]
    columns = np.arange(n, dtype=np.uint64)[None, :]
    A = ((rows * 251 + depths * 17 + 3) & 0xFFFF).astype(np.uint16)
    B = ((rows * 197 + depths * 29 + 7) & 0xFFFF).astype(np.uint16)
    C = ((rows * 13 + columns * 43 + 5) & 0xFFFF).astype(np.uint16)
    original = C.copy()

    ab = _matmul_u16(A, B.T.copy())
    first = (
        np.uint64(workload.SYR2K_ALPHA) * ab.astype(np.uint64)
        + np.uint64(workload.SYR2K_BETA) * original.astype(np.uint64)
    ) & np.uint64(0xFFFF)
    first = first.astype(np.uint16)
    ba = _matmul_u16(B, A.T.copy())
    expected = (
        np.uint64(workload.SYR2K_ALPHA) * ba.astype(np.uint64) + first.astype(np.uint64)
    ) & np.uint64(0xFFFF)
    expected = expected.astype(np.uint16)

    import conftest

    with conftest.apu_g2_board_lock():
        run = compiled(A, B, C)

    np.testing.assert_array_equal(C, expected)
    np.testing.assert_array_equal(run.extra["outputs"]["C"], expected)
    assert run.backend == "apu_v2"
    assert run.extra["hardware_tasks"] == compiled.execution_graph.metadata[
        "hardware_tasks"
    ]
    assert run.extra["schedule"]["resident_accumulator"] is True
    assert run.extra["stage_epilogues"] == [
        {"alpha": workload.SYR2K_ALPHA, "beta": workload.SYR2K_BETA},
        {"alpha": workload.SYR2K_ALPHA, "beta": 1},
    ]
    assert run.extra["vectorization_certificate"]["scalar_tensor_updates"] == 0
    assert compiled.execution_graph.metadata["program"] == (
        "transport_aware_contraction_chain_u16"
    )
    predicted = compiled.estimate().cycles
    assert abs(run.cycles - predicted) / predicted <= 0.10
