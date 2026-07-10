# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-card one-task BiCG through structured independent contractions."""

import allo
import numpy as np
import pytest

from lib import cell
from lib.cost import bind_cost
from lib.targets import build_target


def _dot(matrix, vector):
    return (
        np.sum(
            matrix.astype(np.uint64) * vector.astype(np.uint64)[None, :],
            axis=1,
            dtype=np.uint64,
        )
        & np.uint64(0xFFFF)
    ).astype(np.uint16)


@pytest.mark.apu_g2_device
def test_bicg_apu_g2_one_task_independent_hardware(request, apu_g2_device_gate):
    workload = cell.load_workload(request.path.parent, "bicg")
    target = build_target("apu_v2")
    compiled = allo.compile(workload.build(), target, bind_cost(target))

    m, n = workload.M, workload.N
    rows = np.arange(m, dtype=np.uint64)[:, None]
    columns = np.arange(n, dtype=np.uint64)[None, :]
    A = ((rows * 251 + columns * 509 + rows * columns * 17 + 3) & 0xFFFF).astype(
        np.uint16
    )
    p = ((np.arange(n, dtype=np.uint64) * 997 + 11) & 0xFFFF).astype(np.uint16)
    r = ((np.arange(m, dtype=np.uint64) * 1877 + 41) & 0xFFFF).astype(np.uint16)
    q = np.zeros(m, dtype=np.uint16)
    s = np.zeros(n, dtype=np.uint16)
    expected_q = _dot(A, p)
    expected_s = _dot(np.ascontiguousarray(A.T), r)

    import conftest

    with conftest.apu_g2_board_lock():
        run = compiled(A, p, r, q, s)

    np.testing.assert_array_equal(q, expected_q)
    np.testing.assert_array_equal(s, expected_s)
    np.testing.assert_array_equal(run.extra["outputs"]["q"], expected_q)
    np.testing.assert_array_equal(run.extra["outputs"]["s"], expected_s)
    assert run.backend == "apu_v2"
    assert run.extra["hardware_tasks"] == 1
    assert run.extra["contraction_topology"] == "independent"
    assert run.extra["vectorization_certificate"]["scalar_tensor_updates"] == 0
    assert run.extra["vectorization_certificate"]["fully_vectorized"] is True
    assert compiled.execution_graph.metadata["program"] == (
        "independent_contractions_u16"
    )
    assert compiled.execution_graph.metadata["work_grid"] == (16,)
    predicted = compiled.estimate().cycles
    assert abs(run.cycles - predicted) / predicted <= 0.05

    source = run.extra["sources"]["device/apu_g2_u16_dot_tile.cc"]
    assert "L1Vectors right_low" in source
    assert "mul(" in source and "sum(" in source
    assert "simulator" not in source.lower()
