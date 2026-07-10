# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-card MVT gate through ordinary Allo MLIR GEMV recognition."""

import allo
import numpy as np
import pytest

from lib import cell
from lib.cost import bind_cost
from lib.targets import build_target


def _accumulated_gemv(matrix, vector, initial):
    product = (
        (matrix.astype(np.uint64) @ vector.astype(np.uint64)) & np.uint64(0xFFFF)
    ).astype(np.uint16)
    return (
        (initial.astype(np.uint64) + product.astype(np.uint64)) & np.uint64(0xFFFF)
    ).astype(np.uint16)


@pytest.mark.apu_g2_device
def test_mvt_apu_g2_ordinary_allo_hardware(request, apu_g2_device_gate):
    workload = cell.load_workload(request.path.parent, "mvt")
    target = build_target("apu_v2")
    compiled = allo.compile(workload.build(), target, bind_cost(target))

    n = workload.N
    rows = np.arange(n, dtype=np.uint64)[:, None]
    columns = np.arange(n, dtype=np.uint64)[None, :]
    A = ((rows * 251 + columns * 509 + rows * columns * 17 + 3) & 0xFFFF).astype(
        np.uint16
    )
    y1 = ((np.arange(n, dtype=np.uint64) * 997 + 11) & 0xFFFF).astype(np.uint16)
    y2 = ((np.arange(n, dtype=np.uint64) * 1237 + 29) & 0xFFFF).astype(np.uint16)
    x1 = ((np.arange(n, dtype=np.uint64) * 1877 + 41) & 0xFFFF).astype(np.uint16)
    x2 = ((np.arange(n, dtype=np.uint64) * 1999 + 53) & 0xFFFF).astype(np.uint16)
    expected_x1 = _accumulated_gemv(A, y1, x1)
    expected_x2 = _accumulated_gemv(A.T, y2, x2)

    import conftest

    with conftest.apu_g2_board_lock():
        run = compiled(A, y1, y2, x1, x2)

    np.testing.assert_array_equal(x1, expected_x1)
    np.testing.assert_array_equal(x2, expected_x2)
    np.testing.assert_array_equal(run.extra["outputs"]["x1"], expected_x1)
    np.testing.assert_array_equal(run.extra["outputs"]["x2"], expected_x2)
    assert run.backend == "apu_v2"
    assert run.cycles > 0
    assert compiled.execution_graph.metadata["program"] == "independent_contractions_u16"
    assert run.extra["contraction_topology"] == "independent"
