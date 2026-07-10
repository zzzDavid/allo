# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-card TRMM gate through upper-triangular APUg2 dot tiles."""

import allo
import numpy as np
import pytest

from allo.pim.apu_g2_vector_program import APUG2TrmmCallable
from lib import cell
from lib.cost import bind_cost
from lib.targets import build_target


_U16 = np.uint64(0xFFFF)


def _trmm_reference(A, B, alpha):
    m = A.shape[0]
    rows = np.arange(m)[:, None]
    depth = np.arange(m)[None, :]
    triangular = np.where(depth == rows, np.uint16(1), np.uint16(0)).astype(np.uint16)
    triangular = np.where(depth > rows, A.T, triangular).astype(np.uint16)
    dot = (triangular.astype(np.uint64) @ B.astype(np.uint64)) & _U16
    expected = (np.uint64(alpha) * dot) & _U16
    return expected.astype(np.uint16)


@pytest.mark.apu_g2_device
def test_trmm_apu_g2_upper_triangular_hardware(request, apu_g2_device_gate):
    workload = cell.load_workload(request.path.parent, "trmm")
    target = build_target("apu_v2")
    compiled = allo.compile(workload.build(), target, bind_cost(target))
    assert isinstance(compiled, APUG2TrmmCallable)
    assert compiled.alpha == workload.TRMM_ALPHA

    m, n = workload.M, workload.N
    rows = np.arange(m, dtype=np.uint64)[:, None]
    columns_m = np.arange(m, dtype=np.uint64)[None, :]
    columns_n = np.arange(n, dtype=np.uint64)[None, :]
    A = ((rows * 251 + columns_m * 509 + 3) & _U16).astype(np.uint16)
    B = ((rows * 197 + columns_n * 613 + 7) & _U16).astype(np.uint16)
    expected = _trmm_reference(A, B, workload.TRMM_ALPHA)

    import conftest

    with conftest.apu_g2_board_lock():
        run = compiled(A, B)

    np.testing.assert_array_equal(B, expected)
    np.testing.assert_array_equal(run.extra["outputs"]["B"], expected)
    np.testing.assert_array_equal(run.extra["outputs"]["out"], expected)
    assert run.backend == "apu_v2"
    assert run.extra["hardware_tasks"] == compiled.execution_graph.metadata[
        "hardware_tasks"
    ]
    assert run.extra["schedule"]["resident_accumulator"] is True
    assert run.extra["contraction_topology"] == "trmm_upper_triangular"
    assert run.extra["epilogue"] == {"alpha": workload.TRMM_ALPHA, "beta": 0}
    assert run.extra["vectorization_certificate"]["scalar_tensor_updates"] == 0
    assert compiled.execution_graph.metadata["program"] == "column_batched_trmm_u16"
    assert compiled.execution_graph.metadata["calibration_extrapolated"] is False
    predicted = compiled.estimate().cycles
    assert abs(run.cycles - predicted) / predicted <= 0.10
