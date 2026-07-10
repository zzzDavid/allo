# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-card Gemver gate through a structured APUg2 VL64 chain."""

import allo
import numpy as np
import pytest

from allo.pim.apu_g2_vector_program import APUG2GemverCallable
from lib import cell
from lib.cost import bind_cost
from lib.targets import build_target


_U16 = np.uint64(0xFFFF)


def _dot_u16(lhs, rhs):
    return (lhs.astype(np.uint64) @ rhs.astype(np.uint64) & _U16).astype(np.uint16)


def _gemver_reference(A, u1, u2, v1, v2, x, y, w, z):
    outer = (
        u1.astype(np.uint64)[:, None] * v1.astype(np.uint64)[None, :]
        + u2.astype(np.uint64)[:, None] * v2.astype(np.uint64)[None, :]
    )
    A1 = ((A.astype(np.uint64) + outer) & _U16).astype(np.uint16)
    x_after_gemv = (
        x.astype(np.uint64) + _dot_u16(A1.T.copy(), y).astype(np.uint64)
    ) & _U16
    x1 = ((x_after_gemv + z.astype(np.uint64)) & _U16).astype(np.uint16)
    w1 = (w.astype(np.uint64) + _dot_u16(A1, x1).astype(np.uint64)) & _U16
    return A1, x1, w1.astype(np.uint16)


@pytest.mark.apu_g2_device
def test_gemver_apu_g2_structured_chain_hardware(request, apu_g2_device_gate):
    workload = cell.load_workload(request.path.parent, "gemver")
    target = build_target("apu_v2")
    compiled = allo.compile(workload.build(), target, bind_cost(target))
    assert isinstance(compiled, APUG2GemverCallable)

    n = workload.N
    rows = np.arange(n, dtype=np.uint64)[:, None]
    columns = np.arange(n, dtype=np.uint64)[None, :]
    vector = np.arange(n, dtype=np.uint64)

    A = ((rows * 251 + columns * 509 + 3) & _U16).astype(np.uint16)
    u1 = ((vector * 17 + 5) & _U16).astype(np.uint16)
    u2 = ((vector * 19 + 7) & _U16).astype(np.uint16)
    v1 = ((vector * 23 + 11) & _U16).astype(np.uint16)
    v2 = ((vector * 29 + 13) & _U16).astype(np.uint16)
    x = ((vector * 31 + 17) & _U16).astype(np.uint16)
    y = ((vector * 37 + 19) & _U16).astype(np.uint16)
    w = ((vector * 41 + 23) & _U16).astype(np.uint16)
    z = ((vector * 43 + 29) & _U16).astype(np.uint16)
    original_y = y.copy()
    expected_A, expected_x, expected_w = _gemver_reference(
        A.copy(), u1, u2, v1, v2, x.copy(), y, w.copy(), z
    )

    import conftest

    with conftest.apu_g2_board_lock():
        run = compiled(A, u1, u2, v1, v2, x, y, w, z)

    np.testing.assert_array_equal(A, expected_A)
    np.testing.assert_array_equal(x, expected_x)
    np.testing.assert_array_equal(w, expected_w)
    np.testing.assert_array_equal(y, original_y)
    np.testing.assert_array_equal(run.extra["outputs"]["A"], expected_A)
    np.testing.assert_array_equal(run.extra["outputs"]["x"], expected_x)
    np.testing.assert_array_equal(run.extra["outputs"]["w"], expected_w)
    np.testing.assert_array_equal(run.extra["outputs"]["out"], expected_w)
    assert run.backend == "apu_v2"
    assert run.extra["hardware_tasks"] == 4
    assert run.extra["contraction_topology"] == "gemver_affine_chain"
    assert run.extra["vectorization_certificate"]["scalar_tensor_updates"] == 0
    assert compiled.execution_graph.metadata["program"] == "gemver_u16"
    assert compiled.execution_graph.metadata["calibration_extrapolated"] is False
    predicted = compiled.estimate().cycles
    assert abs(run.cycles - predicted) / predicted <= 0.10
