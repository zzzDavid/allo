# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-card SYMM gate through lower-triangular APUg2 dot tiles."""

import allo
import numpy as np
import pytest

from allo.pim.apu_g2_vector_program import APUG2SymmCallable
from lib import cell
from lib.cost import bind_cost
from lib.targets import build_target


_U16 = np.uint64(0xFFFF)


def _symm_reference(A, B, C, alpha, beta):
    rows = np.arange(A.shape[0])[:, None]
    columns = np.arange(A.shape[0])[None, :]
    logical_a = np.where(columns <= rows, A, A.T)
    dot = (logical_a.astype(np.uint64) @ B.astype(np.uint64)) & _U16
    expected = (np.uint64(alpha) * dot + np.uint64(beta) * C.astype(np.uint64)) & _U16
    return expected.astype(np.uint16)


@pytest.mark.apu_g2_device
def test_symm_apu_g2_lower_symmetric_hardware(request, apu_g2_device_gate):
    workload = cell.load_workload(request.path.parent, "symm")
    target = build_target("apu_v2")
    compiled = allo.compile(workload.build(), target, bind_cost(target))
    assert isinstance(compiled, APUG2SymmCallable)
    assert (compiled.alpha, compiled.beta) == (workload.SYMM_ALPHA, workload.SYMM_BETA)

    m, n = workload.M, workload.N
    rows = np.arange(m, dtype=np.uint64)[:, None]
    columns_m = np.arange(m, dtype=np.uint64)[None, :]
    columns_n = np.arange(n, dtype=np.uint64)[None, :]
    A = ((rows * 251 + columns_m * 509 + 3) & _U16).astype(np.uint16)
    B = ((rows * 197 + columns_n * 613 + 7) & _U16).astype(np.uint16)
    C = ((rows * 17 + columns_n * 29 + 11) & _U16).astype(np.uint16)
    expected = _symm_reference(A, B, C, workload.SYMM_ALPHA, workload.SYMM_BETA)

    import conftest

    with conftest.apu_g2_board_lock():
        run = compiled(A, B, C)

    np.testing.assert_array_equal(C, expected)
    np.testing.assert_array_equal(run.extra["outputs"]["C"], expected)
    np.testing.assert_array_equal(run.extra["outputs"]["out"], expected)
    assert run.backend == "apu_v2"
    assert run.extra["hardware_tasks"] == compiled.execution_graph.metadata[
        "hardware_tasks"
    ]
    assert run.extra["schedule"]["resident_accumulator"] is True
    assert run.extra["contraction_topology"] == "symm_lower_symmetric"
    assert run.extra["epilogue"] == {
        "alpha": workload.SYMM_ALPHA,
        "beta": workload.SYMM_BETA,
    }
    assert run.extra["vectorization_certificate"]["scalar_tensor_updates"] == 0
    assert compiled.execution_graph.metadata["program"] == "column_batched_symm_u16"
    assert compiled.execution_graph.metadata["calibration_extrapolated"] is False
    predicted = compiled.estimate().cycles
    assert abs(run.cycles - predicted) / predicted <= 0.10
