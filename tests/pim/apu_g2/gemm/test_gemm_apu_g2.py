# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-card GEMM gate through the column-batched uint16 schedule."""

import allo
import numpy as np
import pytest

from lib import cell
from lib.cost import bind_cost
from lib.targets import build_target


@pytest.mark.apu_g2_device
def test_gemm_apu_g2_column_batched_hardware(request, apu_g2_device_gate):
    workload = cell.load_workload(request.path.parent, "gemm")
    target = build_target("apu_v2")
    compiled = allo.compile(workload.build(), target, bind_cost(target))

    p, q, r = workload.P, workload.Q, workload.R
    rows = np.arange(p, dtype=np.uint64)[:, None]
    depths = np.arange(q, dtype=np.uint64)[None, :]
    columns = np.arange(r, dtype=np.uint64)[None, :]
    A = ((rows * 251 + depths * 509 + 3) & 0xFFFF).astype(np.uint16)
    B = ((depths.T * 197 + columns * 613 + 7) & 0xFFFF).astype(np.uint16)
    C = ((rows * 17 + columns * 29 + 11) & 0xFFFF).astype(np.uint16)
    expected = (
        A.astype(np.uint64) @ B.astype(np.uint64) + C.astype(np.uint64)
    ) & np.uint64(0xFFFF)
    expected = expected.astype(np.uint16)

    import conftest

    with conftest.apu_g2_board_lock():
        run = compiled(A, B, C)

    np.testing.assert_array_equal(C, expected)
    np.testing.assert_array_equal(run.extra["outputs"]["C"], expected)
    assert run.backend == "apu_v2"
    expected_tasks = ((r + 30) // 31) * ((q + 127) // 128)
    assert run.extra["hardware_tasks"] == expected_tasks
    assert run.extra["schedule"]["batch_columns"] == 31
    assert run.extra["schedule"]["resident_accumulator"] is True
    assert run.extra["vectorization_certificate"]["scalar_tensor_updates"] == 0
    assert run.extra["plan"]["flat_output_extent"] == p * r
    assert run.extra["plan"]["tile_count"] == 3
    assert compiled.execution_graph.metadata["program"] == "column_batched_gemm_u16"
