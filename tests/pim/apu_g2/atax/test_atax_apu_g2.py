# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-card fused ATAX gate through ordinary Allo compilation."""

import allo
import numpy as np
import pytest

from lib import cell
from lib.cost import bind_cost
from lib.targets import build_target


def _reference(matrix, vector, initial):
    tmp = (
        (matrix.astype(np.uint64) @ vector.astype(np.uint64)) & np.uint64(0xFFFF)
    ).astype(np.uint16)
    out = (
        initial.astype(np.uint64) + matrix.astype(np.uint64).T @ tmp.astype(np.uint64)
    ) & np.uint64(0xFFFF)
    return tmp, out.astype(np.uint16)


@pytest.mark.apu_g2_device
def test_atax_apu_g2_one_task_resident_hardware(request, apu_g2_device_gate):
    workload = cell.load_workload(request.path.parent, "atax")
    target = build_target("apu_v2")
    compiled = allo.compile(workload.build(), target, bind_cost(target))

    m, n = workload.M, workload.N
    rows = np.arange(m, dtype=np.uint64)[:, None]
    columns = np.arange(n, dtype=np.uint64)[None, :]
    A = ((rows * 251 + columns * 509 + rows * columns * 17 + 3) & 0xFFFF).astype(
        np.uint16
    )
    x = ((np.arange(n, dtype=np.uint64) * 997 + 11) & 0xFFFF).astype(np.uint16)
    y = ((np.arange(n, dtype=np.uint64) * 1877 + 41) & 0xFFFF).astype(np.uint16)
    expected_tmp, expected_y = _reference(A, x, y.copy())

    import conftest

    with conftest.apu_g2_board_lock():
        run = compiled(A, x, y)

    np.testing.assert_array_equal(y, expected_y)
    np.testing.assert_array_equal(run.extra["outputs"]["tmp"], expected_tmp)
    np.testing.assert_array_equal(run.extra["outputs"]["y"], expected_y)
    assert run.backend == "apu_v2"
    assert run.extra["hardware_tasks"] == 1
    assert run.extra["resident_intermediate"] is True
    assert run.extra["repetitions"] == 4
    assert run.extra["layout"]["stage2_resident_rebroadcast"] is True
    assert compiled.execution_graph.metadata["program"] == "atax_u16"
    assert compiled.execution_graph.metadata["work_grid"] == (16,)
    predicted = compiled.estimate().cycles
    assert abs(run.cycles - predicted) / predicted <= 0.05

    device_source = run.extra["sources"]["device/apu_g2_u16_atax.cc"]
    host_source = run.extra["sources"]["host_atax.cc"]
    assert "GSI_LIBRARY_ENTRY_POINT(tenon_apu_g2_u16_atax" in device_source
    assert "squeeze_rows_inplace" in device_source
    assert "spread_blk" in device_source
    assert "mul(" in device_source and "sum(" in device_source
    assert host_source.count('run_task("tenon_apu_g2_u16_atax"') == 1
    assert "simulator" not in device_source.lower()
