# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-card gate for modulo accumulation across reduction chunks."""

import allo
import numpy as np
import pytest

from lib import cell
from lib.cost import bind_cost
from lib.targets import build_target


@pytest.mark.apu_g2_device
def test_wide_reduction_accumulates_two_physical_chunks(request, apu_g2_device_gate):
    workload = cell.load_workload(request.path.parent, "reduction_chunks")
    target = build_target("apu_v2")
    compiled = allo.compile(workload.build(), target, bind_cost(target))

    rows = np.arange(workload.M, dtype=np.uint64)[:, None]
    depths = np.arange(workload.K, dtype=np.uint64)[None, :]
    A = ((rows * 251 + depths * 509 + 3) & 0xFFFF).astype(np.uint16)
    x = (
        (np.arange(workload.K, dtype=np.uint64) * 613 + 7) & 0xFFFF
    ).astype(np.uint16)
    y = (
        (np.arange(workload.M, dtype=np.uint64) * 29 + 11) & 0xFFFF
    ).astype(np.uint16)
    expected = (
        A.astype(np.uint64) @ x.astype(np.uint64) + y.astype(np.uint64)
    ) & np.uint64(0xFFFF)
    expected = expected.astype(np.uint16)

    import conftest

    with conftest.apu_g2_board_lock():
        result = compiled(A, x, y)

    np.testing.assert_array_equal(y, expected)
    np.testing.assert_array_equal(result.extra["outputs"]["y"], expected)
    assert compiled.plan.tiling.reduction_tile_count == 2
    assert compiled.execution_graph.metadata["hardware_tasks"] == 2
    assert result.extra["hardware_tasks"] == 2
    assert result.extra["vectorization_certificate"]["scalar_tensor_updates"] == 0
