# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-card batched rank-N Doitgen gate."""

import allo
import numpy as np
import pytest

from lib import cell
from lib.cost import bind_cost
from lib.targets import build_target


@pytest.mark.apu_g2_device
def test_doitgen_apu_g2_batched_rank_n_hardware(request, apu_g2_device_gate):
    workload = cell.load_workload(request.path.parent, "doitgen")
    target = build_target("apu_v2")
    compiled = allo.compile(workload.build(), target, bind_cost(target))

    r, q, p, s = workload.R, workload.Q, workload.P, workload.S
    rows = np.arange(r, dtype=np.uint64)[:, None, None]
    batches = np.arange(q, dtype=np.uint64)[None, :, None]
    depths = np.arange(s, dtype=np.uint64)[None, None, :]
    outputs = np.arange(p, dtype=np.uint64)[None, None, :]
    A = ((rows * 251 + batches * 509 + depths * 17 + 3) & 0xFFFF).astype(
        np.uint16
    )
    x = ((depths.reshape(s, 1) * 197 + outputs.reshape(1, p) * 613 + 7) & 0xFFFF).astype(
        np.uint16
    )
    initial = ((rows * 11 + batches * 13 + outputs * 29 + 5) & 0xFFFF).astype(
        np.uint16
    )
    expected = (
        np.einsum("rqs,sp->rqp", A.astype(np.uint64), x.astype(np.uint64))
        + initial.astype(np.uint64)
    ) & np.uint64(0xFFFF)
    expected = expected.astype(np.uint16)

    import conftest

    with conftest.apu_g2_board_lock():
        run = compiled(A, x, initial)

    np.testing.assert_array_equal(initial, expected)
    np.testing.assert_array_equal(run.extra["outputs"]["output"], expected)
    assert run.backend == "apu_v2"
    assert run.extra["hardware_tasks"] == 2
    assert run.extra["plan"]["batch_local_accumulator"] is False
    assert run.extra["vectorization_certificate"]["scalar_tensor_updates"] == 0
    assert run.extra["schedule"]["resident_accumulator"] is True
    assert compiled.execution_graph.metadata["program"] == "column_batched_gemm_u16"
