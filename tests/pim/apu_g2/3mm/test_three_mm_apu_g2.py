# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-card 3mm gate through APUg2 rank-N DAG dot tiles."""

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
def test_three_mm_apu_g2_dag_rank_n_hardware(request, apu_g2_device_gate):
    workload = cell.load_workload(request.path.parent, "3mm")
    target = build_target("apu_v2")
    compiled = allo.compile(workload.build(), target, bind_cost(target))

    p, q, r, nt, s = workload.P, workload.Q, workload.R, workload.NT, workload.S
    p_axis = np.arange(p, dtype=np.uint64)[:, None]
    q_axis = np.arange(q, dtype=np.uint64)[None, :]
    r_axis = np.arange(r, dtype=np.uint64)
    s_axis = np.arange(s, dtype=np.uint64)
    t_axis = np.arange(nt, dtype=np.uint64)[None, :]
    A = ((p_axis * 251 + q_axis * 17 + 3) & 0xFFFF).astype(np.uint16)
    B = ((q_axis.T * 197 + r_axis[None, :] * 29 + 7) & 0xFFFF).astype(np.uint16)
    C = ((r_axis[:, None] * 193 + s_axis[None, :] * 31 + 11) & 0xFFFF).astype(np.uint16)
    D = ((s_axis[:, None] * 211 + t_axis * 37 + 13) & 0xFFFF).astype(np.uint16)
    output = np.zeros((p, nt), dtype=np.uint16)

    ab = _matmul_u16(A, B)
    cd = _matmul_u16(C, D)
    expected = _matmul_u16(ab, cd)

    import conftest

    with conftest.apu_g2_board_lock():
        run = compiled(A, B, C, D, output)

    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(run.extra["outputs"]["AB"], ab)
    np.testing.assert_array_equal(run.extra["outputs"]["CD"], cd)
    np.testing.assert_array_equal(run.extra["outputs"]["output"], expected)
    assert run.backend == "apu_v2"
    assert run.extra["hardware_tasks"] == compiled.execution_graph.metadata[
        "hardware_tasks"
    ]
    assert run.extra["contraction_topology"] == "dag"
    assert run.extra["vectorization_certificate"]["scalar_tensor_updates"] == 0
    assert compiled.execution_graph.metadata["program"] == (
        "transport_aware_contraction_chain_u16"
    )
    assert run.extra["schedule"]["resident_accumulator"] is True
    assert run.cycles is not None and run.cycles > 0
