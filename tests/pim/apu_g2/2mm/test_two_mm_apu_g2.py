# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-card 2mm gate through chained APUg2 rank-N dot tiles."""

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
def test_two_mm_apu_g2_chained_rank_n_hardware(request, apu_g2_device_gate):
    workload = cell.load_workload(request.path.parent, "2mm")
    target = build_target("apu_v2")
    compiled = allo.compile(workload.build(), target, bind_cost(target))
    assert compiled.stage_epilogues == (
        (1, 1),
        (workload.TWO_MM_BETA, workload.TWO_MM_ALPHA),
    )

    p, q, r, s = workload.P, workload.Q, workload.R, workload.S
    rows = np.arange(p, dtype=np.uint64)[:, None]
    q_axis = np.arange(q, dtype=np.uint64)[None, :]
    r_axis = np.arange(r, dtype=np.uint64)
    s_axis = np.arange(s, dtype=np.uint64)[None, :]
    A = ((rows * 251 + q_axis * 17 + 3) & 0xFFFF).astype(np.uint16)
    B = ((q_axis.T * 197 + r_axis[None, :] * 29 + 7) & 0xFFFF).astype(np.uint16)
    C = ((r_axis[:, None] * 193 + s_axis * 31 + 11) & 0xFFFF).astype(np.uint16)
    D = ((rows * 13 + s_axis * 43 + 5) & 0xFFFF).astype(np.uint16)
    original_d = D.copy()

    ab = _matmul_u16(A, B)
    abc = _matmul_u16(ab, C)
    expected = (
        np.uint64(workload.TWO_MM_BETA) * abc.astype(np.uint64)
        + np.uint64(workload.TWO_MM_ALPHA) * original_d.astype(np.uint64)
    ) & np.uint64(0xFFFF)
    expected = expected.astype(np.uint16)

    import conftest

    with conftest.apu_g2_board_lock():
        run = compiled(A, B, C, D)

    np.testing.assert_array_equal(D, expected)
    np.testing.assert_array_equal(run.extra["outputs"]["D"], expected)
    np.testing.assert_array_equal(run.extra["outputs"]["AB"], ab)
    assert run.backend == "apu_v2"
    assert run.extra["hardware_tasks"] == compiled.execution_graph.metadata[
        "hardware_tasks"
    ]
    assert run.extra["contraction_topology"] == "linked_chain"
    assert run.extra["stage_epilogues"] == [
        {"alpha": 1, "beta": 1},
        {"alpha": workload.TWO_MM_BETA, "beta": workload.TWO_MM_ALPHA},
    ]
    assert run.extra["vectorization_certificate"]["scalar_tensor_updates"] == 0
    assert compiled.execution_graph.metadata["program"] == (
        "transport_aware_contraction_chain_u16"
    )
    assert run.extra["schedule"]["resident_accumulator"] is True
    assert run.cycles is not None and run.cycles > 0
