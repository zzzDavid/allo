# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-card gate for the direct-VL64 uint16 PolyBench GESUMMV slice."""

import hashlib

import allo
import numpy as np
import pytest

from allo.pim import APUG2Program
from allo.pim.apu_g2_vector_program import APUG2ChunkedGesummvCallable
from lib import cell
from lib.cost import bind_cost
from lib.targets import build_target


def _reference(A, B, x, alpha, beta):
    """Stored-stage semantics in Z/(2**16), widened to avoid host overflow."""

    tmp_a = ((A.astype(np.uint64) @ x.astype(np.uint64)) & 0xFFFF).astype(np.uint16)
    tmp_b = ((B.astype(np.uint64) @ x.astype(np.uint64)) & 0xFFFF).astype(np.uint16)
    out = (
        (
            np.uint64(alpha) * tmp_a.astype(np.uint64)
            + np.uint64(beta) * tmp_b.astype(np.uint64)
        )
        & 0xFFFF
    ).astype(np.uint16)
    return tmp_a, tmp_b, out


@pytest.mark.apu_g2_device
def test_gesummv_apu_g2_hardware(request, apu_g2_device_gate):
    workload = cell.load_workload(request.path.parent, "gesummv")
    target = build_target("apu_v2")
    compiled = allo.compile(workload.build(), target, bind_cost(target))

    n = workload.N
    rows = np.arange(n, dtype=np.uint64)[:, None]
    reductions = np.arange(n, dtype=np.uint64)[None, :]
    x = np.arange(n, dtype=np.uint16)
    A = ((rows * reductions + 1) % n).astype(np.uint16)
    B = ((rows * reductions + 2) % n).astype(np.uint16)
    tmp_a, tmp_b, expected = _reference(A, B, x, workload.ALPHA, workload.BETA)
    out = np.empty(n, dtype=np.uint16)

    import conftest

    with conftest.apu_g2_board_lock():
        result = compiled(A, B, x, out)

    np.testing.assert_array_equal(result.extra["outputs"]["tmp_a"], tmp_a)
    np.testing.assert_array_equal(result.extra["outputs"]["tmp_b"], tmp_b)
    np.testing.assert_array_equal(result.extra["outputs"]["out"], expected)
    np.testing.assert_array_equal(out, expected)
    assert hashlib.sha256(tmp_a.astype("<u2", copy=False).tobytes()).hexdigest() == (
        "93507f1fd52dbb3318ecde0651e8ac7e85f54c3d382e1200f6f1fa06e9502eab"
    )
    assert hashlib.sha256(tmp_b.astype("<u2", copy=False).tobytes()).hexdigest() == (
        "5475fd5e9b56eec7004c1991cdd55eb474a44f800555c94b5f3fd3027f355374"
    )
    assert hashlib.sha256(expected.astype("<u2", copy=False).tobytes()).hexdigest() == (
        "9628121cee1e587b4a9d315d84f4c5f309f1e410d8c5dd266d2786a5c310dee6"
    )

    assert result.backend == "apu_v2"
    assert result.cycles > 0
    assert result.extra["repetitions"] == 8
    assert result.extra["total_ticks"] > 0
    assert result.extra["final_pipeline_ticks"] > 0

    device_source = result.extra["sources"]["device/apu_g2_u16_gesummv.cc"]
    assert "gsi::g2_64vl" in device_source
    assert device_source.count("mul(") >= 1
    assert device_source.count("sum(") >= 1
    assert device_source.count("add(") >= 1
    assert "sum(v.sum_source, log_block_size" in device_source
    assert result.extra["layout"]["log_block_size"] == 7
    assert "seu_barrier();" in device_source
    assert "simulator" not in device_source.lower()


@pytest.mark.apu_g2_device
def test_gesummv_streams_reductions_larger_than_one_dot_tile(apu_g2_device_gate):
    rows, reduction = 17, 257
    alpha, beta = 5, 4
    target = build_target("apu_v2")
    compiled = allo.compile(
        APUG2Program(
            operation="gesummv_u16",
            shape=(rows, reduction),
            alpha=alpha,
            beta=beta,
            repetitions=2,
        ),
        target,
        bind_cost(target),
    )
    assert isinstance(compiled, APUG2ChunkedGesummvCallable)

    row_axis = np.arange(rows, dtype=np.uint64)[:, None]
    reduction_axis = np.arange(reduction, dtype=np.uint64)[None, :]
    A = ((row_axis * 251 + reduction_axis * 509 + 3) & 0xFFFF).astype(np.uint16)
    B = ((row_axis * 197 + reduction_axis * 613 + 7) & 0xFFFF).astype(np.uint16)
    x = ((np.arange(reduction, dtype=np.uint64) * 29 + 11) & 0xFFFF).astype(
        np.uint16
    )
    tmp_a, tmp_b, expected = _reference(A, B, x, alpha, beta)
    out = np.zeros(rows, dtype=np.uint16)

    import conftest

    with conftest.apu_g2_board_lock():
        result = compiled(A, B, x, out)

    np.testing.assert_array_equal(result.extra["outputs"]["tmp_a"], tmp_a)
    np.testing.assert_array_equal(result.extra["outputs"]["tmp_b"], tmp_b)
    np.testing.assert_array_equal(result.extra["outputs"]["out"], expected)
    np.testing.assert_array_equal(out, expected)
    assert result.extra["hardware_tasks"] == 7
    assert compiled.execution_graph.metadata["execution"] == "streaming_gemv_composition"
