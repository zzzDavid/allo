# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench correlation through the public APUg2 compile path."""

import allo
import numpy as np
import pytest

from allo.pim.apu_g2_correlation_runtime import correlation_u16_reference
from allo.pim.apu_g2_vector_program import APUG2CorrelationCallable
from lib import cell
from lib.cost import bind_cost
from lib.targets import build_target


@pytest.mark.apu_g2_device
def test_correlation_apu_g2_compiled_hardware(request, apu_g2_device_gate):
    workload = cell.load_workload(request.path.parent, "correlation")
    target = build_target("apu_v2")
    compiled = allo.compile(workload.build(), target, bind_cost(target))
    assert isinstance(compiled, APUG2CorrelationCallable)

    m, n = workload.M, workload.N
    rows = np.arange(n, dtype=np.uint64)[:, None]
    cols = np.arange(m, dtype=np.uint64)[None, :]
    data = ((rows * 251 + cols * 509 + 17) & 0xFFFF).astype(np.uint16)
    reference = correlation_u16_reference(data, data.copy(), data.copy())
    corr = np.zeros((m, m), dtype=np.uint16)

    import conftest

    with conftest.apu_g2_board_lock():
        result = compiled(data.copy(), data.copy(), data.copy(), corr)

    np.testing.assert_array_equal(corr, reference["correlation"])
    np.testing.assert_array_equal(
        result.extra["outputs"]["corr"], reference["correlation"]
    )
    np.testing.assert_array_equal(
        result.extra["outputs"]["out"], reference["correlation"]
    )
    np.testing.assert_array_equal(
        result.extra["outputs"]["normalized"], reference["normalized"]
    )
    assert result.backend == "apu_v2"
    assert result.cycles > 0
    assert result.extra["total_ticks"] > 0
    assert result.extra["hardware_tasks"] == 17
    assert result.extra["primitive_programs"] == {
        "dot_tile": 1,
        "column_batched_gemm": 2,
        "fill": 3,
        "div": 3,
        "sub": 2,
        "sqrt": 1,
        "minmax": 1,
        "mul": 1,
        "select_lt": 1,
    }
    assert result.extra["tiling"]["corr_hardware_tasks"] == 3
    assert result.extra["vectorization_certificate"]["scalar_tensor_updates"] == 0
    assert result.extra["compiled_workload"] is True
    assert result.extra["contraction_topology"] == "correlation_primitive_chain"
    assert compiled.execution_graph.metadata["program"] == "transport_aware_correlation_u16"
    assert compiled.execution_graph.metadata["work_grid"] == (16,)
    assert compiled.execution_graph.metadata["hardware_tasks"] == 17
    assert compiled.execution_graph.metadata["calibrated_device_ticks"] is False
    assert compiled.estimate().cycles > 0
