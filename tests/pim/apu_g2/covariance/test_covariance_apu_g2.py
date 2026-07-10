# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench covariance through the public APUg2 compile path."""

import allo
import numpy as np
import pytest

from allo.pim.apu_g2_covariance_runtime import covariance_u16_reference
from allo.pim.apu_g2_vector_program import APUG2CovarianceCallable
from lib import cell
from lib.cost import bind_cost
from lib.targets import build_target


@pytest.mark.apu_g2_device
def test_covariance_apu_g2_compiled_hardware(request, apu_g2_device_gate):
    workload = cell.load_workload(request.path.parent, "covariance")
    target = build_target("apu_v2")
    compiled = allo.compile(workload.build(), target, bind_cost(target))
    assert isinstance(compiled, APUG2CovarianceCallable)

    m, n = workload.M, workload.N
    rows = np.arange(n, dtype=np.uint64)[:, None]
    cols = np.arange(m, dtype=np.uint64)[None, :]
    data = ((rows * 251 + cols * 509 + 17) & 0xFFFF).astype(np.uint16)
    reference = covariance_u16_reference(data)
    mean = np.zeros(m, dtype=np.uint16)
    cov = np.zeros((m, m), dtype=np.uint16)

    import conftest

    with conftest.apu_g2_board_lock():
        result = compiled(data, mean, cov)

    np.testing.assert_array_equal(data, reference["centered"])
    np.testing.assert_array_equal(mean, reference["mean"])
    np.testing.assert_array_equal(cov, reference["covariance"])
    np.testing.assert_array_equal(
        result.extra["outputs"]["data"], reference["centered"]
    )
    np.testing.assert_array_equal(result.extra["outputs"]["mean"], reference["mean"])
    np.testing.assert_array_equal(
        result.extra["outputs"]["cov"], reference["covariance"]
    )
    np.testing.assert_array_equal(
        result.extra["outputs"]["out"], reference["covariance"]
    )
    assert result.backend == "apu_v2"
    assert result.cycles > 0
    assert result.extra["total_ticks"] > 0
    assert result.extra["hardware_tasks"] == 9
    assert result.extra["primitive_programs"] == {
        "dot_tile": 0,
        "column_batched_gemm": 2,
        "fill": 2,
        "div": 2,
        "sub": 1,
    }
    assert result.extra["tiling"]["gram_hardware_tasks"] == 3
    assert result.extra["vectorization_certificate"]["scalar_tensor_updates"] == 0
    assert result.extra["compiled_workload"] is True
    assert result.extra["contraction_topology"] == "covariance_primitive_chain"
    assert compiled.execution_graph.metadata["program"] == "transport_aware_covariance_u16"
    assert compiled.execution_graph.metadata["work_grid"] == (16,)
    assert compiled.execution_graph.metadata["hardware_tasks"] == 9
    assert compiled.execution_graph.metadata["calibrated_device_ticks"] is False
    assert compiled.estimate().cycles > 0
