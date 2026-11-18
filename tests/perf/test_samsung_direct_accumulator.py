# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct output zero initialization reaches the Samsung executable sink."""
import numpy as np
import pytest
import allo
from allo.ir.types import float16
from allo.pim.costs import samsung_cost
from allo.pim.targets import build_samsung_target


def fused(W: float16[48, 40], x: float16[40], y: float16[48]):
    for i in range(48):
        y[i] = 0.0
        for k in allo.reduction(40):
            y[i] += W[i, k] * x[k]


def separate(W: float16[48, 40], x: float16[40], y: float16[48]):
    for i in range(48):
        y[i] = 0.0
    for i in range(48):
        for k in allo.reduction(40):
            y[i] += W[i, k] * x[k]


@pytest.mark.parametrize("workload", [fused, separate])
def test_zero_initialized_argument_executes_exactly(workload):
    rng = np.random.default_rng(9214)
    matrix = rng.integers(-3, 4, (48, 40)).astype(np.float16)
    vector = rng.integers(-3, 4, 40).astype(np.float16)
    output = np.full(48, 17, dtype=np.float16)
    # Every product and partial sum is an exactly representable FP16 integer.
    expected = (matrix.astype(np.int32) @ vector.astype(np.int32)).astype(np.float16)
    compiled = allo.compile(workload, build_samsung_target(), samsung_cost)
    matches = [m for m in compiled.trace.matches if m.target_op_name == "MAC"]
    assert matches and all(m.extra["accumulator_init"] == "zero" for m in matches)
    assert all(m.extra["output_sinks"][0][0] == "y" for m in matches)
    result = compiled(matrix, vector, output)
    assert result.kind == "simulator"
    np.testing.assert_array_equal(output.view(np.uint16), expected.view(np.uint16))
    np.testing.assert_array_equal(
        result.outputs["y"].view(np.uint16), expected.view(np.uint16)
    )
