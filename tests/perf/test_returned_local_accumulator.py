# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Returned local reduction outputs fail for their ABI, not a missing fill."""
# Allo annotations allocate arrays; these functions are compiled, not run by Python.
# pylint: disable=unsupported-assignment-operation,used-before-assignment
import pytest
import allo
from allo.ir.types import float16
from allo.pim.costs import aim_cost, apu_v1_cost, samsung_cost, upmem_cost
from allo.pim.targets import (
    build_aim_target,
    build_apu_v1_target,
    build_samsung_target,
    build_upmem_target,
)
from allo.spmw_match import UnsupportedWorkloadError
from allo.spmw_match_engine import match_workload, require_match_coverage


def zero_filled(W: float16[8, 8], x: float16[8]) -> float16[8]:
    y: float16[8] = 0.0
    for i in range(8):
        for k in allo.reduction(8):
            y[i] += W[i, k] * x[k]
    return y


def nonzero_filled(W: float16[8, 8], x: float16[8]) -> float16[8]:
    y: float16[8] = 1.0
    for i in range(8):
        for k in allo.reduction(8):
            y[i] += W[i, k] * x[k]
    return y


def late_filled(W: float16[8, 8], x: float16[8]) -> float16[8]:
    y: float16[8]
    for i in range(8):
        for k in allo.reduction(8):
            y[i] += W[i, k] * x[k]
    y = 0.0
    return y


@pytest.mark.parametrize("workload", [zero_filled, nonzero_filled, late_filled])
@pytest.mark.parametrize(
    "target,cost",
    [
        (build_samsung_target, samsung_cost),
        (build_aim_target, aim_cost),
        (build_upmem_target, upmem_cost),
        (build_apu_v1_target, apu_v1_cost),
    ],
)
def test_returned_local_output_names_the_abi_boundary(workload, target, cost):
    with pytest.raises(UnsupportedWorkloadError) as caught:
        allo.compile(workload, target(), cost, backend="virtual")
    message = str(caught.value)
    assert "returned local output buffer is unsupported" in message
    assert "pass the output as an argument" in message
    assert "never initialized" not in message
    assert "initialize the accumulator with literal 0" not in message


def test_late_direct_fill_is_not_a_zero_initializer():
    schedule = allo.customize(zero_filled)
    body = list(schedule.top_func.entry_block.operations)
    fill = next(op for op in body if op.operation.name == "linalg.fill")
    loop = next(op for op in body if op.operation.name == "affine.for")
    fill.operation.move_after(loop.operation)
    assert schedule.module.operation.verify()
    target = build_samsung_target()
    trace = match_workload(target, schedule.module)
    with pytest.raises(UnsupportedWorkloadError, match="returned local output buffer"):
        require_match_coverage(trace, target=target)
