# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Returned local reduction outputs fail for their ABI, not a missing fill."""
# Allo annotations allocate arrays; these functions are compiled, not run by Python.
# pylint: disable=unsupported-assignment-operation,used-before-assignment,unsubscriptable-object
import pytest
import re
import numpy as np
import allo
from allo.ir.types import float16, int32
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


def separate_zero_store(W: float16[8, 8], x: float16[8]) -> float16[8]:
    y: float16[8]
    for i in range(8):
        y[i] = 0.0
    for i in range(8):
        for k in allo.reduction(8):
            y[i] += W[i, k] * x[k]
    return y


def fused_zero_store(W: float16[8, 8], x: float16[8]) -> float16[8]:
    y: float16[8]
    for i in range(8):
        y[i] = 0.0
        for k in allo.reduction(8):
            y[i] += W[i, k] * x[k]
    return y


def scalar_to_return(W: float16[8, 8], x: float16[8]) -> float16[8]:
    y: float16[8]
    for i in range(8):
        acc: float16 = 0.0
        for k in allo.reduction(8):
            acc += W[i, k] * x[k]
        y[i] = acc
    return y


def elementwise_return(a: float16[64], b: float16[64]) -> float16[64]:
    c: float16[64]
    for i in range(64):
        c[i] = a[i] + b[i]
    return c


def copied_return(W: float16[8, 8], x: float16[8]) -> float16[8]:
    t: float16[8] = 0.0
    y: float16[8]
    for i in range(8):
        for k in allo.reduction(8):
            t[i] += W[i, k] * x[k]
    for i in range(8):
        y[i] = t[i]
    return y


def separate_integer_return(W: int32[8, 8], x: int32[8]) -> int32[8]:
    y: int32[8]
    for i in range(8):
        y[i] = 0
    for i in range(8):
        for k in allo.reduction(8):
            y[i] += W[i, k] * x[k]
    return y


def integer_return(W: int32[8, 8], x: int32[8]) -> int32[8]:
    y: int32[8]
    for i in range(8):
        y[i] = 0
        for k in allo.reduction(8):
            y[i] += W[i, k] * x[k]
    return y


def integer_elementwise_return(a: int32[64], b: int32[64]) -> int32[64]:
    c: int32[64]
    for i in range(64):
        c[i] = a[i] + b[i]
    return c


def temporary_fill(W: float16[8, 8], x: float16[8], y: float16[8]):
    t: float16[8] = 0.0
    for i in range(8):
        for k in allo.reduction(8):
            t[i] += W[i, k] * x[k]
        y[i] = t[i]


def temporary_store(W: float16[8, 8], x: float16[8], y: float16[8]):
    t: float16[8]
    for i in range(8):
        t[i] = 0.0
        for k in allo.reduction(8):
            t[i] += W[i, k] * x[k]
        y[i] = t[i]


def temporary_nonzero(W: float16[8, 8], x: float16[8], y: float16[8]):
    t: float16[8] = 7.0
    for i in range(8):
        for k in allo.reduction(8):
            t[i] += W[i, k] * x[k]
        y[i] = t[i]


def output_argument(W: float16[8, 8], x: float16[8], y: float16[8]):
    for i in range(8):
        acc: float16 = 0.0
        for k in allo.reduction(8):
            acc += W[i, k] * x[k]
        y[i] = acc


TARGETS = [
    (build_samsung_target, samsung_cost),
    (build_aim_target, aim_cost),
    (build_upmem_target, upmem_cost),
    (build_apu_v1_target, apu_v1_cost),
]
ABI_REASON = (
    "a returned local output buffer is unsupported by the matched output ABI; "
    "pass the output as an argument"
)


@pytest.mark.parametrize(
    "workload",
    [
        zero_filled,
        nonzero_filled,
        late_filled,
        separate_zero_store,
        fused_zero_store,
        scalar_to_return,
        elementwise_return,
        integer_return,
        separate_integer_return,
        copied_return,
        integer_elementwise_return,
    ],
)
@pytest.mark.parametrize("target,cost", TARGETS)
def test_returned_local_output_names_the_abi_boundary(
    workload, target, cost, monkeypatch
):
    def no_candidates(*args, **kwargs):
        pytest.fail("output ABI refusal must precede candidate enumeration")

    monkeypatch.setattr("allo.spmw_codegen.autoschedule", no_candidates)
    with pytest.raises(UnsupportedWorkloadError) as caught:
        allo.compile(workload, target(), cost, backend="virtual")
    message = str(caught.value)
    assert message == ABI_REASON
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


@pytest.mark.parametrize("target,cost", TARGETS)
@pytest.mark.parametrize("late,nonzero", [(False, False), (True, False), (False, True)])
def test_local_fill_initialization_is_checked(target, cost, late, nonzero):
    schedule = allo.customize(temporary_nonzero if nonzero else temporary_fill)
    if late:
        body = list(schedule.top_func.entry_block.operations)
        fill = next(op for op in body if op.operation.name == "linalg.fill")
        loop = next(op for op in body if op.operation.name == "affine.for")
        fill.operation.move_after(loop.operation)
        assert schedule.module.operation.verify()
    trace = match_workload(target(), schedule.module)
    if late or nonzero:
        reason = "comes after the reduction" if late else "nonzero constant"
        with pytest.raises(UnsupportedWorkloadError, match=reason) as caught:
            require_match_coverage(trace, target=target())
        assert "never initialized" not in str(caught.value)
    else:
        # Copy layout may still be unsupported. Initialization itself must
        # reach the same classification as element stores, not a false error.
        assert not any(
            "initializer" in m or "never initialized" in m
            for _functions, m in trace.coverage.problems
        )
        accumulating = [m for m in trace.matches if "accumulator_init" in m.extra]
        assert accumulating
        assert all(m.extra["accumulator_init"] == "zero" for m in accumulating)


@pytest.mark.parametrize("target,cost", TARGETS)
def test_fill_and_element_store_reach_the_same_next_verdict(target, cost):
    outcomes = []
    for program in (temporary_fill, temporary_store):
        try:
            allo.compile(program, target(), cost, backend="virtual")
        except Exception as error:  # compare the next supported-domain boundary
            outcomes.append(
                (
                    type(error),
                    re.sub(
                        r"%[A-Za-z0-9_]+(?:#\d+)?",
                        "%ssa",
                        str(error).replace(program.__name__, "program"),
                    ),
                )
            )
        else:
            outcomes.append(None)
    assert outcomes[0] == outcomes[1]


def test_output_argument_form_executes_on_functional_backend():
    compiled = allo.compile(
        output_argument, build_apu_v1_target(), apu_v1_cost, backend="functional"
    )
    # Small exact products keep this positive control independent of the
    # target's inexact-product rounding; no tolerance hides an output error.
    W = (np.arange(64).reshape(8, 8) % 5 - 2).astype(np.float16)
    x = (np.arange(8) % 3 - 1).astype(np.float16)
    y = np.full(8, 99, np.float16)
    compiled(W, x, y)
    expected = (W.astype(np.int32) @ x.astype(np.int32)).astype(np.float16)
    np.testing.assert_array_equal(y.view(np.uint16), expected.view(np.uint16))
