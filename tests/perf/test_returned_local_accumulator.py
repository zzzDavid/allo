# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every explicit return needs a binding before matched work is enumerated."""
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


def argument_zero_stores(W: float16[8, 8], x: float16[8], y: float16[8]):
    for i in range(8):
        y[i] = 0.0
    for i in range(8):
        for k in allo.reduction(8):
            y[i] += W[i, k] * x[k]


def array_assignment_before(W: float16[8, 8], x: float16[8], y: float16[8]):
    y = 0.0
    for i in range(8):
        for k in allo.reduction(8):
            y[i] += W[i, k] * x[k]


def array_assignment_after(W: float16[8, 8], x: float16[8], y: float16[8]):
    for i in range(8):
        for k in allo.reduction(8):
            y[i] += W[i, k] * x[k]
    y = 0.0


def copy_values(src: int32[8], dst: int32[8]):
    dst = src


def unrelated_filled_return(
    W: float16[8, 8], x: float16[8], y: float16[8]
) -> float16[8]:
    z: float16[8] = 3.0
    for i in range(8):
        acc: float16 = 0.0
        for k in allo.reduction(8):
            acc += W[i, k] * x[k]
        y[i] = acc
    return z


def unrelated_unwritten_return(
    W: float16[8, 8], x: float16[8], y: float16[8]
) -> float16[8]:
    z: float16[8]
    for i in range(8):
        acc: float16 = 0.0
        for k in allo.reduction(8):
            acc += W[i, k] * x[k]
        y[i] = acc
    return z


def returned_output(W: float16[8, 8], x: float16[8], y: float16[8]) -> float16[8]:
    for i in range(8):
        acc: float16 = 0.0
        for k in allo.reduction(8):
            acc += W[i, k] * x[k]
        y[i] = acc
    return y


def returned_input(W: float16[8, 8], x: float16[8], y: float16[8]) -> float16[8]:
    for i in range(8):
        acc: float16 = 0.0
        for k in allo.reduction(8):
            acc += W[i, k] * x[k]
        y[i] = acc
    return x


def scalar_literal_return(W: float16[8, 8], x: float16[8], y: float16[8]) -> float16:
    for i in range(8):
        acc: float16 = 0.0
        for k in allo.reduction(8):
            acc += W[i, k] * x[k]
        y[i] = acc
    return 4.0


def scalar_loaded_return(W: float16[8, 8], x: float16[8], y: float16[8]) -> float16:
    for i in range(8):
        acc: float16 = 0.0
        for k in allo.reduction(8):
            acc += W[i, k] * x[k]
        y[i] = acc
    return x[0]


def local_fill_operand(W: float16[8, 8], y: float16[8]):
    t: float16[8] = 2.0
    for i in range(8):
        for k in allo.reduction(8):
            y[i] += W[i, k] * t[k]


def local_array_operand(a: float16[8], b: float16[8], W: float16[8, 8], y: float16[8]):
    t: float16[8] = a + b
    for i in range(8):
        for k in allo.reduction(8):
            y[i] += W[i, k] * t[k]


def local_library_operand(
    a: float16[8], b: float16[8], W: float16[8, 8], y: float16[8]
):
    t = allo.add(a, b)
    for i in range(8):
        for k in allo.reduction(8):
            y[i] += W[i, k] * t[k]


def local_elementwise_operand(a: float16[64], y: float16[64]):
    t: float16[64] = 2.0
    for i in range(64):
        y[i] = a[i] + t[i]


TARGETS = [
    (build_samsung_target, samsung_cost),
    (build_aim_target, aim_cost),
    (build_upmem_target, upmem_cost),
    (build_apu_v1_target, apu_v1_cost),
]
ABI_REASON = (
    "explicit return values are unsupported by the matched output "
    "ABI; use output arguments without a return statement"
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
        unrelated_filled_return,
        unrelated_unwritten_return,
        scalar_literal_return,
        scalar_loaded_return,
    ],
)
@pytest.mark.parametrize("target,cost", TARGETS)
def test_explicit_return_names_the_abi_boundary(workload, target, cost, monkeypatch):
    def no_candidates(*args, **kwargs):
        pytest.fail("output ABI refusal must precede candidate enumeration")

    monkeypatch.setattr("allo.spmw_codegen.autoschedule", no_candidates)
    with pytest.raises(UnsupportedWorkloadError) as caught:
        allo.compile(workload, target(), cost, backend="virtual")
    message = str(caught.value)
    assert message == ABI_REASON
    assert "use output arguments without a return statement" in message
    assert "never initialized" not in message
    assert "initialize the accumulator with literal 0" not in message


@pytest.mark.parametrize("workload", [returned_output, returned_input])
@pytest.mark.parametrize("target,cost", TARGETS)
def test_returned_argument_has_one_shared_abi_policy(
    workload, target, cost, monkeypatch
):
    monkeypatch.setattr(
        "allo.spmw_codegen.autoschedule",
        lambda *args, **kwargs: pytest.fail(
            "return ABI must be checked before enumeration"
        ),
    )
    with pytest.raises(UnsupportedWorkloadError) as caught:
        allo.compile(workload, target(), cost, backend="virtual")
    assert str(caught.value) == ABI_REASON


@pytest.mark.parametrize(
    "workload",
    [
        local_fill_operand,
        local_array_operand,
        local_library_operand,
        local_elementwise_operand,
    ],
)
@pytest.mark.parametrize("target,cost", TARGETS)
def test_unproduced_local_operand_is_refused_before_enumeration(
    workload, target, cost, monkeypatch
):
    monkeypatch.setattr(
        "allo.spmw_codegen.autoschedule",
        lambda *args, **kwargs: pytest.fail(
            "local operand must be checked before enumeration"
        ),
    )
    with pytest.raises(
        UnsupportedWorkloadError,
        match="local buffer not produced by a matched operation",
    ):
        allo.compile(workload, target(), cost, backend="virtual")


def test_late_direct_fill_is_not_a_zero_initializer():
    schedule = allo.customize(zero_filled)
    body = list(schedule.top_func.entry_block.operations)
    fill = next(op for op in body if op.operation.name == "linalg.fill")
    loop = next(op for op in body if op.operation.name == "affine.for")
    fill.operation.move_after(loop.operation)
    assert schedule.module.operation.verify()
    target = build_samsung_target()
    trace = match_workload(target, schedule.module)
    with pytest.raises(UnsupportedWorkloadError, match="matched output ABI"):
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


@pytest.mark.parametrize("workload", [output_argument, argument_zero_stores])
def test_output_argument_form_executes_on_functional_backend(workload):
    compiled = allo.compile(
        workload, build_apu_v1_target(), apu_v1_cost, backend="functional"
    )
    # Small exact products keep this positive control independent of the
    # target's inexact-product rounding; no tolerance hides an output error.
    W = (np.arange(64).reshape(8, 8) % 5 - 2).astype(np.float16)
    x = (np.arange(8) % 3 - 1).astype(np.float16)
    y = np.full(8, 99, np.float16)
    compiled(W, x, y)
    expected = (W.astype(np.int32) @ x.astype(np.int32)).astype(np.float16)
    np.testing.assert_array_equal(y.view(np.uint16), expected.view(np.uint16))


@pytest.mark.parametrize("target,cost", TARGETS)
@pytest.mark.parametrize("workload", [array_assignment_before, array_assignment_after])
def test_whole_array_assignment_cannot_be_dropped(workload, target, cost, monkeypatch):
    def no_candidates(*args, **kwargs):
        pytest.fail("unsupported whole-buffer copy must fail before enumeration")

    monkeypatch.setattr("allo.spmw_codegen.autoschedule", no_candidates)
    with pytest.raises(UnsupportedWorkloadError, match="whole-buffer memref.copy"):
        allo.compile(workload, target(), cost, backend="functional")


def test_record_copy_effects_are_retained_and_c11_refuses_before_emission(monkeypatch):
    from allo.backend import c as c_backend
    from allo.pim.apu_v1_record_frontend import normalize_apu_v1_program

    record = allo.APUv1Program((allo.APUv1Phase(copy_values),))
    source = normalize_apu_v1_program(record, build_apu_v1_target())
    (effect,) = source.facts.effects
    assert effect.store.startswith("memref.copy@")
    assert effect.whole and effect.writes_whole_buffer and effect.instances == 8
    assert effect.destination.parameter == "dst"
    assert [read.parameter for read in effect.reads] == ["src"]

    def no_emitter(*args, **kwargs):
        pytest.fail("C11 must refuse the unsupported copy before its fatal emitter")

    monkeypatch.setattr(c_backend.allo_d, "emit_vhls", no_emitter)
    with pytest.raises(c_backend.PortableCError, match="does not lower memref.copy"):
        allo.compile(record, build_apu_v1_target(), apu_v1_cost, backend="functional")
