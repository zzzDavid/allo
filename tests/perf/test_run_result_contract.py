# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""RunResult.kind contract and the CompiledCallable output contract."""

from types import SimpleNamespace

import numpy as np
import pytest

import allo
from allo.compiler import CompiledCallable
from allo.ir.types import int32, uint16
from allo.pim.costs import apu_v1_cost
from allo.pim.targets import build_apu_v1_target
from allo.spmw_codegen import RUN_KINDS, RunResult, SourceBackendBinding


def _workload(A, B, C):
    del A, B, C


class _Backend:
    def __init__(self, result, gather=("C",)):
        self.result = result
        self.host_moves = [
            SimpleNamespace(verb=SimpleNamespace(name="gather"), buffer_role=role)
            for role in gather
        ]

    def run(self, **_inputs):
        return self.result


def _callable(result, gather=("C",)):
    return CompiledCallable(
        _workload, SimpleNamespace(name="fake"), None, None, _Backend(result, gather)
    )


def _arrays():
    return np.ones(4, np.float32), np.ones(4, np.float32), np.full(4, 7, np.float32)


def test_run_kinds_are_the_documented_five():
    assert RUN_KINDS == {
        "device",
        "simulator",
        "simulator-timing-only",
        "functional",
        "virtual",
    }


def test_run_result_rejects_unknown_and_unspecified_kinds():
    with pytest.raises(ValueError, match="RunResult.kind must be one of"):
        RunResult(1, "", "fake", kind="hardware")
    unspecified = RunResult(1, "", "fake")
    assert unspecified.kind == "unspecified"
    with pytest.raises(RuntimeError, match="without a kind"):
        unspecified.validate()
    with pytest.raises(RuntimeError, match="without a kind"):
        _callable(unspecified)(*_arrays())


def test_non_executing_result_may_not_carry_outputs():
    result = RunResult(
        1, "", "fake", kind="virtual", extra={"outputs": {"C": np.zeros(4)}}
    )
    with pytest.raises(RuntimeError, match="carrying outputs"):
        result.validate()


@pytest.mark.parametrize("kind", ["virtual", "simulator-timing-only"])
def test_non_executing_runs_leave_arrays_untouched(kind):
    A, B, C = _arrays()
    result = _callable(RunResult(5, "", "fake", kind=kind))(A, B, C)
    assert result.kind == kind and not result.executed
    np.testing.assert_array_equal(C, np.full(4, 7, np.float32))


@pytest.mark.parametrize("kind", ["device", "simulator", "functional"])
def test_executed_run_without_gathered_output_fails(kind):
    A, B, C = _arrays()
    with pytest.raises(RuntimeError, match=r"produced no output for role\(s\) \['C'\]"):
        _callable(RunResult(5, "", "fake", kind=kind))(A, B, C)
    np.testing.assert_array_equal(C, np.full(4, 7, np.float32))


def test_executed_run_with_other_output_fails():
    A, B, C = _arrays()
    outputs = {"p": np.zeros(4), "q": np.zeros(4)}
    result = RunResult(5, "", "fake", kind="simulator", extra={"outputs": outputs})
    with pytest.raises(RuntimeError, match=r"no output for role\(s\) \['C'\]"):
        _callable(result)(A, B, C)


def test_executed_run_without_bound_output_fails():
    A, B, C = _arrays()
    compiled = _callable(RunResult(5, "", "fake", kind="simulator"), gather=())
    compiled._source_binding = SourceBackendBinding(inputs={"X": "A"}, output="C")
    with pytest.raises(RuntimeError, match=r"no output for role\(s\) \['C'\]"):
        compiled(A, B, C)


def test_executed_all_zero_output_is_copied():
    A, B, C = _arrays()
    result = RunResult(
        5, "", "fake", kind="simulator", extra={"outputs": {"y": np.zeros(4)}}
    )
    _callable(result)(A, B, C)
    np.testing.assert_array_equal(C, np.zeros(4, np.float32))


def _scan(values: int32[64], out: int32[64]):
    acc: int32 = 0
    for i in range(64):
        if values[i] > 3:
            acc += values[i]
        out[i] = acc


def _histogram(values: uint16[32768]) -> uint16[256]:
    counts: uint16[256] = 0
    for bin_index in range(256):
        for item in allo.reduction(32768):
            if values[item] == bin_index:
                counts[bin_index] += 1
    return counts


def _program(kernel, **phase):
    return allo.APUv1Program((allo.APUv1Phase(kernel, **phase),), name=kernel.__name__)


@pytest.fixture
def no_apu_v1_device(monkeypatch):
    monkeypatch.setenv("TENON_APU_V1_PCI_NODE", "/nonexistent/apu-pci-node")


def test_apu_v1_program_virtual_never_touches_device(no_apu_v1_device):
    from allo.spmw_codegen import SimulatorUnavailable

    target = build_apu_v1_target()
    values = np.arange(64, dtype=np.int32) % 7
    out = np.full(64, -1, np.int32)
    virtual = allo.compile(_program(_scan), target, apu_v1_cost, backend="virtual")
    result = virtual(values, out)
    assert result.kind == "virtual" and result.backend == "virtual"
    assert result.cycles == virtual.estimate().cycles > 0
    assert result.outputs == {}
    np.testing.assert_array_equal(out, np.full(64, -1, np.int32))

    device = allo.compile(_program(_scan), target, apu_v1_cost)
    with pytest.raises(SimulatorUnavailable, match="GSI device not present"):
        device(values, out)


def test_apu_v1_native_program_virtual_never_touches_device(no_apu_v1_device):
    program = _program(_histogram, result_names=("counts",), vectorize="required")
    target = build_apu_v1_target()
    virtual = allo.compile(program, target, apu_v1_cost, backend="virtual")
    result = virtual(np.zeros(32768, np.uint16))
    assert result.kind == "virtual"
    assert result.cycles == virtual.estimate().cycles > 0
    with pytest.raises(NotImplementedError, match="no host execution path"):
        allo.compile(program, target, apu_v1_cost, backend="functional")


def test_apu_v1_program_functional_executes_scalar_c_on_host(no_apu_v1_device):
    functional = allo.compile(
        _program(_scan), build_apu_v1_target(), apu_v1_cost, backend="functional"
    )
    values = np.arange(64, dtype=np.int32) % 7
    out = np.zeros(64, np.int32)
    result = functional(values, out)
    expected = np.cumsum(np.where(values > 3, values, 0)).astype(np.int32)
    assert result.kind == "functional" and result.cycles is None
    np.testing.assert_array_equal(result.outputs["out"], expected)
    np.testing.assert_array_equal(out, expected)


def test_apu_v1_program_virtual_requires_cost():
    with pytest.raises(ValueError, match="needs a cost spec"):
        allo.pim.apu_v1_program.compile_apu_v1_program(
            _program(_scan), build_apu_v1_target(), backend="virtual"
        )
