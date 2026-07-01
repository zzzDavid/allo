# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

import allo
from allo.spmw_codegen import RunResult


class _FakeCompiled:
    def __init__(self):
        self.execution_graph = object()
        self.host_moves = [
            SimpleNamespace(verb=SimpleNamespace(name="gather"), buffer_role="C")
        ]
        self.calls = []

    def run(self, **inputs):
        self.calls.append(inputs)
        return RunResult(
            cycles=17,
            stdout="fake",
            backend="virtual",
            extra={"outputs": {"out": np.asarray(inputs["A"]) + inputs["B"]}},
        )


def _workload(A, B, C):
    del A, B, C


def _patch_pipeline(monkeypatch):
    compiler_api = importlib.import_module("allo.compiler")
    fake = _FakeCompiled()
    captured = {}
    monkeypatch.setattr(
        compiler_api,
        "customize",
        lambda workload, enable_tensor=False: SimpleNamespace(module="module"),
    )
    monkeypatch.setattr(
        compiler_api,
        "match_workload",
        lambda target, module: SimpleNamespace(target_name=target.name),
    )

    def compile_fake(target, trace, **kwargs):
        captured.update(kwargs)
        return fake

    monkeypatch.setattr(compiler_api, "compile_for_target", compile_fake)
    return fake, captured


def test_allo_compile_returns_signature_compatible_numpy_callable(monkeypatch):
    fake, captured = _patch_pipeline(monkeypatch)
    target = SimpleNamespace(name="fake")

    module = allo.compile(_workload, target, backend="virtual")
    A = np.arange(4, dtype=np.float32)
    B = np.ones(4, dtype=np.float32)
    C = np.zeros(4, dtype=np.float32)
    result = module(A, B=B, C=C)

    assert callable(module)
    assert inspect.signature(module) == inspect.signature(_workload)
    assert result.cycles == 17
    assert module.last_result is result
    assert fake.calls[0] == {"A": A, "B": B, "C": C}
    np.testing.assert_array_equal(C, A + B)
    assert captured["backend"] == "virtual"
    assert captured["cost"] is None


def test_compile_rejects_missing_numpy_operand_before_backend(monkeypatch):
    _patch_pipeline(monkeypatch)
    target = SimpleNamespace(name="fake")
    module = allo.compile(_workload, target)

    with pytest.raises(TypeError, match="missing a required argument: 'C'"):
        module(np.ones(2), np.ones(2))


def test_compile_binds_executable_cost_spec(monkeypatch):
    _fake, captured = _patch_pipeline(monkeypatch)
    from allo.pim.costs import samsung_cost
    from allo.pim.targets import build_samsung_target

    target = build_samsung_target()
    module = allo.compile(_workload, target, cost=samsung_cost)

    assert module.cost.spec is samsung_cost
    assert module.cost.target is target
    assert captured["cost"] is module.cost


def test_run_backend_accepts_lowered_roles_outside_workload_signature(monkeypatch):
    fake, _captured = _patch_pipeline(monkeypatch)
    target = SimpleNamespace(name="fake")
    module = allo.compile(_workload, target)
    A = np.ones(4, dtype=np.float32)
    B = np.ones(4, dtype=np.float32)

    result = module.run_backend(A=A, B=B)

    assert result.cycles == 17
    assert fake.calls[-1] == {"A": A, "B": B}
