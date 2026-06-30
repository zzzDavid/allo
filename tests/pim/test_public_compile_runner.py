# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The PIM suite delegates compilation to the public ``allo.compile`` API."""

from types import SimpleNamespace

import numpy as np

from lib import runner
from allo.pim.performance import SAMSUNG_BASE_PROFILE
from allo.spmw_codegen import RunResult


def _workload(A, B, C):
    del A, B, C


def test_suite_runner_binds_profile_and_invokes_backend_roles(monkeypatch):
    captured = {}

    class FakeCallable:
        def run_backend(self, **inputs):
            captured["inputs"] = inputs
            return RunResult(23, "ok", "samsung_hbm_pim")

    def fake_compile(workload, target, cost, **kwargs):
        captured.update(
            workload=workload,
            target=target,
            cost=cost,
            compile_kwargs=kwargs,
        )
        return FakeCallable()

    monkeypatch.setattr(runner.allo, "compile", fake_compile)
    target = SimpleNamespace(name="samsung_hbm_pim")
    A = np.ones((2, 2), dtype=np.float16)
    B = np.ones((2, 2), dtype=np.float16)

    result = runner.compile_and_run(
        target,
        _workload,
        backend="virtual",
        host_moves=["move"],
        A=A,
        B=B,
    )

    assert result.cycles == 23
    assert captured["workload"] is _workload
    assert captured["target"] is target
    assert captured["cost"] is SAMSUNG_BASE_PROFILE
    assert captured["compile_kwargs"] == {
        "backend": "virtual",
        "host_moves": ["move"],
    }
    assert captured["inputs"] == {"A": A, "B": B}


def test_unported_target_uses_same_public_interface_with_no_profile(monkeypatch):
    captured = {}

    class FakeCallable:
        def run_backend(self, **inputs):
            del inputs
            return RunResult(None, "unavailable", "aim")

    def fake_compile(workload, target, cost, **kwargs):
        del workload, target, kwargs
        captured["cost"] = cost
        return FakeCallable()

    monkeypatch.setattr(runner.allo, "compile", fake_compile)

    result = runner.compile_and_run(
        SimpleNamespace(name="aim"), _workload, backend=None
    )

    assert result.backend == "aim"
    assert captured["cost"] is None
