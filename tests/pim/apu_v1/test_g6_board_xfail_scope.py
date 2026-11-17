# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""No board failure may be hidden by xfail after the V1-16 model repair.

Host-only subprocess injections prove launch errors and bit mismatches fail,
while model-equal readback passes. This retains the round-167 regression for
an unrestricted xfail that once absorbed a pre-execution scratch refusal.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
TARGET = "tests/pim/apu_v1/test_g6_board_semantics.py::test_strict_and_reassociative_gemv_bit_equal_to_models_on_board[full_group]"

PLUGIN = """
import numpy as np
import pytest
from types import SimpleNamespace
import tests.pim.conftest as conftest_module

MODE = {mode!r}


def pytest_configure(config):
    # no hardware: the device gate is bypassed for this injected run
    conftest_module.apu_v1_unavailable_reason = lambda: None


def pytest_runtest_setup(item):
    import importlib
    module = importlib.import_module(item.module.__name__)

    def injected(M, K, seed):
        rng = np.random.default_rng(seed)
        W = (rng.standard_normal((M, K)) * 0.1).astype(np.float16)
        x = (rng.standard_normal(K) * 0.1).astype(np.float16)
        exact = W.astype(np.float32) @ x.astype(np.float32)
        plan = SimpleNamespace(reduction_strategy=SimpleNamespace(group_size=4))
        if MODE == "launch_error":
            raise RuntimeError("ledag-ssh cannot read the device log: injected pre-execution refusal")
        source = module.G6._fp16_source_order(np.zeros(M), (module.G6.mul_f16(W[:, k], x[k]) for k in range(K)))
        tree = module.G6._adjacent_tree_fp16(module.G6.mul_f16(W, x), plan.reduction_strategy.group_size)
        if MODE == "mismatch":
            y_strict = source.copy(); y_strict.view(np.uint16)[::2] ^= 1
            y_relaxed = tree.copy(); y_relaxed.view(np.uint16)[1::2] ^= 1
        else:  # "equal": the device agrees with the models bit for bit
            y_strict, y_relaxed = source.copy(), tree.copy()
        return W, x, exact, y_strict, y_relaxed, plan

    module._strict_and_reassociative_runs = injected
"""


def _run_with_injection(tmp_path, mode):
    plugin_dir = tmp_path / f"plugin_{mode}"
    plugin_dir.mkdir()
    (plugin_dir / f"inject_{mode}.py").write_text(PLUGIN.format(mode=mode))
    root = HERE.parents[2]
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = f"{plugin_dir}:{root}" + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )
    env.pop("PYTHONSAFEPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-p",
            f"inject_{mode}",
            "--basetemp",
            str(tmp_path / f"bt_{mode}"),
            TARGET,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    return result.returncode, result.stdout + result.stderr


def test_a_launch_error_is_a_failure_not_an_expected_one(tmp_path):
    code, out = _run_with_injection(tmp_path, "launch_error")
    assert code != 0, out[-1500:]
    assert "1 failed" in out and "xfailed" not in out, out[-1500:]
    assert "injected pre-execution refusal" in out


def test_a_bit_model_mismatch_is_a_failure(tmp_path):
    code, out = _run_with_injection(tmp_path, "mismatch")
    assert code != 0, out[-1500:]
    assert "1 failed" in out and "xfailed" not in out, out[-1500:]
    assert "BitModelMismatch" in out


def test_model_equal_results_pass(tmp_path):
    code, out = _run_with_injection(tmp_path, "equal")
    assert code == 0, out[-1500:]
    assert "1 passed" in out, out[-1500:]
