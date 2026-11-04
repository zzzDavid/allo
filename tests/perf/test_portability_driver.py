# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The maintained portability driver (paper Sec. 5): its historical inputs,
its independent oracle, its execution-kind rule and its evidence retention,
without hardware."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.portability import gemv_source_order as driver


def test_historical_inputs_have_exact_products_and_the_cancellation_rows():
    A, B = driver.historical_inputs()
    assert A.shape == (driver.M, driver.K) and B.shape == (driver.K, driver.N)
    assert A.dtype == np.float16 and np.all(B == 1)
    assert driver.products_are_exact(A, B)
    for row, pattern in enumerate(driver.CANCELLATION_ROWS):
        assert list(A[row, : len(pattern)]) == pattern
        assert not np.any(A[row, len(pattern) :])
    # Deterministic: the seed reproduces the audit's operands.
    A2, _ = driver.historical_inputs()
    assert np.array_equal(A.view(np.uint16), A2.view(np.uint16))
    gaussian, Bg = driver.gaussian_inputs()
    assert not driver.products_are_exact(gaussian, Bg)


def test_oracle_rounds_each_step_and_exposes_the_regrouping():
    A, B = driver.historical_inputs()
    oracle = driver.source_order_fp16(A, B)
    # Row 0 is 2048 + 1 - 2048: the +1 is lost in FP16 at 2048 (ulp 2), so
    # source order gives 0; a regrouped (2048 - 2048) + 1 gives 1.
    assert oracle[0, 0] == np.float16(0)
    assert oracle[4, 0] == np.float16(0)  # 4096 + 1 + 1 - 4096 in order
    grouped = driver.grouped_numpy_fp16(A, B)
    assert np.count_nonzero(oracle.view(np.uint16) != grouped.view(np.uint16)) > 0
    # The oracle is a plain NumPy computation of the stated rule.
    acc = np.zeros((driver.M, 1), np.float16)
    for k in range(driver.K):
        acc = (acc + (A[:, k, None] * B[k][None, :]).astype(np.float16)).astype(np.float16)
    assert np.array_equal(acc.view(np.uint16), oracle.view(np.uint16))


class _Result:
    def __init__(self, kind, out_value, extra=None, cycles=5):
        self.kind, self.cycles, self.extra, self.backend = kind, cycles, extra or {}, "fake"
        self._out = out_value

    def fill(self, out):
        out[...] = self._out


def _fake_compiled(result, plan_name="fake_plan"):
    class Plan:
        name = plan_name
        metadata = {"k": 1}
        reduction_strategy = None

    class Artifact:
        layout = None
        layout_ctx = None
        cmds = ()

    class Schedule:
        module = "module { }"

    class Compiled:
        selected_plan = Plan()
        compiled = Artifact()
        schedule = Schedule()

        def __call__(self, A, B, out):
            result.fill(out)
            return result

    return Compiled()


def test_driver_fails_loudly_and_keeps_evidence(monkeypatch, tmp_path):
    A, B = driver.historical_inputs()
    oracle = driver.source_order_fp16(A, B)
    wrong = oracle.copy()
    wrong[7, 0] = np.float16(1.5)  # one wrong output
    calls = {}

    def fake_compile(workload, target, cost, **kwargs):
        calls.setdefault("n", 0)
        calls["n"] += 1
        # apu_v1 executes but one output is wrong; samsung ran virtually.
        if "samsung" in str(target).lower() or calls["n"] == 1:
            return _fake_compiled(_Result("virtual", oracle))
        return _fake_compiled(_Result("device", wrong, {"binary_sha256": "ab" * 32, "profile_text": "total crun: 5"}))

    monkeypatch.setattr(driver.allo, "compile", fake_compile)
    monkeypatch.setattr(driver, "_target", lambda name: (name, None))
    monkeypatch.setattr(driver, "_board_lock", lambda name: __import__("contextlib").nullcontext())
    monkeypatch.setattr(driver, "_samsung_identity", lambda: {"pim_driver_sha256": "cd" * 32})
    status = driver.main(["--out", str(tmp_path / "ev")])
    assert status == 1
    report = json.loads((tmp_path / "ev" / "report.json").read_text())
    assert report["status"] == "fail"
    problems = " ".join(report["problems"])
    assert "dry run" in problems and "1 of 4096 outputs differ" in problems
    labels = {r["label"]: r for r in report["runs"]}
    assert labels["samsung_strict"]["dry_run"] is True
    assert labels["apu_v1_strict"]["mismatching_outputs"] == 1
    assert labels["apu_v1_strict"]["binary_sha256"] == "ab" * 32
    # Evidence files: inputs/oracle, every output, the retained IR, profile text.
    ev = tmp_path / "ev"
    assert (ev / "inputs_and_oracle.npz").is_file()
    assert np.load(ev / "apu_v1_strict.output.npy")[7, 0] == np.float16(1.5)
    assert (ev / "apu_v1_strict.retained.mlir").read_text() == "module { }"
    assert (ev / "apu_v1_strict.profile_text.txt").read_text() == "total crun: 5"
    assert report["environment"]["evidence_class"] in ("checkout", "installed artifact")


def test_driver_records_a_failed_run_before_exiting(monkeypatch, tmp_path):
    class Boom:
        selected_plan = None
        compiled = type("A", (), {"layout": None, "layout_ctx": None, "cmds": ()})()
        schedule = type("S", (), {"module": "module { }"})()

        def __call__(self, A, B, out):
            raise RuntimeError("board unavailable")

    monkeypatch.setattr(driver.allo, "compile", lambda *a, **k: Boom())
    monkeypatch.setattr(driver, "_target", lambda name: (name, None))
    monkeypatch.setattr(driver, "_board_lock", lambda name: __import__("contextlib").nullcontext())
    assert driver.main(["--out", str(tmp_path / "ev"), "--targets", "apu_v1"]) == 1
    report = json.loads((tmp_path / "ev" / "report.json").read_text())
    assert report["runs"][0]["run_error"].startswith("RuntimeError: board unavailable")
    assert (tmp_path / "ev" / "apu_v1_strict.output.npy").is_file()
    assert report["problems"] == ["apu_v1_strict: did not execute"]


def test_failed_or_misplaced_control_fails_the_command(monkeypatch, tmp_path):
    """Reviewer round 35: a requested reassociation control that fails or
    runs elsewhere than the paper executed fails the command, while a control
    that executes may differ from the strict oracle."""
    A, B = driver.historical_inputs()
    oracle = driver.source_order_fp16(A, B)
    different = oracle.copy()
    different[3, 0] = np.float16(2.0)

    def scenario(control):
        calls = {"n": 0}

        def fake_compile(workload, target, cost, **kwargs):
            calls["n"] += 1
            if kwargs.get("reduction_order") == "reassociate":
                return control()
            return _fake_compiled(_Result("device", oracle))

        monkeypatch.setattr(driver.allo, "compile", fake_compile)
        monkeypatch.setattr(driver, "_target", lambda name: (name, None))
        monkeypatch.setattr(driver, "_board_lock", lambda name: __import__("contextlib").nullcontext())
        out = tmp_path / f"ev-{calls['n']}-{id(control)}"
        status = driver.main(["--out", str(out), "--targets", "apu_v1", "--reassociate"])
        return status, json.loads((out / "report.json").read_text())

    class Failing:
        selected_plan = None
        compiled = type("A", (), {"layout": None, "layout_ctx": None, "cmds": ()})()
        schedule = type("S", (), {"module": "m"})()

        def __call__(self, A, B, out):
            raise RuntimeError("control launch failed")

    status, report = scenario(lambda: Failing())
    assert status == 1 and report["problems"] == ["apu_v1_reassociate: did not execute"]
    assert report["runs"][1]["run_error"].startswith("RuntimeError: control launch failed")

    status, report = scenario(lambda: _fake_compiled(_Result("virtual", different)))
    assert status == 1 and "apu_v1_reassociate: executed as 'virtual'" in report["problems"][0]

    status, report = scenario(lambda: _fake_compiled(_Result("device", different)))
    assert status == 0 and report["problems"] == []
    assert report["runs"][1]["mismatching_outputs"] == 1  # allowed for the control


def test_plan_provenance_is_structural_not_truncated():
    """Reviewer round 35: APUV1Plan.metadata is a mappingproxy; it and the
    plan's manifest must be retained structurally (a real compiled plan,
    virtual backend, no hardware)."""
    import types

    import allo
    from allo.pim.costs import apu_v1_cost
    from allo.pim.targets import build_apu_v1_target

    proxy = types.MappingProxyType({"a": 1, "nested": {"b": (1, 2)}})
    assert driver._json_ready(proxy) == {"a": 1, "nested": {"b": [1, 2]}}
    with pytest.raises(driver.ProvenanceError, match="deeper"):
        deep = {}
        cur = deep
        for _ in range(driver._MAX_PROVENANCE_DEPTH + 2):
            cur["x"] = {}
            cur = cur["x"]
        driver._json_ready(deep)
    loop = {}
    loop["self"] = loop
    with pytest.raises(driver.ProvenanceError, match="cycle"):
        driver._json_ready(loop)
    # The real plan's effects manifest nests past 12 levels; it must come
    # through whole (a dry run of round 36 failed on the old 12-level cap).
    deep_but_fine = {}
    cur = deep_but_fine
    for _ in range(20):
        cur["x"] = {}
        cur = cur["x"]
    assert driver._json_ready(deep_but_fine) is not None

    compiled = allo.compile(driver.gemv_source(), build_apu_v1_target(), apu_v1_cost, backend="virtual")
    plan = compiled.selected_plan
    assert isinstance(plan.metadata, types.MappingProxyType)
    metadata = driver._json_ready(plan.metadata)
    assert isinstance(metadata, dict)
    for key in ("operation_counts", "transfer_facts", "accumulator_persistence", "accesses"):
        assert key in metadata and isinstance(metadata[key], (dict, list)), key
    manifest = driver._json_ready(plan.manifest())
    assert {"name", "iteration_layout", "value_layouts", "transfers", "reduction_strategy", "operations"} <= set(manifest)
    assert manifest["name"] == plan.name
    effects = (getattr(compiled.compiled.layout, "extra", None) or {}).get("effects")
    assert effects is not None, "the selected placement carries the shared candidate effects"
    effects_manifest = driver._json_ready(effects)
    assert isinstance(effects_manifest, dict) and effects_manifest
    text = json.dumps({"m": metadata, "p": manifest, "e": effects_manifest})
    assert "mappingproxy" not in text and "..." not in text[:0]
