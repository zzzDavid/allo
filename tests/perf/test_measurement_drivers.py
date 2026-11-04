# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The paper-measurement drivers report failure through their exit status
(release gate G10): a missing asset, a failed launch, a wrong output or a
violated experiment invariant exits nonzero while the record is preserved;
a clean run exits zero."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from benchmarks.apu_g2 import paper_rows as g2_paper_rows
from benchmarks.apu_g2 import selection_traces
from benchmarks.apu_v1 import paper_rows


def test_apu_v1_missing_assets_exit_nonzero(tmp_path):
    out = tmp_path / "rows.json"
    status = paper_rows.main(
        [
            "--artifacts",
            str(tmp_path / "nonexistent"),
            "--cases",
            "matrix_multiply",
            "--out",
            str(out),
        ]
    )
    assert status == 1
    session = json.loads(out.read_text())
    assert session["status"].startswith("fail")
    (record,) = session["records"]
    assert "FileNotFoundError" in record["error"]


def test_apu_v1_virtual_pass_exits_zero(tmp_path):
    root = Path(os.environ.get("TENON_ARTIFACTS", ""))
    if not (root / "apuv1" / "tenon" / "word_count" / "workload.py").is_file():
        pytest.skip("BLOCKED-ENV(tenon artifacts not found; set TENON_ARTIFACTS)")
    out = tmp_path / "rows.json"
    status = paper_rows.main(
        [
            "--artifacts",
            str(root),
            "--backend",
            "virtual",
            "--expert",
            "--cases",
            "word_count",
            "--out",
            str(out),
        ]
    )
    assert status == 0
    session = json.loads(out.read_text())
    (record,) = session["records"]
    assert record["lanes"]["tenon"]["native_vector_route"]
    assert record["lanes"]["expert"]["binary_matches_archive"] is True
    assert record["problems"] == []


def test_apu_v1_statistics_and_invalid_launches_are_separated():
    stats = paper_rows._statistics([3, 1, 2, 10])
    assert stats["median"] == 2.5 and stats["min"] == 1 and stats["max"] == 10
    assert stats["q1"] <= stats["median"] <= stats["q3"]
    assert paper_rows._statistics([]) == {}


def test_apu_g2_failed_invariant_exits_nonzero(monkeypatch, tmp_path):
    def perturbation(seed):
        return {
            "experiment": "cost_perturbation",
            "runs": {},
            "invariant_problems": ["the perturbation did not change the selected plan"],
        }

    def forced(seed):
        raise RuntimeError("card unavailable")

    monkeypatch.setattr(selection_traces, "perturbation", perturbation)
    monkeypatch.setattr(selection_traces, "forced_alternative", forced)
    out = tmp_path / "traces.json"
    assert selection_traces.main(["--out", str(out), "--seed", "1"]) == 1
    session = json.loads(out.read_text())
    assert session["status"] == "fail (2 experiment(s))"
    assert "RuntimeError" in session["experiments"][1]["error"]


def test_apu_g2_clean_run_exits_zero(monkeypatch, tmp_path):
    def ok(seed):
        return {"experiment": "x", "runs": {}, "invariant_problems": []}

    monkeypatch.setattr(selection_traces, "perturbation", ok)
    monkeypatch.setattr(selection_traces, "forced_alternative", ok)
    out = tmp_path / "traces.json"
    assert selection_traces.main(["--out", str(out), "--seed", "1"]) == 0
    assert json.loads(out.read_text())["status"] == "pass"


def test_apu_g2_amortized_ticks_keep_fractions():
    class Result:
        extra = {
            "launches": [
                {"raw_ticks": {"pipeline": 34245}},
                {"raw_ticks": {"pipeline": 20}},
            ]
        }

    assert selection_traces.amortized_ticks(Result(), 20) == 34245 / 20 + 1.0

    class Missing:
        extra = {"launches": [{"raw_ticks": {}}]}

    assert selection_traces.amortized_ticks(Missing(), 20) is None


def test_apu_g2_vendor_lane_checks_the_loaded_device_library(tmp_path):
    import hashlib

    archived = tmp_path / "device.update.bin"
    archived.write_bytes(b"archived library")
    expected = hashlib.sha256(archived.read_bytes()).hexdigest()
    same = tmp_path / "compiled-in.bin"
    same.write_bytes(b"archived library")
    other = tmp_path / "other.bin"
    other.write_bytes(b"another library")

    ok = g2_paper_rows.loaded_device_library({"device_library": str(same)}, expected)
    assert ok["problem"] is None and ok["loaded_device_library_sha256"] == expected
    wrong = g2_paper_rows.loaded_device_library({"device_library": str(other)}, expected)
    assert "other than the archived one" in wrong["problem"]
    missing = g2_paper_rows.loaded_device_library(
        {"device_library": str(tmp_path / "gone.bin")}, expected
    )
    assert "not a file" in missing["problem"]
    silent = g2_paper_rows.loaded_device_library({}, expected)
    assert "did not report" in silent["problem"]
