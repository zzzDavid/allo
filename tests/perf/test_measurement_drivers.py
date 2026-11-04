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


def test_apu_v1_paired_ratios_use_same_round_valid_launches_only():
    launches = [
        {"lane": "tenon", "kind": "warmup", "round": 0, "crun": 999, "outputs_match_oracle": True},
        {"lane": "tenon", "kind": "measured", "round": 1, "crun": 100, "outputs_match_oracle": True},
        {"lane": "control:paper", "kind": "measured", "round": 1, "crun": 90, "outputs_match_oracle": True},
        {"lane": "expert", "kind": "measured", "round": 1, "crun": 110, "outputs_match_oracle": True},
        {"lane": "tenon", "kind": "measured", "round": 2, "crun": 200, "outputs_match_oracle": True},
        {"lane": "control:paper", "kind": "measured", "round": 2, "crun": 100, "outputs_match_oracle": False},
        {"lane": "expert", "kind": "measured", "round": 2, "crun": 220, "error": "boom"},
        {"lane": "tenon", "kind": "measured", "round": 3, "crun": 0, "outputs_match_oracle": True},
        {"lane": "expert", "kind": "measured", "round": 3, "crun": 300, "outputs_match_oracle": True},
    ]
    assert paper_rows._paired_ratios(launches, "control:paper") == [0.9]
    assert paper_rows._paired_ratios(launches, "expert") == [1.1]
    assert paper_rows._control_lane_name("/x/word_count_narrow_counters_device.c") == (
        "control:word_count_narrow_counters_device"
    )


def test_ledag_drain_removes_the_tool_scratch_it_created(monkeypatch, tmp_path):
    import subprocess

    from allo.pim import apu_v1_program as program

    monkeypatch.setattr(program, "_LEDAG_SCRATCH", tmp_path)
    older = tmp_path / "online_log_2026_11_04_00_00_00"
    older.write_bytes(b"kept: existed before the read")
    unrelated = tmp_path / "notes.txt"
    unrelated.write_text("kept: not the tool's")

    def fake_run(argv, **kwargs):
        (tmp_path / "online_log_2026_11_04_04_23_40").write_bytes(b"x" * 64)
        (tmp_path / "ledag_ssh_nz264_1_log.txt").write_text("session log")
        return subprocess.CompletedProcess(argv, 0, stdout=b"total crun: 7 hits: 1\n")

    monkeypatch.setattr(program.subprocess, "run", fake_run)
    assert "crun: 7" in program._ledag_drain()
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "notes.txt",
        "online_log_2026_11_04_00_00_00",
    ]

    def failing_run(argv, **kwargs):
        (tmp_path / "online_log_2026_11_04_04_23_48").write_bytes(b"")
        return subprocess.CompletedProcess(argv, 255, stdout=b"", stderr=b"ENOSPC")

    monkeypatch.setattr(program.subprocess, "run", failing_run)
    with pytest.raises(RuntimeError, match="exited with 255"):
        program._ledag_drain()
    assert not (tmp_path / "online_log_2026_11_04_04_23_48").exists()
