# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The paper-measurement drivers report failure through their exit status
(release gate G10): a missing asset, a failed launch, a wrong output or a
violated experiment invariant exits nonzero while the record is preserved;
a clean run exits zero."""

from __future__ import annotations

import hashlib
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
    wrong = g2_paper_rows.loaded_device_library(
        {"device_library": str(other)}, expected
    )
    assert "other than the archived one" in wrong["problem"]
    missing = g2_paper_rows.loaded_device_library(
        {"device_library": str(tmp_path / "gone.bin")}, expected
    )
    assert "not a file" in missing["problem"]
    silent = g2_paper_rows.loaded_device_library({}, expected)
    assert "did not report" in silent["problem"]


def test_apu_v1_paired_ratios_use_same_round_valid_launches_only():
    launches = [
        {
            "lane": "tenon",
            "kind": "warmup",
            "round": 0,
            "crun": 999,
            "outputs_match_oracle": True,
        },
        {
            "lane": "tenon",
            "kind": "measured",
            "round": 1,
            "crun": 100,
            "outputs_match_oracle": True,
        },
        {
            "lane": "control:paper",
            "kind": "measured",
            "round": 1,
            "crun": 90,
            "outputs_match_oracle": True,
        },
        {
            "lane": "expert",
            "kind": "measured",
            "round": 1,
            "crun": 110,
            "outputs_match_oracle": True,
        },
        {
            "lane": "tenon",
            "kind": "measured",
            "round": 2,
            "crun": 200,
            "outputs_match_oracle": True,
        },
        {
            "lane": "control:paper",
            "kind": "measured",
            "round": 2,
            "crun": 100,
            "outputs_match_oracle": False,
        },
        {
            "lane": "expert",
            "kind": "measured",
            "round": 2,
            "crun": 220,
            "error": "boom",
        },
        {
            "lane": "tenon",
            "kind": "measured",
            "round": 3,
            "crun": 0,
            "outputs_match_oracle": True,
        },
        {
            "lane": "expert",
            "kind": "measured",
            "round": 3,
            "crun": 300,
            "outputs_match_oracle": True,
        },
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


def _usage(free):
    import collections

    usage = collections.namedtuple("usage", "total used free")
    return usage(10**12, 10**12 - free, free)


def test_ledag_drain_refuses_without_scratch_headroom(monkeypatch, tmp_path):
    """X-24 (rounds 130-131): with TMPDIR unset the bootloader extracts into the
    log directory too, so the combined headroom is required there; 9 MiB
    free passes neither; the refusal names the directory, the free space and
    the requirement, and no session is started."""
    import subprocess

    from allo.pim import apu_v1_program as program

    logs = tmp_path / "shared-tmp"
    logs.mkdir()
    monkeypatch.setattr(program, "_LEDAG_SCRATCH", logs)
    monkeypatch.delenv("TMPDIR", raising=False)
    monkeypatch.setattr(program.tempfile, "gettempdir", lambda: str(logs))
    monkeypatch.setattr(program, "_same_filesystem", lambda a, b: True)
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(kwargs.get("env", {}))
        return subprocess.CompletedProcess(argv, 0, stdout=b"total crun: 3 hits: 1\n")

    monkeypatch.setattr(program.subprocess, "run", fake_run)
    monkeypatch.setattr(
        program.shutil, "disk_usage", lambda path: _usage(9 * 1024 * 1024)
    )
    with pytest.raises(RuntimeError, match="9.0 MiB free") as excinfo:
        program._ledag_drain()
    message = str(excinfo.value)
    assert str(logs) in message and "24.0 MiB" in message and "unset" in message
    assert calls == []
    # with the combined headroom the session runs and the child extracts there
    monkeypatch.setattr(program.shutil, "disk_usage", lambda path: _usage(10**9))
    assert program.ledag_scratch_headroom_problem() is None
    assert "crun: 3" in program._ledag_drain()
    assert calls[0]["TMPDIR"] == str(logs)


def test_ledag_drain_with_a_separately_provisioned_extraction_directory(
    monkeypatch, tmp_path
):
    """TMPDIR set to another filesystem: the log directory needs 8 MiB, the
    extraction directory 16 MiB, each checked on its own; the child receives
    that TMPDIR; a short extraction directory is refused by name."""
    import subprocess

    from allo.pim import apu_v1_program as program

    logs = tmp_path / "shared-tmp"
    extraction = tmp_path / "work-scratch"
    logs.mkdir()
    extraction.mkdir()
    monkeypatch.setattr(program, "_LEDAG_SCRATCH", logs)
    monkeypatch.setenv("TMPDIR", str(extraction))
    monkeypatch.setattr(program, "_same_filesystem", lambda a, b: False)
    free = {str(logs): 9 * 1024 * 1024, str(extraction): 10**9}
    monkeypatch.setattr(
        program.shutil, "disk_usage", lambda path: _usage(free[str(path)])
    )
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(kwargs.get("env", {}))
        return subprocess.CompletedProcess(argv, 0, stdout=b"total crun: 5 hits: 1\n")

    monkeypatch.setattr(program.subprocess, "run", fake_run)
    assert program.ledag_scratch_headroom_problem() is None
    assert "crun: 5" in program._ledag_drain()
    assert calls[0]["TMPDIR"] == str(extraction)
    free[str(extraction)] = 12 * 1024 * 1024  # below the 16 MiB the extraction needs
    with pytest.raises(RuntimeError, match="bundle extraction") as excinfo:
        program._ledag_drain()
    assert str(extraction) in str(excinfo.value) and "16.0 MiB" in str(excinfo.value)
    assert len(calls) == 1


def test_apu_v1_archived_lane_never_writes_into_the_archive(monkeypatch, tmp_path):
    import numpy as np

    from allo.pim import apu_v1_program as runtime

    archive = tmp_path / "archive" / "expert" / "case"
    (archive / "io").mkdir(parents=True)
    original = archive / "io" / "counts.uint16.bin"
    original.write_bytes(b"original archive evidence")
    before = original.read_bytes()
    scratch = tmp_path / "scratch"

    lane = object.__new__(paper_rows.ExpertLane)
    lane.case, lane.name, lane.dir, lane.count = "case", "expert", archive, 0
    lane.scratch = scratch
    lane.binary = archive / "binary"
    lane.binary_sha256 = "0" * 64
    lane.inputs = []
    lane.outputs = [("counts", original, "uint16", (4,))]
    lane.oracle = np.arange(4, dtype=np.uint16)

    def successful(argv, cwd, **kwargs):
        out = Path(argv[-1])
        assert archive not in out.resolve().parents and out.parent == Path(cwd)
        lane.oracle.astype("<u2").tofile(out)
        return "log", "total crun: 5 hits: 1", [5]

    monkeypatch.setattr(runtime, "launch_on_board", successful)
    launch = lane.launch()
    assert launch["crun"] == 5 and launch["outputs_match_oracle"]
    assert original.read_bytes() == before
    # Round 137 (V1-21 review): the successful launch's actual outputs,
    # device log, profiler text, argv and identities are kept, not removed.
    first = scratch / "expert-launch-001"
    assert launch["launch_dir"] == str(first)
    assert (first / "counts.uint16.bin").read_bytes() == lane.oracle.astype(
        "<u2"
    ).tobytes()
    assert (first / "profile.txt").read_text() == "total crun: 5 hits: 1"
    assert (first / "device.log").read_text() == "log"
    identity = json.loads((first / "launch.json").read_text())
    assert identity["binary_sha256"] == "0" * 64 and identity["crun"] == 5
    assert (
        identity["outputs_sha256"]["counts"]
        == hashlib.sha256(lane.oracle.astype("<u2").tobytes()).hexdigest()
    )
    assert json.loads((first / "argv.json").read_text())[0].endswith("binary")

    def failing(argv, cwd, **kwargs):
        Path(argv[-1]).write_bytes(b"partial")
        raise runtime.APUV1LaunchFailure(
            "simulated failed launch",
            {
                "argv": list(argv),
                "device_log_before": "old",
                "device_log_reads_after": ["old"],
                "attempts": [{"returncode": 0, "process_output": "ran"}],
                "fresh_records_found": 0,
                "expected_records": 1,
            },
        )

    monkeypatch.setattr(runtime, "launch_on_board", failing)
    with pytest.raises(RuntimeError, match="simulated failed launch"):
        lane.launch()
    assert original.read_bytes() == before
    kept = scratch / "expert-launch-002"
    assert (kept / "error.txt").read_text().startswith("APUV1LaunchFailure")
    assert (kept / "counts.uint16.bin").read_bytes() == b"partial"
    # the runtime's structured diagnostics are serialized beside the launch
    diagnostics = json.loads((kept / "diagnostics.json").read_text())
    assert (
        diagnostics["fresh_records_found"] == 0 and diagnostics["expected_records"] == 1
    )
    assert (kept / "device_log_before.txt").read_text() == "old"
    assert (kept / "device_log_after_1.txt").read_text() == "old"
    assert (kept / "process_output_1.txt").read_text() == "ran"


def test_apu_v1_driver_rejects_empty_cases_and_nonpositive_counts(tmp_path):
    out = tmp_path / "rows.json"
    with pytest.raises(SystemExit):
        paper_rows.main(["--cases", "--out", str(out)])
    with pytest.raises(SystemExit):
        paper_rows.main(["--measured", "0", "--out", str(out)])
    with pytest.raises(SystemExit):
        paper_rows.main(["--cases", "not_a_case", "--out", str(out)])


def test_apu_g2_timing_evidence_requires_complete_positive_ticks():
    problem = g2_paper_rows.timing_evidence_problem
    assert problem([], 1) == "0 launch record(s) for a plan of 1 launch(es)"
    assert problem([34245], 1) is None
    assert problem([34245, None], 2) == "launch 1: no raw device ticks"
    assert "not a positive finite count" in problem([0], 1)
    assert "not a positive finite count" in problem([float("nan")], 1)
    assert problem([1, 2], 1).startswith("2 launch record(s)")
    assert problem([1], 0) == "the selected plan declares no launches"

    class Empty:
        extra = {"launches": []}

    assert selection_traces.amortized_ticks(Empty(), 20) is None

    class Zero:
        extra = {"launches": [{"raw_ticks": {"pipeline": 0}}]}

    assert selection_traces.amortized_ticks(Zero(), 20) is None
    assert selection_traces._timing_problems(
        {
            "stock": {
                "amortized_ticks_per_pipeline": None,
                "launches": [],
                "expected_launches": 2,
            }
        }
    ) == ["stock: no complete device tick evidence (0 launch record(s) for 2 expected)"]


def test_apu_g2_driver_rejects_empty_cases_and_nonpositive_rounds(tmp_path):
    out = tmp_path / "rows.json"
    with pytest.raises(SystemExit):
        g2_paper_rows.main(["--cases", "--out", str(out), "--no-vendor"])
    with pytest.raises(SystemExit):
        g2_paper_rows.main(["--rounds", "0", "--out", str(out), "--no-vendor"])


class _Launch:
    def __init__(self, fingerprint):
        self.layout_fingerprint = fingerprint


class _Plan:
    def __init__(self, name, fingerprints):
        self.name = name
        self.estimate_cycles = 100
        self.launches = tuple(_Launch(f) for f in fingerprints)


def test_apu_g2_selection_traces_require_the_plans_launches():
    problem = selection_traces.launch_evidence_problem
    plan = _Plan("single/k2", ("aaaa" * 10, "bbbb" * 10))
    good = [
        {"index": 0, "layout_fingerprint": "aaaa" * 10, "raw_ticks": {"pipeline": 100}},
        {"index": 1, "layout_fingerprint": "bbbb" * 10, "raw_ticks": {"pipeline": 90}},
    ]
    assert problem(good, plan) is None
    assert problem(good[:1], plan) == "1 launch record(s) for a plan of 2 launch(es)"
    assert problem(good + good[:1], plan).startswith("3 launch record(s)")
    assert "index 1" in problem([good[1], good[0]], plan)
    swapped = [dict(good[0], layout_fingerprint="bbbb" * 10), good[1]]
    assert "is not the plan's" in problem(swapped, plan)
    assert "not a positive finite count" in problem(
        [good[0], dict(good[1], raw_ticks={"pipeline": 0})], plan
    )
    assert problem(good, _Plan("empty", ())) == "the selected plan declares no launches"

    class Compiled:
        plan = _Plan("single/k2", ("aaaa" * 10, "bbbb" * 10))

        def __call__(self, **args):
            return type(
                "R",
                (),
                {"kind": "device", "cycles": 100, "extra": {"launches": good[:1]}},
            )()

    run = selection_traces._run(Compiled(), {})
    assert run["amortized_ticks_per_pipeline"] is None
    assert run["expected_launches"] == 2
    assert selection_traces._timing_problems({"stock": run}) == [
        "stock: 1 launch record(s) for a plan of 2 launch(es) (1 launch record(s) for 2 expected)"
    ]


def test_apu_g2_selection_traces_command_rejects_a_missing_stock_launch(
    monkeypatch, tmp_path
):
    """Exact outputs with one of the stock plan's two launches missing: the
    command exits 1 and keeps the invalid evidence."""
    import numpy as np

    calls = {"n": 0}

    class FakeCompiled:
        def __init__(self, label, parameters, outputs, drop_last):
            self.plan = _Plan(label, ("f1" * 20, "f2" * 20))
            self._outputs = outputs
            self._drop = drop_last
            self.repetitions = None

        def __call__(self, **args):
            # exact outputs: fill the destinations from the arguments
            self._outputs(args)
            launches = [
                {
                    "index": i,
                    "layout_fingerprint": f,
                    "raw_ticks": {"pipeline": 100 + i},
                }
                for i, f in enumerate(("f1" * 20, "f2" * 20))
            ]
            if self._drop:
                launches = launches[:1]
            return type(
                "R",
                (),
                {"kind": "device", "cycles": 10, "extra": {"launches": launches}},
            )()

    def fake_compile(workload, target, cost, layout=None):
        calls["n"] += 1
        n = calls["n"]
        if n in (1, 2):  # perturbation: stock (drops a launch), perturbed

            def gemm_outputs(args):
                args["C"][...] = (
                    args["A"].astype(np.int64) @ args["B"].astype(np.int64)
                ).astype(np.int16)

            return FakeCompiled(
                "single/k2" if n == 1 else "single/k1", None, gemm_outputs, n == 1
            )

        def bicg_outputs(args):
            args["q"][...] = (args["A"].astype(np.int64) @ args["p"]).astype(np.int16)
            args["s"][...] = (args["r"].astype(np.int64) @ args["A"]).astype(np.int16)

        return FakeCompiled(
            "fused/k1" if n == 3 else "separate/k1", None, bicg_outputs, False
        )

    monkeypatch.setattr(selection_traces.allo, "compile", fake_compile)
    out = tmp_path / "traces.json"
    assert selection_traces.main(["--out", str(out), "--seed", "3"]) == 1
    session = json.loads(out.read_text())
    stock = session["experiments"][0]["runs"]["stock"]
    assert stock["outputs_exact"] is True
    assert (
        stock["timing_evidence_problem"]
        == "1 launch record(s) for a plan of 2 launch(es)"
    )
    assert stock["amortized_ticks_per_pipeline"] is None
    assert len(stock["launches"]) == 1  # the invalid evidence is retained
    assert any(
        "stock: 1 launch record(s)" in p
        for p in session["experiments"][0]["invariant_problems"]
    )


def test_apu_v1_archived_lane_accepts_a_relative_scratch_directory(
    monkeypatch, tmp_path
):
    import subprocess
    import sys

    import numpy as np

    from allo.pim import apu_v1_program as runtime

    monkeypatch.chdir(tmp_path)
    archive = tmp_path / "archive"
    archive.mkdir()
    original = archive / "counts.bin"
    original.write_bytes(b"original archive evidence")
    before = original.read_bytes()
    lane = object.__new__(paper_rows.ExpertLane)
    lane.case, lane.name, lane.dir, lane.count = "case", "expert", Path("archive"), 0
    lane.scratch = Path("relative_scratch")
    lane.binary = archive / "binary"
    lane.binary_sha256 = "0" * 64
    lane.inputs = []
    lane.outputs = [("counts", Path("archive/counts.bin"), "uint16", (4,))]
    lane.oracle = np.arange(4, dtype=np.uint16)

    def child_writes(argv, cwd, **kwargs):
        # The archived binary runs with the launch directory as cwd and the
        # output argument as given: it must be absolute.
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, numpy as np; np.arange(4, dtype='<u2').tofile(sys.argv[1])",
                argv[-1],
            ],
            cwd=cwd,
            check=True,
            capture_output=True,
        )
        return "", "total crun: 5 hits: 1", [5]

    monkeypatch.setattr(runtime, "launch_on_board", child_writes)
    assert lane.launch()["outputs_match_oracle"]
    assert original.read_bytes() == before
    assert (
        tmp_path / "relative_scratch" / "expert-launch-001" / "launch.json"
    ).is_file()

    def child_fails(argv, cwd, **kwargs):
        Path(argv[-1]).write_bytes(b"partial")
        raise RuntimeError("simulated failed launch")

    monkeypatch.setattr(runtime, "launch_on_board", child_fails)
    with pytest.raises(RuntimeError):
        lane.launch()
    assert original.read_bytes() == before
    kept = tmp_path / "relative_scratch" / "expert-launch-002"
    assert kept.is_absolute() and (kept / "error.txt").exists()
    assert (kept / "counts.bin").read_bytes() == b"partial"


def test_apu_g2_launch_evidence_requires_index_and_declared_fingerprint():
    problem = selection_traces.launch_evidence_problem
    plan = _Plan("two", ("expected-a", "expected-b"))
    ticks = [{"raw_ticks": {"pipeline": 100}}, {"raw_ticks": {"pipeline": 90}}]
    assert problem(ticks, plan) == "launch record 0 carries no launch index"
    indexed = [dict(r, index=i) for i, r in enumerate(ticks)]
    assert problem(indexed, plan) == "launch 0: no layout fingerprint recorded"
    assert problem([dict(indexed[0], layout_fingerprint=None), indexed[1]], plan) == (
        "launch 0: no layout fingerprint recorded"
    )
    swapped = [
        dict(indexed[0], layout_fingerprint="expected-b"),
        dict(indexed[1], layout_fingerprint="expected-a"),
    ]
    assert "is not the plan's" in problem(swapped, plan)
    complete = [
        dict(indexed[0], layout_fingerprint="expected-a"),
        dict(indexed[1], layout_fingerprint="expected-b"),
    ]
    assert problem(complete, plan) is None
    # A plan that declares no fingerprint only needs the index and ticks.
    assert problem(indexed, _Plan("two", (None, None))) is None


def test_apu_g2_paper_rows_lane_applies_the_plan_correspondence(monkeypatch):
    import numpy as np

    lane = object.__new__(g2_paper_rows.TenonLane)
    lane.name = "gemm"
    lane.fixture = type("F", (), {"oracle": np.zeros((1,), dtype=np.int64)})()
    lane.arguments = {"C": np.zeros((1,), dtype=np.int16)}
    monkeypatch.setitem(g2_paper_rows.ORACLE_OUTPUTS, "gemm", ("C",))
    records = [{"raw_ticks": {"pipeline": 100}}, {"raw_ticks": {"pipeline": 90}}]
    lane.compiled = type(
        "Compiled",
        (),
        {
            "plan": _Plan("two", ("expected-a", "expected-b")),
            "__call__": lambda self, **a: type(
                "R",
                (),
                {"kind": "device", "cycles": 10, "extra": {"launches": records}},
            )(),
        },
    )()
    launch = lane.launch()
    assert launch["amortized_ticks_per_pipeline"] is None
    assert (
        launch["timing_evidence_problem"] == "launch record 0 carries no launch index"
    )


def test_apu_v1_cases_of_one_session_take_separate_scratch_directories(tmp_path):
    """Round 85 (X-38): a session run with ``--scratch`` over several cases
    gave every case's lanes the same launch directory names, so the second
    case's expert and archived launches failed with ``FileExistsError``;
    each case now takes its own subdirectory of the supplied scratch."""
    root = Path(os.environ.get("TENON_ARTIFACTS", ""))
    if not (root / "apuv1" / "tenon" / "word_count" / "workload.py").is_file():
        pytest.skip("BLOCKED-ENV(tenon artifacts not found; set TENON_ARTIFACTS)")
    out = tmp_path / "rows.json"
    shared = tmp_path / "shared"
    status = paper_rows.main(
        [
            "--artifacts",
            str(root),
            "--backend",
            "virtual",
            "--expert",
            "--measured",
            "2",
            "--cases",
            "word_count",
            "kmeans",
            "--scratch",
            str(shared),
            "--out",
            str(out),
        ]
    )
    assert status == 0
    session = json.loads(out.read_text())
    records = {record["case"]: record for record in session["records"]}
    assert set(records) == {"word_count", "kmeans"}
    scratches = {case: Path(record["scratch"]) for case, record in records.items()}
    assert scratches["word_count"] != scratches["kmeans"]
    assert {path.parent for path in scratches.values()} == {shared.resolve()}
    assert {path.name for path in scratches.values()} == {"word_count", "kmeans"}
    for record in records.values():
        assert record["problems"] == []


def test_apu_v1_failed_baseline_launches_of_two_cases_do_not_collide(
    monkeypatch, tmp_path
):
    """Round 86 (X-38, the reviewer's failure-path probe): the archived and
    expert lanes of two cases launch for real up to the board call, which is
    made to fail so every launch directory is retained; the current driver
    makes all eight attempts with no collision, where the parent expression
    met four ``FileExistsError`` after the first case's four directories."""
    import contextlib

    import allo.pim.apu_v1_program as runtime

    root = Path(os.environ.get("TENON_ARTIFACTS", ""))
    if not (root / "apuv1" / "tenon" / "word_count" / "workload.py").is_file():
        pytest.skip("BLOCKED-ENV(tenon artifacts not found; set TENON_ARTIFACTS)")

    class FakeTenon:
        def __init__(self, *args, **kwargs):
            pass

        def describe(self):
            return {}

        def launch(self):
            return {"crun": 100, "outputs_match_oracle": True}

    attempts = []

    def no_hardware(argv, cwd, **kwargs):
        attempts.append(Path(cwd))
        raise RuntimeError("deliberate read failure; no device invoked")

    monkeypatch.setattr(paper_rows, "TenonLane", FakeTenon)
    monkeypatch.setattr(
        runtime, "apu_v1_board_lock", lambda: contextlib.nullcontext("no-device")
    )
    monkeypatch.setattr(runtime, "launch_on_board", no_hardware)
    shared = tmp_path / "shared"
    records = [
        paper_rows.measure_case(
            case,
            root.resolve(),
            backend="device",
            measured=1,
            expert=True,
            layout=None,
            device_source=None,
            label=None,
            historical_binary=True,
            scratch=shared,
        )
        for case in ("word_count", "kmeans")
    ]
    errors = [
        launch["error"]
        for record in records
        for launch in record["launches"]
        if launch.get("error")
    ]
    # Two cases x two baseline lanes x (one warm-up + one measured) launches.
    assert len(attempts) == 8
    assert len(errors) == 8 and all("deliberate read failure" in e for e in errors)
    assert not any("FileExistsError" in e for e in errors)
    assert {Path(r["scratch"]) for r in records} == {
        shared.resolve() / "word_count",
        shared.resolve() / "kmeans",
    }
    assert len(list(shared.rglob("error.txt"))) == 8
    # Each retained launch directory sits directly under its case's subdirectory.
    assert {p.parent.name for p in attempts} == {"word_count", "kmeans"}


def test_launch_on_board_missing_profile_preserves_diagnostics(monkeypatch, tmp_path):
    """Round 137 (ISSUES V1-21): a launch whose device log shows no fresh
    ``total`` record raises ``APUV1LaunchFailure`` carrying the argv, the
    binary identity, the process output, the device log read before the
    launch and every read after it, and the record counts, so the evidence
    survives the caller's build-directory cleanup; the freshness check itself
    is unchanged (no cycle count is produced)."""
    import contextlib
    import subprocess
    from types import SimpleNamespace

    from allo.pim import apu_v1_program as program

    binary = tmp_path / "tenon-kernel"
    binary.write_bytes(b"\x7fELF fake binary")
    stale = "ARCT[0]: *** total - hits:1 seu:1 crun:4242 iall:1 icm:0 dcm:0"
    monkeypatch.setattr(program, "apu_v1_board_lock", lambda: contextlib.nullcontext())
    monkeypatch.setattr(
        program, "_ledag_drain", lambda: stale
    )  # nothing new after the launch
    monkeypatch.setattr(program.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        program.subprocess,
        "run",
        lambda argv, **kwargs: SimpleNamespace(
            returncode=0, stdout=b"PASS!\n", stderr=b""
        ),
    )
    argv = [str(binary), str(tmp_path / "in.bin"), str(tmp_path / "out.bin")]
    with pytest.raises(
        program.APUV1LaunchFailure, match="freshness check failed"
    ) as info:
        program.launch_on_board(argv, tmp_path, expected_records=1, timeout=5)
    diagnostics = info.value.diagnostics
    assert diagnostics["argv"] == argv and diagnostics["cwd"] == str(tmp_path)
    assert (
        diagnostics["binary_sha256"] == hashlib.sha256(binary.read_bytes()).hexdigest()
    )
    assert diagnostics["binary_bytes"] == binary.stat().st_size
    assert diagnostics["attempts"] == [{"returncode": 0, "process_output": "PASS!\n"}]
    assert diagnostics["device_log_before"] == stale
    assert len(diagnostics["device_log_reads_after"]) == 4  # every read, not the last
    assert (
        diagnostics["fresh_records_found"] == 0 and diagnostics["expected_records"] == 1
    )
    assert "board_lock" in diagnostics and "time" in diagnostics
    # a failed device run carries the same evidence
    monkeypatch.setattr(
        program.subprocess,
        "run",
        lambda argv, **kwargs: SimpleNamespace(
            returncode=3, stdout=b"", stderr=b"boom"
        ),
    )
    with pytest.raises(program.APUV1LaunchFailure, match="device run failed") as info:
        program.launch_on_board(
            argv, tmp_path, expected_records=1, timeout=5, attempts=1
        )
    assert info.value.diagnostics["attempts"] == [
        {"returncode": 3, "process_output": "boom"}
    ]
    assert info.value.diagnostics["fresh_records_found"] is None


def _tenon_lane(tmp_path, compiled):
    import numpy as np

    lane = object.__new__(paper_rows.TenonLane)
    lane.case, lane.name, lane.count = "case", "tenon", 0
    lane.scratch = tmp_path / "scratch"
    lane.compiled = compiled
    lane.keyword_call, lane.parameters = False, ["x"]
    lane.arguments = {"x": np.zeros(4, dtype=np.uint16)}
    lane.in_place_outputs, lane.returned = [], ("y",)
    lane.oracle = np.arange(4, dtype=np.uint16)
    lane.device_source = "int main(void) { return 0; }\n"
    lane.device_source_sha256 = hashlib.sha256(lane.device_source.encode()).hexdigest()
    lane.substituted_source = None
    return lane


def test_apu_v1_tenon_lane_keeps_per_launch_evidence_and_failed_diagnostics(tmp_path):
    """Round 137 (ISSUES V1-21): a successful Tenon launch retains the
    generated source, the actual output arrays, the profiler text, the
    device log and the identities tying them together under the scratch
    directory; a launch the runtime could not attribute is recorded with its
    serialized diagnostics and no cycle count, and the session's launch
    record carries that evidence instead of the bare error."""
    import numpy as np
    from types import SimpleNamespace

    from allo.pim import apu_v1_program as program

    source = "int main(void) { return 0; }\n"

    def succeed(*args):
        return SimpleNamespace(
            kind="device",
            cycles=7,
            stdout="device log",
            extra={
                "outputs": {"y": np.arange(4, dtype=np.uint16)},
                "device_source": source,
                "binary_sha256": "a" * 64,
                "binary_bytes": 123,
                "profile_text": "total crun: 7 hits: 1",
                "profile_counters": {"crun": 7, "hits": 1},
                "board_lock": "/tmp/lock",
                "build_mode": "release",
            },
        )

    lane = _tenon_lane(tmp_path, succeed)
    launch = paper_rows.attempt_launch(lane)
    assert launch["crun"] == 7 and launch["outputs_match_oracle"]
    kept = tmp_path / "scratch" / "tenon-launch-001"
    assert launch["launch_dir"] == str(kept)
    assert (kept / "device.c").read_text() == source
    assert (
        launch["launched_source_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    )
    output = kept / "outputs" / "y.uint16.bin"
    assert np.fromfile(output, dtype=np.uint16).tolist() == [0, 1, 2, 3]
    assert (
        launch["outputs_sha256"]["y"] == hashlib.sha256(output.read_bytes()).hexdigest()
    )
    assert (kept / "profile.txt").read_text() == "total crun: 7 hits: 1"
    assert (kept / "device.log").read_text() == "device log"
    identity = json.loads((kept / "launch.json").read_text())
    assert identity["binary_sha256"] == "a" * 64 and identity["binary_bytes"] == 123
    assert (
        identity["crun"] == 7 and identity["outputs_sha256"] == launch["outputs_sha256"]
    )

    def lose_the_profile(*args):
        raise program.APUV1LaunchFailure(
            "APU v1 profile freshness check failed: expected 1 fresh 'total' "
            "PROF_PRINT record(s) after the launch, found 0",
            {
                "argv": ["/build/tenon-kernel", "in.bin", "out.bin"],
                "cwd": "/build",
                "binary_sha256": "b" * 64,
                "binary_bytes": 456,
                "board_lock": "/tmp/lock",
                "expected_records": 1,
                "attempts": [{"returncode": 0, "process_output": "PASS!"}],
                "device_log_before": "old record",
                "device_log_reads_after": ["old record", "old record"],
                "fresh_records_found": 0,
                "device_source": source,
                "device_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "native_vector_route": "gvml_route",
            },
        )

    lane = _tenon_lane(tmp_path / "second", lose_the_profile)
    with pytest.raises(paper_rows.LaunchFailure) as info:
        lane.launch()
    record = info.value.record
    failed = tmp_path / "second" / "scratch" / "tenon-launch-001"
    assert record["crun"] is None and record["outputs_match_oracle"] is None
    assert record["error"].startswith(
        "APUV1LaunchFailure: APU v1 profile freshness check failed"
    )
    assert record["launch_dir"] == str(failed)
    assert record["binary_sha256"] == "b" * 64 and record["binary_bytes"] == 456
    assert record["argv"] == ["/build/tenon-kernel", "in.bin", "out.bin"]
    assert (
        record["launched_source_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    )
    assert record["fresh_records_found"] == 0 and record["expected_records"] == 1
    assert (failed / "device.c").read_text() == source
    assert (failed / "device_log_before.txt").read_text() == "old record"
    assert (failed / "device_log_after_2.txt").read_text() == "old record"
    assert (failed / "process_output_1.txt").read_text() == "PASS!"
    diagnostics = json.loads((failed / "diagnostics.json").read_text())
    assert diagnostics["native_vector_route"] == "gvml_route"
    assert "device_source" not in diagnostics  # kept as device.c, not duplicated
    assert set(record["diagnostics"]) >= {"device.c", "diagnostics.json", "error.txt"}
    # the session keeps that record, with the error and without a cycle count
    lane = _tenon_lane(tmp_path / "third", lose_the_profile)
    entry = paper_rows.attempt_launch(lane)
    assert entry["error"].startswith("APUV1LaunchFailure") and entry["crun"] is None
    assert entry["binary_sha256"] == "b" * 64 and entry["launch_dir"].endswith(
        "tenon-launch-001"
    )
    # a lane raising without diagnostics still yields an error entry without timing
    lane = _tenon_lane(
        tmp_path / "fourth", lambda *a: (_ for _ in ()).throw(OSError("board gone"))
    )
    entry = paper_rows.attempt_launch(lane)
    assert entry["error"] == "OSError: board gone" and entry["crun"] is None
    assert (
        tmp_path / "fourth" / "scratch" / "tenon-launch-001" / "error.txt"
    ).is_file()
