# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static and behavioural checks for the ledag-ssh PROF_PRINT drain.

PROF_PRINT output from the ARC binary on Gemini does *not* appear on
the binary's stdout — it goes to the device system log, which is only
reachable via `ledag-ssh flo`. `_run_apu_v1` therefore drains that
channel after the binary exits and feeds the combined text to the
`crun=` parser. These tests assert that wiring without requiring a
real APU host.
"""

from __future__ import annotations

import inspect
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from allo import spmw_codegen
from allo.spmw_codegen import (
    RunResult,
    _parse_apu_v1_prof_print,
    _run_apu_v1,
)


# ----- Static source-level guarantees ----------------------------------------

_RUN_SRC = inspect.getsource(_run_apu_v1)


def test_run_apu_v1_invokes_ledag_ssh():
    """The fix wires `ledag-ssh -o localhost` into the run path."""
    assert "ledag-ssh" in _RUN_SRC
    assert '"-o"' in _RUN_SRC and '"localhost"' in _RUN_SRC


def test_run_apu_v1_sends_flo_quit_to_ledag():
    """The drain script is exactly `flo\\nquit\\n` — `flo` dumps the
    rolling device log, `quit` releases the session so the next run
    isn't blocked on a stale connection."""
    assert b"flo\nquit\n" in _RUN_SRC.encode() or "flo\\nquit\\n" in _RUN_SRC


def test_run_apu_v1_filters_ledag_bytes_to_printable():
    """ledag's wire format is binary-framed; the run path keeps only
    printable ASCII (plus tab/newline/CR) so the `crun=` regex can
    match against it."""
    assert "32 <= b < 127" in _RUN_SRC
    # tab(9), newline(10), CR(13) preserved
    assert "(9, 10, 13)" in _RUN_SRC


def test_run_apu_v1_guards_missing_ledag():
    """On hosts without `ledag-ssh`, the run path degrades to
    cycles=None rather than crashing."""
    assert "shutil.which" in _RUN_SRC
    assert '"ledag-ssh"' in _RUN_SRC


def test_run_apu_v1_preserves_cycles_none_on_missing_prof_print():
    """Spec 017 / SPEC-001 §3.5: a missing PROF_PRINT line yields
    cycles=None, never an exception. Scan the post-parse window for
    any `raise ` statement (ignoring comments) and fail if one
    appears -- the old behaviour raised RuntimeError here."""
    tail = _RUN_SRC.split("cycles = _parse_apu_v1_prof_print")[1].split("return RunResult")[0]
    code_lines = [
        ln for ln in tail.splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    for ln in code_lines:
        # An actual statement, not a substring inside a string literal.
        stripped = ln.strip()
        assert not stripped.startswith("raise "), (
            f"unexpected raise after PROF_PRINT parse: {ln!r}"
        )


# ----- Behavioural: end-to-end via mocked subprocess -------------------------

class _MockMake:
    """Stand-in for `subprocess.run(['make'], ...)` return value."""
    returncode = 0
    stdout = b""
    stderr = b""


class _MockBinary:
    """Stand-in for `subprocess.run([binary, *args], ...)` return value.

    Crucially this returns NO `crun=` text — the only place `crun=`
    can come from is the ledag drain, exactly as on real hardware.
    """
    returncode = 0
    stdout = b"host: wrote outputs\n"
    stderr = b""


class _MockLedag:
    """Stand-in for `subprocess.run(['ledag-ssh', ...], ...)` return.

    The real device emits binary-framed log records; this mock mixes a
    couple of high bytes around a PROF_PRINT line so the printable
    filter actually has work to do.
    """
    returncode = 0
    # Frame: 0xFE marker, "total: crun=12345 iall=...", 0xFE marker.
    stdout = (
        b"\xfe\xfe"
        b"total: crun=12345 iall=999 seu=0 dcm=0 @500MHz\n"
        b"\xfe"
    )
    stderr = b""


def _fake_run(args, **kwargs):
    """Subprocess router for the test: dispatches by argv[0]."""
    argv = list(args)
    if argv and argv[0] == "make":
        return _MockMake()
    if argv and argv[0] == "ledag-ssh":
        # Validate the call shape we promised: `-o localhost`, stdin
        # carries `flo\nquit\n`, and capture_output is on.
        assert argv[:3] == ["ledag-ssh", "-o", "localhost"], argv
        assert kwargs.get("input") == b"flo\nquit\n"
        assert kwargs.get("capture_output") is True
        return _MockLedag()
    # Anything else is the kernel binary.
    return _MockBinary()


def test_run_apu_v1_parses_crun_from_ledag_output(tmp_path, monkeypatch):
    """End-to-end: stub the env gate, build, and subprocess layer; the
    only `crun=` token lives in the mocked ledag stdout. Assert that
    `_run_apu_v1` still returns `cycles=12345`."""

    # 1. Pretend the APU host is present.
    monkeypatch.setattr(
        spmw_codegen, "_apu_v1_unavailable_reason", lambda: None
    )

    # 2. Stub the SDK probe and the project emitter.
    import allo.spmw_apu_v1_build as build_mod

    monkeypatch.setattr(
        build_mod, "_assert_gvml_sdk_present", lambda: None
    )

    def _fake_gen_project(*, dst_dir, compiled, inputs, output_specs, lab_name):
        project_dir = Path(dst_dir)
        (project_dir / "build" / "debug").mkdir(parents=True, exist_ok=True)
        binary = project_dir / "build" / "debug" / "tenon-kernel"
        binary.write_bytes(b"#!/bin/sh\nexit 0\n")
        binary.chmod(0o755)
        return project_dir

    monkeypatch.setattr(
        build_mod, "gen_apu_v1_low_mode_project", _fake_gen_project
    )

    # 3. Stub IO prep so we don't depend on a real Compiled.target.
    monkeypatch.setattr(
        spmw_codegen,
        "_apu_v1_prepare_io",
        lambda compiled, inputs: (inputs, {}),
    )
    monkeypatch.setattr(
        spmw_codegen, "_apu_v1_kernel_src", lambda compiled: ""
    )

    # 4. Stub subprocess and shutil.which.
    monkeypatch.setattr(spmw_codegen.subprocess, "run", _fake_run)
    monkeypatch.setattr(
        spmw_codegen.shutil, "which", lambda name: "/usr/bin/" + name
    )
    # Skip the post-binary settle sleep to keep the test fast.
    monkeypatch.setattr(spmw_codegen.time, "sleep", lambda _s: None)

    # 5. Run.
    class _FakeCompiled:
        target = type("T", (), {"name": "apu_v1"})()
        cmds = []

    import numpy as np

    result = _run_apu_v1(_FakeCompiled(), x=np.zeros(4, dtype=np.uint16))

    assert isinstance(result, RunResult)
    assert result.backend == "apu_v1"
    assert result.cycles == 12345, (
        f"expected cycles parsed from ledag drain, got {result.cycles!r}; "
        f"stdout tail:\n{result.stdout[-400:]}"
    )
    # Sanity: the printable filter dropped the 0xFE framing bytes.
    assert "\xfe" not in result.stdout


def test_run_apu_v1_returns_cycles_none_without_ledag(tmp_path, monkeypatch):
    """When `ledag-ssh` is not on PATH the run path must not crash; it
    returns RunResult(cycles=None) with the binary's own stdout."""

    monkeypatch.setattr(
        spmw_codegen, "_apu_v1_unavailable_reason", lambda: None
    )

    import allo.spmw_apu_v1_build as build_mod

    monkeypatch.setattr(
        build_mod, "_assert_gvml_sdk_present", lambda: None
    )

    def _fake_gen_project(*, dst_dir, compiled, inputs, output_specs, lab_name):
        project_dir = Path(dst_dir)
        (project_dir / "build" / "debug").mkdir(parents=True, exist_ok=True)
        binary = project_dir / "build" / "debug" / "tenon-kernel"
        binary.write_bytes(b"#!/bin/sh\nexit 0\n")
        binary.chmod(0o755)
        return project_dir

    monkeypatch.setattr(
        build_mod, "gen_apu_v1_low_mode_project", _fake_gen_project
    )
    monkeypatch.setattr(
        spmw_codegen,
        "_apu_v1_prepare_io",
        lambda compiled, inputs: (inputs, {}),
    )
    monkeypatch.setattr(
        spmw_codegen, "_apu_v1_kernel_src", lambda compiled: ""
    )

    def _no_ledag_run(args, **kwargs):
        argv = list(args)
        if argv and argv[0] == "ledag-ssh":
            pytest.fail("ledag-ssh should not be invoked when missing on PATH")
        if argv and argv[0] == "make":
            return _MockMake()
        return _MockBinary()

    monkeypatch.setattr(spmw_codegen.subprocess, "run", _no_ledag_run)
    monkeypatch.setattr(spmw_codegen.shutil, "which", lambda name: None)
    monkeypatch.setattr(spmw_codegen.time, "sleep", lambda _s: None)

    class _FakeCompiled:
        target = type("T", (), {"name": "apu_v1"})()
        cmds = []

    import numpy as np

    result = _run_apu_v1(_FakeCompiled(), x=np.zeros(4, dtype=np.uint16))

    assert isinstance(result, RunResult)
    assert result.backend == "apu_v1"
    assert result.cycles is None


# ----- Regression: colon-separator PROF_PRINT format ------------------------

def test_parse_apu_v1_prof_print_accepts_colon_separator():
    """APU v1 hardware emits PROF_PRINT with `crun:<N>` (colon), not
    `crun=<N>`. The exact line shape from a real Gemini run is:
        ARCT[0]: ***  total - hits:1 seu:374 crun:170227 iall:37027 ...
    The previous regex used `\\bcrun\\s*=\\s*(\\d+)` and silently
    returned None on real hardware. Regression-lock the colon path."""
    hw_line = (
        "ARCT[0]: ***  total - hits:1 seu:374 crun:170227 iall:37027 "
        "icm:292 dcm:6 microsec@500Mhz:340\n"
    )
    assert _parse_apu_v1_prof_print(hw_line) == 170227


def test_parse_apu_v1_prof_print_still_accepts_equals_separator():
    """Forward-compat: the original `=` format must keep working so old
    captured logs and the existing mocked-ledag test stay green."""
    legacy = "total: crun=12345 iall=999 seu=0 dcm=0 @500MHz\n"
    assert _parse_apu_v1_prof_print(legacy) == 12345


def test_parse_apu_v1_prof_print_returns_none_when_absent():
    assert _parse_apu_v1_prof_print("no prof line here\n") is None
