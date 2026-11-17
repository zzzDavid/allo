# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The real subprocess boundary of direct profile copies; no device execution."""

import json
import os
from pathlib import Path
import sys

import pytest

from allo.pim import apu_v1_program as runtime


@pytest.fixture
def storage(monkeypatch, tmp_path):
    paths = {
        name: tmp_path / name for name in ("session", "extract", "profiles", "bin")
    }
    for path in paths.values():
        path.mkdir()
    monkeypatch.setattr(runtime, "_LEDAG_SCRATCH", paths["session"])
    monkeypatch.setenv("TMPDIR", str(paths["extract"]))
    monkeypatch.setenv("TENON_APU_V1_PROFILE_DIR", str(paths["profiles"]))
    monkeypatch.setenv("TENON_APU_V1_LOCK", str(tmp_path / "board.lock"))
    monkeypatch.setenv("PATH", str(paths["bin"]) + os.pathsep + os.environ["PATH"])
    return paths


def fake_cli(storage, mode="ok"):
    # A subprocess (not a mocked CompletedProcess) exercises session ownership,
    # communicate retries, failures, stdout preservation and actual file copies.
    source = f"""#!{sys.executable}
import os, pathlib, sys, time
print("Version: fake-cli-1")
print("'Version': 'fake-firmware-2'")
session = pathlib.Path({str(storage['session'])!r})
(session / ("ledag_ssh_test_%d_log.txt" % os.getpid())).write_text("owned session evidence")
(session / "ledag_ssh_other_99999999_log.txt").write_text("unrelated simultaneous session")
time.sleep(0.3)
mode = {mode!r}
if mode == 'timeout':
    print('partial timeout output', flush=True)
    time.sleep(20)
for line in sys.stdin:
    if line == 'quit\\n': break
    cmd, direction, source, destination = line.split()
    assert (cmd, direction) == ('scp', 'd2h')
    assert source.startswith('/run/media/mmcblk0p2/logs/')
    slot = int(source.rsplit('/', 1)[1])
    # Wrap around: slot 5 oldest, slot 4 newest; lexical filename order is wrong.
    order = (slot - 5) % 10
    stamp = 'TIME STAMP: 11/17/26 05:00:%02d.000000001 UTC\\n' % order
    body = stamp + 'total crun: %d hits: 1\\n' % (order + 1)
    if slot == 4:
        if mode == 'missing': continue
        if mode == 'stamp': body = 'total crun: 10 hits: 1\\n'
        if mode == 'backwards': body += 'TIME STAMP: 11/17/26 04:00:00.000000001 UTC\\n'
        if mode == 'overlap': body = body.replace('05:00:09', '05:00:08')
    pathlib.Path(destination).write_bytes(body.encode() + b'\\0')
    if mode != 'ack': print('Download Success', flush=True)
if mode == 'exit':
    print('copy error', file=sys.stderr)
    sys.exit(3)
"""
    path = storage["bin"] / "ledag-ssh"
    path.write_text(source)
    path.chmod(0o755)


def test_direct_copy_orders_rotation_preserves_evidence_and_owns_cleanup(storage):
    fake_cli(storage)
    older = storage["session"] / "online_log_old"
    older.write_text("unrelated flo history")
    text = runtime._ledag_drain()
    records = list(runtime._TOTAL_RECORD.finditer(text))
    assert [int(r.group(1)) for r in records] == list(range(1, 11))
    (evidence,) = storage["profiles"].iterdir()
    manifest = json.loads((evidence / "manifest.json").read_text())
    assert manifest["status"] == "copied"
    assert manifest["ledag_ssh_version"] == "fake-cli-1"
    assert manifest["firmware_versions"] == ["fake-firmware-2"]
    assert manifest["elapsed_seconds"] > 0
    assert [f["slot"] for f in manifest["files"]] == [5, 6, 7, 8, 9, 0, 1, 2, 3, 4]
    assert str(evidence) in text and manifest["extraction"] == str(storage["extract"])
    assert len(list(evidence.glob("ledag_ssh_test_*"))) == 1
    assert sorted(p.name for p in storage["session"].iterdir()) == [
        "ledag_ssh_other_99999999_log.txt",
        "online_log_old",
    ]
    # The return still enters the strict launch freshness parser, including a
    # rotated prefix. Old records alone are never an attributable measurement.
    assert runtime.parse_fresh_profile(text, text + "total crun: 19 hits: 1\n", 1) == (
        19,
    )
    with pytest.raises(RuntimeError, match="expected 1 fresh"):
        runtime.parse_fresh_profile(text, text, 1)


@pytest.mark.parametrize(
    "mode, message",
    [
        ("missing", "did not confirm"),
        ("ack", "did not confirm"),
        ("exit", "exited with 3"),
        ("stamp", "missing or nonmonotonic"),
        ("backwards", "missing or nonmonotonic"),
        ("overlap", "ambiguous chronology"),
        ("timeout", "timed out"),
    ],
)
def test_copy_failures_retain_partial_evidence(storage, monkeypatch, mode, message):
    fake_cli(storage, mode)
    if mode == "timeout":
        monkeypatch.setattr(runtime, "_LEDAG_COPY_TIMEOUT_SECONDS", 0.5)
    with pytest.raises(RuntimeError, match=message) as error:
        runtime._ledag_drain()
    (evidence,) = storage["profiles"].iterdir()
    assert str(evidence) in str(error.value)
    manifest = json.loads((evidence / "manifest.json").read_text())
    assert manifest["status"] == "incomplete" and message in manifest["error"]
    assert (evidence / "stdout.txt").exists() and (evidence / "stderr.txt").exists()
    if mode == "timeout":
        assert "partial timeout output" in (evidence / "stdout.txt").read_text()
    assert not list(storage["session"].glob("ledag_ssh_test_*"))
    assert list(storage["session"].glob("ledag_ssh_other_*"))


def test_missing_file_cannot_hide_behind_success_messages(storage):
    for slot in range(9):
        (storage["profiles"] / str(slot)).write_text(
            f"TIME STAMP: 11/17/26 05:00:{slot:02d}.000000001 UTC\n"
        )
    with pytest.raises(FileNotFoundError):
        runtime._ledag_order_logs(storage["profiles"])


def test_headroom_groups_actual_filesystems_and_checks_residual_tmp(
    storage, monkeypatch
):
    from types import SimpleNamespace

    # Model three distinct mounts; actual temp dirs in a unit test share one.
    monkeypatch.setattr(runtime, "_same_filesystem", lambda a, b: a == b)
    free = {str(p): 10**9 for p in storage.values()}
    monkeypatch.setattr(
        runtime.shutil, "disk_usage", lambda p: SimpleNamespace(free=free[str(p)])
    )
    free[str(storage["session"])] = 128 * 1024  # enough for small residual, not old flo
    assert runtime.ledag_scratch_headroom_problem() is None
    for key, minimum, message in [
        ("session", runtime._LEDAG_LOG_REQUIRED_BYTES, "vendor session log"),
        ("extract", runtime._LEDAG_EXTRACTION_REQUIRED_BYTES, "bundle extraction"),
        ("profiles", runtime._LEDAG_COPY_REQUIRED_BYTES, "direct-copy firmware logs"),
    ]:
        previous = free[str(storage[key])]
        free[str(storage[key])] = minimum - 1
        with pytest.raises(RuntimeError, match=message):
            runtime._ledag_drain()  # refuses before starting any subprocess
        assert not list(storage["profiles"].iterdir())
        free[str(storage[key])] = previous
    monkeypatch.setattr(runtime, "_same_filesystem", lambda a, b: True)
    total = (
        runtime._LEDAG_LOG_REQUIRED_BYTES
        + runtime._LEDAG_EXTRACTION_REQUIRED_BYTES
        + runtime._LEDAG_COPY_REQUIRED_BYTES
    )
    free[str(storage["session"])] = total - 1
    assert runtime.ledag_scratch_headroom_problem() is not None
    free[str(storage["session"])] = total
    assert runtime.ledag_scratch_headroom_problem() is None


def test_unsafe_vendor_path_is_refused_before_command(storage, monkeypatch):
    unsafe = storage["profiles"] / "path with spaces"
    unsafe.mkdir()
    monkeypatch.setenv("TENON_APU_V1_PROFILE_DIR", str(unsafe))
    with pytest.raises(RuntimeError, match="must use only"):
        runtime._ledag_drain()
    assert not list(unsafe.iterdir())


def test_unconfigured_retention_refuses_without_touching_tmp(storage, monkeypatch):
    monkeypatch.delenv("TENON_APU_V1_PROFILE_DIR")
    with pytest.raises(RuntimeError, match="set TENON_APU_V1_PROFILE_DIR"):
        runtime._ledag_drain()
    assert not list(storage["extract"].iterdir())
    assert not list(storage["profiles"].iterdir())


def test_copy_reserve_and_timeout_are_configurable(storage, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        runtime.shutil, "disk_usage", lambda p: SimpleNamespace(free=128 * 1024**2)
    )
    monkeypatch.setenv("TENON_APU_V1_PROFILE_COPY_BYTES", str(256 * 1024**2))
    assert "direct-copy firmware logs" in runtime.ledag_scratch_headroom_problem()
    monkeypatch.setenv("TENON_APU_V1_PROFILE_COPY_BYTES", str(64 * 1024**2))
    monkeypatch.setenv("TENON_APU_V1_PROFILE_TIMEOUT_SECONDS", "1")
    fake_cli(storage, "timeout")
    with pytest.raises(RuntimeError, match="timed out after 1 seconds"):
        runtime._ledag_drain()
    monkeypatch.setenv("TENON_APU_V1_PROFILE_TIMEOUT_SECONDS", "0")
    assert "must be a positive integer" in runtime.ledag_scratch_headroom_problem()
