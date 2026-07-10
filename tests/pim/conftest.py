# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench-on-PIM suite conftest (spec Answer 5).

Adds the `apu_v1_device` marker that gates every APU v1 test to this server
(the only host with the board), a session fixture that skips + records
`BLOCKED-DEVICE(<reason>)` when the board is unreachable, and a file-lock that
serializes board access (the board is a shared resource). All ADDITIVE -- this
conftest only governs the `tests/pim/` tree; it changes no `spmw_*` behavior and
does not touch the spmw suite's import setup.

The device path NEVER sim-substitutes (spec Answer 5 ban): a board-unreachable
apu_v1 cell is `BLOCKED-DEVICE`, never an `l1_sim`/`virtual`/analytical number.
"""

from __future__ import annotations

import contextlib
import pathlib
import shutil
import subprocess

import pytest

# Importing conftest's dir onto sys.path is automatic (pytest prepends the
# rootdir of the conftest); that makes `import lib.<m>` resolve from any kernel
# folder under tests/pim/.

_BOARD_LOCK = pathlib.Path(__file__).resolve().parent / ".apu_v1.lock"
_APU_G2_LOCK = pathlib.Path(__file__).resolve().parent / ".apu_g2.lock"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "apu_v1_device: APU v1 real-device test; gated to the server with the "
        "Gemini board (skips + records BLOCKED-DEVICE when unreachable). Never "
        "sim-substituted.",
    )
    config.addinivalue_line(
        "markers",
        "apu_g2_device: APUg2 real-device VL64 test; never simulation-substituted.",
    )


def apu_v1_unavailable_reason() -> str | None:
    """The device precondition (ARC toolchain / gvml template / GSI PCI node),
    via the existing `_apu_v1_unavailable_reason()` -- re-exported, not
    re-implemented. None means the board is reachable on this host."""
    from allo.spmw_codegen import _apu_v1_unavailable_reason

    return _apu_v1_unavailable_reason()


@pytest.fixture
def apu_v1_device_gate():
    """Skip (recording BLOCKED-DEVICE) when the APU v1 board is unreachable.

    A skip means 'this environment cannot run the cell' (spec Answer 3: skipif
    is reserved for environment gating). A kernel test records
    `BLOCKED-DEVICE(<reason>)` in its results.json before the skip fires so the
    coverage matrix carries the honest verdict."""
    reason = apu_v1_unavailable_reason()
    if reason is not None:
        pytest.skip(f"BLOCKED-DEVICE({reason})")
    yield


@contextlib.contextmanager
def board_lock():
    """Serialize APU v1 board access across pytest workers (shared resource).

    Uses an flock on `.apu_v1.lock`; a no-op fallback where flock is
    unavailable. The lock lives in the suite's conftest, not in `spmw_*`."""
    try:
        import fcntl
    except ImportError:  # non-POSIX: best-effort, no lock
        yield
        return
    with open(_BOARD_LOCK, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def apu_g2_unavailable_reason() -> str | None:
    """Return why the installed Gemini-II hardware path cannot run."""

    required = (
        pathlib.Path("/dev/gsi/g2apu/apu-00"),
        pathlib.Path("/opt/gsi/g2/vector_core/lib/libg2_64vl.a"),
        pathlib.Path("/opt/gsi/share/g2_transport"),
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        return "missing " + ", ".join(missing)
    if shutil.which("gsi_tool") is None or shutil.which("cmake") is None:
        return "gsi_tool/cmake is not on PATH"
    try:
        probe = subprocess.run(
            ["gsi_tool", "info", "apu-00", "-v"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return f"card probe failed: {error}"
    if probe.returncode != 0:
        return f"card probe exited {probe.returncode}"
    if "Status          : Available" not in probe.stdout:
        return "card is not Available"
    return None


@pytest.fixture
def apu_g2_device_gate():
    reason = apu_g2_unavailable_reason()
    if reason is not None:
        pytest.skip(f"BLOCKED-DEVICE({reason})")
    yield


@contextlib.contextmanager
def apu_g2_board_lock():
    """Serialize G2 card allocation without sharing the unrelated v1 lock."""

    try:
        import fcntl
    except ImportError:
        yield
        return
    with open(_APU_G2_LOCK, "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
