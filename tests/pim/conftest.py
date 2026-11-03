# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PIM suite conftest (spec Answer 5).

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

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "apu_v1_device: APU v1 real-device test; gated to the server with the "
        "Gemini board (skips + records BLOCKED-DEVICE when unreachable). Never "
        "sim-substituted.",
    )
    config.addinivalue_line(
        "markers",
        "paper_full: long paper-golden rows; run with `-m paper_full`.",
    )


def pytest_collection_modifyitems(config, items):
    """`paper_full` rows run only when selected with `-m paper_full`."""
    if "paper_full" in (config.getoption("markexpr") or ""):
        return
    skip = pytest.mark.skip(reason="paper_full row; select with -m paper_full")
    for item in items:
        if "paper_full" in item.keywords:
            item.add_marker(skip)


def apu_v1_unavailable_reason() -> str | None:
    """The device precondition (ARC toolchain / gvml template / GSI PCI node /
    GVML SDK). None means the board is reachable on this host."""
    from allo.spmw_simenv import apu_v1_unavailable_reason as reason

    return reason()


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


def _simulator_gate(target_name: str):
    from allo.spmw_codegen import simulator_unavailable_reason

    reason = simulator_unavailable_reason(target_name)
    if reason is not None:
        pytest.skip(f"BLOCKED-SIM({reason})")


@pytest.fixture
def samsung_sim_gate():
    """Skip when PIMSimulator's `pim_driver` cannot run here. A simulator test
    that does not request its gate and hits `SimulatorUnavailable` fails."""
    _simulator_gate("samsung_hbm_pim")
    yield


@pytest.fixture
def aim_sim_gate():
    """Skip when the ramulator2 AiM Docker image or its config is absent."""
    _simulator_gate("aim")
    yield


@contextlib.contextmanager
def board_lock():
    """Serialize APU v1 board access across pytest workers and ad hoc runs.

    Delegates to the production lock (`allo.pim.apu_v1_program.
    apu_v1_board_lock`: one flock per PCI device node, overridable with
    `TENON_APU_V1_LOCK`), which every device launch also takes; it is
    reentrant within a process, so holding it around `compiled(...)` is safe."""
    from allo.pim.apu_v1_program import apu_v1_board_lock

    with apu_v1_board_lock():
        yield
