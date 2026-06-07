# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# verifier-owned: written by the verifier agent for task 012 end-to-end validation.
"""End-to-end pipeline test on all five backends using the MLP workload.

Pipeline: workload -> match -> autoschedule -> codegen -> run

Each non-hardware backend test:
  - Builds target + workload
  - Runs allo.match_workload
  - Runs allo.compile_for_target (autoschedule + codegen)
  - Calls compiled.run()
  - Asserts RunResult shape and (where available) cycle counts > 0

APU v1 test is marked @pytest.mark.hardware and skipped if device unavailable.

IMPORTANT: When a simulator is unavailable, RunResult.cycles is None.
  The test only asserts cycles > 0 when the simulator IS available
  (i.e., "simulator unavailable" is NOT in stdout).
"""

from __future__ import annotations

import pytest

import allo
from allo.spmw_codegen import RunResult, _pimsim_root, _docker_image_exists, _upim_root

from _fixtures import (
    build_mlp_workload,
    build_samsung_target,
    build_aim_target,
    build_upmem_target,
    build_apu_v1_target,
    build_apu_v2_target,
)


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _pipeline(target, workload):
    """Run the full workload -> match -> autoschedule -> compile pipeline.

    Returns (trace, compiled).
    """
    schedule = allo.customize(workload, enable_tensor=False)
    trace = allo.match_workload(target, schedule.module)
    compiled = allo.compile_for_target(target, trace)
    return trace, compiled


def _sim_unavailable(result: RunResult) -> bool:
    """Return True if the simulator/hardware was not reachable."""
    return "simulator unavailable" in result.stdout


# --------------------------------------------------------------------- #
# Cycle-count summary (printed at end of session by the fixture below)
# --------------------------------------------------------------------- #

_CYCLE_RESULTS: list[tuple[str, int | None, str]] = []


@pytest.fixture(autouse=True)
def _record_cycles(request):
    """Collect cycle counts from each test into the global table."""
    yield
    # Collect from request.node's stored result (set by tests below).
    info = getattr(request.node, "_e2e_result", None)
    if info is not None:
        _CYCLE_RESULTS.append(info)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print the cycle-count summary table when all tests have run."""
    if not _CYCLE_RESULTS:
        return
    terminalreporter.write_sep("=", "E2E MLP cycle-count summary")
    header = f"{'Backend':<22}  {'Cycles':>12}  Status"
    terminalreporter.write_line(header)
    terminalreporter.write_line("-" * 55)
    for backend, cycles, note in _CYCLE_RESULTS:
        cyc_s = str(cycles) if cycles is not None else "N/A"
        terminalreporter.write_line(f"{backend:<22}  {cyc_s:>12}  {note}")


# --------------------------------------------------------------------- #
# Samsung HBM-PIM
# --------------------------------------------------------------------- #


def test_e2e_mlp_samsung(request):
    """Full pipeline for Samsung HBM-PIM."""
    import numpy as np

    target = build_samsung_target()
    workload = build_mlp_workload()
    trace, compiled = _pipeline(target, workload)

    # Trace must contain MAC matches (MLP has two MAC layers).
    macs = trace.by_target_op("MAC")
    assert len(macs) >= 2, f"Expected >=2 MAC matches, got {len(macs)}"

    # SPEC-020: measure the actual MLP, not a padded substitute. The
    # cmd stream contains one MAC per layer; `_run_samsung` splits at
    # JUMP boundaries and runs `pim_driver` once per layer, summing the
    # PIM_CYCLES totals. Zero inputs are fine: we are measuring cycles
    # for the CRF microcode `compile_for_target` actually emitted, not
    # numerical correctness.
    W1 = np.zeros((256, 128), dtype=np.float16)
    x  = np.zeros(128,        dtype=np.float16)
    W2 = np.zeros((64, 256),  dtype=np.float16)
    h  = np.zeros(256,        dtype=np.float16)
    result = compiled.run(layers=[
        {"W": W1, "x": x},
        {"W": W2, "x": h},
    ])
    assert isinstance(result, RunResult), f"Expected RunResult, got {type(result).__name__}"
    assert result.backend == "samsung_hbm_pim", result.backend

    if not _sim_unavailable(result):
        assert result.cycles is not None and result.cycles > 0, (
            f"Samsung: expected positive cycles; got {result.cycles!r}; "
            f"stdout tail: {result.stdout[-400:]}"
        )
    else:
        assert result.cycles is None

    note = f"{'sim unavailable' if _sim_unavailable(result) else 'PASS'}"
    request.node._e2e_result = ("samsung_hbm_pim", result.cycles, note)


# --------------------------------------------------------------------- #
# SK-Hynix AiM (GDDR6 PIM)
# --------------------------------------------------------------------- #


def test_e2e_mlp_aim(request):
    """Full pipeline for SK-Hynix AiM. No numerical check (no functional data path)."""
    target = build_aim_target()
    workload = build_mlp_workload()
    trace, compiled = _pipeline(target, workload)

    macs = trace.by_target_op("MAC")
    assert len(macs) >= 2, f"Expected >=2 MAC matches, got {len(macs)}"

    result = compiled.run()
    assert isinstance(result, RunResult), f"Expected RunResult, got {type(result).__name__}"
    assert result.backend == "aim", result.backend

    if not _sim_unavailable(result):
        assert result.cycles is not None and result.cycles > 0, (
            f"AiM: expected positive cycles; got {result.cycles!r}; "
            f"stdout tail: {result.stdout[-400:]}"
        )
    else:
        assert result.cycles is None

    note = f"{'sim unavailable' if _sim_unavailable(result) else 'PASS'}"
    request.node._e2e_result = ("aim_gddr6_pim", result.cycles, note)


# --------------------------------------------------------------------- #
# UPMEM DPU
# --------------------------------------------------------------------- #


def test_e2e_mlp_upmem(request):
    """Full pipeline for UPMEM DPU."""
    target = build_upmem_target()
    workload = build_mlp_workload()
    trace, compiled = _pipeline(target, workload)

    macs = trace.by_target_op("MAC")
    assert len(macs) >= 2, f"Expected >=2 MAC matches, got {len(macs)}"

    result = compiled.run()
    assert isinstance(result, RunResult), f"Expected RunResult, got {type(result).__name__}"
    assert result.backend == "upmem", result.backend

    if not _sim_unavailable(result):
        assert result.cycles is not None and result.cycles > 0, (
            f"UPMEM: expected positive cycles; got {result.cycles!r}; "
            f"stdout tail: {result.stdout[-400:]}"
        )
    else:
        assert result.cycles is None

    note = f"{'sim unavailable' if _sim_unavailable(result) else 'PASS'}"
    request.node._e2e_result = ("upmem_dpu", result.cycles, note)


# --------------------------------------------------------------------- #
# APU v1 (real hardware, gated by @pytest.mark.hardware)
# --------------------------------------------------------------------- #


def _apu_v1_device_available() -> bool:
    """True iff ARC toolchain, PCI device, AND GVML SDK headers are all present."""
    import pathlib
    from allo.spmw_apu_v1_build import _gvml_sdk_available
    toolchain_bins = (
        list(pathlib.Path("/usr/local/gsi-apu").rglob("arc-elf32-gcc"))
        if pathlib.Path("/usr/local/gsi-apu").is_dir() else []
    )
    pci_present = pathlib.Path("/sys/bus/pci/devices/0000:41:00.0").exists()
    return bool(toolchain_bins) and pci_present and _gvml_sdk_available()


@pytest.mark.hardware
@pytest.mark.skipif(
    not _apu_v1_device_available(),
    reason="APU v1 hardware not available: ARC toolchain, PCI 41:00.0, or GVML SDK absent",
)
def test_e2e_mlp_apu_v1(request):
    """Full pipeline for GSI APU v1 real hardware (PCI 41:00.0).

    Hardware/SDK absence is handled entirely by the @pytest.mark.skipif
    decorator above (via `_apu_v1_device_available`). Once the test runs,
    cycle count must be present and positive -- build/runtime failures
    in `_run_apu_v1` now raise `RuntimeError` rather than returning a
    silent `cycles=None`.
    """
    target = build_apu_v1_target()
    workload = build_mlp_workload()
    trace, compiled = _pipeline(target, workload)

    macs = trace.by_target_op("MAC")
    assert len(macs) >= 2, f"Expected >=2 MAC matches, got {len(macs)}"

    result = compiled.run()
    assert isinstance(result, RunResult), f"Expected RunResult, got {type(result).__name__}"
    assert result.backend == "apu_v1", result.backend
    # Real-hardware run: cycles must be present and positive. Hardware
    # absence is handled by @pytest.mark.skipif above; we never expect
    # to reach this assertion with cycles=None.
    assert result.cycles is not None and result.cycles > 0, (
        f"APU v1 hardware returned cycles={result.cycles!r}; "
        f"stdout tail: {result.stdout[-400:]}"
    )

    request.node._e2e_result = ("apu_v1", result.cycles, "PASS (hw)")


# --------------------------------------------------------------------- #
# APU v2 (functional only, cycles=None by design)
# --------------------------------------------------------------------- #


def test_e2e_mlp_apu_v2(request):
    """Full pipeline for GSI APU v2 (l1_sim, functional only).

    l1_sim does not report cycle counts; cycles must always be None.
    """
    target = build_apu_v2_target()
    workload = build_mlp_workload()
    trace, compiled = _pipeline(target, workload)

    macs = trace.by_target_op("MAC")
    assert len(macs) >= 2, f"Expected >=2 MAC matches, got {len(macs)}"

    result = compiled.run()
    assert isinstance(result, RunResult), f"Expected RunResult, got {type(result).__name__}"
    assert result.backend == "apu_v2", result.backend
    assert result.cycles is None, (
        f"APU v2 l1_sim is functional-only; expected cycles=None, got {result.cycles!r}"
    )

    request.node._e2e_result = ("apu_v2", None, "PASS (functional)")


# --------------------------------------------------------------------- #
# Parametrised convenience alias (used by the task spec)
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("target_name", [
    "samsung_hbm_pim",
    "aim_gddr6_pim",
    "upmem_dpu",
    pytest.param("apu_v1", marks=pytest.mark.hardware),
    "apu_v2",
])
def test_e2e_mlp_parametrised(target_name):
    """Parametrised smoke test that mirrors the individual tests above.

    This is a lighter version: verifies the pipeline runs and RunResult
    has the right shape; doesn't check cycles > 0. Use the individual
    tests above for the authoritative assertion set.
    """
    _TARGET_MAP = {
        "samsung_hbm_pim": (build_samsung_target, "samsung_hbm_pim"),
        "aim_gddr6_pim": (build_aim_target, "aim"),
        "upmem_dpu": (build_upmem_target, "upmem"),
        "apu_v1": (build_apu_v1_target, "apu_v1"),
        "apu_v2": (build_apu_v2_target, "apu_v2"),
    }
    builder, backend_key = _TARGET_MAP[target_name]

    if target_name == "apu_v1" and not _apu_v1_device_available():
        pytest.skip("APU v1 hardware not available")

    target = builder()
    workload = build_mlp_workload()
    trace, compiled = _pipeline(target, workload)

    assert len(trace.by_target_op("MAC")) >= 2

    result = compiled.run()
    assert isinstance(result, RunResult)
    assert result.backend == backend_key
    assert result.cycles is None or (isinstance(result.cycles, int) and result.cycles >= 0)
    assert isinstance(result.stdout, str)


if __name__ == "__main__":
    test_e2e_mlp_samsung(None)
    test_e2e_mlp_aim(None)
    test_e2e_mlp_upmem(None)
    test_e2e_mlp_apu_v2(None)
    print("ALL PASSED")
