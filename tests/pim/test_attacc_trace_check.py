# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The AttAcc trace check (``benchmarks/attacc/trace_check.py``): the
complete command stream a selected program implies, opcode and address of
every command, derived from the program and the target geometry and
compared with a trace command by command.  Corruptions that keep opcode
totals and distinct-address counts (zeroed store, drain and barrier
addresses; every address shifted by one channel; the matrix base shifted
by one column; two commands swapped) must fail for GEMV and attention
traces.  The last tests run DCC's own generator when a checkout is
available (``ATTACC_DCC_ROOT``) and require an exact match."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from allo.pim.attacc_tiling import (
    AttAccKernel,
    AttAccTiling,
    attacc_geometry,
    attacc_program,
)
from allo.pim.targets import build_attacc_target

from benchmarks.attacc import trace_check


@pytest.fixture(scope="module")
def geometry():
    return attacc_geometry(build_attacc_target())


QKV_NH8 = [[1, 8, 1], [8, 8, 80], [2, 1, 192]]
SCORE_480 = [[8, 1, 1], [2, 4, 16], [1, 16, 30]]


def _program(geometry, kind, dims, tiling):
    return attacc_program(
        AttAccKernel(kind, *dims), AttAccTiling.coerce(tiling), geometry
    )


def _mutations(lines, g):
    def split(line):
        op, addr = line.split()
        return op, int(addr, 16)

    yield "zero_st_drain_and_barrier_addresses", [
        f"{op} 0x00000000" if op in ("ST", "PIM_MV_SB", "PIM_BARRIER") else line
        for line in lines
        for op, _ in [split(line)]
    ]
    yield "shift_all_addresses_one_channel", [
        f"{op} 0x{addr + g['ch']:08x}" for line in lines for op, addr in [split(line)]
    ]
    yield "shift_mac_matrix_base_one_column", [
        f"{op} 0x{addr + g['col']:08x}" if op == "PIM_MAC_AB" else line
        for line in lines
        for op, addr in [split(line)]
    ]
    swapped = list(lines)
    i = next(k for k, l in enumerate(swapped) if l.startswith("PIM_MAC_AB"))
    swapped[i], swapped[i + 16] = swapped[i + 16], swapped[i]
    yield "swap_two_mac_commands", swapped
    yield "drop_last_result_store", lines[:-1]
    moved = list(lines)
    j = next(k for k, l in enumerate(moved) if l.startswith("PIM_WR_GB"))
    moved.append(moved.pop(j))
    yield "move_one_gb_write_to_the_end", moved
    yield "zero_softmax_and_context_moves", [
        f"{op} 0x00000000" if op in ("PIM_SFM", "PIM_MV_GB") else line
        for line in lines
        for op, _ in [split(line)]
    ]


@pytest.mark.parametrize(
    "kind, dims, tiling",
    [("gemv", (8, 5120, 384), QKV_NH8), ("attention", (8, 128, 480), SCORE_480)],
)
def test_derived_stream_matches_itself_and_its_counts(
    tmp_path, geometry, kind, dims, tiling
):
    program = _program(geometry, kind, dims, tiling)
    trace = trace_check.write_expected_trace(program, tmp_path / "ok.trace")
    report = trace_check.check_trace(trace, program)
    assert (
        report["match"]
        and report["totals_match"]
        and report["first_difference"] is None
    )
    assert report["totals"] == dict(program.counts.commands())
    assert report["summary"]["out_of_device_commands"] == 0
    assert report["regions"]["fits_bank"]


@pytest.mark.parametrize(
    "kind, dims, tiling",
    [("gemv", (8, 5120, 384), QKV_NH8), ("attention", (8, 128, 480), SCORE_480)],
)
def test_corruptions_that_keep_totals_fail(tmp_path, geometry, kind, dims, tiling):
    program = _program(geometry, kind, dims, tiling)
    lines = (
        trace_check.write_expected_trace(program, tmp_path / "ok.trace")
        .read_text()
        .splitlines()
    )
    g = trace_check.granules(geometry)
    for name, mutated in _mutations(lines, g):
        if kind == "gemv" and name == "zero_softmax_and_context_moves":
            continue  # no such commands in a GEMV trace
        path = tmp_path / f"{name}.trace"
        path.write_text("\n".join(mutated))
        report = trace_check.check_trace(path, program)
        assert not report["match"], name
        assert report["first_difference"] is not None, name
        if name == "shift_all_addresses_one_channel":
            assert report["summary"]["out_of_device_commands"] > 0
        if name != "drop_last_result_store":
            assert report["totals_match"], name  # the corruption is invisible to totals


def test_regions_are_the_generators_placement(geometry):
    """Vector at 0, matrix at the matrix size, result after the vector size
    (inside the matrix's address range), the block fitting the bank."""
    program = _program(geometry, "gemv", (8, 5120, 384), QKV_NH8)
    regions = trace_check.regions(program)
    assert (
        regions["vector"] == (0, 160)
        and regions["matrix"] == (30720, 30720)
        and regions["result"] == (30880, 384)
    )
    assert regions["fits_bank"] and regions["result_inside_matrix_range"]
    stream = trace_check.expected_trace(program)
    assert stream[0] == ("ST", 0)
    first_mac = next(addr for op, addr in stream if op == "PIM_MAC_AB")
    assert first_mac == 30720
    barrier = [addr for op, addr in stream if op == "PIM_BARRIER"][:16]
    assert barrier == [c * trace_check.granules(geometry)["ch"] for c in range(16)]


def test_wrong_program_for_a_trace_fails(tmp_path, geometry):
    """A trace of one tiling checked against the program of another."""
    a = _program(geometry, "gemv", (8, 5120, 384), QKV_NH8)
    b = _program(
        geometry, "gemv", (8, 5120, 384), [[1, 8, 1], [14, 8, 46], [1, 1, 384]]
    )
    trace = trace_check.write_expected_trace(a, tmp_path / "a.trace")
    assert not trace_check.check_trace(trace, b)["match"]


@pytest.mark.skipif(
    not os.environ.get("ATTACC_DCC_ROOT"),
    reason="needs a DCC checkout (ATTACC_DCC_ROOT)",
)
@pytest.mark.parametrize(
    "kind, dims, tiling",
    [
        ("gemv", (8, 5120, 384), QKV_NH8),
        ("gemv", (32, 128, 5120), [[1, 32, 1], [2, 2, 32], [8, 1, 640]]),
        ("gemv", (8, 5120, 384), [[8, 1, 1], [1, 4, 1280], [2, 16, 12]]),
        ("attention", (8, 128, 480), SCORE_480),
        ("attention", (32, 128, 4096), [[1, 32, 1], [1, 2, 64], [16, 1, 256]]),
        ("attention", (8, 128, 144), [[8, 1, 1], [2, 4, 16], [1, 16, 9]]),
    ],
)
def test_dcc_generator_trace_equals_the_derived_stream(
    tmp_path, geometry, kind, dims, tiling
):
    """DCC's own generator output equals the stream derived from the
    program, command by command (selected programs, a fixed-form baseline
    candidate, the worst-loss, a golden and the shortest attention shape)."""
    from benchmarks.attacc.scripts.run_ramulator import _dcc_root, write_trace

    program = _program(geometry, kind, dims, tiling)
    trace = write_trace(_dcc_root(), kind, program.tiling.dcc_tiling(), tmp_path)
    report = trace_check.check_trace(trace, program)
    assert report["match"], report["first_difference"]


# --------------------------------------------------------------------- #
# Failure-path retention of the timing driver
# --------------------------------------------------------------------- #


def _fake_generator(program):
    def write_trace(dcc_root, family, tiling, workdir):
        trace = trace_check.write_expected_trace(program, workdir / "kernel.trace")
        (workdir / "kernel.yaml").write_text("Frontend: {path: %s}\n" % trace)
        return trace

    return write_trace


def test_failed_simulation_retains_inputs_and_output(tmp_path, geometry, monkeypatch):
    """A simulator that exits nonzero: the trace, the configuration, the
    complete stdout and stderr and a structured failure record are retained
    under the artifacts directory, the work directory is removed, and the
    error propagates (the driver exits nonzero)."""
    from benchmarks.attacc.scripts import run_ramulator as driver

    program = _program(geometry, "gemv", (8, 5120, 384), QKV_NH8)
    monkeypatch.setattr(driver, "write_trace", _fake_generator(program))
    fake = tmp_path / "ramulator2"
    fake.write_text(
        "#!/bin/sh\necho 'partial output line'\necho 'injected simulator failure' >&2\nexit 3\n"
    )
    fake.chmod(0o755)
    artifacts = tmp_path / "artifacts"
    with pytest.raises(driver.SimulatorFailure, match="failed \\(3\\)") as error:
        driver.simulate_program(tmp_path, fake, "gemv", program, artifacts=artifacts)
    assert error.value.retained == str(artifacts)
    assert (artifacts / "kernel.trace").is_file() and (
        artifacts / "kernel.yaml"
    ).is_file()
    assert (artifacts / "stdout.txt").read_text().startswith("partial output line")
    assert "injected simulator failure" in (artifacts / "stderr.txt").read_text()
    failure = json.loads((artifacts / "failure.json").read_text())
    assert (
        failure["stage"] == "simulator"
        and failure["returncode"] == 3
        and failure["tiling"] == QKV_NH8
    )
    assert failure["trace_check"]["match"] and "-f" in failure["command"]
    assert not Path(failure["workdir"]).exists()


def test_failed_simulation_without_artifacts_keeps_the_work_directory(
    tmp_path, geometry, monkeypatch
):
    from benchmarks.attacc.scripts import run_ramulator as driver

    program = _program(geometry, "gemv", (8, 5120, 384), QKV_NH8)
    monkeypatch.setattr(driver, "write_trace", _fake_generator(program))
    fake = tmp_path / "ramulator2"
    fake.write_text("#!/bin/sh\nexit 1\n")
    fake.chmod(0o755)
    with pytest.raises(driver.SimulatorFailure) as error:
        driver.simulate_program(tmp_path, fake, "gemv", program)
    workdir = Path(error.value.retained)
    try:
        assert (workdir / "kernel.trace").is_file() and (
            workdir / "failure.json"
        ).is_file()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_simulator_timeout_is_a_retained_failure(tmp_path, geometry, monkeypatch):
    from benchmarks.attacc.scripts import run_ramulator as driver

    program = _program(geometry, "gemv", (8, 5120, 384), QKV_NH8)
    monkeypatch.setattr(driver, "write_trace", _fake_generator(program))
    fake = tmp_path / "ramulator2"
    fake.write_text("#!/bin/sh\necho started\nsleep 30\n")
    fake.chmod(0o755)
    artifacts = tmp_path / "artifacts"
    with pytest.raises(driver.SimulatorFailure, match="timed out"):
        driver.simulate_program(
            tmp_path, fake, "gemv", program, artifacts=artifacts, timeout_s=1
        )
    failure = json.loads((artifacts / "failure.json").read_text())
    assert (
        failure["stage"] == "simulator"
        and failure["returncode"] is None
        and failure["timeout_s"] == 1
    )
    assert (artifacts / "kernel.trace").is_file()


def test_trace_mismatch_retains_the_trace_and_the_check(
    tmp_path, geometry, monkeypatch
):
    from benchmarks.attacc.scripts import run_ramulator as driver

    program = _program(geometry, "gemv", (8, 5120, 384), QKV_NH8)
    other = _program(
        geometry, "gemv", (8, 5120, 384), [[1, 8, 1], [14, 8, 46], [1, 1, 384]]
    )
    monkeypatch.setattr(driver, "write_trace", _fake_generator(other))
    artifacts = tmp_path / "artifacts"
    with pytest.raises(driver.TraceMismatch):
        driver.simulate_program(
            tmp_path, tmp_path / "unused", "gemv", program, artifacts=artifacts
        )
    failure = json.loads((artifacts / "failure.json").read_text())
    assert failure["stage"] == "trace-check" and not failure["trace_check"]["match"]
    assert (artifacts / "kernel.trace").is_file() and not (
        artifacts / "stdout.txt"
    ).exists()


def test_driver_exits_nonzero_on_a_failed_shape(tmp_path, geometry, monkeypatch):
    from benchmarks.attacc.scripts import run_ramulator as driver

    program = _program(geometry, "gemv", (8, 5120, 384), QKV_NH8)
    monkeypatch.setattr(driver, "write_trace", _fake_generator(program))
    fake = tmp_path / "ramulator2"
    fake.write_text("#!/bin/sh\nexit 2\n")
    fake.chmod(0o755)
    monkeypatch.setenv("ATTACC_DCC_ROOT", str(tmp_path))
    (tmp_path / "attacc_trace_gen").mkdir()
    monkeypatch.setenv("ATTACC_RAMULATOR", str(fake))
    out = tmp_path / "session.json"
    status = driver.main(
        [
            "--gemv",
            "8",
            "5120",
            "384",
            "--artifacts",
            str(tmp_path / "art"),
            "--out",
            str(out),
        ]
    )
    assert status == 1
    session = json.loads(out.read_text())
    assert session["runs"] == [] and len(session["failures"]) == 1
    assert Path(session["failures"][0]["retained"]).joinpath("failure.json").is_file()
