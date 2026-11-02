# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gemini-II real-card tests (release audit G2-04).

Every test that needs the card skips with ``BLOCKED-DEVICE(<reason>)`` when
the SDK, ``gsi_tool`` or an Available card is missing (the APU v1 convention
of ``tests/pim/conftest.py``); a skip is never a pass.  Card access is
serialized by the runtime's own ``flock`` (``TENON_APU_G2_LOCK``, default
``/tmp/tenon-apu-g2-<device>.lock``), which these tests also hold.

Inputs are fresh random values (seed printed); exactness is enforced inside
the runtimes (they raise on any mismatch with their int64 oracle) and checked
again here against plain NumPy.  Cycles are device ticks per pipeline body
(``repetitions=20``), printed with the calibrated estimate; nothing is
compared with paper numbers.  Run with ``-s -rA`` to see the records.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time

import numpy as np
import pytest

import allo
from allo import spmw_simenv
from allo.ir.types import int4, int15
from allo.pim.apu_g2_typed_program import APUG2ScalarType, APUG2TypedProgram
from allo.pim.costs.apu_g2 import apu_g2_cost
from allo.pim.targets import build_apu_g2_target
from benchmarks.apu_g2.workloads import PAPER_KERNELS

REPETITIONS = 20
SEED = int(os.environ.get("TENON_APU_G2_TEST_SEED", time.time_ns() % (1 << 31)))


def apu_g2_card_unavailable_reason() -> str | None:
    """None when the Gemini-II SDK, gsi_tool and an Available card exist."""

    missing = [
        str(path) for path in spmw_simenv.apu_g2_required_sdk_paths() if not path.is_file()
    ]
    if missing:
        return f"Gemini-II SDK incomplete under {spmw_simenv.apu_g2_sdk_root()}: missing {missing[0]}"
    tool = spmw_simenv.apu_g2_gsi_tool()
    if tool is None:
        return "gsi_tool not found (PATH or TENON_APU_G2_GSI_TOOL)"
    device = spmw_simenv.apu_g2_device()
    try:
        status = subprocess.run(
            [tool, "info", device, "-v"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return f"gsi_tool info {device} failed: {error}"
    text = status.stdout + status.stderr
    if status.returncode:
        last = text.strip().splitlines()[-1:] or [""]
        return f"gsi_tool info {device} exited {status.returncode}: {last[0][:200]}"
    match = re.search(r"^Status\s*:\s*(.+?)\s*$", text, re.MULTILINE)
    if match is None or match.group(1) != "Available":
        return f"{device} status is {None if match is None else match.group(1)!r}"
    return None


@pytest.fixture
def apu_g2_card():
    """Skip (BLOCKED-DEVICE) without a card; hold the device lock otherwise."""

    reason = apu_g2_card_unavailable_reason()
    if reason is not None:
        pytest.skip(f"BLOCKED-DEVICE({reason})")
    from allo.pim.apu_g2_runtime import apu_g2_device_lock

    with apu_g2_device_lock() as path:
        yield path


def _record(kind, name, result, estimate=None, **fields):
    record = {
        "kind": kind,
        "name": name,
        "seed": SEED,
        "run_kind": result.kind,
        "cycles": result.cycles,
        "estimate_cycles": estimate,
        "final_pipeline_ticks": result.extra.get("final_pipeline_ticks"),
        "device_lock": result.extra.get("device_lock"),
        **fields,
    }
    print("APU_G2_RECORD " + json.dumps(record, sort_keys=True, default=str))
    return record


def S(bits, signed=True):
    return APUG2ScalarType(bits, signed)


def _typed_cases(rng):
    full = (4, 65_536)
    i16 = lambda: rng.integers(-(1 << 15), 1 << 15, size=full).astype(np.int16)
    i8 = lambda: rng.integers(-128, 128, size=full).astype(np.int8)
    u16 = lambda low: rng.integers(low, 1 << 16, size=full).astype(np.uint16)
    return [
        ("add", APUG2TypedProgram("add", (S(16), S(16)), S(16), full, repetitions=REPETITIONS),
         (i16(), i16()), lambda a, b: (a.astype(np.int64) + b).astype(np.int16)),
        ("mul", APUG2TypedProgram("mul", (S(8), S(8)), S(16), full, repetitions=REPETITIONS),
         (i8(), i8()), lambda a, b: a.astype(np.int64) * b),
        ("div", APUG2TypedProgram("div", (S(16, False),) * 2, S(16, False), full, repetitions=REPETITIONS),
         (u16(0), u16(1)), lambda a, b: a // b),
        ("block_sum", APUG2TypedProgram("block_sum", (S(8, False),), S(16, False), full, 256, repetitions=REPETITIONS),
         (rng.integers(0, 256, size=full).astype(np.uint8),),
         lambda a: a.astype(np.int64).reshape(-1, 256).sum(axis=1)),
        ("dot", APUG2TypedProgram("dot", (S(4), S(4)), S(15), (1985, 128), 128, repetitions=REPETITIONS),
         (rng.integers(-8, 8, size=(1985, 128)).astype(np.int8),
          rng.integers(-8, 8, size=(1985, 128)).astype(np.int8)),
         lambda a, b: (a.astype(np.int64) * b).sum(axis=1)),
    ]


def test_typed_program_per_operation_on_card(apu_g2_card):
    rng = np.random.default_rng(SEED)
    target = build_apu_g2_target()
    for name, program, operands, reference in _typed_cases(rng):
        compiled = allo.compile(program, target, apu_g2_cost)
        result = compiled(*operands)
        assert result.kind == "device"
        out = result.extra["outputs"]["out"]
        np.testing.assert_array_equal(out.astype(np.int64), reference(*operands))
        np.testing.assert_array_equal(out, result.extra["oracle"])
        _record(
            "typed",
            name,
            result,
            estimate=int(compiled.estimate().cycles),
            device_library_sha256=result.extra["project"]["artifact_sha256"]["device_library"],
        )


def _source_args(compiled, rng):
    contract = compiled.compiled.contract
    args = {}
    for name in contract.parameters:
        scalar = contract.parameter_type(name)
        shape = contract.parameter_shape(name)
        low, high = scalar.minimum, scalar.maximum
        carried = any(s.accumulator == "carried" and s.output == name for s in contract.streams)
        if name in contract.outputs and not carried:
            args[name] = np.zeros(shape, dtype=np.int16)
            continue
        if carried:
            dot_bits = contract.streams[0].exact_dot_type.bits
            low, high = -(1 << (dot_bits - 1)), (1 << (dot_bits - 1)) - 1
        args[name] = rng.integers(low, high + 1, size=shape).astype(np.int16)
    return args


def _reference(name, args):
    a = {key: value.astype(np.int64) for key, value in args.items()}
    if name in ("gemm", "batched_gemm"):
        return {"C": a["A"] @ a["B"]}
    if name == "gesummv":
        return {"y": 3 * (a["A"] @ a["x"]) + 2 * (a["B"] @ a["x"])}
    if name == "bicg":
        return {"q": a["A"] @ a["p"], "s": a["r"] @ a["A"]}
    return {"x1": a["x1"] + a["A"] @ a["y1"], "x2": a["x2"] + a["A"].T @ a["y2"]}


@pytest.mark.parametrize("name", sorted(PAPER_KERNELS))
def test_paper_source_kernel_on_card(apu_g2_card, name):
    rng = np.random.default_rng(SEED + sum(map(ord, name)))
    compiled = allo.compile(PAPER_KERNELS[name], build_apu_g2_target(), apu_g2_cost)
    compiled.compiled.repetitions = REPETITIONS
    args = _source_args(compiled, rng)
    expected = _reference(name, args)
    result = compiled(**args)
    assert result.kind == "device"
    for output, value in expected.items():
        np.testing.assert_array_equal(args[output].astype(np.int64), value)
    (launch,) = result.extra["launches"]
    (selected,) = compiled.compiled.plan.launches
    assert launch["layout_fingerprint"] == selected.layout_fingerprint
    _record(
        "source",
        name,
        result,
        estimate=compiled.compiled.plan.estimate_cycles,
        plan=compiled.compiled.plan.name,
        layout_fingerprint=launch["layout_fingerprint"],
        program_fingerprint=launch["program_fingerprint"],
        device_library_sha256=launch["artifact_sha256"]["device_library"],
        raw_ticks=launch["raw_ticks"],
    )


def gemm_n33(A: int4[64, 128], B: int4[128, 33], C: int15[64, 33]):
    for i in range(64):
        for j in range(33):
            acc: int15 = 0
            for k in range(128):
                acc += A[i, k] * B[k, j]
            C[i, j] = acc


def test_nearby_gemm_beyond_one_carrier_on_card(apu_g2_card):
    """Selected two-launch plan (extrapolated estimate): blind confirmation."""

    rng = np.random.default_rng(SEED + 33)
    compiled = allo.compile(gemm_n33, build_apu_g2_target(), apu_g2_cost)
    compiled.compiled.repetitions = REPETITIONS
    args = _source_args(compiled, rng)
    expected = args["A"].astype(np.int64) @ args["B"]
    result = compiled(**args)
    np.testing.assert_array_equal(args["C"].astype(np.int64), expected)
    _record(
        "source",
        "gemm_64x128x33",
        result,
        estimate=compiled.compiled.plan.estimate_cycles,
        plan=compiled.compiled.plan.name,
        estimate_validity=compiled.compiled.plan.estimate_validity,
        ranking=compiled.compiled.placement_ranking,
        launch_cycles=[item["cycles"] for item in result.extra["launches"]],
    )


# --------------------------------------------------------------------------- #
# Negative controls
# --------------------------------------------------------------------------- #


def test_wrong_oracle_is_reported_as_a_mismatch_on_card(apu_g2_card, monkeypatch):
    import allo.pim.apu_g2_composed_runtime as runtime

    original = runtime.expected_apu_g2_composed_output
    monkeypatch.setattr(
        runtime,
        "expected_apu_g2_composed_output",
        lambda *args, **kwargs: original(*args, **kwargs) + 1,
    )
    compiled = allo.compile(PAPER_KERNELS["gemm"], build_apu_g2_target(), apu_g2_cost)
    compiled.compiled.repetitions = REPETITIONS
    args = _source_args(compiled, np.random.default_rng(SEED + 1))
    with pytest.raises(RuntimeError, match="differs from NumPy"):
        compiled(**args)


def test_echoed_fingerprint_mismatch_is_refused_on_card(apu_g2_card, monkeypatch):
    import allo.pim.apu_g2_composed_runtime as runtime

    original = runtime.build_apu_g2_composed_host_command

    def wrong_fingerprint(program, host, files):
        return (*original(program, host, files)[:-1], "f" * 64)

    monkeypatch.setattr(runtime, "build_apu_g2_composed_host_command", wrong_fingerprint)
    compiled = allo.compile(PAPER_KERNELS["gemm"], build_apu_g2_target(), apu_g2_cost)
    compiled.compiled.repetitions = REPETITIONS
    args = _source_args(compiled, np.random.default_rng(SEED + 2))
    with pytest.raises(RuntimeError, match="layout_fingerprint"):
        compiled(**args)


def test_selected_fingerprint_mismatch_is_refused_before_launch(monkeypatch, tmp_path):
    """Needs no card: the runtime refuses before probing or building."""

    import allo.pim.apu_g2_composed_runtime as runtime

    monkeypatch.setenv("TENON_APU_G2_LOCK", str(tmp_path / "g2.lock"))
    probed = []
    monkeypatch.setattr(runtime, "_require_hardware_stack", lambda: probed.append(1))
    compiled = allo.compile(PAPER_KERNELS["gemm"], build_apu_g2_target(), apu_g2_cost)
    (launch,) = compiled.compiled.plan.launches
    operand = np.zeros(launch.program.dot_shape, dtype=np.int8)
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        runtime.run_apu_g2_composed(
            launch.program, operand, operand, expected_layout_fingerprint="0" * 64
        )
    assert probed == []
