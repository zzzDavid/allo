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
        str(path)
        for path in spmw_simenv.apu_g2_required_sdk_paths()
        if not path.is_file()
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
        (
            "add",
            APUG2TypedProgram(
                "add", (S(16), S(16)), S(16), full, repetitions=REPETITIONS
            ),
            (i16(), i16()),
            lambda a, b: (a.astype(np.int64) + b).astype(np.int16),
        ),
        (
            "mul",
            APUG2TypedProgram(
                "mul", (S(8), S(8)), S(16), full, repetitions=REPETITIONS
            ),
            (i8(), i8()),
            lambda a, b: a.astype(np.int64) * b,
        ),
        (
            "div",
            APUG2TypedProgram(
                "div", (S(16, False),) * 2, S(16, False), full, repetitions=REPETITIONS
            ),
            (u16(0), u16(1)),
            lambda a, b: a // b,
        ),
        (
            "block_sum",
            APUG2TypedProgram(
                "block_sum",
                (S(8, False),),
                S(16, False),
                full,
                256,
                repetitions=REPETITIONS,
            ),
            (rng.integers(0, 256, size=full).astype(np.uint8),),
            lambda a: a.astype(np.int64).reshape(-1, 256).sum(axis=1),
        ),
        (
            "dot",
            APUG2TypedProgram(
                "dot", (S(4), S(4)), S(15), (1985, 128), 128, repetitions=REPETITIONS
            ),
            (
                rng.integers(-8, 8, size=(1985, 128)).astype(np.int8),
                rng.integers(-8, 8, size=(1985, 128)).astype(np.int8),
            ),
            lambda a, b: (a.astype(np.int64) * b).sum(axis=1),
        ),
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
            device_library_sha256=result.extra["project"]["artifact_sha256"][
                "device_library"
            ],
        )


def _source_args(compiled, rng):
    contract = compiled.compiled.contract
    args = {}
    for name in contract.parameters:
        scalar = contract.parameter_type(name)
        shape = contract.parameter_shape(name)
        low, high = scalar.minimum, scalar.maximum
        carried = any(
            s.accumulator == "carried" and s.output == name for s in contract.streams
        )
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

    monkeypatch.setattr(
        runtime, "build_apu_g2_composed_host_command", wrong_fingerprint
    )
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


# --------------------------------------------------------------------------- #
# G6 semantic probes on the card.  tests/pim/test_g6_semantics.py runs the
# same host path against the emulated card; these confirm the device.
# --------------------------------------------------------------------------- #


def g6_gemm_square(A: int4[32, 32], B: int4[32, 32], C: int15[32, 32]):
    for i in range(32):
        for j in range(32):
            acc: int15 = 0
            for k in range(32):
                acc += A[i, k] * B[k, j]
            C[i, j] = acc


def test_call_aliases_read_pre_call_values_on_card(apu_g2_card):
    """One array passed for two parameters: every input is staged before any
    output is written, so the result is computed from the pre-call values."""

    rng = np.random.default_rng(SEED + 61)
    target = build_apu_g2_target()
    gemm = allo.compile(g6_gemm_square, target, apu_g2_cost)
    gemm.compiled.repetitions = REPETITIONS
    A0 = rng.integers(-8, 8, size=(32, 32)).astype(np.int16)
    B0 = rng.integers(-8, 8, size=(32, 32)).astype(np.int16)

    shared = A0.copy()
    out = np.full((32, 32), 7, np.int16)
    result = gemm(shared, shared, out)
    assert result.kind == "device"
    np.testing.assert_array_equal(out, A0.astype(np.int64) @ A0)
    _record("g6_alias", "gemm_lhs_is_rhs", result)

    output_is_lhs = A0.copy()
    result = gemm(output_is_lhs, B0, output_is_lhs)
    np.testing.assert_array_equal(output_is_lhs, A0.astype(np.int64) @ B0)
    _record("g6_alias", "gemm_output_is_lhs", result)

    mvt = allo.compile(PAPER_KERNELS["mvt"], target, apu_g2_cost)
    mvt.compiled.repetitions = REPETITIONS
    M = rng.integers(-4, 4, size=(256, 256)).astype(np.int16)
    y1 = rng.integers(-4, 4, size=256).astype(np.int16)
    y2 = rng.integers(-4, 4, size=256).astype(np.int16)
    x2 = rng.integers(-1000, 1000, size=256).astype(np.int16)
    x1_is_y1, x2_out = y1.copy(), x2.copy()
    result = mvt(M, x1_is_y1, y2, x1_is_y1, x2_out)
    np.testing.assert_array_equal(x1_is_y1, y1 + M.astype(np.int64) @ y1)
    np.testing.assert_array_equal(x2_out, x2 + M.T.astype(np.int64) @ y2)
    _record("g6_alias", "mvt_x1_is_y1", result)


def test_extreme_operands_on_card(apu_g2_card):
    """i8 products at -128/127 into i16, and an int4 GEMM reaching both ends
    of its exact int15 dot range (8192 = (-8)(-8)128, -7168 = (-8)(7)128)."""

    target = build_apu_g2_target()
    full = (4, 65_536)
    lhs = np.resize(np.array([-128, 127, 0, -128, 127], np.int8), full)
    rhs = np.resize(np.array([-128, -128, 127, 127, 127, -1, 0], np.int8), full)
    mul = allo.compile(
        APUG2TypedProgram("mul", (S(8), S(8)), S(16), full, repetitions=REPETITIONS),
        target,
        apu_g2_cost,
    )
    out = np.full(full, 7, np.int16)
    result = mul(lhs, rhs, out)
    np.testing.assert_array_equal(out.astype(np.int64), lhs.astype(np.int64) * rhs)
    assert (out.max(), out.min()) == (16384, -16256)
    _record("g6_extreme", "mul_i8_i16", result)

    gemm = allo.compile(PAPER_KERNELS["gemm"], target, apu_g2_cost)
    gemm.compiled.repetitions = REPETITIONS
    A = np.full((64, 128), -8, np.int16)
    B = np.full((128, 31), -8, np.int16)
    B[:, 1::2] = 7
    C = np.full((64, 31), 7, np.int16)
    result = gemm(A, B, C)
    np.testing.assert_array_equal(C, A.astype(np.int64) @ B)
    assert (C.max(), C.min()) == (8192, -7168)
    _record("g6_extreme", "gemm_int4_edges", result)


def test_zero_results_overwrite_a_sentinel_on_card(apu_g2_card):
    rng = np.random.default_rng(SEED + 62)
    target = build_apu_g2_target()
    full = (4, 65_536)
    a = rng.integers(-(1 << 15), 1 << 15, size=full).astype(np.int16)
    b = (-a.astype(np.int32)).astype(np.int16)  # -(-32768) wraps to -32768
    add = allo.compile(
        APUG2TypedProgram("add", (S(16), S(16)), S(16), full, repetitions=REPETITIONS),
        target,
        apu_g2_cost,
    )
    out = np.full(full, 7, np.int16)
    result = add(a, b, out)
    np.testing.assert_array_equal(out, np.zeros(full, np.int16))
    np.testing.assert_array_equal(
        result.extra["outputs"]["out"], np.zeros(full, np.int16)
    )
    _record("g6_zero", "add_i16_a_minus_a", result)

    gemm = allo.compile(PAPER_KERNELS["gemm"], target, apu_g2_cost)
    gemm.compiled.repetitions = REPETITIONS
    B = rng.integers(-8, 8, size=(128, 31)).astype(np.int16)
    C = np.full((64, 31), 7, np.int16)
    result = gemm(np.zeros((64, 128), np.int16), B, C)
    np.testing.assert_array_equal(C, np.zeros((64, 31), np.int16))
    _record("g6_zero", "gemm_zero_lhs", result)


def test_fused_and_separate_bicg_plans_agree_on_card(apu_g2_card):
    rng = np.random.default_rng(SEED + 63)
    A = rng.integers(-4, 4, size=(256, 256)).astype(np.int16)
    p = rng.integers(-4, 4, size=256).astype(np.int16)
    r = rng.integers(-4, 4, size=256).astype(np.int16)
    runs = {}
    for layout in (None, "separate/k1"):
        compiled = allo.compile(
            PAPER_KERNELS["bicg"], build_apu_g2_target(), apu_g2_cost, layout=layout
        )
        compiled.compiled.repetitions = REPETITIONS
        q = np.full(256, 7, np.int16)
        s = np.full(256, 7, np.int16)
        result = compiled(A, p, r, q, s)
        runs[compiled.compiled.plan.name] = (q, s, result)
        _record(
            "g6_two_plans",
            compiled.compiled.plan.name,
            result,
            estimate=compiled.compiled.plan.estimate_cycles,
            launches=len(result.extra["launches"]),
        )
    (fq, fs, fused), (sq, ss, separate) = runs["fused/k1"], runs["separate/k1"]
    np.testing.assert_array_equal(fq, A.astype(np.int64) @ p)
    np.testing.assert_array_equal(fs, r.astype(np.int64) @ A)
    np.testing.assert_array_equal(fq, sq)
    np.testing.assert_array_equal(fs, ss)
    assert (len(fused.extra["launches"]), len(separate.extra["launches"])) == (1, 2)


def test_typed_div_ignores_garbage_in_padded_lanes_on_card(apu_g2_card, monkeypatch):
    """A partial shape leaves 262144 - 3000 lanes padded; they are filled
    with random garbage after packing (nonzero divisors), and the logical
    quotients must still be exact."""

    import allo.pim.apu_g2_typed_runtime as runtime

    rng = np.random.default_rng(SEED + 64)
    shape = (3, 1000)
    original = runtime.pack_apu_g2_typed_operands

    def with_garbage(program, operands):
        carriers = []
        for index, carrier in enumerate(original(program, operands)):
            flat = np.array(carrier, copy=True).reshape(-1)
            low = 1 if index == 1 else 0
            garbage = rng.integers(low, 1 << 16, size=flat.size - 3000)
            flat[3000:] = garbage.astype(flat.dtype)
            carriers.append(flat.reshape(carrier.shape))
        return tuple(carriers)

    monkeypatch.setattr(runtime, "pack_apu_g2_typed_operands", with_garbage)
    div = allo.compile(
        APUG2TypedProgram(
            "div", (S(16, False),) * 2, S(16, False), shape, repetitions=REPETITIONS
        ),
        build_apu_g2_target(),
        apu_g2_cost,
    )
    lhs = rng.integers(0, 1 << 16, size=shape).astype(np.uint16)
    rhs = rng.integers(1, 1 << 16, size=shape).astype(np.uint16)
    rhs[0, :8] = 1
    out = np.full(shape, 7, np.uint16)
    result = div(lhs, rhs, out)
    np.testing.assert_array_equal(out, lhs // rhs)
    _record("g6_padding", "div_u16_3x1000_garbage", result)
