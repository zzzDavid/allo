# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gemini-II source route: ordinary @allo contractions -> composed programs.

Compile-only.  The device path is exercised with a software emulation of the
composed host binary (``_EmulatedCard``) that computes the documented VL64
semantics on the packed physical carriers; real-card runs live in
``tests/pim/test_apu_g2_hardware.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import allo
from allo.ir.types import int2, int3, int4, int8, int11, int12, int14, int15, int16, int32
from allo.ir.types import uint4, float16
from allo.pim.apu_g2_composed_contraction import (
    APUG2ComposedContractionProgram,
    APUG2DotEpilogue,
)
from allo.pim.apu_g2_composed_layout import (
    attest_apu_g2_composed_packing,
    interleave_apu_g2_dot_streams,
    pack_apu_g2_batched_gemm_dots,
    pack_apu_g2_composed_operand,
    pack_apu_g2_matrix_vector_dots,
)
from allo.pim.apu_g2_composed_program import build_apu_g2_composed_recipe
from allo.pim.apu_g2_pipeline import (
    APUG2_TYPED_TICK_MEASUREMENTS,
    APUG2_TYPED_TICK_MODEL,
)
from allo.pim.apu_g2_recipe import build_apu_g2_priced_recipe_graph
from allo.pim.apu_g2_source_route import (
    APUG2SourceRouteError,
    apu_g2_source_oracle,
    apu_g2_stream_dots,
)
from allo.pim.apu_g2_typed_program import APUG2ScalarType
from allo.pim.costs.apu_g2 import (
    apu_g2_cost,
    apu_g2_typed_calibration_programs,
    build_apu_g2_cost,
    fit_apu_g2_typed_tick_model,
)
from allo.pim.targets import build_apu_g2_target
from benchmarks.apu_g2.workloads import PAPER_KERNELS

I3 = APUG2ScalarType(3, True)
I4 = APUG2ScalarType(4, True)
I15 = APUG2ScalarType(15, True)

# (artifact case, layout, program and recipe fingerprints) from
# tenon-artifacts/apuv2-composed-v2/provenance/implementations/tenon/cases/
# <case>/layout.json.  The layout fingerprints also equal the Round 2 card runs
# (docs/release/VALIDATION.md, tenon-release-evidence/g2/step4.log).
ARTIFACT = {
    "gemm": (
        "gemm_i4_m64_k128_n31",
        "20a85096d210583720079f6f6475ad85376815a62a830753dadc63baec6b9156",
        "86b7bdd981cf52d25c7147478eace7632b76d5442afc90dc718cd18aec4106c0",
        "4671896b85d3e2974e985543c1835600d631528fae91f2508cb30886c1167afa",
    ),
    "batched_gemm": (
        "batched_gemm_i4_b4_m32_k128_n16",
        "07f71036ff4d8c2d047fc0e6bd64a04c773bef41e8c6fd841bfcfe198d0f6151",
        "b4ea0cdeff6638706b24747f75364a71664e539ee61f96bb030e12e512c6d062",
        "86eb6d861d9d8feed4107aed5fe68242ac91693339b5bd422594a21c68b7d3b8",
    ),
    "gesummv": (
        "gesummv_i2_m1024_k128",
        "b4afa6b998f537edef114ee189e660f3afa4cd80f3a72d76085a89b23b817ff2",
        "295a0ac40c96aeca25d1411d53b10231266f409416399bef804cf9056d40176f",
        "d62b53578cde3a5f0a5b1de17329c0fb8be65576491907eaedd877973cb6307e",
    ),
    "bicg": (
        "bicg_i3_n256",
        "a06509fdcb3496c7649b3eecc5cbb9d25eec2d97289fed5311f53caaab666c7b",
        "fbe0a233eb89e3ee827d657c4411f90309377c840750e1ac2161471914390a42",
        "2b9aa49c4848bf0431840e8c7a8385ce86644b116291073bd8f99d0081165f93",
    ),
    "mvt": (
        "mvt_i3_n256",
        "a06509fdcb3496c7649b3eecc5cbb9d25eec2d97289fed5311f53caaab666c7b",
        "0744614cfd9a9aedc478bfb1b759161193ac6aeda6ea859a51a6a533137a49e0",
        "45971069ac979c132bd1e4f0d3e83873089c529515e52be1d3c8c6c9d3ca34dc",
    ),
}
# SHA-256 of the physical carriers the campaign adapter packed (run
# measured-r00 child-measurement.raw.json "packed_files").
ARTIFACT_PACKED = {
    "gemm": (
        "115182478967444ef3d3116ebc3478f874ce797f113429b374abeeb83a7bb16a",
        "5b28d03d60fa93e7e98187c44f65c3b96cf90a6f93e8744697b1f0a8caaa7bb1",
        None,
    ),
    "batched_gemm": (
        "0d1c54b07e23d97d3444abd61569b58778736ae0e12b762e8b42d0223340b488",
        "170259a39ea73d74c2e612d26aad1c623a038e1a2301f029b8994da71e26c3c5",
        None,
    ),
    "gesummv": (
        "e9a351fa1fb1cd89b7259b0615a848da33588fff67b128c6a2322fec20aba8aa",
        "474e0056ca2bec7d96e2cff3dedf4b3c066030cbf1a7f3d0f916749378f94b29",
        "d1bd994e6114dbebfe4c4b1c9149099ee950c2f424c07feb715792197fc4c408",
    ),
    "bicg": (
        "82288e53d6411844b96a7e4c254d0dfa012edb958a08a642b633870bf27a0db1",
        "28ddae66c1349f124f3b2236fa504eb3209b939f04758247572bef50b5957032",
        None,
    ),
    "mvt": (
        "15c9275e4397f6eb3962b50a053157d7e7373ccc609c8a1112c8c6f6bdf9a2a4",
        "26cff2d1f2ed59bd62a89c611e19da45e7fa3fe432a76295de2496ab7845d25a",
        "cc3fe8042aaad6c937ead0274f4eb202a936d926cabb9dfcf218ce65a7677e54",
    ),
}
ARTIFACT_ROOT = (
    Path(__file__).resolve().parents[4] / "tenon-artifacts" / "apuv2-composed-v2"
)


def _compile(workload, cost=apu_g2_cost, **kwargs):
    return allo.compile(workload, build_apu_g2_target(), cost, **kwargs)


@pytest.fixture(scope="module")
def paper():
    return {name: _compile(fn) for name, fn in PAPER_KERNELS.items()}


def _random_args(compiled, rng, *, carried_bits=None):
    contract = compiled.compiled.contract
    args = {}
    outputs = set(contract.outputs)
    for name in contract.parameters:
        scalar = contract.parameter_type(name)
        shape = contract.parameter_shape(name)
        low, high = scalar.minimum, scalar.maximum
        if name in outputs:
            carried = any(
                s.accumulator == "carried" and s.output == name
                for s in contract.streams
            )
            if carried and carried_bits is not None:
                low, high = -(1 << (carried_bits - 1)), (1 << (carried_bits - 1)) - 1
            elif not carried:
                args[name] = np.full(shape, 7, dtype=np.int16)
                continue
        args[name] = rng.integers(low, high + 1, size=shape).astype(np.int16)
    return args


# --------------------------------------------------------------------------- #
# G2-02: the five paper kernels compile to the artifact's programs
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", sorted(PAPER_KERNELS))
def test_paper_kernel_selects_the_artifact_program(paper, name):
    compiled = paper[name]
    plan = compiled.compiled.plan
    assert compiled.compiled.selection == "cost"
    assert len(plan.launches) == 1
    (launch,) = plan.launches
    _case, layout, program, recipe = ARTIFACT[name]
    assert launch.program.layout_manifest()["fingerprint"] == layout
    assert launch.program.structural_fingerprint == program
    assert build_apu_g2_composed_recipe(launch.program).structural_fingerprint == recipe
    assert plan.estimate_validity == "calibrated"
    # The selected plan is the cheapest legal one; the others are kept.
    estimates = [item["estimate_cycles"] for item in compiled.placement_ranking]
    assert estimates == sorted(estimates)
    assert compiled.placement_ranking[0]["plan"] == plan.name


def test_paper_kernel_derivations(paper):
    forms = {
        name: [s.packing_form for s in compiled.compiled.contract.streams]
        for name, compiled in paper.items()
    }
    assert forms == {
        "gemm": ["pack_apu_g2_batched_gemm_dots"],
        "batched_gemm": ["pack_apu_g2_batched_gemm_dots"],
        "gesummv": ["pack_apu_g2_matrix_vector_dots(transpose=False)"] * 2,
        "bicg": [
            "pack_apu_g2_matrix_vector_dots(transpose=False)",
            "pack_apu_g2_matrix_vector_dots(transpose=True)",
        ],
        "mvt": [
            "pack_apu_g2_matrix_vector_dots(transpose=False)",
            "pack_apu_g2_matrix_vector_dots(transpose=True)",
        ],
    }
    gesummv = paper["gesummv"].compiled
    assert gesummv.contract.coefficients == (3, 2)
    assert gesummv.plan.name == "fused/k1"
    assert [p["plan"] for p in gesummv.placement_ranking] == ["fused/k1"]
    bicg = paper["bicg"].compiled
    assert [s.output for s in bicg.contract.streams] == ["q", "s"]
    assert {p["plan"] for p in bicg.placement_ranking} == {
        "fused/k1",
        "fused/k2",
        "separate/k1",
        "separate/k2",
    }
    mvt = paper["mvt"].compiled
    assert [s.accumulator for s in mvt.contract.streams] == ["carried"] * 2
    (launch,) = mvt.plan.launches
    assert launch.program.epilogue.accumulator_type == APUG2ScalarType(14, True)
    assert launch.program.output_shape == (256, 2)


def _helper_dots(name, args):
    """The artifact adapter's packing, from the public host helpers."""

    if name in ("gemm", "batched_gemm"):
        return pack_apu_g2_batched_gemm_dots(args["A"], args["B"])
    if name == "gesummv":
        return interleave_apu_g2_dot_streams(
            pack_apu_g2_matrix_vector_dots(args["A"], args["x"]),
            pack_apu_g2_matrix_vector_dots(args["B"], args["x"]),
        )
    first = ("p", "r") if name == "bicg" else ("y1", "y2")
    return interleave_apu_g2_dot_streams(
        pack_apu_g2_matrix_vector_dots(args["A"], args[first[0]]),
        pack_apu_g2_matrix_vector_dots(args["A"], args[first[1]], transpose=True),
    )


@pytest.mark.parametrize("name", sorted(PAPER_KERNELS))
def test_paper_kernel_packing_equals_host_helpers(paper, name):
    compiled = paper[name]
    rng = np.random.default_rng(11)
    args = _random_args(compiled, rng, carried_bits=14)
    (packed,) = compiled.compiled.pack(**args)
    (launch,) = compiled.compiled.plan.launches
    lhs, rhs = _helper_dots(name, args)
    program = launch.program
    lhs = lhs.astype(program.input_types[0].numpy_dtype)
    rhs = rhs.astype(program.input_types[1].numpy_dtype)
    np.testing.assert_array_equal(packed["lhs"], pack_apu_g2_composed_operand(program, lhs, 0))
    np.testing.assert_array_equal(packed["rhs"], pack_apu_g2_composed_operand(program, rhs, 1))
    attest_apu_g2_composed_packing(
        program,
        lhs,
        rhs,
        packed["lhs"],
        packed["rhs"],
        packed["auxiliary"],
        accumulator=(
            None
            if name != "mvt"
            else np.stack((args["x1"], args["x2"]), axis=1).astype(np.int16)
        ),
    )


def _artifact_fixture(case):
    manifest = json.loads((ARTIFACT_ROOT / "inputs" / case / "manifest.json").read_text())
    arrays = {}
    for record in manifest["arrays"]:
        if record["role"] != "input":
            continue
        bits = int(record["logical_type"][3:])
        raw = np.fromfile(
            ARTIFACT_ROOT / "inputs" / case / record["file"], dtype="<u2"
        ).astype(np.int64) & ((1 << bits) - 1)
        values = np.where(raw >= 1 << (bits - 1), raw - (1 << bits), raw)
        arrays[record["name"]] = values.reshape(record["shape"])
    return arrays


@pytest.mark.parametrize("name", sorted(PAPER_KERNELS))
def test_paper_kernel_packing_is_byte_identical_to_the_campaign(paper, name):
    case = ARTIFACT[name][0]
    if not (ARTIFACT_ROOT / "inputs" / case).is_dir():
        pytest.skip(f"artifact {ARTIFACT_ROOT} is not present")
    compiled = paper[name]
    contract = compiled.compiled.contract
    fixture = {key.lower(): value for key, value in _artifact_fixture(case).items()}
    args = {
        parameter: (
            fixture[parameter.lower()].astype(np.int16)
            if parameter.lower() in fixture
            else np.zeros(contract.parameter_shape(parameter), dtype=np.int16)
        )
        for parameter in contract.parameters
    }
    (packed,) = compiled.compiled.pack(**args)
    digests = tuple(
        None if packed[key] is None else hashlib.sha256(packed[key].tobytes()).hexdigest()
        for key in ("lhs", "rhs", "auxiliary")
    )
    assert digests == ARTIFACT_PACKED[name]
    arrays, initial = compiled.compiled._validated_inputs(args)
    oracle = apu_g2_source_oracle(contract, arrays, initial)
    serialized = np.concatenate(
        [oracle[output].reshape(-1) for output in contract.outputs]
    ).astype("<i2")
    archived = np.fromfile(ARTIFACT_ROOT / "inputs" / case / "oracle.i16le.bin", dtype="<i2")
    np.testing.assert_array_equal(serialized, archived)


# --------------------------------------------------------------------------- #
# Generalization: renamed, reordered and nearby programs; finite rejections
# --------------------------------------------------------------------------- #


def renamed_gesummv(
    Left: int2[1024, 128], Right: int2[1024, 128], v: int2[128], out: int15[1024]
):
    for row in range(1024):
        first: int11 = 0
        second: int11 = 0
        for col in range(128):
            first += Left[row, col] * v[col]
            second += Right[row, col] * v[col]
        out[row] = 3 * first + 2 * second


def gemm_accumulate_in_place(A: int4[64, 128], B: int4[128, 31], C: int15[64, 31]):
    for i in range(64):
        for j in range(31):
            C[i, j] = 0
    for i in range(64):
        for j in range(31):
            for k in range(128):
                C[i, j] += A[i, k] * B[k, j]


def gemm_n33(A: int4[64, 128], B: int4[128, 33], C: int15[64, 33]):
    for i in range(64):
        for j in range(33):
            acc: int15 = 0
            for k in range(128):
                acc += A[i, k] * B[k, j]
            C[i, j] = acc


def gemm_n32(A: int4[64, 128], B: int4[128, 32], C: int15[64, 32]):
    for i in range(64):
        for j in range(32):
            acc: int15 = 0
            for k in range(128):
                acc += A[i, k] * B[k, j]
            C[i, j] = acc


def gesummv_m512(A: int2[512, 128], B: int2[512, 128], x: int2[128], y: int15[512]):
    for i in range(512):
        t: int11 = 0
        u: int11 = 0
        for k in range(128):
            t += A[i, k] * x[k]
            u += B[i, k] * x[k]
        y[i] = 3 * t + 2 * u


def bicg_n512(A: int3[512, 512], p: int3[512], r: int3[512], q: int15[512], s: int15[512]):
    for i in range(512):
        s[i] = 0
    for i in range(512):
        q[i] = 0
        for j in range(512):
            q[i] += A[i, j] * p[j]
            s[j] += r[i] * A[i, j]


def gemm_wide_containers(A: int4[64, 128], B: int4[128, 31], C: int32[64, 31]):
    for i in range(64):
        for j in range(31):
            acc: int32 = 0
            for k in range(128):
                acc += A[i, k] * B[k, j]
            C[i, j] = acc


def gemm_transposed_lhs(A: int4[128, 64], B: int4[128, 32], C: int15[64, 32]):
    for i in range(64):
        for j in range(32):
            acc: int16 = 0
            for k in range(128):
                acc += A[k, i] * B[k, j]
            C[i, j] = acc


def test_rename_only_variant_compiles_to_the_same_program(paper):
    renamed = _compile(renamed_gesummv)
    (launch,) = renamed.compiled.plan.launches
    (original,) = paper["gesummv"].compiled.plan.launches
    assert launch.program.structural_fingerprint == original.program.structural_fingerprint
    assert launch.layout_fingerprint == original.layout_fingerprint
    in_place = _compile(gemm_accumulate_in_place)
    (launch,) = in_place.compiled.plan.launches
    assert launch.program.structural_fingerprint == ARTIFACT["gemm"][2]
    wide = _compile(gemm_wide_containers)
    (launch,) = wide.compiled.plan.launches
    assert launch.program.structural_fingerprint == ARTIFACT["gemm"][2]


def test_nearby_shapes_compile_and_tile_beyond_one_carrier():
    exact = _compile(gemm_n32)
    assert [len(p.launches) for p in exact.compiled.ranking if p.name == "single/k1"] == [1]
    beyond = _compile(gemm_n33)
    plans = {plan.name: plan for plan in beyond.compiled.ranking}
    # 2112 dots exceed the 2048-dot K=128 carrier: output tiles 2048 + 64 ...
    assert [l.coordinates for l in plans["single/k1"].launches] == [(0, 2048), (2048, 2112)]
    # ... or one 4096-dot K=64 carrier per reduction chunk, chained exactly.
    assert [l.accumulator for l in plans["single/k2"].launches] == ["none", "previous"]
    assert beyond.compiled.plan.name == "single/k2"
    assert beyond.compiled.plan.estimate_validity == "extrapolated"
    m512 = _compile(gesummv_m512)
    assert m512.compiled.plan.launches[0].program.output_shape == (512,)
    k512 = _compile(bicg_n512)
    assert k512.compiled.plan.name == "fused/k2"
    assert [l.program.reduction_extent for l in k512.compiled.plan.launches] == [256, 256]
    transposed = _compile(gemm_transposed_lhs)
    assert transposed.compiled.contract.streams[0].packing_form == "broadcast_gather"


def k96(A: int4[64, 96], x: int4[96], y: int15[64]):
    for i in range(64):
        acc: int15 = 0
        for k in range(96):
            acc += A[i, k] * x[k]
        y[i] = acc


def mixed_signedness(A: int4[64, 128], x: uint4[128], y: int16[64]):
    for i in range(64):
        acc: int16 = 0
        for k in range(128):
            acc += A[i, k] * x[k]
        y[i] = acc


def three_products(
    A: int4[64, 128], x: int4[128], y0: int15[64], y1: int15[64], y2: int15[64]
):
    for i in range(64):
        a: int15 = 0
        b: int15 = 0
        c: int15 = 0
        for k in range(128):
            a += A[i, k] * x[k]
            b += A[i, k] * x[k]
            c += A[i, k] * x[k]
        y0[i] = a
        y1[i] = b
        y2[i] = c


def narrow_accumulator(A: int4[64, 128], x: int4[128], y: int15[64]):
    for i in range(64):
        acc: int12 = 0
        for k in range(128):
            acc += A[i, k] * x[k]
        y[i] = acc


def float_gemv(A: float16[64, 128], x: float16[128], y: float16[64]):
    for i in range(64):
        acc: float16 = 0
        for k in range(128):
            acc += A[i, k] * x[k]
        y[i] = acc


def mixed_combined_types(A: int2[64, 128], B: int3[64, 128], x: int2[128], y: int16[64]):
    for i in range(64):
        t: int12 = 0
        u: int12 = 0
        for k in range(128):
            t += A[i, k] * x[k]
            u += B[i, k] * x[k]
        y[i] = 3 * t + 2 * u


def k1024(A: int2[16, 1024], x: int2[1024], y: int16[16]):
    for i in range(16):
        acc: int16 = 0
        for k in range(1024):
            acc += A[i, k] * x[k]
        y[i] = acc


def offset_access(A: int4[65, 128], x: int4[128], y: int15[64]):
    for i in range(64):
        acc: int15 = 0
        for k in range(128):
            acc += A[i + 1, k] * x[k]
        y[i] = acc


def extra_effect(A: int4[64, 128], x: int4[128], y: int15[64], z: int8[64]):
    for i in range(64):
        acc: int15 = 0
        for k in range(128):
            acc += A[i, k] * x[k]
        y[i] = acc
        z[i] = A[i, 0] - x[0]


@pytest.mark.parametrize(
    "workload, reason",
    [
        (k96, "power of two"),
        (mixed_signedness, "mix signed and unsigned"),
        (three_products, "at most two products"),
        # the matcher refuses the widening copy of an int12 accumulator
        # (its range exceeds int12 but the device computes an exact int15)
        (narrow_accumulator, "copies a i12 result into a i15 buffer"),
        (float_gemv, "not an integer buffer"),
        (mixed_combined_types, "mixed operand types"),
        (k1024, "at most 256"),
        (offset_access, "plain loop variable"),
        (extra_effect, "outside the contraction family"),
    ],
)
def test_unsupported_forms_raise_a_finite_error(workload, reason):
    with pytest.raises(APUG2SourceRouteError, match=reason) as error:
        _compile(workload)
    assert isinstance(error.value, NotImplementedError)
    assert "APUG2TypedProgram" in str(error.value)


def test_layout_forces_a_named_plan_and_rejects_unknown_names():
    forced = _compile(PAPER_KERNELS["bicg"], layout="separate/k1")
    assert forced.compiled.selection == "forced"
    assert len(forced.compiled.plan.launches) == 2
    with pytest.raises(ValueError, match="names no legal plan"):
        _compile(PAPER_KERNELS["bicg"], layout="fused/k3")
    with pytest.raises(ValueError, match="host_moves"):
        _compile(PAPER_KERNELS["bicg"], host_moves=[])


def test_composed_record_beyond_one_carrier_names_the_capacity():
    with pytest.raises(ValueError, match="exceeds capacity 2048"):
        APUG2ComposedContractionProgram((I4, I4), I15, (64, 33), 128)


# --------------------------------------------------------------------------- #
# G2-03: calibrated cost, virtual kind, perturbation
# --------------------------------------------------------------------------- #


def test_tick_model_refits_from_the_device_measurements():
    coefficients, report = fit_apu_g2_typed_tick_model()
    stored = APUG2_TYPED_TICK_MODEL.coefficient_map
    assert {k: round(v, 3) for k, v in coefficients.items()} == stored
    for name, row in report.items():
        assert abs(row["in_sample_relative_error"]) <= 0.02, name
        if name.startswith("composed_"):
            assert abs(row["leave_one_out_relative_error"]) <= 0.03, name


def test_typed_and_composed_estimates_are_calibrated_device_ticks():
    target = build_apu_g2_target()
    bound = apu_g2_cost.bind(target)
    measured = {m.program: m.ticks_per_pipeline for m in APUG2_TYPED_TICK_MEASUREMENTS}
    for name, recipe in apu_g2_typed_calibration_programs().items():
        graph = build_apu_g2_priced_recipe_graph(recipe, target, bound)
        estimate = bound.evaluate(graph).cycles
        # <= 2% fit error plus at most half a tick of rounding per issued call.
        assert abs(estimate / measured[name] - 1) <= 0.025, name
        assert graph.metadata["latency_unit"] == "device_ticks_per_pipeline_body"
        assert graph.metadata["estimate_validity"] == "calibrated"


def test_virtual_run_is_labelled_and_leaves_outputs(paper):
    compiled = _compile(PAPER_KERNELS["gemm"], backend="virtual")
    args = _random_args(compiled, np.random.default_rng(1))
    before = args["C"].copy()
    result = compiled(**args)
    assert result.kind == "virtual"
    assert result.backend == "virtual"
    assert result.cycles == compiled.compiled.plan.estimate_cycles == compiled.estimate().cycles
    assert result.extra["cost_model"] == "apu_g2_cost"
    assert result.extra["estimate_validity"] == "calibrated"
    np.testing.assert_array_equal(args["C"], before)


def test_estimates_depend_on_shape():
    def estimate(workload):
        return _compile(workload).compiled.plan.estimate_cycles

    def gemm_k64(A: int4[64, 64], B: int4[64, 31], C: int15[64, 31]):
        for i in range(64):
            for j in range(31):
                acc: int15 = 0
                for k in range(64):
                    acc += A[i, k] * B[k, j]
                C[i, j] = acc

    def gemm_i2(A: int2[64, 128], B: int2[128, 31], C: int15[64, 31]):
        for i in range(64):
            for j in range(31):
                acc: int15 = 0
                for k in range(128):
                    acc += A[i, k] * B[k, j]
                C[i, j] = acc

    base = estimate(PAPER_KERNELS["gemm"])
    assert estimate(gemm_k64) < base  # shallower reduction
    assert estimate(gemm_i2) < base  # narrower operands and sums
    # Within one carrier every VL64 call covers all 262,144 lanes, so the dot
    # count does not change device time; beyond it, launches add up.
    assert estimate(gemm_n32) == base
    two_tiles = _compile(gemm_n33).compiled.ranking
    assert {p.name: p.estimate_cycles for p in two_tiles}["single/k1"] == 2 * base


def test_cost_perturbation_changes_estimate_and_selection():
    perturbed = build_apu_g2_cost(
        APUG2_TYPED_TICK_MODEL.scaled(row_bits=20.0), name="apu_g2_cost_rows_x20"
    )
    target = build_apu_g2_target()
    assert perturbed.bind(target).fingerprint != apu_g2_cost.bind(target).fingerprint
    base = _compile(PAPER_KERNELS["gemm"])
    moved = _compile(PAPER_KERNELS["gemm"], cost=perturbed)
    assert moved.compiled.plan.estimate_cycles > base.compiled.plan.estimate_cycles
    # Same program either way: legality and lowering ignore the cost.
    assert (
        moved.compiled.plan.launches[0].program.structural_fingerprint
        == base.compiled.plan.launches[0].program.structural_fingerprint
    )
    # 64x128x33: the reduction split wins under the calibrated model; making
    # row sweeps (which the split's accumulate epilogue adds) 20x dearer flips
    # the selection to output tiling.
    assert _compile(gemm_n33).compiled.plan.name == "single/k2"
    assert _compile(gemm_n33, cost=perturbed).compiled.plan.name == "single/k1"


# --------------------------------------------------------------------------- #
# Device path with an emulated card (no hardware)
# --------------------------------------------------------------------------- #


def _storage(bits, signed):
    return APUG2ScalarType(bits, signed).numpy_dtype


class _EmulatedCard:
    """Software stand-in for cmake + the composed host binary.

    Executes the documented VL64 pipeline on the physical carriers: lane-wise
    product, sum over aligned 2**log_reduction column blocks into the block
    head, then the epilogue (pair-affine: odd MMB set scaled and added into
    the even set; accumulate: add the auxiliary carrier).  ``corrupt`` adds
    one to every head to model a wrong device.
    """

    def __init__(self, corrupt=False):
        self.corrupt = corrupt
        self.launches = []

    def run(self, command, *, cwd, timeout, label):
        command = [str(part) for part in command]
        if command[0] == "cmake":
            if "--build" in command:
                build = Path(command[command.index("--build") + 1])
                if build.name == "device":
                    (build / "bin").mkdir(parents=True, exist_ok=True)
                    (build / "bin" / "tenon_apu_g2_tasks.update.bin").write_bytes(b"arc")
                else:
                    build.mkdir(parents=True, exist_ok=True)
                    (build / "tenon_apu_g2_composed_dot").write_bytes(b"host")
            return SimpleNamespace(stdout="", stderr="", returncode=0)
        return self._execute(command)

    def _execute(self, argv):
        (host, lhs, rhs, aux, out, lb, ls, rb, rs, db, ab, asg, ob, osg, log, epi, reps, fp) = argv
        lhs_v = np.fromfile(lhs, dtype=_storage(int(lb), ls == "1")).astype(np.int64).reshape(4, 65536)
        rhs_v = np.fromfile(rhs, dtype=_storage(int(rb), rs == "1")).astype(np.int64).reshape(4, 65536)
        block = 1 << int(log)
        dots = (lhs_v * rhs_v).reshape(4, -1, block).sum(axis=2)
        heads = np.zeros((4, 65536), dtype=np.int64)
        heads[:, ::block] = dots
        if epi != "0":
            aux_v = np.fromfile(aux, dtype=_storage(int(ab), asg == "1")).astype(np.int64).reshape(4, 65536)
            if epi == "1":
                scaled = heads * aux_v
                heads = scaled.copy()
                heads[0::2] = scaled[0::2] + scaled[1::2]
            else:
                heads = heads + aux_v
        if self.corrupt:
            heads[:, ::block] += 1
        heads.astype(_storage(int(ob), osg == "1")).tofile(out)
        self.launches.append(argv)
        device = Path(host).parent.parent / "device" / "bin" / "tenon_apu_g2_tasks.update.bin"
        text = "\n".join(
            [
                f"device_pipeline_ticks={1000 * int(reps)}",
                "device_final_pipeline_ticks=1000",
                "h2d_us=1.0",
                "host_task_us=1.0",
                "d2h_us=1.0",
                "end_to_end_us=3.0",
                "target=hardware",
                f"device_library={device}",
                "task_status=0",
                "timed_scope=composed_dot_vl64_pipeline",
                "completion_barrier_included=1",
                "independent_final_correctness_call=1",
                f"layout_fingerprint={fp}",
                f"epilogue_mode={epi}",
                f"lhs_bits={lb}",
                f"lhs_signed={ls}",
                f"rhs_bits={rb}",
                f"rhs_signed={rs}",
                f"dot_bits={db}",
                f"auxiliary_bits={ab}",
                f"auxiliary_signed={asg}",
                f"out_bits={ob}",
                f"out_signed={osg}",
                f"log_reduction={log}",
                f"repetitions={reps}",
                "PASS physical_outputs=262144",
            ]
        )
        return SimpleNamespace(stdout=text + "\n", stderr="", returncode=0)


@pytest.fixture
def emulated_card(monkeypatch, tmp_path):
    card = _EmulatedCard()
    module = "allo.pim.apu_g2_composed_runtime"
    monkeypatch.setattr(f"{module}._require_hardware_stack", lambda: "emulated card")
    monkeypatch.setattr(f"{module}._run_command", card.run)
    monkeypatch.setenv("TENON_APU_G2_LOCK", str(tmp_path / "card.lock"))
    return card


def _reference(name, args):
    """Plain NumPy semantics of the five kernels (independent of the route)."""

    a = {key: value.astype(np.int64) for key, value in args.items()}
    if name in ("gemm", "batched_gemm", "gemm_n33"):
        return {"C": a["A"] @ a["B"]}
    if name == "gesummv":
        return {"y": 3 * (a["A"] @ a["x"]) + 2 * (a["B"] @ a["x"])}
    if name == "bicg":
        return {"q": a["A"] @ a["p"], "s": a["r"] @ a["A"]}
    return {"x1": a["x1"] + a["A"] @ a["y1"], "x2": a["x2"] + a["A"].T @ a["y2"]}


@pytest.mark.parametrize("name", sorted(PAPER_KERNELS) + ["gemm_n33"])
def test_emulated_device_run_writes_exact_outputs(emulated_card, name):
    workload = gemm_n33 if name == "gemm_n33" else PAPER_KERNELS[name]
    compiled = _compile(workload)
    compiled.compiled.repetitions = 20
    args = _random_args(compiled, np.random.default_rng(5), carried_bits=14)
    expected = _reference(name, args)
    result = compiled(**args)
    assert result.kind == "device"
    for output, value in expected.items():
        np.testing.assert_array_equal(args[output], value)
    launches = result.extra["launches"]
    assert len(launches) == len(compiled.compiled.plan.launches)
    assert result.cycles == 1000 * len(launches)
    for record, launch in zip(launches, compiled.compiled.plan.launches):
        assert record["layout_fingerprint"] == launch.layout_fingerprint
        assert record["packing_attestation"]["layout_fingerprint"] == launch.layout_fingerprint
    assert len(emulated_card.launches) == len(launches)
    assert all(argv[16] == "20" for argv in emulated_card.launches)


def test_emulated_wrong_device_result_is_a_mismatch(monkeypatch, emulated_card):
    emulated_card.corrupt = True
    compiled = _compile(PAPER_KERNELS["gemm"])
    args = _random_args(compiled, np.random.default_rng(6))
    with pytest.raises(RuntimeError, match="differs from NumPy"):
        compiled(**args)


def test_emulated_wrong_oracle_is_reported(monkeypatch, emulated_card):
    import allo.pim.apu_g2_composed_runtime as runtime

    original = runtime.expected_apu_g2_composed_output
    monkeypatch.setattr(
        runtime,
        "expected_apu_g2_composed_output",
        lambda *args, **kwargs: original(*args, **kwargs) + 1,
    )
    compiled = _compile(PAPER_KERNELS["gesummv"])
    args = _random_args(compiled, np.random.default_rng(7))
    with pytest.raises(RuntimeError, match="differs from NumPy"):
        compiled(**args)


def test_fingerprint_mismatch_is_refused_before_launch(emulated_card):
    import allo.pim.apu_g2_composed_runtime as runtime

    compiled = _compile(PAPER_KERNELS["gemm"])
    (launch,) = compiled.compiled.plan.launches
    lhs = np.zeros(launch.program.dot_shape, dtype=np.int8)
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        runtime.run_apu_g2_composed(
            launch.program, lhs, lhs, expected_layout_fingerprint="0" * 64
        )
    assert emulated_card.launches == []


def test_tampered_packing_is_refused_before_launch(monkeypatch, emulated_card):
    import allo.pim.apu_g2_composed_runtime as runtime

    original = runtime.pack_apu_g2_composed_operand

    def shifted(program, value, index):
        return np.roll(original(program, value, index), 1, axis=1)

    monkeypatch.setattr(runtime, "pack_apu_g2_composed_operand", shifted)
    compiled = _compile(PAPER_KERNELS["gemm"])
    args = _random_args(compiled, np.random.default_rng(8))
    with pytest.raises(RuntimeError, match="lanes of the program layout"):
        compiled(**args)
    assert emulated_card.launches == []


def test_carried_accumulator_outside_device_width_is_rejected(emulated_card):
    compiled = _compile(PAPER_KERNELS["mvt"])
    args = _random_args(compiled, np.random.default_rng(9), carried_bits=15)
    args["x1"][0] = 1 << 13
    with pytest.raises(ValueError, match="carried accumulator values must fit int14"):
        compiled(**args)


def test_device_lock_is_exclusive_and_reentrant(monkeypatch, tmp_path):
    import fcntl
    import os

    from allo.pim.apu_g2_runtime import apu_g2_device_lock, apu_g2_lock_path

    monkeypatch.delenv("TENON_APU_G2_LOCK", raising=False)
    monkeypatch.setenv("TENON_APU_G2_DEVICE", "apu-00")
    assert apu_g2_lock_path() == Path("/tmp/tenon-apu-g2-apu-00.lock")
    path = tmp_path / "g2.lock"
    monkeypatch.setenv("TENON_APU_G2_LOCK", str(path))
    with apu_g2_device_lock() as held:
        assert held == path
        with apu_g2_device_lock():
            pass
        fd = os.open(path, os.O_RDONLY)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)
    fd = os.open(path, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(fd)


def test_cmake_arguments_follow_the_resolved_sdk_and_device(monkeypatch):
    from allo.pim.apu_g2_runtime import _require_hardware_stack, apu_g2_cmake_arguments

    monkeypatch.setenv("TENON_APU_G2_SDK_ROOT", "/srv/gsi-g2")
    monkeypatch.setenv("TENON_APU_G2_DEVICE", "apu-00")
    assert apu_g2_cmake_arguments() == [
        "-DTENON_APU_G2_SDK_ROOT=/srv/gsi-g2",
        "-DTENON_APU_G2_DEVICE_INDEX=0",
    ]
    monkeypatch.setenv("TENON_APU_G2_DEVICE", "apu-01")
    with pytest.raises(RuntimeError, match="only apu-00 has been validated"):
        _require_hardware_stack()
    monkeypatch.setenv("TENON_APU_G2_DEVICE", "card0")
    with pytest.raises(RuntimeError, match="apu-NN"):
        apu_g2_cmake_arguments()


def test_templates_take_the_sdk_root_and_device_index():
    from allo.pim.apu_g2_runtime import _TEMPLATE

    for relative in ("CMakeLists.txt", "device/CMakeLists.txt"):
        text = (_TEMPLATE / relative).read_text()
        assert text.count("/opt/gsi") == 1  # the cache default only
        assert "${TENON_APU_G2_SDK_ROOT}" in text
    for host in _TEMPLATE.glob("host*.cc"):
        assert "DeviceConfig{TENON_APU_G2_DEVICE_INDEX, 0}" in host.read_text()
