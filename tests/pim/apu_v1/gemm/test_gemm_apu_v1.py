# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical hybrid GEMM plus the scalar-mapping GVML integration test."""

import importlib.util

import numpy as np
import pytest

import allo
from lib import apu_v1, cell
from lib.cost import bind_cost
from lib.shapes import shape
from lib.targets import build_target

_RUN_CMD = (
    "python -m pytest tests/pim/apu_v1/gemm/test_gemm_apu_v1.py"
    "::test_gemm_apu_v1_scalar_baseline "
    "-p no:cacheprovider -q"
)


def _assert_hybrid_manifest(compiled):
    phase = compiled.hybrid_manifest.phases[0]
    assert [(region.function, region.kind) for region in phase.regions] == [
        ("mm1", "vector"),
        ("ele_add", "scalar"),
    ]
    assert phase.regions[1].dependencies == (phase.regions[0].id,)
    assert phase.barriers[0].values == ("out_AB",)
    assert phase.conversions == ()
    assert phase.regions[0].compute_analysis.numeric_type == "ui16"
    assert phase.regions[0].selected_plan is not None


def test_gemm_apu_v1_hybrid_manifest(request):
    workload = cell.load_workload(request.path.parent, "gemm")
    target = build_target("apu_v1")
    compiled = allo.compile(
        workload.build(), target, bind_cost(target), backend="virtual"
    )
    _assert_hybrid_manifest(compiled)


def test_gemm_apu_v1_hybrid_functional_uint16(request):
    workload = cell.load_workload(request.path.parent, "gemm")
    target = build_target("apu_v1")
    compiled = allo.compile(
        workload.build(), target, bind_cost(target), backend="functional"
    )
    _assert_hybrid_manifest(compiled)
    case = apu_v1.get_case("gemm")
    inputs = case.make_inputs()
    canonical = case.run_reference(inputs)["output"]
    run = compiled(**inputs)
    observed = run.extra["outputs"]["output"]
    product = (inputs["A"].astype(np.uint64) @ inputs["B"].astype(np.uint64)).astype(
        np.uint16
    )
    staged = (
        product.astype(np.uint64)
        + np.uint64(case.scalars["beta"]) * inputs["C"].astype(np.uint64)
    ).astype(np.uint16)
    np.testing.assert_array_equal(observed, staged)
    np.testing.assert_array_equal(observed, canonical)
    assert run.extra["host_intermediate_round_trips"] == 0


@pytest.mark.apu_v1_device
def test_gemm_apu_v1_hybrid_device(request, apu_v1_device_gate):
    """Four shard-local GVML tasks feed one retained ARC epilogue."""

    workload = cell.load_workload(request.path.parent, "gemm")
    target = build_target("apu_v1")
    compiled = allo.compile(workload.build(), target, bind_cost(target))
    _assert_hybrid_manifest(compiled)
    case = apu_v1.get_case("gemm")
    inputs = case.make_inputs()
    canonical = case.run_reference(inputs)["output"]
    product = (inputs["A"].astype(np.uint64) @ inputs["B"].astype(np.uint64)).astype(
        np.uint16
    )
    staged = (
        product.astype(np.uint64)
        + np.uint64(case.scalars["beta"]) * inputs["C"].astype(np.uint64)
    ).astype(np.uint16)

    import conftest

    with conftest.board_lock():
        result = compiled(**inputs)
    observed = result.extra["outputs"]["output"]
    np.testing.assert_array_equal(observed, staged)
    np.testing.assert_array_equal(observed, canonical)
    assert result.cycles is not None and result.cycles > 0
    assert result.extra["host_intermediate_round_trips"] == 0
    assert result.extra["persistent_l4"] is True
    assert result.extra["vector_apucs"] == (0, 1, 2, 3)
    assert result.extra["scalar_apucs"] == (0,)


@pytest.mark.apu_v1_device
def test_gemm_apu_v1_scalar_baseline(request, apu_v1_device_gate):
    target = build_target("apu_v1")
    compiled = allo.compile(apu_v1.build_program("gemm"), target, bind_cost(target))
    result, verdict, _record = apu_v1.run_compiled(
        compiled, "gemm", folder=request.path.parent, run_cmd=_RUN_CMD
    )
    apu_v1.assert_result(result, verdict)


@pytest.mark.apu_v1_device
def test_gemm_scalar_mapping_eight_groups(request, apu_v1_device_gate):
    path = request.path.parent / "group_workload.py"
    spec = importlib.util.spec_from_file_location("apu_v1_gemm_group_workload", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = build_target("apu_v1")
    compiled = allo.compile(module.build(), target, bind_cost(target))

    placement = compiled.compiled.layout
    assert placement.extra["groups_per_vr"] == 8
    assert placement.extra["group_size"] == 4096

    dims = shape("gemm")
    rng = np.random.default_rng(7)
    A = (rng.standard_normal((dims["P"], dims["Q"])) * 0.05).astype(np.float16)
    B = (rng.standard_normal((dims["Q"], dims["R"])) * 0.05).astype(np.float16)
    C = np.zeros((dims["P"], dims["R"]), dtype=np.float16)
    expected = A.astype(np.float32) @ B.astype(np.float32)

    import conftest

    with conftest.board_lock():
        result = compiled(A, B, C)
    np.testing.assert_allclose(C, expected, rtol=2e-2, atol=2e-2)
    assert result.cycles is not None and result.cycles > 0
