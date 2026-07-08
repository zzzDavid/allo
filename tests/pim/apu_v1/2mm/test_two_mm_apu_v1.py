# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical 2mm through two GVML contractions and one ARC epilogue."""

import numpy as np

import pytest

import allo
from lib import apu_v1, cell
from lib.cost import bind_cost
from lib.targets import build_target

_KERNEL = "2mm"
_RUN_CMD = (
    "python -m pytest tests/pim/apu_v1/2mm/test_two_mm_apu_v1.py"
    "::test_two_mm_apu_v1_scalar_baseline "
    "-p no:cacheprovider -q"
)


def _assert_hybrid_manifest(compiled):
    phase = compiled.hybrid_manifest.phases[0]
    assert [(region.function, region.kind) for region in phase.regions] == [
        ("mm1", "vector"),
        ("mm2", "vector"),
        ("ele_add", "scalar"),
    ]
    assert phase.regions[1].dependencies == (phase.regions[0].id,)
    assert phase.regions[2].dependencies == (phase.regions[1].id,)
    assert [barrier.values for barrier in phase.barriers] == [
        ("out_AB",),
        ("out_ABC",),
    ]
    assert all(region.selected_plan is not None for region in phase.regions[:2])
    assert phase.regions[2].selected_plan is None
    assert phase.conversions == ()
    assert all(
        region.compute_analysis.numeric_type == "ui16" for region in phase.regions[:2]
    )


def test_two_mm_apu_v1_hybrid_manifest(request):
    workload = cell.load_workload(request.path.parent, _KERNEL)
    target = build_target("apu_v1")
    compiled = allo.compile(
        workload.build(), target, bind_cost(target), backend="virtual"
    )
    _assert_hybrid_manifest(compiled)


def test_two_mm_apu_v1_hybrid_functional_uint16(request):
    workload = cell.load_workload(request.path.parent, _KERNEL)
    target = build_target("apu_v1")
    compiled = allo.compile(
        workload.build(), target, bind_cost(target), backend="functional"
    )
    _assert_hybrid_manifest(compiled)
    case = apu_v1.get_case(_KERNEL)
    inputs = case.make_inputs()
    canonical = case.run_reference(inputs)["output"]
    run = compiled(**inputs)
    observed = run.extra["outputs"]["output"]
    ab = (inputs["A"].astype(np.uint64) @ inputs["B"].astype(np.uint64)).astype(
        np.uint16
    )
    abc = (ab.astype(np.uint64) @ inputs["C"].astype(np.uint64)).astype(np.uint16)
    staged = (
        np.uint64(case.scalars["beta"]) * abc.astype(np.uint64)
        + np.uint64(case.scalars["alpha"]) * inputs["D"].astype(np.uint64)
    ).astype(np.uint16)
    np.testing.assert_array_equal(observed, staged)
    np.testing.assert_array_equal(observed, canonical)
    assert run.extra["host_intermediate_round_trips"] == 0


@pytest.mark.apu_v1_device
def test_two_mm_apu_v1_hybrid_device(request, apu_v1_device_gate):
    workload = cell.load_workload(request.path.parent, _KERNEL)
    target = build_target("apu_v1")
    compiled = allo.compile(workload.build(), target, bind_cost(target))
    _assert_hybrid_manifest(compiled)
    case = apu_v1.get_case(_KERNEL)
    inputs = case.make_inputs()
    canonical = case.run_reference(inputs)["output"]
    ab = (inputs["A"].astype(np.uint64) @ inputs["B"].astype(np.uint64)).astype(
        np.uint16
    )
    abc = (ab.astype(np.uint64) @ inputs["C"].astype(np.uint64)).astype(np.uint16)
    staged = (
        np.uint64(case.scalars["beta"]) * abc.astype(np.uint64)
        + np.uint64(case.scalars["alpha"]) * inputs["D"].astype(np.uint64)
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
    assert len(result.extra["phase_cycles"]["batches"]) == 4


@pytest.mark.apu_v1_device
def test_two_mm_apu_v1_scalar_baseline(request, apu_v1_device_gate):
    target = build_target("apu_v1")
    compiled = allo.compile(apu_v1.build_program(_KERNEL), target, bind_cost(target))
    result, verdict, _record = apu_v1.run_compiled(
        compiled,
        _KERNEL,
        folder=request.path.parent,
        run_cmd=_RUN_CMD,
    )
    apu_v1.assert_result(result, verdict)
