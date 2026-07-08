# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical 3mm through three structurally selected GVML contractions."""

import numpy as np

import pytest

import allo
from lib import apu_v1, cell
from lib.cost import bind_cost
from lib.targets import build_target

_KERNEL = "3mm"
_RUN_CMD = (
    "python -m pytest tests/pim/apu_v1/3mm/test_three_mm_apu_v1.py"
    "::test_three_mm_apu_v1_scalar_baseline "
    "-p no:cacheprovider -q"
)


def _assert_hybrid_manifest(compiled):
    phase = compiled.hybrid_manifest.phases[0]
    assert [(region.function, region.kind) for region in phase.regions] == [
        ("mm1", "vector"),
        ("mm2", "vector"),
        ("mm3", "vector"),
    ]
    assert phase.regions[0].dependencies == ()
    assert phase.regions[1].dependencies == ()
    assert phase.regions[2].dependencies == (
        phase.regions[0].id,
        phase.regions[1].id,
    )
    assert phase.barriers[0].values == ("out_AB", "out_CD")
    assert all(region.selected_plan is not None for region in phase.regions)
    assert phase.conversions == ()
    assert all(
        region.compute_analysis.numeric_type == "ui16" for region in phase.regions
    )


def test_three_mm_apu_v1_hybrid_manifest(request):
    workload = cell.load_workload(request.path.parent, _KERNEL)
    target = build_target("apu_v1")
    compiled = allo.compile(
        workload.build(), target, bind_cost(target), backend="virtual"
    )
    _assert_hybrid_manifest(compiled)


def test_three_mm_apu_v1_hybrid_functional_uint16(request):
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
    cd = (inputs["C"].astype(np.uint64) @ inputs["D"].astype(np.uint64)).astype(
        np.uint16
    )
    staged = (ab.astype(np.uint64) @ cd.astype(np.uint64)).astype(np.uint16)
    np.testing.assert_array_equal(observed, staged)
    np.testing.assert_array_equal(observed, canonical)
    assert run.extra["host_intermediate_round_trips"] == 0


@pytest.mark.apu_v1_device
def test_three_mm_apu_v1_hybrid_device_uint16(request, apu_v1_device_gate):
    """Three uint16 GVML contractions preserve exact modular semantics."""

    workload = cell.load_workload(request.path.parent, _KERNEL)
    target = build_target("apu_v1")
    compiled = allo.compile(workload.build(), target, bind_cost(target))
    _assert_hybrid_manifest(compiled)
    case = apu_v1.get_case(_KERNEL)
    inputs = case.make_inputs()
    canonical = case.run_reference(inputs)["output"]

    def matmul_u16(lhs, rhs):
        return (lhs.astype(np.uint64) @ rhs.astype(np.uint64)).astype(np.uint16)

    ab = matmul_u16(inputs["A"], inputs["B"])
    cd = matmul_u16(inputs["C"], inputs["D"])
    staged = matmul_u16(ab, cd)

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
    assert result.extra["scalar_apucs"] == ()
    assert result.extra["final_gather_apucs"] == (0,)
    assert len(result.extra["phase_cycles"]["batches"]) == 5


@pytest.mark.apu_v1_device
def test_three_mm_apu_v1_scalar_baseline(request, apu_v1_device_gate):
    target = build_target("apu_v1")
    compiled = allo.compile(apu_v1.build_program(_KERNEL), target, bind_cost(target))
    result, verdict, _record = apu_v1.run_compiled(
        compiled,
        _KERNEL,
        folder=request.path.parent,
        run_cmd=_RUN_CMD,
    )
    apu_v1.assert_result(result, verdict)
