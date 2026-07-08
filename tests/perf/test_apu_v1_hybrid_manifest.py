# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retained-MLIR hybrid manifests for canonical PolyBench phases."""

from pathlib import Path
import sys

import pytest

import allo
from allo.pim.apu_v1_hybrid import APUv1PrecisionPolicy
from allo.pim.targets import build_apu_v1_target


sys.path.insert(0, str(Path(__file__).parents[1] / "pim"))
from lib.upmem_polybench import get_case  # pylint: disable=wrong-import-position


def _program(name, *, precision=True, vectorize="required", **phase_kwargs):
    case = get_case(name)
    arguments = {argument.name for argument in case.arguments}
    result_names = tuple(
        result.name for result in case.results if result.name not in arguments
    )
    phase = allo.APUv1Phase(
        case.kernel,
        name=name.replace("2mm", "two_mm").replace("3mm", "three_mm"),
        instantiate=tuple(case.instantiate),
        result_names=result_names,
        vectorize=vectorize,
        precision_policy=(APUv1PrecisionPolicy.f32_to_f16() if precision else None),
        **phase_kwargs,
    )
    return allo.APUv1Program((phase,), name=f"hybrid_{name}")


def _compile(name, **kwargs):
    return allo.compile(
        _program(name, **kwargs), build_apu_v1_target(), backend="virtual"
    )


def test_canonical_two_mm_discovers_vector_vector_scalar_dataflow():
    manifest = _compile("2mm").hybrid_manifest
    phase = manifest.phases[0]

    assert manifest.has_vector_regions
    assert [region.function for region in phase.regions] == ["mm1", "mm2", "ele_add"]
    assert [region.kind for region in phase.regions] == ["vector", "vector", "scalar"]
    assert phase.intermediates == ("out_AB", "out_ABC")
    assert phase.regions[1].dependencies == ("region0",)
    assert phase.regions[2].dependencies == ("region1",)
    assert [barrier.values for barrier in phase.barriers] == [
        ("out_AB",),
        ("out_ABC",),
    ]
    assert phase.barriers[1].reason == "engine_transition"

    first = phase.regions[0]
    assert first.storage_analysis.numeric_type == "f32"
    assert first.compute_analysis.numeric_type == "f16"
    assert [operand.role for operand in first.operands] == ["lhs", "rhs", "output"]
    assert len(first.plans) == 4
    assert first.selected_plan.name == "temporal_dma_coalescing_broadcast_friendly"
    assert first.produced_intermediates == ("out_AB",)
    assert first.consumed_intermediates == ()

    conversions = {
        (item.value, item.before_region, item.after_region)
        for item in phase.conversions
    }
    assert ("A", "region0", None) in conversions
    assert ("B", "region0", None) in conversions
    assert ("C", "region1", None) in conversions
    assert ("out_ABC", None, "region1") in conversions
    # Zero-filled vector accumulators are initialized directly at compute dtype.
    assert ("out_AB", "region0", None) not in conversions


def test_canonical_three_mm_retains_parallel_producers_and_join_barrier():
    phase = _compile("3mm").hybrid_manifest.phases[0]
    assert [region.kind for region in phase.regions] == ["vector", "vector", "vector"]
    assert phase.regions[0].dependencies == ()
    assert phase.regions[1].dependencies == ()
    assert phase.regions[2].dependencies == ("region0", "region1")
    # Physical planning uses the caller-visible retained value, not mm3's
    # helper-local ``out_ABC`` label for the same accumulator argument.
    assert phase.regions[2].compute_analysis.output.value == "output"
    assert phase.regions[2].compute_analysis.accumulator.value == "output"
    assert len(phase.barriers) == 1
    assert phase.barriers[0].after_regions == ("region0", "region1")
    assert phase.barriers[0].values == ("out_AB", "out_CD")
    assert {item.value for item in phase.conversions if item.before_region} == {
        "A",
        "B",
        "C",
        "D",
    }
    assert any(
        item.value == "output" and item.after_region == "region2"
        for item in phase.conversions
    )


def test_canonical_gemm_records_vector_to_scalar_precision_boundary():
    phase = _compile("gemm").hybrid_manifest.phases[0]
    assert [region.kind for region in phase.regions] == ["vector", "scalar"]
    assert phase.regions[1].dependencies == ("region0",)
    assert phase.barriers[0].reason == "engine_transition"
    boundary = [item for item in phase.conversions if item.value == "out_AB"]
    assert len(boundary) == 1
    assert boundary[0].source_dtype == "f16"
    assert boundary[0].target_dtype == "f32"
    assert boundary[0].after_region == "region0"


def test_required_f32_vectorization_needs_explicit_precision_policy():
    with pytest.raises(ValueError, match="explicit precision_policy"):
        _compile("2mm", precision=False)


def test_false_vectorize_keeps_canonical_calls_explicitly_scalar():
    phase = _compile("2mm", precision=False, vectorize=False).hybrid_manifest.phases[0]
    assert [region.kind for region in phase.regions] == ["scalar", "scalar", "scalar"]
    assert not phase.conversions


def test_phase_schema_freezes_bindings_producers_and_dependencies():
    case = get_case("gemm")
    phase0 = allo.APUv1Phase(
        case.kernel,
        name="producer",
        instantiate=tuple(case.instantiate),
        produces=("tmp",),
        zero_initialize=("tmp",),
        bindings={"A": "logical_A"},
    )
    phase1 = allo.APUv1Phase(
        case.kernel,
        name="consumer",
        instantiate=tuple(case.instantiate),
        produces=("output",),
        dependencies=("producer",),
    )
    program = allo.APUv1Program((phase0, phase1))
    assert dict(phase0.bindings) == {"A": "logical_A"}
    assert program.phases[1].dependencies == ("producer",)
    with pytest.raises(TypeError):
        phase0.bindings["A"] = "changed"

    with pytest.raises(ValueError, match="unknown/forward"):
        allo.APUv1Program((phase1, phase0))
    with pytest.raises(ValueError, match="multiple producing"):
        allo.APUv1Program(
            (phase0, allo.APUv1Phase(case.kernel, name="again", produces=("tmp",)))
        )


def test_hybrid_manifest_is_json_ready_and_exposed_on_public_callable():
    compiled = _compile("2mm")
    payload = compiled.hybrid_manifest.manifest()
    assert payload["precision_policy"]["compute_dtype"] == "f16"
    assert payload["phases"][0]["regions"][0]["kind"] == "vector"
    assert payload["values"][0]["program_input"] is True
