# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect

import allo
import numpy as np
import pytest

from lib.upmem_polybench import NAMES, REGISTRY, build_upmem_program, get_case


EXPECTED = (
    "2mm",
    "3mm",
    "adi",
    "atax",
    "bicg",
    "cholesky",
    "correlation",
    "covariance",
    "deriche",
    "doitgen",
    "durbin",
    "fdtd_2d",
    "floyd_warshall",
    "gemm",
    "gemver",
    "gesummv",
    "gramschmidt",
    "heat_3d",
    "jacobi_1d",
    "jacobi_2d",
    "lu",
    "ludcmp",
    "mvt",
    "nussinov",
    "seidel_2d",
    "symm",
    "syr2k",
    "syrk",
    "trisolv",
    "trmm",
)


def test_registry_is_the_complete_30_kernel_suite():
    assert NAMES == EXPECTED
    assert tuple(REGISTRY) == EXPECTED
    assert get_case("two_mm") is REGISTRY["2mm"]
    assert get_case("three_mm") is REGISTRY["3mm"]


def test_upmem_program_chooses_widest_legal_partition_axis():
    assert build_upmem_program("2mm").arrays["output"].partition_axis == 1
    assert build_upmem_program("3mm").arrays["output"].partition_axis == 1
    assert build_upmem_program("gemm").arrays["output"].partition_axis == 1


@pytest.mark.parametrize("name", EXPECTED)
def test_frontend_abi_instantiation_and_execution_metadata(name):
    case = get_case(name)
    kernel, instantiate = case.kernel_and_instantiate()

    assert callable(kernel)
    assert kernel is getattr(case.module, case.kernel_name)
    assert case.reference is getattr(case.module, case.reference_name)
    assert list(inspect.signature(kernel).parameters) == [
        a.name for a in case.arguments
    ]
    assert instantiate[1:] == [case.dims[key] for key in case.instantiate_dimensions]

    assert case.execution.phases
    for phase in case.execution.phases:
        assert phase.name
        assert phase.kind in {
            "independent",
            "temporal",
            "pivot",
            "wavefront",
            "serial",
            "serial-line",
            "reduction",
        }
        assert phase.partition
        assert phase.parallel
        assert phase.barrier


@pytest.mark.parametrize("name", EXPECTED)
def test_canonical_kernel_enters_mlir_with_registered_instantiation(name):
    case = get_case(name)
    kernel, instantiate = case.kernel_and_instantiate()
    schedule = allo.customize(kernel, instantiate=instantiate)
    assert schedule.module is not None


@pytest.mark.parametrize("name", EXPECTED)
def test_complete_upmem_program_manifest(name):
    case = get_case(name)
    program = build_upmem_program(case)
    source_arguments = {argument.name for argument in case.arguments}
    expected_returns = tuple(
        result.name for result in case.results if result.name not in source_arguments
    )

    assert program.phases[0].result_names == expected_returns
    assert expected_returns == (("output",) if name in {"2mm", "3mm"} else ())
    assert program.orchestration == program.execution_manifest
    assert program.manifest["execution"] == program.execution_manifest
    assert program.execution_manifest["num_dpus"] == 64
    assert program.execution_manifest["parallel_region_policy"] == (
        "retained-mlir-dependence-analysis"
    )
    assert program.phases[0].parallel_loops == ()
    assert len(program.execution_manifest["phases"]) == len(case.execution.phases)
    assert set(program.arrays) == source_arguments | set(expected_returns)


@pytest.mark.parametrize("name", EXPECTED)
def test_deterministic_input_generation_matches_declared_abi(name):
    case = get_case(name)
    first = case.make_inputs(seed=19)
    second = case.make_inputs(seed=19)

    assert tuple(first) == tuple(a.name for a in case.arguments)
    for arg in case.arguments:
        assert first[arg.name].shape == arg.shape
        assert first[arg.name].dtype == np.float32
        np.testing.assert_array_equal(first[arg.name], second[arg.name])
        assert not np.shares_memory(first[arg.name], second[arg.name])


@pytest.mark.parametrize("name", EXPECTED)
def test_repository_reference_contract_at_small(name):
    case = get_case(name)
    inputs = case.make_inputs(seed=23)
    saved = {key: value.copy() for key, value in inputs.items()}
    outputs = case.run_reference(inputs)

    # Reference functions are mutating by design; the registry isolates them.
    for key in inputs:
        np.testing.assert_array_equal(inputs[key], saved[key])

    assert tuple(outputs) == tuple(result.name for result in case.results)
    for result in case.results:
        value = outputs[result.name]
        assert value.shape == result.shape
        assert value.dtype == np.float32
        assert np.isfinite(value).all()


@pytest.mark.parametrize(
    "name,bindings",
    (
        ("2mm", {"alpha", "beta"}),
        ("adi", {"a", "b", "c", "d", "e", "f"}),
        ("correlation", {"N_float", "epsilon"}),
        (
            "deriche",
            {"a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "b1", "b2", "c1", "c2"},
        ),
        ("gemm", {"beta"}),
        ("gemver", {"alpha", "beta"}),
        ("gesummv", {"alpha", "beta"}),
        ("jacobi_2d", {"TSTEPS"}),
        ("symm", {"alpha", "beta"}),
        ("syr2k", {"alpha", "beta"}),
        ("syrk", {"alpha", "beta"}),
        ("trmm", {"alpha"}),
    ),
)
def test_ambient_frontend_scalars_are_bound_explicitly(name, bindings):
    case = get_case(name)
    module = case.bind_ambient()
    assert set(case.scalars) == bindings
    for key, value in case.scalars.items():
        assert getattr(module, key) == value
