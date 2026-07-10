# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structural retained-MLIR program-graph tests for APUg2."""

from dataclasses import FrozenInstanceError
import json

import allo
import numpy as np
import pytest
from allo.ir.types import uint16

from allo.pim.apu_g2_ir import (
    APUG2ModuleManifest,
    APUG2RegionManifest,
    APUG2UseDefManifest,
    APUG2ValueManifest,
    discover_apu_g2_module_manifest,
)
from allo.pim.contraction_analysis import analyze_contractions
from allo.pim.costs.apu_g2 import apu_g2_cost
from allo.pim.targets import build_apu_g2_target
from allo.pim.apu_g2_vector_program import APUG2IndependentContractionsCallable


def atax_stage_m(A: uint16[7, 5], x: uint16[5], tmp: uint16[7]):
    for row in allo.grid(7):
        for depth in allo.reduction(5):
            tmp[row] += A[row, depth] * x[depth]


def atax_stage_n(A: uint16[7, 5], tmp: uint16[7], y: uint16[5]):
    for column in allo.grid(5):
        for depth in allo.reduction(7):
            y[column] += A[depth, column] * tmp[depth]


def atax_program(A: uint16[7, 5], x: uint16[5], y: uint16[5]):
    tmp: uint16[7] = 0
    atax_stage_m(A, x, tmp)
    atax_stage_n(A, tmp, y)


def bicg_stage_s(A: uint16[7, 5], r: uint16[7], s: uint16[5]):
    for column in allo.grid(5):
        for depth in allo.reduction(7):
            s[column] += A[depth, column] * r[depth]


def bicg_stage_q(A: uint16[7, 5], p: uint16[5], q: uint16[7]):
    for row in allo.grid(7):
        for depth in allo.reduction(5):
            q[row] += A[row, depth] * p[depth]


def bicg_program(
    A: uint16[7, 5],
    p: uint16[5],
    r: uint16[7],
    q: uint16[7],
    s: uint16[5],
):
    bicg_stage_s(A, r, s)
    bicg_stage_q(A, p, q)


def _module(function):
    return allo.customize(function, enable_tensor=False).module


def test_atax_contractions_form_one_use_def_chain_without_name_matching():
    graph = discover_apu_g2_module_manifest(_module(atax_program))

    assert isinstance(graph, APUG2ModuleManifest)
    assert len(graph.regions) == 2
    assert graph.contraction_topology == "linked_chain"
    assert len(graph.roots) == len(graph.sinks) == 1
    assert graph.roots != graph.sinks

    dependency = graph.dependencies
    assert len(dependency) == 1
    assert isinstance(dependency[0], APUG2UseDefManifest)
    assert dependency[0].value == "tmp"
    assert dependency[0].producer_region == graph.roots[0]
    assert dependency[0].consumer_region == graph.sinks[0]
    assert dependency[0].operand_role in {"lhs", "rhs"}
    assert graph.region(graph.roots[0]).function == "atax_stage_m"
    assert graph.region(graph.sinks[0]).function == "atax_stage_n"

    tmp = graph.value("tmp")
    assert isinstance(tmp, APUG2ValueManifest)
    assert tmp.dtype == "ui16"
    assert tmp.shape == (7,)
    assert tmp.definition_regions == graph.roots
    assert set(tmp.use_regions) == set(graph.regions[index].id for index in (0, 1))


def test_bicg_contractions_share_inputs_without_a_dependency_edge():
    graph = discover_apu_g2_module_manifest(_module(bicg_program))

    assert len(graph.regions) == 2
    assert graph.contraction_topology == "independent"
    assert graph.dependencies == ()
    assert graph.roots == tuple(region.id for region in graph.regions)
    assert graph.sinks == tuple(region.id for region in graph.regions)
    assert {region.function for region in graph.regions} == {
        "bicg_stage_s",
        "bicg_stage_q",
    }

    matrix_uses = tuple(edge for edge in graph.use_defs if edge.value == "A")
    assert len(matrix_uses) == 2
    assert all(edge.producer_region is None for edge in matrix_uses)
    assert {edge.consumer_region for edge in matrix_uses} == set(graph.roots)
    assert graph.value("A").definition_regions == ()


def test_manifest_accepts_preanalyzed_regions_and_is_json_serializable():
    module = _module(atax_program)
    analyses = analyze_contractions(module)
    graph = discover_apu_g2_module_manifest(analyses)

    payload = graph.manifest()
    assert payload["kind"] == "apu-g2-module"
    assert payload["contraction_topology"] == "linked_chain"
    assert len(payload["values"]) == len(graph.values)
    assert len(payload["regions"]) == 2
    assert len(payload["use_defs"]) == 6
    json.dumps(payload)


def test_manifest_values_are_deeply_immutable():
    graph = discover_apu_g2_module_manifest(_module(bicg_program))

    assert all(isinstance(value, APUG2ValueManifest) for value in graph.values)
    assert all(isinstance(region, APUG2RegionManifest) for region in graph.regions)
    with pytest.raises(FrozenInstanceError):
        graph.regions = ()
    with pytest.raises(FrozenInstanceError):
        graph.regions[0].function = "renamed"
    with pytest.raises(TypeError):
        graph.regions[0].input_values[0] = "changed"


def test_public_dispatch_uses_topology_and_never_mislabels_bicg_as_atax():
    target = build_apu_g2_target()
    bicg = allo.compile(bicg_program, target, apu_g2_cost, backend="virtual")
    assert isinstance(bicg, APUG2IndependentContractionsCallable)
    assert bicg.module_manifest.contraction_topology == "independent"
    assert bicg.execution_graph.metadata["hardware_tasks"] == 1
    assert bicg.recipe.certificate.scalar_tensor_updates == 0
    A = np.arange(35, dtype=np.uint16).reshape(7, 5)
    p = np.arange(5, dtype=np.uint16)
    r = np.arange(7, dtype=np.uint16)
    q = np.zeros(7, dtype=np.uint16)
    s = np.zeros(5, dtype=np.uint16)
    run = bicg(A, p, r, q, s)
    assert run.backend == "virtual"
    assert run.extra["hardware_tasks"] == 1
    assert not np.any(q) and not np.any(s)

    compiled = allo.compile(atax_program, target, apu_g2_cost, backend="virtual")
    assert compiled.module_manifest.contraction_topology == "linked_chain"
