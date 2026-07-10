# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic rank-N retained-MLIR contraction planning for APUg2."""

import json
import re

import allo
import numpy as np
import pytest
from allo.ir.types import float16, uint16

from allo.pim.apu_g2_contraction import (
    APUG2AffineAccessMap,
    APUG2RankNContractionPlan,
    UnsupportedAPUG2RankNContractionError,
    plan_apu_g2_rank_n_contraction,
    plan_apu_g2_rank_n_contractions,
)
from allo.pim.apu_g2_vector_program import (
    APUG2ColumnBatchedGemmCallable,
    APUG2ContractionChainCallable,
    APUG2RankNContractionCallable,
    APUG2StreamingGemvCallable,
)
from allo.pim.costs.apu_g2 import apu_g2_cost
from allo.pim.targets import build_apu_g2_target


def ordinary_gemm(A: uint16[60, 80], B: uint16[80, 70], output: uint16[60, 70]):
    for row, column in allo.grid(60, 70):
        for depth in allo.reduction(80):
            output[row, column] += A[row, depth] * B[depth, column]


def ordinary_doitgen(A: uint16[25, 20, 30], x: uint16[30, 30], scratch: uint16[30]):
    for r, q in allo.grid(25, 20):
        for p in allo.grid(30):
            scratch[p] = 0
            for s in allo.reduction(30):
                scratch[p] += A[r, q, s] * x[s, p]
        for p_out in allo.grid(30):
            A[r, q, p_out] = scratch[p_out]


def floating_gemm(A: float16[3, 7], B: float16[7, 5], output: float16[3, 5]):
    for row, column in allo.grid(3, 5):
        for depth in allo.reduction(7):
            output[row, column] += A[row, depth] * B[depth, column]


def ordinary_large_gemv(
    A: uint16[40, 300], x: uint16[300], output: uint16[40]
):
    for row in allo.grid(40):
        for depth in allo.reduction(300):
            output[row] += A[row, depth] * x[depth]


def scaled_small_gemv(A: uint16[4, 5], x: uint16[5], output: uint16[4]):
    for row in allo.grid(4):
        output[row] = output[row] * 4
        for depth in allo.reduction(5):
            output[row] += 5 * A[row, depth] * x[depth]


def two_mm_stage_ab(A: uint16[4, 5], B: uint16[5, 3], AB: uint16[4, 3]):
    for row, column in allo.grid(4, 3):
        for depth in allo.reduction(5):
            AB[row, column] += A[row, depth] * B[depth, column]


def two_mm_stage_abc(AB: uint16[4, 3], C: uint16[3, 6], D: uint16[4, 6]):
    for row, column in allo.grid(4, 6):
        D[row, column] = D[row, column] * 1
        for depth in allo.reduction(3):
            D[row, column] += 5 * AB[row, depth] * C[depth, column]


def ordinary_two_mm(A: uint16[4, 5], B: uint16[5, 3], C: uint16[3, 6], D: uint16[4, 6]):
    AB: uint16[4, 3] = 0
    two_mm_stage_ab(A, B, AB)
    two_mm_stage_abc(AB, C, D)


def three_mm_stage_ab(A: uint16[4, 5], B: uint16[5, 3], AB: uint16[4, 3]):
    for row, column in allo.grid(4, 3):
        for depth in allo.reduction(5):
            AB[row, column] += A[row, depth] * B[depth, column]


def three_mm_stage_cd(C: uint16[3, 7], D: uint16[7, 6], CD: uint16[3, 6]):
    for row, column in allo.grid(3, 6):
        for depth in allo.reduction(7):
            CD[row, column] += C[row, depth] * D[depth, column]


def three_mm_stage_out(AB: uint16[4, 3], CD: uint16[3, 6], output: uint16[4, 6]):
    for row, column in allo.grid(4, 6):
        for depth in allo.reduction(3):
            output[row, column] += AB[row, depth] * CD[depth, column]


def ordinary_three_mm(
    A: uint16[4, 5],
    B: uint16[5, 3],
    C: uint16[3, 7],
    D: uint16[7, 6],
    output: uint16[4, 6],
):
    AB: uint16[4, 3] = 0
    CD: uint16[3, 6] = 0
    three_mm_stage_ab(A, B, AB)
    three_mm_stage_cd(C, D, CD)
    three_mm_stage_out(AB, CD, output)


def _module(function):
    return allo.customize(function, enable_tensor=False).module


def _alpha_rename(text):
    text = re.sub(r"@[-\w.$]+", "@renamed_function", text)
    names = {}

    def replace_ssa(match):
        source = match.group(0)
        return names.setdefault(source, f"%renamed_{len(names)}")

    return re.sub(r"%[-\w.$]+", replace_ssa, text)


def test_gemm_plan_flattens_rank_two_output_and_recovers_access_maps():
    plan = plan_apu_g2_rank_n_contraction(_module(ordinary_gemm))

    assert isinstance(plan, APUG2RankNContractionPlan)
    assert plan.dot_axes == (("row", 60), ("column", 70))
    assert plan.batch_axes == ()
    assert plan.output_axes == ("row", "column")
    assert plan.reduction_axis == ("depth", 80)
    assert plan.flat_output_extent == 60 * 70
    assert plan.tiling.padded_reduction_extent == 128
    assert plan.tiling.tile_capacity == 2048
    assert plan.tiling.tile_count == 3
    assert [plan.tiling.tile_bounds(tile) for tile in range(3)] == [
        (0, 2048),
        (2048, 4096),
        (4096, 4200),
    ]

    assert isinstance(plan.lhs, APUG2AffineAccessMap)
    assert plan.lhs.value == "A"
    assert plan.lhs.domain_axes == ("row", "column", "depth")
    assert plan.lhs.result_axes == ("row", "depth")
    assert plan.lhs.axis_positions == (0, 2)
    assert plan.lhs.broadcast_axes == ("column",)
    assert plan.rhs.value == "B"
    assert plan.rhs.result_axes == ("depth", "column")
    assert plan.rhs.axis_positions == (2, 1)
    assert plan.initial_output.value == plan.output.value == "output"
    assert plan.initial_output.result_axes == ("row", "column")
    assert plan.batch_local_accumulator is False
    assert plan.module.contraction_topology == "single"

    assert plan.flatten_output(row=59, column=69) == 4199
    assert plan.unflatten_output(4199) == (59, 69)
    assert plan.lhs.apply((4, 5, 6)) == (4, 6)
    assert plan.rhs.apply((4, 5, 6)) == (6, 5)


def test_doitgen_plan_preserves_batch_axes_and_local_accumulator():
    plan = plan_apu_g2_rank_n_contraction(_module(ordinary_doitgen))

    assert plan.dot_axes == (("r", 25), ("q", 20), ("p", 30))
    assert plan.batch_axes == ("r", "q")
    assert plan.output_axes == ("p",)
    assert plan.reduction_axis == ("s", 30)
    assert plan.flat_output_extent == 25 * 20 * 30
    assert plan.tiling.padded_reduction_extent == 32
    assert plan.tiling.tile_capacity == 8192
    assert plan.tiling.tile_count == 2
    assert plan.tiling.tile_bounds(0) == (0, 8192)
    assert plan.tiling.tile_bounds(1) == (8192, 15000)

    assert plan.lhs.value == "A"
    assert plan.lhs.domain_axes == ("r", "q", "p", "s")
    assert plan.lhs.result_axes == ("r", "q", "s")
    assert plan.lhs.axis_positions == (0, 1, 3)
    assert plan.lhs.broadcast_axes == ("p",)
    assert plan.rhs.value == "x"
    assert plan.rhs.result_axes == ("s", "p")
    assert plan.rhs.axis_positions == (3, 2)
    assert plan.rhs.broadcast_axes == ("r", "q")
    assert plan.initial_output.value == plan.output.value == "scratch"
    assert plan.initial_output.result_axes == ("p",)
    assert plan.batch_local_accumulator is True

    assert plan.flatten_output(r=24, q=19, p=29) == 14999
    assert plan.unflatten_output(14999) == (24, 19, 29)
    assert plan.lhs.apply((2, 3, 4, 5)) == (2, 3, 5)
    assert plan.rhs.apply((2, 3, 4, 5)) == (5, 4)
    assert plan.initial_output.apply((2, 3, 4, 5)) == (4,)


def test_rank_n_plan_manifest_is_json_serializable_and_structural():
    gemm = plan_apu_g2_rank_n_contraction(_module(ordinary_gemm))
    doitgen = plan_apu_g2_rank_n_contraction(_module(ordinary_doitgen))

    for plan in (gemm, doitgen):
        payload = plan.manifest()
        assert payload["kind"] == "apu-g2-rank-n-contraction"
        assert payload["region_id"] == plan.module.regions[0].id
        assert payload["flat_output_extent"] == plan.flat_output_extent
        json.dumps(payload)
    assert gemm.batch_local_accumulator is False
    assert doitgen.batch_local_accumulator is True


def test_rank_n_plan_rejects_unsupported_numeric_semantics():
    with pytest.raises(UnsupportedAPUG2RankNContractionError, match="uint16"):
        plan_apu_g2_rank_n_contraction(_module(floating_gemm))


def test_public_compile_routes_large_gemv_to_streaming_schedule():
    compiled = allo.compile(
        ordinary_large_gemv,
        build_apu_g2_target(),
        apu_g2_cost,
        backend="virtual",
    )

    assert isinstance(compiled, APUG2StreamingGemvCallable)
    assert compiled.execution_graph.metadata["program"] == "streaming_gemv_u16"
    assert compiled.execution_graph.metadata["hardware_tasks"] == 3
    schedule = compiled.execution_graph.metadata["transport_schedule"]
    assert schedule["reduction_tile"] == 128
    assert schedule["output_readbacks"] == 1
    assert schedule["resident_accumulator"] is True


def test_public_compile_routes_gemm_to_column_batched_schedule_without_execution():
    compiled = allo.compile(
        ordinary_gemm,
        build_apu_g2_target(),
        apu_g2_cost,
        backend="virtual",
    )

    assert isinstance(compiled, APUG2ColumnBatchedGemmCallable)
    assert compiled.execution_graph.metadata["program"] == "column_batched_gemm_u16"
    assert compiled.execution_graph.metadata["hardware_tasks"] == 3
    transport = dict(compiled.execution_graph.metadata["transport_schedule"])
    wall_estimate = transport.pop("host_wall_estimate")
    assert transport == {
        "kind": "column_batched_u16_gemm",
        "rows": 60,
        "columns": 70,
        "reduction": 80,
        "batch_columns": 31,
        "column_batches": 3,
        "reduction_tile": 128,
        "reduction_tiles": 1,
        "weight_uploads": 3,
        "output_readbacks": 3,
        "resident_accumulator": True,
        "contiguous_readback": True,
    }
    assert wall_estimate["hardware_tasks"] == 3
    assert wall_estimate["wall_us"] > 0
    assert compiled.execution_graph.metadata["output_shape"] == (60, 70)
    assert compiled.execution_graph.metadata["epilogue"] == (
        "alpha_dot_plus_beta_accumulator"
    )
    assert compiled.execution_graph.metadata["alpha"] == 1
    assert compiled.execution_graph.metadata["beta"] == 1
    assert compiled.estimate().cycles == 9_806 + 9_806 + 9_911

    rows = np.arange(60, dtype=np.uint16)[:, None]
    depths = np.arange(80, dtype=np.uint16)[None, :]
    columns = np.arange(70, dtype=np.uint16)[None, :]
    A = (rows * np.uint16(251) + depths * np.uint16(509)).astype(np.uint16)
    B = (
        np.arange(80, dtype=np.uint16)[:, None] * np.uint16(997)
        + columns * np.uint16(17)
    ).astype(np.uint16)
    output = (rows * np.uint16(1877) + columns * np.uint16(41)).astype(np.uint16)
    original = output.copy()

    bound = compiled.signature.bind(A, B, output)
    left, right, accumulator, sites = compiled._pack_tile(bound.arguments, 0)
    assert left.shape == right.shape == (2048, 80)
    assert accumulator.shape == (2048,)
    np.testing.assert_array_equal(left[0], A[0])
    np.testing.assert_array_equal(right[0], B[:, 0])
    np.testing.assert_array_equal(left[70], A[1])
    np.testing.assert_array_equal(right[70], B[:, 0])
    assert accumulator[70] == output[1, 0]
    assert sites[70] == (1, 0)

    result = compiled(A, B, output)
    assert result.backend == "virtual"
    assert result.cycles == compiled.estimate().cycles
    assert result.extra["outputs"] == {}
    assert result.extra["hardware_tasks"] == 3
    np.testing.assert_array_equal(output, original)


def test_rank_n_planner_discovers_chained_and_dag_regions():
    two_mm = plan_apu_g2_rank_n_contractions(_module(ordinary_two_mm))
    three_mm = plan_apu_g2_rank_n_contractions(_module(ordinary_three_mm))

    assert [plan.analysis.function for plan in two_mm] == [
        "two_mm_stage_ab",
        "two_mm_stage_abc",
    ]
    assert two_mm[0].module.contraction_topology == "linked_chain"
    assert two_mm[0].module.dependencies[0].value == "AB"
    assert [plan.flat_output_extent for plan in two_mm] == [12, 24]

    assert [plan.analysis.function for plan in three_mm] == [
        "three_mm_stage_ab",
        "three_mm_stage_cd",
        "three_mm_stage_out",
    ]
    assert three_mm[0].module.contraction_topology == "dag"
    assert {edge.value for edge in three_mm[0].module.dependencies} == {"AB", "CD"}
    assert [plan.flat_output_extent for plan in three_mm] == [12, 18, 24]


def test_public_compile_routes_2mm_chain_to_column_batched_gemm():
    target = build_apu_g2_target()
    compiled = allo.compile(ordinary_two_mm, target, apu_g2_cost, backend="virtual")

    assert isinstance(compiled, APUG2ContractionChainCallable)
    assert compiled.module_manifest.contraction_topology == "linked_chain"
    assert compiled.execution_graph.metadata["program"] == (
        "transport_aware_contraction_chain_u16"
    )
    assert compiled.execution_graph.metadata["hardware_tasks"] == 2
    assert compiled.stage_epilogues == ((1, 1), (5, 1))
    assert compiled.recipe.certificate.scalar_tensor_updates == 0
    assert compiled.execution_graph.metadata["transport_schedule"][
        "resident_accumulator"
    ] is True

    A = np.arange(20, dtype=np.uint16).reshape(4, 5)
    B = np.arange(15, dtype=np.uint16).reshape(5, 3)
    C = np.arange(18, dtype=np.uint16).reshape(3, 6)
    D = np.full((4, 6), 19, dtype=np.uint16)
    original = D.copy()
    run = compiled(A, B, C, D)
    assert run.backend == "virtual"
    assert run.extra["hardware_tasks"] == 2
    assert run.extra["stage_epilogues"] == [
        {"alpha": 1, "beta": 1},
        {"alpha": 5, "beta": 1},
    ]
    np.testing.assert_array_equal(D, original)


def test_scaled_epilogues_are_structural_and_alpha_rename_invariant():
    retained = str(_module(ordinary_two_mm))
    original = plan_apu_g2_rank_n_contractions(retained)
    renamed = plan_apu_g2_rank_n_contractions(_alpha_rename(retained))

    assert tuple(plan.epilogue for plan in original) == ((1, 1), (5, 1))
    assert tuple(plan.epilogue for plan in renamed) == ((1, 1), (5, 1))
    assert [plan.manifest()["epilogue"] for plan in original] == [
        {"alpha": 1, "beta": 1},
        {"alpha": 5, "beta": 1},
    ]

    mutated = retained.replace("arith.constant 5", "arith.constant 7")
    assert tuple(
        plan.epilogue for plan in plan_apu_g2_rank_n_contractions(mutated)
    ) == ((1, 1), (7, 1))


def test_unproven_prescale_fails_closed_instead_of_defaulting_to_unit_beta():
    retained = str(_module(ordinary_two_mm))
    lines = retained.splitlines()
    prescale_store = next(
        index
        for index, line in enumerate(lines)
        if 'to = "D"' in line and "affine.store" in line
    )
    prescale_multiply = max(
        index
        for index, line in enumerate(lines[:prescale_store])
        if "arith.muli" in line
    )
    lines[prescale_multiply] = lines[prescale_multiply].replace(
        "arith.muli", "arith.addi"
    )

    with pytest.raises(
        UnsupportedAPUG2RankNContractionError,
        match="unsupported contribution|not a proven prescale",
    ):
        plan_apu_g2_rank_n_contractions("\n".join(lines))


def test_nonunit_small_gemv_uses_coefficient_aware_rank_n_route():
    target = build_apu_g2_target()
    compiled = allo.compile(
        scaled_small_gemv, target, apu_g2_cost, backend="virtual"
    )

    assert isinstance(compiled, APUG2RankNContractionCallable)
    assert compiled.epilogue == (5, 4)
    assert compiled.execution_graph.metadata["runtime_epilogue"] == {
        "alpha": 5,
        "beta": 4,
    }


def test_public_compile_routes_3mm_dag_to_column_batched_gemm():
    target = build_apu_g2_target()
    compiled = allo.compile(ordinary_three_mm, target, apu_g2_cost, backend="virtual")

    assert isinstance(compiled, APUG2ContractionChainCallable)
    assert compiled.module_manifest.contraction_topology == "dag"
    assert compiled.execution_graph.metadata["program"] == (
        "transport_aware_contraction_chain_u16"
    )
    assert compiled.execution_graph.metadata["hardware_tasks"] == 3
    assert compiled.stage_epilogues == ((1, 1), (1, 1), (1, 1))
    assert compiled.recipe.certificate.scalar_tensor_updates == 0

    A = np.arange(20, dtype=np.uint16).reshape(4, 5)
    B = np.arange(15, dtype=np.uint16).reshape(5, 3)
    C = np.arange(21, dtype=np.uint16).reshape(3, 7)
    D = np.arange(42, dtype=np.uint16).reshape(7, 6)
    output = np.zeros((4, 6), dtype=np.uint16)
    run = compiled(A, B, C, D, output)
    assert run.backend == "virtual"
    assert run.extra["hardware_tasks"] == 3
    np.testing.assert_array_equal(output, np.zeros((4, 6), dtype=np.uint16))
