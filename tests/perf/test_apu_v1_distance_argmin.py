# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import allo
from allo.ir.types import uint16
from allo.pim.apu_v1_distance_argmin import (
    APUv1SquaredL2ArgminLowering,
    analyze_apu_v1_squared_l2_argmin,
)
from allo.pim.costs import apu_v1_cost
from allo.pim.targets import build_apu_v1_target


P = 32768
D = 24
K = 10


def squared_l2_argmin(
    points: uint16[P, D],
    centers: uint16[K, D],
    labels: uint16[P],
):
    for point in range(P):
        best_distance: uint16 = 0
        for dim in range(D):
            delta: uint16 = points[point, dim] - centers[0, dim]
            best_distance += delta * delta
        best_center: uint16 = 0
        for center in range(1, K):
            distance: uint16 = 0
            for dim in range(D):
                delta: uint16 = points[point, dim] - centers[center, dim]
                distance += delta * delta
            if distance < best_distance:
                best_distance = distance
                best_center = center
        labels[point] = best_center


def _compile():
    phase = allo.APUv1Phase(
        squared_l2_argmin,
        vectorize="required",
        argument_bounds={"points": (0, 31), "centers": (0, 31)},
    )
    program = allo.APUv1Program((phase,), name="squared_l2_argmin")
    return allo.compile(
        program,
        build_apu_v1_target(),
        apu_v1_cost,
        backend="virtual",
    )


def test_required_vector_phase_selects_complete_center_streaming_route():
    callable_program = _compile()
    compiled = callable_program.compiled
    lowering = compiled.native_vector_lowering
    assert isinstance(lowering, APUv1SquaredL2ArgminLowering)
    assert lowering.route == "gvml_squared_l2_argmin_center_streaming"
    assert compiled.hybrid_callable is None
    assert compiled.scratch_bytes == 8

    source = compiled.device_source
    assert "gvml_init_once();" in source
    assert source.index("gvml_init_once();") > source.index("PROF_PRINT(total)")
    assert "for (uint16_t dim = 0; dim < ARGMIN_DIMS; ++dim)" in source
    assert source.count("direct_dma_l4_to_l1_32k(") == 1
    assert source.count("gvml_load_16(") == 2
    assert "gvml_mul_u16(product_vr, point_vr, center_vr);" in source
    assert "gvml_add_u16(dot_vr, dot_vr, product_vr);" in source
    assert "gvml_xor_16(distance_vr, distance_vr, center_vr);" not in source
    assert "gvml_cpy_imm_16_mrk(label_vr, center, GVML_MRK0);" in source
    assert "data->mem_hndl_result_labels" in source
    assert "raw_result_labels[i]" not in source

    inventory = lowering.operation_inventory()
    assert inventory["dma_l4_to_l1_32k"] == D
    assert inventory["gvml_load_16"] == D * (K + 1)
    assert inventory["gvml_mul_u16"] == D * (K + 1)
    assert inventory["gvml_lt_u16"] == K - 1
    assert inventory["gvml_xor_16"] == 0
    assert lowering.analysis.maximum_squared_distance == D * 31 * 31
    assert lowering.analysis.unsigned_order_proven
    assert 500_000 < callable_program.estimate().cycles < 800_000


def test_physical_abi_transposes_points_and_pads_centers_without_mutation():
    lowering = _compile().compiled.native_vector_lowering
    points = np.arange(P * D, dtype=np.uint16).reshape(P, D)
    centers = np.arange(K * D, dtype=np.uint16).reshape(K, D)
    labels = np.zeros(P, dtype=np.uint16)
    logical = {
        "points": points,
        "centers": centers,
        "labels": labels,
        "allo_scratch": np.zeros(8, dtype=np.uint8),
    }
    physical = lowering.pack_inputs(logical)

    assert physical["points"].shape == (D, P)
    assert physical["points"].flags.c_contiguous
    np.testing.assert_array_equal(physical["points"], points.T)
    assert physical["centers"].shape == (256,)
    np.testing.assert_array_equal(physical["centers"][: K * D], centers.reshape(-1))
    assert not np.any(physical["centers"][K * D :])
    assert physical["labels"] is labels
    assert logical["points"] is points
    assert logical["centers"] is centers


def test_ring_identity_and_sign_flip_preserve_retained_i16_argmin():
    rng = np.random.default_rng(2026072204)
    points = rng.integers(0, 65536, size=(113, D), dtype=np.uint16)
    centers = rng.integers(0, 65536, size=(K, D), dtype=np.uint16)

    # Retained MLIR truncates each delta and accumulation to i16.  Squaring in
    # the uint16 ring has the same low bits for either signed representation.
    delta = points[:, None, :] - centers[None, :, :]
    direct = np.sum(
        delta.astype(np.uint32) * delta.astype(np.uint32),
        axis=2,
        dtype=np.uint32,
    ).astype(np.uint16)
    point_norm = np.sum(
        points.astype(np.uint32) * points.astype(np.uint32),
        axis=1,
        dtype=np.uint32,
    ).astype(np.uint16)
    center_norm = np.sum(
        centers.astype(np.uint32) * centers.astype(np.uint32),
        axis=1,
        dtype=np.uint32,
    ).astype(np.uint16)
    dot = np.sum(
        points[:, None, :].astype(np.uint32) * centers[None, :, :].astype(np.uint32),
        axis=2,
        dtype=np.uint32,
    ).astype(np.uint16)
    algebraic = (point_norm[:, None] + center_norm[None, :] - (dot + dot)).astype(
        np.uint16
    )
    np.testing.assert_array_equal(algebraic, direct)

    expected = direct.view(np.int16).argmin(axis=1)
    sign_flipped = np.bitwise_xor(algebraic, np.uint16(0x8000))
    actual = sign_flipped.argmin(axis=1)
    np.testing.assert_array_equal(actual, expected)


def test_structural_recognizer_fails_closed_when_square_is_removed():
    compiled = _compile().compiled
    analysis = compiled.native_vector_lowering.analysis
    assert analysis.candidates == K
    mutated = compiled.artifact.source_mlir.replace("arith.muli", "arith.addi", 1)
    assert (
        analyze_apu_v1_squared_l2_argmin(
            mutated,
            compiled.arguments,
            function=compiled.schedule.top_func_name,
        )
        is None
    )


def test_unbounded_signed_program_emits_exact_sign_order_transform():
    phase = allo.APUv1Phase(squared_l2_argmin, vectorize="required")
    program = allo.APUv1Program((phase,), name="unbounded_squared_l2_argmin")
    compiled = allo.compile(
        program,
        build_apu_v1_target(),
        apu_v1_cost,
        backend="virtual",
    ).compiled
    lowering = compiled.native_vector_lowering
    assert not lowering.analysis.unsigned_order_proven
    assert lowering.analysis.maximum_squared_distance is None
    assert lowering.operation_inventory()["gvml_xor_16"] == K
    assert "gvml_xor_16(distance_vr, distance_vr, center_vr);" in compiled.device_source
