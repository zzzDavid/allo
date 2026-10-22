# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import allo
from allo.ir.types import uint16
from allo.pim.apu_v1_distance_argmin import APUv1SquaredL2ArgminLowering
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
