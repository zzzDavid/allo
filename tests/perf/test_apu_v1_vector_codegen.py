# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused source, ABI, and numeric tests for the APU v1 plan realizer."""

import hashlib
import json

import numpy as np
import pytest

from allo.pim.apu_v1_layout import (
    APUV1Plan,
    AffineTiledLayout,
    IterationLayout,
    ReductionStrategy,
    ReuseWindow,
    Transfer,
    TransferLayout,
    TransferRouteStep,
    ValueLayout,
)
from allo.spmw_linear_layout import LinearLayout
from allo.pim.apu_v1_vector_codegen import (
    APULayoutPackingError,
    UnsupportedVectorOperation,
    VRCapacityError,
    VectorOp,
    VectorValue,
    binary_matmul_ops,
    fp16_contraction_ops,
    realize_apu_v1_plan,
    uint16_contraction_ops,
    xnor_popcount_ops,
)


def _plan(
    extent,
    *,
    transfers,
    value_names=("lhs", "rhs", "out"),
    layout=None,
    metadata=None,
):
    layout = layout or AffineTiledLayout.packed({"element": extent})
    return APUV1Plan(
        "test_vector_plan",
        IterationLayout(("element",), layout),
        tuple(ValueLayout(name, ("element",), layout=layout) for name in value_names),
        tuple(transfers),
        "single_pass",
        ReductionStrategy("element", group_size=extent),
        metadata=metadata or {},
    )


def _descriptors(extent, *, dtype=np.float16):
    return (
        VectorValue("lhs", (extent,), dtype, "in"),
        VectorValue("rhs", (extent,), dtype, "in"),
        VectorValue("out", (extent // 4,), dtype, "out"),
    )


def _simple_realization():
    plan = _plan(
        8,
        transfers=(
            Transfer("lhs", "in", coalesced=True),
            Transfer("rhs", "in", coalesced=True),
            Transfer("out", "out", coalesced=True),
        ),
    )
    return realize_apu_v1_plan(
        plan,
        operations=fp16_contraction_ops("lhs", "rhs", "out", group_size=4),
        values=_descriptors(8),
    )


def test_fp16_contraction_emits_verified_dma_and_gvml_calls():
    plan = _plan(
        8,
        transfers=(
            Transfer("lhs", "in", coalesced=True),
            Transfer("rhs", "in", coalesced=True),
            Transfer("out", "out", coalesced=True),
        ),
    )
    realization = realize_apu_v1_plan(
        plan,
        operations=fp16_contraction_ops("lhs", "rhs", "out", group_size=4),
        values=_descriptors(8),
    )

    source = realization.device_source()
    assert "direct_dma_l4_to_l1_32k" in source
    assert "gvml_load_16" in source
    assert "gvml_mul_f16" in source
    assert "gvml_add_subgrps_f16_grp" in source
    assert "gvml_store_16" in source
    assert "direct_dma_l1_to_l4_32k" in source
    assert "GVML_VR16_15" not in source
    assert "apu_pio_store_layout" not in source


def test_runtime_artifact_freezes_complete_project_driver_build_and_abi(tmp_path):
    realization = _simple_realization()
    artifact = realization.runtime_artifact

    assert artifact is not None
    project = dict(artifact.project_files)
    assert project["Makefile"]
    assert project["host.c"]
    assert project["struct.h"]
    assert realization.device_source().encode("utf-8") in project["device.c"]
    assert any(path.startswith("Common/") for path in project)
    source_hashes = dict(artifact.source_hashes)
    assert source_hashes["contract/build.json"]
    assert source_hashes["contract/abi.json"]
    assert source_hashes["contract/executor.json"]
    assert any(path.startswith("runtime/") for path in source_hashes)
    assert len(artifact.source_fingerprint) == 64
    assert len(realization.promotion_materialization_fingerprint) == 64
    assert realization.promotion_platform_fingerprint is None
    written = artifact.write_project(tmp_path / "project")
    for relative, mode in artifact.project_modes:
        assert (written / relative).stat().st_mode & 0o777 == mode


def test_runtime_project_mutation_after_realization_fails_closed(monkeypatch):
    from allo.pim import apu_v1_vector_runtime as runtime

    realization = _simple_realization()
    artifact = realization.runtime_artifact
    assert len(realization.promotion_materialization_fingerprint) == 64
    original_inventory = runtime._template_inventory

    def mutated_inventory():
        hashes, modes = original_inventory()
        hashes = dict(hashes)
        path = next(iter(hashes))
        hashes[path] = "f" * 64
        return tuple(sorted(hashes.items())), modes

    monkeypatch.setattr(runtime, "_template_inventory", mutated_inventory)

    assert realization.promotion_materialization_fingerprint is None
    with pytest.raises(RuntimeError, match="template changed"):
        artifact.assert_current(realization)


def test_caller_supplied_g1_files_cannot_attest_hardware_platform(
    monkeypatch, tmp_path
):
    components = {}
    for category in ("sdk", "toolchain", "firmware"):
        path = tmp_path / f"{category}.contract"
        path.write_bytes(f"{category}-revision-1".encode("ascii"))
        components[category] = {
            category: {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        }
    manifest = {
        "schema": "tenon-promotion-platform-v1",
        "target": "apu_v1",
        "hardware_family": "gemini-i",
        **components,
    }
    contract = tmp_path / "platform.json"
    contract.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv("TENON_APU_V1_PLATFORM_CONTRACT", str(contract))

    realization = _simple_realization()
    artifact = realization.runtime_artifact
    assert artifact.platform_fingerprint is None
    assert artifact.current_platform_fingerprint() is None
    assert realization.promotion_platform_fingerprint is None
    assert len(realization.promotion_materialization_fingerprint) == 64

    (tmp_path / "firmware.contract").write_bytes(b"firmware-revision-2")
    assert realization.promotion_platform_fingerprint is None
    artifact.assert_current(realization)


@pytest.mark.parametrize("extent", [32, 256, 4096, 32768])
def test_four_micro_binary_candidate_shapes_lower_without_kernel_names(extent):
    plan = _plan(
        extent,
        transfers=(
            Transfer("lhs", "in", coalesced=True),
            Transfer("rhs", "in", coalesced=True),
            Transfer("out", "out", coalesced=True),
        ),
    )
    values = (
        VectorValue("lhs", (extent,), np.uint16, "in"),
        VectorValue("rhs", (extent,), np.uint16, "in"),
        VectorValue("out", (1,), np.int16, "out"),
    )
    source = realize_apu_v1_plan(
        plan,
        operations=binary_matmul_ops("lhs", "rhs", "out", group_size=extent),
        values=values,
    ).device_source()

    for api in (
        "gvml_xor_16",
        "gvml_not_16",
        "gvml_popcount_16",
        "gvml_sl_imm_16",
        "gvml_sub_s16",
        "gvml_add_subgrps_s16_grp",
    ):
        assert api in source
    assert "matmul" not in source.lower()


def test_transfer_choice_uses_duplicate_lookup_and_real_pio_lane_reads():
    plan = _plan(
        32,
        transfers=(
            Transfer("lhs", "in", coalesced=True, broadcast=True),
            Transfer("rhs", "in", coalesced=False, broadcast=True),
            Transfer("out", "out", coalesced=False),
        ),
        metadata={"subgroup_size": 1, "rhs_lookup_table_size": 32},
    )
    source = realize_apu_v1_plan(
        plan,
        operations=(VectorOp("ADD_F16", "out", ("lhs", "rhs")),),
        values=(
            VectorValue("lhs", (32,), np.float16, "in"),
            VectorValue("rhs", (32,), np.float16, "in"),
            VectorValue("out", (32,), np.float16, "out"),
        ),
    ).device_source()

    assert "gvml_duplicate_subgrp_16_grp_sgidx" in source
    assert "gvml_lookup_16" in source
    assert "uint16_t *rhs_L3ptr" in source
    assert "gvml_get_entry_16" in source


def test_bitwise_and_or_lowering_and_numpy_semantics():
    plan = _plan(
        4,
        transfers=(
            Transfer("lhs", "in", coalesced=True),
            Transfer("rhs", "in", coalesced=True),
            Transfer("out", "out", coalesced=True),
        ),
    )
    values = (
        VectorValue("lhs", (4,), np.uint16, "in"),
        VectorValue("rhs", (4,), np.uint16, "in"),
        VectorValue("out", (4,), np.uint16, "out"),
    )
    operations = (
        VectorOp("AND_16", "out", ("lhs", "rhs")),
        VectorOp("OR_16", "out", ("out", "lhs")),
    )
    realization = realize_apu_v1_plan(plan, operations=operations, values=values)
    source = realization.device_source()
    assert "gvml_and_16" in source
    assert "gvml_or_16" in source
    lhs = np.array([0x0000, 0x00FF, 0xAAAA, 0x1234], dtype=np.uint16)
    rhs = np.array([0xFFFF, 0x0F0F, 0x5555, 0x4321], dtype=np.uint16)
    actual = realization.execute_numpy({"lhs": lhs, "rhs": rhs})["out"]
    np.testing.assert_array_equal(actual, np.bitwise_or(np.bitwise_and(lhs, rhs), lhs))


def test_layout_driven_pack_and_gather_replicates_exactly():
    layout = AffineTiledLayout.packed({"element": 5, "replica": 4})
    plan = APUV1Plan(
        "replicated_abi",
        IterationLayout(("element",), layout),
        (ValueLayout("out", ("element",), ("replica",), layout),),
        (Transfer("out", "out", coalesced=True),),
        "single_pass",
        ReductionStrategy("element", kind="none"),
    )
    realization = realize_apu_v1_plan(
        plan,
        operations=(VectorOp("CPY_IMM_16", "out", attrs={"value": 0}),),
        values=(VectorValue("out", (5,), np.float16, "out"),),
    )
    original = np.array([1.0, -2.5, 3.25, 7.0, 0.125], dtype=np.float16)
    packed = realization.abi.pack({"out": original})["out"]

    for element in range(5):
        coordinates = realization.abi._coordinates(  # pylint: disable=protected-access
            realization.values[0], (element,)
        )
        assert len(coordinates) == 4
        assert len({int(packed.data[coordinate]) for coordinate in coordinates}) == 1
    gathered = realization.abi.gather({"out": packed})["out"]
    np.testing.assert_array_equal(gathered, original)

    bad = packed.data.copy()
    bad[realization.abi._coordinates(realization.values[0], (0,))[1]] ^= 1
    with pytest.raises(APULayoutPackingError, match="replica values disagree"):
        realization.abi.gather({"out": bad})


def test_numpy_semantics_match_fp16_grouped_contraction():
    plan = _plan(
        8,
        transfers=(
            Transfer("lhs", "in", coalesced=True),
            Transfer("rhs", "in", coalesced=True),
            Transfer("out", "out", coalesced=True),
        ),
    )
    realization = realize_apu_v1_plan(
        plan,
        operations=fp16_contraction_ops("lhs", "rhs", "out", group_size=4),
        values=_descriptors(8),
    )
    lhs = np.arange(1, 9, dtype=np.float16)
    rhs = np.array([1, -1, 2, 0.5, 3, 2, -1, 4], dtype=np.float16)
    actual = realization.execute_numpy({"lhs": lhs, "rhs": rhs})["out"]
    expected = (lhs * rhs).reshape(2, 4).sum(axis=1)
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)


def test_uint16_contraction_uses_native_gvml_and_wraps_modulo_2_to_16():
    plan = _plan(
        8,
        transfers=(
            Transfer("lhs", "in", coalesced=True),
            Transfer("rhs", "in", coalesced=True),
            Transfer("out", "out", coalesced=True),
        ),
    )
    realization = realize_apu_v1_plan(
        plan,
        operations=uint16_contraction_ops("lhs", "rhs", "out", group_size=4),
        values=_descriptors(8, dtype=np.uint16),
    )

    source = realization.device_source()
    assert "gvml_mul_u16" in source
    assert "gvml_add_subgrps_u16_grp" in source
    assert "gvml_mul_f16" not in source

    lhs = np.array([0xFFFF, 2, 3, 4, 300, 400, 500, 600], dtype=np.uint16)
    rhs = np.array([2, 3, 4, 5, 300, 400, 500, 600], dtype=np.uint16)
    actual = realization.execute_numpy({"lhs": lhs, "rhs": rhs})["out"]
    wide = lhs.astype(np.uint64) * rhs.astype(np.uint64)
    expected = (wide.reshape(2, 4).sum(axis=1) & 0xFFFF).astype(np.uint16)
    np.testing.assert_array_equal(actual, expected)


def test_numpy_semantics_match_packed_binary_dot_products():
    plan = _plan(
        4,
        transfers=(
            Transfer("lhs", "in", coalesced=True),
            Transfer("rhs", "in", coalesced=True),
            Transfer("out", "out", coalesced=True),
        ),
    )
    values = (
        VectorValue("lhs", (4,), np.uint16, "in"),
        VectorValue("rhs", (4,), np.uint16, "in"),
        VectorValue("out", (2,), np.int16, "out"),
    )
    realization = realize_apu_v1_plan(
        plan,
        operations=binary_matmul_ops("lhs", "rhs", "out", group_size=2),
        values=values,
    )
    lhs = np.array([0xFFFF, 0xAAAA, 0x0F0F, 0x1234], dtype=np.uint16)
    rhs = np.array([0xFFFF, 0x5555, 0x00FF, 0x4321], dtype=np.uint16)
    actual = realization.execute_numpy({"lhs": lhs, "rhs": rhs})["out"]
    contribution = np.array(
        [2 * ((~(int(a) ^ int(b))) & 0xFFFF).bit_count() - 16 for a, b in zip(lhs, rhs)]
    )
    np.testing.assert_array_equal(actual, contribution.reshape(2, 2).sum(axis=1))


def test_target_neutral_xnor_popcount_preserves_raw_count_semantics():
    plan = _plan(
        4,
        transfers=(
            Transfer("lhs", "in", coalesced=True),
            Transfer("rhs", "in", coalesced=True),
            Transfer("out", "out", coalesced=True),
        ),
    )
    values = (
        VectorValue("lhs", (4,), np.uint16, "in"),
        VectorValue("rhs", (4,), np.uint16, "in"),
        VectorValue("out", (4,), np.int16, "out"),
    )
    realization = realize_apu_v1_plan(
        plan,
        operations=xnor_popcount_ops("lhs", "rhs", "out", accumulator="out"),
        values=values,
    )
    lhs = np.array([0xFFFF, 0xAAAA, 0x0F0F, 0x1234], dtype=np.uint16)
    rhs = np.array([0xFFFF, 0x5555, 0x00FF, 0x4321], dtype=np.uint16)
    actual = realization.execute_numpy(
        {"lhs": lhs, "rhs": rhs, "out": np.zeros(4, np.int16)}
    )["out"]
    expected = np.array(
        [((~(int(a) ^ int(b))) & 0xFFFF).bit_count() for a, b in zip(lhs, rhs)],
        dtype=np.int16,
    )
    np.testing.assert_array_equal(actual, expected)
    source = realization.device_source()
    assert "gvml_popcount_16" in source
    assert "gvml_sl_imm_16" not in source
    assert "gvml_sub_s16" not in source


def test_realizer_fails_closed_for_unbound_plan_operations_and_vr_pressure():
    plan = _plan(
        4,
        transfers=(Transfer("lhs", "in", coalesced=True),),
        value_names=("lhs",),
    )
    values = (VectorValue("lhs", (4,), np.float16, "in"),)
    with pytest.raises(UnsupportedVectorOperation, match="no realizable compute"):
        realize_apu_v1_plan(plan, operations=(), values=values)

    operations = tuple(
        VectorOp("CPY_IMM_16", f"temp{i}", attrs={"value": i}) for i in range(4)
    ) + (
        VectorOp("ADD_S16", "temp4", ("temp0", "temp1")),
        VectorOp("ADD_S16", "lhs", ("temp2", "temp3")),
    )
    with pytest.raises(VRCapacityError, match="more than 2"):
        realize_apu_v1_plan(plan, operations=operations, values=values, vr_capacity=2)


def _micro_route_plan(m=64, n=1024, k=16, rows_per_vr=8):
    """Small full-geometry plan: two output batches and two resident RHS VRs."""

    row_bits = m.bit_length() - 1
    column_bits = n.bit_length() - 1
    depth_bits = k.bit_length() - 1
    iteration = LinearLayout(
        {
            "row": [
                ((n << bit), 0) if bit < 5 else (0, k << (bit - 5))
                for bit in range(row_bits)
            ],
            "column": [((1 << bit), 0) for bit in range(column_bits)],
            "depth": [(0, 1 << bit) for bit in range(depth_bits)],
        },
        ("vr_lane", "vr_batch"),
        (32768, (m // 32) * k),
    )
    output = LinearLayout(
        {
            "row": [
                ((n << bit), 0) if bit < 5 else (0, 1 << (bit - 5))
                for bit in range(row_bits)
            ],
            "column": [((1 << bit), 0) for bit in range(column_bits)],
        },
        ("vr_lane", "vr_batch"),
        (32768, m // 32),
    )

    lhs_compact = LinearLayout(
        {
            "row": [
                ((1 << bit),) if bit < 5 else ((32 * k) << (bit - 5),)
                for bit in range(row_bits)
            ],
            "depth": [((32 << bit),) for bit in range(depth_bits)],
        },
        ("l4_element",),
        (m * k,),
    )
    lhs_l4 = TransferLayout("l4", ("row", "depth"), lhs_compact, role="compact")
    lhs_l3 = TransferLayout("l3", ("row", "depth"), lhs_compact, role="compact")
    lhs_vr = TransferLayout(
        "compute_vr", ("row", "depth"), iteration, ("column",), role="compute"
    )

    resident_replicas = 32768 // (rows_per_vr * n)
    rhs_resident = LinearLayout(
        {
            "depth": [
                (
                    ((n << bit), 0)
                    if bit < (rows_per_vr.bit_length() - 1)
                    else (0, 1 << (bit - (rows_per_vr.bit_length() - 1)))
                )
                for bit in range(depth_bits)
            ],
            "column": [((1 << bit), 0) for bit in range(column_bits)],
            "resident_replica": [
                ((rows_per_vr * n) << bit, 0)
                for bit in range(resident_replicas.bit_length() - 1)
            ],
        },
        ("vr_lane", "vr_batch"),
        (32768, k // rows_per_vr),
    )
    rhs_l4 = TransferLayout(
        "l4",
        ("depth", "column"),
        rhs_resident,
        ("resident_replica",),
        role="resident",
    )
    rhs_l1 = TransferLayout(
        "l1",
        ("depth", "column"),
        rhs_resident,
        ("resident_replica",),
        role="resident",
    )
    rhs_resident_vr = TransferLayout(
        "resident_vr",
        ("depth", "column"),
        rhs_resident,
        ("resident_replica",),
        role="resident",
    )
    rhs_compute_vr = TransferLayout(
        "compute_vr", ("depth", "column"), iteration, ("row",), role="compute"
    )

    lhs_route = (
        TransferRouteStep("dma_l4_l3", lhs_l4, lhs_l3, temporal_axis="depth"),
        TransferRouteStep(
            "lookup",
            lhs_l3,
            lhs_vr,
            temporal_axis="depth",
            parameters={"table_size": 32, "group_size": n},
        ),
    )
    reuse = (ReuseWindow("row", 32, extent=m),)
    rhs_route = (
        TransferRouteStep(
            "dma_l4_l1_32k",
            rhs_l4,
            rhs_l1,
            temporal_axis="depth",
            resident_across=reuse,
        ),
        TransferRouteStep(
            "load_vr",
            rhs_l1,
            rhs_resident_vr,
            temporal_axis="depth",
            resident_across=reuse,
        ),
        TransferRouteStep(
            "duplicate_subgroup",
            rhs_resident_vr,
            rhs_compute_vr,
            temporal_axis="depth",
            resident_across=reuse,
            parameters={
                "rows_per_vr": rows_per_vr,
                "subgroup_size": n,
                "group_size": rows_per_vr * n,
            },
        ),
    )
    plan = APUV1Plan(
        "explicit_transit",
        IterationLayout(("row", "column", "depth"), iteration),
        (
            ValueLayout("lhs", ("row", "depth"), ("column",), iteration),
            ValueLayout("rhs", ("depth", "column"), ("row",), iteration),
            ValueLayout("out", ("row", "column"), layout=output),
        ),
        (
            Transfer("lhs", "in", route=lhs_route),
            Transfer("rhs", "in", route=rhs_route),
            Transfer("out", "in", coalesced=True),
            Transfer("out", "out", coalesced=True),
        ),
        "temporal",
        ReductionStrategy("depth", kind="temporal_accumulate"),
        metadata={"output_tiles": m // 32, "temporal_steps": k},
    )
    return plan


def test_explicit_transit_routes_hoist_resident_rhs_and_pack_compact_images():
    plan = _micro_route_plan()
    values = (
        VectorValue("lhs", (64, 16), np.float16, "in"),
        VectorValue("rhs", (16, 1024), np.float16, "in"),
        VectorValue("out", (64, 1024), np.float16, "inout"),
    )
    realization = realize_apu_v1_plan(
        plan,
        operations=(
            VectorOp("MUL_F16", "product", ("lhs", "rhs")),
            VectorOp("ADD_F16", "out", ("out", "product")),
        ),
        values=values,
    )
    bindings = realization.binding_map
    resident = {
        binding.concrete for name, binding in bindings.items() if "rhs_resident" in name
    }
    loop_live = {
        binding.concrete
        for name, binding in bindings.items()
        if name in {"lhs", "rhs", "out", "product"} or "index" in name
    }
    assert len(resident) == 2
    assert resident.isdisjoint(loop_live)
    source = realization.device_source()
    outer = source.index("for (uint32_t output_batch")
    assert source.index("direct_dma_l4_to_l1_32k") < outer
    assert source.count("direct_dma_l4_to_l1_32k") == 3  # 2 resident + carried C
    assert "reduction_step / 8" in source
    assert "reduction_step % 8" in source
    assert "output_batch * 16 + reduction_step" in source
    assert source.count("gvml_lookup_16") == 1
    assert source.count("gvml_duplicate_subgrp_16_grp_sgidx") == 1

    lhs = np.arange(64 * 16, dtype=np.float16).reshape(64, 16)
    rhs = np.arange(16 * 1024, dtype=np.float16).reshape(16, 1024)
    images = realization.abi.transfer_input_images(
        {"lhs": lhs, "rhs": rhs, "out": np.zeros((64, 1024), np.float16)}
    )
    expected_lhs = np.stack(
        [lhs[batch * 32 : (batch + 1) * 32].T for batch in range(2)]
    ).reshape(-1)
    np.testing.assert_array_equal(
        images["lhs"][: expected_lhs.size].view(np.float16), expected_lhs
    )
    rhs_bits = rhs.view(np.uint16)
    first_group = rhs_bits[:8].reshape(-1)
    for replica in range(4):
        begin = replica * 8192
        np.testing.assert_array_equal(images["rhs"][begin : begin + 8192], first_group)
