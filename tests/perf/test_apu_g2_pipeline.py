# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for typed direct-VL64 pipeline calibrations."""

from dataclasses import FrozenInstanceError, fields, replace
from enum import Enum
import json

import pytest

from allo.pim.apu_g2_pipeline import (
    CallInventoryEntry,
    NORMALIZED_DUAL_STREAM_REDUCTION_CALIBRATION,
    NORMALIZED_RESIDENT_TWO_STAGE_REDUCTION_CALIBRATION,
    NORMALIZED_SINGLE_STREAM_REDUCTION_CALIBRATION,
    Opcode,
    OpAttribution,
    PipelineCalibration,
    PipelineSignature,
    PipelineTopology,
    STANDALONE_4X64K_ADD_CALIBRATION,
    ShapeValidityDomain,
    VL64Scope,
)


def _inventory(calibration):
    return tuple(
        (entry.opcode, entry.count)
        for entry in calibration.signature.call_inventory
    )


def _attributions(calibration):
    return tuple(
        (attribution.opcode, attribution.cycles_per_call)
        for attribution in calibration.attributions
    )


@pytest.mark.parametrize(
    "calibration,expected_total,expected_inventory,expected_attributions",
    [
        (
            STANDALONE_4X64K_ADD_CALIBRATION,
            266,
            (
                (Opcode.COPY_L1_TO_MMB_SEGMENT_0, 1),
                (Opcode.COPY_L1_TO_MMB_SEGMENT_1, 1),
                (Opcode.ADD, 1),
                (Opcode.COPY_MMB_TO_L1, 1),
                (Opcode.BARRIER, 1),
            ),
            (
                (Opcode.COPY_L1_TO_MMB_SEGMENT_0, 56),
                (Opcode.COPY_L1_TO_MMB_SEGMENT_1, 56),
                (Opcode.ADD, 98),
                (Opcode.COPY_MMB_TO_L1, 55),
                (Opcode.BARRIER, 1),
            ),
        ),
        (
            NORMALIZED_SINGLE_STREAM_REDUCTION_CALIBRATION,
            6_893,
            (
                (Opcode.MUL, 3),
                (Opcode.REDUCE, 3),
                (Opcode.SHIFT, 2),
                (Opcode.ADD, 3),
                (Opcode.BARRIER, 1),
            ),
            (
                (Opcode.MUL, 580),
                (Opcode.REDUCE, 1_474),
                (Opcode.SHIFT, 179),
                (Opcode.ADD, 124),
                (Opcode.BARRIER, 1),
            ),
        ),
        (
            NORMALIZED_DUAL_STREAM_REDUCTION_CALIBRATION,
            8_001,
            (
                (Opcode.MUL, 6),
                (Opcode.REDUCE, 3),
                (Opcode.SHIFT, 4),
                (Opcode.ADD, 5),
                (Opcode.BARRIER, 1),
            ),
            (
                (Opcode.MUL, 426),
                (Opcode.REDUCE, 1_152),
                (Opcode.SHIFT, 142),
                (Opcode.ADD, 284),
                (Opcode.BARRIER, 1),
            ),
        ),
        (
            NORMALIZED_RESIDENT_TWO_STAGE_REDUCTION_CALIBRATION,
            76_118,
            (
                (Opcode.MUL, 15),
                (Opcode.REDUCE, 15),
                (Opcode.SHIFT, 10),
                (Opcode.ADD, 16),
                (Opcode.SQUEEZE, 20),
                (Opcode.SPREAD, 4),
                (Opcode.BARRIER, 1),
            ),
            (
                (Opcode.MUL, 580),
                (Opcode.REDUCE, 1_474),
                (Opcode.SHIFT, 179),
                (Opcode.ADD, 124),
                (Opcode.SQUEEZE, 1_200),
                (Opcode.SPREAD, 4_382),
                (Opcode.BARRIER, 6),
            ),
        ),
    ],
)
def test_calibrated_regimes_have_exact_inventory_and_total(
    calibration,
    expected_total,
    expected_inventory,
    expected_attributions,
):
    assert calibration.total_cycles == expected_total
    assert _inventory(calibration) == expected_inventory
    assert _attributions(calibration) == expected_attributions
    assert calibration.cycles_for(calibration.signature) == expected_total


def test_signatures_capture_only_complete_structural_facts():
    standalone = STANDALONE_4X64K_ADD_CALIBRATION.signature
    single = NORMALIZED_SINGLE_STREAM_REDUCTION_CALIBRATION.signature
    dual = NORMALIZED_DUAL_STREAM_REDUCTION_CALIBRATION.signature
    resident = NORMALIZED_RESIDENT_TWO_STAGE_REDUCTION_CALIBRATION.signature

    assert standalone.topology is PipelineTopology.STANDALONE
    assert standalone.coalesced_matrix_streams == 0
    assert standalone.shape_domain.extent_bounds == ((4, 4), (65_536, 65_536))

    assert single.topology is PipelineTopology.NORMALIZED
    assert single.coalesced_matrix_streams == 1
    assert dual.coalesced_matrix_streams == 2
    assert not single.resident_intermediate
    assert not dual.resident_intermediate
    assert resident.resident_intermediate
    assert resident.coalesced_matrix_streams == 1

    for signature in (standalone, single, dual, resident):
        assert signature.vl64_scope is VL64Scope.CORE_WIDE
        assert signature.vl64_group_count == 16


def test_equivalent_models_have_deterministic_manifest_key_and_fingerprint():
    original = NORMALIZED_DUAL_STREAM_REDUCTION_CALIBRATION
    domain = original.signature.shape_domain
    equivalent_domain = ShapeValidityDomain(
        extent_bounds=list(domain.extent_bounds),
        power_of_two_padding=list(domain.power_of_two_padding),
        padded_extent_limits=list(domain.padded_extent_limits),
        max_padded_volume=domain.max_padded_volume,
    )
    equivalent_signature = PipelineSignature(
        topology=original.signature.topology,
        coalesced_matrix_streams=original.signature.coalesced_matrix_streams,
        call_inventory=[
            CallInventoryEntry(entry.opcode, entry.count)
            for entry in original.signature.call_inventory
        ],
        resident_intermediate=original.signature.resident_intermediate,
        vl64_scope=original.signature.vl64_scope,
        vl64_group_count=original.signature.vl64_group_count,
        shape_domain=equivalent_domain,
    )
    equivalent = PipelineCalibration(
        signature=equivalent_signature,
        attributions=[
            OpAttribution(item.opcode, item.cycles_per_call)
            for item in reversed(original.attributions)
        ],
    )

    assert equivalent == original
    assert equivalent.canonical_manifest == original.canonical_manifest
    assert equivalent.key == original.key
    assert equivalent.fingerprint == original.fingerprint
    assert len(original.fingerprint) == 64
    assert json.loads(original.key) == original.canonical_manifest


def test_structural_models_have_no_diagnostic_or_workload_identity_fields():
    model_types = (
        CallInventoryEntry,
        ShapeValidityDomain,
        PipelineSignature,
        OpAttribution,
        PipelineCalibration,
    )
    forbidden = {
        "name",
        "diagnostic",
        "benchmark",
        "workload",
        "kernel",
        "event_id",
        "metadata",
    }

    for model_type in model_types:
        assert model_type.__dataclass_params__.frozen
        assert forbidden.isdisjoint(field.name for field in fields(model_type))
    assert issubclass(Opcode, Enum)
    assert issubclass(PipelineTopology, Enum)
    assert issubclass(VL64Scope, Enum)


def test_fingerprint_is_sensitive_to_every_signature_fact_and_attribution():
    calibration = NORMALIZED_SINGLE_STREAM_REDUCTION_CALIBRATION
    signature = calibration.signature
    domain = signature.shape_domain
    inventory = signature.call_inventory
    signature_variants = (
        replace(signature, topology=PipelineTopology.STANDALONE),
        replace(signature, coalesced_matrix_streams=2),
        replace(signature, call_inventory=(inventory[1], inventory[0], *inventory[2:])),
        replace(
            signature,
            call_inventory=(replace(inventory[0], count=4), *inventory[1:]),
        ),
        replace(signature, resident_intermediate=True),
        replace(signature, vl64_scope=VL64Scope.PER_GROUP),
        replace(signature, vl64_group_count=8),
        replace(
            signature,
            shape_domain=replace(
                domain,
                extent_bounds=((1, 32_768), domain.extent_bounds[1]),
            ),
        ),
        replace(
            signature,
            shape_domain=replace(domain, power_of_two_padding=(False, True)),
        ),
        replace(
            signature,
            shape_domain=replace(domain, padded_extent_limits=(32_768, 256)),
        ),
        replace(
            signature,
            shape_domain=replace(domain, max_padded_volume=32_768),
        ),
    )

    variant_fingerprints = {
        PipelineCalibration(variant, calibration.attributions).fingerprint
        for variant in signature_variants
    }
    assert len(variant_fingerprints) == len(signature_variants)
    assert calibration.fingerprint not in variant_fingerprints

    changed_attributions = tuple(
        replace(item, cycles_per_call=item.cycles_per_call + 1)
        if item.opcode is Opcode.ADD
        else item
        for item in calibration.attributions
    )
    changed = PipelineCalibration(signature, changed_attributions)
    assert changed.fingerprint != calibration.fingerprint
    assert changed.total_cycles == calibration.total_cycles + 3


def test_call_order_changes_structural_identity_even_when_total_is_unchanged():
    calibration = NORMALIZED_SINGLE_STREAM_REDUCTION_CALIBRATION
    inventory = calibration.signature.call_inventory
    reordered_signature = replace(
        calibration.signature,
        call_inventory=(*inventory[1:3], inventory[0], *inventory[3:]),
    )
    reordered = PipelineCalibration(reordered_signature, calibration.attributions)

    assert reordered.total_cycles == calibration.total_cycles
    assert reordered.signature.key != calibration.signature.key
    assert reordered.fingerprint != calibration.fingerprint


def test_direct_reduction_shape_domain_enforces_padding_and_capacity():
    domain = NORMALIZED_SINGLE_STREAM_REDUCTION_CALIBRATION.signature.shape_domain

    assert domain.validate(1, 1) == (1, 1)
    assert domain.validate((128, 255)) == (128, 255)
    assert domain.contains(256, 128)
    assert not domain.contains(257, 256)
    with pytest.raises(ValueError, match=r"\[1, 256\]"):
        domain.validate(1, 257)
    with pytest.raises(ValueError, match="capacity"):
        domain.validate(257, 256)


@pytest.mark.parametrize(
    "extents,exception",
    [
        ((0, 1), ValueError),
        ((1,), ValueError),
        ((1, True), TypeError),
        ((1, 1.5), TypeError),
    ],
)
def test_shape_domain_rejects_invalid_extent_inputs(extents, exception):
    domain = NORMALIZED_DUAL_STREAM_REDUCTION_CALIBRATION.signature.shape_domain

    with pytest.raises(exception):
        domain.validate(*extents)


def test_fixed_and_resident_shape_domains_enforce_physical_bounds():
    standalone = STANDALONE_4X64K_ADD_CALIBRATION.signature
    resident = NORMALIZED_RESIDENT_TWO_STAGE_REDUCTION_CALIBRATION.signature

    assert standalone.validate_shape(4, 65_536) == (4, 65_536)
    assert resident.validate_shape(1, 128) == (1, 128)
    assert resident.validate_shape(128, 1) == (128, 1)
    for shape in ((3, 65_536), (4, 65_535)):
        with pytest.raises(ValueError):
            standalone.validate_shape(*shape)
    for shape in ((0, 1), (1, 0), (129, 1), (1, 129)):
        with pytest.raises(ValueError):
            resident.validate_shape(*shape)


@pytest.mark.parametrize(
    "constructor,args,exception",
    [
        (CallInventoryEntry, (Opcode.ADD, 0), ValueError),
        (CallInventoryEntry, (Opcode.ADD, True), TypeError),
        (OpAttribution, (Opcode.ADD, 0), ValueError),
        (OpAttribution, (Opcode.ADD, 1.5), TypeError),
    ],
)
def test_inventory_counts_and_attributed_cycles_must_be_positive(
    constructor, args, exception
):
    with pytest.raises(exception):
        constructor(*args)


def test_signature_rejects_duplicate_inventory_entries():
    signature = NORMALIZED_SINGLE_STREAM_REDUCTION_CALIBRATION.signature

    with pytest.raises(ValueError, match="repeat an opcode"):
        replace(
            signature,
            call_inventory=(
                *signature.call_inventory,
                CallInventoryEntry(Opcode.ADD, 1),
            ),
        )


def test_calibration_rejects_duplicate_missing_and_extra_attributions():
    calibration = NORMALIZED_SINGLE_STREAM_REDUCTION_CALIBRATION
    attributions = calibration.attributions

    with pytest.raises(ValueError, match="repeat an opcode"):
        PipelineCalibration(
            calibration.signature,
            (*attributions, OpAttribution(Opcode.ADD, 1)),
        )
    with pytest.raises(ValueError, match="cover exactly"):
        PipelineCalibration(calibration.signature, attributions[:-1])
    with pytest.raises(ValueError, match="cover exactly"):
        PipelineCalibration(
            calibration.signature,
            (*attributions, OpAttribution(Opcode.SQUEEZE, 1)),
        )


def test_calibration_rejects_a_structurally_different_signature():
    single = NORMALIZED_SINGLE_STREAM_REDUCTION_CALIBRATION
    dual = NORMALIZED_DUAL_STREAM_REDUCTION_CALIBRATION

    with pytest.raises(ValueError, match="does not match"):
        single.cycles_for(dual.signature)
    with pytest.raises(ValueError, match="does not match"):
        single.require_signature(replace(single.signature, vl64_group_count=8))


def test_model_instances_are_frozen():
    calibration = NORMALIZED_SINGLE_STREAM_REDUCTION_CALIBRATION

    with pytest.raises(FrozenInstanceError):
        calibration.total_cycles = 1
    with pytest.raises(FrozenInstanceError):
        calibration.signature.vl64_group_count = 1
