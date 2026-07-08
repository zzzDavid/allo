# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Layout-exact vector ABI coverage for canonical hybrid SMALL regions."""

from pathlib import Path
import sys

import numpy as np
import pytest

import allo
from allo.pim.apu_v1_hybrid import (
    build_apu_v1_region_executables,
    partition_apu_v1_region,
    realize_apu_v1_region_shards,
)
from allo.pim.apu_v1_vector_codegen import APUVectorABI, VectorValue, VR_LANES
from allo.pim.targets import build_apu_v1_target


sys.path.insert(0, str(Path(__file__).parents[1] / "pim"))
from lib.apu_v1 import build_hybrid_program  # pylint: disable=wrong-import-position


@pytest.mark.parametrize("case", ["gemm", "2mm", "3mm"])
def test_selected_canonical_region_abis_pack_transfer_and_gather_edges(case):
    compiled = allo.compile(
        build_hybrid_program(case),
        build_apu_v1_target(),
        backend="functional",
    )
    manifest = compiled.hybrid_manifest
    executables = build_apu_v1_region_executables(manifest, compiled.compiled.schedule)

    for phase in manifest.phases:
        for region in phase.regions:
            if region.kind != "vector":
                continue
            realization = executables[region.id].artifact
            # Ones make physical padding observable: every valid expanded
            # replica has non-zero bits and every masked edge must remain zero.
            arrays = {
                value.name: np.ones(value.shape, dtype=value.dtype)
                for value in realization.values
            }
            outputs = tuple(
                value.name
                for value in realization.values
                if value.intent in {"out", "inout"}
            )
            packed_outputs = realization.abi.pack(arrays, values=outputs)
            gathered = realization.abi.gather(packed_outputs, outputs=outputs)
            for name in outputs:
                np.testing.assert_array_equal(gathered[name], arrays[name])
                packed = packed_outputs[name]
                assert np.count_nonzero(packed.data[~packed.valid]) == 0

            routed = realization.abi.transfer_input_images(arrays)
            assert set(routed) == {
                value.name for value in realization.values if value.intent == "in"
            }
            assert all(image.size % VR_LANES == 0 for image in routed.values())
            logical_iteration_extent = np.prod(
                tuple(region.compute_analysis.axis_extents.values())
            )
            assert all(
                np.count_nonzero(image) == logical_iteration_extent
                for image in routed.values()
            )

            # A patterned probe verifies F2 batch/lane order, not only the
            # number of valid lanes.  Corner and midpoint coordinates cover
            # both logical values and their replica axes.
            patterned = {
                value.name: np.arange(np.prod(value.shape), dtype=value.dtype).reshape(
                    value.shape
                )
                for value in realization.values
            }
            patterned_routes = realization.abi.transfer_input_images(patterned)
            for value in realization.values:
                if value.intent != "in":
                    continue
                transfer = next(
                    item
                    for item in realization.plan.transfers
                    if item.value == value.name and item.direction == "in"
                )
                layout = transfer.route[-1].destination.layout
                value_layout = realization.plan.value_layout(value.name)
                logical_probes = (
                    tuple(0 for _ in value.shape),
                    tuple(extent // 2 for extent in value.shape),
                    tuple(extent - 1 for extent in value.shape),
                )
                bits = np.ascontiguousarray(patterned[value.name]).view(np.uint16)
                for probe_number, logical_index in enumerate(logical_probes):
                    indices = dict(zip(value_layout.axes, logical_index))
                    for replica in value_layout.replica_axes:
                        extent = layout.input_extents[replica]
                        indices[replica] = (0, extent // 2, extent - 1)[probe_number]
                    coordinate = layout.coordinate(**indices)
                    image_index = (
                        coordinate.get("vr_batch", 0) * VR_LANES + coordinate["vr_lane"]
                    )
                    assert (
                        patterned_routes[value.name][image_index] == bits[logical_index]
                    )


def test_resident_rhs_route_zero_pads_non_power_of_two_payload():
    compiled = allo.compile(
        build_hybrid_program("2mm"),
        build_apu_v1_target(),
        backend="virtual",
    )
    region = next(
        region
        for region in compiled.hybrid_manifest.phases[0].regions
        if region.kind == "vector"
    )
    broadcast_plan = next(
        plan
        for plan in region.plans
        if plan.name == "temporal_dma_coalescing_broadcast_friendly"
    )
    extents = region.compute_analysis.axis_extents
    values = tuple(
        VectorValue(
            layout.value,
            tuple(extents[axis] for axis in layout.axes),
            np.float16,
            "inout" if layout.value == region.compute_analysis.output.value else "in",
        )
        for layout in broadcast_plan.value_layouts
    )
    abi = APUVectorABI(broadcast_plan, values)
    rhs_name = next(
        operand.value for operand in region.operands if operand.role == "rhs"
    )
    arrays = {
        value.name: np.arange(np.prod(value.shape), dtype=np.float16).reshape(
            value.shape
        )
        for value in values
    }
    image = abi.transfer_input_images(arrays)[rhs_name]
    transfer = next(item for item in broadcast_plan.transfers if item.value == rhs_name)
    duplicate = next(
        step for step in transfer.route if step.kind == "duplicate_subgroup"
    )
    rows = int(duplicate.parameters["rows_per_vr"])
    subgroup = int(duplicate.parameters["subgroup_size"])
    payload = arrays[rhs_name].shape[1]
    first_group = image[: rows * subgroup].view(np.float16).reshape(rows, subgroup)

    np.testing.assert_array_equal(first_group[:, :payload], arrays[rhs_name][:rows])
    assert np.count_nonzero(first_group[:, payload:]) == 0


def test_gemm_has_four_shard_local_realizations_with_replicated_rhs():
    compiled = allo.compile(
        build_hybrid_program("gemm"),
        build_apu_v1_target(),
        backend="virtual",
    )
    region = next(
        region
        for phase in compiled.hybrid_manifest.phases
        for region in phase.regions
        if region.kind == "vector"
    )
    partition = partition_apu_v1_region(region)
    shards = realize_apu_v1_region_shards(region, partition)

    assert tuple(item.shard.apuc for item in shards) == (0, 1, 2, 3)
    assert tuple(item.valid_extent for item in shards) == tuple(
        item.extent for item in partition.shards
    )
    lhs = region.compute_analysis.lhs.value
    rhs = region.compute_analysis.rhs.value
    output = region.compute_analysis.output.value
    for item in shards:
        assert item.plan.name == region.selected_plan.name
        assert item.value_slice(lhs)[0] == slice(item.shard.start, item.shard.stop)
        assert item.value_slice(output)[0] == slice(item.shard.start, item.shard.stop)
        assert item.value_slice(rhs) == (slice(None), slice(None))
        shapes = {value.name: value.shape for value in item.artifact.values}
        assert shapes[lhs][0] == item.valid_extent
        assert shapes[output][0] == item.valid_extent
        assert shapes[rhs] == region.compute_analysis.rhs.shape
        assert item.padded_extent >= item.valid_extent

        arrays = {
            value.name: np.ones(value.shape, dtype=value.dtype)
            for value in item.artifact.values
        }
        routed = item.artifact.abi.transfer_input_images(arrays)
        assert set(routed) == {lhs, rhs}
        assert all(image.size % VR_LANES == 0 for image in routed.values())
