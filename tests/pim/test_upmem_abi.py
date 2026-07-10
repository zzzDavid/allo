# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from allo.pim.upmem_abi import (
    LaunchABI,
    ProgramABI,
    ScalarABI,
    TensorABI,
    TensorDirection,
    TensorLayout,
    UPMEM_ABI_MAGIC,
    UPMEM_ABI_VERSION,
)


def _replace_slot(packed, launch, dpu, name, value):
    images = [bytearray(v.mram) for v in packed]
    slot = launch.tensor_slot(name)
    shard = slot.shards[dpu]
    payload_shape = list(slot.tensor.shape)
    payload_shape[slot.tensor.partition_axis] = shard.transfer_extent
    local = np.asarray(value, dtype=slot.tensor.dtype).reshape(payload_shape)
    images[dpu][slot.offset : slot.offset + local.nbytes] = local.tobytes()
    return images


def test_block_partition_is_balanced_padded_and_round_trips():
    tensor = TensorABI(
        "out", (10, 3), np.int32, TensorDirection.INOUT, TensorLayout.BLOCK, 0
    )
    launch = LaunchABI("kernel", (tensor,), num_dpus=4, num_tasklets=11)
    source = np.arange(30, dtype=np.int32).reshape(10, 3)
    packed = launch.pack({"out": source})

    assert [v.owned_extent for v in launch.tensor_slot("out").shards] == [3, 3, 2, 2]
    assert launch.tensor_slot("out").slot_bytes == 40
    assert all(len(v.mram) == launch.mram_bytes for v in packed)
    np.testing.assert_array_equal(
        launch.gather([v.mram for v in packed])["out"], source
    )


def test_nonleading_axis_and_halo_gather_only_owned_elements():
    tensor = TensorABI(
        "grid",
        (2, 9),
        np.int16,
        TensorDirection.INOUT,
        TensorLayout.BLOCK,
        partition_axis=1,
        halo=(1, 2),
    )
    launch = LaunchABI("stencil", (tensor,), num_dpus=4)
    source = np.arange(18, dtype=np.int16).reshape(2, 9)
    packed = launch.pack({"grid": source})
    slot = launch.tensor_slot("grid")
    assert [(v.owned_start, v.owned_extent) for v in slot.shards] == [
        (0, 3),
        (3, 2),
        (5, 2),
        (7, 2),
    ]
    assert [(v.transfer_start, v.transfer_extent) for v in slot.shards] == [
        (0, 5),
        (2, 5),
        (4, 5),
        (6, 3),
    ]

    images = [bytearray(v.mram) for v in packed]
    for dpu, shard in enumerate(slot.shards):
        shape = (2, shard.transfer_extent)
        local = np.frombuffer(
            images[dpu], dtype=np.int16, count=np.prod(shape), offset=slot.offset
        ).reshape(shape)
        local[:] = 100 + dpu
    gathered = launch.gather(images)["grid"]
    expected = np.empty((2, 9), dtype=np.int16)
    expected[:, 0:3] = 100
    expected[:, 3:5] = 101
    expected[:, 5:7] = 102
    expected[:, 7:9] = 103
    np.testing.assert_array_equal(gathered, expected)


def test_more_dpus_than_elements_have_valid_empty_shards():
    tensor = TensorABI("x", (3,), np.uint8, TensorDirection.INOUT)
    launch = LaunchABI("tiny", (tensor,), num_dpus=8)
    source = np.array([4, 5, 6], dtype=np.uint8)
    packed = launch.pack({"x": source})
    assert [v.owned_extent for v in launch.tensor_slot("x").shards] == [
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
    ]
    assert launch.tensor_slot("x").slot_bytes == 8
    np.testing.assert_array_equal(launch.gather([v.mram for v in packed])["x"], source)


def test_broadcast_float_bits_and_scalar_metadata_are_exact():
    tensor = TensorABI(
        "weights", (3,), np.float32, TensorDirection.INOUT, TensorLayout.BROADCAST
    )
    launch = LaunchABI(
        "broadcast",
        (tensor,),
        (ScalarABI("alpha", np.float32), ScalarABI("steps", np.int32)),
        num_dpus=3,
        launch_id=7,
    )
    values = np.array([np.nan, -0.0, np.inf], dtype=np.float32)
    packed = launch.pack({"weights": values}, {"alpha": np.float32(-1.25), "steps": 13})
    assert len({v.mram for v in packed}) == 1
    header = struct.unpack_from("<8I", packed[2].metadata)
    assert header == (UPMEM_ABI_MAGIC, UPMEM_ABI_VERSION, 7, 2, 3, 11, 1, 2)
    gathered = launch.gather([v.mram for v in packed])["weights"]
    np.testing.assert_array_equal(gathered.view(np.uint32), values.view(np.uint32))


def test_broadcast_output_detects_replica_disagreement():
    tensor = TensorABI(
        "state", (2,), np.int32, TensorDirection.OUTPUT, TensorLayout.BROADCAST
    )
    launch = LaunchABI("replicated", (tensor,), num_dpus=2)
    packed = launch.pack({})
    images = _replace_slot(packed, launch, 1, "state", [1, 0])
    with pytest.raises(ValueError, match="differs across DPUs"):
        launch.gather(images)


def test_program_manifest_records_host_barriers(tmp_path):
    tensor = TensorABI("x", (64,), np.int32, TensorDirection.INOUT)
    program = ProgramABI(
        (
            LaunchABI("phase_zero", (tensor,), num_dpus=64, launch_id=0),
            LaunchABI("phase_one", (tensor,), num_dpus=64, launch_id=1),
        )
    )
    path = tmp_path / "abi.json"
    program.write_manifest(path)
    manifest = json.loads(path.read_text())
    assert manifest["synchronization"] == "host-global-barrier-between-launches"
    assert [v["launch"] for v in manifest["launches"]] == [
        "phase_zero",
        "phase_one",
    ]
    assert manifest["launches"][0]["num_dpus"] == 64


def test_device_c_declaration_matches_descriptor_order():
    launch = LaunchABI(
        "phase",
        (
            TensorABI("left", (4,), np.int32, TensorDirection.INPUT),
            TensorABI("out", (4,), np.int32, TensorDirection.OUTPUT),
        ),
        (ScalarABI("step", np.int32),),
    )
    source = launch.c_declaration()
    assert "tenon_upmem_tensor_t tensors[2]" in source
    assert "uint8_t scalar_bytes[8]" in source
    assert "TENON_UPMEM_TENSOR_left 0" in source
    assert "TENON_UPMEM_TENSOR_out 1" in source
    assert "TENON_UPMEM_SCALAR_step_OFFSET 0" in source
    assert "uint32_t owned_stride;" in source
    assert "uint32_t transfer_stride;" in source


def test_rejects_mram_overflow_and_inconsistent_topology():
    tensor = TensorABI(
        "huge", (4096,), np.int32, TensorDirection.INPUT, TensorLayout.BROADCAST
    )
    with pytest.raises(MemoryError, match="MRAM bytes"):
        LaunchABI("overflow", (tensor,), mram_capacity=1024)

    small = TensorABI("x", (4,), np.int32, TensorDirection.INPUT)
    with pytest.raises(ValueError, match="same DPU topology"):
        ProgramABI(
            (
                LaunchABI("a", (small,), num_dpus=4, launch_id=0),
                LaunchABI("b", (small,), num_dpus=8, launch_id=1),
            )
        )
