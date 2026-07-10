# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Target-independent host/device ABI planning for UPMEM programs.

The cycle simulator and real UPMEM SDK both require every DPU participating in
one collective transfer to use the same offset and transfer size.  This module
turns logical NumPy tensors into equal-sized, 8-byte-aligned MRAM slots while
retaining the exact logical extent owned by each DPU in a compact WRAM metadata
record.  It deliberately contains no simulator invocation logic: MLIR-to-C
lowering and either a real-device or simulator runner can consume the same
``LaunchABI``.

The ABI is versioned and little endian.  Tensor and scalar names are compile-
time information and therefore are not copied to every DPU; descriptors and
scalar values appear in the declaration order of ``LaunchABI.tensors`` and
``LaunchABI.scalars``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
import json
from pathlib import Path
import struct
from typing import Mapping, Sequence

import numpy as np

from ..spmw_linear_layout import LinearLayout


UPMEM_ABI_MAGIC = 0x55414249  # ASCII "UABI" in a uint32_t.
UPMEM_ABI_VERSION = 2
UPMEM_DMA_ALIGNMENT = 8
UPMEM_MRAM_BYTES = 64 * 1024 * 1024

_HEADER = struct.Struct("<8I")
# offset, slot bytes, payload bytes, element bytes, rank, direction, layout,
# partition axis, owned start/extent/stride, transferred start/extent/stride,
# total elements.
_TENSOR_DESCRIPTOR = struct.Struct("<15I")


def _align_up(value: int, alignment: int = UPMEM_DMA_ALIGNMENT) -> int:
    if value < 0 or alignment <= 0:
        raise ValueError("alignment operands must be positive")
    return (value + alignment - 1) // alignment * alignment


class TensorDirection(IntEnum):
    """Whether a launch reads, writes, or updates a logical tensor."""

    INPUT = 1
    OUTPUT = 2
    INOUT = 3


class TensorLayout(IntEnum):
    """How a logical tensor is placed across DPUs."""

    BROADCAST = 1
    BLOCK = 2


@dataclass(frozen=True)
class TensorABI:
    """One tensor argument in a UPMEM kernel launch.

    ``halo`` applies only to ``BLOCK`` layouts. Without a LinearLayout, BLOCK
    uses the legacy balanced contiguous partition. A carried layout may select
    contiguous or strided ownership; halo-bearing layouts must remain
    contiguous. Halos are input data only: gather writes back the owned
    elements and discards halo copies.
    """

    name: str
    shape: tuple[int, ...]
    dtype: np.dtype | str | type
    direction: TensorDirection
    layout: TensorLayout = TensorLayout.BLOCK
    partition_axis: int = 0
    halo: tuple[int, int] = (0, 0)
    linear_layout: LinearLayout | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(int(v) for v in self.shape))
        object.__setattr__(self, "dtype", np.dtype(self.dtype))
        object.__setattr__(self, "direction", TensorDirection(self.direction))
        object.__setattr__(self, "layout", TensorLayout(self.layout))
        if not self.name or not self.name.isidentifier():
            raise ValueError(f"invalid tensor name {self.name!r}")
        if not self.shape or any(v < 0 for v in self.shape):
            raise ValueError(f"tensor {self.name!r} must have a non-negative rank")
        if self.dtype.hasobject:
            raise TypeError(f"tensor {self.name!r} has unsupported object dtype")
        if self.dtype.itemsize not in (1, 2, 4, 8):
            raise TypeError(
                f"tensor {self.name!r} has unsupported {self.dtype.itemsize}-byte elements"
            )
        axis = int(self.partition_axis)
        if axis < 0:
            axis += len(self.shape)
        if not 0 <= axis < len(self.shape):
            raise ValueError(f"tensor {self.name!r} partition axis is out of range")
        object.__setattr__(self, "partition_axis", axis)
        halo = tuple(int(v) for v in self.halo)
        if len(halo) != 2 or any(v < 0 for v in halo):
            raise ValueError(
                f"tensor {self.name!r} halo must be two non-negative values"
            )
        if self.layout == TensorLayout.BROADCAST and halo != (0, 0):
            raise ValueError("broadcast tensors cannot have a halo")
        object.__setattr__(self, "halo", halo)
        linear_layout = self.linear_layout
        if linear_layout is not None:
            if not isinstance(linear_layout, LinearLayout):
                raise TypeError("linear_layout must be a LinearLayout")
            required_outputs = {"dpu", "tasklet", "local"}
            if not required_outputs.issubset(linear_layout.out_dims):
                raise ValueError(
                    f"tensor {self.name!r} layout outputs must contain "
                    f"{sorted(required_outputs)}"
                )
            required_inputs = {"local_partition", "inner"}
            if self.layout == TensorLayout.BROADCAST:
                required_inputs.add("replica")
                owns_dpu = True
            else:
                owns_dpu = bool({"dpu_block", "dpu_lane"} & set(linear_layout.bases))
            if not required_inputs.issubset(linear_layout.bases) or not owns_dpu:
                raise ValueError(
                    f"tensor {self.name!r} layout inputs must contain "
                    f"{sorted(required_inputs)} and a DPU ownership axis"
                )

    @property
    def num_elements(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))


@dataclass(frozen=True)
class ScalarABI:
    """One scalar copied to each DPU's launch metadata."""

    name: str
    dtype: np.dtype | str | type = np.dtype("int32")

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", np.dtype(self.dtype))
        if not self.name or not self.name.isidentifier():
            raise ValueError(f"invalid scalar name {self.name!r}")
        if self.dtype.hasobject or self.dtype.itemsize not in (1, 2, 4, 8):
            raise TypeError(f"scalar {self.name!r} has unsupported dtype {self.dtype}")


@dataclass(frozen=True)
class DPUShard:
    """Logical and transferred strided ranges for one tensor on one DPU."""

    owned_start: int
    owned_extent: int
    transfer_start: int
    transfer_extent: int
    owned_stride: int = 1
    transfer_stride: int = 1

    @property
    def owned_stop(self) -> int:
        return (
            self.owned_start
            if self.owned_extent == 0
            else self.owned_start + (self.owned_extent - 1) * self.owned_stride + 1
        )

    @property
    def transfer_stop(self) -> int:
        return (
            self.transfer_start
            if self.transfer_extent == 0
            else self.transfer_start
            + (self.transfer_extent - 1) * self.transfer_stride
            + 1
        )

    @property
    def owned_local_start(self) -> int:
        return self.owned_start - self.transfer_start


@dataclass(frozen=True)
class TensorSlot:
    """MRAM allocation shared by every DPU for one tensor."""

    tensor: TensorABI
    offset: int
    slot_bytes: int
    shards: tuple[DPUShard, ...]


@dataclass(frozen=True)
class PackedDPU:
    """Bytes supplied to one DPU for one launch."""

    dpu_id: int
    metadata: bytes
    mram: bytes


@dataclass(frozen=True)
class LaunchABI:
    """A complete, statically laid-out UPMEM kernel launch."""

    name: str
    tensors: tuple[TensorABI, ...]
    scalars: tuple[ScalarABI, ...] = ()
    num_dpus: int = 64
    num_tasklets: int = 11
    launch_id: int = 0
    mram_capacity: int = UPMEM_MRAM_BYTES
    slots: tuple[TensorSlot, ...] = field(init=False, repr=False)
    mram_bytes: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tensors", tuple(self.tensors))
        object.__setattr__(self, "scalars", tuple(self.scalars))
        if not self.name or not self.name.isidentifier():
            raise ValueError(f"invalid launch name {self.name!r}")
        if not self.tensors:
            raise ValueError("a launch needs at least one tensor")
        if self.num_dpus <= 0 or self.num_tasklets <= 0:
            raise ValueError("a launch needs at least one DPU and one tasklet")
        if self.launch_id < 0:
            raise ValueError("launch_id must be non-negative")
        names = [v.name for v in self.tensors] + [v.name for v in self.scalars]
        if len(names) != len(set(names)):
            raise ValueError("tensor and scalar names must be unique within a launch")

        offset = 0
        slots: list[TensorSlot] = []
        for tensor in self.tensors:
            shards = tuple(self._shard(tensor, dpu) for dpu in range(self.num_dpus))
            max_payload = max(self._payload_bytes(tensor, shard) for shard in shards)
            slot_bytes = _align_up(max_payload)
            # The host API permits an empty logical shard, but collective MRAM
            # transfers use one common non-zero size.  Reserve one DMA quantum.
            slot_bytes = max(slot_bytes, UPMEM_DMA_ALIGNMENT)
            offset = _align_up(offset)
            slots.append(TensorSlot(tensor, offset, slot_bytes, shards))
            offset += slot_bytes
        if offset > self.mram_capacity:
            raise MemoryError(
                f"launch {self.name!r} needs {offset} MRAM bytes per DPU; "
                f"capacity is {self.mram_capacity}"
            )
        object.__setattr__(self, "slots", tuple(slots))
        object.__setattr__(self, "mram_bytes", offset)

    def _shard(self, tensor: TensorABI, dpu_id: int) -> DPUShard:
        length = tensor.shape[tensor.partition_axis]
        if tensor.layout == TensorLayout.BROADCAST:
            return DPUShard(0, length, 0, length)
        if tensor.linear_layout is not None:
            local_span = tensor.linear_layout.size_of("local_partition")
            striped = "dpu_lane" in tensor.linear_layout.bases
            if striped:
                owned_start = dpu_id
                owned_extent = (
                    0
                    if owned_start >= length
                    else (length - 1 - owned_start) // self.num_dpus + 1
                )
                owned_stride = self.num_dpus
                layout_inputs = {
                    "dpu_lane": dpu_id,
                    "local_partition": 0,
                    "inner": 0,
                }
            else:
                owned_start = dpu_id * local_span
                owned_extent = max(0, min(local_span, length - owned_start))
                owned_stride = 1
                layout_inputs = {
                    "dpu_block": dpu_id,
                    "local_partition": 0,
                    "inner": 0,
                }
            coordinate = dict(
                zip(
                    tensor.linear_layout.out_dims,
                    tensor.linear_layout.apply(**layout_inputs),
                )
            )
            if coordinate["dpu"] != dpu_id:
                raise ValueError(
                    f"tensor {tensor.name!r} LinearLayout maps DPU block "
                    f"{dpu_id} to {coordinate['dpu']}"
                )
            if owned_extent == 0:
                return DPUShard(
                    owned_start,
                    0,
                    owned_start,
                    0,
                    owned_stride,
                    owned_stride,
                )
            if striped:
                if tensor.halo != (0, 0):
                    raise ValueError("striped LinearLayout does not support halos")
                return DPUShard(
                    owned_start,
                    owned_extent,
                    owned_start,
                    owned_extent,
                    owned_stride,
                    owned_stride,
                )
            transfer_start = max(0, owned_start - tensor.halo[0])
            transfer_stop = min(length, owned_start + owned_extent + tensor.halo[1])
            return DPUShard(
                owned_start,
                owned_extent,
                transfer_start,
                transfer_stop - transfer_start,
                owned_stride,
                1,
            )
        quotient, remainder = divmod(length, self.num_dpus)
        owned_extent = quotient + int(dpu_id < remainder)
        owned_start = dpu_id * quotient + min(dpu_id, remainder)
        if owned_extent == 0:
            return DPUShard(owned_start, 0, owned_start, 0)
        transfer_start = max(0, owned_start - tensor.halo[0])
        transfer_stop = min(length, owned_start + owned_extent + tensor.halo[1])
        return DPUShard(
            owned_start,
            owned_extent,
            transfer_start,
            transfer_stop - transfer_start,
        )

    @staticmethod
    def _payload_shape(tensor: TensorABI, shard: DPUShard) -> tuple[int, ...]:
        shape = list(tensor.shape)
        shape[tensor.partition_axis] = shard.transfer_extent
        return tuple(shape)

    @classmethod
    def _payload_bytes(cls, tensor: TensorABI, shard: DPUShard) -> int:
        return int(np.prod(cls._payload_shape(tensor, shard), dtype=np.int64)) * int(
            tensor.dtype.itemsize
        )

    @property
    def metadata_bytes(self) -> int:
        scalar_bytes = sum(_align_up(v.dtype.itemsize) for v in self.scalars)
        return _align_up(
            _HEADER.size + len(self.tensors) * _TENSOR_DESCRIPTOR.size + scalar_bytes
        )

    def tensor_slot(self, name: str) -> TensorSlot:
        for slot in self.slots:
            if slot.tensor.name == name:
                return slot
        raise KeyError(name)

    def tensor_coordinates(
        self,
        name: str,
        logical_index: Sequence[int],
        *,
        dpu_id: int | None = None,
    ) -> dict[str, int]:
        """Map one logical tensor element through its carried LinearLayout."""
        slot = self.tensor_slot(name)
        tensor = slot.tensor
        layout = tensor.linear_layout
        if layout is None:
            raise ValueError(f"tensor {name!r} has no LinearLayout")
        logical_index = tuple(int(value) for value in logical_index)
        if len(logical_index) != len(tensor.shape):
            raise ValueError(
                f"tensor {name!r} index rank {len(logical_index)} != {len(tensor.shape)}"
            )
        if any(
            value < 0 or value >= extent
            for value, extent in zip(logical_index, tensor.shape)
        ):
            raise IndexError(f"tensor {name!r} logical index is out of bounds")

        partition = logical_index[tensor.partition_axis]
        inner = 0
        for axis, (value, extent) in enumerate(zip(logical_index, tensor.shape)):
            if axis == tensor.partition_axis:
                continue
            inner = inner * extent + value
        if tensor.layout == TensorLayout.BROADCAST:
            if dpu_id is None:
                raise ValueError("broadcast coordinate mapping requires dpu_id")
            inputs = {
                "replica": int(dpu_id),
                "local_partition": partition,
                "inner": inner,
            }
        else:
            local_span = layout.size_of("local_partition")
            if "dpu_lane" in layout.bases:
                inputs = {
                    "dpu_lane": partition % self.num_dpus,
                    "local_partition": partition // self.num_dpus,
                    "inner": inner,
                }
            else:
                inputs = {
                    "dpu_block": partition // local_span,
                    "local_partition": partition % local_span,
                    "inner": inner,
                }
        return dict(zip(layout.out_dims, layout.apply(**inputs)))

    def spatial_parallelism(self, name: str) -> int:
        """Number of non-empty DPU owners implied by one tensor layout."""
        slot = self.tensor_slot(name)
        if slot.tensor.layout == TensorLayout.BROADCAST:
            return 1
        return sum(shard.owned_extent > 0 for shard in slot.shards)

    def tasklet_parallelism(self, name: str) -> int:
        """Active tasklet fanout reached by a tensor's local layout axes."""
        tensor = self.tensor_slot(name).tensor
        layout = tensor.linear_layout
        if layout is None:
            return 1
        fanout = layout.image_size(
            varying_inputs=("local_partition", "inner"),
            output_dims=("tasklet",),
        )
        return max(1, min(self.num_tasklets, fanout))

    def pack(
        self,
        arrays: Mapping[str, np.ndarray],
        scalar_values: Mapping[str, object] | None = None,
    ) -> tuple[PackedDPU, ...]:
        """Pack NumPy arguments and scalar values into per-DPU launch images."""

        scalar_values = scalar_values or {}
        normalized: dict[str, np.ndarray] = {}
        for tensor in self.tensors:
            if tensor.direction == TensorDirection.OUTPUT and tensor.name not in arrays:
                normalized[tensor.name] = np.zeros(tensor.shape, dtype=tensor.dtype)
                continue
            if tensor.name not in arrays:
                raise KeyError(f"missing tensor argument {tensor.name!r}")
            value = np.asarray(arrays[tensor.name], dtype=tensor.dtype, order="C")
            if value.shape != tensor.shape:
                raise ValueError(
                    f"tensor {tensor.name!r} has shape {value.shape}; expected {tensor.shape}"
                )
            normalized[tensor.name] = value
        missing_scalars = [v.name for v in self.scalars if v.name not in scalar_values]
        if missing_scalars:
            raise KeyError(f"missing scalar arguments: {', '.join(missing_scalars)}")

        packed: list[PackedDPU] = []
        for dpu_id in range(self.num_dpus):
            mram = bytearray(self.mram_bytes)
            descriptors: list[tuple[int, ...]] = []
            for slot in self.slots:
                tensor = slot.tensor
                shard = slot.shards[dpu_id]
                indices = (
                    shard.transfer_start
                    + np.arange(shard.transfer_extent) * shard.transfer_stride
                )
                payload = np.ascontiguousarray(
                    np.take(
                        normalized[tensor.name],
                        indices,
                        axis=tensor.partition_axis,
                    )
                ).tobytes()
                if len(payload) > slot.slot_bytes:
                    raise AssertionError(
                        "payload exceeds its statically allocated slot"
                    )
                mram[slot.offset : slot.offset + len(payload)] = payload
                descriptors.append(
                    (
                        slot.offset,
                        slot.slot_bytes,
                        len(payload),
                        tensor.dtype.itemsize,
                        len(tensor.shape),
                        int(tensor.direction),
                        int(tensor.layout),
                        tensor.partition_axis,
                        shard.owned_start,
                        shard.owned_extent,
                        shard.owned_stride,
                        shard.transfer_start,
                        shard.transfer_extent,
                        shard.transfer_stride,
                        tensor.num_elements,
                    )
                )
            metadata = bytearray()
            metadata.extend(
                _HEADER.pack(
                    UPMEM_ABI_MAGIC,
                    UPMEM_ABI_VERSION,
                    self.launch_id,
                    dpu_id,
                    self.num_dpus,
                    self.num_tasklets,
                    len(self.tensors),
                    len(self.scalars),
                )
            )
            for descriptor in descriptors:
                metadata.extend(_TENSOR_DESCRIPTOR.pack(*descriptor))
            for scalar in self.scalars:
                raw = (
                    np.asarray(scalar_values[scalar.name], dtype=scalar.dtype)
                    .reshape(())
                    .tobytes()
                )
                metadata.extend(raw)
                metadata.extend(bytes(_align_up(len(raw)) - len(raw)))
            metadata.extend(bytes(self.metadata_bytes - len(metadata)))
            packed.append(PackedDPU(dpu_id, bytes(metadata), bytes(mram)))
        return tuple(packed)

    def gather(
        self,
        dpu_mram: Sequence[bytes | bytearray | memoryview],
        *,
        outputs: Sequence[str] | None = None,
        check_broadcast_replicas: bool = True,
    ) -> dict[str, np.ndarray]:
        """Reconstruct logical output tensors from per-DPU MRAM snapshots."""

        if len(dpu_mram) != self.num_dpus:
            raise ValueError(
                f"expected {self.num_dpus} DPU images, got {len(dpu_mram)}"
            )
        selected = (
            set(outputs)
            if outputs is not None
            else {
                tensor.name
                for tensor in self.tensors
                if tensor.direction in (TensorDirection.OUTPUT, TensorDirection.INOUT)
            }
        )
        unknown = selected - {v.name for v in self.tensors}
        if unknown:
            raise KeyError(f"unknown output tensors: {', '.join(sorted(unknown))}")

        result: dict[str, np.ndarray] = {}
        for slot in self.slots:
            tensor = slot.tensor
            if tensor.name not in selected:
                continue
            value = np.empty(tensor.shape, dtype=tensor.dtype)
            broadcast_reference: np.ndarray | None = None
            for dpu_id, image in enumerate(dpu_mram):
                if len(image) < self.mram_bytes:
                    raise ValueError(
                        f"DPU {dpu_id} image has {len(image)} bytes; expected {self.mram_bytes}"
                    )
                shard = slot.shards[dpu_id]
                payload_bytes = self._payload_bytes(tensor, shard)
                raw = memoryview(image)[slot.offset : slot.offset + payload_bytes]
                local = np.frombuffer(raw, dtype=tensor.dtype).reshape(
                    self._payload_shape(tensor, shard)
                )
                if tensor.layout == TensorLayout.BROADCAST:
                    if broadcast_reference is None:
                        broadcast_reference = local.copy()
                        value[...] = local
                    elif check_broadcast_replicas and not np.array_equal(
                        broadcast_reference, local, equal_nan=True
                    ):
                        raise ValueError(
                            f"broadcast output {tensor.name!r} differs across DPUs"
                        )
                    continue
                if shard.owned_extent == 0:
                    continue
                for offset in range(shard.owned_extent):
                    local_index = [slice(None)] * len(tensor.shape)
                    local_index[tensor.partition_axis] = (
                        shard.owned_local_start + offset
                    )
                    global_index = [slice(None)] * len(tensor.shape)
                    global_index[tensor.partition_axis] = (
                        shard.owned_start + offset * shard.owned_stride
                    )
                    value[tuple(global_index)] = local[tuple(local_index)]
            result[tensor.name] = value
        return result

    def metadata_manifest(self) -> dict[str, object]:
        """Return the compile-time part of the ABI as JSON-compatible data."""

        return {
            "abi": "upmem",
            "version": UPMEM_ABI_VERSION,
            "launch": self.name,
            "launch_id": self.launch_id,
            "num_dpus": self.num_dpus,
            "num_tasklets": self.num_tasklets,
            "metadata_bytes": self.metadata_bytes,
            "mram_bytes_per_dpu": self.mram_bytes,
            "tensors": [
                {
                    "name": slot.tensor.name,
                    "shape": list(slot.tensor.shape),
                    "dtype": slot.tensor.dtype.str,
                    "direction": slot.tensor.direction.name.lower(),
                    "layout": slot.tensor.layout.name.lower(),
                    "partition_axis": slot.tensor.partition_axis,
                    "halo": list(slot.tensor.halo),
                    "linear_layout": (
                        slot.tensor.linear_layout.manifest()
                        if slot.tensor.linear_layout is not None
                        else None
                    ),
                    "offset": slot.offset,
                    "slot_bytes": slot.slot_bytes,
                }
                for slot in self.slots
            ],
            "scalars": [
                {"name": scalar.name, "dtype": scalar.dtype.str}
                for scalar in self.scalars
            ],
        }

    def c_declaration(self, symbol: str = "DPU_INPUT_ARGUMENTS") -> str:
        """Emit the device-C declaration matching :meth:`pack` metadata.

        Scalars remain raw bytes because the 52-byte tensor descriptor can put
        the scalar region at a four-byte boundary.  Generated code should copy
        a scalar's bytes to an aligned local variable before using it.  Keeping
        the wire struct packed makes the Python, simulator, and real SDK layouts
        independent of host-compiler padding rules.
        """

        if not symbol.isidentifier():
            raise ValueError(f"invalid C symbol {symbol!r}")
        lines = [
            "#include <stdint.h>",
            "typedef struct __attribute__((packed)) {",
            "    uint32_t offset;",
            "    uint32_t slot_bytes;",
            "    uint32_t payload_bytes;",
            "    uint32_t element_bytes;",
            "    uint32_t rank;",
            "    uint32_t direction;",
            "    uint32_t layout;",
            "    uint32_t partition_axis;",
            "    uint32_t owned_start;",
            "    uint32_t owned_extent;",
            "    uint32_t owned_stride;",
            "    uint32_t transfer_start;",
            "    uint32_t transfer_extent;",
            "    uint32_t transfer_stride;",
            "    uint32_t total_elements;",
            "} tenon_upmem_tensor_t;",
            "typedef struct __attribute__((packed, aligned(8))) {",
            "    uint32_t magic;",
            "    uint32_t version;",
            "    uint32_t launch_id;",
            "    uint32_t dpu_id;",
            "    uint32_t num_dpus;",
            "    uint32_t num_tasklets;",
            "    uint32_t num_tensors;",
            "    uint32_t num_scalars;",
            f"    tenon_upmem_tensor_t tensors[{len(self.tensors)}];",
        ]
        if self.scalars:
            lines.append(f"    uint8_t scalar_bytes[{8 * len(self.scalars)}];")
        lines.extend(
            [
                f"}} tenon_upmem_launch_{self.name}_t;",
                f"__host tenon_upmem_launch_{self.name}_t {symbol};",
            ]
        )
        lines.extend(
            f"#define TENON_UPMEM_TENSOR_{tensor.name} {index}"
            for index, tensor in enumerate(self.tensors)
        )
        lines.extend(
            f"#define TENON_UPMEM_SCALAR_{scalar.name}_OFFSET {8 * index}"
            for index, scalar in enumerate(self.scalars)
        )
        return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class ProgramABI:
    """Ordered launches with global synchronization between adjacent launches.

    UPMEM DPUs cannot synchronize directly.  A synchronous launch is therefore
    the global barrier: all DPUs finish, optional host transfers/halo exchange
    run, and only then may the next launch begin.  This class records that order
    and validates a stable topology; the runner owns the actual orchestration.
    """

    launches: tuple[LaunchABI, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "launches", tuple(self.launches))
        if not self.launches:
            raise ValueError("a UPMEM program needs at least one launch")
        topology = (self.launches[0].num_dpus, self.launches[0].num_tasklets)
        if any((v.num_dpus, v.num_tasklets) != topology for v in self.launches):
            raise ValueError(
                "all launches in one program must use the same DPU topology"
            )
        ids = [v.launch_id for v in self.launches]
        if len(ids) != len(set(ids)):
            raise ValueError("launch IDs must be unique")

    @property
    def num_dpus(self) -> int:
        return self.launches[0].num_dpus

    @property
    def num_tasklets(self) -> int:
        return self.launches[0].num_tasklets

    def write_manifest(self, path: str | Path) -> None:
        manifest = {
            "abi": "upmem-program",
            "version": UPMEM_ABI_VERSION,
            "synchronization": "host-global-barrier-between-launches",
            "launches": [launch.metadata_manifest() for launch in self.launches],
        }
        Path(path).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


__all__ = [
    "DPUShard",
    "LaunchABI",
    "PackedDPU",
    "ProgramABI",
    "ScalarABI",
    "TensorABI",
    "TensorDirection",
    "TensorLayout",
    "TensorSlot",
    "UPMEM_ABI_MAGIC",
    "UPMEM_ABI_VERSION",
    "UPMEM_DMA_ALIGNMENT",
    "UPMEM_MRAM_BYTES",
]
