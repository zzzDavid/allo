# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compiler-owned physical plans for calibrated UPMEM kernels.

The classes in this module describe the byte-exact MRAM image consumed by a
complete UPMEM SDK C translation unit.  They intentionally keep packing,
code generation, and analytical features together: an exporter can construct
inputs from :meth:`manifest` without interpreting the generated C source.

All plans use twelve tasklets.  Their tasklet ownership is represented by a
16-way F2 :class:`~allo.spmw_linear_layout.LinearLayout`; tasklets 12--15 are
explicitly invalid.  This lets schedule search use the existing linear-layout
algebra without pretending that twelve is a power of two.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import operator
from typing import ClassVar

from ..spmw_linear_layout import LinearLayout


UPMEM_ACTIVE_TASKLETS = 12
UPMEM_PADDED_TASKLETS = 16
UPMEM_MIN_DMA_BYTES = 8
UPMEM_MAX_DMA_BYTES = 2048
UPMEM_WRAM_BYTES = 64 * 1024
UPMEM_MRAM_BYTES = 64 * 1024 * 1024


def _positive(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _power_of_two(name: str, value: int) -> int:
    value = _positive(name, value)
    if value & (value - 1):
        raise ValueError(f"{name} must be a power of two")
    return value


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _validate_dma(name: str, size: int) -> int:
    size = _positive(name, size)
    if not UPMEM_MIN_DMA_BYTES <= size <= UPMEM_MAX_DMA_BYTES:
        raise ValueError(
            f"{name} must be between {UPMEM_MIN_DMA_BYTES} and "
            f"{UPMEM_MAX_DMA_BYTES} bytes"
        )
    if size % 8:
        raise ValueError(f"{name} must be 8-byte aligned")
    return size


def _validate_capacity(wram_bytes: int, mram_bytes: int) -> None:
    if wram_bytes > UPMEM_WRAM_BYTES:
        raise ValueError(
            f"physical plan needs {wram_bytes} WRAM bytes; UPMEM limit is "
            f"{UPMEM_WRAM_BYTES}"
        )
    if mram_bytes > UPMEM_MRAM_BYTES:
        raise ValueError(
            f"physical plan needs {mram_bytes} MRAM bytes; UPMEM limit is "
            f"{UPMEM_MRAM_BYTES}"
        )


def _contiguous_region(
    *,
    offset: int,
    size: int,
    dtype: str,
    direction: str,
    logical_shape: tuple[int, ...],
    physical_shape: tuple[int, ...],
    **extra,
) -> dict[str, object]:
    if offset % 8 or size % 8:
        raise ValueError("MRAM region offsets and sizes must be 8-byte aligned")
    return {
        "offset": offset,
        "bytes": size,
        "dtype": dtype,
        "direction": direction,
        "logical_shape": list(logical_shape),
        "physical_shape": list(physical_shape),
        "storage": "contiguous",
        **extra,
    }


def _segmented_view(
    *,
    offset: int,
    payload_bytes: int,
    dtype: str,
    logical_shape: tuple[int, ...],
    physical_shape: tuple[int, ...],
    segment_bytes: int,
    segment_stride_bytes: int,
    segments: int,
    replication: str | None = None,
) -> dict[str, object]:
    result = {
        "offset": offset,
        "bytes": payload_bytes,
        "dtype": dtype,
        "direction": "input",
        "logical_shape": list(logical_shape),
        "physical_shape": list(physical_shape),
        "storage": "segmented-view",
        "segment_bytes": segment_bytes,
        "segment_stride_bytes": segment_stride_bytes,
        "segments": segments,
    }
    if replication is not None:
        result["replication"] = replication
    return result


def _packet_layout(fields: int, lanes: int) -> tuple[LinearLayout, int]:
    """Return an exact contiguous F2 carrier plus its valid field count."""

    fields = _positive("packet fields", fields)
    lanes = _power_of_two("packet lanes", lanes)
    padded_fields = _next_power_of_two(fields)
    layout = LinearLayout(
        bases={
            "field": [
                ((lanes << bit),) for bit in range(padded_fields.bit_length() - 1)
            ],
            "lane": [((1 << bit),) for bit in range(lanes.bit_length() - 1)],
        },
        out_dims=("packet_element",),
        out_sizes=(padded_fields * lanes,),
    )
    return layout, fields


@dataclass(frozen=True)
class MaskedTaskletLayout:
    """Contiguous tasklet blocks carried by a padded 16-way F2 map.

    ``block_elements`` is a power-of-two number of consecutive values owned
    by one tasklet.  The F2 carrier has sixteen tasklet coordinates, while
    :meth:`coordinate` admits only the twelve executable tasklets.
    """

    block_elements: int
    layout: LinearLayout = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        block = _power_of_two("block_elements", self.block_elements)
        tasklet_bases = [
            ((block << bit),) for bit in range(UPMEM_PADDED_TASKLETS.bit_length() - 1)
        ]
        lane_bases = [((1 << bit),) for bit in range(block.bit_length() - 1)]
        object.__setattr__(
            self,
            "layout",
            LinearLayout(
                {"tasklet": tasklet_bases, "lane": lane_bases},
                ("element",),
                (UPMEM_PADDED_TASKLETS * block,),
            ),
        )

    @property
    def active_tasklets(self) -> int:
        return UPMEM_ACTIVE_TASKLETS

    @property
    def padded_tasklets(self) -> int:
        return UPMEM_PADDED_TASKLETS

    def is_valid(self, tasklet: int, lane: int) -> bool:
        return (
            0 <= int(tasklet) < UPMEM_ACTIVE_TASKLETS
            and 0 <= int(lane) < self.block_elements
        )

    def coordinate(self, tasklet: int, lane: int) -> int:
        if not self.is_valid(tasklet, lane):
            raise IndexError("tasklet/lane is outside the masked 12-tasklet domain")
        return self.layout.apply(tasklet=int(tasklet), lane=int(lane))[0]

    def manifest(self) -> dict[str, object]:
        return {
            "kind": "upmem-masked-tasklet-linear-layout",
            "active_tasklets": UPMEM_ACTIVE_TASKLETS,
            "padded_tasklets": UPMEM_PADDED_TASKLETS,
            "block_elements": self.block_elements,
            "validity": {"tasklet": [0, UPMEM_ACTIVE_TASKLETS], "masked": [12, 16]},
            "linear_layout": self.layout.manifest(),
        }


def masked_12_tasklet_linear_layout(block_elements: int) -> LinearLayout:
    """Construct the padded F2 carrier used by twelve-tasklet plans."""

    return MaskedTaskletLayout(block_elements).layout


def _matrix_vector_inputs(
    rows: int,
    columns: int,
    batches: int,
    matrix,
    vectors,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Validate and snapshot flattened integer matrix/vector operands."""

    rows = _positive("rows", rows)
    columns = _positive("columns", columns)
    batches = _positive("batches", batches)

    def snapshot(name: str, values, expected: int) -> tuple[int, ...]:
        try:
            raw_values = tuple(values)
        except TypeError as error:
            raise TypeError(f"{name} must be an iterable of integers") from error
        if len(raw_values) != expected:
            raise ValueError(
                f"{name} has {len(raw_values)} values; expected {expected}"
            )
        result: list[int] = []
        for value in raw_values:
            if isinstance(value, bool):
                raise TypeError(f"{name} values must be integers, not bool")
            try:
                result.append(operator.index(value))
            except TypeError as error:
                raise TypeError(f"{name} values must be integers") from error
        return tuple(result)

    return (
        snapshot("matrix", matrix, batches * rows * columns),
        snapshot("vectors", vectors, batches * columns),
    )


def matrix_vector_row_mul_step_costs(
    rows: int,
    columns: int,
    batches: int,
    matrix,
    vectors,
) -> tuple[int, ...]:
    """Return calibrated ``__mulsi3`` loop steps for each logical MV row.

    uPIMulator's software multiply iterates once per significant bit of the
    smaller *unsigned* operand.  Keeping the conversion explicit here makes
    the input-aware layout model agree with that device behavior for negative
    int32 values as well as positive values.
    """

    matrix_values, vector_values = _matrix_vector_inputs(
        rows, columns, batches, matrix, vectors
    )
    costs: list[int] = []
    for logical_row in range(batches * rows):
        batch = logical_row // rows
        matrix_base = logical_row * columns
        vector_base = batch * columns
        costs.append(
            sum(
                min(
                    matrix_values[matrix_base + column] & 0xFFFFFFFF,
                    vector_values[vector_base + column] & 0xFFFFFFFF,
                ).bit_length()
                for column in range(columns)
            )
        )
    return tuple(costs)


def balance_matrix_vector_rows(
    rows: int,
    columns: int,
    batches: int,
    matrix,
    vectors,
    *,
    phase_rotation: int = 1,
) -> tuple[int, ...]:
    """Build a deterministic physical-to-logical MV row permutation.

    Logical rows are scheduled largest-processing-time first into twelve
    capacity-equal tasklet buckets.  Cost ties use logical-row order; eligible
    buckets are ordered by ``(cost sum, row count, tasklet id)``.  Each bucket
    is then rotated left by ``phase_rotation * tid`` slots before the
    tasklet-major concatenation.

    The current byte-stable device kernel treats physical rows at and beyond
    ``total_rows`` as padding.  Consequently this helper fails closed unless
    all twelve tasklet buckets are full; generic padded plans retain identity
    layout rather than moving a real row into that implicit padded tail.
    """

    rows = _positive("rows", rows)
    columns = _positive("columns", columns)
    batches = _positive("batches", batches)
    if isinstance(phase_rotation, bool):
        raise TypeError("phase_rotation must be an integer")
    try:
        phase_rotation = operator.index(phase_rotation)
    except TypeError as error:
        raise TypeError("phase_rotation must be an integer") from error

    total_rows = batches * rows
    rows_per_tasklet = _next_power_of_two(_ceil_div(total_rows, UPMEM_ACTIVE_TASKLETS))
    padded_rows = UPMEM_ACTIVE_TASKLETS * rows_per_tasklet
    if total_rows != padded_rows:
        raise ValueError(
            "balanced MV row layout requires all twelve capacity-equal "
            "tasklet buckets to be full; use identity layout for padded plans"
        )

    costs = matrix_vector_row_mul_step_costs(rows, columns, batches, matrix, vectors)
    buckets: list[list[int]] = [[] for _ in range(UPMEM_ACTIVE_TASKLETS)]
    loads = [0] * UPMEM_ACTIVE_TASKLETS
    for logical_row in sorted(range(total_rows), key=lambda row: (-costs[row], row)):
        tasklet = min(
            (
                tid
                for tid in range(UPMEM_ACTIVE_TASKLETS)
                if len(buckets[tid]) < rows_per_tasklet
            ),
            key=lambda tid: (loads[tid], len(buckets[tid]), tid),
        )
        buckets[tasklet].append(logical_row)
        loads[tasklet] += costs[logical_row]

    physical_to_logical: list[int] = []
    for tasklet, bucket in enumerate(buckets):
        shift = (phase_rotation * tasklet) % rows_per_tasklet
        physical_to_logical.extend(bucket[shift:] + bucket[:shift])
    return tuple(physical_to_logical)


class UPMEMPhysicalPlan:
    """Small common protocol shared by concrete physical-plan dataclasses."""

    num_tasklets: ClassVar[int] = UPMEM_ACTIVE_TASKLETS
    active_tasklets: ClassVar[int] = UPMEM_ACTIVE_TASKLETS
    compile_flags: ClassVar[tuple[str, ...]] = ("-DNR_TASKLETS=12",)

    @property
    def compiler_tasklets(self) -> int:
        return self.num_tasklets

    @property
    def mram_offsets(self) -> dict[str, int]:
        return {
            name: int(region["offset"]) for name, region in self.mram_regions.items()
        }

    @property
    def output_region(self) -> dict[str, object]:
        return next(iter(self.output_regions.values()))

    def _common_manifest(self) -> dict[str, object]:
        return {
            "num_tasklets": self.num_tasklets,
            "active_tasklets": self.active_tasklets,
            "compile_flags": list(self.compile_flags),
            "mram_image_bytes": self.mram_image_bytes,
            "mram_regions": self.mram_regions,
            "output_regions": self.output_regions,
            "cost_features": self.cost_features(),
            "source_sha256": hashlib.sha256(
                self.device_source().encode("utf-8")
            ).hexdigest(),
            "sdk_complete": True,
            "runtime_mram_inputs": True,
        }


def _source_prelude(title: str) -> str:
    return f"""// {title}
// Complete Tenon UPMEM SDK translation unit; all operands are runtime MRAM data.
#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <stdint.h>

#if NR_TASKLETS != 12
#error "This physical plan must be compiled with NR_TASKLETS=12"
#endif
#define TENON_ACTIVE_TASKLETS 12
"""


@dataclass(frozen=True)
class UPMEMElementwisePlan(UPMEMPhysicalPlan):
    """Streaming int32 add/AXPBY with calibrated fused chunk packing.

    With the default 128-byte operand chunk, fused storage is a sequence of
    ``[A_chunk32, B_chunk32]`` 256-byte records.  Each record is fetched with
    one DMA and produces one 128-byte output chunk.
    """

    elements: int
    operation: str = "add"
    dma_bytes: int = 128
    interleaved: bool = True
    alpha: int = 2
    beta: int = -1
    runtime_coefficients: bool = False

    def __post_init__(self):
        _positive("elements", self.elements)
        _validate_dma("dma_bytes", self.dma_bytes)
        if self.dma_bytes % 4:
            raise ValueError("int32 DMA chunks must contain whole elements")
        if self.operation not in {"add", "axpby"}:
            raise ValueError("operation must be 'add' or 'axpby'")
        for name in ("alpha", "beta"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.interleaved:
            _validate_dma("fused operand DMA", 2 * self.dma_bytes)
        _validate_capacity(self.wram_bytes, self.mram_image_bytes)

    @property
    def chunk_elements(self) -> int:
        return self.dma_bytes // 4

    @property
    def chunks(self) -> int:
        return _ceil_div(self.elements, self.chunk_elements)

    @property
    def chunks_per_tasklet(self) -> int:
        return _next_power_of_two(_ceil_div(self.chunks, UPMEM_ACTIVE_TASKLETS))

    @property
    def padded_chunks(self) -> int:
        return UPMEM_ACTIVE_TASKLETS * self.chunks_per_tasklet

    @property
    def padded_elements(self) -> int:
        return self.padded_chunks * self.chunk_elements

    @property
    def tasklet_layout(self) -> MaskedTaskletLayout:
        return MaskedTaskletLayout(self.chunks_per_tasklet * self.chunk_elements)

    @property
    def packet_layout(self) -> LinearLayout:
        return _packet_layout(2, self.chunk_elements)[0]

    @property
    def wram_bytes(self) -> int:
        return UPMEM_ACTIVE_TASKLETS * 2 * self.dma_bytes + (
            8 if self.operation == "axpby" and self.runtime_coefficients else 0
        )

    @property
    def mram_regions(self) -> dict[str, dict[str, object]]:
        operand_bytes = self.padded_elements * 4
        if self.interleaved:
            packed_bytes = 2 * operand_bytes
            regions = {
                "packed_operands": _contiguous_region(
                    offset=0,
                    size=packed_bytes,
                    dtype="int32",
                    direction="input",
                    logical_shape=(self.chunks, 2, self.chunk_elements),
                    physical_shape=(self.chunks, 2, self.chunk_elements),
                    packet_order=["x", "y"],
                    packet_bytes=2 * self.dma_bytes,
                ),
                "x": _segmented_view(
                    offset=0,
                    payload_bytes=operand_bytes,
                    dtype="int32",
                    logical_shape=(self.elements,),
                    physical_shape=(self.padded_elements,),
                    segment_bytes=self.dma_bytes,
                    segment_stride_bytes=2 * self.dma_bytes,
                    segments=self.chunks,
                ),
                "y": _segmented_view(
                    offset=self.dma_bytes,
                    payload_bytes=operand_bytes,
                    dtype="int32",
                    logical_shape=(self.elements,),
                    physical_shape=(self.padded_elements,),
                    segment_bytes=self.dma_bytes,
                    segment_stride_bytes=2 * self.dma_bytes,
                    segments=self.chunks,
                ),
            }
            cursor = packed_bytes
        else:
            regions = {
                "x": _contiguous_region(
                    offset=0,
                    size=operand_bytes,
                    dtype="int32",
                    direction="input",
                    logical_shape=(self.elements,),
                    physical_shape=(self.padded_elements,),
                ),
                "y": _contiguous_region(
                    offset=operand_bytes,
                    size=operand_bytes,
                    dtype="int32",
                    direction="input",
                    logical_shape=(self.elements,),
                    physical_shape=(self.padded_elements,),
                ),
            }
            cursor = 2 * operand_bytes
        if self.operation == "axpby" and self.runtime_coefficients:
            regions["coefficients"] = _contiguous_region(
                offset=cursor,
                size=8,
                dtype="int32",
                direction="input",
                logical_shape=(2,),
                physical_shape=(2,),
                names=["alpha", "beta"],
            )
            cursor += 8
        regions["output"] = _contiguous_region(
            offset=cursor,
            size=operand_bytes,
            dtype="int32",
            direction="output",
            logical_shape=(self.elements,),
            physical_shape=(self.padded_elements,),
        )
        return regions

    @property
    def output_regions(self) -> dict[str, dict[str, object]]:
        return {"output": self.mram_regions["output"]}

    @property
    def mram_image_bytes(self) -> int:
        region = self.mram_regions["output"]
        return int(region["offset"]) + int(region["bytes"])

    def cost_features(self) -> dict[str, int]:
        coefficient_calls = int(self.operation == "axpby" and self.runtime_coefficients)
        calibrated_shift_sub = (
            self.operation == "axpby"
            and not self.runtime_coefficients
            and (self.alpha, self.beta) == (2, -1)
        )
        general_axpby = self.operation == "axpby" and not calibrated_shift_sub
        reads_per_chunk = 1 if self.interleaved else 2
        read_bytes = 2 * self.chunks * self.dma_bytes + 8 * coefficient_calls
        write_bytes = self.chunks * self.dma_bytes
        read_calls = reads_per_chunk * self.chunks + coefficient_calls
        write_calls = self.chunks
        return {
            "mram_read_calls": read_calls,
            "mram_write_calls": write_calls,
            "mram_calls": read_calls + write_calls,
            "mram_read_bytes": read_bytes,
            "mram_write_bytes": write_bytes,
            "mram_bytes": read_bytes + write_bytes,
            "barriers": coefficient_calls,
            "wram_bytes": self.wram_bytes,
            "int32_adds": self.elements * int(self.operation == "add" or general_axpby),
            "int32_subtracts": self.elements * int(calibrated_shift_sub),
            "int32_shifts": self.elements * int(calibrated_shift_sub),
            "int32_multiplies": 2 * self.elements * int(general_axpby),
        }

    def manifest(self) -> dict[str, object]:
        return {
            "kind": "upmem-streaming-int32-elementwise",
            "operation": self.operation,
            "coefficients": (
                {"alpha": self.alpha, "beta": self.beta, "binding": "compile_time"}
                if self.operation == "axpby" and not self.runtime_coefficients
                else (
                    {"alpha": self.alpha, "beta": self.beta, "binding": "runtime_mram"}
                    if self.operation == "axpby"
                    else None
                )
            ),
            "elements": self.elements,
            "padded_elements": self.padded_elements,
            "operand_dma_bytes": self.dma_bytes,
            "fused_dma_bytes": 2 * self.dma_bytes if self.interleaved else None,
            "chunks": self.chunks,
            "chunks_per_tasklet": self.chunks_per_tasklet,
            "interleaved": self.interleaved,
            "tasklet_layout": self.tasklet_layout.manifest(),
            "packet_layout": self.packet_layout.manifest(),
            **self._common_manifest(),
        }

    def device_source(self) -> str:
        chunk = self.chunk_elements
        regions = self.mram_regions
        output_offset = int(regions["output"]["offset"])
        coefficient_code = ""
        expression = "packet[i] + packet[i + TENON_CHUNK_ELEMENTS]"
        barrier = ""
        if self.operation == "axpby" and self.runtime_coefficients:
            coefficient_offset = int(regions["coefficients"]["offset"])
            coefficient_code = f"""
__dma_aligned int32_t tenon_coefficients[2];
"""
            expression = (
                "tenon_coefficients[0] * packet[i] + "
                "tenon_coefficients[1] * packet[i + TENON_CHUNK_ELEMENTS]"
            )
            barrier = f"""
  if (tid == 0)
    mram_read((const __mram_ptr void *)(uintptr_t)(heap + {coefficient_offset}),
              tenon_coefficients, 8);
  barrier_wait(&tenon_barrier);
"""
        elif self.operation == "axpby":
            if (self.alpha, self.beta) == (2, -1):
                expression = "(packet[i] << 1) - " "packet[i + TENON_CHUNK_ELEMENTS]"
            else:
                expression = (
                    f"({self.alpha} * packet[i]) + "
                    f"({self.beta} * packet[i + TENON_CHUNK_ELEMENTS])"
                )
        if self.interleaved:
            read_code = f"""
    mram_read((const __mram_ptr void *)(uintptr_t)(heap + block * {2 * self.dma_bytes}),
              packet, {2 * self.dma_bytes});
"""
        else:
            y_offset = int(regions["y"]["offset"])
            read_code = f"""
    mram_read((const __mram_ptr void *)(uintptr_t)(heap + block * {self.dma_bytes}),
              packet, {self.dma_bytes});
    mram_read((const __mram_ptr void *)(uintptr_t)(heap + {y_offset} +
                                                   block * {self.dma_bytes}),
              &packet[TENON_CHUNK_ELEMENTS], {self.dma_bytes});
"""
        return (
            _source_prelude("Tenon calibrated fused int32 elementwise plan.")
            + f"""
#define TENON_ELEMENTS {self.elements}u
#define TENON_CHUNKS {self.chunks}u
#define TENON_CHUNKS_PER_TASKLET {self.chunks_per_tasklet}u
#define TENON_CHUNK_ELEMENTS {chunk}u

__dma_aligned int32_t tenon_packets[NR_TASKLETS][2 * TENON_CHUNK_ELEMENTS];
{coefficient_code}BARRIER_INIT(tenon_barrier, NR_TASKLETS);

int main(void) {{
  const uint32_t tid = me();
  const uint32_t heap = (uint32_t)DPU_MRAM_HEAP_POINTER;
{barrier}  int32_t *packet = tenon_packets[tid];
  for (uint32_t local_block = 0; local_block < TENON_CHUNKS_PER_TASKLET;
       ++local_block) {{
    const uint32_t block = tid * TENON_CHUNKS_PER_TASKLET + local_block;
    if (block >= TENON_CHUNKS)
      continue;{read_code}
    for (uint32_t i = 0; i < TENON_CHUNK_ELEMENTS; ++i) {{
      const uint32_t element = block * TENON_CHUNK_ELEMENTS + i;
      packet[i] = element < TENON_ELEMENTS ? {expression} : 0;
    }}
    mram_write(packet,
               (__mram_ptr void *)(uintptr_t)(heap + {output_offset} +
                                               block * {self.dma_bytes}),
               {self.dma_bytes});
  }}
  return 0;
}}
"""
        )


@dataclass(frozen=True)
class UPMEMSumReductionPlan(UPMEMPhysicalPlan):
    """Int32-to-int64 sum using one register accumulator per tasklet."""

    elements: int
    dma_bytes: int = 64

    def __post_init__(self):
        _positive("elements", self.elements)
        _validate_dma("dma_bytes", self.dma_bytes)
        if self.dma_bytes % 4:
            raise ValueError("int32 DMA chunks must contain whole elements")
        _validate_capacity(self.wram_bytes, self.mram_image_bytes)

    @property
    def chunk_elements(self) -> int:
        return self.dma_bytes // 4

    @property
    def chunks(self) -> int:
        return _ceil_div(self.elements, self.chunk_elements)

    @property
    def chunks_per_tasklet(self) -> int:
        return _next_power_of_two(_ceil_div(self.chunks, UPMEM_ACTIVE_TASKLETS))

    @property
    def padded_chunks(self) -> int:
        return UPMEM_ACTIVE_TASKLETS * self.chunks_per_tasklet

    @property
    def padded_elements(self) -> int:
        return self.padded_chunks * self.chunk_elements

    @property
    def tasklet_layout(self) -> MaskedTaskletLayout:
        return MaskedTaskletLayout(self.chunks_per_tasklet * self.chunk_elements)

    @property
    def wram_bytes(self) -> int:
        return UPMEM_ACTIVE_TASKLETS * self.dma_bytes + 8 * (UPMEM_ACTIVE_TASKLETS + 1)

    @property
    def mram_regions(self) -> dict[str, dict[str, object]]:
        input_bytes = self.padded_elements * 4
        return {
            "input": _contiguous_region(
                offset=0,
                size=input_bytes,
                dtype="int32",
                direction="input",
                logical_shape=(self.elements,),
                physical_shape=(self.padded_elements,),
            ),
            "output": _contiguous_region(
                offset=input_bytes,
                size=8,
                dtype="int64",
                direction="output",
                logical_shape=(1,),
                physical_shape=(1,),
            ),
        }

    @property
    def output_regions(self) -> dict[str, dict[str, object]]:
        return {"output": self.mram_regions["output"]}

    @property
    def mram_image_bytes(self) -> int:
        return self.padded_elements * 4 + 8

    def cost_features(self) -> dict[str, int]:
        return {
            "mram_read_calls": self.chunks,
            "mram_write_calls": 1,
            "mram_calls": self.chunks + 1,
            "mram_read_bytes": self.chunks * self.dma_bytes,
            "mram_write_bytes": 8,
            "mram_bytes": self.chunks * self.dma_bytes + 8,
            "barriers": 1,
            "wram_bytes": self.wram_bytes,
            "int64_adds": self.elements + UPMEM_ACTIVE_TASKLETS,
            "partial_wram_stores": UPMEM_ACTIVE_TASKLETS,
        }

    def manifest(self) -> dict[str, object]:
        return {
            "kind": "upmem-int32-int64-sum-reduction",
            "elements": self.elements,
            "padded_elements": self.padded_elements,
            "dma_bytes": self.dma_bytes,
            "chunk_elements": self.chunk_elements,
            "chunks": self.chunks,
            "chunks_per_tasklet": self.chunks_per_tasklet,
            "reduction": "register-accumulate/wram-partial/barrier/serial-merge",
            "tasklet_layout": self.tasklet_layout.manifest(),
            **self._common_manifest(),
        }

    def device_source(self) -> str:
        output_offset = int(self.mram_regions["output"]["offset"])
        return (
            _source_prelude("Tenon calibrated int32-to-int64 reduction plan.")
            + f"""
#define TENON_ELEMENTS {self.elements}u
#define TENON_CHUNKS {self.chunks}u
#define TENON_CHUNKS_PER_TASKLET {self.chunks_per_tasklet}u
#define TENON_CHUNK_ELEMENTS {self.chunk_elements}u

__dma_aligned int32_t tenon_values[NR_TASKLETS][TENON_CHUNK_ELEMENTS];
__dma_aligned int64_t tenon_partials[NR_TASKLETS];
__dma_aligned int64_t tenon_result[1];
BARRIER_INIT(tenon_barrier, NR_TASKLETS);

int main(void) {{
  const uint32_t tid = me();
  const uint32_t heap = (uint32_t)DPU_MRAM_HEAP_POINTER;
  int64_t accumulator = 0;
  for (uint32_t local_block = 0; local_block < TENON_CHUNKS_PER_TASKLET;
       ++local_block) {{
    const uint32_t block = tid * TENON_CHUNKS_PER_TASKLET + local_block;
    if (block >= TENON_CHUNKS)
      continue;
    mram_read((const __mram_ptr void *)(uintptr_t)(heap + block * {self.dma_bytes}),
              tenon_values[tid], {self.dma_bytes});
    for (uint32_t i = 0; i < TENON_CHUNK_ELEMENTS; ++i) {{
      const uint32_t element = block * TENON_CHUNK_ELEMENTS + i;
      if (element < TENON_ELEMENTS)
        accumulator += (int64_t)tenon_values[tid][i];
    }}
  }}
  tenon_partials[tid] = accumulator;
  barrier_wait(&tenon_barrier);
  if (tid == 0) {{
    int64_t total = 0;
    for (uint32_t tasklet = 0; tasklet < TENON_ACTIVE_TASKLETS; ++tasklet)
      total += tenon_partials[tasklet];
    tenon_result[0] = total;
    mram_write(tenon_result,
               (__mram_ptr void *)(uintptr_t)(heap + {output_offset}), 8);
  }}
  return 0;
}}
"""
        )


@dataclass(frozen=True)
class UPMEMMatrixVectorPlan(UPMEMPhysicalPlan):
    """Batched row-owned int32 MV with a fail-closed physical vector mode.

    The generic path uses replicated ``[A16, x16]`` packets.  The calibrated
    MMTV fast path keeps one batch vector in each owning tasklet's private
    WRAM and uses a separate contiguous ``A, B, output`` heap ABI.
    """

    rows: int
    columns: int
    batches: int = 1
    scale: bool = False
    scale_factor: int | None = None
    dma_bytes: int = 64
    vector_mode: str = "fused_replicated"
    physical_to_logical_rows: tuple[int, ...] | None = None

    def __post_init__(self):
        _positive("rows", self.rows)
        _positive("columns", self.columns)
        _positive("batches", self.batches)
        _validate_dma("operand dma_bytes", self.dma_bytes)
        if self.dma_bytes % 4:
            raise ValueError("int32 DMA chunks must contain whole elements")
        _power_of_two("chunk_elements", self.chunk_elements)
        if self.vector_mode not in {"fused_replicated", "tasklet_private_wram"}:
            raise ValueError(
                "MVP supports vector_mode='fused_replicated' or "
                "'tasklet_private_wram'"
            )
        if self.vector_mode == "tasklet_private_wram" and (
            (self.batches, self.rows, self.columns, self.dma_bytes)
            != (UPMEM_ACTIVE_TASKLETS, 16, 32, 64)
            or self.effective_scale_factor != 1
        ):
            raise ValueError(
                "tasklet_private_wram MV is calibrated only for 12 batches, "
                "16 rows per batch, 32 columns, 64-byte matrix chunks, and "
                "unit scale"
            )
        if self.scale_factor is not None and (
            isinstance(self.scale_factor, bool)
            or not isinstance(self.scale_factor, int)
        ):
            raise TypeError("scale_factor must be an integer or None")
        permutation = self.physical_to_logical_rows
        if permutation is None:
            permutation = tuple(range(self.total_rows))
        else:
            try:
                raw_permutation = tuple(permutation)
            except TypeError as error:
                raise TypeError(
                    "physical_to_logical_rows must be an iterable of integers"
                ) from error
            normalized: list[int] = []
            for logical_row in raw_permutation:
                if isinstance(logical_row, bool):
                    raise TypeError(
                        "physical_to_logical_rows values must be integers, not bool"
                    )
                try:
                    normalized.append(operator.index(logical_row))
                except TypeError as error:
                    raise TypeError(
                        "physical_to_logical_rows values must be integers"
                    ) from error
            permutation = tuple(normalized)
        if len(permutation) != self.total_rows:
            raise ValueError(
                "physical_to_logical_rows must contain exactly one entry per "
                f"logical row ({self.total_rows})"
            )
        if set(permutation) != set(range(self.total_rows)):
            raise ValueError(
                "physical_to_logical_rows must be a permutation of "
                f"range({self.total_rows})"
            )
        object.__setattr__(self, "physical_to_logical_rows", permutation)
        if self.vector_mode == "tasklet_private_wram" and not (
            permutation == tuple(range(self.total_rows))
        ):
            raise ValueError(
                "tasklet_private_wram MV requires value-independent identity "
                "row order"
            )
        if self.vector_mode == "fused_replicated":
            _validate_dma("fused matrix/vector DMA", 2 * self.dma_bytes)
        _validate_dma("grouped MV output DMA", self.output_dma_bytes)
        _validate_capacity(self.wram_bytes, self.mram_image_bytes)

    @property
    def chunk_elements(self) -> int:
        return self.dma_bytes // 4

    @property
    def column_chunks(self) -> int:
        return _ceil_div(self.columns, self.chunk_elements)

    @property
    def padded_columns(self) -> int:
        return self.column_chunks * self.chunk_elements

    @property
    def total_rows(self) -> int:
        return self.batches * self.rows

    @property
    def rows_per_tasklet(self) -> int:
        return _next_power_of_two(_ceil_div(self.total_rows, UPMEM_ACTIVE_TASKLETS))

    @property
    def padded_rows(self) -> int:
        return UPMEM_ACTIVE_TASKLETS * self.rows_per_tasklet

    @property
    def output_dma_bytes(self) -> int:
        return self.rows_per_tasklet * 4

    @property
    def effective_scale_factor(self) -> int:
        if self.scale_factor is not None:
            return self.scale_factor
        return 2 if self.scale else 1

    @property
    def tasklet_layout(self) -> MaskedTaskletLayout:
        return MaskedTaskletLayout(self.rows_per_tasklet)

    @property
    def logical_to_physical_rows(self) -> tuple[int, ...]:
        inverse = [0] * self.total_rows
        for physical_row, logical_row in enumerate(self.physical_to_logical_rows):
            inverse[logical_row] = physical_row
        return tuple(inverse)

    @property
    def is_identity_row_layout(self) -> bool:
        return self.physical_to_logical_rows == tuple(range(self.total_rows))

    @property
    def row_layout_manifest(self) -> dict[str, object]:
        return {
            "kind": "upmem-masked-f2-tasklet-row-permutation",
            "composition": (
                "physical_row = carrier(tasklet, local_row); "
                "logical_row = physical_to_logical_rows[physical_row]"
            ),
            "carrier": self.tasklet_layout.manifest(),
            "physical_to_logical_rows": list(self.physical_to_logical_rows),
            "logical_to_physical_rows": list(self.logical_to_physical_rows),
            "logical_rows": self.total_rows,
            "physical_rows": self.padded_rows,
            "identity": self.is_identity_row_layout,
            "padding": {
                "physical_rows": [self.total_rows, self.padded_rows],
                "policy": "implicit-fixed-zero-tail",
            },
        }

    @property
    def packet_layout(self) -> LinearLayout:
        return _packet_layout(2, self.chunk_elements)[0]

    @property
    def packet_bytes(self) -> int:
        return 2 * self.dma_bytes

    @property
    def packet_count(self) -> int:
        return self.padded_rows * self.column_chunks

    @property
    def wram_bytes(self) -> int:
        if self.vector_mode == "tasklet_private_wram":
            return UPMEM_ACTIVE_TASKLETS * (
                self.columns * 4 + self.dma_bytes + self.output_dma_bytes
            )
        return UPMEM_ACTIVE_TASKLETS * (self.packet_bytes + self.output_dma_bytes)

    @property
    def mram_regions(self) -> dict[str, dict[str, object]]:
        if self.vector_mode == "tasklet_private_wram":
            matrix_bytes = self.total_rows * self.columns * 4
            vector_bytes = self.batches * self.columns * 4
            output_bytes = self.total_rows * 4
            return {
                "matrix": _contiguous_region(
                    offset=0,
                    size=matrix_bytes,
                    dtype="int32",
                    direction="input",
                    logical_shape=(self.batches, self.rows, self.columns),
                    physical_shape=(self.batches, self.rows, self.columns),
                    flattened_order="batch-major-row-major-column-major",
                ),
                "vector": _contiguous_region(
                    offset=matrix_bytes,
                    size=vector_bytes,
                    dtype="int32",
                    direction="input",
                    logical_shape=(self.batches, self.columns),
                    physical_shape=(self.batches, self.columns),
                    flattened_order="batch-major-column-major",
                    residency="tasklet-private-wram-one-vector-per-owning-batch",
                ),
                "output": _contiguous_region(
                    offset=matrix_bytes + vector_bytes,
                    size=output_bytes,
                    dtype="int32",
                    direction="output",
                    logical_shape=(self.batches, self.rows),
                    physical_shape=(self.batches, self.rows),
                    flattened_order="batch-major-row-major",
                    row_layout="manifest.row_layout",
                    tasklet_group_bytes=self.output_dma_bytes,
                ),
            }
        packets_bytes = self.packet_count * self.packet_bytes
        regions = {
            "packed_matrix_vector": _contiguous_region(
                offset=0,
                size=packets_bytes,
                dtype="int32",
                direction="input",
                logical_shape=(
                    self.total_rows,
                    self.column_chunks,
                    2,
                    self.chunk_elements,
                ),
                physical_shape=(
                    self.padded_rows,
                    self.column_chunks,
                    2,
                    self.chunk_elements,
                ),
                packet_order=["matrix", "vector"],
                packet_bytes=self.packet_bytes,
                physical_row_order="tasklet-major-local-row-major",
                row_layout="manifest.row_layout",
            ),
            "matrix": _segmented_view(
                offset=0,
                payload_bytes=self.padded_rows * self.padded_columns * 4,
                dtype="int32",
                logical_shape=(self.batches, self.rows, self.columns),
                physical_shape=(self.padded_rows, self.padded_columns),
                segment_bytes=self.dma_bytes,
                segment_stride_bytes=self.packet_bytes,
                segments=self.packet_count,
            ),
            "vector": _segmented_view(
                offset=self.dma_bytes,
                payload_bytes=self.packet_count * self.dma_bytes,
                dtype="int32",
                logical_shape=(self.batches, self.columns),
                physical_shape=(self.padded_rows, self.padded_columns),
                segment_bytes=self.dma_bytes,
                segment_stride_bytes=self.packet_bytes,
                segments=self.packet_count,
                replication="once per batch/output-row/reduction-chunk",
            ),
        }
        output_bytes = self.padded_rows * 4
        regions["output"] = _contiguous_region(
            offset=packets_bytes,
            size=output_bytes,
            dtype="int32",
            direction="output",
            logical_shape=(self.batches, self.rows),
            physical_shape=(self.padded_rows,),
            flattened_order=(
                "batch-major-row-major"
                if self.is_identity_row_layout
                else "physical-tasklet-slot-order"
            ),
            logical_flattened_order="batch-major-row-major",
            row_layout="manifest.row_layout",
            tasklet_group_bytes=self.output_dma_bytes,
        )
        return regions

    @property
    def output_regions(self) -> dict[str, dict[str, object]]:
        return {"output": self.mram_regions["output"]}

    @property
    def mram_image_bytes(self) -> int:
        output = self.mram_regions["output"]
        return int(output["offset"]) + int(output["bytes"])

    def row_mul_step_costs(self, matrix, vectors) -> tuple[int, ...]:
        """Return input-aware calibrated multiply cost in logical-row order."""

        return matrix_vector_row_mul_step_costs(
            self.rows, self.columns, self.batches, matrix, vectors
        )

    def pack_fused_inputs(self, matrix, vectors) -> tuple[int, ...]:
        """Pack physical ``[matrix_chunk, vector_chunk]`` int32 records."""

        if self.vector_mode != "fused_replicated":
            raise ValueError("pack_fused_inputs requires fused_replicated mode")

        matrix_values, vector_values = _matrix_vector_inputs(
            self.rows, self.columns, self.batches, matrix, vectors
        )
        values: list[int] = []
        for physical_row in range(self.padded_rows):
            valid_row = physical_row < self.total_rows
            logical_row = (
                self.physical_to_logical_rows[physical_row] if valid_row else 0
            )
            batch = logical_row // self.rows
            for block in range(self.column_chunks):
                column_begin = block * self.chunk_elements
                for lane in range(self.chunk_elements):
                    column = column_begin + lane
                    values.append(
                        matrix_values[logical_row * self.columns + column]
                        if valid_row and column < self.columns
                        else 0
                    )
                for lane in range(self.chunk_elements):
                    column = column_begin + lane
                    values.append(
                        vector_values[batch * self.columns + column]
                        if valid_row and column < self.columns
                        else 0
                    )
        return tuple(values)

    def reference_outputs(self, matrix, vectors) -> tuple[int, ...]:
        """Evaluate logical batch-major MV outputs with the specialized scale."""

        matrix_values, vector_values = _matrix_vector_inputs(
            self.rows, self.columns, self.batches, matrix, vectors
        )
        return tuple(
            self.effective_scale_factor
            * sum(
                matrix_values[logical_row * self.columns + column]
                * vector_values[(logical_row // self.rows) * self.columns + column]
                for column in range(self.columns)
            )
            for logical_row in range(self.total_rows)
        )

    def pack_output(self, logical_values) -> tuple[int, ...]:
        """Scatter logical batch-major outputs into physical tasklet slots."""

        values, _ = _matrix_vector_inputs(self.total_rows, 1, 1, logical_values, (0,))
        physical = [0] * self.padded_rows
        for physical_row, logical_row in enumerate(self.physical_to_logical_rows):
            physical[physical_row] = values[logical_row]
        return tuple(physical)

    def gather_output(self, physical_values) -> tuple[int, ...]:
        """Gather a complete physical output region into logical row order."""

        try:
            raw_values = tuple(physical_values)
        except TypeError as error:
            raise TypeError(
                "physical output must be an iterable of integers"
            ) from error
        if len(raw_values) != self.padded_rows:
            raise ValueError(
                f"physical output has {len(raw_values)} values; "
                f"expected {self.padded_rows}"
            )
        values: list[int] = []
        for value in raw_values:
            if isinstance(value, bool):
                raise TypeError("physical output values must be integers, not bool")
            try:
                values.append(operator.index(value))
            except TypeError as error:
                raise TypeError("physical output values must be integers") from error
        return tuple(values[row] for row in self.logical_to_physical_rows)

    def cost_features(self) -> dict[str, int]:
        if self.vector_mode == "tasklet_private_wram":
            matrix_bytes = self.total_rows * self.columns * 4
            vector_bytes = self.batches * self.columns * 4
            read_calls = (
                self.total_rows * self.column_chunks + self.batches * self.column_chunks
            )
            write_calls = self.batches
            read_bytes = matrix_bytes + vector_bytes
            write_bytes = self.total_rows * 4
            return {
                "mram_read_calls": read_calls,
                "mram_write_calls": write_calls,
                "mram_calls": read_calls + write_calls,
                "mram_read_bytes": read_bytes,
                "mram_write_bytes": write_bytes,
                "mram_bytes": read_bytes + write_bytes,
                "barriers": 1,
                "wram_bytes": self.wram_bytes,
                "int32_macs": self.total_rows * self.columns,
                "int32_multiplies": 0,
                "wram_allocation_calls": 3 * UPMEM_ACTIVE_TASKLETS,
                "wram_allocation_bytes": self.wram_bytes,
                "static_wram_bytes": 0,
                "mem_reset_calls": 1,
            }
        # Invalid padded rows are written as deterministic zero records but do
        # not fetch input packets.
        compute_packets = self.total_rows * self.column_chunks
        read_calls = compute_packets
        write_calls = UPMEM_ACTIVE_TASKLETS
        read_bytes = compute_packets * self.packet_bytes
        write_bytes = self.padded_rows * 4
        return {
            "mram_read_calls": read_calls,
            "mram_write_calls": write_calls,
            "mram_calls": read_calls + write_calls,
            "mram_read_bytes": read_bytes,
            "mram_write_bytes": write_bytes,
            "mram_bytes": read_bytes + write_bytes,
            "barriers": 0,
            "wram_bytes": self.wram_bytes,
            "int32_macs": self.total_rows * self.columns,
            "int32_multiplies": self.total_rows * int(self.effective_scale_factor != 1),
        }

    def manifest(self) -> dict[str, object]:
        if self.vector_mode == "tasklet_private_wram":
            return {
                "kind": "upmem-batch-resident-int32-matrix-vector",
                "shape": [self.batches, self.rows, self.columns],
                "scale": self.scale,
                "scale_factor": self.effective_scale_factor,
                "layout": "separate",
                "vector_mode": self.vector_mode,
                "vector_residency": "tasklet_private_wram",
                "operand_dma_bytes": self.dma_bytes,
                "matrix_chunk_elements": self.chunk_elements,
                "matrix_chunks_per_row": self.column_chunks,
                "flattened_rows": self.total_rows,
                "padded_shape": [self.padded_rows, self.padded_columns],
                "rows_per_tasklet": self.rows_per_tasklet,
                "output_dma_bytes": self.output_dma_bytes,
                "tasklet_layout": self.tasklet_layout.manifest(),
                "row_layout": self.row_layout_manifest,
                "row_ownership": "tasklet-id-is-batch-id",
                "mram_addressing": "heap-relative-typed-bases",
                "heap_abi_order": ["matrix", "vector", "output"],
                "wram_allocation": {
                    "kind": "per-tasklet-mem-alloc",
                    "calls_per_tasklet": 3,
                    "vector_bytes_per_tasklet": self.columns * 4,
                    "matrix_chunk_bytes_per_tasklet": self.dma_bytes,
                    "output_bytes_per_tasklet": self.output_dma_bytes,
                    "total_payload_bytes": self.wram_bytes,
                },
                **self._common_manifest(),
            }
        return {
            "kind": "upmem-fused-replicated-int32-matrix-vector",
            "shape": [self.batches, self.rows, self.columns],
            "scale": self.scale,
            "scale_factor": self.effective_scale_factor,
            "vector_mode": self.vector_mode,
            "operand_dma_bytes": self.dma_bytes,
            "packet_bytes": self.packet_bytes,
            "packet_order": ["matrix_chunk", "vector_chunk"],
            "flattened_rows": self.total_rows,
            "padded_shape": [self.padded_rows, self.padded_columns],
            "rows_per_tasklet": self.rows_per_tasklet,
            "output_dma_bytes": self.output_dma_bytes,
            "tasklet_layout": self.tasklet_layout.manifest(),
            "row_layout": self.row_layout_manifest,
            "packet_layout": self.packet_layout.manifest(),
            **self._common_manifest(),
        }

    def device_source(self) -> str:
        if self.vector_mode == "tasklet_private_wram":
            return self._tasklet_private_wram_device_source()
        regions = self.mram_regions
        output_offset = int(regions["output"]["offset"])
        scale_expression = (
            "accumulator"
            if self.effective_scale_factor == 1
            else f"{self.effective_scale_factor} * accumulator"
        )
        return (
            _source_prelude("Tenon calibrated fused-replicated int32 MV plan.")
            + f"""
#define TENON_TOTAL_ROWS {self.total_rows}u
#define TENON_COLUMN_CHUNKS {self.column_chunks}u
#define TENON_CHUNK_ELEMENTS {self.chunk_elements}u
#define TENON_ROWS_PER_TASKLET {self.rows_per_tasklet}u
#define TENON_PADDED_ROWS {self.padded_rows}u
#define TENON_PACKET_BYTES {self.packet_bytes}u

__dma_aligned int32_t tenon_packets[NR_TASKLETS][2 * TENON_CHUNK_ELEMENTS];
__dma_aligned int32_t tenon_outputs[NR_TASKLETS][TENON_ROWS_PER_TASKLET];

int main(void) {{
  const uint32_t tid = me();
  const uint32_t heap = (uint32_t)DPU_MRAM_HEAP_POINTER;
  int32_t *packet = tenon_packets[tid];
  int32_t *result = tenon_outputs[tid];
  for (uint32_t local_row = 0; local_row < TENON_ROWS_PER_TASKLET;
       ++local_row) {{
    const uint32_t row = tid * TENON_ROWS_PER_TASKLET + local_row;
    int32_t accumulator = 0;
#if TENON_PADDED_ROWS != TENON_TOTAL_ROWS
    if (row >= TENON_TOTAL_ROWS) {{
      result[local_row] = 0;
      continue;
    }}
#endif
    for (uint32_t block = 0; block < TENON_COLUMN_CHUNKS; ++block) {{
      const uint32_t packet_id = row * TENON_COLUMN_CHUNKS + block;
      mram_read((const __mram_ptr void *)(uintptr_t)
                    (heap + packet_id * TENON_PACKET_BYTES),
                packet, TENON_PACKET_BYTES);
      for (uint32_t i = 0; i < TENON_CHUNK_ELEMENTS; ++i)
        accumulator += packet[i] * packet[TENON_CHUNK_ELEMENTS + i];
    }}
    accumulator = {scale_expression};
    result[local_row] = accumulator;
  }}
  mram_write(result,
             (__mram_ptr void *)(uintptr_t)(heap + {output_offset} +
                                             tid * {self.output_dma_bytes}),
             {self.output_dma_bytes});
  return 0;
}}
"""
        )

    def _tasklet_private_wram_device_source(self) -> str:
        """Emit the exact one-batch-per-tasklet MMTV resident-vector path."""

        regions = self.mram_regions
        vector_offset = int(regions["vector"]["offset"])
        output_offset = int(regions["output"]["offset"])
        return (
            _source_prelude("Tenon value-independent batch-resident int32 MMTV plan.")
            + f"""
#define TENON_BATCHES {self.batches}u
#define TENON_ROWS_PER_BATCH {self.rows}u
#define TENON_COLUMNS {self.columns}u
#define TENON_CHUNK_ELEMENTS {self.chunk_elements}u
#define TENON_CHUNK_BYTES {self.dma_bytes}u
#define TENON_COLUMN_CHUNKS {self.column_chunks}u
#define TENON_OUTPUT_BYTES {self.output_dma_bytes}u

BARRIER_INIT(tenon_barrier, NR_TASKLETS);

int main(void) {{
  const uint32_t tid = me();
  const uint32_t heap = (uint32_t)DPU_MRAM_HEAP_POINTER;
  if (tid == 0)
    mem_reset();
  barrier_wait(&tenon_barrier);

  const __mram_ptr int32_t *matrix =
      (const __mram_ptr int32_t *)(uintptr_t)heap;
  const __mram_ptr int32_t *vectors =
      (const __mram_ptr int32_t *)(uintptr_t)(heap + {vector_offset}u);
  __mram_ptr int32_t *outputs =
      (__mram_ptr int32_t *)(uintptr_t)(heap + {output_offset}u);
  int32_t *vector = (int32_t *)mem_alloc(TENON_COLUMNS * sizeof(int32_t));
  int32_t *matrix_chunk = (int32_t *)mem_alloc(TENON_CHUNK_BYTES);
  int32_t *output = (int32_t *)mem_alloc(TENON_OUTPUT_BYTES);

  for (uint32_t block = 0; block < TENON_COLUMN_CHUNKS; ++block)
    mram_read((const __mram_ptr void *)
                  &vectors[tid * TENON_COLUMNS +
                           block * TENON_CHUNK_ELEMENTS],
              vector + block * TENON_CHUNK_ELEMENTS, TENON_CHUNK_BYTES);
  for (uint32_t row = 0; row < TENON_ROWS_PER_BATCH; ++row) {{
    int32_t accumulator = 0;
    for (uint32_t block = 0; block < TENON_COLUMN_CHUNKS; ++block) {{
      mram_read((const __mram_ptr void *)
                    &matrix[(tid * TENON_ROWS_PER_BATCH + row) * TENON_COLUMNS +
                            block * TENON_CHUNK_ELEMENTS],
                matrix_chunk, TENON_CHUNK_BYTES);
      for (uint32_t column = 0; column < TENON_CHUNK_ELEMENTS; ++column)
        accumulator += matrix_chunk[column] *
                       vector[block * TENON_CHUNK_ELEMENTS + column];
    }}
    output[row] = accumulator;
  }}
  mram_write(output,
             (__mram_ptr void *)
                 &outputs[tid * TENON_ROWS_PER_BATCH],
             TENON_OUTPUT_BYTES);
  return 0;
}}
"""
        )


@dataclass(frozen=True)
class UPMEMGEMMPlan(UPMEMPhysicalPlan):
    """Row-owned int32 GEMM with calibrated private fused A/RHS packets.

    A packet is ``[A[Kc], B_col0[Kc], ..., B_colNc-1[Kc]]``.  The default
    ``Kc=Nc=16`` packet therefore contains 272 int32 values (1088 bytes), and
    each packet produces one 64-byte row-major output tile.
    """

    rows: int
    columns: int
    reduction: int
    column_tile: int = 16
    reduction_tile: int = 16
    rhs_mode: str = "fused_replicated"

    def __post_init__(self):
        _positive("rows", self.rows)
        _positive("columns", self.columns)
        _positive("reduction", self.reduction)
        _power_of_two("column_tile", self.column_tile)
        _power_of_two("reduction_tile", self.reduction_tile)
        if self.rhs_mode != "fused_replicated":
            raise ValueError("MVP supports rhs_mode='fused_replicated'")
        _validate_dma("GEMM fused packet", self.packet_bytes)
        _validate_dma("GEMM output tile", self.output_tile_bytes)
        _validate_capacity(self.wram_bytes, self.mram_image_bytes)

    @property
    def rows_per_tasklet(self) -> int:
        return _next_power_of_two(_ceil_div(self.rows, UPMEM_ACTIVE_TASKLETS))

    @property
    def padded_rows(self) -> int:
        return UPMEM_ACTIVE_TASKLETS * self.rows_per_tasklet

    @property
    def column_tiles(self) -> int:
        return _ceil_div(self.columns, self.column_tile)

    @property
    def reduction_tiles(self) -> int:
        return _ceil_div(self.reduction, self.reduction_tile)

    @property
    def padded_columns(self) -> int:
        return self.column_tiles * self.column_tile

    @property
    def padded_reduction(self) -> int:
        return self.reduction_tiles * self.reduction_tile

    @property
    def packet_elements(self) -> int:
        return (1 + self.column_tile) * self.reduction_tile

    @property
    def packet_bytes(self) -> int:
        return self.packet_elements * 4

    @property
    def output_tile_bytes(self) -> int:
        return self.column_tile * 4

    @property
    def packet_count(self) -> int:
        return self.padded_rows * self.column_tiles * self.reduction_tiles

    @property
    def exact_row_fast_path(self) -> bool:
        """Whether every tasklet owns exactly one real output row."""

        return (
            self.rows == UPMEM_ACTIVE_TASKLETS
            and self.padded_rows == self.rows
            and self.rows_per_tasklet == 1
        )

    @property
    def codegen_path(self) -> str:
        return (
            "tasklet-private-mem-alloc-exact-row"
            if self.exact_row_fast_path
            else "static-wram-generic-row-loop"
        )

    @property
    def tasklet_layout(self) -> MaskedTaskletLayout:
        return MaskedTaskletLayout(self.rows_per_tasklet)

    @property
    def packet_layout(self) -> LinearLayout:
        return _packet_layout(1 + self.column_tile, self.reduction_tile)[0]

    @property
    def wram_bytes(self) -> int:
        return UPMEM_ACTIVE_TASKLETS * (self.packet_bytes + self.output_tile_bytes)

    @property
    def mram_regions(self) -> dict[str, dict[str, object]]:
        packets_bytes = self.packet_count * self.packet_bytes
        output_bytes = self.padded_rows * self.padded_columns * 4
        return {
            "packed_lhs_rhs": _contiguous_region(
                offset=0,
                size=packets_bytes,
                dtype="int32",
                direction="input",
                logical_shape=(
                    self.rows,
                    self.column_tiles,
                    self.reduction_tiles,
                    1 + self.column_tile,
                    self.reduction_tile,
                ),
                physical_shape=(
                    self.padded_rows,
                    self.column_tiles,
                    self.reduction_tiles,
                    1 + self.column_tile,
                    self.reduction_tile,
                ),
                packet_order=["lhs_chunk", "rhs_column_chunks"],
                packet_bytes=self.packet_bytes,
            ),
            "lhs": _segmented_view(
                offset=0,
                payload_bytes=self.packet_count * self.reduction_tile * 4,
                dtype="int32",
                logical_shape=(self.rows, self.reduction),
                physical_shape=(
                    self.padded_rows,
                    self.column_tiles,
                    self.padded_reduction,
                ),
                segment_bytes=self.reduction_tile * 4,
                segment_stride_bytes=self.packet_bytes,
                segments=self.packet_count,
                replication="once per output-column tile",
            ),
            "rhs": _segmented_view(
                offset=self.reduction_tile * 4,
                payload_bytes=(
                    self.packet_count * self.column_tile * self.reduction_tile * 4
                ),
                dtype="int32",
                logical_shape=(self.reduction, self.columns),
                physical_shape=(
                    self.padded_rows,
                    self.column_tiles,
                    self.reduction_tiles,
                    self.column_tile,
                    self.reduction_tile,
                ),
                segment_bytes=self.column_tile * self.reduction_tile * 4,
                segment_stride_bytes=self.packet_bytes,
                segments=self.packet_count,
                replication="once per output-row owner",
            ),
            "output": _contiguous_region(
                offset=packets_bytes,
                size=output_bytes,
                dtype="int32",
                direction="output",
                logical_shape=(self.rows, self.columns),
                physical_shape=(self.padded_rows, self.padded_columns),
                row_major=True,
            ),
        }

    @property
    def output_regions(self) -> dict[str, dict[str, object]]:
        return {"output": self.mram_regions["output"]}

    @property
    def mram_image_bytes(self) -> int:
        output = self.mram_regions["output"]
        return int(output["offset"]) + int(output["bytes"])

    def cost_features(self) -> dict[str, int]:
        compute_packets = self.rows * self.column_tiles * self.reduction_tiles
        read_calls = compute_packets
        write_calls = self.padded_rows * self.column_tiles
        read_bytes = read_calls * self.packet_bytes
        write_bytes = write_calls * self.output_tile_bytes
        fast_path = int(self.exact_row_fast_path)
        return {
            "mram_read_calls": read_calls,
            "mram_write_calls": write_calls,
            "mram_calls": read_calls + write_calls,
            "mram_read_bytes": read_bytes,
            "mram_write_bytes": write_bytes,
            "mram_bytes": read_bytes + write_bytes,
            "barriers": fast_path,
            "wram_bytes": self.wram_bytes,
            "wram_allocation_calls": 2 * UPMEM_ACTIVE_TASKLETS * fast_path,
            "wram_allocation_bytes": self.wram_bytes * fast_path,
            "static_wram_bytes": self.wram_bytes * (1 - fast_path),
            "mem_reset_calls": fast_path,
            "int32_macs": self.rows * self.padded_columns * self.padded_reduction,
        }

    def manifest(self) -> dict[str, object]:
        return {
            "kind": "upmem-fused-replicated-row-major-int32-gemm",
            "shape": [self.rows, self.columns, self.reduction],
            "padded_shape": [
                self.padded_rows,
                self.padded_columns,
                self.padded_reduction,
            ],
            "rhs_mode": self.rhs_mode,
            "codegen_path": self.codegen_path,
            "wram_allocation": {
                "kind": (
                    "per-tasklet-mem-alloc"
                    if self.exact_row_fast_path
                    else "static-tasklet-indexed-arrays"
                ),
                "calls_per_tasklet": 2 if self.exact_row_fast_path else 0,
                "packet_bytes_per_tasklet": self.packet_bytes,
                "accumulator_bytes_per_tasklet": self.output_tile_bytes,
                "total_payload_bytes": self.wram_bytes,
            },
            "mram_addressing": "heap-relative-typed-bases",
            "row_ownership": (
                "tasklet-id-is-output-row"
                if self.exact_row_fast_path
                else "contiguous-row-block-per-tasklet"
            ),
            "column_tile": self.column_tile,
            "reduction_tile": self.reduction_tile,
            "column_tiles": self.column_tiles,
            "reduction_tiles": self.reduction_tiles,
            "packet_elements": self.packet_elements,
            "packet_bytes": self.packet_bytes,
            "packet_order": ["lhs_chunk", "rhs_column_chunks"],
            "output_tile_bytes": self.output_tile_bytes,
            "tasklet_layout": self.tasklet_layout.manifest(),
            "packet_layout": self.packet_layout.manifest(),
            **self._common_manifest(),
        }

    def device_source(self) -> str:
        if self.exact_row_fast_path:
            return self._exact_row_device_source()

        output_offset = int(self.mram_regions["output"]["offset"])
        row_guard_open = ""
        row_guard_close = ""
        if self.padded_rows != self.rows:
            row_guard_open = "      if (row < TENON_ROWS) {\n"
            row_guard_close = "      }\n"
        return (
            _source_prelude("Tenon calibrated fused-replicated int32 GEMM plan.")
            + f"""
#define TENON_ROWS {self.rows}u
#define TENON_ROWS_PER_TASKLET {self.rows_per_tasklet}u
#define TENON_PADDED_ROWS {self.padded_rows}u
#define TENON_COLUMN_TILES {self.column_tiles}u
#define TENON_REDUCTION_TILES {self.reduction_tiles}u
#define TENON_COLUMN_TILE {self.column_tile}u
#define TENON_REDUCTION_TILE {self.reduction_tile}u
#define TENON_PACKET_ELEMENTS {self.packet_elements}u
#define TENON_PACKET_BYTES {self.packet_bytes}u

__dma_aligned int32_t tenon_packets[NR_TASKLETS][TENON_PACKET_ELEMENTS];
__dma_aligned int32_t tenon_accumulators[NR_TASKLETS][TENON_COLUMN_TILE];

int main(void) {{
  const uint32_t tid = me();
  const uint32_t heap = (uint32_t)DPU_MRAM_HEAP_POINTER;
  int32_t *packet = tenon_packets[tid];
  int32_t *accumulator = tenon_accumulators[tid];
  for (uint32_t local_row = 0; local_row < TENON_ROWS_PER_TASKLET;
       ++local_row) {{
    const uint32_t row = tid * TENON_ROWS_PER_TASKLET + local_row;
    for (uint32_t column_block = 0; column_block < TENON_COLUMN_TILES;
         ++column_block) {{
      for (uint32_t column = 0; column < TENON_COLUMN_TILE; ++column)
        accumulator[column] = 0;
{row_guard_open}        for (uint32_t reduction_block = 0;
             reduction_block < TENON_REDUCTION_TILES; ++reduction_block) {{
          const uint32_t packet_id =
              (row * TENON_COLUMN_TILES + column_block) *
                  TENON_REDUCTION_TILES + reduction_block;
          mram_read((const __mram_ptr void *)(uintptr_t)
                        (heap + packet_id * TENON_PACKET_BYTES),
                    packet, TENON_PACKET_BYTES);
          for (uint32_t column = 0; column < TENON_COLUMN_TILE; ++column)
            for (uint32_t k = 0; k < TENON_REDUCTION_TILE; ++k)
              accumulator[column] +=
                  packet[k] * packet[TENON_REDUCTION_TILE +
                                     column * TENON_REDUCTION_TILE + k];
        }}
{row_guard_close}      const uint32_t output_element =
          row * {self.padded_columns}u + column_block * TENON_COLUMN_TILE;
      mram_write(accumulator,
                 (__mram_ptr void *)(uintptr_t)(heap + {output_offset} +
                                                 output_element * 4),
                 {self.output_tile_bytes});
    }}
  }}
  return 0;
}}
"""
        )

    def _exact_row_device_source(self) -> str:
        """Emit the calibrated one-real-row-per-tasklet GEMM fast path."""

        output_offset = int(self.mram_regions["output"]["offset"])
        return (
            _source_prelude(
                "Tenon tasklet-private mem_alloc fused int32 GEMM fast path."
            )
            + f"""
#define TENON_COLUMN_TILES {self.column_tiles}u
#define TENON_REDUCTION_TILES {self.reduction_tiles}u
#define TENON_COLUMN_TILE {self.column_tile}u
#define TENON_REDUCTION_TILE {self.reduction_tile}u
#define TENON_PADDED_COLUMNS {self.padded_columns}u
#define TENON_PACKET_ELEMENTS {self.packet_elements}u
#define TENON_PACKET_BYTES {self.packet_bytes}u
#define TENON_OUTPUT_TILE_BYTES {self.output_tile_bytes}u

BARRIER_INIT(tenon_barrier, NR_TASKLETS);

int main(void) {{
  const uint32_t tid = me();
  const uint32_t heap = (uint32_t)DPU_MRAM_HEAP_POINTER;
  if (tid == 0)
    mem_reset();
  barrier_wait(&tenon_barrier);

  const __mram_ptr int32_t *packets =
      (const __mram_ptr int32_t *)(uintptr_t)heap;
  __mram_ptr int32_t *output =
      (__mram_ptr int32_t *)(uintptr_t)(heap + {output_offset}u);
  int32_t *packet = (int32_t *)mem_alloc(TENON_PACKET_BYTES);
  int32_t *accumulator = (int32_t *)mem_alloc(TENON_OUTPUT_TILE_BYTES);

  for (uint32_t column_block = 0; column_block < TENON_COLUMN_TILES;
       ++column_block) {{
    for (uint32_t column = 0; column < TENON_COLUMN_TILE; ++column)
      accumulator[column] = 0;
    for (uint32_t reduction_block = 0;
         reduction_block < TENON_REDUCTION_TILES; ++reduction_block) {{
      const uint32_t packet_id =
          (tid * TENON_COLUMN_TILES + column_block) *
              TENON_REDUCTION_TILES + reduction_block;
      mram_read((const __mram_ptr void *)
                    &packets[packet_id * TENON_PACKET_ELEMENTS],
                packet, TENON_PACKET_BYTES);
      for (uint32_t column = 0; column < TENON_COLUMN_TILE; ++column)
        for (uint32_t k = 0; k < TENON_REDUCTION_TILE; ++k)
          accumulator[column] +=
              packet[k] * packet[TENON_REDUCTION_TILE +
                                 column * TENON_REDUCTION_TILE + k];
    }}
    mram_write(accumulator,
               (__mram_ptr void *)
                   &output[tid * TENON_PADDED_COLUMNS +
                           column_block * TENON_COLUMN_TILE],
               TENON_OUTPUT_TILE_BYTES);
  }}
  return 0;
}}
"""
        )


# Descriptive aliases used by artifact/export code.
UPMEMInt64SumPlan = UPMEMSumReductionPlan
UPMEMMatVecPlan = UPMEMMatrixVectorPlan
UPMEMInt32GEMMPlan = UPMEMGEMMPlan


__all__ = [
    "MaskedTaskletLayout",
    "UPMEMPhysicalPlan",
    "UPMEMElementwisePlan",
    "UPMEMSumReductionPlan",
    "UPMEMInt64SumPlan",
    "UPMEMMatrixVectorPlan",
    "UPMEMMatVecPlan",
    "UPMEMGEMMPlan",
    "UPMEMInt32GEMMPlan",
    "masked_12_tasklet_linear_layout",
]
