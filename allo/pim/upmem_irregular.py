# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Physical UPMEM plans for irregular integer kernels.

The general UPMEM path deliberately keeps portable MLIR lowering separate
from device-specific physical programs.  This module contains the small set
of physical programs needed by the canonical irregular benchmark suite:

* a private-WRAM histogram followed by an on-DPU merge;
* stable selection with local compaction, a tasklet prefix, and aligned MRAM
  boundary stitching;
* one complete assignment/update iteration of integer k-means; and
* linear-fixed-point or logistic-at-zero feature gradients.

Every plan owns its runtime MRAM ABI, complete SDK translation unit, WRAM
proof, and analytical cost features.  The logical work map is represented by
an F2 :class:`LinearLayout` over a padded 16-tasklet coordinate.  The emitted
program has exactly 12 physical tasklets; coordinates 12--15 are explicitly
masked in the compiler-owned layout manifest.  Padding is therefore an ABI
fact rather than fabricated parallel work.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from typing import Mapping

from ..spmw_linear_layout import LinearLayout


UPMEM_IRREGULAR_TASKLETS = 12
UPMEM_LAYOUT_TASKLETS = 16
UPMEM_WRAM_BYTES = 64 * 1024
UPMEM_MRAM_BYTES = 64 * 1024 * 1024
UPMEM_DMA_ALIGNMENT = 8
UPMEM_MIN_DMA_BYTES = 8
UPMEM_MAX_DMA_BYTES = 2048

# The 2023.1 SDK linker reserves a 1 KiB stack section per enabled tasklet.
# Keep a further fixed quantum for barriers, the software cache, and runtime
# bookkeeping so ``wram_bytes`` is a link-time capacity proof, not just a sum
# of user arrays.
_STACK_RESERVE_PER_TASKLET = 1024
_WRAM_RUNTIME_RESERVE = 1024


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _align_up(value: int, alignment: int = UPMEM_DMA_ALIGNMENT) -> int:
    value = int(value)
    if value < 0 or alignment <= 0:
        raise ValueError("alignment operands must be non-negative and positive")
    return (value + alignment - 1) // alignment * alignment


def _power_of_two_ceiling(value: int) -> int:
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def _validate_tasklets(num_tasklets: int) -> int:
    num_tasklets = _positive_int(num_tasklets, "num_tasklets")
    if num_tasklets != UPMEM_IRREGULAR_TASKLETS:
        raise ValueError(
            "irregular UPMEM physical plans require exactly "
            f"{UPMEM_IRREGULAR_TASKLETS} tasklets"
        )
    return num_tasklets


def _validate_dma_words(dma_elements: int, name: str = "dma_elements") -> int:
    dma_elements = _positive_int(dma_elements, name)
    size = dma_elements * 4
    if size < UPMEM_MIN_DMA_BYTES or size > UPMEM_MAX_DMA_BYTES or size % 8:
        raise ValueError(f"{name} must describe an 8-byte-aligned DMA of 8..2048 bytes")
    return dma_elements


def _validate_regions(regions: Mapping[str, "MRAMRegion"]) -> int:
    end = 0
    for name, region in regions.items():
        if name != region.name:
            raise ValueError("MRAM region mapping keys must equal region names")
        if region.offset % UPMEM_DMA_ALIGNMENT or region.bytes % UPMEM_DMA_ALIGNMENT:
            raise ValueError(f"MRAM region {name!r} is not 8-byte aligned")
        if region.offset < end:
            raise ValueError(f"MRAM region {name!r} overlaps its predecessor")
        end = region.offset + region.bytes
    if end > UPMEM_MRAM_BYTES:
        raise ValueError("UPMEM physical plan exceeds the 64 MiB MRAM capacity")
    return end


def _validate_wram(wram_bytes: int) -> None:
    if int(wram_bytes) > UPMEM_WRAM_BYTES:
        raise ValueError(
            f"UPMEM physical plan requires {wram_bytes} WRAM bytes; "
            f"the 64 KiB WRAM capacity is {UPMEM_WRAM_BYTES} bytes"
        )


@dataclass(frozen=True)
class MRAMRegion:
    """One non-overlapping, DMA-aligned region in the DPU heap image."""

    name: str
    offset: int
    bytes: int
    direction: str
    dtype: str
    shape: tuple[int, ...]
    logical_bytes: int | None = None

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"invalid MRAM region name {self.name!r}")
        if self.direction not in {"input", "output", "inout"}:
            raise ValueError(f"invalid MRAM direction {self.direction!r}")
        if int(self.offset) < 0 or int(self.bytes) <= 0:
            raise ValueError("MRAM region offset/bytes must be non-negative/positive")
        if int(self.offset) % 8 or int(self.bytes) % 8:
            raise ValueError("MRAM region offset and allocation must be 8-byte aligned")
        logical_bytes = self.bytes if self.logical_bytes is None else self.logical_bytes
        if int(logical_bytes) < 0 or int(logical_bytes) > int(self.bytes):
            raise ValueError("MRAM logical_bytes must fit in the allocated region")
        shape = tuple(int(value) for value in self.shape)
        if any(value < 0 for value in shape):
            raise ValueError("MRAM region shape must be non-negative")
        object.__setattr__(self, "offset", int(self.offset))
        object.__setattr__(self, "bytes", int(self.bytes))
        object.__setattr__(self, "logical_bytes", int(logical_bytes))
        object.__setattr__(self, "shape", shape)

    @property
    def padding_bytes(self) -> int:
        return self.bytes - int(self.logical_bytes)

    def manifest(self) -> dict[str, object]:
        return {
            "offset": self.offset,
            "bytes": self.bytes,
            "logical_bytes": self.logical_bytes,
            "padding_bytes": self.padding_bytes,
            "direction": self.direction,
            "dtype": self.dtype,
            "shape": list(self.shape),
        }


def _masked_padded_16_layout(
    logical_extent: int, *, logical_unit: str
) -> tuple[LinearLayout, dict[str, object]]:
    """Return the local compiler-owned 12-active/16-coordinate F2 layout.

    ``LinearLayout`` is intentionally power-of-two-only.  A 12-tasklet DPU
    plan therefore has a 16-coordinate tasklet output and a validity mask.
    This helper stays private until the same masked-layout policy is useful to
    another backend component.
    """

    logical_extent = _positive_int(logical_extent, "logical_extent")
    local_extent = _ceil_div(logical_extent, UPMEM_IRREGULAR_TASKLETS)
    padded_local_extent = _power_of_two_ceiling(local_extent)
    tasklet_vectors = [
        (1 << bit, 0) for bit in range(UPMEM_LAYOUT_TASKLETS.bit_length() - 1)
    ]
    local_vectors = [
        (0, 1 << bit) for bit in range(padded_local_extent.bit_length() - 1)
    ]
    layout = LinearLayout(
        bases={
            "tasklet_block": tasklet_vectors,
            "tasklet_local": local_vectors,
        },
        out_dims=("tasklet", "local"),
        out_sizes=(UPMEM_LAYOUT_TASKLETS, padded_local_extent),
    )
    manifest = layout.manifest()
    manifest.update(
        {
            "owner": "compiler",
            "policy": "masked-padded-16",
            "logical_unit": logical_unit,
            "logical_extent": logical_extent,
            "logical_extent_per_tasklet": local_extent,
            "padded_local_extent": padded_local_extent,
            "physical_tasklets": UPMEM_IRREGULAR_TASKLETS,
            "padded_tasklet_extent": UPMEM_LAYOUT_TASKLETS,
            "active_tasklets": list(range(UPMEM_IRREGULAR_TASKLETS)),
            "masked_tasklets": list(
                range(UPMEM_IRREGULAR_TASKLETS, UPMEM_LAYOUT_TASKLETS)
            ),
            "active_mask": (1 << UPMEM_IRREGULAR_TASKLETS) - 1,
            "validity": (
                "tasklet < 12 && "
                "tasklet * logical_extent_per_tasklet + local < logical_extent"
            ),
        }
    )
    return layout, manifest


class _UPMEMIrregularPlan:
    """Shared immutable ABI/manifest surface for physical plans."""

    num_tasklets: int

    @property
    def compile_flags(self) -> tuple[str, ...]:
        return (f"-DNR_TASKLETS={self.num_tasklets}",)

    @property
    def tasklets(self) -> int:
        return self.num_tasklets

    @property
    def mram_offsets(self) -> dict[str, int]:
        return {name: region.offset for name, region in self.mram_regions.items()}

    @property
    def output_regions(self) -> dict[str, MRAMRegion]:
        return {
            name: region
            for name, region in self.mram_regions.items()
            if region.direction in {"output", "inout"}
        }

    @property
    def mram_image_bytes(self) -> int:
        return _validate_regions(self.mram_regions)

    @property
    def output_contract(self) -> dict[str, object]:
        outputs = self.output_regions
        first = min(region.offset for region in outputs.values())
        last = max(region.offset + region.bytes for region in outputs.values())
        return {
            "kind": "exact-full-output",
            "combined_offset": first,
            "combined_bytes": last - first,
            "regions": {name: region.manifest() for name, region in outputs.items()},
        }

    def _base_manifest(self, kind: str) -> dict[str, object]:
        source = self.device_source()
        return {
            "kind": kind,
            "num_tasklets": self.num_tasklets,
            "compile_flags": list(self.compile_flags),
            "sdk_compilable_translation_unit": True,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "wram_bytes": self.wram_bytes,
            "wram_capacity_bytes": UPMEM_WRAM_BYTES,
            "mram_image_bytes": self.mram_image_bytes,
            "mram_regions": {
                name: region.manifest() for name, region in self.mram_regions.items()
            },
            "output_regions": {
                name: region.manifest() for name, region in self.output_regions.items()
            },
            "output_contract": self.output_contract,
            "linear_layout": self.layout_manifest,
            "cost_features": self.cost_features,
        }


@dataclass(frozen=True)
class UPMEMHistogramPlan(_UPMEMIrregularPlan):
    """Private-WRAM histogram with a deterministic on-DPU merge."""

    elements: int
    bins: int
    depth: int
    num_tasklets: int = UPMEM_IRREGULAR_TASKLETS
    dma_elements: int = 128

    def __post_init__(self) -> None:
        object.__setattr__(self, "elements", _positive_int(self.elements, "elements"))
        object.__setattr__(self, "bins", _positive_int(self.bins, "bins"))
        object.__setattr__(self, "depth", _positive_int(self.depth, "depth"))
        object.__setattr__(self, "num_tasklets", _validate_tasklets(self.num_tasklets))
        object.__setattr__(self, "dma_elements", _validate_dma_words(self.dma_elements))
        if self.depth >= 31:
            raise ValueError("histogram depth must be below 31 for int32 inputs")
        _validate_regions(self.mram_regions)
        _validate_wram(self.wram_bytes)

    @property
    def local_elements(self) -> int:
        return _ceil_div(self.elements, self.num_tasklets)

    @property
    def linear_layout(self) -> LinearLayout:
        return _masked_padded_16_layout(
            self.elements, logical_unit="histogram-element"
        )[0]

    @property
    def layout_manifest(self) -> dict[str, object]:
        return _masked_padded_16_layout(
            self.elements, logical_unit="histogram-element"
        )[1]

    @property
    def mram_regions(self) -> dict[str, MRAMRegion]:
        input_bytes = _align_up(4 * self.elements)
        output_bytes = _align_up(4 * self.bins)
        return {
            "input": MRAMRegion(
                "input",
                0,
                input_bytes,
                "input",
                "int32",
                (self.elements,),
                4 * self.elements,
            ),
            "histogram": MRAMRegion(
                "histogram",
                input_bytes,
                output_bytes,
                "output",
                "uint32",
                (self.bins,),
                4 * self.bins,
            ),
        }

    @property
    def wram_bytes(self) -> int:
        private_histograms = 4 * self.num_tasklets * self.bins
        merged = _align_up(4 * self.bins)
        dma_buffers = 4 * self.num_tasklets * (self.dma_elements + 2)
        stacks = _STACK_RESERVE_PER_TASKLET * self.num_tasklets
        return (
            private_histograms + merged + dma_buffers + stacks + _WRAM_RUNTIME_RESERVE
        )

    @property
    def cost_features(self) -> dict[str, object]:
        return {
            "tasklet_fanout": self.num_tasklets,
            "layout_tasklet_extent": UPMEM_LAYOUT_TASKLETS,
            "logical_work_items": self.elements,
            "wram_bytes": self.wram_bytes,
            "barriers": 3,
            "primitive_iterations": {
                "MUL": self.elements,
                "ADD": self.elements + self.bins * self.num_tasklets,
                "CMP": self.elements,
            },
            "dma_bytes": {
                "LD_MRAM": 4 * self.elements,
                "ST_MRAM": _align_up(4 * self.bins),
            },
            "dma_calls": {
                "LD_MRAM": _ceil_div(self.elements, self.dma_elements),
                "ST_MRAM": _ceil_div(_align_up(4 * self.bins), UPMEM_MAX_DMA_BYTES),
            },
        }

    def manifest(self) -> dict[str, object]:
        return {
            **self._base_manifest("upmem-private-wram-histogram"),
            "elements": self.elements,
            "bins": self.bins,
            "depth": self.depth,
            "dma_elements": self.dma_elements,
            "algorithm": ["private_histogram", "barrier", "merge", "aligned_write"],
        }

    def reference_histogram(self, values) -> tuple[int, ...]:
        """Evaluate the device plan's exact valid-bin behavior."""

        values = tuple(int(value) for value in values)
        if len(values) != self.elements:
            raise ValueError(f"expected {self.elements} histogram inputs")
        result = [0] * self.bins
        for value in values:
            bin_index = (value * self.bins) >> self.depth
            if 0 <= bin_index < self.bins:
                result[bin_index] += 1
        return tuple(result)

    def device_source(self) -> str:
        input_offset = self.mram_regions["input"].offset
        output = self.mram_regions["histogram"]
        output_words = output.bytes // 4
        n, bins, depth = self.elements, self.bins, self.depth
        nt, dma, local = self.num_tasklets, self.dma_elements, self.local_elements
        return f"""// Tenon private-WRAM histogram physical plan for UPMEM.
// Runtime ABI: int32 input[{n}] -> uint32 histogram[{bins}].
#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <stdint.h>

#if !defined(NR_TASKLETS) || NR_TASKLETS != {nt}
#error "Tenon irregular UPMEM plan requires NR_TASKLETS={nt}"
#endif

#define TENON_NUM_TASKLETS {nt}
#define TENON_LAYOUT_TASKLETS {UPMEM_LAYOUT_TASKLETS}
#define TENON_N {n}u
#define TENON_BINS {bins}u
#define TENON_LOCAL_ELEMENTS {local}u
#define TENON_DMA_WORDS {dma}u

__dma_aligned uint32_t tenon_private_hist[{nt}][{bins}];
__dma_aligned uint32_t tenon_merged_hist[{output_words}];
__dma_aligned int32_t tenon_input_buf[{nt}][{dma + 2}];
BARRIER_INIT(tenon_barrier, NR_TASKLETS);

static void tenon_read_input(uint32_t tid, uint32_t heap,
                             uint32_t begin, uint32_t count) {{
  uint32_t byte_begin = begin * sizeof(int32_t);
  uint32_t aligned_begin = byte_begin & ~7u;
  uint32_t shift = (byte_begin - aligned_begin) / sizeof(int32_t);
  uint32_t bytes = (shift + count) * sizeof(int32_t);
  bytes = (bytes + 7u) & ~7u;
  mram_read((const __mram_ptr void *)(uintptr_t)(heap + {input_offset}u + aligned_begin),
            tenon_input_buf[tid], bytes);
  for (uint32_t i = 0; i < count; ++i) {{
    int32_t value = tenon_input_buf[tid][shift + i];
    int64_t scaled = (int64_t)value * (int64_t)TENON_BINS;
    int64_t bin = scaled >> {depth};
    if (bin >= 0 && bin < (int64_t)TENON_BINS)
      tenon_private_hist[tid][(uint32_t)bin] += 1u;
  }}
}}

int main(void) {{
  uint32_t tid = me();
  uint32_t heap = (uint32_t)DPU_MRAM_HEAP_POINTER;
  if (tid == 0) mem_reset();
  for (uint32_t bin = 0; bin < TENON_BINS; ++bin)
    tenon_private_hist[tid][bin] = 0u;
  barrier_wait(&tenon_barrier);

  uint32_t begin = tid * TENON_LOCAL_ELEMENTS;
  uint32_t end = begin + TENON_LOCAL_ELEMENTS;
  if (end > TENON_N) end = TENON_N;
  for (uint32_t cursor = begin; cursor < end;) {{
    uint32_t count = end - cursor;
    if (count > TENON_DMA_WORDS) count = TENON_DMA_WORDS;
    tenon_read_input(tid, heap, cursor, count);
    cursor += count;
  }}
  barrier_wait(&tenon_barrier);

  for (uint32_t bin = tid; bin < TENON_BINS; bin += TENON_NUM_TASKLETS) {{
    uint32_t total = 0u;
    for (uint32_t owner = 0; owner < TENON_NUM_TASKLETS; ++owner)
      total += tenon_private_hist[owner][bin];
    tenon_merged_hist[bin] = total;
  }}
  if (tid == 0)
    for (uint32_t bin = TENON_BINS; bin < {output_words}u; ++bin)
      tenon_merged_hist[bin] = 0u;
  barrier_wait(&tenon_barrier);

  if (tid == 0) {{
    uint32_t cursor = 0u;
    while (cursor < {output.bytes}u) {{
      uint32_t bytes = {output.bytes}u - cursor;
      if (bytes > {UPMEM_MAX_DMA_BYTES}u) bytes = {UPMEM_MAX_DMA_BYTES}u;
      mram_write((const void *)((const uint8_t *)tenon_merged_hist + cursor),
                 (__mram_ptr void *)(uintptr_t)(heap + {output.offset}u + cursor),
                 bytes);
      cursor += bytes;
    }}
  }}
  return 0;
}}
"""


@dataclass(frozen=True)
class UPMEMStableSelectionPlan(_UPMEMIrregularPlan):
    """Stable odd-value selection with exact values and count output.

    Each tasklet compacts a contiguous logical slice into private WRAM.  A
    prefix over the twelve counts establishes global stable positions.  Bulk
    writes cover aligned pairs in parallel; tasklet zero stitches only the
    at-most-eleven odd boundaries and the optional final padding word.  Thus
    every MRAM operation remains 8-byte aligned without changing value order.
    """

    elements: int
    num_tasklets: int = UPMEM_IRREGULAR_TASKLETS
    dma_elements: int = 32
    predicate: str = "odd"

    def __post_init__(self) -> None:
        object.__setattr__(self, "elements", _positive_int(self.elements, "elements"))
        object.__setattr__(self, "num_tasklets", _validate_tasklets(self.num_tasklets))
        object.__setattr__(self, "dma_elements", _validate_dma_words(self.dma_elements))
        if self.predicate != "odd":
            raise ValueError(
                "the stable selection physical plan supports predicate='odd'"
            )
        _validate_regions(self.mram_regions)
        _validate_wram(self.wram_bytes)

    @property
    def local_capacity(self) -> int:
        return _ceil_div(self.elements, self.num_tasklets)

    @property
    def linear_layout(self) -> LinearLayout:
        return _masked_padded_16_layout(
            self.elements, logical_unit="selection-element"
        )[0]

    @property
    def layout_manifest(self) -> dict[str, object]:
        return _masked_padded_16_layout(
            self.elements, logical_unit="selection-element"
        )[1]

    @property
    def mram_regions(self) -> dict[str, MRAMRegion]:
        input_bytes = _align_up(4 * self.elements)
        # Count has an explicit uint32 zero pad so the following value stream
        # starts at an aligned address and can be compared as one region.
        count_offset = input_bytes
        values_offset = count_offset + 8
        values_bytes = _align_up(4 * self.elements)
        return {
            "input": MRAMRegion(
                "input",
                0,
                input_bytes,
                "input",
                "int32",
                (self.elements,),
                4 * self.elements,
            ),
            "selected_count": MRAMRegion(
                "selected_count",
                count_offset,
                8,
                "output",
                "uint32",
                (1,),
                4,
            ),
            "selected_values": MRAMRegion(
                "selected_values",
                values_offset,
                values_bytes,
                "output",
                "int32",
                (self.elements,),
                4 * self.elements,
            ),
        }

    @property
    def output_contract(self) -> dict[str, object]:
        count = self.mram_regions["selected_count"]
        values = self.mram_regions["selected_values"]
        return {
            "kind": "exact-count-delimited-stable-selection",
            "combined_offset": count.offset,
            "combined_bytes": count.bytes + values.bytes,
            "header": {
                "count_offset": count.offset,
                "count_dtype": "uint32",
                "padding_offset": count.offset + 4,
                "padding_value": 0,
            },
            "values_offset": values.offset,
            "values_dtype": "int32",
            "logical_values": "selected_count",
            "capacity_values": self.elements,
            "comparison_bytes": "8 + 4 * selected_count",
            "order": "stable-input-order",
            "regions": {
                name: region.manifest() for name, region in self.output_regions.items()
            },
        }

    @property
    def wram_bytes(self) -> int:
        private_values = 4 * self.num_tasklets * self.local_capacity
        io_buffers = 4 * self.num_tasklets * (self.dma_elements + 2)
        counts_and_prefixes = 8 * self.num_tasklets
        stacks = _STACK_RESERVE_PER_TASKLET * self.num_tasklets
        return (
            private_values
            + io_buffers
            + counts_and_prefixes
            + stacks
            + _WRAM_RUNTIME_RESERVE
        )

    @property
    def cost_features(self) -> dict[str, object]:
        # Output traffic is conservatively priced at full capacity because the
        # selected count is data dependent; the exact count remains in the ABI.
        return {
            "tasklet_fanout": self.num_tasklets,
            "layout_tasklet_extent": UPMEM_LAYOUT_TASKLETS,
            "logical_work_items": self.elements,
            "wram_bytes": self.wram_bytes,
            "barriers": 4,
            "data_dependent_output": True,
            "primitive_iterations": {
                "CMP": self.elements,
                "ADD": self.elements + 2 * self.num_tasklets,
                "BRANCH": self.elements,
            },
            "dma_bytes": {
                "LD_MRAM": 4 * self.elements,
                "ST_MRAM_upper_bound": 8 + _align_up(4 * self.elements),
            },
            "dma_calls": {
                "LD_MRAM": _ceil_div(self.elements, self.dma_elements),
                "ST_MRAM_upper_bound": (
                    _ceil_div(self.elements, self.dma_elements) + self.num_tasklets + 1
                ),
            },
        }

    def manifest(self) -> dict[str, object]:
        return {
            **self._base_manifest("upmem-stable-selection"),
            "elements": self.elements,
            "predicate": self.predicate,
            "dma_elements": self.dma_elements,
            "local_capacity": self.local_capacity,
            "algorithm": [
                "tasklet_local_compaction",
                "tasklet_count_prefix",
                "aligned_parallel_bulk_write",
                "aligned_boundary_stitch",
                "count_write",
            ],
        }

    def reference_selection(self, values) -> tuple[tuple[int, ...], int]:
        """Return the exact stable values/count pair described by the ABI."""

        values = tuple(int(value) for value in values)
        if len(values) != self.elements:
            raise ValueError(f"expected {self.elements} selection inputs")
        selected = tuple(value for value in values if value & 1)
        return selected, len(selected)

    def device_source(self) -> str:
        regions = self.mram_regions
        input_offset = regions["input"].offset
        count_offset = regions["selected_count"].offset
        values_offset = regions["selected_values"].offset
        n, nt = self.elements, self.num_tasklets
        dma, local = self.dma_elements, self.local_capacity
        return f"""// Tenon stable odd-selection physical plan for UPMEM.
// Runtime ABI: int32 input[{n}] -> {{uint32 count, uint32 zero, int32 values[count]}}.
#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <stdint.h>

#if !defined(NR_TASKLETS) || NR_TASKLETS != {nt}
#error "Tenon irregular UPMEM plan requires NR_TASKLETS={nt}"
#endif

#define TENON_NUM_TASKLETS {nt}
#define TENON_LAYOUT_TASKLETS {UPMEM_LAYOUT_TASKLETS}
#define TENON_N {n}u
#define TENON_LOCAL_CAPACITY {local}u
#define TENON_DMA_WORDS {dma}u

__dma_aligned int32_t tenon_private_values[{nt}][{local}];
__dma_aligned int32_t tenon_io[{nt}][{dma + 2}];
__dma_aligned uint32_t tenon_counts[{nt}];
__dma_aligned uint32_t tenon_prefixes[{nt}];
BARRIER_INIT(tenon_barrier, NR_TASKLETS);

static uint32_t tenon_compact_chunk(uint32_t tid, uint32_t heap,
                                    uint32_t begin, uint32_t count,
                                    uint32_t local_count) {{
  uint32_t byte_begin = begin * sizeof(int32_t);
  uint32_t aligned_begin = byte_begin & ~7u;
  uint32_t shift = (byte_begin - aligned_begin) / sizeof(int32_t);
  uint32_t bytes = (shift + count) * sizeof(int32_t);
  bytes = (bytes + 7u) & ~7u;
  mram_read((const __mram_ptr void *)(uintptr_t)(heap + {input_offset}u + aligned_begin),
            tenon_io[tid], bytes);
  for (uint32_t i = 0; i < count; ++i) {{
    int32_t value = tenon_io[tid][shift + i];
    if ((value & 1) != 0)
      tenon_private_values[tid][local_count++] = value;
  }}
  return local_count;
}}

static void tenon_write_private(uint32_t tid, uint32_t heap,
                                uint32_t local_begin,
                                uint32_t global_begin,
                                uint32_t count) {{
  while (count != 0u) {{
    uint32_t words = count;
    if (words > TENON_DMA_WORDS) words = TENON_DMA_WORDS;
    words &= ~1u;
    for (uint32_t i = 0; i < words; ++i)
      tenon_io[tid][i] = tenon_private_values[tid][local_begin + i];
    mram_write(tenon_io[tid],
               (__mram_ptr void *)(uintptr_t)(heap + {values_offset}u +
                                               global_begin * sizeof(int32_t)),
               words * sizeof(int32_t));
    local_begin += words;
    global_begin += words;
    count -= words;
  }}
}}

int main(void) {{
  uint32_t tid = me();
  uint32_t heap = (uint32_t)DPU_MRAM_HEAP_POINTER;
  if (tid == 0) mem_reset();
  barrier_wait(&tenon_barrier);

  uint32_t begin = tid * TENON_LOCAL_CAPACITY;
  uint32_t end = begin + TENON_LOCAL_CAPACITY;
  if (end > TENON_N) end = TENON_N;
  uint32_t local_count = 0u;
  for (uint32_t cursor = begin; cursor < end;) {{
    uint32_t count = end - cursor;
    if (count > TENON_DMA_WORDS) count = TENON_DMA_WORDS;
    local_count = tenon_compact_chunk(tid, heap, cursor, count, local_count);
    cursor += count;
  }}
  tenon_counts[tid] = local_count;
  barrier_wait(&tenon_barrier);

  if (tid == 0) {{
    uint32_t prefix = 0u;
    for (uint32_t owner = 0; owner < TENON_NUM_TASKLETS; ++owner) {{
      tenon_prefixes[owner] = prefix;
      prefix += tenon_counts[owner];
    }}
  }}
  barrier_wait(&tenon_barrier);

  // Exclude an odd leading position and an odd trailing position from each
  // slice.  The remaining even-length interior has aligned source/destination
  // staging and is written independently by its owning tasklet.
  uint32_t global_begin = tenon_prefixes[tid];
  uint32_t local_begin = global_begin & 1u;
  uint32_t interior = tenon_counts[tid] > local_begin ?
                      (tenon_counts[tid] - local_begin) & ~1u : 0u;
  if (interior != 0u)
    tenon_write_private(tid, heap, local_begin,
                        global_begin + local_begin, interior);
  barrier_wait(&tenon_barrier);

  if (tid == 0) {{
    __dma_aligned int32_t pair[2];
    // Stitch every odd prefix. Empty owners are skipped; the preceding value
    // is taken from the nearest non-empty owner, preserving global stability.
    for (uint32_t owner = 1; owner < TENON_NUM_TASKLETS; ++owner) {{
      uint32_t prefix = tenon_prefixes[owner];
      if (tenon_counts[owner] != 0u && (prefix & 1u) != 0u) {{
        uint32_t previous = owner;
        do {{ --previous; }} while (tenon_counts[previous] == 0u);
        pair[0] = tenon_private_values[previous][tenon_counts[previous] - 1u];
        pair[1] = tenon_private_values[owner][0];
        mram_write(pair,
                   (__mram_ptr void *)(uintptr_t)(heap + {values_offset}u +
                                                   (prefix - 1u) * sizeof(int32_t)),
                   8u);
      }}
    }}

    uint32_t total = tenon_prefixes[TENON_NUM_TASKLETS - 1u] +
                     tenon_counts[TENON_NUM_TASKLETS - 1u];
    if ((total & 1u) != 0u) {{
      uint32_t last = TENON_NUM_TASKLETS;
      do {{ --last; }} while (tenon_counts[last] == 0u);
      pair[0] = tenon_private_values[last][tenon_counts[last] - 1u];
      pair[1] = 0;
      mram_write(pair,
                 (__mram_ptr void *)(uintptr_t)(heap + {values_offset}u +
                                                 (total - 1u) * sizeof(int32_t)),
                 8u);
    }}
    __dma_aligned uint32_t header[2] = {{total, 0u}};
    mram_write(header,
               (__mram_ptr void *)(uintptr_t)(heap + {count_offset}u), 8u);
  }}
  return 0;
}}
"""


@dataclass(frozen=True)
class UPMEMSelectionFlagsPlan(_UPMEMIrregularPlan):
    """Device-comparable odd-selection flag/materialization phase.

    The archived ATiM comparison times the DPU phase that maps every input to
    either the original odd value or zero; stable compaction is performed by
    the host and is outside its reported kernel counter.  This plan preserves
    that exact partition so Tenon can be compared fairly.  The full on-DPU
    stable-compaction plan remains available as :class:`UPMEMStableSelectionPlan`.
    """

    elements: int
    num_tasklets: int = UPMEM_IRREGULAR_TASKLETS
    dma_elements: int = 16
    predicate: str = "odd"

    def __post_init__(self) -> None:
        object.__setattr__(self, "elements", _positive_int(self.elements, "elements"))
        object.__setattr__(self, "num_tasklets", _validate_tasklets(self.num_tasklets))
        object.__setattr__(self, "dma_elements", _validate_dma_words(self.dma_elements))
        if self.predicate != "odd":
            raise ValueError("the selection-flags plan supports predicate='odd'")
        if self.elements % self.num_tasklets:
            raise ValueError("selection-flags elements must divide evenly by tasklets")
        if self.local_elements % self.dma_elements:
            raise ValueError("selection-flags local extent must divide by dma_elements")
        _validate_regions(self.mram_regions)
        _validate_wram(self.wram_bytes)

    @property
    def local_elements(self) -> int:
        return self.elements // self.num_tasklets

    @property
    def linear_layout(self) -> LinearLayout:
        return _masked_padded_16_layout(
            self.elements, logical_unit="selection-flag-element"
        )[0]

    @property
    def layout_manifest(self) -> dict[str, object]:
        return _masked_padded_16_layout(
            self.elements, logical_unit="selection-flag-element"
        )[1]

    @property
    def mram_regions(self) -> dict[str, MRAMRegion]:
        allocation = _align_up(4 * self.elements)
        return {
            "input": MRAMRegion(
                "input", 0, allocation, "input", "int32", (self.elements,)
            ),
            "flags": MRAMRegion(
                "flags",
                allocation,
                allocation,
                "output",
                "int32",
                (self.elements,),
            ),
        }

    @property
    def output_contract(self) -> dict[str, object]:
        flags = self.mram_regions["flags"]
        return {
            "kind": "exact-selection-device-flags",
            "combined_offset": flags.offset,
            "combined_bytes": flags.bytes,
            "equation": "flags[i] = input[i] if (input[i] & 1) else 0",
            "host_compaction": "excluded-to-match-archived-device-kernel-scope",
            "result": flags.manifest(),
        }

    @property
    def wram_bytes(self) -> int:
        return (
            4 * self.num_tasklets * self.dma_elements
            + _STACK_RESERVE_PER_TASKLET * self.num_tasklets
            + _WRAM_RUNTIME_RESERVE
        )

    @property
    def cost_features(self) -> dict[str, object]:
        return {
            "tasklet_fanout": self.num_tasklets,
            "layout_tasklet_extent": UPMEM_LAYOUT_TASKLETS,
            "logical_work_items": self.elements,
            "wram_bytes": self.wram_bytes,
            "barriers": 1,
            "comparison_partition": "device_flags_only",
            "primitive_iterations": {
                "AND": 2 * self.elements,
                "SUB": self.elements,
            },
            "dma_bytes": {
                "LD_MRAM": 4 * self.elements,
                "ST_MRAM": 4 * self.elements,
            },
            "dma_calls": {
                "LD_MRAM": self.elements // self.dma_elements,
                "ST_MRAM": self.elements // self.dma_elements,
            },
        }

    def manifest(self) -> dict[str, object]:
        return {
            **self._base_manifest("upmem-selection-device-flags"),
            "elements": self.elements,
            "predicate": self.predicate,
            "dma_elements": self.dma_elements,
            "local_elements": self.local_elements,
            "algorithm": [
                "contiguous_tasklet_ownership",
                "aligned_streaming",
                "calibrated_branch_odd_value_or_zero",
            ],
        }

    def reference_flags(self, values) -> tuple[int, ...]:
        values = tuple(int(value) for value in values)
        if len(values) != self.elements:
            raise ValueError(f"expected {self.elements} selection inputs")
        return tuple(value if value & 1 else 0 for value in values)

    def device_source(self) -> str:
        input_region = self.mram_regions["input"]
        flags_region = self.mram_regions["flags"]
        n, nt = self.elements, self.num_tasklets
        dma, local = self.dma_elements, self.local_elements
        return f"""// Tenon device-comparable odd-selection flags for UPMEM.
// Runtime ABI: int32 input[{n}] -> int32 flags[{n}].
#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <stdint.h>

#if !defined(NR_TASKLETS) || NR_TASKLETS != {nt}
#error "Tenon irregular UPMEM plan requires NR_TASKLETS={nt}"
#endif

#define TENON_NUM_TASKLETS {nt}
#define TENON_LAYOUT_TASKLETS {UPMEM_LAYOUT_TASKLETS}
#define TENON_LOCAL_ELEMENTS {local}u
#define TENON_DMA_WORDS {dma}u
BARRIER_INIT(tenon_barrier, NR_TASKLETS);

int main(void) {{
  const uint32_t tid = me();
  const uint32_t heap = (uint32_t)DPU_MRAM_HEAP_POINTER;
  if (tid == 0) mem_reset();
  barrier_wait(&tenon_barrier);
  int32_t *buffer = (int32_t *)mem_alloc(TENON_DMA_WORDS * sizeof(int32_t));
  const uint32_t begin = tid * TENON_LOCAL_ELEMENTS;
  for (uint32_t local = 0; local < TENON_LOCAL_ELEMENTS;
       local += TENON_DMA_WORDS) {{
    const uint32_t element = begin + local;
    mram_read((const __mram_ptr void *)(uintptr_t)(
                  heap + {input_region.offset}u + element * sizeof(int32_t)),
              buffer, TENON_DMA_WORDS * sizeof(int32_t));
    for (uint32_t i = 0; i < TENON_DMA_WORDS; ++i) {{
      if ((buffer[i] & 1) == 0)
        buffer[i] = 0;
    }}
    mram_write(buffer,
               (__mram_ptr void *)(uintptr_t)(
                   heap + {flags_region.offset}u + element * sizeof(int32_t)),
               TENON_DMA_WORDS * sizeof(int32_t));
  }}
  return 0;
}}
"""


@dataclass(frozen=True)
class UPMEMKMeansPlan(_UPMEMIrregularPlan):
    """One exact squared-L2 assignment and rounded centroid update.

    This is the full one-iteration application partition.  The archived ATiM
    comparison offloads distances only; use :class:`UPMEMKMeansDistancesPlan`
    when reproducing that device-kernel boundary.
    """

    points: int
    dimension: int
    clusters: int
    iterations: int = 1
    num_tasklets: int = UPMEM_IRREGULAR_TASKLETS

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", _positive_int(self.points, "points"))
        object.__setattr__(
            self, "dimension", _positive_int(self.dimension, "dimension")
        )
        object.__setattr__(self, "clusters", _positive_int(self.clusters, "clusters"))
        object.__setattr__(
            self, "iterations", _positive_int(self.iterations, "iterations")
        )
        object.__setattr__(self, "num_tasklets", _validate_tasklets(self.num_tasklets))
        if self.iterations != 1:
            raise ValueError("UPMEMKMeansPlan is exactly one k-means iteration")
        if self.dimension % 2:
            raise ValueError(
                "k-means dimension must be even so every point/centroid DMA is aligned"
            )
        if 4 * self.dimension > UPMEM_MAX_DMA_BYTES:
            raise ValueError("one k-means point exceeds the 2048-byte DMA maximum")
        _validate_regions(self.mram_regions)
        _validate_wram(self.wram_bytes)

    @property
    def local_points(self) -> int:
        return _ceil_div(self.points, self.num_tasklets)

    @property
    def linear_layout(self) -> LinearLayout:
        return _masked_padded_16_layout(self.points, logical_unit="kmeans-point")[0]

    @property
    def layout_manifest(self) -> dict[str, object]:
        return _masked_padded_16_layout(self.points, logical_unit="kmeans-point")[1]

    @property
    def mram_regions(self) -> dict[str, MRAMRegion]:
        point_logical = 4 * self.points * self.dimension
        centroid_logical = 4 * self.clusters * self.dimension
        count_logical = 4 * self.clusters
        point_bytes = _align_up(point_logical)
        input_centroid_bytes = _align_up(centroid_logical)
        output_centroid_bytes = _align_up(centroid_logical)
        count_bytes = _align_up(count_logical)
        initial_offset = point_bytes
        output_offset = initial_offset + input_centroid_bytes
        count_offset = output_offset + output_centroid_bytes
        return {
            "points": MRAMRegion(
                "points",
                0,
                point_bytes,
                "input",
                "int32",
                (self.points, self.dimension),
                point_logical,
            ),
            "initial_centroids": MRAMRegion(
                "initial_centroids",
                initial_offset,
                input_centroid_bytes,
                "input",
                "int32",
                (self.clusters, self.dimension),
                centroid_logical,
            ),
            "centroids": MRAMRegion(
                "centroids",
                output_offset,
                output_centroid_bytes,
                "output",
                "int32",
                (self.clusters, self.dimension),
                centroid_logical,
            ),
            "counts": MRAMRegion(
                "counts",
                count_offset,
                count_bytes,
                "output",
                "uint32",
                (self.clusters,),
                count_logical,
            ),
        }

    @property
    def output_contract(self) -> dict[str, object]:
        centroids = self.mram_regions["centroids"]
        counts = self.mram_regions["counts"]
        return {
            "kind": "exact-kmeans-one-iteration",
            "combined_offset": centroids.offset,
            "combined_bytes": centroids.bytes + counts.bytes,
            "region_order": ["centroids", "counts"],
            "centroid_empty_cluster_value": 0,
            "centroid_rounding": "signed-round-closest-ties-away-from-zero",
            "assignment_tie_break": "lowest-cluster-index",
            "regions": {
                name: region.manifest() for name, region in self.output_regions.items()
            },
        }

    @property
    def wram_bytes(self) -> int:
        centroid_bytes = _align_up(4 * self.clusters * self.dimension)
        point_buffers = 4 * self.num_tasklets * self.dimension
        # Canonical UPMEM k-means inputs are int32 values in [-50, 50], so a
        # 120-point coordinate sum is safely int32.  Keeping the merge/divide
        # path in 32 bits also avoids the SDK's ``__divdi3`` instruction
        # sequence, which the frozen uPIMulator cannot execute.
        private_sums = 4 * self.num_tasklets * self.clusters * self.dimension
        private_counts = 4 * self.num_tasklets * self.clusters
        output_centroids = centroid_bytes
        output_counts = _align_up(4 * self.clusters)
        stacks = _STACK_RESERVE_PER_TASKLET * self.num_tasklets
        return (
            centroid_bytes
            + point_buffers
            + private_sums
            + private_counts
            + output_centroids
            + output_counts
            + stacks
            + _WRAM_RUNTIME_RESERVE
        )

    @property
    def cost_features(self) -> dict[str, object]:
        assignments = self.points * self.clusters
        distance_terms = assignments * self.dimension
        updates = self.points * self.dimension
        return {
            "tasklet_fanout": self.num_tasklets,
            "layout_tasklet_extent": UPMEM_LAYOUT_TASKLETS,
            "logical_work_items": self.points,
            "wram_bytes": self.wram_bytes,
            "barriers": 2,
            "primitive_iterations": {
                "SUB": distance_terms,
                "MUL": distance_terms,
                "ADD": distance_terms + updates,
                "CMP": self.points * (self.clusters - 1),
                "DIV": self.clusters * self.dimension,
            },
            "dma_bytes": {
                "LD_MRAM": (
                    4 * self.points * self.dimension
                    + _align_up(4 * self.clusters * self.dimension)
                ),
                "ST_MRAM": (
                    _align_up(4 * self.clusters * self.dimension)
                    + _align_up(4 * self.clusters)
                ),
            },
            "dma_calls": {
                "LD_MRAM": self.points
                + _ceil_div(
                    _align_up(4 * self.clusters * self.dimension),
                    UPMEM_MAX_DMA_BYTES,
                ),
                "ST_MRAM": _ceil_div(
                    _align_up(4 * self.clusters * self.dimension),
                    UPMEM_MAX_DMA_BYTES,
                )
                + _ceil_div(_align_up(4 * self.clusters), UPMEM_MAX_DMA_BYTES),
            },
        }

    def manifest(self) -> dict[str, object]:
        return {
            **self._base_manifest("upmem-kmeans-one-iteration"),
            "points": self.points,
            "dimension": self.dimension,
            "clusters": self.clusters,
            "iterations": self.iterations,
            "local_points": self.local_points,
            "algorithm": [
                "centroid_cache",
                "squared_l2_assignment",
                "private_sums_and_counts",
                "merge",
                "signed_round_closest",
            ],
            "accumulator_contract": {
                "dtype": "int32",
                "required": "every per-cluster coordinate sum fits signed int32",
                "canonical_input_bound": [-50, 50],
            },
        }

    @staticmethod
    def signed_round_closest(numerator: int, denominator: int) -> int:
        """C/Python-independent signed round-to-nearest, ties away from zero."""

        numerator, denominator = int(numerator), int(denominator)
        if denominator <= 0:
            raise ValueError("k-means centroid denominator must be positive")
        adjusted = (
            numerator - denominator // 2
            if numerator < 0
            else numerator + denominator // 2
        )
        magnitude = abs(adjusted) // denominator
        return -magnitude if adjusted < 0 else magnitude

    def reference_iteration(
        self, point_values, initial_centroid_values
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Evaluate the exact assignment, tie-break, and update contract."""

        points = tuple(int(value) for value in point_values)
        centroids = tuple(int(value) for value in initial_centroid_values)
        if len(points) != self.points * self.dimension:
            raise ValueError("point_values has the wrong flattened extent")
        if len(centroids) != self.clusters * self.dimension:
            raise ValueError("initial_centroid_values has the wrong extent")
        sums = [[0] * self.dimension for _ in range(self.clusters)]
        counts = [0] * self.clusters
        for point in range(self.points):
            point_base = point * self.dimension
            chosen = min(
                range(self.clusters),
                key=lambda cluster: sum(
                    (
                        points[point_base + feature]
                        - centroids[cluster * self.dimension + feature]
                    )
                    ** 2
                    for feature in range(self.dimension)
                ),
            )
            counts[chosen] += 1
            for feature in range(self.dimension):
                sums[chosen][feature] += points[point_base + feature]
        updated = tuple(
            (
                0
                if counts[cluster] == 0
                else self.signed_round_closest(sums[cluster][feature], counts[cluster])
            )
            for cluster in range(self.clusters)
            for feature in range(self.dimension)
        )
        return updated, tuple(counts)

    def device_source(self) -> str:
        regions = self.mram_regions
        point_offset = regions["points"].offset
        initial_offset = regions["initial_centroids"].offset
        centroid_output = regions["centroids"]
        count_output = regions["counts"]
        centroid_words = centroid_output.bytes // 4
        count_words = count_output.bytes // 4
        p, d, k = self.points, self.dimension, self.clusters
        nt, local = self.num_tasklets, self.local_points
        return f"""// Tenon one-iteration integer k-means physical plan for UPMEM.
// Runtime ABI: points[{p},{d}], initial_centroids[{k},{d}]
//           -> centroids[{k},{d}], counts[{k}].
#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <stdint.h>

#if !defined(NR_TASKLETS) || NR_TASKLETS != {nt}
#error "Tenon irregular UPMEM plan requires NR_TASKLETS={nt}"
#endif

#define TENON_NUM_TASKLETS {nt}
#define TENON_LAYOUT_TASKLETS {UPMEM_LAYOUT_TASKLETS}
#define TENON_POINTS {p}u
#define TENON_DIMENSION {d}u
#define TENON_CLUSTERS {k}u
#define TENON_LOCAL_POINTS {local}u

__dma_aligned int32_t tenon_centroid_cache[{centroid_words}];
__dma_aligned int32_t tenon_point[{nt}][{d}];
__dma_aligned int32_t tenon_private_sums[{nt}][{k}][{d}];
__dma_aligned uint32_t tenon_private_counts[{nt}][{k}];
__dma_aligned int32_t tenon_output_centroids[{centroid_words}];
__dma_aligned uint32_t tenon_output_counts[{count_words}];
BARRIER_INIT(tenon_barrier, NR_TASKLETS);

static int32_t tenon_div_round_closest(int32_t numerator,
                                       int32_t denominator) {{
  // Avoid the SDK division helpers entirely: the frozen simulator does not
  // reproduce their result registers reliably.  Canonical coordinate means
  // have magnitude at most 50, so this exact repeated-subtraction quotient is
  // both bounded and cheaper than making the benchmark non-replayable.
  uint32_t magnitude = numerator < 0 ? (uint32_t)(-numerator) :
                                       (uint32_t)numerator;
  uint32_t remainder = magnitude + ((uint32_t)denominator >> 1u);
  uint32_t quotient = 0u;
  while (remainder >= (uint32_t)denominator) {{
    remainder -= (uint32_t)denominator;
    quotient += 1u;
  }}
  return numerator < 0 ? -(int32_t)quotient : (int32_t)quotient;
}}

static void tenon_copy_from_mram(uint32_t heap, uint32_t mram_offset,
                                 void *wram, uint32_t bytes) {{
  uint32_t cursor = 0u;
  while (cursor < bytes) {{
    uint32_t chunk = bytes - cursor;
    if (chunk > {UPMEM_MAX_DMA_BYTES}u) chunk = {UPMEM_MAX_DMA_BYTES}u;
    mram_read((const __mram_ptr void *)(uintptr_t)(heap + mram_offset + cursor),
              (void *)((uint8_t *)wram + cursor), chunk);
    cursor += chunk;
  }}
}}

static void tenon_copy_to_mram(uint32_t heap, uint32_t mram_offset,
                               const void *wram, uint32_t bytes) {{
  uint32_t cursor = 0u;
  while (cursor < bytes) {{
    uint32_t chunk = bytes - cursor;
    if (chunk > {UPMEM_MAX_DMA_BYTES}u) chunk = {UPMEM_MAX_DMA_BYTES}u;
    mram_write((const void *)((const uint8_t *)wram + cursor),
               (__mram_ptr void *)(uintptr_t)(heap + mram_offset + cursor),
               chunk);
    cursor += chunk;
  }}
}}

int main(void) {{
  uint32_t tid = me();
  uint32_t heap = (uint32_t)DPU_MRAM_HEAP_POINTER;
  if (tid == 0) {{
    mem_reset();
    tenon_copy_from_mram(heap, {initial_offset}u, tenon_centroid_cache,
                         {regions['initial_centroids'].bytes}u);
  }}
  for (uint32_t cluster = 0; cluster < TENON_CLUSTERS; ++cluster) {{
    tenon_private_counts[tid][cluster] = 0u;
    for (uint32_t feature = 0; feature < TENON_DIMENSION; ++feature)
      tenon_private_sums[tid][cluster][feature] = 0;
  }}
  barrier_wait(&tenon_barrier);

  uint32_t begin = tid * TENON_LOCAL_POINTS;
  uint32_t end = begin + TENON_LOCAL_POINTS;
  if (end > TENON_POINTS) end = TENON_POINTS;
  for (uint32_t point = begin; point < end; ++point) {{
    mram_read((const __mram_ptr void *)(uintptr_t)(heap + {point_offset}u +
                                                   point * {4 * d}u),
              tenon_point[tid], {4 * d}u);
    uint32_t best_cluster = 0u;
    uint64_t best_distance = 0u;
    for (uint32_t cluster = 0; cluster < TENON_CLUSTERS; ++cluster) {{
      uint64_t distance = 0u;
      for (uint32_t feature = 0; feature < TENON_DIMENSION; ++feature) {{
        int64_t delta = (int64_t)tenon_point[tid][feature] -
                        (int64_t)tenon_centroid_cache[
                            cluster * TENON_DIMENSION + feature];
        distance += (uint64_t)(delta * delta);
      }}
      if (cluster == 0u || distance < best_distance) {{
        best_distance = distance;
        best_cluster = cluster;
      }}
    }}
    tenon_private_counts[tid][best_cluster] += 1u;
    for (uint32_t feature = 0; feature < TENON_DIMENSION; ++feature)
      tenon_private_sums[tid][best_cluster][feature] +=
          (int64_t)tenon_point[tid][feature];
  }}
  barrier_wait(&tenon_barrier);

  if (tid == 0) {{
    for (uint32_t cluster = 0; cluster < TENON_CLUSTERS; ++cluster) {{
      uint32_t count = 0u;
      for (uint32_t owner = 0; owner < TENON_NUM_TASKLETS; ++owner)
        count += tenon_private_counts[owner][cluster];
      tenon_output_counts[cluster] = count;
      for (uint32_t feature = 0; feature < TENON_DIMENSION; ++feature) {{
        int32_t sum = 0;
        for (uint32_t owner = 0; owner < TENON_NUM_TASKLETS; ++owner)
          sum += tenon_private_sums[owner][cluster][feature];
        tenon_output_centroids[cluster * TENON_DIMENSION + feature] =
            count == 0u ? 0 :
            tenon_div_round_closest(sum, (int32_t)count);
      }}
    }}
    for (uint32_t word = TENON_CLUSTERS * TENON_DIMENSION;
         word < {centroid_words}u; ++word)
      tenon_output_centroids[word] = 0;
    for (uint32_t word = TENON_CLUSTERS; word < {count_words}u; ++word)
      tenon_output_counts[word] = 0u;
    tenon_copy_to_mram(heap, {centroid_output.offset}u,
                        tenon_output_centroids, {centroid_output.bytes}u);
    tenon_copy_to_mram(heap, {count_output.offset}u,
                        tenon_output_counts, {count_output.bytes}u);
  }}
  return 0;
}}
"""


@dataclass(frozen=True)
class UPMEMKMeansDistancesPlan(_UPMEMIrregularPlan):
    """Fair archived-ATiM partition: fused-packet squared distances only.

    Input packets are ordered point-major then cluster-major and contain one
    complete point followed by one complete centroid.  Each tasklet owns one
    contiguous group of packets, performs exactly one packet DMA per pair,
    accumulates a statically bounded int32 squared distance, and performs one
    grouped int64 output DMA.  Argmin, counts, and centroid update deliberately
    remain outside this plan because they are outside the archived ATiM DPU
    counter as well.
    """

    points: int
    dimension: int
    clusters: int
    max_abs_value: int = 50
    num_tasklets: int = UPMEM_IRREGULAR_TASKLETS

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", _positive_int(self.points, "points"))
        object.__setattr__(
            self, "dimension", _positive_int(self.dimension, "dimension")
        )
        object.__setattr__(self, "clusters", _positive_int(self.clusters, "clusters"))
        object.__setattr__(
            self,
            "max_abs_value",
            _nonnegative_int(self.max_abs_value, "max_abs_value"),
        )
        object.__setattr__(self, "num_tasklets", _validate_tasklets(self.num_tasklets))
        if self.total_pairs % self.num_tasklets:
            raise ValueError(
                "k-means distance pairs must divide evenly across 12 tasklets"
            )
        if not UPMEM_MIN_DMA_BYTES <= self.packet_bytes <= UPMEM_MAX_DMA_BYTES:
            raise ValueError(
                "k-means fused packet DMA must be between 8 and 2048 bytes"
            )
        if self.packet_bytes % UPMEM_DMA_ALIGNMENT:
            raise ValueError("k-means fused packet DMA must be 8-byte aligned")
        if not UPMEM_MIN_DMA_BYTES <= self.grouped_output_bytes <= UPMEM_MAX_DMA_BYTES:
            raise ValueError(
                "k-means grouped output DMA must be between 8 and 2048 bytes"
            )
        if self.grouped_output_bytes % UPMEM_DMA_ALIGNMENT:
            raise ValueError("k-means grouped output DMA must be 8-byte aligned")
        if self.distance_upper_bound > (1 << 31) - 1:
            raise ValueError(
                "declared k-means distance bound does not fit signed int32: "
                f"dimension * (2 * max_abs_value)^2 = {self.distance_upper_bound}"
            )
        _validate_regions(self.mram_regions)
        _validate_wram(self.wram_bytes)

    @property
    def total_pairs(self) -> int:
        return self.points * self.clusters

    @property
    def pairs_per_tasklet(self) -> int:
        return self.total_pairs // self.num_tasklets

    @property
    def packet_words(self) -> int:
        return 2 * self.dimension

    @property
    def packet_bytes(self) -> int:
        return 4 * self.packet_words

    @property
    def grouped_output_bytes(self) -> int:
        return 8 * self.pairs_per_tasklet

    @property
    def distance_upper_bound(self) -> int:
        return self.dimension * (2 * self.max_abs_value) ** 2

    @property
    def linear_layout(self) -> LinearLayout:
        return _masked_padded_16_layout(
            self.total_pairs, logical_unit="point-centroid-pair"
        )[0]

    @property
    def layout_manifest(self) -> dict[str, object]:
        return _masked_padded_16_layout(
            self.total_pairs, logical_unit="point-centroid-pair"
        )[1]

    @property
    def mram_regions(self) -> dict[str, MRAMRegion]:
        input_bytes = self.total_pairs * self.packet_bytes
        output_bytes = 8 * self.total_pairs
        return {
            "fused_pairs": MRAMRegion(
                "fused_pairs",
                0,
                input_bytes,
                "input",
                "int32",
                (self.points, self.clusters, 2, self.dimension),
                input_bytes,
            ),
            "distances": MRAMRegion(
                "distances",
                input_bytes,
                output_bytes,
                "output",
                "int64",
                (self.points, self.clusters),
                output_bytes,
            ),
        }

    @property
    def output_contract(self) -> dict[str, object]:
        distances = self.mram_regions["distances"]
        return {
            "kind": "exact-kmeans-squared-distances",
            "combined_offset": distances.offset,
            "combined_bytes": distances.bytes,
            "dtype": "int64",
            "shape": [self.points, self.clusters],
            "order": "point-major-cluster-major",
            "equation": ("distances[p,c] = sum_d " "(point[p,d] - centroid[c,d])^2"),
            "device_partition": "distance-generation-only",
            "host_postprocessing": ["argmin", "counts", "centroid_update"],
            "result": distances.manifest(),
        }

    @property
    def wram_bytes(self) -> int:
        tasklet_buffers = self.num_tasklets * (
            self.packet_bytes + self.grouped_output_bytes
        )
        stacks = _STACK_RESERVE_PER_TASKLET * self.num_tasklets
        return tasklet_buffers + stacks + _WRAM_RUNTIME_RESERVE

    @property
    def cost_features(self) -> dict[str, object]:
        distance_terms = self.total_pairs * self.dimension
        return {
            "tasklet_fanout": self.num_tasklets,
            "layout_tasklet_extent": UPMEM_LAYOUT_TASKLETS,
            "logical_work_items": self.total_pairs,
            "pairs_per_tasklet": self.pairs_per_tasklet,
            "wram_bytes": self.wram_bytes,
            "barriers": 1,
            "comparison_partition": "device_squared_distances_only",
            "primitive_iterations": {
                "SUB": distance_terms,
                "MUL": distance_terms,
                "ADD": distance_terms,
            },
            "dma_bytes": {
                "LD_MRAM": self.total_pairs * self.packet_bytes,
                "ST_MRAM": 8 * self.total_pairs,
            },
            "dma_calls": {
                "LD_MRAM": self.total_pairs,
                "ST_MRAM": self.num_tasklets,
            },
        }

    def manifest(self) -> dict[str, object]:
        return {
            **self._base_manifest("upmem-kmeans-squared-distances"),
            "points": self.points,
            "dimension": self.dimension,
            "clusters": self.clusters,
            "total_pairs": self.total_pairs,
            "pairs_per_tasklet": self.pairs_per_tasklet,
            "packet": {
                "order": ["point", "centroid"],
                "words": self.packet_words,
                "bytes": self.packet_bytes,
            },
            "grouped_output_bytes": self.grouped_output_bytes,
            "distance_bound": {
                "input_dtype": "int32",
                "input_range": [-self.max_abs_value, self.max_abs_value],
                "accumulator_dtype": "int32",
                "upper_bound": self.distance_upper_bound,
                "required_maximum": (1 << 31) - 1,
            },
            "comparison_partition": {
                "device": "squared_distances",
                "host": ["argmin", "counts", "centroid_update"],
                "full_iteration_plan": "UPMEMKMeansPlan",
            },
            "algorithm": [
                "contiguous_pair_ownership",
                "one_fused_packet_dma_per_pair",
                "bounded_int32_squared_distance",
                "one_grouped_int64_write_per_tasklet",
            ],
        }

    def pack_fused_pairs(self, point_values, centroid_values) -> tuple[int, ...]:
        """Pack ordinary point/centroid arrays into the runtime MRAM order."""

        points = tuple(int(value) for value in point_values)
        centroids = tuple(int(value) for value in centroid_values)
        if len(points) != self.points * self.dimension:
            raise ValueError("point_values has the wrong flattened extent")
        if len(centroids) != self.clusters * self.dimension:
            raise ValueError("centroid_values has the wrong flattened extent")
        self._validate_input_bound(points + centroids)
        return tuple(
            value
            for point in range(self.points)
            for cluster in range(self.clusters)
            for value in (
                points[point * self.dimension : (point + 1) * self.dimension]
                + centroids[cluster * self.dimension : (cluster + 1) * self.dimension]
            )
        )

    def _validate_input_bound(self, values) -> None:
        if any(abs(int(value)) > self.max_abs_value for value in values):
            raise ValueError(
                "runtime k-means value exceeds the declared max_abs_value contract"
            )

    def reference_distances(
        self, fused_or_points, centroid_values=None
    ) -> tuple[int, ...]:
        """Evaluate exact point-major/cluster-major distance outputs.

        With one argument, ``fused_or_points`` is the physical packet stream.
        With two arguments, ordinary flattened points and centroids are packed
        first using :meth:`pack_fused_pairs`.
        """

        if centroid_values is None:
            fused = tuple(int(value) for value in fused_or_points)
            expected = self.total_pairs * self.packet_words
            if len(fused) != expected:
                raise ValueError(f"expected {expected} fused int32 packet words")
            self._validate_input_bound(fused)
        else:
            fused = self.pack_fused_pairs(fused_or_points, centroid_values)
        return tuple(
            sum(
                (
                    fused[pair * self.packet_words + feature]
                    - fused[pair * self.packet_words + self.dimension + feature]
                )
                ** 2
                for feature in range(self.dimension)
            )
            for pair in range(self.total_pairs)
        )

    def device_source(self) -> str:
        input_region = self.mram_regions["fused_pairs"]
        output_region = self.mram_regions["distances"]
        p, d, k = self.points, self.dimension, self.clusters
        pairs, local = self.total_pairs, self.pairs_per_tasklet
        nt = self.num_tasklets
        return f"""// Tenon archived-partition k-means distances for UPMEM.
// Runtime ABI: fused [point[{d}], centroid[{d}]] packets[{pairs}]
//           -> int64 distances[{p},{k}] in point-major/cluster-major order.
#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <stdint.h>

#if !defined(NR_TASKLETS) || NR_TASKLETS != {nt}
#error "Tenon irregular UPMEM plan requires NR_TASKLETS={nt}"
#endif

#define TENON_NUM_TASKLETS {nt}
#define TENON_LAYOUT_TASKLETS {UPMEM_LAYOUT_TASKLETS}
#define TENON_DIMENSION {d}u
#define TENON_PACKET_WORDS {self.packet_words}u
#define TENON_PACKET_BYTES {self.packet_bytes}u
#define TENON_LOCAL_PAIRS {local}u
#define TENON_GROUPED_OUTPUT_BYTES {self.grouped_output_bytes}u
BARRIER_INIT(tenon_barrier, NR_TASKLETS);

int main(void) {{
  const uint32_t tid = me();
  const uint32_t heap = (uint32_t)DPU_MRAM_HEAP_POINTER;
  if (tid == 0) mem_reset();
  barrier_wait(&tenon_barrier);
  int32_t *packet =
      (int32_t *)mem_alloc(TENON_PACKET_BYTES);
  int64_t *distances =
      (int64_t *)mem_alloc(TENON_GROUPED_OUTPUT_BYTES);
  const uint32_t pair_begin = tid * TENON_LOCAL_PAIRS;
  for (uint32_t local_pair = 0; local_pair < TENON_LOCAL_PAIRS;
       ++local_pair) {{
    const uint32_t pair = pair_begin + local_pair;
    mram_read((const __mram_ptr void *)(uintptr_t)(
                  heap + {input_region.offset}u + pair * TENON_PACKET_BYTES),
              packet, TENON_PACKET_BYTES);
    int32_t distance = 0;
    for (uint32_t feature = 0; feature < TENON_DIMENSION; ++feature) {{
      const int32_t delta = packet[feature] - packet[TENON_DIMENSION + feature];
      distance += delta * delta;
    }}
    distances[local_pair] = (int64_t)distance;
  }}
  mram_write(distances,
             (__mram_ptr void *)(uintptr_t)(
                 heap + {output_region.offset}u +
                 pair_begin * sizeof(int64_t)),
             TENON_GROUPED_OUTPUT_BYTES);
  return 0;
}}
"""


class GradientFormula(str, Enum):
    """Exact feature-gradient formula selected by a physical plan."""

    LINEAR_FIXED_POINT = "linear_fixed_point"
    LOGISTIC_ZERO = "logistic_zero"


def _gradient_formula(value: GradientFormula | str) -> GradientFormula:
    aliases = {
        "linear": GradientFormula.LINEAR_FIXED_POINT,
        "linear_fixed_point": GradientFormula.LINEAR_FIXED_POINT,
        "logistic": GradientFormula.LOGISTIC_ZERO,
        "logistic_zero": GradientFormula.LOGISTIC_ZERO,
        "logistic_at_zero": GradientFormula.LOGISTIC_ZERO,
    }
    if isinstance(value, GradientFormula):
        return value
    try:
        return aliases[str(value)]
    except KeyError as exc:
        raise ValueError(
            "gradient formula must be 'linear_fixed_point' or 'logistic_zero'"
        ) from exc


@dataclass(frozen=True)
class UPMEMFeatureGradientPlan(_UPMEMIrregularPlan):
    """Feature-parallel gradient over packed ``[features..., label]`` rows."""

    samples: int
    features: int
    formula: GradientFormula | str
    shift: int = 5
    overflow_shift: int = 8
    num_tasklets: int = UPMEM_IRREGULAR_TASKLETS

    def __post_init__(self) -> None:
        object.__setattr__(self, "samples", _positive_int(self.samples, "samples"))
        object.__setattr__(self, "features", _positive_int(self.features, "features"))
        object.__setattr__(self, "formula", _gradient_formula(self.formula))
        object.__setattr__(self, "shift", _nonnegative_int(self.shift, "shift"))
        object.__setattr__(
            self,
            "overflow_shift",
            _nonnegative_int(self.overflow_shift, "overflow_shift"),
        )
        object.__setattr__(self, "num_tasklets", _validate_tasklets(self.num_tasklets))
        if self.shift > 30:
            raise ValueError("linear fixed-point shift must be at most 30")
        if self.overflow_shift > 62:
            raise ValueError("linear overflow_shift must be at most 62")
        if self.record_dma_bytes > UPMEM_MAX_DMA_BYTES:
            raise ValueError(
                "one packed feature record exceeds the 2048-byte DMA maximum"
            )
        _validate_regions(self.mram_regions)
        _validate_wram(self.wram_bytes)

    @property
    def local_samples(self) -> int:
        return _ceil_div(self.samples, self.num_tasklets)

    @property
    def record_words(self) -> int:
        return self.features + 1

    @property
    def record_bytes(self) -> int:
        return 4 * self.record_words

    @property
    def record_dma_bytes(self) -> int:
        # Odd-word rows alternate between aligned and four-byte-offset starts.
        # One extra word makes both cases one fixed, aligned DMA transaction.
        return (
            self.record_bytes if self.record_bytes % 8 == 0 else self.record_bytes + 4
        )

    @property
    def linear_layout(self) -> LinearLayout:
        return _masked_padded_16_layout(self.samples, logical_unit="feature-sample")[0]

    @property
    def layout_manifest(self) -> dict[str, object]:
        return _masked_padded_16_layout(self.samples, logical_unit="feature-sample")[1]

    @property
    def mram_regions(self) -> dict[str, MRAMRegion]:
        input_logical = self.samples * self.record_bytes
        input_bytes = _align_up(input_logical)
        output_bytes = 8 * self.features
        return {
            "samples": MRAMRegion(
                "samples",
                0,
                input_bytes,
                "input",
                "int32",
                (self.samples, self.record_words),
                input_logical,
            ),
            "gradient": MRAMRegion(
                "gradient",
                input_bytes,
                output_bytes,
                "output",
                "int64",
                (self.features,),
                output_bytes,
            ),
        }

    @property
    def output_contract(self) -> dict[str, object]:
        gradient = self.mram_regions["gradient"]
        if self.formula is GradientFormula.LINEAR_FIXED_POINT:
            equation = (
                "gradient[j] = sum_i floor_div_pow2("
                "x[i,j] * (-(y[i] * 2^shift)), overflow_shift)"
            )
        else:
            equation = "gradient[j] = sum_i x[i,j] * (1 - 2*y[i])"
        return {
            "kind": "exact-feature-gradient",
            "combined_offset": gradient.offset,
            "combined_bytes": gradient.bytes,
            "formula": self.formula.value,
            "equation": equation,
            "result": gradient.manifest(),
        }

    @property
    def wram_bytes(self) -> int:
        sample_buffers = self.num_tasklets * self.record_dma_bytes
        private_gradients = 8 * self.num_tasklets * self.features
        merged_gradient = 8 * self.features
        stacks = _STACK_RESERVE_PER_TASKLET * self.num_tasklets
        return (
            sample_buffers
            + private_gradients
            + merged_gradient
            + stacks
            + _WRAM_RUNTIME_RESERVE
        )

    @property
    def cost_features(self) -> dict[str, object]:
        terms = self.samples * self.features
        if self.formula is GradientFormula.LINEAR_FIXED_POINT:
            primitive_iterations = {
                "MUL": 2 * terms,
                "ADD": terms + self.features * self.num_tasklets,
                "BRANCH": terms,
            }
        else:
            primitive_iterations = {
                "MUL": 2 * terms,
                "SUB": terms,
                "ADD": terms + self.features * self.num_tasklets,
            }
        return {
            "tasklet_fanout": self.num_tasklets,
            "layout_tasklet_extent": UPMEM_LAYOUT_TASKLETS,
            "logical_work_items": self.samples,
            "formula": self.formula.value,
            "wram_bytes": self.wram_bytes,
            "barriers": 2,
            "primitive_iterations": primitive_iterations,
            "dma_bytes": {
                "LD_MRAM": self.samples * self.record_dma_bytes,
                "ST_MRAM": 8 * self.features,
            },
            "dma_calls": {
                "LD_MRAM": self.samples,
                "ST_MRAM": _ceil_div(8 * self.features, UPMEM_MAX_DMA_BYTES),
            },
        }

    def manifest(self) -> dict[str, object]:
        result = {
            **self._base_manifest("upmem-feature-gradient"),
            "samples": self.samples,
            "features": self.features,
            "packed_record_words": self.record_words,
            "record_dma_bytes": self.record_dma_bytes,
            "formula": self.formula.value,
            "algorithm": [
                "packed_record_load",
                "tasklet_private_feature_gradient",
                "feature_merge",
                "aligned_int64_write",
            ],
        }
        if self.formula is GradientFormula.LINEAR_FIXED_POINT:
            result["fixed_point"] = {
                "label_shift": self.shift,
                "product_floor_shift": self.overflow_shift,
                "lowering": "upmem-signed-arithmetic-right-shift",
            }
        return result

    def reference_gradient(self, packed_samples) -> tuple[int, ...]:
        """Evaluate the exact mathematical output without NumPy."""

        values = tuple(int(value) for value in packed_samples)
        expected = self.samples * self.record_words
        if len(values) != expected:
            raise ValueError(f"expected {expected} packed int32 values")
        gradient = [0] * self.features
        for sample in range(self.samples):
            base = sample * self.record_words
            label = values[base + self.features]
            for feature in range(self.features):
                x = values[base + feature]
                if self.formula is GradientFormula.LINEAR_FIXED_POINT:
                    product = x * (-(label * (1 << self.shift)))
                    gradient[feature] += product // (1 << self.overflow_shift)
                else:
                    gradient[feature] += x * (1 - 2 * label)
        return tuple(gradient)

    def device_source(self) -> str:
        sample_region = self.mram_regions["samples"]
        gradient_region = self.mram_regions["gradient"]
        s, f = self.samples, self.features
        nt, local = self.num_tasklets, self.local_samples
        record_words = self.record_words
        dma_words = self.record_dma_bytes // 4
        if self.formula is GradientFormula.LINEAR_FIXED_POINT:
            formula_helpers = ""
            formula_body = f"""      int64_t product =
          (int64_t)tenon_sample[tid][shift_words + feature] * (int64_t)label;
      tenon_private_gradient[tid][feature] +=
          (product * -((int64_t)1 << {self.shift})) >> {self.overflow_shift};"""
        else:
            formula_helpers = ""
            formula_body = """      int64_t factor = (int64_t)1 - (int64_t)2 * (int64_t)label;
      tenon_private_gradient[tid][feature] +=
          (int64_t)tenon_sample[tid][shift_words + feature] * factor;"""
        return f"""// Tenon {self.formula.value} feature-gradient physical plan for UPMEM.
// Runtime ABI: packed int32 samples[{s}][{record_words}] -> int64 gradient[{f}].
#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <stdint.h>

#if !defined(NR_TASKLETS) || NR_TASKLETS != {nt}
#error "Tenon irregular UPMEM plan requires NR_TASKLETS={nt}"
#endif

#define TENON_NUM_TASKLETS {nt}
#define TENON_LAYOUT_TASKLETS {UPMEM_LAYOUT_TASKLETS}
#define TENON_SAMPLES {s}u
#define TENON_FEATURES {f}u
#define TENON_RECORD_BYTES {self.record_bytes}u
#define TENON_RECORD_DMA_BYTES {self.record_dma_bytes}u
#define TENON_LOCAL_SAMPLES {local}u

__dma_aligned int32_t tenon_sample[{nt}][{dma_words}];
__dma_aligned int64_t tenon_private_gradient[{nt}][{f}];
__dma_aligned int64_t tenon_gradient[{f}];
BARRIER_INIT(tenon_barrier, NR_TASKLETS);
{formula_helpers}
int main(void) {{
  uint32_t tid = me();
  uint32_t heap = (uint32_t)DPU_MRAM_HEAP_POINTER;
  if (tid == 0) mem_reset();
  for (uint32_t feature = 0; feature < TENON_FEATURES; ++feature)
    tenon_private_gradient[tid][feature] = 0;
  barrier_wait(&tenon_barrier);

  uint32_t begin = tid * TENON_LOCAL_SAMPLES;
  uint32_t end = begin + TENON_LOCAL_SAMPLES;
  if (end > TENON_SAMPLES) end = TENON_SAMPLES;
  for (uint32_t sample = begin; sample < end; ++sample) {{
    uint32_t record_begin = sample * TENON_RECORD_BYTES;
    uint32_t aligned_begin = record_begin & ~7u;
    uint32_t shift_words = (record_begin - aligned_begin) / sizeof(int32_t);
    mram_read((const __mram_ptr void *)(uintptr_t)(heap + {sample_region.offset}u +
                                                   aligned_begin),
              tenon_sample[tid], TENON_RECORD_DMA_BYTES);
    int32_t label = tenon_sample[tid][shift_words + TENON_FEATURES];
    for (uint32_t feature = 0; feature < TENON_FEATURES; ++feature) {{
{formula_body}
    }}
  }}
  barrier_wait(&tenon_barrier);

  for (uint32_t feature = tid; feature < TENON_FEATURES;
       feature += TENON_NUM_TASKLETS) {{
    int64_t total = 0;
    for (uint32_t owner = 0; owner < TENON_NUM_TASKLETS; ++owner)
      total += tenon_private_gradient[owner][feature];
    tenon_gradient[feature] = total;
  }}
  barrier_wait(&tenon_barrier);

  if (tid == 0) {{
    uint32_t cursor = 0u;
    while (cursor < {gradient_region.bytes}u) {{
      uint32_t bytes = {gradient_region.bytes}u - cursor;
      if (bytes > {UPMEM_MAX_DMA_BYTES}u) bytes = {UPMEM_MAX_DMA_BYTES}u;
      mram_write((const void *)((const uint8_t *)tenon_gradient + cursor),
                 (__mram_ptr void *)(uintptr_t)(heap + {gradient_region.offset}u + cursor),
                 bytes);
      cursor += bytes;
    }}
  }}
  return 0;
}}
"""


# Explicit aliases make the physical-plan role discoverable while preserving
# concise names for workload modules.
UPMEMHistogramPhysicalPlan = UPMEMHistogramPlan
UPMEMSelectionPlan = UPMEMStableSelectionPlan
UPMEMSelectionFlagsPhysicalPlan = UPMEMSelectionFlagsPlan
UPMEMStableSelectionPhysicalPlan = UPMEMStableSelectionPlan
UPMEMKMeansDistancePlan = UPMEMKMeansDistancesPlan
UPMEMKMeansDistancesPhysicalPlan = UPMEMKMeansDistancesPlan
UPMEMKMeansOneIterationPlan = UPMEMKMeansPlan
UPMEMKMeansPhysicalPlan = UPMEMKMeansPlan
UPMEMFeatureGradientPhysicalPlan = UPMEMFeatureGradientPlan


__all__ = [
    "GradientFormula",
    "MRAMRegion",
    "UPMEMFeatureGradientPhysicalPlan",
    "UPMEMFeatureGradientPlan",
    "UPMEMHistogramPhysicalPlan",
    "UPMEMHistogramPlan",
    "UPMEMKMeansDistancePlan",
    "UPMEMKMeansDistancesPhysicalPlan",
    "UPMEMKMeansDistancesPlan",
    "UPMEMKMeansOneIterationPlan",
    "UPMEMKMeansPhysicalPlan",
    "UPMEMKMeansPlan",
    "UPMEMSelectionPlan",
    "UPMEMSelectionFlagsPhysicalPlan",
    "UPMEMSelectionFlagsPlan",
    "UPMEMStableSelectionPhysicalPlan",
    "UPMEMStableSelectionPlan",
    "UPMEM_IRREGULAR_TASKLETS",
    "UPMEM_LAYOUT_TASKLETS",
]
