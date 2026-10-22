# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Linear-layout contracts and packing helpers for Gemini-II VL64.

The milestone-one layout describes a complete four-vector pack.  Reduction
workloads additionally need a validity-masked logical view because their row
and reduction extents need not be powers of two.  :class:`APUG2ReductionPlan`
keeps that logical shape beside an exact F2 carrier and is the single source
of truth for host packing and result gathering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from types import MappingProxyType

import numpy as np

from ..spmw_linear_layout import LinearLayout


APUG2_GROUPS = 16
APUG2_LANES_PER_GROUP = 4096
APUG2_VECTOR_LANES = APUG2_GROUPS * APUG2_LANES_PER_GROUP
APUG2_MMB_SETS = 4
APUG2_U16_SHAPE = (APUG2_MMB_SETS, APUG2_VECTOR_LANES)


def _positive_extent(name: str, value) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _freeze_layout(layout: LinearLayout) -> LinearLayout:
    """Retain an immutable snapshot of a legacy mutable ``LinearLayout``."""

    frozen = LinearLayout(
        {
            name: [tuple(vector) for vector in vectors]
            for name, vectors in layout.bases.items()
        },
        tuple(layout.out_dims),
        tuple(layout.out_sizes),
    )
    frozen.bases = MappingProxyType(
        {name: tuple(vectors) for name, vectors in frozen.bases.items()}
    )
    return frozen


@dataclass(frozen=True)
class APUG2ReductionPlan:
    """Padded F2 layout for independent reductions over logical output rows.

    Independent matrix streams occupy MMB sets while rows are striped across
    the sixteen 4K-column L1 groups.  Additional rows occupy contiguous
    reduction-sized slots inside each group::

        mmb_set = stream
        group = row & 15
        slot = row >> 4
        l1_column = group * 4096 + slot * padded_reduction_extent + reduction

    Both logical extents are padded to powers of two so this mapping remains
    representable by :class:`LinearLayout`.  Packing helpers explicitly leave
    the non-logical padded domain zero.
    """

    output_extent: int
    reduction_extent: int
    stream_extent: int = 1
    padded_output_extent: int = field(init=False)
    padded_reduction_extent: int = field(init=False)
    padded_stream_extent: int = field(init=False)
    layout: LinearLayout = field(init=False, repr=False)

    def __post_init__(self):
        output_extent = _positive_extent("output_extent", self.output_extent)
        reduction_extent = _positive_extent("reduction_extent", self.reduction_extent)
        stream_extent = _positive_extent("stream_extent", self.stream_extent)
        if stream_extent > APUG2_MMB_SETS:
            raise ValueError(f"stream_extent must not exceed {APUG2_MMB_SETS} MMB sets")
        padded_output = _next_power_of_two(output_extent)
        padded_reduction = _next_power_of_two(reduction_extent)
        padded_stream = _next_power_of_two(stream_extent)

        # Every stream uses the same column image in its own MMB set.  One
        # group owns this many padded output rows; their blocks must fit
        # entirely inside 4K columns so sum(log_block_size) never crosses a
        # group boundary.
        slots_per_group = max(1, padded_output // APUG2_GROUPS)
        columns_per_group = slots_per_group * padded_reduction
        if columns_per_group > APUG2_LANES_PER_GROUP:
            raise ValueError(
                "padded APUg2 reduction layout does not fit: "
                f"{slots_per_group} row slots * {padded_reduction} "
                f"reduction values needs {columns_per_group} columns per "
                f"group, limit is {APUG2_LANES_PER_GROUP}"
            )

        output_bits = padded_output.bit_length() - 1
        reduction_bits = padded_reduction.bit_length() - 1
        row_bases = []
        for bit in range(output_bits):
            if bit < 4:
                # row % 16 selects one of the sixteen L1 groups.
                row_bases.append((APUG2_LANES_PER_GROUP << bit, 0))
            else:
                # Remaining row bits select a reduction-aligned in-group slot.
                row_bases.append((padded_reduction << (bit - 4), 0))

        layout = LinearLayout(
            bases={
                "stream": [
                    ((0, 1 << bit)) for bit in range(padded_stream.bit_length() - 1)
                ],
                "row": row_bases,
                "reduction": [((1 << bit), 0) for bit in range(reduction_bits)],
            },
            out_dims=("l1_column", "mmb_set"),
            out_sizes=(APUG2_VECTOR_LANES, APUG2_MMB_SETS),
        )
        object.__setattr__(self, "output_extent", output_extent)
        object.__setattr__(self, "reduction_extent", reduction_extent)
        object.__setattr__(self, "stream_extent", stream_extent)
        object.__setattr__(self, "padded_output_extent", padded_output)
        object.__setattr__(self, "padded_reduction_extent", padded_reduction)
        object.__setattr__(self, "padded_stream_extent", padded_stream)
        object.__setattr__(self, "layout", _freeze_layout(layout))

    @property
    def log_block_size(self) -> int:
        """The direct-VL64 ``sum`` block-size argument."""

        return self.padded_reduction_extent.bit_length() - 1

    @property
    def active_block_count(self) -> int:
        """Number of distinct L1 column blocks containing logical rows.

        Streams share the same column image in different MMB sets, so they do
        not multiply the count needed by the broadcast L1 vector.
        """

        return self.output_extent

    def physical_coordinate(
        self, row: int, reduction: int = 0, *, stream: int = 0
    ) -> tuple[int, int]:
        """Map one logical coordinate to ``(l1_column, mmb_set)``."""

        if isinstance(stream, (bool, np.bool_)) or not isinstance(stream, Integral):
            raise TypeError("stream must be an integer")
        if isinstance(row, (bool, np.bool_)) or not isinstance(row, Integral):
            raise TypeError("row must be an integer")
        if isinstance(reduction, (bool, np.bool_)) or not isinstance(
            reduction, Integral
        ):
            raise TypeError("reduction must be an integer")
        stream = int(stream)
        row = int(row)
        reduction = int(reduction)
        if not 0 <= stream < self.stream_extent:
            raise IndexError(
                f"stream {stream} is outside logical extent {self.stream_extent}"
            )
        if not 0 <= row < self.output_extent:
            raise IndexError(
                f"row {row} is outside logical output extent {self.output_extent}"
            )
        if not 0 <= reduction < self.reduction_extent:
            raise IndexError(
                "reduction index "
                f"{reduction} is outside logical extent {self.reduction_extent}"
            )
        return self.layout.apply(stream=stream, row=row, reduction=reduction)

    def pack_matrix(self, matrix, *, stream: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Pack a logical uint16 matrix into low/high physical byte packs.

        Both returned arrays have shape ``(4, 65536)`` and dtype ``uint8``.
        Coordinates outside the logical matrix, including power-of-two
        padding, remain zero.
        """

        if isinstance(stream, (bool, np.bool_)) or not isinstance(stream, Integral):
            raise TypeError("stream must be an integer")
        stream = int(stream)
        if not 0 <= stream < self.stream_extent:
            raise IndexError(
                f"stream {stream} is outside logical extent {self.stream_extent}"
            )
        matrix = np.asarray(matrix)
        expected_shape = (self.output_extent, self.reduction_extent)
        if matrix.dtype != np.dtype(np.uint16):
            raise TypeError(f"matrix must have dtype uint16, got {matrix.dtype}")
        if matrix.shape != expected_shape:
            raise ValueError(
                f"matrix must have shape {expected_shape}, got {matrix.shape}"
            )

        low = np.zeros(APUG2_U16_SHAPE, dtype=np.uint8)
        high = np.zeros(APUG2_U16_SHAPE, dtype=np.uint8)
        reductions = np.arange(self.reduction_extent, dtype=np.int64)
        for row in range(self.output_extent):
            column, mmb_set = self.layout.apply(stream=stream, row=row, reduction=0)
            columns = column + reductions
            values = matrix[row]
            low[mmb_set, columns] = (values & np.uint16(0xFF)).astype(np.uint8)
            high[mmb_set, columns] = (values >> np.uint16(8)).astype(np.uint8)
        return low, high

    def pack_matrices(self, matrices) -> tuple[np.ndarray, np.ndarray]:
        """Pack all logical matrix streams into their corresponding MMB sets."""

        matrices = np.asarray(matrices)
        expected_shape = (
            self.stream_extent,
            self.output_extent,
            self.reduction_extent,
        )
        if matrices.dtype != np.dtype(np.uint16):
            raise TypeError(f"matrices must have dtype uint16, got {matrices.dtype}")
        if matrices.shape != expected_shape:
            raise ValueError(
                f"matrices must have shape {expected_shape}, got {matrices.shape}"
            )

        low = np.zeros(APUG2_U16_SHAPE, dtype=np.uint8)
        high = np.zeros(APUG2_U16_SHAPE, dtype=np.uint8)
        reductions = np.arange(self.reduction_extent, dtype=np.int64)
        for stream in range(self.stream_extent):
            for row in range(self.output_extent):
                column, mmb_set = self.layout.apply(stream=stream, row=row, reduction=0)
                columns = column + reductions
                values = matrices[stream, row]
                low[mmb_set, columns] = (values & np.uint16(0xFF)).astype(np.uint8)
                high[mmb_set, columns] = (values >> np.uint16(8)).astype(np.uint8)
        return low, high

    def pack_broadcast_vector(self, vector) -> tuple[np.ndarray, np.ndarray]:
        """Broadcast one logical uint16 reduction vector over active blocks.

        The returned one-dimensional byte vectors model the single
        :class:`gsi::g2_64vl::L1Vector` operand shared by all four MMB sets.
        A block is populated if at least one logical output row maps to it.
        """

        vector = np.asarray(vector)
        expected_shape = (self.reduction_extent,)
        if vector.dtype != np.dtype(np.uint16):
            raise TypeError(f"vector must have dtype uint16, got {vector.dtype}")
        if vector.shape != expected_shape:
            raise ValueError(
                f"vector must have shape {expected_shape}, got {vector.shape}"
            )

        low = np.zeros(APUG2_VECTOR_LANES, dtype=np.uint8)
        high = np.zeros(APUG2_VECTOR_LANES, dtype=np.uint8)
        value_low = (vector & np.uint16(0xFF)).astype(np.uint8)
        value_high = (vector >> np.uint16(8)).astype(np.uint8)
        populated_blocks = set()
        for row in range(self.output_extent):
            column, _mmb_set = self.layout.apply(stream=0, row=row, reduction=0)
            if column in populated_blocks:
                continue
            populated_blocks.add(column)
            stop = column + self.reduction_extent
            low[column:stop] = value_low
            high[column:stop] = value_high
        return low, high

    def unpack_reduction(self, packed, *, stream: int = 0) -> np.ndarray:
        """Gather block-first uint16 reduction results in logical row order."""

        if isinstance(stream, (bool, np.bool_)) or not isinstance(stream, Integral):
            raise TypeError("stream must be an integer")
        stream = int(stream)
        if not 0 <= stream < self.stream_extent:
            raise IndexError(
                f"stream {stream} is outside logical extent {self.stream_extent}"
            )
        packed = np.asarray(packed)
        if packed.dtype != np.dtype(np.uint16):
            raise TypeError(
                f"packed reduction must have dtype uint16, got {packed.dtype}"
            )
        if packed.shape != APUG2_U16_SHAPE:
            raise ValueError(
                f"packed reduction must have shape {APUG2_U16_SHAPE}, "
                f"got {packed.shape}"
            )

        output = np.empty(self.output_extent, dtype=np.uint16)
        for row in range(self.output_extent):
            column, mmb_set = self.layout.apply(stream=stream, row=row, reduction=0)
            output[row] = packed[mmb_set, column]
        return output

    def unpack_reductions(self, packed) -> np.ndarray:
        """Gather block-first results for every logical stream."""

        return np.stack(
            [
                self.unpack_reduction(packed, stream=stream)
                for stream in range(self.stream_extent)
            ],
            axis=0,
        )


@dataclass(frozen=True)
class APUG2DotTilePlan:
    """Four-set tile for arbitrary independent uint16 dot products.

    Unlike :class:`APUG2ReductionPlan`, which reserves each MMB set for a
    matrix stream sharing one broadcast vector, this layout spends all four
    sets on independent output rows.  Both operands therefore use
    ``L1Vectors`` and may differ for every dot product::

        mmb_set = row & 3
        group = (row >> 2) & 15
        slot = row >> 6
        l1_column = group * 4096 + slot * padded_reduction + reduction

    This is the common physical carrier for GEMM, batched contractions, Gram
    products, triangular contractions, and the existing GEMV family.
    """

    output_extent: int
    reduction_extent: int
    padded_output_extent: int = field(init=False)
    padded_reduction_extent: int = field(init=False)
    output_capacity: int = field(init=False)
    layout: LinearLayout = field(init=False, repr=False)

    def __post_init__(self):
        output_extent = _positive_extent("output_extent", self.output_extent)
        reduction_extent = _positive_extent("reduction_extent", self.reduction_extent)
        padded_output = _next_power_of_two(output_extent)
        padded_reduction = _next_power_of_two(reduction_extent)
        if padded_reduction > 256:
            raise ValueError(
                "APUg2 uint16 dot reductions require padded extent at most 256"
            )
        output_capacity = APUG2_MMB_SETS * APUG2_VECTOR_LANES // padded_reduction
        if padded_output > output_capacity:
            raise ValueError(
                "APUg2 dot tile does not fit: padded output extent "
                f"{padded_output} exceeds capacity {output_capacity} for "
                f"padded reduction {padded_reduction}"
            )

        row_bases = []
        for bit in range(padded_output.bit_length() - 1):
            if bit < 2:
                row_bases.append((0, 1 << bit))
            elif bit < 6:
                row_bases.append((APUG2_LANES_PER_GROUP << (bit - 2), 0))
            else:
                row_bases.append((padded_reduction << (bit - 6), 0))
        layout = LinearLayout(
            bases={
                "row": row_bases,
                "reduction": [
                    ((1 << bit), 0) for bit in range(padded_reduction.bit_length() - 1)
                ],
            },
            out_dims=("l1_column", "mmb_set"),
            out_sizes=(APUG2_VECTOR_LANES, APUG2_MMB_SETS),
        )
        object.__setattr__(self, "output_extent", output_extent)
        object.__setattr__(self, "reduction_extent", reduction_extent)
        object.__setattr__(self, "padded_output_extent", padded_output)
        object.__setattr__(self, "padded_reduction_extent", padded_reduction)
        object.__setattr__(self, "output_capacity", output_capacity)
        object.__setattr__(self, "layout", _freeze_layout(layout))

    @property
    def log_block_size(self) -> int:
        return self.padded_reduction_extent.bit_length() - 1

    def physical_coordinate(self, row: int, reduction: int = 0) -> tuple[int, int]:
        for name, value, extent in (
            ("row", row, self.output_extent),
            ("reduction", reduction, self.reduction_extent),
        ):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if not 0 <= int(value) < extent:
                raise IndexError(f"{name} {value} is outside logical extent {extent}")
        return self.layout.apply(row=int(row), reduction=int(reduction))

    def _pack_operand(self, operand, name: str) -> tuple[np.ndarray, np.ndarray]:
        operand = np.asarray(operand)
        expected = (self.output_extent, self.reduction_extent)
        if operand.dtype != np.dtype(np.uint16):
            raise TypeError(f"{name} must have dtype uint16, got {operand.dtype}")
        if operand.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {operand.shape}")
        low = np.zeros(APUG2_U16_SHAPE, dtype=np.uint8)
        high = np.zeros(APUG2_U16_SHAPE, dtype=np.uint8)
        offsets = np.arange(self.reduction_extent, dtype=np.int64)
        for row in range(self.output_extent):
            column, mmb_set = self.layout.apply(row=row, reduction=0)
            columns = column + offsets
            values = operand[row]
            low[mmb_set, columns] = (values & np.uint16(0xFF)).astype(np.uint8)
            high[mmb_set, columns] = (values >> np.uint16(8)).astype(np.uint8)
        return low, high

    def pack_operands(self, left, right):
        """Pack two independently indexed ``[output,reduction]`` operands."""

        left_low, left_high = self._pack_operand(left, "left")
        right_low, right_high = self._pack_operand(right, "right")
        return left_low, left_high, right_low, right_high

    def unpack_reduction(self, packed) -> np.ndarray:
        packed = np.asarray(packed)
        if packed.dtype != np.dtype(np.uint16):
            raise TypeError(
                f"packed reduction must have dtype uint16, got {packed.dtype}"
            )
        if packed.shape != APUG2_U16_SHAPE:
            raise ValueError(
                f"packed reduction must have shape {APUG2_U16_SHAPE}, "
                f"got {packed.shape}"
            )
        result = np.empty(self.output_extent, dtype=np.uint16)
        for row in range(self.output_extent):
            column, mmb_set = self.layout.apply(row=row, reduction=0)
            result[row] = packed[mmb_set, column]
        return result


@dataclass(frozen=True)
class APUG2DotTiling:
    """Temporal tiling of independent dots over output and reduction axes.

    The VL64 ``sum`` result must fit one 24-bit MMB segment, which limits a
    physical uint16 reduction block to 256 values.  Larger logical reductions
    are therefore split into 256-value chunks and accumulated modulo 2^16.
    ``tile_count`` remains the output-tile count for compatibility; callers
    use ``task_count`` for the complete output-by-reduction launch grid.
    """

    output_extent: int
    reduction_extent: int
    padded_reduction_extent: int = field(init=False)
    tile_capacity: int = field(init=False)
    tile_count: int = field(init=False)
    reduction_tile_extent: int = field(init=False)
    reduction_tile_count: int = field(init=False)
    task_count: int = field(init=False)

    def __post_init__(self):
        output_extent = _positive_extent("output_extent", self.output_extent)
        reduction_extent = _positive_extent("reduction_extent", self.reduction_extent)
        reduction_tile_extent = min(reduction_extent, 256)
        padded_reduction = _next_power_of_two(reduction_tile_extent)
        capacity = APUG2_MMB_SETS * APUG2_VECTOR_LANES // padded_reduction
        tile_count = (output_extent + capacity - 1) // capacity
        reduction_tile_count = (
            reduction_extent + reduction_tile_extent - 1
        ) // reduction_tile_extent
        object.__setattr__(self, "output_extent", output_extent)
        object.__setattr__(self, "reduction_extent", reduction_extent)
        object.__setattr__(self, "padded_reduction_extent", padded_reduction)
        object.__setattr__(self, "tile_capacity", capacity)
        object.__setattr__(self, "tile_count", tile_count)
        object.__setattr__(self, "reduction_tile_extent", reduction_tile_extent)
        object.__setattr__(self, "reduction_tile_count", reduction_tile_count)
        object.__setattr__(self, "task_count", tile_count * reduction_tile_count)

    def tile_bounds(self, tile: int) -> tuple[int, int]:
        if isinstance(tile, (bool, np.bool_)) or not isinstance(tile, Integral):
            raise TypeError("tile must be an integer")
        tile = int(tile)
        if not 0 <= tile < self.tile_count:
            raise IndexError(f"tile {tile} is outside extent {self.tile_count}")
        begin = tile * self.tile_capacity
        return begin, min(begin + self.tile_capacity, self.output_extent)

    def reduction_bounds(self, tile: int) -> tuple[int, int]:
        if isinstance(tile, (bool, np.bool_)) or not isinstance(tile, Integral):
            raise TypeError("reduction tile must be an integer")
        tile = int(tile)
        if not 0 <= tile < self.reduction_tile_count:
            raise IndexError(
                f"reduction tile {tile} is outside extent "
                f"{self.reduction_tile_count}"
            )
        begin = tile * self.reduction_tile_extent
        return begin, min(begin + self.reduction_tile_extent, self.reduction_extent)

    def tile_plan(self, tile: int, reduction_tile: int = 0) -> APUG2DotTilePlan:
        begin, end = self.tile_bounds(tile)
        reduction_begin, reduction_end = self.reduction_bounds(reduction_tile)
        return APUG2DotTilePlan(end - begin, reduction_end - reduction_begin)


def build_apu_g2_u16_layout() -> LinearLayout:
    """Map ``(group, lane_in_group, vector)`` to column and MMB set.

    L1 groups are consecutive 4K portions of the 64K column dimension.  The
    four-vector pack is a data axis mapped to the four MMB sets, not another
    Tenon work-grid axis.  A single VL64 instruction therefore realizes every
    point in this layout.
    """

    return LinearLayout(
        bases={
            "group": [((APUG2_LANES_PER_GROUP << bit), 0) for bit in range(4)],
            "lane_in_group": [((1 << bit), 0) for bit in range(12)],
            "vector": [((0, 1 << bit)) for bit in range(2)],
        },
        out_dims=("l1_column", "mmb_set"),
        out_sizes=(APUG2_VECTOR_LANES, APUG2_MMB_SETS),
    )


__all__ = [
    "APUG2_GROUPS",
    "APUG2_LANES_PER_GROUP",
    "APUG2_MMB_SETS",
    "APUG2DotTilePlan",
    "APUG2DotTiling",
    "APUG2ReductionPlan",
    "APUG2_U16_SHAPE",
    "APUG2_VECTOR_LANES",
    "build_apu_g2_u16_layout",
]
