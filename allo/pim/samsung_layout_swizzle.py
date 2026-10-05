# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A minimal, executable Samsung bank-layout swizzle experiment.

The experiment separates an access schedule from its physical data layout.
One logical round reads every ``tile`` at a fixed logical ``bank`` and
``column``.  The no-swizzle layout places those reads in different rows of
one physical bank, so they serialize.  The synthesized layout is the F2 map

.. code-block:: text

   physical_bank = logical_bank XOR tile
   physical_row  = tile
   physical_col  = column

where only the declared tile bits participate in the XOR.  Keeping ``tile``
as the row coordinate makes the map bijective: the host can pack the same
logical tensor into either layout and recover it exactly.  The overlapping
bank/tile basis vectors are intentionally evaluated by
:class:`allo.spmw_linear_layout.LinearLayout`; this is not the affine
``stride * bank + tile`` address form used by the current Samsung operand
classifier.

The experiment can obtain its bank projection from the registered production
Samsung autoscheduler.  The row/column lift, exact buffer packing, functional
execution, and grouped trace emission remain an explicit software harness;
they are not the production Samsung GEMV packer.  The traces are small enough
to feed to a simulator or an on-board microbenchmark.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ..spmw_linear_layout import LinearLayout


LOGICAL_DIMS = ("bank", "tile", "column")
PHYSICAL_DIMS = ("bank", "row", "column")


def _power_of_two_extent(name: str, value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value <= 0 or value & (value - 1):
        raise ValueError(f"{name} must be a positive power of two")
    return value


def _basis_bits(extent: int) -> range:
    return range(extent.bit_length() - 1)


@dataclass(frozen=True)
class SamsungBankGeometry:
    """Power-of-two geometry for the minimal bank/row/column experiment.

    ``tile_count`` is both the number of concurrent accesses in a round and
    the physical row extent.  It may not exceed ``bank_count`` because an
    injective low-bit XOR needs one bank bit for every tile bit.
    """

    bank_count: int = 2
    tile_count: int = 2
    column_count: int = 1

    def __post_init__(self) -> None:
        bank_count = _power_of_two_extent("bank_count", self.bank_count)
        tile_count = _power_of_two_extent("tile_count", self.tile_count)
        column_count = _power_of_two_extent("column_count", self.column_count)
        if tile_count > bank_count:
            raise ValueError(
                "tile_count must not exceed bank_count for a conflict-free XOR"
            )
        object.__setattr__(self, "bank_count", bank_count)
        object.__setattr__(self, "tile_count", tile_count)
        object.__setattr__(self, "column_count", column_count)

    @property
    def logical_shape(self) -> tuple[int, int, int]:
        return (self.bank_count, self.tile_count, self.column_count)

    @property
    def physical_shape(self) -> tuple[int, int, int]:
        return (self.bank_count, self.tile_count, self.column_count)

    def manifest(self) -> dict[str, object]:
        return {
            "logical_dims": [
                {"name": "bank", "size": self.bank_count},
                {"name": "tile", "size": self.tile_count},
                {"name": "column", "size": self.column_count},
            ],
            "physical_dims": [
                {"name": "bank", "size": self.bank_count},
                {"name": "row", "size": self.tile_count},
                {"name": "column", "size": self.column_count},
            ],
            "round_policy": "fixed-bank-and-column/all-tiles-concurrent",
        }


@dataclass(frozen=True)
class SamsungAutoscheduleSelection:
    """The production Samsung enumerator's choice, lifted for this probe.

    ``source_layout`` is attached to the placement selected by the registered
    Samsung autoscheduler.  ``layout`` retains that exact bank projection and
    adds identity row/column coordinates so the microexperiment can pack a
    bijective physical buffer and emit complete bank/row/column addresses.
    """

    layout: LinearLayout
    source_layout: LinearLayout
    placement_mode: str

    def manifest(self) -> dict[str, object]:
        return {
            "selection_path": "registered-samsung-autoscheduler",
            "placement_mode": self.placement_mode,
            "source_layout": self.source_layout.manifest(),
            "lifted_experiment_layout": self.layout.manifest(),
            "lift_contract": (
                "preserve the selected bank projection; retain tile as row "
                "and column as column for bijective experimental packing"
            ),
        }


def build_identity_layout(geometry: SamsungBankGeometry) -> LinearLayout:
    """Build the no-swizzle ``(bank, tile, col) -> (bank, row, col)`` map."""

    return LinearLayout(
        bases={
            "bank": [(1 << bit, 0, 0) for bit in _basis_bits(geometry.bank_count)],
            "tile": [(0, 1 << bit, 0) for bit in _basis_bits(geometry.tile_count)],
            "column": [(0, 0, 1 << bit) for bit in _basis_bits(geometry.column_count)],
        },
        out_dims=PHYSICAL_DIMS,
        out_sizes=geometry.physical_shape,
    )


def synthesize_xor_layout(geometry: SamsungBankGeometry) -> LinearLayout:
    """Synthesize a conflict-free low-bit bank XOR from ``geometry``.

    The result is discovered by :meth:`LinearLayout.optimal_swizzle`; no
    synthesized basis vector is pasted into this constructor.  The base map
    already retains ``tile`` in ``row``, and the optimizer augments each tile
    basis vector with a distinct bank bit until the tile span is injective on
    the physical-bank output.
    """

    base = build_identity_layout(geometry)
    layout = LinearLayout.optimal_swizzle(
        base,
        vec_dims=("column",),
        bank_dims=("bank",),
        segment_dims=("tile",),
    )
    if not layout.describes_conflict_free(
        bank_dims=("bank",), varying_inputs=("tile",)
    ):
        raise RuntimeError("LinearLayout synthesized a bank-conflicting tile map")
    return layout


def build_manual_xor_oracle_layout(geometry: SamsungBankGeometry) -> LinearLayout:
    """Construct the expected XOR map independently for equivalence tests.

    This is deliberately a simple mathematical oracle, not the production
    synthesis path.  Tile bit ``i`` contributes the same ``2**i`` mask to
    both the bank and row outputs, while logical-bank bit ``i`` contributes
    that mask to bank.  F2 evaluation therefore cancels equal low bits.
    """

    return LinearLayout(
        bases={
            "bank": [(1 << bit, 0, 0) for bit in _basis_bits(geometry.bank_count)],
            "tile": [
                (1 << bit, 1 << bit, 0) for bit in _basis_bits(geometry.tile_count)
            ],
            "column": [(0, 0, 1 << bit) for bit in _basis_bits(geometry.column_count)],
        },
        out_dims=PHYSICAL_DIMS,
        out_sizes=geometry.physical_shape,
    )


def select_autoscheduled_xor_layout(
    geometry: SamsungBankGeometry,
) -> SamsungAutoscheduleSelection:
    """Run the registered Samsung autoscheduler and lift its chosen layout.

    The production Samsung target currently declares two bank fibers per PIM,
    so this bridge intentionally accepts the same one-bit ``tile`` extent.  It
    does not claim that the production GEMV packer consumes the added row and
    column coordinates; those are the explicit microexperiment boundary.
    """

    if geometry.tile_count != 2:
        raise ValueError(
            "the production Samsung autoscheduler exposes exactly two tile fibers"
        )

    # Local imports avoid introducing an autoscheduler dependency for callers
    # that use only the standalone LinearLayout packing helpers.
    from .costs import samsung_cost
    from .targets import build_samsung_target
    from ..spmw_autoschedule import autoschedule
    from ..spmw_match import MatchTrace, MatchedOp, OperandBinding

    target = build_samsung_target()
    match = MatchedOp(
        target_op_name="MAC",
        func_name="linear_layout_swizzle_probe",
        work_id=(0, 0),
        enclosing_loops=[("k", "0", str(geometry.tile_count), 1)],
        operands=[
            OperandBinding("x", "x"),
            OperandBinding("y", "W"),
            OperandBinding("acc", "acc", is_loop_carried=True),
        ],
        result_memref_name="acc",
        op_range=("layout_probe_begin", "layout_probe_end"),
    )
    trace = MatchTrace(target.name, "linear_layout_swizzle_probe", [match])
    placement = autoschedule(target, trace, samsung_cost)[0]
    source = placement.layout
    if not isinstance(source, LinearLayout):
        raise RuntimeError(
            "Samsung autoscheduler selected a placement without LinearLayout"
        )
    for input_dim in ("bank", "tile"):
        if input_dim not in source.bases:
            raise RuntimeError(
                f"Samsung autoscheduler layout has no {input_dim!r} input dimension"
            )
    if "bank" not in source.out_dims:
        raise RuntimeError("Samsung autoscheduler layout has no bank output dimension")

    source_bank = source.out_dims.index("bank")
    bank_bits = geometry.bank_count.bit_length() - 1
    tile_bits = geometry.tile_count.bit_length() - 1
    if len(source.bases["bank"]) < bank_bits:
        raise ValueError(
            "experiment bank extent exceeds the Samsung layout bank extent"
        )
    if len(source.bases["tile"]) != tile_bits:
        raise RuntimeError(
            "Samsung autoscheduler tile basis does not match its target geometry"
        )

    lifted = LinearLayout(
        bases={
            "bank": [
                (source.bases["bank"][bit][source_bank], 0, 0)
                for bit in _basis_bits(geometry.bank_count)
            ],
            "tile": [
                (
                    source.bases["tile"][bit][source_bank],
                    1 << bit,
                    0,
                )
                for bit in _basis_bits(geometry.tile_count)
            ],
            "column": [(0, 0, 1 << bit) for bit in _basis_bits(geometry.column_count)],
        },
        out_dims=PHYSICAL_DIMS,
        out_sizes=geometry.physical_shape,
    )
    if not lifted.describes_conflict_free(
        bank_dims=("bank",), varying_inputs=("tile",)
    ):
        raise RuntimeError(
            "selected Samsung layout is not conflict-free on tile fibers"
        )
    return SamsungAutoscheduleSelection(
        layout=lifted,
        source_layout=source,
        placement_mode=placement.mode,
    )


def _validate_layout(layout: LinearLayout, geometry: SamsungBankGeometry) -> None:
    if not isinstance(layout, LinearLayout):
        raise TypeError("layout must be a LinearLayout")
    if tuple(layout.bases) != LOGICAL_DIMS:
        raise ValueError(
            f"layout input dims must be {LOGICAL_DIMS}, got {tuple(layout.bases)}"
        )
    input_sizes = tuple(layout.size_of(name) for name in LOGICAL_DIMS)
    if input_sizes != geometry.logical_shape:
        raise ValueError(
            "layout input sizes do not match geometry: "
            f"{input_sizes} != {geometry.logical_shape}"
        )
    if layout.out_dims != PHYSICAL_DIMS:
        raise ValueError(
            f"layout output dims must be {PHYSICAL_DIMS}, got {layout.out_dims}"
        )
    if layout.out_sizes != geometry.physical_shape:
        raise ValueError(
            "layout output sizes do not match geometry: "
            f"{layout.out_sizes} != {geometry.physical_shape}"
        )


def _logical_coordinates(geometry: SamsungBankGeometry):
    for bank in range(geometry.bank_count):
        for tile in range(geometry.tile_count):
            for column in range(geometry.column_count):
                yield bank, tile, column


def pack_logical_tensor(
    logical: np.ndarray,
    layout: LinearLayout,
    geometry: SamsungBankGeometry,
) -> np.ndarray:
    """Pack ``logical[bank, tile, column]`` by an exact F2 layout."""

    _validate_layout(layout, geometry)
    logical = np.asarray(logical)
    if logical.shape != geometry.logical_shape:
        raise ValueError(
            f"logical tensor must have shape {geometry.logical_shape}, "
            f"got {logical.shape}"
        )
    if not np.issubdtype(logical.dtype, np.number):
        raise TypeError(
            f"logical tensor must have a numeric dtype, got {logical.dtype}"
        )

    packed = np.empty(geometry.physical_shape, dtype=logical.dtype)
    occupied: set[tuple[int, int, int]] = set()
    for bank, tile, column in _logical_coordinates(geometry):
        physical = layout.apply(bank=bank, tile=tile, column=column)
        if physical in occupied:
            raise ValueError(
                f"layout is not one-to-one: physical coordinate {physical} repeats"
            )
        occupied.add(physical)
        packed[physical] = logical[bank, tile, column]
    if len(occupied) != packed.size:
        raise ValueError("layout does not cover the declared physical tensor")
    return packed


def unpack_physical_tensor(
    packed: np.ndarray,
    layout: LinearLayout,
    geometry: SamsungBankGeometry,
) -> np.ndarray:
    """Recover the logical tensor from a buffer packed by ``layout``."""

    _validate_layout(layout, geometry)
    packed = np.asarray(packed)
    if packed.shape != geometry.physical_shape:
        raise ValueError(
            f"packed tensor must have shape {geometry.physical_shape}, "
            f"got {packed.shape}"
        )
    if not np.issubdtype(packed.dtype, np.number):
        raise TypeError(f"packed tensor must have a numeric dtype, got {packed.dtype}")

    logical = np.empty(geometry.logical_shape, dtype=packed.dtype)
    for bank, tile, column in _logical_coordinates(geometry):
        physical = layout.apply(bank=bank, tile=tile, column=column)
        logical[bank, tile, column] = packed[physical]
    return logical


@dataclass(frozen=True, order=True)
class BankRowColumnAccess:
    """One logical access and the exact physical address chosen for it."""

    logical_bank: int
    tile: int
    logical_column: int
    bank: int
    row: int
    column: int

    @property
    def physical(self) -> tuple[int, int, int]:
        return (self.bank, self.row, self.column)

    def manifest(self) -> dict[str, object]:
        return {
            "logical": {
                "bank": self.logical_bank,
                "tile": self.tile,
                "column": self.logical_column,
            },
            "physical": {
                "bank": self.bank,
                "row": self.row,
                "column": self.column,
            },
        }


@dataclass(frozen=True)
class BankAccessRound:
    """Concurrent accesses issued at one fixed logical bank and column."""

    index: int
    logical_bank: int
    logical_column: int
    accesses: tuple[BankRowColumnAccess, ...]

    @property
    def bank_occupancy(self) -> tuple[tuple[int, int], ...]:
        counts = Counter(access.bank for access in self.accesses)
        return tuple(sorted(counts.items()))

    @property
    def conflict_count(self) -> int:
        """Extra accesses that must serialize after one access per bank."""

        return sum(count - 1 for _, count in self.bank_occupancy)

    @property
    def wavefront_count(self) -> int:
        """Minimum one-access-per-bank wavefronts needed for this round."""

        return max((count for _, count in self.bank_occupancy), default=0)

    @property
    def conflict_free(self) -> bool:
        return self.conflict_count == 0

    def manifest(self) -> dict[str, object]:
        return {
            "round": self.index,
            "logical_bank": self.logical_bank,
            "logical_column": self.logical_column,
            "accesses": [access.manifest() for access in self.accesses],
            "bank_occupancy": [
                {"bank": bank, "accesses": count} for bank, count in self.bank_occupancy
            ],
            "conflict_count": self.conflict_count,
            "wavefront_count": self.wavefront_count,
        }


def grouped_bank_row_column_trace(
    layout: LinearLayout,
    geometry: SamsungBankGeometry,
) -> tuple[BankAccessRound, ...]:
    """Emit all rounds, grouped into simultaneous bank/row/column accesses."""

    _validate_layout(layout, geometry)
    rounds = []
    round_index = 0
    for logical_bank in range(geometry.bank_count):
        for logical_column in range(geometry.column_count):
            accesses = []
            for tile in range(geometry.tile_count):
                bank, row, column = layout.apply(
                    bank=logical_bank,
                    tile=tile,
                    column=logical_column,
                )
                accesses.append(
                    BankRowColumnAccess(
                        logical_bank=logical_bank,
                        tile=tile,
                        logical_column=logical_column,
                        bank=bank,
                        row=row,
                        column=column,
                    )
                )
            rounds.append(
                BankAccessRound(
                    index=round_index,
                    logical_bank=logical_bank,
                    logical_column=logical_column,
                    accesses=tuple(accesses),
                )
            )
            round_index += 1
    return tuple(rounds)


def format_simulator_trace(
    layout: LinearLayout,
    geometry: SamsungBankGeometry,
) -> str:
    """Format physical accesses as ``round bank row column`` lines.

    The text has no header or comments so a simulator harness can consume it
    directly.  A final newline is included whenever the trace is non-empty.
    """

    lines = [
        f"{round_.index} {access.bank} {access.row} {access.column}"
        for round_ in grouped_bank_row_column_trace(layout, geometry)
        for access in round_.accesses
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def identity_and_swizzle_trace_texts(
    geometry: SamsungBankGeometry = SamsungBankGeometry(),
) -> dict[str, str]:
    """Return directly comparable identity and synthesized-XOR trace texts."""

    return {
        "identity": format_simulator_trace(build_identity_layout(geometry), geometry),
        "xor-swizzle": format_simulator_trace(
            synthesize_xor_layout(geometry), geometry
        ),
    }


def trace_manifest(
    layout: LinearLayout,
    geometry: SamsungBankGeometry,
    *,
    name: str,
) -> dict[str, object]:
    """Return a JSON-compatible exact layout and grouped-address manifest."""

    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string")
    rounds = grouped_bank_row_column_trace(layout, geometry)
    return {
        "kind": "samsung-bank-layout-trace",
        "name": name,
        "geometry": geometry.manifest(),
        "linear_layout": layout.manifest(),
        "rounds": [round_.manifest() for round_ in rounds],
        "summary": {
            "round_count": len(rounds),
            "access_count": sum(len(round_.accesses) for round_ in rounds),
            "conflict_count": sum(round_.conflict_count for round_ in rounds),
            "wavefront_count": sum(round_.wavefront_count for round_ in rounds),
            "max_wavefronts_per_round": max(
                (round_.wavefront_count for round_ in rounds), default=0
            ),
        },
    }


def deterministic_workload(geometry: SamsungBankGeometry) -> np.ndarray:
    """Return a minimal integer tensor with a unique value at every index."""

    count = int(np.prod(geometry.logical_shape))
    return np.arange(1, count + 1, dtype=np.int64).reshape(geometry.logical_shape)


def reference_tile_reduction(
    logical: np.ndarray, geometry: SamsungBankGeometry
) -> np.ndarray:
    """Reference ``sum_tile(logical * (tile + 1))`` result."""

    logical = np.asarray(logical)
    if logical.shape != geometry.logical_shape:
        raise ValueError(
            f"logical tensor must have shape {geometry.logical_shape}, "
            f"got {logical.shape}"
        )
    if not np.issubdtype(logical.dtype, np.number):
        raise TypeError(
            f"logical tensor must have a numeric dtype, got {logical.dtype}"
        )
    weights = np.arange(1, geometry.tile_count + 1, dtype=np.int64)
    return np.sum(logical.astype(np.int64) * weights[None, :, None], axis=1)


def execute_packed_tile_reduction(
    packed: np.ndarray,
    layout: LinearLayout,
    geometry: SamsungBankGeometry,
) -> np.ndarray:
    """Execute the deterministic reduction through grouped physical traces."""

    _validate_layout(layout, geometry)
    packed = np.asarray(packed)
    if packed.shape != geometry.physical_shape:
        raise ValueError(
            f"packed tensor must have shape {geometry.physical_shape}, "
            f"got {packed.shape}"
        )
    if not np.issubdtype(packed.dtype, np.number):
        raise TypeError(f"packed tensor must have a numeric dtype, got {packed.dtype}")

    output = np.zeros((geometry.bank_count, geometry.column_count), dtype=np.int64)
    for round_ in grouped_bank_row_column_trace(layout, geometry):
        for access in round_.accesses:
            output[round_.logical_bank, round_.logical_column] += int(
                packed[access.physical]
            ) * (access.tile + 1)
    return output


@dataclass(frozen=True)
class SamsungLayoutRun:
    """One fully reproducible software run for an identity or XOR layout."""

    name: str
    geometry: SamsungBankGeometry
    layout: LinearLayout
    logical_input: np.ndarray
    packed_input: np.ndarray
    unpacked_input: np.ndarray
    output: np.ndarray
    rounds: tuple[BankAccessRound, ...]

    def manifest(self) -> dict[str, object]:
        manifest = trace_manifest(self.layout, self.geometry, name=self.name)
        manifest["numeric_workload"] = {
            "operation": "sum_tile(input[bank,tile,column] * (tile + 1))",
            "logical_input": self.logical_input.tolist(),
            "packed_input": self.packed_input.tolist(),
            "unpacked_input": self.unpacked_input.tolist(),
            "output": self.output.tolist(),
        }
        return manifest


def run_layout_microexperiment(
    name: str,
    layout: LinearLayout,
    geometry: SamsungBankGeometry,
    *,
    logical_input: np.ndarray | None = None,
) -> SamsungLayoutRun:
    """Pack, unpack, trace, and execute one layout deterministically."""

    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string")
    if logical_input is None:
        logical_input = deterministic_workload(geometry)
    logical_input = np.asarray(logical_input)
    packed = pack_logical_tensor(logical_input, layout, geometry)
    unpacked = unpack_physical_tensor(packed, layout, geometry)
    output = execute_packed_tile_reduction(packed, layout, geometry)
    return SamsungLayoutRun(
        name=name,
        geometry=geometry,
        layout=layout,
        logical_input=logical_input.copy(),
        packed_input=packed,
        unpacked_input=unpacked,
        output=output,
        rounds=grouped_bank_row_column_trace(layout, geometry),
    )


def run_identity_and_swizzle(
    geometry: SamsungBankGeometry = SamsungBankGeometry(),
) -> tuple[SamsungLayoutRun, SamsungLayoutRun]:
    """Run the same workload under no-swizzle and synthesized XOR layouts."""

    logical = deterministic_workload(geometry)
    identity = run_layout_microexperiment(
        "identity", build_identity_layout(geometry), geometry, logical_input=logical
    )
    swizzled = run_layout_microexperiment(
        "xor-swizzle", synthesize_xor_layout(geometry), geometry, logical_input=logical
    )
    return identity, swizzled


__all__ = [
    "LOGICAL_DIMS",
    "PHYSICAL_DIMS",
    "SamsungBankGeometry",
    "SamsungAutoscheduleSelection",
    "BankRowColumnAccess",
    "BankAccessRound",
    "SamsungLayoutRun",
    "build_identity_layout",
    "synthesize_xor_layout",
    "build_manual_xor_oracle_layout",
    "select_autoscheduled_xor_layout",
    "pack_logical_tensor",
    "unpack_physical_tensor",
    "grouped_bank_row_column_trace",
    "format_simulator_trace",
    "identity_and_swizzle_trace_texts",
    "trace_manifest",
    "deterministic_workload",
    "reference_tile_reduction",
    "execute_packed_tile_reduction",
    "run_layout_microexperiment",
    "run_identity_and_swizzle",
]
