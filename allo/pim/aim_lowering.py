# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AiM lowering IR and the shared lowerer.

The AimOp records are the AiM lowering IR: ``AimCtx.emit_program`` derives them
from matched kernels and ``_AimLowerer`` expands each record into simulator
commands.  The helpers below are shared by the enumerator, the cost program and
codegen.

The AiM simulator is timing-only, so the records describe the shape and
placement of data, not its values.  ``input_source="banks"`` records the
intended source but the timing simulator does not check payload routing, and
``partitions_per_replica`` is a layout assertion, not a numerical coverage
proof.  Benchmark manifests must retain these assumptions and must not claim
simulator-backed numerical correctness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Literal


def _positive(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative(name: str, value: int) -> int:
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _normalize_channels(channels) -> tuple[int, ...] | None:
    if channels is None:
        return None
    values = tuple(int(channel) for channel in channels)
    if not values:
        raise ValueError("channels must not be empty")
    if any(channel < 0 for channel in values):
        raise ValueError("channel indices must be nonnegative")
    if len(values) != len(set(values)):
        raise ValueError("channels must be unique")
    return values


class AimOp:
    """Marker base class for typed AiM program operations."""

    @property
    def kind(self) -> str:
        name = type(self).__name__
        return name.removeprefix("Aim").lower()

    def manifest(self) -> dict:
        data = asdict(self)
        data["kind"] = self.kind
        return data


@dataclass(frozen=True)
# The fields are the public layout contract, not mutable implementation state.
# pylint: disable-next=too-many-instance-attributes
class AimContraction(AimOp):
    """Dense reductions mapped over channel-local, all-bank MAC engines.

    ``outputs`` is the number of scalar results per independent instance and
    ``reduction`` is K.  ``batches`` and ``replicas`` describe explicit
    logical work and never become a simulator-side cycle multiplier.  The
    default ``"flattened"`` mapping flattens them into one spatial output
    frontier and lays out matrix rows launch-major then K-row-major from
    ``row``; the specialized mappings below publish their different row
    ownership formulas in the compiled manifest.

    ``batch_mapping`` selects how independent batches occupy the physical
    matrix/GB layout.  ``"auto"`` applies a shape- and residency-only policy
    and records its decision in the compiled manifest.  ``"flattened"``
    preserves the ordinary dense-output frontier.  ``"row_packed"`` packs
    complete reduction vectors into a bank row and is useful when several
    batches share the same output frontier.  ``"channels"`` maps one batch to
    each channel and distributes its outputs over that channel's banks.  The
    latter may reserve a larger
    ``reduction_storage_extent`` than the logical reduction, so a short live
    reduction can address a cache laid out for a longer maximum extent.

    A reuse group keeps up to ``reuse_group_size`` accumulator launches live
    together, ordering all WR_BIAS commands before all MAC commands and all
    readbacks.  When omitted, lowering picks the largest valid generic
    candidate no greater than the target accumulator depth.  Groups are made
    as even as possible under that limit to avoid a needlessly tiny tail.
    CENT's fused MAC/AF schedule budgets half of the configured reuse window
    (equivalently, two window entries per live output).
    """

    outputs: int
    reduction: int
    batches: int = 1
    replicas: int = 1
    row: int = 0
    channels: tuple[int, ...] | None = None
    channels_per_replica: int | None = None
    input_source: Literal["gb", "banks"] = "gb"
    batch_mapping: Literal["auto", "flattened", "row_packed", "channels"] = "flattened"
    reduction_storage_extent: int | None = None
    input_gpr: int = 0
    bias_gpr: int = 0
    result_gpr: int = 0
    reuse_group_size: int | None = None
    reuse_group_candidates: tuple[int, ...] = ()
    activation: bool = False
    name: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "outputs", _positive("outputs", self.outputs))
        object.__setattr__(self, "reduction", _positive("reduction", self.reduction))
        object.__setattr__(self, "batches", _positive("batches", self.batches))
        object.__setattr__(self, "replicas", _positive("replicas", self.replicas))
        object.__setattr__(self, "row", _nonnegative("row", self.row))
        object.__setattr__(self, "input_gpr", _nonnegative("input_gpr", self.input_gpr))
        object.__setattr__(self, "bias_gpr", _nonnegative("bias_gpr", self.bias_gpr))
        object.__setattr__(
            self, "result_gpr", _nonnegative("result_gpr", self.result_gpr)
        )
        object.__setattr__(self, "channels", _normalize_channels(self.channels))
        if self.channels_per_replica is not None:
            object.__setattr__(
                self,
                "channels_per_replica",
                _positive("channels_per_replica", self.channels_per_replica),
            )
        input_source = str(self.input_source).lower()
        if input_source not in {"gb", "banks"}:
            raise ValueError("AimContraction.input_source must be 'gb' or 'banks'")
        object.__setattr__(self, "input_source", input_source)
        batch_mapping = str(self.batch_mapping).lower()
        if batch_mapping not in {
            "auto",
            "flattened",
            "row_packed",
            "channels",
        }:
            raise ValueError(
                "AimContraction.batch_mapping must be 'auto', 'flattened', "
                "'row_packed', or 'channels'"
            )
        object.__setattr__(self, "batch_mapping", batch_mapping)
        if self.reduction_storage_extent is not None:
            storage_extent = _positive(
                "reduction_storage_extent", self.reduction_storage_extent
            )
            if storage_extent < self.reduction:
                raise ValueError(
                    "reduction_storage_extent cannot be smaller than reduction"
                )
            object.__setattr__(self, "reduction_storage_extent", storage_extent)
        if self.reuse_group_size is not None:
            object.__setattr__(
                self,
                "reuse_group_size",
                _positive("reuse_group_size", self.reuse_group_size),
            )
        candidates = tuple(
            _positive("reuse_group_candidates entry", candidate)
            for candidate in self.reuse_group_candidates
        )
        if len(candidates) != len(set(candidates)):
            raise ValueError("reuse_group_candidates must be unique")
        object.__setattr__(self, "reuse_group_candidates", candidates)
        if self.name is not None and not self.name:
            raise ValueError("contraction name must be non-empty")


@dataclass(frozen=True)
class AimElementwise(AimOp):
    """Elementwise MUL in bank groups or ADD in the host-visible GPR."""

    operation: Literal["mul", "add"]
    elements: int
    row: int = 0
    channels: tuple[int, ...] | None = None
    replicas: int = 1
    channels_per_replica: int | None = None
    partitions_per_replica: int | None = None
    gpr_addr_0: int = 0
    gpr_addr_1: int = 0
    name: str | None = None

    def __post_init__(self):
        operation = str(self.operation).lower()
        if operation not in {"mul", "add"}:
            raise ValueError("AimElementwise.operation must be 'mul' or 'add'")
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "elements", _positive("elements", self.elements))
        object.__setattr__(self, "row", _nonnegative("row", self.row))
        object.__setattr__(self, "channels", _normalize_channels(self.channels))
        object.__setattr__(self, "replicas", _positive("replicas", self.replicas))
        if self.channels_per_replica is not None:
            object.__setattr__(
                self,
                "channels_per_replica",
                _positive("channels_per_replica", self.channels_per_replica),
            )
        if self.partitions_per_replica is not None:
            object.__setattr__(
                self,
                "partitions_per_replica",
                _positive("partitions_per_replica", self.partitions_per_replica),
            )
        object.__setattr__(
            self, "gpr_addr_0", _nonnegative("gpr_addr_0", self.gpr_addr_0)
        )
        object.__setattr__(
            self, "gpr_addr_1", _nonnegative("gpr_addr_1", self.gpr_addr_1)
        )
        if self.name is not None and not self.name:
            raise ValueError("elementwise name must be non-empty")


@dataclass(frozen=True)
class AimActivation(AimOp):
    """Apply the configured activation function to accumulator latches."""

    groups: int = 1
    channels: tuple[int, ...] | None = None
    readback: bool = True
    result_gpr: int = 0
    name: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "groups", _positive("groups", self.groups))
        object.__setattr__(self, "channels", _normalize_channels(self.channels))
        object.__setattr__(
            self, "result_gpr", _nonnegative("result_gpr", self.result_gpr)
        )
        if self.name is not None and not self.name:
            raise ValueError("activation name must be non-empty")


@dataclass(frozen=True)
class AimHostTransfer(AimOp):
    """Raw host read or write bursts to one bank row.

    One simulator ``R/W MEM`` line represents one 256-bit burst.  ``bursts``
    therefore materializes that many explicit requests and is never encoded as
    an analytical multiplier.
    """

    direction: Literal["read", "write"]
    channel: int
    bank: int
    row: int
    bursts: int = 1
    name: str | None = None

    def __post_init__(self):
        direction = str(self.direction).lower()
        if direction not in {"read", "write"}:
            raise ValueError("AimHostTransfer.direction must be 'read' or 'write'")
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "channel", _nonnegative("channel", self.channel))
        object.__setattr__(self, "bank", _nonnegative("bank", self.bank))
        object.__setattr__(self, "row", _nonnegative("row", self.row))
        object.__setattr__(self, "bursts", _positive("bursts", self.bursts))
        if self.name is not None and not self.name:
            raise ValueError("host-transfer name must be non-empty")


@dataclass(frozen=True)
class AimBankCopy(AimOp):
    """Copy a distributed vector between one bank and each channel's GB."""

    direction: Literal["bank_to_gb", "gb_to_bank"]
    elements: int
    bank: int
    row: int
    channels: tuple[int, ...] | None = None
    copies: int = 1
    bank_stride: int = 1
    replicas: int = 1
    channels_per_replica: int | None = None
    partitions_per_replica: int | None = None
    name: str | None = None

    def __post_init__(self):
        direction = str(self.direction).lower()
        if direction not in {"bank_to_gb", "gb_to_bank"}:
            raise ValueError(
                "AimBankCopy.direction must be 'bank_to_gb' or 'gb_to_bank'"
            )
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "elements", _positive("elements", self.elements))
        object.__setattr__(self, "bank", _nonnegative("bank", self.bank))
        object.__setattr__(self, "row", _nonnegative("row", self.row))
        object.__setattr__(self, "channels", _normalize_channels(self.channels))
        object.__setattr__(self, "copies", _positive("copies", self.copies))
        object.__setattr__(
            self, "bank_stride", _positive("bank_stride", self.bank_stride)
        )
        object.__setattr__(self, "replicas", _positive("replicas", self.replicas))
        if self.channels_per_replica is not None:
            object.__setattr__(
                self,
                "channels_per_replica",
                _positive("channels_per_replica", self.channels_per_replica),
            )
        if self.partitions_per_replica is not None:
            object.__setattr__(
                self,
                "partitions_per_replica",
                _positive("partitions_per_replica", self.partitions_per_replica),
            )
        if self.name is not None and not self.name:
            raise ValueError("bank-copy name must be non-empty")


@dataclass(frozen=True)
class AimAllBankWrite(AimOp):
    """Write one GPR vector to every bank of each selected channel.

    WR_ABK accepts exactly one channel per command.  ``rows`` and ``copies``
    compactly describe regular cache-append layouts while lowering still emits
    every physical command explicitly.
    """

    row: int
    rows: int = 1
    row_stride: int = 1
    copies: int = 1
    copy_row_stride: int | None = None
    channels: tuple[int, ...] | None = None
    gpr_addr: int = 0
    name: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "row", _nonnegative("row", self.row))
        object.__setattr__(self, "rows", _positive("rows", self.rows))
        object.__setattr__(self, "row_stride", _positive("row_stride", self.row_stride))
        object.__setattr__(self, "copies", _positive("copies", self.copies))
        if self.copy_row_stride is not None:
            object.__setattr__(
                self,
                "copy_row_stride",
                _positive("copy_row_stride", self.copy_row_stride),
            )
        object.__setattr__(self, "channels", _normalize_channels(self.channels))
        object.__setattr__(self, "gpr_addr", _nonnegative("gpr_addr", self.gpr_addr))
        if self.name is not None and not self.name:
            raise ValueError("all-bank write name must be non-empty")


@dataclass(frozen=True)
class AimDistributedHostTransfer(AimOp):
    """Regular linear-layout host traffic expanded into raw MEM bursts.

    Each replica occupies ``channels_per_replica`` disjoint channels.  Banks
    at ``bank_offset + i * bank_stride`` are the partition axis, and adjacent
    offsets can be replicated with ``copies``.  The lowering order is row,
    burst, partition, replica, copy, making large traces reproducible without
    retaining hundreds of thousands of Python objects.
    """

    direction: Literal["read", "write"]
    elements: int
    replicas: int
    channels_per_replica: int
    row: int
    bank_stride: int = 1
    bank_offset: int = 0
    copies: int = 1
    name: str | None = None

    def __post_init__(self):
        direction = str(self.direction).lower()
        if direction not in {"read", "write"}:
            raise ValueError(
                "AimDistributedHostTransfer.direction must be 'read' or 'write'"
            )
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "elements", _positive("elements", self.elements))
        object.__setattr__(self, "replicas", _positive("replicas", self.replicas))
        object.__setattr__(
            self,
            "channels_per_replica",
            _positive("channels_per_replica", self.channels_per_replica),
        )
        object.__setattr__(self, "row", _nonnegative("row", self.row))
        object.__setattr__(
            self, "bank_stride", _positive("bank_stride", self.bank_stride)
        )
        object.__setattr__(
            self, "bank_offset", _nonnegative("bank_offset", self.bank_offset)
        )
        object.__setattr__(self, "copies", _positive("copies", self.copies))
        if self.name is not None and not self.name:
            raise ValueError("distributed host-transfer name must be non-empty")


@dataclass(frozen=True)
class AimSync(AimOp):
    """Device-wide synchronization point."""

    name: str | None = None

    def __post_init__(self):
        if self.name is not None and not self.name:
            raise ValueError("sync name must be non-empty")


@dataclass(frozen=True)
class _AimGeometry:
    channels: int
    banks: int
    bank_groups: int
    rows: int
    row_elements: int
    element_bits: int
    vector_bits: int
    lanes: int
    gb_columns: int
    reuse_window: int
    gpr_entries: int


def _target_geometry(target) -> _AimGeometry:
    if getattr(target, "name", None) != "aim":
        raise ValueError("AiM lowering requires the aim target")
    banks = getattr(target, "banks", None)
    gb = getattr(target, "gb", None)
    mac_reg = getattr(target, "mac_reg", None)
    gpr = getattr(target, "gpr", None)
    channel = target.unit("channel")
    try:
        channels = math.prod(int(value) for value in channel.axes.values())
        bank_count = int(banks.geometry["banks"])
        bank_groups = int(banks.geometry.get("bank_groups", 1))
        rows = int(banks.geometry["rows"])
        row_elements = int(banks.geometry["cols"])
        element_bits = int(banks.geometry["width"])
        gb_columns = int(gb.geometry["entries"])
        reuse_window = int(mac_reg.slots)
        gpr_entries = int(gpr.geometry["entries"])
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ValueError("aim target is missing required storage geometry") from error
    vector_bits = int(getattr(gb, "width", gb.geometry.get("width", 0)))
    if channels <= 0 or bank_count <= 0 or rows <= 0 or row_elements <= 0:
        raise ValueError("aim target geometry must be positive")
    if vector_bits <= 0 or element_bits <= 0 or vector_bits % element_bits:
        raise ValueError("aim target vector width must divide into element lanes")
    lanes = vector_bits // element_bits
    if gb_columns * lanes < row_elements:
        raise ValueError("aim GB cannot hold one complete bank row")
    return _AimGeometry(
        channels,
        bank_count,
        bank_groups,
        rows,
        row_elements,
        element_bits,
        vector_bits,
        lanes,
        gb_columns,
        reuse_window,
        gpr_entries,
    )


def _channels_or_all(
    requested: tuple[int, ...] | None,
    geometry: _AimGeometry,
) -> tuple[int, ...]:
    channels = tuple(range(geometry.channels)) if requested is None else requested
    if any(channel >= geometry.channels for channel in channels):
        raise ValueError(
            f"channel index exceeds target channel count {geometry.channels}"
        )
    return channels


def _contraction_channels(
    operation: AimContraction,
    geometry: _AimGeometry,
    total_outputs: int,
) -> tuple[int, ...]:
    if operation.channels is not None:
        return _channels_or_all(operation.channels, geometry)
    required = math.ceil(total_outputs / geometry.banks)
    return tuple(range(min(geometry.channels, max(1, required))))


def _channel_mask(channels: tuple[int, ...], geometry: _AimGeometry) -> int:
    """Encode simulator masks, whose low bit denotes the highest channel."""
    mask = 0
    for channel in channels:
        mask |= 1 << (geometry.channels - 1 - channel)
    return mask


def _replica_channel_layout(
    requested: tuple[int, ...] | None,
    replicas: int,
    channels_per_replica: int | None,
    geometry: _AimGeometry,
) -> tuple[tuple[int, ...], int]:
    """Resolve disjoint channel groups used by replicated tensor layouts."""
    if requested is None:
        if channels_per_replica is None:
            if geometry.channels % replicas:
                raise ValueError(
                    "target channels are not evenly divisible by replicas; "
                    "specify channels_per_replica"
                )
            channels_per_replica = geometry.channels // replicas
        required = replicas * channels_per_replica
        if required > geometry.channels:
            raise ValueError(
                f"{replicas} replicas * {channels_per_replica} channels exceeds "
                f"target count {geometry.channels}"
            )
        return tuple(range(required)), channels_per_replica

    channels = _channels_or_all(requested, geometry)
    if channels_per_replica is None:
        if len(channels) % replicas:
            raise ValueError(
                "selected channels are not evenly divisible by replicas; "
                "specify channels_per_replica"
            )
        channels_per_replica = len(channels) // replicas
    if replicas * channels_per_replica != len(channels):
        raise ValueError(
            "selected channel count must equal replicas * channels_per_replica"
        )
    return channels, channels_per_replica


def _default_reuse_candidates(limit: int) -> tuple[int, ...]:
    candidates = []
    value = 1
    while value <= limit:
        candidates.append(value)
        value *= 2
    if candidates[-1] != limit:
        candidates.append(limit)
    return tuple(candidates)


def _select_reuse_group(
    operation: AimContraction,
    geometry: _AimGeometry,
    launches: int,
) -> tuple[int, int, int, tuple[int, ...]]:
    window_entries_per_launch = 2 if operation.activation else 1
    capacity = geometry.reuse_window // window_entries_per_launch
    if capacity <= 0:
        raise ValueError("aim target has an insufficient reuse window")
    if operation.reuse_group_candidates:
        candidates = operation.reuse_group_candidates
    else:
        candidates = _default_reuse_candidates(capacity)
    if any(candidate > capacity for candidate in candidates):
        raise ValueError(
            "reuse group candidate exceeds target reuse-window capacity " f"{capacity}"
        )
    if operation.reuse_group_size is None:
        configured = max(candidates)
    else:
        configured = operation.reuse_group_size
        if configured > capacity:
            raise ValueError(
                f"reuse_group_size {configured} exceeds target reuse-window "
                f"capacity {capacity}"
            )
        if operation.reuse_group_candidates and configured not in candidates:
            raise ValueError("reuse_group_size must be one of reuse_group_candidates")
        if configured not in candidates:
            candidates = tuple(sorted((*candidates, configured)))
    # First determine how many groups are necessary under the resource limit,
    # then spread launches across those groups.  For example, 86 launches with
    # capacity 32 become 29/29/28 instead of 32/32/22.  Both are legal, but the
    # former avoids unnecessarily extreme tails and matches the target's
    # generic GB-reuse schedule.
    group_count = math.ceil(launches / configured)
    selected = math.ceil(launches / group_count)
    return selected, configured, capacity, candidates


class _AimLowerer:
    def __init__(self, target):
        self.target = target
        self.geometry = _target_geometry(target)
        self.commands: list[str] = []
        self.operations: list[dict] = []

    def emit(self, line: str) -> None:
        if line == "AiM EOC":
            raise ValueError("operations cannot emit EOC directly")
        self.commands.append(line)

    def _check_rows(self, first: int, count: int) -> None:
        if first < 0 or count <= 0 or first + count > self.geometry.rows:
            raise ValueError(
                f"bank rows [{first}, {first + count}) exceed target row "
                f"capacity {self.geometry.rows}"
            )

    def _check_gpr(self, address: int, columns: int = 1) -> None:
        if address < 0 or columns <= 0 or address + columns > self.geometry.gpr_entries:
            raise ValueError(
                f"GPR range [{address}, {address + columns}) exceeds target "
                f"capacity {self.geometry.gpr_entries}"
            )

    def lower(self, operation: AimOp, index: int) -> None:
        begin = len(self.commands)
        if isinstance(operation, AimContraction):
            details = self._contraction(operation)
        elif isinstance(operation, AimElementwise):
            details = self._elementwise(operation)
        elif isinstance(operation, AimActivation):
            details = self._activation(operation)
        elif isinstance(operation, AimHostTransfer):
            details = self._host_transfer(operation)
        elif isinstance(operation, AimDistributedHostTransfer):
            details = self._distributed_host_transfer(operation)
        elif isinstance(operation, AimBankCopy):
            details = self._bank_copy(operation)
        elif isinstance(operation, AimAllBankWrite):
            details = self._all_bank_write(operation)
        elif isinstance(operation, AimSync):
            self.emit("AiM SYNC")
            details = {"barrier": "device"}
        else:
            raise TypeError(f"unsupported AiM operation {type(operation).__name__}")
        end = len(self.commands)
        self.operations.append(
            {
                "index": index,
                "kind": operation.kind,
                "name": getattr(operation, "name", None),
                "command_span": [begin, end],
                "command_count": end - begin,
                **details,
            }
        )

    def _contraction(self, operation: AimContraction) -> dict:
        selected_mapping = operation.batch_mapping
        selection_reason = "explicit batch mapping"
        if selected_mapping == "auto":
            aligned_reduction = (
                math.ceil(operation.reduction / self.geometry.lanes)
                * self.geometry.lanes
            )
            if operation.input_source == "banks":
                selected_mapping = "flattened"
                selection_reason = "bank-resident input needs no GB batch layout"
            elif operation.reduction_storage_extent is not None:
                selected_mapping = "channels"
                selection_reason = (
                    "reserved reduction storage requires channel-batched "
                    "output strides"
                )
            elif operation.batches == 1:
                selected_mapping = "flattened"
                selection_reason = "one GB payload needs no batch packing"
            elif aligned_reduction <= self.geometry.row_elements:
                selected_mapping = "row_packed"
                selection_reason = (
                    "independent GB batches fit complete aligned reductions "
                    "within one bank row"
                )
            else:
                raise ValueError(
                    "the 'auto' batch mapping found no dependency-safe "
                    "layout: multiple GB batches have reductions larger "
                    "than one bank row and no reduction_storage_extent"
                )
        if (
            selected_mapping != "channels"
            and operation.reduction_storage_extent is not None
        ):
            raise ValueError(
                "reduction_storage_extent is only meaningful for the "
                "'channels' batch mapping"
            )
        if selected_mapping == "row_packed":
            details = self._contraction_row_packed(operation)
        elif selected_mapping == "channels":
            details = self._contraction_channels_batched(operation)
        else:
            details = self._contraction_flattened(operation)
        if operation.batch_mapping == "auto":
            details["requested_batch_mapping"] = "auto"
            details["batch_mapping_selection"] = {
                "policy": "shape_and_residency_v1",
                "selected": selected_mapping,
                "reason": selection_reason,
                "uses_operation_name": False,
            }
        return details

    def _emit_contraction_reuse_scope(
        self,
        operation: AimContraction,
        mask: int,
        matrix_rows: tuple[int, ...],
        op_size: int,
        *,
        final_reduction_chunk: bool,
    ) -> dict:
        """Emit one dependency scope while its WR_GB payload remains live."""
        if not matrix_rows:
            raise ValueError("a contraction reuse scope cannot be empty")
        (
            selected_reuse,
            configured_reuse,
            reuse_capacity,
            candidates,
        ) = _select_reuse_group(operation, self.geometry, len(matrix_rows))
        group_sizes = []
        for group_start in range(0, len(matrix_rows), selected_reuse):
            group = matrix_rows[
                group_start : min(len(matrix_rows), group_start + selected_reuse)
            ]
            group_sizes.append(len(group))
            for _matrix_row in group:
                self.emit(f"AiM WR_BIAS {operation.bias_gpr} {mask}")
            for matrix_row in group:
                self.emit(f"AiM MAC_ABK {op_size} {mask} {matrix_row}")
            if operation.activation and final_reduction_chunk:
                # AF and its readback are paired while the corresponding
                # accumulator latch is selected.  Keeping these commands in
                # the current GB scope also prevents a later WR_GB from
                # invalidating an activation's input dependency.
                for _matrix_row in group:
                    self.emit(f"AiM AF {mask}")
                    self.emit(f"AiM RD_AF {operation.result_gpr} {mask}")
            for _matrix_row in group:
                self.emit(f"AiM RD_MAC {operation.result_gpr} {mask}")
        return {
            "launches": len(matrix_rows),
            "group_sizes": group_sizes,
            "selected_group_size": selected_reuse,
            "configured_group_size": configured_reuse,
            "capacity": reuse_capacity,
            "candidates": list(candidates),
        }

    def _specialized_contraction_channels(
        self, operation: AimContraction
    ) -> tuple[tuple[int, ...], int]:
        if operation.input_source != "gb":
            raise ValueError(
                f"the '{operation.batch_mapping}' batch mapping requires "
                "input_source='gb'"
            )
        return _replica_channel_layout(
            operation.channels,
            operation.replicas,
            operation.channels_per_replica,
            self.geometry,
        )

    def _contraction_row_packed(self, operation: AimContraction) -> dict:
        """Pack short batch reduction vectors into complete bank rows."""
        geometry = self.geometry
        channels, channels_per_replica = self._specialized_contraction_channels(
            operation
        )
        replica_channel_groups = tuple(
            channels[
                replica * channels_per_replica : (replica + 1) * channels_per_replica
            ]
            for replica in range(operation.replicas)
        )
        reduction_columns = math.ceil(operation.reduction / geometry.lanes)
        aligned_reduction = reduction_columns * geometry.lanes
        batch_pack = min(
            operation.batches,
            geometry.row_elements // aligned_reduction,
        )
        if batch_pack == 0:
            # Falling back after the caller selected this dependency layout
            # would be unsafe: the flattened batched form can stage several
            # payloads before their consumers.  Fail closed until a genuine
            # multi-row packed mapping is selected by a future scheduler.
            raise ValueError(
                "the 'row_packed' batch mapping requires an aligned "
                "reduction that fits in one bank row"
            )

        batch_groups = math.ceil(operation.batches / batch_pack)
        output_capacity = channels_per_replica * geometry.banks
        output_tiles = math.ceil(operation.outputs / output_capacity)
        output_tile_layouts = []
        for output_tile in range(output_tiles):
            output_begin = output_tile * output_capacity
            valid_outputs = min(output_capacity, operation.outputs - output_begin)
            local_channels = math.ceil(valid_outputs / geometry.banks)
            tile_channels = tuple(
                channel
                for replica_channels in replica_channel_groups
                for channel in replica_channels[:local_channels]
            )
            output_tile_layouts.append(
                {
                    "index": output_tile,
                    "output_begin": output_begin,
                    "valid_outputs_per_replica": valid_outputs,
                    "active_channels_per_replica": local_channels,
                    "active_channels": list(tile_channels),
                    "channel_mask": _channel_mask(tile_channels, geometry),
                    "physical_output_slots_per_replica": (
                        local_channels * geometry.banks
                    ),
                    "padded_output_slots_per_replica": (
                        local_channels * geometry.banks - valid_outputs
                    ),
                }
            )
        physical_rows = batch_groups * output_tiles
        self._check_rows(operation.row, physical_rows)
        self._check_gpr(operation.input_gpr, batch_pack * reduction_columns)
        self._check_gpr(operation.bias_gpr)
        self._check_gpr(operation.result_gpr)

        input_groups = []
        reuse_scopes = []
        for batch_group in range(batch_groups):
            batch_begin = batch_group * batch_pack
            packed_batches = min(batch_pack, operation.batches - batch_begin)
            wr_gb_op_size = packed_batches * reduction_columns
            output_group_rows = []
            for output_tile in range(output_tiles):
                tile = output_tile_layouts[output_tile]
                mask = tile["channel_mask"]
                matrix_row = operation.row + output_tile * batch_groups + batch_group
                # A new WR_GB begins a new dependency scope.  Reuse may only
                # combine the packed batch launches below; it can never cross
                # this write and observe a later group's payload.
                self.emit(f"AiM WR_GB {wr_gb_op_size} {operation.input_gpr} {mask}")
                scope = self._emit_contraction_reuse_scope(
                    operation,
                    mask,
                    (matrix_row,) * packed_batches,
                    reduction_columns,
                    final_reduction_chunk=True,
                )
                scope.update(
                    {
                        "batch_group": batch_group,
                        "output_tile": output_tile,
                        "matrix_row": matrix_row,
                        "active_channels": tile["active_channels"],
                        "channel_mask": mask,
                    }
                )
                reuse_scopes.append(scope)
                output_group_rows.append(matrix_row)
            input_groups.append(
                {
                    "index": batch_group,
                    "batch_begin": batch_begin,
                    "batches": packed_batches,
                    "wr_gb_op_size": wr_gb_op_size,
                    "matrix_rows": output_group_rows,
                }
            )

        nominal_reuse = _select_reuse_group(operation, geometry, batch_pack)
        logical_scalar_macs = (
            operation.outputs
            * operation.reduction
            * operation.batches
            * operation.replicas
        )
        physical_output_slots_per_batch = sum(
            len(tile["active_channels"]) * geometry.banks
            for tile in output_tile_layouts
        )
        physical_wr_gb_channel_copies = sum(
            len(tile["active_channels"]) for tile in output_tile_layouts
        )
        physical_padded_scalar_macs = (
            operation.batches * physical_output_slots_per_batch * aligned_reduction
        )
        unique_gb_payload_elements = (
            operation.batches * operation.reduction * operation.replicas
        )
        physical_gb_transfer_elements = (
            operation.batches * aligned_reduction * physical_wr_gb_channel_copies
        )
        actually_active_channels = tuple(
            dict.fromkeys(
                channel
                for tile in output_tile_layouts
                for channel in tile["active_channels"]
            )
        )
        return {
            "outputs_per_instance": operation.outputs,
            "batches": operation.batches,
            "replicas": operation.replicas,
            "total_outputs": (
                operation.outputs * operation.batches * operation.replicas
            ),
            "batch_mapping": "row_packed",
            "layout": {
                "kind": "row_packed_batches",
                "iteration_order": [
                    "batch_group",
                    "output_tile",
                    "reuse_group",
                ],
                "matrix_row_address": (
                    "row + output_tile * batch_groups + batch_group"
                ),
                "reuse_dependency_scope": "one WR_GB live range",
                "reuse_crosses_wr_gb": False,
                "matrix_storage_contract": {
                    "logical_axes": [
                        "replica",
                        "batch",
                        "output",
                        "reduction",
                    ],
                    "batch_group": "batch // batch_pack",
                    "output_tile": ("output // output_capacity_per_tile_per_replica"),
                    "channel_within_replica": (
                        "(output % output_capacity_per_tile_per_replica) "
                        "// active_banks_per_channel"
                    ),
                    "physical_channel": (
                        "replica_channel_groups[replica]" "[channel_within_replica]"
                    ),
                    "bank": "output % active_banks_per_channel",
                    "row": ("row_base + output_tile * batch_groups + batch_group"),
                    "column": ("(batch % batch_pack) * aligned_reduction + reduction"),
                    "packed_segment_padding": ("aligned_reduction - reduction"),
                    "output_tail_policy": (
                        "mask whole unused channels in every replica; "
                        "zero-pad only the final active channel's unused banks"
                    ),
                    "row_base": operation.row,
                    "row_elements": geometry.row_elements,
                    "replica_channel_groups": [
                        list(group) for group in replica_channel_groups
                    ],
                    "required_replica_coverage": list(range(operation.replicas)),
                },
            },
            "configured_channels": list(channels),
            "active_channels": list(actually_active_channels),
            "active_channel_count": len(actually_active_channels),
            "active_banks_per_channel": geometry.banks,
            "channels_per_replica": channels_per_replica,
            "replica_channel_groups": [list(group) for group in replica_channel_groups],
            "output_capacity_per_tile_per_replica": output_capacity,
            "output_tiles": output_tiles,
            "output_tile_layouts": output_tile_layouts,
            "output_utilization": (
                operation.outputs * operation.replicas / physical_output_slots_per_batch
            ),
            "physical_output_slots_per_batch": physical_output_slots_per_batch,
            "padded_output_slots_per_batch": (
                physical_output_slots_per_batch - operation.outputs * operation.replicas
            ),
            "reduction": operation.reduction,
            "aligned_reduction": aligned_reduction,
            "reduction_rows": 1,
            "reduction_storage_extent": operation.reduction,
            "reduction_storage_rows": 1,
            "batch_pack": batch_pack,
            "batch_groups": batch_groups,
            "input_groups": input_groups,
            "wr_gb_launches": batch_groups * output_tiles,
            "wr_gb_launches_per_reduction_row": (batch_groups * output_tiles),
            "allocated_matrix_rows": physical_rows,
            "accessed_matrix_rows": physical_rows,
            "matrix_row_range": [operation.row, operation.row + physical_rows],
            "reuse_group_candidates": list(nominal_reuse[3]),
            "reuse_group_size": nominal_reuse[0],
            "configured_reuse_group_size": nominal_reuse[1],
            "reuse_window_capacity": geometry.reuse_window,
            "reuse_window_entries_per_launch": (2 if operation.activation else 1),
            "effective_reuse_capacity": nominal_reuse[2],
            "reuse_scopes": reuse_scopes,
            "logical_scalar_macs": logical_scalar_macs,
            "physical_mac_slots": physical_padded_scalar_macs,
            "unique_gb_payload_elements": unique_gb_payload_elements,
            "unique_logical_gb_payload_elements": (unique_gb_payload_elements),
            "physical_gb_transfer_elements": physical_gb_transfer_elements,
            "physical_wr_gb_elements": physical_gb_transfer_elements,
            "gb_replication_and_reload_factor": (
                physical_gb_transfer_elements / unique_gb_payload_elements
            ),
            "physical_padded_scalar_macs": physical_padded_scalar_macs,
            "physical_to_logical_mac_ratio": (
                physical_padded_scalar_macs / logical_scalar_macs
            ),
            "input_source": operation.input_source,
            "activation": operation.activation,
            "runtime_repeat": 1,
        }

    def _contraction_channels_batched(self, operation: AimContraction) -> dict:
        """Map batches to channels and output coordinates to their banks."""
        geometry = self.geometry
        channels, channels_per_replica = self._specialized_contraction_channels(
            operation
        )
        replica_channel_groups = tuple(
            channels[
                replica * channels_per_replica : (replica + 1) * channels_per_replica
            ]
            for replica in range(operation.replicas)
        )
        batch_groups = math.ceil(operation.batches / channels_per_replica)
        output_tiles = math.ceil(operation.outputs / geometry.banks)
        output_tile_layouts = [
            {
                "index": output_tile,
                "output_begin": output_tile * geometry.banks,
                "valid_outputs_per_batch": min(
                    geometry.banks,
                    operation.outputs - output_tile * geometry.banks,
                ),
                "physical_output_slots_per_batch": geometry.banks,
                "padded_output_slots_per_batch": (
                    geometry.banks
                    - min(
                        geometry.banks,
                        operation.outputs - output_tile * geometry.banks,
                    )
                ),
            }
            for output_tile in range(output_tiles)
        ]
        reduction_rows = math.ceil(operation.reduction / geometry.row_elements)
        storage_extent = (
            operation.reduction
            if operation.reduction_storage_extent is None
            else operation.reduction_storage_extent
        )
        storage_rows = math.ceil(storage_extent / geometry.row_elements)
        allocated_rows = batch_groups * output_tiles * storage_rows
        accessed_rows = batch_groups * output_tiles * reduction_rows
        self._check_rows(operation.row, allocated_rows)
        max_chunk_columns = math.ceil(
            min(operation.reduction, geometry.row_elements) / geometry.lanes
        )
        self._check_gpr(operation.input_gpr, max_chunk_columns)
        self._check_gpr(operation.bias_gpr)
        self._check_gpr(operation.result_gpr)

        groups = []
        reuse_scopes = []
        physical_gb_transfer_elements = 0
        for batch_group in range(batch_groups):
            batch_begin = batch_group * channels_per_replica
            group_batches = min(channels_per_replica, operation.batches - batch_begin)
            group_channels = tuple(
                channel
                for replica_channels in replica_channel_groups
                for channel in replica_channels[:group_batches]
            )
            mask = _channel_mask(group_channels, geometry)
            chunks = []
            for reduction_row in range(reduction_rows):
                reduction_begin = reduction_row * geometry.row_elements
                row_elements = min(
                    geometry.row_elements,
                    operation.reduction - reduction_begin,
                )
                op_size = math.ceil(row_elements / geometry.lanes)
                self.emit(f"AiM WR_GB {op_size} {operation.input_gpr} {mask}")
                matrix_rows = tuple(
                    operation.row
                    + batch_group * output_tiles * storage_rows
                    + output_tile * storage_rows
                    + reduction_row
                    for output_tile in range(output_tiles)
                )
                scope = self._emit_contraction_reuse_scope(
                    operation,
                    mask,
                    matrix_rows,
                    op_size,
                    final_reduction_chunk=(reduction_row == reduction_rows - 1),
                )
                scope.update(
                    {
                        "batch_group": batch_group,
                        "reduction_row": reduction_row,
                        "matrix_rows": list(matrix_rows),
                    }
                )
                reuse_scopes.append(scope)
                chunks.append(
                    {
                        "index": reduction_row,
                        "element_begin": reduction_begin,
                        "elements": row_elements,
                        "op_size": op_size,
                        "matrix_rows": list(matrix_rows),
                    }
                )
                physical_gb_transfer_elements += (
                    op_size * geometry.lanes * len(group_channels)
                )
            groups.append(
                {
                    "index": batch_group,
                    "batch_begin": batch_begin,
                    "batches": group_batches,
                    "active_channels": list(group_channels),
                    "channel_mask": mask,
                    "row_base": (
                        operation.row + batch_group * output_tiles * storage_rows
                    ),
                    "row_chunks": chunks,
                }
            )

        nominal_reuse = _select_reuse_group(operation, geometry, output_tiles)
        aligned_reduction = (
            math.ceil(operation.reduction / geometry.lanes) * geometry.lanes
        )
        logical_scalar_macs = (
            operation.outputs
            * operation.reduction
            * operation.batches
            * operation.replicas
        )
        physical_padded_scalar_macs = (
            operation.batches
            * operation.replicas
            * output_tiles
            * geometry.banks
            * aligned_reduction
        )
        unique_gb_payload_elements = (
            operation.batches * operation.reduction * operation.replicas
        )
        actually_active_channels = tuple(
            dict.fromkeys(
                channel for group in groups for channel in group["active_channels"]
            )
        )
        return {
            "outputs_per_instance": operation.outputs,
            "batches": operation.batches,
            "replicas": operation.replicas,
            "total_outputs": (
                operation.outputs * operation.batches * operation.replicas
            ),
            "batch_mapping": "channels",
            "layout": {
                "kind": "channel_batched_outputs",
                "iteration_order": [
                    "batch_group",
                    "reduction_row",
                    "reuse_group",
                ],
                "matrix_row_address": (
                    "row + batch_group * output_tiles * storage_rows + "
                    "output_tile * storage_rows + reduction_row"
                ),
                "reuse_dependency_scope": "one WR_GB live range",
                "reuse_crosses_wr_gb": False,
                "tail_channels_repeat_per_replica": True,
                "output_tail_policy": (
                    "zero-pad unused banks in the final output tile"
                ),
                "matrix_storage_contract": {
                    "logical_axes": [
                        "replica",
                        "batch",
                        "output",
                        "reduction",
                    ],
                    "batch_group": "batch // channels_per_replica",
                    "channel_within_replica": ("batch % channels_per_replica"),
                    "physical_channel": (
                        "replica_channel_groups[replica]" "[channel_within_replica]"
                    ),
                    "output_tile": "output // active_banks_per_channel",
                    "bank": "output % active_banks_per_channel",
                    "reduction_row": "reduction // row_elements",
                    "row": (
                        "row_base + batch_group * output_tiles * "
                        "reduction_storage_rows + output_tile * "
                        "reduction_storage_rows + reduction_row"
                    ),
                    "column": "reduction % row_elements",
                    "row_base": operation.row,
                    "row_elements": geometry.row_elements,
                    "replica_channel_groups": [
                        list(group) for group in replica_channel_groups
                    ],
                    "required_replica_coverage": list(range(operation.replicas)),
                    "producer_must_populate_all_replica_channel_groups": True,
                },
            },
            "configured_channels": list(channels),
            "active_channels": list(actually_active_channels),
            "active_channel_count": len(actually_active_channels),
            "active_banks_per_channel": geometry.banks,
            "channels_per_replica": channels_per_replica,
            "replica_channel_groups": [list(group) for group in replica_channel_groups],
            "output_capacity_per_tile_per_batch": geometry.banks,
            "output_tiles": output_tiles,
            "output_tile_layouts": output_tile_layouts,
            "output_utilization": (operation.outputs / (output_tiles * geometry.banks)),
            "physical_output_slots_per_batch_per_replica": (
                output_tiles * geometry.banks
            ),
            "padded_output_slots_per_batch_per_replica": (
                output_tiles * geometry.banks - operation.outputs
            ),
            "reduction": operation.reduction,
            "aligned_reduction": aligned_reduction,
            "reduction_rows": reduction_rows,
            "reduction_storage_extent": storage_extent,
            "reduction_storage_rows": storage_rows,
            "batch_pack": channels_per_replica,
            "batch_groups": batch_groups,
            "groups": groups,
            "wr_gb_launches": batch_groups * reduction_rows,
            "wr_gb_launches_per_reduction_row": batch_groups,
            "allocated_matrix_rows": allocated_rows,
            "accessed_matrix_rows": accessed_rows,
            "matrix_row_range": [operation.row, operation.row + allocated_rows],
            "reuse_group_candidates": list(nominal_reuse[3]),
            "reuse_group_size": nominal_reuse[0],
            "configured_reuse_group_size": nominal_reuse[1],
            "reuse_window_capacity": geometry.reuse_window,
            "reuse_window_entries_per_launch": (2 if operation.activation else 1),
            "effective_reuse_capacity": nominal_reuse[2],
            "reuse_scopes": reuse_scopes,
            "logical_scalar_macs": logical_scalar_macs,
            "physical_mac_slots": physical_padded_scalar_macs,
            "unique_gb_payload_elements": unique_gb_payload_elements,
            "unique_logical_gb_payload_elements": (unique_gb_payload_elements),
            "physical_gb_transfer_elements": physical_gb_transfer_elements,
            "physical_wr_gb_elements": physical_gb_transfer_elements,
            "gb_replication_and_reload_factor": (
                physical_gb_transfer_elements / unique_gb_payload_elements
            ),
            "physical_padded_scalar_macs": physical_padded_scalar_macs,
            "physical_to_logical_mac_ratio": (
                physical_padded_scalar_macs / logical_scalar_macs
            ),
            "input_source": operation.input_source,
            "activation": operation.activation,
            "runtime_repeat": 1,
        }

    def _contraction_flattened(self, operation: AimContraction) -> dict:
        geometry = self.geometry
        total_outputs = operation.outputs * operation.batches * operation.replicas
        channels = _contraction_channels(operation, geometry, total_outputs)
        mask = _channel_mask(channels, geometry)
        output_capacity = len(channels) * geometry.banks
        launches = math.ceil(total_outputs / output_capacity)
        reduction_rows = math.ceil(operation.reduction / geometry.row_elements)
        self._check_rows(operation.row, launches * reduction_rows)
        (
            selected_reuse,
            configured_reuse,
            reuse_capacity,
            candidates,
        ) = _select_reuse_group(operation, geometry, launches)

        reduction_columns = math.ceil(operation.reduction / geometry.lanes)
        derived_channels_per_replica = min(
            len(channels),
            max(1, math.ceil(operation.outputs / geometry.banks)),
        )
        channels_per_replica = (
            derived_channels_per_replica
            if operation.channels_per_replica is None
            else operation.channels_per_replica
        )
        if channels_per_replica > len(channels):
            raise ValueError(
                "channels_per_replica exceeds the active contraction channels"
            )
        gb_launches_per_row = (
            math.ceil(operation.batches / channels_per_replica)
            if operation.input_source == "gb"
            else 0
        )
        if operation.input_source == "gb" and operation.batches > 1:
            raise ValueError(
                "the 'flattened' batch mapping cannot represent distinct "
                "batch GB payloads; select 'row_packed' or 'channels'"
            )
        if operation.input_source == "gb":
            self._check_gpr(
                operation.input_gpr,
                reduction_columns * gb_launches_per_row,
            )
        self._check_gpr(operation.bias_gpr)
        self._check_gpr(operation.result_gpr)

        row_chunks = []
        for reduction_row in range(reduction_rows):
            row_offset = reduction_row * geometry.row_elements
            row_elements = min(geometry.row_elements, operation.reduction - row_offset)
            op_size = math.ceil(row_elements / geometry.lanes)
            gpr_addresses = []
            if operation.input_source == "gb":
                for gb_launch in range(gb_launches_per_row):
                    gpr_addr = (
                        operation.input_gpr
                        + gb_launch * reduction_columns
                        + row_offset // geometry.lanes
                    )
                    self.emit(f"AiM WR_GB {op_size} {gpr_addr} {mask}")
                    gpr_addresses.append(gpr_addr)

            for group_start in range(0, launches, selected_reuse):
                group = tuple(
                    range(group_start, min(launches, group_start + selected_reuse))
                )
                for _launch in group:
                    self.emit(f"AiM WR_BIAS {operation.bias_gpr} {mask}")
                for launch in group:
                    matrix_row = operation.row + launch * reduction_rows + reduction_row
                    self.emit(f"AiM MAC_ABK {op_size} {mask} {matrix_row}")
                if operation.activation and reduction_row == reduction_rows - 1:
                    for _launch in group:
                        self.emit(f"AiM AF {mask}")
                        self.emit(f"AiM RD_AF {operation.result_gpr} {mask}")
                for _launch in group:
                    self.emit(f"AiM RD_MAC {operation.result_gpr} {mask}")
            row_chunks.append(
                {
                    "index": reduction_row,
                    "elements": row_elements,
                    "op_size": op_size,
                    "input_gpr_addresses": gpr_addresses,
                }
            )

        padded_reduction_elements = sum(
            chunk["op_size"] * geometry.lanes for chunk in row_chunks
        )
        logical_scalar_macs = total_outputs * operation.reduction
        physical_mac_slots = (
            launches * len(channels) * geometry.banks * padded_reduction_elements
        )
        unique_gb_payload_elements = (
            operation.batches * operation.reduction * operation.replicas
            if operation.input_source == "gb"
            else 0
        )
        physical_wr_gb_elements = (
            gb_launches_per_row * len(channels) * padded_reduction_elements
        )
        return {
            "outputs_per_instance": operation.outputs,
            "batches": operation.batches,
            "replicas": operation.replicas,
            "total_outputs": total_outputs,
            "batch_mapping": "flattened",
            "layout": {
                "kind": "flattened_output_frontier",
                "iteration_order": [
                    "reduction_row",
                    "reuse_group",
                ],
                "matrix_row_address": (
                    "row + output_launch * reduction_rows + reduction_row"
                ),
                "matrix_storage_contract": {
                    "logical_axes": [
                        "replica",
                        "batch",
                        "output",
                        "reduction",
                    ],
                    "flattened_output": (
                        "(replica * batches + batch) * outputs + output"
                    ),
                    "output_launch": ("flattened_output // output_capacity_per_launch"),
                    "channel": (
                        "(flattened_output % output_capacity_per_launch) "
                        "// active_banks_per_channel"
                    ),
                    "bank": ("flattened_output % active_banks_per_channel"),
                    "row": (
                        "row_base + output_launch * reduction_rows + " "reduction_row"
                    ),
                    "column": "reduction % row_elements",
                },
            },
            "active_channels": list(channels),
            "active_channel_count": len(channels),
            "active_banks_per_channel": geometry.banks,
            "output_capacity_per_launch": output_capacity,
            "output_launches": launches,
            "output_utilization": total_outputs / (launches * output_capacity),
            "channels_per_replica": channels_per_replica,
            "wr_gb_launches": gb_launches_per_row * reduction_rows,
            "wr_gb_launches_per_reduction_row": gb_launches_per_row,
            "reduction": operation.reduction,
            "aligned_reduction": padded_reduction_elements,
            "input_source": operation.input_source,
            "reduction_rows": reduction_rows,
            "reduction_storage_extent": operation.reduction,
            "reduction_storage_rows": reduction_rows,
            "allocated_matrix_rows": launches * reduction_rows,
            "accessed_matrix_rows": launches * reduction_rows,
            "matrix_row_range": [
                operation.row,
                operation.row + launches * reduction_rows,
            ],
            "row_chunks": row_chunks,
            "reuse_group_candidates": list(candidates),
            "reuse_group_size": selected_reuse,
            "configured_reuse_group_size": configured_reuse,
            "reuse_window_capacity": geometry.reuse_window,
            "reuse_window_entries_per_launch": (2 if operation.activation else 1),
            "effective_reuse_capacity": reuse_capacity,
            "logical_scalar_macs": logical_scalar_macs,
            "physical_mac_slots": physical_mac_slots,
            "physical_padded_scalar_macs": physical_mac_slots,
            "physical_to_logical_mac_ratio": (physical_mac_slots / logical_scalar_macs),
            "unique_gb_payload_elements": unique_gb_payload_elements,
            "unique_logical_gb_payload_elements": (unique_gb_payload_elements),
            "physical_gb_transfer_elements": physical_wr_gb_elements,
            "physical_wr_gb_elements": physical_wr_gb_elements,
            "gb_replication_and_reload_factor": (
                physical_wr_gb_elements / unique_gb_payload_elements
                if unique_gb_payload_elements
                else 0.0
            ),
            "activation": operation.activation,
            "runtime_repeat": 1,
        }

    def _elementwise(self, operation: AimElementwise) -> dict:
        geometry = self.geometry
        if operation.operation == "add":
            columns = math.ceil(operation.elements / geometry.lanes)
            self._check_gpr(operation.gpr_addr_0, columns)
            self._check_gpr(operation.gpr_addr_1, columns)
            self.emit(
                f"AiM EWADD {columns} {operation.gpr_addr_0} " f"{operation.gpr_addr_1}"
            )
            return {
                "operation": "add",
                "elements": operation.elements,
                "op_size": columns,
                "active_channels": [],
                "logical_elements": operation.elements,
                "physical_element_slots": columns * geometry.lanes,
                "physical_to_logical_element_ratio": (
                    columns * geometry.lanes / operation.elements
                ),
            }

        channels, channels_per_replica = _replica_channel_layout(
            operation.channels,
            operation.replicas,
            operation.channels_per_replica,
            geometry,
        )
        mask = _channel_mask(channels, geometry)
        partitions = (
            channels_per_replica * geometry.bank_groups
            if operation.partitions_per_replica is None
            else operation.partitions_per_replica
        )
        if partitions < channels_per_replica:
            raise ValueError("partitions_per_replica cannot be fewer than channels")
        span = math.ceil(operation.elements / partitions)
        rows = math.ceil(span / geometry.row_elements)
        self._check_rows(operation.row, rows)
        chunks = []
        for row_offset in range(rows):
            elements = min(
                geometry.row_elements, span - row_offset * geometry.row_elements
            )
            columns = math.ceil(elements / geometry.lanes)
            self.emit(f"AiM EWMUL {columns} {mask} {operation.row + row_offset}")
            chunks.append(columns)
        return {
            "operation": "mul",
            "elements": operation.elements,
            "replicas": operation.replicas,
            "active_channels": list(channels),
            "channels_per_replica": channels_per_replica,
            "partitions_per_replica": partitions,
            "elements_per_partition": span,
            "row_op_sizes": chunks,
            "logical_elements": operation.elements * operation.replicas,
            "physical_element_slots": (
                sum(chunks) * geometry.lanes * len(channels) * geometry.bank_groups
            ),
            "physical_to_logical_element_ratio": (
                sum(chunks)
                * geometry.lanes
                * len(channels)
                * geometry.bank_groups
                / (operation.elements * operation.replicas)
            ),
        }

    def _activation(self, operation: AimActivation) -> dict:
        channels = _channels_or_all(operation.channels, self.geometry)
        mask = _channel_mask(channels, self.geometry)
        self._check_gpr(operation.result_gpr)
        for _group in range(operation.groups):
            self.emit(f"AiM AF {mask}")
            if operation.readback:
                self.emit(f"AiM RD_AF {operation.result_gpr} {mask}")
        return {
            "groups": operation.groups,
            "active_channels": list(channels),
            "readback": operation.readback,
        }

    def _host_transfer(self, operation: AimHostTransfer) -> dict:
        geometry = self.geometry
        if operation.channel >= geometry.channels:
            raise ValueError(
                f"channel {operation.channel} exceeds target count {geometry.channels}"
            )
        if operation.bank >= geometry.banks:
            raise ValueError(
                f"bank {operation.bank} exceeds target count {geometry.banks}"
            )
        self._check_rows(operation.row, 1)
        prefix = "R" if operation.direction == "read" else "W"
        for _burst in range(operation.bursts):
            self.emit(
                f"{prefix} MEM {operation.channel} {operation.bank} {operation.row}"
            )
        return {
            "direction": operation.direction,
            "bursts": operation.bursts,
            "burst_bits": geometry.vector_bits,
        }

    def _distributed_host_transfer(self, operation: AimDistributedHostTransfer) -> dict:
        geometry = self.geometry
        if geometry.banks % operation.bank_stride:
            raise ValueError("bank_stride must divide the target bank count")
        banks_per_channel = geometry.banks // operation.bank_stride
        if operation.bank_offset + operation.copies > operation.bank_stride:
            raise ValueError(
                "bank_offset + copies must fit within one bank-stride group"
            )
        physical_channels = operation.replicas * operation.channels_per_replica
        if physical_channels > geometry.channels:
            raise ValueError(
                "distributed host-transfer replicas exceed target channels"
            )
        partitions = operation.channels_per_replica * banks_per_channel
        span = math.ceil(operation.elements / partitions)
        rows = math.ceil(span / geometry.row_elements)
        self._check_rows(operation.row, rows)
        prefix = "R" if operation.direction == "read" else "W"
        emitted = 0

        # Row/burst/partition/replica/copy is an intentional stable ABI.
        for row_offset in range(rows):
            row_start = row_offset * geometry.row_elements
            max_row_elements = min(geometry.row_elements, span - row_start)
            max_bursts = math.ceil(max_row_elements / geometry.lanes)
            for burst in range(max_bursts):
                for partition in range(partitions):
                    partition_start = partition * span
                    partition_elements = min(span, operation.elements - partition_start)
                    if partition_elements <= row_start + burst * geometry.lanes:
                        continue
                    local_channel = partition // banks_per_channel
                    bank_slot = partition % banks_per_channel
                    for replica in range(operation.replicas):
                        channel = (
                            replica * operation.channels_per_replica + local_channel
                        )
                        for copy in range(operation.copies):
                            bank = (
                                operation.bank_offset
                                + copy
                                + bank_slot * operation.bank_stride
                            )
                            self.emit(
                                f"{prefix} MEM {channel} {bank} "
                                f"{operation.row + row_offset}"
                            )
                            emitted += 1
        return {
            "direction": operation.direction,
            "elements_per_replica": operation.elements,
            "replicas": operation.replicas,
            "channels_per_replica": operation.channels_per_replica,
            "partitions_per_replica": partitions,
            "elements_per_partition": span,
            "rows_per_partition": rows,
            "bank_stride": operation.bank_stride,
            "bank_offset": operation.bank_offset,
            "copies": operation.copies,
            "bursts": emitted,
            "iteration_order": [
                "row",
                "burst",
                "partition",
                "replica",
                "copy",
            ],
        }

    def _bank_copy(self, operation: AimBankCopy) -> dict:
        geometry = self.geometry
        channels, channels_per_replica = _replica_channel_layout(
            operation.channels,
            operation.replicas,
            operation.channels_per_replica,
            geometry,
        )
        last_bank = operation.bank + (operation.copies - 1) * operation.bank_stride
        if last_bank >= geometry.banks:
            raise ValueError(f"bank {last_bank} exceeds target count {geometry.banks}")
        mask = _channel_mask(channels, geometry)
        partitions = (
            channels_per_replica * operation.copies
            if operation.partitions_per_replica is None
            else operation.partitions_per_replica
        )
        span = math.ceil(operation.elements / partitions)
        rows = math.ceil(span / geometry.row_elements)
        self._check_rows(operation.row, rows)
        opcode = "COPY_BKGB" if operation.direction == "bank_to_gb" else "COPY_GBBK"
        chunks = []
        for row_offset in range(rows):
            elements = min(
                geometry.row_elements, span - row_offset * geometry.row_elements
            )
            columns = math.ceil(elements / geometry.lanes)
            for copy in range(operation.copies):
                bank = operation.bank + copy * operation.bank_stride
                self.emit(
                    f"AiM {opcode} {columns} {mask} {bank} "
                    f"{operation.row + row_offset}"
                )
            chunks.append(columns)
        return {
            "direction": operation.direction,
            "elements": operation.elements,
            "replicas": operation.replicas,
            "active_channels": list(channels),
            "channels_per_replica": channels_per_replica,
            "partitions_per_replica": partitions,
            "elements_per_partition": span,
            "copies": operation.copies,
            "banks": [
                operation.bank + copy * operation.bank_stride
                for copy in range(operation.copies)
            ],
            "row_op_sizes": chunks,
        }

    def _all_bank_write(self, operation: AimAllBankWrite) -> dict:
        geometry = self.geometry
        channels = _channels_or_all(operation.channels, geometry)
        self._check_gpr(operation.gpr_addr)
        copy_stride = (
            operation.rows * operation.row_stride
            if operation.copy_row_stride is None
            else operation.copy_row_stride
        )
        rows = []
        for copy in range(operation.copies):
            for row_index in range(operation.rows):
                row = (
                    operation.row
                    + copy * copy_stride
                    + row_index * operation.row_stride
                )
                self._check_rows(row, 1)
                rows.append(row)
                for channel in channels:
                    mask = _channel_mask((channel,), geometry)
                    self.emit(f"AiM WR_ABK {operation.gpr_addr} {mask} {row}")
        return {
            "active_channels": list(channels),
            "rows": rows,
            "copies": operation.copies,
            "row_stride": operation.row_stride,
            "copy_row_stride": copy_stride,
            "gpr_addr": operation.gpr_addr,
        }


def _loop_extent(bound) -> int | None:
    from ..spmw_tripcount import _parse_loop_bound

    return _parse_loop_bound(bound)


def _memref_shape(type_text: str | None) -> tuple[int, ...]:
    if not type_text or not type_text.startswith("memref<"):
        return ()
    body = type_text[len("memref<") :].split(">", 1)[0]
    dims = []
    for token in body.split("x")[:-1]:
        try:
            dims.append(int(token))
        except ValueError:
            return ()
    return tuple(dims)


def mac_operands(match):
    """Return ``(matrix, vector)`` operand bindings of one MAC match.

    The matrix is the bank-resident operand.  It is the input whose index set
    strictly contains the other input's (``K[h, s, d]`` over ``q[h, d]``,
    ``W[i, j]`` over ``x[j]``); otherwise the pattern's ``x`` role.  When both
    roles bind one memref, both entries are that operand.
    """
    roles = {operand.role: operand for operand in match.operands}
    x_operand, y_operand = roles.get("x"), roles.get("y")
    if x_operand is None or y_operand is None:
        raise NotImplementedError(
            f"AiM MAC match in {match.func_name} lacks an x or y operand"
        )
    if x_operand.memref_name == y_operand.memref_name:
        return x_operand, y_operand
    x_indices, y_indices = set(x_operand.indices), set(y_operand.indices)
    if y_indices > x_indices:
        return y_operand, x_operand
    return x_operand, y_operand


def contraction_shape(
    match, *, replicated: bool, replicas: int, channels: int
) -> dict:
    """Shape of one AiM contraction from one bucket's MAC match.

    A replicated group takes its shape from the bucket's loop extents: the
    innermost loop is the reduction, loops indexing the GB vector are the
    batch, and the remaining loops are the outputs.  A partitioned group's
    buckets cover one aggregate problem, so its shape comes from the matrix's
    memref extents.  ``storage_extent`` is the matrix extent along the
    reduction axis when it exceeds the reduction.
    """
    if not match.enclosing_loops:
        raise ValueError("AiM MAC match has no reduction loop")
    matrix, vector = mac_operands(match)
    reduction_var, _lb, reduction_ub, _step = match.enclosing_loops[-1]
    reduction = _loop_extent(reduction_ub)
    if reduction is None or reduction <= 0:
        raise ValueError(f"AiM reduction extent is not static: {reduction_ub}")

    # A bank-pair MAC (x and y bind one buffer) has no GB vector, so no batch.
    same_buffer = matrix.memref_name == vector.memref_name
    if replicated:
        batches = 1
        outputs = 1
        for var, _lb, ub, _step in match.enclosing_loops[:-1]:
            extent = _loop_extent(ub)
            if extent is None or extent <= 0:
                raise ValueError(f"AiM loop extent is not static: {ub}")
            if not same_buffer and var in vector.indices:
                batches *= extent
            else:
                outputs *= extent
    else:
        replicas = 1
        batch_var = match.extra.get("batch_loop_var")
        batches = 1 if same_buffer else int(match.extra.get("batch_dim", 1) or 1)
        outputs = None
        for operand in match.operands:
            if operand.is_loop_carried:
                continue
            shape = _memref_shape(getattr(operand, "memref_type", None))
            if len(shape) != len(operand.indices):
                continue
            if reduction_var not in operand.indices:
                continue
            for extent, index in zip(shape, operand.indices):
                if index in (reduction_var, batch_var):
                    continue
                outputs = extent
                break
            if outputs is not None:
                break
        if outputs is None:
            outer = _loop_extent(match.enclosing_loops[0][2])
            if outer is None:
                raise ValueError("AiM cannot recover output extent from match")
            outputs = outer * channels

    storage_extent = None
    shape = _memref_shape(getattr(matrix, "memref_type", None))
    if len(shape) == len(matrix.indices):
        for extent, index in zip(shape, matrix.indices):
            if index == reduction_var and extent > reduction:
                storage_extent = extent
    return {
        "outputs": int(outputs),
        "reduction": int(reduction),
        "batches": int(batches),
        "replicas": int(replicas),
        "storage_extent": storage_extent,
        "input_source": "banks" if same_buffer else "gb",
        "matrix": matrix.memref_name,
        "vector": vector.memref_name,
    }


def build_contraction(shape: dict, knobs: dict, *, activation=False, name=None):
    """The AimContraction one kernel lowers to under the given knob values.

    ``knobs`` maps ``batch_mapping``, ``reuse_group`` and
    ``replica_partitions``; a missing knob keeps the typed default.
    ``replicas`` overrides the shape's replica count (the profile prices one).
    """
    partitions = knobs.get("replica_partitions")
    replicas = int(knobs.get("replicas", shape["replicas"]))
    batch_mapping = knobs.get("batch_mapping") or "auto"
    reuse = knobs.get("reuse_group")
    return AimContraction(
        outputs=shape["outputs"],
        reduction=shape["reduction"],
        batches=shape["batches"],
        replicas=replicas,
        row=0,
        channels=(
            None if partitions is None else tuple(range(replicas * int(partitions)))
        ),
        channels_per_replica=None if partitions is None else int(partitions),
        input_source=shape["input_source"],
        batch_mapping=batch_mapping,
        reduction_storage_extent=shape["storage_extent"],
        reuse_group_size=None if not reuse else int(reuse),
        activation=bool(activation),
        name=name,
    )


class _ProfileCache:
    """Memo for ``aim_contraction_profile`` keyed on frozen record + geometry.

    The cost fingerprint walks every object a rule references; this cache is
    pure memoization of a deterministic function, so its manifest is constant.
    """

    def __init__(self):
        self.entries: dict = {}

    def __allo_fingerprint_manifest__(self):
        return "aim-contraction-profile-cache-v1"


_PROFILE_CACHE = _ProfileCache()


def lower_contraction(operation, geometry) -> list[str]:
    """Lower one AimContraction with the shared expander; raise ValueError if
    the record is unrealizable on ``geometry``."""
    lowerer = _AimLowerer.__new__(_AimLowerer)
    lowerer.target = None
    lowerer.geometry = geometry
    lowerer.commands = []
    lowerer.operations = []
    lowerer.lower(operation, 0)
    return lowerer.commands


def aim_contraction_profile(operation, geometry) -> dict:
    """Opcode counts, MAC column total, reuse groups and row activations.

    ``row_activations`` counts MAC lines whose ``(mask, row)`` differs from the
    previous MAC line; ``reuse_groups`` counts maximal WR_BIAS runs.
    """
    key = (operation, geometry)
    cached = _PROFILE_CACHE.entries.get(key)
    if cached is not None:
        return cached
    counts: dict[str, int] = {}
    mac_columns = 0
    mac_lines = 0
    row_activations = 0
    reuse_groups = 0
    previous_open = None
    previous_opcode = None
    for line in lower_contraction(operation, geometry):
        fields = line.split(" ")
        opcode = fields[1]
        counts[opcode] = counts.get(opcode, 0) + 1
        if opcode == "WR_BIAS" and previous_opcode != "WR_BIAS":
            reuse_groups += 1
        if opcode == "MAC_ABK":
            op_size, mask, row = (int(value) for value in fields[2:5])
            mac_lines += 1
            mac_columns += op_size
            if (mask, row) != previous_open:
                row_activations += 1
                previous_open = (mask, row)
        previous_opcode = opcode
    profile = {
        "counts": counts,
        "mac_lines": mac_lines,
        "mac_columns": mac_columns,
        "row_activations": row_activations,
        "reuse_groups": reuse_groups,
    }
    _PROFILE_CACHE.entries[key] = profile
    return profile


def _missing_fingerprint(match) -> ValueError:
    return ValueError(
        f"AiM kernel {match.func_name} carries a matcher work scope but no "
        "spmw_body_fingerprint; allo.compile must stamp it so replication is "
        "known (spec 002 A2)"
    )


def group_replicated(group_matches) -> bool:
    """Whether every bucket of one kernel group runs the same body.

    Reads the retained ``spmw_body_fingerprint`` of each match. A match that
    has a work scope but no fingerprint is an error, never a guess. Matches
    without any scope (synthetic traces) form one bucket per function.
    """
    functions = {match.func_name for match in group_matches}
    fingerprints = set()
    for match in group_matches:
        fingerprint = match.extra.get("spmw_body_fingerprint")
        if fingerprint is None:
            if "spmw_work_scope" in match.extra:
                raise _missing_fingerprint(match)
            return len(functions) == 1
        fingerprints.add(fingerprint)
    return len(fingerprints) == 1


def group_replication(match) -> tuple[bool, int]:
    """``(replicated, bucket_count)`` of the group owning one bucket's match.

    The whole-group flag ``spmw_group_replicated`` is stamped by
    ``rank_matcher_placements`` from the per-match fingerprints.
    """
    scope = match.extra.get("spmw_work_scope")
    shape = tuple(getattr(scope, "group_shape", ()) or ())
    buckets = math.prod(shape) if shape else 1
    replicated = match.extra.get("spmw_group_replicated")
    if replicated is not None:
        return bool(replicated), buckets
    if buckets == 1:
        return True, buckets
    if match.extra.get("spmw_body_fingerprint") is None:
        raise _missing_fingerprint(match)
    raise ValueError(
        f"AiM kernel {match.func_name}: group replication was not stamped; "
        "enumerate through rank_matcher_placements"
    )


def replica_partition_domain(replicas: int, channels: int) -> list[int]:
    """Powers of two ``p`` with ``replicas * p <= channels``, ascending."""
    values = []
    p = 1
    while replicas * p <= channels:
        values.append(p)
        p *= 2
    return values


def aligned_reduction(reduction: int, lanes: int) -> int:
    return math.ceil(reduction / lanes) * lanes


def resolve_batch_mapping(shape: dict, requested, geometry) -> str:
    """The batch mapping the lowerer realizes (its ``auto`` rule otherwise)."""
    if requested not in (None, "auto", "none"):
        return requested
    if shape["input_source"] == "banks":
        return "flattened"
    if shape["storage_extent"] is not None:
        return "channels"
    if shape["batches"] == 1:
        return "flattened"
    return "row_packed"


def matrix_axes(match) -> list[str]:
    """Role of each matrix dimension: ``reduction``, ``batch`` or ``output``."""
    matrix, vector = mac_operands(match)
    reduction_var = match.enclosing_loops[-1][0]
    same_buffer = matrix.memref_name == vector.memref_name
    roles = []
    for index in matrix.indices:
        if index == reduction_var:
            roles.append("reduction")
        elif not same_buffer and index in vector.indices:
            roles.append("batch")
        else:
            roles.append("output")
    return roles
