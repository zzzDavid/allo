# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable APU v1 iteration, storage-layout, and execution-plan values.

The physical mapping core remains :class:`allo.spmw_linear_layout.LinearLayout`:
an exact F2-linear map over power-of-two axes.  ``AffineTiledLayout`` adds a
validity-masked logical domain and an affine output view for non-power-of-two
problems without weakening those invariants.

This module is intentionally declarative.  It does not schedule, allocate, or
emit GVML code; candidate generators, cost programs, and codegen can consume
the same validated plan later.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field, is_dataclass
from itertools import product
import math
from types import MappingProxyType
from typing import Mapping

from ..spmw_linear_layout import LinearLayout


class UnsupportedLayoutError(ValueError):
    """Raised when a metric or coordinate cannot be represented faithfully."""


def _clone_linear(layout: LinearLayout) -> LinearLayout:
    if not isinstance(layout, LinearLayout):
        raise TypeError(f"expected LinearLayout, got {type(layout).__name__}")
    cloned = LinearLayout(
        {
            name: [tuple(vector) for vector in vectors]
            for name, vectors in layout.bases.items()
        },
        tuple(layout.out_dims),
        tuple(layout.out_sizes),
    )
    # LinearLayout predates immutable plan values and exposes mutable basis
    # lists.  Freeze only the retained snapshot; the original layout and the
    # core class remain unchanged for existing transformation code.
    cloned.bases = MappingProxyType(
        {name: tuple(vectors) for name, vectors in cloned.bases.items()}
    )
    return cloned


def _pairs(value, *, label: str) -> tuple[tuple[str, int], ...]:
    items = value.items() if isinstance(value, Mapping) else value
    result = tuple((str(name), int(extent)) for name, extent in items)
    names = [name for name, _extent in result]
    if len(names) != len(set(names)):
        raise ValueError(f"{label} axes must be unique")
    if any(not name.isidentifier() for name in names):
        raise ValueError(f"{label} axes must be identifiers")
    if any(extent <= 0 for _name, extent in result):
        raise ValueError(f"{label} extents must be positive")
    return result


def _linear_input_extents(layout: LinearLayout) -> dict[str, int]:
    return {name: layout.size_of(name) for name in layout.bases}


@dataclass(frozen=True)
class AffineTiledLayout:
    """Validity-masked affine view over an exact F2 carrier.

    ``logical_extents`` may be non-power-of-two but cannot exceed the carrier
    input extent.  The carrier therefore describes the padded tile; invalid
    coordinates are explicit rather than silently folded.  ``output_offsets``
    and ``output_strides`` are ordered like ``carrier.out_dims``.
    """

    carrier: LinearLayout
    logical_extents: tuple[tuple[str, int], ...] | Mapping[str, int]
    output_offsets: tuple[int, ...] = ()
    output_strides: tuple[int, ...] = ()
    physical_out_sizes: tuple[int, ...] | None = None

    def __post_init__(self):
        carrier = _clone_linear(self.carrier)
        object.__setattr__(self, "carrier", carrier)
        requested = dict(_pairs(self.logical_extents, label="logical"))
        carrier_extents = _linear_input_extents(carrier)
        unknown = set(requested) - set(carrier_extents)
        if unknown:
            raise ValueError(f"logical axes absent from carrier: {sorted(unknown)}")
        logical = []
        for name, padded in carrier_extents.items():
            extent = requested.get(name, padded)
            if extent > padded:
                raise ValueError(
                    f"logical extent {name}={extent} exceeds padded carrier extent {padded}"
                )
            logical.append((name, extent))
        object.__setattr__(self, "logical_extents", tuple(logical))

        n_out = len(carrier.out_dims)
        offsets = tuple(int(value) for value in self.output_offsets) or (0,) * n_out
        strides = tuple(int(value) for value in self.output_strides) or (1,) * n_out
        if len(offsets) != n_out or len(strides) != n_out:
            raise ValueError("output offsets/strides must match carrier.out_dims")
        if any(value < 0 for value in offsets) or any(value <= 0 for value in strides):
            raise ValueError("output offsets must be non-negative and strides positive")
        object.__setattr__(self, "output_offsets", offsets)
        object.__setattr__(self, "output_strides", strides)
        physical = (
            tuple(int(value) for value in self.physical_out_sizes)
            if self.physical_out_sizes is not None
            else tuple(carrier.out_sizes)
        )
        if len(physical) != n_out or any(value <= 0 for value in physical):
            raise ValueError(
                "physical_out_sizes must be positive and match carrier outputs"
            )
        object.__setattr__(self, "physical_out_sizes", physical)

    @staticmethod
    def _padded(value: int) -> int:
        value = int(value)
        if value <= 0:
            raise ValueError("logical extents must be positive")
        return 1 << (value - 1).bit_length()

    @classmethod
    def identity(
        cls, logical_extents: Mapping[str, int] | tuple[tuple[str, int], ...]
    ) -> "AffineTiledLayout":
        """Build a padded identity map with one physical output per axis."""

        logical = _pairs(logical_extents, label="logical")
        padded = {name: cls._padded(extent) for name, extent in logical}
        return cls(LinearLayout.identity(padded), logical)

    @classmethod
    def packed(
        cls,
        logical_extents: Mapping[str, int] | tuple[tuple[str, int], ...],
        *,
        out_dim: str = "vr_lane",
        max_extent: int | None = 32768,
    ) -> "AffineTiledLayout":
        """Pack padded logical axes row-major into one F2 output dimension."""

        logical = _pairs(logical_extents, label="logical")
        padded = [(name, cls._padded(extent)) for name, extent in logical]
        physical_extent = math.prod(extent for _name, extent in padded)
        if max_extent is not None and physical_extent > int(max_extent):
            raise UnsupportedLayoutError(
                f"packed layout needs {physical_extent} lanes; limit is {max_extent}"
            )
        shifts = {}
        trailing = 1
        for name, extent in reversed(padded):
            shifts[name] = trailing
            trailing *= extent
        bases = {}
        for name, extent in padded:
            bits = extent.bit_length() - 1
            bases[name] = [((shifts[name] << bit),) for bit in range(bits)]
        carrier = LinearLayout(bases, (out_dim,), (physical_extent,))
        return cls(carrier, logical)

    @property
    def input_extents(self) -> dict[str, int]:
        return dict(self.logical_extents)

    @property
    def padded_extents(self) -> dict[str, int]:
        return _linear_input_extents(self.carrier)

    @property
    def out_dims(self) -> tuple[str, ...]:
        return self.carrier.out_dims

    @property
    def out_sizes(self) -> tuple[int, ...]:
        return self.physical_out_sizes

    def is_valid(self, **indices: int) -> bool:
        logical = self.input_extents
        return set(indices) == set(logical) and all(
            0 <= int(indices[name]) < extent for name, extent in logical.items()
        )

    def apply(self, **logical_and_replica_indices: int) -> tuple[int, ...]:
        if not self.is_valid(**logical_and_replica_indices):
            raise ValueError(
                "AffineTiledLayout.apply requires every input axis within its logical extent"
            )
        base = self.carrier.apply(**logical_and_replica_indices)
        output = tuple(
            offset + stride * coordinate
            for offset, stride, coordinate in zip(
                self.output_offsets, self.output_strides, base
            )
        )
        if any(coordinate >= size for coordinate, size in zip(output, self.out_sizes)):
            raise UnsupportedLayoutError(
                f"affine output {output} exceeds physical extents {self.out_sizes}"
            )
        return output

    def coordinate(self, **logical_and_replica_indices: int) -> dict[str, int]:
        return dict(zip(self.out_dims, self.apply(**logical_and_replica_indices)))

    def lane_for(self, *, out_dim: str = "vr_lane", **indices: int) -> int:
        try:
            return self.coordinate(**indices)[out_dim]
        except KeyError as error:
            raise KeyError(f"layout has no output dimension {out_dim!r}") from error

    def manifest(self) -> dict[str, object]:
        return {
            "kind": "affine-tiled-f2",
            "carrier": self.carrier.manifest(),
            "logical_extents": dict(self.logical_extents),
            "padded_extents": self.padded_extents,
            "output_offsets": list(self.output_offsets),
            "output_strides": list(self.output_strides),
            "physical_out_sizes": list(self.out_sizes),
        }


Layout = LinearLayout | AffineTiledLayout


@dataclass(frozen=True)
class Validity:
    """Logical validity and deterministic padding for one stored value."""

    axes: tuple[str, ...]
    logical_shape: tuple[int, ...]
    padded_shape: tuple[int, ...]
    padding_value: object = 0

    def __post_init__(self):
        axes = tuple(self.axes)
        logical = tuple(int(value) for value in self.logical_shape)
        padded = tuple(int(value) for value in self.padded_shape)
        if not axes or len(set(axes)) != len(axes):
            raise ValueError("validity axes must be non-empty and unique")
        if len(axes) != len(logical) or len(axes) != len(padded):
            raise ValueError(
                "validity axes/logical_shape/padded_shape must have equal rank"
            )
        if any(value <= 0 for value in logical + padded):
            raise ValueError("validity shapes must be positive")
        if any(valid > pad for valid, pad in zip(logical, padded)):
            raise ValueError("logical validity cannot exceed padded shape")
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "logical_shape", logical)
        object.__setattr__(self, "padded_shape", padded)

    @property
    def valid_elements(self) -> int:
        return math.prod(self.logical_shape)

    @property
    def padded_elements(self) -> int:
        return math.prod(self.padded_shape)

    @property
    def padding_elements(self) -> int:
        return self.padded_elements - self.valid_elements

    def contains(self, **indices: int) -> bool:
        return set(indices) == set(self.axes) and all(
            0 <= int(indices[axis]) < extent
            for axis, extent in zip(self.axes, self.logical_shape)
        )

    def manifest(self) -> dict[str, object]:
        return {
            "axes": list(self.axes),
            "logical_shape": list(self.logical_shape),
            "padded_shape": list(self.padded_shape),
            "padding_value": self.padding_value,
        }


@dataclass(frozen=True)
class LayoutMetrics:
    domain_size: int
    padded_domain_size: int
    image_size: int
    fiber_size: int | None
    replication_factor: int | None
    occupancy: float
    unique_elements: int
    padding_elements: int
    address_span: int | None
    low_bit_contiguous: bool | None


def _layout_parts(layout: Layout):
    if isinstance(layout, AffineTiledLayout):
        return (
            layout.input_extents,
            layout.padded_extents,
            layout.out_dims,
            layout.out_sizes,
            layout.apply,
        )
    if isinstance(layout, LinearLayout):
        frozen = _clone_linear(layout)
        extents = _linear_input_extents(frozen)

        def apply(**indices):
            if set(indices) != set(extents):
                raise ValueError("LinearLayout metrics require every input coordinate")
            if any(
                not 0 <= int(indices[name]) < extent for name, extent in extents.items()
            ):
                raise ValueError(
                    "LinearLayout input coordinate is outside its F2 domain"
                )
            return frozen.apply(**indices)

        return extents, extents, frozen.out_dims, frozen.out_sizes, apply
    raise UnsupportedLayoutError(f"unsupported layout type {type(layout).__name__}")


def derive_layout_metrics(
    layout: Layout,
    *,
    varying_inputs: tuple[str, ...] | None = None,
    enumeration_limit: int = 1 << 14,
) -> LayoutMetrics:
    """Enumerate an exact finite layout image and derive storage metrics."""

    logical, padded, _out_dims, out_sizes, apply = _layout_parts(layout)
    varying = tuple(logical) if varying_inputs is None else tuple(varying_inputs)
    if len(set(varying)) != len(varying) or not set(varying) <= set(logical):
        raise ValueError("varying_inputs must be unique layout input axes")
    domain = math.prod(logical[name] for name in varying)
    padded_domain = math.prod(padded[name] for name in varying)
    carrier = layout.carrier if isinstance(layout, AffineTiledLayout) else layout
    full_padded_domain = all(logical[name] == padded[name] for name in varying)

    # The image of a full F2 domain is exactly 2**rank.  This is the common
    # large-model path (including domains with millions of replicated points)
    # and must never enumerate the domain.  A positive per-output affine scale
    # preserves equality/distinctness, so the carrier rank also applies to an
    # unmasked AffineTiledLayout.
    if full_padded_domain:
        image = carrier.image_size(varying_inputs=varying)
        fiber = domain // image
        capacity = math.prod(out_sizes)
        address_span = None
        low_bit_contiguous = None
        if domain <= enumeration_limit:
            # Address metrics are optional physical-locality details.  Compute
            # them only when their exact enumeration is representable.
            counts = Counter()
            fixed = {name: 0 for name in logical}
            for coordinate in product(*(range(logical[name]) for name in varying)):
                indices = dict(fixed)
                indices.update(zip(varying, coordinate))
                counts[apply(**indices)] += 1
            addresses = []
            for output in counts:
                address = 0
                for coordinate, size in zip(output, out_sizes):
                    address = address * size + coordinate
                addresses.append(address)
            address_span = max(addresses) - min(addresses) + 1 if addresses else 0
            low_bit_contiguous = bool(addresses) and set(addresses) == set(range(image))
        return LayoutMetrics(
            domain_size=domain,
            padded_domain_size=padded_domain,
            image_size=image,
            fiber_size=fiber,
            replication_factor=fiber,
            occupancy=image / capacity,
            unique_elements=image,
            padding_elements=0,
            address_span=address_span,
            low_bit_contiguous=low_bit_contiguous,
        )

    # A masked restriction of an injective F2 carrier remains injective.  This
    # common non-power-of-two tiled path has an exact analytical image equal to
    # its logical domain and must not enumerate hundreds of thousands of
    # coordinates merely to rediscover uniqueness.
    if (
        carrier.image_size(varying_inputs=varying) == padded_domain
        and domain > enumeration_limit
    ):
        capacity = math.prod(out_sizes)
        return LayoutMetrics(
            domain_size=domain,
            padded_domain_size=padded_domain,
            image_size=domain,
            fiber_size=1,
            replication_factor=1,
            occupancy=domain / capacity,
            unique_elements=domain,
            padding_elements=padded_domain - domain,
            address_span=None,
            low_bit_contiguous=None,
        )

    if domain > enumeration_limit:
        raise UnsupportedLayoutError(
            f"masked non-power-of-two domain {domain} exceeds exact enumeration "
            f"limit {enumeration_limit}; split it into smaller affine tiles"
        )

    counts = Counter()
    fixed = {name: 0 for name in logical}
    for coordinate in product(*(range(logical[name]) for name in varying)):
        indices = dict(fixed)
        indices.update(zip(varying, coordinate))
        output = apply(**indices)
        counts[output] += 1
    image = len(counts)
    fibers = set(counts.values())
    fiber = next(iter(fibers)) if len(fibers) == 1 else None
    capacity = math.prod(out_sizes)
    addresses = []
    for output in counts:
        if len(output) != len(out_sizes) or any(
            coordinate < 0 or coordinate >= size
            for coordinate, size in zip(output, out_sizes)
        ):
            raise UnsupportedLayoutError(
                f"layout output {output} cannot be represented by extents {out_sizes}"
            )
        address = 0
        for coordinate, size in zip(output, out_sizes):
            address = address * size + coordinate
        addresses.append(address)
    address_span = max(addresses) - min(addresses) + 1 if addresses else 0
    low_bit_contiguous = bool(addresses) and set(addresses) == set(range(image))
    return LayoutMetrics(
        domain_size=domain,
        padded_domain_size=padded_domain,
        image_size=image,
        fiber_size=fiber,
        replication_factor=fiber,
        occupancy=image / capacity,
        unique_elements=image,
        padding_elements=padded_domain - domain,
        address_span=address_span,
        low_bit_contiguous=low_bit_contiguous,
    )


@dataclass(frozen=True)
class TemporalAxis:
    name: str
    extent: int
    order: str = "forward"
    carried_values: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.name.isidentifier() or int(self.extent) <= 0:
            raise ValueError("temporal axis needs an identifier and positive extent")
        if self.order not in {"forward", "reverse", "wavefront", "host"}:
            raise ValueError(f"unsupported temporal order {self.order!r}")
        object.__setattr__(self, "extent", int(self.extent))
        object.__setattr__(self, "carried_values", tuple(self.carried_values))


@dataclass(frozen=True)
class IterationLayout:
    axes: tuple[str, ...]
    layout: Layout
    temporal_axes: tuple[TemporalAxis, ...] = ()

    def __post_init__(self):
        axes = tuple(self.axes)
        if not axes or len(axes) != len(set(axes)):
            raise ValueError("iteration axes must be non-empty and unique")
        layout = (
            self.layout
            if isinstance(self.layout, AffineTiledLayout)
            else _clone_linear(self.layout)
        )
        inputs = set(_layout_parts(layout)[0])
        if not set(axes) <= inputs:
            raise ValueError(
                f"iteration axes absent from layout: {sorted(set(axes)-inputs)}"
            )
        temporal = tuple(self.temporal_axes)
        if any(not isinstance(axis, TemporalAxis) for axis in temporal):
            raise TypeError("temporal_axes must contain TemporalAxis values")
        if not {axis.name for axis in temporal} <= set(axes):
            raise ValueError("temporal axes must also be iteration axes")
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "temporal_axes", temporal)

    def metrics(self) -> LayoutMetrics:
        return derive_layout_metrics(self.layout, varying_inputs=self.axes)

    def coordinate(self, **indices: int) -> dict[str, int]:
        logical, _padded, out_dims, _out_sizes, apply = _layout_parts(self.layout)
        if set(indices) != set(logical):
            raise ValueError("coordinate requires every iteration-layout input axis")
        return dict(zip(out_dims, apply(**indices)))

    def lane_for(self, *, out_dim: str = "vr_lane", **indices: int) -> int:
        try:
            return self.coordinate(**indices)[out_dim]
        except KeyError as error:
            raise KeyError(
                f"iteration layout has no output dimension {out_dim!r}"
            ) from error

    def manifest(self) -> dict[str, object]:
        return {
            "axes": list(self.axes),
            "layout": self.layout.manifest(),
            "temporal_axes": [_manifest_value(axis) for axis in self.temporal_axes],
        }


@dataclass(frozen=True)
class ValueMetrics:
    layout: LayoutMetrics
    logical_elements: int
    physical_elements: int
    explicit_replication_factor: int
    padding_elements: int


@dataclass(frozen=True)
class ValueLayout:
    value: str
    axes: tuple[str, ...]
    replica_axes: tuple[str, ...] = ()
    layout: Layout | None = None
    validity: Validity | None = None
    storage: str = "vr"

    def __post_init__(self):
        axes = tuple(self.axes)
        replicas = tuple(self.replica_axes)
        if not self.value.isidentifier() or not self.storage:
            raise ValueError("value must be an identifier and storage non-empty")
        if len(axes) != len(set(axes)) or len(replicas) != len(set(replicas)):
            raise ValueError("value and replica axes must be unique")
        if set(axes) & set(replicas):
            raise ValueError("replica_axes must be disjoint from logical value axes")
        layout = self.layout
        if layout is not None and not isinstance(
            layout, (LinearLayout, AffineTiledLayout)
        ):
            raise UnsupportedLayoutError(
                f"unsupported value layout {type(layout).__name__}"
            )
        if isinstance(layout, LinearLayout):
            layout = _clone_linear(layout)
        if self.validity is not None and self.validity.axes != axes:
            raise ValueError("validity axes must exactly match ValueLayout.axes")
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "replica_axes", replicas)
        object.__setattr__(self, "layout", layout)

    def effective_layout(
        self, iteration_layout: IterationLayout | None = None
    ) -> Layout:
        if self.layout is not None:
            return self.layout
        if iteration_layout is None:
            raise UnsupportedLayoutError(
                f"value {self.value!r} inherits its layout but no IterationLayout was supplied"
            )
        return iteration_layout.layout

    def metrics(self, iteration_layout: IterationLayout | None = None) -> ValueMetrics:
        layout = self.effective_layout(iteration_layout)
        logical, _padded, _out_dims, _out_sizes, _apply = _layout_parts(layout)
        varying = self.axes + self.replica_axes
        missing = set(varying) - set(logical)
        if missing:
            raise ValueError(
                f"value axes absent from storage layout: {sorted(missing)}"
            )
        base = derive_layout_metrics(layout, varying_inputs=varying)
        if self.validity is None:
            logical_elements = math.prod(logical[axis] for axis in self.axes)
            padding = 0
        else:
            logical_elements = self.validity.valid_elements
            padding = self.validity.padding_elements
        replicas = math.prod(logical[axis] for axis in self.replica_axes)
        return ValueMetrics(base, logical_elements, base.image_size, replicas, padding)

    def coordinate(
        self, iteration_layout: IterationLayout | None = None, **indices: int
    ) -> dict[str, int]:
        layout = self.effective_layout(iteration_layout)
        logical, _padded, out_dims, _out_sizes, apply = _layout_parts(layout)
        if set(indices) != set(logical):
            raise ValueError("coordinate requires every storage-layout input axis")
        return dict(zip(out_dims, apply(**indices)))

    def lane_for(
        self,
        iteration_layout: IterationLayout | None = None,
        *,
        out_dim: str = "vr_lane",
        **indices: int,
    ) -> int:
        try:
            return self.coordinate(iteration_layout, **indices)[out_dim]
        except KeyError as error:
            raise KeyError(
                f"value layout has no output dimension {out_dim!r}"
            ) from error

    def manifest(self) -> dict[str, object]:
        return {
            "value": self.value,
            "axes": list(self.axes),
            "replica_axes": list(self.replica_axes),
            "layout": (
                self.layout.manifest() if self.layout is not None else "iteration"
            ),
            "validity": self.validity.manifest() if self.validity is not None else None,
            "storage": self.storage,
        }


_TRANSFER_DIRECTIONS = {
    "in",
    "out",
    "inout",
    "load",
    "store",
    "l4_to_vr",
    "vr_to_l4",
    "vr_to_vr",
    "host_to_l4",
    "l4_to_host",
}


@dataclass(frozen=True)
class TransferLayout:
    """One physical representation participating in a value transfer.

    Unlike :class:`ValueLayout`, this value describes an intermediate
    representation rather than the canonical compute representation retained
    by an :class:`APUV1Plan`.  Its logical and replica axes make compaction and
    expansion visible to code generation and costing; storage names merely
    select the target memory/register level.
    """

    storage: str
    axes: tuple[str, ...]
    layout: Layout
    replica_axes: tuple[str, ...] = ()
    role: str = ""
    padded_tile_extents: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self):
        axes = tuple(self.axes)
        replicas = tuple(self.replica_axes)
        if not self.storage:
            raise ValueError("transfer layout storage must be non-empty")
        if len(axes) != len(set(axes)) or len(replicas) != len(set(replicas)):
            raise ValueError("transfer layout axes must be unique")
        if set(axes) & set(replicas):
            raise ValueError("transfer layout replica axes must be disjoint")
        layout = self.layout
        if not isinstance(layout, (LinearLayout, AffineTiledLayout)):
            raise UnsupportedLayoutError(
                f"unsupported transfer layout {type(layout).__name__}"
            )
        if isinstance(layout, LinearLayout):
            layout = _clone_linear(layout)
        inputs = set(_layout_parts(layout)[0])
        missing = set(axes + replicas) - inputs
        if missing:
            raise ValueError(
                f"transfer axes absent from physical layout: {sorted(missing)}"
            )
        padded_tiles = {
            str(axis): int(value) for axis, value in self.padded_tile_extents.items()
        }
        unknown_padding = set(padded_tiles) - set(axes + replicas)
        if unknown_padding or any(value <= 0 for value in padded_tiles.values()):
            raise ValueError(
                "padded transfer tiles must be positive and reference layout axes"
            )
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "replica_axes", replicas)
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "padded_tile_extents", MappingProxyType(padded_tiles))

    @property
    def all_axes(self) -> tuple[str, ...]:
        return self.axes + self.replica_axes

    @property
    def axis_extents(self) -> dict[str, int]:
        logical, _padded, _dims, _sizes, _apply = _layout_parts(self.layout)
        return {axis: logical[axis] for axis in self.all_axes}

    def metrics(self) -> LayoutMetrics:
        return derive_layout_metrics(self.layout, varying_inputs=self.all_axes)

    def manifest(self) -> dict[str, object]:
        return {
            "storage": self.storage,
            "axes": list(self.axes),
            "replica_axes": list(self.replica_axes),
            "layout": self.layout.manifest(),
            "role": self.role,
            "padded_tile_extents": dict(self.padded_tile_extents),
        }


@dataclass(frozen=True)
class ReuseWindow:
    """A logical axis tiled in time while a transfer representation is live."""

    axis: str
    tile_extent: int
    extent: int | None = None

    def __post_init__(self):
        if not self.axis.isidentifier() or int(self.tile_extent) <= 0:
            raise ValueError("reuse windows need an axis and positive tile extent")
        object.__setattr__(self, "tile_extent", int(self.tile_extent))
        if self.extent is not None:
            extent = int(self.extent)
            if extent <= 0:
                raise ValueError("reuse window extent must be positive")
            object.__setattr__(self, "extent", extent)

    def factor(self, layout: TransferLayout | None = None) -> int:
        if self.extent is not None:
            return (self.extent + self.tile_extent - 1) // self.tile_extent
        if layout is None:
            raise ValueError("a layout is required when window extent is implicit")
        extents = layout.axis_extents
        if self.axis not in extents:
            raise ValueError(f"reuse axis {self.axis!r} is absent from transfer layout")
        return (extents[self.axis] + self.tile_extent - 1) // self.tile_extent


@dataclass(frozen=True)
class TransferRouteMetrics:
    """Layout-derived traffic for one route step over the full problem."""

    source_elements: int
    destination_elements: int
    source_elements_per_call: int
    destination_elements_per_call: int
    call_count: int
    expansion_factor: float
    resident_reuse_factor: int


_TRANSFER_ROUTE_KINDS = {
    "direct",
    "dma_l4_l3",
    "dma_l4_l1_32k",
    "dma_vr_l4",
    "load_vr",
    "lookup",
    "duplicate_subgroup",
}


@dataclass(frozen=True)
class TransferRouteStep:
    """A target operation relating two explicit physical layouts.

    ``executed_at`` partitions the full logical domain into calls.  In
    contrast, ``resident_across`` records axes whose tiles reuse the source
    representation without reloading it.  Keeping the two separate prevents
    cost programs from guessing call counts from informal broadcast metadata.
    """

    kind: str
    source: TransferLayout
    destination: TransferLayout
    temporal_axis: str | None = None
    executed_at: tuple[ReuseWindow, ...] = ()
    resident_across: tuple[ReuseWindow, ...] = ()
    parameters: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in _TRANSFER_ROUTE_KINDS:
            raise ValueError(f"unsupported transfer route kind {self.kind!r}")
        if not isinstance(self.source, TransferLayout) or not isinstance(
            self.destination, TransferLayout
        ):
            raise TypeError("route endpoints must be TransferLayout values")
        executed = tuple(self.executed_at)
        resident = tuple(self.resident_across)
        if any(not isinstance(item, ReuseWindow) for item in executed + resident):
            raise TypeError("route windows must contain ReuseWindow values")
        if len({item.axis for item in executed}) != len(executed):
            raise ValueError("executed_at axes must be unique")
        if len({item.axis for item in resident}) != len(resident):
            raise ValueError("resident_across axes must be unique")
        all_extents = {**self.source.axis_extents, **self.destination.axis_extents}
        unknown = {
            item.axis
            for item in executed + resident
            if item.axis not in all_extents and item.extent is None
        }
        if unknown:
            raise ValueError(f"route windows reference unknown axes: {sorted(unknown)}")
        parameters = MappingProxyType(dict(self.parameters))
        object.__setattr__(self, "executed_at", executed)
        object.__setattr__(self, "resident_across", resident)
        object.__setattr__(self, "parameters", parameters)

    @property
    def replication_axes_added(self) -> tuple[str, ...]:
        return tuple(
            axis
            for axis in self.destination.replica_axes
            if axis not in self.source.replica_axes
        )

    @staticmethod
    def _elements_per_call(
        layout: TransferLayout, executed_at: tuple[ReuseWindow, ...]
    ) -> int:
        tiles = {item.axis: item.tile_extent for item in executed_at}
        return math.prod(
            max(
                min(extent, tiles.get(axis, extent)),
                layout.padded_tile_extents.get(axis, 0),
            )
            for axis, extent in layout.axis_extents.items()
        )

    def metrics(self) -> TransferRouteMetrics:
        source = self.source.metrics().image_size
        destination = self.destination.metrics().image_size
        extents = {**self.source.axis_extents, **self.destination.axis_extents}
        for item in self.executed_at + self.resident_across:
            if item.extent is not None:
                extents[item.axis] = item.extent
        calls = math.prod(
            (extents[item.axis] + item.tile_extent - 1) // item.tile_extent
            for item in self.executed_at
        )
        source_per_call = self._elements_per_call(self.source, self.executed_at)
        destination_per_call = self._elements_per_call(
            self.destination, self.executed_at
        )
        resident_reuse = math.prod(
            (extents[item.axis] + item.tile_extent - 1) // item.tile_extent
            for item in self.resident_across
        )
        return TransferRouteMetrics(
            source_elements=source,
            destination_elements=destination,
            source_elements_per_call=source_per_call,
            destination_elements_per_call=destination_per_call,
            call_count=calls,
            expansion_factor=destination_per_call / source_per_call,
            resident_reuse_factor=resident_reuse,
        )

    def manifest(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "source": self.source.manifest(),
            "destination": self.destination.manifest(),
            "temporal_axis": self.temporal_axis,
            "executed_at": _manifest_value(self.executed_at),
            "resident_across": _manifest_value(self.resident_across),
            "parameters": _manifest_value(self.parameters),
            "metrics": _manifest_value(self.metrics()),
        }


@dataclass(frozen=True)
class Transfer:
    value: str
    direction: str
    coalesced: bool = False
    broadcast: bool = False
    source: str | None = None
    destination: str | None = None
    temporal_axis: str | None = None
    route: tuple[TransferRouteStep, ...] = ()

    def __post_init__(self):
        if not self.value.isidentifier() or self.direction not in _TRANSFER_DIRECTIONS:
            raise ValueError(f"unsupported transfer {self.value!r}/{self.direction!r}")
        route = tuple(self.route)
        if any(not isinstance(step, TransferRouteStep) for step in route):
            raise TypeError("transfer route must contain TransferRouteStep values")
        for previous, following in zip(route, route[1:]):
            if previous.destination is not following.source and (
                previous.destination.manifest() != following.source.manifest()
            ):
                raise ValueError("transfer route layouts must form a continuous chain")
        route_broadcast = any(
            step.kind in {"lookup", "duplicate_subgroup"} for step in route
        )
        route_coalesced = any(step.kind.startswith("dma_") for step in route)
        broadcast = route_broadcast if route else bool(self.broadcast)
        coalesced = route_coalesced if route else bool(self.coalesced)
        if broadcast and self.direction in {
            "out",
            "store",
            "vr_to_l4",
            "l4_to_host",
        }:
            raise ValueError("broadcast is only valid for ingress/read transfers")
        if route and self.source is None:
            object.__setattr__(self, "source", route[0].source.storage)
        if route and self.destination is None:
            object.__setattr__(self, "destination", route[-1].destination.storage)
        object.__setattr__(self, "coalesced", coalesced)
        object.__setattr__(self, "broadcast", broadcast)
        object.__setattr__(self, "route", route)

    @property
    def source_layout(self) -> TransferLayout | None:
        return self.route[0].source if self.route else None

    @property
    def transit_layouts(self) -> tuple[TransferLayout, ...]:
        return tuple(step.destination for step in self.route[:-1])

    @property
    def transit_layout(self) -> TransferLayout | None:
        layouts = self.transit_layouts
        return layouts[0] if layouts else None

    @property
    def destination_layout(self) -> TransferLayout | None:
        return self.route[-1].destination if self.route else None


TransferStep = Transfer


@dataclass(frozen=True)
class ReductionStrategy:
    axis: str
    kind: str = "group_tree"
    group_size: int | None = None
    identity: object = 0
    partial: bool = False

    def __post_init__(self):
        aliases = {"spatial": "group_tree", "temporal": "temporal_accumulate"}
        kind = aliases.get(self.kind, self.kind)
        if kind not in {
            "none",
            "serial",
            "group_tree",
            "pairwise",
            "host",
            "partial",
            "temporal_accumulate",
        }:
            raise ValueError(f"unsupported reduction kind {kind!r}")
        object.__setattr__(self, "kind", kind)
        if kind != "none" and not self.axis.isidentifier():
            raise ValueError("reduction axis must be an identifier")
        if self.group_size is not None:
            size = int(self.group_size)
            if size <= 0 or size & (size - 1):
                raise ValueError("reduction group_size must be a positive power of two")
            object.__setattr__(self, "group_size", size)
        object.__setattr__(self, "partial", bool(self.partial))


APUReduction = ReductionStrategy


@dataclass(frozen=True)
class TemporalStrategy:
    axes: tuple[TemporalAxis, ...] = ()
    barrier: str = "phase"

    def __post_init__(self):
        axes = tuple(self.axes)
        if any(not isinstance(axis, TemporalAxis) for axis in axes):
            raise TypeError("TemporalStrategy axes must contain TemporalAxis values")
        if not self.barrier:
            raise ValueError("temporal barrier must be non-empty")
        object.__setattr__(self, "axes", axes)


@dataclass(frozen=True)
class OutputTilePlacement:
    """Where one compute-output tile lands in the dense output VR image."""

    work_tile: int
    physical_output_batch: int
    lane_offset: int
    logical_origin: tuple[int, ...]
    logical_shape: tuple[int, ...]

    def __post_init__(self):
        if any(
            int(value) < 0
            for value in (self.work_tile, self.physical_output_batch, self.lane_offset)
        ):
            raise ValueError("output placement indices must be non-negative")
        origin = tuple(int(value) for value in self.logical_origin)
        shape = tuple(int(value) for value in self.logical_shape)
        if (
            not origin
            or len(origin) != len(shape)
            or any(value <= 0 for value in shape)
        ):
            raise ValueError("output placement needs equal-rank origin and shape")
        object.__setattr__(self, "work_tile", int(self.work_tile))
        object.__setattr__(
            self, "physical_output_batch", int(self.physical_output_batch)
        )
        object.__setattr__(self, "lane_offset", int(self.lane_offset))
        object.__setattr__(self, "logical_origin", origin)
        object.__setattr__(self, "logical_shape", shape)


@dataclass(frozen=True)
class OutputBatching:
    """Exact relation between compute work tiles and physical output VRs."""

    output_axes: tuple[str, ...]
    axis_extents: tuple[tuple[str, int], ...] | Mapping[str, int]
    work_tile_extents: tuple[tuple[str, int], ...] | Mapping[str, int]
    physical_tile_extents: tuple[tuple[str, int], ...] | Mapping[str, int]
    reduction_axis: str
    reduction_extent: int
    reduction_tile_extent: int
    placements: tuple[OutputTilePlacement, ...]

    def __post_init__(self):
        axes = tuple(self.output_axes)
        extents = _pairs(self.axis_extents, label="output")
        work = _pairs(self.work_tile_extents, label="work tile")
        physical = _pairs(self.physical_tile_extents, label="physical tile")
        if tuple(name for name, _ in extents) != axes:
            raise ValueError("output batching extents must follow output_axes")
        if (
            tuple(name for name, _ in work) != axes
            or tuple(name for name, _ in physical) != axes
        ):
            raise ValueError("output batching tile axes must follow output_axes")
        reduction_extent = int(self.reduction_extent)
        reduction_tile = int(self.reduction_tile_extent)
        if (
            not self.reduction_axis.isidentifier()
            or reduction_extent <= 0
            or reduction_tile <= 0
        ):
            raise ValueError("output batching reduction extents must be positive")
        placements = tuple(sorted(self.placements, key=lambda item: item.work_tile))
        work_sizes = dict(work)
        expected = math.prod(
            (extent + work_sizes[axis] - 1) // work_sizes[axis]
            for axis, extent in extents
        )
        if len(placements) != expected:
            raise ValueError(
                f"output batching needs {expected} work placements, got {len(placements)}"
            )
        if sorted(item.work_tile for item in placements) != list(range(expected)):
            raise ValueError("output work tiles must be numbered densely")
        physical_sizes = dict(physical)
        physical_batches = math.prod(
            (extent + physical_sizes[axis] - 1) // physical_sizes[axis]
            for axis, extent in extents
        )
        if any(item.physical_output_batch >= physical_batches for item in placements):
            raise ValueError("placement exceeds physical output batch count")
        object.__setattr__(self, "output_axes", axes)
        object.__setattr__(self, "axis_extents", extents)
        object.__setattr__(self, "work_tile_extents", work)
        object.__setattr__(self, "physical_tile_extents", physical)
        object.__setattr__(self, "reduction_extent", reduction_extent)
        object.__setattr__(self, "reduction_tile_extent", reduction_tile)
        object.__setattr__(self, "placements", placements)

    @property
    def work_output_tiles(self) -> int:
        return len(self.placements)

    @property
    def physical_output_batches(self) -> int:
        extents = dict(self.axis_extents)
        return math.prod(
            (extents[axis] + tile - 1) // tile
            for axis, tile in self.physical_tile_extents
        )

    @property
    def reduction_tiles(self) -> int:
        return (
            self.reduction_extent + self.reduction_tile_extent - 1
        ) // self.reduction_tile_extent

    @property
    def vector_batches(self) -> int:
        return self.work_output_tiles * self.reduction_tiles

    @property
    def work_tile_counts(self) -> tuple[int, ...]:
        count = Counter(item.physical_output_batch for item in self.placements)
        return tuple(count[index] for index in range(self.physical_output_batches))

    @property
    def uniform_work_tiles_per_output_batch(self) -> int | None:
        sizes = set(self.work_tile_counts)
        return next(iter(sizes)) if len(sizes) == 1 else None

    @property
    def work_tiles_per_output_batch(self) -> int:
        """Maximum work-tile capacity of a physical output batch."""

        return max(self.work_tile_counts)

    @property
    def work_steps_per_output_batch(self) -> int:
        return self.work_tiles_per_output_batch * self.reduction_tiles

    def placement(self, work_tile: int) -> OutputTilePlacement:
        return self.placements[int(work_tile)]

    def manifest(self) -> dict[str, object]:
        return {
            "output_axes": list(self.output_axes),
            "axis_extents": dict(self.axis_extents),
            "work_tile_extents": dict(self.work_tile_extents),
            "physical_tile_extents": dict(self.physical_tile_extents),
            "reduction_axis": self.reduction_axis,
            "reduction_extent": self.reduction_extent,
            "reduction_tile_extent": self.reduction_tile_extent,
            "work_output_tiles": self.work_output_tiles,
            "physical_output_batches": self.physical_output_batches,
            "reduction_tiles": self.reduction_tiles,
            "vector_batches": self.vector_batches,
            "work_tiles_per_output_batch": self.work_tiles_per_output_batch,
            "work_tile_counts": list(self.work_tile_counts),
            "work_steps_per_output_batch": self.work_steps_per_output_batch,
            "placements": _manifest_value(self.placements),
        }


@dataclass(frozen=True)
class VRAllocation:
    abstract: str
    concrete: int | str
    value: str | None = None
    live_range: tuple[int, int] = (0, 1)
    lane_offset: int = 0
    lane_extent: int = 32768

    def __post_init__(self):
        if not self.abstract.isidentifier():
            raise ValueError("abstract VR name must be an identifier")
        concrete = self.concrete
        if isinstance(concrete, str):
            prefix = "GVML_VR16_"
            if not concrete.startswith(prefix) or not concrete[len(prefix) :].isdigit():
                raise ValueError(f"invalid concrete VR {concrete!r}")
            concrete = int(concrete[len(prefix) :])
        concrete = int(concrete)
        if not 0 <= concrete < 15:
            raise ValueError("APU v1 has concrete VR indices 0..14")
        start, stop = (int(value) for value in self.live_range)
        if start < 0 or stop <= start:
            raise ValueError("VR live_range must be a non-empty half-open interval")
        if int(self.lane_offset) < 0 or int(self.lane_extent) <= 0:
            raise ValueError("VR lane allocation must be positive")
        if int(self.lane_offset) + int(self.lane_extent) > 32768:
            raise ValueError("VR lane allocation exceeds 32K lanes")
        object.__setattr__(self, "concrete", concrete)
        object.__setattr__(self, "live_range", (start, stop))
        object.__setattr__(self, "lane_offset", int(self.lane_offset))
        object.__setattr__(self, "lane_extent", int(self.lane_extent))

    @property
    def concrete_name(self) -> str:
        return f"GVML_VR16_{self.concrete}"


@dataclass(frozen=True)
class PlanOperation:
    name: str
    count: int
    dtype: str = "f16"
    loop_roles: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.name or int(self.count) < 0:
            raise ValueError("plan operation needs a name and non-negative count")
        object.__setattr__(self, "count", int(self.count))
        object.__setattr__(self, "loop_roles", tuple(self.loop_roles))


def _manifest_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _manifest_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_manifest_value(item) for item in value]
    manifest = getattr(value, "manifest", None)
    if callable(manifest):
        return _manifest_value(manifest())
    if isinstance(value, LinearLayout):
        return value.manifest()
    if is_dataclass(value):
        return _manifest_value(asdict(value))
    return repr(value)


@dataclass(frozen=True)
class APUV1Plan:
    name: str
    iteration_layout: IterationLayout
    value_layouts: tuple[ValueLayout, ...]
    transfers: tuple[Transfer, ...]
    temporal_strategy: TemporalStrategy | str
    reduction_strategy: ReductionStrategy | str | None
    operations: tuple[PlanOperation, ...] = ()
    vr_allocations: tuple[VRAllocation, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)
    output_batching: OutputBatching | None = None

    def __post_init__(self):
        if not self.name.isidentifier():
            raise ValueError("plan name must be a C identifier")
        if not isinstance(self.iteration_layout, IterationLayout):
            raise TypeError("iteration_layout must be an IterationLayout")
        values = tuple(self.value_layouts)
        transfers = tuple(self.transfers)
        operations = tuple(self.operations)
        allocations = tuple(self.vr_allocations)
        if any(not isinstance(value, ValueLayout) for value in values):
            raise TypeError("value_layouts must contain ValueLayout values")
        if any(not isinstance(transfer, Transfer) for transfer in transfers):
            raise TypeError("transfers must contain Transfer values")
        if any(not isinstance(operation, PlanOperation) for operation in operations):
            raise TypeError("operations must contain PlanOperation values")
        if any(not isinstance(allocation, VRAllocation) for allocation in allocations):
            raise TypeError("vr_allocations must contain VRAllocation values")
        if self.output_batching is not None and not isinstance(
            self.output_batching, OutputBatching
        ):
            raise TypeError("output_batching must be an OutputBatching value")
        names = [value.value for value in values]
        if len(names) != len(set(names)):
            raise ValueError("value layout names must be unique")
        unknown_transfers = {step.value for step in transfers} - set(names)
        if unknown_transfers:
            raise ValueError(
                f"transfers reference unknown values: {sorted(unknown_transfers)}"
            )

        iteration_inputs = set(_layout_parts(self.iteration_layout.layout)[0])
        for value in values:
            layout = value.effective_layout(self.iteration_layout)
            inputs = set(_layout_parts(layout)[0])
            missing = set(value.axes + value.replica_axes) - inputs
            if missing:
                raise ValueError(
                    f"value {value.value!r} axes absent from storage layout: {sorted(missing)}"
                )
        if isinstance(self.temporal_strategy, str):
            if not self.temporal_strategy:
                raise ValueError("temporal_strategy string must be non-empty")
        elif not isinstance(self.temporal_strategy, TemporalStrategy):
            raise TypeError("temporal_strategy must be a string or TemporalStrategy")
        reduction = self.reduction_strategy
        if isinstance(reduction, str):
            reduction = ReductionStrategy(
                axis="" if reduction == "none" else self.iteration_layout.axes[0],
                kind=reduction,
            )
        if reduction is not None and not isinstance(reduction, ReductionStrategy):
            raise TypeError("reduction_strategy must be a string or ReductionStrategy")
        if (
            reduction is not None
            and reduction.kind != "none"
            and reduction.axis not in iteration_inputs
        ):
            raise ValueError(
                f"reduction axis {reduction.axis!r} is absent from iteration layout"
            )

        abstract = [allocation.abstract for allocation in allocations]
        if len(abstract) != len(set(abstract)):
            raise ValueError("abstract VR allocations must be unique")
        for index, lhs in enumerate(allocations):
            if lhs.value is not None and lhs.value not in names:
                raise ValueError(
                    f"VR allocation references unknown value {lhs.value!r}"
                )
            for rhs in allocations[index + 1 :]:
                overlap = max(lhs.live_range[0], rhs.live_range[0]) < min(
                    lhs.live_range[1], rhs.live_range[1]
                )
                if lhs.concrete == rhs.concrete and overlap:
                    raise ValueError(
                        f"concrete {lhs.concrete_name} is assigned to overlapping live ranges"
                    )

        object.__setattr__(self, "value_layouts", values)
        object.__setattr__(self, "transfers", transfers)
        object.__setattr__(self, "operations", operations)
        object.__setattr__(self, "vr_allocations", allocations)
        object.__setattr__(self, "reduction_strategy", reduction)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def value_layout(self, name: str) -> ValueLayout:
        for value in self.value_layouts:
            if value.value == name:
                return value
        raise KeyError(name)

    def value_metrics(self, name: str) -> ValueMetrics:
        return self.value_layout(name).metrics(self.iteration_layout)

    @property
    def temporal_extent(self) -> int:
        if isinstance(self.temporal_strategy, TemporalStrategy):
            return math.prod(axis.extent for axis in self.temporal_strategy.axes)
        return math.prod(axis.extent for axis in self.iteration_layout.temporal_axes)

    def manifest(self) -> dict[str, object]:
        return {
            "name": self.name,
            "iteration_layout": _manifest_value(self.iteration_layout),
            "value_layouts": _manifest_value(self.value_layouts),
            "transfers": _manifest_value(self.transfers),
            "temporal_strategy": _manifest_value(self.temporal_strategy),
            "reduction_strategy": _manifest_value(self.reduction_strategy),
            "operations": _manifest_value(self.operations),
            "vr_allocations": _manifest_value(self.vr_allocations),
            "metadata": _manifest_value(self.metadata),
            "output_batching": _manifest_value(self.output_batching),
        }


__all__ = [
    "APUReduction",
    "APUV1Plan",
    "AffineTiledLayout",
    "IterationLayout",
    "LayoutMetrics",
    "OutputBatching",
    "OutputTilePlacement",
    "PlanOperation",
    "ReductionStrategy",
    "TemporalAxis",
    "TemporalStrategy",
    "Transfer",
    "TransferLayout",
    "TransferRouteMetrics",
    "TransferRouteStep",
    "TransferStep",
    "UnsupportedLayoutError",
    "VRAllocation",
    "Validity",
    "ValueLayout",
    "ValueMetrics",
    "ReuseWindow",
    "derive_layout_metrics",
]
