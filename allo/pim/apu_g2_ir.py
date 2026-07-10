# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable compiler manifests for multi-region APUg2 programs.

This module is an additive compiler boundary.  It converts the existing
target-neutral retained-MLIR contraction analyses into a small program graph;
it does not choose a runtime, mutate compiler dispatch, or recognize workload
names.  Producer-consumer links are recovered solely from value definitions
and later operand uses.  Consequently a linked pair such as ``A @ x`` followed
by ``A.T @ tmp`` is distinct from two contractions that merely share an input
matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .contraction_analysis import (
    ContractionAnalysis,
    ValueAccess,
    analyze_contractions,
)


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values))


@dataclass(frozen=True)
class APUG2UseDefManifest:
    """One region operand use and its nearest preceding definition, if any."""

    value: str
    consumer_region: str
    operand_role: str
    indices: tuple[str, ...]
    producer_region: str | None = None

    def __post_init__(self):
        if not self.value or not self.consumer_region:
            raise ValueError("APUg2 use-def records require value and consumer names")
        if self.operand_role not in {"lhs", "rhs", "accumulator"}:
            raise ValueError(f"unknown APUG2 operand role {self.operand_role!r}")
        object.__setattr__(self, "indices", tuple(self.indices))

    @property
    def is_dependency(self) -> bool:
        return self.producer_region is not None

    def manifest(self) -> dict[str, object]:
        return {
            "value": self.value,
            "producer_region": self.producer_region,
            "consumer_region": self.consumer_region,
            "operand_role": self.operand_role,
            "indices": list(self.indices),
        }


@dataclass(frozen=True)
class APUG2ValueManifest:
    """A logical memref value and its definition/use sites in program order."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    definition_regions: tuple[str, ...] = ()
    use_regions: tuple[str, ...] = ()
    entry_use_regions: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.name or not self.dtype:
            raise ValueError("APUg2 values require nonempty names and dtypes")
        shape = tuple(int(extent) for extent in self.shape)
        if any(extent <= 0 for extent in shape):
            raise ValueError(f"APUg2 value {self.name!r} has invalid shape {shape}")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "definition_regions", _unique(self.definition_regions))
        object.__setattr__(self, "use_regions", _unique(self.use_regions))
        object.__setattr__(self, "entry_use_regions", _unique(self.entry_use_regions))

    def manifest(self) -> dict[str, object]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "definition_regions": list(self.definition_regions),
            "use_regions": list(self.use_regions),
            "entry_use_regions": list(self.entry_use_regions),
        }


@dataclass(frozen=True)
class APUG2RegionManifest:
    """One retained-MLIR compute region in the APUg2 program graph."""

    id: str
    function: str
    kind: str
    input_values: tuple[str, ...]
    output_values: tuple[str, ...]
    axes: tuple[tuple[str, int], ...]
    output_axes: tuple[str, ...]
    reduction_axes: tuple[str, ...]
    numeric_type: str
    operation: str

    def __post_init__(self):
        if not self.id or not self.function:
            raise ValueError("APUg2 regions require nonempty ids and functions")
        if self.kind != "contraction":
            raise ValueError(f"unsupported APUG2 region kind {self.kind!r}")
        axes = tuple((str(name), int(extent)) for name, extent in self.axes)
        if len({name for name, _extent in axes}) != len(axes):
            raise ValueError(f"APUg2 region {self.id!r} has duplicate axes")
        if any(extent <= 0 for _name, extent in axes):
            raise ValueError(f"APUg2 region {self.id!r} has nonpositive axes")
        object.__setattr__(self, "input_values", _unique(self.input_values))
        object.__setattr__(self, "output_values", _unique(self.output_values))
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "output_axes", tuple(self.output_axes))
        object.__setattr__(self, "reduction_axes", tuple(self.reduction_axes))

    def manifest(self) -> dict[str, object]:
        return {
            "id": self.id,
            "function": self.function,
            "kind": self.kind,
            "input_values": list(self.input_values),
            "output_values": list(self.output_values),
            "axes": [{"name": name, "extent": extent} for name, extent in self.axes],
            "output_axes": list(self.output_axes),
            "reduction_axes": list(self.reduction_axes),
            "numeric_type": self.numeric_type,
            "operation": self.operation,
        }


@dataclass(frozen=True)
class APUG2ModuleManifest:
    """Immutable value/region/use-def graph discovered from retained MLIR."""

    values: tuple[APUG2ValueManifest, ...]
    regions: tuple[APUG2RegionManifest, ...]
    use_defs: tuple[APUG2UseDefManifest, ...]

    def __post_init__(self):
        values = tuple(self.values)
        regions = tuple(self.regions)
        use_defs = tuple(self.use_defs)
        if not regions:
            raise ValueError("APUg2 module manifest requires at least one region")
        if any(not isinstance(value, APUG2ValueManifest) for value in values):
            raise TypeError("values must contain APUG2ValueManifest instances")
        if any(not isinstance(region, APUG2RegionManifest) for region in regions):
            raise TypeError("regions must contain APUG2RegionManifest instances")
        if any(not isinstance(edge, APUG2UseDefManifest) for edge in use_defs):
            raise TypeError("use_defs must contain APUG2UseDefManifest instances")
        value_names = [value.name for value in values]
        region_ids = [region.id for region in regions]
        if len(value_names) != len(set(value_names)):
            raise ValueError("APUg2 module value names must be unique")
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("APUg2 module region ids must be unique")
        known_values = set(value_names)
        known_regions = set(region_ids)
        for edge in use_defs:
            if edge.value not in known_values:
                raise ValueError(f"use-def references unknown value {edge.value!r}")
            if edge.consumer_region not in known_regions:
                raise ValueError(
                    f"use-def references unknown consumer {edge.consumer_region!r}"
                )
            if edge.producer_region not in known_regions | {None}:
                raise ValueError(
                    f"use-def references unknown producer {edge.producer_region!r}"
                )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "regions", regions)
        object.__setattr__(self, "use_defs", use_defs)

    def value(self, name: str) -> APUG2ValueManifest:
        try:
            return next(value for value in self.values if value.name == name)
        except StopIteration as error:
            raise KeyError(name) from error

    def region(self, region_id: str) -> APUG2RegionManifest:
        try:
            return next(region for region in self.regions if region.id == region_id)
        except StopIteration as error:
            raise KeyError(region_id) from error

    @property
    def dependencies(self) -> tuple[APUG2UseDefManifest, ...]:
        """Producer-consumer uses, excluding entry/initializer uses."""

        return tuple(edge for edge in self.use_defs if edge.is_dependency)

    @property
    def roots(self) -> tuple[str, ...]:
        consumers = {edge.consumer_region for edge in self.dependencies}
        return tuple(region.id for region in self.regions if region.id not in consumers)

    @property
    def sinks(self) -> tuple[str, ...]:
        producers = {edge.producer_region for edge in self.dependencies}
        return tuple(region.id for region in self.regions if region.id not in producers)

    @property
    def contraction_topology(self) -> str:
        """Return ``single``, ``independent``, ``linked_chain``, or ``dag``."""

        if len(self.regions) == 1:
            return "single"
        unique_edges = {
            (edge.producer_region, edge.consumer_region) for edge in self.dependencies
        }
        if not unique_edges:
            return "independent"
        incoming = {region.id: 0 for region in self.regions}
        outgoing = {region.id: 0 for region in self.regions}
        for producer, consumer in unique_edges:
            outgoing[producer] += 1
            incoming[consumer] += 1
        if (
            len(unique_edges) == len(self.regions) - 1
            and all(degree <= 1 for degree in incoming.values())
            and all(degree <= 1 for degree in outgoing.values())
            and len(self.roots) == 1
            and len(self.sinks) == 1
        ):
            return "linked_chain"
        return "dag"

    def manifest(self) -> dict[str, object]:
        return {
            "kind": "apu-g2-module",
            "contraction_topology": self.contraction_topology,
            "roots": list(self.roots),
            "sinks": list(self.sinks),
            "values": [value.manifest() for value in self.values],
            "regions": [region.manifest() for region in self.regions],
            "use_defs": [edge.manifest() for edge in self.use_defs],
        }


def _region_id(index: int, analysis: ContractionAnalysis) -> str:
    return f"region_{index}_{analysis.function}"


def _region_manifest(index: int, analysis: ContractionAnalysis) -> APUG2RegionManifest:
    return APUG2RegionManifest(
        id=_region_id(index, analysis),
        function=analysis.function,
        kind="contraction",
        input_values=(
            analysis.lhs.value,
            analysis.rhs.value,
            analysis.accumulator.value,
        ),
        output_values=(analysis.output.value,),
        axes=tuple((axis.name, axis.extent) for axis in analysis.axes),
        output_axes=analysis.output_axes,
        reduction_axes=(analysis.reduction_axis,),
        numeric_type=analysis.numeric_type,
        operation=f"{analysis.multiply_operation}+{analysis.combine_operation}",
    )


def _accesses(analysis: ContractionAnalysis):
    return (
        ("lhs", analysis.lhs),
        ("rhs", analysis.rhs),
        ("accumulator", analysis.accumulator),
    )


def discover_apu_g2_module_manifest(module_or_analyses) -> APUG2ModuleManifest:
    """Discover an immutable multi-contraction graph from retained MLIR.

    A tuple of :class:`ContractionAnalysis` objects is also accepted so target
    planners can reuse a parse they already performed.  Definitions become
    visible only after a region's operand uses, faithfully modeling in-place
    accumulator updates and preventing spurious self-dependencies.
    """

    if isinstance(module_or_analyses, tuple) and all(
        isinstance(item, ContractionAnalysis) for item in module_or_analyses
    ):
        analyses = module_or_analyses
    else:
        analyses = analyze_contractions(module_or_analyses)
    if not analyses:
        raise ValueError("APUg2 module discovery requires at least one contraction")

    regions = tuple(
        _region_manifest(index, analysis) for index, analysis in enumerate(analyses)
    )
    latest_definition: dict[str, str] = {}
    use_defs: list[APUG2UseDefManifest] = []
    accesses_by_value: dict[str, list[ValueAccess]] = {}
    definitions_by_value: dict[str, list[str]] = {}

    for region, analysis in zip(regions, analyses):
        for role, access in _accesses(analysis):
            accesses_by_value.setdefault(access.value, []).append(access)
            use_defs.append(
                APUG2UseDefManifest(
                    value=access.value,
                    producer_region=latest_definition.get(access.value),
                    consumer_region=region.id,
                    operand_role=role,
                    indices=access.indices,
                )
            )
        output = analysis.output
        accesses_by_value.setdefault(output.value, []).append(output)
        definitions_by_value.setdefault(output.value, []).append(region.id)
        latest_definition[output.value] = region.id

    values: list[APUG2ValueManifest] = []
    for name, accesses in accesses_by_value.items():
        dtypes = {access.dtype for access in accesses}
        shapes = {tuple(access.shape) for access in accesses}
        if len(dtypes) != 1 or len(shapes) != 1:
            raise ValueError(
                f"retained MLIR gives inconsistent type information for {name!r}: "
                f"dtypes={sorted(dtypes)}, shapes={sorted(shapes)}"
            )
        related_uses = [edge for edge in use_defs if edge.value == name]
        values.append(
            APUG2ValueManifest(
                name=name,
                dtype=next(iter(dtypes)),
                shape=next(iter(shapes)),
                definition_regions=tuple(definitions_by_value.get(name, ())),
                use_regions=tuple(edge.consumer_region for edge in related_uses),
                entry_use_regions=tuple(
                    edge.consumer_region
                    for edge in related_uses
                    if edge.producer_region is None
                ),
            )
        )
    return APUG2ModuleManifest(tuple(values), regions, tuple(use_defs))


__all__ = [
    "APUG2ModuleManifest",
    "APUG2RegionManifest",
    "APUG2UseDefManifest",
    "APUG2ValueManifest",
    "discover_apu_g2_module_manifest",
]
