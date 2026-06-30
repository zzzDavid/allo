# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Target-owned resource topology.

Resources are named, instanced capacity constraints.  They intentionally carry
no cycle values: timing belongs to a ``TimingModel``.  A pipelined resource is
reserved for an activity's initiation interval; a non-pipelined resource is
reserved for its full latency.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ResourceSpec:
    """One homogeneous pool of target resource instances.

    ``instances`` distinguishes physically independent engines. ``capacity``
    permits multiple simultaneous users of one instance. ``pipelined`` changes
    admission occupancy from latency to initiation interval; completion still
    occurs after the activity latency.
    """

    name: str
    instances: int = 1
    capacity: int = 1
    pipelined: bool = False
    parent: str | None = None
    description: str = ""
    attributes: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            raise ValueError("resource name must be non-empty")
        if self.instances <= 0:
            raise ValueError(f"resource {self.name!r}: instances must be positive")
        if self.capacity <= 0:
            raise ValueError(f"resource {self.name!r}: capacity must be positive")


@dataclass(frozen=True)
class ResourceRequest:
    """Capacity requested from concrete instances of a resource pool.

    A gang operation names multiple instances. Acquiring all instances is
    atomic, so a broadcast can reserve a command bus and several execution
    engines at the same start cycle.
    """

    resource: str
    instances: tuple[int, ...] = (0,)
    amount: int = 1

    def __post_init__(self):
        if not self.resource:
            raise ValueError("resource request must name a resource")
        if not self.instances:
            raise ValueError("resource request must select at least one instance")
        if len(set(self.instances)) != len(self.instances):
            raise ValueError("resource request contains duplicate instances")
        if min(self.instances) < 0:
            raise ValueError("resource instance indices must be non-negative")
        if self.amount <= 0:
            raise ValueError("resource request amount must be positive")


class ResourceTopology:
    """Validated collection of resource pools for a target."""

    def __init__(self, resources: Iterable[ResourceSpec] = ()):
        self._resources: dict[str, ResourceSpec] = {}
        for resource in resources:
            self.add(resource)

    def add(self, resource: ResourceSpec) -> ResourceSpec:
        if resource.name in self._resources:
            raise ValueError(f"duplicate resource {resource.name!r}")
        self._resources[resource.name] = resource
        return resource

    def __contains__(self, name: str) -> bool:
        return name in self._resources

    def __iter__(self):
        return iter(self._resources.values())

    def __len__(self) -> int:
        return len(self._resources)

    def get(self, name: str) -> ResourceSpec:
        try:
            return self._resources[name]
        except KeyError as exc:
            raise KeyError(
                f"unknown resource {name!r}; declared: {sorted(self._resources)}"
            ) from exc

    def validate_request(self, request: ResourceRequest) -> None:
        spec = self.get(request.resource)
        invalid = [i for i in request.instances if i >= spec.instances]
        if invalid:
            raise ValueError(
                f"resource {spec.name!r} has {spec.instances} instances; "
                f"requested {invalid}"
            )
        if request.amount > spec.capacity:
            raise ValueError(
                f"resource {spec.name!r} capacity is {spec.capacity}; "
                f"requested {request.amount}"
            )

    @property
    def resources(self) -> Mapping[str, ResourceSpec]:
        return dict(self._resources)
