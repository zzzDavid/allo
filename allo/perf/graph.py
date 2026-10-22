# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lowered cycle program over concrete target-handle instances."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True, order=True)
class HandleInstance:
    """One physical instance of a target unit, op, memory, register, or move."""

    path: str
    coordinates: tuple[object, ...] = ()
    capacity: int = 1
    kind: str = "handle"

    def __post_init__(self):
        if not self.path:
            raise ValueError("handle instance path must be non-empty")
        if self.capacity <= 0:
            raise ValueError("handle instance capacity must be positive")

    @property
    def name(self) -> str:
        if not self.coordinates:
            return self.path
        coords = ",".join(
            f"{axis}={value}" if isinstance(item, tuple) else str(item)
            for item in self.coordinates
            for axis, value in ([item] if isinstance(item, tuple) else [("", item)])
        )
        return f"{self.path}[{coords}]"


@dataclass(frozen=True)
class Occupancy:
    """Capacity held on one concrete handle for a number of cycles."""

    handle: HandleInstance
    cycles: int
    amount: int = 1

    def __post_init__(self):
        if self.cycles < 0:
            raise ValueError("occupancy cycles must be non-negative")
        if self.amount <= 0:
            raise ValueError("occupancy amount must be positive")
        if self.amount > self.handle.capacity:
            raise ValueError(
                f"{self.handle.name} capacity is {self.handle.capacity}; "
                f"requested {self.amount}"
            )


@dataclass(frozen=True)
class Activity:
    """One executable step emitted by a cost-spec rule."""

    id: str
    primitive: str
    latency_cycles: int
    occupancy: tuple[Occupancy, ...] = ()
    depends_on: tuple[str, ...] = ()
    label: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            raise ValueError("activity id must be non-empty")
        if not self.primitive:
            raise ValueError(f"activity {self.id!r}: primitive must be non-empty")
        if self.latency_cycles < 0:
            raise ValueError("activity latency must be non-negative")
        if self.id in self.depends_on:
            raise ValueError(f"activity {self.id!r} cannot depend on itself")


class ExecutionGraph:
    """Dependency DAG produced by executing a CostSpec."""

    def __init__(self, name: str = "", metadata: Mapping[str, object] | None = None):
        self.name = name
        self.metadata = dict(metadata or {})
        self._activities: dict[str, Activity] = {}

    def add(self, activity: Activity) -> Activity:
        if activity.id in self._activities:
            raise ValueError(f"duplicate activity id {activity.id!r}")
        self._activities[activity.id] = activity
        return activity

    def extend(self, activities: Iterable[Activity]) -> None:
        for activity in activities:
            self.add(activity)

    def get(self, activity_id: str) -> Activity:
        try:
            return self._activities[activity_id]
        except KeyError as exc:
            raise KeyError(f"unknown activity {activity_id!r}") from exc

    @property
    def activities(self) -> tuple[Activity, ...]:
        return tuple(self._activities.values())

    def topological_order(self) -> tuple[Activity, ...]:
        position = {name: i for i, name in enumerate(self._activities)}
        indegree = {name: 0 for name in self._activities}
        successors: dict[str, list[str]] = {name: [] for name in self._activities}
        for activity in self._activities.values():
            for dep in activity.depends_on:
                if dep not in self._activities:
                    raise ValueError(
                        f"activity {activity.id!r} depends on unknown activity {dep!r}"
                    )
                indegree[activity.id] += 1
                successors[dep].append(activity.id)

        ready = sorted(
            (name for name, degree in indegree.items() if degree == 0),
            key=position.__getitem__,
        )
        ordered = []
        while ready:
            name = ready.pop(0)
            ordered.append(self._activities[name])
            for successor in successors[name]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    ready.append(successor)
                    ready.sort(key=position.__getitem__)
        if len(ordered) != len(self._activities):
            cyclic = sorted(name for name, degree in indegree.items() if degree)
            raise ValueError(f"execution graph contains a cycle involving {cyclic}")
        return tuple(ordered)
