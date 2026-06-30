# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Candidate-specific performance execution graph."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from .resources import ResourceRequest


@dataclass(frozen=True)
class Invocation:
    """Typed analytical inputs for one activity.

    Models consume numeric values from ``metrics``. Shapes, dtypes, and access
    summaries remain attached for diagnostics and calibration-domain checks.
    """

    operand_shapes: tuple[tuple[int, ...], ...] = ()
    dtypes: tuple[str, ...] = ()
    metrics: Mapping[str, int | float] = field(default_factory=dict)
    attributes: Mapping[str, object] = field(default_factory=dict)

    def model_inputs(self) -> dict[str, int | float]:
        out = dict(self.metrics)
        out.setdefault("operand_count", len(self.operand_shapes))
        return out


@dataclass(frozen=True)
class Activity:
    """One schedulable operation in an ``ExecutionGraph``."""

    id: str
    primitive: str
    timing_model: str
    invocation: Invocation = field(default_factory=Invocation)
    resources: tuple[ResourceRequest, ...] = ()
    depends_on: tuple[str, ...] = ()
    label: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            raise ValueError("activity id must be non-empty")
        if not self.primitive:
            raise ValueError(f"activity {self.id!r}: primitive must be non-empty")
        if not self.timing_model:
            raise ValueError(f"activity {self.id!r}: timing_model must be non-empty")
        if self.id in self.depends_on:
            raise ValueError(f"activity {self.id!r} cannot depend on itself")


class ExecutionGraph:
    """A dependency DAG for one concrete autoscheduler candidate."""

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
        """Return stable insertion-order topological order and validate edges."""

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

        ready = [name for name, degree in indegree.items() if degree == 0]
        ready.sort(key=position.__getitem__)
        ordered: list[Activity] = []
        while ready:
            name = ready.pop(0)
            ordered.append(self._activities[name])
            for succ in successors[name]:
                indegree[succ] -= 1
                if indegree[succ] == 0:
                    ready.append(succ)
                    ready.sort(key=position.__getitem__)
        if len(ordered) != len(self._activities):
            cyclic = sorted(name for name, degree in indegree.items() if degree)
            raise ValueError(f"execution graph contains a cycle involving {cyclic}")
        return tuple(ordered)

    def terminal_ids(self) -> tuple[str, ...]:
        depended_on = {
            dep for activity in self._activities.values() for dep in activity.depends_on
        }
        return tuple(name for name in self._activities if name not in depended_on)
