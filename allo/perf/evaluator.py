# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic calendar evaluator for lowered cost programs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from .graph import ExecutionGraph, HandleInstance


@dataclass(frozen=True)
class ActivitySpan:
    activity_id: str
    start_cycle: int
    end_cycle: int
    latency_cycles: int
    handle_occupancy_cycles: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Estimate:
    cycles: int
    spans: Mapping[str, ActivitySpan]
    critical_path: tuple[str, ...]
    utilization: Mapping[str, float]
    bottlenecks: tuple[str, ...]
    model_fingerprint: str


@dataclass(frozen=True)
class _Reservation:
    start: int
    end: int
    amount: int
    activity_id: str


class _Calendar:
    def __init__(self, capacity):
        self.capacity = capacity
        self.reservations: list[_Reservation] = []

    def fits(self, start, duration, amount):
        if duration <= 0:
            return True
        end = start + duration
        points = {start}
        for reservation in self.reservations:
            if reservation.end <= start or reservation.start >= end:
                continue
            points.add(max(start, reservation.start))
            points.add(min(end, reservation.end))
        for point in sorted(points):
            if point >= end:
                continue
            usage = sum(
                reservation.amount
                for reservation in self.reservations
                if reservation.start <= point < reservation.end
            )
            if usage + amount > self.capacity:
                return False
        return True

    def earliest(self, start, duration, amount):
        candidate = start
        blocker = None
        while not self.fits(candidate, duration, amount):
            overlapping = [
                reservation
                for reservation in self.reservations
                if reservation.end > candidate
                and reservation.start < candidate + duration
            ]
            if not overlapping:
                raise RuntimeError("handle calendar failed to make progress")
            next_end = min(
                reservation.end
                for reservation in overlapping
                if reservation.end > candidate
            )
            blocker = next(
                reservation.activity_id
                for reservation in reversed(overlapping)
                if reservation.end == next_end
            )
            candidate = next_end
        return candidate, blocker

    def reserve(self, start, duration, amount, activity_id):
        if duration <= 0:
            return
        if not self.fits(start, duration, amount):
            raise RuntimeError("attempted to overbook a target handle")
        self.reservations.append(
            _Reservation(start, start + duration, amount, activity_id)
        )


class Evaluator:
    """ASAP scheduler over dependencies and concrete target-handle calendars."""

    def evaluate(self, graph: ExecutionGraph) -> Estimate:
        calendars: dict[HandleInstance, _Calendar] = {}
        spans = {}
        predecessor = {}

        for activity in graph.topological_order():
            dep_end = 0
            dep_predecessor = None
            for dependency in activity.depends_on:
                end = spans[dependency].end_cycle
                if end >= dep_end:
                    dep_end = end
                    dep_predecessor = dependency

            start = dep_end
            handle_predecessor = None
            while True:
                moved = False
                for use in activity.occupancy:
                    calendar = calendars.setdefault(
                        use.handle, _Calendar(use.handle.capacity)
                    )
                    available, blocker = calendar.earliest(
                        start, use.cycles, use.amount
                    )
                    if available > start:
                        start = available
                        handle_predecessor = blocker
                        moved = True
                if not moved:
                    break

            occupancy = {}
            for use in activity.occupancy:
                calendar = calendars.setdefault(
                    use.handle, _Calendar(use.handle.capacity)
                )
                calendar.reserve(start, use.cycles, use.amount, activity.id)
                occupancy[use.handle.name] = (
                    occupancy.get(use.handle.name, 0) + use.cycles * use.amount
                )

            end = start + activity.latency_cycles
            spans[activity.id] = ActivitySpan(
                activity.id, start, end, activity.latency_cycles, occupancy
            )
            predecessor[activity.id] = (
                handle_predecessor if start > dep_end else dep_predecessor
            )

        cycles = max((span.end_cycle for span in spans.values()), default=0)
        critical = self._critical_path(spans, predecessor)
        utilization = self._utilization(calendars, cycles)
        bottlenecks = tuple(
            name
            for name, _value in sorted(
                utilization.items(), key=lambda item: (-item[1], item[0])
            )[:5]
        )
        return Estimate(
            cycles=cycles,
            spans=spans,
            critical_path=critical,
            utilization=utilization,
            bottlenecks=bottlenecks,
            model_fingerprint=str(graph.metadata.get("cost_fingerprint", "")),
        )

    @staticmethod
    def _critical_path(spans, predecessor):
        if not spans:
            return ()
        current = max(spans, key=lambda name: (spans[name].end_cycle, name))
        path = []
        seen = set()
        while current is not None and current not in seen:
            path.append(current)
            seen.add(current)
            current = predecessor[current]
        path.reverse()
        return tuple(path)

    @staticmethod
    def _utilization(calendars, cycles):
        if cycles <= 0:
            return {handle.name: 0.0 for handle in calendars}
        return {
            handle.name: min(
                1.0,
                sum(
                    (reservation.end - reservation.start) * reservation.amount
                    for reservation in calendar.reservations
                )
                / (cycles * calendar.capacity),
            )
            for handle, calendar in calendars.items()
        }
