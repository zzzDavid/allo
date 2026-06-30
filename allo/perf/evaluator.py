# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic resource-constrained cycle estimator."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from .calibration import CalibrationProfile
from .graph import Activity, ExecutionGraph
from .resources import ResourceSpec, ResourceTopology
from .timing import CycleTiming, TimingLibrary


@dataclass(frozen=True)
class ActivitySpan:
    activity_id: str
    start_cycle: int
    end_cycle: int
    latency_cycles: int
    resource_occupancy_cycles: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Estimate:
    cycles: int
    cycle_interval: tuple[int, int]
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
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.reservations: list[_Reservation] = []

    def _usage_at(self, cycle: int) -> int:
        return sum(r.amount for r in self.reservations if r.start <= cycle < r.end)

    def fits(self, start: int, duration: int, amount: int) -> bool:
        if duration <= 0:
            return True
        end = start + duration
        points = {start}
        for reservation in self.reservations:
            if reservation.end <= start or reservation.start >= end:
                continue
            points.add(max(start, reservation.start))
            points.add(min(end, reservation.end))
        return not any(
            point < end and self._usage_at(point) + amount > self.capacity
            for point in sorted(points)
        )

    def earliest(
        self, start: int, duration: int, amount: int
    ) -> tuple[int, str | None]:
        candidate = start
        blocker = None
        while not self.fits(candidate, duration, amount):
            overlapping = [
                r
                for r in self.reservations
                if r.end > candidate and r.start < candidate + duration
            ]
            if not overlapping:
                raise RuntimeError("resource calendar failed to make progress")
            next_end = min(r.end for r in overlapping if r.end > candidate)
            ending = [r for r in overlapping if r.end == next_end]
            blocker = ending[-1].activity_id
            candidate = next_end
        return candidate, blocker

    def reserve(self, start: int, duration: int, amount: int, activity_id: str) -> None:
        if duration <= 0:
            return
        if not self.fits(start, duration, amount):
            raise RuntimeError("attempted to overbook a resource calendar")
        self.reservations.append(
            _Reservation(start, start + duration, amount, activity_id)
        )


class Evaluator:
    """ASAP list scheduler over a dependency DAG and target resource calendars."""

    def evaluate(
        self,
        graph: ExecutionGraph,
        topology: ResourceTopology,
        timings: TimingLibrary,
        profile: CalibrationProfile,
    ) -> Estimate:
        nominal = self._schedule(graph, topology, timings, profile, bound="nominal")
        lower = self._schedule(graph, topology, timings, profile, bound="lower")
        upper = self._schedule(graph, topology, timings, profile, bound="upper")
        cycles, spans, predecessor, calendars = nominal
        lower_cycles = min(lower[0], cycles)
        upper_cycles = max(upper[0], cycles)

        critical = self._critical_path(spans, predecessor)
        utilization = self._utilization(calendars, topology, cycles)
        bottlenecks = tuple(
            name
            for name, _ in sorted(
                utilization.items(), key=lambda item: (-item[1], item[0])
            )[:5]
        )
        return Estimate(
            cycles=cycles,
            cycle_interval=(lower_cycles, upper_cycles),
            spans=spans,
            critical_path=critical,
            utilization=utilization,
            bottlenecks=bottlenecks,
            model_fingerprint=profile.fingerprint(),
        )

    def _timing_for_bound(
        self,
        activity: Activity,
        timings: TimingLibrary,
        profile: CalibrationProfile,
        bound: str,
    ) -> CycleTiming:
        timing = timings.evaluate(
            activity.timing_model, activity.invocation.model_inputs(), profile
        )
        if bound == "nominal" or timing.cycle_interval is None:
            return timing
        latency = timing.cycle_interval[0 if bound == "lower" else 1]
        if timing.initiation_interval_cycle_interval is None:
            ii = timing.initiation_interval_cycles
        else:
            ii = timing.initiation_interval_cycle_interval[0 if bound == "lower" else 1]
        return CycleTiming(latency, max(0, ii))

    def _schedule(self, graph, topology, timings, profile, *, bound):
        calendars: dict[tuple[str, int], _Calendar] = {}
        for spec in topology:
            for instance in range(spec.instances):
                calendars[(spec.name, instance)] = _Calendar(spec.capacity)

        spans: dict[str, ActivitySpan] = {}
        predecessor: dict[str, str | None] = {}
        for activity in graph.topological_order():
            for request in activity.resources:
                topology.validate_request(request)
            timing = self._timing_for_bound(activity, timings, profile, bound)
            dep_end = 0
            dep_predecessor = None
            for dep in activity.depends_on:
                end = spans[dep].end_cycle
                if end >= dep_end:
                    dep_end = end
                    dep_predecessor = dep

            start = dep_end
            resource_predecessor = None
            # Atomic acquisition: moving one request later can invalidate an
            # earlier request, so iterate until every calendar accepts the same
            # start cycle.
            while True:
                moved = False
                for request in activity.resources:
                    spec = topology.get(request.resource)
                    duration = self._occupancy(spec, timing)
                    for instance in request.instances:
                        available, blocker = calendars[(spec.name, instance)].earliest(
                            start, duration, request.amount
                        )
                        if available > start:
                            start = available
                            resource_predecessor = blocker
                            moved = True
                if not moved:
                    break

            occupancy: dict[str, int] = {}
            for request in activity.resources:
                spec = topology.get(request.resource)
                duration = self._occupancy(spec, timing)
                for instance in request.instances:
                    calendars[(spec.name, instance)].reserve(
                        start, duration, request.amount, activity.id
                    )
                    key = f"{spec.name}[{instance}]"
                    occupancy[key] = duration * request.amount

            end = start + timing.latency_cycles
            spans[activity.id] = ActivitySpan(
                activity.id, start, end, timing.latency_cycles, occupancy
            )
            predecessor[activity.id] = (
                resource_predecessor if start > dep_end else dep_predecessor
            )

        cycles = max((span.end_cycle for span in spans.values()), default=0)
        return cycles, spans, predecessor, calendars

    @staticmethod
    def _occupancy(spec: ResourceSpec, timing: CycleTiming) -> int:
        if timing.latency_cycles == 0:
            return 0
        if spec.pipelined:
            return max(1, timing.initiation_interval_cycles)
        return timing.latency_cycles

    @staticmethod
    def _critical_path(spans, predecessor) -> tuple[str, ...]:
        if not spans:
            return ()
        current = max(spans, key=lambda name: (spans[name].end_cycle, name))
        path = []
        seen = set()
        while current is not None and current not in seen:
            path.append(current)
            seen.add(current)
            current = predecessor.get(current)
        path.reverse()
        return tuple(path)

    @staticmethod
    def _utilization(calendars, topology, cycles) -> dict[str, float]:
        if cycles <= 0:
            return {
                f"{spec.name}[{i}]": 0.0
                for spec in topology
                for i in range(spec.instances)
            }
        out = {}
        for spec in topology:
            for instance in range(spec.instances):
                calendar = calendars[(spec.name, instance)]
                occupied = sum(
                    (r.end - r.start) * r.amount for r in calendar.reservations
                )
                out[f"{spec.name}[{instance}]"] = occupied / (cycles * spec.capacity)
        return out
