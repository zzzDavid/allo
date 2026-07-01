# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Executable cost specifications.

A cost spec is ordinary Python code.  Binding it to a structural target runs
the builder and registers rules against target handle identity.  Executing a
rule emits cycle steps over concrete instances derived from the SPMW work
coordinate and the target unit tree.
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
from dataclasses import dataclass, field
from types import MappingProxyType

from .evaluator import Evaluator
from .graph import Activity, ExecutionGraph, HandleInstance, Occupancy


def _unit_path(unit):
    names = []
    while unit is not None:
        names.append(unit.name)
        unit = unit.parent
    return "/".join(reversed(names))


def handle_path(handle):
    """Return the stable structural path for a target handle."""
    from ..spmw_target import MemoryRef, Unit

    if isinstance(handle, Unit):
        return _unit_path(handle)
    if isinstance(handle, MemoryRef):
        return f"{handle_path(handle.memory)}[{handle.idx!r}]"
    owner = getattr(handle, "owner", None)
    name = getattr(handle, "name", None)
    if owner is None or not name:
        raise TypeError(f"{handle!r} is not a target unit/op/move/memory/register")
    return f"{_unit_path(owner)}/{name}"


def _handle_owner(handle):
    from ..spmw_target import MemoryRef, Unit

    if isinstance(handle, Unit):
        return handle
    if isinstance(handle, MemoryRef):
        return handle.memory.owner
    owner = getattr(handle, "owner", None)
    if owner is None:
        raise TypeError(f"{handle!r} has no target-tree owner")
    return owner


def _scope_coordinates(owner, work_id):
    units = []
    unit = owner
    while unit is not None and unit.parent is not None:
        if unit.mode not in ("device", "host") and unit.mapping:
            units.append(unit)
        unit = unit.parent
    units.reverse()
    values = tuple(work_id[: len(units)])
    if len(values) < len(units):
        values += (0,) * (len(units) - len(values))
    coordinates = []
    for coordinate, unit in zip(values, units):
        extent = 1
        for factor in unit.mapping:
            extent *= factor
        if coordinate < 0 or coordinate >= extent:
            raise ValueError(
                f"work coordinate {coordinate} is outside {unit.name!r} "
                f"extent {extent}"
            )
        axis_name = next(iter(unit.axes), unit.name)
        coordinates.append((axis_name, coordinate))
    return tuple(coordinates)


def _evaluate_index(value, work_id):
    """Substitute a work coordinate into a target ``SymExpr`` index."""
    from ..spmw_target import SymExpr, UnitId

    if isinstance(value, UnitId):
        if value.level >= len(work_id):
            raise ValueError(f"work coordinate {work_id!r} has no axis {value.level}")
        return int(work_id[value.level])
    if not isinstance(value, SymExpr):
        return value
    lhs = _evaluate_index(value.args[0], work_id)
    rhs = _evaluate_index(value.args[1], work_id)
    operations = {
        "add": lambda: lhs + rhs,
        "sub": lambda: lhs - rhs,
        "mul": lambda: lhs * rhs,
        "floordiv": lambda: lhs // rhs,
        "mod": lambda: lhs % rhs,
    }
    try:
        return operations[value.op]()
    except KeyError as exc:
        raise ValueError(f"unsupported target index expression {value.op!r}") from exc


def concrete_instance(handle, work_id):
    """Bind a structural target handle to one spatial instance."""
    from ..spmw_target import Memory, MemoryRef, Move, Op, Register, Unit

    owner = _handle_owner(handle)
    kind = "handle"
    for cls, label in (
        (Unit, "unit"),
        (Op, "op"),
        (Move, "move"),
        (MemoryRef, "memory"),
        (Memory, "memory"),
        (Register, "register"),
    ):
        if isinstance(handle, cls):
            kind = label
            break
    path = handle_path(handle)
    if isinstance(handle, MemoryRef):
        index = _evaluate_index(handle.idx, tuple(work_id))
        path = f"{handle_path(handle.memory)}[{index!r}]"
        capacity = handle.memory.capacity
    else:
        capacity = int(getattr(handle, "capacity", 1))
    return HandleInstance(
        path=path,
        coordinates=_scope_coordinates(owner, tuple(work_id)),
        capacity=capacity,
        kind=kind,
    )


@dataclass(frozen=True)
class CostEvent:
    """One target-bound operation or move presented to a cost rule."""

    id: str
    primitive: object
    work_id: tuple[int, ...] = ()
    metrics: MappingProxyType = field(default_factory=lambda: MappingProxyType({}))
    attributes: MappingProxyType = field(default_factory=lambda: MappingProxyType({}))

    @classmethod
    def create(cls, event_id, primitive, work_id=(), metrics=None, attributes=None):
        return cls(
            id=event_id,
            primitive=primitive,
            work_id=tuple(work_id),
            metrics=MappingProxyType(dict(metrics or {})),
            attributes=MappingProxyType(dict(attributes or {})),
        )

    def __getattr__(self, name):
        try:
            return self.metrics[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def instance(self, handle):
        return concrete_instance(handle, self.work_id)

    @property
    def operation_instance(self):
        return self.instance(self.primitive)


@dataclass(frozen=True)
class CostUse:
    """A cost-program request to occupy a structural or concrete handle."""

    handle: object
    cycles: int | None = None
    amount: int = 1


class CostContext:
    """Builder used while one cost rule emits activities."""

    def __init__(self, graph, event, dependencies):
        self.graph = graph
        self.event = event
        self._current = tuple(dependencies)
        self._counter = 0
        self._repeat = [1]
        self._parallel = []

    def use(self, handle, *, cycles=None, amount=1):
        return CostUse(handle, cycles, amount)

    def step(self, *, cycles=None, latency=None, occupy=(), name="step"):
        """Emit one cycle step; program order is sequential by default."""
        if latency is None:
            latency = cycles
        if latency is None:
            raise TypeError("cost.step requires cycles= or latency=")
        multiplier = self._repeat[-1]
        latency = int(latency) * multiplier
        if latency < 0:
            raise ValueError("cost step latency must be non-negative")

        occupancies = []
        for use in occupy:
            if not isinstance(use, CostUse):
                use = CostUse(use)
            handle = (
                use.handle
                if isinstance(use.handle, HandleInstance)
                else self.event.instance(use.handle)
            )
            duration = latency if use.cycles is None else int(use.cycles) * multiplier
            occupancies.append(Occupancy(handle, duration, use.amount))

        activity_id = f"{self.event.id}:cost:{self._counter}:{name}"
        self._counter += 1
        dependencies = self._parallel[-1]["entry"] if self._parallel else self._current
        self.graph.add(
            Activity(
                id=activity_id,
                primitive=handle_path(self.event.primitive),
                latency_cycles=latency,
                occupancy=tuple(occupancies),
                depends_on=tuple(dependencies),
                label=name,
                metadata={
                    **dict(self.event.attributes),
                    "work_id": self.event.work_id,
                },
            )
        )
        if self._parallel:
            self._parallel[-1]["terminals"].append(activity_id)
        else:
            self._current = (activity_id,)
        return activity_id

    @contextlib.contextmanager
    def repeat(self, count):
        """Summarize a sequentially repeated cost region without unrolling it."""
        count = int(count)
        if count < 0:
            raise ValueError("cost.repeat count must be non-negative")
        self._repeat.append(self._repeat[-1] * count)
        try:
            yield self
        finally:
            self._repeat.pop()

    @contextlib.contextmanager
    def parallel(self):
        """Emit enclosed steps from one common dependency frontier."""
        state = {"entry": self._current, "terminals": []}
        self._parallel.append(state)
        try:
            yield self
        finally:
            self._parallel.pop()
            self._current = tuple(state["terminals"]) or tuple(state["entry"])

    @property
    def terminals(self):
        return self._current


_BIND_STACK = []


class BoundCostSpec:
    """A CostSpec whose rules are bound to one concrete target tree."""

    def __init__(self, spec, target):
        self.spec = spec
        self.target = target
        self.rules = {}
        _BIND_STACK.append(self)
        try:
            spec.builder(target)
        finally:
            popped = _BIND_STACK.pop()
            assert popped is self
        if not self.rules:
            raise ValueError(f"cost spec {spec.name!r} registered no rules")

    @property
    def fingerprint(self):
        source = inspect.getsource(self.spec.builder)
        rules = "".join(
            inspect.getsource(function)
            for _path, function in sorted(self.rules.items())
        )
        return hashlib.sha256((source + rules).encode()).hexdigest()[:16]

    def add_rule(self, handle, function):
        path = handle_path(handle)
        if path in self.rules:
            raise ValueError(f"duplicate cost rule for {path!r}")
        self.rules[path] = function

    def emit(self, graph, event, dependencies=()):
        path = handle_path(event.primitive)
        try:
            function = self.rules[path]
        except KeyError as exc:
            raise KeyError(
                f"cost spec {self.spec.name!r} implements no rule for {path!r}"
            ) from exc
        context = CostContext(graph, event, dependencies)
        function(event, context)
        if not context.terminals:
            raise ValueError(f"cost rule for {path!r} emitted no steps")
        return context.terminals

    def evaluate(self, graph):
        graph.metadata["cost_fingerprint"] = self.fingerprint
        return Evaluator().evaluate(graph)


class CostSpec:
    """Reusable executable cost program."""

    def __init__(self, builder, *, target=None, name=None):
        self.builder = builder
        self.target_name = target
        self.name = name or builder.__name__

    def bind(self, target):
        target_name = getattr(target, "name", None)
        if self.target_name is not None and self.target_name != target_name:
            raise ValueError(
                f"cost spec {self.name!r} targets {self.target_name!r}, "
                f"not {target_name!r}"
            )
        return BoundCostSpec(self, target)

    def __repr__(self):
        return f"CostSpec({self.name!r}, target={self.target_name!r})"


def cost(function=None, *, target=None, name=None):
    """Decorate an executable cost-spec builder."""

    def decorate(builder):
        return CostSpec(builder, target=target, name=name)

    if function is None:
        return decorate
    return decorate(function)


def rule(handle):
    """Register a handle-specific rule while a CostSpec is binding."""
    if not _BIND_STACK:
        raise RuntimeError("@allo.rule must be declared inside an @allo.cost builder")

    def decorate(function):
        _BIND_STACK[-1].add_rule(handle, function)
        return function

    return decorate
