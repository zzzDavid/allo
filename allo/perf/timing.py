# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analytical operation timing models expressed exclusively in cycles."""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping
import itertools
import math
import operator
from dataclasses import dataclass

from .calibration import CalibrationProfile


def _ceil_div(a, b):
    if b == 0:
        raise ZeroDivisionError("ceil_div divisor is zero")
    return math.ceil(a / b)


_FUNCTIONS = {
    "ceil_div": _ceil_div,
    "ceil": math.ceil,
    "floor": math.floor,
    "min": min,
    "max": max,
    "abs": abs,
}

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}

_UNARY_OPERATORS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_ALLOWED_NODES = (
    ast.Expression,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.USub,
    ast.UAdd,
)


class CycleExpression:
    """Validated, serializable arithmetic expression."""

    def __init__(self, text: str | int | float):
        self.text = str(text)
        tree = ast.parse(self.text, mode="eval")
        for node in ast.walk(tree):
            if not isinstance(node, _ALLOWED_NODES):
                raise ValueError(
                    f"unsupported node {type(node).__name__} in cycle expression "
                    f"{self.text!r}"
                )
            if isinstance(node, ast.Call):
                if (
                    not isinstance(node.func, ast.Name)
                    or node.func.id not in _FUNCTIONS
                ):
                    raise ValueError(
                        f"unsupported function in cycle expression {self.text!r}"
                    )
                if node.keywords:
                    raise ValueError(
                        "cycle-expression functions do not accept keywords"
                    )
        self._tree = tree
        self.names = frozenset(
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id not in _FUNCTIONS
        )

    def evaluate(self, values: Mapping[str, int | float]) -> int:
        missing = self.names - values.keys()
        if missing:
            raise KeyError(
                f"cycle expression {self.text!r} is missing values for {sorted(missing)}"
            )
        value = self._evaluate_node(self._tree.body, values)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"cycle expression {self.text!r} returned {value!r}")
        return int(math.ceil(value))

    def _evaluate_node(self, node, values):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in _FUNCTIONS:
                return _FUNCTIONS[node.id]
            return values[node.id]
        if isinstance(node, ast.BinOp):
            return _BINARY_OPERATORS[type(node.op)](
                self._evaluate_node(node.left, values),
                self._evaluate_node(node.right, values),
            )
        if isinstance(node, ast.UnaryOp):
            return _UNARY_OPERATORS[type(node.op)](
                self._evaluate_node(node.operand, values)
            )
        if isinstance(node, ast.Call):
            function = self._evaluate_node(node.func, values)
            return function(
                *[self._evaluate_node(argument, values) for argument in node.args]
            )
        raise TypeError(f"unsupported expression node {type(node).__name__}")


@dataclass(frozen=True)
class CycleTiming:
    latency_cycles: int
    initiation_interval_cycles: int = 0
    cycle_interval: tuple[int, int] | None = None
    initiation_interval_cycle_interval: tuple[int, int] | None = None

    def __post_init__(self):
        if self.latency_cycles < 0:
            raise ValueError("latency_cycles must be non-negative")
        if self.initiation_interval_cycles < 0:
            raise ValueError("initiation_interval_cycles must be non-negative")
        if self.cycle_interval is not None:
            lo, hi = self.cycle_interval
            if lo < 0 or lo > self.latency_cycles or hi < self.latency_cycles:
                raise ValueError(
                    "cycle_interval must enclose latency_cycles and be non-negative"
                )
        if self.initiation_interval_cycle_interval is not None:
            lo, hi = self.initiation_interval_cycle_interval
            if (
                lo < 0
                or lo > self.initiation_interval_cycles
                or hi < self.initiation_interval_cycles
            ):
                raise ValueError(
                    "initiation_interval_cycle_interval must enclose the II"
                )


class TimingModel:
    """An analytical latency/throughput formula with named parameters."""

    def __init__(
        self,
        name: str,
        *,
        latency_cycles: str | int | float,
        initiation_interval_cycles: str | int | float = 0,
        inputs: Iterable[str] = (),
        parameters: Iterable[str] = (),
        description: str = "",
        valid_when=None,
    ):
        if not name:
            raise ValueError("timing model name must be non-empty")
        self.name = name
        self.latency = CycleExpression(latency_cycles)
        self.initiation_interval = CycleExpression(initiation_interval_cycles)
        self.inputs = frozenset(inputs)
        self.parameters = frozenset(parameters)
        self.description = description
        self.valid_when = valid_when
        used = self.latency.names | self.initiation_interval.names
        undeclared = used - self.inputs - self.parameters
        if undeclared:
            raise ValueError(
                f"timing model {name!r} uses undeclared names {sorted(undeclared)}"
            )

    def evaluate(
        self,
        inputs: Mapping[str, int | float],
        profile: CalibrationProfile,
    ) -> CycleTiming:
        missing_inputs = self.inputs - inputs.keys()
        if missing_inputs:
            raise KeyError(
                f"timing model {self.name!r} missing inputs {sorted(missing_inputs)}"
            )
        if self.valid_when is not None and not self.valid_when(inputs):
            raise ValueError(
                f"invocation is outside timing model {self.name!r}'s validity domain"
            )

        nominal = dict(inputs)
        parameter_ranges = []
        for name in self.parameters:
            parameter = profile.get(name)
            nominal[name] = parameter.value
            lower = parameter.lower if parameter.lower is not None else parameter.value
            upper = parameter.upper if parameter.upper is not None else parameter.value
            parameter_ranges.append((name, (lower, upper)))

        latency = self.latency.evaluate(nominal)
        ii = self.initiation_interval.evaluate(nominal)
        latency_bounds = [latency]
        ii_bounds = [ii]
        # Evaluate the corners of the (typically tiny) parameter box. This is
        # correct for the affine/ceil/min/max formulas accepted by the DSL and
        # avoids the false assumption that every formula is monotone in the
        # same direction for every parameter.
        for choices in itertools.product(
            *[bounds for _name, bounds in parameter_ranges]
        ):
            corner = dict(inputs)
            for (name, _bounds), value in zip(parameter_ranges, choices):
                corner[name] = value
            latency_bounds.append(self.latency.evaluate(corner))
            ii_bounds.append(self.initiation_interval.evaluate(corner))
        latency_interval = (min(latency_bounds), max(latency_bounds))
        ii_interval = (min(ii_bounds), max(ii_bounds))
        if latency_interval == (latency, latency):
            latency_interval = None
        if ii_interval == (ii, ii):
            ii_interval = None
        return CycleTiming(latency, ii, latency_interval, ii_interval)


class TimingLibrary:
    """Closed vocabulary of timing models for a virtual target."""

    def __init__(self, models: Iterable[TimingModel] = ()):
        self._models: dict[str, TimingModel] = {}
        for model in models:
            self.add(model)

    def add(self, model: TimingModel) -> TimingModel:
        if model.name in self._models:
            raise ValueError(f"duplicate timing model {model.name!r}")
        self._models[model.name] = model
        return model

    def get(self, name: str) -> TimingModel:
        try:
            return self._models[name]
        except KeyError as exc:
            raise KeyError(
                f"unknown timing model {name!r}; declared: {sorted(self._models)}"
            ) from exc

    def evaluate(
        self,
        name: str,
        inputs: Mapping[str, int | float],
        profile: CalibrationProfile,
    ) -> CycleTiming:
        return self.get(name).evaluate(inputs, profile)

    @property
    def models(self) -> Mapping[str, TimingModel]:
        return dict(self._models)
