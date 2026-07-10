# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public hardware callables for direct Gemini-II VL64 programs.

This is a deliberately small vertical backend.  ``APUG2Program`` describes a
lowering owned by Tenon, and :func:`allo.compile` turns it into a callable that
builds and executes an ARC ``gsi::g2_64vl`` task.  No functional simulator is
used by the device path.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from enum import Enum
from numbers import Integral

import numpy as np

from ..perf import BoundCostSpec, ExecutionGraph
from ..spmw_codegen import RunResult
from .apu_g2_layout import (
    APUG2ReductionPlan,
    APUG2_U16_SHAPE,
    build_apu_g2_u16_layout,
)
from .costs.apu_g2 import (
    build_apu_g2_add_graph,
    build_apu_g2_gemv_graph,
    build_apu_g2_gesummv_graph,
)


class APUG2Operation(str, Enum):
    """Typed semantic operation selected by an explicit direct program."""

    ADD_U16 = "add_u16"
    GEMV_U16 = "gemv_u16"
    GESUMMV_U16 = "gesummv_u16"

    def __str__(self):
        return self.value


def _positive_uint32(name: str, value) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"APUg2 {name} must be an integer")
    value = int(value)
    if value <= 0 or value > np.iinfo(np.uint32).max:
        raise ValueError(f"APUg2 {name} must fit a nonzero uint32")
    return value


def _uint16_scalar(name: str, value) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"APUg2 {name} must be an integer")
    value = int(value)
    if not 0 <= value <= np.iinfo(np.uint16).max:
        raise ValueError(f"APUg2 {name} must fit uint16")
    return value


@dataclass(frozen=True)
class APUG2Program:
    """A direct-VL64 uint16 kernel owned by the APUg2 backend.

    ``gesummv_u16`` implements the modular specialization
    ``y = alpha*(A@x) + beta*(B@x)``.  Each stored reduction and the final
    result are reduced modulo 2**16.
    """

    operation: APUG2Operation | str = APUG2Operation.ADD_U16
    shape: tuple[int, int] | None = None
    alpha: int = 5
    beta: int = 4
    repetitions: int = 256
    name: str | None = None

    def __post_init__(self):
        try:
            operation = APUG2Operation(self.operation)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "APUg2 operation must be one of "
                + ", ".join(item.value for item in APUG2Operation)
            ) from error
        object.__setattr__(self, "operation", operation)
        repetitions = _positive_uint32("profiling repetitions", self.repetitions)
        object.__setattr__(self, "repetitions", repetitions)

        if self.operation is APUG2Operation.ADD_U16:
            if self.shape is not None:
                raise ValueError("add_u16 owns the fixed physical shape (4, 65536)")
            default_name = "apu_g2_u16_add"
        else:
            if not isinstance(self.shape, tuple) or len(self.shape) != 2:
                raise TypeError(
                    f"{self.operation} shape must be a (rows, reduction) tuple"
                )
            rows = _positive_uint32("row extent", self.shape[0])
            reduction = _positive_uint32("reduction extent", self.shape[1])
            if self.operation is APUG2Operation.GEMV_U16:
                plan = APUG2ReductionPlan(rows, reduction, stream_extent=1)
                if plan.log_block_size > 8:
                    raise ValueError(
                        "gemv_u16 padded reduction extent must not exceed 256"
                    )
            object.__setattr__(self, "shape", (rows, reduction))
            if self.operation is APUG2Operation.GESUMMV_U16:
                object.__setattr__(self, "alpha", _uint16_scalar("alpha", self.alpha))
                object.__setattr__(self, "beta", _uint16_scalar("beta", self.beta))
                default_name = "apu_g2_u16_gesummv"
            else:
                default_name = "apu_g2_u16_gemv"

        if self.name is None:
            object.__setattr__(self, "name", default_name)
        elif not isinstance(self.name, str) or not self.name:
            raise TypeError("APUg2 program name must be a nonempty string")

    def build(self):
        return self


def build_apu_g2_execution_graph(
    program: APUG2Program, target, cost: BoundCostSpec
) -> ExecutionGraph:
    """Build the target-bound VL64 cost graph for ``program``."""

    if not isinstance(program, APUG2Program):
        raise TypeError("program must be an APUG2Program")
    if getattr(target, "name", None) != "apu_v2":
        raise ValueError("APUg2 programs require the apu_v2 target key")
    if not isinstance(cost, BoundCostSpec) or cost.target is not target:
        raise TypeError("APUg2 execution graph requires a cost bound to its target")

    if program.operation is APUG2Operation.ADD_U16:
        layout = build_apu_g2_u16_layout()
        graph = build_apu_g2_add_graph(target, cost)
        layout_manifest = layout.manifest()
        vl64_calls = 1
    elif program.operation is APUG2Operation.GESUMMV_U16:
        rows, reduction = program.shape
        plan = APUG2ReductionPlan(rows, reduction, stream_extent=2)
        graph = build_apu_g2_gesummv_graph(
            target,
            cost,
            output_extent=rows,
            reduction_extent=reduction,
        )
        layout_manifest = plan.layout.manifest()
        vl64_calls = graph.metadata["vl64_compute_calls"]
    else:
        rows, reduction = program.shape
        plan = APUG2ReductionPlan(rows, reduction, stream_extent=1)
        graph = build_apu_g2_gemv_graph(
            target,
            cost,
            output_extent=rows,
            reduction_extent=reduction,
        )
        layout_manifest = plan.layout.manifest()
        vl64_calls = graph.metadata["vl64_compute_calls"]

    graph.metadata.update(
        {
            "target": "apu_v2",
            "hardware": "GSI Gemini-II",
            "program": program.operation.value,
            "work_grid": (16,),
            "coalesced_groups": 16,
            "vl64_calls": vl64_calls,
            "layout": layout_manifest,
            "cost": cost.spec.name,
            "cost_fingerprint": cost.fingerprint,
            "analytical": True,
            "execution": "direct_vl64",
            "dtype": "uint16",
        }
    )
    return graph


class APUG2Callable:
    """NumPy-callable hardware-only uint16 VL64 program."""

    def __init__(self, program, target, *, cost, backend=None):
        if not isinstance(program, APUG2Program):
            raise TypeError("program must be an APUG2Program")
        if getattr(target, "name", None) != "apu_v2":
            raise ValueError("APUG2Program requires build_apu_g2_target()")
        if backend not in (None, "device", "virtual"):
            raise ValueError("APUg2 supports only hardware device or virtual cost")
        if cost is None:
            raise ValueError("APUg2 requires its executable cost spec")
        if not isinstance(cost, BoundCostSpec) or cost.target is not target:
            raise TypeError("APUg2 requires a cost bound to the same target")
        self.program = program
        self.target = target
        self.cost = cost
        self.backend = "device" if backend is None else backend
        if program.operation is APUG2Operation.ADD_U16:
            self.layout = build_apu_g2_u16_layout()
        else:
            streams = 2 if program.operation is APUG2Operation.GESUMMV_U16 else 1
            self.layout = APUG2ReductionPlan(*program.shape, stream_extent=streams)
        self.execution_graph = build_apu_g2_execution_graph(program, target, cost)
        self.estimate_result = cost.evaluate(self.execution_graph)
        self.last_result = None
        self.__name__ = program.name
        if program.operation is APUG2Operation.ADD_U16:
            names = ("lhs", "rhs")
        elif program.operation is APUG2Operation.GESUMMV_U16:
            names = ("A", "B", "x")
        else:
            names = ("A", "x", "accumulator")
        self.__signature__ = inspect.Signature(
            parameters=tuple(
                inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for name in names
            )
            + (
                inspect.Parameter(
                    "out", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None
                ),
            )
        )

    def estimate(self):
        return self.estimate_result

    @staticmethod
    def _array(value, name, shape):
        array = np.asarray(value)
        if array.dtype != np.uint16:
            raise TypeError(f"APUg2 operand {name!r} must have dtype uint16")
        if array.shape != shape:
            raise ValueError(
                f"APUg2 operand {name!r} must have shape {shape}, got {array.shape}"
            )
        return np.ascontiguousarray(array)

    @staticmethod
    def _output(value, shape):
        if value is None:
            return
        if not isinstance(value, np.ndarray) or value.dtype != np.uint16:
            raise TypeError("APUg2 output must be a NumPy uint16 array")
        if value.shape != shape:
            raise ValueError(f"APUg2 output must have shape {shape}")

    def _virtual_result(self):
        return RunResult(
            cycles=int(self.estimate_result.cycles),
            stdout="virtual APUg2 cost evaluation; hardware was not executed",
            backend="virtual",
            extra={
                "outputs": {},
                "layout": self.execution_graph.metadata["layout"],
                "vl64_calls": self.execution_graph.metadata["vl64_calls"],
            },
        )

    def __call__(self, *args, **kwargs):
        out = kwargs.pop("out", None)
        if kwargs:
            unknown = next(iter(kwargs))
            raise TypeError(f"unexpected keyword argument {unknown!r}")
        expected_args = 2 if self.program.operation is APUG2Operation.ADD_U16 else 3
        if len(args) == expected_args + 1:
            if out is not None:
                raise TypeError("output was supplied both positionally and by keyword")
            args, out = args[:-1], args[-1]
        if len(args) != expected_args:
            raise TypeError(
                f"{self.program.operation} expects {expected_args} inputs and optional out"
            )

        if self.program.operation is APUG2Operation.ADD_U16:
            lhs = self._array(args[0], "lhs", APUG2_U16_SHAPE)
            rhs = self._array(args[1], "rhs", APUG2_U16_SHAPE)
            self._output(out, APUG2_U16_SHAPE)
            if self.backend == "virtual":
                result = self._virtual_result()
            else:
                from .apu_g2_runtime import run_apu_g2_u16_add

                result = run_apu_g2_u16_add(
                    lhs, rhs, repetitions=self.program.repetitions
                )
        elif self.program.operation is APUG2Operation.GESUMMV_U16:
            rows, reduction = self.program.shape
            matrix_a = self._array(args[0], "A", (rows, reduction))
            matrix_b = self._array(args[1], "B", (rows, reduction))
            vector = self._array(args[2], "x", (reduction,))
            self._output(out, (rows,))
            if self.backend == "virtual":
                result = self._virtual_result()
            else:
                from .apu_g2_gesummv_runtime import run_apu_g2_u16_gesummv

                result = run_apu_g2_u16_gesummv(
                    matrix_a,
                    matrix_b,
                    vector,
                    alpha=self.program.alpha,
                    beta=self.program.beta,
                    repetitions=self.program.repetitions,
                )
        else:
            rows, reduction = self.program.shape
            matrix = self._array(args[0], "A", (rows, reduction))
            vector = self._array(args[1], "x", (reduction,))
            accumulator = self._array(args[2], "accumulator", (rows,))
            self._output(out, (rows,))
            if self.backend == "virtual":
                result = self._virtual_result()
            else:
                from .apu_g2_gemv_runtime import run_apu_g2_u16_gemv

                result = run_apu_g2_u16_gemv(
                    matrix,
                    vector,
                    accumulator,
                    repetitions=self.program.repetitions,
                )

        if out is not None and self.backend != "virtual":
            np.copyto(out, np.asarray(result.extra["outputs"]["out"], dtype=np.uint16))
        self.last_result = result
        return result

    run = __call__


def compile_apu_g2_program(program, target, *, cost, backend=None):
    if program.operation is APUG2Operation.GESUMMV_U16:
        rows, reduction = program.shape
        try:
            plan = APUG2ReductionPlan(rows, reduction, stream_extent=2)
            direct = plan.log_block_size <= 8
        except ValueError:
            direct = False
        if not direct:
            from .apu_g2_vector_program import APUG2ChunkedGesummvCallable

            return APUG2ChunkedGesummvCallable(
                program, target, cost=cost, backend=backend
            )
    return APUG2Callable(program, target, cost=cost, backend=backend)


__all__ = [
    "APUG2Callable",
    "APUG2Operation",
    "APUG2Program",
    "build_apu_g2_execution_graph",
    "compile_apu_g2_program",
]
