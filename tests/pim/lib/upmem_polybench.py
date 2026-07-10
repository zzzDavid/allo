# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical SMALL PolyBench registry for the UPMEM MLIR/C path.

Unlike the historical UPMEM leaf workloads, this module describes the complete
programs in :mod:`examples.polybench`.  Every entry resolves the original Allo
kernel dynamically, gives its exact generic instantiation and array ABI, binds
the ambient constants used by the frontend, constructs deterministic inputs,
and invokes the repository's own ``*_np`` reference.

The registry contains descriptions and reference data only.  It deliberately
does not customize, schedule, lower, or execute a backend.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from types import ModuleType
from typing import Callable, Iterable, Mapping

import numpy as np
from allo.ir.types import float32

from .shapes import shape


_DTYPE = np.dtype(np.float32)


@dataclass(frozen=True)
class ArgumentSpec:
    """One array in the canonical Allo function ABI."""

    name: str
    shape: tuple[int, ...]
    intent: str
    dtype: np.dtype = _DTYPE

    def __post_init__(self):
        if self.intent not in {"in", "out", "inout"}:
            raise ValueError(f"invalid intent {self.intent!r} for {self.name}")


@dataclass(frozen=True)
class ResultSpec:
    """One returned or externally observable result."""

    name: str
    shape: tuple[int, ...]
    dtype: np.dtype = _DTYPE


@dataclass(frozen=True)
class ExecutionPhase:
    """One host-visible DPU launch phase.

    ``partition`` describes the logical objects assigned to DPUs, ``parallel``
    names the source loop or derived frontier, and ``barrier`` describes the
    host action required before the next phase.
    """

    name: str
    kind: str
    partition: str
    parallel: str
    barrier: str


@dataclass(frozen=True)
class ExecutionSpec:
    """Declarative phased orchestration used to construct a UPMEM program."""

    phases: tuple[ExecutionPhase, ...]

    @property
    def partition_axes(self) -> tuple[str, ...]:
        return tuple(phase.partition for phase in self.phases)


@dataclass(frozen=True)
class PolyBenchCase:
    """Complete frontend and NumPy contract for one SMALL kernel."""

    name: str
    module_name: str
    kernel_name: str
    reference_name: str
    dimensions: tuple[tuple[str, int], ...]
    instantiate_dimensions: tuple[str, ...]
    arguments: tuple[ArgumentSpec, ...]
    results: tuple[ResultSpec, ...]
    execution: ExecutionSpec
    ambient_bindings: tuple[tuple[str, float | int], ...] = ()

    @property
    def dims(self) -> dict[str, int]:
        return dict(self.dimensions)

    @property
    def scalars(self) -> dict[str, float | int]:
        return dict(self.ambient_bindings)

    @property
    def module(self) -> ModuleType:
        """Dynamically import the canonical example module."""
        return importlib.import_module(f"examples.polybench.{self.module_name}")

    def bind_ambient(self) -> ModuleType:
        """Bind every free scalar before the Allo frontend inspects the kernel."""
        module = self.module
        for name, value in self.ambient_bindings:
            setattr(module, name, value)
        return module

    @property
    def kernel(self) -> Callable:
        module = self.bind_ambient()
        return getattr(module, self.kernel_name)

    @property
    def reference(self) -> Callable:
        return getattr(self.module, self.reference_name)

    @property
    def instantiate(self) -> list[object]:
        """Exact ``allo.customize(..., instantiate=...)`` list at SMALL."""
        dims = self.dims
        return [float32, *(dims[name] for name in self.instantiate_dimensions)]

    def kernel_and_instantiate(self) -> tuple[Callable, list[object]]:
        """Resolve and bind the frontend kernel in one operation."""
        return self.kernel, self.instantiate

    def make_inputs(self, seed: int = 0) -> dict[str, np.ndarray]:
        """Return deterministic, numerically safe arrays in ABI-name order."""
        values = _make_inputs(self, np.random.default_rng(seed))
        expected = tuple(arg.name for arg in self.arguments)
        if tuple(values) != expected:
            raise AssertionError(
                f"{self.name}: generator order {tuple(values)} != ABI {expected}"
            )
        for arg in self.arguments:
            value = values[arg.name]
            if value.shape != arg.shape or value.dtype != arg.dtype:
                raise AssertionError(
                    f"{self.name}.{arg.name}: got {value.shape}/{value.dtype}, "
                    f"expected {arg.shape}/{arg.dtype}"
                )
        return values

    def make_argument_tuple(self, seed: int = 0) -> tuple[np.ndarray, ...]:
        values = self.make_inputs(seed)
        return tuple(values[arg.name] for arg in self.arguments)

    def run_reference(
        self,
        inputs: Mapping[str, np.ndarray] | None = None,
        *,
        seed: int = 0,
    ) -> dict[str, np.ndarray]:
        """Run this module's own ``*_np`` function on private input copies."""
        if inputs is None:
            inputs = self.make_inputs(seed)
        missing = [arg.name for arg in self.arguments if arg.name not in inputs]
        if missing:
            raise KeyError(f"{self.name}: missing reference inputs {missing}")
        work = {
            arg.name: np.array(inputs[arg.name], copy=True) for arg in self.arguments
        }
        outputs = _run_reference(self, work)
        expected = tuple(result.name for result in self.results)
        if tuple(outputs) != expected:
            raise AssertionError(
                f"{self.name}: reference results {tuple(outputs)} != {expected}"
            )
        for result in self.results:
            value = np.asarray(outputs[result.name])
            if value.shape != result.shape:
                raise AssertionError(
                    f"{self.name}.{result.name}: got {value.shape}, expected {result.shape}"
                )
            outputs[result.name] = value.astype(result.dtype, copy=False)
        return outputs


def _arg(name: str, dims: tuple[int, ...], intent: str = "in") -> ArgumentSpec:
    return ArgumentSpec(name, tuple(int(v) for v in dims), intent)


def _result(name: str, dims: tuple[int, ...]) -> ResultSpec:
    return ResultSpec(name, tuple(int(v) for v in dims))


def _adi_bindings(dims: Mapping[str, int]) -> dict[str, float]:
    n, tsteps = dims["N"], dims["TSTEPS"]
    dx = 1.0 / n
    dy = 1.0 / n
    dt = 1.0 / tsteps
    mul1 = 2.0 * dt / (dx * dx)
    mul2 = dt / (dy * dy)
    a = -mul1 / 2.0
    d = -mul2 / 2.0
    return {"a": a, "b": 1.0 + mul1, "c": a, "d": d, "e": 1.0 + mul2, "f": d}


def _deriche_bindings() -> dict[str, float]:
    alpha = 0.25
    exp = math.exp
    k = ((1.0 - exp(-alpha)) ** 2) / (
        1.0 + 2.0 * alpha * exp(-alpha) - exp(2.0 * alpha)
    )
    return {
        "a1": k,
        "a2": k * exp(-alpha) * (alpha - 1.0),
        "a3": k * exp(-alpha) * (alpha + 1.0),
        "a4": -k * exp(-2.0 * alpha),
        "a5": k,
        "a6": k * exp(-alpha) * (alpha - 1.0),
        "a7": k * exp(-alpha) * (alpha + 1.0),
        "a8": -k * exp(-2.0 * alpha),
        "b1": 2.0 ** (-alpha),
        "b2": -exp(-2.0 * alpha),
        "c1": 1.0,
        "c2": 1.0,
    }


def _dims(name: str) -> dict[str, int]:
    return {key: int(value) for key, value in shape(name).items()}


def _p(
    name: str, kind: str, partition: str, parallel: str, barrier: str
) -> ExecutionPhase:
    return ExecutionPhase(name, kind, partition, parallel, barrier)


# Source-loop/frontier metadata from design/08-upmem-polybench-execution-matrix.md.
# Barrier vocabulary is intentionally descriptive rather than tied to one host
# runtime: ``host`` is a launch boundary, while suffixes say which collective is
# required to make the next phase's operands physically available.
_EXECUTION: dict[str, ExecutionSpec] = {
    "2mm": ExecutionSpec(
        (
            _p(
                "mm1",
                "independent",
                "flatten(AB[i0,j0])",
                "mm1.(i0,j0)",
                "host:gather+redistribute",
            ),
            _p("mm2", "independent", "flatten(ABC[i1,j1])", "mm2.(i1,j1)", "host"),
            _p(
                "combine",
                "independent",
                "flatten(output[i2,j2])",
                "ele_add.(i2,j2)",
                "host:gather",
            ),
        )
    ),
    "3mm": ExecutionSpec(
        (
            _p(
                "mm1",
                "independent",
                "flatten(AB[i0,j0])",
                "mm1.(i0,j0)",
                "host:gather+redistribute",
            ),
            _p(
                "mm2",
                "independent",
                "flatten(CD[i1,j1])",
                "mm2.(i1,j1)",
                "host:gather+redistribute",
            ),
            _p(
                "mm3",
                "independent",
                "flatten(output[i2,j2])",
                "mm3.(i2,j2)",
                "host:gather",
            ),
        )
    ),
    "adi": ExecutionSpec(
        (
            _p(
                "column_solve",
                "temporal",
                "independent line i",
                "first-sweep.i; serial j/j_rev",
                "host:redistribute-orientation",
            ),
            _p(
                "row_solve",
                "temporal",
                "independent line i",
                "second-sweep.i; serial j/j_rev",
                "host:timestep",
            ),
        )
    ),
    "atax": ExecutionSpec(
        (
            _p("ax", "independent", "tmp row m", "stage_M.m", "host:gather+broadcast"),
            _p("atx", "independent", "y column n", "stage_N.n", "host:gather"),
        )
    ),
    "bicg": ExecutionSpec(
        (
            _p("q", "independent", "A row i1", "stageQ.i1", "host:gather"),
            _p("s", "independent", "A column j0", "outlined stageS.j0", "host:gather"),
        )
    ),
    "cholesky": ExecutionSpec(
        (
            _p(
                "lower_entry",
                "pivot",
                "dot fragments k<j",
                "partial-reduction.k",
                "host:reduce+broadcast",
            ),
            _p(
                "upper_row",
                "pivot",
                "upper entries j>=i",
                "second-region.j",
                "host:pivot-row",
            ),
            _p(
                "diagonal",
                "pivot",
                "dot fragments k<i",
                "partial-reduction.k",
                "host:reduce+sqrt+broadcast",
            ),
        )
    ),
    "correlation": ExecutionSpec(
        (
            _p(
                "mean",
                "independent",
                "data column x",
                "compute_mean.x",
                "host:broadcast",
            ),
            _p(
                "stddev",
                "independent",
                "data column x",
                "compute_stddev.x",
                "host:broadcast",
            ),
            _p(
                "center",
                "independent",
                "flatten(data[x,y])",
                "center_reduce.(x,y)",
                "host",
            ),
            _p(
                "corr",
                "independent",
                "upper-triangle pair (i,j)",
                "compute_corr.(i,j)",
                "host:gather",
            ),
        )
    ),
    "covariance": ExecutionSpec(
        (
            _p("mean", "independent", "data column x", "mean.x", "host:broadcast"),
            _p(
                "cov",
                "independent",
                "matrix pair (i,j)",
                "covariance.(i,j)",
                "host:gather",
            ),
        )
    ),
    "deriche": ExecutionSpec(
        (
            _p(
                "horizontal",
                "serial-line",
                "image row i",
                "horizontal.i; serial j/j_inv",
                "host:redistribute-orientation",
            ),
            _p(
                "vertical",
                "serial-line",
                "image column j",
                "vertical.j; serial i/i_inv",
                "host:gather",
            ),
        )
    ),
    "doitgen": ExecutionSpec(
        (
            _p(
                "batched_product",
                "independent",
                "flatten(r,q)",
                "kernel_doitgen.(r,q)",
                "host:gather",
            ),
        )
    ),
    "durbin": ExecutionSpec(
        (
            _p(
                "dot",
                "serial",
                "prefix reduction fragments i<k",
                "reduction.i",
                "host:reduce+broadcast",
            ),
            _p(
                "prefix_update",
                "pivot",
                "prefix elements i<k",
                "update.i",
                "host:iteration-k",
            ),
        )
    ),
    "fdtd_2d": ExecutionSpec(
        (
            _p(
                "electric",
                "temporal",
                "flatten field cells (i,j)",
                "ey.(i,j) + ex.(i,j)",
                "host:halo",
            ),
            _p(
                "magnetic",
                "temporal",
                "flatten hz cells (i,j)",
                "hz.(i,j)",
                "host:timestep+halo",
            ),
        )
    ),
    "floyd_warshall": ExecutionSpec(
        (
            _p(
                "k_update",
                "pivot",
                "path rows i / cells (i,j)",
                "(i,j) at fixed k",
                "host:pivot-k+broadcast-row",
            ),
        )
    ),
    "gemm": ExecutionSpec(
        (
            _p(
                "product_combine",
                "independent",
                "flatten(output[i,j])",
                "mm1.(i0,j0) then ele_add.(i2,j2)",
                "host:gather",
            ),
        )
    ),
    "gemver": ExecutionSpec(
        (
            _p("rank2", "independent", "A row/cells (i,j)", "first.(i,j)", "host"),
            _p(
                "aty",
                "reduction",
                "x output i / A column",
                "second.i; reduction j",
                "host:reduce-or-redistribute",
            ),
            _p("add_z", "independent", "x element i", "third.i", "host:broadcast"),
            _p("ax", "independent", "w row i", "fourth.i; reduction j", "host:gather"),
        )
    ),
    "gesummv": ExecutionSpec(
        (
            _p(
                "rows",
                "independent",
                "output row i",
                "compute_tmp.i + compute_y.i0",
                "host:gather",
            ),
        )
    ),
    "gramschmidt": ExecutionSpec(
        (
            _p(
                "norm",
                "serial",
                "row reduction fragments i",
                "norm.i at fixed k",
                "host:reduce+broadcast",
            ),
            _p("normalize", "pivot", "Q row i", "normalize.i", "host"),
            _p(
                "project",
                "pivot",
                "remaining column j",
                "projection.j; reduction i",
                "host:reduce+broadcast",
            ),
            _p(
                "update",
                "pivot",
                "flatten remaining A[i,j]",
                "update.i for each j",
                "host:iteration-k",
            ),
        )
    ),
    "heat_3d": ExecutionSpec(
        (
            _p(
                "fused_cells",
                "wavefront",
                "3-D frontier i+j+k",
                "derived i+j+k; B then A per cell",
                "host:wavefront+halo",
            ),
        )
    ),
    "jacobi_1d": ExecutionSpec(
        (
            _p("B_from_A", "temporal", "interior index i0", "i0", "host:halo"),
            _p("A_from_B", "temporal", "interior index i1", "i1", "host:timestep+halo"),
        )
    ),
    "jacobi_2d": ExecutionSpec(
        (
            _p(
                "B_from_A",
                "temporal",
                "flatten interior (i0,j0)",
                "compute_A.(i0,j0)",
                "host:halo",
            ),
            _p(
                "A_from_B",
                "temporal",
                "flatten interior (i1,j1)",
                "compute_B.(i1,j1)",
                "host:timestep+halo",
            ),
        )
    ),
    "lu": ExecutionSpec(
        (
            _p(
                "lower_prefix",
                "pivot",
                "dot fragments k<j",
                "lower.k; serial j",
                "host:reduce+broadcast",
            ),
            _p("upper_row", "pivot", "upper entry j>=i", "upper.j", "host:pivot-row"),
        )
    ),
    "ludcmp": ExecutionSpec(
        (
            _p(
                "lu_lower",
                "pivot",
                "dot fragments k<j",
                "lower.k; serial j",
                "host:reduce+broadcast",
            ),
            _p("lu_upper", "pivot", "upper entry j>=i", "upper.j", "host:pivot-row"),
            _p(
                "forward",
                "serial",
                "dot fragments j<i",
                "forward.j",
                "host:reduce+broadcast",
            ),
            _p(
                "backward",
                "serial",
                "dot fragments j>i",
                "backward.j at i_inv",
                "host:reduce+divide+broadcast",
            ),
        )
    ),
    "mvt": ExecutionSpec(
        (
            _p("Ay1", "independent", "output row i0", "stageA.i0", "host:gather"),
            _p("ATy2", "independent", "output column i1", "stageB.i1", "host:gather"),
        )
    ),
    "nussinov": ExecutionSpec(
        (
            _p(
                "intervals",
                "wavefront",
                "intervals (i,j) at fixed length",
                "derived interval-length diagonal",
                "host:diagonal",
            ),
        )
    ),
    "seidel_2d": ExecutionSpec(
        (
            _p(
                "sweep",
                "wavefront",
                "skewed frontier 2*i+j",
                "derived 2*i+j",
                "host:wavefront+timestep+halo",
            ),
        )
    ),
    "symm": ExecutionSpec(
        (
            _p(
                "columns",
                "serial-line",
                "output column j",
                "whole algorithm per j; serial i",
                "host:gather",
            ),
        )
    ),
    "syr2k": ExecutionSpec(
        (
            _p(
                "triangle",
                "independent",
                "flatten lower-triangle (i,j)",
                "output pair (i1,j1)",
                "host:gather",
            ),
        )
    ),
    "syrk": ExecutionSpec(
        (
            _p(
                "triangle",
                "independent",
                "flatten lower-triangle (i,j)",
                "output pair (i1,j1)",
                "host:gather",
            ),
        )
    ),
    "trisolv": ExecutionSpec(
        (
            _p(
                "row_dot",
                "serial",
                "dot fragments j<i",
                "reduction.j at fixed i",
                "host:reduce+divide+broadcast",
            ),
        )
    ),
    "trmm": ExecutionSpec(
        (
            _p("columns", "serial-line", "B column j1", "S0.j1; serial i1", "host"),
            _p("scale", "independent", "flatten B[i0,j0]", "S1.(i0,j0)", "host:gather"),
        )
    ),
}


def _case(
    name: str,
    module: str,
    kernel: str,
    reference: str,
    instantiate: Iterable[str],
    arguments: Iterable[ArgumentSpec],
    results: Iterable[ResultSpec],
    bindings: Mapping[str, float | int] | None = None,
) -> PolyBenchCase:
    dims = _dims(name)
    return PolyBenchCase(
        name=name,
        module_name=module,
        kernel_name=kernel,
        reference_name=reference,
        dimensions=tuple(dims.items()),
        instantiate_dimensions=tuple(instantiate),
        arguments=tuple(arguments),
        results=tuple(results),
        execution=_EXECUTION[name],
        ambient_bindings=tuple((bindings or {}).items()),
    )


def _build_registry() -> dict[str, PolyBenchCase]:
    cases: dict[str, PolyBenchCase] = {}

    def add(case: PolyBenchCase):
        if case.name in cases:
            raise KeyError(f"duplicate PolyBench case {case.name}")
        cases[case.name] = case

    d = _dims("2mm")
    P, Q, R, S = d["P"], d["Q"], d["R"], d["S"]
    add(
        _case(
            "2mm",
            "two_mm",
            "kernel_2mm",
            "two_mm_np",
            ("P", "R", "Q", "S"),
            (
                _arg("A", (P, Q)),
                _arg("B", (Q, R)),
                _arg("C", (R, S)),
                _arg("D", (P, S)),
            ),
            (_result("output", (P, S)),),
            {"alpha": 0.1, "beta": 0.5},
        )
    )

    d = _dims("3mm")
    P, Q, R, S, T = d["P"], d["Q"], d["R"], d["S"], d["T"]
    add(
        _case(
            "3mm",
            "three_mm",
            "kernel_3mm",
            "three_mm_np",
            ("P", "Q", "R", "S", "T"),
            (
                _arg("A", (P, Q)),
                _arg("B", (Q, R)),
                _arg("C", (R, S)),
                _arg("D", (S, T)),
            ),
            (_result("output", (P, T)),),
        )
    )

    d = _dims("adi")
    N = d["N"]
    add(
        _case(
            "adi",
            "adi",
            "kernel_adi",
            "adi_np",
            ("TSTEPS", "N"),
            tuple(_arg(x, (N, N), "inout") for x in ("u", "v", "p", "q")),
            tuple(_result(x, (N, N)) for x in ("u", "v", "p", "q")),
            _adi_bindings(d),
        )
    )

    d = _dims("atax")
    M, N = d["M"], d["N"]
    add(
        _case(
            "atax",
            "atax",
            "kernel_atax",
            "atax_np",
            ("M", "N"),
            (_arg("A", (M, N)), _arg("x", (N,)), _arg("y", (N,), "out")),
            (_result("y", (N,)),),
        )
    )

    d = _dims("bicg")
    M, N = d["M"], d["N"]
    add(
        _case(
            "bicg",
            "bicg",
            "kernel_bicg",
            "bicg_np",
            ("M", "N"),
            (
                _arg("A", (N, M)),
                _arg("A_copy", (N, M)),
                _arg("p", (M,)),
                _arg("r", (N,)),
                _arg("q", (N,), "inout"),
                _arg("s", (M,), "inout"),
            ),
            (_result("q", (N,)), _result("s", (M,))),
        )
    )

    d = _dims("cholesky")
    N = d["N"]
    add(
        _case(
            "cholesky",
            "cholesky",
            "kernel_cholesky",
            "cholesky_np",
            ("N",),
            (_arg("A", (N, N), "inout"),),
            (_result("A", (N, N)),),
        )
    )

    d = _dims("correlation")
    M, N = d["M"], d["N"]
    add(
        _case(
            "correlation",
            "correlation",
            "kernel_correlation",
            "correlation_np",
            ("M", "N"),
            (
                _arg("data_mean", (N, M)),
                _arg("data_stddev", (N, M)),
                _arg("data_for_center", (N, M)),
                _arg("corr", (M, M), "out"),
            ),
            (_result("corr", (M, M)),),
            {"N_float": float(N), "epsilon": 1e-5},
        )
    )

    d = _dims("covariance")
    M, N = d["M"], d["N"]
    add(
        _case(
            "covariance",
            "covariance",
            "kernel_covariance",
            "covariance_np",
            ("M", "N"),
            (
                _arg("data", (N, M)),
                _arg("mean", (M,), "out"),
                _arg("cov", (M, M), "out"),
            ),
            (_result("mean", (M,)), _result("cov", (M, M))),
        )
    )

    d = _dims("deriche")
    W, H = d["W"], d["H"]
    add(
        _case(
            "deriche",
            "deriche",
            "kernel_deriche",
            "deriche_np",
            ("W", "H"),
            (
                _arg("imgIn", (W, H)),
                _arg("imgOut", (W, H), "out"),
                _arg("y1", (W, H), "out"),
                _arg("y2", (W, H), "out"),
            ),
            (_result("imgOut", (W, H)), _result("y1", (W, H)), _result("y2", (W, H))),
            _deriche_bindings(),
        )
    )

    d = _dims("doitgen")
    R, Q, P, S = d["R"], d["Q"], d["P"], d["S"]
    add(
        _case(
            "doitgen",
            "doitgen",
            "kernel_doitgen",
            "doitgen_np",
            ("R", "Q", "P", "S"),
            (
                _arg("A", (R, Q, S), "inout"),
                _arg("x", (P, S)),
                _arg("sum_", (P,), "out"),
            ),
            (_result("A", (R, Q, S)), _result("sum_", (P,))),
        )
    )

    d = _dims("durbin")
    N = d["N"]
    add(
        _case(
            "durbin",
            "durbin",
            "kernel_durbin",
            "durbin_np",
            ("N",),
            (_arg("r", (N,)), _arg("y", (N,), "out")),
            (_result("y", (N,)),),
        )
    )

    d = _dims("fdtd_2d")
    X, Y, T = d["Nx"], d["Ny"], d["Tmax"]
    add(
        _case(
            "fdtd_2d",
            "fdtd_2d",
            "kernel_fdtd_2d",
            "fdtd_2d_np",
            ("Nx", "Ny", "Tmax"),
            (
                _arg("ex", (X, Y), "inout"),
                _arg("ey", (X, Y), "inout"),
                _arg("hz", (X, Y), "inout"),
                _arg("fict", (T,)),
            ),
            (_result("ex", (X, Y)), _result("ey", (X, Y)), _result("hz", (X, Y))),
        )
    )

    d = _dims("floyd_warshall")
    N = d["N"]
    add(
        _case(
            "floyd_warshall",
            "floyd_warshall",
            "kernel_floyd_warshall",
            "floyd_warshall_np",
            ("N",),
            (_arg("path", (N, N), "inout"),),
            (_result("path", (N, N)),),
        )
    )

    d = _dims("gemm")
    P, Q, R = d["P"], d["Q"], d["R"]
    add(
        _case(
            "gemm",
            "gemm",
            "kernel_gemm",
            "gemm_np",
            ("P", "Q", "R"),
            (
                _arg("A", (P, Q)),
                _arg("B", (Q, R)),
                _arg("C", (P, R)),
                _arg("output", (P, R), "out"),
            ),
            (_result("output", (P, R)),),
            {"beta": 0.1},
        )
    )

    d = _dims("gemver")
    N = d["N"]
    add(
        _case(
            "gemver",
            "gemver",
            "kernel_gemver",
            "gemver_np",
            ("N",),
            (
                _arg("A", (N, N), "inout"),
                *tuple(_arg(x, (N,)) for x in ("u1", "u2", "v1", "v2")),
                _arg("x", (N,), "inout"),
                _arg("y", (N,)),
                _arg("w", (N,), "inout"),
                _arg("z", (N,)),
            ),
            (_result("A", (N, N)), _result("x", (N,)), _result("w", (N,))),
            {"alpha": 0.1, "beta": 0.1},
        )
    )

    d = _dims("gesummv")
    N = d["N"]
    add(
        _case(
            "gesummv",
            "gesummv",
            "kernel_gesummv",
            "gesummv_np",
            ("N",),
            (
                _arg("A", (N, N)),
                _arg("B", (N, N)),
                _arg("x", (N,)),
                _arg("y", (N,), "out"),
            ),
            (_result("y", (N,)),),
            {"alpha": 0.1, "beta": 0.1},
        )
    )

    d = _dims("gramschmidt")
    M, N = d["M"], d["N"]
    add(
        _case(
            "gramschmidt",
            "gramschmidt",
            "kernel_gramschmidt",
            "gramschmidt_np",
            ("M", "N"),
            (
                _arg("A", (M, N), "inout"),
                _arg("Q", (M, N), "out"),
                _arg("R", (N, N), "out"),
            ),
            (_result("A", (M, N)), _result("Q", (M, N)), _result("R", (N, N))),
        )
    )

    d = _dims("heat_3d")
    N = d["N"]
    add(
        _case(
            "heat_3d",
            "heat_3d",
            "kernel_heat_3d",
            "heat_3d_np",
            ("TSTEPS", "N"),
            (_arg("A", (N, N, N), "inout"), _arg("B", (N, N, N), "inout")),
            (_result("A", (N, N, N)), _result("B", (N, N, N))),
        )
    )

    d = _dims("jacobi_1d")
    N = d["N"]
    add(
        _case(
            "jacobi_1d",
            "jacobi_1d",
            "kernel_jacobi_1d",
            "jacobi_1d_np",
            ("TSTEPS", "N"),
            (_arg("A", (N,), "inout"), _arg("B", (N,), "inout")),
            (_result("A", (N,)), _result("B", (N,))),
        )
    )

    d = _dims("jacobi_2d")
    N = d["N"]
    add(
        _case(
            "jacobi_2d",
            "jacobi_2d",
            "kernel_jacobi_2d",
            "jacobi_2d_np",
            ("N",),
            (_arg("A", (N, N), "inout"), _arg("B", (N, N), "inout")),
            (_result("A", (N, N)), _result("B", (N, N))),
            {"TSTEPS": d["TSTEPS"]},
        )
    )

    d = _dims("lu")
    N = d["N"]
    add(
        _case(
            "lu",
            "lu",
            "kernel_lu",
            "lu_np",
            ("N",),
            (_arg("A", (N, N), "inout"),),
            (_result("A", (N, N)),),
        )
    )

    d = _dims("ludcmp")
    N = d["N"]
    add(
        _case(
            "ludcmp",
            "ludcmp",
            "kernel_ludcmp",
            "ludcmp_np",
            ("N",),
            (
                _arg("A", (N, N), "inout"),
                _arg("b", (N,)),
                _arg("x", (N,), "out"),
                _arg("y", (N,), "out"),
            ),
            (_result("A", (N, N)), _result("x", (N,)), _result("y", (N,))),
        )
    )

    d = _dims("mvt")
    N = d["N"]
    add(
        _case(
            "mvt",
            "mvt",
            "kernel_mvt",
            "mvt_np",
            ("N",),
            (
                _arg("A", (N, N)),
                _arg("A_copy", (N, N)),
                _arg("y1", (N,)),
                _arg("y2", (N,)),
                _arg("x1", (N,)),
                _arg("x2", (N,)),
                _arg("x1_out", (N,), "out"),
                _arg("x2_out", (N,), "out"),
            ),
            (_result("x1_out", (N,)), _result("x2_out", (N,))),
        )
    )

    d = _dims("nussinov")
    N = d["N"]
    add(
        _case(
            "nussinov",
            "nussinov",
            "kernel_nussinov",
            "nussinov_np",
            ("N",),
            (_arg("seq", (N,)), _arg("table", (N, N), "inout")),
            (_result("table", (N, N)),),
        )
    )

    d = _dims("seidel_2d")
    N = d["N"]
    add(
        _case(
            "seidel_2d",
            "seidel_2d",
            "kernel_seidel_2d",
            "seidel_2d_np",
            ("TSTEPS", "N"),
            (_arg("A", (N, N), "inout"),),
            (_result("A", (N, N)),),
        )
    )

    d = _dims("symm")
    M, N = d["M"], d["N"]
    add(
        _case(
            "symm",
            "symm",
            "kernel_symm",
            "symm_np",
            ("M", "N"),
            (
                _arg("A0", (M, M)),
                _arg("A1", (M, M)),
                _arg("B0", (M, N)),
                _arg("B1", (M, N)),
                _arg("C", (M, N), "inout"),
            ),
            (_result("C", (M, N)),),
            {"alpha": 1.5, "beta": 1.2},
        )
    )

    d = _dims("syr2k")
    M, N = d["M"], d["N"]
    add(
        _case(
            "syr2k",
            "syr2k",
            "kernel_syr2k",
            "syr2k_np",
            ("N", "M"),
            (
                _arg("A", (N, M)),
                _arg("A_copy", (N, M)),
                _arg("B", (N, M)),
                _arg("B_copy", (N, M)),
                _arg("Cin", (N, N)),
                _arg("Cout", (N, N), "out"),
            ),
            (_result("Cout", (N, N)),),
            {"alpha": 1.5, "beta": 1.2},
        )
    )

    d = _dims("syrk")
    M, N = d["M"], d["N"]
    add(
        _case(
            "syrk",
            "syrk",
            "kernel_syr2k",
            "syrk_np",
            ("N", "M"),
            (
                _arg("A", (N, M)),
                _arg("A_copy", (N, M)),
                _arg("Cin", (N, N)),
                _arg("Cout", (N, N), "out"),
            ),
            (_result("Cout", (N, N)),),
            {"alpha": 1.5, "beta": 1.2},
        )
    )

    d = _dims("trisolv")
    N = d["N"]
    add(
        _case(
            "trisolv",
            "trisolv",
            "kernel_trisolv",
            "trisolv_np",
            ("N",),
            (_arg("L", (N, N)), _arg("b", (N,)), _arg("x", (N,), "out")),
            (_result("x", (N,)),),
        )
    )

    d = _dims("trmm")
    M, N = d["M"], d["N"]
    add(
        _case(
            "trmm",
            "trmm",
            "kernel_trmm",
            "trmm_np",
            ("M", "N"),
            (_arg("A", (M, M)), _arg("B", (M, N), "inout")),
            (_result("B", (M, N)),),
            {"alpha": 1.5},
        )
    )

    return cases


REGISTRY = _build_registry()
NAMES = tuple(REGISTRY)


_HALO_CASES = {
    "fdtd_2d": (1, 1),
    "heat_3d": (1, 1),
    "jacobi_1d": (1, 1),
    "jacobi_2d": (1, 1),
    "seidel_2d": (1, 1),
}


def _array_declaration(case: PolyBenchCase, argument: ArgumentSpec) -> UPMEMArray:
    from allo.pim.upmem_abi import TensorLayout
    from allo.pim.upmem_program import UPMEMArray

    # Read-only operands are replicated because the canonical whole-program
    # MLIR may consume them in more than one orientation.  Observable outputs
    # use BLOCK ownership.  The conceptual phase manifest records the tighter
    # per-launch redistributions that a device runtime should eventually use.
    if argument.intent == "in":
        return UPMEMArray(argument.name, layout=TensorLayout.BROADCAST)

    halo = _HALO_CASES.get(case.name, (0, 0))
    axis = 0
    if halo == (0, 0):
        # Choose the widest legal tensor axis. The LinearLayout then stripes
        # that axis over DPUs, maximizing rank utilization without flattening
        # or copying the complete tensor.
        axis = max(range(len(argument.shape)), key=argument.shape.__getitem__)
    return UPMEMArray(
        argument.name,
        layout=TensorLayout.BLOCK,
        partition_axis=axis,
        halo=halo,
    )


def execution_manifest(case: PolyBenchCase) -> dict[str, object]:
    """JSON-compatible conceptual host/DPU launch graph for one case."""
    return {
        "kernel": case.name,
        "num_dpus": 64,
        "num_tasklets": 16,
        "phases": [
            {
                "name": phase.name,
                "kind": phase.kind,
                "partition": phase.partition,
                "parallel": phase.parallel,
                "barrier": phase.barrier,
            }
            for phase in case.execution.phases
        ],
        "parallel_region_policy": "retained-mlir-dependence-analysis",
    }


def build_upmem_program(case_or_name: PolyBenchCase | str) -> UPMEMProgram:
    """Build a complete canonical MLIR program and its physical array ABI.

    Only 2mm and 3mm return memrefs from MLIR.  Every other observable result
    aliases a source argument and therefore must not be declared as an extra C
    result.
    """
    from allo.pim.upmem_abi import TensorLayout
    from allo.pim.upmem_program import UPMEMArray, UPMEMPhase, UPMEMProgram

    case = get_case(case_or_name) if isinstance(case_or_name, str) else case_or_name
    argument_names = {argument.name for argument in case.arguments}
    result_names = tuple(
        result.name for result in case.results if result.name not in argument_names
    )
    phase = UPMEMPhase(
        case.kernel,
        name=case.name.replace("2mm", "two_mm").replace("3mm", "three_mm"),
        instantiate=tuple(case.instantiate),
        result_names=result_names,
        parallel_workers=64,
    )
    declarations = [_array_declaration(case, argument) for argument in case.arguments]
    for result in case.results:
        if result.name not in argument_names:
            partition_axis = max(range(len(result.shape)), key=result.shape.__getitem__)
            declarations.append(
                UPMEMArray(
                    result.name,
                    layout=TensorLayout.BLOCK,
                    partition_axis=partition_axis,
                )
            )
    program = UPMEMProgram(
        (phase,),
        name=f"polybench_{case.name.replace('2mm', 'two_mm').replace('3mm', 'three_mm')}",
        arrays=declarations,
        num_tasklets=16,
        orchestration=execution_manifest(case),
    )
    # UPMEMProgram intentionally has a small compiler-facing constructor.  The
    # reference suite attaches richer orchestration metadata without teaching
    # target/cost/backend code about PolyBench.
    program.polybench_case = case
    program.execution_spec = case.execution
    program.execution_manifest = program.orchestration
    program.manifest = {
        "program": program.name,
        "execution": program.execution_manifest,
        "arrays": [
            {
                "name": declaration.name,
                "layout": declaration.layout.name,
                "partition_axis": declaration.partition_axis,
                "halo": list(declaration.halo),
            }
            for declaration in declarations
        ],
    }
    return program


def get_case(name: str) -> PolyBenchCase:
    """Return a canonical case, accepting source spellings for 2mm and 3mm."""
    name = {"two_mm": "2mm", "three_mm": "3mm"}.get(name, name)
    try:
        return REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown PolyBench kernel {name!r}; choices: {NAMES}") from exc


def _random(
    rng: np.random.Generator, dims: tuple[int, ...], scale: float = 1.0
) -> np.ndarray:
    return rng.uniform(-scale, scale, size=dims).astype(_DTYPE)


def _zeros(dims: tuple[int, ...]) -> np.ndarray:
    return np.zeros(dims, dtype=_DTYPE)


def _make_inputs(
    case: PolyBenchCase, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    d, n = case.dims, case.name
    a: dict[str, np.ndarray]

    if n == "2mm":
        P, Q, R, S = d["P"], d["Q"], d["R"], d["S"]
        a = {
            "A": _random(rng, (P, Q)),
            "B": _random(rng, (Q, R)),
            "C": _random(rng, (R, S)),
            "D": _random(rng, (P, S)),
        }
    elif n == "3mm":
        P, Q, R, S, T = d["P"], d["Q"], d["R"], d["S"], d["T"]
        a = {
            "A": _random(rng, (P, Q)),
            "B": _random(rng, (Q, R)),
            "C": _random(rng, (R, S)),
            "D": _random(rng, (S, T)),
        }
    elif n == "adi":
        N = d["N"]
        a = {x: _random(rng, (N, N), 0.25) for x in ("u", "v", "p", "q")}
    elif n == "atax":
        M, N = d["M"], d["N"]
        a = {"A": _random(rng, (M, N)), "x": _random(rng, (N,)), "y": _zeros((N,))}
    elif n == "bicg":
        M, N = d["M"], d["N"]
        A = _random(rng, (N, M))
        a = {
            "A": A,
            "A_copy": A.copy(),
            "p": _random(rng, (M,)),
            "r": _random(rng, (N,)),
            "q": _zeros((N,)),
            "s": _zeros((M,)),
        }
    elif n == "cholesky":
        N = d["N"]
        X = _random(rng, (N, N), 0.1).astype(np.float64)
        A = X @ X.T + np.eye(N, dtype=np.float64)
        a = {"A": A.astype(_DTYPE)}
    elif n == "correlation":
        M, N = d["M"], d["N"]
        data = rng.uniform(0.1, 1.0, (N, M)).astype(_DTYPE)
        a = {
            "data_mean": data.copy(),
            "data_stddev": data.copy(),
            "data_for_center": data.copy(),
            "corr": _zeros((M, M)),
        }
    elif n == "covariance":
        M, N = d["M"], d["N"]
        a = {"data": _random(rng, (N, M)), "mean": _zeros((M,)), "cov": _zeros((M, M))}
    elif n == "deriche":
        W, H = d["W"], d["H"]
        a = {
            "imgIn": rng.random((W, H), dtype=np.float32),
            "imgOut": _zeros((W, H)),
            "y1": _zeros((W, H)),
            "y2": _zeros((W, H)),
        }
    elif n == "doitgen":
        R, Q, P, S = d["R"], d["Q"], d["P"], d["S"]
        a = {
            "A": _random(rng, (R, Q, S)),
            "x": _random(rng, (P, S)),
            "sum_": _zeros((P,)),
        }
    elif n == "durbin":
        N = d["N"]
        a = {"r": _random(rng, (N,), 0.01), "y": _zeros((N,))}
    elif n == "fdtd_2d":
        X, Y, T = d["Nx"], d["Ny"], d["Tmax"]
        a = {
            "ex": _random(rng, (X, Y)),
            "ey": _random(rng, (X, Y)),
            "hz": _random(rng, (X, Y)),
            "fict": _random(rng, (T,)),
        }
    elif n == "floyd_warshall":
        N = d["N"]
        path = rng.uniform(0.1, 5.0, (N, N)).astype(_DTYPE)
        np.fill_diagonal(path, 0.0)
        a = {"path": path}
    elif n == "gemm":
        P, Q, R = d["P"], d["Q"], d["R"]
        a = {
            "A": _random(rng, (P, Q)),
            "B": _random(rng, (Q, R)),
            "C": _random(rng, (P, R)),
            "output": _zeros((P, R)),
        }
    elif n == "gemver":
        N = d["N"]
        a = {
            "A": _random(rng, (N, N)),
            **{
                x: _random(rng, (N,))
                for x in ("u1", "u2", "v1", "v2", "x", "y", "w", "z")
            },
        }
    elif n == "gesummv":
        N = d["N"]
        a = {
            "A": _random(rng, (N, N)),
            "B": _random(rng, (N, N)),
            "x": _random(rng, (N,)),
            "y": _zeros((N,)),
        }
    elif n == "gramschmidt":
        M, N = d["M"], d["N"]
        a = {
            "A": rng.uniform(0.1, 1.0, (M, N)).astype(_DTYPE),
            "Q": _zeros((M, N)),
            "R": _zeros((N, N)),
        }
    elif n == "heat_3d":
        N = d["N"]
        a = {"A": _random(rng, (N, N, N), 0.25), "B": _random(rng, (N, N, N), 0.25)}
    elif n == "jacobi_1d":
        N = d["N"]
        a = {"A": _random(rng, (N,)), "B": _random(rng, (N,))}
    elif n == "jacobi_2d":
        N = d["N"]
        a = {"A": _random(rng, (N, N)), "B": _random(rng, (N, N))}
    elif n in {"lu", "ludcmp"}:
        N = d["N"]
        A = _random(rng, (N, N), 0.1)
        A += np.eye(N, dtype=_DTYPE) * 2.0
        a = {"A": A}
        if n == "ludcmp":
            a.update({"b": _random(rng, (N,)), "x": _zeros((N,)), "y": _zeros((N,))})
    elif n == "mvt":
        N = d["N"]
        A = _random(rng, (N, N))
        a = {
            "A": A,
            "A_copy": A.copy(),
            "y1": _random(rng, (N,)),
            "y2": _random(rng, (N,)),
            "x1": _random(rng, (N,)),
            "x2": _random(rng, (N,)),
            "x1_out": _zeros((N,)),
            "x2_out": _zeros((N,)),
        }
    elif n == "nussinov":
        N = d["N"]
        a = {"seq": rng.integers(0, 4, size=N).astype(_DTYPE), "table": _zeros((N, N))}
    elif n == "seidel_2d":
        N = d["N"]
        a = {"A": _random(rng, (N, N))}
    elif n == "symm":
        M, N = d["M"], d["N"]
        A = _random(rng, (M, M))
        B = _random(rng, (M, N))
        a = {
            "A0": A,
            "A1": A.copy(),
            "B0": B,
            "B1": B.copy(),
            "C": _random(rng, (M, N)),
        }
    elif n == "syr2k":
        M, N = d["M"], d["N"]
        A = _random(rng, (N, M))
        B = _random(rng, (N, M))
        C = _random(rng, (N, N))
        a = {
            "A": A,
            "A_copy": A.copy(),
            "B": B,
            "B_copy": B.copy(),
            "Cin": C,
            "Cout": C.copy(),
        }
    elif n == "syrk":
        M, N = d["M"], d["N"]
        A = _random(rng, (N, M))
        C = _random(rng, (N, N))
        a = {"A": A, "A_copy": A.copy(), "Cin": C, "Cout": C.copy()}
    elif n == "trisolv":
        N = d["N"]
        L = np.tril(_random(rng, (N, N), 0.1))
        L += np.eye(N, dtype=_DTYPE) * 2.0
        a = {"L": L, "b": _random(rng, (N,)), "x": _zeros((N,))}
    elif n == "trmm":
        M, N = d["M"], d["N"]
        a = {"A": _random(rng, (M, M)), "B": _random(rng, (M, N))}
    else:  # pragma: no cover - registry construction makes this unreachable
        raise KeyError(n)
    return a


def _run_reference(
    case: PolyBenchCase, a: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    n, d, s, ref = case.name, case.dims, case.scalars, case.reference
    if n == "2mm":
        return {"output": ref(a["A"], a["B"], a["C"], a["D"], s["alpha"], s["beta"])}
    if n == "3mm":
        return {"output": ref(a["A"], a["B"], a["C"], a["D"])}
    if n == "adi":
        out = ref(a["u"], a["v"], a["p"], a["q"], d["TSTEPS"], d["N"])
        return dict(zip(("u", "v", "p", "q"), out))
    if n == "atax":
        return {"y": ref(a["A"], a["x"])}
    if n == "bicg":
        sout, qout = ref(a["A"], a["s"], a["q"], a["p"], a["r"])
        return {"q": qout, "s": sout}
    if n == "cholesky":
        return {"A": ref(a["A"])}
    if n == "correlation":
        mean = _zeros((d["M"],))
        std = _zeros((d["M"],))
        _, _, _, corr = ref(
            a["data_mean"],
            mean,
            std,
            a["corr"],
            d["M"],
            d["N"],
            s["N_float"],
            s["epsilon"],
        )
        return {"corr": corr}
    if n == "covariance":
        _, mean, cov = ref(a["data"], a["mean"], a["cov"], d["M"], d["N"])
        return {"mean": mean, "cov": cov}
    if n == "deriche":
        coeff = [
            s[x]
            for x in (
                "a1",
                "a2",
                "a3",
                "a4",
                "a5",
                "a6",
                "a7",
                "a8",
                "b1",
                "b2",
                "c1",
                "c2",
            )
        ]
        _, img, y1, y2 = ref(a["imgIn"], a["imgOut"], a["y1"], a["y2"], *coeff)
        return {"imgOut": img, "y1": y1, "y2": y2}
    if n == "doitgen":
        A, _, summ = ref(a["A"], a["x"], a["sum_"])
        return {"A": A, "sum_": summ}
    if n == "durbin":
        return {"y": ref(a["r"], a["y"])}
    if n == "fdtd_2d":
        ex, ey, hz, _ = ref(a["ex"], a["ey"], a["hz"], a["fict"])
        return {"ex": ex, "ey": ey, "hz": hz}
    if n == "floyd_warshall":
        return {"path": ref(a["path"])}
    if n == "gemm":
        return {"output": ref(a["A"], a["B"], a["C"], s["beta"])}
    if n == "gemver":
        A, x, _, w = ref(
            a["A"],
            a["u1"],
            a["u2"],
            a["v1"],
            a["v2"],
            a["x"],
            a["y"],
            a["w"],
            a["z"],
            s["alpha"],
            s["beta"],
        )
        return {"A": A, "x": x, "w": w}
    if n == "gesummv":
        return {"y": ref(a["A"], a["B"], a["x"], a["y"], s["alpha"], s["beta"])}
    if n == "gramschmidt":
        A, Q, R = ref(a["A"], a["Q"], a["R"])
        return {"A": A, "Q": Q, "R": R}
    if n == "heat_3d":
        A, B = ref(a["A"], a["B"], d["TSTEPS"], d["N"])
        return {"A": A, "B": B}
    if n == "jacobi_1d":
        A, B = ref(a["A"], a["B"], d["TSTEPS"], d["N"])
        return {"A": A, "B": B}
    if n == "jacobi_2d":
        A, B = ref(a["A"], a["B"], d["TSTEPS"])
        return {"A": A, "B": B}
    if n == "lu":
        return {"A": ref(a["A"])}
    if n == "ludcmp":
        A, _, x, y = ref(a["A"], a["b"], a["x"], a["y"])
        return {"A": A, "x": x, "y": y}
    if n == "mvt":
        x1, x2 = ref(a["A"], a["x1"], a["x2"], a["y1"], a["y2"])
        return {"x1_out": x1, "x2_out": x2}
    if n == "nussinov":
        return {"table": ref(a["seq"], a["table"])}
    if n == "seidel_2d":
        return {"A": ref(a["A"], d["TSTEPS"])}
    if n == "symm":
        return {
            "C": ref(a["A0"], a["B0"], a["C"], s["alpha"], s["beta"], d["M"], d["N"])
        }
    if n == "syr2k":
        return {"Cout": ref(a["A"], a["B"], a["Cin"], s["alpha"], s["beta"])}
    if n == "syrk":
        return {"Cout": ref(a["A"], a["Cin"], s["alpha"], s["beta"])}
    if n == "trisolv":
        return {"x": ref(a["L"], a["x"], a["b"])}
    if n == "trmm":
        return {"B": ref(a["A"], a["B"], s["alpha"])}
    raise KeyError(n)  # pragma: no cover


__all__ = [
    "ArgumentSpec",
    "ResultSpec",
    "ExecutionPhase",
    "ExecutionSpec",
    "PolyBenchCase",
    "REGISTRY",
    "NAMES",
    "get_case",
    "execution_manifest",
    "build_upmem_program",
]
