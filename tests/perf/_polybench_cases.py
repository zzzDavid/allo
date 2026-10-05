# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SMALL PolyBench gemm/2mm/3mm cases for the APU v1 hybrid perf tests.

Moved from the deleted ``tests/pim/lib`` leaf harness (``upmem_polybench`` and
``apu_v1``). ``get_base_case`` returns the float32 canonical contract;
``get_case`` returns its uint16 APU v1 specialization. Inputs are generated
exactly as before, so test data is unchanged.
"""

from __future__ import annotations

import importlib
import json
import pathlib
from dataclasses import dataclass, replace
from types import ModuleType
from typing import Callable, Iterable, Mapping

import numpy as np

import allo
from allo.ir.types import float32, uint16


_DTYPE = np.dtype(np.float32)
_U16 = np.dtype(np.uint16)

# tests/perf/_polybench_cases.py -> parents[2] == experiments/allo
_PSIZE = json.loads(
    (
        pathlib.Path(__file__).resolve().parents[2]
        / "examples"
        / "polybench"
        / "psize.json"
    ).read_text()
)
_PSIZE_ALIAS = {"2mm": "two_mm", "3mm": "three_mm"}


def _dims(name: str) -> dict[str, int]:
    small = _PSIZE[_PSIZE_ALIAS.get(name, name)]["small"]
    return {key: int(value) for key, value in small.items()}


@dataclass(frozen=True)
class ArgumentSpec:
    name: str
    shape: tuple[int, ...]
    intent: str
    dtype: np.dtype = _DTYPE

    def __post_init__(self):
        if self.intent not in {"in", "out", "inout"}:
            raise ValueError(f"invalid intent {self.intent!r} for {self.name}")


@dataclass(frozen=True)
class ResultSpec:
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype = _DTYPE


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
    ambient_bindings: tuple[tuple[str, float | int], ...] = ()

    @property
    def dims(self) -> dict[str, int]:
        return dict(self.dimensions)

    @property
    def scalars(self) -> dict[str, float | int]:
        return dict(self.ambient_bindings)

    @property
    def module(self) -> ModuleType:
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
        dims = self.dims
        return [float32, *(dims[name] for name in self.instantiate_dimensions)]

    def make_inputs(self, seed: int = 0) -> dict[str, np.ndarray]:
        values = _make_inputs(self, np.random.default_rng(seed))
        expected = tuple(arg.name for arg in self.arguments)
        if tuple(values) != expected:
            raise AssertionError(
                f"{self.name}: generator order {tuple(values)} != ABI {expected}"
            )
        return values

    def run_reference(self, inputs=None, *, seed: int = 0) -> dict[str, np.ndarray]:
        if inputs is None:
            inputs = self.make_inputs(seed)
        work = {
            arg.name: np.array(inputs[arg.name], copy=True) for arg in self.arguments
        }
        outputs = _run_reference(self, work)
        return {
            result.name: np.asarray(outputs[result.name]).astype(
                result.dtype, copy=False
            )
            for result in self.results
        }


def _arg(name: str, dims: tuple[int, ...], intent: str = "in") -> ArgumentSpec:
    return ArgumentSpec(name, tuple(int(v) for v in dims), intent)


def _result(name: str, dims: tuple[int, ...]) -> ResultSpec:
    return ResultSpec(name, tuple(int(v) for v in dims))


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
        ambient_bindings=tuple((bindings or {}).items()),
    )


def _build_registry() -> dict[str, PolyBenchCase]:
    cases: dict[str, PolyBenchCase] = {}

    d = _dims("2mm")
    P, Q, R, S = d["P"], d["Q"], d["R"], d["S"]
    cases["2mm"] = _case(
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

    d = _dims("3mm")
    P, Q, R, S, T = d["P"], d["Q"], d["R"], d["S"], d["T"]
    cases["3mm"] = _case(
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

    d = _dims("gemm")
    P, Q, R = d["P"], d["Q"], d["R"]
    cases["gemm"] = _case(
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
    return cases


def _random(rng: np.random.Generator, dims: tuple[int, ...]) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, size=dims).astype(_DTYPE)


def _make_inputs(
    case: PolyBenchCase, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    d, n = case.dims, case.name
    if n == "2mm":
        P, Q, R, S = d["P"], d["Q"], d["R"], d["S"]
        return {
            "A": _random(rng, (P, Q)),
            "B": _random(rng, (Q, R)),
            "C": _random(rng, (R, S)),
            "D": _random(rng, (P, S)),
        }
    if n == "3mm":
        P, Q, R, S, T = d["P"], d["Q"], d["R"], d["S"], d["T"]
        return {
            "A": _random(rng, (P, Q)),
            "B": _random(rng, (Q, R)),
            "C": _random(rng, (R, S)),
            "D": _random(rng, (S, T)),
        }
    P, Q, R = d["P"], d["Q"], d["R"]
    return {
        "A": _random(rng, (P, Q)),
        "B": _random(rng, (Q, R)),
        "C": _random(rng, (P, R)),
        "output": np.zeros((P, R), dtype=_DTYPE),
    }


def _run_reference(
    case: PolyBenchCase, a: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    n, s, ref = case.name, case.scalars, case.reference
    if n == "2mm":
        return {"output": ref(a["A"], a["B"], a["C"], a["D"], s["alpha"], s["beta"])}
    if n == "3mm":
        return {"output": ref(a["A"], a["B"], a["C"], a["D"])}
    return {"output": ref(a["A"], a["B"], a["C"], s["beta"])}


REGISTRY = _build_registry()


def _canonical_name(name: str) -> str:
    return {"two_mm": "2mm", "three_mm": "3mm"}.get(name, name)


def get_base_case(name: str) -> PolyBenchCase:
    """Return the float32 canonical case (former ``upmem_polybench.get_case``)."""
    return REGISTRY[_canonical_name(name)]


def _uint16_scalar(value):
    if isinstance(value, (int, np.integer)):
        return int(value) & 0xFFFF
    value = float(value)
    if value == 0.0:
        return 0
    return int(round(value)) & 0xFFFF


def _uint16_input(value, *, zero_output=False):
    if zero_output:
        return np.zeros(value.shape, dtype=_U16)
    source = np.asarray(value)
    if np.issubdtype(source.dtype, np.floating):
        magnitude = np.rint(np.abs(source.astype(np.float64)) * 16.0)
        converted = np.where(source == 0, 0, np.maximum(1, magnitude))
        return np.asarray(converted, dtype=_U16)
    return np.asarray(source, dtype=_U16)


@dataclass(frozen=True)
class APUv1PolyBenchCase:
    """APU-specific uint16 specialization of one canonical PolyBench case."""

    base: PolyBenchCase

    def __post_init__(self):
        bindings = tuple(
            (name, _uint16_scalar(value)) for name, value in self.base.ambient_bindings
        )
        contract = replace(
            self.base,
            arguments=tuple(replace(item, dtype=_U16) for item in self.base.arguments),
            results=tuple(replace(item, dtype=_U16) for item in self.base.results),
            ambient_bindings=bindings,
        )
        object.__setattr__(self, "contract", contract)

    def __getattr__(self, name):
        return getattr(self.contract, name)

    @property
    def instantiate(self):
        dims = self.dims
        return [uint16, *(dims[name] for name in self.instantiate_dimensions)]

    def make_inputs(self, seed=0):
        source = self.base.make_inputs(seed)
        return {
            argument.name: _uint16_input(
                source[argument.name], zero_output=argument.intent == "out"
            )
            for argument in self.arguments
        }


_APU_CASES = {name: APUv1PolyBenchCase(case) for name, case in REGISTRY.items()}


def get_case(name: str) -> APUv1PolyBenchCase:
    """Return the uint16 APU v1 case (former ``lib.apu_v1.get_case``)."""
    return _APU_CASES[_canonical_name(name)]


def build_hybrid_program(case_or_name):
    """Build a canonical phase whose MLIR regions are selected structurally."""

    case = get_case(case_or_name) if isinstance(case_or_name, str) else case_or_name
    argument_names = {argument.name for argument in case.arguments}
    result_names = tuple(
        result.name for result in case.results if result.name not in argument_names
    )
    phase = allo.APUv1Phase(
        case.kernel,
        name=case.name.replace("2mm", "two_mm").replace("3mm", "three_mm"),
        instantiate=tuple(case.instantiate),
        result_names=result_names,
        vectorize=True,
        # Canonical SMALL reductions may need more resident RHS banks than the
        # fifteen writable VRs, so use the shape-independent DMA plan.
        vector_layout="temporal_dma_coalescing",
    )
    return allo.APUv1Program((phase,), name=f"polybench_hybrid_{phase.phase_name}")
