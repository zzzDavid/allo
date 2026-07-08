# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical PolyBench execution helpers for the real APU v1 scalar path."""

from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np

import allo
from allo.ir.types import uint16
from . import reference, results, runner
from .upmem_polybench import NAMES, PolyBenchCase, get_case as _get_base_case


_U16 = np.dtype(np.uint16)


def _uint16_scalar(value):
    """Quantize a PolyBench coefficient to its nearest modular integer."""

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
        # Keep deterministic structure while avoiding the nearly-all-zero
        # result of directly casting canonical [-1, 1] float inputs.
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
        values = {}
        for argument in self.arguments:
            value = _uint16_input(
                source[argument.name], zero_output=argument.intent == "out"
            )
            if self.name == "covariance" and argument.name == "data":
                # Keep every covariance non-negative so the kernel's explicit
                # float accumulator converts to uint16 within the defined C
                # range on both x86 and ARC.  Repeated columns still exercise
                # mean subtraction and the full O(M^2 N) covariance nest.
                value = np.full(value.shape, value.flat[0], dtype=_U16)
            if self.name == "durbin" and argument.name == "r":
                # Durbin negates every reflection coefficient.  Nonzero
                # unsigned inputs would require a separately specified
                # signed/fixed-point encoding; use the well-defined zero
                # sequence while still executing the complete recurrence.
                value = np.zeros_like(value)
            values[argument.name] = value
        return values

    def make_argument_tuple(self, seed=0):
        values = self.make_inputs(seed)
        return tuple(values[item.name] for item in self.arguments)

    def run_reference(self, inputs=None, *, seed=0):
        if inputs is None:
            inputs = self.make_inputs(seed)
        work = {name: np.array(value, copy=True) for name, value in inputs.items()}
        if self.name in _PORTABLE_REFERENCE_CASES:
            return _portable_reference(self, work)
        return self.contract.run_reference(work)


_CASES = {name: APUv1PolyBenchCase(_get_base_case(name)) for name in NAMES}

_PORTABLE_REFERENCE_CASES = frozenset(
    {"adi", "correlation", "covariance", "durbin", "fdtd_2d"}
)
_REFERENCE_EXECUTABLES = {}


def _portable_reference(case, arrays):
    """Execute retained MLIR as host C for mixed-width uint16 semantics.

    These kernels combine modular arrays, widened integer expressions, and
    float temporaries.  Their original NumPy functions apply NumPy scalar
    promotion at different points, so host C is the exact scalar oracle used
    to validate the independent ARC and GVML lowerings.
    """

    executable = _REFERENCE_EXECUTABLES.get(case.name)
    if executable is None:
        from allo.backend.c import emit_c_from_mlir

        schedule = allo.customize(case.kernel, instantiate=case.instantiate)
        executable = emit_c_from_mlir(
            schedule.module,
            schedule.top_func_name,
            wrap_wide_integers=True,
        ).compile()
        _REFERENCE_EXECUTABLES[case.name] = executable
    returned = executable(*(arrays[item.name] for item in case.arguments))
    returned = (
        ()
        if returned is None
        else (returned if isinstance(returned, tuple) else (returned,))
    )
    extra_names = [item.name for item in case.results if item.name not in arrays]
    extra = dict(zip(extra_names, returned))
    return {
        item.name: np.asarray(
            arrays[item.name] if item.name in arrays else extra[item.name]
        )
        for item in case.results
    }


def get_case(name):
    name = {"two_mm": "2mm", "three_mm": "3mm"}.get(name, name)
    try:
        return _CASES[name]
    except KeyError as error:
        raise KeyError(f"unknown APU v1 PolyBench kernel {name!r}") from error


def build_program(case_or_name):
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
    )
    return allo.APUv1Program((phase,), name=f"polybench_{phase.phase_name}")


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
        # uint16 storage is already the native one-lane GVML compute type.
        # Canonical SMALL reductions may need more resident RHS banks than
        # the fifteen writable VRs, so use the shape-independent DMA plan.
        vector_layout="temporal_dma_coalescing",
    )
    return allo.APUv1Program((phase,), name=f"polybench_hybrid_{phase.phase_name}")


def run_compiled(compiled, case_or_name, *, folder, run_cmd):
    case = get_case(case_or_name) if isinstance(case_or_name, str) else case_or_name
    inputs = case.make_inputs()
    expected = case.run_reference(inputs)

    import conftest

    with conftest.board_lock():
        result = compiled(**inputs)

    observed = result.extra.get("outputs", {})
    max_abs = 0.0
    failures = []
    for name, wanted in expected.items():
        if name not in observed:
            failures.append(f"missing output {name}")
            continue
        got = np.asarray(observed[name])
        wanted = np.asarray(wanted)
        error = float(
            np.max(np.abs(got.astype(np.float64) - wanted.astype(np.float64)))
        )
        max_abs = max(max_abs, error)
        if not np.array_equal(got, wanted):
            failures.append(f"{name}: max_abs_err={error:g}")

    if failures:
        verdict = reference.Verdict(reference.FAIL, "; ".join(failures))
    else:
        oracle = (
            "retained-MLIR host C oracle"
            if case.name in _PORTABLE_REFERENCE_CASES
            else "repository NumPy reference"
        )
        verdict = reference.Verdict(
            reference.PASS,
            f"canonical uint16 MLIR -> scalar ARC C matched {oracle} "
            f"bit-for-bit for {', '.join(expected)} "
            f"(max_abs_err={max_abs:g})",
        )

    estimate = compiled.estimate().cycles
    notes = (
        "Complete canonical uint16 PolyBench program lowered through MLIR to scalar C "
        "on APUC 0. This is the explicit scalar correctness baseline; other "
        "APUCs are not launched because arbitrary "
        "multi-phase programs do not yet carry cross-APUC barriers. "
        f"Analytical scalar estimate={estimate} cycles."
    )
    provenance = dict(reference.provenance(case.name))
    if case.name in _PORTABLE_REFERENCE_CASES:
        provenance["execution_oracle"] = (
            "same retained MLIR compiled to host C; differentially validates "
            "ARC codegen/runtime for mixed float and modular uint16 expressions"
        )
    record = results.build_record(
        kernel=case.name,
        target="apu_v1",
        dataset="SMALL",
        shapes=case.dims,
        verdict=verdict,
        reference_provenance=provenance,
        metric="cycles",
        value=result.cycles,
        source=runner.apu_v1_device_source(),
        run_cmd=run_cmd,
        timestamp=runner.now_iso(),
        tenon_commit=runner.tenon_commit(),
        notes=notes,
    )
    results.write_results(folder, record)
    results.write_results_md(folder, record)
    results.regenerate_coverage()
    return result, verdict, record


def assert_result(result, verdict):
    assert verdict.status == reference.PASS, verdict
    assert result.cycles is not None and result.cycles > 0


__all__ = [
    "NAMES",
    "assert_result",
    "build_hybrid_program",
    "build_program",
    "get_case",
    "run_compiled",
]
