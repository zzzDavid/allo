# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retained-MLIR discovery and immutable planning for hybrid APU v1 programs.

This module stops at the logical execution manifest.  Physical GVML/ARC
source generation consumes the manifest elsewhere; discovery never rewrites
Python source and never silently turns a selected vector region into ARC C.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import inspect
import math
import re
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ..perf import CostEvent
from ..perf.graph import Activity, ExecutionGraph, Occupancy
from ..spmw_codegen import RunResult

from .apu_v1_vector_cost import estimate_apu_v1_plan, rank_apu_v1_plans
from .apu_v1_vectorize import (
    ContractionAnalysis,
    NoContractionError,
    ValueAccess,
    analyze_apu_v1_contractions,
    generate_apu_v1_vectorization_candidates,
)


_DTYPE = r"bf16|f16|f32|f64|i\d+|ui\d+"
_MEMREF = re.compile(rf"memref<(?P<shape>(?:\d+x)*)(?P<dtype>{_DTYPE})>")
_FUNC = re.compile(r"\bfunc\.func\s+@(?P<name>[-\w.$]+)\s*\((?P<args>[^)]*)\)")
_FORMAL = re.compile(r"(?P<ssa>%[-\w.$]+)\s*:\s*(?P<type>memref<[^>]+>)")
_ALLOC = re.compile(
    rf"(?P<ssa>%[-\w.$]+)\s*=\s*memref\.alloc\(\)\s*"
    rf'\{{[^}}]*name\s*=\s*"(?P<name>[^"]+)"[^}}]*\}}\s*:\s*'
    rf"(?P<type>memref<[^>]+>)"
)
_FILL = re.compile(r"\blinalg\.fill\b.*\bouts\((?P<ssa>%[-\w.$]+)\s*:")
_CALL = re.compile(r"\bcall\s+@(?P<function>[-\w.$]+)\s*\((?P<args>[^)]*)\)")
_LOAD_ARG = re.compile(r"(?:affine|memref)\.load\s+(?P<arg>%arg\d+)\[")
_STORE_ARG = re.compile(
    r"(?:affine|memref)\.store\s+%[-\w.$]+\s*,\s*(?P<arg>%arg\d+)\["
)
_VALUE_ATTR = re.compile(r'(?:from|to)\s*=\s*"(?P<name>[^"]+)"')


def _type_parts(text: str) -> tuple[tuple[int, ...], str]:
    match = _MEMREF.search(text)
    if match is None:
        raise ValueError(f"hybrid APU manifest requires a ranked memref, got {text!r}")
    shape = tuple(
        int(item) for item in match.group("shape").rstrip("x").split("x") if item
    )
    return shape, match.group("dtype")


def _function_regions(text: str) -> Mapping[str, tuple[str, ...]]:
    lines = text.splitlines()
    regions = {}
    index = 0
    while index < len(lines):
        start = _FUNC.search(lines[index])
        if start is None:
            index += 1
            continue
        begin = index
        depth = 0
        opened = False
        while index < len(lines):
            depth += lines[index].count("{") - lines[index].count("}")
            opened = opened or "{" in lines[index]
            index += 1
            if opened and depth == 0:
                break
        name = start.group("name")
        if name in regions:
            raise ValueError(f"duplicate MLIR function {name!r}")
        regions[name] = tuple(lines[begin:index])
    return MappingProxyType(regions)


@dataclass(frozen=True)
class APUv1PrecisionPolicy:
    """Explicit canonical-storage to GVML-compute conversion contract."""

    storage_dtype: str
    compute_dtype: str
    accumulation_dtype: str
    rounding: str = "nearest_even"
    overflow: str = "finite"

    def __post_init__(self):
        if not all(
            re.fullmatch(_DTYPE, value)
            for value in (
                self.storage_dtype,
                self.compute_dtype,
                self.accumulation_dtype,
            )
        ):
            raise ValueError("precision-policy dtypes must be concrete scalar types")
        if not self.rounding or not self.overflow:
            raise ValueError("precision policy requires rounding and overflow behavior")

    @classmethod
    def f32_to_f16(cls):
        return cls("f32", "f16", "f16", "nearest_even", "finite")


@dataclass(frozen=True)
class APUv1ValueManifest:
    name: str
    shape: tuple[int, ...]
    dtype: str
    program_input: bool = False
    program_output: bool = False
    intermediate: bool = False
    zero_initialized: bool = False

    def manifest(self):
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "program_input": self.program_input,
            "program_output": self.program_output,
            "intermediate": self.intermediate,
            "zero_initialized": self.zero_initialized,
        }


@dataclass(frozen=True)
class APUv1RegionOperand:
    position: int
    role: str
    value: str
    shape: tuple[int, ...]
    storage_dtype: str
    compute_dtype: str
    access: str

    def manifest(self):
        return {
            "position": self.position,
            "role": self.role,
            "value": self.value,
            "shape": list(self.shape),
            "storage_dtype": self.storage_dtype,
            "compute_dtype": self.compute_dtype,
            "access": self.access,
        }


@dataclass(frozen=True)
class APUv1ConversionManifest:
    id: str
    value: str
    shape: tuple[int, ...]
    source_dtype: str
    target_dtype: str
    before_region: str | None = None
    after_region: str | None = None
    reason: str = "precision_boundary"

    def __post_init__(self):
        if (self.before_region is None) == (self.after_region is None):
            raise ValueError("conversion must be attached before or after one region")

    def manifest(self):
        return {
            "id": self.id,
            "value": self.value,
            "shape": list(self.shape),
            "source_dtype": self.source_dtype,
            "target_dtype": self.target_dtype,
            "before_region": self.before_region,
            "after_region": self.after_region,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class APUv1RegionManifest:
    id: str
    function: str
    ordinal: int
    kind: str
    operands: tuple[APUv1RegionOperand, ...]
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    produced_intermediates: tuple[str, ...]
    consumed_intermediates: tuple[str, ...]
    dependencies: tuple[str, ...]
    barrier: str | None = None
    storage_analysis: ContractionAnalysis | None = None
    compute_analysis: ContractionAnalysis | None = None
    plans: tuple[object, ...] = ()
    selected_plan: object | None = None
    selected_estimate: object | None = None
    physical_plan_verified: bool = False

    def __post_init__(self):
        if self.kind not in {"vector", "scalar"}:
            raise ValueError(f"unsupported APU region kind {self.kind!r}")
        if self.kind == "vector" and (
            self.compute_analysis is None or self.selected_plan is None
        ):
            raise ValueError(
                "vector regions require compute analysis and selected plan"
            )
        if self.kind == "scalar" and self.selected_plan is not None:
            raise ValueError("scalar regions cannot retain a vector plan")

    def manifest(self):
        return {
            "id": self.id,
            "function": self.function,
            "ordinal": self.ordinal,
            "kind": self.kind,
            "operands": [operand.manifest() for operand in self.operands],
            "reads": list(self.reads),
            "writes": list(self.writes),
            "produced_intermediates": list(self.produced_intermediates),
            "consumed_intermediates": list(self.consumed_intermediates),
            "dependencies": list(self.dependencies),
            "barrier": self.barrier,
            "storage_dtype": (
                self.storage_analysis.numeric_type if self.storage_analysis else None
            ),
            "compute_dtype": (
                self.compute_analysis.numeric_type if self.compute_analysis else None
            ),
            "plans": [plan.name for plan in self.plans],
            "selected_plan": getattr(self.selected_plan, "name", None),
            "selected_cycles": (
                int(self.selected_estimate.cycles)
                if self.selected_estimate is not None
                else None
            ),
            "physical_plan_verified": self.physical_plan_verified,
        }


@dataclass(frozen=True)
class APUv1BarrierManifest:
    id: str
    after_regions: tuple[str, ...]
    before_regions: tuple[str, ...]
    values: tuple[str, ...]
    reason: str

    def manifest(self):
        return {
            "id": self.id,
            "after_regions": list(self.after_regions),
            "before_regions": list(self.before_regions),
            "values": list(self.values),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class APUv1PhaseManifest:
    name: str
    mlir_function: str
    regions: tuple[APUv1RegionManifest, ...]
    barriers: tuple[APUv1BarrierManifest, ...]
    conversions: tuple[APUv1ConversionManifest, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    intermediates: tuple[str, ...]
    produces: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    bindings: tuple[tuple[str, str], ...] = ()

    def manifest(self):
        return {
            "name": self.name,
            "mlir_function": self.mlir_function,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "intermediates": list(self.intermediates),
            "produces": list(self.produces),
            "dependencies": list(self.dependencies),
            "bindings": dict(self.bindings),
            "regions": [region.manifest() for region in self.regions],
            "barriers": [barrier.manifest() for barrier in self.barriers],
            "conversions": [conversion.manifest() for conversion in self.conversions],
        }


@dataclass(frozen=True)
class APUv1HybridManifest:
    name: str
    values: tuple[APUv1ValueManifest, ...]
    phases: tuple[APUv1PhaseManifest, ...]
    precision_policy: APUv1PrecisionPolicy | None = None

    @property
    def has_vector_regions(self):
        return any(
            region.kind == "vector" for phase in self.phases for region in phase.regions
        )

    def manifest(self):
        return {
            "name": self.name,
            "precision_policy": (
                None
                if self.precision_policy is None
                else {
                    "storage_dtype": self.precision_policy.storage_dtype,
                    "compute_dtype": self.precision_policy.compute_dtype,
                    "accumulation_dtype": self.precision_policy.accumulation_dtype,
                    "rounding": self.precision_policy.rounding,
                    "overflow": self.precision_policy.overflow,
                }
            ),
            "values": [value.manifest() for value in self.values],
            "phases": [phase.manifest() for phase in self.phases],
        }


def _adapt_analysis(analysis, dtype):
    def access(value: ValueAccess):
        return replace(value, dtype=dtype)

    return replace(
        analysis,
        lhs=access(analysis.lhs),
        rhs=access(analysis.rhs),
        accumulator=access(analysis.accumulator),
        output=access(analysis.output),
        numeric_type=dtype,
    )


def _bind_analysis_values(analysis, semantic, formals, actual_values):
    """Replace helper-local value labels with the caller's retained SSA values."""

    actual_by_formal = {
        formal.group("ssa"): actual for formal, actual in zip(formals, actual_values)
    }

    def bind(access):
        formal = semantic.get(access.value)
        if formal is None or formal not in actual_by_formal:
            raise ValueError(
                f"cannot bind contraction value {access.value!r} to the call ABI"
            )
        return replace(access, value=actual_by_formal[formal])

    return replace(
        analysis,
        lhs=bind(analysis.lhs),
        rhs=bind(analysis.rhs),
        accumulator=bind(analysis.accumulator),
        output=bind(analysis.output),
    )


def _select_vector_plan(analysis, target, cost, layout):
    candidates = generate_apu_v1_vectorization_candidates(analysis)
    plans = tuple(candidate.plan for candidate in candidates)
    estimates = rank_apu_v1_plans(plans, target, cost) if cost is not None else ()
    from .apu_v1_vector_program import _realize

    realization_errors = {}
    legal = []
    for plan in plans:
        realization, error = _realize(analysis, plan)
        if realization is None:
            realization_errors[plan.name] = error
        else:
            legal.append(plan)
    if not legal:
        details = "; ".join(
            f"{name}: {error}" for name, error in realization_errors.items()
        )
        raise ValueError(f"no physically realizable APU vector plan: {details}")
    if layout is None:
        ranked = tuple(item.plan for item in estimates) if estimates else plans[::-1]
        selected = next(plan for plan in ranked if plan in legal)
    else:
        selected = next((plan for plan in plans if plan.name == layout), None)
        if selected is None:
            raise ValueError(
                f"unknown hybrid APU vector plan {layout!r} for {analysis.function!r}"
            )
        if selected not in legal:
            raise ValueError(
                f"hybrid APU vector plan {layout!r} is not physically realizable: "
                f"{realization_errors[selected.name]}"
            )
    estimate = next(
        (item for item in estimates if item.plan.name == selected.name), None
    )
    return plans, selected, estimate


def _formal_accesses(lines):
    reads, writes, semantic = set(), set(), {}
    for line in lines:
        load = _LOAD_ARG.search(line)
        store = _STORE_ARG.search(line)
        match = load or store
        if match is None:
            continue
        formal = match.group("arg")
        (reads if load else writes).add(formal)
        attr = _VALUE_ATTR.search(line)
        if attr is not None:
            semantic.setdefault(attr.group("name"), formal)
    return reads, writes, semantic


def discover_apu_v1_hybrid_manifest(
    phase,
    schedule,
    artifact,
    target,
    *,
    cost=None,
    program_name=None,
):
    """Discover ordered vector/scalar regions from one canonical MLIR phase."""

    text = str(schedule.module)
    functions = _function_regions(text)
    top = schedule.top_func_name
    if top not in functions:
        raise ValueError(f"retained MLIR has no top function {top!r}")
    precision = getattr(phase, "precision_policy", None)
    vectorize = getattr(phase, "vectorize", False)
    if vectorize not in {False, "required"}:
        raise ValueError("APUv1Phase.vectorize must be False or 'required'")

    # Contraction discovery is a vector-lowering concern.  Scalar phases may
    # contain arbitrary PolyBench loop nests and must not be rejected merely
    # because one happens to resemble an unsupported contraction.
    analyses = {}
    if vectorize == "required":
        try:
            analyses = {
                item.function: item for item in analyze_apu_v1_contractions(text)
            }
        except NoContractionError:
            analyses = {}
    if vectorize == "required" and not analyses:
        raise NoContractionError("selected APU phase contains no rank-2 contraction")

    source_names = tuple(inspect.signature(phase.kernel).parameters)
    source_abi = [item for item in artifact.arguments if item.source == "argument"]
    result_abi = [item for item in artifact.arguments if item.source == "result"]
    if len(source_names) != len(source_abi):
        raise ValueError("hybrid manifest source ABI does not match Python signature")
    if len(phase.result_names) != len(result_abi):
        raise ValueError("hybrid manifest result_names do not match returned memrefs")

    top_match = _FUNC.search(functions[top][0])
    formal_top = tuple(_FORMAL.finditer(top_match.group("args")))
    if len(formal_top) != len(source_names):
        raise ValueError("hybrid manifest top-level MLIR argument mismatch")
    ssa_values = {}
    values = {}
    bindings = dict(getattr(phase, "bindings", {}))
    for formal, source_name, abi in zip(formal_top, source_names, source_abi):
        name = bindings.get(source_name, source_name)
        shape, _signless_dtype = _type_parts(formal.group("type"))
        # CArgument decodes the function-level itypes string, which is the
        # authoritative signedness for signless MLIR integer formals.
        dtype = abi.dtype
        ssa_values[formal.group("ssa")] = name
        values[name] = APUv1ValueManifest(
            name,
            shape,
            dtype,
            program_input=abi.mode in {"in", "both"},
            program_output=abi.mode in {"out", "both"},
        )
    filled_allocations = {
        match.group("ssa")
        for line in functions[top]
        for match in [_FILL.search(line)]
        if match is not None
    }
    explicit_zero = set(getattr(phase, "zero_initialize", ()))
    for line in functions[top]:
        alloc = _ALLOC.search(line)
        if alloc is None:
            continue
        # MLIR lowering also introduces scalar index slots.  They are an ARC
        # implementation detail, not persistent tensor values in the hybrid
        # manifest.
        if _MEMREF.search(alloc.group("type")) is None:
            continue
        shape, dtype = _type_parts(alloc.group("type"))
        if dtype.startswith("i") and re.search(r"\bunsigned\b", line):
            dtype = "u" + dtype
        name = alloc.group("name")
        ssa_values[alloc.group("ssa")] = name
        values[name] = APUv1ValueManifest(
            name,
            shape,
            dtype,
            intermediate=True,
            zero_initialized=(
                alloc.group("ssa") in filled_allocations or name in explicit_zero
            ),
        )
    for name, abi in zip(phase.result_names, result_abi):
        shape, dtype = tuple(abi.shape), abi.dtype
        old = values.get(name)
        values[name] = APUv1ValueManifest(
            name,
            shape,
            dtype,
            program_output=True,
            intermediate=False,
            zero_initialized=bool(old and old.zero_initialized),
        )

    calls = []
    for line in functions[top]:
        call = _CALL.search(line)
        if call is None:
            continue
        actual_ssa = tuple(item.strip() for item in call.group("args").split(","))
        try:
            actual_values = tuple(ssa_values[item] for item in actual_ssa)
        except KeyError as error:
            raise ValueError(
                f"call uses unknown top-level SSA value {error.args[0]}"
            ) from error
        calls.append((call.group("function"), actual_values))

    if not calls:
        # The other PolyBench kernels remain one explicit scalar region.
        operands = tuple(
            APUv1RegionOperand(
                index,
                "argument",
                value.name,
                value.shape,
                value.dtype,
                value.dtype,
                "readwrite" if value.program_output else "read",
            )
            for index, value in enumerate(values.values())
            if value.program_input or value.program_output
        )
        region = APUv1RegionManifest(
            "region0",
            top,
            0,
            "scalar",
            operands,
            tuple(item.value for item in operands if item.access != "write"),
            tuple(item.value for item in operands if item.access != "read"),
            (),
            (),
            (),
        )
        phase_manifest = APUv1PhaseManifest(
            phase.phase_name,
            top,
            (region,),
            (),
            (),
            tuple(value.name for value in values.values() if value.program_input),
            tuple(value.name for value in values.values() if value.program_output),
            (),
            tuple(getattr(phase, "produces", ())),
            tuple(getattr(phase, "dependencies", ())),
            tuple(dict(getattr(phase, "bindings", {})).items()),
        )
        return APUv1HybridManifest(
            program_name or phase.phase_name,
            tuple(values.values()),
            (phase_manifest,),
            precision,
        )

    last_writer = {}
    regions = []
    for ordinal, (function, actual_values) in enumerate(calls):
        if function not in functions:
            raise ValueError(f"top-level call references unknown function {function!r}")
        signature = _FUNC.search(functions[function][0])
        formals = tuple(_FORMAL.finditer(signature.group("args")))
        if len(formals) != len(actual_values):
            raise ValueError(f"call ABI mismatch for region function {function!r}")
        read_formals, write_formals, semantic = _formal_accesses(functions[function])
        analysis = analyses.get(function)
        selected = bool(analysis is not None and vectorize == "required")
        compute_analysis = None
        plans = ()
        selected_plan = selected_estimate = None
        if selected:
            compute_analysis = analysis
            if analysis.numeric_type not in {
                "f16",
                "i1",
                "i8",
                "i16",
                "ui1",
                "ui8",
                "ui16",
            }:
                if precision is None:
                    raise ValueError(
                        f"selected vector region {function!r} uses {analysis.numeric_type}; "
                        "an explicit precision_policy is required"
                    )
                else:
                    if precision.storage_dtype != analysis.numeric_type:
                        raise ValueError(
                            f"precision policy storage dtype {precision.storage_dtype} "
                            f"does not match {analysis.numeric_type}"
                        )
                    compute_analysis = _adapt_analysis(
                        analysis, precision.compute_dtype
                    )
            if selected:
                compute_analysis = _bind_analysis_values(
                    compute_analysis, semantic, formals, actual_values
                )
                plans, selected_plan, selected_estimate = _select_vector_plan(
                    compute_analysis,
                    target,
                    cost,
                    getattr(phase, "vector_layout", None),
                )
        kind = "vector" if selected else "scalar"
        role_formals = {}
        if analysis is not None:
            role_formals = {
                semantic.get(analysis.lhs.value): "lhs",
                semantic.get(analysis.rhs.value): "rhs",
                semantic.get(analysis.output.value): "output",
            }
        operands = []
        reads, writes = [], []
        for position, (formal, value_name) in enumerate(zip(formals, actual_values)):
            formal_name = formal.group("ssa")
            is_read = formal_name in read_formals
            is_write = formal_name in write_formals
            access = (
                "readwrite" if is_read and is_write else "read" if is_read else "write"
            )
            if not is_read and not is_write:
                access = "none"
            if is_read:
                reads.append(value_name)
            if is_write:
                writes.append(value_name)
            value = values[value_name]
            role = role_formals.get(formal_name) or (
                "output" if is_write else f"input{position}"
            )
            compute_dtype = compute_analysis.numeric_type if selected else value.dtype
            operands.append(
                APUv1RegionOperand(
                    position,
                    role,
                    value_name,
                    value.shape,
                    value.dtype,
                    compute_dtype,
                    access,
                )
            )
        dependency_values = {}
        for value_name in reads + writes:
            if value_name in last_writer:
                dependency_values.setdefault(last_writer[value_name], []).append(
                    value_name
                )
        dependencies = tuple(dependency_values)
        region_id = f"region{ordinal}"
        intermediate_names = {
            name for name, value in values.items() if value.intermediate
        }
        region = APUv1RegionManifest(
            region_id,
            function,
            ordinal,
            kind,
            tuple(operands),
            tuple(dict.fromkeys(reads)),
            tuple(dict.fromkeys(writes)),
            tuple(name for name in dict.fromkeys(writes) if name in intermediate_names),
            tuple(
                name
                for name in dict.fromkeys(reads)
                if name in intermediate_names and name in last_writer
            ),
            dependencies,
            storage_analysis=analysis,
            compute_analysis=compute_analysis if selected else None,
            plans=plans,
            selected_plan=selected_plan,
            selected_estimate=selected_estimate,
            physical_plan_verified=selected,
        )
        regions.append(region)
        for value_name in writes:
            last_writer[value_name] = region_id

    # A barrier is an explicit data-dependency frontier, not an assumption
    # that source order alone synchronizes ARC and GVML engines.
    barriers = []
    for region in regions:
        if not region.dependencies:
            continue
        values_crossing = tuple(
            value
            for value in region.reads + region.writes
            if any(
                value in producer.writes
                for producer in regions
                if producer.id in region.dependencies
            )
        )
        kinds = {region.kind} | {
            producer.kind for producer in regions if producer.id in region.dependencies
        }
        reason = "engine_transition" if len(kinds) > 1 else "data_dependency"
        barrier = APUv1BarrierManifest(
            f"barrier{len(barriers)}",
            region.dependencies,
            (region.id,),
            tuple(dict.fromkeys(values_crossing)),
            reason,
        )
        barriers.append(barrier)
        regions[region.ordinal] = replace(region, barrier=barrier.id)

    conversions = []
    for region in regions:
        if region.kind != "vector":
            continue
        compute_dtype = region.compute_analysis.numeric_type
        for operand in region.operands:
            if operand.access not in {"read", "readwrite"}:
                continue
            producer = next(
                (
                    item
                    for item in regions
                    if item.id in region.dependencies and operand.value in item.writes
                ),
                None,
            )
            if producer is not None and producer.kind == "vector":
                continue
            value_manifest = values[operand.value]
            if (
                producer is None
                and operand.role == "output"
                and value_manifest.zero_initialized
            ):
                continue
            if operand.storage_dtype != compute_dtype:
                conversions.append(
                    APUv1ConversionManifest(
                        f"convert{len(conversions)}",
                        operand.value,
                        operand.shape,
                        operand.storage_dtype,
                        compute_dtype,
                        before_region=region.id,
                        reason="program_or_scalar_to_vector",
                    )
                )
        for operand in region.operands:
            if operand.access not in {"write", "readwrite"}:
                continue
            consumers = [
                item
                for item in regions
                if region.id in item.dependencies and operand.value in item.reads
            ]
            value = values[operand.value]
            if value.program_output or any(item.kind == "scalar" for item in consumers):
                if operand.storage_dtype != compute_dtype:
                    conversions.append(
                        APUv1ConversionManifest(
                            f"convert{len(conversions)}",
                            operand.value,
                            operand.shape,
                            compute_dtype,
                            operand.storage_dtype,
                            after_region=region.id,
                            reason="vector_to_scalar_or_program",
                        )
                    )

    phase_manifest = APUv1PhaseManifest(
        phase.phase_name,
        top,
        tuple(regions),
        tuple(barriers),
        tuple(conversions),
        tuple(value.name for value in values.values() if value.program_input),
        tuple(value.name for value in values.values() if value.program_output),
        tuple(value.name for value in values.values() if value.intermediate),
        tuple(getattr(phase, "produces", ())),
        tuple(getattr(phase, "dependencies", ())),
        tuple(dict(getattr(phase, "bindings", {})).items()),
    )
    return APUv1HybridManifest(
        program_name or phase.phase_name,
        tuple(values.values()),
        (phase_manifest,),
        precision,
    )


def _bound_cost(target, cost):
    if cost is None:
        return None
    if hasattr(cost, "target"):
        if cost.target is not target:
            raise ValueError("hybrid cost is bound to another target")
        return cost
    return cost.bind(target)


def _copy_one_cost_subgraph(
    destination,
    source,
    prefix,
    entry_dependencies,
    *,
    apuc=None,
    numerator=1,
    denominator=1,
):
    def scaled(value, activity):
        count = max(1, int(activity.metadata.get("cost_count", 1)))
        if count == 1:
            # A nonempty shard still issues a fixed/single-call operation.
            return int(value)
        shard_count = max(1, math.ceil(count * numerator / denominator))
        return max(0, math.ceil(int(value) * shard_count / count))

    id_map = {activity.id: f"{prefix}:{activity.id}" for activity in source.activities}
    referenced = {
        dependency
        for activity in source.activities
        for dependency in activity.depends_on
    }
    for activity in source.activities:
        internal = tuple(id_map[item] for item in activity.depends_on)
        dependencies = internal or tuple(entry_dependencies)
        occupancy = activity.occupancy
        if apuc is not None:
            occupancy = tuple(
                Occupancy(
                    replace(
                        use.handle,
                        coordinates=tuple(
                            (axis, apuc) if axis == "apuc" else (axis, coordinate)
                            for axis, coordinate in use.handle.coordinates
                        ),
                    ),
                    scaled(use.cycles, activity),
                    use.amount,
                )
                for use in occupancy
            )
        destination.add(
            Activity(
                id_map[activity.id],
                activity.primitive,
                scaled(activity.latency_cycles, activity),
                occupancy,
                dependencies,
                activity.label,
                {**dict(activity.metadata), "hybrid_region": prefix},
            )
        )
    terminals = tuple(
        id_map[activity.id]
        for activity in source.activities
        if activity.id not in referenced
    )
    return terminals or tuple(entry_dependencies)


def _copy_cost_subgraph(destination, source, prefix, entry_dependencies, partition):
    """Shard one full-work plan graph using its physical partition contract.

    Candidate plans describe the complete output domain.  Physical hybrid
    launches scale aggregate repeated work by each declared half-open shard's
    fraction of the partition axis while retaining the same internal DAG.
    """

    terminals = []
    for shard in partition.shards:
        terminals.extend(
            _copy_one_cost_subgraph(
                destination,
                source,
                f"{prefix}:apuc{shard.apuc}",
                entry_dependencies,
                apuc=shard.apuc,
                numerator=shard.extent,
                denominator=partition.extent,
            )
        )
    return tuple(terminals) or tuple(entry_dependencies)


def build_apu_v1_hybrid_execution_graph(manifest, target, cost, partitions=None):
    """Compose selected vector plans, conversions, scalar regions, and barriers."""

    if not isinstance(manifest, APUv1HybridManifest):
        raise TypeError("expected an APUv1HybridManifest")
    bound = _bound_cost(target, cost)
    if bound is None:
        raise ValueError("hybrid execution graph requires an APU v1 cost spec")
    partition_map = dict(partitions or {})
    vector_regions = {
        region.id: region
        for phase in manifest.phases
        for region in phase.regions
        if region.kind == "vector"
    }
    missing_partitions = set(vector_regions) - set(partition_map)
    if missing_partitions:
        raise ValueError(
            "hybrid cost graph requires a physical partition for vector "
            f"regions {sorted(missing_partitions)}"
        )
    for region_id, region in vector_regions.items():
        partition = partition_map[region_id]
        if not isinstance(partition, APUv1RegionPartition):
            raise TypeError(f"partition for {region_id!r} has the wrong type")
        if partition.region_id != region_id:
            raise ValueError("region partition id does not match its vector region")
        if partition.axis != region.compute_analysis.output_axes[0]:
            raise ValueError("region partition axis does not match contraction output")
    graph = ExecutionGraph(
        manifest.name,
        {
            "target": "apu_v1",
            "hybrid": True,
            "region_partitions": {
                region_id: partition_map[region_id].manifest()
                for region_id in vector_regions
            },
            "partition_cost_method": (
                "full-domain repeat counts scaled by shard extent; "
                "single-call primitive latency preserved"
            ),
            "cost_fingerprint": bound.fingerprint,
            "precision_policy": manifest.manifest()["precision_policy"],
        },
    )
    region_terminals = {}
    for logical_phase in manifest.phases:
        before = {}
        after = {}
        for conversion in logical_phase.conversions:
            table = before if conversion.before_region is not None else after
            table.setdefault(
                conversion.before_region or conversion.after_region, []
            ).append(conversion)
        barriers = {item.id: item for item in logical_phase.barriers}
        emitted_barriers = set()
        for region in sorted(logical_phase.regions, key=lambda item: item.ordinal):
            dependencies = tuple(
                terminal
                for logical_id in region.dependencies
                for terminal in region_terminals.get(logical_id, ())
            )
            if region.barrier and region.barrier not in emitted_barriers:
                barrier = barriers[region.barrier]
                barrier_dependencies = tuple(
                    terminal
                    for logical_id in barrier.after_regions
                    for terminal in region_terminals.get(logical_id, ())
                )
                barrier_id = f"hybrid:{logical_phase.name}:{barrier.id}"
                graph.add(
                    Activity(
                        barrier_id,
                        "apu_v1/barrier",
                        0,
                        (),
                        barrier_dependencies,
                        "hybrid_barrier",
                        {
                            "phase": logical_phase.name,
                            "values": barrier.values,
                            "reason": barrier.reason,
                        },
                    )
                )
                emitted_barriers.add(region.barrier)
                dependencies = (barrier_id,)

            conversion_terminals = []
            for conversion in before.get(region.id, ()):
                terminals = []
                elements = max(1, math.prod(conversion.shape))
                for apuc in range(4):
                    event = CostEvent.create(
                        f"hybrid:{conversion.id}:apuc{apuc}",
                        target.op("SCALAR_C"),
                        work_id=(apuc,),
                        metrics={"instructions": (elements + 3) // 4},
                        attributes={
                            "kind": "precision_conversion",
                            "value": conversion.value,
                            "source_dtype": conversion.source_dtype,
                            "target_dtype": conversion.target_dtype,
                        },
                    )
                    terminals.extend(bound.emit(graph, event, dependencies))
                conversion_terminals.extend(terminals)
            entry = tuple(dict.fromkeys(dependencies + tuple(conversion_terminals)))

            if region.kind == "vector":
                estimate = region.selected_estimate or estimate_apu_v1_plan(
                    region.selected_plan, target, bound
                )
                terminals = _copy_cost_subgraph(
                    graph,
                    estimate.graph,
                    f"hybrid:{region.id}",
                    entry,
                    partition_map[region.id],
                )
            else:
                elements = max(
                    (math.prod(operand.shape) for operand in region.operands),
                    default=1,
                )
                event = CostEvent.create(
                    f"hybrid:{region.id}:scalar",
                    target.op("SCALAR_C"),
                    work_id=(0,),
                    metrics={"instructions": elements},
                    attributes={"kind": "arc_scalar", "region": region.id},
                )
                terminals = tuple(bound.emit(graph, event, entry))

            after_terminals = []
            for conversion in after.get(region.id, ()):
                elements = max(1, math.prod(conversion.shape))
                for apuc in range(4):
                    event = CostEvent.create(
                        f"hybrid:{conversion.id}:apuc{apuc}",
                        target.op("SCALAR_C"),
                        work_id=(apuc,),
                        metrics={"instructions": (elements + 3) // 4},
                        attributes={
                            "kind": "precision_conversion",
                            "value": conversion.value,
                            "source_dtype": conversion.source_dtype,
                            "target_dtype": conversion.target_dtype,
                        },
                    )
                    after_terminals.extend(bound.emit(graph, event, terminals))
            region_terminals[region.id] = (
                tuple(after_terminals) if after_terminals else tuple(terminals)
            )
    if not graph.activities:
        raise ValueError("hybrid manifest emitted no cost activities")
    return graph


# ---------------------------------------------------------------------------
# Physical hybrid execution
# ---------------------------------------------------------------------------


def _numpy_dtype(dtype: str) -> np.dtype:
    aliases = {
        "f16": "float16",
        "f32": "float32",
        "f64": "float64",
        "bf16": "float16",
        "i1": "bool",
        "ui1": "bool",
    }
    value = aliases.get(str(dtype), str(dtype))
    if value.startswith("ui"):
        value = "uint" + value[2:]
    elif value.startswith("i") and value[1:].isdigit():
        value = "int" + value[1:]
    return np.dtype(value)


@dataclass(frozen=True)
class APUv1L4Allocation:
    """One persistent physical value in the hybrid program's L4 arena."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    owner: str
    host_input: bool = False
    host_output: bool = False
    intermediate: bool = False
    zero_initialized: bool = False

    def __post_init__(self):
        if not self.name or not self.owner:
            raise ValueError("L4 allocation needs a name and owner")
        shape = tuple(int(item) for item in self.shape)
        if any(item <= 0 for item in shape):
            raise ValueError("L4 allocation shape must be positive")
        _numpy_dtype(self.dtype)
        object.__setattr__(self, "shape", shape)

    @property
    def nbytes(self):
        return math.prod(self.shape) * _numpy_dtype(self.dtype).itemsize

    def manifest(self):
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "owner": self.owner,
            "host_input": self.host_input,
            "host_output": self.host_output,
            "intermediate": self.intermediate,
            "zero_initialized": self.zero_initialized,
            "residency": "persistent_l4",
        }


@dataclass(frozen=True)
class APUv1APUCShard:
    apuc: int
    start: int
    stop: int

    def __post_init__(self):
        if (
            not 0 <= int(self.apuc) < 4
            or int(self.start) < 0
            or int(self.stop) <= int(self.start)
        ):
            raise ValueError("APUC shard needs core 0..3 and a non-empty interval")

    @property
    def extent(self):
        return int(self.stop) - int(self.start)

    def manifest(self):
        return {
            "apuc": int(self.apuc),
            "start": int(self.start),
            "stop": int(self.stop),
            "extent": self.extent,
        }


@dataclass(frozen=True)
class APUv1RegionPartition:
    """Structural four-APUC ownership of one contraction's output-row axis."""

    region_id: str
    axis: str
    extent: int
    shards: tuple[APUv1APUCShard, ...]
    sharded_values: tuple[str, ...]
    replicated_values: tuple[str, ...]
    estimate_scope: str = "full_domain"

    def __post_init__(self):
        shards = tuple(self.shards)
        if not self.region_id or not self.axis or int(self.extent) <= 0:
            raise ValueError("region partition needs an axis and positive extent")
        if tuple(item.apuc for item in shards) != (0, 1, 2, 3):
            raise ValueError("region partition must explicitly cover APUC 0..3")
        if shards[0].start != 0 or shards[-1].stop != int(self.extent):
            raise ValueError("region shards must cover the full axis")
        if any(lhs.stop != rhs.start for lhs, rhs in zip(shards, shards[1:])):
            raise ValueError("region shards must be contiguous and disjoint")
        sharded = tuple(self.sharded_values)
        replicated = tuple(self.replicated_values)
        if set(sharded) & set(replicated):
            raise ValueError("values cannot be both sharded and replicated")
        if self.estimate_scope != "full_domain":
            raise ValueError("selected vector plan estimates are full-domain")
        object.__setattr__(self, "extent", int(self.extent))
        object.__setattr__(self, "shards", shards)
        object.__setattr__(self, "sharded_values", sharded)
        object.__setattr__(self, "replicated_values", replicated)

    @property
    def apucs(self):
        return tuple(item.apuc for item in self.shards)

    def manifest(self):
        return {
            "region_id": self.region_id,
            "axis": self.axis,
            "extent": self.extent,
            "apucs": list(self.apucs),
            "shards": [item.manifest() for item in self.shards],
            "sharded_values": list(self.sharded_values),
            "replicated_values": list(self.replicated_values),
            "estimate_scope": self.estimate_scope,
        }


@dataclass(frozen=True)
class APUv1ShardRealization:
    """One APUC's re-based, validity-masked vector realization.

    ``analysis`` and ``plan`` use a local row axis starting at zero.  The
    global half-open interval remains in ``shard`` so a runner can slice
    sharded L4 values before packing and scatter the result afterwards.
    Replicated values retain their full logical shape.
    """

    partition: APUv1RegionPartition
    shard: APUv1APUCShard
    analysis: ContractionAnalysis
    plan: object
    artifact: object

    def __post_init__(self):
        if self.shard not in self.partition.shards:
            raise ValueError("shard realization is not owned by its partition")
        if self.analysis.axis_extents[self.partition.axis] != self.shard.extent:
            raise ValueError("shard analysis does not use its local valid extent")

    @property
    def valid_extent(self):
        return self.shard.extent

    @property
    def padded_extent(self):
        return 1 << (self.valid_extent - 1).bit_length()

    def value_slice(self, value):
        """Return the global L4 slice consumed or produced by ``value``."""

        accesses = (
            self.analysis.lhs,
            self.analysis.rhs,
            self.analysis.accumulator,
            self.analysis.output,
        )
        access = next((item for item in accesses if item.value == value), None)
        if access is None:
            raise KeyError(value)
        slices = [slice(None)] * len(access.shape)
        if value in self.partition.sharded_values:
            try:
                dimension = access.indices.index(self.partition.axis)
            except ValueError as error:
                raise ValueError(
                    f"sharded value {value!r} does not reference partition axis"
                ) from error
            slices[dimension] = slice(self.shard.start, self.shard.stop)
        return tuple(slices)

    def manifest(self):
        return {
            "region_id": self.partition.region_id,
            "apuc": self.shard.apuc,
            "global_interval": [self.shard.start, self.shard.stop],
            "valid_extent": self.valid_extent,
            "padded_extent": self.padded_extent,
            "plan": self.plan.name,
            "value_shapes": {
                value.name: list(value.shape) for value in self.artifact.values
            },
        }


def partition_apu_v1_region(region):
    """Derive the balanced four-APUC row partition for a vector region."""
    analysis = region.compute_analysis
    axis = analysis.output_axes[0]
    extent = int(analysis.axis_extents[axis])
    boundaries = [extent * index // 4 for index in range(5)]
    # SMALL dimensions are all at least four.  Fail closed rather than launch
    # empty APUCs for a future tiny region without an explicit policy.
    if len(set(boundaries)) != 5:
        raise ValueError(
            f"vector region {region.id!r} cannot use all four APUCs on axis "
            f"{axis!r} extent {extent}"
        )
    shards = tuple(
        APUv1APUCShard(apuc, boundaries[apuc], boundaries[apuc + 1])
        for apuc in range(4)
    )
    access_by_role = {
        "lhs": analysis.lhs,
        "rhs": analysis.rhs,
        "output": analysis.output,
    }
    sharded, replicated = [], []
    for operand in region.operands:
        access = access_by_role.get(operand.role)
        if access is not None and axis in access.indices:
            sharded.append(operand.value)
        else:
            replicated.append(operand.value)
    return APUv1RegionPartition(
        region.id,
        axis,
        extent,
        shards,
        tuple(dict.fromkeys(sharded)),
        tuple(dict.fromkeys(replicated)),
    )


def realize_apu_v1_region_shards(region, partition=None):
    """Replan a vector region as four independent APUC-local realizations.

    The partitioned output axis is re-based to ``[0, shard.extent)``.  Layout
    generation then supplies physical power-of-two padding and validity masks;
    padding is never represented as extra logical work.  Row-indexed operands
    shrink with the shard while row-independent operands (the canonical RHS)
    remain replicated.
    """

    if region.kind != "vector" or region.compute_analysis is None:
        raise ValueError("shard realization requires a vector region")
    partition = partition or partition_apu_v1_region(region)
    if partition.region_id != region.id:
        raise ValueError("partition id does not match its vector region")
    analysis = region.compute_analysis
    if partition.axis != analysis.output_axes[0]:
        raise ValueError("partition axis does not match contraction output")

    from .apu_v1_vector_program import _realize

    result = []
    for shard in partition.shards:

        def local_access(access):
            shape = list(access.shape)
            references_axis = partition.axis in access.indices
            if access.value in partition.sharded_values:
                if not references_axis:
                    raise ValueError(
                        f"sharded value {access.value!r} does not reference "
                        f"axis {partition.axis!r}"
                    )
                for dimension, index in enumerate(access.indices):
                    if index == partition.axis:
                        shape[dimension] = shard.extent
            elif references_axis:
                raise ValueError(
                    f"replicated value {access.value!r} references sharded "
                    f"axis {partition.axis!r}"
                )
            return replace(access, shape=tuple(shape))

        local_analysis = replace(
            analysis,
            axes=tuple(
                (
                    replace(
                        axis,
                        extent=shard.extent,
                        lower_bound=0,
                        upper_bound=shard.extent,
                        step=1,
                    )
                    if axis.name == partition.axis
                    else axis
                )
                for axis in analysis.axes
            ),
            lhs=local_access(analysis.lhs),
            rhs=local_access(analysis.rhs),
            accumulator=local_access(analysis.accumulator),
            output=local_access(analysis.output),
        )
        local_plan = next(
            (
                candidate.plan
                for candidate in generate_apu_v1_vectorization_candidates(
                    local_analysis
                )
                if candidate.name == region.selected_plan.name
            ),
            None,
        )
        if local_plan is None:
            raise ValueError(
                f"selected plan {region.selected_plan.name!r} has no shard-local form"
            )
        artifact, error = _realize(local_analysis, local_plan)
        if artifact is None:
            raise RuntimeError(
                f"APUC {shard.apuc} shard of region {region.id!r} is not "
                f"physically realizable: {error}"
            )
        result.append(
            APUv1ShardRealization(
                partition, shard, local_analysis, local_plan, artifact
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class APUv1RegionExecutable:
    """Physical artifact bound to one retained-MLIR region manifest.

    ``functional`` receives ``(buffers, phase)`` where buffers are physical
    arrays keyed by the region's logical operand names.  Returning a mapping
    replaces named buffers.  Device execution may supply a source fragment and
    function name; handwritten benchmark selection is intentionally absent.
    """

    region_id: str
    artifact: object | None = None
    functional: object | None = None
    device_source: str | None = None
    device_function: str | None = None
    cycles: int | None = None
    partition: APUv1RegionPartition | None = None

    def __post_init__(self):
        if not self.region_id:
            raise ValueError("region executable needs a region id")
        if self.functional is not None and not callable(self.functional):
            raise TypeError("region functional implementation must be callable")
        if self.cycles is not None and int(self.cycles) < 0:
            raise ValueError("region cycles must be non-negative")


@dataclass(frozen=True)
class APUv1PhysicalPhase:
    name: str
    kind: str
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    bindings: tuple[tuple[str, str], ...] = ()
    dependencies: tuple[str, ...] = ()
    apucs: tuple[int, ...] = ()
    executable: APUv1RegionExecutable | None = None
    conversion: APUv1ConversionManifest | None = None
    barrier: APUv1BarrierManifest | None = None

    def __post_init__(self):
        if self.kind not in {"vector", "arc_scalar", "convert", "barrier"}:
            raise ValueError(f"unsupported physical APU phase {self.kind!r}")
        reads, writes = tuple(self.reads), tuple(self.writes)
        apucs = tuple(int(item) for item in self.apucs)
        if len(set(apucs)) != len(apucs) or any(not 0 <= item < 4 for item in apucs):
            raise ValueError("APUC launch set must contain unique indices 0..3")
        if self.kind == "vector" and apucs != (0, 1, 2, 3):
            raise ValueError("vector regions must launch all four APUCs")
        if self.kind == "arc_scalar" and apucs != (0,):
            raise ValueError("scalar ARC regions must launch APUC 0")
        if self.kind == "convert" and apucs != (0, 1, 2, 3):
            raise ValueError("precision conversion must shard over four APUCs")
        if self.kind == "barrier" and apucs:
            raise ValueError("barriers do not launch an APUC")
        if self.kind in {"vector", "arc_scalar"} and self.executable is None:
            raise ValueError(f"{self.kind} phase requires an executable artifact")
        if self.kind == "vector" and self.executable.partition is None:
            raise ValueError("vector phase requires an explicit APUC partition")
        if self.kind == "convert" and self.conversion is None:
            raise ValueError("conversion phase requires its manifest")
        if self.kind == "barrier" and self.barrier is None:
            raise ValueError("barrier phase requires its manifest")
        object.__setattr__(self, "reads", reads)
        object.__setattr__(self, "writes", writes)
        object.__setattr__(self, "bindings", tuple(self.bindings))
        object.__setattr__(self, "dependencies", tuple(self.dependencies))
        object.__setattr__(self, "apucs", apucs)

    def manifest(self):
        return {
            "name": self.name,
            "kind": self.kind,
            "reads": list(self.reads),
            "writes": list(self.writes),
            "bindings": dict(self.bindings),
            "dependencies": list(self.dependencies),
            "apucs": list(self.apucs),
            "blocking_barrier_after_launch": self.kind
            in {"vector", "arc_scalar", "convert"},
            "conversion": self.conversion.manifest() if self.conversion else None,
            "barrier": self.barrier.manifest() if self.barrier else None,
            "partition": (
                self.executable.partition.manifest()
                if self.executable is not None and self.executable.partition is not None
                else None
            ),
        }


@dataclass(frozen=True)
class APUv1PhysicalHybridManifest:
    name: str
    logical: APUv1HybridManifest
    allocations: tuple[APUv1L4Allocation, ...]
    phases: tuple[APUv1PhysicalPhase, ...]

    def __post_init__(self):
        allocations = tuple(self.allocations)
        phases = tuple(self.phases)
        names = [item.name for item in allocations]
        if len(names) != len(set(names)):
            raise ValueError("physical L4 allocation names must be unique")
        known = set(names)
        completed = set()
        phase_names = [phase.name for phase in phases]
        if len(phase_names) != len(set(phase_names)):
            raise ValueError("physical hybrid phase names must be unique")
        initialized = {item.name for item in allocations if item.host_input}
        initialized.update(item.name for item in allocations if item.zero_initialized)
        for phase in phases:
            unknown = set(phase.reads + phase.writes) - known
            if unknown:
                raise ValueError(
                    f"physical phase {phase.name!r} references unknown L4 values "
                    f"{sorted(unknown)}"
                )
            missing = set(phase.dependencies) - completed
            if missing:
                raise ValueError(
                    f"physical phase {phase.name!r} has unsatisfied dependencies "
                    f"{sorted(missing)}"
                )
            uninitialized = set(phase.reads) - initialized
            if uninitialized:
                raise ValueError(
                    f"physical phase {phase.name!r} reads uninitialized L4 values "
                    f"{sorted(uninitialized)}"
                )
            initialized.update(phase.writes)
            completed.add(phase.name)
        missing_outputs = {
            item.name for item in allocations if item.host_output
        } - initialized
        if missing_outputs:
            raise ValueError(
                f"hybrid program leaves host outputs uninitialized: "
                f"{sorted(missing_outputs)}"
            )
        object.__setattr__(self, "allocations", allocations)
        object.__setattr__(self, "phases", phases)

    @property
    def persistent_l4_bytes(self):
        return sum(item.nbytes for item in self.allocations)

    def manifest(self):
        return {
            "name": self.name,
            "persistent_l4_bytes": self.persistent_l4_bytes,
            "host_intermediate_round_trips": 0,
            "allocations": [item.manifest() for item in self.allocations],
            "phases": [item.manifest() for item in self.phases],
        }


def _conversion_buffer_name(conversion, *, source=False):
    dtype = conversion.source_dtype if source else conversion.target_dtype
    return f"{conversion.value}__{dtype}__{conversion.id}"


def lower_apu_v1_hybrid_manifest(manifest, executables):
    """Bind logical regions to artifacts and materialize persistent L4 phases."""

    if not isinstance(manifest, APUv1HybridManifest):
        raise TypeError("expected an APUv1HybridManifest")
    executable_map = {
        key: (
            value
            if isinstance(value, APUv1RegionExecutable)
            else APUv1RegionExecutable(key, artifact=value)
        )
        for key, value in dict(executables).items()
    }
    value_map = {item.name: item for item in manifest.values}
    allocations = {
        item.name: APUv1L4Allocation(
            item.name,
            item.shape,
            item.dtype,
            "host" if item.program_input else "program",
            item.program_input,
            item.program_output,
            item.intermediate,
            item.zero_initialized,
        )
        for item in manifest.values
    }
    current = {name: name for name in value_map}
    latest_writer = {
        name: None
        for name, allocation in allocations.items()
        if allocation.host_input or allocation.zero_initialized
    }
    region_terminals = {}
    physical = []
    emitted_barriers = set()

    for logical_phase in manifest.phases:
        conversions_before = {}
        conversions_after = {}
        for conversion in logical_phase.conversions:
            mapping = (
                conversions_before
                if conversion.before_region is not None
                else conversions_after
            )
            key = conversion.before_region or conversion.after_region
            mapping.setdefault(key, []).append(conversion)
        barrier_map = {item.id: item for item in logical_phase.barriers}

        for region in sorted(logical_phase.regions, key=lambda item: item.ordinal):
            entry_dependencies = []
            if region.barrier and region.barrier not in emitted_barriers:
                barrier = barrier_map[region.barrier]
                barrier_dependencies = tuple(
                    dict.fromkeys(
                        terminal
                        for logical_id in barrier.after_regions
                        for terminal in region_terminals.get(logical_id, (logical_id,))
                    )
                )
                physical.append(
                    APUv1PhysicalPhase(
                        barrier.id,
                        "barrier",
                        reads=tuple(current[value] for value in barrier.values),
                        dependencies=barrier_dependencies,
                        barrier=barrier,
                    )
                )
                emitted_barriers.add(barrier.id)
                entry_dependencies.append(barrier.id)
            else:
                entry_dependencies.extend(
                    terminal
                    for logical_id in region.dependencies
                    for terminal in region_terminals.get(logical_id, (logical_id,))
                )

            before_conversion_ids = []
            for conversion in conversions_before.get(region.id, ()):
                source = current[conversion.value]
                destination = _conversion_buffer_name(conversion)
                allocations.setdefault(
                    destination,
                    APUv1L4Allocation(
                        destination,
                        conversion.shape,
                        conversion.target_dtype,
                        conversion.id,
                        intermediate=True,
                    ),
                )
                physical.append(
                    APUv1PhysicalPhase(
                        conversion.id,
                        "convert",
                        (source,),
                        (destination,),
                        ((conversion.value, destination),),
                        tuple(
                            dict.fromkeys(
                                entry_dependencies
                                + (
                                    [latest_writer[source]]
                                    if latest_writer.get(source) is not None
                                    else []
                                )
                            )
                        ),
                        apucs=(0, 1, 2, 3),
                        conversion=conversion,
                    )
                )
                current[conversion.value] = destination
                latest_writer[destination] = conversion.id
                before_conversion_ids.append(conversion.id)

            bindings = {name: current[name] for name in region.reads + region.writes}
            # A write-only vector result has no ingress conversion but still
            # needs compute-precision storage before its explicit egress.
            if region.kind == "vector":
                for operand in region.operands:
                    if operand.access not in {"write", "readwrite"}:
                        continue
                    if operand.storage_dtype == operand.compute_dtype:
                        continue
                    existing = bindings.get(operand.value)
                    if existing == operand.value:
                        compute_name = (
                            f"{operand.value}__{operand.compute_dtype}__resident"
                        )
                        allocations.setdefault(
                            compute_name,
                            APUv1L4Allocation(
                                compute_name,
                                operand.shape,
                                operand.compute_dtype,
                                region.id,
                                intermediate=True,
                                zero_initialized=value_map[
                                    operand.value
                                ].zero_initialized,
                            ),
                        )
                        if allocations[compute_name].zero_initialized:
                            latest_writer.setdefault(compute_name, None)
                        bindings[operand.value] = compute_name
                        current[operand.value] = compute_name

            executable = executable_map.get(region.id)
            if executable is None:
                raise ValueError(f"no executable artifact for region {region.id!r}")
            if region.kind == "vector" and executable.partition is None:
                raise ValueError(
                    f"vector region {region.id!r} requires an explicit APUC partition"
                )
            region_dependencies = list(entry_dependencies) + before_conversion_ids
            region_dependencies.extend(
                latest_writer[name]
                for name in tuple(bindings[value] for value in region.reads)
                if latest_writer.get(name) is not None
            )
            physical.append(
                APUv1PhysicalPhase(
                    region.id,
                    "vector" if region.kind == "vector" else "arc_scalar",
                    tuple(bindings[name] for name in region.reads),
                    tuple(bindings[name] for name in region.writes),
                    tuple(bindings.items()),
                    tuple(dict.fromkeys(region_dependencies)),
                    (executable.partition.apucs if region.kind == "vector" else (0,)),
                    executable,
                )
            )
            for physical_name in tuple(bindings[name] for name in region.writes):
                latest_writer[physical_name] = region.id

            terminal_ids = []
            for conversion in conversions_after.get(region.id, ()):
                source = current[conversion.value]
                if _numpy_dtype(conversion.source_dtype) != _numpy_dtype(
                    allocations[source].dtype
                ):
                    source = bindings[conversion.value]
                destination = conversion.value
                physical.append(
                    APUv1PhysicalPhase(
                        conversion.id,
                        "convert",
                        (source,),
                        (destination,),
                        ((conversion.value, destination),),
                        (region.id,),
                        (0, 1, 2, 3),
                        conversion=conversion,
                    )
                )
                current[conversion.value] = destination
                latest_writer[destination] = conversion.id
                terminal_ids.append(conversion.id)
            region_terminals[region.id] = tuple(terminal_ids) or (region.id,)

    used_allocations = {
        name for phase in physical for name in phase.reads + phase.writes
    }
    used_allocations.update(
        item.name
        for item in allocations.values()
        if item.host_input or item.host_output
    )
    retained_allocations = tuple(
        item for name, item in allocations.items() if name in used_allocations
    )
    return APUv1PhysicalHybridManifest(
        manifest.name, manifest, retained_allocations, tuple(physical)
    )


def emit_apu_v1_hybrid_host_source(physical):
    """Emit the explicit persistent-L4 launch/barrier skeleton for the GDL host."""

    lines = [
        "/* All L4 allocations below live for the complete hybrid program. */",
        f"/* persistent_l4_bytes={physical.persistent_l4_bytes} */",
        "static int run_hybrid_program(gdl_context_handle_t ctx) {",
        "    int ret = 0;",
    ]
    for allocation in physical.allocations:
        lines.append(
            f"    gdl_mem_handle_t l4_{allocation.name} = "
            f"gdl_mem_alloc_aligned(ctx, {allocation.nbytes}, "
            "GDL_CONST_MAPPED_POOL, GDL_ALIGN_32);"
        )
        if allocation.host_input:
            lines.append(f"    /* host_to_l4_once: {allocation.name} */")
    for ordinal, phase in enumerate(physical.phases):
        if phase.kind == "barrier":
            lines.append(
                f"    /* {phase.name}: explicit host launch barrier; L4 retained */"
            )
            continue
        lines.append(
            f"    ret = schedule_hybrid_phase(ctx, {ordinal}, {len(phase.apucs)}); "
            f"/* {phase.name}:{phase.kind} apucs={phase.apucs} */"
        )
        lines.append("    if (ret) return ret; /* blocking batch = barrier */")
    for allocation in physical.allocations:
        if allocation.host_output:
            lines.append(f"    /* l4_to_host_once: {allocation.name} */")
    lines.extend(["    return ret;", "}"])
    return "\n".join(lines)


def emit_apu_v1_hybrid_device_source(physical):
    """Compose retained region fragments and explicit conversion dispatch.

    The emitted source is an inspectable physical contract.  A project runner
    binds the named L4 handles and ``phase_id``/``core_id`` fields when it
    materializes the final GDL task ABI.
    """

    fragments = [
        "#include <stdint.h>",
        "#include <stddef.h>",
        "/* Hybrid regions share persistent L4 handles across host barriers. */",
    ]
    seen = set()
    for phase in physical.phases:
        executable = phase.executable
        if executable is None:
            continue
        source = executable.device_source
        artifact = executable.artifact
        if source is None and getattr(artifact, "realization", None) is not None:
            source = artifact.realization.device_source()
        if source and source not in seen:
            fragments.append(source)
            seen.add(source)
    fragments.extend(
        [
            "static uint16_t allo_f32_to_f16_rne_finite(float value);",
            "static float allo_f16_to_f32(uint16_t value);",
            "",
            "static int run_hybrid_phase(uint32_t phase_id, uint32_t core_id) {",
            "    switch (phase_id) {",
        ]
    )
    for ordinal, phase in enumerate(physical.phases):
        if phase.kind == "barrier":
            continue
        fragments.append(
            f"    case {ordinal}: /* {phase.name}:{phase.kind}; "
            f"apucs={phase.apucs}; reads={phase.reads}; writes={phase.writes} */"
        )
        if phase.kind == "convert":
            fragments.append(
                "        /* Four-way ARC shard performs explicit "
                f"{phase.conversion.source_dtype}->{phase.conversion.target_dtype} "
                f"({phase.conversion.reason}). */"
            )
        elif phase.kind == "arc_scalar":
            fragments.append("        if (core_id != 0) return 0;")
        fragments.append("        return 0;")
    fragments.extend(
        [
            "    default: return -1;",
            "    }",
            "}",
        ]
    )
    return "\n".join(fragments)


def _functional_conversion(value, conversion, precision_policy):
    target = _numpy_dtype(conversion.target_dtype)
    source = np.asarray(value)
    if conversion.source_dtype == "f32" and conversion.target_dtype == "f16":
        if precision_policy is None:
            raise RuntimeError("f32->f16 conversion requires a precision policy")
        if precision_policy.rounding != "nearest_even":
            raise NotImplementedError(
                f"unsupported APU conversion rounding {precision_policy.rounding!r}"
            )
        if precision_policy.overflow != "finite":
            raise NotImplementedError(
                f"unsupported APU conversion overflow {precision_policy.overflow!r}"
            )
        limit = np.finfo(np.float16).max
        source = np.clip(source, -limit, limit)
    return source.astype(target, copy=True)


def _vector_region_function(region):
    analysis = region.compute_analysis
    axes = []
    for access in (analysis.lhs, analysis.rhs, analysis.output):
        for axis in access.indices:
            if axis not in axes:
                axes.append(axis)
    symbols = {axis: chr(ord("a") + index) for index, axis in enumerate(axes)}
    lhs_subscript = "".join(symbols[axis] for axis in analysis.lhs.indices)
    rhs_subscript = "".join(symbols[axis] for axis in analysis.rhs.indices)
    output_subscript = "".join(symbols[axis] for axis in analysis.output.indices)
    expression = f"{lhs_subscript},{rhs_subscript}->{output_subscript}"
    role_values = {operand.role: operand.value for operand in region.operands}
    lhs_name = role_values["lhs"]
    rhs_name = role_values["rhs"]
    output_name = role_values["output"]

    def execute(buffers, _phase):
        contribution = np.einsum(
            expression, buffers[lhs_name], buffers[rhs_name]
        ).astype(buffers[output_name].dtype)
        buffers[output_name][:] = (buffers[output_name] + contribution).astype(
            buffers[output_name].dtype
        )

    return execute


def _scalar_region_function(artifact, region):
    compiled = None

    def execute(buffers, _phase):
        nonlocal compiled
        if compiled is None:
            compiled = artifact.compile()
        arguments = [buffers[operand.value] for operand in region.operands]
        returned = compiled(*arguments)
        if returned is None:
            return None
        writes = [
            operand.value
            for operand in region.operands
            if operand.access in {"write", "readwrite"}
        ]
        if len(writes) == 1:
            return {writes[0]: returned}
        if isinstance(returned, tuple) and len(returned) == len(writes):
            return dict(zip(writes, returned))
        raise RuntimeError(
            f"scalar region {region.id!r} returned values inconsistent with writes"
        )

    return execute


def build_apu_v1_region_executables(manifest, schedule):
    """Build physical artifacts directly from each retained-MLIR region."""

    from ..backend.c import emit_c_from_mlir
    from .apu_v1_vector_program import _realize

    executables = {}
    for phase in manifest.phases:
        for region in phase.regions:
            if region.kind == "vector":
                realization, error = _realize(
                    region.compute_analysis, region.selected_plan
                )
                if realization is None:
                    raise RuntimeError(
                        f"selected vector region {region.id!r} is not executable: {error}"
                    )
                executables[region.id] = APUv1RegionExecutable(
                    region.id,
                    artifact=realization,
                    functional=_vector_region_function(region),
                    device_source=realization.device_source(),
                    device_function=(
                        re.sub(r"\W", "_", region.selected_plan.name) + "_vector"
                    ),
                    cycles=(
                        int(region.selected_estimate.cycles)
                        if region.selected_estimate is not None
                        else None
                    ),
                    partition=partition_apu_v1_region(region),
                )
            else:
                artifact = emit_c_from_mlir(
                    schedule.module,
                    region.function,
                    wrap_wide_integers=True,
                )
                executables[region.id] = APUv1RegionExecutable(
                    region.id,
                    artifact=artifact,
                    functional=_scalar_region_function(artifact, region),
                    device_source=artifact.c_source,
                    device_function=region.function,
                )
    return MappingProxyType(executables)


class APUv1HybridCallable:
    """Execute one physical hybrid manifest functionally or on a device runner."""

    def __init__(self, physical, *, backend="functional", device_runner=None):
        self.physical_manifest = physical
        self.hybrid_manifest = physical.logical
        self.backend = str(backend)
        if self.backend not in {"functional", "virtual", "device"}:
            raise ValueError("hybrid backend must be functional, virtual, or device")
        self.device_runner = device_runner
        self.input_names = tuple(
            item.name for item in physical.allocations if item.host_input
        )
        self.output_names = tuple(
            item.name for item in physical.allocations if item.host_output
        )
        self.signature = inspect.Signature(
            inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for name in self.input_names
        )
        self.__signature__ = self.signature
        self.last_result = None

    def host_source(self):
        return emit_apu_v1_hybrid_host_source(self.physical_manifest)

    def device_source(self):
        return emit_apu_v1_hybrid_device_source(self.physical_manifest)

    def __call__(self, *args, **kwargs):
        bound = self.signature.bind(*args, **kwargs)
        if self.backend == "device":
            if self.device_runner is None:
                raise RuntimeError(
                    "hybrid device execution requires a physical GDL project runner"
                )
            result = self.device_runner(self, dict(bound.arguments))
        else:
            result = self._run_functional(dict(bound.arguments))
        self.last_result = result
        return result

    run = __call__

    def _run_functional(self, inputs):
        allocations = {item.name: item for item in self.physical_manifest.allocations}
        state = {}
        for allocation in allocations.values():
            if allocation.host_input:
                value = inputs[allocation.name]
                if not isinstance(value, np.ndarray):
                    raise TypeError(f"{allocation.name} must be a NumPy array")
                expected = _numpy_dtype(allocation.dtype)
                if value.shape != allocation.shape or value.dtype != expected:
                    raise TypeError(
                        f"{allocation.name} must be {expected}{allocation.shape}, "
                        f"got {value.dtype}{value.shape}"
                    )
                state[allocation.name] = np.ascontiguousarray(value).copy()
            else:
                state[allocation.name] = np.zeros(
                    allocation.shape, dtype=_numpy_dtype(allocation.dtype)
                )

        trace = []
        total_cycles = 0
        for phase in self.physical_manifest.phases:
            if phase.kind == "barrier":
                trace.append({"phase": phase.name, "kind": "barrier", "cycles": 0})
                continue
            if phase.kind == "convert":
                source, destination = phase.reads[0], phase.writes[0]
                state[destination] = _functional_conversion(
                    state[source],
                    phase.conversion,
                    self.hybrid_manifest.precision_policy,
                )
                trace.append(
                    {
                        "phase": phase.name,
                        "kind": "convert",
                        "apucs": phase.apucs,
                        "cycles": 0,
                    }
                )
                continue

            executable = phase.executable
            buffers = {logical: state[physical] for logical, physical in phase.bindings}
            functional = executable.functional
            if functional is None:
                artifact = executable.artifact
                if hasattr(artifact, "workload"):
                    functional = lambda values, _phase: artifact.workload(**values)
                elif callable(artifact):
                    functional = lambda values, _phase: artifact(**values)
                else:
                    raise RuntimeError(
                        f"region {phase.name!r} has no functional implementation"
                    )
            returned = functional(buffers, phase)
            if isinstance(returned, Mapping):
                for logical, value in returned.items():
                    physical_name = dict(phase.bindings)[logical]
                    state[physical_name] = np.asarray(value)
            phase_cycles = int(executable.cycles or 0)
            total_cycles += phase_cycles
            trace.append(
                {
                    "phase": phase.name,
                    "kind": phase.kind,
                    "apucs": phase.apucs,
                    "cycles": phase_cycles,
                }
            )

        outputs = {name: state[name].copy() for name in self.output_names}
        for name, value in outputs.items():
            if name in inputs and inputs[name].shape == value.shape:
                np.copyto(inputs[name], value, casting="same_kind")
        return RunResult(
            total_cycles,
            "functional execution of explicit APU v1 hybrid phase manifest",
            f"apu_v1-{self.backend}",
            {
                "outputs": outputs,
                "phase_trace": tuple(trace),
                "persistent_l4_bytes": self.physical_manifest.persistent_l4_bytes,
                "host_intermediate_round_trips": 0,
                "host_source": self.host_source(),
                "device_source": self.device_source(),
            },
        )


def compile_apu_v1_hybrid(
    manifest, executables, target, *, backend="functional", device_runner=None
):
    if getattr(target, "name", None) != "apu_v1":
        raise ValueError("hybrid APU execution requires the apu_v1 target")
    physical = lower_apu_v1_hybrid_manifest(manifest, executables)
    return APUv1HybridCallable(physical, backend=backend, device_runner=device_runner)


__all__ = [
    "APUv1BarrierManifest",
    "APUv1APUCShard",
    "APUv1ConversionManifest",
    "APUv1HybridManifest",
    "APUv1HybridCallable",
    "APUv1L4Allocation",
    "APUv1PhysicalHybridManifest",
    "APUv1PhysicalPhase",
    "APUv1PhaseManifest",
    "APUv1PrecisionPolicy",
    "APUv1RegionManifest",
    "APUv1RegionOperand",
    "APUv1RegionPartition",
    "APUv1ShardRealization",
    "APUv1RegionExecutable",
    "APUv1ValueManifest",
    "build_apu_v1_hybrid_execution_graph",
    "compile_apu_v1_hybrid",
    "build_apu_v1_region_executables",
    "discover_apu_v1_hybrid_manifest",
    "emit_apu_v1_hybrid_device_source",
    "emit_apu_v1_hybrid_host_source",
    "lower_apu_v1_hybrid_manifest",
    "partition_apu_v1_region",
    "realize_apu_v1_region_shards",
]
