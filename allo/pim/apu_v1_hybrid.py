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
import re
from types import MappingProxyType
from typing import Mapping

from .apu_v1_vector_cost import rank_apu_v1_plans
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


__all__ = [
    "APUv1BarrierManifest",
    "APUv1ConversionManifest",
    "APUv1HybridManifest",
    "APUv1PhaseManifest",
    "APUv1PrecisionPolicy",
    "APUv1RegionManifest",
    "APUv1RegionOperand",
    "APUv1ValueManifest",
    "discover_apu_v1_hybrid_manifest",
]
