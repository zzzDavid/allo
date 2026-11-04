# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retained-IR discovery and immutable planning for APU v1 scalar programs.

The manifest of an ``APUv1Program`` phase (its values, the scalar regions its
top function calls, their operand access directions, the data dependencies
between them) is read from the retained MLIR module through the IR bindings:
function signatures and argument types, named allocations and their
``linalg.fill`` initializers, ``func.call`` operations in program order and
the loads and stores of each callee on its formals.  No MLIR text is parsed.
This module stops at the logical execution manifest; physical GVML/ARC source
generation consumes the manifest elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import inspect
import re
from typing import Mapping

from .apu_v1_vectorize import ContractionAnalysis


_DTYPE = r"bf16|f16|f32|f64|i\d+|ui\d+"
_DTYPE_RE = re.compile(_DTYPE)


def _ir():
    from .._mlir import ir

    return ir


def _attr_text(attribute) -> str:
    ir = _ir()
    for kind in (ir.StringAttr, ir.FlatSymbolRefAttr):
        try:
            return kind(attribute).value
        except (TypeError, ValueError):
            continue
    return str(attribute).strip('"').lstrip("@")


def _module_functions(module) -> Mapping[str, object]:
    """``{symbol name: func.func operation}`` of the retained module."""
    functions = {}
    for operation in module.body.operations:
        if operation.operation.name != "func.func":
            continue
        name = _attr_text(operation.attributes["sym_name"])
        if name in functions:
            raise ValueError(f"duplicate MLIR function {name!r}")
        functions[name] = operation
    return functions


def _walk(block, out):
    for operation in block.operations:
        out.append(operation)
        for region in operation.regions:
            for nested in region.blocks:
                _walk(nested, out)
    return out


def _memref_parts(mlir_type) -> tuple[tuple[int, ...], str]:
    """``(shape, element dtype text)`` of a ranked memref type; a type that
    is not a ranked memref of a scalar element is refused."""
    ir = _ir()
    if not ir.MemRefType.isinstance(mlir_type):
        raise ValueError(
            f"hybrid APU manifest requires a ranked memref, got {mlir_type}"
        )
    memref = ir.MemRefType(mlir_type)
    dtype = str(memref.element_type)
    if not _DTYPE_RE.fullmatch(dtype):
        raise ValueError(
            f"hybrid APU manifest requires a scalar element type, got {dtype}"
        )
    return tuple(int(d) for d in memref.shape), dtype


def _entry_block(function):
    return function.regions[0].blocks[0]


def _formal_number(function, value) -> int | None:
    """The position of ``value`` among ``function``'s formals, else None."""
    ir = _ir()
    if not ir.BlockArgument.isinstance(value):
        return None
    argument = ir.BlockArgument(value)
    if argument.owner != _entry_block(function):
        return None
    return argument.arg_number


def _formal_accesses(function) -> tuple[set, set]:
    """``(read formals, written formals)`` by position: the formals the
    function's loads read and its stores write (nested loops included)."""
    reads, writes = set(), set()
    for operation in _walk(_entry_block(function), []):
        name = operation.operation.name
        if name in ("affine.load", "memref.load"):
            position = _formal_number(function, operation.operands[0])
            if position is not None:
                reads.add(position)
        elif name in ("affine.store", "memref.store"):
            position = _formal_number(function, operation.operands[1])
            if position is not None:
                writes.add(position)
    return reads, writes


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


def discover_apu_v1_hybrid_manifest(
    phase,
    schedule,
    artifact,
    target,
    *,
    cost=None,
    program_name=None,
):
    """Discover the ordered scalar regions of one phase from its retained IR.

    Values are the top function's formals (the phase's source arguments, in
    signature order, typed by the artifact's ABI), its named allocations
    (intermediates; zero-initialized when a ``linalg.fill`` initializes them)
    and the returned buffers (``phase.result_names``).  Regions are the top
    function's ``func.call`` operations in program order, each callee's
    formals bound to the call's actual values with the access direction the
    callee's loads and stores give them; a top function without calls is
    one scalar region over the program's inputs and outputs.  Dependencies
    follow the last writer of every value a region reads or writes.
    """
    del cost, target
    ir = _ir()
    from ..spmw_match_engine import ValueNames

    functions = _module_functions(schedule.module)
    top = schedule.top_func_name
    if top not in functions:
        raise ValueError(f"retained MLIR has no top function {top!r}")
    precision = getattr(phase, "precision_policy", None)
    vectorize = getattr(phase, "vectorize", False)
    if vectorize not in {False, "required"}:
        raise ValueError("APUv1Phase.vectorize must be False or 'required'")

    if vectorize == "required":
        # Required-vector phases are proven by the native recipes
        # (`apu_v1_native_vector.discover_apu_v1_native_vector_lowering`) before
        # this manifest is discovered; the hybrid vector regions and their own
        # MLIR-text contraction analysis were removed (release round 14).
        # Ordinary contractions compile through `allo.compile`.
        raise NotImplementedError(
            "APUv1Phase(vectorize='required') is proven by the native recipes; "
            "hybrid vector regions were removed. Compile a contraction as a plain "
            "@allo function through allo.compile for the APU v1 vector plans."
        )

    source_names = tuple(inspect.signature(phase.kernel).parameters)
    source_abi = [item for item in artifact.arguments if item.source == "argument"]
    result_abi = [item for item in artifact.arguments if item.source == "result"]
    if len(source_names) != len(source_abi):
        raise ValueError("hybrid manifest source ABI does not match Python signature")
    if len(phase.result_names) != len(result_abi):
        raise ValueError("hybrid manifest result_names do not match returned memrefs")

    top_function = functions[top]
    names = ValueNames(top_function, ir.AsmState(top_function))
    formal_top = tuple(_entry_block(top_function).arguments)
    if len(formal_top) != len(source_names):
        raise ValueError("hybrid manifest top-level MLIR argument mismatch")
    value_of_ssa = {}
    values = {}
    bindings = dict(getattr(phase, "bindings", {}))
    for formal, source_name, abi in zip(formal_top, source_names, source_abi):
        name = bindings.get(source_name, source_name)
        shape, _signless_dtype = _memref_parts(formal.type)
        # CArgument decodes the function-level itypes string, which is the
        # authoritative signedness for signless MLIR integer formals.
        dtype = abi.dtype
        value_of_ssa[names(formal)] = name
        values[name] = APUv1ValueManifest(
            name,
            shape,
            dtype,
            program_input=abi.mode in {"in", "both"},
            program_output=abi.mode in {"out", "both"},
        )
    top_ops = _walk(_entry_block(top_function), [])
    filled = {
        names(operation.operands[-1])
        for operation in top_ops
        if operation.operation.name == "linalg.fill" and len(operation.operands) >= 2
    }
    explicit_zero = set(getattr(phase, "zero_initialize", ()))
    for operation in top_ops:
        if (
            operation.operation.name != "memref.alloc"
            or "name" not in operation.attributes
        ):
            continue
        result = operation.results[0]
        try:
            shape, dtype = _memref_parts(result.type)
        except ValueError:
            # MLIR lowering also introduces scalar index slots.  They are an
            # ARC implementation detail, not persistent tensor values.
            continue
        if dtype.startswith("i") and "unsigned" in operation.attributes:
            dtype = "u" + dtype
        name = _attr_text(operation.attributes["name"])
        ssa = names(result)
        value_of_ssa[ssa] = name
        values[name] = APUv1ValueManifest(
            name,
            shape,
            dtype,
            intermediate=True,
            zero_initialized=(ssa in filled or name in explicit_zero),
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
    for operation in top_ops:
        if operation.operation.name != "func.call":
            continue
        callee = _attr_text(operation.attributes["callee"])
        actual_values = []
        for operand in operation.operands:
            ssa = names(operand)
            if ssa not in value_of_ssa:
                raise ValueError(f"call uses unknown top-level SSA value {ssa}")
            actual_values.append(value_of_ssa[ssa])
        calls.append((callee, tuple(actual_values)))

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
    intermediate_names = {name for name, value in values.items() if value.intermediate}
    for ordinal, (function_name, actual_values) in enumerate(calls):
        if function_name not in functions:
            raise ValueError(
                f"top-level call references unknown function {function_name!r}"
            )
        function = functions[function_name]
        formals = tuple(_entry_block(function).arguments)
        if len(formals) != len(actual_values):
            raise ValueError(f"call ABI mismatch for region function {function_name!r}")
        read_formals, write_formals = _formal_accesses(function)
        # Every region is a scalar ARC region (vector regions were removed).
        operands = []
        reads, writes = [], []
        for position, value_name in enumerate(actual_values):
            is_read = position in read_formals
            is_write = position in write_formals
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
            role = "output" if is_write else f"input{position}"
            operands.append(
                APUv1RegionOperand(
                    position,
                    role,
                    value_name,
                    value.shape,
                    value.dtype,
                    value.dtype,
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
        region = APUv1RegionManifest(
            region_id,
            function_name,
            ordinal,
            "scalar",
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

    phase_manifest = APUv1PhaseManifest(
        phase.phase_name,
        top,
        tuple(regions),
        tuple(barriers),
        (),
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
