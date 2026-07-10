# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""MLIR-driven functional programs for the UPMEM backend.

An :class:`UPMEMProgram` is an ordered sequence of MLIR-lowered phases.  It is
separate from the SPMW operation matcher: general control flow remains in MLIR
and is emitted through Allo's portable C backend.  The host shared-library
execution path is a functional oracle only and intentionally reports no device
or simulator cycles.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import inspect
import json
import math
import re
import shutil

import numpy as np

from ..backend.c import CArgument, emit_c_from_mlir
from ..customize import customize
from ..perf import BoundCostSpec, CostEvent
from ..perf.graph import ExecutionGraph
from ..spmw_linear_layout import LinearLayout
from ..spmw_codegen import RunResult
from .upmem_abi import (
    LaunchABI,
    ProgramABI as DeviceProgramABI,
    ScalarABI,
    TensorABI,
    TensorDirection,
    TensorLayout,
)
from .upmem_analysis import analyze_upmem_mlir
from .schedule_promotion import validate_schedule_promotion_gate
from .schedule_search import (
    DecisionDomain,
    InfeasibleSchedule,
    ScheduleObjectiveDomain,
    grid_search,
    guarded_schedule_activation,
)


_UPMEM_WRAM_BYTES = 64 * 1024
_UPMEM_MRAM_BYTES = 64 * 1024 * 1024
_UPMEM_MIN_DMA_BYTES = 8
_UPMEM_MAX_DMA_BYTES = 2048


def _validate_upmem_dma_size(size: int, name: str) -> None:
    if not _UPMEM_MIN_DMA_BYTES <= size <= _UPMEM_MAX_DMA_BYTES:
        raise ValueError(f"{name} MRAM DMA size must be between 8 and 2048 bytes")
    if size % 8:
        raise ValueError(f"{name} MRAM DMA size must be 8-byte aligned")


def _validate_upmem_mram(offsets, image_bytes: int) -> None:
    if any(offset % 8 for offset in offsets):
        raise ValueError("UPMEM MRAM offsets must be 8-byte aligned")
    if image_bytes > _UPMEM_MRAM_BYTES:
        raise ValueError("UPMEM tile exceeds the 64 MiB MRAM capacity per DPU")


def _merge_modes(lhs: str, rhs: str) -> str:
    if lhs == rhs:
        return lhs
    if "both" in (lhs, rhs):
        return "both"
    if {lhs, rhs} <= {"in", "out", "func"}:
        return (
            "both" if {lhs, rhs} == {"in", "out"} else (rhs if lhs == "func" else lhs)
        )
    if "scalar" in (lhs, rhs):
        return "scalar"
    return rhs


@dataclass(frozen=True)
class ProgramArgument:
    """One named array/scalar in a UPMEM program's public NumPy ABI."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    mode: str

    def manifest(self) -> dict:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "mode": self.mode,
        }


@dataclass(frozen=True)
class UPMEMParallelRegion:
    """Canonical, dependence-safe loop frontier retained from MLIR structure."""

    site_id: str
    loop_ordinal: int
    lower_bound: int
    upper_bound: int
    step: int

    @property
    def trip_count(self) -> int:
        distance = self.upper_bound - self.lower_bound
        return 0 if distance <= 0 else (distance + self.step - 1) // self.step

    def manifest(self) -> dict:
        return {
            "site_id": self.site_id,
            "loop_ordinal": self.loop_ordinal,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "step": self.step,
            "trip_count": self.trip_count,
        }


@dataclass(frozen=True)
class PhaseABI:
    """Stable ABI and parallel-frontier description for one phase."""

    name: str
    arguments: tuple[ProgramArgument, ...]
    parallel_regions: tuple[UPMEMParallelRegion, ...] = ()
    oracle_parallel_workers: int = 1

    def manifest(self) -> dict:
        return {
            "name": self.name,
            "arguments": [argument.manifest() for argument in self.arguments],
            "parallel_regions": [region.manifest() for region in self.parallel_regions],
            "oracle_parallel_workers": self.oracle_parallel_workers,
        }


@dataclass(frozen=True)
class MLIRProgramABI:
    """JSON-serializable manifest for a complete phased program."""

    name: str
    arguments: tuple[ProgramArgument, ...]
    phases: tuple[PhaseABI, ...]

    def manifest(self) -> dict:
        return {
            "name": self.name,
            "arguments": [argument.manifest() for argument in self.arguments],
            "phases": [phase.manifest() for phase in self.phases],
        }


@dataclass(frozen=True)
class UPMEMPhase:
    """One serial launch phase lowered from an Allo callable through MLIR.

    Parallel frontiers are discovered from retained MLIR structure.  The
    ``parallel_loops`` compatibility field rejects non-empty display labels so
    workload names cannot select scheduling or realizability decisions.
    """

    kernel: object
    name: str | None = None
    instantiate: tuple = ()
    result_names: tuple[str, ...] = ()
    parallel_loops: tuple[str, ...] = ()
    parallel_workers: int = 64

    def __post_init__(self):
        if not callable(self.kernel):
            raise TypeError("UPMEMPhase kernel must be callable")
        if not 1 <= int(self.parallel_workers) <= 64:
            raise ValueError("UPMEMPhase parallel_workers must be between 1 and 64")
        object.__setattr__(self, "instantiate", tuple(self.instantiate))
        object.__setattr__(self, "result_names", tuple(self.result_names))
        object.__setattr__(self, "parallel_loops", tuple(self.parallel_loops))
        if self.parallel_loops:
            raise ValueError(
                "display-string UPMEM parallel_loops are no longer supported; "
                "parallel regions are derived from retained MLIR structure"
            )

    @property
    def phase_name(self) -> str:
        return self.name or getattr(self.kernel, "__name__", "phase")


@dataclass(frozen=True)
class UPMEMArray:
    """Declared distribution of one program array across UPMEM DPUs."""

    name: str
    layout: TensorLayout = TensorLayout.BLOCK
    partition_axis: int = 0
    halo: tuple[int, int] = (0, 0)
    linear_layout: LinearLayout | None = None

    def __post_init__(self):
        if not self.name.isidentifier():
            raise ValueError(f"invalid UPMEM array name {self.name!r}")
        object.__setattr__(self, "layout", TensorLayout(self.layout))
        object.__setattr__(self, "partition_axis", int(self.partition_axis))
        object.__setattr__(self, "halo", tuple(int(value) for value in self.halo))
        if self.linear_layout is not None and not isinstance(
            self.linear_layout, LinearLayout
        ):
            raise TypeError("UPMEMArray linear_layout must be a LinearLayout")


class UPMEMProgram:
    """Ordered MLIR phases defining a complete UPMEM workload program."""

    def __init__(
        self,
        phases,
        *,
        name: str | None = None,
        arrays=(),
        num_tasklets: int = 16,
        orchestration=None,
    ):
        phases = tuple(phases)
        if not phases:
            raise ValueError("UPMEMProgram requires at least one phase")
        if not all(isinstance(phase, UPMEMPhase) for phase in phases):
            raise TypeError("UPMEMProgram phases must be UPMEMPhase objects")
        names = [phase.phase_name for phase in phases]
        if len(names) != len(set(names)):
            raise ValueError("UPMEMProgram phase names must be unique")
        self.phases = phases
        self.name = name or names[0]
        if not self.name.isidentifier():
            raise ValueError("UPMEMProgram name must be a C identifier")
        arrays = tuple(arrays)
        if not all(isinstance(array, UPMEMArray) for array in arrays):
            raise TypeError("UPMEMProgram arrays must be UPMEMArray objects")
        if len({array.name for array in arrays}) != len(arrays):
            raise ValueError("UPMEMProgram array declarations must be unique")
        self.arrays = {array.name: array for array in arrays}
        self.num_tasklets = int(num_tasklets)
        if not 1 <= self.num_tasklets <= 24:
            raise ValueError("UPMEMProgram num_tasklets must be between 1 and 24")
        self.orchestration = orchestration

    def build(self):
        """Allow workload modules to return a program from their own build()."""

        return self


@dataclass
class _CompiledPhase:
    phase: UPMEMPhase
    artifact: object
    executable: object
    abi: PhaseABI
    source: str
    source_argument_names: tuple[str, ...]


_DIAGNOSTIC_MLIR_ATTRIBUTES = {
    "from",
    "itypes",
    "loop_name",
    "op_name",
    "otypes",
    "sym_name",
    "to",
    "top",
}
_AFFINE_CONSTANT_MAP = re.compile(r"affine_map<\(\) -> \((-?\d+)\)>")
_MEMORY_LOAD_OPS = {"affine.load", "memref.load"}
_MEMORY_STORE_OPS = {"affine.store", "memref.store"}
_UNSAFE_MEMORY_OPS = {
    "affine.vector_load",
    "affine.vector_store",
    "func.call",
    "memref.alloc",
    "memref.alloca",
    "memref.atomic_rmw",
    "memref.copy",
    "scf.while",
}


def _operation_children(operation):
    for region in operation.regions:
        for block in region.blocks:
            yield from block.operations


def _walk_operations(operation):
    yield operation
    for child in _operation_children(operation):
        yield from _walk_operations(child)


def _canonical_operation(operation):
    text = str(operation)
    for attribute in _DIAGNOSTIC_MLIR_ATTRIBUTES:
        text = re.sub(
            rf"\s*{re.escape(attribute)}\s*=\s*(?:\"[^\"]*\"|[^,}}]+),?",
            "",
            text,
        )
    text = re.sub(r"\bloc\([^\n]*\)", "loc(?)", text)
    return " ".join(text.split())


def _constant_affine_bound(attribute) -> int | None:
    match = _AFFINE_CONSTANT_MAP.fullmatch(str(attribute))
    return None if match is None else int(match.group(1))


def _loop_bounds(operation):
    if operation.operation.name != "affine.for":
        return None
    attributes = dict(operation.attributes)
    lower = _constant_affine_bound(attributes.get("lowerBoundMap"))
    upper = _constant_affine_bound(attributes.get("upperBoundMap"))
    step_match = re.match(r"(-?\d+)", str(attributes.get("step", "")))
    if lower is None or upper is None or step_match is None:
        return None
    step = int(step_match.group(1))
    if step <= 0:
        return None
    return lower, upper, step


def _memory_access(operation):
    name = operation.operation.name
    operands = tuple(operation.operands)
    if name in _MEMORY_LOAD_OPS:
        return "load", operands[0], operands[1:]
    if name in _MEMORY_STORE_OPS:
        return "store", operands[1], operands[2:]
    return None


def _affine_map_results(mapping) -> tuple[str, ...]:
    text = str(mapping)
    arrow = text.find("->")
    opening = text.find("(", arrow + 2)
    if arrow < 0 or opening < 0:
        return ()
    depth = 0
    result_start = opening + 1
    results = []
    item_start = result_start
    for index in range(opening, len(text)):
        token = text[index]
        if token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
            if depth == 0:
                results.append(text[item_start:index].strip())
                break
        elif token == "," and depth == 1:
            results.append(text[item_start:index].strip())
            item_start = index + 1
    return tuple(result for result in results if result)


def _direct_affine_induction(operation, indices, induction) -> bool:
    positions = [index for index, value in enumerate(indices) if value == induction]
    if not positions:
        return False
    mapping = dict(operation.attributes).get("map")
    if mapping is None:
        return True
    results = _affine_map_results(mapping)
    return any(result == f"d{position}" for position in positions for result in results)


def _is_dependence_safe_parallel_loop(operation) -> bool:
    bounds = _loop_bounds(operation)
    if bounds is None or len(operation.regions) != 1:
        return False
    blocks = tuple(operation.regions[0].blocks)
    if len(blocks) != 1 or len(blocks[0].arguments) != 1 or operation.results:
        return False
    induction = blocks[0].arguments[0]
    accesses = []
    for nested in _walk_operations(operation):
        name = nested.operation.name
        if name in _UNSAFE_MEMORY_OPS or (
            name.startswith("memref.")
            and name not in _MEMORY_LOAD_OPS
            and name not in _MEMORY_STORE_OPS
            and name not in {"memref.cast", "memref.dim"}
        ):
            return False
        access = _memory_access(nested)
        if access is not None:
            accesses.append((nested, *access))
    stores = [access for access in accesses if access[1] == "store"]
    if not stores:
        return False
    for nested, _, _, indices in stores:
        if not _direct_affine_induction(nested, indices, induction):
            return False
    for nested, _, _, indices in accesses:
        if not _direct_affine_induction(nested, indices, induction):
            return False
    return True


def _discover_parallel_regions(module) -> tuple[UPMEMParallelRegion, ...]:
    functions = [
        operation
        for operation in module.body.operations
        if operation.operation.name == "func.func"
    ]
    if len(functions) != 1:
        return ()
    regions = []
    next_loop_ordinal = 0

    def visit(operation, inside_selected=False):
        nonlocal next_loop_ordinal
        is_loop = operation.operation.name in {"affine.for", "scf.for"}
        ordinal = next_loop_ordinal if is_loop else None
        if is_loop:
            next_loop_ordinal += 1
        selected = (
            is_loop
            and not inside_selected
            and _is_dependence_safe_parallel_loop(operation)
        )
        if selected:
            lower, upper, step = _loop_bounds(operation)
            structural = {
                "schema": "upmem-parallel-region-v1",
                "loop_ordinal": ordinal,
                "operation": _canonical_operation(operation),
            }
            site_id = hashlib.sha256(
                json.dumps(structural, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            regions.append(UPMEMParallelRegion(site_id, ordinal, lower, upper, step))
        for child in _operation_children(operation):
            visit(child, inside_selected or selected)

    visit(functions[0])
    return tuple(regions)


def _parallelize_source(
    source: str,
    regions: tuple[UPMEMParallelRegion, ...],
    workers: int,
) -> str:
    if not regions:
        return source

    loop_line = re.compile(
        r"^(?P<indent>[ \t]*)(?P<label>l_[A-Za-z0-9_]+):[ \t]*"
        r"(?P<loop>for[ \t]*\()",
        re.MULTILINE,
    )
    requested = {region.loop_ordinal for region in regions}
    matched = set()
    loop_ordinal = -1

    def replace(match):
        nonlocal loop_ordinal
        loop_ordinal += 1
        if loop_ordinal not in requested:
            return match.group(0)
        matched.add(loop_ordinal)
        indent = match.group("indent")
        return (
            f"{indent}{match.group('label')}:\n"
            f"{indent}#pragma omp parallel for schedule(static) "
            f"num_threads({workers})\n"
            f"{indent}{match.group('loop')}"
        )

    source = loop_line.sub(replace, source)
    missing = sorted(requested - matched)
    if missing:
        raise RuntimeError(
            "retained MLIR parallel region(s) did not resolve to emitted C loop "
            "ordinal(s): " + ", ".join(map(str, missing))
        )
    return source


def _compile_phase(phase: UPMEMPhase) -> _CompiledPhase:
    kwargs = {"enable_tensor": False}
    if phase.instantiate:
        kwargs["instantiate"] = list(phase.instantiate)
    schedule = customize(phase.kernel, **kwargs)
    # UPMEM's native integer model is modular.  All ABI-visible integers are
    # at most 64 bits, while Allo may introduce i65 temporaries to express a
    # mathematically widened i32 MAC before truncating it back to i32.  Keep
    # the low 64 bits in portable C; this exactly preserves the later ABI
    # truncation and avoids requiring an HLS-only ap_int implementation.
    artifact = emit_c_from_mlir(
        schedule.module, schedule.top_func_name, wrap_wide_integers=True
    )
    source_parameters = tuple(inspect.signature(phase.kernel).parameters)
    source_arguments = tuple(
        argument for argument in artifact.arguments if argument.source == "argument"
    )
    result_arguments = tuple(
        argument for argument in artifact.arguments if argument.source == "result"
    )
    if len(source_parameters) != len(source_arguments):
        raise ValueError(
            f"phase {phase.phase_name!r} source signature has {len(source_parameters)} "
            f"arguments but emitted ABI has {len(source_arguments)}"
        )
    if len(phase.result_names) != len(result_arguments):
        raise ValueError(
            f"phase {phase.phase_name!r} declares {len(phase.result_names)} result "
            f"names but emitted ABI has {len(result_arguments)} results"
        )

    def named(name: str, argument: CArgument) -> ProgramArgument:
        return ProgramArgument(name, argument.dtype, argument.shape, argument.mode)

    arguments = tuple(
        named(name, argument)
        for name, argument in zip(source_parameters, source_arguments)
    ) + tuple(
        named(name, argument)
        for name, argument in zip(phase.result_names, result_arguments)
    )
    parallel_regions = _discover_parallel_regions(artifact.module)
    phase_abi = PhaseABI(
        phase.phase_name,
        arguments,
        parallel_regions,
        phase.parallel_workers if parallel_regions else 1,
    )
    source = _parallelize_source(
        artifact.c_source,
        parallel_regions,
        phase.parallel_workers,
    )
    executable = artifact.compile(
        source=source,
        compile_flags=("-fopenmp",) if parallel_regions else (),
    )
    return _CompiledPhase(
        phase,
        artifact,
        executable,
        phase_abi,
        source,
        source_parameters,
    )


def _program_abi(program: UPMEMProgram, phases) -> MLIRProgramABI:
    ordered = {}
    for phase in phases:
        for argument in phase.abi.arguments:
            previous = ordered.get(argument.name)
            if previous is None:
                ordered[argument.name] = argument
                continue
            if previous.dtype != argument.dtype or previous.shape != argument.shape:
                raise ValueError(
                    f"program argument {argument.name!r} has inconsistent phase ABI: "
                    f"{previous.dtype}{previous.shape} vs {argument.dtype}{argument.shape}"
                )
            ordered[argument.name] = ProgramArgument(
                argument.name,
                argument.dtype,
                argument.shape,
                _merge_modes(previous.mode, argument.mode),
            )
    return MLIRProgramABI(
        program.name,
        tuple(ordered.values()),
        tuple(phase.abi for phase in phases),
    )


_MLIR_NUMPY_DTYPES = {
    "i1": np.bool_,
    "ui1": np.bool_,
    "i8": np.int8,
    "ui8": np.uint8,
    "i16": np.int16,
    "ui16": np.uint16,
    "i32": np.int32,
    "ui32": np.uint32,
    "i64": np.int64,
    "ui64": np.uint64,
    "index": np.int64,
    "f32": np.float32,
    "f64": np.float64,
}


def _direction(mode: str) -> TensorDirection:
    return {
        "in": TensorDirection.INPUT,
        "out": TensorDirection.OUTPUT,
        "both": TensorDirection.INOUT,
        "func": TensorDirection.INPUT,
    }[mode]


def _target_extent(target, unit_name: str) -> int:
    unit = target.unit(unit_name)
    result = 1
    for factor in unit.mapping:
        result *= int(factor)
    return result


def _power_of_two_ceiling(value: int) -> int:
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


def _tensor_linear_layout(
    shape,
    partition_axis,
    tensor_layout,
    *,
    num_dpus,
    num_tasklets,
    halo=(0, 0),
):
    """Construct the physical ``logical -> (dpu, tasklet, local)`` F2 map.

    Arbitrary logical extents are padded to powers of two. Halo-free BLOCK
    tensors stripe their low partition bits over DPUs so every processor is
    used before a DPU receives a second element. Halo-bearing BLOCK tensors use
    contiguous power-of-two slices for neighborhood locality. BROADCAST tensors
    add an explicit replica input that maps identically to the DPU axis. Within
    a DPU, low payload bits stripe over a padded tasklet axis and remaining bits
    select a tasklet-local offset.
    """
    shape = tuple(int(extent) for extent in shape)
    num_dpus = int(num_dpus)
    if num_dpus <= 0 or num_dpus & (num_dpus - 1):
        raise ValueError("UPMEM LinearLayout requires a power-of-two DPU count")
    # F2 layouts require power-of-two axes. Use the largest physical subgroup
    # contained in the requested active tasklet count; the remaining tasklets
    # may service runtime/DMA work but own no layout coordinate.
    tasklet_span = 1 << (max(1, int(num_tasklets)).bit_length() - 1)
    axis_extent = shape[partition_axis]
    inner_extent = math.prod(
        extent for axis, extent in enumerate(shape) if axis != partition_axis
    )
    inner_span = _power_of_two_ceiling(inner_extent)
    if tensor_layout == TensorLayout.BROADCAST:
        local_partition_span = _power_of_two_ceiling(axis_extent)
    else:
        local_partition_span = _power_of_two_ceiling(math.ceil(axis_extent / num_dpus))

    inner_bits = inner_span.bit_length() - 1
    partition_bits = local_partition_span.bit_length() - 1
    payload_bits = inner_bits + partition_bits
    tasklet_bits = min(tasklet_span.bit_length() - 1, payload_bits)
    local_bits = payload_bits - tasklet_bits

    def payload_vector(bit_position):
        if bit_position < tasklet_bits:
            return (0, 1 << bit_position, 0)
        return (0, 0, 1 << (bit_position - tasklet_bits))

    bases = {}
    striped = tensor_layout == TensorLayout.BLOCK and tuple(halo) == (0, 0)
    if tensor_layout == TensorLayout.BROADCAST:
        bases["replica"] = [
            (1 << bit, 0, 0) for bit in range(num_dpus.bit_length() - 1)
        ]
    elif striped:
        bases["dpu_lane"] = [
            (1 << bit, 0, 0) for bit in range(num_dpus.bit_length() - 1)
        ]
    else:
        bases["dpu_block"] = [
            (1 << bit, 0, 0) for bit in range(num_dpus.bit_length() - 1)
        ]
    bases["local_partition"] = [
        payload_vector(inner_bits + bit) for bit in range(partition_bits)
    ]
    bases["inner"] = [payload_vector(bit) for bit in range(inner_bits)]
    return LinearLayout(
        bases=bases,
        out_dims=("dpu", "tasklet", "local"),
        out_sizes=(num_dpus, tasklet_span, 1 << local_bits),
    )


def _device_program_abi(program, target, phases) -> DeviceProgramABI:
    num_dpus = _target_extent(target, "dpu")
    launches = []
    for launch_id, phase in enumerate(phases):
        tensors = []
        scalars = []
        for argument in phase.abi.arguments:
            dtype = _MLIR_NUMPY_DTYPES.get(argument.dtype)
            if dtype is None:
                raise TypeError(
                    f"UPMEM ABI does not support MLIR type {argument.dtype!r}"
                )
            if not argument.shape:
                scalars.append(ScalarABI(argument.name, dtype))
                continue
            declaration = program.arrays.get(argument.name, UPMEMArray(argument.name))
            linear_layout = declaration.linear_layout or _tensor_linear_layout(
                argument.shape,
                declaration.partition_axis,
                declaration.layout,
                num_dpus=num_dpus,
                num_tasklets=program.num_tasklets,
                halo=declaration.halo,
            )
            tensors.append(
                TensorABI(
                    argument.name,
                    argument.shape,
                    dtype,
                    _direction(argument.mode),
                    declaration.layout,
                    declaration.partition_axis,
                    declaration.halo,
                    linear_layout,
                )
            )
        launches.append(
            LaunchABI(
                phase.phase.phase_name,
                tuple(tensors),
                tuple(scalars),
                num_dpus=num_dpus,
                num_tasklets=program.num_tasklets,
                launch_id=launch_id,
            )
        )
    return DeviceProgramABI(tuple(launches))


def _build_cost_graph(program, target, phases, device_abi, device_artifacts, cost):
    graph = ExecutionGraph(
        program.name,
        metadata={
            "backend": "upmem",
            "analysis": "retained-mlir-linear-layout",
            "simulator_cycles": False,
            "cost": getattr(getattr(cost, "spec", None), "name", None),
            "orchestration": _manifest_value(program.orchestration),
        },
    )
    if cost is None:
        return graph
    dependencies = ()
    for phase_index, (phase, launch, device_artifact) in enumerate(
        zip(phases, device_abi.launches, device_artifacts)
    ):
        distributed = [
            tensor
            for tensor in launch.tensors
            if tensor.layout == TensorLayout.BLOCK
            and tensor.direction in (TensorDirection.OUTPUT, TensorDirection.INOUT)
        ]
        if not distributed:
            distributed = [
                tensor
                for tensor in launch.tensors
                if tensor.layout == TensorLayout.BLOCK
            ]
        dpu_parallelism = max(
            (launch.spatial_parallelism(tensor.name) for tensor in distributed),
            default=1,
        )
        tasklet_mapping = device_artifact.schedule_realizable and bool(
            phase.abi.parallel_regions
        )
        tasklet_parallelism = device_artifact.active_tasklets if tasklet_mapping else 1
        graph.metadata.setdefault("layout_parallelism", []).append(
            {
                "phase": phase.phase.phase_name,
                "dpu": dpu_parallelism,
                "tasklet": tasklet_parallelism,
            }
        )
        for tensor in launch.tensors:
            if tensor.direction not in (TensorDirection.INPUT, TensorDirection.INOUT):
                continue
            move = (
                "BCAST_MRAM"
                if tensor.layout == TensorLayout.BROADCAST
                else "SCATTER_MRAM"
            )
            event = CostEvent.create(
                f"phase:{phase_index}:ingress:{tensor.name}",
                target.move(move),
                metrics={"bytes": tensor.num_elements * tensor.dtype.itemsize},
                attributes={"phase": phase.phase.phase_name, "analytical": True},
            )
            dependencies = tuple(cost.emit(graph, event, dependencies))

        summary = analyze_upmem_mlir(phase.artifact.source_mlir)
        graph.metadata.setdefault("mlir_summaries", []).append(
            {
                "phase": phase.phase.phase_name,
                "exact": summary.is_exact,
                "diagnostics": list(summary.diagnostics),
                "total_instruction_count": summary.total_instruction_count,
            }
        )
        for op_index, (target_name, raw_metrics) in enumerate(summary.cost_records()):
            metrics = dict(raw_metrics)
            if tasklet_mapping:
                metrics["iterations"] = (
                    int(metrics.get("iterations", 1)) + dpu_parallelism - 1
                ) // dpu_parallelism
                if "bytes" in metrics:
                    metrics["bytes"] = (
                        int(metrics["bytes"]) + dpu_parallelism - 1
                    ) // dpu_parallelism
            metrics["candidate"] = {
                "tasklet_fanout": tasklet_parallelism,
                "tasklet_mapping": tasklet_mapping,
                "dpu_fanout": dpu_parallelism,
            }
            primitive = (
                target.move(target_name)
                if target_name in {"LD_WRAM", "ST_WRAM"}
                else target.op(target_name)
            )
            event = CostEvent.create(
                f"phase:{phase_index}:op:{op_index}:{target_name}",
                primitive,
                work_id=(0,),
                metrics=metrics,
                attributes={
                    "phase": phase.phase.phase_name,
                    "analytical": True,
                    "source": "retained-mlir-summary",
                },
            )
            dependencies = tuple(cost.emit(graph, event, dependencies))

        for tensor in launch.tensors:
            if tensor.direction not in (TensorDirection.OUTPUT, TensorDirection.INOUT):
                continue
            event = CostEvent.create(
                f"phase:{phase_index}:egress:{tensor.name}",
                target.move("GATHER_MRAM"),
                metrics={"bytes": tensor.num_elements * tensor.dtype.itemsize},
                attributes={"phase": phase.phase.phase_name, "analytical": True},
            )
            dependencies = tuple(cost.emit(graph, event, dependencies))
    return graph


def _manifest_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return _manifest_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): _manifest_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_manifest_value(item) for item in value]
    manifest = getattr(value, "manifest", None)
    if callable(manifest):
        return _manifest_value(manifest())
    return repr(value)


@dataclass(frozen=True)
class UPMEMDeviceSourceManifest:
    """Immutable identity of one exact DPU translation unit and launch ABI."""

    kind: str
    num_tasklets: int
    active_tasklets: int
    compile_flags: tuple[str, ...]
    source_fingerprint: str
    abi_fingerprint: str
    parallel_region_ids: tuple[str, ...]
    sdk_complete: bool

    def manifest(self) -> dict:
        return {
            "kind": self.kind,
            "num_tasklets": self.num_tasklets,
            "active_tasklets": self.active_tasklets,
            "compile_flags": list(self.compile_flags),
            "source_fingerprint": self.source_fingerprint,
            "abi_fingerprint": self.abi_fingerprint,
            "parallel_region_ids": list(self.parallel_region_ids),
            "sdk_complete": self.sdk_complete,
        }


@dataclass(frozen=True)
class UPMEMDeviceCFragment:
    """Frozen DPU source artifact, complete only when capabilities are proven.

    Unsupported generic phases retain an exact ABI and compute fragment, but
    cannot enter schedule search or promotion.  Supported structural lowering
    supplies a complete translation unit, compile flags, and exact tasklet count.
    """

    launch: str
    abi_declaration: str
    source_fragment: str
    sdk_compiler: str | None
    num_tasklets: int = 1
    compile_flags: tuple[str, ...] = ()
    abi_fingerprint: str = ""
    parallel_region_ids: tuple[str, ...] = ()
    active_tasklets: int = 1
    lowering_diagnostic: str | None = None
    missing_capabilities: tuple[str, ...] = (
        "dpu_main",
        "mram_descriptor_loads",
        "tasklet_strided_execution",
        "compute_argument_binding",
        "mram_output_stores",
    )

    @property
    def sdk_complete(self) -> bool:
        return not self.missing_capabilities

    @property
    def sdk_compilable(self) -> bool:
        return self.sdk_complete and self.sdk_compiler is not None

    @property
    def schedule_realizable(self) -> bool:
        marker = f"#define TENON_NUM_TASKLETS {self.num_tasklets}"
        flag = f"-DNR_TASKLETS={self.num_tasklets}"
        return (
            self.sdk_complete
            and marker in self.source_fragment
            and f"#define TENON_ACTIVE_TASKLETS {self.active_tasklets}"
            in self.source_fragment
            and flag in self.compile_flags
            and bool(self.abi_fingerprint)
        )

    @property
    def source_manifest(self) -> UPMEMDeviceSourceManifest:
        kind = (
            "upmem-dpu-translation-unit"
            if self.sdk_complete
            else "upmem-dpu-translation-unit-fragment"
        )
        return UPMEMDeviceSourceManifest(
            kind,
            self.num_tasklets,
            self.active_tasklets,
            self.compile_flags,
            hashlib.sha256(self.source_fragment.encode()).hexdigest(),
            self.abi_fingerprint,
            self.parallel_region_ids,
            self.sdk_complete,
        )

    def capability_manifest(self) -> dict:
        return {
            "launch": self.launch,
            **self.source_manifest.manifest(),
            "sdk_compiler_available": self.sdk_compiler is not None,
            "sdk_compilable": self.sdk_compilable,
            "schedule_realizable": self.schedule_realizable,
            "active_tasklets": self.active_tasklets,
            "lowering_diagnostic": self.lowering_diagnostic,
            "missing_capabilities": list(self.missing_capabilities),
        }

    def require_sdk_compilable(self) -> None:
        if self.sdk_compilable:
            return
        if not self.sdk_complete:
            reasons = ", ".join(self.missing_capabilities)
            raise RuntimeError(
                f"UPMEM launch {self.launch!r} is a translation-unit fragment, "
                f"not an SDK-compilable DPU program; missing: {reasons}"
            )
        raise RuntimeError(
            f"UPMEM launch {self.launch!r} is a complete DPU translation unit, "
            "but the UPMEM SDK compiler is unavailable"
        )

    def require_schedule_realizable(self) -> None:
        if self.schedule_realizable:
            return
        diagnostic = self.lowering_diagnostic or ", ".join(self.missing_capabilities)
        raise RuntimeError(
            f"UPMEM launch {self.launch!r} has no complete candidate-specific "
            f"device translation unit: {diagnostic}"
        )


@dataclass(frozen=True)
class UPMEMDenseTile:
    """A concrete, SDK-compilable int32 GEMM tile for UPMEM DPUs.

    PolyBench's runnable P1/P2 comparison cells decompose into dense matrix
    products.  This artifact is the physical 16x16xK launch used by that
    decomposition: K is tiled through WRAM, rows are owned by tasklets, and
    the complete A/B/C tile resides in MRAM.  It is intentionally a narrow
    physical lowering rather than a claim that arbitrary portable C is
    automatically DPU-compatible.
    """

    rows: int
    columns: int
    reduction: int
    reduction_tile: int
    num_tasklets: int = 8

    def __post_init__(self):
        values = (
            int(self.rows),
            int(self.columns),
            int(self.reduction),
            int(self.reduction_tile),
            int(self.num_tasklets),
        )
        if any(value <= 0 for value in values):
            raise ValueError("UPMEM dense-tile dimensions must be positive")
        if self.rows % self.num_tasklets:
            raise ValueError("tile rows must divide evenly across tasklets")
        if self.reduction % self.reduction_tile:
            raise ValueError("reduction extent must be a multiple of its tile")
        if self.num_tasklets > 24:
            raise ValueError("UPMEM supports at most 24 tasklets")
        _validate_upmem_dma_size(4 * self.columns, "dense column")
        _validate_upmem_dma_size(4 * self.reduction_tile, "dense reduction tile")
        if self.wram_bytes > _UPMEM_WRAM_BYTES:
            raise ValueError(
                "UPMEM dense tile exceeds the shared 64 KiB WRAM capacity per DPU"
            )
        _validate_upmem_mram(self.mram_offsets, self.mram_image_bytes)

    @property
    def rows_per_tasklet(self):
        return self.rows // self.num_tasklets

    @property
    def reduction_tiles(self):
        return self.reduction // self.reduction_tile

    @property
    def wram_words_per_tasklet(self):
        rows = self.rows_per_tasklet
        return (
            rows * self.columns
            + self.reduction_tile * self.columns
            + rows * self.reduction_tile
        )

    @property
    def wram_bytes(self):
        return 4 * self.num_tasklets * self.wram_words_per_tasklet

    @property
    def mram_offsets(self):
        a_offset = 0
        b_offset = 4 * self.rows * self.reduction
        c_offset = 4 * (self.rows * self.reduction + self.reduction * self.columns)
        return a_offset, b_offset, c_offset

    @property
    def mram_image_bytes(self):
        return self.mram_offsets[2] + 4 * self.rows * self.columns

    def manifest(self):
        return {
            "kind": "upmem-dense-int32-tile",
            "shape": [self.rows, self.columns, self.reduction],
            "reduction_tile": self.reduction_tile,
            "reduction_tiles": self.reduction_tiles,
            "num_tasklets": self.num_tasklets,
            "rows_per_tasklet": self.rows_per_tasklet,
            "wram_words_per_tasklet": self.wram_words_per_tasklet,
            "mram_offsets": list(self.mram_offsets),
            "sdk_compilable_translation_unit": True,
        }

    def device_source(self):
        """Emit the complete DPU translation unit used by the simulator."""

        m, n, k = self.rows, self.columns, self.reduction
        kc, nt = self.reduction_tile, self.num_tasklets
        rpt, chunks = self.rows_per_tasklet, self.reduction_tiles
        a_offset, b_offset, c_offset = self.mram_offsets
        return f"""// Tenon dense-tile physical lowering for UPMEM.
// C[{m},{n}] += A[{m},{k}] * B[{k},{n}], int32; KC={kc}, tasklets={nt}.
#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <stdint.h>

__dma_aligned int32_t acc_buf[NR_TASKLETS][{rpt} * {n}];
__dma_aligned int32_t b_buf[NR_TASKLETS][{kc} * {n}];
__dma_aligned int32_t a_buf[NR_TASKLETS][{rpt} * {kc}];

static void tenon_gemm_tasklet(uint32_t tid, const int32_t *A,
                               const int32_t *B, int32_t *C) {{
  int32_t *acc = acc_buf[tid];
  int32_t *b = b_buf[tid];
  int32_t *a = a_buf[tid];
  for (uint32_t i = 0; i < {rpt}; ++i)
    for (uint32_t j = 0; j < {n}; ++j)
      acc[i * {n} + j] = 0;
  for (uint32_t kt = 0; kt < {chunks}; ++kt) {{
    for (uint32_t k0 = 0; k0 < {kc}; ++k0)
      mram_read((const __mram_ptr int32_t *)(uintptr_t)&B[(k0 + {kc} * kt) * {n}],
                (int32_t *)&b[k0 * {n}], {n} * sizeof(int32_t));
    for (uint32_t i = 0; i < {rpt}; ++i)
      mram_read((const __mram_ptr int32_t *)(uintptr_t)&A[(i + {rpt} * tid) * {k} + {kc} * kt],
                (int32_t *)&a[i * {kc}], {kc} * sizeof(int32_t));
    for (uint32_t i = 0; i < {rpt}; ++i)
      for (uint32_t j = 0; j < {n}; ++j)
        for (uint32_t k0 = 0; k0 < {kc}; ++k0)
          acc[i * {n} + j] += a[i * {kc} + k0] * b[k0 * {n} + j];
  }}
  for (uint32_t i = 0; i < {rpt}; ++i)
    mram_write((const int32_t *)&acc[i * {n}],
               (__mram_ptr int32_t *)(uintptr_t)&C[({rpt} * tid + i) * {n}],
               {n} * sizeof(int32_t));
}}

BARRIER_INIT(tenon_barrier, NR_TASKLETS);

int main(void) {{
  uint32_t tid = me();
  uint32_t heap = (uint32_t)DPU_MRAM_HEAP_POINTER;
  if (tid == 0) mem_reset();
  barrier_wait(&tenon_barrier);
  if (tid < {nt})
    tenon_gemm_tasklet(tid, (const int32_t *)(heap + {a_offset}),
                       (const int32_t *)(heap + {b_offset}),
                       (int32_t *)(heap + {c_offset}));
  barrier_wait(&tenon_barrier);
  return 0;
}}
"""


@dataclass(frozen=True)
class UPMEMDotTile:
    """Complete multi-tasklet int32 dot-product DPU translation unit."""

    elements: int
    num_tasklets: int = 16
    dma_chunk: int = 128

    def __post_init__(self):
        if min(int(self.elements), int(self.num_tasklets), int(self.dma_chunk)) <= 0:
            raise ValueError("UPMEM dot-tile parameters must be positive")
        if self.num_tasklets > 24:
            raise ValueError("UPMEM supports at most 24 tasklets")
        if self.elements % (self.num_tasklets * self.dma_chunk):
            raise ValueError("dot extent must divide into equal tasklet DMA chunks")
        _validate_upmem_dma_size(4 * self.dma_chunk, "dot tile")
        if self.wram_bytes > _UPMEM_WRAM_BYTES:
            raise ValueError(
                "UPMEM dot tile exceeds the shared 64 KiB WRAM capacity per DPU"
            )
        _validate_upmem_mram(self.mram_offsets, self.mram_image_bytes)

    @property
    def chunks_per_tasklet(self):
        return self.elements // (self.num_tasklets * self.dma_chunk)

    @property
    def mram_offsets(self):
        return 0, 4 * self.elements, 8 * self.elements

    @property
    def wram_bytes(self):
        stack_bytes = 8 * self.dma_chunk + 8
        return 4 * self.num_tasklets + self.num_tasklets * stack_bytes

    @property
    def mram_image_bytes(self):
        return self.mram_offsets[2] + 8

    def manifest(self):
        return {
            "kind": "upmem-dot-int32-tile",
            "elements": self.elements,
            "num_tasklets": self.num_tasklets,
            "dma_chunk": self.dma_chunk,
            "chunks_per_tasklet": self.chunks_per_tasklet,
            "mram_offsets": list(self.mram_offsets),
            "sdk_compilable_translation_unit": True,
        }

    def device_source(self):
        n, nt, chunk = self.elements, self.num_tasklets, self.dma_chunk
        chunks = self.chunks_per_tasklet
        a_offset, b_offset, c_offset = self.mram_offsets
        return f"""// Tenon int32 dot-product physical lowering for UPMEM.
// dot(A[{n}], B[{n}]); tasklets={nt}, DMA chunk={chunk} int32 values.
#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <stdint.h>

__dma_aligned int32_t tenon_partial[{nt}];
BARRIER_INIT(tenon_barrier, NR_TASKLETS);

int main(void) {{
  uint32_t tid = me();
  uint32_t heap = (uint32_t)DPU_MRAM_HEAP_POINTER;
  const int32_t *A = (const int32_t *)(heap + {a_offset});
  const int32_t *B = (const int32_t *)(heap + {b_offset});
  tenon_partial[tid] = 0;
  for (uint32_t block = 0; block < {chunks}; ++block) {{
    __dma_aligned int32_t a[{chunk}];
    __dma_aligned int32_t b[{chunk}];
    uint32_t base = {chunk} * block + {chunk * chunks} * tid;
    mram_read((const __mram_ptr int32_t *)(uintptr_t)&A[base], a, {4 * chunk});
    mram_read((const __mram_ptr int32_t *)(uintptr_t)&B[base], b, {4 * chunk});
    for (uint32_t i = 0; i < {chunk}; ++i)
      tenon_partial[tid] += a[i] * b[i];
  }}
  barrier_wait(&tenon_barrier);
  if (tid == 0) {{
    __dma_aligned int32_t result[2] = {{0, 0}};
    for (uint32_t tasklet = 0; tasklet < {nt}; ++tasklet)
      result[0] += tenon_partial[tasklet];
    mram_write(result, (__mram_ptr int32_t *)(uintptr_t)(heap + {c_offset}), 8);
  }}
  return 0;
}}
"""


@dataclass(frozen=True)
class UPMEMRank1Tile:
    """Complete tasklet-sharded int32 rank-1 row update for one DPU."""

    elements: int
    num_tasklets: int = 16

    def __post_init__(self):
        if int(self.elements) <= 0 or not 1 <= int(self.num_tasklets) <= 24:
            raise ValueError("UPMEM rank-1 tile parameters must be positive")
        if self.elements % self.num_tasklets:
            raise ValueError("rank-1 row must divide evenly across tasklets")
        _validate_upmem_dma_size(4 * self.elements_per_tasklet, "rank-1 tile")
        if self.wram_bytes > _UPMEM_WRAM_BYTES:
            raise ValueError(
                "UPMEM rank-1 tile exceeds the shared 64 KiB WRAM capacity per DPU"
            )
        _validate_upmem_mram(self.mram_offsets, self.mram_image_bytes)

    @property
    def elements_per_tasklet(self):
        return self.elements // self.num_tasklets

    @property
    def mram_offsets(self):
        return 0, 4 * self.elements, 8 * self.elements

    @property
    def wram_bytes(self):
        return self.num_tasklets * (8 + 8 * self.elements_per_tasklet)

    @property
    def mram_image_bytes(self):
        return self.mram_offsets[2] + 8

    def manifest(self):
        return {
            "kind": "upmem-rank1-int32-tile",
            "elements": self.elements,
            "num_tasklets": self.num_tasklets,
            "elements_per_tasklet": self.elements_per_tasklet,
            "mram_offsets": list(self.mram_offsets),
            "sdk_compilable_translation_unit": True,
        }

    def device_source(self):
        n, nt, chunk = self.elements, self.num_tasklets, self.elements_per_tasklet
        a_offset, v_offset, u_offset = self.mram_offsets
        return f"""// Tenon int32 rank-1 row update physical lowering for UPMEM.
// A[{n}] += u * V[{n}]; tasklets={nt}, contiguous slice={chunk}.
#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <stdint.h>

BARRIER_INIT(tenon_barrier, NR_TASKLETS);

int main(void) {{
  uint32_t tid = me();
  uint32_t heap = (uint32_t)DPU_MRAM_HEAP_POINTER;
  int32_t *A = (int32_t *)(heap + {a_offset});
  const int32_t *V = (const int32_t *)(heap + {v_offset});
  __dma_aligned int32_t scalar[2];
  __dma_aligned int32_t a[{chunk}];
  __dma_aligned int32_t v[{chunk}];
  mram_read((const __mram_ptr int32_t *)(uintptr_t)(heap + {u_offset}), scalar, 8);
  mram_read((const __mram_ptr int32_t *)(uintptr_t)&A[{chunk} * tid], a,
            {4 * chunk});
  mram_read((const __mram_ptr int32_t *)(uintptr_t)&V[{chunk} * tid], v,
            {4 * chunk});
  for (uint32_t i = 0; i < {chunk}; ++i)
    a[i] += scalar[0] * v[i];
  mram_write(a, (__mram_ptr int32_t *)(uintptr_t)&A[{chunk} * tid], {4 * chunk});
  barrier_wait(&tenon_barrier);
  return 0;
}}
"""


def _device_c_fragment(phase, launch):
    return (
        "/* Tenon UPMEM DPU TRANSLATION-UNIT FRAGMENT. This is not a complete "
        "DPU program: no main/MRAM/tasklet wrapper is emitted. */\n"
        "#include <defs.h>\n"
        "#include <mram.h>\n"
        + launch.c_declaration()
        + "\n/* MLIR-derived compute translation unit. */\n"
        + phase.source
    )


class _DeviceLoweringUnsupported(ValueError):
    pass


_C_TYPE_BY_MLIR = {
    "i1": "bool",
    "ui1": "bool",
    "i8": "int8_t",
    "ui8": "uint8_t",
    "i16": "int16_t",
    "ui16": "uint16_t",
    "i32": "int32_t",
    "ui32": "uint32_t",
    "i64": "int64_t",
    "ui64": "uint64_t",
    "f32": "float",
    "f64": "double",
}
_C_FOR_LOOP = re.compile(
    r"(?P<indent>^[ \t]*)(?:(?P<label>[A-Za-z_]\w*):\s*)?"
    r"for\s*\(\s*int\s+(?P<iv>[A-Za-z_]\w*)\s*=\s*0\s*;\s*"
    r"(?P=iv)\s*<\s*(?P<extent>\d+)\s*;\s*(?P=iv)\+\+\s*\)\s*\{",
    re.MULTILINE,
)


def _launch_abi_fingerprint(launch) -> str:
    return hashlib.sha256(
        json.dumps(
            launch.metadata_manifest(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _rank1_pointwise_device_source(phase, launch):
    regions = phase.abi.parallel_regions
    if len(regions) != 1 or regions[0].loop_ordinal != 0:
        raise _DeviceLoweringUnsupported(
            "requires exactly one outermost dependence-safe affine region"
        )
    functions = [
        operation
        for operation in phase.artifact.module.body.operations
        if operation.operation.name == "func.func"
    ]
    if len(functions) != 1:
        raise _DeviceLoweringUnsupported("requires one retained MLIR function")
    if any(
        operation.operation.name in _UNSAFE_MEMORY_OPS
        for operation in _walk_operations(functions[0])
    ):
        raise _DeviceLoweringUnsupported(
            "calls or function-local memory are not included in the WRAM manifest"
        )

    memory_outside_loop = False

    def check_memory_scope(operation, inside_loop=False):
        nonlocal memory_outside_loop
        inside_loop = inside_loop or operation.operation.name in {
            "affine.for",
            "scf.for",
        }
        if _memory_access(operation) is not None and not inside_loop:
            memory_outside_loop = True
        for child in _operation_children(operation):
            check_memory_scope(child, inside_loop)

    check_memory_scope(functions[0])
    if memory_outside_loop:
        raise _DeviceLoweringUnsupported(
            "memory accesses outside the parallel region are not tasklet-safe"
        )
    if launch.scalars or any(not argument.shape for argument in phase.abi.arguments):
        raise _DeviceLoweringUnsupported("scalar launch arguments are not lowered")
    if any(len(argument.shape) != 1 for argument in phase.abi.arguments):
        raise _DeviceLoweringUnsupported("only rank-one pointwise phases are lowered")

    owners = [
        (index, tensor)
        for index, tensor in enumerate(launch.tensors)
        if tensor.direction in (TensorDirection.OUTPUT, TensorDirection.INOUT)
        and tensor.layout == TensorLayout.BLOCK
    ]
    if not owners:
        raise _DeviceLoweringUnsupported(
            "requires a block-owned output or inout tensor"
        )
    if any(
        tensor.direction in (TensorDirection.OUTPUT, TensorDirection.INOUT)
        and tensor.layout != TensorLayout.BLOCK
        for tensor in launch.tensors
    ):
        raise _DeviceLoweringUnsupported(
            "pointwise outputs and inout tensors require block ownership"
        )
    owner_index, owner = owners[0]
    if regions[0].lower_bound != 0 or regions[0].step != 1:
        raise _DeviceLoweringUnsupported("requires a zero-based unit-stride region")
    if regions[0].upper_bound != owner.shape[0]:
        raise _DeviceLoweringUnsupported(
            "parallel region extent must equal the block-owned output extent"
        )
    if any(tensor.shape != owner.shape for tensor in launch.tensors):
        raise _DeviceLoweringUnsupported(
            "all pointwise tensors must have the block-owned output shape"
        )
    owner_shards = launch.tensor_slot(owner.name).shards
    for tensor in launch.tensors:
        if tensor.layout != TensorLayout.BLOCK:
            continue
        if launch.tensor_slot(tensor.name).shards != owner_shards:
            raise _DeviceLoweringUnsupported(
                "all block tensors must have identical structural ownership"
            )
    active_tasklets = min(
        launch.tasklet_parallelism(owner.name),
        max(shard.owned_extent for shard in owner_shards),
    )
    active_tasklets = max(1, active_tasklets)

    source = phase.artifact.c_source
    loops = tuple(_C_FOR_LOOP.finditer(source))
    if len(loops) != 1:
        raise _DeviceLoweringUnsupported(
            "portable C must contain exactly one canonical pointwise loop"
        )
    loop = loops[0]
    induction = loop.group("iv")
    if int(loop.group("extent")) != regions[0].trip_count:
        raise _DeviceLoweringUnsupported(
            "portable-C and retained-MLIR loop extents disagree"
        )

    for index, argument in enumerate(phase.abi.arguments):
        if argument.dtype not in _C_TYPE_BY_MLIR:
            raise _DeviceLoweringUnsupported(
                f"unsupported pointwise element type {argument.dtype}"
            )
        subscripts = re.findall(rf"\bv{index}\s*\[\s*([^\]]+)\s*\]", source)
        if not subscripts or any(
            subscript.strip() != induction and not subscript.strip().isdigit()
            for subscript in subscripts
        ):
            raise _DeviceLoweringUnsupported(
                "all pointwise tensor accesses must use the parallel induction"
            )
        access = re.compile(rf"\bv{index}\s*\[\s*{re.escape(induction)}\s*\]")
        source, replacements = access.subn(
            f"v{index}[tenon_payload_index(&DPU_INPUT_ARGUMENTS.tensors[{index}], "
            f"(uint32_t){induction})]",
            source,
        )
        if replacements == 0:
            raise _DeviceLoweringUnsupported(
                "all pointwise tensor accesses must use the parallel induction"
            )

    label = f"{loop.group('label')}: " if loop.group("label") else ""
    replacement = (
        f"{loop.group('indent')}{label}for (uint32_t tenon_local = tid; "
        "tenon_local < tenon_owner->owned_extent; "
        "tenon_local += TENON_ACTIVE_TASKLETS) {\n"
        f"{loop.group('indent')}  int {induction} = "
        "(int)(tenon_owner->owned_start + "
        "tenon_local * tenon_owner->owned_stride);"
    )
    source = source[: loop.start()] + replacement + source[loop.end() :]
    function_name = re.escape(phase.artifact.top_func_name)
    source, substitutions = re.subn(
        rf"\bvoid\s+({function_name})\s*\(",
        r"void \1(uint32_t tid, const tenon_upmem_tensor_t *tenon_owner, ",
        source,
        count=1,
    )
    if substitutions != 1:
        raise _DeviceLoweringUnsupported("portable-C entry function was not found")

    slots = tuple(launch.slots)
    wram_bytes = sum(slot.slot_bytes for slot in slots)
    if wram_bytes > _UPMEM_WRAM_BYTES - 4096:
        raise _DeviceLoweringUnsupported(
            "pointwise tensor slots leave insufficient WRAM for the DPU runtime"
        )
    buffers = "\n".join(
        f"__dma_aligned uint8_t tenon_tensor_{index}[{slot.slot_bytes}];"
        for index, slot in enumerate(slots)
    )
    call_arguments = ", ".join(
        f"({_C_TYPE_BY_MLIR[argument.dtype]} *)(void *)tenon_tensor_{index}"
        for index, argument in enumerate(phase.abi.arguments)
    )
    reads = "\n".join(
        "    tenon_dma_read(tenon_tensor_{index}, "
        "&DPU_INPUT_ARGUMENTS.tensors[{index}]);".format(index=index)
        for index, tensor in enumerate(launch.tensors)
        if tensor.direction in (TensorDirection.INPUT, TensorDirection.INOUT)
    )
    writes = "\n".join(
        "    tenon_dma_write(tenon_tensor_{index}, "
        "&DPU_INPUT_ARGUMENTS.tensors[{index}]);".format(index=index)
        for index, tensor in enumerate(launch.tensors)
        if tensor.direction in (TensorDirection.OUTPUT, TensorDirection.INOUT)
    )
    num_tasklets = int(launch.num_tasklets)
    complete = f"""// Tenon retained-MLIR rank-one pointwise lowering for UPMEM.
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <stdint.h>

#define TENON_NUM_TASKLETS {num_tasklets}
#define TENON_ACTIVE_TASKLETS {active_tasklets}
#ifndef NR_TASKLETS
#define NR_TASKLETS TENON_NUM_TASKLETS
#endif
#if NR_TASKLETS != TENON_NUM_TASKLETS
#error "NR_TASKLETS must match this frozen Tenon candidate"
#endif

{launch.c_declaration()}
{buffers}
BARRIER_INIT(tenon_barrier, TENON_NUM_TASKLETS);

static uint32_t tenon_payload_index(const tenon_upmem_tensor_t *tensor,
                                    uint32_t logical_index) {{
  if (tensor->layout == {int(TensorLayout.BROADCAST)})
    return logical_index;
  return (logical_index - tensor->transfer_start) / tensor->transfer_stride;
}}

static void tenon_dma_read(uint8_t *destination,
                           const tenon_upmem_tensor_t *tensor) {{
  uint32_t heap = (uint32_t)DPU_MRAM_HEAP_POINTER;
  for (uint32_t copied = 0; copied < tensor->slot_bytes; copied += 2048) {{
    uint32_t remaining = tensor->slot_bytes - copied;
    uint32_t bytes = remaining < 2048 ? remaining : 2048;
    mram_read((const __mram_ptr void *)(uintptr_t)(heap + tensor->offset + copied),
              destination + copied, bytes);
  }}
}}

static void tenon_dma_write(const uint8_t *source,
                            const tenon_upmem_tensor_t *tensor) {{
  uint32_t heap = (uint32_t)DPU_MRAM_HEAP_POINTER;
  for (uint32_t copied = 0; copied < tensor->slot_bytes; copied += 2048) {{
    uint32_t remaining = tensor->slot_bytes - copied;
    uint32_t bytes = remaining < 2048 ? remaining : 2048;
    mram_write(source + copied,
               (__mram_ptr void *)(uintptr_t)(heap + tensor->offset + copied), bytes);
  }}
}}

{source}

int main(void) {{
  uint32_t tid = me();
  if (DPU_INPUT_ARGUMENTS.num_tasklets != TENON_NUM_TASKLETS)
    return 1;
  if (tid == 0) {{
{reads}
  }}
  barrier_wait(&tenon_barrier);
  if (tid < TENON_ACTIVE_TASKLETS)
    {phase.artifact.top_func_name}(
        tid, &DPU_INPUT_ARGUMENTS.tensors[{owner_index}], {call_arguments});
  barrier_wait(&tenon_barrier);
  if (tid == 0) {{
{writes}
  }}
  barrier_wait(&tenon_barrier);
  return 0;
}}
"""
    return complete, active_tasklets


def _device_c_artifact(phase, launch, sdk_compiler):
    abi_fingerprint = _launch_abi_fingerprint(launch)
    region_ids = tuple(region.site_id for region in phase.abi.parallel_regions)
    try:
        source, active_tasklets = _rank1_pointwise_device_source(phase, launch)
    except _DeviceLoweringUnsupported as error:
        return UPMEMDeviceCFragment(
            phase.phase.phase_name,
            launch.c_declaration(),
            _device_c_fragment(phase, launch),
            sdk_compiler,
            num_tasklets=int(launch.num_tasklets),
            abi_fingerprint=abi_fingerprint,
            parallel_region_ids=region_ids,
            lowering_diagnostic=str(error),
        )
    return UPMEMDeviceCFragment(
        phase.phase.phase_name,
        launch.c_declaration(),
        source,
        sdk_compiler,
        num_tasklets=int(launch.num_tasklets),
        compile_flags=(f"-DNR_TASKLETS={launch.num_tasklets}",),
        abi_fingerprint=abi_fingerprint,
        parallel_region_ids=region_ids,
        active_tasklets=active_tasklets,
        missing_capabilities=(),
    )


class CompiledUPMEMProgram:
    """Executable phased artifact retained by :class:`UPMEMProgramCallable`."""

    def __init__(self, program, target, cost=None):
        self.program = program
        self.target = target
        self.cost = cost
        self.phases = tuple(_compile_phase(phase) for phase in program.phases)
        self.mlir_abi = _program_abi(program, self.phases)
        self.device_abi = _device_program_abi(program, target, self.phases)
        self.abi = self.device_abi
        self.mlir_abi_manifest = self.mlir_abi.manifest()
        sdk_compiler = shutil.which("dpu-upmem-dpurte-clang")
        self.device_c_artifacts = tuple(
            _device_c_artifact(phase, launch, sdk_compiler)
            for phase, launch in zip(self.phases, self.device_abi.launches)
        )
        self.device_source_manifests = tuple(
            artifact.source_manifest for artifact in self.device_c_artifacts
        )
        self.schedule_realizable = all(
            artifact.schedule_realizable for artifact in self.device_c_artifacts
        )
        self.abi_manifest = {
            "abi": "upmem-program",
            "synchronization": "host-global-barrier-between-launches",
            "oracle_launches": [
                launch.metadata_manifest() for launch in self.device_abi.launches
            ],
            # The declarative plan may contain more physical launches than the
            # canonical functional MLIR phase.  Retain both without pretending
            # the host oracle executed every planned device launch.
            "planned_orchestration": _manifest_value(program.orchestration),
            "mlir_signature": self.mlir_abi_manifest,
            "device_c_artifacts": [
                artifact.capability_manifest() for artifact in self.device_c_artifacts
            ],
        }
        self.source_mlir = tuple(phase.artifact.source_mlir for phase in self.phases)
        self.lowered_mlir = tuple(phase.artifact.lowered_mlir for phase in self.phases)
        self.c_source = tuple(phase.source for phase in self.phases)
        self.device_c_abi = tuple(
            launch.c_declaration() for launch in self.device_abi.launches
        )
        self.device_c_fragment = tuple(
            artifact.source_fragment for artifact in self.device_c_artifacts
        )
        if self.schedule_realizable:
            self.device_c_source = self.device_c_fragment
        self.execution_graph = _build_cost_graph(
            program,
            target,
            self.phases,
            self.device_abi,
            self.device_c_artifacts,
            cost,
        )
        self.last_packed_manifest = None

    def require_schedule_realizable(self) -> None:
        for artifact in self.device_c_artifacts:
            artifact.require_schedule_realizable()

    def _materialization_payload(self) -> dict:
        return {
            "kind": "upmem-program-materialization-v2",
            "num_tasklets": int(self.program.num_tasklets),
            "device_launches": [
                launch.metadata_manifest() for launch in self.device_abi.launches
            ],
            "source_mlir": self.source_mlir,
            "lowered_mlir": self.lowered_mlir,
            "portable_c_source": self.c_source,
            "device_c_abi": self.device_c_abi,
            "device_sources": [
                manifest.manifest() for manifest in self.device_source_manifests
            ],
        }

    @property
    def legacy_materialization_fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self._materialization_payload(),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    @property
    def promotion_materialization_fingerprint(self) -> str:
        self.require_schedule_realizable()
        return self.legacy_materialization_fingerprint

    def run(self, **inputs) -> RunResult:
        environment = dict(inputs)
        packed_manifests = []
        for compiled_phase, launch in zip(self.phases, self.device_abi.launches):
            tensor_values = {
                tensor.name: environment[tensor.name] for tensor in launch.tensors
            }
            scalar_values = {
                scalar.name: environment[scalar.name] for scalar in launch.scalars
            }
            packed_before = launch.pack(tensor_values, scalar_values)
            arguments = [
                environment[name] for name in compiled_phase.source_argument_names
            ]
            returned = compiled_phase.executable(*arguments)
            result_names = compiled_phase.phase.result_names
            if result_names:
                values = returned if isinstance(returned, tuple) else (returned,)
                for name, value in zip(result_names, values):
                    destination = environment[name]
                    if isinstance(destination, np.ndarray):
                        np.copyto(destination, np.asarray(value), casting="same_kind")
                    else:
                        environment[name] = value

            # Repack and gather through the exact device ABI.  The host oracle
            # executes full logical arrays, while this round trip verifies that
            # BLOCK ownership, BROADCAST replication, padding, and output
            # reconstruction preserve the same observable arrays.
            tensor_values = {
                tensor.name: environment[tensor.name] for tensor in launch.tensors
            }
            packed_after = launch.pack(tensor_values, scalar_values)
            gathered = launch.gather([packed.mram for packed in packed_after])
            for name, value in gathered.items():
                np.copyto(environment[name], value, casting="same_kind")
            packed_manifests.append(
                {
                    **launch.metadata_manifest(),
                    "packed_dpus": len(packed_before),
                    "metadata_image_bytes": len(packed_before[0].metadata),
                    "mram_image_bytes": len(packed_before[0].mram),
                }
            )

        outputs = {
            argument.name: environment[argument.name]
            for argument in self.mlir_abi.arguments
            if argument.mode in {"out", "both"}
        }
        self.last_packed_manifest = tuple(packed_manifests)
        return RunResult(
            cycles=None,
            stdout="MLIR portable-C functional oracle; no UPMEM simulator cycles",
            backend=self.target.name,
            extra={
                "outputs": outputs,
                "functional_oracle": True,
                "simulator_cycles": False,
                "abi": self.abi_manifest,
                "packed_launches": packed_manifests,
            },
        )


def _with_upmem_tasklets(program, num_tasklets):
    if int(num_tasklets) == program.num_tasklets:
        return program
    return UPMEMProgram(
        program.phases,
        name=program.name,
        arrays=tuple(program.arrays.values()),
        num_tasklets=int(num_tasklets),
        orchestration=program.orchestration,
    )


class UPMEMScheduleUnavailable(RuntimeError):
    """Raised before scoring when no complete generic DPU lowering exists."""

    def __init__(self, reason: str, incumbent: CompiledUPMEMProgram):
        self.incumbent = incumbent
        super().__init__(reason)


def search_upmem_program_schedule(
    program,
    target,
    cost,
    *,
    incumbent_materialized=None,
):
    """Rank tasklet fanouts only after exact complete DPU materialization."""

    if not isinstance(cost, BoundCostSpec) or cost.target is not target:
        raise TypeError("UPMEM schedule search requires cost bound to the target")
    if not isinstance(program, UPMEMProgram):
        raise TypeError("UPMEM schedule search requires a UPMEMProgram")
    incumbent_compiled = incumbent_materialized or CompiledUPMEMProgram(
        program,
        target,
        cost=cost,
    )
    if (
        incumbent_compiled.program is not program
        or incumbent_compiled.target is not target
        or incumbent_compiled.cost is not cost
    ):
        raise ValueError("UPMEM incumbent materialization does not match the search")
    if not incumbent_compiled.schedule_realizable:
        diagnostics = tuple(
            artifact.lowering_diagnostic or "incomplete device translation unit"
            for artifact in incumbent_compiled.device_c_artifacts
            if not artifact.schedule_realizable
        )
        raise UPMEMScheduleUnavailable(
            "autoschedule requires complete candidate-specific UPMEM device source; "
            + "; ".join(diagnostics),
            incumbent_compiled,
        )
    if not any(phase.abi.parallel_regions for phase in incumbent_compiled.phases):
        raise UPMEMScheduleUnavailable(
            "retained MLIR exposes no dependence-safe parallel region",
            incumbent_compiled,
        )
    tasklet_domain = tuple(range(1, 25))

    def materialize(selected):
        if selected is program:
            return incumbent_compiled
        compiled = CompiledUPMEMProgram(selected, target, cost=cost)
        try:
            compiled.require_schedule_realizable()
        except RuntimeError as error:
            raise InfeasibleSchedule(str(error)) from error
        return compiled

    def score(compiled):
        compiled.require_schedule_realizable()
        return cost.evaluate(compiled.execution_graph)

    return grid_search(
        (DecisionDomain("num_tasklets", tasklet_domain),),
        build=lambda decisions: _with_upmem_tasklets(
            program, decisions["num_tasklets"]
        ),
        materialize=materialize,
        score=score,
        objective=lambda estimate: int(estimate.cycles),
        objective_domain=ScheduleObjectiveDomain.fingerprinted_target(
            metric="cycles",
            target="upmem",
            model_fingerprint=cost.fingerprint,
            fidelity="analytical",
            scope="whole_program",
            unit="cycles",
            direction="minimize",
        ),
        incumbent={"num_tasklets": program.num_tasklets},
    )


class UPMEMProgramCallable:
    """Signature-compatible NumPy callable returned by ``allo.compile``."""

    def __init__(
        self,
        compiled: CompiledUPMEMProgram,
        schedule_search_result=None,
        schedule_activation=None,
        fallback_reason=None,
    ):
        self.compiled = compiled
        self.program = compiled.program
        self.target = compiled.target
        self.cost = compiled.cost
        self.schedule = None
        self.trace = None
        self.abi = compiled.abi
        self.mlir_abi = compiled.mlir_abi
        self.abi_manifest = compiled.abi_manifest
        self.signature = inspect.Signature(
            inspect.Parameter(argument.name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for argument in self.mlir_abi.arguments
        )
        self.__signature__ = self.signature
        self.__name__ = self.program.name
        self.last_result = None
        self.schedule_search_result = schedule_search_result
        self.schedule_activation = schedule_activation
        self.fallback_reason = fallback_reason
        if schedule_activation is not None:
            self.fallback_reason = schedule_activation.fallback_reason

    @property
    def execution_graph(self):
        return self.compiled.execution_graph

    def __call__(self, *args, **kwargs) -> RunResult:
        bound = self.signature.bind(*args, **kwargs)
        result = self.compiled.run(**bound.arguments)
        self.last_result = result
        return result

    run = __call__

    def run_backend(self, **inputs) -> RunResult:
        return self(**inputs)

    def estimate(self):
        if self.cost is None:
            raise RuntimeError("compiled UPMEM program has no executable cost spec")
        if not self.execution_graph.activities:
            raise RuntimeError("UPMEM MLIR cost analysis emitted no operations")
        return self.cost.evaluate(self.execution_graph)


def compile_upmem_program(
    program,
    target,
    *,
    cost=None,
    promotion_gate=None,
) -> UPMEMProgramCallable:
    """Compile a general UPMEM program without invoking the SPMW matcher."""

    promotion_gate = validate_schedule_promotion_gate(promotion_gate)
    if target.name != "upmem":
        raise ValueError("UPMEMProgram can only be compiled for the upmem target")
    if cost is None:
        if promotion_gate is not None:
            raise ValueError(
                "promotion_gate requires automatic cost-ranked UPMEM search"
            )
        return UPMEMProgramCallable(
            CompiledUPMEMProgram(program, target, cost=None),
            schedule_search_result=None,
        )
    incumbent = CompiledUPMEMProgram(program, target, cost=cost)
    try:
        search_result = search_upmem_program_schedule(
            program,
            target,
            cost,
            incumbent_materialized=incumbent,
        )
    except UPMEMScheduleUnavailable as error:
        return UPMEMProgramCallable(
            error.incumbent,
            schedule_search_result=None,
            fallback_reason=f"autoschedule_unavailable: {error}",
        )
    activation = guarded_schedule_activation(
        search_result,
        promotion_gate=promotion_gate,
    )
    return UPMEMProgramCallable(
        activation.active_materialized,
        schedule_search_result=search_result,
        schedule_activation=activation,
    )


__all__ = [
    "CompiledUPMEMProgram",
    "PhaseABI",
    "MLIRProgramABI",
    "ProgramArgument",
    "UPMEMPhase",
    "UPMEMParallelRegion",
    "UPMEMArray",
    "UPMEMDeviceCFragment",
    "UPMEMDeviceSourceManifest",
    "UPMEMDenseTile",
    "UPMEMDotTile",
    "UPMEMRank1Tile",
    "UPMEMProgram",
    "UPMEMProgramCallable",
    "UPMEMScheduleUnavailable",
    "compile_upmem_program",
    "search_upmem_program_schedule",
]
