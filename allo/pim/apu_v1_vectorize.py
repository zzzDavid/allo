# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""MLIR contraction analysis and APU v1 vector-plan candidate generation.

This module is deliberately independent of code generation and runtime.  It
recognizes reduction updates from retained Allo MLIR, recovers logical loop
axes and memory accesses, proves the output/reduction independence needed for
vector execution, and materializes named layout candidates.  No decision
depends on a Python function or kernel name.

The accepted analysis core is a dense one-reduction contraction with one or
more parallel output axes.  APU plan generation currently realizes the
two-output-axis matrix form, for floating point, integer bitwise, and packed
XNOR/popcount arithmetic::

    C[i, j] = C[i, j] + A[i, k] * B[k, j]

Commuted add/multiply operands are accepted.  Casts between loads and the
product are transparent.  ``math.ctpop(xori(xori(a, b), all_ones))`` is
recognized structurally as raw XNOR/popcount; it is not silently reinterpreted
as the bipolar ``2*popcount-word_bits`` operation.

This boundary also makes the PolyBench subset precise. GEMM and the GEMM
stages of 2mm/3mm fit the dense rank-2 form. Matrix-vector reductions expressed
with an explicit singleton second output axis use a native spatial-K group
reduction plan; a genuinely rank-1 MLIR result still needs normalization to
that layout. Batched contractions, triangular domains, and multi-stage fusion
remain separate legality and program-composition concerns.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import re
from typing import Iterable


class ContractionAnalysisError(ValueError):
    """Base class for conservative contraction-analysis failures."""


class NoContractionError(ContractionAnalysisError):
    """Raised when retained MLIR contains no supported contraction update."""


class IllegalContractionError(ContractionAnalysisError):
    """Raised when an update resembles a contraction but is not vector-safe."""


@dataclass(frozen=True)
class LogicalAxis:
    name: str
    extent: int
    lower_bound: int
    upper_bound: int
    step: int
    reduction: bool
    ssa_name: str


@dataclass(frozen=True)
class ValueAccess:
    value: str
    indices: tuple[str, ...]
    mode: str
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class ContractionAnalysis:
    function: str
    axes: tuple[LogicalAxis, ...]
    output_axes: tuple[str, ...]
    parallel_axes: tuple[str, ...]
    reduction_axis: str
    lhs: ValueAccess
    rhs: ValueAccess
    accumulator: ValueAccess
    output: ValueAccess
    multiply_operation: str
    combine_operation: str
    numeric_type: str
    packed_word_bits: int | None = None

    @property
    def axis_extents(self) -> dict[str, int]:
        return {axis.name: axis.extent for axis in self.axes}

    @property
    def values(self) -> tuple[str, str, str]:
        return self.lhs.value, self.rhs.value, self.output.value


@dataclass(frozen=True)
class VectorizationCandidate:
    """A named APU layout plan tied to the proven contraction."""

    name: str
    analysis: ContractionAnalysis
    plan: object


@dataclass
class _RawLoop:
    ssa_name: str
    lower_bound: int
    upper_bound: int
    step: int
    name: str | None = None
    reduction: bool = False

    @property
    def extent(self) -> int:
        distance = self.upper_bound - self.lower_bound
        return 0 if distance <= 0 else (distance + self.step - 1) // self.step


@dataclass(frozen=True)
class _LoadExpr:
    access: ValueAccess


@dataclass(frozen=True)
class _BinaryExpr:
    operation: str
    lhs: str
    rhs: str


@dataclass(frozen=True)
class _AliasExpr:
    source: str


@dataclass(frozen=True)
class _ConstantExpr:
    value: int
    dtype: str


@dataclass(frozen=True)
class _UnaryExpr:
    operation: str
    source: str


@dataclass(frozen=True)
class _ProductMatch:
    operation: str
    lhs: ValueAccess
    rhs: ValueAccess
    packed_word_bits: int | None = None


@dataclass(frozen=True)
class _Store:
    value_ssa: str
    access: ValueAccess
    loop_ssa_names: tuple[str, ...]


_FUNC_START = re.compile(r"\bfunc\.func\s+@(?P<name>[-\w.$]+)")
_LOOP_START = re.compile(
    r"\b(?:affine|scf)\.for\s+(?P<ssa>%[-\w.$]+)\s*=\s*"
    r"(?P<lb>-?\d+)\s+to\s+(?P<ub>-?\d+)"
    r"(?:\s+step\s+(?P<step>\d+))?\s*\{"
)
_LOOP_END = re.compile(r"^\s*}\s*(?:\{(?P<attrs>[^}]*)})?\s*$")
_LOOP_NAME = re.compile(r'loop_name\s*=\s*"(?P<name>[^"]+)"')
_RESULT = r"(?P<result>%[-\w.$]+)"
_LOAD = re.compile(
    rf"{_RESULT}\s*=\s*(?:affine|memref)\.load\s+"
    r"(?P<memref>%[-\w.$]+)\[(?P<indices>[^]]*)]"
    r"(?P<tail>.*)$"
)
_STORE = re.compile(
    r"(?:affine|memref)\.store\s+(?P<value>%[-\w.$]+)\s*,\s*"
    r"(?P<memref>%[-\w.$]+)\[(?P<indices>[^]]*)]"
    r"(?P<tail>.*)$"
)
_BINARY = re.compile(
    rf"{_RESULT}\s*=\s*(?P<op>arith\.(?:(?:add|mul)[fi]|(?:and|or|xor)i))\s+"
    r"(?P<lhs>%[-\w.$]+)\s*,\s*(?P<rhs>%[-\w.$]+)"
)
_CONSTANT = re.compile(
    rf"{_RESULT}\s*=\s*arith\.constant\s+(?P<value>-?\d+)\s*:\s*(?P<dtype>i\d+|ui\d+)"
)
_POPCOUNT = re.compile(
    rf"{_RESULT}\s*=\s*(?P<op>math\.ctpop|allo\.popcount|llvm(?:\.intr)?\.ctpop)"
    r"(?:\s+|\s*\(\s*)(?P<source>%[-\w.$]+)"
)
_GENERIC_LLVM_POPCOUNT = re.compile(
    rf'{_RESULT}\s*=\s*"(?P<op>llvm(?:\.intr)?\.ctpop)"\s*\('
    r"\s*(?P<source>%[-\w.$]+)"
)
_ALIAS = re.compile(
    rf"{_RESULT}\s*=\s*(?:arith\.(?:extf|truncf|extsi|extui|trunci)|"
    r"builtin\.unrealized_conversion_cast)\s+(?P<source>%[-\w.$]+)"
)
_VALUE_ATTR = re.compile(r'(?:from|to)\s*=\s*"(?P<name>[^"]+)"')
_MEMREF_TYPE = re.compile(
    r"memref<(?P<shape>(?:\d+x)*)(?P<dtype>bf16|f16|f32|f64|i\d+|ui\d+)>"
)


def _function_regions(text: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    lines = text.splitlines()
    result: list[tuple[str, tuple[str, ...]]] = []
    index = 0
    while index < len(lines):
        match = _FUNC_START.search(lines[index])
        if match is None:
            index += 1
            continue
        start = index
        depth = 0
        opened = False
        while index < len(lines):
            line = lines[index]
            depth += line.count("{") - line.count("}")
            opened = opened or "{" in line
            index += 1
            if opened and depth == 0:
                break
        result.append((match.group("name"), tuple(lines[start:index])))
    return tuple(result)


def _split_indices(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _memref_type(tail: str) -> tuple[tuple[int, ...], str]:
    matches = tuple(_MEMREF_TYPE.finditer(tail))
    if not matches:
        return (), "unknown"
    match = matches[-1]
    dimensions = tuple(
        int(value) for value in match.group("shape").rstrip("x").split("x") if value
    )
    dtype = match.group("dtype")
    # Allo represents unsigned integer memrefs with signless MLIR integer
    # types and preserves source signedness on memory operations.
    if dtype.startswith("i") and re.search(r"\bunsigned\b", tail):
        dtype = "u" + dtype
    return dimensions, dtype


def _value_name(tail: str, fallback: str) -> str:
    match = _VALUE_ATTR.search(tail)
    return match.group("name") if match is not None else fallback.lstrip("%")


def _parse_function(name: str, lines: tuple[str, ...]):
    loops: list[_RawLoop] = []
    all_loops: list[_RawLoop] = []
    definitions: dict[str, object] = {}
    stores: list[_Store] = []

    for raw_line in lines:
        line = raw_line.split("//", 1)[0].strip()
        if not line:
            continue
        loop = _LOOP_START.search(line)
        if loop is not None:
            parsed = _RawLoop(
                loop.group("ssa"),
                int(loop.group("lb")),
                int(loop.group("ub")),
                int(loop.group("step") or 1),
            )
            loops.append(parsed)
            all_loops.append(parsed)
            continue

        load = _LOAD.search(line)
        if load is not None:
            shape, dtype = _memref_type(load.group("tail"))
            access = ValueAccess(
                _value_name(load.group("tail"), load.group("memref")),
                _split_indices(load.group("indices")),
                "read",
                dtype,
                shape,
            )
            definitions[load.group("result")] = _LoadExpr(access)
            continue

        binary = _BINARY.search(line)
        if binary is not None:
            definitions[binary.group("result")] = _BinaryExpr(
                binary.group("op"), binary.group("lhs"), binary.group("rhs")
            )
            continue

        constant = _CONSTANT.search(line)
        if constant is not None:
            definitions[constant.group("result")] = _ConstantExpr(
                int(constant.group("value")), constant.group("dtype")
            )
            continue

        popcount = _POPCOUNT.search(line) or _GENERIC_LLVM_POPCOUNT.search(line)
        if popcount is not None:
            definitions[popcount.group("result")] = _UnaryExpr(
                "popcount", popcount.group("source")
            )
            continue

        alias = _ALIAS.search(line)
        if alias is not None:
            definitions[alias.group("result")] = _AliasExpr(alias.group("source"))
            continue

        store = _STORE.search(line)
        if store is not None:
            shape, dtype = _memref_type(store.group("tail"))
            stores.append(
                _Store(
                    store.group("value"),
                    ValueAccess(
                        _value_name(store.group("tail"), store.group("memref")),
                        _split_indices(store.group("indices")),
                        "write",
                        dtype,
                        shape,
                    ),
                    tuple(loop.ssa_name for loop in loops),
                )
            )
            continue

        end = _LOOP_END.match(line)
        if end is not None and loops:
            closed = loops.pop()
            attrs = end.group("attrs") or ""
            loop_name = _LOOP_NAME.search(attrs)
            closed.name = (
                loop_name.group("name") if loop_name else closed.ssa_name.lstrip("%")
            )
            closed.reduction = "reduction" in attrs

    # Malformed textual snippets may omit closing attributes.  Stable fallback
    # names still allow legality inference, but not silent dynamic bounds.
    for loop in all_loops:
        if loop.name is None:
            loop.name = loop.ssa_name.lstrip("%")
    return tuple(all_loops), definitions, tuple(stores)


def _resolve_alias(ssa: str, definitions: dict[str, object]) -> str:
    seen = set()
    while isinstance(definitions.get(ssa), _AliasExpr):
        if ssa in seen:
            raise IllegalContractionError("cyclic SSA alias while tracing contraction")
        seen.add(ssa)
        ssa = definitions[ssa].source
    return ssa


def _load(ssa: str, definitions: dict[str, object]) -> ValueAccess | None:
    expression = definitions.get(_resolve_alias(ssa, definitions))
    return expression.access if isinstance(expression, _LoadExpr) else None


def _integer_width(dtype: str) -> int | None:
    match = re.fullmatch(r"u?i(\d+)", str(dtype))
    return int(match.group(1)) if match is not None else None


def _all_ones(ssa: str, definitions: dict[str, object], width: int) -> bool:
    expression = definitions.get(_resolve_alias(ssa, definitions))
    if not isinstance(expression, _ConstantExpr):
        return False
    constant_width = _integer_width(expression.dtype)
    return constant_width == width and expression.value in {-1, (1 << width) - 1}


def _match_product(ssa: str, definitions: dict[str, object]) -> _ProductMatch | None:
    """Recover a two-load product, including canonical packed XNOR/popcount."""

    expression = definitions.get(_resolve_alias(ssa, definitions))
    if isinstance(expression, _BinaryExpr) and expression.operation in {
        "arith.mulf",
        "arith.muli",
        "arith.andi",
        "arith.ori",
        "arith.xori",
    }:
        lhs = _load(expression.lhs, definitions)
        rhs = _load(expression.rhs, definitions)
        if lhs is not None and rhs is not None:
            return _ProductMatch(expression.operation, lhs, rhs)
        return None

    if not isinstance(expression, _UnaryExpr) or expression.operation != "popcount":
        return None
    inverted = definitions.get(_resolve_alias(expression.source, definitions))
    if not isinstance(inverted, _BinaryExpr) or inverted.operation != "arith.xori":
        return None
    for xor_ssa, constant_ssa in (
        (inverted.lhs, inverted.rhs),
        (inverted.rhs, inverted.lhs),
    ):
        xor = definitions.get(_resolve_alias(xor_ssa, definitions))
        if not isinstance(xor, _BinaryExpr) or xor.operation != "arith.xori":
            continue
        lhs = _load(xor.lhs, definitions)
        rhs = _load(xor.rhs, definitions)
        if lhs is None or rhs is None:
            continue
        lhs_width = _integer_width(lhs.dtype)
        rhs_width = _integer_width(rhs.dtype)
        if lhs_width is None or lhs_width != rhs_width:
            raise IllegalContractionError(
                "packed XNOR/popcount operands must have equal integer widths"
            )
        if not _all_ones(constant_ssa, definitions, lhs_width):
            continue
        return _ProductMatch("allo.xnor_popcount", lhs, rhs, lhs_width)
    return None


def _same_location(lhs: ValueAccess, rhs: ValueAccess) -> bool:
    return lhs.value == rhs.value and lhs.indices == rhs.indices


def _axis_name(token: str, by_ssa: dict[str, _RawLoop]) -> str:
    loop = by_ssa.get(token)
    return loop.name if loop is not None else token


def _resolved_access(access: ValueAccess, by_ssa: dict[str, _RawLoop]) -> ValueAccess:
    return ValueAccess(
        access.value,
        tuple(_axis_name(index, by_ssa) for index in access.indices),
        access.mode,
        access.dtype,
        access.shape,
    )


def _analyze_store(function, loops, definitions, store) -> ContractionAnalysis | None:
    root_ssa = _resolve_alias(store.value_ssa, definitions)
    combine = definitions.get(root_ssa)
    if not isinstance(combine, _BinaryExpr) or combine.operation not in {
        "arith.addf",
        "arith.addi",
    }:
        return None

    by_ssa = {loop.ssa_name: loop for loop in loops}
    output = _resolved_access(store.access, by_ssa)
    children = (combine.lhs, combine.rhs)
    accumulator_ssa = product_ssa = None
    for candidate, other in (children, tuple(reversed(children))):
        access = _load(candidate, definitions)
        if access is not None and _same_location(
            _resolved_access(access, by_ssa), output
        ):
            accumulator_ssa, product_ssa = candidate, other
            break
    if accumulator_ssa is None:
        return None

    product = _match_product(product_ssa, definitions)
    if product is None:
        return None
    lhs = product.lhs
    rhs = product.rhs
    accumulator = _load(accumulator_ssa, definitions)
    if accumulator is None:
        return None
    lhs = _resolved_access(lhs, by_ssa)
    rhs = _resolved_access(rhs, by_ssa)
    accumulator = _resolved_access(accumulator, by_ssa)

    active = [by_ssa[ssa] for ssa in store.loop_ssa_names if ssa in by_ssa]
    active_names = {loop.name for loop in active}
    output_axes = tuple(index for index in output.indices if index in active_names)
    if len(output_axes) != len(output.indices) or len(set(output_axes)) != len(
        output_axes
    ):
        raise IllegalContractionError(
            f"{function}: output {output.value}{output.indices} is not indexed by "
            "distinct enclosing loop axes"
        )
    if not output_axes:
        raise IllegalContractionError(
            f"{function}: vector contraction needs at least one output axis"
        )

    marked_reductions = [loop for loop in active if loop.reduction]
    inferred = [
        loop
        for loop in active
        if loop.name not in output_axes
        and loop.name in lhs.indices
        and loop.name in rhs.indices
    ]
    reductions = marked_reductions or inferred
    if len(reductions) != 1:
        raise IllegalContractionError(
            f"{function}: expected one reduction axis, found "
            f"{[loop.name for loop in reductions]}"
        )
    reduction = reductions[0]
    if reduction.name in output.indices:
        raise IllegalContractionError(
            f"{function}: reduction axis {reduction.name!r} indexes the output"
        )
    if reduction.name not in lhs.indices or reduction.name not in rhs.indices:
        raise IllegalContractionError(
            f"{function}: both multiplicands must reference reduction axis "
            f"{reduction.name!r}"
        )
    if _same_location(lhs, output) or _same_location(rhs, output):
        raise IllegalContractionError(
            f"{function}: product reads the accumulator location, creating a "
            "non-reduction loop-carried recurrence"
        )
    legal_indices = set(output_axes) | {reduction.name}
    for access in (lhs, rhs):
        unknown = set(access.indices) - legal_indices
        if unknown:
            raise IllegalContractionError(
                f"{function}: access {access.value}{access.indices} uses "
                f"non-contraction indices {sorted(unknown)}"
            )
    for axis in output_axes:
        if axis not in lhs.indices and axis not in rhs.indices:
            raise IllegalContractionError(
                f"{function}: output axis {axis!r} does not index either input"
            )

    logical_axes = tuple(
        LogicalAxis(
            loop.name,
            loop.extent,
            loop.lower_bound,
            loop.upper_bound,
            loop.step,
            loop is reduction,
            loop.ssa_name,
        )
        for loop in active
    )
    numeric_type = output.dtype if output.dtype != "unknown" else lhs.dtype
    # Allo's unsigned memrefs use signless MLIR types.  Loads retain an
    # ``unsigned`` attribute, while stores currently do not; therefore the
    # store alone can make an unsigned contraction look signed.  The
    # accumulator load names the same location and is the authoritative
    # element signedness for this update.
    if (
        numeric_type.startswith("i")
        and accumulator.dtype.startswith("ui")
        and _integer_width(numeric_type) == _integer_width(accumulator.dtype)
    ):
        numeric_type = accumulator.dtype
        output = ValueAccess(
            output.value,
            output.indices,
            output.mode,
            numeric_type,
            output.shape,
        )
    return ContractionAnalysis(
        function,
        logical_axes,
        output_axes,
        output_axes,
        reduction.name,
        lhs,
        rhs,
        accumulator,
        output,
        product.operation,
        combine.operation,
        numeric_type,
        product.packed_word_bits,
    )


def analyze_apu_v1_contractions(module_or_text) -> tuple[ContractionAnalysis, ...]:
    """Return every legal dense contraction in retained Allo MLIR."""
    text = module_or_text if isinstance(module_or_text, str) else str(module_or_text)
    analyses: list[ContractionAnalysis] = []
    illegal: list[IllegalContractionError] = []
    for function, lines in _function_regions(text):
        loops, definitions, stores = _parse_function(function, lines)
        for store in stores:
            try:
                analysis = _analyze_store(function, loops, definitions, store)
            except IllegalContractionError as error:
                illegal.append(error)
                continue
            if analysis is not None:
                analyses.append(analysis)
    if analyses:
        return tuple(analyses)
    if illegal:
        raise illegal[0]
    raise NoContractionError("retained MLIR contains no supported mul-add contraction")


def analyze_apu_v1_contraction(module_or_text, *, function: str | None = None):
    """Return one contraction, optionally selected by MLIR function name."""
    analyses = analyze_apu_v1_contractions(module_or_text)
    if function is not None:
        analyses = tuple(item for item in analyses if item.function == function)
        if not analyses:
            raise NoContractionError(
                f"no supported contraction in function {function!r}"
            )
    if len(analyses) != 1:
        raise ContractionAnalysisError(
            f"expected exactly one contraction, found {len(analyses)}; select function="
        )
    return analyses[0]


def _layout_api():
    # Kept local so the analyzer remains usable while the provisional planning
    # API evolves independently.
    from .apu_v1_layout import (
        APUReduction,
        APUV1Plan,
        AffineTiledLayout,
        IterationLayout,
        PlanOperation,
        OutputBatching,
        OutputTilePlacement,
        TemporalAxis,
        Transfer,
        TransferLayout,
        TransferRouteStep,
        ReuseWindow,
        ValueLayout,
    )

    return (
        APUReduction,
        APUV1Plan,
        AffineTiledLayout,
        IterationLayout,
        PlanOperation,
        OutputBatching,
        OutputTilePlacement,
        TemporalAxis,
        Transfer,
        TransferLayout,
        TransferRouteStep,
        ReuseWindow,
        ValueLayout,
    )


def _candidate_recipes(analysis: ContractionAnalysis):
    output_axes = analysis.output_axes
    reduction = analysis.reduction_axis
    singleton_gemv = (
        len(output_axes) == 2 and analysis.axis_extents[output_axes[1]] == 1
    )
    return (
        {
            "name": "baseline_spatial_reduction",
            "spatial_axes": output_axes + (reduction,),
            "temporal_axes": (),
            "temporal_strategy": "none",
            "reduction_strategy": "spatial",
            "coalesced": False,
            "broadcast": False,
            "accumulator_block": 1,
            "lookup_storage": "l3",
        },
        {
            "name": "temporal_svp",
            "spatial_axes": output_axes,
            "temporal_axes": (reduction,),
            "temporal_strategy": "svp",
            "reduction_strategy": "temporal",
            "coalesced": False,
            "broadcast": False,
            "accumulator_block": 1,
            "lookup_storage": "l3",
        },
        {
            "name": "temporal_dma_coalescing",
            "spatial_axes": output_axes,
            "temporal_axes": (reduction,),
            "temporal_strategy": "svp",
            "reduction_strategy": "temporal",
            "coalesced": True,
            "broadcast": False,
            "accumulator_block": 1,
            "lookup_storage": "l3",
        },
        {
            "name": "temporal_dma_coalescing_broadcast_friendly",
            "spatial_axes": output_axes,
            "temporal_axes": (reduction,),
            "temporal_strategy": "svp",
            "reduction_strategy": "temporal",
            "coalesced": True,
            "broadcast": True,
            "accumulator_block": 1,
            "lookup_storage": "l3",
        },
        *(
            {
                "name": (
                    "temporal_dma_coalescing_broadcast_friendly_"
                    f"acc{accumulator_block}"
                    + ("_l4" if lookup_storage == "l4" else "")
                ),
                "spatial_axes": output_axes,
                "temporal_axes": (reduction,),
                "temporal_strategy": "svp",
                "reduction_strategy": "temporal",
                "coalesced": True,
                "broadcast": True,
                "accumulator_block": accumulator_block,
                "lookup_storage": lookup_storage,
            }
            for accumulator_block in (2, 4, 8)
            for lookup_storage in ("l3", "l4")
        ),
        *(
            (
                {
                    "name": "spatial_gemv_group_reduction",
                    "spatial_axes": output_axes + (reduction,),
                    "temporal_axes": (),
                    "temporal_strategy": "none",
                    "reduction_strategy": "spatial",
                    "coalesced": True,
                    "broadcast": False,
                    "accumulator_block": 1,
                    "lookup_storage": "l3",
                    "gemv": True,
                },
            )
            if singleton_gemv
            else ()
        ),
    )


def generate_apu_v1_vectorization_candidates(
    module_or_analysis,
) -> tuple[VectorizationCandidate, ...]:
    """Create MICRO-motivated plans for a proven contraction."""
    analysis = (
        module_or_analysis
        if isinstance(module_or_analysis, ContractionAnalysis)
        else analyze_apu_v1_contraction(module_or_analysis)
    )
    if len(analysis.output_axes) != 2:
        next_subset = (
            "One-dimensional matrix-vector reductions are the next enabled "
            "vector subset."
            if len(analysis.output_axes) == 1
            else "Batched-output contractions require an additional output "
            "layout axis."
        )
        raise IllegalContractionError(
            "APU v1 plan generation currently requires two output axes; "
            f"analysis discovered {analysis.output_axes}. {next_subset}"
        )
    (
        APUReduction,
        APUV1Plan,
        AffineTiledLayout,
        IterationLayout,
        PlanOperation,
        OutputBatching,
        OutputTilePlacement,
        TemporalAxis,
        Transfer,
        TransferLayout,
        TransferRouteStep,
        ReuseWindow,
        ValueLayout,
    ) = _layout_api()
    extents = analysis.axis_extents
    axis_order = tuple(axis.name for axis in analysis.axes)
    total_iterations = 1
    for extent in extents.values():
        total_iterations *= extent
    output_elements = 1
    for axis in analysis.output_axes:
        output_elements *= extents[axis]

    def padded(extent):
        return 1 << (int(extent) - 1).bit_length()

    def spatial_tiles(spatial_axes):
        # Grow a validity-masked power-of-two tile within one 32K VR.  Favor
        # the reduction axis first for the baseline group reduction, then
        # round-robin output axes.  Full problem extents remain in metadata.
        tiles = {axis: 1 for axis in spatial_axes}
        priority = tuple(reversed(spatial_axes))
        for axis in priority:
            while True:
                candidate = min(extents[axis], tiles[axis] * 2)
                if candidate == tiles[axis]:
                    break
                trial = dict(tiles)
                trial[axis] = candidate
                lanes = 1
                for value in trial.values():
                    lanes *= padded(value)
                if lanes <= 32768:
                    tiles = trial
                else:
                    break
        return tiles

    def batched_layout(
        logical_extents,
        tile_extents,
        *,
        within_dim="vr_lane",
        batch_dim="vr_batch",
        physical_within_size=None,
    ):
        from ..spmw_linear_layout import LinearLayout

        names = tuple(logical_extents)
        padded_full = {axis: padded(logical_extents[axis]) for axis in names}
        padded_tile = {axis: padded(tile_extents.get(axis, 1)) for axis in names}
        lane_extent = 1
        for extent in padded_tile.values():
            lane_extent *= extent
        if lane_extent > 32768:
            raise IllegalContractionError(
                f"spatial tile needs {lane_extent} lanes, exceeding one 32K VR"
            )
        batch_extents = {axis: padded_full[axis] // padded_tile[axis] for axis in names}
        batch_extent = 1
        for extent in batch_extents.values():
            batch_extent *= extent

        lane_shift = {}
        batch_shift = {}
        trailing_lane = trailing_batch = 1
        for axis in reversed(names):
            lane_shift[axis] = trailing_lane
            batch_shift[axis] = trailing_batch
            trailing_lane *= padded_tile[axis]
            trailing_batch *= batch_extents[axis]
        bases = {}
        for axis in names:
            tile_bits = padded_tile[axis].bit_length() - 1
            full_bits = padded_full[axis].bit_length() - 1
            vectors = [(lane_shift[axis] << bit, 0) for bit in range(tile_bits)]
            vectors.extend(
                (0, batch_shift[axis] << (bit - tile_bits))
                for bit in range(tile_bits, full_bits)
            )
            bases[axis] = vectors
        carrier = LinearLayout(
            bases,
            (within_dim, batch_dim),
            (lane_extent, batch_extent),
        )
        return AffineTiledLayout(
            carrier,
            logical_extents,
            physical_out_sizes=(
                int(physical_within_size or lane_extent),
                batch_extent,
            ),
        )

    def iteration_layout(recipe):
        if recipe.get("gemv"):
            row_axis, column_axis = analysis.output_axes
            reduction_extent = extents[analysis.reduction_axis]
            choices = []
            for reduction_tile in (32, 64, 128, 256, 512, 1024, 2048):
                tile = min(reduction_extent, reduction_tile)
                padded_tile = padded(tile)
                resident_count = (reduction_extent + padded_tile - 1) // padded_tile
                if resident_count > 10:
                    continue
                rows_per_vr = 32768 // padded_tile
                matrix_tiles = (
                    (extents[row_axis] + rows_per_vr - 1) // rows_per_vr
                ) * resident_count
                choices.append(
                    (matrix_tiles, resident_count, -padded_tile, tile, rows_per_vr)
                )
            if not choices:
                raise IllegalContractionError(
                    "spatial GEMV needs at most ten resident vector tiles"
                )
            _calls, _resident, _neg_tile, reduction_tile, rows_per_vr = min(choices)
            spatial = {
                row_axis: min(extents[row_axis], rows_per_vr),
                column_axis: 1,
                analysis.reduction_axis: reduction_tile,
            }
        else:
            spatial = spatial_tiles(recipe["spatial_axes"])
        tiles = {axis: spatial.get(axis, 1) for axis in axis_order}
        layout = batched_layout(extents, tiles)
        temporal = tuple(
            TemporalAxis(
                axis,
                extents[axis],
                carried_values=(analysis.output.value,),
            )
            for axis in recipe["temporal_axes"]
        )
        return IterationLayout(axis_order, layout, temporal), tiles

    candidates = []
    for recipe in _candidate_recipes(analysis):
        planned_iteration_layout, tile_sizes = iteration_layout(recipe)
        output_tile_sizes = spatial_tiles(analysis.output_axes)

        def value_layout(access, *, output=False):
            replicas = (
                ()
                if output
                else tuple(
                    axis for axis in analysis.output_axes if axis not in access.indices
                )
            )
            # Storage follows iteration-axis order, independent of source
            # operand index order.  This keeps the physical lane meaning
            # consistent (row-major output lanes, reduction in vr_batch) and
            # makes replication an actual layout property rather than an ABI
            # transpose.
            varying = set(access.indices) | set(replicas)
            value_axes = tuple(axis for axis in axis_order if axis in varying)
            value_extents = {axis: extents[axis] for axis in value_axes}
            value_tiles = {
                axis: (
                    output_tile_sizes.get(axis, 1)
                    if output
                    else tile_sizes.get(axis, 1)
                )
                for axis in value_axes
            }
            return ValueLayout(
                access.value,
                access.indices,
                replica_axes=replicas,
                layout=batched_layout(value_extents, value_tiles),
            )

        value_layouts = (
            value_layout(analysis.lhs),
            value_layout(analysis.rhs),
            value_layout(analysis.output, output=True),
        )

        row_axis, column_axis = analysis.output_axes
        row_tile = padded(output_tile_sizes[row_axis])
        column_tile = padded(output_tile_sizes[column_axis])
        reduction_tile = padded(tile_sizes.get(analysis.reduction_axis, 1))

        def window(axis, tile):
            return ReuseWindow(axis, tile, extents[axis])

        compute_windows = (
            window(row_axis, row_tile),
            window(column_axis, column_tile),
            window(analysis.reduction_axis, reduction_tile),
        )

        def compact_layout(access, *, out_dim="l4_offset"):
            logical = {axis: extents[axis] for axis in access.indices}
            return AffineTiledLayout.packed(logical, out_dim=out_dim, max_extent=None)

        def endpoint(
            storage,
            access,
            layout,
            *,
            replicas=(),
            role="",
            padded_tile_extents=None,
        ):
            return TransferLayout(
                storage,
                access.indices,
                layout,
                replica_axes=tuple(replicas),
                role=role,
                padded_tile_extents=padded_tile_extents or {},
            )

        def input_transfer(access, compute):
            compute_endpoint = endpoint(
                "compute_vr",
                access,
                compute.layout,
                replicas=compute.replica_axes,
                role="expanded_compute",
            )
            if recipe.get("gemv"):
                if access.value == analysis.lhs.value:
                    gemv_compute_windows = (
                        window(row_axis, padded(tile_sizes[row_axis])),
                        window(column_axis, padded(tile_sizes[column_axis])),
                        window(
                            analysis.reduction_axis,
                            padded(tile_sizes[analysis.reduction_axis]),
                        ),
                    )
                    l4 = endpoint(
                        "l4_expanded",
                        access,
                        compute.layout,
                        replicas=compute.replica_axes,
                        role="packed_gemv_matrix_tiles",
                    )
                    l1 = endpoint(
                        "l1",
                        access,
                        compute.layout,
                        replicas=compute.replica_axes,
                        role="gemv_matrix_dma_staging",
                    )
                    return Transfer(
                        access.value,
                        "in",
                        route=(
                            TransferRouteStep(
                                "dma_l4_l1_32k",
                                l4,
                                l1,
                                executed_at=gemv_compute_windows,
                            ),
                            TransferRouteStep(
                                "load_vr",
                                l1,
                                compute_endpoint,
                                executed_at=gemv_compute_windows,
                            ),
                        ),
                    )

                # Collapse high output-row replica bits in storage. One
                # pre-expanded vector VR is therefore shared by every matrix
                # row tile instead of being materialized once per tile.
                from ..spmw_linear_layout import LinearLayout

                carrier = compute.layout.carrier
                gemv_row_tile = padded(tile_sizes[row_axis])
                row_bits = gemv_row_tile.bit_length() - 1
                resident_bases = {
                    axis: [tuple(vector) for vector in vectors]
                    for axis, vectors in carrier.bases.items()
                }
                resident_bases[row_axis] = resident_bases[row_axis][:row_bits]
                carrier_reduction_batches = max(
                    1,
                    padded(extents[analysis.reduction_axis]) // reduction_tile,
                )
                resident_count = (
                    extents[analysis.reduction_axis] + reduction_tile - 1
                ) // reduction_tile
                resident_carrier = LinearLayout(
                    resident_bases,
                    carrier.out_dims,
                    (32768, carrier_reduction_batches),
                )
                resident_layout = AffineTiledLayout(
                    resident_carrier,
                    {
                        **compute.layout.input_extents,
                        row_axis: min(gemv_row_tile, extents[row_axis]),
                    },
                    physical_out_sizes=(32768, carrier_reduction_batches),
                )
                l4 = endpoint(
                    "l4_expanded",
                    access,
                    resident_layout,
                    replicas=compute.replica_axes,
                    role="resident_gemv_vector_image",
                )
                l1 = endpoint(
                    "l1",
                    access,
                    resident_layout,
                    replicas=compute.replica_axes,
                    role="gemv_vector_dma_staging",
                )
                resident = endpoint(
                    "resident_vr",
                    access,
                    resident_layout,
                    replicas=compute.replica_axes,
                    role="resident_gemv_vector",
                )
                vector_window = (window(analysis.reduction_axis, reduction_tile),)
                row_residency = (window(row_axis, gemv_row_tile),)
                parameters = {
                    "gemv_resident_vector": True,
                    "reduction_tile": reduction_tile,
                    "resident_count": resident_count,
                }
                return Transfer(
                    access.value,
                    "in",
                    route=(
                        TransferRouteStep(
                            "dma_l4_l1_32k",
                            l4,
                            l1,
                            executed_at=vector_window,
                            resident_across=row_residency,
                            parameters=parameters,
                        ),
                        TransferRouteStep(
                            "load_vr",
                            l1,
                            resident,
                            executed_at=vector_window,
                            resident_across=row_residency,
                            parameters=parameters,
                        ),
                    ),
                )
            if not recipe["broadcast"]:
                if recipe["coalesced"]:
                    source = endpoint(
                        "l4_expanded",
                        access,
                        compute.layout,
                        replicas=compute.replica_axes,
                        role="preexpanded_dma_image",
                    )
                    route = (
                        TransferRouteStep(
                            "dma_l4_l1_32k",
                            source,
                            compute_endpoint,
                            temporal_axis=analysis.reduction_axis,
                            executed_at=compute_windows,
                        ),
                    )
                else:
                    source = endpoint(
                        "l4_expanded",
                        access,
                        compute.layout,
                        replicas=compute.replica_axes,
                        role="preexpanded_pio_image",
                    )
                    route = (
                        TransferRouteStep(
                            "direct",
                            source,
                            compute_endpoint,
                            temporal_axis=analysis.reduction_axis,
                            executed_at=compute_windows,
                        ),
                    )
                return Transfer(access.value, "in", route=route)

            replicas = set(compute.replica_axes)
            if column_axis in replicas:
                # A compact row operand is one lookup table per (output-row
                # tile, reduction step).  The unblocked MICRO plan may retain
                # it in L3.  Accumulator-blocked plans lookup directly from L4
                # so dense workloads are not constrained by ARC cache capacity.
                table_tiles = {
                    axis: (
                        min(row_tile, padded(extents[axis])) if axis == row_axis else 1
                    )
                    for axis in access.indices
                }
                table_layout = batched_layout(
                    {axis: extents[axis] for axis in access.indices},
                    table_tiles,
                    within_dim="lookup_entry",
                    batch_dim="lookup_table",
                    physical_within_size=max(32, row_tile),
                )
                l4 = endpoint(
                    "l4",
                    access,
                    table_layout,
                    role="compact_lookup_tables",
                    padded_tile_extents={row_axis: max(32, row_tile)},
                )
                l3 = endpoint(
                    "l3",
                    access,
                    table_layout,
                    role="resident_lookup_tables",
                    padded_tile_extents={row_axis: max(32, row_tile)},
                )
                table_windows = (
                    window(row_axis, row_tile),
                    window(column_axis, column_tile),
                    window(analysis.reduction_axis, 1),
                )
                lookup_source = l3 if recipe["lookup_storage"] == "l3" else l4
                route = (
                    *(
                        (
                            TransferRouteStep(
                                "dma_l4_l3",
                                l4,
                                l3,
                                parameters={"whole_operand": True},
                            ),
                        )
                        if recipe["lookup_storage"] == "l3"
                        else ()
                    ),
                    TransferRouteStep(
                        "lookup",
                        lookup_source,
                        compute_endpoint,
                        temporal_axis=analysis.reduction_axis,
                        executed_at=table_windows,
                        parameters={
                            "table_size": max(32, row_tile),
                            "group_size": column_tile,
                            "subgroup_size": 1,
                        },
                    ),
                )
                return Transfer(access.value, "in", route=route)

            if row_axis in replicas:
                # Eight contiguous reduction rows share each resident VR.  The
                # packed RHS is loaded once and remains live while all output
                # row batches execute; subgroup duplication expands the chosen
                # 1K row into the 32 row groups of a compute VR.
                rows_per_vr = min(8, padded(extents[analysis.reduction_axis]), row_tile)
                stage_extents = {axis: extents[axis] for axis in access.indices}
                stage_tiles = {
                    axis: (
                        rows_per_vr
                        if axis == analysis.reduction_axis
                        else min(column_tile, padded(extents[axis]))
                    )
                    for axis in access.indices
                }
                l4_layout = batched_layout(
                    stage_extents,
                    stage_tiles,
                    within_dim="l4_chunk_offset",
                    batch_dim="resident_chunk",
                    physical_within_size=32768,
                )
                l1_layout = batched_layout(
                    stage_extents,
                    stage_tiles,
                    within_dim="l1_offset",
                    batch_dim="resident_chunk",
                    physical_within_size=32768,
                )
                duplicate_group_size = rows_per_vr * column_tile
                active_vr_extent = row_tile * column_tile
                if active_vr_extent % duplicate_group_size:
                    raise IllegalContractionError(
                        "resident RHS group must divide the active compute VR"
                    )
                resident_repeats = active_vr_extent // duplicate_group_size
                resident_extents = {
                    "resident_repeat": resident_repeats,
                    **stage_extents,
                }
                resident_tiles = {
                    "resident_repeat": resident_repeats,
                    **stage_tiles,
                }
                resident_layout = batched_layout(
                    resident_extents,
                    resident_tiles,
                    within_dim="vr_lane",
                    batch_dim="resident_vr",
                    physical_within_size=32768,
                )
                l4 = endpoint("l4", access, l4_layout, role="packed_rhs")
                l1 = endpoint("l1", access, l1_layout, role="dma_staging")
                resident = endpoint(
                    "resident_vr",
                    access,
                    resident_layout,
                    replicas=("resident_repeat",),
                    role="resident_rhs",
                )
                load_window = (
                    window(column_axis, column_tile),
                    window(analysis.reduction_axis, rows_per_vr),
                )
                row_residency = (window(row_axis, row_tile),)
                expand_windows = (
                    window(row_axis, row_tile),
                    window(column_axis, column_tile),
                    window(analysis.reduction_axis, 1),
                )
                shared_parameters = {
                    "rows_per_vr": rows_per_vr,
                    "group_size": duplicate_group_size,
                    "subgroup_size": column_tile,
                    "resident_vr_extent": 32768,
                    "active_vr_extent": active_vr_extent,
                    "replication_factor": resident_repeats,
                }
                route = (
                    TransferRouteStep(
                        "dma_l4_l1_32k",
                        l4,
                        l1,
                        executed_at=load_window,
                        resident_across=row_residency,
                        parameters=shared_parameters,
                    ),
                    TransferRouteStep(
                        "load_vr",
                        l1,
                        resident,
                        executed_at=load_window,
                        resident_across=row_residency,
                        parameters=shared_parameters,
                    ),
                    TransferRouteStep(
                        "duplicate_subgroup",
                        resident,
                        compute_endpoint,
                        temporal_axis=analysis.reduction_axis,
                        executed_at=expand_windows,
                        parameters=shared_parameters,
                    ),
                )
                return Transfer(access.value, "in", route=route)
            raise IllegalContractionError(
                f"input {access.value!r} has no output-axis replication to expand"
            )

        output_compute = endpoint(
            "compute_vr",
            analysis.output,
            value_layouts[2].layout,
            role="output_compute",
        )
        output_l4 = endpoint(
            "l4",
            analysis.output,
            value_layouts[2].layout,
            role="tiled_compact_output",
        )
        output_l1 = endpoint(
            "l1",
            analysis.output,
            value_layouts[2].layout,
            role="output_dma_staging",
        )
        output_windows = (
            window(row_axis, row_tile),
            window(column_axis, column_tile),
        )
        output_ingress_route = (
            (
                TransferRouteStep(
                    "dma_l4_l1_32k",
                    output_l4,
                    output_l1,
                    executed_at=output_windows,
                ),
                TransferRouteStep(
                    "load_vr",
                    output_l1,
                    output_compute,
                    executed_at=output_windows,
                ),
            )
            if recipe["coalesced"] and not recipe.get("gemv")
            else (
                TransferRouteStep(
                    "direct",
                    output_l4,
                    output_compute,
                    executed_at=output_windows,
                ),
            )
        )
        output_egress_route = (
            TransferRouteStep(
                (
                    "dma_vr_l4"
                    if recipe["coalesced"] and not recipe.get("gemv")
                    else "direct"
                ),
                output_compute,
                output_l4,
                executed_at=output_windows,
            ),
        )
        transfers = (
            input_transfer(analysis.lhs, value_layouts[0]),
            input_transfer(analysis.rhs, value_layouts[1]),
            Transfer(analysis.output.value, "in", route=output_ingress_route),
            Transfer(analysis.output.value, "out", route=output_egress_route),
        )
        reduction_strategy = APUReduction(
            analysis.reduction_axis,
            kind=recipe["reduction_strategy"],
            group_size=(
                padded(tile_sizes[analysis.reduction_axis])
                if analysis.reduction_axis in tile_sizes
                and recipe["reduction_strategy"] == "spatial"
                else None
            ),
            partial=(
                recipe["reduction_strategy"] == "temporal"
                or tile_sizes.get(analysis.reduction_axis, 1)
                < extents[analysis.reduction_axis]
            ),
        )
        loop_roles = tuple(
            "reduction" if axis == analysis.reduction_axis else "parallel_output"
            for axis in axis_order
        )
        product_primitive = {
            "arith.mulf": "MUL",
            "arith.muli": "MUL",
            "arith.andi": "AND",
            "arith.ori": "OR",
            "arith.xori": "XOR",
            "allo.xnor_popcount": "XNOR_POPCOUNT",
        }[analysis.multiply_operation]
        output_tiles = 1
        for axis in analysis.output_axes:
            tile = tile_sizes.get(axis, 1)
            output_tiles *= (extents[axis] + tile - 1) // tile
        reduction_tile = tile_sizes.get(analysis.reduction_axis, 1)
        reduction_tiles = (
            extents[analysis.reduction_axis] + reduction_tile - 1
        ) // reduction_tile
        binary_primitives = (
            "XOR_16",
            "NOT_16",
            "POPCOUNT_16",
        )
        if recipe["reduction_strategy"] == "spatial":
            vr_batches = output_tiles * reduction_tiles
            vector_compute_calls = vr_batches
            group_reduce_calls = vr_batches
            temporal_steps = reduction_tiles
            product_operations = (
                tuple(
                    PlanOperation(
                        primitive,
                        vector_compute_calls,
                        dtype=analysis.numeric_type,
                        loop_roles=loop_roles,
                    )
                    for primitive in binary_primitives
                )
                if product_primitive == "XNOR_POPCOUNT"
                else (
                    PlanOperation(
                        product_primitive,
                        vector_compute_calls,
                        dtype=analysis.numeric_type,
                        loop_roles=loop_roles,
                    ),
                )
            )
            operations = product_operations + (
                PlanOperation(
                    "GROUP_REDUCE",
                    group_reduce_calls,
                    dtype=analysis.numeric_type,
                    loop_roles=("reduction",),
                ),
            )
        else:
            temporal_steps = extents[analysis.reduction_axis]
            vr_batches = output_tiles * temporal_steps
            vector_compute_calls = vr_batches
            group_reduce_calls = 0
            product_operations = (
                tuple(
                    PlanOperation(
                        primitive,
                        vector_compute_calls,
                        dtype=analysis.numeric_type,
                        loop_roles=loop_roles,
                    )
                    for primitive in binary_primitives
                )
                if product_primitive == "XNOR_POPCOUNT"
                else (
                    PlanOperation(
                        product_primitive,
                        vector_compute_calls,
                        dtype=analysis.numeric_type,
                        loop_roles=loop_roles,
                    ),
                )
            )
            operations = product_operations + (
                PlanOperation(
                    "ADD",
                    vector_compute_calls,
                    dtype=analysis.numeric_type,
                    loop_roles=loop_roles,
                ),
            )
        temporal_trip_count = (
            extents[analysis.reduction_axis]
            if recipe["temporal_axes"]
            else (
                extents[analysis.reduction_axis]
                + tile_sizes[analysis.reduction_axis]
                - 1
            )
            // tile_sizes[analysis.reduction_axis]
        )

        work_counts = {
            axis: (extents[axis] + tile_sizes[axis] - 1) // tile_sizes[axis]
            for axis in analysis.output_axes
        }
        placements = []
        for tile_indices in product(
            *(range(work_counts[axis]) for axis in analysis.output_axes)
        ):
            origin = {
                axis: index * tile_sizes[axis]
                for axis, index in zip(analysis.output_axes, tile_indices)
            }
            iteration_indices = {
                axis: (0 if axis == analysis.reduction_axis else origin[axis])
                for axis in axis_order
            }
            output_coordinate = value_layouts[2].layout.coordinate(**origin)
            placements.append(
                OutputTilePlacement(
                    # Valid logical tiles are a compact execution stream.
                    # The F2 carrier may contain padded outer coordinates;
                    # those are storage capacity, not executable work IDs.
                    work_tile=len(placements),
                    physical_output_batch=output_coordinate["vr_batch"],
                    lane_offset=output_coordinate["vr_lane"],
                    logical_origin=tuple(origin[axis] for axis in analysis.output_axes),
                    logical_shape=tuple(
                        min(tile_sizes[axis], extents[axis] - origin[axis])
                        for axis in analysis.output_axes
                    ),
                )
            )
        placements.sort(key=lambda item: item.work_tile)
        output_batching = OutputBatching(
            analysis.output_axes,
            tuple((axis, extents[axis]) for axis in analysis.output_axes),
            tuple((axis, tile_sizes[axis]) for axis in analysis.output_axes),
            tuple((axis, output_tile_sizes[axis]) for axis in analysis.output_axes),
            analysis.reduction_axis,
            extents[analysis.reduction_axis],
            reduction_tile,
            tuple(placements),
        )
        physical_output_batches = output_batching.physical_output_batches
        if physical_output_batches > value_layouts[2].layout.out_sizes[1]:
            raise IllegalContractionError(
                "output batching exceeds the physical output layout capacity"
            )
        accumulator_block = int(recipe["accumulator_block"])
        if accumulator_block > physical_output_batches:
            continue
        compute_tiles_per_output = output_batching.work_tiles_per_output_batch
        work_steps_per_output = output_batching.work_steps_per_output_batch
        metadata = {
            "analysis": "mlir-access-structure",
            "function": analysis.function,
            "dtype": analysis.numeric_type,
            "numeric_kind": (
                "floating"
                if analysis.numeric_type.startswith(("f", "bf"))
                else "integer"
            ),
            "problem_shape": {
                "M": extents[analysis.output_axes[0]],
                "N": (
                    extents[analysis.output_axes[1]]
                    if len(analysis.output_axes) > 1
                    else 1
                ),
                "K": extents[analysis.reduction_axis],
            },
            "axis_extents": dict(extents),
            "loop_roles": {
                **{axis: "parallel_output" for axis in analysis.output_axes},
                analysis.reduction_axis: "reduction",
            },
            "tile_sizes": {
                **{axis: tile_sizes.get(axis, 1) for axis in axis_order},
            },
            "temporal_trip_count": temporal_trip_count,
            "operation_counts": {
                "scalar_work_items": total_iterations,
                "output_elements": output_elements,
                "output_tiles": physical_output_batches,
                "physical_output_batches": physical_output_batches,
                "work_output_tiles": output_tiles,
                "compute_tiles_per_output": compute_tiles_per_output,
                "reduction_tiles": reduction_tiles,
                "work_steps_per_output": work_steps_per_output,
                "vr_batches": vr_batches,
                "vector_compute_calls": vector_compute_calls,
                "group_reduce_calls": group_reduce_calls,
                "temporal_steps": temporal_steps,
            },
            "scalar_work_items": total_iterations,
            "output_tiles": physical_output_batches,
            "physical_output_batches": physical_output_batches,
            "work_output_tiles": output_tiles,
            "compute_tiles_per_output": compute_tiles_per_output,
            "reduction_tiles": reduction_tiles,
            "work_steps_per_output": work_steps_per_output,
            "vr_batches": vr_batches,
            "vector_compute_calls": vector_compute_calls,
            "group_reduce_calls": group_reduce_calls,
            "temporal_steps": temporal_steps,
            "transfer_facts": {
                "input_values": (analysis.lhs.value, analysis.rhs.value),
                "output_value": analysis.output.value,
                "input_transfer_calls": 2 * vr_batches,
                "output_transfer_calls": physical_output_batches,
                "coalesced": recipe["coalesced"],
                "broadcast_values": tuple(
                    transfer.value for transfer in transfers if transfer.broadcast
                ),
            },
            "accumulator_persistence": {
                "scope": (
                    "work_tile"
                    if compute_tiles_per_output > 1
                    else "physical_output_batch"
                ),
                "reduction_tiles": reduction_tiles,
                "commit": (
                    "scatter_to_physical_output"
                    if compute_tiles_per_output > 1
                    else "egress_after_work_steps"
                ),
            },
            "accumulator_block": accumulator_block,
            "gemv_spatial_reduction": bool(recipe.get("gemv")),
            "product_operation": analysis.multiply_operation,
            "packed_word_bits": analysis.packed_word_bits,
            "combine_operation": analysis.combine_operation,
            "accesses": {
                "lhs": {
                    "value": analysis.lhs.value,
                    "indices": analysis.lhs.indices,
                },
                "rhs": {
                    "value": analysis.rhs.value,
                    "indices": analysis.rhs.indices,
                },
                "output": {
                    "value": analysis.output.value,
                    "indices": analysis.output.indices,
                },
            },
        }
        plan = APUV1Plan(
            name=recipe["name"],
            iteration_layout=planned_iteration_layout,
            value_layouts=value_layouts,
            transfers=transfers,
            temporal_strategy=recipe["temporal_strategy"],
            reduction_strategy=reduction_strategy,
            operations=operations,
            metadata=metadata,
            output_batching=output_batching,
            accumulator_block=accumulator_block,
        )
        candidates.append(VectorizationCandidate(recipe["name"], analysis, plan))
    return tuple(candidates)


def generate_apu_v1_plans(module_or_analysis) -> tuple[object, ...]:
    """Return only the immutable :class:`APUV1Plan` values."""

    return tuple(
        candidate.plan
        for candidate in generate_apu_v1_vectorization_candidates(module_or_analysis)
    )


__all__ = [
    "ContractionAnalysisError",
    "NoContractionError",
    "IllegalContractionError",
    "LogicalAxis",
    "ValueAccess",
    "ContractionAnalysis",
    "VectorizationCandidate",
    "analyze_apu_v1_contraction",
    "analyze_apu_v1_contractions",
    "generate_apu_v1_vectorization_candidates",
    "generate_apu_v1_plans",
]
