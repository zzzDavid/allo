# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic rank-N single-reduction planning for APUg2.

The existing target-neutral contraction parser intentionally rejects batch
axes absent from an accumulator memref.  That is correct for its original
matrix-vector legality boundary, but batched kernels such as Doitgen use a
local output vector inside enclosing batch loops.  This module reuses the same
retained-MLIR parser and expression matcher while generalizing only that
legality rule.  It then builds immutable projection maps, structured program
manifests, and :class:`APUG2DotTiling` values without consulting workload or
function names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re

from . import apu_v1_vectorize as _analysis_impl
from .apu_g2_ir import APUG2ModuleManifest, discover_apu_g2_module_manifest
from .apu_g2_layout import APUG2DotTiling
from .contraction_analysis import (
    ContractionAnalysis,
    IllegalContractionError,
    LogicalAxis,
    NoContractionError,
    ValueAccess,
)


class UnsupportedAPUG2RankNContractionError(IllegalContractionError):
    """Retained MLIR is not a legal APUg2 rank-N uint16 dot contraction."""


_SSA_TOKEN = re.compile(r"%[-\w.$]+")


def _uniquify_scoped_loop_ssa(lines: tuple[str, ...]) -> tuple[str, ...]:
    """Give reused affine loop IV names stable scope-unique spellings.

    MLIR permits a textual SSA spelling to be reused after its defining loop
    closes.  The shared parser normally resolves loop names through one flat
    table, so a later Doitgen writeback loop can otherwise rename the earlier
    contraction's ``p`` axis.  Rewriting only loop-IV spellings preserves the
    retained program while making lexical scope explicit to that parser.
    """

    active: list[tuple[str, str]] = []
    counts: dict[str, int] = {}

    def replace_active(line: str) -> str:
        mapping = {source: target for source, target in active}
        return _SSA_TOKEN.sub(
            lambda match: mapping.get(match.group(), match.group()), line
        )

    rewritten = []
    for raw_line in lines:
        loop = _analysis_impl._LOOP_START.search(raw_line)
        if loop is not None:
            source = loop.group("ssa")
            line = replace_active(raw_line)
            count = counts.get(source, 0)
            counts[source] = count + 1
            target = f"{source}__apu_g2_{count}"
            line = _SSA_TOKEN.sub(
                lambda match: target if match.group() == source else match.group(),
                line,
            )
            rewritten.append(line)
            active.append((source, target))
            continue
        line = replace_active(raw_line)
        rewritten.append(line)
        if _analysis_impl._LOOP_END.match(line.strip()) is not None and active:
            active.pop()
    return tuple(rewritten)


@dataclass(frozen=True)
class APUG2AffineAccessMap:
    """A direct affine projection/permutation from a dot domain to a memref.

    The retained contraction parser accepts direct loop-IV indices.  Such an
    access is an affine map with zero offsets and unit coefficients, represented
    exactly by ``result_axes`` and ``axis_positions``.
    """

    value: str
    dtype: str
    shape: tuple[int, ...]
    mode: str
    domain_axes: tuple[str, ...]
    result_axes: tuple[str, ...]
    axis_positions: tuple[int, ...] = field(init=False)

    def __post_init__(self):
        if not self.value or not self.dtype:
            raise ValueError("APUg2 affine access maps require value and dtype")
        shape = tuple(int(extent) for extent in self.shape)
        domain = tuple(self.domain_axes)
        results = tuple(self.result_axes)
        if len(shape) != len(results):
            raise ValueError(
                f"access {self.value!r} rank {len(shape)} does not match "
                f"index map {results}"
            )
        if len(domain) != len(set(domain)):
            raise ValueError("APUg2 affine access domain axes must be unique")
        if any(axis not in domain for axis in results):
            unknown = sorted(set(results) - set(domain))
            raise ValueError(f"access {self.value!r} uses unknown axes {unknown}")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "domain_axes", domain)
        object.__setattr__(self, "result_axes", results)
        object.__setattr__(
            self, "axis_positions", tuple(domain.index(axis) for axis in results)
        )

    @property
    def broadcast_axes(self) -> tuple[str, ...]:
        return tuple(axis for axis in self.domain_axes if axis not in self.result_axes)

    def apply(self, coordinates: tuple[int, ...]) -> tuple[int, ...]:
        if len(coordinates) != len(self.domain_axes):
            raise ValueError(
                f"access {self.value!r} needs {len(self.domain_axes)} coordinates"
            )
        return tuple(int(coordinates[position]) for position in self.axis_positions)

    def manifest(self) -> dict[str, object]:
        return {
            "value": self.value,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "mode": self.mode,
            "domain_axes": list(self.domain_axes),
            "result_axes": list(self.result_axes),
            "axis_positions": list(self.axis_positions),
            "broadcast_axes": list(self.broadcast_axes),
        }


@dataclass(frozen=True)
class APUG2RankNContractionPlan:
    """One flattened rank-N dot domain and its physical temporal tiling."""

    analysis: ContractionAnalysis = field(repr=False)
    module: APUG2ModuleManifest = field(repr=False)
    region_id: str
    dot_axes: tuple[tuple[str, int], ...]
    batch_axes: tuple[str, ...]
    output_axes: tuple[str, ...]
    reduction_axis: tuple[str, int]
    lhs: APUG2AffineAccessMap
    rhs: APUG2AffineAccessMap
    initial_output: APUG2AffineAccessMap
    output: APUG2AffineAccessMap
    tiling: APUG2DotTiling

    def __post_init__(self):
        dot_axes = tuple((str(name), int(extent)) for name, extent in self.dot_axes)
        if not dot_axes or len({name for name, _extent in dot_axes}) != len(dot_axes):
            raise ValueError("APUg2 dot axes must be nonempty and unique")
        if any(extent <= 0 for _name, extent in dot_axes):
            raise ValueError("APUg2 dot axis extents must be positive")
        reduction_axis = (str(self.reduction_axis[0]), int(self.reduction_axis[1]))
        if reduction_axis[1] <= 0:
            raise ValueError("APUg2 reduction extent must be positive")
        if reduction_axis[0] in {name for name, _extent in dot_axes}:
            raise ValueError("APUg2 reduction axis cannot also be a dot output axis")
        object.__setattr__(self, "dot_axes", dot_axes)
        object.__setattr__(self, "batch_axes", tuple(self.batch_axes))
        object.__setattr__(self, "output_axes", tuple(self.output_axes))
        object.__setattr__(self, "reduction_axis", reduction_axis)
        if self.tiling.output_extent != math.prod(extent for _name, extent in dot_axes):
            raise ValueError("APUg2 dot tiling does not cover the flattened output")
        if self.tiling.reduction_extent != reduction_axis[1]:
            raise ValueError("APUg2 dot tiling reduction extent does not match")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= 0xFFFF
            for value in self.epilogue
        ):
            raise ValueError("APUg2 contraction epilogue must fit uint16")

    @property
    def flat_output_extent(self) -> int:
        return self.tiling.output_extent

    @property
    def batch_local_accumulator(self) -> bool:
        """Whether the accumulator omits enclosing batch axes and needs reset."""

        return any(
            axis not in self.initial_output.result_axes for axis in self.batch_axes
        )

    @property
    def epilogue(self) -> tuple[int, int]:
        """Return ``alpha * dot + beta * accumulator`` from retained semantics."""

        return (
            int(self.analysis.product_coefficient),
            int(self.analysis.accumulator_coefficient),
        )

    def flatten_output(self, **coordinates: int) -> int:
        expected = {name for name, _extent in self.dot_axes}
        if set(coordinates) != expected:
            raise ValueError(
                f"flatten_output needs axes {sorted(expected)}, got "
                f"{sorted(coordinates)}"
            )
        flat = 0
        for name, extent in self.dot_axes:
            coordinate = int(coordinates[name])
            if not 0 <= coordinate < extent:
                raise IndexError(f"axis {name}={coordinate} is outside extent {extent}")
            flat = flat * extent + coordinate
        return flat

    def unflatten_output(self, flat: int) -> tuple[int, ...]:
        flat = int(flat)
        if not 0 <= flat < self.flat_output_extent:
            raise IndexError(
                f"flat output {flat} is outside extent {self.flat_output_extent}"
            )
        result = [0] * len(self.dot_axes)
        for index in range(len(self.dot_axes) - 1, -1, -1):
            extent = self.dot_axes[index][1]
            result[index] = flat % extent
            flat //= extent
        return tuple(result)

    def manifest(self) -> dict[str, object]:
        return {
            "kind": "apu-g2-rank-n-contraction",
            "region_id": self.region_id,
            "dot_axes": [
                {"name": name, "extent": extent} for name, extent in self.dot_axes
            ],
            "batch_axes": list(self.batch_axes),
            "output_axes": list(self.output_axes),
            "reduction_axis": {
                "name": self.reduction_axis[0],
                "extent": self.reduction_axis[1],
            },
            "flat_output_extent": self.flat_output_extent,
            "tile_capacity": self.tiling.tile_capacity,
            "tile_count": self.tiling.tile_count,
            "reduction_tile_extent": self.tiling.reduction_tile_extent,
            "reduction_tile_count": self.tiling.reduction_tile_count,
            "hardware_tasks": self.tiling.task_count,
            "batch_local_accumulator": self.batch_local_accumulator,
            "epilogue": {
                "alpha": self.epilogue[0],
                "beta": self.epilogue[1],
            },
            "lhs": self.lhs.manifest(),
            "rhs": self.rhs.manifest(),
            "initial_output": self.initial_output.manifest(),
            "output": self.output.manifest(),
        }


def _generalized_analysis(
    function,
    loops,
    definitions,
    store,
    *,
    accumulator_coefficient=1,
):
    """Generalize the shared parser's legality check to enclosing batch axes."""

    root_ssa = _analysis_impl._resolve_alias(store.value_ssa, definitions)
    combine = definitions.get(root_ssa)
    if not isinstance(combine, _analysis_impl._BinaryExpr) or combine.operation not in {
        "arith.addf",
        "arith.addi",
    }:
        return None

    by_ssa = {loop.ssa_name: loop for loop in loops}
    output = _analysis_impl._resolved_access(store.access, by_ssa)
    accumulator_ssa = product_ssa = None
    for candidate, other in (
        (combine.lhs, combine.rhs),
        (combine.rhs, combine.lhs),
    ):
        access = _analysis_impl._load(candidate, definitions)
        if access is not None and _analysis_impl._same_location(
            _analysis_impl._resolved_access(access, by_ssa), output
        ):
            accumulator_ssa, product_ssa = candidate, other
            break
    if accumulator_ssa is None:
        return None

    product = _analysis_impl._match_product(
        product_ssa,
        definitions,
        allow_integer_coefficient=True,
    )
    accumulator = _analysis_impl._load(accumulator_ssa, definitions)
    if accumulator is None:
        return None
    if product is None:
        raise UnsupportedAPUG2RankNContractionError(
            f"{function}: accumulator update has an unsupported contribution"
        )
    lhs = _analysis_impl._resolved_access(product.lhs, by_ssa)
    rhs = _analysis_impl._resolved_access(product.rhs, by_ssa)
    accumulator = _analysis_impl._resolved_access(accumulator, by_ssa)

    active = [by_ssa[ssa] for ssa in store.loop_ssa_names if ssa in by_ssa]
    active_names = {loop.name for loop in active}
    output_axes = tuple(index for index in output.indices if index in active_names)
    if len(output_axes) != len(output.indices) or len(set(output_axes)) != len(
        output_axes
    ):
        raise UnsupportedAPUG2RankNContractionError(
            f"{function}: output {output.value}{output.indices} is not indexed by "
            "distinct enclosing axes"
        )
    if not output_axes:
        raise UnsupportedAPUG2RankNContractionError(
            f"{function}: contraction needs at least one output axis"
        )

    marked = [loop for loop in active if loop.reduction]
    inferred = [
        loop
        for loop in active
        if loop.name not in output_axes
        and loop.name in lhs.indices
        and loop.name in rhs.indices
    ]
    reductions = marked or inferred
    if len(reductions) != 1:
        raise UnsupportedAPUG2RankNContractionError(
            f"{function}: expected one reduction axis, found "
            f"{[loop.name for loop in reductions]}"
        )
    reduction = reductions[0]
    if reduction.name in output.indices:
        raise UnsupportedAPUG2RankNContractionError(
            f"{function}: reduction axis indexes the output"
        )
    if reduction.name not in lhs.indices or reduction.name not in rhs.indices:
        raise UnsupportedAPUG2RankNContractionError(
            f"{function}: both operands must reference the reduction axis"
        )
    if _analysis_impl._same_location(lhs, output) or _analysis_impl._same_location(
        rhs, output
    ):
        raise UnsupportedAPUG2RankNContractionError(
            f"{function}: product reads its accumulator location"
        )

    parallel = tuple(loop for loop in active if loop is not reduction)
    parallel_axes = tuple(loop.name for loop in parallel)
    legal_indices = set(parallel_axes) | {reduction.name}
    for access in (lhs, rhs):
        unknown = set(access.indices) - legal_indices
        if unknown:
            raise UnsupportedAPUG2RankNContractionError(
                f"{function}: access {access.value}{access.indices} uses unknown "
                f"axes {sorted(unknown)}"
            )
    for axis in parallel_axes:
        if (
            axis not in lhs.indices
            and axis not in rhs.indices
            and axis not in output.indices
        ):
            raise UnsupportedAPUG2RankNContractionError(
                f"{function}: enclosing axis {axis!r} does not index the dot"
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
    if (
        numeric_type.startswith("i")
        and accumulator.dtype.startswith("ui")
        and _analysis_impl._integer_width(numeric_type)
        == _analysis_impl._integer_width(accumulator.dtype)
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
        function=function,
        axes=logical_axes,
        output_axes=output_axes,
        parallel_axes=parallel_axes,
        reduction_axis=reduction.name,
        lhs=lhs,
        rhs=rhs,
        accumulator=accumulator,
        output=output,
        multiply_operation=product.operation,
        combine_operation=combine.operation,
        numeric_type=numeric_type,
        packed_word_bits=product.packed_word_bits,
        product_coefficient=product.coefficient,
        accumulator_coefficient=accumulator_coefficient,
    )


def _preceding_accumulator_coefficient(
    function, stores, store_index, loops, definitions
) -> int:
    by_ssa = {loop.ssa_name: loop for loop in loops}
    current = _analysis_impl._resolved_access(stores[store_index].access, by_ssa)
    expected_scope = tuple(
        ssa
        for ssa in stores[store_index].loop_ssa_names
        if ssa in by_ssa and not by_ssa[ssa].reduction
    )
    for previous in reversed(stores[:store_index]):
        target = _analysis_impl._resolved_access(previous.access, by_ssa)
        if not _analysis_impl._same_location(target, current):
            continue
        if previous.loop_ssa_names != expected_scope:
            previous_analysis = _generalized_analysis(
                function,
                loops,
                definitions,
                previous,
            )
            if previous_analysis is not None:
                return 1
            raise UnsupportedAPUG2RankNContractionError(
                f"{function}: accumulator prescale does not dominate its reduction"
            )
        scaled = _analysis_impl._match_scaled_integer_load(
            previous.value_ssa, definitions
        )
        if scaled is not None:
            source, coefficient = scaled
            source = _analysis_impl._resolved_access(source, by_ssa)
            if _analysis_impl._same_location(source, current):
                return coefficient
        constant = _analysis_impl._integer_constant_value(
            previous.value_ssa, definitions
        )
        if constant == 0:
            return 0
        previous_analysis = _generalized_analysis(
            function,
            loops,
            definitions,
            previous,
        )
        if previous_analysis is not None:
            return 1
        raise UnsupportedAPUG2RankNContractionError(
            f"{function}: preceding accumulator write is not a proven prescale"
        )
    return 1


def analyze_apu_g2_rank_n_contractions(module_or_text):
    """Return legal rank-N, one-reduction contractions from retained MLIR."""

    text = module_or_text if isinstance(module_or_text, str) else str(module_or_text)
    analyses = []
    illegal = []
    for function, lines in _analysis_impl._function_regions(text):
        scoped_lines = _uniquify_scoped_loop_ssa(lines)
        loops, definitions, stores = _analysis_impl._parse_function(
            function, scoped_lines
        )
        for store_index, store in enumerate(stores):
            try:
                analysis = _generalized_analysis(
                    function,
                    loops,
                    definitions,
                    store,
                    accumulator_coefficient=_preceding_accumulator_coefficient(
                        function,
                        stores,
                        store_index,
                        loops,
                        definitions,
                    ),
                )
            except UnsupportedAPUG2RankNContractionError as error:
                illegal.append(error)
                continue
            if analysis is not None:
                analyses.append(analysis)
    if illegal:
        raise illegal[0]
    if analyses:
        return tuple(analyses)
    raise NoContractionError("retained MLIR contains no rank-N mul-add contraction")


def _access_map(
    access: ValueAccess, domain_axes: tuple[str, ...]
) -> APUG2AffineAccessMap:
    return APUG2AffineAccessMap(
        value=access.value,
        dtype=access.dtype,
        shape=access.shape,
        mode=access.mode,
        domain_axes=domain_axes,
        result_axes=access.indices,
    )


def plan_apu_g2_rank_n_contractions(module_or_text):
    """Plan every structurally legal uint16 rank-N contraction in a module."""

    analyses = analyze_apu_g2_rank_n_contractions(module_or_text)
    module = discover_apu_g2_module_manifest(analyses)
    plans = []
    for index, analysis in enumerate(analyses):
        if (
            analysis.numeric_type != "ui16"
            or analysis.multiply_operation != "arith.muli"
            or analysis.combine_operation != "arith.addi"
        ):
            raise UnsupportedAPUG2RankNContractionError(
                "APUg2 rank-N dots require uint16 multiply-add semantics"
            )
        extents = analysis.axis_extents
        dot_axis_names = tuple(analysis.parallel_axes)
        dot_axes = tuple((axis, extents[axis]) for axis in dot_axis_names)
        output_axes = tuple(analysis.output_axes)
        batch_axes = tuple(axis for axis in dot_axis_names if axis not in output_axes)
        reduction_extent = extents[analysis.reduction_axis]
        tiling = APUG2DotTiling(
            math.prod(extent for _name, extent in dot_axes), reduction_extent
        )
        domain_axes = dot_axis_names + (analysis.reduction_axis,)
        plans.append(
            APUG2RankNContractionPlan(
                analysis=analysis,
                module=module,
                region_id=module.regions[index].id,
                dot_axes=dot_axes,
                batch_axes=batch_axes,
                output_axes=output_axes,
                reduction_axis=(analysis.reduction_axis, reduction_extent),
                lhs=_access_map(analysis.lhs, domain_axes),
                rhs=_access_map(analysis.rhs, domain_axes),
                initial_output=_access_map(analysis.accumulator, domain_axes),
                output=_access_map(analysis.output, domain_axes),
                tiling=tiling,
            )
        )
    return tuple(plans)


def plan_apu_g2_rank_n_contraction(module_or_text) -> APUG2RankNContractionPlan:
    """Plan exactly one rank-N contraction."""

    plans = plan_apu_g2_rank_n_contractions(module_or_text)
    if len(plans) != 1:
        raise UnsupportedAPUG2RankNContractionError(
            f"expected exactly one rank-N contraction, found {len(plans)}"
        )
    return plans[0]


__all__ = [
    "APUG2AffineAccessMap",
    "APUG2RankNContractionPlan",
    "UnsupportedAPUG2RankNContractionError",
    "analyze_apu_g2_rank_n_contractions",
    "plan_apu_g2_rank_n_contraction",
    "plan_apu_g2_rank_n_contractions",
]
