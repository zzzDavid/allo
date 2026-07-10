# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""MLIR-driven uint16 GEMV planning for Gemini-II VL64.

The contraction parser is shared with the existing APU vector path.  This
module adds only the APUg2 legality and physical-layout decisions for a dense
one-output-axis matrix-vector reduction; no function or benchmark names are
consulted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .apu_g2_layout import APUG2ReductionPlan
from .contraction_analysis import (
    ContractionAnalysis,
    IllegalContractionError,
    ValueAccess,
    analyze_contraction,
    analyze_contractions,
)


class UnsupportedAPUG2ContractionError(IllegalContractionError):
    """The MLIR is a contraction, but not a legal APUg2 uint16 GEMV."""


@dataclass(frozen=True)
class APUG2GemvPlan:
    """Logical roles and the immutable physical reduction layout for GEMV."""

    analysis: ContractionAnalysis = field(repr=False)
    matrix: ValueAccess
    vector: ValueAccess
    output: ValueAccess
    output_extent: int
    reduction_extent: int
    matrix_transposed: bool
    layout: APUG2ReductionPlan = field(repr=False)

    @property
    def log_block_size(self) -> int:
        return self.layout.log_block_size


@dataclass(frozen=True)
class APUG2AtaxPlan:
    """Two linked GEMVs whose intermediate remains resident on APUg2."""

    stage_m: APUG2GemvPlan
    stage_n: APUG2GemvPlan
    matrix: ValueAccess
    vector: ValueAccess
    intermediate: ValueAccess
    output: ValueAccess
    row_extent: int
    column_extent: int


def plan_apu_g2_gemv(module_or_analysis) -> APUG2GemvPlan:
    """Prove and plan one ordinary Allo ``uint16`` GEMV contraction."""

    analysis = (
        module_or_analysis
        if isinstance(module_or_analysis, ContractionAnalysis)
        else analyze_contraction(module_or_analysis)
    )
    if len(analysis.output_axes) != 1:
        raise UnsupportedAPUG2ContractionError(
            "APUg2 GEMV requires exactly one output axis; discovered "
            f"{analysis.output_axes}"
        )
    if (
        analysis.numeric_type != "ui16"
        or analysis.multiply_operation != "arith.muli"
        or analysis.combine_operation != "arith.addi"
    ):
        raise UnsupportedAPUG2ContractionError(
            "APUg2 GEMV requires a uint16 multiply-add contraction"
        )

    operands = (analysis.lhs, analysis.rhs)
    matrices = tuple(access for access in operands if len(access.shape) == 2)
    vectors = tuple(access for access in operands if len(access.shape) == 1)
    if len(matrices) != 1 or len(vectors) != 1:
        raise UnsupportedAPUG2ContractionError(
            "APUg2 GEMV requires one rank-2 matrix and one rank-1 vector"
        )
    matrix = matrices[0]
    vector = vectors[0]
    output_axis = analysis.output_axes[0]
    reduction_axis = analysis.reduction_axis
    if vector.indices != (reduction_axis,):
        raise UnsupportedAPUG2ContractionError(
            f"APUg2 GEMV vector must be indexed by {reduction_axis!r}"
        )
    if matrix.indices == (output_axis, reduction_axis):
        transposed = False
    elif matrix.indices == (reduction_axis, output_axis):
        transposed = True
    else:
        raise UnsupportedAPUG2ContractionError(
            "APUg2 GEMV matrix indices must be (output,reduction) or "
            "(reduction,output)"
        )
    if analysis.output.indices != (output_axis,):
        raise UnsupportedAPUG2ContractionError(
            "APUg2 GEMV output must be indexed by its single output axis"
        )

    extents = analysis.axis_extents
    output_extent = extents[output_axis]
    reduction_extent = extents[reduction_axis]
    layout = APUG2ReductionPlan(output_extent, reduction_extent, stream_extent=1)
    if layout.log_block_size > 8:
        raise UnsupportedAPUG2ContractionError(
            "APUg2 GEMV padded reduction extent must not exceed 256"
        )
    return APUG2GemvPlan(
        analysis=analysis,
        matrix=matrix,
        vector=vector,
        output=analysis.output,
        output_extent=output_extent,
        reduction_extent=reduction_extent,
        matrix_transposed=transposed,
        layout=layout,
    )


def plan_apu_g2_atax(module_or_analyses) -> APUG2AtaxPlan:
    """Prove ``tmp=A@x; y+=A.T@tmp`` from ordinary retained Allo MLIR."""

    if isinstance(module_or_analyses, tuple) and all(
        isinstance(item, ContractionAnalysis) for item in module_or_analyses
    ):
        analyses = module_or_analyses
    else:
        analyses = analyze_contractions(module_or_analyses)
    if len(analyses) != 2:
        raise UnsupportedAPUG2ContractionError(
            f"APUg2 ATAX requires exactly two contractions, found {len(analyses)}"
        )
    gemvs = tuple(plan_apu_g2_gemv(analysis) for analysis in analyses)
    linked = tuple(
        (first, second)
        for first in gemvs
        for second in gemvs
        if first is not second and first.output.value == second.vector.value
    )
    if len(linked) != 1:
        raise UnsupportedAPUG2ContractionError(
            "APUg2 ATAX requires the first GEMV output to be the second GEMV vector"
        )
    stage_m, stage_n = linked[0]
    if stage_m.matrix.value != stage_n.matrix.value:
        raise UnsupportedAPUG2ContractionError(
            "APUg2 ATAX requires both contractions to use the same matrix"
        )
    if stage_m.matrix_transposed or not stage_n.matrix_transposed:
        raise UnsupportedAPUG2ContractionError(
            "APUg2 ATAX requires direct A access followed by transposed A access"
        )
    if (
        stage_m.output_extent != stage_n.reduction_extent
        or stage_m.reduction_extent != stage_n.output_extent
        or stage_m.matrix.shape != stage_n.matrix.shape
    ):
        raise UnsupportedAPUG2ContractionError(
            "APUg2 ATAX contraction extents do not form A followed by A.T"
        )
    if stage_m.output_extent > 128 or stage_m.reduction_extent > 128:
        raise UnsupportedAPUG2ContractionError(
            "resident APUg2 ATAX currently supports matrix extents up to 128"
        )
    return APUG2AtaxPlan(
        stage_m=stage_m,
        stage_n=stage_n,
        matrix=stage_m.matrix,
        vector=stage_m.vector,
        intermediate=stage_m.output,
        output=stage_n.output,
        row_extent=stage_m.output_extent,
        column_extent=stage_m.reduction_extent,
    )


__all__ = [
    "APUG2AtaxPlan",
    "APUG2GemvPlan",
    "UnsupportedAPUG2ContractionError",
    "plan_apu_g2_atax",
    "plan_apu_g2_gemv",
]
