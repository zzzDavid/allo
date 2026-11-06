# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compilation surface for LinearLayout-aware composed APUg2 dots."""

from __future__ import annotations

import math


from ..perf import BoundCostSpec
from .apu_g2_composed_contraction import (
    APUG2ComposedContractionProgram,
    APUG2DotEpilogue,
    APUG2DotEpilogueMode,
)
from .apu_g2_recipe import (
    APUG2Descriptor,
    APUG2DescriptorUse,
    APUG2Recipe,
    APUG2RecipeOperation,
    APUG2RecipeOpKind,
    APUG2VectorizationCertificate,
    build_apu_g2_priced_recipe_graph,
)


APUG2_COMPOSED_PHYSICAL_ELEMENTS = 4 * 65_536
APUG2_COMPOSED_L1_ROWS = {
    "lhs": 64,
    "rhs": 192,
    "auxiliary": 320,
    "out": 448,
    "scratch": 576,
}


def _descriptor(
    storage: str,
    scalar_type,
    *,
    start_row: int,
    segment: str | None = None,
) -> APUG2Descriptor:
    return APUG2Descriptor(
        storage,
        scalar_type.bits,
        num_vectors=4,
        value_type="int" if scalar_type.signed else "uint",
        start_row=start_row,
        segment=segment,
    )


def _use(role: str, descriptor: APUG2Descriptor) -> APUG2DescriptorUse:
    return APUG2DescriptorUse(role, descriptor)


def build_apu_g2_composed_recipe(
    program: APUG2ComposedContractionProgram,
) -> APUG2Recipe:
    """Record the exact full-carrier device call sequence."""

    if not isinstance(program, APUG2ComposedContractionProgram):
        raise TypeError("composed recipe requires APUG2ComposedContractionProgram")
    operations = []
    last = None
    layout = program.layout_manifest()
    common_metrics = {
        "layout_fingerprint": layout["fingerprint"],
        "program_fingerprint": program.structural_fingerprint,
        "logical_dots": program.dot_count,
        "reduction_extent": program.reduction_extent,
        "padded_output_extent": program.layout_plan.padded_output_extent,
        "padded_reduction_extent": program.layout_plan.padded_reduction_extent,
        "epilogue_mode": program.epilogue.mode.value,
    }

    def issue(opcode, kind, descriptors=(), *, updates=0, **attributes):
        nonlocal last
        identifier = f"call_{len(operations):04d}_{opcode.lower()}"
        operations.append(
            APUG2RecipeOperation(
                identifier,
                opcode,
                kind,
                dependencies=() if last is None else (last,),
                descriptors=tuple(descriptors),
                metrics={
                    **common_metrics,
                    "vector_lane_updates": updates,
                },
                attributes=attributes,
            )
        )
        last = identifier

    lhs_type, rhs_type = program.input_types
    product_type = program.product_type
    dot_type = program.dot_type
    auxiliary_type = program.auxiliary_type
    lhs_l1 = _descriptor("l1", lhs_type, start_row=APUG2_COMPOSED_L1_ROWS["lhs"])
    rhs_l1 = _descriptor("l1", rhs_type, start_row=APUG2_COMPOSED_L1_ROWS["rhs"])
    out_l1 = _descriptor(
        "l1", program.output_type, start_row=APUG2_COMPOSED_L1_ROWS["out"]
    )
    product_scratch = _descriptor(
        "l1",
        product_type,
        start_row=APUG2_COMPOSED_L1_ROWS["scratch"],
    )
    dot_scratch = _descriptor(
        "l1", dot_type, start_row=APUG2_COMPOSED_L1_ROWS["scratch"]
    )
    lhs_mmb = _descriptor("mmb", lhs_type, start_row=0, segment="seg0")
    rhs_mmb = _descriptor("mmb", rhs_type, start_row=lhs_type.bits, segment="seg0")
    product_mmb = _descriptor("mmb", product_type, start_row=24, segment="seg1")
    sum_source = _descriptor("mmb", product_type, start_row=0, segment="seg0")
    dot_result = _descriptor("mmb", dot_type, start_row=24, segment="seg1")
    dot_source = _descriptor("mmb", dot_type, start_row=0, segment="seg0")

    issue(
        "COPY_L1_VECTORS_TO_MMB",
        APUG2RecipeOpKind.TRANSFER,
        (_use("src", lhs_l1), _use("dst", lhs_mmb)),
    )
    issue(
        "COPY_L1_VECTORS_TO_MMB",
        APUG2RecipeOpKind.TRANSFER,
        (_use("src", rhs_l1), _use("dst", rhs_mmb)),
    )
    issue(
        "MUL_TYPED",
        APUG2RecipeOpKind.VL64,
        (
            _use("lhs", lhs_mmb),
            _use("rhs", rhs_mmb),
            _use("dst", product_mmb),
        ),
    )
    issue(
        "COPY_MMB_TO_L1_VECTORS",
        APUG2RecipeOpKind.TRANSFER,
        (_use("src", product_mmb), _use("dst", product_scratch)),
    )
    issue(
        "COPY_L1_VECTORS_TO_MMB",
        APUG2RecipeOpKind.TRANSFER,
        (_use("src", product_scratch), _use("dst", sum_source)),
    )
    issue(
        "GROUP_REDUCE_ADD_TYPED",
        APUG2RecipeOpKind.VL64,
        (_use("src", sum_source), _use("dst", dot_result)),
        log_reduction=int(math.log2(program.reduction_extent)),
    )

    mode = program.epilogue.mode
    if mode is APUG2DotEpilogueMode.IDENTITY:
        issue(
            "COPY_MMB_TO_L1_VECTORS",
            APUG2RecipeOpKind.TRANSFER,
            (_use("src", dot_result), _use("dst", out_l1)),
            updates=APUG2_COMPOSED_PHYSICAL_ELEMENTS,
        )
    else:
        auxiliary_l1 = _descriptor(
            "l1",
            auxiliary_type,
            start_row=APUG2_COMPOSED_L1_ROWS["auxiliary"],
        )
        issue(
            "COPY_MMB_TO_L1_VECTORS",
            APUG2RecipeOpKind.TRANSFER,
            (_use("src", dot_result), _use("dst", dot_scratch)),
        )
        issue(
            "COPY_L1_VECTORS_TO_MMB",
            APUG2RecipeOpKind.TRANSFER,
            (_use("src", dot_scratch), _use("dst", dot_source)),
        )
        if mode is APUG2DotEpilogueMode.PAIR_AFFINE:
            scaled_type = type(dot_type)(
                dot_type.bits + auxiliary_type.bits,
                dot_type.signed,
            )
            auxiliary_source = _descriptor(
                "mmb",
                auxiliary_type,
                start_row=dot_type.bits,
                segment="seg0",
            )
            scaled_result = _descriptor(
                "mmb", scaled_type, start_row=24, segment="seg1"
            )
            scaled_scratch = _descriptor(
                "l1",
                scaled_type,
                start_row=APUG2_COMPOSED_L1_ROWS["scratch"],
            )
            scaled_source = _descriptor("mmb", scaled_type, start_row=0, segment="seg0")
            odd_to_even = _descriptor("mmb", scaled_type, start_row=24, segment="seg1")
            pair_result = _descriptor(
                "mmb", program.output_type, start_row=24, segment="seg1"
            )
            issue(
                "COPY_L1_VECTORS_TO_MMB",
                APUG2RecipeOpKind.TRANSFER,
                (_use("src", auxiliary_l1), _use("dst", auxiliary_source)),
            )
            issue(
                "MUL_TYPED",
                APUG2RecipeOpKind.VL64,
                (
                    _use("lhs", dot_source),
                    _use("rhs", auxiliary_source),
                    _use("dst", scaled_result),
                ),
            )
            issue(
                "COPY_MMB_TO_L1_VECTORS",
                APUG2RecipeOpKind.TRANSFER,
                (_use("src", scaled_result), _use("dst", scaled_scratch)),
            )
            issue(
                "COPY_L1_VECTORS_TO_MMB",
                APUG2RecipeOpKind.TRANSFER,
                (_use("src", scaled_scratch), _use("dst", scaled_source)),
            )
            issue(
                "COPY_ODD_TO_EVEN_VECTORS",
                APUG2RecipeOpKind.VL64,
                (_use("src", scaled_source), _use("dst", odd_to_even)),
            )
            issue(
                "ADD_TYPED",
                APUG2RecipeOpKind.VL64,
                (
                    _use("lhs", scaled_source),
                    _use("rhs", odd_to_even),
                    _use("dst", pair_result),
                ),
            )
            epilogue_result = pair_result
        else:
            accumulator_mmb = _descriptor(
                "mmb", auxiliary_type, start_row=24, segment="seg1"
            )
            accumulated = _descriptor(
                "mmb", program.output_type, start_row=24, segment="seg1"
            )
            issue(
                "COPY_L1_VECTORS_TO_MMB",
                APUG2RecipeOpKind.TRANSFER,
                (_use("src", auxiliary_l1), _use("dst", accumulator_mmb)),
            )
            issue(
                "ADD_TYPED",
                APUG2RecipeOpKind.VL64,
                (
                    _use("lhs", dot_source),
                    _use("rhs", accumulator_mmb),
                    _use("dst", accumulated),
                ),
            )
            epilogue_result = accumulated
        issue(
            "COPY_MMB_TO_L1_VECTORS",
            APUG2RecipeOpKind.TRANSFER,
            (_use("src", epilogue_result), _use("dst", out_l1)),
            updates=APUG2_COMPOSED_PHYSICAL_ELEMENTS,
        )

    issue("SEU_BARRIER", APUG2RecipeOpKind.BARRIER)
    vector_updates = sum(
        operation.metrics["vector_lane_updates"] for operation in operations
    )
    return APUG2Recipe(
        program.name,
        tuple(operations),
        APUG2VectorizationCertificate(
            vector_lane_updates=vector_updates,
            scalar_control_ops=0,
            scalar_tensor_updates=0,
        ),
        metadata={
            "typed": True,
            "composed": True,
            "program": program.manifest(),
            "program_fingerprint": program.structural_fingerprint,
            "layout": layout,
            "layout_fingerprint": layout["fingerprint"],
            "canonical_boundary": "logical_dot_inputs_to_logical_output",
        },
    )


def build_apu_g2_composed_execution_graph(program, target, cost):
    if not isinstance(program, APUG2ComposedContractionProgram):
        raise TypeError("composed graph requires APUG2ComposedContractionProgram")
    if getattr(target, "name", None) != "apu_v2":
        raise ValueError("composed contractions require the apu_v2 target")
    if not isinstance(cost, BoundCostSpec) or cost.target is not target:
        raise TypeError("composed contractions require a target-bound cost")
    recipe = build_apu_g2_composed_recipe(program)
    graph = build_apu_g2_priced_recipe_graph(recipe, target, cost)
    graph.metadata.update(
        {
            "cost": cost.spec.name,
            "cost_fingerprint": cost.fingerprint,
            "recipe": recipe.manifest(),
            "recipe_fingerprint": recipe.structural_fingerprint,
            "program_fingerprint": program.structural_fingerprint,
            "layout_fingerprint": program.layout_manifest()["fingerprint"],
            "execution": "composed_direct_vl64",
        }
    )
    return graph


__all__ = [
    "APUG2_COMPOSED_L1_ROWS",
    "APUG2_COMPOSED_PHYSICAL_ELEMENTS",
    "APUG2ComposedContractionProgram",
    "APUG2DotEpilogue",
    "APUG2DotEpilogueMode",
    "build_apu_g2_composed_execution_graph",
    "build_apu_g2_composed_recipe",
]
