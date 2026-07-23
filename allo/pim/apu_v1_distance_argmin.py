# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact GVML lowering for batched squared-distance argmin programs.

This is a structural compiler route, not a benchmark implementation.  It
recognizes the canonical retained-MLIR form of

``labels[p] = argmin_c sum_d (points[p, d] - centers[c, d])**2``

with strict first-index tie breaking.  One 32K point frontier maps directly to
the APUv1 lanes.  The physical ABI transposes points to dimension-major order
so each dimension is one VMR.

The selected plan applies the ring identity

``(p - c)^2 == p^2 + c^2 - 2*p*c (mod 2**16)``.

It computes the point norm once and streams one candidate dot product through
low-numbered VRs.  This plan has a larger primitive count than keeping every
dot resident, but real APUv1 profiles show lower end-to-end CRUN because it
avoids the resident plan's dependency/register-scheduling penalty.  Signed i16
ordering is preserved exactly by flipping the sign bit before the unsigned
GVML comparison.  Consequently the lowering does not depend on a
benchmark-specific value bound; bounds are only needed when a caller wants to
interpret the wrapped i16 program as mathematical squared L2.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

import numpy as np


VR_LANES = 32768
MAX_CANDIDATES = 65535


@dataclass(frozen=True)
class APUv1SquaredL2ArgminAnalysis:
    """A proven canonical squared-L2/strict-argmin loop nest."""

    function: str
    points_role: str
    centers_role: str
    labels_role: str
    points: int
    dimensions: int
    candidates: int
    comparison: str
    unsigned_order_proven: bool = False
    maximum_squared_distance: int | None = None

    @property
    def center_values(self) -> int:
        return self.candidates * self.dimensions

    @property
    def center_image_values(self) -> int:
        # L4->L3 uses complete 512-byte transactions.
        return math.ceil(self.center_values / 256) * 256

    @property
    def center_image_bytes(self) -> int:
        return self.center_image_values * 2

    def manifest(self) -> dict:
        return {
            "kind": "squared_l2_argmin_u16",
            "function": self.function,
            "points_role": self.points_role,
            "centers_role": self.centers_role,
            "labels_role": self.labels_role,
            "points": self.points,
            "dimensions": self.dimensions,
            "candidates": self.candidates,
            "comparison": self.comparison,
            "unsigned_order_proven": self.unsigned_order_proven,
            "maximum_squared_distance": self.maximum_squared_distance,
            "point_layout": "dimension_major_32k_vmr",
            "center_image_values": self.center_image_values,
            "candidate_residency": "one_streamed_dot_product_in_vr",
            "algebra": "sum(p*p)+sum(c*c)-2*sum(p*c) modulo 2^16",
        }


def _count_opcode(text: str, opcode: str) -> int:
    return len(re.findall(rf"\b{re.escape(opcode)}\b", text))


def _loop_count(text: str, lower: int, upper: int) -> int:
    return len(
        re.findall(
            rf"\baffine\.for\s+%[-\w.$]+\s*=\s*{lower}\s+to\s+{upper}\b",
            text,
        )
    )


# The recognizer deliberately rejects every structural mismatch independently.
# pylint: disable=too-many-return-statements
def analyze_apu_v1_squared_l2_argmin(
    source_mlir: str,
    arguments,
    *,
    function: str,
    argument_bounds=None,
) -> APUv1SquaredL2ArgminAnalysis | None:
    """Recognize the exact canonical Allo loop/def-use skeleton.

    Shape matching alone is intentionally insufficient.  The accepted module
    must contain the two canonical squared-distance reductions (candidate zero
    and the remaining candidates), one strict conditional update, and no
    additional arithmetic/control-flow operations of those classes.
    """

    source_arguments = [arg for arg in arguments if arg.source == "argument"]
    if len(source_arguments) != 3:
        return None
    if any(arg.dtype != "ui16" for arg in source_arguments):
        return None

    rank1 = [arg for arg in source_arguments if len(arg.shape) == 1]
    rank2 = [arg for arg in source_arguments if len(arg.shape) == 2]
    if len(rank1) != 1 or len(rank2) != 2:
        return None
    labels = rank1[0]
    points = labels.shape[0]
    if points != VR_LANES or labels.mode not in {"out", "both"}:
        return None

    point_arrays = [arg for arg in rank2 if arg.shape[0] == points]
    if len(point_arrays) != 1:
        return None
    point_array = point_arrays[0]
    dimensions = point_array.shape[1]
    center_arrays = [
        arg for arg in rank2 if arg is not point_array and arg.shape[1] == dimensions
    ]
    if len(center_arrays) != 1:
        return None
    centers = center_arrays[0]
    candidates = centers.shape[0]
    if not (2 <= candidates <= MAX_CANDIDATES):
        return None
    if not (1 <= dimensions <= 48):
        return None
    if math.ceil(candidates * dimensions * 2 / 512) * 512 > (1 << 20):
        return None

    # Canonical Allo emits one outer loop, two statically represented distance
    # reductions, and one candidate loop.  Requiring the complete inventory
    # makes this recognizer fail closed if the source computation changes.
    required_counts = {
        "affine.for": 4,
        "arith.subi": 2,
        "arith.muli": 2,
        "arith.addi": 2,
        "arith.trunci": 7,
        "scf.if": 1,
    }
    if any(
        _count_opcode(source_mlir, op) != count for op, count in required_counts.items()
    ):
        return None
    comparisons = re.findall(r"\barith\.cmpi\s+(slt|ult)\b", source_mlir)
    if comparisons not in (["slt"], ["ult"]):
        return None
    if _loop_count(source_mlir, 0, points) != 1:
        return None
    if _loop_count(source_mlir, 0, dimensions) != 2:
        return None
    if _loop_count(source_mlir, 1, candidates) != 1:
        return None

    # Tie the operation skeleton to the three formal SSA values.  The
    # canonical top-level ABI numbers formals in argument order.
    positions = {id(arg): index for index, arg in enumerate(source_arguments)}
    point_ssa = rf"%arg{positions[id(point_array)]}"
    center_ssa = rf"%arg{positions[id(centers)]}"
    label_ssa = rf"%arg{positions[id(labels)]}"
    if len(re.findall(rf"\baffine\.load\s+{point_ssa}\[", source_mlir)) != 2:
        return None
    if len(re.findall(rf"\baffine\.load\s+{center_ssa}\[", source_mlir)) != 2:
        return None
    if (
        len(
            re.findall(rf"\baffine\.store\s+%[-\w.$]+\s*,\s*{label_ssa}\[", source_mlir)
        )
        != 1
    ):
        return None

    maximum_squared_distance = None
    bounds = dict(argument_bounds or ())
    point_bounds = bounds.get(point_array.name)
    center_bounds = bounds.get(centers.name)
    if point_bounds is not None and center_bounds is not None:
        if not (
            0 <= point_bounds[0] <= point_bounds[1] <= np.iinfo(np.uint16).max
            and 0 <= center_bounds[0] <= center_bounds[1] <= np.iinfo(np.uint16).max
        ):
            return None
        maximum_delta = max(
            abs(point_bounds[0] - center_bounds[1]),
            abs(point_bounds[1] - center_bounds[0]),
        )
        maximum_squared_distance = dimensions * maximum_delta * maximum_delta

    return APUv1SquaredL2ArgminAnalysis(
        function=function,
        points_role=point_array.name,
        centers_role=centers.name,
        labels_role=labels.name,
        points=points,
        dimensions=dimensions,
        candidates=candidates,
        comparison=comparisons[0],
        unsigned_order_proven=(
            comparisons[0] == "ult"
            or (
                maximum_squared_distance is not None
                and maximum_squared_distance <= np.iinfo(np.int16).max
            )
        ),
        maximum_squared_distance=maximum_squared_distance,
    )


@dataclass(frozen=True)
class APUv1SquaredL2ArgminLowering:
    """Complete physical ABI and source for the squared-L2 argmin route."""

    analysis: APUv1SquaredL2ArgminAnalysis

    @property
    def route(self) -> str:
        return "gvml_squared_l2_argmin_center_streaming"

    @property
    def scratch_bytes(self) -> int:
        return 8

    def operation_inventory(self) -> dict[str, int]:
        d = self.analysis.dimensions
        k = self.analysis.candidates
        # Masked copies and comparison operations are reported separately even
        # though the current general target cost vocabulary does not yet carry
        # calibrated primitives for them.
        return {
            "dma_l4_to_l3_512b": self.analysis.center_image_bytes // 512,
            "dma_l4_to_l1_32k": d,
            "gvml_load_16": d * (k + 1),
            "gvml_reset_16": k + 2,
            "gvml_cpy_imm_16": d * k
            + k
            + (
                k
                if self.analysis.comparison == "slt"
                and not self.analysis.unsigned_order_proven
                else 0
            ),
            "gvml_mul_u16": d * (k + 1),
            "gvml_add_u16": d * (k + 1) + k,
            "gvml_sl_imm_16": k,
            "gvml_sub_u16": k,
            "gvml_xor_16": (
                k
                if self.analysis.comparison == "slt"
                and not self.analysis.unsigned_order_proven
                else 0
            ),
            "gvml_lt_u16": k - 1,
            "gvml_cpy_16_msk_mrk": k - 1,
            "gvml_cpy_imm_16_mrk": k - 1,
            "dma_l1_to_l4_32k": 1,
        }

    def pack_inputs(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Apply the declared physical ABI without changing logical arrays."""

        analysis = self.analysis
        packed = dict(inputs)
        points = inputs[analysis.points_role]
        if points.shape != (analysis.points, analysis.dimensions):
            raise ValueError("squared-L2 argmin points do not match analyzed shape")
        packed[analysis.points_role] = np.ascontiguousarray(points.T)

        centers = inputs[analysis.centers_role]
        if centers.shape != (analysis.candidates, analysis.dimensions):
            raise ValueError("squared-L2 argmin centers do not match analyzed shape")
        center_image = np.zeros(analysis.center_image_values, dtype=np.uint16)
        center_image[: analysis.center_values] = centers.reshape(-1)
        packed[analysis.centers_role] = center_image
        return packed

    def manifest(self) -> dict:
        return {
            "route": self.route,
            "analysis": self.analysis.manifest(),
            "operation_inventory": self.operation_inventory(),
            "physical_input_transforms": {
                self.analysis.points_role: "transpose_point_dimension_to_dimension_point",
                self.analysis.centers_role: "row_major_then_zero_pad_to_512_bytes",
            },
            "output_transform": "identity",
        }

    def build_execution_graph(self, target, cost):
        """Build the calibrated primitive graph used for route reporting.

        Comparison and masked-copy calls do not yet have distinct structural
        target primitives.  They are conservatively represented by the
        calibrated 16-bit copy issue cost; DMA and arithmetic retain their
        exact target operations and measured counts.
        """

        from ..perf import CostEvent
        from ..perf.graph import ExecutionGraph

        inventory = self.operation_inventory()
        graph = ExecutionGraph(
            self.route,
            metadata={
                "target": "apu_v1",
                "route": self.route,
                "operation_inventory": inventory,
                "analytical": True,
            },
        )
        dependencies = ()
        event_index = 0

        def emit(handle, label, *, count=1, **metrics):
            nonlocal dependencies, event_index
            event = CostEvent.create(
                f"native-vector:{event_index}:{label}",
                handle,
                work_id=(0,),
                metrics={"count": int(count), **metrics},
                attributes={"route": self.route, "operation": label},
            )
            dependencies = tuple(cost.emit(graph, event, dependencies))
            event_index += 1

        emit(
            target.move("DMA_L4_TO_L3"),
            "DMA_L4_TO_L3",
            bytes=self.analysis.center_image_bytes,
        )
        for move_name, inventory_name in (
            ("DMA_L4_TO_L1_32K", "dma_l4_to_l1_32k"),
            ("LOAD_L1_TO_VR16", "gvml_load_16"),
        ):
            emit(
                target.move(move_name),
                move_name,
                count=inventory[inventory_name],
            )
        for opcode, inventory_name in (
            ("RESET_16", "gvml_reset_16"),
            ("CPY_IMM_16", "gvml_cpy_imm_16"),
            ("MUL_U16", "gvml_mul_u16"),
            ("ADD_U16", "gvml_add_u16"),
            ("SHL_IMM_16", "gvml_sl_imm_16"),
            ("SUB_U16", "gvml_sub_u16"),
            ("XOR_16", "gvml_xor_16"),
        ):
            count = inventory[inventory_name]
            if count:
                emit(target.op(opcode), opcode, count=count)
        unmodeled_vector_calls = (
            1
            + inventory["gvml_lt_u16"]
            + inventory["gvml_cpy_16_msk_mrk"]
            + inventory["gvml_cpy_imm_16_mrk"]
        )
        emit(
            target.op("CPY_IMM_16"),
            "comparison_and_masked_copy_issue_proxy",
            count=unmodeled_vector_calls,
        )
        emit(target.move("STORE_VR16_TO_L1"), "STORE_VR16_TO_L1")
        emit(target.move("DMA_L1_TO_L4_32K"), "DMA_L1_TO_L4_32K")
        return graph

    def emit_device_source(self) -> str:
        analysis = self.analysis
        k = analysis.candidates
        d = analysis.dimensions
        signed_transform = ""
        if analysis.comparison == "slt" and not analysis.unsigned_order_proven:
            signed_transform = """
        gvml_cpy_imm_16(center_vr, UINT16_C(0x8000));
        gvml_xor_16(distance_vr, distance_vr, center_vr);
"""

        return f"""#include <stdint.h>

#include <gsi/libsys/assert.h>
#include <gsi/libsys.h>
#include <gsi/libgal.h>
#include <gsi/gal-fast-funcs.h>
#include <gsi/libgvml_element_wise.h>
#include <gsi/libgvml_memory.h>

#include "gsi_dma.h"
#include "struct.h"
#include <gsi_device_profiling.h>

enum {{
    ARGMIN_POINTS = {analysis.points},
    ARGMIN_DIMS = {d},
    ARGMIN_CANDIDATES = {k},
    ARGMIN_CENTER_VALUES = {analysis.center_values},
    ARGMIN_CENTER_IMAGE_BYTES = {analysis.center_image_bytes},
}};

_Static_assert(ARGMIN_POINTS == 32768, "one point per APUv1 lane is required");
_Static_assert(ARGMIN_CANDIDATES <= 65535, "labels use uint16 candidate indices");

PROF_VAR(total);

static void copy_centers_l4_to_l3(uint16_t *dst, const uint16_t *src)
{{
    gal_fast_l2dma_async_memcpy_init(GAL_L2DMA_APC_ID_0);
    for (uint32_t offset = 0; offset < ARGMIN_CENTER_IMAGE_BYTES; offset += 512) {{
        gal_fast_l2dma_mem_to_mem_512(
            (uint8_t *)dst + offset,
            (uint8_t *)src + offset,
            GAL_L2DMA_APC_ID_0);
    }}
    gal_fast_l2dma_async_memcpy_end(GAL_L2DMA_APC_ID_0);
}}

static int squared_l2_argmin(struct program_data *data)
{{
    const uint16_t *points_l4 = (const uint16_t *)gal_mem_handle_to_apu_ptr(
        data->mem_hndl_{analysis.points_role});
    const uint16_t *centers_l4 = (const uint16_t *)gal_mem_handle_to_apu_ptr(
        data->mem_hndl_{analysis.centers_role});
    uint16_t *labels_l4 = (uint16_t *)gal_mem_handle_to_apu_ptr(
        data->mem_hndl_result_{analysis.labels_role});
    uint16_t *centers_l3 = (uint16_t *)gal_fast_malloc_cache_aligned(
        ARGMIN_CENTER_IMAGE_BYTES, true);
    if (GSI_IS_ERR_PTR_OR_NULL(centers_l3)) {{
        gsi_error("squared-L2 argmin: failed to allocate center image in L3\\n");
        return gsi_status(ENOMEM);
    }}

    enum gvml_vr16 center_vr = GVML_VR16_0;
    enum gvml_vr16 point_vr = GVML_VR16_1;
    enum gvml_vr16 product_vr = GVML_VR16_2;
    enum gvml_vr16 point_norm_vr = GVML_VR16_3;
    enum gvml_vr16 dot_vr = GVML_VR16_4;
    enum gvml_vr16 distance_vr = GVML_VR16_5;
    enum gvml_vr16 min_distance_vr = GVML_VR16_6;
    enum gvml_vr16 label_vr = GVML_VR16_7;

    arc_counters_init();
    PROF_INIT(total);
    PROF_START(total);

    gal_set_l2dma_dma_mode(GAL_L2DMA_MODE_DIRECT);
    copy_centers_l4_to_l3(centers_l3, centers_l4);
    for (uint16_t dim = 0; dim < ARGMIN_DIMS; ++dim) {{
        direct_dma_l4_to_l1_32k(
            (enum gvml_vm_reg)dim,
            points_l4 + (uint32_t)dim * ARGMIN_POINTS);
    }}

    gvml_reset_16(point_norm_vr);
    for (uint16_t dim = 0; dim < ARGMIN_DIMS; ++dim) {{
        gvml_load_16(point_vr, (enum gvml_vm_reg)dim);
        gvml_mul_u16(product_vr, point_vr, point_vr);
        gvml_add_u16(point_norm_vr, point_norm_vr, product_vr);
    }}

    gvml_reset_16(label_vr);
    for (uint16_t center = 0; center < ARGMIN_CANDIDATES; ++center) {{
        uint16_t center_norm = 0;
        gvml_reset_16(dot_vr);

        for (uint16_t dim = 0; dim < ARGMIN_DIMS; ++dim) {{
            const uint16_t center_coord =
                centers_l3[(uint32_t)center * ARGMIN_DIMS + dim];
            center_norm = (uint16_t)(
                center_norm + (uint16_t)(center_coord * center_coord));
            gvml_cpy_imm_16(center_vr, center_coord);
            gvml_load_16(point_vr, (enum gvml_vm_reg)dim);
            gvml_mul_u16(product_vr, point_vr, center_vr);
            gvml_add_u16(dot_vr, dot_vr, product_vr);
        }}

        /* Reconstruct the original i16 distance exactly in the uint16 ring. */
        gvml_sl_imm_16(dot_vr, dot_vr, 1);
        gvml_cpy_imm_16(center_vr, center_norm);
        gvml_add_u16(distance_vr, point_norm_vr, center_vr);
        gvml_sub_u16(distance_vr, distance_vr, dot_vr);
{signed_transform}
        if (center == 0) {{
            gvml_cpy_16(min_distance_vr, distance_vr);
        }} else {{
            gvml_lt_u16(GVML_MRK0, distance_vr, min_distance_vr);
            gvml_cpy_16_msk_mrk(
                min_distance_vr, distance_vr, UINT16_MAX, GVML_MRK0);
            gvml_cpy_imm_16_mrk(label_vr, center, GVML_MRK0);
        }}
    }}

    gvml_store_16(GVML_VM_0, label_vr);
    direct_dma_l1_to_l4_32k(labels_l4, GVML_VM_0);

    PROF_END(total);
    PROF_PRINT(total);
    return 0;
}}

GAL_TASK_ENTRY_POINT(apu_kernel_task, in, out)
{{
    struct program_cmd *cmd = (struct program_cmd *)in;
    gvml_init_once();
    return squared_l2_argmin(&cmd->data);
}}
"""


__all__ = [
    "APUv1SquaredL2ArgminAnalysis",
    "APUv1SquaredL2ArgminLowering",
    "MAX_CANDIDATES",
    "analyze_apu_v1_squared_l2_argmin",
]
