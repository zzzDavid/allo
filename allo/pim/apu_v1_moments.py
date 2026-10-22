# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused GVML lowering for five first/second bivariate moments.

The structural route recognizes a freshly initialized result with columns
``sum(x), sum(y), sum(x*x), sum(y*y), sum(x*y)``.  Independent datasets are
mapped to power-of-two lane groups whose combined size is one 32K APUv1
frontier.  Both inputs are loaded once, the three products are formed in
parallel VRs, and all five quantities use native grouped reduction.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np


VR_LANES = 32768
MOMENT_COUNT = 5


def _power2_enum(extent: int) -> str:
    if extent <= 0 or extent & (extent - 1):
        raise ValueError("APUv1 grouped moments require a power-of-two group")
    labels = {
        1: "GVML_P2_1",
        2: "GVML_P2_2",
        4: "GVML_P2_4",
        8: "GVML_P2_8",
        16: "GVML_P2_16",
        32: "GVML_P2_32",
        64: "GVML_P2_64",
        128: "GVML_P2_128",
        256: "GVML_P2_256",
        512: "GVML_P2_512",
        1024: "GVML_P2_1K",
        2048: "GVML_P2_2K",
        4096: "GVML_P2_4K",
        8192: "GVML_P2_8K",
        16384: "GVML_P2_16K",
        32768: "GVML_P2_32K",
    }
    try:
        return labels[extent]
    except KeyError as error:
        raise ValueError(f"unsupported APUv1 moment group size {extent}") from error


@dataclass(frozen=True)
class APUv1BivariateMomentsAnalysis:
    """A proven five-column bivariate-moment reduction."""

    function: str
    x_role: str
    y_role: str
    stats_role: str
    batches: int
    points_per_batch: int

    def manifest(self) -> dict:
        return {
            "kind": "bivariate_moments_u16",
            "function": self.function,
            "x_role": self.x_role,
            "y_role": self.y_role,
            "stats_role": self.stats_role,
            "batches": self.batches,
            "points_per_batch": self.points_per_batch,
            "stat_order": ["sum_x", "sum_y", "sum_xx", "sum_yy", "sum_xy"],
            "input_layout": "batch_major_contiguous_lane_groups",
            "arithmetic": "uint16_ring",
        }


def _opcode_count(source: str, opcode: str) -> int:
    return len(re.findall(rf"\b{re.escape(opcode)}\b", source))


def _load_counts(source: str, role: str) -> int:
    return len(re.findall(rf"\bfrom\s*=\s*\"{re.escape(role)}\"", source))


# The recognizer deliberately rejects every structural mismatch independently.
# pylint: disable=too-many-return-statements,too-many-boolean-expressions
def analyze_apu_v1_bivariate_moments(
    source_mlir: str,
    arguments,
    *,
    function: str,
) -> APUv1BivariateMomentsAnalysis | None:
    """Recognize the exact five-moment source skeleton, independent of names."""

    source = [arg for arg in arguments if arg.source == "argument"]
    results = [arg for arg in arguments if arg.source == "result"]
    if len(source) != 2 or len(results) != 1:
        return None
    if any(arg.dtype != "ui16" for arg in source + results):
        return None
    x, y = source
    stats = results[0]
    if (
        len(x.shape) != 2
        or y.shape != x.shape
        or stats.shape != (x.shape[0], MOMENT_COUNT)
        or x.mode != "in"
        or y.mode != "in"
        or stats.mode != "out"
    ):
        return None
    batches, points = x.shape
    if batches * points != VR_LANES:
        return None
    try:
        _power2_enum(points)
    except ValueError:
        return None

    inventory = {
        "affine.for": 2,
        "arith.muli": 3,
        "arith.addi": 5,
        "affine.load": 13,
        "affine.store": 5,
        "memref.alloc": 1,
        "linalg.fill": 1,
    }
    if any(_opcode_count(source_mlir, op) != count for op, count in inventory.items()):
        return None
    if _load_counts(source_mlir, x.name) != 4:
        return None
    if _load_counts(source_mlir, y.name) != 4:
        return None
    if _load_counts(source_mlir, stats.name) != 5:
        return None
    if len(re.findall(rf"\bto\s*=\s*\"{re.escape(stats.name)}\"", source_mlir)) != 5:
        return None
    if (
        len(
            re.findall(
                rf"\baffine\.for\s+%[-\w.$]+\s*=\s*0\s+to\s+{batches}\b", source_mlir
            )
        )
        != 1
    ):
        return None
    if (
        len(
            re.findall(
                rf"\baffine\.for\s+%[-\w.$]+\s*=\s*0\s+to\s+{points}\b", source_mlir
            )
        )
        != 1
    ):
        return None
    for stat in range(MOMENT_COUNT):
        if (
            len(
                re.findall(
                    rf"affine\.store\s+%[-\w.$]+\s*,\s*%[-\w.$]+\["
                    rf"%[-\w.$]+,\s*{stat}\][^\n]*to\s*=\s*\"{re.escape(stats.name)}\"",
                    source_mlir,
                )
            )
            != 1
        ):
            return None

    # The canonical Allo emission orders the three multiplications as x*x,
    # y*y, x*y.  Prove their operand origins across only the widening aliases
    # that appear between loads and multiplication.
    definitions: dict[str, tuple] = {}
    for line in source_mlir.splitlines():
        load = re.search(
            r"(?P<dst>%[-\w.$]+)\s*=\s*affine\.load\s+%[-\w.$]+\[[^]]+]"
            r"[^\n]*from\s*=\s*\"(?P<role>[^\"]+)\"",
            line,
        )
        if load is not None:
            definitions[load.group("dst")] = ("load", load.group("role"))
            continue
        alias = re.search(
            r"(?P<dst>%[-\w.$]+)\s*=\s*arith\.(?:extui|extsi|trunci)\s+"
            r"(?P<src>%[-\w.$]+)",
            line,
        )
        if alias is not None:
            definitions[alias.group("dst")] = ("alias", alias.group("src"))
            continue
        multiply = re.search(
            r"(?P<dst>%[-\w.$]+)\s*=\s*arith\.muli\s+"
            r"(?P<lhs>%[-\w.$]+)\s*,\s*(?P<rhs>%[-\w.$]+)",
            line,
        )
        if multiply is not None:
            definitions[multiply.group("dst")] = (
                "multiply",
                multiply.group("lhs"),
                multiply.group("rhs"),
            )

    def load_origin(value: str) -> str | None:
        seen = set()
        while value not in seen:
            seen.add(value)
            definition = definitions.get(value)
            if definition is None:
                return None
            if definition[0] == "load":
                return definition[1]
            if definition[0] != "alias":
                return None
            value = definition[1]
        return None

    products = []
    for definition in definitions.values():
        if definition[0] != "multiply":
            continue
        products.append((load_origin(definition[1]), load_origin(definition[2])))
    expected_products = [(x.name, x.name), (y.name, y.name), (x.name, y.name)]
    if products != expected_products:
        return None

    return APUv1BivariateMomentsAnalysis(
        function=function,
        x_role=x.name,
        y_role=y.name,
        stats_role=stats.name,
        batches=batches,
        points_per_batch=points,
    )


@dataclass(frozen=True)
class APUv1BivariateMomentsLowering:
    """Complete physical ABI and source for fused five-moment reduction."""

    analysis: APUv1BivariateMomentsAnalysis

    @property
    def route(self) -> str:
        return "gvml_bivariate_moments_fused_group_reduce"

    @property
    def scratch_bytes(self) -> int:
        return 8

    def pack_inputs(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        analysis = self.analysis
        expected = (analysis.batches, analysis.points_per_batch)
        x = inputs[analysis.x_role]
        y = inputs[analysis.y_role]
        if x.shape != expected or y.shape != expected:
            raise ValueError("bivariate-moment inputs do not match analyzed shape")
        return {
            **inputs,
            analysis.x_role: np.ascontiguousarray(x),
            analysis.y_role: np.ascontiguousarray(y),
        }

    def operation_inventory(self) -> dict[str, int]:
        return {
            "dma_l4_to_l1_32k": 2,
            "gvml_load_16": 2,
            "gvml_mul_u16": 3,
            "gvml_add_subgrps_u16_grp": 5,
            "gvml_get_entry_16": self.analysis.batches * MOMENT_COUNT,
            "scalar_l4_store_u16": self.analysis.batches * MOMENT_COUNT,
        }

    def build_execution_graph(self, target, cost):
        """Cost the fused five-reduction physical plan structurally."""

        from .apu_v1_native_cost import APUv1NativeCostGraphBuilder

        inventory = self.operation_inventory()
        graph = APUv1NativeCostGraphBuilder(self.route, inventory, target, cost)
        graph.move(
            "DMA_L4_TO_L1_32K",
            "dma_l4_to_l1_32k",
            count=inventory["dma_l4_to_l1_32k"],
        )
        graph.move(
            "LOAD_L1_TO_VR16",
            "gvml_load_16",
            count=inventory["gvml_load_16"],
        )
        graph.op(
            "MUL_U16",
            "gvml_mul_u16",
            count=inventory["gvml_mul_u16"],
        )
        graph.op(
            "GROUP_REDUCE_ADD_U16",
            "gvml_add_subgrps_u16_grp",
            count=inventory["gvml_add_subgrps_u16_grp"],
            group_size=self.analysis.points_per_batch,
        )
        graph.op(
            "GET_ENTRY_16",
            "gvml_get_entry_16",
            count=inventory["gvml_get_entry_16"],
        )
        graph.move(
            "ARC_STORE_L4_U16",
            "scalar_l4_store_u16",
            count=inventory["scalar_l4_store_u16"],
        )
        return graph.finish()

    def manifest(self) -> dict:
        return {
            "route": self.route,
            "analysis": self.analysis.manifest(),
            "operation_inventory": self.operation_inventory(),
            "physical_input_transforms": {},
            "output_transform": "identity",
        }

    def emit_device_source(self) -> str:
        analysis = self.analysis
        group_enum = _power2_enum(analysis.points_per_batch)
        return f"""#include <stdint.h>

#include <gsi/libsys/assert.h>
#include <gsi/libsys.h>
#include <gsi/libgal.h>
#include <gsi/gal-fast-funcs.h>
#include <gsi/libgvml_debug.h>
#include <gsi/libgvml_element_wise.h>
#include <gsi/libgvml_iv.h>
#include <gsi/libgvml_memory.h>

#include "gsi_dma.h"
#include "struct.h"
#include <gsi_device_profiling.h>

enum {{
    MOMENT_BATCHES = {analysis.batches},
    MOMENT_POINTS = {analysis.points_per_batch},
    MOMENT_STATS = 5,
}};

_Static_assert(MOMENT_BATCHES * MOMENT_POINTS == 32768,
               "moment groups must fill one APUv1 frontier");

PROF_VAR(total);

static int bivariate_moments(struct program_data *data)
{{
    const uint16_t *x_l4 = (const uint16_t *)gal_mem_handle_to_apu_ptr(
        data->mem_hndl_{analysis.x_role});
    const uint16_t *y_l4 = (const uint16_t *)gal_mem_handle_to_apu_ptr(
        data->mem_hndl_{analysis.y_role});
    uint16_t *stats_l4 = (uint16_t *)gal_mem_handle_to_apu_ptr(
        data->mem_hndl_{analysis.stats_role});

    enum gvml_vr16 x_vr = GVML_VR16_0;
    enum gvml_vr16 y_vr = GVML_VR16_1;
    enum gvml_vr16 xx_vr = GVML_VR16_2;
    enum gvml_vr16 yy_vr = GVML_VR16_3;
    enum gvml_vr16 xy_vr = GVML_VR16_4;
    enum gvml_vr16 reduce_tmp_vr = GVML_VR16_5;

    arc_counters_init();
    PROF_INIT(total);
    PROF_START(total);
    gal_set_l2dma_dma_mode(GAL_L2DMA_MODE_DIRECT);

    direct_dma_l4_to_l1_32k(GVML_VM_0, x_l4);
    direct_dma_l4_to_l1_32k(GVML_VM_1, y_l4);
    gvml_load_16(x_vr, GVML_VM_0);
    gvml_load_16(y_vr, GVML_VM_1);
    gvml_mul_u16(xx_vr, x_vr, x_vr);
    gvml_mul_u16(yy_vr, y_vr, y_vr);
    gvml_mul_u16(xy_vr, x_vr, y_vr);

    gvml_add_subgrps_u16_grp(
        x_vr, x_vr, {group_enum}, GVML_P2_1, 0, GVML_VM_2, reduce_tmp_vr);
    gvml_add_subgrps_u16_grp(
        y_vr, y_vr, {group_enum}, GVML_P2_1, 0, GVML_VM_2, reduce_tmp_vr);
    gvml_add_subgrps_u16_grp(
        xx_vr, xx_vr, {group_enum}, GVML_P2_1, 0, GVML_VM_2, reduce_tmp_vr);
    gvml_add_subgrps_u16_grp(
        yy_vr, yy_vr, {group_enum}, GVML_P2_1, 0, GVML_VM_2, reduce_tmp_vr);
    gvml_add_subgrps_u16_grp(
        xy_vr, xy_vr, {group_enum}, GVML_P2_1, 0, GVML_VM_2, reduce_tmp_vr);

    for (uint16_t batch = 0; batch < MOMENT_BATCHES; ++batch) {{
        const uint32_t lane = (uint32_t)batch * MOMENT_POINTS;
        stats_l4[batch * MOMENT_STATS + 0] = gvml_get_entry_16(x_vr, lane);
        stats_l4[batch * MOMENT_STATS + 1] = gvml_get_entry_16(y_vr, lane);
        stats_l4[batch * MOMENT_STATS + 2] = gvml_get_entry_16(xx_vr, lane);
        stats_l4[batch * MOMENT_STATS + 3] = gvml_get_entry_16(yy_vr, lane);
        stats_l4[batch * MOMENT_STATS + 4] = gvml_get_entry_16(xy_vr, lane);
    }}

    PROF_END(total);
    PROF_PRINT(total);
    return 0;
}}

GAL_TASK_ENTRY_POINT(apu_kernel_task, in, out)
{{
    struct program_cmd *cmd = (struct program_cmd *)in;
    gvml_init_once();
    return bivariate_moments(&cmd->data);
}}
"""


__all__ = [
    "APUv1BivariateMomentsAnalysis",
    "APUv1BivariateMomentsLowering",
    "analyze_apu_v1_bivariate_moments",
]
