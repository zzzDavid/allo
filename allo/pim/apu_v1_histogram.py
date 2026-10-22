# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact paired-count GVML lowering for dense unsigned histograms.

The recognizer accepts the retained-MLIR form of a freshly zero-initialized
256-bin histogram over one complete 32K APUv1 lane frontier.  Bin identities
must come from the outer induction variable itself; no input-value or kernel
name is used to infer the histogram domain.

The physical plan loads the complete 32K lane frontier once, compares two
implicit bin values at a time, and uses the hardware's paired marker-count
primitive.  Unlike the performance-only Phoenix source, the bin identities
are proven from the retained program and all 256 logical counts are returned
to L4 exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np


VR_LANES = 32768
HISTOGRAM_BINS = 256


@dataclass(frozen=True)
class APUv1DenseHistogramAnalysis:
    """A proven canonical dense-u8 histogram loop nest."""

    function: str
    values_role: str
    counts_role: str
    items: int
    bins: int

    def manifest(self) -> dict:
        return {
            "kind": "dense_histogram_u16",
            "function": self.function,
            "values_role": self.values_role,
            "counts_role": self.counts_role,
            "items": self.items,
            "bins": self.bins,
            "bin_source": "outer_induction_variable",
            "input_layout": "one_u16_value_per_lane",
            "physical_plan": "one_32k_load_then_128_paired_immediate_counts",
            "count_pairing": "consecutive_implicit_bins",
        }


def _count(text: str, pattern: str) -> int:
    return len(re.findall(pattern, text, flags=re.MULTILINE))


# The recognizer deliberately rejects every structural mismatch independently.
# pylint: disable=too-many-return-statements,too-many-boolean-expressions
def analyze_apu_v1_dense_histogram(
    source_mlir: str,
    arguments,
    *,
    function: str,
) -> APUv1DenseHistogramAnalysis | None:
    """Recognize a complete structural histogram, failing closed on changes."""

    source = [arg for arg in arguments if arg.source == "argument"]
    results = [arg for arg in arguments if arg.source == "result"]
    if len(source) != 1 or len(results) != 1:
        return None
    values, counts = source[0], results[0]
    if (
        values.dtype != "ui16"
        or counts.dtype != "ui16"
        or values.shape != (VR_LANES,)
        or counts.shape != (HISTOGRAM_BINS,)
        or values.mode != "in"
        or counts.mode != "out"
    ):
        return None

    inventory = {
        r"\baffine\.for\b": 2,
        r"\barith\.cmpi\s+eq\b": 1,
        r"\bscf\.if\b": 1,
        r"\barith\.addi\b": 1,
        r"\baffine\.store\b": 1,
        r"\bmemref\.alloc\b": 1,
        r"\blinalg\.fill\b": 1,
        r"\barith\.index_cast\b": 1,
    }
    if any(
        _count(source_mlir, pattern) != wanted for pattern, wanted in inventory.items()
    ):
        return None
    if _count(source_mlir, r"\baffine\.load\b") != 2:
        return None
    if _count(source_mlir, r"\baffine\.for\s+%[-\w.$]+\s*=\s*0\s+to\s+256\b") != 1:
        return None
    if _count(source_mlir, r"\baffine\.for\s+%[-\w.$]+\s*=\s*0\s+to\s+32768\b") != 1:
        return None

    # Prove that equality is between the loaded value and the outer induction
    # variable after only width/index casts.  This is what makes bins 0..255 a
    # static semantic fact instead of a fixture-dependent assumption.
    outer_match = re.search(
        r"affine\.for\s+(%[-\w.$]+)\s*=\s*0\s+to\s+256\b",
        source_mlir,
    )
    inner_match = re.search(
        r"affine\.for\s+(%[-\w.$]+)\s*=\s*0\s+to\s+32768\b",
        source_mlir,
    )
    if outer_match is None or inner_match is None:
        return None
    outer, inner = outer_match.group(1), inner_match.group(1)
    value_load = re.search(
        rf"(?P<load>%[-\w.$]+)\s*=\s*affine\.load\s+%arg0\[{re.escape(inner)}\]"
        rf"[^\n]*from\s*=\s*\"{re.escape(values.name)}\"",
        source_mlir,
    )
    if value_load is None:
        return None
    widened = re.search(
        rf"(?P<wide>%[-\w.$]+)\s*=\s*arith\.extui\s+{re.escape(value_load.group('load'))}\b",
        source_mlir,
    )
    index = re.search(
        rf"(?P<index>%[-\w.$]+)\s*=\s*arith\.index_cast\s+{re.escape(outer)}\b",
        source_mlir,
    )
    if widened is None or index is None:
        return None
    comparison = re.search(
        rf"arith\.cmpi\s+eq,\s*{re.escape(widened.group('wide'))},\s*"
        rf"{re.escape(index.group('index'))}\b",
        source_mlir,
    )
    if comparison is None:
        return None
    if not re.search(
        rf"affine\.store\s+%[-\w.$]+\s*,\s*%[-\w.$]+\[{re.escape(outer)}\]"
        rf"[^\n]*to\s*=\s*\"{re.escape(counts.name)}\"",
        source_mlir,
    ):
        return None

    return APUv1DenseHistogramAnalysis(
        function=function,
        values_role=values.name,
        counts_role=counts.name,
        items=VR_LANES,
        bins=HISTOGRAM_BINS,
    )


@dataclass(frozen=True)
class APUv1DenseHistogramLowering:
    """Complete physical ABI and device source for paired marker counts."""

    analysis: APUv1DenseHistogramAnalysis

    @property
    def route(self) -> str:
        return "gvml_dense_histogram_pair_count"

    @property
    def scratch_bytes(self) -> int:
        return 8

    def pack_inputs(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        values = inputs[self.analysis.values_role]
        if values.shape != (self.analysis.items,) or values.dtype != np.uint16:
            raise ValueError("dense histogram values do not match analyzed ABI")
        return {**inputs, self.analysis.values_role: np.ascontiguousarray(values)}

    def operation_inventory(self) -> dict[str, int]:
        return {
            "dma_l4_to_l1_32k": 1,
            "gvml_load_16": 1,
            "gvml_eq_imm_16": HISTOGRAM_BINS,
            "gvml_2_fast_count_m_g32k": HISTOGRAM_BINS // 2,
            "scalar_l4_store_u16": HISTOGRAM_BINS,
        }

    def build_execution_graph(self, target, cost):
        """Cost the proven paired-count plan, not its scalar source loop."""

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
            "EQ_IMM_16",
            "gvml_eq_imm_16",
            count=inventory["gvml_eq_imm_16"],
        )
        graph.op(
            "FAST_COUNT_2M_G32K",
            "gvml_2_fast_count_m_g32k",
            count=inventory["gvml_2_fast_count_m_g32k"],
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
            "correctness_repairs": [
                "derive every immediate bin from the proven outer induction variable",
                "return all 256 bins without overwrite",
            ],
        }

    def emit_device_source(self) -> str:
        analysis = self.analysis
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
    HIST_ITEMS = {analysis.items},
    HIST_BINS = {analysis.bins},
}};

_Static_assert(HIST_ITEMS == 32768, "one complete APUv1 lane frontier required");
_Static_assert(HIST_BINS == 256, "paired plan requires 256 dense bins");

PROF_VAR(total);

static int dense_histogram(struct program_data *data)
{{
    const uint16_t *values_l4 = (const uint16_t *)gal_mem_handle_to_apu_ptr(
        data->mem_hndl_{analysis.values_role});
    uint16_t *counts_l4 = (uint16_t *)gal_mem_handle_to_apu_ptr(
        data->mem_hndl_{analysis.counts_role});
    enum gvml_vr16 values_vr = GVML_VR16_0;

    arc_counters_init();
    PROF_INIT(total);
    PROF_START(total);
    gal_set_l2dma_dma_mode(GAL_L2DMA_MODE_DIRECT);

    direct_dma_l4_to_l1_32k(GVML_VM_0, values_l4);
    gvml_load_16(values_vr, GVML_VM_0);

    for (uint16_t bin = 0; bin < HIST_BINS; bin += 2) {{
        unsigned int count0 = 0;
        unsigned int count1 = 0;
        gvml_eq_imm_16(GVML_MRK0, values_vr, bin);
        gvml_eq_imm_16(GVML_MRK1, values_vr, (uint16_t)(bin + 1));
        gvml_2_fast_count_m_g32k(
            &count0, &count1, GVML_MRK0, GVML_MRK1, GVML_VR16_1);
        counts_l4[bin] = (uint16_t)count0;
        counts_l4[bin + 1] = (uint16_t)count1;
    }}

    PROF_END(total);
    PROF_PRINT(total);
    return 0;
}}

GAL_TASK_ENTRY_POINT(apu_kernel_task, in, out)
{{
    struct program_cmd *cmd = (struct program_cmd *)in;
    gvml_init_once();
    return dense_histogram(&cmd->data);
}}
"""


__all__ = [
    "APUv1DenseHistogramAnalysis",
    "APUv1DenseHistogramLowering",
    "analyze_apu_v1_dense_histogram",
]
