# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact paired-count GVML lowering for dense unsigned histograms.

Supported semantics (proved structurally from the retained MLIR; see
:mod:`apu_v1_native_vector`)::

    counts: uint16[B] = 0                 # whole-array zero initializer
    for b in range(B):                    # bin loop (either nesting order)
        for i in range(N):                # item loop
            if values[i] == b:            # equality, value vs bin IV
                counts[b] += 1            # increment exactly 1, mod 2**16

with ``values: uint16[N]`` the only source argument and ``counts`` the only
returned array.  The comparison may compare the 16-bit patterns directly or
after zero extension (sign extension is accepted only when every bin is below
``2**15``).  Function, argument, and loop names are irrelevant.

Rejected with :class:`APUv1NativeRecipeMismatch` naming the difference: any
other increment, a nonzero or missing initializer, any predicate other than
``==``, a comparison that is not ``values[item]`` against the bin induction
variable, extra stores/loads-with-effects, loop-carried scalars, non-uint16
dtypes, non-unit steps or offset bounds.

Shape domain: ``N`` equals the target's VR lane count (one complete 32K
frontier on Gemini-I); ``1 <= B <= 2**16`` (bins are 16-bit immediates).
Odd ``B`` counts the final bin with the single-marker count primitive.

The physical plan loads the lane frontier once, compares two implicit bin
values at a time, and uses the hardware's paired marker-count primitive; all
``B`` logical counts are written to L4 exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .apu_v1_native_vector import (
    APUv1NativeRecipeMismatch,
    APUv1NativeTargetFacts,
    E,
    bind_abi,
    compare_view,
    describe,
    iv_of,
    match_increment,
    predicate_text,
    require_fill_zero,
    require_loop,
    require_no_carried_locals,
    ring16,
    summarize_native_function,
)


RECIPE = "dense_histogram"


def _mismatch(reason: str, stage: int = 2) -> APUv1NativeRecipeMismatch:
    return APUv1NativeRecipeMismatch(RECIPE, reason, stage=stage)


def _signed16(value: int) -> int:
    return value - (1 << 16) if value >= 1 << 15 else value


@dataclass(frozen=True)
class APUv1DenseHistogramAnalysis:
    """A proven dense-u16 histogram loop nest."""

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
            "bin_source": "bin_induction_variable",
            "increment": 1,
            "initial_value": 0,
            "input_layout": "one_u16_value_per_lane",
            "physical_plan": "one_lane_frontier_load_then_paired_immediate_counts",
            "count_pairing": "consecutive_implicit_bins",
        }


# The recognizer deliberately rejects every semantic mismatch independently.
# pylint: disable=too-many-locals,too-many-branches,too-many-statements
def analyze_apu_v1_dense_histogram(
    source_mlir: str,
    arguments,
    *,
    function: str,
    target=None,
) -> APUv1DenseHistogramAnalysis:
    """Prove the dense-histogram semantics or raise a named mismatch."""

    facts = APUv1NativeTargetFacts.from_target(target)
    sources = [arg for arg in arguments if arg.source == "argument"]
    results = [arg for arg in arguments if arg.source == "result"]
    if len(sources) != 1 or len(results) != 1:
        raise _mismatch(
            "expects one source array (values) and one returned array (counts); "
            f"got {len(sources)} source and {len(results)} returned arrays",
            stage=0,
        )
    values, counts = sources[0], results[0]
    if len(values.shape) != 1 or len(counts.shape) != 1:
        raise _mismatch("values and counts must both be rank-1 arrays", stage=0)

    if values.dtype != "ui16" or counts.dtype != "ui16":
        raise _mismatch(
            f"dtypes are {values.name}:{values.dtype}, {counts.name}:{counts.dtype}; "
            "the template implements uint16 values and uint16 counts"
        )

    summary = summarize_native_function(source_mlir, function, RECIPE)
    abi = bind_abi(summary, arguments, RECIPE)
    names = abi.names
    ((values_formal, _),) = abi.sources
    ((counts_array, _),) = abi.results
    require_no_carried_locals(summary, RECIPE)

    if len(summary.effects) != 1:
        raise _mismatch(
            f"expects exactly one store (the counter update); found "
            f"{len(summary.effects)}",
            stage=1,
        )
    (effect,) = summary.effects
    if effect.base is not counts_array:
        raise _mismatch(
            f"the only store writes {names.get(effect.base, effect.base.name)!r}, "
            f"not the returned counts {counts.name!r}",
            stage=1,
        )
    if len(effect.loops) != 2:
        raise _mismatch(
            "the counter update must sit inside exactly two loops (bins x items); "
            f"found {len(effect.loops)}",
            stage=1,
        )
    if len(effect.indices) != 1 or iv_of(effect.indices[0]) not in effect.loops:
        raise _mismatch("counts must be indexed by an enclosing loop variable")
    bin_loop = iv_of(effect.indices[0])
    (item_loop,) = [loop for loop in effect.loops if loop is not bin_loop]
    if len(effect.guards) != 1 or effect.guards[0].kind != "cmp":
        raise _mismatch(
            "the counter update must be guarded by exactly one comparison "
            "(no else branch)",
            stage=1,
        )

    predicate, lhs, rhs = effect.guards[0].args
    value_load = E("load", (values_formal, (("iv", item_loop),)), 16)
    bin_value = E("iv", (bin_loop,), 16)
    views = (compare_view(lhs), compare_view(rhs))
    value_side = [view for view in views if view[0] == value_load]
    bin_side = [view for view in views if view[0] == bin_value]
    if len(value_side) != 1 or len(bin_side) != 1:
        raise _mismatch(
            "the guard compares "
            f"{describe(views[0][0], names)} with {describe(views[1][0], names)}; "
            f"the template compares {values.name}[item] with the bin index"
        )
    value_extension = value_side[0][1]
    bin_extension = bin_side[0][1]
    if value_extension not in {None, "u", "s"} or bin_extension not in {
        None,
        "index",
        "u",
    }:
        raise _mismatch("the comparison operands are not exact 16-bit values")
    if predicate != "eq":
        raise _mismatch(
            f"the guard uses {predicate_text(predicate)}; the template counts "
            "equality (==) only"
        )

    counter = E("load", (counts_array, effect.indices), 16)
    update = ring16(effect.value)
    increment = match_increment(update, counter)
    if increment is None:
        raise _mismatch(
            f"the update is {counts.name}[bin] = {describe(update, names)}; the "
            f"template implements {counts.name}[bin] += 1"
        )
    if increment != 1:
        raise _mismatch(
            f"{counts.name}[bin] is incremented by {_signed16(increment)}; the "
            "template implements += 1"
        )
    require_fill_zero(summary, counts_array, counts.name, RECIPE)

    if values.mode != "in":
        raise _mismatch(f"{values.name!r} is written; the template reads it only")
    items, bins = values.shape[0], counts.shape[0]
    require_loop(bin_loop, lower=0, upper=bins, what="bin", recipe=RECIPE)
    require_loop(item_loop, lower=0, upper=items, what="item", recipe=RECIPE)
    if value_extension == "s" and bins > 1 << (facts.lane_bits - 1):
        raise _mismatch(
            "sign-extended values never equal bins >= 2**15, but the template "
            "compares raw 16-bit patterns"
        )
    if items != facts.lanes:
        raise _mismatch(
            f"{values.name} has {items} items; the template needs exactly one "
            f"complete frontier of {facts.lanes} lanes",
            stage=3,
        )
    if not 1 <= bins <= 1 << facts.lane_bits:
        raise _mismatch(
            f"{bins} bins exceed the {facts.lane_bits}-bit immediate range",
            stage=3,
        )

    return APUv1DenseHistogramAnalysis(
        function=function,
        values_role=values.name,
        counts_role=counts.name,
        items=items,
        bins=bins,
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
        pairs, odd = divmod(self.analysis.bins, 2)
        return {
            "dma_l4_to_l1_32k": 1,
            "gvml_load_16": 1,
            "gvml_eq_imm_16": self.analysis.bins,
            "gvml_2_fast_count_m_g32k": pairs,
            "gvml_count_m_g32k": odd,
            "scalar_l4_store_u16": self.analysis.bins,
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
        graph.op(
            "COUNT_M_G32K",
            "gvml_count_m_g32k",
            count=inventory["gvml_count_m_g32k"],
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
                "derive every immediate bin from the proven bin induction variable",
                "return all bins without overwrite",
            ],
        }

    def emit_device_source(self) -> str:
        analysis = self.analysis
        odd_tail = ""
        if analysis.bins % 2:
            odd_tail = """
    {
        unsigned int count = 0;
        gvml_eq_imm_16(GVML_MRK0, values_vr, (uint16_t)(HIST_BINS - 1));
        gvml_count_m_g32k(&count, GVML_MRK0);
        counts_l4[HIST_BINS - 1] = (uint16_t)count;
    }
"""
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

_Static_assert(HIST_BINS <= 65536, "bins are 16-bit immediates");

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

    for (uint32_t bin = 0; bin + 1 < HIST_BINS; bin += 2) {{
        unsigned int count0 = 0;
        unsigned int count1 = 0;
        gvml_eq_imm_16(GVML_MRK0, values_vr, (uint16_t)bin);
        gvml_eq_imm_16(GVML_MRK1, values_vr, (uint16_t)(bin + 1));
        gvml_2_fast_count_m_g32k(
            &count0, &count1, GVML_MRK0, GVML_MRK1, GVML_VR16_1);
        counts_l4[bin] = (uint16_t)count0;
        counts_l4[bin + 1] = (uint16_t)count1;
    }}
{odd_tail}
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
