# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused GVML lowering for five first/second bivariate moments.

Supported semantics (proved structurally from the retained MLIR; see
:mod:`apu_v1_native_vector`)::

    stats: uint16[B, 5] = 0                    # whole-array zero initializer
    for b in range(B):                         # batch loop (either order)
        for p in range(P):                     # point loop
            stats[b, 0] += x[b, p]
            stats[b, 1] += y[b, p]
            stats[b, 2] += x[b, p] * x[b, p]
            stats[b, 3] += y[b, p] * y[b, p]
            stats[b, 4] += x[b, p] * y[b, p]   # (or y * x)

all modulo ``2**16`` with ``x, y: uint16[B, P]`` and ``stats`` the only
returned array.  The five statements may appear in any order, each column is
updated exactly once per iteration, and every update adds its term with
coefficient one.  Operand roles are bound from the IR: ``x`` is the argument
accumulated into column 0, ``y`` the other one, and columns 1-4 must then
match the table above; a program whose columns 0 and 1 are swapped relative
to its products is rejected.

Rejected with :class:`APUv1NativeRecipeMismatch` naming the difference: a
nonzero initializer, a different term or coefficient in any column, a
repeated or missing column, guards, extra stores, loop-carried scalars,
non-uint16 dtypes, offset indices.

Shape domain: ``P`` is a power of two and ``B * P`` is at most the VR lane
count.  Batch ``b`` occupies lanes ``[b*P, (b+1)*P)`` of one VR; a frontier
with ``B * P`` below the lane count is zero padded (padding groups are reduced
but never read back).  Both inputs are loaded once, the three products are
formed in parallel VRs, and all five quantities use native grouped reduction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .apu_v1_native_vector import (
    APUv1NativeRecipeMismatch,
    APUv1NativeTargetFacts,
    E,
    bind_abi,
    describe,
    iv_of,
    require_fill_zero,
    require_loop,
    require_no_carried_locals,
    ring16,
    ring_product,
    summarize_native_function,
)


RECIPE = "bivariate_moments"
MOMENT_COUNT = 5
_VRS_USED = 6
_VMRS_USED = 3


def _mismatch(reason: str, stage: int = 2) -> APUv1NativeRecipeMismatch:
    return APUv1NativeRecipeMismatch(RECIPE, reason, stage=stage)


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
    lanes: int = 32768

    def manifest(self) -> dict:
        return {
            "kind": "bivariate_moments_u16",
            "function": self.function,
            "x_role": self.x_role,
            "y_role": self.y_role,
            "stats_role": self.stats_role,
            "batches": self.batches,
            "points_per_batch": self.points_per_batch,
            "lanes": self.lanes,
            "stat_order": ["sum_x", "sum_y", "sum_xx", "sum_yy", "sum_xy"],
            "initial_value": 0,
            "input_layout": "batch_major_contiguous_lane_groups_zero_padded",
            "arithmetic": "uint16_ring",
        }


# The recognizer deliberately rejects every semantic mismatch independently.
# pylint: disable=too-many-locals,too-many-branches,too-many-statements
def analyze_apu_v1_bivariate_moments(
    source_mlir: str,
    arguments,
    *,
    function: str,
    target=None,
) -> APUv1BivariateMomentsAnalysis:
    """Prove the five-moment semantics or raise a named mismatch."""

    facts = APUv1NativeTargetFacts.from_target(target)
    sources = [arg for arg in arguments if arg.source == "argument"]
    results = [arg for arg in arguments if arg.source == "result"]
    if len(sources) != 2 or len(results) != 1:
        raise _mismatch(
            "expects two source arrays (x, y) and one returned array (stats); "
            f"got {len(sources)} source and {len(results)} returned",
            stage=0,
        )
    stats = results[0]
    if any(len(arg.shape) != 2 for arg in sources + results):
        raise _mismatch("x, y, and stats must be rank-2 arrays", stage=0)
    if sources[0].shape != sources[1].shape:
        raise _mismatch("x and y must have the same shape", stage=0)
    if stats.shape != (sources[0].shape[0], MOMENT_COUNT):
        raise _mismatch(
            f"stats has shape {stats.shape}; the template returns "
            f"({sources[0].shape[0]}, {MOMENT_COUNT})",
            stage=0,
        )

    if any(arg.dtype != "ui16" for arg in sources + results):
        raise _mismatch(
            "the template implements uint16 inputs and statistics; got "
            + ", ".join(f"{arg.name}:{arg.dtype}" for arg in sources + results)
        )

    summary = summarize_native_function(source_mlir, function, RECIPE)
    abi = bind_abi(summary, arguments, RECIPE)
    names = abi.names
    ((stats_array, _),) = abi.results
    formals = dict(abi.sources)
    require_no_carried_locals(summary, RECIPE)

    if len(summary.effects) != MOMENT_COUNT:
        raise _mismatch(
            f"expects exactly {MOMENT_COUNT} stores (one per statistic); found "
            f"{len(summary.effects)}",
            stage=1,
        )
    first = summary.effects[0]
    if len(first.loops) != 2 or not first.indices:
        raise _mismatch(
            "every statistic update must sit inside exactly two loops", stage=1
        )
    batch_loop = iv_of(first.indices[0])
    if batch_loop not in first.loops:
        raise _mismatch("stats must be indexed by the batch loop variable", stage=1)
    (point_loop,) = [loop for loop in first.loops if loop is not batch_loop]

    terms = {}
    for effect in summary.effects:
        if effect.base is not stats_array:
            raise _mismatch(
                f"a store writes {names.get(effect.base, effect.base.name)!r}, not "
                f"the returned {stats.name!r}",
                stage=1,
            )
        if effect.loops != first.loops:
            raise _mismatch("all statistic updates must share one loop nest", stage=1)
        if effect.guards:
            raise _mismatch("statistic updates must be unconditional")
        if (
            len(effect.indices) != 2
            or iv_of(effect.indices[0]) is not batch_loop
            or effect.indices[1][0] != "const"
        ):
            raise _mismatch(f"{stats.name} must be indexed [batch, <literal column>]")
        column = effect.indices[1][1]
        if column in terms or not 0 <= column < MOMENT_COUNT:
            raise _mismatch(
                f"column {column} is updated more than once or is out of range"
            )
        update = ring16(effect.value)
        current = E("load", (stats_array, effect.indices), 16)
        if update.kind != "add" or current not in update.args:
            raise _mismatch(
                f"column {column} is set to {describe(update, names)}; the "
                f"template accumulates {stats.name}[b, {column}] += term"
            )
        lhs, rhs = update.args
        terms[column] = rhs if lhs == current else lhs

    def operand(formal):
        return E("load", (formal, (("iv", batch_loop), ("iv", point_loop))), 16)

    x_term = terms[0]
    if x_term.kind != "load" or x_term.args[0] not in formals:
        raise _mismatch(
            f"column 0 accumulates {describe(x_term, names)}; the template's "
            "column 0 is x[b, p] for one source argument"
        )
    x_formal = x_term.args[0]
    (y_formal,) = [formal for formal in formals if formal is not x_formal]
    x_value, y_value = operand(x_formal), operand(y_formal)
    expected = {
        0: x_value,
        1: y_value,
        2: ring_product(x_value, x_value),
        3: ring_product(y_value, y_value),
        4: ring_product(x_value, y_value),
    }
    labels = {0: "x", 1: "y", 2: "x*x", 3: "y*y", 4: "x*y"}
    x, y = formals[x_formal], formals[y_formal]
    for column in range(MOMENT_COUNT):
        if terms[column] != expected[column]:
            raise _mismatch(
                f"column {column} accumulates {describe(terms[column], names)}; "
                f"with x = {x.name!r} (column 0) the template's column {column} is "
                f"{labels[column]} = {describe(expected[column], names)}"
            )
    require_fill_zero(summary, stats_array, stats.name, RECIPE)

    if x.mode != "in" or y.mode != "in":
        raise _mismatch("x and y must be read-only")
    batches, points = x.shape
    require_loop(batch_loop, lower=0, upper=batches, what="batch", recipe=RECIPE)
    require_loop(point_loop, lower=0, upper=points, what="point", recipe=RECIPE)
    if points & (points - 1) or batches * points > facts.lanes:
        raise _mismatch(
            f"{batches}x{points}: the grouped reduction needs a power-of-two "
            f"group size and at most {facts.lanes} lanes in total",
            stage=3,
        )
    if facts.writable_vrs < _VRS_USED or facts.vmr_slots < _VMRS_USED:
        raise _mismatch("the fused plan needs six VRs and three VMRs", stage=3)

    return APUv1BivariateMomentsAnalysis(
        function=function,
        x_role=x.name,
        y_role=y.name,
        stats_role=stats.name,
        batches=batches,
        points_per_batch=points,
        lanes=facts.lanes,
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

        def frontier(values):
            image = np.zeros(analysis.lanes, dtype=np.uint16)
            image[: values.size] = np.asarray(values, dtype=np.uint16).reshape(-1)
            return image

        return {
            **inputs,
            analysis.x_role: frontier(x),
            analysis.y_role: frontier(y),
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
            "physical_input_transforms": {
                self.analysis.x_role: "row_major_then_zero_pad_to_lane_frontier",
                self.analysis.y_role: "row_major_then_zero_pad_to_lane_frontier",
            },
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

_Static_assert(MOMENT_BATCHES * MOMENT_POINTS <= {analysis.lanes},
               "moment groups must fit one APUv1 frontier");

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
