# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact GVML lowering for batched squared-distance argmin programs.

Supported semantics (proved structurally from the retained MLIR; see
:mod:`apu_v1_native_vector`)::

    for p in range(P):
        best: uint16 = 0
        for d in range(D):
            best += (points[p, d] - centers[0, d]) ** 2      # via any temps
        index: uint16 = 0
        for k in range(1, K):
            dist: uint16 = 0
            for d in range(D):
                dist += (points[p, d] - centers[k, d]) ** 2
            if dist < best:          # strict; or `best > dist`
                best = dist
                index = k
        labels[p] = index

all arithmetic modulo ``2**16`` with ``points: uint16[P, D]``,
``centers: uint16[K, D]``, ``labels: uint16[P]`` (an output argument that is
completely overwritten).  The difference may be written either way round and
through scalar temporaries.  The comparison is the 16-bit signed (``slt``,
Allo's lowering of ``uint16 <``) or unsigned (``ult``, after zero extension)
order of the wrapped distances; the template reproduces either exactly.
Roles are bound from the access pattern (which array is indexed by the point
loop), not from argument order or names.

Rejected with :class:`APUv1NativeRecipeMismatch` naming the difference: a
nonzero initial distance or index, ``labels`` storing anything but the argmin
index (e.g. the running minimum distance), a non-strict or reversed
predicate, a candidate loop not starting at 1, any other distance term,
extra stores, guarded label writes, non-16-bit locals, non-uint16 dtypes.

Shape domain: ``P`` equals the VR lane count (one point per lane);
``1 <= D <= VMR slots`` (one resident VMR per dimension, 48 on Gemini-I);
``2 <= K <= 65535`` (uint16 labels); the 512-byte-padded center image fits in
the per-APUC L3.

The physical ABI transposes points to dimension-major order so each dimension
is one VMR.  The plan applies ``(p - c)^2 == p^2 + c^2 - 2*p*c (mod 2**16)``,
computes the point norm once, and streams one candidate dot product through
low-numbered VRs.  Signed i16 ordering is preserved by flipping the sign bit
before the unsigned GVML comparison unless ``argument_bounds`` prove every
distance below ``2**15`` (then signed and unsigned orders coincide).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .apu_v1_native_vector import (
    APUv1NativeRecipeMismatch,
    APUv1NativeTargetFacts,
    E,
    bind_abi,
    compare_view,
    describe,
    effective_predicate,
    loop_name,
    predicate_text,
    require_loop,
    ring16,
    summarize_native_function,
)


RECIPE = "squared_l2_argmin"
MAX_CANDIDATES = 65535
_MIRROR = {
    "slt": "sgt",
    "sgt": "slt",
    "sle": "sge",
    "sge": "sle",
    "ult": "ugt",
    "ugt": "ult",
    "ule": "uge",
    "uge": "ule",
    "eq": "eq",
    "ne": "ne",
}


def _mismatch(reason: str, stage: int = 2) -> APUv1NativeRecipeMismatch:
    return APUv1NativeRecipeMismatch(RECIPE, reason, stage=stage)


@dataclass(frozen=True)
class APUv1SquaredL2ArgminAnalysis:
    """A proven squared-L2/strict-argmin loop nest."""

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
            "tie_breaking": "first_minimum",
            "unsigned_order_proven": self.unsigned_order_proven,
            "maximum_squared_distance": self.maximum_squared_distance,
            "point_layout": "dimension_major_32k_vmr",
            "center_image_values": self.center_image_values,
            "candidate_residency": "one_streamed_dot_product_in_vr",
            "algebra": "sum(p*p)+sum(c*c)-2*sum(p*c) modulo 2^16",
        }


def _squared_difference(term, point_loop, dim_loop, center_index, formals):
    """Return (points formal, centers formal) when ``term`` is the squared delta."""

    if term.kind != "mul" or term.args[0] != term.args[1]:
        return None
    delta = term.args[0]
    if delta.kind != "sub":
        return None
    point_indices = (("iv", point_loop), ("iv", dim_loop))
    center_indices = (center_index, ("iv", dim_loop))
    loads = delta.args
    if any(item.kind != "load" or item.args[0] not in formals for item in loads):
        return None
    by_indices = {item.args[1]: item.args[0] for item in loads}
    if set(by_indices) != {point_indices, center_indices}:
        return None
    return by_indices[point_indices], by_indices[center_indices]


def _distance_reduction(summary, value, parents):
    """Return (loop, local, term, init) when ``value`` is a summed reduction."""

    if value.kind != "loopout":
        return None
    loop, local = value.args
    loop_summary = summary.summaries.get(loop)
    if loop_summary is None or tuple(loop_summary.parent_loops) != parents:
        return None
    if set(loop_summary.updates) != {local}:
        return None
    update = ring16(loop_summary.updates[local])
    carry = E("carry", (local, loop), 16)
    if update.kind != "add" or carry not in update.args:
        return None
    lhs, rhs = update.args
    term = rhs if lhs == carry else lhs
    return loop, local, term, ring16(loop_summary.init[local])


# The recognizer deliberately rejects every semantic mismatch independently.
# pylint: disable=too-many-locals,too-many-branches,too-many-statements
def analyze_apu_v1_squared_l2_argmin(
    source_mlir: str,
    arguments,
    *,
    function: str,
    argument_bounds=None,
    target=None,
) -> APUv1SquaredL2ArgminAnalysis:
    """Prove the squared-L2 strict-argmin semantics or raise a named mismatch."""

    facts = APUv1NativeTargetFacts.from_target(target)
    sources = [arg for arg in arguments if arg.source == "argument"]
    results = [arg for arg in arguments if arg.source == "result"]
    if len(sources) != 3 or results:
        raise _mismatch(
            "expects three source arrays (points, centers, labels) and no "
            f"returned array; got {len(sources)} source and {len(results)} returned",
            stage=0,
        )
    if sorted(len(arg.shape) for arg in sources) != [1, 2, 2]:
        raise _mismatch(
            "expects one rank-1 labels array and two rank-2 point/center arrays",
            stage=0,
        )

    if any(arg.dtype != "ui16" for arg in sources):
        raise _mismatch(
            "the template implements uint16 points, centers, and labels; got "
            + ", ".join(f"{arg.name}:{arg.dtype}" for arg in sources)
        )

    summary = summarize_native_function(source_mlir, function, RECIPE)
    abi = bind_abi(summary, arguments, RECIPE)
    names = abi.names
    formals = dict(abi.sources)

    if len(summary.effects) != 1:
        raise _mismatch(
            f"expects exactly one array store (labels[point]); found "
            f"{len(summary.effects)}",
            stage=1,
        )
    (effect,) = summary.effects
    if effect.base not in formals or len(formals[effect.base].shape) != 1:
        raise _mismatch("the only store must write the rank-1 labels argument", stage=1)
    labels_formal = effect.base
    labels = formals[labels_formal]
    if len(effect.loops) != 1 or effect.guards:
        raise _mismatch(
            f"{labels.name}[point] must be written unconditionally inside exactly "
            "one loop (the point loop)",
            stage=1,
        )
    (point_loop,) = effect.loops
    if effect.indices != (("iv", point_loop),):
        raise _mismatch(f"{labels.name} must be indexed by the point loop variable")

    stored = ring16(effect.value)
    if stored.kind != "loopout":
        raise _mismatch(
            f"{labels.name}[point] stores {describe(stored, names)}; the template "
            "stores the argmin candidate index"
        )
    candidate_loop, stored_local = stored.args
    candidate_summary = summary.summaries[candidate_loop]
    if tuple(candidate_summary.parent_loops) != (point_loop,):
        raise _mismatch("the candidate loop must be nested directly in the point loop")
    if len(candidate_summary.updates) != 2:
        raise _mismatch(
            f"candidate loop {loop_name(candidate_loop)!r} must update exactly two "
            "locals (best distance and best index); it updates "
            f"{sorted(local.name or 'local' for local in candidate_summary.updates)}"
        )
    updates = {
        local: ring16(value) for local, value in candidate_summary.updates.items()
    }
    candidate_index = E("iv", (candidate_loop,), 16)
    index_locals = [
        local
        for local, value in updates.items()
        if value.kind == "select" and value.args[1] == candidate_index
    ]
    if len(index_locals) != 1:
        raise _mismatch("no local records the candidate index on a strict improvement")
    (index_local,) = index_locals
    (best_local,) = [local for local in updates if local is not index_local]
    if stored_local is not index_local:
        raise _mismatch(
            f"{labels.name}[point] stores {stored_local.name or 'a local'!r} (the "
            f"running minimum distance), not the argmin index "
            f"{index_local.name or 'local'!r}"
        )
    for local in (index_local, best_local):
        if local.width != 16:
            raise _mismatch(
                f"local {local.name!r} is i{local.width}; the template keeps "
                "16-bit distances and indices"
            )

    index_update = updates[index_local]
    best_update = updates[best_local]
    condition = index_update.args[0]
    best_carry = E("carry", (best_local, candidate_loop), 16)
    index_carry = E("carry", (index_local, candidate_loop), 16)
    if index_update.args[2] != index_carry:
        raise _mismatch(
            f"{index_local.name!r} is not kept when the candidate does not improve"
        )
    if (
        best_update.kind != "select"
        or best_update.args[0] != condition
        or best_update.args[2] != best_carry
    ):
        raise _mismatch(
            f"{best_local.name!r} is not replaced under the same condition that "
            f"records the index (update: {describe(best_update, names)})"
        )
    candidate_distance = best_update.args[1]
    reduction = _distance_reduction(
        summary, candidate_distance, (point_loop, candidate_loop)
    )
    if reduction is None:
        raise _mismatch(
            f"{best_local.name!r} is replaced by {describe(candidate_distance, names)}"
            "; the template replaces it by the candidate's summed distance"
        )
    dim_loop, distance_local, term, distance_init = reduction
    if distance_local.width != 16:
        raise _mismatch(f"local {distance_local.name!r} is not a 16-bit distance")
    if distance_init != E("const", (0,), 16):
        raise _mismatch(
            f"candidate distance {distance_local.name!r} starts at "
            f"{describe(distance_init, names)}; the template requires 0"
        )
    roles = _squared_difference(
        term, point_loop, dim_loop, ("iv", candidate_loop), formals
    )
    if roles is None:
        raise _mismatch(
            f"candidate distance accumulates {describe(term, names)}; the template "
            "accumulates (points[point, d] - centers[candidate, d])**2"
        )
    points_formal, centers_formal = roles

    if condition.kind != "cmp":
        raise _mismatch("the improvement condition is not an integer comparison")
    predicate, lhs, rhs = condition.args
    (lhs_value, lhs_extension), (rhs_value, rhs_extension) = (
        compare_view(lhs),
        compare_view(rhs),
    )
    meaning = effective_predicate(predicate, lhs_extension, rhs_extension)
    if meaning is None:
        raise _mismatch("the comparison mixes signed and unsigned extensions")
    if lhs_value == candidate_distance and rhs_value == best_carry:
        normalized = meaning
    elif lhs_value == best_carry and rhs_value == candidate_distance:
        normalized = _MIRROR[meaning]
    else:
        raise _mismatch(
            f"the condition compares {describe(lhs_value, names)} with "
            f"{describe(rhs_value, names)}; the template compares the candidate "
            f"distance with {best_local.name!r}"
        )
    if normalized not in {"slt", "ult"}:
        detail = {
            "sle": "non-strict <= keeps the last minimum on ties",
            "ule": "non-strict <= keeps the last minimum on ties",
            "sgt": "> selects the maximum",
            "ugt": "> selects the maximum",
            "sge": ">= selects the last maximum",
            "uge": ">= selects the last maximum",
        }.get(normalized, "it is not an ordering")
        raise _mismatch(
            f"candidate distance {predicate_text(normalized)} best: {detail}; the "
            "template implements strict < with first-minimum tie breaking"
        )

    initial_best = ring16(candidate_summary.init[best_local])
    first = _distance_reduction(summary, initial_best, (point_loop,))
    if first is None or first[1] is not best_local:
        raise _mismatch(
            f"{best_local.name!r} enters the candidate loop as "
            f"{describe(initial_best, names)}; the template starts from candidate "
            "0's summed distance"
        )
    first_dim_loop, _, first_term, first_init = first
    if first_init != E("const", (0,), 16):
        raise _mismatch(
            f"{best_local.name!r} starts at {describe(first_init, names)} before "
            "accumulating candidate 0; the template requires 0"
        )
    if _squared_difference(
        first_term, point_loop, first_dim_loop, ("const", 0), formals
    ) != (points_formal, centers_formal):
        raise _mismatch(
            f"candidate 0's distance accumulates {describe(first_term, names)}; "
            "the template accumulates (points[point, d] - centers[0, d])**2"
        )
    initial_index = ring16(candidate_summary.init[index_local])
    if initial_index != E("const", (0,), 16):
        raise _mismatch(
            f"{index_local.name!r} starts at {describe(initial_index, names)}; "
            "candidate 0 must be the initial index"
        )

    points_record = formals[points_formal]
    centers_record = formals[centers_formal]
    if points_record.mode != "in" or centers_record.mode != "in":
        raise _mismatch("points and centers must be read-only")
    point_count, dimensions = points_record.shape
    candidates, center_dimensions = centers_record.shape
    if center_dimensions != dimensions or labels.shape != (point_count,):
        raise _mismatch(
            f"shapes {points_record.shape}, {centers_record.shape}, {labels.shape} "
            "do not form points[P, D], centers[K, D], labels[P]"
        )
    require_loop(point_loop, lower=0, upper=point_count, what="point", recipe=RECIPE)
    require_loop(
        candidate_loop, lower=1, upper=candidates, what="candidate", recipe=RECIPE
    )
    for loop in (first_dim_loop, dim_loop):
        require_loop(loop, lower=0, upper=dimensions, what="dimension", recipe=RECIPE)

    if point_count != facts.lanes:
        raise _mismatch(
            f"{point_count} points; the template maps one point per lane of one "
            f"{facts.lanes}-lane frontier",
            stage=3,
        )
    if not 2 <= candidates <= min(MAX_CANDIDATES, (1 << facts.lane_bits) - 1):
        raise _mismatch(
            f"{candidates} candidates; the template needs 2..{MAX_CANDIDATES} "
            "(uint16 labels)",
            stage=3,
        )
    if not 1 <= dimensions <= facts.vmr_slots:
        raise _mismatch(
            f"{dimensions} dimensions; each dimension stays resident in one of "
            f"{facts.vmr_slots} VMRs",
            stage=3,
        )
    center_image = math.ceil(candidates * dimensions * 2 / 512) * 512
    if center_image > facts.l3_bytes:
        raise _mismatch(
            f"the center image needs {center_image} bytes of L3; the target has "
            f"{facts.l3_bytes}",
            stage=3,
        )

    maximum_squared_distance = None
    bounds = dict(argument_bounds or ())
    point_bounds = bounds.get(points_record.name)
    center_bounds = bounds.get(centers_record.name)
    if point_bounds is not None and center_bounds is not None:
        limit = np.iinfo(np.uint16).max
        if not (
            0 <= point_bounds[0] <= point_bounds[1] <= limit
            and 0 <= center_bounds[0] <= center_bounds[1] <= limit
        ):
            raise _mismatch("argument_bounds for points/centers must lie within uint16")
        maximum_delta = max(
            abs(point_bounds[0] - center_bounds[1]),
            abs(point_bounds[1] - center_bounds[0]),
        )
        maximum_squared_distance = dimensions * maximum_delta * maximum_delta

    return APUv1SquaredL2ArgminAnalysis(
        function=function,
        points_role=points_record.name,
        centers_role=centers_record.name,
        labels_role=labels.name,
        points=point_count,
        dimensions=dimensions,
        candidates=candidates,
        comparison=normalized,
        unsigned_order_proven=(
            normalized == "ult"
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
