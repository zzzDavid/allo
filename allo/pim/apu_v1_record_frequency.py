# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact GVML lowering for fixed-width record-frequency reductions.

Supported semantics (proved structurally from the retained MLIR; see
:mod:`apu_v1_native_vector`)::

    counts: uint16[Q] = 0                       # whole-array zero initializer
    for q in range(Q):                          # query loop (either order)
        for i in range(N):                      # record loop
            if records[c0, i] == queries[c0, q]:    # one nested equality per
                if records[c1, i] == queries[c1, q]:  # chunk, matching chunk
                    ...                                 # indices, any order
                        counts[q] += 1          # increment exactly 1

i.e. ``counts[q] = #{i : all_c records[c, i] == queries[c, q]}`` modulo
``2**16``, with ``records: uint16[C, N]``, ``queries: uint16[C, Q]`` and
``counts`` the only returned array.  Each chunk ``0..C-1`` is compared exactly
once, records chunk ``c`` with queries chunk ``c``; operands may be compared
as raw 16-bit patterns or after the same extension.  Roles are bound from the
access pattern (which argument is indexed by the record loop), not from
argument order or names.

Rejected with :class:`APUv1NativeRecipeMismatch` naming the difference: any
increment other than 1, a nonzero initializer, a predicate other than ``==``,
a cross-chunk comparison, a missing or repeated chunk, an else branch, extra
stores, loop-carried scalars, non-uint16 dtypes, or bounds that do not cover
the arrays.

Shape domain: ``N`` equals the VR lane count (one record per lane); ``C``
records chunks stay resident, so ``1 <= C <= min(writable VRs - 1, VMRs)``
(one VR is the count scratch; 14 on Gemini-I); four markers; the compact
query image plus ``Q`` staged counts fit in the per-APUC L3; ``Q >= 1`` (odd
``Q`` counts its last query with the single-marker primitive).
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
    iv_of,
    match_increment,
    predicate_text,
    require_fill_zero,
    require_loop,
    require_no_carried_locals,
    ring16,
    summarize_native_function,
)


RECIPE = "record_frequency"
_MARKERS_USED = 4


def _mismatch(reason: str, stage: int = 2) -> APUv1NativeRecipeMismatch:
    return APUv1NativeRecipeMismatch(RECIPE, reason, stage=stage)


def _query_image_values(queries: int, chunks: int) -> int:
    # L4->L3 uses complete 512-byte (256 x uint16) transactions.
    return math.ceil(queries * chunks / 256) * 256


def loop_counter_type(queries: int) -> str:
    """The C type of the ARC loop counters of the record-frequency kernel.

    The store loop's counter reaches ``FREQ_QUERIES``, so 16-bit counters
    are correct up to 65,535 queries; the chunk loops are bounded by the
    vector-register file.  The width matters on the board: see
    :meth:`emit_device_source`.
    """

    return "uint16_t" if 0 < queries <= 0xFFFF else "uint32_t"


@dataclass(frozen=True)
class APUv1RecordFrequencyAnalysis:
    """A proven nested-equality fixed-record frequency computation."""

    function: str
    records_role: str
    queries_role: str
    counts_role: str
    records: int
    queries: int
    chunks: int

    @property
    def query_values(self) -> int:
        return self.queries * self.chunks

    @property
    def query_image_values(self) -> int:
        return _query_image_values(self.queries, self.chunks)

    @property
    def query_image_bytes(self) -> int:
        return self.query_image_values * 2

    def manifest(self) -> dict:
        return {
            "kind": "fixed_record_frequency_u16",
            "function": self.function,
            "records_role": self.records_role,
            "queries_role": self.queries_role,
            "counts_role": self.counts_role,
            "records": self.records,
            "queries": self.queries,
            "chunks": self.chunks,
            "increment": 1,
            "initial_value": 0,
            "record_layout": "chunk_major_32k_vmr",
            "query_layout": "chunk_major_compact_l3",
            "predicate": "conjunction_of_chunkwise_equality",
            "query_blocking": 2,
        }


def _classify(view, record_loop, query_loop, formals):
    """Return ("record"|"query", formal, chunk) for one comparison operand."""

    value, extension = view
    if value.kind != "load" or extension == "opaque":
        return None
    formal, indices = value.args
    if formal not in formals or len(indices) != 2 or indices[0][0] != "const":
        return None
    loop = iv_of(indices[1])
    if loop is record_loop:
        return "record", formal, indices[0][1], extension
    if loop is query_loop:
        return "query", formal, indices[0][1], extension
    return None


# The recognizer deliberately rejects every semantic mismatch independently.
# pylint: disable=too-many-locals,too-many-branches,too-many-statements
def analyze_apu_v1_record_frequency(
    source_mlir: str,
    arguments,
    *,
    function: str,
    target=None,
) -> APUv1RecordFrequencyAnalysis:
    """Prove the record-frequency semantics or raise a named mismatch."""

    facts = APUv1NativeTargetFacts.from_target(target)
    sources = [arg for arg in arguments if arg.source == "argument"]
    results = [arg for arg in arguments if arg.source == "result"]
    if len(sources) != 2 or len(results) != 1:
        raise _mismatch(
            "expects two source arrays (records, queries) and one returned array "
            f"(counts); got {len(sources)} source and {len(results)} returned",
            stage=0,
        )
    counts = results[0]
    if len(counts.shape) != 1 or any(len(arg.shape) != 2 for arg in sources):
        raise _mismatch("records and queries must be rank-2 and counts rank-1", stage=0)

    if any(arg.dtype != "ui16" for arg in sources + results):
        raise _mismatch(
            "the template implements uint16 records, queries, and counts; got "
            + ", ".join(f"{arg.name}:{arg.dtype}" for arg in sources + results)
        )

    summary = summarize_native_function(source_mlir, function, RECIPE)
    abi = bind_abi(summary, arguments, RECIPE)
    names = abi.names
    ((counts_array, _),) = abi.results
    formals = dict(abi.sources)
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
            "the counter update must sit inside exactly two loops "
            f"(queries x records); found {len(effect.loops)}",
            stage=1,
        )
    if len(effect.indices) != 1 or iv_of(effect.indices[0]) not in effect.loops:
        raise _mismatch("counts must be indexed by an enclosing loop variable")
    query_loop = iv_of(effect.indices[0])
    (record_loop,) = [loop for loop in effect.loops if loop is not query_loop]
    if not effect.guards:
        raise _mismatch("the counter update is not guarded by any comparison")

    records_formal = queries_formal = None
    chunks_seen = []
    for position, guard in enumerate(effect.guards):
        if guard.kind != "cmp":
            raise _mismatch(
                f"guard {position} is not a comparison (the update sits in an "
                "else branch or behind a non-comparison condition)"
            )
        predicate, lhs, rhs = guard.args
        sides = [
            _classify(compare_view(item), record_loop, query_loop, formals)
            for item in (lhs, rhs)
        ]
        kinds = sorted(side[0] for side in sides if side is not None)
        if kinds != ["query", "record"]:
            raise _mismatch(
                f"guard {position} compares {describe(compare_view(lhs)[0], names)} "
                f"with {describe(compare_view(rhs)[0], names)}; the template "
                "compares records[c, record] with queries[c, query]"
            )
        record_side = next(side for side in sides if side[0] == "record")
        query_side = next(side for side in sides if side[0] == "query")
        if predicate != "eq":
            raise _mismatch(
                f"guard {position} uses {predicate_text(predicate)}; the "
                "template implements equality (==) only"
            )
        if record_side[3] != query_side[3]:
            raise _mismatch(
                f"guard {position} extends its operands differently "
                f"({record_side[3]} vs {query_side[3]})"
            )
        records_formal = records_formal or record_side[1]
        queries_formal = queries_formal or query_side[1]
        if record_side[1] is not records_formal or query_side[1] is not queries_formal:
            raise _mismatch("the guards read records/queries from different arrays")
        if record_side[2] != query_side[2]:
            raise _mismatch(
                f"guard {position} compares records chunk {record_side[2]} with "
                f"queries chunk {query_side[2]}; the template compares matching "
                "chunks"
            )
        chunks_seen.append(record_side[2])
    if records_formal is queries_formal:
        raise _mismatch("records and queries must be distinct arguments")
    records, queries = formals[records_formal], formals[queries_formal]
    chunks = records.shape[0]
    if len(set(chunks_seen)) != len(chunks_seen):
        raise _mismatch(f"chunks {sorted(chunks_seen)} are compared more than once")
    if sorted(chunks_seen) != list(range(chunks)):
        missing = sorted(set(range(chunks)) - set(chunks_seen))
        raise _mismatch(
            f"compared chunks {sorted(chunks_seen)} do not cover all {chunks} "
            f"record chunks (missing {missing}) or exceed them"
        )

    counter = E("load", (counts_array, effect.indices), 16)
    update = ring16(effect.value)
    increment = match_increment(update, counter)
    if increment is None:
        raise _mismatch(
            f"the update is {counts.name}[query] = {describe(update, names)}; the "
            f"template implements {counts.name}[query] += 1"
        )
    if increment != 1:
        shown = increment - (1 << 16) if increment >= 1 << 15 else increment
        raise _mismatch(
            f"{counts.name}[query] is incremented by {shown}; the template "
            "implements += 1"
        )
    require_fill_zero(summary, counts_array, counts.name, RECIPE)

    if records.mode != "in" or queries.mode != "in":
        raise _mismatch("records and queries must be read-only")
    query_count = counts.shape[0]
    if queries.shape != (chunks, query_count):
        raise _mismatch(
            f"{queries.name} has shape {queries.shape}; expected "
            f"({chunks}, {query_count}) to match {records.name} and {counts.name}"
        )
    record_count = records.shape[1]
    require_loop(query_loop, lower=0, upper=query_count, what="query", recipe=RECIPE)
    require_loop(record_loop, lower=0, upper=record_count, what="record", recipe=RECIPE)

    if record_count != facts.lanes:
        raise _mismatch(
            f"{record_count} records; the template maps exactly one record per "
            f"lane of one {facts.lanes}-lane frontier",
            stage=3,
        )
    max_chunks = min(facts.writable_vrs - 1, facts.vmr_slots)
    if not 1 <= chunks <= max_chunks:
        raise _mismatch(
            f"{chunks} chunks; resident chunks need one VR each plus a count "
            f"scratch VR (at most {max_chunks})",
            stage=3,
        )
    if facts.marker_slots < _MARKERS_USED:
        raise _mismatch("the paired plan needs four marker registers", stage=3)
    l3_bytes = (_query_image_values(query_count, chunks) + query_count) * 2
    if l3_bytes > facts.l3_bytes:
        raise _mismatch(
            f"query image and staged counts need {l3_bytes} bytes of L3; the "
            f"target has {facts.l3_bytes}",
            stage=3,
        )

    return APUv1RecordFrequencyAnalysis(
        function=function,
        records_role=records.name,
        queries_role=queries.name,
        counts_role=counts.name,
        records=record_count,
        queries=query_count,
        chunks=chunks,
    )


@dataclass(frozen=True)
class APUv1RecordFrequencyLowering:
    """Complete physical ABI and source for resident record frequency."""

    analysis: APUv1RecordFrequencyAnalysis

    @property
    def route(self) -> str:
        return "gvml_record_frequency_resident_chunks_pair_count"

    @property
    def scratch_bytes(self) -> int:
        return 8

    def pack_inputs(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        analysis = self.analysis
        records = inputs[analysis.records_role]
        queries = inputs[analysis.queries_role]
        if records.shape != (analysis.chunks, analysis.records):
            raise ValueError("record-frequency records do not match analyzed shape")
        if queries.shape != (analysis.chunks, analysis.queries):
            raise ValueError("record-frequency queries do not match analyzed shape")
        query_image = np.zeros(analysis.query_image_values, dtype=np.uint16)
        query_image[: analysis.query_values] = queries.reshape(-1)
        return {
            **inputs,
            analysis.records_role: np.ascontiguousarray(records),
            analysis.queries_role: query_image,
        }

    def operation_inventory(self) -> dict[str, int]:
        analysis = self.analysis
        pairs, odd = divmod(analysis.queries, 2)
        return {
            "dma_l4_to_l3_512b": analysis.query_image_bytes // 512,
            "dma_l4_to_l1_32k": analysis.chunks,
            "gvml_load_16": analysis.chunks,
            "gvml_eq_imm_16": analysis.chunks * analysis.queries,
            "gvml_and_m": (analysis.chunks - 1) * analysis.queries,
            "gvml_2_fast_count_m_g32k": pairs,
            "gvml_count_m_g32k": odd,
            "scalar_l4_store_u16": analysis.queries,
        }

    def build_execution_graph(self, target, cost):
        """Cost the resident marker plan instead of the nested scalar loops."""

        from .apu_v1_native_cost import APUv1NativeCostGraphBuilder

        inventory = self.operation_inventory()
        graph = APUv1NativeCostGraphBuilder(self.route, inventory, target, cost)
        graph.move(
            "DMA_L4_TO_L3",
            "dma_l4_to_l3_512b",
            bytes=self.analysis.query_image_bytes,
        )
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
            "AND_M",
            "gvml_and_m",
            count=inventory["gvml_and_m"],
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
            "physical_input_transforms": {
                self.analysis.queries_role: "row_major_then_zero_pad_to_512_bytes"
            },
            "output_transform": "identity",
        }

    def emit_device_source(self) -> str:
        analysis = self.analysis
        count_tmp_vr = analysis.chunks
        # The ARC loop counters are as narrow as the extents allow.  The
        # paper's program counted queries and chunks in ``uint16_t``; the
        # ``uint32_t`` counters introduced with the structural recognizer
        # (2026-11-02) cost 2.4% on the board for the same instruction
        # count (release issue X-22: in one session the current program
        # measured 176,042 cycles, the same source with 16-bit counters
        # 172,178, the paper-era source 171,984).  The counter reaches
        # ``FREQ_QUERIES`` in the store loop, so 16 bits need
        # ``FREQ_QUERIES <= 65535``; the chunk loops are bounded by the
        # vector-register file (``FREQ_CHUNKS < 15``).
        counter_t = loop_counter_type(analysis.queries)
        return f"""#include <stdint.h>

#include <gsi/libsys/assert.h>
#include <gsi/libsys.h>
#include <gsi/libgal.h>
#include <gsi/gal-fast-funcs.h>
#include <gsi/libgvml_element_wise.h>
#include <gsi/libgvml_iv.h>
#include <gsi/libgvml_memory.h>

#include "gsi_dma.h"
#include "struct.h"
#include <gsi_device_profiling.h>

enum {{
    FREQ_RECORDS = {analysis.records},
    FREQ_QUERIES = {analysis.queries},
    FREQ_CHUNKS = {analysis.chunks},
    FREQ_QUERY_VALUES = {analysis.query_values},
    FREQ_QUERY_IMAGE_BYTES = {analysis.query_image_bytes},
}};

_Static_assert(FREQ_CHUNKS < 15, "resident chunks plus one count scratch VR");

PROF_VAR(total);

static void copy_queries_l4_to_l3(uint16_t *dst, const uint16_t *src)
{{
    gal_fast_l2dma_async_memcpy_init(GAL_L2DMA_APC_ID_0);
    for (uint32_t offset = 0; offset < FREQ_QUERY_IMAGE_BYTES; offset += 512) {{
        gal_fast_l2dma_mem_to_mem_512(
            (uint8_t *)dst + offset,
            (uint8_t *)src + offset,
            GAL_L2DMA_APC_ID_0);
    }}
    gal_fast_l2dma_async_memcpy_end(GAL_L2DMA_APC_ID_0);
}}

static int record_frequency(struct program_data *data)
{{
    const uint16_t *records_l4 = (const uint16_t *)gal_mem_handle_to_apu_ptr(
        data->mem_hndl_{analysis.records_role});
    const uint16_t *queries_l4 = (const uint16_t *)gal_mem_handle_to_apu_ptr(
        data->mem_hndl_{analysis.queries_role});
    uint16_t *counts_l4 = (uint16_t *)gal_mem_handle_to_apu_ptr(
        data->mem_hndl_{analysis.counts_role});
    uint16_t *queries_l3 = (uint16_t *)gal_fast_malloc_cache_aligned(
        FREQ_QUERY_IMAGE_BYTES, true);
    uint16_t *counts_l3 = (uint16_t *)gal_fast_malloc_cache_aligned(
        FREQ_QUERIES * sizeof(uint16_t), true);
    if (GSI_IS_ERR_PTR_OR_NULL(queries_l3) ||
        GSI_IS_ERR_PTR_OR_NULL(counts_l3)) {{
        gsi_error("record frequency: failed to allocate L3 images\\n");
        return gsi_status(ENOMEM);
    }}

    enum gvml_vr16 count_tmp_vr =
        (enum gvml_vr16)(GVML_VR16_0 + {count_tmp_vr});

    arc_counters_init();
    PROF_INIT(total);
    PROF_START(total);
    gal_set_l2dma_dma_mode(GAL_L2DMA_MODE_DIRECT);
    copy_queries_l4_to_l3(queries_l3, queries_l4);

    for (uint16_t chunk = 0; chunk < FREQ_CHUNKS; ++chunk) {{
        direct_dma_l4_to_l1_32k(
            (enum gvml_vm_reg)chunk,
            records_l4 + (uint32_t)chunk * FREQ_RECORDS);
        gvml_load_16(
            (enum gvml_vr16)(GVML_VR16_0 + chunk),
            (enum gvml_vm_reg)chunk);
    }}

    {counter_t} query = 0;
    for (; query + 1 < FREQ_QUERIES; query += 2) {{
        for ({counter_t} chunk = 0; chunk < FREQ_CHUNKS; ++chunk) {{
            enum gvml_vr16 record_vr =
                (enum gvml_vr16)(GVML_VR16_0 + chunk);
            gvml_eq_imm_16(
                chunk == 0 ? GVML_MRK0 : GVML_MRK2,
                record_vr,
                queries_l3[(uint32_t)chunk * FREQ_QUERIES + query]);
            if (chunk != 0)
                gvml_and_m(GVML_MRK0, GVML_MRK0, GVML_MRK2);

            gvml_eq_imm_16(
                chunk == 0 ? GVML_MRK1 : GVML_MRK3,
                record_vr,
                queries_l3[(uint32_t)chunk * FREQ_QUERIES + query + 1]);
            if (chunk != 0)
                gvml_and_m(GVML_MRK1, GVML_MRK1, GVML_MRK3);
        }}
        unsigned int first_count = 0;
        unsigned int second_count = 0;
        gvml_2_fast_count_m_g32k(
            &first_count,
            &second_count,
            GVML_MRK0,
            GVML_MRK1,
            count_tmp_vr);
        counts_l3[query] = (uint16_t)first_count;
        counts_l3[query + 1] = (uint16_t)second_count;
    }}
    if (query < FREQ_QUERIES) {{
        for ({counter_t} chunk = 0; chunk < FREQ_CHUNKS; ++chunk) {{
            enum gvml_vr16 record_vr =
                (enum gvml_vr16)(GVML_VR16_0 + chunk);
            gvml_eq_imm_16(
                chunk == 0 ? GVML_MRK0 : GVML_MRK2,
                record_vr,
                queries_l3[(uint32_t)chunk * FREQ_QUERIES + query]);
            if (chunk != 0)
                gvml_and_m(GVML_MRK0, GVML_MRK0, GVML_MRK2);
        }}
        unsigned int count = 0;
        gvml_count_m_g32k(&count, GVML_MRK0);
        counts_l3[query] = (uint16_t)count;
    }}

    for ({counter_t} index = 0; index < FREQ_QUERIES; ++index)
        counts_l4[index] = counts_l3[index];

    PROF_END(total);
    PROF_PRINT(total);
    return 0;
}}

GAL_TASK_ENTRY_POINT(apu_kernel_task, in, out)
{{
    struct program_cmd *cmd = (struct program_cmd *)in;
    gvml_init_once();
    return record_frequency(&cmd->data);
}}
"""


# ``MAX_RESIDENT_CHUNKS`` was a fixed hardware limit this module no longer
# defines (the resident-chunk bound is derived per program); its stale export
# made ``from ... import *`` fail (round 87, P-23 triage).
__all__ = [
    "APUv1RecordFrequencyAnalysis",
    "APUv1RecordFrequencyLowering",
    "analyze_apu_v1_record_frequency",
]
