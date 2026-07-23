# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact GVML lowering for fixed-width record-frequency reductions.

This route structurally recognizes

``counts[q] = sum_i all_c(records[c, i] == queries[c, q])``

for a complete 32K APUv1 lane frontier.  Record chunks stay resident in VRs,
two queries are compared concurrently with four markers, and the pair is
reduced with GVML's native two-marker count.  The implementation is generic
over role/function names and supports up to ten uint16 chunks per record.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

import numpy as np


VR_LANES = 32768
MAX_RESIDENT_CHUNKS = 10


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
        return math.ceil(self.query_values / 256) * 256

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
            "record_layout": "chunk_major_32k_vmr",
            "query_layout": "chunk_major_compact_l3",
            "predicate": "conjunction_of_chunkwise_equality",
            "query_blocking": 2,
        }


def _matches(text: str, pattern: str) -> list[re.Match[str]]:
    return list(re.finditer(pattern, text, flags=re.MULTILINE))


# The recognizer deliberately rejects every structural mismatch independently.
# pylint: disable=too-many-return-statements
def analyze_apu_v1_record_frequency(
    source_mlir: str,
    arguments,
    *,
    function: str,
) -> APUv1RecordFrequencyAnalysis | None:
    """Recognize the exact fixed-record count skeleton from retained MLIR."""

    source = [arg for arg in arguments if arg.source == "argument"]
    results = [arg for arg in arguments if arg.source == "result"]
    if len(source) != 2 or len(results) != 1:
        return None
    if any(arg.dtype != "ui16" for arg in source + results):
        return None
    counts = results[0]
    if len(counts.shape) != 1 or counts.mode != "out":
        return None
    query_count = counts.shape[0]

    candidates = []
    for records in source:
        if len(records.shape) != 2 or records.shape[1] != VR_LANES:
            continue
        chunks = records.shape[0]
        peers = [
            value
            for value in source
            if value is not records and value.shape == (chunks, query_count)
        ]
        if len(peers) == 1:
            candidates.append((records, peers[0], chunks))
    if len(candidates) != 1:
        return None
    records, queries, chunks = candidates[0]
    if not (1 <= chunks <= MAX_RESIDENT_CHUNKS and 1 <= query_count <= 256):
        return None
    if records.mode != "in" or queries.mode != "in":
        return None

    exact_counts = {
        r"\baffine\.for\b": 2,
        r"\barith\.cmpi\s+eq\b": chunks,
        r"\bscf\.if\b": chunks,
        r"\barith\.addi\b": 1,
        r"\baffine\.store\b": 1,
        r"\bmemref\.alloc\b": 1,
        r"\blinalg\.fill\b": 1,
        r"\baffine\.load\b": 2 * chunks + 1,
    }
    if any(
        len(_matches(source_mlir, pattern)) != wanted
        for pattern, wanted in exact_counts.items()
    ):
        return None
    outer_matches = _matches(
        source_mlir,
        rf"\baffine\.for\s+(%[-\w.$]+)\s*=\s*0\s+to\s+{query_count}\b",
    )
    inner_matches = _matches(
        source_mlir,
        r"\baffine\.for\s+(%[-\w.$]+)\s*=\s*0\s+to\s+32768\b",
    )
    if len(outer_matches) != 1 or len(inner_matches) != 1:
        return None
    outer, inner = outer_matches[0].group(1), inner_matches[0].group(1)

    source_positions = {id(arg): index for index, arg in enumerate(source)}
    record_ssa = f"%arg{source_positions[id(records)]}"
    query_ssa = f"%arg{source_positions[id(queries)]}"
    for chunk in range(chunks):
        record_loads = _matches(
            source_mlir,
            rf"(?P<value>%[-\w.$]+)\s*=\s*affine\.load\s+"
            rf"{re.escape(record_ssa)}\[{chunk},\s*{re.escape(inner)}\]",
        )
        query_loads = _matches(
            source_mlir,
            rf"(?P<value>%[-\w.$]+)\s*=\s*affine\.load\s+"
            rf"{re.escape(query_ssa)}\[{chunk},\s*{re.escape(outer)}\]",
        )
        if len(record_loads) != 1 or len(query_loads) != 1:
            return None
        lhs = re.escape(record_loads[0].group("value"))
        rhs = re.escape(query_loads[0].group("value"))
        if not re.search(
            rf"arith\.cmpi\s+eq,\s*(?:{lhs},\s*{rhs}|{rhs},\s*{lhs})\b",
            source_mlir,
        ):
            return None
    if not re.search(
        rf"affine\.store\s+%[-\w.$]+\s*,\s*%[-\w.$]+\[{re.escape(outer)}\]"
        rf"[^\n]*to\s*=\s*\"{re.escape(counts.name)}\"",
        source_mlir,
    ):
        return None

    return APUv1RecordFrequencyAnalysis(
        function=function,
        records_role=records.name,
        queries_role=queries.name,
        counts_role=counts.name,
        records=VR_LANES,
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
        if count_tmp_vr + 2 > 15:
            raise ValueError(
                "record-frequency resident plan exceeds writable VR capacity"
            )
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

_Static_assert(FREQ_RECORDS == 32768, "one record per APUv1 lane required");
_Static_assert(FREQ_CHUNKS <= 10, "resident-chunk plan supports ten chunks");

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

    uint16_t query = 0;
    for (; query + 1 < FREQ_QUERIES; query += 2) {{
        for (uint16_t chunk = 0; chunk < FREQ_CHUNKS; ++chunk) {{
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
        for (uint16_t chunk = 0; chunk < FREQ_CHUNKS; ++chunk) {{
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

    for (uint16_t index = 0; index < FREQ_QUERIES; ++index)
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


__all__ = [
    "APUv1RecordFrequencyAnalysis",
    "APUv1RecordFrequencyLowering",
    "MAX_RESIDENT_CHUNKS",
    "analyze_apu_v1_record_frequency",
]
