# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""FEATHER+ accelerator with MINISA support using Allo dataflow.

This module provides the full-matrix Allo dataflow implementation for FEATHER+.
A single Allo invocation handles complete input matrices and the full MINISA
instruction list, performing tiling, instruction decode, crossbar reordering,
NEST computation, BIRRD reduction, and output accumulation on-chip.

Supports all Gr values (1, 2, ..., AW) per tile via power-of-2 bit operations,
enabling full dataflow switching (output/weight/input stationary and mixed).

Architecture (5 pipelined dataflow kernels):
1. dram_loader: Sole DRAM reader — loads on-chip buffers, streams to pe_array
2. pe_array[AH+1,AW]: Compute PEs (rows 0..AH-1) + gather (row AH)
3. inst_rw: Distribute BIRRD switch instructions
4. BIRRD[P0,P1]: Butterfly reduction/reorder (per-tile configuration)
5. output_accum: Column remap + tile accumulation into output matrix

The MINISA instructions (SetIVNLayout, SetWVNLayout, SetOVNLayout, SetMapping)
are encoded as an int32 array and decoded on-chip by the dataflow kernels.

Temporal N-iteration and M-batching (n_inner > 1):
When n_inner > 1, each ISA tile contains multiple inner sub-operations that
share the same K-range and BIRRD configuration but have different (m_start,
n_start) offsets. This matches the RTL's VN buffer temporal iteration, where
each ExecuteMapping covers all M-batches and N-sub-tiles internally. The
per-sub-operation m_start/n_start are stored in separate DRAM lookup tables.
"""

from math import log2
from typing import Tuple

import allo
from allo.ir.types import int8, int32, UInt, Stream
import allo.dataflow as df
from allo.customize import Partition
import numpy as np

# BIRRD switch operations
PS = 0  # Pass: outputs = inputs
AR = 1  # Add Right: left_out = left, right_out = left + right
AL = 2  # Add Left: left_out = left + right, right_out = right
SW = 3  # Swap: left_out = right, right_out = left


def reverse_bits(data: int, bit_range: int) -> int:
    """Reverse the lower bit_range bits of data.

    Used for butterfly network routing in BIRRD.
    """
    mask = (1 << bit_range) - 1
    reversed_bits = 0
    for i in range(0, bit_range):
        if data & (1 << i):
            reversed_bits |= 1 << (bit_range - 1 - i)
    return (data & ~mask) | reversed_bits


def compute_birrd_params(AW: int) -> Tuple[int, int]:
    """Compute BIRRD network parameters from array width.

    Returns:
        P0: Number of stages in BIRRD network
        P1: Number of switches per stage
    """
    LOG2_AW = int(log2(AW))
    P0 = 2 * LOG2_AW if AW > 4 else 2 * LOG2_AW - 1
    P1 = AW // 2
    return P0, P1


def get_feather_full_matrix_top(M, K, N, AW, AH, Ty, num_inst,
                                            n_inner=1, k_passes=1,
                                            Nt_local=None):
    """Create FEATHER+ dataflow region with unified PE array.

    Uses dram_loader(mapping=[1]) as sole DRAM reader with on-chip streaming
    and stationary buffers, plus pe_array(mapping=[AH+1, AW]): rows 0..AH-1 =
    compute PEs (AH*AW instances), row AH = gather (AW instances). Parallelism
    is structural (guaranteed by construction).

    Architecture (5 dataflow kernels):
    1. dram_loader(1): Sole DRAM reader with on-chip buffers, streams to PEs
    2. pe_array[AH+1,AW]: Compute PEs + per-column gather
    3. inst_rw(1): Distribute BIRRD switch instructions
    4. BIRRD[P0,P1]: Butterfly reduction/reorder (per-tile configuration)
    5. output_accum(1): Column remap + block accumulation into output matrix

    Each tile covers exactly Kt = (AW // Gr) * AH K-elements (one pass).
    K-decomposition is handled at the ISA level, not inside the PE.

    When n_inner > 1, each tile contains n_inner sub-operations with different
    (m_start, n_start) offsets, matching the RTL's temporal VN iteration.

    Args:
        M, K, N: Matrix dimensions
        AW: Array width (must be power of 2)
        AH: Array height
        Ty: Input data type (e.g., int8)
        num_inst: Total instruction count (3 layout + tile mappings)
        n_inner: Inner iterations per tile (default 1). When > 1, each tile
            processes n_inner sub-operations with different m_start/n_start.
        k_passes: Number of K-decomposition tiles per output block (default 1).
            Consecutive k_passes tiles share the same (m_start, n_start) and
            are accumulated in a local register buffer before flushing to C.
        Nt_local: N-dimension of local accumulation buffer (default AH).
            Must be >= max N-coverage per tile. For sc>0 mappings (e.g.,
            Figure 7 with sc=4, Gc=2), Nt_local = sr*(AH-1)+sc*(Gc-1)+1.
    """
    if Nt_local is None:
        Nt_local = AH
    TyOut = int32
    LOG2_AW = int(log2(AW))
    LOG2_AH = int(log2(AH))
    P0, P1 = compute_birrd_params(AW)
    num_tiles = num_inst - 3
    total_ops = num_tiles * n_inner
    num_blocks = num_tiles // k_passes

    num_accum_params = 2 + num_tiles  # quant_scale, quant_zp, sr[0..num_tiles-1]

    @df.region()
    def full_matrix_top(
        A_pe: int32[M, K],
        B_pe: int32[K, N],
        inst_pe: int32[num_inst, 13],
        loader_m_start: int32[total_ops],
        loader_n_start: int32[total_ops],
        birrd_inst: int8[num_tiles, P0, P1],
        output_col_map: int32[num_tiles, AW],
        output_num_m: int32[num_tiles],
        output_n_base: int32[num_tiles, AW],
        accum_m_start: int32[total_ops],
        accum_n_start: int32[total_ops],
        accum_params: int32[num_accum_params],
        C: int32[M, N],
    ):
        """Unified PE array FEATHER+ dataflow region.

        dram_loader is the sole reader of A_pe, B_pe, inst_pe, loader_m_start,
        loader_n_start. output_accum reads quant/sr params from accum_params,
        per-op m_start/n_start from accum arrays. Each DRAM buffer has exactly
        one reader kernel (HLS single-reader compliance).
        """

        # dram_loader -> pe_array streams (row 0..AH-1 compute PEs)
        pe_a_in: Stream[int32, AH][AH, AW]
        pe_w_in: Stream[int32, AH][AH, AW]

        # pe_array internal streams: rows 0..AH-1 -> row AH (gather)
        pe_out: Stream[int32, total_ops][AH, AW]

        # BIRRD streams
        connection: Stream[TyOut, AH][P0 + 1, P1 * 2]
        inst_input: Stream[int8, total_ops][P0, P1]

        @df.kernel(mapping=[1], args=[A_pe, B_pe, inst_pe, loader_m_start, loader_n_start])
        def dram_loader(
            local_A: int32[M, K],
            local_B: int32[K, N],
            local_inst: int32[num_inst, 13],
            local_loader_m_start: int32[total_ops],
            local_loader_n_start: int32[total_ops],
        ):
            """Sole DRAM reader. Streams A/B directly to PEs (no on-chip buffers).

            Single merged nk loop reads A and B from DRAM and puts directly to
            PE streams. When n_inner > 1, each tile has multiple inner iterations
            with different (m_start, n_start) from DRAM lookup tables.
            """
            iacts_zp: int32 = local_inst[0, 6]
            weights_zp: int32 = local_inst[1, 6]

            for tile in range(num_tiles):
                inst_idx: int32 = tile + 3
                Gr: int32 = local_inst[inst_idx, 3]
                Gc: int32 = local_inst[inst_idx, 4]
                sr: int32 = local_inst[inst_idx, 5]
                sc: int32 = local_inst[inst_idx, 6]
                k_start_tile: int32 = local_inst[inst_idx, 11]

                # Compute log2_Gr via comparison chain (Gr is power of 2)
                log2_Gr: int32 = 0
                if Gr >= 2:
                    log2_Gr = 1
                if Gr >= 4:
                    log2_Gr = 2
                if Gr >= 8:
                    log2_Gr = 3
                if Gr >= 16:
                    log2_Gr = 4
                mask_Gr: int32 = Gr - 1
                mask_Gc: int32 = Gc - 1

                for inner in range(n_inner):
                    op_idx: int32 = tile * n_inner + inner
                    m_start: int32 = local_loader_m_start[op_idx]
                    n_start: int32 = local_loader_n_start[op_idx]

                    # Stream A/B directly to PEs (no on-chip buffers)
                    for nk in range(AH):
                        with allo.meta_for(AW) as nj:
                            m_idx: int32 = m_start + (nj & mask_Gr)
                            k_idx: int32 = k_start_tile + nk + (nj >> log2_Gr) * AH
                            a_val: int32 = local_A[m_idx, k_idx] - iacts_zp
                            with allo.meta_for(AH) as pe_row:
                                wn_idx: int32 = n_start + sr * pe_row + sc * (nj & mask_Gc)
                                w_val: int32 = local_B[k_idx, wn_idx] - weights_zp
                                pe_a_in[pe_row, nj].put(a_val)
                                pe_w_in[pe_row, nj].put(w_val)

        @df.kernel(mapping=[AH + 1, AW])
        def pe_array():
            """Compute PEs (rows 0..AH-1) + gather (row AH). Stream-only, no DRAM args.

            Rows 0..AH-1 (AH*AW instances): Simple MAC units. Read from
                pe_a_in/pe_w_in, accumulate per sub-operation, output to pe_out.
            Row AH (AW instances): Collects PE results from its column,
                sends to BIRRD input connections.

            total_ops = num_tiles * n_inner — the PE doesn't distinguish
            between tiles and inner iterations.
            """
            ni, nj = df.get_pid()

            with allo.meta_if(ni == AH):
                # === ROW AH: GATHER (AW parallel instances) ===
                for _op in range(total_ops):
                    buf: int32[AH]
                    with allo.meta_for(AH) as pe_row:
                        buf[pe_row] = pe_out[pe_row, nj].get()
                    for row in range(AH):
                        connection[0, nj].put(buf[row])

            with allo.meta_else():
                # === ROWS 0..AH-1: COMPUTE PEs (AH*AW instances) ===
                for _op in range(total_ops):
                    tile_accum: int32 = 0
                    for nk in range(AH):
                        a_val: int32 = pe_a_in[ni, nj].get()
                        w_val: int32 = pe_w_in[ni, nj].get()
                        tile_accum += a_val * w_val
                    pe_out[ni, nj].put(tile_accum)

        @df.kernel(mapping=[1], args=[birrd_inst])
        def inst_rw(local_birrd_inst: int8[num_tiles, P0, P1]):
            """Distribute per-tile BIRRD switch instructions.

            Each tile's instruction is repeated n_inner times so that
            the BIRRD network receives one instruction per sub-operation.
            """
            for tile in range(num_tiles):
                for _rep in range(n_inner):
                    with allo.meta_for(P0) as i:
                        with allo.meta_for(P1) as j:
                            inst_input[i, j].put(local_birrd_inst[tile, i, j])

        @df.kernel(mapping=[P0, P1])
        def BIRRD():
            """BIRRD butterfly reduction/reorder network (int32 values)."""
            i, j = df.get_pid()

            for _op in range(total_ops):
                inst_val = inst_input[i, j].get()

                for _ in range(AH):
                    in_left: TyOut = connection[i, 2 * j].get()
                    in_right: TyOut = connection[i, 2 * j + 1].get()

                    out_left: TyOut = 0
                    out_right: TyOut = 0

                    if inst_val == 0:  # PS: pass
                        out_left = in_left
                        out_right = in_right
                    elif inst_val == 1:  # AR: add-right
                        out_left = in_left
                        out_right = in_left + in_right
                    elif inst_val == 2:  # AL: add-left
                        out_left = in_left + in_right
                        out_right = in_right
                    else:  # SW: swap
                        out_left = in_right
                        out_right = in_left

                    with allo.meta_if(i != P0 - 1):
                        connection[
                            i + 1,
                            reverse_bits(
                                2 * j,
                                2 if i == 0 else min(LOG2_AW, 2 + i, 2 * LOG2_AW - i),
                            ),
                        ].put(out_left)
                        connection[
                            i + 1,
                            reverse_bits(
                                2 * j + 1,
                                2 if i == 0 else min(LOG2_AW, 2 + i, 2 * LOG2_AW - i),
                            ),
                        ].put(out_right)
                    with allo.meta_else():
                        connection[P0, 2 * j].put(out_left)
                        connection[P0, 2 * j + 1].put(out_right)

        @df.kernel(mapping=[1], args=[output_col_map, output_num_m, output_n_base, accum_m_start, accum_n_start, accum_params, C])
        def output_accum(
            local_output_col_map: int32[num_tiles, AW],
            local_output_num_m: int32[num_tiles],
            local_output_n_base: int32[num_tiles, AW],
            local_accum_m_start: int32[total_ops],
            local_accum_n_start: int32[total_ops],
            local_accum_params: int32[num_accum_params],
            local_C: int32[M, N],
        ):
            """Accumulate BIRRD output into C[M,N] with block-level register accumulation.

            Uses a small local_acc[AW,AH] register buffer (fully partitioned) instead
            of a large accum[M,N] buffer. Consecutive k_passes tiles form a "block"
            targeting the same output region — their partial K-sums are accumulated
            in local_acc, then flushed to C once per block.

            This eliminates data-dependent scatter into a large array, replacing it
            with a 16-register buffer that HLS can fully partition without memory
            port contention.
            """
            quant_scale: int32 = local_accum_params[0]
            quant_zp: int32 = local_accum_params[1]

            # Phase 1: Communication — read all BIRRD outputs into buffer
            all_tile_out: int32[total_ops, AH, AW]
            for op in range(total_ops):
                for d in range(AH):
                    with allo.meta_for(AW) as aw_i:
                        all_tile_out[op, d, aw_i] = connection[P0, aw_i].get()

            # Phase 2: Block-level accumulation with local register buffer
            # Each block = k_passes consecutive tiles sharing same (m_start, n_start)
            local_acc: int32[AW, Nt_local]

            for block in range(num_blocks):
                # Zero local_acc
                for _i0 in range(AW):
                    for _j0 in range(Nt_local):
                        local_acc[_i0, _j0] = 0

                base_tile: int32 = block * k_passes
                num_m: int32 = local_output_num_m[base_tile]
                sr_val: int32 = local_accum_params[2 + base_tile]

                # Accumulate k_passes tiles into local_acc
                for k in range(k_passes):
                    tile: int32 = base_tile + k
                    for inner in range(n_inner):
                        op_idx: int32 = tile * n_inner + inner
                        for col in range(AW):
                            m_pos: int32 = local_output_col_map[tile, col]
                            n_base_col: int32 = local_output_n_base[tile, col]
                            col_mask: int32 = 0
                            m_safe: int32 = 0
                            if m_pos < num_m:
                                col_mask = 1
                                m_safe = m_pos
                            for on in range(AH):
                                sr_mask: int32 = 0
                                if sr_val != 0:
                                    sr_mask = 1
                                if on == 0:
                                    sr_mask = 1
                                n_off: int32 = sr_val * on + n_base_col
                                local_acc[m_safe, n_off] = (
                                    local_acc[m_safe, n_off]
                                    + all_tile_out[op_idx, on, col] * col_mask * sr_mask
                                )

                # Flush to C with additive accumulation (supports mixed-Gr tiles)
                m_start: int32 = local_accum_m_start[base_tile * n_inner]
                n_start: int32 = local_accum_n_start[base_tile * n_inner]
                for mi in range(AW):
                    for ni in range(Nt_local):
                        val: int32 = local_acc[mi, ni]
                        if quant_scale != 0:
                            val = (val * quant_scale + quant_zp) & 255
                        local_C[m_start + mi, n_start + ni] = (
                            local_C[m_start + mi, n_start + ni] + val
                        )

    return full_matrix_top


class FeatherModule:
    """Wrapper for FEATHER+ with clean (A, B, instructions, C) interface.

    Supports all Gr values per tile. Precomputes BIRRD tables for all
    OVN orders at build time. At call time, selects per-tile BIRRD
    configuration based on Gr: reduction (Gr < AW) or pass-through (Gr == AW).

    When n_inner > 1, the caller must provide inner_params dict with
    per-sub-operation m_starts and n_starts arrays.
    """

    def __init__(self, allo_mod, AW, n_inner=1):
        from minisa.lowering import (
            lower_ovn_layout, compute_col_to_m_map,
            compute_output_col_map, _simulate_birrd_passthrough_perm,
            generate_birrd_instructions, _simulate_birrd_output_col_map_general,
        )
        from minisa.isa import SetOVNLayout

        self._mod = allo_mod
        self._AW = AW
        self._n_inner = n_inner
        P0, P1 = compute_birrd_params(AW)
        self._P0 = P0
        self._P1 = P1
        log2_aw = int(log2(AW))
        valid_grs = set()
        for i in range(log2_aw + 1):
            valid_grs.add(1 << i)  # 1, 2, 4, ..., AW

        # Precompute BIRRD tables for all (order, Gr) combinations
        # - Gr=AW: all-PS passthrough
        # - Gr=AW//2: hand-coded order-dependent 2-way tables
        # - Gr<AW//2: algorithmically generated multi-way tables
        self._birrd_tables = {}
        self._multiway_birrd = {}  # generated tables keyed by Gr
        for order in range(6):
            ovn = SetOVNLayout(order=order, PL0=AW, PL1=1, QL0=AW, QL1=1)
            self._birrd_tables[order] = lower_ovn_layout(ovn, AW, AW)
        for gr in valid_grs:
            if gr < AW // 2:
                self._multiway_birrd[gr] = generate_birrd_instructions(AW, gr)

        # Precompute col->M maps for all (order, Gr) combinations
        self._col_to_m_maps = {}
        for order in range(6):
            for gr in valid_grs:
                self._col_to_m_maps[(order, gr)] = compute_col_to_m_map(AW, order, gr)

        # Precompute passthrough permutation (for n_base computation)
        self._passthrough_perm = _simulate_birrd_passthrough_perm(AW)
        # Precompute pair->col maps for 2-way reduction n_base
        self._pair_to_col = {}
        for order in range(6):
            self._pair_to_col[order] = compute_output_col_map(AW, order)
        # Precompute m->col maps for multi-way reduction n_base
        self._multiway_m_to_col = {}
        for gr in valid_grs:
            if gr < AW // 2:
                inst = self._multiway_birrd[gr]
                self._multiway_m_to_col[gr] = _simulate_birrd_output_col_map_general(
                    inst, AW, gr
                )

    def __call__(self, A, B, instructions, C, inner_params=None):
        """Execute FEATHER+ dataflow.

        Args:
            A: Input activations [M, K] (int8)
            B: Weight matrix [K, N] (int8)
            instructions: Encoded MINISA program [num_inst, 13] (int32)
            C: Output matrix [M, N] (int32), modified in-place
            inner_params: Optional dict with per-sub-operation arrays:
                'm_starts': int32[total_ops] — m_start per sub-operation
                'n_starts': int32[total_ops] — n_start per sub-operation
                Required when n_inner > 1.
        """
        ovn_order = int(instructions[2, 1])
        AW = self._AW
        P0, P1 = self._P0, self._P1
        n_inner = self._n_inner
        num_tiles = len(instructions) - 3
        total_ops = num_tiles * n_inner

        # Build per-tile BIRRD instructions, output mappings, and N-base offsets
        birrd_per_tile = np.zeros((num_tiles, P0, P1), dtype=np.int8)
        col_map_per_tile = np.zeros((num_tiles, AW), dtype=np.int32)
        num_m_per_tile = np.zeros(num_tiles, dtype=np.int32)
        n_base_per_tile = np.zeros((num_tiles, AW), dtype=np.int32)

        for t in range(num_tiles):
            Gr = int(instructions[3 + t, 3])
            Gc = int(instructions[3 + t, 4])
            sc = int(instructions[3 + t, 6])
            mask_Gc = Gc - 1

            if Gr == AW:
                # Pass-through: each column is a unique M row
                col_map_per_tile[t] = self._col_to_m_maps[(ovn_order, AW)]
                num_m_per_tile[t] = AW
                for col in range(AW):
                    orig_pe = int(self._passthrough_perm[col])
                    n_base_per_tile[t, col] = sc * (orig_pe & mask_Gc)
            elif Gr == AW // 2:
                # 2-way reduction: hand-coded order-dependent BIRRD tables
                birrd_per_tile[t] = self._birrd_tables[ovn_order]
                col_map_per_tile[t] = self._col_to_m_maps[(ovn_order, Gr)]
                num_m_per_tile[t] = Gr
                pair_to_col = self._pair_to_col[ovn_order]
                for pair_idx in range(AW // 2):
                    col = int(pair_to_col[pair_idx])
                    n_base_per_tile[t, col] = sc * (pair_idx & mask_Gc)
            else:
                # Multi-way reduction (Gr < AW//2): generated BIRRD tables
                birrd_per_tile[t] = self._multiway_birrd[Gr]
                col_map_per_tile[t] = self._col_to_m_maps[(ovn_order, Gr)]
                num_m_per_tile[t] = Gr
                m_to_col = self._multiway_m_to_col[Gr]
                for m in range(Gr):
                    col = int(m_to_col[m])
                    n_base_per_tile[t, col] = sc * (m & mask_Gc)

        # Per-sub-operation m_start/n_start arrays
        if inner_params is not None:
            m_start_per_op = inner_params['m_starts'].astype(np.int32)
            n_start_per_op = inner_params['n_starts'].astype(np.int32)
        else:
            # n_inner=1: extract from instructions (backward compat)
            m_start_per_op = np.array(
                [int(instructions[3 + t, 7]) for t in range(num_tiles)],
                dtype=np.int32,
            )
            n_start_per_op = np.array(
                [int(instructions[3 + t, 9]) for t in range(num_tiles)],
                dtype=np.int32,
            )

        # Prepare accum_params: [quant_scale, quant_zp, sr[0], sr[1], ...]
        accum_params = np.zeros(2 + num_tiles, dtype=np.int32)
        accum_params[0] = int(instructions[2, 6])  # quant_scale
        accum_params[1] = int(instructions[2, 7])  # quant_zp
        for t in range(num_tiles):
            accum_params[2 + t] = int(instructions[3 + t, 5])  # sr

        # Call order matches Allo's kernel-grouped reordering:
        # dram_loader args (A_pe, B_pe, inst_pe, loader_m_start, loader_n_start),
        # inst_rw args (birrd_inst),
        # output_accum args (output_col_map, ..., C)
        # (BIRRD has no DRAM args — streams only)
        self._mod(
            A.astype(np.int32),         # dram_loader: A_pe
            B.astype(np.int32),         # dram_loader: B_pe
            instructions,               # dram_loader: inst_pe
            m_start_per_op.copy(),      # dram_loader: loader_m_start
            n_start_per_op.copy(),      # dram_loader: loader_n_start
            birrd_per_tile,             # inst_rw: birrd_inst
            col_map_per_tile,           # output_accum: output_col_map
            num_m_per_tile,             # output_accum: output_num_m
            n_base_per_tile,            # output_accum: output_n_base
            m_start_per_op,             # output_accum: accum_m_start
            n_start_per_op,             # output_accum: accum_n_start
            accum_params,               # output_accum: accum_params
            C,
        )


def build_feather_simulator(M, K, N, AW, AH, Ty, num_inst,
                                        n_inner=1, k_passes=1, Nt_local=None):
    """Build FEATHER+ dataflow for simulation."""
    top = get_feather_full_matrix_top(
        int(M), int(K), int(N), int(AW), int(AH), Ty, int(num_inst),
        int(n_inner), int(k_passes), Nt_local=Nt_local,
    )
    allo_mod = df.build(top, target="simulator")
    return FeatherModule(allo_mod, AW, n_inner)


def schedule_feather_hls(s, K, AH, AW):
    """Apply HLS scheduling optimizations to FEATHER+ dataflow.

    With unified PE array kernel (mapping=[AH+2, AW]), parallelism is structural —
    no pipeline/unroll scheduling needed. Array partitions enable parallel
    reads in pe_array's AW row-0 data loader instances.

    Args:
        s: Allo schedule object from df.customize()
        K: K dimension (for array partitioning)
        AH: Array height (for C partitioning)
        AW: Array width (for A/B partitioning, optional)
    """
    # Pipeline the dram_loader tile loop for II=1
    s.pipeline("dram_loader_0:tile")
    # Partition C along N (dim=2) for parallel output writes
    s.partition("full_matrix_top:C", dim=2, factor=AH)
    # Partition A and B inside dram_loader for AW parallel reads
    s.partition("dram_loader_0:local_A", dim=2, factor=K)
    s.partition("dram_loader_0:local_B", dim=0)
    # Fully partition local_acc[AW,AH] — only 16 registers, eliminates all
    # memory port contention for data-dependent indexing
    s.partition("output_accum_0:local_acc", dim=1, partition_type=Partition.Complete)
    s.partition("output_accum_0:local_acc", dim=2, partition_type=Partition.Complete)


def build_feather_hls(M, K, N, AW, AH, Ty, num_inst,
                                  mode="csim", project=None, n_inner=1,
                                  k_passes=1, Nt_local=None):
    """Build FEATHER+ dataflow for Vitis HLS."""
    if project is None:
        project = "feather.prj"
    top = get_feather_full_matrix_top(
        int(M), int(K), int(N), int(AW), int(AH), Ty, int(num_inst),
        int(n_inner), int(k_passes), Nt_local=Nt_local,
    )
    s = df.customize(top)
    schedule_feather_hls(s, int(K), int(AH), int(AW))
    allo_mod = s.build(target="vitis_hls", mode=mode, project=project)
    return FeatherModule(allo_mod, AW, n_inner)


def run_sequential_gemm_layers(A_input, layer_weights, layer_program_kwargs,
                                AW, AH):
    """Execute multiple GEMM layers sequentially with int8 intermediates.

    Each non-final layer must enable post-quantization (quant_scale != 0)
    so that its int32 accumulator is quantized to uint8 via
    (accum * scale + zp) & 255. The uint8 output is reinterpreted as
    signed int8 for the next layer's input — matching the RTL auto-quant
    pipeline (OB -> quant_post -> StaB PONG write).

    Args:
        A_input: Initial input matrix (int8, shape [M, K0])
        layer_weights: List of weight matrices (int8), one per layer
        layer_program_kwargs: List of dicts passed to create_gemm_program(),
            e.g. [dict(M=8, N=8, K=16, quant_scale=1, quant_zp=128), ...]
        AW, AH: Array dimensions

    Returns:
        List of output arrays (int32), one per layer.
        Non-final layers contain post-quantized uint8 values (0-255) in int32.
    """
    from minisa.isa import create_gemm_program, encode_program

    outputs = []
    current_input = A_input

    for i, (B, kwargs) in enumerate(zip(layer_weights, layer_program_kwargs)):
        program = create_gemm_program(AH=AH, AW=AW, **kwargs)
        instructions = encode_program(program)

        M_i = current_input.shape[0]
        K_i = current_input.shape[1]
        N_i = B.shape[1]

        mod = build_feather_simulator(
            M_i, K_i, N_i, AW, AH, int8, len(instructions),
        )
        C = np.zeros((M_i, N_i), dtype=np.int32)
        mod(current_input, B, instructions, C)
        outputs.append(C)

        # Convert quantized output to int8 for next layer's input
        if i < len(layer_weights) - 1:
            current_input = C.astype(np.uint8).view(np.int8)

    return outputs
