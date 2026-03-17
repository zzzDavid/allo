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
"""

from math import log2
from typing import Tuple

import allo
from allo.ir.types import int8, int32, UInt, Stream
import allo.dataflow as df
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


def get_feather_full_matrix_top_kstreaming(M, K, N, AW, AH, Ty, num_inst):
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
    5. output_accum(1): Column remap + tile accumulation into output matrix

    Each tile covers exactly Kt = (AW // Gr) * AH K-elements (one pass).
    K-decomposition is handled at the ISA level, not inside the PE.

    Args:
        M, K, N: Matrix dimensions
        AW: Array width (must be power of 2)
        AH: Array height
        Ty: Input data type (e.g., int8)
        num_inst: Total instruction count (3 layout + tile mappings)
    """
    TyOut = int32
    LOG2_AW = int(log2(AW))
    LOG2_AH = int(log2(AH))
    P0, P1 = compute_birrd_params(AW)
    num_tiles = num_inst - 3

    num_accum_params = 2 + num_tiles  # quant_scale, quant_zp, sr[0..num_tiles-1]

    @df.region()
    def full_matrix_top(
        A_pe: int32[M, K],
        B_pe: int32[K, N],
        inst_pe: int32[num_inst, 13],
        birrd_inst: int8[num_tiles, P0, P1],
        output_col_map: int32[num_tiles, AW],
        output_num_m: int32[num_tiles],
        output_n_base: int32[num_tiles, AW],
        accum_m_start: int32[num_tiles],
        accum_n_start: int32[num_tiles],
        accum_params: int32[num_accum_params],
        C: int32[M, N],
    ):
        """Unified PE array FEATHER+ dataflow region.

        dram_loader is the sole reader of A_pe, B_pe, inst_pe.
        output_accum reads quant/sr params from accum_params DRAM array.
        Each DRAM buffer has exactly one reader kernel (HLS compliant).
        """

        # dram_loader → pe_array streams (row 0..AH-1 compute PEs)
        pe_a_in: Stream[int32, AH][AH, AW]
        pe_w_in: Stream[int32, AH][AH, AW]

        # pe_array internal streams: rows 0..AH-1 → row AH (gather)
        pe_out: Stream[int32, num_tiles][AH, AW]

        # BIRRD streams
        connection: Stream[TyOut, AH][P0 + 1, P1 * 2]
        inst_input: Stream[int8, num_tiles][P0, P1]

        @df.kernel(mapping=[1], args=[A_pe, B_pe, inst_pe])
        def dram_loader(
            local_A: int32[M, K],
            local_B: int32[K, N],
            local_inst: int32[num_inst, 13],
        ):
            """Sole DRAM reader. Loads A/B/inst into on-chip buffers, streams to PEs.

            Per-column streaming buffer (iacts_buf) and stationary buffer (weight_buf)
            are loaded per tile, then streamed to all AH×AW PEs via meta_for.
            """
            iacts_zp: int32 = local_inst[0, 6]
            weights_zp: int32 = local_inst[1, 6]

            for tile in range(num_tiles):
                inst_idx: int32 = tile + 3
                Gr: int32 = local_inst[inst_idx, 3]
                Gc: int32 = local_inst[inst_idx, 4]
                sr: int32 = local_inst[inst_idx, 5]
                sc: int32 = local_inst[inst_idx, 6]
                m_start: int32 = local_inst[inst_idx, 7]
                n_start: int32 = local_inst[inst_idx, 9]
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

                # For each column, load buffers and stream to PEs
                with allo.meta_for(AW) as nj:
                    # --- Streaming buffer: load activations ---
                    iacts_buf: int32[AH]
                    m_idx: int32 = m_start + (nj & mask_Gr)
                    for nk in range(AH):
                        k_idx: int32 = k_start_tile + nk + (nj >> log2_Gr) * AH
                        iacts_buf[nk] = local_A[m_idx, k_idx] - iacts_zp

                    # --- Stationary buffer: load weights ---
                    weight_buf: int32[AH, AH]
                    for nk in range(AH):
                        k_idx_w: int32 = k_start_tile + nk + (nj >> log2_Gr) * AH
                        for pe_row_rt in range(AH):
                            wn_idx: int32 = n_start + sr * pe_row_rt + sc * (nj & mask_Gc)
                            weight_buf[pe_row_rt, nk] = local_B[k_idx_w, wn_idx] - weights_zp

                    # --- Stream from on-chip buffers to PEs ---
                    for nk in range(AH):
                        with allo.meta_for(AH) as pe_row:
                            pe_a_in[pe_row, nj].put(iacts_buf[nk])
                            pe_w_in[pe_row, nj].put(weight_buf[pe_row, nk])

        @df.kernel(mapping=[AH + 1, AW])
        def pe_array():
            """Compute PEs (rows 0..AH-1) + gather (row AH). Stream-only, no DRAM args.

            Rows 0..AH-1 (AH*AW instances): Simple MAC units. Read from
                pe_a_in/pe_w_in, accumulate per tile, output to pe_out.
            Row AH (AW instances): Collects PE results from its column,
                sends to BIRRD input connections.
            """
            ni, nj = df.get_pid()

            with allo.meta_if(ni == AH):
                # === ROW AH: GATHER (AW parallel instances) ===
                for _tile in range(num_tiles):
                    buf: int32[AH]
                    with allo.meta_for(AH) as pe_row:
                        buf[pe_row] = pe_out[pe_row, nj].get()
                    for row in range(AH):
                        connection[0, nj].put(buf[row])

            with allo.meta_else():
                # === ROWS 0..AH-1: COMPUTE PEs (AH*AW instances) ===
                for tile in range(num_tiles):
                    tile_accum: int32 = 0
                    for nk in range(AH):
                        a_val: int32 = pe_a_in[ni, nj].get()
                        w_val: int32 = pe_w_in[ni, nj].get()
                        tile_accum += a_val * w_val
                    pe_out[ni, nj].put(tile_accum)

        @df.kernel(mapping=[1], args=[birrd_inst])
        def inst_rw(local_birrd_inst: int8[num_tiles, P0, P1]):
            """Distribute per-tile BIRRD switch instructions."""
            for tile in range(num_tiles):
                with allo.meta_for(P0) as i:
                    with allo.meta_for(P1) as j:
                        inst_input[i, j].put(local_birrd_inst[tile, i, j])

        @df.kernel(mapping=[P0, P1])
        def BIRRD():
            """BIRRD butterfly reduction/reorder network (int32 values)."""
            i, j = df.get_pid()

            for _tile in range(num_tiles):
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
            local_accum_m_start: int32[num_tiles],
            local_accum_n_start: int32[num_tiles],
            local_accum_params: int32[num_accum_params],
            local_C: int32[M, N],
        ):
            """Accumulate BIRRD output into C[M,N] with col remapping and optional post-quantization.

            Reads quant params and per-tile sr directly from accum_params DRAM
            array: [quant_scale, quant_zp, sr[0], sr[1], ...].

            N-offset uses generalized MINISA mapping:
                n_off = sr * on + n_base[tile, col]
            where n_base encodes sc * (original_pe_col & mask_Gc) per BIRRD output column.
            When sr=0, only first temporal row is used (others are duplicates).

            Post-quantization (when quant_scale != 0) matches RTL quant_post.v:
                result = (sign_extend_64(data) * scale + zero_extend_64(zp))[7:0]
            Implemented as: (accum * quant_scale + quant_zp) & 255
            """
            accum: int32[M, N]
            for _ai in range(M):
                for _aj in range(N):
                    accum[_ai, _aj] = 0

            # Read post-quantization params from DRAM array
            quant_scale: int32 = local_accum_params[0]
            quant_zp: int32 = local_accum_params[1]

            for tile in range(num_tiles):
                m_start: int32 = local_accum_m_start[tile]
                n_start: int32 = local_accum_n_start[tile]
                num_m: int32 = local_output_num_m[tile]
                sr_val: int32 = local_accum_params[2 + tile]

                tile_out: int32[AH, AW]
                for d in range(AH):
                    with allo.meta_for(AW) as aw_i:
                        tile_out[d, aw_i] = connection[P0, aw_i].get()

                for col in range(AW):
                    m_pos: int32 = local_output_col_map[tile, col]
                    n_base_col: int32 = local_output_n_base[tile, col]
                    if m_pos < num_m:
                        for on in range(AH):
                            # When sr=0, all temporal rows are identical;
                            # only accumulate first row to avoid AH-fold duplication
                            skip: int32 = 0
                            if sr_val == 0:
                                if on > 0:
                                    skip = 1
                            if skip == 0:
                                n_off: int32 = sr_val * on + n_base_col
                                accum[m_start + m_pos, n_start + n_off] = (
                                    accum[m_start + m_pos, n_start + n_off] + tile_out[on, col]
                                )

            for _wi in range(M):
                for _wj in range(N):
                    val: int32 = accum[_wi, _wj]
                    if quant_scale != 0:
                        val = (val * quant_scale + quant_zp) & 255
                    local_C[_wi, _wj] = val

    return full_matrix_top


class FeatherKStreamingModule:
    """Wrapper for FEATHER+ with clean (A, B, instructions, C) interface.

    Supports all Gr values per tile. Precomputes BIRRD tables for all
    OVN orders at build time. At call time, selects per-tile BIRRD
    configuration based on Gr: reduction (Gr < AW) or pass-through (Gr == AW).
    """

    def __init__(self, allo_mod, AW):
        from minisa.lowering import (
            lower_ovn_layout, compute_col_to_m_map,
            compute_output_col_map, _simulate_birrd_passthrough_perm,
            generate_birrd_instructions, _simulate_birrd_output_col_map_general,
        )
        from minisa.isa import SetOVNLayout

        self._mod = allo_mod
        self._AW = AW
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

    def __call__(self, A, B, instructions, C):
        ovn_order = int(instructions[2, 1])
        AW = self._AW
        P0, P1 = self._P0, self._P1
        num_tiles = len(instructions) - 3

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

        m_start_per_tile = np.array(
            [int(instructions[3 + t, 7]) for t in range(num_tiles)],
            dtype=np.int32,
        )
        n_start_per_tile = np.array(
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
        # pe_array args (A_pe, B_pe, inst_pe), inst_rw args, output_accum args
        # (BIRRD has no DRAM args — streams only)
        self._mod(
            A.astype(np.int32),         # pe_array: A_pe (int32 for spatial PE)
            B.astype(np.int32),         # pe_array: B_pe (int32 for spatial PE)
            instructions,               # pe_array: inst_pe
            birrd_per_tile,             # inst_rw
            col_map_per_tile,           # output_accum
            num_m_per_tile,
            n_base_per_tile,
            m_start_per_tile,
            n_start_per_tile,
            accum_params,               # output_accum: quant + sr params
            C,
        )


def build_feather_kstreaming_simulator(M, K, N, AW, AH, Ty, num_inst):
    """Build FEATHER+ dataflow for simulation."""
    top = get_feather_full_matrix_top_kstreaming(
        int(M), int(K), int(N), int(AW), int(AH), Ty, int(num_inst),
    )
    allo_mod = df.build(top, target="simulator")
    return FeatherKStreamingModule(allo_mod, AW)


def schedule_feather_hls(s, K, AH, AW=None):
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
    # Partition C along N (dim=2) for parallel output writes
    s.partition("full_matrix_top:C", dim=2, factor=AH)
    # Partition A_pe and B_pe to enable parallel reads in pe_array's
    # AW row-0 instances — AW parallel column reads of A and B
    if AW is not None:
        # A_pe[M,K]: partition K dimension for parallel reads across nj groups
        s.partition("full_matrix_top:A_pe", dim=2, factor=K)
        # B_pe[K,N]: partition N dimension for parallel reads across nj columns
        s.partition("full_matrix_top:B_pe", dim=2, factor=AW * AH)


def build_feather_kstreaming_hls(M, K, N, AW, AH, Ty, num_inst,
                                  mode="csim", project=None):
    """Build FEATHER+ dataflow for Vitis HLS."""
    if project is None:
        project = "feather_kstreaming.prj"
    top = get_feather_full_matrix_top_kstreaming(
        int(M), int(K), int(N), int(AW), int(AH), Ty, int(num_inst),
    )
    s = df.customize(top)
    schedule_feather_hls(s, int(K), int(AH))
    allo_mod = s.build(target="vitis_hls", mode=mode, project=project)
    return FeatherKStreamingModule(allo_mod, AW)


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

        mod = build_feather_kstreaming_simulator(
            M_i, K_i, N_i, AW, AH, int8, len(instructions),
        )
        C = np.zeros((M_i, N_i), dtype=np.int32)
        mod(current_input, B, instructions, C)
        outputs.append(C)

        # Convert quantized output to int8 for next layer's input
        if i < len(layer_weights) - 1:
            current_input = C.astype(np.uint8).view(np.int8)

    return outputs
