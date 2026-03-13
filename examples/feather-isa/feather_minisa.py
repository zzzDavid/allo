# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""FEATHER+ accelerator with MINISA support using Allo dataflow.

This module provides the full-matrix Allo dataflow implementation for FEATHER+.
A single Allo invocation handles complete input matrices and the full MINISA
instruction list, performing tiling, instruction decode, crossbar reordering,
NEST computation, BIRRD reduction, and output accumulation on-chip.

Supports all Gr values (1, 2, ..., AW) per tile via power-of-2 bit operations,
enabling full dataflow switching (output/weight/input stationary and mixed).

Architecture (7 pipelined dataflow kernels):
1. crossbar_load: Read A/B with parametric Gr crossbar (bit ops), pack into
   UInt(128), stream to nest_compute
2. nest_compute: Unpack, run NEST MAC, accumulate across K-passes
3. bus: Unpack packed NEST output to BIRRD connection streams
4. inst_rw: Distribute BIRRD switch instructions
5. BIRRD[P0,P1]: Butterfly reduction/reorder (per-tile configuration)
6. output_accum: Column remap + tile accumulation into output matrix

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


def get_feather_full_matrix_top_kstreaming(M, K, N, AW, AH, Ty, num_inst,
                                            max_k_passes, Kt_per_pass=None):
    """Create FEATHER+ dataflow region with split crossbar/NEST and flexible Gr.

    Supports all power-of-2 Gr values per tile via bit operations in the
    crossbar index arithmetic. No runtime dividers — uses (ic_j & (Gr-1))
    for modulo and (ic_j >> log2_Gr) for division.

    Supports mixed Kt_per_pass across tiles: each tile computes its own
    actual_passes at runtime from its Gr and K-range. Padding passes
    (beyond actual_passes) stream zeros — harmless to NEST (0*x=0).

    Architecture (7 dataflow kernels):
    1. crossbar_load: Read A/B with parametric Gr crossbar (bit ops),
       pack into UInt(128), stream to nest_compute
    2. nest_compute: Unpack, run NEST MAC, accumulate across K-passes
    3. bus: Unpack packed NEST output to BIRRD connections
    4. inst_rw: Distribute BIRRD switch instructions
    5. BIRRD[P0,P1]: Butterfly reduction/reorder (per-tile configuration)
    6. output_accum: Column remap + tile accumulation into C

    Data transfer protocol per K-pass:
      crossbar_load sends 1 iActs packet + AH weights packets via streams
      nest_compute receives and unpacks them, runs NEST MAC, accumulates

    Args:
        M, K, N: Matrix dimensions
        AW: Array width (must be power of 2)
        AH: Array height
        Ty: Input data type (e.g., int8)
        num_inst: Total instruction count (3 layout + tile mappings)
        max_k_passes: Max K-passes across all tiles (compile-time loop bound)
        Kt_per_pass: Ignored (kept for backward compatibility)
    """
    TyOut = int32
    TyPacked = UInt(TyOut.bits * AW)       # UInt(128) for packing 4 int32
    TyCrossbarPacked = UInt(AH * AW * Ty.bits)  # UInt(128) for packing 16 int8
    LOG2_AW = int(log2(AW))
    LOG2_AH = int(log2(AH))
    P0, P1 = compute_birrd_params(AW)
    num_tiles = num_inst - 3
    # Allow crossbar_load to run ~2 tiles ahead of nest_compute
    xbar_i_depth = max_k_passes * 2
    xbar_w_depth = max_k_passes * AH * 2

    @df.region()
    def full_matrix_top(
        A: Ty[M, K],
        B: Ty[K, N],
        instructions: int32[num_inst, 13],
        birrd_inst: int8[num_tiles, P0, P1],
        output_col_map: int32[num_tiles, AW],
        output_num_m: int32[num_tiles],
        output_n_base: int32[num_tiles, AW],
        accum_m_start: int32[num_tiles],
        accum_n_start: int32[num_tiles],
        C: int32[M, N],
    ):
        """K-streaming FEATHER+ dataflow region with split crossbar/NEST."""

        # Intermediate streams: crossbar_load → nest_compute
        iacts_stream: Stream[TyCrossbarPacked, xbar_i_depth]
        weights_stream: Stream[TyCrossbarPacked, xbar_w_depth]

        # Compute pipeline streams (unchanged from v1)
        nest_out: Stream[TyPacked, AH * 2]
        connection: Stream[TyOut, AH][P0 + 1, P1 * 2]
        inst_input: Stream[int8, num_tiles][P0, P1]

        @df.kernel(mapping=[1], args=[A, B, instructions])
        def crossbar_load(
            local_A: Ty[M, K],
            local_B: Ty[K, N],
            local_instructions: int32[num_inst, 13],
        ):
            """Load crossbar data and stream packed iActs/weights to nest_compute.

            Supports parametric Gr per tile via power-of-2 bit operations:
              ic_j % Gr  →  ic_j & (Gr - 1)    (AND with bitmask)
              ic_j // Gr →  ic_j >> log2_Gr     (right shift)

            Per-tile actual_passes computed at runtime; padding passes stream
            zeros (harmless to NEST: 0*x=0). Stream puts are always unconditional
            to maintain dataflow balance with nest_compute.

            IVN/WVN layout orders (instructions[0,1] and instructions[1,1]):
            These control VN buffer memory layout in the physical architecture.
            In our direct-indexing model (no VN buffer), the crossbar routing
            is fully determined by Gr/Gc/sr/sc from SetMapping. The index
            formula m_idx = m_start + (ic_j & mask_Gr), k_idx = k_start +
            ic_i + (ic_j >> log2_Gr) * AH produces correct GEMM for all 6
            IVN/WVN orders because it maps PE positions to matrix coordinates
            independently of the buffer memory layout.

            Per K-pass: 1 iActs packet (AH*AW int8 → UInt(128)) +
                        AH weights packets (AW*AH int8 each → UInt(128)).
            """
            for tile in range(num_tiles):
                inst_idx: int32 = tile + 3
                Gr: int32 = local_instructions[inst_idx, 3]
                Gc: int32 = local_instructions[inst_idx, 4]
                sr: int32 = local_instructions[inst_idx, 5]
                sc: int32 = local_instructions[inst_idx, 6]
                m_start: int32 = local_instructions[inst_idx, 7]
                n_start: int32 = local_instructions[inst_idx, 9]
                k_start_tile: int32 = local_instructions[inst_idx, 11]

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

                # Per-tile K-pass computation via shifts (no runtime dividers)
                k_end_tile: int32 = local_instructions[inst_idx, 12]
                k_range: int32 = k_end_tile - k_start_tile
                kt_per_pass: int32 = (AW * AH) >> log2_Gr
                log2_kt: int32 = (LOG2_AW + LOG2_AH) - log2_Gr
                actual_passes: int32 = k_range >> log2_kt

                for k_pass in range(max_k_passes):
                    k_start: int32 = k_start_tile + k_pass * kt_per_pass

                    # === Input crossbar (Gr-based bit ops) ===
                    packed_iacts: TyCrossbarPacked = 0
                    if k_pass < actual_passes:
                        for ic_i in range(AH):
                            for ic_j in range(AW):
                                m_idx: int32 = m_start + (ic_j & mask_Gr)
                                k_idx: int32 = k_start + ic_i + (ic_j >> log2_Gr) * AH
                                packed_iacts[
                                    (ic_i * AW + ic_j) * Ty.bits :
                                    (ic_i * AW + ic_j + 1) * Ty.bits
                                ] = local_A[m_idx, k_idx]
                    iacts_stream.put(packed_iacts)

                    # === Weight crossbar (Gr/sr/sc bit ops) ===
                    for wc_i in range(AH):
                        packed_w: TyCrossbarPacked = 0
                        if k_pass < actual_passes:
                            for wc_w in range(AW):
                                for wc_k in range(AH):
                                    wk_idx: int32 = k_start + wc_k + (wc_w >> log2_Gr) * AH
                                    wn_idx: int32 = n_start + sr * wc_i + sc * (wc_w & mask_Gc)
                                    packed_w[
                                        (wc_w * AH + wc_k) * Ty.bits :
                                        (wc_w * AH + wc_k + 1) * Ty.bits
                                    ] = local_B[wk_idx, wn_idx]
                        weights_stream.put(packed_w)

        @df.kernel(mapping=[1], args=[instructions])
        def nest_compute(local_instructions: int32[num_inst, 13]):
            """Receive crossbar data, compute NEST MAC, accumulate, stream result.

            Receives packed iActs/weights from crossbar_load, unpacks them,
            runs AH×AW NEST MAC with int32 accumulation across K-passes,
            then streams packed result to bus.

            Zero point subtraction: decodes iacts_zp and weights_zp from
            instructions and computes (iact - iacts_zp) * (weight - weights_zp),
            matching RTL PE behavior (feather_pe.v).
            """
            iActs: Ty[AH, AW]
            weights: Ty[AH, AW, AH]

            # Decode zero points (layer-level, from SetIVNLayout/SetWVNLayout)
            iacts_zp: int32 = local_instructions[0, 6]
            weights_zp: int32 = local_instructions[1, 6]

            for tile in range(num_tiles):
                nest_accum: int32[AH, AW]
                for _ai in range(AH):
                    for _aj in range(AW):
                        nest_accum[_ai, _aj] = 0

                for k_pass in range(max_k_passes):
                    # Unpack iActs from stream
                    packed_iacts: TyCrossbarPacked = iacts_stream.get()
                    for ic_i in range(AH):
                        for ic_j in range(AW):
                            iActs[ic_i, ic_j] = packed_iacts[
                                (ic_i * AW + ic_j) * Ty.bits :
                                (ic_i * AW + ic_j + 1) * Ty.bits
                            ]

                    # Unpack weights from stream (AH gets)
                    for wc_i in range(AH):
                        packed_w: TyCrossbarPacked = weights_stream.get()
                        for wc_w in range(AW):
                            for wc_k in range(AH):
                                weights[wc_i, wc_w, wc_k] = packed_w[
                                    (wc_w * AH + wc_k) * Ty.bits :
                                    (wc_w * AH + wc_k + 1) * Ty.bits
                                ]

                    # NEST compute + accumulate (with zero point subtraction)
                    for ni in range(AH):
                        for nj in range(AW):
                            temp: int32 = 0
                            for nk in range(AH):
                                a_val: int32 = iActs[nk, nj]
                                w_val: int32 = weights[ni, nj, nk]
                                temp += (a_val - iacts_zp) * (w_val - weights_zp)
                            nest_accum[ni, nj] = nest_accum[ni, nj] + temp

                # Stream accumulated result (after all K-passes)
                for ni in allo.grid(AH, name="nest_stream"):
                    local_result: TyPacked = 0
                    for nj in range(AW):
                        local_result[
                            nj * TyOut.bits : (nj + 1) * TyOut.bits
                        ] = nest_accum[ni, nj]
                    nest_out.put(local_result)

        @df.kernel(mapping=[1])
        def bus():
            """Unpack packed NEST output and distribute to BIRRD input stage."""
            for _ in range(num_tiles * AH):
                array: TyPacked = nest_out.get()
                with allo.meta_for(AW) as i:
                    connection[0, i].put(
                        array[i * TyOut.bits : (i + 1) * TyOut.bits]
                    )

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

        @df.kernel(mapping=[1], args=[output_col_map, output_num_m, output_n_base, accum_m_start, accum_n_start, instructions, C])
        def output_accum(
            local_output_col_map: int32[num_tiles, AW],
            local_output_num_m: int32[num_tiles],
            local_output_n_base: int32[num_tiles, AW],
            local_accum_m_start: int32[num_tiles],
            local_accum_n_start: int32[num_tiles],
            local_instructions: int32[num_inst, 13],
            local_C: int32[M, N],
        ):
            """Accumulate BIRRD output into C[M,N] with col remapping and optional post-quantization.

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

            # Decode post-quantization params from SetOVNLayout
            quant_scale: int32 = local_instructions[2, 6]
            quant_zp: int32 = local_instructions[2, 7]

            for tile in range(num_tiles):
                inst_idx: int32 = tile + 3
                m_start: int32 = local_accum_m_start[tile]
                n_start: int32 = local_accum_n_start[tile]
                num_m: int32 = local_output_num_m[tile]
                sr_val: int32 = local_instructions[inst_idx, 5]

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

        # Precompute col→M maps for all (order, Gr) combinations
        self._col_to_m_maps = {}
        for order in range(6):
            for gr in valid_grs:
                self._col_to_m_maps[(order, gr)] = compute_col_to_m_map(AW, order, gr)

        # Precompute passthrough permutation (for n_base computation)
        self._passthrough_perm = _simulate_birrd_passthrough_perm(AW)
        # Precompute pair→col maps for 2-way reduction n_base
        self._pair_to_col = {}
        for order in range(6):
            self._pair_to_col[order] = compute_output_col_map(AW, order)
        # Precompute m→col maps for multi-way reduction n_base
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

        self._mod(
            A, B, instructions,
            birrd_per_tile,
            col_map_per_tile,
            num_m_per_tile,
            n_base_per_tile,
            m_start_per_tile,
            n_start_per_tile,
            C,
        )


def compute_max_k_passes(instructions, AW, AH):
    """Compute max K-passes across all tiles from encoded instructions.

    Each tile may have a different Kt_per_pass = (AW*AH) / Gr, so the
    number of K-passes varies per tile. Returns the maximum, which is
    used as the compile-time loop bound (padding passes stream zeros).
    """
    num_tiles = len(instructions) - 3
    max_passes = 0
    for t in range(num_tiles):
        Gr = int(instructions[3 + t, 3])
        k_range = int(instructions[3 + t, 12]) - int(instructions[3 + t, 11])
        kt_per_pass = (AW // Gr) * AH
        max_passes = max(max_passes, k_range // kt_per_pass)
    return max_passes


def build_feather_kstreaming_simulator(M, K, N, AW, AH, Ty, num_inst,
                                        max_k_passes, Kt_per_pass=None):
    """Build K-streaming FEATHER+ dataflow for simulation."""
    top = get_feather_full_matrix_top_kstreaming(
        int(M), int(K), int(N), int(AW), int(AH), Ty, int(num_inst),
        int(max_k_passes),
    )
    allo_mod = df.build(top, target="simulator")
    return FeatherKStreamingModule(allo_mod, AW)


def build_feather_kstreaming_hls(M, K, N, AW, AH, Ty, num_inst,
                                  max_k_passes, Kt_per_pass=None,
                                  mode="csim", project=None):
    """Build K-streaming FEATHER+ dataflow for Vitis HLS."""
    if project is None:
        project = "feather_kstreaming.prj"
    top = get_feather_full_matrix_top_kstreaming(
        int(M), int(K), int(N), int(AW), int(AH), Ty, int(num_inst),
        int(max_k_passes),
    )
    s = df.customize(top)
    s.partition("full_matrix_top:C", dim=2, factor=AH)
    allo_mod = s.build(target="vitis_hls", mode=mode, project=project)
    return FeatherKStreamingModule(allo_mod, AW)


def run_sequential_gemm_layers(A_input, layer_weights, layer_program_kwargs,
                                AW, AH):
    """Execute multiple GEMM layers sequentially with int8 intermediates.

    Each non-final layer must enable post-quantization (quant_scale != 0)
    so that its int32 accumulator is quantized to uint8 via
    (accum * scale + zp) & 255. The uint8 output is reinterpreted as
    signed int8 for the next layer's input — matching the RTL auto-quant
    pipeline (OB → quant_post → StaB PONG write).

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
        max_kp = compute_max_k_passes(instructions, AW, AH)

        M_i = current_input.shape[0]
        K_i = current_input.shape[1]
        N_i = B.shape[1]

        mod = build_feather_kstreaming_simulator(
            M_i, K_i, N_i, AW, AH, int8, len(instructions), max_kp,
        )
        C = np.zeros((M_i, N_i), dtype=np.int32)
        mod(current_input, B, instructions, C)
        outputs.append(C)

        # Convert quantized output to int8 for next layer's input
        if i < len(layer_weights) - 1:
            current_input = C.astype(np.uint8).view(np.int8)

    return outputs
