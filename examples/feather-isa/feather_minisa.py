# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""FEATHER+ accelerator with MINISA support using Allo dataflow.

This module provides the full-matrix Allo dataflow implementation for FEATHER+.
A single Allo invocation handles complete input matrices and the full MINISA
instruction list, performing tiling, instruction decode, crossbar reordering,
NEST computation, BIRRD reduction, and output accumulation on-chip.

Architecture (5 pipelined dataflow kernels):
1. crossbar_and_NEST: Instruction decode + input/weight crossbar + PE array
2. bus: Unpack packed NEST output to BIRRD connection streams
3. inst_rw: Distribute BIRRD switch instructions
4. BIRRD[P0,P1]: Butterfly reduction/reorder with compile-time routing
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


def get_feather_full_matrix_top(M, K, N, AW, AH, Ty, num_inst):
    """Create full-matrix FEATHER+ Allo dataflow region.

    Returns a dataflow region that processes complete input matrices and a
    MINISA instruction list in a single invocation, handling tiling, decode,
    crossbar reordering, NEST computation, BIRRD reduction, and output
    accumulation through pipelined dataflow kernels.

    Supports parametric mapping: the Gr field from each SetMapping instruction
    controls K-group splitting in the crossbar. When Gr < AW, BIRRD performs
    reduction and produces Mt=AW//2 outputs. When Gr == AW, BIRRD passes
    through and produces AW outputs (no reduction).

    Args:
        M: Number of rows in A and C
        K: Shared dimension (cols of A, rows of B)
        N: Number of columns in B and C
        AW: Array width (must be power of 2: 4, 8, or 16)
        AH: Array height (number of VN elements)
        Ty: Data type (e.g., int8)
        num_inst: Total instruction count (3 layout + tile mappings)

    Returns:
        Allo dataflow region function that can be built and executed.
    """
    TyPacked = UInt(Ty.bits * AW)
    LOG2_AW = int(log2(AW))
    P0, P1 = compute_birrd_params(AW)
    Mt = AW // 2
    HalfAW = AW // 2
    num_tiles = num_inst - 3  # 3 layout instructions + tile mappings

    @df.region()
    def full_matrix_top(
        A: Ty[M, K],
        B: Ty[K, N],
        instructions: int32[num_inst, 13],
        birrd_inst: int8[num_tiles, P0, P1],
        output_col_map: int32[num_tiles, AW],
        output_num_m: int32[num_tiles],
        accum_m_start: int32[num_tiles],
        accum_n_start: int32[num_tiles],
        C: int32[M, N],
    ):
        """Full-matrix FEATHER+ dataflow region.

        Kernels:
        1. crossbar_and_NEST: instruction decode + crossbar + NEST compute
        2. bus: unpack TyPacked to individual connection streams
        3. inst_rw: distribute per-tile BIRRD switch instructions
        4. BIRRD[P0,P1]: butterfly reduction with compile-time routing
        5. output_accum: col-remap + accumulate tile results into C[M,N]

        Note: output_accum receives precomputed m_start/n_start arrays
        (accum_m_start, accum_n_start) instead of reading from the shared
        instructions array.  This avoids an HLS dataflow violation where two
        kernels would read the same buffer.
        """

        # Stream from NEST to bus
        nest_out: Stream[TyPacked, AH]

        # BIRRD inter-stage connections
        connection: Stream[Ty, 1][P0 + 1, P1 * 2]

        # BIRRD instruction distribution
        inst_input: Stream[int8, 1][P0, P1]

        @df.kernel(mapping=[1], args=[A, B, instructions])
        def crossbar_and_NEST(
            local_A: Ty[M, K],
            local_B: Ty[K, N],
            local_instructions: int32[num_inst, 13],
        ):
            """Instruction decode + input/weight crossbar + NEST compute.

            Decodes IVN/WVN layout orders from instructions, then for each
            tile: decodes Gr from SetMapping and stages inputs via crossbar
            (parametric for order-0, hardcoded HalfAW for other orders),
            runs NEST, and streams packed results.
            """
            # Decode layout orders from first 3 instructions
            ivn_order: int32 = local_instructions[0, 1]
            wvn_order: int32 = local_instructions[1, 1]

            # Tile-level buffers
            iActs: Ty[AH, AW]
            weights: Ty[AH, AW, AH]

            # Weight stationarity: track previous tile's weight config
            # using 1-element arrays (lowered to registers by HLS).
            # Avoids re-reading instructions BRAM for comparison.
            prev_Gr: int32[1]
            prev_n: int32[1]
            prev_k: int32[1]
            prev_Gr[0] = -1
            prev_n[0] = -1
            prev_k[0] = -1

            for tile in range(num_tiles):
                inst_idx: int32 = tile + 3
                # Decode tile bounds and mapping parameter
                Gr: int32 = local_instructions[inst_idx, 3]
                m_start: int32 = local_instructions[inst_idx, 7]
                n_start: int32 = local_instructions[inst_idx, 9]
                k_start: int32 = local_instructions[inst_idx, 11]

                # === Input Crossbar (IVN order-dependent reordering) ===
                # Order-0 uses parametric Gr; other orders use HalfAW
                for ic_i in range(AH):
                    for ic_j in range(AW):
                        m_idx: int32 = 0
                        k_idx: int32 = 0
                        if ivn_order == 0:  # ORDER_012 (parametric)
                            m_idx = m_start + (ic_j % Gr)
                            k_idx = k_start + ic_i + (ic_j // Gr) * AH
                        elif ivn_order == 1:  # ORDER_021
                            m_idx = m_start + (ic_j // Mt)
                            k_idx = k_start + ic_i + (ic_j % Mt) * AH
                        elif ivn_order == 2:  # ORDER_102
                            m_idx = m_start + ic_i
                            k_idx = k_start + (ic_j % Mt) + (ic_j // Mt) * AH
                        elif ivn_order == 3:  # ORDER_120
                            m_idx = m_start + ic_i
                            k_idx = k_start + (ic_j // Mt) + (ic_j % Mt) * AH
                        elif ivn_order == 4:  # ORDER_201
                            m_idx = m_start + (ic_j // Mt)
                            k_idx = k_start + (ic_j % Mt) + ic_i * AH
                        else:  # ORDER_210
                            m_idx = m_start + (ic_j % Mt)
                            k_idx = k_start + (ic_j // Mt) + ic_i * AH
                        iActs[ic_i, ic_j] = local_A[m_idx, k_idx]

                # === Weight Crossbar (WVN order-dependent reordering) ===
                # Weight stationarity: skip loading when consecutive tiles
                # share the same Gr, n_start, and k_start (identical weights).
                # Compare with register-cached previous tile values.
                load_w: int32 = 1
                if prev_Gr[0] == Gr:
                    if prev_n[0] == n_start:
                        if prev_k[0] == k_start:
                            load_w = 0
                prev_Gr[0] = Gr
                prev_n[0] = n_start
                prev_k[0] = k_start

                if load_w == 1:
                    # Order-0 uses parametric Gr; other orders use HalfAW
                    for wc_i in range(AH):
                        for wc_w in range(AW):
                            for wc_k in range(AH):
                                wk_idx: int32 = 0
                                wn_idx: int32 = 0
                                if wvn_order == 0:  # ORDER_012 (parametric)
                                    wk_idx = k_start + wc_k + (wc_w // Gr) * AH
                                    wn_idx = n_start + wc_i
                                elif wvn_order == 1:  # ORDER_021
                                    wk_idx = k_start + wc_k + wc_i * AH
                                    wn_idx = n_start + (wc_w // HalfAW)
                                elif wvn_order == 2:  # ORDER_102
                                    wk_idx = k_start + (wc_w // HalfAW) + wc_k * AH
                                    wn_idx = n_start + wc_i
                                elif wvn_order == 3:  # ORDER_120
                                    wk_idx = k_start + wc_i + wc_k * AH
                                    wn_idx = n_start + (wc_w // HalfAW)
                                elif wvn_order == 4:  # ORDER_201
                                    wk_idx = k_start + (wc_w // HalfAW) + wc_i * AH
                                    wn_idx = n_start + wc_k
                                else:  # ORDER_210
                                    wk_idx = k_start + wc_i + (wc_w // HalfAW) * AH
                                    wn_idx = n_start + wc_k
                                weights[wc_i, wc_w, wc_k] = local_B[wk_idx, wn_idx]

                # === NEST: AH x AW PE array ===
                for ni in allo.grid(AH, name="nest"):
                    local_buffer: Ty[AW] = 0
                    for nj in range(AW):
                        temp: Ty = 0
                        for nk in range(AH):
                            temp += iActs[nk, nj] * weights[ni, nj, nk]
                        local_buffer[nj] = temp

                    # Pack AW results into single word for bus transfer
                    local_result: TyPacked = 0
                    for nj in range(AW):
                        local_result[nj * Ty.bits : (nj + 1) * Ty.bits] = local_buffer[nj]
                    nest_out.put(local_result)

        @df.kernel(mapping=[1])
        def bus():
            """Unpack packed NEST output and distribute to BIRRD input stage."""
            for _ in range(num_tiles * AH):
                array: TyPacked = nest_out.get()
                with allo.meta_for(AW) as i:
                    connection[0, i].put(array[i * Ty.bits : (i + 1) * Ty.bits])

        @df.kernel(mapping=[1], args=[birrd_inst])
        def inst_rw(local_birrd_inst: int8[num_tiles, P0, P1]):
            """Distribute per-tile BIRRD switch instructions."""
            for tile in range(num_tiles):
                with allo.meta_for(P0) as i:
                    with allo.meta_for(P1) as j:
                        inst_input[i, j].put(local_birrd_inst[tile, i, j])

        @df.kernel(mapping=[P0, P1])
        def BIRRD():
            """BIRRD butterfly reduction/reorder network.

            Each instance is one switch. Reads a new instruction per tile,
            then processes AH iterations for that tile.
            Uses compile-time reverse_bits() routing.
            """
            i, j = df.get_pid()

            for _tile in range(num_tiles):
                inst_val = inst_input[i, j].get()

                for _ in range(AH):
                    in_left: Ty = connection[i, 2 * j].get()
                    in_right: Ty = connection[i, 2 * j + 1].get()

                    out_left: Ty = 0
                    out_right: Ty = 0

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

                    # Route to next stage with bit-reversal
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
                        # Last stage: direct output
                        connection[P0, 2 * j].put(out_left)
                        connection[P0, 2 * j + 1].put(out_right)

        @df.kernel(mapping=[1], args=[output_col_map, output_num_m, accum_m_start, accum_n_start, C])
        def output_accum(
            local_output_col_map: int32[num_tiles, AW],
            local_output_num_m: int32[num_tiles],
            local_accum_m_start: int32[num_tiles],
            local_accum_n_start: int32[num_tiles],
            local_C: int32[M, N],
        ):
            """Accumulate BIRRD output into C[M,N] with col remapping.

            For each tile: reads precomputed m_start/n_start, collects
            BIRRD output, remaps columns, and accumulates into C.
            Supports both reduction mode (Gr < AW, Mt outputs) and
            pass-through mode (Gr == AW, AW outputs).

            Uses precomputed m_start/n_start arrays instead of reading from
            the shared instructions buffer (avoids HLS dataflow violation).
            Uses a local accumulation buffer so that local_C is write-only
            (avoids load_buf/output_accum multi-writer violation).
            """
            # Local accumulation buffer — avoids read-modify-write on local_C
            # so that the HLS backend treats C as output-only (no load_buf).
            accum: int32[M, N]
            for _ai in range(M):
                for _aj in range(N):
                    accum[_ai, _aj] = 0

            for tile in range(num_tiles):
                m_start: int32 = local_accum_m_start[tile]
                n_start: int32 = local_accum_n_start[tile]
                num_m: int32 = local_output_num_m[tile]

                # Collect BIRRD output for this tile
                tile_out: Ty[AH, AW]
                for d in range(AH):
                    with allo.meta_for(AW) as aw_i:
                        tile_out[d, aw_i] = connection[P0, aw_i].get()

                # Accumulate into local buffer with column remapping
                for om in range(AW):
                    if om < num_m:
                        col: int32 = local_output_col_map[tile, om]
                        for on in range(AH):
                            accum[m_start + om, n_start + on] = (
                                accum[m_start + om, n_start + on] + tile_out[on, col]
                            )

            # Write final result to output (write-only access to local_C)
            for _wi in range(M):
                for _wj in range(N):
                    local_C[_wi, _wj] = accum[_wi, _wj]

    return full_matrix_top


class FeatherFullMatrixModule:
    """Wrapper that provides clean (A, B, instructions, C) interface.

    Precomputes all BIRRD configs at build time for all 6 OVN orders.
    At call time, extracts OVN order and per-tile Gr from instructions
    to build per-tile BIRRD instructions and output column maps.

    Per-tile logic:
    - Gr < AW (reduction mode): uses standard BIRRD reduction, Mt outputs
    - Gr == AW (pass-through mode): all-PS BIRRD, AW outputs with identity map

    The dataflow BIRRD uses compile-time reverse_bits() routing, so
    route_left/route_right are not needed.
    """

    def __init__(self, allo_mod, AW):
        from minisa.lowering import lower_ovn_layout, compute_output_col_map
        from minisa.isa import SetOVNLayout

        self._mod = allo_mod
        self._AW = AW
        P0, P1 = compute_birrd_params(AW)
        self._P0 = P0
        self._P1 = P1
        # Precompute BIRRD tables + col maps for all 6 OVN orders
        self._birrd_tables = {}
        self._col_maps = {}
        for order in range(6):
            ovn = SetOVNLayout(order=order, PL0=AW, PL1=1, QL0=AW, QL1=1)
            self._birrd_tables[order] = lower_ovn_layout(ovn, AW, AW)
            self._col_maps[order] = compute_output_col_map(AW, order)

    def __call__(self, A, B, instructions, C):
        # OVN order is at instructions[2, 1] (row 2 = SetOVNLayout, field 1 = order)
        ovn_order = int(instructions[2, 1])
        AW = self._AW
        Mt = AW // 2
        P0, P1 = self._P0, self._P1
        num_tiles = len(instructions) - 3

        # Build per-tile BIRRD instructions and output mappings
        birrd_per_tile = np.zeros((num_tiles, P0, P1), dtype=np.int8)
        col_map_per_tile = np.zeros((num_tiles, AW), dtype=np.int32)
        num_m_per_tile = np.zeros(num_tiles, dtype=np.int32)

        for t in range(num_tiles):
            Gr = int(instructions[3 + t, 3])
            if Gr < AW:
                # Reduction mode: standard BIRRD + col_map
                birrd_per_tile[t] = self._birrd_tables[ovn_order]
                col_map_per_tile[t, :Mt] = self._col_maps[ovn_order]
                num_m_per_tile[t] = Mt
            else:
                # Pass-through mode: all-PS BIRRD + identity col_map
                # birrd_per_tile[t] is already all zeros (PS=0)
                col_map_per_tile[t] = np.arange(AW, dtype=np.int32)
                num_m_per_tile[t] = AW

        # Precompute m_start/n_start per tile for output_accum
        # (avoids sharing the instructions buffer with crossbar_and_NEST)
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
            m_start_per_tile,
            n_start_per_tile,
            C,
        )


def get_feather_full_matrix_top_kstreaming(M, K, N, AW, AH, Ty, num_inst,
                                            num_k_passes, Kt_per_pass):
    """Create K-streaming FEATHER+ dataflow region.

    Specialized for Gr=AW (pass-through BIRRD, no runtime dividers).
    The crossbar_and_NEST kernel has an inner K-loop that accumulates
    NEST partial products across K-passes in int32 before streaming
    to BIRRD.

    Key differences from get_feather_full_matrix_top():
    - Uses int32 for intermediate types (NEST output, BIRRD, output_accum)
    - Inner K-loop in crossbar_and_NEST accumulates across K-passes
    - Crossbar logic hardcoded for Gr=AW ORDER_012 (no dividers)
    - No weight stationarity tracking (K-loop handles this naturally)

    Args:
        M, K, N: Matrix dimensions
        AW: Array width (must be power of 2)
        AH: Array height
        Ty: Input data type (e.g., int8)
        num_inst: Total instruction count (3 layout + tile mappings)
        num_k_passes: Number of K-passes per tile
        Kt_per_pass: K elements per pass (typically AH)
    """
    TyOut = int32  # Wider type for accumulated output
    TyPacked = UInt(TyOut.bits * AW)
    LOG2_AW = int(log2(AW))
    P0, P1 = compute_birrd_params(AW)
    num_tiles = num_inst - 3

    @df.region()
    def full_matrix_top(
        A: Ty[M, K],
        B: Ty[K, N],
        instructions: int32[num_inst, 13],
        birrd_inst: int8[num_tiles, P0, P1],
        output_col_map: int32[num_tiles, AW],
        output_num_m: int32[num_tiles],
        accum_m_start: int32[num_tiles],
        accum_n_start: int32[num_tiles],
        C: int32[M, N],
    ):
        """K-streaming FEATHER+ dataflow region."""

        nest_out: Stream[TyPacked, AH * 2]
        connection: Stream[TyOut, AH][P0 + 1, P1 * 2]
        inst_input: Stream[int8, num_tiles][P0, P1]

        @df.kernel(mapping=[1], args=[A, B, instructions])
        def crossbar_and_NEST(
            local_A: Ty[M, K],
            local_B: Ty[K, N],
            local_instructions: int32[num_inst, 13],
        ):
            """K-streaming crossbar + NEST with inner K-loop.

            Specialized for Gr=AW ORDER_012:
              m_idx = m_start + ic_j  (no division)
              k_idx = k_start + ic_i  (no division)
            Accumulates NEST partial products across K-passes in int32.
            """
            iActs: Ty[AH, AW]
            weights: Ty[AH, AW, AH]

            for tile in range(num_tiles):
                inst_idx: int32 = tile + 3
                m_start: int32 = local_instructions[inst_idx, 7]
                n_start: int32 = local_instructions[inst_idx, 9]
                k_start_tile: int32 = local_instructions[inst_idx, 11]

                # Accumulation buffer for K-streaming (int32)
                nest_accum: int32[AH, AW]
                for _ai in range(AH):
                    for _aj in range(AW):
                        nest_accum[_ai, _aj] = 0

                for k_pass in range(num_k_passes):
                    k_start: int32 = k_start_tile + k_pass * Kt_per_pass

                    # === Input Crossbar (Gr=AW, ORDER_012, no division) ===
                    for ic_i in range(AH):
                        for ic_j in range(AW):
                            iActs[ic_i, ic_j] = local_A[
                                m_start + ic_j, k_start + ic_i
                            ]

                    # === Weight Crossbar (Gr=AW, ORDER_012, no division) ===
                    for wc_i in range(AH):
                        for wc_w in range(AW):
                            for wc_k in range(AH):
                                weights[wc_i, wc_w, wc_k] = local_B[
                                    k_start + wc_k, n_start + wc_i
                                ]

                    # === NEST compute → accumulate ===
                    for ni in range(AH):
                        for nj in range(AW):
                            temp: int32 = 0
                            for nk in range(AH):
                                temp += iActs[nk, nj] * weights[ni, nj, nk]
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

        @df.kernel(mapping=[1], args=[output_col_map, output_num_m, accum_m_start, accum_n_start, C])
        def output_accum(
            local_output_col_map: int32[num_tiles, AW],
            local_output_num_m: int32[num_tiles],
            local_accum_m_start: int32[num_tiles],
            local_accum_n_start: int32[num_tiles],
            local_C: int32[M, N],
        ):
            """Accumulate BIRRD output into C[M,N] with col remapping (int32)."""
            accum: int32[M, N]
            for _ai in range(M):
                for _aj in range(N):
                    accum[_ai, _aj] = 0

            for tile in range(num_tiles):
                m_start: int32 = local_accum_m_start[tile]
                n_start: int32 = local_accum_n_start[tile]
                num_m: int32 = local_output_num_m[tile]

                tile_out: int32[AH, AW]
                for d in range(AH):
                    with allo.meta_for(AW) as aw_i:
                        tile_out[d, aw_i] = connection[P0, aw_i].get()

                for om in range(AW):
                    if om < num_m:
                        col: int32 = local_output_col_map[tile, om]
                        for on in range(AH):
                            accum[m_start + om, n_start + on] = (
                                accum[m_start + om, n_start + on] + tile_out[on, col]
                            )

            for _wi in range(M):
                for _wj in range(N):
                    local_C[_wi, _wj] = accum[_wi, _wj]

    return full_matrix_top


def get_feather_full_matrix_top_kstreaming_v2(M, K, N, AW, AH, Ty, num_inst,
                                               num_k_passes, Kt_per_pass):
    """K-streaming FEATHER+ with split crossbar/NEST kernels (Phase 2).

    Splits crossbar_and_NEST into separate crossbar_load and nest_compute
    dataflow kernels connected by intermediate streams. This allows Vitis HLS
    to pipeline them: crossbar_load can prefetch next tile/K-pass data while
    nest_compute processes current data.

    Data transfer protocol per K-pass:
      crossbar_load sends 1 iActs packet + AH weights packets via streams
      nest_compute receives and unpacks them, runs NEST MAC, accumulates

    All other kernels (bus, inst_rw, BIRRD, output_accum) are unchanged.
    """
    TyOut = int32
    TyPacked = UInt(TyOut.bits * AW)       # UInt(128) for packing 4 int32
    TyCrossbarPacked = UInt(AH * AW * Ty.bits)  # UInt(128) for packing 16 int8
    LOG2_AW = int(log2(AW))
    P0, P1 = compute_birrd_params(AW)
    num_tiles = num_inst - 3
    # Allow crossbar_load to run ~2 tiles ahead of nest_compute
    xbar_i_depth = num_k_passes * 2
    xbar_w_depth = num_k_passes * AH * 2

    @df.region()
    def full_matrix_top(
        A: Ty[M, K],
        B: Ty[K, N],
        instructions: int32[num_inst, 13],
        birrd_inst: int8[num_tiles, P0, P1],
        output_col_map: int32[num_tiles, AW],
        output_num_m: int32[num_tiles],
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

            Per K-pass: 1 iActs packet (16 int8 → UInt(128)) +
                        AH weights packets (16 int8 each → UInt(128)).
            """
            for tile in range(num_tiles):
                inst_idx: int32 = tile + 3
                m_start: int32 = local_instructions[inst_idx, 7]
                n_start: int32 = local_instructions[inst_idx, 9]
                k_start_tile: int32 = local_instructions[inst_idx, 11]

                for k_pass in range(num_k_passes):
                    k_start: int32 = k_start_tile + k_pass * Kt_per_pass

                    # === Pack and stream iActs[AH, AW] → 1 x UInt(128) ===
                    packed_iacts: TyCrossbarPacked = 0
                    for ic_i in range(AH):
                        for ic_j in range(AW):
                            packed_iacts[
                                (ic_i * AW + ic_j) * Ty.bits :
                                (ic_i * AW + ic_j + 1) * Ty.bits
                            ] = local_A[m_start + ic_j, k_start + ic_i]
                    iacts_stream.put(packed_iacts)

                    # === Pack and stream weights[AH, AW, AH] → AH x UInt(128) ===
                    for wc_i in range(AH):
                        packed_w: TyCrossbarPacked = 0
                        for wc_w in range(AW):
                            for wc_k in range(AH):
                                packed_w[
                                    (wc_w * AH + wc_k) * Ty.bits :
                                    (wc_w * AH + wc_k + 1) * Ty.bits
                                ] = local_B[k_start + wc_k, n_start + wc_i]
                        weights_stream.put(packed_w)

        @df.kernel(mapping=[1])
        def nest_compute():
            """Receive crossbar data, compute NEST MAC, accumulate, stream result.

            Receives packed iActs/weights from crossbar_load, unpacks them,
            runs 4x4 NEST MAC with int32 accumulation across K-passes,
            then streams packed result to bus.
            """
            iActs: Ty[AH, AW]
            weights: Ty[AH, AW, AH]

            for tile in range(num_tiles):
                nest_accum: int32[AH, AW]
                for _ai in range(AH):
                    for _aj in range(AW):
                        nest_accum[_ai, _aj] = 0

                for k_pass in range(num_k_passes):
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

                    # NEST compute + accumulate
                    for ni in range(AH):
                        for nj in range(AW):
                            temp: int32 = 0
                            for nk in range(AH):
                                temp += iActs[nk, nj] * weights[ni, nj, nk]
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

        @df.kernel(mapping=[1], args=[output_col_map, output_num_m, accum_m_start, accum_n_start, C])
        def output_accum(
            local_output_col_map: int32[num_tiles, AW],
            local_output_num_m: int32[num_tiles],
            local_accum_m_start: int32[num_tiles],
            local_accum_n_start: int32[num_tiles],
            local_C: int32[M, N],
        ):
            """Accumulate BIRRD output into C[M,N] with col remapping (int32)."""
            accum: int32[M, N]
            for _ai in range(M):
                for _aj in range(N):
                    accum[_ai, _aj] = 0

            for tile in range(num_tiles):
                m_start: int32 = local_accum_m_start[tile]
                n_start: int32 = local_accum_n_start[tile]
                num_m: int32 = local_output_num_m[tile]

                tile_out: int32[AH, AW]
                for d in range(AH):
                    with allo.meta_for(AW) as aw_i:
                        tile_out[d, aw_i] = connection[P0, aw_i].get()

                for om in range(AW):
                    if om < num_m:
                        col: int32 = local_output_col_map[tile, om]
                        for on in range(AH):
                            accum[m_start + om, n_start + on] = (
                                accum[m_start + om, n_start + on] + tile_out[on, col]
                            )

            for _wi in range(M):
                for _wj in range(N):
                    local_C[_wi, _wj] = accum[_wi, _wj]

    return full_matrix_top


class FeatherKStreamingModule:
    """Wrapper for K-streaming FEATHER+ with clean (A, B, instructions, C) interface.

    All tiles use Gr=AW (pass-through BIRRD, identity column map).
    """

    def __init__(self, allo_mod, AW):
        self._mod = allo_mod
        self._AW = AW
        P0, P1 = compute_birrd_params(AW)
        self._P0 = P0
        self._P1 = P1

    def __call__(self, A, B, instructions, C):
        AW = self._AW
        P0, P1 = self._P0, self._P1
        num_tiles = len(instructions) - 3

        # All tiles use Gr=AW → pass-through BIRRD (all PS=0)
        birrd_per_tile = np.zeros((num_tiles, P0, P1), dtype=np.int8)
        # Identity column map for all tiles
        col_map_per_tile = np.tile(
            np.arange(AW, dtype=np.int32), (num_tiles, 1)
        )
        # All tiles produce AW outputs (no reduction)
        num_m_per_tile = np.full(num_tiles, AW, dtype=np.int32)

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
            m_start_per_tile,
            n_start_per_tile,
            C,
        )


def build_feather_kstreaming_simulator(M, K, N, AW, AH, Ty, num_inst,
                                        num_k_passes, Kt_per_pass):
    """Build K-streaming FEATHER+ dataflow for simulation."""
    top = get_feather_full_matrix_top_kstreaming(
        int(M), int(K), int(N), int(AW), int(AH), Ty, int(num_inst),
        int(num_k_passes), int(Kt_per_pass),
    )
    allo_mod = df.build(top, target="simulator")
    return FeatherKStreamingModule(allo_mod, AW)


def build_feather_kstreaming_hls(M, K, N, AW, AH, Ty, num_inst,
                                  num_k_passes, Kt_per_pass,
                                  mode="csim", project=None):
    """Build K-streaming FEATHER+ dataflow for Vitis HLS."""
    if project is None:
        project = "feather_kstreaming.prj"
    top = get_feather_full_matrix_top_kstreaming(
        int(M), int(K), int(N), int(AW), int(AH), Ty, int(num_inst),
        int(num_k_passes), int(Kt_per_pass),
    )
    s = df.customize(top)
    s.partition("full_matrix_top:C", dim=2, factor=AH)
    allo_mod = s.build(target="vitis_hls", mode=mode, project=project)
    return FeatherKStreamingModule(allo_mod, AW)


def build_feather_kstreaming_v2_simulator(M, K, N, AW, AH, Ty, num_inst,
                                            num_k_passes, Kt_per_pass):
    """Build split-kernel K-streaming FEATHER+ dataflow for simulation."""
    top = get_feather_full_matrix_top_kstreaming_v2(
        int(M), int(K), int(N), int(AW), int(AH), Ty, int(num_inst),
        int(num_k_passes), int(Kt_per_pass),
    )
    allo_mod = df.build(top, target="simulator")
    return FeatherKStreamingModule(allo_mod, AW)


def build_feather_kstreaming_v2_hls(M, K, N, AW, AH, Ty, num_inst,
                                      num_k_passes, Kt_per_pass,
                                      mode="csim", project=None):
    """Build split-kernel K-streaming FEATHER+ dataflow for Vitis HLS."""
    if project is None:
        project = "feather_kstreaming_v2.prj"
    top = get_feather_full_matrix_top_kstreaming_v2(
        int(M), int(K), int(N), int(AW), int(AH), Ty, int(num_inst),
        int(num_k_passes), int(Kt_per_pass),
    )
    s = df.customize(top)
    s.partition("full_matrix_top:C", dim=2, factor=AH)
    allo_mod = s.build(target="vitis_hls", mode=mode, project=project)
    return FeatherKStreamingModule(allo_mod, AW)


def build_feather_full_matrix_simulator(M, K, N, AW, AH, Ty, num_inst):
    """Build full-matrix FEATHER+ dataflow for simulation.

    Returns a module with clean interface: mod(A, B, instructions, C).
    BIRRD configs are precomputed internally for all OVN orders.
    """
    top = get_feather_full_matrix_top(
        int(M), int(K), int(N), int(AW), int(AH), Ty, int(num_inst),
    )
    allo_mod = df.build(top, target="simulator")
    return FeatherFullMatrixModule(allo_mod, AW)


def build_feather_full_matrix_hls(M, K, N, AW, AH, Ty, num_inst,
                                   mode="csim", project=None):
    """Build full-matrix FEATHER+ dataflow for Vitis HLS.

    Returns a module with clean interface: mod(A, B, instructions, C).
    BIRRD configs are precomputed internally for all OVN orders.
    """
    if project is None:
        project = "feather_full_matrix.prj"
    top = get_feather_full_matrix_top(
        int(M), int(K), int(N), int(AW), int(AH), Ty, int(num_inst),
    )
    s = df.customize(top)
    s.partition("full_matrix_top:C", dim=2, factor=AH)

    allo_mod = s.build(target="vitis_hls", mode=mode, project=project)
    return FeatherFullMatrixModule(allo_mod, AW)


def run_full_matrix_gemm(
    M, N, K,
    AW=8, AH=8,
    Ty_data=int8,
    verbose=False,
    A=None, B=None,
    seed=42,
    build_target="simulator",
    build_mode="csim",
    project_dir=None,
):
    """Run GEMM using the full-matrix execution model.

    Single Allo invocation per GEMM — no Python tile loop.

    Args:
        M, N, K: Matrix dimensions for C[M,N] = A[M,K] * B[K,N]
        AW: Array width
        AH: Array height
        Ty_data: Data type
        verbose: Enable verbose logging
        A: Optional input matrix (generated if None)
        B: Optional weight matrix (generated if None)
        seed: Random seed for input generation
        build_target: "simulator" or "vitis_hls"
        build_mode: HLS build mode (for vitis_hls target)
        project_dir: HLS project directory (for vitis_hls target)

    Returns:
        (C, reference, passed): Output, numpy reference, and verification status
    """
    from minisa.isa import create_gemm_program, encode_program

    # Create and encode MINISA program
    program = create_gemm_program(M, N, K, AH, AW)
    instructions = encode_program(program)
    num_inst = len(instructions)

    if verbose:
        print(f"Full-matrix GEMM: C[{M},{N}] = A[{M},{K}] x B[{K},{N}]")
        print(f"  AW={AW}, AH={AH}, Ty={Ty_data}")
        print(f"  Encoded {num_inst} instructions ({num_inst - 3} tile mappings)")

    # Generate or use provided inputs
    if A is None or B is None:
        np.random.seed(seed)
        A = np.random.randint(-4, 4, size=(M, K)).astype(np.int8)
        B = np.random.randint(-4, 4, size=(K, N)).astype(np.int8)

    # Build Allo module (single invocation)
    if build_target == "simulator":
        mod = build_feather_full_matrix_simulator(M, K, N, AW, AH, Ty_data, num_inst)
    else:
        mod = build_feather_full_matrix_hls(
            M, K, N, AW, AH, Ty_data, num_inst,
            mode=build_mode, project=project_dir,
        )

    # Allocate output
    C = np.zeros((M, N), dtype=np.int32)

    # Execute — single Allo call (wrapper handles BIRRD config selection)
    mod(A, B, instructions, C)

    # Compute numpy reference
    ref = np.dot(A, B)

    # Verify
    try:
        np.testing.assert_allclose(C, ref, atol=1e-5)
        passed = True
    except AssertionError:
        passed = False

    if verbose:
        print(f"  Verification: {'PASSED' if passed else 'FAILED'}")
        if not passed:
            diff = np.abs(C - ref)
            print(f"  Max abs diff: {diff.max()}")
            print(f"  Diff locations: {np.argwhere(diff > 0).tolist()[:5]}...")

    return C, ref, passed


if __name__ == "__main__":
    # Quick verification test
    C, ref, passed = run_full_matrix_gemm(
        M=8, N=8, K=16, AW=8, AH=8, verbose=True
    )
    print(f"\nReference (numpy):\n{ref}")
    print(f"\nFull-matrix output (Allo):\n{C}")
    assert passed, "Verification failed"
    print("FEATHER+ MINISA full-matrix verification complete!")
