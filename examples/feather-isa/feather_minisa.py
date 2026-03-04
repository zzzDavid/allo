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
                                            num_k_passes, Kt_per_pass):
    """Create FEATHER+ dataflow region with split crossbar/NEST and flexible Gr.

    Supports all power-of-2 Gr values per tile via bit operations in the
    crossbar index arithmetic. No runtime dividers — uses (ic_j & (Gr-1))
    for modulo and (ic_j >> log2_Gr) for division.

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
        num_k_passes: Number of K-passes per tile
        Kt_per_pass: K elements per pass (typically AH)
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

            Supports parametric Gr per tile via power-of-2 bit operations:
              ic_j % Gr  →  ic_j & (Gr - 1)    (AND with bitmask)
              ic_j // Gr →  ic_j >> log2_Gr     (right shift)

            Per K-pass: 1 iActs packet (AH*AW int8 → UInt(128)) +
                        AH weights packets (AW*AH int8 each → UInt(128)).
            """
            for tile in range(num_tiles):
                inst_idx: int32 = tile + 3
                Gr: int32 = local_instructions[inst_idx, 3]
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

                for k_pass in range(num_k_passes):
                    k_start: int32 = k_start_tile + k_pass * Kt_per_pass

                    # === Input crossbar (ORDER_012 with bit ops) ===
                    packed_iacts: TyCrossbarPacked = 0
                    for ic_i in range(AH):
                        for ic_j in range(AW):
                            m_idx: int32 = m_start + (ic_j & mask_Gr)
                            k_idx: int32 = k_start + ic_i + (ic_j >> log2_Gr) * AH
                            packed_iacts[
                                (ic_i * AW + ic_j) * Ty.bits :
                                (ic_i * AW + ic_j + 1) * Ty.bits
                            ] = local_A[m_idx, k_idx]
                    iacts_stream.put(packed_iacts)

                    # === Weight crossbar (ORDER_012 with bit ops) ===
                    for wc_i in range(AH):
                        packed_w: TyCrossbarPacked = 0
                        for wc_w in range(AW):
                            for wc_k in range(AH):
                                wk_idx: int32 = k_start + wc_k + (wc_w >> log2_Gr) * AH
                                wn_idx: int32 = n_start + wc_i
                                packed_w[
                                    (wc_w * AH + wc_k) * Ty.bits :
                                    (wc_w * AH + wc_k + 1) * Ty.bits
                                ] = local_B[wk_idx, wn_idx]
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
    """Wrapper for FEATHER+ with clean (A, B, instructions, C) interface.

    Supports all Gr values per tile. Precomputes BIRRD tables for all
    OVN orders at build time. At call time, selects per-tile BIRRD
    configuration based on Gr: reduction (Gr < AW) or pass-through (Gr == AW).
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
