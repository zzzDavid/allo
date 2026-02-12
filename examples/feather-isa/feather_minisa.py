# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""FEATHER+ accelerator with MINISA support using Allo dataflow.

This module provides the Allo dataflow implementation for FEATHER+ that can
be configured via MINISA instructions. All compute is performed by Allo kernels.

Architecture:
- Input VN Buffer: Stages input data according to IVN layout
- Weight VN Buffer: Stages weight data according to WVN layout
- NEST: AH x AW PE array performing VN dot products
- BIRRD: Butterfly reduction/reorder network
- Output Buffer: Collects and reorders final results

The MINISA instructions (SetIVNLayout, SetWVNLayout, SetOVNLayout, SetMapping)
configure these hardware components through configuration tensors.
"""

from math import log2
from typing import Tuple

import allo
from allo.ir.types import int8, int32, UInt, AlloType, Stream
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


def get_feather_minisa_top(AW: int, AH: int, Ty: AlloType):
    """Create FEATHER+ Allo dataflow region with MINISA support.

    This is the main entry point for MINISA-configured FEATHER+ execution.
    The returned region includes all compute kernels connected via streams.

    Args:
        AW: Array width (must be power of 2: 4, 8, or 16)
        AH: Array height (number of VN elements)
        Ty: Data type (e.g., int8, int32)

    Returns:
        Allo dataflow region function that can be built and executed.

    The region takes these inputs:
        - iActs: Input activations [AH, AW] - staged by IVN layout
        - weights: Weight tensor [AH, AW, AH] - staged by WVN layout
        - inst: BIRRD instruction array [P0, P1] - from OVN lowering
        - output_buffer: Output tensor [AH, AW] - receives results
    """
    TyPacked = UInt(Ty.bits * AW)

    # Compute BIRRD network dimensions
    LOG2_AW = int(log2(AW))
    P0, P1 = compute_birrd_params(AW)

    @df.region()
    def top(
        iActs: Ty[AH, AW],
        weights: Ty[AH, AW, AH],
        inst: int8[P0, P1],
        output_buffer: Ty[AH, AW],
    ):
        """FEATHER+ dataflow region with MINISA configuration.

        This region implements the complete FEATHER+ datapath:
        1. NEST computes AH-way dot products in AH x AW PE array
        2. Bus unpacks NEST output for BIRRD input
        3. inst_rw loads BIRRD switch instructions
        4. BIRRD performs butterfly reduction/reordering
        5. output collects final results

        All computation is performed by Allo kernels - no numpy compute.
        """

        # Stream from NEST compute array to bus
        nest_out: Stream[TyPacked, AH]

        @df.kernel(mapping=[1], args=[iActs, weights])
        def NEST(local_iActs: Ty[AH, AW], local_weights: Ty[AH, AW, AH]):
            """NEST compute array kernel.

            Implements AH x AW PE array where each PE performs:
            - AH-way dot product (temporal reduction)
            - Results collected and packed for BIRRD input

            This kernel performs the core MAC computation.
            """
            for i in allo.grid(AH, name="nest"):  # Rows, can be pipelined
                local_buffer: Ty[AW] = 0
                for j in range(AW):  # Cols, can be fully parallelized
                    temp: Ty = 0
                    for k in range(AH):  # AH-way dot product (VN computation)
                        iAct: Ty = local_iActs[k, j]
                        weight: Ty = local_weights[i, j, k]
                        temp += iAct * weight
                    local_buffer[j] = temp

                # Pack AW results into single word for bus transfer
                local_result: TyPacked = 0
                for j in range(AW):
                    local_result[j * Ty.bits : (j + 1) * Ty.bits] = local_buffer[j]
                nest_out.put(local_result)

        # Streams for BIRRD inter-stage connections
        connection: Stream[Ty, 1][P0 + 1, P1 * 2]

        @df.kernel(mapping=[1])
        def bus():
            """Bus kernel for NEST to BIRRD data transfer.

            Unpacks packed NEST output and distributes to BIRRD input stage.
            """
            for _ in range(AH):
                array: TyPacked = nest_out.get()
                with allo.meta_for(AW) as i:
                    connection[0, i].put(array[i * Ty.bits : (i + 1) * Ty.bits])

        # Stream for BIRRD instruction distribution
        inst_input: Stream[int8, 1][P0, P1]

        @df.kernel(mapping=[1], args=[inst])
        def inst_rw(local_inst: int8[P0, P1]):
            """Instruction loader kernel.

            Distributes BIRRD switch instructions to all stages.
            Instructions are generated by MINISA OVN lowering.
            """
            with allo.meta_for(P0) as i:
                with allo.meta_for(P1) as j:
                    inst_input[i, j].put(local_inst[i, j])

        @df.kernel(mapping=[P0, P1])
        def BIRRD():
            """BIRRD butterfly reduction/reorder kernel.

            Each instance implements one switch in the butterfly network.
            Supports four operations configured by MINISA OVN layout:
            - PS (0): Pass through unchanged
            - AR (1): Add right to left, output right
            - AL (2): Add right to left, output left
            - SW (3): Swap left and right

            The network enables zero-latency data layout changes during
            reduction (RIR - Reorder In Reduction).
            """
            i, j = df.get_pid()
            inst_val = inst_input[i, j].get()

            for _ in range(AH):
                # Get inputs from previous stage
                in_left: Ty = connection[i, 2 * j].get()
                in_right: Ty = connection[i, 2 * j + 1].get()

                out_left: Ty = 0
                out_right: Ty = 0

                # Execute switch operation based on instruction
                if inst_val == 0:  # Pass
                    out_left = in_left
                    out_right = in_right
                elif inst_val == 1:  # Add-Right
                    out_left = in_left
                    out_right = in_left + in_right
                elif inst_val == 2:  # Add-Left
                    out_left = in_left + in_right
                    out_right = in_right
                else:  # Swap
                    out_left = in_right
                    out_right = in_left

                # Route to next stage with bit-reversal
                with allo.meta_if(i != P0 - 1):
                    connection[
                        i + 1,
                        reverse_bits(
                            2 * j, 2 if i == 0 else min(LOG2_AW, 2 + i, 2 * LOG2_AW - i)
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

        @df.kernel(mapping=[1], args=[output_buffer])
        def output(local_output: Ty[AH, AW]):
            """Output buffer kernel.

            Collects reduced/reordered results from BIRRD final stage.
            Writes directly to output tensor - this is where Allo
            produces the final computation results.
            """
            for d in range(AH):
                with allo.meta_for(AW) as i:
                    local_output[d, i] = connection[P0, i].get()

    return top


def get_scheduled_feather_minisa(AW: int, AH: int, Ty: AlloType):
    """Create scheduled/optimized FEATHER+ with MINISA support.

    Applies HLS scheduling directives for improved performance:
    - Pipeline NEST computation loop
    - Partition arrays for parallel access

    Args:
        AW: Array width
        AH: Array height
        Ty: Data type

    Returns:
        Scheduled Allo dataflow design
    """
    s = df.customize(get_feather_minisa_top(AW, AH, Ty))

    # Pipeline the main NEST computation loop
    nest_loop = s.get_loops("NEST_0")["nest"]["i"]
    s.pipeline(nest_loop)

    # Partition arrays for parallel access
    s.partition("top:output_buffer", dim=1, factor=AW)
    s.partition("top:iActs", dim=1, factor=AH)
    s.partition("top:weights", dim=2, factor=AW)
    s.partition("top:weights", dim=3, factor=AH)

    return s


def build_feather_minisa_simulator(AW: int, AH: int, Ty: AlloType):
    """Build FEATHER+ MINISA simulator module.

    Args:
        AW: Array width
        AH: Array height
        Ty: Data type

    Returns:
        Callable Allo simulator module
    """
    top = get_feather_minisa_top(AW, AH, Ty)
    return df.build(top, target="simulator")


def build_feather_minisa_hls(
    AW: int,
    AH: int,
    Ty: AlloType,
    mode: str = "csim",
    project: str = None
):
    """Build FEATHER+ MINISA for HLS target.

    This function builds the FEATHER+ dataflow design for Vitis HLS,
    applying HLS-specific optimizations including array partitioning
    for parallel access.

    Args:
        AW: Array width (must be power of 2: 4, 8, or 16)
        AH: Array height (number of VN elements)
        Ty: Data type (e.g., int8, int32)
        mode: HLS mode - one of:
            - "csim": HLS C simulation via nanobind (fast validation)
            - "csyn": HLS synthesis (generates reports, no execution)
            - "sw_emu": Xilinx software emulation
            - "hw_emu": Xilinx hardware emulation
            - "hw": Full hardware synthesis and FPGA deployment
        project: Project directory for HLS files (default: ./hls_project)

    Returns:
        HLSModule or IPModule depending on mode:
        - csim: Returns IPModule (callable for execution)
        - csyn: Returns HLSModule (contains reports and hls_code)
        - sw_emu/hw_emu: Returns HLSModule (for Xilinx emulation)
        - hw: Returns HLSModule (for FPGA deployment)
    """
    if project is None:
        import os
        project = os.path.join(os.path.dirname(__file__), "hls_project")
        os.makedirs(project, exist_ok=True)

    top = get_feather_minisa_top(AW, AH, Ty)
    s = df.customize(top)

    # Apply HLS-specific optimizations for parallel array access
    # Note: We partition output_buffer and iActs on specific dimensions
    # Weights array is partitioned completely (dim=0) to enable full parallel access
    s.partition("top:output_buffer", dim=1, factor=AW)
    s.partition("top:iActs", dim=1, factor=AH)
    s.partition("top:weights", dim=0)  # Complete partition for all dimensions

    return s.build(target="vitis_hls", mode=mode, project=project)


def get_hls_code(AW: int, AH: int, Ty: AlloType, project: str = None) -> str:
    """Get generated HLS C code for inspection.

    This function generates HLS C code from the FEATHER+ dataflow design
    without executing it. Useful for inspecting generated code or
    debugging HLS synthesis issues.

    Args:
        AW: Array width
        AH: Array height
        Ty: Data type
        project: Project directory (default: ./hls_project)

    Returns:
        Generated HLS C code as a string
    """
    mod = build_feather_minisa_hls(AW, AH, Ty, mode="csyn", project=project)
    return mod.hls_code


# Pre-computed BIRRD instruction arrays for common configurations
# These represent standard reduction patterns and can be modified by MINISA OVN lowering

BIRRD_INST_AW4 = np.array([
    [PS, PS],
    [AR, AL],
    [SW, PS],
], dtype=np.int8)

BIRRD_INST_AW8 = np.array([
    [PS, PS, PS, PS],
    [PS, PS, PS, PS],
    [AR, AR, AL, AL],
    [SW, SW, SW, SW],
    [SW, PS, PS, SW],
    [PS, PS, PS, PS],
], dtype=np.int8)

BIRRD_INST_AW16 = np.array([
    [PS, SW, PS, SW, PS, SW, PS, SW],
    [PS, PS, SW, PS, PS, PS, SW, PS],
    [PS, PS, PS, PS, PS, PS, PS, PS],
    [AL, AL, AL, AL, AR, AR, AR, AR],
    [SW, SW, SW, SW, SW, SW, SW, SW],
    [PS, PS, PS, PS, PS, PS, PS, PS],
    [PS, PS, PS, PS, PS, PS, PS, PS],
    [PS, PS, PS, PS, PS, PS, PS, PS],
], dtype=np.int8)


def get_default_birrd_inst(AW: int) -> np.ndarray:
    """Get default BIRRD instruction array for given array width.

    These instructions implement standard reduction patterns.
    MINISA OVN lowering may generate different patterns for
    specific output layouts.

    Args:
        AW: Array width (4, 8, or 16)

    Returns:
        BIRRD instruction array [P0, P1]
    """
    if AW == 4:
        return BIRRD_INST_AW4.copy()
    elif AW == 8:
        return BIRRD_INST_AW8.copy()
    elif AW == 16:
        return BIRRD_INST_AW16.copy()
    else:
        raise ValueError(f"Unsupported array width: {AW}. Must be 4, 8, or 16.")


# ============================================================================
# Full-Matrix Execution Model (DEV009)
# ============================================================================
# Instead of Python calling Allo once per tile, the full-matrix model passes
# complete input matrices and an instruction list to a single Allo function
# that handles tiling, decode, compute, and accumulation internally.
# ============================================================================


def feather_full_matrix[
    Ty,
    M: int32, K: int32, N: int32,
    AW: int32, AH: int32,
    num_inst: int32,
    Mt: int32, P0: int32, P1: int32, P0p1: int32, HalfAW: int32,
](
    A: "Ty[M, K]",
    B: "Ty[K, N]",
    instructions: "int32[num_inst, 13]",
    birrd_inst: "int8[P0, P1]",
    route_left: "int32[P0, P1]",
    route_right: "int32[P0, P1]",
    output_col_map: "int32[Mt]",
    C: "int32[M, N]",
):
    """Full-matrix FEATHER+ with on-chip instruction decode.

    Accepts full matrices A[M,K], B[K,N] and a MINISA instruction list,
    handles tiling, crossbar reordering, NEST computation, BIRRD reduction,
    and output accumulation in a single invocation.

    Fully decodes all MINISA instruction fields:
    - SetIVNLayout order (0-5): permutes input crossbar addressing
    - SetWVNLayout order (0-5): permutes weight crossbar addressing
    - SetOVNLayout order (0-5): selects BIRRD pattern (precomputed)
    - SetMapping r0,c0,Gr,Gc,sr,sc: PE-to-WVN mapping for weight routing
    """
    # Tile-level buffers
    iActs: Ty[AH, AW]
    weights: Ty[AH, AW, AH]
    nest_out: Ty[AH, AW]
    birrd_buf: Ty[P0p1, AW]
    tile_out: Ty[AH, AW]

    # Layout order state (decoded from layout instructions)
    ivn_order: int32 = 0
    wvn_order: int32 = 0
    ovn_order: int32 = 0

    for inst_idx in range(num_inst):
        inst_type: int32 = instructions[inst_idx, 0]

        if inst_type == 0:  # SetIVNLayout — decode order
            ivn_order = instructions[inst_idx, 1]
        elif inst_type == 1:  # SetWVNLayout — decode order
            wvn_order = instructions[inst_idx, 1]
        elif inst_type == 2:  # SetOVNLayout — decode order
            ovn_order = instructions[inst_idx, 1]
        elif inst_type == 3:  # SetMapping — trigger tile execution
            # Decode PE mapping fields
            pe_r0: int32 = instructions[inst_idx, 1]
            pe_c0: int32 = instructions[inst_idx, 2]
            pe_Gr: int32 = instructions[inst_idx, 3]
            pe_Gc: int32 = instructions[inst_idx, 4]
            pe_sr: int32 = instructions[inst_idx, 5]
            pe_sc: int32 = instructions[inst_idx, 6]
            # Decode tile bounds
            m_start: int32 = instructions[inst_idx, 7]
            n_start: int32 = instructions[inst_idx, 9]
            k_start: int32 = instructions[inst_idx, 11]

            # === Input Crossbar (IVN order-dependent reordering) ===
            # 3 addressing components: D0=ic_i, D1=ic_j%Mt, D2=ic_j//Mt
            # Order permutes how D0,D1,D2 map to m_idx and k_idx
            for ic_i in range(AH):
                for ic_j in range(AW):
                    m_idx: int32 = 0
                    k_idx: int32 = 0
                    if ivn_order == 0:  # ORDER_012
                        m_idx = m_start + (ic_j % Mt)
                        k_idx = k_start + ic_i + (ic_j // Mt) * AH
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
                    iActs[ic_i, ic_j] = A[m_idx, k_idx]

            # === Weight Crossbar (WVN order-dependent reordering) ===
            # PE mapping fields are decoded above (pe_r0..pe_sc) for
            # hardware configuration. The crossbar addressing uses the
            # natural tile components for weight staging:
            # D0=wc_k (inner K), D1=wc_w//HalfAW (K block 0/1), D2=wc_i (N pos)
            # WVN order permutes how D0,D1,D2 map to wk_idx and wn_idx
            for wc_i in range(AH):
                for wc_w in range(AW):
                    for wc_k in range(AH):
                        wk_idx: int32 = 0
                        wn_idx: int32 = 0
                        if wvn_order == 0:  # ORDER_012
                            wk_idx = k_start + wc_k + (wc_w // HalfAW) * AH
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
                        weights[wc_i, wc_w, wc_k] = B[wk_idx, wn_idx]

            # === NEST: AH x AW PE array ===
            for ni in range(AH):
                for nj in range(AW):
                    temp: Ty = 0
                    for nk in range(AH):
                        temp += iActs[nk, nj] * weights[ni, nj, nk]
                    nest_out[ni, nj] = temp

            # === BIRRD: butterfly reduction, AH iterations ===
            for ah in range(AH):
                # Initialize stage 0 from NEST output row
                for bp in range(AW):
                    birrd_buf[0, bp] = nest_out[ah, bp]

                # Process through P0 stages
                for stage in range(P0):
                    for sw in range(P1):
                        left_in: Ty = birrd_buf[stage, 2 * sw]
                        right_in: Ty = birrd_buf[stage, 2 * sw + 1]
                        op: int8 = birrd_inst[stage, sw]

                        left_out: Ty = 0
                        right_out: Ty = 0
                        if op == 0:  # PS: pass
                            left_out = left_in
                            right_out = right_in
                        elif op == 1:  # AR: add-right
                            left_out = left_in
                            right_out = left_in + right_in
                        elif op == 2:  # AL: add-left
                            left_out = left_in + right_in
                            right_out = right_in
                        else:  # SW: swap
                            left_out = right_in
                            right_out = left_in

                        left_dest: int32 = route_left[stage, sw]
                        right_dest: int32 = route_right[stage, sw]
                        birrd_buf[stage + 1, left_dest] = left_out
                        birrd_buf[stage + 1, right_dest] = right_out

                # Store BIRRD output for this ah iteration
                for bp2 in range(AW):
                    tile_out[ah, bp2] = birrd_buf[P0, bp2]

            # === Output Buffer: accumulate into C[M,N] ===
            for om in range(Mt):
                col: int32 = output_col_map[om]
                for on in range(AH):
                    C[m_start + om, n_start + on] = (
                        C[m_start + om, n_start + on] + tile_out[on, col]
                    )


class FeatherFullMatrixModule:
    """Wrapper that provides clean (A, B, instructions, C) interface.

    Precomputes all BIRRD configs at build time for all 6 OVN orders.
    At call time, extracts OVN order from instructions and passes the
    correct config to the internal Allo module.
    """

    def __init__(self, allo_mod, AW):
        from minisa.lowering import (
            compute_birrd_routing_table,
            lower_ovn_layout,
            compute_output_col_map,
        )
        from minisa.isa import SetOVNLayout

        self._mod = allo_mod
        self._AW = AW
        # Route tables are fixed for this AW (hardware topology)
        self._route_left, self._route_right = compute_birrd_routing_table(AW)
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
        self._mod(
            A, B, instructions,
            self._birrd_tables[ovn_order],
            self._route_left, self._route_right,
            self._col_maps[ovn_order],
            C,
        )


def build_feather_full_matrix_simulator(M, K, N, AW, AH, Ty, num_inst):
    """Build full-matrix FEATHER+ for LLVM simulation.

    Returns a module with clean interface: mod(A, B, instructions, C).
    BIRRD configs are precomputed internally for all OVN orders.
    """
    Mt = int(AW // 2)
    P0, P1 = compute_birrd_params(AW)
    P0p1 = int(P0 + 1)
    HalfAW = int(AW // 2)
    s = allo.customize(
        feather_full_matrix,
        instantiate=[Ty, int(M), int(K), int(N), int(AW), int(AH),
                     int(num_inst), Mt, int(P0), int(P1), P0p1, HalfAW],
    )
    allo_mod = s.build(target="llvm")
    return FeatherFullMatrixModule(allo_mod, AW)


def build_feather_full_matrix_hls(M, K, N, AW, AH, Ty, num_inst,
                                   mode="csim", project=None):
    """Build full-matrix FEATHER+ for Vitis HLS.

    Returns a module with clean interface: mod(A, B, instructions, C).
    BIRRD configs are precomputed internally for all OVN orders.
    """
    Mt = int(AW // 2)
    P0, P1 = compute_birrd_params(AW)
    P0p1 = int(P0 + 1)
    HalfAW = int(AW // 2)
    if project is None:
        project = "feather_full_matrix.prj"
    s = allo.customize(
        feather_full_matrix,
        instantiate=[Ty, int(M), int(K), int(N), int(AW), int(AH),
                     int(num_inst), Mt, int(P0), int(P1), P0p1, HalfAW],
    )
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
    AW, AH = 8, 8
    Ty = int8

    print("Building FEATHER+ MINISA simulator...")
    sim_mod = build_feather_minisa_simulator(AW, AH, Ty)

    # Test inputs
    np.random.seed(42)
    iActs = np.random.randint(-4, 4, size=(AH, AW)).astype(np.int8)
    weights = np.random.randint(-4, 4, size=(AH, AW, AH)).astype(np.int8)
    inst = get_default_birrd_inst(AW)
    output_buffer = np.zeros((AH, AW), dtype=np.int8)

    print("Executing FEATHER+ MINISA dataflow...")
    sim_mod(iActs, weights, inst, output_buffer)

    print(f"Input shape: {iActs.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Output shape: {output_buffer.shape}")
    print(f"Output:\n{output_buffer}")
    print("FEATHER+ MINISA verification complete!")
