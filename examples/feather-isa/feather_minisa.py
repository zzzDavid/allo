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
