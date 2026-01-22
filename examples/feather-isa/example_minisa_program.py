#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Example: Programming FEATHER+ with MINISA Instructions

This example demonstrates how to create and execute a MINISA program
for matrix multiplication at Virtual Neuron (VN) granularity.

The key insight: VN-level programming is the right abstraction that:
1. Preserves inter-PE mapping flexibility
2. Avoids unnecessary control overhead
3. Achieves orders of magnitude reduction in instruction footprint
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from feather_isa import (
    FEATHER_ISA,
    SetIVNLayout,
    SetWVNLayout,
    SetOVNLayout,
    SetMapping,
    create_minisa_program_gemm,
    print_minisa_program,
)


def example_1_basic_gemm():
    """
    Example 1: Basic GEMM with MINISA
    Computes C = A @ B where A: (8, 8), B: (8, 8), C: (8, 8)
    """
    print("=" * 80)
    print("Example 1: Basic GEMM with MINISA")
    print("=" * 80)

    # Problem size
    M, N, K = 8, 8, 8
    AH, AW = 4, 4  # PE array configuration

    print(f"\nProblem: C[{M}, {N}] = A[{M}, {K}] @ B[{K}, {N}]")
    print(f"PE Array: {AH}×{AW} (VN size = {AH})")

    # Create input data
    np.random.seed(42)
    A = np.random.randint(-4, 4, size=(M, K)).astype(np.int8)
    B = np.random.randint(-4, 4, size=(K, N)).astype(np.int8)

    # Initialize FEATHER-ISA accelerator
    feather = FEATHER_ISA(AH, AW)

    # Create MINISA program
    print("\n1. Creating MINISA program...")
    ivn_layout = SetIVNLayout(
        order=0,
        ML0=AH, ML1=M//AH,  # Partition M = ML1 * ML0
        JL0=AH, JL1=K//AH   # Partition K = JL1 * JL0
    )

    wvn_layout = SetWVNLayout(
        order=0,
        KL0=AH, KL1=K//AH,  # Partition K = KL1 * KL0
        NL0=AW, NL1=N//AW   # Partition N = NL1 * NL0
    )

    ovn_layout = SetOVNLayout(
        order=0,
        PL0=AH, PL1=M//AH,  # Partition M = PL1 * PL0
        QL0=AH, QL1=N//AH   # Partition N = QL1 * QL0
    )

    mapping = SetMapping(
        r0=0, c0=0,   # Start from WVN(0,0)
        Gr=1, Gc=AW,  # No row replication, full width replication group
        sr=0, sc=1    # No temporal stride, unit spatial stride
    )

    # Execute MINISA program
    print("\n2. Executing MINISA instructions...")
    feather.set_ivn_layout(ivn_layout)
    feather.set_wvn_layout(wvn_layout)
    feather.set_ovn_layout(ovn_layout)
    C = feather.execute_mapping(mapping, A, B)

    # Verify result
    C_ref = A @ B
    if np.array_equal(C, C_ref):
        print("\n✅ Result verified! Matrix multiplication correct.")
        print(f"   Sample: C[0,0] = {C[0,0]}, expected = {C_ref[0,0]}")
    else:
        print("\n❌ Result mismatch!")
        print(f"   Max error: {np.max(np.abs(C - C_ref))}")


def example_2_instruction_analysis():
    """
    Example 2: Analyzing MINISA Instruction Footprint

    This example demonstrates the dramatic reduction in instruction footprint
    achieved by MINISA compared to micro-instruction programming.
    """
    print("\n\n" + "=" * 80)
    print("Example 2: MINISA Instruction Footprint Analysis")
    print("=" * 80)

    # Test different array sizes
    sizes = [
        (4, 4),
        (8, 8),
        (16, 16),
        (32, 32),
    ]

    M, N, K = 16, 16, 16  # Fixed problem size

    print(f"\nProblem size: M={M}, N={N}, K={K}")
    print(f"{'AH×AW':<10} {'MINISA Insts':<15} {'Bytes/Inst':<15} {'Total Bytes':<15}")
    print("-" * 60)

    for AH, AW in sizes:
        if M % AH != 0 or N % AH != 0 or K % AH != 0:
            continue

        # Create MINISA program
        ivn, wvn, ovn, mappings = create_minisa_program_gemm(M, N, K, AH, AW)

        # Count instructions
        num_layout_insts = 3  # SetIVN, SetWVN, SetOVN
        num_mapping_insts = len(mappings)
        total_insts = num_layout_insts + num_mapping_insts

        # Estimate bytes (simplified)
        # Layout inst: ~3 bits for order + dimension parameters
        # Mapping inst: ~6 parameters × ~log2(max_dim) bits each
        bytes_per_layout = 2  # Simplified estimate
        bytes_per_mapping = 3  # Simplified estimate
        total_bytes = num_layout_insts * bytes_per_layout + num_mapping_insts * bytes_per_mapping

        print(f"{AH}×{AW:<7} {total_insts:<15} {total_bytes/total_insts:<15.1f} {total_bytes:<15}")

    print("\nKey insight: MINISA instruction count is independent of array size!")
    print("In contrast, micro-instructions grow O(AH * AW * cycles)")


def example_3_vn_level_computation():
    """
    Example 3: Understanding VN-level Computation

    This example shows how Virtual Neurons work at the hardware level.
    """
    print("\n\n" + "=" * 80)
    print("Example 3: Virtual Neuron (VN) Level Computation")
    print("=" * 80)

    AH = 4  # VN size
    print(f"\nVN Size: {AH} elements")
    print(f"Each VN performs one {AH}-way dot product")

    # Create a simple VN computation
    input_vn = np.array([1, 2, 3, 4], dtype=np.int8)
    weight_vn = np.array([2, 1, 3, 2], dtype=np.int8)

    # Manual VN computation
    output_scalar = 0
    print(f"\nComputing: output = input_vn · weight_vn")
    print(f"Input VN:  {input_vn}")
    print(f"Weight VN: {weight_vn}")
    print(f"\nStep-by-step:")

    for i in range(AH):
        product = input_vn[i] * weight_vn[i]
        output_scalar += product
        print(f"  [{i}] {input_vn[i]} × {weight_vn[i]} = {product}, sum = {output_scalar}")

    print(f"\nFinal result: {output_scalar}")
    print(f"Verification: {np.dot(input_vn, weight_vn)}")

    print("\n💡 Key insight: VN is the atomic unit of computation")
    print("   - One PE computes one VN per cycle")
    print("   - One SetMapping instruction controls all PEs")
    print("   - No need for per-cycle micro-control!")


def example_4_mapping_flexibility():
    """
    Example 4: Demonstrating Mapping Flexibility

    Shows how SetMapping parameters enable different PE mappings.
    """
    print("\n\n" + "=" * 80)
    print("Example 4: Mapping Flexibility with SetMapping")
    print("=" * 80)

    AH, AW = 4, 4

    # Create different mappings
    mappings = [
        SetMapping(r0=0, c0=0, Gr=1, Gc=4, sr=0, sc=1),  # Sequential columns
        SetMapping(r0=1, c0=0, Gr=2, Gc=4, sr=1, sc=1),  # With row offset and temporal stride
        SetMapping(r0=0, c0=4, Gr=1, Gc=4, sr=0, sc=2),  # Column offset with stride 2
    ]

    print(f"\nPE Array: {AH}×{AW}")
    print("\nDifferent mappings demonstrate flexibility:")

    for idx, mapping in enumerate(mappings):
        print(f"\n--- Mapping {idx+1} ---")
        print(f"Parameters: r0={mapping.r0}, c0={mapping.c0}, Gr={mapping.Gr}, Gc={mapping.Gc}, sr={mapping.sr}, sc={mapping.sc}")

        # Compute WVN indices for each PE
        indices = mapping.compute_wvn_indices(AH, AW)

        print(f"WVN(r,c) assigned to each PE:")
        for ah in range(AH):
            row_str = ""
            for aw in range(AW):
                r, c = indices[ah, aw]
                row_str += f"({r},{c}) "
            print(f"  PE row {ah}: {row_str}")

    print("\n💡 Key insight: One SetMapping instruction controls entire PE array")
    print("   - Parametric mapping enables diverse dataflows")
    print("   - No per-PE configuration needed!")


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("MINISA Programming Examples for FEATHER+")
    print("="*80)
    print("\nThese examples demonstrate the power of Virtual Neuron (VN) abstraction")
    print("for programming reconfigurable accelerators with minimal control overhead.\n")

    # Run examples
    example_1_basic_gemm()
    example_2_instruction_analysis()
    example_3_vn_level_computation()
    example_4_mapping_flexibility()

    # Final summary
    print("\n\n" + "="*80)
    print("Summary: Why MINISA Matters")
    print("="*80)
    print("""
MINISA achieves:
1. ✅ 10^5× reduction in instruction footprint (geometric mean)
2. ✅ Eliminates 98% instruction-fetch stalls in large arrays
3. ✅ Preserves full hardware flexibility for dataflow/layout
4. ✅ Enables larger on-chip tiles and higher arithmetic intensity

Key principle: VN-level abstraction is the sweet spot
- Coarser than VN: loses inter-PE mapping flexibility
- Finer than VN: introduces unnecessary control overhead

Result: Near-zero control overhead with maximum flexibility!
    """)
    print("="*80)


if __name__ == "__main__":
    main()
