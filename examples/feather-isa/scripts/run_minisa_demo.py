#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MINISA Demonstration Script

Demonstrates MINISA programming model for FEATHER+ accelerator:
1. Single-tile GEMM execution
2. Multi-tile GEMM with tiling
3. Layout switching between layers
4. Instruction count analysis

Usage:
    python run_minisa_demo.py [--verbose]
"""

import argparse
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minisa import (
    SetIVNLayout,
    SetWVNLayout,
    SetOVNLayout,
    SetMapping,
    MINISAInterpreter,
    MINISALowering,
)
from minisa.isa import FeatherPlusConfig, create_gemm_mapping
from minisa.layout import WVNLayout, LayoutOrder


def demo_single_tile_gemm(verbose: bool = False):
    """Demonstrate single-tile GEMM execution."""
    print("\n" + "=" * 60)
    print("DEMO 1: Single-Tile GEMM (4x4x4)")
    print("=" * 60)

    AH, AW = 4, 4
    M, K, N = 4, 4, 4

    config = FeatherPlusConfig(AH=AH, AW=AW)
    interp = MINISAInterpreter(config)

    # Create test matrices
    np.random.seed(42)
    A = np.array([
        [1, 2, 0, -1],
        [0, 1, 1, 2],
        [-1, 0, 2, 1],
        [1, -1, 1, 0]
    ], dtype=np.int8)

    B = np.array([
        [1, 0, 1, -1],
        [2, 1, 0, 1],
        [0, -1, 1, 2],
        [1, 1, -1, 0]
    ], dtype=np.int8)

    expected = np.matmul(A.astype(np.int32), B.astype(np.int32))

    print(f"\nInput A ({M}x{K}):")
    print(A)
    print(f"\nWeight B ({K}x{N}):")
    print(B)
    print(f"\nExpected C = A @ B ({M}x{N}):")
    print(expected)

    # Load data
    interp.load_inputs(A)
    interp.load_weights(B)
    interp.allocate_output(M, N)

    # Configure layouts
    print("\n--- MINISA Instructions ---")
    print("1. SetIVNLayout(order=0, M_L0=4, M_L1=1, J_L1=1)")
    interp.execute(SetIVNLayout(order=0, M_L0=4, M_L1=1, J_L1=1))

    print("2. SetWVNLayout(order=0, N_L0=4, N_L1=1, K_L1=1)")
    interp.execute(SetWVNLayout(order=0, N_L0=4, N_L1=1, K_L1=1))

    print("3. SetOVNLayout(order=0, P_L0=4, P_L1=1, Q_L1=1)")
    interp.execute(SetOVNLayout(order=0, P_L0=4, P_L1=1, Q_L1=1))

    print("4. SetMapping(r0=0, c0=0, G_r=4, G_c=4, s_r=0, s_c=1)")
    interp.execute(SetMapping(r0=0, c0=0, G_r=4, G_c=4, s_r=0, s_c=1))

    result = interp.get_output()
    print(f"\nComputed C:")
    print(result)

    # Verify
    match = np.array_equal(result, expected)
    print(f"\nCorrectness: {'PASS' if match else 'FAIL'}")

    counts = interp.get_instruction_counts()
    print(f"\nInstruction counts: {counts}")

    return match


def demo_multi_tile_gemm(verbose: bool = False):
    """Demonstrate multi-tile GEMM with K-dimension tiling."""
    print("\n" + "=" * 60)
    print("DEMO 2: Multi-Tile GEMM (8x16x8) with K-tiling")
    print("=" * 60)

    AH, AW = 8, 8
    M, K, N = 8, 16, 8

    config = FeatherPlusConfig(AH=AH, AW=AW)
    interp = MINISAInterpreter(config)

    np.random.seed(123)
    A = np.random.randint(-2, 2, size=(M, K), dtype=np.int8)
    B = np.random.randint(-2, 2, size=(K, N), dtype=np.int8)
    expected = np.matmul(A.astype(np.int32), B.astype(np.int32))

    print(f"\nMatrix dimensions: A({M}x{K}) @ B({K}x{N}) = C({M}x{N})")
    print(f"Array dimensions: AH={AH}, AW={AW}")
    print(f"K-dimension tiling: K_L1 = {K // AH} tiles")

    interp.load_inputs(A)
    interp.load_weights(B)
    interp.allocate_output(M, N)

    print("\n--- MINISA Instructions ---")

    # Layouts with K_L1=2
    print("1. SetIVNLayout(order=0, M_L0=8, M_L1=1, J_L1=2)")
    interp.execute(SetIVNLayout(order=0, M_L0=8, M_L1=1, J_L1=2))

    print("2. SetWVNLayout(order=0, N_L0=8, N_L1=1, K_L1=2)")
    interp.execute(SetWVNLayout(order=0, N_L0=8, N_L1=1, K_L1=2))

    print("3. SetOVNLayout(order=0, P_L0=8, P_L1=1, Q_L1=1)")
    interp.execute(SetOVNLayout(order=0, P_L0=8, P_L1=1, Q_L1=1))

    # Two mapping instructions for K tiles
    print("4. SetMapping(r0=0, ...) - First K tile")
    interp.execute(SetMapping(r0=0, c0=0, G_r=8, G_c=8, s_r=0, s_c=1))

    print("5. SetMapping(r0=1, ...) - Second K tile (accumulates)")
    interp.execute(SetMapping(r0=1, c0=0, G_r=8, G_c=8, s_r=0, s_c=1))

    result = interp.get_output()

    match = np.array_equal(result, expected)
    print(f"\nCorrectness: {'PASS' if match else 'FAIL'}")

    counts = interp.get_instruction_counts()
    print(f"Instruction counts: {counts}")
    print(f"Total MINISA instructions: {counts['total_instructions']}")

    return match


def demo_layout_variations(verbose: bool = False):
    """Demonstrate different layout orderings."""
    print("\n" + "=" * 60)
    print("DEMO 3: Layout Order Variations")
    print("=" * 60)

    AH, AW = 4, 4
    K, N = 8, 8

    print("\nWVN Layout addressing with different orders:")
    print(f"Weight matrix: {K}x{N}, VN dimensions: K_L1={K//AH}, N_L0={AW}, N_L1={N//AW}")
    print()

    order_names = {
        0: "k_L1 -> n_L0 -> n_L1",
        1: "k_L1 -> n_L1 -> n_L0",
        2: "n_L0 -> k_L1 -> n_L1",
        3: "n_L0 -> n_L1 -> k_L1",
        4: "n_L1 -> k_L1 -> n_L0",
        5: "n_L1 -> n_L0 -> k_L1",
    }

    for order in [0, 2, 4]:
        layout = WVNLayout(
            order=LayoutOrder(order),
            AW=AW,
            N_L0=4,
            N_L1=2,
            K_L1=2
        )

        print(f"Order {order} ({order_names[order]}):")

        # Show VN to buffer mapping for first few VNs
        for r in range(2):
            for c in range(4):
                buf_row, buf_col = layout.vn_to_buffer_addr(r, c)
                print(f"  WVN({r},{c}) -> buffer[{buf_row}][{buf_col}]")
        print()


def demo_pe_mapping(verbose: bool = False):
    """Demonstrate PE-to-VN mapping computation."""
    print("\n" + "=" * 60)
    print("DEMO 4: PE-to-WVN Mapping")
    print("=" * 60)

    AH, AW = 4, 4

    print("\nMapping with G_r=2 (2:1 reduction groups):")
    mapping = SetMapping(r0=0, c0=0, G_r=2, G_c=4, s_r=0, s_c=1)
    pe_map = mapping.get_pe_to_wvn_map(AH, AW)

    print(f"\nParameters: r0={mapping.r0}, c0={mapping.c0}, G_r={mapping.G_r}, G_c={mapping.G_c}")
    print(f"Formula: r(ah,aw) = r0 + floor(aw/G_r), c(ah,aw) = c0 + s_r*ah + s_c*(aw mod G_c)")
    print()

    # Display mapping as grid
    print("PE(ah,aw) -> WVN(r,c):")
    for ah in range(AH):
        row_str = f"  ah={ah}: "
        for aw in range(AW):
            wvn_r, wvn_c = pe_map[(ah, aw)]
            row_str += f"({wvn_r},{wvn_c}) "
        print(row_str)

    print("\nReduction groups (columns with same WVN row):")
    from collections import defaultdict
    groups = defaultdict(list)
    for aw in range(AW):
        wvn_r = pe_map[(0, aw)][0]
        groups[wvn_r].append(aw)

    for wvn_r, cols in sorted(groups.items()):
        print(f"  WVN row {wvn_r}: PE columns {cols}")


def demo_birrd_config(verbose: bool = False):
    """Demonstrate BIRRD configuration generation."""
    print("\n" + "=" * 60)
    print("DEMO 5: BIRRD Configuration")
    print("=" * 60)

    from minisa.lowering import generate_birrd_for_gemm, PS, AR, AL, SW

    op_names = {PS: 'PS', AR: 'AR', AL: 'AL', SW: 'SW'}

    for AW in [4, 8]:
        print(f"\nBIRRD for AW={AW} (2:1 reduction):")
        birrd = generate_birrd_for_gemm(AW, reduction_ratio=2)
        P0, P1 = birrd.shape

        print(f"  Stages (P0): {P0}")
        print(f"  Switches/stage (P1): {P1}")
        print(f"  Instruction matrix:")

        for stage in range(P0):
            row_str = f"    Stage {stage}: ["
            row_str += ", ".join(op_names[birrd[stage, sw]] for sw in range(P1))
            row_str += "]"
            print(row_str)


def demo_lowering(verbose: bool = False):
    """Demonstrate MINISA to FEATHER lowering."""
    print("\n" + "=" * 60)
    print("DEMO 6: MINISA Lowering to FEATHER Control")
    print("=" * 60)

    AH, AW = 4, 4
    config = FeatherPlusConfig(AH=AH, AW=AW)
    lowering = MINISALowering(config)

    mapping = SetMapping(r0=0, c0=0, G_r=2, G_c=4, s_r=0, s_c=1)

    print(f"\nLowering SetMapping instruction:")
    print(f"  r0={mapping.r0}, c0={mapping.c0}, G_r={mapping.G_r}, G_c={mapping.G_c}")

    tile_config = lowering.lower_mapping(mapping)

    print(f"\nGenerated TileConfig:")
    print(f"  Reduction groups: {tile_config.reduction_groups}")
    print(f"  BIRRD stages: {tile_config.birrd.num_stages}")
    print(f"  BIRRD switches/stage: {tile_config.birrd.switches_per_stage}")

    print(f"\n  Input crossbar config:")
    for aw, ivns in tile_config.input_crossbar.items():
        print(f"    PE col {aw} <- IVNs: {ivns}")

    print(f"\n  Weight crossbar config:")
    for aw, wvns in tile_config.weight_crossbar.items():
        print(f"    PE col {aw} <- WVNs: {wvns}")


def demo_instruction_efficiency(verbose: bool = False):
    """Demonstrate instruction count efficiency."""
    print("\n" + "=" * 60)
    print("DEMO 7: Instruction Count Analysis")
    print("=" * 60)

    AH, AW = 8, 8

    print("\nMINISA instruction counts for various GEMM sizes:")
    print("-" * 50)
    print(f"{'Size':<20} {'Layout':<10} {'Mapping':<10} {'Total':<10}")
    print("-" * 50)

    test_sizes = [
        (8, 8, 8),
        (8, 16, 8),
        (16, 8, 16),
        (16, 16, 16),
        (32, 32, 32),
    ]

    for M, K, N in test_sizes:
        config = FeatherPlusConfig(AH=AH, AW=AW)
        interp = MINISAInterpreter(config)
        interp.allocate_output(M, N)

        # Calculate tiling
        K_tiles = (K + AH - 1) // AH
        M_tiles = (M + AW - 1) // AW
        N_tiles = (N + AW - 1) // AW

        # Execute layouts
        interp.execute(SetIVNLayout(order=0, M_L0=min(AW, M), M_L1=M_tiles, J_L1=K_tiles))
        interp.execute(SetWVNLayout(order=0, N_L0=min(AW, N), N_L1=N_tiles, K_L1=K_tiles))
        interp.execute(SetOVNLayout(order=0, P_L0=min(AW, M), P_L1=M_tiles, Q_L1=N_tiles))

        # Execute mappings for all tiles
        for k_tile in range(K_tiles):
            for m_tile in range(M_tiles):
                for n_tile in range(N_tiles):
                    interp.execute(SetMapping(
                        r0=k_tile,
                        c0=n_tile * min(AW, N),
                        G_r=AW,
                        G_c=AW,
                        s_r=m_tile,
                        s_c=1
                    ))

        counts = interp.get_instruction_counts()
        size_str = f"{M}x{K}x{N}"
        print(f"{size_str:<20} {counts['layout_instructions']:<10} {counts['mapping_instructions']:<10} {counts['total_instructions']:<10}")

    print("-" * 50)
    print("\nNote: MINISA achieves O(1) layout instructions per layer,")
    print("with mapping instructions scaling with number of tiles.")


def main():
    parser = argparse.ArgumentParser(description='MINISA Demonstration Script')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--demo', type=int, choices=range(1, 8),
                       help='Run specific demo (1-7)')
    args = parser.parse_args()

    demos = [
        demo_single_tile_gemm,
        demo_multi_tile_gemm,
        demo_layout_variations,
        demo_pe_mapping,
        demo_birrd_config,
        demo_lowering,
        demo_instruction_efficiency,
    ]

    print("=" * 60)
    print("MINISA: Minimal Instruction Set Architecture for FEATHER+")
    print("=" * 60)

    if args.demo:
        demos[args.demo - 1](args.verbose)
    else:
        all_pass = True
        for demo in demos:
            result = demo(args.verbose)
            if result is False:
                all_pass = False

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        if all_pass:
            print("All demos completed successfully.")
        else:
            print("Some demos had failures. Check output above.")


if __name__ == '__main__':
    main()
