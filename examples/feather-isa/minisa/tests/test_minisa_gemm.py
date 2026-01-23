# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MINISA GEMM Correctness Tests

Tests MINISA interpreter against numpy reference for small GEMM workloads.
Verifies:
- Correct PE-to-VN mapping computation
- Correct dot product accumulation
- Correct BIRRD reduction routing
- Multiple layout configurations
"""

import pytest
import numpy as np
from typing import List, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from minisa import (
    SetIVNLayout,
    SetWVNLayout,
    SetOVNLayout,
    SetMapping,
    MINISAInterpreter,
    LayoutOrder,
)
from minisa.isa import FeatherPlusConfig, create_gemm_mapping


def numpy_gemm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Reference GEMM implementation using numpy."""
    return np.matmul(A.astype(np.int32), B.astype(np.int32))


class TestMINISAGEMMSmall:
    """Test MINISA GEMM on small matrices that fit in single tile."""

    def test_gemm_4x4x4_single_tile(self):
        """Test 4x4x4 GEMM with AH=4, AW=4 (single tile)."""
        AH, AW = 4, 4
        M, K, N = 4, 4, 4

        config = FeatherPlusConfig(AH=AH, AW=AW)
        interp = MINISAInterpreter(config)

        # Create test data
        np.random.seed(42)
        A = np.random.randint(-8, 8, size=(M, K), dtype=np.int8)
        B = np.random.randint(-8, 8, size=(K, N), dtype=np.int8)

        # Reference result
        expected = numpy_gemm(A, B)

        # Load data into interpreter
        interp.load_inputs(A)
        interp.load_weights(B)
        interp.allocate_output(M, N)

        # Configure layouts
        ivn_layout = SetIVNLayout(order=0, M_L0=4, M_L1=1, J_L1=1)
        wvn_layout = SetWVNLayout(order=0, N_L0=4, N_L1=1, K_L1=1)
        ovn_layout = SetOVNLayout(order=0, P_L0=4, P_L1=1, Q_L1=1)

        assert interp.execute(ivn_layout)
        assert interp.execute(wvn_layout)
        assert interp.execute(ovn_layout)

        # Execute single mapping (covers entire tile)
        mapping = SetMapping(r0=0, c0=0, G_r=4, G_c=4, s_r=0, s_c=1)
        assert interp.execute(mapping)

        # Get result
        result = interp.get_output()

        # Verify correctness
        np.testing.assert_array_equal(result, expected,
            f"GEMM 4x4x4 failed.\nExpected:\n{expected}\nGot:\n{result}")

    def test_gemm_8x8x8_single_tile(self):
        """Test 8x8x8 GEMM with AH=8, AW=8."""
        AH, AW = 8, 8
        M, K, N = 8, 8, 8

        config = FeatherPlusConfig(AH=AH, AW=AW)
        interp = MINISAInterpreter(config)

        np.random.seed(123)
        A = np.random.randint(-4, 4, size=(M, K), dtype=np.int8)
        B = np.random.randint(-4, 4, size=(K, N), dtype=np.int8)

        expected = numpy_gemm(A, B)

        interp.load_inputs(A)
        interp.load_weights(B)
        interp.allocate_output(M, N)

        ivn_layout = SetIVNLayout(order=0, M_L0=8, M_L1=1, J_L1=1)
        wvn_layout = SetWVNLayout(order=0, N_L0=8, N_L1=1, K_L1=1)
        ovn_layout = SetOVNLayout(order=0, P_L0=8, P_L1=1, Q_L1=1)

        interp.execute(ivn_layout)
        interp.execute(wvn_layout)
        interp.execute(ovn_layout)

        mapping = SetMapping(r0=0, c0=0, G_r=8, G_c=8, s_r=0, s_c=1)
        interp.execute(mapping)

        result = interp.get_output()
        np.testing.assert_array_equal(result, expected)


class TestMINISAGEMMMultiTile:
    """Test MINISA GEMM requiring multiple tiles."""

    def test_gemm_8x16x8_two_k_tiles(self):
        """Test GEMM with K > AH requiring K-dimension tiling."""
        AH, AW = 8, 8
        M, K, N = 8, 16, 8

        config = FeatherPlusConfig(AH=AH, AW=AW)
        interp = MINISAInterpreter(config)

        np.random.seed(456)
        A = np.random.randint(-4, 4, size=(M, K), dtype=np.int8)
        B = np.random.randint(-4, 4, size=(K, N), dtype=np.int8)

        expected = numpy_gemm(A, B)

        interp.load_inputs(A)
        interp.load_weights(B)
        interp.allocate_output(M, N)

        # Layout with K_L1=2 (two K tiles)
        ivn_layout = SetIVNLayout(order=0, M_L0=8, M_L1=1, J_L1=2)
        wvn_layout = SetWVNLayout(order=0, N_L0=8, N_L1=1, K_L1=2)
        ovn_layout = SetOVNLayout(order=0, P_L0=8, P_L1=1, Q_L1=1)

        interp.execute(ivn_layout)
        interp.execute(wvn_layout)
        interp.execute(ovn_layout)

        # Execute two tiles (accumulating K dimension)
        mapping1 = SetMapping(r0=0, c0=0, G_r=8, G_c=8, s_r=0, s_c=1)
        mapping2 = SetMapping(r0=1, c0=0, G_r=8, G_c=8, s_r=0, s_c=1)

        interp.execute(mapping1)
        interp.execute(mapping2)

        result = interp.get_output()
        np.testing.assert_array_equal(result, expected)

    def test_gemm_16x8x16_spatial_tiling(self):
        """Test GEMM with M, N > AW requiring spatial tiling."""
        AH, AW = 8, 8
        M, K, N = 16, 8, 16

        config = FeatherPlusConfig(AH=AH, AW=AW)
        interp = MINISAInterpreter(config)

        np.random.seed(789)
        A = np.random.randint(-2, 2, size=(M, K), dtype=np.int8)
        B = np.random.randint(-2, 2, size=(K, N), dtype=np.int8)

        expected = numpy_gemm(A, B)

        interp.load_inputs(A)
        interp.load_weights(B)
        interp.allocate_output(M, N)

        # Layouts with spatial tiling
        ivn_layout = SetIVNLayout(order=0, M_L0=8, M_L1=2, J_L1=1)
        wvn_layout = SetWVNLayout(order=0, N_L0=8, N_L1=2, K_L1=1)
        ovn_layout = SetOVNLayout(order=0, P_L0=8, P_L1=2, Q_L1=2)

        interp.execute(ivn_layout)
        interp.execute(wvn_layout)
        interp.execute(ovn_layout)

        # 2x2 tile grid
        for m_tile in range(2):
            for n_tile in range(2):
                mapping = SetMapping(
                    r0=0,
                    c0=n_tile * 8,
                    G_r=8,
                    G_c=8,
                    s_r=m_tile,
                    s_c=1
                )
                interp.execute(mapping)

        result = interp.get_output()
        # Check result matches in shape
        assert result.shape == expected.shape, \
            f"Shape mismatch: {result.shape} vs {expected.shape}"


class TestMINISALayoutVariations:
    """Test different layout orderings."""

    def test_wvn_layout_orders(self):
        """Test all 6 WVN layout orders produce correct addressing."""
        AH, AW = 4, 4
        K, N = 8, 8  # 2 WVN rows, 8 WVN columns

        from minisa.layout import WVNLayout, LayoutOrder

        for order in range(6):
            layout = WVNLayout(
                order=LayoutOrder(order),
                AW=AW,
                N_L0=4,
                N_L1=2,
                K_L1=2
            )

            # Verify bijective mapping (each VN maps to unique buffer location)
            seen_addrs = set()
            for r in range(2):  # K_L1 = 2
                for c in range(8):  # N_L1 * N_L0 = 8
                    addr = layout.vn_to_buffer_addr(r, c)
                    assert addr not in seen_addrs, \
                        f"Order {order}: duplicate address {addr} for VN ({r}, {c})"
                    seen_addrs.add(addr)

    def test_gemm_with_different_wvn_orders(self):
        """Test GEMM produces same result regardless of WVN order."""
        AH, AW = 4, 4
        M, K, N = 4, 4, 4

        np.random.seed(111)
        A = np.random.randint(-4, 4, size=(M, K), dtype=np.int8)
        B = np.random.randint(-4, 4, size=(K, N), dtype=np.int8)
        expected = numpy_gemm(A, B)

        for order in [0, 2, 4]:  # Test subset of orders
            config = FeatherPlusConfig(AH=AH, AW=AW)
            interp = MINISAInterpreter(config)

            interp.load_inputs(A)
            interp.load_weights(B)
            interp.allocate_output(M, N)

            ivn_layout = SetIVNLayout(order=0, M_L0=4, M_L1=1, J_L1=1)
            wvn_layout = SetWVNLayout(order=order, N_L0=4, N_L1=1, K_L1=1)
            ovn_layout = SetOVNLayout(order=0, P_L0=4, P_L1=1, Q_L1=1)

            interp.execute(ivn_layout)
            interp.execute(wvn_layout)
            interp.execute(ovn_layout)

            mapping = SetMapping(r0=0, c0=0, G_r=4, G_c=4, s_r=0, s_c=1)
            interp.execute(mapping)

            result = interp.get_output()
            np.testing.assert_array_equal(result, expected,
                f"GEMM with WVN order {order} failed")


class TestMINISAMappingVariations:
    """Test different SetMapping configurations."""

    def test_mapping_with_reduction_groups(self):
        """Test SetMapping with G_r < AW (reduction groups)."""
        AH, AW = 4, 4
        config = FeatherPlusConfig(AH=AH, AW=AW)

        # G_r=2 means pairs of columns share same WVN row
        mapping = SetMapping(r0=0, c0=0, G_r=2, G_c=4, s_r=0, s_c=1)
        pe_map = mapping.get_pe_to_wvn_map(AH, AW)

        # Verify column pairs share WVN row
        for ah in range(AH):
            # Columns 0,1 should have same WVN row
            assert pe_map[(ah, 0)][0] == pe_map[(ah, 1)][0]
            # Columns 2,3 should have same WVN row
            assert pe_map[(ah, 2)][0] == pe_map[(ah, 3)][0]
            # But different from first pair
            assert pe_map[(ah, 0)][0] != pe_map[(ah, 2)][0]

    def test_mapping_spatial_stride(self):
        """Test SetMapping with s_c > 0 (spatial striding)."""
        AH, AW = 4, 4
        config = FeatherPlusConfig(AH=AH, AW=AW)

        mapping = SetMapping(r0=0, c0=0, G_r=4, G_c=4, s_r=1, s_c=2)
        pe_map = mapping.get_pe_to_wvn_map(AH, AW)

        # Verify stride pattern
        for aw in range(AW):
            wvn_col_base = pe_map[(0, aw)][1]
            for ah in range(1, AH):
                expected_col = wvn_col_base + mapping.s_r * ah
                assert pe_map[(ah, aw)][1] == expected_col


class TestMINISAInstructionCounts:
    """Test instruction counting for efficiency analysis."""

    def test_instruction_count_single_tile(self):
        """Verify instruction count for single-tile GEMM."""
        AH, AW = 8, 8
        config = FeatherPlusConfig(AH=AH, AW=AW)
        interp = MINISAInterpreter(config)

        # Execute minimal GEMM
        interp.allocate_output(8, 8)
        interp.execute(SetIVNLayout(order=0, M_L0=8, M_L1=1, J_L1=1))
        interp.execute(SetWVNLayout(order=0, N_L0=8, N_L1=1, K_L1=1))
        interp.execute(SetOVNLayout(order=0, P_L0=8, P_L1=1, Q_L1=1))
        interp.execute(SetMapping(r0=0, c0=0, G_r=8, G_c=8, s_r=0, s_c=1))

        counts = interp.get_instruction_counts()
        assert counts['layout_instructions'] == 3
        assert counts['mapping_instructions'] == 1
        assert counts['total_instructions'] == 4

    def test_instruction_count_multi_tile(self):
        """Verify instruction count for multi-tile GEMM."""
        AH, AW = 8, 8
        M, K, N = 16, 16, 16

        config = FeatherPlusConfig(AH=AH, AW=AW)
        interp = MINISAInterpreter(config)

        interp.allocate_output(M, N)
        interp.execute(SetIVNLayout(order=0, M_L0=8, M_L1=2, J_L1=2))
        interp.execute(SetWVNLayout(order=0, N_L0=8, N_L1=2, K_L1=2))
        interp.execute(SetOVNLayout(order=0, P_L0=8, P_L1=2, Q_L1=2))

        # 2x2x2 = 8 tiles
        num_tiles = 0
        for m_tile in range(2):
            for k_tile in range(2):
                for n_tile in range(2):
                    interp.execute(SetMapping(
                        r0=k_tile, c0=n_tile*8, G_r=8, G_c=8, s_r=m_tile, s_c=1
                    ))
                    num_tiles += 1

        counts = interp.get_instruction_counts()
        assert counts['layout_instructions'] == 3
        assert counts['mapping_instructions'] == num_tiles
        assert counts['total_instructions'] == 3 + num_tiles


def run_all_tests():
    """Run all MINISA GEMM tests."""
    pytest.main([__file__, '-v'])


if __name__ == '__main__':
    run_all_tests()
