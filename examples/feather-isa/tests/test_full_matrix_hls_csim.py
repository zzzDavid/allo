# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Full-matrix FEATHER+ HLS C simulation tests.

These tests verify that the full-matrix dataflow model works correctly
with the Vitis HLS C simulation backend. Tests are skipped when Vitis HLS
is not available.
"""

import os
import sys
import tempfile

import numpy as np

# Setup path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from allo.ir.types import int8
from allo.backend.hls import is_available

from minisa.isa import create_gemm_program, encode_program
from feather_minisa import (
    build_feather_full_matrix_simulator,
    build_feather_full_matrix_hls,
    run_full_matrix_gemm,
)

HLS_AVAILABLE = is_available("vitis_hls")


def test_hls_csim_gemm_8x8x16():
    """Test full-matrix GEMM 8x8x16 through HLS C simulation."""
    print("\n" + "=" * 60)
    print("Test: HLS CSim GEMM 8x8x16")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        C, ref, passed = run_full_matrix_gemm(
            M=8, N=8, K=16, AW=8, AH=8,
            verbose=True,
            build_target="vitis_hls",
            build_mode="csim",
            project_dir=tmpdir,
        )

    assert passed, "HLS csim GEMM 8x8x16 failed"
    print("PASSED: HLS CSim GEMM 8x8x16")
    return True


def test_hls_csim_gemm_16x8x32():
    """Test full-matrix GEMM 16x8x32 through HLS C simulation."""
    print("\n" + "=" * 60)
    print("Test: HLS CSim GEMM 16x8x32")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        C, ref, passed = run_full_matrix_gemm(
            M=16, N=8, K=32, AW=8, AH=8,
            verbose=True,
            build_target="vitis_hls",
            build_mode="csim",
            project_dir=tmpdir,
        )

    assert passed, "HLS csim GEMM 16x8x32 failed"
    print("PASSED: HLS CSim GEMM 16x8x32")
    return True


def test_hls_csim_gemm_16x16x32():
    """Test full-matrix GEMM 16x16x32 through HLS C simulation."""
    print("\n" + "=" * 60)
    print("Test: HLS CSim GEMM 16x16x32")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        C, ref, passed = run_full_matrix_gemm(
            M=16, N=16, K=32, AW=8, AH=8,
            verbose=True,
            build_target="vitis_hls",
            build_mode="csim",
            project_dir=tmpdir,
        )

    assert passed, "HLS csim GEMM 16x16x32 failed"
    print("PASSED: HLS CSim GEMM 16x16x32")
    return True


def test_hls_csim_ovn_orders():
    """Test all OVN orders produce correct GEMM via HLS csim."""
    print("\n" + "=" * 60)
    print("Test: HLS CSim OVN Orders")
    print("=" * 60)

    M, N, K, AW, AH = 8, 8, 16, 8, 8
    np.random.seed(42)
    A = np.random.randint(-4, 4, size=(M, K)).astype(np.int8)
    B = np.random.randint(-4, 4, size=(K, N)).astype(np.int8)
    ref = np.dot(A, B)

    for ovn_order in range(6):
        program = create_gemm_program(
            M=M, N=N, K=K, AH=AH, AW=AW, ovn_order=ovn_order,
        )
        instructions = encode_program(program)

        with tempfile.TemporaryDirectory() as tmpdir:
            mod = build_feather_full_matrix_hls(
                M, K, N, AW, AH, int8, len(instructions),
                mode="csim", project=tmpdir,
            )
            C = np.zeros((M, N), dtype=np.int32)
            mod(A, B, instructions, C)

        np.testing.assert_allclose(C, ref, atol=1e-5)
        print(f"  OVN order={ovn_order}: matches numpy reference")

    print("PASSED: HLS CSim OVN orders")
    return True


def test_hls_csim_matches_simulator():
    """Verify HLS csim produces identical results to LLVM simulator."""
    print("\n" + "=" * 60)
    print("Test: HLS CSim Matches Simulator")
    print("=" * 60)

    M, N, K, AW, AH = 16, 16, 32, 8, 8
    np.random.seed(42)
    A = np.random.randint(-4, 4, size=(M, K)).astype(np.int8)
    B = np.random.randint(-4, 4, size=(K, N)).astype(np.int8)

    program = create_gemm_program(M=M, N=N, K=K, AH=AH, AW=AW)
    instructions = encode_program(program)
    num_inst = len(instructions)

    # Run with LLVM simulator
    sim_mod = build_feather_full_matrix_simulator(M, K, N, AW, AH, int8, num_inst)
    C_sim = np.zeros((M, N), dtype=np.int32)
    sim_mod(A, B, instructions, C_sim)

    # Run with HLS csim
    with tempfile.TemporaryDirectory() as tmpdir:
        hls_mod = build_feather_full_matrix_hls(
            M, K, N, AW, AH, int8, num_inst,
            mode="csim", project=tmpdir,
        )
        C_hls = np.zeros((M, N), dtype=np.int32)
        hls_mod(A, B, instructions, C_hls)

    np.testing.assert_array_equal(C_sim, C_hls)
    print(f"  Simulator output sum: {C_sim.sum()}")
    print(f"  HLS csim output sum: {C_hls.sum()}")
    print("  Results match exactly")
    print("PASSED: HLS CSim matches simulator")
    return True


def run_hls_csim_tests():
    """Run all full-matrix HLS csim tests."""
    print("=" * 70)
    print("FULL-MATRIX HLS C SIMULATION TESTS")
    print("=" * 70)

    if not HLS_AVAILABLE:
        print("SKIPPED: Vitis HLS not available")
        return True

    results = {}

    tests = [
        ("HLS csim GEMM 8x8x16", test_hls_csim_gemm_8x8x16),
        ("HLS csim GEMM 16x8x32", test_hls_csim_gemm_16x8x32),
        ("HLS csim GEMM 16x16x32", test_hls_csim_gemm_16x16x32),
        ("HLS csim OVN orders", test_hls_csim_ovn_orders),
        ("HLS csim matches simulator", test_hls_csim_matches_simulator),
    ]

    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            import traceback
            print(f"FAILED: {name}")
            print(f"  Error: {e}")
            traceback.print_exc()
            results[name] = False

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll HLS csim tests PASSED")
    else:
        print("\nSome tests FAILED")

    return all_passed


if __name__ == "__main__":
    success = run_hls_csim_tests()
    sys.exit(0 if success else 1)
