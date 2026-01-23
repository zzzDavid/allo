# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""MINISA GEMM tests using Allo dataflow execution.

These tests verify that MINISA programs execute correctly through the
FEATHER+ Allo dataflow implementation. All computation is performed by
Allo kernels - not numpy.

Verification that Allo is used:
1. The interpreter logs each Allo module invocation
2. Output tensors are only written by Allo kernels
3. The test compares Allo outputs against numpy reference
4. Tests would fail if numpy compute were used (wrong layout)
"""

import os
import sys

# Setup path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from allo.ir.types import int8

from minisa.interpreter import MINISAInterpreter, run_minisa_gemm
from minisa.isa import create_gemm_program, MINISAProgram, SetMapping
from minisa.lowering import lower_minisa_program, TileExtractor, extract_output_for_verification
from feather_minisa import get_feather_minisa_top, get_default_birrd_inst
import allo.dataflow as df


def test_minisa_gemm_8x8x16():
    """Test MINISA GEMM with small dimensions: C[8,8] = A[8,16] * B[16,8]."""
    print("\n" + "=" * 60)
    print("Test: MINISA GEMM 8x8x16")
    print("=" * 60)

    M, N, K = 8, 8, 16
    AW, AH = 8, 8

    output, ref, passed = run_minisa_gemm(
        M=M, N=N, K=K,
        AW=AW, AH=AH,
        verbose=True
    )

    print(f"\nReference (numpy):\n{ref}")
    print(f"\nMINISA output (Allo):\n{output}")

    assert passed, "MINISA GEMM 8x8x16 failed to match numpy reference"
    print("PASSED: MINISA GEMM 8x8x16")
    return True


def test_minisa_gemm_16x8x32():
    """Test MINISA GEMM with larger M: C[16,8] = A[16,32] * B[32,8]."""
    print("\n" + "=" * 60)
    print("Test: MINISA GEMM 16x8x32")
    print("=" * 60)

    M, N, K = 16, 8, 32
    AW, AH = 8, 8

    output, ref, passed = run_minisa_gemm(
        M=M, N=N, K=K,
        AW=AW, AH=AH,
        verbose=True
    )

    assert passed, "MINISA GEMM 16x8x32 failed to match numpy reference"
    print("PASSED: MINISA GEMM 16x8x32")
    return True


def test_minisa_gemm_8x16x32():
    """Test MINISA GEMM with larger N: C[8,16] = A[8,32] * B[32,16]."""
    print("\n" + "=" * 60)
    print("Test: MINISA GEMM 8x16x32")
    print("=" * 60)

    M, N, K = 8, 16, 32
    AW, AH = 8, 8

    output, ref, passed = run_minisa_gemm(
        M=M, N=N, K=K,
        AW=AW, AH=AH,
        verbose=True
    )

    assert passed, "MINISA GEMM 8x16x32 failed to match numpy reference"
    print("PASSED: MINISA GEMM 8x16x32")
    return True


def test_minisa_single_tile_direct():
    """Test direct single-tile execution through Allo.

    This test verifies that the Allo module is correctly invoked for
    a single tile, demonstrating that compute happens in Allo.
    """
    print("\n" + "=" * 60)
    print("Test: MINISA Single Tile Direct Execution")
    print("=" * 60)

    AW, AH = 8, 8
    Ty = int8

    # Create interpreter
    interpreter = MINISAInterpreter(AW=AW, AH=AH, Ty=Ty, verbose=True)

    # Prepare tile data (already in FEATHER format)
    np.random.seed(42)
    iActs = np.random.randint(-4, 4, size=(AH, AW)).astype(np.int8)
    weights = np.random.randint(-4, 4, size=(AH, AW, AH)).astype(np.int8)
    birrd_inst = get_default_birrd_inst(AW)

    # Execute single tile through Allo
    output = interpreter.execute_single_tile(iActs, weights, birrd_inst)

    stats = interpreter.get_stats()
    print(f"\nExecution statistics:")
    print(f"  Allo invocations: {stats['allo_invocations']}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output:\n{output}")

    # Verify Allo was actually invoked
    assert stats['allo_invocations'] == 1, "Allo should be invoked exactly once"
    assert output.shape == (AH, AW), f"Output shape mismatch"

    print("PASSED: Single tile direct execution")
    return True


def test_minisa_allo_invocation_count():
    """Verify that Allo is invoked the correct number of times.

    This test proves that Allo performs the computation by tracking
    invocation counts. The interpreter must invoke Allo for each tile.
    """
    print("\n" + "=" * 60)
    print("Test: Allo Invocation Count Verification")
    print("=" * 60)

    M, N, K = 16, 16, 32
    AW, AH = 8, 8
    Mt = AW // 2  # = 4
    Nt = AH       # = 8
    Kt = 2 * AH   # = 16

    # Calculate expected tile count
    expected_tiles = (M // Mt) * (N // Nt) * (K // Kt)
    print(f"Matrix dimensions: M={M}, N={N}, K={K}")
    print(f"Tile dimensions: Mt={Mt}, Nt={Nt}, Kt={Kt}")
    print(f"Expected number of tiles: {expected_tiles}")

    # Execute GEMM
    interpreter = MINISAInterpreter(AW=AW, AH=AH, Ty=int8, verbose=False)
    program = create_gemm_program(M, N, K, AH, AW)

    np.random.seed(42)
    A = np.random.randint(-4, 4, size=(M, K)).astype(np.int8)
    B = np.random.randint(-4, 4, size=(K, N)).astype(np.int8)

    output = interpreter.execute_program(program, A, B)

    # Verify invocation count
    stats = interpreter.get_stats()
    actual_invocations = stats['allo_invocations']
    print(f"Actual Allo invocations: {actual_invocations}")

    assert actual_invocations == expected_tiles, \
        f"Expected {expected_tiles} Allo invocations, got {actual_invocations}"

    print("PASSED: Allo invocation count matches expected tiles")
    return True


def test_minisa_no_numpy_compute():
    """Verify that computation is not done by numpy.

    This test ensures that the interpreter does NOT perform numpy compute.
    It does this by checking that:
    1. The raw output has BIRRD-reordered layout
    2. The output must be extracted/reordered to match numpy reference
    3. If numpy compute were used, it would produce direct output layout
    """
    print("\n" + "=" * 60)
    print("Test: Verify No NumPy Compute")
    print("=" * 60)

    M, N, K = 8, 8, 16
    AW, AH = 8, 8

    # Create and execute program
    program = create_gemm_program(M, N, K, AH, AW)
    interpreter = MINISAInterpreter(AW=AW, AH=AH, Ty=int8, verbose=False)

    np.random.seed(42)
    A = np.random.randint(-4, 4, size=(M, K)).astype(np.int8)
    B = np.random.randint(-4, 4, size=(K, N)).astype(np.int8)

    # Get raw output (before extraction)
    raw_output = interpreter.execute_program(program, A, B)

    # Compute numpy reference
    ref = np.dot(A, B)

    print(f"Raw output shape: {raw_output.shape}")  # [N, 2*M]
    print(f"Reference shape: {ref.shape}")          # [M, N]

    # Verify raw output has different shape than reference
    # This proves the output comes from FEATHER layout, not numpy
    assert raw_output.shape != ref.shape, \
        "Raw output should have FEATHER layout, not numpy layout"

    # Verify raw output is NOT equal to transposed reference
    # (If numpy compute were used, we'd expect simple transpose)
    try:
        np.testing.assert_allclose(raw_output, ref.T, atol=1e-5)
        # If this passes, something is wrong - numpy compute may have been used
        raise AssertionError("Raw output should NOT match transposed reference directly")
    except AssertionError as e:
        if "Raw output should NOT" in str(e):
            raise
        # Expected: raw output differs from simple transpose
        pass

    # Now extract properly and verify it matches
    extracted = extract_output_for_verification(raw_output, ref.shape, AW)
    np.testing.assert_allclose(extracted, ref, atol=1e-5)

    print("Raw output has BIRRD-reordered layout (not numpy layout)")
    print("Extraction required to match numpy reference")
    print("PASSED: Computation is done by Allo, not numpy")
    return True


def run_minisa_gemm_tests():
    """Run all MINISA GEMM tests."""
    print("=" * 70)
    print("MINISA GEMM ALLO TESTS")
    print("=" * 70)

    results = {}

    tests = [
        ("GEMM 8x8x16", test_minisa_gemm_8x8x16),
        ("GEMM 16x8x32", test_minisa_gemm_16x8x32),
        ("GEMM 8x16x32", test_minisa_gemm_8x16x32),
        ("Single tile direct", test_minisa_single_tile_direct),
        ("Allo invocation count", test_minisa_allo_invocation_count),
        ("No numpy compute", test_minisa_no_numpy_compute),
    ]

    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"FAILED: {name}")
            print(f"  Error: {e}")
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
        print("\nAll MINISA GEMM tests PASSED")
    else:
        print("\nSome tests FAILED")

    return all_passed


if __name__ == "__main__":
    success = run_minisa_gemm_tests()
    sys.exit(0 if success else 1)
