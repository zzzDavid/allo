# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-layer sequential execution tests for FEATHER+ (TICKET-009 Phase 1).

Tests chaining of GEMM layers with quantized int8 intermediate values:
  Layer i output (int32) → post-quantization (uint8) → reinterpret as int8
  → Layer i+1 input (int8)

This matches the RTL auto-quant pipeline:
  OB write → auto OB read → quant_post → StaB PONG write
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from allo.ir.types import int8

from minisa.isa import create_gemm_program, encode_program
from feather_minisa import (
    build_feather_simulator,
    run_sequential_gemm_layers,
)


def _quant_ref(accum_int32, quant_scale, quant_zp):
    """Numpy golden reference for post-quantization."""
    return (accum_int32.astype(np.int64) * quant_scale
            + np.int64(quant_zp)).astype(np.int32) & 255


def test_two_layer_gemm():
    """2-layer GEMM: layer1 quantized int8 output feeds layer2 input.

    Layer 1: C1[8,8] = A[8,16] x B1[16,8]  (quant_scale=1, quant_zp=128)
             → uint8 values reinterpreted as int8
    Layer 2: C2[8,8] = C1_int8[8,8] x B2[8,8]  (no quantization)
    """
    print("\n" + "=" * 60)
    print("Test: Two-Layer Sequential GEMM")
    print("=" * 60)

    AW, AH = 8, 8
    quant_scale, quant_zp = 1, 128

    np.random.seed(42)
    A = np.random.randint(-4, 4, size=(8, 16)).astype(np.int8)
    B1 = np.random.randint(-4, 4, size=(16, 8)).astype(np.int8)
    B2 = np.random.randint(-4, 4, size=(8, 8)).astype(np.int8)

    # --- Layer 1: with post-quantization ---
    layer1_kwargs = dict(M=8, N=8, K=16, quant_scale=quant_scale, quant_zp=quant_zp)
    program1 = create_gemm_program(AH=AH, AW=AW, **layer1_kwargs)
    inst1 = encode_program(program1)

    mod1 = build_feather_simulator(8, 16, 8, AW, AH, int8, len(inst1))
    C1 = np.zeros((8, 8), dtype=np.int32)
    mod1(A, B1, inst1, C1)

    # Golden reference for layer 1
    ref1 = _quant_ref(A.astype(np.int32) @ B1.astype(np.int32), quant_scale, quant_zp)
    assert np.array_equal(C1, ref1), (
        f"Layer 1 mismatch. Max diff: {np.max(np.abs(C1 - ref1))}"
    )
    print(f"  Layer 1: C1[8,8] = A[8,16] x B1[16,8] + quant → PASSED")

    # Convert to int8 for layer 2 input
    C1_int8 = C1.astype(np.uint8).view(np.int8)

    # --- Layer 2: no quantization, gr=AW so Kt=AH=8 divides K=8 ---
    layer2_kwargs = dict(M=8, N=8, K=8, gr=AW)
    program2 = create_gemm_program(AH=AH, AW=AW, **layer2_kwargs)
    inst2 = encode_program(program2)

    mod2 = build_feather_simulator(8, 8, 8, AW, AH, int8, len(inst2))
    C2 = np.zeros((8, 8), dtype=np.int32)
    mod2(C1_int8, B2, inst2, C2)

    # Golden reference for layer 2
    ref2 = C1_int8.astype(np.int32) @ B2.astype(np.int32)
    assert np.array_equal(C2, ref2), (
        f"Layer 2 mismatch. Max diff: {np.max(np.abs(C2 - ref2))}"
    )
    print(f"  Layer 2: C2[8,8] = C1_int8[8,8] x B2[8,8] → PASSED")

    # End-to-end golden: A → quant → int8 → matmul
    e2e_ref = ref1.astype(np.uint8).view(np.int8).astype(np.int32) @ B2.astype(np.int32)
    assert np.array_equal(C2, e2e_ref), "End-to-end reference mismatch"
    print(f"  End-to-end: matches numpy golden reference")
    print("PASSED: Two-layer sequential GEMM")
    return True


def test_three_layer_gemm():
    """3-layer GEMM: two quantized intermediate layers.

    Layer 1: C1[8,8]  = A[8,16] x B1[16,8]  (quant_scale=1, quant_zp=128) → int8
    Layer 2: C2[8,8]  = C1[8,8]  x B2[8,8]  (quant_scale=2, quant_zp=64)  → int8
    Layer 3: C3[8,8]  = C2[8,8]  x B3[8,8]  (no quantization) → int32
    """
    print("\n" + "=" * 60)
    print("Test: Three-Layer Sequential GEMM")
    print("=" * 60)

    AW, AH = 8, 8

    np.random.seed(123)
    A = np.random.randint(-3, 3, size=(8, 16)).astype(np.int8)
    B1 = np.random.randint(-3, 3, size=(16, 8)).astype(np.int8)
    B2 = np.random.randint(-3, 3, size=(8, 8)).astype(np.int8)
    B3 = np.random.randint(-3, 3, size=(8, 8)).astype(np.int8)

    # Use the helper function
    outputs = run_sequential_gemm_layers(
        A,
        [B1, B2, B3],
        [
            dict(M=8, N=8, K=16, quant_scale=1, quant_zp=128),
            dict(M=8, N=8, K=8, gr=AW, quant_scale=2, quant_zp=64),
            dict(M=8, N=8, K=8, gr=AW),
        ],
        AW, AH,
    )

    # Golden reference chain
    ref1 = _quant_ref(A.astype(np.int32) @ B1.astype(np.int32), 1, 128)
    ref1_int8 = ref1.astype(np.uint8).view(np.int8)

    ref2 = _quant_ref(ref1_int8.astype(np.int32) @ B2.astype(np.int32), 2, 64)
    ref2_int8 = ref2.astype(np.uint8).view(np.int8)

    ref3 = ref2_int8.astype(np.int32) @ B3.astype(np.int32)

    assert np.array_equal(outputs[0], ref1), (
        f"Layer 1 mismatch. Max diff: {np.max(np.abs(outputs[0] - ref1))}"
    )
    print(f"  Layer 1: quant(A[8,16] x B1[16,8]) → PASSED")

    assert np.array_equal(outputs[1], ref2), (
        f"Layer 2 mismatch. Max diff: {np.max(np.abs(outputs[1] - ref2))}"
    )
    print(f"  Layer 2: quant(C1[8,8] x B2[8,8]) → PASSED")

    assert np.array_equal(outputs[2], ref3), (
        f"Layer 3 mismatch. Max diff: {np.max(np.abs(outputs[2] - ref3))}"
    )
    print(f"  Layer 3: C2[8,8] x B3[8,8] → PASSED")

    print("PASSED: Three-layer sequential GEMM")
    return True


def test_two_layer_with_zero_points():
    """2-layer GEMM with both zero points and post-quantization.

    Layer 1: (A - iacts_zp) x (B1 - weights_zp) + quant → int8
    Layer 2: C1_int8 x B2 → int32
    """
    print("\n" + "=" * 60)
    print("Test: Two-Layer with Zero Points")
    print("=" * 60)

    AW, AH = 8, 8
    iacts_zp, weights_zp = 2, 1
    quant_scale, quant_zp = 1, 128

    np.random.seed(77)
    A = np.random.randint(-4, 4, size=(8, 16)).astype(np.int8)
    B1 = np.random.randint(-4, 4, size=(16, 8)).astype(np.int8)
    B2 = np.random.randint(-4, 4, size=(8, 8)).astype(np.int8)

    outputs = run_sequential_gemm_layers(
        A,
        [B1, B2],
        [
            dict(M=8, N=8, K=16,
                 iacts_zp=iacts_zp, weights_zp=weights_zp,
                 quant_scale=quant_scale, quant_zp=quant_zp),
            dict(M=8, N=8, K=8, gr=AW),
        ],
        AW, AH,
    )

    # Golden: layer 1 with zero point subtraction + quantization
    A_shifted = A.astype(np.int32) - iacts_zp
    B1_shifted = B1.astype(np.int32) - weights_zp
    ref1 = _quant_ref(A_shifted @ B1_shifted, quant_scale, quant_zp)
    ref1_int8 = ref1.astype(np.uint8).view(np.int8)

    # Golden: layer 2 (no zero points, no quantization)
    ref2 = ref1_int8.astype(np.int32) @ B2.astype(np.int32)

    assert np.array_equal(outputs[0], ref1), (
        f"Layer 1 mismatch. Max diff: {np.max(np.abs(outputs[0] - ref1))}"
    )
    print(f"  Layer 1: quant((A-{iacts_zp})[8,16] x (B1-{weights_zp})[16,8]) → PASSED")

    assert np.array_equal(outputs[1], ref2), (
        f"Layer 2 mismatch. Max diff: {np.max(np.abs(outputs[1] - ref2))}"
    )
    print(f"  Layer 2: C1_int8[8,8] x B2[8,8] → PASSED")

    print("PASSED: Two-layer with zero points")
    return True


def test_two_layer_different_dataflow():
    """2-layer GEMM with different dataflows per layer.

    Layer 1: Output-stationary (Gr=4) with quantization → int8
    Layer 2: Gr=8 (passthrough) without quantization → int32

    Demonstrates dataflow switching between layers, which is a key
    FEATHER+ capability.
    """
    print("\n" + "=" * 60)
    print("Test: Two-Layer Different Dataflows")
    print("=" * 60)

    AW, AH = 8, 8

    np.random.seed(55)
    A = np.random.randint(-3, 3, size=(8, 16)).astype(np.int8)
    B1 = np.random.randint(-3, 3, size=(16, 8)).astype(np.int8)
    B2 = np.random.randint(-3, 3, size=(8, 8)).astype(np.int8)

    outputs = run_sequential_gemm_layers(
        A,
        [B1, B2],
        [
            # Layer 1: default output_stationary (Gr=AW//2=4)
            dict(M=8, N=8, K=16, quant_scale=1, quant_zp=100),
            # Layer 2: passthrough (Gr=AW=8), different OVN order
            dict(M=8, N=8, K=8, gr=AW, ovn_order=3),
        ],
        AW, AH,
    )

    ref1 = _quant_ref(A.astype(np.int32) @ B1.astype(np.int32), 1, 100)
    ref1_int8 = ref1.astype(np.uint8).view(np.int8)
    ref2 = ref1_int8.astype(np.int32) @ B2.astype(np.int32)

    assert np.array_equal(outputs[0], ref1), (
        f"Layer 1 mismatch. Max diff: {np.max(np.abs(outputs[0] - ref1))}"
    )
    print(f"  Layer 1: OS (Gr=4) + quant → PASSED")

    assert np.array_equal(outputs[1], ref2), (
        f"Layer 2 mismatch. Max diff: {np.max(np.abs(outputs[1] - ref2))}"
    )
    print(f"  Layer 2: passthrough (Gr=8, ovn_order=3) → PASSED")

    print("PASSED: Two-layer different dataflows")
    return True


def test_two_layer_aw4():
    """2-layer GEMM on AW=4 array.

    Layer 1: A[4,8] x B1[8,4]  (gr=4, quant) → int8
    Layer 2: C1[4,4] x B2[4,4] (gr=4, no quant) → int32
    """
    print("\n" + "=" * 60)
    print("Test: Two-Layer AW=4")
    print("=" * 60)

    AW, AH = 4, 4

    np.random.seed(88)
    A = np.random.randint(-4, 4, size=(4, 8)).astype(np.int8)
    B1 = np.random.randint(-4, 4, size=(8, 4)).astype(np.int8)
    B2 = np.random.randint(-4, 4, size=(4, 4)).astype(np.int8)

    outputs = run_sequential_gemm_layers(
        A,
        [B1, B2],
        [
            dict(M=4, N=4, K=8, gr=AW, quant_scale=1, quant_zp=128),
            dict(M=4, N=4, K=4, gr=AW),
        ],
        AW, AH,
    )

    ref1 = _quant_ref(A.astype(np.int32) @ B1.astype(np.int32), 1, 128)
    ref1_int8 = ref1.astype(np.uint8).view(np.int8)
    ref2 = ref1_int8.astype(np.int32) @ B2.astype(np.int32)

    assert np.array_equal(outputs[0], ref1), f"Layer 1 mismatch"
    print(f"  Layer 1: A[4,8] x B1[8,4] + quant → PASSED")
    assert np.array_equal(outputs[1], ref2), f"Layer 2 mismatch"
    print(f"  Layer 2: C1[4,4] x B2[4,4] → PASSED")
    print("PASSED: Two-layer AW=4")
    return True


def run_multi_layer_tests():
    """Run all multi-layer execution tests."""
    print("=" * 70)
    print("MULTI-LAYER SEQUENTIAL EXECUTION TESTS (TICKET-009 Phase 1)")
    print("=" * 70)

    results = {}
    tests = [
        ("Two-layer GEMM", test_two_layer_gemm),
        ("Three-layer GEMM", test_three_layer_gemm),
        ("Two-layer with zero points", test_two_layer_with_zero_points),
        ("Two-layer different dataflows", test_two_layer_different_dataflow),
        ("Two-layer AW=4", test_two_layer_aw4),
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
        print(f"\nAll {len(results)} multi-layer tests PASSED")
    else:
        print("\nSome tests FAILED")

    return all_passed


if __name__ == "__main__":
    success = run_multi_layer_tests()
    sys.exit(0 if success else 1)
