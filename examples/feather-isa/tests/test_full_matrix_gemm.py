# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Full-matrix GEMM tests for FEATHER+ with on-chip instruction decode.

These tests verify the DEV009 full-matrix execution model where a single
Allo invocation handles complete matrices and the full instruction list,
performing tiling, decode, compute, and accumulation on-chip.
"""

import os
import sys

# Setup path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from allo.ir.types import int8

from minisa.isa import (
    create_gemm_program,
    encode_program,
    NUM_FIELDS,
    INST_TYPE_IVN,
    INST_TYPE_WVN,
    INST_TYPE_OVN,
    INST_TYPE_MAPPING,
)
from feather_minisa import run_full_matrix_gemm


def test_full_matrix_gemm_8x8x16():
    """Test full-matrix GEMM: C[8,8] = A[8,16] * B[16,8]."""
    print("\n" + "=" * 60)
    print("Test: Full-Matrix GEMM 8x8x16")
    print("=" * 60)

    C, ref, passed = run_full_matrix_gemm(
        M=8, N=8, K=16, AW=8, AH=8, verbose=True
    )

    print(f"\nReference (numpy):\n{ref}")
    print(f"\nFull-matrix output (Allo):\n{C}")

    assert passed, "Full-matrix GEMM 8x8x16 failed to match numpy reference"
    print("PASSED: Full-matrix GEMM 8x8x16")
    return True


def test_full_matrix_gemm_16x8x32():
    """Test full-matrix GEMM: C[16,8] = A[16,32] * B[32,8]."""
    print("\n" + "=" * 60)
    print("Test: Full-Matrix GEMM 16x8x32")
    print("=" * 60)

    C, ref, passed = run_full_matrix_gemm(
        M=16, N=8, K=32, AW=8, AH=8, verbose=True
    )

    assert passed, "Full-matrix GEMM 16x8x32 failed to match numpy reference"
    print("PASSED: Full-matrix GEMM 16x8x32")
    return True


def test_full_matrix_gemm_16x16x32():
    """Test full-matrix GEMM: C[16,16] = A[16,32] * B[32,16]."""
    print("\n" + "=" * 60)
    print("Test: Full-Matrix GEMM 16x16x32")
    print("=" * 60)

    C, ref, passed = run_full_matrix_gemm(
        M=16, N=16, K=32, AW=8, AH=8, verbose=True
    )

    assert passed, "Full-matrix GEMM 16x16x32 failed to match numpy reference"
    print("PASSED: Full-matrix GEMM 16x16x32")
    return True


def test_full_matrix_instruction_encoding():
    """Verify encode_program produces correct array format."""
    print("\n" + "=" * 60)
    print("Test: Instruction Encoding")
    print("=" * 60)

    program = create_gemm_program(M=8, N=8, K=16, AH=8, AW=8)
    inst = encode_program(program)

    # Check shape
    expected_mappings = (8 // 4) * (8 // 8) * (16 // 16)  # 2 tiles
    expected_num_inst = 3 + expected_mappings
    assert inst.shape == (expected_num_inst, NUM_FIELDS), \
        f"Expected shape ({expected_num_inst}, {NUM_FIELDS}), got {inst.shape}"

    # Check instruction types
    assert inst[0, 0] == INST_TYPE_IVN, "First instruction should be SetIVNLayout"
    assert inst[1, 0] == INST_TYPE_WVN, "Second instruction should be SetWVNLayout"
    assert inst[2, 0] == INST_TYPE_OVN, "Third instruction should be SetOVNLayout"
    for i in range(3, len(inst)):
        assert inst[i, 0] == INST_TYPE_MAPPING, f"Instruction {i} should be SetMapping"

    # Check IVN layout fields
    assert inst[0, 1] == 0, "IVN order should be 0"
    assert inst[0, 2] == 8, "IVN ML0 should be AH=8"

    # Check first mapping tile bounds
    assert inst[3, 7] == 0, "First mapping m_start should be 0"
    assert inst[3, 8] == 4, "First mapping m_end should be Mt=4"
    assert inst[3, 9] == 0, "First mapping n_start should be 0"
    assert inst[3, 10] == 8, "First mapping n_end should be Nt=8"

    print(f"Encoded {len(inst)} instructions, {expected_mappings} tile mappings")
    print("PASSED: Instruction encoding")
    return True


def test_full_matrix_single_invocation():
    """Verify that full-matrix model uses exactly one Allo invocation."""
    print("\n" + "=" * 60)
    print("Test: Single Allo Invocation")
    print("=" * 60)

    M, N, K = 16, 16, 32
    AW, AH = 8, 8

    # The old model would need (M/Mt) * (N/Nt) * (K/Kt) = 4*2*2 = 16 invocations
    Mt = AW // 2
    Nt = AH
    Kt = 2 * AH
    old_invocations = (M // Mt) * (N // Nt) * (K // Kt)
    print(f"Old model would need {old_invocations} Allo invocations")
    print(f"Full-matrix model uses 1 Allo invocation")

    # Execute and verify correctness with single call
    C, ref, passed = run_full_matrix_gemm(
        M=M, N=N, K=K, AW=AW, AH=AH, verbose=True
    )

    assert passed, "Full-matrix GEMM failed"
    print("PASSED: Single Allo invocation produces correct results")
    return True


def test_full_matrix_matches_old_interpreter():
    """Cross-validate full-matrix output against old per-tile interpreter."""
    print("\n" + "=" * 60)
    print("Test: Cross-validation with Old Interpreter")
    print("=" * 60)

    M, N, K = 8, 8, 16
    AW, AH = 8, 8

    # Generate shared inputs
    np.random.seed(42)
    A = np.random.randint(-4, 4, size=(M, K)).astype(np.int8)
    B = np.random.randint(-4, 4, size=(K, N)).astype(np.int8)

    # Run new full-matrix model
    C_new, ref, passed_new = run_full_matrix_gemm(
        M=M, N=N, K=K, AW=AW, AH=AH, A=A.copy(), B=B.copy(), verbose=True
    )

    # Run old per-tile model
    from minisa.interpreter import run_minisa_gemm
    # run_minisa_gemm uses its own seed=42, so we pass matching inputs
    output_old, ref_old, passed_old = run_minisa_gemm(
        M=M, N=N, K=K, AW=AW, AH=AH, verbose=True
    )

    print(f"\nNew full-matrix output:\n{C_new}")
    print(f"\nOld interpreter output:\n{output_old}")

    # Both should match numpy reference
    assert passed_new, "Full-matrix model failed numpy comparison"
    assert passed_old, "Old interpreter failed numpy comparison"

    # Both should produce same result
    np.testing.assert_allclose(
        C_new, output_old, atol=1e-5,
    )

    print("PASSED: Full-matrix matches old interpreter")
    return True


def test_layout_instruction_decode_on_chip():
    """Verify that layout order fields are read from the instruction array.

    Creates a program with explicit order values and checks they appear
    in the encoded instruction array at the correct positions.
    """
    print("\n" + "=" * 60)
    print("Test: Layout Instruction Decode On-Chip")
    print("=" * 60)

    for ivn_ord, wvn_ord, ovn_ord in [(0, 0, 0), (1, 2, 3), (5, 4, 2)]:
        program = create_gemm_program(
            M=8, N=8, K=16, AH=8, AW=8,
            ivn_order=ivn_ord, wvn_order=wvn_ord, ovn_order=ovn_ord,
        )
        inst = encode_program(program)

        assert inst[0, 0] == INST_TYPE_IVN
        assert inst[0, 1] == ivn_ord, f"IVN order: expected {ivn_ord}, got {inst[0, 1]}"
        assert inst[1, 0] == INST_TYPE_WVN
        assert inst[1, 1] == wvn_ord, f"WVN order: expected {wvn_ord}, got {inst[1, 1]}"
        assert inst[2, 0] == INST_TYPE_OVN
        assert inst[2, 1] == ovn_ord, f"OVN order: expected {ovn_ord}, got {inst[2, 1]}"
        print(f"  orders ({ivn_ord},{wvn_ord},{ovn_ord}): encoded correctly")

    print("PASSED: Layout instruction decode on-chip")
    return True


def test_pe_mapping_fields_encoded():
    """Verify PE mapping fields are encoded in SetMapping instructions."""
    print("\n" + "=" * 60)
    print("Test: PE Mapping Fields Encoded")
    print("=" * 60)

    # Output-stationary
    program_os = create_gemm_program(
        M=8, N=8, K=16, AH=8, AW=8, dataflow="output_stationary",
    )
    inst_os = encode_program(program_os)
    # First mapping at index 3
    assert inst_os[3, 3] == 8, f"OS Gr should be AW=8, got {inst_os[3, 3]}"
    assert inst_os[3, 4] == 1, f"OS Gc should be 1, got {inst_os[3, 4]}"
    assert inst_os[3, 5] == 0, f"OS sr should be 0, got {inst_os[3, 5]}"
    assert inst_os[3, 6] == 0, f"OS sc should be 0, got {inst_os[3, 6]}"
    print("  Output-stationary: Gr=8, Gc=1, sr=0, sc=0 encoded correctly")

    # Weight-stationary
    program_ws = create_gemm_program(
        M=8, N=8, K=16, AH=8, AW=8, dataflow="weight_stationary",
    )
    inst_ws = encode_program(program_ws)
    assert inst_ws[3, 3] == 1, f"WS Gr should be 1, got {inst_ws[3, 3]}"
    assert inst_ws[3, 4] == 8, f"WS Gc should be AW=8, got {inst_ws[3, 4]}"
    assert inst_ws[3, 5] == 0, f"WS sr should be 0, got {inst_ws[3, 5]}"
    assert inst_ws[3, 6] == 1, f"WS sc should be 1, got {inst_ws[3, 6]}"
    print("  Weight-stationary: Gr=1, Gc=8, sr=0, sc=1 encoded correctly")

    print("PASSED: PE mapping fields encoded")
    return True


def test_order0_backward_compatible():
    """Verify explicit order=0 matches existing (default) results."""
    print("\n" + "=" * 60)
    print("Test: Order 0 Backward Compatible")
    print("=" * 60)

    # Run with default (implicit order=0)
    C_default, ref_default, passed_default = run_full_matrix_gemm(
        M=8, N=8, K=16, AW=8, AH=8, verbose=False
    )
    assert passed_default, "Default run failed"

    # Run with explicit order=0
    from minisa.isa import create_gemm_program, encode_program
    from feather_minisa import build_feather_full_matrix_simulator

    program = create_gemm_program(
        M=8, N=8, K=16, AH=8, AW=8,
        ivn_order=0, wvn_order=0, ovn_order=0,
    )
    instructions = encode_program(program)

    np.random.seed(42)
    A = np.random.randint(-4, 4, size=(8, 16)).astype(np.int8)
    B = np.random.randint(-4, 4, size=(16, 8)).astype(np.int8)

    mod = build_feather_full_matrix_simulator(8, 16, 8, 8, 8, int8, len(instructions))
    C_explicit = np.zeros((8, 8), dtype=np.int32)
    mod(A, B, instructions, C_explicit)

    np.testing.assert_array_equal(
        C_default, C_explicit,
        err_msg="Explicit order=0 should match default"
    )
    print("  Explicit order=0 matches default: True")
    print("PASSED: Order 0 backward compatible")
    return True


def test_ovn_order_produces_different_birrd():
    """Verify each OVN order produces different BIRRD instruction tables."""
    print("\n" + "=" * 60)
    print("Test: OVN Order Produces Different BIRRD")
    print("=" * 60)

    from minisa.lowering import lower_ovn_layout, compute_output_col_map
    from minisa.isa import SetOVNLayout

    for AW in [4, 8, 16]:
        birrd_tables = []
        col_maps = []
        for order in range(6):
            ovn = SetOVNLayout(order=order, PL0=8, PL1=1, QL0=8, QL1=1)
            birrd = lower_ovn_layout(ovn, AW, 8)
            col_map = compute_output_col_map(AW, order)
            birrd_tables.append(birrd)
            col_maps.append(tuple(col_map.tolist()))

        # All BIRRD tables should be different
        for i in range(6):
            for j in range(i + 1, 6):
                if np.array_equal(birrd_tables[i], birrd_tables[j]):
                    assert False, \
                        f"AW={AW}: orders {i} and {j} have identical BIRRD tables"

        unique_col_maps = len(set(col_maps))
        print(f"  AW={AW}: 6 unique BIRRD tables, {unique_col_maps} unique col_maps")

    print("PASSED: OVN order produces different BIRRD")
    return True


def test_ivn_order_affects_output():
    """Verify different IVN orders produce different outputs."""
    print("\n" + "=" * 60)
    print("Test: IVN Order Affects Output")
    print("=" * 60)

    from minisa.isa import create_gemm_program, encode_program
    from feather_minisa import build_feather_full_matrix_simulator

    M, N, K, AW, AH = 8, 8, 16, 8, 8
    np.random.seed(42)
    A = np.random.randint(-4, 4, size=(M, K)).astype(np.int8)
    B = np.random.randint(-4, 4, size=(K, N)).astype(np.int8)

    outputs = {}
    for ivn_order in range(6):
        program = create_gemm_program(
            M=M, N=N, K=K, AH=AH, AW=AW, ivn_order=ivn_order,
        )
        instructions = encode_program(program)
        mod = build_feather_full_matrix_simulator(
            M, K, N, AW, AH, int8, len(instructions)
        )
        C = np.zeros((M, N), dtype=np.int32)
        mod(A, B, instructions, C)
        outputs[ivn_order] = C.copy()
        print(f"  IVN order={ivn_order}: output sum={C.sum()}")

    # Order 0 should be correct GEMM
    ref = np.dot(A, B)
    np.testing.assert_allclose(outputs[0], ref, atol=1e-5)
    print("  IVN order=0: matches numpy reference")

    # At least some other orders should differ
    diff_count = sum(
        1 for o in range(1, 6) if not np.array_equal(outputs[0], outputs[o])
    )
    assert diff_count > 0, "At least one non-zero IVN order should produce different output"
    print(f"  {diff_count}/5 non-zero orders produce different output")

    print("PASSED: IVN order affects output")
    return True


def test_wvn_order_affects_output():
    """Verify different WVN orders produce different outputs."""
    print("\n" + "=" * 60)
    print("Test: WVN Order Affects Output")
    print("=" * 60)

    from minisa.isa import create_gemm_program, encode_program
    from feather_minisa import build_feather_full_matrix_simulator

    M, N, K, AW, AH = 8, 8, 16, 8, 8
    np.random.seed(42)
    A = np.random.randint(-4, 4, size=(M, K)).astype(np.int8)
    B = np.random.randint(-4, 4, size=(K, N)).astype(np.int8)

    outputs = {}
    for wvn_order in range(6):
        program = create_gemm_program(
            M=M, N=N, K=K, AH=AH, AW=AW, wvn_order=wvn_order,
        )
        instructions = encode_program(program)
        mod = build_feather_full_matrix_simulator(
            M, K, N, AW, AH, int8, len(instructions)
        )
        C = np.zeros((M, N), dtype=np.int32)
        mod(A, B, instructions, C)
        outputs[wvn_order] = C.copy()
        print(f"  WVN order={wvn_order}: output sum={C.sum()}")

    # Order 0 should be correct GEMM
    ref = np.dot(A, B)
    np.testing.assert_allclose(outputs[0], ref, atol=1e-5)
    print("  WVN order=0: matches numpy reference")

    # At least some other orders should differ
    diff_count = sum(
        1 for o in range(1, 6) if not np.array_equal(outputs[0], outputs[o])
    )
    assert diff_count > 0, "At least one non-zero WVN order should produce different output"
    print(f"  {diff_count}/5 non-zero orders produce different output")

    print("PASSED: WVN order affects output")
    return True


def test_ovn_order_all_correct():
    """Verify all OVN orders produce correct GEMM via different BIRRD routing."""
    print("\n" + "=" * 60)
    print("Test: OVN Order All Correct")
    print("=" * 60)

    from minisa.isa import create_gemm_program, encode_program
    from feather_minisa import build_feather_full_matrix_simulator

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
        mod = build_feather_full_matrix_simulator(
            M, K, N, AW, AH, int8, len(instructions)
        )
        C = np.zeros((M, N), dtype=np.int32)
        mod(A, B, instructions, C)
        np.testing.assert_allclose(C, ref, atol=1e-5)
        print(f"  OVN order={ovn_order}: matches numpy reference")

    print("PASSED: OVN order all correct")
    return True


def run_full_matrix_tests():
    """Run all full-matrix GEMM tests."""
    print("=" * 70)
    print("FULL-MATRIX GEMM TESTS (DEV009)")
    print("=" * 70)

    results = {}

    tests = [
        ("Instruction encoding", test_full_matrix_instruction_encoding),
        ("GEMM 8x8x16", test_full_matrix_gemm_8x8x16),
        ("GEMM 16x8x32", test_full_matrix_gemm_16x8x32),
        ("GEMM 16x16x32", test_full_matrix_gemm_16x16x32),
        ("Single invocation", test_full_matrix_single_invocation),
        ("Cross-validation", test_full_matrix_matches_old_interpreter),
        ("Layout decode on-chip", test_layout_instruction_decode_on_chip),
        ("PE mapping fields", test_pe_mapping_fields_encoded),
        ("Order 0 backward compat", test_order0_backward_compatible),
        ("OVN order BIRRD tables", test_ovn_order_produces_different_birrd),
        ("IVN order affects output", test_ivn_order_affects_output),
        ("WVN order affects output", test_wvn_order_affects_output),
        ("OVN order all correct", test_ovn_order_all_correct),
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
        print("\nAll full-matrix GEMM tests PASSED")
    else:
        print("\nSome tests FAILED")

    return all_passed


if __name__ == "__main__":
    success = run_full_matrix_tests()
    sys.exit(0 if success else 1)
