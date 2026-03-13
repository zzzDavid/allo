# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for crossbar flexibility: different Gr values on AW=4 NEST.

Verifies that the parametric Gr crossbar (power-of-2 bit operations) produces
correct GEMM results for different dataflow configurations:

- Gr=AW (pass-through): each PE column handles independent M row, no BIRRD reduction
- Gr=AW//2 (2-way reduction): paired columns handle different K-stripes,
  BIRRD reduces partial sums
- Mixed Gr tiles: some tiles Gr=AW, others Gr=AW//2, in a single program

Also tests that bit operations (& and >>) match modulo/division for all
valid Gr and index values.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from allo.ir.types import int8

from minisa.isa import (
    MINISAProgram,
    SetIVNLayout,
    SetWVNLayout,
    SetOVNLayout,
    SetMapping,
    create_figure7_program,
    encode_program,
)
from feather_minisa import build_feather_kstreaming_simulator, compute_max_k_passes


# Hardware parameters for 4x4 NEST
AH = 4
AW = 4


def test_gr_equals_aw():
    """Gr=AW=4: pass-through BIRRD mode (Figure 7 regression).

    Each PE column handles a distinct M row. No BIRRD reduction needed.
    All tiles use Gr=4 with K-streaming across 3 K-passes.
    """
    M, K, N = 16, 12, 8
    program = create_figure7_program()
    instructions = encode_program(program)
    max_k_passes = compute_max_k_passes(instructions, AW, AH)

    np.random.seed(7)
    A = np.random.randint(-4, 4, size=(M, K)).astype(np.int8)
    B = np.random.randint(-4, 4, size=(K, N)).astype(np.int8)

    mod = build_feather_kstreaming_simulator(
        M, K, N, AW, AH, int8, len(instructions),
        max_k_passes,
    )
    C = np.zeros((M, N), dtype=np.int32)
    mod(A, B, instructions, C)

    ref = A.astype(np.int32) @ B.astype(np.int32)
    np.testing.assert_array_equal(C, ref)


def test_gr_half_aw():
    """Gr=AW//2=2: standard BIRRD reduction mode.

    Paired columns handle different K-stripes for the same M rows.
    BIRRD reduces the partial K-sums to produce correct GEMM output.

    Workload: C[8,4] = A[8,8] x B[8,4]
    - 4 tiles (4 M-blocks x 1 N-group), each with Gr=2, Mt=2
    - Kt_per_pass = (AW/Gr)*AH = 2*4 = 8
    - Each tile covers full K=8 in 1 pass
    """
    M, K, N = 8, 8, 4
    Gr = AW // 2  # 2
    Mt = Gr  # 2 M rows per tile (from crossbar: ic_j & (Gr-1))
    Nt = AH  # 4

    program = MINISAProgram(
        name="gr2_test", AH=AH, AW=AW,
        ivn_layout=SetIVNLayout(order=0, ML0=AH, ML1=M // AH, JL0=AH, JL1=K // AH),
        wvn_layout=SetWVNLayout(order=0, KL0=AH, KL1=K // AH, NL0=min(N, AW), NL1=max(1, N // AW)),
        ovn_layout=SetOVNLayout(order=0, PL0=AH, PL1=M // AH, QL0=AH, QL1=N // AH),
    )

    # Gr=2, Gc=2, sr=1, sc=4 (same column mapping as Figure 7 tile 1)
    for n_tile in range(N // Nt):
        for m_tile in range(M // Mt):
            program.add_mapping(SetMapping(
                r0=0, c0=n_tile * Nt,
                Gr=Gr, Gc=2, sr=1, sc=4,
                m_start=m_tile * Mt, m_end=(m_tile + 1) * Mt,
                n_start=n_tile * Nt, n_end=(n_tile + 1) * Nt,
                k_start=0, k_end=K,
            ))

    instructions = encode_program(program)
    max_k_passes = compute_max_k_passes(instructions, AW, AH)

    np.random.seed(42)
    A = np.random.randint(-4, 4, size=(M, K)).astype(np.int8)
    B = np.random.randint(-4, 4, size=(K, N)).astype(np.int8)

    mod = build_feather_kstreaming_simulator(
        M, K, N, AW, AH, int8, len(instructions),
        max_k_passes,
    )
    C = np.zeros((M, N), dtype=np.int32)
    mod(A, B, instructions, C)

    ref = A.astype(np.int32) @ B.astype(np.int32)
    np.testing.assert_array_equal(C, ref)


def test_mixed_gr_tiles():
    """Mixed Gr values: Gr=2 and Gr=4 tiles in a single program.

    Workload: C[8,4] = A[8,12] x B[12,4]
    - Gr=2 tiles for K=[0,8): 4 tiles (M/Mt=4), BIRRD reduction, Mt=2
    - Gr=4 tiles for K=[8,12): 2 tiles (M/Mt=2), pass-through, Mt=4

    All tiles share Kt_per_pass=8 (max stride needed for Gr=2).
    Gr=4 tiles only read 4 of 8 K-elements per pass (ic_j >> 2 = 0).
    """
    M, K, N = 8, 12, 4
    Nt = AH  # 4

    program = MINISAProgram(
        name="mixed_gr_test", AH=AH, AW=AW,
        ivn_layout=SetIVNLayout(order=0, ML0=AH, ML1=M // AH, JL0=AH, JL1=K // AH),
        wvn_layout=SetWVNLayout(order=0, KL0=AH, KL1=K // AH, NL0=min(N, AW), NL1=max(1, N // AW)),
        ovn_layout=SetOVNLayout(order=0, PL0=AH, PL1=M // AH, QL0=AH, QL1=N // AH),
    )

    for n_tile in range(N // Nt):
        # Gr=2 tiles: K=[0,8), Mt=2
        Mt_gr2 = 2
        for m_tile in range(M // Mt_gr2):
            program.add_mapping(SetMapping(
                r0=0, c0=n_tile * Nt,
                Gr=2, Gc=2, sr=1, sc=4,
                m_start=m_tile * Mt_gr2, m_end=(m_tile + 1) * Mt_gr2,
                n_start=n_tile * Nt, n_end=(n_tile + 1) * Nt,
                k_start=0, k_end=8,
            ))

        # Gr=4 tiles: K=[8,12), Mt=4
        Mt_gr4 = 4
        for m_tile in range(M // Mt_gr4):
            program.add_mapping(SetMapping(
                r0=2, c0=n_tile * Nt,
                Gr=4, Gc=2, sr=1, sc=4,
                m_start=m_tile * Mt_gr4, m_end=(m_tile + 1) * Mt_gr4,
                n_start=n_tile * Nt, n_end=(n_tile + 1) * Nt,
                k_start=8, k_end=12,
            ))

    instructions = encode_program(program)
    max_k_passes = compute_max_k_passes(instructions, AW, AH)

    np.random.seed(99)
    A = np.random.randint(-4, 4, size=(M, K)).astype(np.int8)
    B = np.random.randint(-4, 4, size=(K, N)).astype(np.int8)

    mod = build_feather_kstreaming_simulator(
        M, K, N, AW, AH, int8, len(instructions),
        max_k_passes,
    )
    C = np.zeros((M, N), dtype=np.int32)
    mod(A, B, instructions, C)

    ref = A.astype(np.int32) @ B.astype(np.int32)
    np.testing.assert_array_equal(C, ref)


def test_gr_1_aw4():
    """Gr=1 on AW=4: 4-way reduction (weight-stationary-like).

    Each PE column handles a different K-stripe for the same M row.
    BIRRD does 2-way reduction, output_accum accumulates the remaining 2 pairs.

    Workload: C[1,4] = A[1,16] x B[16,4]
    - 1 tile, Gr=1, Mt=1, Kt_per_pass = (4/1)*4 = 16
    """
    M, K, N = 1, 16, 4
    Gr = 1
    Mt = Gr  # 1
    Nt = AH  # 4

    program = MINISAProgram(
        name="gr1_aw4_test", AH=AH, AW=AW,
        ivn_layout=SetIVNLayout(order=0, ML0=AH, ML1=M // AH if M >= AH else 1, JL0=AH, JL1=K // AH),
        wvn_layout=SetWVNLayout(order=0, KL0=AH, KL1=K // AH, NL0=min(N, AW), NL1=max(1, N // AW)),
        ovn_layout=SetOVNLayout(order=0, PL0=AH, PL1=1, QL0=AH, QL1=N // AH),
    )

    for n_tile in range(N // Nt):
        for m_tile in range(M // Mt):
            program.add_mapping(SetMapping(
                r0=0, c0=n_tile * Nt,
                Gr=Gr, Gc=1, sr=1, sc=0,
                m_start=m_tile * Mt, m_end=(m_tile + 1) * Mt,
                n_start=n_tile * Nt, n_end=(n_tile + 1) * Nt,
                k_start=0, k_end=K,
            ))

    instructions = encode_program(program)
    max_k_passes = compute_max_k_passes(instructions, AW, AH)

    np.random.seed(123)
    A = np.random.randint(-4, 4, size=(M, K)).astype(np.int8)
    B = np.random.randint(-4, 4, size=(K, N)).astype(np.int8)

    mod = build_feather_kstreaming_simulator(
        M, K, N, AW, AH, int8, len(instructions),
        max_k_passes,
    )
    C = np.zeros((M, N), dtype=np.int32)
    mod(A, B, instructions, C)

    ref = A.astype(np.int32) @ B.astype(np.int32)
    np.testing.assert_array_equal(C, ref)


def test_gr_2_aw8():
    """Gr=2 on AW=8: 4-way reduction.

    Each Gr=2 group of PE columns shares M rows, 4 groups total.
    BIRRD does 2-way, output_accum sums the 2 remaining pairs per M.

    Workload: C[2,8] = A[2,32] x B[32,8]
    - 1 tile, Gr=2, Mt=2, Kt_per_pass = (8/2)*8 = 32
    """
    AW_8, AH_8 = 8, 8
    M, K, N = 2, 32, 8
    Gr = 2
    Mt = Gr  # 2
    Nt = AH_8  # 8

    program = MINISAProgram(
        name="gr2_aw8_test", AH=AH_8, AW=AW_8,
        ivn_layout=SetIVNLayout(order=0, ML0=AH_8, ML1=1, JL0=AH_8, JL1=K // AH_8),
        wvn_layout=SetWVNLayout(order=0, KL0=AH_8, KL1=K // AH_8, NL0=min(N, AW_8), NL1=max(1, N // AW_8)),
        ovn_layout=SetOVNLayout(order=0, PL0=AH_8, PL1=1, QL0=AH_8, QL1=N // AH_8),
    )

    for n_tile in range(N // Nt):
        for m_tile in range(M // Mt):
            program.add_mapping(SetMapping(
                r0=0, c0=n_tile * Nt,
                Gr=Gr, Gc=1, sr=1, sc=0,
                m_start=m_tile * Mt, m_end=(m_tile + 1) * Mt,
                n_start=n_tile * Nt, n_end=(n_tile + 1) * Nt,
                k_start=0, k_end=K,
            ))

    instructions = encode_program(program)
    max_k_passes = compute_max_k_passes(instructions, AW_8, AH_8)

    np.random.seed(456)
    A = np.random.randint(-4, 4, size=(M, K)).astype(np.int8)
    B = np.random.randint(-4, 4, size=(K, N)).astype(np.int8)

    mod = build_feather_kstreaming_simulator(
        M, K, N, AW_8, AH_8, int8, len(instructions),
        max_k_passes,
    )
    C = np.zeros((M, N), dtype=np.int32)
    mod(A, B, instructions, C)

    ref = A.astype(np.int32) @ B.astype(np.int32)
    np.testing.assert_array_equal(C, ref)


def test_gr_1_aw8():
    """Gr=1 on AW=8: 8-way reduction (full weight-stationary).

    All PE columns handle different K-stripes for the same M row.
    BIRRD does 2-way (4 pairs), output_accum sums all 4 pairs to 1 M.

    Workload: C[1,8] = A[1,64] x B[64,8]
    - 1 tile, Gr=1, Mt=1, Kt_per_pass = (8/1)*8 = 64
    """
    AW_8, AH_8 = 8, 8
    M, K, N = 1, 64, 8
    Gr = 1
    Mt = Gr  # 1
    Nt = AH_8  # 8

    program = MINISAProgram(
        name="gr1_aw8_test", AH=AH_8, AW=AW_8,
        ivn_layout=SetIVNLayout(order=0, ML0=AH_8, ML1=1, JL0=AH_8, JL1=K // AH_8),
        wvn_layout=SetWVNLayout(order=0, KL0=AH_8, KL1=K // AH_8, NL0=min(N, AW_8), NL1=max(1, N // AW_8)),
        ovn_layout=SetOVNLayout(order=0, PL0=AH_8, PL1=1, QL0=AH_8, QL1=N // AH_8),
    )

    for n_tile in range(N // Nt):
        for m_tile in range(M // Mt):
            program.add_mapping(SetMapping(
                r0=0, c0=n_tile * Nt,
                Gr=Gr, Gc=1, sr=1, sc=0,
                m_start=m_tile * Mt, m_end=(m_tile + 1) * Mt,
                n_start=n_tile * Nt, n_end=(n_tile + 1) * Nt,
                k_start=0, k_end=K,
            ))

    instructions = encode_program(program)
    max_k_passes = compute_max_k_passes(instructions, AW_8, AH_8)

    np.random.seed(789)
    A = np.random.randint(-4, 4, size=(M, K)).astype(np.int8)
    B = np.random.randint(-4, 4, size=(K, N)).astype(np.int8)

    mod = build_feather_kstreaming_simulator(
        M, K, N, AW_8, AH_8, int8, len(instructions),
        max_k_passes,
    )
    C = np.zeros((M, N), dtype=np.int32)
    mod(A, B, instructions, C)

    ref = A.astype(np.int32) @ B.astype(np.int32)
    np.testing.assert_array_equal(C, ref)


def test_mixed_kt_per_pass():
    """Mixed Kt_per_pass: Gr=2 tiles (2 passes) and Gr=4 tiles (4 passes).

    Workload: C[4,4] = A[4,32] x B[32,4] on AW=4, AH=4

    | Tiles  | Gr | Mt | K-range  | Kt_per_pass | actual_passes |
    |--------|----|----|----------|-------------|---------------|
    | 2 tiles| 2  | 2  | [0,16)   | 8           | 2             |
    | 1 tile | 4  | 4  | [16,32)  | 4           | 4             |

    max_k_passes = 4. Gr=2 tiles get 2 padding passes (zeros).
    Verifies correct GEMM with truly mixed K-pass counts.
    """
    M, K, N = 4, 32, 4
    Nt = AH  # 4

    program = MINISAProgram(
        name="mixed_kt_test", AH=AH, AW=AW,
        ivn_layout=SetIVNLayout(order=0, ML0=AH, ML1=1, JL0=AH, JL1=K // AH),
        wvn_layout=SetWVNLayout(order=0, KL0=AH, KL1=K // AH, NL0=min(N, AW), NL1=max(1, N // AW)),
        ovn_layout=SetOVNLayout(order=0, PL0=AH, PL1=1, QL0=AH, QL1=N // AH),
    )

    # Gr=2 tiles: K=[0,16), Mt=2, Kt_per_pass=(4/2)*4=8, actual_passes=16/8=2
    Mt_gr2 = 2
    for m_tile in range(M // Mt_gr2):
        program.add_mapping(SetMapping(
            r0=0, c0=0,
            Gr=2, Gc=2, sr=1, sc=4,
            m_start=m_tile * Mt_gr2, m_end=(m_tile + 1) * Mt_gr2,
            n_start=0, n_end=N,
            k_start=0, k_end=16,
        ))

    # Gr=4 tile: K=[16,32), Mt=4, Kt_per_pass=(4/4)*4=4, actual_passes=16/4=4
    program.add_mapping(SetMapping(
        r0=4, c0=0,
        Gr=4, Gc=2, sr=1, sc=4,
        m_start=0, m_end=4,
        n_start=0, n_end=N,
        k_start=16, k_end=32,
    ))

    instructions = encode_program(program)
    max_k_passes = compute_max_k_passes(instructions, AW, AH)

    # Verify mixed K-pass counts
    assert max_k_passes == 4, f"Expected max_k_passes=4, got {max_k_passes}"

    np.random.seed(202)
    A = np.random.randint(-4, 4, size=(M, K)).astype(np.int8)
    B = np.random.randint(-4, 4, size=(K, N)).astype(np.int8)

    mod = build_feather_kstreaming_simulator(
        M, K, N, AW, AH, int8, len(instructions),
        max_k_passes,
    )
    C = np.zeros((M, N), dtype=np.int32)
    mod(A, B, instructions, C)

    ref = A.astype(np.int32) @ B.astype(np.int32)
    np.testing.assert_array_equal(C, ref)


def test_ivn_wvn_orders_aw4():
    """Verify all IVN/WVN layout orders produce correct GEMM on 4x4 NEST.

    Tests all 6 IVN orders and all 6 WVN orders independently, plus
    a combined non-zero case. The crossbar routing (determined by Gr)
    is independent of the VN buffer layout order.
    """
    M, K, N = 8, 8, 4
    Gr = AW // 2  # 2
    Mt = Gr

    np.random.seed(77)
    A = np.random.randint(-4, 4, size=(M, K)).astype(np.int8)
    B = np.random.randint(-4, 4, size=(K, N)).astype(np.int8)
    ref = A.astype(np.int32) @ B.astype(np.int32)

    # Test all 6 IVN orders
    for ivn_order in range(6):
        program = MINISAProgram(
            name=f"ivn_order_{ivn_order}", AH=AH, AW=AW,
            ivn_layout=SetIVNLayout(order=ivn_order, ML0=AH, ML1=M // AH,
                                     JL0=AH, JL1=K // AH),
            wvn_layout=SetWVNLayout(order=0, KL0=AH, KL1=K // AH,
                                     NL0=min(N, AW), NL1=max(1, N // AW)),
            ovn_layout=SetOVNLayout(order=0, PL0=AH, PL1=M // AH,
                                     QL0=AH, QL1=N // AH),
        )
        for n_tile in range(N // AH):
            for m_tile in range(M // Mt):
                program.add_mapping(SetMapping(
                    r0=0, c0=n_tile * AH,
                    Gr=Gr, Gc=2, sr=1, sc=4,
                    m_start=m_tile * Mt, m_end=(m_tile + 1) * Mt,
                    n_start=n_tile * AH, n_end=(n_tile + 1) * AH,
                    k_start=0, k_end=K,
                ))
        instructions = encode_program(program)
        max_k_passes = compute_max_k_passes(instructions, AW, AH)
        mod = build_feather_kstreaming_simulator(
            M, K, N, AW, AH, int8, len(instructions), max_k_passes,
        )
        C = np.zeros((M, N), dtype=np.int32)
        mod(A, B, instructions, C)
        np.testing.assert_array_equal(C, ref)

    # Test all 6 WVN orders
    for wvn_order in range(6):
        program = MINISAProgram(
            name=f"wvn_order_{wvn_order}", AH=AH, AW=AW,
            ivn_layout=SetIVNLayout(order=0, ML0=AH, ML1=M // AH,
                                     JL0=AH, JL1=K // AH),
            wvn_layout=SetWVNLayout(order=wvn_order, KL0=AH, KL1=K // AH,
                                     NL0=min(N, AW), NL1=max(1, N // AW)),
            ovn_layout=SetOVNLayout(order=0, PL0=AH, PL1=M // AH,
                                     QL0=AH, QL1=N // AH),
        )
        for n_tile in range(N // AH):
            for m_tile in range(M // Mt):
                program.add_mapping(SetMapping(
                    r0=0, c0=n_tile * AH,
                    Gr=Gr, Gc=2, sr=1, sc=4,
                    m_start=m_tile * Mt, m_end=(m_tile + 1) * Mt,
                    n_start=n_tile * AH, n_end=(n_tile + 1) * AH,
                    k_start=0, k_end=K,
                ))
        instructions = encode_program(program)
        max_k_passes = compute_max_k_passes(instructions, AW, AH)
        mod = build_feather_kstreaming_simulator(
            M, K, N, AW, AH, int8, len(instructions), max_k_passes,
        )
        C = np.zeros((M, N), dtype=np.int32)
        mod(A, B, instructions, C)
        np.testing.assert_array_equal(C, ref)

    # Test combined non-zero IVN + WVN orders
    program = MINISAProgram(
        name="mixed_ivn_wvn", AH=AH, AW=AW,
        ivn_layout=SetIVNLayout(order=3, ML0=AH, ML1=M // AH,
                                 JL0=AH, JL1=K // AH),
        wvn_layout=SetWVNLayout(order=5, KL0=AH, KL1=K // AH,
                                 NL0=min(N, AW), NL1=max(1, N // AW)),
        ovn_layout=SetOVNLayout(order=2, PL0=AH, PL1=M // AH,
                                 QL0=AH, QL1=N // AH),
    )
    for n_tile in range(N // AH):
        for m_tile in range(M // Mt):
            program.add_mapping(SetMapping(
                r0=0, c0=n_tile * AH,
                Gr=Gr, Gc=2, sr=1, sc=4,
                m_start=m_tile * Mt, m_end=(m_tile + 1) * Mt,
                n_start=n_tile * AH, n_end=(n_tile + 1) * AH,
                k_start=0, k_end=K,
            ))
    instructions = encode_program(program)
    max_k_passes = compute_max_k_passes(instructions, AW, AH)
    mod = build_feather_kstreaming_simulator(
        M, K, N, AW, AH, int8, len(instructions), max_k_passes,
    )
    C = np.zeros((M, N), dtype=np.int32)
    mod(A, B, instructions, C)
    np.testing.assert_array_equal(C, ref)


def test_bit_ops_equivalence():
    """Verify bit operations match modulo/division for all valid Gr and j values.

    For power-of-2 Gr:
      j & (Gr - 1)  ==  j % Gr
      j >> log2(Gr)  ==  j // Gr
    """
    for AW_test in [4, 8, 16]:
        for Gr in [1, 2, AW_test // 2, AW_test]:
            if Gr < 1:
                continue
            log2_Gr = int(np.log2(Gr)) if Gr > 0 else 0
            mask_Gr = Gr - 1
            for j in range(AW_test):
                assert (j & mask_Gr) == (j % Gr), \
                    f"AW={AW_test}, Gr={Gr}, j={j}: {j & mask_Gr} != {j % Gr}"
                assert (j >> log2_Gr) == (j // Gr), \
                    f"AW={AW_test}, Gr={Gr}, j={j}: {j >> log2_Gr} != {j // Gr}"


if __name__ == "__main__":
    tests = [
        test_gr_equals_aw,
        test_gr_half_aw,
        test_mixed_gr_tiles,
        test_gr_1_aw4,
        test_gr_2_aw8,
        test_gr_1_aw8,
        test_mixed_kt_per_pass,
        test_ivn_wvn_orders_aw4,
        test_bit_ops_equivalence,
    ]
    for t in tests:
        print(f"  {t.__name__} ... ", end="", flush=True)
        t()
        print("PASSED")
    print(f"\nAll {len(tests)} crossbar flexibility tests passed.")
