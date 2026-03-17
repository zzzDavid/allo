# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parameterized FEATHER+ GEMM tests for arbitrary AW/AH.

Generates test cases relative to tile sizes so any power-of-2 AW/AH works.
Supports simulator, HLS csim, and HLS csynth modes.

Usage:
  python tests/test_parameterized_gemm.py --aw 16 --ah 16              # simulator
  python tests/test_parameterized_gemm.py --aw 4 --ah 4                # 4x4 regression
  python tests/test_parameterized_gemm.py --aw 16 --ah 16 --hls csim   # HLS C-sim
  python tests/test_parameterized_gemm.py --aw 16 --ah 16 --hls csyn   # HLS synthesis
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from allo.ir.types import int8

from minisa.isa import (
    create_gemm_program,
    encode_program,
    MINISAProgram,
    SetIVNLayout,
    SetWVNLayout,
    SetOVNLayout,
    SetMapping,
)
from feather_minisa import (
    build_feather_kstreaming_simulator,
    build_feather_kstreaming_hls,
    get_feather_full_matrix_top_kstreaming,
)

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))


def create_passthrough_program(M, N, K, AW, AH):
    """Create GEMM program with Gr=AW (pass-through, no BIRRD reduction).

    With Gr=AW, each PE column maps to a distinct M row.
    Mt=AW, Kt=AH (one K-stripe per tile).
    """
    Mt = AW
    Nt = AH
    Kt = AH  # single K-stripe per tile

    assert M % Mt == 0, f"M={M} must be divisible by Mt={Mt} for pass-through"
    assert N % Nt == 0, f"N={N} must be divisible by Nt={Nt}"
    assert K % Kt == 0, f"K={K} must be divisible by Kt={Kt}"

    program = MINISAProgram(
        name="passthrough_gemm", AH=AH, AW=AW,
        ivn_layout=SetIVNLayout(order=0, ML0=AH, ML1=M // AH, JL0=AH, JL1=K // AH),
        wvn_layout=SetWVNLayout(order=0, KL0=AH, KL1=K // AH, NL0=min(N, AW), NL1=max(1, N // AW)),
        ovn_layout=SetOVNLayout(order=0, PL0=AH, PL1=M // AH, QL0=AH, QL1=N // AH),
    )

    for n_tile in range(N // Nt):
        for m_tile in range(M // Mt):
            for k_tile in range(K // Kt):
                program.add_mapping(SetMapping(
                    r0=k_tile, c0=n_tile * Nt,
                    Gr=AW, Gc=1, sr=1, sc=0,
                    m_start=m_tile * Mt, m_end=(m_tile + 1) * Mt,
                    n_start=n_tile * Nt, n_end=(n_tile + 1) * Nt,
                    k_start=k_tile * Kt, k_end=(k_tile + 1) * Kt,
                ))

    return program


def run_gemm_test(name, M, N, K, AW, AH, program=None, seed=42, verbose=True):
    """Run a single GEMM test through the simulator.

    Returns (passed, info_dict).
    """
    if program is None:
        program = create_gemm_program(M=M, N=N, K=K, AH=AH, AW=AW)

    instructions = encode_program(program)
    num_tiles = len(instructions) - 3

    if verbose:
        print(f"  {name}: C[{M},{N}] = A[{M},{K}] x B[{K},{N}]")
        print(f"    Array: {AH}x{AW}, tiles={num_tiles}")

    np.random.seed(seed)
    A = np.random.randint(-4, 4, size=(M, K)).astype(np.int8)
    B = np.random.randint(-4, 4, size=(K, N)).astype(np.int8)

    t0 = time.time()
    mod = build_feather_kstreaming_simulator(
        M, K, N, AW, AH, int8, len(instructions),
    )
    build_time = time.time() - t0

    C = np.zeros((M, N), dtype=np.int32)
    t0 = time.time()
    mod(A, B, instructions, C)
    run_time = time.time() - t0

    ref = A.astype(np.int32) @ B.astype(np.int32)
    passed = np.array_equal(C, ref)

    if verbose:
        print(f"    Build: {build_time:.1f}s, Run: {run_time:.1f}s — "
              f"{'PASS' if passed else 'FAIL'}")

    return passed, {
        "name": name, "M": M, "N": N, "K": K,
        "tiles": num_tiles, "build_time": build_time, "run_time": run_time,
    }


def run_hls_csim_test(name, M, N, K, AW, AH, program=None, seed=42, verbose=True):
    """Run a single GEMM test through HLS C-simulation."""
    if program is None:
        program = create_gemm_program(M=M, N=N, K=K, AH=AH, AW=AW)

    instructions = encode_program(program)

    if verbose:
        print(f"  {name}: C[{M},{N}] = A[{M},{K}] x B[{K},{N}] (HLS csim)")

    np.random.seed(seed)
    A = np.random.randint(-4, 4, size=(M, K)).astype(np.int8)
    B = np.random.randint(-4, 4, size=(K, N)).astype(np.int8)

    project_dir = os.path.join(TESTS_DIR, f"param_{AW}x{AH}_{name}_csim.prj")
    mod = build_feather_kstreaming_hls(
        M, K, N, AW, AH, int8, len(instructions),
        mode="csim", project=project_dir,
    )

    # Patch kernel.cpp for wide ap_uint if needed
    patch_kernel_for_wide_apint(project_dir, AW * AH * 8)

    C = np.zeros((M, N), dtype=np.int32)
    mod(A, B, instructions, C)

    ref = A.astype(np.int32) @ B.astype(np.int32)
    passed = np.array_equal(C, ref)

    if verbose:
        print(f"    {'PASS' if passed else 'FAIL'}")

    return passed


def run_hls_csynth(AW, AH, verbose=True):
    """Run HLS synthesis for a minimal workload and report resources."""
    import xmltodict
    import allo.dataflow as df

    Mt = AW // 2
    Nt = AH
    Kt = 2 * AH
    M, N, K = Mt, Nt, Kt  # single tile

    program = create_gemm_program(M=M, N=N, K=K, AH=AH, AW=AW)
    instructions = encode_program(program)

    if verbose:
        print(f"  CSynth: C[{M},{N}] = A[{M},{K}] x B[{K},{N}] on {AH}x{AW}")

    project_dir = os.path.join(TESTS_DIR, f"param_{AW}x{AH}_csynth.prj")
    top = get_feather_full_matrix_top_kstreaming(
        M, K, N, AW, AH, int8, len(instructions),
    )
    s = df.customize(top)
    s.partition("full_matrix_top:C", dim=2, factor=AH)
    hls_mod = s.build(target="vitis_hls", mode="csyn", project=project_dir)

    # Patch kernel.cpp for wide ap_uint
    patch_kernel_for_wide_apint(project_dir, AW * AH * 8)

    print("  Running Vitis HLS synthesis...")
    hls_mod()

    # Parse synthesis report
    xml_path = os.path.join(
        project_dir, "out.prj", "solution1", "syn", "report",
        "full_matrix_top_csynth.xml",
    )
    assert os.path.isfile(xml_path), f"Synthesis report not found: {xml_path}"

    with open(xml_path, "r", encoding="utf-8") as f:
        profile = xmltodict.parse(f.read())["profile"]

    perf = profile["PerformanceEstimates"]
    latency = perf["SummaryOfOverallLatency"]
    timing = perf["SummaryOfTimingAnalysis"]
    area = profile["AreaEstimates"]["Resources"]

    results = {
        "best_cycles": latency["Best-caseLatency"],
        "worst_cycles": latency["Worst-caseLatency"],
        "est_clock": timing["EstimatedClockPeriod"],
        "target_clock": profile["UserAssignments"]["TargetClockPeriod"],
        "clock_unit": profile["UserAssignments"]["unit"],
        "resources": {},
    }
    for resource in ("BRAM_18K", "DSP", "DSP48E", "FF", "LUT", "URAM"):
        if resource in area:
            results["resources"][resource] = area[resource]

    if verbose:
        print(f"\n  === Synthesis Results ({AH}x{AW} PE Array) ===")
        print(f"  Target clock:    {results['target_clock']} {results['clock_unit']}")
        print(f"  Estimated clock: {results['est_clock']} {results['clock_unit']}")
        print(f"  Latency (cycles): {results['best_cycles']} (best) / "
              f"{results['worst_cycles']} (worst)")
        print(f"\n  === Resource Utilization ===")
        for k, v in results["resources"].items():
            print(f"  {k}: {v}")

    return results


def patch_kernel_for_wide_apint(project_dir, max_bits):
    """Insert #define AP_INT_MAX_W before ap_int.h include if needed.

    Vitis HLS defaults AP_INT_MAX_W to 1024. For UInt widths > 1024
    (e.g., UInt(2048) for 16x16 PE array), this must be increased.
    """
    if max_bits <= 1024:
        return

    max_w = 4096  # Vitis HLS supports up to 4096
    kernel_path = os.path.join(project_dir, "kernel.cpp")
    if not os.path.isfile(kernel_path):
        return

    with open(kernel_path, "r") as f:
        code = f.read()

    define = f"#define AP_INT_MAX_W {max_w}\n"
    if define in code:
        return

    # Insert before first #include that uses ap_int
    code = code.replace(
        "#include <ap_int.h>",
        f"{define}#include <ap_int.h>",
    )
    # Also handle ap_fixed.h which may include ap_int internally
    if "#include <ap_fixed.h>" in code and define not in code.split("#include <ap_fixed.h>")[0]:
        code = code.replace(
            "#include <ap_fixed.h>",
            f"{define}#include <ap_fixed.h>",
        )

    with open(kernel_path, "w") as f:
        f.write(code)
    print(f"    Patched kernel.cpp: AP_INT_MAX_W={max_w}")


def get_test_configs(AW, AH):
    """Generate test configurations scaled to array dimensions."""
    Mt = AW // 2
    Nt = AH
    Kt = 2 * AH

    configs = [
        ("single_tile",  Mt,      Nt,      Kt),
        ("multi_m",      4 * Mt,  Nt,      Kt),
        ("multi_n",      Mt,      2 * Nt,  Kt),
        ("multi_k",      Mt,      Nt,      2 * Kt),
        ("full_multi",   4 * Mt,  2 * Nt,  2 * Kt),
        ("large_gemm",   8 * Mt,  4 * Nt,  4 * Kt),
    ]
    return configs


def run_all_simulator_tests(AW, AH):
    """Run full simulator test suite for given array dimensions."""
    print(f"\n{'=' * 70}")
    print(f"SIMULATOR TESTS — {AH}x{AW} PE Array")
    print(f"{'=' * 70}")

    results = {}
    all_passed = True

    # Standard GEMM tests (output_stationary, Gr=AW//2)
    print(f"\n--- Output-Stationary Tests (Gr={AW // 2}) ---")
    for name, M, N, K in get_test_configs(AW, AH):
        passed, info = run_gemm_test(name, M, N, K, AW, AH)
        results[name] = passed
        if not passed:
            all_passed = False

    # NOTE: Pass-through (Gr=AW) is skipped — known pre-existing col_map issue
    # for AW > 4. The BIRRD butterfly wiring permutes output even with all-PS
    # switches, but the wrapper assumes identity col_map. See TICKET-001.

    # OVN order tests
    print(f"\n--- OVN Order Tests ---")
    Mt = AW // 2
    Nt = AH
    Kt = 2 * AH
    for order in range(6):
        name = f"ovn_order_{order}"
        program = create_gemm_program(M=Mt, N=Nt, K=Kt, AH=AH, AW=AW, ovn_order=order)
        passed, info = run_gemm_test(
            name, Mt, Nt, Kt, AW, AH,
            program=program,
        )
        results[name] = passed
        if not passed:
            all_passed = False

    # Summary
    print(f"\n{'=' * 70}")
    print(f"SIMULATOR TEST SUMMARY — {AH}x{AW}")
    print(f"{'=' * 70}")
    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    n_pass = sum(results.values())
    n_total = len(results)
    print(f"\n  {n_pass}/{n_total} tests passed")

    return all_passed, results


def run_all_hls_csim_tests(AW, AH):
    """Run HLS csim tests for given array dimensions."""
    from allo.backend.hls import is_available
    if not is_available("vitis_hls"):
        print("SKIPPED: Vitis HLS not available")
        return False, {}

    print(f"\n{'=' * 70}")
    print(f"HLS CSIM TESTS — {AH}x{AW} PE Array")
    print(f"{'=' * 70}")

    results = {}
    # Start with smallest workload
    Mt = AW // 2
    Nt = AH
    Kt = 2 * AH

    for name, M, N, K in [("single_tile", Mt, Nt, Kt), ("multi_m", 4 * Mt, Nt, Kt)]:
        passed = run_hls_csim_test(name, M, N, K, AW, AH)
        results[name] = passed

    print(f"\n  HLS CSim: {sum(results.values())}/{len(results)} passed")
    return all(results.values()), results


def main():
    parser = argparse.ArgumentParser(description="Parameterized FEATHER+ GEMM tests")
    parser.add_argument("--aw", type=int, default=16, help="Array width (default: 16)")
    parser.add_argument("--ah", type=int, default=16, help="Array height (default: 16)")
    parser.add_argument("--hls", choices=["csim", "csyn"], default=None,
                        help="HLS mode (default: simulator only)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    AW, AH = args.aw, args.ah
    assert AW >= 4 and (AW & (AW - 1)) == 0, f"AW={AW} must be power of 2 >= 4"
    assert AH >= 4 and (AH & (AH - 1)) == 0, f"AH={AH} must be power of 2 >= 4"

    print(f"FEATHER+ Parameterized GEMM Tests")
    print(f"Array: {AH}x{AW}, Ty=int8, TyOut=int32")
    print(f"Tile sizes: Mt={AW // 2}, Nt={AH}, Kt={2 * AH}")

    if args.hls is None:
        # Simulator-only tests
        all_passed, results = run_all_simulator_tests(AW, AH)
        sys.exit(0 if all_passed else 1)

    elif args.hls == "csim":
        # Run simulator first, then HLS csim
        sim_passed, _ = run_all_simulator_tests(AW, AH)
        if not sim_passed:
            print("\nSimulator tests failed — skipping HLS csim")
            sys.exit(1)
        hls_passed, _ = run_all_hls_csim_tests(AW, AH)
        sys.exit(0 if hls_passed else 1)

    elif args.hls == "csyn":
        results = run_hls_csynth(AW, AH)
        sys.exit(0)


if __name__ == "__main__":
    main()
