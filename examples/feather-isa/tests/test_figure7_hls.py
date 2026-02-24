# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""HLS tests for MINISA Figure 7: irregular tiling with mapping adaptation.

Tests the Figure 7 case study (C[16,8] = A[16,12] x B[12,8] on 4x4 NEST
with adaptive Gr) through:
  - Vitis HLS C simulation (csim): functional correctness
  - Vitis HLS C synthesis (csynth): resource/latency reports with cycle count
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import xmltodict

from allo.ir.types import int8
from allo.backend.hls import is_available
import allo.dataflow as df

from minisa.isa import create_figure7_program, encode_program
from feather_minisa import (
    get_feather_full_matrix_top,
    build_feather_full_matrix_hls,
)

HLS_AVAILABLE = is_available("vitis_hls")

# Figure 7 dimensions
M, K, N = 16, 12, 8
AH, AW = 4, 4

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))


def test_figure7_hls_csim():
    """Test Figure 7 GEMM through HLS C simulation.

    Builds the full FEATHER+ dataflow with parametric mapping support
    (adaptive Gr=2 and Gr=4 tiles) and runs through Vitis HLS csim.
    Verifies output matches numpy reference.
    """
    print("\n" + "=" * 60)
    print("Test: Figure 7 HLS CSim")
    print("=" * 60)

    program = create_figure7_program()
    instructions = encode_program(program)
    num_inst = len(instructions)

    print(f"  Workload: C[{M},{N}] = A[{M},{K}] x B[{K},{N}]")
    print(f"  NEST: {AH}x{AW}, {num_inst - 3} tile mappings")

    np.random.seed(7)
    A = np.random.randint(-4, 4, size=(M, K)).astype(np.int8)
    B = np.random.randint(-4, 4, size=(K, N)).astype(np.int8)

    project_dir = os.path.join(TESTS_DIR, "figure7_csim.prj")
    mod = build_feather_full_matrix_hls(
        M, K, N, AW, AH, int8, num_inst,
        mode="csim", project=project_dir,
    )
    C = np.zeros((M, N), dtype=np.int32)
    mod(A, B, instructions, C)

    ref = A.astype(np.int32) @ B.astype(np.int32)
    np.testing.assert_array_equal(C, ref)
    print("  Output matches numpy reference")
    print("PASSED: Figure 7 HLS CSim")


def test_figure7_hls_csynth():
    """Synthesize Figure 7 GEMM and report cycle count.

    Runs Vitis HLS C synthesis on the full FEATHER+ dataflow with
    parametric mapping support. Parses the synthesis report to extract
    and print latency (cycle count) and resource utilization.
    """
    print("\n" + "=" * 60)
    print("Test: Figure 7 HLS CSynth")
    print("=" * 60)

    program = create_figure7_program()
    instructions = encode_program(program)
    num_inst = len(instructions)

    print(f"  Workload: C[{M},{N}] = A[{M},{K}] x B[{K},{N}]")
    print(f"  NEST: {AH}x{AW}, {num_inst - 3} tile mappings")

    # Build HLS module in csyn mode.  The dataflow top function takes
    # separate accum_m_start/accum_n_start arrays and uses a local
    # accumulation buffer, so the generated kernel.cpp is HLS-dataflow-clean
    # (no shared-buffer or multi-writer violations).
    project_dir = os.path.join(TESTS_DIR, "figure7_csynth.prj")
    top = get_feather_full_matrix_top(M, K, N, AW, AH, int8, num_inst)
    s = df.customize(top)
    hls_mod = s.build(
        target="vitis_hls", mode="csyn", project=project_dir,
    )

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

    best_cycles = latency["Best-caseLatency"]
    worst_cycles = latency["Worst-caseLatency"]
    interval_min = latency["Interval-min"]
    interval_max = latency["Interval-max"]
    est_cp = timing["EstimatedClockPeriod"]
    target_cp = profile["UserAssignments"]["TargetClockPeriod"]
    clock_unit = profile["UserAssignments"]["unit"]

    print()
    print(f"  === Synthesis Results ===")
    print(f"  Target clock:    {target_cp} {clock_unit}")
    print(f"  Estimated clock: {est_cp} {clock_unit}")
    print(f"  Latency (cycles): {best_cycles} (best) / {worst_cycles} (worst)")
    print(f"  Interval (cycles): {interval_min} (min) / {interval_max} (max)")
    print()
    print(f"  === Resource Utilization ===")
    for resource in ("BRAM_18K", "DSP", "DSP48E", "FF", "LUT", "URAM"):
        if resource in area:
            print(f"  {resource}: {area[resource]}")
    print()

    # Also print the full report via allo's parse_xml
    from allo.backend.report import parse_xml
    parse_xml(project_dir, "Vitis HLS", "full_matrix_top", print_flag=True)

    print(f"\n  Cycle count: {worst_cycles}")
    print("PASSED: Figure 7 HLS CSynth")


if __name__ == "__main__":
    print("=" * 70)
    print("FIGURE 7 HLS TESTS")
    print("=" * 70)

    if not HLS_AVAILABLE:
        print("SKIPPED: Vitis HLS not available")
        sys.exit(0)

    tests = [
        ("Figure 7 HLS CSim", test_figure7_hls_csim),
        ("Figure 7 HLS CSynth", test_figure7_hls_csynth),
    ]

    results = {}
    for name, test_fn in tests:
        try:
            test_fn()
            results[name] = True
        except Exception as e:
            import traceback
            print(f"FAILED: {name}")
            print(f"  Error: {e}")
            traceback.print_exc()
            results[name] = False

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")

    all_passed = all(results.values())
    print(f"\n{'All' if all_passed else 'Some'} Figure 7 HLS tests {'PASSED' if all_passed else 'FAILED'}")
    sys.exit(0 if all_passed else 1)
