# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test for Phase 2: split crossbar/NEST kernels (K-streaming v2).

Verifies that the split-kernel architecture produces correct GEMM results
and measures cycle count via HLS csim/csynth/cosim.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from allo.ir.types import int8
from minisa.isa import create_figure7_program, encode_program

# Hardware parameters
AH = 4
AW = 4
M, K, N = 16, 12, 8


def test_figure7_v2_functional_gemm():
    """End-to-end GEMM with split crossbar/NEST kernels (simulator)."""
    from feather_minisa import build_feather_kstreaming_v2_simulator

    program = create_figure7_program()
    instructions = encode_program(program)
    num_k_passes = K // AH  # 3
    Kt_per_pass = AH        # 4

    np.random.seed(7)
    A = np.random.randint(-4, 4, size=(M, K)).astype(np.int8)
    B = np.random.randint(-4, 4, size=(K, N)).astype(np.int8)

    mod = build_feather_kstreaming_v2_simulator(
        M, K, N, AW, AH, int8, len(instructions),
        num_k_passes, Kt_per_pass,
    )
    C = np.zeros((M, N), dtype=np.int32)
    mod(A, B, instructions, C)

    ref = A.astype(np.int32) @ B.astype(np.int32)
    np.testing.assert_array_equal(C, ref)


def test_figure7_v2_hls_csim():
    """HLS C-simulation correctness check for split-kernel design."""
    from feather_minisa import build_feather_kstreaming_v2_hls

    program = create_figure7_program()
    instructions = encode_program(program)
    num_k_passes = K // AH
    Kt_per_pass = AH

    np.random.seed(7)
    A = np.random.randint(-4, 4, size=(M, K)).astype(np.int8)
    B = np.random.randint(-4, 4, size=(K, N)).astype(np.int8)

    mod = build_feather_kstreaming_v2_hls(
        M, K, N, AW, AH, int8, len(instructions),
        num_k_passes, Kt_per_pass,
        mode="csim",
        project=os.path.join(os.path.dirname(__file__), "figure7_v2_csim.prj"),
    )
    C = np.zeros((M, N), dtype=np.int32)
    mod(A, B, instructions, C)

    ref = A.astype(np.int32) @ B.astype(np.int32)
    np.testing.assert_array_equal(C, ref)


def test_figure7_v2_hls_csynth():
    """HLS synthesis cycle estimate for split-kernel design."""
    import xmltodict
    import allo.dataflow as df
    from feather_minisa import get_feather_full_matrix_top_kstreaming_v2

    program = create_figure7_program()
    instructions = encode_program(program)
    num_k_passes = K // AH
    Kt_per_pass = AH
    project_dir = os.path.join(os.path.dirname(__file__), "figure7_v2_csynth.prj")

    top = get_feather_full_matrix_top_kstreaming_v2(
        M, K, N, AW, AH, int8, len(instructions),
        num_k_passes, Kt_per_pass,
    )
    s = df.customize(top)
    s.partition("full_matrix_top:C", dim=2, factor=AH)
    hls_mod = s.build(target="vitis_hls", mode="csyn", project=project_dir)

    print("\n  Running Vitis HLS synthesis...")
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
    area = profile["AreaEstimates"]["Resources"]

    best_cycles = latency["Best-caseLatency"]
    worst_cycles = latency["Worst-caseLatency"]
    print(f"  Cycles: {best_cycles} (best) / {worst_cycles} (worst)")
    print(f"  Resources: BRAM={area['BRAM_18K']}, DSP={area['DSP']}, "
          f"FF={area['FF']}, LUT={area['LUT']}")


def test_figure7_v2_cosim():
    """RTL co-simulation for split-kernel design (cycle-accurate).

    Uses the same approach as test_figure7_cosim.py: generate kernel.cpp via
    csyn mode, patch m_axi depths, write C testbench, run csynth + cosim.
    """
    import re
    import subprocess
    import allo.dataflow as df
    from allo.backend.hls import is_available
    from feather_minisa import (
        get_feather_full_matrix_top_kstreaming_v2,
        compute_birrd_params,
    )
    from test_figure7_cosim import (
        patch_kernel_for_cosim,
        generate_cosim_testbench,
        generate_cosim_tcl,
    )

    if not is_available("vitis_hls"):
        print("SKIPPED: Vitis HLS not available")
        return

    program = create_figure7_program()
    instructions = encode_program(program)
    num_inst = len(instructions)
    num_tiles = num_inst - 3
    num_k_passes = K // AH
    Kt_per_pass = AH

    np.random.seed(7)
    A = np.random.randint(-4, 4, size=(M, K)).astype(np.int8)
    B = np.random.randint(-4, 4, size=(K, N)).astype(np.int8)
    C_ref = A.astype(np.int32) @ B.astype(np.int32)

    project_dir = os.path.join(os.path.dirname(__file__), "figure7_v2_cosim.prj")

    # Build in csyn mode to generate kernel.cpp
    top = get_feather_full_matrix_top_kstreaming_v2(
        M, K, N, AW, AH, int8, num_inst, num_k_passes, Kt_per_pass,
    )
    s = df.customize(top)
    s.partition("full_matrix_top:C", dim=2, factor=AH)
    hls_mod = s.build(target="vitis_hls", mode="csyn", project=project_dir)

    # Patch kernel, write testbench, generate TCL
    patch_kernel_for_cosim(project_dir, num_tiles)
    generate_cosim_testbench(project_dir, A, B, instructions, C_ref)
    generate_cosim_tcl(project_dir)

    # Run csynth + cosim
    print("\n  Running csynth + cosim_design...")
    cmd = f"cd {project_dir}; vitis_hls -f run_cosim.tcl"
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    stdout, _ = process.communicate()
    log = stdout.decode("utf-8", errors="replace")

    log_path = os.path.join(project_dir, "cosim.log")
    with open(log_path, "w") as f:
        f.write(log)

    if process.returncode != 0:
        print(f"  Cosim failed (exit code {process.returncode})")
        for line in log.strip().split("\n")[-30:]:
            print(f"    {line}")
        raise RuntimeError("Cosim failed")

    # Parse cycle count
    cycles = None
    sim_dir = os.path.join(project_dir, "out.prj", "solution1", "sim")
    cosim_rpt = os.path.join(sim_dir, "report", "full_matrix_top_cosim.rpt")
    if os.path.isfile(cosim_rpt):
        with open(cosim_rpt, "r") as f:
            rpt = f.read()
        for line in rpt.split("\n"):
            m = re.search(r"Verilog\|\s*Pass\|\s*(\d+)", line)
            if m:
                cycles = int(m.group(1))

    if cycles is None:
        for subdir in ["verilog", "vhdl"]:
            txn_path = os.path.join(sim_dir, "report", subdir, "result.transaction.rpt")
            if os.path.isfile(txn_path):
                with open(txn_path, "r") as f:
                    txn = f.read()
                for line in txn.split("\n"):
                    m = re.search(r"transaction\s+\d+:\s+(\d+)", line)
                    if m:
                        cycles = int(m.group(1))

    assert cycles is not None, f"Could not extract cycle count. See {log_path}"
    print(f"  V2 split-kernel RTL cosim: {cycles} cycles")
    print(f"  V1 K-streaming cosim: 1208 cycles")
    print(f"  RTL reference: 1120 cycles")
    print(f"  Ratio vs RTL: {cycles/1120:.2f}x")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["sim", "csim", "csynth", "cosim", "all"],
                        default="sim", help="Which test to run")
    args = parser.parse_args()

    if args.test in ("sim", "all"):
        print("  test_figure7_v2_functional_gemm ... ", end="", flush=True)
        test_figure7_v2_functional_gemm()
        print("PASSED")

    if args.test in ("csim", "all"):
        print("  test_figure7_v2_hls_csim ... ", end="", flush=True)
        test_figure7_v2_hls_csim()
        print("PASSED")

    if args.test in ("csynth", "all"):
        print("  test_figure7_v2_hls_csynth ... ", end="", flush=True)
        test_figure7_v2_hls_csynth()
        print("PASSED")

    if args.test in ("cosim", "all"):
        print("  test_figure7_v2_cosim ... ", end="", flush=True)
        test_figure7_v2_cosim()
        print("PASSED")
