#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""U280 FPGA deployment test for FEATHER+ MINISA.

Generates an HLS hardware project, builds the xclbin bitstream,
and runs on the Xilinx U280 FPGA to measure real-device latency.

Usage:
    # Step 1: Generate project + start build (takes 4-12 hours)
    python tests/test_u280_deploy.py --stage build

    # Step 2: Run on device (after build completes)
    python tests/test_u280_deploy.py --stage run

    # Or do everything in one shot (build + run):
    python tests/test_u280_deploy.py --stage all
"""

import os
import sys
import re
import argparse
import subprocess
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from allo.ir.types import int8
from allo.backend.hls import is_available
import allo.dataflow as df

from minisa.isa import create_figure7_program, encode_program
from feather_minisa import (
    get_feather_full_matrix_top_kstreaming,
    FeatherKStreamingModule,
    compute_birrd_params,
)

# Figure 7 dimensions
M, K, N = 16, 12, 8
AH, AW = 4, 4

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(TESTS_DIR, "feather_u280_hw.prj")


def generate_hw_project():
    """Generate HLS project with mode='hw' for U280 deployment."""
    print("\n" + "=" * 60)
    print("Step 1: Generate FEATHER+ HLS Project for U280")
    print("=" * 60)

    program = create_figure7_program()
    instructions = encode_program(program)
    num_inst = len(instructions)
    num_tiles = num_inst - 3
    print(f"  Workload: C[{M},{N}] = A[{M},{K}] x B[{K},{N}]")
    print(f"  NEST: {AH}x{AW}, {num_tiles} tile mappings")
    print(f"  Project: {PROJECT_DIR}")

    # Build with mode="hw" — generates kernel.cpp, host.cpp, Makefile
    top = get_feather_full_matrix_top_kstreaming(
        M, K, N, AW, AH, int8, num_inst,
    )
    s = df.customize(top)
    s.partition("full_matrix_top:C", dim=2, factor=AH)
    s.partition("full_matrix_top:A", dim=2, factor=K)
    s.partition("full_matrix_top:B", dim=2, factor=N)
    hls_mod = s.build(target="vitis_hls", mode="hw", project=PROJECT_DIR)

    print("  Project files generated.")
    # List generated files
    for f in sorted(os.listdir(PROJECT_DIR)):
        fpath = os.path.join(PROJECT_DIR, f)
        size = os.path.getsize(fpath) if os.path.isfile(fpath) else 0
        print(f"    {f:30s} {size:>8d} bytes" if size else f"    {f:30s} (dir)")

    return hls_mod


def write_input_data():
    """Write input data files for the host executable."""
    print("\n  Writing input data files...")
    from minisa.isa import SetOVNLayout
    from minisa.lowering import (
        lower_ovn_layout, compute_col_to_m_map,
        compute_output_col_map, _simulate_birrd_passthrough_perm,
    )

    program = create_figure7_program()
    instructions = encode_program(program)
    num_inst = len(instructions)
    num_tiles = num_inst - 3
    P0, P1 = compute_birrd_params(AW)

    np.random.seed(7)
    A = np.random.randint(-4, 4, size=(M, K)).astype(np.int8)
    B = np.random.randint(-4, 4, size=(K, N)).astype(np.int8)
    C_ref = A.astype(np.int32) @ B.astype(np.int32)

    # Compute BIRRD tables (same as FeatherKStreamingModule.__call__)
    ovn_order = int(instructions[2, 1])
    ovn = SetOVNLayout(order=ovn_order, PL0=AW, PL1=1, QL0=AW, QL1=1)
    birrd_table = lower_ovn_layout(ovn, AW, AW)
    passthrough_perm = _simulate_birrd_passthrough_perm(AW)
    pair_to_col = compute_output_col_map(AW, ovn_order)

    birrd_per_tile = np.zeros((num_tiles, P0, P1), dtype=np.int8)
    col_map_per_tile = np.zeros((num_tiles, AW), dtype=np.int32)
    num_m_per_tile = np.zeros(num_tiles, dtype=np.int32)
    n_base_per_tile = np.zeros((num_tiles, AW), dtype=np.int32)

    for t in range(num_tiles):
        Gr = int(instructions[3 + t, 3])
        Gc = int(instructions[3 + t, 4])
        sc = int(instructions[3 + t, 6])
        mask_Gc = Gc - 1
        if Gr == AW:
            col_map_per_tile[t] = compute_col_to_m_map(AW, ovn_order, AW)
            num_m_per_tile[t] = AW
            for col in range(AW):
                orig_pe = int(passthrough_perm[col])
                n_base_per_tile[t, col] = sc * (orig_pe & mask_Gc)
        elif Gr == 1:
            col_map_per_tile[t] = np.zeros(AW, dtype=np.int32)
            num_m_per_tile[t] = 1
            for col in range(AW):
                orig_pe = int(passthrough_perm[col])
                n_base_per_tile[t, col] = sc * (orig_pe & mask_Gc)
        else:
            birrd_per_tile[t] = birrd_table
            col_map_per_tile[t] = compute_col_to_m_map(AW, ovn_order, Gr)
            num_m_per_tile[t] = Gr
            for pair_idx in range(AW // 2):
                col = int(pair_to_col[pair_idx])
                n_base_per_tile[t, col] = sc * (pair_idx & mask_Gc)

    m_start_per_tile = np.array(
        [int(instructions[3 + t, 7]) for t in range(num_tiles)], dtype=np.int32
    )
    n_start_per_tile = np.array(
        [int(instructions[3 + t, 9]) for t in range(num_tiles)], dtype=np.int32
    )

    C = np.zeros((M, N), dtype=np.int32)

    # Write binary data files in Allo's expected order:
    # (kernel-grouped reordering: crossbar_load args, inst_rw args, output_accum args)
    arrays = [
        ("A", A),
        ("B", B),
        ("instructions", instructions),
        ("birrd_inst", birrd_per_tile),
        ("output_col_map", col_map_per_tile),
        ("output_num_m", num_m_per_tile),
        ("output_n_base", n_base_per_tile),
        ("accum_m_start", m_start_per_tile),
        ("accum_n_start", n_start_per_tile),
        ("C", C),
    ]

    for i, (name, arr) in enumerate(arrays):
        path = os.path.join(PROJECT_DIR, f"input{i}.data")
        with open(path, "wb") as f:
            f.write(arr.tobytes())
        print(f"    input{i}.data: {name:20s} shape={str(arr.shape):16s} {arr.nbytes} bytes")

    # Save reference for verification
    ref_path = os.path.join(PROJECT_DIR, "c_ref.npy")
    np.save(ref_path, C_ref)
    print(f"    c_ref.npy: reference output saved")

    return C_ref


def build_bitstream():
    """Run v++ compilation to generate xclbin bitstream."""
    print("\n" + "=" * 60)
    print("Step 2: Build FPGA Bitstream (v++ compile + link)")
    print("=" * 60)

    xdevice = os.environ.get("XDEVICE", "")
    if not xdevice:
        print("  ERROR: XDEVICE environment variable not set")
        print("  Set it to: /opt/xilinx/platforms/xilinx_u280_gen3x16_xdma_1_202211_1/xilinx_u280_gen3x16_xdma_1_202211_1.xpfm")
        return False

    print(f"  XDEVICE: {xdevice}")
    print(f"  This will take several hours...")

    cmd = f"cd {PROJECT_DIR} && make build TARGET=hw PLATFORM=$XDEVICE"
    print(f"  Command: {cmd}")
    print(f"  Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    stdout, _ = process.communicate()
    log = stdout.decode("utf-8", errors="replace")

    log_path = os.path.join(PROJECT_DIR, "build_hw.log")
    with open(log_path, "w") as f:
        f.write(log)

    print(f"  End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Build log: {log_path}")

    if process.returncode != 0:
        print(f"  BUILD FAILED (exit code {process.returncode})")
        for line in log.strip().split("\n")[-20:]:
            print(f"    {line}")
        return False

    # Check for xclbin
    xsa = xdevice.rsplit("/", 1)[-1].split(".")[0]
    xclbin_path = os.path.join(PROJECT_DIR, f"build_dir.hw.{xsa}", "full_matrix_top.xclbin")
    if os.path.exists(xclbin_path):
        size_mb = os.path.getsize(xclbin_path) / (1024 * 1024)
        print(f"  Bitstream generated: {xclbin_path} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"  WARNING: xclbin not found at {xclbin_path}")
        # List build dir
        build_dir = os.path.join(PROJECT_DIR, f"build_dir.hw.{xsa}")
        if os.path.isdir(build_dir):
            for f in os.listdir(build_dir):
                print(f"    {f}")
        return False


def run_on_device():
    """Run the compiled design on U280 FPGA and measure latency."""
    print("\n" + "=" * 60)
    print("Step 3: Run on U280 FPGA")
    print("=" * 60)

    xdevice = os.environ.get("XDEVICE", "")
    xsa = xdevice.rsplit("/", 1)[-1].split(".")[0] if xdevice else ""

    # Find xclbin
    xclbin_path = None
    for candidate in [
        os.path.join(PROJECT_DIR, f"build_dir.hw.{xsa}", "full_matrix_top.xclbin"),
        os.path.join(PROJECT_DIR, "build_dir.hw.*", "full_matrix_top.xclbin"),
    ]:
        import glob
        matches = glob.glob(candidate)
        if matches:
            xclbin_path = matches[0]
            break

    if xclbin_path is None or not os.path.exists(xclbin_path):
        print(f"  ERROR: No xclbin found. Run --stage build first.")
        return None

    print(f"  Bitstream: {xclbin_path}")

    # Build host if needed
    host_exe = os.path.join(PROJECT_DIR, "full_matrix_top")
    if not os.path.exists(host_exe):
        print("  Building host executable...")
        cmd = f"cd {PROJECT_DIR} && make host PLATFORM=$XDEVICE"
        process = subprocess.Popen(cmd, shell=True)
        process.wait()
        if process.returncode != 0:
            print("  ERROR: Failed to build host")
            return None

    # Run on device
    print(f"  Running on U280...")
    cmd = f"cd {PROJECT_DIR} && ./full_matrix_top {xclbin_path}"
    print(f"  Command: {cmd}")

    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    stdout, _ = process.communicate()
    output = stdout.decode("utf-8", errors="replace")

    print("\n  === Device Output ===")
    for line in output.strip().split("\n"):
        print(f"    {line}")

    # Parse execution time
    exe_time_ns = None
    for line in output.split("\n"):
        # Look for "| full_matrix_top |          NNNN |" pattern
        m = re.search(r"\|\s*\w+\s*\|\s*(\d+)\s*\|", line)
        if m:
            exe_time_ns = int(m.group(1))

    # Verify output
    ref_path = os.path.join(PROJECT_DIR, "c_ref.npy")
    out_path = os.path.join(PROJECT_DIR, "output0.data")
    if os.path.exists(ref_path) and os.path.exists(out_path):
        C_ref = np.load(ref_path)
        C_out = np.fromfile(out_path, dtype=np.int32).reshape(M, N)
        if np.array_equal(C_out, C_ref):
            print("\n  OUTPUT VERIFICATION: PASSED")
        else:
            mismatches = np.sum(C_out != C_ref)
            print(f"\n  OUTPUT VERIFICATION: FAILED ({mismatches} mismatches)")
            print(f"    Expected:\n{C_ref}")
            print(f"    Got:\n{C_out}")
    else:
        print("\n  Could not verify output (missing reference or output file)")

    if exe_time_ns is not None:
        # Compute cycles at 300 MHz
        freq_mhz = 300
        clock_period_ns = 1000.0 / freq_mhz
        cycles = exe_time_ns / clock_period_ns
        print(f"\n  === Latency Results ===")
        print(f"  Wall-clock time: {exe_time_ns} ns")
        print(f"  Clock frequency: {freq_mhz} MHz ({clock_period_ns:.2f} ns period)")
        print(f"  Estimated cycles: {cycles:.0f}")
        print(f"  RTL cosim cycles: 1052")
        print(f"  RTL reference:    1120")
        print(f"  Ratio vs cosim:   {cycles/1052:.2f}x")
        print(f"  Ratio vs RTL ref: {cycles/1120:.2f}x")

    if process.returncode != 0:
        print(f"\n  Device run failed (exit code {process.returncode})")
        return None

    return exe_time_ns


def main():
    parser = argparse.ArgumentParser(description="U280 FPGA deployment test")
    parser.add_argument("--stage", choices=["build", "run", "all", "generate"],
                        default="generate",
                        help="Stage to run: generate (project only), build, run, or all")
    args = parser.parse_args()

    if args.stage in ("generate", "build", "all"):
        if not is_available("vitis_hls"):
            print("ERROR: Vitis HLS not available. Source settings64.sh first.")
            sys.exit(1)

        hls_mod = generate_hw_project()
        write_input_data()

    if args.stage in ("build", "all"):
        success = build_bitstream()
        if not success and args.stage == "all":
            print("\nBuild failed. Cannot proceed to run stage.")
            sys.exit(1)

    if args.stage in ("run", "all"):
        exe_time = run_on_device()
        if exe_time is not None:
            print(f"\nU280 deployment test completed: {exe_time} ns")
        else:
            print("\nU280 deployment test: could not measure latency")


if __name__ == "__main__":
    main()
