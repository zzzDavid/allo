# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""RTL co-simulation for Figure 7 via Vitis HLS cosim_design.

Generates the HLS project, writes a C testbench with actual Figure 7 data,
runs csynth + cosim_design, and extracts the cycle-accurate latency from
the RTL simulation report.
"""

import os
import sys
import re
import subprocess

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


def _c_array_1d(name, dtype, data):
    """Generate a C array initializer for a 1D array."""
    vals = ", ".join(str(int(x)) for x in data.flat)
    return f"{dtype} {name}[{data.size}] = {{{vals}}};\n"


def generate_cosim_testbench(project_dir, A, B, instructions, C_ref):
    """Write a C testbench that calls full_matrix_top with Figure 7 data."""
    from minisa.isa import SetOVNLayout
    from minisa.lowering import (
        lower_ovn_layout, compute_col_to_m_map,
        compute_output_col_map, _simulate_birrd_passthrough_perm,
    )

    num_inst = len(instructions)
    num_tiles = num_inst - 3
    P0, P1 = compute_birrd_params(AW)

    # Compute BIRRD configs (same logic as FeatherKStreamingModule.__call__)
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
    # Write testbench
    tb_path = os.path.join(project_dir, "tb.cpp")
    with open(tb_path, "w") as f:
        f.write('#include <cstdint>\n#include <cstdio>\n#include <cstring>\n\n')
        f.write('#include "kernel.h"\n\n')
        f.write("int main() {\n")

        # Flatten all arrays as 1D C arrays
        # Order must match Allo's kernel-grouped reordering:
        # crossbar_load: A, B, instructions
        # inst_rw: birrd_inst
        # output_accum: output_col_map, output_num_m, output_n_base,
        #               accum_m_start, accum_n_start, C
        # (nest_compute and bus have no DRAM args — they use streams)
        f.write("  // Input matrices (crossbar_load)\n")
        f.write("  " + _c_array_1d("A", "int8_t", A.flatten()))
        f.write("  " + _c_array_1d("B", "int8_t", B.flatten()))
        f.write("  // Instructions (crossbar_load)\n")
        f.write("  " + _c_array_1d("instructions", "int32_t", instructions.flatten()))
        f.write("  // BIRRD per-tile instructions (inst_rw)\n")
        f.write("  " + _c_array_1d("birrd_inst", "int8_t", birrd_per_tile.flatten()))
        f.write("  // Output column map (output_accum)\n")
        f.write("  " + _c_array_1d("output_col_map", "int32_t", col_map_per_tile.flatten()))
        f.write("  // Output num_m per tile\n")
        f.write("  " + _c_array_1d("output_num_m", "int32_t", num_m_per_tile.flatten()))
        f.write("  // Output n_base per tile\n")
        f.write("  " + _c_array_1d("output_n_base", "int32_t", n_base_per_tile.flatten()))
        f.write("  // Accum m_start per tile\n")
        f.write("  " + _c_array_1d("accum_m_start", "int32_t", m_start_per_tile.flatten()))
        f.write("  // Accum n_start per tile\n")
        f.write("  " + _c_array_1d("accum_n_start", "int32_t", n_start_per_tile.flatten()))
        f.write("  // Output matrix (initialized to zero)\n")
        f.write(f"  int32_t C[{M * N}];\n")
        f.write(f"  memset(C, 0, sizeof(C));\n\n")

        # Call top function (order matches HLS kernel-grouped reordering)
        f.write("  // Run FEATHER+ dataflow\n")
        f.write("  full_matrix_top(A, B, instructions, birrd_inst, "
                "output_col_map, output_num_m, output_n_base, "
                "accum_m_start, accum_n_start, C);\n\n")

        # Verify output
        f.write("  // Reference output\n")
        f.write("  " + _c_array_1d("C_ref", "int32_t", C_ref.flatten()))
        f.write("  // Verify\n")
        f.write("  int errors = 0;\n")
        f.write(f"  for (int i = 0; i < {M * N}; i++) {{\n")
        f.write("    if (C[i] != C_ref[i]) {\n")
        f.write('      printf("MISMATCH at %d: got %d, expected %d\\n", i, C[i], C_ref[i]);\n')
        f.write("      errors++;\n")
        f.write("    }\n")
        f.write("  }\n")
        f.write('  if (errors == 0) printf("COSIM PASSED: all outputs match\\n");\n')
        f.write(f'  else printf("COSIM FAILED: %d mismatches\\n", errors);\n')
        f.write("  return errors;\n")
        f.write("}\n")

    return tb_path


def patch_kernel_for_cosim(project_dir, num_tiles):
    """Add depth specs to m_axi pragmas in kernel.cpp for cosim.

    Dynamically detects port variable names from the generated kernel
    (codegen names vary across builds) and adds depth=N so that
    cosim_design can allocate RTL simulation memory.
    """
    kernel_path = os.path.join(project_dir, "kernel.cpp")
    with open(kernel_path, "r") as f:
        code = f.read()

    # Ordered depths matching the 10 kernel-grouped function arguments:
    # crossbar_load: A, B, instructions
    # inst_rw: birrd_inst
    # output_accum: output_col_map, output_num_m, output_n_base,
    #               accum_m_start, accum_n_start, C
    # (nest_compute and bus use streams only — no DRAM args)
    P0, _ = compute_birrd_params(AW)
    num_inst = num_tiles + 3
    ordered_depths = [
        M * K,                        # A: int8[16,12]
        K * N,                        # B: int8[12,8]
        num_inst * 13,                # instructions: int32[num_inst,13]
        num_tiles * P0 * (AW // 2),   # birrd_inst
        num_tiles * AW,               # output_col_map
        num_tiles,                    # output_num_m
        num_tiles * AW,               # output_n_base
        num_tiles,                    # accum_m_start
        num_tiles,                    # accum_n_start
        M * N,                        # C: int32[16,8]
    ]

    # Find all m_axi pragmas in order and add depth
    maxi_re = re.compile(
        r"(#pragma HLS interface m_axi port=\w+ offset=slave bundle=gmem\d+)"
    )
    matches = list(maxi_re.finditer(code))
    assert len(matches) == len(ordered_depths), (
        f"Expected {len(ordered_depths)} m_axi pragmas, found {len(matches)}"
    )
    # Replace in reverse order to preserve string positions
    for match, depth in zip(reversed(matches), reversed(ordered_depths)):
        old = match.group(1)
        new = f"{old} depth={depth}"
        code = code[:match.start()] + new + code[match.end():]

    with open(kernel_path, "w") as f:
        f.write(code)


def generate_cosim_tcl(project_dir):
    """Write a TCL script that runs csynth + cosim."""
    tcl_path = os.path.join(project_dir, "run_cosim.tcl")
    with open(tcl_path, "w") as f:
        f.write("set hls_prj out.prj\n")
        f.write("open_project ${hls_prj} -reset\n")
        f.write("open_solution -reset solution1 -flow_target vivado\n")
        f.write("set_top full_matrix_top\n")
        f.write("add_files kernel.cpp\n")
        f.write('add_files -tb tb.cpp -cflags "-std=gnu++0x"\n')
        f.write('open_solution "solution1"\n')
        f.write("set_part {xcu280-fsvh2892-2L-e}\n")
        f.write("create_clock -period 3.33\n")
        f.write("csynth_design\n")
        f.write("cosim_design\n")
        f.write("exit\n")
    return tcl_path


def run_figure7_cosim():
    """Run Figure 7 RTL co-simulation and report cycle count."""
    print("\n" + "=" * 60)
    print("Figure 7 RTL Co-Simulation (Vitis HLS cosim_design)")
    print("=" * 60)

    if not is_available("vitis_hls"):
        print("SKIPPED: Vitis HLS not available")
        return None

    # Generate test data
    program = create_figure7_program()
    instructions = encode_program(program)
    num_inst = len(instructions)

    np.random.seed(7)
    A = np.random.randint(-4, 4, size=(M, K)).astype(np.int8)
    B = np.random.randint(-4, 4, size=(K, N)).astype(np.int8)
    C_ref = A.astype(np.int32) @ B.astype(np.int32)

    print(f"  Workload: C[{M},{N}] = A[{M},{K}] x B[{K},{N}]")
    print(f"  NEST: {AH}x{AW}, {num_inst - 3} tile mappings")

    num_tiles = num_inst - 3

    # Build HLS project (generates kernel.cpp, kernel.h)
    project_dir = os.path.join(TESTS_DIR, "figure7_cosim.prj")
    top = get_feather_full_matrix_top_kstreaming(
        M, K, N, AW, AH, int8, num_inst,
    )
    s = df.customize(top)
    s.partition("full_matrix_top:C", dim=2, factor=AH)
    # Partition A and B to reduce crossbar_load II (32→8)
    s.partition("full_matrix_top:A", dim=2, factor=K)
    s.partition("full_matrix_top:B", dim=2, factor=N)
    # Build in csyn mode just to generate the project files
    hls_mod = s.build(target="vitis_hls", mode="csyn", project=project_dir)

    # Patch kernel.cpp with depth specs for cosim, write testbench and TCL
    print("  Patching kernel for cosim + generating testbench...")
    patch_kernel_for_cosim(project_dir, num_tiles)
    generate_cosim_testbench(project_dir, A, B, instructions, C_ref)
    generate_cosim_tcl(project_dir)

    # Run csynth + cosim
    print("  Running csynth + cosim (this may take several minutes)...")
    cmd = f"cd {project_dir}; vitis_hls -f run_cosim.tcl"
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    stdout, _ = process.communicate()
    log = stdout.decode("utf-8", errors="replace")

    # Save full log
    log_path = os.path.join(project_dir, "cosim.log")
    with open(log_path, "w") as f:
        f.write(log)

    if process.returncode != 0:
        print(f"  Cosim failed (exit code {process.returncode})")
        print(f"  See log: {log_path}")
        # Print last 30 lines
        for line in log.strip().split("\n")[-30:]:
            print(f"    {line}")
        return None

    # Parse cosim results from all available sources
    cycles = None
    sim_dir = os.path.join(project_dir, "out.prj", "solution1", "sim")

    # 1. Check cosim report
    cosim_rpt = os.path.join(sim_dir, "report", "full_matrix_top_cosim.rpt")
    if os.path.isfile(cosim_rpt):
        with open(cosim_rpt, "r") as f:
            rpt = f.read()
        print(f"\n  === Cosim Report ===")
        for line in rpt.strip().split("\n"):
            print(f"    {line}")
        # Parse "Verilog|Pass|NNNN|NNNN|NNNN|..." table row
        for line in rpt.split("\n"):
            m = re.search(r"Verilog\|\s*Pass\|\s*(\d+)", line)
            if m:
                cycles = int(m.group(1))

    # 2. Check transaction report (Verilog RTL sim)
    for subdir in ["verilog", "vhdl"]:
        txn_path = os.path.join(sim_dir, "report", subdir, "result.transaction.rpt")
        if os.path.isfile(txn_path):
            with open(txn_path, "r") as f:
                txn = f.read()
            print(f"\n  === Transaction Report ({subdir}) ===")
            for line in txn.strip().split("\n"):
                print(f"    {line}")
            # Parse "transaction 0: NNNN" pattern
            for line in txn.split("\n"):
                m = re.search(r"transaction\s+\d+:\s+(\d+)", line)
                if m and cycles is None:
                    cycles = int(m.group(1))

    # 3. Search all report files for latency info
    if cycles is None:
        report_dir = os.path.join(sim_dir, "report")
        if os.path.isdir(report_dir):
            for root, dirs, files in os.walk(report_dir):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    with open(fpath, "r", errors="replace") as f:
                        content = f.read()
                    for line in content.split("\n"):
                        m = re.search(r"(?:Latency|latency)\s*[=:]\s*(\d+)", line)
                        if m:
                            cycles = int(m.group(1))
                            print(f"  Found latency in {fname}: {cycles}")

    # 4. Check log for pass/fail and cycle info
    for line in log.split("\n"):
        line_s = line.strip()
        if "cosim" in line_s.lower() and ("pass" in line_s.lower() or "fail" in line_s.lower()):
            print(f"  LOG: {line_s}")
        if "latency" in line_s.lower() or "cycle" in line_s.lower():
            m = re.search(r"(?:Latency|latency)\s*[=:]\s*(\d+)", line_s)
            if m and cycles is None:
                cycles = int(m.group(1))

    if cycles is not None:
        print(f"\n  RTL Co-Simulation Cycle Count: {cycles}")
    else:
        print("\n  Could not extract cycle count automatically.")
        print(f"  Check log: {log_path}")

    return cycles


if __name__ == "__main__":
    cycles = run_figure7_cosim()
    if cycles is not None:
        print(f"\nFigure 7 RTL Cosim (K-streaming): {cycles} cycles")
        print(f"Previous Allo (24 tiles): 1792 cycles")
        print(f"RTL reference: 1120 cycles")
        print(f"Ratio vs RTL: {cycles/1120:.2f}x")
        print(f"Speedup vs previous: {1792/cycles:.2f}x")
