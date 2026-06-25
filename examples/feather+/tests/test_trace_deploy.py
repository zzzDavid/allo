#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""FEATHER+ DEPLOYABLE-variant trace test runner.

Exercises `feather_plus_deploy.py` (baseline multi-process dataflow + the iter70
nearest-neighbour w_chain weight cascade) the same way `tests/test_trace_input.py`
exercises the baseline, by reusing that runner's machinery and swapping in the
deploy builders.

Usage:
    python tests/test_trace_deploy.py instr_trace/figure7_16x12x8_4x4.json
    python tests/test_trace_deploy.py instr_trace/figure7_16x12x8_4x4.json --sim
    python tests/test_trace_deploy.py instr_trace/figure7_16x12x8_4x4.json --hls csim
    python tests/test_trace_deploy.py instr_trace/trace_m24k48n512_16x16.json --hls csyn
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from allo.ir.types import int8
import allo.dataflow as df

import test_trace_input as base
from minisa.trace_parser import load_trace
from feather_plus_deploy import (
    get_feather_deploy_top,
    schedule_deploy_hls,
)
from feather_plus import FeatherModule


def _deploy_top(M, K, N, AW, AH, Ty, num_inst, n_inner=1, k_passes=1,
                Nt_local=None):
    return get_feather_deploy_top(M, K, N, AW, AH, Ty, num_inst, n_inner,
                                  k_passes, Nt_local=Nt_local)


def _deploy_schedule(s, K, N, AH, AW):
    return schedule_deploy_hls(s, K, N, AH, AW)


def _install_deploy_overrides():
    base.get_feather_full_matrix_top = _deploy_top
    base.schedule_feather_hls = _deploy_schedule
    # The baseline csyn flow applies _patch_load_bufs_for_throughput — a
    # POST-GENERATION C++ rewrite that hard-gate 3 forbids the Allo variant
    # from relying on (and at N=512 it packs an ap_uint that overflows
    # AP_INT_MAX_W). Disable so deploy csynth reflects only Allo-generated HLS.
    base._patch_load_bufs_for_throughput = lambda project_dir: None
    # The deploy region's generated HLS top is `deploy_top` — point the cosim
    # flow (set_top, tb.cpp call, *_cosim.rpt filename) at it. Harness wiring only.
    base.TOP_NAME = "deploy_top"
    base.TESTS_DIR = os.path.join(base.TESTS_DIR, "deploy")
    os.makedirs(base.TESTS_DIR, exist_ok=True)


def run_deploy_simulator(trace_info, seed=42):
    """Bit-exact functional check of the deploy variant via the Allo simulator."""
    M = trace_info["M"]; K = trace_info["K"]; N = trace_info["N"]
    M_padded = trace_info["M_padded"]; AH = trace_info["AH"]; AW = trace_info["AW"]
    instructions = trace_info["instructions"]
    n_inner = trace_info.get("n_inner", 1)
    k_passes = trace_info.get("k_passes", 1)

    print(f"\n{'=' * 70}")
    print("ALLO SIMULATOR TEST (deploy variant — w_chain weight cascade)")
    print(f"{'=' * 70}")
    print(f"  Workload: C[{M},{N}] = A[{M},{K}] x B[{K},{N}]   Array: {AH}x{AW}")

    top = get_feather_deploy_top(
        M_padded, K, N, AW, AH, int8, len(instructions), n_inner, k_passes,
    )
    allo_mod = df.build(top, target="simulator")
    mod = FeatherModule(allo_mod, AW, n_inner)

    np.random.seed(seed)
    A_orig = np.random.randint(-4, 4, size=(M, K)).astype(np.int8)
    B = np.random.randint(-4, 4, size=(K, N)).astype(np.int8)
    C_ref = A_orig.astype(np.int32) @ B.astype(np.int32)

    if M_padded != M:
        A = np.zeros((M_padded, K), dtype=np.int8); A[:M, :] = A_orig
    else:
        A = A_orig

    C_padded = np.zeros((M_padded, N), dtype=np.int32)
    inner_params = None
    if "inner_m_starts" in trace_info:
        inner_params = {
            "m_starts": trace_info["inner_m_starts"],
            "n_starts": trace_info["inner_n_starts"],
        }
    mod(A, B, instructions, C_padded, inner_params=inner_params)
    C = C_padded[:M, :]

    passed = np.array_equal(C, C_ref)
    print(f"  OUTPUT VERIFICATION: {'PASS' if passed else 'FAIL'}")
    if not passed:
        n_mismatch = int(np.sum(C != C_ref))
        print(f"  Mismatches: {n_mismatch}/{C.size}")
        print(f"  C[0,:8]   = {C[0,:8]}")
        print(f"  Ref[0,:8] = {C_ref[0,:8]}")
    return passed


def main():
    parser = argparse.ArgumentParser(description="FEATHER+ deploy-variant test")
    parser.add_argument("trace", help="Path to trace JSON file")
    parser.add_argument("--hls", choices=["csim", "csyn", "cosim"], default=None)
    parser.add_argument("--sim", action="store_true",
                        help="Run the Allo simulator functional check")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    trace_path = args.trace
    if not os.path.isabs(trace_path) and not os.path.exists(trace_path):
        trace_path = os.path.join(os.path.dirname(__file__), "..", args.trace)
    if not os.path.exists(trace_path):
        print(f"ERROR: Trace file not found: {args.trace}")
        sys.exit(1)

    print(f"Parsing trace: {trace_path}")
    trace_info = load_trace(trace_path)

    _install_deploy_overrides()

    passed, _ = base.run_reference_test(trace_info, seed=args.seed)
    if not passed:
        print("\nReference model test FAILED — aborting")
        sys.exit(1)

    if args.sim:
        if not run_deploy_simulator(trace_info, seed=args.seed):
            sys.exit(1)

    if args.hls:
        base.run_hls_test(trace_info, mode=args.hls, seed=args.seed)


if __name__ == "__main__":
    main()
