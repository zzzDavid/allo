# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""MINISA Hardware Synthesis and Deployment Test.

This test demonstrates building FEATHER+ MINISA for actual FPGA hardware
deployment using Vitis HLS with mode="hw".

Unlike csim or hw_emu modes, hw mode performs full hardware synthesis
and generates bitstreams that can be deployed on physical FPGA boards.

Usage:
    python test_minisa_hw.py [--build-only] [--AW 8] [--AH 8]

Options:
    --build-only    Only synthesize hardware, do not run on FPGA
    --AW            Array width (4, 8, or 16). Default: 8
    --AH            Array height. Default: 8
"""

import os
import sys
import argparse
import tempfile

import numpy as np

# Setup path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from allo.ir.types import int8
from allo.backend.hls import is_available
import allo.dataflow as df

from minisa.interpreter import MINISAInterpreter
from minisa.isa import create_gemm_program
from minisa.lowering import extract_output_for_verification
from feather_minisa import (
    get_feather_minisa_top,
    get_scheduled_feather_minisa,
    get_default_birrd_inst,
    build_feather_minisa_hls,
)


def test_hw_code_generation(AW: int, AH: int, project_dir: str):
    """Test that valid HLS code is generated for hw synthesis."""
    print("-" * 70)
    print("Test: HW Code Generation")
    print("-" * 70)

    top = get_feather_minisa_top(AW, AH, int8)
    s = df.customize(top)
    mod = s.build(target="vitis_hls", mode="csyn", project=project_dir)

    # Verify code was generated
    assert len(mod.hls_code) > 0, "HLS code should not be empty"
    assert "void" in mod.hls_code, "HLS code should contain function definitions"

    # Verify files exist
    kernel_cpp = os.path.join(project_dir, "kernel.cpp")
    kernel_h = os.path.join(project_dir, "kernel.h")
    assert os.path.exists(kernel_cpp), f"kernel.cpp should be generated at {kernel_cpp}"
    assert os.path.exists(kernel_h), f"kernel.h should be generated at {kernel_h}"

    print(f"  Generated HLS code length: {len(mod.hls_code)} bytes")
    print(f"  kernel.cpp: {kernel_cpp}")
    print(f"  kernel.h: {kernel_h}")
    print("  PASSED")
    return True


def test_hw_build(AW: int, AH: int, project_dir: str):
    """Test building FEATHER+ MINISA for hw target."""
    print("-" * 70)
    print("Test: HW Build (mode='hw')")
    print("-" * 70)

    print(f"  Building FEATHER+ MINISA for FPGA hardware...")
    print(f"  Project directory: {project_dir}")

    s = get_scheduled_feather_minisa(AW, AH, int8)
    hw_mod = s.build(
        target="vitis_hls",
        mode="hw",
        project=project_dir,
    )

    print(f"  Hardware build initiated successfully!")

    # Verify project files
    expected_files = ["kernel.cpp", "kernel.h", "host.cpp", "Makefile"]
    for filename in expected_files:
        filepath = os.path.join(project_dir, filename)
        if os.path.exists(filepath):
            print(f"    {filename}: EXISTS")
        else:
            print(f"    {filename}: MISSING (may be generated during synthesis)")

    print("  PASSED")
    return hw_mod


def test_hw_single_tile(AW: int, AH: int, project_dir: str):
    """Test single tile execution on FPGA hardware."""
    print("-" * 70)
    print("Test: HW Single Tile Execution")
    print("-" * 70)

    interpreter = MINISAInterpreter(
        AW=AW, AH=AH, Ty=int8,
        build_target="vitis_hls",
        build_mode="hw",
        project_dir=project_dir,
        verbose=True
    )

    np.random.seed(42)
    iActs = np.random.randint(-4, 4, size=(AH, AW)).astype(np.int8)
    weights = np.random.randint(-4, 4, size=(AH, AW, AH)).astype(np.int8)
    birrd_inst = get_default_birrd_inst(AW)

    print(f"  Input activations shape: {iActs.shape}")
    print(f"  Weights shape: {weights.shape}")
    print(f"  BIRRD instruction shape: {birrd_inst.shape}")

    output = interpreter.execute_single_tile(iActs, weights, birrd_inst)

    print(f"  Output shape: {output.shape}")
    assert output.shape == (AH, AW), f"Output shape mismatch: {output.shape}"
    assert interpreter.get_stats()['allo_invocations'] == 1

    print("  PASSED")
    return output


def test_hw_gemm(AW: int, AH: int, project_dir: str):
    """Test GEMM execution on FPGA hardware."""
    print("-" * 70)
    print("Test: HW GEMM Execution")
    print("-" * 70)

    M, N, K = 8, 8, 16

    interpreter = MINISAInterpreter(
        AW=AW, AH=AH, Ty=int8,
        build_target="vitis_hls",
        build_mode="hw",
        project_dir=project_dir
    )

    program = create_gemm_program(M, N, K, AH, AW)

    np.random.seed(42)
    A = np.random.randint(-4, 4, size=(M, K)).astype(np.int8)
    B = np.random.randint(-4, 4, size=(K, N)).astype(np.int8)

    print(f"  GEMM dimensions: M={M}, N={N}, K={K}")
    print(f"  Input A shape: {A.shape}")
    print(f"  Input B shape: {B.shape}")

    output = interpreter.execute_program(program, A, B)

    # Compute reference
    ref = np.dot(A, B)
    extracted = extract_output_for_verification(output, ref.shape, AW)

    np.testing.assert_allclose(extracted, ref, atol=1e-5)
    print(f"  Output verified against numpy reference")
    print("  PASSED")
    return True


def test_hw_matches_simulator(AW: int, AH: int, hw_project_dir: str):
    """Verify HW produces same results as LLVM simulator."""
    print("-" * 70)
    print("Test: HW vs Simulator Equivalence")
    print("-" * 70)

    np.random.seed(42)
    iActs = np.random.randint(-4, 4, size=(AH, AW)).astype(np.int8)
    weights = np.random.randint(-4, 4, size=(AH, AW, AH)).astype(np.int8)
    birrd_inst = get_default_birrd_inst(AW)

    # Run with LLVM simulator
    print("  Running LLVM simulator...")
    sim_interpreter = MINISAInterpreter(
        AW=AW, AH=AH, Ty=int8,
        build_target="simulator"
    )
    sim_output = sim_interpreter.execute_single_tile(iActs, weights, birrd_inst)

    # Run with HW
    print("  Running on FPGA hardware...")
    hw_interpreter = MINISAInterpreter(
        AW=AW, AH=AH, Ty=int8,
        build_target="vitis_hls",
        build_mode="hw",
        project_dir=hw_project_dir
    )
    hw_output = hw_interpreter.execute_single_tile(iActs.copy(), weights.copy(), birrd_inst.copy())

    # Results must match exactly
    np.testing.assert_array_equal(sim_output, hw_output)
    print("  Simulator and HW outputs match exactly!")
    print("  PASSED")
    return True


def run_build_only_tests(AW: int, AH: int, project_dir: str):
    """Run tests that only build/synthesize without FPGA execution."""
    print("=" * 70)
    print("MINISA HARDWARE BUILD TESTS (Build-Only Mode)")
    print("=" * 70)
    print(f"Configuration: AW={AW}, AH={AH}")
    print(f"Project directory: {project_dir}")
    print("=" * 70)

    results = {}

    # Test 1: Code generation
    try:
        test_hw_code_generation(AW, AH, project_dir)
        results["HW code generation"] = True
    except Exception as e:
        print(f"  FAILED: {e}")
        results["HW code generation"] = False

    # Test 2: HW build
    try:
        test_hw_build(AW, AH, project_dir)
        results["HW build"] = True
    except Exception as e:
        print(f"  FAILED: {e}")
        results["HW build"] = False

    return results


def run_full_tests(AW: int, AH: int, project_dir: str):
    """Run full tests including FPGA execution."""
    print("=" * 70)
    print("MINISA HARDWARE TESTS (Full Execution Mode)")
    print("=" * 70)
    print(f"Configuration: AW={AW}, AH={AH}")
    print(f"Project directory: {project_dir}")
    print("=" * 70)

    results = {}

    # Test 1: Code generation
    try:
        test_hw_code_generation(AW, AH, project_dir)
        results["HW code generation"] = True
    except Exception as e:
        print(f"  FAILED: {e}")
        results["HW code generation"] = False

    # Test 2: HW build
    try:
        test_hw_build(AW, AH, project_dir)
        results["HW build"] = True
    except Exception as e:
        print(f"  FAILED: {e}")
        results["HW build"] = False

    # Test 3: Single tile execution
    try:
        test_hw_single_tile(AW, AH, project_dir)
        results["HW single tile"] = True
    except Exception as e:
        print(f"  FAILED: {e}")
        results["HW single tile"] = False

    # Test 4: GEMM execution
    try:
        test_hw_gemm(AW, AH, project_dir)
        results["HW GEMM"] = True
    except Exception as e:
        print(f"  FAILED: {e}")
        results["HW GEMM"] = False

    # Test 5: Simulator equivalence
    try:
        test_hw_matches_simulator(AW, AH, project_dir)
        results["HW vs simulator"] = True
    except Exception as e:
        print(f"  FAILED: {e}")
        results["HW vs simulator"] = False

    return results


def print_summary(results: dict):
    """Print test summary."""
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    all_passed = all(results.values())
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    print("-" * 70)
    if all_passed:
        print("All hardware tests PASSED")
    else:
        failed = [name for name, passed in results.items() if not passed]
        print(f"Some tests FAILED: {', '.join(failed)}")
    print("=" * 70)

    return all_passed


def print_deployment_instructions(project_dir: str):
    """Print manual deployment instructions."""
    print("\n" + "=" * 70)
    print("MANUAL DEPLOYMENT INSTRUCTIONS")
    print("=" * 70)
    print(f"Build artifacts are generated in: {project_dir}")
    print("\nTo synthesize and deploy manually:")
    print(f"  cd {project_dir}")
    print("  make all TARGET=hw PLATFORM=<your_platform>")
    print("\nCommon platforms:")
    print("  - xilinx_u250_gen3x16_xdma_4_1_202210_1")
    print("  - xilinx_u280_gen3x16_xdma_1_202211_1")
    print("  - xilinx_vck190_base_202310_1")
    print("\nAfter synthesis completes, deploy the generated xclbin:")
    print("  ./host <kernel>.xclbin")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="MINISA Hardware Synthesis and Deployment Test"
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Only build the hardware, do not run on FPGA"
    )
    parser.add_argument(
        "--AW",
        type=int,
        default=8,
        choices=[4, 8, 16],
        help="Array width (must be power of 2). Default: 8"
    )
    parser.add_argument(
        "--AH",
        type=int,
        default=8,
        help="Array height. Default: 8"
    )
    parser.add_argument(
        "--project-dir",
        type=str,
        default=None,
        help="Project directory for HLS files. Default: auto-generated"
    )
    args = parser.parse_args()

    # Check Vitis HLS availability
    if not is_available("vitis_hls"):
        print("ERROR: Vitis HLS not available.")
        print("Please ensure Vitis HLS is installed and in PATH.")
        print("Check with: python -c \"from allo.backend.hls import is_available; print(is_available('vitis_hls'))\"")
        sys.exit(1)

    # Set up project directory
    if args.project_dir:
        project_dir = args.project_dir
        os.makedirs(project_dir, exist_ok=True)
    else:
        project_dir = os.path.join(
            os.path.dirname(__file__),
            f"feather_minisa_{args.AW}_{args.AH}_hw.prj"
        )
        os.makedirs(project_dir, exist_ok=True)

    # Run tests
    if args.build_only:
        results = run_build_only_tests(args.AW, args.AH, project_dir)
        success = print_summary(results)
        print_deployment_instructions(project_dir)
    else:
        results = run_full_tests(args.AW, args.AH, project_dir)
        success = print_summary(results)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
