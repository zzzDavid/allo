# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test MINISA with pre-compiled bitstream.

This test demonstrates using a pre-compiled xclbin file to skip
the time-consuming hardware synthesis step.
"""

import os
import sys

# Setup path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from allo.ir.types import int8

from minisa.interpreter import MINISAInterpreter
from minisa.isa import create_gemm_program
from minisa.lowering import extract_output_for_verification
from feather_minisa import get_default_birrd_inst

# Path to pre-compiled bitstream
BITSTREAM_PATH = os.path.join(
    os.path.dirname(__file__),
    "feather_minisa_8_8_hw.prj/build_dir.hw.xilinx_u280_gen3x16_xdma_1_202211_1/top.xclbin"
)

# Path to existing project (for host code, makefile, etc.)
PROJECT_DIR = os.path.join(
    os.path.dirname(__file__),
    "feather_minisa_8_8_hw.prj"
)


def test_single_tile_with_bitstream():
    """Test single tile execution using pre-compiled bitstream."""
    print("=" * 70)
    print("Test: Single Tile with Pre-compiled Bitstream")
    print("=" * 70)

    if not os.path.exists(BITSTREAM_PATH):
        print(f"Bitstream not found: {BITSTREAM_PATH}")
        print("Skipping test - no pre-compiled bitstream available")
        return None

    print(f"Using bitstream: {BITSTREAM_PATH}")
    print(f"Using project: {PROJECT_DIR}")

    AW, AH = 8, 8

    # Create interpreter with pre-compiled bitstream
    interpreter = MINISAInterpreter(
        AW=AW, AH=AH, Ty=int8,
        build_target="vitis_hls",
        build_mode="hw",
        project_dir=PROJECT_DIR,
        bitstream=BITSTREAM_PATH,
        verbose=True
    )

    # Test data - same as used in original test
    np.random.seed(42)
    iActs = np.random.randint(-4, 4, size=(AH, AW)).astype(np.int8)
    weights = np.random.randint(-4, 4, size=(AH, AW, AH)).astype(np.int8)
    birrd_inst = get_default_birrd_inst(AW)

    print(f"Input activations shape: {iActs.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"BIRRD instruction shape: {birrd_inst.shape}")

    # Execute
    output = interpreter.execute_single_tile(iActs, weights, birrd_inst)

    print(f"Output shape: {output.shape}")
    print(f"Output:\n{output}")

    # Compare with simulator
    print("\n--- Comparing with LLVM simulator ---")
    sim_interpreter = MINISAInterpreter(
        AW=AW, AH=AH, Ty=int8,
        build_target="simulator",
        verbose=False
    )
    sim_output = sim_interpreter.execute_single_tile(
        iActs.copy(), weights.copy(), birrd_inst.copy()
    )

    print(f"Simulator output:\n{sim_output}")

    # Check match
    if np.array_equal(output, sim_output):
        print("\nRESULT: HW output matches simulator!")
        return True
    else:
        print("\nRESULT: HW output DIFFERS from simulator!")
        print(f"Difference:\n{output - sim_output}")
        return False


def test_gemm_with_bitstream():
    """Test GEMM execution using pre-compiled bitstream."""
    print("\n" + "=" * 70)
    print("Test: GEMM with Pre-compiled Bitstream")
    print("=" * 70)

    if not os.path.exists(BITSTREAM_PATH):
        print(f"Bitstream not found: {BITSTREAM_PATH}")
        print("Skipping test - no pre-compiled bitstream available")
        return None

    M, N, K = 8, 8, 16
    AW, AH = 8, 8

    # Create interpreter with pre-compiled bitstream
    interpreter = MINISAInterpreter(
        AW=AW, AH=AH, Ty=int8,
        build_target="vitis_hls",
        build_mode="hw",
        project_dir=PROJECT_DIR,
        bitstream=BITSTREAM_PATH,
        verbose=True
    )

    # Create GEMM program
    program = create_gemm_program(M, N, K, AH, AW)

    # Test data
    np.random.seed(42)
    A = np.random.randint(-4, 4, size=(M, K)).astype(np.int8)
    B = np.random.randint(-4, 4, size=(K, N)).astype(np.int8)

    print(f"Matrix A shape: {A.shape}")
    print(f"Matrix B shape: {B.shape}")

    # Execute
    output_raw = interpreter.execute_program(program, A, B)

    # Numpy reference
    ref = np.dot(A, B)

    # Extract for comparison
    output = extract_output_for_verification(output_raw, ref.shape, AW)

    print(f"Numpy reference:\n{ref}")
    print(f"HW output (extracted):\n{output}")

    # Check
    try:
        np.testing.assert_allclose(output, ref, atol=1e-5)
        print("\nRESULT: HW GEMM matches numpy reference!")
        return True
    except AssertionError as e:
        print(f"\nRESULT: HW GEMM DIFFERS from numpy!")
        print(f"Difference:\n{output - ref}")
        return False


def main():
    print("=" * 70)
    print("MINISA HW TEST WITH PRE-COMPILED BITSTREAM")
    print("=" * 70)

    # Check environment
    if "XDEVICE" not in os.environ:
        print("WARNING: XDEVICE not set. Setting to default platform.")
        os.environ["XDEVICE"] = "xilinx_u280_gen3x16_xdma_1_202211_1"

    print(f"XDEVICE: {os.environ.get('XDEVICE', 'not set')}")
    print(f"Bitstream path: {BITSTREAM_PATH}")
    print(f"Bitstream exists: {os.path.exists(BITSTREAM_PATH)}")
    print()

    results = {}

    # Test 1: Single tile
    result = test_single_tile_with_bitstream()
    if result is not None:
        results["Single tile"] = result

    # Test 2: GEMM
    result = test_gemm_with_bitstream()
    if result is not None:
        results["GEMM"] = result

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    if not results:
        print("No tests ran (bitstream not found)")
        return 1

    all_passed = all(results.values())
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    if all_passed:
        print("\nAll tests PASSED!")
        return 0
    else:
        print("\nSome tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
