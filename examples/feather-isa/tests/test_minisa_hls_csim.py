# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""MINISA HLS C Simulation tests.

These tests verify that the FEATHER+ MINISA implementation works correctly
with the Vitis HLS C simulation backend. Tests are skipped when Vitis HLS
is not available.

The HLS csim backend:
1. Generates HLS C code from the Allo dataflow design
2. Compiles the code via nanobind (IPModule)
3. Executes through the compiled module
4. Should produce identical results to the LLVM simulator
"""

import os
import sys
import tempfile

import pytest
import numpy as np

# Setup path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from allo.ir.types import int8
from allo.backend.hls import is_available

from minisa.interpreter import MINISAInterpreter
from minisa.isa import create_gemm_program
from minisa.lowering import extract_output_for_verification
from feather_minisa import (
    get_feather_minisa_top,
    get_default_birrd_inst,
    build_feather_minisa_hls,
    get_hls_code,
)
import allo.dataflow as df


# Skip all tests if Vitis HLS not available
pytestmark = pytest.mark.skipif(
    not is_available("vitis_hls"),
    reason="Vitis HLS not available"
)


class TestHLSCodeGeneration:
    """Tests for HLS code generation (no execution)."""

    def test_hls_code_generation_basic(self):
        """Test that valid HLS code is generated."""
        AW, AH = 8, 8

        with tempfile.TemporaryDirectory() as tmpdir:
            top = get_feather_minisa_top(AW, AH, int8)
            s = df.customize(top)
            mod = s.build(target="vitis_hls", mode="csyn", project=tmpdir)

            # Verify code was generated
            assert len(mod.hls_code) > 0, "HLS code should not be empty"

            # Verify key constructs are present
            assert "void" in mod.hls_code, "HLS code should contain function definitions"

            # Verify files exist
            assert os.path.exists(os.path.join(tmpdir, "kernel.cpp")), \
                "kernel.cpp should be generated"

    def test_hls_code_via_convenience_function(self):
        """Test get_hls_code convenience function."""
        AW, AH = 8, 8

        code = get_hls_code(AW, AH, int8)

        assert len(code) > 0, "HLS code should not be empty"
        assert "void" in code, "HLS code should contain function definitions"

    def test_hls_project_files_created(self):
        """Test that HLS project files are properly created."""
        AW, AH = 8, 8

        with tempfile.TemporaryDirectory() as tmpdir:
            mod = build_feather_minisa_hls(AW, AH, int8, mode="csyn", project=tmpdir)

            # Verify project structure
            assert os.path.exists(os.path.join(tmpdir, "kernel.cpp")), \
                "kernel.cpp should exist"
            assert os.path.exists(os.path.join(tmpdir, "kernel.h")), \
                "kernel.h should exist"


class TestHLSCSim:
    """Tests for HLS C simulation execution."""

    def test_hls_csim_single_tile(self):
        """Test single tile execution through HLS C simulation."""
        AW, AH = 8, 8

        with tempfile.TemporaryDirectory() as tmpdir:
            interpreter = MINISAInterpreter(
                AW=AW, AH=AH, Ty=int8,
                build_target="vitis_hls",
                build_mode="csim",
                project_dir=tmpdir,
                verbose=True
            )

            # Execute single tile (this triggers module build)
            np.random.seed(42)
            iActs = np.random.randint(-4, 4, size=(AH, AW)).astype(np.int8)
            weights = np.random.randint(-4, 4, size=(AH, AW, AH)).astype(np.int8)
            birrd_inst = get_default_birrd_inst(AW)

            output = interpreter.execute_single_tile(iActs, weights, birrd_inst)

            # Verify HLS project was created (after execution triggers build)
            assert os.path.exists(os.path.join(tmpdir, "kernel.cpp")), \
                "kernel.cpp should be generated"
            assert os.path.exists(os.path.join(tmpdir, "kernel.h")), \
                "kernel.h should be generated"

            assert output.shape == (AH, AW), f"Output shape mismatch: {output.shape}"
            assert interpreter.get_stats()['allo_invocations'] == 1, \
                "Should have exactly one Allo invocation"

    def test_hls_csim_gemm(self):
        """Test GEMM execution through HLS C simulation."""
        M, N, K = 8, 8, 16
        AW, AH = 8, 8

        with tempfile.TemporaryDirectory() as tmpdir:
            interpreter = MINISAInterpreter(
                AW=AW, AH=AH, Ty=int8,
                build_target="vitis_hls",
                build_mode="csim",
                project_dir=tmpdir
            )

            program = create_gemm_program(M, N, K, AH, AW)

            np.random.seed(42)
            A = np.random.randint(-4, 4, size=(M, K)).astype(np.int8)
            B = np.random.randint(-4, 4, size=(K, N)).astype(np.int8)

            output = interpreter.execute_program(program, A, B)

            # Compute reference
            ref = np.dot(A, B)
            extracted = extract_output_for_verification(output, ref.shape, AW)

            np.testing.assert_allclose(extracted, ref, atol=1e-5)

    def test_hls_csim_larger_gemm(self):
        """Test larger GEMM execution through HLS C simulation."""
        M, N, K = 16, 8, 32
        AW, AH = 8, 8

        with tempfile.TemporaryDirectory() as tmpdir:
            interpreter = MINISAInterpreter(
                AW=AW, AH=AH, Ty=int8,
                build_target="vitis_hls",
                build_mode="csim",
                project_dir=tmpdir
            )

            program = create_gemm_program(M, N, K, AH, AW)

            np.random.seed(42)
            A = np.random.randint(-4, 4, size=(M, K)).astype(np.int8)
            B = np.random.randint(-4, 4, size=(K, N)).astype(np.int8)

            output = interpreter.execute_program(program, A, B)

            # Compute reference
            ref = np.dot(A, B)
            extracted = extract_output_for_verification(output, ref.shape, AW)

            np.testing.assert_allclose(extracted, ref, atol=1e-5)


class TestSimulatorHLSEquivalence:
    """Tests verifying HLS csim produces same results as LLVM simulator."""

    def test_hls_csim_matches_simulator_single_tile(self):
        """Verify HLS csim produces same results as LLVM simulator for single tile."""
        AW, AH = 8, 8

        np.random.seed(42)
        iActs = np.random.randint(-4, 4, size=(AH, AW)).astype(np.int8)
        weights = np.random.randint(-4, 4, size=(AH, AW, AH)).astype(np.int8)
        birrd_inst = get_default_birrd_inst(AW)

        # Run with LLVM simulator
        sim_interpreter = MINISAInterpreter(
            AW=AW, AH=AH, Ty=int8,
            build_target="simulator"
        )
        sim_output = sim_interpreter.execute_single_tile(iActs, weights, birrd_inst)

        # Run with HLS csim
        with tempfile.TemporaryDirectory() as tmpdir:
            hls_interpreter = MINISAInterpreter(
                AW=AW, AH=AH, Ty=int8,
                build_target="vitis_hls",
                build_mode="csim",
                project_dir=tmpdir
            )
            hls_output = hls_interpreter.execute_single_tile(iActs, weights, birrd_inst)

        # Results must match exactly
        np.testing.assert_array_equal(sim_output, hls_output)

    def test_hls_csim_matches_simulator_gemm(self):
        """Verify HLS csim produces same results as LLVM simulator for GEMM."""
        M, N, K = 8, 8, 16
        AW, AH = 8, 8

        np.random.seed(42)
        A = np.random.randint(-4, 4, size=(M, K)).astype(np.int8)
        B = np.random.randint(-4, 4, size=(K, N)).astype(np.int8)

        program = create_gemm_program(M, N, K, AH, AW)

        # Run with LLVM simulator
        sim_interpreter = MINISAInterpreter(
            AW=AW, AH=AH, Ty=int8,
            build_target="simulator"
        )
        sim_output = sim_interpreter.execute_program(program, A, B)

        # Run with HLS csim
        with tempfile.TemporaryDirectory() as tmpdir:
            hls_interpreter = MINISAInterpreter(
                AW=AW, AH=AH, Ty=int8,
                build_target="vitis_hls",
                build_mode="csim",
                project_dir=tmpdir
            )
            hls_output = hls_interpreter.execute_program(program, A.copy(), B.copy())

        # Results must match exactly
        np.testing.assert_array_equal(sim_output, hls_output)


class TestInterpreterConfiguration:
    """Tests for interpreter build target configuration."""

    def test_default_target_is_simulator(self):
        """Verify default build target is simulator."""
        interpreter = MINISAInterpreter(AW=8, AH=8, Ty=int8)
        assert interpreter.build_target == "simulator"
        assert interpreter.build_mode == "csim"

    def test_invalid_target_raises_error(self):
        """Verify invalid build target raises error."""
        interpreter = MINISAInterpreter(
            AW=8, AH=8, Ty=int8,
            build_target="invalid_target"
        )

        np.random.seed(42)
        iActs = np.random.randint(-4, 4, size=(8, 8)).astype(np.int8)
        weights = np.random.randint(-4, 4, size=(8, 8, 8)).astype(np.int8)
        birrd_inst = get_default_birrd_inst(8)

        with pytest.raises(ValueError, match="Unsupported build target"):
            interpreter.execute_single_tile(iActs, weights, birrd_inst)

    def test_auto_project_dir_creation(self):
        """Test that project_dir is auto-created when not provided."""
        interpreter = MINISAInterpreter(
            AW=8, AH=8, Ty=int8,
            build_target="vitis_hls",
            build_mode="csim",
            project_dir=None  # Should auto-create
        )

        np.random.seed(42)
        iActs = np.random.randint(-4, 4, size=(8, 8)).astype(np.int8)
        weights = np.random.randint(-4, 4, size=(8, 8, 8)).astype(np.int8)
        birrd_inst = get_default_birrd_inst(8)

        # Execute to trigger module build
        interpreter.execute_single_tile(iActs, weights, birrd_inst)

        # Verify project_dir was set
        assert interpreter.project_dir is not None
        assert os.path.exists(interpreter.project_dir)


def run_hls_csim_tests():
    """Run all HLS csim tests manually."""
    print("=" * 70)
    print("MINISA HLS C SIMULATION TESTS")
    print("=" * 70)

    if not is_available("vitis_hls"):
        print("SKIPPED: Vitis HLS not available")
        return True

    results = {}

    tests = [
        ("HLS code generation", TestHLSCodeGeneration().test_hls_code_generation_basic),
        ("HLS convenience function", TestHLSCodeGeneration().test_hls_code_via_convenience_function),
        ("HLS project files", TestHLSCodeGeneration().test_hls_project_files_created),
        ("HLS csim single tile", TestHLSCSim().test_hls_csim_single_tile),
        ("HLS csim GEMM", TestHLSCSim().test_hls_csim_gemm),
        ("HLS csim larger GEMM", TestHLSCSim().test_hls_csim_larger_gemm),
        ("HLS matches simulator (tile)", TestSimulatorHLSEquivalence().test_hls_csim_matches_simulator_single_tile),
        ("HLS matches simulator (GEMM)", TestSimulatorHLSEquivalence().test_hls_csim_matches_simulator_gemm),
        ("Default target", TestInterpreterConfiguration().test_default_target_is_simulator),
        ("Invalid target error", TestInterpreterConfiguration().test_invalid_target_raises_error),
    ]

    for name, test_fn in tests:
        try:
            test_fn()
            results[name] = True
            print(f"  {name}: PASS")
        except Exception as e:
            print(f"  {name}: FAIL")
            print(f"    Error: {e}")
            results[name] = False

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    all_passed = all(results.values())
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    if all_passed:
        print("\nAll HLS csim tests PASSED")
    else:
        print("\nSome tests FAILED")

    return all_passed


if __name__ == "__main__":
    success = run_hls_csim_tests()
    sys.exit(0 if success else 1)
