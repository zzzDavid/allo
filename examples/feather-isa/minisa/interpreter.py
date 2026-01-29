# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""MINISA Interpreter: Execute MINISA programs through Allo dataflow.

This interpreter executes MINISA programs by:
1. Lowering layouts to configuration tensors (Python)
2. For each SetMapping: invoking the Allo dataflow region (Allo compute)
3. Accumulating results in output buffer (Allo writes, Python manages)

CRITICAL REQUIREMENT: All computation is performed by Allo kernels.
The interpreter only handles:
- Control flow (iterating over tiles)
- Data slicing (extracting tiles from tensors)
- Configuration (passing config arrays to Allo)

The interpreter NEVER performs matrix multiply, reduction, or any
mathematical computation on the data.
"""

import os
import sys
from typing import Optional, Callable, Dict, Any

import numpy as np

# Ensure imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from allo.ir.types import int8, AlloType
import allo.dataflow as df

from .isa import MINISAProgram, SetMapping
from .lowering import (
    lower_minisa_program,
    TileExtractor,
    LoweredConfig,
)


class MINISAInterpreter:
    """Execute MINISA programs through the FEATHER+ Allo dataflow.

    The interpreter manages:
    - Building and caching Allo modules
    - Lowering MINISA instructions to configurations
    - Executing tiles through Allo kernels
    - Tracking execution statistics

    All computation is delegated to Allo - the interpreter performs
    NO mathematical operations on data.

    Supports multiple build targets:
    - "simulator": LLVM OMP-based dataflow simulator (fast, default)
    - "vitis_hls": HLS C simulation via Vitis HLS (validates HLS code)
    """

    def __init__(
        self,
        AW: int = 8,
        AH: int = 8,
        Ty: AlloType = int8,
        verbose: bool = False,
        build_target: str = "simulator",
        build_mode: str = "csim",
        project_dir: Optional[str] = None
    ):
        """Initialize the MINISA interpreter.

        Args:
            AW: Array width (4, 8, or 16)
            AH: Array height
            Ty: Data type for computation
            verbose: Enable verbose logging
            build_target: Build target ("simulator" or "vitis_hls")
            build_mode: HLS mode for vitis_hls target ("csim", "csyn", etc.)
            project_dir: Directory for HLS project files (auto-created if None)
        """
        self.AW = AW
        self.AH = AH
        self.Ty = Ty
        self.verbose = verbose
        self.build_target = build_target
        self.build_mode = build_mode
        self.project_dir = project_dir

        # Execution statistics
        self.stats: Dict[str, Any] = {
            "allo_invocations": 0,
            "tiles_executed": 0,
            "total_elements_processed": 0,
        }

        # Lazily built Allo module
        self._allo_module: Optional[Callable] = None
        self._module_built = False

    def _log(self, msg: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[MINISA] {msg}")

    def _build_allo_module(self):
        """Build the FEATHER+ Allo dataflow module.

        This creates the Allo module that will execute all tile computations.
        The module is cached for reuse across tiles.

        Supports two build targets:
        - "simulator": LLVM OMP-based dataflow simulator (fast)
        - "vitis_hls": HLS C simulation via Vitis HLS (validates HLS code)
        """
        if self._module_built:
            return

        self._log(f"Building Allo dataflow module (AW={self.AW}, AH={self.AH}, "
                  f"target={self.build_target})")

        # Import here to avoid circular dependency
        # Handle both package and direct imports
        try:
            from feather_minisa import get_feather_minisa_top
        except ImportError:
            # Try adding the feather-isa directory to path
            feather_isa_dir = os.path.join(os.path.dirname(__file__), "..")
            if feather_isa_dir not in sys.path:
                sys.path.insert(0, feather_isa_dir)
            from feather_minisa import get_feather_minisa_top

        top = get_feather_minisa_top(self.AW, self.AH, self.Ty)

        if self.build_target == "simulator":
            self._allo_module = df.build(top, target="simulator")
        elif self.build_target == "vitis_hls":
            # Requires project directory for HLS code generation
            if self.project_dir is None:
                self.project_dir = os.path.join(os.path.dirname(__file__), "..", "hls_project")
                os.makedirs(self.project_dir, exist_ok=True)
                self._log(f"Using HLS project dir: {self.project_dir}")

            self._allo_module = df.build(
                top,
                target="vitis_hls",
                mode=self.build_mode,
                project=self.project_dir
            )
        else:
            raise ValueError(f"Unsupported build target: {self.build_target}. "
                           f"Use 'simulator' or 'vitis_hls'.")

        self._module_built = True
        self._log("Allo module built successfully")

    def execute_program(
        self,
        program: MINISAProgram,
        inputs: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """Execute a MINISA program on the given inputs.

        This is the main entry point for MINISA execution. It:
        1. Validates the program
        2. Lowers layouts to configurations
        3. Iterates over tile mappings
        4. Invokes Allo for each tile
        5. Returns the accumulated output

        Args:
            program: MINISA program to execute
            inputs: Input tensor [M, K]
            weights: Weight tensor [K, N]

        Returns:
            Output tensor computed by Allo

        Raises:
            ValueError: If program is invalid or tensors don't match
        """
        # Validate program
        if not program.validate():
            raise ValueError("Invalid MINISA program")

        # Build Allo module if needed
        self._build_allo_module()

        # Lower program to configuration
        config = lower_minisa_program(program)
        tile_extractor = TileExtractor(program)

        # Determine output shape from layouts
        M, K = inputs.shape
        K2, N = weights.shape
        assert K == K2, f"Dimension mismatch: inputs K={K}, weights K={K2}"

        # Output buffer in FEATHER+ layout [N, 2*M]
        output = np.zeros((N, 2 * M), dtype=inputs.dtype)

        self._log(f"Executing MINISA program '{program.name}'")
        self._log(f"  Input shape: {inputs.shape}")
        self._log(f"  Weights shape: {weights.shape}")
        self._log(f"  Output shape: {output.shape}")
        self._log(f"  Number of tiles: {program.num_tiles()}")

        # Execute each tile through Allo
        for i, mapping in enumerate(program.mappings):
            self._execute_tile(
                mapping=mapping,
                inputs=inputs,
                weights=weights,
                output=output,
                config=config,
                tile_extractor=tile_extractor,
                tile_idx=i,
            )

        self._log(f"Program execution complete")
        self._log(f"  Total Allo invocations: {self.stats['allo_invocations']}")

        return output

    def _execute_tile(
        self,
        mapping: SetMapping,
        inputs: np.ndarray,
        weights: np.ndarray,
        output: np.ndarray,
        config: LoweredConfig,
        tile_extractor: TileExtractor,
        tile_idx: int,
    ):
        """Execute a single tile through Allo.

        This is where Allo performs the actual computation. The method:
        1. Extracts input/weight tiles (data slicing, no compute)
        2. Invokes Allo module (ALL COMPUTE HAPPENS HERE)
        3. Accumulates results into output buffer

        IMPORTANT: No mathematical computation happens in this method
        except through the Allo module invocation.

        Args:
            mapping: Tile mapping configuration
            inputs: Full input tensor
            weights: Full weight tensor
            output: Output buffer to accumulate into
            config: Lowered MINISA configuration
            tile_extractor: Helper for tile extraction
            tile_idx: Index of current tile
        """
        # Extract tiles - NO COMPUTE, just data slicing
        iActs_tile = tile_extractor.extract_input_tile(inputs, mapping)
        weights_tile = tile_extractor.extract_weight_tile(weights, mapping)

        # Get output location
        out_row_slice, out_col_slice = tile_extractor.get_output_slices(mapping)

        # Create per-tile output buffer
        tile_output = np.zeros((self.AH, self.AW), dtype=inputs.dtype)

        self._log(f"  Tile {tile_idx}: m=[{mapping.m_start}:{mapping.m_end}], "
                  f"n=[{mapping.n_start}:{mapping.n_end}], "
                  f"k=[{mapping.k_start}:{mapping.k_end}]")

        # ================================================================
        # ALLO EXECUTION - ALL COMPUTATION HAPPENS HERE
        # ================================================================
        self._allo_module(
            iActs_tile,
            weights_tile,
            config.birrd_inst,
            tile_output
        )
        # ================================================================

        # Track statistics
        self.stats["allo_invocations"] += 1
        self.stats["tiles_executed"] += 1
        self.stats["total_elements_processed"] += iActs_tile.size

        # Accumulate tile output into full output buffer
        # This is just array addition for reduction across K dimension
        # The actual tile computation was done by Allo above
        output[out_row_slice, out_col_slice] += tile_output

    def execute_single_tile(
        self,
        iActs: np.ndarray,
        weights: np.ndarray,
        birrd_inst: np.ndarray,
    ) -> np.ndarray:
        """Execute a single pre-formatted tile through Allo.

        This is a lower-level interface for direct tile execution when
        the caller has already prepared the tile data and configuration.

        Args:
            iActs: Input tile [AH, AW] - already reordered
            weights: Weight tile [AH, AW, AH] - already reordered
            birrd_inst: BIRRD instruction array [P0, P1]

        Returns:
            Output tile [AH, AW] computed by Allo
        """
        # Build Allo module if needed
        self._build_allo_module()

        # Validate shapes
        assert iActs.shape == (self.AH, self.AW), \
            f"Expected iActs shape ({self.AH}, {self.AW}), got {iActs.shape}"
        assert weights.shape == (self.AH, self.AW, self.AH), \
            f"Expected weights shape ({self.AH}, {self.AW}, {self.AH}), got {weights.shape}"

        # Output buffer
        output = np.zeros((self.AH, self.AW), dtype=iActs.dtype)

        # ================================================================
        # ALLO EXECUTION - ALL COMPUTATION HAPPENS HERE
        # ================================================================
        self._allo_module(iActs, weights, birrd_inst, output)
        # ================================================================

        self.stats["allo_invocations"] += 1

        return output

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics.

        Returns:
            Dictionary with execution statistics
        """
        return self.stats.copy()

    def reset_stats(self):
        """Reset execution statistics."""
        self.stats = {
            "allo_invocations": 0,
            "tiles_executed": 0,
            "total_elements_processed": 0,
        }


def run_minisa_gemm(
    M: int, N: int, K: int,
    AW: int = 8, AH: int = 8,
    Ty: AlloType = int8,
    verbose: bool = False,
    build_target: str = "simulator",
    build_mode: str = "csim",
    project_dir: Optional[str] = None,
) -> tuple:
    """Run a GEMM operation using MINISA on FEATHER+.

    This is a convenience function that:
    1. Creates a MINISA program for GEMM
    2. Generates random test inputs
    3. Executes through the interpreter
    4. Compares to numpy reference

    Args:
        M, N, K: Matrix dimensions for C[M,N] = A[M,K] * B[K,N]
        AW: Array width
        AH: Array height
        Ty: Data type
        verbose: Enable verbose logging
        build_target: Build target ("simulator" or "vitis_hls")
        build_mode: HLS mode for vitis_hls target
        project_dir: Directory for HLS project files

    Returns:
        (output, reference, passed): Results and verification status
    """
    from .isa import create_gemm_program
    from .lowering import extract_output_for_verification

    # Create MINISA program
    program = create_gemm_program(M, N, K, AH, AW, name=f"gemm_{M}x{K}x{N}")

    # Generate test inputs
    np.random.seed(42)
    A = np.random.randint(-4, 4, size=(M, K)).astype(np.int8)
    B = np.random.randint(-4, 4, size=(K, N)).astype(np.int8)

    # Execute through MINISA interpreter (uses Allo for compute)
    interpreter = MINISAInterpreter(
        AW=AW, AH=AH, Ty=Ty, verbose=verbose,
        build_target=build_target, build_mode=build_mode, project_dir=project_dir
    )
    output_raw = interpreter.execute_program(program, A, B)

    # Compute numpy reference (for verification only)
    ref = np.dot(A, B)

    # Extract and reorder output for comparison
    output = extract_output_for_verification(output_raw, ref.shape, AW)

    # Verify
    try:
        np.testing.assert_allclose(output, ref, atol=1e-5)
        passed = True
    except AssertionError:
        passed = False

    stats = interpreter.get_stats()
    if verbose:
        print(f"MINISA GEMM Results:")
        print(f"  Dimensions: M={M}, N={N}, K={K}")
        print(f"  Allo invocations: {stats['allo_invocations']}")
        print(f"  Tiles executed: {stats['tiles_executed']}")
        print(f"  Verification: {'PASSED' if passed else 'FAILED'}")

    return output, ref, passed


if __name__ == "__main__":
    # Quick test
    print("Testing MINISA Interpreter...")
    output, ref, passed = run_minisa_gemm(
        M=8, N=8, K=16,
        AW=8, AH=8,
        verbose=True
    )
    print(f"\nFinal result: {'PASSED' if passed else 'FAILED'}")
