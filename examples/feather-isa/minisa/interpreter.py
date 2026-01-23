# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MINISA Interpreter

Implements the MINISA execution model:
- Maintains active VN layout configurations
- Executes SetMapping instructions as tile computations
- Coordinates with FEATHER+ lowering layer

Reference: MINISA paper Section IV-E (Execution Model)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
import numpy as np

from .isa import (
    MINISAInstruction,
    SetIVNLayout,
    SetWVNLayout,
    SetOVNLayout,
    SetMapping,
    FeatherPlusConfig,
)
from .layout import (
    IVNLayout,
    WVNLayout,
    OVNLayout,
    LayoutOrder,
)


@dataclass
class InterpreterState:
    """
    MINISA interpreter state.

    Tracks:
    - Active layout configurations for IVN, WVN, OVN
    - Buffer contents (input, weight, output)
    - Execution statistics
    """
    # Active layouts
    ivn_layout: Optional[IVNLayout] = None
    wvn_layout: Optional[WVNLayout] = None
    ovn_layout: Optional[OVNLayout] = None

    # Buffer states (numpy arrays)
    input_buffer: Optional[np.ndarray] = None
    weight_buffer: Optional[np.ndarray] = None
    output_buffer: Optional[np.ndarray] = None

    # Execution counters
    num_layout_instructions: int = 0
    num_mapping_instructions: int = 0
    num_tiles_executed: int = 0


@dataclass
class MINISAInterpreter:
    """
    MINISA instruction interpreter for FEATHER+.

    Provides a software model for MINISA execution that can:
    1. Execute MINISA programs and produce correct results
    2. Generate equivalent FEATHER control signals for hardware
    3. Track instruction counts for efficiency analysis

    Usage:
        config = FeatherPlusConfig(AH=8, AW=8)
        interp = MINISAInterpreter(config)

        # Load data
        interp.load_inputs(input_matrix)
        interp.load_weights(weight_matrix)

        # Execute MINISA program
        for inst in minisa_program:
            interp.execute(inst)

        # Get results
        output = interp.get_output()
    """
    config: FeatherPlusConfig
    state: InterpreterState = field(default_factory=InterpreterState)

    # Callback for hardware execution (optional)
    hardware_executor: Optional[Callable] = None

    def __post_init__(self):
        """Initialize interpreter state."""
        self.reset()

    def reset(self):
        """Reset interpreter to initial state."""
        self.state = InterpreterState()

    def execute(self, instruction: MINISAInstruction) -> bool:
        """
        Execute a single MINISA instruction.

        Args:
            instruction: MINISA instruction to execute

        Returns:
            True if execution successful, False otherwise
        """
        if isinstance(instruction, SetIVNLayout):
            return self._execute_set_ivn_layout(instruction)
        elif isinstance(instruction, SetWVNLayout):
            return self._execute_set_wvn_layout(instruction)
        elif isinstance(instruction, SetOVNLayout):
            return self._execute_set_ovn_layout(instruction)
        elif isinstance(instruction, SetMapping):
            return self._execute_set_mapping(instruction)
        else:
            raise ValueError(f"Unknown instruction type: {type(instruction)}")

    def execute_program(self, program: List[MINISAInstruction]) -> bool:
        """
        Execute a complete MINISA program.

        Args:
            program: List of MINISA instructions

        Returns:
            True if all instructions executed successfully
        """
        for inst in program:
            if not self.execute(inst):
                return False
        return True

    def _execute_set_ivn_layout(self, inst: SetIVNLayout) -> bool:
        """Execute SetIVNLayout instruction."""
        # Validate
        if not inst.validate(self.config):
            return False

        # Create layout descriptor
        self.state.ivn_layout = IVNLayout(
            order=LayoutOrder(inst.order),
            AW=self.config.AW,
            M_L0=inst.M_L0,
            M_L1=inst.M_L1,
            J_L1=inst.J_L1,
        )

        self.state.num_layout_instructions += 1
        return True

    def _execute_set_wvn_layout(self, inst: SetWVNLayout) -> bool:
        """Execute SetWVNLayout instruction."""
        if not inst.validate(self.config):
            return False

        self.state.wvn_layout = WVNLayout(
            order=LayoutOrder(inst.order),
            AW=self.config.AW,
            N_L0=inst.N_L0,
            N_L1=inst.N_L1,
            K_L1=inst.K_L1,
        )

        self.state.num_layout_instructions += 1
        return True

    def _execute_set_ovn_layout(self, inst: SetOVNLayout) -> bool:
        """Execute SetOVNLayout instruction."""
        if not inst.validate(self.config):
            return False

        self.state.ovn_layout = OVNLayout(
            order=LayoutOrder(inst.order),
            AW=self.config.AW,
            P_L0=inst.P_L0,
            P_L1=inst.P_L1,
            Q_L1=inst.Q_L1,
        )

        # Clear output buffer if requested
        if inst.clear_mode and self.state.output_buffer is not None:
            self.state.output_buffer.fill(0)

        self.state.num_layout_instructions += 1
        return True

    def _execute_set_mapping(self, inst: SetMapping) -> bool:
        """
        Execute SetMapping instruction (tile execution).

        This is the main compute operation:
        1. Get PE-to-WVN mapping from instruction parameters
        2. For each PE, fetch assigned IVN and WVN
        3. Compute dot products
        4. Route results through BIRRD to output buffer
        """
        if not inst.validate(self.config):
            return False

        # Ensure layouts are configured
        if self.state.wvn_layout is None:
            raise RuntimeError("WVN layout not configured")
        if self.state.output_buffer is None:
            raise RuntimeError("Output buffer not initialized")

        # Get PE-to-WVN mapping
        pe_to_wvn = inst.get_pe_to_wvn_map(self.config.AH, self.config.AW)

        # Execute tile computation (software model)
        self._execute_tile(pe_to_wvn, inst)

        self.state.num_mapping_instructions += 1
        self.state.num_tiles_executed += 1
        return True

    def _execute_tile(self, pe_to_wvn: Dict[tuple, tuple], mapping: SetMapping):
        """
        Execute a single compute tile.

        Software model of FEATHER+ tile execution:
        1. Each PE computes AH-way dot product
        2. Results flow through BIRRD for reduction/reordering
        3. Output written to buffer according to OVN layout

        Key insight: PE(ah, aw) computes a partial sum for output C[m, n] where:
        - m is determined by ah and the M-tiling
        - n is determined by WVN column c(ah, aw)
        - The partial sum covers one K-tile (AH elements of K)
        """
        AH, AW = self.config.AH, self.config.AW

        # Track output contributions: (m_idx, n_idx) -> partial_sum
        # m_idx comes from PE row (ah), n_idx comes from WVN column
        output_contributions: Dict[tuple, int] = {}

        for aw in range(AW):
            for ah in range(AH):
                wvn_r, wvn_c = pe_to_wvn[(ah, aw)]

                # Get WVN data (with bounds checking)
                if self.state.weight_buffer is not None:
                    wvn_data = self._get_wvn(wvn_r, wvn_c)
                else:
                    wvn_data = np.zeros(AH, dtype=np.int8)

                # Get IVN data
                # ah indexes the M dimension (input row)
                # wvn_r indexes the K dimension (which K-tile)
                if self.state.input_buffer is not None:
                    ivn_data = self._get_ivn(ah, wvn_r)
                else:
                    ivn_data = np.zeros(AH, dtype=np.int8)

                # Compute dot product
                dot_product = np.dot(ivn_data.astype(np.int32),
                                    wvn_data.astype(np.int32))

                # Output position: (ah, wvn_c) in the output matrix
                # ah = row index (M dimension)
                # wvn_c = column index (N dimension)
                key = (ah, wvn_c)
                if key in output_contributions:
                    output_contributions[key] += dot_product
                else:
                    output_contributions[key] = dot_product

        # Write contributions to output buffer
        self._write_output_contributions(output_contributions, mapping)

    def _get_wvn(self, r: int, c: int) -> np.ndarray:
        """Get WVN data from weight buffer."""
        if self.state.weight_buffer is None:
            return np.zeros(self.config.AH, dtype=np.int8)

        # Use layout to find buffer address
        if self.state.wvn_layout is None:
            # Direct indexing if no layout
            if r < self.state.weight_buffer.shape[0] and c < self.state.weight_buffer.shape[1]:
                return self.state.weight_buffer[r, c, :]
            return np.zeros(self.config.AH, dtype=np.int8)

        buf_row, buf_col = self.state.wvn_layout.vn_to_buffer_addr(r, c)
        if buf_row < self.state.weight_buffer.shape[0]:
            return self.state.weight_buffer[buf_row, buf_col, :]
        return np.zeros(self.config.AH, dtype=np.int8)

    def _get_ivn(self, m: int, k: int) -> np.ndarray:
        """Get IVN data from input buffer."""
        if self.state.input_buffer is None:
            return np.zeros(self.config.AH, dtype=np.int8)

        # Direct indexing for functional model
        if m < self.state.input_buffer.shape[0]:
            k_start = k * self.config.AH
            k_end = k_start + self.config.AH
            if k_end <= self.state.input_buffer.shape[1]:
                return self.state.input_buffer[m, k_start:k_end]
        return np.zeros(self.config.AH, dtype=np.int8)

    def _write_output_contributions(self, contributions: Dict[tuple, int],
                                      mapping: SetMapping):
        """
        Write computed partial sums to output buffer.

        Args:
            contributions: Dict mapping (m_idx, n_idx) -> partial_sum
            mapping: SetMapping instruction (for potential offset information)
        """
        if self.state.output_buffer is None:
            return

        for (m_idx, n_idx), value in contributions.items():
            if (m_idx < self.state.output_buffer.shape[0] and
                n_idx < self.state.output_buffer.shape[1]):
                self.state.output_buffer[m_idx, n_idx] += value

    # =========================================================================
    # Data loading methods
    # =========================================================================

    def load_inputs(self, inputs: np.ndarray):
        """
        Load input matrix into streaming buffer.

        Args:
            inputs: Input matrix of shape (M, K)
        """
        self.state.input_buffer = inputs.copy()

    def load_weights(self, weights: np.ndarray):
        """
        Load weight matrix into stationary buffer.

        For GEMM with (M,K) × (K,N), weights have shape (K, N).
        VN organization: K_L0 = AH, so buffer shape is (K/AH, N, AH)

        Args:
            weights: Weight matrix
        """
        K, N = weights.shape
        AH = self.config.AH

        if K % AH != 0:
            # Pad K dimension
            pad_k = AH - (K % AH)
            weights = np.pad(weights, ((0, pad_k), (0, 0)), mode='constant')
            K = weights.shape[0]

        num_wvn_rows = K // AH

        # Reshape into VN format: (num_wvn_rows, N, AH)
        weight_buffer = np.zeros((num_wvn_rows, N, AH), dtype=weights.dtype)
        for r in range(num_wvn_rows):
            k_start = r * AH
            weight_buffer[r, :, :] = weights[k_start:k_start + AH, :].T

        self.state.weight_buffer = weight_buffer

    def allocate_output(self, M: int, N: int):
        """
        Allocate output buffer.

        Args:
            M, N: Output dimensions
        """
        self.state.output_buffer = np.zeros((M, N), dtype=np.int32)

    def get_output(self) -> np.ndarray:
        """Get output buffer contents."""
        return self.state.output_buffer.copy() if self.state.output_buffer is not None else None

    # =========================================================================
    # Statistics and debugging
    # =========================================================================

    def get_instruction_counts(self) -> Dict[str, int]:
        """Get instruction execution counts."""
        return {
            'layout_instructions': self.state.num_layout_instructions,
            'mapping_instructions': self.state.num_mapping_instructions,
            'tiles_executed': self.state.num_tiles_executed,
            'total_instructions': (self.state.num_layout_instructions +
                                  self.state.num_mapping_instructions),
        }

    def print_state(self):
        """Print interpreter state for debugging."""
        print("=" * 60)
        print("MINISA Interpreter State")
        print("=" * 60)
        print(f"Hardware Config: AH={self.config.AH}, AW={self.config.AW}")
        print(f"IVN Layout: {self.state.ivn_layout}")
        print(f"WVN Layout: {self.state.wvn_layout}")
        print(f"OVN Layout: {self.state.ovn_layout}")
        print(f"Input buffer shape: {self.state.input_buffer.shape if self.state.input_buffer is not None else None}")
        print(f"Weight buffer shape: {self.state.weight_buffer.shape if self.state.weight_buffer is not None else None}")
        print(f"Output buffer shape: {self.state.output_buffer.shape if self.state.output_buffer is not None else None}")
        print(f"Instructions executed: {self.get_instruction_counts()}")
        print("=" * 60)
