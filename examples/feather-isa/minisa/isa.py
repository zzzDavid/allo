# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MINISA Instruction Set Architecture

Defines the four MINISA instructions for programming FEATHER+:
- SetIVNLayout: Configure streaming buffer layout for input VNs
- SetWVNLayout: Configure stationary buffer layout for weight VNs
- SetOVNLayout: Configure output buffer layout for output VNs and clear buffer
- SetMapping: Specify VN-level mapping and trigger tile execution
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
from enum import IntEnum


class InstructionOpCode(IntEnum):
    """MINISA instruction opcodes."""
    SET_IVN_LAYOUT = 0b01
    SET_WVN_LAYOUT = 0b10
    SET_OVN_LAYOUT = 0b10  # Same opcode, different context
    SET_MAPPING = 0b11


@dataclass
class MINISAInstruction(ABC):
    """Base class for all MINISA instructions."""

    @property
    @abstractmethod
    def opcode(self) -> int:
        """Return the instruction opcode."""
        pass

    @abstractmethod
    def encode(self) -> int:
        """Encode instruction to binary representation."""
        pass

    @abstractmethod
    def validate(self, config: 'FeatherPlusConfig') -> bool:
        """Validate instruction parameters against hardware config."""
        pass


@dataclass
class SetIVNLayout(MINISAInstruction):
    """
    Configure streaming buffer layout for input VNs.

    Triggers loading of input data from off-chip memory into on-chip buffer.

    Attributes:
        order: 3-bit layout order permutation (0-5)
        M_L0: Inner M dimension factor (non-reduction)
        M_L1: Outer M dimension factor
        J_L1: Outer J (reduction) dimension factor

    Constraints:
        - J_L0 = AH (VN constraint)
        - M_L1 * J_L1 <= D_str / M_L0 (buffer capacity)
    """
    order: int  # 3-bit (0-5)
    M_L0: int   # Inner non-reduction factor
    M_L1: int   # Outer non-reduction factor
    J_L1: int   # Outer reduction factor (J_L0 = AH implicitly)

    @property
    def opcode(self) -> int:
        return InstructionOpCode.SET_IVN_LAYOUT

    def encode(self) -> int:
        """Encode to binary: [opcode:2][order:3][M_L0:?][M_L1:?][J_L1:?]"""
        # Bit widths depend on hardware config, simplified here
        encoded = (self.opcode << 30)
        encoded |= (self.order & 0x7) << 27
        # Additional fields depend on specific bitwidths
        return encoded

    def validate(self, config: 'FeatherPlusConfig') -> bool:
        """Validate against hardware configuration."""
        if not (0 <= self.order <= 5):
            return False
        if not (1 <= self.M_L0 <= config.AW):
            return False
        if self.M_L1 * self.J_L1 > config.D_str // self.M_L0:
            return False
        return True


@dataclass
class SetWVNLayout(MINISAInstruction):
    """
    Configure stationary buffer layout for weight VNs.

    Triggers loading of weights from off-chip memory into on-chip buffer.

    Attributes:
        order: 3-bit layout order permutation (0-5)
        N_L0: Inner N dimension factor
        N_L1: Outer N dimension factor
        K_L1: Outer K (reduction) dimension factor

    Constraints:
        - K_L0 = AH (VN constraint)
        - N_L1 * K_L1 <= D_sta / N_L0 (buffer capacity)
    """
    order: int  # 3-bit (0-5)
    N_L0: int   # Inner non-reduction factor
    N_L1: int   # Outer non-reduction factor
    K_L1: int   # Outer reduction factor (K_L0 = AH implicitly)

    @property
    def opcode(self) -> int:
        return InstructionOpCode.SET_WVN_LAYOUT

    def encode(self) -> int:
        """Encode to binary."""
        encoded = (self.opcode << 30)
        encoded |= (self.order & 0x7) << 27
        return encoded

    def validate(self, config: 'FeatherPlusConfig') -> bool:
        """Validate against hardware configuration."""
        if not (0 <= self.order <= 5):
            return False
        if not (1 <= self.N_L0 <= config.AW):
            return False
        if self.N_L1 * self.K_L1 > config.D_sta // self.N_L0:
            return False
        return True


@dataclass
class SetOVNLayout(MINISAInstruction):
    """
    Configure output buffer layout for output VNs.

    Also clears (initializes) the output buffer for accumulation.

    Attributes:
        order: 3-bit layout order permutation (0-5)
        P_L0: Inner P dimension factor
        P_L1: Outer P dimension factor
        Q_L1: Outer Q dimension factor

    Constraints:
        - Q_L0 = AH (VN constraint)
        - P_L1 * Q_L1 <= D_str / P_L0 (buffer capacity)
    """
    order: int  # 3-bit (0-5)
    P_L0: int   # Inner non-reduction factor
    P_L1: int   # Outer non-reduction factor
    Q_L1: int   # Outer output dimension factor (Q_L0 = AH implicitly)
    clear_mode: bool = True  # Whether to clear buffer

    @property
    def opcode(self) -> int:
        return InstructionOpCode.SET_OVN_LAYOUT

    def encode(self) -> int:
        """Encode to binary."""
        encoded = (self.opcode << 30)
        encoded |= (self.order & 0x7) << 27
        encoded |= (1 if self.clear_mode else 0) << 26
        return encoded

    def validate(self, config: 'FeatherPlusConfig') -> bool:
        """Validate against hardware configuration."""
        if not (0 <= self.order <= 5):
            return False
        if not (1 <= self.P_L0 <= config.AW):
            return False
        if self.P_L1 * self.Q_L1 > config.D_str // self.P_L0:
            return False
        return True


@dataclass
class SetMapping(MINISAInstruction):
    """
    Specify VN-level mapping parameters and trigger tile execution.

    Maps Weight VNs to PEs according to parametric formula:
        r(ah, aw) = r0 + floor(aw / G_r)  [WVN row index]
        c(ah, aw) = c0 + s_r * ah + s_c * (aw mod G_c)  [WVN col index]

    Attributes:
        r0: Base WVN row index
        c0: Base WVN column index
        G_r: Number of PE columns sharing same WVN row (reduction group)
        G_c: Replication group size
        s_r: Temporal stride across PE rows
        s_c: Spatial stride within replication group

    Constraints:
        - 0 <= r0 < K/AH (number of WVN rows)
        - 0 <= c0 < N (number of WVN columns)
        - 1 <= G_r <= AW
        - 1 <= G_c <= AW
        - s_r, s_c define non-overlapping VN access
    """
    r0: int    # Base WVN row index
    c0: int    # Base WVN column index
    G_r: int   # Columns sharing same WVN row
    G_c: int   # Replication group size
    s_r: int   # Temporal stride
    s_c: int   # Spatial stride

    @property
    def opcode(self) -> int:
        return InstructionOpCode.SET_MAPPING

    def encode(self) -> int:
        """Encode to binary."""
        encoded = (self.opcode << 30)
        # Simplified encoding - actual bitwidths depend on config
        return encoded

    def validate(self, config: 'FeatherPlusConfig') -> bool:
        """Validate against hardware configuration."""
        if not (1 <= self.G_r <= config.AW):
            return False
        if not (1 <= self.G_c <= config.AW):
            return False
        # r0 and c0 bounds depend on workload, not just hardware
        return True

    def get_pe_to_wvn_map(self, AH: int, AW: int) -> Dict[tuple, tuple]:
        """
        Compute the PE-to-WVN mapping for this instruction.

        Returns:
            Dict mapping (ah, aw) -> (wvn_row, wvn_col)
        """
        mapping = {}
        for ah in range(AH):
            for aw in range(AW):
                r = self.r0 + (aw // self.G_r)
                c = self.c0 + self.s_r * ah + self.s_c * (aw % self.G_c)
                mapping[(ah, aw)] = (r, c)
        return mapping


@dataclass
class FeatherPlusConfig:
    """
    FEATHER+ hardware configuration.

    Attributes:
        AH: Array height (PE rows, also VN size)
        AW: Array width (PE columns, must be power of 2)
        D_sta: Stationary buffer depth (in VN rows)
        D_str: Streaming buffer depth (in VN rows)
    """
    AH: int
    AW: int
    D_sta: int = 256  # Default stationary buffer depth
    D_str: int = 256  # Default streaming buffer depth

    def __post_init__(self):
        # Validate power of 2
        assert self.AW > 0 and (self.AW & (self.AW - 1)) == 0, \
            f"AW must be power of 2, got {self.AW}"
        assert self.AH > 0, f"AH must be positive, got {self.AH}"

    @property
    def vn_size(self) -> int:
        """Virtual Neuron size = AH."""
        return self.AH

    @property
    def num_birrd_stages(self) -> int:
        """Number of BIRRD stages."""
        from math import log2
        LOG2_AW = int(log2(self.AW))
        return 2 * LOG2_AW if self.AW > 4 else 2 * LOG2_AW - 1

    @property
    def switches_per_stage(self) -> int:
        """Number of EGG switches per BIRRD stage."""
        return self.AW // 2


# Convenience factory functions
def create_gemm_mapping(M: int, K: int, N: int, AH: int, AW: int) -> list:
    """
    Create MINISA instruction sequence for a GEMM workload.

    Args:
        M, K, N: GEMM dimensions (M×K) × (K×N) = (M×N)
        AH, AW: FEATHER+ array dimensions

    Returns:
        List of MINISA instructions
    """
    instructions = []

    # Calculate VN dimensions
    num_ivn_rows = M
    num_ivn_cols = K // AH  # VN constraint
    num_wvn_rows = K // AH  # VN constraint
    num_wvn_cols = N
    num_ovn_rows = M
    num_ovn_cols = N // AH  # VN constraint for output

    # SetIVNLayout - configure input buffer
    # Using K-major order (elements along K are contiguous)
    ivn_layout = SetIVNLayout(
        order=0b000,  # m_L1 -> m_L0 -> j_L1
        M_L0=min(AW, M),
        M_L1=(M + AW - 1) // AW,
        J_L1=num_ivn_cols,
    )
    instructions.append(ivn_layout)

    # SetWVNLayout - configure weight buffer
    wvn_layout = SetWVNLayout(
        order=0b010,  # n_L0 -> k_L1 -> n_L1 (matches paper example)
        N_L0=min(AW, N),
        N_L1=(N + AW - 1) // AW,
        K_L1=num_wvn_rows,
    )
    instructions.append(wvn_layout)

    # SetOVNLayout - configure output buffer
    ovn_layout = SetOVNLayout(
        order=0b010,  # p_L0 -> p_L1 -> q_L1
        P_L0=min(AW, M),
        P_L1=(M + AW - 1) // AW,
        Q_L1=num_ovn_cols,
        clear_mode=True,
    )
    instructions.append(ovn_layout)

    # SetMapping - trigger execution for each tile
    # Number of tiles depends on how workload maps to array
    num_r_tiles = (num_wvn_rows + AW - 1) // AW if num_wvn_rows > 0 else 1
    num_c_tiles = (num_wvn_cols + AW - 1) // AW if num_wvn_cols > 0 else 1

    for r_tile in range(num_r_tiles):
        for c_tile in range(num_c_tiles):
            mapping = SetMapping(
                r0=r_tile * (AW // 2),  # Adjust based on reduction group
                c0=c_tile * AW,
                G_r=2,  # 2 columns share same WVN row (2:1 reduction)
                G_c=AW,
                s_r=1,
                s_c=1,
            )
            instructions.append(mapping)

    return instructions
