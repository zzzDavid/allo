# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MINISA Lowering Layer

Translates MINISA instructions into FEATHER control signals:
- Layout instructions -> buffer addressing functions
- SetMapping -> PE-to-VN mapping + BIRRD instruction array

Reference: MINISA paper Section IV-D (FEATHER+ Configuration Generation)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable
from math import log2
import numpy as np

from .isa import (
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


# BIRRD EGG instructions
PS = 0  # Pass
AR = 1  # Add Right
AL = 2  # Add Left
SW = 3  # Swap


def reverse_bits(data: int, bit_range: int) -> int:
    """
    Reverse bits in data within the specified bit range.
    Used for BIRRD inter-stage routing.
    """
    mask = (1 << bit_range) - 1
    reversed_bits = 0
    for i in range(bit_range):
        if data & (1 << i):
            reversed_bits |= 1 << (bit_range - 1 - i)
    return (data & ~mask) | reversed_bits


@dataclass
class BIRRDConfig:
    """
    BIRRD configuration for a tile execution.

    Attributes:
        num_stages: Number of BIRRD stages (P0)
        switches_per_stage: Switches per stage (P1)
        instructions: Instruction array of shape (P0, P1)
        output_addresses: Output buffer write addresses
    """
    num_stages: int
    switches_per_stage: int
    instructions: np.ndarray
    output_addresses: Optional[np.ndarray] = None


@dataclass
class TileConfig:
    """
    Complete configuration for a compute tile.

    Generated from SetMapping instruction and active layouts.
    """
    # PE-to-VN mapping
    pe_to_wvn: Dict[Tuple[int, int], Tuple[int, int]]
    pe_to_ivn: Dict[Tuple[int, int], Tuple[int, int]]

    # Distribution crossbar config
    input_crossbar: Dict[int, List[int]]  # PE column -> IVN indices
    weight_crossbar: Dict[int, List[int]]  # PE column -> WVN indices

    # BIRRD config
    birrd: BIRRDConfig

    # Reduction groups
    reduction_groups: List[List[int]]  # Groups of PE columns that reduce together


class MINISALowering:
    """
    MINISA to FEATHER lowering layer.

    Translates MINISA instructions into hardware control signals.
    """

    def __init__(self, config: FeatherPlusConfig):
        self.config = config
        self.AH = config.AH
        self.AW = config.AW
        self.LOG2_AW = int(log2(self.AW))
        self.P0 = 2 * self.LOG2_AW if self.AW > 4 else 2 * self.LOG2_AW - 1
        self.P1 = self.AW // 2

    def lower_mapping(self, mapping: SetMapping,
                      wvn_layout: Optional[WVNLayout] = None,
                      ivn_layout: Optional[IVNLayout] = None,
                      ovn_layout: Optional[OVNLayout] = None) -> TileConfig:
        """
        Lower SetMapping instruction to tile configuration.

        Args:
            mapping: SetMapping instruction
            wvn_layout: Active weight VN layout
            ivn_layout: Active input VN layout
            ovn_layout: Active output VN layout

        Returns:
            TileConfig with all hardware control signals
        """
        # Generate PE-to-VN mappings
        pe_to_wvn = mapping.get_pe_to_wvn_map(self.AH, self.AW)

        # Generate IVN mapping (derived from WVN mapping)
        pe_to_ivn = self._derive_ivn_mapping(pe_to_wvn, mapping)

        # Generate distribution crossbar configs
        input_xbar = self._generate_input_crossbar(pe_to_ivn)
        weight_xbar = self._generate_weight_crossbar(pe_to_wvn)

        # Identify reduction groups
        reduction_groups = self._identify_reduction_groups(pe_to_wvn)

        # Generate BIRRD instructions
        birrd = self._generate_birrd_config(reduction_groups, pe_to_wvn, ovn_layout)

        return TileConfig(
            pe_to_wvn=pe_to_wvn,
            pe_to_ivn=pe_to_ivn,
            input_crossbar=input_xbar,
            weight_crossbar=weight_xbar,
            birrd=birrd,
            reduction_groups=reduction_groups,
        )

    def _derive_ivn_mapping(self, pe_to_wvn: Dict, mapping: SetMapping) -> Dict:
        """
        Derive IVN mapping from WVN mapping.

        For GEMM: IVN row maps to M dimension, IVN col maps to K dimension
        WVN row maps to K dimension, WVN col maps to N dimension

        Since WVN row = k_L1, IVN column should match WVN row for correct
        reduction alignment.
        """
        pe_to_ivn = {}
        for (ah, aw), (wvn_r, wvn_c) in pe_to_wvn.items():
            # IVN row indexed by PE row (ah) -> M dimension
            # IVN col indexed by WVN row (wvn_r) -> K dimension
            ivn_r = ah  # M index
            ivn_c = wvn_r  # K/AH index
            pe_to_ivn[(ah, aw)] = (ivn_r, ivn_c)
        return pe_to_ivn

    def _generate_input_crossbar(self, pe_to_ivn: Dict) -> Dict[int, List[int]]:
        """
        Generate input distribution crossbar configuration.

        Maps which IVNs route to which PE columns.
        """
        crossbar = {}
        for aw in range(self.AW):
            ivn_indices = set()
            for ah in range(self.AH):
                ivn_r, ivn_c = pe_to_ivn[(ah, aw)]
                ivn_indices.add((ivn_r, ivn_c))
            crossbar[aw] = list(ivn_indices)
        return crossbar

    def _generate_weight_crossbar(self, pe_to_wvn: Dict) -> Dict[int, List[int]]:
        """
        Generate weight distribution crossbar configuration.

        Maps which WVNs route to which PE columns.
        """
        crossbar = {}
        for aw in range(self.AW):
            wvn_indices = set()
            for ah in range(self.AH):
                wvn_r, wvn_c = pe_to_wvn[(ah, aw)]
                wvn_indices.add((wvn_r, wvn_c))
            crossbar[aw] = list(wvn_indices)
        return crossbar

    def _identify_reduction_groups(self, pe_to_wvn: Dict) -> List[List[int]]:
        """
        Identify which PE columns reduce together.

        Columns with the same WVN row index form a reduction group.
        """
        row_to_columns: Dict[int, List[int]] = {}
        for aw in range(self.AW):
            # All PEs in a column have same WVN row (FEATHER constraint)
            wvn_r = pe_to_wvn[(0, aw)][0]
            if wvn_r not in row_to_columns:
                row_to_columns[wvn_r] = []
            row_to_columns[wvn_r].append(aw)

        return list(row_to_columns.values())

    def _generate_birrd_config(self, reduction_groups: List[List[int]],
                                pe_to_wvn: Dict,
                                ovn_layout: Optional[OVNLayout]) -> BIRRDConfig:
        """
        Generate BIRRD instruction array for the given reduction pattern.

        The BIRRD must:
        1. Reduce partial sums from columns in each reduction group
        2. Route reduced results to correct output positions

        This is a simplified implementation that handles common patterns.
        """
        instructions = np.full((self.P0, self.P1), PS, dtype=np.int8)

        # Determine reduction pattern
        num_groups = len(reduction_groups)
        group_sizes = [len(g) for g in reduction_groups]

        # Generate instructions based on reduction pattern
        if num_groups == self.AW:
            # No reduction needed (1:1 mapping) - all pass
            pass
        elif all(size == 2 for size in group_sizes):
            # 2:1 reduction pattern
            instructions = self._generate_2_to_1_reduction()
        elif all(size == 4 for size in group_sizes):
            # 4:1 reduction pattern
            instructions = self._generate_4_to_1_reduction()
        else:
            # Mixed or complex pattern - use generic algorithm
            instructions = self._generate_generic_reduction(reduction_groups)

        return BIRRDConfig(
            num_stages=self.P0,
            switches_per_stage=self.P1,
            instructions=instructions,
        )

    def _generate_2_to_1_reduction(self) -> np.ndarray:
        """
        Generate BIRRD instructions for 2:1 reduction.

        Adjacent column pairs reduce together.
        """
        inst = np.full((self.P0, self.P1), PS, dtype=np.int8)

        if self.AW == 4:
            # 3-stage BIRRD for AW=4
            inst[1, :] = [AR, AL]  # Stage 1: reduce pairs
            inst[2, :] = [SW, PS]  # Stage 2: reorder
        elif self.AW == 8:
            # 6-stage BIRRD for AW=8
            inst[2, :] = [AR, AR, AL, AL]
            inst[3, :] = [SW, SW, SW, SW]
            inst[4, :] = [SW, PS, PS, SW]
        elif self.AW == 16:
            # 8-stage BIRRD for AW=16
            inst[3, :] = [AL, AL, AL, AL, AR, AR, AR, AR]
            inst[4, :] = [SW, SW, SW, SW, SW, SW, SW, SW]

        return inst

    def _generate_4_to_1_reduction(self) -> np.ndarray:
        """
        Generate BIRRD instructions for 4:1 reduction.

        Groups of 4 columns reduce together.
        """
        inst = np.full((self.P0, self.P1), PS, dtype=np.int8)

        if self.AW == 8:
            # First reduce 4:2, then 2:1
            inst[1, :] = [AR, AR, AL, AL]  # 4:2
            inst[2, :] = [AR, AR, AL, AL]  # Combine
            inst[3, :] = [AR, PS, AL, PS]  # 2:1

        return inst

    def _generate_generic_reduction(self, reduction_groups: List[List[int]]) -> np.ndarray:
        """
        Generate BIRRD instructions for arbitrary reduction pattern.

        Uses a routing-based algorithm to determine switch configurations.
        """
        inst = np.full((self.P0, self.P1), PS, dtype=np.int8)

        # Simplified: just generate pass-through for now
        # A full implementation would use the butterfly routing algorithm
        # from the FEATHER paper

        return inst

    # =========================================================================
    # Buffer addressing
    # =========================================================================

    def lower_wvn_layout(self, layout: SetWVNLayout) -> Callable[[int, int], Tuple[int, int]]:
        """
        Generate buffer address function from WVN layout instruction.

        Returns:
            Function mapping (vn_row, vn_col) -> (buffer_row, buffer_col)
        """
        wvn = WVNLayout(
            order=LayoutOrder(layout.order),
            AW=self.AW,
            N_L0=layout.N_L0,
            N_L1=layout.N_L1,
            K_L1=layout.K_L1,
        )
        return wvn.vn_to_buffer_addr

    def lower_ivn_layout(self, layout: SetIVNLayout) -> Callable[[int, int], Tuple[int, int]]:
        """
        Generate buffer address function from IVN layout instruction.
        """
        ivn = IVNLayout(
            order=LayoutOrder(layout.order),
            AW=self.AW,
            M_L0=layout.M_L0,
            M_L1=layout.M_L1,
            J_L1=layout.J_L1,
        )
        return ivn.vn_to_buffer_addr

    def lower_ovn_layout(self, layout: SetOVNLayout) -> Callable[[int, int], Tuple[int, int]]:
        """
        Generate buffer address function from OVN layout instruction.
        """
        ovn = OVNLayout(
            order=LayoutOrder(layout.order),
            AW=self.AW,
            P_L0=layout.P_L0,
            P_L1=layout.P_L1,
            Q_L1=layout.Q_L1,
        )
        return ovn.vn_to_buffer_addr


def generate_birrd_for_gemm(AW: int, reduction_ratio: int = 2) -> np.ndarray:
    """
    Generate BIRRD instruction array for GEMM workload.

    Args:
        AW: Array width
        reduction_ratio: How many columns reduce together (typically 2)

    Returns:
        BIRRD instruction array of shape (P0, P1)
    """
    LOG2_AW = int(log2(AW))
    P0 = 2 * LOG2_AW if AW > 4 else 2 * LOG2_AW - 1
    P1 = AW // 2

    if AW == 4:
        return np.array([[PS, PS], [AR, AL], [SW, PS]], dtype=np.int8)
    elif AW == 8:
        return np.array([
            [PS, PS, PS, PS],
            [PS, PS, PS, PS],
            [AR, AR, AL, AL],
            [SW, SW, SW, SW],
            [SW, PS, PS, SW],
            [PS, PS, PS, PS],
        ], dtype=np.int8)
    elif AW == 16:
        return np.array([
            [PS, SW, PS, SW, PS, SW, PS, SW],
            [PS, PS, SW, PS, PS, PS, SW, PS],
            [PS, PS, PS, PS, PS, PS, PS, PS],
            [AL, AL, AL, AL, AR, AR, AR, AR],
            [SW, SW, SW, SW, SW, SW, SW, SW],
            [PS, PS, PS, PS, PS, PS, PS, PS],
            [PS, PS, PS, PS, PS, PS, PS, PS],
            [PS, PS, PS, PS, PS, PS, PS, PS],
        ], dtype=np.int8)
    else:
        # Default pass-through
        return np.full((P0, P1), PS, dtype=np.int8)
