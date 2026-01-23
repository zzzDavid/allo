# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""MINISA lowering: convert MINISA instructions to Allo configuration tensors.

This module translates high-level MINISA instructions into low-level
configuration arrays that drive the FEATHER+ Allo dataflow hardware.

The lowering produces:
- BIRRD instruction arrays from SetOVNLayout
- Input tile extraction functions from SetIVNLayout
- Weight tile extraction functions from SetWVNLayout
- Tile slice bounds from SetMapping

IMPORTANT: This module performs NO COMPUTE. It only generates configuration
data structures. All actual computation is performed by Allo kernels.
"""

from dataclasses import dataclass
from math import log2
from typing import Tuple, Callable, Optional

import numpy as np

from .isa import (
    SetIVNLayout,
    SetWVNLayout,
    SetOVNLayout,
    SetMapping,
    MINISAProgram,
)


# BIRRD switch operation codes
PS = 0  # Pass
AR = 1  # Add Right
AL = 2  # Add Left
SW = 3  # Swap


@dataclass
class LoweredConfig:
    """Configuration tensors produced by MINISA lowering.

    These tensors are passed to the Allo dataflow region to configure
    the hardware for a specific workload.

    Attributes:
        birrd_inst: BIRRD instruction array [P0, P1]
        AW: Array width
        AH: Array height
        P0: Number of BIRRD stages
        P1: Number of switches per stage
    """
    birrd_inst: np.ndarray
    AW: int
    AH: int
    P0: int
    P1: int


def compute_birrd_params(AW: int) -> Tuple[int, int]:
    """Compute BIRRD network parameters from array width.

    Args:
        AW: Array width (must be power of 2)

    Returns:
        (P0, P1): Number of stages, switches per stage
    """
    LOG2_AW = int(log2(AW))
    P0 = 2 * LOG2_AW if AW > 4 else 2 * LOG2_AW - 1
    P1 = AW // 2
    return P0, P1


def lower_ovn_layout(
    ovn: SetOVNLayout,
    AW: int,
    AH: int
) -> np.ndarray:
    """Lower SetOVNLayout to BIRRD instruction array.

    The OVN layout determines how partial sums are reduced and reordered
    in the BIRRD network. Different layouts produce different instruction
    patterns that implement the required reduction tree.

    Args:
        ovn: Output VN layout configuration
        AW: Array width
        AH: Array height

    Returns:
        BIRRD instruction array [P0, P1]
    """
    P0, P1 = compute_birrd_params(AW)

    # Generate BIRRD instructions based on OVN order and dimensions
    # Default patterns are provided for common array widths
    # These implement standard reduction with output reordering

    if AW == 4:
        # AW=4: 3 stages, 2 switches
        inst = np.array([
            [PS, PS],
            [AR, AL],
            [SW, PS],
        ], dtype=np.int8)
    elif AW == 8:
        # AW=8: 6 stages, 4 switches
        # Standard reduction pattern from FEATHER paper
        inst = np.array([
            [PS, PS, PS, PS],
            [PS, PS, PS, PS],
            [AR, AR, AL, AL],
            [SW, SW, SW, SW],
            [SW, PS, PS, SW],
            [PS, PS, PS, PS],
        ], dtype=np.int8)
    elif AW == 16:
        # AW=16: 8 stages, 8 switches
        inst = np.array([
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
        raise ValueError(f"Unsupported array width: {AW}")

    # Apply layout-specific modifications based on OVN order
    # Different orders may require different reduction trees
    if ovn.order != 0:
        # For non-default orders, modify the instruction pattern
        # This is a simplified version - full implementation would
        # compute the exact butterfly configuration for each order
        pass

    return inst


def lower_ivn_layout_to_reorder(
    ivn: SetIVNLayout,
    AW: int,
    AH: int
) -> Callable[[np.ndarray], np.ndarray]:
    """Lower SetIVNLayout to input tile reordering function.

    The IVN layout determines how input data is reordered before being
    fed to the PE array. This function returns a Python callable that
    performs the reordering (NO COMPUTE - just data movement).

    Args:
        ivn: Input VN layout configuration
        AW: Array width
        AH: Array height

    Returns:
        Function that reorders input tiles: (Mt, Kt) -> (AH, AW)
    """
    Mt = AW // 2
    Kt = 2 * AH

    def reorder_input_tile(tile: np.ndarray) -> np.ndarray:
        """Reorder input tile for FEATHER+ PE array.

        This implements the IVN layout transformation. The tile is split
        and transposed to match the PE array's expected input format.

        Args:
            tile: Input tile [Mt, Kt]

        Returns:
            Reordered tile [AH, AW]
        """
        if tile.shape != (Mt, Kt):
            raise ValueError(f"Expected tile shape ({Mt}, {Kt}), got {tile.shape}")

        # Split on K dimension for two-phase reduction
        B_left, B_right = np.hsplit(tile, 2)  # Each: (Mt, Kt/2) = (AW//2, AH)

        # Transpose and concatenate for PE array layout
        C = np.hstack([B_left.transpose(), B_right.transpose()])

        assert C.shape == (AH, AW)
        return np.ascontiguousarray(C)

    return reorder_input_tile


def lower_wvn_layout_to_reorder(
    wvn: SetWVNLayout,
    AW: int,
    AH: int
) -> Callable[[np.ndarray], np.ndarray]:
    """Lower SetWVNLayout to weight tile reordering function.

    The WVN layout determines how weight data is reordered before being
    loaded into the PE array's weight buffers.

    Args:
        wvn: Weight VN layout configuration
        AW: Array width
        AH: Array height

    Returns:
        Function that reorders weight tiles: (Kt, Nt) -> (AH, AW, AH)
    """
    Kt = 2 * AH
    Nt = AH

    def reorder_weight_tile(tile: np.ndarray) -> np.ndarray:
        """Reorder weight tile for FEATHER+ PE array.

        This implements the WVN layout transformation. The tile is
        transformed to match the PE array's 3D weight buffer format.

        Args:
            tile: Weight tile [Kt, Nt]

        Returns:
            Reordered weights [AH, AW, AH]
        """
        if tile.shape != (Kt, Nt):
            raise ValueError(f"Expected tile shape ({Kt}, {Nt}), got {tile.shape}")

        # Split on K dimension
        B_left, B_right = np.vsplit(tile, 2)  # Each: (Kt/2, Nt) = (AH, AH)

        # Replicate for each PE column and arrange in 3D
        C_left = np.array([B_left.transpose()] * (AW // 2))
        C_right = np.array([B_right.transpose()] * (AW // 2))

        D = np.vstack([C_left, C_right]).transpose(1, 0, 2)

        assert D.shape == (AH, AW, AH)
        return np.ascontiguousarray(D)

    return reorder_weight_tile


def lower_mapping_to_slices(
    mapping: SetMapping,
    AW: int,
    AH: int
) -> Tuple[slice, slice, slice, slice]:
    """Lower SetMapping to tensor slice objects.

    Converts mapping bounds to Python slice objects for extracting
    tiles from input, weight, and output tensors.

    Args:
        mapping: Tile mapping configuration
        AW: Array width
        AH: Array height

    Returns:
        (m_slice, n_slice, k_slice, output_slice): Slices for tensor access
    """
    Mt = AW // 2
    Nt = AH
    Kt = 2 * AH

    m_slice = slice(mapping.m_start, mapping.m_end)
    n_slice = slice(mapping.n_start, mapping.n_end)
    k_slice = slice(mapping.k_start, mapping.k_end)

    # Output slice depends on the specific mapping configuration
    # For standard GEMM layout:
    output_m_slice = slice(mapping.n_start, mapping.n_end)  # Output rows = N
    output_n_slice = slice(mapping.m_start * 2, mapping.m_end * 2)  # Output cols = 2*M

    return m_slice, n_slice, k_slice, (output_m_slice, output_n_slice)


def lower_minisa_program(program: MINISAProgram) -> LoweredConfig:
    """Lower a complete MINISA program to Allo configuration.

    This is the main entry point for MINISA lowering. It processes
    all layout instructions and generates the configuration tensors
    needed to execute the program on FEATHER+.

    Args:
        program: MINISA program to lower

    Returns:
        LoweredConfig with all configuration tensors
    """
    AW = program.AW
    AH = program.AH
    P0, P1 = compute_birrd_params(AW)

    # Lower OVN layout to BIRRD instructions
    birrd_inst = lower_ovn_layout(program.ovn_layout, AW, AH)

    return LoweredConfig(
        birrd_inst=birrd_inst,
        AW=AW,
        AH=AH,
        P0=P0,
        P1=P1,
    )


class TileExtractor:
    """Helper class for extracting and reordering tiles from tensors.

    This encapsulates the layout-specific tile extraction logic generated
    from MINISA IVN and WVN layouts. It performs NO COMPUTE - only
    data movement and reordering.
    """

    def __init__(self, program: MINISAProgram):
        """Initialize tile extractor from MINISA program.

        Args:
            program: MINISA program with layout configurations
        """
        self.AW = program.AW
        self.AH = program.AH
        self.Mt = self.AW // 2
        self.Nt = self.AH
        self.Kt = 2 * self.AH

        # Lower layouts to reorder functions
        self.reorder_input = lower_ivn_layout_to_reorder(
            program.ivn_layout, self.AW, self.AH
        )
        self.reorder_weight = lower_wvn_layout_to_reorder(
            program.wvn_layout, self.AW, self.AH
        )

    def extract_input_tile(
        self,
        inputs: np.ndarray,
        mapping: SetMapping
    ) -> np.ndarray:
        """Extract and reorder input tile for Allo execution.

        Args:
            inputs: Full input tensor [M, K]
            mapping: Tile mapping specifying slice bounds

        Returns:
            Reordered input tile [AH, AW]
        """
        # Extract raw tile (no compute)
        tile = inputs[mapping.m_start:mapping.m_end,
                      mapping.k_start:mapping.k_end]

        # Reorder for PE array layout (no compute)
        return self.reorder_input(tile)

    def extract_weight_tile(
        self,
        weights: np.ndarray,
        mapping: SetMapping
    ) -> np.ndarray:
        """Extract and reorder weight tile for Allo execution.

        Args:
            weights: Full weight tensor [K, N]
            mapping: Tile mapping specifying slice bounds

        Returns:
            Reordered weight tile [AH, AW, AH]
        """
        # Extract raw tile (no compute)
        tile = weights[mapping.k_start:mapping.k_end,
                       mapping.n_start:mapping.n_end]

        # Reorder for PE array layout (no compute)
        return self.reorder_weight(tile)

    def get_output_slices(self, mapping: SetMapping) -> Tuple[slice, slice]:
        """Get output tensor slices for storing tile results.

        Args:
            mapping: Tile mapping specifying output location

        Returns:
            (row_slice, col_slice) for output tensor access
        """
        # Output layout: [N, 2*M] with specific reordering from BIRRD
        row_slice = slice(mapping.n_start, mapping.n_end)
        col_slice = slice(mapping.m_start * 2, mapping.m_end * 2)
        return row_slice, col_slice


def extract_output_for_verification(
    oActs: np.ndarray,
    ref_shape: Tuple[int, int],
    AW: int
) -> np.ndarray:
    """Extract and reorder output for numpy reference comparison.

    The BIRRD network produces output in a specific layout that needs
    to be reordered to match the standard GEMM output layout.

    This is used ONLY FOR VERIFICATION - not part of the compute path.

    Args:
        oActs: Raw output from FEATHER+ [N, 2*M]
        ref_shape: Expected output shape [M, N]
        AW: Array width

    Returns:
        Reordered output matching numpy reference layout
    """
    M, N = ref_shape
    Mt = AW // 2

    oActs = oActs.transpose()
    extracted_data = np.zeros(shape=ref_shape, dtype=oActs.dtype)

    for m in range(M // Mt):
        if AW == 16:  # Mt == 8
            np.copyto(extracted_data[m * Mt], oActs[m * 2 * Mt + 8])
            np.copyto(extracted_data[m * Mt + 1], oActs[m * 2 * Mt + 10])
            np.copyto(extracted_data[m * Mt + 2], oActs[m * 2 * Mt + 11])
            np.copyto(extracted_data[m * Mt + 3], oActs[m * 2 * Mt + 9])
            np.copyto(extracted_data[m * Mt + 4], oActs[m * 2 * Mt + 5])
            np.copyto(extracted_data[m * Mt + 5], oActs[m * 2 * Mt + 6])
            np.copyto(extracted_data[m * Mt + 6], oActs[m * 2 * Mt + 7])
            np.copyto(extracted_data[m * Mt + 7], oActs[m * 2 * Mt + 4])
        elif AW == 8:  # Mt == 4
            np.copyto(extracted_data[m * Mt], oActs[m * 2 * Mt + 6])
            np.copyto(extracted_data[m * Mt + 1], oActs[m * 2 * Mt + 5])
            np.copyto(extracted_data[m * Mt + 2], oActs[m * 2 * Mt + 2])
            np.copyto(extracted_data[m * Mt + 3], oActs[m * 2 * Mt + 1])
        elif AW == 4:  # Mt == 2
            np.copyto(extracted_data[m * Mt], oActs[m * 2 * Mt + 2])
            np.copyto(extracted_data[m * Mt + 1], oActs[m * 2 * Mt])

    return extracted_data
