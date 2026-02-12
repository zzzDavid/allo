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


def _simulate_birrd_output_col_map(birrd_inst, AW):
    """Simulate BIRRD network to derive the output column map.

    Traces symbolic inputs through the BIRRD butterfly network to determine
    which output positions contain the reduced values for each M-position.
    Inputs at positions j and j+Mt (where Mt=AW//2) are paired for reduction.

    Args:
        birrd_inst: BIRRD instruction array [P0, P1]
        AW: Array width

    Returns:
        int32 array of shape [Mt] mapping M-position to BIRRD output column,
        or None if the BIRRD config doesn't produce valid 2-way reduction.
    """
    from feather_minisa import reverse_bits

    Mt = AW // 2
    P0, P1 = birrd_inst.shape
    LOG2_AW = int(log2(AW))

    # Compute routing tables
    route_left_table = np.zeros((P0, P1), dtype=int)
    route_right_table = np.zeros((P0, P1), dtype=int)
    for stage in range(P0):
        if stage == P0 - 1:
            for sw in range(P1):
                route_left_table[stage, sw] = 2 * sw
                route_right_table[stage, sw] = 2 * sw + 1
        else:
            rev_bits_factor = (
                2 if stage == 0
                else min(LOG2_AW, 2 + stage, 2 * LOG2_AW - stage)
            )
            for sw in range(P1):
                route_left_table[stage, sw] = reverse_bits(2 * sw, rev_bits_factor)
                route_right_table[stage, sw] = reverse_bits(2 * sw + 1, rev_bits_factor)

    # Represent each wire value as a frozenset of contributing input indices
    buf = {}
    for pos in range(AW):
        buf[(0, pos)] = frozenset({pos})

    for stage in range(P0):
        for sw in range(P1):
            left_in = buf.get((stage, 2 * sw), frozenset())
            right_in = buf.get((stage, 2 * sw + 1), frozenset())
            op = int(birrd_inst[stage, sw])

            if op == PS:
                left_out, right_out = left_in, right_in
            elif op == AR:
                left_out = left_in
                right_out = left_in | right_in
            elif op == AL:
                left_out = left_in | right_in
                right_out = right_in
            else:  # SW
                left_out, right_out = right_in, left_in

            left_dest = int(route_left_table[stage, sw])
            right_dest = int(route_right_table[stage, sw])
            buf[(stage + 1, left_dest)] = left_out
            buf[(stage + 1, right_dest)] = right_out

    # Find output columns containing reduced pairs (m, m+Mt)
    col_map = []
    for m in range(Mt):
        target = frozenset({m, m + Mt})
        found = False
        for col in range(AW):
            if buf.get((P0, col), frozenset()) == target:
                col_map.append(col)
                found = True
                break
        if not found:
            return None  # This BIRRD config doesn't produce valid 2-way reduction
    return np.array(col_map, dtype=np.int32)


# BIRRD instruction tables for all (AW, order) combinations.
# Order 0: standard reduction pattern from FEATHER paper.
# Orders 1-5: variations that produce different output permutations
# by modifying switch operations at different stages.

_BIRRD_INST_TABLES = {
    # AW=4: P0=3 stages, P1=2 switches
    # Pairs to reduce: (0,2) and (1,3)
    # Stage 0 must be all-PS or all-SW to keep pairs co-located at stage 1
    (4, 0): np.array([
        [PS, PS],
        [AR, AL],
        [SW, PS],
    ], dtype=np.int8),
    (4, 1): np.array([
        [PS, PS],
        [AL, AR],
        [SW, PS],
    ], dtype=np.int8),
    (4, 2): np.array([
        [PS, PS],
        [AR, AL],
        [PS, PS],
    ], dtype=np.int8),
    (4, 3): np.array([
        [PS, PS],
        [AR, AL],
        [PS, SW],
    ], dtype=np.int8),
    (4, 4): np.array([
        [SW, SW],
        [AR, AL],
        [PS, PS],
    ], dtype=np.int8),
    (4, 5): np.array([
        [SW, SW],
        [AR, AL],
        [SW, PS],
    ], dtype=np.int8),

    # AW=8: P0=6 stages, P1=4 switches
    (8, 0): np.array([
        [PS, PS, PS, PS],
        [PS, PS, PS, PS],
        [AR, AR, AL, AL],
        [SW, SW, SW, SW],
        [SW, PS, PS, SW],
        [PS, PS, PS, PS],
    ], dtype=np.int8),
    (8, 1): np.array([
        [PS, PS, PS, PS],
        [PS, PS, PS, PS],
        [AL, AL, AR, AR],
        [SW, SW, SW, SW],
        [SW, PS, PS, SW],
        [PS, PS, PS, PS],
    ], dtype=np.int8),
    (8, 2): np.array([
        [PS, PS, PS, PS],
        [PS, PS, PS, PS],
        [AR, AR, AL, AL],
        [PS, PS, PS, PS],
        [SW, PS, PS, SW],
        [PS, PS, PS, PS],
    ], dtype=np.int8),
    (8, 3): np.array([
        [PS, PS, PS, PS],
        [PS, PS, PS, PS],
        [AL, AL, AR, AR],
        [PS, PS, PS, PS],
        [SW, PS, PS, SW],
        [PS, PS, PS, PS],
    ], dtype=np.int8),
    (8, 4): np.array([
        [PS, PS, PS, PS],
        [PS, PS, PS, PS],
        [AR, AR, AL, AL],
        [PS, PS, PS, PS],
        [PS, SW, SW, PS],
        [PS, PS, PS, PS],
    ], dtype=np.int8),
    (8, 5): np.array([
        [PS, PS, PS, PS],
        [PS, PS, PS, PS],
        [AL, AL, AR, AR],
        [PS, PS, PS, PS],
        [PS, SW, SW, PS],
        [PS, PS, PS, PS],
    ], dtype=np.int8),

    # AW=16: P0=8 stages, P1=8 switches
    (16, 0): np.array([
        [PS, SW, PS, SW, PS, SW, PS, SW],
        [PS, PS, SW, PS, PS, PS, SW, PS],
        [PS, PS, PS, PS, PS, PS, PS, PS],
        [AL, AL, AL, AL, AR, AR, AR, AR],
        [SW, SW, SW, SW, SW, SW, SW, SW],
        [PS, PS, PS, PS, PS, PS, PS, PS],
        [PS, PS, PS, PS, PS, PS, PS, PS],
        [PS, PS, PS, PS, PS, PS, PS, PS],
    ], dtype=np.int8),
    (16, 1): np.array([
        [PS, SW, PS, SW, PS, SW, PS, SW],
        [PS, PS, SW, PS, PS, PS, SW, PS],
        [PS, PS, PS, PS, PS, PS, PS, PS],
        [AR, AR, AR, AR, AL, AL, AL, AL],
        [SW, SW, SW, SW, SW, SW, SW, SW],
        [PS, PS, PS, PS, PS, PS, PS, PS],
        [PS, PS, PS, PS, PS, PS, PS, PS],
        [PS, PS, PS, PS, PS, PS, PS, PS],
    ], dtype=np.int8),
    (16, 2): np.array([
        [PS, SW, PS, SW, PS, SW, PS, SW],
        [PS, PS, SW, PS, PS, PS, SW, PS],
        [PS, PS, PS, PS, PS, PS, PS, PS],
        [AL, AL, AL, AL, AR, AR, AR, AR],
        [PS, PS, PS, PS, PS, PS, PS, PS],
        [PS, PS, PS, PS, PS, PS, PS, PS],
        [PS, PS, PS, PS, PS, PS, PS, PS],
        [PS, PS, PS, PS, PS, PS, PS, PS],
    ], dtype=np.int8),
    (16, 3): np.array([
        [PS, SW, PS, SW, PS, SW, PS, SW],
        [PS, PS, SW, PS, PS, PS, SW, PS],
        [PS, PS, PS, PS, PS, PS, PS, PS],
        [AR, AR, AR, AR, AL, AL, AL, AL],
        [PS, PS, PS, PS, PS, PS, PS, PS],
        [PS, PS, PS, PS, PS, PS, PS, PS],
        [PS, PS, PS, PS, PS, PS, PS, PS],
        [PS, PS, PS, PS, PS, PS, PS, PS],
    ], dtype=np.int8),
    (16, 4): np.array([
        [SW, PS, SW, PS, SW, PS, SW, PS],
        [PS, PS, SW, PS, PS, PS, SW, PS],
        [PS, PS, PS, PS, PS, PS, PS, PS],
        [AL, AL, AL, AL, AR, AR, AR, AR],
        [SW, SW, SW, SW, SW, SW, SW, SW],
        [PS, PS, PS, PS, PS, PS, PS, PS],
        [PS, PS, PS, PS, PS, PS, PS, PS],
        [PS, PS, PS, PS, PS, PS, PS, PS],
    ], dtype=np.int8),
    (16, 5): np.array([
        [SW, PS, SW, PS, SW, PS, SW, PS],
        [PS, PS, SW, PS, PS, PS, SW, PS],
        [PS, PS, PS, PS, PS, PS, PS, PS],
        [AR, AR, AR, AR, AL, AL, AL, AL],
        [SW, SW, SW, SW, SW, SW, SW, SW],
        [PS, PS, PS, PS, PS, PS, PS, PS],
        [PS, PS, PS, PS, PS, PS, PS, PS],
        [PS, PS, PS, PS, PS, PS, PS, PS],
    ], dtype=np.int8),
}


def lower_ovn_layout(
    ovn: SetOVNLayout,
    AW: int,
    AH: int
) -> np.ndarray:
    """Lower SetOVNLayout to BIRRD instruction array.

    The OVN layout determines how partial sums are reduced and reordered
    in the BIRRD network. Different orders produce different switch
    operation patterns that implement different reduction/reorder trees.

    Args:
        ovn: Output VN layout configuration
        AW: Array width
        AH: Array height

    Returns:
        BIRRD instruction array [P0, P1]
    """
    key = (AW, ovn.order)
    if key in _BIRRD_INST_TABLES:
        return _BIRRD_INST_TABLES[key].copy()
    raise ValueError(f"Unsupported (AW={AW}, order={ovn.order}) combination")


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


def compute_birrd_routing_table(AW: int) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute BIRRD butterfly routing tables.

    For each BIRRD stage and switch, computes the destination positions
    for left and right outputs using the bit-reversal routing formula
    from the FEATHER+ butterfly network.

    Args:
        AW: Array width (must be power of 2)

    Returns:
        (route_left, route_right): int32 arrays of shape [P0, P1]
        route_left[stage, sw] = destination position for left output of switch sw at stage
        route_right[stage, sw] = destination position for right output of switch sw at stage
    """
    from feather_minisa import reverse_bits

    LOG2_AW = int(log2(AW))
    P0, P1 = compute_birrd_params(AW)

    route_left = np.zeros((P0, P1), dtype=np.int32)
    route_right = np.zeros((P0, P1), dtype=np.int32)

    for stage in range(P0):
        if stage == P0 - 1:
            # Last stage: direct output (no bit-reversal)
            for sw in range(P1):
                route_left[stage, sw] = 2 * sw
                route_right[stage, sw] = 2 * sw + 1
        else:
            rev_bits_factor = (
                2 if stage == 0
                else min(LOG2_AW, 2 + stage, 2 * LOG2_AW - stage)
            )
            for sw in range(P1):
                route_left[stage, sw] = reverse_bits(2 * sw, rev_bits_factor)
                route_right[stage, sw] = reverse_bits(2 * sw + 1, rev_bits_factor)

    return route_left, route_right


def compute_output_col_map(AW: int, ovn_order: int = 0) -> np.ndarray:
    """Compute the BIRRD output column to M-position mapping.

    The BIRRD butterfly network produces outputs in a permuted order.
    This function returns the mapping from local M position to the
    BIRRD output column that contains the corresponding reduced value.

    The mapping depends on both AW and the OVN order, since different
    OVN orders use different BIRRD switch patterns that route reduced
    values to different output positions.

    Args:
        AW: Array width (4, 8, or 16)
        ovn_order: OVN layout order (0-5)

    Returns:
        int32 array of shape [Mt] where Mt = AW // 2
    """
    key = (AW, ovn_order)
    if key not in _BIRRD_INST_TABLES:
        raise ValueError(f"Unsupported (AW={AW}, order={ovn_order})")

    birrd_inst = _BIRRD_INST_TABLES[key]
    col_map = _simulate_birrd_output_col_map(birrd_inst, AW)
    if col_map is None:
        raise ValueError(
            f"BIRRD config for (AW={AW}, order={ovn_order}) does not "
            f"produce valid 2-way reduction"
        )
    return col_map


def precompute_full_matrix_configs(program: MINISAProgram) -> dict:
    """Precompute all configuration arrays for full-matrix Allo execution.

    Bundles BIRRD instructions, routing tables, and output column map
    into a single dict for passing to the Allo function.

    Args:
        program: MINISA program with layout configurations

    Returns:
        dict with keys: birrd_inst, route_left, route_right, output_col_map
    """
    AW = program.AW
    AH = program.AH
    ovn_order = program.ovn_layout.order

    birrd_inst = lower_ovn_layout(program.ovn_layout, AW, AH)
    route_left, route_right = compute_birrd_routing_table(AW)
    output_col_map = compute_output_col_map(AW, ovn_order)

    return {
        'birrd_inst': birrd_inst,
        'route_left': route_left,
        'route_right': route_right,
        'output_col_map': output_col_map,
    }


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
