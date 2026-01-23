# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
VN Layout Descriptors for MINISA

Defines layout representations for Virtual Neurons in on-chip buffers:
- Each layout specifies the nested-loop order for VN placement
- The VN constraint (e.g., K_L0 = AH for weights) is implicit
- A 3-bit order encoding captures the 6 legal permutations

Reference: MINISA paper Section IV-C (VN-Granularity Layouts)
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Tuple, Callable, Optional
import numpy as np


class LayoutOrder(IntEnum):
    """
    3-bit encoding for VN layout order permutation.

    For WVN (weights K×N with K_L0=AH constraint):
      Remaining ranks: {k_L1, n_L0, n_L1}
      Order specifies outer→inner loop order

    Similar encoding applies to IVN and OVN with their respective ranks.
    """
    # Format: outer -> middle -> inner
    ORDER_000 = 0b000  # e.g., k_L1 -> n_L0 -> n_L1 for WVN
    ORDER_001 = 0b001  # e.g., k_L1 -> n_L1 -> n_L0 for WVN
    ORDER_010 = 0b010  # e.g., n_L0 -> k_L1 -> n_L1 for WVN
    ORDER_011 = 0b011  # e.g., n_L0 -> n_L1 -> k_L1 for WVN
    ORDER_100 = 0b100  # e.g., n_L1 -> k_L1 -> n_L0 for WVN
    ORDER_101 = 0b101  # e.g., n_L1 -> n_L0 -> k_L1 for WVN


# Permutation tables for each order
WVN_PERMUTATIONS = {
    LayoutOrder.ORDER_000: ('k_L1', 'n_L0', 'n_L1'),
    LayoutOrder.ORDER_001: ('k_L1', 'n_L1', 'n_L0'),
    LayoutOrder.ORDER_010: ('n_L0', 'k_L1', 'n_L1'),
    LayoutOrder.ORDER_011: ('n_L0', 'n_L1', 'k_L1'),
    LayoutOrder.ORDER_100: ('n_L1', 'k_L1', 'n_L0'),
    LayoutOrder.ORDER_101: ('n_L1', 'n_L0', 'k_L1'),
}

IVN_PERMUTATIONS = {
    LayoutOrder.ORDER_000: ('m_L1', 'm_L0', 'j_L1'),
    LayoutOrder.ORDER_001: ('m_L1', 'j_L1', 'm_L0'),
    LayoutOrder.ORDER_010: ('m_L0', 'm_L1', 'j_L1'),
    LayoutOrder.ORDER_011: ('m_L0', 'j_L1', 'm_L1'),
    LayoutOrder.ORDER_100: ('j_L1', 'm_L1', 'm_L0'),
    LayoutOrder.ORDER_101: ('j_L1', 'm_L0', 'm_L1'),
}

OVN_PERMUTATIONS = {
    LayoutOrder.ORDER_000: ('p_L1', 'p_L0', 'q_L1'),
    LayoutOrder.ORDER_001: ('p_L1', 'q_L1', 'p_L0'),
    LayoutOrder.ORDER_010: ('p_L0', 'p_L1', 'q_L1'),
    LayoutOrder.ORDER_011: ('p_L0', 'q_L1', 'p_L1'),
    LayoutOrder.ORDER_100: ('q_L1', 'p_L1', 'p_L0'),
    LayoutOrder.ORDER_101: ('q_L1', 'p_L0', 'p_L1'),
}


@dataclass
class VNLayout:
    """
    Base class for VN layout descriptors.

    A VN layout specifies how Virtual Neurons are organized in an on-chip buffer.
    The layout is parameterized by:
    - order: Permutation of partitioned ranks (3-bit encoding)
    - Dimension factors: How tensor ranks are partitioned

    The VN constraint (L0 = AH for the reduction dimension) is implicit.
    """
    order: LayoutOrder
    AW: int  # Buffer width (VNs per row)

    def get_permutation_indices(self) -> Tuple[int, int, int]:
        """
        Get the permutation as indices into the dimension arrays.

        Returns:
            Tuple of (outer, middle, inner) indices corresponding to
            positions in the D (dimensions) and I (indices) arrays.
        """
        # Map order to permutation indices
        # D = [dim0, dim1, dim2], I = [idx0, idx1, idx2]
        # Order determines which of D/I goes in outer/middle/inner position
        permutations = {
            LayoutOrder.ORDER_000: (0, 1, 2),  # dim0 -> dim1 -> dim2
            LayoutOrder.ORDER_001: (0, 2, 1),  # dim0 -> dim2 -> dim1
            LayoutOrder.ORDER_010: (1, 0, 2),  # dim1 -> dim0 -> dim2
            LayoutOrder.ORDER_011: (1, 2, 0),  # dim1 -> dim2 -> dim0
            LayoutOrder.ORDER_100: (2, 0, 1),  # dim2 -> dim0 -> dim1
            LayoutOrder.ORDER_101: (2, 1, 0),  # dim2 -> dim1 -> dim0
        }
        return permutations[self.order]

    def compute_linear_index(self, D: List[int], I: List[int]) -> int:
        """
        Compute linear VN index given dimensions and indices.

        Args:
            D: List of dimension sizes [D0, D1, D2]
            I: List of indices [I0, I1, I2]

        Returns:
            Linear index L based on layout order
        """
        p0, p1, p2 = self.get_permutation_indices()
        L = I[p0] * D[p1] * D[p2] + I[p1] * D[p2] + I[p2]
        return L

    def linear_to_buffer_addr(self, L: int) -> Tuple[int, int]:
        """
        Convert linear VN index to buffer row and column.

        Args:
            L: Linear VN index

        Returns:
            (row, col) buffer address
        """
        row = L // self.AW
        col = L % self.AW
        return row, col


@dataclass
class WVNLayout(VNLayout):
    """
    Weight VN layout for stationary buffer.

    Weights have shape (K, N) where:
    - K is the reduction dimension (partitioned as K = K_L1 * K_L0)
    - N is the non-reduction dimension (partitioned as N = N_L1 * N_L0)
    - VN constraint: K_L0 = AH

    WVN(r, c) indexing:
    - r = k_L1 (row in VN matrix)
    - c = n_L1 * N_L0 + n_L0 (column in VN matrix)
    """
    N_L0: int   # Inner non-reduction factor (1 <= N_L0 <= AW)
    N_L1: int   # Outer non-reduction factor
    K_L1: int   # Outer reduction factor (K_L0 = AH implicit)

    @property
    def AH(self) -> int:
        """VN size (implicit K_L0)."""
        # This is set by the hardware config, typically stored elsewhere
        # For now, we don't store it here to avoid redundancy
        return 0  # Placeholder - should be passed from config

    def get_dimensions(self) -> List[int]:
        """Get dimension sizes [K_L1, N_L0, N_L1]."""
        return [self.K_L1, self.N_L0, self.N_L1]

    def vn_to_buffer_addr(self, r: int, c: int) -> Tuple[int, int]:
        """
        Convert WVN(r, c) to buffer (row, col) address.

        Args:
            r: WVN row index (k_L1)
            c: WVN column index (n_L1 * N_L0 + n_L0)

        Returns:
            (buffer_row, buffer_col)
        """
        # Decompose c into n_L0 and n_L1
        n_L0 = c % self.N_L0
        n_L1 = c // self.N_L0
        k_L1 = r

        # Get dimension sizes and indices
        D = self.get_dimensions()  # [K_L1, N_L0, N_L1]
        I = [k_L1, n_L0, n_L1]

        # Compute linear index
        L = self.compute_linear_index(D, I)

        # Convert to buffer address
        return self.linear_to_buffer_addr(L)

    def buffer_addr_to_vn(self, row: int, col: int) -> Tuple[int, int]:
        """
        Convert buffer address to WVN(r, c) indices.

        Args:
            row: Buffer row
            col: Buffer column

        Returns:
            (r, c) WVN indices, or (-1, -1) if invalid
        """
        L = row * self.AW + col
        D = self.get_dimensions()
        p0, p1, p2 = self.get_permutation_indices()

        # Reverse the linear index computation
        total = D[p0] * D[p1] * D[p2]
        if L >= total:
            return (-1, -1)

        # Extract indices
        I = [0, 0, 0]
        I[p2] = L % D[p2]
        I[p1] = (L // D[p2]) % D[p1]
        I[p0] = L // (D[p1] * D[p2])

        k_L1, n_L0, n_L1 = I
        r = k_L1
        c = n_L1 * self.N_L0 + n_L0
        return (r, c)


@dataclass
class IVNLayout(VNLayout):
    """
    Input VN layout for streaming buffer.

    Inputs have shape (M, J) where:
    - M is the non-reduction dimension (partitioned as M = M_L1 * M_L0)
    - J is the reduction dimension (partitioned as J = J_L1 * J_L0)
    - VN constraint: J_L0 = AH

    IVN(r, c) indexing:
    - r = m_L1 * M_L0 + m_L0 (row in VN matrix)
    - c = j_L1 (column in VN matrix)
    """
    M_L0: int   # Inner non-reduction factor
    M_L1: int   # Outer non-reduction factor
    J_L1: int   # Outer reduction factor (J_L0 = AH implicit)

    def get_dimensions(self) -> List[int]:
        """Get dimension sizes [M_L1, M_L0, J_L1]."""
        return [self.M_L1, self.M_L0, self.J_L1]

    def vn_to_buffer_addr(self, r: int, c: int) -> Tuple[int, int]:
        """
        Convert IVN(r, c) to buffer address.

        Args:
            r: IVN row index
            c: IVN column index (j_L1)

        Returns:
            (buffer_row, buffer_col)
        """
        # Decompose r into m_L0 and m_L1
        m_L0 = r % self.M_L0
        m_L1 = r // self.M_L0
        j_L1 = c

        D = self.get_dimensions()
        I = [m_L1, m_L0, j_L1]
        L = self.compute_linear_index(D, I)
        return self.linear_to_buffer_addr(L)


@dataclass
class OVNLayout(VNLayout):
    """
    Output VN layout for output buffer.

    Outputs have shape (P, Q) where:
    - P is the M-derived dimension (partitioned as P = P_L1 * P_L0)
    - Q is the N-derived dimension (partitioned as Q = Q_L1 * Q_L0)
    - VN constraint: Q_L0 = AH (output VN aligns with dot-product output)

    OVN(r, c) indexing:
    - r = p_L1 * P_L0 + p_L0 (row in VN matrix)
    - c = q_L1 (column in VN matrix)
    """
    P_L0: int   # Inner P factor
    P_L1: int   # Outer P factor
    Q_L1: int   # Outer Q factor (Q_L0 = AH implicit)

    def get_dimensions(self) -> List[int]:
        """Get dimension sizes [P_L1, P_L0, Q_L1]."""
        return [self.P_L1, self.P_L0, self.Q_L1]

    def vn_to_buffer_addr(self, r: int, c: int) -> Tuple[int, int]:
        """
        Convert OVN(r, c) to buffer address.

        Args:
            r: OVN row index
            c: OVN column index (q_L1)

        Returns:
            (buffer_row, buffer_col)
        """
        p_L0 = r % self.P_L0
        p_L1 = r // self.P_L0
        q_L1 = c

        D = self.get_dimensions()
        I = [p_L1, p_L0, q_L1]
        L = self.compute_linear_index(D, I)
        return self.linear_to_buffer_addr(L)


def create_wvn_buffer(K: int, N: int, AH: int, AW: int,
                      layout: WVNLayout,
                      weights: np.ndarray) -> np.ndarray:
    """
    Create weight VN buffer with specified layout.

    Args:
        K, N: Weight matrix dimensions
        AH, AW: Array dimensions (AH is VN size)
        layout: WVNLayout specifying buffer organization
        weights: Weight matrix of shape (K, N)

    Returns:
        Buffer array with VNs arranged according to layout
    """
    assert weights.shape == (K, N), f"Expected shape ({K}, {N}), got {weights.shape}"
    assert K % AH == 0, f"K ({K}) must be divisible by AH ({AH})"

    num_wvn_rows = K // AH
    num_wvn_cols = N

    # Calculate buffer size
    total_vns = num_wvn_rows * num_wvn_cols
    buffer_rows = (total_vns + AW - 1) // AW
    buffer = np.zeros((buffer_rows, AW, AH), dtype=weights.dtype)

    # Fill buffer according to layout
    for r in range(num_wvn_rows):
        for c in range(num_wvn_cols):
            # Extract VN data
            k_start = r * AH
            vn_data = weights[k_start:k_start + AH, c]

            # Get buffer address
            buf_row, buf_col = layout.vn_to_buffer_addr(r, c)

            if buf_row < buffer_rows:
                buffer[buf_row, buf_col, :] = vn_data

    return buffer


def extract_wvn_from_buffer(buffer: np.ndarray, r: int, c: int,
                            layout: WVNLayout) -> np.ndarray:
    """
    Extract a WVN from buffer given its VN coordinates.

    Args:
        buffer: VN buffer array of shape (rows, AW, AH)
        r, c: WVN coordinates
        layout: WVNLayout

    Returns:
        VN data array of shape (AH,)
    """
    buf_row, buf_col = layout.vn_to_buffer_addr(r, c)
    if buf_row >= buffer.shape[0]:
        return np.zeros(buffer.shape[2], dtype=buffer.dtype)
    return buffer[buf_row, buf_col, :]
