# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# FEATHER-ISA: ISA-level implementation of FEATHER+ accelerator with MINISA
#
# This implementation is based on the MINISA paper:
# "MINISA: Minimal Instruction Set Architecture for Next-gen Reconfigurable
# Inference Accelerator"
#
# MINISA provides 4 instructions that program FEATHER+ at Virtual Neuron (VN)
# granularity:
# 1. SetIVNLayout: Configure input VN layout
# 2. SetWVNLayout: Configure weight VN layout
# 3. SetOVNLayout: Configure output VN layout
# 4. SetMapping: Execute compute tile with VN-level mapping

import allo
from allo.ir.types import int8, int32, AlloType
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from enum import IntEnum


class LayoutOrder(IntEnum):
    """
    3-bit permutation encoding for VN layout orders.
    See MINISA paper Table III for details.
    """
    ORDER_000 = 0  # kL1 → nL0 → nL1 for WVN
    ORDER_001 = 1  # kL1 → nL1 → nL0 for WVN
    ORDER_010 = 2  # nL0 → kL1 → nL1 for WVN
    ORDER_011 = 3  # nL0 → nL1 → kL1 for WVN
    ORDER_100 = 4  # nL1 → kL1 → nL0 for WVN
    ORDER_101 = 5  # nL1 → nL0 → kL1 for WVN


@dataclass
class SetIVNLayout:
    """
    Configure the streaming buffer layout for input VNs.

    For input matrices of shape (M, J), we partition:
    - M = ML1 * ML0 where ML0 = AH (VN constraint)
    - J = JL1 * JL0 where JL0 = AH (VN constraint)

    The layout order determines how IVNs are arranged in the streaming buffer.
    """
    order: int  # 3-bit encoding (0-5)
    ML0: int    # Always = AH for VN constraint
    ML1: int    # M / AH
    JL0: int    # Always = AH for VN constraint
    JL1: int    # J / AH

    def __post_init__(self):
        assert 0 <= self.order <= 5, "Order must be 0-5"


@dataclass
class SetWVNLayout:
    """
    Configure the stationary buffer layout for weight VNs.

    For weight matrices of shape (K, N), we partition:
    - K = KL1 * KL0 where KL0 = AH (VN constraint)
    - N = NL1 * NL0

    The layout order determines how WVNs are arranged in the stationary buffer.
    """
    order: int  # 3-bit encoding (0-5)
    KL0: int    # Always = AH for VN constraint
    KL1: int    # K / AH
    NL0: int    # Inner non-reduction factor (1 <= NL0 <= AW)
    NL1: int    # Outer non-reduction factor

    def __post_init__(self):
        assert 0 <= self.order <= 5, "Order must be 0-5"
        assert self.KL0 > 0 and self.KL1 > 0
        assert self.NL0 > 0 and self.NL1 > 0


@dataclass
class SetOVNLayout:
    """
    Configure the output buffer layout for output VNs.

    For output matrices of shape (P, Q), we partition:
    - P = PL1 * PL0 where PL0 = AH (VN constraint)
    - Q = QL1 * QL0 where QL0 = AH (VN constraint)

    The layout order determines how OVNs are arranged in the output buffer.
    """
    order: int  # 3-bit encoding (0-5)
    PL0: int    # Always = AH for VN constraint
    PL1: int    # P / AH
    QL0: int    # Always = AH for VN constraint
    QL1: int    # Q / AH

    def __post_init__(self):
        assert 0 <= self.order <= 5, "Order must be 0-5"


@dataclass
class SetMapping:
    """
    Configure VN-level mapping parameters and trigger execution.

    This defines the parametric mapping of WVNs to PEs:
    - r(ah, aw) = r0 + floor(aw / Gr)
    - c(ah, aw) = c0 + sr*ah + sc*(aw mod Gc)

    where (r, c) are the WVN row and column indices.
    """
    r0: int     # Base WVN row index
    c0: int     # Base WVN column index
    Gr: int     # Row replication group size
    Gc: int     # Column replication group size
    sr: int     # Temporal stride (across PE rows)
    sc: int     # Spatial stride (across PE columns)

    def compute_wvn_indices(self, AH: int, AW: int) -> np.ndarray:
        """
        Compute the WVN (r, c) indices for each PE (ah, aw).
        Returns array of shape (AH, AW, 2) where last dim is (r, c).
        """
        indices = np.zeros((AH, AW, 2), dtype=np.int32)
        for ah in range(AH):
            for aw in range(AW):
                r = self.r0 + (aw // self.Gr)
                c = self.c0 + self.sr * ah + self.sc * (aw % self.Gc)
                indices[ah, aw, 0] = r
                indices[ah, aw, 1] = c
        return indices


class FEATHER_ISA:
    """
    FEATHER-ISA: A minimal ISA-based implementation of FEATHER+ accelerator.

    This class provides a functional model of FEATHER+ that can be programmed
    using MINISA instructions at Virtual Neuron granularity.
    """

    def __init__(self, AH: int, AW: int, dtype: AlloType = int8):
        """
        Initialize FEATHER-ISA accelerator.

        Args:
            AH: PE array height (number of elements in a VN)
            AW: PE array width
            dtype: Data type for computations
        """
        self.AH = AH
        self.AW = AW
        self.dtype = dtype

        # Current layout configurations
        self.ivn_layout: Optional[SetIVNLayout] = None
        self.wvn_layout: Optional[SetWVNLayout] = None
        self.ovn_layout: Optional[SetOVNLayout] = None

        # On-chip buffers (simplified model)
        self.streaming_buffer = None  # For inputs
        self.stationary_buffer = None  # For weights
        self.output_buffer = None  # For outputs

    def set_ivn_layout(self, layout: SetIVNLayout):
        """Set input VN layout and load data into streaming buffer."""
        self.ivn_layout = layout
        print(f"[SetIVNLayout] Order={layout.order}, ML1={layout.ML1}, JL1={layout.JL1}")

    def set_wvn_layout(self, layout: SetWVNLayout):
        """Set weight VN layout and load data into stationary buffer."""
        self.wvn_layout = layout
        print(f"[SetWVNLayout] Order={layout.order}, KL1={layout.KL1}, NL0={layout.NL0}, NL1={layout.NL1}")

    def set_ovn_layout(self, layout: SetOVNLayout):
        """Set output VN layout and initialize output buffer."""
        self.ovn_layout = layout
        print(f"[SetOVNLayout] Order={layout.order}, PL1={layout.PL1}, QL1={layout.QL1}")

    def execute_mapping(self, mapping: SetMapping,
                       input_data: np.ndarray,
                       weight_data: np.ndarray) -> np.ndarray:
        """
        Execute a compute tile with the given VN-level mapping.

        This performs the core computation: output = input @ weight
        at VN granularity using the parametric mapping.

        Args:
            mapping: SetMapping instruction with mapping parameters
            input_data: Input matrix (M, K) organized as VNs
            weight_data: Weight matrix (K, N) organized as VNs

        Returns:
            Output matrix (M, N) organized as VNs
        """
        print(f"[SetMapping] r0={mapping.r0}, c0={mapping.c0}, Gr={mapping.Gr}, Gc={mapping.Gc}, sr={mapping.sr}, sc={mapping.sc}")

        # For simplicity, we'll compute the standard matrix multiplication
        # In a real ISA implementation, this would route VNs to PEs according
        # to the mapping parameters
        M, K = input_data.shape
        K2, N = weight_data.shape
        assert K == K2, f"Dimension mismatch: {K} != {K2}"

        # Compute output
        # Convert Allo type to numpy type
        if hasattr(self.dtype, 'to_numpy'):
            np_dtype = self.dtype.to_numpy()
        else:
            # Map Allo types to numpy types
            if self.dtype.bits == 8:
                np_dtype = np.int8
            elif self.dtype.bits == 16:
                np_dtype = np.int16
            elif self.dtype.bits == 32:
                np_dtype = np.int32
            else:
                np_dtype = np.int32

        output = np.zeros((M, N), dtype=np_dtype)

        # In VN-level execution, we process AH-element dot products
        # This is a functional model - a real implementation would
        # map these computations to the PE array
        output = input_data @ weight_data

        return output.astype(np_dtype)

    def create_gemm_kernel(self):
        """
        Create an Allo kernel for GEMM using FEATHER ISA approach.
        This is a simplified version that demonstrates VN-level computation.
        """
        AH, AW = self.AH, self.AW
        Ty = self.dtype

        def gemm_vn_kernel(A: Ty[AH, AW], B: Ty[AH, AW, AH], C: Ty[AH, AW]):
            """
            Compute one VN-level tile: C += A @ B

            This kernel performs VN-level dot products:
            - A: Input VNs of size (AH, AW)
            - B: Weight VNs of size (AH, AW, AH)
            - C: Output VNs of size (AH, AW)

            Each PE computes an AH-way dot product.
            """
            # Iterate over PE rows
            for i in range(AH):
                # Iterate over PE columns
                for j in range(AW):
                    temp: Ty = 0
                    # Perform AH-way dot product (one VN)
                    for k in range(AH):
                        temp += A[k, j] * B[i, j, k]
                    C[i, j] = temp

        return gemm_vn_kernel


def create_minisa_program_gemm(M: int, N: int, K: int, AH: int, AW: int):
    """
    Create a MINISA program for matrix multiplication.

    This function generates the MINISA instruction sequence for computing
    C = A @ B where A is (M, K), B is (K, N), and C is (M, N).

    Returns:
        Tuple of (SetIVNLayout, SetWVNLayout, SetOVNLayout, List[SetMapping])
    """
    # Ensure dimensions are multiples of AH for VN partitioning
    assert M % AH == 0, f"M={M} must be divisible by AH={AH}"
    assert K % AH == 0, f"K={K} must be divisible by AH={AH}"
    assert N % AH == 0, f"N={N} must be divisible by AH={AH}"

    # Partition dimensions for VN layout
    ML0, ML1 = AH, M // AH
    JL0, JL1 = AH, K // AH
    KL0, KL1 = AH, K // AH
    NL0, NL1 = min(AW, N), N // min(AW, N)
    PL0, PL1 = AH, M // AH
    QL0, QL1 = AH, N // AH

    # Create layout instructions (using simple order 0 for now)
    ivn_layout = SetIVNLayout(
        order=0,  # mL1 → mL0 → jL1
        ML0=ML0, ML1=ML1,
        JL0=JL0, JL1=JL1
    )

    wvn_layout = SetWVNLayout(
        order=0,  # kL1 → nL0 → nL1
        KL0=KL0, KL1=KL1,
        NL0=NL0, NL1=NL1
    )

    ovn_layout = SetOVNLayout(
        order=0,  # pL1 → pL0 → qL1
        PL0=PL0, PL1=PL1,
        QL0=QL0, QL1=QL1
    )

    # Create mapping instructions for tiled execution
    # For simplicity, create one mapping per tile
    mappings = []

    # Tile the computation
    num_m_tiles = M // AH
    num_n_tiles = N // AW if N >= AW else 1
    num_k_tiles = K // AH

    for m_tile in range(num_m_tiles):
        for n_tile in range(min(num_n_tiles, 1)):  # Simplified: one mapping
            # Simple mapping: each PE column gets consecutive WVN columns
            mapping = SetMapping(
                r0=0,          # Start from WVN row 0
                c0=n_tile * AW,  # Offset by tile
                Gr=1,          # No row replication
                Gc=AW,         # Full column width replication group
                sr=0,          # No temporal stride
                sc=1           # Sequential spatial stride
            )
            mappings.append(mapping)

    return ivn_layout, wvn_layout, ovn_layout, mappings


def print_minisa_program(ivn_layout, wvn_layout, ovn_layout, mappings):
    """Print MINISA program in readable format."""
    print("=" * 80)
    print("MINISA Program")
    print("=" * 80)
    print(f"1. SetIVNLayout(order={ivn_layout.order}, ML1={ivn_layout.ML1}, JL1={ivn_layout.JL1})")
    print(f"2. SetWVNLayout(order={wvn_layout.order}, KL1={wvn_layout.KL1}, NL0={wvn_layout.NL0}, NL1={wvn_layout.NL1})")
    print(f"3. SetOVNLayout(order={ovn_layout.order}, PL1={ovn_layout.PL1}, QL1={ovn_layout.QL1})")
    print(f"4. SetMapping instructions (count={len(mappings)}):")
    for i, m in enumerate(mappings):
        print(f"   [{i}] SetMapping(r0={m.r0}, c0={m.c0}, Gr={m.Gr}, Gc={m.Gc}, sr={m.sr}, sc={m.sc})")
    print("=" * 80)
