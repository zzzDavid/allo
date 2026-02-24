# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""MINISA Instruction Set Architecture definitions.

MINISA (Minimal ISA) provides a VN-level (Virtual Neuron) programming interface
for the FEATHER+ accelerator. It abstracts away low-level PE array details
while enabling efficient dataflow configuration.

The ISA consists of four instruction types:
- SetIVNLayout: Configure input VN buffer layout
- SetWVNLayout: Configure weight VN buffer layout
- SetOVNLayout: Configure output VN buffer layout
- SetMapping: Trigger tile execution with VN mapping

These instructions are lowered to Allo configuration tensors that drive
the FEATHER+ dataflow hardware.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Tuple, Union

import numpy as np


# Instruction encoding constants
NUM_FIELDS = 13  # Fields per encoded instruction row
INST_TYPE_IVN = 0
INST_TYPE_WVN = 1
INST_TYPE_OVN = 2
INST_TYPE_MAPPING = 3


class LayoutOrder(IntEnum):
    """Layout dimension ordering for VN buffers.

    3-bit encoding supports 6 orderings for 3 logical dimensions.
    The order determines how data is arranged in memory and
    accessed by the PE array.
    """
    ORDER_012 = 0  # Default order
    ORDER_021 = 1
    ORDER_102 = 2
    ORDER_120 = 3
    ORDER_201 = 4
    ORDER_210 = 5


@dataclass
class SetIVNLayout:
    """Set Input Virtual Neuron layout configuration.

    Configures how input activations are laid out in the streaming buffer.
    The IVN (Input VN) represents AH consecutive input elements that form
    the input to a single VN dot product.

    Attributes:
        order: Dimension ordering (3-bit encoding)
        ML0: Inner M factor - always equals AH (VN constraint)
        ML1: Outer M factor - M / AH tiles in M dimension
        JL0: Inner J factor - always equals AH (VN constraint)
        JL1: Outer J factor - J / AH tiles in J (reduction) dimension

    The layout configuration generates:
    - Input buffer addressing patterns
    - Crossbar select signals for input distribution
    """
    order: int = 0
    ML0: int = 8   # Inner M (always = AH)
    ML1: int = 1   # Outer M = M / AH
    JL0: int = 8   # Inner J (always = AH)
    JL1: int = 1   # Outer J = J / AH

    def validate(self, AH: int) -> bool:
        """Validate layout against hardware constraints."""
        if self.ML0 != AH:
            return False
        if self.JL0 != AH:
            return False
        if self.order < 0 or self.order > 5:
            return False
        return True

    def total_m(self) -> int:
        """Total M dimension size."""
        return self.ML0 * self.ML1

    def total_j(self) -> int:
        """Total J (reduction) dimension size."""
        return self.JL0 * self.JL1


@dataclass
class SetWVNLayout:
    """Set Weight Virtual Neuron layout configuration.

    Configures how weights are laid out in the stationary buffer.
    The WVN (Weight VN) represents AH consecutive weight elements that
    multiply with an IVN to produce partial sums.

    Attributes:
        order: Dimension ordering (3-bit encoding)
        KL0: Inner K factor - always equals AH (VN constraint)
        KL1: Outer K factor - K / AH tiles in K dimension
        NL0: Inner N factor - 1 <= NL0 <= AW (flexible)
        NL1: Outer N factor - N / NL0 tiles in N dimension

    The layout configuration generates:
    - Weight buffer addressing patterns
    - Crossbar select signals for weight distribution
    """
    order: int = 0
    KL0: int = 8   # Inner K (always = AH)
    KL1: int = 1   # Outer K = K / AH
    NL0: int = 8   # Inner N (1 <= NL0 <= AW)
    NL1: int = 1   # Outer N = N / NL0

    def validate(self, AH: int, AW: int) -> bool:
        """Validate layout against hardware constraints."""
        if self.KL0 != AH:
            return False
        if self.NL0 < 1 or self.NL0 > AW:
            return False
        if self.order < 0 or self.order > 5:
            return False
        return True

    def total_k(self) -> int:
        """Total K dimension size."""
        return self.KL0 * self.KL1

    def total_n(self) -> int:
        """Total N dimension size."""
        return self.NL0 * self.NL1


@dataclass
class SetOVNLayout:
    """Set Output Virtual Neuron layout configuration.

    Configures how outputs are laid out and how reduction/reordering
    is performed in the BIRRD network.

    Attributes:
        order: Dimension ordering (3-bit encoding)
        PL0: Inner P factor - always equals AH (VN constraint)
        PL1: Outer P factor - P / AH tiles in P (output rows)
        QL0: Inner Q factor - always equals AH (VN constraint)
        QL1: Outer Q factor - Q / AH tiles in Q (output cols)

    The layout configuration generates:
    - BIRRD instruction arrays for reduction/reordering
    - Output buffer addressing patterns
    """
    order: int = 0
    PL0: int = 8   # Inner P (always = AH)
    PL1: int = 1   # Outer P = P / AH
    QL0: int = 8   # Inner Q (always = AH)
    QL1: int = 1   # Outer Q = Q / AH

    def validate(self, AH: int) -> bool:
        """Validate layout against hardware constraints."""
        if self.PL0 != AH:
            return False
        if self.QL0 != AH:
            return False
        if self.order < 0 or self.order > 5:
            return False
        return True

    def total_p(self) -> int:
        """Total P (output rows) dimension size."""
        return self.PL0 * self.PL1

    def total_q(self) -> int:
        """Total Q (output cols) dimension size."""
        return self.QL0 * self.QL1


@dataclass
class SetMapping:
    """Set VN-level mapping and trigger tile execution.

    Specifies how Virtual Neurons map to the physical PE array and
    triggers execution of a tile. Each SetMapping instruction results
    in one invocation of the Allo dataflow region.

    Attributes:
        r0: Base WVN row index
        c0: Base WVN column index
        Gr: Replication group size for rows
        Gc: Replication group size for columns
        sr: Temporal stride (across rows)
        sc: Spatial stride (across columns)

    Mapping formula for PE at position (ah, aw):
        r(ah, aw) = r0 + floor(aw / Gr)
        c(ah, aw) = c0 + sr * ah + sc * (aw mod Gc)

    Common mappings:
    - Output stationary: Gr=AW, Gc=1, sr=0, sc=0
    - Weight stationary: Gr=1, Gc=AW, sr=0, sc=1
    - Input stationary: Gr=1, Gc=1, sr=1, sc=0
    """
    r0: int = 0    # Base WVN row
    c0: int = 0    # Base WVN column
    Gr: int = 8    # Row replication group
    Gc: int = 1    # Column replication group
    sr: int = 0    # Temporal stride
    sc: int = 0    # Spatial stride

    # Optional: tile bounds for slicing input/weight tensors
    m_start: int = 0
    m_end: int = 0
    n_start: int = 0
    n_end: int = 0
    k_start: int = 0
    k_end: int = 0

    def validate(self, AH: int, AW: int) -> bool:
        """Validate mapping against hardware constraints."""
        if self.Gr < 1 or self.Gr > AW:
            return False
        if self.Gc < 1 or self.Gc > AW:
            return False
        return True

    def get_pe_mapping(self, ah: int, aw: int) -> Tuple[int, int]:
        """Compute WVN indices for PE at position (ah, aw).

        Returns:
            (r, c): WVN row and column indices
        """
        r = self.r0 + (aw // self.Gr)
        c = self.c0 + self.sr * ah + self.sc * (aw % self.Gc)
        return (r, c)


@dataclass
class MINISAProgram:
    """A complete MINISA program for a workload.

    Contains layout configurations and a sequence of mapping operations.
    The program is encoded and passed to the FEATHER+ dataflow region,
    which decodes instructions on-chip and executes the full matrix
    computation in a single invocation.

    Attributes:
        name: Program name/identifier
        AH: Hardware array height
        AW: Hardware array width
        ivn_layout: Input VN layout configuration
        wvn_layout: Weight VN layout configuration
        ovn_layout: Output VN layout configuration
        mappings: Sequence of tile execution operations
    """
    name: str = "minisa_program"
    AH: int = 8
    AW: int = 8
    ivn_layout: Optional[SetIVNLayout] = None
    wvn_layout: Optional[SetWVNLayout] = None
    ovn_layout: Optional[SetOVNLayout] = None
    mappings: List[SetMapping] = field(default_factory=list)

    def __post_init__(self):
        """Initialize default layouts if not provided."""
        if self.ivn_layout is None:
            self.ivn_layout = SetIVNLayout(ML0=self.AH, JL0=self.AH)
        if self.wvn_layout is None:
            self.wvn_layout = SetWVNLayout(KL0=self.AH, NL0=self.AW)
        if self.ovn_layout is None:
            self.ovn_layout = SetOVNLayout(PL0=self.AH, QL0=self.AH)

    def add_mapping(self, mapping: SetMapping):
        """Add a tile mapping operation to the program."""
        self.mappings.append(mapping)

    def validate(self) -> bool:
        """Validate all instructions against hardware constraints."""
        if not self.ivn_layout.validate(self.AH):
            return False
        if not self.wvn_layout.validate(self.AH, self.AW):
            return False
        if not self.ovn_layout.validate(self.AH):
            return False
        for mapping in self.mappings:
            if not mapping.validate(self.AH, self.AW):
                return False
        return True

    def num_tiles(self) -> int:
        """Return number of tile operations in the program."""
        return len(self.mappings)


def create_gemm_program(
    M: int, N: int, K: int,
    AH: int = 8, AW: int = 8,
    name: str = "gemm",
    ivn_order: int = 0,
    wvn_order: int = 0,
    ovn_order: int = 0,
    dataflow: str = "output_stationary",
) -> MINISAProgram:
    """Create a MINISA program for GEMM: C[M,N] = A[M,K] * B[K,N].

    This generates the layout configurations and tile mappings needed
    to execute a matrix multiplication on FEATHER+.

    Args:
        M: Number of rows in A and C
        N: Number of columns in B and C
        K: Shared dimension (cols of A, rows of B)
        AH: Hardware array height
        AW: Hardware array width
        name: Program name
        ivn_order: Input VN layout order (0-5)
        wvn_order: Weight VN layout order (0-5)
        ovn_order: Output VN layout order (0-5)
        dataflow: PE mapping strategy:
            "output_stationary" (default): Gr=AW, Gc=1, sr=0, sc=0
            "weight_stationary": Gr=1, Gc=AW, sr=0, sc=1

    Returns:
        MINISAProgram configured for GEMM
    """
    # Tile sizes based on hardware dimensions
    Mt = AW // 2   # M tile size
    Nt = AH        # N tile size
    Kt = 2 * AH    # K tile size (reduction dimension)

    # Ensure dimensions are tileable
    assert M % Mt == 0, f"M={M} must be divisible by Mt={Mt}"
    assert N % Nt == 0, f"N={N} must be divisible by Nt={Nt}"
    assert K % Kt == 0, f"K={K} must be divisible by Kt={Kt}"

    # PE mapping based on dataflow strategy
    if dataflow == "output_stationary":
        Gr, Gc, sr, sc = AW, 1, 0, 0
    elif dataflow == "weight_stationary":
        Gr, Gc, sr, sc = 1, AW, 0, 1
    else:
        raise ValueError(f"Unsupported dataflow: {dataflow}. "
                        f"Use 'output_stationary' or 'weight_stationary'.")

    # Create program with layouts
    program = MINISAProgram(
        name=name,
        AH=AH,
        AW=AW,
        ivn_layout=SetIVNLayout(
            order=ivn_order,
            ML0=AH,
            ML1=M // AH,
            JL0=AH,
            JL1=K // AH,
        ),
        wvn_layout=SetWVNLayout(
            order=wvn_order,
            KL0=AH,
            KL1=K // AH,
            NL0=min(N, AW),
            NL1=max(1, N // AW),
        ),
        ovn_layout=SetOVNLayout(
            order=ovn_order,
            PL0=AH,
            PL1=M // AH,
            QL0=AH,
            QL1=N // AH,
        ),
    )

    # Generate tile mappings
    # Loop order: N tiles -> M tiles -> K tiles (reduction)
    for n_tile in range(N // Nt):
        for m_tile in range(M // Mt):
            for k_tile in range(K // Kt):
                mapping = SetMapping(
                    r0=k_tile * Kt // AH,
                    c0=n_tile * Nt,
                    Gr=Gr,
                    Gc=Gc,
                    sr=sr,
                    sc=sc,
                    m_start=m_tile * Mt,
                    m_end=(m_tile + 1) * Mt,
                    n_start=n_tile * Nt,
                    n_end=(n_tile + 1) * Nt,
                    k_start=k_tile * Kt,
                    k_end=(k_tile + 1) * Kt,
                )
                program.add_mapping(mapping)

    return program


# Type alias for MINISA instruction union
MINISAInstruction = Union[SetIVNLayout, SetWVNLayout, SetOVNLayout, SetMapping]


def encode_program(program: MINISAProgram) -> np.ndarray:
    """Encode a MINISA program as a flat int32 array for Allo execution.

    Converts the program's layout instructions and tile mappings into a
    fixed-width integer array that can be passed to the full-matrix Allo
    function for on-chip instruction decode.

    Encoding format (NUM_FIELDS=13 columns per row):
        SetIVNLayout:  [0, order, ML0, ML1, JL0, JL1, 0, 0, 0, 0, 0, 0, 0]
        SetWVNLayout:  [1, order, KL0, KL1, NL0, NL1, 0, 0, 0, 0, 0, 0, 0]
        SetOVNLayout:  [2, order, PL0, PL1, QL0, QL1, 0, 0, 0, 0, 0, 0, 0]
        SetMapping:    [3, r0, c0, Gr, Gc, sr, sc, m_start, m_end, n_start, n_end, k_start, k_end]

    Args:
        program: MINISA program to encode

    Returns:
        int32 array of shape [num_inst, NUM_FIELDS]
    """
    num_inst = 3 + len(program.mappings)  # 3 layout instructions + mappings
    inst = np.zeros((num_inst, NUM_FIELDS), dtype=np.int32)

    # Encode SetIVNLayout
    ivn = program.ivn_layout
    inst[0] = [INST_TYPE_IVN, ivn.order, ivn.ML0, ivn.ML1, ivn.JL0, ivn.JL1,
               0, 0, 0, 0, 0, 0, 0]

    # Encode SetWVNLayout
    wvn = program.wvn_layout
    inst[1] = [INST_TYPE_WVN, wvn.order, wvn.KL0, wvn.KL1, wvn.NL0, wvn.NL1,
               0, 0, 0, 0, 0, 0, 0]

    # Encode SetOVNLayout
    ovn = program.ovn_layout
    inst[2] = [INST_TYPE_OVN, ovn.order, ovn.PL0, ovn.PL1, ovn.QL0, ovn.QL1,
               0, 0, 0, 0, 0, 0, 0]

    # Encode SetMapping instructions
    for i, mapping in enumerate(program.mappings):
        inst[3 + i] = [INST_TYPE_MAPPING, mapping.r0, mapping.c0,
                        mapping.Gr, mapping.Gc, mapping.sr, mapping.sc,
                        mapping.m_start, mapping.m_end,
                        mapping.n_start, mapping.n_end,
                        mapping.k_start, mapping.k_end]

    return inst
