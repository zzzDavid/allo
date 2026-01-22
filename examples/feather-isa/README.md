# FEATHER-ISA: Minimal Instruction Set Architecture Implementation

This directory contains an implementation of **MINISA** (Minimal Instruction Set Architecture) for the FEATHER+ reconfigurable AI accelerator, implemented using the Allo accelerator design language.

## Overview

MINISA is a minimal instruction set that programs FEATHER+ at the **Virtual Neuron (VN)** granularity—the smallest hardware dot-product atom. This implementation demonstrates how to achieve dramatic reductions in control overhead while preserving the full flexibility of reconfigurable dataflow execution.

### What is MINISA?

MINISA is based on the paper:
> "MINISA: Minimal Instruction Set Architecture for Next-gen Reconfigurable Inference Accelerator"

Key insights:
- **Virtual Neuron (VN)**: AH-element dot product (smallest hardware compute atom)
- **Programming at VN granularity** is the coarsest control that preserves inter-PE mapping flexibility
- **4 instructions** are sufficient to express arbitrary dataflows and layouts
- **Up to 99.4× speedup** by eliminating instruction-fetch stalls

### Why MINISA?

Traditional reconfigurable accelerators suffer from control overhead:
- Fine-grained micro-instructions configure every switch and cycle
- Instruction footprint grows with array size
- Instruction fetch can dominate execution time (up to 98% stall cycles)

MINISA solves this by:
- Abstracting control to VN level
- Reducing instruction bytes by 10^5× (geometric mean)
- Enabling larger on-chip tiles and higher arithmetic intensity

## Architecture

### FEATHER+ Components

1. **NEST**: AH×AW PE array
   - Each PE performs AH-way dot products
   - One PE = one VN computation

2. **Three-level reduction**:
   - Temporal reduction within PE (AH-way dot product)
   - Spatial reduction over PE rows (via BIRRD network)
   - Temporal reduction over PE columns (in output buffer)

3. **All-to-all distribution**: Flexible routing of VNs to any PE column

### Virtual Neuron (VN) Abstraction

A VN is the smallest operand fragment that matches the hardware atom:
- **IVN** (Input VN): AH consecutive elements from input matrix
- **WVN** (Weight VN): AH consecutive elements from weight matrix
- **OVN** (Output VN): AH accumulated partial sums

## MINISA Instructions

### 1. SetIVNLayout

Configure the streaming buffer layout for input VNs.

**Parameters**:
- `order`: 3-bit encoding (0-5) for loop ordering
- `ML0`, `ML1`: Input matrix M dimension partition (ML0 = AH)
- `JL0`, `JL1`: Input matrix J dimension partition (JL0 = AH)

**Effect**: Defines how IVNs are organized in the streaming buffer.

### 2. SetWVNLayout

Configure the stationary buffer layout for weight VNs.

**Parameters**:
- `order`: 3-bit encoding (0-5) for loop ordering
- `KL0`, `KL1`: Weight matrix K dimension partition (KL0 = AH)
- `NL0`, `NL1`: Weight matrix N dimension partition

**Effect**: Defines how WVNs are organized in the stationary buffer.

### 3. SetOVNLayout

Configure the output buffer layout for output VNs.

**Parameters**:
- `order`: 3-bit encoding (0-5) for loop ordering
- `PL0`, `PL1`: Output matrix P dimension partition (PL0 = AH)
- `QL0`, `QL1`: Output matrix Q dimension partition (QL0 = AH)

**Effect**: Initializes output buffer and defines OVN organization.

### 4. SetMapping

Execute a compute tile with parametric VN-level mapping.

**Parameters**:
- `r0`, `c0`: Base WVN row/column indices
- `Gr`, `Gc`: Replication group sizes
- `sr`, `sc`: Temporal/spatial strides

**Effect**: Maps WVNs to PEs and triggers execution:
```
r(ah, aw) = r0 + floor(aw / Gr)
c(ah, aw) = c0 + sr*ah + sc*(aw mod Gc)
```

## Files

- `feather_isa.py`: Core MINISA implementation
  - `FEATHER_ISA` class: Functional model of FEATHER+ accelerator
  - Instruction dataclasses: `SetIVNLayout`, `SetWVNLayout`, `SetOVNLayout`, `SetMapping`
  - Helper functions for creating MINISA programs

- `feather_isa_hardware.py`: Complete hardware implementation with Allo
  - `create_feather_isa()`: Full FEATHER+ dataflow graph with NEST and BIRRD
  - `create_simplified_gemm_feather()`: VN-level GEMM kernel for testing
  - `generate_birrd_config()`: BIRRD configuration from MINISA instructions
  - Hardware components: NEST, BIRRD, Output Buffer, Instruction Config

- `test_gemm_minisa.py`: Test suite demonstrating MINISA programming
  - Matrix multiplication using MINISA instructions
  - VN-level Allo kernel implementation
  - Verification against reference

- `test_feather_hardware_integration.py`: Integration test for hardware + MINISA
  - End-to-end test combining MINISA instructions with hardware execution
  - Verifies functional model and hardware implementation match
  - Tests multiple PE array configurations

- `example_minisa_program.py`: Educational examples
  - 4 detailed examples demonstrating VN-level programming concepts
  - Instruction footprint analysis
  - Mapping flexibility demonstration

- `MINISA.pdf`: Original paper specification
- `feather.pdf`: FEATHER architecture documentation

## Usage

### Basic Test

Run the test suite with default parameters:

```bash
conda activate py312
cd /home/nz264/shared/allo/examples/feather-isa
python test_gemm_minisa.py
```

### Custom Matrix Dimensions

Test with specific dimensions (must be multiples of AH):

```bash
python test_gemm_minisa.py --M 16 --N 16 --K 16 --AH 8 --AW 8 --verbose
```

### Example MINISA Program

Here's a complete example for matrix multiplication C = A @ B:

```python
from feather_isa import (
    FEATHER_ISA,
    SetIVNLayout,
    SetWVNLayout,
    SetOVNLayout,
    SetMapping
)

# Initialize accelerator
AH, AW = 4, 4
feather = FEATHER_ISA(AH, AW)

# Step 1: Configure input VN layout
ivn_layout = SetIVNLayout(
    order=0,  # Loop order: mL1 → mL0 → jL1
    ML0=AH, ML1=M//AH,
    JL0=AH, JL1=K//AH
)
feather.set_ivn_layout(ivn_layout)

# Step 2: Configure weight VN layout
wvn_layout = SetWVNLayout(
    order=0,  # Loop order: kL1 → nL0 → nL1
    KL0=AH, KL1=K//AH,
    NL0=AW, NL1=N//AW
)
feather.set_wvn_layout(wvn_layout)

# Step 3: Configure output VN layout
ovn_layout = SetOVNLayout(
    order=0,  # Loop order: pL1 → pL0 → qL1
    PL0=AH, PL1=M//AH,
    QL0=AH, QL1=N//AH
)
feather.set_ovn_layout(ovn_layout)

# Step 4: Execute compute tiles
mapping = SetMapping(
    r0=0, c0=0,  # Start from WVN(0,0)
    Gr=1, Gc=AW,  # Replication groups
    sr=0, sc=1    # Strides
)
output = feather.execute_mapping(mapping, input_data, weight_data)
```

## Hardware Implementation

### FEATHER+ Architecture in Allo

The hardware implementation (`feather_isa_hardware.py`) uses Allo's dataflow API to create a complete FEATHER+ accelerator:

#### NEST (Neural Engine with Spatial forwarding and Temporal reduction)
```python
@df.kernel(mapping=[1])
def NEST(input_acts: Ty[AH, AW], weights: Ty[AH, AW, AH], nest_output: TyAccum[AH, AW]):
    # Phase 1: Local Temporal Reduction (AH-way dot product per PE)
    # Phase 2: Interleaved Spatial Forwarding (to BIRRD)
```

**Features**:
- AH×AW PE array, each PE has AH local registers
- Temporal reduction: AH-way dot product in each PE
- Spatial forwarding: Row-by-row output to BIRRD
- Peak throughput: AH × AH × AW MACs/cycle

#### BIRRD (Butterfly Interconnect for Reduction and Reordering)
```python
@df.kernel(mapping=[NUM_STAGES, SWITCHES_PER_STAGE])
def BIRRD_Switch():
    # 4 operations: PASS (=), SWAP (×), ADD_LEFT (∓), ADD_RIGHT (±)
    # Butterfly bit-reversal for inter-stage connections
```

**Features**:
- Multi-stage butterfly network (2×log₂(AW) stages)
- 4 switch operations: PASS, SWAP, ADD_LEFT, ADD_RIGHT
- RIR capability: Reorder In Reduction (zero-latency layout switching)
- Bit-reversal routing between stages

#### Stream-Based Communication
- `nest_to_birrd`: AW streams from NEST to BIRRD input
- `birrd_connections`: (NUM_STAGES+1) × AW streams for inter-stage data
- `birrd_config`: NUM_STAGES × SWITCHES_PER_STAGE configuration streams

#### VN-Level GEMM Kernel
Simplified implementation for testing:
```python
def gemm_vn_tile(input_vns: int8[AH, AW], weight_vns: int8[AH, AW, AH], output_vns: int32[AH, AW]):
    # Each PE(i, j) computes AH-way dot product
    # Implements NEST Phase 1 behavior
```

## Implementation Notes

### VN Constraint

For all operands, one dimension must be partitioned with size AH:
- Input (M, J): `JL0 = AH` (reduction dimension)
- Weight (K, N): `KL0 = AH` (reduction dimension)
- Output (P, Q): `QL0 = AH` (matches output structure)

### Layout Encoding

The 3-bit `order` field encodes 6 possible loop permutations:
- `000`: Leftmost → Rightmost ordering
- `001-101`: Various permutations
- See MINISA paper Table III for complete encoding

### Mapping Constraints

FEATHER+ enforces reduction-aware mapping:
- All PEs in a column must share the same WVN row index
- This ensures correct temporal local reduction
- Column-wise independence allows arbitrary WVN column replication

## Testing

### Test 1: MINISA Functional Model

Tests the core MINISA programming model:
- Creates MINISA program for GEMM
- Executes all 4 instructions in sequence
- Verifies correctness against NumPy reference

**Status**: ✅ Fully functional

```bash
python test_gemm_minisa.py --M 16 --N 16 --K 16 --AH 4 --AW 4 --verbose
```

### Test 2: Hardware Integration Test

Tests complete FEATHER-ISA with hardware implementation:
- Creates MINISA instruction sequence
- Executes using VN-level hardware kernels
- Verifies functional model and hardware match
- Tests multiple PE array configurations

**Status**: ✅ Functional model verified, hardware execution ready for LLVM

```bash
# Single configuration
python test_feather_hardware_integration.py --M 8 --N 8 --K 8 --AH 4 --AW 4 --verbose

# Multiple configurations
python test_feather_hardware_integration.py --multi
```

### Test 3: Allo VN-level Kernel [Optional]

Tests VN-level computation expressed in Allo:
- Defines VN-level dot product kernel
- Builds for LLVM backend (requires `LLVM_BUILD_DIR`)
- Demonstrates hardware-software co-design

**Status**: ⚠️ Optional (requires LLVM setup)

## Performance Benefits

From the MINISA paper, benefits include:

1. **Instruction Compression**:
   - Micro-instructions: 58× instruction-to-data ratio (128×128 array)
   - MINISA: 0.00013× instruction-to-data ratio
   - **Reduction: ~445,000×**

2. **Stall Elimination**:
   - Micro-instructions: 98% of cycles stalled on instruction fetch
   - MINISA: <0.1% instruction overhead
   - **Speedup: up to 99.4×**

3. **Larger Tiles**:
   - Freed on-chip storage enables bigger compute tiles
   - Higher arithmetic intensity
   - Better end-to-end throughput

## Future Extensions

This implementation can be extended with:

1. **Full FEATHER+ Architecture**:
   - Implement BIRRD spatial reduction network
   - Add ping-pong buffer management
   - Support mixed dataflow/layout execution

2. **More Operators**:
   - Convolution with MINISA
   - Attention mechanisms
   - Fully Homomorphic Encryption (FHE) NTT operations

3. **Hardware Backend**:
   - Generate RTL from Allo/MINISA programs
   - Target FPGAs with Vitis HLS
   - Support AIE tiles for AMD Versal

4. **Compiler Integration**:
   - Automatic MINISA program generation
   - Dataflow/layout co-optimization
   - Tile size selection for target architecture

## References

1. MINISA Paper: "MINISA: Minimal Instruction Set Architecture for Next-gen Reconfigurable Inference Accelerator"

2. FEATHER Paper: "FEATHER: A Reconfigurable Accelerator with Data Reordering Support for Low-Cost On-Chip Dataflow Switching" (ISCA 2024)

3. Allo: Python-based Accelerator Design Language
   - GitHub: https://github.com/cornell-zhang/allo

## Citation

If you use this implementation, please cite:

```bibtex
@article{minisa2024,
  title={MINISA: Minimal Instruction Set Architecture for Next-gen Reconfigurable Inference Accelerator},
  author={...},
  year={2024}
}

@inproceedings{feather2024,
  title={FEATHER: A Reconfigurable Accelerator with Data Reordering Support for Low-Cost On-Chip Dataflow Switching},
  author={Tong, Jianming and others},
  booktitle={ISCA},
  year={2024}
}
```

## License

Copyright Allo authors. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
