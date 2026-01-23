# MINISA Verification Report

## Overview

This report documents the verification of the MINISA (Minimal Instruction Set Architecture) implementation for FEATHER+ accelerator programming.

## Implementation Summary

### Core Components

| Component | File | Status | Description |
|-----------|------|--------|-------------|
| ISA Instructions | `minisa/isa.py` | Complete | SetIVNLayout, SetWVNLayout, SetOVNLayout, SetMapping |
| Layout Descriptors | `minisa/layout.py` | Complete | VNLayout base class with IVN, WVN, OVN specializations |
| Interpreter | `minisa/interpreter.py` | Complete | Software execution model with state machine |
| Lowering Layer | `minisa/lowering.py` | Complete | MINISA to FEATHER control translation |

### Instruction Encoding

| Instruction | Opcode | Parameters | Bits |
|------------|--------|------------|------|
| SetIVNLayout | 0b01 | order(3), M_L0, M_L1, J_L1 | Variable |
| SetWVNLayout | 0b10 | order(3), N_L0, N_L1, K_L1 | Variable |
| SetOVNLayout | 0b10 | order(3), P_L0, P_L1, Q_L1, clear | Variable |
| SetMapping | 0b11 | r0, c0, G_r, G_c, s_r, s_c | Variable |

## Test Results

### Single-Tile GEMM Tests

| Test | Array Size | GEMM Size | Status |
|------|-----------|-----------|--------|
| test_gemm_4x4x4_single_tile | 4x4 | 4x4x4 | PASS |
| test_gemm_8x8x8_single_tile | 8x8 | 8x8x8 | PASS |

### Multi-Tile GEMM Tests

| Test | Array Size | GEMM Size | Tiles | Status |
|------|-----------|-----------|-------|--------|
| test_gemm_8x16x8_two_k_tiles | 8x8 | 8x16x8 | 2 (K) | PASS |
| test_gemm_16x8x16_spatial_tiling | 8x8 | 16x8x16 | 4 (M,N) | PASS |

### Layout Variation Tests

| Test | Description | Status |
|------|-------------|--------|
| test_wvn_layout_orders | All 6 permutation orders | PASS |
| test_gemm_with_different_wvn_orders | Same result across orders | PASS |

### Multi-Layer Tests

| Test | Layers | Dimensions | Status |
|------|--------|------------|--------|
| test_two_layer_same_size | 2 | 4x4x4x4 | PASS |
| test_two_layer_different_sizes | 2 | 4x8x4x8 | PASS |
| test_three_layer_mlp | 3 | 4x8x4x4x8 | PASS |
| test_layout_order_switch | 2 | Layout change | PASS |

## Verification Methodology

### Correctness Verification

1. **Reference Implementation**: All GEMM computations verified against `numpy.matmul`
2. **Integer Precision**: Tests use int8 inputs with int32 accumulation
3. **Dimension Checks**: Output shape verification for all tests

### Coverage Areas

1. **VN Layout Addressing**
   - Verified bijective mapping (each VN maps to unique buffer location)
   - Tested all 6 layout order permutations
   - Confirmed inverse mapping (buffer_addr_to_vn) correctness

2. **PE-to-VN Mapping**
   - Verified SetMapping formula: `r(ah,aw) = r0 + floor(aw/G_r)`
   - Verified column index: `c(ah,aw) = c0 + s_r*ah + s_c*(aw mod G_c)`
   - Tested reduction group identification (columns with same WVN row)

3. **BIRRD Configuration**
   - Verified instruction array generation for AW=4, 8, 16
   - Tested 2:1 and 4:1 reduction patterns
   - Confirmed correct stage count: P0 = 2*log2(AW) for AW>4

4. **Interpreter State Machine**
   - Verified layout instruction execution updates state
   - Verified mapping instruction triggers tile computation
   - Confirmed accumulation across K-dimension tiles

## Hardware Constraint Verification

### VN Constraints (Paper Section IV-A)

| Constraint | Implementation | Verified |
|------------|---------------|----------|
| K_L0 = AH for WVN | `load_weights` reshapes to (K/AH, N, AH) | Yes |
| J_L0 = AH for IVN | `_get_ivn` extracts AH elements | Yes |
| Q_L0 = AH for OVN | Output VN aligns with dot-product | Yes |

### Buffer Capacity Constraints

| Buffer | Constraint | Implementation |
|--------|------------|---------------|
| Stationary | N_L1 * K_L1 <= D_sta / N_L0 | Validated in SetWVNLayout |
| Streaming | M_L1 * J_L1 <= D_str / M_L0 | Validated in SetIVNLayout |

### Mapping Constraints

| Constraint | Implementation | Notes |
|------------|---------------|-------|
| 1 <= G_r <= AW | Validated in SetMapping | Reduction group size |
| 1 <= G_c <= AW | Validated in SetMapping | Replication group size |
| Non-overlapping VN access | s_r, s_c parameters | Spatial/temporal stride |

## Known Limitations

1. **BIRRD Generic Reduction**: The `_generate_generic_reduction` method returns pass-through; full butterfly routing not yet implemented for arbitrary patterns.

2. **Hardware Integration**: Current implementation is software-only interpreter; Allo dataflow integration pending.

3. **Quantization**: Tests use int8 inputs; quantization-aware training integration not implemented.

## Acceptance Criteria Status

| Criterion | Status | Evidence |
|-----------|--------|----------|
| GEMM correctness vs numpy | PASS | test_minisa_gemm.py |
| Multiple layout/mapping correctness | PASS | test_minisa_gemm.py |
| Layout switching across layers | PASS | test_minisa_multilayer.py |
| Instruction count tracking | PASS | TestMINISAInstructionCounts |

## Conclusion

The MINISA implementation correctly implements the instruction set architecture as specified in the MINISA paper. All core functionality has been verified:

- **4 instructions** properly defined with validation
- **6 layout orders** correctly encode rank permutations
- **PE-to-VN mapping** formula implemented per paper Section IV-B
- **Multi-tile execution** with K-dimension accumulation verified
- **Layout switching** between consecutive layers functional

The implementation provides a complete software model for MINISA program execution and is ready for integration with Allo dataflow hardware generation.
