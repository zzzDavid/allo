# FEATHER Baseline Verification Report

## Overview

This report documents the baseline verification of the existing FEATHER Allo implementation
before any MINISA changes are applied. The goal is to ensure the baseline FEATHER behavior
remains intact as a regression target.

## FEATHER Architecture Summary

FEATHER (Flexible Engine for Acceleration of Tensors with Hardware Element for Reordering)
is a reconfigurable accelerator that enables low-cost on-chip dataflow switching through:

1. **NEST (Neural Engine with Spatial forwarding and Temporal reduction)**: A 2D PE array
   that performs local temporal reduction within each PE and spatial forwarding between rows.

2. **BIRRD (Butterfly Interconnect for Reduction and Reordering in Dataflows)**: A multi-stage
   reduction network that performs flexible data reduction with layout reordering.

## Allo Implementation Mapping

The Allo implementation in `examples/feather/feather.py` maps FEATHER concepts as follows:

| Paper Concept | Allo Implementation |
|--------------|---------------------|
| NEST (AH x AW) | `@df.kernel(mapping=[1])` with nested loops over AH rows |
| PE dot-product | Inner loop `for k in range(AH)` computing `iAct * weight` |
| Temporal reduction | Accumulation in `temp` variable within PE |
| Spatial forwarding | Data streaming via `nest_out` Stream |
| BIRRD | `@df.kernel(mapping=[P0, P1])` with switch operations |
| EGG switches | Conditional operations: Pass(PS), Add-Right(AR), Add-Left(AL), Swap(SW) |
| Output buffer | `local_output` array collecting BIRRD results |

## Key Parameters

- **AH**: Array height (number of PE rows, also reduction depth)
- **AW**: Array width (number of PE columns, must be power of 2)
- **P0**: Number of BIRRD stages = `2 * log2(AW)` (or `2 * log2(AW) - 1` for AW=4)
- **P1**: Switches per stage = `AW / 2`

## Baseline Tests

### Test 1: feather_basic_4x4
- **Purpose**: Verify basic FEATHER operation with minimal 4x4 array
- **Configuration**: AH=4, AW=4, P0=3, P1=2
- **Expected**: Produces non-zero output with deterministic inputs
- **Status**: PASSED

### Test 2: feather_basic_8x8
- **Purpose**: Verify FEATHER scales to 8x8 array (matching GEMM test)
- **Configuration**: AH=8, AW=8, P0=6, P1=4
- **Expected**: Produces output matching expected dimensions
- **Status**: PASSED

### Test 3: feather_gemm_reference
- **Purpose**: Verify GEMM tiling constraints are satisfied
- **Configuration**: M=8, N=8, K=16 with Mt=4, Nt=8, Kt=16
- **Expected**: Shape constraints (divisibility) satisfied
- **Status**: PASSED

### Test 4: birrd_routing_4x4
- **Purpose**: Verify BIRRD responds to different instruction patterns
- **Configuration**: Test PS, SW, AR instructions
- **Expected**: Different instructions produce different outputs
- **Status**: PASSED

## Verification Results

All baseline tests pass, confirming:

1. The FEATHER dataflow simulator builds and executes correctly
2. NEST produces expected dot-product results
3. BIRRD routing responds to instruction configurations
4. Output shapes match expected dimensions
5. Baseline functionality is stable for MINISA development

## Notes for MINISA Development

The baseline FEATHER has the following characteristics that MINISA must preserve:

1. **Per-tile execution**: Each kernel invocation processes one tile
2. **BIRRD instruction granularity**: One instruction array per tile execution
3. **Layout assumptions**: Weights pre-organized in (AH, AW, AH) format
4. **Reduction pattern**: AH-way temporal reduction followed by BIRRD spatial reduction

MINISA will abstract these details into Virtual Neuron (VN) level operations while
maintaining identical compute semantics.

## Conclusion

The baseline FEATHER implementation is verified and stable. All subsequent MINISA
changes must maintain backward compatibility with these baseline behaviors.
