# MINISA Support Plan for FEATHER Allo Implementation

## 1. Overview

This document describes how to extend the existing FEATHER Allo implementation to support
the MINISA (Minimal Instruction Set Architecture) programming model. MINISA provides a
VN-level (Virtual Neuron) abstraction that reduces control overhead while preserving
FEATHER's dataflow and layout flexibility.

## 2. FEATHER Paper to Allo Code Mapping

### 2.1 NEST (Neural Engine with Spatial forwarding and Temporal reduction)

**Paper Description:**
- 2D PE array of size AH × AW
- Each PE performs AH-way dot product (local temporal reduction)
- Rows forward results sequentially to BIRRD (spatial forwarding)

**Allo Implementation (`feather.py`):**
```python
@df.kernel(mapping=[1], args=[iActs, weights])
def NEST(local_iActs: Ty[AH, AW], local_weights: Ty[AH, AW, AH]):
    for i in allo.grid(AH, name="nest"):  # Rows (time-multiplexed)
        local_buffer: Ty[AW] = 0
        for j in range(AW):  # Cols (parallel)
            temp: Ty = 0
            for k in range(AH):  # AH-way dot product
                temp += local_iActs[k, j] * local_weights[i, j, k]
            local_buffer[j] = temp
        nest_out.put(packed_result)  # Forward to BIRRD
```

**Mapping:**
| Paper Concept | Allo Code |
|---------------|-----------|
| PE array | Single kernel with `mapping=[1]` |
| AH-way dot product | Inner `for k in range(AH)` loop |
| Temporal reduction | Accumulation in `temp` variable |
| Spatial forwarding | `nest_out.put()` stream output |
| Time-multiplexed rows | Outer `for i in allo.grid(AH)` loop |

### 2.2 BIRRD (Butterfly Interconnect for Reduction and Reordering)

**Paper Description:**
- Multi-stage butterfly network with 2×log₂(AW) stages
- Each stage has AW/2 EGG switches
- Supports 4 operations: Pass(=), Add-Right(±), Add-Left(∓), Swap(×)
- Enables arbitrary reduction and reordering patterns

**Allo Implementation:**
```python
@df.kernel(mapping=[P0, P1])  # P0 stages, P1 switches per stage
def BIRRD():
    i, j = df.get_pid()  # Stage and switch indices
    inst_val = inst_input[i, j].get()
    for _ in range(AH):
        in_left = connection[i, 2*j].get()
        in_right = connection[i, 2*j + 1].get()
        # Execute EGG operation based on instruction
        if inst_val == PS: out_left, out_right = in_left, in_right
        elif inst_val == AR: out_left, out_right = in_left, in_left + in_right
        # ... routing via reverse_bits()
```

**Mapping:**
| Paper Concept | Allo Code |
|---------------|-----------|
| BIRRD stages | `P0 = 2 * log2(AW)` |
| EGG switches | Conditional operations in kernel |
| Inter-stage routing | `reverse_bits()` function |
| Reduction/reordering | AR/AL instructions |

## 3. FEATHER+ Changes for MINISA

### 3.1 Motivation

The baseline FEATHER has limitations for dynamic workloads:

1. **Point-to-point buffer connections**: Weights must be pre-organized offline
2. **Heavy control overhead**: Per-switch instructions don't scale

### 3.2 FEATHER+ Enhancements

**Enhancement 1: All-to-All Distribution Crossbars**

```
Baseline FEATHER:
  Buffer[col] → PE[col]  (fixed connection)

FEATHER+:
  Buffer[any] → Crossbar → PE[any]  (flexible routing)
```

This enables:
- Any VN to route to any PE column
- Multicast of VNs to multiple columns
- Dynamic input/weight distribution at runtime

**Enhancement 2: Simplified Streaming Buffer**

```
Baseline: Multi-bank streaming buffer with per-column constraints
FEATHER+: Single logical bank with VN-level addressing
```

### 3.3 Implementation in Allo

The FEATHER+ refinement is implemented as a wrapper/extension:

```python
# feather_plus.py (conceptual)
class FeatherPlusConfig:
    def __init__(self, AH, AW):
        self.AH = AH  # Array height = VN size
        self.AW = AW  # Array width
        self.vn_size = AH

    def create_input_crossbar(self):
        """All-to-all distribution from input buffer to NEST"""
        # Routes IVN[any] → PE column[any]

    def create_weight_crossbar(self):
        """All-to-all distribution from weight buffer to NEST"""
        # Routes WVN[any] → PE column[any]
```

## 4. MINISA Instruction Mapping

### 4.1 Virtual Neuron (VN) Abstraction

A Virtual Neuron is the smallest hardware dot-product atom:
- **Size**: AH elements (matching PE reduction depth)
- **Types**: IVN (input), WVN (weight), OVN/PVN (output/partial sum)

```
For GEMM O[M,N] = I[M,K] × W[K,N]:
  IVN has AH elements from K dimension
  WVN has AH elements from K dimension
  OVN has AH elements from N dimension (output layout)
```

### 4.2 MINISA Instructions

| Instruction | Purpose | Parameters |
|-------------|---------|------------|
| `SetIVNLayout` | Configure streaming buffer for inputs | Order, M_L0, M_L1, J_L1 |
| `SetWVNLayout` | Configure stationary buffer for weights | Order, N_L0, N_L1, K_L1 |
| `SetOVNLayout` | Configure output buffer layout + clear | Order, P_L0, P_L1, Q_L1 |
| `SetMapping` | Trigger tile execution | r₀, c₀, G_r, G_c, s_r, s_c |

### 4.3 Layout Encoding

VN layouts use a 3-bit order encoding with rank partitioning:

```
For WVN layout (weights K×N):
  VN constraint: K_L0 = AH (each VN is AH elements along K)
  Layout order: permutation of {k_L1, n_L0, n_L1}

  order=000: k_L1 → n_L0 → n_L1 (K-major)
  order=010: n_L0 → k_L1 → n_L1 (N_L0-major)
  ...
```

### 4.4 SetMapping Parametric Encoding

The mapping parameters define how WVNs are assigned to PEs:

```
WVN(r, c) assigned to PE(ah, aw) where:
  r = r₀ + ⌊aw / G_r⌋           # Row index
  c = c₀ + s_r × ah + s_c × (aw mod G_c)  # Column index
```

Parameters:
- `r₀, c₀`: Base VN indices
- `G_r`: PE columns sharing same WVN row (reduction group)
- `G_c`: Replication group size
- `s_r, s_c`: Temporal and spatial strides

## 5. Lowering: MINISA → FEATHER Control

### 5.1 Layout Lowering

`SetWVNLayout(Order, N_L0, N_L1, K_L1)` generates:
1. Buffer address mapping function
2. VN index to buffer row/column translation

```python
def lower_wvn_layout(order, N_L0, N_L1, K_L1, AW):
    # Compute linear VN index based on order permutation
    def vn_to_buffer_addr(r, c):
        # r = k_L1, c = n_L1 * N_L0 + n_L0
        k_L1 = r
        n_L0 = c % N_L0
        n_L1 = c // N_L0

        # Apply permutation order
        D = [K_L1, N_L0, N_L1]
        I = [k_L1, n_L0, n_L1]
        P = PERMUTATIONS[order]
        L = I[P[0]] * D[P[1]] * D[P[2]] + I[P[1]] * D[P[2]] + I[P[2]]

        row = L // AW
        col = L % AW
        return row, col
    return vn_to_buffer_addr
```

### 5.2 Mapping Lowering

`SetMapping(r₀, c₀, G_r, G_c, s_r, s_c)` generates:
1. PE-to-VN assignment
2. Input distribution crossbar configuration
3. Weight distribution crossbar configuration
4. BIRRD instruction array

```python
def lower_mapping(params, AH, AW):
    r0, c0, G_r, G_c, s_r, s_c = params

    # Generate WVN-to-PE mapping
    pe_wvn_map = {}
    for ah in range(AH):
        for aw in range(AW):
            r = r0 + aw // G_r
            c = c0 + s_r * ah + s_c * (aw % G_c)
            pe_wvn_map[(ah, aw)] = (r, c)

    # Generate BIRRD instructions based on reduction pattern
    birrd_inst = generate_birrd_from_mapping(pe_wvn_map, AH, AW)

    return pe_wvn_map, birrd_inst
```

### 5.3 BIRRD Instruction Generation

Given the PE-to-VN mapping, BIRRD instructions are generated to route
partial sums to correct output positions while performing reduction:

```python
def generate_birrd_instructions(pe_wvn_map, reduction_groups, output_layout):
    """
    Generate BIRRD instruction array for a tile mapping.

    Args:
        pe_wvn_map: Dict mapping (ah, aw) -> (vn_row, vn_col)
        reduction_groups: List of PE columns that reduce together
        output_layout: Target OVN layout

    Returns:
        inst: np.ndarray of shape (P0, P1) with EGG instructions
    """
    # Determine which columns need reduction (same VN row)
    # Route reduced results to output positions
    # Use butterfly routing algorithm
```

## 6. Implementation Plan

### Phase 1: Core Infrastructure
1. `minisa/isa.py`: Instruction class definitions
2. `minisa/layout.py`: VN layout descriptors and encoding
3. `minisa/interpreter.py`: MINISA interpreter state machine

### Phase 2: Lowering Layer
1. `minisa/lowering.py`: Layout/mapping to FEATHER config translation
2. `minisa/birrd_codegen.py`: BIRRD instruction generation

### Phase 3: FEATHER+ Integration
1. Distribution crossbar models (functional)
2. VN buffer addressing
3. Integration with existing FEATHER kernels

### Phase 4: Verification
1. Correctness tests vs numpy reference
2. Multi-layout/mapping tests
3. Multi-layer chain tests

## 7. Mapping Constraints

### 7.1 Reduction-Aware Constraint

Within each PE column, all PEs must share the same WVN row index:

```
∀ ah₁, ah₂ ∈ [0, AH): WVN_row(ah₁, aw) = WVN_row(ah₂, aw)
```

This is enforced by Eq. (2): `r(ah, aw) = r₀ + ⌊aw/G_r⌋` (independent of `ah`)

### 7.2 WVN Replication

The same WVN may be replicated across multiple PE columns:
- `G_r` columns share the same WVN row
- `G_c` columns form replication groups

### 7.3 Out-of-Bounds Handling

VN indices exceeding workload bounds are implicitly zero-padded:

```python
if r >= num_wvn_rows or c >= num_wvn_cols:
    wvn_data = zeros(AH)  # Zero-padded VN
```

## 8. Verification Strategy

### 8.1 Unit Tests
- Layout encoding/decoding roundtrip
- Mapping parameter validation
- BIRRD instruction generation correctness

### 8.2 Integration Tests
- Small GEMM (4×4, 8×8) vs numpy reference
- Different layout configurations
- Different mapping patterns

### 8.3 System Tests
- Multi-layer execution with layout switching
- Instruction counting and efficiency metrics
- Comparison with micro-instruction baseline

## 9. Expected Outcomes

1. **Correctness**: MINISA produces identical results to direct FEATHER execution
2. **Flexibility**: All legal FEATHER mappings are expressible via MINISA
3. **Efficiency**: Dramatic reduction in instruction count (per MINISA paper: ~10⁵× reduction)
4. **Simplicity**: 4 instructions capture full dataflow/layout space
