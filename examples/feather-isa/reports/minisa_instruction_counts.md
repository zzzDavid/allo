# MINISA Instruction Count Analysis

## Overview

This report analyzes the instruction count efficiency of MINISA compared to traditional element-level ISA approaches for FEATHER+ programming.

## MINISA Instruction Model

### Instruction Types

| Instruction | Purpose | Frequency per Layer |
|-------------|---------|-------------------|
| SetIVNLayout | Configure streaming buffer | 1 (or reuse) |
| SetWVNLayout | Configure stationary buffer | 1 (or reuse) |
| SetOVNLayout | Configure output + clear | 1 |
| SetMapping | Trigger tile execution | # of tiles |

### Key Insight

MINISA operates at **VN granularity**, where each VN represents AH elements (the dot-product size). This reduces control overhead by a factor of AH compared to element-level programming.

## Instruction Count Analysis

### Single GEMM Layer

For a GEMM of size M×K×N with array dimensions AH×AW:

```
Layout instructions: 3 (IVN, WVN, OVN)
Mapping instructions: ceil(M/AW) × ceil(K/AH) × ceil(N/AW)
Total: 3 + (M_tiles × K_tiles × N_tiles)
```

### Benchmark Results

| GEMM Size | Array (AH×AW) | Tiles | Layout | Mapping | Total MINISA |
|-----------|---------------|-------|--------|---------|--------------|
| 8×8×8 | 8×8 | 1 | 3 | 1 | 4 |
| 8×16×8 | 8×8 | 2 | 3 | 2 | 5 |
| 16×8×16 | 8×8 | 4 | 3 | 4 | 7 |
| 16×16×16 | 8×8 | 8 | 3 | 8 | 11 |
| 32×32×32 | 8×8 | 64 | 3 | 64 | 67 |
| 64×64×64 | 8×8 | 512 | 3 | 512 | 515 |
| 128×128×128 | 8×8 | 4096 | 3 | 4096 | 4099 |

### Comparison with Element-Level ISA

Traditional element-level control requires instructions per element operation:

```
Element-level instructions ≈ M × K × N × (compute + routing)
```

| GEMM Size | MINISA | Element-Level (est.) | Reduction Factor |
|-----------|--------|---------------------|------------------|
| 8×8×8 | 4 | ~512 | 128× |
| 16×16×16 | 11 | ~4096 | 372× |
| 32×32×32 | 67 | ~32768 | 489× |
| 64×64×64 | 515 | ~262144 | 509× |
| 128×128×128 | 4099 | ~2097152 | 512× |

The reduction factor approaches **AH × some constant** for large workloads, reflecting the VN abstraction benefit.

## Multi-Layer Analysis

### MLP-Style Networks

For an L-layer network where each layer has dimensions Mᵢ×Kᵢ×Nᵢ:

**With Layout Reuse:**
```
Initial layout: 2 (IVN, WVN if shapes match)
Per layer: 1 (OVN for clear) + tiles
Total ≈ 2 + L × (1 + tiles_per_layer)
```

**Without Layout Reuse:**
```
Per layer: 3 + tiles
Total = L × (3 + tiles_per_layer)
```

### Example: 3-Layer MLP (8→4→4→8)

| Layer | Input×Output | Tiles | Layout | Mapping |
|-------|--------------|-------|--------|---------|
| 1 | 4×8 → 4×4 | 2 | 3 | 2 |
| 2 | 4×4 → 4×4 | 1 | 3 | 1 |
| 3 | 4×4 → 4×8 | 2 | 3 | 2 |
| **Total** | | | **9** | **5** |

With layout reuse (same M_L0, J_L1 between layers):
- Initial IVN/WVN: 2
- Per-layer OVN: 3
- Mappings: 5
- **Total with reuse: 10** (vs 14 without)

## Scaling Analysis

### Instruction Count Growth

| Dimension Scaling | MINISA Growth | Element Growth |
|------------------|---------------|----------------|
| M × 2 | +tiles | ×2 |
| K × 2 | +tiles | ×2 |
| N × 2 | +tiles | ×2 |
| All × 2 | ~×8 tiles | ×8 |

MINISA instruction count scales with **tiles**, not **elements**.

### Large Workload Efficiency

For a 1024×1024×1024 GEMM on 8×8 array:

```
Tiles = 128 × 128 × 128 = 2,097,152
MINISA instructions = 3 + 2,097,152 = 2,097,155
Element-level (est.) = 1024³ × 2 ≈ 2.1 billion
Reduction: ~1000×
```

## Layout Order Impact

The 3-bit layout order encoding enables optimal buffer access patterns without additional instructions:

| Order | Access Pattern | Use Case |
|-------|---------------|----------|
| 0b000 | K-major | Standard GEMM |
| 0b010 | N-major inner | Transposed access |
| 0b100 | N-major outer | Large N tiling |

Layout order selection has **zero instruction overhead** - it's encoded in the layout instruction itself.

## Reduction Pattern Impact

The SetMapping G_r parameter controls reduction group size:

| G_r | Reduction Ratio | BIRRD Usage | Tiles for K |
|-----|-----------------|-------------|-------------|
| AW | 1:1 (no reduction) | Minimal | K/AH |
| AW/2 | 2:1 | Moderate | K/(2×AH) |
| AW/4 | 4:1 | Heavy | K/(4×AH) |

Higher reduction ratios reduce tile count but increase BIRRD complexity.

## Conclusions

1. **O(1) Layout Overhead**: Only 3 layout instructions per layer regardless of GEMM size.

2. **Linear Tile Scaling**: Mapping instructions scale with number of tiles, not elements.

3. **VN Abstraction Benefit**: Reduction factor approaches AH (array height) for large workloads.

4. **Layout Reuse**: Same-shape consecutive layers can share IVN/WVN configurations.

5. **Encoding Efficiency**: Layout permutations encoded in 3 bits without extra instructions.

### Recommendation

For optimal instruction efficiency:
- Use largest feasible array dimensions (AH, AW)
- Enable layout reuse for same-shape layers
- Choose G_r to minimize total tiles while matching BIRRD capability
