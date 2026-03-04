# FEATHER+ Figure 7: K-Streaming Optimization

**Date:** 2026-03-04
**Status:** Implemented and verified via RTL cosim
**Branch:** minisa

## Summary

K-streaming fuses multiple K-passes within each tile, reducing tile count from 24 to 8
and eliminating all 14 runtime integer dividers. This brought the Allo cycle count from
**1792 to 1213** (1.48x speedup), closing the gap vs RTL from 1.6x to **1.08x**.

| Metric | Before | After |
|---|---|---|
| RTL cosim cycles | 1792 | **1213** |
| HLS csynth estimate | 1804 | **812** |
| Tile count | 24 | 8 |
| Runtime dividers | 14 (sdiv/srem) | 0 |
| FF utilization | 45,973 | 18,503 |
| LUT utilization | 45,407 | 21,580 |
| DSP utilization | 9 | 49 |
| vs RTL reference (1120) | 1.60x | **1.08x** |

## Approach

### Key Insight

Using Gr=AW=4 for all tiles simultaneously solves two problems:
1. **K-streaming**: With Gr=AW, all PE columns share the same WVN row. The BIRRD
   becomes pass-through (no reduction), so we can accumulate NEST partial products
   across K-passes *before* streaming to BIRRD. This fuses K into the tile loop.
2. **Divider elimination**: When Gr=AW=4, `ic_j % Gr = ic_j % 4 = ic_j` and
   `ic_j // Gr = ic_j // 4 = 0` for ic_j in {0,1,2,3}. The crossbar indexing
   becomes pure addition — no division needed.

### No ISA Changes Required

The MINISA ISA already has `k_start`/`k_end` fields in SetMapping. The change is purely
in (a) program generation (fewer tiles with full K-range) and (b) the kernel (inner
K-loop with int32 accumulation).

### Type Widening

With 3 K-passes of int8 x int8 MAC, the accumulated value can reach 3 x 4 x 127 x 127
= 193,548, which overflows int8. The K-streaming kernel uses:
- `TyOut = int32` for NEST accumulation, BIRRD, and output_accum intermediate values
- `TyPacked = UInt(128)` for packing 4 int32 values (was `UInt(32)` for 4 int8)
- Input types `A: int8[M,K]` and `B: int8[K,N]` remain unchanged

## Implementation

### 1. Program generation (`minisa/isa.py`)

`create_figure7_program()` changed from 24 tiles to 8:

```
Before: 16 Gr=2 tiles (K=[0,8)) + 8 Gr=4 tiles (K=[8,12)) = 24 tiles
After:  8 Gr=4 tiles (4 M-blocks x 2 N-groups), each K=[0,12)
```

### 2. K-streaming kernel (`feather_minisa.py`)

New function `get_feather_full_matrix_top_kstreaming(M, K, N, AW, AH, Ty, num_inst, num_k_passes, Kt_per_pass)`:

**crossbar_and_NEST** changes:
- Hardcoded ORDER_012 with Gr=AW: `m_idx = m_start + ic_j`, `k_idx = k_start + ic_i`
- Inner K-loop: `for k_pass in range(num_k_passes):`
  - Each pass computes `k_start = k_start_tile + k_pass * Kt_per_pass`
  - Fills iActs and weights for this K-pass
  - Runs NEST, accumulates into `nest_accum: int32[AH, AW]`
- After K-loop: packs int32 values into UInt(128) and streams to nest_out

**bus**: Unpacks int32 from UInt(128) instead of int8 from UInt(32).

**BIRRD**: Operates on int32. Same switch logic (all pass-through with Gr=AW).

**output_accum**: Receives int32 from BIRRD streams.

### 3. Wrapper (`FeatherKStreamingModule`)

Simplified wrapper — all tiles are Gr=AW, so BIRRD is always pass-through and
col_map is identity. Interface: `mod(A, B, instructions, C)`.

### 4. Build helpers

- `build_feather_kstreaming_simulator()` — for functional testing
- `build_feather_kstreaming_hls()` — for csim/csynth/cosim

## Synthesis Details

### crossbar_and_NEST (critical path kernel)

```
Total: 609 cycles (8 tiles x 76 cycles/tile)
  Per tile:
    accum_init (l_S_ai_0_ai_l_S_aj_0_aj): 18 cycles
    K-pass loop (l_S_k_pass_2_k_pass):    44 cycles (3 passes x ~14 cycles)
    stream_out (l_nest_stream_ni1):         6 cycles
    overhead:                               8 cycles
```

The K-pass pipeline achieves II=14 with trip count 3 = 44 cycles. No sdiv/srem
instances. Uses 48 DSPs for 16 MAC units (fully unrolled 4x4 NEST per K-pass).

### output_accum

396 cycles (was 652). Fewer tiles to process (8 vs 24).

## Verification

| Test | Result |
|---|---|
| Simulator (`test_figure7_functional_gemm`) | PASSED |
| HLS csim (`test_figure7_hls_csim`) | PASSED |
| HLS csynth (`test_figure7_hls_csynth`) | 812 cycles |
| RTL cosim (`test_figure7_cosim`) | **1213 cycles, PASSED** |
| Regression (`test_full_matrix_gemm`) | PASSED |
| Regression (`test_full_matrix_hls_csim`) | PASSED |

## Next Steps

The remaining 8% gap (1213 vs 1120 = 93 cycles) is primarily from lack of
**ping-pong buffering**. The RTL double-buffers crossbar data to overlap the next
tile's data loading with the current tile's NEST compute. Adding this could save
~8-10 cycles per tile (64-80 cycles total), potentially bringing Allo within 2-3%
of the RTL reference.
