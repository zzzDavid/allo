# FEATHER+ Figure 7: Allo vs RTL Cycle Count Comparison

**Date:** 2026-03-04
**Status:** K-streaming + FIFO tuning applied, gap reduced from 1.6x to 1.08x
**Branch:** minisa

## Workload

**C[16,8] = A[16,12] x B[12,8]** on 4x4 NEST PE array (AH=AW=4)

## Results Summary

| Implementation | Cycles | Tiles | Notes |
|---|---|---|---|
| **Allo K-streaming + FIFO tuning (current)** | **1208** | 8 | Gr=AW=4, K-fused, no dividers, tuned FIFOs |
| Allo K-streaming (before FIFO tuning) | 1213 | 8 | Gr=AW=4, K-fused, no dividers |
| Allo baseline (previous) | 1792 | 24 | Gr=2/4, per-K-group tiles, 14 dividers |
| **RTL reference** | **1120** | 8 | Icarus Verilog, K-distribution mode |

**Current gap: 1.08x** (was 1.6x). K-streaming delivered a **1.48x speedup** (1792 -> 1208).

## Optimization History

### Baseline: 1792 cycles (24 tiles)

Root causes identified:
1. **24 tiles vs 8**: Each (M-block, N-group, K-group) = separate tile. RTL fuses K-passes.
2. **14 runtime dividers**: `ic_j % Gr` / `ic_j // Gr` compiled to sdiv/srem, inflating
   input crossbar II from 1 to 2, weight crossbar II from 1 to 8.
3. **No ping-pong buffering**: Crossbar fill and NEST compute are sequential per tile.

See `reports/figure7_gap_analysis.md` for the full baseline analysis.

### K-streaming optimization: 1213 cycles (8 tiles)

Changes:
- `create_figure7_program()`: 8 Gr=4 tiles (was 24 mixed Gr=2/4), each covering K=[0,12)
- New `get_feather_full_matrix_top_kstreaming()`: inner K-loop in crossbar_and_NEST,
  accumulates NEST partial products in int32 before streaming to BIRRD
- Gr=AW=4 eliminates all dividers: `ic_j % 4 = ic_j`, `ic_j // 4 = 0`
- int32 intermediate type (TyOut) prevents accumulation overflow

Per-kernel synthesis (K-streaming):

| Kernel | Cycles | II | Notes |
|---|---|---|---|
| crossbar_and_NEST_0 | 609 | - | 8 tiles x 76 cycles/tile, K-pass pipeline II=14 |
| output_accum_0 | 396 | - | Was 652 (fewer tiles to process) |
| BIRRD_{0..2}_{0,1} | ~99 | 1 | Unchanged (pass-through with Gr=AW) |
| bus_0 | ~98 | 1 | Unpacks int32 from UInt(128) |

Resources (K-streaming):

| Resource | Used | Baseline Used |
|---|---|---|
| BRAM_18K | 18 | 21 |
| DSP | 49 | 9 |
| FF | 18,503 | 45,973 |
| LUT | 21,580 | 45,407 |

DSP usage increased (48 DSPs for 16 MAC units in unrolled NEST), but FF/LUT dropped
dramatically due to elimination of all 14 divider instances (was 32k FF + 24k LUT).

### FIFO depth tuning: 1208 cycles (5 cycles saved)

Increased inter-kernel FIFO depths to test backpressure hypothesis:
- `nest_out`: depth 4 → 8, `connection`: depth 1 → 4, `inst_input`: depth 1 → 8
- **Result: only 5 cycles saved** — gap is structural, not from FIFO backpressure
- See `reports/figure7_kstreaming.md` for detailed FIFO experiment

## Remaining Gap Analysis (1208 vs 1120 = 88 cycles)

The remaining 8% gap is structural:

1. **Sequential crossbar fill + NEST compute** (~50-70 cycles): Within each K-pass,
   crossbar fill (read A/B into iActs/weights) and NEST MAC are sequential due to WAR
   dependencies. RTL double-buffers crossbar data, overlapping K-pass T+1's load with
   K-pass T's compute. With 8 tiles x 3 K-passes, this overlap saves ~2-3 cycles/K-pass.

2. **Pipeline startup/drain** (~20-30 cycles): Dataflow pipeline fill/drain between
   the 5 kernel stages (crossbar_and_NEST -> bus -> BIRRD -> output_accum -> store).

### Next step: Phase 2 — Split crossbar_and_NEST

Split into `crossbar_load` and `nest_compute` kernels connected by intermediate PIPO
buffers. This allows Vitis HLS to overlap crossbar loading with NEST computation across
K-passes, potentially reducing K-pass pipeline II from 14 toward ~7.

## Verification

All tests pass:
- Simulator functional: `test_figure7_functional_gemm` — PASSED
- HLS csim: `test_figure7_hls_csim` — PASSED (output matches numpy reference)
- HLS csynth: `test_figure7_hls_csynth` — PASSED (812 cycle estimate)
- RTL cosim: `test_figure7_cosim` — PASSED (1208 cycles with FIFO tuning, all outputs match)
- Regression: `test_full_matrix_gemm`, `test_full_matrix_hls_csim` — PASSED

## Files

| File | Description |
|---|---|
| `feather_minisa.py` | K-streaming kernel (`get_feather_full_matrix_top_kstreaming`) |
| `minisa/isa.py` | Updated `create_figure7_program()` (8 Gr=4 tiles) |
| `tests/test_figure7_cosim.py` | RTL cosim (1213 cycles) |
| `tests/test_figure7_hls.py` | HLS csim/csynth tests |
| `tests/test_figure7_mapping.py` | ISA mapping + functional GEMM |
| `reports/figure7_gap_analysis.md` | Baseline gap analysis (1792 cycles) |
| `reports/figure7_kstreaming.md` | K-streaming optimization details |
