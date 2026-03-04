# FEATHER+ Figure 7: Allo vs RTL Cycle Count Comparison

**Date:** 2026-03-04
**Status:** Allo beats RTL reference — 1001 vs 1120 cycles (0.89x)
**Branch:** minisa

## Workload

**C[16,8] = A[16,12] x B[12,8]** on 4x4 NEST PE array (AH=AW=4)

## Results Summary

| Implementation | Cycles | Tiles | Notes |
|---|---|---|---|
| **Allo split-kernel v2 (current)** | **1001** | 8 | Split crossbar_load + nest_compute, K-pass II=4 |
| Allo K-streaming v1 + FIFO tuning | 1208 | 8 | Fused crossbar_and_NEST, K-pass II=14 |
| Allo K-streaming v1 | 1213 | 8 | Gr=AW=4, K-fused, no dividers |
| Allo baseline | 1792 | 24 | Gr=2/4, per-K-group tiles, 14 dividers |
| **RTL reference** | **1120** | 8 | Icarus Verilog, K-distribution mode |

**Allo is 11% faster than RTL** (1001 / 1120 = 0.89x). Total speedup: **1.79x** (1792 → 1001).

## Optimization History

### Phase 0: Baseline — 1792 cycles (24 tiles)

Root causes: 24 tiles (vs RTL's 8), 14 runtime dividers (sdiv/srem), no ping-pong
buffering. See `reports/figure7_gap_analysis.md`.

### Phase 1: K-streaming — 1213 cycles (8 tiles)

Fused K-passes within tiles, reduced tile count from 24 to 8, eliminated all dividers.
K-pass pipeline II=14. See `reports/figure7_kstreaming.md`.

### Phase 1b: FIFO depth tuning — 1208 cycles

Increased inter-kernel FIFO depths (nest_out 4→8, connection 1→4, inst_input 1→8).
Only 5 cycles saved — confirmed gap was structural, not from backpressure.

### Phase 2: Split crossbar/NEST kernels — 1001 cycles

Split `crossbar_and_NEST` into separate `crossbar_load` and `nest_compute` dataflow
kernels connected by intermediate UInt(128) streams. This provides two key benefits:

1. **K-pass pipeline II: 14 → 4** (3.5x improvement). Without array read latency in
   the critical path, HLS achieves much better pipelining for the NEST computation.

2. **Dataflow overlap**: crossbar_load (198 cycles) runs concurrently with nest_compute
   (401 cycles), overlapping data loading with computation across tiles.

Per-kernel synthesis (v2):

| Kernel | Cycles | Notes |
|---|---|---|
| crossbar_load_0 | 198 | Flattened tile+K-pass loop, II=8 |
| nest_compute_0 | 401 | 50 cycles/tile, K-pass II=4 |
| output_accum_0 | 396 | Unchanged from v1 |
| BIRRD_{0..2}_{0,1} | ~99 | Pass-through (Gr=AW) |
| bus_0 | ~98 | Unpacks int32 from UInt(128) |

Resources (v2):

| Resource | V2 | V1 | Baseline |
|---|---|---|---|
| BRAM_18K | 18 | 18 | 21 |
| DSP | 49 | 49 | 9 |
| FF | 17,439 | 18,503 | 45,973 |
| LUT | 22,238 | 21,580 | 45,407 |

FF slightly reduced, LUT slightly increased. Overall similar resource footprint.

## Data Transfer Protocol (crossbar_load → nest_compute)

Per K-pass, crossbar_load sends:
- 1 x UInt(128): packed iActs[4,4] (16 int8 values)
- 4 x UInt(128): packed weights[wc_i,:,:] (16 int8 each, one per wc_i)

Total: 5 stream operations per K-pass, 15 per tile, 120 for all 8 tiles.

Stream depths: `iacts_stream` depth=6, `weights_stream` depth=24 (allows ~2 tiles buffering).

## Verification

All tests pass:
- Simulator functional: `test_figure7_v2_functional_gemm` — PASSED
- HLS csim: `test_figure7_v2_hls_csim` — PASSED (output matches numpy reference)
- HLS csynth: `test_figure7_v2_hls_csynth` — PASSED (764 cycle estimate)
- RTL cosim: `test_figure7_v2_cosim` — **1001 cycles, PASSED**
- V1 regression: `test_figure7_functional_gemm`, `test_figure7_mapping` — PASSED

## Files

| File | Description |
|---|---|
| `feather_minisa.py` | V2 split-kernel (`get_feather_full_matrix_top_kstreaming_v2`) |
| `feather_minisa.py` | V1 K-streaming (`get_feather_full_matrix_top_kstreaming`) |
| `minisa/isa.py` | `create_figure7_program()` (8 Gr=4 tiles) |
| `tests/test_figure7_v2.py` | V2 tests (sim, csim, csynth, cosim) |
| `tests/test_figure7_cosim.py` | V1 RTL cosim |
| `tests/test_figure7_hls.py` | V1 HLS csim/csynth |
| `tests/test_figure7_mapping.py` | ISA mapping + functional GEMM |
| `reports/figure7_gap_analysis.md` | Baseline gap analysis (1792 cycles) |
| `reports/figure7_kstreaming.md` | K-streaming + FIFO tuning details |
| `reports/figure7_split_kernel.md` | Phase 2 split-kernel details |
