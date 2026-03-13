# FEATHER+ MINISA Allo — Ticket Progress Summary

**Date**: 2026-03-13
**Branch**: minisa
**Status**: 8/9 tickets resolved, 1 remaining

## Resolved Tickets

### TICKET-001: Gr < AW//2 Reduction (P0, resolved)
Support for arbitrary power-of-2 Gr values in the crossbar via bit operations.
`ic_j % Gr` → `ic_j & (Gr-1)`, `ic_j // Gr` → `ic_j >> log2_Gr`.
Compiles to AND gates and shift muxes — zero pipeline penalty, no integer dividers.

### TICKET-002: Mixed Kt_per_pass Across Tiles (P0, resolved)
Different Gr values imply different `Kt_per_pass = (AW/Gr)*AH`. Each tile computes
`actual_passes` at runtime via shifts; `max_k_passes` is the compile-time loop bound.
Padding passes stream zeros (harmless: 0*x=0).

### TICKET-003: IVN/WVN Layout Permutations (P1, resolved)
All 6 IVN and 6 WVN layout orders produce correct GEMM. In our direct-indexing
model (no VN buffer), crossbar routing is determined by Gr/Gc/sr/sc — buffer
layout order only affects OVN (BIRRD routing). Verified with 13 order combinations.

### TICKET-004: 16x16 FPGA Deployment (P1, resolved)
AW=16 synthesis with `AP_INT_MAX_W 4096` patch. Parameterized GEMM test supports
arbitrary AW/AH with optional HLS csim/csynth modes.

### TICKET-005: Zero Point Subtraction (P0, resolved)
NEST compute decodes `iacts_zp` and `weights_zp` from instructions, computing
`(iact - iacts_zp) * (weight - weights_zp)` per PE. Matches RTL `feather_pe.v`.
6 test cases including negative zero points.

### TICKET-006: Post-Quantization int32 → int8 (P0, resolved)
Output accumulator applies `(accum * quant_scale + quant_zp) & 255` when
`quant_scale != 0`. Matches RTL `quant_post.v` formula
`(sign_extend_64(data) * scale + zero_extend_64(zp))[7:0]`.
7 test cases including combined zero points + quantization on AW=4 and AW=8.

### TICKET-007: Gc/sr/sc Crossbar for Full Dataflow Flexibility (P1, resolved)
Weight crossbar generalized to `wn_idx = n_start + sr * wc_i + sc * (wc_w & mask_Gc)`.
Added `output_n_base` array for per-column N-offset encoding BIRRD permutation.
sr=0 guard prevents AH-fold duplication. Tested with sr=0 output-stationary mapping.

### TICKET-008: Multi-Way BIRRD Reduction (P1, resolved)
Greedy forward-pass algorithm generates BIRRD instructions for arbitrary Gr.
At each butterfly stage, sets AL when both switch inputs belong to the same
reduction group. Produces full `log2(AW/Gr)` levels of tree reduction inside
BIRRD hardware. Verified for all 12 (AW, Gr) combinations across AW=4,8,16.

## Remaining Ticket

### TICKET-009: Multi-Layer RIR (Run-Infer-Run) (P3, open)
Chain multiple GEMM layers with quantized int8 output feeding the next layer's
int8 input. Requires post-quantization (done) plus layer-to-layer buffer management.

## Test Summary

| Test File | Tests | Status |
|-----------|-------|--------|
| test_figure7_mapping.py | 8 | PASS |
| test_full_matrix_gemm.py | 17 | PASS |
| test_crossbar_flexibility.py | 13 | PASS |
| test_parameterized_gemm.py (AW=8) | 12 | PASS |
| test_parameterized_gemm.py (AW=4) | 12 | PASS |
| **Total** | **62** | **ALL PASS** |

## Architecture Overview

The FEATHER+ Allo implementation uses 7 pipelined dataflow kernels:

1. **crossbar_load** — Parametric Gr/Gc/sr/sc crossbar with bit operations
2. **nest_compute** — AH×AW NEST MAC with zero point subtraction
3. **bus** — Unpack NEST output to BIRRD connections
4. **inst_rw** — Distribute per-tile BIRRD switch instructions
5. **BIRRD** — Butterfly reduction/reorder (2-way or multi-way per tile)
6. **output_accum** — Column remap + tile accumulation + post-quantization

Key metrics:
- RTL cosim: **1004 cycles** (Allo) vs 1120 (RTL reference) = **0.90x**
- Single Allo invocation handles complete matrices (no Python-level tiling loop)
- All operations HLS-friendly: shifts, masks, comparisons — no runtime dividers

## Changes in This Commit

### Files modified (7):
- `CLAUDE.md` — Updated design decisions for multi-way BIRRD and sr/sc support
- `feather_minisa.py` — Gc/sr/sc crossbar, output_n_base, multi-way BIRRD dispatch
- `minisa/isa.py` — quant_scale/quant_zp in SetOVNLayout and create_gemm_program
- `minisa/lowering.py` — Multi-way BIRRD generation algorithm + generalized simulator
- `tests/test_crossbar_flexibility.py` — sr=0 test, multi-way BIRRD GEMM tests
- `tests/test_figure7_cosim.py` — output_n_base port in cosim testbench
- `tests/test_full_matrix_gemm.py` — Post-quantization tests (7 cases)

### New files (6):
- `reports/ticket_progress_summary.md` — This report
- `tickets/TICKET-005-zero-point-subtraction.md` — Resolved
- `tickets/TICKET-006-post-quantization.md` — Resolved
- `tickets/TICKET-007-gc-sr-sc-crossbar.md` — Resolved
- `tickets/TICKET-008-multi-way-birrd.md` — Resolved
- `tickets/TICKET-009-multi-layer-rir.md` — Open (P3)

### Net change: +783 lines across 7 modified files, 6 new ticket/report files
