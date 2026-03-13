# FEATHER+ MINISA (Allo)

## Environment Setup

All commands require two environment activations:

```bash
source /home/nz264/.local/bin/allo-env.sh          # conda py312 + LLVM
source /opt/xilinx/Vitis_HLS/2023.2/settings64.sh  # Vitis HLS (for csim/csynth/cosim)
```

The first sets up Python 3.12, LLVM, and LD_LIBRARY_PATH.
The second adds `vitis_hls` to PATH (required for HLS tests).

## Running Tests

```bash
# Simulator-only tests (no Vitis HLS required)
python tests/test_figure7_mapping.py     # ISA mapping + functional GEMM
python tests/test_full_matrix_gemm.py    # Full-matrix GEMM regression (AW=8)
python tests/test_crossbar_flexibility.py # Multi-Gr crossbar + sr=0/sc=0 tests
python tests/test_multi_layer.py          # Multi-layer sequential execution

# Parameterized tests for any AW/AH (default 16x16)
python tests/test_parameterized_gemm.py --aw 16 --ah 16           # simulator
python tests/test_parameterized_gemm.py --aw 8 --ah 8             # 8x8 regression
python tests/test_parameterized_gemm.py --aw 16 --ah 16 --hls csim  # HLS C-sim
python tests/test_parameterized_gemm.py --aw 16 --ah 16 --hls csyn  # HLS synthesis

# HLS tests (require Vitis HLS)
python tests/test_figure7_hls.py         # HLS csim + csynth (770 cycles)

# RTL co-simulation (requires Vitis HLS, takes several minutes)
python tests/test_figure7_cosim.py       # RTL cosim cycle count
```

## Project Structure

- `feather_minisa.py` — Unified FEATHER+ dataflow kernel
  - `get_feather_full_matrix_top_kstreaming()` — supports all power-of-2 Gr values per tile via bit operations in crossbar index arithmetic (no runtime dividers)
  - `FeatherKStreamingModule` — wrapper with per-tile BIRRD configuration (reduction for Gr<AW, pass-through for Gr=AW)
  - `build_feather_kstreaming_simulator()` / `build_feather_kstreaming_hls()` — build helpers
  - `run_sequential_gemm_layers()` — multi-layer chaining with int8 intermediates
- `minisa/isa.py` — MINISA ISA definitions and program generation
- `minisa/lowering.py` — BIRRD lowering and output column mapping
- `tests/` — All test files
- `reports/` — Analysis reports and synthesis results

## Key Design Decisions

- Single unified kernel handles all Gr values via power-of-2 bit operations:
  `ic_j % Gr` → `ic_j & (Gr-1)`, `ic_j // Gr` → `ic_j >> log2_Gr`.
  Compiles to AND gates and shift muxes (zero pipeline penalty, no integer dividers).
- `TyOut = int32` for K-streaming intermediate type (int8 accumulation overflows with
  multiple K-passes).
- Split-kernel packs crossbar data into UInt(AH*AW*Ty.bits) streams: 1 iActs packet + AH
  weights packets per K-pass. This breaks the WAR dependency that caused fused kernel's II=14.
- For AW=16 (UInt(2048)), kernel.cpp must be patched with `#define AP_INT_MAX_W 4096`
  before `#include <ap_int.h>` (Vitis HLS default is 1024).
- Per-tile BIRRD configuration: Gr=AW → all-PS pass-through with AW outputs;
  Gr=AW//2 → hand-coded 2-way reduction (order-dependent);
  Gr<AW//2 → algorithmically generated multi-way reduction via greedy forward pass.
- All power-of-2 Gr values (1 ≤ Gr ≤ AW) produce correct GEMM. Multi-way BIRRD
  reduction uses `generate_birrd_instructions(AW, Gr)` which greedily sets AL switches
  when both inputs belong to the same reduction group on the butterfly topology.
- Weight crossbar uses full MINISA mapping: `wn_idx = n_start + sr * wc_i + sc * (wc_w & mask_Gc)`.
  When sr=1, sc=0 (output stationary), this reduces to the original `wn_idx = n_start + wc_i`.
  When sr=0, all temporal rows see the same N column; output_accum has an sr=0 guard to
  prevent AH-fold duplication (only on=0 is accumulated).
- output_n_base[num_tiles, AW]: precomputed per-tile, per-output-column N-offset base that
  encodes `sc * (original_pe_col & mask_Gc)`, accounting for BIRRD butterfly permutation.
  Used in output_accum: `n_off = sr * on + n_base[tile, col]`.
- Kt_per_pass = (AW/Gr)*AH determines how many K elements each K-pass covers.
  Mixed Kt_per_pass is supported: each tile computes its own actual_passes at
  runtime; max_k_passes is the compile-time loop bound (padding passes stream zeros).
- Zero point subtraction: nest_compute decodes iacts_zp from instructions[0,6] and
  weights_zp from instructions[1,6], computing (iact - iacts_zp) * (weight - weights_zp)
  per PE, matching RTL feather_pe.v behavior. Set via create_gemm_program(iacts_zp=, weights_zp=).
- Post-quantization (int32 → uint8): output_accum decodes quant_scale from instructions[2,6]
  and quant_zp from instructions[2,7]. When quant_scale != 0, applies
  `(accum * quant_scale + quant_zp) & 255`, matching RTL quant_post.v formula
  `(sign_extend_64(data) * scale + zero_extend_64(zp))[7:0]`. Disabled by default (scale=0).
  Set via create_gemm_program(quant_scale=, quant_zp=).
- IVN/WVN layout orders (ORDER_012 through ORDER_210) control VN buffer memory layout.
  In our direct-indexing model (no VN buffer), the crossbar routing is determined by
  Gr/Gc/sr/sc from SetMapping, so all 6 orders produce correct results. OVN order
  DOES affect computation via BIRRD butterfly routing (already implemented).
- Multi-layer sequential execution: `run_sequential_gemm_layers()` chains GEMM layers with
  int8 intermediates. Each non-final layer uses post-quantization (quant_scale != 0) to
  produce uint8 output, which is reinterpreted as int8 for the next layer's input. Matches
  RTL pipeline: OB → quant_post → StaB PONG write → next layer iActs read.
  Supports different dataflows per layer (e.g., OS layer 1 → passthrough layer 2).
- For cosim, use `mode="csyn"` to generate kernel.cpp, then manually patch m_axi depths
  and write C testbench (see `test_figure7_cosim.py` for pattern).
