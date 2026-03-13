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
python tests/test_crossbar_flexibility.py # Multi-Gr crossbar tests (Gr=AW, AW//2, mixed)

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
- Per-tile BIRRD configuration: Gr<AW → standard BIRRD reduction with Mt=AW//2 outputs;
  Gr=AW → all-PS pass-through with AW outputs (identity col_map).
- Supported Gr values: AW and AW//2 (limited by BIRRD 2-way reduction).
  Smaller Gr values (e.g., Gr=1 weight stationary) need additional reduction stages.
- Kt_per_pass = (AW/Gr)*AH determines how many K elements each K-pass covers.
  Mixed Kt_per_pass is supported: each tile computes its own actual_passes at
  runtime; max_k_passes is the compile-time loop bound (padding passes stream zeros).
- IVN/WVN layout orders (ORDER_012 through ORDER_210) control VN buffer memory layout.
  In our direct-indexing model (no VN buffer), the crossbar routing is determined by
  Gr/Gc/sr/sc from SetMapping, so all 6 orders produce correct results. OVN order
  DOES affect computation via BIRRD butterfly routing (already implemented).
- For cosim, use `mode="csyn"` to generate kernel.cpp, then manually patch m_axi depths
  and write C testbench (see `test_figure7_cosim.py` for pattern).
