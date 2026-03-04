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
python tests/test_full_matrix_gemm.py    # Full-matrix GEMM regression

# HLS tests (require Vitis HLS)
python tests/test_figure7_hls.py         # HLS csim + csynth (764 cycles)
python tests/test_full_matrix_hls_csim.py     # Full-matrix HLS csim regression

# RTL co-simulation (requires Vitis HLS, takes several minutes)
python tests/test_figure7_cosim.py       # RTL cosim (1001 cycles)
```

## Project Structure

- `feather_minisa.py` — FEATHER+ dataflow kernels
  - Original kernel: `get_feather_full_matrix_top()` — general-purpose, supports all Gr values
  - K-streaming kernel: `get_feather_full_matrix_top_kstreaming()` — Gr=AW only, split crossbar_load + nest_compute (best: 1001 cycles)
- `minisa/isa.py` — MINISA ISA definitions and program generation
- `minisa/lowering.py` — BIRRD lowering and output column mapping
- `tests/` — All test files
- `reports/` — Analysis reports and synthesis results

## Key Design Decisions

- Each kernel variant is a separate function (not a mode flag) because Allo parses kernel
  source via `inspect.getsource()`, so Python-level conditionals inside kernels don't
  compile to HLS branches as expected.
- `TyOut = int32` for K-streaming intermediate type (int8 accumulation overflows with
  multiple K-passes).
- Split-kernel packs crossbar data into UInt(128) streams: 1 iActs packet + 4 weights
  packets per K-pass. This breaks the WAR dependency that caused fused kernel's II=14.
- Original kernel left unchanged for backward compatibility with non-Figure-7 workloads.
- For cosim, use `mode="csyn"` to generate kernel.cpp, then manually patch m_axi depths
  and write C testbench (see `test_figure7_cosim.py` for pattern).
