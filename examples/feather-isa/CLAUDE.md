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
python tests/test_figure7_hls.py         # HLS csim + csynth (cycle estimate)
python tests/test_full_matrix_hls_csim.py  # Full-matrix HLS csim regression

# RTL co-simulation (requires Vitis HLS, takes several minutes)
python tests/test_figure7_cosim.py       # Cycle-accurate RTL measurement
```

## Project Structure

- `feather_minisa.py` — FEATHER+ dataflow kernels (crossbar_and_NEST, bus, BIRRD, output_accum, inst_rw)
  - Original kernel: `get_feather_full_matrix_top()` — general-purpose, supports all Gr values
  - K-streaming kernel: `get_feather_full_matrix_top_kstreaming()` — Gr=AW only, fuses K-passes within tiles
- `minisa/isa.py` — MINISA ISA definitions and program generation
- `minisa/lowering.py` — BIRRD lowering and output column mapping
- `tests/` — All test files
- `reports/` — Analysis reports and synthesis results

## Key Design Decisions

- K-streaming kernel is a separate function (not a mode flag) because Allo parses kernel
  source via `inspect.getsource()`, so Python-level conditionals inside kernels don't
  compile to HLS branches as expected.
- `TyOut = int32` for K-streaming intermediate type (int8 accumulation overflows with
  multiple K-passes).
- Original kernel left unchanged for backward compatibility with non-Figure-7 workloads.
