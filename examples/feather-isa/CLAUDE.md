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

# Trace-based tests (parse instruction trace JSON → MINISA → execute)
# All traces stored in instr_trace/ — supports RTL, MINISA, manual, multi-layer formats
python tests/test_trace_input.py instr_trace/trace_m24k48n512_16x16.json             # RTL trace
python tests/test_trace_input.py instr_trace/gemm_8x16x8_aw8.json                   # MINISA trace
python tests/test_trace_input.py instr_trace/crossbar_mixed_gr_aw4.json              # manual tiles
python tests/test_trace_input.py instr_trace/multi_layer_2layer_aw8.json             # multi-layer
python tests/test_trace_input.py instr_trace/trace_m24k48n512_16x16.json --hls csim  # HLS csim
python tests/test_trace_input.py instr_trace/trace_m24k48n512_16x16.json --hls csyn  # HLS synthesis
python tests/test_trace_input.py instr_trace/trace_m24k48n512_16x16.json --deploy    # U280

# HLS tests (require Vitis HLS)
python tests/test_figure7_hls.py         # HLS csim + csynth (770 cycles)

# RTL co-simulation (requires Vitis HLS, takes several minutes)
python tests/test_figure7_cosim.py       # RTL cosim cycle count
```

## Project Structure

- `feather_minisa.py` — Unified FEATHER+ dataflow kernel with spatial PE array
  - `get_feather_full_matrix_top_kstreaming()` — 5-kernel architecture: dram_loader(1) → pe_array[AH+1,AW] → BIRRD[P0,P1] → output_accum, plus inst_rw. dram_loader has on-chip streaming/stationary buffers. pe_array rows 0..AH-1 = compute PEs, row AH = gather. All power-of-2 Gr values via bit operations
  - `FeatherKStreamingModule` — wrapper with per-tile BIRRD configuration (reduction for Gr<AW, pass-through for Gr=AW)
  - `schedule_feather_hls()` — HLS scheduling (minimal — parallelism is structural via mapping=[AH+1,AW])
  - `build_feather_kstreaming_simulator()` / `build_feather_kstreaming_hls()` — build helpers
  - `run_sequential_gemm_layers()` — multi-layer chaining with int8 intermediates
- `minisa/isa.py` — MINISA ISA definitions and program generation
- `minisa/lowering.py` — BIRRD lowering and output column mapping
- `minisa/trace_parser.py` — Instruction trace parser with `load_trace()` unified entry point
  - Supports 4 formats: RTL trace, MINISA simple, MINISA manual, multi-layer
- `instr_trace/` — Instruction trace JSON files (57 files covering all test configs)
- `tests/` — All test files
- `reports/` — Analysis reports and synthesis results

## Key Design Decisions

- `dram_loader` (mapping=[1]) is the sole reader of A_pe, B_pe, inst_pe, loader_m_start,
  loader_n_start. Per-column streaming buffer (iacts_buf[AH]) and stationary buffer
  (weight_buf[AH,AH]) are loaded per sub-operation, then streamed to PEs via
  meta_for(AW) × meta_for(AH). A and B passed as int32 (int8 causes LLVM crashes in
  spatial kernels); `FeatherKStreamingModule.__call__()` converts via `.astype(np.int32)`.
- `pe_array` (mapping=[AH+1, AW]) is stream-only (no DRAM args). Rows 0..AH-1 = AH*AW
  compute PEs (each doing 1 MAC/cycle). Row AH = AW gather instances. Uses `meta_if` on
  `get_pid()` for row specialization. Parallelism is structural (guaranteed by construction).
- Each compute PE reads from pe_a_in[ni,nj] and pe_w_in[ni,nj], does `a * w` + accumulate,
  and puts result on pe_out[ni,nj]. The multiply maps to 3 DSP48E2 slices.
- 5 dataflow kernels: dram_loader(1) → pe_array[AH+1,AW] → BIRRD[P0,P1] → output_accum,
  plus inst_rw. Each DRAM buffer has exactly one reader kernel (HLS single-reader compliance).
  13 DRAM args total: A_pe, B_pe, inst_pe, loader_m_start, loader_n_start (dram_loader);
  birrd_inst (inst_rw); output_col_map, output_num_m, output_n_base, accum_m_start,
  accum_n_start, accum_params, C (output_accum). loader_m/n and accum_m/n are separate
  copies of the same per-op data (single-reader compliance).
- Row AH (gather): each column instance reads AH PE outputs via meta_for(AH) into
  local buf[AH], then writes sequentially to connection[0, nj]. AW-way parallel (no
  single-instance bottleneck).
- Single unified kernel handles all Gr values via power-of-2 bit operations:
  `ic_j % Gr` → `ic_j & (Gr-1)`, `ic_j // Gr` → `ic_j >> log2_Gr`.
  Compiles to AND gates and shift muxes (zero pipeline penalty, no integer dividers).
- `TyOut = int32` for accumulation type (int8 accumulation overflows).
- Per-tile BIRRD configuration: Gr=AW → all-PS pass-through with AW outputs;
  Gr=AW//2 → hand-coded 2-way reduction (order-dependent);
  Gr<AW//2 → algorithmically generated multi-way reduction via greedy forward pass.
- All power-of-2 Gr values (1 ≤ Gr ≤ AW) produce correct GEMM. Multi-way BIRRD
  reduction uses `generate_birrd_instructions(AW, Gr)` which greedily sets AL switches
  when both inputs belong to the same reduction group on the butterfly topology.
- Weight crossbar uses full MINISA mapping: `wn_idx = n_start + sr * ni + sc * (nj & mask_Gc)`.
  When sr=1, sc=0 (output stationary), this reduces to `wn_idx = n_start + ni`.
  When sr=0, all temporal rows see the same N column; output_accum has an sr=0 guard to
  prevent AH-fold duplication (only on=0 is accumulated).
- output_n_base[num_tiles, AW]: precomputed per-tile, per-output-column N-offset base that
  encodes `sc * (original_pe_col & mask_Gc)`, accounting for BIRRD butterfly permutation.
  Used in output_accum: `n_off = sr * on + n_base[tile, col]`.
- Each tile processes exactly one K-slice (k_start to k_end). The ISA decomposes K
  at the tile level — there is no multi-pass K accumulation within a tile.
  Kt = (AW/Gr)*AH elements per tile. K > Kt is handled by multiple tiles.
- Temporal N-iteration and M-batching (n_inner > 1): matches RTL's VN temporal iteration
  where each ExecuteMapping covers all M-batches and N-sub-tiles internally. Instead of
  creating n_spatial * n_m_batches * n_sub_tiles separate ISA tiles, these are folded into
  n_inner sub-operations per tile. Only K-decomposition creates separate tiles.
  Per-sub-op m_start/n_start stored in DRAM lookup tables (loader_m_start, loader_n_start
  for dram_loader; accum_m_start, accum_n_start for output_accum). PE/gather/BIRRD loop
  over total_ops = num_tiles * n_inner (transparent to tile/inner boundary). inst_rw repeats
  each tile's BIRRD instruction n_inner times. For RTL trace M=24,K=48,N=512 on 16x16:
  192 tiles → 3 tiles × 64 inner iterations (same 192 total ops, ~3K target cycles vs 946K).
- output_accum reads quant params and per-tile sr from `accum_params` DRAM array:
  `[quant_scale, quant_zp, sr[0], sr[1], ...]`. This replaced the old stream-based
  parameter scatter (crossbar_load → accum_param_stream) to maintain one-reader-per-DRAM
  compliance after removing crossbar_load.
- Zero point subtraction: nest_pe reads iacts_zp and weights_zp directly from inst_pe
  (instructions[0,6] and instructions[1,6]), computing
  (iact - iacts_zp) * (weight - weights_zp) per PE, matching RTL feather_pe.v behavior.
  Set via create_gemm_program(iacts_zp=, weights_zp=).
- Post-quantization (int32 → uint8): output_accum reads quant_scale and quant_zp from
  accum_params[0] and accum_params[1].
  When quant_scale != 0, applies `(accum * quant_scale + quant_zp) & 255`, matching
  RTL quant_post.v formula `(sign_extend_64(data) * scale + zero_extend_64(zp))[7:0]`.
  Disabled by default (scale=0). Set via create_gemm_program(quant_scale=, quant_zp=).
- IVN/WVN layout orders (ORDER_012 through ORDER_210) control VN buffer memory layout.
  Uses corrected TABLE_II_OUTER_TO_INNER from RTL implementation (differs from original
  MINISA paper — see TICKET-011). Corrected encoding ensures same order index gives same
  structural loop nesting for IVN and WVN. Per-VN canonical dimensions:
  IVN=(jL1,mL0,mL1), WVN=(kL1,nL0,nL1), OVN=(pL1,pL0,qL1).
  In our direct-indexing model (no VN buffer), the crossbar routing is determined by
  Gr/Gc/sr/sc from SetMapping, so all 6 orders produce correct results. OVN order
  DOES affect computation via BIRRD butterfly routing (already implemented).
- Multi-layer sequential execution: `run_sequential_gemm_layers()` chains GEMM layers with
  int8 intermediates. Each non-final layer uses post-quantization (quant_scale != 0) to
  produce uint8 output, which is reinterpreted as int8 for the next layer's input. Matches
  RTL pipeline: OB → quant_post → StaB PONG write → next layer iActs read.
  Supports different dataflows per layer (e.g., OS layer 1 → passthrough layer 2).
- HLS scheduling (`schedule_feather_hls()`): minimal — only partitions C for parallel
  output writes. Parallelism is structural via mapping=[AH+2,AW], no pipeline/unroll needed.
- LLVM JIT compilation scales with PE count: 4×4 (16 PEs) compiles in ~8s, 8×8 (64 PEs)
  with multi-tile loops takes 15+ min. Acceptable for HLS synthesis (one-time cost).
- For cosim, use `mode="csyn"` to generate kernel.cpp, then manually patch m_axi depths
  and write C testbench (see `test_figure7_cosim.py` for pattern).
- Trace parser (`minisa/trace_parser.py`): `load_trace()` auto-detects 4 trace formats:
  (1) RTL trace ("layer" key) — parses ExecuteMapping entries, N-decomposition;
  (2) type="minisa" — simple GEMM params, maps to create_gemm_program();
  (3) type="minisa_manual" — explicit tile specifications (mixed Gr, sr=0);
  (4) type="minisa_multi_layer" — sequential GEMM layers with int8 intermediates.
  RTL trace: K-only tile decomposition (k_passes tiles) with temporal M/N folded into
  n_inner = n_spatial_tiles * n_m_batches * n_sub_tiles sub-operations per tile.
  Returns inner_m_starts/inner_n_starts arrays for per-sub-op DRAM lookup tables.
