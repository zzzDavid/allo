# gemm on aim (SMALL)

- correctness: **CYCLES-ONLY** -- aim/gemm: ramulator2 trace sim -- N/A (no functional numerics)
- reference: PolyBench/C 4.2.1 SMALL_DATASET (validated by `experiments/scripts/R_polybench_ref_validation.py`)
- shapes: {'P': 60, 'R': 70, 'Q': 80}
- cycles: 272
- source: `ramulator2@0f28a07bdb83e42b9305ad3d45410ebd3aa2c091`
- run_cmd: `python -m pytest tests/pim/aim/gemm/test_gemm_aim.py -p no:cacheprovider -q`
- timestamp: 2026-06-29T11:36:19.615391
- tenon_commit: `ed217633e316c4b45d19368b4cfadd61c028aa21`

Tier-1 single-output; AiM ramulator2 is a trace sim -> CYCLES-ONLY (no functional numerics).
