# syr2k on aim (SMALL)

- correctness: **CYCLES-ONLY** -- aim/syr2k: ramulator2 trace sim -- N/A (no functional numerics)
- reference: PolyBench/C 4.2.1 SMALL_DATASET (validated by `experiments/scripts/R_polybench_ref_validation.py`)
- shapes: {'M': 60, 'N': 80}
- cycles: 439
- source: `ramulator2@0f28a07bdb83e42b9305ad3d45410ebd3aa2c091`
- run_cmd: `python -m pytest tests/pim/aim/syr2k/test_syr2k_aim.py -p no:cacheprovider -q`
- timestamp: 2026-06-29T10:14:44.835373
- tenon_commit: `ed217633e316c4b45d19368b4cfadd61c028aa21`

Tier-1 single-output; AiM ramulator2 is a trace sim -> CYCLES-ONLY (no functional numerics).
