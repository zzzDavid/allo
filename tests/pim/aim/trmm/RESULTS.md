# trmm on aim (SMALL)

- correctness: **CYCLES-ONLY** -- aim/trmm: ramulator2 trace sim -- N/A (no functional numerics)
- reference: PolyBench/C 4.2.1 SMALL_DATASET (validated by `experiments/scripts/R_polybench_ref_validation.py`)
- shapes: {'M': 60, 'N': 80}
- cycles: 232
- source: `ramulator2@0f28a07bdb83e42b9305ad3d45410ebd3aa2c091`
- run_cmd: `python -m pytest tests/pim/aim/trmm/test_trmm_aim.py -p no:cacheprovider -q`
- timestamp: 2026-06-29T11:14:44.722772
- tenon_commit: `ed217633e316c4b45d19368b4cfadd61c028aa21`

Tier-2 triangular; AiM ramulator2 is a trace sim -> CYCLES-ONLY (no functional numerics).
