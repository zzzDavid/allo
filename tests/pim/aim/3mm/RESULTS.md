# 3mm on aim (SMALL)

- correctness: **CYCLES-ONLY** -- aim/3mm: ramulator2 trace sim -- N/A (no functional numerics)
- reference: PolyBench/C 4.2.1 SMALL_DATASET (validated by `experiments/scripts/R_polybench_ref_validation.py`)
- shapes: {'P': 40, 'R': 50, 'Q': 60, 'T': 70, 'S': 80}
- cycles: 666
- source: `ramulator2@0f28a07bdb83e42b9305ad3d45410ebd3aa2c091`
- run_cmd: `python -m pytest tests/pim/aim/3mm/test_three_mm_aim.py -p no:cacheprovider -q`
- timestamp: 2026-06-29T11:36:15.936532
- tenon_commit: `ed217633e316c4b45d19368b4cfadd61c028aa21`

Tier-1 single-output; AiM ramulator2 is a trace sim -> CYCLES-ONLY (no functional numerics).
