# atax on aim (SMALL)

- correctness: **CYCLES-ONLY** -- aim/atax: ramulator2 trace sim -- N/A (no functional numerics)
- reference: PolyBench/C 4.2.1 SMALL_DATASET (validated by `experiments/scripts/R_polybench_ref_validation.py`)
- shapes: {'M': 116, 'N': 124}
- cycles: 679
- source: `ramulator2@0f28a07bdb83e42b9305ad3d45410ebd3aa2c091`
- run_cmd: `python -m pytest tests/pim/aim/atax/test_atax_aim.py -p no:cacheprovider -q`
- timestamp: 2026-06-29T11:36:16.947541
- tenon_commit: `ed217633e316c4b45d19368b4cfadd61c028aa21`

Tier-1 multi-output; AiM ramulator2 is a trace sim -> CYCLES-ONLY (no functional numerics).
