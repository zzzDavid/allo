# syr2k on samsung_hbm_pim (SMALL)

- correctness: **CYCLES-ONLY** -- samsung/syr2k: run path reports cycles only (no output array); computed at the GEMV design point
- reference: PolyBench/C 4.2.1 SMALL_DATASET (validated by `experiments/scripts/R_polybench_ref_validation.py`)
- shapes: {'M': 60, 'N': 80}
- cycles: 5626
- source: `PIMSimulator@bin-sha256:3f032694bfe1`
- run_cmd: `python -m pytest tests/pim/samsung_hbm_pim/syr2k/test_syr2k_samsung_hbm_pim.py -p no:cacheprovider -q`
- timestamp: 2026-06-29T10:56:36.588351
- tenon_commit: `ed217633e316c4b45d19368b4cfadd61c028aa21`

Tier-1 single-output; Samsung reports cycles only (no output array) -> CYCLES-ONLY at the GEMV design point; a shape the reference sim cannot express -> BLOCKED-SIM.
