# doitgen on samsung_hbm_pim (SMALL)

- correctness: **PASS** -- samsung doitgen output matches the contraction reference within rtol=0.02 (max_abs_err=2.80589e-05)
- reference: PolyBench/C 4.2.1 SMALL_DATASET (validated by `experiments/scripts/R_polybench_ref_validation.py`)
- shapes: {'Q': 20, 'R': 25, 'P': 30, 'S': 30}
- cycles: 34444
- source: `PIMSimulator@bin-sha256:121250edea00`
- run_cmd: `python -m pytest tests/pim/samsung_hbm_pim/doitgen/test_doitgen_samsung_hbm_pim.py -p no:cacheprovider -q`
- timestamp: 2026-06-30T01:03:22.417689
- tenon_commit: `ee1e52a539d6385a60d149ca238b592d9763103e`

Tier-1 single-output; Samsung reports cycles only (no output array) -> CYCLES-ONLY at the GEMV design point; a shape the reference sim cannot express -> BLOCKED-SIM.
