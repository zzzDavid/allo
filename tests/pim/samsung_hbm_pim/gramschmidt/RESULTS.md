# gramschmidt on samsung_hbm_pim (SMALL)

- correctness: **CYCLES-ONLY** -- samsung/gramschmidt: run path reports cycles only (no output array); computed at the GEMV design point
- reference: PolyBench/C 4.2.1 SMALL_DATASET (validated by `experiments/scripts/R_polybench_ref_validation.py`)
- shapes: {'M': 60, 'N': 80}
- cycles: 2813
- source: `PIMSimulator@bin-sha256:3f032694bfe1`
- run_cmd: `python -m pytest tests/pim/samsung_hbm_pim/gramschmidt/test_gramschmidt_samsung_hbm_pim.py -p no:cacheprovider -q`
- timestamp: 2026-06-29T11:14:48.995421
- tenon_commit: `ed217633e316c4b45d19368b4cfadd61c028aa21`

Tier-2 triangular; Samsung reports cycles only (no output array) -> CYCLES-ONLY at the GEMV design point; a shape the reference sim cannot express -> BLOCKED-SIM.
