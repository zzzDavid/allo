# bicg on samsung_hbm_pim (SMALL)

- correctness: **CYCLES-ONLY** -- samsung/bicg: run path reports cycles only (no output array); computed at the GEMV design point
- reference: PolyBench/C 4.2.1 SMALL_DATASET (validated by `experiments/scripts/R_polybench_ref_validation.py`)
- shapes: {'M': 116, 'N': 124}
- cycles: 5626
- source: `PIMSimulator@bin-sha256:3f032694bfe1`
- run_cmd: `python -m pytest tests/pim/samsung_hbm_pim/bicg/test_bicg_samsung_hbm_pim.py -p no:cacheprovider -q`
- timestamp: 2026-06-29T10:59:00.203252
- tenon_commit: `ed217633e316c4b45d19368b4cfadd61c028aa21`

Tier-1 multi-output; Samsung reports cycles only (no functional readback on the faithful path) -> CYCLES-ONLY at the GEMV design point; shape the sim cannot express -> BLOCKED-SIM.
