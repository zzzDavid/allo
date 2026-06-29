# gemver on samsung_hbm_pim (SMALL)

- correctness: **CYCLES-ONLY** -- samsung/gemver: run path reports cycles only (no output array); computed at the GEMV design point
- reference: PolyBench/C 4.2.1 SMALL_DATASET (validated by `experiments/scripts/R_polybench_ref_validation.py`)
- shapes: {'N': 120}
- cycles: 11252
- source: `PIMSimulator@bin-sha256:3f032694bfe1`
- run_cmd: `python -m pytest tests/pim/samsung_hbm_pim/gemver/test_gemver_samsung_hbm_pim.py -p no:cacheprovider -q`
- timestamp: 2026-06-29T10:58:55.862107
- tenon_commit: `ed217633e316c4b45d19368b4cfadd61c028aa21`

Tier-1 multi-output; Samsung reports cycles only (no functional readback on the faithful path) -> CYCLES-ONLY at the GEMV design point; shape the sim cannot express -> BLOCKED-SIM.
