# symm on samsung_hbm_pim (SMALL)

- correctness: **BLOCKED-SIM** -- samsung_hbm_pim/symm: reference sim cored on the shape (env limit, not a Tenon gap): Samsung pim_driver returned but stdout missing 'PIM_CYCLES total=...' line; tail: 
malloc(): invalid size (unsorted)

- reference: PolyBench/C 4.2.1 SMALL_DATASET (validated by `experiments/scripts/R_polybench_ref_validation.py`)
- shapes: {'M': 60, 'N': 80}
- cycles: N/A
- source: `PIMSimulator@bin-sha256:3f032694bfe1 (cored on shape)`
- run_cmd: `python -m pytest tests/pim/samsung_hbm_pim/symm/test_symm_samsung_hbm_pim.py -p no:cacheprovider -q`
- timestamp: 2026-06-29T11:14:46.590606
- tenon_commit: `ed217633e316c4b45d19368b4cfadd61c028aa21`

Tier-2 triangular; Samsung reports cycles only (no output array) -> CYCLES-ONLY at the GEMV design point; a shape the reference sim cannot express -> BLOCKED-SIM.
