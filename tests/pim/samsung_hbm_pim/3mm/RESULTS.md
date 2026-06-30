# 3mm on samsung_hbm_pim (SMALL)

- correctness: **PASS** -- samsung 3mm REDUCE chain final output matches the data-dependent composition within rtol=0.02 (max_abs_err=0.00113106)
- reference: PolyBench/C 4.2.1 SMALL_DATASET (validated by `experiments/scripts/R_polybench_ref_validation.py`)
- shapes: {'P': 40, 'R': 50, 'Q': 60, 'T': 70, 'S': 80}
- cycles: 205119
- source: `PIMSimulator@bin-sha256:121250edea00`
- run_cmd: `python -m pytest tests/pim/samsung_hbm_pim/3mm/test_three_mm_samsung_hbm_pim.py -p no:cacheprovider -q`
- timestamp: 2026-06-30T11:48:06.807163
- tenon_commit: `1af5ccd18a995ba016570ea802403affd91cad2a`

Tier-1 single-output; Samsung reports cycles only (no output array) -> CYCLES-ONLY at the GEMV design point; a shape the reference sim cannot express -> BLOCKED-SIM. | SPEC-04 cross-stage REDUCE chain (logical shape; fabric M padded to 4096, K to 256 per stage).
