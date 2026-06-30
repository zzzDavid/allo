# bicg on samsung_hbm_pim (SMALL)

- correctness: **PASS** -- samsung bicg REDUCE chain final output matches the data-dependent composition within rtol=0.02 (max_abs_err=0.000190765)
- reference: PolyBench/C 4.2.1 SMALL_DATASET (validated by `experiments/scripts/R_polybench_ref_validation.py`)
- shapes: {'M': 116, 'N': 124}
- cycles: 8870
- source: `PIMSimulator@bin-sha256:121250edea00`
- run_cmd: `python -m pytest tests/pim/samsung_hbm_pim/bicg/test_bicg_samsung_hbm_pim.py -p no:cacheprovider -q`
- timestamp: 2026-06-30T11:48:11.313048
- tenon_commit: `1af5ccd18a995ba016570ea802403affd91cad2a`

Tier-1 multi-output; Samsung reports cycles only (no functional readback on the faithful path) -> CYCLES-ONLY at the GEMV design point; shape the sim cannot express -> BLOCKED-SIM. | SPEC-04 cross-stage REDUCE chain (logical shape; fabric M padded to 4096, K to 256 per stage).
