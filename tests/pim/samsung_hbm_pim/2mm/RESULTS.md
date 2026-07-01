# 2mm on samsung_hbm_pim (SMALL)

- correctness: **PASS** -- samsung 2mm REDUCE chain final output matches the data-dependent composition within rtol=0.02 (max_abs_err=0.00031817)
- reference: PolyBench/C 4.2.1 SMALL_DATASET (validated by `experiments/scripts/R_polybench_ref_validation.py`)
- shapes: {'P': 40, 'R': 50, 'Q': 70, 'S': 80}
- cycles: 140313
- source: `PIMSimulator@bin-sha256:121250edea00`
- run_cmd: `python -m pytest tests/pim/samsung_hbm_pim/2mm/test_two_mm_samsung_hbm_pim.py -p no:cacheprovider -q`
- timestamp: 2026-06-30T12:33:15.784137
- tenon_commit: `837aab5ab7def05b5255b98cf66eee65362bb546`

Tier-1 two-stage GEMM chain; genuine fp16 PASS against the composed reference. | SPEC-04 cross-stage REDUCE chain (logical shape; fabric M padded to 4096, K to 256 per stage).
