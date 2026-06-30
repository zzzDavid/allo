# gemm on samsung_hbm_pim (SMALL)

- correctness: **PASS** -- samsung gemm output matches the contraction reference within rtol=0.02 (max_abs_err=7.67112e-05)
- reference: PolyBench/C 4.2.1 SMALL_DATASET (validated by `experiments/scripts/R_polybench_ref_validation.py`)
- shapes: {'P': 60, 'R': 70, 'Q': 80}
- cycles: 75205
- source: `PIMSimulator@bin-sha256:121250edea00`
- run_cmd: `python -m pytest tests/pim/samsung_hbm_pim/gemm/test_gemm_samsung_hbm_pim.py -p no:cacheprovider -q`
- timestamp: 2026-06-30T11:53:51.776912
- tenon_commit: `1af5ccd18a995ba016570ea802403affd91cad2a`

SPEC-05 SPMW slice-form gemm (mapping=[16,8]); genuine fp16 PASS vs A@B; partition mapping-driven (128 work-id buckets, per-PE slice loop bound = P//128).
