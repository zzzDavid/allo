# gemm on samsung_hbm_pim (SMALL)

- correctness: **PASS** -- samsung gemm output matches the contraction reference within rtol=0.02 (max_abs_err=7.67112e-05)
- reference: PolyBench/C 4.2.1 SMALL_DATASET (validated by `experiments/scripts/R_polybench_ref_validation.py`)
- shapes: {'P': 60, 'R': 70, 'Q': 80}
- cycles: 75205
- source: `PIMSimulator@bin-sha256:121250edea00`
- run_cmd: `python -m pytest tests/pim/samsung_hbm_pim/gemm/test_gemm_samsung_hbm_pim.py -p no:cacheprovider -q`
- timestamp: 2026-06-30T15:58:05.054280
- tenon_commit: `837aab5ab7def05b5255b98cf66eee65362bb546`

SPMW slice-form GEMM on the 128-PE mapping; genuine fp16 PASS against A@B.
