# covariance on samsung_hbm_pim (SMALL)

- correctness: **PASS** -- samsung covariance output matches the contraction reference within rtol=0.02 (max_abs_err=0.000112414)
- reference: PolyBench/C 4.2.1 SMALL_DATASET (validated by `experiments/scripts/R_polybench_ref_validation.py`)
- shapes: {'M': 80, 'N': 100}
- cycles: 85604
- source: `PIMSimulator@bin-sha256:121250edea00`
- run_cmd: `python -m pytest tests/pim/samsung_hbm_pim/covariance/test_covariance_samsung_hbm_pim.py -p no:cacheprovider -q`
- timestamp: 2026-06-30T11:50:08.109015
- tenon_commit: `1af5ccd18a995ba016570ea802403affd91cad2a`

mean/centering/normalize/symmetrize host-side; on-device = cdata^T@cdata (GEMM nest). GEMM-shaped emitted stream -> CYCLES-ONLY (no W@x match); a shape the reference sim cannot express -> BLOCKED-SIM.
