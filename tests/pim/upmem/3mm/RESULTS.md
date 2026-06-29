# 3mm on upmem (SMALL)

- correctness: **PASS** -- uPIMulator GEMV host numeric check passed (W@x vs numpy ref, host tol); recorded ref rtol=0.0001
- reference: PolyBench/C 4.2.1 SMALL_DATASET (validated by `experiments/scripts/R_polybench_ref_validation.py`)
- shapes: {'P': 40, 'R': 50, 'Q': 60, 'T': 70, 'S': 80}
- cycles: 173490
- source: `uPIMulator@870d916334e9ff0b190f555f951a9ec3c4257781`
- run_cmd: `python -m pytest tests/pim/upmem/3mm/test_three_mm_upmem.py -p no:cacheprovider -q`
- timestamp: 2026-06-29T10:15:12.681764
- tenon_commit: `ed217633e316c4b45d19368b4cfadd61c028aa21`

Tier-1 single-output; UPMEM GEMV-host slot verifies W@x internally (PASS w/ cycles); a VA-slot route is CYCLES-ONLY.
