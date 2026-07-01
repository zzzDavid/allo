# gesummv on upmem (SMALL)

- correctness: **PASS** -- uPIMulator GEMV host numeric check passed (W@x vs numpy ref, host tol); recorded ref rtol=0.0001
- reference: PolyBench/C 4.2.1 SMALL_DATASET (validated by `experiments/scripts/R_polybench_ref_validation.py`)
- shapes: {'N': 90}
- cycles: 326445
- source: `uPIMulator@870d916334e9ff0b190f555f951a9ec3c4257781`
- run_cmd: `python -m pytest tests/pim/upmem/gesummv/test_gesummv_upmem.py -p no:cacheprovider -q`
- timestamp: 2026-06-29T12:22:21.961398
- tenon_commit: `ee1e52a539d6385a60d149ca238b592d9763103e`

Tier-1 single-output; UPMEM GEMV-host slot verifies W@x internally (PASS w/ cycles); a VA-slot route is CYCLES-ONLY.
