# trmm on aim (SMALL)

- correctness: **CYCLES-ONLY** -- aim/trmm: ramulator2 trace sim -- N/A (no functional numerics)
- reference: PolyBench/C 4.2.1 SMALL_DATASET (validated by `experiments/scripts/R_polybench_ref_validation.py`)
- shapes: {'M': 60, 'N': 80}
- cycles: 7234
- source: `ramulator2@0f28a07bdb83e42b9305ad3d45410ebd3aa2c091`
- run_cmd: `python -m pytest tests/pim/aim/trmm/test_trmm_aim.py -p no:cacheprovider -q`
- timestamp: 2026-06-30T18:12:18.474541
- tenon_commit: `837aab5ab7def05b5255b98cf66eee65362bb546`

AiM-local SPMW workload; ramulator2 trace simulation; cycles only because the simulator exposes no functional numerics.
