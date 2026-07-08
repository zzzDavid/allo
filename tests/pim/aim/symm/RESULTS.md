# symm on aim (SMALL)

- correctness: **CYCLES-ONLY** -- aim/symm: ramulator2 trace sim -- N/A (no functional numerics)
- reference: PolyBench/C 4.2.1 SMALL_DATASET (validated by `experiments/scripts/R_polybench_ref_validation.py`)
- shapes: {'M': 60, 'N': 80}
- cycles: 71280
- source: `ramulator2@0f28a07bdb83e42b9305ad3d45410ebd3aa2c091`
- run_cmd: `python -m pytest tests/pim/aim/symm/test_symm_aim.py -p no:cacheprovider -q`
- timestamp: 2026-07-08T09:57:14.360509
- tenon_commit: `fc0ab691deb957bbb5b9193660d11615c0275283`

AiM-local SPMW workload; ramulator2 trace simulation; cycles only because the simulator exposes no functional numerics.
