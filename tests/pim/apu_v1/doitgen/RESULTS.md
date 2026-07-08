# doitgen on apu_v1 (SMALL)

- correctness: **PASS** -- canonical uint16 MLIR -> scalar ARC C matched repository NumPy reference bit-for-bit for A, sum_ (max_abs_err=0)
- reference: PolyBench/C 4.2.1 SMALL_DATASET (validated by `experiments/scripts/R_polybench_ref_validation.py`)
- shapes: {'Q': 20, 'R': 25, 'P': 30, 'S': 30}
- cycles: 44754196
- source: `apu_v1_device@zhang-capra-xcel.ece.cornell.edu/gsi-13.7.1`
- run_cmd: `python -m pytest tests/pim/apu_v1/doitgen/test_doitgen_apu_v1.py -p no:cacheprovider -q`
- timestamp: 2026-07-08T00:43:24.584256
- tenon_commit: `6bc7474a09fa7d2d64007e6b929b6f68660729d3`

Complete canonical uint16 PolyBench program lowered through MLIR to scalar C on APUC 0. This is the explicit scalar correctness baseline; other APUCs are not launched because arbitrary multi-phase programs do not yet carry cross-APUC barriers. Analytical scalar estimate=311400507 cycles.
