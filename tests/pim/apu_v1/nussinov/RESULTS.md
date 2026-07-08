# nussinov on apu_v1 (SMALL)

- correctness: **PASS** -- canonical uint16 MLIR -> scalar ARC C matched repository NumPy reference bit-for-bit for table (max_abs_err=0)
- reference: PolyBench/C 4.2.1 SMALL_DATASET (validated by `experiments/scripts/R_polybench_ref_validation.py`)
- shapes: {'N': 180}
- cycles: 232178398
- source: `apu_v1_device@zhang-capra-xcel.ece.cornell.edu/gsi-13.7.1`
- run_cmd: `python -m pytest tests/pim/apu_v1/nussinov/test_nussinov_apu_v1.py -p no:cacheprovider -q`
- timestamp: 2026-07-08T00:40:39.454481
- tenon_commit: `6bc7474a09fa7d2d64007e6b929b6f68660729d3`

Complete canonical uint16 PolyBench program lowered through MLIR to scalar C on APUC 0. This is the explicit scalar correctness baseline; other APUCs are not launched because arbitrary multi-phase programs do not yet carry cross-APUC barriers. Analytical scalar estimate=7827831247 cycles.
