# 2mm on apu_v1 (SMALL)

- correctness: **PASS** -- canonical uint16 MLIR -> scalar ARC C matched repository NumPy reference bit-for-bit for output (max_abs_err=0)
- reference: PolyBench/C 4.2.1 SMALL_DATASET (validated by `experiments/scripts/R_polybench_ref_validation.py`)
- shapes: {'P': 40, 'R': 50, 'Q': 70, 'S': 80}
- cycles: 27359120
- source: `apu_v1_device@zhang-capra-xcel.ece.cornell.edu/gsi-13.7.1`
- run_cmd: `python -m pytest tests/pim/apu_v1/2mm/test_two_mm_apu_v1.py::test_two_mm_apu_v1_scalar_baseline -p no:cacheprovider -q`
- timestamp: 2026-07-08T17:45:34.752300
- tenon_commit: `a2586d1b0aa59a7b477cf91402dec06daa3f83be`

Complete canonical uint16 PolyBench program lowered through MLIR to scalar C on APUC 0. This is the explicit scalar correctness baseline; other APUCs are not launched because arbitrary multi-phase programs do not yet carry cross-APUC barriers. Analytical scalar estimate=206199151 cycles.
