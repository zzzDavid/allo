# 2mm on apu_v1 (SMALL)

- correctness: **PASS** -- canonical uint16 MLIR -> scalar ARC C matched repository NumPy reference bit-for-bit for output (max_abs_err=0)
- reference: PolyBench/C 4.2.1 SMALL_DATASET (validated by `experiments/scripts/R_polybench_ref_validation.py`)
- shapes: {'P': 40, 'R': 50, 'Q': 70, 'S': 80}
- cycles: 27358199
- source: `apu_v1_device@zhang-capra-xcel.ece.cornell.edu/gsi-13.7.1`
- run_cmd: `python -m pytest tests/pim/apu_v1/2mm/test_two_mm_apu_v1.py::test_two_mm_apu_v1_scalar_baseline -p no:cacheprovider -q`
- timestamp: 2026-07-08T00:42:06.437560
- tenon_commit: `6bc7474a09fa7d2d64007e6b929b6f68660729d3`

Complete canonical uint16 PolyBench program lowered through MLIR to scalar C on APUC 0. This is the explicit scalar correctness baseline; other APUCs are not launched because arbitrary multi-phase programs do not yet carry cross-APUC barriers. Analytical scalar estimate=206199151 cycles.

## Four-APUC uint16 hybrid

- correctness: **PASS**, bit-exact modulo-2^16
- regions: `mm1:vector -> mm2:vector -> ele_add:scalar`
- precision conversions: 0
- persistent L4; host intermediate round trips: 0
- phase critical paths: 3,893,847 + 9,732,698 repack + 2,701,142 + 460,965 cycles
- total critical path: **16,788,652 cycles**
