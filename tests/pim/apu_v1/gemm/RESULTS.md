# gemm on apu_v1 (SMALL)

- correctness: **PASS** -- canonical uint16 MLIR -> scalar ARC C matched repository NumPy reference bit-for-bit for output (max_abs_err=0)
- reference: PolyBench/C 4.2.1 SMALL_DATASET (validated by `experiments/scripts/R_polybench_ref_validation.py`)
- shapes: {'P': 60, 'R': 70, 'Q': 80}
- cycles: 30741941
- source: `apu_v1_device@zhang-capra-xcel.ece.cornell.edu/gsi-13.7.1`
- run_cmd: `python -m pytest tests/pim/apu_v1/gemm/test_gemm_apu_v1.py::test_gemm_apu_v1_scalar_baseline -p no:cacheprovider -q`
- timestamp: 2026-07-08T00:38:36.312164
- tenon_commit: `6bc7474a09fa7d2d64007e6b929b6f68660729d3`

Complete canonical uint16 PolyBench program lowered through MLIR to scalar C on APUC 0. This is the explicit scalar correctness baseline; other APUCs are not launched because arbitrary multi-phase programs do not yet carry cross-APUC barriers. Analytical scalar estimate=230503639 cycles.

## Four-APUC uint16 hybrid

- correctness: **PASS**, bit-exact modulo-2^16
- regions: `mm1:vector -> ele_add:scalar`
- GVML: `gvml_mul_u16` + `gvml_add_u16`
- precision conversions: 0
- vector APUCs: 0, 1, 2, 3
- persistent L4; host intermediate round trips: 0
- vector critical path: 4,419,068 cycles
- gather + scalar epilogue: 691,572 cycles
- total critical path: **5,110,640 cycles**
