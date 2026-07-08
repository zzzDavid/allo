# 3mm on apu_v1 (SMALL)

- correctness: **PASS** -- canonical uint16 MLIR -> scalar ARC C matched repository NumPy reference bit-for-bit for output (max_abs_err=0)
- reference: PolyBench/C 4.2.1 SMALL_DATASET (validated by `experiments/scripts/R_polybench_ref_validation.py`)
- shapes: {'P': 40, 'R': 50, 'Q': 60, 'T': 70, 'S': 80}
- cycles: 48825932
- source: `apu_v1_device@zhang-capra-xcel.ece.cornell.edu/gsi-13.7.1`
- run_cmd: `python -m pytest tests/pim/apu_v1/3mm/test_three_mm_apu_v1.py::test_three_mm_apu_v1_scalar_baseline -p no:cacheprovider -q`
- timestamp: 2026-07-08T00:42:15.781831
- tenon_commit: `6bc7474a09fa7d2d64007e6b929b6f68660729d3`

Complete canonical uint16 PolyBench program lowered through MLIR to scalar C on APUC 0. This is the explicit scalar correctness baseline; other APUCs are not launched because arbitrary multi-phase programs do not yet carry cross-APUC barriers. Analytical scalar estimate=366428415 cycles.

## Four-APUC uint16 hybrid

- correctness: **PASS**, bit-exact modulo-2^16
- regions: independent `mm1`/`mm2`, join repack, then `mm3`
- precision conversions: 0
- persistent L4; host intermediate round trips: 0
- phase critical paths: 3,358,726 + 4,281,142 + 20,873,332 join + 2,700,893 + 363,308 gather cycles
- total critical path: **31,577,401 cycles**

The former compounded-FP16 out-of-tolerance result is obsolete. With native
uint16 GVML arithmetic, all outputs match the canonical modular reference.
