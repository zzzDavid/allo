# trisolv on upmem (SMALL)

- correctness: **PASS** -- canonical MLIR -> portable C matched repository NumPy reference for x (max_abs_err=0, rtol=0.0001, atol=0.0001); ABI packed and gathered a full 64-DPU rank
- reference: examples/polybench/trisolv.py::trisolv_np SMALL_DATASET (validated by `tests/pim/test_upmem_polybench_registry.py`)
- shapes: {'N': 120}
- cycles: 18568737
- source: `upmem_cost@1e9f869de726a65f (analytical MLIR operation graph); portable-C functional oracle`
- run_cmd: `python -m pytest tests/pim/upmem/trisolv/test_trisolv_upmem.py -p no:cacheprovider -q`
- timestamp: 2026-07-01T09:39:40.009385
- tenon_commit: `6bc7474a09fa7d2d64007e6b929b6f68660729d3`

The complete canonical Allo kernel is lowered through MLIR to portable C. Its NumPy-visible results are checked against the repository reference; the UPMEM ABI uses 64 DPUs and retains the declarative partition, barrier, collective, temporal, pivot, or wavefront orchestration plan. Reported cycles are analytical cost-program estimates, not simulator measurements. Tensor MRAM ownership and analytical DPU/tasklet fanout are derived from the compiled F2 LinearLayout.
