# vector_gemm on APU v1

Full FP16 layout milestone: `M=1024`, `N=1024`, `K=64`, one APUC.

| plan | analytical cycles | measured CRUN | max absolute error |
|---|---:|---:|---:|
| temporal + DMA + broadcast-friendly | 8,552,316 | 7,349,987 | 2.94e-4 |
| temporal + DMA coalescing | 93,711,488 | 93,449,834 | 2.81e-4 |
| baseline spatial reduction | 7,785,192,064 | 6,254,825,903 | 9.81e-5 |
| temporal SVP | 7,789,201,728 | 6,258,832,854 | 3.60e-4 |

- correctness: **PASS** for all 1,048,576 outputs in every plan.
- predicted ranking: broadcast-friendly < coalesced < baseline < temporal SVP.
- measured ranking: broadcast-friendly < coalesced < baseline < temporal SVP.
- measured broadcast-friendly speedup over coalesced DMA: **12.71x**.
- source: `apu_v1_device@zhang-capra-xcel.ece.cornell.edu/gsi-13.7.1`.
- timestamp: `2026-07-07T17:29:24-04:00`.

The four candidates come from the same ordinary Allo contraction. Compact LHS
lookup tables, resident RHS VR banks, subgroup expansion, transfer counts, and
reuse are represented by explicit layout relations shared by the ABI, GVML
lowering, and executable cost model. The spatial baseline currently supports a
zero-initialized accumulator and rejects nonzero `C` rather than silently
producing incorrect results.
