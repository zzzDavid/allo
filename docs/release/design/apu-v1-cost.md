# APU v1 (Gemini-I): cost provenance, validity, tile selection, scalar sqrt

Status: describes the code after the V1-05, V1-06, V1-11 and V1-12 repairs
(audit `docs/release/audit/apu-v1.md`). All numbers below were produced
compile-only on this host on 2026-11-02 (no board launch); measured values are
the archived device results named in each table.

## 1. The fitted affine event model (V1-05)

`allo/pim/costs/apu_v1.py` now carries the model fitted by the paper's
agent-assisted calibration (`APU_V1_AFFINE_EVENT_MODEL`). For one launch of a
native recipe, every GVML primitive family is one aggregated graph event
(`APUv1NativeCostGraphBuilder`, `native_aggregate=True`), and

```
event_cycles = round(activation + slope * count)      (0 when count == 0)
route        = route_intercept + sum(event_cycles)
l4_l3        = round(activation_l4_l3 * transactions + per_byte * bytes)
```

The rounding happens once per aggregated event, exactly as in the frozen
study CostSpec. Units: ARC CRUN device cycles (the `PROF total` counter, no
clock conversion).

| Term | Activation (cycles) | Slope | Microbenchmarks (low, high count) | Medians incl. intercept |
|---|---:|---:|---|---|
| route intercept (empty route) | 2,558 | | q00-empty | 2,558 |
| L4->L3 query image (`DMA_L4_TO_L3`) | 1,842.33 | 0.7767 / byte | q01-l4-l3-b1, q02-l4-l3-b4 | |
| `direct_dma_l4_to_l1_32k` | 8,477.00 | 21,808.00 / call | q03 (1), q04 (4) | 32,843 / 98,267 |
| `gvml_load_16` | 1,871.37 | 25.63 / call | q05 (1), q06 (64) | 4,455 / 6,070 |
| `gvml_eq_imm_16` | 1,880.70 | 24.30 / call | q07 (1), q08 (256) | 4,463 / 10,659 |
| `gvml_and_m` | 908.79 | 27.21 / call | q09 (1), q10 (192) | 3,494 / 8,691 |
| `gvml_2_fast_count_m_g32k` (paper's count rule) | **5,887.79** | **876.21** / call | q11 (1), q12 (64) | 9,322 / 64,523 |
| `scalar_l4_store_u16` | 1,235.13 | 17.87 / call | q13 (1), q14 (128) | 3,811 / 6,081 |

Provenance (also in `APU_V1_AFFINE_EVENT_MODEL.provenance`):

* Coefficients copied at full precision from
  `tenon-artifacts/agent-cost-calibration/2026-07-27-gemini-i/confirmatory-affine-event-v1/freeze/affine-event-model.json`
  (model fingerprint `0f26de9d...dcfef`, algorithm `paired-affine-event-v1`);
  the rebuttal transfer studies use identical coefficients
  (`frozen-affine-model.json`, sha256 `f89d8de6...be16`). A test checks
  equality against the archive when it is present.
* Measurements: 15 single-primitive microbenchmarks,
  `negative-pilot-c4q96-v3/measurements/calibration/q00..q14*.json`, one
  warm-up plus seven runs each, median CRUN, measured 2026-07-27
  22:31-22:57 UTC.
* Board: GSI Gemini-I, PCI `1e4c:0010` rev 00 on `zhang-capra-xcel`, APUC 0
  (platform fingerprint `1c28bbaf...3974`). Software: GSI APU SDK 13.7.1,
  ARC GNU 2021.09, `ledag-ssh` 100.12.0.1.1000.25. The board firmware version
  is not recorded in the archive.
* The 15-observation fit has 15 parameters (saturated interpolation); the
  pairs were not refitted for any transfer study below.

Scope of pricing: `priced_routes` = the native record-frequency route
(`gvml_record_frequency_resident_chunks_pair_count`) and the dense histogram
route (`gvml_dense_histogram_pair_count`). All other events (vector
contraction plans, moments, squared-L2 argmin, scalar ARC, the odd-count
`gvml_count_m_g32k`) keep the per-call coefficients documented in the same
module. Legality, lowering and emitted GVML are unchanged; only estimates move.

### 1.1 Validity label

`apu_v1_cost_validity(route, inventory, cost)` labels every APU v1 estimate;
the label is stored in `graph.metadata["cost_validity"]` and
`graph.metadata["estimate_validity"]` (as on Gemini-II) and in
`RunResult.extra["cost_validity"]` for virtual/functional runs.

* `calibrated`: the cost program carries `apu_v1_calibration`, the route is a
  validated family, no event lacks a fitted pair, and the shape is inside the
  validated envelope: record frequency with 1 <= chunks <= 9 and
  16 <= queries <= 160, even queries; histogram with 256 bins (even) and 1 to
  128 frontier loads.
* `extrapolated` otherwise, with the reasons (`basis`): other routes (vector
  contraction plans, moments, argmin, scalar ARC), odd pair counts
  (`gvml_count_m_g32k` has no fitted pair), shapes outside the envelope, or a
  cost program without the calibration record.

### 1.2 Accuracy on the archived plans (re-evaluated, not refitted)

"Before" is the released program with the fitted model disabled (identical to
the archived `frozen_uncalibrated` predictions); "after" is the released
program. Record frequency compiles through `allo.compile` (each register-offset
pair shares one inventory, so one estimate); histogram residency plans are
evaluated through `APUv1NativeCostGraphBuilder` with the archived inventories,
because the released histogram recipe only materializes the one-load plan
(block 256, whose public estimate is the same 166,694).

| Family (archive) | Plans | MAPE before | MAPE after | Paper |
|---|---:|---:|---:|---|
| Record frequency (`2026-09-29-gemini-calibration-v4`) | 8 | 7.41% | 1.23% | 7.41 -> 1.23 |
| Histogram (`2026-09-29-gemini-histogram-verification-v1`) | 8 | 25.92% | 0.41% | 25.92 -> 0.41 |
| Confirmatory record frequency (2026-07-27) | 3 | 10.67% | 1.29% | 10.67 -> 1.29 |

The port reproduces every frozen prediction to the cycle (for example
c1-q128 107,459; histogram block 2 2,939,566, block 256 166,694), so there is
no gap to the paper. Worst single cases after calibration: record frequency
+2.86% (c1-q128-vr8), histogram -0.92% (block 32). Paper kernels on the
default program: word count (C3, Q128) 165,618 vs measured 171,988 (-3.7%,
was -18.6%), histogram 166,694 vs 166,783 (-0.05%, was -56.5%); both
`calibrated`. Linear regression (54,299 vs 94,618.5) and k-means (643,130 vs
624,677) are unchanged and `extrapolated`; pricing the moments route with the
same model would move it to 65,781 (-30.5%), which is outside the validated
families and left as a decision.

## 2. Tile selection (V1-06)

The spatial GEMV recipe's reduction tile is a ranked decision.
`gemv_reduction_tile_domain` (`allo/pim/apu_v1_vectorize.py`) enumerates the
legal tiles: powers of two from 32 to 2048 (capped at the reduction extent),
padded width dividing the 32K VR lanes, at most 10 resident reduction tiles;
L3/VR capacity is re-checked when each plan is realized. Each tile is a
catalog candidate `spatial_gemv_group_reduction_k<tile>`, realized and
ranked by the cost program like every other plan (ties keep the former
heuristic order: fewest matrix tiles, fewest resident tiles, widest tile).
`layout="spatial_gemv_group_reduction_k64"` forces one tile;
`layout="spatial_gemv_group_reduction"` keeps the tile cost-ranked within the
family. Each plan's `metadata["tile_decision"]` and its estimate graph record
the decision; `APUv1VectorCallable.tile_decision` lists every tile with its
estimate (or `None` when not realizable) and the selected tile, and
functional/virtual runs return it in `RunResult.extra["tile_decision"]`.

The matcher route (`_apu_v1_enumerate`) has no tile decision: its group size
is the authored `mapping`, so it was not changed.

### 2.1 Paper GEMV (FP16 4096x256 times 256x1, reassociation permitted)

| Tile | Estimate (cycles) | Measured median (archive) | Error |
|---:|---:|---:|---:|
| 256 (selected) | 1,424,455 | 1,244,229 | +14.5% |
| 128 | 1,435,492 | 1,258,976 | +14.0% |
| 64 | 1,467,902 | 1,292,070 | +13.6% |
| 32 | 1,545,490 | 1,368,885 | +12.9% |

Measured: `reviews/experiments/results/2026-09-29-common-gemv-owner-v2/apu-runs-v2/results.json`
(two runs per tile). The cost program predicts the measured order and
tile 64 3.05% slower than tile 256 (measured 3.845%); the default selects
tile 256, the measured fastest. The paper's 3.845% is the loss of the shared
unit-weight movement objective, which ranked tiles 64 and 32 first
(777,728 movement units each) before cost ranking; the released route ranks
all four tiles by cost. These are vector-route estimates: `extrapolated`.
Under the default `reduction_order="strict"` all four GEMV tiles are pruned
(the group reduction regroups the FP16 sum).

### 2.2 FP16 1024x64x1024 GEMM (12.71x row)

Catalog: `baseline_spatial_reduction` (pruned under strict), `temporal_svp`,
`temporal_dma_coalescing`, `temporal_dma_coalescing_broadcast_friendly`,
`..._acc{2,4,8}` and `..._acc{2,4,8}_l4`. No tile family (the second output
axis is not a singleton). Realizable and ranked (strict):

| Plan | Estimate | Measured |
|---|---:|---:|
| `..._broadcast_friendly_acc2` (default winner) | 6,553,468 | not measured |
| `temporal_dma_coalescing_broadcast_friendly` (row "adapted") | 8,552,316 | 7,349,987 (paper), 7,145,109 (2026-10-04 golden) |
| `..._broadcast_friendly_acc2_l4` | 9,291,112 | not measured |
| `temporal_dma_coalescing` (row "baseline") | 93,711,488 | 93,449,834 |
| `temporal_svp` | 7,789,201,728 | |

`acc4`/`acc8` (both storages) are not realizable (they need 3/7 more
writable VRs). The default winner differs from the row's plan: the row is a
fixed-mechanism ablation (coalesced DMA, then broadcast-friendly transfer)
measured before the accumulator-block variants entered the catalog;
`acc2` (`accumulator_block=2`) keeps two output-batch accumulator VRs live,
so each broadcast operand serves two output batches; the cost program prices
it 23% below the row's plan. That
ranking is unvalidated on the device. Both row plans remain reachable with
`layout=`.

## 3. Public surface (V1-11)

Removed: `APUv1PrecisionPolicy` from `allo` and `allo.pim` exports and from
`apu_v1_program.__all__` (the class remains internal to `apu_v1_hybrid`,
whose manifest still reads it); `APUv1Phase.precision_policy`,
`vector_layout`, `produces`, `bindings`, `dependencies`, `zero_initialize`
(passing them is a `TypeError`). `APUv1Program` accepts exactly one phase
(multi-phase raises `ValueError` at construction instead of at compile).
`APUv1Phase` keeps `kernel`, `name`, `instantiate`, `result_names`,
`vectorize`, `argument_bounds`.

## 4. Scalar-route sqrt (V1-12)

The device link has no libm, so `sqrt(`/`sqrtf(` in the portable C are
redirected to `allo_sqrt` (double) and `allo_sqrtf` (rounds `allo_sqrt`)
(`APU_V1_SQRT_HELPERS` in `allo/pim/apu_v1_program.py`): IEEE special cases
(+-0, +inf, NaN, negative -> NaN), subnormals scaled by 2^54, an
exponent-halved initial guess within a factor of two, then Newton until the
iterate stops decreasing (at most 6 steps needed; bound 10). Host checks: at
most 1 ulp from the C library on 20,000 random doubles across the full range
and at both ends (smallest subnormal, largest finite); the float helper equals
NumPy's correctly rounded `float32` sqrt on 20,000 random floats; 1e10 and 1e12
(the audit's 0.74 and 14.3 relative errors) are exact. The double path no
longer truncates to float. A release-mode ARC project with the helper builds
and links (no launch).

## 5. What the device run must verify

1. Word count (C3, Q128) and histogram (N32768, B256) medians against the new
   `calibrated` estimates 165,618 and 166,694 (the paper golden rows record
   171,988 and 166,783).
2. The FP16 GEMM default winner `temporal_dma_coalescing_broadcast_friendly_acc2`
   (estimate 6,553,468): correctness of all 1,048,576 outputs and its median,
   compared with the broadcast-friendly plan (golden 7,145,109). If it is not
   faster, the accumulator-block pricing needs a microbenchmark.
3. The paper GEMV (reassociate): the default `..._k256` plan builds, runs and
   is the fastest of the four tiles on the current tree.
4. A scalar-route program using `allo.sqrt` on float32 and float64 inputs up
   to 1e12 returns the host results (device helper vs host libm).
