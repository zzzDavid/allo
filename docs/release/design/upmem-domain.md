# UPMEM backend: supported domain, legality, initialization and cost scope

Status: describes the code after the U-03/U-04/U-05/U-06/U-09/U-10/U-11..U-15
repair (audit `docs/release/audit/upmem.md`). Executed numbers are from
uPIMulator via the release route (`allo.compile(..., build_upmem_target(),
upmem_cost)`, `GENERIC` fixture, 12 tasklets, 350 MHz logic clock) on this
host, 2026-11-02.

## 1. Route and what each family proves

`spmw_upmem.classify_upmem_kernel` sees one kernel group's `MatchedOp`s, one
stream per work id (`@allo.work(mapping=[N])`, one DPU per work id). A family
recognizer proves the complete stored semantics of the stream: operation,
operand roles and indices, element types (`UPMEM_ROLE_ELEMENT_TYPES`), literal
coefficients and predicate, and the accumulator's initial value. Nothing
dispatches on names, shapes or benchmark identity. A program the recognizer
cannot prove is rejected with the reason.

The legality of a physical decision is **the template's**: `_realize(kernel,
decision)` runs the physical model's structural checks
(`upmem_physical_search.physical_features`: decision fields that belong to the
family, DMA sizes and alignment, WRAM) and the plan constructor's own
preconditions. An extent that does not divide a tile, chunk or tasklet count
is never a legality condition. The template zero-pads and masks it, and the
cost features describe the padded program. The enumerator only fans out
realizable catalog decisions. When none is realizable it raises with every
decision's reason; it no longer returns "no candidate layouts produced".

### 1.1 Per-family legality and padding

| Family | Recognized source form | Padded (zeros, result masked or discarded) | Rejected (template constraint named in the error) | Uneven work items | `carried` accumulator |
|---|---|---|---|---|---|
| `pointwise` | `out[i] = x[i] + y[i]`, `x - y`, `x * y`, `a * x`, `a * x + b * y`, `a * x - b * y`, `a * x + b * y + c` (matched ADD/SUB/MUL/SCALE/AXPBY/AXMBY/AXPBYC, int32 literals) | N to whole DMA chunks; the template masks `element >= N` | DMA chunk outside 8..2048 B or not 8-B aligned (decision); fused packet needs two operands (`a * x` streams one) | yes, `elements` | no accumulator |
| `reduction` | `out[c] += in[i]` (int32 into int64) | N to `12 * pow2(ceil(chunks/12))` chunks; masked `element >= N` | none beyond MRAM capacity | yes, `elements` | yes: tasklet 0 loads the cell before the merge |
| `matrix_vector`, fused packets | `y[o] += M[o, k] * v[k]` (rows flattened over outer loops), MMTV `C[b, r] += A[b, r, k] * B[b, k]`, scaled `y = alpha * acc` | K to whole chunks; rows to `12 * max(2, pow2(ceil(rows/12)))` (padded rows fetch nothing and write 0) | `2 * chunk` packet > 2048 B; per-tasklet output group (`rows_per_tasklet * 4`) > 2048 B, i.e. more than 6144 rows | yes, `rows` (batches = 1) or `batches` | yes (unscaled form): each tasklet loads its output group first |
| `matrix_vector`, tasklet-private WRAM | same; any batch count (tasklet `t` owns batches `t, t+12, ...`) | K to whole chunks; rows per batch to an even count (8-B output DMA) | per-batch output DMA > 2048 B (rows per batch > 512); WRAM `12 * (Kpad*4 + dma + out) + 12 KiB stacks + 1 KiB > 64 KiB` (K above about 1056 at small rows) | as above | yes |
| `matrix_vector`, shared WRAM | in the catalog, no template | never enumerated: "MV lowering does not materialize the requested residency" | | | |
| `gemm`, fused RHS packets | `C[i, j] += A[i, k] * B[k, j]` | columns to Nc tiles, K to Kc tiles, rows to `12 * pow2(ceil(rows/12))` | packet `(Nc+1) * Kc * 4` > 2048 B or output tile < 8 B (decision); separate/shared-WRAM entry has no template (never enumerated) | yes, `rows` | yes: each output tile is loaded before its reduction |
| `histogram` | `H[(x[i] * m) >> k] += 1`, `m == bins`, guard empty or exactly `0 <= bin < bins` | none needed (tasklet ranges clamp to N) | `depth >= 31`; WRAM `4 * 12 * bins + ...` > 64 KiB; any other index/guard form | **no**: a zero-padded element would be counted in bin `(0 * m) >> k` | yes: the merge starts from the initial histogram |
| `selection` | compaction form (`COMPACT` + `INC` under one guard) with guard `(x & mask) cmp k` or `x cmp k`, `cmp` in `== != < <= > >=`, signed or unsigned, int32 literals; flags form `SELECT_ODD` (odd only: that is the target op) | N to `12 * dma_elements` multiples; padded flags are discarded | a guard of any other form ("the selection plan supports '(x & mask) cmp k' and 'x cmp k' ...") | yes, `elements` | yes, count: the host compaction appends after the initial count (`out[c0 : c0 + k]`) |
| `kmeans_distances` | `D[p, c] += (P[p, d] - C[c, d])^2` | none | `points * clusters` not a multiple of 12 ("k-means distance pairs must divide evenly across 12 tasklets"); packet `8 * D` > 2048 B; values beyond `isqrt((2^31-1)/D)/2`, checked on the inputs at run time | no | **rejected** on every route: the template writes `D` |
| `feature_gradient` | linear `G[f] += ((S[s,f] * S[s,F]) * -(2^s)) >> beta`, logistic `G[f] += S[s,f] * (1 - 2 * S[s,F])` | none | other coefficients; label column not `F`; record > 2048 B; products >= 2^63 (run time) | no | **rejected** on every route |
| `kmeans_update` (host) | argmin over `D[p, c]`, `counts[idx] += 1`, `sums[idx, d] += P[p, d]`, then per-cluster division (section 4) | | any other op, operand relation or epilogue | | `counts` must be zeroed |

The minus spelling `a * x - b * y` and the constant term
`a * x + b * y + c` are the target ops `AXMBY` and `AXPBYC` in
`pim/targets.py`. Both lower to the int32 AXPBY template: the subtraction is
`a * x + (-b) * y` (the negated coefficient must itself be an int32, so
`- (-2147483648) * y` is rejected) and `c` is one more int32 add
(`UPMEMElementwisePlan.gamma`, compile-time only). A coefficient or constant
outside int32 raises "coefficient ... is not an int32 literal". uPIMulator
runs of `3*A - 7*B` and `2*A + 5*B + 11` on int32 inputs including the int32
extremes are exact (`tests/pim/test_upmem_generality.py::test_sim_pointwise_forms_off_grid`).
Forms with other association (`c + a * x + b * y`, `a * x - b * y + c`,
`a * x - y`) still match no target op.

All new pointwise expressions (`sub`, `mul`, `scale`, general `axpby`) are
emitted with explicit `uint32_t` wraparound, so int32 overflow is defined two's
complement, as in the source. The calibrated `add` and `(2, -1)` `axpby`
expressions are unchanged byte for byte. `UPMEMElementwisePlan.evaluate` is
the template's per-element result. Selection predicates are evaluated by
`SelectionPredicate.holds` (Python) and `c_condition` (C) with the same int32
and uint32 rules.

### 1.2 Shape-exact paths removed

* `tasklet_private_wram` MV was legal only at `(12, 16, 32, 12, 64, 16)`. It
  is now legal wherever its WRAM footprint and per-batch output DMA fit (table
  above). The calibrated one-batch-per-tasklet kernel is still emitted, byte
  for byte, when its structural condition holds: 12 batches, no column, row
  or output padding, unit scale, zero-initialized. Otherwise the strided-batch
  kernel `_strided_private_wram_device_source` is emitted. The old exact
  kernel also wrote `pow2(rows) * 4` bytes per batch, so it overran for a
  non-power-of-two batch row count. The structural condition excludes that.
* `UPMEMGEMMPlan.exact_row_fast_path` (rows == 12) is a structural template
  choice (one real row per tasklet), not a shape gate. Other row counts use
  the generic template.

## 2. Uneven partitions (U-04)

Work items may differ in one padding-neutral axis (`UpmemFamily.uneven`).
Classification merges the streams. The kernel geometry takes the largest
extent, and `UpmemKernel.streams` keeps each work item's own. One DPU binary
serves every work item (one `task.c` per segment). Smaller work items are
zero-padded to the largest (`_padded_slices`, flat padding at the end of the
partitioned outermost axis), and their padded outputs are dropped on gather.
The run record carries `geometry`, `work_item_geometry` and
`padding_fraction` = 1 - sum(logical extents) / (largest extent x DPUs).

Ranking: the F2 object a placement carries is now the extent-independent
tasklet-to-block carrier (section 6). The work items of an uneven group
therefore share decision keys and are ranked as one group. Each bucket is
priced at its own extent, while every DPU actually runs the padded program.
The device time is the slowest DPU's, which is the largest work item, priced
exactly. If the buckets nevertheless chose different decisions, the
executed decision is the largest work item's.

Source form (the paper partition, the last DPU takes the remainder):

```python
R, LAST = M // N, M - (N - 1) * (M // N)
@allo.work(mapping=[N], args=[A, x, y])
def mv(lA, lx, ly):
    (d,) = allo.get_wid()
    with allo.meta_if(d < N - 1):
        for i in range(R): ...          # rows d*R .. d*R+R-1
    with allo.meta_else():
        for i in range(LAST): ...       # rows M-LAST .. M-1
```

### 2.1 391x65 GEMV, 1-32 DPUs (executed)

Inputs: int8-range values (`rng(0)`, -8..7). The decision is fused MV
128 B/16 in every row. K = 65 pads to 80 columns. Every output is exact
(`y == A @ x`). Cycles are the slowest DPU's `logic_cycle`. All DPUs report
the same value because they run the same padded binary.

| DPUs | rows per DPU (last) | padding | slowest-DPU cycles, last-takes-remainder | balanced split (first `M mod N` DPUs get one more row) | virtual estimate (remainder split) | paper |
|---|---|---|---|---|---|---|
| 1 | 391 | 0 | 1,146,612 | (same) | 1,594,496 (1.39x) | 1,149,430 |
| 2 | 195 (196) | 0.26% | 576,115 | 575,423 | 797,579 (1.38x) | |
| 4 | 97 (100) | 2.25% | 296,106 | 292,741 (max 98) | 399,368 (1.35x) | |
| 8 | 48 (55) | 11.1% | 155,727 | 149,542 (max 49) | 200,756 (1.29x) | |
| 16 | 24 (31) | 21.2% | 84,665 | 77,998 (max 25) | 101,203 (1.20x) | |
| 17 | 23 (23), even | 0 | 61,287 | (same) | 54,257 (0.89x) | |
| 32 | 12 (19) | 35.7% | 49,421 | 40,058 (max 13) | 51,426 (1.04x) | 38,484 |

At 1 DPU the release route reproduces the paper's 1,149,430 within 0.25%.
Inputs differ, and `__mulsi3` time is data-dependent: it runs one step per
significant bit of the smaller operand. At 32 DPUs the paper's 38,484 matches
a balanced split (13 rows on the slowest DPU, 40,058 here, 4% off), not the
last-takes-remainder split (19 rows, 49,421). The remaining 4% is the input
data and the K padding: the paper's kernel tiles 65 columns differently from
the 16-element chunks here. The row block is
`max(2, pow2(ceil(rows/12)))` per tasklet, so 13..24 rows all use 2-row
blocks. 31 rows (16 DPUs, remainder split) use 4-row blocks with 8 busy
tasklets. That accounts for most of the gap between the two splits. The
paper's "fixed work per DPU stays at 73,803 cycles" (weak scaling) was not
re-run.

The table was produced while a concurrent C-04 change to
`spmw_codegen._check_work_grid` briefly rejected UPMEM `mapping=[N]` for
1 < N < 64; the probe bypassed that check. The check now admits an outer
subset of DPUs on UPMEM (`_OUTER_SUBSET_LAUNCH_BACKENDS`), and
`tests/pim/test_upmem_generality.py` compiles the 2/4/32-DPU partitions and
runs the 1-32 sweep (`paper_full`) and 64-DPU uneven GEMV/reduction through
the unmodified route.

## 3. Accumulator initialization (U-03)

`spmw_match_engine` sets `MatchedOp.extra["accumulator_init"]` on every match
with a loop-carried operand. The value is `"zero"` when a dominating zero
store covers the element or the whole buffer, or `"carried"` when there is no
initializer and the accumulator is a caller argument. Every other case raises
in the matcher. Every UPMEM recognizer with an accumulator reads the fact
(`_accumulator_init`):

* MV, GEMM and reduction now check initialization at all. Before, they had no
  check, and the existence-only `_check_zero_init` covered just four other
  families. That check is removed.
* `"carried"` is implemented by the template where that is a one-load
  extension. MV loads each tasklet's output group (fused) or each batch's
  outputs (private). GEMM loads each output tile before its reduction.
  Reduction loads the cell before the merge. Histogram loads the initial
  counts before the merge. The selection count is offset by the host
  compaction. The initial output is packed into the output region of the
  input MRAM image (`pack_inputs(..., initial_out=...)`). The cost features
  add the plan's own extra reads (`upmem_kernel_features`).
* `kmeans_distances` and `feature_gradient` reject `"carried"` in
  classification, so the rejection happens on every route, including
  `backend="virtual"`.
* A scaled MV (`y = alpha * acc`) uses a local accumulator that the matcher
  proves zeroed. A carried scaled form cannot reach a template: the plan
  rejects scale with a carried output.

Executed on uPIMulator with nonzero initial outputs, all exact
(`tests/pim/test_upmem_generality.py::test_sim_carried_*`): MV 36x40, GEMM
20x24x18, histogram 1000 inputs / 64 bins, reduction 1000, selection
`(x & 6) != 2` with an initial count of 5.

## 4. Host-placed work (U-10, U-11)

`kmeans_update` is placed on the host and runs its own lowered MLIR (the
portable C backend). The recognizer now admits only this matched structure:

* exactly one each of `MIN_SEL`, `ARGMIN_IDX`, `INC` and `SCATTER_ADD`;
* the argmin reads `D[p, c]` over the whole (points, clusters) nest;
* `counts[idx] += 1` and `sums[idx, d] += P[p, d]`, indexed by the argmin's
  result, with consistent cluster, point and dimension extents;
* `COMPACT` copies only of `D`, `sums` or `counts` elements;
* a zero-initialized `counts`.

Any other op rejects. At final compile, `_check_kmeans_update_epilogue`
proves on the IR that the only other caller-visible output is one
(clusters, dimension) array. Every value stored to it must be computed only
from `sums`/`counts` elements, constants, local scalars and arithmetic, and
must include a division. Tests cover a copy without division, an
index-dependent value and a points-sourced value.

The host group is priced: `costs/upmem.py::host_group_cycles`
(argmin, count and sum operations plus a 52-instruction signed divide per
centroid element) times `HOST_OP_CYCLES = 1`. The unit is cost-graph cycles,
with a coarse uncalibrated host model (`HOST_TIMING_SCOPE`). It occupies the
`host` unit. In a simulator run it is recorded as `modeled_host_cycles` and
kept out of the device-only `RunResult.cycles` (paper metric).

Host steps of DPU segments are part of the retained description (`_host_steps` in
every DPU run record):

* `gather_outputs`: the per-DPU output window gathered into the caller's
  arrays. Priced by explicit `host_moves` GATHER steps when the program
  declares them. Without host moves the virtual graph has no transfer step.
* `selection_compaction`: the stable compaction of the device flags,
  priced in the cost program (`selection_compaction_cycles`, a host step at
  the selection anchor). The step is described by the matched `COMPACT`/`INC`
  semantics: value-or-zero flags for the odd kernel, a 0/1 mask for general
  predicates, then append after the initial count.

The cross-DPU reduction "merge" is not compiler work. The reduction source
writes one partial per work item (`lP[d]`), and summing the partials is the
caller's code outside the program. The partials' transfer is the
`gather_outputs` step.

## 5. Cost validity and scope (U-09)

`upmem_physical_search.calibration_validity(problem, decision)` labels the
estimate `calibrated` only when an anchor of the decision's calibration
stratum has exactly this problem (shape and operation). Everything else is
`extrapolated`: a represented stratum at another shape scales by the stratum
mean, and an unrepresented stratum uses factor 1.0. The result carries the
nearest calibrated point (same kind, most shared stratum fields, then closest
logical work). Families priced by per-op rules (reduction, histogram,
k-means, gradients) and the host group are always `extrapolated`, with their
basis named. The label is in each placement's `extra["upmem_cost_validity"]`
and in `UpmemCtx.virtual_run_extra()`:

```
timing_scope = "analytical: device logic cycles incl. modeled host transfers"
cost_validity = "calibrated" | "extrapolated"   (all groups calibrated -> calibrated)
upmem_cost_validity = [per group: cost_validity, calibration_status, factor, basis, nearest_calibrated_point]
```

`spmw_codegen._run_virtual` (outside this repair's files) must merge
`layout_ctx.virtual_run_extra()` into `RunResult.extra`. Until it does,
`test_virtual_run_result_carries_scope_when_the_runner_forwards_it` skips. The
simulator's `RunResult.cycles` stays device-only (`"uPIMulator DPU
logic_cycle only"`). The virtual estimate includes explicit host moves and
modeled host steps, so the two are not comparable one-to-one.

Measured gaps (virtual / simulator; tolerances were not set in advance, so
these are observations, not a validation):

| Case | Decision | Validity | Virtual | uPIMulator | Ratio |
|---|---|---|---|---|---|
| VA 12288 | fused 256/32 | calibrated | 115,048 | 114,534 | 1.004 |
| VA 1000 | fused 256/32 | extrapolated | 10,983 | 20,374 | 0.54 |
| MUL 1000 | separate 64/16 | extrapolated | 80,714 | 57,246 | 1.41 |
| MV 100x100 | fused 128/16 | extrapolated | 558,884 | 452,464 | 1.24 |
| MV 48x64 | fused 128/16 | extrapolated | 90,255 | 99,499 | 0.91 |
| MMTV 6x8x32 | fused 128/16 | extrapolated | 55,157 | 61,259 | 0.90 |
| GEMM 20x32x32 | fused 1088 (Nc=Kc=16) | extrapolated | 717,319 | 813,235 | 0.88 |
| GEMV 391x65, 1..32 DPUs | fused 128/16 | extrapolated | section 2.1 | | 0.89-1.39 |

Outside the calibrated stratum the model is off by 0.54x to 1.41x in this
sample. The audit saw 1.39x to 1.53x. Small kernels, where fixed launch and
DMA latency dominate, are underestimated.

Features are computed for the padded program and add a busiest-tasklet issue
bound (`_issue_bound`: one tasklet issues at most once per 11-cycle revolver
period). The reference shapes and every pinned paper shape are balanced and
unpadded, so their features, the calibration factors and the 13 golden cycle
counts are unchanged.

## 6. Catalogs, knobs and the carried F2 layout (U-12, U-13)

* `UPMEM_DECISION_DOMAINS` are finite curated catalogs (the calibration
  campaign's lists). Selection among their realizable members is by cost,
  with no optimality claim beyond the catalog. Unrealizable members (MV and
  GEMM shared-WRAM, non-conditional-zero selection lowerings) are never
  enumerated or priced.
* `tasklets` only takes 12. That is a schedule policy, not a hardware limit
  (the target declares 24). Every template is a 12-tasklet plan. One-decision
  families (reduction, histogram, k-means, gradients) carry their single
  decision's fields, and their templates have one DMA policy each. The plan
  builders say so where they ignore the decision.
* The placement carries `UPMEM_TASKLET_CARRIER`, the masked 12-of-16 F2 map
  from tasklet coordinate to block index. The movement search may compose
  program dimensions onto it. Codegen proves the emitted plan's ownership
  `tid * BLOCK + local` factors through the carrier's `element` output with
  the plan's block (`_check_tasklet_ownership`). The block size is the plan's,
  derived from the extent, so the carrier stays extent-independent (section 2).
  Cost reads the tasklet count from the `tasklets` knob, which equals the
  carrier's active count by that check.

## 7. Diagnostics and tests (U-14, U-15)

* No-candidate kernels list every catalog decision's template reason.
* Selection predicates outside the domain state the domain.
* `kmeans_update` rejections state the expected structure.
* `tests/pim/test_upmem_matcher.py` no longer imports
  `scripts/prepare_upmem_tenon_campaign.py`. It checks the knob fan-out
  against the realizable catalog directly.

## 8. Owner-map row placement (movement ties, 2026-11-02)

Claim row 16 (movement ties survive to the cost ranking and execute) on
UPMEM. The movement stage (`allo/spmw_layout_problem.py`, `layout-search.md`
section 8) states a contraction's layout as a one-to-one GF(2) map from work
bits (owner level `dpu`, then local bits) to iteration bits (`work0`, the row
loop, the reduction loop). `tasklet` is not modelled for a one-dimensional
grid; inside a DPU the rows stay in ascending order, so the plan's
`tid * BLOCK + local` tasklet blocks and the carried F2 tasklet map are
unchanged (`_check_tasklet_ownership` still passes).

### 8.1 Realizable family

`row_map_unrealizable_reason` accepts a tie on UPMEM when:

* the kernel is one `MAC` with one accumulator of one output-row coordinate
  (GEMM's two-coordinate output and scaled GEMV's MAC + SCALE are outside);
* the owner bits come from the output-row bits only (no bit of `k`). A tie
  with an owner bit from `k` splits a row's reduction across DPUs. UPMEM
  accumulates in int32, so the reduction policy permits the regrouping, but
  the templates do not combine partial sums across DPUs (each DPU writes
  whole output rows); the rejection says so;
* the work grid is one-dimensional, partitions the output rows into
  consecutive per-work-item blocks in work-id order (a replicated grid is
  outside), and no input reads the row loops without the whole row index
  (MMTV batches are outside);
* every DPU gets the same number of rows, counted on the active rows (a
  masked row loop is accepted when it divides evenly).

The lowering (`spmw_upmem._owner_row_map`) also requires the single-batch
`matrix_vector` family on an even work grid and rejects other carried maps
with that reason (InfeasibleSchedule at materialization).

### 8.2 Realization

`_owner_row_map` evaluates `owner_rows` (the rows of each flat owner unit,
ascending) for the carried map and for the program map. DPU `d` (the `d`-th
work item in work-id order) holds the rows of unit `d`. When that equals the
program map the group carries no row map, so the identity reproduces today's
fixtures byte for byte. Otherwise `_GroupProgram.dpu_rows` holds the
assignment:

* pack: DPU `d`'s `matrix` and `out` (and `initial_out`) slices are
  `W[rows_d, cols]` and `y[rows_d]` (`_role_slice`); the vector is unchanged;
  `pack_inputs`/`pack_expected` and the plan are unchanged, so the task image
  (`task.c`) is identical and only the per-DPU MRAM images differ;
* gather: `_run_upmem` writes DPU `d`'s output back to `y[rows_d]`;
* `_bind_module` checks that each work item's IR windows are exactly the
  rows the program map gives it (`_check_row_windows`);
* the segment command gains ` row_map=<FNV-1a 64 digest>` and the run record
  a `row_map` entry (`rows_moved`, `rows_per_dpu`, digest, sample row ->
  `(dpu, local_row)`).

### 8.3 Ties and simulator runs

GEMV 1024 x 128 int32 over 8 DPUs (128 rows per DPU, unmasked; `M >
DPUS * K / 2`, so splitting `k` never pays): minimum 3,584 B = program,
proven; ties: the program (block) map and `rows cyclic over dpu` (DPU `d`
holds rows `d, d + 8, ...`); `tie_count == realizable_tie_count == 2`,
`chosen_label == "minimal"`, 4 scored candidates (2 ties x 2 keys), virtual
estimate 646,015 for both. The structural tie-break selects the cyclic map
in the default compile.

Expected before running: the per-DPU work is identical (same task image, 128
rows x 128 columns packed the same way), so cycles are equal when every DPU
sees the same values; with random values they may differ because the DPU
multiply's cost depends on the operand values and the DPUs hold other rows.

| Tie | Input | Outputs | Cycles (max over DPUs) |
|---|---|---|---|
| program `dpu<-work0[0,1,2]` | random int32 in [-8, 8) | `W @ x` exact | 595,371 |
| cyclic `dpu<-loop0[0,1,2]` | random | exact | 593,932 |
| program | every row equal | exact | 604,829 |
| cyclic | every row equal | exact | 604,829 |

uPIMulator `logic_cycle`, 12 tasklets, `fused_replicated` decision; each run
reported the same count on all 8 DPUs. Test:
`tests/pim/test_upmem_owner_maps.py`.

The pinned all-targets GEMV (768 x 128, 8 DPUs, 96 masked rows per DPU) is
unchanged: its minimum (2,560 B against the program's 3,584) is `dpu <-
loop0[4,5,6]`, which gives 6 DPUs 128 rows and 2 none; it is rejected for
unequal row counts, the cyclic distribution has the program's movement and
is not a tie. The 64-DPU x 12-row GEMV's minimum ties take owner bits from
`k` and are rejected as reduction splits.

## 9. Open

* `_run_virtual` forwarding of `virtual_run_extra()` (spmw_codegen.py).
* `a * x - b * y` and `+ c` need target-description patterns.
* The 12-tasklet policy and power-of-two row blocks cost up to about 2x at
  awkward row counts (31 rows -> 8 busy tasklets). A non-power-of-two block
  would change the pinned paper programs, so it was not done here.
* Weak-scaling "73,803 cycles per DPU" not re-run.
