<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Tenon backend capability matrix

Scope: `experiments/allo`, branch `tenon`, working tree on top of `6efaca5`
(round 3, 2026-11-02). Sources: `docs/release/audit/*.md`,
`docs/release/ISSUES.md`, `docs/release/VALIDATION.md`,
`docs/release/design/*.md`, and the runs noted in the "Evidence" rows.
Finding IDs (S-xx, M-xx, U-xx, V1-xx, G2-xx, C-xx, P-xx, F-xx) refer to
`ISSUES.md`; check it for the current status of each. The implementation map
(modules, stages, inspection fields) is `SPMW_ARCHITECTURE.md`.

Terms used below:

- **Input kind.** *Source*: an ordinary `@allo` workload (usually an
  `allo.dataflow.region` with `@allo.work(mapping=...)`; a plain function
  with a matched contraction on APU v1 and Gemini-II, whose vector plans and
  launch plans are candidates of the same shared pipeline) passed to
  `allo.compile(workload, target, cost)`. *Typed*: a typed program object
  (`APUv1Program`, `APUG2TypedProgram`, `APUG2ComposedContractionProgram`)
  passed to `allo.compile`; they have no source for the matcher and enter
  the shared pipeline as frontends (their semantics stated in the shared
  program contract, their realization a candidate of the common selection).
- **Modes**, matching `RunResult.kind`: *compile-only* (emit and inspect, no
  run), *analytical* (`backend="virtual"`, `kind="virtual"`; cost-program
  evaluation of the retained execution graph; nothing executes and caller
  arrays are not written), *simulator functional* (`kind="simulator"`;
  outputs read back from the simulator), *simulator timing-only*
  (`kind="simulator-timing-only"`; no outputs exist), *hardware*
  (`kind="device"`), *host functional* (`kind="functional"`; host execution
  of the lowered program, cycles are not a measurement).
- **Reduction order.** `allo.compile(..., reduction_order=)` is `"strict"`
  by default (source accumulation order and rounding for floating-point
  accumulators; integer accumulators may always be split) or
  `"reassociate"`. Rows below say which programs each policy admits.
- **Resource budget.** `allo.compile(..., resources={"channel": 8})` caps
  the structural units one program may occupy (`allo.spmw_resources`): the
  enumerators bound their partition domains by it (AiM replica and token
  partition counts), the shared ranking selects among the candidates that
  fit, a supplied placement outside the budgeted enumeration is refused, the
  realization check compares the lowered program's unit use with the cap
  (a feasibility rejection the ranking records), and the budget is on every
  result (`extra["resource_budget"]`). A cap above the device's count or an
  undeclared unit is refused. It is a search policy, not a device
  description: target and simulator geometry are unchanged.
- **Evidence status.** *Verified* means run at the round 3 tree on the
  authors' hosts through the public API and recorded in `VALIDATION.md`;
  **UNVERIFIED** means no run at the candidate revision, whatever older
  artifacts say. No simulator result is a hardware measurement.

## Summary

| Backend | Target builder / cost | Input | Compile-only | Analytical | Simulator functional | Simulator timing-only | Hardware |
|---|---|---|---|---|---|---|---|
| Samsung HBM-PIM | `build_samsung_target()` / `samsung_cost` | source | yes | yes | yes (PIMSimulator `pim_driver`, Tenon conductor) | no | no |
| SK hynix AiM | `build_aim_target()` / `aim_cost` | source | yes | yes | no | yes (Ramulator2 `aim_simulator`) | no |
| UPMEM | `build_upmem_target()` / `upmem_cost` | source | yes | yes (validity labelled) | yes (uPIMulator + DPU SDK) | no | no |
| AttAcc (HBM3-PIM) | `build_attacc_target()` / `attacc_cost` | source (FP16 GEMV, attention scoring) | yes | yes (closed-form ranking model, validity labelled) | no | external evaluation script only (Ramulator2-AttAcc, `benchmarks/attacc/scripts/run_ramulator.py`; not a runner) | no (academic architecture) |
| GSI APU v1, source contraction route | `build_apu_v1_target()` / `apu_v1_cost` | source (rank-2 contractions, 1-D GEMV) | yes | yes (see note) | no simulator exists | no | yes (Gemini-I board) |
| GSI APU v1, `APUv1Program` scalar route | same | typed (`APUv1Phase(vectorize=False)`) | yes | yes | no | no | yes; plus host functional |
| GSI APU v1, native recipes | same | typed (`APUv1Phase(vectorize="required")`) | yes | yes | no | no | yes |
| GSI APU v2 (Gemini-II), `APUG2TypedProgram` | `build_apu_g2_target()` (= `build_apu_v2_target()`) / `apu_g2_cost` | typed | yes | yes (card-fitted tick model) | no | no | yes (Gemini-II card) |
| GSI APU v2, `APUG2ComposedContractionProgram` | same | typed | yes | yes | no | no | yes |
| GSI APU v2, source | same | source (integer contractions) | yes | yes | no | no | yes |

Every PIM target compiles and runs analytically with only Allo installed.
Everything else needs the external pieces in `simulators.md` and the
discovery variables in `INSTALL.md` section 3.

## Samsung HBM-PIM

| Aspect | Status |
|---|---|
| Input | Source FP16 contractions classified structurally (`spmw_samsung.samsung_contraction`): nests `[k]`, `[row, k]` (GEMV), `[row, column, k]` (GEMM), either operand order (`W[i,k]*x[k]` or `x[k]*W[i,k]`, S-05 fixed), literal-zero accumulator; work items must partition the output rows exactly (full `[16, 8]` grid or `mapping=[1]`, S-10). Elementwise `ADD`/`MUL`/`MAD`/`RELU` for one work item (`mapping=[1]`) walking a 1-D FP16 extent (S-07 fixed); partitioned elementwise grids are rejected with the reason. A carried accumulator (`y[i] += ...` with no initializer) is rejected before scheduling. |
| Compile-only | `compiled.compiled.cmds` is the emitted per-pass CRF (`PIMCmd` list); `compiled.compiled.layout_ctx.programs` the `SamsungProgram` records; `compiled.placement_ranking` the ranked placements, pruned modes (`policy_pruned`) and movement record. |
| Analytical | `samsung_cost` over the retained graph. Unit: memory-clock cycles of the modeled activities (GRF fills, MAC/elementwise, drains, explicit host moves) of one work item on the declared 16-channel device, unpadded shape; preload, CRF upload and readback are outside the graph. Not comparable in absolute value with simulated cycles; compared by candidate ranking (4 of 6 placement pairs agree per GEMV shape, `design/samsung-execution.md` section 7). |
| Simulator | PIMSimulator `pim_driver --op GENERIC --op-kind REDUCE | TEMPORAL_REDUCE | ELTWISE`: the conductor uploads the compiled CRF verbatim and never calls the vendor command generator (S-01 fixed); `kind="simulator"`, outputs read back (`out.bin` absent or short raises; all-zero is a value, S-02 fixed). |
| Numerical scope | FP16 inputs. `reduction_order="strict"` (default) selects the ordered TEMPORAL program: one GRF_B lane per output row, K issued in source order, bit-equal to source-order FP16 (4096/4096 rows at 4096x256, including a 0.5 initial value through the runtime). `"reassociate"` admits the REDUCE programs (`bank_row`, `dual_fiber`): lane `k mod 16` partials summed on the host in FP32, within 1e-2 of FP32 (343/4096 rows bit-equal to source order). Elementwise results are bit-exact against NumPy FP16. |
| What cycles mean | PIMSimulator memory-clock cycles of the **padded** program (preload + execution + readback on the simulated 64-channel system). REDUCE pads K to 128-element tiles and rows to 4,096-slot passes; TEMPORAL pads rows to 8,192-row tiles; elementwise pads to 131,072-element tiles. `RunResult.extra` reports `logical_shape`, `padded_shape`, `padding_fraction`, `timing_scope`, `cycle_phases`, `plan`, `crf`, `conductor`, `levers` (S-06 fixed). |
| Selection | Default under strict: TEMPORAL (26,807 cycles at 4096x256). Under `"reassociate"`: bank_row + host GRF_A fill, 4,389 / 7,947 / 8,153 cycles at 4096x256 / 4096x512 / 8192x256, 1.0-1.2% fewer than the vendor program; the paper's vendor-equivalent placement (dual_fiber, host fill, shared CRF) is selected explicitly with `allo.spmw_samsung.samsung_vendor_gemv_layout(target, trace)` and reproduces 4,435 / 8,041 / 8,247. Unrealizable levers (`grf_staged`, `crf_issue=per_workid` with more than one item, more than 32 CRF words, non-identity `tile`, `double_buffer`) are rejected at materialization and recorded in `KernelRanking.rejections`. |
| Hardware | none (no HBM-PIM device). |
| Evidence | Verified at round 3: `tests/pim/test_paper_golden.py` Samsung rows (vendor layout 4,435 / 8,041 / 8,247), `tests/pim/samsung_hbm_pim/test_samsung_lowering.py`, `test_samsung_conductor_runs.py`, `test_samsung_reduction_order.py`, `gemv/` tests; executed table in `design/samsung-execution.md` sections 6-7. The running `pim_driver` is built from the overlay in the containing repository's `experiments/simulators/setup/PIMSimulator/`, which is not yet committed there (S-04, P-04, P-05). Open: S-11 (docs page), S-12 (CRF-fill preload outside the model), S-13 (XOR swizzle 148 -> 116 only as a harness, `tests/pim/samsung_hbm_pim/layout_swizzle/`). Multi-kernel programs run their `@allo.host_program` launch schedule, one conductor invocation per launch, gathered results re-staged as declared; device residency across launches is rejected (`design/samsung-execution.md` section 9, `tests/pim/test_host_program_analysis.py`). |

## SK hynix AiM

| Aspect | Status |
|---|---|
| Input | Source GEMV and attention-style batched MAC (`acc += W*x`), elementwise `MUL`/`ADD` and `AF` through target ops; mapping a prefix of `[32, 4, 4]` or an outer subset. Output, reduction, batch and work axes are derived structurally (`axis_roles`, M-01 fixed); undecidable forms (split-K grids, uneven per-item extents, `y[row0+i] += ...`) raise; issued MACs are checked against the program (`AimCoverageError`). Carried accumulators are rejected (M-09/C-14). |
| Compile-only | `compiled.compiled.cmds` is the exact Ramulator2 trace text (`AiM WR_GB ...`, `MAC_ABK ...`, `EOC`); `compiled.compiled.layout_ctx.host_steps` lists host reductions. |
| Analytical | `aim_cost`: one route prices the exact lowered record (M-06 fixed; GEMV 256x256 ABK 223 vs 222 simulated, SBK 1,528 vs 1,527); host transfers priced by the moved slice and residency (M-05 fixed; `attention_l128` 10,108 estimated vs 11,010 simulated). Long streams remain 1.25x to 1.59x high (token-partition study). |
| Simulator | Ramulator2 timing simulator in Docker (`TENON_AIM_SIM_IMAGE`, default `aim-simulator-build`) or a natively built binary (`TENON_AIM_RAMULATOR2`); every run records the executor (image id, or the binary's SHA-256 with the shared library it resolves and its SHA-256), the configuration file's SHA-256, each segment's exact trace lines, digest, repeat count, cycles and simulator output, and the invocations (`extra["simulator"]`, `extra["segment_records"]`, `extra["invocations"]`); `kind="simulator-timing-only"`, no `extra["outputs"]`, caller arrays untouched; a nonzero simulator exit raises (M-08 fixed). Trace files go to a private temp directory (P-21 fixed). |
| Numerical scope | **None in the simulator** (no data path). Under `"strict"` every BF16 reduction is refused before scheduling with `ReductionOrderError` (each MAC regroups 16 products through the bank adder tree); use `reduction_order="reassociate"` or integer accumulators. The token-partition merge (`reduction_partitions` knob, `AimHostReduction`) is checked numerically by `aim_lowering.reference_partial_sums` (exact on integers, within 3.7e-3 on BF16). |
| What cycles mean | Max over channels of Ramulator2 `memory_system_cycles` per distinct trace segment, times the compile-time repeat count, summed. Host steps are listed in `extra["host_steps"]` and never added. Simulator geometry is fixed at the 32-channel preset; other targets are rejected (M-10). |
| Hardware | none. |
| Evidence | Verified at round 3: 14 CENT traces byte-identical to pins with golden cycles (`test_paper_golden.py`), `tests/pim/test_aim_axis_roles.py`, `test_aim_cost_routes.py`, `test_aim_token_partition.py` (6 of 9 simulated minima reached, max gap 1.34%; paper 7 of 9), `test_aim_virtual_study.py`. The AttAcc compiler path is the `attacc` backend below (M-04). |

## AttAcc (HBM3-PIM, analytical backend)

| Aspect | Status |
|---|---|
| Input | Source FP16 kernels on one work item (`mapping=[1]`): GEMV `y[h, n] = sum_k W[h, n, k] * x[h, k]` (one `MAC`) and attention scoring `QK^T -> exp/sum/divide -> PV` (`MAC`, `SFM_EXP`, `SFM_SUM`, `SFM_DIV`, `MAC`). Axes come from `axis_roles`; other dtypes, carried accumulators, multi-item grids and other op mixes raise `AttAccDomainError`. The max-stabilized softmax is **not** accepted (needs a `-inf` accumulator initializer the matcher does not support); the un-stabilized form is. Requires `reduction_order="reassociate"`. Sources: `benchmarks/attacc/workloads.py`. |
| Compile-only | `compiled.compiled.cmds` lists the command inventory (`"PIM_MAC_AB count=15360"`); `compiled.compiled.layout_ctx.programs` the `AttAccProgram` records (kernel, tiling, counts, `dcc_tiling()`); `compiled.placement_ranking` every legal tiling ranked (3,043 for qkv batch 1). |
| Analytical | `attacc_cost`: the paper's closed forms (`m8` GEMV, `serial_q_cf` attention) over the command counts, one device, priced once per kernel record. Ranking model: paper Spearman 0.94 / 0.98, MAPE 30.78% / 14.19% vs Ramulator2-AttAcc; row-buffer state and issue order omitted; `extra["cost_validity"]`, `extra["timing_scope"]`. Selected tiling in `extra["attacc_tiling"]`. Details: `design/attacc-backend.md`. |
| Simulator | none in the package; `backend=None` raises `NotImplementedError` (analytical-only). `benchmarks/attacc/scripts/run_ramulator.py` times the selected program in Ramulator2-AttAcc as an external evaluation step (`kind: "simulator-timing-only"`), using DCC's unlicensed trace generator from a user checkout (`simulators.md`). |
| Numerical scope | none (no data path anywhere; caller arrays untouched). |
| What cycles mean | Closed-form per-channel makespan of one device's command inventory, in Ramulator2 memory cycles (integer, truncated). Not a simulator measurement. |
| Hardware | none (academic architecture). |
| Evidence | 2026-11-02: `tests/pim/test_attacc_virtual.py` (target legality, space sizes 1,391 / 2,273 / 3,043 / 4,478, 24 DCC count fixtures, the four GEMV instances 7,040 / 25,600 / 2,432 / 8,192 with the paper tilings, attention worst-loss and tie shapes, cost perturbation, domain errors, renamed source, `layout=` legality); all 500 paper instances through `allo.compile` select the paper tiling (`design/attacc-backend.md` section 5); external Ramulator2-AttAcc smoke run 11,454 cycles (qkv batch 1) and 953 (worst-loss attention), equal to the archived values. Decode composition (`compose_e2e.py`) is external and not validated at batch 4. |

## UPMEM

| Aspect | Status |
|---|---|
| Input | Source kernels proven as one of nine families (`spmw_upmem.classify_upmem_kernel`): `pointwise` (ADD, SUB, MUL, SCALE, `a*x + b*y`, `a*x - b*y`, `a*x + b*y + c` with int32 literals, all on the int32 AXPBY template; U-06 fixed), `reduction`, `matrix_vector` (incl. MMTV and scaled), `gemm`, `histogram`, `selection` (`(x & mask) cmp k`, `x cmp k`, odd flags), `kmeans_distances`, `kmeans_update` (host-placed), `feature_gradient`; `mapping=[N]`, `N <= 64`, uneven partitions through `meta_if`/`meta_else` (last DPU takes the remainder; U-04 fixed). Legality is the template's (DMA size and alignment, WRAM footprint); non-dividing extents are zero-padded and masked (U-05, U-06 fixed). Carried accumulators are loaded by the MV, GEMM, reduction, histogram and selection templates and rejected for k-means distances and gradients (U-03 fixed). |
| Compile-only | `compiled.compiled.cmds` names the segments; `compiled.compiled.layout_ctx.segments[i].device_source` is the DPU `task.c`. |
| Analytical | `upmem_cost` over the retained graph; every virtual result carries `cost_validity` (`calibrated` only at a calibration anchor, else `extrapolated` with the nearest point) and `upmem_cost_validity` per group (U-09 fixed). Outside the calibrated stratum the model was 0.54x to 1.41x off in the recorded sample (`design/upmem-domain.md` section 5). The host-placed group and selection compaction are priced with a coarse host unit (`HOST_TIMING_SCOPE`). |
| Simulator | DPU program built with the DPU SDK in Docker, run on uPIMulator with the `GENERIC` fixture; uPIMulator byte-compares the observed MRAM window with the portable-C host oracle; decoded outputs overwrite the caller's arrays (`kind="simulator"`). Two routes: pinned-campaign (archived binary, snapshot, pinned SDK manifest) and user-build (`UPIMULATOR_ROOT` + `UPIMULATOR_BIN`); each record carries `sdk_build` (`pinned-campaign` or `user-build:<manifest12>`). |
| Numerical scope | Element types enforced per role (`UPMEM_ROLE_ELEMENT_TYPES`: int32 operands, int64 accumulators for reduction, k-means distances and gradients; U-01 fixed); wide-integer host code is rejected instead of truncated (U-02 fixed). Results exact against the oracle on every executed case, including the 391x65 GEMV sweep over 1-32 DPUs and carried-accumulator runs. |
| What cycles mean | Sum over DPU segments of the max per-DPU uPIMulator `logic_cycle` (350 MHz logic clock); host packing, host transfers and host-placed groups are excluded (`extra["timing_scope"]`); modeled host work appears as `modeled_host_cycles` in the records. |
| Hardware | none (no UPMEM DIMMs). |
| Evidence | Verified at round 3: 13 `paper_full` golden rows with pinned cycles (`test_paper_golden.py`), `tests/pim/test_upmem_generality.py` (391x65: 1 DPU 1,146,612 cycles vs paper 1,149,430; 32 DPUs 49,421 remainder split / 40,058 balanced vs paper 38,484), `test_upmem_dtypes.py`, `test_upmem_matcher.py`, user-build route (`user-build:d45e285c34b5`). A build from public sources is **not yet possible**: the Tenon uPIMulator patch is unpublished (P-06) and the SDK image reuses an upstream name (P-17). |

## GSI APU v1 (Gemini-I)

Four routes share `build_apu_v1_target()`. Every board launch takes the
board lock (`apu_v1_program.apu_v1_board_lock`, `TENON_APU_V1_LOCK`,
default `/tmp/tenon-apu-v1-<pci node>.lock`) and proves profile freshness
by diffing the device log before and after the launch (V1-07 fixed).

| Aspect | Source contraction route | `APUv1Program` scalar route | Native recipes |
|---|---|---|---|
| Input | Plain `@allo` function (not a dataflow region) whose matched effect is one rank-2 contraction `C[i,j] += A[i,k]*B[k,j]` (FP16/uint16 multiply-add; the accumulator a function argument, carried or zero-filled, or since round 12 a local scalar copied once to the output `acc = 0; ...; C[i,j] = acc`); 1-D GEMV lifted with a unit axis. The shared front stages run first (matcher coverage, alias, cast, copy and precision contracts, `ProgramFacts`); bitwise and XNOR-popcount products are not matchable on the target yet (ISSUES V1-18) | `APUv1Program([APUv1Phase(fn, vectorize=False)])`: the whole function lowered to C11 with `emit_c_from_mlir`, runs on APUC 0; one phase only | `APUv1Program([APUv1Phase(fn, vectorize="required")])` proven by exactly one of four recipes (dense 256-bin histogram, squared-L2 argmin, bivariate moments, record frequency) with a strict MLIR parser and symbolic executor; mismatches raise `APUv1NativeRecipeMismatch` (V1-01 fixed) |
| Plan selection | Finite recipe catalog (`temporal_svp`, `temporal_dma_coalescing*`, `..._acc{2,4,8}`, `spatial_gemv_group_reduction_k<tile>`, ...); regrouping plans pruned under `"strict"` (`reduction_policy_pruned`); every surviving plan realized to GVML, infeasible ones dropped, ranked by `apu_v1_cost` on the materialized instructions (`schedule_search.grid_search`); `layout=<plan>` forces a plan; `cost=None` without `layout=` is an error (V1-10) | none | the recognizer's single template |
| Analytical | yes (`kind="virtual"`: the selected plan's retained cost graph only, no outputs, caller arrays untouched, `cost_validity` label; `tests/perf/test_run_result_contract.py`) | yes (`kind="virtual"`, V1-04 fixed) | yes |
| Host functional | yes (`kind="functional"`: a host-side NumPy model of the selected plan's arithmetic order, not compiled device code; `cycles` is the analytical estimate) | yes (retained portable C on the host) | rejected at compile time |
| Hardware | yes: GVML project built with the ARC toolchain, launched on the board, outputs read back; temporal plans upload the packed initial output and accumulate onto it, spatial group-reduction plans reject a nonzero initial output (V1-02 fixed) | yes | yes |
| Numerical scope | FP16, uint16, bitwise. Under `reduction_order="reassociate"` the default FP16 GEMV plan is `spatial_gemv_group_reduction_k<tile>`, a tree reduction (`gvml_add_subgrps`) whose rounding differs from source order: 425 of 4096 outputs of the 4096x256 GEMV are bit-equal to the source-order FP16 sum in the plan-order model (a halving FP16 tree per group, then the groups in ascending order; `tests/perf/test_apu_v1_reduction_policy.py`). Under `"strict"` (the default) the tree plans are pruned and `temporal_dma_coalescing` is selected, bit-equal to source order (4096/4096) in the same model. Neither is board-checked bit for bit: the halving order is a model of `gvml_add_subgrps` never compared with the board, and the board GEMV test (`test_gemv_apu_v1`, strict) checks a 5e-2 tolerance only (the paper's source-order board match came from an external script, `CLAIMS.md` E11) | C11 semantics of the C backend; `sqrt` lowers to a double-precision helper within 1 ulp of the C library (V1-12 fixed) | fixed shapes and dtypes per recipe (histogram 32,768 values x 256 bins, argmin P=32,768, D<=48) |
| What cycles mean | ARC core cycles (`crun`) of the profiled `total` section printed by the device build (`PROF_PRINT`), exactly one fresh record per launch | same | same |
| Evidence | Verified at round 3 on this server: `test_gemv_apu_v1` and two `tests/pim/apu_v1/vector_gemm` plans passed on the board (4 device tests, 197 s). Paper numbers (E9-E12) came from an external locked runner (`CLAIMS.md`); open: V1-05 (fitted count rule absent, native estimates 19-56% off without a validity flag), V1-06 (fixed tile heuristic; 12.71x row plans reachable by `layout=`, golden drift 2.9%); V1-08 fixed (only the eight GSI SDK files the build reads are copied from `TENON_APU_V1_TEMPLATE_DIR`, origins in the project's `THIRD_PARTY.txt`; an ARC build of a generated project succeeded on this server 2026-11-02, not launched) | virtual/functional covered by `tests/perf/test_run_result_contract.py`; hardware **UNVERIFIED** at round 3 | histogram golden median row passed on the board at round 3; other recipes compile-only at round 3 (`test_paper_golden.py`) |

A legacy matcher route (`spmw_apu_v1.APUv1Ctx`, grouped FP16 MAC over four
APUCs with `gvml_add_subgrps`) is reachable for dataflow regions only; it
refuses float accumulators under `"strict"`, reads output shapes from the
compiled ABI and raises on missing output files (V1-09 fixed).

## GSI APU v2 (Gemini-II)

Three routes share `build_apu_g2_target()` (`target.name == "apu_v2"`).
Every runtime takes an in-process `flock` (`TENON_APU_G2_LOCK`, default
`/tmp/tenon-apu-g2-<device>.lock`) around status check, build and launch;
`extra["device_lock"]` records the path (G2-06, partially fixed: only
`apu-00` and SDK root `/opt/gsi`-compatible installs are accepted).

| Aspect | `APUG2TypedProgram` | `APUG2ComposedContractionProgram` | Source route |
|---|---|---|---|
| Input | Typed op record: `add`, `mul`, `div`, `block_sum`, `dot` over `input_types`/`output_type`, `shape`, `reduction_extent`, `repetitions` | Typed record of independent dot products `(dot_count, K)` with an epilogue (identity, two-coefficient pair-affine, accumulate); GEMM/GEMV/batched forms pre-packed by the caller with the host helpers | Plain `@allo` function with one or two integer dot-product reductions (`acc += x*y`) and optionally an `alpha*t + beta*u` epilogue (`allo/pim/apu_g2_source_route.py`, G2-02 fixed); `benchmarks/apu_g2/workloads.py` holds the five paper kernels |
| Legality | exact output width (`out_bits == lhs + rhs (+ log2 K)`), shared signedness, unsigned-only div/block_sum, <= 24-bit segments, K power of two in [2, 4096], up to 262,144 elements | one `(4, 65536)` carrier: `padded(dot_count) <= 4*65536/K_pad`, `K_pad <= 256`; larger problems are rejected naming the capacity | exact widths derived from the source (`D = a + b + log2 K <= 24`, carried output `>= D + 1`), one signedness, K power of two `<= 512`; dots beyond one carrier are tiled, K up to 512 split into two launches (G2-05 fixed); anything else raises `APUG2SourceRouteError` |
| Plan selection | none: fixed layout; `layout=`/`host_moves=` rejected | none: one deterministic `APUG2DotTilePlan` (`LinearLayout` over F2) | `enumerate_apu_g2_launch_plans` (`single`/`fused`/`separate` x `k1`/`k2` x output tiles) ranked by `rank_apu_g2_launch_plans` (estimate, launches, catalog order); `layout="<plan>"` forces one; `compiled.compiled.placement_ranking`, `.rejected`, `.manifest()` |
| Analytical | `kind="virtual"`; `apu_g2_cost` prices every issued recipe call from seven structural features fitted on ten card programs (in-sample within 1.9%, leave-one-out within 2.6%; G2-03 fixed); `estimate_validity="extrapolated"` outside reduction depths 7-8 | same | same, summed over launches |
| Hardware | project template (`allo/pim/templates/apu_g2`) built per call with CMake, run on the card after `gsi_tool info` reports it Available and core 0 idle; outputs compared with an int64 NumPy oracle inside the runtime, mismatch raises; device values returned | same, plus the layout fingerprint bound to the packed carriers before launch (G2-08 fixed) | same, each launch through `run_apu_g2_composed`, results checked against an `einsum` oracle of the matched contractions |
| Numerical scope | exact integer arithmetic modulo the declared widths (1-24 bit) | same | same |
| What cycles mean | `round(device pipeline ticks / repetitions)`: ARC running-state ticks over `repetitions` bodies plus one barrier, host transfers excluded (`extra` keeps host microseconds separately); default `repetitions=256` | same | sum of launch cycles; host readback between launches outside the unit |
| Evidence | Verified at round 3 on zhang-x4 `apu-00` (FW V1.24.1): `tests/pim/test_apu_g2_hardware.py` 10 passed (G2-04); typed add/mul/div/block_sum/dot exact, cycles 315 / 311 / 1,014 / 2,585 / 1,725 vs estimates 340 / 307 / 1,020 / 2,588 / 1,714 | five paper programs exact in round 2 (1,711 / 1,720 / 1,876 / 2,569 / 2,754 vs paper medians within 1.3%) | round 3: GEMM, batched GEMM, GESUMMV, BiCG, MVT from source exact with cycles 1,736 / 1,719 / 1,869 / 2,554 / 2,766 (estimates within 8%), GEMM 64x128x33 two-launch `single/k2` 2,685 (extrapolated estimate 2,550); layout fingerprints equal the artifact's |

G2-09 to G2-11 fixed: device readback and fixture decode raise on values
outside the declared output type instead of wrapping (typed ADD keeps its
declared modulo-2^bits semantics); padded divisor lanes of a partial-shape
typed `div` carry 1 (packing tested; not yet run on the card); the template
ships only the uint16 add, typed and composed programs. Open: G2-12 (docs),
`DeviceConfig` semantics for cards other than index 0, `t + u` epilogue
without coefficients.

## Cross-cutting status

- Source semantics the matcher enforces before any backend is consulted
  (C-02, C-14, C-17, C-19, C-20, C-21): every store of every work-item
  function is classified and an unmatched effect raises
  `UnsupportedWorkloadError`; accumulators are `zero` or `carried`; an
  operand may not alias its own result at another element; a conversion is
  looked through only when it is exact (`cast_is_exact`), a constant only
  when proven representable, a store-side narrowing only when proven exact
  (integer: the leaves fit the destination width or the value provably
  fits; float: a single operation on destination-type leaves, where
  rounding once in f32 and then to f16/bf16 equals the device's correctly
  rounded operation; a compound expression such as
  `float(a)*float(b)+float(c)` keeps f32 intermediates and is rejected
  unless the target op declares `intermediate="f32"`, as the AiM `AF`
  lookup table does), and any other conversion (`int(acc)` on an FP16
  output, `fptosi`, an unproven `trunci`) matches no target op and is
  rejected naming the conversion; an operation the source evaluates in a
  wider type than its operands and stores wide (`y_f32[i] = float(a[i]) +
  float(b[i])` on FP16 arrays, `i32 + i32` stored to int64) binds only to
  an op declaring that `intermediate` precision, otherwise it is rejected
  naming the precision (`arithmetic_precision_problem`), while an FP16
  operation whose result is widened at the store is the FP16 operation; an output copy must run under exactly its
  producer's guards, exactly once per produced element (never inside the
  reduction loop), read the element its producer stored, and write a
  buffer of the producer's type, otherwise it is rejected
  (`if cond: out[i] = acc` is unsupported, not silently unconditional).
  Output access maps are the identity: every store and operand coordinate
  is resolved to an affine form and a contraction's result must be stored
  at the coordinate its matrix row was read from (`out[M-1-row]`,
  `out[0]` for every row, `out[row+1]` are rejected on Samsung, AiM and
  AttAcc and by the layout derivation; no backend realizes a permuted
  output); an elementwise result is written at the element it was read
  from (Samsung and AiM reject reversed, shifted or broadcast destinations
  and shifted inputs). Copies of a result into another buffer are accepted
  only element for element and count as effects: every externally visible
  write of the source (matched stores and accepted copies into function
  arguments, `external_effects`) must be declared realized by the backend
  (`require_effects_realized`) or compilation stops; Samsung's elementwise
  program writes the result into every such sink, the other backends refuse
  copies of external array results (C-25). Signless loads are read as signed or unsigned according to the
  frontend's extension; unknown signedness uses the union range.
- Placement legality (`allo/spmw_legality.py`) filters enumerated
  candidates and rejects illegal `layout=` (C-01 fixed); the work grid must
  map onto the unit tree in a documented form (C-04 fixed).
- Reduction order is explicit (`reduction_order`, default `"strict"`,
  C-09 fixed); every reassociating paper row passes `"reassociate"`.
- Ties are broken by a documented structural key and exposed as
  `KernelRanking.cost_ties` (C-06 fixed).
- The minimum-movement layout stage runs per kernel group and records
  ties, termination and optimality (`KernelRanking.movement`, C-11 fixed);
  owner maps other than the work-grid map are realized on Samsung, UPMEM
  and AiM; APU v1 realizes only the work-grid map and rejects other ties
  with a stated reason (CLAIMS row 16 partial on APU v1).
- Candidates are materialized in cost order and backend rejections are
  recorded (C-03 fixed); the reconciled whole program is re-priced and,
  within `SelectionPolicy.joint_budget`, every complete assignment is
  evaluated as one program (C-05 fixed, bounded and labelled).
- Shared frontend: annotation names resolve from the defining scope only
  (F-01 fixed); integer literals compile to their exact value or raise
  (F-02 fixed).
- `RunResult.kind` is set on every route and validated; non-executed kinds
  never write caller arrays (G2-07, S-02, V1-09, M-08 fixed).
- One shared pipeline for ordinary source since rounds 12-13 (gate G9,
  ISSUES X-01, in progress): every target passes
  `allo/spmw_pipeline.py::normalize_source` (frontend, function-boundary
  check, matcher against the target or its declared `spmw_match_view`, the
  coverage/semantic contracts above, the shared facts
  `allo/spmw_semantics.py::ProgramFacts`) and then `compile_for_target`
  (enumerator, legality, movement stage, cost, `rank_matcher_placements`,
  materialization, runner). The APU v1 GVML vector plans and the Gemini-II
  composed-dot launch plans are candidates of that ranking with their
  effects stated in the typed candidate contract
  (`allo/spmw_candidate.py::CandidateEffects`), validated per candidate,
  on the selected placements and against the realized artifact; `layout=`
  names a candidate or family on every target and a supplied placement is
  rebound to the current program's candidate under the current policy
  (C-28); a host program the APU backends cannot honor, or a launch whose
  operands rebind a kernel's ABI, is refused before selection (C-29,
  C-29b); the contract is enforced field by field against the facts, the
  target, the priced graph and the realized program (C-30). Plain
  helper-function calls in ordinary source are refused at the shared
  boundary (C-27); dataflow regions remain the interprocedural form. The
  Gemini-II typed records and `APUv1Program` are frontends of this
  pipeline (rounds 15 and 16); the matcher backends' candidates do not
  declare `CandidateEffects` yet.
