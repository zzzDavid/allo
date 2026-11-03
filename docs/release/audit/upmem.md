<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# UPMEM backend release audit

Audit date: 2026-11-01. Code: `experiments/allo` branch `tenon` @ `8160d6d`
(working tree had only untracked `scripts/verifier_check_*.py`). Read-only
audit; probe scripts and logs under `/tmp/tenon-release/probes/upmem/`
(`p1_e2e.py`, `p2_zero_init_compile.py`, `p2_run.py`, `p3_domain.py`,
`p4_cbackend.py`, `p5_cost.py`, `p6_hostgroup.py`, `p8_misc.py`, `p9_forms.py`,
plus `*.log`). Every claim is tagged **executed** (observed by running code
here), **read** (established from source), or **suspected** (inferred, not
run). Line numbers are for revision `8160d6d`.

Environment used: `PYTHONPATH` to the in-tree MLIR python package, the
`spmw` conda interpreter, docker image `bongjoonhyun/upimulator`
(`sha256:57dc7591...`, created 2026-07-21), provenance archive at
`~/shared/tenon-artifacts/upmem/provenance/upimulator/` (binary sha256
`0ab6af91...`, snapshot tar sha256 `13ed0d80...`). `upmem_unavailable_reason()`
returned `None` here (executed), so every simulator probe below ran on the
paper's provenance uPIMulator.

Headline: **NOT READY** for this backend. Two blocker-grade silent wrong-code
paths were reproduced on the public route (U-01 int64/float operands silently
computed as int32; U-02 >64-bit signed arithmetic in host-placed groups and
in the functional oracle), the initialization checks of A08 are confirmed as
compile-time gaps (detected only by the runtime byte oracle), the paper's
391x65 multi-DPU scaling claim is not reproducible on the release route, and
normal execution still depends on an unpublished archive (A11).

---

## 1. Routes and entry points

### 1.1 Public route (read; executed end-to-end in P1)

```
allo.compile(workload, build_upmem_target(), upmem_cost[, backend="virtual"][, host_moves=...])
  -> compiler.py:692 customize() -> MLIR
  -> compiler.py:710 match_workload(target, module)           # MatchedOp trace (ADD/AXPBY/MAC/SCALE/INC/COMPACT/...)
  -> compile_for_target(..., module=schedule.module)          # compiler.py:719-729 (module is passed on all backends)
     -> spmw_autoschedule.py:1002 @register_enumerator("upmem") _upmem_enumerate
        -> spmw_upmem.py:1225 upmem_base_placements -> classify_upmem_kernel (:193)
        -> spmw_knobs.cross_with_knobs over UPMEM_KNOBS (:1247) -> upmem_knob_candidates/upmem_knob_emit (:1258/:1273)
           last knob attaches Placement.layout = masked_12_tasklet_linear_layout(_tasklet_block(...)) (:1282)
     -> spmw_autoschedule.py:323 derive_layout_properties -> upmem_kernel_features (physical features for pricing)
     -> rank_matcher_placements (cost program costs/upmem.py:289 upmem_cost; family priced once at anchor op, :300 family_event)
     -> spmw_codegen.py:1046 _BACKENDS["upmem"] = ("allo.spmw_upmem", "UpmemCtx", "_run_upmem")
     -> UpmemCtx.emit_program (spmw_upmem.py:2042): classify again, decision_from_extra, family.plan_for -> plan.device_source()
        -> UpmemCtx._bind_module (:2149): _check_coverage (:1690), _check_zero_init (:1728), _access_window (:1465) per role,
           functional oracle = backend/c.py emit_c_from_mlir(module, func, wrap_wide_integers=True).compile() (:2263)
           host-placed groups: same emit_c_from_mlir is the *production* host computation (:2164-2168)
  -> CompiledCallable.__call__ (compiler.py:481) -> backend_runner("upmem") = _run_upmem (spmw_upmem.py:2483)
     -> ctx.prepare(inputs) (:2300): runs the oracle on copies, packs MRAM heap image (pack_inputs :1877) and
        expected output window (pack_expected :1934) per DPU per execution -> GENERIC fixture (manifest.json + blobs)
     -> _stage_task_c writes benchmark/GENERIC/dpu/task.c in the staged provenance root (:2409)
     -> docker run bongjoonhyun/upimulator python3 benchmark/build.py --num_dpus 1 --num_tasklets 12 --benchmark GENERIC (:2547)
     -> SDK manifest re-check (:2573-2575) -> provenance uPIMulator --benchmark GENERIC --fixture_dir ... --skip_compilation true (:2552)
     -> uPIMulator byte-compares observed MRAM window vs expected; exit 2 on mismatch -> RuntimeError with first mismatch (:2586-2594)
     -> parse Logic[c_r_d]_logic_cycle per DPU (:2596); segment cycles = max over DPUs (:2603); total = sum over segments (:2604)
     -> gather_outputs (:1979) decodes observed bytes and overwrites the caller's arrays in place (:2615-2626)
     -> RunResult(cycles=total, backend="upmem", extra={"upmem": [records], "timing_scope": "uPIMulator DPU logic_cycle only"})
```

`backend="virtual"` (executed): same pipeline through `_bind_module`
(coverage/zero-init checks run, the oracle is compiled), then
`_run_virtual` evaluates the cost graph: `RunResult(backend="virtual",
cycles=<analytical>)` with `extra` holding critical path / utilization /
`model_fingerprint`. There is no typed `UPMEMProgram` route any more
(`compiler.py:660-663`, commit `eb4ebde` deleted `upmem_program.py` and
`upmem_abi.py`, executed `git log`).

### 1.2 Families (read, `spmw_upmem.py`)

| Family | Recognizer | Structural predicate (admits) | Plan template | Decision domain |
|---|---|---|---|---|
| `pointwise` | `:284` | one stream op in `{ADD, AXPBY}` (`_POINTWISE_OPERATIONS :267`), one static loop `[0,N)` step 1, both operands rank-1 memrefs indexed `[iv]`, not loop-carried, result memref distinct from both, store index `[iv]`. **No element-type check.** | `UPMEMElementwisePlan` (`upmem_physical.py:411`) | `_POINTWISE_DOMAIN :345` (5 curated decisions) |
| `reduction` | `:365` | one `ADD`, one loop, cell `out[c]` with `iv` not in `c`, value rank-1 `[iv]`, cell type ends `xi64>`, value `xi32>` | `UPMEMSumReductionPlan (:697)` | `_REDUCTION_DOMAIN :407` (1) |
| `matrix_vector` | `:421` | exactly one `MAC` (+ at most one `SCALE`), 2 or 3 static loops, loop-carried `acc`, matrix indexed `outer+[red]` with rank = loop count, vector `[red]` (TTV/flatten) or `[outer0, red]` (MMTV); scaled form needs rank-0 acc + SCALE store. **No element-type check.** | `UPMEMMatrixVectorPlan (:845)` | `_MATRIX_VECTOR_DOMAIN :525` (3; `shared_wram` has no plan builder) |
| `gemm` | `:558` | one `MAC`, 3 loops `(i,j,k)`, acc `[i,j]` loop-carried, lhs `[i,k]`, rhs `[k,j]`. **No element-type check.** | `UPMEMGEMMPlan (:1447)` | `_GEMM_DOMAIN :618` (5; `separate/shared_wram` has no builder) |
| `histogram` | `:734` | one `INC`, index term exactly `(x[iv]*m)>>k`, acc shape `(m,)`, guard empty or exactly `0<=bin<m` | `UPMEMHistogramPlan` (`upmem_irregular.py:284`) | `_HISTOGRAM_DOMAIN :784` (1; plan ignores decision) |
| `selection` | `:812` | `SELECT_ODD` alone, or `COMPACT`+`INC` under the same guard which must be `x&1 != 0`/`==1` on the copied value | `UPMEMSelectionFlagsPlan (:817)`; host Python compaction `_selection_epilogue :2454` | `_SELECTION_DOMAIN :895` (10; only `conditional_zero` has a builder) |
| `kmeans_distances` | `:930` | one `SQDIST_ACC` over `(p,c,d)`, x `[p,d]`, y `[c,d]`, acc `[p,c]` | `UPMEMKMeansDistancesPlan (:1405)` | `_SINGLE_DOMAIN :966` (1, ignored) |
| `kmeans_update` | `:980` | op set is a **superset** of `{MIN_SEL, ARGMIN_IDX, SCATTER_ADD, INC}`; no geometry, no roles; `placement="host"` | none: whole kernel runs on host via `emit_c_from_mlir` | `_SINGLE_DOMAIN` |
| `feature_gradient` | `:1019` | one `GRAD_LINEAR`/`GRAD_LOGISTIC`, 2 loops, label column `== features`, samples shape `(S, F+1)`, linear coeffs `alpha=-(2**s)`, `0<=beta<=62`; logistic `(1,2)` only | `UPMEMFeatureGradientPlan (:1747)` | `_SINGLE_DOMAIN` |

Rejection points (read): `classify_upmem_kernel :193` raises
`NotImplementedError("no UPMEM kernel family matches ops ...")` or "needs
exactly one <anchor> per work-id stream"; `_legal_decisions :1217` filters by
`physical_features` legality, and an empty result surfaces as
`RuntimeError("no candidate layouts produced for kernel ...")` (executed);
plan builders raise `UPMEMPlanBuilderRejected` -> `InfeasibleSchedule`
(executed: 12x16 MV "grouped MV output DMA must be between 8 and 2048
bytes"); `_check_coverage :1690` raises on any store no match covers
(executed: "store to lz ... is not covered by a matched UPMEM op");
`_check_zero_init :1728` raises only for the four `_ZERO_INIT_ROLES`
families (:1098) and only when **no** zero store to that memref exists.

### 1.3 Benchmark-identity dispatch (executed grep + read)

No production code dispatches on workload name, filename, or PolyBench
identity. `classify_upmem_kernel` docstring and implementation read only
`MatchedOp`s; the renamed-equivalent probe (P9, different region/kernel/
argument/loop names) produced byte-identical `task.c` to the benchmark
factory's GEMV (**executed**). Hits of kernel names in production files are:
docstrings/comments (`spmw_upmem.py:424-427`, `:758`, `:763`), region names
inside plan manifests ("histogram", "kmeans-point"), and the calibration
evidence IDs `_DEFAULT_RAW_EVIDENCE_BINDINGS` in
`upmem_physical_search.py:1483-1640` ("run-va-dma64", "gemv-shared-vector",
...). Those IDs name archived runs; the keys that select a factor at
compile time are structural (`_calibration_key`, `costs/upmem.py:214`), with
one shape-exact stratum for batched MV discussed in U-07.

---

## 2. Enforced vs advertised domain

| Advertised (paper 5-evaluation.tex UPMEM subsection; SKILL.md; docs) | Enforced by the release route | Evidence |
|---|---|---|
| 16 kernels (VA, RED, MTV, GEMV, GEVA, TTV, MMTV, HIST, SEL, KMEANS, LINREG, LOGREG, 1mm, 2mm, 3mm, CONV) fastest on all 16 | All 16 sources in `benchmarks/upmem/workloads.py` compile and the 16 layer-1 provenance rows pass; layer-2 cycles pass for va/red/hist (114534 for VA reproduced here) | executed: `pytest -k upmem test_paper_golden.py`: 19 passed, 13 skipped (`paper_full`) |
| "1-32 simulated DPUs, fixed 391x65 GEMV ... including uneven partitions" | 391x65 is rejected at 1 DPU and at 17 DPUs ("no candidate layouts produced"); `_split` (`workloads.py:46`) requires even division; `_matrix_vector_features` requires `rows % 12 == 0` and `reduction % 16 == 0`. The maintained scaling test uses 1152x128 instead (`test_upmem_fanout_gemv`). 1152x128 at 1 vs 32 DPUs executed here: 5,532,920 -> 180,154 cycles, outputs correct | executed P3, P8 |
| "coalesced layouts, buffered writes, tiled reductions" | Present as mechanisms: fused `[A_chunk, x_chunk]` packets (`UPMEMDataLayout.FUSED_REPLICATED`), per-tasklet output buffers with one `mram_write`, GEMM `Nc/Kc` tiles. Selected by cost over the curated lists | read |
| Section 3: shared autoscheduling lifecycle (match -> enumerate -> cost -> codegen) | Yes, through `_upmem_enumerate` and `rank_matcher_placements`; however candidates without a plan builder are priced and then dropped by codegen-feasibility probing (U-13) | read, executed (selection: 10 legal decisions, 1 realizable) |
| Section 4: layout as F2 `LinearLayout`; DPU/tasklet compute levels | `Placement.layout` is a masked 12-of-16 F2 carrier attached at the last knob; it is **not read** by codegen or costing (`derive_layout_properties` sets `tasklet_fanout` from the `tasklets` knob, `spmw_autoschedule.py:323-330`); templates compute tasklet ownership themselves. DPU level = `@allo.work(mapping=[N])` work-id partition, proven affine by `_access_window` | read (U-14) |
| SKILL.md: "Program lowering: allo/pim/upmem_program.py"; "emits DPU kernel C + host driver Go/C"; TENON slot | Stale: `upmem_program.py` deleted; slot is `GENERIC` from the provenance snapshot; host driver is Python | executed `ls`, `git log` (U-08) |
| SPMW_ARCHITECTURE.md:46-47, 69-86: `UPMEMProgram`, `upmem_abi.py`, "arbitrary multi-phase MLIR workloads", "matcher path has been removed" | Inverted: the matcher path is the only route; `UPMEMProgram`/`upmem_abi.py` do not exist | executed (U-08) |
| design/06, design/08: 30 PolyBench leaves as a functional virtual target with 64-DPU ABI | No such route exists in the package | read (U-08) |
| dtype: int32 data, int64 accumulators (SKILL/workloads) | Not enforced for pointwise/MV/GEMM: int64 and float32 operands are accepted and computed as int32 | executed (U-01) |
| Shapes "in the family" | Divisibility constraints from the cost-model feature functions (pointwise `N % chunk`, MV `rows % 12`, `K % 16`, GEMM `rows % 12`, `N % nc`, `K % kc`) reject realizable shapes; MV also needs `> 12` total rows per DPU | executed (U-06) |

Supported-domain summary that the source actually enforces (read + executed):
int32 operands only in practice (not checked), `mapping=[N]` even row/element
partitions with the work id appearing as `d*S + i`, pointwise ADD and
`a*A + b*B` with literal `a, b` (not `a*A - b*B`, not SUB/MUL/SCALE), MV with
`rows % 12 == 0`, `K % 16 == 0`, `rows > 12`; GEMM with `rows % 12 == 0`,
`N % 4..128`, `K % 8|16|64`; histogram `(x*m)>>k` with `m == bins`; selection
of odd values only; k-means distances with `|v| <= isqrt((2^31-1)/D)/2`;
gradients with the two coefficient patterns.

---

## 3. Findings

Severity: **blocker** = silent wrong code or a paper capability with no
release route; **major** = gate violation needing repair before release;
**minor** = quality/clarity.

### U-01 (blocker) Non-int32 operands are silently computed in int32 and wrapped

- Files: `allo/spmw_upmem.py:284` (`_recognize_pointwise`), `:421`
  (`_recognize_matrix_vector`), `:558` (`_recognize_gemm`) check memref rank
  and indices but never the element type; `pack_inputs :1877` and
  `pack_expected :1934` go through `_i32 :1794`
  (`np.asarray(values, dtype=np.int64).astype("<i4")`), which truncates both
  the device inputs **and the oracle's expected bytes** identically; `gather_outputs
  :1979` writes the int32 window back into the caller's int64/float array
  (`view[...] = ...astype(view.dtype)`, `:2626`).
- What happens: the DPU byte compare passes because the expected image was
  truncated the same way, so the run reports success with int32-wrapped
  results. Gate G2 ("integer widths ... overflow ... explicit failure").
- Reproduction (executed, `p2_run.py`):
  - `gemv-int64 large values`: `int64[48,64] @ int64[64]` with |values| < 1e5
    -> `RAN backend=upmem cycles=176648`; `y==A@x: False`,
    `y==int32-wrapped: True`, max |true| = 45,683,307,424.
  - `va-int64 large`: `int64` VA with |values| < 2^40 -> ran, `C==A+B: False`,
    `C==wrap32(A+B): True`.
  - `va-float32 fractional`: compiled; run failed with "first mismatch ...
    got 2, expected 3" (float 1.5+1.5 truncated); `va-float32 integral`: ran
    and "passed" via the int path.
- Repair: recognizers must require the plan's element type (i32 operands;
  i64 cell only for reduction/kmeans/gradient where the template is int64)
  and reject others with a message naming the type; `pack_expected` must
  never narrow the oracle's values (compare in the oracle's width or fail
  when a value does not fit); add int64/float negative tests.

### U-02 (blocker) >64-bit signed arithmetic is wrong in host-placed groups and in the functional oracle

- Files: `allo/backend/c.py:106-120` (`_portable_c11.replace_ap_integer`
  maps `ap_int<W>`/`ap_uint<W>` with `W > 64` to `uint64_t` when
  `wrap_wide_integers=True`); `spmw_upmem.py:2164-2168` uses that artifact
  as the **production** computation of `placement == "host"` groups
  (`kmeans_update`), and `:2263` as the oracle for every DPU group.
- What the code does: a signed 96/128-bit intermediate becomes unsigned
  64-bit, so arithmetic right shift becomes logical, signed compare becomes
  unsigned, floor division loses sign, and high-bit extraction returns 0.
- Reproduction (executed, `p4_cbackend.py`, through `emit_c_from_mlir`):
  `(A*B)>>8` on int64 -> got `72057594037927935` where `-1` expected;
  `A*B < 0` -> all `0`; `(A*B)//16` wrong sign; `(A*B)>>64` -> all 0;
  int32 `((A*B)*-32)>>8` (the natural GRAD_LINEAR source, 96-bit) -> got
  `72057594037927931` where `-5` expected.
  Production path (executed, `p6_hostgroup.py`): a k-means `update` host group
  computing `uC = (int64 sums * -(2**40)) >> 40` ran with
  `RAN backend=upmem cycles=49205`, `C_new` all zeros, expected `-sums`;
  `host record: [('dpu', 49205), ('host', 0)]`. No error.
- Evidence the hazard already shaped the benchmark: `benchmarks/upmem/
  workloads.py:265-269` adds `t: int64 = ...` with the comment "An int64
  temporary keeps the shift signed in the portable-C oracle (backend/c.py
  stores >64-bit temporaries unsigned)".
- Gate G2 (wrong numeric lowering; covert wrong host computation). Also
  A09 escalated from oracle-only to production.
- Repair: in `c.py` either lower wide signed temporaries to `__int128`/
  `unsigned __int128` with correct signedness (then reject only when the
  compiler lacks it), or reject `W > 64` for shift/compare/div/trunc-high
  ops and allow the wrap only for add/mul/low-trunc chains proven by a
  small pass. Do not use the same transformation for both oracle and
  device/host production; keep an independent oracle.

### U-03 (major) Initialization/accumulator semantics are not proven at compile time (A08 confirmed)

- Files: `_ZERO_INIT_ROLES :1098` lists histogram/selection/kmeans_distances/
  feature_gradient only; `matrix_vector`, `gemm`, `reduction` have no
  initialization check although their templates start from
  `accumulator = 0` / `total = 0` and overwrite the output
  (`upmem_physical.py:1355`, `:831`); `_check_zero_init :1728` accepts the
  existence of any zero store to the memref name anywhere in the function
  (no dominance, no index coverage, no guard analysis).
- Reproduction (executed, `p2_zero_init_compile.py` then `p2_run.py`): all
  of the following **compiled** on both `virtual` and simulator backends:
  (a1) GEMV `ly[i] += A*x` with no zero store; (b) histogram whose `lH[j]=0`
  loop runs **after** the counting loop; (c) histogram zeroing only bins
  `0..63`; (f) reduction `lP[0] += lA[i]` with no zero store. At run with
  nonzero initial outputs each failed only through the uPIMulator byte
  compare: a1 "got -5801, expected -6716"; b "got 14, expected 0"; c
  "element 64: got 13, expected 20"; f "got 9804, expected 22149". With
  zero initial outputs all four ran and matched NumPy. `backend="virtual"`
  returned cycles for all four without complaint. Correctly rejected:
  histogram with no zero store (d), guarded zero store (c2, by index
  analysis, not by the init check), `y = y + acc` and `C[i] = C[i] + A[i]`
  (no family).
- Gate G2 ("Finding one zero store somewhere is not a proof that an
  accumulator is initialized"; virtual/emitted artifacts must not succeed
  for programs the plan does not implement). Runtime detection exists only
  on the simulator route and reports a confusing build-log tail.
- Repair: make initialization part of each family's recognizer contract:
  either require a dominating full-window zero store (walk the IR: a store
  loop over the exact role window, before the accumulating loop, same
  control depth) or carry the initial value into the template (read-add-write
  of the output window). Apply to MV/GEMM/reduction too.

### U-04 (major) Paper scaling claim (391x65 GEMV, 1-32 DPUs, uneven partitions) has no release route

- Files: `allo/pim/upmem_physical_search.py:740` (`problem.rows %
  decision.tasklets`), `:622` (`logical_elements % chunk_elements`),
  `benchmarks/upmem/workloads.py:46` (`_split` raises unless even).
- Reproduction (executed, `p3_domain.py`): `gemv-391x65` and
  `gemv-391x65-17dpu` -> "no candidate layouts produced"; `gemv-96x64-8dpu`
  -> infeasible (12 rows/DPU gives a 4-byte output DMA). Fanout evidence
  exists only at 1152x128 (`test_upmem_fanout_gemv`, `paper_full`), which I
  ran at 1 and 32 DPUs: 5,532,920 and 180,154 cycles, outputs correct.
- Gate G1 (every paper evaluation path needs a documented execution route;
  "uneven partitions" cannot be expressed since all work-id streams must
  classify to identical geometry, `:206-213`).
- Repair: either implement uneven/last-DPU tails (different static extents
  per work id are already specialized per function; relax the identical-
  geometry requirement and the `rows % 12` legality, since the template pads
  rows) or record in CLAIMS.md that the 391x65 figure is historical and not
  reproducible by the release.

### U-05 (major) Legality catalogs reject realizable nearby shapes and come from the cost model, not the target (A07)

- Files: `upmem_physical_search.py:622,683,740,852,856` divisibility checks
  inside `_*_features`; `_legal :1206` uses them as the **only** legality
  filter; the templates themselves pad (`UPMEMElementwisePlan.padded_elements`,
  tail guard `element < TENON_ELEMENTS`, `UPMEMMatrixVectorPlan.padded_rows`,
  `row >= TENON_TOTAL_ROWS` guard).
- Reproduction (executed P3): pointwise N=1000, 12289, 8 rejected; N=1008
  admitted but only as `separate/64` (fused/256 illegal because 1008 % 32 !=
  0), so a divisibility artifact changes the selected layout; MV 100x100,
  96x100, 24x8 rejected; GEMM 20x32x32, 12x20x32 rejected; MMTV 4x8x32
  rejected. Error text "no candidate layouts produced for kernel 'k_0'" does
  not say why.
- Gates G3/G4 (legal domain derived from target facts and explicit policy;
  honest coverage scope). The curated decision lists are an acceptable
  finite catalog if labelled, but the legality should be the template's
  (DMA alignment, WRAM capacity, output DMA >= 8 bytes), with the cost
  features computed for the padded program.
- Repair: move divisibility out of legality (compute features on padded
  extents), keep true hardware constraints, and emit a reason string with
  the rejection.

### U-06 (major) Shape-exact and paper-shaped template paths

- `tasklet_private_wram` MV residency is legal only for exactly
  `(batches, rows, K, tasklets, dma, chunk) == (12, 16, 32, 12, 64, 16)`
  (`upmem_physical_search.py:792`, `upmem_physical.py:877`), the paper's
  MMTV shape; `_calibration_key` adds `(rows, reduction)` to the stratum for
  batched MV (`:1168`), so any other MMTV gets factor 1.0 (executed:
  `mmtv-12x16x64` and `mmtv-6x8x32` fall back to `fused_replicated`).
- `UPMEMGEMMPlan.exact_row_fast_path` (`upmem_physical.py:1515`) is a
  separate C template when `rows == 12` (the paper's 1mm/2mm/3mm rows);
  structural trigger, executed correct on 12x32x32 and 24x32x32.
- Selection supports the odd predicate only (`:812-905`; executed:
  `sel-even`, `sel-positive` rejected). Pointwise supports ADD and literal
  AXPBY only; `A - B`, `A * B`, `3*A`, and `3*A - 7*B` are rejected
  (executed; the last with the unhelpful "retained matcher kernel has no
  matched implementation"). `3*A + (-7)*B` compiles and runs correctly
  (executed).
- Gate G3: a new op or predicate in a promised family requires a
  compiler-side entry (`_POINTWISE_OPERATIONS`, a new `register_upmem_family`,
  a new plan class). Fail-closed, so not wrong code, but the release must
  state this domain explicitly and not call it "elementwise"/"selection"
  in general.
- Repair: document the exact domain in CLAIMS.md; generalize the pointwise
  template to the matched expression (it already parameterizes alpha/beta)
  and the selection predicate to the matched guard; retire the
  shape-exact `tasklet_private_wram` branch or derive its parameters.

### U-07 (major) Normal execution depends on an unpublished archive; no compatible-build path (A11 confirmed)

- Files: `allo/spmw_simenv.py:40` (`UPMEM_SDK_BUILD_MANIFEST_SHA256`),
  `:112` (`upimulator_bin` default under `~/shared/tenon-artifacts`), `:136`
  (`ensure_upimulator_root` extracts `source-snapshot.tar`), `:288-321`
  (`upmem_unavailable_reason`), `spmw_upmem.py:2573-2575` (hard failure on
  manifest drift). No environment override exists for the hash (executed
  grep).
- Reproduction (executed): with an empty provenance dir ->
  "uPIMulator binary not found at .../uPIMulator (set UPIMULATOR_BIN)"; with
  `UPIMULATOR_ROOT` = the in-tree checkout -> "uPIMulator benchmark slot
  missing at .../benchmark/GENERIC/dpu/CMakeLists.txt"; the in-tree binary
  has 0 occurrences of `fixture_dir`/`GENERIC` (provenance binary: 5); the
  in-tree `sdk/build` manifest is `9c2122ef...` vs pinned `d45e285c...`, so
  even with GENERIC restored the route fails with "not the pinned campaign
  build (set UPIMULATOR_ROOT)", whose suggested remedy is the setting that
  triggered it. `experiments/simulators/setup/uPIMulator/` ships only a
  Dockerfile; the `GENERIC` slot sources (`benchmark/GENERIC/*`,
  `src/assembler/prim/generic.go`) and the 145 prebuilt SDK objects exist
  only inside the archived tar (executed `tar tf`).
- Gate G7 ("a normal user must have a documented compatible-build path that
  does not require an unpublished archive of prebuilt objects").
- Repair: publish `tracked-changes.patch` + GENERIC slot as a versioned
  patch set under `experiments/simulators/setup/uPIMulator/`, document the
  SDK build recipe, and make the manifest pin a provenance *label*
  (`sdk_build_sha256` in records, golden tests assert it) rather than an
  execution precondition; add `TENON_UPMEM_ALLOW_UNPINNED_SDK` or similar
  with the result labelled "unpinned".

### U-08 (major) Architecture and design docs describe a deleted route

- `SPMW_ARCHITECTURE.md:46-47` list `allo/pim/upmem_program.py` and
  `allo/pim/upmem_abi.py`; `:69-86` say UPMEM "supports arbitrary
  multi-phase MLIR workloads through `UPMEMProgram`" and that "the
  contraction-only `UPMEMCtx`/uPIMulator matcher path has been removed",
  the reverse of the current code; `design/06-upmem-reference-backend.md`
  and `design/08-upmem-polybench-execution-matrix.md` describe a 30-leaf
  PolyBench virtual target and 64-DPU ABI that no longer exist;
  `.claude/skills/pim-upmem/SKILL.md` cites `upmem_program.py` and the
  `TENON` slot. Executed: the modules are absent; `git log` shows deletion
  in `eb4ebde`. `docs/source` has no UPMEM page (executed grep).
- `allo/pim/__init__.py:124-158` still exports `UPMEMKMeansPlan`,
  `UPMEMStableSelectionPlan`, `rank_upmem_physical_candidates`,
  `select_upmem_physical_plan`, and the `upmem_calibration` report API; none
  is reached from `allo.compile` (executed grep: only `scripts/` and
  `tests/perf` use them). `allo/pim/upmem_analysis.py` is imported only by
  `apu_v1_program.py`.
- Gate G7 (architecture docs must match the implementation and explain how
  to add a primitive/backend). Repair: rewrite the UPMEM sections around
  `spmw_upmem.py` families, the GENERIC fixture runner, provenance
  requirements, and the extension path (`register_upmem_family`); prune or
  mark the unreachable exports.

### U-09 (major) Virtual cycles outside the calibration stratum are reported without status; virtual and simulated scopes differ

- Files: `allo/pim/costs/upmem.py:249` (`_physical_kernel_cycles` applies
  `PHYSICAL_CALIBRATION_FACTORS.get(key, 1.0)`), `:98` (24 factors,
  "Generated by: python /work/shared/users/phd/nz264/tmp/coder014_factors.py",
  an unpublished path); `upmem_physical_search.py:1383-1420` computes
  `calibration_status` but the matcher route never surfaces it
  (`RunResult.extra` from `_run_virtual` has only `model_fingerprint`).
- Executed (`p5_cost.py`, `p1_e2e.py`): VA 12288 (calibrated exact stratum,
  factor 2.0687): virtual 115,048 vs simulated 114,534 (0.4%). VA 1024
  ("shape extrapolation"): 10,188 vs 15,603 (1.53x). MV 48x64: 90,255 vs
  125,689 (1.39x). With `host_moves=gemv_host_program`, virtual includes
  SCATTER/BCAST/GATHER steps (493,803 vs 359,488 without) while the
  simulator total excludes host transfers by design (`_TIMING_SCOPE`);
  both are called `cycles`.
- Units/provenance (read): all cycles are uPIMulator one-DPU cumulative
  `logic_cycle` at 350 MHz logic clock; factors are mean ratios
  `anchor.logic_cycles / raw_analytical` over `DEFAULT_UPMEM_CALIBRATION_EVIDENCE`
  (27 anchors at the reference shapes 12288 / 96x128 / 12x192x32 /
  12x128x64, each bound to archived run hashes). Per-op scalar rules
  (`ADD_INSTRUCTIONS` ... `FLOAT_SQRT_INSTRUCTIONS`, DRAM timings) are
  labelled analytical assumptions. The perturbation test
  `test_upmem_calibration_flips_ranking` passes (executed): scaling one
  factor by 10 changes the fingerprint and flips the VA winner.
- Gate G5 ("a model without a validated performance domain must not silently
  advertise performance predictions"; scopes documented). Repair: carry
  `calibration_status` and `timing_scope` into placement extra and virtual
  `RunResult.extra`; state the host-transfer scope difference in docs.

### U-10 (major) `kmeans_update` recognizer is an unbounded zero-cost host path

- `spmw_upmem.py:980-996`: any kernel whose op set contains
  `{MIN_SEL, ARGMIN_IDX, SCATTER_ADD, INC}` (superset test, no geometry, no
  roles) is placed on the host; `costs/upmem.py:309-310` prices it at
  latency 0; `:2164` executes its full MLIR through the wide-int-unsafe C
  backend (U-02). Executed: the probe kernel in `p6_hostgroup.py` with an
  extra wide-shift epilogue was accepted as `kmeans_update` and ran on the
  host at `logic_cycles: 0`.
- Gates G2/G5 (host partition must be explicit and priced; no covert CPU
  substitution). Repair: tighten the recognizer to the argmin+scatter
  structure with roles/geometry, require an explicit user-visible host
  placement (e.g. in the host program), and price host work in a labelled
  host-time scope.

### U-11 (minor) Selection compaction and cross-DPU merge are not compiler artifacts

- `_selection_epilogue :2454` performs the stable compaction and count in
  Python on the simulated flags and only cross-checks against the oracle
  (executed `sel-odd`: count ok, prefix ok, tail untouched). The reduction
  fan-out returns per-DPU partials; the merge is done by the caller
  (`test_upmem_fanout_reduction`, executed `red12288-4dpu-sim`: partials
  summed in NumPy). Both are consistent with the paper's device-only
  metric but should be documented as host steps outside generated code and
  outside the cost graph.

### U-12 (minor) Priced-but-unrealizable candidates and vestigial knobs

- Selection: 10 legal decisions, only `conditional_zero` has a plan builder
  (`:882-887`); MV: `separate/shared_wram` is legal in features but rejected
  by `_matrix_vector_plan :498`; GEMM `separate/shared_wram` likewise.
  Resolution is by codegen-feasibility probing after pricing (A02).
  Histogram/kmeans/gradient/reduction plans `del decision`, so their
  `dma_bytes`/`chunk` knob values are carried but unused. Executed: selection
  N=1536 -> 10 legal, winner `conditional_zero/16`.

### U-13 (minor) Carried F2 layout does not determine the program

- `upmem_knob_emit :1282` attaches `masked_12_tasklet_linear_layout(block)`;
  `derive_layout_properties` (`spmw_autoschedule.py:323-330`) only tests
  `isinstance(layout, LinearLayout)` and reads `tasklets` from the knob.
  Templates hardcode `tid * ROWS_PER_TASKLET + local_row`. Agreement is by
  construction, not derivation (G4 "selected layout determines the
  executable" holds for the decision fields, not for the F2 object).

### U-14 (minor) Diagnostics

- "no candidate layouts produced for kernel 'k_0'" (no reason);
  "retained matcher kernel has no matched implementation" for `3*A - 7*B`;
  "UPMEM index analysis: unbound value %arg2" for a guarded zero store;
  `ly[i] = acc` matched as `COMPACT` so the scaled-GEMV form without a
  multiplier reports "no family matches ['COMPACT','MAC']"; run failures
  append 40 lines of SDK liveness-analysis noise after the useful "first
  mismatch" line (executed).

### U-15 (minor) Test/package hygiene

- `tests/pim/test_upmem_matcher.py:37-48` imports
  `scripts/prepare_upmem_tenon_campaign.py` by path (not a package);
  `test_paper_golden.py` and `benchmarks/upmem/workloads.py:artifacts_root`
  default to `~/shared/tenon-artifacts` (evaluation-only, skip-guarded; fine
  if documented as optional). `setup.py:108` uses `find_packages()` with no
  data declaration; the UPMEM route needs no data files (templates are
  Python strings), so packaging risk here is low (read).

Positive observations (executed): coverage check fails closed on an
uncovered store; rename invariance holds; the simulator byte compare is real
and independent of the DPU output (mismatches raise rather than substitute
oracle values); records carry simulator/snapshot/SDK/`task.c`/`task.c.o`/
fixture hashes and `timing_scope`; the three default golden rows reproduce
exactly; `test_gemv_upmem` passes (58 s); cost perturbation reaches
selection.

---

## 4. Constants / specialization inventory

| Item (file:line) | Category (TASK.md) | Judgement |
|---|---|---|
| `UPMEM_ACTIVE_TASKLETS = 12`, `UPMEM_PADDED_TASKLETS = 16` (`upmem_physical.py:27-28`); `-DNR_TASKLETS=12` | Schedule policy frozen as a constant | Not a hardware limit (max 24, `targets.py` declares 24). Every plan is 12-tasklet; the `tasklets` knob only ever takes 12. Label as policy or make it a decision. |
| `UPMEM_MIN/MAX_DMA_BYTES = 8/2048`, 8-byte alignment, WRAM 64 KiB, MRAM 64 MiB (`upmem_physical.py:29-32`, `_search.py:38-42`) | Physical capacities/ISA | Sourced (UPMEM SDK/uPIMulator); acceptable. |
| `targets.py:465-906` `build_upmem_target` (64 DPUs, 24 tasklets, mram/wram/iram/atomic sizes, 24 GPRs) | Target facts | Acceptable; note `mapping={"dpu": 64}` while the runner accepts any DPU count. |
| `_POINTWISE_DOMAIN :345`, `_MATRIX_VECTOR_DOMAIN :525`, `_GEMM_DOMAIN :618`, `_SELECTION_DOMAIN :895` ("campaign's curated list") | Finite candidate catalog | Acceptable only with explicit "heuristic catalog, not optimal" scope; currently tied to `scripts/prepare_upmem_tenon_campaign.py` by test. Unrealizable members (U-12). |
| `_REDUCTION_DOMAIN`, `_HISTOGRAM_DOMAIN`, `_SINGLE_DOMAIN` (one decision each, plans ignore it) | Placeholder | Vestigial; no cost choice exists (A07). |
| Divisibility legality (`_search.py:622,683,740,852,856`) | Mis-filed: cost-model reference restriction used as legality | Defect (U-05). |
| `(12, 16, 32, 12, 64, 16)` tasklet-private MV shape (`_search.py:792`, `physical.py:877`) | Template precondition equal to one paper shape | Defect/limit (U-06). |
| `_calibration_key` MMTV stratum `(batches, rows, reduction)` (`_search.py:1168`) | Empirical model validity domain | Acceptable as documented non-extrapolation, but status must surface (U-09). |
| `PHYSICAL_CALIBRATION_FACTORS` (24 hex floats, `costs/upmem.py:98`) + `DEFAULT_UPMEM_CALIBRATION_EVIDENCE` (27 anchors with run hashes, `_search.py:1715`) | Measured cost coefficients | Provenance and units present; generator path is an unpublished `/tmp` script; validity domain = reference shapes; extrapolation factor 1.0 silently (U-09). |
| `ANALYTICAL_*` (8 B/cycle DMA, 24 cycles/call, 256/barrier, 4/tasklet, divisor 4, fill 13, revolver 11, 14 stages) (`costs/upmem.py:28-92`, `_search.py:42-52`) | Analytical model constants | Labelled as initial assumptions / uPIMulator defaults; acceptable. |
| Per-op instruction counts (`MUL_STEP_INSTRUCTIONS = 32`, `FLOAT_* `, DRAM `T_*`) (`costs/upmem.py:29-74`) | Analytical assumptions | Labelled; only reach non-family events (reduction/irregular families, which have one candidate anyway). |
| `GEMM backpressure` proxies `underfill`, `accumulator_pressure` with literal 16/64 (`_search.py:924-925`) | Dimensionless heuristic terms | Commented as proxies; acceptable but uncalibrated. |
| `_kmeans_max_abs` (`spmw_upmem.py:924`), `_check_preconditions :1761` (int32/int64 exactness bounds) | Derived numeric precondition | Good: derived from dimension and checked on actual inputs at prepare time (runtime, with clear `ValueError`). |
| Histogram `depth < 31`, guard exactly `0<=bin<bins`, `m == bins` (`:734-770`, `irregular.py:300`) | Template precondition | Structural; acceptable, message is clear. |
| Feature-gradient coefficient patterns `alpha=-(2**s)`, logistic `(1,2)`, `shift=5, overflow=8` (`:1019-1063`) | Template precondition with literal magic for logistic | `formula, shift, overflow = 1, 5, 8` is undocumented; needs derivation or doc. |
| `_LOCAL_ABI_ID = 1 << 62` (`:74`) | ABI convention shared with `ir/builder.py` | Acceptable, cross-referenced. |
| `UPMEM_SDK_BUILD_MANIFEST_SHA256`, binary/snapshot sha pins (`spmw_simenv.py:40`, `test_paper_golden.py:251-253`) | Archived provenance pins | Acceptable for golden rows; defect as an execution precondition (U-07). |
| `GEVA_ALPHA/BETA`, `GEMV_ALPHA`, `_HIST_*`, `_KM_*`, `_GRAD_*`, `UPMEM_CASES` (`benchmarks/upmem/workloads.py`) | Benchmark dimensions | Correctly outside the compiler; the production code never reads them (executed grep). |
| `_i32`/`_i64` packers (`spmw_upmem.py:1794/1836`) | Lowering rule | Defect: silent narrowing (U-01). |
| `wrap_wide_integers=True` -> `uint64_t` (`c.py:112-120`) | Lowering rule | Defect for shift/compare/div/high bits (U-02). |
| `_selection_epilogue` odd-mask Python (`:2454`) | Host partition implemented by hand | Document or generate (U-11). |
| `--logic_frequency 350`, `--num_simulation_threads 16`, `--num_channels 1` (`:2553-2558`) | Simulator configuration | Hard-coded simulator settings; frequency does not affect cycle counts; document. |

---

## 5. Open questions

1. Should `cycles` in `RunResult` for UPMEM remain "max over DPUs of DPU
   logic cycles, summed over segments" (device-only, excluding host
   transfer/merge), while the virtual graph includes explicit host moves?
   Both are labelled `cycles`; the paper's metric is the former.
2. Is the 12-tasklet policy a deliberate release decision or an artifact of
   the calibration campaign? The target declares 24; the knob exposes only
   12.
3. Which of the 16 paper kernels are intended as the released UPMEM
   "supported family" domain, versus examples? The paper text says
   "16 kernels"; SKILL.md says PrIM; the release domain as enforced is
   narrower than either name suggests (U-06).
4. The 391x65 scaling and "uneven partitions" figures: are they to be
   re-expressed on the maintained route (needs tail support) or declared
   historical in CLAIMS.md?
5. What is the intended public story for the `GENERIC` slot and the patched
   SDK: a published uPIMulator fork/patch set, or a documented "paper
   reproduction only" dependency with an unpinned compatible mode?
6. Should `kmeans_update`-style host groups exist on the matcher route at
   all, or should host work be expressed only through `allo.host_program`
   with a priced host scope?
7. Is `allo/pim/upmem_analysis.py` (MLIR summarizer, now used only by the
   APU v1 program route) still part of the UPMEM story, or should it move?
