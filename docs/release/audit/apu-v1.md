# GSI APU v1 (Gemini-I) backend: release-readiness audit

Read-only audit of `experiments/allo` (branch `tenon`, 8160d6d) against TASK.md
gates G1-G8. No device launch, no board lock touched; all probes are compile-only
and live under `/tmp/tenon-release/probes/apuv1/` (`p1_routes.py`,
`p2_recognizers.py`, `p5_gemm_plans.py`, `p7_runtime_gates.py`, `p10_misc.py`,
`probe_util.py`). Each claim below is labelled **executed (compile-only)**,
**read**, or **suspected**. File paths are relative to `experiments/allo/`.

Verdict for this backend: **NOT READY**. Two blockers (silent wrong-code in the
four native recognizers; silently dropped nonzero accumulator on the vector
device route), several majors (unauthorized FP16 reassociation by default,
`backend="virtual"` on `APUv1Program` launches hardware, paper evidence
produced by out-of-tree harnesses, no production board lock / profile freshness,
template assets discovered from a user home directory).

---

## 1. Routes and entry points

### 1.1 Dispatch in `allo.compile` (read)

`allo/compiler.py:598-630, 683-703`:

1. Typed route: `isinstance(workload, APUv1Program)` -> `compile_apu_v1_program`
   (`allo/pim/apu_v1_program.py:674`). Accepts `backend in {None,"virtual","functional"}`
   but the backend value is **never used** (`CompiledAPUv1Program.__init__`
   ignores `backend`; `run()` always builds and launches, lines 522-632).
2. Source route, contraction frontend: a plain `@allo` function on `apu_v1` without
   retained matcher scopes goes to `compile_apu_v1_vector_workload`
   (`allo/pim/apu_v1_vector_program.py:659`); `NoContractionError` falls through.
3. Source route, matcher: everything else goes to `match_workload` +
   `compile_for_target` with `allo/spmw_apu_v1.py` (`APUv1Ctx`, `_run_apu_v1`),
   the legacy four-APUC grouped FP16 MAC path.

### 1.2 Route map for the five paper kernels (executed, compile-only; `p1_routes.py`)

| Paper kernel | Workload file (artifact) | Frontend | Production symbol | Selected route/plan | Estimate (cycles) |
|---|---|---|---|---|---|
| MatMul 256x1024x1024 uint16 | `tenon-artifacts/apuv1/tenon/matrix_multiply/workload.py` (plain `@allo` grid/reduction) | source, contraction | `APUv1VectorCallable` | `temporal_dma_coalescing_broadcast_friendly_acc8` (default ranking agrees with the producer's forced `layout=`) | 13,681,547 |
| K-Means P32768 D24 K10 | `.../kmeans/workload.py` (`APUv1Program([APUv1Phase(vectorize="required", argument_bounds=...)])`) | typed | `APUv1ProgramCallable` -> `apu_v1_distance_argmin.py` | `gvml_squared_l2_argmin_center_streaming` | 643,130 |
| Histogram N32768 B256 | `.../histogram/workload.py` (typed, `result_names=("counts",)`) | typed | `apu_v1_histogram.py` | `gvml_dense_histogram_pair_count` | 72,605 |
| Linear Regression 8xN4096 | `.../linear_regression/workload.py` (typed) | typed | `apu_v1_moments.py` | `gvml_bivariate_moments_fused_group_reduce` | 54,299 |
| Word Count Q128 N32768 | `.../word_count/workload.py` (typed) | typed | `apu_v1_record_frequency.py` | `gvml_record_frequency_resident_chunks_pair_count` | 140,070 |

So only MatMul is an ordinary `@allo` source workload; the other four are typed
`APUv1Program` objects whose single phase requests `vectorize="required"` and is
matched by one of four **per-kernel recognizer+template modules** registered in
`allo/pim/apu_v1_native_vector.py:98-103`. The artifact producer
(`tenon-artifacts/apuv1/scripts/prepare_tenon.py:117-123, 237-258, 320-330`) used
`allo.compile(..., backend="virtual", layout=MATRIX_PLAN)` for MatMul and
`allo.compile(..., backend="virtual")` for the other four, then called the
private `allo.pim.apu_v1_program._generate_project` to materialize projects and
ran them with its own locked runner (`scripts/run_campaign.py`,
`scripts/measure_once.py`), **not** with the production `run()` paths.
`tests/pim/test_paper_golden.py:520-550` mirrors this (compile-only for four
kernels; only `histogram` has a board test).

### 1.3 What each recognizer proves (read: `apu_v1_histogram.py:60-155`, `apu_v1_distance_argmin.py:104-229`, `apu_v1_moments.py:87-233`, `apu_v1_record_frequency.py:75-179`)

All four are **regex scans over the retained MLIR text** plus ABI shape/dtype
checks. Common pattern: count opcodes (`affine.for`, `arith.addi`, `arith.muli`,
`arith.cmpi eq`, `scf.if`, `affine.load/store`, `memref.alloc`, `linalg.fill`),
check loop bounds literally (`0 to 32768`, `0 to 256`, ...), tie a few SSA values
(comparison operands, store index) to the formal `%argN`. Parameters derived from
IR/ABI: role names, `dimensions`, `candidates`, `batches`, `points_per_batch`,
`queries`, `chunks`, comparison predicate (`slt|ult`), optional
`argument_bounds` (argmin only). Hard-coded: 32768 lanes, 256 bins, 5 moment
columns, products order `x*x, y*y, x*y`.

**Not proven** (and therefore silently ignored; see V1-01): the increment
constant of the counted update (`+= 1`), the initial fill value of the result
(`= 0`), which scalar local is stored to `labels` (argmin), the mapping of the
first two moment columns to `x`/`y`, the initial value of `best_distance`.

Classification per TASK.md table: these are "operator/kernel templates" whose
structural recognition is **incomplete**; they are closer to per-benchmark
adapters than general lowerings (one module per paper kernel, fixed lane count,
fail-closed on unrelated changes but fail-open on several semantic ones). They
do not consult benchmark names, filenames, or fixture values (verified: renames
keep route, estimate and emitted source identical modulo role identifiers;
`p10_misc.py` output `identical after role-identifier substitution: True`).

### 1.4 Vector contraction route (read: `apu_v1_vectorize.py`, `apu_v1_vector_program.py`, `apu_v1_vector_codegen.py`, `apu_v1_vector_cost.py`)

`analyze_apu_v1_contractions` (`apu_v1_vectorize.py:661`) recovers one
`C[i,j] += A[i,k]*B[k,j]` update per store (mulf/muli/and/or/xor/xnor-popcount),
checks output axes are distinct enclosing loops, exactly one reduction axis, no
accumulator read inside the product, no foreign indices. 1-D GEMV is lifted with
a unit axis (`:832`). `_candidate_recipes` (`:735`) yields a finite catalog:
`baseline_spatial_reduction`, `temporal_svp`, `temporal_dma_coalescing`,
`temporal_dma_coalescing_broadcast_friendly`, `..._acc{2,4,8}[_l4]`, and
`spatial_gemv_group_reduction` when the second output axis is a singleton.
Selection (`apu_v1_vector_program.py:454-475`): every candidate is realized to
GVML (`_realize`), infeasible ones are dropped (VR capacity, L3 size), feasible
ones are scored with `estimate_apu_v1_realization` on the **materialized**
instruction inventory, ranked by cycles, ties broken by catalog order.
`layout=<name|APUV1Plan>` bypasses ranking. The selected realization's
`runtime_artifact` (frozen project files + ABI manifest) is what
`run_apu_v1_vector` builds and launches (`apu_v1_vector_runtime.py:268-430`),
and the ABI manifest is re-derived at launch and compared
(`:342-350`) -- good G4 linkage from plan to executable.

### 1.5 Scalar ARC route and hybrid manifest (read)

`APUv1Phase(vectorize=False)` (the default) lowers the whole function via
`emit_c_from_mlir` to C11, regex-rewrites local arrays onto an L4 bump
allocator and replaces `sqrt` by a Newton helper (`apu_v1_program.py:198-262`),
and launches on APUC 0 only (`NUM_APUC = 1` string patch at `:365`).
`apu_v1_hybrid.py` only produces a manifest; any discovered vector region raises
`NotImplementedError("... removed ...")` (`apu_v1_program.py:472-477`), and
multi-phase programs raise at `:408`. No benchmark-name usage was found in any
APU v1 module (`grep -niE "phoenix|kmeans|histogram|word_count|linear_regression|matrix_multiply" allo/pim/apu_v1_*.py` -> only the histogram module's own docstring).

---

## 2. Enforced vs advertised domain

| Family | Advertised (docs/paper) | Enforced by code (executed unless noted) | Gap |
|---|---|---|---|
| Dense histogram | "dense 256-bin histogram over one 32K frontier" (`design/10-...md:375`) | accepts only `values: ui16[32768]`, `counts: ui16[256]`, `== bin` on the outer IV, 1 addi; rejects `!=`, `<`, `-=`, B=255, N=32769 (shape rejects are by construction: `values.shape != (32768,)`) | **accepts `+= 2` and nonzero init** (V1-01) |
| Squared-L2 argmin | "bounded uint16 squared-L2 plus strict argmin" | P=32768 only; 1<=D<=48; 2<=K<=65535; `<`/`ult` only; D=23,K=11 and D=48,K=2 accepted; D=49, K=1, `<=`, `>` rejected | **accepts `labels[p]=best_distance` and `best_distance=100` init** (V1-01); rejection messages come from the contraction fallback, not the recognizer (V1-13) |
| Bivariate moments | "batched bivariate sufficient statistics" | `batches*points == 32768`, points power of two; 16x2048 and 1x32768 accepted; 8x4095, 4x4096 rejected | **accepts swapped `sum_x`/`sum_y` columns and nonzero init** (V1-01) |
| Record frequency | "fixed-width record/query frequency", <=10 chunks | records=32768 only; 1<=C<=10; 1<=Q<=256; odd Q (127) accepted; C=11, Q=257, N=16384 rejected | **accepts `+= 2` and nonzero init** (V1-01) |
| Contraction (FP16/uint16/bitwise/xnor-popcount) | "rank-2 contractions, singleton-axis GEMV" | 1000x1024x64, 1024x1000x64, 100x100x100 fp16, 1000x64x1024 and 256x1024x1000 uint16, 7x5x3 uint16 all compile and realize; 1-D GEMV lifted | nonzero initial `C` silently zeroed on device for temporal plans (V1-02); default picks a reassociating plan for GEMV (V1-03) |
| Scalar ARC | "complete programs on APUC 0" | any Allo function the C backend accepts; `_C_TYPES` dtypes | `sqrt` replaced by Newton helper with wrong results above ~1e8 (V1-12) |
| Typed route backends | `"APUv1Program supports the device, virtual, or functional backend"` (`compiler.py:607`) | backend ignored; always device | V1-04 |
| Matcher route | grouped FP16 MAC over four APUCs (`spmw_apu_v1.py`) | vadd fails with `NotImplementedError: apu_v1 enumerator: trace is missing one of x/y/acc roles` | reachable but mostly dead; output-shape guessing and missing-output tolerance (V1-09) |

---

## 3. Findings

Severity: **blocker** = silent wrong code or misleading success on a released
route; **major** = gate violation needing a repair before release; **minor** =
documentation/hygiene.

### V1-01 (blocker, G2/G3/G6) Native recognizers fail open on semantic mutations and emit the paper kernel

- Files: `allo/pim/apu_v1_histogram.py:83-147`, `apu_v1_record_frequency.py:114-169`,
  `apu_v1_moments.py:120-224`, `apu_v1_distance_argmin.py:156-194`.
- What the code does: counts `arith.addi` occurrences but never checks the added
  constant; counts `linalg.fill` but never checks the fill value; in the argmin
  recipe checks that *one* store to `labels` exists but not which local is stored;
  in the moments recipe checks load counts per role (4/4/5) which are invariant
  under swapping columns 0 and 1.
- Executed (compile-only, `p2_recognizers.py`): with `vectorize="required"` the
  following all return the **same route, same estimate and byte-identical device
  source (modulo role names)** as the canonical kernel:
  `counts[b] += 2`, `counts: uint16[256] = 7` (histogram);
  `labels[point] = best_distance`, `best_distance: uint16 = 100` (k-means);
  `stats[batch,0] += y; stats[batch,1] += x` (moments), `stats = 3` init;
  `counts[q] += 2`, `counts = 5` init (word count).
  The existing regression `tests/perf/test_apu_v1_native_reductions.py:182` only
  mutates `cmpi eq -> ne` and `muli -> addi`.
- Gate: G2 ("Recognizing one MAC is not proof that an entire kernel is ..."),
  G3 (template must prove complete semantics), G6 (change operations/constants at
  fixed shape).
- Repair: either (a) prove the missing facts from IR (increment constant == 1
  via the `arith.constant` feeding the addi; `linalg.fill` operand is the zero
  constant; the value stored to `labels` is the SSA chain rooted at the
  `best_center` slot updated inside the `scf.if`; column-to-role binding for
  moments via the load origin of each `affine.store` value), or (b) replace the
  regex recognizers with a structural matcher over the parsed MLIR (the
  contraction analyzer in `apu_v1_vectorize.py` already has a def-use parser that
  could be reused) and parameterize the increment/init so the template emits
  `count*k + init` instead of assuming them.

### V1-02 (blocker, G2) Vector device route silently discards a nonzero carried output

- Files: `allo/pim/apu_v1_vector_runtime.py:309-328` (output image is allocated
  as an *output-only* role; comment at 322-324 says "current contraction plans
  initialize the carried output to zero"), `allo/spmw_apu_v1_build.py:482-492`
  (`host.c` `calloc`s every output role and copies zeros to L4 before launch),
  `apu_v1_vector_runtime.py:396-410` (device result overwrites `arrays[name]`).
- Executed (compile-only, `p7_runtime_gates.py`): the frozen ABI manifest for
  FP16 1024x64x1024 is `inputs: ["left","right"], outputs: ["result"]` -- the
  inout `result` is never transferred to the device. Only the spatial-scatter
  case checks `np.any(arrays[out] != 0)` and raises (`:300-307`); temporal plans
  (the default winners) have no check.
- Consequence (suspected, needs board run to demonstrate numerically): a program
  `result += A@B` called with nonzero `result` returns `A@B` on hardware while
  `backend="functional"`/`"virtual"` return `result0 + A@B` -- a device/virtual
  disagreement.
- Repair: transfer the packed output image as an input (the ingress route
  `dma_l4_l1_32k`/`load_vr` for `output_compute` already exists in the plan,
  `apu_v1_vectorize.py:1490-1514`; the host just never fills it), or reject
  nonzero carried outputs for every plan kind, not only spatial.

### V1-03 (major, G2/G1) FP16 reassociation is chosen by cost with no permission flag; the paper's "source-order" Gemini-I result is not the default route

- Files: `apu_v1_vectorize.py:805-823` (`spatial_gemv_group_reduction` recipe),
  `apu_v1_vector_codegen.py:290-315` (`fp16_contraction_ops` -> `gvml_add_subgrps_f16_grp`),
  `apu_v1_vector_program.py:454-475` (cost ranking).
- Executed (compile-only, `p5_gemm_plans.py`): for `y[i] += W[i,k]*x[k]` with
  4096x256 FP16 the default selection is `spatial_gemv_group_reduction`
  (1,424,455 cycles) whose device source contains `gvml_add_subgrps_f16_grp`
  (tree reduction). Source-order candidates `temporal_svp` and
  `temporal_dma_coalescing` realize but are only reachable by name via `layout=`.
  `grep -rni "reassoc|source_order|strict" allo/pim/apu_v1_*.py allo/compiler.py`
  finds no permission flag. The `virtual` run of the same compiled object
  reproduces **source-order FP16** exactly (`virtual output == source-order f16
  reference: True`; 3,668/4,096 elements differ from fp32-rounded), i.e. the
  virtual backend's numerics do not correspond to the selected device plan.
- Paper mapping: Sec. 5.2 says source-order FP16 on Gemini-I matched 4,096
  outputs and that regrouping "requires explicit permission". That run was
  produced by `reviews/experiments/strict_fp16_gemv_audit.py` /
  `shared_gemv_*` (outside `experiments/allo`), not by a release API flag.
- Gate: G2 ("unauthorized reassociation"), G1 (mechanism must be reachable via a
  documented release API).
- Repair: add an explicit reduction-order policy (e.g. `allo.compile(...,
  reduction_order="source"|"reassociate")` or a per-phase/precision policy),
  filter candidates by it before ranking, default to "source" for floating
  point, and make the functional backend execute the realization's
  `execute_numpy` (`apu_v1_vector_codegen.py:2704`) rather than the Python
  function so virtual numerics match the selected plan.

### V1-04 (major, G2/G8) `APUv1Program` ignores `backend`; "virtual"/"functional" launch hardware

- Files: `allo/compiler.py:603-609` (advertises virtual/functional),
  `allo/pim/apu_v1_program.py:404, 522-528, 674-679` (backend parameter unused;
  `run()` always gates on the device and launches).
- Executed (compile-only with `TENON_APU_V1_PCI_NODE=/nonexistent`,
  `p7_runtime_gates.py`): `allo.compile(histogram_program, ..., backend="virtual")(values)`
  raises `SimulatorUnavailable: GSI device not present ...`, proving the call
  path is the device runner. On this host (board present) the same call would
  build and launch on the board. `test_apu_v1_paper_kernel_compiles_to_gvml`
  never calls the compiled object, so the suite does not notice.
- Repair: implement `virtual` (return `estimate()` cycles with `backend="apu_v1-virtual"`,
  no outputs or NumPy outputs from the retained C/MLIR) and `functional`
  (execute the portable C or an interpreter) or reject those backends for the
  typed route with a clear error.

### V1-05 (major, G1/G5) Paper calibration result (5,887.79 + 876.21/call) is not in the released cost program

- Files: `allo/pim/costs/apu_v1.py:118-123` prices `FAST_COUNT_2M_G32K` as a flat
  245 cycles per call (`MICRO_FAST_COUNT_2M_G32K = MICRO_COUNT_M_G32K`, an
  inference), with no activation term. The fitted affine model lives only in
  `tenon-artifacts/agent-cost-calibration/2026-07-27-gemini-i/confirmatory-affine-event-v1/freeze/affine-event-model.json`
  (`activation_cycles: 5887.79`, `slope_cycles_per_call: 876.21`) and the study
  code under `studies/agent_cost_calibration*` (read).
- Consequence: `TENON_ANALYSIS.md` records route estimates off by -56.5%
  (histogram), -42.6% (moments), -18.6% (word count) versus measured medians; the
  graphs carry `analytical: True` but no validity-domain or "uncalibrated" flag
  (`apu_v1_native_cost.py:26-35`).
- Repair: land the fitted event model (with provenance, units, validity domain
  = even-Q record-frequency/histogram routes) in `costs/apu_v1.py` or an
  explicitly named calibrated cost spec, and mark uncalibrated native estimates.

### V1-06 (major, G1/G4) The "tile 64 vs tile 256 (3.845%)" and four-candidate ranking claims are not reproducible through the public API

- Read: `apu_v1_vectorize.py:1008-1035` picks the GEMV reduction tile
  deterministically by `min(choices)` (fewest matrix tiles, then fewest resident
  VRs, then largest tile); there is no cost-ranked tile candidate set in
  production. The 3.845% figure comes from `reviews/experiments/run_common_gemv_apu.py`
  (per-tile `device.c` payloads) and is discussed in
  `reviews/experiments/Q2-UTILIZATION-AND-OVERLAP.md:55-59`.
- Executed (compile-only): FP16 1024x64x1024 now has **10** catalog candidates,
  6 realizable (acc4/acc8 rejected: "needs 3/7 additional writable VRs; only 1
  remain"), and the default winner is `..._acc2` (6,553,468), not either of the
  two plans in the 12.71x row. Both row plans are reachable by `layout=` and
  realize (`temporal_dma_coalescing` 93,711,488 est.; `..._broadcast_friendly`
  8,552,316 est.; the second emits 9 `direct_dma_l4_to_l1_32k`, lookups and
  subgroup duplication, matching `tests/pim/apu_v1/vector_gemm/test_vector_gemm_apu_v1.py:49-57`).
  Cost perturbation does reach selection: `MICRO_LOOKUP_L4_SLOPE=0` flips the
  winner to `..._acc2_l4`; `MICRO_DMA_L4_L1_32K=2e6` re-ranks
  `temporal_dma_coalescing` below the spatial baseline (G5 satisfied for the
  contraction route).
- Gap: `tests/perf/test_apu_v1_vector_compile.py:33-38` still asserts exactly four
  candidates (only true for tiny shapes where acc variants are skipped at
  `apu_v1_vectorize.py:1699`); `RESULTS.md` and the golden row
  `vector_gemm_temporal_dma_coalescing_broadcast_friendly` = 7,145,109 (re-captured
  2026-10-04) differ from the frozen paper's 7,349,987 (2.9%, outside the 2%
  tolerance applied to the paper value) -- document as a measurement drift, not a
  code change.
- Repair: expose the GEMV tile domain as ranked candidates (or state in docs that
  tile selection is a fixed heuristic), move the review-harness experiments'
  reproduction recipe into `docs/release/`, and reconcile the golden/paper numbers
  in `CLAIMS.md`.

### V1-07 (major, G8) Production runtimes have no board lock and no profile-freshness check

- Read: `apu_v1_program.py:379-400, 604, 615` and `apu_v1_vector_runtime.py:193-197, 411-413`
  drain the cumulative `ledag-ssh flo` log once after the run and take the
  **last** `total ... crun:` match; a run whose PROF_PRINT did not flush silently
  reports an older run's counter or `cycles=None`. `spmw_apu_v1.py:213-230` takes
  the max of the last four. The only lock is the pytest fixture
  `tests/pim/conftest.py:97-113` (`tests/pim/.apu_v1.lock`); `run()` paths retry
  3x on "no valid context" instead (`apu_v1_program.py:590-599`). The artifact
  runner that produced the paper numbers does both correctly
  (`tenon-artifacts/apuv1/scripts/measure_once.py:102-139`: flock + pre/post
  `flo` snapshot + prefix proof + exactly one fresh total).
- `.claude/skills/pim-apu-v1/SKILL.md` ("if that line is missing the harness
  raises") contradicts the code (`spmw_apu_v1.py:396-402` docstring: yields
  `cycles=None`).
- Repair: move the lock + pre/post snapshot + single-fresh-total logic from the
  artifact scripts into `allo/pim/apu_v1_*runtime`, raise on missing/ambiguous
  totals, record board/SDK identity in `RunResult.extra`.

### V1-08 (major, G7) Build harness copies GSI/Synopsys SDK example files discovered from a user home path

- Read: `allo/spmw_simenv.py:195-200` defaults the template to
  `Path.home()/shared/accelerator-hub/gsi-apu/example-gvml`;
  `spmw_apu_v1_build.py:51` and `apu_v1_program.py:350-353` copy `Common/` wholesale
  plus `gsi_dma.c`, `gsi_dma.h`, `gsi_device_profiling.h`.
- Executed (compile-only, `p7_runtime_gates.py`): the frozen vector project
  contains 12 files byte-identical to the template (`Common/arc.tcf` 59 KB,
  `Common/arc_for_gcc_for_gnu-ccac.tcf` 68 KB, `Common/common.mk` 32 KB authored
  "Gal Shalif" (GSI), `memory.x`, two shell scripts, `.clang-format`,
  `gsi_dma.c/.h`, `gsi_device_profiling.h`) and 4 generated files (`Makefile`,
  `struct.h`, `host.c`, `device.c`). None of the template files carries a
  license header; the `.tcf` files are Synopsys ARC configuration files.
- Gate: G7 ("Ordinary compilation does not import ... a particular user's home
  directory"; "proprietary dependencies need documented installation rather than
  unauthorized bundling").
- Repair: require `TENON_APU_V1_TEMPLATE_DIR` (or an SDK-relative default such
  as `/usr/local/gsi-apu/<ver>/examples`), document how to obtain the GSI example
  project, and consider replacing `gsi_dma.c` with Tenon-owned code (it is 4.8 KB
  of DMA helpers) if GSI cannot license redistribution.

### V1-09 (major, G2) Legacy matcher runner guesses output shapes and tolerates missing outputs

- Read: `allo/spmw_apu_v1.py:232-270` (`_apu_v1_output_specs` picks the largest
  input's shape and `uint16` as the output spec), `:536-542` (`if p.exists():`
  silently omits a missing output file), `:396-402` (`cycles=None` on missing
  profile). `_apu_v1_prepare_io` raises `NotImplementedError` for anything but a
  dense FP16 matrix contraction (`:375-379`).
- Executed (compile-only): a uint16 vadd on `apu_v1` dies in the autoscheduler
  (`NotImplementedError: apu_v1 enumerator: trace is missing one of x/y/acc
  roles`) -- a finite error, but the route is effectively dead code for the
  release and its runner violates G2 if reached (e.g. via `@allo.work` dataflow
  GEMV, `tests/pim/test_gemv_all_targets.py:284-290` currently takes the
  contraction frontend instead).
- Repair: delete `spmw_apu_v1.py`'s runner or make it raise on unknown output
  roles and missing files; route all APU v1 source workloads through the
  contraction/native/scalar routes.

### V1-10 (minor, G3) Cost-less compile silently selects the slowest plan

- Executed (compile-only, `p10_misc.py`): `allo.compile(gemm, target, None)` for
  FP16 1024x64x1024 selects `baseline_spatial_reduction` (7.8e9 est. cycles
  versus 6.5e6 for the ranked default) because `_select_plan` falls back to
  `candidates[-1]` (`apu_v1_vector_program.py:228-232`, comment claims "most
  capable MICRO recipe") which is now `..._acc8_l4`, is infeasible, and the
  recovery loop (`:495-513`) takes the first realizable catalog entry.
- Repair: require a cost spec for selection or default to the first feasible
  entry of an explicitly documented preference order.

### V1-11 (minor, G7) Dead/removed API surface and stale docs

- Read: `APUv1Phase` accepts `precision_policy`, `vector_layout`, `produces`,
  `bindings`, `dependencies`, `zero_initialize` (`apu_v1_program.py:72-79`) but
  any vector region raises `NotImplementedError` (`:472-477`) and multi-phase
  programs raise (`:408-411`); `APUv1PrecisionPolicy` is exported from
  `allo/__init__.py:83` with no consumer. `SPMW_ARCHITECTURE.md:57-58` lists
  `allo/pim/apu_v1_hybrid_runtime.py` ("four-APUC shard packing ... hybrid device
  execution") which does not exist. `design/10-apu-v1-reference-backend.md:297-366`
  still describes hybrid 2mm/3mm execution. `spmw_apu_v1_build.py:8` cites
  `.orchestrator/specs/017-apu-v1-run-spec.md` (not in tree).
- Repair: remove or implement; update architecture docs to the three live routes.

### V1-12 (minor, G2) Scalar route `sqrt` replacement is numerically wrong for large inputs

- Read: `apu_v1_program.py:240, 248-254` replaces `sqrt(` with `allo_sqrtf(`, a
  16-iteration Newton loop starting at `max(value, 1)`.
- Executed (NumPy float32 emulation, `p10_misc.py`): rel. error 0 up to 1e6,
  4e-6 at 1e8, **0.74 at 1e10, 14.3 at 1e12**. Any float program with large
  squared sums on the scalar route is silently wrong.
- Repair: use the toolchain `sqrtf` (link `-lm`) or iterate to convergence with a
  magnitude-aware initial guess.

### V1-13 (minor, G2) Rejection diagnostics are misattributed

- Executed: a mutated k-means (`<=`) fails with
  `IllegalContractionError: kmeans_assignment: vector contraction needs at least
  one output axis`; a mutated histogram fails with `NoContractionError: selected
  APU phase contains no rank-2 contraction`. The native registry returns `None`
  without a reason (`apu_v1_native_vector.py:106-127`) and the contraction
  fallback's error is surfaced instead.
- Repair: have discoverers return structured rejection reasons and raise a
  dedicated `NoNativeVectorRouteError` listing the closest recipe and the failed
  condition.

### V1-14 (minor, G2) Readback errors are finite but not actionable

- Read: `apu_v1_program.py:608-611` and `apu_v1_vector_runtime.py:399-401` do
  `np.fromfile(...).reshape(shape)`; a short or empty output file raises
  NumPy's `cannot reshape array` instead of naming the role and expected bytes.
  `host.c` itself checks `fread`/`fwrite` byte counts (`spmw_apu_v1_build.py:470-475, 500-510`).

---

## 4. Constants and specialization inventory (classification per TASK.md)

| Item | Location | Category | Assessment |
|---|---|---|---|
| 32768 lanes, 15 writable VRs + index VR, 48 VMRs, 8 markers, 4 APUCs, 1 MiB L3, 14 GiB L4 | `targets.py:909-960`, `VR_LANES` constants in codegen/recipes | ISA/physical capacity | Named, sourced (MICRO'25 Fig. 3); fine. Duplicated as literals in five modules; centralize. |
| Power-of-two `GVML_P2_*` enum table | `apu_v1_moments.py:24-48`, `apu_v1_vector_codegen.py:278` | ISA encoding | Legitimate instruction-selection table. |
| 512-byte L4->L3 transaction rounding | `apu_v1_distance_argmin.py:61-63`, `apu_v1_record_frequency.py:45-50` | derived dimension | Computed from shape; units clear. |
| `MAX_CANDIDATES=65535`, `1<=dims<=48`, `<=10 chunks`, `Q<=256`, `(1<<20)` center image cap | recipe modules | legality limits | 65535 and 1 MiB are hardware facts; 48 and 10 are VMR/VR residency limits (should be derived from the target's 48 VMRs / 15 VRs rather than literals); `Q<=256` is unexplained. |
| Histogram `HISTOGRAM_BINS=256`, `items=32768` | `apu_v1_histogram.py:25-26` | template specialization | Shape-only template; general bins would need a different plan. Document as the supported domain. |
| Increment `1`, fill `0`, column order, stored local | implicit in all four recipes | **unproven template precondition** | V1-01. |
| `argument_bounds` -> `unsigned_order_proven` | `apu_v1_distance_argmin.py:196-227` | explicit user policy | Legitimate: only removes a sign-flip; semantics otherwise exact in the uint16 ring. |
| Measured GVML/DMA cycles (`F16_*`, `MICRO_*`, `DMA_*`), fits in `_reduce_cycles`, `GROUPED_*`, `ARC_SCALAR_CPI=52` | `costs/apu_v1.py:30-123, 161-185` | measured cost coefficients | Provenance comments present (dates, commits, shapes); units = ARC cycles. `OR_16=13`, `FAST_COUNT=COUNT`, `AND_M=13`, `ARC_STORE=PIO store` are stated inferences; `ARC_SCALAR_CPI` from one GEMM. No validity-domain flags; paper's fitted model absent (V1-05). |
| `MICRO_PIO_CALL_STARTUP=3365` fitted from "baseline/SVP delta" | `costs/apu_v1.py:78-84` | fitted coefficient | Fitted on the same 1024x64x1024 layout experiment the paper reports; held-out validation not recorded here. |
| Candidate catalog (`_candidate_recipes`), GEMV tile heuristic `min(choices)` | `apu_v1_vectorize.py:735-823, 1013-1030` | schedule policy / finite catalog | Explicit, structurally applicable, legality-checked at realization. Must be labelled heuristic (no optimality claim); GEMV tile is not ranked (V1-06). |
| `rows_per_vr = min(8, ...)`, `table_slots = 32768 // column_tile`, `resident_count <= 10` | `apu_v1_vectorize.py:1016-1017, 1295, 1364` | derived dimensions / capacity | Derived from extents and VR count; 8 and 10 are residency budgets that should reference `WRITABLE_VRS`. |
| `scratch_bytes = 8`, `allo_l4_alloc` 8-byte alignment | recipes, `apu_v1_program.py:223-262` | derived/ABI | Fine. |
| `NUM_APUC = 4` -> `1` string patch; `core * (sz/NUM_APUC)` per-core slicing | `apu_v1_program.py:365`, `spmw_apu_v1_build.py:530-534` | ABI | Works but fragile; make host emitter take `num_apuc`. |
| Default paths: `/usr/local/gsi-apu/13.7.1/...`, `/usr/local/include`, PCI vendor `0x1e4c`, `~/shared/accelerator-hub/...` | `spmw_simenv.py:44-48, 188-240` | environment discovery | Env-overridable; vendor scan is good. Home-path template default violates G7 (V1-08). |
| Retries `3`, backoff `[1,2]`, timeouts 600/900/300 s, `"no valid context"` marker | runtimes | operational policy | Documented in build manifest; OK. |
| Paper medians, `_APU_V1_MATRIX_PLAN`, `_APU_V1_NATIVE_ROUTES` | `tests/pim/test_paper_golden.py:92-104, 125-131` | evaluation oracles | Correctly confined to tests; production selection agrees without them (executed). |
| `_LOCAL_ARRAY` regex + `sqrt` rewrite | `apu_v1_program.py:55-61, 240` | lowering by text substitution | Fragile; V1-12. |

---

## 5. Open questions / what needs a board run to verify

1. **V1-02 numerically**: run `result += A@B` with nonzero `result` through
   `allo.compile(gemm, target, apu_v1_cost)(A, B, C0)` on the board and compare
   with `C0 + A@B`; expected (suspected) wrong result on temporal plans.
2. **V1-03 numerically**: run the 4096x256 FP16 GEMV with `layout="temporal_dma_coalescing"`
   versus the default `spatial_gemv_group_reduction` and confirm which matches
   source-order FP16 bit-exactly (paper Sec. 5.2 claims the former behaviour).
3. **V1-07**: confirm whether `PROF_PRINT` of a failed/early-exit kernel leaves the
   previous run's `total` as the "newest" line (stale-cycle attribution).
4. Re-measure the two 12.71x-row plans at the current revision to reconcile
   7,349,987 (paper) vs 7,145,109 (golden row re-captured 2026-10-04).
5. Whether the four native routes still pass their independent oracles after the
   V1-01 repair (any change to the recognizers must be re-validated on the board
   with the `tenon-artifacts/apuv1/inputs/*` fixtures and fresh reviewer inputs).
6. GEMM shape neighbours that compile (1000x1024x64 fp16, 1000x64x1024 uint16,
   100x100x100 fp16, 7x5x3 uint16) have no execution evidence; the artifact and
   golden tests only exercise the paper shapes and 1024x64x1024.
7. Whether redistribution of `example-gvml` files (`Common/*.tcf`, `common.mk`,
   `gsi_dma.c`) is permitted by GSI/Synopsys; if not, the release must document
   installation and ship no copies (currently none are in the tree, but the
   build cannot run without them).

### Not verified in this audit (read-only limits)

- Device execution of any route (board intentionally untouched).
- The `_emit_device_c` legacy grouped path (`spmw_apu_v1_build.py:267-409`) was
  read only; it is not reachable from the five paper kernels.
- `apu_v1_layout.py` relations were read at the API level (`APUV1Plan.__post_init__`
  validation, `TransferRouteStep.metrics`) but their algebra was not
  independently re-derived; the realization-time ABI check (`abi_manifest_json`
  equality) is the mechanism that ties plan to launch.
