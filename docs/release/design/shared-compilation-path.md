<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Shared compilation path: required migration

Status: **required, in progress**. User scope addition on 2026-11-02.
This is gate G9 of the containing repository's `TASK.md`; release cannot be
declared ready without it. Slices 1 and 2 of the migration map below landed
in round 12 and slice 3 in round 13 (section "Implemented stage contracts");
slices 4 and 5 remain. This document is the migration map and the record of
what is implemented, not a completion receipt.

## Implemented stage contracts (round 12)

Every ordinary source workload, for every target, now enters `allo.compile`
through the same front stages (`allo/spmw_pipeline.py::normalize_source`):

| Stage | Contract | Implementation |
|---|---|---|
| frontend normalization | retained MLIR module of the `@allo` function | `customize(workload, enable_tensor=False)` |
| matching | `MatchTrace` with store coverage, axis roles, accumulator initialization, output sinks, resolved store/operand coordinates, guards, source loop names (`extra["loop_names"]`, new) | `match_workload(matcher_view(target), module)`; a target whose hardware description declares no matchable operation names its source-idiom view through `target.spmw_match_view` (Gemini-II: `build_apu_g2_contraction_patterns()`), so the view is a declared property of the target, not a branch in the driver |
| semantic validation | dropped effects, unsupported accumulator initializers, unproven conversions, operand/result aliases, copy and precision contracts raise `UnsupportedWorkloadError` before any planner | `require_match_coverage(trace, target=view, defer_unknown=True)` |
| semantic analysis | `ProgramFacts` (`allo/spmw_semantics.py`): parameters (ordinal, shape, element type with signedness from the frontend's `itypes` / `unsigned` attributes, retained identity); per contraction: loop nest with static trips and source labels, reduction/output/batch loops from the matcher's roles, inputs and accumulator as `BufferAccess` records with resolved coordinates and plain-dimension loop variables, accumulator initialization, output sinks (`OutputSink`), guards, product/accumulate operations, element types, reduction policy, alias facts; epilogues (`Epilogue`: operands, literal constants, result, store coordinates); matches outside those two shapes (`other_matches`); matches in other functions (`foreign_functions`) | `analyze_program(view, module, trace, parameter_names=...)`; derived once per compilation (`SourceProgram.program_facts()`), never from MLIR text |

### Round 13: both APUs' plans are candidates of the common selection stage

After the shared front stages every target, including both APUs, continues
through the same driver (`allo/spmw_codegen.py::compile_for_target`):

| Stage | Contract | APU v1 (plain-function contraction) | Gemini-II |
|---|---|---|---|
| candidate enumeration | `autoschedule` enumerator `(matcher view, bucket matches) -> [Placement]`; a placement's `extra` states `candidate_plan`, `candidate_decision` (name-free physical identity, part of the schedule-search key), `candidate_families`, `coverage`, `value_layouts`, `transfers`, `launch_order`, `abi`, `whole_device_launch`, `reduction_policy_pruned`; the backend's intrinsic recipe is the `payload` | `apu_v1_vector_program.apu_v1_vector_candidates`: the recipe catalog of the facts-derived `ContractionAnalysis`, regrouping plans pruned under a strict float policy (nothing left raises `ReductionOrderError`), each plan realized to GVML (`APUv1VectorCandidate`, unrealizable plans kept with their reason); operands placed on the VR classes the `MAC` declaration admits; a dataflow region's work items still enumerate the grouped program (`_apu_v1_grouped_enumerate`), which refuses a strict float policy and a carried accumulator | `apu_g2_source_route.apu_g2_source_candidates`: the contract from the shared facts, `enumerate_apu_g2_launch_plans` (fusion x reduction chunks x output tiles), one `APUG2SourceCandidate` per legal plan with the whole catalog and the rejected plans; operands placed on the view's `seg0`/`seg1` |
| legality | `spmw_legality.check_placement(view, matches, candidate)` on every candidate (operand storage classes against the op declaration) | `MAC` src/dst `any_vr` | view `MAC` (`seg0`, `seg0` -> `seg1`), `AXPBY` (`seg1`, `seg1` -> `seg1`) |
| movement stage | `analyze_kernel_movement(view, ...)`; a plain function retains no group shape, so the stage reports `not-analyzed` and the candidates keep their own layouts | not analyzed (recorded) | not analyzed (recorded) |
| cost | `spmw_plan.build_execution_graph` delegates to the context's `candidate_execution_graph(target, trace, layout, bound_cost)` when the placements are its candidates (None otherwise: generic lowering) | `APUv1Ctx.candidate_execution_graph`: `estimate_apu_v1_realization` over the materialized GVML inventory; an unrealizable plan raises `InfeasibleSchedule` (a recorded rejection) | `APUG2SourceCtx.candidate_execution_graph`: `price_apu_g2_launch_plan` (every launch's issued recipe, back to back) |
| selection | `rank_matcher_placements`: scores, ties, rejections, joint evaluation, `KernelRanking.candidates` (new: the realized placement per scored position); `layout=` is a `candidate_filter` (`select_candidates`: a candidate name, a declared family name or a backend plan object restricts the ranking; an unknown name raises naming the candidates and those the policy pruned) | `layout="temporal_svp"`, `layout="spatial_gemv_group_reduction"` (family, cost-ranked among tiles), `layout=<APUV1Plan>` | `layout="single/k2"` |
| materialization | `ctx.emit_program(trace, layouts)` in cost order (probe); the context's `program` is the realized backend program; `runtime_route` names it | `APUv1Ctx.emit_program` -> `APUv1VectorProgram` (device source = the command stream; route `apu-v1-vector-gvml-v1`) | `APUG2SourceCtx.emit_program` -> `APUG2SourceProgram` (launch list = the command stream; route `apu-g2-composed-dot-v1`) |
| work grid | `_check_work_grid(target, trace, layouts=...)`: a single item is accepted when every selected placement declares `whole_device_launch` | the plan decomposes the one item over the APUC | the launches cover the whole problem |
| artifact and runner | `Compiled` (graph, cmds, ranking, runtime ABI) wrapped by `CompiledCallable`; `backend_program` is the context's program and its documented `INTROSPECTION` names are forwarded (`compiled.selected_plan`, `compiled.plan`) | `_run_apu_v1`: vector program on the board or the functional host model; grouped program through the low-mode project | `_run_apu_g2`: pack, launch, oracle check |

### Round 14: the candidate contract is enforced, not described

- `allo/spmw_candidate.py::CandidateEffects` (coverage: function,
  operations, inputs, outputs; value layouts; transfers with direction and
  route; launches; ABI inputs/outputs; device scope) is the one typed record
  both plan backends attach to their candidates (`extra["effects"]`).
  `validate_candidate_effects` checks it against the work item's semantic
  program (every external destination covered, every value a program
  buffer, every ABI name a parameter, at least one launch) in
  `autoschedule` before costing (a violation is a recorded rejection) and
  in `compile_for_target` on the selected placements; the context's
  `check_realization` compares the realized program (APU v1 realization
  ABI; Gemini-II contract outputs and launch count) with the declared
  effects after materialization.
- `rebind_supplied_layouts`: a caller-supplied placement is a decision.
  The current enumeration (legality, current reduction-order policy) is
  searched for the same physical decision; a backend plan without one is
  rebound by plan name or refused; a handle-only placement whose mode the
  policy pruned raises `ReductionOrderError`. No supplied payload is ever
  executed (C-28).
- `validate_orchestration` on the backend contexts: the APU plan backends
  declare that they execute one launch of the whole problem per kernel
  group and own their transfers; a host program with repeated launches,
  host transfer steps or explicit host moves is refused before selection
  instead of being dropped (C-29). Implementing repeated launches on these
  backends is future work; the matcher backends keep their schedule
  handling (Samsung honors schedules; AiM/UPMEM/AttAcc price them through
  the graph and are not changed here).
- The APU v1 MLIR-text contraction parser is gone; the hybrid route's dead
  vector-region planning went with it.

### Round 15: launch bindings, the enforced contract, typed Gemini-II records

- Launch bindings (C-29b). `LaunchStep.operands` retains the base buffer
  names of every `allo.launch`; `launch_schedule(host_program, trace,
  module=...)` derives each kernel's declared ABI buffers (`spmw.abi_value_ids`
  for dataflow kernels, the parameters for a plain entry) and refuses a
  launch with a different arity or order before selection, for every
  target. The APU backends' `_single_launch_orchestration` also requires the
  launches to cover the kernel groups once each in program order.
- Enforced contract (C-30). `CandidateLaunch(index, kind, operations,
  instances, reduction, accumulator, covers)`; `validate_candidate_effects`
  checks coverage equality, a value layout for every input and output over
  the program's loop labels (plus declared `synthetic_axes`) on a storage
  the target declares, every loop covered by some layout, transfer
  completeness, launches indexed 0..n-1 whose instance ranges tile each
  covered operation's result instances and whose reduction ranges tile its
  reduction extent, ABI equality; `check_graph_consistency` compares the
  priced graph (launch count, plan) with the declaration; the contexts'
  `check_realization` compares value layouts, transfers and launches with
  the materialized program.
- Typed Gemini-II records (`allo/pim/apu_g2_record_frontend.py`).
  `normalize_apu_g2_record` states a record's semantics as synthetic
  matches (the structural facts the matcher stamps: loops, axis roles,
  resolved coordinates, accumulator initialization, reduction policy) and
  the derived `ProgramFacts` (`frontend` = the record); the `apu_v2`
  enumerator emits the record's realization as one candidate
  (`apu_g2_record_candidates`), legality checks it against the view (which
  declares the typed `ADD`/`MUL`/`DIV`/`SUM` operations, not matchable from
  source), the candidate contract, the record's recipe graph through
  `candidate_execution_graph`, ranking, `APUG2RecordProgram` as the
  materialized program and the shared `Compiled`/`CompiledCallable`
  artifact (the record's call convention is the `RecordWorkload`
  signature; ``out`` may be omitted). Accumulate epilogues are a
  zero-initialized local dot plus an `ADD` with the seed input. The typed
  and composed callables are deleted. `APUv1Program` remains on
  `_PROGRAM_ROUTES`: its scalar ARC programs are whole-function lowerings
  and its native recipes are proven by their own structural executor; that
  migration is the remaining slice 4 item.

Removed in round 13: the planner-hook registry `_SOURCE_PLANNERS` and
`compile()`'s target-name branches; `APUv1VectorCallable`,
`compile_apu_v1_vector_workload`, `search_apu_v1_vector_plans` and the
route's own `grid_search`; `compile_apu_g2_source_workload` and
`APUG2SourceProgram._select` (the program is the realized selected plan;
`repetitions` rebuilds that plan's launches). `require_matcher_route_policy`
treats `apu_v1` and `apu_v2` as policy-aware routes because their
enumerators consult the policy.

Function boundary (review round 12): `normalize_source` refuses ordinary
source whose entry function calls plain helpers
(`require_single_function_or_dataflow`), for every target, before matching:
the matcher would otherwise analyze the helper's own argument order and one
call site, so a reordered call computed `R @ L` and a repeated call planned
one contraction. Dataflow regions (`df.kernel` callees) remain the
supported interprocedural form. The facts are always derived for the entry
function and stamped on the trace (`trace.semantics`,
`match.extra["program_facts"]`, `extra["semantic_facts"]`) for the
enumerators.

Backend planning after the shared stages (round 12 state, superseded by the
table above for the APUs):

| Target | Planner | Consumes | Status |
|---|---|---|---|
| Samsung, AiM, UPMEM, AttAcc | matcher route (`compile_for_target`: `autoschedule` enumerators, legality, movement stage, `rank_matcher_placements`, backend ctx) | the trace (the facts are available through `SourceProgram.program_facts()` but these enumerators still read `MatchedOp` directly) | unchanged |
| Gemini-II (`apu_v2`) | `allo/compiler.py::_plan_apu_g2_source` -> `apu_g2_source_route.compile_apu_g2_source_workload(program=...)` -> `analyze_apu_g2_source(ProgramFacts)` -> `APUG2SourceContract` -> launch-plan enumeration, pricing, ranking -> `APUG2SourceProgram` | the shared facts only. Removed: the route's own `_FunctionFacts` module walk (argument `itypes`, local `unsigned`, identity-store check), its `_map_dims` access-map parser, `_memref_shape_and_bits` and `_loop_trips`; the contract's domain rules (integer lanes 2..24 bits, power-of-two reductions, at most two products, result covers the output, `_check_widths`) remain as target legality over the facts | slice 1 done |
| APU v1 (`apu_v1`), plain function with a matched contraction | `allo/compiler.py::_plan_apu_v1_vector` -> `apu_v1_vector_program.compile_apu_v1_vector_workload(program=...)` -> `apu_v1_vectorize.apu_v1_contraction_analysis(ProgramFacts)` -> `ContractionAnalysis` -> recipe catalog -> `grid_search` over realized plans -> `APUv1VectorCallable` (`.trace`, `.program_facts` retained) | the shared facts only. Removed from the public route: `analyze_apu_v1_contractions`, the regex MLIR-text parser (it remains, marked, for the typed hybrid route `apu_v1_hybrid.discover_apu_v1_hybrid_manifest`, slice 4). The vector plans' restrictions are applied as legality over the facts (`contraction_analysis_from_facts`): one unguarded `range(N)` nest, one reduction loop indexing both inputs, plain loop-variable subscripts, two distinct input buffers, accumulator a function argument (carried or zero-filled) or a local scalar copied once to an argument (new: the `acc = 0; ...; C[i, j] = acc` form compiles and executes), one 16-bit element type, loop extents equal to buffer extents (new explicit rejection of partial coverage) | slice 2 done |
| APU v1 dataflow region (`@allo.work` scopes) or plain function without a contraction | matcher route (`spmw_apu_v1.APUv1Ctx`, grouped FP16 MAC) | the trace | unchanged |
| typed records (`APUv1Program`, `APUG2TypedProgram`, `APUG2ComposedContractionProgram`) | `_PROGRAM_ROUTES` | their own records | slice 4 |

Consequences observed and tested (`tests/pim/test_shared_semantics.py`):

- The shared semantic checks now run before every APU plan is selected. A
  contraction under `if r < 4:` was planned by the APU v1 text analysis as
  the whole 8-row problem (ISSUES V1-17; virtual plan `temporal_svp`, axis
  extents misread); it is now refused on APU v1, Samsung and AiM with the
  same coverage reason ("under a guard ... matches no target op"). The
  nonzero / partial output initializers, the operand/result alias and the
  Gemini-II narrowing-copy and wide-operand cases are refused by the shared
  stage with the shared error type; the tests that expected the routes' own
  error types were updated to the shared contract (the semantic reason is
  unchanged).
- Renamed programs give equal `ProgramFacts.manifest()` and the same APU v1
  plan and estimate; the five Gemini-II paper kernels compile to the same
  contracts, packings and fingerprints as before (`tests/perf/test_apu_g2_source_route.py`).
- Domain note (APU v1): the text parser also accepted `and`/`or`/`xor`
  products and the `popcount(xnor(a, b))` binary form. Those forms are not
  matchable on the APU v1 target yet (only `MAC: acc + x * y` is declared
  matchable), so they are refused by the shared coverage check with the
  "matches no target op" reason instead of planned; ISSUES V1-18 records the
  extension path (declare the matchable patterns; `math.ctpop` is already
  traced as `popcount`).

Not yet done (the remaining slices):

- `APUv1Program` does not normalize into the shared contracts (slice 4,
  remaining); the hybrid manifest still reads the scalar ABI from MLIR
  text;
- the matcher backends' enumerators read `MatchedOp` directly and declare
  no `CandidateEffects`; the facts are available on every match
  (`extra["program_facts"]`) but Samsung, AiM, UPMEM and AttAcc do not
  consume them yet;
- stage traces on the Gemini-II card and the installed artifact (slice 5);
  the APU v1 board and the three simulators ran through the round 13 tree
  (VALIDATION round 13).

## Current separation

At the time of the requirement, `allo/compiler.py::compile` has:

- the matcher/legality/placement path for Samsung, AiM, UPMEM and AttAcc;
- an APU-v1 vector route before that path, using
  `apu_v1_vectorize` contraction analysis and `APUV1Plan` recipe search;
- a Gemini-II source route using the shared matcher but its own source
  contract, launch planner and `APUG2ComposedContractionProgram` records;
- typed APU records dispatched to separate compilation functions.

Shared costs, a common call signature, or a shared result type are useful,
but do not make these one compilation pipeline. Hardware-specific lowerers
and runtimes are expected. Separate source-semantics analyses and end-to-end
planning bypasses are the migration target.

## Required boundaries

Design a common semantic program and candidate-plan contract, with one shared
driver from normalized frontend input to selected executable artifact. Keep
casts, guards, initialization, dependencies, access relations, exact numeric
contracts and reduction permissions explicit. Describe layouts, allocations,
transfers, residency, launch order, ABI and costing at the plan boundary.
APU vector recipes and instruction decomposition become backend hooks whose
effects are visible and checked through those contracts.

Reuse the existing frontend, matcher, target legality, layout algebra/search,
schedule search, cost evaluator and artifact machinery wherever sound. Avoid
an opaque registry of independent compilers or a parallel replacement stack.
Do not force target-dependent instructions or runtimes to be identical.

## Implementation order

**User priority update, 2026-11-02:** implement this shared-pipeline migration
before further standalone arithmetic/precision repairs or unrelated release
polishing. This supersedes the earlier semantic-repair-first sequencing.
Deferred defects remain recorded release blockers. Both actor and reviewer
must prioritize real G9 implementation slices; only a demonstrated dependency
of the next migration slice warrants an immediate prerequisite repair.

1. Preserve the independent reviewer's findings, existing repairs and executed
   reproducers. Checkpoint unfinished lower-priority work without extending
   it into another repair/whole-suite/wheel-validation cycle.
2. Inventory stage contracts and duplicated analyses, write the concrete IR
   and plan design, migration slices, and compatibility constraints here.
3. Introduce the shared boundaries through a working vertical slice of an
   existing matcher backend; preserve real code generation and execution.
4. Migrate APU-v1 and Gemini-II source paths, then typed inputs, into those
   boundaries; remove superseded public bypasses and duplicated analyses.
5. Prove common stage use, generality and selected-program fidelity with
   source walkthroughs, independent program variants, cost/plan perturbations,
   simulator execution and both APU hardware hosts. Validate installation
   outside the source tree and refresh the release claim/capability matrix.

Retain semantic facts and known restrictions in the shared contracts while
delivering bounded migration increments to the independent RLAR reviewer.
Do not wait for the unrelated arithmetic backlog to be closed before starting.
Follow `zzzDavid/allo:tenon`, normal fast-forward pushes only, and the user's
absolute prohibition on all PRs and upstream-main merges.

## Stage inventory from source (round 8, `allo/compiler.py` at the round 8 tree)

`compile()` (`allo/compiler.py:759`) dispatches in this order:

| Order | Condition | Route | Semantic analysis | Candidate plans | Legality | Cost / selection | Realization and runtime |
|---|---|---|---|---|---|---|---|
| 1 | `isinstance(workload, route.program_type)` for `_PROGRAM_ROUTES` | typed records `APUv1Program`, `APUG2TypedProgram`, `APUG2ComposedContractionProgram` via `_compile_program` | none on source: the record is the semantics | none (one fixed plan) or a per-record enumeration | per record | per record | `compile_apu_v1_program`, `compile_apu_g2_typed_program`, `compile_apu_g2_composed_program` |
| 2 | `target.name == "apu_v2"` | Gemini-II source route `compile_apu_g2_source_workload` (`allo/pim/apu_g2_source_route.py:1464`) | `customize` + `match_workload` + `require_match_coverage` (shared), then `analyze_apu_g2_source` builds `APUG2SourceContract` (its own operand/width/signedness facts over the matches) | `enumerate_apu_g2_launch_plans` (`single`/`fused`/`separate` x `k1`/`k2` x output tiles) | `_check_widths` and the contract's own domain rules | `price_apu_g2_launch_plan` -> `ExecutionGraph`, `rank_apu_g2_launch_plans` | `APUG2SourceProgram` -> GTML project template -> card; oracle `apu_g2_source_oracle` inside the runtime |
| 3 | `target.name == "apu_v1"` and no retained matcher scopes | APU v1 vector route `compile_apu_v1_vector_workload` (`allo/pim/apu_v1_vector_program.py:979`) | `analyze_apu_v1_contractions` (`allo/pim/apu_v1_vectorize.py:757`): a **text parser** of the printed MLIR (`_function_regions`, `_parse_function`) rediscovering products, accumulators and initializers; does **not** run `match_workload`, so none of the round 8 cast/copy/alias contracts apply | `generate_apu_v1_vectorization_candidates` (recipe catalog incl. `gemv_reduction_tile_domain`) | inside the recipes | `search_apu_v1_vector_plans`, `_apply_reduction_policy`, `_select_plan` | `_realize` -> GVML project -> board (`APUv1VectorCallable`) |
| 4 | otherwise | matcher route | `customize` -> `match_workload` -> `require_match_coverage` -> `require_matcher_route_policy` -> `_stamp_matcher_work_scopes` | `autoschedule` enumerators per target (`_samsung_enumerate`, `_aim_enumerate`, `_upmem_enumerate`, `_apu_v1_enumerate`, `_attacc_enumerate` in `allo/spmw_autoschedule.py`) | `allo/spmw_legality.py` in `autoschedule()` | `rank_matcher_placements` (movement stage `allo/pim/layout_search.py`, cost evaluator, lazy materialization, bounded joint evaluation, ties) | `compile_for_target` (`allo/spmw_codegen.py:1525`) -> backend ctx (`SamsungCtx`, UPMEM, AiM, AttAcc, the matcher-side APU v1 ctx) -> runner; `CompiledCallable` |

Observations that fix the migration order:

- Only route 4 and (for its semantic facts) route 2 run the one maintained
  semantic analysis. Route 3 reads printed MLIR text; a program accepted by
  route 3 never sees store coverage, exact-cast, copy-contract, alias or
  initialization checks. This is the first thing to remove: route 3 must
  consume `MatchTrace` (plus a shared contraction-facts object derived from
  the matches) instead of re-parsing text.
- Routes 2 and 3 both derive "contraction facts" (operand widths,
  signedness, reduction trip, output initializer, epilogue) from different
  representations: `APUG2SourceContract` from matches, `ContractionAnalysis`
  from text. A single `ContractionFacts` derived from `MatchedOp` accesses
  and `axis_roles` (already structural, M-01) is the shared semantic
  program contract for both.
- Route 2's launch plans and route 3's recipes are candidate plans with
  the same observable effects as a matcher `Placement` (coverage, layouts,
  allocation, transfers, launch order, ABI, cost graph) but no common type.
  The matcher side already has `Placement`, `KernelRanking` (ties,
  rejections, movement record), `PlacementRanking` and `ExecutionGraph`;
  the APU plans must become enumerator outputs of the same shape so that
  `rank_matcher_placements` owns search, budgets, ties and selection for
  them too. Their payloads (GVML recipe, VL64 launch program) stay
  backend-specific.
- Route 1's typed records can remain frontends if they normalize into the
  same contraction-facts and plan contracts and then enter the same
  legality, selection and materialization stages.

### Migration slices

1. **Shared contraction facts** (done, round 12: `allo/spmw_semantics.py`,
   `ProgramFacts`/`Contraction`/`Epilogue`). `analyze_apu_g2_source`
   consumes it; its tests and the card run are the contract.
2. **APU v1 through the matcher** (done, round 12): the text parser is out
   of the public route; the recipe catalog is kept; the matcher's checks run
   in front of it. The `NoContractionError` fall-through is now the
   planner's `applies` predicate (`_apu_v1_vector_applies`); it goes away
   when the recipes are enumerator candidates.
3. **Plans as enumerator outputs** (done, round 13): the APU v1 recipes and
   the Gemini-II launch plans are `autoschedule` enumerators producing
   `Placement`s with target-specific payloads and explicit effects; priced
   through the graph path (`candidate_execution_graph`); selected in
   `rank_matcher_placements`; materialized through `compile_for_target`
   backend contexts. The `apu_v2` and `apu_v1` branches of `compile()` and
   the planner hooks are gone.
4. **Typed records.** `APUG2TypedProgram` and
   `APUG2ComposedContractionProgram` normalize into the slice 1/3 contracts
   (done, round 15); the hybrid route's MLIR-text parser is removed (round
   14); `APUv1Program` remains to be normalized.
5. **Evidence.** Stage traces for one representative program per target,
   forced alternatives and cost perturbations reaching generated code and
   launches, simulator and both hardware hosts, installed artifact. At
   `9255d29` (round 15) the installed wheel ran the shared pipeline on the
   three simulators, the APU v1 board and the Gemini-II card, and the full
   suites passed (VALIDATION.md round 15); the forced-alternative and
   cost-perturbation traces on hardware remain.
