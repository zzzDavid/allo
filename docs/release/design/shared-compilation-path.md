# Shared compilation path: required migration

Status: **required, not implemented**. User scope addition on 2026-11-02.
This is gate G9 of the containing repository's `TASK.md`; release cannot be
declared ready without it. This document seeds the work, not an implementation
or a completion receipt. Actor and independent reviewer must refine it from
current source before migration.

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

1. Preserve and complete the independent reviewer's conversion and guarded
   output-store repairs. Keep their executed reproducers as regressions.
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

This outline is not permission to delay semantic fixes until after a broad
refactor. Deliver bounded increments to the independent RLAR reviewer.
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

1. **Shared contraction facts.** Add a `ContractionFacts` builder over
   `MatchTrace` (widths, signedness from the frontend's extensions,
   reduction trips, initializer kind, epilogue, aliasing) and make
   `analyze_apu_g2_source` consume it (behavior-preserving; its tests and
   the card run are the contract).
2. **APU v1 through the matcher.** Replace `analyze_apu_v1_contractions`'s
   text parser with the same facts; keep the recipe catalog; add the
   matcher's checks in front of it. Remove the `NoContractionError`
   fall-through once the recipes are enumerator candidates.
3. **Plans as enumerator outputs.** Register the APU v1 recipes and the
   Gemini-II launch plans as `autoschedule` enumerators producing
   `Placement`s with target-specific payloads and explicit effects; price
   them through the existing graph path; select in
   `rank_matcher_placements`; materialize through `compile_for_target`
   backend contexts.
4. **Remove the bypasses.** Delete the `apu_v2` and `apu_v1` branches in
   `compile()`; typed records normalize into slice 1/3 contracts.
5. **Evidence.** Stage traces for one representative program per target,
   forced alternatives and cost perturbations reaching generated code and
   launches, simulator and both hardware hosts, installed artifact.
