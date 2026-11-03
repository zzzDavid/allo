<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Tenon architecture in Allo

This document maps the Tenon (SPMW) implementation as it is in the working
tree of branch `tenon` (on top of `6efaca5`, 2026-11-02). It names the module
and symbol that implements each stage of the paper's compilation flow
(`paper/latex/sections/3-language-and-compiler.tex`, Sections III-E and
III-F) and states the invariant that holds at each stage boundary. Release
status, claim-by-claim correspondence and open findings are kept in
`docs/release/STATUS.md`, `docs/release/CLAIMS.md`, `docs/release/ISSUES.md`
and `docs/release/REVIEW.md`; finding identifiers below (C-xx, S-xx, U-xx,
M-xx, V1-xx, G2-xx, P-xx, F-xx) refer to `ISSUES.md`. The historical design
notes under `design/` are rationale, not a description of the current tree;
`design/README.md` lists which of their statements no longer hold.

Paper notation uses `@tn.*`; the Python surface is `allo.*`.

Contents

1. The three languages and their public symbols
2. The compile pipeline, stage by stage
3. Backend routes
4. Extending Tenon (primitives, knobs, backends)
5. Inspecting a chosen program
6. External dependencies and discovery
7. Module map
8. Open items and compatibility boundary

## 1. The three languages and their public symbols

### 1.1 Target language (`allo/spmw_target.py`)

A target is a tree of units with attached storage, moves and operations. It
carries structure and legality only; no cycle figure is stored on a target.

| Paper | Python | Implementation |
|---|---|---|
| `@tn.target` | `allo.target(name)` | `spmw_target.target`; returns a `Target` (root `Unit`, flat handle index, `Target.op(name)`, `Target.move(name)`, `Target.unit(name)`, `Target.work_grid()`) |
| `@tn.unit(mapping)` | `allo.unit(mapping=None, *, mode=None, capacity=1)` | `spmw_target.unit`; `Unit.mapping`, `Unit.axes`; `mode="host"` marks the host scope |
| `@tn.device` | `allo.device` | `spmw_target.device` returning a `DeviceScope` whose attributes name device storage from a sibling host scope (`hbm_pim.banks`) |
| `tn.mem`, `tn.reg` | `allo.mem(**geometry)` (= `spmw_target.memory`), `allo.reg(lanes, width, name=None, slots=None, ports=1, axes=None)` | `Memory`, `MemoryRef` (a `Memory` indexed by a `SymExpr`/`UnitId`), `Register` |
| `tn.get_uid` | `allo.get_uid()` | symbolic unit coordinates of the enclosing scope |
| `tn.op(fn, emit, src, dst)` | `allo.op(name, src, dst, fn, accumulates=False, emit=None, capacity=1, matchable=True, vector_width=1, const_params=(), guarded=False, dst_index="affine", iv_params=())` | `Op`; `fn` is a Python lambda compiled to a pattern by `spmw_match_engine.compile_op_pattern` |
| `tn.move` | `allo.move(name, src, dst, emit=None, verb=move_only, capacity=1)` | `Move`; verbs `allo.broadcast`, `allo.scatter`, `allo.gather`, `allo.move_only` tag host collectives |
| `tn.any`, `tn.or` | `allo.any_(candidates)`, `allo.or_(*alternatives)` | `AnyOf`, `OrOf`; admission is `spmw_target.pattern_admits` |

Host programs: `allo.host_xfer` (a `RecordingBackend` whose verb calls
record `HostMoveRecord`s), `allo.record_host_moves`, `allo.host_program`,
`allo.launch`, `allo.HostProgram`, `allo.BufferToken`, `allo.LaunchRecord`.
`allo.compile(..., host_moves=...)` accepts a list of records or a
`HostProgram`; nothing is inferred implicitly. `allo/spmw_host_program.py`
(`analyze`, `HostSchedule`, `schedule_residency`, `schedule_cost`) is a
residency analysis of such programs that has no production caller (C-10).

Reference targets, `allo/pim/targets.py`: `build_samsung_target`,
`build_aim_target`, `build_upmem_target`, `build_apu_v1_target`,
`build_apu_v2_target` (alias `build_apu_g2_target`; `target.name` stays
`"apu_v2"`), and `build_apu_g2_contraction_patterns` (a matcher-only view of
the two Gemini-II source idioms, never executed).

### 1.2 Workload language

| Paper | Python | Implementation |
|---|---|---|
| `@tn.work(mapping)` | `allo.work` | alias of `allo.dataflow.kernel` (`allo/__init__.py`: `from .dataflow import kernel as work`) inside an `allo.dataflow.region` |
| `tn.get_wid()` | `allo.get_wid` | alias of `allo.dataflow.get_pid` |
| typed ops, loops, accesses | ordinary Allo frontend | `allo.customize` lowers to retained MLIR (`func`, `affine`/`scf`, `memref`, `arith`, `math`) |

The frontend stamps structural scope attributes on each work-item function
(`allo/ir/builder.py`: `spmw.group_id`, `spmw.work_id`, `spmw.group_shape`,
`spmw.coalesced_axes`, `spmw.abi_value_ids`, `spmw.group_fingerprint`,
`spmw.body_fingerprint`); `allo/compiler.py::_stamp_matcher_work_scopes`
reads them back into `spmw_autoschedule.MatcherWorkScope` records on every
`MatchedOp`.

Two other workload kinds reach `allo.compile`:

* a plain `@allo` function (no dataflow region): the APU v1 vector
  contraction route (`allo/pim/apu_v1_vector_program.py`) and the Gemini-II
  source route (`allo/pim/apu_g2_source_route.py`) accept it; on the
  matcher backends it is matched as a single work item;
* typed program records: `allo.APUG2TypedProgram`
  (`allo/pim/apu_g2_typed_program.py`) and
  `allo.APUG2ComposedContractionProgram`
  (`allo/pim/apu_g2_composed_contraction.py`) are frontends of the shared
  pipeline (`allo/pim/apu_g2_record_frontend.py`, Section 2.0);
  `allo.APUv1Program` / `allo.APUv1Phase` (`allo/pim/apu_v1_program.py`)
  is a frontend too (`allo/pim/apu_v1_record_frontend.py`, rounds 16-17):
  its phase's source goes through the shared front stages (customize,
  matcher, coverage with the verbatim exemption, semantic facts that
  retain every store's loops, guards, reads, conversions and carried
  state), and its whole-function lowering (scalar ARC C or one proven
  native GVML recipe) is the record's candidate of the common pipeline,
  whose contract is derived from those facts.

### 1.3 Cost language (`allo/perf/`)

| Paper | Python | Implementation |
|---|---|---|
| `@tn.cost` | `allo.cost(function=None, *, target=None, name=None, fingerprint_data=None, materialization_scorer=None)` | `allo/perf/cost.py::cost` returns a `CostSpec`; `CostSpec.bind(target)` runs the builder once and yields a `BoundCostSpec` (rules keyed by handle path, `fingerprint`) |
| `@tn.rule(handle)` | `allo.rule(handle)` | `allo/perf/cost.py::rule`, legal only while a builder is binding; one rule per op or move handle, `function(event, ctx)` |
| `ctx.step(latency, occupy=[ctx.use(...)])` | `CostContext.step(cycles=/latency=, occupy=(), name=)`, `CostContext.use(handle, cycles=None, amount=1)`, `CostContext.repeat(count)`, `CostContext.parallel()` | `allo/perf/cost.py::CostContext`; a `CostEvent` exposes `metrics` (shapes, extents, `candidate` knob values, `lanes`, `host_slice`, ...) and `attributes` |
| evaluator | `BoundCostSpec.evaluate(graph) -> Estimate` | `allo/perf/evaluator.py::Evaluator`: ASAP over `depends_on` plus per-handle-instance calendars with capacity; `Estimate(cycles, spans, critical_path, utilization, bottlenecks, model_fingerprint)` |
| execution graph | `allo/perf/graph.py::ExecutionGraph`, `Activity`, `Occupancy`, `HandleInstance` | |

Reference cost programs (`allo/pim/costs/`): `samsung_cost`, `aim_cost`,
`upmem_cost`, `apu_v1_cost` (each `@cost(target=...)`), and
`apu_g2_cost = build_apu_g2_cost()` (`allo/pim/costs/apu_g2.py`, rules built from an
`APUG2TypedTickModel`). Constants are module-level Python with provenance in
the module docstrings; calibrating a backend means editing that file.

### 1.4 Composition

```python
compiled = allo.compile(workload, target, cost, *, backend=None, host_moves=None,
                        layout=None, reduction_order=None)
result = compiled(*arrays)        # RunResult
estimate = compiled.estimate()    # perf.Estimate from the retained graph
```

`allo/compiler.py::compile` returns a `CompiledWorkload`
(`CompiledCallable` for matcher and Gemini-II source routes,
`APUv1VectorCallable`, `APUv1ProgramCallable`, `APUG2TypedCallable`,
`APUG2ComposedContractionCallable` otherwise). `reduction_order` is
`"strict"` (default) or `"reassociate"` (`allo/spmw_reduction_policy.py`);
typed program records reject it. `backend="virtual"` evaluates the cost
program only.

## 2. The compile pipeline, stage by stage

The paper's flow (III-E) is: lower the workload to target-agnostic IR; derive
device model, operation definitions and rewrite patterns from the target;
match; autoschedule (decision domains, legality and pruning, layout
construction, materialization, ranking of materialized candidates with
deterministic tie-breaking); lower the selected program and link the runtime;
virtual execution follows the same flow and returns only the cost. The
implementation follows this order; candidates are generated in cost order
and only a generated program can win, and complete assignments are
evaluated jointly within a documented budget (Section 2.6 and
`docs/release/design/selection.md`).

### 2.0 Route dispatch (`allo/compiler.py::compile`)

1. `_materialize_workload` (calls `build()` when present),
   `_materialize_target` (calls a zero-argument builder), `_resolve_cost`
   (binds a `CostSpec` to the target).
2. Program records are frontends of the shared pipeline; there is no typed
   dispatch table. `APUG2TypedProgram` and `APUG2ComposedContractionProgram`:
   `allo.pim.apu_g2_record_frontend.normalize_apu_g2_record` states the
   record's semantics as the shared program contract (synthetic matches with
   the matcher's structural facts and the derived `ProgramFacts`), and the
   record's realization is one candidate of step 4 (the view declares the
   typed operations; the record's recipe graph prices it;
   `APUG2RecordProgram` is the materialized program). `APUv1Program`:
   `allo.pim.apu_v1_record_frontend.normalize_apu_v1_program` runs the
   shared front stages on the phase's source (one `customize`,
   `match_workload`, `require_match_coverage` with the function marked
   verbatim so its unmatched stores are not dropped effects, and
   `analyze_program(verbatim=True)`, whose `ProgramFacts.effects` retain
   every store's loop nest, guards, reads through local temporaries,
   data-dependent indices, conversions and carried state, with returned
   buffers as parameters); one anchor match of `ARC_SCALAR_PROGRAM` /
   `NATIVE_VECTOR_PROGRAM` is the scheduled work item. The `apu_v1`
   enumerator builds the lowering from the same retained schedule
   (`CompiledAPUv1Program`) and declares the record's candidate from the
   facts (`allo.spmw_candidate.verbatim_values`: an argument is an input
   when the function reads it, reads the element it writes, or does not
   overwrite it whole and unconditionally; one whole-function launch
   advancing every store effect; L4 value layouts; the transfers and
   scratch allocation its runtime performs). The shared validator checks
   the declaration against the facts, and the realization check compares
   the declared transfers and allocations with the runtime's actual
   movement. Records fix their own reduction grouping and refuse
   `reduction_order`; `host_moves` and `layout` follow the common rules.
3. Shared front stages for every ordinary source workload
   (`allo/spmw_pipeline.py::normalize_source`, gate G9):
   `customize(workload, enable_tensor=False)` produces the MLIR module;
   `match_workload(matcher_view(target), module)` matches it against the
   target's patterns, or against the source-idiom view a backend declares
   as `target.spmw_match_view` (Gemini-II: `build_apu_g2_contraction_patterns()`,
   because its hardware description has no matchable primitive);
   `require_match_coverage(..., defer_unknown=True)` enforces the coverage,
   accumulator-initialization, conversion, alias, copy and precision
   contracts (`UnsupportedWorkloadError`) before any planner; the shared
   semantic facts (`allo/spmw_semantics.py::analyze_program` ->
   `ProgramFacts`: parameters with element types and signedness, each
   contraction's loop nest, roles, accesses, accumulator initialization,
   output sinks, guards, operations, policy; epilogues) are derived from
   that trace on first use (`SourceProgram.program_facts()`).
4. Common candidate pipeline: `require_matcher_route_policy`,
   `_stamp_matcher_work_scopes`, then `spmw_codegen.compile_for_target`:
   the target's enumerator (Samsung, AiM, UPMEM, AttAcc placements; APU v1
   vector plans of a plain-function contraction or the grouped program of
   a dataflow region; Gemini-II launch plans), legality against the
   matcher view, the movement stage, cost through
   `spmw_plan.build_execution_graph` (a backend plan through its context's
   `candidate_execution_graph`), `rank_matcher_placements`, materialization
   by the backend context and the runner.  ``layout=`` names a candidate,
   a candidate family or a backend plan object and restricts the selection
   (`spmw_autoschedule.select_candidates`); every candidate stays scored.
   A caller-supplied placement is rebound to the current program's
   candidate under the current reduction policy
   (`spmw_candidate.rebind_supplied_layouts`); a backend plan's effects are
   the typed `CandidateEffects`, validated per candidate, on the selected
   placements, against the priced graph and against the realized program;
   a host program is checked for launch arity and operand binding
   (`spmw_plan.launch_schedule`) and for what the backend can honor
   (`validate_orchestration`).  The realized backend program of the
   selected candidate (the context's ``program``) is reachable as
   ``CompiledCallable.backend_program``; its documented ``INTROSPECTION``
   names are forwarded (``compiled.selected_plan``, ``compiled.plan``).

### 2.1 Frontend (`allo/customize.py`, `allo/ir/`)

`customize` lowers the Python function to retained MLIR. Two shared-frontend
facts matter to Tenon:

* Name resolution for annotations (`allo/ir/utils.py::_get_global_vars`):
  the function's module globals, then locals of call-stack frames whose code
  lexically encloses the definition (nearest first, `functools.wraps`
  unwrapped), then closure cells. Frames that do not enclose the definition
  contribute nothing, so an unrelated `M = 256` in a calling script cannot
  change a shape (F-01, fixed; `tests/test_global_vars_scope.py`).
* Integer literals are typed `i32` and silently truncated (`-(1 << 40)`
  becomes `arith.constant 0 : i32`) (F-02, open).

Invariant at the boundary: the module contains one `func.func` per work item,
each carrying the `spmw.*` scope attributes, with loop bounds retained as
affine text (`spmw_tripcount._parse_loop_bound` reads them).

### 2.2 Matcher (`allo/spmw_match_engine.py`, `allo/spmw_match.py`)

`match_workload(target, module, *, reduction_order=None) -> MatchTrace`.
`compile_target_patterns(target)` compiles every matchable `Op.fn` lambda
into an `OpPattern` (`PVar`, `PConst`, `PBinOp`, `PUnary`, `PCmp`,
`PSelect`); `_try_match_at_store` unifies the term stored at each
`memref.store`/`affine.store` site against the patterns (commutative
operands, constant folding, `const_params`, `guarded`, `dst_index`,
`iv_params`). Each hit is a `MatchedOp(target_op_name, func_name, work_id,
enclosing_loops, operands: list[OperandBinding], result_memref_name,
op_range, extra, result_value_ref, vector_width)`; operands carry
`memref_name`, `indices`, `index_map`, `memref_type`, `is_loop_carried` and
an exact `IRValueRef`.

Facts stamped on `MatchedOp.extra`:

| Key | Meaning | Producer |
|---|---|---|
| `accumulator_init` | `"zero"` (a dominating zero store or whole-buffer fill) or `"carried"` (caller buffer, no initializer); any other initializer raises | `_FunctionCoverage` (C-14) |
| `axis_roles` | `{"reduction", "output", "batch", "trips", "matrix", "vector", "unresolved", "work_axis", ...}` derived from index sets including the store index; consumers must not re-derive roles from extents | `_axis_roles`, `_stamp_batch_dims` (M-01) |
| `batch_dim` | batch extent | `_stamp_batch_dims` |
| `reduction_policy` | `ReductionPolicy(order, accumulator_type)` on every accumulating match | `spmw_reduction_policy.stamp_reduction_policy` via `match_workload` |
| `constants` | literal `const_params` values | `_try_match_at_store` |
| `output_sinks`, `store_indices` | where a scalar accumulator is stored, and the store's loop variables | coverage analysis |

Store coverage: every store of every function is classified
(`spmw_match.StoreCoverage.category` in `matched`, `copy`,
`accumulator_init`, `local`, `unmatched`; `MatchCoverage.problems`).
`require_match_coverage(trace, *, target=None, module=None,
defer_unknown=False)` raises `UnsupportedWorkloadError` for any store the
target does not implement, except in functions a backend executes verbatim
on the host (`_host_executed_functions`, today the UPMEM `kmeans_update`
group); a trace without coverage is re-matched from `module` when given
(C-02).

Invariant: a trace that reaches codegen implements every store of the
source; a device may zero an accumulator only when the source does.

### 2.3 Reduction-order policy (`allo/spmw_reduction_policy.py`)

`reduction_order="strict"` (default) forbids forming and regrouping partial
sums of a floating-point accumulator; integer accumulators (`i*`, `ui*`,
`si*`, `index`) may always be split (wrapping addition). `reduction_policy
(match)`, `may_regroup(matches)`, `ReductionOrderError`.
`require_matcher_route_policy(target, trace)` fails before scheduling on
`aim` and the `apu_v1` matcher route under strict with a float accumulator
(`_REGROUPING_MATCHER_ROUTES`); Samsung and UPMEM are policy-aware
enumerators (`_POLICY_AWARE_MATCHER_ROUTES`). Design:
`docs/release/design/reduction-policy.md`. Candidates pruned by the policy are
recorded on survivors as `extra["reduction_policy_pruned"]` and surface as
`KernelRanking.policy_pruned` and `CompiledCallable.reduction_policy_pruned`.

### 2.4 Legality (`allo/spmw_legality.py`)

`check_target(target)`, `check_placement(target, matches, placement)` and
`check_layouts(target, trace, layouts)` return a `LegalityVerdict`
(`violations`, `raise_if_illegal(context)` raising `IllegalPlacement`).
Rules, applied per matched op and its selected target operation
(`selected_operation`, honouring `placement.extra["operation_name"]`):
`declaration` (op/move handles declared on the owner or an ancestor; `src`
arity fits `fn`), `access-scope` (every placed operand handle is declared on
`op.owner` or an ancestor), `operand-pattern` (each operand admitted by its
`src`/`dst` pattern up to the symmetries of `fn`), `move` (a register-placed
source needs a declared fill, a register-placed result a declared drain, and
`spmw_plan._moves_for_handles` must be unambiguous), `register-aliasing`
(an accumulator may not share a register with a distinct source),
`register-capacity` (distinct values per register bounded by
`Register.slots`). The pass filters enumerated candidates in
`autoschedule()` (rejections recorded) and rejects a user `layout=` in
`compile_for_target` (C-01). Placements that name no handles (UPMEM)
satisfy the rules vacuously; their legality is the template's (Section 3.3).
Replacing the cost program cannot legalize a placement (paper III-C).

### 2.5 Movement stage (`allo/spmw_layout_problem.py`, `allo/pim/layout_*.py`, `allo/pim/gf2.py`)

The paper's Section IV machinery lives in the package:
`layout_problem.py` (`OwnerHierarchy`, `owner_hierarchy_from_target`),
`layout_movement.py` (`LayoutAccess`, `full_carrier_rank_incidence`,
`exact_active_owner_incidence`, `layout_movement_cost`,
`strict_reduction_kernel`), `layout_search.py`
(`search_min_movement_layout` dispatching to the bounded Algorithm 1, its
masked variant and the scalable flag search; `LayoutSearchPolicy`,
`LayoutSearchResult`; plus `search_coordinate_min_movement`, an exact
coordinate-variable search with a stated dominance lemma for problems beyond
the 4-bit domain). `spmw_layout_problem.derive_layout_problem` builds the
problem of one kernel group from the matched ops (work-grid axes and loops
as variables, affine injective coordinates, `READ`/`WRITE_ACCUMULATE`
accesses, `reassociation_allowed` from the policy stamp, owner levels from
`COMPUTE_LEVELS`), and `analyze_kernel_movement` runs the search and
classifies ties.

What is proven: inside the bounded domain, the complete tie set of minimum
movement (`optimality_proven=True`, `ties_complete`); for larger full-carrier
problems the coordinate search is exact over all one-to-one layouts; on a
mask it is exact over coordinate layouts only (`optimality_scope`). A
node-budget exit sets `incomplete`. Movement weights are uniform (1 per
byte) because no package cost program defines `movement_cost_per_byte`
(`movement_weights="uniform"`).

What is recorded: `KernelRanking.movement` is a `KernelMovement` (`status`,
`implementation`, `termination`, `optimality_proven`, `optimality_scope`,
`incomplete`, `min_cost`, `program_cost`, `program_label`, `chosen_cost`,
`chosen_label`, `tie_count`, `ties_crossed`, `visited`, `pruned`,
`movement_weights`, `assumptions`, `notes`, `ties: tuple[MovementTie]`).
`PlacementRanking.incomplete` aggregates node-budget exits.

Realizability limitation: every backend emits one program per work item on
the unit its work id names and none re-maps iterations to owners, so a tie is
realizable only when its owner flag equals the program's work-grid map.
Other ties are crossed with the candidates and rejected before emission with
the reason (`KernelRanking.movement_rejections`). CLAIMS row 16 is therefore
"partial". Policy: `MovementSelectionPolicy`, `movement_selection_policy()`
context manager (default node budget 20,672, `max_tie_candidates=64`).
Design: `docs/release/design/layout-search.md`; tests
`tests/perf/test_layout_*.py`, `tests/pim/test_layout_selection.py`.

### 2.6 Placement enumeration, knobs, ranking (`allo/spmw_autoschedule.py`, `allo/spmw_knobs.py`)

Enumeration. `autoschedule(target, trace, cost, *, host_moves=(),
buffer_metrics=None, movement_policy=None) -> list[Placement]` looks up the
backend enumerator registered with `@register_enumerator(name)`
(`_ENUMERATORS`: `_samsung_enumerate`, `_aim_enumerate`, `_upmem_enumerate`,
`_apu_v1_enumerate`). A `Placement(placements: memref -> handle, mode, extra,
layout: LinearLayout | None)` is the unit of decision; `mode` is a label only
and `extra` carries allowlisted structural fields
(`_MATCHER_PHYSICAL_EXTRA_FIELDS`). Each enumerator produces base candidates
and calls `cross_with_knobs(target, bases, matches, role_to_memref)`.

Knobs. `spmw_knobs.Knob(name, candidates_fn, emit_fn)`,
`register_knob(target_name, knob)`, `registered_knobs(target_name)`,
`register_default_knobs()` (run at import; registration order is the cross
order). Registered today: Samsung `grf_residency`, `crf_issue`,
`stage_resident`, `residency`, `tile`, `double_buffer`; APU v1 (matcher
route) `residency`, `tile`, `double_buffer`; AiM `bank_scope`,
`batch_mapping`, `reuse_group`, `reduction_partitions`,
`replica_partitions`; UPMEM the family decision fields
(`_UPMEM_KNOB_NAMES`, from `UPMEM_DECISION_DOMAINS` catalogs). A knob's
`emit` writes `extra[name]`; cost rules read it through
`event.metrics["candidate"]`. `SPMW_DISABLE_SCHEDULE_SEARCH=1` skips
`residency`/`tile`/`double_buffer` (baseline-versus-search comparisons).
`derive_layout_properties(target, placement)` adds facts derived from a
carried `LinearLayout` (AiM bank fanout, UPMEM tasklet features) so that
removing the F2 map changes both cost and emitted code.

Ranking. `rank_matcher_placements(target, trace, enumerator, bound_cost,
*, host_moves, buffer_metrics, liveness, movement_policy)`:

1. buckets the trace by work item (`_bucket_for_autoschedule`) and groups
   buckets by `MatcherWorkScope.group_id` (one `@allo.work` kernel);
2. the kernel choice set is the decision keys
   (`_structural_decision_key`) every bucket enumerates; the legality pass
   has already removed illegal candidates inside `autoschedule()`;
3. `analyze_kernel_movement` runs the movement stage and
   `_movement_entries` crosses every realizable tie class with every key
   (the program's own layout is kept, labelled `not-minimal`, when no tie
   contains it); the movement layout joins the backend layout as a product
   over `mm.`-prefixed dims;
4. every candidate is priced through the same executable cost program as
   virtual execution (`spmw_plan.build_execution_graph` +
   `bound_cost.evaluate`); a cost rule raising
   `pim.schedule_search.InfeasibleSchedule` is a recorded rejection;
5. candidates sort by `(cycles, structural key, position)`
   (`TIE_BREAK_RULE`: cycles, then the carried layout's F2 map, then knob
   and feature values, then operand handle paths; C-06) and the first whose
   codegen probe (`spmw_codegen._materialize_matcher_codegen` on the
   kernel's sub-trace) does not raise `InfeasibleSchedule` wins;
6. kernels whose buckets share no decision key fall back to a per-bucket
   stable argmin (`fallback="per_bucket"`);
7. `_reconcile_resident_pairs` honours a cross-kernel `resident` residency
   only when both endpoints chose it, without re-costing.

Records: `KernelRanking(group_id, bucket_count, fallback, scores,
chosen_position, rejections, movement, cost_ties, tie_break,
movement_rejections, policy_pruned)` and `PlacementRanking(kernels,
incomplete)` on `MatcherScheduledPlacements.placement_ranking`.

Selection contract (`docs/release/design/selection.md`, C-03 and C-05
closed): candidates are code-generated in cost order and the first that
generates wins; `InfeasibleSchedule`, `NotImplementedError` and `ValueError`
from an emitter are recorded as rejections (`PlacementRanking.materialization`
reports how many candidates were probed; `SelectionPolicy(materialize=
"eager")` generates every candidate first). After per-group choices the
reconciled whole program is re-priced (`program_cycles` equals
`estimate().cycles`) and any group a revert touched is re-ranked under the
whole-program cost. When the product of the groups' surviving candidate
counts is within `SelectionPolicy.joint_budget` (default 256), every
complete assignment is evaluated as one program through
`allo/pim/schedule_search.py::grid_search` over the whole-trace graph and
the record says `joint_evaluation="complete"`; above the budget the
per-group route runs and the record says `"per-group heuristic"`. The
paper's "each surviving complete assignment is evaluated as one program"
therefore holds within the budget, and the label states when it does not.

### 2.7 Materialization and code generation (`allo/spmw_codegen.py`)

`compile_for_target(target, trace, layout=None, backend=None, host_moves=None,
buffer_metrics=None, cost=None, module=None, launch_schedule=None) ->
Compiled`:

1. `_resolve_host_moves` matches recorded `host_xfer.*` calls to declared
   host moves by verb and device-handle identity; a `LaunchSchedule`
   (`spmw_plan.launch_schedule(host_program, trace)`) must have one host
   step per move;
2. `require_match_coverage` again (fail closed for traces that deferred it);
   AiM and the APU v1 matcher route refuse `accumulator_init == "carried"`
   with `NotImplementedError` (they zero the accumulator); Samsung prunes
   REDUCE candidates for it and the ordered program loads the value, but
   `compile_for_target` still refuses `carried` on Samsung before
   scheduling (open, `design/reduction-policy.md` section 7);
3. a user `layout=` is checked by `check_layouts`; otherwise
   `autoschedule()` picks one placement per kernel and the survivors are
   re-checked after reconciliation;
4. `_check_work_grid(target, trace)` (C-04) raises unless the kernel grid is
   one of the documented forms: full fanout, outer prefix, single item on
   `_SINGLE_ITEM_LAUNCH_BACKENDS` (`samsung_hbm_pim`, `upmem`, `aim`),
   outer-level subset on `_OUTER_SUBSET_LAUNCH_BACKENDS` (`upmem`, `aim`),
   or the APU v1 coalesced lane axis;
5. the backend `CodegenContext` subclass (`_BACKENDS`: `samsung_hbm_pim ->
   spmw_samsung.SamsungCtx/_run_samsung`, `aim -> spmw_aim.AimCtx/_run_aim`,
   `apu_v1 -> spmw_apu_v1.APUv1Ctx/_run_apu_v1`, `upmem ->
   spmw_upmem.UpmemCtx/_run_upmem`) lowers the program through
   `emit_program(trace, layouts, *, schedule, host_moves, module)`; the
   default walks buckets and calls each matched op's `emit` with placed
   handles (`_walk_buckets`, `_schedule_moves`, `_emit_move`); Samsung, AiM
   and UPMEM override it. `emit_program` must raise `InfeasibleSchedule` for
   a layout the backend cannot realize and `NotImplementedError` for an op it
   cannot lower;
6. `spmw_plan.build_execution_graph` lowers the selected placements to the
   retained `ExecutionGraph` when a cost is supplied;
7. the result is `Compiled(target, trace, cmds, layout, ctx=layout_ctx,
   backend, host_moves, execution_graph, cost, placement_ranking,
   launch_schedule)` with `Compiled.run(**inputs)` dispatching to the
   backend runner or `_run_virtual`.

The matcher materialization manifest (`MatcherCodegenArtifact`: commands,
host schedule, host preloads, resolved host moves, `runtime_abi`,
`runtime_segments`) is what the codegen probe freezes; the backend runners
consume `Compiled.cmds` and the context state (CLAIMS row 23).

### 2.8 Execution graph and estimate (`allo/spmw_plan.py`)

`build_execution_graph(target, trace, layout, cost_spec, *, host_moves=(),
buffer_metrics=None, launch_schedule=None)` emits one `CostEvent` per
matched op and per structural move (`_moves_for_handles` resolves the fill
and drain of each register-placed operand) per work item, plus host
transfer events; `_host_transfer_metrics` adds `host_slice` and
`host_resident` so a sliced or resident buffer is priced for what moves
(M-05). Metrics such as `lanes` come from the placement register (C-08).
`CompiledCallable.estimate()` evaluates the retained graph with the bound
cost and returns an `Estimate`. `estimate_candidate(target, trace, layout,
cost_spec)` prices one placement standalone.

### 2.9 Runners, result kinds and cost validity

`RunResult(cycles, stdout, backend, extra, kind)` (`spmw_codegen.py`).
`kind` is one of `device`, `simulator`, `simulator-timing-only`,
`functional`, `virtual` (`RUN_KINDS`); `validate()` rejects an unset kind and
a non-executing result carrying outputs; `CompiledCallable.run_backend`
validates every result, stamps `extra["reduction_order"]` and
`extra["reduction_policy"]`, and copies `extra["outputs"]` into the caller's
arrays only for `device`, `simulator` and `functional` kinds, raising when a
gathered output is missing (S-02, G2-07, V1-09).

`backend="virtual"` (`_run_virtual`) evaluates the retained graph and
returns `kind="virtual"` with `extra` keys `critical_path`, `utilization`,
`bottlenecks`, `model_fingerprint`, `cost_model`, `priced_target`, merged
with the backend context's `virtual_run_extra()` when it defines one
(UPMEM: `timing_scope`, `cost_validity` in `calibrated`/`extrapolated`,
`upmem_cost_validity` per group; Gemini-II: `estimate_validity`,
`latency_unit`). `simulator_unavailable_reason(target_name)` and
`SimulatorUnavailable(target, reason)` report missing external pieces
without running anything.

## 3. Backend routes

Each subsection gives the input kind, the supported domain, what executes
versus what is analytical, and open findings. Evidence status per route is
in `docs/release/CAPABILITIES.md` and `docs/release/VALIDATION.md`.

### 3.1 Samsung HBM-PIM (`allo/spmw_samsung.py`, `allo/spmw_autoschedule.py::_samsung_enumerate`, `allo/pim/costs/samsung.py`)

Input: source dataflow workloads matched to the target's `MAC`, `ADD`,
`MUL`, `MAD`, `RELU` ops (`build_samsung_target`). Runner: PIMSimulator
`pim_driver --op GENERIC` with the Tenon conductor
(`experiments/simulators/setup/PIMSimulator`, S-04/P-04 for its licensing
state); `kind="simulator"`, outputs read back (all-zero is a value, a short
`out.bin` raises).

* **Contractions, REDUCE program.** `samsung_contraction(match)` classifies
  operands structurally (matrix indexed by row and reduction, vector by
  reduction only; nests `[k]`, `[row, k]`, `[row, column, k]`; FP16; S-05).
  Work items must partition the output rows exactly (S-10). Base candidates
  `bank_row`, `dual_fiber`, `grf_staged` cross the knobs; the legality pass
  and `SamsungCtx._emit_reduce` reject what the conductor cannot run
  (`grf_staged`, `crf_issue=per_workid` with more than one item, more than
  32 CRF words, non-identity `tile`, `double_buffer`) with
  `InfeasibleSchedule`. The emitted per-pass CRF is uploaded verbatim; the
  conductor never calls the vendor command generator (S-01). Reduction
  element `k` accumulates in lane `k mod 16` of a GRF_B entry and the host
  sums the 16 lane partials in FP32, so REDUCE regroups and is pruned under
  `reduction_order="strict"`. `samsung_vendor_gemv_layout(target, trace)`
  selects the vendor-equivalent placement (dual_fiber, host-broadcast
  GRF_A, shared CRF) that reproduces the paper's 4,435 / 8,041 / 8,247
  cycles; the cost-selected default under `"reassociate"` is bank_row with
  host fill (4,389 / 7,947 / 8,153 cycles). The MAC rule serializes both
  fibers on the channel bus (`FIBER_TILE_CYCLES`); the fill-residency
  disagreements that remain are preload cost outside the graph (S-12).
* **Contractions, TEMPORAL (ordered) program.** The `temporal` base
  candidate (`extra["reduction_program"]="temporal"`, layout
  `identity({"lane": 16, "bank": 16})`, `grf_residency=host`,
  `stage_resident=False`) is the only survivor under strict for FP16.
  `SamsungCtx._emit_temporal` emits the five-word CRF (`MOV`, `MAC`,
  `JUMP K-1`, `NOP`, `EXIT`); `_run_samsung_temporal` calls `--op-kind
  TEMPORAL_REDUCE`; one GRF_B lane per output row; bit-equal to source-order
  FP16 (26,807 cycles at 4096x256); cost rule `temporal_mac`
  (`TEMPORAL_STEP_CYCLES=51`, `TEMPORAL_PASS_CYCLES=681`). `K <= 4096`.
* **Elementwise.** `samsung_elementwise(matches)`: one work item
  (`mapping=[1]`) walking a 1-D FP16 extent; `ADD`/`MUL`/`MAD`/`RELU`
  through `SamsungCtx._emit_eltwise`, one CRF pass per 131,072-element
  tile, bit-exact against NumPy FP16 (S-07). Partitioned grids are rejected
  with the reason.

Executed versus analytical: `RunResult.cycles` is PIMSimulator memory-clock
cycles of the padded program (preload + execution + readback, 64-channel
configuration); `extra` carries `logical_shape`, `padded_shape`,
`padding_fraction`, `timing_scope`, `cycle_phases`, `plan`, `crf`,
`conductor`, `simulator_geometry`, `levers`, `kernel` (S-06). The cost graph
prices only the modeled activities of the declared 16-channel device on the
unpadded shape; the two are compared by ranking only
(`docs/release/design/samsung-execution.md`). Open: S-11, S-13 (XOR swizzle
148 -> 116 reachable only through
`tests/pim/samsung_hbm_pim/layout_swizzle/` and
`allo/pim/samsung_layout_swizzle.py`), carried accumulators rejected before
scheduling. Multi-kernel programs run their host program's launch schedule,
one `pim_driver` invocation per launch (`_run_samsung_launches`; gathered
results are re-staged by the declared broadcast/scatter, a value left
device-resident across launches is rejected; per-launch cycles in
`extra["launches"]`; `samsung-execution.md` section 9).

### 3.2 SK hynix AiM (`allo/spmw_aim.py`, `allo/pim/aim_lowering.py`, `allo/pim/costs/aim.py`)

Input: source dataflow workloads (GEMV, attention-style batched MAC,
elementwise `MUL`/`ADD`, `AF`) with `mapping` a prefix of `[32, 4, 4]`.
`AimCtx.emit_program` turns matched kernels into typed `AimOp` records
(`AimContraction`, `AimElementwise`, `AimActivation`, `AimHostTransfer`,
`AimBankCopy`, `AimAllBankWrite`, `AimDistributedHostTransfer`,
`AimHostReduction`, `AimSync`) that `_AimLowerer` expands into the
ramulator2 text trace; `aim_lowering` is the only ISR encoder (M-11). Axis
roles come from the matcher (`contraction_shape`), and
`check_contraction_coverage` raises `AimCoverageError` when issued MACs
differ from the program's (M-01). Knobs: `bank_scope` (SBK/ABK from the
`LinearLayout` bank image), `batch_mapping`, `reuse_group`,
`reduction_partitions` (token partition: the reduction split cyclically over
P channel groups plus an explicit `AimHostReduction` host step, offered only
when the policy permits regrouping; M-03), `replica_partitions`. Under
`reduction_order="strict"` every BF16 program is refused before scheduling
(`ReductionOrderError`), since each MAC reduces 16 products through the bank
adder tree.

Executed versus analytical: the simulator (ramulator2 in Docker,
`TENON_AIM_SIM_IMAGE`) is timing-only; `kind="simulator-timing-only"`, no
outputs, caller arrays untouched; a nonzero exit raises (M-08). `cycles` is
the per-segment `memory_system_cycles` maximum over channels times the
compile-time repeat, summed; `extra` carries `returncode`, `segments`,
`unique_segments`, `host_steps` (host reductions are listed, never added).
Numerics of the token-partition merge are checked separately by
`aim_lowering.reference_partial_sums`. One cost route prices the exact
lowered record (`aim_contraction_profile`; M-06); host transfers are priced
by slice (M-05). `_check_simulator_geometry` rejects targets whose geometry
differs from the 32-channel preset (M-10). Design:
`docs/release/design/aim-token-partition.md`. Open: skill-doc bank geometry.

### 3.6 AttAcc (`attacc`, analytical only)

`build_attacc_target()` describes the paper's AttAcc device (5 HBM3 PIM
devices x 16 pseudo-channels x 64 banks, FP16; `MAC`, `SFM_EXP`, `SFM_SUM`,
`SFM_DIV`; no timing). `allo/pim/attacc_tiling.py` enumerates the legal
tilings from the geometry and the matched contraction, the `attacc_tiling`
knob makes each a candidate, and `allo/pim/costs/attacc.py` prices the
retained command inventory with the paper's closed-form rules
(re-implemented from the paper's source, no DCC code; validity domain in
the docstring). `allo/spmw_attacc.py::AttAccCtx` emits the command
inventory; there is no runner, so `backend=None` raises and only
`backend="virtual"` (`kind="virtual"`) is supported. All 500 paper
instances select the paper tiling through `allo.compile`
(`benchmarks/attacc/`, `tests/pim/test_attacc_virtual.py`). The
Ramulator2-AttAcc timing and the DCC decode composition are external
scripts with named dependencies (`benchmarks/attacc/scripts/`). Design:
`docs/release/design/attacc-backend.md`.

### 3.3 UPMEM (`allo/spmw_upmem.py`, `allo/pim/upmem_physical*.py`, `allo/pim/upmem_irregular.py`, `allo/pim/costs/upmem.py`)

Input: source dataflow workloads with `mapping=[N]` (one DPU per work item;
uneven partitions through `meta_if`/`meta_else`, U-04).
`classify_upmem_kernel(target, matches)` proves one of the registered
families (`register_upmem_family`: `pointwise`, `reduction`,
`matrix_vector`, `gemm`, `histogram`, `selection`, `kmeans_distances`,
`kmeans_update` (host-placed), `feature_gradient`), including operand roles,
element types (`UPMEM_ROLE_ELEMENT_TYPES`: int32 operands, int64
accumulators where the template says so; U-01), literal coefficients and
`accumulator_init` (carried outputs are loaded by MV, GEMM, reduction,
histogram and selection templates; rejected for k-means distances and
gradients; U-03). `_upmem_enumerate` fans out the family's finite decision
catalog (`UPMEM_DECISION_DOMAINS`) and keeps only decisions the template
realizes (`upmem_physical_search.physical_features`: DMA sizes and
alignment, WRAM footprint); non-dividing extents are padded and masked, not
rejected (U-05, U-06). `UpmemCtx.emit_program` instantiates the plan
(`UPMEMElementwisePlan`, `UPMEMSumReductionPlan`, `UPMEMMatrixVectorPlan`,
`UPMEMGEMMPlan`, `UPMEMHistogramPlan`, `UPMEMStableSelectionPlan`,
`UPMEMSelectionFlagsPlan`, `UPMEMKMeansDistancesPlan`,
`UPMEMFeatureGradientPlan`) whose `device_source()` is the DPU `task.c`;
`_check_tasklet_ownership` proves the emitted ownership factors through the
carried 12-of-16 tasklet `LinearLayout` (`UPMEM_TASKLET_CARRIER`). The
host-placed `kmeans_update` group runs its own portable C
(`allo/backend/c.py::emit_c_from_mlir`; wide integers rejected by
`WideIntegerError`, U-02) and is priced by `allo/pim/costs/upmem.py::host_group_cycles`
(coarse, `HOST_TIMING_SCOPE`).

Executed versus analytical: `_run_upmem` builds the DPU program with the
SDK Docker image, runs uPIMulator on a `GENERIC` fixture whose expected
bytes come from an independent portable-C oracle, and overwrites caller
arrays (`kind="simulator"`). `cycles` is the sum over segments of the max
per-DPU `logic_cycle` (`timing_scope="uPIMulator DPU logic_cycle only"`);
each record in `extra["upmem"]` carries `family`, `sdk_build`
(`pinned-campaign` or `user-build:<manifest12>`), `num_dpus`,
`num_tasklets`, `logic_cycles`, `geometry`, `work_item_geometry`,
`padding_fraction`, `host_steps`, source and simulator hashes. Virtual
runs label validity (`calibration_validity`: `calibrated` only at a
calibration anchor, else `extrapolated` with the nearest point; U-09).
Routes: pinned-campaign (archived binary + snapshot + pinned SDK manifest)
or user-build (`UPIMULATOR_ROOT` + `UPIMULATOR_BIN`; U-07/P-03). The Tenon
uPIMulator patch is not yet published (P-06). Design:
`docs/release/design/upmem-domain.md`. Pointwise `a*x - b*y` and
`a*x + b*y + c` (target ops `AXMBY`, `AXPBYC`) lower to the int32 AXPBY
template (U-06). Open: 12-tasklet policy, P-06, P-17.

### 3.4 GSI APU v1 / Gemini-I (`allo/pim/apu_v1_*.py`, `allo/spmw_apu_v1.py`, `allo/spmw_apu_v1_build.py`, `allo/pim/costs/apu_v1.py`)

Four routes share `build_apu_v1_target()` and the device. Every board
launch holds `apu_v1_program.apu_v1_board_lock()` (`flock` on
`TENON_APU_V1_LOCK`, default `/tmp/tenon-apu-v1-<pci>.lock`) and proves
profile freshness by diffing the device log before and after the launch
(`fresh_profile_records`, `parse_fresh_profile`; V1-07).

* **Vector contraction route** (`apu_v1_vector_program.py`,
  `apu_v1_vectorize.py`, `apu_v1_layout.py`, `apu_v1_vector_codegen.py`,
  `apu_v1_vector_cost.py`, `apu_v1_vector_runtime.py`). Input: the shared
  facts of a plain `@allo` function whose matched effect is one rank-2
  contraction (FP16/uint16 multiply-add; accumulator a function argument,
  carried or zero-filled, or a local scalar copied once to the output);
  1-D GEMV is lifted with a unit axis. `apu_v1_contraction_analysis`
  (`apu_v1_vectorize.py`) applies the vector plans' legality to the
  `ProgramFacts` (one unguarded `range(N)` nest, one reduction loop on both
  inputs, plain subscripts, distinct input buffers, one 16-bit element
  type, loop extents equal to buffer extents) and yields the
  `ContractionAnalysis`; the older MLIR-text parser remains only for the
  hybrid typed route. `generate_apu_v1_vectorization_candidates` builds a
  finite recipe catalog of `APUV1Plan`s (temporal SVP, DMA coalescing with
  accumulator depths, spatial group reduction, ...); `apu_v1_vector_candidates`
  (the ``autoschedule`` enumerator) prunes it under the reduction-order
  policy (`apu_v1_plan_regroups`), realizes every plan to GVML
  (`realize_apu_v1_plan`) and emits one ``Placement`` per plan with its
  effects (coverage, value layouts, transfers, launch order, ABI) and a
  name-free decision in the search key; `APUv1Ctx.candidate_execution_graph`
  prices each from the materialized inventory with `apu_v1_cost` and the
  shared `rank_matcher_placements` ranks them; `layout=<plan name>`, a
  family name or an ``APUV1Plan`` restricts the selection.
  `APUv1Ctx.emit_program` instantiates `APUv1VectorProgram` for the
  selected plan (forwarded by the callable: `plans`, `candidates`,
  `ranked_plans`, `candidate_estimates`, `selected_plan`, `realization`,
  `plan_selection`, `tile_decision`, `device_source()`).
  Backends: `None` builds the GVML project with the ARC toolchain and runs on
  the board (`run_apu_v1_vector`, `kind="device"`, manifest v2 uploads the
  packed initial output for temporal plans; V1-02); `"virtual"` evaluates
  the selected plan's retained cost graph only (`kind="virtual"`, no
  outputs, caller arrays untouched, `extra["cost_validity"]` as on the
  other routes; a nonzero initial output the plan resets is still
  rejected); `"functional"` is a host-side NumPy model of the selected
  plan's arithmetic order, not compiled device code (`kind="functional"`,
  outputs written, `cycles` the analytical estimate; a regrouping FP16
  plan runs `_grouped_fp16_contraction`, any other plan the source
  function, whose order it keeps). GEMV reduction tiles are cost-ranked
  candidates (`spatial_gemv_group_reduction_k<tile>`; `tile_decision` in the
  estimate and run record). The cost program carries the paper's fitted
  affine event model with provenance and labels estimates `calibrated` or
  `extrapolated` (`docs/release/design/apu-v1-cost.md`).
  Reduction order (V1-03): under `reduction_order="reassociate"` the default
  FP16 GEMV plan is `spatial_gemv_group_reduction_k<tile>`, a tree reduction
  (`gvml_add_subgrps`) whose rounding differs from source order; on the
  4096x256 GEMV 425 of 4096 outputs are bit-equal to the source-order FP16
  sum in the plan-order model (halving FP16 tree per group, groups added in
  ascending order). Under `"strict"` (the default) the tree plans are pruned
  and the temporal plan `temporal_dma_coalescing` is selected, bit-equal to
  source order (4096/4096) in the same model. The halving-tree numerics are
  a model of `gvml_add_subgrps` and are not board-checked; the board GEMV
  test checks a 5e-2 tolerance, not bit equality.
  The ARC project copies only the eight GSI files its `make` reads from the
  user's GSI SDK `example-gvml` project (`TENON_APU_V1_TEMPLATE_DIR`;
  `spmw_apu_v1_build._TEMPLATE_FILES`) and records their origin and SHA-256
  in the project's `THIRD_PARTY.txt`; Tenon does not redistribute them (V1-08).
* **`APUv1Program` scalar programs** (`apu_v1_program.py`,
  `apu_v1_record_frontend.py`): one `APUv1Phase(kernel, vectorize=False)`;
  the whole function lowers to portable C11 (`emit_c_from_mlir`) and runs
  on APUC 0 as the record's candidate of the shared pipeline. Backends
  `None` (device), `"virtual"` (the shared virtual runner over the retained
  cost graph), `"functional"` (host execution of the same C; V1-04).
  Multi-phase programs raise.
* **Native recipes** (`apu_v1_native_vector.py`, `apu_v1_histogram.py`,
  `apu_v1_distance_argmin.py`, `apu_v1_moments.py`,
  `apu_v1_record_frequency.py`, `apu_v1_native_cost.py`):
  `APUv1Phase(kernel, vectorize="required")` is proven by exactly one of
  `APU_V1_NATIVE_VECTOR_DISCOVERERS` with a strict MLIR parser plus symbolic
  executor; mismatches raise `APUv1NativeRecipeMismatch` and no match raises
  `NoNativeVectorRouteError` (V1-01). `"functional"` is rejected at compile
  time.
* **Legacy matcher route** (`spmw_apu_v1.APUv1Ctx`, `_run_apu_v1`,
  `_apu_v1_enumerate`): reached only by dataflow regions on `apu_v1`;
  grouped FP16 MAC over four APUCs with `gvml_add_subgrps`, refused under
  strict for float accumulators; output shapes come from the compiled ABI
  (V1-09).

`allo/pim/apu_v1_hybrid.py` (`discover_apu_v1_hybrid_manifest`) supplies the
ABI manifest of a single phase; the four-APUC hybrid runtime described in
`design/10-apu-v1-reference-backend.md` is not in the tree.

### 3.5 GSI Gemini-II / APU v2 (`allo/pim/apu_g2_*.py`, `allo/pim/costs/apu_g2.py`, `allo/pim/templates/apu_g2/`)

Three routes share `build_apu_g2_target()` (`target.name == "apu_v2"`).
Every runtime takes `apu_g2_runtime.apu_g2_device_lock()` (`flock` on
`TENON_APU_G2_LOCK`, default `/tmp/tenon-apu-g2-<device>.lock`) around
status check, build and launch, and `_require_hardware_stack` requires the
SDK files, `gsi_tool`, an `Available` card with core 0 idle, and device
index 0 (other indices rejected until `DeviceConfig` semantics are verified;
G2-06).

* **Typed records** (`APUG2TypedProgram`, `apu_g2_typed_program.py`,
  `apu_g2_typed_runtime.run_apu_g2_typed`: exact-width `add`, `mul`, `div`,
  `block_sum`, `dot`; and `APUG2ComposedContractionProgram`,
  `apu_g2_composed_program.py`, `apu_g2_composed_layout.py`,
  `apu_g2_composed_runtime.run_apu_g2_composed`: independent dot products
  `(dot_count, K)` with identity, pair-affine or accumulate epilogue on one
  `(4, 65536)` carrier, `APUG2DotTilePlan`, a `LinearLayout` over F2; larger
  problems rejected naming the capacity; packing attested against the plan
  before launch, G2-08).  Both are frontends of the shared pipeline
  (`apu_g2_record_frontend.py`): `normalize_apu_g2_record` states the
  record's semantics as synthetic matches with the matcher's structural
  facts and the derived `ProgramFacts`; the ``apu_v2`` enumerator emits the
  record's realization as one candidate with its `CandidateEffects`;
  legality (the view declares `SUM`/`ADD`/`MUL`/`DIV` for them), the
  candidate contract, the record's recipe graph, ranking,
  `APUG2RecordProgram` and the shared artifact follow; ``out`` may be
  omitted in a call and is then allocated and returned in
  ``RunResult.extra["outputs"]``.
* **Source route** (`apu_g2_source_route.py`,
  `compile_apu_g2_source_workload`): a plain integer contraction function
  passes the shared front stages, matched against the target's declared
  view `build_apu_g2_contraction_patterns()` (`MAC`, `AXPBY`, the latter
  declaring `intermediate="i24"` for its wide epilogue);
  `analyze_apu_g2_source(ProgramFacts)` (called by the ``autoschedule``
  enumerator `apu_g2_source_candidates`) derives the `APUG2SourceContract` (K, result
  coordinates, transposition, broadcast, epilogue, exact widths; rejects
  wrapping, non-power-of-two K, mixed signedness, K > 512),
  `enumerate_apu_g2_launch_plans` builds `single`/`fused`/`separate` x
  `k1`/`k2` x output-tile plans (G2-05), each a candidate ``Placement``
  priced by `APUG2SourceCtx.candidate_execution_graph`
  (`price_apu_g2_launch_plan`) and ranked by `rank_matcher_placements`
  (`allo/spmw_apu_g2.py`); the context materializes `APUG2SourceProgram`,
  which exposes `plan`, `ranking`, `rejected`, `selection`,
  `placement_ranking`, `manifest()`, `pack()`; `layout="<plan name>"`
  restricts the selection. Results are checked
  against an `einsum` oracle before `kind="device"` is returned (G2-02).
  The five paper kernels as source (`benchmarks/apu_g2/workloads.py`)
  reproduce the artifact's layout and program fingerprints.

Executed versus analytical: `cycles` is `round(device pipeline ticks /
repetitions)` (ARC running-state ticks of the timed batch plus one barrier;
host transfers reported separately in microseconds). The cost program
(`build_apu_g2_cost(APUG2TypedTickModel)`) prices every issued recipe call
from seven structural features fitted on ten card programs; outside its
validity domain the estimate is labelled `estimate_validity="extrapolated"`
(G2-03). `backend="virtual"` returns `kind="virtual"` with `cost_model`,
`latency_unit`, `estimate_validity`. Card-gated test:
`tests/pim/test_apu_g2_hardware.py` (G2-04). Design:
`docs/release/design/apu-g2-source-route.md`. Readback range-checks the
declared output type instead of wrapping (G2-09); padded `div` divisor lanes
are 1 (G2-10); the template holds the uint16 add, typed and composed programs
only, and the typed and composed runtimes build their one host target
(G2-11). Open: G2-12, device index semantics, `t + u` epilogue without
coefficients.

## 4. Extending Tenon

### 4.1 Adding a target primitive (op or move)

Worked example in the tree: Samsung `RELU` (`allo/pim/targets.py`,
`build_samsung_target`), priced by `elementwise` in
`allo/pim/costs/samsung.py`, lowered by `SamsungCtx._emit_eltwise`, and
recognized by `spmw_samsung.samsung_elementwise`.

1. Declare the primitive inside the unit that executes it:
   `allo.op(name, src=(pattern, ...), dst=pattern, fn=lambda ...,
   accumulates=..., emit=lambda ..., ctx: ctx.cmd(...))` or
   `allo.move(name, src, dst, emit=..., verb=...)`. `fn` is what the matcher
   unifies against (commutative operands, `const_params` for literal
   coefficients, `guarded`, `dst_index`, `iv_params`); `src`/`dst` patterns
   (`allo.any_`, `allo.or_`) are what the legality pass admits; `emit`
   receives placed handles and the backend `CodegenContext`. A declaration
   whose handles are out of scope fails `spmw_legality.check_target`
   (rule `declaration`); do not declare an operand source the backend
   cannot emit (the Samsung `srf` precedent, C-01).
2. Add a cost rule: inside the backend's `@cost` builder,
   `@rule(target.op(name))` (or `target.move(name)`) with
   `def r(event, ctx): ctx.step(latency=..., occupy=[ctx.use(...)])`.
   `BoundCostSpec.emit` raises `KeyError` for an event with no rule, so every
   reachable primitive needs one. Read shape facts from `event.metrics`, not
   from the target.
3. Legality implications: a new register destination needs declared fill
   and drain moves (rule `move`); an accumulating op's `dst` must not alias
   its sources (rule `register-aliasing`); `Register.slots` bounds distinct
   values (rule `register-capacity`). Run `tests/perf/test_legality.py`.
4. Backend lowering and runtime: if the backend overrides `emit_program`,
   teach it the new op (Samsung: `samsung_elementwise` plus `_emit_eltwise`
   and the conductor plan `samsung_eltwise_plan`; AiM: a new `AimOp` and
   `_AimLowerer` case; UPMEM: a family recognizer plus template). Make the
   runner's `RunResult.extra` describe what executed.
5. Tests: a matcher test (`tests/pim/test_match_patterns.py`,
   `tests/pim/test_match_coverage.py`), a compile-only test through
   `allo.compile(..., backend="virtual")`, and the backend's simulator or
   device test gated by `simulator_unavailable_reason`.

### 4.2 Adding a schedule knob

Worked example: AiM `reduction_partitions` (`spmw_knobs.
_aim_reduction_partitions_candidates`, `_aim_reduction_partitions_emit`;
layout carrier `aim_lowering.reduction_partitioned_layout`; codegen reader
`spmw_aim._reduction_partitions`; cost `allo/pim/costs/aim.py::host_reduction_cycles`).
`register_knob(target_name, Knob(name, candidates_fn, emit_fn))` in
`register_default_knobs`; the candidates function reads `KnobCtx(target,
matches, role_to_memref, base, liveness)` and may consult
`spmw_reduction_policy.reduction_policy(match)`; `emit` writes
`extra[name]` (and the `LinearLayout` when the knob changes the layout, so it
enters the decision key). Cost rules read `event.metrics["candidate"][name]`;
codegen reads `placement.extra[name]` or the layout and must raise
`InfeasibleSchedule` for values it cannot realize.

### 4.3 Adding a backend

Follow the AiM backend as the smallest complete example
(`allo/spmw_aim.py`, `allo/pim/aim_lowering.py`, `allo/pim/costs/aim.py`,
`spmw_autoschedule._aim_enumerate`, `spmw_simenv.aim_unavailable_reason`).

1. Target: `build_<name>_target()` in `allo/pim/targets.py` (structure only;
   `check_target` must pass). Export it from `allo/pim/__init__.py`.
2. Cost: `allo/pim/costs/<name>.py` with `@cost(target="<name>")` and one
   rule per reachable primitive; export from `allo/pim/costs/__init__.py`.
3. Enumerator: `@register_enumerator("<name>")` in `spmw_autoschedule.py`
   returning base `Placement`s crossed with `cross_with_knobs`; add the
   backend's compute levels to `spmw_layout_problem.COMPUTE_LEVELS` so the
   movement stage can derive the problem.
4. Codegen context: a `CodegenContext` subclass with `cmd` or an
   `emit_program` override, `runtime_route(commands)`, and optionally
   `virtual_run_extra()`; register it in `spmw_codegen._BACKENDS` as
   `(module, class name, runner name)`. If the backend accepts
   `mapping=[1]` or outer-subset grids, add it to
   `_SINGLE_ITEM_LAUNCH_BACKENDS` / `_OUTER_SUBSET_LAUNCH_BACKENDS`.
5. Runner: `_run_<name>(compiled, **inputs) -> RunResult` that raises
   `SimulatorUnavailable` when the external piece is missing, raises on
   missing outputs or a failed sub-process, and sets `kind` honestly
   (`simulator-timing-only` when no data path exists).
6. Discovery: every external path or image name goes through
   `allo/spmw_simenv.py` with an environment override; add
   `<name>_unavailable_reason()` and wire it into
   `spmw_codegen.simulator_unavailable_reason`; document the variables in
   `docs/release/INSTALL.md` and the external software in
   `docs/release/simulators.md`.
7. Policy: decide whether the backend has an order-preserving program and
   either consult `reduction_policy(match)` in the enumerator
   (`_POLICY_AWARE_MATCHER_ROUTES`) or add the reason to
   `_REGROUPING_MATCHER_ROUTES`.
8. Tests: `tests/pim/test_gemv_all_targets.py` (one GEMV per backend),
   `tests/perf/test_simulator_unavailable.py`, `tests/perf/test_run_result_contract.py`,
   `tests/pim/test_import_surface.py`, and `docs/release/CAPABILITIES.md`.

A program record is a *frontend* of the same pipeline instead: a
`normalize_<record>` function states its semantics as `ProgramFacts` (from
the shared front stages run on the record's source when it has one, as
`APUv1Program` does; as synthetic matches with the matcher's structural
facts when it has none, as the Gemini-II typed records do;
`allo/pim/record_frontend.py` holds the shared helpers), the target's
enumerator turns the record into candidates whose `CandidateEffects` the
shared validator checks, and the backend context materializes and runs
them (`apu_g2_record_frontend.py`, `apu_v1_record_frontend.py`).

## 5. Inspecting a chosen program

For a matcher-route `CompiledCallable`:

| What | Where |
|---|---|
| ranked placements per kernel | `compiled.placement_ranking.kernels[i]` (`KernelRanking`): `scores` `(position, cycles, mode)` in rank order, `chosen_position`, `cost_ties`, `tie_break`, `rejections` `(position, reason)`, `movement_rejections`, `policy_pruned` `(mode, reason)`, `fallback`, `bucket_count` |
| movement record | `compiled.placement_ranking.kernels[i].movement` (`KernelMovement`): `status`, `implementation`, `termination`, `optimality_proven`, `optimality_scope`, `incomplete`, `min_cost`, `program_cost`, `chosen_cost`, `chosen_label`, `tie_count`, `ties_crossed`, `ties`, `assumptions`, `notes`; `compiled.placement_ranking.incomplete` |
| reduction policy | `compiled.reduction_order`, `compiled.reduction_policy_pruned`; per match `compiled.trace.matches[j].extra["reduction_policy"]` |
| matcher facts | `compiled.trace.matches[j].extra` (`accumulator_init`, `axis_roles`, `batch_dim`, `constants`, `output_sinks`), `compiled.trace.coverage.stores` |
| chosen placement(s) | `compiled.compiled.layout` (a `Placement` or a list): `placements`, `extra` (knob values, `movement`, `reduction_program`, `reduction_policy_pruned`), `layout` (`LinearLayout`) |
| emitted program | `compiled.compiled.cmds` (Samsung `PIMCmd` list, AiM trace lines, UPMEM segment names); `compiled.compiled.layout_ctx` for backend state (Samsung `programs`, UPMEM `segments` with `device_source()`, AiM `host_steps`) |
| legality rejections | raised as `IllegalPlacement` for `layout=`; enumerated candidates the pass removed are appended to `KernelRanking.rejections` as `(None, "illegal placement <mode>: <rule> ...")` entries by `autoschedule()` |
| cost graph and estimate | `compiled.execution_graph` (`ExecutionGraph.activities`), `compiled.estimate()` (`Estimate.cycles`, `critical_path`, `utilization`, `bottlenecks`) |
| run outcome | `result.kind`, `result.cycles`, `result.extra` (Samsung: `timing_scope`, `logical_shape`, `padded_shape`, `padding_fraction`, `cycle_phases`, `plan`, `crf`, `conductor`, `levers`, `outputs`; AiM: `returncode`, `segments`, `unique_segments`, `host_steps`; UPMEM: `upmem` records with `sdk_build`, `logic_cycles`, `geometry`, `padding_fraction`, `host_steps`, plus `timing_scope`, `outputs`; virtual: `cost_model`, `priced_target`, `critical_path`, and `cost_validity`/`upmem_cost_validity` on UPMEM); every result also carries `reduction_order` and `reduction_policy` |

For the APU v1 vector route: `compiled.plans`, `compiled.ranked_plans`,
`compiled.candidate_estimates`, `compiled.selected_plan`,
`compiled.selected_estimate`, `compiled.realization`,
`compiled.plan_selection`, `compiled.reduction_policy_pruned`,
`compiled.execution_graph`, `compiled.estimate(plan=None)`;
`result.extra["plan"]`, `["plan_selection"]`, `["plan_regroups_reduction"]`,
`["numerics"]`, `["board_lock"]`. For `APUv1Program`: `compiled.route`,
`compiled.hybrid_manifest`, `compiled.native_vector_lowering`,
`compiled.device_source`, `compiled.lowered` (the `CompiledAPUv1Program`).
For Gemini-II
source: `compiled.compiled.plan`, `.ranking`, `.rejected`, `.selection`,
`.placement_ranking`, `.manifest()`, `.repetitions`; `result.extra`
carries `layout_fingerprint`, `program_fingerprint`, `packing_attestation`,
`estimate_validity`, `device_lock`. The same table is kept short in
`docs/release/INSTALL.md` section 5.

## 6. External dependencies and discovery

All locations outside the package are owned by `allo/spmw_simenv.py`; each
has an environment override (`TENON_SIMULATORS_DIR`, `PIMSIMULATOR_ROOT`,
`AIM_SIMULATOR_ROOT`, `TENON_AIM_SIM_IMAGE`, `UPIMULATOR_ROOT`,
`UPIMULATOR_BIN`, `TENON_UPMEM_SDK_IMAGE`, `TENON_UPMEM_BENCHMARK_SLOT`,
`TENON_UPMEM_SCRATCH`, `TENON_UPMEM_PROVENANCE_DIR`, `TENON_ARTIFACTS`,
`TENON_APU_V1_TOOLCHAIN_BASE`, `TENON_APU_V1_GVML_INCLUDE_ROOT`,
`TENON_APU_V1_TEMPLATE_DIR`, `TENON_APU_V1_PCI_NODE`, `TENON_APU_V1_LOCK`,
`TENON_APU_G2_SDK_ROOT`, `TENON_APU_G2_GSI_TOOL`, `TENON_APU_G2_DEVICE`,
`TENON_APU_G2_LOCK`, `TENON_APU_G2_KEEP_TMP`, `TENON_APU_G2_BUILD_JOBS`).
The variable table, defaults, timeouts, temporary directories and locks are
in `docs/release/INSTALL.md` section 3; the simulators and vendor SDKs,
their revisions and licenses are in `docs/release/simulators.md`. Nothing
external is bundled; every target compiles and runs `backend="virtual"` with
Allo alone.

## 7. Module map

| Module | Responsibility |
|---|---|
| `allo/compiler.py` | `allo.compile` (record frontends, shared front stages, common candidate pipeline), `CompiledCallable`, work-scope stamping, carried-accumulator binding |
| `allo/pim/record_frontend.py`, `allo/pim/apu_v1_record_frontend.py`, `allo/pim/apu_g2_record_frontend.py` | program records as frontends: synthetic semantic contract, record candidates with declared effects, realized record programs |
| `allo/spmw_pipeline.py` | shared front stages of every source workload: frontend, matcher view, coverage contracts, `SourceProgram` |
| `allo/spmw_semantics.py` | shared semantic facts (`ProgramFacts`, `Contraction`, `Epilogue`, `BufferAccess`, `OutputSink`, `ElementType`) derived from the match trace and retained IR types |
| `allo/spmw_target.py` | target language: units, memories, registers, ops, moves, patterns, host-move recording, host programs |
| `allo/spmw_match.py`, `allo/spmw_match_engine.py` | matcher data model, pattern compilation, store coverage, axis roles |
| `allo/spmw_reduction_policy.py` | reduction-order policy, stamps, route gates |
| `allo/spmw_legality.py` | target-neutral placement legality |
| `allo/spmw_autoschedule.py` | enumerators, `Placement`, ranking, `KernelRanking`/`PlacementRanking`, residency reconciliation |
| `allo/spmw_knobs.py` | typed knob registry and generic cross product |
| `allo/spmw_layout_problem.py` | movement-stage problem derivation, search dispatch, tie classification, `KernelMovement` |
| `allo/pim/gf2.py`, `allo/pim/layout_problem.py`, `allo/pim/layout_movement.py`, `allo/pim/layout_search.py` | Section IV algebra, movement formula, bounded/masked/scalable/coordinate searches |
| `allo/spmw_codegen.py` | `compile_for_target`, `CodegenContext`, `_BACKENDS`, work-grid check, materialization manifest, `Compiled`, `RunResult`, virtual runner |
| `allo/spmw_plan.py` | execution-graph construction, host transfer metrics, launch schedules |
| `allo/spmw_linear_layout.py` | `LinearLayout` F2 algebra (`compose`, `product`, `identity`, `to_gf2`/`from_gf2`, `materialise_handle`) |
| `allo/spmw_liveness.py`, `allo/spmw_tiling.py`, `allo/spmw_tripcount.py`, `allo/spmw_fingerprint.py` | whole-trace liveness for the residency knob, capacity tiling candidates, affine trip counts, canonical digests |
| `allo/spmw_host_program.py` | host-program residency analysis (no production caller, C-10) |
| `allo/spmw_simenv.py` | external paths, images, devices, availability probes |
| `allo/perf/` | cost language and evaluator |
| `allo/pim/targets.py`, `allo/pim/costs/` | reference targets and cost programs |
| `allo/spmw_samsung.py`, `allo/pim/samsung_layout_swizzle.py` | Samsung classification, CRF lowering, conductor plans, runner; XOR swizzle harness |
| `allo/spmw_aim.py`, `allo/pim/aim_lowering.py` | AiM program derivation, ISR lowering, coverage invariant, runner |
| `allo/spmw_upmem.py`, `allo/pim/upmem_physical.py`, `allo/pim/upmem_physical_search.py`, `allo/pim/upmem_irregular.py`, `allo/pim/upmem_analysis.py`, `allo/pim/upmem_calibration.py` | UPMEM families, templates, physical decision catalog and calibration, MLIR instruction summary, calibration tooling |
| `allo/backend/c.py` | MLIR to portable C (UPMEM oracle and host groups, APU v1 scalar programs) |
| `allo/spmw_apu_v1.py`, `allo/spmw_apu_v1_build.py` | APU v1 backend context (vector-plan candidates and the grouped program), runner, GVML project emission |
| `allo/spmw_apu_g2.py` | Gemini-II backend context (launch-plan and record candidates) and runner |
| `allo/spmw_candidate.py` | common candidate contract: `CandidateEffects`, per-candidate/artifact validation, supplied-placement rebinding, graph consistency |
| `allo/pim/apu_g2_record_frontend.py` | Gemini-II typed program records as frontends: synthetic semantic contract, record candidate, realized program |
| `allo/pim/apu_v1_*.py` | APU v1 vector plans, codegen, cost bridge, runtime, typed scalar programs, native recipes, hybrid manifest |
| `allo/pim/apu_g2_*.py`, `allo/pim/templates/apu_g2/` | Gemini-II typed, composed and source routes, tick model, runtimes, project templates |
| `allo/pim/schedule_search.py` | generic materialize-then-score grid search (joint evaluation of complete assignments) |
| `benchmarks/cent_aim/`, `benchmarks/upmem/`, `benchmarks/apu_g2/` | paper workloads as source |

## 8. Open items and compatibility boundary

Open findings that shape the architecture (details and status in
`docs/release/ISSUES.md`): C-10, S-11/S-13, P-17 (UPMEM SDK is
proprietary), V1-03 (default FP16 GEMV plan under `"reassociate"` is a tree
plan; strict selects a temporal plan; documented in section 3.4), G2-12,
P-04 (Samsung
simulator patches need Samsung's permission to publish; the overlay lives
in the containing repository). CLAIMS row 16 (movement ties) is partial for
the realizability reason in Section 2.5.

Tenon does not change upstream Allo's `allo.memory.Layout` or the FPGA/AIE
build paths; the PIM surface is additive (`allo/__init__.py` eager exports
plus `_LAZY_EXPORTS`, checked by `tests/pim/test_import_surface.py`). Cost
formulas stay out of target descriptions, and no runner falls back from an
unavailable simulator or device to an analytical estimate: it raises
`SimulatorUnavailable`, and only `backend="virtual"` returns an estimate.
