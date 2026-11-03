<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Core (target-neutral) compiler audit

Scope: `experiments/allo` branch `tenon` @ `8160d6d`, the target-neutral part of
the Tenon compiler (target/workload/cost languages, matcher, autoscheduler,
generic codegen, cost evaluator, linear layouts, host program) and the location
of the paper's minimum-movement layout search. Read-only audit; no tracked file
was edited. Probe scripts: `/tmp/tenon-release/probes/core/p*.py`.

Evidence labels: **executed** (probe or test run in this audit, output quoted),
**read** (source read, `file:line`), **suspected** (inference not yet executed).

Release criteria: `TASK.md` ("What counts as good compiler code", gates G1-G8).
Prior findings A01 (layout search outside the package), A02 (per-kernel ranking
then reconciliation) and A05 (`_eval_sym` default) were rechecked; all three
still hold at `8160d6d` (see C-02, C-05, C-07).

Tests executed during this audit (all pass; **executed**):
`tests/pim/test_import_surface.py`, `tests/perf/test_resource_model.py`,
`tests/perf/test_schedule_search_core.py`, `tests/pim/test_host_program_analysis.py`,
`tests/pim/test_host_transfer_codegen.py` (30 passed, 5.5 s);
`tests/perf/test_matcher_ranked_placement.py`, `tests/pim/test_match_patterns.py`
(24 passed, 90 s).

---

## 1. Paper claim -> package symbol map

Paper locations are the frozen `paper/latex/sections/*.tex`. Status: **implemented** /
**partial** / **missing**.

| # | Claim | Paper location | Package symbol(s) (`allo/...`) | Status | Notes |
|---|-------|----------------|-------------------------------|--------|-------|
| 1 | `@tn.target` creates the root; `@tn.unit(mapping=...)` adds a named physical axis; nesting forms the hierarchy | 3-language-and-compiler.tex, Sec. III-A | `spmw_target.py:439-457 target`, `:460-489 unit`, `:321-358 Unit` (`axes`, `mapping`, `parent`, `children`, `level`) | implemented | `Target.work_grid()` (`:411-430`) derives the PE grid from the tree. |
| 2 | `tn.mem` / `tn.reg` return handles recording owner unit and dimensions | III-A | `spmw_target.py:554-576 memory` (`mem` alias `:576`), `:579-602 reg`; `Memory.owner/geometry :72-104`; `Register.owner :119-163` | implemented | Geometry is a free-form dict; Samsung-shaped attributes (`banks/rows/cols/width`) are bound for convenience. |
| 3 | A memory belongs to its unit; descendants may access it, units outside the subtree may not ("handles determine whether a particular placement is legal") | III-A, Table I | **no symbol**. `Op.__init__ :250-316` and `op() :711-749` never compare `src/dst` owners with the op's unit; `_validate_move :666-690` checks only the host/device boundary; `spmw_codegen._resolve_layout :446-473` maps memref->handle with no legality test | **missing** | Probe P1 (C-01): a cross-subtree `src` is accepted at declaration; a hand-supplied placement that violates `MAC.dst=grf_b` is accepted, emitted and costs less than the legal one. |
| 4 | `tn.op(fn, emit, src, dst)`; `fn` matches workload expressions; `tn.any_` denotes any member of a storage set | III-A | `spmw_target.py:241-320 Op`, `:752-759 any_`, `:762-770 or_`; `spmw_match_engine.compile_op_pattern :194-238` (AST of the lambda) | implemented | `src/dst` are stored but only consumed by backend enumerators/emitters by name, not by a generic legality analysis (row 3). |
| 5 | `tn.move` declares a legal data path; host collectives scatter/broadcast/gather are moves | III-A | `spmw_target.py:218-239 Move`, `:693-708 move`, verbs `:211-215` (`broadcast/scatter/gather/move_only`), `BackendHandle.resolve :1225+`, `spmw_codegen._resolve_host_moves :1200-1244` | implemented | Device moves are selected structurally by `spmw_plan._moves_for_handles :83-110` (register endpoint identity). |
| 6 | `@tn.device` names the device subtree; `mode="host"` names the host side | III-A | `spmw_target.py:531-552 device`, `DeviceScope :500-528`, `Unit.mode`, `_is_host_side :651-657` | implemented | |
| 7 | `@tn.work(mapping=...)` grid; `tn.get_wid()`; workload records typed ops, loops, tensor accesses, dependences | III-B | `__init__.py:14` (`work = dataflow.kernel`, `get_wid = dataflow.get_pid`), `dataflow.py:542-561`, `ir/builder.py:94-100` (`spmw.group_id/work_id/group_shape/...`), `:186-240 _spmw_group_contract`, `compiler._stamp_matcher_work_scopes :365-444` | implemented | Work grid is the dataflow kernel instance grid; the retained-IR scope attributes are rename/relocation invariant (tests in `tests/perf/test_compile_interface.py`). |
| 8 | Matching replaces supported expression graphs with target ops while preserving loops, accesses, dependences | III-E | `spmw_match_engine.match_workload :1670-1721`, `_walk_loops :977-1035` (loop stack, `scf.if`/`affine.if` guards), `_try_match_at_store :1374-1584`; `spmw_match.MatchedOp :123-166` (`enclosing_loops`, `operands[].indices/index_map/value_ref`, `result_memref_name`, `is_loop_carried`) | partial | Preserves loops/accesses of **matched** stores. Unmatched stores are silently dropped (`:1399-1401`, `:1584` return `[]`); no whole-kernel coverage check; dependences are represented only as the accumulator flag, not a dependence graph (C-02). |
| 9 | `@tn.cost(target=...)`, `@tn.rule(handle)`, `ctx.step(latency, occupy=[ctx.use(...)])` | III-C | `perf/cost.py:1280 cost`, `:1304 rule`, `CostContext.use :203`, `.step :206-251`, `.repeat :253`, `.parallel :265`; example rule `pim/costs/samsung.py:43-52,82-84` | implemented | Signature matches the paper (`latency=`, `occupy=[ctx.use(pim, cycles=26), ctx.use(channel, cycles=4)]`). |
| 10 | Evaluator schedules under dependencies and resource capacity, exposing serialization and overlap | III-C | `perf/evaluator.py:102-172 Evaluator.evaluate` (ASAP over `depends_on` + per-`HandleInstance` calendars), `_Calendar :40-100`, `perf/graph.py:12-36 HandleInstance.capacity` | implemented | Probe P4 (**executed**): disjoint units overlap (26 cy), same unit serializes (52), shared channel issue slot (30), dependency serializes (20), capacity-2 handle admits two. Units are abstract cycles; `Estimate.cycles` is the makespan. |
| 11 | Replacing the cost program changes the winner but cannot make an illegal op/transfer legal | III-C | legality lives in enumerators (`spmw_autoschedule._samsung_enumerate :543-673`, `_aim_layout_candidates :676`, `_upmem_enumerate :1003`, `_apu_v1_enumerate :1024`, `_apu_v2_enumerate :1102`) and backend emitters | partial | True for the autoscheduler route (enumerators only emit handles they know are legal). False for the public `layout=` route (C-01). |
| 12 | `tn.compile` composes workload, target, cost; `estimate()`; virtual execution returns only the cost result | III-D, III-E | `compiler.compile :635-742`, `CompiledCallable.estimate :514-520`, `spmw_codegen.compile_for_target :1351-1523`, `_run_virtual :843-862` (`RunResult.backend="virtual"`) | implemented | `cost=None` is allowed by `compile` but `autoschedule` raises `TypeError` (`spmw_autoschedule.py:1168`). |
| 13 | "Combining partial sums in a different order requires explicit permission"; strict FP order can rule out splitting a reduction before search | III-D, III-E, VI | **no symbol** in the package (`grep -rn reassoc|permission|strict` over `allo/spmw_*.py`, `allo/perf`, `allo/pim/costs`, `allo/pim/targets.py` finds nothing) | **missing** | The snapshot's `typed_reuse_analysis.PartialResultPolicy`/`NumericPolicy` carries this; the package has no policy field, so no route can refuse a split (C-09). |
| 14 | Autoscheduler lifecycle: decision domains -> legality/pruning -> layout construction -> materialize exact program -> rank only materialized candidates, deterministic tie-breaking | III-F, Fig. 2 | domains: `spmw_knobs.register_default_knobs :446-515` + `cross_with_knobs :518-570`; legality: enumerator `ValueError`s, `derive_layout_properties :311-375`, `InfeasibleSchedule` from codegen; layout construction: `LinearLayout.identity/optimal_swizzle` in `_samsung_enumerate :571-584`; materialization: `spmw_codegen._materialize_matcher_codegen :1297-1348`; ranking: `rank_matcher_placements :1261-1414` (`scored.sort()` on `(cycles, position)` `:1371`) | partial | Order differs from the paper: cost ranking first (`:1349-1371`), then codegen of candidates in cost order until the first succeeds (`:1373-1385`); only `InfeasibleSchedule` is treated as a rejection, any other exception aborts the compile (C-03). Per-group, not whole-program (C-05). The per-bucket fallback (`_per_bucket_argmin_index :1441-1466`) ranks without any materialization probe. |
| 15 | "Each surviving complete assignment is evaluated as one program" | III-F | `rank_matcher_placements` scores each **kernel group** with `spmw_plan.build_execution_graph(kernel_trace, ...)` `:1356-1365`; cross-kernel decisions are reconciled afterwards by `_reconcile_resident_pairs :1469-1537` without re-costing | partial (A02) | Probe P3 (**executed**, MVT, two groups): each group ranked independently (128 buckets each, 16 tied minima at 188 cy, chosen = lowest position), reconciliation was a no-op, whole-program cycles 376 = 188 + 188 because groups are serialized in the graph. `pim/schedule_search.grid_search :754` (true complete-assignment search with incumbent, cap and ties) exists but is used only by `pim/apu_v1_vector_program.py:391` and UPMEM physical search, not by the matcher route. |
| 16 | Ties in movement survive to final cost ranking | III-E | **no movement stage in the package**; cost ties are broken by enumeration position (`:1371`) | **missing** | See Sec. 3. |
| 17 | Linear layout over F2; `D_x = A_x L_W` | II-C, IV-A | `spmw_linear_layout.LinearLayout :47` (`compose :444`, `product :483`, `invert :525-594`, `is_surjective :596`, `image_size :144-200`, `matrix :419`, `apply :399`, `optimal_swizzle :272`, `materialise_handle :725-870`) | partial | No access-matrix type, no `kernel`/`nullspace`/`rank` API (`image_size` computes 2^rank of a column block internally `:187-200`). Power-of-two sizes only (`_ilog2_exact :32-39`). |
| 18 | Rank formula `r_{j,s} = rank(Q_s|K_j)`; movement cost `sum c_js B_j (2^r - 1)`; owner/replication/partial-sum analysis | IV-B | **no symbol** in the package (`grep -rn "rank\|kernel\|nullspace\|movement\|owner" allo/spmw_linear_layout.py allo/pim/schedule_search.py` -> only F2 elimination inside `image_size` and search telemetry words) | **missing** (A01) | Implemented in the snapshot worktree (`prelayout_movement_algebra.full_carrier_rank_incidence`, `paper_partial_basis_search._paper_bound_for_prefix_v1`), Sec. 3. |
| 19 | Locality-chain search, optimistic rank bound, best-first with ties retained, budget exit returns best + remaining lower bounds; bounded (<=4 bits, 3 levels) and separate scalable implementation | IV-C, Alg. 1 | **no symbol** in the package | **missing** (A01) | Snapshot: `paper_partial_basis_search.py` (bounded, literal paper bound), `scalable_layout_synthesis.py` (scalable flag search). Sec. 3. |
| 20 | Masked domains: exact active-owner count, rank bound disabled, exact only when complete; non-power-of-two axes padded with a mask | VI, V-C | **no symbol**; package `LinearLayout` rejects non-power-of-two sizes (probe P9, **executed**: `ValueError size 3 is not a positive power of 2`) | **missing** | Snapshot: `prelayout_movement_algebra.exact_active_owner_incidence :238`, `admissible_layout_search` masked mode, `scalable_masked_layout_synthesis.py`. |
| 21 | 4-bit example: 20,160 layouts -> 315 chains -> 576 visited | IV-C | not reproducible from the package | **missing** | Reproduced by snapshot tests (`tests/perf/test_paper_partial_basis_search.py:117-131, 205-216`) and `reviews/experiments/BATCH-2026-09-29.md:102-105`. Sec. 3. |
| 22 | 121,968 small masked layouts checked exhaustively | V-B | not in the package | **missing** | `reviews/experiments/irregular_audit.py:46-110` + `verify_second_batch.py:63` against snapshot APIs. Sec. 3. |
| 23 | Host-device ABI, buffers, transfers are part of the materialized program | III-F | `spmw_codegen._materialize_matcher_codegen :1297-1348` (`commands`, `host_schedule`, `host_preloads`, `runtime_abi`, `runtime_segments` via `ctx.runtime_route :309`) | implemented | The ABI tuple is a manifest; its consumer is the backend runner. |
| 24 | Backend code generators and runtimes are target-specific | III-E | `spmw_codegen._BACKENDS :1040-1046`, `CodegenContext :202-317` (abstract `cmd`, `resolve_moves`, `emit_program`, `runtime_route`) | implemented | Out of scope here (per-backend audits). |
| 25 | Public surface `allo.target/unit/device/mem/reg/op/move/any_/or_/get_uid/work/get_wid/compile/cost/rule/...` | III | `allo/__init__.py:14-48` eager, `_LAZY_EXPORTS :55-100` (`compile`, `CostSpec`, `cost`, `rule`, `LinearLayout`, `Placement`, `autoschedule`, `compile_for_target`, host-program helpers, APU typed programs) | implemented | `tests/pim/test_import_surface.py` asserts: `import allo` loads none of `allo.pim`, `allo.compiler`, `allo.spmw_codegen`, `allo.spmw_autoschedule`, `allo.perf` (subprocess probe); every lazy export resolves; `compile`/`CompiledWorkload` are in `dir(allo)`; `allo.compile is allo.compiler.compile`. **executed**: passes. Note `schedule_residency`/`schedule_cost` are exported but have no production caller (C-10). |

---

## 2. Findings

Severity: **blocker** (contradicts a paper mechanism or a G2/G4 pass condition in the public route), **major** (release-quality defect, user-visible wrong or misleading behavior), **minor**.

### C-01 (blocker) Access-scope and operand legality are not analyzed; an illegal hand-supplied placement is accepted, emitted, and wins on cost

- Gate: G1 (paper mechanism without implementation), G2 (no explicit failure), G4 (legal layouts derived from the target).
- Where: `allo/spmw_target.py:250-316` (`Op.__init__` stores `src/dst`, no owner check), `:711-749` (`op()`), `:666-690` (`_validate_move` checks only host/device straddling), `allo/spmw_codegen.py:446-473` (`_resolve_layout` is a name->handle lookup), `allo/spmw_codegen.py:1457-1470` (`compile_for_target` with `layout=` bypasses the enumerators).
- What the code does (**read**): the only legality in the matcher route is the *absence* of illegal handles from the per-backend enumerators and whatever the backend emitter happens to reject by name (`spmw_samsung.py:128` rejects `srf` because it has no `PIMOpdType` mapping; `:65` says the bank-dst-with-GRF-src ISA rule is "enforced at emit"). Nothing reads `Op.src/dst` or the unit tree to decide whether a placement is reachable.
- Reproduction (**executed**, `/tmp/tenon-release/probes/core/p1_access_scope.py`, `p1b_opconstraint.py`):
  ```
  (a) allo.op/allo.move accepted cross-subtree src: Op('ADD_X', accumulates=False) | src owner: unit_a | op owner: unit_b
  (b) x_on_host_dram backend=None: REJECTED NotImplementedError: SamsungCtx: unknown handle type Memory.
  acc_on_grf_a (MAC declares dst=grf_b); x,y grf_a: ACCEPTED cmds=512 cycles=130 first=PIMCmd(type_='FILL', dst_='GRF_A', src0_='EVEN_BANK', ...)
  x on grf_b, y on grf_b, acc grf_b (all one register): ACCEPTED cmds=512 cycles=130
  target-legal but backend-unsupported: y on srf: REJECTED NotImplementedError: SamsungCtx: register 'srf' has no PIMOpdType mapping.
  baseline autoscheduler-like: x grf_a, y grf_a, acc grf_b: ACCEPTED cmds=512 cycles=182
  ```
  The placement violating the declared `MAC.dst=grf_b` constraint (and aliasing all three operands onto one register) is emitted as a 512-command stream and priced at 130 cycles, cheaper than the legal baseline (182). The host-DRAM placement is rejected only because the Samsung emitter cannot print a `Memory` handle, not because `host_dram` is outside the `pim` subtree. Conversely `srf`, declared legal by `any_([grf_a, grf_b, srf])` in `pim/targets.py:126`, is rejected by the emitter: the target declaration and the backend capability disagree.
- Proposed repair: add a target-neutral legality pass (new `allo/spmw_legality.py` or inside `spmw_autoschedule`) that, for every `MatchedOp` and candidate `Placement`, checks (i) each operand handle's owner is an ancestor-or-self of the matched op's unit (`Unit.parent` chain), (ii) the handle satisfies the op's `src`/`dst` patterns (`AnyOf`/`OrOf`/direct identity), (iii) each device move used by `_moves_for_handles` connects reachable storage. Run it in `rank_matcher_placements` (as `InfeasibleSchedule`) and in `compile_for_target` when `layout` is user-supplied (hard error). Reconcile `pim/targets.py` declarations with emitter capability (either remove `srf` from MAC operands or implement its `PIMOpdType`).

### C-02 (blocker) Unmatched operations are silently dropped; a kernel whose output is never produced compiles without warning

- Gate: G2 ("no dropped unmatched effect", "missing-output success").
- Where: `allo/spmw_match_engine.py:1399-1401` (non-computed store -> `return []`), `:1584` (no pattern matched -> `return []`), `:1569-1583` (multi-term flatten aborts silently), `match_workload :1702-1716` (collects only matches); `allo/compiler.py:414` raises only when a kernel has **zero** matches; no coverage check of stores/effects.
- Reproduction (**executed**, `p2_unmatched_op.py`, Samsung target, `backend="virtual"`):
  ```
  Samsung ops: ['MUL', 'ADD', 'MAC', 'MAD', 'RELU']
  compile OK; matches: 128 target ops matched: ['MAC']
  result memrefs: ['acc']
  warnings: []
  estimate cycles: 182
  variant2 compile OK; matches: 128 ops: ['MAC']
  ```
  Workload 1 stores `local_y[row0] = acc / local_x[0]`; the division has no target op, so the store of `y` disappears: the trace writes only `acc`, `y` is never produced, and compile/estimate succeed. Workload 2 (`y = acc; y = y - x[0]`) likewise drops the subtraction. On the Samsung runner the output would then come from the canonical reduce driver (A03), i.e. `W@x`, which is a covert substitution for the user's program.
- Proposed repair: after `match_workload`, walk every store/effect site collected by `_walk_loops` and require each to be covered by a `MatchedOp.op_range` or to be a recognised copy; raise an actionable error listing the unmatched op, its loop nest and the target ops considered. Also require every kernel output memref (ABI arguments written in the body) to be the `result_memref_name` of some match or of a declared host move.

### C-03 (major) Ranking precedes materialization and only `InfeasibleSchedule` counts as a rejection; the per-bucket fallback never materializes

- Gate: G4 ("accepted candidates have complete realizable code"), G2.
- Where: `allo/spmw_autoschedule.py:1349-1371` (score by `build_execution_graph`, catch only `InfeasibleSchedule`), `:1373-1385` (codegen probe of candidates in cost order, catch only `InfeasibleSchedule`), `:1441-1466` (`_per_bucket_argmin_index` ranks without any codegen), `:1321-1347` (fallback path used when buckets of a kernel do not share a decision key).
- What the code does (**read**): the result equals "cheapest materializable candidate" only if every codegen failure is an `InfeasibleSchedule`. Backend emitters raise `NotImplementedError`/`ValueError` (probe P1b shows `NotImplementedError` from `SamsungCtx`), which propagate out of `rank_matcher_placements` and abort the compile instead of falling through to the next candidate. `derive_layout_properties` `ValueError`s (`:338-341`, `:360`) are raised inside `realize()` outside the `try`.
- Proposed repair: wrap the materialization probe to convert backend `NotImplementedError`/`ValueError` into `Rejection`s with diagnostics (or make emitters raise `InfeasibleSchedule`), record them in `KernelRanking.rejections`, and run the same probe in the per-bucket fallback. Report "ranked N, materialized M, rejected R" in `PlacementRanking`.

### C-04 (major) Work-grid / target-grid mismatch is advisory only

- Gate: G2 (unsupported inputs must produce finite errors).
- Where: `allo/spmw_codegen.py:1139-1197` (`_check_work_grid`: "Warns + auto-fills; never raises"), `:1178-1196`; `spmw_codegen.py:747-760` (Samsung shared-CRF trigger-count guard, also a warning).
- Reproduction (**executed**, `p2_unmatched_op.py` variant 3, `mapping=[4,4]` on the 16x8 Samsung tree):
  ```
  variant3 (grid mismatch) compile OK; warnings: ['samsung shared-CRF: single-kernel trace has 16 work-id buckets but target geometry yields 128 (unit-tree fanout product); the shared-CRF trigger schedule may be ...']
  ```
  The compile proceeds with a trigger schedule the code itself says "may be mispriced"; the paper states mapping a work coordinate to a unit is a schedule decision, but the package neither maps a 16-item grid onto 128 units nor rejects it.
- Proposed repair: make the mismatch an error unless the target declares a sub-grid mapping policy (e.g. `@allo.work` on a prefix product is accepted as today; everything else rejected), or implement the paper's logical-to-physical grid mapping as a decision domain.

### C-05 (major, confirms A02) Per-kernel ranking followed by reconciliation; whole-program cost is not what selected the program

- Gate: G4 (joint optimization claim; "final ranking uses the graph corresponding to the realization").
- Where: `allo/spmw_autoschedule.py:1261-1414` (`kernels` loop, one choice per group), `:1413` (`_reconcile_resident_pairs` after all choices), `:1469-1537` (reverts to `restage` without re-scoring).
- Evidence (**executed**, `p3_two_groups.py`, Samsung MVT): two groups, each 128 buckets; each group ranked over its own choice set (16 ties at 188 cy), `chosen=32` = lowest position; final placements all `('dual_fiber+crf_shared', residency='restage', grf_residency={'local_A': 'crf'}, crf_issue='shared', stage_resident=False)`; whole-program graph of the final program = 376 = 188 + 188 (groups serialized). No `residency_pairs` were produced, so reconciliation did nothing here. **Read**: when the `residency` knob does produce a `resident` arm (`spmw_knobs._residency_candidates :214`), the reverted program is never re-priced, and `compile_for_target :1500-1511` builds the final `execution_graph` only for `Compiled.execution_graph`, not for selection.
- Proposed repair: either (a) route the matcher through `pim/schedule_search.grid_search` with one domain per kernel group and `max_complete_assignments` (the snapshot's `_search_autoschedule_program` did this as a shadow), scoring the whole-trace graph after reconciliation, or (b) document the route as a per-kernel heuristic and remove the joint-selection wording from release docs. Either way re-cost after `_reconcile_resident_pairs` and expose both numbers in `PlacementRanking`.

### C-06 (major) Deterministic tie-breaking by enumeration position, undocumented

- Gate: G4 (tie retention / deterministic tie-breaking).
- Where: `allo/spmw_autoschedule.py:1371` (`scored.sort()` on `(cycles, position)`), `position` = first-occurrence order of decision keys from `_bucket_decision_map :1222-1243`.
- Evidence (**executed**, P3): 16 candidates tied at 188 cycles; the chosen position was 32, the lowest tied position; `KernelRanking.scores` retains all ties. Deterministic, but the tie-break key is the knob registration order (`spmw_knobs.register_default_knobs :446-515`) rather than a documented structural key (contrast `pim/schedule_search._decision_order_key :205`).
- Proposed repair: break ties with a structural decision key (as `grid_search` does) and state the rule in the user docs; surface the tie set in `PlacementRanking`.

### C-07 (major, confirms A05) `_eval_sym` defaults unbound `UnitId`s to 0

- Gate: G2 ("no silent default unit coordinate").
- Where: `allo/spmw_codegen.py:319-359`, default at `:336-338`. Call sites: `allo/spmw_aim.py:120` and `:178` (bank index from `handle.idx`), with `env = {level: value for level, value in enumerate(self._active_work_id)}` (`spmw_aim.py:116-119`, `:174-177`); recursive use at `spmw_codegen.py:342`. No other backend calls it (Samsung uses `_bank_fiber_class :390` parity classification; UPMEM has its own affine evaluator `spmw_upmem.py:1361-1450`).
- Evidence (**executed**, `p4_evaluator_and_evalsym.py`):
  ```
  2*pid+1 with empty env -> 1
  2*pid+1 with env {1: 3} -> 7
  level-2 uid with env only level 0,1 (AiM style enumerate(work_id)) -> 0
  ```
  Classification: the AiM call sites are **concrete execution bindings** (the integer goes into the ramulator2 trace, `spmw_aim.py:115`), so a `UnitId` at a level deeper than the work-id tuple (e.g. the `bank` level under a channel-only `mapping=[32]`) is silently bank 0. The "representative trace" rationale in the docstring belongs to the cost path, which does not call `_eval_sym`.
- Proposed repair: make `_eval_sym` raise on an unbound `UnitId` and add an explicit `representative=True` parameter (or a separate helper) for callers that intentionally want a representative; have AiM bind every level from the placement/layout, not from `enumerate(work_id)`.

### C-08 (major) Target-neutral plan builder hard-codes Samsung GRF lanes

- Gate: G3, G5.
- Where: `allo/spmw_plan.py:667` (`lanes=8` in `build_execution_graph` metrics for every target), consumed by `allo/pim/costs/samsung.py:57` (`event.metrics.get("lanes", 8)`).
- What it does (**read**): the number of FP16 lanes per GRF (a Samsung ISA fact, `target.grf_a.lanes == 8`) is pasted into the generic graph builder instead of being read from the placement handle; a target with a different register width would be mispriced silently.
- Proposed repair: derive `lanes` from the operand/accumulator `Register.lanes` of the placement (or drop it and let the Samsung rule read `event.instance(handle)`), and delete the default in the rule.

### C-09 (blocker) No floating-point reduction-order policy exists in the package

- Gate: G2 ("unauthorized reassociation"), G1 (paper III-D/VI mechanism).
- Where: no symbol (`grep` over `allo/spmw_*.py`, `allo/perf/*.py`, `allo/pim/costs/*.py`, `allo/pim/targets.py`, `allo/compiler.py`). The Samsung `dual_fiber` candidate (`spmw_autoschedule.py:647-661`) splits the K reduction across even/odd bank fibers and the Samsung reduce reader sums partial lanes in FP32 (A04), both without any permission check; UPMEM `nc/kc` knobs (`spmw_knobs.py:435-441`) split reductions likewise.
- Proposed repair: add an explicit reduction policy to the workload or `compile(...)` (e.g. `reassociate=True|False` per reduction or per compile) retained on `MatchedOp.extra`, make every enumerator/knob that splits a reduction consult it (prune the split arm when strict), and make backends that cannot honour strict order fail explicitly. The snapshot's `PartialResultPolicy` / `NumericPolicy` enums are a usable model.

### C-10 (minor) Host program: analysis and "cost" are Python-side; no lowered host computation; two exported helpers have no production caller

- Gate: G2 ("a legitimate host partition is explicit and executes its real lowered host computation"), G7.
- Where: `allo/spmw_host_program.py:120-189` (`analyze`: residency/coalescing by scanning `HostMoveRecord`/`LaunchRecord`), `:192-205` (`schedule_residency`), `:208-227` (`schedule_cost` = `P + B*(E+R)` arithmetic with caller-supplied callables); executor `allo/spmw_samsung.py:714-765` (`execute_host_schedule_samsung`: one `run_samsung_reduce` per group, numpy stacking, "host pre/post-pass (alpha/beta, transpose, centering) is the caller's concern" `:709-711`). `grep -rn schedule_residency|schedule_cost allo/` finds no caller outside the module and `allo/__init__.py:61-62`.
- Assessment (**read**): the host partition is a Python orchestration of device launches; any host-side arithmetic a workload needs is left to test code. `schedule_cost` is a second, non-executable cost path parallel to `CostSpec`; it is exported publicly but unused by `compile`.
- Proposed repair: either lower host steps to an executable host program (and price them through `CostSpec` rules on the `mode="host"` unit, which `spmw_plan._emit_host :546-565` already does for transfers) or remove `schedule_residency`/`schedule_cost` from the public surface and label `HostSchedule` as the launch-coalescing analysis only.

### C-11 (major) Minimum-movement layout machinery is absent from the package (A01 restated with locations)

- Gate: G1, G4.
- Where (package): `allo/spmw_linear_layout.py` has no access matrices, kernel/nullspace/rank, owner images, chains or bounds; `allo/pim/schedule_search.py` is a generic finite grid search; backend enumerators hand-build one or a few layouts (`_samsung_enumerate :571-584` builds exactly one swizzled layout; `_apu_v1_enumerate :1057-1066` one; `_apu_v2_enumerate` one; AiM two). `derive_layout_properties :311-375` reads fan-out/conflicts from a carried layout but never movement.
- Consequence: paper claims 16-22 in Sec. 1 cannot be demonstrated from the release. See Sec. 3 for the salvage plan.

### C-12 (minor) `LinearLayout.identity` ignores input dims whose names are not output dims

- Where: `allo/spmw_linear_layout.py:224-255` ("Input dims whose name is not in `out_dims` get all-zero basis vectors").
- Evidence (**executed**, P4/P7): `LinearLayout.identity({"m": 4, "k": 4}, out_dims=("lane",))` yields a `(0, 4)` matrix and `is_surjective() == True` (vacuously), `image_size(varying_inputs=("k",)) == 1`. A caller expecting a 16-lane identity gets a layout that maps everything to lane 0 without error.
- Proposed repair: raise when an input dim has no matching output dim unless `allow_zero=True`; `is_surjective` should return `False` for zero output bits unless all output sizes are 1.

### C-13 (minor) Broad exception swallowing in the generic path

- Where: `allo/spmw_match_engine.py:656` and `:802` (`except Exception`), `allo/spmw_codegen.py:1324` (manifest failure recorded as `("unavailable", ...)` instead of failing the ABI), `allo/spmw_autoschedule.py:177-178` (`except (AttributeError, TypeError, ValueError): pass` while freezing features: an unfreezable feature silently leaves the decision key), `allo/compiler.py:708` (`except NoContractionError: pass`, legitimate fall-through).
- Proposed repair: narrow the matcher exceptions to the MLIR binding errors actually expected; make a non-manifestable host move or feature a hard error (it is part of schedule identity/ABI).

---

## 3. Minimum-movement search: candidate implementations outside the package

### 3.1 Candidates

| Candidate | Location (size) | Entry points | Algorithm vs. paper | Dependencies | Tests / evidence |
|-----------|-----------------|--------------|---------------------|--------------|------------------|
| **Prototype exhaustive synthesizer** | `dev/07292026-shared-loop-layout-autoscheduler/prototype/shared_autoscheduler.py` (483 lines); `synthesize_min_movement_layout :459-483`, `enumerate_invertible_work_layouts :297-327`, `_movement_terms :341-372`, `linear_layout_for_candidate :415-456` | one function; returns a single `winner` by `min(byte cost, rank cost, canonical bytes)` | Implements the rank formula `r = dim K - dim(K ∩ T_{s+1})` and `2^r - 1` for READ relations only; exhaustive over `GL(n,2)`, `n <= 4`; **no chains, no bound, no tie set (single winner), no masks, no budget exit**; unit bytes/unit cost per byte | imports `allo.pim.gf2`, `gf2_access_analysis`, `layout_problem`, `affine_access_analysis` from the **snapshot** (absent in `tenon`) | `prototype/test_shared_autoscheduler.py:68-79` pins GL counts `{1,6,168,20160}` and cost 12 on a 4-bit fixture |
| **Snapshot algebra core** | `.worktrees/allo-snapshot/allo/pim/gf2.py` (546), `layout_problem.py` (736), `prelayout_movement_algebra.py` (474) | `GF2Matrix.rank/nullspace/image/restricted_rank/inverse`, `GF2Subspace` (RREF, `intersection`); `OwnerAxis/OwnerLevel/OwnerHierarchy/LocalAxis/LocalSpaceFlag`; `full_carrier_rank_incidence :190`, `exact_active_owner_incidence :238`, `scale_extra_owner_incidence :408` | Exactly the paper's objects: owner hierarchy = compute levels `U_s`, local flag `T_s`, rank formula, byte scaling; masked exact active-owner count | `gf2` has no allo imports; `layout_problem` -> `gf2`; `prelayout` -> `gf2`, `layout_problem`, plus type-only imports from `affine_access_analysis`, `access_reduction_coverage`, `typed_reuse_analysis` (for `PrelayoutIncidenceKind` validation); caps `_MAX_CARRIER_BITS=6`, `_MAX_OWNER_LEVELS=3` | `tests/perf/test_gf2*.py`, `test_layout_problem*.py`, `test_prelayout_movement_algebra{,_oracle,_independent,_adversarial}.py` |
| **Snapshot bounded paper search (Alg. 1)** | `allo/pim/canonical_layout_enumeration.py` (1644) + `paper_partial_basis_search.py` (1445) | `enumerate_canonical_layout_candidates :1573`; `search_full_carrier_paper_partial_basis_layout_candidates :1361`; `PaperPartialBasisSearchPolicy(max_search_nodes=20_672, enable_pruning, coordinate_flags_only) :106`; outcome `:1261` (`best_seen_cost`, `best_seen_candidates` = tie set, `pruned_prefix_keys`, `unresolved_frontier_keys`, `termination`, `optimality_proven`) | Chains as nested RREF local-space flags built one basis vector at a time (`_prefix_children_v1 :809`); literal paper bound `u = min(dim K, dim(B ∩ K) + ell - q)` at `_paper_bound_for_prefix_v1 :911-985`; heap best-first (`_run_search_v1 :1081`); strict-greater pruning keeps ties; node budget with `unresolved_frontier_keys` returned. Domain: <= 4 carrier bits, <= 3 levels (`CanonicalLayoutEnumerationPolicy.max_carrier_bits=4`, `max_accesses=4`) | -> `gf2`, `gf2_access_analysis` (`IterationCarrierAxis`), `layout_problem`, `prelayout_movement_algebra`, `typed_reuse_analysis` (`TypedOperandReuseEntry` inside `CanonicalLayoutAccess :172-237`, which also re-validates the typed kernel against the access nullspace) | `tests/perf/test_paper_partial_basis_search.py:205-216`: 315 flags, 64 layouts/flag, 20,160 candidates, 511 prefix nodes, default budget 20,672; `:117-131`: 576-layout tie set preserved (`best_seen_candidates == 576`); `test_canonical_layout_enumeration_independent.py:351-379`: 35 flags x 576 frames; `reviews/experiments/BATCH-2026-09-29.md:102-105`: pruned search visits 576 of 20,160 at 2 and 3 levels (4,032 at 1 level) and agrees with exhaustive enumeration on cost and complete tie set. `design_doc/compiler/q7-ilp-subdomain-formulation.md:44` records 576 = 9 tied flags x 64 frames at 4 bits / 3 levels. |
| **Snapshot admissible search (bounded, incl. masked)** | `allo/pim/admissible_layout_search.py` (1313) | `search_admissible_layout_candidates(input_axes, hierarchy, accesses, active_iteration_points, forced_local, policies) :1108`; `LayoutSearchMode.MASKED_PRUNING_DISABLED` | Full-carrier: complete-flag rank bound; partial carrier (mask): pruning disabled, direct per-layout active-owner incidence, exact only on exhaustion (matches paper Sec. VI) | same as above | `tests/perf/test_admissible_layout_search{,_oracle,_independent,_adversarial}.py`; **121,968 check**: `reviews/experiments/irregular_audit.py:46-110` enumerates with `enumerate_canonical_layout_candidates`, cross-checks each layout with an integer-parity oracle (bijection, active coverage, poison padding, modular partial sums, owner terms), then asserts `search_admissible_layout_candidates` returns the same minimum and tie set with `pruned_candidate_leaves == 0`; `verify_second_batch.py:63` asserts the total `121968` over 12 cases |
| **Snapshot scalable flag search** | `allo/pim/scalable_layout_synthesis.py` (4589) | `validate_scalable_full_carrier_request :1040`, `search_scalable_layout_flags :3015`, `complete_scalable_layout_representative :3455`; `ScalableLayoutSearchTermination {EXHAUSTED, BOUND_PROVED, NODE_BUDGET}`; result `:2897` with `common_minimum_ties`, `pruned_frontier_bounds`, `unresolved_frontier_bounds` | "Searches chains directly": lazy relative-RREF prefix traversal without materializing basis frames; exact weighted lower bound; flag-level best-first; deterministic representative completion; node budget | -> `gf2`, `gf2_access_analysis`, `layout_problem`, `loop_nest_analysis` (`LoopId`), `prelayout_movement_algebra`, `typed_reuse_analysis` | `tests/pim/test_scalable_*`; Q7 scaling in `BATCH-2026-09-29.md:107-111` (one-level 5/6-bit optima proved at a 256-node budget; multi-level cases return node-budget results; 8-32-bit cases time out) |
| **Snapshot scalable masked search** | `scalable_masked_layout_synthesis.py` (2385), `scalable_masked_layout_fallback.py` (477), `scalable_masked_layout_completion.py` (489) | `synthesize_scalable_masked_layout :2154` (admissible coset bound, strict-worse pruning only, point-term budget), fallback to the bounded admissible search | Beyond the paper (paper says the masked search disables bounds; this adds a sound coset bound) | as above | `tests/pim/test_scalable_masked_layout_{synthesis,completion,fallback}.py` |
| **Snapshot GL(n,2) bounded synthesizer** | `bounded_layout_search.py` (636) | `synthesize_bounded_linear_layout_v1 :504` | exhaustive `GL(n,2)`, `n<=4`, READ relations, no masks (productionised version of the prototype) | `gf2`, `gf2_access_analysis`, `layout_problem`, `affine_access_analysis` | `tests/perf/test_bounded_layout_search.py` |
| **Snapshot wiring into compilation** | `allo/spmw_autoschedule.py:383-400` (`MatcherScheduledPlacements` carrying `schedule_search_result`), `:1767-1800` (`_search_autoscheduler_program`, "Multi-group winners remain shadow-only") | | Even in the snapshot the movement search never selected the executable program; it produced shadow evidence while the legacy per-group incumbent was activated | `bounded_joint_search.py` (812) and ~70 `apu_g2_*`/`shared_*`/`source_derived_*` adapters and `*_paper_receipt.py` modules | |

Input-side dependency chain (sizes, snapshot): `affine_access_analysis.py` 1720, `loop_nest_analysis.py` 1320, `gf2_access_analysis.py` 1792, `typed_reuse_analysis.py` 2036, `typed_semantic_analysis.py` 3513, `reduction_analysis.py` 1501, `access_reduction_coverage.py` 1123, `reuse_space_analysis.py` 771. `reviews/experiments/layout_fixture.py` (73 lines) shows the minimum a caller must construct by hand: `IterationCarrierAxis`, `OwnerHierarchy`, `GF2Matrix` access rows, `TypedOperandReuseEntry` (12 fields drawn from 7 modules), `CanonicalLayoutAccess(typed_entry, matrix, offset, incidence_kind, element_bytes, cost_per_byte)`.

### 3.2 Assessment

- The algorithm the paper describes exists and is well tested, but only in the snapshot worktree (`df35093`) and only as a "source-less", "shadow" surface. Its public API requires a `TypedOperandReuseEntry` built by the snapshot's typed-semantics stack (reduction identity, initializer evidence, numeric policy, partial-reuse eligibility), which is what makes it unusable from `tenon` and heavy to carry.
- The prototype in `dev/` is a strict subset (exhaustive, READ-only, single winner) and should not be salvaged; `bounded_layout_search.py` already supersedes it.
- The 20,160 / 315 / 576 and 121,968 numbers are reproducible from snapshot tests and `reviews/experiments` drivers; none is reproducible from the release package.

### 3.3 Salvage recommendation (concrete)

Create `allo/pim/layout_search.py` (plus `allo/pim/gf2.py`) in the package, with
`spmw_linear_layout.py` gaining the bridge to/from `GF2Matrix`:

1. **Copy verbatim, drop nothing**: `gf2.py` (no allo imports). Add `LinearLayout.to_gf2()` / `LinearLayout.from_gf2(matrix, in_dims, out_dims)` to `spmw_linear_layout.py` (the snapshot's `scalable_masked_layout_completion._matrix_from_linear_layout :111` and `paper...linear_layout_for_bounded_candidate` show the conversion).
2. **Copy with trimmed imports**: `layout_problem.py` (`OwnerAxis`, `OwnerLevel`, `OwnerHierarchy`, `LocalAxis`, `LocalSpaceFlag`); build the `OwnerHierarchy` from the `Target` unit tree (`Unit.mapping` extents, power-of-two check) and from `Register.axes` for lane-type owners, so the paper's "any hierarchy of compute axes" comes from the target description, not from a fixture.
3. **Copy and simplify**: `prelayout_movement_algebra.py` -> keep `full_carrier_rank_incidence`, `exact_active_owner_incidence`, `scale_extra_owner_incidence`; replace `TypedOperandReuseEntry`/`PrelayoutIncidenceKind` validation with a small `LayoutAccess` dataclass: `(name, access_matrix: GF2Matrix, mode: READ|WRITE_ACCUMULATE, element_bytes, cost_per_level: tuple[Fraction, ...], reassociation_allowed: bool)`. The `reassociation_allowed` flag is the package-side home for C-09 (a strict reduction contributes no distributable owner term and forbids owner bits that split its kernel).
4. **Copy the search kernels**: `canonical_layout_enumeration.py` (`_prefix_children_v1`, flag ranking, frame completion) and `paper_partial_basis_search.py` (`_paper_bound_for_prefix_v1`, `_run_search_v1`, policy, outcome) as the **bounded** implementation; `scalable_layout_synthesis.py` `search_scalable_layout_flags` + `complete_scalable_layout_representative` as the **scalable** one; `admissible_layout_search.py` masked mode (and optionally `scalable_masked_layout_synthesis.py`) for masks. Remove the `*_sha256` authority/manifest fields, `_reject` enums that only encode task numbers, and the `type(x) is not T` guards that forbid subclassing; keep the telemetry (`visited`, `pruned`, `unresolved_frontier`) because the paper's budget-exit contract depends on it.
5. **Derive accesses from the matcher**: a new `spmw_layout_problem.py` builds `LayoutAccess` rows from `MatchedOp.operands[].indices/index_map` and `enclosing_loops` (the affine maps are already retained as text; power-of-two bounds via `spmw_tripcount.resolve_trip_count`), with the iteration-bit order fixed by the loop nest and the work-grid bits from `MatcherWorkScope.group_shape`. Do **not** port `affine_access_analysis`/`loop_nest_analysis`/`typed_semantic_analysis` (8.9k lines) in this step; reject non-affine or non-power-of-two accesses explicitly (masks come in step 7).
6. **Expose**: `search_min_movement_layout(target, trace_group, *, policy) -> LayoutSearchResult(ties: tuple[LinearLayout,...], cost, termination, remaining_lower_bounds)` and wire it into `rank_matcher_placements` as the layout-construction stage: every tied minimum-movement layout becomes a `Placement.layout` candidate crossed with the knobs, so "ties in movement survive to final cost ranking" is true by construction; a `NODE_BUDGET` termination must mark `PlacementRanking` as incomplete.
7. **Masks**: carry an `active_points` mask on the layout problem (from non-power-of-two loop bounds) and dispatch to the admissible masked mode; port the `irregular_audit.py` oracle as a package test to re-establish the 121,968 figure, and port `test_paper_partial_basis_search.py:117-131, 205-216` for the 315/576/20,160 figures.
8. **Do not transplant**: `bounded_joint_search.py`, every `apu_g2_*`/`shared_*`/`source_derived_*`/`streamhls_*`/`*_paper_receipt.py` adapter, the snapshot's `MatcherScheduledPlacements.schedule_search_result` shadow plumbing, and the `dev/` prototype.

Estimated carried code after trimming: `gf2.py` (~550) + `layout_problem.py` (~500) + movement algebra (~250) + enumeration/paper search (~1,500) + scalable search (~2,000, optional for a first release but required for the paper's "separate scalable implementation") + masked mode (~800).

---

## 4. Constants / specialization inventory (core files)

| Item | Location | Category (TASK.md table) | Judgment |
|------|----------|--------------------------|----------|
| `lanes=8` metric for every target | `spmw_plan.py:667`; default `8` in `pim/costs/samsung.py:57` | architectural vector width pasted into a target-neutral path | **defect** (C-08): derive from the placement `Register.lanes`. |
| `banks_per_pim ... else 2`, `"grf": 8`, `"bank": bank_out or 16` | `spmw_autoschedule.py:575-578` | Samsung ISA facts with fallbacks | the fallbacks mask a missing target fact; make `_bank_out_size`/`_pim_unit_count` failures errors. |
| `group_size` default `32768` | `spmw_codegen.py:633` | APU v1 VR lane count | should come from `target.vr0.axes["lane"]` (as `_apu_v1_enumerate :1046` already does). |
| `UnitId` default `0` | `spmw_codegen.py:336-338` | guessed coordinate | **defect** (C-07). |
| Samsung cycle constants `MAC_CYCLES=4`, `BANK_LOAD_CYCLES=26`, ... | `pim/costs/samsung.py:20-27` | measured/datasheet coefficients | acceptable: in the cost program, documented as the calibration surface; provenance (PIMSimulator version, units) should be stated in the module docstring. |
| `_MATCHER_PHYSICAL_EXTRA_FIELDS` allowlist | `spmw_autoschedule.py:112-146` | schedule-identity allowlist | acceptable as an explicit identity contract; `_freeze_matcher_feature` silently drops unfreezable values (`:177-178`), see C-13. |
| `_SCHEDULE_SEARCH_KNOBS` + `SPMW_DISABLE_SCHEDULE_SEARCH` env var | `spmw_knobs.py:44`, `:541` | search policy switch | acceptable for A/B evaluation; document it or remove it from the release. |
| `_HOST_STAGING_STUB_NAMES`, `_HOST_STAGING_STUB_PREFIXES` | `spmw_target.py:634-635` | name-based rejection of removed stubs | historical; harmless but name-keyed; consider removing with the history. |
| `_SAMSUNG_MAC_ROLE_KWARGS = (("x","A"),("y","B"))` | `spmw_codegen.py:880` | backend ABI role table in the generic module | move to `spmw_samsung.py`; structural binding (`source_backend_binding :883-935`) is otherwise sound. |
| `max_complete_assignments`, incumbent handling | `pim/schedule_search.py:754-840` | explicit search policy with coverage telemetry | acceptable; unused by the matcher route (C-05). |
| Work-grid auto-fill when `observed == 1` | `spmw_codegen.py:1178-1189` | silent default | see C-04. |
| `Memory.banks/rows/cols/width` convenience attributes | `spmw_target.py:88-93` | Samsung-shaped naming in the neutral model | cosmetic; fine if documented. |
| `pass`/broad `except` | listed in C-13 | | |

Power-of-two restriction (`spmw_linear_layout._ilog2_exact :32-39`) is a real limit of the F2 algebra and is enforced explicitly (good), but the mask path that the paper promises for non-power-of-two shapes does not exist (row 20).

---

## 5. Open questions

1. **Which route should host the movement stage?** `rank_matcher_placements` scores per kernel group; the layout search is per group too (one iteration space), so step 6 of Sec. 3.3 fits, but whole-program ties (C-05) still need the `grid_search` route or an explicit heuristic label.
2. **How much of the snapshot's typed-semantics stack is required for soundness?** The snapshot validates the typed reduction kernel against the access nullspace (`CanonicalLayoutAccess :224-236`) and uses `PartialResultPolicy`/`NumericPolicy` to decide whether an output access contributes a distributable owner term. The lightweight `LayoutAccess` proposed above must reproduce exactly those two decisions (reduction identity and reassociation permission) from the matcher's `MatchedOp` (accumulator role + a new user policy); whether `MatchedOp` carries enough (initialization evidence, aliasing) is unverified (**suspected** gap, overlaps with G2 initialization checks).
3. **576 interpretation.** The paper says the bounded search "visits 576 of the 20,160 layouts"; the snapshot evidence shows 576 = 9 tied flags x 64 frames (3 levels) and = the 576 frames of one flag (2 owner bits), i.e. the visited set is exactly the tie set. If the release reproduces this figure it should state it as "576 complete layouts, all tied at the minimum" to avoid implying 576 distinct bound evaluations (**suspected**).
4. **Masked search scope.** The paper (Sec. VI) says the masked search disables the rank bound and is exact only when complete; the snapshot's `scalable_masked_layout_synthesis` adds an admissible coset bound. Decide whether the release ships the paper's simpler contract (admissible masked mode only) or the stronger one; both need the 121,968 oracle as a package test.
5. **Virtual run output contract.** `CompiledCallable.__call__` with `backend="virtual"` returns a `RunResult(backend="virtual")` and leaves output arrays untouched (`_run_virtual :843-862`, `_copy_outputs :540-583` finds no `outputs`). This is distinguishable by `RunResult.backend`, but a caller comparing `y` without checking `backend` sees stale data; consider raising when outputs are requested from a virtual run (**read**).
6. **`srf` legality.** `pim/targets.py:126` declares `srf` a legal MAC operand; `spmw_samsung.py:128` cannot emit it. Which is the hardware truth (ISA-1.0 allows `SRF_M/SRF_A` sources per the comment at `pim/targets.py:122-124`)? If the emitter is incomplete, the enumerator's omission of `srf` candidates is a hidden domain restriction.
7. **Dependences.** The matcher records the accumulator dependence only (`is_loop_carried`); whether cross-statement dependences inside a kernel body (e.g. `y` produced then consumed) are preserved for scheduling is untested here; `spmw_liveness.trace_liveness` covers cross-kernel memref spans only (**suspected**).
