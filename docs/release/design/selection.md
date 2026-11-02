# Placement selection contract (paper Section III-F, "Autoscheduling")

Status: implemented in `allo/spmw_autoschedule.py::rank_matcher_placements`
(matcher route: Samsung, AiM, UPMEM, APU v1 matcher route). Addresses release
audit findings C-03 (ranking precedes materialization) and C-05 (per-kernel
ranking, whole program not re-costed) in `docs/release/audit/core.md`, claim
rows 14 and 15 of `docs/release/CLAIMS.md`. Tests:
`tests/perf/test_joint_selection.py`, `tests/perf/test_matcher_ranked_placement.py`,
`tests/perf/test_schedule_search_core.py`.

## 1. Terms

| Term | Meaning |
|---|---|
| group | One `@allo.work` kernel (`MatcherWorkScope.group_id`); its buckets are its work items. |
| candidate | One position in a group's choice set: a decision key (backend placement crossed with knobs) crossed with one movement tie (`spmw_layout_problem`). |
| surviving candidate | A candidate that passed legality (`spmw_legality.check_placement`), the movement realizability check, the reduction-order policy pruning and every cost rule (no `InfeasibleSchedule`), and was not rejected by a code generation probe. |
| complete assignment | One surviving candidate per group, in `PlacementRanking.kernels` order. |
| probe | `spmw_codegen._materialize_matcher_codegen` (the backend's `emit_program` over a sub-trace, frozen into a `MatcherCodegenArtifact`). |
| whole-program cycles | `bound_cost.evaluate(spmw_plan.build_execution_graph(target, trace, placements, ..., launch_schedule=...))` over the full trace, after `_reconcile_resident_pairs`. This is the graph `compile_for_target` retains for `estimate()`. |

## 2. Stages

Selection runs inside `autoschedule()`; the policy is
`SelectionPolicy(joint_budget=256, max_reconcile_iterations=4,
materialize="lazy")` (`DEFAULT_SELECTION_POLICY`), overridden with
`with selection_policy(SelectionPolicy(...)):` or the `selection_policy=`
argument of `autoschedule`.

1. **Legality and pruning.** The backend enumerator yields base placements
   crossed with the registered knobs; illegal candidates are recorded on
   `KernelRanking.rejections` with position `None`; policy-pruned base modes
   on `KernelRanking.policy_pruned`.
2. **Per-group scoring.** Each group's candidates are realized
   (`derive_layout_properties`, cross-kernel stamps, host moves) and scored
   with the executable cost program over the group's sub-trace. A program
   with one group is scored over the whole trace with the launch schedule,
   so its group scores are whole-program cycles. A `ValueError` raised while
   realizing a candidate and an `InfeasibleSchedule` raised by a cost rule
   are recorded rejections.
3. **Per-group pick.** Candidates are ordered by `(cycles, TIE_BREAK_RULE
   key, position)` and probed in that order; the pick
   (`KernelRanking.per_group_position`) is the first whose probe succeeds.
4. **Reconciliation.** The picks are assembled and
   `_reconcile_resident_pairs` keeps a cross-kernel `resident` choice only
   when both endpoints chose it; it returns the reverted
   `(placement index, memref)` pairs. When it reverted anything, each
   affected group is re-ranked: every surviving candidate of that group is
   scored as a whole program (others fixed, reconciliation applied), and the
   cheapest whose probe succeeds replaces the pick (ties keep the group's
   rank order). Rounds repeat until no choice changes or
   `max_reconcile_iterations` rounds ran (`ReconciliationRecord.fixed_point`).
   The result's whole-program cycles are `PlacementRanking.per_group_cycles`.
5. **Joint evaluation.** `joint_assignments` is the product of the groups'
   surviving candidate counts. When it is at most `joint_budget`,
   `pim.schedule_search.grid_search` runs over one `DecisionDomain` per group
   (values: surviving positions in group rank order): `build` assembles and
   reconciles the assignment, `score` prices the whole program, the
   objective is whole-program cycles, the objective domain is
   `("whole-program cycles", target, cost fingerprint)`, and the stage 4
   result is the incumbent. Ranked assignments are then probed in order and
   the first that materializes is selected (`joint_evaluation="complete"`).
   Otherwise the stage 4 result is selected
   (`joint_evaluation="per-group heuristic"`).
6. **Final program.** `compile_for_target` re-checks legality of the
   selected placements, emits the whole program with the module and launch
   schedule, builds the retained execution graph, and sets
   `PlacementRanking.program_cycles` from it (equal to
   `CompiledCallable.estimate().cycles`).

## 3. What is materialized when

The contract implemented is **lazy materialization with fall-through**
(`materialize="lazy"`, the default):

- Probes run in cost order. A probe failure of type `InfeasibleSchedule`,
  `NotImplementedError` (including `SamsungUnsupported`) or `ValueError`
  (`candidate_rejection_types()`) is a recorded rejection
  (`"<ExceptionType>: <message>"`, bare message for `InfeasibleSchedule`)
  and the next candidate is probed. Any other exception type is a
  programming error and propagates.
- The selected program is identical to the one an eager pipeline would
  select (materialize every surviving candidate, then rank only the
  materialized ones by the same key): cost scores do not depend on the
  probe, so the first materializable candidate in cost order is the minimum
  over the materialized set. What differs is the record: under the lazy
  contract the candidates ranked after the selected one are scored but not
  generated. `KernelRanking.probe_count` and `KernelRanking.materialized`
  (positions whose probe succeeded) say how many were generated;
  `PlacementRanking.materialization` aggregates
  `ranked / probed / materialized / rejected / assignments_probed`.
- Probe scope: a group's candidate is probed over the group's sub-trace
  (the whole trace for a one-group program). In the joint evaluation of a
  program with several groups, a complete assignment is probed over the
  whole trace, reconciled, as one program. In the per-group heuristic route
  only the group sub-traces are probed; the whole program is first emitted
  by `compile_for_target`.
- `materialize="eager"` probes every scored candidate of every group before
  the pick, and in the joint evaluation probes every complete assignment in
  `grid_search`'s materialize stage (before scoring). It selects the same
  program (`test_eager_materialization_selects_the_lazy_program`).
- Per-bucket fallback (groups whose buckets share no decision key): each
  bucket ranks its candidates by `(cycles, index)`; the kernel assembled
  from the per-bucket argmins is probed over the group sub-trace (a single
  bucket is not a program a backend accepts, e.g. Samsung row coverage).
  On failure every bucket advances to its next-ranked candidate together
  (rank levels 1, 2, ...), up to `FALLBACK_REPAIR_PROBES = 64` further
  probes. This is a heuristic repair, not an enumeration of per-bucket
  combinations; a lockstep backend (Samsung) refuses kernels whose work
  items carry different placements.
- No candidate materializes: the first `NotImplementedError` or
  `ValueError` a backend raised (in cost order, i.e. the cheapest
  candidate's domain diagnostic) is re-raised with a note listing every
  rejection; when every rejection is an `InfeasibleSchedule` the error is a
  `RuntimeError` with the list.

## 4. Ties

- Within a group: `(cycles, structural key, position)` with the structural
  key of `TIE_BREAK_RULE` (carried F2 layout map, knob and feature values,
  operand handle paths, all role-normalized text). `KernelRanking.cost_ties`
  lists the candidates at the pick's cycles.
- Across complete assignments: `grid_search` orders by whole-program cycles,
  then the incumbent (the stage 4 result) first, then enumeration order,
  which is lexicographic over the groups' rank orders. Equal-cycle
  assignments therefore keep the per-group pick; a different program is
  selected only when it is strictly cheaper as a whole.
  `JointEvaluation.ties` counts the feasible assignments at the selected
  cycles, `incumbent_rank` the incumbent's rank.

## 5. Guarantees

| Label | Domain | Guarantee |
|---|---|---|
| `"complete"` | product of surviving candidate counts at most `joint_budget` | The selected program has the minimum whole-program cycles among complete assignments whose code generation succeeded, over the surviving candidates of every group, with the tie order of Section 4. `JointEvaluation.termination == "exhausted"` and `assignments_evaluated == assignments_total == joint_assignments`. |
| `"per-group heuristic"` | product above the budget | Each group's pick minimizes its group-scope cycles among its materialized candidates; groups touched by a reconciliation revert are re-ranked under whole-program cycles with the other groups fixed (coordinate descent, bounded). No optimality claim across groups: a coupling that needs two groups to change together is not found (`test_over_budget_selection_is_labelled_heuristic`). |

In both routes `selection_cycles` and `program_cycles` are whole-program
cycles of the selected program; `per_group_cycles` is the whole-program
cost of the stage 4 result, so `per_group_cycles - program_cycles` is what
the joint evaluation gained.

What the guarantee does not cover: the surviving candidates are those the
backend enumerators, knobs and the movement stage produce (at most
`MovementSelectionPolicy.max_tie_candidates` movement ties per group); the
per-bucket fallback groups contribute their one assembled candidate; the
cost program is the model of record, not the simulator.

## 6. Fields a user reads

`compiled.compiled.placement_ranking` (`PlacementRanking`):

| Field | Meaning |
|---|---|
| `joint_evaluation` | `"complete"` or `"per-group heuristic"`. |
| `joint_budget`, `joint_assignments` | The policy budget and the product of surviving candidate counts it was compared with. |
| `joint` | `JointEvaluation` (complete route only): `assignments_total`, `assignments_evaluated`, `feasible`, `rejections_by_stage`, `rejections`, `incumbent`, `incumbent_rank`, `ties`, `chosen` (`(group_id, position)` per group), `chosen_rank`, `probed`, `probe_rejections`, `whole_program_evaluations`, `termination`, `elapsed_seconds`. |
| `per_group_cycles`, `selection_cycles`, `program_cycles` | Whole-program cycles of the stage 4 result, of the selected program as scored by selection, and of the retained final graph. |
| `reconciliation` | `ReconciliationRecord(reverted, iterations, fixed_point, changed_groups, remaining)`. |
| `materialize`, `materialization` | The contract that ran and its counts. |
| `kernels[i]` | `KernelRanking`: group-scope `scores`, `per_group_position`, `chosen_position` (the selected program's choice), `rejections`, `probe_count`, `materialized`, `cost_ties`, `movement`, `policy_pruned`. |

## 7. Compile time

Wall-clock `allo.compile` on the shared host (48 cores, other jobs
running), before (`63ea0e8` worktree) and after (this change) run
alternately, two runs each:

| Workload | Groups | Route after | Before (s) | After (s) |
|---|---|---|---|---|
| Samsung paper GEMV `gemv_m4096_k256` (reassociate) | 1 | complete (48 assignments, 5 probes) | 16.2, 16.1 | 16.1, 16.2 |
| AiM `attention_l128` (golden row) | 22 | per-group heuristic (about 7.3e26 assignments, 24 probes) | 15.6, 15.6 | 15.5, 15.7 |
| UPMEM GEMV (`test_gemv_all_targets`) | 1 | complete (2 assignments, 1 probe) | 2.47, 2.44 | 2.49, 2.51 |
| Samsung MVT two groups (audit P3 script) | 2 | complete (4 assignments, 6 group probes and 1 whole-program probe) | 19.7, 19.7 | 20.5, 20.9 |

Eager materialization was measured as too slow for the default: a probe
costs about 0.06 s on AiM and Samsung, and AiM `attention_l128` has about
1,080 scored candidates over 22 groups (about +65 s).

## 8. Findings while implementing

- The cross-kernel `resident` arm is inert today. `trace_liveness` marks a
  span `crosses_kernel` only when exactly one producer work item and one
  consumer work item touch the value, and in dataflow regions each kernel
  argument is its own ABI value, so no matcher program reaches
  `_reconcile_resident_pairs` with a cross-kernel pair; no cost rule reads
  `residency` and no emitter reads `residency_pairs`. The conflict test
  therefore declares the coupling (monkeypatched
  `_residency_crossing_memrefs`) and prices it with a test cost spec.
- On the default cost specs the two routes agree on every program measured
  (Section 9): the whole-program graph serializes groups, so without a
  residency coupling the whole-program cost is separable by group.

## 9. Evidence

Two routes on the same programs (`SelectionPolicy(joint_budget=0)` versus
the default 256; `samsung_cost`, virtual backend):

| Program | Assignments | Heuristic `program_cycles` | Complete `program_cycles` | Choices |
|---|---|---|---|---|
| MVT two groups (audit P3, default reduction order) | 4 | 14,666 | 14,666 (incumbent rank 0, 4 ties) | identical |
| `test_host_program_analysis` two kernels, no host program | 4 | 7,990 | 7,990 | identical |
| same, `_chained` host program (launch schedule) | 4 | 8,530 | 8,530 | identical |
| coupled chain (`test_joint_selection`, residency coupling cost) | 16 | 7,820 | 7,750 | producer and consumer both `resident` |
