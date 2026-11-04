<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Release status

Updated: 2026-11-03 (round 15: launch operand bindings validated (C-29b), the candidate contract enforced field by field (C-30), Gemini-II typed records migrated into the shared pipeline; committed as `9255d29`, exercised from the installed wheel on the simulators, the APU v1 board and the Gemini-II card; round 16: the reviewer's three contract gaps closed (C-30b) and the last typed route removed; round 17: the `APUv1Program` facade replaced by the shared front stages on the phase's source with retained store effects (X-02); round 18: indirect loads, fill initializers and unused arguments retained or represented correctly (X-03), whole launches reserved for verbatim programs (C-32); round 19: effects in program order (X-04) and Samsung as the first matcher backend declaring effects from the shared facts; round 20: carried destinations are reads on every backend (X-05), Samsung storage, packing and transfers tied to the emitted program (X-06), realization checked at the materialization boundary (C-33); round 21: distinct input/output images and exact vector residency complete X-06, AiM is the second matcher backend declaring effects from the shared facts; round 22: complete assignments materialize under the real launch schedule and host moves before acceptance (C-33b), the AiM check covers the batch mapping, the GPR readout of results (X-07) and host transfers bound by retained buffer identity (X-08)). Read `TASK.md` in the containing repository,
then this file, then `ISSUES.md` for the next action.

## Identity

- Release target: `experiments/allo`, branch `tenon`, publishing remote
  `fork` = `git@github.com:zzzDavid/allo.git` (`origin` is upstream Cornell
  Allo and must never be pushed to). Surveyed revision `8160d6d`
  (= `fork/tenon` at the start of round 1).
- Frozen paper: `zzzDavid/tenon-hpca27` `main` @ `30b6f49e`
  (`paper/SUBMITTED_VERSION.json`).
- Dirty state inherited at the start of round 1: six untracked
  `scripts/verifier_check_*.py` files in `experiments/allo` (left untouched,
  not committed); the containing repository has extensive uncommitted
  research history (design docs, agent memory, paper source changes) that
  is out of scope and is preserved.
- Commits so far: see `git log --oneline 8160d6d..tenon` (round 1: 33f53fb,
  7530405, ffb4a78; round 2: see below).
- Containing repository (not committed by this work, user decision pending):
  `experiments/simulators/setup/PIMSimulator/` now holds the exact overlay
  that builds the running `pim_driver` (our files in full, Samsung-licensed
  files as patches, README, apply.sh); `experiments/simulators/setup/apply.sh`
  calls it; the `dev/.../pimsim-conductor` tree mirrors the installed
  simulator sources. The installed `pim_driver` was rebuilt
  (sha256 `0ae6a3d9...`, previous binary kept as `pim_driver.bak-20261101`).

## Verdict

**NOT READY.** Gate judgments are in `REVIEW.md`. Blockers, in order:

1. **G9 (X-01)**: the shared compilation pipeline for all targets including
   APU v1 and Gemini-II is required by the user (2026-11-02) and in
   progress: slices 1-2 landed in round 12 (shared front stages
   `allo/spmw_pipeline.py`, semantic facts `allo/spmw_semantics.py`) and
   slice 3 in round 13 (the APU v1 vector plans and the Gemini-II launch
   plans are `autoschedule` candidates priced, ranked, materialized and run
   through `compile_for_target`; the planner seam is gone), the enforced
   candidate contract and the Gemini-II typed-record frontends in rounds
   14-15. At `9255d29` the installed wheel ran the shared pipeline on the
   simulators, the APU v1 board and the Gemini-II card (VALIDATION.md
   round 15). Round 16 closed the reviewer's three contract gaps (C-30b)
   and removed the typed route, but its `APUv1Program` migration was a
   facade (X-02); round 17 runs the shared front stages on the phase's
   source and derives the record's contract from retained store effects
   (guards, carried reads, preserved output regions); rounds 19-22 put
   Samsung and AiM on the enforced contract (effects from the shared facts,
   realization compared at the materialization boundary under the real
   launch schedule and host moves, C-33b, X-05 to X-08). Open: independent
   review of round 22, UPMEM and AttAcc effects declarations and facts
   consumption, the hybrid manifest's scalar ABI from retained IR, and the
   slice 5 traces of forced alternatives and cost perturbations on both
   hardware hosts. Map and status: `design/shared-compilation-path.md`.
2. **G2**: the independent reviewer disproved the round 7 PASS with two
   executed wrong-code probes at `e41ebbd` (C-19 casts erased, C-20 guarded
   copies), repaired in round 8 (plus C-21), and then found two more at
   `ee8f522` (C-22 float precision silently changed, C-23 output access
   maps lost), and two more at `620a230` (C-24 elementwise destination
   maps, C-25 accepted copies dropped), and two more at `f732b22` (C-25's
   fixed output key displaced by a destination named `out`, C-26 wide
   arithmetic bound to the operands'-type op). Rounds 9 to 11 repair them with
   stated contracts, an external-effect realization check before selection
   and value-checked regressions; awaiting independent review.
3. **G7/G8 evidence**: at `7bdb475` (round 9), `e60973f` (round 10) and
   `9255d29` (round 15) the installed artifact (wheel from the committed
   sdist, clean venv, run outside the checkout) was exercised on the three
   simulators, the APU v1 board and, from a wheel built on zhang-x4, the
   Gemini-II card (VALIDATION.md rounds 9, 10 and 15). Still open for G7:
   the Samsung simulator route is not buildable from the release repository
   without Samsung's permission for the five patches (P-04); the compatible
   build path for a user without the patches is the open question the
   reviewer named.
4. Open minor/boundary items: V1-16, C-18, S-13, P-17, APU v1 owner maps.

## Round 21 (2026-11-03): what was done

- Reviewer findings on `91401b6` (X-06 partial): `CandidateValueLayout.image`
  separates the input and output images of one buffer (the MAD addend that
  is also the result's copy sink keeps its swapped input image; the
  reviewer's 4113-element program compiles and executes exactly again);
  the Samsung vector residency check requires the exact realized storage
  and a transfer ending at it (vector-only SRF corruption refused).
- G9, second matcher backend: AiM declares every candidate's effects from
  the shared facts and the placement's handles and knobs
  (`spmw_aim.aim_candidate_effects`), is in `EFFECTS_REQUIRED_TARGETS` and
  `EFFECTS_DERIVERS`, and `AimCtx.check_realization` compares the lowered
  records with the selected declaration at materialization. Section r of
  `tests/pim/test_shared_semantics.py`.
- Evidence: VALIDATION.md round 21.

## Round 20 (2026-11-03): what was done

- Reviewer findings on `4fda263`: X-05 (carried destinations not semantic
  reads) fixed in the shared derivation and in the Samsung, APU v1 vector
  and Gemini-II declarations and realization checks; X-06 (Samsung storage
  and transfers not tied to realization) fixed with backing storage,
  transient registers, packing and conductor transfers declared in the
  emitter's terms and compared with the emitted program, plus a route
  vocabulary check on every target; C-33 (realization checked only after
  selection) fixed by running the check at the materialization boundary as
  a recorded feasibility rejection. Regressions in sections o and p of
  `tests/pim/test_shared_semantics.py`; the reviewer's probes behave.
- Evidence: VALIDATION.md round 20.

## Round 19 (2026-11-03): what was done

- Reviewer finding on `1a5ebf9` (X-04): fills and stores are retained in
  one traversal in program order; the dependency-bearing case is a
  regression and the reviewer's test passes.
- G9, first matcher backend: the pipeline derives and stamps shared facts
  for every matched dataflow work body (`trace.function_facts`); the
  Samsung enumerator declares every candidate's effects from the retained
  facts and the placement (`samsung_candidate_effects`); handle-only
  Samsung candidates are refused for programs with shared facts
  (`EFFECTS_REQUIRED_TARGETS`); corrupted declarations are refused before
  costing; `SamsungCtx.check_realization` compares the lowered CRF program
  with the selected declaration; the selected GEMV executes on the
  simulator (section n of `tests/pim/test_shared_semantics.py`).
- Evidence: VALIDATION.md round 19 (targeted; no full suites or hardware
  reruns this round, as directed).

## Round 18 (2026-11-03): what was done

- Reviewer findings on `a73a62d`: three supported-program regressions of
  the retained-effect contract (X-03: loads inside load addresses dropped,
  `linalg.fill` initializers not effects, the runtime's copy of an unused
  argument refused) and one validation hole (C-32: whole launches bypassed
  the tiling checks on planned programs). Fixed: `_retained_loads` recurses
  through addresses; fills are retained effects with whole extent, value
  and order; `CandidateTransfer.physical` represents the runtime's copy of
  a parameter the program neither reads nor writes; a whole launch on a
  non-verbatim program is a contract problem. Regressions in section m of
  `tests/pim/test_shared_semantics.py`; the reviewer's probes execute all
  five programs with the baseline's results.
- Evidence: VALIDATION.md round 18.

## Round 17 (2026-11-03): what was done

- Reviewer finding on `bba11f0` (X-02): the round 16 `APUv1Program`
  frontend built the lowering first and stood a body-less synthetic match
  for the function, so a guarded write declared its output write-only.
  Replaced: the phase's source goes through the shared front stages; the
  matcher retains every store's loop nest, guards, reads (through local
  temporaries and data-dependent indices), conversions and carried state
  (`StoreCoverage`), the semantics state them as `ProgramFacts.effects`
  (`StoreEffect`, returned buffers as parameters, `verbatim=True`); the
  enumerator builds the lowering from the same retained schedule; the
  candidate's contract is derived by `spmw_candidate.verbatim_values`
  (preserved output regions and carried reads are inputs), one
  whole-function launch advances every effect, and the realization check
  compares declared transfers and allocations with the runtime's actual
  movement. The reviewer's reproducer passes; section l of
  `tests/pim/test_shared_semantics.py` holds the regressions.
- Documentation corrected: slice 4 is described as done pending review of
  round 17, not as completed in round 16.
- Evidence: VALIDATION.md round 17.

## Round 16 (2026-11-03): what was done

- Reviewer findings on `9255d29`/`0f6c321` (C-30b): the shared validator
  derives an extent for every matched operation, requires the launches of
  one instance range to be an ordered partition of the reduction with
  multiplicity, and checks the accumulator chain against the source
  initialization; the realization checks compare instances, reduction and
  initializer (APU v1 vector plans) or the recomputed effects (record
  programs). The reviewer's three reproducers are rejection regressions.
- G9 slice 4 completed: `APUv1Program` is a frontend of the shared pipeline
  (`allo/pim/apu_v1_record_frontend.py`; `record_frontend.py` holds the
  helpers both record frontends share); `_PROGRAM_ROUTES`,
  `_compile_program`, `compile_apu_v1_program` and `APUv1ProgramCallable`
  are deleted; `compiled.native_vector_lowering`, `.device_source`,
  `.hybrid_manifest`, `.route`, `.lowered` are forwarded by the shared
  `CompiledCallable`. A layout request naming only contract-violating
  candidates is refused with their reasons.
- Evidence: VALIDATION.md round 16.

## Round 15 (2026-11-03): what was done

- Reviewer findings on `5a07639`: C-29b (launch operands dropped; now
  retained on `LaunchStep` and validated against each kernel's declared
  ABI before selection, for every target) and C-30 (candidate effects
  descriptive; now a typed contract with per-field validation against the
  facts, the target, the priced graph and the realized program) fixed with
  regressions (`tests/pim/test_shared_semantics.py` sections h and i).
- G9 slice 4, Gemini-II records: `allo/pim/apu_g2_record_frontend.py`
  normalizes `APUG2TypedProgram` and `APUG2ComposedContractionProgram` into
  the shared program contract; their realization is a candidate of the
  common pipeline; the typed callables are deleted; `_PROGRAM_ROUTES` holds
  `APUv1Program` only.
- Verdict item 1 and the next-action list updated: `APUv1Program` is the
  remaining typed route; card and installed-artifact traces pending.
- Evidence at the commit `9255d29`: full `tests/perf` (533 passed; one
  stale expectation about the Gemini-II backend table corrected, no code
  change) and `tests/pim` non-device suites; the sdist and wheel of the
  committed tree installed into a clean venv and run outside the checkout
  on the Samsung, AiM and UPMEM simulators with every reviewer probe of
  rounds 8 to 14; the APU v1 board (3 passed) and, from a wheel built on
  zhang-x4, the Gemini-II card (15 passed) through the shared pipeline
  (VALIDATION.md round 15). The PIM suite found one regression from the
  round 11 precision contract (C-31: the UPMEM linear-regression gradient
  refused); fixed in the narrowing proof with a regression test, suites
  rerun.

## Round 14 (2026-11-03): what was done

- Reviewer findings on `d2c7a65`: C-28 (supplied placements executed as
  stale artifacts; now rebound through `spmw_candidate.rebind_supplied_layouts`
  under the current program and policy, or refused) and C-29 (APU hooks
  dropped launch schedules and host moves; now refused before selection by
  `validate_orchestration`) fixed with executed regressions on both APUs.
- `allo/spmw_candidate.py`: `CandidateEffects`, the typed common candidate
  contract of the plan backends, validated per candidate before costing, on
  the selected placements and against the realized artifact.
- The APU v1 MLIR-text contraction parser and the hybrid route's dead
  vector-region planning are deleted; `tests/perf/test_apu_v1_tile_selection.py`
  reads the analysis from the shared facts.

## Round 13 (2026-11-02): what was done

- Reviewer findings on `7bb372e`: C-27 (helper calls replaced the entry
  ABI; refused at the shared boundary for every target, with executed
  regressions for a reordered and a repeated call) and G2-13
  (`build_apu_v2_target()` lacked the matcher view) fixed.
- G9 slice 3: `apu_v1_vector_program.apu_v1_vector_candidates` and
  `apu_g2_source_route.apu_g2_source_candidates` are the ``autoschedule``
  enumerators of the two APUs; `candidate_execution_graph` on the backend
  contexts prices their candidates; `select_candidates` turns ``layout=``
  into a selection restriction over a fully scored ranking;
  `KernelRanking.candidates` exposes the realized choice set; the APU v1
  context materializes `APUv1VectorProgram`, the new `allo/spmw_apu_g2.py`
  context materializes `APUG2SourceProgram`; `CompiledCallable` forwards
  the programs' documented introspection. Removed: `_SOURCE_PLANNERS`,
  `APUv1VectorCallable`, `compile_apu_v1_vector_workload`,
  `compile_apu_g2_source_workload`, the Gemini-II program's own selection.
- Tests: `tests/pim/test_shared_semantics.py` (boundary and slice 3
  sections); Gemini-II and APU v1 tests moved to the forwarded surface.

## Round 12 (2026-11-02): what was done

- Inherited uncommitted round 11 checkpoint inspected: its focused tests
  failed (Gemini-II `gesummv` refused by the integer precision contract,
  one Samsung check order, one stale key expectation); reconciled in a
  bounded way (integer `Op.intermediate`, `AXPBY` declares `i24`, check
  order, test expectation) and committed on its own.
- G9 slice 1: `allo/spmw_semantics.py` (`ProgramFacts`, `Contraction`,
  `Epilogue`, `BufferAccess`, `OutputSink`, `ElementType`) derived from the
  `MatchTrace` and retained IR types; the matcher retains source loop names
  (`extra["loop_names"]`) and traces `math.ctpop` as `popcount`.
- G9 slice 2: `allo/spmw_pipeline.py::normalize_source` is the one front
  end of `allo.compile` for every target (`matcher_view(target)` for the
  Gemini-II source-idiom view declared on the target); the Gemini-II
  contract and the APU v1 vector analysis consume the facts; the APU v1
  text parser and the Gemini-II module walk are out of the public route;
  planner hooks `_SOURCE_PLANNERS` replace the name branches in `compile()`
  as an explicit seam for slice 3.
- Found by the migration: V1-17 (guarded contraction planned unguarded by
  the old APU v1 analysis; closed), V1-18 (bitwise/XNOR products not
  matchable; recorded). New supported form on APU v1: local scalar
  accumulator copied to the output (executed on the functional backend).
- Tests: `tests/pim/test_shared_semantics.py` (10); expectations in
  `test_g6_semantics.py`, `test_apu_g2_source_route.py`,
  `test_compile_interface.py` moved to the shared error type.

## Round 11 (2026-11-02): what was done

- Destination identities (C-25 follow-up): outputs keyed by memref name in
  the Samsung runtime and `SourceBackendBinding.outputs`; required
  destinations derived from `external_effects` in `_copy_outputs`; output
  extent and element type checked in the Samsung elementwise classifier.
- Arithmetic precision (C-26): `arithmetic_precision_problem` enforced at
  match time; narrowing proofs retag the accepted operation; wide integer
  and float operations refused unless the op declares `intermediate`.
- STATUS line 6 corrected: the complete lint command still exits in the
  pylint stage (P-23).

## Round 10 (2026-11-02): what was done

- Elementwise destination maps (C-24): Samsung and AiM elementwise
  lowerings require the resolved identity map.
- Dropped copies (C-25): copy destination contract in the matcher;
  `external_effects` / `require_effects_realized` in `compile_for_target`;
  Samsung elementwise realizes identity copies (`sink_memrefs`,
  `SourceBackendBinding.outputs`, `_copy_outputs` requires every
  destination); other backends refuse such copies at compile time.
- Full `task_lint.sh` header check passes against `origin/main`.

## Round 9 (2026-11-02): what was done

- Float precision (C-22): `truncf` at a store is transparent only for a
  single operation on destination-typed leaves (double-rounding theorem);
  compound expressions bind only to an op declaring `intermediate="f32"`
  (AiM `AF` lookup table); `precision_notes` removed; the reviewer's
  program is rejected; numerical tests independent of the matcher.
- Output maps (C-23): resolved affine coordinates on stores, operands and
  sinks; `output_map_problem` identity contract enforced by Samsung, AiM,
  AttAcc, the layout derivation and the copy contract; Samsung's
  opaque-coordinate trust removed; the reviewer's reversed-row program is
  rejected; simulator value check with row-identifying inputs.
- License headers on all release documents; P-22 corrected.

## Round 8 (2026-11-02): what was done

- Reproduced the reviewer's two probes (`/tmp/tenon_reviewer_samsung_{cast,guard}.py`)
  and kept them as regressions in `tests/pim/test_match_coverage.py`.
- Matcher contracts (`allo/spmw_match_engine.py`): `cast_is_exact`,
  per-value constant proofs, store-side integer narrowing accepted only as
  the destination-width evaluation of fitting leaves or a proven range,
  store-side float narrowing accepted only over destination-type leaves and
  recorded in `MatchTrace.precision_notes`, every other conversion a visible
  unary term that fails coverage; `_copy_contract_problem` (guards, result
  loops, element type, truncation range proof via `_carried_interval`);
  `WLoad.signed` from the frontend's extension (C-21). UPMEM histogram names
  an unproven bin-index truncation; one Gemini-II expectation updated to the
  shared copy contract's message.
- Simulator recipes distributed in the release repository under
  `simulators/` with `tests/perf/test_simulator_recipes.py`; the Samsung
  patches are excluded by `.gitignore` and `MANIFEST.in` (P-04).
- Ledgers made consistent: ISSUES C-19/C-20/C-21/X-01; REVIEW gate table
  (G2 FAIL-then-repaired, G8 UNVERIFIED at this revision, G9 FAIL);
  CLAIMS evaluation rows updated for closed issues; CAPABILITIES
  cross-cutting status rewritten.

## Next concrete actions (in order)

0. Independent review of rounds 17-21 (the `APUv1Program` frontend, the
   retained effects, the Samsung and AiM slices and their realization
   contracts).
1. G9, matcher backends: UPMEM and AttAcc candidates declare
   `CandidateEffects` (coverage, value layouts on their handles, transfers,
   launches) and their enumerators consume `ProgramFacts` instead of
   re-reading `MatchedOp` fields; the shared validator then applies to
   every target.
2. G9, APU v1: the hybrid manifest's scalar ABI from the retained IR
   (`apu_v1_hybrid.discover_apu_v1_hybrid_manifest`) instead of MLIR text.
3. G9 slice 5: traces of forced alternatives and cost perturbations
   reaching generated code and launches on the APU v1 board and the
   Gemini-II card at one revision, run with `-rA`/`-s` so each case's
   printed provenance is kept in the logs; installed-wheel runs at that
   revision.
4. Independent review; then the deferred installation and lint work (P-04
   Samsung patches, P-23 pylint stage) and the open minor items.

## Round 6 (2026-11-02): what was done

- G6 probes (`tests/pim/test_g6_semantics.py`, `test_g6_decisions.py`,
  board and card additions) covering aliases, extreme integers, nonzero
  accumulators, executed schedule pairs, all-zero results, masks, strict
  versus reassociative pairs, cost perturbations traced to execution,
  target-parameter variation, renames with normalized decisions, op and
  constant changes at fixed shape, composed stages, archive removal and an
  sdist smoke test. Four silent-wrong-result defects found and fixed
  (C-17, S-14, V1-15 and the APU v1 zero-fill disagreement).
- Owner maps realized on UPMEM and AiM (CLAIMS row 16 now holds on three
  of four matcher backends).
- Multi-kernel Samsung programs execute on the simulator through the host
  schedule (S-15); APU v1 `backend="virtual"` is analytical only.

## Next concrete actions (in order)

1. Commit and push round 6 (done when this file is committed).
2. User decisions listed above.
3. C-18 evaluator diagnosis; M-14 confirmation; APU v1 owner maps.
4. Re-run every suite and both hardware hosts at the final revision and
   re-judge the gates for a READY verdict.

## Round 1 (2026-11-01): what was done

- Seven source audits with executed probes: `audit/*.md`.
- Repairs (all with tests): matcher store coverage and accumulator-init
  facts (C-02, C-14/S-08/M-09); `RunResult.kind` contract and missing-output
  errors (G2-07, M-08 part); Samsung zero/truncated readback (S-02); removal
  of the dead Gemini-II source route (G2-01); `APUv1Program` virtual and
  functional backends (V1-04); AiM structural axis roles and MAC-count
  invariant (M-01, M-10), `_eval_sym` no longer guesses unit 0 (C-07); UPMEM
  element-type enforcement and exact packing (U-01); portable C wide-integer
  soundness (U-02/A09).
- Ledger: `CLAIMS.md`, `ISSUES.md`, `REVIEW.md`, `VALIDATION.md`.
- Gemini-II host inspected (card available, FW V1.24.1); SSH IP-alias host
  key issue resolved without disabling checking.

## Next concrete actions (in order)

1. Commit and push round 1 (fast-forward to `fork/tenon`) after the final
   suite run recorded in `VALIDATION.md`.
2. F-01: resolve annotation names from the defining scope only
   (`allo/ir/utils.py`); run the upstream suites that exercise
   `customize`/dataflow to confirm no regression. F-02: make integer literal
   typing in the frontend fail on truncation.
3. S-01: carry the selected Samsung realization into the conductor (or
   document a proven lowering equivalence for the canonical route and make
   the placement levers that do not reach the executable explicit); S-04:
   commit the exact simulator overlay that builds the running `pim_driver`
   (license review first: Samsung notice, P-04).
4. C-11: port the minimum-movement algebra and bounded search per
   `audit/core.md` §3.3 into `allo/pim/layout_search.py` and wire the tie set
   into `rank_matcher_placements`; re-establish the 315/576/20,160 and
   121,968 checks as package tests.
5. C-09: explicit reassociation policy consulted by every reduction-splitting
   knob; C-01 target-neutral legality pass.
6. V1-01: replace regex recognizers with structural checks that bind the
   semantics (constants, fill values, stored locals) or reject; V1-02: carry
   the output or reject nonzero initial outputs on the vector route.
7. G8: build and run the Gemini-II typed/composed routes on zhang-x4 with a
   shared `flock` (plan in `audit/apu-g2.md` §5); re-run the APU v1 device
   rows after V1 repairs with the board lock.
8. G7: packaging (package data, extension build or documented bindings
   path, UPMEM compatible-build route), README/architecture rewrite,
   capability matrix.

## External blockers

- None hard at the moment. Gemini-II validation needs card time on
  zhang-x4 (25 users logged in; card idle at inspection).
