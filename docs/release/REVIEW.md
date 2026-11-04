<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Release review: architecture, constants, specialization, gate judgments

This file records the expert review of the release candidate against gates
G1-G8 of the task, the rejected shortcuts, and the remaining limitations. It
is updated at each round; judgments name the revision they apply to.

## Method

Seven read-only source audits (`audit/*.md`) covered every public route:
core target/workload/cost languages and matcher, Samsung HBM-PIM, UPMEM,
SK-Hynix AiM and AttAcc, GSI APU v1, GSI Gemini-II, packaging. Each audit
executed probes against `8160d6d` and labels claims as executed, read, or
suspected. `ISSUES.md` consolidates the findings; `CLAIMS.md` maps the frozen
paper to symbols. Repairs are reviewed by diff before commit; this file notes
what was accepted and why.

## Invariants at stage boundaries (current candidate)

1. **Frontend to matcher.** `customize()` produces retained MLIR with
   `spmw.*` scope attributes on work-item functions. `match_workload` now
   classifies every store of every function (`matched`, `copy`,
   `accumulator_init`, `local`, `unmatched`) and raises
   `UnsupportedWorkloadError` for any unmatched effect. Every accumulating
   match carries `accumulator_init` in {`zero`, `carried`}; anything else
   raises. Round 8 (C-19 to C-21): a conversion is transparent only when
   `cast_is_exact` holds or the matcher proves it for the value (constant,
   provable range, destination-width ring evaluation, destination-type
   float leaves); an output copy is accepted only under its producer's
   guards, once per produced element, into the producer's type. Invariant:
   a trace that reaches codegen implements every store of the source with
   the source's values (up to the recorded float precision contract), and
   the device may zero an accumulator only when the source does.
   Round 12: this stage runs for every target through
   `spmw_pipeline.normalize_source` and yields the shared facts
   `spmw_semantics.ProgramFacts`; a backend planner may refine them with
   target legality, never reinterpret them.
2. **Matcher to autoscheduler.** Axis roles (`axis_roles`: reduction,
   output, batch, matrix/vector operand, work-axis) are stamped structurally
   from index sets including the store index. Consumers must not re-derive
   them from memref extents or target counts.
3. **Autoscheduler to codegen.** Candidates pass the target-neutral
   legality pass (C-01) before ranking; materialization runs in cost order
   with rejections recorded (C-03); the reconciled whole program is
   re-priced and bounded joint evaluation is labelled (C-05). Invariant: the
   selected placement is legal, materialized, and priced from the graph of
   the program that is emitted. Since rounds 12-17 this holds for every
   target: the APU v1 vector plans, the Gemini-II launch plans and the
   typed records are candidates of the same pipeline (G9, X-01); since
   rounds 19-24 every backend declares `CandidateEffects` from the shared
   facts and is compared with them at materialization; since round 23 the
   host program's transfer availability is a shared stage; since round 28
   the `APUv1Program` manifest is read from the retained IR.
4. **Codegen to runtime.** `RunResult.kind` states what a result evidences
   (`device`, `simulator`, `simulator-timing-only`, `functional`, `virtual`).
   Executed kinds must carry `extra["outputs"]` for every gathered role;
   `CompiledCallable` raises otherwise and never writes caller arrays for
   non-executed kinds. Samsung readback treats all-zero as a value and
   truncation as an error. AiM lowering checks issued MACs against program
   MACs (`AimCoverageError`). UPMEM rejects element types its templates do
   not compute and never truncates on pack or gather. Portable C rejects
   operations on >64-bit integers whose result depends on bits above 63.
   Remaining breach: the Samsung REDUCE conductor regenerates the vendor
   microcode regardless of the selected placement (S-01).

## Accepted repairs (increment 1) and review notes

- Matcher coverage (`spmw_match_engine._FunctionCoverage`). Accepted. Notes:
  the `local` category uses a greatest-fixpoint over local stores and
  requires every reader of a local that reaches a matched op as an operand
  to be a constant or copy; the whole-buffer-fill rule for zero initializers
  requires each dimension indexed by its own `[0, dim)` loop with step 1.
  The host-exempt set for UPMEM host-placed groups couples the matcher to
  `spmw_upmem.classify_upmem_kernel` (import inside the function); acceptable
  for now, flagged for a cleaner "backend executes function verbatim"
  capability declaration. A hand-built trace without `coverage` cannot be
  checked; `compile_for_target` re-matches when a module is available and
  otherwise trusts only the per-match stamps.
- `RunResult.kind`. Accepted. Default `"unspecified"` keeps construction
  sites compiling; `validate()` rejects it on every public route. Twenty-two
  construction sites updated. Known residue: `apu_v1_vector_program` labels a
  Python-source execution as `functional` (it is neither the lowered program
  nor virtual; closed 2026-11-02: `backend="virtual"` there now evaluates the
  cost graph only and returns `kind="virtual"`, and `"functional"` is
  documented and labelled as a host NumPy model of the selected plan's
  arithmetic order, not compiled device code); UPMEM `extra["outputs"]` exposes every argument array;
  Samsung/AiM still tolerate a nonzero simulator exit code when a cycles
  line exists.
- Removal of the Gemini-II source route. Accepted: there was no emission and
  no execution; a finite `NotImplementedError` replaces a success-shaped
  result. `build_apu_v2_target` stays for the typed routes.
- `APUv1Program` backends. Accepted: `virtual` evaluates the retained graph;
  `functional` executes the retained portable C (the same C the ARC core
  runs for scalar programs); native-vector programs reject `functional` at
  compile time rather than silently running Python.
- AiM axis roles. Accepted. The `work_axis` fact is derived by comparing
  operand accesses across work-item functions; an undecidable case raises
  instead of guessing. Forms that used to compile but cannot be proven
  (split-K grids, uneven per-item extents, `y[row0+i] += ...`) now raise;
  this narrows accepted syntax to what the lowering can prove, which the
  task prefers over silent over-issue.
- UPMEM element types and exact packers. Accepted. The type table is a
  template fact (int32 MRAM words), not a tuning parameter.
- Portable C wide integers. Accepted with one deviation from the brief:
  `extsi`/`extui` from a wide type to a wider type is allowed because the
  low 64 bits are unchanged and any later observation of high bits is
  rejected at that op. The k-means paper kernel's oracle needs this.

## Accepted repairs (round 2) and review notes

- Defining-scope name resolution (`allo/ir/utils.py`). Shared Allo frontend
  file. Accepted because it implements Python scoping (module globals,
  lexically enclosing frames nearest first, closure cells) and the upstream
  suites that exercise `customize` and dataflow regions are unchanged; the
  remaining upstream failures pre-exist at `8160d6d` on this host.
- Section IV port. Accepted: `gf2.py` verbatim; the search, bound and masked
  mode carry their telemetry and the paper's reproduction tests. Two
  deliberate departures are documented in `design/layout-search.md`: a
  faster child generator in the scalable search (checked against brute force
  on 1,196 prefixes) and a fallback incumbent on budget exit (the snapshot
  could return none, which fails G4). Not yet a selection stage.
- Samsung selected-program execution. Accepted. Reviewer note: the old
  emitted stream was not a valid program (JUMP counts ignored lanes and GRF
  depth; roles inverted), which is why the conductor had regenerated the
  vendor microcode. The corrected MAC rule changes the default selection
  away from the vendor-equivalent program; the paper rows are pinned by an
  explicit selector rather than by weakening the model, and the default's
  cycles are recorded beside them.
- APU v1 recognizers. Accepted: a strict MLIR parser plus a symbolic
  executor replaces regex counting; every recipe documents the exact
  semantics it implements. Reviewer repair after the device run: the
  profile-freshness scheme assumed `flo` returns unread records only; it
  prints the whole retained log, so the check now diffs the before/after
  reads (`fresh_profile_records`).
- Packaging. Accepted; the wheel is host-specific (libstdc++) and the
  supported installation remains the source build, which INSTALL.md states.

## Accepted repairs (round 3) and review notes

- Reduction-order policy with `strict` as the default. Accepted: the paper
  says regrouping needs explicit permission. Consequence accepted knowingly:
  the default Samsung compile now selects the ordered TEMPORAL program
  (26,807 cycles at 4096x256) and AiM/APU v1 matcher routes refuse float
  reductions under strict with a finite error; every reassociating golden
  row passes `reduction_order="reassociate"` explicitly. Integer reductions
  may always split (wraparound makes grouping irrelevant).
- Target legality pass. Accepted. `srf` removed from the Samsung MAC
  operand set because neither the emitter nor the conductor can carry SRF
  data; the honest choice was to stop declaring it. Reviewer note: the
  work-grid check briefly rejected multi-DPU UPMEM programs until the
  outer-subset form was added; the accepted forms are now a documented
  table in `_check_work_grid`.
- Layout selection stage. Accepted with the finding it surfaced: the ported
  search runs per kernel group (an exact coordinate search handles problems
  beyond the 4-bit bounded domain, with a stated lemma), ties reach the
  ranking, but no backend can realize an iteration-to-owner map other than
  the kernel's own work-grid map; alternative ties are rejected with a
  stated reason. CLAIMS row 16 is therefore "partial", not "implemented".
- Gemini-II source route. Accepted: the five paper kernels as plain source
  reproduce the artifact's fingerprints and packed carriers byte for byte,
  and the plan candidates are ranked by a cost program fitted on card
  measurements with a stated validity domain. Reviewer repair after the
  card run: the typed runtime lacked the artifact hash record.
- UPMEM and AiM generality. Accepted; cross-file hooks (virtual-run extra,
  host-slice metrics, Samsung carried-accumulator gate) were applied by the
  reviewer after reading the proposed patches.

## Accepted repairs (round 4) and review notes

- Selection contract (C-03, C-05). Accepted: lazy materialization in cost
  order with recorded rejections selects the same program as eager
  materialization and keeps compile time flat; the joint evaluation is
  bounded by a documented budget and labelled. Reviewer note: the
  cross-kernel residency coupling that reconciliation handles is never
  produced by liveness on dataflow regions today, so the conflict test
  fakes it; this is recorded in ISSUES C-05.
- AttAcc analytical backend (M-04). Accepted: the paper's closed-form rules
  are re-implemented from the paper's own source without DCC code, the
  tiling space comes from the target geometry, and the 500-instance
  selection reproduces the paper's selector exactly. The inherited
  power-of-two rounding quirk of the reference space is kept and
  documented. What stays external is named with its dependency and
  license status; the e2e script's batch-4 discrepancy is reported, not
  hidden. Open question for the user: DCC has no license.
- APU v1 cost provenance and tiles (V1-05, V1-06). Accepted: the fitted
  coefficients are copied at full precision from the archived freeze with
  a provenance test; the released cost program reproduces the paper's
  error figures exactly; the tile is a ranked decision whose predicted
  order matches the archived measurements. The 1024x64x1024 default winner
  is a catalog variant never measured on the device, which the record
  states.
- Frontend literal typing (F-02). Accepted as a shared-file change after a
  module-level diff of 866 built modules showed two intended differences;
  one upstream test that depended on silent truncation was corrected.
- Simulator recipes (P-05, P-06, P-21). Accepted: each simulator rebuilt
  from committed overlay/patch plus a fresh upstream clone reproduces the
  pinned cycles; the Samsung patches still need Samsung's permission to
  publish (P-04).

## Accepted repairs (round 5) and review notes

- Samsung owner maps. Accepted: the realizable family is stated (owner
  bits from output-row bits; no reduction split), the identity map yields
  the byte-identical vendor plan, the conductor verifies a row-map digest,
  and the finding that tied layouts have identical cycles on Samsung (the
  lockstep program issues every slot) is recorded rather than hidden.
- Minor closures (U-06, G2-09, G2-10, G2-11, V1-08). Accepted; the reduced
  Gemini-II templates were rebuilt and run on the card (10 passed).
- EVIDENCE.md. Accepted as the G6 record; its gap list is the remaining
  G6 work.

## Accepted repairs (round 6) and review notes

- G6 probe campaign. Accepted; the probes used fresh random inputs and
  independent oracles and found four defects that returned wrong results
  without error (source alias, in-place Samsung elementwise, APU v1 aliased
  output, APU v1 zero-fill); each is fixed with the probe kept as the
  regression. One cost-evaluator anomaly (C-18) is recorded as an xfail
  rather than hidden.
- Owner maps on UPMEM and AiM. Accepted with the stated families; the
  expected cycle relations were written down before the simulator runs
  and held. Reviewer note: on skinny AiM GEMVs the routed reduction split
  can become the only realizable tie and displaces the unsplit program
  from the candidates; the simulator favored the split in the one case
  measured, but the enumerator should keep the program's own map as a
  candidate (recorded as a follow-up in the design note).
- Multi-kernel Samsung execution. Accepted: residency the conductor cannot
  keep across invocations is rejected before any simulator call instead of
  being silently re-staged.
- Board finding V1-16. The bit-equality board test written by the G6 agent
  failed: the Gemini-I device is not bit-equal to the functional route's
  FP16 models for either plan (a few ulps on about half of the outputs,
  all within FP16 tolerance of FP32). Treated as a documented limitation of
  the host model, not weakened: the test is a strict xfail with the
  mismatch counts printed, the strict policy's APU v1 guarantee is restated
  as accumulation order rather than bit-exactness, and the issue stays open
  until the GVML arithmetic is characterized.

## Round 8 (2026-11-02): independent reviewer's rejection of e41ebbd and the repairs

The independent reviewer reproduced two silent wrong-code cases through
`allo.compile` and the Samsung simulator at `e41ebbd`, which disproved the
round 7 judgment that G2 had passed and that no compiler-side blocker
remained. Both were mechanism defects in the shared matcher, not Samsung
spellings:

- Casts erased during matching (C-19). `_trace_value` and `_copy_source`
  treated every `_CAST_OPS` member as transparent. The repair makes
  exactness the only permission to look through a cast (`cast_is_exact`),
  proves constants per value, accepts store-side integer narrowing only as
  the destination-width evaluation of an expression whose leaves provably
  fit (truncation distributes over the ring ops; anything else needs a
  proven range), accepts store-side float narrowing only over leaves of the
  destination type and records it as a precision contract
  (`MatchTrace.precision_notes`), and otherwise keeps the conversion as a
  visible unary term that no target op matches. Review note: the legitimate
  frontend forms (`acc: fp16 = 0` as `sitofp(0)`, `i4*i4` evaluated in i8/i16
  and truncated at the i15 store, `i32 + i32` in i33, bf16 arithmetic in
  f32) all had to be proven rather than special-cased; the proofs are the
  interval analysis and the ring-homomorphism argument in the source
  docstrings. Reviewer-side finding during this round: the interval
  analysis first read every signless load as signed, which understates a
  `uint8` operand; fixed by carrying the extension kind on the load (C-21).
- Guarded or misplaced output copies (C-20). A copy of a matched result was
  accepted without comparing its guards or loop position to its producer.
  The repair is a copy contract: identical guard frames (then/else
  distinguished), exactly the producers' result loops (once per produced
  element, never inside the reduction) and the producers' element type (or
  a proven-exact truncation). Unsupported copies fail before any backend
  is consulted; the Samsung sink consumer is unchanged because it never
  sees them.

Scope of the review: the repair was checked on the four matcher backends
(Samsung, UPMEM, AiM, AttAcc share `match_workload`) and the Gemini-II source
route, which uses the same matcher; one UPMEM recognizer (histogram bin
index) and one Gemini-II test expectation changed because the copy contract
now names the real problem first. The APU v1 vector route does not run the
matcher and therefore does not run these checks; that is part of G9 (X-01).

Also in this round: the redistributable simulator recipes moved into the
release repository (`simulators/`, `tests/perf/test_simulator_recipes.py`);
the Samsung-derived patches stay out (P-04) and `apply.sh` refuses without
them, so the Samsung simulator route is documented as not buildable from
the release repository alone.

## Round 9 (2026-11-02): second independent rejection (ee8f522) and the repairs

The reviewer executed two further programs through `allo.compile`, the
Samsung simulator and the installed wheel and obtained wrong values:

- Float precision (C-22). The round 8 `_narrow_float` accepted any
  expression with destination-typed leaves and wrote a note. Review
  judgment: a note is neither a proof nor a permission. The repair keeps a
  store-side `truncf` transparent only where it is provably the device's
  operation (one operation on destination-typed leaves; the double-rounding
  theorem with 24 >= 2*11+2 and 24 >= 2*8+2; `min`/`max` round nothing) and
  otherwise leaves the cast visible. The only way to bind a compound
  expression under the cast is a target-description fact: the op declares
  `intermediate="f32"`, meaning the device evaluates its whole expression in
  that precision and rounds once. The AiM activation is a lookup table and
  declares it; no Samsung op does, so the reviewer's program is refused.
  The numerical basis is a test independent of the matcher (random pairs
  against rational arithmetic; the reviewer's two counterexamples verbatim).
- Output access maps (C-23). The Samsung classifier read coordinates as
  text and trusted the store row whenever the matrix row was an SSA name;
  the copy contract never related the copied element to the produced one.
  The repair resolves every coordinate to an affine form in the matcher
  (`affine_form`, `store_coords`, operand `coords`, `output_sink_coords`)
  and states one contract, `output_map_problem`: every non-reduction
  operand coordinate is literally a result coordinate. Samsung, AiM, AttAcc
  and the layout derivation require it; the trust branch is deleted; the
  copy contract compares the copied and produced elements. Reviewer note on
  scope: the contract is enforced where contraction lowerings assume the
  identity; elementwise families still rely on their own structural checks,
  and no backend realizes a permuted output (a permutation at readback
  would be a legitimate host transfer, not implemented).

Other corrections in this round: license headers on the release documents
(lint), and the P-22 ledger entry no longer claims development requirements
in the wheel metadata (the wheel carries exactly the ten runtime entries).

## Round 10 (2026-11-02): third independent rejection (620a230) and the repairs

The reviewer's simulator probes found two more silent defects, through the
checkout and the recorded installed wheel:

- Elementwise destination maps (C-24). The round 9 contract was enforced
  for contractions; the Samsung elementwise classifier still read
  affine-map operands. Repair: the classifier and AiM's elementwise
  derivation use the resolved coordinate forms and the shared identity
  contract. Review judgment: no reversal-specific check; shifted and
  broadcast destinations and shifted inputs fail the same rule.
- Disappearing copies (C-25). A copy of an array result was "covered" by
  the coverage analysis without anyone realizing it, and the runtime's
  output check only knew the one bound output. Repair in three parts: the
  copy contract now fixes the destination relation (same element), the
  coverage records externally visible writes and `compile_for_target`
  refuses any effect the backend does not declare it realizes, and the
  Samsung elementwise program realizes identity copies by writing every
  sink (declared through `realized_effects`, mapped through
  `SourceBackendBinding.outputs`, required by `_copy_outputs`). Reviewer
  note: the realization is a host-side duplicate of the gathered result,
  which is exactly the source's `y[i] = t[i]`; it is reported in
  `RunResult.extra["copies"]` and is not device work. Backends without a
  declaration fall back to the pre-existing implicit set (own result,
  sinks of a local producer), so a copy of an external array result on
  AiM, UPMEM or AttAcc is now a compile-time refusal instead of a dropped
  write.

Lint: the full `scripts/lint/task_lint.sh` header check against
`origin/main` now passes (headers added to 17 fork-authored files: the
architecture document, the historical `design/` notes, two source modules,
two tests, a verifier probe and both simulator `apply.sh`).

## Round 11 (2026-11-02): fourth independent rejection (f732b22) and the repairs

- Destination identities (C-25 follow-up). The round 10 realization keyed
  the producer's output by the fixed word `out`, in the binding and in the
  Samsung runtime, so a source buffer named `out` displaced it. Repair:
  outputs are keyed by the destinations' own memref names end to end, and
  the runtime's required-destination check is derived from the trace's
  external effects by retained identity, independent of the binding. No
  source name is reserved. The classifier also checks the output's extent
  and element type before execution.
- Arithmetic precision (C-26). Erasing exact widening casts is value-exact
  but loses *where the arithmetic happens*; the match then bound an f32 add
  on f16 operands to the FP16 ADD. Repair: the term carries its evaluation
  type, `arithmetic_precision_problem` states the contract, and a wide
  operation binds only to an op declaring that intermediate precision; the
  narrowing proofs retag accepted operations to the destination type so the
  valid `trunc` cases remain. Review judgment on the control: `y_f32[i] =
  a[i] * b[i]` multiplies in FP16 and widens the result exactly, so the
  device's FP16 multiply (inf for 256 * 256) is the source's value; it
  stays accepted and is executed as a regression.

## Round 12 (2026-11-02): gate G9 slices 1-2, shared front stages and semantic facts

User priority (2026-11-02): implement the shared compilation path before
the remaining arithmetic repairs. Reviewed increments:

- Inherited round 11 checkpoint (uncommitted at the start of the round;
  C-25 follow-up and C-26). Its focused tests did not pass as left: the
  precision contract refused the Gemini-II `gesummv` epilogue (integer
  intermediate), one Samsung check fired before the more specific
  coordinate message, and one test still expected the old `out` key.
  Accepted after a bounded reconciliation: `Op.intermediate` accepts an
  integer width and `AXPBY` declares `i24` (truthful: the epilogue runs in
  typed VL64 lanes and `_check_widths` proves the combined width); check
  order in `samsung_elementwise`; the test asserts the memref-named keys.
  The reviewer's round 10 probes are regressions in
  `tests/pim/test_match_coverage.py`. Committed separately (`c8a513c`).
- Shared front stages (`allo/spmw_pipeline.py`) and semantic facts
  (`allo/spmw_semantics.py`). Review questions and answers: the facts are
  derived from the matcher's structural analysis and the retained IR types
  (`_build_ir_value_refs` gives the same canonical identities the matcher
  used; `itypes`/`unsigned` give signedness), never from text; a fact that
  is not established is recorded (`roles_unresolved`, `dims` entries of
  `None`, element `None`) and both planners fail closed on it. The
  Gemini-II target declares its source-idiom view as a property
  (`spmw_match_view`); the driver does not branch on the name to choose a
  pattern set. The planner hooks (`_SOURCE_PLANNERS`) are a name-keyed
  registry: judged an explicit seam for slice 3, not unification (the
  design document says so; X-01 stays open).
- Gemini-II contract over the facts: the five paper kernels keep their
  contracts, packings and fingerprints (`test_apu_g2_source_route.py`);
  the domain rules (lanes, power-of-two reductions, widths) are unchanged
  target legality. Removed: the route's own module walk and access-map
  parser.
- APU v1 over the facts: the recipe catalog, realizer, cost model and
  runtime are unchanged; the vector plans' restrictions are now explicit
  legality over the facts, including two new explicit refusals (loop
  extents must equal buffer extents; mixed element types) and one new
  accepted form (local scalar accumulator copied to the output), executed
  on the functional backend against an independent FP16 source-order loop;
  the board GEMV test ran through the new path. Found and closed: V1-17
  (a guarded contraction planned unguarded by the text analysis). Domain
  change recorded honestly: V1-18 (bitwise/XNOR products no longer
  reachable until declared matchable).
- Tests whose expectations changed were each judged on semantics: the
  shared stage now raises `UnsupportedWorkloadError` where a route raised
  its own type for the same reason (alias, nonzero/partial initializer,
  narrowing copy, wide operand, extra effect); no rejection became an
  acceptance.

Gate G9 remains **FAIL**: the APU candidate plans are not common
candidates of the selection stage, typed records are not normalized, and
the stage traces on the Gemini-II card and the installed artifact are
pending.

## Round 13 (2026-11-02): independent review of 7bb372e, gate G9 slice 3

The independent reviewer accepted the shared front end and both
facts-consuming adapters as substantive progress and found two defects in
the semantic boundary plus the incompleteness of the candidate stage:

- C-27 (blocker, fixed). The facts were derived for the first function with
  matches, so a plain helper called with reordered arguments was planned as
  `R @ L` (APU v1 functional model 10 instead of 5; Gemini-II packed the
  same wrong assignment) and a helper called twice was planned once. The
  shared boundary now refuses plain helper calls for every target before
  matching, naming the call and its count; dataflow regions remain the
  interprocedural form; the facts are always the entry function's.
  Judgment: an explicit refusal is the sound boundary today; faithful
  inlining is an extension, not a requirement. Executed regressions on APU
  v1, Gemini-II and Samsung (reordered call and repeated call) check the
  rejection, not Python-source execution.
- G2-13 (major, fixed). `build_apu_v2_target()` lacked the matcher view;
  both builders now return it.
- Slice 3 (the reviewer's third item): both APUs' plans are candidates of
  the common candidate pipeline. Review questions and answers: the
  enumerators expose every feasible alternative (not a preselected winner)
  with its effects in the placement contract and a name-free physical
  decision in the schedule-search key; the cost stage prices each one
  through the backend's `candidate_execution_graph` (APU v1 from the
  materialized GVML inventory, Gemini-II from the issued recipes), so the
  selected program's graph is the estimate's graph; `layout=` restricts
  selection while every candidate stays scored; materialization instantiates
  the realized program from the selected candidate's payload, and the runner
  executes that program (board, host model, card). The planner-hook seam
  and the routes' own selection code are deleted. The movement stage reports
  `not-analyzed` for both APUs (a plain function retains no group shape);
  that is recorded on the ranking, not hidden. Remaining under X-01: typed
  records (slice 4), the matcher backends' consumption of the facts, and the
  card and installed-artifact traces (slice 5).

Gate G9 remains **FAIL** until slices 4 and 5 are complete and reviewed.

## Round 14 (2026-11-03): independent review of d2c7a65, candidate boundary

The reviewer accepted slice 3's shared enumeration, ranking and
materialization and found that the new boundary admitted wrong programs:

- C-28 (blocker, fixed). A supplied placement was executed as an artifact
  (stale payload, stale contract, policy bypass). Judgment: a placement is
  a decision; the common boundary now rebinds it to the current program's
  candidate under the current policy or refuses it
  (`spmw_candidate.rebind_supplied_layouts`), for every target; the
  reviewer's transposed-access and strict-reuse probes are regressions on
  both APUs, checking the planned program, the packing and the functional
  run rather than Python-source execution. Handle-only placements of the
  matcher backends keep their legality path and are refused when their mode
  was policy-pruned (Samsung's own emit-time check remains).
- C-29 (blocker, fixed by explicit refusal). The APU hooks dropped launch
  schedules and host moves. Judgment: until repeated launches and
  interleaved transfers are implemented for these backends, the capability
  is declared and checked before selection (`validate_orchestration`); the
  graph and the runtime never see an orchestration they cannot honor. The
  single-launch host program and `host_moves=[]` remain valid.
- Candidate effects as an enforced contract (the reviewer's third item).
  `CandidateEffects` is one typed record for every plan backend; the driver
  validates it per candidate before costing (a violating candidate is
  recorded as a contract rejection, never priced), on the selected
  placements, and against the realized artifact (`check_realization`: APU
  v1 realization ABI, Gemini-II contract outputs and launch count). A
  regression injects a wrong coverage and shows the rejection. The matcher
  backends declare no effects yet (recorded under X-01).
- The APU v1 MLIR-text contraction parser is deleted together with the
  hybrid route's dead vector-region planning (the typed route already
  refused vector regions; required-vector phases are proven by the native
  recipes). The hybrid manifest still reads the scalar ABI from MLIR text
  (recorded).

Gate G9 remains **FAIL**: typed records still dispatch through
`_PROGRAM_ROUTES`; card and installed-artifact traces pending.

## Round 15 (2026-11-03): independent review of 5a07639; launch bindings, enforced effects, typed records

The reviewer found the round 14 boundaries incomplete in two ways and
asked for the typed-record migration:

- C-29b (fixed). Operand bindings of `allo.launch` were dropped; permuted
  and empty operand lists executed. Judgment: a launch's operands are part
  of the orchestration contract; they are retained on the schedule and
  checked against the kernel's declared ABI (dataflow kernels through their
  retained ABI identities, plain functions through their parameters) before
  selection, for every target; permutations are refused rather than
  lowered. The APU rule additionally checks that launches cover the kernel
  groups once each in program order.
- C-30 (fixed). The candidate effects were descriptive. Judgment: the
  contract is now typed (launches carry operations, instance and reduction
  ranges) and every field is checked against the semantic facts and the
  target: coverage equality, layouts over program loops on declared
  storage with iteration coverage, transfer completeness, ordered launches
  tiling instances and reductions, ABI equality, graph consistency, and a
  realization check that compares layouts, transfers and launches with the
  materialized program. The reviewer's fault injection, re-expressed in the
  typed fields, is a six-way regression on both APUs. The matcher backends
  still declare no effects (recorded).
- Slice 4, Gemini-II records (done). `APUG2TypedProgram` and
  `APUG2ComposedContractionProgram` are frontends: their semantics are
  stated as synthetic matches carrying the structural facts the matcher
  stamps (loops, roles, accesses, accumulator initialization, policy) and
  the derived `ProgramFacts`; the `apu_v2` enumerator produces the record's
  realization as one candidate; legality (the view declares the typed
  operations), the candidate contract, pricing through the record's recipe
  graph, ranking, materialization (`APUG2RecordProgram`) and the shared
  artifact follow. The separate callables are deleted. Judgment on the
  accumulate epilogue: modeled as a zero-initialized local dot plus an ADD
  with the seed input, which is what the device computes. `APUv1Program`
  remains on the typed route (its scalar ARC programs and native recipes
  are whole-function lowerings the matcher does not cover); that is the
  remaining slice 4 item and is stated as such.

Evidence at the commit `9255d29` (VALIDATION.md round 15): the full
`tests/perf` and non-device `tests/pim` suites; the installed wheel on the
three simulators with the reviewer probes of rounds 8 to 14; the APU v1
board and the Gemini-II card from installed wheels (3 and 15 passed). One
perf test still asserted the pre-round-13 backend table and was corrected
to the current contract (no code change). The PIM suite exposed one
regression from round 11 (C-31): the paper's UPMEM linear-regression
gradient was refused because the integer narrowing proof had no rule for
a right shift whose operand the interval analysis can no longer bound.
Judgment: the operation's own result width bounds it, so the rule is
sound and narrow; it is fixed with a regression test and the suites rerun
(VALIDATION.md round 15). The round 10 suite had passed this row on an
interval the IR does not hold, so the earlier PASS was not evidence for
it.

Gate G9 remains **FAIL** until `APUv1Program` is migrated, the matcher
backends declare their effects, and the slice 5 traces (forced alternatives
and cost perturbations on both hardware hosts) exist.

## Round 16 (2026-11-03): independent review of 9255d29 and 0f6c321; contract gaps, APUv1Program

The reviewer accepted the round 15 progress (records in the shared
pipeline, launch bindings checked, no manufactured pass in the changed
expectations, frozen paper intact) and found C-30 partially repaired, with
three accepted-invalid contracts (C-30b):

- Repeated reduction chunks (fixed). `_tiles` worked on a set of ranges,
  so a plan summing one chunk twice passed when its effects described it
  accurately; the runtime oracle caught the wrong value after execution.
  Judgment: the contract must establish the ordered accumulation chain
  before selection. Per operation and instance range, the launches'
  reduction ranges now have to be an ordered partition with multiplicity,
  and the accumulator chain has to start as the source initializes the
  accumulator and continue from the previous launch.
- Accumulator modes unchecked (fixed). A carried contraction declared as
  zero-started was accepted, and the APU v1 realization check compared
  launch count, kind and plan name only. Judgment: initialization and
  continuation modes are checked against the semantic facts in the shared
  validator and the realization checks compare instances, reduction range
  and initializer (APU v1 vector plans) or the complete recomputed effects
  (record programs on both APUs).
- Extents of non-contraction operations (fixed). The extent check indexed
  a filtered contraction list by original match indices and skipped
  elementwise operations, so a 64-element typed add accepted one declared
  instance. Judgment: every matched operation has an extent (contraction
  trips, epilogue loop trips, a record's declared result instances) or the
  candidate is refused.

The reviewer's three reproducers are rejection regressions
(`test_shared_semantics.py` section k); their original file now fails at
compile on all three.

Slice 4 completed as directed: `APUv1Program` normalizes into the shared
contracts (`allo/pim/apu_v1_record_frontend.py`) and the early return in
`allo/compiler.py` is gone with the typed route's callable. Judgment on
the semantic statement of a whole-function program: the matcher cannot
recognize it, so its contract is its external effects (the scalar ABI's
reads and writes, one launch over the result buffer's elements), stated on
non-matchable target operations declared on the ARC unit over L4; the
shared validator checks the candidate against exactly that program, and
the realization check recomputes the effects from the lowered program.
Native recipes keep their structural proof as the lowering; their
`backend="functional"` refusal is made at materialization, before an
artifact exists.

Gate G9 remains **FAIL** until the matcher backends declare their effects
and consume the facts, the hybrid manifest reads the scalar ABI from the
retained IR, and the slice 5 traces (forced alternatives and cost
perturbations on both hardware hosts, with per-case provenance kept in the
logs) exist.

## Round 17 (2026-11-03): independent review of bba11f0; the APUv1Program facade

The reviewer accepted the C-30b repairs and the removal of the typed
dispatcher, and rejected the `APUv1Program` migration as a facade around
an independent lowering (X-02): the lowering was constructed before any
shared semantics, the function's body was replaced by one synthetic
body-less match, and the contract came from the scalar ABI's modes, so a
guarded write declared its output as write-only while its initial
contents reached the result; the shared validators never saw the
function. Judgment: an `APUv1Phase` contains source, and its semantics
are the shared front stages' business. Round 17 runs them on the phase
(one `customize`, the matcher, the coverage check with the function marked
verbatim so that its unmatched stores are implemented, not dropped, and
the semantic analysis), retains every store's facts (loop nest, guards,
reads through local temporaries and data-dependent indices, conversions,
carried state) as `ProgramFacts.effects`, builds the lowering from the
same retained schedule, derives the candidate's contract from those facts
in the shared validator's own terms (an argument is an input when read,
read at the element written, or not overwritten whole and
unconditionally), and compares the declared transfers and allocations
with what the runtime actually moves and allocates. The reviewer's
reproducer passes; guarded, shifted, carried and whole writes are
regressions, and hiding a read, a transfer or the scratch allocation is
refused. The matched operations of a verbatim function stay in the facts
(their contracts were checked by the matcher) and are not placed against
op patterns the verbatim program never uses.

The lint script's Black stage (Black 24.8.0) had one unformatted test
file from round 16; it is formatted in this round.

Gate G9 remains **FAIL** until this round is reviewed, the matcher
backends declare their effects and consume the facts, the hybrid manifest
reads the scalar ABI from the retained IR, and the slice 5 traces exist.

## Round 18 (2026-11-03): independent review of a73a62d; retained-effect regressions

The reviewer accepted the round 17 direction (the matcher and the semantic
analysis run before lowering, the lowering uses the retained schedule) and
found three supported-program regressions and one validation hole:

- Indirect loads (X-03, fixed). Loads inside a load's address were not
  retained, so `src[index[i]]` lost its dependency on `index` and the
  program was refused. Judgment: dependencies follow addresses; the
  retained loads now recurse through values, guards and destination
  indices, and the regression checks reads, ABI, transfers and results.
- Fill initializers (X-03, fixed). A declaration's `linalg.fill` was not an
  effect, so a returned fill-only buffer "wrote nothing" and an
  initialized, partially overwritten return omitted the initialization of
  its untouched elements. Judgment: the fill is an observable effect of a
  returned buffer; it is retained with its destination, value, whole
  extent and program order, and the realized lowering reproduces it. The
  design note's claim that local declarations made fills irrelevant is
  withdrawn.
- Unused arguments (X-03, fixed). The runtime copies every source argument;
  declaring that copy as a transfer of a non-program buffer was refused.
  Judgment: physical movement of a declared parameter is represented
  separately from semantic reads (`CandidateTransfer.physical`), checked as
  a parameter that is neither read nor written, and compared with the
  runtime; an unused parameter is never marked as a semantic input.
- Whole launches (C-32, fixed). A whole-function launch bypassed the tiling
  checks on planned programs; it is now refused before pricing for any
  program not executed verbatim.

Gate G9 remains **FAIL** until the complete migration is reviewed, the
matcher backends declare their effects and consume the facts, the hybrid
manifest reads the scalar ABI from the retained IR, and the slice 5 traces
exist.

## Round 19 (2026-11-03): independent review of 1a5ebf9; effect order, first matcher backend

The reviewer accepted the round 18 repairs and found one remaining
shared-contract defect: retained effects did not preserve program order
(every fill before every store), so a fill reading a value an earlier
store defined was placed before its defining write (X-04). Judgment: the
effect set is the function's program-ordered effect sequence; fills and
stores are now retained in one traversal by site index, and the
dependency-bearing case is a regression.

As directed, the increment also started the matcher-backend migration
with an ordinary-source vertical slice on Samsung HBM-PIM: the pipeline
derives and stamps shared facts for every matched dataflow work body
(their arguments named by the buffers the matches bind them to); the
Samsung enumerator declares every candidate's effects from those facts
and the placement; a handle-only Samsung candidate is a contract problem
for any program with shared facts; the shared validator refuses corrupted
declarations (coverage, reduction partition, layout axes, missing effects)
before costing; the context compares the lowered CRF program with the
selected declaration; the selected GEMV program executes on the simulator
under that contract. Judgment on scope: the Samsung classification code
(`samsung_contraction`, `samsung_elementwise`) still reads the matches'
structural fields for its own domain checks; the candidate contract is
derived from the facts, and the realization is checked against it, which
is the invariant the gate asks for. AiM, UPMEM and AttAcc remain
handle-only (X-01).

Gate G9 remains **FAIL** until the remaining matcher backends are
migrated and reviewed, the hybrid manifest reads the scalar ABI from the
retained IR, and the slice 5 traces exist.

## Round 20 (2026-11-03): independent review of 4fda263; carried reads, Samsung realization, search-time checks

The reviewer accepted the ordered traversal and the Samsung slice's reach
into shared validation and lowering, and found the Samsung contract
incomplete in three ways:

- Carried destinations (X-05, fixed). A destination whose incoming
  contents seed the reduction was not a semantic read on any backend, so
  removing its ingress passed. Judgment: the read is a property of the
  program, so it belongs to the shared derivation; every backend's
  declaration now includes it, its realization checks expect it, and
  removal is refused before costing and at realization.
- Storage and transfers (X-06, fixed). Declarations named an existing
  storage the program did not use, routes were not checked, and the MAD
  addend's fiber packing was absent. Judgment: the contract states backing
  storage, transient registers, packing and the conductor's transfers in
  the emitter's own terms and the realization check compares them with
  the emitted program; routes are checked against declared storage, moves,
  host steps and registered steps on every target.
- Search-time realization (C-33, fixed). The realization check now runs at
  the materialization boundary of the ranking; a mismatch is a recorded
  feasibility rejection and another candidate is selected.

AiM, UPMEM and AttAcc remain handle-only (X-01). Gate G9 remains **FAIL**.

## Round 21 (2026-11-03): independent review of 91401b6; X-06 completion, AiM

The reviewer accepted the carried-read repair and the search-time
realization rejection and found X-06 partial: an operand that is also a
copy sink lost its swapped input packing to the output's (a supported
program regressed), and the vector residency check accepted any storage
but the banks under a host broadcast. Judgment: one semantic buffer can
have distinct input and output images, so the contract names the image
(`CandidateValueLayout.image`) and the realization compares each side with
its image; the vector check requires the exact realized storage and a
transfer ending at it. The reviewer's program executes exactly again and a
vector-only SRF corruption is refused during the search.

As directed, AiM is the second matcher backend on the shared contract:
effects derived from the facts and the placement's handles and knobs,
required and re-declared through the registry, and compared at
materialization with the lowered records (instances, reduction, bank
scope, reduction partitions and their host merge, vector and matrix
residency, elementwise elements and home, host transfer names). The AiM
GEMV's declaration, corrupted declarations refused before costing, a
realization mismatch rejected in the search with a fallback, and the
timing-only simulator run are regressions. UPMEM and AttAcc remain
handle-only (X-01).

Gate G9 remains **FAIL**.

## Round 22 (2026-11-03): independent review of cc2354f; scheduled materialization, AiM contract depth

The reviewer found three defects in the round 21 work and a count error.
(1) C-33 was not closed for scheduled programs: the materialization probe
emitted every candidate without the launch schedule and the host moves, so
the realization compared at materialization was a kernel-only program and a
mismatch under the host program surfaced only after selection (the
reviewer's `probe_final_only.py` printed FAILURE). Judgment: the ranking
keeps the per-kernel probes as schedule-free preliminary filters and accepts
a complete assignment, on the joint path and on the over-budget heuristic
path, only after the whole program materializes under the real launch
schedule and the resolved host moves (C-33b); the probe prints FALLBACK.
(2) The AiM realization check ignored the declared batch mapping and the
results' residency: a wrong-but-valid `channels` mapping and results
declared in `gb` passed. Judgment: the declared request must be the
record's (an unspecified one is the lowering's `auto`), an explicit one the
realized mapping of every launch, the realized one the mapping the
derivation laid the host transfers out for; results live in `gpr` and
leave through it (X-07). (3) The host-transfer check matched the record's
label text against function names and covered one record kind. Judgment:
the derivation stamps every host record with the retained identities of the
buffers it moves, their slices and the group whose layout it follows, and
the check binds each record to the declared transfer of the work-body value
with the same identity, comparing direction, slice extent and device
endpoint; under a host program every declared input must be host-written,
resident or produced on the device (X-08). The round 21 shared-semantics
count is corrected to 68.

Gate G9 remains **FAIL**: UPMEM and AttAcc effects, the hybrid scalar ABI
from retained IR, the installed-artifact and both-hardware-host traces are
open.

## Round 23 (2026-11-03): independent review of 29b477b; X-08 reopened

The reviewer accepted the scheduled materialization (the scheduled-failure
probe also fell back with `joint_budget=0`) and found X-08 not closed, with
three probes outside the checkout (`reviewer-r22/`). (1) Input availability
ignored launch order and access coverage: a 128-element multiply compiled
after a scatter of `A[:64]`, and a first launch before the only scatter of
`B` compiled. Judgment: availability is a property of the host program and
the candidate's declared transfers, not of one backend, so it is the shared
`spmw_plan.check_transfer_availability`, walked in schedule order for every
launch against the regions the retained accesses read, with host slices,
explicit residency and earlier launches' written regions as the only
sources (X-09). Programs that relied on an implicit KV cache now declare it
resident. (2) Sliced transfers lost their coordinates: two half scatters of
a 524288-element vector emitted identical streams over rows 0 and 1 while
the kernel read rows 0 to 3. Judgment: a transfer is a slice of the kernel's
vector; the record carries the slice frame and the lowering issues the
bursts of the whole vector's layout that the slice touches; unsupported
slice forms are refused (X-10). (3) The endpoint check compared the storage
name only and skipped unknown owners. Judgment: every host record is
compared with the launched kernel record's addressing (row, replicas,
channels, stride, slot per coalesced constituent, slice frame; matrix rows
for row writes and broadcasts) and must belong to a launched group (X-11).
The round 22 "fixed" claim on X-08 is corrected in ISSUES.md.

Gate G9 remains **FAIL**.

## Round 24 (2026-11-03): round 23 verified; UPMEM and AttAcc on the shared contract

Round 23 had ended without verification (the launcher lost its shell when
/tmp filled). This round first ran its last edits: the reviewer's three
probes (`reviewer-r22/probe_transfer_contract.py`, `probe_large_slices.py`)
now reject the partial and the late scatter in schedule order, emit distinct
command streams for the two half scatters of the 524288-element vector
(rows 0 to 3, as the multiply reads), and refuse every physical mutation
(row, bank slot, channels, unknown owner, and the pre-emission row
redirection, which now fails selection and the final compile). Two of its
tests needed correction, not the mechanism: the "extent" corruption is now
caught by the slice-frame message rather than the round 22 wording, and the
split-scatter test compared the halves with a whole-program record that
coalesces A and B into two bank slots (A's own slot's bursts are the
comparison).

Then the last two matcher backends joined the contract. Judgments:

- *UPMEM effects are derived from the facts, the physical decision names the
  realization.* The family classifier (`classify_upmem_kernel`) remains the
  structural recognition that chooses the template; it is consulted by the
  deriver only to name the operand the residency decision applies to. Every
  value's backing storage is the DPU MRAM heap image the runtime packs
  (`host_pack`) and decodes (`host_readback`); WRAM is a transient
  allocation; the packing carries the decision's data layout and the
  resident operand's residency so `check_realization` can compare them with
  `decision_fields`. Launch extents are compared with the family template's
  geometry per work item (`upmem_family_extents`), the accumulator chain
  with the kernel's `accumulator_init` and the plan's `carried_output`, and
  the carried role is the output except the selection count (the host
  compaction step carries it, the template never loads it).
- *A host partition is declared, not hidden.* The k-means update group runs
  its whole function on the host from its retained IR (the paper's
  partition). The contract gains `CandidateEffects.host_function`: a
  whole-function launch over the function's retained store effects (the
  same check a verbatim device program gets) with values in `host_dram`.
  A device backend cannot use it to hide tiling: the realization check
  requires the group's placement to be the host, no template plan, and the
  compiled host function at the final compile.
- *AttAcc's inventory is one program with phases.* One launch per matched
  operation, in match order, names the command phase and the tiling; the
  realization check compares the emitted inventory's kernel kind, the MAC
  phases' extents (rows x outputs over the reduction, then rows x d_head
  over the length for attention), zero-started accumulators, the tiling and
  the fixed datapath homes.
- *Two shared rules were incomplete and are now stated once.* A destination
  an operation accumulates into after a proven zero initializer is not an
  input (the operand view had exempted only the pattern's loop-carried
  parameter), and buffers read only through computed indices or guards are
  inputs (`indirect_reads`, from the retained store effects; the per-body
  facts name such arguments from the coverage's loads). Both came out of
  faithful UPMEM declarations the previous rule would have forced to lie
  (X-12, X-13); they apply to every backend.

- *The availability check found a misdeclared AiM ingress.* The CENT
  attention and projection kernels never broadcast their MAC vector; the
  lowering writes the GB with `WR_GB` from a host-loaded GPR. The round 21
  declaration `("host_broadcast", "gb")` had described a host step that
  does not exist, which only the schedule-ordered availability check could
  expose. The declaration now names the realized route
  (`host_gpr_preload`, `gpr`, `WR_GB`, `gb`), the same form round 23 gave
  EWADD's GPR operands (X-14). This is the intended effect of the shared
  stage: a declaration that is not what the backend does cannot survive
  the host program.

`EFFECTS_REQUIRED_TARGETS` lists every released backend; a handle-only
candidate is a contract problem everywhere. Open for G9: the hybrid
manifest's scalar ABI from retained IR, the slice 5 hardware traces, and
independent review of rounds 23-24. Gate G9 remains **FAIL**.

## Round 25 (2026-11-04): independent review of 9d13666; AttAcc orchestration and datapath fidelity, availability shapes

The reviewer accepted the UPMEM/AttAcc migration as real progress and found
four G9 gaps, all reproduced by `reviewer-r24/probe_attacc_contract.py`.
Judgments: (1) an analytical backend is still bound by the retained
schedule: AttAcc now emits one inventory per launch in schedule order, so
the artifact, the realization check and the priced graph describe the same
program (X-15); (2) a realization check that accepts "either home" checks
nothing: the AttAcc check binds every external MAC value to the datapath
role its facts give it, with the exact route, and checks the phase order and
the softmax scratch (X-16); (3) the full allocation shape is a fact of the
normalized program, not of a factory's annotations: the availability stage
reads it from the entry facts by retained identity before residency (X-17);
(4) the flat-slice conversion was a plain bug, fixed and pinned by a test
whose rejected case names the missing row (X-18). The round 23 one-dimensional
transfer probes therefore did not establish complete X-09 closure; with X-17
and X-18 the stage is checked on multi-dimensional buffers too.

Gate G9 remains **FAIL** (hybrid scalar ABI, slice 5 hardware traces);
G10 is open.

### G10 start (round 25)

The inventory (`paper_reproduction/inventory.csv`, 82 rows) was measured
first where the shared pipeline is usable locally: Samsung's three GEMVs
(default selection faster than the paper, outputs checked), UPMEM's sixteen
kernels (exact, NumPy-exact outputs), ten of fourteen AiM CENT rows (exact,
timing-only) and the 500 AttAcc instances (shared selection equals the
paper's selection on all 500 through `allo.compile`). The four
RoPE-containing AiM rows did not compile: the round 23 frame check exposed
that the paper's program multiplied a strided half of the scattered RoPE
data (X-19). Judgment: the paper's digits for those rows are not a target to
force; the kernel is corrected to the work CENT performs, the measurement is
reported against both the paper and CENT, and the golden rows keep the paper
values while naming the deviation. The direct selector call in the 500
instance golden test does not establish shared selection; the measured
route is `benchmarks/attacc/scripts/verify_paper_selection.py`.

## Round 26 (2026-11-04): independent review of da041af; flat-range encoding, attention dataflow

The reviewer accepted the G9 progress and the first G10 measurements and
found two contract defects (`reviewer-r25/`). (1) The round 25 flat-slice
repair had kept an ambiguous encoding (a flat range as a Cartesian box
ending in unit dimensions), so a genuine Cartesian scatter of four scattered
elements was taken as the flat prefix supplying a whole matrix. Judgment: a
region is either a Cartesian box or a flat range, by type, never by shape
(`spmw_plan.FlatRange`); every consumer was moved to the typed form and
X-18 was reopened rather than left closed. (2) AttAcc recognized fused
attention from operation names and extents only: a softmax over another
buffer and a normalization across heads both lowered to the same inventory.
Judgment: the fused template is selected only after a structural proof from
the shared facts (value identities along score to exp to sum to div to
context, own-element access relations, the sum's initializer scope under
exactly the row loops, the division in the same row iteration); the proof
reads the retained `accumulator_init` store effect, which is why the
initializer scope is a shared fact and not a backend parser (X-20).

G10 reporting corrected as the reviewer asked: the 500-instance selector
equivalence is a diagnostic row; the published 398/500 timing-quality result
stays UNVERIFIED without Ramulator2-AttAcc timings; the four corrected AiM
rows are an OPEN 32-cycle regression against the paper, not parity.

Gates G9 and G10 remain **FAIL**.

## Round 27 (2026-11-04): independent review of f3ef033; X-20 reopened

The reviewer accepted X-18 and the G10 reporting corrections and reopened
X-20 with two demonstrated defects: the division's operands were compared
as a set (the reversed division selected the fused inventory), and axis
correspondence used source loop labels (alpha-renamed loops were refused).
The round 26 ledger's "no source-name checks" was wrong about that
correspondence and is corrected. Judgment: a structural proof names axes
by buffer dimension and loops by retained identity; operand roles are
ordered facts. The proof now derives the length axis from the dimension
of the score buffer the context MAC reduces over, proves own-element
access relations per softmax op, checks numerator and denominator roles,
and compares initializer and consumer scope by retained loop variables.
The AttAcc declaration gained the local score buffer's layout so the
shared iteration-coverage rule does not require the softmax loops to share
the MAC's spelling. Regressions cover the reversed division and three
renamings alongside the retained disconnected and global-normalization
rejections.

Gates G9 and G10 remain **FAIL**.

## Round 28 (2026-11-04): independent review of 4056f67 accepted; the APUv1Program manifest from the retained IR

The reviewer accepted the X-20 repair and asked for the hybrid scalar-ABI
migration as production code. `discover_apu_v1_hybrid_manifest` now walks
the retained module through the IR bindings (function symbols and formal
types, named allocations and their `linalg.fill` initializers, `func.call`
operations in program order, each callee's loads and stores on its
formals) and the regex analysis of the printed module is deleted.
Judgment: the manifest's facts (ABI identities, call order, access
directions, initialization, intermediate lifetimes and barriers) are
properties of the retained IR, so they are read from it; the dead
vector-region branches (analysis, conversions) were removed with the text
parser instead of being carried along. The manifests of returned-local,
nested-call, in/out and unsigned-local programs are byte-identical to the
text-derived ones, and a renamed neighbour yields the same structure.

Gates G9 and G10 remain **FAIL**; the gate table above was refreshed.

## Round 29 (2026-11-04): independent review of 90c6cbd accepted; manifest fidelity (X-21)

The reviewer accepted the parser replacement and found that the manifest
still called every fill a zero initialization and joined a returned local
to its result name by spelling. Judgment: both are facts of the retained
IR (the fill's constant; the operands of `func.return`), so they are read
from it, and the public-route regression uses different allocation and
result names with a nonzero initializer so spelling cannot carry the
test. The remaining increment of this round is the hardware session the
reviewer asked for (G9 slice 5 with the G10 APU rows).

## Rejected shortcuts

- Relabelling the Gemini-II emitted-only route as "functional-only" instead
  of removing it.
- Downgrading the coverage check to a warning for the CENT and UPMEM
  multi-statement kernels; instead each legitimately implemented store got
  a precise category (`copy`, whole-buffer zero fill, host-executed function).
- Editing golden rows or widening tolerances: no golden value changed in
  increment 1; the 4 upstream failures were verified to pre-exist.
- Keeping the Samsung dual-fiber cost rule because it reproduced the paper's
  default selection: rejected in round 2; the rule now matches the
  simulator and the paper placement is selected explicitly in the tests.
- Treating the Gemini-II typed-record runs as evidence for a source route:
  they validate the typed routes only (G2-02 was closed later by the source
  route and its own card run).
- Presenting the 26-minute 500-instance AttAcc sweep as the paper's 24 s
  selector time, or ranking AttAcc tilings with the archived replay model
  that contradicts the paper.
- Round 8: rejecting the reviewer's two source spellings by name (an
  `int()` call on the output, a guard on the final store) instead of
  restating the cast and copy contracts; and removing the store-side
  narrowing check to keep the integer kernels compiling instead of proving
  when the narrowing is exact.
- Round 9: keeping the bf16 activation compiling by widening a numeric
  tolerance or by treating `reduction_order="reassociate"` as permission
  to change intermediate precision; and realizing the reversed output map
  by renumbering the Samsung rows (that would again write a result at a
  row the source did not name).

## Constants and specialization inventory

Per-area inventories with the task's classification are in `audit/*.md`
section 4 (core, Samsung, UPMEM, AiM, APU v1, G2). Items judged defects and
their status are in `ISSUES.md` (C-08 `lanes=8`; S-09 fallback 8 blocks;
M-06 cost geometry duplication; V1-05 flat count price; G2-11 dead uint16
templates and `APUG2_U16_GEMM_BMAX = 31`; U-06 shape-exact paths).

## Gate judgments at the current candidate

Refreshed in round 28 (2026-11-04) at the candidate `4056f67`+ (this
commit). Earlier judgments are kept in the round sections above.

| Gate | Judgment | Basis |
|---|---|---|
| G1 correspondence | **PARTIAL** | every substantive mechanism of Sections III and IV has a maintained implementation reachable through `allo.compile` (CLAIMS.md), now including both APUs through the shared pipeline; the claim matrix must be re-read against the G9 pipeline and the G10 inventory (same-or-better performance is part of G1 since the 2026-11-03 scope addition) |
| G2 semantic preservation / explicit failure | **PARTIAL** | the rounds 8-11 repairs (C-19 to C-26) and the shared contract rounds (X-02 to X-14, X-17 to X-20) are implemented with regressions; the shared semantic checks run before every backend's selection; open: the deferred arithmetic findings (ISSUES), independent re-challenge of the whole domain on every runner |
| G3 general mechanisms | **PARTIAL** | structural recognizers and template-derived legality on every backend; the AttAcc attention template is proven structurally (X-20); renamed and neighbouring programs compile to the same normalized decisions in the round 24-28 regressions; open: M-14, the constants inventory re-read against the current tree |
| G4 search/layout/selected-program integrity | **PARTIAL** | complete assignments materialize under the real orchestration before acceptance (C-33b); every backend's realization is compared with its declaration (X-05 to X-16); transfer availability and physical addressing are checked (X-09 to X-11, X-17, X-18); open: the slice 5 forced-alternative traces through packing and launches on both hardware hosts |
| G5 cost models | **PARTIAL** | provenance and validity labels everywhere; cost perturbations traced to executed programs on Samsung, UPMEM and APU v1 in earlier rounds; open: C-18, uniform movement weights, the perturbation traces at the current candidate on both hardware hosts |
| G6 evidence beyond examples | **PARTIAL** | EVIDENCE.md plus the reviewer-driven probes of rounds 8-27 (renamings, operation changes, shape and slice neighbours, wrong declarations, reversed arithmetic); open: hardware-only confirmations, the probe list re-run at the candidate |
| G7 install/interfaces/maintainability | **PARTIAL** | installed-artifact runs at `7bdb475`, `e60973f`, `9255d29`; architecture docs follow the pipeline (`design/shared-compilation-path.md`); open: a fresh install at the candidate, the Samsung patches (P-04), P-23 lint, the public-payload review |
| G8 backend execution | **PARTIAL** | installed-artifact evidence on the simulators, the APU v1 board and the Gemini-II card at `9255d29`; the UPMEM, AiM and Samsung paper rows re-run through the simulators at `da041af` (G10); open: both hardware hosts at the current candidate |
| G9 one shared compilation pipeline incl. both APUs | **FAIL (in progress)** | shared front stages and facts (rounds 12-18); both APUs' plans and typed records as candidates of the common pipeline with an enforced contract (rounds 13-17); Samsung, AiM, UPMEM and AttAcc declare effects from the facts and are checked at materialization (rounds 19-27, X-04 to X-20); shared transfer availability (X-09, X-17, X-18); the `APUv1Program` manifest from the retained IR (round 28); open: slice 5 traces of forced alternatives and cost perturbations reaching generated code and launches on the APU v1 board and the Gemini-II card at one revision, with installed artifacts |
| G10 paper results through the shared pipeline | **FAIL (in progress)** | inventory of 83 rows (`paper_reproduction/inventory.csv`); measured through default selection: Samsung 3/3 faster than the paper (outputs checked), UPMEM 16/16 exact, AiM 10/14 exact with the 4 RoPE rows an OPEN 32-cycle regression (X-19), the 500-instance selector equivalence as a diagnostic; UNVERIFIED: both hardware tables, token partitioning, UPMEM scaling, packing/selection/DCC studies, AttAcc Ramulator2 timings and decode, calibration, the virtual study, compile cost |

Verdict: **NOT READY**. Blockers: G9 (slice 5 hardware traces at the
candidate), G10 (hardware rows, the remaining studies, the four AiM
regressions), the independent re-challenge of G1-G8 at the candidate, P-04
and P-23 (deferred), the open minor items in ISSUES.md.
