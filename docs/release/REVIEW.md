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
test.

The hardware sessions then ran from installed wheels at `653b059` on both
hosts (VALIDATION round 29). G9 slice 5: on the APU v1 board a forced
alternative plan and a cost perturbation each changed the selected plan,
the generated device source and the launched program, with exact outputs
and the stock model's pick faster as predicted; on the Gemini-II card the
rows-x20 perturbation moved the GEMM's plan and layout fingerprints and
the forced BiCG alternative doubled the launches, all exact, the stock
pick faster as predicted. G10: the five Gemini-II paper kernels are at or
below the paper's values over five sessions; four of the five Gemini-I
kernels are at or below, word count is 1.5% slower (X-22, open; the
expert baseline is still beaten). Judgment: the shared selection, not a
forced historical plan, produced these programs; the matrix multiply's
default selection is the paper's plan by cost, so no separate forced row
was needed, and the forced rows are labelled as such.

## Round 30 (2026-11-04): independent review of 657fa95; measurement validity, the archived protocols with their baselines, launch provenance, X-22 attribution

The reviewer accepted the round 29 hardware sessions as measurements but
not as the G10 comparison: the drivers exited 0 on failed launches and
wrong outputs, the sessions used five launches without the archived
protocols or their baselines, the Gemini-II statistic rounded ticks, and
launch provenance (binary hashes, profile counters, generated source,
SDK/device identity) was discarded with the temporary projects. Judgment:
all four are defects of the evidence, not of the compiler, and were
repaired in `69773fc`/`d2e3fda` before any new measurement was taken:
both drivers return 1 on a failed execution, a wrong output, a missing
round or a violated invariant and keep the failed record; statistics are
computed over valid launches only; the APU v1 driver runs the archived
comparison's protocol (one warm-up, ten measured launches per lane,
interleaved launch by launch with the archived corrected expert Phoenix
binary, counters per launch) and the Gemini-II driver the archived
campaign's (one warm-up per lane, twenty interleaved paired rounds against
the archived public-GTML host, raw batch ticks divided by the repetitions);
every APU v1 launch route records the binary hash and size, the parsed
profile counters and the profile text; the Gemini-II runtime already
recorded the layout and program fingerprints, packing attestation and
artifact hashes per launch.

Two things the new drivers caught on their first runs are kept as
evidence. The Gemini-II vendor lane failed every launch of run 1 because
the archived host loads its device library from a compiled-in path that
no longer existed; the session exited 1 and reported no comparison. The
path was restored with a byte copy of the archived library (documented in
place), the driver now hashes the library at the location the host reports
and fails the launch unless it is the archived one, and run 2 completed
with every output exact: the five Tenon medians are below the paper's
Tenon column and the re-run GTML baseline is within 0.6% of the paper's
GTML column, so the paired GTML/Tenon ratios are at or above the paper's
(VALIDATION round 30). Judgment: the five Gemini-II rows are REPRODUCED
under the archived protocol with the baseline re-measured; the paper's
value lies inside the session's interquartile range for gemm,
batched_gemm and mvt, which the inventory states.

The APU v1 session reproduced matrix_multiply and histogram below the
paper's medians with the expert lane re-measured within 0.3% of its
archived values; kmeans and linear_regression came out 0.1% and 1.0%
above the paper's medians with the paper's values inside or just below the
session's interquartile ranges (round 29 had both below the paper), and
word_count came out 3.0% above (round 29: 1.5%). The same-session control
that rebuilt the paper-era generated source through today's template
measured 172,416 against the paper's 171,988 with a quarter of the current
program's dispersion, which places X-22 in the generated program rather
than in the board or firmware. A second paired session from a wheel at
`d2e3fda` and a five-lane attribution session (current program, paper-era
source, the current source with the paper-era loop-counter widths, the
archived paper-era binary, the expert binary, all in the same rounds) were
pre-registered and run next (VALIDATION round 30): the five-lane session put
the whole word-count gap in the loop-counter width of the generated
program (the same source with 16-bit counters measured as the paper-era
source, at the paper's value, in the same rounds), the repair in `dc41df7`
restored the paper's program and its advantage over the expert, and the
controls showed the board measuring the paper's own kmeans and
linear-regression binaries within the current programs' dispersion.
Judgment: X-22 was a compiler regression and is repaired; the kmeans,
linear-regression and repaired word-count rows stay OPEN (nominal). The
round 30 closure of kmeans on one favourable session was wrong and was
withdrawn in round 31 (pooled-session rule).

The suites at one revision: `tests/perf` at `69773fc` passed except the
device-lock test, whose failure is an artifact of an NFS tmpdir (the probe
takes `flock` on a read-only descriptor), fixed by keeping the test's lock
file on a local filesystem; `tests/pim` at `69773fc` is recorded in the
addendum. Installed-wheel simulator checks at `69773fc` passed with no
checkout module loaded. CLAIMS.md rows 24 and E13 no longer describe typed
or separate routes (removed in rounds 13-17), and E9/E13 cite the round 30
protocols.

## Round 31 (2026-11-04): independent review of 41a7237; driver defects, acceptance wording, an installed-artifact finding

The reviewer verified the round 30 measurements against the raw records,
ran the `dc41df7` wheel on the board with a renamed record-frequency
program (two chunks, 129 queries, reversed equality, a zero-result query:
exact outputs, 142,092 cycles), and found two bounded defects in the
measurement drivers and one defect in the acceptance wording.

Drivers. The archived-baseline lane launched into the archive's own
output paths after unlinking them, so a failed launch destroyed historical
evidence; the Gemini-II drivers computed zero ticks for an empty launch
list (`all([])`) and the paper-row driver accepted the zero as a
measurement; the APU v1 driver validated the Tenon lane's count only and
ignored a binary that did not match the archive; empty case lists and
nonpositive counts were accepted. Judgment: all four are defects of the
evidence-producing code and were repaired before any further session
(VALIDATION round 31): per-launch output paths outside the archive with
failed outputs and diagnostics retained; timing evidence requires one
record per launch of the selected plan with finite positive raw ticks;
every lane's count is validated; an archive mismatch invalidates the case;
usage errors for empty or nonpositive arguments. The reviewer's probes now
fail loudly and the archive survives a failed launch.

Acceptance. The round 30 wording let a row become REPRODUCED once any
session's median crossed the paper's value, which permits favourable-
session selection; kmeans had been closed on one such session while the
pooled 30 samples sit above the paper (624,950 vs 624,677). Judgment: the
closure was wrong. The rule is now pooled over every complete session at
the revision, with at most one pre-registered confirmation campaign per
nominal row; kmeans and linear regression are OPEN (nominal) on the pooled
statistics, with their historical-binary controls reported, and no session
is discarded.

Installed artifact. The functional reference route relies on a NumPy
conversion that NumPy 2.4+ rejects; three existing tests fail with the
installed wheel (V1-19, recorded, repair deferred behind G9/G10, tests not
weakened).

G10 then advanced on an available path: the Samsung XOR layout probe
measured 148 to 116 cycles exactly on PIMSimulator and the Gemini-I vector
GEMM layout row ran on the board (VALIDATION round 31). The Samsung row
was marked REPRODUCED in round 31 as a "mechanism row"; the reviewer
rejected that in round 32 (the probe fabricates a MatchedOp and bypasses
the shared stages), the status is withdrawn to UNVERIFIED with the probe
kept as diagnostic evidence, and the shared-pipeline route is the open
implementation item. The 20
Docker-blocked simulator tests stay UNVERIFIED until the host's `/var` is
freed; they are not rerun blindly.

## Round 32 (2026-11-04): independent review of 257b91d; a G10 closure through a bypass, two driver defects, the Samsung route

The reviewer found the Samsung XOR row closed in round 31 through a
bypass: the probe fabricates a `MatchedOp`, calls the autoscheduler
directly and lifts the layout into a separate trace harness, and G10
excludes direct-selector reproduction. Judgment: the reviewer is right and
the "mechanism row" label was an exception G10 does not grant; the status
is withdrawn (inventory, CLAIMS E3, S-13 raised to major), the exact
148/116 cycles are kept as diagnostic evidence.

Two more driver defects were repaired before any further session
(VALIDATION round 32): the selection-trace driver now requires one launch
record per launch of the selected plan, in order and with the plan's layout
fingerprint and a positive tick count, so omitted work is a named problem
and the command exits 1 while keeping the record; the archived-lane
`--scratch` path is resolved to an absolute path before the launch
directories and arguments are built, and a relative scratch with a
successful and a failed launch leaves the archive byte-identical.

The round then went to the missing route rather than to the ledger. Reading
the production lowering and the conductor, it concluded that every
reduction tile is staged at DRAM row 0 (corrected in round 33: the address
generator carries columns into rows and a tile spans two rows; what the
stream lacks is two rows of one bank inside one barrier group). The fiber-swizzle lever
was measured on the real conductor through `allo.compile`: the vendor
`dual_fiber` placement 8,247 cycles (the paper's vendor row), `bank_row`
with the same levers 8,153. Judgment: the paper's mechanism exists in the
layout algebra the enumerator uses, but the row-packing it de-conflicts is
not a lever of the maintained Samsung backend; the route that makes it one
(a `matrix_rows` candidate family, a row-activation cost term, `tile_row`
through the program and the conductor plan, a program-derived
`bank_access_trace()` artifact) is specified in
`design/samsung-bank-row-swizzle-route.md` and is the next G10
implementation item for this row. The row stays UNVERIFIED.

## Round 33 (2026-11-04): independent review of e357628; the row-0 diagnosis corrected with the conductor's own trace; the first production slice

The reviewer showed the round 32 diagnosis wrong: `addrGenSafe` carries
column overflow into the row, so with 32 burst columns per row each
64-column tile spans two physical rows, and the measured A/B did not
establish an absence of row changes. Judgment: accepted in full; the
conductor overlay now dumps its normalized stream on request, the dump of
both placements of the 8192x256 GEMV (and of the 4096x512 neighbour)
confirms the reviewer's arithmetic (bank_row: bank 0 rows 0-3 and 4-7,
seven same-bank row changes; dual_fiber: banks 0 and 1 rows 0-3, three
each), and every copy of the wrong statement was corrected (design,
ISSUES S-13, inventory, STATUS, VALIDATION, this file, handoff).

What the stream does establish is narrower than round 32 claimed and is
now evidenced rather than asserted: every barrier group is eight
consecutive columns of one row of one bank, so the within-round two-row
conflict the paper's four-request probe measures never occurs in the
production program; the layout-dependent cost it has is the row change
between successive groups. The multiplicity, ordering and timing boundary
of the probe therefore cannot be obtained by compressing the GEMV stream,
and the design now says so instead of proposing a one-request-per-tile
trace; its cost proposal no longer calibrates against 148/116.

The production slice: `samsung_access_stream` derives the executable
pass-phase stream from the lowered program and the geometry; it is
byte-identical to the conductor's dump for both placements of two GEMVs
(a gated test runs the conductor with the trace enabled; a unit test pins
the counts, rows and row changes); the layout-trace harness was made able
to replay a program-length stream; the replay of the MAC and FILL
addresses as plain reads with a barrier per group gives 1,188 vs 1,248 and
1,228 vs 1,248 harness cycles while the full programs run 8,247 vs 8,153
and 8,041 vs 7,947, both through `allo.compile` with explicit placements.
Judgment: the stream itself is shared-pipeline evidence (what the
materialized program addresses, verified against the running conductor,
on two workloads); the harness numbers are a diagnostic of the address
sequence without PIM mode, fills, drains or staging, they do not show the
conductor's bank-read phase to be faster nor attribute the full-program
difference, and the modelled "row changes" are address changes, not
measured activations (narrowed in round 34 at the reviewer's request).
It is not the paper row, which stays UNVERIFIED.

Also this round: the Gemini-II drivers require a launch index and the
plan's layout fingerprint on every record (reviewer's provenance probe);
CLAIMS E4 reports 10 exact AiM rows and four OPEN (X-19).

## Round 34 (2026-11-04): independent review of d4c6703; the stream's defaults, the overlay, the wording; the portability row

The reviewer found the new access stream defaulting to one result column
and substituting the offline geometry when the installed driver's query
failed, the trace instrumentation missing from the release overlay, and
the performance wording overreaching. Judgment: all accepted and repaired
before the stream is used as evidence (VALIDATION round 34): the column
count comes from the retained program and any other count is an error;
geometry errors propagate and offline streams are labelled; the release
overlay carries the instrumentation and builds the identical driver; the
gated test fails instead of skipping; the design, REVIEW round 33 and
VALIDATION round 33 call the harness replay a diagnostic of the address
sequence and the summary's transitions modelled address changes.

G10 then advanced on an available row, the paper's portability claim (the
same 4096x256 FP16 GEMV source, rounding-sensitive inputs, bit-exact on
physical Gemini-I and PIMSimulator under source order). With Gaussian
inputs the plain source is bit-exact on PIMSimulator and 1,900 of 4,096 on
the board, which is V1-16 (the device's FP16 multiply is not an
exact-product rounding; no standard rounding model fits). The paper's own
audit used inputs whose products are exact (signed powers of two against a
vector of ones, with cancellation rows), so the second pass re-ran that
exact experiment through `allo.compile` on both targets; its outcome is in
the VALIDATION round 34 addendum and the inventory row. Judgment: the
claim as the paper states it is about accumulation order, which the strict
policy guarantees on both targets; bit-exactness of arbitrary products on
Gemini-I is not claimed by the paper and is not reproducible (V1-16).

## Round 35 (2026-11-04): independent review of 790e7a9; the portability reproduction brought into the repository

The reviewer accepted the round 34 repairs and the measurement, and
required the reproduction itself to live in the release repository with
its source, historical inputs, independent oracle, execution-kind
assertions and full evidence retention, since the inventory cited an
external script and the maintained tests check a different experiment.
Judgment: accepted; `benchmarks/portability/gemv_source_order.py` is that
driver, its tests run without hardware, and the inventory and CLAIMS E11
now cite an executable in-repo command. The driver's run from the checkout
reproduces the paper's claim on both targets (4,096 of 4,096 bit-exact
under strict order with the paper's inputs; the regrouping board plan 859
of 4,096) and retains every output, the retained IR, the generated source
and binary identity, the launch and profile evidence, the CRF words and
conductor receipt, and the environment. The observation is labelled
checkout evidence until the installed-artifact run; the Gaussian-input
failure and V1-16 stay open. The stream summary's docstring was aligned
with the narrowed diagnostic scope.

## Round 36 (2026-11-04): independent review of e6f54f0; two driver defects; the tile-row levers implemented

The reviewer found the portability driver skipping validation of its
reassociation controls and truncating the selected plan's provenance
(a `mappingproxy` rendered as 400 characters). Judgment: both accepted and
repaired with regressions before the driver is cited again; the Samsung
command re-run passes from the repaired driver (VALIDATION round 36).

The substantive increment is the grouped-REDUCE slice of S-13 through the
shared pipeline. The design's first idea, pairing two k-tiles under one
barrier, was rejected before coding because two tiles need two vector
slices in one GRF_A; the sound pairing is the two physical rows of one
tile, whose GRF_B entries the row parity already separates, so the
accumulation order per entry and the results are unchanged. Three
candidates were added with the shared contract, legality at lowering, the
plan fields the conductor executes, the lever comparison at realization
and the access-stream model; the conductor overlay gained `issue_order`
and `split_rows`. Measured on the conductor through `allo.compile` on two
GEMVs: identity rows under one barrier cost about 2,845 cycles per pair
(53,634 against 8,153); the XOR row packing with the vendor's barriers is
bit identical and 1.124x / 1.126x faster than `bank_row` on the whole
program; the XOR packing under one barrier executed with wrong results
(the controller may reorder two banks' commands inside a barrier group
while the lockstep CRF runs fixed-bank MAC instructions) and is now
refused at lowering with that reason. Judgment: the mechanism the paper's
row describes, the swizzled rows across the fibers against the identity
rows in one bank, is now an ordinary-source choice executed by the
conductor with comparable timing and exact outputs; the paper's probe
metric (within-round conflicts on plain reads) is unreachable by the
production program, so the row is OPEN (measured), not REPRODUCED, and the
default selection does not yet earn the faster packing (S-16, the staging
saving is outside the MAC rule). The measured per-pair constant prices the
identity pairing; the split layout's rank is an open cost-model item, not
a fitted one.

## Round 37 (2026-11-04): independent review of 4378bf7; the selected map made the contract; the staging term; the default earns the swizzled rows

The reviewer accepted the tile-row levers as substantive and found that
the selected half -> bank map was a diagnostic attachment: the lowering
followed a label, the conductor hardcoded the upper-half bank, the shared
matrix layout omitted its packing, the map was outside the decision
identity, and an edited map with a bogus packing compiled unchanged.
Judgment: accepted in full. The map now names the packing the candidate
declares, the lowering derives the row placement from it and rejects a
disagreeing label, the plan carries the map's rendering and the conductor
addresses both halves from it, the realization check compares the
program's row fibers with the declared packing, the map is a physical
decision field, and the shared validation compares every declared value
layout and transfer with the target's derivation before costing while a
caller-supplied declaration is no longer overwritten. The reviewer's probe
now reports a changed decision and a rejection before selection.

The reviewer also required the ranking repair without a fitted discount.
Judgment: the estimate's scope now includes the first matrix staging,
derived from the conductor's staging schedule (verified equal to the
staging dump for every placement on three workloads) and priced from the
device ini; the split-rows accounting applies for any issue order; the
row-pair constant states its validity domain. With that term the default
selection picks `split_rows` on both paper GEMVs and on the reviewer's
held-out 37x129x2 contraction, and the conductor confirms it is the
fastest placement on all three (7,255 / 7,057 / 8,163 against bank_row
8,153 / 7,947 / 8,724), bit identical. Judgment: the ranking is validated
on a shape the model was not built on; absolute estimates remain on the
declared device and are compared by ranking only, as the module states.
The XOR paper row stays OPEN (program-level metric); the cost-model issue
is S-16 (the earlier S-14 label collided with the output-binding issue).

## Round 38 (2026-11-04): independent review of c29287d; the staging accounting and the declaration axes

The reviewer accepted the map-to-packing-to-conductor repair and found
two gaps: the staging term charged every work-item event a whole-fabric
pass divided by the fabric (equal work priced differently, rows not
scaling, the temporal family outside the scope, the staging after the MAC),
and the declaration checks compared storage and packing but not the
semantic axes, so an emptied matrix layout passed through coverage.
Judgment: both accepted. Each work item is now charged its own rows' share
of the realized staging (bus time on the channel, the bank's row occupancy
as latency with banks in parallel), the shares partition the conductor's
staging exactly (checked against the dump counts for the paper GEMV), the
staging precedes the MAC, and the ordered program carries the same scope
from its own preload pattern; the declaration checks compare axes, a
supplied declaration that disagrees on any realized field is refused rather
than replaced, and the map must keep the bank input as the base fiber.
With the repaired accounting the default still selects the swizzled rows on
both paper GEMVs, the held-out 37x129x2 and the reviewer's 19x65x3, the
measured fastest wherever measured. S-16 is repaired with its remaining
scope stated (declared-device absolutes; the ordered program's preload 16%
under its measurement). The XOR paper row stays OPEN (probe metric).

## Round 39 (2026-11-04): independent review of 0a4b543; the supplied-declaration boundary and the next inventory study

The reviewer accepted the staging repair (its own 23x81x2 contraction
selects `split_rows` and executes exactly at 3,241 cycles) and found the
declaration fix incomplete: the compile boundary compared three fields and
then replaced the whole supplied record before validation, so an edited
image, buffer identity, launch domain or allocation set passed
`allo.compile(..., layout=...)` while failing direct validation.
Judgment: accepted (C-34). A supplied declaration is now a claim about the
program: normalized onto its work item by structure (function and value
bindings by operand role, which is what lets one body's declaration serve
every body of a lockstep kernel), validated in full, and compared field by
field with the current candidate's declaration for the same decision; any
difference refuses the compilation before anything is replaced, in every
rebinding branch. Reuse is preserved structurally, not by names: a bare
decision and another program's plan (a declaration covering a function
outside the program) supply their decision and get the re-derived
declaration, recorded in the placement. The derivation check itself now
compares the whole record. The reviewer asked next for an inventory study
through actual shared selection and simulator execution; Docker is out of
disk on this host, so the AiM simulator was built natively from the same
tree and validated on the exact golden rows before the token-partitioning
study ran through the maintained driver (`benchmarks/cent_aim/
token_partition.py`); it reproduces in kind (6 of 9 simulated minima, largest gap 1.34%, against the paper's 7 of 9 and 2.63%), with every compared plan's declared effects, lowering entry, estimate, device cycles and executor recorded, and the row is OPEN (measured), not REPRODUCED: one fewer minimum, and the 1-head head-only ratio is not comparable because the ranking finds an in-budget plan outside the paper's candidate set.

## Round 40 (2026-11-04): independent review of 128aef8; duplicates, the comparable boundary, compiler-owned budget, provenance

The reviewer verified the round 39 tests and the native executor on the
default golden rows, reproduced the 1-head/128-token row, and found four
gaps. (1) C-34 incomplete: the comparisons collapsed layouts and transfers
into dictionaries, so a contradicting duplicate record passed validation
and was replaced. Judgment: accepted; duplicate (value, image) and (value,
direction) keys are refused by the validator and by both comparisons, in
either order, with regressions at both boundaries. (2) The token study does
not measure the frozen paper's boundary (no softmax bridge, no host-stage
time, no readout/upload link volumes; 37 distinct plans, merge checks only
on split plans). Judgment: accepted; the inventory row is DIAGNOSTIC, the
driver, the design doc and the ledger say what the measured boundary is and
is not, and M-15 holds the bridge and full accounting as the next
increment; the bridge is not implemented in this round. (3) The eight-channel
selection lived in the benchmark. Judgment: accepted; the budget is now a
general compiler search policy (`resources={"channel": 8}`) read by the
enumerators, the ranking, the rebinding boundary and the realization check,
and the headline plan is the normal shared selection under it; the forced
plans remain as ablations. (4) Provenance. Judgment: accepted; every AiM run
records the executor with the resolved shared library and configuration
digests, the exact segment traces, repeats and simulator output and the
invocations; the driver keeps every plan's artifacts and raw result before
advancing, failed plans included, with the compiler revision. S-17 and the
earlier blockers stay open; G9 and G10 remain FAIL.

## Round 41 (2026-11-04): independent review of 0694791; study failure handling, the budget contract, failure provenance

The reviewer accepted the duplicate-declaration repair, reproduced the
1-head/128-token row natively with matching provenance, and found three
bounded gaps. (1) The study could misreport: a failed distinct control was
aliased to P1, partial alternative sets were compared for minima, and the
`paper_full` test passed with a failed plan. Judgment: accepted (M-16);
aliases are decided before execution and never substitute for a failure,
a comparison is complete or not made, and the test gate requires every
expected plan. (2) The resource budget accepted units it ignored.
Judgment: accepted (C-35); the contract is bounded to the enforced pairs
(AiM channel) and refuses the rest before enumeration. (3) Failed-plan
provenance was overwritten and a simulator failure kept nothing.
Judgment: accepted (M-16); the record is established before fallible
steps and the failure appended, and the runner raises structured evidence
with the exact trace, identity, command, return code and output. The M-15
bridge follows these fixes; the study stays DIAGNOSTIC.

## Round 42 (2026-11-04): independent review of 6e988e9; the M-15 bridge as the main deliverable

The reviewer accepted the round 41 repairs, reproduced the 1x128 row
natively, found one bounded provenance defect (a simulator timeout's
partial streams were dropped; fixed, with a regression) and required the
M-15 bridge through the shared plan. Judgment: implemented as a vertical
slice. `allo.host_compute` declares the global softmax between QK and SV
in the host program; the launch schedule orders it with its declared
volumes; the availability walk binds its inputs to the producing launch
and its outputs to the consumer's host ingress; the AiM program lowers it
as a host step and the realization check binds it; the program graph the
normal selection ranks prices it as the target's `HOST_COMPUTE` operation
(readout and upload over the declared link plus declared host vector
passes; the partial readout stays in the host reduction, nothing counted
twice); other backends refuse it. The driver compiles the bridged program,
records independent functional evidence per plan (the lowered QK and SV
programs' oracles around the compiled stage's own function, against a
float64 reference; the timing-only simulator produces no values and none
are attributed to it), measured host-stage samples beside the modelled
price, the CENT attention program as the control for the 32-head workloads
with its boundary stated, and a neighbouring 2x256 workload; every plan's
provenance, traces and executor identity are retained and failed plans
are kept. The study stays DIAGNOSTIC: the host stage's price is a declared
per-vector model (measured samples beside it) and the CENT control's
boundary is CENT's own. G9/G10 and the recorded blockers remain open.

## Round 43 (2026-11-04): independent review of 3a40971; host-stage invariants, attention semantics, timing accounting

The reviewer accepted the bridge's reach into scheduling, costing and
lowering, reproduced the 1x128 row, and found five bounded gaps (M-17):
the softmax lacked the 1/sqrt(head_dim) scale and so did its reference;
the realization check read the declared schedule, not the realized
sequence, and ignored the computation; the host-time model was chosen by
the stage's label; the measured-host total dropped the stage's link term;
and the realization refused host inputs the shared availability walk
admits. Judgment: all accepted and repaired. The stage is the scaled
softmax of the workload's head dimension, verified on operands that expose
the scale; the lowered stage is bound to the recorded callable, the
declared operation and buffers and its position among the realized launch
occurrences, refused at materialization otherwise; the semantic operation
is an explicit declared domain the policy keys on, carried through the
schedule, graph, record and manifest; the price is stored as link and
compute terms and the measured total replaces the compute term only; a
stage may read a region input no launch writes or an earlier stage's
output. The study is re-run on the repaired boundary and stays DIAGNOSTIC;
the one-head misses are examined from the retained graphs (design doc).
G9/G10 and the recorded blockers remain open.

## Round 44 (2026-11-04): independent review of b0c667b; launch occurrences and completion

The reviewer accepted the round 43 repairs and showed the realization-order
part of M-17 only partially closed: the per-launch record count came from
the records under check (a duplicated QK contraction passed and reached
the simulator), and host reductions were not ordered against stages (a
stage before the producing launch's merge, or that merge dropped while a
repeated launch kept its own, passed). Judgment: accepted. The expected
inventory now comes from the retained facts and the selected candidates,
per scheduled launch occurrence, and the lowered records are walked
against it; a duplicated, missing or displaced record, or a host reduction
with another partition count, fails materialization before selection, and
the unmodified repeated-launch program is accepted. (Correction, round 45:
that round bound only the reduction's partition count; its outputs,
batches and replicas were not bound to the producing contraction, so a
mis-parameterized reduction still materialized with a manifest price that
disagreed with the priced graph. Round 45 binds the whole work domain.)
The study's label stays DIAGNOSTIC and 6 of 9 minima does not establish
the paper's 7 of 9. G9/G10 and the recorded blockers remain open.

## Round 45 (2026-11-04): independent review of 25c482d; the reduction's work domain

The reviewer accepted the occurrence and count repairs and showed the
closure overstated: a host reduction's outputs, batches and replicas were
not bound to its producing contraction, so a reduction with outputs 1,
batches 1 or replicas 2 (2 heads x 128 tokens, P = 2) materialized with a
manifest price of 8, 32 or 128 cycles while the selection graph kept 64,
and only the functional merge later failed on a shape mismatch. Judgment:
accepted. The inventory walk now binds each reduction occurrence to the
contraction record of the same occurrence that produces its partials
(outputs, batches, replicas and the contraction's partition count), which
is itself checked against the candidate's declared launch; the manifest
price follows from those fields and equals the priced graph's, verified
for the single and the repeated program. The reviewer's three mutations,
on the first and on a later repeated launch, are refused at
materialization. (Correction, round 46: the producer itself was still
validated by the product outputs x batches, so a coordinated change of the
contraction and its merge to outputs 256 x batches 1 passed; round 46
binds each contraction occurrence's outputs, batches, replicas, reduction
and vector source to the retained semantic shape under the selected
candidate.) The study stays DIAGNOSTIC; G9/G10 and the recorded blockers
remain open.

## Round 46 (2026-11-04): independent review of 152095e; the producer's own dimensions

The reviewer accepted the merge-to-producer binding and showed the
producer still validated by the product outputs x batches: changing the SV
contraction and its merge together from outputs 128 x batches 2 to
outputs 256 x batches 1 was accepted (57 commands and 2,066 simulated
cycles against 33 and 1,178, the merge still priced 64), the functional
check alone refusing it. Judgment: accepted. Every launch's contraction
record is now bound dimension by dimension (outputs, batches, replicas,
reduction, vector source) to the retained semantic shape of its
contraction under the selected candidate, grouped realizations included;
the coordinated mutation and the replica and reduction variants are
refused on the single and the repeated program, and the unmodified
programs and the reviewer's neighbouring 4 heads x 256 tokens x head
dimension 64 workload pass with the merge priced as the graph prices it.
The study stays DIAGNOSTIC; G9/G10 and the recorded blockers remain open.

## Round 47 (2026-11-04): independent review of 65750db; realization fidelity beyond dimensions

The reviewer accepted the dimensional repair and showed that the emitted
activation, reuse scheduling and GPR addressing of a contraction record
were still unbound: an added activation, another reuse group or another
result GPR address on the QK record was accepted under the unchanged
selected plan and estimate, with more commands and more simulated cycles.
Judgment: accepted (M-18). Rather than another field patch, each launch
occurrence's realized launch-owned records are now compared record by
record with the records the retained program and the selected decision
derive afresh for one launch of the group, every emitted field included;
a difference names the fields and fails materialization before selection.
The reviewer's three mutations and two more (input GPR, channel set) are
refused on the single and the repeated program; the unmodified programs
and the 3 heads x 192 tokens x head dimension 64 neighbour pass. Relocation
bank copies, syncs and host transfer records are not yet under this
comparison (M-18 says so); the round 46 closure stays dimensional. G9/G10
and the recorded blockers remain open; the study stays DIAGNOSTIC.

## Round 48 (2026-11-04): independent review of 35b6351; the emission's bank payload

The reviewer accepted the record-by-record comparison and showed it
discarded half of the emission input: the bank payload the emission
expands MAC_ABK into MAC_SBK with. An all-bank QK record whose payload was
replaced by sixteen banks, or by bank 0 sixteen times, was accepted with
the all-bank plan's scope and estimate, the scope check reading the
group's expectation. Judgment: accepted (M-18 extended). The emission
input is retained in full, each record's emitted payload is compared with
the derived one and must be distinct banks of the device, and the scope
check reads what was emitted; the two payload mutations and a four-bank
one are refused, and genuine single-bank and all-bank candidates lower
with their derived payloads. Relocation bank copies, syncs and host
transfer records remain outside the whole comparison, as M-18 says.
G9/G10 and the recorded blockers remain open; the study stays DIAGNOSTIC.

## Round 49 (2026-11-04): M-15 advanced; the CENT-derived eight-channel control, the host protocol, the calibration of the misses

The reviewer accepted the bank-payload repair and asked for the M-15
workstream. Delivered: the CENT-derived control is now structurally
derived (CENT's mapping policy, read from the shared pipeline's exact
selection on the CENT golden attention cases, matched by decision knobs in
each workload's budgeted enumeration and compiled under the study's own
budget, bridge and accounting), replacing the incomparable 32-channel
case; the historical host protocol (decode of the readout, scaled softmax,
staging, merge) is measured step by step on every compiled program and
composed with the modelled link terms, beside the priced total; the nine
workloads and the neighbour are re-measured with every variant and control
accounted for; and the one-head misses are investigated with an
independent command-level calibration driver (prefix simulation), which
attributes them to the contraction profile's register I/O, WR_GB and MAC
charges (M-19), not to the bridge or the accounting. The study stays
DIAGNOSTIC: the priced host compute is a declared model and the profile's
charges await recalibration from command classes; 6 of 9 minima does not
establish the paper's 7 of 9. G9/G10 and the recorded blockers remain open.

## Round 50 (2026-11-04): the frozen study's control restored, the host protocol on the attention's inputs, the contraction profile recalibrated (M-19)

The reviewer's round 50 findings, all accepted: (1) the round 49 control
(current CENT policy, 16-slot reuse, heads up to eight) was not the frozen
study's `cent-repaired` control (padded heads, reuse 1, eight channels,
reserved row pitch, normalized against CENT's events); (2) the host
protocol was timed on masked random words (scores up to 3.4e38, 127 of
128 probabilities zero) with a duplicated BF16 rounding and without the
historical warm-up/batching; (3) the calibration driver took the empty
prefix as zero while EOC alone takes 2 cycles, and the profile's charges
needed a focused production repair validated on controlled variations and
held-out programs before any confirmation outcome; (4) stale docstring and
`not_comparable_because` text.

Delivered: the historical control restored through the shared pipeline
(`cent_repaired_program`, `cent_repaired_layouts`, `CENT_REPAIRED_POLICY`,
`CENT_REPAIRED_PROVENANCE`, command inventories per kernel; the round 49
control kept as the `cent-policy` ablation); the host protocol measured on
the functional computation's BF16 scores, encoded once, timed whole with
the historical methodology and broken down beside it; the calibration
baseline measured; the contraction profile recalibrated per command class
from controlled single-factor command streams (`contraction_profile_charges`,
the profile's activation classes, WR_GB columns, mode switches and
register runs), validated on the standalone probe, on held-out programs
(every budgeted candidate of each kernel simulated to its boundary) and
only then on the nine workloads, which the normal selection now minimizes
on all nine (every gap 0; estimates within 0.3% of the measured totals);
docstrings and the driver's statement updated.  Not done on purpose: no
benchmark selector, no tie rule, no fitting to archived winners; the paper
frozen; the row stays DIAGNOSTIC (declared host compute price; the
recalibrated selections await the reviewer); M-15 and M-19 remain open
until reviewed; the deferred items unchanged.  Per the user's scope note,
this closes the bounded token-partition/cost checkpoint; the next
increment moves to the required backend rows and the AttAcc figures.

## Round 51 (2026-11-05): the AttAcc kernel-tiling figure (required G10), documentation defects

The reviewer accepted `0cc8187` as the bounded M-15/M-19 checkpoint and
asked for the required AttAcc figures, starting with a fresh
shared-pipeline-selected QKV timing, plus three documentation defects.
Done: the inventory's `release_scope` column (55 required / 28 optional,
statuses unchanged), the narrowed blocker list below, the stale SBK and
error-band text in `allo/pim/costs/aim.py` and the qualified "never
reorder" statement (the single-bank contextual residual).  The AttAcc
increment: Ramulator2-AttAcc built natively from its MIT sources with a
shipped recipe (the container route is unavailable while Docker's disk is
full); `run_ramulator.py` keeps the compiled artifact, records the
selection's provenance, checks DCC's generator trace against the selected
program command by command and address by address (`trace_check.py`;
validated against DCC's own output), labels forced tilings and the
fixed-form baseline as controlled references, and retains every artifact
with digests; all six kernel results measured: the four GEMV rows at the
paper's values with their fixed baselines re-run, the two score rows at
the figure's speedups with the text's 500-instance tally against DCC
reproduced exactly.  Observed and recorded, not hidden: `tables/eval-attacc-kernels.tex`
is not input by the frozen paper and its superseded values (score 0.0019 /
0.00677 ms, "Tenon <= DCC on every shape") differ from the frozen figure
and text, which the fresh measurements match; the score baselines are
DCC's archived per-shape simulations, not re-run.  Next: the six decode
configurations through `compose_e2e.py` with the fresh kernel times
(analytical composition, labelled), then the AiM X-19 and Gemini-I nominal
rows.

## Round 52 (2026-11-05): trace fidelity made per-command, failed runs retained, the decode configurations

The reviewer found round 51's fidelity claim larger than its check
(cardinalities of address sets, stores / drains / barriers unchecked) and
the driver discarding a failed run's inputs.  Both repaired without
renaming or weakening: the checker now derives the whole command stream
(opcode and address of every command, the generator's region placement
and composition) from the selected program and the target geometry and
compares command by command; the reviewer's three mutations and four more
fail; DCC's own output matches exactly on GEMV and attention shapes; all
506 retained traces re-checked.  The driver retains inputs, complete
output and a structured failure record on trace mismatch, simulator error
and timeout, and exits nonzero.  Then the six decode configurations were
composed from the fresh shared-pipeline kernel times with the input
discipline the reviewer asked for (shared-pipeline records only, conflicts
rejected, no reference substitution, unmeasured rows NaN): Tenon, DCC and
the fixed baseline reproduce the frozen figure, table and text, and the
archived batch-4 discrepancy is explained (missing measurements and the
fixed variant's placement).  Required G10 set: 48 of 55 REPRODUCED, 7
OPEN (X-19, X-22 / X-23), 0 UNVERIFIED.  Next: the four AiM RoPE rows
(X-19), then the three nominal Gemini-I rows.

## Round 53 (2026-11-05): X-19 investigated to its mechanism; the Gemini-I campaign pre-registered

The reviewer asked for the four AiM RoPE regressions to be investigated
and repaired without restoring the omitted work, changing the paper's
values or repinning timings.  Investigated: per-stage prefix simulation
attributes the corrected program; the paper's digit (129,605) is reproduced
exactly by the same trace with the two multiplies at the paper's half op
size, so the whole 32-cycle difference is the execution of the omitted half
of the RoPE data; every kernel group's selection is the cost minimum of its
enumerated candidates (the projections' candidates tie on one command
stream for a structural reason: one MAC register per bank and a 1,024-element
global buffer); the multiply's op size is the device's geometric minimum for
the full vector; a neighbouring hidden size lowers and simulates
consistently; a regression test pins the width and the slot coverage.
Limits of the search: the candidates were compared by estimated cost, not
simulated one by one, and the width bound covers the multiply alone, so
the attribution does not establish the minimum simulated time of a
faithful complete program.  The rows stay OPEN: the same-or-better rule is
not met and the explanation does not waive it; nothing was repinned.  The
Gemini-I confirmation campaign (X-22 / X-23) is pre-registered with its
protocol, session count and decision rule and deliberately not launched:
the protocol binds it to the final candidate and sessions at an earlier
revision would not pool.  Next: the remaining candidate-wide gates (G9
independent review, installed-artifact and backend validation at a
declared final candidate, at which point the campaign runs).

## Round 54 (2026-11-05): Gemini-II launch recipes bound to the contract (X-25); X-19 claims narrowed

The reviewer showed by fault injection that a Gemini-II launch's intrinsic
recipe was never compared with the retained contract: a halved recipe
under an unchanged declared reduction passed costing, selection and
materialization and failed only in public packing.  Repaired through the
existing materialization hook: the enumerator's own derivation of a
launch's recipe from the contract and the launch partition is now one
function (`launch_program_for`), and `check_realization` re-derives every
realized launch's recipe and refuses a mismatch in reduction extent,
operand and output types, output geometry, epilogue, fingerprint,
accumulator chaining or final-result routing, plus incomplete reduction
chains and final tiles that do not partition a stream's coordinates; the
refusal is the shared feasibility rejection, so the ranking takes the
intact alternative.  Regressions cover the damaged cheapest candidate,
the forced damaged candidate, the unmodified transposed program and two
further inconsistencies.  No new route; no change to the APU v1 path.
Also done: the X-19 statements in CLAIMS, PAPER_REPRODUCTION, REVIEW, the
inventory and the new test's documentation now state the demonstrated
attribution and the search limits (estimated-cost minimum over the
enumerated candidates, a width bound on the multiply alone) instead of an
impossibility claim; the rows stay OPEN under the same-or-better rule;
STATUS's required counts refreshed (48 / 7 / 0).  Next: name an integration
candidate, validate its installed artifact and backend paths, and run the
pre-registered Gemini-I campaign.

## Round 55 (2026-11-05): pair-affine coefficients bound to the stream identities (X-25 completed)

The reviewer's second fault injection reversed a pair-affine launch's
streams through the enumerator; the round 54 check re-derived the same
contract-order coefficients and accepted a recipe that would compute
3 (b x) + 2 (a x) for a source requiring 3 (a x) + 2 (b x).  Repaired in
the single derivation: the coefficients follow the stream identities into
the launch's order, and a pair-affine contract's streams must share one
launch.  The reversed-stream recipe with unchanged coefficients is now
refused at materialization (the compilation fails: it was the only legal
plan), the same launch carrying its coefficients in its own order is
accepted and computes the source result, and the ordinary pair is
unchanged (fused/k1, coefficients (3, 2), 112 per output).  Also corrected:
the VALIDATION wording on slot capacity versus coverage and the stale
statement in `spmw_candidate.py` that matcher backends declare no effects.
Next: the integration candidate, its installed artifact and backend paths,
then the pre-registered Gemini-I campaign.

## Rounds 56-57 (2026-11-05): integration candidate `54cbb08` validated from installed artifacts

The candidate is `54cbb08` (X-25 completed, clean tree).  From clean venvs
outside the checkout on both hosts: the round 8 probes (fresh install,
Samsung simulator, AiM timing-only through the native executor; UPMEM
blocked by the host's full Docker data root), eleven test files green, the
shared-semantics suite 131 of 132 with the one failure the recorded NumPy
2.4+ item V1-19; on zhang-x4 the card tests (15) and four paired sessions
of the five Gemini-II rows (four REPRODUCED, batched_gemm OPEN (nominal)
after its pre-registered confirmation campaign X-26, which is spent); the
fourteen AiM rows through the native simulator, default and `paper_full`,
with per-row selection, trace and executor provenance (ten exact, the four
corrected RoPE programs 32 cycles above the paper, OPEN); the three Samsung
rows (vendor-equivalent programs at the paper's values, defaults faster).
The Gemini-I campaign could not be completed: one attempt misused a driver
flag and launched nothing, one lost launches to the vendor tool's scratch
writes when the host's shared `/tmp` reached 100% (not our files); both are
recorded, nothing pooled, a watcher launches the campaign when the partition
has room.  Reporting discipline: the inventory's 47 REPRODUCED / 8 OPEN is
historical coverage at the recorded revisions; validation at the candidate
is listed separately (Gemini-II, AiM and Samsung executed; UPMEM, AttAcc
and Gemini-I not yet).  Lesson recorded: my kept Gemini-II temporaries had
filled zhang-x4's `/tmp` and were removed.

## Round 58 (2026-11-05): the Gemini-II repetitions update bound to the validated plan (X-27)

The reviewer found that the public `repetitions` setter re-enumerated the
plans and installed a replacement by name without the shared checks, so a
validated artifact could acquire a recipe compilation would reject (shown
by fault injection into the enumeration).  Repaired: the update now
derives each launch's recipe from the retained contract and the launch's
own partition at the new count, refuses any difference from the validated
launch beyond the repetition count, re-runs the recipe check, prices the
rebuilt plan, checks the priced graph against the candidate's declared
effects, and only then installs plan, graph and count together; a failure
leaves the previous artifact in place.  The enumerator is not consulted and
nothing is selected by name.  Regressions cover the corrupted enumeration
(no effect, enumerator not called), drifted derivations and reordered pair
coefficients (refused, previous plan kept, artifact still packing the
source result) and a valid update of the forced two-launch plan (partition
kept, count in every recipe and in the manifest, re-priced, packs).  The
reviewer's reproducer now stops at its own "expected recipe mismatch"
assertion: the update no longer admits the corruption.  Hardware and
UPMEM candidate execution remain blocked by the hosts' full `/tmp` and
Docker data root; X-19 and X-26 stay OPEN; no session was added.

## Round 59 (2026-11-05): the graph check of the repetitions update actually runs (X-27 correction)

The reviewer showed that the round 58 setter never invoked the shared graph
consistency check on the public route: it looked for the layout under a
nested attribute the attached artifact does not have, so an injected
priced graph claiming three launches for the two-launch plan was
installed.  Corrected: the setter reads the attached artifact's own layout
and runs `check_graph_consistency` on every placement before installing
anything.  The regression injects the inconsistent graph on the forced
`single/k2` plan (refused; the previous plan object, both graph references,
the count and the estimate retained; the artifact still packs) and then
performs a valid update (the check is invoked and passes, the consistent
graph installed).  The round 58 claim that the check ran before
installation was false for that commit and is corrected in ISSUES,
VALIDATION and the handoff.  The installed-artifact candidate `54cbb08`
predates X-25's completion and X-27; the next candidate must include them.

## Round 60 (2026-11-05): integration candidate `cd744bb` validated from installed artifacts; the required rows re-executed there

The candidate containing the completed X-25 repair and X-27 with its
correction (`cd744bb`) was archived into an sdist and built into wheels
(local `83fd95dd...`, zhang-x4 `65604d75...`), installed into clean venvs
and exercised with the tests and benchmarks copied out of the checkout:
the round 8 probes, the installed test subsets, the shared-semantics suite
(137 passed; the one failure is the recorded V1-19 under numpy 2.5.4), the
Gemini-II card tests (15 passed), the fourteen AiM rows (cycles, trace
digests, line and opcode counts and lowering manifests identical to the
`54cbb08` run), the three Samsung rows (the same default and
vendor-equivalent cycles), the twelve AttAcc results (every selection,
cycle count, trace check and composed value equal to round 57; only the
evidence paths, the config digests that embed the temporary trace path and
the simulator wall times differ) and one pre-registered Gemini-II paired
session of 20 rounds, run because the paper-row driver assigns the
repetition count through the public setter X-27 changed: plans, layout and
program fingerprints and packing attestations identical to the `54cbb08`
sessions, every output exact; gemm, gesummv and bicg at or below the
paper, batched_gemm (1.0003) and mvt (1.0002) nominally above it in this
session.  Under the pooled-session rule mvt is OPEN (nominal) at `cd744bb`
(REPRODUCED at `54cbb08` over four sessions); no further session was added
at this revision.  Required set: 46 REPRODUCED, 9 OPEN (X-19 x4, X-22 /
X-23 x3, X-26, mvt at this revision).  Still not executed at the
candidate: the sixteen UPMEM rows (the host's Docker data root) and the
five Gemini-I rows (the campaign scripts re-pointed at `cd744bb`, the
watcher waiting for the host's `/tmp`).  The AttAcc stage of the chain
first failed on my sed-adapted script, which named a benchmarks copy that
did not exist in the new evidence directory (kept under `r59/aborted/`);
it was rerun with the copy in place.

## Round 61 (2026-11-05): the APU v1 grouped route on the candidate contract (X-28)

The reviewer's round 60 probe showed that the advertised APU v1 dataflow
route had no candidate contract: its enumerator returned `grouped_f16`
placements without effects, an ordinary supported region failed public
compilation from the `cd744bb` wheel with "declares no effects", and the
context's realization check returned for grouped programs.  Repaired as a
bounded G9 increment: a route module derives every grouped candidate's
coverage, layouts, allocations, transfers, launch and ABI from the work
body's retained facts and the group layout and registers as the target's
effects deriver; the enumerator attaches the declaration (and refuses a
non-MAC body by name); the context records what its emitters emit and
compares the SPMD body with every work item's declaration at
materialization (fault injection into the group tree is refused); the
runtime re-derives the packing from the facts, requires the declared
geometry and batch count, binds the caller's arrays by retained identity
and refuses a program without a device packing with the reason.  The
derivation exposed a disagreement between the old compile-time batch count
and the runtime's packing for a region GEMM with more rows than APUCs; both
are now the one derived count.  Regressions on the reviewer's source, a
renamed program and a nearby GEMM region, the runtime binding, the explicit
restrictions, emission fault injection and an edited declaration (section
u, 7 tests).  Nothing was fixed by removing shared facts, exempting grouped
candidates, deleting the route or forwarding it into another compiler.  Not
board-verified (the host's `/tmp`); `cd744bb` predates this repair, so the
next candidate must include it.

## Round 62 (2026-11-05): the executable template of the grouped route proven, not assumed (X-29)

The reviewer showed that round 61 declared a region executable (`gemm`
packing) from buffer shapes, element type and reduction extent alone: a
reversed reduction access, a half-row or half-column domain, a repeated
or permuted row access all kept the same shapes, selected the packing and
emitted the same device C as the dense base while their source semantics
differed; the realization check compared registers and group sizes, not
the accesses.  Repaired in production: the dense GEMM template is proven
structurally from the retained access relations of every work item of the
kernel group (operands at `[r, k]` and `[k, c]` with the reduction in lane
order, the result at `[r, c]` with the operands' own coordinates, one
rectangular block per work item, the blocks a disjoint cover of the whole
output, a zero accumulator, FP16, the shapes), the ranking stamping every
kernel group's facts on its matches before enumeration so the deriver, the
enumerator, the named-candidate path and the runtime read the whole group.
A region that fails a condition is declared `virtual_only` with the
condition named and is refused before scheduling for an executable backend
(the functional backend, which has no model of the grouped program, is
refused as such); the runtime packs from the proven template only.
Regressions: the reviewer's five variants plus overlapping work items,
the canonical and a column-block partition, the named path.  X-28's
closure text is corrected and X-29 recorded as the linked finding.
Nothing was exempted, erased or removed.  Not board-verified; `cd744bb`
predates X-28 and X-29, so the next candidate must include both.

## Round 63 (2026-11-05): integration candidate `8795e8f` validated from installed artifacts

The candidate containing X-28 and X-29 was archived, built into wheels
locally and on zhang-x4, installed into clean venvs and exercised with the
tests and benchmarks copied out of the checkout: probes (UPMEM still
Docker-blocked), the shared-semantics suite from the wheel (151 passed,
the one failure V1-19), the grouped route's fourteen section u tests from
the wheel, the per-file subsets, the fourteen AiM rows and three Samsung
rows (identical to the two previous candidates), the twelve AttAcc results
(identical), the Gemini-II card tests (15 passed) and a compile-only check
that the five Gemini-II paper kernels compile from the new wheel to the
same plans, estimates and launch fingerprints as the `cd744bb` session (no
session launched).  The grouped route's board validation is written as an
installed-wheel script (three GEMM regions, multiple batches, a
neighbouring shape and a column-block partition, exact outputs) and waits
with the Gemini-I campaign for the host's `/tmp`; it is recorded as an
explicit open verification blocker, not as completed hardware validation.
Required set unchanged: 46 REPRODUCED, 9 OPEN.

## Round 64 (2026-11-05): the grouped route's batch count validated at the shared boundary (X-30)

The reviewer showed that a grouped candidate whose device loop covered
fewer batches than the proven packing needs passed the contract and the
materialization check, because the declaration's batch count came from the
placement's own field and the emission check compared against that same
field; only the runtime caught it.  Repaired: the declaration's batch count
is the proven packing's for an executable GEMM, so an under-batched body
disagrees with its own declaration and is refused at materialization as a
per-candidate feasibility rejection (normal search keeps and selects the
intact candidate, a forced one fails during compilation); the check also
re-derives the packing from the kernel group's facts independently of the
declaration; an edited declaration is refused by the contract; the runtime
guard stays.  Two regressions (supplied-layout and normal-selection paths;
section u now 16).  Not an installed-candidate repair yet (`8795e8f`
predates it); the grouped board validation and the Gemini-I campaign still
wait for the host's `/tmp`.

## Round 65 (2026-11-05): the board checker corrected, the candidate `c43061f` validated, the X-19 search bounded

The reviewer showed that the queued grouped-route board checker exited 0
on compilation failures and inexact outputs and that the watchers
discarded the child status.  Replaced: the checker keeps every record,
archives the generated project with the host's packed inputs, the raw
output image, the inputs, outputs, oracle, profile text and kernel source
beside the wheel and revision identity and the binary digest, and exits 0
only when all three cases ran on the device with four fresh counters and
exact outputs; its self-test refuses the negative records; the watchers
and the Gemini-I campaign script propagate status while keeping every
session.  The candidate `c43061f` (X-30 included) was built and validated
from installed artifacts: probes, the shared suite (153 passed, V1-19 the
one failure), the sixteen grouped-route tests from the wheel, the subsets,
the AiM, Samsung and AttAcc rows identical to the three earlier
candidates, the card tests and the compile-only Gemini-II equivalence.
The X-19 study simulated the four symmetric candidate choices of the RoPE
kernel groups natively for the RoPE row and a full-block row: the default is
the fastest of those and all are above the paper's digits, so the four rows
stay OPEN (round 65 overclaimed this as the exhausted space; round 66
measures all 16 pairs for all four rows).  Hardware validation of the
grouped route and the Gemini-I campaign remain blocked by the host's
`/tmp`, UPMEM by Docker; the watchers now target the `c43061f` wheel.

## Round 66 (2026-11-05): the grouped placement's map must be the executable's (X-31); the X-19 overclaim corrected

The reviewer showed that the grouped route validated a placement's group
cardinalities but not its physical map: an interleaved and a collapsed
map with the canonical cardinalities compiled to the canonical device
program and packing, and an injected interleaved clone was selected in
normal search.  Repaired: `grouped_map_problem` states the executable's
map (contiguous groups: lane bit b to physical bit b, group bit b above
the group), the deriver declares every packing invalid when the map is
another so the contract refuses the candidate per candidate (a supplied
one with the reason, an injected one filtered while the canonical
candidate is kept and selected), and the realization check refuses such a
map at materialization too; canonical layouts and their movement
annotations are unchanged.  Regressions on both routes (section u, 19).
The round 65 X-19 wording overclaimed four symmetric choices on two rows
as the exhausted faithful space; it is corrected in place, and all 16
pairs of the two RoPE groups' candidates are now simulated for all four
rows with the default pair the minimum in each and every pair above the
paper's digits (OPEN, with a bound over the RoPE groups' joint candidate
space).  Not yet in an installed candidate; the grouped board run, the
Gemini-I campaign and UPMEM stay blocked by the host.

## Round 67 (2026-11-05): candidate `f418762` validated from installed artifacts; UPMEM through a native SDK executor

The candidate containing X-31 was built into wheels locally and on
zhang-x4, installed into clean venvs and exercised from outside the
checkout: probes, the shared suite (154 passed, V1-19 the one failure),
the nineteen grouped-route tests and the reviewer's supplied / injected
map probe from the wheel, the subsets, the AiM, Samsung and AttAcc rows
identical to every earlier candidate, the card tests and the compile-only
Gemini-II equivalence.  The host's Docker data root has been full since
2026-11-04, so a native DPU SDK executor was added (an environment
selection like the native AiM executor): the same SDK compiler runs
through an unprivileged user-namespace bind at the image's path, the gate
and run records name it, the pinned manifest check is unchanged.  All
sixteen UPMEM rows reproduce the paper's cycles exactly with NumPy-exact
outputs at the checkout through it; this is checkout evidence until the
next installed candidate contains the executor.  The watchers now target
the `f418762` wheel; `/tmp` is still full.  Required set unchanged:
46 REPRODUCED, 9 OPEN.

## Round 68 (2026-11-05): candidate `ba2f58d` harvested (UPMEM rows from the installed wheel); U-16 repaired

The `ba2f58d` chain was harvested: probes, the shared suite (156 passed,
V1-19), the grouped tests and the reviewer's map probe from the wheel, the
subsets, the AiM, Samsung and AttAcc rows identical to every earlier
candidate, the card tests, the compile-only Gemini-II equivalence, and for
the first time since `0e48d81` the sixteen UPMEM rows from an installed
wheel, through the native SDK executor.  A compile-only ranking shows no
Gemini-II plan estimated cheaper than the selected one for any paper
kernel, so the nominal rows have no production selection opportunity in
the route's candidate space (the RoPE rows' grid was completed in round
66).  U-16 is repaired: the UPMEM family recognizers read the matcher's
resolved affine coordinates instead of raw index names, so a row base
hoisted into an address-only temporary classifies like the inline
spelling, and the portable-C host oracle accepts the index cast of the
frontend's 65-bit sum as the truncation it is; both forms compile to the
same device source and object and simulate to the same cycles with exact
outputs.  Board and Gemini-I work stay blocked by `/tmp`.

## Round 69 (2026-11-05): load identity on full resolved coordinates (U-16 completed)

The reviewer showed that round 68 fixed the operand geometry but left the
selection and histogram families comparing the guard's load with the
copied operand by raw index names: an equivalent hoisted selection was
refused while a guard over another element was accepted and failed in
packing.  Repaired in the shared retained analysis and the recognizers:
load identity now compares the full resolved coordinates of both sides,
including affine maps and constants, through the engine's own coordinate
derivation; the geometry projection stays a projection; the engine's term
equality treats loads with equal resolved coordinates and version as the
same element.  Equivalent spellings compile and execute alike on the native
simulator, differing guard and copy accesses are refused before selection,
and the round 68 closure text is narrowed.  Checkout evidence; the next
installed candidate must include it.  Nothing else changed: hardware work
stays behind `/tmp`, the nine required rows stay OPEN.

## Round 70 (2026-11-05): the grouped route validated on the board from the installed `472bec9` wheel; the Gemini-I campaign still held by `/tmp`

The temporary-space dependency was examined rather than waited on: my own
scratch was removed (other users' files and the user's own shells hold the
rest; the vendor tool ignores `TMPDIR`), the runtime's device-log drain
then worked, and the grouped APU v1 route was validated on the board from
the installed wheel of `472bec9`: three dense FP16 GEMM regions (row
blocks and column blocks with eight batches per APUC, a neighbouring 12 x
17 x 5 shape with four) built, launched under the board lock with fresh
profile records, exact outputs, binaries, projects, inputs, outputs and
profiles archived beside the wheel identity; the checker's verdict PASS.
That closes the hardware validation of X-28 to X-31 at an installed
candidate.  The forty grouped-route and U-16 regressions pass from the
same wheel.  Other users consumed the freed space during that stage, so
the pre-registered Gemini-I campaign was refused by its guard and stays
queued behind a watcher with an attainable threshold and per-session
guards; the X-22 / X-23 rows stay OPEN.  P-24 (the lint chain's header
stop) recorded and deferred.

## Round 71 (2026-11-05): replicated AiM elementwise grids realized once over the device (X-32); X-19 below the paper with the complete program

The round 70 direction was a bounded production improvement for the X-19
rows through `allo/spmw_aim.py` and its lowering.  Reading the corrected
`qkvo_rope_l128` program's lowering showed that every elementwise group
replicated with a neighbouring contraction (the RMSNorm scale and norm,
RoPE, the softmax multiply, SiLU and the gate) was realized once per
replica: four copies of each host write, each EWMUL / EWADD and each
readback for one vector of work, as CENT's expert traces also do.  The
derivation now realizes such a group once over the device's channels, the
knob domain offers that form, a reduced-copy producer whose device reader
needs another layout is a feasibility rejection, and the realization check
compares the realized replicas and channels with the single-copy
expectation (X-32).  Selection chooses the form by estimated cost, and the
estimates order the forms as the simulator does.  Every CENT row was
re-simulated natively: the four X-19 rows are below the paper's digits
with the full D2-element multiply, CENT's host protocol and the timing
boundary kept (123,448 / 397,080 / 408,823 / 529,019 against 129,605 /
421,159 / 446,803 / 697,316), six more rows are faster than the paper, and
the four rows whose cycles are unchanged have traces that are the earlier
pins shifted down by the three rows the smaller elementwise regions free
(verified byte for byte).  The neighbouring D = 2048 block shows the same
gain.  The golden rows keep the paper values as the baseline and pin the
selected programs; no expected value was moved toward the paper's digits
and no omitted work restored.  X-19 is REPRODUCED at the checkout; the
installed-candidate re-execution is the next step.  The Gemini-I campaign
stays queued behind the `/tmp` watcher; X-22 / X-23 / X-26 and mvt stay
OPEN.  NOT READY.

## Round 72 (2026-11-05): every device-resident value keeps its writer's distribution (the reviewer's fan-out and inverse probes)

The reviewer showed, with ordinary source and candidates from production
enumeration, that the round 71 compatibility check was partial: it stopped
at the first reader and ran only when the producer was the reduced-copy
side.  A fan-out whose second reader differs and a producer whose readers
differ were both accepted, and the address witnesses show the second reader
addressing data the writer never placed in its bank partition.  The repair
is a walk of the schedule before any record is lowered: per buffer, the
distribution its last writer left on the device (a launch's store, or a host
scatter in the layout it is bound to; a host stage producing the buffer ends
its residency), and a feasibility rejection of the candidate assignment at
the first device read in another distribution, for every reader of every
live range and whichever side is reduced, because no redistribution between
slot layouts is lowered.  Relocations now serve every distinct reader slot
of the live range, not the first reader's.  Both probes are refused, the
default and the all-per-item alternative compile, a host reupload between
readers ends the live range, and the fourteen golden rows and the CENT suite
are unchanged.  Recorded under X-32; G4 and G9 updated.  The installed
candidate `2096239` (wheel `803d1ea4...`) re-executed the fourteen AiM rows,
the CENT suite, the value-layout regressions and the reviewer's probe
from the wheel with identical results.  (Round 73: this closure was
premature; the relocations were not lifetime-safe, see round 73.)  X-22 /
X-23 / X-26 and mvt stay OPEN.  NOT READY.

## Round 73 (2026-11-05): physical liveness of bank slots (the reviewer's slot-liveness witness); X-32 reopened and repaired

The round 72 closure of X-32 was wrong, and the reviewer's one-element
witness showed it on my own test program: with every elementwise group in
one bank row, the copies placed right after the producer and the original
result were overwritten by the next scatters and the first reader before
the second reader ran, so the generated records computed `d * c` where the
source says `d * t`.  The distribution check tracked logical buffers, not
cells, and the test counted copies.  The repair changes the allocation and
the materialization: every elementwise group has a bank row region of its
own, a value is copied into each reader's region right after its writer
while the source is live, and the emitted records are walked cell by cell
(rows, bank slots, the global buffer) so that every EWMUL operand, copy
source and gather finds the value program order placed there; a mismatch is
a feasibility rejection of the candidate assignment.  The witness now gives
the source semantics, the new regressions are address/value witnesses over
several compositions rather than counts, and the production check refuses
the old one-row allocation when forced.  The CENT rows re-simulate at or
below the paper; four traces return to their typed-route pins and the
RMSNorm and FFN chains carry their copies across regions at a few hundred
cycles.  Round 72's G4 and G9 statements are corrected below.  The installed
candidate `26fd884` (wheel `4bf38997...`) re-executed the fourteen rows, the CENT
suite, the value-layout regressions and both reviewer probes from the wheel
with identical results.
(Round 74: this closure was premature for vectors spanning several bank
rows; see round 74.)
X-22 / X-23 / X-26 and mvt stay OPEN.  NOT READY.

## Round 74 (2026-11-06): multi-row relocations move one row chunk at a time (the reviewer's multi-row witness); X-32 reopened and repaired again

The reviewer's command-level witness showed that the round 73 relocations
were right for one row and wrong for several: a whole-vector copy record is
expanded row by row, the global buffer holds one row, so the last source
row was written into every destination row, and the round 73 liveness
check could not see it because it tracked which value the GB held, not
which row chunk.  The repair emits the copies one row chunk at a time
(source row r to the GB, then destination row r, then the next chunk),
binds each record to its chunk, and makes the liveness model track row
chunks in every cell and in the GB, walking each copy record row by row as
the lowering does; a whole-vector record is refused.  The priced work is
unchanged.  The witness now gives the source semantics; the regressions
are command-level over every row with values that differ from row to row,
a partial final row, both layouts and the chain.  None of the fourteen
CENT rows relocates a multi-row vector, so their traces, pins and cycles
are unchanged.  Round 73's closure statements are corrected below.  The installed
candidate `db2ad84` (wheel `7c8734a7...`) re-executed the fourteen rows, the CENT
suite, the value-layout regressions and the three reviewer probes from the
wheel with identical results.
(Round 75: this closure did not cover readers of another logical frame;
see round 75.)
X-22 / X-23 / X-26 and mvt stay OPEN.  NOT READY.

## Round 75 (2026-11-06): the logical frame is part of the value-layout contract (the reviewer's prefix witness); X-32 reopened and repaired again

The reviewer's prefix composition passed every round 72 to 74 check:
producer and consumer agreed on copies, channels, bank family and row
chunks, yet the consumer covered half the vector and so assigned different
elements to the same bank partitions; the partition-preserving copy
delivered the wrong elements.  The contracts now carry the logical frame
(first element, extent, elements per bank partition) of every slot layout:
a device reader whose frame differs from its writer's is refused before
selection, naming both frames and the remedy (equal coverage, or a host
scatter of the region the reader needs), and the physical cells carry the
frame as well.  No redistribution between bank partitions is realized,
because none exists on the device (the global buffer is per channel) and a
host round trip is the program's to write.  Equal-frame compositions and
host-scattered prefixes are unchanged.  The regression witness now walks
every channel, bank and row with values that differ per partition and per
row chunk.  None of the fourteen CENT rows composes device values of
different frames, so no trace, pin or cycle changed.  Round 74's closure
statements are qualified below.  The installed candidate `b221972` (wheel
`ace1b7b0...`) re-executed the fourteen rows, the CENT suite, the value-layout
regressions and the four reviewer probes from the wheel with identical results.
X-22 / X-23 / X-26 and mvt stay OPEN.  NOT READY.

## Round 76 (2026-11-06): the elementwise lowering proves a dense domain (the reviewer's stride-two witness, X-33); the Gemini-I campaign running

The reviewer found a pre-existing defect independent of X-32: the AiM
elementwise lowering took its element count from the loops' upper bounds
and treated the access bounding box as an exact image, so a stride-two
kernel lowered to dense EWMULs that overwrote the lanes the source leaves
unchanged.  The lowering now proves the domain first (loops stepping by
one, result coordinates one loop variable each with coefficient one, every
loop indexing the result once, the count from the trip counts) and refuses
strided or repeated stores before selection with the loop or coordinate
named; dense kernels and dense suffixes are unchanged, and no required
paper row has a strided elementwise domain.  Recorded as X-33 under G2 and
G9, with the reviewer's failing witness preserved.  The queued Gemini-I
campaign launched itself from the round 69 watcher when the host's `/tmp`
freed (348 MB at launch); it runs from the installed wheel of `472bec9`,
whose APU v1 code is unchanged since, under the pre-registered three-session
protocol; its sessions are pooled when it finishes.  It did not get
there: the host's `/tmp` refilled within minutes and session 1 lost its
launches to the vendor tool's scratch (kmeans kept eight of ten measured
launches per lane, the other two cases none), sessions 2 and 3 were
refused by the guard, the record is kept and not pooled, and a retry of
the same protocol is queued at a higher threshold.
X-26 and mvt stay OPEN.  NOT READY.

## Round 77 (2026-11-06): the Gemini-II batched_gemm and mvt gaps attributed to measurement drift (no new launches)

The remaining Gemini-II gaps are half a tick and six tenths of a tick at
`cd744bb`.  Every recorded session was re-read against its own re-run GTML
lane: the generated work is one program throughout (one plan, one program
and layout fingerprint, one packing attestation since round 30; the
ranking's only alternatives are 49 and 100 percent more expensive by
estimate), the statistic is device pipeline ticks per body with host
launch overhead outside it, and what moves between sessions is the card.
batched_gemm's identical program measured from 0.987 to 1.003 of the
paper's column across seven sessions while the GTML lane in the same
rounds measured 1.002 to 1.007 of its own column; the vendor's drift
exceeds Tenon's in five of six paired sessions and the paired same-round
GTML/Tenon ratio is at or above the paper's in five of six.  mvt pools
below the paper over the six paired sessions, its paired ratio within 0.1
percent of the paper's.  The gaps are measurement drift the baseline
shares, not generated work, selection or launch overhead; no mechanism
repair is supported, no session is added to move a median, and the
statuses stand under the pooled-session rule until the final-candidate
validation session.  The stale 46 / 9 summary in STATUS.md is refreshed
to the recorded 50 / 5.  The Gemini-I retry stays queued behind `/tmp`.
NOT READY.

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

Refreshed in round 76 (2026-11-06); the integration candidate `b221972` (round 75, installed wheel `ace1b7b0...`) plus the round 76 checkout (X-33: the elementwise lowering proves a dense domain; X-32 repaired with physical liveness, row chunks and logical frames included; the fourteen AiM rows at or below the paper; X-19 REPRODUCED with the complete program; the grouped route's board validation and the installed grouped / U-16 regressions at `472bec9`); the candidate `472bec9`
(X-25 completed, X-27 and its correction, X-28 to X-31, the native UPMEM SDK
executor, U-16): the grouped route validated on the APU v1 board and the
grouped and U-16 regressions run from its installed wheel; the simulator rows
and the Gemini-II card tests last run from the `ba2f58d` wheel; the grouped route's board validation is complete (round
70); the Gemini-I campaign waits for the host's `/tmp` behind a guarded, status-propagating watcher. Earlier judgments are kept in the round sections above.

| Gate | Judgment | Basis |
|---|---|---|
| G1 correspondence | **PARTIAL** | every substantive mechanism of Sections III and IV has a maintained implementation reachable through `allo.compile` (CLAIMS.md), now including both APUs through the shared pipeline; the claim matrix must be re-read against the G9 pipeline and the G10 inventory (same-or-better performance is part of G1 since the 2026-11-03 scope addition) |
| G2 semantic preservation / explicit failure | **PARTIAL** | the rounds 8-11 repairs (C-19 to C-26) and the shared contract rounds (X-02 to X-14, X-17 to X-20) are implemented with regressions; round 76: the AiM elementwise lowering proves a dense iteration domain and store image before it is used and refuses strided or repeated stores before selection (X-33, the reviewer's stride-two witness); the shared semantic checks run before every backend's selection; round 62: the APU v1 grouped route refuses, before scheduling and by name, a region whose accesses or written domain are not the dense GEMM its device program executes (X-29); open: the deferred arithmetic findings (ISSUES), independent re-challenge of the whole domain on every runner |
| G3 general mechanisms | **PARTIAL** | structural recognizers and template-derived legality on every backend; the AttAcc attention template is proven structurally (X-20); the UPMEM recognizers read resolved coordinates so hoisted and inline address spellings classify alike, and guard / copy load identity is proven on full coordinates (U-16, rounds 68-69); renamed and neighbouring programs compile to the same normalized decisions in the round 24-28 regressions; open: M-14, the constants inventory re-read against the current tree |
| G4 search/layout/selected-program integrity | **PARTIAL** | complete assignments materialize under the real orchestration before acceptance (C-33b); round 72: an AiM assignment whose device reader takes a value in another distribution than its last writer left is a feasibility rejection for every reader of every live range (X-32, the reviewer's fan-out and inverse probes); round 73 (the reviewer's slot-liveness witness showed the round 72 relocations were not lifetime-safe): every elementwise group in its own bank row region, copies into each reader's region while the source is live, and the emitted records walked cell by cell so every slot read finds the value program order placed there, else a feasibility rejection; round 74 (the reviewer's multi-row witness): relocations one row chunk at a time through the global buffer, the liveness model tracking row chunks in cells and the GB, whole-vector copy records refused; round 75 (the reviewer's prefix witness): the logical frame and partition span of every slot layout are part of the compatibility and cell contracts, a reader of another frame refused before selection with both frames and the remedy named; every backend's realization is compared with its declaration (X-05 to X-16; the APU v1 grouped route's map, batches, packing and emission since rounds 61-66, X-28 to X-31); transfer availability and physical addressing are checked (X-09 to X-11, X-17, X-18); open: the slice 5 forced-alternative traces through packing and launches on both hardware hosts |
| G5 cost models | **PARTIAL** | provenance and validity labels everywhere; the AiM single-copy elementwise form selected by estimated cost and confirmed by native simulation on every CENT row and a neighbouring block (round 71); cost perturbations traced to executed programs on Samsung, UPMEM and APU v1 in earlier rounds; the Samsung estimate includes each work item's share of the first matrix staging (partitioning the conductor's staging exactly, staging before the MAC, the same scope for the ordered program) and ranks the swizzled-rows placement first on both paper GEMVs, a held-out and a neighbouring shape, as the conductor measures (rounds 37-38, S-16 repaired); the AiM contraction profile recalibrated per command class from controlled simulator measurements and validated on held-out programs before the nine workloads (round 50, M-19 awaiting review); open: C-18, uniform movement weights, the perturbation traces at the current candidate on both hardware hosts |
| G6 evidence beyond examples | **PARTIAL** | EVIDENCE.md plus the reviewer-driven probes of rounds 8-27 (renamings, operation changes, shape and slice neighbours, wrong declarations, reversed arithmetic); open: hardware-only confirmations, the probe list re-run at the candidate |
| G7 install/interfaces/maintainability | **PARTIAL** | installed-artifact runs at `7bdb475`, `e60973f`, `9255d29`, `54cbb08`, `cd744bb`, `8795e8f`, `c43061f`, `f418762`, `ba2f58d`, `472bec9` (round 70), `2096239` (round 72), `26fd884` (round 73), `db2ad84` (round 74) and the candidate `b221972` (round 75: sdist and wheel from the clean tree, clean venv, the fourteen AiM rows, the CENT suite, the value-layout regressions and the four reviewer probes from the wheel) (round 68: sdist and wheels, clean venvs, probes and test subsets including the grouped route's section u and the reviewer's map probe, the card tests, the UPMEM rows through the native SDK executor); architecture docs follow the pipeline (`design/shared-compilation-path.md`); open: the Samsung patches (P-04), P-23 lint, the public-payload review |
| G8 backend execution | **PARTIAL** | installed-artifact evidence on the simulators at `69773fc` (Samsung, AiM, UPMEM probes, no checkout module loaded) and on the APU v1 board and the Gemini-II card at `69773fc`/`28cc922` (round 30: the archived paired protocols with the baselines re-run, exact outputs, device locks, launch provenance on every route); `tests/perf` at `69773fc` (545 passed, the one failure an NFS-tmpdir artifact fixed in `d2e3fda`), `tests/pim` at `69773fc` 564 passed with 20 Docker-backed simulator tests failing on the host's full `/var` (containerd ENOSPC, an administrator matter; the same rows passed from the installed wheel earlier in the round); open: the 20 simulator tests re-run when the host has space, the suites and installed-wheel checks at the final round 30 candidate, the ledag scratch fix (X-24) exercised only through its unit test and the sessions 3 and later; round 60: installed-wheel probes and test subsets at `cd744bb` on the simulators (the UPMEM probe still blocked by the host's Docker data root) and the Gemini-II card tests and one paired session from the `cd744bb` wheel |
| G9 one shared compilation pipeline incl. both APUs | **FAIL (in progress)** | shared front stages and facts (rounds 12-18); both APUs' plans and typed records as candidates of the common pipeline with an enforced contract (rounds 13-17); Samsung, AiM, UPMEM and AttAcc declare effects from the facts and are checked at materialization (rounds 19-27, X-04 to X-20); shared transfer availability (X-09, X-17, X-18); the `APUv1Program` manifest from the retained IR (round 28); slice 5 traces at `653b059` from installed wheels (round 29); round 30: the paper rows measured through default selection from installed wheels under the archived protocols with launch provenance on every route; round 54: Gemini-II launch recipes bound to the retained contract and the launch partition at materialization (X-25, found by the reviewer's fault injection; a mismatch is the shared feasibility rejection); round 58: the public repetitions update re-derives the validated plan's launches from their own partitions, validates, prices and installs atomically (X-27, the reviewer's second fault injection; round 59 correction: the graph check actually runs on the public route); round 60: the repaired setter exercised from the installed wheel of `cd744bb` on the card through the paper-row driver with identical plans and fingerprints; round 61: the APU v1 grouped route of a dataflow region declares its effects from the shared facts, is checked at materialization and binds its runtime by retained identity (X-28, the reviewer's public-source probe); round 62: its executable dense GEMM template proven from every work item's retained accesses and the written domain before it is declared executable, non-template regions refused for execution with the failing condition (X-29); round 63: X-28 and X-29 exercised from the installed wheel of `8795e8f` (section u, 14 passed) with the board validation of the grouped GEMM runtime pending on the host's `/tmp` (an explicit blocker); round 64: the grouped candidates' batch count validated against the proven packing at materialization, under-batched candidates feasibility-rejected (X-30, the reviewer's damaged-candidate injection); round 65: X-30 exercised from the installed wheel of `c43061f` (section u, 16 passed) and the board checker corrected to fail on any incomplete case; round 66: a grouped placement's map must be the contiguous-group map the executable realizes, other maps refused per candidate (X-31, the reviewer's interleaved and collapsed maps); round 67: X-31 exercised from the installed wheel of `f418762` (section u, 19 passed; the reviewer's map probe refused from the wheel); round 70: the grouped route executed on the board from the installed wheel of `472bec9` (three GEMM regions, multiple batches, a column-block partition and a neighbouring shape, exact outputs, fresh profile records, archived provenance); round 71-73: the AiM single-copy elementwise realization (X-32) with producer-consumer distribution compatibility established over every live range before lowering (the reviewer's round 71 probes refused per candidate) and, after the reviewer's round 72 witness showed the relocations were not lifetime-safe, per-group bank row regions, copies into the readers' regions and a cell-level liveness check of the emitted program (round 73), and row-chunked relocations with row chunks tracked after the reviewer's multi-row witness (round 74), and the logical frame of every slot layout in the contracts after the reviewer's prefix witness (round 75); open: independent review of rounds 28-30, the full suites and installed-artifact runs at the final candidate, the remaining minor items |
| G10 paper results through the shared pipeline | **FAIL (in progress)** | inventory of 83 rows (`paper_reproduction/inventory.csv`); measured through default selection: Samsung 3/3 faster than the paper (outputs checked), UPMEM 16/16 exact, AiM 10/14 exact with the 4 RoPE rows an OPEN 32-cycle regression (X-19), Gemini-II 5/5 REPRODUCED under the archived paired protocol with the GTML baseline re-run in the same rounds (round 30), Gemini-I under the archived paired protocol with the expert binaries re-run: 2/5 REPRODUCED (matrix multiply, histogram), kmeans and linear regression OPEN (nominal) on the pooled round 30 sessions (X-23; the single-session closure of kmeans was withdrawn in round 31), word count attributed to the generated program's loop-counter width and repaired (X-22, `dc41df7`): the repaired program at parity with the paper's own binary in the same rounds and 0.19% above the paper's archived median, OPEN (nominal); the Gemini-I vector GEMM layout row faster than the paper (round 31); the portability row REPRODUCED on both targets with the paper's inputs through the maintained in-repo driver (rounds 34-35, checkout evidence); the Samsung XOR row OPEN (measured): the identity-rows and F2-swizzled-rows placements are ordinary-source candidates executed by the conductor, the XOR packing 1.12-1.13x faster and bit identical on two GEMVs, the paper's within-round probe metric unreachable by the production program (round 36); the selected map made the contract and the default selection earning the swizzled rows on the paper shapes and a held-out shape (round 37); the 500-instance selector equivalence as a diagnostic; round 51: the six AttAcc kernel-tiling results REPRODUCED through the shared pipeline on the natively built Ramulator2-AttAcc (QKV / projection at the paper's values with fixed baselines re-run; score 1.00x / 1.13x over fixed as plotted; 123 / 316 / 61 against DCC's archived per-shape times, median loss 7.07%), every trace checked against the selected program command by command (round 52); round 52: the six decode configurations REPRODUCED as an analytical composition from the shared-pipeline kernel times (Tenon, DCC and fixed variants re-run through DCC's model; the frozen figure, table and text values); required set 48 of 55 REPRODUCED, 7 OPEN (X-19, X-22/X-23), 0 UNVERIFIED; rounds 56-57 at the integration candidate `54cbb08`: Gemini-II re-measured from the installed wheel over four paired sessions (four REPRODUCED, batched_gemm OPEN (nominal), X-26), the fourteen AiM rows and three Samsung rows re-executed from the installed wheel (10 exact, 4 corrected RoPE programs OPEN, Samsung faster), the Gemini-I campaign pending; historical coverage 47 of 55 REPRODUCED, 8 OPEN; the twelve AttAcc results re-executed from the installed wheel with identical results; at the candidate itself the UPMEM and Gemini-I rows not yet re-executed; round 60 at the candidate `cd744bb` from installed artifacts: the fourteen AiM, three Samsung and twelve AttAcc results identical to the `54cbb08` runs, Gemini-II in one pre-registered session (gemm, gesummv, bicg at or below the paper; batched_gemm 1.0003 and mvt 1.0002 nominally above, mvt therefore OPEN (nominal) at this revision under the pooled rule), required set 46 REPRODUCED, 9 OPEN, UPMEM and Gemini-I still not re-executed at the candidate; round 63 at `8795e8f` from installed artifacts: AiM 14, Samsung 3 and AttAcc 12 identical again, the Gemini-II kernels compiling to the `cd744bb` session's plans and fingerprints (no new session), UPMEM and Gemini-I still blocked; round 65 at `c43061f` and round 67 at `f418762`: the same three sets identical again and the same Gemini-II equivalence; round 67: the sixteen UPMEM rows exact at the checkout through the native SDK executor (the Docker data root being full); round 68: the same sixteen rows exact from the installed wheel of `ba2f58d`, and no Gemini-II plan estimated cheaper than the selected one for any paper kernel; the X-19 RoPE rows: all 16 candidate pairs of the two RoPE groups simulated natively for all four rows in round 66 (the default pair the minimum in each, every pair above the paper's digits: OPEN with a measured bound over the RoPE groups' joint candidate space); round 71 at the checkout: the replicated elementwise grids realized once over the device (X-32), every AiM row's selected program at or below the paper (four exact, ten faster; the four RoPE rows 123,448 / 397,080 / 408,823 / 529,019 with the complete program), X-19 REPRODUCED; round 72 at the candidate `2096239` from installed artifacts: the fourteen AiM rows, the CENT suite, the value-layout regressions and the reviewer's fan-out probe from the wheel; round 73 at the candidate `26fd884` from installed artifacts (`r73/installed_run.log`, `r73/aim_rows_provenance.json`): the fourteen rows at the per-group-region pins (norm_residual 7,052, ffn_activation 3,672, ffn_complete 254,166, full_block 397,577 / 409,320 / 529,516, the others unchanged), the CENT suite, the value-layout regressions, the fan-out probe refused and the slot-liveness witness at the source semantics from the wheel; round 74 at the candidate `db2ad84` from installed artifacts (`r74/installed_run.log`, `r74/aim_rows_provenance.json`): the fourteen rows unchanged, the CENT suite, the value-layout regressions and the three reviewer probes from the wheel; round 75 at the candidate `b221972` from installed artifacts (`r75/installed_run.log`, `r75/aim_rows_provenance.json`): the fourteen rows unchanged, the CENT suite, the value-layout regressions and the four reviewer probes from the wheel; required set at the latest recorded revisions 50 REPRODUCED, 5 OPEN; round 77: the Gemini-II batched_gemm (X-26) and mvt gaps attributed, from every recorded session against the re-run GTML lane, to the card's session-to-session drift (identical program, plan and fingerprints; no cheaper feasible plan; host overhead outside the tick statistic), statuses unchanged, no session added; optional: token partitioning DIAGNOSTIC, UPMEM scaling, packing/selection/DCC studies, calibration, the virtual study, compile cost UNVERIFIED |

Verdict: **NOT READY**. Blockers: G9 (independent review of rounds 28-30,
the suites at the final candidate), G10 on its narrowed required scope
(user, 2026-11-04: the Samsung, AiM, UPMEM, Gemini-I and Gemini-II kernel
comparisons and the AttAcc kernel-tiling and decode figures; the inventory's
`release_scope` column): the four AiM RoPE rows X-19 (REPRODUCED at the candidate `b221972` from its installed wheel through
X-32, the complete programs below the paper's digits), the three nominal Gemini-I
rows X-22/X-23 awaiting their pre-registered confirmation campaign at the
candidate (blocked since round 56 by the host's full `/tmp`; re-pointed at `472bec9` in round 70 with an attainable watcher threshold and per-session guards), the nominal
Gemini-II batched_gemm row X-26 and the mvt row nominal at `cd744bb` in its one
pre-registered session, and the Gemini-I rows not re-executed at the candidate (the AiM, Samsung,
AttAcc and, since round 68, UPMEM rows were, from the installed wheel of
`ba2f58d`; Gemini-II from `cd744bb`); the grouped
route's board validation done at `472bec9` (round 70); the independent re-challenge of G1-G8 at the candidate; P-04
and P-23 (deferred); the open minor items in ISSUES.md.  The optional
supporting studies (scaling, layouts, packing, DCC, token partitioning,
calibration, virtual sweeps, compile cost, LoC) keep their recorded
statuses and are not release blockers on reproduction grounds.
