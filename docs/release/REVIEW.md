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
2. **Matcher to autoscheduler.** Axis roles (`axis_roles`: reduction,
   output, batch, matrix/vector operand, work-axis) are stamped structurally
   from index sets including the store index. Consumers must not re-derive
   them from memref extents or target counts.
3. **Autoscheduler to codegen.** Candidates pass the target-neutral
   legality pass (C-01) before ranking; materialization runs in cost order
   with rejections recorded (C-03); the reconciled whole program is
   re-priced and bounded joint evaluation is labelled (C-05). Invariant: the
   selected placement is legal, materialized, and priced from the graph of
   the program that is emitted. This holds for the matcher backends; the
   APU v1 vector route and the Gemini-II source route have their own
   planners (G9, X-01).
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

| Gate | Judgment | Basis |
|---|---|---|
| G1 correspondence | **PASS (round 7 judgment, to be re-challenged under G9)** | every substantive mechanism of Sections III and IV has a maintained implementation reachable through `allo.compile` (CLAIMS.md); row 16 realized on Samsung, UPMEM and AiM; the remaining external items are evaluation scripts documented as such (E11, E14 DRAM half, E15, the AttAcc reference-best study and e2e composition); under the user's G9 framing the APU v1 vector route and the Gemini-II source route are release APIs but not the shared pipeline (X-01) |
| G2 semantic preservation / explicit failure | **FAIL at f732b22; repaired in round 11, pending independent review** | the reviewer's renamed-sink and wide-output probes executed with a lost write and wrong values at f732b22 (C-25 follow-up, C-26); outputs are now keyed by destination identity with the required set derived from the trace's effects, and the match enforces the operation's evaluation precision; the rounds 8 to 10 probes still behave; evidence at the new revision in `VALIDATION.md` round 11 |
| G3 general mechanisms | **PASS (round 7 judgment, to be re-challenged)** | structural recognizers and template-derived legality on every backend; renamed and neighbouring programs compile to the same normalized decisions; open minor: M-14; round 8 added no case entries: the cast and copy contracts are proofs over the IR, not spellings |
| G4 search/layout/selected-program integrity | **PARTIAL (reopened in rounds 9 and 10)** | the executed program dropped accepted copies until C-25; required outputs are now derived from the trace's complete external effects and checked against the backend's declaration before selection; re-judge after independent review; the APU routes' planners remain outside the shared selection stage until G9 |
| G5 cost models | **PARTIAL** | provenance and validity labels everywhere; cost perturbations traced to executed programs on Samsung, UPMEM and APU v1; open: C-18 evaluator anomaly, uniform movement weights |
| G6 evidence beyond examples | **PARTIAL (reopened in round 8)** | EVIDENCE.md tables plus the round 6 probe files; remaining gaps are hardware-only confirmations listed in VALIDATION.md; the independent reviewer's probes varied the output conversion and a store guard, which the round 6 campaign had not, and found two silent defects; the campaign is kept as regression evidence, not as proof of generality; new probes must target the shared semantic contracts on every route |
| G7 install/interfaces/maintainability | **PARTIAL** | fresh-venv install at ffb4a78 (needs repeating at the candidate, INSTALL.md section 6); AiM and UPMEM simulator recipes now ship in `simulators/` and were rebuilt from the release copies; Samsung's route is not buildable from the release repository without Samsung's permission for the five patches (P-04, documented, `apply.sh` refuses); architecture doc update for the shared pipeline pending (G9); P-17 documented proprietary SDK |
| G8 backend execution | **evidence complete at 67f89fd, pending independent review** | installed-artifact runs at the round 10 repair: Samsung virtual and simulator, AiM timing-only, UPMEM simulator, APU v1 board (local lock) and Gemini-II card (wheel built on zhang-x4, clean venv, shared lock, 15 passed); all reviewer probes of rounds 8 to 10 rejected or executed correctly through the installed wheel (VALIDATION.md round 10); AttAcc analytical by design; AiM timing-only as accepted |
| G9 one shared compilation pipeline incl. both APUs | **FAIL** | required by the user on 2026-11-02; three end-to-end routes remain in `allo/compiler.py` (ISSUES X-01); design seed in `design/shared-compilation-path.md` |

Verdict: **NOT READY**. Blockers: G9 (X-01) is not implemented; G2's round 8 repair awaits independent review and execution evidence from the installed artifact on the simulators and both hardware hosts at the candidate revision; G7's Samsung simulator route depends on Samsung's permission (P-04); G5 keeps C-18 and the uniform movement weights open.
