<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# G9 evidence packet: one shared compilation pipeline, including both APUs

Prepared in round 78 (2026-11-06) for the outstanding G9 source review, at
the named candidate `d120084` (its code tree is that of `9d7dc81`), and
updated in rounds 79 and 80 for the APU v1 vector route's realization
boundary (X-35 in `6ce37ad`, X-36 in `7fb19a4`, `5965d1f` and the round 82 commit;
these commits change production code after `d120084`, so the next named
candidate must contain them).  Line numbers are those of `d120084` unless a row says
otherwise.  The design history is in
`design/shared-compilation-path.md`; this packet lists what the reviewer
has to read, in pipeline order, with the evidence that exercises it.

## 1. The common orchestration (`allo/compiler.py::compile`, line 829)

| Stage | Where | What it fixes for every target |
|---|---|---|
| Source normalization | `allo/spmw_pipeline.py::normalize_source` (146), `SourceProgram` (48), `program_facts` (70) | one `customize` of the workload, the matcher trace and the semantic facts (`allo/spmw_semantics.py::ProgramFacts`, 441; `analyze_program`, 819; `StoreEffect`, 330; `BufferAccess`, 172; `LoopAxis`, 145), identical for every target; enumerators and contexts read the facts, never IR text |
| Route policy | `allo/spmw_reduction_policy.py::require_matcher_route_policy` (198) | the reduction-order policy (`strict` / `reassociate`) refused or admitted before any backend |
| Work scopes | `allo/compiler.py::_stamp_matcher_work_scopes` (361) | the typed retained-IR work scopes copied onto matcher records |
| Grouped APU v1 gate | `allo/pim/apu_v1_grouped_route.py::require_grouped_device_packing` (548) | a dataflow region is declared executable on APU v1 only when every work item proves the dense GEMM template (`prove_dense_gemm_template`, 302) and its packing (`grouped_device_packing`, 507) |
| Launch schedule | `allo/spmw_plan.py::launch_schedule` (566) | the host program's launches bound to the kernels' ABI in declaration order; host transfers resolved against the target |
| Selection and materialization | `allo/spmw_codegen.py::compile_for_target` (1699) -> `allo/spmw_autoschedule.py::autoschedule` (1490), `select_candidates` (1431); `allo/pim/schedule_search.py::evaluate_candidate` (947): build, materialize, score, in that order; `allo/spmw_codegen.py::_materialize_matcher_codegen` (1498) | every complete assignment is materialized under the real orchestration (the launch schedule and the host moves) before it is scored or accepted; only `InfeasibleSchedule` (and the backend rejection types of `spmw_autoschedule.candidate_rejection_types`) is an ordinary rejection with fallback |

## 2. The candidate contract (`allo/spmw_candidate.py`)

| Element | Where |
|---|---|
| `CandidateEffects` (declared inputs, outputs, transfers, launches, allocations) | 126; `CandidateTransfer` 87, `CandidateLaunch` 98, `CandidateValueLayout` 66, `CandidateCoverage` 56 |
| Derivation from the shared facts | `candidate_effects` (196), `verbatim_values` (232), `indirect_reads` (282), `_program_values` (311) |
| Enforcement before selection | `require_candidate_effects` (875): a caller-supplied placement's declared effects are checked field by field against the current derivation |
| Per-backend derivers (registered at import) | `EFFECTS_DERIVERS["samsung_hbm_pim"]` `allo/spmw_samsung.py:844`; `["aim"]` `allo/spmw_aim.py:272`; `["upmem"]` `allo/spmw_upmem.py:2160`; `["attacc"]` `allo/spmw_attacc.py:233`; `["apu_v1"]` `allo/pim/apu_v1_grouped_route.py:814` |

## 3. Backend hooks on the same stages

| Backend | Enumerator (`allo/spmw_autoschedule.py`) | Effects deriver | Realization check (declared vs emitted) | Run |
|---|---|---|---|---|
| Samsung HBM-PIM | `_samsung_enumerate` 555 | `spmw_samsung._derive_samsung_effects` 833 | `SamsungCtx.check_realization` `allo/spmw_samsung.py:1242` | PIMSimulator conductor |
| SK hynix AiM | `_aim_enumerate` 1051 | `spmw_aim._derive_aim_effects` 264 | `AimCtx.check_realization` `allo/spmw_aim.py:1084`; the program derivation's own checks before any record is lowered: `_dense_elementwise_domain` (X-33), `_check_device_value_layouts` (X-32, frames), `_check_physical_liveness` (X-32, cells and row chunks) | native ramulator2 (timing-only) |
| UPMEM | `_upmem_enumerate` 1180 | `spmw_upmem._derive_upmem_effects` 2150 | `UpmemCtx.check_realization` `allo/spmw_upmem.py:3586` | uPIMulator through the native SDK executor |
| AttAcc | `_attacc_enumerate` 1387 | `spmw_attacc._derive_attacc_effects` 223 | `AttAccCtx.check_realization` `allo/spmw_attacc.py:349` | Ramulator2-AttAcc traces, analytical composition |
| GSI APU v1, vector route | `_apu_v1_enumerate` 1227 (`allo/pim/apu_v1_vector_program.py::apu_v1_vector_candidates` 638; `apu_v1_record_frontend.py::apu_v1_program_candidates` 353 for the record route) | `APUv1VectorCandidate.effects` (`apu_v1_vector_program.py` 534) from the contraction analysis of the shared facts | `APUv1Ctx.check_realization` `allo/spmw_apu_v1.py:202`: ABI, value layouts, transfers, launch extents, and `_check_vector_compute` (rounds 79 and 80, X-35 and X-36): `apu_v1_vector_codegen.validate_realization` re-derives the operations from the contraction and the plan, the whole instruction stream from `_instruction_stream` (the realizer's own builder), the physical emission and the register bindings, and checks the structured `EmissionSchedule` against the plan (round 81: loop roles and counts, guards, bank sections, every call's execution count, the declared register substitutions and pointer indices, readiness and store finality, the spatial forms' resets and scatters; round 82: `_ScheduleInterpreter` executes the plan's instruction stream under the plan's batching and the structured schedule concretely on two abstract machines and compares every store's value, destination and multiplicity, the scatter coverage, the compute, reset and setup counts and the priced inventory; round 83: both machines key their cells by the concrete VR after the call's substitutions, so a clobber through a register the allocator shares across live ranges changes the stored value); the context then compares the emitted source with the schedule's rendering line for line as a tamper check | the board (ledag) |
| GSI APU v1, grouped route | `_apu_v1_grouped_enumerate` 1278 | `apu_v1_grouped_route.apu_v1_grouped_candidate_effects` 596 (`_derive_apu_v1_effects` 804) | `APUv1Ctx._check_grouped_realization` `allo/spmw_apu_v1.py:335` (declaration, independently re-derived packing, map) | the board |
| GSI Gemini-II | `_apu_g2_enumerate` 1270 (`allo/pim/apu_g2_record_frontend.py::apu_g2_record_candidates` 957; `apu_g2_source_route.APUG2SourceContract` 208) | the record frontend's contract bound to the retained contract and the launch partition (X-25, X-27) | `APUG2Ctx.check_realization` `allo/spmw_apu_g2.py:150` | the zhang-x4 card |

No backend has a route around `compile`: there is no planner seam and no
`_PROGRAM_ROUTES` (removed in round 16; `allo/compiler.py` has one entry).

## 4. Tests that exercise the contracts (file, tests at the candidate)

| Suite | Tests | What it covers |
|---|---|---|
| `tests/pim/test_shared_semantics.py` | 110 | the lettered sections of rounds 12 to 70: facts, effects, coverage, launches, transfers, the grouped route (the `apu_v1_grouped` tests, X-28 to X-31), UPMEM and AttAcc effects (u), AttAcc's schedule and datapath (v), flat ranges (w), the structural attention proof (x) |
| `tests/pim/test_g6_semantics.py`, `test_g6_decisions.py` | 28, 21 | reviewer probes of rounds 8 to 27: renamings, reversed arithmetic, cost flips reaching execution on every backend |
| `tests/pim/test_aim_value_layouts.py`, `test_aim_dense_domains.py` | 17, 3 parametrized to 4 | X-32 (distributions, physical liveness, row chunks, frames) and X-33 (dense domains) |
| `tests/pim/test_cent_aim_workloads.py`, `test_paper_golden.py` | 7 parametrized over the 14 CENT cases and the paper rows | trace pins, vendor inventories, golden cycles for AiM, Samsung and UPMEM |
| `tests/pim/test_upmem_sdk_executor.py`, `test_upmem_coordinates.py` | 4, 4 | the native SDK executor; load identity on full resolved coordinates (U-16) |
| `tests/pim/test_attacc_virtual.py`, `test_attacc_trace_check.py`, `test_attacc_compose.py` | 15, 10, 4 | AttAcc declarations, traces checked command by command, the decode composition |
| `tests/pim/test_apu_g2_hardware.py` | 11 | the card: typed and source kernels, fingerprint mismatches refused before and after launch, aliasing, extremes, fused against separate plans |
| `tests/pim/samsung_hbm_pim/` | per file | the conductor, row pairing, supplied declarations, owner maps |

## 5. Installed-artifact evidence at the candidate

Round 78 (`tenon-release-evidence/r78/`, `VALIDATION.md` round 78): sdist
`ccbd0ce5...` from `git archive` of `d120084` (0 dirty entries), local wheel
`81e37bac...` and zhang-x4 wheel `e7aff69d...` installed into clean venvs;
every backend module imported from site-packages, no module from the
checkout.  From the installed artifacts, with the tests and benchmarks copied
out of the checkout:

| Evidence | Result |
|---|---|
| X-33 and its neighbours | the stride-two witness refused, the dense program accepted, `test_aim_dense_domains.py` + `test_aim_value_layouts.py` 21 passed |
| The reviewer's round 65, 71, 72, 73, 74 probes | the interleaved / collapsed maps refused (round 65); fan-out and inverse refused; slot-liveness and multi-row witnesses at the source semantics; the prefix composition refused |
| Shared semantics | 156 passed, 1 failed (the known deferred V1-19 under the venv's NumPy 2.5.4); the grouped route's section u 19 passed |
| AiM | golden rows 3 + 11, the CENT suite 8 + 13, the fourteen-row provenance identical to rounds 73 to 75 |
| Samsung | 3 passed, the default selection faster than the paper (4,108 / 7,057 / 7,255 against 4,435 / 8,041 / 8,247) |
| UPMEM (native SDK executor) | default 19 passed, `paper_full` 13 passed, the GEMV test 1 passed |
| AttAcc | the twelve results identical to the `54cbb08` set and every later candidate; the AttAcc tests 65 passed |
| Gemini-II (zhang-x4) | the card tests 11 passed; the pre-registered validation session: status pass, every output exact (batched_gemm 0.9976, gesummv 0.9853, bicg 0.9961, mvt 0.9985 of the paper; gemm 1.0042, nominal, X-34) |
| Gemini-I | not run at this candidate (the board host's `/tmp`) |

Round 82 (`tenon-release-evidence/r82/`, `VALIDATION.md` round 82): the
named candidate `92d4120`, the first that contains every production fix
after `d120084` (`6ce37ad`, `7fb19a4`, `5965d1f`, `92d4120`, all in the APU
v1 vector route); sdist `e40df359...` (0 dirty entries), local wheel
`dd91e58e...`, zhang-x4 wheel `32c005b6...`, clean venvs, every module from
site-packages.  From the installed artifacts: the X-35 / X-36 refusals (the
reviewers' round 78, 79 and 81 probes and the round 82 probe, every fault
refused, every intact control accepted), the 29 realization-binding
regressions, the paper APU v1 kernels' compile-only golden rows; the AiM,
Samsung, UPMEM and AttAcc rows identical to the `d120084` table above (the
chain's AttAcc stage re-run alone; the stale comparison lines it printed are
disregarded); shared semantics 156 passed and the one V1-19 failure; the
Gemini-II card tests 15 passed and the one pre-registered session with the
same attestations (gemm 0.9999, batched_gemm 0.9992, gesummv 0.9880, bicg
0.9986, mvt 1.0002 of the paper; mvt nominal, X-37); Gemini-I not run.  This
is installed-artifact evidence of the repair; it is not the independent
review of it, which G9 still needs.

Round 84 (`tenon-release-evidence/r84/`, `VALIDATION.md` round 84): the
named candidate `88852b7` (code of `8d9136f`, every production fix since
`d120084`), sdist from `git archive 88852b7`, local wheel `9d7b6be6...`
and a zhang-x4 wheel in clean venvs.  From the installed artifacts: the
reviewers' round 78, 79, 81 and 82 probes (the alias probe included) and
the round 82 probe, every fault refused and every control accepted; the
32 realization regressions and the compile-only golden rows; the AiM,
Samsung, UPMEM and AttAcc rows identical to the `d120084` table above
(the AttAcc stage inside the chain this time); shared semantics 156 passed
and the one V1-19 failure; the Gemini-II card tests 15 passed and the one
pre-registered session with the same attestations (gemm 0.9973,
batched_gemm 0.9945, gesummv 0.9901, bicg 0.9959, mvt 1.0013 of the paper;
mvt nominal, X-37); Gemini-I not run (the board host's `/tmp`).  The
reviewer accepted the round 83 repair; the manifest and slice 5 review
(rounds 28 to 30) and the Gemini-I execution at a candidate remain the open
G9 items.

## 6. What remains for G9

The reviewer's round 78 source review of this packet found that the APU v1
vector route's realization check did not bind the realized computation to
the retained contraction (X-35); repaired in round 79 (section 3, the
vector row) with the reviewer's two injections refused and five regressions
at the checkout and from the installed wheel of `6ce37ad`
(`r79/installed_probes.log`); the reviewer's round 79 review found the
check still partial (ingress and emission order, X-36), repaired in round 80
with one invariant over the instruction stream and the bindings; the
reviewer's round 80 injections (a forged call binding, an early egress, an
empty reduction loop) showed the schedule part still partial, repaired in
round 81 with a structured schedule (loops, guards, bank sections, resets,
scatters, calls with declared substitutions) checked against the plan's
counts, roles, substitutions and ordering (all injections of rounds 78 to
80 refused, 15 regressions, 21 plans rendered identically to the parent
renderer by the reviewer's own comparator); the reviewer's round 81
injections (zeroed indices, zeroed pointer batches, an undeclared call,
folded accumulator slots, a scatter writing zeros) showed that shape checks
do not establish meaning, repaired in round 82 with an abstract interpreter
that executes the plan's batching and the schedule on two machines and
compares stores, coverage and counts (all injections of rounds 78 to 81
refused, 29 regressions, the specialized forms intact, 21 plans identical).
The reviewer's round 82 alias probe (a reset moved across the live range
of a value sharing its hardware register) showed the interpreter keyed
storage by logical name; repaired in round 83 by keying every cell by the
concrete VR (the reviewer's probe refused, the intact reuse accepted, 32
regressions, 21 plans identical).  The reviewer accepted that repair in
round 83 (`reviewer-r83/`: the 32 regressions from checkout and installed
code, the alias probe and a renamed uint16 11 x 17 x 7 contraction, the
21-plan comparator, the retained-IR manifest suite 7 passed), closing the
physical-alias finding; this packet still does not claim G9 closed, and
the checker establishes no hardware correctness.

Round 84 review (`reviewer-r84/README.txt`): the manifest source review of
rounds 28 to 30 is complete for the reviewed domain.  The reviewer read
`allo/compiler.py::compile` (line 829) through the common frontend,
candidate validation, materialization and scoring and the APU hooks,
`allo/pim/apu_v1_hybrid.py` (line 375, retained IR), and the X-21 repair
(returned allocations and initialization values bound correctly), and ran
the installed-wheel manifest and measurement-driver suites (25 passed,
`manifest_driver_tests.log`).  Source acceptance, not G9 closure.  The
reviewer also supplied current Gemini-II slice 5 evidence from the round 84
wheel on zhang-x4 under the runtime's device lock
(`g2_focused.py`, `g2_selection_traces.json`): a cost perturbation moves
the 64 x 128 x 33 GEMM from `single/k2` to `single/k1` (2,652 to 3,444
ticks per body), forced selection moves BiCG from `fused/k1` to
`separate/k1` (2,534 to 5,065), every output exact, every launch carrying
packing attestations, program and layout fingerprints and artifact hashes;
a reviewer-chosen renamed 48 x 64 x 35 int4/int15 contraction executed
exactly with inputs, actual outputs and independent expected arrays
retained (`g2_neighbor_*`); a K96 neighbour was refused before launch by
the documented power-of-two reduction rule.  Mechanism checks, not paper
confirmation samples.

The candidate's repaired APU v1 vector path executed on the Gemini-I
board in round 136 (VALIDATION.md round 136, the packaged working
candidate `c762d4d` from its installed wheel, pre-registered before launch):
the default vector programs matrix_multiply (plan
`temporal_dma_coalescing_broadcast_friendly_acc8`) and histogram
reproduce the paper's rows with exact outputs; the forced
`temporal_dma_coalescing_broadcast_friendly_acc4` plan and the
`MICRO_DMA_L4_L1_32K x 0.01` cost perturbation reach selection, a
different generated source and execution with exact outputs, the model's
order confirmed by the board.  Rounds 85 to 135 had recorded this as
blocked by the board host's `/tmp` (VALIDATION.md round 85); the host
recovered before round 136.  The manifest and slice 5 review of rounds 28 to 30 is no longer pending.

Specifically outstanding for G9 (round 139, replacing the earlier
"remaining minor items" wording; item (1) carried to `40f9b64` in the round 187
review: the reviewer accepts the round 141 contract-boundary source review
for the verified unchanged paths plus the inspected S-18 and G2-16 changes,
a limited acceptance that is neither hardware nor whole-release acceptance): (1) the independent review, at the final
candidate, of the shared candidate contract as implemented: the invariant
that every selected placement's declared `CandidateEffects`
(`allo/spmw_candidate.py`: `CandidateCoverage`, `CandidateValueLayout`,
`CandidateTransfer`, `CandidateLaunch`) equals the derivation from the
current program facts (`_derivation_problems`, `rebind_supplied_layouts`
for a supplied `layout=`) and passes `validate_candidate_effects`
(`_check_value_layouts`, `_check_transfers`, `_check_launches` /
`_check_operation_launches`; `require_candidate_effects` for every target in
`EFFECTS_REQUIRED_TARGETS`; `check_graph_consistency` against the priced
graph), with the per-backend derivers `samsung_candidate_effects`,
`aim_candidate_effects`, `upmem_candidate_effects`,
`attacc_candidate_effects`, `apu_v1_grouped_candidate_effects` and the APU
v1 vector and Gemini-II candidates' effects, and the realization checks
`SamsungCtx.check_realization`, `AimCtx.check_realization`,
`UpmemCtx.check_realization`, `AttAccCtx.check_realization`,
`spmw_apu_v1.check_realization` (the plan interpreter of rounds 79 to 83)
and `spmw_apu_g2.check_realization`; the rounds 78 to 84 reviews covered
these at earlier revisions, and X-01 stays in-progress until that review
is recorded at the final candidate.  (2) The Gemini-II slice 5 traces at
the current candidate: `benchmarks/apu_g2/selection_traces.py` (a forced
alternative and a cost perturbation reaching generated programs and
launches on the card with exact outputs and packing attestations) was last
executed from the round 84 wheel (`88852b7`, the reviewer's
`g2_selection_traces.json`); the maintained card chain
(`g2_card_tests.sh`, `tests/pim/test_apu_g2_hardware.py`, 15 tests) does
not run it; executed in round 140 from the `f57c840` wheel on the card
under the device lock (VALIDATION round 140: the cost perturbation moves
the 64x128x33 GEMM from `single/k2` to `single/k1`, 2,713 to 3,441 cycles,
the forced `separate/k1` BiCG against the default `fused/k1`, 2,516 to
5,082, every launch exact), and repeated in round 174 from the accepted
candidate `40f9b64`'s wheel on the card (VALIDATION round 174: `single/k2`
to `single/k1`, 2,716 to 3,457 cycles; the forced `separate/k1` against the
default `fused/k1`, 2,552 to 5,054; every launch exact; mechanism traces,
not performance sessions), and again in round 192 from the accepted candidate
`f32b8e3`'s wheel (`single/k2` 2,678 to `single/k1` 3,446; `fused/k1` 2,550 to
`separate/k1` 5,058; every launch exact), and again in round 196 from the working
candidate's `448a51f` wheel built on the card host (`single/k2` 2,699 to
`single/k1` 3,476; `fused/k1` 2,510 to `separate/k1` 5,077; every
launch exact), so item (2) is executed at the working candidate's package.  (3) The APU
v1 side of slice 5 is demonstrated at `c762d4d` (round 136) and the
runtime's failure evidence is complete since `dbe0543` (V1-21); the round
136 lost sample itself stays unverified.  (4) The round 139 review's own
pass over item (1) found the contract boundary open on the payload routes
(ISSUES X-45): `_derivation_problems` skipped the comparison when the
target had no deriver (Gemini-II) or the deriver returned None (the APU v1
vector plans and program records), so an edited `CandidateValueLayout.
packing` passed selection and both realization checks; repaired in round
140 (`f57c840`): `derive_candidate_effects` takes the executable payload's
own derivation (`payload.effects()`) where no deriver applies, a required
target with neither is a problem, and `spmw_apu_v1.check_realization` and
`spmw_apu_g2.check_realization` compare the complete declaration with the
realized payload's derivation (regressions
`tests/pim/test_candidate_contract_payload_routes.py`).  The round
140 review accepted the repaired boundary (`reviewer-r140/README.txt`:
the 31 contract regressions and both adjusted record tests from the
`89e7b745...` wheel, the three changed modules byte-identical to the
source, reviewer probes on renamed irregular uint16 contractions with
overflow and carried outputs, a guarded whole-function record and a
non-mutating Gemini-II virtual result; no weakened numerical check,
benchmark-specific dispatch or fabricated execution found).  That is the
acceptance of the X-45 boundary, not a declaration that every G9 gate
passes.  **Item (1) accepted (round 141 review, `reviewer-r141/README.txt`)
at the final compiler candidate `f57c840`**: the reviewer followed
normalization, facts, candidate validation, movement, supplied-layout
rebinding, materialization, graph construction and final realization
through the backend hooks on the installed wheel `89e7b745...` and
accepted the source invariant; recorded here as that acceptance, not as
every G9 and G10 requirement passed.  The hardware obligations stay
explicit by revision: the APU v1 vector-path execution with the forced
plan and the cost perturbation was executed at `c762d4d` (round 136) and
has not been re-executed at `f57c840` (the runtime's failure path and the
contract's derivation changed since, not its launch procedure or
programs; any claim of execution at `f57c840` requires that run); the
Gemini-II slice 5 traces were executed at `f57c840` (round 140); the round
136 lost linear-regression launch stays unverified; the Gemini-I
confirmation campaign at `f57c840` is incomplete (VALIDATION rounds 141
and 142).

**Candidate transition (round 158 review, 2026-11-12).** `273305c` (the
S-18 diagnostic repair: the Samsung enumerator's op-set check before role
binding, the only compiler source change after `f57c840`) is accepted as
the replacement compiler candidate for completing the gates.  Executed at
`273305c` from its installed wheel `1e86935a...`: the Samsung lowering
suite with the mixed-body regression, the joint-selection suite and the
host-program suite (38 passed, VALIDATION round 158).  Inherited from
`f57c840` through artifact identity, limited to the verified unchanged
paths: 498 of the 501 installed files are byte-identical and the two
compiled bindings are identical after the embedded build path is
substituted, so items (1) and (2) above and the rounds 140, 141 and 157
installed evidence describe the same code at `273305c` except the Samsung
enumerator's refusal ordering; `f57c840`'s evidence keeps its own identity
and is not relabeled.  The Gemini-II card tests were executed at `273305c` in round 165 (the
zhang-x4 wheel `8a7f5bb2126e...` from the `273305c` sdist, 15 passed under the
device lock), so they are no longer inherited.  Still executed at no
candidate but `c762d4d`: item (3), the APU v1 vector-path execution with the
forced plan and the cost perturbation; still owed at the accepted
candidate: that execution and the Gemini-I remaining quota.  The Gemini-I pooler and campaign accept the
transitions only through verified installation-equivalence manifests
(round 159, `r159/installation_equivalence.json`, `f57c840` to `273305c`;
round 170, `r170/installation_equivalence.json`, `f57c840`, `273305c` and the
accepted candidate `40f9b64`; round 192, `r192/installation_equivalence.json`, adding `f32b8e3`; round 195, `r195/installation_equivalence.json`, adding the `448a51f` build with `spmw_host_program.py` explicitly allowed absent, the pooler extended for that one file in `48a6a70`; ISSUES X-49).  Apart from items (1), (2) and the unverified sample,
nothing else of G9 is open: the migration, the matcher-backend contracts,
the retained-IR manifest and the hardware provenance bindings are recorded
above.

**Candidate transition (round 194 review, 2026-11-12).** `56f37e5` is adopted as
the working source candidate for completing validation, not release approval:
`f32b8e3` plus the C-10 resolution (the callerless host-scheduling module, its
seven exports and its five analysis tests removed; the maintained-path
host-program tests restored).  Its `allo/` is identical to `448a51f`, the
revision the `r193` wheel `88e61410171715cb...` was built at; the wheel is the
package-build identity and `56f37e5` the source identity, kept distinct.
Executed from that wheel: the public-surface, public-exports, APU v1
scalar-surface and packaging suites (134 passed, 27 skipped, round 193) and
the restored host-program tests with the joint-selection and Samsung
composition cases (21 passed, round 194).  Inherited from `f32b8e3` through
artifact identity, limited to the verified unchanged paths: 496 of the 501
installed files are byte-identical, the two compiled libraries identical after
the build-path substitution, `allo/__init__.py` and `allo/spmw_target.py`
changed and `allo/spmw_host_program.py` absent (`r193/run/installed_file_identity.json`).  In round
196 the Gemini-II card tests, the pair-affine record, the selection traces
and item (2) were executed at `448a51f` itself from the wheel built on the
card host from the `448a51f` sdist, and the 42-stage installed-artifact
milestone passed from the `r193` wheel, so those no longer rest on the
inheritance; `f32b8e3`'s evidence keeps its own identity and is not relabeled.  Still owed at the
working candidate: item (3), the APU v1 vector-path execution with the forced
plan and the cost perturbation, and the Gemini-I remaining quota (kmeans 2,
word count 2, linear regression 3), both blocked on the board host's `/tmp`.

