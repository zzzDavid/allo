<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# G9 evidence packet: one shared compilation pipeline, including both APUs

Prepared in round 78 (2026-11-06) for the outstanding G9 source review, at
the named candidate `d120084` (its code tree is that of `9d7dc81`), and
updated in rounds 79 and 80 for the APU v1 vector route's realization
boundary (X-35 in `6ce37ad`, X-36 in the round 80 commit; these two commits
change production code after `d120084`, so the next named candidate must
contain them).  Line numbers are those of `d120084` unless a row says
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
| GSI APU v1, vector route | `_apu_v1_enumerate` 1227 (`allo/pim/apu_v1_vector_program.py::apu_v1_vector_candidates` 638; `apu_v1_record_frontend.py::apu_v1_program_candidates` 353 for the record route) | `APUv1VectorCandidate.effects` (`apu_v1_vector_program.py` 534) from the contraction analysis of the shared facts | `APUv1Ctx.check_realization` `allo/spmw_apu_v1.py:202`: ABI, value layouts, transfers, launch extents, and `_check_vector_compute` (rounds 79 and 80, X-35 and X-36): `apu_v1_vector_codegen.validate_realization` re-derives the operations from the contraction and the plan, the whole instruction stream from `_instruction_stream` (the realizer's own builder), the physical emission and the register bindings, and checks the `EmissionSchedule` (every instruction emitted or declared replaced by a spatial form's reset and scatter, compute calls whole repetitions of the operation sequence, every register read defined earlier); the context then compares the emitted source with the schedule's rendering line for line | the board (ledag) |
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

## 6. What remains for G9

The reviewer's round 78 source review of this packet found that the APU v1
vector route's realization check did not bind the realized computation to
the retained contraction (X-35); repaired in round 79 (section 3, the
vector row) with the reviewer's two injections refused and five regressions
at the checkout and from the installed wheel of `6ce37ad`
(`r79/installed_probes.log`); the reviewer's round 79 review found the
check still partial (ingress and emission order, X-36), repaired in round 80
with one realization-boundary invariant over the whole instruction stream,
the bindings and an explicit emission schedule the device source is
rendered from (all four injections refused, 11 regressions, 21 plans
rendered identically to the previous renderer).  Still
open: the independent review of the repair and of the remaining packet
(rounds 28 to 30: the `APUv1Program` manifest from the retained IR, the
slice 5 traces); the Gemini-I rows at a candidate (the board's `/tmp`); the
open minor items in `ISSUES.md`.
