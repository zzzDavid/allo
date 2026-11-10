<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Release status

**User scope correction, confirmed 2026-11-04:** G10 requires only the main
Samsung/AiM/UPMEM/Gemini-I/Gemini-II performance comparisons and the AttAcc
kernel/decode figures. All supporting studies are optional. The inventory's
`release_scope` column is authoritative: currently 55 required performance
entries (historical coverage at the recorded revisions, round 84: 51 reproduced, 4 open (the three Gemini-I rows X-22 / X-23 whose pre-registered campaign is queued behind the board host's `/tmp`, and the Gemini-II mvt row nominal at the named candidates `92d4120` and `88852b7`, X-37; gemm, nominal at `d120084`, X-34, is REPRODUCED at `92d4120` and `88852b7`), 0 unverified; at `88852b7` (and before it `92d4120`) from installed artifacts: the fourteen AiM rows, the three Samsung rows, the sixteen UPMEM rows, the twelve AttAcc results and the five Gemini-II rows (one pre-registered session), identical AiM / Samsung / UPMEM / AttAcc values to `d120084`; earlier: at `d120084` itself from installed artifacts: the fourteen AiM rows, the three Samsung rows, the sixteen UPMEM rows, the twelve AttAcc results and the five Gemini-II rows (one pre-registered session); earlier text follows: executed at the integration candidate ba2f58d itself as of round 68 from installed wheels: the fourteen AiM rows (10 exact, 4 corrected RoPE programs whose RoPE-group candidate pairs were all simulated natively in round 66, the default the minimum), the three Samsung rows and the twelve AttAcc results (all identical to every earlier candidate's runs); the sixteen UPMEM rows exact from the installed wheel through the native SDK executor (the host's Docker data root being full); the five Gemini-II rows measured at cd744bb in one pre-registered session and compiling from every later wheel to the same plans and fingerprints (gemm, gesummv and bicg at or below the paper; batched_gemm and mvt nominally above, so mvt is OPEN (nominal) at this revision under the pooled rule while it was REPRODUCED at 54cbb08); UPMEM (Docker) and Gemini-I (campaign pending on the host's /tmp) rows not yet re-executed there) and 28 optional supporting
entries. These are evidence at the recorded revisions, not an assertion of
current-candidate readiness. Optional reproduction gaps do not block release;
actual production defects and G9 still need their appropriate assessment.
This overrides older exhaustive-campaign next-action lists below. Checkpoint
the current optional token-partition/cost increment, then prioritize the
remaining required comparisons and release validation.

Updated: 2026-11-10 (round 114: `ab239e3` the packaged working candidate by the reviewer's decision, the `b994fc0` Gemini-II campaign inherited with its identities; the AttAcc evidence attribution corrected (last installed figure run `88852b7`, round 84; the milestone's AttAcc stage was contract tests only); the AttAcc figure stages added to the milestone script and executed from the `ab239e3` wheel; the copy helper's optional semantics made explicit and exercised; round 113: the premature P-08 closure amended after the review found four deleted result wrappers and a removed route entry still named as the API; the architecture document and three design documents corrected to the single `CompiledCallable` return and the shared enumerator entry; a definition-based symbol check preserved; the `ab239e3` installed validation completed on both hosts without a Gemini-II session (local milestone every stage PASS with results identical to `b994fc0`'s; zhang-x4 wheel and card tests 15 passed); `ab239e3` proposed as the packaged candidate's replacement, pending the reviewer's call; round 112: candidate and artifact identity consolidated below (inherited against fresh evidence per revision, no rebuild); U-08's documentary portions and P-08 closed against the current source (architecture section order, design-note banners, the UPMEM skill document), the unreachable-export portion of U-08 left open; the board still blocked (`/tmp` 1.1 MB); round 111: the P-27 repair's call-after-close crash fixed (`ab239e3`, lifecycle regression); the round 110 investigation script corrected to the benchmark's repetitions with its run preserved beside the stale one; the owed installed-artifact validation of the changed backend done from the `ab239e3` wheel (UPMEM probe exact, 0 cleanup tracebacks, reproducer refused, backend tests and UPMEM rows from the wheel); round 110: the Gemini-II gemm/mvt cause investigation found the generated work identical to the frozen experiment (fingerprints, operation stream, device kernel source, build flags, timed unit) and established no production cause, the rows staying OPEN (nominal) under the fixed rule; P-27 repaired in `allo/backend/c.py` (library unloaded before scratch cleanup; 0 tracebacks), the first compiler-source change after `b994fc0`, owing installed-artifact validation of that backend; the board still blocked (`/tmp` 1.1 MB); round 109: the pooler now verifies raw device ticks, round pairing and fixture identity (P-26 completed for the listed checks, 39 tests, the reviewer's counterexamples rejected, the campaign recomputed identically; the round 108 claim that the scripts cannot report false success is withdrawn); P-27 recorded (UPMEM probe tempfile cleanup tracebacks after an exact result); round 108: the validation scripts' first repair (P-26: the pooler validates sessions before pooling, the probes assert their contracts; negative injections fail the milestone, the campaign recomputes identically, the CENT `paper_full` stage executed from the wheel, 13 passed); round 107: `b994fc0` declared the final compiler candidate; the milestone scripts repaired for failure reporting (`docs/release/scripts/`); the pre-registered Gemini-II confirmation campaign run at `b994fc0` and pooled: gemm 1.0009 and mvt 1.0001 OPEN (nominal) and fixed, the other three REPRODUCED, required set 50 REPRODUCED / 5 OPEN; the Gemini-I host's `/tmp` still full; round 106: the milestone re-run at the named candidate `b994fc0`: the bundled Samsung simulator built and discovered from the wheel, the shared paths and golden rows executed, the Gemini-II card tests 15 passed and the session reproducing four rows with gemm nominal (1.0015); every local installed stage passed, the AttAcc and allocation set 77 passed where round 105 had the X-44 failure; round 105: the integration milestone begun at the named candidate `f6b19bd` (wheel, clean venv, the bundled Samsung simulator built and discovered from the wheel, the shared paths executed; the Gemini-II card tests and session on zhang-x4: four rows reproduced, gemm nominal, one typed program refused by the round 99 bounds rule, X-44, repaired); V1-19 closed; V1-19's local rebinding extended to closure cells and the `dsl` submodule, the reviewer's three probes passing under NumPy 2.5.4; round 104: V1-19 repaired locally, the functional reference rebinds the workload's own globals (imported and aliased `grid`/`reduction` spellings, no process-wide patch), the reviewer's probe passing under NumPy 2.5.4; round 103: the board host's `/tmp` rechecked at 2.4 MB (hardware still blocked); V1-19 repaired, the APU v1 functional reference runs under NumPy 2.5.4 without the suppressed deprecation; round 102: S-17 closed, the three Samsung decision tests restated from the current model (derived flip factor executed on the conductor, odd-fiber term per candidate, first-fit fill schedule) and the allocation test parametrized by orientation; round 101: X-43, the AiM row footprint now covers the realized mapping's batch groups and the realization check refuses matrix rows outside their region (the reviewer's 48x320 probe reserved one row for two addressed); round 100: X-41 closed, AiM enumerates both storage orientations of a two-sided contraction as cost-ranked candidates (`contraction_orientations` shared with AttAcc), the layout resolver no longer reads the `matrix` name; the reviewer's batched GEMV selects one program in both spellings (223 cycles where the convention gave 342); round 99: X-42's check judges each operand's coordinates together (a repeated loop variable is a diagonal gather, refused; staged extents must fit the buffer), the reviewer's `b[k, k]` probe refused; round 98: X-42's result-offset records keyed by operand, so same-buffer operands keep both accesses (the reviewer's duplicate-operand probe refused in both orders); round 97: X-42's work-grid offsets proven against the realized frontier (item `w` reads offset `w x extent`) instead of an item-count exemption, the reviewer's shifted-grid probe refused; round 96: X-42 completed over the loop domain (effective accesses after `lower + step * t`; `REALIZES_RESULT_OFFSETS` declared False by AiM and AttAcc), the reviewer's two probes refused; round 95: the reviewer's strided-access finding (X-42) fixed by one shared check before selection, accepting dense flattenings; AttAcc orientations as cost-ranked candidates (X-41, second increment), operand order no longer changes the AttAcc program; round 94: the Gemini-II candidate session at `fdee44b` on zhang-x4 (card tests 15 passed; mvt REPRODUCED 0.9968, batched_gemm, gesummv, bicg REPRODUCED, gemm OPEN (regression) 1.0048 under the pre-registered rule, all 210 outputs exact); X-41's resolved-coordinate half repaired with AiM reading the same index sets; the reviewer's build_py test hole closed; the shared filesystem's hard quota reached and 39 regenerable build trees removed; round 93: round 92 finished (the conductor-test failure was a stale expectation; the GEMM residency execution test exposed X-40, repaired, with X-41 recorded open), the Samsung simulator bundled with the tool under the user's direction (P-04: `simulators/PIMSimulator/{upstream,patches,NOTICE.md}`, `python -m allo.pim.samsung_simulator build`, `TENON_HOME` discovery, the wheel copy), its build byte-identical to the evaluation's conductor after stripping, and validated from the `39449d6` wheel in a clean venv outside the checkout (verify, build, discovery, a compiled GEMV executed: 4,108 cycles, max error 0.0004); the board host's `/tmp` still at about 7 MB; round 15: launch operand bindings validated (C-29b), the candidate contract enforced field by field (C-30), Gemini-II typed records migrated into the shared pipeline; committed as `9255d29`, exercised from the installed wheel on the simulators, the APU v1 board and the Gemini-II card; round 16: the reviewer's three contract gaps closed (C-30b) and the last typed route removed; round 17: the `APUv1Program` facade replaced by the shared front stages on the phase's source with retained store effects (X-02); round 18: indirect loads, fill initializers and unused arguments retained or represented correctly (X-03), whole launches reserved for verbatim programs (C-32); round 19: effects in program order (X-04) and Samsung as the first matcher backend declaring effects from the shared facts; round 20: carried destinations are reads on every backend (X-05), Samsung storage, packing and transfers tied to the emitted program (X-06), realization checked at the materialization boundary (C-33); round 21: distinct input/output images and exact vector residency complete X-06, AiM is the second matcher backend declaring effects from the shared facts; round 22: complete assignments materialize under the real launch schedule and host moves before acceptance (C-33b), the AiM check covers the batch mapping, the GPR readout of results (X-07) and host transfers bound by retained buffer identity (X-08, found partial by the reviewer); round 23: transfer availability checked in schedule order on every target (X-09), AiM slices lowered from the kernel's vector layout (X-10), host records compared with the launched kernel's physical addressing (X-11); round 24: UPMEM and AttAcc declare their candidates' effects from the shared facts and check their realization, the last two matcher backends on the enforced contract; two shared coverage rules completed (X-12 zero-initialized own-element accumulates, X-13 index/guard reads); the AiM GB vector's ingress declared as the lowering realizes it (X-14); round 25: the reviewer's four G9 findings closed (AttAcc one inventory per launch X-15, datapath roles bound X-16, allocation shapes from the IR facts X-17, flat slices converted correctly X-18); G10 inventory and first reproductions (Samsung, UPMEM, AiM, AttAcc selection), the RoPE program corrected (X-19); round 26: flat ranges a distinct region type (X-18 reopened and closed again), AttAcc attention proves its dataflow and per-row normalization from the shared facts (X-20), the 500-instance selection reported as a diagnostic and the four corrected AiM rows as an open regression; round 27: the attention proof made structural (operand order, axes by buffer dimension, loops by retained identity; X-20 reopened and closed again); round 28: the APUv1Program scalar manifest read from the retained IR, the MLIR-text regex analysis deleted; round 29: manifest fidelity (X-21), installed-wheel hardware sessions on the APU v1 board and the Gemini-II card at 653b059 with forced alternatives and cost perturbations traced to launches, and the eval-apuv1/eval-apuv2 paper rows measured (word count an open regression, X-22); round 30: the measurement drivers exit nonzero on any failure and keep the records, the archived paired protocols re-run with their baselines interleaved from installed wheels (Gemini-II 5/5 REPRODUCED with GTML re-measured; Gemini-I matrix multiply and histogram REPRODUCED, kmeans and linear regression nominal (X-23), word count attributed to the generated program's loop-counter width and repaired, the repaired program at parity with the paper's own binary and 0.19% above the paper's archived median (X-22, `dc41df7`)), launch provenance on every APU v1 route, the ledag scratch leak that filled /tmp fixed (X-24); round 31: the reviewer's driver defects repaired (archived outputs preserved outside the archive, incomplete timing evidence rejected, every lane and argument validated), the hardware acceptance rule rewritten to pooled sessions with one bounded confirmation campaign (kmeans returned to OPEN nominal), the NumPy 2.4+ functional-route failure recorded (V1-19), the Samsung XOR layout probe exact and the Gemini-I vector GEMM layout row measured (VALIDATION round 31); round 32: the Samsung XOR row's G10 status withdrawn (the probe bypasses the shared stages, S-13 major), two more driver defects repaired (plan-launch correspondence, absolute scratch), the fiber-swizzle lever measured on the real conductor (dual_fiber 8,247 vs bank_row 8,153) and the missing route specified; round 33: the round 32 row-0 diagnosis corrected with the conductor's own address dump (tiles span two rows; what the stream lacks is two rows of one bank inside one barrier group), `samsung_access_stream` derives the executable pass-phase stream from the lowered program byte-identically to the dump for two GEMVs and replays it on the harness (dual_fiber 1,188 vs bank_row 1,248 cycles), the Gemini-II drivers require launch indexes and declared fingerprints, CLAIMS E4 corrected to 10 exact / 4 OPEN; round 34: the stream's column count from the retained program and no silent geometry default, the trace instrumentation committed in the release overlay, the harness wording narrowed to a diagnostic, and the paper's portability row REPRODUCED through `allo.compile` on PIMSimulator and the board (4,096 of 4,096 bit-exact under strict order with the paper's inputs; V1-16 bounds what Gemini-I can match with inexact products); round 35: that experiment is a maintained in-repo driver (`benchmarks/portability/gemv_source_order.py`, tests, full evidence retention), re-run from the checkout with the same result and cited by the inventory and CLAIMS E11; round 36: the driver's controls validated and its provenance structural (reviewer), and the Samsung tile-row levers implemented through the shared pipeline (`bank_row_rowpair`, `split_rows`, the paired split refused as unsound), measured on the conductor on two GEMVs (XOR rows across the fibers 1.12-1.13x faster than identity rows and bit identical; identity rows under one barrier 6.5x slower), the XOR inventory row OPEN (measured), S-16 opened for the cost model; round 37: the selected half -> bank map made the contract (declared packing, lowering, conductor plan `half_fiber`, decision identity, realization check, rejection of edited declarations before selection), the first staging added to the estimate from the conductor's schedule with device constants, and the default selection earning `split_rows` on both paper GEMVs and a held-out shape as the conductor measures (S-16 repaired); round 38: the staging accounting partitioned per work item (equal work equal cost, rows scale, staging before the MAC, the ordered program in scope; shares equal the conductor's staging) and declarations compared on their axes with contradictory supplied declarations refused; round 39: the supplied-declaration boundary completed (C-34: normalized onto the work item, validated in full, compared on every field before any replacement; structural reuse kept), the AiM simulator built natively while Docker is out of disk and validated on the exact golden rows, and the token-partitioning study run through a maintained driver (6 of 9 simulated minima, largest gap 1.34%, against the paper's 7 of 9 and 2.63%); round 40: contradicting duplicate declarations refused (C-34), the eight-channel budget made a compiler search policy (`resources=`) with the normal selection measured, every AiM run and every plan of the study carrying full provenance, and the study relabelled DIAGNOSTIC because its boundary lacks the softmax bridge and the host/link accounting (M-15); round 41: the study's failure handling made sound (no alias for a failed control, no partial minima comparison, the paper_full gate requires every plan; M-16), the budget contract bounded to the enforced AiM channel pair (C-35), and failed plans and simulator failures retained with structured evidence; round 42: the attention bridge (QK scores -> host global softmax -> probabilities -> SV) as a declared host stage of the shared plan, scheduled, availability-checked, lowered, realization-bound and priced in the graph the normal selection ranks (AiM `HOST_COMPUTE`), the study re-measured on that boundary with independent functional evidence per plan, measured host-stage samples beside the modelled price, the CENT attention control and a neighbouring workload; still DIAGNOSTIC, M-15 narrowed; round 43: the slice's invariants repaired (M-17: scaled softmax semantics verified on scale-exposing operands, the lowered stage bound to the recorded computation, operation and realized position, an explicit host-operation domain for pricing, link and compute terms separated in the measured total, the shared host-input contract); round 44: the lowered records checked against the launch-occurrence inventory derived from the facts and the selected candidates, so duplicated, missing or displaced primary records and host reductions fail materialization; round 45: each host reduction bound to its producing contraction's whole work domain, its price equal to the priced graph's; round 46: each contraction record bound dimension by dimension to the retained semantic shape under the selected candidate; round 47: each launch occurrence's realized records compared record by record with the records the retained program derives afresh, every emitted field included (M-18; relocations, syncs and host transfers still outside that comparison); round 48: the emitted bank payload of each record compared with the derivation and the scope read from it; round 49: M-15 advanced with the structurally derived eight-channel CENT control, the historical host protocol measured beside the price, the study re-measured, and the one-head misses traced by independent command-level calibration to the contraction profile's charges (M-19, open); round 50: the frozen study's `cent-repaired` control restored through the shared pipeline with the round 49 control kept as the `cent-policy` ablation, the host protocol measured on the attention's own scores with the historical methodology, and the contraction profile recalibrated per command class from controlled simulator measurements, validated on held-out programs, after which the normal selection reaches all nine simulated minima (M-19 repaired, awaiting review; the study stays DIAGNOSTIC); round 51: the inventory's `release_scope` column (55 required / 28 optional), and the six required AttAcc kernel-tiling results reproduced through the shared pipeline on a natively built Ramulator2-AttAcc with every generator trace checked against the selected program (ordering and address mapping), the four GEMV rows with their fixed baselines re-run and the score rows at the figure's speedups with the text's 500-instance tally reproduced; required G10 set 42 of 55 REPRODUCED, 7 OPEN, 6 UNVERIFIED (decode); round 52: the trace check made per-command (every command's opcode and address derived from the selected program; all 506 retained traces exact), failed runs retained with structured records, and the six decode configurations composed from the shared-pipeline kernel times with Tenon, DCC and fixed variants reproducing the frozen figure, table and text; required G10 set 48 of 55 REPRODUCED, 7 OPEN (X-19, X-22/X-23), 0 UNVERIFIED; round 53: X-19 attributed to its mechanism (the 32 cycles are exactly the omitted half of the RoPE multiply; the paper's digit reproduced by the inconsistent program's stream; the enumerated candidates searched by estimated cost, which does not establish the minimum simulated time of a faithful program; rows OPEN) and the Gemini-I confirmation campaign pre-registered for the final candidate; round 54: Gemini-II launch recipes bound to the retained contract and the launch partition at materialization (X-25, reviewer fault injection; the damaged cheapest candidate is now a recorded feasibility rejection and the intact alternative is selected); round 55: pair-affine coefficients bound to the stream identities in the same derivation (a reversed-stream recipe with unchanged coefficients is refused); integration candidate named: `54cbb08` (clean tree; local sdist `5aa0ad9c...`, local wheel and zhang-x4 wheel `ba7bedab...` built from it; installed-artifact checks, the Gemini-II card tests and the pre-registered Gemini-I campaign run at this revision, results in VALIDATION round 56: installed-artifact probes and test subsets green except the recorded V1-19 and the Docker-blocked UPMEM probe, Gemini-II four rows REPRODUCED and batched_gemm OPEN (nominal) after its pre-registered campaign (X-26), the Gemini-I campaign pending on the host's temporary space); round 58: the Gemini-II public repetitions update re-derives the validated plan's launches from their own partitions, validates, prices and installs atomically (X-27, the reviewer's second fault injection); round 59: the setter's graph check corrected to run on the public route; round 60: the candidate `cd744bb` (X-25 completed, X-27 and its correction) validated from installed artifacts on the simulators and the Gemini-II card, the AiM, Samsung and AttAcc rows identical to the 54cbb08 runs, Gemini-II in one pre-registered session, results in VALIDATION round 60; round 61: the APU v1 grouped route of a dataflow region put on the candidate contract (X-28: effects derived from the shared facts, the emission checked at materialization, the runtime bound by retained identity); round 62: the route's executable dense GEMM template proven from every work item's retained accesses and written domain before it is declared executable, non-template regions refused for execution with the failing condition (X-29); round 63: the candidate `8795e8f` (both included) validated from installed artifacts on the simulators and the Gemini-II card, the grouped route's board validation and the Gemini-I campaign pending on the host's `/tmp`, results in VALIDATION round 63; round 64: the grouped candidates' batch count validated against the proven packing at materialization, under-batched candidates feasibility-rejected (X-30); round 65: the candidate `c43061f` validated from installed artifacts, the grouped-route board checker corrected to fail on any incomplete case, the X-19 RoPE rows' four symmetric candidate choices simulated natively (default fastest, all above the paper), results in VALIDATION round 65; round 66: the grouped placement's map must be the contiguous-group map the executable realizes (X-31), and all 16 RoPE candidate pairs simulated for all four X-19 rows with the default pair the minimum in each; round 67: the candidate `f418762` validated from installed artifacts, and a native UPMEM SDK executor (`TENON_UPMEM_SDK_HOME`) with which the sixteen UPMEM rows reproduce exactly at the checkout while the host's Docker is unusable; round 68: the candidate `ba2f58d` validated from installed artifacts with the UPMEM rows from the wheel, and U-16 repaired (hoisted address temporaries classify like inline spellings; the host oracle's index cast); round 69: guard and copy load identity on full resolved coordinates in the selection and histogram families, differing accesses refused before selection; round 70: the grouped APU v1 route validated on the board from the installed wheel of `472bec9` (X-28 to X-31 hardware validation complete), the Gemini-I campaign still held by the host's `/tmp` behind guarded scripts, P-24 recorded). Read `TASK.md` in the containing repository, Round 71 (2026-11-05): replicated AiM elementwise grids realized once over the device (X-32); every AiM row at or below the paper at the checkout, the four X-19 rows REPRODUCED with the complete program; installed-candidate re-execution pending; the Gemini-I campaign still queued. Round 72 (2026-11-05): the reviewer's fan-out and inverse probes (a device reader in another distribution than its writer) refused for every live range before lowering (X-32 closure); the integration candidate `2096239` built (wheel `803d1ea4...`) and the fourteen AiM rows, the CENT suite, the value-layout regressions and the reviewer's probe run from it with identical results. Round 73 (2026-11-05): the reviewer's slot-liveness witness reopened X-32 (copies overwritten before a later reader); repaired with per-group bank row regions, copies into the readers' regions and a cell-level liveness check; CENT rows at or below the paper; the integration candidate `26fd884` built (wheel `4bf38997...`) and the fourteen rows, the suites and both reviewer probes run from it with identical results. Round 74 (2026-11-06): the reviewer's multi-row witness reopened X-32 (whole-vector copies replicated the last row); repaired with row-chunked relocations and row chunks in the liveness model; no CENT row affected; the integration candidate `db2ad84` built (wheel `7c8734a7...`) and the fourteen rows, the suites and the three reviewer probes run from it with identical results. Round 75 (2026-11-06): the reviewer's prefix witness reopened X-32 (equal counts, different logical frames over the same bank partitions); the frame is now part of the compatibility and cell contracts, a reader of another frame refused before selection; no CENT row affected; the integration candidate `b221972` built (wheel `ace1b7b0...`) and the fourteen rows, the suites and the four reviewer probes run from it with identical results. Round 76 (2026-11-06): the reviewer's stride-two witness recorded as X-33 (pre-existing: the elementwise lowering assumed a dense domain) and repaired (dense domain proven, strided or repeated stores refused before selection); the Gemini-I campaign launched from the queued watcher and lost its launches when `/tmp` refilled (session 1 incomplete, sessions 2 and 3 guarded; record kept, not pooled); a retry of the same protocol queued at a 120 MB threshold; X-22 / X-23 OPEN. Round 77 (2026-11-06): the Gemini-II batched_gemm (X-26) and mvt gaps read against every recorded session and its re-run GTML lane (one program and packing attestation throughout; consistent with measurement variability; no cause established, no repair supported); both rows OPEN, no session added; the summary above refreshed to 50 / 5. Round 78 (2026-11-06): the named candidate `d120084` built (local wheel `81e37bac...`, zhang-x4 wheel `e7aff69d...`) and the pre-registered milestone run from the installed artifacts: X-33's refusal and dense neighbour, the reviewer probes, the AiM, Samsung, UPMEM and AttAcc rows, the suites, the card tests and the one Gemini-II validation session (batched_gemm and mvt REPRODUCED at the candidate, gemm nominal there, X-34); Gemini-I not run (`/tmp`); the G9 source-review packet `G9_EVIDENCE.md` supplied; the round 77 evidence claims corrected. Round 79 (2026-11-06): the reviewer's APU v1 injections recorded as X-35 (the vector route's realization check did not bind the computation) and repaired (operations, instructions and emitted calls re-derived from the contraction and the plan; a mismatch the shared feasibility rejection); the stale summaries replaced by one current snapshot (51 / 4). Round 80 (2026-11-06): the reviewer's transfer and emission-order injections recorded as X-36 and repaired with one realization-boundary invariant (the whole instruction stream, the bindings, an explicit emission schedule the device source is rendered from); the specialized paths preserved (21 plans rendered identically); the next named candidate must contain the rounds 79 and 80 code. Round 81 (2026-11-06): the reviewer's forged binding, early egress and empty loop injections reopened X-36; the emission schedule is now structured (loops, guards, bank sections, resets, scatters, calls with declared substitutions) and checked against the plan's counts, roles, substitutions and ordering; 21 plans render identically to the parent renderer; the next named candidate must contain the rounds 79 to 81 code. Round 82 (2026-11-06): the reviewer's five round 81 injections (zeroed indices, zeroed pointer batches, an undeclared call, folded accumulator slots, a zeroed scatter) reopened X-36; the contract is now an abstract interpreter that executes the plan's batching and the structured schedule on two machines and compares every store's value, destination and multiplicity, the scatter coverage and the counts with the reference and the priced inventory; all injections of rounds 78 to 81 refused, the accumulator-block, spatial and resident GEMV forms intact, 21 plans identical, 29 regressions; G4 and G9 not claimed closed from the checker; the next named candidate must contain the rounds 79 to 82 code. The named candidate `92d4120` (containing all of it) then built and validated from installed artifacts: the X-35 / X-36 refusals and 29 regressions from the wheel; AiM, Samsung, UPMEM and AttAcc rows identical to `d120084`; the one pre-registered Gemini-II session: gemm 0.9999 REPRODUCED, mvt 1.0002 OPEN (nominal, X-37), the rest REPRODUCED, same attestations; Gemini-I not run (the board host's `/tmp`); snapshot 51 / 4 (kmeans, linear regression, word count, mvt). NOT READY. Round 83 (2026-11-06): the reviewer's alias probe (a reset moved across the live range of a value sharing its VR) reopened X-36; the interpreter's cells are now keyed by the concrete VR after substitutions, so clobbers through shared registers change the stored value; the probe refused, legal reuse and the specialized forms intact, 32 regressions, 21 plans identical, suites unchanged; G4 / G9 await the reviewer's read; the next named candidate must contain the round 83 commit. NOT READY. Round 84 (2026-11-06): the reviewer accepted the round 83 repair (the alias finding closed; G4 / G9 and hardware correctness not established by it); the named candidate `88852b7` (code of `8d9136f`) built and validated from installed artifacts (VALIDATION round 84): X-35 / X-36 refusals and 32 regressions from the wheel; AiM, Samsung, UPMEM and AttAcc rows identical to `d120084` and `92d4120`; the one pre-registered Gemini-II session four rows REPRODUCED (gemm 0.9973) and mvt OPEN nominal (1.0013, X-37), same attestations; Gemini-I explicitly blocked (the board host's `/tmp`); snapshot 51 / 4. NOT READY. Round 85 (2026-11-06): the reviewer accepted round 84, completed the manifest review (rounds 28 to 30) and supplied Gemini-II slice 5 evidence; the `/tmp` blocker recovered within this user's own scratch (552 kB to 174 MB free, the vendor tool's self-extraction moved through `TMPDIR`); the candidate `88852b7` Gemini-I runs pre-registered and launched from the round 84 wheel; the freed 174 MB was refilled by other users within minutes, kmeans kept three valid launches per lane (exact, incomplete, no verdict), the other rows and the vector-route runs were lost or guarded out; the external blocker recorded precisely (VALIDATION round 85); the driver's multi-case `--scratch` collision fixed (X-38) and the vendor tool's temp-directory behaviour documented in INSTALL.md. NOT READY. Round 86 (2026-11-06): the X-38 fix accepted and the reviewer's failure-path probe made a regression; three ledger corrections applied; P-24 fixed (header) and the lint chain now runs through black (31 fork files reformatted, formatting only) and clang-format to pylint's pre-existing style debt (P-23, 629 findings, 9.84/10); `/tmp` unchanged, no board run. NOT READY. Round 87 (2026-11-06): round 86 accepted; a stale `__all__` export fixed (X-39) with an export test over the PIM and SPMW modules; the 49 E-class pylint diagnostics triaged (1 defect, 48 runtime-demonstrated false positives; the W/C/R classes untriaged, not claimed harmless); the payload audit of the round 84 sdist begun (387 files, headers, no evidence or Samsung-derived files, one host-name provenance tuple observed); the active verdict in REVIEW refreshed to the current record. NOT READY. Round 88 (2026-11-06): round 87 accepted; the third-party license texts (LLVM/MLIR with its exceptions, nanobind, uPIMulator) bundled into the sdist and the wheel with a notices inventory and a test (P-25; the first build caught a `*.txt` ignore rule dropping two texts, fixed); payload counts corrected (325 regular files); the distribution audit's wheel-versus-sdist and README/INSTALL checks done; the next named candidate must contain `cf43398`. NOT READY. Round 89 (2026-11-06): P-25 qualified by the reviewer (symbol-confirmed omissions: robin-map, xxHash, Unicode conversion; the UPMEM runtime hunks; nanobind's two origins) and completed: ten texts bundled, the inventory redone from symbols, archives rebuilt from `aae8b23` and validated; the next named candidate must contain `aae8b23`. NOT READY. Round 90 (2026-11-06): round 89 accepted; the W0640 closure class (89 diagnostics) triaged by AST escape analysis, 0 defects, 0 suppressed, 491 W/C/R diagnostics untriaged; the active verdict refreshed (the reviewed packaging items closed narrowly; P-04 and the final-candidate checks retained). NOT READY. Round 91 (2026-11-06): round 90 accepted; the unused-import class (20) triaged for re-exports and registration effects and removed, 0 suppressed, pylint unused-import 0, focused suites pass; 471 W/C/R diagnostics untriaged; `/tmp` 124 KiB, nothing launched. NOT READY. Round 92 (2026-11-06): round 91 accepted; the unused-argument class (27) triaged before any change: 19 required signatures kept, the Samsung staging share's ignored columns-per-row made effective (default unchanged, tested at three geometries), 7 vestigial parameters removed with their callers, nothing suppressed; 444 W/C/R untriaged; `/tmp` 124 KiB, nothing launched. NOT READY.
then this file, then `ISSUES.md` for the next action.


## Candidate and artifact identity (consolidated round 112; no rebuild for this record)

| Revision | Role | Compiler sources | Artifacts (sha256 prefix) | Evidence that is fresh at this revision | Evidence inherited (identical code, not re-run here) |
|---|---|---|---|---|---|
| `b994fc0` (2026-11-10) | the final compiler candidate declared in round 107 and the packaged candidate through round 113; **superseded as the packaged working candidate by `ab239e3` (the reviewer's decision, round 114)**; its Gemini-II campaign is inherited evidence (below) | the X-40 to X-44, V1-19 and S-17 repairs; the Gemini-II modules unchanged since `fdee44b` | local wheel `5ba2e348...` and sdist `4e97a518...` built from `1b4282d` (records-only commit on top, code identical; VALIDATION round 106); zhang-x4 wheel `d5cb59f8...` (`r106/g2`) | round 106: every local installed stage (bundled Samsung built from the wheel, golden and `paper_full` rows, Samsung conductor runs, CENT, the shared-contract set, APU v1 functional reference under two NumPy lines), the Gemini-II card tests (15) and validation session; round 107: the Gemini-II confirmation campaign from the same zhang-x4 wheel (fixed outcomes: gemm and mvt OPEN (nominal), the other three REPRODUCED); round 108: the five repaired probes and the CENT `paper_full` stage from the `5ba2e348...` wheel | the twelve AttAcc figure results (six kernel rows on Ramulator2-AttAcc, six decode configurations): the `b994fc0` milestone's AttAcc stage ran only the analytical and contract tests (`tests/pim/test_attacc_virtual.py` runs no simulator), so those figures were last re-executed from an installed wheel at `88852b7` (round 84, `r84/attacc_*.json`) and are inherited at `b994fc0`; the Gemini-I rows and the APU v1 installed hardware execution are not available at any revision (the board host's `/tmp`) |
| `ab239e3` (2026-11-10) | the P-27 backend repair; **the packaged working candidate for the remaining work (the reviewer's decision, round 114 review of round 113)**. The `b994fc0` Gemini-II confirmation campaign is carried as explicitly inherited evidence: revision `b994fc0`, zhang-x4 wheel `d5cb59f8...`, the round 106 validation session `r106/g2_candidate_rows.json` and the three round 107 sessions `r107/g2_campaign_s{1,2,3}.json`, pooled in `r107/g2_pooled.json` (recomputed identically in `r109`), outcomes gemm 1.0009 and mvt 1.0001 OPEN (nominal), the other three REPRODUCED, fixed; no further campaign. The reviewer's independent compile of the five paper kernels from the `r106` and `r111` wheels found plans, rankings, program, layout and recipe manifests identical (`reviewer-r113/g2_compile_identity.py`, `g2_{old,new}.json`), which supports carrying the evidence forward; inherited timing is not a fresh measurement | differs from `b994fc0` in `allo/backend/c.py` only (`git diff --stat b994fc0 ab239e3 -- allo setup.py simulators`: one file); Gemini-II modules, templates and benchmark byte-identical | sdist `19a80d2f...`, wheel `d280a4d3...` (`r111/pkg`) | round 111: the UPMEM probe exact with 0 cleanup tracebacks and 0 scratch directories, the use-after-close reproducer refused from the installed package, the backend tests (37) and all sixteen UPMEM rows from the wheel; round 113: the complete local installed milestone (`run_installed_milestone.sh`, every stage PASS, results identical to `b994fc0`'s) and the Gemini-II card tests (15) from a zhang-x4 wheel `11f659ba...` of its own | the Gemini-II validation session and campaign (inherited from `b994fc0` as above, no new samples) and the unavailable Gemini-I / APU v1 hardware evidence; the AttAcc figure results are re-executed from this revision's wheel in round 114 (VALIDATION) |
| `c21b510` and later | records and scripts | none beyond `ab239e3` | none built | ledgers, `docs/release/scripts/`, tests of the scripts | all of the above |

Replacing the packaged candidate by `ab239e3` (or a later revision) requires the full installed-artifact milestone at that revision (`run_installed_milestone.sh`, the Gemini-II chain), not this table.

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
  (sha256 `0ae6a3d9...`, previous binary kept as `pim_driver.bak-20261101`); rebuilt again in round 33 with the address-trace switch (`TENON_CONDUCTOR_ADDRESS_TRACE`) in `TenonConductor.inc`, sha256 `4e4c82ad...`, in round 36 with the plan fields `issue_order`/`split_rows`, sha256 `82db787a...`, and in round 37 with `half_fiber` (the map's rendering) replacing `split_rows`, sha256 `6c50596b...` (previous binaries kept as `pim_driver.bak-20261104-r33`/`-r36`); the cycles of the paper GEMVs are unchanged (8,247 / 8,153 and 8,041 / 7,947).

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
   launch schedule and host moves, C-33b, X-05 to X-08); round 23 made
   transfer availability a shared, schedule-ordered contract (X-09) and
   tied AiM's sliced transfers and host records to the physical layout the
   launched kernel addresses (X-10, X-11); round 24 put UPMEM and AttAcc
   on the contract (every candidate's effects from the shared facts and the
   physical decision or tiling, realization compared at materialization;
   the host-placed k-means update is a declared `host_function`
   candidate), so every released backend declares and is checked against
   the same candidate contract (`EFFECTS_REQUIRED_TARGETS`); round 28 read
   the `APUv1Program` scalar manifest from the retained IR and deleted the
   MLIR-text regex analysis; round 29 ran the slice 5 traces from
   installed wheels on both hardware hosts (forced alternatives and cost
   perturbations reaching generated source and launches, exact outputs,
   VALIDATION round 29). Open: independent review of rounds 28-29, the
   simulators from the installed wheel and the full suites at one
   revision. Map and status: `design/shared-compilation-path.md`.
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
4. **G10 (user addition, 2026-11-03)**: reproduce the frozen paper's
   results through the shared pipeline at matching or better comparable
   performance. Round 25: `PAPER_REPRODUCTION.md` and the machine-readable
   `paper_reproduction/inventory.csv` (82 rows) exist; measured through
   `allo.compile` default selection: Samsung 3/3 faster than the paper
   (outputs checked), UPMEM 16/16 exact (NumPy-exact outputs), AiM 10/14
   exact (timing-only) and 4/14 corrected programs 32 cycles above the
   paper and below CENT (X-19, an OPEN regression under the same-or-better
   rule through round 70; round 71: with the replicated elementwise grids
   realized once over the device (X-32) all 14 rows are at or below the
   paper at the checkout, the 4 RoPE rows REPRODUCED with the complete
   program; round 72: re-executed from the installed wheel of `2096239`), AttAcc shared selection equal to the paper's archived selection
   on 500/500 instances (a diagnostic; the published 398/500 timing-quality
   result is UNVERIFIED without Ramulator2-AttAcc timings); round 30 (the
   archived paired protocols with the baselines re-run in the same rounds,
   from installed wheels, VALIDATION round 30): Gemini-II 5/5 REPRODUCED
   (Tenon medians below the paper's column, the re-run GTML within 0.6% of
   the paper's), Gemini-I matrix multiply and histogram REPRODUCED; kmeans and linear
   regression OPEN (nominal) on the pooled round 30 sessions (pooled medians
   1.0004 and 1.0059 of the paper's, the paper's values inside the pooled
   IQRs, the generated programs the paper's own, X-23; round 31 withdrew the
   single-session closure of kmeans and fixed a pooled-session acceptance
   rule with one bounded confirmation campaign), word count an OPEN regression
   attributed by a five-lane same-round session to the loop-counter width
   of the generated program and repaired (X-22, `dc41df7`): the repaired
   program measures as the paper's own binary measures today (paired
   0.9998) with the paper's advantage over the expert restored (1.0201 vs
   1.0198), 0.19% above the paper's archived median, so OPEN (nominal)
   under the parity rule. UNVERIFIED: token partitioning, UPMEM scaling,
   packing/shared-selection/DCC studies, AttAcc Ramulator2 timings and
   decode composition, calibration, virtual study, compile-cost rows.
5. Open minor/boundary items: V1-16, C-18, S-13, P-17, APU v1 owner maps.

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
