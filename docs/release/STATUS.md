# Release status

Updated: 2026-11-02 (round 3). Read `TASK.md` in the containing repository,
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

**NOT READY.** Gate judgments are in `REVIEW.md` (all PARTIAL after round
3). Open items that still block a READY verdict (see `ISSUES.md`):

1. C-03/C-05: ranking precedes materialization and is per kernel group
   with reconciliation afterwards; the paper's "each complete assignment
   evaluated as one program" is not what runs. Row 16: movement ties reach
   the ranking but no backend can realize an alternative owner map.
2. M-04 AttAcc compiler/virtual-model path is outside the release (needs a
   port or a documented evaluation-only boundary); E11/E12/E14/E15 are
   external reproduction scripts.
3. V1-05/V1-06 APU v1 cost provenance (fitted count rule; fixed tile
   heuristic); F-02 i32 literal truncation in the shared frontend.
4. Simulator reproducibility: Samsung overlay uncommitted in the containing
   repository with Samsung-licensed patches (user decision), the Tenon
   uPIMulator patch unpublished (P-06), UPMEM SDK proprietary.
5. P-08 `SPMW_ARCHITECTURE.md` still describes removed modules; needs a
   rewrite against the current implementation (routes, policies, extension
   points, inspection).

## Round 3 (2026-11-02): what was done

- Target legality pass (`spmw_legality.py`), work-grid forms as errors,
  `lanes` from the placement, `LinearLayout.identity` strictness.
- Reduction-order policy (`reduction_order`, strict by default); Samsung
  ordered program selectable and cost-modeled; APU v1 tree plans pruned
  under strict; AiM/APU v1 matcher routes refuse float under strict.
- Movement stage in `rank_matcher_placements` (`spmw_layout_problem.py`,
  exact coordinate search, ties/termination/optimality recorded,
  structural tie-break).
- Gemini-II source route (five paper kernels as plain source reproduce the
  artifact's fingerprints), card-fitted cost rules, carrier tiling,
  in-process device lock, fingerprint bound to packed carriers, card-gated
  hardware test (10 passed on zhang-x4, 11 exact programs).
- UPMEM: uneven partitions (391x65 sweep), template-derived legality with
  masked padding, carried accumulators loaded, virtual validity labels,
  exact k-means update recognizer, host steps recorded and priced.
- AiM: token-partition knob with explicit priced host reduction (6 of 9
  minima, max gap 1.34%), slice-aware host cost (4.2M -> 10,108 vs 11,010
  simulated), one cost route (SBK 1528 vs 1527), ties resolved, exit codes.
- Device evidence: APU v1 4 tests on the board after the repairs; Gemini-II
  round 3 table in `VALIDATION.md`.

## Next concrete actions (in order)

1. (done in round 3) Commit and push round 3.
2. C-03/C-05: materialize before ranking (convert backend rejections into
   recorded rejections), re-cost the reconciled whole program, and either
   route the matcher through `pim/schedule_search.grid_search` for a
   bounded joint evaluation or document the per-kernel heuristic in
   CLAIMS row 15 and the architecture doc.
3. M-04: decide the AttAcc release boundary; if ported, add the AttAcc
   target and closed-form cost as a virtual backend with the tiling knob.
4. P-08: rewrite `SPMW_ARCHITECTURE.md` from the current tree (routes,
   policies, legality, movement stage, result kinds, extension points,
   inspection) and retire stale `design/` statements.
5. V1-05/V1-06, F-02, remaining minor items; UPMEM public build path
   (publish the uPIMulator patch from the archive diff).
6. Independent-oracle evidence document for G6 summarizing the probes and
   regressions per mechanism.

## External blockers

- Redistribution of Samsung-derived simulator patches needs the user's
  decision (P-04); the UPMEM SDK is proprietary (P-17).
- Committing the simulator overlay in the containing repository is a user
  action (the task restricts commits to `experiments/allo`).

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
