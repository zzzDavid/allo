# Release status

Updated: 2026-11-02 (round 7, final evidence at `1ea8756`). Read `TASK.md` in the containing repository,
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

**NOT READY** pending user decisions and the final hardware re-runs
recorded below; the compiler-side blockers identified by the audits are
closed. Gate judgments are in `REVIEW.md`. What remains:

1. User decisions: redistribution of the Samsung-derived simulator patches
   (P-04); whether to seek a license from the DCC authors for the AttAcc
   evaluation scripts; committing the simulator overlays and skill-doc
   corrections in the containing repository; running the index removals in
   `RELEASE_CHECKLIST.md`.
2. Open minor items: V1-16 (Gemini-I FP16 arithmetic not bit-reproducible
   by an IEEE model; characterized on the board, needs GSI's GVML
   definition), APU v1 realizes only the work-grid owner map, P-17
   (proprietary UPMEM SDK, documented). C-18 is a documented evaluator
   policy and M-14 (an AiM elementwise lowering that dropped work on
   partitioned grids) is fixed.
3. Final evidence at `1ea8756` is recorded in `VALIDATION.md` (PIM suite,
   APU v1 board, Gemini-II card). Any later code commit requires a re-run
   before a READY verdict.

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
