# Release status

Updated: 2026-11-02 (round 2). Read `TASK.md` in the containing repository,
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

**NOT READY.** Gate judgments are in `REVIEW.md`. Blockers still open after
round 2 (see `ISSUES.md` for all):

1. C-01 no access-scope legality; C-09 no reassociation permission in the
   public API (the layout algebra now carries `reassociation_allowed`, but
   nothing sets it from the workload); C-03/C-05 ranking before
   materialization and per kernel; C-11 the ported Section IV search is not
   yet wired into placement selection and `LayoutAccess` rows are not yet
   derived from `MatchedOp`.
2. S-03 Samsung ordered (source-order FP16) fallback is in the simulator
   overlay but not selectable from `allo.compile`.
3. G2-02 no source route to Gemini-II (typed records built by hand); G2-03
   the Gemini-II cost program does not price the typed opcodes; G2-04 no
   hardware test in the suite (hardware evidence exists, VALIDATION.md).
4. M-03 token partitioning and M-04 AttAcc path external to the release;
   U-04 UPMEM uneven partitions; U-05/U-06 UPMEM shape-exact paths.
5. F-02 i32 literal truncation in the shared frontend.
6. P-04/P-06/S-04 simulator reproducibility: Samsung overlay uncommitted in
   the containing repository (and Samsung-licensed patches need the user's
   redistribution decision); the Tenon uPIMulator patch (GENERIC slot,
   fixture loader) exists only in the artifact archive.
7. V1-05 the paper's fitted APU v1 count rule is not in the cost program;
   V1-06 tile choice is a fixed heuristic.

## Round 2 (2026-11-02): what was done

- Frontend: annotation names resolve from the defining scope only (F-01).
- Section IV port: `allo/pim/{gf2,layout_problem,layout_movement,layout_search}.py`
  with the paper's 20,160 / 315 / 576 and 121,968 figures reproduced from
  the package (`design/layout-search.md`).
- Samsung executes the selected program: per-placement CRF lowering, the
  conductor uploads it verbatim, unrealizable levers rejected, structural
  operand roles, padding reported, elementwise ops reachable, MAC cost rule
  corrected (default now beats the vendor program by about 1%; paper rows
  pinned through `samsung_vendor_gemv_layout`); `design/samsung-execution.md`.
- APU v1: structural native recognizers (symbolic executor), carried
  outputs uploaded on temporal plans, board lock and profile freshness,
  ABI-derived output shapes. Device run on this server after the repairs:
  `test_gemv_apu_v1`, two `vector_gemm` plans and the histogram golden
  median row pass (4 passed, `VALIDATION.md`).
- Gemini-II: eleven programs executed on `zhang-x4` `apu-00`, all exact,
  composed paper programs within 1.3% of the paper medians.
- Packaging: package data, wheel with bindings, fresh-venv install verified,
  every external location through `spmw_simenv` with documented variables,
  UPMEM user-build route, `INSTALL.md`, `CAPABILITIES.md`, `simulators.md`,
  README section.

## Next concrete actions (in order)

1. Commit and push round 2 after the final suite runs recorded in
   `VALIDATION.md`.
2. Wire `search_min_movement_layout` into `rank_matcher_placements`: derive
   `LayoutAccess` rows (access matrices, element bytes, reassociation
   policy) and the owner hierarchy from `MatchedOp` and the target; feed the
   tie set as `Placement.layout` candidates; mark `PlacementRanking`
   incomplete on a budget exit (C-11, row 16 of CLAIMS.md).
3. C-09: a reduction-order policy on the workload or `compile(...)`
   consulted by `dual_fiber`, UPMEM `nc/kc` and the APU v1 plan ranking;
   expose Samsung TEMPORAL_REDUCE as the strict-order placement (S-03).
4. C-01 target-neutral legality pass (operand owner reachability and
   `Op.src/dst` patterns) applied to user `layout=` and to enumerators.
5. C-03/C-05: materialize before ranking (convert backend rejections into
   `Rejection`s) and re-cost the reconciled whole program; document the
   tie-break key (C-06).
6. Gemini-II: add a hardware test gated on the card (G2-04), run the
   negative controls, decide G2-02 (source route or documented typed-only
   boundary) and price typed opcodes (G2-03).
7. UPMEM: U-04 uneven partitions, U-05/U-06 template-derived legality,
   publish the uPIMulator patch (P-06).
8. F-02 literal typing in the shared frontend; V1-05/V1-06 cost provenance.

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
