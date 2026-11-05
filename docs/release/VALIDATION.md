<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Validation record

Reproducible commands and pointers to compact raw evidence for the release
candidate. Every result names the revision it was produced at. Evidence
produced by simulators is labelled as such; nothing here is a physical
measurement unless the row says `device`.

## Environment (this server)

```sh
cd /work/shared/users/phd/nz264/spmw-for-pim/experiments/allo
export PYTHONPATH=$PWD:$PYTHONPATH
PY=/work/zhang-capra/users/nz264/miniforge3/envs/spmw/bin/python   # Python 3.12, numpy 2.3.5, pytest 9.1.1
```

`allo/_mlir` is a symlink into `mlir/build/tools/allo/_mlir` (built against
`/work/shared/common/llvm-project-main/build-rhel8`); `externals/llvm-project`
is not checked out here, so `LLVM_BUILD_DIR` is not needed for the Python
path but some tests read it. Simulator discovery (`allo/spmw_simenv.py`):
Samsung `experiments/simulators/PIMSimulator/pim_driver`, AiM docker image
`aim-simulator-build` + `experiments/simulators/aim_simulator/test/example.yaml`,
UPMEM provenance under `~/shared/tenon-artifacts/upmem/provenance/upimulator`
+ docker image `bongjoonhyun/upimulator`. All four reported available on
2026-11-01. APU v1 board: PCI vendor `0x1e4c` at `0000:41:00.0`, toolchain
`/usr/local/gsi-apu/13.7.1`, GVML headers `/usr/local/include/gsi`.

## Baseline at 8160d6d (before repairs), 2026-11-01

| Suite | Command | Result |
|---|---|---|
| PIM (no paper golden, no APU device row) | `$PY -m pytest tests/pim -q -p no:cacheprovider --deselect tests/pim/test_paper_golden.py -rs` | 88 passed, 13 skipped (paper_full), 56 deselected, 331 s; includes `test_gemv_apu_v1` on the board |
| perf | `$PY -m pytest tests/perf -q -p no:cacheprovider -m "not apu_v1_device"` | 96 passed |
| upstream smoke | `tests/test_builder.py tests/test_types.py tests/test_linalg.py` | 124 passed, 2 skipped (packaging audit run); a later run of a wider upstream set shows 4 pre-existing failures (`test_slice`, `test_bf16_array`, `test_concatenate[True/False]`: `RuntimeError: Unknown function slice`) that reproduce identically on a pristine `8160d6d` worktree |

## After repair increment 1 (uncommitted at the time of the run; see STATUS.md for the commit)

| Suite | Result |
|---|---|
| perf (`-m "not apu_v1_device"`) | 112 passed, 93 s |
| upstream smoke (`test_builder test_types test_linalg test_c_backend test_bitop test_popcount test_static_slice`) | 154 passed, 2 skipped, 4 failed (the same 4 pre-existing failures) |
| PIM (no paper golden, no APU device row) | see STATUS.md (run in progress when this file was written; one expected failure `test_gemv_upmem` fixed by updating its expected exception) |
| AiM `paper_full` CENT rows | 11 passed; all 14 CENT traces byte-identical to pinned SHA-256; golden cycles 11,750 / 129,605 / 11,010 / 6,307 reproduce (`fix-aim-axes` agent run) |
| UPMEM `paper_full` golden rows | 13 passed, pinned cycles unchanged (`fix-upmem-dtypes` agent run) |

## Probe evidence (audit stage)

Probe scripts and logs are kept under `/tmp/tenon-release/probes/<area>/` on
this server (not part of the release). The audit files under `audit/` quote
the executed outputs. Key executed observations at `8160d6d`:

- Samsung: forcing `bank_row`, `grf_staged`, `dual_fiber+per_workid+wresident`
  changed `compiled.cmds` (4/6/768 commands) but not argv, CRF, cycles (4435)
  or output bytes; `x = 0, y = 1` returned no outputs and left `y = 1`;
  `acc: f16 = 0.5` returned `W@x`.
- AiM: GEMV spelled `W[ch*ROWS+i, j]` issued 8 MAC_ABK vs 1 for the `row0`
  spelling (simulated 858 vs 222 cycles); 96x256 with mapping [3] issued 341x
  the logical MACs.
- UPMEM: `gemv-int64` ran "successfully" with int32-wrapped results; a
  k-means update with a wide shift returned zeros.
- Gemini-II source route: empty `main()`, `RunResult(cycles=None)`, output
  untouched.
- APU v1: `backend="virtual"` on `APUv1Program` hit the device gate.

## Hardware evidence

| Host | Device | Status at this revision |
|---|---|---|
| this server | GSI Gemini-I (APU v1), PCI 0000:41:00.0 | baseline suite ran `test_gemv_apu_v1` on the board (15 s). Post-repair device run (round 2): see "Round 2 device evidence" below. |
| zhang-x4 (en-ec-zhang-x4) | GSI Gemini-II `apu-00`, PCI 0000:61:00.0, FW `GSI_G2_Infrastructure_V1.24.1`, 4 cores idle, Status Available; SDK `/opt/gsi` (`g2`, `include/64vl`, `gsi_tool`), docker present, Python 3.10 system; shared home visible (`/work/shared/users/phd/nz264/spmw-for-pim` is mounted there too) | inspected 2026-11-01 via the user's SSH config (host key for `zhang-x4.ece.cornell.edu` already in `known_hosts`; the IP alias `128.253.128.31` lacked the ED25519 entry, which caused `Host key verification failed` under `CheckHostIP`; the trusted key was appended for the IP, backup `~/.ssh/known_hosts.bak-20261101`). Executed 2026-11-02: see "Round 2 device evidence". |

## Round 2 device evidence (2026-11-02)

### Gemini-II on zhang-x4 (raw logs: `/work/shared/users/phd/nz264/tenon-release-evidence/g2/`)

Environment: host `en-ec-zhang-x4`, card `apu-00` PCI 0000:61:00.0, FW
`GSI_G2_Infrastructure_V1.24.1`, Status Available, 4 cores idle; SDK
`/opt/gsi` (`gsi_tool` at `/opt/gsi/g2/bin/amd64`); interpreter
`/work/shared/users/phd/nz264/anaconda3/envs/allo-dev/bin/python` (3.12.12,
numpy 2.4.3); the checkout is the same shared working tree as this server
(`ffb4a78` plus the uncommitted wave 2 tree, 59 dirty paths at run time);
every launch wrapped in `flock /tmp/tenon-apu-g2.lock` on the device host.
Scripts: `step2_3.sh`, `step4.sh` in the evidence directory.

| Step | Program | kind | cycles (ticks per call, batch of 20) | Check |
|---|---|---|---|---|
| software gates | `tests/perf/test_apu_g2_typed_program.py`, `test_apu_g2_composed_contraction.py`, `test_apu_g2_target_layout.py` | | | 13 passed |
| 2 | `run_apu_g2_u16_add` (4 x 65536 u16) | device | 236 (pipeline 4,710 raw ticks / 20) | outputs returned, exact |
| 3 | typed add i16+i16->i16 | device | 333 | exact (runtime oracle) |
| 3 | typed mul i8*i8->i16 | device | 312 | exact |
| 3 | typed div u16/u16->u16 | device | 1,019 | exact |
| 3 | typed block_sum u8->u16, K=256 | device | 2,614 | exact |
| 3 | typed dot i4.i4->i15, 1985x128 | device | 1,713 | exact |
| 4 | composed GEMM INT4 64x128x31 | device | 1,711 (paper median 1,733) | exact |
| 4 | composed batched GEMM INT4 4x32x128x16 | device | 1,720 (1,739) | exact |
| 4 | composed GESUMMV INT2 M1024 K128 | device | 1,876 (1,860) | exact |
| 4 | composed BiCG INT3 N256 | device | 2,569 (2,549) | exact |
| 4 | composed MVT INT3 N256 | device | 2,754 (2,767) | exact |

"exact" means the runtime's int64 NumPy oracle comparison passed on the
device readback (the runtime raises otherwise; device values are what is
returned). Cycles are device ARC running-state ticks per call, never host
time; host h2d/d2h/end-to-end microseconds are in the logs separately. The
device library sha256 for all step 4 runs is
`8ea3600c0a60ccc78dd2d1a9f8d1fd9b71ee05912e4f9d5f208d45331a57ffc5`. The
paper medians are listed for orientation only; they were not a pass
criterion. Negative controls (step 5 of `audit/apu-g2.md`) have not been
run yet.

### APU v1 on this server (raw logs: `/work/shared/users/phd/nz264/tenon-release-evidence/apuv1/`)

First post-repair device run (`w2-device-tests.log`): 4 failed, all with the
new profile-freshness check reporting ~2,058 `total` records: each
`ledag-ssh` session's `flo` prints the whole retained device log, so a
"drain before, read after" scheme sees every historical record. Repaired by
diffing the log read before the launch against the log read after it
(`allo.pim.apu_v1_program.fresh_profile_records`). Second run:
`w2-device-tests-2.log` (result recorded in STATUS.md when complete).

## Round 3 (2026-11-02): merged tree after the legality, reduction-policy, layout-selection, Gemini-II source-route, UPMEM and AiM repairs

| Suite | Result |
|---|---|
| golden (`tests/pim/test_paper_golden.py`, default and `paper_full` rows, APU board row deselected) | 55 passed in 838 s: Samsung vendor-layout rows 4,435 / 8,041 / 8,247, 14 AiM CENT rows (traces byte-identical to pins), 13 UPMEM rows with pinned cycles unchanged |
| perf (`tests/perf`, `tests/test_global_vars_scope.py`, `tests/test_c_backend.py`) | 462 passed, 5 failed in the first run; the 5 were test fixtures (a stub lacking the new `reduction_order` argument, a synthetic placement without a register for the `lanes` metric, an assertion on the old enumeration-order tie-break, an AiM bf16 compile needing explicit `reduction_order="reassociate"`); fixtures updated, all pass |
| PIM non-device (`tests/pim`, paper golden and APU board rows deselected, `-m "not apu_v1_device"`) | 266 passed, 24 skipped (`paper_full` rows), 30 min, after the review fixes below |
| APU v1 device (this server) | `test_gemv_apu_v1`, two `vector_gemm` plans, histogram golden median row: 4 passed, 197 s (`apuv1/w3-device-tests.log`) |
| Gemini-II hardware test (`tests/pim/test_apu_g2_hardware.py` on zhang-x4 `apu-00`) | 10 passed, 90 s (`g2/round3b-hardware.log`); an earlier run exposed a missing `artifact_sha256` record in the typed runtime, which was added |

Gemini-II round 3 records (device ARC ticks per call, batch of 20, fresh random inputs, exactness enforced by the runtime oracle; estimate = the new cost program; `plan` = the source route's cost-selected plan):

| Kind | Program | cycles | estimate | plan |
|---|---|---|---|---|
| typed | add i16 | 315 | 340 | |
| typed | mul i8 -> i16 | 311 | 307 | |
| typed | div u16 | 1,014 | 1,020 | |
| typed | block_sum u8 -> u16 | 2,585 | 2,588 | |
| typed | dot i4 -> i15 | 1,725 | 1,714 | |
| source | GEMM INT4 64x128x31 | 1,736 | 1,714 | single/k1 |
| source | batched GEMM INT4 4x32x128x16 | 1,719 | 1,714 | single/k1 |
| source | GESUMMV INT2 M1024 K128 | 1,869 | 1,882 | fused/k1 |
| source | BiCG INT3 N256 | 2,554 | 2,610 | fused/k1 |
| source | MVT INT3 N256 | 2,766 | 2,740 | fused/k1 |
| source | GEMM INT4 64x128x33 (beyond one carrier) | 2,685 | 2,550 (extrapolated) | single/k2 |

The agreement tolerance stated before the run was 10% for calibrated single-launch programs; every calibrated row is within 8%, the extrapolated row within 5.3%. Negative controls passed: a wrong oracle is reported as a mismatch and an echoed fingerprint mismatch is refused by the host binary. The source-route programs carry the same layout fingerprints as the artifact's per-case programs (20a85096, 07f71036, b4afa6b9, a06509fd). Device library sha256 `8ea3600c0a60...` for all rows.

Review fixes applied while merging round 3 (all re-run above): legality and
pattern admission now compare units and handles structurally
(`Unit.path`, `storage_key`), so placements from a second target instance
are judged correctly; the MAC pattern accepts an accumulator addressed
through distinct SSA values (`y[row0 + r]`), which makes the carried
`y[i] += W[i,k]*x[k]` form a MAC with `accumulator_init="carried"` and lets
Samsung run it as the ordered program (`test_carried_accumulator_through_the_public_api`,
simulator, bit-equal to source-order FP16 from the caller's initial `y`);
axis roles resolve result loops through the recorded store-index
dependences; the Gemini-II source route ignores a carried accumulator's own
store when checking for copies. Upstream smoke (same file list as round 2
minus the three pre-existing `test_nn` failures): 313 passed. Golden: 55
passed (781 s). Gemini-II hardware test re-run on the final tree
(`g2/round3c-hardware.log`): 10 passed.

## Round 4 (2026-11-02): selection, AttAcc backend, APU v1 cost, literal typing, simulator recipes

| Suite | Result |
|---|---|
| PIM non-device (`tests/pim`, paper golden and APU board rows deselected) | 311 passed, 25 skipped, 59 deselected in 1776.14s (0:29:36) |
| perf (`tests/perf`, `tests/test_global_vars_scope.py`, `tests/test_c_backend.py`, `tests/test_literal_typing.py`) | 539 passed, 614 s |
| golden (default and `paper_full`, APU board row deselected) | 55 passed, 834 s (Samsung vendor rows, 14 AiM CENT rows byte-identical, 13 UPMEM rows pinned) |
| upstream smoke + three dataflow files | 318 passed; the 3 dataflow failures are the pre-existing OpenMP JIT symbol failures (`__kmpc_*`) that reproduce on a pristine `8160d6d` worktree |
| APU v1 device (this server) | `test_gemv_apu_v1`, two `vector_gemm` plans, histogram golden median row: 4 passed, 199 s (`apuv1/w4-device-tests.log`) |
| artifact-free execution probe (`TENON_ARTIFACTS` and `TENON_UPMEM_PROVENANCE_DIR` pointed at nonexistent paths) | Samsung GEMV `kind="simulator"` 4,389 cycles, max error 4.2e-4 vs FP32; AiM GEMV `kind="simulator-timing-only"` 222 cycles (`/tmp/tenon-release/probes/round3/no_artifacts.log`) |
| AttAcc 500-instance sweep through `allo.compile(..., backend="virtual")` | 500/500 select the paper tiling; 1,565 s wall on a loaded host (`/tmp/attacc-port/sweep500.json`); selector alone 40 s CPU; not the paper's 24 s |
| simulator rebuilds from committed recipes (scratch, installed copies untouched) | Samsung: fresh SAITPublic 3703d1f clone + overlay reproduces 4,435 / 8,041 / 8,247 (vendor layout) and 4,389 / 7,947 / 8,153 (default), binaries identical after stripping build paths; AiM: fresh 0f28a07 clone + Dockerfile (`tenon-aim-simulator-build:0f28a07`) gives 222 cycles; UPMEM: fresh 870d916 clone + `setup/uPIMulator/tenon-upimulator.patch` + new SDK image (`tenon-upimulator-sdk:2023.1.0`) reproduces all 16 paper rows exactly on the user-build route (`sdk_build=user-build:d45e285c34b5`) |

Ramulator2-AttAcc (external, MIT, in another user's DCC checkout, GLIBC 2.34) was run once read-only inside the `aim-simulator-build` image: 11,454 cycles for the qkv batch-1 selected tiling and 953 for the worst-loss attention shape, equal to the archived and paper values. The DCC decode composition script reproduces the archived batch-1 speedups but not batch-4 and is marked unvalidated.

## Round 5 (2026-11-02): Samsung owner maps, minor closures, device checks

### APU v1 board check of the ranked GEMV tiles and the retiled GEMM default (`apuv1/tile_ranking_board.{py,log,json}`)

Paper GEMV 4096x256 FP16, `reduction_order="reassociate"`, fresh random inputs (seed 7), outputs checked against FP32 NumPy (max abs error 4.6e-4 to 6.6e-4):

| Plan | cost estimate | device cycles (CRUN) |
|---|---|---|
| `spatial_gemv_group_reduction_k32` | 1,536,466 | 1,369,443 |
| `spatial_gemv_group_reduction_k64` | 1,458,814 | 1,292,886 |
| `spatial_gemv_group_reduction_k128` | 1,426,276 | 1,258,531 |
| `spatial_gemv_group_reduction_k256` (cost-selected default) | 1,414,983 | 1,243,617 |

The cost program's order equals the device order; tile 64 is 3.96 percent slower than tile 256 on the board (paper: 3.845 percent). Estimates run about 12 to 14 percent high, as the design note predicted.

GEMM 1024x64x1024 FP16 (the 12.71x row's workload), fresh random inputs, outputs within FP16 tolerance (relative error 3.2e-3) on every plan:

| Plan | cost estimate | device cycles |
|---|---|---|
| default, cost-selected (`temporal_dma_coalescing_broadcast_friendly_acc2`) | 6,553,468 | 5,044,711 |
| `temporal_dma_coalescing` (row baseline) | 93,711,488 | 91,622,057 (paper 93,449,834) |
| `temporal_dma_coalescing_broadcast_friendly` (row adapted) | 8,552,316 | 7,147,558 (paper 7,349,987; golden 7,145,109) |

The cost-ranked default, a catalog variant that had never been measured before, runs 1.42x faster than the paper's adapted plan with correct outputs. These are device cycles (CRUN) with no host time; a single launch each.

### Gemini-II card test after the template reduction (`g2/round5-hardware.log`)

`tests/pim/test_apu_g2_hardware.py` on zhang-x4 `apu-00`: 10 passed in 70 s with the reduced project templates (device library and host targets rebuilt on the card; padded divisor lanes hold 1).

### Round 5 suites (final tree)

| Suite | Result |
|---|---|
| PIM non-device (`tests/pim`, paper golden and APU board rows deselected) | 325 passed, 25 skipped, 59 deselected in 2056.99s (0:34:16) |
| perf (`tests/perf`, scope, C backend, literal typing) | 553 passed, 597 s |
| golden (default and `paper_full`, APU board row deselected) | 55 passed, 847 s; pinned Samsung, AiM and UPMEM cycles unchanged |
| Gemini-II card test (zhang-x4) | 10 passed (`g2/round5-hardware.log`) |
| APU v1 board | tile ranking and retiled GEMM default, see above |

## Round 6 (2026-11-02): G6 probe campaign, owner maps on UPMEM and AiM, multi-kernel Samsung

| Suite | Result |
|---|---|
| perf (`tests/perf`, scope, C backend, literal typing) | 555 passed, 630 s |
| golden (default and `paper_full`, APU board row deselected) | 55 passed, 838 s; pinned cycles and CENT traces unchanged |
| upstream smoke | 313 passed (same pre-existing environment deselections) |
| PIM non-device | 395 passed, 30 skipped, 66 deselected, 1 xfailed in 3390.56s (0:56:30) (the one xfail is the C-18 evaluator anomaly) |
| APU v1 board (`tests/pim/apu_v1`, `test_gemv_apu_v1`, histogram golden median) | first run 9 passed, 2 failed on bit-equality with the host FP16 models (finding V1-16); after splitting the test: 2 value tests passed, 2 bit-equality tests xfailed as designed (`apuv1/w6-device-tests.log`, `apuv1/w6-bitmodel.log`): strict plan 259/512 and 194/384 outputs differ from the source-order model by a few ulps, relaxed plan 365/512 and 289/384 from the halving tree, all within FP16 tolerance of FP32 |
| Gemini-II card (`tests/pim/test_apu_g2_hardware.py`, now 15 tests incl. aliases, extremes, zero results, fused versus separate BiCG, typed div with garbage lanes) | 15 passed, 123 s (`g2/round6-hardware.log`) |

Simulator evidence added by the probes (all with fresh random inputs and NumPy oracles; details in the test docstrings): Samsung cost perturbation (bank_row + host fill 4,389 cycles versus the perturbed model's pick bank_row + CRF fill 4,589, the stock pick faster as predicted), UPMEM perturbation (fused 114,534 versus separate 131,002), Samsung 8-channel variant executes in 4,389 cycles, renamed Samsung/UPMEM GEMVs give identical decisions and cycles, two-kernel Samsung chain executes (2,813 + 2,813 reassociate; 7,485 + 7,485 strict) with `z` bit-identical to the second kernel compiled alone, UPMEM k-means distances plus host update exact on the simulator, UPMEM owner-map ties 595,371 versus 593,932 cycles (random data; 604,829 both on equal rows) with exact outputs, AiM owner-map ties identical traces (507 cycles).

## Round 7 (2026-11-02): Gemini-I FP16 arithmetic characterization (V1-16) and AiM elementwise fix

Board probes (`apuv1/fp16_characterization.{py,log,json}`, `apuv1/fp16_rounding_probe.{py,log,json}`), strict temporal plan, fresh random FP16 inputs:

| K | rows | device vs per-op FP16 RN model | vs FP32 accumulate then round | vs FMA-style FP16 accumulator | max relative error vs FP32 |
|---|---|---|---|---|---|
| 1 | 512 | 83 mismatches | 83 | 83 | 7.0e-4 |
| 2 | 512 | 33 | 202 | 161 | 1.4e-2 |
| 4 | 512 | 92 | 276 | 217 | 0.19 (cancellation) |
| 16 | 512 | 143 | 332 | 303 | 0.10 |
| 256 | 512 | 250 | 462 | 403 | 0.12 |

Rounding-mode test on the single product (K=1, 1,024 rows): round-to-nearest 66 mismatches, toward zero 435, down 514, up 510, away from zero 589. On K=2 with RN products: per-op add RN 188 mismatches; toward zero 378; a single rounding of the exact sum 429. Conclusion: the device rounds per operation and is RN-like, but its FP16 multiply is not an exact-product round-to-nearest, so no IEEE model reproduces it bit for bit; a vendor-level definition of the GVML FP16 arithmetic is required for bit-exact modelling (V1-16 open). The relative errors are FP16-class.

AiM elementwise over a partitioned 32-item grid (M-14): the lowering had emitted one work item's slice on one channel (`EWMUL 2 2147483648`, 128 of 4,096 elements); it now spreads the grid over the channels (`EWMUL 2 4294967295`, all 4,096 elements, 30 simulated cycles against an estimate of 28) or issues the whole vector for ADD (`EWADD 256`), and refuses grids that do not cover the operands; the ADD cost rule prices the issued columns (8 -> 256 for the 32-item case). The evaluator's non-monotone response to a per-step cost (C-18) is a documented list-scheduling anomaly of its backfilling placement policy, now asserted by a regular test with the worked example.

Round 7 regression: `tests/perf` 510 passed (617 s); AiM suites, `test_g6_decisions.py`, `test_gemv_all_targets.py` and the full golden file (default and `paper_full`) 134 passed (2,337 s), pinned cycles and CENT traces unchanged.

## Final evidence at the release candidate `1ea8756` (2026-11-02)

The runs below were made on the committed tree of `1ea8756` (working tree
clean apart from the pre-existing untracked `scripts/verifier_check_*.py`);
the documentation commit that records them changes no code.

| Evidence | Result |
|---|---|
| PIM non-device (`tests/pim`, paper golden and APU board rows deselected, `-m "not apu_v1_device"`) | 398 passed, 30 skipped, 68 deselected in 3606.77s (1:00:06) (the skips are `paper_full` rows and card-gated Gemini-II tests; the xfail is the documented C-18 schedule check) |
| perf (`tests/perf`) | 510 passed at this tree (round 7 run) |
| golden (default and `paper_full`) | 55 passed in the round 7 subset at this tree; pinned Samsung, AiM and UPMEM cycles and the 14 CENT traces unchanged |
| APU v1 board (this server, `tests/pim/apu_v1`, `test_gemv_apu_v1`, histogram golden median) | 11 passed, 2 xfailed (V1-16 bit-model checks), 485 s (`apuv1/final-device-tests.log`) |
| Gemini-II card (zhang-x4 `apu-00`, `tests/pim/test_apu_g2_hardware.py`) | 15 passed, 124 s (`g2/final-hardware.log`) |
| upstream smoke | 313 passed at the round 6 tree; the only files changed since are AiM backend and evaluator files not imported by those tests |

## Round 8 (2026-11-02): reviewer probes, matcher cast/copy contracts, simulator recipes

Environment as above (`spmw` env, `PYTHONPATH=$PWD`,
`LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build-rhel8`). Raw
logs: `/tmp/tenon-release/r9/*.log` on this server, copied with the scripts
and the reviewer's probe files to
`/work/shared/users/phd/nz264/tenon-release-evidence/r8/`.

Independent reviewer's probes (their scripts, unchanged:
`/tmp/tenon_reviewer_samsung_cast.py`, `/tmp/tenon_reviewer_samsung_guard.py`):

| Probe | At e41ebbd (reviewer's run) | Round 8 tree |
|---|---|---|
| FP16 GEMV, `out[row] = int(acc)`, W[:,0]=0.25, x=1 (source: all outputs 0) | compiled, `kind='simulator'`, returned 0.25 everywhere | `UnsupportedWorkloadError`: "computes arith.sitofp<i32->f16>(arith.fptosi<f16->i32>(acc[])) ... which matches no target op" (`probe-cast.log`) |
| FP16 GEMV, `if v[0] > 0: out[row] = acc`, x=-1, y initialized to 99 (source: all outputs stay 99) | compiled, overwrote every output with -0.25 | `UnsupportedWorkloadError`: "is a copy of a matched result under a different guard than its producer (copy guards if(%9), producer guards none); a conditional output copy would be executed unconditionally by the device" (`probe-guard.log`) |

Both probes are regressions in `tests/pim/test_match_coverage.py`
(`test_value_changing_conversion_on_the_output_is_rejected`,
`test_conditional_output_copy_is_rejected`), with the neighbours that must
keep compiling (`acc: fp16 = 0` constant initializer, `i4*i4` accumulation
truncated to i15, `i32 + i32` pointwise, bf16 activation rounded at the
store, loop-index narrowing) and the neighbours that must be refused (copy
inside the reduction loop, truncating copy of an unbounded int64
accumulator, f32 constant stored into bf16, `uint8` products copied into
int17).

Suites on the round 8 tree (before the commit; the committed tree differs
only by the ledger text written after these runs):

| Suite | Command | Result |
|---|---|---|
| matcher contracts, legality, run-result contract | `pytest tests/pim/test_match_coverage.py tests/perf/test_legality.py tests/perf/test_run_result_contract.py` | 59 passed (`core.log`), plus the later `test_unsigned_loads_use_the_unsigned_range` (1 passed) |
| CENT AiM traces and UPMEM dtypes | `pytest tests/pim/test_cent_aim_workloads.py tests/pim/test_upmem_dtypes.py` | 25 passed, 11 skipped (`cent-upmem.log`; the skips are `paper_full` rows) |
| Gemini-II source route | `pytest tests/perf/test_apu_g2_source_route.py` | 48 passed (`g2.log`) |
| perf | `pytest tests/perf -q -rfxs` | 534 passed in 614.91 s (`final-perf.log`) |
| PIM non-device (`tests/pim -m "not apu_v1_device"`) | `pytest tests/pim -q -rfxs -m "not apu_v1_device"` | 438 passed, 54 skipped, 12 deselected, 3 failed in 4033.60 s (`final-pim.log`). The three failures were test expectations overtaken by the new contracts, each re-judged on its semantics: the Gemini-II alias probe used an `int16` operand with an `int15` accumulator (now refused visibly for the truncation; the alias probe keeps `int15` and the wide-operand case is its own test); a `uint32` pointwise add and an `int32` affine form with the constant 2,147,483,648 were refused by the first narrowing rule before UPMEM's ABI and literal checks could name the real problem, fixed by judging narrowing leaves by bit width (ISSUES C-19) |
| re-run after those fixes | `pytest tests/pim/test_upmem_dtypes.py tests/pim/test_upmem_generality.py tests/pim/test_g6_semantics.py -m "not apu_v1_device"` | 119 passed, 1 skipped (`refix-a.log`) |
| re-run after those fixes | `pytest tests/pim/test_match_coverage.py tests/perf/test_apu_g2_source_route.py tests/pim/test_cent_aim_workloads.py tests/perf/test_legality.py` | 96 passed, 11 skipped (`refix-b.log`) |

Suites on the committed tree `9cd8fb3` (checkout, `spmw` env, logs
`commit-pim.log`, `commit-perf.log`):

| Suite | Command | Result |
|---|---|---|
| PIM non-device | `pytest tests/pim -q -rfxs -m "not apu_v1_device"` | 442 passed, 54 skipped (`paper_full` rows, card-gated Gemini-II tests), 12 deselected, 0 failed in 3877.56 s (1:04:37) |
| perf | `pytest tests/perf -q -rfxs` | 534 passed in 686.37 s |

### Installed-artifact runs at `9cd8fb3` (2026-11-02)

Build: `setup.py sdist` from the clean committed tree, then `pip wheel` with
the LLVM 22 clang and the `LDFLAGS` of INSTALL.md section 6
(`/tmp/tenon-release/r9/wheel-build.log`); the sdist contains `simulators/`
(no `patches/`, no `scripts/verifier_check_*`). Wheel
`allo-0.5-cp312-cp312-linux_x86_64.whl`, sha256
`788e5e13f38bef03f12f9d907c2db7da3a5ec7a02d8e956a74695654732a2e69`. Installed
into a clean venv (`/tmp/tenon-release/r9/pkg/venv2`) with `--no-deps
--force-reinstall`; every run below used `cd /tmp`, `env -u PYTHONPATH`,
`TENON_SIMULATORS_DIR=<repo>/experiments/simulators` and the installed
simulators of this server (`/tmp/tenon-release/r9/run_installed.sh`, log
`installed2.log`; scripts in `/tmp/tenon-release/r9/scripts/`).

| Check | Result |
|---|---|
| provenance | `allo` from `venv2/.../site-packages/allo/__init__.py`; modules loaded from the checkout: none; Gemini-II and IR templates shipped |
| Samsung GEMV 4096x256 FP16, `backend="virtual"` | kind `virtual`, estimate 27,598 (strict TEMPORAL program) |
| Samsung GEMV 4096x256 FP16 on PIMSimulator, random inputs | kind `simulator`, 26,807 cycles, max abs error 2.2e-3 against FP32 |
| AiM GEMV 256x256 BF16 over 32 channels, `reassociate`, y pre-filled with 7.0 | kind `simulator-timing-only`, 222 cycles, `outputs == {}`, caller array untouched; chosen `all_bank+bank_scope=abk+...` |
| UPMEM GEMV 768x128 int32 over 8 DPUs, random inputs in [-1000, 1000), y pre-filled | kind `simulator`, 563,207 DPU cycles, outputs exact against the int64 oracle |
| reviewer cast probe through the installed module | `UnsupportedWorkloadError` naming `arith.sitofp<i32->f16>(arith.fptosi<f16->i32>(acc[]))` |
| reviewer guard probe through the installed module | `UnsupportedWorkloadError` naming the guard mismatch (`copy guards if(%9), producer guards none`) |

| APU v1 board (this server, PCI `0000:41:00.0`, shared `flock`), tests run from the installed wheel with `cd /tmp`, `PYTHONSAFEPATH=1`, `env -u PYTHONPATH` (`run_board.sh`, log `board-installed.log`) | `allo` from `venv2`; `test_vector_gemm_apu_v1[temporal_dma_coalescing]`, `[temporal_dma_coalescing_broadcast_friendly]` and `test_apu_v1_board_median_cycles[apu_v1-histogram]`: 3 passed in 180.83 s (device builds, launches and readback through the installed package). `tests/pim/test_g6_semantics.py` and `apu_v1/test_g6_board_semantics.py` cannot be collected outside the checkout because they import evaluation workloads from the repository's `benchmarks/` tree, which is not part of the package; they ran from the checkout in the full PIM suite |

Found on the way: the hoisted-row-base GEMV form is refused by the UPMEM
family classifier (U-16, open); installing into a `--system-site-packages`
venv over the editable development environment does not install the wheel,
and running `python -m pytest` with the checkout as the working directory
puts the checkout's `allo` first on `sys.path` even inside a clean venv
(P-22; use a working directory outside the checkout and `PYTHONSAFEPATH=1`).
Not done at this revision: the Gemini-II card run from the installed
artifact on zhang-x4 (G8 stays UNVERIFIED there at `9cd8fb3`; APU v1 and
the three simulators have installed-artifact evidence above).

## Round 9 (2026-11-02): second reviewer rejection (float precision, output maps)

Environment as in round 8. Raw logs: `/tmp/tenon-release/r9/{probes-r9,r9a,r9b,final9-pim,final9-perf}.log`
on this server, copied to `/work/shared/users/phd/nz264/tenon-release-evidence/r9/`.
The reviewer's scripts and logs are under
`/work/shared/users/phd/nz264/tenon-release-evidence/reviewer-r8/`.

| Probe (reviewer's script, unchanged) | At ee8f522 (reviewer) | Round 9 tree |
|---|---|---|
| `tenon_review_r8_float_overflow.py`: one work item, `y[i] = float(a[i])*float(b[i])+float(c[i])`, FP16 arrays; a=b=256, c=-65504 (source: 32) | compiled, Samsung simulator returned inf everywhere | `UnsupportedWorkloadError`: "computes arith.truncf<f32->f16>(add(mul(a[%arg4], b[%arg4]), c[%arg4])) ... matches no target op" |
| `tenon_review_r8_reversed.py`: FP16 GEMV 128x16, mapping [16, 8], `out[M-1-row] = acc`, W[:,0]=arange, x=1 (source: 127..0) | compiled, returned 0..127 | `SamsungUnsupported`: "the output map is not the identity the lowering realizes: operand 'w' is read at coordinate 0 but the result is stored at [127]" |

Suites and installed-artifact runs at `7bdb475` (the repair commit; the
ledger commit after it changes no code):

| Check | Result |
|---|---|
| affected groups before the commit (`r9c.log`, `r9d.log`) | joint selection, layout selection/problem, legality, Samsung performance, match coverage, AiM axis roles, AttAcc virtual: 201 passed, 1 skipped; layout-swizzle probe, Samsung lowering, GEMV on all targets, G6 semantics, CENT traces, Gemini-II source route: 105 passed, 11 skipped |
| perf (`final9-perf.log`) | `pytest tests/perf -q -rfxs`: 534 passed in 619.72 s |
| PIM non-device (`final9-pim.log`) | `pytest tests/pim -q -rfxs -m "not apu_v1_device"`: 449 passed, 54 skipped, 12 deselected, 1 failed in 3737.07 s. The failure, `samsung_hbm_pim/gemv/test_gemv_samsung_readback.py::test_paper_gemv_zero_input_writes_zero_output`, died in `numpy.save` with `OSError: 1048576 requested and 1984 written`: `/tmp` was full at that minute (the wheel build running alongside, `wheel-build9.log` line 582 shows the same `Errno 28`); after freeing space the file passes alone (3 passed, 20.75 s). Not a code defect; the run is otherwise clean |
| wheel | built from the clean committed tree (`wheel-build9.log`), `allo-0.5-cp312-cp312-linux_x86_64.whl` sha256 `900b92bdc71a60cdc29c5cf2255897c11b6629ca7e86bd396e1869e97e04bae7`; sdist sha256 `6f5c9763c6f936db884277b655125441b4c9acce3e0dde065d93c3995fcc7a02` |
| installed (clean venv `pkg9/venv2`, `cd /tmp`, no `PYTHONPATH`; `installed9.log`) | `allo` from the venv, no checkout modules; Samsung virtual estimate 27,598; Samsung simulator 26,807 cycles, max abs error 2.2e-3; AiM timing-only 222 cycles, outputs untouched; UPMEM 563,207 cycles, exact; all four reviewer probes (round 8 cast and guard, round 9 float overflow and reversed rows) rejected with the semantic reason |
| APU v1 board from the installed wheel (`board-installed9.log`, shared `flock`) | `test_vector_gemm_apu_v1[temporal_dma_coalescing]`, `[temporal_dma_coalescing_broadcast_friendly]`, `test_apu_v1_board_median_cycles[apu_v1-histogram]`: 3 passed in 179.72 s |
| Gemini-II card from an installed wheel built on zhang-x4 (`/work/shared/users/phd/nz264/tenon-release-evidence/r9/{g2_installed_run.sh,g2-installed-run.log,g2/}`) | host `en-ec-zhang-x4`, card `apu-00` Available, FW `GSI_G2_Infrastructure_V1.24.1`; the sdist of `7bdb475` (sha256 `6f5c9763...7a02`) built into a wheel on that host with its gcc 11 against the shared LLVM 22 build (ninja and cmake 4.4.4 from the throwaway venv; the LLVM build's zstd dependency, absent on that host, mirrored from the build host under `r9/zstd-install/`, see `ORIGIN.txt`), wheel sha256 `65c09608895f91353789fba5e5db0c2a9853e9a24dab96053085c2910ad0ddcc`; installed into a clean venv of the `allo-dev` interpreter (Python 3.12.12), run from `/tmp` with `PYTHONSAFEPATH=1` and only a copy of `benchmarks/` on `PYTHONPATH`; `allo` imported from the venv; `tests/pim/test_apu_g2_hardware.py` under `flock /tmp/tenon-apu-g2.lock`: **15 passed in 145.81 s** (device builds, launches, readback and the runtime's int64 oracle checks through the installed package) |

## Round 10 (2026-11-02): third reviewer rejection (elementwise maps, dropped copies)

Reviewer material: `/work/shared/users/phd/nz264/tenon-release-evidence/reviewer-r9/`
(`tenon_review_r9_maps.py`, logs). Raw logs of this round:
`/tmp/tenon-release/r9/{r10a,r10b,lint10c,final10-pim,final10-perf,installed10,board-installed10}.log`,
copied to `/work/shared/users/phd/nz264/tenon-release-evidence/r10/`.

| Probe (reviewer's script, unchanged) | At 620a230 (reviewer) | Round 10 tree |
|---|---|---|
| `reverse_add`: one work item, `y[N-1-i] = a[i] + b[i]`, a = arange % 128, b = 1 (source: 128, 127, ...) | compiled, simulator wrote forward order (1, 2, ...) | `SamsungUnsupported`: "the result is stored at coordinate ['-1*%arg3 + 4095'], not at the element loop index %arg3; ... a reversed, shifted or broadcast destination is not realized" |
| `copy_add`: `t[i] = a[i] + b[i]; y[N-1-i] = t[i]` | compiled, t correct, y untouched (99) | `UnsupportedWorkloadError`: "copies the result produced at element [%arg4] to element [-1*%arg4 + 4095]; a copy is realized as the same element of another buffer, and no target lowering realizes a permuted or shifted copy" |
| `plain_copy_add`: `t[i] = a[i] + b[i]; y[i] = t[i]` | compiled, y untouched (99) | compiles; `kind='simulator'`, y == t == a + b (both written; `RunResult.extra["copies"] == ["y"]`) |
| same programs on AiM (`probe_maps_other.py`) | `aim_rev` and `aim_copy` compiled (timing-only, wrong effect description) | `aim_rev` refused (output map), `aim_copy` refused at compile time: "the lowering does not realize the write(s) to 'y' in function 'calc_0' (copy store)"; UPMEM refused both already |

Suites and installed-artifact runs at the round 10 repair commit `67f89fd`
(the lint commit after it, `3418938`, changes formatting, headers and two
script mode bits only; the wheels below were built from the tree whose
`allo/` package is identical to both):

| Check | Result |
|---|---|
| regression groups before the commit (`r10a.log`, `r10b.log`) | match coverage, run-result contract, Samsung lowering, G6 semantics, legality: 112 passed after the partitioned-grid message update; AiM axis roles, CENT traces, token partition, AttAcc virtual, UPMEM dtypes/generality, GEMV on all targets, host-program analysis: 171 passed, 14 skipped |
| perf (`final10-perf.log`) | `pytest tests/perf -q -rfxs`: 533 passed, 1 failed (`test_simulator_recipes.py::test_apply_scripts_are_executable`: the header insertion had dropped the two `apply.sh` executable bits; restored in the lint commit, the file passes: 24 passed) |
| PIM non-device (`final10-pim.log`) | `pytest tests/pim -q -rfxs -m "not apu_v1_device"`: 455 passed, 54 skipped, 12 deselected, 0 failed in 3820.82 s (1:03:40) |
| wheel (this server) | `wheel-build10.log`; sha256 `6f899ef7e90bad0b5a7754697782a5c43d40b360a1ec86701fd3c1f16e7af2ec`; installed into a clean venv `pkg10/venv2` |
| installed (`installed10.log`, `cd /tmp`, no `PYTHONPATH`) | `allo` from the venv only; Samsung virtual 27,598; Samsung simulator 26,807 cycles, max abs error 2.2e-3; AiM timing-only 222 cycles, outputs untouched; UPMEM 563,207 cycles, exact; the round 8 and 9 reviewer probes rejected |
| reviewer's round 10 probe through the installed wheel (`installed10-maps.log`) | `reverse_add` refused (destination coordinate), `copy_add` refused (permuted copy), `plain_copy_add` runs on the simulator with y == t == a + b |
| APU v1 board from the installed wheel (`board-installed10.log`, shared `flock`) | 3 passed in 203.25 s (two vector GEMM plans, histogram golden median) |
| Gemini-II card from a wheel built on zhang-x4 (`r10/g2-installed-run.log`, `r10/g2/`) | sdist sha256 `70a3346d...3736`, wheel sha256 `64d2a72a5c416a39e421f2db876c80e73b6b05839d87b0ecb33f02a336a7fb56`, clean venv of the `allo-dev` interpreter, run from `/tmp` under `flock /tmp/tenon-apu-g2.lock`: `tests/pim/test_apu_g2_hardware.py` 15 passed in 146.93 s |

Lint (`bash scripts/lint/task_lint.sh`, `lint10d.log`): the license-header,
black and clang-format stages pass against both HEAD~1 and origin/main after
the round 10 lint commit; the pylint stage exits with 365 pre-existing
findings across the fork (P-23), so the complete command still fails at
that last stage.

## Round 11 (2026-11-02): fourth reviewer rejection (destination identities, arithmetic precision)

Reviewer material: `/work/shared/users/phd/nz264/tenon-release-evidence/reviewer-r10/`
(`tenon_review_r10_effects.py`, `tenon_review_r10_wide_output.py`, logs).
Raw logs of this round: `/tmp/tenon-release/r9/{r11a,final11-pim,final11-perf,installed11,board-installed11}.log`,
copied to `/work/shared/users/phd/nz264/tenon-release-evidence/r11/`.

| Probe (reviewer's script, unchanged) | At f732b22 (reviewer) | Round 11 tree |
|---|---|---|
| `renamed_sink`: `t[i] = a[i] + b[i]; out[i] = t[i]` (T and Y external) | compiled, binding `{'out': 'Y'}`, Y written, T left at 98 | binding `{'t': 'T', 'out': 'Y'}`; simulator outputs `['t', 'out']`; T and Y both equal a + b |
| `larger_output`: 4097-element output for a 4096-element loop | compiled; failed only at readback with a shape error | refused at compile time: "the output 'y' is a f16 buffer of shape (4097,) but the element loop writes 4096 FP16 results" |
| `wide_sum` / `wide_product`: FP16 inputs, FP32 output, `float(a[i]) + float(b[i])` and `float(a[i]) * float(b[i])` (source 65536) | compiled, simulator returned inf | refused: "computes add(a, b) ... the source evaluates this operation in f32 on f16 operands ... a target op computes in its operands' type f16" |
| `narrow_then_widen` control: `y_f32[i] = a[i] * b[i]` | inf (correct) | compiles and executes: inf, equal to the source |

## Round 12 (2026-11-02): gate G9 slices 1-2 (shared front stages, semantic facts)

Interpreter `/work/zhang-capra/users/nz264/miniforge3/envs/spmw/bin/python`
(3.12.13, numpy 2.3.5, pytest 9.1.1), `LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build-rhel8`,
checkout as working directory. Raw logs under `/tmp/g9/`, copied to
`/work/shared/users/phd/nz264/tenon-release-evidence/r12/`.

Inherited round 11 checkpoint, as found (uncommitted at `f732b22`,
`baseline-a.log`): `test_apu_v1_vector_compile/tile_selection/reduction_policy`,
`test_apu_g2_source_route`, `test_compile_interface`, `test_match_coverage`:
7 failed, 93 passed, 18 errors. Causes: the C-26 contract refused the
Gemini-II `gesummv` epilogue (integer intermediate; 16 errors/failures),
`test_elementwise_destination_must_be_the_element_read` saw the new extent
message first, `test_identity_copies_are_required_outputs` expected the old
`out` key, `test_dataflow_build_does_not_validate_retained_scopes` needs
`LLVM_BUILD_DIR`. After the bounded reconciliation (`checkpoint-recon.log`):
`test_match_coverage` + `test_apu_g2_source_route`: 87 passed.

| Check (round 12 tree) | Result |
|---|---|
| probe of the guarded contraction (`probe_guard_old2.py`) against the installed round 10 wheel (`/tmp/tenon-release/r9/pkg10/venv2`, `cd /tmp`, `python -I`) | `guarded_outer` (`if r < 4:` around the reduction): compiled, plan `temporal_svp`, axis extents `{col: 8, arg4: 4, d: 16}` (whole 8-row problem; V1-17); `guarded_rows`/`guarded_data` refused for a wrong reason ("reduction axis 'd' indexes the output"); functional run of `guarded_outer` crashed (`ValueError: setting an array element with a sequence`) |
| same probe on the round 12 tree | all three refused by the shared coverage check: "store ... under a guard: computes add(C, mul(L, R)) ... which matches no target op" |
| smoke (`smoke1.py`): GEMM carried, GEMM with local accumulator, 1-D GEMV on APU v1 (`backend="functional"`), Gemini-II gemm | plans `baseline_spatial_reduction` x2 (`output_initializer` carried / zero), `spatial_gemv_group_reduction_k32`; max abs error vs float32 reference 2.2e-3, 2.2e-3, 8.0e-3; Gemini-II `single/k1`, packing form `pack_apu_g2_batched_gemm_dots` |
| `tests/pim/test_shared_semantics.py` (new) | 10 passed |
| `test_apu_g2_source_route`, `test_compile_interface`, `test_g6_semantics -k "apu or unsupported or compile" -m "not apu_v1_device"` (`g9-c.log`) | 88 passed, 13 deselected (five expectations moved to the shared `UnsupportedWorkloadError`: APU v1 alias, Gemini-II aliases, Gemini-II wide operand, APU v1 nonzero/partial initializers; three Gemini-II unsupported-form rows and the pipeline monkeypatches updated) |
| matcher/APU regression (`g9-d.log`): `test_match_patterns`, `test_aim_matcher`, `test_upmem_matcher`, `test_layout_selection`, `test_legality`, `test_matcher_ranked_placement`, `test_host_transfer_dispatch`, `test_aim_axis_roles`, `test_reduction_policy`, `test_apu_v1_vector_compile`, `test_apu_v1_tile_selection`, `test_apu_v1_reduction_policy`, `test_run_result_contract`, `test_apu_g2_typed_program`, `test_apu_g2_composed_contraction`, `test_import_surface` | 130 passed, 2 skipped in 544 s |
| `tests/pim/test_gemv_all_targets.py` (Samsung and AiM simulators, UPMEM uPIMulator, APU v1 board under the board lock; `g9-e-gemv.log`) | 6 passed in 141 s: Samsung simulator, Gemini-II source rejection, AiM simulator (two tests), UPMEM uPIMulator and `test_gemv_apu_v1` on the Gemini-I board, all through the shared front stages (the board launch went through `compile_apu_v1_vector_workload(program=...)` with the facts-derived analysis) |
| black on the changed files | clean; pylint on the new modules: no findings (`Contraction` carries a documented `too-many-instance-attributes` disable) |

Not run this round (pending, slice 5): the full `tests/perf` and `tests/pim`
suites, the installed-artifact runs, the Gemini-II card. The round 12 code
is the first increment of G9 and is submitted for architectural review
before those runs.

## Round 13 (2026-11-02): review findings C-27 and G2-13; gate G9 slice 3

Same interpreter and `LLVM_BUILD_DIR` as round 12; checkout as working
directory. Raw logs under `/tmp/g9/s3-*.log`, copied to
`/work/shared/users/phd/nz264/tenon-release-evidence/r13/`.

| Check (round 13 tree) | Result |
|---|---|
| the reviewer's round 12 probes (`tenon-release-evidence/reviewer-r12/tenon_review_g9.py`, rerun unchanged: `reviewer-r12-probes-after.log`; and every probe guarded: `reviewer_r12_probes_guarded.py`, `reviewer-r12-probes-guarded.log`) | `int_square_helper`, `square_helper`, `twice`, `through_helper`, `int_helper`: refused before matching with `UnsupportedWorkloadError` naming the call ("function 'twice' calls 'square_leaf' 2 times"); `intleaf` on `build_apu_v2_target()`: compiles (`single/k1`); `nonsquare_acc`: compiles with the transposed output access `C[c, r]` retained |
| `tests/pim/test_shared_semantics.py` (now 19: facts, planners, shared rejections, function boundary on APU v1 / Gemini-II / Samsung, both Gemini-II builders, both APUs' plans as candidates of the shared ranking with forced alternatives reaching the generated GVML source and the launch list) | 19 passed |
| APU v1 vector compile/tile/reduction-policy, Gemini-II source route, compile interface, run-result contract, work grid, legality, matcher ranked placement, reduction policy, shared semantics (`s3-a.log`; the Gemini-II and interface files rerun after the last test reference fix: 69 passed) | 173 passed, then 69 passed |
| G6 APU semantics (`-k apu`, no board), match coverage, G6 decisions `-k "apu or decision or flip"` (`s3-b.log`) | 44 passed, 51 deselected in 701 s (the APU v1 device-route recorder now returns the staged output, as the executed-run contract requires) |
| `tests/pim/test_gemv_all_targets.py` (`s3-e-gemv.log`): Samsung and AiM simulators, UPMEM, `test_gemv_apu_v1` on the board through the common candidate pipeline | 5 passed, 1 failed in 144 s; the failure (`test_gemv_apu_v2_source_rejected`) expected the Gemini-II family message for a dataflow region, which the enumerator now raises (`APUG2SourceRouteError`); the test passes alone after that change (1 passed, 11 s). The board GEMV: plan selected by `rank_matcher_placements`, materialized by `APUv1Ctx.emit_program`, launched by `APUv1VectorProgram` |
| broader matcher and typed-route regression (`s3-d.log`): match patterns, AiM/UPMEM matchers, layout selection, joint selection, host transfer dispatch, AiM axis roles, Gemini-II typed and composed records, import surface, APU v1 scalar surface and recognizers | 138 passed, 2 skipped, 2 failed in 558 s; the two failures (`test_apu_v1_recognizers.py`) were contracts of the removed route object that the shared driver had not taken over: a virtual run with a nonzero initial output on a plan without output ingress must still be refused (now the context's `virtual_run_check` hook), and `cost=None` with `layout=<name>` must materialize that candidate without ranking while `cost=None` alone must say that candidates are selected by cost ranking (`_select_named_candidates`); after these repairs `test_apu_v1_recognizers.py`, `test_run_result_contract.py`, `test_apu_v1_vector_compile.py` and `test_shared_semantics.py`: 86 passed |
| black on the changed files; pylint on the new modules | clean; remaining pylint findings in the touched files are pre-existing (unused imports in `apu_g2_source_route.py`, dict literals) or structural (`too-many-*`) |

Test expectations changed in this round, each judged on semantics:
`isinstance(compiled, CompiledCallable)` replaces the removed route class;
Gemini-II tests read the realized program through the forwarded surface
(`compiled.plan`, `compiled.contract`, `compiled.backend_program.placement_ranking`);
`layout=` with an unknown name reports "names no candidate" (shared
message); `host_moves=[]` is accepted as "no explicit moves" on every
target; the APU v1 device-route recorder returns the staged output.

Not run this round: the full `tests/perf` and `tests/pim` suites, the
installed artifact, the Gemini-II card (slice 5).

## Round 14 (2026-11-03): review findings C-28 and C-29; candidate contract; text parser removed

Same interpreter and `LLVM_BUILD_DIR` as round 12; checkout as working
directory. Raw logs under `/tmp/g9/s4-*.log`, copied to
`/work/shared/users/phd/nz264/tenon-release-evidence/r14/`.

| Check (round 14 tree) | Result |
|---|---|
| the reviewer's round 13 probes (`tenon-release-evidence/reviewer-r13/`, rerun unchanged; `*-after.log` in `r14/`) | `plan_binding_probe`: APU v1 stale placement now computes `L.T @ R` (equals the fresh compile and the source); Gemini-II stale and fresh contracts both `itranspose`, packed carriers equal. `policy_probe`: the reassociative placement reused under `strict` raises `ReductionOrderError` naming the plan. `host_schedule_probe`: the two-launch host program raises `NotImplementedError` naming the capability on APU v1 (and on Gemini-II in the pytest file). `test_g2_plan_binding.py`: the stale-source test passes (emulated card computes `L.T @ R`); the launch-multiplicity test now stops at the explicit refusal instead of executing one launch for two |
| `tests/pim/test_shared_semantics.py` (24: previous sections plus supplied-placement rebinding on APU v1 and Gemini-II, strict-policy refusal of a reused regrouping placement, unsupported host programs refused on both APUs, an injected wrong coverage rejected by the candidate contract) | 24 passed |
| APU route suites, interface, run-result contract, work grid, legality, matcher ranking, reduction policy, APU v1 recognizers, layout selection, joint selection, host transfer dispatch, host program analysis (`s4-a.log`) | 253 passed in 792 s |
| G6 semantics (all, no board), match coverage, G6 decisions, AttAcc virtual, AiM and UPMEM matchers (`s4-b.log`) | 142 passed, 3 skipped, 2 failed in 993 s; both failures were the first rebinding rule applied to handle-only placements (AiM: the enumerator needs the group-replication stamp the ranking applies; UPMEM: a supplied placement whose decision is current must keep its own derived fields); after the repair the two tests and the shared-semantics suite pass (24 passed) |
| APU v1 typed route after the text-parser removal: scalar surface, recognizers, native reductions, tile selection, run-result contract, distance argmin, cost provenance, shared semantics (`s4-c.log`) | 122 passed in 71 s |
| `tests/pim/test_gemv_all_targets.py` (Samsung and AiM simulators, UPMEM, Gemini-II dataflow rejection, APU v1 board), Gemini-II source route, compile interface at the final tree (`s4-e.log`) | 75 passed in 163 s (the APU v1 board GEMV went through the common candidate pipeline with the rebinding boundary and the effects checks in place) |
| black on the changed files; license headers | clean |

Not run this round: the full `tests/perf` and `tests/pim` suites, the
installed artifact, the Gemini-II card (slice 5).

## Round 15 (2026-11-03): review findings C-29b and C-30; Gemini-II records in the shared pipeline

Same interpreter and `LLVM_BUILD_DIR` as round 12; checkout as working
directory. Raw logs under `/tmp/g9/s6-*.log`, copied to
`/work/shared/users/phd/nz264/tenon-release-evidence/r15/`.

| Check (round 15 tree) | Result |
|---|---|
| the reviewer's round 14 probes (`tenon-release-evidence/reviewer-r14/`): `host_binding_probe.py` rerun unchanged (`host_binding_probe-after.log`) | the swapped launch, the launch without operands (APU v1) and the swapped Gemini-II launch all raise `ValueError` naming the kernel's declared buffers before selection |
| `effects_probe.py`: the reviewer's fault injection constructs the old `CandidateLaunch` shape; re-expressed in the typed contract (`r15/effects_probe_typed.py`, `effects_probe_typed.log`) | both APUs refuse the corrupted candidates before costing ("reads [], the program reads ['A', 'B']; layout of 'Z' uses axis 'nonexistent_axis' ...; storage 'not_a_storage' ...; never transferred ...; launch -7 is out of order ...") |
| `tests/pim/test_shared_semantics.py` (32: previous sections plus launch bindings, six-way fault injection on both APUs, graph consistency, declared-versus-realized effects, Gemini-II typed and composed records through the common pipeline) | 32 passed |
| APU routes, interface, run-result contract, work grid, legality, matcher ranking, reduction policy, recognizers, layout selection, joint selection, host transfer dispatch, host program analysis, shared semantics, Gemini-II typed/composed/payload/layout/value-range tests (`s6-a.log`) | 309 passed in 825 s |
| G6 semantics (all, no board), match coverage, G6 decisions, AttAcc virtual, AiM and UPMEM matchers, CENT AiM workloads, UPMEM dtypes (`s6-b.log`) | 167 passed, 14 skipped, 2 failed in 1130 s; the two failures (`test_upmem_dtypes.py`, int16 operands added into an int32 output, int32 into int64) are rejections the round 11 precision contract (C-26, committed as `c8a513c`) now makes at the shared stage with `UnsupportedWorkloadError` where the tests expected UPMEM's own recognition/ABI errors; both programs are still refused, the expectations were widened to the shared contract and the two parametrized tests pass after the change (13 passed) |
| APU v1 typed routes (scalar surface, native reductions, distance argmin, cost provenance), import surface, GEMV on all targets (Samsung and AiM simulators, UPMEM, the APU v1 board), match patterns, AiM axis roles (`s6-c.log`) | 66 passed in 231 s (the board GEMV through the shared pipeline with the launch-binding and effects checks in place) |
| black on the changed files; license headers | clean |

### Round 15 evidence at the committed revision `9255d29`

The round 15 increment was committed as `9255d29` and pushed fast-forward
to `fork/tenon`; the checks below ran against that commit (checkout clean
apart from the six pre-existing untracked `scripts/verifier_check_*.py`).
Raw logs and scripts in `tenon-release-evidence/r15/`.

| Check (at `9255d29`) | Result |
|---|---|
| perf (`final15-perf.log`, `pytest tests/perf -q -rfxs`) | 533 passed, 1 failed in 596 s. The failure, `test_simulator_unavailable.py::test_gemini_ii_has_no_source_runner`, asserted the pre-round-13 state (`apu_v2` absent from the backend table); the Gemini-II card runner `allo.spmw_apu_g2` has been the shared backend entry since round 13 and the round 13-15 subsets never included this file. The test now asserts the current contract (the shared entry exists; there is still no simulator for the target; no `spmw_apu_v2` module) as `test_gemini_ii_has_no_simulator_runner`; the file passes (8 passed). No code changed |
| PIM non-device, first run at `9255d29` (`final15-pim-attempt1.log`, `pytest tests/pim -q -rfxs -m "not apu_v1_device"`) | 491 passed, 54 skipped, 12 deselected, 1 failed in 3916 s. The failure, `test_paper_golden.py::test_upmem_program_provenance[upmem-linear_reg]`, is a regression from the round 11 precision contract (C-31): the paper's UPMEM linear-regression gradient was refused because the integer narrowing proof had no rule for the right shift of an int64 temporary under the i65 sum once the interval analysis (correctly) stopped bounding the wrapped product. The round 10 wheel compiles the kernel (63,373 virtual cycles); the round 15 tree refused it (`probe_linreg.py`). Fixed with a fallback rule in `_narrow_integer` (an operation the IR evaluates in at most the destination width is a destination-width value) and `test_match_coverage.py::test_shift_of_a_wrapped_temporary_narrows_by_its_own_width`; the kernel compiles to the same 63,373-cycle program again |
| perf and PIM non-device, rerun on the fixed tree (`final15-perf.log`, `final15-pim.log`) | perf: 534 passed in 594 s; PIM: 493 passed, 54 skipped, 12 deselected, 0 failed in 3815 s (1:03:34); the skips are the `paper_full` rows and the Gemini-II card tests, which need the card host |
| sdist and wheel (this server) | sdist from the committed tree (`r15/allo-0.5.tar.gz`, sha256 `4d8dc51a...9881`; 20 `simulators/` entries, no `patches/`, no `verifier_check`, ships `apu_g2_record_frontend.py`); wheel built with the shared LLVM 22 build (`build_wheel15.sh`, `wheel-build15.log`; a first attempt failed only because the script left the environment's cmake 4.4 off `PATH`, `wheel-build15-attempt1-cmake-path.log`), `allo-0.5-cp312-cp312-linux_x86_64.whl` sha256 `9338762d2483b1831207d2a6a2972b64fae4d956a11dc0fc81c31c371aea79d8`; installed into a clean venv (`pkg15/venv2` under `/work/zhang-capra/users/nz264/tenon-release/r15`) |
| installed (`run_installed15.sh`, `installed15.log`; `cd /tmp`, `PYTHONSAFEPATH=1`, no `PYTHONPATH`) | `allo` from the venv only, no module loaded from the checkout; Samsung virtual 27,598; Samsung simulator 26,807 cycles, max abs error 2.2e-3; AiM timing-only 222 cycles, outputs untouched; UPMEM 563,207 cycles, exact; the reviewer probes of rounds 8 to 10 rejected or executed as before (`plain_copy_add` y == t == a + b on the simulator); the round 14 probes through the installed wheel: the swapped and empty launches raise `ValueError` naming the kernel's ABI, the typed fault injection is refused on both APUs |
| APU v1 board from the installed wheel (`run_board15.sh`, `board-installed15.log`, shared `flock`) | `test_vector_gemm_apu_v1[temporal_dma_coalescing]`, `[temporal_dma_coalescing_broadcast_friendly]`, `test_apu_v1_board_median_cycles[apu_v1-histogram]`: 3 passed in 181.26 s (vector plans enumerated, validated against the candidate contract, priced, materialized and launched on the board through the installed package) |
| Gemini-II card from a wheel built on zhang-x4 (`g2_installed_run15.sh`, `g2-installed-run.log`, `r15/g2/`) | host `en-ec-zhang-x4`, card `apu-00` Available, FW `GSI_G2_Infrastructure_V1.24.1`; the same sdist built into a wheel with the host's gcc 11 (sha256 `3e6c7f6f6d12b9e0233b658cec43bac99509f90418edbca2443697548811a87f`), clean venv of the `allo-dev` interpreter (Python 3.12.12), run from `/tmp` under `flock /tmp/tenon-apu-g2.lock`: `tests/pim/test_apu_g2_hardware.py` **15 passed in 145.52 s** (source launch plans and the typed and composed records through the shared pipeline, device builds, launches, readback and the int64 oracle checks through the installed package) |

The wheel, installed, board and card rows above were produced from
`9255d29`, before the C-31 narrowing fix; the fix touches one proof rule
in the matcher that none of those runs exercised (the UPMEM
linear-regression gradient is a simulator-only golden row), and the full
suites were rerun from the checkout with the fix. Still open for slice 5:
stage traces of forced alternatives and cost perturbations reaching
generated code and launches on both hardware hosts at one revision (the
board and card runs above exercise the selected candidates and the
fixed-layout GEMM plans).

## Round 16 (2026-11-03): review findings C-30b; APUv1Program in the shared pipeline

Same interpreter and `LLVM_BUILD_DIR` as round 12; checkout as working
directory. Raw logs under `/tmp/g9/r16-*.log`, copied to
`/work/shared/users/phd/nz264/tenon-release-evidence/r16/`.

| Check (round 16 tree) | Result |
|---|---|
| the reviewer's round 15 reproducers (`tenon-release-evidence/reviewer-r15/test_contract_gaps.py`, run unchanged) | the valid 3x8x5 int4 source still executes exactly on the emulated card; the three accepted-invalid cases now fail at compile: the duplicated partial is refused ("sum [(0, 4), (4, 8), (4, 8)] in order, not an ordered partition of [0, 8)"), the carried accumulator declared zero is refused ("starts its accumulator as 'zero'; the source initializes it as 'carried'"), the one-instance typed add is refused ("cover instances [(0, 1)], not a tiling of [0, 64)") |
| `tests/pim/test_shared_semantics.py` (36: previous sections plus (j) `APUv1Program` records through the common pipeline and (k) the launch chain, accumulator modes and every operation's extent) | 36 passed (within `r16-a.log`) |
| APU v1 program tests through the shared pipeline (`test_apu_v1_native_reductions`, `test_apu_v1_distance_argmin`, `test_apu_v1_recognizers`, `test_apu_v1_scalar_surface`, `test_apu_v1_cost_provenance`, `test_run_result_contract`, `test_compile_interface`), the Gemini-II record and source tests, APU v1 vector compile, tile selection and reduction policy, import surface (`r16-a.log`) | 222 passed in 109 s; the reviewer's file run afterwards: 1 passed (the valid source), 3 failed at compile (the three accepted-invalid contracts) |
| perf (`r16-perf.log`) | 534 passed in 601 s |
| PIM non-device (`r16-pim.log`) | 493 passed, 54 skipped, 12 deselected, 4 failed in 3901 s. The four failures, `test_paper_golden.py::test_apu_v1_paper_kernel_compiles_to_gvml[apu_v1-{kmeans,histogram,linear_regression,word_count}]`, read the native lowering through the old typed callable's `.compiled` attribute path; the lowering is now the selected candidate of the shared pipeline and its introspection is forwarded by the artifact (`compiled.native_vector_lowering`, `.device_source`, `.route`); the test was pointed at the forwarded attributes (no code change) and its five rows pass (5 passed). The virtual estimates of the paper kernels are unchanged (the same retained graphs price the same lowerings) |
| APU v1 board from the checkout (`r16-board.log`, `-m apu_v1_device -rA`: vector GEMM plans, GEMV on all targets, the paper golden board rows including the histogram `APUv1Program` through the shared pipeline, G6 board semantics; per-case printed records kept) | 10 passed, 2 xfailed, 93 deselected in 466 s (`test_vector_gemm_apu_v1` x2, `test_apu_v1_board_median_cycles[apu_v1-histogram]` through the record frontend and the shared pipeline, seven G6 board semantics cases; the two xfails are the documented V1-16 bit-equality expectations); the log keeps every case's printed records |
| Gemini-II card from a wheel built on zhang-x4 from the round 16 sdist (`g2_installed_run16.sh`, `g2-installed-run.log`, `r16/g2/`; `-rA` keeps each case's printed records) | host `en-ec-zhang-x4`, card `apu-00` Available, FW `GSI_G2_Infrastructure_V1.24.1`; sdist of the round 16 tree sha256 `05ae7a97...afa6a` built into wheel sha256 `a1dd070a57b20f21ec01d17f68be24484bad19086fb013edc6a2d486e9a3eeb5` with the host's gcc 11, clean venv of the `allo-dev` interpreter, run from `/tmp` under `flock /tmp/tenon-apu-g2.lock`: `tests/pim/test_apu_g2_hardware.py` **15 passed in 146.08 s** (the typed and composed records through the record frontend with the recomputed-effects realization check, the source launch plans under the ordered-partition and accumulator-chain rules); per-case output kept with `-rA` |
| black on `allo` and `tests`; license headers; pylint on the new and changed modules (`apu_v1_record_frontend.py`, `record_frontend.py`, `spmw_candidate.py`, `apu_g2_record_frontend.py`) | clean; no pylint findings |

## Round 17 (2026-11-03): review finding X-02; APUv1Program through the shared front stages

Same interpreter and `LLVM_BUILD_DIR` as round 12; checkout as working
directory. Raw logs under `/tmp/g9/r17-*.log`, copied to
`/work/shared/users/phd/nz264/tenon-release-evidence/r17/`.

| Check (round 17 tree) | Result |
|---|---|
| the reviewer's round 16 reproducer (`tenon-release-evidence/reviewer-r16/test_record_contract.py`, run unchanged: a guarded write's output must be an input of the contract) | 1 passed: `out` is in the ABI inputs and moved `inout`; both sentinel executions still agree with NumPy |
| the reviewer's probe (`probe_record_frontend.py`, which asserts the shared coverage check and semantic analysis are bypassed) | now fails its own assertion at the first program: both shared stages run on the phase's source |
| retained facts of four small programs (`r17/probe_r17_facts.py`, `probe_r17_facts.log`) | the unconditional whole overwrite reads `src` only; the guarded write keeps the guard `if(gt(src[i], 0))` and its guard read, is not a whole write, and makes `out` an input; the shifted write `out[i+1]` covers 7 of 8 elements and makes `out` an input; `out[i] = out[i] + src[i]` is a carried read; all four declare the runtime's movement (`src` in, `out` inout) and the 8-byte scratch |
| `tests/pim/test_shared_semantics.py` (41: sections a-k plus (l) guarded, shifted, carried and whole writes; hidden reads, allocations and transfers refused) | 41 passed (within `r17-a.log`) |
| APU v1 program tests, Gemini-II record and source tests, match coverage, interface and run-result contracts, paper golden compile rows (`r17-a.log`) | 301 passed, 2 skipped in 1036 s (shared semantics, match coverage, interface and run-result contracts, APU v1 recognizers, native reductions, argmin, scalar surface, cost provenance, Gemini-II typed/composed/source, APU v1 vector compile, tile selection, reduction policy, import surface, UPMEM and AiM matchers, G6 decisions, the APU v1 paper-kernel compile rows); the reviewer's reproducer run afterwards: 1 passed |
| perf (`r17-perf.log`) | 534 passed in 601 s |
| PIM non-device (`r17-pim.log`) | 502 passed, 54 skipped, 12 deselected, 0 failed in 3863 s (1:04:23); the skips are the `paper_full` rows and the Gemini-II card tests, which need the card host |
| APU v1 board from the checkout (`r17-board.log`, `-m apu_v1_device -rA`) | 10 passed, 2 xfailed (V1-16), 93 deselected in 471 s: the two vector GEMM plans, the histogram `APUv1Program` row through the shared front stages and the record candidate, seven G6 board semantics cases; per-case printed records kept |
| Gemini-II card from a wheel built on zhang-x4 from the round 17 sdist (`g2_installed_run17.sh`, `g2-installed-run.log`, `r17/g2/`; `-rA`) | host `en-ec-zhang-x4`, card `apu-00`; sdist of the round 17 tree (sha256 `07141fc8...`) built into wheel `2d6fa3484eabec2b...` with the host's gcc 11, clean venv of the `allo-dev` interpreter, run from `/tmp` under `flock /tmp/tenon-apu-g2.lock`. First run (`g2-installed-run.log`): 14 passed, 1 failed, the typed `div` device build dying with `ld: final link failed: No space left on device`: the host's `/tmp` was full of the build directories the earlier rounds' runs had kept (`TENON_APU_G2_KEEP_TMP=1`, 138 directories, 3.9 GB, all mine). Those were removed and the file rerun with the same wheel and venv without keeping temporaries (`g2_rerun17.sh`, `g2-installed-rerun.log`): **15 passed in 123.04 s**; per-case output kept with `-rA` |
| black on `allo` and `tests` (Black 24.8.0, the lint script's stage; the one unformatted round 16 test file formatted); license headers; pylint on the new and changed modules | black clean on `allo` and `tests` (24.8.0); pylint: no findings in `apu_v1_record_frontend.py`, `record_frontend.py`, `spmw_candidate.py`, `spmw_semantics.py` (the pre-existing P-23 findings of `spmw_match_engine.py` are unchanged); license headers checked at commit |

## Round 18 (2026-11-03): review findings X-03 and C-32; retained effects completed

Same interpreter and `LLVM_BUILD_DIR` as round 12; checkout as working
directory. Raw logs under `/tmp/g9/r18-*.log`, copied to
`/work/shared/users/phd/nz264/tenon-release-evidence/r18/`.

| Check (round 18 tree) | Result |
|---|---|
| the reviewer's `probe_effects.py` (`tenon-release-evidence/reviewer-r17/`, run unchanged; `r18/probe_effects-after.log`) | all five programs compile and execute with the baseline's results: `gather` and `indirect_guard` retain `index` as a read (value and guard), `fill_only` returns eight sevens from a retained fill effect, `initialized_return` retains the fill before the partial store, `unused_argument` moves `src` as a physical transfer with no semantic input |
| the reviewer's `probe_whole.py` (`r18/probe_whole-after.log`) | the whole launch on the non-verbatim Gemini-II add is refused by the shared validator ("declared whole-function, but the program is not executed verbatim") and the full compile fails before pricing |
| `tests/pim/test_shared_semantics.py` (45: sections a-l plus (m) indirect loads, fills, unused arguments, whole launches reserved for verbatim programs) | 45 passed (within `r18-a.log`) |
| targeted batch (`r18-a.log`: the round 17 batch) | 305 passed, 2 skipped in 1044 s |
| perf (`r18-perf.log`) | 534 passed in 644 s |
| PIM non-device (`r18-pim.log`) | 506 passed, 54 skipped, 12 deselected, 0 failed in 4007 s (1:06:47); the skips are the `paper_full` rows and the Gemini-II card tests, which need the card host |
| APU v1 board from the checkout (`r18-board.log`, `-m apu_v1_device -rA`) | 10 passed, 2 xfailed (V1-16), 93 deselected in 477 s: the two vector GEMM plans, the histogram `APUv1Program` row (its facts now hold the zero fill and the guarded carried count), seven G6 board semantics cases; per-case printed records kept |
| Gemini-II card from a wheel built on zhang-x4 from the round 18 sdist (`g2_installed_run18.sh`, `g2-installed-run.log`, `r18/g2/`; `-rA`; temporaries not kept) | host `en-ec-zhang-x4`, card `apu-00`; sdist of the round 18 tree (sha256 `04b8e485...`) built into wheel `000f91ae8d2ca86d...` with the host's gcc 11, clean venv of the `allo-dev` interpreter, run from `/tmp` under `flock /tmp/tenon-apu-g2.lock`, build temporaries not kept: `tests/pim/test_apu_g2_hardware.py` **15 passed in 146.50 s**; per-case output kept with `-rA` |
| black on `allo` and `tests` (24.8.0); license headers; pylint on the changed modules | black clean (24.8.0); pylint: no findings in `apu_v1_record_frontend.py`, `spmw_candidate.py`, `spmw_semantics.py` (the pre-existing P-23 findings of `spmw_match.py` and `spmw_match_engine.py` are unchanged); license headers checked at commit |

## Round 19 (2026-11-03): review finding X-04; Samsung on the shared contract

Same interpreter and `LLVM_BUILD_DIR` as round 12; checkout as working
directory. Targeted runs only this round, as the reviewer directed (no full
suites, no hardware hosts). Raw logs under `/tmp/g9/r19-*.log`, copied to
`/work/shared/users/phd/nz264/tenon-release-evidence/r19/`.

| Check (round 19 tree) | Result |
|---|---|
| the reviewer's `test_effect_order.py` (`tenon-release-evidence/reviewer-r18/`, run unchanged) | 1 passed: the defining store, the seed store and the fill are retained in that order |
| `tests/pim/test_shared_semantics.py` (52: sections a-m plus the ordering regression and (n) Samsung candidates declare effects from the facts, four corrupted declarations refused before costing, the selected GEMV executes on the simulator) | 52 passed (within `r19-a.log`); the Samsung simulator execution ran behind its gate |
| Samsung-sensitive files (`r19-a.log`: shared semantics, the `samsung_hbm_pim` tests, GEMV on all targets except the board, match coverage, work grid, matcher ranking, layout and joint selection, host transfer dispatch, legality, G6 decisions) | 351 passed, 26 skipped, 2 deselected in 6477 s (1:47:56): shared semantics, the `samsung_hbm_pim` directory (conductor runs incl. the elementwise ops and the GEMM residency lever, lowering, owner maps, reduction order, GEMV and readback, layout swizzle), GEMV on all targets except the board, match coverage, work grid, matcher ranking, joint selection, legality, G6 decisions and semantics, paper golden rows, compile interface, run-result contract, layout selection search, Samsung performance, AiM and UPMEM matchers; an earlier run of the same files on the tree before the per-body re-declaration of supplied placements (`r19-a-prefix.log`) had the lockstep-kernel declarations mismatch that the fix removes. A second batch (`r19-b.log`: layout search, Samsung performance, AiM and UPMEM matchers, CENT AiM workloads): 29 passed, 13 skipped |
| black on `allo` and `tests` (24.8.0); license headers; pylint on the changed modules | black clean (24.8.0); pylint: no findings in the new code of `spmw_samsung.py`, `spmw_candidate.py`, `spmw_pipeline.py`, `spmw_semantics.py` (pre-existing P-23 findings unchanged); license headers checked at commit |

## Round 20 (2026-11-03): review findings X-05, X-06, C-33

Same interpreter and `LLVM_BUILD_DIR` as round 12; checkout as working
directory. Targeted runs only this round, as directed. Raw logs under
`/tmp/g9/r20-*.log`, copied to
`/work/shared/users/phd/nz264/tenon-release-evidence/r20/`.

| Check (round 20 tree) | Result |
|---|---|
| the reviewer's `probe_carried.py` (`tenon-release-evidence/reviewer-r19/`, run unchanged; `r20/probe_carried-after.log`) | the 2048x100 carried GEMV declares inputs (matrix, vector, destination) with the destination moved `inout` (scatter before, gather after); the out-only transfer is refused by the validator ("inputs ['destination'] are never transferred in") and by `SamsungCtx.check_realization` ("realization moves ... declared ..."); the simulator run still preserves the nonzero initial output |
| the reviewer's `probe_samsung.py` (`r20/probe_samsung-after.log`) | `flat_add` and `flat_mad` declare every operand in the banks as `all_bank` (the MAD addend `all_bank:swapped_fibers`, matching the program's swap parity `(False, False, True)`), transient registers and scatter/gather transfers; every value in `srf` with route `imaginary_route` is refused before costing |
| `tests/pim/test_shared_semantics.py` (60: sections a-n plus (o) carried destinations on every backend and the refused ingress removal, (p) the elementwise bank images and fiber packing, four valid-looking-but-wrong declarations refused, a realization mismatch rejected in the search with a fallback) | 60 passed (within `r20-a.log` and `r20-b.log`); the Samsung simulator execution ran behind its gate |
| Samsung-sensitive batch (`r20-a.log`, the round 19 file set) | 428 passed, 40 skipped, 2 deselected, 1 failed in 6616 s (1:50:16): the round 19 file set plus the APU v1 vector, reduction-policy and tile-selection files, the Gemini-II source, typed, composed and card-emulation files and the native reductions. The one failure (`test_match_coverage.py::test_local_intermediate_copied_to_an_external_output`) was the elementwise realization check demanding a bank image for the op's *local* result, which the declaration correctly omits; the check now covers the declared outputs and sinks only. Rerun on the final tree (`r20-b.log`: match coverage, shared semantics, the Samsung conductor runs, G6 semantics without the APU rows): 107 passed, 34 deselected in 1609 s |
| black on `allo` and `tests` (24.8.0); license headers; pylint on the changed modules | black clean (24.8.0); pylint: no findings in the new code of `spmw_candidate.py`, `spmw_autoschedule.py`, `spmw_samsung.py`, `spmw_codegen.py`, the APU declarations (pre-existing P-23 findings unchanged); license headers checked at commit |

## Round 21 (2026-11-03): X-06 completion; AiM on the shared contract

Same interpreter and `LLVM_BUILD_DIR` as round 12; checkout as working
directory. Targeted runs only this round, as directed. Raw logs under
`/tmp/g9/r21-*.log`, copied to
`/work/shared/users/phd/nz264/tenon-release-evidence/r21/`.

| Check (round 21 tree) | Result |
|---|---|
| the reviewer's `probe_copyback.py` (`tenon-release-evidence/reviewer-r20/`, run unchanged through `runpy` from the checkout; `r21/probe_copyback-after.log`) | compiles; the 4113-element `temp[i] = a[i]*b[i] + c[i]; c[i] = temp[i]` executes on PIMSimulator and equals the NumPy oracle (1910 cycles, as at the parent) |
| the reviewer's `probe_vector_storage.py` (`r21/probe_vector_storage-after.log`) | the vector-only SRF declaration is refused: every candidate is rejected at materialization ("loads the vector from a GRF_A broadcast (grf_a); the candidate declared ... srf") |
| `tests/pim/test_shared_semantics.py` (68 at `cc2354f`: sections a-p plus (q) distinct images and exact residency, (r) AiM declares effects from the facts, corrupted declarations refused, a realization mismatch rejected in the search with a fallback, the timing-only simulator run) | 68 passed in 564.96 s (`r21-b.log`; this row first said 70, a miscount corrected in round 22 after the reviewer re-ran the file); the Samsung and AiM simulator executions ran behind their gates |
| targeted batch (`r21-a.log`: the round 20 file set plus the AiM files `test_aim_matcher`, `test_aim_axis_roles`, `test_aim_cost_routes`, `test_aim_owner_maps`, `test_aim_program`, `test_aim_token_partition`, `test_aim_virtual_study`, `test_cent_aim_workloads`) | first run (`r21-a.log`): 479 passed, 52 skipped, 2 deselected, 10 failed in 7796 s (2:09:56); the ten failures were all the new AiM realization check on the CENT multi-kernel programs: a group the case's host program never launches lowers no record (now skipped as not realized by that program), and a group launched twice lowers two matching records (now every record is compared with the declaration). Rerun of the AiM files, GEMV on all targets and the shared semantics on the fixed tree (`r21-b.log`): 130 passed, 23 skipped in 1301 s; the CENT workloads and the AiM shared-semantics rows on the final tree (after the last lint edit): 12 passed |
| black on `allo` and `tests` (24.8.0); license headers; pylint on the changed modules | black clean (24.8.0); pylint: no findings in the new code of `spmw_aim.py`, `spmw_candidate.py`, `spmw_samsung.py` (pre-existing P-23 findings, including `AimCtx`'s abstract-method notes, unchanged); license headers checked at commit |


## Round 22 (2026-11-03): scheduled materialization (C-33b); AiM mapping, results and host-transfer binding (X-07, X-08)

Same interpreter and `LLVM_BUILD_DIR` as round 12; checkout as working
directory. Targeted runs only this round, as directed. Raw logs under
`/tmp/g9/r22/`, copied to
`/work/shared/users/phd/nz264/tenon-release-evidence/r22/`.

| Check (round 22 tree) | Result |
|---|---|
| the reviewer's `probe_final_only.py` (`tenon-release-evidence/reviewer-r21/`, unchanged; `r22/probe_final_only.log`) | `FALLBACK`: the winning mode's scheduled realization failure (injected only when the program has a schedule) is rejected and another candidate selected; the kernel-only probes ran as filters (`without_schedule`), the scheduled probes and the final emission under the schedule (`with_schedule`) |
| the reviewer's `probe_contract.py` (`r22/probe_contract.log`) | the `channels` declaration on every candidate selects a candidate that realizes `channels` (reduction partitions 2) instead of one that does not; the renamed, resized, reversed host transfer is refused by `check_realization` ("reads 'A' ('left'); the candidate declares it 'in'") |
| `tests/pim/test_shared_semantics.py` (79: sections a-r plus (s) scheduled realization failure rejected before acceptance; AiM wrong batch mapping and wrong result residency rejected with a fallback; host records carry retained identities; records mutated in direction, extent, identity or unbound are refused; relabelling is not the binding; a host program that never supplies a bank-landing input is refused) with `test_aim_token_partition.py`, `test_aim_program.py`, `test_host_transfer_dispatch.py`, `test_cent_aim_workloads.py` on the fixed tree (`r22/final_tree.log`) | 117 passed, 12 skipped in 742 s; an earlier batch of the AiM files (`r22/aim_more_tests.log`: owner maps, matcher, cost routes, CENT workloads, layout selection) had 3 CENT failures from the first form of the ingress check, which demanded a host write for every declared input: one scatter serves every reader of a buffer and EWADD's GPR operands have no lowered ingress; both are now stated in the check. The AiM sections and the token partition file on the final tree after the last lint-only refactor (`r22/final_aim_sections.log`): 22 passed, 1 skipped |
| ranking-sensitive batch (`r22/ranking_batch.log`: shared semantics, G6 decisions and semantics, GEMV on all targets except the board, match coverage, the UPMEM matcher, work grid) | 206 passed, 2 skipped in 2754 s (45:53); the per-kernel probe is unchanged for programs without a host program and the whole-program probe adds one emission per accepted assignment |
| AiM host-program files before the final edits (`r22/aim_host_tests.log`: token partition, AiM program, host transfer codegen and dispatch, host program analysis) | 53 passed, 1 skipped in 51 s |
| black on the changed files (24.8.0); pylint on the changed modules (`r22/pylint.log` is the first pass; the new code's own findings, a tuple membership test, a `dict` call, a 12-attribute record and a 28-branch function, were removed; the remaining findings in these modules pre-date this round, P-23); license headers | clean |

## Round 23 (2026-11-03): X-09, X-10, X-11 (verified in round 24)

Round 23's actor lost its shell before verification; the edits were
verified at the start of round 24 on the same tree (same interpreter and
`LLVM_BUILD_DIR` as round 12; logs under
`/work/shared/users/phd/nz264/tenon-release-evidence/r23/`).

| Check (round 23 tree) | Result |
|---|---|
| the reviewer's `probe_transfer_contract.py` (`reviewer-r22/`, unchanged; `r23/probe_transfer_contract.log`) | `partial` and `late` are rejected before selection ("launch 0 of 'multiply' ... reads 'A' over [(0, 128)] but the host program has not brought [(64, 128)] to the device"; "reads 'B' ... has not brought [(0, 128)]"); `split` compiles with distinct command streams for the two halves; the row, bank slot, channel and group-id mutations are refused by `check_realization`; the pre-emission row redirection fails selection and the final compile ("addresses 'left' at (row, ...) (37, ...)") |
| the reviewer's `probe_large_slices.py` (`r23/probe_large_slices.log`) | the 524288-element program compiles; the two half scatters emit different commands over rows 0 to 3 |
| `tests/pim/test_shared_semantics.py` with `test_aim_token_partition.py` (`r23/pytest_shared_semantics_r23.log`) | 93 passed, 1 skipped, 1 failed in 660 s; the failure (`test_split_scatters_cover_the_vector_once_at_their_own_addresses`) was the test comparing the halves with the whole program's coalesced `A+B` record; it now compares with A's own bank slot. The "extent" case of `test_aim_host_transfer_records_are_checked_by_identity` was matched to the round 23 slice-frame message before this run |

## Round 24 (2026-11-03): UPMEM and AttAcc on the shared contract (G9); X-12, X-13

Same interpreter and `LLVM_BUILD_DIR`; checkout as working directory.
Targeted runs. Logs under
`/work/shared/users/phd/nz264/tenon-release-evidence/r23/` (the round 24
files are named by their content).

| Check (round 24 tree) | Result |
|---|---|
| every UPMEM family through `allo.compile` with the new deriver (`r23/probe_upmem_families.py`: pointwise, matrix-vector carried, MMTV, GEMM, histogram plain and carried, reduction carried, selection plain and carried, k-means distances, logistic regression, the k-means program with its host-partitioned update) | every program compiles; every selected candidate declares effects that pass the contract and the realization check (the k-means update as a `host_function` whole-function launch) |
| AttAcc GEMV, attention, the renamed GEMV (`r23/probe_attacc_effects.py`) | effects declared and realized as documented; the paper GEMV estimate is unchanged (7040) and the renamed program keeps the same tiling |
| `tests/pim/test_shared_semantics.py` section u (19 tests: UPMEM declarations from the facts, the scaled GEMV's two operations in one launch, the zero-initialized partial as an output and the carried one as `inout`, indirect reads as inputs, the host partition, four corrupted UPMEM declarations refused, three accepted-but-unrealized declarations rejected at materialization with a fallback, a foreign plan refused by the final check; AttAcc declarations, the attention's five phases, three corrupted declarations refused, a wrong tiling rejected with a fallback) with the UPMEM carried/zero-init rows of `test_upmem_generality.py` (`r23/section_u_and_carried.log`) | 86 passed, 1 skipped in 494 s |
| `tests/pim/test_attacc_virtual.py` (without the 500-instance paper selector row), `test_upmem_matcher.py`, `test_upmem_owner_maps.py` (`r23/pytest_attacc_upmem_a.log`) | 53 passed, 2 skipped in 402 s |
| `tests/pim/test_upmem_generality.py`, `test_upmem_dtypes.py`, `test_g6_semantics.py`, `test_g6_decisions.py` on the tree before the last two deriver fixes (`r23/pytest_upmem_b.log`) | 139 passed, 1 skipped, 4 failed in 2127 s: the histogram's launch declared a reduction range of 1 for an accumulate without a reduction loop (now `None`, as the shared extents allow) and the selection family's carried *count* was checked as if it were the output (now `_CARRIED_ROLE`); the non-simulator rows rerun on the fixed tree (`r23/pytest_upmem_b2.log`, before the carried-role fix): 74 passed, 1 failed (the selection row, fixed next); the carried rows and section u after both fixes: see the row above |
| cross-backend batch on the tree before the X-14 fix (`r23/pytest_cross_c.log`: GEMV on all targets, layout selection, reduction policy, the AiM matcher, program, axis-role and CENT files, the Samsung directory) | 96 passed, 11 skipped, 2 failed in 3442 s; both failures were the CENT attention case rejected by the round 23 availability check because the AiM GB vector declared a host broadcast nothing performs (X-14). After the route fix (`r23/pytest_aim_gb_route.log`: the CENT file, the AiM/availability/scatter rows of the shared semantics, `test_aim_program.py`, `test_aim_token_partition.py`): 52 passed, 12 skipped in 211 s |
| `tests/pim/test_shared_semantics.py`, whole file on the round 24 tree before the X-14 route fix (`r23/pytest_shared_semantics_final.log`) | 108 passed in 680 s (sections a-u) |
| the UPMEM simulator rows for the carried histogram, the carried reduction and the selection with a carried count (`r23/pytest_upmem_sim_final.log`, uPIMulator through its gate) | 3 passed |
| black (24.8.0) on the changed files; pylint on `spmw_candidate.py`, `spmw_upmem.py`, `spmw_attacc.py`, `spmw_pipeline.py` (`r23/pylint_r24.log`) | black clean; pylint: the new derivers and realization checks carry the same size notes as the Samsung and AiM ones (too-many-locals/branches/statements, imports inside the function to avoid the import cycle with `spmw_candidate`), recorded under P-23; the two `use-maxsplit-arg` notes were fixed; no other new findings |

## Round 25 (2026-11-04): review findings X-15 to X-18

Same interpreter and `LLVM_BUILD_DIR`; checkout as working directory. Logs
under `/work/shared/users/phd/nz264/tenon-release-evidence/r25/`.

| Check (round 25 tree) | Result |
|---|---|
| the reviewer's `probe_attacc_contract.py` (`reviewer-r24/`, unchanged; `r25/probe_attacc_contract.log`) | stops at its first case with the intended outcome: every swapped-home candidate is rejected at materialization ("keeps 'lW' in 'banks' through ('host_scatter', 'banks'); the candidate declared 'gemv_buffer' ...") and the compile fails with no feasible placement; the probe's later cases are the section v tests |
| `tests/pim/test_shared_semantics.py` section v with the AttAcc, availability, residency and scatter rows (`r25/section_v.log`) | 15 passed |
| regression batch on the round 25 tree (`r25/pytest_r25_batch.log`: `test_attacc_virtual.py` without the 500-instance row, `test_cent_aim_workloads.py`, `test_aim_token_partition.py`, `test_upmem_matcher.py`, the whole `test_shared_semantics.py`) | 170 passed, 14 skipped in 819 s |
| G10 first measurements through `allo.compile` default shared selection (`r25/g10_*.log`, `r25/probe_rope_case_*.log`) | Samsung GEMV rows: default 4389 / 7947 / 8153 cycles, forced vendor layout 4435 / 8041 / 8247 (= paper), outputs within rtol 1e-2 of the fp32 reference; UPMEM: 16 of 16 rows exact (default and `paper_full`), NumPy-exact outputs; AiM: 10 of 14 rows exact (timing-only), the 4 RoPE rows corrected (X-19): qkvo_rope_l128 129637, full_block_l128 421191, full_block_l512 446835, full_block_l4096 697348 cycles against the paper's 129605, 421159, 446803, 697316; AttAcc: 500 of 500 instances select the paper's tiling through `allo.compile` (668,575 candidates, 1912 s); AiM regression after the re-upload modelling (`r25/pytest_aim_reupload.log`): 53 passed, 12 skipped |

## Round 26 (2026-11-04): review findings X-18 (reopened) and X-20

Same interpreter and `LLVM_BUILD_DIR`; checkout as working directory. Logs
under `/work/shared/users/phd/nz264/tenon-release-evidence/r26/`.

| Check (round 26 tree) | Result |
|---|---|
| the reviewer's `probe_slice_identity.py` (`reviewer-r25/`, unchanged; `r26/probe_slice_identity.log`) | `W[0:4]` is `FlatRange(0, 4)`; `W[0:4, 0, 0]` is the box `((0, 4), (0, 1), (0, 1))` with no flat range; the scalar `(1, 0, 0)` has flat offset 4; availability rejects `W[0:4, 0, 0]` naming the missing `[(0, 1), (0, 1), (1, 2)]` and accepts `W[0:4]` and `W[0, :, :]` |
| the reviewer's `probe_attacc_dataflow.py` and `probe_attacc_global.py` (`r26/probe_attacc_*.log`) | the ordinary attention lowers to the 465-cycle inventory; the disconnected softmax is refused ("the exponential reads 'p' and writes 'p', not the score buffer 's' the first MAC writes"); the global normalization is refused ("the softmax sum is not re-initialized per row ... a normalization across rows is not the AttAcc softmax") |
| `tests/pim/test_shared_semantics.py`, whole file on the final tree (`r26/pytest_shared_semantics_r26.log`; sections a-w, 119 tests) | 119 passed in 697 s |
| regression batch (`r26/pytest_r26_batch.log`: `test_attacc_virtual.py` without the 500-instance row, the shared semantics, `test_aim_token_partition.py`, `test_cent_aim_workloads.py`, `test_upmem_matcher.py`) | 173 passed, 14 skipped, 3 failed in 812 s; the three failures were the first form of the section w tests (a wrong whole-row expectation for `W[0:4]`, and a Python variable captured inside a kernel body, which the frontend refuses), corrected before the whole-file run above |
| pylint on `spmw_plan.py` and `attacc_tiling.py` (`r26/pylint_r26.log`) | no new findings beyond two docstrings (added) and the size notes on `_prove_attention_dataflow` recorded under P-23 |

## Round 27 (2026-11-04): review finding X-20 reopened (operand order, loop renaming)

Same interpreter and `LLVM_BUILD_DIR`; checkout as working directory. Logs
under `/work/shared/users/phd/nz264/tenon-release-evidence/r27/`.

| Check (round 27 tree) | Result |
|---|---|
| the reviewer's `probe_attention.py` (`reviewer-r26/`, unchanged; `r27/probe_attention.log`) | ordinary attention 465 cycles; `total / s[h, t]` refused ("the division is not (exponentiated score) / (softmax sum): the operands are 'total' / 's'"); the context-loop renaming and the softmax-row-loop renaming accepted with the same 465 cycles and the same command digest as the ordinary program |
| `tests/pim/test_attacc_virtual.py` (without the 500-instance row) and the whole `tests/pim/test_shared_semantics.py` (sections a-x, 123 tests) (`r27/pytest_r27_batch.log`) | 167 passed in 710 s |

## Round 28 (2026-11-04): the APUv1Program manifest from the retained IR

Same interpreter and `LLVM_BUILD_DIR`; checkout as working directory. Logs
under `/work/shared/users/phd/nz264/tenon-release-evidence/r28/`.

| Check (round 28 tree) | Result |
|---|---|
| manifests of four representative scalar programs (returned local buffer with scratch, nested calls through two callees, in/out argument, unsigned returned local) dumped before and after the rewrite (`r28/dump_manifests.py`, `r28/apu_v1_manifest_kernels.py`, `r28/manifests_{before,after}.json`) | byte-identical JSON (`diff` empty): values, ABI roles, call order, access directions, zero initialization, intermediates, dependencies and barriers unchanged |
| `tests/perf/test_apu_v1_hybrid_manifest.py` (6 tests) | 6 passed |
| APU v1 batch (`r28/pytest_r28_apu_v1.log`: `test_apu_v1_scalar_surface.py`, `test_apu_v1_recognizers.py`, `test_apu_v1_native_reductions.py`, `test_apu_v1_cost_provenance.py`, `test_apu_v1_distance_argmin.py`, `test_compile_interface.py`, `test_run_result_contract.py`, the non-backend sections of `test_shared_semantics.py`) | 171 passed in 380 s |
| `grep` of `allo/pim/apu_v1_hybrid.py` for text analysis (`str(schedule.module)`, `re.search`, the deleted regex names) | none left; the one remaining pattern validates scalar element-type spellings |

## Round 29 (2026-11-04): X-21; installed-artifact hardware sessions on the APU v1 board and the Gemini-II card (G9 slice 5, G10 hardware rows)

Candidate revision `653b059`. Local sdist sha256 `705b7598...16b6` built
from the committed tree; local wheel sha256 `6abf554c...1819` (LLVM 22
clang, clean venv `r29/pkg/venv`); zhang-x4 wheel sha256 `713ba6c6...63af`
built from the same sdist with the host's gcc 11 (clean venv
`r29/g2/venv`). Every run below executed from the installed wheel with the
checkout off `sys.path` (`PYTHONSAFEPATH=1`, only a copy of `benchmarks/`
on `PYTHONPATH`), under the device-host locks
(`/tmp/tenon-apu-v1-0000:41:00.0.lock`, `/tmp/tenon-apu-g2-apu-00.lock`).
Predictions were written before the runs (`r29/apu_v1_prediction.txt`,
`r29/g2_prediction.txt`). Logs and JSON records under
`/work/shared/users/phd/nz264/tenon-release-evidence/r29/`.

| Check | Result |
|---|---|
| `tests/perf/test_apu_v1_hybrid_manifest.py` after X-21 (7 tests, public route with a nonzero initializer and different allocation/result names) | 7 passed; the reviewer's `probe_manifest.py` now reports one buffer `answer` with `zero_initialized=False` and values `x + 7` (`r29/probe_manifest.log`) |
| APU v1 paper rows, default shared selection, 5 launches each, outputs against the archived oracles (`r29/apu_v1-installed-default.log`, `apu_v1_default_rows.json`) | matrix_multiply (plan `temporal_dma_coalescing_broadcast_friendly_acc8`, the paper's forced plan, selected by cost) median 8,757,824 vs paper 8,764,900 (0.9992); kmeans 624,488 vs 624,677 (0.9997); histogram 166,636 vs 166,783 (0.9991); linear_regression 94,328 vs 94,618.5 (0.9969); word_count 174,620 vs 171,988 (**1.0153, open regression X-22**); every launch's outputs equal the oracle |
| APU v1 forced alternative (`apu_v1-installed-alternative.log`) | `layout=temporal_dma_coalescing_broadcast_friendly_acc4`: different device source (sha `68fa795e...`), median 13,534,246 cycles (predicted slower than acc8: confirmed), outputs equal the oracle |
| APU v1 cost perturbation (`apu_v1-installed-perturbed.log`): `MICRO_DMA_L4_L1_32K` x0.01 | cost fingerprint `3345fcb9...` vs stock `0c6e5955...`; the selection moves to `temporal_dma_coalescing` (perturbed estimate 4,849,168 vs acc8 10,682,747); different device source (sha `066bf67b...`); launched: median 359,246,921 cycles, outputs equal the oracle; the stock model's pick is 41x faster (its own estimate for the perturbed pick was 366,292,768, within 2% of the measurement), as predicted |
| Gemini-II paper rows, default shared selection, 5 card sessions (`g2-installed-run.log`, `g2-paper-repeat{2,3,4,5}.log`, `g2_paper_rows_sessions.json`), exact int64 outputs enforced in the runtime and re-checked | medians of ticks per pipeline body (repetitions 20): gemm 1,725 vs paper 1,733.225 (0.9953); batched_gemm 1,736 vs 1,739.15 (0.9982); gesummv 1,806 vs 1,860.075 (0.9709); bicg 2,540 vs 2,548.825 (0.9965); mvt 2,733 vs 2,766.625 (0.9878); spreads at most 43 ticks; one device library `ca39fc23...` for every session |
| Gemini-II card tests from the installed wheel (`g2-installed-run.log`) | `tests/pim/test_apu_g2_hardware.py`: 15 passed in 147 s |
| Gemini-II selection traces (`benchmarks/apu_g2/selection_traces.py`, `g2-selection-traces.log`, `g2_selection_traces.json`) | cost perturbation (rows x20): the 64x128x33 GEMM moves from `single/k2` to `single/k1` (fingerprints `1539818...` vs `0ac887c7...`), distinct layout fingerprints per launch, both exact; stock 2,712 ticks vs perturbed 3,462 (prediction "stock not slower": confirmed). Forced alternative: BiCG `fused/k1` (one launch) 2,520 ticks vs forced `separate/k1` (two launches) 5,074, both exact |

## Round 30 (2026-11-04): measurement drivers that fail loudly; the archived protocols re-run with their baselines; launch provenance; X-22 attribution

Candidate revisions: `69773fc` (drivers, provenance on the scalar route,
CLAIMS/CAPABILITIES text) for the installed-wheel checks and the suites,
`d2e3fda` (provenance on the vector and matcher routes, the Gemini-II paired
driver, the device-lock test on a local filesystem) for the second APU v1
session. Local sdist `0da34450...` and wheel `8de30c98...` at `69773fc`
(LLVM 22 clang, clean venv `r30/pkg/venv`); zhang-x4 wheel `d9261935...`
from the same sdist (gcc 11, clean venv `r30/g2/venv`); local sdist
`f2b4dfdc...` and wheel `1cd4bc6f...` at `d2e3fda` (`r30/pkg2/venv`).
Every hardware run executed from an installed wheel with the checkout off
`sys.path` (`PYTHONSAFEPATH=1`, only a copy of `benchmarks/` on
`PYTHONPATH`), under the device-host locks. Predictions were written before
the runs (`r30/predictions.txt`, with an addendum before the second APU v1
session and the attribution session). Evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r30/`.

| Check | Result |
|---|---|
| Driver failure semantics (`tests/perf/test_measurement_drivers.py`) | 7 passed, 1 skipped (the virtual pass needs `TENON_ARTIFACTS`): missing assets, a failed launch, a wrong output or a violated invariant exit 1 with the record preserved; statistics exclude invalid launches; fractional ticks kept; the Gemini-II vendor lane flags a device library other than the archived one; same-round paired ratios use valid launches of the same round only |
| Installed wheel at `69773fc`, simulators (`installed30.log`, the round 8 probe scripts, no checkout module loaded) | Samsung virtual 27,598 / PIMSimulator 26,807 cycles, max abs err 2.2e-3; AiM timing-only 222 cycles, outputs untouched; UPMEM 563,207 cycles exact; `modules loaded from the checkout: []` |
| `tests/perf` at `69773fc` (`full_perf_69773fc.log`) | 545 passed, 1 failed, 1 skipped in 10:45; the failure is `test_device_lock_is_exclusive_and_reentrant`: `flock(LOCK_EX)` on a read-only descriptor returns EBADF when pytest's tmpdir is on NFS (this session's `TMPDIR`); the same test passes with `--basetemp=/tmp`; `d2e3fda` keeps the test's lock file on a local filesystem (the production lock is under `/tmp`); the skip is the `TENON_ARTIFACTS` virtual pass |
| `tests/pim` at `69773fc` (`full_pim_69773fc.log`, `-m "not apu_v1_device"`) | running at the time of writing (see the round 30 addendum below when complete) |
| Gemini-II paired protocol, run 1 (`g2-paired-rows.log`, `g2_paired_rows.json`) | the Tenon lane completed 20 rounds, all exact (gemm 1,714.05, batched_gemm 1,716.35, gesummv 1,813.25, bicg 2,522.05, mvt 2,743.85 ticks per body); the vendor lane failed every launch: the archived host has its device library location compiled in (`.tenon-apuv2-vector-gtml-stage/.../gtml_vector_baseline_tasks.update.bin`, no longer present) and reported `LoadLibrary failed, status=5`; the session exited 1 and no comparison was reported, as required |
| Gemini-II paired protocol, run 2 (`g2-paired-rows-run2.log`, `g2_paired_rows_run2.json`, exit 0, no problems) after restoring a byte copy of the archived `device.update.bin` (sha256 `c46f7f08...`) at the compiled-in location (documented in `ORIGIN.txt` beside it; the driver now hashes the file at the location the host reports) | one warm-up per lane, 20 interleaved paired rounds, repetitions 20, fractional ticks per pipeline body, one device lock, 210 launches all exact. Tenon medians (q1-q3): gemm 1,722.05 (1,717.0-1,733.4) vs paper 1,733.225; batched_gemm 1,731.55 (1,719.9-1,744.4) vs 1,739.15; gesummv 1,839.30 (1,829.1-1,844.5) vs 1,860.075; bicg 2,530.05 (2,525.9-2,547.5) vs 2,548.825; mvt 2,755.03 (2,750.5-2,770.6) vs 2,766.625. GTML re-run in the same rounds: 1,914.4 / 1,916.1 / 3,099.5 / 2,734.8 / 3,090.6 (paper 1,903.85 / 1,906.4 / 3,090.975 / 2,726.5 / 3,089.65, within 0.6%). Paired GTML/Tenon medians 1.109 / 1.105 / 1.687 / 1.079 / 1.122 (IQR 0.024 / 0.020 / 0.012 / 0.010 / 0.011) vs the paper's 1.098 / 1.096 / 1.662 / 1.070 / 1.117. Predictions confirmed |
| APU v1 paired protocol, session 1 at `69773fc` (`apu_v1-paired.log`, `apu_v1_paired_rows.json`, exit 0, no problems) | one warm-up and ten measured launches per lane and case, Tenon default selection interleaved launch by launch with the archived corrected expert Phoenix binaries (hashes equal to the archive), all 110 launches' outputs equal the oracles. Tenon medians (q1-q3) vs paper: matrix_multiply 8,758,698 (8,757,583-8,759,827) vs 8,764,900 (0.9993); kmeans 625,280 (624,452-625,516) vs 624,677 (1.0010); histogram 165,997 (165,747-166,315) vs 166,783 (0.9953); linear_regression 95,526 (94,829-96,221) vs 94,618.5 (1.0096); word_count 177,178 (175,030-177,284) vs 171,988 (1.0302). Expert medians measured now: 8,786,101 / 641,165 / 167,804 / 95,185 / 175,948.5 (paper 8,784,431 / 641,471 / 167,384 / 95,785 / 175,397.5). Paired expert/Tenon medians: 1.0030 / 1.0261 / 1.0107 / 0.9961 / 0.9922 (paper 1.0022 / 1.0269 / 1.0036 / 1.0123 / 1.0198). Profile counters per launch retained (the matrix multiply's Tenon lane has none at `69773fc`: the vector route recorded no provenance before `d2e3fda`) |
| APU v1 word-count control, same wheel, after session 1 (`apu_v1-control.log`, `apu_v1_word_count_historical.json`, exit 0) | the paper-era generated source (`tenon/word_count/compile/emitted-gvml.c`, sha256 `4e4df52e...`) built by today's project template and launched through the same route: median 172,416 (q1 172,232, q3 172,734, IQR 502) vs paper 171,988 (1.0025), outputs equal the oracle; counters iall 63,062.5 / dcm 79 / icm 96 / seu 13,259 against the current program's 63,145.5 / 80 / 98 / 13,259 in session 1. The two sources differ only in four loop-counter declarations (`uint16_t` then, `uint32_t` since `f3eceeb`) and a static assertion |
| APU v1 paired protocol, session 2 at `d2e3fda` (`apu_v1-paired-s2.log`, `apu_v1_paired_rows_s2.json`, exit 1) | cut short by the host: `ledag-ssh` writes a copy of the device log (about 5 MB) and a session log into `/tmp` on every read, the host's 15 GB `/tmp` filled after about 150 launches of the round, and every later read failed (`ledag-ssh exited with 255`, then `profile freshness check failed ... found 0`). The driver recorded each failure, reported `tenon: 5 valid measured launches of 10` for the matrix multiply (its five valid rounds: median 8,759,424, IQR 608, counters now recorded on the vector route: iall 3,159,328, seu 3,096,646, dcm 33,876) and no comparison for the other four cases, and exited 1. 338 log copies (1.75 GB) and 936 session logs of this user were removed; `28cc922` removes the scratch a read created after capturing the log (`test_ledag_drain_removes_the_tool_scratch_it_created`) |
| APU v1 paired protocol, session 3 at `28cc922` (`apu_v1-paired-s3.log`, `apu_v1_paired_rows_s3.json`, exit 0, no problems; the device-log scratch now removed per read, `/tmp` stable) | Tenon medians (q1-q3) vs paper: matrix_multiply 8,757,878 (8,757,305-8,758,599) vs 8,764,900 (0.9992); kmeans 625,200 (624,624-626,483) vs 624,677 (1.0008); histogram 166,141 (165,630-166,166) vs 166,783 (0.9962); linear_regression 95,807.5 (94,864-96,082) vs 94,618.5 (1.0126); word_count 176,415 (175,696-177,209) vs 171,988 (1.0257). Expert medians 8,785,367 / 641,341 / 167,916 / 95,249 / 175,805; paired expert/Tenon medians 1.0031 / 1.0249 / 1.0111 / 0.9974 / 0.9967 (paper 1.0022 / 1.0269 / 1.0036 / 1.0123 / 1.0198); all 110 launches exact; counters on every lane including the matrix multiply's vector route (iall 3,158,750, seu 3,096,646, dcm 33,876 against the expert's 3,157,077 / 3,096,646 / 33,878) |
| Generated sources against the paper-era sources (`<case>_current_device.c` vs `tenon-artifacts/apuv1/tenon/<case>/compile/emitted-gvml.c`) | matrix_multiply and kmeans byte-identical; linear_regression differs in one static assertion only (`== 32768` became `<= 32768`); histogram differs in the bin loop's counter width and a static assertion; word_count differs in four loop-counter declarations (`uint16_t` to `uint32_t`) and a static assertion |
| X-22 attribution session at `28cc922`, word count, five lanes in the same rounds (`apu_v1-x22.log`, `apu_v1_word_count_x22.json`, exit 0, no problems; one warm-up, ten rounds) | medians (q1-q3): current program 176,041.5 (175,365-176,328), ratio to paper 1.0236; the paper-era source rebuilt today 171,984 (171,842-172,792), 1.0000; the current source with the paper-era 16-bit loop counters 172,178 (171,773-172,856), 1.0011; the archived paper-era binary 172,648 (172,382-172,978), 1.0038; the expert binary 176,167 (175,664-176,502). Same-round ratios over the current program: 0.9794 / 0.9802 / 0.9817 / 1.0014 (IQR at most 0.007). Instruction counts: 63,149 / 63,100 / 63,068.5 / 63,079.5 / 61,758 (expert); seu 13,259 on every lane; dcm 80 / 79 / 79 / 79 / 45; icm 97 / 95 / 96.5 / 98 / 96. Prediction confirmed: the loop-counter width accounts for the whole gap (2.4% of the cycles at equal instruction counts); the board measures the paper-era program at the paper's value today |
| `tests/pim` at `69773fc` (`full_pim_69773fc.log`, `-m "not apu_v1_device"`, 2:04:57) | 564 passed, 20 failed, 54 skipped, 12 deselected. Every failure is the same host condition, not a product result: Docker could not start a container (`failed to start shim: mkdir /var/lib/containerd/...: no space left on device`; the host's `/var` is a 15 GB volume at 100%, `/var/log` 13 GB, root-owned) while the suite ran, so the AiM Ramulator2 rows (`test_aim_cent_paper_cycles[norm_residual_l128, attention_l128, softmax_pim_l128]`, `test_aim_selected_program_runs_timing_only_under_the_declared_contract`), the UPMEM uPIMulator rows (`test_upmem_paper_cycles[va, red, hist]`, 12 `test_upmem_generality` simulator tests, `test_upmem_owner_maps::test_two_ties_execute_with_identical_outputs`) raised the runtime's "no cycles are reported from a failed run" / "DPU build failed (exit 125)" errors as designed. `docker run --rm alpine true` fails the same way at the end of the round; the 20 tests are to be re-run when the host's `/var` has space (no administrator access from this account). The same rows passed from the installed wheel at `69773fc` earlier in the round (`installed30.log`: AiM 222 cycles, UPMEM 563,207 exact) and in rounds 25-29 |
| Nominal-difference control at `28cc922`, kmeans and linear regression, three lanes in the same rounds (`apu_v1-ctrl.log`, `apu_v1_nominal_controls.json`, exit 0, no problems) | kmeans: current program 624,530 (q1 624,428, q3 624,922; 0.9998 of the paper), the archived paper-era Tenon binary 625,038 (1.0006; paper-era/current 1.0007), expert 641,629 (paired expert/current 1.0269 = the paper's 1.0269); the generated source is byte-identical to the paper's. linear regression: current 94,933 (94,767-95,008; 1.0033), the paper-era binary 95,047.5 (1.0045; paper-era/current 1.0040), expert 95,521 (archived 95,785; paired expert/current 1.0047 vs the paper's 1.0123). Predictions confirmed: the paper-era binaries measure within the current programs' dispersion; the board measures the paper's own linear-regression binary 0.45% above the paper's value today. X-23: kmeans REPRODUCED (parity established over four sessions), linear regression OPEN (nominal) |
| X-22 repair check at `dc41df7` (wheel `6897f6d7...` from sdist `77f88dbf...`, clean venv `r30/pkg4/venv`), word count through default shared selection interleaved with the archived paper-era Tenon binary and the expert binary (`apu_v1-repair.log`, `apu_v1_word_count_repaired.json`, exit 0, no problems) | repaired default selection (route `gvml_record_frequency_resident_chunks_pair_count`, generated source sha256 `12fba5a4...`, 16-bit loop counters) median 172,312 (q1 172,090, q3 172,882, IQR 792, MAD 334, min 171,718, max 174,286) vs paper 171,988 (1.0019); the archived paper-era binary in the same rounds 172,372 (172,337-172,528; paper-era/repaired 0.9998, IQR 0.004); expert 175,635 (175,420-176,500), paired expert/repaired 1.0201 (q1 1.0148, q3 1.0247) against the paper's 1.0198; counters iall 63,099.5 / 63,057 / 61,764.5, dcm 79 / 79 / 45, seu 13,259 on all lanes; all 33 launches exact. Prediction confirmed: the repaired program measures as the paper's own program measures today and the paper's advantage over the expert is restored; the session median is 0.19% above the paper's archived median (the paper's own binary 0.22% above it), so under the parity rule the row is OPEN (nominal), not REPRODUCED, until a session's median is at or below 171,988; no tolerance was applied |

Round 30 candidate for review: `dc41df7` plus the docs commit that follows it. The suites ran at `69773fc` (`tests/perf` 545 passed with the one NFS-tmpdir artifact; `tests/pim` 564 passed with 20 Docker-ENOSPC environment failures to re-run); the commits after `69773fc` change the measurement drivers, the launch-record provenance fields, the device-log reader's scratch removal, the device-lock test's tmpdir and the record-frequency loop-counter width (covered by `tests/perf/test_measurement_drivers.py`, 8 passed, and `tests/perf/test_apu_v1_recognizers.py -k record_frequency`, 12 passed, on the final tree).

## Round 31 (2026-11-04): reviewer's driver defects repaired; acceptance rule rewritten; layout rows

Working tree on top of `41a7237`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r31/`.

| Check | Result |
|---|---|
| Reviewer's round 30 driver probes (`reviewer-r30/driver_probes.py`) re-run against the repaired drivers (`r31/driver_probes_rerun.py`, `driver_probes_rerun.log`; mocks, no hardware) | a device result with exact outputs and no launch records is no longer a measurement: the Gemini-II Tenon lane reports `timing_evidence_problem` "0 launch record(s) for a plan of 1 launch(es)" with no amortized ticks, `selection_traces.amortized_ticks` returns `None`, and the complete command exits 1 with `status: fail` and no comparison (`r31/g2_empty_launches.json`); the archived historical output survives a failed launch (`True`) |
| `tests/perf/test_measurement_drivers.py` (12 passed, 1 skipped) | new: the archived lane writes each launch's outputs under a per-session scratch directory outside the archive and keeps a failed launch's partial outputs, `error.txt` and `argv.json` there (the archive is byte-identical after a successful and a failed launch); empty case lists, unknown cases and nonpositive measurement counts or rounds are usage errors; timing evidence needs one record per launch of the selected plan with a finite positive raw tick count (empty, `None`, zero, NaN and mismatched counts are named problems); `selection_traces` invariants reject runs without complete ticks |
| APU v1 driver on the virtual backend with the expert and historical-binary lanes (`r31/apu_v1_virtual_lanes.json`) | both cases describe their lanes and record the scratch directory; every lane's measurement count is now validated and a lane whose binary does not match the archive makes the case invalid |
| Acceptance rule (PAPER_REPRODUCTION.md) | rewritten: all complete sessions at a revision are pooled, the pooled median decides, at most one pre-registered confirmation campaign per nominal row, no favourable-session selection; kmeans reassessed on the pooled 30 samples (median 624,950 vs 624,677, 1.0004) and returned to OPEN (nominal); linear regression pooled 95,180 (1.0059); the repaired word count has one session at `dc41df7` (172,312) |
| Samsung XOR layout row (`tests/pim/samsung_hbm_pim/layout_swizzle`, PIMSimulator, `r31/samsung_layout_swizzle.log`) | 2 passed in 3.3 s: the autoscheduled XOR layout is conflict-free and the identity layout is not; the simulator traces run at exactly 148 (identity) and 116 (synthesized XOR) cycles, the golden rows with zero tolerance, i.e. the paper's 1.28x. This row exercises the layout synthesizer (`select_autoscheduled_xor_layout`) and the simulator trace, not `allo.compile`; the inventory says so |
| Gemini-I vector GEMM layout row (`tests/pim/apu_v1/vector_gemm`, board, `r31/apu_v1_vector_gemm.log`; `r31/vector_gemm_rows.py`, `apu_v1_vector_gemm_rows.json`, 3 interleaved launches per plan under the board lock) | pytest: 2 passed in 163 s (golden rows within 2%). Recorded run: coalesced DMA 91,620,078 cycles [91,620,078, 91,627,191, 91,619,215] vs paper 93,449,834 (0.9804); broadcast-friendly 7,146,206 [7,146,206, 7,145,169, 7,148,065] vs paper 7,349,987 (0.9723; re-captured golden 7,145,109); speedup 12.82x (paper 12.71x); max abs error <= 2.94e-04 against the fp32 product in all six launches; both plans selected by explicit `layout=` as the paper's producer did; the default selection for this shape is `temporal_dma_coalescing_broadcast_friendly_acc2` (not one of the paper's two rows; not measured here) |

## Round 32 (2026-11-04): reviewer's second set of driver defects; the Samsung XOR row withdrawn from G10

| Check | Result |
|---|---|
| Reviewer's round 31 probes (`reviewer-r31/driver_probes.py`) against the repaired drivers (`r32/driver_probes_rerun.log`, mocks) | a two-launch plan returning one launch record: `amortized_ticks_per_pipeline` None, `_timing_problems` names "1 launch record(s) for a plan of 2 launch(es)"; a relative `--scratch` works (outputs written by a child process with the launch directory as cwd) |
| `tests/perf/test_measurement_drivers.py` (15 passed, 1 skipped) | new: `selection_traces.launch_evidence_problem` requires one record per launch of the selected plan, in order (`index`), with the plan's layout fingerprint and a finite positive tick count; `_run` records `expected_launches` and `timing_evidence_problem`; a command-level regression with exact outputs and a missing stock launch exits 1 and retains the one record; the archived lane resolves the scratch directory to an absolute path, passes absolute output arguments, and a relative scratch with a successful and a failed launch leaves the archive byte-identical |
| Samsung XOR layout row | the round 31 `REPRODUCED` status is withdrawn (inventory, REVIEW, CLAIMS E3): the probe reaches the autoscheduler through a fabricated `MatchedOp` and a separate trace harness; its 148/116 cycles are kept as diagnostic evidence; the shared-pipeline route is the open implementation item of this round |
| Why the Samsung row has no shared-pipeline realization (conductor `TenonConductor.inc` `address(...)`, `stage_matrix`, `run_pass`) | the production REDUCE program stages every reduction tile at DRAM row 0 of its bank fiber and separates tiles by column; the candidates' only layout lever is the fiber of each tile (`tile_fiber`), so the row conflicts the paper's micro-experiment de-conflicts cannot arise in any ordinary program today (design: `design/samsung-bank-row-swizzle-route.md`, S-13) |
| Fiber-swizzle A/B on the real PIMSimulator through `allo.compile` with `layout=` (`r32/samsung_fiber_ab.py`, `samsung_fiber_ab_m8192_k256.json`, `samsung_fiber_ab.log`, exit 0): GEMV 8192x256 fp16, reassociate | vendor-equivalent `dual_fiber` (tiles alternate the even and odd fiber by the enumerator's F2 swizzle, host GRF_A fill, shared CRF) 8,247 cycles = the paper's vendor row; `bank_row` with the same levers (every tile in the even fiber) 8,153; default selection (`bank_row+crf_shared`) 8,153; all three outputs within 1.42e-4 of the fp32 reference. The fiber swizzle does not speed up the production program (nothing to de-conflict at row 0); this is the shared-route evidence available for the mechanism, not a reproduction of the paper's 148/116 |

## Round 33 (2026-11-04): the row-0 diagnosis corrected with the conductor's address trace; the program-derived access stream

Working tree on top of `e357628`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r33/`. The PIMSimulator
driver was rebuilt from the overlay with an address-trace switch
(`TenonConductor.inc`, `TENON_CONDUCTOR_ADDRESS_TRACE=<path>`; `pim_driver`
sha256 `4e4c82ad...`, the previous binary kept as
`pim_driver.bak-20261104-r33`); the paper GEMVs' cycles are unchanged.

| Check | Result |
|---|---|
| Reviewer's provenance probe (`reviewer-r32/provenance_probe.py`) and `tests/perf/test_measurement_drivers.py` (18 passed with `TENON_ARTIFACTS`) | `launch_evidence_problem` now requires a launch index on every record and the plan's layout fingerprint whenever the plan declares one (missing, null and swapped fields are named problems); `benchmarks/apu_g2/paper_rows.py` applies the same plan correspondence after the tick check |
| Address model vs the conductor (`r33/reduce_address_model.py`, `trace_gemv_m8192_k256_{dual_fiber_vendor,bank_row_same_levers}.txt`, `samsung_fiber_ab_trace.log`) | the conductor's dump of the 8192x256 GEMV, channel 0: per pass 16 auto-MAC groups of 8 consecutive columns, each closed by a barrier, plus a host GRF_A fill per tile and one 8-column drain to bank 1; `bank_row` reads bank 0 at rows 0-3 (pass 0) and 4-7 (pass 1), seven same-bank row changes over the two passes; `dual_fiber` reads banks 0 and 1 at rows 0-3, three row changes in each; the matrix staging writes rows 0-7 (bank_row) or 0-3 (dual_fiber) of every bank. The round 32 statement that every tile stays at row 0 was wrong (`addrGenSafe` carries columns into rows, 32 burst columns per row, 64 columns per tile), as the reviewer said; the model's group order, banks, rows and columns equal the dump. Cycles unchanged by the rebuild: dual_fiber 8,247, bank_row 8,153 |
| `samsung_access_stream` (new, `allo/spmw_samsung.py`): the executable pass-phase stream derived from the lowered `SamsungProgram` and the geometry; `tests/pim/samsung_hbm_pim/test_access_stream.py` (3 passed in 346 s: one unit test, two conductor-gated) | derived stream byte-identical to the conductor's dump (`PASS` to the readback) for both placements of the 8192x256 GEMV (two tiles) and of the 4096x512 GEMV (four tiles, `tile_fiber` 0,1,0,1 against 0,0,0,0), with exact simulator outputs; the unit test pins 32 MAC rounds of 8 columns, 4 host fills, the drains, the rows per bank and the row-change counts for both placements without the simulator (`r33/access_stream_{m8192_k256,m4096_k512}.json`, `access_stream_*.log`) |
| Harness replay of the derived MAC+FILL read rounds (`scripts/run_pimsim_layout_trace.py`; the harness `pimsim_layout_trace.cc` now advances the memory system while a long stream is queued, bounded retries; the original probe still passes, 2 passed) | 8192x256: dual_fiber 1,188 cycles, bank_row 1,248 (32 rounds, 256 requests each); 4096x512: dual_fiber 1,228, bank_row 1,248. The replay is plain reads with a barrier per group (no PIM mode or execution, host fills, drains, staging, CRF programming or mode changes): a diagnostic of the address sequence under the DRAM timing, not a measurement of the conductor's bank-read phase and not an attribution of the full-program difference; the full programs through `allo.compile` run 8,247 vs 8,153 and 8,041 vs 7,947 (`samsung_fiber_ab_trace_m4096_k512.json`); the default selection picks `bank_row+crf_shared` on both; the summary's "row transitions" are address changes between MAC groups in the modelled stream, not measured activations (wording narrowed in round 34) |
| What this does and does not establish for the paper row | every barrier group of the production stream is eight consecutive columns of one row of one bank, so the within-round two-row conflict of the paper's four-request probe never occurs in the program; the probe's request multiplicity, ordering and timing boundary cannot be obtained by compressing the GEMV stream; the row stays UNVERIFIED and the route (a REDUCE variant with two tiles' rows in one group, `matrix_rows` with `tile_row` through program and conductor plan, GRF_B row-parity indexing preserved, costs from device timing facts) is in `design/samsung-bank-row-swizzle-route.md` |
| CLAIMS E4 | corrected: 10 AiM rows exact through default selection, the 4 RoPE rows OPEN 32-cycle regressions (X-19) with the paper's expectations preserved |

## Round 34 (2026-11-04): the stream's column default and geometry handling; the instrumentation in the release overlay; the portability row

Working tree on top of `d4c6703`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r34/`.

| Check | Result |
|---|---|
| Reviewer's access probe (`reviewer-r33/access_probe.py`) re-run (`r34/access_probe_rerun.log`) | a two-column FP16 GEMM (4096x128 by 128x2) now yields 20 rounds covering columns 0 and 1 by default (the count comes from the retained `contraction.N`); a failing geometry query raises `RuntimeError: installed driver has broken describe` instead of yielding a stream |
| `tests/pim/samsung_hbm_pim/test_access_stream.py` (unit tests, 2 passed in 154 s without the simulator; the gated conductor-dump tests in `r34/access_stream_tests.log`) | `samsung_access_stream` returns a `SamsungAccessStream` carrying the column count, the geometry and its source (`installed pim_driver --describe` or the explicitly requested `SAMSUNG_DEFAULT_GEOMETRY` offline model); `columns` other than the program's count or non-positive are errors; a geometry without `channels`/`cols_per_row` is an error; the summary labels the geometry source and its "row transitions" as modelled address changes. The gated test fails (does not skip) with rebuild instructions when the installed `pim_driver` writes no trace |
| Release overlay `simulators/PIMSimulator/src/tests/TenonConductor.inc` | now carries the Allo-authored `TENON_CONDUCTOR_ADDRESS_TRACE` instrumentation (identical to the research tree's overlay copy); the installed driver rebuilt from the release copy is byte-identical to the round 33 build (sha256 `4e4c82ad...`) and describes the same geometry; the Samsung-derived patches stay outside the release (P-04), as the overlay README states |
| Wording | the design, REVIEW round 33 and VALIDATION round 33 no longer present the harness replay as the conductor's bank-read phase or as an attribution of the full-program difference, and the stream summary's transitions as measured activations |
| GEMV portability row, first pass (`r34/portability_gemv.py`, `portability_gemv.json`, `portability_gemv.log`, exit 1): the same plain 4096x256 FP16 GEMV source (`y[i] += W[i,k]*x[k]`) with Gaussian 0.1-scaled FP16 inputs, strict (default) order, on PIMSimulator and the local board; reference: every multiply and add rounded to FP16 in source order | Samsung strict: `GENERIC_TEMPORAL_REDUCE`, 26,807 cycles, 4,096 of 4,096 outputs bit-exact (the plain source compiles on Samsung too; the reassociating compile of this source also selected the ordered program and was bit-exact). Gemini-I strict: `temporal_dma_coalescing`, 11,423,897 cycles, 1,900 of 4,096 bit-exact, max abs error 2.1e-3 against FP32; `temporal_svp` gives the same 1,900 (`apu_v1_scalar_probe.log`); the scalar `grouped_f16` candidate is not enumerated for this source. Offline hypothesis tests on the saved device output (`apu_v1_strict_probe.log`, `strict_4096x256.npz`): no IEEE rounding mode of product or sum (nearest-even, away, truncation), no fused step, no FP32 accumulation, no partial-sum grouping and no flush-to-zero matches better than source-order RNE (1,900), consistent with V1-16 (the device's FP16 multiply is not an exact-product rounding; `test_g6_board_semantics` bit-model test xfails for the same reason, 2 passed 2 xfailed this round). The paper's audit (`reviews/experiments/strict_fp16_gemv_audit.py`, CLAIMS E11) used inputs whose products are exact (powers of two times ±1 against a vector of ones, with cancellation patterns), where only the addition order matters; see the second pass below |
| GEMV portability row, second pass: the paper's experiment through `allo.compile` (`r34/portability_paper_inputs.py`, `portability_paper_inputs.json`, `.log`, exit 0): the paper's shared GEMV source (`out[row,column] += A[row,depth]*B[depth,column]`, 4096x256x1) with its rounding-sensitive inputs (signed powers of two 2^-8..2^8, B = ones, the six cancellation rows, seed 2026092910; products exact, so the accumulation order is what is tested), strict (default) reduction order, PIMSimulator and the local board | Samsung strict `GENERIC_TEMPORAL_REDUCE` 26,807 cycles: 4,096 of 4,096 outputs bit-exact against the source-order FP16 reference, cancellation rows exact; Gemini-I strict `temporal_dma_coalescing` 11,420,120 cycles: 4,096 of 4,096 bit-exact, cancellation rows exact. The paper's second claim too: the reassociating Gemini-I plan (`spatial_gemv_group_reduction_k256`, 1,245,149 cycles) matches only 859 of 4,096 and misses the cancellation rows; the source-order reference differs from NumPy's grouped FP16 sum on the rows the inputs were built to expose. Row REPRODUCED: the paper's claim is about accumulation order under strict policy, which both targets honour through the shared pipeline; bit-exactness of arbitrary FP16 products on Gemini-I is not claimed by the paper and is not reproducible (V1-16, first pass above) |

## Round 35 (2026-11-04): the portability experiment as a maintained in-repo driver

Working tree on top of `790e7a9`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r35/`.

| Check | Result |
|---|---|
| `benchmarks/portability/gemv_source_order.py` (new) and `tests/perf/test_portability_driver.py` (4 passed) | the paper's shared GEMV source, its historical inputs (signed powers of two against ones, six cancellation rows, seed 2026092910; `products_are_exact` True), an independent source-order FP16 oracle (differs from NumPy's grouped FP16 sum on 3,272 rows), execution-kind assertions (Samsung `simulator`, Gemini-I `device`; a virtual run is a dry run and never a reproduction), and evidence retention under `--out`; the tests cover the inputs, the oracle, a wrong output and a dry run failing loudly with evidence kept, and a failed run recorded before exit |
| Dry run on the virtual backend (`r35/dryrun/`) | exits 0 with status "dry run (virtual backend): not a reproduction" and both targets flagged as not executed where the paper executed |
| `python -m benchmarks.portability.gemv_source_order --out r35/portability --reassociate` from the checkout (`r35/portability.log`, `report.json`; PIMSimulator `pim_driver` sha256 `4e4c82ad...`, local board, NumPy 2.3.5) | Samsung strict `GENERIC_TEMPORAL_REDUCE` (`temporal+crf_per_workid`), 26,807 cycles, kind `simulator`, 4,096 of 4,096 bit-exact, cancellation rows exact; Gemini-I strict `temporal_dma_coalescing`, 11,422,897 cycles, kind `device`, 4,096 of 4,096 bit-exact, cancellation rows exact (binary sha256 `ca85f59b...`, device source `4ad4ae20...`, profile counters and text retained); the regrouping Gemini-I plan `spatial_gemv_group_reduction_k256` (1,245,645 cycles) matches 859 of 4,096; the reassociating Samsung compile again selects the ordered program. Retained per run: output `.npy`, retained MLIR, device source, `RunResult.extra` (launch roles, raw outputs, CRF words, conductor receipt, cycle phases), the environment (`evidence_class: checkout`, allo revision 790e7a9). Status: pass, exit 0. This is checkout evidence; the installed-artifact run is pending |
| Docstring residue (`samsung_access_stream_summary`) | describes modelled address changes, not row activations |

## Round 36 (2026-11-04): the portability driver's validation and provenance; the tile-row levers through the shared pipeline

Working tree on top of `e6f54f0`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r36/`. The PIMSimulator
driver was rebuilt twice from the overlay (`TenonConductor.inc/.h`: plan
fields `issue_order`, `split_rows`; final sha256 `82db787a...`, previous
binaries kept as `pim_driver.bak-20261104-r36`); the release overlay copy is
identical to the research tree's.

| Check | Result |
|---|---|
| Portability driver (`tests/perf/test_portability_driver.py`, 6 passed) | a requested reassociation control that fails or executes elsewhere than the paper executed fails the command (an executing control may differ from the oracle); provenance is serialized structurally (Mapping views, dataclasses, manifests; depth 64 with a cycle guard; nothing truncated) and the record keeps the selected plan's manifest and the shared candidate-effects manifest; a real APU v1 plan's metadata (operation counts, transfers, accumulator persistence, accesses) comes through whole (`r36/dryrun_apu_v1/report.json`, 203 kB). Samsung re-run from the repaired driver (`r36/portability_samsung/`): 4,096 of 4,096 bit-exact, 26,807 cycles, pass |
| Tile-row levers, simulator-free tests (`tests/pim/samsung_hbm_pim/test_row_pairing.py`, 3 passed) | `bank_row_rowpair`, `split_rows` and `split_rows_rowpair` are enumerated with the shared contract and the `half -> bank` F2 map (conflict-free for the split families); the lowering emits the vendor CRF for the identity pairing and the two-MAC loop body for the split layout, the plan carries `issue_order`/`split_rows`, the access-stream model gives 16 rounds of 16 for the pairing and alternating even/odd rounds for the split; the paired split and a CRF-filled split are refused with their reasons; the cost model ranks the identity pairing last |
| Tile-row levers on the conductor (`r36/run2/rowpair_gemv_m8192_k256.json`, `rowpair_gemv_m4096_k512.json`, `rowpair_measure_run2.log`, both exit 0; gated tests in `r36/row_pairing_tests.log`) | through `allo.compile` with `layout=`: 8192x256: bank_row 8,153, dual_fiber 8,247, bank_row_rowpair 53,634 (exec 47,697), split_rows 7,255 (preload 4,595, exec 2,351); 4096x512: 7,947 / 8,041 / 53,506 / 7,057; every derived stream byte-identical to the conductor's dump; `bank_row_rowpair` and `split_rows` bit identical to `bank_row`; `split_rows_rowpair` refused at lowering (`InfeasibleSchedule`, the lockstep-CRF reason). The first measurement run (`r36/rowpair_measure.log`, before the refusal) executed the paired split and got wrong results (max error 0.139 and 0.175 against 1.4e-4 and 2.2e-4), the evidence behind the refusal |
| Interpretation (design doc, round 36 section) | two rows of one bank inside one barrier group cost about 2,845 cycles per pair on both shapes (the measured constant now prices `issue_order=row_pair`); the XOR row packing with the vendor's barriers is 1.124x / 1.126x faster than `bank_row` on the whole program, with the gain in staging; the cost model still prefers `bank_row` (S-16); the paper's within-round conflict on plain reads is unreachable by the production program, so the inventory row is OPEN (measured), not REPRODUCED |
| Harness replay of the derived pass-phase reads of the four legal placements (`r36/harness_replay_run2.log`, plain reads with a barrier per round, a diagnostic of the address sequence) | 8192x256: bank_row 1,248, dual_fiber 1,188, bank_row_rowpair 45,298 (16 rounds of 16), split_rows 1,276; 4096x512: 1,248 / 1,228 / 45,298 / 1,276. The two-rows-of-one-bank barrier group costs about 2,750 cycles per round on plain reads as well, so the identity pairing's penalty is the DRAM controller's handling of that group, not PIM execution; the split layout's reads are not faster than bank_row's on plain reads (its whole-program gain is in staging, S-16) |


## Round 37 (2026-11-04): the selected map as the contract; the staging term; default selection earns the swizzled rows

Working tree on top of `4378bf7`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r37/`. The conductor
overlay now reads `half_fiber a b` from the plan (the rendering of the
selected half -> bank map) and derives `split_rows` from it; `pim_driver`
sha256 `6c50596b...` (previous kept as `pim_driver.bak-20261104-r36`); the
release overlay copy is identical to the research tree's.

| Check | Result |
|---|---|
| Reviewer's contract probe (`reviewer-r36/contract_probe.py`) re-run (`r37/contract_probe_rerun.log`) | changing the `split_rows` candidate's `half_bank_map` to the identity map changes the search decision (`unchanged: False`); the edited map with a bogus matrix packing is rejected before selection: "violates the candidate contract: declares 'left' in storage 'banks' packing 'bogus-unrealized-packing'; the derivation from the placement's selected map and handles gives 'banks' packing 'half_bank:0,0'" |
| The map as the contract (`allo/spmw_samsung.py`, `spmw_candidate.py`, `spmw_autoschedule.py`; `tests/pim/samsung_hbm_pim/test_row_pairing.py::test_the_selected_map_is_the_contract_not_a_label`) | `samsung_half_fibers` reads the fiber of each tile row from the selected `LinearLayout`; the matrix value layout declares `packing=half_bank:h0,h1` from it; the lowering derives the row placement from the map and rejects a disagreeing `tile_rows` label; the plan carries `half_fiber`; `_check_lever_realization` compares the program's row fibers with the declared packing; `half_bank_map` is a physical decision field; `validate_candidate_effects` compares every declared value layout's storage and packing and every transfer route with the target's derivation (`_derivation_problems`), `_redeclared` no longer overwrites a differing declaration, and the rebinding of a supplied placement to the current candidate with the same decision (`rebind_supplied_layouts` / `_with_current_effects`) refuses a supplied declaration that disagrees with the current candidate's on storage, packing or transfer routes instead of replacing it (found by the regression's third step: with the map intact and only the packing edited, the old rebinding silently substituted the fresh declaration) |
| Cost model (`allo/pim/costs/samsung.py`) | `split_fibers` priced for any issue order (the odd-fiber term per tile), the row-pair constant's validity domain stated; the first matrix staging added to the selection graph (`mac_stage`, scope changed and documented) from `samsung_staging_schedule`, the conductor's channel-0 staging order, priced from the device ini (a write every tCCDS = 2, an exposed same-bank row change tWR + tRP + tRCDWR - tCCDL = 36); no constant fitted. The schedule equals the conductor's staging dump (first staging, channel 0) for every placement on all three workloads (`r37/staging2/staging_and_selection.json`, `schedule_equals_dump_first_staging` True) |
| Default selection and measured cycles through `allo.compile` on the conductor (`r37/staging2/`, `staging_and_selection2.log`; held-out streams `heldout_stream_check.log`, restaging excluded) | 8192x256: default `split_rows+crf_shared`; measured bank_row 8,153 (preload 5,628), dual_fiber 8,247, split_rows 7,255 (preload 4,595), bank_row_rowpair 53,634; predicted staging on the 64-channel system 5,248 vs 4,096 (measured preload gap 1,033, modelled 1,152). 4096x512: default `split_rows+crf_shared`; measured 7,947 / 8,041 / 7,057 / 53,506. Held-out 37x129x2 (the reviewer's contraction): default `split_rows+crf_per_workid+wresident`; measured bank_row 8,724, dual_fiber 8,817, split_rows 8,163, bank_row_rowpair 49,734; the default is the measured fastest on all three; every output within 2.3e-4 of FP32; every derived stream equals the dump (the held-out program restages per column; those lines are outside the pass-phase stream). Estimates are on the declared 16-channel device (absolute values not comparable to the conductor, module docstring) |
| Tests | `tests/perf/test_samsung_performance.py`: the graph test expects the `mac_stage` step and the shared-cost-program test expects the `split_rows` default (both updated for the scope change); `tests/pim/samsung_hbm_pim/test_row_pairing.py`: cost ordering split_rows < bank_row < dual_fiber < bank_row_rowpair and the default `split_rows`; final runs in `r37/final_tests.log` and `r37/perf_samsung_tests2.log` |


## Round 38 (2026-11-04): the staging accounting partitioned per work item; declarations compared on their axes

Working tree on top of `c29287d`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r38/`.

| Check | Result |
|---|---|
| Reviewer's cost probe (`reviewer-r37/cost_probe.py`) re-run (`r38/cost_probe_rerun.log`) | the staging step now precedes the MAC (`mac_stage` then `mac`, the MAC depending on it); work items 64x256, 32x512 and 128x128 (1,024 padded bursts each) all cost 2,624; 8x128 costs 208 (its bank's row-occupancy floor) and 1024x128 costs 20,992 (eight times 128x128) |
| `tests/perf/test_samsung_performance.py` (5 passed): `test_staging_shares_partition_the_conductors_staging`, the graph test's order | the shares of the 8192x256 GEMV's 128 work items sum to 64 channels x 2,048 writes and x 32 exposed row changes (the conductor's staging and the whole-pass schedule) for the vendor rows, and to no exposed change for the split rows; the split share is cheaper; the ordered program carries the same scope from its own preload pattern |
| Default selection with the repaired accounting (`r38/selection_recheck.json`, virtual) | `split_rows` on gemv 8192x256 (35,014 against bank_row 43,094, dual_fiber 43,478, bank_row_rowpair 72,088, temporal 86,906), gemv 4096x512 (37,166 against 44,118 / 44,878 / 87,875 / 89,582), the held-out 37x129x2 (2,181; split_rows with the vendor levers 2,514 against bank_row 2,912) and the reviewer's 19x65x3 (903; 1,093 against 1,261); the measured cycles of round 37 and of the reviewer's neighbor run (`r38/neighbor_rerun.log`) confirm the fastest placement on every measured shape |
| Reviewer's declaration probe (`reviewer-r37/probe.py`) re-run (`r38/probe_rerun.log`) | the matrix layout with its axes emptied is reported by `validate_candidate_effects` ("declares 'a' laid out over axes (); the derivation ... gives ('r', 'd')") and refused by `allo.compile`; the map with a corrupted bank basis is refused by `samsung_half_fibers` ("must keep the bank input as the tile's base fiber") |
| `tests/pim/samsung_hbm_pim/test_row_pairing.py::test_supplied_declarations_must_match_axes_and_a_valid_map` and the other simulator-free suites (`r38/unit_tests.log`); ranking perf suites (`r38/perf_tests.log`) | 31 passed, 4 conductor-gated deselected (`test_row_pairing.py`, `test_access_stream.py`, `test_portability_driver.py`, `test_measurement_drivers.py`); 62 passed (`test_matcher_ranked_placement.py`, `test_joint_selection.py`, `test_legality.py`, `test_compile_interface.py`); the reviewer's neighbor 19x65x3 run on the conductor selects `split_rows+crf_per_workid+wresident` by default and is exact at 3,822 cycles (`r38/neighbor_rerun.log`) |


## Round 39 (2026-11-04): the supplied-declaration boundary completed; the token-partitioning study on the simulator

Working tree on top of `0a4b543`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r39/`.

| Check | Result |
|---|---|
| Reviewer's contract probe (`reviewer-r38/contract_probe.py`) re-run (`r39/contract_probe_rerun.log`) | all five mutations of the 23x81x2 contraction's `split_rows` declaration are refused at direct validation and by `allo.compile(..., layout=...)`: the image (`declares no layout for 'left'`), the buffer identity (`no value layout for ['left']`), the launch domain (`cover instances [(0, 1)], not a tiling of [0, 46)` and the launch difference), the allocation set (`declares allocations [('unrealized', ...)]; the derivation gives ...`) and the axes |
| `tests/pim/samsung_hbm_pim/test_supplied_declarations.py` (7 passed) | the five mutations at both boundaries with the validator's own problem in the refusal and the unedited candidate compiling; a bare decision (no declaration) and another program's plan (coverage of `another_program`, foreign launch domain) supply their decision, the declaration re-derived and the rebinding recorded; one body's declaration supplied for all 16 work items of a lockstep kernel (`mapping=[16]`) compiles with each body's own declaration, and an edited one is refused for every body |
| `tests/pim/test_shared_semantics.py -k "supplied_placement or fault_injection or corrupted or validated_against or rebound or explicit_layout or layout"` (30 passed, `r39/shared_semantics_boundary.log`) | the cross-program rebinding (`L @ R` placement supplied to `L.T @ R` on APU v1 and Gemini-II) and the fault-injection suites of every backend pass with the full-record derivation comparison |
| `tests/pim/test_aim_token_partition.py -m "not paper_full"` (5 passed) | the compile-only tests pass against the study moved into `benchmarks/cent_aim/token_partition.py` |
| AiM simulator, native executor (`TENON_AIM_RAMULATOR2`; `r39/aim_native_golden.log`, `r39/aim_native_golden_full.log`) | Docker on this host is out of space (`/docker` 300 GB and `/var` 15 GB full; the ENOSPC that blocks the Docker-run tests since round 25). `ramulator2` built natively from the same simulator tree (upstream 0f28a07, conda gcc 12.4, out of tree, dependency sources reused offline; `docs/release/simulators.md`): the smoke trace gives the documented 689 cycles; `test_gemv_all_targets.py -k aim` and the default CENT rows pass (5 passed, the AiM GEMV 222 cycles); the 11 `paper_full` AiM rows pass exactly as well (`-m paper_full -k aim`: 11 passed in 247 s), including the four X-19 corrected programs (e.g. `qkvo_rope_l128` 129,637) |
| Token-partitioning study (`python -m benchmarks.cent_aim.token_partition --out r39/token_partition_nine.json`, `r39/token_partition_nine.log`) | pass (nine workloads, 45 compared plans, every run `simulator-timing-only` on the native binary `f5bf8671e3d0`): the cost selection reaches 6 of 9 simulated minima with a largest gap of 1.34% (the paper: 7 of 9, 2.63%); the misses are the three 1-head workloads (selected P2 vs best P4 at 0.34%, P4 vs P8 at 0.49%, P1 vs P8 at 1.34%); every 8- and 32-head workload selects and measures P1 best; the merge is exact on integers and within 3.7e-3 on BF16 on every split plan; for 1 head x 4096 tokens head-only / selected is 2.17x (12,003 / 5,535) against the paper's 1.071x, the in-budget P1 plan being the flattened output mapping over eight channels rather than the paper's head-only plan; the unrestricted default selection on the simulated 32-channel device (32 replicas on most workloads) is recorded and is not the metric; inventory row OPEN (measured) |
| `tests/perf` whole (`r39/perf_all.log`) | 567 passed (16 min); `tests/perf/test_token_partition_driver.py` 3 passed |
| AiM suites with the native executor (`r39/aim_suites_native.log`: `test_aim_*.py`, `test_cent_aim_workloads.py`, `test_gemv_all_targets.py`) | 54 passed, 11 skipped; the one failure is `test_gemv_upmem` (uPIMulator's Docker SDK image, the same ENOSPC) |
| Samsung simulator-free suites (`r39/samsung_unit.log`, 31 tests) | 26 passed, 5 failed, all five stale expectations predating this round and not run in rounds 37-38: `test_default_selection_follows_the_serialized_fiber_cost` asserted the pre-round-37 default (every tile on the even bank; the default now splits each tile's rows, the measured fastest placement) and is rewritten as `test_default_selection_splits_each_tiles_rows_across_the_fibers`; four `test_samsung_owner_maps.py` tests supplied the vendor declaration (identity row packing) with an edited movement layout, which the round 37 derivation check refuses (`row_slots:identity` declared, `row_slots:movement` derived): the helper now supplies the edited decision without the stale declaration (`r39/samsung_owner_maps.log`: 7 passed in 11 min, the conductor runs included) |
| `tests/pim/test_shared_semantics.py` whole (`r39/shared_semantics_full.log`) | 123 passed (12 min; the native AiM executor set) |
| UPMEM, AttAcc and shared-stage suites with the native AiM executor (`r39/pim_backends_targeted.log`: `test_attacc_virtual`, `test_upmem_*`, `test_layout_selection`, `test_g6_*`, `test_reduction_policy`, `test_host_*`, `test_work_grid`, `test_match_*`, `test_import_surface`; launched before the test repairs below) | 299 passed, 29 failed, 4 skipped in 50 min: 22 UPMEM tests on the uPIMulator Docker ENOSPC, the 3 S-17 tests, and 4 since repaired (the three `test_reduction_policy.py` Samsung tests and the sdist test; `r39/redeclared_recheck.log`); every AiM `test_g6_*` run passes natively |
| `tests/pim` without the Samsung and hardware files (`r39/pim_other_backends.log`), Docker executor | stopped at `--maxfail=30`: 30 failed, 268 passed, 37 skipped in 64 min. Attribution: 10 AiM simulator tests failing on the Docker ENOSPC (all pass with the native executor above: `test_aim_axis_roles`, `test_aim_owner_maps`, `test_gemv_all_targets -k aim`, the AiM golden rows; the `test_g6_*` AiM runs in `r39/pim_backends_targeted.log`); 13 UPMEM tests failing on the same ENOSPC (uPIMulator's SDK image; blocked); 7 Samsung expectations stale since rounds 36-38, never run in those rounds: 4 repaired in this round (`test_reduction_policy.py` candidate and pruned sets with the three tile-row families, its temporal ranking restated above the per-tile families under the staging share, `test_samsung_forced_regrouping_layout_fails_under_strict` fixed by the re-declaration normalization itself (one body's declaration supplied for the other body), `test_sdist_tree_compiles_samsung_gemv_outside_the_checkout` expecting `bank_row`), 3 recorded open as S-17 (`test_g6_decisions.py` cost-perturbation tests). Re-check after the repairs (`r39/redeclared_recheck.log`: `test_reduction_policy.py`, `test_supplied_declarations.py`, `test_row_pairing.py`, the whole `test_shared_semantics.py`, the sdist test): 143 passed, the one failure being the ranking assertion before its final form, which passes alone (`1 passed`) |

The native build wrote `libramulator.so` into the shared simulator tree
(the upstream CMake's library output directory is the source root); the
Docker-built `build/ramulator2` will load it when Docker runs again, and
the Docker route is to be re-smoked (689 cycles) then.


## Round 40 (2026-11-04): duplicate declarations refused; the channel budget as a compiler policy; the token study as a diagnostic with full provenance

Working tree on top of `128aef8`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r40/`.

| Check | Result |
|---|---|
| Reviewer's duplicate-declaration probe (`reviewer-r39/declaration_duplicates.py`) re-run (`r40/declaration_duplicates_rerun.log`) | a second layout of `left` contradicting the first's packing and a second transfer of `left` contradicting the first's route are refused by direct validation (`declares two layouts of 'left' (image 'value'): storage 'banks' packing 'unrealized_packing' and ... 'half_bank:0,1'`; `declares two transfers of 'left' in: ('host_copy',) and ('host_scatter', 'banks')`) and by `allo.compile(..., layout=...)` |
| `tests/pim/samsung_hbm_pim/test_supplied_declarations.py` (11 passed) | the four duplicate cases (layout and transfer, duplicate first and duplicate last) refused at both boundaries with the validator's own problem, beside the five earlier mutations, the lockstep reuse and the bare/foreign reuse |
| Resource budget (`r40/budget_probe.log`; `tests/pim/test_resource_budget.py`, 3 passed; `tests/pim/test_aim_token_partition.py -m "not paper_full"`, 5 passed) | under `resources={"channel": 8}` the AiM replica and token partition domains are {1, 2, 4, 8} on both kernels of the 1x512 attention (216 and 72 scored plans become 144 and 48), the normal selection and the lowered program occupy at most 8 channels (QK 8, SV 4; unbudgeted QK 32), the budget is on every result, a placement selected without the budget is refused under it, a cap above the device (64), an undeclared unit and a non-positive cap are refused, and a tightened realization cap is a feasibility rejection the search acts on (one-channel programs selected) |
| Token-partitioning driver (`tests/perf/test_token_partition_driver.py`, 3 passed; `r40/tp_1x128.*`) | the headline plan is the normal budgeted selection, the forced plans are ablations under the same budget, every plan's record and artifacts (plan.json with placements' declared effects, complete lowering manifest, graph activities, resource budget; the exact segment traces; the simulator output) are written before the next plan runs, a failed plan is recorded without discarding the others, a run of the wrong kind is refused, the summary states the figures are not comparable with the paper's |
| AiM run provenance (one plan's record, `r40/tp_1x128.artifacts/h1_l128/default/plan.json`) | executor `native`, binary `f5bf8671e3d0...`, the resolved `libramulator.so` (`df5bb9218c7c...`, the file this round's native build wrote into the simulator tree), the configuration's SHA-256 (`d692087...`), one segment of 30 lines with its digest, repeat 1, 1,141 raw cycles, return code 0, the invocation command and working directory, the resource budget and the compiler revision (`128aef8`, dirty) |
| Nine-workload study under the budget (`r40/token_partition_nine.{json,log}`, artifacts beside) | pass, no failed plan; the normal selection under the budget reaches 6 of 9 simulated minima with a largest gap of 1.34% and equals the forced plan of its P on every workload; the misses are the three 1-head workloads (P2 selected vs P4 best at 0.34%, P4 vs P8 at 0.49%, P1 vs P8 at 1.34%); 37 distinct plans over the nine workloads, 29 split plans merge-checked (exact on integers, within 3.7e-3 on BF16), the unsplit plans recorded as having no merge; for 1 head x 4096 tokens head-only / selected 2.17x; the same cells as the round 39 and the 2026-11-02 runs; the figures are a diagnostic (M-15), not a reproduction of the paper's row |
| Suites at this revision | `tests/perf` whole (`r40/perf_all.log`): 563 passed, 7 failed, of which 4 APU v1 tests hit a transient syntax error while `apu_v1_vector_program.py` was being edited (re-run: 57 passed) and 3 Docker-executor tests ran with the native executor configured in the environment and saw one more Docker call from the run's image-id lookup (the lookup is removed, the tests clear `TENON_AIM_RAMULATOR2`; re-run: 24 passed). AiM suites with the native executor (`r40/aim_suites_native.log`: `test_aim_*`, `test_cent_aim_workloads`, `test_paper_golden -k aim`, `test_g6_*` and `test_layout_selection -k aim`): 60 passed, 22 skipped, the one failure the sdist test in the same transient window (re-run: 1 passed, `r40/sdist_rerun.log`). The APU v1 vector plans' declarations: the DMA plans declared the result's inbound and outbound routes as two `inout` transfers, which the duplicate rule refuses; they are now one `in` and one `out` record (a route used in both directions stays one `inout`), and the 21 APU v1 / carried / fault-injection shared-semantics tests pass. Contract suites (`r40/contract_suites.log`: whole `test_shared_semantics.py`, `test_supplied_declarations.py`, `test_reduction_policy.py`, `test_row_pairing.py` simulator-free): 146 passed, 1 failed in 37 min, the failure being the APU v1 ranking test before the DMA plans' transfer records were corrected (it passes in the 21-test re-run above) |


## Round 41 (2026-11-04): study failures never misreported; the budget contract bounded; failure provenance retained

Working tree on top of `0694791`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r41/`.

| Check | Result |
|---|---|
| Reviewer's control-failure probe (`reviewer-r40/control_failure_probe.py`, 1 head x 4096 tokens with the distinct head-only control failing) re-run (`r41/control_failure_probe_rerun.log`) | `failed_plans` {head-only}, aliases {fixed-8: P8} only, `head_only_over_selected` None; P1 stays the flattened mapping, never relabelled as the control |
| Reviewer's budget probe (`reviewer-r40/budget_probe.py`) re-run (`r41/budget_probe_rerun.log`) | `{'pseudo_channel': 1}` on Samsung and `{'bank': 1}` on AiM are refused on both backends before enumeration: "enforces no budget on ... budgets are enforced for aim: channel" |
| Reviewer's paper_full probe (`reviewer-r40/failure_test_probe.py` with the repository's P2 wrong-kind injection) re-run (`r41/failure_test_probe_rerun.log`) | `test_nine_workload_selection` fails: "P2 failed at simulate ...; the minima comparison is incomplete: missing ['P2']" |
| `tests/perf/test_token_partition_driver.py` (7 passed), `tests/pim/test_resource_budget.py` (3 passed), `tests/pim/test_aim_token_partition.py -m "not paper_full"` (5 passed) | a failed forced alternative keeps modes, effects, lowering manifest, graph and completed stages in memory and on disk and voids the comparison; a failed distinct control is never aliased and voids the head-only ratio while the partition comparison stands; aliases are decided before execution; the summary counts complete comparisons only; a nonzero ramulator2 exit (fake binary, fake subprocess) raises `AimSimulatorFailure` whose evidence carries the executor, config digest, exact trace lines and digest, command, return code, stdout and stderr, retained in the plan record, with no cycles reported; the budget refuses `bank`, `bank_group` and `pseudo_channel` |
| Suites at this revision | `tests/perf` whole 574 passed (`r41/perf_all.log`); AiM suites with the native executor plus `test_resource_budget.py` 65 passed, 22 skipped (`r41/aim_suites_native.log`); the contract suites (whole `test_shared_semantics.py`, `test_supplied_declarations.py`, `test_reduction_policy.py`) 142 passed (`r41/contract_suites.log`) |
| M-15 bridge | not implemented in this round; the design probe (`r41/bridge_probe.py`: a bank `gather` of the QK scores is refused, "no kernel layout to follow", because results live in the GPR) fixes the accounting route (host-GPR link volumes under the declared link model) and the pieces to build, recorded on M-15; the study stays DIAGNOSTIC |


## Round 42 (2026-11-04): the host bridge through the shared plan; the study on its bridged boundary

Working tree on top of `6e988e9`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r41/` (`tp_bridge_probe.*`, `token_partition_bridged.*`).

| Check | Result |
|---|---|
| Timeout evidence (reviewer, round 41) | a `TimeoutExpired` from the simulator keeps the partial stdout and stderr it carried beside the trace, the executor identity and the command; `tests/perf/test_token_partition_driver.py::test_simulator_timeout_keeps_the_partial_streams` |
| Host stage through the shared plan (`tests/pim/test_host_compute.py`, 10 passed) | `allo.host_compute` recorded and scheduled with the declared volumes (scores and probs, 1x128: 256 B each); a stage before its producer is refused; Samsung refuses a host stage; the selection graph prices `host_stage:softmax` at the declared link + passes (1x512: 2,048 B / 32 + 32 vectors x 4 = 192 cycles) on the host handle, depending on the QK group and feeding SV; the lowered program carries one `hoststage` entry after QK's commands with readout and upload bytes; a dropped or renamed stage fails realization; an unknown stage kind is refused, not priced at zero; `reference_outputs` matches the reference on an unsplit contraction (and exact on integers); the functional pipeline QK -> stage -> SV passes for an unsplit SV, a split SV (P4) and the neighbouring 2x256 (P2) |
| Bridged compile probe (`r41/bridge_compile_probe.py`) | 1x128: estimate 1,634 (1,586 + the 48-cycle stage), manifest `hoststage` after command 4, realization accepted, functional pipeline error 4.9e-3 |
| Driver on the simulator (`r41/tp_bridge_probe.*`) | 1x128 and 32x128 complete, no failed plan: 1x128 normal selection P2 1,221 total (1,141 device + 48 stage + 32 reduction) against P4 1,217, gap 0.33%; 32x128 P1 7,424 (5,888 + 1,536 stage); every plan's functional pipeline within 3.5e-3; measured host-stage medians 18-35 us per call on this host beside the 48 / 1,536-cycle prices; the CENT attention control for 32x128 11,010 cycles on CENT's boundary, unavailable for 1 head |
| Nine workloads + neighbour, bridged (`r41/token_partition_bridged.{json,log}`) | pass, no failed plan, every comparison and control complete: the normal budgeted selection reaches 6 of 9 simulated minima with a largest gap of 1.04% (round 40 without the bridge: 1.34%; the paper: 7 of 9, 2.63%); the misses are the three 1-head workloads (P2 selected vs P4 best at 0.33%, P4 vs P8 at 0.44%, P1 vs P8 at 1.04%); the normal selection equals the forced plan of its P on every workload; head-only / selected 1.91x for 1 head x 4096 tokens (13,539 / 7,071); the neighbour 2x256 selects P2 against P4 best at 0.46%; 37 distinct plans, 29 split plans merge-checked exactly on integers, every plan's functional pipeline within 4.9e-3 of the float64 reference; the CENT attention control 11,010 / 18,126 / 95,090 cycles for 32 heads x 128 / 512 / 4,096 tokens (CENT's boundary: cache append included, 32 channels, softmax outside); measured host-stage medians 18 us (1x128) to 530 us (32x4096) per call on this host against priced 48 to 49,152 cycles, recorded separately |
| Suites at this revision | `tests/perf` whole 575 passed (`r41/perf_all_b.log`); contract and host-program suites (whole `test_shared_semantics.py`, `test_host_program_analysis`, `test_host_transfer_*`, `test_supplied_declarations`, `test_reduction_policy`, `test_resource_budget`, `test_host_compute`, `test_aim_token_partition`) 193 passed (`r41/contract_suites_b.log`); AiM, decision, layout-selection, work-grid and matcher suites with the native executor 188 passed, 22 skipped, 3 failed, the three being the open S-17 Samsung cost-perturbation tests (`r41/aim_suites_b.log`) |


## Round 43 (2026-11-04): host-stage invariants, scaled attention semantics, timing accounting (M-17)

Working tree on top of `3a40971`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r43/`.

| Check | Result |
|---|---|
| Reviewer's host-stage probe (`reviewer-r42/host_stage_probe.py`, the `operation=` argument added) re-run (`r43/host_stage_probe_rerun.log`) | the lowered stage moved before QK is refused ("orders the host stages differently from the schedule: realized [stage, launch, launch], declared [launch, stage, launch]"), moved after SV refused likewise, carrying another computation refused ("carries a computation other than the one the host program recorded"); the stage renamed `normalization` with `operation="softmax"` compiles; the stage reading the region input `q` compiles |
| Reviewer's attention-semantics probe re-run (`r43/attention_semantics_probe_rerun.log`) | the stage's peak probability equals the historical scaled softmax's (0.0108642578125 against the earlier unscaled 0.061035); end-to-end error against the scaled reference 0.0; the round 42 measured-host total omitted the 16-cycle link term, now retained |
| `tests/pim/test_host_compute.py` (18 passed) | the label is free and the operation prices (two labels one price; one label two operations two prices; an undeclared operation refused at recording; `AimHostStage` without an operation refused); the realization binds position and computation (early, late, function, a stage between repeated SV records); a renamed stage prices identically through the graph and lowers with its operation; a host-input stage compiles; the scaled softmax verified on linspace scores (peak 0.011 against 0.06 unscaled); the functional pipeline passes on unit-variance operands for an unsplit SV, a split SV and the neighbour, with the unscaled-reference discrepancy above 10x the tolerance |
| `tests/perf/test_token_partition_driver.py` (9 passed) | the measured-host total composes device + reduction + stage link + measured compute (16 + 32 cycles split for 1x128); the functional check records the head dimension |
| Nine workloads + neighbour re-run on the repaired boundary (`r43/token_partition_bridged.{json,log}`) | pass, no failed plan, every comparison and control complete; the same cells as round 42 (the simulator's cycles are data independent): 6 of 9 simulated minima, largest gap 1.04%, the neighbour 2x256 at 0.46%, head-only / selected 1.91x for 1 head x 4096 tokens; every plan's functional pipeline within 9.6e-3 of the scaled float64 reference on unit-variance operands, the unscaled reference off by at least 5.3x the reference peak; the host price stored as link and compute terms (1x128: 16 + 32; 32x4096: 16,384 + 32,768) and the measured-host totals retaining the link term (1x128 default 20,273); CENT attention control 11,010 / 18,126 / 95,090 for the 32-head workloads |
| Suites at this revision | `tests/perf` whole 576 passed (`r43/perf_all.log`); contract and host-program suites (whole `test_shared_semantics.py`, host program and transfer tests, `test_supplied_declarations`, `test_reduction_policy`, `test_resource_budget`, `test_host_compute`, `test_aim_token_partition`) 200 passed (`r43/contract_suites.log`); AiM, decision, layout-selection, work-grid and GEMV suites with the native executor, the three S-17 tests deselected, 132 passed, 22 skipped (`r43/aim_suites.log`) |


## Round 44 (2026-11-04): launch occurrences and completion in the realization check

Working tree on top of `b0c667b`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r44/`.

| Check | Result |
|---|---|
| Reviewer's realization probe (`reviewer-r43/realization_probe.py`) re-run (`r44/realization_probe_rerun.log`) | the duplicated QK contraction is refused ("lowered primary record 'attention_qk_0' where the schedule expects host stage 'softmax'"), so it no longer reaches the simulator; the repeated program (QK, softmax, split SV, a stage reading heads_out, SV again) is accepted with its seven records; the stage moved before the first SV's host reduction and the dropped first reduction are both refused ("launch 0 of group 'attention_sv' is incomplete before the next step: missing ... host reductions ['attention_sv_0']") |
| `tests/pim/test_host_compute.py` (24 passed) | the repeated-launch program with two stages lowers three contractions, two reductions and two stages in schedule order; the duplicate (with and without host stages), the stage before the merge, the dropped merge and a reduction with another P are refused at materialization; the earlier mutations (early, late, function, between repeats, dropped, renamed outputs) still refused |
| Suites at this revision | `tests/perf` whole 576 passed (`r44/perf_all.log`); contract and host-program suites (whole `test_shared_semantics.py`, host program and transfer tests, `test_supplied_declarations`, `test_reduction_policy`, `test_host_compute`) 198 passed (`r44/contract_suites.log`); AiM, decision, layout-selection, GEMV, token-partition and budget suites with the native executor, the S-17 tests deselected, 121 passed (`r44/aim_suites.log`); the 11 `paper_full` CENT golden rows, repeated launches included, exact under the inventory check (`r44/aim_golden_full.log`) |


## Round 45 (2026-11-04): the host reduction's work domain bound to its producing contraction

Working tree on top of `25c482d`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r45/`.

| Check | Result |
|---|---|
| Reviewer's reduction-volume probe (`reviewer-r44/reduction_volume_probe.py`) re-run (`r45/reduction_volume_probe_rerun.log`) | the unmodified 2 heads x 128 tokens, P = 2 program is accepted with the lowered merge priced at 64 cycles, equal to the selection graph's 64, and its functional merge within 1.6e-3; the reductions with outputs 1, batches 1 and replicas 2 are refused at materialization ("merges (outputs, batches, replicas, partitions) (1, 2, 1, 2) of 'attention_sv_0'; its contraction produces (128, 2, 1, 2)", and likewise for the other two) |
| `tests/pim/test_host_compute.py` (31 passed) | the three field mutations refused on the first SV launch and on the later repeated launch (six cases); the lowered merge cost equals the priced graph's `host_reduction` latency (64 cycles) for the single and the repeated program; every earlier occurrence, order, computation, operation and declaration regression still passes |
| Eight-channel CENT control probe (`r45/cent_budget_probe.{py,log}`) | the CENT attention case (`attention_l128`, 32 heads, 4 replicas x 8 channels) selects and simulates unbudgeted (QK and SV on 32 channels, 11,010 cycles) but under `resources={"channel": 8}` no placement of its first grouped reduction materializes ("2 partitions give a 256-element bank-pair span, the reduction is 64"): CENT's program cannot be re-selected within the study's device budget; a comparable eight-channel control needs a CENT-derived SV program at the study's shapes (M-15, next) |
| Suites at this revision | `tests/perf` whole 576 passed (`r45/perf_all.log`); contract and host-program suites 205 passed (`r45/contract_suites.log`); AiM, decision, layout-selection, GEMV, token-partition and budget suites with the native executor, the S-17 tests deselected, 121 passed (`r45/aim_suites.log`); the 11 `paper_full` CENT golden rows exact (`r45/aim_golden_full.log`) |


## Round 46 (2026-11-04): each contraction record bound to the retained shape under the selected candidate

Working tree on top of `152095e`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r46/`.

| Check | Result |
|---|---|
| Reviewer's producer-domain probe (`reviewer-r45/producer_domain_probe.py`) re-run (`r46/producer_domain_probe_rerun.log`) | the unmodified program is accepted (33 commands, 1,178 simulated cycles, merge 64 = graph 64); the coordinated contraction-and-merge change to outputs 256 x batches 1 is refused at materialization ("realizes {'outputs': 256, 'batches': 1, ...}; the retained program under the selected candidate gives {'outputs': 128, 'batches': 2, ...}"); the replica change is refused as not realizable (channel count must equal replicas x channels per replica) |
| `tests/pim/test_host_compute.py` (39 passed) | the coordinated outputs/batches, the replica and the reduction mutations refused on the single and the repeated program (the reduction one by the coverage check: 16,384 logical scalar MACs declared against 32,768 performed); the neighbouring 4 heads x 256 tokens x head dimension 64 workload passes unmodified with P1 and P2, functionally checked, its merge priced as the graph prices it |
| Suites at this revision | `tests/perf` whole 576 passed (`r46/perf_all.log`); contract and host-program suites 211 passed, 2 failed (`r46/contract_suites.log`), the two being the reduction-mutation regression before its expected message included the coverage check's refusal, launched before that edit; the whole `test_host_compute.py` re-run 39 passed (`r46/host_compute_final.log`); AiM, decision, layout-selection, GEMV, token-partition and budget suites with the native executor, the S-17 tests deselected, 119 passed, 2 failed (`r46/aim_suites.log`): the two APU v1 board tests failed with `ledag-ssh exited with 255 while reading the device log` because the shared `/tmp` was 100% full (2.8 MB free; X-24's scratch pattern), and both pass after my own stale `/tmp/tenon-release` scratch (venv, probes, builds from 1-2 November) was removed (477 MB free; `r46/apu_v1_rerun2.log`); the 11 `paper_full` CENT golden rows exact (`r46/aim_golden_full.log`) |


## Round 47 (2026-11-04): realized launch records compared with the derivation record by record (M-18)

Working tree on top of `65750db`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r47/`.

| Check | Result |
|---|---|
| Reviewer's contract probe (`reviewer-r46/contract_probe.py`) re-run (`r47/contract_probe_rerun.log`) | the unmodified program is accepted (estimate 1,798, 33 commands, 1,178 simulated cycles); the QK record with an added activation, another result GPR address or another reuse group is refused at materialization with the field named ("activation: realized True, expected False", "result_gpr: realized 7, expected 0", "reuse_group_size: realized 1, expected 16"); the storage-extent change stays refused as not realizable; the 3 heads x 192 tokens x head dimension 64 neighbour passes its functional check |
| `tests/pim/test_host_compute.py` (50 passed, `r47/host_compute_final.log`) | activation, reuse group, result GPR, input GPR and channel-set mutations of the QK record refused on the single and the repeated program with the differing fields named; the unmodified single, repeated and neighbouring programs match their derivation record by record; every earlier regression still passes |
| Suites at this revision | `tests/perf` whole 576 passed (`r47/perf_all.log`); contract and host-program suites 222 passed, 2 failed (`r47/contract_suites.log`), the two being the dimension regression before its expected message included the whole-record comparison's wording, launched before that edit, the file re-run 50 passed; AiM, decision, layout-selection, GEMV, token-partition and budget suites with the native executor, the S-17 tests deselected, 121 passed (`r47/aim_suites.log`), the APU v1 board tests included with `/tmp` at 97%; the 11 `paper_full` CENT golden rows exact (`r47/aim_golden_full.log`) |


## Round 48 (2026-11-04): the emitted bank payload bound to the derivation (M-18)

Working tree on top of `35b6351`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r48/`.

| Check | Result |
|---|---|
| Reviewer's bank-payload probes (`reviewer-r47/bank_payload_probe.py`, `duplicate_bank_probe.py`) re-run (`r48/*_rerun.log`) | the unmodified program is accepted (estimate 1,798, 33 commands, 1,178 simulated cycles); the all-bank QK record whose payload is replaced by sixteen banks is refused ("expands 'attention_qk_0' over banks (0, ..., 15); the retained program under the selected decision derives None"), and so is the payload of bank 0 sixteen times; neither reaches the simulator |
| `tests/pim/test_host_compute.py` (54 passed, `r48/host_compute_final.log`) | the sixteen-bank, the repeated-bank-0 and a four-bank payload on the all-bank QK record are refused (the four-bank one by the coverage check: 8,192 scalar MACs issued for 32,768); an enumerated single-bank QK candidate lowers with its derived payload (MAC_SBK over distinct banks, manifest scope sbk) and an all-bank one without, both accepted by the whole emission-input comparison; every earlier regression still passes |
| Suites at this revision | `tests/perf` whole 576 passed (`r48/perf_all.log`); contract and host-program suites 227 passed, 1 failed (`r48/contract_suites.log`), the one being the four-bank payload regression before its expected message included the coverage check's refusal, launched before that edit, the file re-run 54 passed; AiM, decision, layout-selection, GEMV, token-partition and budget suites with the native executor, the S-17 tests deselected, 121 passed (`r48/aim_suites.log`); the 11 `paper_full` CENT golden rows exact (`r48/aim_golden_full.log`) |
