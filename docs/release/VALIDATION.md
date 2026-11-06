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


## Round 49 (2026-11-04): the CENT-derived eight-channel control, the host protocol, the calibration of the one-head misses

Working tree on top of `586f575`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r49/`.

| Check | Result |
|---|---|
| `tests/perf/test_token_partition_driver.py` (13 passed) | the CENT-derived control is the budgeted candidate carrying CENT's policy for 1, 8 and 32 heads (QK row-packed over 8 channels, SV channel-batched over 1 / 8 / 8 channels, unsplit, 16-slot reuse) and compiles as an explicit layout; the measured-protocol total composes device + both link terms + the measured protocol; the reduction price splits into link and compute terms |
| Nine workloads + neighbour with the CENT-derived control and the protocol samples (`r49/token_partition_bridged.{json,log}`) | pass, no failed plan or control, every comparison complete: 6 of 9 simulated minima, largest gap 1.04% (the same device cells as rounds 42-43); 40 distinct plans (the CENT-derived control added), 29 split plans merge-checked, every functional pipeline within 9.6e-3; the CENT-derived control (QK row-packed over 8 channels, SV channel-batched over 1 / 8 / 8 channels, unsplit, reuse 16) is the normal selection's own plan on every 8- and 32-head workload (ratio 1.00) and slower than it on the 1-head ones (1.03x at 128, 1.32x at 512, 2.23x at 4096 tokens) and on the neighbour (1.05x); the measured host protocol (decode + scaled softmax + staging + merge) is 31 us (1x128) to 899 us (32x4096) per call on this host, composed with the modelled link terms into `total_cycles_measured_host_protocol` beside the priced total |
| Command-level calibration (`benchmarks/cent_aim/command_calibration.py`; `r49/calibration_{1x4096,1x128,1x512}.{json,log}`) | per-command increments on the simulator against the profile's class charges: WR_BIAS 2 and RD_MAC 4-7 cycles per command simulated against 35 and 34 charged (QK at 1x4096: 64 + 129 against 1,120 + 1,088); WR_GB 51 (one 128-element vector) to 131 (a flattened 4096-token write) against a constant 35; a reuse group's first MAC_ABK 101-245 against 56; the row-packed QK's consecutive MAC columns 3,534 against 480 + 2,728; SV spans: 1x4096 P1 1,656 simulated against 1,372 modelled, P8 1,455 against 1,795, which inverts the order; 1x128 P2 against P4 951 against 915 (model 1,347 against 1,315, totals tied by the 32-cycle host reduction step); 1x512 P4 against P8 1,023 against 951 (model 1,411 against 1,347, totals tied likewise) |
| Suites at this revision | `tests/perf` whole 580 passed (`r49/perf_all.log`); the AiM token-partition, budget, host-compute, cost-route, program, CENT-workload and golden suites with the native executor 96 passed (`r49/aim_quick.log`) |

## Round 50 (2026-11-04/05): the frozen study's control restored, the host protocol on the attention's inputs, the contraction profile recalibrated (M-19)

Working tree on top of `2e7453e`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r50/`.

| Check | Result |
|---|---|
| Controlled single-factor command streams on the simulator (`r50/controlled_variations.{py,json,log}`, `controlled_variations_sbk.{py,json,log}`; EOC-only baseline 2 cycles) | WR_GB op size 1..64: 2n in register mode, 2n + 3 after a read, +32 as the first command; WR_BIAS runs 1..16: 2 each; first MAC_ABK after WR_BIAS, 1..64 columns, fresh row: 87 + 2(c - 1) including the mode change; consecutive MAC_ABK at new rows (2..9 lines, c = 2 / 8 / 32): 100 + 2(c - 1) per line; at the open row: 2c; one MAC per extra reuse group at a new row: 104 (c = 8), at the open row 16; MAC_SBK lines rotating 16 banks at new rows: 89 / 91 / 103 (c = 1 / 2 / 8), on one bank: 100 / 102 / 114, 16 fresh banks: 58.9 / 60.9 / 72.9 per line (the mode change amortized); RD_MAC runs 1..32: 36 then 3 per extra read; the same group on 1 / 2 / 4 / 8 channels: 188 each; reuse groups 1 / 2 / 4 / 8: 188 / 362 / 710 / 1406 at new rows, 188 / 274 / 446 / 790 at the open row |
| Recalibrated charges (`allo/pim/costs/aim.py::contraction_profile_charges`, `aim_lowering.aim_contraction_profile`) against the round 49 per-command increments (`r49/calibration_*.json`, kernel spans) | QK 1x128: 191 against 190 simulated (before: 207); SV 1x128 P2 / P4: 952 / 916 against 951 / 915 (before: 1347 / 1315); SV 1x512 P4 / P8: 1024 / 952 against 1023 / 951 (before: 1411 / 1347); QK 1x4096: 3880 against 3879 (before: 5539); SV 1x4096 P1 / P8: 1666 / 1456 against 1656 / 1455 (before: 1372 / 1795, the inverted pair) |
| `tests/pim/test_aim_cost_routes.py` (6 passed) | the standalone GEMV probe: ABK 223 against 222 simulated (unchanged), SBK 1513 against 1527 (pin updated from 1528; the one extra cycle per consecutive fresh single-bank line) |
| `tests/pim/test_aim_profile_calibration.py` (5 passed, new) | the constants are the simulator's timings (N_RTP 12, CMD_RD_REG 3 = nCLREG + nBL); the single-group QK's classes and charges (191); the four-way channel-batched SV's classes (one fresh, seven in-run switches, three mode changes; 916); the historical control's reuse-1 SV pays two mode changes and a register run per group and its row-packed QK walks the open row; splitting one group into eight changes the price by exactly the per-group structure (385 cycles), not N_MODCH per register command |
| Held-out validation (`r50/heldout_validation.{py,json,log}`): every distinct budgeted candidate of each kernel, the other kernel at its default selection, compiled through the shared pipeline and simulated whole and to the QK boundary; the kernel's simulated span against the recalibrated charge and the previous charges recomputed from the same profile | 20 kernel sets (four neighbouring shapes 3x192x64, 5x320x64, 2x256x128, 4x256x64; the study's 1x128/512/4096; the golden 32x128/512/4096), 2,520 candidates: mean absolute error 0.76% (largest 6.9%, a single-bank candidate after a kernel that left rows open), 168,272 of 168,328 ordered pairs in the simulator's order (99.97%), the simulator's best candidate the model's best in all 20 sets; the previous charges: 19.3% mean error (largest 107%), 165,830 pairs (98.5%); of the 56 disagreeing pairs the largest simulated gap is 1.14% and 51 are single-bank against all-bank candidates; per kernel below |
| `r50/probe_sbk_lines.{py,log}` | on the 3 x 192 x 64 SV after the QK, every MAC_SBK line of the single-bank channel-batched candidate costs 99 cycles, the 16 lines on banks the profile sees as fresh included (the QK left rows open in all 16 banks): the previous kernel's context costs a single-bank expansion one precharge per bank; recorded as a residual (design doc 3a, ISSUES M-19) |
| Nine workloads + neighbour on the recalibrated profile, both controls, whole host protocol (`r50/token_partition_nine.{json,log}` + `.artifacts/`) | pass, no failed plan or control, every comparison complete: 9 of 9 simulated minima, every gap 0 (the same device cells as rounds 42-49), estimates within 0.1% to 0.3% of the measured totals; 49 distinct plans, 30 split plans merge-checked, every functional pipeline within 3.7e-3; cent-repaired / selected 2.22 / 3.85 / 7.27 (1 head), 1.45 / 1.46 / 1.46 (8 and 32 heads), 2.39 (2 x 256); cent-policy / selected 1.04 / 1.32 / 2.25 (1 head), 1.00 (8 and 32 heads), 1.05 (2 x 256); the whole host protocol 32 us (1 x 128) to 623 us (32 x 4096) per call, inputs' score standard deviation 10 to 12, every probability populated, largest at most 0.09 |
| `tests/perf/test_token_partition_driver.py` (18 passed within the perf suite) | the historical control is the padded program's candidates with the producer's knobs for 1, 8 and 32 heads (physical 8 / 8 / 32), its SV inventory one WR_BIAS run per MAC, its provenance names the producer's lines; the two controls differ and are recorded under their own names; the whole-protocol median is the comparable sample with 31 samples after ten warm-ups, the inputs the attention's (finite, every probability populated), staging encodes once |
| Suites at this revision | `tests/perf` whole 584 passed, 1 skipped (`r50/perf_all.log`); the AiM token-partition, budget, host-compute, cost-route, profile, program, CENT-workload and golden suites with the native executor 144 passed, 23 skipped (`r50/aim_quick.log`; the three UPMEM golden rows fail on Docker ENOSPC as before, deselected here) |

Held-out validation per kernel set (model error mean / largest; ordered
pairs in the simulator's order; the previous charges beside):

| Program | Kernel | Candidates | Error now | Pairs now | Error before | Pairs before | Best |
|---|---|---:|---:|---:|---:|---:|---|
| 3 x 192 x 64 | QK | 156 | 0.7% / 1.4% | 12008 / 12008 | 24.7% / 76.2% | 11625 / 12008 | same |
| 3 x 192 x 64 | SV | 108 | 1.7% / 6.9% | 5669 / 5670 | 16.8% / 43.0% | 5650 / 5670 | same |
| 5 x 320 x 64 | QK | 168 | 0.6% / 1.2% | 13982 / 13992 | 27.1% / 106.8% | 13763 / 13992 | same |
| 5 x 320 x 64 | SV | 120 | 1.0% / 5.4% | 7020 / 7020 | 15.0% / 42.6% | 6993 / 7020 | same |
| 2 x 256 x 128 | QK | 132 | 0.7% / 1.3% | 8537 / 8552 | 19.9% / 48.6% | 8453 / 8552 | same |
| 2 x 256 x 128 | SV | 84 | 1.0% / 3.1% | 3444 / 3444 | 16.9% / 45.5% | 3419 / 3444 | same |
| 4 x 256 x 64 | QK | 156 | 0.7% / 1.3% | 11990 / 12000 | 27.2% / 92.9% | 11768 / 12000 | same |
| 4 x 256 x 64 | SV | 108 | 1.5% / 6.7% | 5670 / 5670 | 16.1% / 43.5% | 5647 / 5670 | same |
| 1 x 128 x 128 | QK | 144 | 0.8% / 1.6% | 9988 / 9988 | 14.2% / 45.1% | 9661 / 9988 | same |
| 1 x 128 x 128 | SV | 48 | 1.6% / 3.3% | 1104 / 1104 | 17.8% / 44.8% | 1085 / 1104 | same |
| 1 x 512 x 128 | QK | 144 | 0.8% / 1.6% | 10144 / 10144 | 17.0% / 49.1% | 10036 / 10144 | same |
| 1 x 512 x 128 | SV | 48 | 1.3% / 3.1% | 1104 / 1104 | 14.4% / 41.6% | 1093 / 1104 | same |
| 1 x 4096 x 128 | QK | 144 | 0.8% / 1.6% | 10164 / 10164 | 17.7% / 49.0% | 10068 / 10164 | same |
| 1 x 4096 x 128 | SV | 96 | 0.6% / 3.0% | 4440 / 4444 | 6.5% / 23.4% | 4400 / 4444 | same |
| 32 x 128 x 128 | QK | 168 | 0.6% / 1.1% | 13941 / 13944 | 24.2% / 84.0% | 13722 / 13944 | same |
| 32 x 128 x 128 | SV | 120 | 0.5% / 1.0% | 7080 / 7080 | 20.1% / 48.3% | 7052 / 7080 | same |
| 32 x 512 x 128 | QK | 168 | 0.5% / 1.1% | 13997 / 14004 | 24.1% / 84.1% | 13720 / 14004 | same |
| 32 x 512 x 128 | SV | 120 | 0.4% / 0.9% | 7080 / 7080 | 15.2% / 44.9% | 7057 / 7080 | same |
| 32 x 4096 x 128 | QK | 168 | 0.5% / 1.1% | 13998 / 14004 | 24.0% / 84.2% | 13706 / 14004 | same |
| 32 x 4096 x 128 | SV | 120 | 0.3% / 0.6% | 6912 / 6912 | 7.4% / 25.5% | 6912 / 6912 | same |

Not evidence of comparability: the row stays DIAGNOSTIC (declared host
compute price; the recalibrated selections await the reviewer); the paper's
seven of nine is on the paper's own boundary.

## Round 51 (2026-11-05): the AttAcc kernel-tiling figure through the shared pipeline (required G10 rows), scope column, documentation defects

Working tree on top of `0cc8187`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r51/`.

| Check | Result |
|---|---|
| Native Ramulator2-AttAcc (`benchmarks/attacc/scripts/build_ramulator_attacc.sh`, sources `DCC/src/src/Ramulator2/AttAcc` (MIT), conda gcc 12; `attacc-native/build.log`) | builds against glibc 2.31 (Docker's data root is full on this host); binary sha256 `01e857a7...`; the qkv batch-1 selected program simulates to 11,454 cycles, equal to the archived container run and the paper's 0.00881 ms |
| `tests/pim/test_attacc_trace_check.py` (9 passed, 1 skipped) and `tests/pim/test_attacc_virtual.py` (49 passed, 2 skipped) | the checker's GEMV and attention sequences re-derive the generator's emission order from the counts record; a reordered trace with equal totals fails, a trace with every MAC on one channel fails the mapping check, a partial barrier group is an error; DCC's own generator traces for the qkv batch-1 selection and three attention shapes (the worst-loss 8 x 128 x 480, 32 x 128 x 4096, the shortest 8 x 128 x 144) match ordering and mapping; the two-row-block attention branch is exercised synthetically only (no paper instance needs two blocks) |
| QKV and projection rows (`r51/attacc_{qkv,proj}_b{1,4}.json` + `.artifacts`) | shared-pipeline selections `[[1,8,1],[8,8,80],[2,1,192]]` 11,454 cycles = 0.00881 ms, `[[1,16,2],[16,4,80],[1,1,384]]` 43,750 = 0.03364, `[[1,8,1],[1,4,32],[16,2,160]]` 3,448 = 0.00265, `[[1,32,1],[2,2,32],[8,1,640]]` 13,318 = 0.01024 (the paper's Tenon values); fixed-form baselines re-run (minimum over 2 / 1 / 2 / 1 fixed-form candidates) 36,504 / 224,170 / 3,448 / 13,398 cycles = 0.02807 / 0.17239 / 0.00265 / 0.01030 ms (the paper's Fixed values); 3,043 / 4,478 / 1,391 / 2,273 candidates ranked, the tied qkv batch-4 and projection batch-1 minima resolved to the paper tiling by the tie-break rule; every trace: 16 channels and 64 units addressed, matrix columns and vector positions as declared |
| Score rows, 248 shapes each (`r51/attacc_score_b{1,4}.json` + `.artifacts`, `r51/attacc_kernel_summary.json`) | no failure; every selection equals `paper_selection.csv` and the archived Tenon manifest's per-shape time on all 496 shapes; batch 1 mean 0.00193 ms, speedup over the fixed tiling 1.00x (mean of per-shape ratios; the figure's 1.00x; DCC's archived 1.01x), 48 wins / 144 ties / 56 losses against DCC's archived per-shape times; batch 4 mean 0.00678 ms, 1.13x (figure 1.13x; DCC 1.12x), 73 / 170 / 5; with the four GEMV rows the 500 instances give 123 wins, 316 ties, 61 losses, median loss 7.07% (text: 123 / 316 / 61, 7.1%); 2 batch-1 shapes slower than the fixed tiling by at most 2.9% (the paper's text already reports losses); the estimate is 0.69-0.95 of the simulated cycles (a ranking model, as documented) |
| `r51/attacc_trace_recheck.json` | all 506 retained traces (10 GEMV incl. fixed-form candidates, 496 attention) re-checked with the final checker: totals, ordering and mapping match the program recorded beside each trace; 8,690,688 commands |
| Not re-run | the fixed and DCC baselines of the 496 score shapes are DCC's archived per-shape simulations (`temp/simulation/@attacc-attn_inference.csv` in the DCC checkout); DCC's own selector was not re-run; the reference-best sweep (398/500) is an optional study |
| `docs/release/paper_reproduction/inventory.csv` | `release_scope` column added to all 83 rows (55 `required_key_result`: 36 REPRODUCED + 6 this round = 42, 7 OPEN, 6 UNVERIFIED; 28 `optional_supporting`), CRLF preserved |
| Suites at this revision | `tests/pim/test_attacc_virtual.py`, `test_attacc_trace_check.py`, `test_aim_profile_calibration.py`, `test_aim_cost_routes.py`: 64 passed, 2 skipped (`r51/pytest_final.log`); no production `allo/` behaviour changed this round (the AiM cost module's docstring and comments only) |

Not evidence of: the six decode configurations (analytical composition,
`compose_e2e.py`, next), the reference-best study, or any numerical
execution (the simulator is timing-only).

## Round 52 (2026-11-05): per-command trace fidelity, failure-path retention, the AttAcc decode configurations

Working tree on top of `eb34f62`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r52/`.

| Check | Result |
|---|---|
| `benchmarks/attacc/trace_check.py` rewritten: the complete command stream a selected program implies (opcode and address of every store, GB write, MAC, bank-group drain, softmax, context move and barrier, in the generator's emission order, with the generator's region placement: vector at 0, matrix at the matrix size, result after the vector size) derived from the program and the target geometry and compared with the trace command by command | the reviewer's three mutations of the retained QKV trace now fail (zeroed store / drain / barrier addresses at command 1; every address shifted by one channel at command 0 with 2,564 commands outside the device; the matrix base shifted by one column at command 10,272), as do a swapped command pair, a dropped store, a displaced GB write and zeroed softmax / context moves on the attention stream (`tests/pim/test_attacc_trace_check.py`, 17 passed) |
| DCC's own generator output against the derived stream | exact on the qkv batch-1 selection, the projection batch-4 selection, a fixed-form baseline candidate and three attention shapes (8 x 128 x 480, 32 x 128 x 4096, 8 x 128 x 144) |
| `r52/attacc_trace_recheck.json` (all retained r51 traces) | 506 of 506 traces equal their program's derived stream command by command (496 attention, 8,707,296 commands including barriers); the r51 timings stand, no trace changed |
| Failure-path retention (`run_ramulator.py::simulate_program`; `test_attacc_trace_check.py` failure tests) | a nonzero simulator exit, a timeout and a trace mismatch each retain the trace, the configuration, the complete stdout and stderr and a structured `failure.json` (stage, command, exit status or timeout, trace check) under the artifacts directory, or keep the work directory with the record when none was given; the driver exits 1 with the retained path in the session's failures |
| Decode composition (`compose_e2e.py`, `r52/attacc_e2e.json`; `tests/pim/test_attacc_compose.py`, 4 passed) | 500 shared-pipeline kernel records (4 reference records ignored), every GPT-13B PIM row measured, the 6 other table rows NaN; Tenon full 6.50 / 6.95 / 7.40 / 3.09 / 3.48 / 3.87x and base 1.47 / 1.56 / 1.66 / 1.34 / 1.50 / 1.65x over the GPU roofline; DCC's selection re-run 6.50 / 6.95 / 7.40 / 3.05 / 3.44 / 3.82x and 1.47 / 1.56 / 1.66 / 1.34 / 1.49 / 1.64x; the fixed baseline with DCC's own placement (attention on PIM, QKV and projection on the GPU) 2.55 / 2.73 / 2.91 / 2.29 / 2.58 / 2.87x and 1.09 / 1.16 / 1.23 / 1.16 / 1.30 / 1.43x; at batch 4 Tenon's decode is 1.27% faster than DCC's in every configuration, QKV 1.2799 to 1.2332 ms of 3.9933 to 3.9431 ms at 128 / 2048; at batch 1 equal to two decimals with Tenon 0.0001 ms slower at 128 / 2048 |
| Explained | the earlier smoke run's batch-4 numbers (1.28 / 1.45 / 1.61x) were the fixed-tiling times substituted for the unmeasured kernels under the PIM placement; with every kernel measured and DCC's placement rule for the fixed variant the composition reproduces all three columns |
| Suites | `tests/pim/test_attacc_trace_check.py` + `test_attacc_virtual.py`: 61 passed, 1 skipped (`r52/pytest_attacc.log`); `test_attacc_compose.py` 4 passed; no production `allo/` change this round |

Not evidence of: numerical execution (the simulator is timing-only), a
measured decode step (the composition is analytical), or DCC's selector
re-run (its recorded per-shape selections are the comparison baseline).

## Round 53 (2026-11-05): X-19 attributed; the Gemini-I confirmation campaign pre-registered

Working tree on top of `a4802c2`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r53/`.

| Check | Result |
|---|---|
| Per-stage prefix simulation of the corrected `qkvo_rope_l128` (`r53/rope_attribution.{py,json}`, native ramulator2, EOC baseline 2) | q / k / v / o projections 30298 / 30300 / 30300 / 30324 cycles; host scatters m_q / m_k / r_q / r_k 2055 / 2048 / 2015 / 2048; multiplies rope_q / rope_k 121 / 126; whole 129,637 (the golden pin) |
| Controlled reference: the same trace with the two EWMULs at the paper's half op size (8 columns) | 129,605 cycles, the paper's digit exactly; the two multiplies 215 against 247 cycles after the same prefix: the 32-cycle difference is the execution of the omitted half of the data |
| Candidate domain of every kernel group (`r53/probe_rope_case.py`) | projections: 48 candidates each, 12 tied at the minimum with identical command streams (reuse groups do not apply: one MAC register per bank, a 1,024-element global buffer, CENT's per-chunk partial readout); elementwise: 4 replica partitions, the cheapest selected (4,272 against 4,288 / 4,320 / ...); no enumerated alternative is cheaper or lowers to a shorter multiply |
| Device lower bound of the multiply | op size = ceil(2D / (channels_per_replica x bank_groups x lanes)) = ceil(8192 / (8 x 4 x 16)) = 16 columns (one bank pair per bank group per channel); the paper's 8 columns cover half the vector |
| Neighbouring program (`r53/rope_neighbor_d2048.json`: D = 2048, H = 16, F = 5504) | the same structure with 8-column multiplies (4 WR_GB + 128 MAC per projection as the counts predict), 35,341 simulated cycles, the two multiplies 215 (equal to the 8-column reference above) |
| `tests/pim/test_cent_aim_workloads.py::test_rope_ewmul_width_is_the_device_minimum_for_the_full_vector` (paper D and the neighbour; 2 passed) | two EWMULs per program at the geometric minimum width, the declared slot capacity at least the logical element count (`physical_element_slots >= logical_elements`: capacity, not address-level coverage of every scattered element), each kernel group's chosen candidate at the minimum estimated cost of its ranking |
| Gemini-I X-22 / X-23 | confirmation campaign pre-registered (ISSUES): three paired sessions per row at the final candidate under the round 30 protocol; not launched at this revision |

Status: the four rows stay OPEN; the attribution and the search by
estimated cost over the enumerated candidates do not establish the minimum
simulated time of a faithful complete program and do not satisfy the
same-or-better rule. No expected value changed, no timing repinned, the
omitted work not restored.

## Round 54 (2026-11-05): Gemini-II launch recipes bound to the contract (X-25)

Working tree on top of `13698c8`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r54/`.

| Check | Result |
|---|---|
| The reviewer's reproducer (`reviewer-r53/g2_realization.py`, `PYTHONPATH=$PWD`, spmw interpreter) after the repair | the damaged `single/k1` (recipe K 16, 10-bit output, declared reduction (0, 32)) is refused at materialization; `single/k2` is selected at 1,494 cycles with recipe K 16 per launch under declared ranges (0, 16) and (16, 32); the artifact packs, so the script's "expected packing failure" assertion fails as intended |
| `tests/pim/test_shared_semantics.py` section q (6 passed) | the unmodified transposed int3 x int3 -> int12[7,12] program selects `single/k1` at 892 cycles (K 32, 11-bit output, final) and packs; the damaged cheapest candidate is rejected with the recorded reason (`reduction_extent realized 16 vs derived 32`, `output_type`) and the intact alternative selected and packed; forcing the damaged candidate fails at compilation; a flattened output geometry and a final launch declared partial are refused; a chained first chunk is refused earlier by the shared candidate contract |
| Suites at this revision | `tests/pim/test_shared_semantics.py` + `tests/pim/test_apu_g2_hardware.py` + `tests/perf/test_compile_interface.py`: 150 passed, 14 skipped, 1 failed (`r54/pytest_shared.log`); the failure is the AiM timing-only run through the Docker route (containerd ENOSPC, the host's full `/var`), which passes with the native executor configured (`r54/pytest_aim_native_rerun.log`, 1 passed); the Gemini-II card tests are skipped on this host |
| Not changed | the APU v1 path; no new compiler route; the AiM, Samsung, UPMEM and AttAcc realization checks |

## Round 55 (2026-11-05): pair-affine coefficients bound to the stream identities (X-25)

Working tree on top of `369e265`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r55/`.

| Check | Result |
|---|---|
| The reviewer's reproducer (`reviewer-r54/g2_stream_order.py`) after the repair | compilation fails: "no feasible placement ... Gemini-II launch 0 (streams (1, 0) ...) carries a recipe the contract does not imply ... epilogue" (the only legal plan is refused at materialization); before, it packed a program computing 128 for a source requiring 112 |
| `tests/pim/test_shared_semantics.py` (3 new, 15 Gemini-II tests passed) | the ordinary pair selects `fused/k1` with streams (0, 1) and coefficients (3, 2), and the recipe's epilogue on the packed operands equals the source result (112 per output); reversed streams with unchanged coefficients are refused; reversed streams carrying the coefficients in their own order ((2, 3)) are accepted and compute 112 |
| Suites at this revision | `tests/pim/test_shared_semantics.py` + `tests/pim/test_apu_g2_hardware.py` + `tests/perf/test_compile_interface.py` with the native AiM executor: 154 passed, 14 skipped, 0 failed (`r55/pytest_shared.log`; the Gemini-II card tests skip on this host) |

## Round 56 (2026-11-05): integration candidate `54cbb08`: installed artifact, backend paths, hardware

Candidate `54cbb08` (clean tree). Evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r55/`.

| Check | Result |
|---|---|
| Local sdist and wheel (`build_wheel.sh`, `pkg/`) | sdist `5aa0ad9c...` from `git archive` of `54cbb08`; wheel `860c82a8...` built with the shared LLVM 22 clang; clean venv (numpy 2.5.4) |
| Installed-artifact probes (`installed.log`, round 8 probe scripts, no PYTHONPATH, safe path, cwd outside the checkout) | fresh install: compile and virtual run from the wheel, no module loaded from the checkout; Samsung simulator probe pass; AiM timing-only 222 cycles through the native executor, outputs untouched; UPMEM probe fails on the host's full Docker data root (containerd ENOSPC, unchanged host condition) |
| Installed-wheel test subsets (`installed-pytest.log`, `installed-cent-aim.log`) | paper golden AiM rows 3 passed 11 skipped (default rows), AttAcc virtual 44 passed 1 skipped, trace check 17, compose 4, CENT AiM workloads (default rows) 6, resource budget 3, host compute 54, AiM profile calibration 5, AiM cost routes 6, AiM program 16, Samsung supplied declarations 11: all passed |
| Installed-wheel `tests/pim/test_shared_semantics.py` (`installed.log`) | 131 passed, 1 failed: `test_local_accumulator_gemm_executes_on_apu_v1_functional_backend`, the recorded NumPy 2.4+ functional-route failure V1-19 (the venv installs numpy 2.5.4; the checkout environment has 2.3.5 where it passes); `installed-apu_v1-functional-rerun.log` |
| zhang-x4 wheel from the same sdist (`g2_build_wheel.sh`, `g2/`) | wheel `ba7bedab...`, clean venv; card `apu-00` Available, FW `GSI_G2_Infrastructure_V1.24.1` |
| Gemini-II card tests from the installed wheel (`g2-card-tests.log`) | 15 passed in 124 s (a first attempt failed when my kept temporaries from earlier rounds filled the host's /tmp; 139 of them removed, the run repeated without `TENON_APU_G2_KEEP_TMP`) |
| Gemini-II paper rows at the candidate (`g2_paired_rows.json`, `g2_paired_rows_s{2,3,4}.json`, `g2_pooled.json`; four paired sessions of 20 rounds, 80 samples per row, GTML interleaved, every output exact) | pooled medians vs paper: gemm 1729.63 / 1733.225 (0.9979), batched_gemm 1743.33 / 1739.15 (1.0024, paper inside the pooled IQR), gesummv 1840.45 / 1860.075 (0.9894), bicg 2538.63 / 2548.825 (0.9960), mvt 2764.48 / 2766.625 (0.9992); GTML pooled medians against the paper's column: gemm 1915.38 / 1903.85 (+0.61%), batched_gemm 1912.93 / 1906.4 (+0.34%), gesummv 3093.93 / 3090.975 (+0.10%), bicg 2728.68 / 2726.5 (+0.08%), mvt 3084.35 / 3089.65 (-0.17%); batched_gemm OPEN (nominal) after its pre-registered campaign (X-26), the other four REPRODUCED at the candidate |
| Gemini-I pre-registered campaign (X-22 / X-23) | not completed at this revision: attempt 1 launched nothing (the driver's `--artifacts` flag is the assets root; misuse mine), attempt 2 lost linear-regression launches 6 to 10 to `ledag-ssh` exit 255 with the host's shared `/tmp` at 100% (consumers not ours; the vendor tool needs about 5 MB of scratch per device-log read); both under `aborted/`, no sample pooled; `campaign_watcher.sh` launches the campaign when `/tmp` has 3 GB free |
| Required G10 set: historical coverage against candidate validation | Historical (at the recorded revisions of the inventory): 47 REPRODUCED, 8 OPEN (X-19 x4, X-22 / X-23 x3, X-26 x1), 0 UNVERIFIED. Executed at the candidate `54cbb08` from installed artifacts in this round: the five Gemini-II rows (4 REPRODUCED, 1 OPEN) and the three default AiM golden rows (installed-wheel subset); not executed at the candidate: the sixteen UPMEM rows (recorded at `0e48d81`; the candidate's UPMEM probe failed on the host's Docker data root), the three Samsung rows and the eleven `paper_full` AiM rows (recorded at `0e48d81` and rounds 25-39), the twelve AttAcc results (rounds 51-52 working trees), the five Gemini-I rows (69773fc / 28cc922) |

## Round 57 (2026-11-05): required AiM and Samsung rows executed at the integration candidate from the installed wheel

Wheel `860c82a8...` of `54cbb08` in the clean venv (`r55/pkg`), tests and
benchmarks copied out of the checkout, `PYTHONSAFEPATH`, the native
timing-only ramulator2 (`r57/run_golden_installed.sh`, `r57/run_aim_provenance.sh`).

| Check | Result |
|---|---|
| `tests/pim/test_paper_golden.py -k aim_cent_paper_cycles` (`golden-aim-default.log`) | 3 passed, 11 skipped (the default rows) |
| the same with `-m paper_full` (`golden-aim-paper-full.log`) | 11 passed: the paper_full rows selected explicitly; the four RoPE rows print their corrected-program pins and measured cycles (`AIM_GOLDEN ... (ISSUES X-19)`) |
| `r57/aim_rows_provenance.json` (every AiM row from the installed wheel: the selection of each kernel group with candidate counts and ties, the lowering manifest, the trace digest, line and opcode counts, the simulated cycles, the executor binary / library / config digests) | see the table below; 10 rows equal the paper's cycles exactly, the 4 RoPE rows are the corrected programs 32 cycles above (OPEN, X-19) |
| `tests/pim/test_paper_golden.py -k samsung_gemv_paper_cycles` (`golden-samsung-default.log`, PIMSimulator) | 3 passed: the vendor-equivalent program at the paper's values (4435 / 8041 / 8247), the default selection faster (4108 / 7057 / 7255, `split_rows+crf_shared`), outputs within the float32 reference; no Samsung row is marked paper_full |
| UPMEM at the candidate | not executed: the host's Docker data root is full (containerd ENOSPC), the UPMEM rows stay at `0e48d81` |

AiM rows at the candidate (timing only; cycles are `memory_system_cycles`):

| Row | Measured | Paper | Difference | Trace sha256 | Lines |
|---|---:|---:|---|---|---:|
| norm_residual_l128 | 11750 | 11750 | exact | `f846600a17b0` | 10271 |
| qkvo_rope_l128 | 129637 | 129605 | +32 (corrected RoPE program, X-19) | `c25203892997` | 9747 |
| attention_l128 | 11010 | 11010 | exact | `29f3190aad66` | 2249 |
| attention_l512 | 18126 | 18126 | exact | `112c7bad9914` | 2549 |
| attention_l4096 | 95090 | 95090 | exact | `103edc8376c5` | 5649 |
| softmax_pim_l128 | 6307 | 6307 | exact | `5bbf4172209a` | 6149 |
| softmax_pim_l512 | 24835 | 24835 | exact | `a6f1c35d91e2` | 24581 |
| softmax_pim_l4096 | 198416 | 198416 | exact | `f40acc441db3` | 196619 |
| ffn_fc_l128 | 250433 | 250433 | exact | `833f14bacd24` | 3312 |
| ffn_activation_l128 | 11775 | 11775 | exact | `e30bbfbe179c` | 11276 |
| ffn_complete_l128 | 262269 | 262269 | exact | `73af1bad404b` | 14587 |
| full_block_l128 | 421191 | 421159 | +32 (corrected RoPE program, X-19) | `05101b9337b0` | 42999 |
| full_block_l512 | 446835 | 446803 | +32 (corrected RoPE program, X-19) | `3603565dc103` | 61731 |
| full_block_l4096 | 697348 | 697316 | +32 (corrected RoPE program, X-19) | `8abfcb04091b` | 236869 |

| The twelve AttAcc results from the installed wheel (`r57/run_attacc_installed.sh`, `attacc_{qkv,proj}_b{1,4}.json` with fixed-form baselines, `attacc_score_b{1,4}.json`, `attacc_e2e.json`) | 504 of 504 selected and baseline timings equal the round 51 values (same tilings, same cycles), every one of the 500 shared-pipeline traces equal to its program command by command, no failure; the six decode configurations equal round 52 to the microsecond (Tenon full 6.50 / 6.95 / 7.40 / 3.09 / 3.48 / 3.87x, base 1.47 / 1.56 / 1.66 / 1.34 / 1.50 / 1.65x; DCC and fixed variants as recorded) |

Executed at the candidate after this round: the five Gemini-II rows (round
56), the fourteen AiM rows, the three Samsung rows and the twelve AttAcc
results; not yet: the sixteen UPMEM rows (the host's Docker data root) and
the five Gemini-I rows (the pre-registered campaign pending on the host's
temporary space).

## Round 58 (2026-11-05): the Gemini-II repetitions update bound to the validated plan (X-27)

Working tree on top of `f1df711` (the candidate `54cbb08` plus this repair);
evidence under `/work/shared/users/phd/nz264/tenon-release-evidence/r58/`.

| Check | Result |
|---|---|
| The reviewer's reproducer (`reviewer-r57/g2_repetitions.py`) after the repair (`r58/reviewer_r57_reproducer_after.log`) | the corrupted enumeration no longer reaches the artifact: after `repetitions = 17` the launch still declares reduction (0, 32) with recipe K = 32, the explicit recipe check passes and the script stops at its own "expected recipe mismatch" assertion |
| `tests/pim/test_shared_semantics.py` section r (5 new; 20 Gemini-II tests passed) | a corrupted enumeration is not consulted by an update (the enumerator is not called); a derivation drifting in reduction extent or output geometry, and a pair derivation with reordered coefficients, are refused with the previous plan, graph and count retained and the artifact packing the source result (112 per output); a valid update of the forced `single/k2` plan keeps both launches' partitions, carries 17 into every recipe and the manifest, re-prices and packs |
| Suites at this revision | `tests/pim/test_shared_semantics.py` + `tests/pim/test_apu_g2_hardware.py` + `tests/perf/test_compile_interface.py` with the native AiM executor: 159 passed, 14 skipped, 0 failed (`r58/pytest_shared.log`; the card tests skip on this host) |
| Not changed | the enumerator and the selection; no production change outside the Gemini-II source route and its context |

## Round 59 (2026-11-05): the repetitions update's graph check runs on the public route (X-27 correction)

Working tree on top of `c39af38`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r59/`.

| Check | Result |
|---|---|
| The reviewer's reproducer (`reviewer-r58/graph_update.py`) after the correction (`r59/reviewer_r58_reproducer_after.log`) | `repetitions = 19` with the injected three-launch graph raises "the cost graph prices 3 launch(es) but the candidate declares 2" from the shared check; nothing is installed |
| `tests/pim/test_shared_semantics.py::test_repetitions_update_runs_the_shared_graph_check_before_installing` (21 Gemini-II tests passed) | the inconsistent graph is refused with the previous plan object, both graph references (`execution_graph` and `plan.execution_graph`), the repetition count (256) and the estimate retained and the artifact packing; a valid update invokes the check (spied: launch counts 3 then 2), installs the consistent graph and the new count in every recipe |
| Correction of round 58's claim | the round 58 setter looked up the layout under a nested attribute the attached codegen artifact does not have, so its graph check never ran on the public route; the claim in round 58's entry that it ran before installation was false for `c39af38` |
| Suites at this revision | `tests/pim/test_shared_semantics.py` + `tests/pim/test_apu_g2_hardware.py` + `tests/perf/test_compile_interface.py` with the native AiM executor: 160 passed, 14 skipped, 0 failed (`r59/pytest_shared.log`) |

## Round 60 (2026-11-05): integration candidate `cd744bb` validated from installed artifacts (X-25 completed, X-27 and its correction)

Candidate `cd744bb` (clean tree: `54cbb08` plus `c39af38` and `cd744bb`).
Evidence under `/work/shared/users/phd/nz264/tenon-release-evidence/r59/`
(`candidate_chain.sh` locally, `g2_remote_chain.sh` on zhang-x4).

| Check | Result |
|---|---|
| Local sdist and wheel (`build_wheel.sh`, `pkg/`) | sdist `61b249b2...` from `git archive` of `cd744bb`; wheel `83fd95dd...` built with the shared LLVM 22 clang; clean venv (numpy 2.5.4) |
| Installed-artifact probes (`installed.log`, the round 8 probe scripts, no PYTHONPATH, safe path, cwd outside the checkout) | compile and virtual run from the wheel (45550 virtual cycles), no module loaded from the checkout; Samsung simulator probe 26807 cycles with max abs error 0.0022 against the reference; AiM timing-only 222 cycles through the native executor, outputs untouched; UPMEM probe fails on the host's full Docker data root (containerd ENOSPC, unchanged host condition) |
| Installed-wheel test subsets (`installed-pytest.log`, `installed-cent-aim.log`) | paper golden AiM default rows 3 passed 11 skipped, AttAcc virtual 44 passed 1 skipped, trace check 17, compose 4, resource budget 3, host compute 54, AiM profile calibration 5, AiM cost routes 6, AiM program 16, Samsung supplied declarations 11, CENT AiM workloads (default rows) 6: all passed (the chain's combined selection with a marker expression ran 9 tests, the per-file reruns above cover the rest) |
| Installed-wheel `tests/pim/test_shared_semantics.py` (`installed.log`) | 137 passed, 1 failed: `test_local_accumulator_gemm_executes_on_apu_v1_functional_backend`, the recorded NumPy 2.4+ functional-route failure V1-19 (numpy 2.5.4 in the venv); the Gemini-II sections q and r (recipe binding, repetitions update including the round 59 graph-check regression) pass from the wheel |
| AiM rows from the installed wheel (`aim_rows_provenance.json`, `aim_rows_provenance.log`, `golden-aim-paper-full.log`) | the fourteen rows' cycles, trace digests, line and opcode counts and lowering manifests equal the `54cbb08` run of round 57 row by row (10 exact, the 4 RoPE rows +32, X-19); `-m paper_full` 11 passed |
| Samsung rows from the installed wheel (`golden-samsung-default.log`, PIMSimulator) | 3 passed: the vendor-equivalent program at the paper's values (4435 / 8041 / 8247), the default selection 4108 / 7057 / 7255 (`split_rows+crf_shared`), the same cycles as at `54cbb08` |
| The twelve AttAcc results from the installed wheel (`run_attacc_installed.sh`, `attacc_{qkv,proj}_b{1,4}.json` with fixed-form baselines, `attacc_score_b{1,4}.json`, `attacc_e2e.json`, `compare_attacc.py`) | every selection, cycle count, per-command trace check and composed decode value equals the round 57 (`54cbb08`) result; the only differing fields are the evidence paths, the executor command lines, the config digests (the YAML embeds the temporary trace path; 504 of 504 configs equal up to that directory name) and the simulator wall times; decode: Tenon full 6.50 / 6.95 / 7.40 / 3.09 / 3.48 / 3.87x, base 1.47 / 1.56 / 1.66 / 1.34 / 1.50 / 1.65x. The chain's first AttAcc attempt failed before any run: my sed-adapted script named a benchmarks copy (`r59/bench`) that did not exist (`aborted/run_attacc_installed_badpath.log`); rerun with the copy in place |
| zhang-x4 wheel from the same sdist (`g2_build_wheel.sh`, `g2/`) | wheel `65604d75...`, clean venv; card `apu-00` Available |
| Gemini-II card tests from the installed wheel (`g2-card-tests.log`) | 15 passed in 125 s (no kept temporaries; the host's `/tmp` at 76%) |
| Gemini-II paper rows at the candidate: one pre-registered validation session (`g2_candidate_session.sh`, `g2_candidate_rows.json`, `g2_candidate_summary.{py,json}`, `g2-candidate-session.log`) | run because `benchmarks/apu_g2/paper_rows.py` assigns the repetition count through the public setter X-27 changed. 20 rounds, GTML interleaved, exit 0, every output exact; plans, layout and program fingerprints and packing attestations identical to the `54cbb08` sessions. Tenon medians vs paper: gemm 1732.375 / 1733.225 (0.9995), batched_gemm 1739.675 / 1739.15 (1.0003, paper inside the IQR 1721.75 to 1748.18), gesummv 1837.225 / 1860.075 (0.9877), bicg 2537.625 / 2548.825 (0.9956), mvt 2767.225 / 2766.625 (1.0002, paper inside the IQR 2754.77 to 2776.47); GTML medians vs the paper's column: gemm 1914.25 / 1903.85 (+0.55%), batched_gemm 1913.45 / 1906.4 (+0.37%), gesummv 3099.75 / 3090.975 (+0.28%), bicg 2727.125 / 2726.5 (+0.02%), mvt 3089.475 / 3089.65 (-0.01%). Under the pooled-session rule at this revision (one session): gemm, gesummv and bicg REPRODUCED; batched_gemm OPEN (nominal, X-26 unchanged); mvt OPEN (nominal) at `cd744bb` after REPRODUCED at `54cbb08` over four sessions. No further session was added at this revision; a confirmation campaign at `cd744bb` would need its own pre-registration |
| Gemini-I pre-registered campaign (X-22 / X-23) | not run: the host's shared `/tmp` is still at 100% (consumers not ours); the round 56 watcher, which targeted the `54cbb08` wheel and launched no complete session, was stopped and replaced by `r59/campaign_watcher.sh` / `r59/apu_v1_campaign.sh` from the `cd744bb` wheel (three sessions, same protocol and acceptance rule) |
| UPMEM at the candidate | not executed: the host's Docker data root is full (containerd ENOSPC); the sixteen rows stay at `0e48d81` |
| Required G10 set | historical coverage (each row at the latest revision the inventory records): 46 REPRODUCED, 9 OPEN (X-19 x4, X-22 / X-23 x3, X-26, mvt at `cd744bb`), 0 UNVERIFIED. Executed at `cd744bb` from installed artifacts: the fourteen AiM rows, the three Samsung rows, the twelve AttAcc results, the five Gemini-II rows; not: the sixteen UPMEM rows and the five Gemini-I rows |

## Round 61 (2026-11-05): the APU v1 grouped route of a dataflow region on the candidate contract (X-28)

Working tree on top of `59e3199` (the candidate `cd744bb` plus this
repair); evidence under `/work/shared/users/phd/nz264/tenon-release-evidence/r60/`.

| Check | Result |
|---|---|
| The reviewer's reproducer (`reviewer-r60/grouped_source.py`) against the repaired checkout (`r60/reviewer_r60_reproducer_after.log`; the same environment as the reviewer's README, `PYTHONPATH` the checkout) | compiles through public `allo.compile(..., backend='virtual', reduction_order='reassociate')`: `COMPILED 185295`, exit 0 (the `cd744bb` wheel fails with "declares no effects", `reviewer-r60/grouped_source.log`) |
| What the route emits and declares for that source (`r60/probe_grouped.py`, `r60/probe_grouped2.py` with the requirement lifted in-process to observe the old emission; `r60/probe_x28.py`, `r60/probe_x28.log` after the repair) | one SPMD body: `mat` into vr0, `vec` into vr1, the output image into vr2, `gvml_mul_f16`, `gvml_add_subgrps_f16_grp(... GVML_P2_8K ...)`, vr2 stored back; four work items of one kernel group, each its own bucket; the declaration per work item: coverage `mat`, `vec` -> `dst`, layouts vr0 / vr1 `group_lanes:8192` and vr2 `group_head:8192`, transfers host_pack / l4 / vmrs / vr and vr2 / vmrs / l4 / host_gather, one `gvml_grouped_mac` launch over instances (0, 1) and reduction (0, 8) with accumulator zero, covers `('grouped_f16', 8192, 4, 4, 1, 'virtual_only')` |
| Fault injection into the emission (a group-sum tree over half the declared group; `test_apu_v1_grouped_emission_is_checked_against_the_declaration`, `probe_x28.log` FAULT) | materialization refuses every candidate: "realization differs from the candidate contract: APU v1 grouped realization of work item (0,) (candidate 'grouped_f16') reduces groups of [4096] lanes; the candidate declares 8192" (a first version of the record captured the group size before the reduce emitter changed it and let the fault through; the record now comes from the reduce emitter's own enum) |
| An edited supplied declaration (`mat` moved to vr5) and a handle-only supplied placement (`test_apu_v1_grouped_supplied_declaration_must_be_the_derivation`) | the edit is refused by the contract before costing ("declares 'mat' in storage 'vr5' ... the derivation ... gives 'vr0'"); the handle-only placement is re-declared from the facts and compiles to the same program and manifest |
| A renamed program and a nearby supported program (`test_apu_v1_grouped_renamed_program_gets_the_same_decision`, `test_apu_v1_grouped_gemm_declares_the_runtime_packing_and_batches`) | the renamed region (arguments and work parameters respelled) gets the same placement, launch, storages, packings, command stream and estimate; a 16 x 32 x 8 FP16 GEMM region whose four work items own row blocks declares covers `('grouped_f16', 8192, 4, 4, 8, 'gemm')`, instances (0, 32), reduction (0, 32), `n_out_tiles` 8 and a device body looping over 8 batches (the old heuristic gave `ceil(8 / 4) = 2` batches while the runtime packed 8: the device would have computed a quarter of the outputs) |
| The runtime's packing and ABI binding without a board (`test_apu_v1_grouped_runtime_binds_the_callers_arrays_by_identity`) | `_apu_v1_prepare_io` binds `A` / `B` / `C` to the caller's `p` / `q` / `r_out` by retained identity (the old code matched spellings and would have found no operands), packs one dot product per 8192-lane group with the rows split over four APUCs (core 1, batch 0, group 2 holds row 4 and column 2, the rest of the group zero), returns the output image spec under the caller's name; arrays passed under the work body's names are refused by name; the GEMV region is refused with the packing reason ("the grouped runtime packs a dense FP16 GEMM ... got mat[4, 8] x vec[8] -> dst[4]") |
| The explicit restrictions (`test_apu_v1_grouped_restrictions_are_explicit`) | strict: `ReductionOrderError` naming `gvml_add_subgrps`; carried destination: `NotImplementedError` with `accumulator_init='carried'`; an elementwise work body: `NotImplementedError` "declares no program for the operations ['ADD']" |
| Suites at this revision | `tests/pim/test_shared_semantics.py` + `tests/perf/test_compile_interface.py` + `tests/pim/test_apu_g2_hardware.py`: 166 passed, 14 skipped, 1 failed (`r60/pytest_shared.log`): the failure is `test_aim_selected_program_runs_timing_only_under_the_declared_contract` through Docker (the host's data root, containerd ENOSPC); with the native executor it passes (`r60/pytest_aim_timing_native.log`, 1 passed). The APU v1 subsets: `-k "apu_v1 and not grouped"` 9 passed (`pytest_shared_apu_v1.log`); the seven new grouped-route tests pass; `tests/perf/test_apu_v1_*`, `tests/perf/test_run_result_contract.py`, `tests/pim/apu_v1` (not board): 25 passed (`pytest_apu_v1_perf2.log`). `tests/pim/test_gemv_all_targets.py::test_gemv_apu_v1` (vector route, board) ran on the board in a first background selection and failed in the profile read: `ledag-ssh` cannot write its scratch to the host's full `/tmp` ("the server is overloaded or there is an error"), the same condition that blocks the Gemini-I campaign; not a defect of this change (`pytest_apu_v1_perf.log`) |
| Not changed | the vector plans and the `APUv1Program` records (their own declarations and realization checks); the paper rows (no required row uses the grouped route); the Gemini-II and other backends |
| Observed, not repaired | V1-20: a region with more work items than APUCs (`mapping=[16]`) fails in `apu_v1_cost` ("work coordinate 4 is outside 'apuc' extent 4"); the grouped enumerator accepts the mapping and the cost model reads it as APUC coordinates |

## Round 62 (2026-11-05): the grouped route's executable template proven from the retained accesses (X-29)

Working tree on top of `d907b3b`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r61/` (the reviewer's
under `reviewer-r61/`).

| Check | Result |
|---|---|
| The reviewer's six same-shaped regions (`reviewer-r61/{base,reverse_k,partial_rows,partial_columns,repeated_row,permuted_rows}.py`) against the repaired checkout, both backends (`r61/probe_x29.py`, `probe_x29_after.log`) | `base`: `gemm` on both backends, the proven blocks rows 0-3 / 4-7 / 8-11 / 12-15 x columns 0-7, the values the packed operands imply under the emitted multiply and group sum equal `A @ B`; the five others: `backend=None` refused before scheduling with the failing condition ("operand 'B' is read at [-1 * k + 31, j] along the reduction; the device packs B[k, c]"; "the work items write 64 of the 128 elements of C[16, 8] (...); the device computes and gathers the whole output, so an untouched element would be overwritten"; "the result 'C' is written at [i, j] while the operands are read at row 0 of 'A'"; "... row -1 * i + 15 of 'A'"), `backend='virtual'` compiled with the launch's `covers` ending in `virtual_only`, the reason in `extra["device_packing_reason"]`, and `_apu_v1_prepare_io` refusing with the same reason |
| The reviewer's board-route reproducer (`reviewer-r61/board_compile.py`, `r61/reviewer_board_compile_after.log`) | `allo.compile(region, ..., reduction_order='reassociate')` with the default `backend=None` raises `NotImplementedError` from `require_grouped_device_packing` naming the reversed reduction access (exit 1); before the repair it compiled to the base case's device C with packing that implied 1 where the source gives 32 (`reviewer-r61/board_compile.log`) |
| `tests/pim/test_shared_semantics.py` section u additions (7: the six non-template regions parametrized and the whole-group proof; 14 grouped tests passed) | each non-template region: refused for `backend=None` with its condition and for `backend='functional'` ("no host functional model"), compiled for the estimate as `virtual_only` with the declared instances the work item's own (32, or 16 for the half-domain regions), refused by the runtime; the canonical row-block partition and a column-block partition are proven over all four work items (blocks `(0,4,0,8)...` and `(0,16,0,2)...`), the packed products gathered equal `A @ B`, and the named-candidate path (`layout='grouped_f16'`, no cost) proves the template too |
| Suites at this revision (native AiM executor) | `tests/pim/test_shared_semantics.py` + `tests/perf/test_compile_interface.py` + `tests/pim/test_apu_g2_hardware.py`: 174 passed, 14 skipped, 0 failed (`r61/pytest_shared.log`); `tests/perf/test_apu_v1_*`, `tests/perf/test_run_result_contract.py`, `tests/pim/apu_v1`, `tests/pim/test_gemv_all_targets.py` (not board, not the UPMEM/Samsung simulator rows): 28 passed (`r61/pytest_apu_v1_perf.log`) |
| Not changed | the vector plans and program records; the emitted grouped body (the same device C as before for the template); the paper rows; other backends. No board launched (the host's `/tmp`) |

## Round 63 (2026-11-05): integration candidate `8795e8f` validated from installed artifacts (X-28 and X-29 included)

Candidate `8795e8f` (clean tree: `cd744bb` plus `d907b3b` and `8795e8f`).
Evidence under `/work/shared/users/phd/nz264/tenon-release-evidence/r62/`
(`candidate_chain.sh` locally, `g2_remote_chain.sh` and
`g2_plan_equivalence.sh` on zhang-x4).

| Check | Result |
|---|---|
| Local sdist and wheel (`build_wheel.sh`, `pkg/`) | sdist `b83b32ec...` from `git archive` of `8795e8f` (no dirty entries); wheel `8feb1122...` built with the shared LLVM 22 clang; clean venv (numpy 2.5.4) |
| Installed-artifact probes (`installed.log`, the round 8 probe scripts, no PYTHONPATH, safe path, cwd outside the checkout) | compile and virtual run from the wheel, no module loaded from the checkout; Samsung simulator probe pass; AiM timing-only through the native executor, outputs untouched; UPMEM probe fails on the host's full Docker data root (containerd ENOSPC, unchanged: `/docker` at 100%) |
| Installed-wheel `tests/pim/test_shared_semantics.py` (`installed.log`) | 151 passed, 1 failed: `test_local_accumulator_gemm_executes_on_apu_v1_functional_backend` (V1-19, numpy 2.5.4); the grouped route's section u from the wheel: 14 passed (the reviewer's source, the renamed program, the GEMM packing and batches, the identity binding, the explicit restrictions, emission fault injection, the edited declaration, the six non-template regions, the whole-group proof and the named path) |
| Installed-wheel test subsets (`installed-pytest.log`, `installed-cent-aim.log`) | paper golden AiM default rows 3 passed 11 skipped, AttAcc virtual 44 passed 1 skipped, trace check 17, compose 4, resource budget 3, host compute 54, AiM profile calibration 5, AiM cost routes 6, AiM program 16, Samsung supplied declarations 11, CENT AiM workloads (default rows) 6: all passed |
| AiM rows from the installed wheel (`aim_rows_provenance.json`, `golden-aim-paper-full.log`) | the fourteen rows' cycles, trace digests, line and opcode counts and lowering manifests equal the `54cbb08` (r57) and `cd744bb` (r59) runs row by row (10 exact, the 4 RoPE rows +32, X-19); `-m paper_full` 11 passed |
| Samsung rows from the installed wheel (`golden-samsung-default.log`) | 3 passed: vendor-equivalent 4435 / 8041 / 8247 at the paper's values, defaults 4108 / 7057 / 7255, the same as at `54cbb08` and `cd744bb` |
| The twelve AttAcc results from the installed wheel (`run_attacc_installed.sh`, `attacc_*.json`, `attacc_e2e.json`, `compare_attacc.py`) | every selection, cycle count, per-command trace check and composed decode value equals the r57 and r59 results (only paths, command lines, config digests and wall times differ); decode: Tenon full 6.50 / 6.95 / 7.40 / 3.09 / 3.48 / 3.87x, base 1.47 / 1.56 / 1.66 / 1.34 / 1.50 / 1.65x |
| zhang-x4 wheel from the same sdist (`g2_build_wheel.sh`, `g2/`) and the card tests (`g2-card-tests.log`) | wheel `2c1056ef...`, clean venv; 15 passed in 125 s |
| Gemini-II paper kernels, compile only, from the installed wheel on zhang-x4 (`g2_plan_equivalence.py`, `g2_plan_equivalence.json`, `g2-plan-equivalence.log`) | each of the five kernels compiles to the same plan (`single/k1` / `fused/k1`), the same estimate and the same launch layout and program fingerprints as the `cd744bb` session recorded on the card (`r59/g2_candidate_rows.json`); no session launched at this candidate (the Gemini-II route is unchanged between `cd744bb` and `8795e8f`; a further session would be the median-chasing the review forbids), so the Gemini-II rows keep their `cd744bb` measurements |
| The grouped route on the APU v1 board (`apu_v1_grouped_board.py`, `grouped_board_watcher.sh`) | **not run: explicit verification blocker.** The script compiles three dense FP16 GEMM regions from the installed wheel through public `allo.compile` with `backend=None` (the canonical 16 x 32 x 8 row blocks and its column-block partition, 8 batches per APUC each; a neighbouring 12 x 17 x 5 shape, 4 batches), runs them on the board and compares the outputs exactly with the NumPy oracle; the watcher launches it when the host's shared `/tmp` has 3 GB free (`ledag-ssh` needs scratch there; `/tmp` at 100%, consumers not ours). Until it runs, X-28 / X-29 hardware validation is not complete |
| Gemini-I pre-registered campaign | not run (the same `/tmp` blocker); the scripts re-pointed at the `8795e8f` wheel (`r62/apu_v1_campaign.sh`, `r62/campaign_watcher.sh`); the r59 watcher stopped without having launched a session |
| UPMEM at the candidate | not executed: the host's Docker data root is full; the sixteen rows stay at `0e48d81` |
| Required G10 set | historical coverage: 46 REPRODUCED, 9 OPEN (X-19 x4, X-22 / X-23 x3, X-26, mvt at `cd744bb`), 0 UNVERIFIED. Executed at `8795e8f` from installed artifacts: the fourteen AiM rows, the three Samsung rows, the twelve AttAcc results; Gemini-II at `cd744bb` (plans and fingerprints equal at `8795e8f`); not: the sixteen UPMEM rows and the five Gemini-I rows |

## Round 64 (2026-11-05): the grouped route's batch count validated against the proven packing (X-30)

Working tree on top of `e4ec027` (the candidate `8795e8f` plus this
repair); evidence under `/work/shared/users/phd/nz264/tenon-release-evidence/r63/`
(the reviewer's under `reviewer-r63/`).

| Check | Result |
|---|---|
| The reviewer's reproducer (`reviewer-r63/batch_probe.py`) against the repaired checkout (`r63/reviewer_batch_probe_after.log`) | the intact region: 974475 cycles, eight batches; the supplied one-batch placements without declarations: compilation refused, "APU v1 grouped realization of work item (0,) (candidate 'grouped_f16') loops over 1 batch(es); the candidate declares 8" (before the repair they compiled at 185295 and failed in the runtime's packing); the injected search (a one-batch clone declared by the production deriver beside every intact candidate): the intact candidate selected (974475, eight batches) and its packing accepted |
| `test_apu_v1_grouped_supplied_placement_with_too_few_batches_is_refused` | supplied one-batch placements without declarations are re-declared with the proven eight batches and refused at materialization; supplied with their declaration edited to one batch they are refused by the contract before costing |
| `test_apu_v1_grouped_search_rejects_an_under_batched_candidate` | with the enumerator monkeypatched to offer the one-batch clones first, the compiled program is the intact candidate (`n_out_tiles` 8, the context looping over 8, the declaration's covers 8, the same estimate as the unpatched compile) and the ranking records the clone's rejection ("loops over 1 batch(es); the candidate declares 8") |
| The materialization check's independent leg | besides the declaration, `check_realization` re-derives the kernel group's packing from the stamped facts and requires the body's loop count to be the batches that packing needs ("the packing the runtime performs for this kernel group needs 8") |
| Suites at this revision (native AiM executor) | `tests/pim/test_shared_semantics.py` + `tests/perf/test_compile_interface.py` + `tests/pim/test_apu_g2_hardware.py`: 176 passed, 14 skipped, 0 failed (`r63/pytest_shared.log`); the APU v1 perf and unit sets (not board): 28 passed (`r63/pytest_apu_v1_perf.log`); section u: 16 passed |
| Not changed | the enumerator's own candidates (they already loop over the packing's batches); the runtime guard; the paper rows; other backends. No board launched (the host's `/tmp` still full; both watchers waiting) |

## Round 65 (2026-11-05): the board checker corrected, the candidate `c43061f` validated from installed artifacts, the X-19 search bounded by measurement

Candidate `c43061f` (clean tree: `8795e8f` plus `e4ec027` and `c43061f`).
Evidence under `/work/shared/users/phd/nz264/tenon-release-evidence/r64/`
(the reviewer's under `reviewer-r64/`).

| Check | Result |
|---|---|
| The queued grouped-route board checker (reviewer: `r62/apu_v1_grouped_board.py` exited 0 with three compilation failures or three inexact outputs, `reviewer-r64/driver_{failure,mismatch}_probe.*`) | replaced by `r64/apu_v1_grouped_board.py`: every case's record is kept whatever the outcome; the generated project with the host's packed inputs and raw output image (the runtime's `TENON_APU_V1_KEEP_TMP=1` directory), the inputs, the outputs, the oracle, the fresh profile text and the kernel source are archived under `<out>.evidence/<case>/` with the wheel and revision identity and the binary digest; `verdict` exits 0 only when every case compiled, ran on the device, returned four positive fresh APUC counters whose maximum is the cycle count, a binary digest, exact outputs and the declared batch count; `--self-test` (`r64` run, exit 0) refuses synthetic records with an exception, an inexact output, a missing counter, a non-device run, a missing case and a cycle count that is not the counters' maximum, and accepts complete ones. `r64/grouped_board_watcher.sh`, `r64/apu_v1_campaign.sh` (every session kept, status the count of incomplete sessions) and `r64/campaign_watcher.sh` propagate the child status; the r62 watchers were stopped (they had launched nothing) and the r64 ones run against the `c43061f` wheel |
| Local sdist and wheel (`build_wheel.sh`, `pkg/`) | sdist `8d22f09a...` from `git archive` of `c43061f` (no dirty entries); wheel `dc9463a2...`; clean venv (numpy 2.5.4) |
| Installed-artifact probes (`installed.log`) | compile, virtual, Samsung and native AiM probes pass, no module from the checkout; UPMEM probe fails on the host's full Docker data root (unchanged) |
| Installed-wheel `tests/pim/test_shared_semantics.py` (`installed.log`) | 153 passed, 1 failed (V1-19 under numpy 2.5.4); the grouped route's section u from the wheel: 16 passed (X-28, X-29 and X-30 invariants outside the checkout) |
| Installed-wheel test subsets (`installed-pytest.log`, `installed-cent-aim.log`) | AiM golden default rows 3 passed 11 skipped, AttAcc virtual 44 passed 1 skipped, trace check 17, compose 4, resource budget 3, host compute 54, AiM profile calibration 5, AiM cost routes 6, AiM program 16, Samsung supplied declarations 11, CENT AiM workloads 6: all passed |
| AiM, Samsung and AttAcc rows from the installed wheel (`aim_rows_provenance.json`, `golden-aim-paper-full.log` 11 passed, `golden-samsung-default.log` 3 passed, `attacc_*.json`, `attacc_e2e.json`, `compare_attacc.py`) | the fourteen AiM rows equal r57, r59 and r62 row by row; Samsung 4435 / 8041 / 8247 vendor-equivalent and 4108 / 7057 / 7255 default; every AttAcc selection, cycle count, trace check and composed value equal to r57, r59 and r62 |
| zhang-x4 wheel, card tests, compile-only Gemini-II plan equivalence (`g2/`, `g2-card-tests.log`, `g2_plan_equivalence.json`) | wheel `538d3f39...`; 15 passed in 125 s; the five paper kernels compile to the `cd744bb` session's plans, estimates and launch fingerprints (no session launched) |
| X-19 bounded faithful-program study on the native simulator (`r64/x19/rope_candidates.{py,json}`, `rope_alternatives.py`, `rope_alternatives_{qkvo_rope_l128,full_block_l128}.{json,log}`) | the two RoPE kernel groups enumerate four candidates each (`replica_partitions` 1 / 2 / 4 / 8; four replicas on 32 channels make 8 the widest); the four symmetric choices (one position applied to both groups, the other groups at their default selection) simulated whole: qkvo_rope_l128 130,310 / 129,850 / 129,701 / 129,637 (EWMUL 64 / 64 / 32 / 16 columns), full_block_l128 421,992 / 421,500 / 421,319 / 421,191; the default (partitions 8) is the fastest of those four in both rows and all are above the paper's digits (129,605 and 421,159). The four rows stay OPEN. (Round 65 stated this as the exhausted faithful candidate space: an overclaim, the two groups being independently selectable and only two rows simulated; corrected and completed in round 66.) Observed, not pursued: the whole-program estimates of full_block_l128 (497,680 to 498,176) sit well above its simulated cycles while ranking the candidates in the measured order |
| Pending on the host (unchanged blockers) | the grouped route's board validation and the Gemini-I campaign wait for `/tmp` (100%, consumers not ours; both r64 watchers running); UPMEM rows blocked by the Docker data root. X-28 / X-29 / X-30 hardware validation remains incomplete |
| Required G10 set | historical coverage 46 REPRODUCED, 9 OPEN, 0 UNVERIFIED; executed at `c43061f` from installed artifacts: AiM 14, Samsung 3, AttAcc 12; Gemini-II at `cd744bb` (plans and fingerprints equal at `c43061f`); not: UPMEM 16, Gemini-I 5 |

## Round 66 (2026-11-05): the grouped placement's map must be the executable's (X-31); the X-19 pair grid measured

Working tree on top of `8ab0a85` (the candidate `c43061f` plus this
repair); evidence under `/work/shared/users/phd/nz264/tenon-release-evidence/r65/`
and `r64/x19/` (the reviewer's under `reviewer-r65/`).

| Check | Result |
|---|---|
| The reviewer's reproducer (`reviewer-r65/layout_probe.py`) against the repaired checkout (`r65/reviewer_layout_probe_after.log`) | the interleaved map (group bits on physical lanes 1 and 2, lane bits on 4 to 16384) is refused at its first compile by the contract: "the placement's selected map is invalid for 'P': the executable addresses contiguous groups (physical lane = group x 8192 + lane_in_group); the placement's map sends the lane_in_group bits to physical lanes [4, 8, ...]" (before the repair it compiled to the canonical program's device C and packing, and an injected clone was selected) |
| `test_apu_v1_grouped_non_contiguous_map_is_refused_when_supplied[interleaved,collapsed]` | both maps supplied without a declaration are refused with the contiguous map named; supplied with the canonical declaration copied onto them they are refused as a derivation disagreement; the canonical map (with its movement annotations) re-supplied compiles to the same program |
| `test_apu_v1_grouped_search_rejects_non_contiguous_maps` | with the enumerator monkeypatched to offer both clones of every intact candidate (declared by the production deriver: every packing `invalid:...`), the contract filters them per candidate and the canonical candidate is selected and emitted (the same command stream and estimate as the unpatched compile) |
| The realization check's second line | `check_realization` evaluates `grouped_map_problem` on every declared placement at materialization and refuses a non-contiguous map as a feasibility rejection even when a declaration slipped past the contract |
| X-19 pair grid on the native simulator (`r64/x19/rope_pairs.py`, `rope_pairs_{qkvo_rope_l128,full_block_l128,full_block_l512,full_block_l4096}.{json,log}`) | all 16 pairs of the two RoPE groups' candidates (`replica_partitions` 1 / 2 / 4 / 8 each; the other kernel groups at their default selection) forced and simulated whole for all four X-19 rows: the minimum is the default pair (8 / 8) in every row, 129,637 / 421,191 / 446,835 / 697,348 against the paper's 129,605 / 421,159 / 446,803 / 697,316; the round 65 wording ("the route's candidate space exhausted" from four symmetric choices on two rows) is corrected in place in ISSUES, VALIDATION, REVIEW, STATUS, PAPER_REPRODUCTION and the inventory; the four rows stay OPEN |
| Suites at this revision (native AiM executor) | `tests/pim/test_shared_semantics.py` + `tests/perf/test_compile_interface.py` + `tests/pim/test_apu_g2_hardware.py`: 179 passed, 14 skipped, 0 failed (`r65/pytest_shared.log`); the APU v1 perf and unit sets (not board): 28 passed (`r65/pytest_apu_v1_perf.log`); section u: 19 passed |
| Not changed | the canonical enumerator's layouts and their movement annotations; the paper rows; other backends. No board launched (the host's `/tmp` still full, both r64 watchers waiting; UPMEM blocked by Docker). Not yet in an installed candidate |

## Round 67 (2026-11-05): candidate `f418762` validated from installed artifacts; the UPMEM rows re-executed through a native SDK executor

Candidate `f418762` (clean tree: `c43061f` plus `8ab0a85` and `f418762`).
Evidence under `/work/shared/users/phd/nz264/tenon-release-evidence/r66/`
(the reviewer's under `reviewer-r66/`).

| Check | Result |
|---|---|
| Local sdist and wheel (`build_wheel.sh`, `pkg/`) | sdist `de58449a...` from `git archive` of `f418762` (no dirty entries); wheel `e2b1bae4...`; clean venv (numpy 2.5.4) |
| Installed-artifact probes (`installed.log`) | compile, virtual, Samsung and native AiM probes pass, no module from the checkout; the UPMEM probe fails on the host's full Docker data root (the wheel predates the native executor below) |
| Installed-wheel `tests/pim/test_shared_semantics.py` and the grouped route (`installed.log`) | 154 passed, 1 failed (V1-19); the grouped route's section u from the wheel: 19 passed (X-28 to X-31 outside the checkout); the reviewer's round 65 map probe from the installed wheel: the first supplied interleaved map refused by the contract ("the placement's selected map is invalid for 'P': the executable addresses contiguous groups ...") |
| Installed-wheel test subsets (`installed-pytest.log`, `installed-cent-aim.log`) | all subsets passed (AiM golden default rows 3, AttAcc virtual 44, trace check 17, compose 4, resource budget 3, host compute 54, AiM calibration 5, AiM cost routes 6, AiM program 16, Samsung declarations 11, CENT AiM workloads 6) |
| AiM, Samsung and AttAcc rows from the installed wheel (`aim_rows_provenance.json`, `golden-aim-paper-full.log` 11 passed, `golden-samsung-default.log` 3 passed, `attacc_*.json`, `attacc_e2e.json`, `compare_attacc.py`) | the fourteen AiM rows equal every earlier candidate row by row; Samsung 4435 / 8041 / 8247 vendor-equivalent and 4108 / 7057 / 7255 default; every AttAcc selection, cycle count, trace check and composed decode value equals the four earlier candidates (only paths, command lines, config digests and wall times differ) |
| zhang-x4 wheel, card tests, compile-only Gemini-II plan equivalence (`g2/`, `g2-card-tests.log`, `g2_plan_equivalence.json`) | wheel `d16c6327...`; 15 passed in 124 s; the five paper kernels compile to the `cd744bb` session's plans, estimates and launch fingerprints (no session launched) |
| The native UPMEM SDK executor (this commit; `allo/spmw_simenv.py::upmem_sdk_build_command`, `TENON_UPMEM_SDK_HOME`; `simulators.md`) | the host's Docker data root has been full since 2026-11-04 (every container start fails, `docker save` too), so the DPU build can now run the same SDK 2023.1.0 compiler natively: `unshare -Urm` binds the uPIMulator root at the image's `/root/uPIMulator` path in an unprivileged user and mount namespace and runs the same `benchmark/build.py` with the SDK on `PATH` (identical source paths; the build tree owned by the user). The uPIMulator binary already ran natively. The availability gate names a missing piece; every run record names its executor (`sdk_executor`) and the exact build command; the pinned-campaign SDK manifest check is unchanged and passes (`d45e285c...`). Unit tests `tests/pim/test_upmem_sdk_executor.py` (4: the Docker command unchanged, the native command's bind and environment, the gate's reasons, the gate preferring the native executor) |
| The sixteen UPMEM rows at the checkout (`f418762` plus this executor; `r66/upmem/golden-upmem-default.log`, `golden-upmem-paper-full.log`, `record_probe.log`, `gemv-upmem-native.log`) | `tests/pim/test_paper_golden.py -k upmem`: 3 default rows passed (va, red, hist) and 16 provenance tests; `-m paper_full`: 13 passed in 543 s; every row's pinned cycles equal (the paper's values) with NumPy-exact outputs, the simulator `0ab6af91...` and the SDK build manifest `d45e285c...` as recorded; a run record: `sdk_executor native-sdk:<home>`, `sdk_build pinned-campaign`; `test_gemv_all_targets.py -k upmem` 1 passed. **Checkout evidence**: the installed candidate `f418762` does not contain the executor, so the UPMEM rows are not yet executed at an installed candidate |
| Watchers | the r64 watchers (c43061f) stopped, their logs kept, nothing launched; the r66 watchers target the `f418762` wheel (`r66/grouped_board_watcher.sh`, `r66/campaign_watcher.sh`); `/tmp` still at 100% |
| Suites at this revision (native AiM executor) | `tests/pim/test_shared_semantics.py` + `tests/perf/test_compile_interface.py` + `tests/pim/test_apu_g2_hardware.py` + `tests/pim/test_upmem_sdk_executor.py` + `tests/perf/test_simulator_unavailable.py`: 191 passed, 14 skipped, 0 failed (`r66/pytest_shared_r67.log`) |
| Required G10 set | historical coverage 46 REPRODUCED, 9 OPEN, 0 UNVERIFIED; executed at `f418762` from installed artifacts: AiM 14, Samsung 3, AttAcc 12; UPMEM 16 at the checkout `f418762` plus the executor (native SDK executor), not yet from an installed wheel; Gemini-II at `cd744bb` (plans and fingerprints equal at `f418762`); Gemini-I not executed (the host's `/tmp`) |

## Round 68 (2026-11-05): candidate `ba2f58d` validated from installed artifacts (UPMEM rows from the wheel); U-16 repaired

Candidate `ba2f58d` (clean tree: `f418762` plus the native UPMEM SDK executor).
Evidence under `/work/shared/users/phd/nz264/tenon-release-evidence/r67/`
(the reviewer's under `reviewer-r67/`); the U-16 repair is this commit's
working tree (checkout evidence, `r67/u16/`).

| Check | Result |
|---|---|
| Local sdist and wheel (`build_wheel.sh`, `pkg/`) | sdist `c8d90e77...` from `git archive` of `ba2f58d` (no dirty entries); wheel `bd0b05fc...`; clean venv (numpy 2.5.4) |
| Installed-artifact probes and suites (`installed.log`) | compile, virtual, Samsung and native AiM probes pass, no module from the checkout; the round 8 UPMEM probe, which sets no executor, still fails on Docker (the executor needs `TENON_UPMEM_SDK_HOME`, set by the UPMEM stage below); `tests/pim/test_shared_semantics.py` from the wheel 156 passed, 1 failed (V1-19); the grouped route's section u 19 passed; the reviewer's round 65 map probe refused at its first supplied map |
| Installed-wheel test subsets (`installed-pytest.log`, `installed-cent-aim.log`) | all subsets passed |
| AiM, Samsung and AttAcc rows from the installed wheel | the fourteen AiM rows equal every earlier candidate row by row (`aim_rows_provenance.json`, `golden-aim-paper-full.log` 11 passed); Samsung 4435 / 8041 / 8247 vendor-equivalent and 4108 / 7057 / 7255 default (`golden-samsung-default.log`); every AttAcc selection, cycle count, trace check and composed decode value equals the five earlier candidates (only paths, command lines, config digests and wall times differ) |
| **The sixteen UPMEM rows from the installed wheel** through the native SDK executor (`run_installed_upmem.sh`, `golden-upmem-default.log`, `golden-upmem-paper-full.log`, `gemv-upmem.log`, `upmem_record_probe.log`) | 3 default rows (va, red, hist) and 16 provenance tests passed, the 13 paper_full rows passed in 507 s, the UPMEM GEMV test passed, the GEMV probe exact at 563,207 cycles with the run record naming the executor (native-sdk), the pinned-campaign SDK build and the manifest d45e285c...: every row's pinned cycles (the paper's) equal with NumPy-exact outputs, from the installed wheel of ba2f58d |
| zhang-x4 wheel, card tests, compile-only Gemini-II plan equivalence and plan ranking (`g2/`, `g2-card-tests.log`, `g2_plan_equivalence.json`, `g2_plan_ranking.json`) | wheel `86565fae...`; 15 passed in 124 s; the five paper kernels compile to the `cd744bb` session's plans, estimates and launch fingerprints (no session launched); the ranked plans of every kernel (gemm and batched_gemm `single/k1` 1714 against `single/k2` 2550; gesummv `fused/k1` alone; bicg `fused/k1` 2610 against 3500 / 5220 / 7000; mvt `fused/k1` 2740 against `separate/k1` 5480): no alternative is estimated cheaper than the selected plan, so the two nominal Gemini-II rows (X-26, mvt) have no production selection opportunity in the route's candidate space |
| U-16 (`r67/u16/repro.py`, `native_both.{py,json}`): the hoisted row base `row0 = dpu * ROWS; y[row0 + i] += W[row0 + i, j] * x[j]` | before: refused by `classify_upmem_kernel` ("no UPMEM kernel family matches ops ['MAC']"), the raw index names (`%12`, `%22`, `%32`) differing from the inline form's loop variables while the matcher's resolved coordinates (`1 * i + 96`) were already equal; after the recognizers read the resolved coordinates it classifies as `matrix_vector` with the inline form's geometry, then the portable-C host oracle refused `arith.index_cast` on the frontend's 65-bit `row0 + i` (a cast to `index` is a truncation to the low 64 bits, now accepted as exact); through the native SDK executor both forms compile to the same device source and DPU object (`b96b90c8...`) and simulate to the same 563,207 cycles with exact outputs. Regressions: `tests/pim/test_gemv_all_targets.py::test_gemv_upmem` (the hoisted form's family, geometry, device source and estimate equal the inline form's) and `tests/test_c_backend.py::test_wide_index_cast_of_an_address_temporary_is_exact` |
| The UPMEM suites after the recognizer change (checkout, native executor; `r67/u16/pytest_upmem_units.log`, `pytest_upmem_more.log`, `golden-upmem-*.log`) | generality, irregular and GEMV tests 69 passed, 1 skipped; the matcher, dtype, owner-map, physical and physical-search suites and the executor tests 66 passed, 2 skipped; the sixteen golden rows again exact (13 paper_full, 3 default) and the C backend 16 passed |
| Suites at this revision | `tests/test_c_backend.py` 16 passed; the shared suites unchanged by this commit's files (the UPMEM recognizer and the portable-C rule) |
| Blockers unchanged | the grouped route's board validation and the Gemini-I campaign wait for `/tmp` (100%; the r67 watchers target the `ba2f58d` wheel, the r66 ones stopped having launched nothing); X-28 to X-31 hardware validation incomplete |
| Required G10 set | historical coverage 46 REPRODUCED, 9 OPEN, 0 UNVERIFIED; executed at `ba2f58d` from installed artifacts: AiM 14, Samsung 3, AttAcc 12, UPMEM 16; Gemini-II at `cd744bb` (plans, fingerprints and ranking equal at `ba2f58d`); Gemini-I not executed (the host's `/tmp`) |

## Round 69 (2026-11-05): UPMEM load identity on full resolved coordinates (U-16 completed)

Working tree on top of `63bc166`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r68/` (the reviewer's
under `reviewer-r68/`). Checkout evidence (not yet an installed candidate).

| Check | Result |
|---|---|
| The reviewer's reproducers against the repaired checkout (`reviewer-r68/coordinate_probe.py`, `selection_native.py`; `r68/reviewer_coordinate_probe_after.log`, `reviewer_selection_native_after.log`) | `base_select` and `hoisted_select` (`row0 = 64; if A[row0 + i] > 0: O[c] = A[row0 + i]`) both compile to the selection template at the same estimate (before: the hoisted one refused "selection guard tests a value other than the copied one"); `shifted_guard` (guard over `A[i]`, copy of `A[64 + i]`) is refused before selection with that message (before: accepted as the selection template and failed in packing, "512 payload bytes do not fit the 384/384-byte region"); `base_select` executes on the native simulator at 3,946 cycles with the exact count and output |
| What changed (`allo/spmw_upmem.py`, `allo/spmw_match_engine.py`) | `_load_is` compares the full resolved coordinates of the guard's load and of the operand binding (affine maps and constants; `_access_coords`: an operand's `coords`, a retained load term's traced indices composed with its access map through the engine's own `_coords_of_terms`), refusing when either side is unresolved unless both are and the raw indices agree; `_coordinate_vars` remains a geometry projection (it drops constants) and now also projects load terms; the engine's `_term_eq` treats two loads of one memref with equal resolved coordinates and memory version as the same element (the histogram guard recomputing a hoisted index). No new parser: the retained analysis is reused |
| `tests/pim/test_upmem_coordinates.py` (4) | the inline and hoisted selection spellings classify as `selection` with equal geometry, predicate constants, device source and estimate; the shifted-guard selection is refused before selection; the inline and hoisted histogram spellings (`(X[row0 + i] * 256) >> 24` under the range guard) classify as `histogram` with the same device source; both selection spellings execute on the simulator (through the native SDK executor) selecting the positive elements of the second half exactly, with equal cycles |
| Suites at this revision | `tests/pim/test_shared_semantics.py` + `tests/perf/test_compile_interface.py` + `tests/pim/test_apu_g2_hardware.py` + `tests/pim/test_match_coverage.py` + `tests/pim/test_g6_semantics.py` + `tests/test_c_backend.py`: 267 passed, 14 skipped, 0 failed (`r68/pytest_shared.log`, native AiM executor); the UPMEM generality, irregular, matcher, dtype, owner-map, physical, physical-search, executor and GEMV suites: 108 passed, 3 skipped (`r68/pytest_upmem.log`); the sixteen UPMEM golden rows again exact through the native SDK executor (13 paper_full, 3 default; `r68/golden-upmem-*.log`) |
| Round 68 claim corrected | ISSUES U-16 no longer claims that equivalent spellings classified alike in every family after round 68: the operand geometry did, the selection and histogram guard / copy loads did not until this round |
| Blockers unchanged | `/tmp` full (the grouped board validation and the Gemini-I campaign queued, the r67 watchers targeting `ba2f58d`); nine required rows OPEN |

## Round 70 (2026-11-05): the grouped APU v1 route validated on the board from the installed wheel of `472bec9`; the Gemini-I campaign still held by the host's `/tmp`

Candidate `472bec9` (clean tree: `ba2f58d` plus `63bc166` and `472bec9`).
Evidence under `/work/shared/users/phd/nz264/tenon-release-evidence/r69/`.

| Check | Result |
|---|---|
| The temporary-space dependency | `/tmp` (15 GB, shared) was at 100% with 356 KB free. Visible owners account for about 1.2 GB (mine 529 MB, root 610 MB); the rest is held by directories I cannot read and by deleted-but-open files of the user's own long-lived interactive shells (22 of them, 8 MB each). I removed only my own scratch (a stale PyInstaller extraction of the vendor tool, an old session cache, empty temporary directories, compiler scratch), leaving the round 1-9 log copies and probe scratch in place, which gave about 50 MB; the vendor tool ignores `TMPDIR` for its session log, so redirection does not help. The runtime's own drain of the device log then succeeded (5 MB read in 7 s, scratch removed). Other users consumed about 37 MB during the board stage below, which left 8 MB and stopped the campaign: the dependency is external and the campaign stays queued behind a watcher with an attainable threshold (60 MB) and a per-session guard (25 MB), every session kept |
| Local sdist and wheel (`build_wheel.sh`, `pkg/`) | sdist `ecaa40e9...` from `git archive` of `472bec9` (no dirty entries); wheel `6dcbfff0...`; clean venv |
| **The grouped APU v1 route on the board** from the installed wheel (`apu_v1_grouped_board.py`, `apu_v1_grouped_board.{log,json}`, `apu_v1_grouped_board.json.evidence/<case>/`, `board_then_campaign.log`) | three dense FP16 GEMM regions through public `allo.compile` with `backend=None`, each built with the ARC toolchain, launched on the board under the board lock with the device log drained before and after, and compared exactly with the NumPy oracle: `row_blocks_16x32x8` (8 batches per APUC) 938,340 cycles (APUCs 933,955 / 938,156 / 938,340 / 935,095), `column_blocks_16x32x8` (8 batches) 934,668 cycles (929,460 / 930,476 / 934,668 / 933,275), `row_blocks_12x17x5` (4 batches) 489,454 cycles (485,907 / 485,557 / 489,454 / 487,586); every output exact (max abs error 0.0), four fresh counters per launch whose maximum is the cycle count, the declared covers `('grouped_f16', 8192, 4, 4, 8 | 4, 'gemm')`, binaries `642e846c...`, `bc5507f4...`, `6c56acfb...`; the checker's verdict PASS, exit 0. Archived per case (247 MB): the generated project with the host's packed inputs and raw output image, the inputs, outputs and oracle, the fresh profile text, the kernel body and source, beside the wheel digest and revision. Estimates 974,475 / 974,475 / 523,515 against the measured values (3.9% / 4.3% / 7.0% above), recorded, not a paper row. **X-28 to X-31 hardware validation complete at the installed candidate `472bec9`** |
| Installed grouped-route and U-16 regressions from the wheel (`installed-grouped-u16.log`) | section u (19), `test_upmem_coordinates.py` (4), the UPMEM GEMV test with the hoisted form (native SDK executor) and the C backend tests: 40 passed |
| The pre-registered Gemini-I campaign | not started: the runner's guard refused it at 8 MB free (`board_then_campaign.log`: "GUARD: /tmp below 25 MB, campaign not started"); queued behind `r69/campaign_watcher.sh` (60 MB) with the per-session guard in `r69/apu_v1_campaign.sh`; the r67 watchers stopped, logs kept, nothing launched. X-22 / X-23 stay OPEN |
| P-24 recorded | `task_lint.sh` exits 255 at the upstream-relative license-header check for `benchmarks/attacc/scripts/build_ramulator_attacc.sh` (reviewer, rounds 64 and 69); P-23's header-stage closure qualified; deferred behind G9/G10 |

## Round 71 (2026-11-05): replicated AiM elementwise grids realized once over the device (X-32); the X-19 rows below the paper with the complete program

Checkout after `597c793` (the round 71 change, committed in this round);
evidence under `/work/shared/users/phd/nz264/tenon-release-evidence/r69/` (the
X-19 study files) and `r71/`.

| Check | Result |
|---|---|
| The mechanism (`allo/spmw_aim.py`, `allo/spmw_knobs.py`) | a replicated elementwise kernel group whose copies times `replica_partitions` would exceed the 32 channels is realized once over the device (`copies = channels // partitions`); the knob domain of such a MUL group gains the full channel count; a reduced-copy producer whose device reader has another layout is an `InfeasibleSchedule` per candidate; the realization check compares the realized (replicas, channels per replica) with the single-copy expectation. The rule applies to elementwise groups only (a replicated MAC group keeps its replicas) |
| Selection and the selected trace (`r69/x19_selection.py`, `x19_selection_qkvo_rope_l128.json`) | the six elementwise groups of the decode block select 32 partitions by estimated cost; the selected `qkvo_rope_l128` program estimates 125,536 and simulates 123,448 natively; the per-item form (forced `replica_partitions` 8) simulates 129,637 as before; the estimates order the two forms as the simulator does |
| Every CENT row re-simulated (`r69/aim_rows_provenance_final.json`, `.log`; the earlier `aim_rows_provenance_single_copy.json` identical row by row) | norm_residual 6,932 (paper 11,750), qkvo_rope 123,448 (129,605), attention 11,010 / 18,126 / 95,090 (exact), softmax_pim 1,686 / 6,313 / 49,545 (6,307 / 24,835 / 198,416), ffn_fc 250,433 (exact), ffn_activation 3,295 (11,775), ffn_complete 253,789 (262,269), full_block 397,080 / 408,823 / 529,019 (421,159 / 446,803 / 697,316); every row at or below the paper and below CENT |
| The unchanged-cycle traces (`r71/row_shift_check.{py,json,log}`) | attention_l128/l512/l4096 and ffn_fc_l128: adding 3 to the row argument of every W MEM, WR_ABK and MAC_ABK line of the new trace reproduces the earlier pin byte for byte (four of four; the scores region shrinks from 4 rows to 1 and the K/V cache and weight regions move down by three rows; `attention_l128` region table: k_cache row 8, v_cache 136) |
| Neighbouring workload (`r69/x19_neighbor.{py,json}`) | the D = 2048, H = 16, F = 5504 decode block: single-copy realization 32,248 against the per-item 35,341 cycles, the same structure |
| Golden rows (`r71/pytest_golden_aim_default.log`, `pytest_golden_aim_paper_full.log`) | 3 passed (default), 11 passed (`paper_full`): the paper values stay the baseline, `AIM_SELECTED_PROGRAM_CYCLES` pins the ten rows whose selected program is not at the paper's digits (all below) |
| CENT workload suite (`r71/pytest_cent_aim_default.log`, `pytest_cent_aim_paper_full.log`) | 8 passed, 13 skipped (default); 13 passed (`paper_full`): the fourteen trace pins re-pinned with provenance (`benchmarks/cent_aim/reference_traces.py`), the vendor inventory equal to CENT's except host writes / reads and EWMUL / EWADD, which must equal the lowered records' counts and be below the vendor's; the three new regressions (single copy over the device, reduced-copy producer refused for a differently laid-out reader, the neighbouring RoPE shape); the width test's single-copy (4 / 2 columns) and forced per-item (16 / 8) forms; the benchmark's shape-derived decode layout sized for the single copy |
| AiM-related unit suites (`r71/pytest_aim_related.log`) | first batch (`pytest_aim_related.log`, fail-fast): 1 failed, 54 passed, 1 skipped, 1 deselected: the failure is `test_g6_decisions.py::test_samsung_host_fill_cost_flip_reaches_execution` (expects the `bank_row` decision, gets `split_rows`), which fails identically at the committed `597c793` in a separate worktree (PIMSimulator configured through `TENON_SIMULATORS_DIR`): pre-existing, Samsung, not touched in this round, recorded; second batch (`pytest_aim_related_2.log`, the remaining files without fail-fast, `test_shared_semantics.py` included): 6 failed, 346 passed, 1 deselected, 162 warnings: the six failures are five `test_g6_semantics.py::test_upmem_*` tests that build the DPU program through Docker (containerd ENOSPC on the host's full `/var`; the run did not set `TENON_UPMEM_SDK_HOME`) and `test_gemv_all_targets.py::test_gemv_apu_v1` (`ledag-ssh exited with 255 while reading the device log`, the host's `/tmp`); the same six fail at `597c793` in the worktree (`pytest_aim_related_2_HEAD_baseline.log`, 6 failed, 189 passed, 1 deselected, 135 warnings, `test_shared_semantics.py` excluded there), so none is introduced by this round; every AiM, shared-semantics, matcher, layout-selection, host-compute, resource-budget and work-grid test passes |
| Not done | the installed-candidate re-execution of the fourteen rows (next integration candidate); the Gemini-I campaign still queued behind the `/tmp` watcher (X-22 / X-23 OPEN); X-26 and mvt unchanged |

## Round 72 (2026-11-05): device-resident values keep their writer's distribution (X-32, the reviewer's round 71 probes)

Checkout after `4627399` (committed in this round); evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r72/`.

| Check | Result |
|---|---|
| The reviewer's reproducer (`reviewer-r71/fanout.py`, re-run as `r72/fanout_rerun.py` writing into `r72/`; `fanout_rerun.log`) | default accepted (producer, both readers and the relocations one copy over 32 channels, 16 elements per bank partition); `fanout` (second reader forced to four copies over 8 channels) refused: `AiM second: reads t from the device as 4 copies over 8 channels each, bank stride 4, but launch producer left it as 1 copy over 32 channels each ... needs readers of the same layout`; `inverse` (producer forced to four copies) refused at the first reader with the mirrored message; both `InfeasibleSchedule` (a per-candidate feasibility rejection) |
| `tests/pim/test_aim_value_layouts.py` (`pytest_value_layouts.log`) | 6 passed: the default's single distribution; the fan-out and the inverse refused; the per-item alternative for every group accepted (four copies over 8 channels, 64 elements per partition, everywhere); a host gather and re-scatter of `t` between the readers ends the producer's live range (the second reader's four-copy form accepted, the re-scatter laid out for it, the same choice refused without the reupload); two readers in different slots both relocated before the first reader |
| Golden AiM rows (`pytest_golden_aim_default.log`, `pytest_golden_aim_paper_full.log`) | 3 passed, 11 passed: every pinned selected program unchanged |
| CENT workload suite (`pytest_cent_aim_default.log`, `pytest_cent_aim_paper_full.log`) | 8 passed, 13 skipped; 13 passed: trace pins, vendor inventory, the round 71 regressions (including the reduced-copy refusal, now produced by the general check) unchanged |
| AiM-related unit suites (`pytest_aim_related_1.log`, `pytest_aim_related_2.log`) | first batch (the AiM unit suites, the decision tests, host compute and transfer dispatch, layout selection, match coverage, reduction policy, resource budget, work grid, verify): 3 failed, 233 passed, 1 skipped, 217 warnings; the three failures are the Samsung decision tests `test_g6_decisions.py::test_samsung_{host_fill_cost_flip_reaches_execution,fiber_term_removal_keeps_bank_row_by_construction,host_burst_cost_shows_the_documented_list_scheduling_anomaly}` (`bank_row` expected, `split_rows` selected), which fail identically at the committed `4627399` in a separate worktree (`pytest_g6_samsung_HEAD_baseline.log`: 3 failed, 9 passed, 1 skipped, 11 deselected): pre-existing, Samsung, not touched, recorded; `test_g6_decisions.py::test_aim_elementwise_grid_simulates_every_work_item` passes; second batch (`test_shared_semantics.py`, `test_g6_semantics.py`, `test_gemv_all_targets.py` with the native UPMEM SDK executor configured, the APU v1 board test deselected): 194 passed, 1 deselected, 53 warnings |
| Integration candidate `2096239` (`build_wheel.{sh,log}`, `pkg/`) | clean tree (0 dirty entries); sdist `974098a9...` from `git archive`; wheel `803d1ea4...`; clean venv; `allo` imported from the venv's site-packages |
| The fourteen AiM rows and the suites from the installed wheel (`run_installed.sh`, `installed_run.log`, `aim_rows_provenance.json`, tests copied out of the checkout, `PYTHONSAFEPATH`) | golden rows 3 passed (default) and 11 passed (`paper_full`) with every printed selected-program value equal to the pins (6,932 / 123,448 / 1,686 / 6,313 / 49,545 / 3,295 / 253,789 / 397,080 / 408,823 / 529,019; attention and ffn_fc exact); the CENT suite 8 passed, 13 skipped and 13 passed; `test_aim_value_layouts.py` 6 passed; the reviewer's fan-out probe from the wheel: default accepted, `fanout` and `inverse` refused with the same messages; the fourteen-row provenance from the wheel identical in cycles and trace digest to the round 71 checkout run (`r69/aim_rows_provenance_final.json`); the inventory's AiM rows record the candidate |
| Not done | the Gemini-I campaign still queued behind the `/tmp` watcher (X-22 / X-23 OPEN); X-26 and mvt unchanged; the Samsung, UPMEM, AttAcc and Gemini-II rows last re-executed from earlier candidates (`ba2f58d`, `cd744bb`) |

## Round 73 (2026-11-05): physical liveness of bank slots (X-32 reopened by the reviewer's witness, repaired)

Checkout after `a35d42b`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r73/`.  The round 72
rows above that call X-32 closed are superseded by this section.

| Check | Result |
|---|---|
| The reviewer's witness (`reviewer-r72/slot_liveness.py`, re-run as `r73/slot_liveness_rerun.py` writing into `r73/`; `slot_liveness_rerun.log`) | at the round 72 tree: `u = 30`, `v = 35` (the second reader multiplied `d` by `c`); at the repaired checkout: `u = 30`, `v = 42`, the source semantics; three row regions, one per group |
| The round 71 fan-out probe re-run (`r73/fanout_rerun.log`) | default accepted; `fanout` and `inverse` refused with the round 72 messages |
| `tests/pim/test_aim_value_layouts.py` (`pytest_value_layouts.log`) | 11 passed: the independent address/value witness (one cell per row and bank slot, the global buffer, constants a = 2, b = 3, c = 5, d = 7) on the two-reader-slot composition (`v = 42`), the default fan-out, the per-item alternative, the reupload composition (the re-uploaded value carried through the host) and a neighbouring chain (`v = u * t = 180`); the production check refusing the one-row allocation when forced (`physical liveness`); the four round 72 refusals |
| CENT rows re-simulated (`aim_rows_provenance_checkout.json`, `.log`) | attention_l128/l512/l4096 and ffn_fc_l128 at 11,010 / 18,126 / 95,090 / 250,433 with their typed-route trace digests again (`29f3190a...`, `112c7bad...`, `103edc83...`, `833f14ba...`); norm_residual 7,052 (paper 11,750), qkvo_rope 123,448 (129,605), softmax_pim 1,686 / 6,313 / 49,545 (6,307 / 24,835 / 198,416), ffn_activation 3,672 (11,775), ffn_complete 254,166 (262,269), full_block 397,577 / 409,320 / 529,516 (421,159 / 446,803 / 697,316); against round 71: +120 cycles where the RMSNorm chain crosses two regions, +377 where the FFN chain does, +497 for the full block; every row at or below the paper |
| Golden rows and the CENT suite (`pytest_golden_aim_{default,paper_full}.log`, `pytest_cent_aim_{default,paper_full}.log`) | golden rows 3 passed (default) and 11 passed (`paper_full`) at the re-pinned selected programs; the CENT suite 8 passed, 13 skipped and 13 passed (the fourteen trace pins, the vendor inventory, the round 71 regressions with the RoPE command rows 6 and 7, the probe test's QK row 11 again, the benchmark's shape-derived layout one region per group) |
| AiM-related unit suites (`pytest_aim_related_1.log`, `pytest_aim_related_2.log`) | first batch (the AiM unit suites, the decision tests without the three pre-existing Samsung failures recorded in round 72, host compute and transfer dispatch, layout selection, match coverage, reduction policy, resource budget, work grid, verify): 233 passed, 1 skipped, 3 deselected, 217 warnings; second batch (`test_shared_semantics.py`, `test_g6_semantics.py`, `test_gemv_all_targets.py` with the native UPMEM SDK executor, the APU v1 board test deselected): 194 passed, 1 deselected, 53 warnings |
| Integration candidate `26fd884` (`build_wheel.{sh,log}`, `pkg/`) | clean tree (0 dirty entries); sdist from `git archive`; wheel `4bf38997...`; clean venv; `allo` imported from the venv's site-packages |
| The fourteen AiM rows, the suites and both reviewer probes from the installed wheel (`run_installed.sh`, `installed_run.log`, `aim_rows_provenance.json`, tests copied out of the checkout, `PYTHONSAFEPATH`) | golden rows 3 passed (default) and 11 passed (`paper_full`) at the re-pinned values; the CENT suite 8 passed, 13 skipped and 13 passed; `test_aim_value_layouts.py` 11 passed; the round 71 fan-out probe: default accepted, `fanout` and `inverse` refused; the round 72 slot-liveness witness: `u = 30`, `v = 42`; the fourteen-row provenance identical in cycles and trace digests to the checkout run; the inventory's AiM rows record the candidate |
| Not done | the Gemini-I campaign still queued behind the `/tmp` watcher (X-22 / X-23 OPEN); X-26 and mvt unchanged; the Samsung, UPMEM, AttAcc and Gemini-II rows last re-executed from earlier candidates (`ba2f58d`, `cd744bb`) |

## Round 74 (2026-11-06): multi-row relocations one row chunk at a time (X-32 reopened by the reviewer's multi-row witness, repaired)

Checkout after `f79e646`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r74/`.  The round 73
rows above that call X-32 repaired hold for vectors of one row per
partition only; this section supersedes them.

| Check | Result |
|---|---|
| The reviewer's witness (`reviewer-r73/multirow.py`, re-run as `r74/multirow_rerun.py` writing into `r74/`; `multirow_rerun.log`, `multirow.trace`) | at the round 73 tree (the reviewer, from the installed wheel of `26fd884`): `u = (60, 60)`, `v = (84, 84)`; at the repaired checkout: `u = (30, 60)`, `v = (42, 84)`, the source semantics; the copy commands alternate `COPY_BKGB` of source row r and `COPY_GBBK` of destination row r (32 commands, the same count as before) |
| `tests/pim/test_aim_value_layouts.py` (`pytest_value_layouts.log`) | 15 passed: the command-level witness over every row of channel 0's first bank group with values differing from row to row; 147,456-element programs (one full and one partial row per partition in the single-copy layout, four full rows and a partial fifth in the per-item layout): the two-reader-slot composition in both layouts, the chain, the copy commands' chunk order and the partial row's smaller column count; a whole-vector copy record refused by the production check when forced (`global buffer holds row chunk`); the single-row compositions and the round 72 refusals unchanged |
| Affected selected programs | none of the fourteen CENT rows relocates a multi-row vector (the RMSNorm scale result and the FFN gate input occupy one row per partition), so no trace, pin or cycle count changes: golden rows 3 passed (default) and 11 passed (`paper_full`) at the round 73 pins; the CENT suite 8 passed, 13 skipped and 13 passed with the round 73 trace pins; the fourteen-row provenance at the checkout identical to round 73 (`aim_rows_provenance_checkout.json`) |
| AiM-related unit suites (`pytest_aim_related_1.log`, `pytest_aim_related_2.log`) | first batch (the AiM unit suites, the decision tests without the three pre-existing Samsung failures recorded in round 72, host compute and transfer dispatch, layout selection, match coverage, reduction policy, resource budget, work grid, verify): 233 passed, 1 skipped, 3 deselected, 217 warnings; second batch (`test_shared_semantics.py`, `test_g6_semantics.py`, `test_gemv_all_targets.py` with the native UPMEM SDK executor, the APU v1 board test deselected): 194 passed, 1 deselected, 53 warnings |
| Integration candidate `db2ad84` (`build_wheel.{sh,log}`, `pkg/`) | clean tree (0 dirty entries); sdist from `git archive`; wheel `7c8734a7...`; clean venv; `allo` imported from the venv's site-packages |
| The fourteen AiM rows, the suites and the three reviewer probes from the installed wheel (`run_installed.sh`, `installed_run.log`, `aim_rows_provenance.json`) | golden rows 3 passed (default) and 11 passed (`paper_full`); the CENT suite 8 passed, 13 skipped and 13 passed; `test_aim_value_layouts.py` 15 passed; the round 71 fan-out probe: default accepted, `fanout` and `inverse` refused; the round 72 slot-liveness witness: `u = 30`, `v = 42`; the round 73 multi-row witness: `u = (30, 60)`, `v = (42, 84)`; the fourteen-row provenance identical in cycles and trace digests to the checkout run; the inventory's AiM rows record the candidate |
| Not done | the Gemini-I campaign still queued behind the `/tmp` watcher (X-22 / X-23 OPEN); X-26 and mvt unchanged; the Samsung, UPMEM, AttAcc and Gemini-II rows last re-executed from earlier candidates (`ba2f58d`, `cd744bb`) |

## Round 75 (2026-11-06): the logical frame in the value-layout contract (X-32 reopened by the reviewer's prefix witness, repaired)

Checkout after `3bb7bfd`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r75/`.  The round 74
rows above that call X-32 repaired did not cover readers of another
logical frame; this section supersedes them.

| Check | Result |
|---|---|
| The reviewer's probe (`reviewer-r74/prefix.py`, re-run as `r75/prefix_rerun.py`; `prefix_rerun.log`) | at the round 74 tree (the reviewer, checkout and installed wheel of `db2ad84`): accepted, `u[0, 1024, 2048, 3072] = [2, 4, 6, 8]` for `[2, 3, 4, 5]`; at the repaired checkout: refused before selection, `AiM consumer: reads t from the device as 1 copy over 32 channels each, bank stride 4, elements [0, 131072) at 1024 per bank partition, but launch producer left it as 1 copy over 32 channels each, bank stride 4, elements [0, 262144) at 2048 per bank partition; ... make the kernels cover the same elements of the buffer, or have the host scatter the region the reader needs` (the per-item candidates refused alike) |
| `tests/pim/test_aim_value_layouts.py` (`pytest_value_layouts.log`) | 17 passed: the witness walks every channel, bank and row (the lane's logical index from the transfer's own layout, values differing per partition and per row chunk); the prefix composition refused with both frames and the remedy; the host-scattered prefix accepted with every observed lane the source's (128 partitions, one row each); the single-row, multi-row (both layouts), chain and reupload compositions re-witnessed over all partitions; the round 72 to 74 refusals unchanged |
| Affected selected programs | none of the fourteen CENT rows composes device values of different frames: golden rows 3 passed (default) and 11 passed (`paper_full`) at the round 73 pins (`pytest_golden_aim_{default,paper_full}.log`); the CENT suite 8 passed, 13 skipped and 13 passed (`pytest_cent_aim_{default,paper_full}.log`); the fourteen-row provenance at the checkout identical in cycles and trace digests to rounds 73 and 74 (`aim_rows_provenance_checkout.json`) |
| AiM-related unit suites (`pytest_aim_related_1.log`, `pytest_aim_related_2.log`) | first batch (the AiM unit suites, the decision tests without the three pre-existing Samsung failures recorded in round 72, host compute and transfer dispatch, layout selection, match coverage, reduction policy, resource budget, work grid, verify): 233 passed, 1 skipped, 3 deselected, 217 warnings; second batch (`test_shared_semantics.py`, `test_g6_semantics.py`, `test_gemv_all_targets.py` with the native UPMEM SDK executor, the APU v1 board test deselected): 194 passed, 1 deselected, 53 warnings |
| Integration candidate `b221972` (`build_wheel.{sh,log}`, `pkg/`) | clean tree (0 dirty entries); sdist from `git archive`; wheel `ace1b7b0...`; clean venv; `allo` imported from the venv's site-packages |
| The fourteen AiM rows, the suites and the four reviewer probes from the installed wheel (`run_installed.sh`, `installed_run.log`, `aim_rows_provenance.json`) | golden rows 3 passed (default) and 11 passed (`paper_full`); the CENT suite 8 passed, 13 skipped and 13 passed; `test_aim_value_layouts.py` 17 passed; the round 71 fan-out probe: default accepted, `fanout` and `inverse` refused (the frames now in the messages); the round 72 slot-liveness witness: `u = 30`, `v = 42`; the round 73 multi-row witness: `u = (30, 60)`, `v = (42, 84)`; the round 74 prefix probe: refused with both frames named (`prefix_installed/prefix.log`); the fourteen-row provenance identical in cycles and trace digests to the checkout run; the inventory's AiM rows record the candidate |
| Not done | the Gemini-I campaign still queued behind the `/tmp` watcher (X-22 / X-23 OPEN); X-26 and mvt unchanged; the Samsung, UPMEM, AttAcc and Gemini-II rows last re-executed from earlier candidates (`ba2f58d`, `cd744bb`) |

## Round 76 (2026-11-06): the elementwise lowering proves a dense domain (X-33); the Gemini-I campaign launched

Checkout after `462399c`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r76/` (and the campaign
under `r69/`, where its watcher was queued).

| Check | Result |
|---|---|
| The reviewer's witness (`reviewer-r75/strided.py even`, re-run as `r76/strided_rerun.py`; `strided_rerun_even.log`) | at the round 75 tree (the reviewer: checkout, `b221972` and `db2ad84` wheels): accepted, `u[1] = 30` for the untouched 99; at the repaired checkout: refused before selection, `AiM alpha: the elementwise lowering covers dense iteration domains only; loop %arg3 steps by 2 (trip 131072), so the EWMUL / EWADD stream would write the elements the source skips` (every candidate) |
| The dense counterpart (`reviewer-r75/dense.py full`, `r76/dense_rerun_full.log`) and the script's suffix mode (`strided.py suffix`, `strided_rerun_suffix.log`) | the dense program accepted with its witness at the source semantics; the suffix mode refused by the round 22 availability check, correctly: the script's host program moves `a[H:N-1]` (the stride mode's slice) while the suffix kernel reads `a[H:N)`, so element N-1 is never brought to the device (`launch 0 of 'alpha' reads 'a' over [(131072, 262144)] but the host program has not brought [(262143, 262144)]`); the dense suffix with matching transfers is covered by the regression below |
| `tests/pim/test_aim_dense_domains.py` | 4 passed: the stride-two loop and the strided coordinate `z[2 * i]` refused with their diagnostics; the dense kernel and the dense suffix compile, the generated work multiplies every lane of every written cell, element counts N and H |
| Affected selected programs | none: every CENT elementwise kernel iterates `range(n)` with identity coordinates; golden rows 3 passed (default) and 11 passed (`paper_full`) at the round 73 pins; the CENT suite 8 passed, 13 skipped and 13 passed; `test_aim_value_layouts.py` and `test_aim_dense_domains.py` 21 passed (`pytest_*.log`) |
| The Gemini-I campaign (`r69/campaign_watcher.log`, `r69/apu_v1_campaign.log`) | the round 69 watcher launched the pre-registered campaign at 2026-11-06T02:47:44 with 348,020 kB free in `/tmp` (the host's other users released space); session 1 started 02:47:48 from the installed wheel of `472bec9` (`6dcbfff0...`; the APU v1 modules and benchmarks are unchanged between `472bec9` and the current head, `git diff --stat` empty), three paired sessions of kmeans, linear regression and word count with ten measured launches per lane, the expert and the archived paper-era binaries interleaved, exact outputs required, every session kept and pooled. the round 69 watcher launched the pre-registered campaign at 02:47:44 with 348,020 kB free in `/tmp`; the host's other users refilled the partition within minutes (89,572 kB at 03:03, 468 kB at 03:10), so session 1 lost its launches to the vendor tool's per-read scratch (`ledag-ssh exited with 255 while reading the device log`): kmeans kept 8 of 10 measured launches per lane (tenon, expert and the archived paper-era binary; every output exact), linear regression and word count none; sessions 2 and 3 were refused by the 25 MB per-session guard; campaign exit 3, the session record kept (`r69/apu_v1_campaign_s1.json`, status `fail (3 case(s))`, incomplete, recorded and not pooled under the acceptance rule). A retry of the same pre-registered protocol (three sessions, the same `472bec9` wheel and sampling rule, sessions written under `r76/`) is queued behind `r76/campaign_watcher.sh` at a 120 MB threshold with the per-session guard; X-22 / X-23 stay OPEN until three complete sessions exist |
| AiM-related unit suites (`pytest_aim_related_1.log`, `pytest_aim_related_2.log`; the APU v1 board tests deselected while the campaign held the board, the three pre-existing Samsung failures of round 72 deselected) | first batch: 229 passed, 1 skipped, 7 deselected, 217 warnings; second batch (`test_shared_semantics.py`, `test_g6_semantics.py`, `test_gemv_all_targets.py`): 155 passed, 40 deselected, 52 warnings |
| Not done | the installed-candidate re-execution of X-33's refusal (with the next candidate); X-26 and mvt unchanged; the Samsung, UPMEM, AttAcc and Gemini-II rows last re-executed from earlier candidates |

## Round 77 (2026-11-06): the Gemini-II batched_gemm (X-26) and mvt gaps attributed from the recorded sessions

No new launches; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r77/`.

| Check | Result |
|---|---|
| Selection and generated work (`g2_plan_probe.py`, `.json`, compile-only from the checkout; the session records' per-launch provenance) | batched_gemm selects `single/k1` (estimate 1,714; the only alternative `single/k2` at 2,550), mvt `fused/k1` (2,740; `separate/k1` 5,480), one launch each; every recorded session carries one program fingerprint, one layout fingerprint and one packing attestation per row, unchanged since round 30 |
| Measurement scope | the statistic is the repeated batch's device pipeline ticks divided by the repetitions (`benchmarks/apu_g2/paper_rows.py`); host launch timings are recorded per launch and are outside it |
| Session-by-session reading (`g2_gap_attribution.py`, `.json`, corrected in round 78: every launch sample pooled, complete packing attestations compared; r30 runs 1 and 2 at `69773fc`, the four `54cbb08` sessions, the `cd744bb` session, the five-round round 29 run) | batched_gemm: Tenon session medians 1,716.35 / 1,731.55 / 1,744.55 / 1,740.58 / 1,743.43 / 1,740.65 / 1,739.68 ticks (0.9869 to 1.0031 of the paper's 1,739.15) with the GTML lane in the same rounds at 1,916.1 / 1,910.9 / 1,912.9 / 1,919.7 / 1,913.4 / 1,913.5 (1.0023 to 1.0070 of the paper's 1,906.4); the vendor's excess exceeds Tenon's in 5 of 6 paired sessions; the paired same-round GTML/Tenon ratio 1.1048 / 1.0946 / 1.0974 / 1.0984 / 1.0983 / 1.1004 against the paper's 1.0962, at or above it in 5 of 6; pooled over the 120 launch samples of the six paired sessions the Tenon median is 1.001150 of the paper's column (the median of the session medians 1.000841) and the paired ratio 1.0984. mvt: Tenon 2,743.85 / 2,755.03 / 2,763.33 / 2,762.13 / 2,755.6 / 2,771.3 / 2,767.23 (0.9918 to 1.0017 of 2,766.625), GTML 0.9961 to 1.0003 of 3,089.65, the paired ratio 1.1149 to 1.1221 against 1.1168; pooled over the 120 samples 0.999223 of the paper (the median of the session medians 0.998590), the paired ratio 1.1160. One program, layout and complete packing attestation per row across every session |
| Conclusion | no generated-program change is identified between the sessions below and above the paper; the enumerated rankings show the selected plan has the lowest estimate among the enumerated candidates, not that no alternative could run faster; the observations are consistent with session-to-session measurement variability of the card that the re-run baseline shares, and do not establish the cause of the residual gaps (0.5 and 0.6 ticks at `cd744bb`); no mechanism repair is supported by the evidence; both rows OPEN under the pooled-session rule (X-26's campaign spent; no session added); the Gemini-II rows are re-measured once more only as the pre-registered final-candidate validation session |
| Not done | the Gemini-I retry still queued behind the `/tmp` watcher (X-22 / X-23 OPEN); the final-candidate runs of the Samsung, UPMEM, AttAcc and Gemini-II rows, X-33's installed refusal and the G9 review at one named candidate |

## Round 78 (2026-11-06): the named-candidate integration milestone (pre-registration, written before any launch)

Candidate: the revision of this commit (the code tree is that of `9d7dc81`,
X-33 included; the commits after it change release documentation only).
Evidence under `/work/shared/users/phd/nz264/tenon-release-evidence/r78/`.

Pre-registered before the launch:

- Local: sdist from `git archive` of the candidate, wheel and clean venv
  (`r78/pkg`); from the installed wheel with the tests and benchmarks copied
  out of the checkout: the round 8 probes, X-33's refusal (`reviewer-r75/strided.py even`) and its dense neighbours (`dense.py full`, `test_aim_dense_domains.py`), the value-layout
  regressions and the reviewer's round 71 to 74 probes, the fourteen AiM rows
  (golden default and `paper_full`, the CENT suite, the fourteen-row
  provenance), the three Samsung rows (PIMSimulator), the sixteen UPMEM rows
  through the native SDK executor, the twelve AttAcc results (six kernel rows
  through `run_ramulator.py` with the fixed-form baselines, six decode
  configurations through `compose_e2e.py`), the shared-semantics suite and
  the grouped route's section u, the reviewer's round 65 map probe.
- Gemini-II: the sdist built into a wheel on zhang-x4 and installed into a
  clean venv there; the card tests from that wheel; then ONE paired session
  of 20 rounds of `benchmarks.apu_g2.paper_rows` for all five rows under the
  device lock (the archived protocol: one warm-up per lane and case, the
  GTML lane interleaved, exact outputs required).  It is the candidate's
  validation session, recorded whatever the outcome; at this revision the
  rows' statistic is the pooled samples of this one session; it is not a
  further sample of the spent batched_gemm campaign at `54cbb08` (X-26), and
  no second session follows at this revision.
- Gemini-I: not run at this candidate (the host's `/tmp`, 140 kB free); the
  queued retry at `472bec9` (`r76/campaign_watcher.sh`) stays as
  pre-registered and cannot establish execution at this candidate; recorded
  as not re-executed here.

Results (the named candidate `d120084`; local sdist `ccbd0ce5...`, wheel
`81e37bac...`, clean venv, 0 dirty entries, `build_wheel.log`; zhang-x4
wheel `e7aff69d...` from the same sdist, `g2-build-wheel.log`):

| Stage | Result |
|---|---|
| Installed origin and the round 8 probes (`installed_run.log`) | `allo` and every backend module imported from the venv's site-packages; the fresh-install, Samsung simulator, AiM and UPMEM probes as in rounds 63 to 70 |
| X-33 from the wheel | `reviewer-r75/strided.py even` refused (`NotImplementedError: AiM alpha: the elementwise lowering covers dense iteration domains only; loop %arg3 steps by 2 (trip 131072) ...`, `x33/strided_even.log`); the dense neighbour `dense.py full` accepted with its witness (`x33/dense_full.log`); `test_aim_dense_domains.py` and `test_aim_value_layouts.py` 21 passed |
| The reviewer's round 71 to 74 probes from the wheel (`probes/`) | fan-out and inverse refused; the slot-liveness and multi-row witnesses at the source semantics (exit 0); the prefix composition refused |
| AiM | golden rows 3 passed (default) and 11 passed (`paper_full`) at the round 73 pins; the CENT suite 8 passed, 13 skipped and 13 passed; the fourteen-row provenance identical in cycles and trace digests to rounds 73 to 75 (`aim_rows_provenance.json`) |
| Samsung (PIMSimulator, `installed_run.log`) | 3 passed: default selection 4,108 / 7,057 / 7,255 cycles against the paper's 4,435 / 8,041 / 8,247 (the vendor-equivalent program at the paper's values), the swizzled split-rows placement with shared CRF issue selected as in rounds 37 to 68 |
| UPMEM (native SDK executor) | default 19 passed (the provenance and cycle rows of the default set), `paper_full` 13 passed, the UPMEM GEMV test 1 passed: the sixteen rows at the paper's cycles with NumPy-exact outputs |
| AttAcc | the six kernel rows through `run_ramulator.py` with the fixed-form baselines for the GEMV rows and the six decode configurations through `compose_e2e.py` from the installed wheel (`run_attacc_installed.log`, `attacc_*.json`, `attacc_e2e.json`): every value equal to the `54cbb08` set and the five later candidates (`compare_attacc.py`, `compare_attacc.log`: 0 differences excluding paths, command lines, config digests and wall times; e.g. the decode table tenon 1.47x / 6.50x, 1.56x / 6.95x, 1.66x / 7.40x, 1.34x / 3.09x, 1.50x / 3.48x, 1.65x / 3.87x as in rounds 52 to 68); the AttAcc tests 65 passed, 1 skipped |
| Shared semantics and the grouped route | the whole suite from the wheel 156 passed, 1 failed (`installed_run.log`, `installed_shared_semantics.log`): the failure is `test_local_accumulator_gemm_executes_on_apu_v1_functional_backend`, `ValueError: setting an array element with a sequence` in the APU v1 functional reference route under the clean venv's NumPy 2.5.4, the known deferred V1-19 (the test passes on the checkout's NumPy 2.3.5, rounds 74 and 75 batch 2); a fourth instance added to the V1-19 row; the grouped route's section u 19 passed; the reviewer's round 65 map probe refused from the wheel |
| Gemini-II (zhang-x4; `g2-card-tests.log`, `g2-candidate-session.log`, `g2_candidate_rows.json`) | the card tests 11 passed from the wheel; the ONE pre-registered validation session of 20 rounds: status pass, every output exact; gemm 1,740.575 ticks per body against the paper's 1,733.225 (ratio 1.0042; q1 1,720.2, q3 1,752.5; GTML in the same rounds 1.0074 of its column; paired GTML/Tenon 1.1023 against the paper's 1.0984), batched_gemm 1,734.9 against 1,739.15 (0.9976; GTML 1.0030; paired 1.1029 against 1.0962), gesummv 1,832.8 against 1,860.075 (0.9853), bicg 2,538.9 against 2,548.825 (0.9961), mvt 2,762.375 against 2,766.625 (0.9985; paired 1.1169 against 1.1168); the same plans, program fingerprints and packing attestations as every earlier session. Under the pooled-session rule at this revision: batched_gemm, gesummv, bicg and mvt REPRODUCED at `d120084`; gemm OPEN (nominal) at `d120084` (X-34, its one confirmation campaign pre-registered for the final candidate, not launched) |
| Gemini-I | not re-executed at this candidate: the board host's `/tmp` (140 kB free at the start of the round); the queued retry at `472bec9` (`r76/campaign_watcher.sh`) stays as pre-registered and cannot establish execution at `d120084` |
| G9 source-review packet | `docs/release/G9_EVIDENCE.md`: the common orchestration, the candidate contract, every backend's hooks on the same stages (both APUs included) with line anchors at this candidate, the suites that exercise them, and this installed evidence |

## Round 79 (2026-11-06): the APU v1 vector route binds its realized computation to the contraction (X-35)

Checkout after `c874397`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r79/`.

| Check | Result |
|---|---|
| The reviewer's injections (`reviewer-r78/v1_realization_probe.py`, `v1_instruction_probe.py`, re-run at the checkout: `v1_realization_probe_rerun.log`, `v1_instruction_probe_rerun.log`) | at `d120084` (the reviewer, installed wheel): both accepted, `gvml_mul_f16(vr___total_product, vr_left, vr_left)` emitted at estimate 130,634 with no rejection; at the repaired checkout: the damaged-operations realization refused under forced `temporal_svp` and under default selection (`realization operations differ from the contraction: realized [('MUL_F16', '__total_product', ('left', 'left'), {}), ('ADD_F16', 'total', ('total', '__total_product'), {})], derived from the shared contraction and plan 'temporal_svp' [('MUL_F16', '__total_product', ('left', 'right'), {}), ...]`); the damaged-instruction realization refused for every plan (`realization instructions differ from the contraction's operations`, the compile ending in `no feasible placement`) |
| `tests/pim/test_apu_v1_realization_binding.py` | 5 passed: the intact program emits `gvml_mul_f16(vr___total_product, vr_left, vr_right)` with no rejection; damaged operations refused when forced and when every plan is damaged; damaged instructions refused; damaging only the plan the default selection takes rejects it (reason recorded in the ranking), selects an intact alternative whose multiply reads both operands, and refuses it when forced |
| APU v1 and shared suites at the checkout | the shared-semantics APU v1 and vector subset 30 passed (`pytest_shared_apu_v1.log`); the APU v1 perf suites (build config, cost provenance, distance argmin, hybrid manifest, native reductions, performance, recognizers, reduction policy, scalar surface, tile selection, vector compile), the compile interface, layout problem, legality, portability driver and fingerprint suites, and the decision, layout-selection, transfer-dispatch and work-grid suites (`pytest_apu_v1_perf.log`): 6 failed, 281 passed, 3 deselected, 220 warnings; the six failures are UPMEM and AiM execution tests of `test_g6_decisions.py` (test_upmem_calibration_flip_reaches_execution, test_upmem_renamed_gemv_same_decision_and_cycles, test_upmem_axpby_coefficients_at_fixed_shape, test_aim_elementwise_grid_simulates_every_work_item, test_upmem_kmeans_distances_and_update_execute, test_aim_executes_without_archives) run in this batch without the simulator executors configured (the batch set neither `TENON_UPMEM_SDK_HOME` nor `TENON_AIM_RAMULATOR2`: the UPMEM tests fell back to the Docker image and met the host's containerd ENOSPC, the AiM tests raised `SimulatorUnavailable`), the same tests having passed in rounds 76 and 78 with them configured; every APU v1 test passed |
| Ledger consistency | PAPER_REPRODUCTION's status paragraph rewritten as one current snapshot (51 REPRODUCED, 4 OPEN at `d120084`) with history separated; REVIEW's G10 row carries one current count; `G9_EVIDENCE.md` updated for X-35 |
| From the installed wheel of `6ce37ad` (`build_wheel.log`, wheel `08b58ac9...`, clean venv; `run_installed_probes.sh`, `installed_probes.log`, `installed/`) | both reviewer probes refused with the same diagnostics as at the checkout (the instruction probe ending in `no feasible placement`); `test_apu_v1_realization_binding.py` 5 passed; the shared-semantics APU v1 subset 29 passed, 1 failed (the known deferred V1-19, NumPy 2.5.4) |
| Not done | Gemini-I (the board host's `/tmp`, 132 kB free at the start of the round); X-34's pre-registered campaign (the final candidate); the independent review of the repair |

## Round 80 (2026-11-06): one realization-boundary invariant for the APU v1 vector route (X-36)

Checkout after `7bce26a`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r80/`.

| Check | Result |
|---|---|
| The reviewer's injections (`reviewer-r79/transfer_probe.py`, `emitted_order_probe.py`; re-run at the checkout: `transfer_probe_rerun.log`, `emitted_order_probe_rerun.log`) | at `6ce37ad` (the reviewer, installed wheel): both accepted; at the repaired checkout: the misdirected ingress refused under forced `temporal_svp`, forced `temporal_dma_coalescing` and default selection (`instruction 2 (transfer_in) differs from the plan's stream: realized (... ('vr', 'p') ...), derived (... ('vr', 'q') ...)`, ending in `no feasible placement`), the intact program still accepted; the swapped emission refused (`APU v1 emitted device source differs from the realization's emission schedule at line 19`) |
| The round 78 injections re-run (`v1_realization_probe_rerun.log`, `v1_instruction_probe_rerun.log`) | still refused |
| The schedule-based renderer against the previous renderer (`differential_emission.py`, `.log`; run while the previous renderer was still present in the working tree, then removed) | 21 plans of the square, rectangular and large contractions, a GEMV and the paper's matrix multiply (accumulator blocks 1, 2, 4 and 8, the L4 variants, the spatial scatter form `baseline_spatial_reduction`): the device source identical byte for byte, the invariant satisfied by every plan |
| `tests/pim/test_apu_v1_realization_binding.py` | 11 passed |
| APU v1 and shared suites at the checkout | the shared-semantics APU v1 and vector subset, the APU v1 perf suites (build config, cost provenance, distance argmin, hybrid manifest, native reductions, performance, recognizers, reduction policy, scalar surface, tile selection, vector compile), the compile interface, layout problem, legality, portability driver and fingerprint suites, and the layout-selection, transfer-dispatch and work-grid suites: 139 passed (`pytest_apu_v1_suites.log`); the paper APU v1 kernels' compile-only golden test, the matrix multiply at its accumulator-block plan among them: 5 passed (`pytest_golden_apu_v1_compile.log`) |
| Not done | the installed re-execution (next candidate, which must contain `6ce37ad` and this commit); Gemini-I (the board host's `/tmp`); X-34's pre-registered campaign (the final candidate); the independent review of the repair |

## Round 81 (2026-11-06): the structured emission schedule checked against the plan (X-36 reopened by the reviewer's round 80 injections, repaired)

Checkout after `7fb19a4`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r81/`.  The round 80
rows that call the schedule bound are superseded by this section.

| Check | Result |
|---|---|
| The reviewer's round 80 injections (`reviewer-r80/schedule_probe.py`) | at `7fb19a4` (the reviewer): the forged call binding, the early egress and the empty reduction loop all accepted with no rejection; at the repaired checkout the probe no longer imports (its flat `EmissionCall` / `EmissionLine` records were replaced by structured nodes, `schedule_probe_rerun.log`); the three injections ported to the structured nodes are refused (`test_apu_v1_realization_binding.py`: `substitutes 'q' by the register of 'p', which the plan does not allow in this scope`; `stores vr_r before a later compute defines it`; `loop 'reduction_step' (reduction_step) runs 0 times; the plan requires 8`) |
| The round 78 and 79 probes re-run (`*_rerun.log`) | still refused (the intact program of the transfer probe accepted) |
| The structured renderer against the parent renderer (`differential_actual.py`, the reviewer's comparator embedding the renderer of `7bce26a`; `differential_actual.log`) | 21 plans (the square, rectangular and large contractions, a GEMV, the paper's matrix multiply with accumulator blocks 1 to 8, the L4 variants and the spatial scatter form) identical byte for byte and valid under the checker: ALL_IDENTICAL_AND_VALID |
| `tests/pim/test_apu_v1_realization_binding.py` | 15 passed |
| APU v1 and shared suites at the checkout | the shared-semantics APU v1 and vector subset, the APU v1 perf suites, the compile interface, layout problem, legality, portability driver and fingerprint suites, the layout-selection, transfer-dispatch and work-grid suites and the paper golden rows selected by `apu_v1 or vector or gvml` (`pytest_apu_v1_suites.log`): 1 failed, 144 passed, 334 deselected, 4 warnings; the one failure is the board row `test_apu_v1_board_median_cycles[apu_v1-histogram]`, a device launch that needs the board host's `/tmp` (the vendor tool exited while reading the device log), the same blocker that holds the Gemini-I campaign; the compile-only golden rows, every realization and every compile test pass |
| Evidence integrity | `r80/differential_emission.py` now exits with an error naming the executable comparator instead of printing an unchecked success |
| Not done | the installed re-execution (the next named candidate must contain `6ce37ad`, `7fb19a4` and this commit); Gemini-I (the board host's `/tmp`); X-34's pre-registered campaign (the final candidate); the independent review of the repair |

## Round 82 (2026-11-06): the schedule executed against the plan's batching (X-36 reopened by the reviewer's round 81 injections, repaired)

Checkout after `5965d1f`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r82/`.  The round 81
rows that call the schedule bound are superseded by this section.  No
installed artifact and no board or card launch in this round.

| Check | Result |
|---|---|
| The reviewer's round 81 injections (`reviewer-r81/schedule_probe.py`) | at `5965d1f` (the reviewer): `index_zero`, `pointer_zero`, `extra_compute`, `duplicate_slot` and `scatter_zero` all accepted with no rejection; at the repaired checkout, the reviewer's probe with `scatter_zero` adapted to the structured scatter and two scatter faults added (`schedule_probe_r82.py`, `schedule_probe_r82.log`): `index_zero` and `pointer_zero` refused (`batch 0 of 'c' receives a value other than the plan's computation (its operands, their order, their addresses or the reduction extent differ)`), `extra_compute` refused (`the emission calls gvml_add_f16, which is not in the plan's instruction stream`), `duplicate_slot` refused (`batch 0 of 'c' is stored 2 times; the plan stores it 1 time(s)`), the scatter of another register refused (`a scatter of a register that is not the output's`), every work tile scattered to element zero refused (`the scatter of 128 lanes (stride 256) of 'c' to elements 0..127 is stored 1536 times; the plan stores it 1 time(s)`), twice the lanes refused (`is never stored`); the intact controls (`fresh` at `temporal_svp`, `wide`, `gemv`, `integer_neighbor` at `temporal_dma_coalescing_broadcast_friendly_acc2` and at `baseline_spatial_reduction`) accepted with no rejection; the reviewer's unmodified probe (`schedule_probe_rerun.log`): four injections refused by the contract, `scatter_zero` failing with `AttributeError` because the scatter no longer carries body text |
| The round 78 and 79 probes re-run (`v1_instruction_probe_rerun.log`, `v1_realization_probe_rerun.log`, `emitted_order_probe_rerun.log`, `transfer_probe_rerun.log`) | still refused with the round 79 to 81 messages (the intact program of the transfer probe accepted) |
| The structured renderer against the parent renderer (`differential_actual.py`, the reviewer's round 81 comparator unchanged but for its docstring; `differential_actual.log`) | 21 plans identical byte for byte and valid under the interpreter: ALL_IDENTICAL_AND_VALID |
| `tests/pim/test_apu_v1_realization_binding.py` (`pytest_realization_binding.log`) | 29 passed: the rounds 78 to 80 cases (the early-egress and hoisted-ingress expectations now the interpreter's messages: the stored value differs; the hoisted loads name `batch` outside its scope), the five round 81 injections, three structured-scatter faults, `acc2`, `acc4` and `baseline_spatial_reduction` accepted intact with the emitted text the schedule's, the cost-preferred `acc4` candidate damaged alone (slots folded) rejected with `plan 'temporal_dma_coalescing_broadcast_friendly_acc4'` and `is stored 4 times` in the recorded reason while an intact alternative is selected, the resident GEMV `spatial_gemv_group_reduction_k64` (four banks) accepted intact, every bank's tile index aliased to bank zero refused (the stored value differs), the banks' substitutions stripped refused (`reads the resident vector in bank 1 without that bank's register`) |
| APU v1 and shared suites at the checkout (`pytest_apu_v1_suites.log`, the file list in `suite_files.txt`) | selected by `apu_v1 or vector or gvml`: 1 failed, 144 passed, 352 deselected, 4 warnings; the one failure is the board row `test_apu_v1_board_median_cycles[apu_v1-histogram]` (`ledag-ssh exited with 255 while reading the device log`), the board host's `/tmp` at 124 KB free, the same blocker that holds the Gemini-I campaign; every compile-only golden row, realization and compile test passes |
| GEMV tile and reduction-policy suites (compile the resident form) | `tests/perf/test_apu_v1_tile_selection.py`, `tests/perf/test_apu_v1_reduction_policy.py`: 8 passed |
| Not done | the installed re-execution (the next named candidate must contain `6ce37ad`, `7fb19a4`, `5965d1f` and this commit); Gemini-I (the board host's `/tmp`, watcher alive); X-34's pre-registered campaign (the final candidate); the independent review of the repair; G4 and G9 are not claimed closed from this checker |

## Round 82, named-candidate milestone at `92d4120` (pre-registration, written 2026-11-06 07:52 EST, before any result was read)

The candidate is `92d4120` on `fork/tenon`, the first commit that contains
every production fix after `d120084` (`6ce37ad`, `7fb19a4`, `5965d1f`,
`92d4120`; `git diff d120084..92d4120 -- allo benchmarks` touches only
`allo/pim/apu_v1_vector_codegen.py` and `allo/spmw_apu_v1.py`, so every
other backend's production code is byte-identical to `d120084`).  Evidence
under `/work/shared/users/phd/nz264/tenon-release-evidence/r82/`.  Launched
in this order, each script a copy of the round 78 chain with the round
changed and the scratch directory moved under `r82/tmp` (the host's `/tmp`
is full):

- `build_wheel.sh` (07:45): sdist from the committed tree (dirty entries 0),
  wheel `dd91e58e...` built with the shared clang into a clean venv under
  `r82/pkg` (`WHEEL_OK` at 07:48).
- `g2_remote_chain.sh` on zhang-x4 (07:47): the same sdist built into a
  wheel there, the card tests from that wheel, then ONE pre-registered
  paired session of 20 rounds of `benchmarks.apu_g2.paper_rows`
  (`g2_candidate_session.sh`; its purpose statement is in the script).  The
  statistic at this revision is that session's pooled samples, recorded
  whatever the outcome.  It is not the X-34 gemm campaign (pre-registered
  for the final candidate, not launched) and not a further sample for the
  exhausted batched_gemm campaign (X-26).
- `run_installed.sh` (07:49): the local stages from the installed wheel
  exactly as in round 78 (fresh-install and simulator probes, X-33's
  refusal and dense neighbour, the AiM dense-domain and value-layout
  suites, the rounds 71 to 74 reviewer probes, the golden AiM rows default
  and `paper_full`, the CENT suite default and `paper_full`, the
  fourteen-row provenance, the golden Samsung rows, the golden UPMEM rows
  default and `paper_full`, the UPMEM GEMV test, the six AttAcc rows and
  the composed decode configurations with the round 67 comparison, the
  AttAcc suites, the whole shared-semantics suite and the grouped route's
  section, the round 65 map probe).  Acceptance: every row the same as at
  `d120084` (same cycles, same trace digests, same pass counts); a
  difference is a finding, not a tolerance.
- `run_installed_probes.sh` (07:49): from the installed wheel, the rounds
  78, 79 and 81 reviewer probes and the round 82 probe (refusals expected,
  intact controls accepted), the 29 realization-binding regressions, the
  paper APU v1 kernels' compile-only golden rows, the GEMV tile and
  reduction-policy suites and the shared-semantics APU v1 subset.
- Not launched: anything on the Gemini-I board (the host's `/tmp` is at
  124 KB free; the Gemini-I rows stay blocked); the X-34 gemm campaign.

### Results (read after every chain finished, 09:25 EST)

| Check | Result |
|---|---|
| Artifacts | sdist `e40df359...` from `git archive` of `92d4120` (0 dirty entries); local wheel `dd91e58e...` (`pkg/`), zhang-x4 wheel `32c005b6...` (`g2/`), both installed into clean venvs; `allo` and every backend module imported from site-packages in every log (`installed_run.log`, `installed_probes.log`, `g2-card-tests.log`, `g2-candidate-session.log`) |
| The X-35 / X-36 refusals from the wheel (`installed_probes.log`, `installed/*.log`) | the module exposes `_ScheduleInterpreter` and no `ScheduleLine`; the round 78 probes refused (operations and instruction differences), the round 79 probes refused (the misdirected ingress under both forced plans and default selection, the emitted-order fault), the reviewer's round 81 probe: four injections refused by the contract and `scatter_zero` failing on the removed body text, the round 82 probe: all seven faults refused and the five intact controls accepted; `test_apu_v1_realization_binding.py` 29 passed; the paper APU v1 kernels' compile-only golden rows 5 passed; GEMV tile and reduction policy 7 passed, 1 failed (V1-19, `perf_rerun.log`); shared semantics `apu_v1 or vector or gvml` 29 passed, 1 failed (V1-19, `shared_rerun.log`) |
| Fresh-install and simulator probes | `fresh_install_check`, `fresh_sim_check`, `aim_check`, `upmem_check` exit 0 |
| X-33 and its neighbours | the stride-two witness refused (exit 1), the dense program accepted, `test_aim_dense_domains.py` + `test_aim_value_layouts.py` 21 passed |
| The reviewer's round 65, 71 to 74 probes from the wheel | fan-out and inverse refused, slot liveness and multi-row at the source semantics, the prefix composition refused (`probes/`), the round 65 first map refused (exit 1) |
| AiM | golden rows default 3 passed (11 skipped), `paper_full` 11 passed; the CENT suite default 8 passed (13 skipped), `paper_full` 13 passed; the fourteen-row provenance (`aim_rows_provenance.json`): every row's cycles and trace digest identical to the `d120084` set (attention 11,010 / 18,126 / 95,090 and ffn_fc 250,433 exact; the other ten below the paper as before) |
| Samsung | 3 passed; default 4,108 / 7,057 / 7,255 against the paper's 4,435 / 8,041 / 8,247, identical to `d120084` |
| UPMEM (native SDK executor) | default 19 passed, 13 skipped; `paper_full` 13 passed; the GEMV test 1 passed (as at `d120084`) |
| AttAcc | the chain's own AttAcc stage failed at once (`ModuleNotFoundError: benchmarks`: the chain copies the benchmarks only after a `cp` of `pytest.ini`, which the tree does not have, so the copy never happens; the same defect as round 78) and the comparison lines it printed into `installed_run.log` compared `r57` with `r62` (the round 67 script's hard-coded rounds): those lines are disregarded. The twelve results re-run alone from the wheel with an explicit benchmarks copy (`run_attacc_installed.sh`, every row exit 0, compose exit 0; `attacc_*.json`, `attacc_e2e.json`); the round 78 comparator with the round changed (`compare_attacc.py`, `compare_attacc.log`): all six kernel rows and the composed configurations identical to the `r57` baseline, 0 differences excluding timestamps and paths; the AttAcc suites 65 passed, 1 skipped |
| Shared semantics (whole) from the wheel | 1 failed, 156 passed (`installed_run.log`; re-run with failure names, `installed/shared_whole_rerun.log`: the one failure is `test_local_accumulator_gemm_executes_on_apu_v1_functional_backend`, V1-19 under the venv's NumPy 2.5.4, identical to the `d120084` run); the grouped route's section 19 passed |
| Gemini-II (zhang-x4, `g2-card-tests.log`, `g2-candidate-session.log`, `g2_candidate_rows.json`) | the card tests 15 passed; the one pre-registered validation session: status pass, 210 launches, no timing problem, every output exact, the same program and layout fingerprints and packing attestations as the `d120084` session for all five rows; medians against the paper: gemm 1,733.025 / 1,733.225 (0.9999), batched_gemm 1,737.675 / 1,739.15 (0.9992), gesummv 1,837.725 / 1,860.075 (0.9880), bicg 2,545.175 / 2,548.825 (0.9986), mvt 2,767.075 / 2,766.625 (1.0002, the paper inside the IQR 2,754.6 to 2,777.0); under the pre-registered rule the statistic at this revision is this session's samples: gemm REPRODUCED at `92d4120` (X-34, reported beside the `d120084` session, not pooled), mvt OPEN (nominal) at `92d4120` (X-37); the confirmation campaign stays pre-registered for the final candidate and was not launched |
| Gemini-I | not run (the board host's `/tmp` at 124 KB free); the three open rows stay blocked |
| Inventory (`docs/release/paper_reproduction/inventory.csv`, `update_inventory.py`, byte-exact CRLF edit, 84 lines, 18 columns) | 55 required rows recorded at `92d4120`: the fourteen AiM, three Samsung, sixteen UPMEM and twelve AttAcc rows REPRODUCED from the installed wheel, the Gemini-II rows from the session (four REPRODUCED, mvt OPEN nominal), the five Gemini-I rows noted as not re-executed; snapshot 51 REPRODUCED, 4 OPEN (kmeans, linear regression, word count, mvt), 0 UNVERIFIED |
| Not done | the independent review of the round 82 repair (G4, G9); the Gemini-I rows (the board host's `/tmp`); the X-34 / X-37 campaign (the final candidate); V1-19 (deferred by direction) |

## Round 83 (2026-11-06): the interpreter's storage keyed by the concrete VR (X-36 reopened by the reviewer's round 82 alias probe, repaired)

Checkout after `f88dde9` (code after `92d4120`); evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r83/`.  Compile-only;
no board or card launch.  The rows of round 82 that describe the
interpreter are superseded only in the identity of its register cells.

| Check | Result |
|---|---|
| The reviewer's alias probe (`reviewer-r82/alias_probe.py`) | at `92d4120` (the reviewer, checkout and installed wheel): the reset moved before the multiply accepted with no rejection; at the repaired checkout (`alias_probe_rerun.log`): the intact program accepted, the moved reset refused by the production checker (`batch 0 of 'z' receives a value other than the plan's computation`), the reviewer's physical-key variant agreeing on both |
| The rounds 78, 79, 81 and 82 probes re-run (`*_rerun.log`) | every fault refused, every intact control accepted (the round 78 and 79 probes exit 1 on the raised refusal as before) |
| The structured renderer against the parent renderer (`differential_actual.py`, the reviewer's round 81 comparator; `differential_actual.log`) | 21 plans identical byte for byte and valid: ALL_IDENTICAL_AND_VALID |
| `tests/pim/test_apu_v1_realization_binding.py` (`pytest_realization_binding.log`) | 32 passed: the 29 earlier cases, the intact aliased lowering accepted (`vr_x` and `vr___z_reduce_tmp` in one VR, no rejection, the emitted text the schedule's), the moved reset refused, and default selection rejecting the damaged preferred plan `baseline_spatial_reduction` with `receives a value other than` recorded and selecting an alternative that shares no register |
| APU v1 and shared suites (`pytest_apu_v1_suites.log`, `suite_files.txt`) | 1 failed, 144 passed, 352 deselected; the failure is the board row `test_apu_v1_board_median_cycles[apu_v1-histogram]` (`ledag-ssh exited with 255`, the host's `/tmp`) |
| From the installed wheel of the round 83 commit `8d9136f` (`build_wheel.log`, wheel `60999e64...`, clean venv, 0 dirty entries; `run_installed_probes.sh`, `installed_probes.log`, `installed/`) | `allo` from site-packages, the module exposing `_ScheduleInterpreter` and no `ScheduleLine`; the reviewer's alias probe (`installed/alias_probe.log`): intact accepted, the moved reset refused, the physical variant agreeing; the rounds 78, 79, 81 and 82 probes refused with their controls accepted; `test_apu_v1_realization_binding.py` 32 passed; the paper APU v1 kernels' compile-only golden rows 5 passed; GEMV tile and reduction policy 7 passed and 1 failed, shared semantics `apu_v1 or vector or gvml` 29 passed and 1 failed, both failures the deferred V1-19 tests under the venv's NumPy 2.5.4 (`installed/v1_19_rerun.log`: `test_functional_numerics_follow_the_selected_plan`, `test_local_accumulator_gemm_executes_on_apu_v1_functional_backend`, `ValueError: setting an array element with a sequence`, as at `92d4120`); that rerun's broadened selection also picked `test_corrupted_upmem_declarations_are_refused_before_costing`, which fails there only because the probe tree copies `tests` without `benchmarks` (`ModuleNotFoundError: No module named 'workloads'`, `installed/upmem_declarations_rerun.log`); it passed in the round 82 chain's whole-suite run from the wheel with the benchmarks copied and is outside the APU v1 subset. The full local and Gemini-II chains were not repeated: `git diff 92d4120..8d9136f -- allo` touches only `allo/pim/apu_v1_vector_codegen.py` |
| Not done | the independent review of the repair (G4, G9); the next named candidate (must contain this commit); Gemini-I (the board host's `/tmp`); the X-34 / X-37 campaign (the final candidate); V1-19 (deferred) |
