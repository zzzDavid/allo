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

## Round 84 (2026-11-06): the round 83 repair accepted; named-candidate milestone at `88852b7` (pre-registration, written 10:05 EST, before any launch)

The reviewer accepted the round 83 repair (`reviewer-r83/`), closing the
round 82 physical-alias finding only.  The named candidate for the
remaining G9 execution evidence and the required G10 comparisons is
`88852b7` on `fork/tenon`: its code tree is that of `8d9136f`, the first
commit containing every production fix since `d120084` (`6ce37ad`,
`7fb19a4`, `5965d1f`, `92d4120`, `8d9136f`; all in the APU v1 vector
route), and `git diff 92d4120..88852b7 -- allo benchmarks` touches only
`allo/pim/apu_v1_vector_codegen.py`.  The working tree carries uncommitted
ledger edits (this pre-registration and the acceptance notes), so the
sdist is taken from `git archive 88852b7` explicitly and the recorded
dirty entries are documentation files only.  Evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r84/`.  Scripts are
the round 82 ones with the round changed and two corrections: the
benchmarks copy no longer depends on a `pytest.ini` that the tree lacks
(round 82's AttAcc stage failed on that), and the AttAcc comparator is the
round 78 one pointed at `r84`.

- `build_wheel.sh`: sdist from `88852b7`, wheel into a clean venv under
  `r84/pkg` (the venv's NumPy 2.5.4 reproduces the deferred V1-19 pair).
- `g2_remote_chain.sh` on zhang-x4: the sdist built there, the card tests
  from that wheel (`tests/pim/test_apu_g2_hardware.py`), then ONE
  pre-registered paired session of 20 rounds of
  `benchmarks.apu_g2.paper_rows` under the device lock with actual
  outputs, selected-program provenance and the existing timing boundaries.
  Under the rule of PAPER_REPRODUCTION.md the statistic at this revision
  is this session's pooled samples, recorded whatever the outcome; the
  earlier sessions at `d120084` and `92d4120` are reported beside it, not
  pooled.  It is not the bounded confirmation campaign (X-34 / X-37; at
  most one, at the final candidate, three sessions pooled with that
  revision's session) and not a further sample for the spent batched_gemm
  campaign (X-26).  No further candidate is named to seek another session.
- `run_installed.sh`: the local stages from the wheel as in round 82
  (fresh-install and simulator probes, X-33, the AiM suites and rows
  default and `paper_full`, the fourteen-row provenance, Samsung, UPMEM
  default and `paper_full` and the GEMV test, the six AttAcc rows and the
  composed configurations, the AttAcc suites, the whole shared-semantics
  suite and the grouped route's section, the round 65 and 71 to 74
  probes).  Acceptance: every row the same as at `92d4120`; a difference is
  a finding.
- `run_installed_probes.sh`: the rounds 78, 79, 81 and 82 reviewer probes
  (the alias probe included) and the round 82 probe from the wheel, the 32
  realization regressions, the compile-only golden rows, the GEMV tile and
  reduction-policy suites, the shared-semantics APU v1 subset.
- `benchmarks/apu_v1/paper_rows.py` (Gemini-I): NOT launched.  The board
  host's `/tmp` is at about 116 KB free and the vendor tool fails there
  (`ledag-ssh exited with 255`); the three open Gemini-I rows stay
  explicitly blocked until execution is possible.  The retry watcher of
  the pre-registered campaign stays queued.

### Results (read after every chain finished, 11:50 EST)

| Check | Result |
|---|---|
| Artifacts | sdist from `git archive 88852b7` (`pkg/dirty.txt`: the five uncommitted ledger files, no code), local wheel `9d7b6be6...` (`pkg/`), zhang-x4 wheel (`g2/`), clean venvs; `allo` and every backend module from site-packages in every log. The first local launch aborted at the test-tree copy because the chain still named a `conftest.py` the tree does not have (`installed_run_attempt1_copyfail.log`; no stage ran); the copy fixed, the chain re-launched at 10:06 (`installed_run.log`) |
| The X-35 / X-36 refusals from the wheel (`installed_probes.log`, `installed/*.log`) | the module exposes `_ScheduleInterpreter` and no `ScheduleLine`; the round 78 probes refused, the round 79 probes refused (the intact transfer control accepted), the reviewer's round 81 probe four refused and `scatter_zero` failing on the removed body text, the round 82 probe seven faults refused and five controls accepted, the reviewer's alias probe intact accepted and the moved reset refused (the physical variant agreeing); `test_apu_v1_realization_binding.py` 32 passed; the compile-only golden rows 5 passed; GEMV tile and reduction policy 7 passed, 1 failed and the shared-semantics APU v1 subset 29 passed, 1 failed, both the deferred V1-19 tests under NumPy 2.5.4 (`installed/v1_19_rerun.log`) |
| Fresh-install and simulator probes | `fresh_install_check`, `fresh_sim_check`, `aim_check`, `upmem_check` exit 0 |
| X-33 and its neighbours | the stride-two witness refused (exit 1), the dense program accepted, `test_aim_dense_domains.py` + `test_aim_value_layouts.py` 21 passed |
| The reviewer's round 65, 71 to 74 probes from the wheel | fan-out and inverse refused, slot liveness and multi-row at the source semantics, the prefix composition refused, the round 65 first map refused (exit 1) |
| AiM | golden rows default 3 passed (11 skipped), `paper_full` 11 passed; the CENT suite default 8 passed (13 skipped), `paper_full` 13 passed; the fourteen-row provenance (`aim_rows_provenance.json`): every row's cycles and trace digest identical to the `d120084` and `92d4120` sets |
| Samsung | 3 passed; default 4,108 / 7,057 / 7,255 against the paper's 4,435 / 8,041 / 8,247, identical to the earlier candidates |
| UPMEM (native SDK executor) | default 19 passed, 13 skipped; `paper_full` 13 passed; the GEMV test 1 passed |
| AttAcc | from the chain's own stage this time (every row exit 0, compose exit 0; `attacc_*.json`, `attacc_e2e.json`): the round 78 comparator pointed at `r84` (`compare_attacc.py`, its lines in `installed_run.log`): all six kernel rows and the composed configurations identical to the `r57` baseline, 0 differences excluding timestamps and paths; the AttAcc suites 65 passed, 1 skipped |
| Shared semantics (whole) from the wheel | 1 failed, 156 passed (`installed_run.log`); the failure named by the rerun `installed/shared_whole_rerun.log` (1 failed, 156 passed): `test_local_accumulator_gemm_executes_on_apu_v1_functional_backend`, the deferred V1-19 test under the venv's NumPy 2.5.4, identical to the `d120084` and `92d4120` runs; the grouped route's section 19 passed |
| Gemini-II (zhang-x4, `g2-card-tests.log`, `g2-candidate-session.log`, `g2_candidate_rows.json`) | the card tests 15 passed; the one pre-registered validation session: status pass, 210 launches, no timing problem, every output exact, the same program and layout fingerprints and packing attestations as the `d120084` and `92d4120` sessions for all five rows; medians against the paper: gemm 1,728.525 / 1,733.225 (0.9973), batched_gemm 1,729.5 / 1,739.15 (0.9945), gesummv 1,841.575 / 1,860.075 (0.9901), bicg 2,538.425 / 2,548.825 (0.9959), mvt 2,770.25 / 2,766.625 (1.0013, the paper inside the IQR 2,760.5 to 2,778.1); under the rule the statistic at this revision is this session's samples: four rows REPRODUCED at `88852b7` (gemm: X-34, beside the earlier sessions, not pooled), mvt OPEN (nominal) at `88852b7` (X-37); the bounded confirmation campaign stays pre-registered for the final candidate and was not launched |
| Gemini-I | not launched: the board host's `/tmp` at about 108 KB free; `benchmarks/apu_v1/paper_rows.py` cannot run until the host's `/tmp` is usable; the three open rows stay explicitly blocked |
| Inventory (`update_inventory.py`, byte-exact CRLF edit, 84 lines, 18 columns) | 55 required rows recorded at `88852b7`: the fourteen AiM, three Samsung, sixteen UPMEM and twelve AttAcc rows REPRODUCED from the installed wheel, the Gemini-II rows from the session (four REPRODUCED, mvt OPEN nominal), the five Gemini-I rows noted as not re-executed; snapshot 51 REPRODUCED, 4 OPEN (kmeans, linear regression, word count, mvt), 0 UNVERIFIED |
| Not done | the independent review of the manifest and slice 5 packet (rounds 28 to 30; G9); the Gemini-I rows (the board host's `/tmp`); the X-34 / X-37 campaign (the final candidate); V1-19 (deferred); the packaging, licensing and final-candidate gates |

## Round 85 (2026-11-06): the `/tmp` recovery and the Gemini-I execution of the candidate `88852b7` (pre-registration, committed as `093f675` at 12:16:31 EST, before the script started at 12:16:33 and its first run at 12:16:37; the heading's earlier "12:20" was a drafting-time estimate, corrected in round 86)

The reviewer accepted the round 84 evidence, the manifest source review
(rounds 28 to 30) and supplied current Gemini-II slice 5 evidence
(`reviewer-r84/`); the remaining execution gap is the candidate's APU v1
vector path on the Gemini-I board, blocked by the host's full `/tmp`.

**What blocked.** `ledag-ssh` is a PyInstaller bundle: it extracts itself
into a temporary directory at start (`TMPDIR`, else `/tmp`) and writes a
session log and an `online_log_<time>` copy of the device log (about 5 MB)
into `/tmp` on every read, with no option to move them (`ledag-ssh --help`
offers only `--host`).  With `/tmp` at a few hundred kB free the extraction
fails (`fwrite: No space left on device`, exit 255; `r85/ledag_diag/`),
and with it above that but below about 5 MB the read fails the same way.

**What was recovered (bounded, owned scratch only).** Every top-level
`/tmp` entry owned by this user except the runtime's device lock was copied
to `tenon-release-evidence/tmp-relocated-20261106/` with `cp -a`, each entry
compared with `diff -rq`, and only the entries that compared identical were
removed (2,001 entries; 46 sockets and lock files kept; 22 entries kept
unverified, among them `tenon-upmem`, whose Docker-era files are root-owned
and cannot be removed; `REMOVED.txt`, `removed.txt`, `kept.txt`).  The
ledger's early `/tmp` citations (`/tmp/tenon-release/r9`, `/tmp/g9`,
`/tmp/upmem-fix`, the two Samsung reviewer scripts) are preserved under the
relocated tree at the same relative paths.  Free space went from 552 kB to
174,148 kB.  Nothing owned by another user was touched; the remaining ~15 GB
is other users' unreadable directories and stays an external condition, so
every run below is guarded on free space and recorded whatever happens.
The vendor tool's self-extraction is pointed at the evidence tree through
`TMPDIR`, a documented PyInstaller setting; its per-read `/tmp` scratch is
removed by the runtime after each read as before.

**What happened meanwhile.** The round 76 watcher fired on its own at
11:34 EST when another user's deletion briefly left 288 MB free, and ran
session 1 of the superseded `472bec9` campaign: kmeans kept 5 of 10 measured
launches per lane before the partition refilled (every kept output exact,
the driver marks the performance evidence invalid), linear regression and
word count lost every launch, sessions 2 and 3 were guarded out
(`r76/apu_v1_campaign.log`, `apu_v1_campaign_s1.json`, exit 3 at 12:01).
It is recorded under X-22 / X-23 as a failed session at `472bec9`, not
pooled with anything, and the watcher was ended so that no further launch
happens at a superseded revision.

**The runs (`r85/apu_v1_candidate.sh`, one pass, in this order, from the
round 84 wheel `9d7b6be6...` of `88852b7`, under the runtime's board lock,
`--scratch` under `r85/scratch`, outputs retained per launch, each run
guarded on 25 MB free):**

1. `protocol_rows`: kmeans, linear regression, word count, the round 30
   protocol (one warm-up, ten measured launches per lane, the expert binary
   and the archived paper-era Tenon binary interleaved, exact outputs
   required).  This is the candidate's validation session; the statistic
   at `88852b7` is this session's samples.  It is not the pre-registered
   final-candidate confirmation campaign (three sessions at the final
   candidate), which stays reserved; no further session is run here to
   move a verdict.
2. `vector_rows`: matrix multiply (the repaired vector route at its
   accumulator-block plan) and histogram, the same protocol.
3. `forced_acc4`: matrix multiply forced to
   `temporal_dma_coalescing_broadcast_friendly_acc4`, five measured
   launches (predicted slower than acc8).
4. `perturbed_dma`: matrix multiply with `MICRO_DMA_L4_L1_32K` x0.01,
   five measured launches (the selection is predicted to move to
   `temporal_dma_coalescing` and to run slower than the stock pick).

Acceptance: every launch's outputs equal the oracle; rows judged by the
rule in PAPER_REPRODUCTION.md on this session's samples; the forced and
perturbed runs are mechanism evidence (different device source, different
fingerprint, the predicted ordering), not paper samples.  A session that
loses launches to `/tmp` is kept, recorded as incomplete and not repeated
in this round.

### Results (read after the pass ended, 12:35 EST)

| Check | Result |
|---|---|
| `/tmp` after the recovery | 174,148 kB free at the launch (12:16:37); by the kmeans expert lane's third measured read (about 12:20) the vendor tool was failing again, and at 12:25 the partition held 100 kB free; it stayed at 100 kB through a two-minute sample (12:27 to 12:29). No entry owned by this user was created in `/tmp` in that window (the runtime removed the tool's two files after each read as designed), and no readable entry grew; the partition's 15 GB is in 128 unreadable directories of other users (the largest counts: `vb354` 63, `ck795` 37, `ss3679` 33), 2,886 of 25,800 inodes in use, so a few large files hold it. The consumer of the freed 174 MB cannot be identified or removed from this account. **Precise external blocker:** the board host's shared `/tmp` is refilled to within 100 kB by processes of other users within minutes of any space appearing; `ledag-ssh` needs about 5 MB there per device-log read and cannot be redirected; until the partition is freed by its owners or the host's administrators, every Gemini-I launch loses its profile read. The recovery was bounded as pre-registered (this user's own scratch only, by verified copy) and is not repeated |
| `protocol_rows` (`apu_v1_protocol_rows.json`, exit 1, 12:16 to 12:25) | kmeans: 33 launches recorded, every launch whose outputs were read equal to the oracle (11 of 11), 3 valid measured launches of 10 on the Tenon lane (median 624,448 against the paper's 624,677, 0.9996), 3 on the archived paper-era lane, 2 on the expert lane, the rest lost to `ledag-ssh exited with 255`; the driver marks the performance evidence invalid, so the session is incomplete and establishes no verdict at `88852b7` (recorded under X-22 / X-23, not pooled). linear regression and word count: every Tenon launch lost to the same read failure; on the expert and archived lanes the first attempts of each case (linear regression: warm-up through measured 2 on the expert lane, warm-up through measured 3 on the archived lane) were lost to the same `ledag-ssh` read failure, and the later attempts to a harness defect: with `--scratch` given for a multi-case session the lanes reused one set of launch directory names across cases, so once the first case had created a directory name every later attempt at that name failed with `FileExistsError` (X-38, fixed below); word count's expert and archived lanes collided from their warm-up launches onward. Both causes stand; neither lane produced a valid sample |
| `vector_rows`, `forced_acc4`, `perturbed_dma` | not started: guarded out at 12:25 with 100 kB free (`GUARD` lines in `apu_v1_candidate.log`). The repaired vector route therefore still has no current board execution |
| X-38 (harness) | `benchmarks/apu_v1/paper_rows.py::measure_case` gives each case its own subdirectory of a supplied scratch; `tests/perf/test_measurement_drivers.py` 19 passed with the new two-case virtual test, which fails without the fix (`pytest_measurement_drivers.log`); no measurement rule changed |
| Installation documentation | `INSTALL.md` section 5 now states the vendor tool's temporary-directory behaviour (self-extraction into `TMPDIR`, fixed `/tmp` writes of about 5 MB per read, no option) |
| Not done | the Gemini-I rows and the vector route's board execution at `88852b7` (external blocker above); the final-candidate campaigns (X-22 / X-23, X-34 / X-37); the remaining packaging, licensing and final-candidate gates; V1-19 (deferred) |

## Round 86 (2026-11-06): failure-path coverage for X-38; the lint chain whole (P-24, P-23)

Checkout after `7fc5e3a`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r86/`.  No board or
card launch; the host's `/tmp` is unchanged (100 kB free).

| Check | Result |
|---|---|
| X-38 failure-path regression (`tests/perf/test_measurement_drivers.py::test_apu_v1_failed_baseline_launches_of_two_cases_do_not_collide`) | the reviewer's probe as a test: `TenonLane` replaced by a stub, the board lock by a null context, `launch_on_board` by a launcher that fails, so every launch directory is retained; two cases with the expert and archived lanes against one scratch: eight attempts, eight retained `error.txt`, no `FileExistsError`, each launch directory directly under its case's subdirectory; with the parent expression restored in place both scratch tests fail (2 failed); the driver suite 20 passed (`pytest_measurement_drivers.log`) |
| P-24 | header added to `benchmarks/attacc/scripts/build_ramulator_attacc.sh`; `check_license_header.py HEAD~1` and `origin/main`: all checks passed (`check_license_header_HEAD1.log`, `task_lint.log`) |
| The lint chain whole (`task_lint.sh`; `task_lint.log` before, `task_lint_after_black.log` after formatting) | before: header passed, black listed 31 fork files (`black_files.txt`), chain stopped (exit 1); after reformatting those 31 files with black only: header passed, black 33 and 261 files unchanged against `HEAD~1` and `origin/main`, clang-format passed against both, pylint 629 findings rated 9.84/10, exit 30 (P-23's open stage). The `OSError: Device or resource busy` lines are black's multiprocessing temporary directory on the NFS `TMPDIR` being removed, not a formatting result |
| Behaviour neutrality of the formatting | every reformatted file compiles; the suites covering them (measurement drivers, simulator recipes, AttAcc compose and trace check, resource budget, host compute, the three Samsung declaration suites, AiM profile calibration, the token-partition and portability drivers, Samsung performance): 172 passed, 6 skipped in 2023.94 s (`pytest_reformatted_modules.log`; the run was still executing when the round 86 commit was made and this cell then said so; the reviewer confirmed the same count and that the 31 files' Python ASTs are identical before and after `2bdd298`); the long shared-semantics suite was not re-run for a formatting-only change |
| P-04 | `tests/perf/test_simulator_recipes.py` 24 passed: the release tree's `apply.sh` refuses without the five Samsung-derived patches, which stay outside the payload pending Samsung's permission; no distributable form of the patches exists that does not carry Samsung's lines |
| Ledger corrections | the round 85 pre-registration heading now cites commit `093f675` at 12:16:31 (before the 12:16:33 start and 12:16:37 first run); REVIEW round 85 gives the per-lane counts 3 / 3 / 2; VALIDATION round 85 distinguishes linear regression's early baseline launches lost to the `ledag-ssh` reads from the later ones lost to the collision |
| Not done | the Gemini-I rows and the vector route's board execution at `88852b7` (the external `/tmp` blocker of round 85, unchanged); the final-candidate campaigns; P-23's pylint stage; P-04 (permission); V1-19 (deferred); the public-payload review and the remaining packaging and final-candidate gates |

## Round 87 (2026-11-06): X-39 (stale export); the E-class pylint triage; the payload audit begun

Checkout after `1b8a6ff`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r87/`.  No board or
card launch; the host's `/tmp` unchanged.

| Check | Result |
|---|---|
| X-39 | `from allo.pim.apu_v1_record_frequency import *` works; `tests/pim/test_public_exports.py` 57 passed, 22 skipped (modules without `__all__`), 2 failed with the stale export restored (`pytest_public_exports.log`); a scan of every `__all__` in the whole `allo` package found no other missing name (the four modules that do not import on this host, `_mlir.libAlloMLIRAggregateCAPI`, `backend.aie`, `frontend.tracer`, `harness.makefile_gen.descgen`, are upstream Allo modules outside the PIM path and were not scanned) |
| P-23 E-class triage (`pylint_e_triage.md`) | 49 E diagnostics: 1 confirmed defect (E0603, X-39), 48 false positives demonstrated at runtime (PEP 604 unions, guard-narrowed subscripts, names of the compiled MLIR extension that import, one guarded negation counted five times, the `ConstantOp(result, value, *, loc, ip)` constructor); the 580 W/C/R diagnostics untriaged |
| Payload audit of the round 84 sdist (`sdist/`, from `r84/pkg/dist/allo-0.5.tar.gz`) | 325 regular files (the 387 first written here counted tar members, directories included; corrected in round 88 after the reviewer's count): `allo` 193 (150 Python files, every one with the Apache-2.0 header; the runtime assets `MANIFEST.in` names: `.mlir` templates, AIE kernels, `harness/`, `pim/templates/`), `mlir` 103 (the dialect and binding sources the sdist rebuilds), `simulators` 14, and 15 build and metadata files; no `tests/`, `benchmarks/`, `dev/`, evidence, lock, wheel, binary or `.so` file; `simulators/PIMSimulator/` carries only Tenon's conductor sources, `apply.sh` and `Sconstruct` (the five Samsung-derived patches absent, as pruned); `simulators/uPIMulator/tenon-upimulator.patch` modifies 47 CMake, README and benchmark files of the MIT-licensed upstream at its pinned commit `870d916` (its README names the license and the upstream); no packaged file contains a `/work/...` path or this user's name. One observation: `allo/pim/apu_g2_pipeline.py` carries `APUG2_TYPED_TICK_PROVENANCE`, a provenance tuple naming the lab host `en-ec-zhang-x4`, the card, the firmware and evidence log paths (`tenon-release-evidence/g2/...`) that are not shipped; it is documentation of where the typed-tick calibration numbers came from, not a dependency, and is recorded here for the distribution review to judge. Not yet audited: the wheel's installed file set against the sdist, the `mlir/` tree's third-party notices, `simulators/PIMSimulator/Sconstruct` (P-04's open question), `README.md` and `INSTALL.md` for lab-specific instructions |
| Not done | the Gemini-I rows and the vector route's board execution at `88852b7` (the external `/tmp` blocker, unchanged); the final-candidate campaigns; the rest of the payload audit; P-23's W/C/R classes; P-04; V1-19 (deferred) |

## Round 88 (2026-11-06): third-party license texts bundled (P-25); the distribution audit finished for this increment

Checkout after `8689630`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r88/`.  No board or
card launch; the host's `/tmp` at about 96 KiB free, unchanged.

| Check | Result |
|---|---|
| Third-party inventory of the round 84 archives (`wheel/`, `sdist_allo_files.txt`, `wheel_allo_files.txt`) | wheel: 422 members; `allo/_mlir/` 223 files = the MLIR Python bindings (149 `.py`, LLVM-headed, plus Allo's own `__init__.py` and `exceptions.py` under Apache-2.0), the C API headers, `libAlloMLIRAggregateCAPI.so.22.0git` and 20 `_*.so` extension modules; `strings` finds `nanobind` in all 20 extension modules and no `pybind11`; `readelf -d` on the aggregate library: `libz.so.1`, `libzstd.so.1` and the C/C++ runtimes linked dynamically, nothing else bundled (`LLVM_ENABLE_LIBXML2`/`LIBEDIT` are build options of the LLVM tree, no such library is linked). sdist: 325 regular files; LLVM headers retained in `mlir/lib/Dialect/AlloOps.cpp` and `AlloDialect.cpp` (the other 103 `mlir/` sources carry Allo's Apache-2.0 marker, none is unmarked); the uPIMulator patch (47 upstream files, MIT at commit `870d916`); Tenon's PIMSimulator conductor sources only |
| Wheel versus sdist | the 193 `allo/` files of the sdist and the 193 non-`_mlir` `allo/` files of the wheel are the same set; the wheel adds the 223 `_mlir` files `build_ext` copies from the CMake build; nothing in the sdist is missing from the wheel |
| README and INSTALL | `README.md` carries no lab-specific path or host; `INSTALL.md` names `/opt/gsi` as the vendor SDK's default with the `TENON_APU_G2_SDK_ROOT` override, and its section 6 fresh-install transcript quotes this host's interpreter and LLVM paths as the transcript of one host, labelled as such |
| Packaging change (commit `7637768`) | `THIRD_PARTY_LICENSES/{LLVM-LICENSE.TXT,NANOBIND-LICENSE.txt,UPIMULATOR-LICENSE.txt}` (sha256 `8d85c105...`, `f305091a...`, `73de9527...`), `THIRD_PARTY_NOTICES.md`, `setup.py` `license`/`license_files`, `MANIFEST.in`; `tests/perf/test_distribution_licenses.py` 3 passed (`pytest_distribution_licenses.log`) |
| Archives rebuilt (`build_wheel.sh`, `build_wheel.log`) | first from `7637768` (`pkg_attempt1_7637768/`): the sdist and the wheel carried `LICENSE`, `THIRD_PARTY_NOTICES.md` and only `THIRD_PARTY_LICENSES/LLVM-LICENSE.TXT`; the nanobind and uPIMulator texts were missing because the repository's `*.txt` ignore rule (`.gitignore:90`) had kept them out of the commit and `git archive` packages the commit. Fixed in `cf43398` (the two files tracked, an ignore negation for the directory, `test_every_bundled_text_is_tracked_by_git`, 4 passed, the new check failing before the fix). Rebuilt from `cf43398` (`pkg/`): sdist `b6e17ca4...`, 329 regular files, members `LICENSE`, `THIRD_PARTY_NOTICES.md`, `THIRD_PARTY_LICENSES/{LLVM-LICENSE.TXT,NANOBIND-LICENSE.txt,UPIMULATOR-LICENSE.txt}`; wheel `ce11a89c...`, 426 members, `allo-0.5.dist-info/licenses/` holding the same five files with the bundled checksums (`8d85c105...`, `f305091a...`, `73de9527...`), `METADATA` `License: Apache-2.0` and five `License-File` entries; the wheel's `allo/` file set identical to the round 84 wheel's (the packaging change adds nothing else) |
| Not done | the Gemini-I rows and the vector route's board execution at `88852b7` (the external `/tmp` blocker, unchanged); the final-candidate campaigns; P-04 (permission; its `Sconstruct` question stays open); P-23's W/C/R classes; V1-19 (deferred); the next named candidate must contain `cf43398` (not `7637768` alone, whose archives lacked two of the texts) and, after round 89, `aae8b23`, for its installed artifacts to carry the complete notices |

## Round 89 (2026-11-06): P-25 completed after the reviewer's symbol-level verification; the inventory corrected

Checkout after `b201db4`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r89/`.  No board or
card launch; the host's `/tmp` 100 percent full (about 128 KiB free at the
reviewer's observation), unchanged.

| Check | Result |
|---|---|
| What round 88 omitted (`reviewer-r88/payload_verification.txt`, confirmed here by `nm -C` on the round 88 wheel) | robin-map template instantiations (`tsl::detail_robin_hash::` symbols in all 20 extension modules, 286 in `_allo`, 405 in `_mlir`), `llvm::xxHash64` / `xxh3_64bits` (4 symbols) and `llvm::ConvertUTF*` (13 symbols) in the aggregate library; none of the bundled texts carried their notices. The enumeration was then extended over the aggregate library's symbols: `llvm_regcomp` / `llvm_regexec` / `llvm_regfree` (3, Henry Spencer's regex), `llvm::BLAKE3` / `llvm_blake3_*` (44), `llvm::MD5` (24), `llvm::SHA1` (14, LLVM's adaptation of public-domain code under the LLVM header), SipHash (6, under the LLVM header), the Itanium and Microsoft demanglers (872, LLVM's own), `llvm::compression::zlib` / `zstd` (dynamic, not bundled); no `absl::` or Google code |
| UPMEM runtime hunks of the uPIMulator patch | the hunks at patch lines 339 to 430 modify `golang/uPIMulator/sdk/misc/CMakeLists.txt`, `sdk/stdlib/stdio.c`, `sdk/stdlib/CMakeLists.txt`, `sdk/syslib/CMakeLists.txt`, `sdk/syslib/dpuruntime.h` and add `sdk/misc/stdoutBufferState.c` under the retained "Copyright 2020 UPMEM" notice; the upstream checkout's `sdk/stdlib/stdio.c` and `sdk/syslib/dpuruntime.h` carry that notice, and the same two files exist in the UPMEM SDK 2023.1.0 `dpu-rt` tree whose `LICENSE` (BSD 3-Clause, "Copyright (c) 2020, UPMEM", sha256 `8f49c23e...`) is now bundled; the notices and the simulator README distinguish these hunks from the MIT components |
| nanobind provenance | the rebuilt tree's `mlir/build/CMakeCache.txt` names `NB_DIR` in the spmw environment (nanobind 2.13.0) and `build.ninja` builds `nanobind-static` from it for the `_allo` module, while the shared LLVM build's cache names 2.10.2 for the nineteen `_mlir*` modules; both origins are recorded, and the notice says a future supported build embeds whatever `setup_requires` resolves |
| The bundle (commit `aae8b23`) | `THIRD_PARTY_LICENSES/` holds ten texts: the three of round 88 plus `ROBIN-MAP-LICENSE.txt`, `XXHASH-LICENSE.txt`, `CONVERTUTF-UNICODE-NOTICE.txt`, `LLVM-REGEX-COPYRIGHT.txt`, `BLAKE3-LICENSE.txt`, `MD5-NOTICE.txt`, `UPMEM-DPU-RT-LICENSE.txt`, each extracted or copied verbatim from the upstream file named in the inventory; `THIRD_PARTY_NOTICES.md` rewritten (components, placement, license, text, checksums, the symbol-based method, the not-shipped list); `tests/perf/test_distribution_licenses.py` pins the ten-file set, the phrases of each text, the inventory entries, the two nanobind origins, the UPMEM distinction and the git tracking of every file (4 passed, `pytest_distribution_licenses.log`) |
| Archives rebuilt from `aae8b23` (`build_wheel.sh`, `build_wheel.log`, `pkg/`, `wheel_new/`, `sdist_new/`, `*_license_shas.txt`) | sdist `0a6746e9...`, 336 regular files, carrying `LICENSE`, `THIRD_PARTY_NOTICES.md` and the ten texts; wheel `7235d510...`, 433 members, `allo-0.5.dist-info/licenses/` holding `LICENSE`, the notices and the ten texts, `METADATA` with `License: Apache-2.0` and twelve `License-File` entries; every bundled text in both archives byte-identical to the repository's (sha256 lists compared); the wheel's `allo/` file set identical to the round 84 wheel's |
| Not done | the Gemini-I rows and the vector route's board execution at `88852b7` (the external `/tmp` blocker, unchanged); the final-candidate campaigns; P-04 (permission); P-23's W/C/R classes; V1-19 (deferred); the next named candidate must contain `aae8b23` |

## Round 90 (2026-11-06): P-23 W0640 triage (compile-only); no board launch

Checkout after `3a16830`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r90/`.  The host's
`/tmp` at 128 KiB free, unchanged; nothing launched.

| Check | Result |
|---|---|
| W0640 escape analysis (`w0640_escape_analysis.txt`) | over `spmw_autoschedule.py`, `spmw_plan.py`, `pim/apu_v1_vectorize.py`, `spmw_samsung.py`, `pim/upmem_irregular.py`, `pim/costs/upmem.py`: eight closures defined inside loops capture loop-scoped names: five are called directly in their loop body (`kernel_probe` at 2327 and 2336, `binding` at 647, `value_layout` at 840 to 842, `input_transfer` at 1281 to 1282, `layout_of` at 1314 to 1431), one is a `key=lambda` consumed synchronously by `min` (`upmem_irregular.py:1342`), and two are decorator registrations (`costs/upmem.py:782`, `:842`) that the cost rule table retains for later execution, safely, because they bind every loop value through default arguments evaluated at definition time; none of the five direct helpers or the lambda is stored, returned or passed to another function (corrected in round 91: the earlier text said six direct helpers and that none escapes) |
| Captured-cell check (`w0640_cells.txt`) | for the five directly called closures, every captured name is assigned before the definition and not reassigned before the last call; the only later reassignments (`label`, `cost` at `spmw_autoschedule.py:2410`, nested, and `:2494`) follow the closure's last call and the closure is referenced nowhere after |
| Verdict | 89 diagnostics, 0 defects, 0 suppressed; 491 W/C/R diagnostics remain untriaged (ISSUES P-23) |
| Not done | the Gemini-I rows and the vector route's board execution at a candidate containing `aae8b23` (the external `/tmp` blocker, unchanged); the final-candidate campaigns; P-04; the remaining W/C/R classes; V1-19 (deferred) |

## Round 91 (2026-11-06): P-23 unused-import triage and removal (compile-only); no board launch

Checkout after `1dfb9db`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r91/`.  The host's
`/tmp` at 124 KiB free; nothing launched.

| Check | Result |
|---|---|
| Triage (`w0611_diagnostics.txt`, `w0611_triage_raw.txt`, `w0611_triage.md`) | 20 diagnostics in twelve fork files; for each: other mentions in the file, `__all__`, `TYPE_CHECKING`, consumers importing the name from the module, attribute-style users, star imports, test monkeypatches; all 20 genuinely unused, none a public re-export, a registration side effect or a monkeypatch target |
| Removal | the 20 names removed from their import statements (two function-local re-imports, two names inside multi-line statements, the rest whole lines); 12 files, 4 insertions and 20 deletions before black, formatting unchanged otherwise; every file compiles and imports; `pylint allo --disable=all --enable=unused-import`: 0 (`pylint_w0611_after.log`) |
| Focused suites over the twelve modules (`focused_suite_files.txt`, `pytest_focused_after_unused_import_removal.log`) | 204 passed, 22 skipped, 1 failed: `tests/perf/test_apu_g2_source_route.py::test_device_lock_is_exclusive_and_reentrant`, which deliberately creates its lock under the local `/tmp` (NFS rejects `flock` on a read-only descriptor) and fails there with `OSError: [Errno 28] No space left on device`; the same test fails identically at the committed tree with the edits stashed, so it is the host's full `/tmp`, not the import removal; every other test of the thirteen files passes |
| Round 90 wording | VALIDATION round 90 and REVIEW round 90 corrected: five direct helpers, one lambda, two decorated registrations retained for later execution and safe through default-argument binding |
| Not done | the Gemini-I rows and the vector route's board execution at a candidate containing `aae8b23` (the external `/tmp` blocker, unchanged); the final-candidate campaigns; P-04; the remaining 471 W/C/R diagnostics; V1-19 (deferred) |

## Round 92 (2026-11-06): P-23 unused-argument triage (compile-only); no board launch

Checkout after `74dc03e`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r92/`.  The host's
`/tmp` at 124 KiB free; nothing launched.

| Check | Result |
|---|---|
| Triage (`w0613_diagnostics.txt`, `w0613_sites.txt`, `w0613_triage.md`) | 27 diagnostics; for each the function's role, registration or overrides, callers and sibling signatures, and whether the ignored value should have influenced the result; 19 required signatures kept, 1 missing resource dependence repaired, 7 vestigial parameters removed with their callers; nothing suppressed; `pylint allo --disable=all --enable=unused-argument`: 19, the kept set (`pylint_w0613_after.log`) |
| The repair: `samsung_staging_share` | `cols_per_row` now derives the physical rows a staging run spans (`ceil(G*E / cols_per_row)`, and `ceil(G*E/2 / cols_per_row)` per bank for the split map), with one exposed row change per row boundary and one row visit per row; at the simulator's 32 columns per row the figures equal the former hard-coded ones, so the whole-pass cross-check against `samsung_staging_schedule` is unchanged; `tests/perf/test_samsung_performance.py::test_staging_share_follows_the_geometry_columns_per_row` pins the default, 16 and 64 columns per row, the split map and the rejection of a non-positive value (1 passed alone) |
| Focused suites over the touched modules (`focused_suite_files.txt`, `pytest_focused_after_w0613.log`) | not established in round 92 (the run was interrupted by the runtime migration and the reboot; the log was empty); run and recorded in round 93 |
| Not done | the Gemini-I rows and the vector route's board execution at a candidate containing `aae8b23` and these changes (the external `/tmp` blocker, unchanged); the final-candidate campaigns; P-04; the remaining 444 W/C/R diagnostics; V1-19 (deferred) |

## Round 93 (2026-11-10): round 92 finished (the conductor-test failure explained and repaired); the Samsung simulator bundled (P-04); X-40 found and repaired

Checkout after `74dc03e` (the round 92 edits inherited uncommitted and
finished here); evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r93/`.  The host's
shared `/tmp` at about 7 MB free throughout (`df -k /tmp`); every run here
used `TMPDIR` on the evidence tree; nothing launched on the board.
Environment: `/work/zhang-capra/users/nz264/miniforge3/envs/spmw/bin/python`,
`LLVM_BUILD_DIR=/work/shared/common/llvm-project-main/build-rhel8`, the
installed PIMSimulator through the checkout's `experiments/simulators`.

| Check | Result |
|---|---|
| The round 92 conductor-subset failure (`r92/pytest_conductor_runs_subset.log`: `test_samsung_conductor_runs.py:200`) | reproduced before any change (`pytest_tampered_crf_first.log`): `test_tampered_crf_is_rejected_not_canonicalized` expects the tampered CRF (word 1, the odd-bank MAC, replaced by `JUMP x3 offset 2`) to be rejected for its MAC-group count, but the conductor rejects it with "CRF program consumes more than 20480 column commands without reaching EXIT". Cause: with two JUMPs the walk's `lastJumpIdx_` rule (PIMRank::doPIM's, which `tenonWalkCrf` mirrors) makes each JUMP reset the other's counter, so the tampered program never reaches EXIT; the conductor's column-command budget (64 x (T+2) x (G+2) x 8 = 20480 for the plan's 2 tiles and 8 groups, `TenonConductor.inc:454`) stops it. The rejection is correct and the test's expectation was stale (the test file last changed in `3418938`, before the budget guard of 2026-11-04); not caused by the round 92 edits, which do not touch the conductor. Repaired in the test: a terminating tamper (the tile loop `JUMP x7` shortened to `x3`) is rejected for its group count (`MAC groups on fiber 0`), and the non-terminating tamper for the budget (`column commands without reaching EXIT`); both asserted |
| `tests/pim/samsung_hbm_pim/test_samsung_conductor_runs.py` whole file (`pytest_conductor_runs.log`) | 8 passed, 1 failed in 519 s: the repaired tampered-CRF test passes; the failure is `test_gemm_matrix_residency_lever_changes_staging`, which fails identically at the committed `74dc03e` in a detached worktree with the same simulator (`pytest_gemm_residency_baseline_74dc03e.log`): pre-existing, diagnosed as X-40 below, repaired, and passing after the repair (`pytest_gemm_residency_fixed.log`, 1 passed in 306 s: both residency settings execute correctly, 1 vs 4 matrix preloads, the resident program faster, the default selects the resident matrix) |
| Focused suites over the round 92 modules (`r92/focused_suite_files.txt`, `pytest_focused_after_w0613.log`) | 161 passed, 4 skipped, 3 failed in 2601 s, run in round 93 after the interruption (`r93/pytest_focused_after_w0613.log`; the round 92 copy of this log was empty): the three failures are the S-17 Samsung decision tests of `tests/pim/test_g6_decisions.py` (`test_samsung_host_fill_cost_flip_reaches_execution`, `test_samsung_fiber_term_removal_keeps_bank_row_by_construction`, `test_samsung_host_burst_cost_shows_the_documented_list_scheduling_anomaly`), open since round 39 with the same observed values (`split_rows`+host selected, estimates 18,646 / 18,654 / 18,662), so not introduced here; the staging-share repair leaves the default geometry's figures unchanged, as the whole-pass cross-check test shows |
| X-40 diagnosis (`gemm_probe.log`, `gemm_probe2.log`, `gemm_probe3.log`, `gemm_probe4.log`, `gemm_probe5.log`) | the test GEMM `acc += lB[k, j] * lA[row0 + i, k]`: shared roles `matrix='x'` (lB), `vector='y'` (lA), `unresolved: vector index ['%21'] may depend on result loop %arg3`; every enumerated REDUCE candidate declared `lB grf_a half_bank:0,0` and `lA banks None`; the default compile chose position 49 `temporal+crf_shared` at 181,942 cycles while the ranking's best was 18,630 (`split_rows_rowpair+crf_per_workid+wresident`, itself rejected for the lockstep CRF as designed); 98 rejections, of which 28 "realization differs from the candidate contract: Samsung realization stages the matrix rows as 'half_bank:0,0'/'0,1'; the candidate declared packing None"; the same GEMM written `lA[row0 + i, k] * lB[k, j]` resolves its roles and selects `split_rows+crf_shared+wresident` at 18,630 cycles with 6 rejections (none for packing) |
| X-40 repair | `samsung_candidate_effects` derives the bank-resident operand from `samsung_contraction(match).matrix_memref` (the REDUCE lowering's own derivation) instead of the shared `matrix` role; `tests/pim/samsung_hbm_pim/test_samsung_lowering.py::test_gemm_operand_order_does_not_change_the_selected_program` compiles both spellings: both REDUCE, matrix `lA` in the banks with the packing the realization reports, `lB` in GRF_A, resident matrix, `split_rows` selected, identical CRF words, binds, tiles and fibers, equal best and chosen cycles, no "declared packing" rejection (`pytest_gemm_operand_order.log`, 1 passed in 134 s). The shared role analysis's own order dependence is recorded open as X-41 |
| P-04: the bundled Samsung simulator (`bundled_build.log`) | `simulators/PIMSimulator/upstream/` extracted from the authors' upstream object store at `3703d1f` (70 files, 772 KiB; `data/`, `dump/` and `.gitignore` excluded) and hashed into `UPSTREAM_MANIFEST.sha256`; against the installed simulator tree exactly the five patched files and the Tenon `Sconstruct` differ. `python -m allo.pim.samsung_simulator verify`: 70 files verified. `python -m allo.pim.samsung_simulator build --jobs 16` with `TENON_HOME` on the evidence tree: the first two attempts failed in the compiler with `No space left on device` because SCons runs the compiler with a minimal environment and the host's `/tmp` is full; the Tenon `Sconstruct` now passes `TMPDIR` through, after which the build took 12.8 s and printed `PIM_GEOMETRY channels=64 ranks=1 pim_blocks=8 banks=16 bank_groups=4 grf_a=8 grf_b=8 lanes=16 crf_words=32 cols_per_row=32 rows=16384`; `locate` reports the default root discovered. The built `pim_driver` (sha256 `055d6c24...`) and the authors' installed one (`6c50596b...`) are identical after `objcopy --strip-all --remove-section=.note.gnu.build-id` (`9aea49bffc5c04b23b033695a7bcc072fe24ac203ad349895dabee61d80ee21d`) |
| The bundled build runs the compiler's programs (`pytest_gemv_samsung_bundled_root.log`) | `PIMSIMULATOR_ROOT` at the bundled build, `tests/pim/test_gemv_all_targets.py -k samsung`: 1 passed in 48 s (the GEMV compiled, executed and checked on the conductor built from the bundle) |
| Recipe, bundle, license and packaging suites (`pytest_bundle_packaging.log`) | `tests/perf/test_simulator_recipes.py` (the manifest against every bundled file, `apply.sh` on a copy of `upstream/` changing exactly the five patched files plus the Tenon files and idempotent on a second run, refusals of an incomplete patch set and a non-source tree, a user's own patch set by argument or variable, the sdist shipping the recipe with the upstream copy and the patches), `tests/perf/test_samsung_simulator_bundle.py` (recipe discovery and override, manifest verification and a tampered copy's four problem kinds, assemble into a root and the refusal of a root inside the recipe, the default root under `TENON_HOME` used only once it holds `pim_driver`, the precedence of `PIMSIMULATOR_ROOT` and `TENON_SIMULATORS_DIR`, the XDG default, the `verify`/`locate` CLI, `setup.py build_py` copying the recipe into `allo/pim/simulators/PIMSimulator` without build outputs, and a real build into a temporary `TENON_HOME` with the geometry checked and a rebuild starting from pristine upstream), `tests/perf/test_distribution_licenses.py` (12 texts, the two new ones inventoried with checksums) and `tests/perf/test_packaging.py` (the unconfigured message names the build command and the default root): 68 passed, 1 deselected (the UPMEM calibration regeneration, unrelated) in 32 s |
| Formatting and lint | black on every changed Python file; `pylint allo/pim/samsung_simulator.py allo/spmw_simenv.py` with the E, unused-import, unused-argument, unused-variable and undefined-variable classes: 0 |
| Installed-artifact validation of the bundled route at the candidate `39449d6` (`build_wheel.sh`, `build_wheel.log`, `pkg/`, `installed-run/run.sh`, `installed-run/run.log`) | sdist from `git archive 39449d6` (`da4736dd2bc948af6d5dbc127ea8dbea179db7c8eca3fd6ccdc62cd1d91561cd`; 70 upstream files and 5 patches under `simulators/PIMSimulator/`), wheel `b190202706a6765f452d0114c0b154fcdb545befce1d5fbd0eb16af9f87f7750` (84 members under `allo/pim/simulators/PIMSimulator/`, `dist-info/licenses/` with the Samsung and DRAMSim2 texts), installed with `--no-deps` into a clean venv (runtime dependencies plus `scons` from pip). From a directory outside the checkout, `PYTHONSAFEPATH=1`, no `PYTHONPATH`, `LLVM_BUILD_DIR`, `PIMSIMULATOR_ROOT` or `TENON_SIMULATORS_DIR`, `TENON_HOME` on the evidence tree: `verify` finds the recipe inside `site-packages` and verifies 70 files; `locate` reports no simulator and the message names the build command; `build --jobs 16` assembles, patches and builds in 16.3 s and prints the geometry; `locate` then reports the default root available; the built `pim_driver` (`fe04c690...`) strips to `9aea49bffc5c04b23b033695a7bcc072fe24ac203ad349895dabee61d80ee21d`, the evaluation conductor's stripped identity; the probe `bundled_samsung_probe.py` imports `allo` from the venv, checks `pimsim_root()` equals the default root, compiles a 4096x256 FP16 GEMV (59 s, REDUCE) and executes it on the bundled conductor: 4,108 cycles, max abs error 0.0004 against float32 numpy, `pimcmdgen_calls` 0 |
| Not done | the Gemini-I rows and the vector route's board execution at a candidate containing `aae8b23` (the external `/tmp` blocker, unchanged: about 7 MB free, held by other users' directories); the final-candidate campaigns; X-41; the remaining W/C/R diagnostics; V1-19 (deferred) |

## Round 94 (2026-11-10): the reviewer's test-hole fix; the Gemini-II candidate-validation session pre-registered

Checkout after `f377ecb`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r94/`.  The host's
shared `/tmp` at about 6.6 MB free (unchanged); `zhang-x4` reachable over
the trusted SSH setup, `gsi_tool info apu-00`: Status Available, four cores
idle, FW `GSI_G2_Infrastructure_V1.24.1`, its `/tmp` 3.6 GB free.

| Check | Result |
|---|---|
| The reviewer's round 93 finding: `test_build_py_bundles_the_recipe_inside_the_package` turned every nonzero `setup.py build_py` exit into a skip | fixed in `fdee44b`: the only prerequisite (setuptools) is detected narrowly before the build; a nonzero exit is an assertion failure that includes the build's stdout and stderr; 1 passed (`pytest_build_py_assert.log`) |
| Named candidate | `fdee44b` (= `f377ecb` plus the test fix; no production change after `39449d6`). sdist from `git archive fdee44b`: `7476d78b121c91e39333075dfc37681203f6f7ab6bee1afcbb6d51182429c7b9` (`pkg/`, `dirty.txt` empty) |
| **Pre-registration (written before any launch): the Gemini-II candidate-validation session at `fdee44b`** | ONE paired session of 20 rounds of `benchmarks.apu_g2.paper_rows` from a wheel built on `zhang-x4` from the candidate's sdist and installed into a clean venv (`g2_build_wheel.sh`), after the card tests from that wheel (`g2_card_tests.sh`), under the device lock (`g2_candidate_session.sh`, launched by `g2_remote_chain.sh` over SSH). Purpose: the candidate's Gemini-II modules differ from the last named candidate `88852b7` only by the unused-import removals and black formatting of rounds 86-91 (`git diff 88852b7..fdee44b -- allo/pim/apu_g2_* allo/spmw_apu_g2.py benchmarks/apu_g2`), and the five required rows must be measured from this candidate's installed artifact. Recorded whatever the outcome; the statistic at this revision is this session's samples, reported beside the earlier revisions' verdicts and not pooled with them. This is not the X-34 / X-37 confirmation campaign, which stays reserved for the final candidate, and not a further sample for the exhausted `batched_gemm` campaign (X-26). Verdict rule per row, fixed now: REPRODUCED if the session median is at or below the paper's value with every output exact; otherwise OPEN (nominal) if the paper's value lies inside the session's interquartile range, OPEN (regression) if the paper's value lies below it (the whole central half of the session slower than the paper) |
| Storage | the shared filesystem's user quota reached its hard limit during the round (330 GB; the remote wheel build failed with `[Errno 122] Disk quota exceeded` while copying the aggregate library, and the first launch of the X-41 suites could not open its log). Removed: the extracted sdist build trees `r*/pkg/src` and `r*/g2/src` of rounds 9 to 94 (39 directories, about 30 GB; each is regenerable from the round's retained `allo-0.5.tar.gz` and wheel, whose hashes the rounds record); nothing else. Usage after: 301 GB (the soft limit of 300 GB still exceeded, grace period running) |
| The Gemini-II chain at `fdee44b` (`g2_remote_chain.sh` over SSH; `g2-build-wheel.log`, `g2-card-tests.log`, `g2-candidate-session.log`, `g2_candidate_rows.json`) | wheel built on `zhang-x4` from the candidate's sdist and installed into a clean venv: `5542eae474aebed3cd2396cfe3d370e86ee6819a6194f2ed98d4fd592411947a` (the first attempt failed on the quota above; relaunched after the cleanup). Card tests from that wheel (`tests/pim/test_apu_g2_hardware.py` under the device lock): 15 passed in 124 s. The pre-registered session (20 paired rounds, one warm-up per lane, `benchmarks.apu_g2.paper_rows` from the installed wheel, `TENON_ARTIFACTS` for the archived GTML host, `/tmp/tenon-apu-g2.lock` plus the runtime's own device lock): status `pass`, no problems, 210 launches with `outputs_exact` true in every one; the Tenon lane's plans `single/k1` (gemm, batched_gemm) and `fused/k1` (gesummv, bicg, mvt), the same cost fingerprint `1539818082e183c3` as the earlier sessions. Per row (median ticks per body, paper, ratio, session IQR, verdict under the rule fixed above): gemm 1,741.55 / 1,733.225 / 1.0048 / 1,735.8 to 1,748.6 / OPEN (regression); batched_gemm 1,735.45 / 1,739.15 / 0.9979 / 1,717.3 to 1,743.9 / REPRODUCED; gesummv 1,829.45 / 1,860.075 / 0.9835 / 1,826.3 to 1,842.4 / REPRODUCED; bicg 2,537.025 / 2,548.825 / 0.9954 / 2,531.9 to 2,550.5 / REPRODUCED; mvt 2,757.775 / 2,766.625 / 0.9968 / 2,753.1 to 2,770.3 / REPRODUCED. GTML in the same rounds: 1,915.75, 1,913.85, 3,108.675, 2,732.55, 3,099.45 (paired GTML/Tenon 1.1000, 1.1028, 1.6992, 1.0771, 1.1239 against the paper's 1.0984, 1.0962, 1.6617, 1.0697, 1.1168). Recorded in the inventory, `PAPER_REPRODUCTION.md`, X-34 and X-37 |
| X-41, first increment (`spmw_match_engine.operand_index_terms`, `_axis_roles`, `aim_lowering.mac_operands`) | `tests/pim/test_axis_roles_coordinates.py`: the helper on synthetic bindings (a resolved SSA row coordinate counts as its loop variable; a coordinate without a form and a non-loop atom stay opaque; the textual fallback unchanged) and both GEMM spellings compiled for Samsung resolve their roles (`unresolved` None, `matrix_by` `pattern_order`, the batch axis the result loop indexing the input named `vector`): 2 passed in 147 s (`pytest_axis_roles_coordinates.log`). AiM, AttAcc and UPMEM suites reading the roles (`test_aim_matcher`, `test_aim_program`, `test_aim_owner_maps`, `test_aim_dense_domains`, `test_cent_aim_workloads`, `test_attacc_trace_check`, `test_upmem_matcher`): 50 passed, 21 skipped (the `paper_full` rows) in 343 s (`pytest_x41_slice2.log`) |
| X-41, first increment: the shared-semantics, matcher, Samsung lowering, AttAcc and layout-selection suites (`pytest_x41_slice.log`: `test_aim_axis_roles`, `test_shared_semantics`, `test_match_coverage`, `samsung_hbm_pim/test_samsung_lowering` with the X-40 operand-order test, `test_attacc_virtual`, `test_attacc_compose`, `test_layout_selection`) | 278 passed, 1 skipped in 2405 s; no decision, role or realization change on any existing program |
| Not done | the structural two-sided definition of the shared `matrix`/`vector` roles and its AttAcc and layout-problem consumers (X-41, next increment); the Gemini-I rows and the vector route's board execution (the board host's `/tmp` at 6.6 MB, held by other users); the X-34 / X-37 confirmation campaign (reserved for the final candidate); the final-candidate validation; the remaining W/C/R diagnostics; V1-19 (deferred) |

## Round 95 (2026-11-10): the reviewer's access-contract finding (X-42) repaired in the shared stage; AttAcc storage orientations as candidates (X-41, second increment)

Checkout after `b578fc2`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r95/`.  The host's
`/tmp` at about 6.6 MB (unchanged); nothing launched on hardware.  The
shared filesystem at 303 GB of the 300 GB soft limit (grace running).

| Check | Result |
|---|---|
| The round 94 wording the reviewer flagged (`VALIDATION.md`, the pre-registered Gemini-II rule) | corrected: OPEN (regression) is the case where the paper's value lies below the session's interquartile range |
| X-42 reproduced and widened (`stride_backends.py`, `stride_backends.log`, `offset_probe.py`, `offset_probe.log`) | the reviewer's `sum_k a[i,k]*b[2*k]`: AiM accepted it with the commands of the plain program; AttAcc accepted it too (kernel rows 1, reduction 256, outputs 32, `PIM_WR_GB count=1024`); Samsung refused it (`must read the reduction loop unshifted in exactly one coordinate`); UPMEM's matcher did not match it. AiM also accepted `b[k + 1]`, `b[k + 128]` and `a[i, k + 8]` with the plain program's commands and preload (`offset_probe.log`): constant shifts of the reduction coordinate are ignored on either operand |
| X-42 repair (`spmw_semantics.contraction_access_problems`, `spmw_codegen.require_contraction_accesses_realizable`) | one shared check in `compile_for_target` before the virtual-only branch and before any backend's candidates. Accepted coordinate forms: one loop variable; a loop variable plus a constant on a result loop; a dense row-major flattening of several loops (outer coefficients equal to the products of the inner trip counts). Refused with the function, operand, coordinate and the form read: an innermost coefficient other than one, a padded flattening, a constant shift of a reduction loop. The first version refused every multi-loop coordinate; the suites caught it on the CENT decode region's `xs[p*Q + k]` and the double-reduction GEMV `lx[jo*KI + ji]` (both dense flattenings the lowerings realize), and the rule was corrected before the commit. `tests/pim/test_contraction_access_forms.py`: 5 passed in 3 s (`pytest_access_forms.log`; the AiM, AttAcc and Samsung refusals, the two shifted forms, the plain program's commands unchanged, the dense flattening accepted and the padded one refused) |
| X-41 second increment: AttAcc orientations (`attacc_tiling.attacc_orientations`, `_oriented_extents`, `AttAccKernel.matrix`; `spmw_attacc.attacc_orientation_of`, the base placement per orientation with `candidate_families`; `_attacc_enumerate`) | the reviewer's 32x64x8 GEMM in both spellings (`reviewer-r94/roles_probe.py` rerun): both select kernel rows 32, reduction 64, outputs 8, matrix `b`, tiling `[[1,32,1],[2,2,16],[5,1,2]]` (before: `(8,64,32)` for one spelling and `(32,64,8)` for the other). `tests/pim/test_attacc_orientation.py` (the two orientations and their kernels `(a,(8,64,32))`, `(b,(32,64,8))` in both spellings, identical selection at the ranking's minimum with no realization rejection, each orientation forced through `layout=` and realized with its declared storage, the GEMV's single orientation) with `test_attacc_virtual` (its kernel manifest now names the matrix, `lW`), `test_attacc_compose`, `test_attacc_trace_check`, `test_aim_axis_roles` and the access-form tests: 79 passed, 7 skipped in 144 s (`pytest_attacc_orientation.log`) |
| Lint and formatting | black on every changed file; pylint (unused-import, unused-variable, unused-argument, undefined-variable) on the changed modules reports only the seven kept hook signatures of `spmw_codegen.py` (round 92's triage) |
| Regression suites after both changes (`pytest_x42_slice1.log`, `pytest_x42_slice2.log`) | shared semantics, matcher coverage, Samsung lowering, layout selection and the role-coordinate tests: 219 passed, 2 failed in 2523 s; the two failures are `test_apu_v1_grouped_non_template_regions_are_refused_for_execution[reversed_k, permuted_rows]`, whose expectation was that a reversed operand access (`B[K-1-k]`, `A[M-1-i]`) is kept as a `virtual_only` estimate: the shared access contract now refuses it before any backend, so the two regions moved to `test_apu_v1_grouped_reversed_accesses_are_refused_on_every_backend` (the grouped route's own refusal for the board, no functional model, the shared refusal for the virtual estimate; 6 passed with the remaining four non-template cases). AiM matcher/program/owner-map/dense-domain/token-partition/value-layout suites, the CENT workloads, UPMEM matcher and generality, the compile interface, host-program analysis, work grid and the paper-golden rows: 216 passed, 41 skipped, 1 failed in 1209 s; the failure is `test_paper_golden.py::test_apu_v1_board_median_cycles[apu_v1-histogram]`, a board test that fails reading the device log (`ledag-ssh exited with 255`, the board host's full `/tmp`), the environmental blocker recorded since round 60 and unrelated to these changes; every AiM, UPMEM, Samsung and AttAcc golden row passes |
| Not done | the AiM tie orientation as a cost decision and the layout problem's naming consumer (X-41, remaining); the Gemini-I rows and the vector route's board execution (`/tmp` at 6.6 MB); the X-34 / X-37 confirmation campaign (reserved for the final candidate); the final-candidate validation; V1-19 (deferred) |

## Round 96 (2026-11-10): X-42 completed over the loop domain (the reviewer's lower-bound and step probes)

Checkout after `f31e0f3`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r96/`.  The host's
`/tmp` at about 6.6 MB (unchanged); nothing launched on hardware.

| Check | Result |
|---|---|
| The reviewer's probes (`reviewer-r95/access_neighbors.py`, `attacc_access_neighbors.py`) | `range(128, K)` and `range(0, K, 2)` reading `b[k]` passed the round 95 rule (which judged coefficients and constants but not `LoopAxis.lower`/`step`) and compiled on AiM and AttAcc to the dense `range(128)` program |
| Widened (`row_offset_probe.py`, `row_offset_probe.log`, `domain_probe2.py`, `domain_probe2.log`) | AiM compiles `range(16, 32)`, `range(0, 32, 2)` and the coordinate constant `a[16 + i]` / `c[16 + i]` on one work item to the rows 0..15 program (`AimContraction(outputs=16, row=0)`, the same `WR_GB`/`MAC_ABK` commands); AttAcc likewise (kernel rows 1, reduction 256, outputs 16 for both the constant and the lower bound). UPMEM's matcher does not match the int32 GEMV spelling at all (its families are explicit). Samsung's own derivation requires `[0, n)` ranges with step 1 and realizes a row constant through `row_offset` |
| Repair (`spmw_semantics.contraction_access_problems`, `spmw_codegen.require_contraction_accesses_realizable`, `CodegenContext.REALIZES_RESULT_OFFSETS`) | every loop variable in a coordinate is substituted by `lower + step * t`: the effective coefficients are judged by the dense-flattening rule (so a step refuses like a stride), the effective constant by the reduction-shift rule (so a lower bound refuses like a shift), and a result-loop effective constant is refused when the match is a single work item and the backend's context sets `REALIZES_RESULT_OFFSETS = False` (`AimCtx`, `AttAccCtx`); Samsung keeps its row offsets, multi-item kernels keep the grid's constants (the CENT rows), a loop without a static domain is reported. The message names the loop domain (`k with k in [128, 256)`, `k in [0, 256) step 2`) |
| `tests/pim/test_contraction_access_forms.py` | 7 passed: the two reviewer probes on AiM and AttAcc, `range(16, 32)` and the single-item `a[16 + i]` on AiM, the earlier stride, shift, dense-flattening and padded cases; with the orientation, role-coordinate and grouped-source tests: 20 passed in 146 s (`pytest_access_domain.log`); the reviewer's `access_neighbors.py` now reports `NotImplementedError` for `lower_bound`, `step_two` and `shifted_rows` |
| Regression suites after the repair (`pytest_slice1.log`, `pytest_slice2.log`) | shared semantics (with the grouped-source and reversed-access cases), matcher coverage, Samsung lowering, layout selection, the role-coordinate, access-form and orientation tests: 233 passed in 2430 s; the AiM matcher/program/owner-map/dense-domain/axis-role/token-partition/value-layout suites, the CENT workloads, UPMEM matcher and generality, the compile interface, host-program analysis, work grid, the AttAcc virtual/compose/trace-check suites and the paper-golden rows (the APU v1 board rows deselected, the board host's `/tmp` being full): 278 passed, 48 skipped, 9 deselected in 1387 s; no decision, role, realization or golden row changed |
| Not done | the AiM tie orientation as a cost decision and the layout problem's naming consumer (X-41, remaining); the Gemini-I rows and the vector route's board execution (`/tmp` at 6.6 MB); the X-34 / X-37 confirmation campaign (reserved for the final candidate); the final-candidate validation; V1-19 (deferred) |

## Round 97 (2026-11-10): X-42, the work-grid offsets proven against the realized frontier (the reviewer's shifted-grid probe)

Checkout after `08fd332`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r97/`.  The host's
`/tmp` at about 2.6 MB (the reviewer's reading; unchanged by this round);
nothing launched on hardware.

| Check | Result |
|---|---|
| The reviewer's probe (`reviewer-r96/grid_offsets.py`, `grid_offsets.log`) | the round 96 check exempted every multi-item kernel from the result-offset rule on the item count alone: on a two-item AiM grid (`W[64, 256]`, `mapping=[2]`, `R = 16`) `a[16 + ch*R + i, k]` / `c[16 + ch*R + i]` and `range(16, 32)` with `a[ch*R + i, k]` compiled to the same contraction records, commands and manifests as the base `a[ch*R + i, k]`, while the realization (`spmw_aim._contraction_row_map`) stages frontier rows 0..31 in work-id order |
| Repair (`spmw_semantics.contraction_result_offsets`, `_effective_access_forms`, `spmw_codegen.require_contraction_accesses_realizable`) | for a backend with `REALIZES_RESULT_OFFSETS = False`, the kernel group's dense result-loop coordinates are collected per work item (effective offset and extent after the loop-domain substitution) and proven against the frontier: per coordinate, either every item reads the same offset, which must then be zero, or item `w` (work-id order over the group shape) reads offset `w * extent` with equal extents; a coordinate not read alike by every item, differing extents, a uniform nonzero offset or offsets that do not tile the frontier are refused before selection, naming the offsets and the frontier. Single items keep the round 96 rule; backends that realize offsets (Samsung, by default) are not judged here |
| `tests/pim/test_contraction_access_forms.py` | 8 passed in 5 s (`pytest_access_grid.log`): the base grid `a[ch*R + i, k]` accepted with two matched items; the shifted grid, the grid with `range(16, 32)` and the strided grid `a[ch*32 + i, k]` (offsets `[16, 32]`, `[16, 32]`, `[0, 32]` against the frontier `[0, 16]`) refused; the earlier stride, shift, loop-domain, single-item and dense-flattening cases unchanged. The reviewer's `grid_offsets.py` rerun: `base` accepted, `shifted` and `lower` `NotImplementedError` (`reviewer_grid_offsets_rerun.txt`) |
| A first form of the proof, corrected before the commit | the first version compared coordinates across the whole kernel group rather than across the same contraction of every item, and refused the CENT `norm_residual_l128` region (`w1_projection_af_0: input 'W' coordinate 0 is not read alike by every work item`, `aim_failures2.log`: an item's several contractions were pooled); the proof now walks the items' contractions in match order and compares each column (the matcher's own column notion for `work_axis`); the CENT inventory test passes again (`11 passed, 11 skipped` with the access tests) |
| Regression suites after the repair (`pytest_slice1.log`, `pytest_slice2.log`) | shared semantics (grouped-source and reversed-access cases included), matcher coverage, Samsung lowering, layout selection, the role-coordinate, access-form and orientation tests: 234 passed in 2445 s; the AiM matcher/program/owner-map/dense-domain/axis-role/token-partition/value-layout suites, the CENT workloads, UPMEM matcher and generality, the compile interface, host-program analysis, work grid, the AttAcc virtual/compose/trace-check suites, the paper-golden rows (the APU v1 board rows deselected) and the G6 decision tests: 298 passed, 48 skipped, 10 deselected, 3 failed in 2761 s; the three failures are the S-17 Samsung decision tests (`test_samsung_host_fill_cost_flip_reaches_execution`, `test_samsung_fiber_term_removal_keeps_bank_row_by_construction`, `test_samsung_host_burst_cost_shows_the_documented_list_scheduling_anomaly`), open since round 39 with the same observed values; every CENT row, every golden row and every multi-item grid of the suites is accepted by the proof |
| Not done | the AiM tie orientation as a cost decision and the layout problem's naming consumer (X-41, remaining); the Gemini-I rows and the vector route's board execution (the board host's `/tmp` at about 2.6 MB); the X-34 / X-37 confirmation campaign (reserved for the final candidate); the final-candidate validation; V1-19 (deferred) |

## Round 98 (2026-11-10): X-42, same-buffer operands keep both accesses (the reviewer's duplicate-operand probe)

Checkout after `708f15d`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r98/`.  The host's
`/tmp` at about 2.6 MB (the reviewer's reading); nothing launched on
hardware.

| Check | Result |
|---|---|
| The reviewer's probe (`reviewer-r97/access_probes2.py`, `summary.txt`, `access_reversed.jsonl`) | `contraction_result_offsets` keyed its records by `(memref, coordinate)`, so for `a[16 + i, k] * a[j, k]` with `c[16 + i, j]` on one AiM item the second operand's record overwrote the first's and the offset 16 was lost: the program was accepted with the lowering of the zero-offset neighbour (a round 97 regression, since the round 96 single-item rule had refused it; the reversed operand order was refused) |
| Repair (`spmw_semantics._effective_access_forms`, `contraction_result_offsets`, `spmw_codegen.require_contraction_accesses_realizable`) | the records and the column comparison are keyed by `(operand role, memref, coordinate)`; every distinct access of a contraction is judged, and the messages name the buffer and the operand role |
| `tests/pim/test_contraction_access_forms.py` | 9 passed (`pytest_access_same_buffer.log` with the grouped-source cases: 15 passed): the zero-offset same-buffer neighbour accepted with both operands' records present, the shifted program refused in both operand orders, the shifted same-buffer grid refused against the frontier (AiM's own work-axis derivation refuses even the tiling same-buffer grid with an explicit error, after the shared check), and every earlier case unchanged. The reviewer's `access_probes2.py` rerun: `duplicate_base` accepted, `duplicate_shift` refused, `fixed_base`/`fixed_shift` refused by the output-map identity as before (`reviewer_access_probes2_rerun.txt`) |
| Regression suites after the repair (`pytest_slice1.log`, `pytest_slice2.log`) | shared semantics (grouped-source and reversed-access cases included), matcher coverage, Samsung lowering, layout selection, the role-coordinate, access-form and orientation tests: 235 passed in 2447 s; the AiM matcher/program/owner-map/dense-domain/axis-role/token-partition/value-layout suites, the CENT workloads, UPMEM matcher and generality, the compile interface, host-program analysis, work grid, the AttAcc virtual/compose/trace-check suites and the paper-golden rows (the APU v1 board rows deselected): 278 passed, 48 skipped, 9 deselected in 1350 s; no decision, role, realization or golden row changed |
| Not done | the AiM tie orientation as a cost decision and the layout problem's naming consumer (X-41, remaining); the Gemini-I rows and the vector route's board execution (the board host's `/tmp` at about 2.6 MB); the X-34 / X-37 confirmation campaign (reserved for the final candidate); the final-candidate validation; V1-19 (deferred) |

## Round 99 (2026-11-10): X-42, repeated loop variables across an operand's coordinates (the reviewer's diagonal probe)

Checkout after `71e6a18`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r99/`.  The host's
`/tmp` at about 2.6 MB (the reviewer's reading); nothing launched on
hardware.

| Check | Result |
|---|---|
| The reviewer's probe (`reviewer-r98/diagonal_probe.py`, `diagonal_probe.jsonl`, `summary.txt`) | `sum_k a[i,k]*b[k,k]` with `b[256, 256]` was accepted on AiM with the dense `b[k]` program's contraction record and commands: the check judged each coordinate on its own and both occurrences of `k` are plain counters; `spmw_candidate.value_axes` then deduplicated the axis and the vector contract declared a dense layout without a diagonal gather (the reviewer's reference calculation: 256 per row from the source, 1 from the contiguous prefix). The flattened `b[257*k]` was refused. `a[i, i, k]` likewise accepted |
| Repair (`spmw_semantics._repeated_axis_problems`, `_bounds_problem`, `contraction_access_problems`) | per operand, a loop variable read in two coordinates is refused as a diagonal gather, naming the coordinates; each dense coordinate's staged extent (`offset + product of the trip counts`) must fit the buffer dimension. The ordinary two-dimensional vector operand of a batched GEMV (`x[bb, k]`, each coordinate its own loop) reports no problem |
| `tests/pim/test_contraction_access_forms.py` | 10 passed (with the grouped-source cases 17 passed in 18 s; the round 99 run was not written to a file, and the reference to `r99/pytest_access_diagonal.log` here was wrong: the recorded run is `r100/pytest_access_diagonal.log`, made at the start of round 100 on the `ffa73ec` tree before any round 100 edit, 17 passed in 19 s): `b[k, k]` and `a[i, i, k]` refused per operand, the batched neighbour accepted with its batch and output roles, every earlier case unchanged. The reviewer's `diagonal_probe.py` rerun (`reviewer_diagonal_rerun.log`): `diagonal` and `flat_diagonal` both `NotImplementedError`, the dense program accepted |
| Regression suites after the repair (`pytest_slice1.log`, `pytest_slice2.log`) | shared semantics (grouped-source and reversed-access cases included), matcher coverage, Samsung lowering, layout selection, the role-coordinate, access-form and orientation tests: 236 passed in 2485 s; the AiM matcher/program/owner-map/dense-domain/axis-role/token-partition/value-layout suites, the CENT workloads, UPMEM matcher and generality, the compile interface, host-program analysis, work grid, the AttAcc virtual/compose/trace-check suites and the paper-golden rows (the APU v1 board rows deselected): 278 passed, 48 skipped, 9 deselected in 1330 s; the repeated-axis and bounds checks refuse no program of the suites (attention's `kc[h, s, d]` and `vc[h, d, s]`, the batched and multi-dimensional operands included); no decision, role, realization or golden row changed |
| Not done | the AiM tie orientation as a cost decision and the layout problem's naming consumer (X-41, remaining); the Gemini-I rows and the vector route's board execution (the board host's `/tmp` at about 2.6 MB); the X-34 / X-37 confirmation campaign (reserved for the final candidate); the final-candidate validation; V1-19 (deferred) |

## Round 100 (2026-11-10): X-41 closed, AiM storage orientations as cost-ranked candidates; the layout resolver freed from the naming

Checkout after `ffa73ec`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r100/`.  The host's
`/tmp` at about 2.4 MB (the reviewer's reading); nothing launched on
hardware.

| Check | Result |
|---|---|
| The round 99 evidence reference the reviewer flagged | `r99/pytest_access_diagonal.log` did not exist (the round 99 run was not written to a file); the recorded run is `r100/pytest_access_diagonal.log`, made at the start of this round on the `ffa73ec` tree before any edit: 17 passed in 19 s; VALIDATION round 99 corrected |
| The reviewer's neighbouring rectangular contraction on AiM (`aim_orientation_probe.py`, `aim_orientation_probe.log`) | a batched GEMV `Y[b, i] = sum_k W[i, k] X[b, k]` (`W[32, 256]`, `X[4, 256]`, one work item) in both operand orders: before this round the pattern-order convention made `W` the bank operand for the matrix-first spelling (342 cycles) and `X` for the vector-first one (223 cycles). Now both spellings enumerate 468 candidates across the two orientations (`aim_bank_operand=a`, `aim_bank_operand=x`), select the same program (`x` resident, `(outputs, batches) = (4, 32)`, 223 cycles, identical commands, no rejection), and `layout="aim_bank_operand=a"` realizes `(32, 4)` at 342 cycles through the realization check with different commands |
| `tests/pim/test_aim_orientation.py`, `test_attacc_orientation.py`, `test_aim_axis_roles.py` | 21 passed in 106 s (`pytest_orientation.log`): the two orientations and their families in both spellings, identical selection and commands, the forced alternative's shape, cycles and commands, the GEMV's single orientation; AttAcc unchanged on the shared helper; every AiM axis-role case unchanged |
| Lint | black on every changed file; pylint (unused-import, unused-variable, undefined-variable) on the six changed modules reports only the pre-existing `geometry` variable of `spmw_aim.py` (P-23 debt, untouched) |
| Regression suites after the change (`pytest_slice1.log`, `pytest_slice2.log`) | shared semantics (grouped-source and reversed-access cases included), matcher coverage, Samsung lowering, layout selection (the opaque-coordinate resolver among them), the role-coordinate, access-form, AttAcc-orientation and AiM-orientation tests: 241 passed in 2531 s; the AiM matcher/program/owner-map/dense-domain/axis-role/token-partition/value-layout/cost-route/profile-calibration suites, the CENT workloads, UPMEM matcher and generality, the compile interface, host-program analysis, work grid, the AttAcc virtual/compose/trace-check suites, the paper-golden rows (the APU v1 board rows deselected) and the G6 decision tests: 309 passed, 48 skipped, 10 deselected, 3 failed in 2838 s; the three failures are the S-17 Samsung decision tests, open since round 39 with the same observed values; every AiM golden row and every CENT trace is unchanged (all containment cases: one orientation) |
| Not done | the Gemini-I rows and the vector route's board execution (the board host's `/tmp` at about 2.4 MB); the X-34 / X-37 confirmation campaign (reserved for the final candidate); the final-candidate installed-artifact validation on both hardware hosts; S-17; V1-19 (deferred) |

## Round 101 (2026-11-10): X-43, the AiM row footprint covers the realized mapping; the row-range invariant at materialization

Checkout after `54ecd18`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r101/`.  The host's
`/tmp` at about 2.4 MB (the reviewer's reading); nothing launched on
hardware.

| Check | Result |
|---|---|
| The reviewer's probe (`reviewer-r100/grid_allocation.py`; `reviewer_grid_allocation_before.log`, rerun at `54ecd18`) | 48x320 contraction, 7 vectors, three partitioned work items: the x-resident orientation (selected in both spellings and forced) reserved `[0, 1)` while its channels-mapped record addressed `[0, 2)` (`allocated_matrix_rows` 2); the a-resident one reserved and addressed `[0, 3)`; zero rejections, the realization check silent |
| Repair (`spmw_aim._AimProgramDerivation._footprint`, `AimCtx._check_matrix_rows_within_regions`) | the footprint takes, per role, the larger of the buffer's extent and the realized shape of the candidate's orientation; every lowered record's `matrix_row_range` is compared with the allocator's regions in `check_realization`, so an under-allocated record is refused before selection (recorded as a rejection naming the reserved region) |
| The probe after the repair (`reviewer_grid_allocation_after.log`) | x-resident regions `[0, 2)` for `[0, 2)` addressed, a-resident `[0, 3)` for `[0, 3)`, in both spellings, selected and forced, zero rejections, cycles unchanged (480 and 525) |
| `tests/pim/test_aim_allocation.py` (`pytest_aim_allocation.log`) | 7 passed in 32 s: both spellings' default orientation reserves the two rows it addresses; the forced a-resident orientation three; row-packed and reduction-partitioned candidates of the ranking forced through `layout=` all fit their regions (the reviewer found that this round's form of the test took the first six candidates, all a-resident, and swallowed compile errors; round 102 parametrizes it by orientation and knob, forces the cheapest candidate of each of the four combinations and lets any error fail the test: 9 passed, `r102/pytest_aim_allocation.log`); a program with a second live contraction gets disjoint regions each containing its record's rows; with the footprint monkeypatched back to the buffer extent, the channels-mapped candidates are rejected with `the row allocator reserved` and the selected program fits |
| Regression slice after the repair (`pytest_aim_slice.log`) | the AiM matcher/program/owner-map/dense-domain/axis-role/orientation/token-partition/value-layout/cost-route/profile-calibration suites, the CENT workloads, the paper-golden rows (the APU v1 board rows deselected), the G6 decision tests and the shared-semantics suite: 260 passed, 38 skipped, 35 deselected, 3 failed in 3082 s; the three failures are the S-17 Samsung decision tests, open since round 39 with the same observed values; every AiM golden row and CENT trace is unchanged (their matrices carry their batch coordinate or none is needed, so the footprint is the same) |
| Not done | the Gemini-I rows and the vector route's board execution (the board host's `/tmp` at about 2.4 MB); the X-34 / X-37 confirmation campaign (reserved for the final candidate); the final-candidate installed-artifact validation on both hardware hosts; S-17; V1-19 (deferred) |

## Round 102 (2026-11-10): the allocation test's orientation coverage; S-17 restated from the current model with conductor evidence

Checkout after `fc9dcdf`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r102/`.  The host's
`/tmp` at about 2.4 MB (the reviewer's reading); the conductor runs below
used the evidence tree's `TMPDIR`; nothing launched on the boards.

| Check | Result |
|---|---|
| The reviewer's round 101 finding on `tests/pim/test_aim_allocation.py` | the forced-knob test took the first six candidates (all a-resident) and swallowed compile errors. It is parametrized by orientation and knob (`a`/`x` x `row_packed`/`reduction_partitions=2`), forces the cheapest priced candidate of each combination, requires compilation, containment and the kept orientation, and lets any error fail: 9 passed in 30 s (`pytest_aim_allocation.log`; `forced_knob_probe.log` shows the cheapest candidate of every combination realizing: x-resident row-packed 16 rows, the others 3) |
| S-17 facts (`s17_probe.log`, `s17_probe2.log`, `s17_probe3.log`) | stock default `split_rows+host` 18,630; the round 36 `x8` host-bus perturbation now gives 18,686 and the **same** selection (the cheapest CRF fills are split-row candidates 16 cycles dearer, refused at realization as having no CRF-filled variant; the cheapest realizable CRF fill, `bank_row+crf`, is 22,118, a gap of 3,488); with `HOST_BURST_CYCLES = 438` (derived: `1 + 3488 // 8 + 1`) the selection flips to `bank_row+crf` (22,118; FILL 2), and on the conductor the stock program runs in 4,108 cycles and the flipped one in 4,589, both within tolerance of the float32 reference. The odd-fiber term removal lowers `split_rows+host` by 1,128 (= 47 x 8 units x 3 tiles), the vendor `dual_fiber` by 376 (= 47 x 8 x 1) and `bank_row+host` by 0 (22,102 both). Host-burst `b = 3, 4, 5`: default 18,646 / 18,654 / 18,662 and `bank_row+host` 22,118 / 22,126 / 22,134 (monotone, +8 per cycle), unit 5's fill at 20 in every case, units 6-7 at 24, 28 for `b >= 4` and pushed to 2,104 / 2,108 (default) and 2,680 / 2,684 (`bank_row`) at `b = 3` |
| The three restated tests (`pytest_s17.log`) | 3 passed in 852 s: the flip factor derived and applied with both programs executed on the conductor; every candidate's estimate shifted by exactly the derived odd-fiber amount; the first-fit fill schedule and the monotone total on both placements |

## Round 103 (2026-11-10): the hardware blocker rechecked; V1-19 repaired (the functional reference under NumPy 2.4+)

Checkout after `87e3ea9`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r103/`.

| Check | Result |
|---|---|
| Device prerequisites (the board host's `/tmp`) | `df -k /tmp`: 2,432 kB free of 15.7 GB (100 percent), held by other users' directories; the vendor tool `ledag-ssh` writes its session log and per-read scratch under `/tmp` regardless of `TMPDIR` (VALIDATION rounds 60, 69, 85), so the APU v1 installed-artifact vector, forced-plan and cost-perturbation runs and the three Gemini-I rows stay blocked; nothing launched, no file of another user touched |
| V1-19 reproduced (`venv-np24/`, NumPy 2.5.4 over the release environment's site packages; `pytest_v1_19_before_np254.log`) | the four recorded tests (`test_temporal_plan_accumulates_nonzero_initial_output[functional]`, `test_plan_without_output_ingress_rejects_nonzero_output_on_every_route`, `test_vector_plan_selection_requires_cost_or_explicit_layout`, `test_local_accumulator_gemm_executes_on_apu_v1_functional_backend`) fail with `ValueError: setting an array element with a sequence` (4 failed, 1 passed: the virtual parameter) |
| Repair (`allo/pim/apu_v1_vector_program.py`), round 103 form | the `warnings.filterwarnings` suppression is removed; `_scalar_one_dimensional_loops` bound Allo's `grid`/`reduction` module attributes to scalar-yielding iterators while the functional reference called the source workload. **Superseded in round 104**: the reviewer showed that bindings imported as `from allo import grid, reduction` kept the originals (the source still failed under NumPy 2.5.4) and that overlapping contexts left the module attribute replaced; see round 104 |
| After the repair | NumPy 2.5.4: 5 passed in 28 s (`pytest_v1_19_after_np254.log`, run with `-W error::DeprecationWarning`); NumPy 2.3.5 (the checkout's `spmw` interpreter; the log does not print the interpreter, attribution from the round's notes, round 120): 5 passed in 29 s with the tuple-conversion deprecation turned into an error (`pytest_v1_19_after_np235.log`), so the suppressed warning is gone rather than hidden |
| APU v1 and shared-semantics suites after the repair (`pytest_apu_v1_slice.log`) | the APU v1 recognizer, build-config, cost-provenance, tile-selection, vector-compile, native-reduction, reduction-policy, scalar-surface, hybrid-manifest and distance-argmin suites, the run-result contract and the shared-semantics suite (the two board tests deselected): 273 passed in 844 s; no decision or program changed |
| Not done | the APU v1 installed-artifact vector, forced-plan and cost-perturbation execution and the three Gemini-I rows (the board host's `/tmp` at 2.4 MB); the X-34 / X-37 confirmation campaign (reserved for the final candidate); the final-candidate installed-artifact validation on the simulators (the bundled Samsung build from the wheel included) and both hardware hosts |

## Round 104 (2026-11-10): V1-19 repaired locally (the reviewer's imported-binding and overlap findings)

Checkout after `1224aa0`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r104/`.  The host's
`/tmp` at about 2.4 MB (the reviewer's reading); nothing launched on
hardware.

| Check | Result |
|---|---|
| The reviewer's probe (`reviewer-r103/probe_imports.py`) against the round 103 code | a 64x32x64 uint16 contraction written with `from allo import grid, reduction` compiled through `allo.compile(..., backend="functional", layout="temporal_svp")` and still failed at execution under NumPy 2.5.4 (the imported names kept the original functions); two overlapping contexts left `allo.grid` permanently replaced |
| Repair (`_scalar_loop_view`, `_ScalarLoopModuleView`, `_scalar_one_dimensional_range`) | the functional reference runs the workload rebound to a copy of its own globals: module references become read-only views whose `grid`/`reduction` are the scalar ranges, direct bindings and aliases of those functions become the scalar range; no process-wide binding is modified, so there is nothing to restore after an exception and overlapping references cannot interfere |
| `tests/perf/test_apu_v1_functional_reference.py` with the four recorded V1-19 tests | NumPy 2.5.4 (`venv-np24`, `-W error::DeprecationWarning`): 11 passed in 32 s (`pytest_v1_19_np254.log`); NumPy 2.3.5 (the checkout's `spmw` interpreter; attribution from the round's notes, round 120) with the tuple-conversion deprecation an error: 11 passed in 32 s (`pytest_v1_19_np235.log`). The new file covers the module-qualified, imported and aliased spellings (every element 32), the process bindings observed unchanged from another module while a rebound workload runs, the view's one-dimensional ranges yielding integers and two-dimensional ones tuples, the view's module delegating other attributes and refusing assignment, two rebound workloads interleaved, an exception inside a rebound workload, and a non-function refused with its type |
| The reviewer's probe rerun (`probe_imports_adapted.py`, its removed context-manager import replaced by the rebound views; `reviewer_probe_imports_rerun.log`) | `qualified` and `imported` both COMPILED `temporal_svp` and PASS `functional` under NumPy 2.5.4; the interleaved views both give every element 32; `allo.grid` is the original and yields tuples |
| APU v1 and shared-semantics suites on the final code (`pytest_apu_v1_slice.log`) | the APU v1 recognizer, functional-reference, build-config, cost-provenance, tile-selection, vector-compile, native-reduction, reduction-policy, scalar-surface, hybrid-manifest and distance-argmin suites, the run-result contract and the shared-semantics suite (the two board tests deselected): 279 passed in 848 s; `pylint` (E, unused-import, unused-variable, undefined-variable) on the changed module: 0 (the view is invoked through `operator.call`, since static inference does not know a `types.FunctionType`-built function to be callable; no diagnostic is suppressed) |
| Not done | the APU v1 installed-artifact vector, forced-plan and cost-perturbation execution and the three Gemini-I rows (the board host's `/tmp` at 2.4 MB); the X-34 / X-37 confirmation campaign (reserved for the final candidate); the final-candidate installed-artifact validation on the simulators (the bundled Samsung build from the wheel included) and both hardware hosts |

## Round 105 (2026-11-10): V1-19, closure cells and the `dsl` submodule follow the local view (the reviewer's round 104 probes)

Checkout after `2556cd2`; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r105/`.  The host's
`/tmp` at about 2.4 MB (the reviewer's reading); nothing launched on
hardware.

| Check | Result |
|---|---|
| The reviewer's probes (`reviewer-r104/probe_closures.py`, `probe_closures.log`) against the round 104 code | three accepted source forms compiled as `temporal_svp` and failed at execution under NumPy 2.5.4: a factory capturing `from allo import grid, reduction` in closure cells, a factory capturing `import allo as local_allo`, and `allo.dsl.grid` reached through the module view's delegation |
| Repair (`_scalar_loop_view`, `_ScalarLoopModuleView.__getattr__`, `_cell_is_bound`) | closure cells are rebound into fresh `types.CellType` cells with the same rule as the globals (module references become views, direct bindings become the scalar range, everything else kept); the module view returns a view for the `allo`/`allo.dsl` modules reached through it. No original cell or module attribute changes |
| `tests/perf/test_apu_v1_functional_reference.py` with the four recorded V1-19 tests | NumPy 2.5.4 (`-W error::DeprecationWarning`): 15 passed in 36 s (`pytest_v1_19_np254.log`); NumPy 2.3.5 (the checkout's `spmw` interpreter; attribution from the round's notes, round 120) with the tuple-conversion deprecation an error: 15 passed in 35 s (`pytest_v1_19_np235.log`). New cases: the two factory forms and the `allo.dsl` form through `allo.compile`'s functional route, and the closure cells shown fresh with the unrelated captured value kept and the original cells' contents unchanged |
| The reviewer's probe rerun (`reviewer_probe_closures_rerun.log`) | all three forms PASS `functional` under NumPy 2.5.4 |
| APU v1 and shared-semantics suites on the final code (`pytest_apu_v1_slice.log`) | the APU v1 recognizer, functional-reference, build-config, cost-provenance, tile-selection, vector-compile, native-reduction, reduction-policy, scalar-surface, hybrid-manifest and distance-argmin suites, the run-result contract and the shared-semantics suite (the two board tests deselected): 283 passed in 893 s; pylint (E, unused-import, unused-variable, undefined-variable) on the changed module: 0 |
| Named candidate for the integration milestone | `f6b19bd` (rounds 93-105 included). sdist from `git archive f6b19bd`: `8a8cef280167980475c9549a665480e5c7682e2e013f0a27f083f6f3eb20ebe1` (`pkg/`, 70 upstream files and 5 patches under `simulators/PIMSimulator/`); the wheel build and the clean venv follow (`build_wheel.log`) |
| **Pre-registration (written before any launch): the Gemini-II candidate-validation session at `f6b19bd`** | ONE paired session of 20 rounds of `benchmarks.apu_g2.paper_rows` from a wheel built on `zhang-x4` from the candidate's sdist and installed into a clean venv, after the card tests from that wheel, under the device lock (`g2_build_wheel.sh`, `g2_card_tests.sh`, `g2_candidate_session.sh`, launched by `g2_remote_chain.sh` over SSH). The Gemini-II modules are unchanged since `fdee44b` (`git diff fdee44b..f6b19bd` over `allo/pim/apu_g2_*`, `allo/spmw_apu_g2.py`, `benchmarks/apu_g2`, the templates: empty); the five required rows must be measured from this candidate's installed artifact. Recorded whatever the outcome; the statistic at this revision is this session's samples, reported beside the earlier revisions' verdicts and not pooled. Not the X-34 / X-37 confirmation campaign (reserved for the final candidate). Verdict rule per row, unchanged from round 94: REPRODUCED if the session median is at or below the paper's value with every output exact; otherwise OPEN (nominal) if the paper's value lies inside the session's interquartile range, OPEN (regression) if it lies below it |
| The candidate wheel and clean venv (`build_wheel.log`, `pkg/`) | wheel `0df0910af83bd769989c324d8f2875152b00a356ab23aca0b52511fbf14c2209` (84 recipe members under `allo/pim/simulators/PIMSimulator/`, the Samsung and DRAMSim2 texts in `dist-info/licenses/`), installed with `--no-deps` into a clean venv with the runtime dependencies and `scons` |
| Installed stages from the wheel, outside the checkout, no `PYTHONPATH`, `PYTHONSAFEPATH=1`, no simulator variable for Samsung (`run_installed.sh`, `run_installed.log`) | the package imports from the venv with every backend module; the bundled Samsung recipe verifies (70 files), `locate` reports none, `build --jobs 16` builds in 15.1 s and prints the geometry, `locate` reports the default root available, the stripped `pim_driver` is the evaluation conductor's identity (`9aea49bf...`); the GEMV probe compiles and executes on it (REDUCE, 4,108 cycles, max abs error 0.0004, `pimcmdgen_calls` 0); `fresh_install_check` (virtual compile and run, no module from the checkout); `aim_check` through the native `ramulator2` (222 cycles, timing-only, outputs untouched); `upmem_check` through the native SDK executor (563,207 cycles, exact; its first appearance in the log was hidden by the log filter and was rerun directly, `upmem_check_direct.log`); `fresh_sim_check` after repointing its stale `/tmp` source path to the evidence tree (`fresh_sim_check.py`: the ordered Samsung program, 26,807 cycles, max abs error 0.0022 on the bundled conductor); the default golden rows for every simulator with Samsung through the bundled build at `TENON_HOME`: 26 passed, 24 skipped (`paper_full`), the three Samsung rows 4,108 / 7,057 / 7,255 against the paper's 4,435 / 8,041 / 8,247 (vendor placements 4,435 / 8,041 / 8,247), the AiM corrected programs 7,052 and 1,686; the remaining stages (the `paper_full` rows, the conductor runs, CENT, AttAcc, the APU v1 functional reference under NumPy 2.5.4 from the wheel) were still running when round 105 closed and are reported in round 106 with the next candidate's run |
| The Gemini-II chain at `f6b19bd` on `zhang-x4` (`g2-build-wheel.log`, `g2-card-tests.log`, `g2-candidate-session.log`, `g2_candidate_rows.json`) | wheel `9e0d0b61...` built there and installed into a clean venv; **card tests: 14 passed, 1 failed** (`test_typed_program_per_operation_on_card`, the typed `block_sum` record refused by the round 99 bounds rule, X-44, repaired below and to be re-run from the next candidate); the pre-registered session ran to completion: status `pass`, 210 of 210 launches with exact outputs; per row (median / paper / ratio / IQR / verdict): gemm 1,738.975 / 1,733.225 / 1.0033 / 1,722.2 to 1,750.1 / OPEN (nominal); batched_gemm 1,732.975 / 1,739.15 / 0.9964 / REPRODUCED; gesummv 1,841.075 / 1,860.075 / 0.9898 / REPRODUCED; bicg 2,536.825 / 2,548.825 / 0.9953 / REPRODUCED; mvt 2,755.1 / 2,766.625 / 0.9958 / REPRODUCED; GTML in the same rounds 1,917.3 / 1,916.2 / 3,105.575 / 2,736.6 / 3,090.75. Recorded in the inventory, `PAPER_REPRODUCTION.md`, X-34 and X-37 |
| X-44 (`block_sum_probe.py`) and the X-43 addendum (`same_buffer_probe.py`, `same_buffer_probe2.py`, `same_buffer_probe3.py`) | the typed `block_sum` refusal reproduced locally with the virtual backend; the bounds rule now bounds a single flat coordinate by the buffer's element count. The same-buffer contraction `a[i, k] * a[j, k]` was found refused by the round 101 invariant (12 candidates `addresses matrix rows [0, 2) but the row allocator reserved [0, 1)`): its slot usage now carries the contraction and the footprint reserves the larger of the slot image and the launch rows. `tests/pim/test_contraction_access_forms.py`, `test_aim_allocation.py` (the same-buffer dot row), `test_aim_orientation.py`: 25 passed, plus the same-buffer allocation test |
| Regression slices after the X-44 and X-43 repairs (`pytest_sliceA.log`, `pytest_sliceB.log`) | the AiM matcher/program/owner-map/dense-domain/axis-role/orientation/allocation/token-partition/value-layout/cost-route/profile-calibration suites, the CENT workloads, the paper-golden rows (the APU v1 board rows deselected), the G6 decision tests (the three restated S-17 tests passing) and the access-form tests: 155 passed, 38 skipped, 7 deselected in 2977 s; shared semantics, matcher coverage, Samsung lowering, layout selection, the role-coordinate and AttAcc-orientation tests, the AttAcc virtual/compose/trace-check suites, UPMEM matcher and generality, the compile interface, host-program analysis, work grid and the APU v1 functional reference: 422 passed, 10 skipped in 3096 s; no decision, realization or golden row changed |

## Round 106 (2026-11-10): the integration milestone re-run at the named candidate `b994fc0`

Checkout after the round 105 commits; evidence under
`/work/shared/users/phd/nz264/tenon-release-evidence/r106/`.

| Check | Result |
|---|---|
| **Pre-registration (written before any launch): the Gemini-II candidate-validation session at `b994fc0`** | the round 105 chain, unchanged (`g2_build_wheel.sh`, `g2_card_tests.sh`, `g2_candidate_session.sh` via `g2_remote_chain.sh` over SSH): a wheel built on `zhang-x4` from this candidate's sdist into a clean venv, the card tests from it, then ONE paired session of 20 rounds of `benchmarks.apu_g2.paper_rows` under the device lock. Purpose: the candidate repairs the shared bounds rule that refused the typed `block_sum` program at `f6b19bd` (X-44) and the bank-sourced footprint (X-43); the Gemini-II modules are otherwise unchanged since `fdee44b`, and the five required rows must be measured from this candidate's installed artifact. Recorded whatever the outcome; the statistic at this revision is this session's samples, reported beside the earlier revisions and not pooled; not the X-34 / X-37 confirmation campaign. Verdict rule per row unchanged from round 94 |
| The candidate's sdist, wheel and clean venv (`build_wheel.log`, `pkg/`), built from `1b4282d` (the ledger commit on top of `b994fc0`; `git diff --stat b994fc0 1b4282d -- allo tests setup.py simulators` is empty, so the code is the named candidate's) | sdist `4e97a518d04d524560d6b02d4e34d69679faa9f46015d42073200e06716dd277`, wheel `5ba2e3486fbd6f466eabd7dcb94d5d3b4487125fcd67c408b378fe16a5d7c39b` (the recipe and the license texts carried as in round 105), installed into a clean venv |
| The Gemini-II chain at `b994fc0` on `zhang-x4` (`g2-build-wheel.log`, `g2-card-tests.log`, `g2-candidate-session.log`, `g2_candidate_rows.json`) | wheel `d5cb59f8...` built there and installed into a clean venv; **card tests: 15 passed in 124 s** (the typed `block_sum` program that X-44 refused at `f6b19bd` now compiles, runs and matches its reference); the pre-registered session: status `pass`, 210 of 210 launches exact; per row (median / paper / ratio / IQR / verdict): gemm 1,735.775 / 1,733.225 / 1.0015 / 1,721.0 to 1,745.1 / OPEN (nominal); batched_gemm 1,735.375 / 1,739.15 / 0.9978 / REPRODUCED; gesummv 1,834.325 / 1,860.075 / 0.9862 / REPRODUCED; bicg 2,531.9 / 2,548.825 / 0.9934 / REPRODUCED; mvt 2,763.675 / 2,766.625 / 0.9989 / REPRODUCED; GTML 1,913.175 / 1,914.425 / 3,108.85 / 2,733.8 / 3,097.525. Recorded in the inventory, `PAPER_REPRODUCTION.md`, X-34 and X-37 |
| Installed stages from the `b994fc0` wheel, outside the checkout (`run_installed.sh`, `run_installed.log`), first part | package imported from the venv with every backend module; the bundled Samsung recipe verified (70 files), built in 15 s with the geometry printed, discovered through `TENON_HOME` with no variable, the stripped `pim_driver` the evaluation conductor's identity (`9aea49bf...`); the GEMV probe on it: REDUCE, 4,108 cycles, max abs error 0.0004, `pimcmdgen_calls` 0; `fresh_install_check` (45,550 virtual cycles, no module from the checkout), `aim_check` (222 cycles, timing-only) and `upmem_check` (563,207 cycles, exact) as in round 105; the script's loop ran the round 8 copy of `fresh_sim_check` whose source path still points at the vanished `/tmp` tree, so that stage exited 1 (recorded as such in `run_installed.log`), and the repointed copy (`fresh_sim_check.py`, the same one round 105 used) was then run from the same venv and environment: the ordered Samsung program, 26,807 cycles, max abs error 0.0022 on the bundled conductor (`fresh_sim_check_rerun.log`); the default golden rows for every simulator with Samsung through the bundled build: 26 passed, 24 skipped, the Samsung rows 4,108 / 7,057 / 7,255 (vendor placements 4,435 / 8,041 / 8,247), the AiM corrected programs 7,052 and 1,686; the remaining stages are in the next row |
| Installed stages from the `b994fc0` wheel, outside the checkout (`run_installed.log`), second part (`DONE 2026-11-10T15:27:02-05:00`) | the `paper_full` golden rows for every simulator: 24 passed; the Samsung conductor runs and the GEMV-on-every-target rows: 14 passed; the CENT suite: 8 passed, 13 skipped (the 13 skips are the `paper_full` selection skips `tests/pim/conftest.py:36-43` adds to every `paper_full`-marked case when `-m paper_full` is not given; this stage did not select them, so those cases did not run here; the review of round 106 corrected my earlier explanation, which wrongly attributed them to checkout variables); the AttAcc virtual, AttAcc orientation, AiM allocation, access-forms, AiM orientation and axis-roles set: **77 passed, 1 skipped** (round 105 at `f6b19bd`: 1 failed, the X-44 flat-view refusal, now passing from the wheel); the APU v1 functional reference from the wheel: 10 passed with DeprecationWarning as error in the clean venv (labelled "NumPy 2.3.5" here until round 120; the venv had installed unpinned numpy 2.5.4, as its `numpy-2.5.4.dist-info` shows, so this run was under 2.5.4 as well), and 13 passed under NumPy 2.5.4 (`venv-np24`, the wheel reinstalled there without dependencies) including the nonzero-initial-output temporal plan and the local-accumulator GEMM on the functional backend; the first run under NumPy 2.3.5 proper from an installed wheel is round 120's. Every local stage of the milestone at this candidate passed; the hardware stages are the Gemini-II chain above (passed) and the Gemini-I stages (not run, the board host's `/tmp` still full) |

## Round 107 (2026-11-10): the review's record corrections, the milestone scripts repaired, `b994fc0` declared the final compiler candidate, the Gemini-II confirmation campaign pre-registered and launched

| Check | Result |
|---|---|
| The review of round 106 (independent re-runs: the installed contraction-access, AiM-allocation and APU v1 functional-reference suites 31 passed, the installed UPMEM probe exact at 563,207 cycles, the 210 exact launches and the five medians recomputed from the batch ticks, the 73 frozen-paper files against their lock) | accepted as the reviewer's own evidence; nothing in it is restated here as mine |
| Record correction 1 (`VALIDATION.md`, the round 106 second-part row) | the 13 CENT skips are the `paper_full` selection skips of `tests/pim/conftest.py:36-43`, not checkout-variable skips; the row now says so and does not claim those cases ran. The repaired milestone script below runs the CENT `paper_full` selection as its own stage |
| Record correction 2 (`REVIEW.md`, the active verdict block) | the current summary now cites `b994fc0`: gemm OPEN (nominal) at 1.0015, the other four Gemini-II rows REPRODUCED, the round 106 local installed results; the historical sections are unchanged |
| The milestone scripts (`docs/release/scripts/`, new; `stage_lib.sh`, `run_installed_milestone.sh`, `probes/`, `g2_build_wheel.sh`, `g2_card_tests.sh`, `g2_session.sh`, `g2_remote_chain.sh`, `g2_campaign.sh`, `g2_pool.py`) | every step is a `stage`: the command's complete output kept in `<round>/stages/<name>.log`, the command's own exit status (PIPESTATUS of the command under `pipefail`, not of a `tail` or `grep`) written to `<name>.status`, a required failure recorded and the whole run exiting nonzero from `milestone_finish`; `DONE` is no longer printed unconditionally. The probes are the round 8 scripts copied into the repository with `fresh_sim_check.py` loading `gemv_src.py` from its own directory (the stale `/tmp/tenon-release/r9` path is gone; `grep /tmp/ probes/*` finds nothing); `g2_session.sh` fails unless the session file records `status: pass`. Harness smoke test (`r107/smoke/stages`): a passing stage PASS rc=0, a failing stage FAIL rc=3 marked required, an optional failure recorded without failing the run, a stage exiting 7 reported as rc=7, `milestone_finish` exit 1 with a required failure and exit 0 without. `g2_pool.py` on the round 106 session alone reproduces the recorded verdicts (gemm 1735.775 OPEN (nominal), the other four REPRODUCED; its quartiles use the median-of-halves convention of the round 55 pooling scripts, so the IQR bounds differ from `paper_rows`' own by at most 0.3 ticks). The round 106 scripts in the evidence tree are left as they ran |
| Gemini-I headroom rechecked once (the board is on this host, `zhang-capra-xcel`) | `/dev/mapper/VG00-tmp 15G 15G 2.2M 100% /tmp`: the blocker remains; the APU v1 installed-artifact vector, forced-alternative and cost-perturbation execution and the three Gemini-I rows are not run. No other user's files touched |
| **Declaration: `b994fc0` is the final compiler candidate** | every commit after it (`1b4282d`, `d0fc602`, this round) changes release records and the milestone scripts only (`git diff --stat b994fc0 HEAD -- allo tests setup.py simulators MANIFEST.in` is empty, checked before this entry). No compiler change is known to be necessary: the open gates need hardware time (Gemini-I) and the bounded Gemini-II campaign, not code. If a compiler change becomes necessary, its concrete dependency is to be stated in ISSUES before any further candidate validation, and the pre-registered rule then applies at that later revision (the campaign run there, this candidate's sessions reported beside it, not pooled) |
| **Pre-registration (written and committed before the launch): the Gemini-II confirmation campaign at the final candidate `b994fc0` (X-34 / X-37)** | the rule as recorded in ISSUES X-34 (round 78) and `PAPER_REPRODUCTION.md`, unchanged: THREE further paired sessions of 20 rounds of all five rows (`benchmarks.apu_g2.paper_rows --rounds 20`) from the SAME `zhang-x4` wheel as this candidate's validation session (`d5cb59f8...`, `r106/g2/venv`, `r106/g2/bench`), under the device lock, launched once by `g2_campaign.sh r107 r106/g2 3`; per row the samples pooled with the round 106 session (80 Tenon samples per row) by `g2_pool.py`; REPRODUCED if the pooled median is at or below the paper's value with every output exact, otherwise OPEN (nominal) when the paper's value lies inside the pooled interquartile range and OPEN (regression) when it lies below it; fixed thereafter, every row reported whatever the outcome, no further sessions to move a status; the per-session medians, the re-run GTML lane and the paired GTML/Tenon ratio reported beside the pooled statistic. This is gemm's one campaign and mvt's one campaign; it is not a further sample for the spent batched_gemm campaign at `54cbb08` (X-26), whose status at this revision is this candidate's own sessions. A session that ends without `status: pass` is reported as incomplete and its samples are not pooled; the campaign is not re-run to replace it |
| **The Gemini-II confirmation campaign at the final candidate `b994fc0` (`g2_campaign.sh`; `r107/g2-campaign.log`, `r107/campaign-stages/`, `r107/g2_campaign_s1.json`, `s2`, `s3`; pooled with `r106/g2_candidate_rows.json` by `g2_pool.py` into `r107/g2_pooled.json`)** | launched 15:50:04, finished 16:15:22 on `en-ec-zhang-x4`, every session a PASS stage with `status: pass`, no problems, 210 launches each, every output exact in all 840 launches; the same wheel `d5cb59f8...` and venv as the round 106 session; one program fingerprint and one layout fingerprint per row across all four sessions (gemm `86b7bdd981cf52d2...`, batched_gemm `b4ea0cdeff663870...`, gesummv `295a0ac40c96aeca...`, bicg `fbe0a233eb89e3ee...`, mvt `0744614cfd9a9aed...`), plans `single/k1`, `single/k1`, `fused/k1`, `fused/k1`, `fused/k1`; 80 Tenon samples per row pooled. Under the rule as pre-registered: **gemm** pooled median 1,734.825 / paper 1,733.225 (ratio 1.0009; pooled IQR 1,723.15 to 1,746.6; per-session medians 1,735.775, 1,734.225, 1,738.975, 1,729.9; GTML pooled median 1,917.35; paired GTML/Tenon median 1.1015 against the paper's 1.0984) **OPEN (nominal)**; **batched_gemm** pooled median 1,737.975 / paper 1,739.15 (ratio 0.9993; pooled IQR 1,724.675 to 1,747.25; per-session medians 1,735.375, 1,740.175, 1,736.275, 1,738.4; GTML pooled median 1,914.325; paired GTML/Tenon median 1.0998 against the paper's 1.0962) **REPRODUCED**; **gesummv** pooled median 1,836.825 / paper 1,860.075 (ratio 0.9875; pooled IQR 1,828.2 to 1,849.775; per-session medians 1,834.325, 1,840.25, 1,838.025, 1,837.125; GTML pooled median 3,110.075; paired GTML/Tenon median 1.6910 against the paper's 1.6617) **REPRODUCED**; **bicg** pooled median 2,538.775 / paper 2,548.825 (ratio 0.9961; pooled IQR 2,531.3 to 2,552.05; per-session medians 2,531.9, 2,539.675, 2,539.9, 2,542.025; GTML pooled median 2,734.425; paired GTML/Tenon median 1.0765 against the paper's 1.0697) **REPRODUCED**; **mvt** pooled median 2,767 / paper 2,766.625 (ratio 1.0001; pooled IQR 2,752.875 to 2,774.875; per-session medians 2,763.675, 2,767.05, 2,767.3, 2,769.3; GTML pooled median 3,095.225; paired GTML/Tenon median 1.1195 against the paper's 1.1168) **OPEN (nominal)**. The statuses of `gemm` and `mvt` at the final candidate are now FIXED by the pooled median (X-34, X-37); no further session is run to move them. `mvt` was REPRODUCED in the round 106 session alone (0.9989) and is OPEN (nominal) on the pooled 80 samples (1.0001): the rule says the pooled statistic decides, so that is the status recorded, and I note the direction honestly rather than reporting the favorable single session |
| What the campaign establishes and does not | all five Gemini-II kernels compile to one program each at the final candidate, execute exactly, and measure within 0.1 percent of the paper's archived medians on either side (gemm +0.09 percent, mvt +0.01 percent, the other three faster); the paired GTML/Tenon ratios stay above the paper's on every row (Tenon's advantage over the vendor baseline re-run in the same rounds is at least the paper's); no cause for the two nominal rows is established and none is claimed; the required set at the final candidate is 50 REPRODUCED, 5 OPEN (the three Gemini-I rows, gemm and mvt), 0 UNVERIFIED |
| Not done | the APU v1 installed-artifact vector, forced-plan and cost-perturbation execution and the three Gemini-I rows (this host's `/tmp` at 2.2 MB); the final verdict (NOT READY until the gates hold) |

## Round 108 (2026-11-10): the validation scripts made unable to report false success (P-26)

| Check | Result |
|---|---|
| The reviewer's incomplete-session probe (`reviewer-r107/incomplete.json`: `status: fail`, a recorded problem, no launch records) through the round 107 `g2_pool.py` | reproduced before the repair: exit 0, four rows REPRODUCED with `exact=True` (the script pooled summarized samples regardless of status and defaulted missing launch evidence to exact) |
| `g2_pool.py` repaired (`docs/release/scripts/g2_pool.py`) | validation before pooling as listed in ISSUES P-26; the same probe now: `REJECTED ... status is 'fail', not 'pass'`, exit 2, no output file written; with `--exclude-invalid` the invalid session is listed under `rejected_sessions` and only the valid one pooled. One rule correction during the work: `paper_rows` numbers the warmup round 0 and the measured rounds 1 to N, so the coverage check requires N consecutive measured rounds (the first draft required rounds 0 to N-1 and rejected every real session, which is recorded here as the kind of false failure the tests now pin) |
| `tests/perf/test_release_pooling_script.py` (synthetic sessions, no evidence tree) | 21 passed: complete sessions pool under the rule (REPRODUCED, OPEN (nominal) and OPEN (regression) verdicts), and rejections for a failed status, recorded problems, no launch records, a missing measured launch, a missing `outputs_exact`, an `outputs_exact=False` on the vendor lane, a timing-evidence problem, summarized samples differing from the launches, a non-consecutive round, a launch without its packing attestation, two program fingerprints in one session, baseline binaries not recorded, a missing field, and incompatible provenance across sessions (installed package path, program fingerprint, layout fingerprint, plan); a rejected session fails the pool unless excluded explicitly; missing exactness evidence never implies exactness |
| The four-session campaign recomputed with the repaired pooler (`r108/g2_pooled_recomputed.json`, `g2_pool_recompute.out`) | every session accepted (all 840 launches carry `outputs_exact: true`, 20 measured rounds per case and lane, summarized samples equal to the launches, one program and layout fingerprint per row, one baseline binary pair, the same host, venv, protocol and device lock); pooled statistics and statuses identical to `r107/g2_pooled.json` (gemm 1,734.825 OPEN (nominal), mvt 2,767.0 OPEN (nominal), the other three REPRODUCED); no new samples collected |
| The probes' contracts (`docs/release/scripts/probes/`) | assertions added as listed in P-26 with the tolerances unchanged (`fresh_sim_check`: `assert_allclose(rtol=1e-2, atol=1e-2)`, the Samsung suites' tolerance; `bundled_samsung_probe`: its existing `err < 0.02` kept); `fresh_install_check` additionally asserts the shipped templates, the virtual cycle count equal to the selected plan's estimate, the caller's output untouched, and no module imported from the checkout |
| Negative exercise (`negative/fake_compile_runner.py`, `r108/probes_negative.sh`, `r108/negative-stages/`): `allo.compile` replaced by a fake returning `kind=virtual`, `cycles=0`, no computed output, from the `b994fc0` wheel | every probe stage FAIL rc=1 on its own contract: `fresh_install_check` `AssertionError: (0, 0)` (cycles not the estimate), `fresh_sim_check` `AssertionError: virtual` (kind), `aim_check` `AssertionError: ('virtual', 'virtual')` (kind, backend), `upmem_check` `AssertionError: ('virtual', 'virtual')`, `bundled_samsung_probe` `AssertionError: 0.6005555` (the error of an unwritten output); `MILESTONE FAILED`, `milestone_finish` exit 1. The first attempt of this run had `aim_check` fail on an `IndexError` in the ranking print before its contract; the contract assertions were moved directly after the run so the contract fails first, and the run was repeated with the result above |
| Positive run of the repaired probes from the `b994fc0` wheel (`r108/probes_positive.sh`, `r108/probes_positive.log`, `r108/positive-stages/`) | first attempt (`positive-stages-attempt1/`): `fresh_install_check` FAIL because the leak check counted the probe script itself, now located in the checkout, as a checkout import (filter excludes `__main__`); `upmem_check` FAIL on an `sdk_build` field the UPMEM result does not record (the contract now requires the per-segment execution records the result does carry); both were my contract mistakes, not product defects, and the UPMEM run in that attempt was exact at 563,207 cycles. Second attempt with the final probe files: all five PASS, `MILESTONE PASSED`, exit 0: `fresh_install_check` virtual 45,550 cycles equal to the estimate, templates shipped, no checkout module; `fresh_sim_check` simulator 26,807 cycles, max abs error 0.0022 within tolerance; `aim_check` simulator-timing-only 222 cycles, no outputs, arrays untouched; `upmem_check` simulator 563,207 cycles, exact; `bundled_samsung_probe` 4,108 cycles, max abs error 0.0004, `pimcmdgen_calls` 0 |
| The CENT `paper_full` stage added in round 107, executed from the `b994fc0` wheel outside the checkout under the stage harness (`r108/cent_paper_full_stage.sh`, `r108/cent-stages/cent_suite_paper_full.log`) | PASS rc=0: **13 passed, 8 deselected in 267 s** (the eleven typed-trace-and-vendor-inventory rows qkvo_rope_l128, attention_l512, attention_l4096, softmax_pim_l512, softmax_pim_l4096, ffn_fc_l128, ffn_activation_l128, ffn_complete_l128, full_block_l128, full_block_l512, full_block_l4096; the RoPE EWMUL width on the neighbouring D2048 shape; the single-copy elementwise realization on the neighbouring RoPE shape); this is the first execution of these 13 cases from an installed wheel in the milestone, and it is recorded as such |
| Scope kept | no new compiler wheel, no Gemini-II session, no tolerance change; the registered pooled outcomes stand |
| Not done | the APU v1 installed-artifact vector, forced-plan and cost-perturbation execution and the three Gemini-I rows (this host's `/tmp`, about 2.2 MB); the final verdict (NOT READY) |

## Round 109 (2026-11-10): the remaining P-26 evidence checks; the absolute claim withdrawn

| Check | Result |
|---|---|
| The review's three counterexamples through the round 108 pooler | reproduced as reported: `raw-disagreement.json` (derived Tenon samples and summaries set to 1.0, raw ticks unchanged) exit 0 and five REPRODUCED; `unpaired-rounds.json` (vendor measured rounds moved by +100) accepted with the old paired ratios; `different-inputs.json` (Tenon `inputs_sha256` replaced) accepted alone and pooled with the round 106 session |
| `g2_pool.py` (round 109) | verifies the raw evidence the derivation in `benchmarks/apu_g2/paper_rows.py:195-224` rests on: `protocol.repetitions` a positive integer; every measured record carries the timing-evidence field explicitly and it is `None`; `repeated_batch_ticks` present, positive and finite, with one record per launch of the selected plan on the Tenon lane (`expected_launches` equal to the launch items and the batch records) and each launch item's `raw_ticks.pipeline` equal to its batch record; the recorded derived sample equal to `sum(raw) / repetitions`; the lane's summarized samples and median equal to the raw-derived per-round values; both lanes measured on exactly rounds 1..N; the paired vendor/Tenon ratios and their median equal to the ratios re-derived by matching round identifiers; the two lanes on one fixture (`inputs_sha256` with the inputs and the oracle, identical) and the fixture identical across sessions; the pooled samples are the raw-derived values. A recorded launch problem next to otherwise complete evidence is a rejection |
| The reviewer's counterexamples through the repaired pooler (`r109/`) | `raw-disagreement`: `gemm/tenon round 1: derived sample 1.0 disagrees with the raw ticks ([34974] / 20 = 1748.7)`; `unpaired-rounds`: `gemm/vendor: measured rounds [101, ...], rounds 1..20 required`; `different-inputs`: `gemm/tenon: inputs_sha256 records no oracle` (alone and pooled with the round 106 session); `missing-timing-evidence`: `gemm/tenon round 1: timing-evidence field absent`; a further probe of mine with the hashes changed but the oracle key kept (`r109/different_fixture_same_shape.json`) pooled with the round 106 session: `gemm inputs differs from the first session`. Every one exit 2, no output written |
| `tests/perf/test_release_pooling_script.py` | the session builder now models the producer's raw records (per-round repeated-batch ticks, launch items with `raw_ticks.pipeline`, repetitions, fixture hashes on both lanes, paired ratios and medians derived by round, several launches per plan); **39 passed**: the earlier cases plus derived-only changes, shifted vendor rounds, contradictory paired ratios and ratio median, zero, infinite and missing raw ticks, a launch item whose pipeline ticks disagree with or lack its batch record, a batch count differing from the plan, the timing field absent, a recorded launch problem, lanes on different fixtures, a fixture without an oracle, a missing fixture, missing or zero repetitions, sessions on different fixtures refused to pool, and the pooled values shown to follow the raw ticks |
| The four-session campaign recomputed (`r109/g2_pooled_recomputed.json`, `g2_pool_recompute.out`) | every session accepted on the full checks (800 measured records, raw ticks positive, each derived sample equal to its raw sum over 20 repetitions, each Tenon launch item's pipeline ticks equal to its batch record, both lanes on rounds 1..20, paired ratios equal to the by-round derivation, one fixture per row shared by both lanes and all four sessions); pooled statistics and statuses identical to `r107/g2_pooled.json`; no new samples |
| The absolute claim of round 108 | withdrawn in ISSUES P-26, REVIEW and STATUS: the scripts refuse the specific inconsistencies listed and tested, no more |
| The reviewer's UPMEM positive run (`r109/reviewer-positive-upmem_check.log`, preserved) | exact at 563,207 cycles, `UPMEM_CHECK_OK`, exit 0, followed by eight tempfile cleanup tracebacks (`Directory not empty` for `allo-c-*` scratch directories on the NFS evidence tree); recorded as the operational finding P-27, separate from the numerical result, not pursued ahead of the required work |
| Scope kept | no compiler rebuild, no Gemini-II session, no tolerance change; the fixed pooled outcomes stand (gemm 1.0009 and mvt 1.0001 OPEN (nominal), the other three REPRODUCED) |
| Not done | the APU v1 installed-artifact vector, forced-plan and cost-perturbation execution and the three Gemini-I rows (this host's `/tmp`, about 1.8 MB); the final verdict (NOT READY) |

## Round 110 (2026-11-10): the Gemini-II gemm / mvt cause investigation (none established); P-27 repaired

| Check | Result |
|---|---|
| Board headroom (the Gemini-I host is this machine) | `/tmp` 1.1 MB free at the start of the round: the APU v1 installed-artifact vector, forced-plan and cost-perturbation execution and the three Gemini-I rows stay blocked; nothing launched, no other user's files touched |
| Method of the causal investigation (`r110/compare_generated_work.py`, host-only: compile and recipe construction, no card launch) | gemm and mvt compiled through `allo.compile` at the final candidate (`PAPER_KERNELS`, `build_apu_g2_target`, `apu_g2_cost`, 20 repetitions as the benchmark sets) and their plan, layout manifest, program manifest, recipe manifest, ordered operation stream, inventory and fingerprints compared with the frozen campaign's archived Tenon-lane provenance (`tenon-artifacts/apuv2-composed-v2/provenance/implementations/tenon/cases/<case>/{operations,layout}.json`, `generated-source.txt`, the archived source files by sha256) and with the archived measurement records (`tenon/<case>/runs/measured-r*/measurement.raw.json`, `results.json`) |
| Generated work, gemm | plan `single/k1`, one launch; program structural fingerprint `86b7bdd981cf52d2...` equal to the archived; layout fingerprint `20a85096d2105837...` equal; recipe fingerprint `4671896b85d3e297...` equal; ordered operation stream equal (`COPY_L1_VECTORS_TO_MMB` x2, `MUL_TYPED`, `COPY_MMB_TO_L1_VECTORS`, `COPY_L1_VECTORS_TO_MMB`, `GROUP_REDUCE_ADD_TYPED`, `COPY_MMB_TO_L1_VECTORS`, `SEU_BARRIER`), inventory equal; program manifest and recipe manifest equal at 20 repetitions. **Record correction (round 111, the reviewer's finding):** the round 110 script `r110/compare_generated_work.py` as first run did not set the benchmark's repetition policy, so its saved dumps carry the default 256 and `r110/generated-work/comparison.json` reports the two manifest equalities false (the 20-repetition equality in this row had been established by a separate interactive run that was not preserved); the script now sets `compiled.repetitions = REPETITIONS` through the public setter before reading the plan and performs every comparison itself, and its corrected run is preserved beside the original in `r110/generated-work-corrected/{gemm,mvt}_dump.json`, `report.json` (every equality `True`, repetitions 20 on both sides); the original `r110/generated-work/` is kept unchanged as the stale run |
| Generated work, mvt | plan `fused/k1`, one launch; program fingerprint `0744614cfd9a9aed...`, layout `a06509fdcb3496c7...`, recipe `45971069ac979c13...` all equal to the archived; the 12-operation stream equal (the gemm stream plus `COPY_L1_VECTORS_TO_MMB` x2, `ADD_TYPED`, `COPY_MMB_TO_L1_VECTORS` before the barrier), inventory equal; program and recipe manifests equal at 20 repetitions (`r110/generated-work-corrected/mvt_dump.json`, `report.json`; the round 110 `r110/generated-work/` run is the stale one described above) |
| The executing sources (`r110/source-diffs/`) | `device/apu_g2_composed_dot.cc` token-identical to the archived device kernel (formatting only); `host_composed_dot.cc` differs in formatting and in `DeviceConfig{TENON_APU_G2_DEVICE_INDEX, 0}` against `{0, 0}` (the index is 0 on the card used); `device/gsi_library.h` drops the entry points of unrelated u16 kernels the archived library also carried; the CMake files parametrize the SDK root and drop those kernels; the device build keeps `-O3 -DNDEBUG` under `GSI_SYSTEM_CMAKE_BUILD_TYPE Release` in both eras and the host is configured `-DCMAKE_BUILD_TYPE=Release` now as the archived `flags.make` shows `-O3 -DNDEBUG` then; `apu_g2_typed_bits.h` identical |
| The timing boundary | both eras: a batch of 20 consecutive pipeline bodies plus one completion barrier measured with `gsi_perf_start/end`, divided by 20, a separate final body plus barrier retained; packing, transfers, gather and the oracle check outside the timed region; the device library and host binaries are rebuilt from the candidate wheel so their hashes differ from the archived ones (`8ea3600c...` / `c85f4970...`) as expected |
| The distributions (paper session n=20 against the pooled current n=80) | gemm: Tenon paper median 1,733.225 (IQR 1,723.2 to 1,751.3) against current 1,734.825 (IQR 1,723.2 to 1,746.6), shift +0.09 percent; GTML paper 1,903.85 against current 1,917.35, shift +0.71 percent. mvt: Tenon paper 2,766.625 (IQR 2,757.6 to 2,777.9) against current 2,767.0 (IQR 2,752.9 to 2,774.9), shift +0.01 percent; GTML paper 3,089.65 against current 3,095.225, shift +0.18 percent. Each era's median lies inside the other's IQR for both rows; the paired GTML/Tenon ratio is above the paper's on both (1.1015 against 1.0984; 1.1195 against 1.1168) |
| **Conclusion** | the compiler emits the same program, layout, recipe and operation stream as the frozen experiment, the device kernel is the same source, and the timed unit is the same; the residual +0.09 and +0.01 percent on the Tenon lane is smaller than the shift the vendor lane shows in the same rounds. **No production cause is established**, and none is claimed; the observations are consistent with card or environment drift between the campaigns. gemm and mvt stay OPEN (nominal) under the completed campaign's fixed rule; no further sessions, no rule change. Recorded in X-34 and X-37 |
| P-27 repair (`allo/backend/c.py`) | the portable C module unloads its compiled library before removing its scratch directory (a `close()` method and a finalizer registered after the `TemporaryDirectory`'s own, so it runs first at exit) and the directory is created with `ignore_cleanup_errors`; `tests/test_c_backend.py` + `tests/test_literal_typing.py` + `tests/test_bitwise_invert.py`: 58 passed with `LLVM_BUILD_DIR` set (without it 23 of them fail identically with and without the patch, an environment matter); `tests/pim/test_upmem_dtypes.py` passed with the backend tests (36 in that run); the UPMEM probe from the patched checkout (`r110/p27/upmem_check_patched.log`): exact at 563,207 cycles, `UPMEM_CHECK_OK`, exit 0, **0 tracebacks, 0 `allo-c-*` directories left** (before: 8 tracebacks, `r108/positive-stages/probe_upmem_check.log` and the reviewer's run) |
| The dependency this change creates | stated in P-27 before any candidate validation: the first compiler-source change after the final candidate `b994fc0`, in `allo/backend/c.py` only; `git diff --stat b994fc0 -- allo/pim/apu_g2_* allo/spmw_apu_g2.py benchmarks/apu_g2 allo/pim/templates/apu_g2` is empty, so the completed Gemini-II campaign measured programs identical to this revision's and its fixed outcomes stand; the revision needs installed-artifact validation of the changed backend (the UPMEM probe and rows from a wheel) before it replaces `b994fc0` as the packaged candidate, not yet done |
| Not done | the APU v1 installed-artifact vector, forced-plan and cost-perturbation execution and the three Gemini-I rows (this host's `/tmp`); the installed-artifact validation of this revision's portable C backend; the final verdict (NOT READY) |

## Round 111 (2026-11-10): the P-27 lifecycle crash repaired; the owed installed-artifact validation of the changed backend

| Check | Result |
|---|---|
| The reviewer's crash (`reviewer-r110/close_probe.py`: a compiled add-one kernel executes, closes twice, then is called again; SIGSEGV, return code -11) | reproduced as reported against `a7dbe67`: the round 110 repair unloaded the library on `close` and left the function pointer callable |
| Repair (`ab239e3`, `allo/backend/c.py`) | `close()` is idempotent (the finalizer is one-shot), drops the library and function references, `closed` is a property, and `__call__` raises `PortableCError` naming the closed module and the remedy before touching the pointer; the cleanup behaviour of round 110 is kept. Lifecycle regression `test_compiled_module_lifecycle_close_is_idempotent_and_use_after_close_is_refused` (`tests/test_c_backend.py`): execution exact, scratch directory gone after `close`, second `close` a no-op, the call after `close` refused with the outputs untouched. `tests/test_c_backend.py` + `tests/test_literal_typing.py` + `tests/test_bitwise_invert.py` + `tests/pim/test_upmem_dtypes.py`: **79 passed** (`LLVM_BUILD_DIR` set). The reviewer's reproducer against the patched checkout (`r111/close_probe.log`): the call after `close` ends in `PortableCError: compiled module 'add_one' is closed ...`, process exit 1, no signal |
| The candidate artifact (`r111/build_wheel.sh`, `build_wheel.log`, `pkg/`) | revision `ab239e3` (`pkg/revision.txt`, clean tree), sdist `19a80d2f7cae8e0a8b8031b469dfc0d8093638601bae2d65125ec20f9c376de7`, wheel `d280a4d3fe4ba1d6d5dfeba3ee0da5f1d579625ad8fada75abb837835dbc079f`, the recipe and license texts carried as before (84 bundled recipe entries, both license texts), installed into a clean venv |
| Installed stages from the `ab239e3` wheel, outside the checkout, no `PYTHONPATH` into it (`r111/run_upmem_installed.sh`, `run_upmem_installed.log`, `r111/stages/`) | `module_origin` PASS (the package from `pkg/venv`, nothing from the checkout on `sys.path`); `probe_upmem_check` PASS: kind `simulator`, backend `upmem`, **563,207 cycles, exact**, `UPMEM_CHECK_OK`; `upmem_probe_cleanup_behaviour` PASS: **0 tracebacks, 0 `allo-c-*` directories left** in `TMPDIR` after exit; `close_probe_from_wheel` PASS: the reviewer's reproducer from the installed package ends in `PortableCError ... is closed`, exit 1; `c_backend_tests` PASS: `tests/test_c_backend.py` + `tests/pim/test_upmem_dtypes.py` 37 passed from the wheel (the lifecycle regression included); `upmem_rows` PASS: the sixteen UPMEM program-provenance rows and the three default-selection cycle rows (va, red, hist) 19 passed, 13 skipped (the `paper_full` selection), 0 tracebacks. The first run was interrupted during the `paper_full` stage by my own shell's ten-minute limit (recorded as such; the harness had already written the six stage statuses above); that stage was relaunched on its own (`r111/run_upmem_paper_full.sh`, `stages-paper-full/`), recorded in the next row |
| The UPMEM `paper_full` rows from the `ab239e3` wheel (`r111/run_upmem_paper_full.sh`, `run_upmem_paper_full.log`, `stages-paper-full/`) | `upmem_rows_paper_full` PASS: **13 passed in 508 s** (mtv, gemv, geva, ttv, mmtv, sel, kmeans, linear_reg, logistic_reg, 1mm, 2mm, 3mm, conv; with va, red and hist from the default-selection stage all sixteen UPMEM cycle rows pass from this wheel), 0 tracebacks in the log. `upmem_cleanup_after_rows` FAIL rc=1: one `allo-c-*` directory present in `TMPDIR`; its mtime is 17:24:31, inside the first run's interrupted `paper_full` stage (begun 17:17:03, killed with the shell at about 17:24:43) and before the relaunch began (17:25:08), and `find -newer` the relaunch's summary finds no scratch directory: the orphan is from the process my shell limit killed (a killed process runs no finalizer), not from the repaired cleanup, whose own window left nothing. Recorded as a failed stage with that cause; the stage's rule counted every directory, which is what made the interruption visible. The orphan is left in place as evidence |
| What this round does and does not establish | the backend change is validated from an installed artifact of its own revision (`ab239e3`, wheel `d280a4d3...`): the UPMEM probe and all sixteen UPMEM rows exact, the cleanup behaviour as designed, the use-after-close refusal from the installed package. It does not by itself replace `b994fc0` as the packaged candidate, does not touch the Gemini-II campaign (modules byte-identical; the fixed OPEN (nominal) outcomes stand with no new sampling and no rule change), and closes no release gate |
| Not done | the APU v1 installed-artifact vector, forced-plan and cost-perturbation execution and the three Gemini-I rows (this host's `/tmp`, about 1 MB); the final verdict (NOT READY) |

## Round 112 (2026-11-10): candidate identity consolidated; U-08 / P-08 reconciled against the current source

| Check | Result |
|---|---|
| Board headroom (this host) | `/tmp` 1.1 MB free: the bounded Gemini-I campaign and the APU v1 installed-artifact execution are not launched; the blocker stands; no other user's files touched |
| Candidate and artifact identity | consolidated in `STATUS.md` ("Candidate and artifact identity"): `b994fc0` the packaged and final compiler candidate (local wheel `5ba2e348...` built from the code-identical `1b4282d`, zhang-x4 wheel `d5cb59f8...`) with the fresh evidence of rounds 106 to 108; `ab239e3` the P-27 backend repair (wheel `d280a4d3...`), fresh for the UPMEM probe, the backend tests and the sixteen UPMEM rows, inheriting every other backend's installed evidence from identical sources and the Gemini-II campaign's fixed outcomes from byte-identical programs; `c21b510` and later records only. No artifact rebuilt for this record |
| P-08 verified against the current source | `SPMW_ARCHITECTURE.md` (1,028 lines): grep finds no `upmem_program`, `upmem_abi`, `apu_v1_hybrid_runtime` or `UPMEMProgram`; every `allo/*.py` path it names exists (67 distinct paths checked, 0 missing, by an executed loop); section 3.3 describes the live uPIMulator route and section 3.5 Gemini-II / APU v2; sections 4, 5 and 7 are the extension, inspection and module-map sections the audit asked for. The one defect found: the AttAcc section numbered 3.6 sat between 3.2 and 3.3; moved after 3.5 (the cross-references to 3.3 and 3.4 at lines 303 and 1017 unchanged). **Closed prematurely (amended in round 113):** this check established module-file existence, not API accuracy; the review found four deleted result wrappers and a removed route entry still named as current (round 113 below) |
| U-08, the documentary portions | `design/README.md` already lists, per note, the statements no longer true of the tree; `06-upmem-reference-backend.md` and `08-upmem-polybench-execution-matrix.md` now open with a SUPERSEDED banner naming the live route and pointing to the index and `docs/release/design/upmem-domain.md`; the skill document `.claude/skills/pim-upmem/SKILL.md` (containing repository, edited in place, not committed there) now names `allo/spmw_upmem.py` as the lowering with the Python host and `GENERIC` fixtures and marks its `TENON` drop-zone section historical. **Closed** |
| U-08, what remains open | `allo/pim/__init__.py` exports the UPMEM plan classes (reached by `UpmemCtx.emit_program`) and the physical-search entry points `rank_upmem_physical_candidates` / `select_upmem_physical_plan` (reached only from `scripts/prepare_upmem_tenon_campaign.py` and `tests/perf/test_upmem_physical_search.py`, not from `allo.compile`); whether they stay public is a maintainers' API decision, left as is and recorded; `docs/source` has no UPMEM page (0 files mention UPMEM) |
| Not done | the APU v1 installed-artifact vector, forced-plan and cost-perturbation execution and the three Gemini-I rows (this host's `/tmp`); the final verdict (NOT READY) |

## Round 113 (2026-11-10): the P-08 closure corrected by a symbol check; the backend-repair revision's installed validation started

| Check | Result |
|---|---|
| The review's finding on round 112 | reproduced: `SPMW_ARCHITECTURE.md` section 1.4 named `APUv1VectorCallable`, `APUv1ProgramCallable`, `APUG2TypedCallable`, `APUG2ComposedContractionCallable` as return types, and the Gemini-II source-route bullet named `compile_apu_g2_source_workload`; none of the five is defined or used anywhere in `allo/`, `benchmarks/`, `scripts/` or `tests/` sources (grep over `.py` files excluding bytecode: 0 definitions, 0 uses). `allo.compile` returns `CompiledCallable` for every input (`allo/compiler.py`: the APU v1 program record and the Gemini-II records enter through their record frontends and the common pipeline); the Gemini-II source route is entered through the `autoschedule` enumerator `apu_g2_source_candidates` (`allo/spmw_autoschedule.py` -> `allo/pim/apu_g2_source_route.py`) |
| Corrections | `SPMW_ARCHITECTURE.md` 1.4 now states the single `CompiledCallable` return (the `CompiledWorkload` protocol's one implementation) with the attributes it carries for every route, and names the four wrappers only as removed; the 3.5 source-route bullet names the enumerator entry and the removed function as removed. `docs/release/design/apu-g2-source-route.md` section 2 describes the common-pipeline entry; `apu-v1-cost.md` points the tile decision at the plan's `metadata["tile_decision"]` reachable from `CompiledCallable`; `reduction-policy.md` names `CompiledCallable.reduction_order` / `.reduction_policy_pruned` for APU v1 and `apu_v1_vector_program._run_functional` for the host model. `shared-compilation-path.md` keeps its mentions: its removal note (round 13) and its tables labelled round 12 state and round 8 inventory are history, not description |
| Symbol check (`r113/architecture_symbol_check.txt`) | every single-token code span of `SPMW_ARCHITECTURE.md`, split at `::`, `.`, `/` and brackets, resolved against class/def/module-level assignments, module files, quoted keys and attribute names in `allo/` and `benchmarks/` sources only: 692 distinct tokens, 28 unresolved, each accounted for: the five deleted names (inside the sentences that call them removed), Python builtins and stdlib (`KeyError`, `ValueError`, `NotImplementedError`, `functools`, `math`), vendor SDK symbols used by the templates and the APU v1 route (`DeviceConfig`, `gvml_add_subgrps`), `layout_swizzle` (`samsung_layout_swizzle.py`, `scripts/run_pimsim_layout_trace.py`), directory names (`design`, `latex`, `spmw`, `task`), and eleven test module names, each present under `tests/`. My first two attempts at this check were themselves wrong (one treated any identifier occurrence as a definition, the other split prose inside multi-word spans) and were discarded; the preserved file is the third |
| What the check does not establish | it resolves names, not signatures, semantics or call-site correctness; prose claims about behaviour remain reviewed by reading |
| Installed validation of the backend-repair revision `ab239e3` (started this round; the artifacts of `r111/pkg`, wheel `d280a4d3...`) | local: `run_installed_milestone.sh r113` running under the stage harness (bundled Samsung verified, built from the wheel and discovered through `TENON_HOME`, the stripped driver identity, the bundled GEMV probe PASS at the time of this entry; the remaining stages recorded when complete); zhang-x4: `g2_chain_no_session.sh` building the wheel there and running the card tests, **no validation session and no campaign** (the Gemini-II modules are byte-identical to `b994fc0`; the completed campaign's fixed outcomes stand and no sample is added) |
| Board headroom | `/tmp` 1.1 MB on this host; the Gemini-I rows and the APU v1 installed execution stay blocked |
| **The `ab239e3` installed-artifact milestone, local part, complete** (`r113/run_installed_milestone.log`, `r113/stages/`, wheel `d280a4d3...` from `r111/pkg`, `TENON_HOME` fresh under `r113`) | **`MILESTONE PASSED`, every required stage exited 0** (18:36:11): package from the venv with every backend module, nothing from the checkout; the bundled Samsung recipe verified, built from the wheel (geometry printed), discovered through `TENON_HOME`, the stripped `pim_driver` the evaluation identity `9aea49bf...`; the bundled GEMV probe 4,108 cycles, `pimcmdgen_calls` 0; the five contract probes PASS (virtual 45,550 equal to the estimate; Samsung simulator 26,807 cycles within tolerance; AiM timing-only 222 cycles with arrays untouched; UPMEM exact 563,207 with 0 cleanup tracebacks and 0 scratch directories left; the bundled route); golden rows 26 passed (Samsung 4,108 / 7,057 / 7,255 against the paper's 4,435 / 8,041 / 8,247 with the vendor placements at the paper's values; AiM corrected programs 7,052 and 1,686); `paper_full` golden rows 24 passed; Samsung conductor runs and GEMV on every target 14 passed; CENT 8 passed and CENT `paper_full` 13 passed; the shared-contract set 77 passed, 1 skipped; APU v1 functional reference 10 passed in the clean venv (which had installed unpinned numpy 2.5.4, not the 2.3.5 this row claimed until round 120) and 13 passed under NumPy 2.5.4 in `venv-np24`. Identical results to the `b994fc0` milestone (rounds 106 and 108) on every stage |
| The `ab239e3` chain on `zhang-x4` (`r113/g2-chain.log`, `r113/g2/stages/`, `stages-copy-resume/`) | wheel `11f659ba...` built there from the same sdist and installed into a clean venv; **card tests 15 passed in 125 s** from it; the copy stage FAILED in the chain because the repository script required a root `tests/conftest.py` that does not exist (the content had been copied; the card tests ran on it); the script now tolerates the missing file and the stage was re-run alone and PASSED (`stages-copy-resume`); the chain's `MILESTONE FAILED` line stays in its log as it was. **No Gemini-II validation session and no campaign were run at `ab239e3`**, deliberately: the Gemini-II modules, templates and benchmark are byte-identical to `b994fc0` and the completed campaign's fixed outcomes are not to be re-sampled |
| What `ab239e3` now has and lacks as a packaged-candidate replacement | has: every local installed stage the `b994fc0` milestone had, the Gemini-II card tests from its own zhang-x4 wheel, and the UPMEM probe and rows with the P-27 behaviour (round 111); lacks: a Gemini-II validation session of its own (withheld by design, the programs byte-identical) and the Gemini-I / APU v1 hardware evidence (unavailable at any revision). Whether byte-identity of the Gemini-II modules lets the `b994fc0` campaign stand for `ab239e3` is the reviewer's call; until it is made, `STATUS.md` keeps `b994fc0` as the packaged candidate and lists `ab239e3` as validated and proposed |
| Not done | the APU v1 installed-artifact vector, forced-plan and cost-perturbation execution and the three Gemini-I rows (this host's `/tmp`, 1.0 MB); the final verdict (NOT READY) |

## Round 114 (2026-11-10): the AttAcc figure evidence attributed correctly and re-executed; the copy semantics made explicit

| Check | Result |
|---|---|
| Candidate decision (the reviewer, round 114 review) | `ab239e3` is the packaged working candidate for the remaining work; the `b994fc0` Gemini-II campaign is inherited evidence with its revision, artifact identities, samples and OPEN gemm / mvt outcomes preserved (`STATUS.md`, the identity table); no further campaign |
| The AttAcc attribution error (the reviewer's finding) | the `b994fc0` and `ab239e3` milestones' AttAcc stage ran `tests/pim/test_attacc_virtual.py`, `test_attacc_orientation.py` and the allocation and access-form sets: analytical and contract tests, no simulator. `REVIEW.md`'s verdict block and `STATUS.md`'s identity table said the AttAcc rows were reproduced from the `b994fc0` wheel and that no simulator evidence was inherited; both corrected. The last re-execution of the twelve AttAcc figure results from an installed wheel was at `88852b7` in round 84 (`r84/attacc_*.json`, `r84/attacc_e2e.json`, identical to the `r57` baseline at `54cbb08`), the checkout execution at round 51 / 52 before it |
| `attacc_compare_frozen.py` (new, `docs/release/scripts/`) | compares a session set with the frozen figures through the inventory's paper values: the four GEMV rows' selected Ramulator2-AttAcc milliseconds against the paper's (the round 114 version allowed a symmetric 0.15 percent, which the round 115 review showed is not the rounding interval of the printed five-decimal values and which let a slower 0.03368 ms pass against 0.03364; replaced in round 115 by the same-or-better rule against the archived per-shape Tenon timing, see round 115) with the fixed-form baseline re-run; the two score rows' mean of per-shape fixed / Tenon over 248 shapes (the figure's statistic) against the figure's value, with the fixed and DCC per-shape times from the archived `tenon_manifest.csv`; the text's 500-instance tally against DCC over all 500 kernels (ties at equal milliseconds); the six decode configurations' base and full speedups against the paper's; every selected run must be `simulator-timing-only` with a matching command-by-command trace check and totals (the fixed-form baseline carries no trace check by design). The round 114 version did not in fact fail on every missing row: an empty decode configuration list and score records relabelled away from `shared-pipeline` with their trace checks removed both passed (the round 115 review's reproductions); the round 115 version requires the complete evidence set, see round 115. Checked against the round 84 sessions first: every row REPRODUCED, the tally 123 / 316 / 61 with median loss 7.07 percent, `ALL REPRODUCED` (`r114_precheck_r84.json`); two mistakes of my first draft (requiring a trace check of the baseline runs; tallying the score rows only, 121 / 314 / 61) were corrected before any new session was read |
| `run_installed_milestone.sh` | gains the AttAcc figure stages: `run_ramulator --row qkv_b1 / qkv_b4 / proj_b1 / proj_b4 --fixed-baseline`, `--row score_b1 / score_b4` (248 shapes each), `compose_e2e` over the six sessions, and `attacc_compare_frozen.py`; traces, configurations and raw outputs retained under `attacc/*.artifacts`; the natively built simulator and the DCC trace generator at the paths of `r84/run_installed.sh` (`ATTACC_RAMULATOR`, `ATTACC_RAMULATOR_LIBDIR`, `ATTACC_DCC_ROOT`), their identities recorded in each session (`simulator.binary_sha256`, `trace_generator.files_sha256`). The test-tree copy is now a stage (`copy_test_tree`) through the new helper |
| `copy_tree_with_optional` (`stage_lib.sh`) and the chain's copy stage | the round 113 expression `[ -f x ] && cp x || true` returned success for a failed copy (the reviewer reproduced it); the helper copies every mandatory path or fails, copies an optional path when present, tolerates its absence, and fails on a copy error; `g2_remote_chain.sh` and the milestone use it. Exercised under the harness (`r114/copy-smoke/stages`): optional absent PASS (`pytest.ini`, `conftest.py` reported tolerated), mandatory missing FAIL rc=1, optional present but unreadable FAIL rc=1 (`copy failed`), optional present copied PASS; the milestone FAILED as it must |
| **The AttAcc figure stages from the `ab239e3` wheel, outside the checkout** (`r114/attacc_stages.sh`, `attacc_stages.log`, `r114/stages/`, `r114/attacc/` with `attacc_*.json`, `attacc_e2e.json`, `attacc_compare_frozen.json` and `*.artifacts`, 184 MB of retained traces, configurations and raw outputs) | `MILESTONE PASSED` (18:53 to 19:41): package from `r111/pkg/venv` (wheel `d280a4d3...`), nothing from the checkout; test tree copied through the new helper; simulator `Ramulator2-AttAcc` binary `01e857a78e59dbd1...` with the DCC trace generator files `gen_trace_AttAcc_GEMV.py` `220a1a22...`, `gen_trace_AttAcc_ATTN.py` `1c313e66...`, `generate_configs.py` `0e5051f2...` recorded in every session; 504 runs, 0 failures, every one of the 500 shared-pipeline runs with a matching command-by-command trace check and totals (the 4 fixed-form baseline runs carry none by design). **Frozen-figure comparison (`attacc_compare_frozen`): every required result REPRODUCED**: qkv_b1 0.008808 ms (paper 0.00881, ratio 0.9998; fixed 0.028072, 3.19x), qkv_b4 0.033644 (0.03364, 1.0001; fixed 0.172387, 5.12x), proj_b1 0.002652 (0.00265, 1.0006; fixed 0.002652, 1.00x), proj_b4 0.010242 (0.01024, 1.0002; fixed 0.010303, 1.01x); score_b1 1.000x over fixed (figure 1.00x) and score_b4 1.128x (figure 1.13x) over 248 shapes each; the 500-instance tally against DCC 123 / 316 / 61 with median loss 7.07 percent (the text's 123 / 316 / 61, 7.1 percent); decode base / full 1.47 / 6.50, 1.56 / 6.95, 1.66 / 7.40, 1.34 / 3.09, 1.50 / 3.48, 1.65 / 3.87 against the paper's same values, `ALL REPRODUCED`, exit 0. Continuity with the round 84 sessions at `88852b7`: the score rows and the composition have 0 differences excluding timestamps, paths, command lines, config digests and wall times; the four GEMV rows differ only in the kernel record's new `matrix` field (`lW`, the operand name the X-41 orientation work added), tilings and milliseconds identical |
| What this establishes | the twelve AttAcc figure results now have fresh installed-artifact evidence at the packaged working candidate `ab239e3`; the inventory rows are updated to this revision. It is simulator evidence on the natively built Ramulator2-AttAcc, not hardware; the Gemini-I rows, the APU v1 installed execution and the two OPEN Gemini-II rows are unchanged |
| Not done | the APU v1 installed-artifact vector, forced-plan and cost-perturbation execution and the three Gemini-I rows (this host's `/tmp`, about 1 MB); the final verdict (NOT READY) |

## Round 115 (2026-11-10): the AttAcc comparator made unable to certify incomplete or relabelled evidence (P-28)

| Check | Result |
|---|---|
| The review's reproductions against the round 114 comparator (on temporary copies of `r114/attacc`) | reproduced before the repair: `attacc_e2e.json` with `configs: []` gave exit 0, `all_reproduced: true`, `decode: {}`; the 496 score records relabelled `forced-tiling` with their `trace_check` fields deleted gave `ALL REPRODUCED`; qkv_b4 changed to 0.03368 ms against the printed 0.03364 passed under the symmetric 0.15 percent allowance |
| `attacc_compare_frozen.py` (round 115, rewritten) | evidence contract: GEMV rows need exactly one `shared-pipeline` run of the row's archived shape with a matching trace check and at least one `fixed-form-baseline` run, no runs of another selection; score rows need exactly the archived manifest's attention shapes of the row's head count, each once, every record `shared-pipeline` with a matching trace check, with records of another selection, another row's shapes, duplicates and missing shapes rejected by name; the composition must be `analytical-composition` with exactly the six expected configurations and no duplicates; all twelve verdicts must be present and REPRODUCED and no problem recorded for success. Performance rule, same-or-better: a selected timing REPRODUCES at or below the archived per-shape Tenon timing of the frozen experiment (`tenon_time_ms`, deterministic; the printed value plus half a unit of its last printed digit only where no archived timing exists); score-row and decode speedups REPRODUCE at or above the printed figure less half a unit of its last digit (`half_ulp("0.00881") = 5e-6`; the example `half_ulp("6.5") = 0.05` given here in round 115 described the inventory's lossy string, not the paper: the frozen table prints `6.50`, so the bound is 6.495, corrected in round 116); improvements pass, regressions fail. The DCC tally is reported as a diagnostic (`matches_text`) and does not decide |
| The round 114 sessions through the repaired comparator (`r115/attacc_compare_frozen_r114.json`) | `ALL REPRODUCED`, exit 0: the four GEMV rows at or below their archived timings (0.008808126, 0.03364375, 0.002651512, 0.010241542 ms), 248 of 248 shapes per score row with 0 slower than archived and the figure speedups met, the six decode configurations at or above the printed values, the diagnostic tally 123 / 316 / 61 over 500 kernels matching the text. The `r114/attacc/attacc_compare_frozen.json` written by the round 114 version is left in place as what that version produced |
| Mutations on temporary copies (`r115/mut_*`) | `empty_configs`: exit 1, 7 problems (six configurations missing, six verdicts missing); `dup_config`: exit 1 (duplicate and missing configuration); `relabel_score`: exit 1, 498 problems (every relabelled record rejected, every shape missing); `missing_shape`: exit 1 (`1 of 248 archived shapes missing`); `slower_qkv_b4` (0.03368 ms): exit 1, verdict `DIFFERS (slower)` with no evidence problem; `faster_qkv_b4` (0.0330 ms): exit 0, REPRODUCED |
| `tests/perf/test_attacc_compare_frozen.py` (synthetic sessions, manifest and inventory, 11 passed) | complete evidence reproduces; empty decode configurations fail with the missing configurations and verdicts named; duplicate configuration and wrong composition kind fail; relabelled score records without traces fail; missing, wrong-row and duplicate score shapes fail by name; a score record without a trace check fails; a GEMV regression fails while an improvement passes; a score shape slower than archived fails; wrong GEMV shape, missing baseline and missing trace check fail; a missing session and recorded failures fail; the half-ulp and printed-number helpers behave as specified |
| Record corrections | the round 114 rows no longer claim that 0.15 percent is the paper's rounding or that every missing row already failed (amended in place with pointers here); the simulator evidence of round 114 and the candidate identity are unchanged |
| Not done | the APU v1 installed-artifact vector, forced-plan and cost-perturbation execution and the three Gemini-I rows (this host's `/tmp`, under 1 MB); the final verdict (NOT READY) |

## Round 116 (2026-11-10): the comparator checks its evidence against itself (P-28 follow-up)

| Check | Result |
|---|---|
| The review's reproductions against the round 115 comparator (`reviewer-r115/probe_comparator.py`) | reproduced before the repair: a consistent-looking `6.46` full speedup and a `1.46` base speedup passed because the inventory stored `6.5`, `7.4` and `1.5` where the frozen table prints `6.50`, `7.40` and `1.50` (bounds 6.45 and 1.45 instead of 6.495 and 1.495); qkv_b4's raw cycle count multiplied by ten with milliseconds unchanged passed; a decode `full_ms` multiplied by ten with the speedup unchanged passed; an infinite speedup passed; every qkv_b4 fixed baseline relabelled to shape `[1, 1, 1]` passed |
| Printed precision | `inventory.csv` now carries the decode table's printed digits for the three rows (`base 1.47x, full 6.50x`, `base 1.66x, full 7.40x`, `base 1.50x, full 3.48x`; `paper/latex/tables/eval-attacc-e2e-detail.tex`, read, not edited); the comparator's bounds on the round 114 sessions are now 6.495 / 1.465, 6.945 / 1.555, 7.395 / 1.655, 3.085 / 1.335, 3.475 / 1.495, 3.865 / 1.645 (`r116/attacc_compare_frozen_r114.json`); a test reads the real inventory and requires two printed decimals on every decode value and five on every GEMV millisecond |
| Evidence consistency (`attacc_compare_frozen.py`, round 116) | every run: positive finite milliseconds and integer cycles, milliseconds equal to `memory_system_cycles x tCK / 1e6` at the session's `simulator.tck_ns` (required positive finite), the row's family; every selected run: the retained raw-output directory present, its `stdout.txt` reporting the same `memory_system_cycles`, its `kernel.trace` hashing to the record's `trace_sha256`; fixed-form baselines: the row's family and shape, timing consistency, unrelated records rejected by name and a row without a baseline of its shape refused; decode: `gpu_roofline_ms`, `base_ms`, `full_ms`, `base_speedup`, `full_speedup` positive finite and the speedups equal to the roofline over the recorded latencies |
| The round 114 sessions through the round 116 comparator | `ALL REPRODUCED`, no problem: the 500 selected runs' cycles, clock, retained outputs and trace hashes agree with their records, the four baselines are the rows' workloads, the six compositions' ratios agree with their latencies; no new simulator sampling |
| The reviewer's probe (`reviewer-r115/probe_comparator.py`) after the repair | `unchanged` rc 0; `rounding_full_6_46` and `rounding_base_1_46` rc 1 (the mutated speedups no longer equal roofline over latency; a consistent 6.46 fails the 6.495 bound, shown in the tests); `tenfold_raw_cycles` rc 1 (milliseconds against cycles, and the retained output's cycle line); `tenfold_decode_latency` rc 1; `wrong_baseline_shape` rc 1 (records rejected, no baseline of the row's shape); `infinite_decode_speedup` rc 1 (`not positive finite`) |
| `tests/perf/test_attacc_compare_frozen.py` (18 passed; the fixture's timings are exact multiples of the clock with retained outputs on disk) | the earlier eleven plus: a consistent decode speedup below the printed value less half an ulp fails on the real frozen strings (6.46 against 6.50, 1.46 against 1.50) while 1.495 passes; decode fields must agree and be finite; kernel cycles, clock and retained output must agree; the retained output directory is required and a tampered trace is caught; baselines must be the row's shape and family; a missing simulator clock fails; the inventory carries the tables' printed precision |
| What the comparator establishes and does not | it establishes that the recorded results are internally consistent with their raw simulator evidence and at or better than the frozen experiment's archived and printed values; it does not re-simulate or re-generate traces (the reviewer's independent reconstruction of the 500 traces is the evidence of that) |
| Not done | the APU v1 installed-artifact vector, forced-plan and cost-perturbation execution and the three Gemini-I rows (this host's `/tmp`, about 1 MB and briefly 0 during this round, which also stalled my own shell); the final verdict (NOT READY) |

## Round 117 (2026-11-10): the release scripts' license headers; the PIM user-documentation page (S-11)

| Check | Result |
|---|---|
| License headers (`scripts/lint/check_license_header.py`) | `check_license_header.py b994fc0` named sixteen files added under `docs/release/scripts/` in rounds 107 to 115 (the probes, the stage harness, the chain and milestone scripts, the pooler, the comparator, the negative runner); the Allo header added to each (after the shebang in the shell scripts), `check_license_header.py b994fc0` and `HEAD~1` now `all checks passed`; `bash -n` and `py_compile` clean, Black unchanged; commit `94e435f`, no behaviour change |
| Board headroom (this host) | `/tmp` 3.4 MB free throughout the round: the APU v1 installed vector, forced-plan and cost-perturbation execution and the three Gemini-I rows stay blocked; nothing launched, no other user's files touched |
| S-11 (no Samsung docs page; stale skill doc; simulator discovery outside package) | `docs/source/backends/pim.rst` added and linked from the Backends toctree: the compile entry and the single `CompiledCallable`, a table of the six targets (constructor and cost program, executor, `RunResult.kind`), installation and discovery per executor (the bundled Samsung build through `allo.pim.samsung_simulator` and `TENON_HOME`; AiM, UPMEM, Gemini-I, Gemini-II variables with pointers to `INSTALL.md` section 3 and `simulators.md`), inspecting a chosen program, and where the capability and reproduction ledgers are; every name the page cites checked against the sources (`build_*_target` in `allo/pim/targets.py`, the cost programs in `allo.pim.costs`, `placement_ranking`, `execution_graph`, `reduction_policy_pruned`, `backend="functional"`, the environment variables in `spmw_simenv.py`). The Samsung skill document's vendor-CLI, 4096-hard-coding and `tn.get_wid` passages replaced by the `@allo.work` region form, the `--op GENERIC` conductor and the bundled build (containing repository, in place, not committed there). Discovery has been inside the package since P-04. **S-11 closed**; the UPMEM page gap noted under U-08 is covered by the same page. Not built with Sphinx (docutils unavailable here); the page follows the existing `backends/simulator.rst` form |
| Scope kept | no compiler, package or benchmark source changed (`git diff --stat ab239e3 HEAD -- allo setup.py simulators benchmarks` empty); `ab239e3` stays the packaged working candidate; the `b994fc0` Gemini-II campaign and its OPEN gemm / mvt outcomes untouched |
| Not done | the APU v1 installed-artifact vector, forced-plan and cost-perturbation execution and the three Gemini-I rows (`/tmp`); the final verdict (NOT READY) |

## Round 118 (2026-11-10): the four documentation corrections; V1-20 resolved at enumeration

| Check | Result |
|---|---|
| Documentation corrections (the review of round 117) | `docs/source/backends/pim.rst`: the AttAcc pair is imported from `allo.pim.targets` / `allo.pim.costs` (an `ImportError` from `allo.pim` reproduced), stated with the import lines, no API change; the Gemini-II SDK root and card variables are documented with their supported boundary (`/opt/gsi` and `apu-00` only; other values refused with a named reason by `apu_g2_unsupported_override_reason`). `docs/release/INSTALL.md`: the sentence "None of the simulators or vendor SDKs is bundled" replaced by the distinction between the bundled Samsung sources and the external AiM, uPIMulator, UPMEM SDK and GSI SDKs. The Samsung skill document's example gains `import allo` (containing repository, in place, not committed there) |
| Board headroom | `/tmp` 3.4 MB free throughout: the Gemini-I campaign and the installed APU v1 execution stay blocked; nothing launched |
| V1-20 reproduced at `ab239e3` (`r118/v1_20_repro.py`, `v1_20_gemm.py`, `v1_20_scopes.py`) | the one-row-per-work-item GEMV region: `mapping=[4]` compiles (estimate 185,295, `grouped_f16`); `[8]` and `[16]` pass enumeration and fail in ranking inside `perf/cost.py::_scope_coordinates` with `work coordinate 4 is outside 'apuc' extent 4` (the plan builder emits one cost stream per work item at the APUC coordinate, spied: work ids 0..4 before the failure); `[2]` is refused by the shared work-grid policy. The dense-GEMM row-block region: `[4]` compiles (estimate 974,475); `[8]` is refused by the policy (`work grid does not partition target 'apu_v1'`), never reaching costing |
| What the realized program does | `grouped_device_packing` and `_apu_v1_prepare_grouped_io`: one SPMD body on the four APUCs, the output rows split `ceil(M / 4)` per core, `batches` 32K-lane batches per core, independent of the work-item count; the cost evaluator prices one stream per unit coordinate (`_check_work_grid`'s documented reason for refusing grids with no unit mapping). An experiment that remapped grouped work items onto the APUCs of their row blocks in the plan builder was tried and reverted: it made the costing pass for the dense-GEMM form only for the shared policy to refuse the same grid afterwards, so it would have added a second reading of the mapping rather than removing one |
| Resolution (`allo/spmw_autoschedule.py::_apu_v1_grouped_enumerate`) | a non-coalesced scope (a dataflow region's grouped program) with more work items than APUCs is refused before any candidate is priced, with the shared policy's wording, the explanation (one SPMD body on the four APUCs, rows split over the cores) and the remedy (`mapping=[4]`, each work item its `ceil(rows / 4)`-row block); the coalesced-axis lane-partition mapping of the scalar route is untouched. After the change: `[4]` compiles with the same estimate 185,295; `[8]` and `[16]` raise the policy message and never reach the coordinate error; the neighbouring dense-GEMM `[4]` region unchanged at 974,475 |
| Tests | `tests/pim/test_apu_v1_grouped_mapping.py` 3 passed; `tests/pim/test_shared_semantics.py -k grouped` 19 passed; `-k apu_v1` 28 passed; the APU v1 `tests/perf` selection (vector compile, tile selection, scalar surface, reduction policy, recognizers) recorded in the next row |
| Dependency | the first compiler-source change after the packaged working candidate `ab239e3`: `allo/spmw_autoschedule.py` only (the APU v1 grouped enumerator); `git diff --stat ab239e3 HEAD -- allo/pim/apu_g2_* allo/spmw_apu_g2.py benchmarks/apu_g2 allo/spmw_samsung.py allo/spmw_aim.py allo/spmw_upmem.py allo/spmw_attacc.py` empty; the revision owes installed-artifact validation of the APU v1 route (the functional reference and section-u tests from a wheel, the board stages when `/tmp` allows) before it can replace `ab239e3`, which stays the packaged working candidate; the inherited `b994fc0` Gemini-II campaign and its OPEN gemm / mvt outcomes are untouched |
| The APU v1 `tests/perf` selection after the change (`r118/pytest_apu_v1_perf.log`) | `test_apu_v1_vector_compile.py`, `test_apu_v1_tile_selection.py`, `test_apu_v1_scalar_surface.py`, `test_apu_v1_reduction_policy.py`, `test_apu_v1_recognizers.py`: **75 passed** (the scalar lane-partition mapping, which the refusal does not touch, exercised throughout) |
| Not done | the APU v1 installed-artifact vector, forced-plan and cost-perturbation execution and the three Gemini-I rows (`/tmp`); the installed validation of this revision's APU v1 route; the final verdict (NOT READY) |

## Round 119 (2026-11-10): the V1-20 revision validated from its installed artifact (the APU v1 route)

| Check | Result |
|---|---|
| The artifact (`r119/build_wheel.sh`, `build_wheel.log`, `pkg/`) | revision `7dc7a85` (`pkg/revision.txt`, clean tree), sdist `b09c4b7d88571660ceacb20307272bb62f8a42af9b7ea75d7bd0b59f5ce2d2a2`, wheel `9a745c1160dd1d1dd4ea896aedc167ec18663a2b1fccde4f3afdc05a11c0656e`, the bundled recipe (77 entries) and both license texts carried, installed into a clean venv |
| Installed stages outside the checkout, no `PYTHONPATH` into it (`r119/run_apu_v1_installed.sh`, `run_apu_v1_installed.log`, `r119/stages/`; `MILESTONE PASSED` 20:30:45 to 20:32:39) | `module_origin` PASS (the package from `pkg/venv`, nothing from the checkout on `sys.path`); `installed_enumerator_carries_the_refusal` PASS (the installed `spmw_autoschedule.py` contains the V1-20 refusal); `copy_test_tree` PASS (tests and benchmarks copied, the optional `pytest.ini` / `conftest.py` reported absent); **`apu_v1_grouped_mapping` 3 passed** (four items compile through the grouped route; eight and sixteen are refused with the policy message and never reach the coordinate error); **`apu_v1_shared_semantics_selection` (`-k "apu_v1 or grouped"`) 28 passed**; `apu_v1_functional_reference` 10 passed with DeprecationWarning as error in `pkg/venv`, **under NumPy 2.5.4, not 2.3.5** (corrected in round 120: `r119/build_wheel.sh` installs unpinned numpy and `pkg/venv` holds `numpy-2.5.4.dist-info` dated 20:29:55, before these tests; the log is preserved as it is); `numpy24_venv` PASS (NumPy 2.5.4, the wheel reinstalled without dependencies); `apu_v1_functional_reference_numpy24` 13 passed; `apu_v1_grouped_mapping_numpy24` 3 passed. The NumPy 2.3.5 run is round 120's |
| `run_installed_milestone.sh` | gains the two stages `apu_v1_grouped_mapping` and `apu_v1_shared_semantics_selection` before the functional-reference stages, which do not exercise the grouped enumerator |
| What this establishes and does not | the changed APU v1 grouped enumerator behaves from an installed artifact of its own revision as it does in the checkout, and the APU v1 functional reference passes under NumPy 2.5.4 (both venvs of this round; the "two NumPy lines" claim made here in round 119 was wrong and the 2.3.5 run is round 120's). It is not the full milestone at `7dc7a85` (the other backends' stages, the Gemini-II card tests and the AttAcc figures are inherited from `ab239e3`'s identical sources and not re-run), it is not final-candidate validation, and it is not hardware evidence: the board stages (the vector path with forced alternatives and cost perturbation, the Gemini-I campaign) stay pending behind this host's `/tmp` (about 4 MB). `ab239e3` stays the packaged working candidate |
| `STATUS.md` identity table | the open-ended `c21b510 and later` row bounded to `c21b510` to `4773d47` (rounds 111 to 117, records and scripts only); a `7dc7a85` row added with its artifact identities and this round's validation status |
| Not done | the APU v1 installed-artifact vector, forced-plan and cost-perturbation execution and the three Gemini-I rows (`/tmp`); the full milestone at a revision that would replace `ab239e3`; the final verdict (NOT READY) |

## Round 120 (2026-11-10): the NumPy labels corrected; the functional reference under NumPy 2.3.5 proper

| Check | Result |
|---|---|
| The review's finding | reproduced: `r119/pkg/venv` and `r119/venv-np24` both hold `numpy-2.5.4.dist-info`; `r119/build_wheel.sh` (like `r106` and `r111`) installs unpinned `numpy`, so every "release NumPy 2.3.5" installed stage since round 106 ran under 2.5.4 (`r106/pkg/venv` and `r111/pkg/venv` also hold 2.5.4). The label came from the checkout environment's NumPy (the `spmw` interpreter, 2.3.5), not from the venv. The round 106, 113 and 119 rows, `STATUS.md`, `REVIEW.md`, ISSUES V1-20 and the handoff are corrected; the logs are preserved as they are |
| The earlier V1-19 rows (rounds 103 to 105, `pytest_v1_19_np235.log`) | those "NumPy 2.3.5" runs were made from the checkout with the `spmw` interpreter, whose NumPy is 2.3.5; the logs do not print the interpreter, so this attribution rests on the rounds' working notes and is marked so in place rather than asserted as verified |
| NumPy 2.3.5 proper from the `7dc7a85` wheel (`r119/run_np235.sh`, `run_np235.log`, `stages-np235/`) | a venv pinned to `numpy==2.3.5` with the wheel `9a745c11...` installed without dependencies; **versions asserted, not assumed** (`numpy 2.3.5`, `allo` from `venv-np235/.../site-packages`, nothing from the checkout on `sys.path`); `apu_v1_functional_reference_numpy235` **13 passed** with DeprecationWarning as error (the functional reference, the nonzero-initial-output temporal plan, the local-accumulator GEMM on the functional backend); `apu_v1_grouped_mapping_numpy235` 3 passed; `MILESTONE PASSED` |
| The maintained milestone (`run_installed_milestone.sh`) | an `environment_versions` stage records Python, NumPy and the package origin from the interpreter and asserts `TENON_MILESTONE_NUMPY` when set; `numpy24_versions_asserted` asserts NumPy >= 2.4 and the `venv-np24` origin; a pinned `release_numpy_venv` (`numpy==${TENON_RELEASE_NUMPY:-2.3.5}`) with `release_numpy_versions_asserted` and `apu_v1_functional_reference_release_numpy` (the functional reference and the grouped-mapping regressions) so the two NumPy lines are two asserted environments, not a name and a guess |
| Scope kept | no wheel rebuilt, no compiler source changed (`git diff --stat 7dc7a85 HEAD -- allo` empty); `ab239e3` stays the packaged working candidate; the inherited `b994fc0` Gemini-II campaign and its OPEN gemm / mvt outcomes untouched; the board blocked (`/tmp` about 3.9 MB) |
| Not done | the APU v1 installed-artifact vector, forced-plan and cost-perturbation execution and the three Gemini-I rows (`/tmp`); the full milestone at a revision that would replace `ab239e3`; the final verdict (NOT READY) |

## Round 121 (2026-11-10): the round 106 attribution corrected; P-07 completed (distribution identity)

| Check | Result |
|---|---|
| `REVIEW.md` round 106 attribution (the review's remaining correction) | the verdict block now states that both environments of round 106's APU v1 functional-reference stage held NumPy 2.5.4, and that the asserted NumPy 2.3.5 evidence is round 120's from the `7dc7a85` wheel and validates nothing about `b994fc0` under 2.3.5 |
| Board headroom | `/tmp` 3.9 MB free throughout: the Gemini-I campaign and the installed APU v1 vector execution stay blocked; nothing launched |
| P-07 (release identity is upstream Allo) | `setup.py` metadata: summary, home page, maintainer and project URLs as recorded in ISSUES P-07; `python setup.py --name --version --description --url --maintainer` prints `allo`, `0.5`, the Tenon summary, the fork URL and the Tenon maintainer; an sdist built from the working tree (`r121/sdist/allo-0.5.tar.gz`, `sdist.log`) carries `Summary`, `Home-page`, `Maintainer` and the four `Project-URL` lines in `PKG-INFO`; `tests/perf/test_packaging.py` 17 passed including the new `test_distribution_metadata_identifies_the_tenon_fork`. The package name and version line stay upstream Allo's by recorded decision (the fork is installed in place of upstream Allo). **Closed** |
| C-13 (broad exception swallowing in the generic path) | the named modules carry no broad handler (grep); the residue outside the named scope is listed by file and line in ISSUES C-13 and left unchanged (a production change there would be a separate bounded item) |
| Scope kept | no compiler module changed (`git diff --stat 7dc7a85 HEAD -- allo` empty); `setup.py` metadata is the one packaging change after `ab239e3` and the next built artifact carries it; no wheel rebuilt for prose; `ab239e3` stays the packaged working candidate; the inherited `b994fc0` Gemini-II campaign and its OPEN gemm / mvt outcomes untouched |
| Not done | the APU v1 installed-artifact vector, forced-plan and cost-perturbation execution and the three Gemini-I rows (`/tmp`); the full milestone at a revision that would replace `ab239e3`; the final verdict (NOT READY) |

## Round 122 (2026-11-10): the consolidated candidate `7efd43a` validated from its installed artifacts on both hosts

| Check | Result |
|---|---|
| The candidate | code revision `7efd43a` (the V1-20 enumerator repair `7dc7a85` and the P-07 metadata; `git diff --stat ab239e3 7efd43a -- allo setup.py` is the grouped enumerator and `setup.py` metadata, nothing else); the milestone script revision `9c93843` adds the installed-metadata stage. Built from the committed tree (`r122/build_wheel.sh`, `pkg/revision.txt`): sdist `de3830a70a9ed1ba0d176ab2c756ba8458cd6109e975c5ff0a81e650978b878d`, wheel `bef38f1819e329135dd407bcf94a91c07257d0dd7fd281fd7ba839361165e551`; zhang-x4 wheel `bfb4ddeb8e8bcb85...` from the same sdist |
| Local milestone, every stage, outside the checkout (`run_installed_milestone.sh r122`; `r122/run_installed_milestone.log`, `r122/stages/`; `MILESTONE PASSED`, 37 stages, 20:55 to 22:41) | `module_origin` (the package from `pkg/venv`, nothing from the checkout); **`distribution_metadata`** (the installed dist-info's Summary names Tenon and the PIM targets, Home-page the fork, Maintainer the Tenon authors, Project-URLs the upstream and the release records); `environment_versions` (Python 3.12.13, **NumPy 2.5.4** in the clean venv, recorded from the interpreter); the bundled Samsung recipe verified, built from the wheel, discovered through `TENON_HOME`, the stripped driver the evaluation identity `9aea49bf...`; the five contract probes (`BUNDLED_ROUTE_OK 4108`, `FRESH_INSTALL_CHECK_OK`, `FRESH_SIM_CHECK_OK`, `AIM_CHECK_OK`, `UPMEM_CHECK_OK`; 0 `allo-c-*` directories left); `copy_test_tree`; golden rows 26 passed; `paper_full` golden rows 24 passed; Samsung conductor runs and GEMV on every target 14 passed; CENT 8 passed and CENT `paper_full` 13 passed; the shared-contract set 77 passed, 1 skipped; **`apu_v1_grouped_mapping` 3 passed and `apu_v1_shared_semantics_selection` 28 passed** (the V1-20 change from the wheel); the AttAcc figure stages: the six kernel rows and the composition PASS and `attacc_compare_frozen` **`ALL REPRODUCED`**; `apu_v1_functional_reference` 10 passed (NumPy 2.5.4, asserted); `release_numpy_venv` + `release_numpy_versions_asserted` (**NumPy 2.3.5**, the wheel from `venv-release-numpy`) and `apu_v1_functional_reference_release_numpy` 13 passed; `numpy24_venv` + `numpy24_versions_asserted` (NumPy 2.5.4) and `apu_v1_functional_reference_numpy24` 13 passed |
| Gemini-II chain on `zhang-x4` (`r122/g2_chain_no_session.sh`; `r122/g2-chain.log`, `r122/g2/stages/`; `MILESTONE PASSED`) | wheel `bfb4ddeb...` built there from the candidate's sdist into a clean venv; tests and benchmarks copied through the helper; **card tests 15 passed in 125 s**. **No validation session and no campaign** (the Gemini-II modules, templates and benchmark are byte-identical to `b994fc0`; the completed campaign's fixed outcomes are inherited and not re-sampled) |
| Fresh against inherited at `7efd43a` | fresh: every local installed stage above, the installed metadata, the two asserted NumPy environments, the twelve AttAcc figure results on Ramulator2-AttAcc, the Gemini-II card tests from a wheel of its own; inherited: the `b994fc0` Gemini-II confirmation campaign (revision, wheel `d5cb59f8...`, the four session files, `r107/g2_pooled.json`, gemm 1.0009 and mvt 1.0001 OPEN (nominal), the other three REPRODUCED, fixed) on byte-identical programs; **UNVERIFIED at every revision:** the APU v1 installed-artifact vector execution with forced alternatives and cost perturbation and the three Gemini-I rows (this host's `/tmp`, about 3.9 MB throughout the round; nothing launched, no other user's files touched) |
| Candidate status | `7efd43a` has every piece of installed evidence `ab239e3` had plus the V1-20 route, the metadata and the AttAcc figures from its own wheel; it is proposed as the packaged working candidate's replacement, the decision left to the reviewer as in round 114 (`STATUS.md`, identity table) |
| Not done | the APU v1 installed-artifact vector, forced-plan and cost-perturbation execution and the three Gemini-I rows (UNVERIFIED, `/tmp`); the final verdict (NOT READY) |

