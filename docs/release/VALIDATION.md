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
