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
logs: `/tmp/tenon-release/r9/*.log` on this server (copied to
`/work/shared/users/phd/nz264/tenon-release-evidence/r8/` when the round
closes).

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

A full `tests/pim` and `tests/perf` run on the committed tree and the
installed-artifact runs are recorded below when they complete.
