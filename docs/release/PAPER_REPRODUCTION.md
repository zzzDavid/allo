<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Paper reproduction through the shared pipeline

Status (2026-11-04, round 28): **in progress, G10 FAIL**. User priority
addition 2026-11-03, gate G10 of the containing repository's TASK.md. The
machine-readable inventory (`paper_reproduction/inventory.csv`, 83 rows) is
the authoritative per-case status; "Results so far" below summarizes it.
Measured through the shared pipeline's default selection so far: Samsung's
three GEMVs (faster than the paper, outputs checked), UPMEM's sixteen kernels
(exact), ten of fourteen AiM CENT rows (exact, timing-only) with the four
RoPE rows an OPEN 32-cycle regression against the paper (X-19), the five
Gemini-II kernels (at or below the paper over five card sessions), four of
the five Gemini-I kernels (word count 1.5% slower, X-22 open), and the
500-instance AttAcc selector equivalence as a diagnostic. UNVERIFIED: the
layout rows, token partitioning, UPMEM scaling, the packing, shared
selection and DCC studies, AttAcc Ramulator2 timings and decode composition,
calibration, the virtual-backend study and the compile-cost rows.
The paper is zzzDavid/tenon-hpca27 main at
30b6f49e64583b5d301d2a5e8b902498f5578e4e; its local mirror and hash lock are
paper/latex and paper/SUBMITTED_VERSION.json in the containing repository.
Do not modify that baseline. More features and richer internal mechanisms are
allowed; preserve the paper's semantic/mechanism contracts and qualifications.

## Work order and acceptance

Complete the current bounded backend integration increment, then inventory
and measure paper workloads on each usable shared backend while finishing G9.
Prioritize kernel performance and shared selection. Measure the program the
shared driver selects and materializes; use explicit alternatives separately
for controlled ablations. Fix implementation and performance gaps as found.
Do not wait for unrelated arithmetic or lint debt to be closed first.

For comparable latency/cycles, current/paper must be <= 1; for throughput,
current/paper must be >= 1. Exact historical digits are unnecessary. Hardware
needs repeated measurements, the stated statistic and reported uncertainty;
a nominal regression stays open until parity is established or it is repaired.
Retain per-case comparisons and aggregate summaries. Do not improve results
by changing arithmetic, shapes, timing boundaries, baseline, or supported
paper capabilities. Missing comparable measurements stay UNVERIFIED.

## Initial coverage map

References below are relative to the frozen paper's latex source. Expand
these families into a machine-readable case/result inventory, including every
reported size and substantive table/figure claim, with reproducible commands.

| Family | Frozen source | Required comparison and execution scope |
| --- | --- | --- |
| Samsung GEMV | tables/eval-samsung.tex; sections/5-evaluation.tex | All three shapes; FP16, stated reassociation policy; placement + execution + readback cycles against stock vendor simulator. Separately preserve source-order portability semantics. |
| AiM CENT workloads | tables/eval-skhynix.tex | All 14 rows and lengths, component and composed stages; same timing model and global memory-system cycle boundary; compare freshly generated work with CENT. Timing-only accepted. |
| UPMEM kernels and scaling | tables/eval-upmem.tex; sections/5-evaluation.tex and eval_upmem figure | All 16 kernels, fastest supported baselines, one DPU / 12 tasklets / modeled 350 MHz; uneven GEMV and reduction scaling over 1-32 DPUs with the stated device-only boundary. |
| Gemini-I kernels | tables/eval-apuv1.tex | All five Phoenix comparison rows on local physical hardware; same shapes, arithmetic and median single-core device-cycle scope. |
| Gemini-II kernels | tables/eval-apuv2.tex; sections/5-evaluation.tex | All five GTML comparisons on zhang-x4; exact bit widths and shapes; median ARC running-state cycles amortized over batches of 20, data transfers excluded as in the paper. |
| Portability and masked layouts | sections/5-evaluation.tex; tables/eval-layout-mechanisms.tex | Shared 4096x256 rounding-sensitive FP16 GEMV and masked-layout correctness; retain source-order and explicit reassociation distinctions. |
| Shared selection, packing and DCC comparison | sections/5-evaluation.tex; figures/samsung-gemv-layout.tex; figures/apu-reduction.tex; tables/eval-layout-mechanisms.tex | Four Gemini-I tiles / five DRAM partitions, fixed-work packing controls, physical query-packing cases, residency-changing contraction separately, six GEMVs / 68 mappings; preserve HBM3 timing and excluded preload/reduction costs. Measure default selection separately from forced alternatives. |
| AiM token partitioning | sections/5-evaluation.tex | Nine workloads, 1/2/4/8 partitions and controls, 45 numerical variants; scoring + global softmax + value sum + host stages/reduction under the stated link/residency assumptions. |
| AttAcc selection and decode | tables/eval-attacc-*.tex; sections/5-evaluation.tex; eval_attacc figures | All 500 selection instances, six decode configurations, selection and command-replay prediction studies; Ramulator2-AttAcc timings and analytical GPU composition retain their distinct labels and assumptions. |
| Compilation and description cost | tables/eval-attacc-compile.tex; kernel tables; sections/5-evaluation.tex | Same machine/candidate set for schedule-selection timing; report relevant compile-time comparison, training exclusion and description scope. |
| Calibration | sections/5-evaluation.tex | Component measurements and held-out record-frequency/histogram shapes/plans; preserve no-refitting/held-out boundaries and distinguish estimate error from measured execution time. |
| Virtual architecture variants | sections/synthesis.tex; sections/5-evaluation.tex | AiM row-payload and global-buffer sweeps with fixed workload/timing; generated-command cost estimates plus conditional replay evidence, preserving hardware assumptions. |

## Evidence and issue records to produce

For each case retain paper location and value, source/API, shape/dtype/policy,
platform and baseline versions/configuration, timing boundary, command, current
revision and installed artifact, selected decision and stage provenance,
generated artifact identity, actual outputs or valid timing-only checks, raw
samples, current metric, ratio/delta and status. Link failures to an issue,
root cause, production repair and independent remeasurement. Preserve failed
attempts. Record available baselines and historical records without executing
old independent compiler routes as the claimed current implementation.

Start from tests/pim/test_paper_golden.py, release/CLAIMS.md and EVIDENCE.md,
benchmarks/attacc/paper_selection.csv and scripts/verify_paper_selection.py,
benchmarks/cent_aim, the UPMEM campaign scripts, both APU benchmark histories
and containing-repository experiment reports. Audit their coverage and routes:
a golden pass or archived winner is not proof of current shared selection.
Include expensive paper_full rows explicitly. Export reusable reproduction
commands/drivers in the release repo; measurements are evidence, not runtime
compiler inputs. G10 remains open until independent review establishes the
full coverage, preserved contracts and matching-or-better performance.

## Machine-readable inventory (round 25, 2026-11-04)

`docs/release/paper_reproduction/inventory.csv` holds one row per
substantive result of the frozen evaluation (82 rows: the five backend
tables, the layout-mechanism and portability rows, the shared-selection,
packing, DCC, token-partitioning, AttAcc tiling/selection/decode/compile,
calibration and virtual-backend studies), with the paper location,
workload source, shape, dtype and arithmetic policy, platform and baseline,
metric, timing boundary, paper value, reproduction command, current value,
current revision and status. Statuses are `REPRODUCED` (measured through
`allo.compile` default shared selection at the named revision, with the
ratio current/paper and the numerical check performed), `OPEN` (measured
and different, or a mechanism defect found), `UNVERIFIED` (not yet
measured, or no comparable measurement is possible today). A `paper_full`
row is reproduced by selecting it explicitly; nothing is hidden.

Rules applied: the measured program is the one the shared driver selects
and materializes (the Samsung row records the forced vendor layout
separately as the paper's own comparison point); archived winners,
vendor traces and golden checksums are comparison evidence only; a case
whose paper program is found inconsistent is reported as such, with the
corrected program's measurement, never by changing the paper.

### Results so far (revision 0e48d81, evidence `tenon-release-evidence/r25/`)

| Family | Rows | Result |
|---|---|---|
| Samsung GEMV (3) | PIMSimulator, placement + execution + readback, reassociation permitted | default shared selection 4389 / 7947 / 8153 cycles against the paper's 4435 / 8041 / 8247 (ratios 0.990 / 0.988 / 0.989); the forced vendor layout reproduces the paper values exactly; outputs match the fp32 NumPy reference (`g10_samsung_gemv.log`) |
| UPMEM kernels (16) | uPIMulator, 1 DPU / 12 tasklets, device-only | all 16 rows exact through default selection (`test_upmem_paper_cycles`, default and `paper_full`), NumPy-exact outputs, recorded simulator/SDK build (`g10_upmem_default.log`, `g10_upmem_full.log`) |
| AiM CENT (14) | Ramulator2 AiM, timing-only, global memory-system cycles | 10 rows exact through default selection (`g10_aim_default.log`, `g10_aim_full.log`); the 4 rows containing RoPE (`qkvo_rope_l128`, `full_block_l128/l512/l4096`) were refused by the round 23 transfer frame check: the paper's program scattered the 2D-element RoPE buffers (as CENT does, 8192 host writes) while its EWMUL kernel iterated D elements, so the kernel multiplied a strided half of each partition's data. The benchmark kernel now covers the D2 vector the host scatters (CENT's one EWMUL per rotated vector) and the post-launch host rewrite of the RoPE result is modelled explicitly; the corrected programs keep CENT's exact opcode inventories and simulate at qkvo_rope_l128: 129637 (paper 129605, CENT 130597), full_block_l128: 421191 (paper 421159, CENT 425471), full_block_l512: 446835 (paper 446803, CENT 456575), full_block_l4096: 697348 (paper 697316, CENT 762668), each 32 cycles above the paper's number (two wider EWMULs) and faster than CENT. Recorded as corrected programs (X-19), not as reproductions of the paper's digits; the golden rows keep the paper values and pin the corrected cycles exactly (`AIM_CORRECTED_PROGRAMS`). Under G10's same-or-better rule these four rows are an OPEN regression of 32 cycles against the paper's reported values; a passing corrected-program golden row is provenance, not parity |
| AttAcc selection (500) | analytical selector through `allo.compile` | Diagnostic only: `benchmarks/attacc/scripts/verify_paper_selection.py --out r25/g10_attacc_selection_500.json` shows the shared-pipeline selection equals the paper's archived selection on 500 of 500 instances (668,575 candidates, 1912 s). The published quality result (398 of 500 instances at the best reference simulator timing) is a different metric and stays UNVERIFIED until the external Ramulator2-AttAcc timings of the selected and reference tilings are produced; the golden test that calls the selector directly establishes neither |
| Gemini-I kernels (5), round 29 | local board, installed wheel, 5 launches each, archived inputs and oracles | default shared selection medians: matrix_multiply 8,757,824 (paper 8,764,900; the selected plan is the paper's forced plan, chosen by cost), kmeans 624,488 (624,677), histogram 166,636 (166,783), linear_regression 94,328 (94,618.5), word_count 174,620 (171,988: **1.5% slower, X-22 open**); every output equals the oracle (`r29/apu_v1_default_rows.json`). Forced alternative and cost perturbation rows are labelled separately (`apu_v1_matmul_forced_acc4.json`, `apu_v1_matmul_perturbed.json`) |
| Gemini-II kernels (5), round 29 | zhang-x4 card, installed wheel, 5 sessions of 20 pipeline bodies, exact int64 outputs | medians: gemm 1,725 (paper 1,733.225), batched_gemm 1,736 (1,739.15), gesummv 1,806 (1,860.075), bicg 2,540 (2,548.825), mvt 2,733 (2,766.625); all at or below the paper's Tenon column and below GTML (`r29/g2_paper_rows_sessions.json`); the GTML baseline itself is the paper's measurement, not re-run |
| layout rows (Samsung XOR, Gemini-I vector GEMM), token partitioning, scaling, packing, DCC, calibration, virtual study, compile cost | | UNVERIFIED this round; next in the order of the inventory |
