<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Paper reproduction through the shared pipeline

## Required scope: key tables and figures only

User correction and confirmation, 2026-11-04: reproduce the key tables and
figures; supporting work is optional. This section supersedes earlier
exhaustive-campaign wording and historical next-action lists in this file.
The inventory's `release_scope` column determines whether a reproduction
result is required. Preserve existing measurements and honest statuses;
optional UNVERIFIED/DIAGNOSTIC rows are not G10 reproduction blockers.

| Required comparison | Frozen paper source | Required performance coverage |
| --- | --- | --- |
| Samsung GEMV | tables/eval-samsung.tex | Three GEMV shapes; cycles and speedups with outputs validated |
| AiM vs CENT | tables/eval-skhynix.tex | Fourteen component/composed rows; timing-only result kind and generated work checked |
| UPMEM kernel comparison | figures/eval_upmem, tables/eval-upmem.tex | Sixteen kernels; kernel latency and baseline comparison |
| Gemini-I vs Phoenix | tables/eval-apuv1.tex | Five kernels; device cycles and speedups; LoC columns optional |
| Gemini-II vs GTML | tables/eval-apuv2.tex | Five kernels; device cycles and speedups; LoC columns optional |
| AttAcc kernel tiling | figures/eval_attacc_kernels, tables/eval-attacc-kernels.tex | QKV, projection and scoring at batches 1/4 (six plotted summaries), with comparable simulator timings |
| AttAcc full-model decode | figures/eval_attacc_e2e, tables/eval-attacc-e2e*.tex | Six plotted prompt/generation/batch configurations; stated simulated/analytical composition and speedups |

The required rows use shared-pipeline compilation, preserved source semantics
and timing boundaries, matching-or-better comparable performance, and real
declared execution modes. Architecture diagrams and mechanism illustrations
remain implementation contracts under G1/G9; they do not create an obligation
to regenerate every isolated probe or historical measurement in a caption.

**Optional supporting work:** UPMEM scaling, exhaustive masked layouts,
shared-selection/packing/DCC studies, isolated layout ablations, AiM token
partitioning, AttAcc 500-instance timing-quality and prediction-error studies,
compilation time and line counts, calibration, and virtual architecture sweeps.
Keep useful existing evidence. Run more only to diagnose a required result
or an actual compiler/shared-pipeline defect, or under a later user request.
Optional reproduction incompleteness is distinct from a production defect.

**Next work:** checkpoint the current token-partition/cost increment without
extending the optional campaign, then address open required backend rows and
the AttAcc performance figures, followed by release-candidate validation.
Report required and optional progress separately; the total number of
unverified entries in the broad inventory is not a release criterion.

Status (2026-11-05, round 56, integration candidate `54cbb08`): **in progress, G10 FAIL**. Historical coverage of the required set (each row at the revision the inventory records): 47 of 55 REPRODUCED, 8 OPEN (the four AiM RoPE rows X-19, the three nominal Gemini-I rows X-22/X-23 whose pre-registered campaign is pending on the host's temporary space, the nominal Gemini-II batched_gemm row X-26), 0 UNVERIFIED. Validation at the candidate itself is separate and incomplete: executed at `54cbb08` from installed artifacts are the five Gemini-II rows (four paired sessions; 4 REPRODUCED, 1 OPEN), the fourteen AiM rows (default and `paper_full`; 10 exact, the 4 corrected RoPE programs OPEN) the three Samsung rows (vendor-equivalent programs at the paper's values, defaults faster) and the twelve AttAcc results (identical to rounds 51-52); the UPMEM and Gemini-I rows carry earlier revisions and remain unverified at the candidate until re-executed there (VALIDATION rounds 56-57). User priority
addition 2026-11-03, gate G10 of the containing repository's TASK.md. The
machine-readable inventory (`paper_reproduction/inventory.csv`, 83 rows) is
the authoritative per-case status; "Results so far" below summarizes it.
Measured through the shared pipeline's default selection so far: Samsung's
three GEMVs (faster than the paper, outputs checked), UPMEM's sixteen kernels
(exact), ten of fourteen AiM CENT rows (exact, timing-only) with the four
RoPE rows an OPEN 32-cycle regression against the paper (X-19; round 53: the difference is exactly the omitted half of the RoPE multiply, the paper's digit is reproduced by the inconsistent program's command stream; the enumerated candidates were searched by estimated cost and the multiply width is the device minimum, which does not establish that no faithful complete program is faster), the five
Gemini-II kernels (REPRODUCED under the archived paired protocol with the
GTML baseline re-run in the same rounds), the Gemini-I kernels under the
archived paired protocol with the expert binaries re-run (matrix multiply
and histogram REPRODUCED; kmeans and linear regression OPEN (nominal) on
the pooled sessions, their generated programs being the paper's, X-23;
word count an OPEN regression
attributed to the generated program's loop-counter width and repaired, the
repaired program at parity with the paper's own binary in the same rounds
and 0.19% above the paper's archived median, so OPEN (nominal) by the
parity rule, X-22), and the 500-instance AttAcc selector equivalence as a diagnostic.
DIAGNOSTIC: token partitioning (the normal selection under the eight-channel
budget on the bridged boundary reaches 9 of 9 simulated minima on the
contraction profile recalibrated in round 50 (rounds 42-49: 6 of 9, largest
gap 1.04%); the softmax bridge, the link volumes, the host stage, the frozen
study's own control and the measured host protocol are in the priced and
measured boundary; not comparable with the paper's row while the host
stage's price is a declared model and the recalibrated selections await
review, M-15, M-19; rounds 39-50).
REPRODUCED (round 51): the six AttAcc kernel-tiling results through the
shared pipeline on the natively built Ramulator2-AttAcc: QKV and projection
at batches 1 and 4 at the paper's values with the fixed baselines re-run,
the two score rows at the figure's speedups (1.00x and 1.13x over fixed)
with every selection equal to the paper's selection file and the text's
500-instance tally against DCC reproduced exactly (123 wins, 316 ties, 61
losses, median loss 7.1%).
REPRODUCED (round 52, analytical composition): the six AttAcc decode
configurations from the fresh shared-pipeline kernel times through DCC's
decode model: Tenon full 6.50 / 6.95 / 7.40 / 3.09 / 3.48 / 3.87x and base
1.47 / 1.56 / 1.66 / 1.34 / 1.50 / 1.65x over the GPU roofline (the frozen
figure and table ranges), DCC re-run 6.50 / 6.95 / 7.40 / 3.05 / 3.44 / 3.82x,
the fixed baseline 2.55 / 2.73 / 2.91 / 2.29 / 2.58 / 2.87x; batch 4 Tenon
1.27% faster than DCC, QKV 1.2799 to 1.2332 ms of 3.9933 to 3.9431 ms at
128 / 2048 (the frozen text).
UNVERIFIED (optional): the layout rows, UPMEM scaling, the packing, shared
selection and DCC studies, calibration, the virtual-backend study and the
compile-cost rows.
The paper is zzzDavid/tenon-hpca27 main at
30b6f49e64583b5d301d2a5e8b902498f5578e4e; its local mirror and hash lock are
paper/latex and paper/SUBMITTED_VERSION.json in the containing repository.
Do not modify that baseline. More features and richer internal mechanisms are
allowed; preserve the paper's semantic/mechanism contracts and qualifications.

## Work order and acceptance

Complete the current bounded backend integration increment, then inventory
and measure paper workloads on each usable shared backend while finishing G9.
Prioritize the required key comparisons listed above. Measure the program the
shared driver selects and materializes; use explicit alternatives separately
for controlled ablations. Fix implementation and performance gaps as found.
Do not wait for unrelated arithmetic or lint debt to be closed first.

For required comparable latency/cycles, current/paper must be <= 1; for throughput,
current/paper must be >= 1. Exact historical digits are unnecessary. Hardware
needs repeated measurements, the stated statistic and reported uncertainty;
a nominal regression stays open until parity is established or it is repaired.
Retain per-case comparisons and aggregate summaries. Do not improve results
by changing arithmetic, shapes, timing boundaries, baseline, or supported
paper capabilities. Missing comparable measurements stay UNVERIFIED.

## Initial coverage map

References below are relative to the frozen paper's latex source. This is
the broader historical coverage map; only the required set above blocks G10.
Retain optional families and their statuses without demanding their completion.

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
compiler inputs. G10 remains open until independent review establishes
coverage of the required key comparisons, preserved contracts and
matching-or-better performance. Optional supporting entries need not be
completed.

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
measured, or no comparable measurement is possible today), `DIAGNOSTIC`
(measured through the shared pipeline, but on a boundary or metric that is
not the paper's; the figures inform the mechanism and are not a
reproduction). A `paper_full`
row is reproduced by selecting it explicitly; nothing is hidden.

Rules applied: the measured program is the one the shared driver selects
and materializes (the Samsung row records the forced vendor layout
separately as the paper's own comparison point); archived winners,
vendor traces and golden checksums are comparison evidence only; a case
whose paper program is found inconsistent is reported as such, with the
corrected program's measurement, never by changing the paper.

Hardware rows (round 30 onward) follow the archived campaigns' protocols
with their baselines re-run in the same rounds: Gemini-I, one warm-up and
ten measured launches per lane and case, the Tenon default selection
interleaved launch by launch with the archived corrected expert Phoenix
binary (`benchmarks/apu_v1/paper_rows.py --expert`); Gemini-II, one warm-up
per lane and twenty interleaved paired rounds against the archived
public-GTML VectorPack host, repetitions 20, raw batch ticks divided by the
repetitions (`benchmarks/apu_g2/paper_rows.py`). Each row records the
session median with quartiles, IQR and MAD, the re-measured baseline, the
same-round baseline/Tenon ratios and the paper's ratio.

Acceptance for hardware rows (rewritten in round 31 after the reviewer
found the round 30 wording open to favourable-session selection): every
complete session under the protocol at a given revision is kept and
pooled; no session is discarded or chosen. The row's statistic is the
median of the pooled measured samples of all complete sessions at the
revision being judged (sessions at earlier revisions are reported beside
it, not pooled). `REPRODUCED` means that pooled median is at or below the
paper's value with every output exact. Otherwise the row is `OPEN
(nominal)` when the paper's value lies inside the pooled interquartile
range, or `OPEN (regression)` when it lies below it; both report the
pooled median, quartiles, the per-session medians, the re-measured
baseline, the paired baseline/Tenon ratio against the paper's, and the
same-round control against the archived paper-era binary where one
exists. Confirmation is bounded: a row that is `OPEN (nominal)` gets at
most one pre-registered confirmation campaign at the final candidate (the
number of sessions written down before the first launch, three unless
stated otherwise), whose samples are pooled with the earlier sessions at
that revision; after it the status is fixed by the pooled median and no
further sessions are run to move it. The Gemini-I campaign for kmeans, linear regression and word count is pre-registered in ISSUES X-22 / X-23 (round 53: three paired sessions each at the final candidate, the round 30 protocol, pooled median) and has not been launched because the candidate is not final. No tolerance closes a row; X-19 and
the pre-repair X-22 measurements are regressions, not nominal. The
Gemini-II vendor host has its device library location compiled
in; a byte copy of the archived `device.update.bin` (sha256 `c46f7f08...`)
was placed at that location on the shared filesystem for the re-run
(`ORIGIN.txt` beside it), and the driver hashes the file the host reports
having loaded and fails the launch unless it equals the archived library.

### Results so far (revision 0e48d81, evidence `tenon-release-evidence/r25/`)

| Family | Rows | Result |
|---|---|---|
| Samsung GEMV (3) | PIMSimulator, placement + execution + readback, reassociation permitted | default shared selection 4389 / 7947 / 8153 cycles against the paper's 4435 / 8041 / 8247 (ratios 0.990 / 0.988 / 0.989); the forced vendor layout reproduces the paper values exactly; outputs match the fp32 NumPy reference (`g10_samsung_gemv.log`) |
| UPMEM kernels (16) | uPIMulator, 1 DPU / 12 tasklets, device-only | all 16 rows exact through default selection (`test_upmem_paper_cycles`, default and `paper_full`), NumPy-exact outputs, recorded simulator/SDK build (`g10_upmem_default.log`, `g10_upmem_full.log`) |
| AiM CENT (14) | Ramulator2 AiM, timing-only, global memory-system cycles | 10 rows exact through default selection (`g10_aim_default.log`, `g10_aim_full.log`); the 4 rows containing RoPE (`qkvo_rope_l128`, `full_block_l128/l512/l4096`) were refused by the round 23 transfer frame check: the paper's program scattered the 2D-element RoPE buffers (as CENT does, 8192 host writes) while its EWMUL kernel iterated D elements, so the kernel multiplied a strided half of each partition's data. The benchmark kernel now covers the D2 vector the host scatters (CENT's one EWMUL per rotated vector) and the post-launch host rewrite of the RoPE result is modelled explicitly; the corrected programs keep CENT's exact opcode inventories and simulate at qkvo_rope_l128: 129637 (paper 129605, CENT 130597), full_block_l128: 421191 (paper 421159, CENT 425471), full_block_l512: 446835 (paper 446803, CENT 456575), full_block_l4096: 697348 (paper 697316, CENT 762668), each 32 cycles above the paper's number (two wider EWMULs) and faster than CENT. Recorded as corrected programs (X-19), not as reproductions of the paper's digits; the golden rows keep the paper values and pin the corrected cycles exactly (`AIM_CORRECTED_PROGRAMS`). Under G10's same-or-better rule these four rows are an OPEN regression of 32 cycles against the paper's reported values; a passing corrected-program golden row is provenance, not parity |
| AttAcc selection (500) | analytical selector through `allo.compile` | Diagnostic only: `benchmarks/attacc/scripts/verify_paper_selection.py --out r25/g10_attacc_selection_500.json` shows the shared-pipeline selection equals the paper's archived selection on 500 of 500 instances (668,575 candidates, 1912 s). The published quality result (398 of 500 instances at the best reference simulator timing) is a different metric and stays UNVERIFIED until the external Ramulator2-AttAcc timings of the selected and reference tilings are produced; the golden test that calls the selector directly establishes neither |
| Gemini-I kernels (5), round 29 | local board, installed wheel, 5 launches each, archived inputs and oracles | default shared selection medians: matrix_multiply 8,757,824 (paper 8,764,900; the selected plan is the paper's forced plan, chosen by cost), kmeans 624,488 (624,677), histogram 166,636 (166,783), linear_regression 94,328 (94,618.5), word_count 174,620 (171,988: **1.5% slower, X-22 open**); every output equals the oracle (`r29/apu_v1_default_rows.json`). Forced alternative and cost perturbation rows are labelled separately (`apu_v1_matmul_forced_acc4.json`, `apu_v1_matmul_perturbed.json`) |
| Gemini-II kernels (5), round 29 | zhang-x4 card, installed wheel, 5 sessions of 20 pipeline bodies, exact int64 outputs | medians: gemm 1,725 (paper 1,733.225), batched_gemm 1,736 (1,739.15), gesummv 1,806 (1,860.075), bicg 2,540 (2,548.825), mvt 2,733 (2,766.625); all at or below the paper's Tenon column and below GTML (`r29/g2_paper_rows_sessions.json`); the GTML baseline itself is the paper's measurement, not re-run |
| Gemini-I kernels (5), round 30 paired protocol | local board, installed wheels at 69773fc and 28cc922, one warm-up and ten measured launches per lane, the expert binaries interleaved launch by launch, exact outputs in every launch | Tenon medians (sessions 1 / 3) vs paper: matrix_multiply 8,758,698 / 8,757,878 vs 8,764,900 (REPRODUCED); kmeans 625,280 / 625,200 / 624,530 (control session) vs 624,677: pooled over the three complete round 30 sessions (30 samples) median 624,950, q1 624,429, q3 625,430 (1.0004), the archived paper-era binary 625,038 in the same rounds as the control, the paired expert ratio 1.0269 at the paper's: OPEN (nominal), identical generated source, X-23; histogram 165,997 / 166,141 vs 166,783 (REPRODUCED); linear_regression 95,526 / 95,807.5 / 94,933 vs 94,618.5: pooled median 95,180 (q1 94,703, q3 95,967; 1.0059), the paper's own binary 95,047.5 in the same rounds as the control: OPEN (nominal), source differs in a static assertion only, X-23; word_count 177,178 / 176,415 vs 171,988 (OPEN regression X-22, attributed to the loop-counter width: in one session the current program 176,041.5, the same source with 16-bit counters 172,178, the paper-era source 171,984, the archived paper-era binary 172,648; repaired in dc41df7; the repair check, the only complete session at dc41df7, measured the repaired default selection at 172,312 (q1 172,090, q3 172,882) in the same rounds as the archived paper-era binary at 172,372 and the expert at 175,635, paired expert/Tenon 1.0201 vs the paper's 1.0198: OPEN (nominal), the paper's value 0.06% below the pooled q1). Expert medians re-run 8,786,101 / 641,165 / 167,804 / 95,185 / 175,948.5 (session 1); paired expert/Tenon 1.0030 / 1.0261 / 1.0107 / 0.9961 / 0.9922 against the paper's 1.0022 / 1.0269 / 1.0036 / 1.0123 / 1.0198 (`r30/apu_v1_paired_rows.json`, `apu_v1_paired_rows_s3.json`, `apu_v1_word_count_x22.json`) |
| Gemini-II kernels (5), round 30 paired protocol | zhang-x4 card, installed wheel at 69773fc, 20 interleaved paired rounds against the archived GTML host, exact outputs in all 210 launches | Tenon medians (q1-q3): gemm 1,722.05 (1,717.0-1,733.4) vs paper 1,733.225; batched_gemm 1,731.55 (1,719.9-1,744.4) vs 1,739.15; gesummv 1,839.30 (1,829.1-1,844.5) vs 1,860.075; bicg 2,530.05 (2,525.9-2,547.5) vs 2,548.825; mvt 2,755.03 (2,750.5-2,770.6) vs 2,766.625: all REPRODUCED. GTML re-run in the same rounds 1,914.4 / 1,916.1 / 3,099.5 / 2,734.8 / 3,090.6 (within 0.6% of the paper's column); paired GTML/Tenon medians 1.109 / 1.105 / 1.687 / 1.079 / 1.122 (paper 1.098 / 1.096 / 1.662 / 1.070 / 1.117) (`r30/g2_paired_rows_run2.json`). The first run of the session reported the vendor lane's library-load failure and exited 1 (`r30/g2_paired_rows.json`) |
| AiM token partitioning (9 workloads), rounds 39-40 | AiM ramulator2 built natively from the 0f28a07 tree (Docker out of disk on this host; the native binary reproduces every exact golden row), `benchmarks/cent_aim/token_partition.py`: the paper's eight-channel budget as the compiler's resource policy (`resources={"channel": 8}`), the normal shared selection under it measured as the headline plan, the forced per-P plans and the head-only control as ablations, every plan's provenance and artifacts kept, host merge priced at 32 GB/s, merge numerics checked on every split plan | DIAGNOSTIC, not comparable (M-15): the normal budgeted selection reaches 6 of 9 simulated minima with a largest gap of 1.34% (paper 7 of 9, 2.63%), the misses being the 1-head workloads (0.34%, 0.49%, 1.34%); the measured boundary lacks the scores-to-softmax-to-probabilities bridge, the host stages and the readout/upload link volumes the frozen evaluation includes, so these figures are comparable with each other only (`r40/token_partition_nine.json`) |
| AiM token partitioning (9 workloads + neighbour), round 42, bridged boundary | the bridged attention (QK scores -> host global softmax -> probabilities -> SV) with the softmax a declared host stage of the shared plan, priced in the selection graph (readout + upload over the 32 B/cycle link, four host vector passes; partial readout in the host reduction), the eight-channel budget as the compiler's policy, every plan with independent functional evidence (lowered QK and SV oracles around the compiled stage's function vs a float64 reference), measured host-stage samples beside the price, the CENT attention program as the 32-head control, every plan's artifacts retained | DIAGNOSTIC (M-15 narrowed): the normal selection reaches 6 of 9 simulated minima with a largest gap of 1.04% (paper 7 of 9, 2.63%), the three misses on the 1-head workloads (0.33%, 0.44%, 1.04%); head-only / selected 1.91x for 1 head x 4096 tokens (paper 1.071x); neighbour 2x256 at 0.46%; functional pipeline within 4.9e-3 on every plan; not yet comparable because the host stage's price is a declared model (this host's measured samples are 18-530 us per call) and the CENT control's boundary is CENT's own (`r41/token_partition_bridged.json`) |
| AiM token partitioning, round 43, repaired bridge | the stage is the scaled softmax of the workload's head dimension bound at realization to the recorded computation, declared operation and realized position; the host price stored as link and compute terms and the measured-host total replacing the compute term only; the functional reference is scaled attention on operands that expose the scale | DIAGNOSTIC (M-15, M-17 fixed): the same cells and figures as round 42 (6 of 9, 1.04%; the simulator's cycles are data independent); every plan's functional pipeline within 9.6e-3 of the scaled float64 reference, the unscaled reference off by at least 5.3x; the one-head misses are two estimate ties and one 12% relative overprice of the 8-way command stream (design doc section 3a) (`r43/token_partition_bridged.json`) |
| AiM token partitioning, round 49, CENT-derived control and host protocol | the eight-channel CENT control derived structurally (CENT's mapping policy from the shared pipeline's exact selection on the CENT golden cases, matched in each workload's budgeted enumeration and compiled under the study's budget, bridge and accounting), the historical host protocol measured step by step beside the price, every variant and control accounted for, the one-head misses calibrated command by command against the simulator | DIAGNOSTIC (M-15, M-19): same device cells, 6 of 9 minima, largest gap 1.04%; the CENT-derived control equals the selection on the 8- and 32-head workloads and is 1.03x-2.23x slower on the 1-head ones; the misses are the contraction profile's register I/O, WR_GB and MAC charges (recalibration pending), not the bridge or the accounting (`r49/token_partition_bridged.json`, `r49/calibration_*.json`) |
| AiM token partitioning, round 50, historical control, host protocol, recalibrated profile | the frozen study's `cent-repaired` control restored through the shared pipeline (padded heads, reuse 1, eight channels; producer provenance and command inventories recorded), the round 49 control kept as the `cent-policy` ablation; the host protocol measured on the attention's own BF16 scores, encoded once, with the historical methodology; the contraction profile recalibrated per command class from controlled single-factor simulator measurements and validated on the standalone probe and held-out programs before the nine workloads | DIAGNOSTIC (M-15; M-19 repaired, awaiting review): same device cells; the normal selection reaches 9 of 9 simulated minima, every gap 0, estimates within 0.1-0.3% of the measured totals (rounds 42-49: 6 of 9, misses 0.33% / 0.44% / 1.04%; paper 7 of 9, 2.63%, on its own boundary); cent-repaired 2.22x / 3.85x / 7.27x over the selection on the 1-head workloads and 1.45x-1.46x on the 8- and 32-head ones; the measured whole protocol 32-623 us per call on this host (`r50/token_partition_nine.json`, `r50/controlled_variations.json`, `r50/heldout_validation.json`) |
| AttAcc kernel tiling (6 required results), round 51 | Ramulator2-AttAcc built natively from the MIT sources (`build_ramulator_attacc.sh`; Docker out of disk), `run_ramulator.py` rewritten to keep the compiled artifact and record the selection's provenance, check the DCC generator's trace against the selected program (totals, ordering, address mapping; `trace_check.py`) and retain trace, configuration, digests and raw output per run; every shape compiled through `allo.compile`; the fixed-form baseline re-run for the four GEMV rows (DCC's rule: the minimum over the fixed-form candidates) | REPRODUCED: QKV batch 1 / 4 and projection batch 1 / 4 selected programs 0.00881 / 0.03364 / 0.00265 / 0.01024 ms (the paper's values; fixed baselines 0.02807 / 0.17239 / 0.00265 / 0.01030, the paper's); score batch 1 / 4 over 248 shapes each: mean 0.00193 / 0.00678 ms, speedup over the fixed tiling 1.00x / 1.13x (mean of per-shape ratios, the figure's 1.00x / 1.13x; DCC's archived per-shape results give 1.01x / 1.12x, the figure's); every selection equals `paper_selection.csv`; the fresh Tenon times against DCC's archived per-shape times give 123 wins, 316 ties, 61 losses with a median loss of 7.07% on the 500 instances (text: 123 / 316 / 61, 7.1%); 506 retained traces re-checked, ordering and mapping as declared (`r51/attacc_*.json`, `r51/attacc_kernel_summary.json`, `r51/attacc_trace_recheck.json`) |
| AttAcc full-model decode (6 required configurations), round 52 | `compose_e2e.py` rewritten: shared-pipeline kernel records only (conflicts rejected, references ignored), every GPT-13B PIM row measured or NaN (no baseline substitution), DCC's decode model (`inference/system.py`, A100 roofline, prefill excluded) run for the GPU roofline and for the Tenon, DCC-selection and fixed variants (the fixed variant with DCC's own placement: attention on PIM, QKV and projection on the GPU), per-stage times recorded; `tests/pim/test_attacc_compose.py` | REPRODUCED (analytical composition): Tenon full 6.50 / 6.95 / 7.40 / 3.09 / 3.48 / 3.87x, base 1.47 / 1.56 / 1.66 / 1.34 / 1.50 / 1.65x (the inventory's paper values and the table ranges 3.09-7.40 / 1.34-1.66); DCC re-run full 6.50 / 6.95 / 7.40 / 3.05 / 3.44 / 3.82x, base 1.47 / 1.56 / 1.66 / 1.34 / 1.49 / 1.64x; fixed full 2.55 / 2.73 / 2.91 / 2.29 / 2.58 / 2.87x, base 1.09 / 1.16 / 1.23 / 1.16 / 1.30 / 1.43x (the table's 2.29-2.91 / 1.09-1.43); batch 1 equal to two decimals with Tenon 0.0001 ms slower at 128 / 2048 (the text's one small loss), batch 4 Tenon 1.27% faster (text 1.2-1.3%), QKV 1.2799 to 1.2332 ms of 3.9933 to 3.9431 ms (`r52/attacc_e2e.json`) |
| layout rows (Samsung XOR, Gemini-I vector GEMM), scaling, packing, DCC, calibration, virtual study, compile cost | | UNVERIFIED this round; next in the order of the inventory |
