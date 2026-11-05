<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# AiM token partitioning and cost-route repairs

Status: implemented on `tenon` (working tree after `6efaca5`). Addresses
release audit `docs/release/audit/aim-attacc.md` M-03, M-05 (cost side; see
Section 6 for the one plumbing edit outside this change), M-06, M-07, M-08,
M-11 and M-12, and the paper subsection "Token Partitioning"
(`paper/latex/sections/5-evaluation.tex`, `sec:eval-token-partition`).

Salvaged from the snapshot worktree (read-only):
`.worktrees/allo-snapshot/allo/pim/aim_sv_partition.py`
(`AiMSVPartitionSpec`, `compile_aim_sv_partition`, `merge_aim_sv_partials`),
`aim_qk_row_packed.py`, and the harness
`reviews/experiments/aim_attention_campaign.py`. Nothing was copied as a
separate compiler: the semantics became a knob, a LinearLayout dimension, a
lowering case and a host-step record of the existing AiM route.

## 1. Mechanism

| Piece | Where | Meaning |
|---|---|---|
| Knob `reduction_partitions` (P) | `allo/spmw_knobs.py::_aim_reduction_partitions_candidates`, `_aim_reduction_partitions_emit` | Registered for `aim` between `reuse_group` and `replica_partitions`. Domain: P = 1, plus powers of two P <= channels / replicas that divide the reduction, only on the `channels` batch mapping with a GB-fed vector. |
| Layout carrier | `allo/pim/aim_lowering.py::reduction_partitioned_layout`, `layout_reduction_partitions` | P > 1 extends the base AiM LinearLayout with output dim `token_partition`: reduction bit i < log2(P) maps to partition bit i, so element t belongs to partition `t % P` (cyclic split). The layout is part of the matcher decision key, so P is a distinct decision without touching the shared physical-feature list. Codegen (`spmw_aim._reduction_partitions`) reads P from the layout and rejects a placement whose `extra` disagrees. |
| Lowering IR | `AimContraction.reduction_partitions`, `_AimLowerer._contraction_reduction_partitioned` | The pair (batch b, partition p) is the channel-batched virtual batch `b * P + p` over `reduction / P` local elements with `ceil(storage / P)` reserved local storage; each accumulates its own partial output vector. The manifest entry carries a `reduction_partition` block (rule, local extents, partial count). |
| Host reduction | `AimHostReduction` (aim_lowering), emitted by `_AimProgramDerivation._launch_records` right after the contraction | The host reads the P partial GPR entries per merged 256-bit vector, adds them, and writes the merged vector back to the GPR. Lowers to no ISR line (AiM has no host instruction); recorded in `lowering_manifest` (`kind="hostreduction"`, `executes_on="host"`, `command_count=0`) and in `AimCtx.host_steps`; `_run_aim` lists it in `RunResult.extra["host_steps"]` and never adds it to simulated `cycles`. `AimHostReduction.merge` is the executable computation (FP32 for floats, int64 for integers). |
| Cache append | `_AimProgramDerivation._matrix_records` | `broadcast(v_cache[:, :, t])` into a split layout writes only the channels of partition `t % P` at local row `(t // P) // row_elements`. The cyclic split keeps every token's owner fixed as the context grows. |
| Cost | `allo/pim/costs/aim.py::host_reduction_cycles`, rule `contraction` | A second, dependent `host_reduction` step on the `host` unit after the device contraction. |
| Oracle | `aim_lowering.reference_partial_sums` | Functional check of the transformation, separate from the timing-only simulator: walks the lowered manifest (channel groups, virtual batches, row chunks), asserts every (batch, element) is owned exactly once and every accumulator has its own channel, and computes the partials (`arithmetic="exact"` for integers; `"bf16"`: BF16 operands, FP32 chunk sums, BF16 readout per row chunk). |

QK: the attention score's token axis is its output axis, so PIMphony-style
token placement for QK is the existing output distribution over channels
(`row_packed` with `replica_partitions`); no reduction is split and no merge
is needed. The knob also applies to QK's `channels` mapping (it splits the
head-dimension reduction) and to any GB-fed contraction on that mapping
(split-K GEMV); the cost decides.

### Reduction-order policy

Splitting reassociates the reduction. The knob reads each accumulating
match's policy through `allo.spmw_reduction_policy.reduction_policy` (the
stamped `extra["reduction_policy"]`; an unstamped match reads as the strict
default): when `may_regroup` is false only P = 1 remains, so a strict BF16
reduction is never split while an integer accumulator or
`reduction_order="reassociate"` permits it. Note that under
`reduction_order="strict"` the reduction-policy route rejects AiM BF16
programs before scheduling (every MAC regroups 16 lanes), so in practice the
split is reachable with `reduction_order="reassociate"` or integer
accumulators.

## 2. Cost model and provenance

Units: ramulator2 `memory_system_cycles` of `aim_simulator` `0f28a07`.
`GDDR6.cpp:602` recomputes `tCK_ps = 1e6 / (rate / 2)` for every preset, so
rate 2000 gives 1 ns per cycle (the preset's listed 500 ps is overwritten).

`host_reduction_cycles(record, geometry)` =
`HOST_LINK_FIXED_CYCLES + ceil(V * (P + 1) * 32 B / HOST_LINK_BYTES_PER_CYCLE) + V * (P - 1) * HOST_VECTOR_ADD_CYCLES`,
V = `ceil(outputs / lanes) * batches * replicas` merged vectors.

| Constant | Value | Provenance / class |
|---|---|---|
| `HOST_LINK_BYTES_PER_CYCLE` | 32 | Paper: 32 GB/s link; harness `host_link_bytes_per_ns=32`; 1 ns cycle. Declared scenario. |
| `HOST_LINK_FIXED_CYCLES` | 0 | Paper: zero fixed latency; harness `host_link_fixed_latency_ns=0`. Declared scenario. |
| `HOST_VECTOR_ADD_CYCLES` | 1 | Analytical: one 256-bit host SIMD add per merged vector. Uncalibrated. |
| GPR entry | `vector_bits / 8` = 32 B | Target (`gpr` width 256). One `RD_MAC` readout per output tile per accumulator. |

The harness measured host CPU stages (QK decode, global softmax, staging,
merge) with `perf_counter_ns`; those are not compiler quantities and the
softmax is common to every P. The compiler prices only the merge and its
transfers.

## 3. The nine-workload experiment (a diagnostic; see M-15)

Driver: `benchmarks/cent_aim/token_partition.py` (`python -m
benchmarks.cent_aim.token_partition --out tp.json`; the test
`tests/pim/test_aim_token_partition.py::test_nine_workload_selection`,
`-m paper_full`, calls it).  Workloads: 1/8/32 heads x 128/512/4096 tokens,
head dim 128, K/V caches reserved for 4096 tokens, inputs resident, QK and
SV in one program (`mapping=[1]`).  Budget: the paper's eight simulated
channels, expressed since round 40 as the compiler's resource policy
(`resources={"channel": 8}`, `allo.spmw_resources`): the replica and token
partition domains are bounded by it and the shared ranking selects the
complete program (QK and SV) within it; that **normal selection** is the
headline plan and its P the "selected" partition count.  The forced plans
are the ablation: the budgeted ranking's cheapest SV plan per P in
{1, 2, 4, 8} and the head-only control (one head per channel, no split),
compiled as explicit layouts under the same budget.  Measured total =
simulated device cycles + modelled host reduction (32 GB/s link).  Every
split plan passes the exact-integer merge check and the BF16 check (max
relative error 3.7e-3, tolerance 2^-6); an unsplit plan has no merge and
its record says so.  Every plan's artifacts are written before the next
plan runs (declared effects, lowering manifest, graph, segment traces,
simulator output, executor/library/config digests, compiler revision),
and a failed plan is recorded without discarding the others.

**What this is not:** the frozen paper's boundary includes the
scores-to-global-softmax-to-probabilities bridge between the kernels, the
host stages' time and the QK readout / score upload / partial readout link
volumes; this driver has `probs` resident and prices only the merge, so its
minima, gaps and head-only ratios are comparable with each other, not with
the paper's row (inventory status DIAGNOSTIC; M-15 holds the plan).

Run 2026-11-02, isolated snapshot HEAD `6efaca5` + this change, simulator
`0f28a07`, image `aim-simulator-build`; re-run 2026-11-04 (rounds 39-40,
native ramulator2 from the same tree, the budget as the compiler's policy)
with the same cells, the normal budgeted selection equal to the forced
plan of its P on every workload (`tenon-release-evidence/r40/
token_partition_nine.json`). Cells: measured total (estimate).

| Heads | Tokens | P1 | P2 | P4 | P8 = fixed 8-way | Head-only | Selected P | Best P | Gap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 1213 (1618) | 1173 (1586) | 1169 (1586) | 1215 (1634) | 1213 | 2 | 4 | 0.34% |
| 1 | 512 | 2002 (2518) | 1746 (2294) | 1634 (2198) | 1626 (2198) | 2002 | 4 | 8 | 0.49% |
| 1 | 4096 | 5535 (6911) | 7973 (10217) | 5974 (7910) | 5462 (7462) | 12003 | 1 | 8 | 1.34% |
| 8 | 128 | 1472 (2206) | 2607 (3777) | 4621 (6663) | 8649 (12435) | 1472 | 1 | 1 | 0 |
| 8 | 512 | 3251 (5071) | 4386 (6642) | 6400 (9528) | 10428 (15300) | 3251 | 1 | 1 | 0 |
| 8 | 4096 | 22492 (35756) | 22748 (36012) | 23004 (36268) | 27032 (42040) | 22492 | 1 | 1 | 0 |
| 32 | 128 | 5888 (9016) | 10428 (15300) | 18484 (26844) | 34596 (49932) | 5888 | 1 | 1 | 0 |
| 32 | 512 | 13004 (20476) | 17544 (26760) | 25600 (38304) | 41712 (61392) | 13004 | 1 | 1 | 0 |
| 32 | 4096 | 89968 (143216) | 90992 (144240) | 92016 (145264) | 108128 (168352) | 89968 | 1 | 1 | 0 |

The selection reaches 6 of 9 simulated minima (paper: 7 of 9, maximum gap
2.63%); maximum gap 1.34%. All three misses are single-head workloads where
the candidates are within 10% and the serial-issue estimate (1.25x to 1.59x
over simulation) misorders them. The P1 column for one head and 4096 tokens
is a `flattened` mapping (head dimension over the eight channels), which the
harness did not offer; against the head-only control the fixed eight-way
split is 2.20x faster for that workload. Numbers differ from the paper's
because the compiler's own program (masked WR_GB/RD_MAC per channel group,
no separate per-channel readout) and cost model replace the harness's
explicit-binding traces and measured host stages.

## 3a. The host bridge (round 42, M-15)

The study's program is the bridged attention: QK scores -> global softmax
on the host -> probabilities -> SV.  The softmax is a declared host stage
of the host program (`allo.host_compute("softmax", fn, inputs=(scores,),
outputs=(probs,))`, `allo/spmw_target.py`), ordered by the launch schedule
as a `HostComputeStep` carrying the stage's buffers with their shapes and
element widths (`allo/spmw_plan.py`).  The availability walk requires the
stage's inputs to be produced by an earlier launch (QK's declared `out`
transfer of the scores to the GPR) and makes its outputs available to a
later launch through that launch's declared host ingress (SV's
`host_gpr_preload` of its GB vector).  The AiM program lowers the stage as
`AimHostStage` (no commands; `after_command`; manifest kind `hoststage`
with readout and upload bytes and GPR entries; in the run's `host_steps`),
and `AimCtx._check_host_stages` binds it to the schedule and to the
producing and consuming launches' declared transfers through the launch's
positional binding and the retained identities.  Other backends refuse a
host stage.

**Semantics.**  The stage is the scaled global softmax of the workload's
head dimension, `softmax(scores / sqrt(d))` in FP32 with BF16 at the
upload (`ScaledSoftmax(head_dim)`), the historical host operation's
semantics; its `operation` is `softmax` from the declared host-operation
domain (`allo.spmw_target.HOST_OPERATIONS`), the only thing the pricing
and the realization key on (the stage's name is a label).

**Pricing, in the graph the normal selection ranks.**  The program graph
emits the stage as the target's `HOST_COMPUTE` operation; the AiM rule
prices it with `host_stage_terms`: a link term, the readout of the inputs
plus the upload of the outputs over the declared link (32 B/cycle, zero
fixed latency), and a compute term, the host's vector passes
(`HOST_OPERATION_PASSES["softmax"] = 4`: row max, scaled exp, row sum,
divide, one `HOST_VECTOR_ADD_CYCLES` each per 16-lane vector).  Units are simulator cycles (1 cycle = 1 ns at the
declared rate).  The partial-result readout and the merge adds of a split
SV remain in the host reduction, so nothing is counted twice.  For 1 head
x 128 tokens the stage prices at 48 cycles (16 link + 32 host).  The host
time is a declared model; the driver records measured host-stage samples
beside it (`host_stage_measured`, `stage.run` only) and a second total in
which the measured median replaces the compute term only, the stage's link
term and the host reduction retained (`total_cycles_measured_host_stage`).

**Realization.**  `AimCtx._check_host_stages` compares the realized
sequence of launch occurrences and stages in the lowered records with the
schedule (repeated launches counted by their primary records), requires
the lowered stage's callable to be the one the host program recorded and
its operation and buffers the declared ones, requires a launch-written
input to be declared an output by an earlier launch (a region input no
launch writes is host data; an earlier stage's output is accepted) and an
output to reach a later launch's declared host ingress; a mismatch is a
materialization failure before selection.

**Functional evidence, separate from the simulator.**  `reference_outputs`
(`allo/pim/aim_lowering.py`) is the functional oracle of a lowered
contraction: an unsplit entry after its coverage fields are asserted, a
split entry through `reference_partial_sums` and the host merge.  The
driver's `functional_check` runs random BF16 operands through the lowered
QK oracle, the compiled stage's own function and the lowered SV oracle
and compares with a float64 reference of softmax(q.K).V (tolerance 2^-6),
for every plan, split and unsplit, and for the neighbouring 2 heads x 256
tokens workload.  The timing-only simulator produces no values and none
are attributed to it.

**Controls (round 49).**  The head-only plan (forced); the CENT-derived
control: CENT's mapping policy, read from the shared pipeline's exact
selection on the CENT golden attention cases (QK row-packed over the
channel group with the 16-slot reuse window; SV channel-batched, one head
per channel, unsplit, same window), matched by its decision knobs in each
workload's budgeted enumeration (QK over the budget's eight channels, SV
over the heads up to eight) and compiled as an explicit layout under the
study's own budget, bridge and accounting (`cent_derived_layouts`); and
the unbudgeted default selection (recorded only).  On every 8- and 32-head
workload the CENT-derived control is the normal selection's own plan
(ratio 1.00); on the 1-head workloads it is 1.03x (128 tokens), 1.32x
(512) and 2.23x (4096) slower than the selection, and 1.05x on the 2x256
neighbour.

**Host protocol (round 49).**  Beside the priced host terms, the historical
bridge protocol is measured step by step on each compiled program: the
decode of the QK readout words (BF16 -> FP32), the compiled scaled softmax,
the staging of the probabilities for the upload (FP32 -> BF16 words) and,
for a split SV, the compiled merge (`host_protocol_measured`); their median
sum replaces every modelled host compute term in
`total_cycles_measured_host_protocol`, the stage's and the reduction's link
terms retained.  On this host the protocol takes 31 us (1x128) to 899 us
(32x4096) per call, NumPy on small arrays, far above the declared price;
the paper's host is not this one, which is one reason the row stays
DIAGNOSTIC.

**The one-head misses, calibrated command by command (round 49,
`benchmarks/cent_aim/command_calibration.py`).**  The k-th command's
increment on the simulator (cycles of the first k commands minus those of
the first k-1) is aggregated by class and set beside the profile's charges
for the same class, independently of any selection outcome:

| Class | Simulated increment per command | Profile charge |
|---|---:|---:|
| WR_BIAS | 2 | 35 (N_MODCH + CMD_WR_GB) |
| RD_MAC | 4 to 7 | 34 (N_MODCH + CMD_RD_REG) |
| WR_GB | 51 (one 128-element vector) to 131 (a flattened 4096-token write) | 35, constant |
| first MAC_ABK of a reuse group | 101 to 245 | 56 (first-row activation) |
| consecutive MAC_ABK, row-packed QK at 4096 tokens | 3,534 over 31 commands | 480 columns + 2,728 row switches |

At 1 head x 4096 tokens the SV spans are 1,656 simulated against 1,372
modelled for the flattened unsplit plan and 1,455 against 1,795 for the
eight-way plan: the register I/O charges inflate the channel-batched split
programs and the constant WR_GB and activation charges deflate the
flattened one, inverting their order.  At 128 and 512 tokens the SV spans
of the tied plans differ by 36 and 72 simulated cycles (951 against 915;
1,023 against 951) while the model's difference (32 and 64) is cancelled
exactly by the host reduction's 32 cycles per doubling, hence the ties.
Neither is a bridge or accounting effect.  The repair is a recalibration of
the register I/O, WR_GB and MAC charges from command classes measured this
way across the CENT golden cases and the study's kernels, never from the
study's winners (M-19).

**Bridged run (2026-11-04, round 42, `tenon-release-evidence/r41/token_partition_bridged.json`).**
Cells: measured total = device + priced host (stage + reduction), with the
estimate in parentheses; the normal budgeted selection equals the forced
plan of its P on every workload.

| Heads | Tokens | P1 | P2 | P4 | P8 | Head-only | Selected P | Best P | Gap | CENT control |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 1261 (1666) | 1221 (1634) | 1217 (1634) | 1263 (1682) | 1261 | 2 | 4 | 0.33% | n/a |
| 1 | 512 | 2194 (2710) | 1938 (2486) | 1826 (2390) | 1818 (2390) | 2194 | 4 | 8 | 0.44% | n/a |
| 1 | 4096 | 7071 (8447) | 9509 (11753) | 7510 (9446) | 6998 (8998) | 13539 | 1 | 8 | 1.04% | n/a |
| 8 | 128 | 1856 (2590) | 2991 (4161) | 5005 (7047) | 9033 (12819) | 1856 | 1 | 1 | 0 | n/a |
| 8 | 512 | 4787 (6607) | 5922 (8178) | 7936 (11064) | 11964 (16836) | 4787 | 1 | 1 | 0 | n/a |
| 8 | 4096 | 34780 (48044) | 35036 (48300) | 35292 (48556) | 39320 (54328) | 34780 | 1 | 1 | 0 | n/a |
| 32 | 128 | 7424 (10552) | 11964 (16836) | 20020 (28380) | 36132 (51468) | 7424 | 1 | 1 | 0 | 11010 |
| 32 | 512 | 19148 (26620) | 23688 (32904) | 31744 (44448) | 47856 (67536) | 19148 | 1 | 1 | 0 | 18126 |
| 32 | 4096 | 139120 (192368) | 140144 (193392) | 141168 (194416) | 157280 (217504) | 139120 | 1 | 1 | 0 | 95090 |
| 2 (neighbour) | 256 | 1813 (2345) | 1733 (2281) | 1725 (2281) | 2732 (3724) | 1813 | 2 | 4 | 0.46% | n/a |

Six of nine minima, largest gap 1.04%; every plan's functional pipeline
within 4.9e-3 of the float64 reference; the measured host-stage samples on
this host (18 us to 530 us per call, NumPy on small arrays) are far above
the declared price (48 to 49,152 cycles) and are recorded separately, not
substituted.

**The one-head misses, from the retained graphs (round 43,
`r41/token_partition_bridged.artifacts/h1_*/P*/plan.json`).**  The
bridge's stage term is identical for every P of a workload and the host
reduction term is small (32 cycles per doubling of P), so the order is
decided by the contraction profile (`mac_abk`, the serial-issue estimate
of the lowered command stream), which the simulator does not confirm at
the sub-percent level:

| Workload | Plans | Estimate (device term) | Measured device | What decided it |
|---|---|---:|---:|---|
| 1 x 128 | P2 vs P4 | 1554 vs 1522 (totals 1634 = 1634) | 1141 vs 1105 | an exact tie of the totals, broken to the smaller P; measured P4 better by 0.33% |
| 1 x 512 | P4 vs P8 | 2134 vs 2070 (totals 2390 = 2390) | 1570 vs 1498 | an exact tie of the totals, broken to the smaller P; measured P8 better by 0.44% |
| 1 x 4096 | P1 vs P8 | 6911 vs 7334 | 5535 vs 5334 | the profile overprices the 8-way program (1.37x of the simulator) more than the flattened unsplit one (1.25x); measured P8 better by 1.04% |

Two of the three misses are estimate ties resolved by the ordinary
smaller-P preference, the third a 12% relative overprice of the 8-way
command stream in the profile.  Neither is a bridge or accounting effect.
A tie rule that prefers the larger split when the host reduction is the
only difference would be a selection heuristic, not evidence; the profile's
systematic overprice of the channel-batched split programs (1.37x against
1.25x here, 1.25x to 1.59x across the study) is the cost-model item to
calibrate from the lowered command streams, not from these winners.

## 4. Cost-route repairs (M-06, M-05, M-07)

* One route (M-06): every contraction placement, enumerated or passed as
  `layout=`, is priced by lowering the exact `AimContraction` codegen emits
  (`_contraction_record`, absent knobs resolve as in codegen) and profiling
  its commands. `_aim_layout_candidates` stamps `aim_shape` (and the SBK bank
  list) on the base layouts. The knob-free formulas (`_columns`,
  `_mac_launches`) and `LANES_PER_PU`/`BANKS_PER_CHANNEL` are gone; geometry
  comes from `_target_geometry`. A contraction placement without a shape is
  rejected with an actionable error.
* Calibration: the profile now tracks open rows per (channel, bank): a MAC
  line opening a row in banks with no open row pays `nRCDRDMAC` (56), one
  replacing an open row pays `nRP + nRCDRDMAC` (88); SBK expansions are
  profiled line by line. GEMV 256x256 mapping [32]: ABK 223 (before: 255
  knobs / 156 `layout=`), simulated 222; SBK 1528 (before 720 / 765),
  simulated 1527. The 14 CENT traces are byte-identical to their pins.
* Host transfers (M-05): the rule prices the moved slice (`host_slice`) and
  zero for a resident buffer (`host_resident`); the docstring states the
  validity domain. See Section 6 for the plumbing that supplies the metrics.
* Ties (M-07): `replica_partitions` values wider than the channels a kernel
  can occupy are no longer enumerated (they either repeat the same command
  stream with an identical mask or only add zero-padded channels, so they tie
  with the narrowest covering group at every cost). The remaining tied minima
  on the GEMV probe (reuse groups above the single launch, `flattened` vs
  `row_packed` for one batch) lower to byte-identical traces
  (`tests/pim/test_aim_cost_routes.py::test_tied_minima_emit_one_program`).
  Surfacing ties in the ranking is the generic `rank_matcher_placements`
  change owned elsewhere.

## 5. Runner and dead code (M-08, M-11, M-12)

* `_run_aim` raises on a nonzero ramulator2 exit status; no cycles are
  reported from a failed run.
* `AimCtx.cmd`, `_ISR_FIELDS`, `_operand_text`, `_handle_to_fields`,
  `resolve_moves`, `_unit_env` and `_human_lines` are removed; the AiM
  target's ops and channel/bank moves declare no `emit` (their lambdas used
  the opposite channel-mask encoding). The only ISR encoder is
  `aim_lowering`. Host moves keep their direction-reporting `emit`.
* CENT mapper slot conventions live on the target's `banks` memory
  (`ewmul_slots`, `bank_pair_stride`, `bank_pair_slots`) and are read by
  `spmw_aim._bank_slot_conventions`.
* Target declarations made legal for the new placement-legality pass:
  MAC/MAC_ABK admit `banks` as the y source (CENT bank-pair MAC), AF reads
  `mac_reg` (MAC -> AF -> RD_AF).

## 6. Open items

* M-05 plumbing: the host-event metrics come from
  `spmw_plan.build_execution_graph._emit_host`, outside this change's files.
  The proposed edit (`_host_transfer_metrics`: add `host_slice` only for a
  sliced role and `host_resident` only for a resident buffer, leaving other
  host moves byte-identical) measured `attention_l128` virtual 4,203,452 ->
  10,108 cycles (simulated 11,010). Until it lands, the shared tree still
  prices whole buffers.
* `benchmarks/cent_aim/workloads.py` still duplicates target facts
  (`TARGET_CHANNELS`, `BANKS_PER_CHANNEL`, ...); its region annotations are
  postponed, so the token-partition workloads are defined in the test module.
* The skill doc `.claude/skills/pim-aim/SKILL.md` states 8 banks x 4 bank
  groups; the target and simulator preset declare 4 x 4 = 16 banks.
* The `channels` mapping issues one masked WR_GB for distinct per-channel
  payloads (timing abstraction inherited from CENT's traces); the oracle
  checks ownership and merge, not GB payload routing.
* Estimate accuracy on long streams (1.25x to 1.59x high) is the next
  calibration target; it was not tuned against these results.

## 7. Owner-map row placement (movement ties, 2026-11-02)

Claim row 16 on AiM. The movement stage (`allo/spmw_layout_problem.py`,
`layout-search.md` section 8) states a contraction's layout as a GF(2) map
from work bits (owner level `channel`; `bank_group` and `bank` are not
modelled for a one-dimensional grid) to iteration bits (`work0`, the row
loop, the reduction loop).

### 7.1 Realizable family

`row_map_unrealizable_reason` accepts a tie on AiM when the kernel is one
`MAC` with one accumulator of one output-row coordinate, the owner bits come
from the row bits only, the one-dimensional work grid partitions the output
rows into consecutive per-work-item blocks in work-id order (the CENT
replicated groups, `replicas = 4`, are outside), no input reads the row
loops without the whole row index (batched contractions are outside), and
every unit gets the same number of rows. A tie with an owner bit from `k`
is a reduction split and is rejected, with one exception: when every owner
bit comes from the low `log2 P` bits of the one reduction loop it is exactly
the `reduction_partitions` split of section 1 (element `t` on channel group
`t % P`, every row in every group, host merge;
`reduction_partition_split`). That tie is realizable, the stage adds the
low-bit member of a minimum k-split class as its own tie (description
`reduction split t % P over channel (reduction_partitions=P)`), and codegen
accepts it only on placements that lower the same `P`
(`spmw_aim._contraction_row_map` raises InfeasibleSchedule otherwise; those
candidates are scored and then rejected at materialization).

### 7.2 Realization

`_contraction_row_map` (in `_AimProgramDerivation._derive_groups`) evaluates
`owner_rows` for the carried and the program map. The work grid partitions
the output frontier in work-id order, so work item `w` owns positions
`[w R, (w + 1) R)`; under a tie, owner unit `u`'s rows occupy those positions
in ascending order: `slot[row] = u R + rank`. The map applies to one
unreplicated single-batch contraction with an unsplit reduction (other
carried maps are rejected with that reason).

* Trace: MAC_ABK/MAC_SBK, WR_GB, WR_BIAS and RD_MAC address channels, banks
  and positions, not rows, so the trace of a GB-fed contraction is
  identical for every realizable tie. Per-row host transfers
  (`_matrix_records`, `row_packed` scatter of `W[o, :]`) are addressed at
  `slot[o]`.
* Host-side result mapping: the lowering-manifest entry gains `row_map`
  (`map`, `rows_moved`, `rows_per_unit`, FNV-1a 64 `digest`, the full
  `row_slots` table and sample rows' `slot`, launch or tile, channel and
  bank under the mapping's storage contract, `_frontier_position`);
  `RunResult.extra["row_map"]` lists it. The identity adds nothing, so the
  pinned CENT traces and manifests are unchanged.

### 7.3 Ties and simulator runs

GEMV 2048 x 64 BF16 over `mapping=[32]` (`reassociate`; 64 rows per work
item; `M > 32 K / 2`, so splitting `k` never pays): minimum 3,968 B =
program, proven; ties: the program (block) map and `rows cyclic over
channel` (unit `u` holds rows `u, u + 32, ...`); `tie_count ==
realizable_tie_count == 2`, `chosen_label == "minimal"`. Expected before
running: equal cycles, because both ties emit the same 14-line trace.

| Tie | Trace | ramulator2 cycles | Estimate |
|---|---|---|---|
| program `channel<-work0[0..4]` | 14 lines | 507 | 691 |
| cyclic `channel<-loop0[0..4]` | identical | 507 | 691 |

Row 63, for example, sits at slot 1985 (launch 3, channel 28, bank 1) under
the cyclic map. Test: `tests/pim/test_aim_owner_maps.py`.

The all-targets GEMV (256 x 256, `mapping=[32]`) is unchanged: its two
minimum classes (5,120 B against the program's 15,872) mix row and high `k`
bits and are rejected as reduction splits that are not the
`reduction_partitions` split. A 64 x 2048 GEMV over 32 channels has the
P = 32 split as its minimum (3,968 B against 126,976): it is realizable and
selected (`chosen_label == "minimal"`). The ranking crosses the program's
own map only when no tie is realizable (`spmw_autoschedule._movement_entries`),
so the unsplit program is no longer a candidate. Without the movement stage
the same kernel selects the unsplit `flattened` program. Estimates: 8,883
(P = 32) against 670 (unsplit). ramulator2: 507 device cycles plus the
modelled 256-cycle host merge = 763 for P = 32, against 828 for the unsplit
program. The minimum-movement choice is faster in simulation although the
cost program ranks it far slower (the serial-issue estimate overstates the
32-group channel-batched program, section 3).
