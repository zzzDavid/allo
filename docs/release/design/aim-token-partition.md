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

## 3. The nine-workload experiment

`tests/pim/test_aim_token_partition.py` (`test_nine_workload_selection`,
`-m paper_full`; table: `python -m tests.pim.test_aim_token_partition`).
Workloads: 1/8/32 heads x 128/512/4096 tokens, head dim 128, K/V caches
reserved for 4096 tokens, inputs resident, QK and SV in one program
(`mapping=[1]`). Budget: eight channels (the paper's eight simulated
channels) = candidates with `replica_partitions <= 8` on the 32-channel
simulator preset (the harness also emitted 8-channel masks on that preset).
For each P in {1, 2, 4, 8} the cost program's best candidate with that P is
compiled through `layout=` and simulated; P1 is the best unsplit candidate
of any mapping. Controls: head-only = best P = 1 candidate on the `channels`
mapping (one head per channel); fixed eight-way = the P = 8 variant.
Measured total = simulated device cycles + modelled host reduction. Every
split variant passes the exact-integer merge check and the BF16 check
(max relative error 3.7e-3, tolerance 2^-6).

Run 2026-11-02, isolated snapshot HEAD `6efaca5` + this change, simulator
`0f28a07`, image `aim-simulator-build`. Cells: measured total (estimate).

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
