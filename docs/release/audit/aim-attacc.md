# Release audit: SK hynix AiM backend and the AttAcc evaluation path

> User clarification: AiM's simulator is timing-only, and this is acceptable
> for release. Its lack of numerical outputs is not a blocker and does not
> require implementing a functional simulator. Findings below concerning
> output/capability reporting should be interpreted as requirements for honest
> timing-only labeling, unchanged output arrays, correct emitted work/trace,
> valid simulator execution and timing provenance. Separate functional checks
> may validate compiler transformations, but must not be represented as
> numerical execution by the timing simulator. See the clarification in
> the containing repository's TASK.md.

Scope: `experiments/allo` branch `tenon` @ `8160d6d` (working tree had only
untracked `scripts/verifier_check_*.py`). Read-only audit; no tracked file was
edited. Probe scripts: `/tmp/tenon-release/probes/aim/p1..p15*.py`
(`common.sh` sets the environment). Every claim is labelled **executed**
(probe or test run in this audit), **read** (source inspection), or
**suspected** (inferred, not demonstrated).

Environment used: `PY=/work/zhang-capra/users/nz264/miniforge3/envs/spmw/bin/python`,
docker image `aim-simulator-build` (215d37b48f08), simulator checkout
`experiments/simulators/aim_simulator` @ `0f28a07` (`build/ramulator2` present).

---

## 1. Routes and entry points

### 1.1 Public AiM route (executed, P1/P4/P7)

```
allo.compile(workload, build_aim_target(), aim_cost[, host_moves=HostProgram][, backend="virtual"][, layout=...])
  -> allo/compiler.py:635  compile()            customize() -> match_workload() -> _stamp_matcher_work_scopes()
  -> allo/spmw_codegen.py:1351 compile_for_target()
       -> allo/spmw_autoschedule.py:1142 autoschedule() -> rank_matcher_placements() (1261)
            enumerator  = _aim_enumerate (855): SBK/ABK LinearLayout bases (_aim_layout_candidates 676)
                          x knobs bank_scope/batch_mapping/reuse_group/replica_partitions (spmw_knobs.py:329-412, 503-511)
            cost        = allo/pim/costs/aim.py::aim_cost (per-candidate execution graph, spmw_plan.build_execution_graph)
            feasibility = _materialize_matcher_codegen probe; InfeasibleSchedule -> next candidate
       -> AimCtx.emit_program (allo/spmw_aim.py:285): _AimProgramDerivation -> AimOp records -> _AimLowerer (allo/pim/aim_lowering.py:592)
          bank_fanout==1 expands each MAC_ABK into per-bank MAC_SBK lines from the LinearLayout (spmw_aim.py:341-347, _sbk_banks 262-281)
  -> CompiledCallable (compiler.py:447).  __call__(*arrays) / run_backend() -> Compiled.run -> _run_aim (spmw_aim.py:1097)
       -> writes the exact `compiled.cmds` text trace to <AIM_ROOT>/test/tmp*.trace
       -> docker run aim-simulator-build ./build/ramulator2 -f test/example.yaml -t <trace>
       -> regex `memory_system_cycles: N`, max over channels -> RunResult(cycles=N, stdout, backend="aim", extra={returncode, segments, unique_segments})
```

Observed (executed, P1, GEMV 256x256 bf16, mapping [32]): compile 7.4 s, 216
candidates, chosen `all_bank+bank_scope=abk+batch_mapping=flattened+reuse_group=1+replica_partitions=16`
(cost 255), trace of 5 lines (`WR_GB 16 0 4294901760`, `WR_BIAS`, `MAC_ABK 16 4294901760 0`,
`RD_MAC`, `EOC`), ramulator2 run 0.7 s, `RunResult(cycles=222, backend='aim',
extra={'returncode': 0, 'segments': 1, 'unique_segments': 1})`.

**Numerics:** none. The output array passed to the callable is untouched after
the run (`y unchanged after run? True`, `outputs` key absent from `extra`).
`AimCtx` emits text ISR lines carrying no payload; ramulator2 produces only
timing statistics (simulator README: "no functional data path"). Nothing in the
release computes or checks AiM numerics; `tests/pim/test_gemv_all_targets.py::test_gemv_aim`
asserts only `cycles > 0`. The paper's "numerical validation of attention
partitioning is evaluated separately" refers to code outside the release
(section 5, M-03). The public `RunResult` carries **no timing-only marker**
(M-08).

### 1.2 Virtual route (executed, P8)

`backend="virtual"` evaluates the retained execution graph with `aim_cost`;
`run_backend()` returns `RunResult(cycles=<estimate>, backend='virtual',
extra={critical_path, utilization, bottlenecks, model_fingerprint, cost_model, priced_target})`.
Distinguishable from the simulator run by `backend` (good), but see M-05 for
what the estimate contains.

### 1.3 CENT reproduction route (executed, P4 + pytest)

`benchmarks/cent_aim/workloads.py::build_case(case_id)` -> `CentCase(region, host_program)`;
`allo.compile(case.region, build_aim_target(), aim_cost, host_moves=case.host_program)`.
`attention_l128`: compile 8.8 s, 2249 lines, SHA matches the pinned typed-route
trace, 22 kernels ranked, simulated **11,010 cycles = golden** (0.7 s sim).
`pytest tests/pim/test_paper_golden.py -k aim`: 3 default rows pass
(norm_residual 10.3 s, softmax_pim 9.8 s, attention 8.2 s), 11 `paper_full`
rows skipped by marker. `tests/pim/test_gemv_all_targets.py -k aim`: 2 pass.
`tests/pim/test_aim_matcher.py` + `test_aim_virtual_study.py`: 5 pass in 36.6 s
(virtual study rows ~10-11 s each, compile-only).

### 1.4 What is not a compiler route

- `benchmarks/` is a plain package inside the checkout (`find_packages()` will
  ship it); it holds the CENT workload builder, pinned trace SHAs and vendor
  opcode inventories (data, test-side).
- Token partitioning and AttAcc are **not** reachable from `allo.compile` (section 5).

---

## 2. Enforced vs advertised domain

| Aspect | Advertised (paper / docs / skill) | Enforced in release (executed unless noted) |
|---|---|---|
| Operations | MAC (GEMV/attention), EWMUL, EWADD, AF (sigmoid-as-tanh) | Matcher accepts `acc += W*x`; `acc += W - x` rejected: `ValueError retained matcher kernel has no matched implementation` (P3). MUL/ADD/AF via target ops (read). |
| dtype | BF16 (16 lanes x 16 bit) | **Not enforced**: `float32` and `int32` GEMVs compile to the identical 27-line trace and identical cost as bf16 (P3). |
| Reduction K | any | K=100, 1040, 4096 compile (row splits derived: `reduction_rows` 1/2/4; P3). |
| Outputs M | any | M=7 ok (1 launch, 2.29x padded issue); but see M-01 for M/mapping combinations that explode work. |
| Work grid | mapping prefix of [32,4,4] | mapping [3], [24] accepted with a `UserWarning` and **wrong shapes** (M-01); [64] rejected (`work coordinate 32 is outside 'channel' extent 32`). |
| Channels | 32 (simulator hard-coded) | Target with 24 channels compiles and emits mask `16776960` (0xFFFF00) for a 32-channel-only simulator; no check (P5/P8). |
| Banks | 16 (4 bg x 4) | Non-power-of-two banks rejected (`AiM LinearLayout requires a power-of-two bank count`). Cost module hard-codes 16 (M-06). |
| GB / row | 2 KiB / 1024 elements; virtual sweep 2/4/8 KiB | `GB cannot hold one complete bank row` rejected (good). Sweep variants compile (test_aim_virtual_study). |
| Reuse window | 32 slots | `slots=0` -> `IndexError: list index out of range` (not actionable). |
| Initial accumulator | zero (WR_BIAS) | `acc = y[i]` (nonzero init) compiles to the same trace as zero-init; bias GPR fixed at 0 (semantics not represented; timing-only). |
| Shape names in annotations | from the workload | **Taken from the caller's stack frames** (M-02). |

---

## 3. Findings

Severity: blocker / major / minor. Gates refer to TASK.md G1-G8.

### M-01 (blocker) Output axis mis-derived as batch; issued work does not match the program
- **Where:** `allo/spmw_match_engine.py:1604-1653` (`batch_dim`: a loop is "batch" iff its var indexes some input and is absent from another input — true for the *output* loop of every GEMV written as `W[f(i), j] * x[j]`), consumed by `allo/pim/aim_lowering.py:1746-1821` (`contraction_shape`, non-replicated branch: `batches = extra["batch_dim"]` **and** `outputs` from the matrix memref extent, skipping only reduction/batch vars) and the fallback `outputs = outer * channels` at `aim_lowering.py:1805-1809` (uses the *target* channel count, not the workload's work grid).
- **Executed (P7):** three semantically identical GEMVs, 256x256, mapping [32]:
  - A `row0 = ch*ROWS; W[row0+i, j]`: `batch_dim=1`, record `{outputs 256, batches 1}`, 1 MAC_ABK, issued MACs 65,536 = logical, cost 255, **sim 222**.
  - B `W[ch*ROWS+i, j]` inline: `batch_dim=8, batch_loop_var=%arg3`, record `{outputs 256, batches 8}`, **8 MAC_ABK**, issued 524,288 (8x), cost 1110, **sim 858**.
  - C mapping [1], `W[i, j]`: correct (replicated branch derives outputs from loops).
- **Executed (P6):** 96x256 mapping [3]: record `{outputs 1024, batches 32}`, 64 MAC_ABK on 32 channels, issued 8,388,608 vs logical 24,576 (**341x**), no error. 240x256 mapping [24]: 13.3x. 512x256 map [32]: 16x. 256x1040: 8x.
- **Gates:** G2 (source meaning not preserved: the emitted program performs a different amount of work), G4 (selected program/cost describe padded work), G6 (renamed-equivalent program changes decisions and simulated cycles 4x).
- **Repair:** derive axis roles from index sets including the *store* index (a loop var in matrix+result and absent from the vector is an output axis; batch = var in vector+result, absent from matrix); compute the work-item output extent from the kernel's own loops and the declared work grid, never from `target.channels`; add an invariant check `issued scalar MACs == logical MACs x replicas` (the manifest already exposes `physical_to_logical_*` ratios) and reject or report overcompute; add a normalized-decision regression over syntactic variants.

### M-02 (blocker, frontend-wide) Shape names resolved from the caller's call stack
- **Where:** `allo/ir/utils.py:31-76` (`_get_global_vars` walks `inspect.currentframe()` outwards to `<module>` and copies every int/float local of every frame into the resolver's global table; only the *top-level* function's own closure values override, lines 68-75). Used by `allo/dataflow.py:596,631` and `allo/customize.py:1357`.
- **Executed (P15, no prior compile):** a script that merely defines `M = 7` at module scope before `build_case("attention_l128")`/`allo.compile` makes the compile fail with `NotImplementedError: AiM host broadcast of v_cache[:, :, 127] into a row_packed layout`; `M = 256` compiles **successfully with `v_cache: memref<32x128x256xbf16>`** (should be `4096`), i.e. a silently wrong K/V-cache geometry; `M = 4096` is correct. Post-mortem (P13) shows the SV kernel ranked over the QK choice set (96 candidates, row_packed) because `storage_extent` became `None`. This is why the earlier P6/P9 runs looked "history dependent": a leftover loop variable `M` in the probe script.
- **Gates:** G2 (silent wrong shapes), G6 (renamed/neighbouring programs), G7 (ordinary compilation depends on the user's frame contents).
- **Repair:** resolve annotation names from `func.__globals__` + the function's closure (and the enclosing region's closure) only; never from unrelated caller frames; raise on unresolved names. This is shared Allo infrastructure (FPGA/AIE blast radius) and needs the architect's review.

### M-03 (blocker) Token partitioning (paper "Token Partitioning") is not in the release
- **Read:** the mechanism (SV split over tokens with per-channel partial sums merged on the host, P in {1,2,4,8}, `grid_search`) is implemented in `.worktrees/allo-snapshot/allo/pim/aim_sv_partition.py` (`AiMSVPartitionSpec`, `compile_aim_sv_partition`, `merge_aim_sv_partials` at lines 36/259/252), `.worktrees/allo-snapshot/allo/pim/aim_qk_row_packed.py`, and the harness `reviews/experiments/aim_attention_campaign.py` (imports `studies.virtual_backend_architecture.aim_model.AimCostProgram`; host merge and host-link timing 32 GB/s are computed in the harness). Evidence bundle: `tenon-artifacts/rebuttal/reviewer-e/q3/` (9 shapes, 45 numerical variants). None of these files exist under `experiments/allo/allo/pim/`.
- **Release has:** the `replica_partitions` knob (`spmw_knobs.py:378-397`, domain `replica_partition_domain` = powers of two with `replicas*p <= 32`) which spreads one replica's **outputs** over `p` channels (`channels_per_replica`); it never splits a reduction axis and emits no host reduction. The only reduction split in the release is the CENT RMS bank-pair idiom (`spmw_aim.py:783-800 _check_bank_pair_span`).
- **Gates:** G1 (paper mechanism not reachable via the release API; research code stranded in a worktree), G2 (host partition not executed in release).
- **Repair:** port `aim_sv_partition` semantics as a reduction-axis partition knob of the AiM enumerator (channel groups over the reduction axis + an explicit host `ADD`/gather step priced by the host rules), with the numerical shadow check as a test oracle; or state in the paper claim matrix that the token-partition study is reproduction-only outside the package.

### M-04 (blocker for G1 unless documented as evaluation-only) AttAcc path is entirely outside the release
See section 5 for locations. No `attacc` entry in `spmw_codegen._BACKENDS` (1043), no target/cost in `allo/pim`. The archived plugin hard-codes another user's absolute paths and stub-imports `allo` from a private inspection copy.

### M-05 (major) Host-transfer cost is slice-blind; virtual estimate is 380x the simulated cycles
- **Where:** `allo/pim/costs/aim.py:356-370` (`host_transfer`: `bytes` -> bursts of 32 B, first burst + `(bursts-1)*nCCD`); `bytes` comes from the buffer manifest (`allo/compiler.py:125`), not from the host slice.
- **Executed (P8, attention_l128, virtual):** total 4,203,452 cycles; top spans `host_scatter` 2,097,186 and `host_broadcast` 2,097,186 (= 36 + 2*(2^20-1): the whole 2^25-byte K/V cache), while the emitted trace moves 1,024 `W MEM` bursts and the simulator reports 11,010 cycles. Each kernel score in `placement_ranking` therefore sits at ~4.19M.
- **Gates:** G5 (transfers do not describe the generated program; a model without validated domain advertises predictions through `backend="virtual"`). Ranking is unaffected because the term is constant across candidates (executed: chosen placements identical to the simulator route).
- **Repair:** price the resolved host slice (`_token_extent`/`AimDistributedHostTransfer.bursts` already know the burst count) and exclude resident buffers; record the cost's validity domain ("device-only command stream; host link not modelled") in the cost module docstring.

### M-06 (major) Two inconsistent cost routes for the same program; cost constants duplicate target geometry
- **Executed (P1/P2):** same 5-line ABK trace costs **255** via the knob path (`_contraction_profile_cycles`) and **156** via `allo.compile(..., layout=abk_layout)` (knob-free `all_bank_mac` rule); simulator 222. SBK: estimate 765 vs simulated 1527 (ranking order preserved).
- **Read:** `costs/aim.py:27-28` `LANES_PER_PU = 16`, `BANKS_PER_CHANNEL = 16` are used in `_bank_fanout` validation, `_columns`, `_mac_launches` while `_target_geometry` already derives lanes/banks from the target; a virtual sweep over banks/lanes would be silently mispriced. Timing constants carry provenance (simulator commit `0f28a07`, `GDDR6.cpp`) but no stated validity domain or accuracy expectation; units are simulator memory-system cycles (no wall time) — consistent with the paper table caption.
- **Gates:** G5. **Repair:** use `geometry.lanes/banks` everywhere; make the knob-free route call the same profile (or drop it); document units, scope ("device command stream only"), and observed error band (ABK +15%, SBK -50% on the GEMV probe).

### M-07 (major) Cost perturbation reaches selection (positive), but ties are resolved positionally and silently
- **Executed:** `test_aim_row_activation_cost_flips_batch_mapping` passes: setting `BATCH_MAPPING_ROW_ACTIVATE_CYCLES=0` flips `row_packed -> channels` (WR_GB 16 -> 4, MAC count unchanged), fingerprint changes. Layout perturbation (P2): forced SBK vs ABK produce different traces (16 `MAC_SBK` vs 1 `MAC_ABK`) and simulated cycles (1527 vs 222). **This satisfies G4/G5 "selected plan determines the executable".**
- **Executed (P1/P6):** 6-8 candidates tie at the minimum (e.g. `reuse_group` 1/2/4 x `replica_partitions` 16/32 all cost 255) and position 0 wins; `replica_partitions=16` vs `32` emit different channel masks (different programs) at equal cost. Ties are visible in `placement_ranking.kernels[*].scores` but not surfaced as ties. Minor-to-major under G4 "preserve ties where promised".

### M-08 (major) Timing-only nature not visible in the public result or release docs; nonzero simulator exit tolerated
- **Executed (P1):** `RunResult` has no `timing_only`/`numerics` flag; `CompiledCallable.__call__` accepts NumPy arrays and leaves outputs untouched. **Read:** `SPMW_ARCHITECTURE.md:62-67` and `docs/source/` never state that AiM is timing-only (only the skill file, `benchmarks/cent_aim/README.md` and `aim_lowering.py` docstring do). `_run_aim` (`spmw_aim.py:1136-1160`) returns success whenever a `memory_system_cycles` line is present even if `returncode != 0` (recorded in `extra` only).
- **Gates:** G2/G8 ("virtual results and emitted-only artifacts distinguishable", capability matrix). **Repair:** add `extra["execution"]="simulator-timing-only"` and `cycles_unit`, reject on nonzero return code, document in the backend capability matrix.

### M-09 (major) dtype and value semantics not enforced
- **Executed (P3):** `float32`/`int32` GEMVs compile to the same trace/cost as bf16 (AiM MAC is BF16 only); nonzero initial accumulator compiles identically to zero-init (bias GPR fixed at 0, `aim_lowering.py:1823-1850` `build_contraction` never sets `bias_gpr`). **Gates:** G2. **Repair:** matcher-level dtype check against `target` element width (16-bit) with an actionable error; represent initial-accumulator payload or reject non-zero init explicitly.

### M-10 (minor) Target variants vs fixed simulator
- **Executed (P5/P8):** 24-channel target emits masks for the 32-channel simulator with no diagnostic; `mac_reg slots=0` -> `IndexError`. Simulator channel count is hard-coded (README). **Repair:** `_run_aim` should compare target geometry with `example.yaml`/preset and refuse mismatches; validate `slots >= 1` in `_target_geometry`.

### M-11 (minor) Dead and inconsistent trace-encoding code
- **Read:** `AimCtx.cmd/_ISR_FIELDS/_operand_text/_handle_to_fields/resolve_moves` (`spmw_aim.py:60-300`) and the target `emit=lambda ctx: ctx.cmd(...)` lambdas (`targets.py:303-400`) are unreachable in the matcher path (`emit_program` goes records -> `_AimLowerer`; `_walk_buckets` is not called for AiM). `cmd()` encodes `channel_mask = 1 << channel_id` while the live lowering uses the reversed `1 << (channels-1-channel)` (`aim_lowering.py:495-500`). The skill file states 8 banks x 4 groups; the target declares 16 banks (4 x 4). **Gates:** G7 maintainability.

### M-12 (minor) CENT mapper conventions and duplicated target facts in production
- **Read:** `spmw_aim.py:378-384` `_EWMUL_SLOTS`, `_BANK_PAIR_STRIDE`, `_BANK_PAIR_SLOTS` ("CENT's mapper conventions"); `benchmarks/cent_aim/workloads.py:546-551` duplicates channel/bank/row facts; `costs/aim.py:27-28` duplicates lanes/banks. Acceptable as named target idioms if derived from the target (bank-group width is derived; slot numbers are not).

### M-13 (minor) Simulator discovery and side effects
- **Read:** `spmw_simenv.py:22-24` default root `<package>/../../simulators/aim_simulator` (outside the release package), override `AIM_SIMULATOR_ROOT`; `_run_aim` writes temp traces into `<root>/test` (write access to the simulator checkout required); docker-only path (host g++ 9.4 cannot build). Not documented in `docs/source`.

### Positive observations (executed)
- Route end-to-end works; selected layout and knobs determine the trace and the simulated cycles (P2, P7).
- Cost perturbation changes selection without changing legality (test executed).
- CENT `attention_l128` byte-identical to the pinned trace; opcode inventory equals the vendor's except the documented `WR_ABK` 1024 vs 256 (Tenon writes all four replica groups); simulated 11,010 = golden; CENT vendor 14,272 (`tenon-artifacts/skhynix/vendor/attention_l128/metrics.json`).
- No filtering of generated commands before simulation: `_run_aim` serializes `compiled.cmds` verbatim; `_aim_runtime_segments` always yields one segment with `repeat=1` (the `raw_cycles * repeat` composition hook exists but is inert). No hard-coded trace fragments or expected totals in production code; pins live in `benchmarks/cent_aim/reference_traces.py`, `vendor_contract.py` and `tests/pim/test_paper_golden.py` (test/provenance side).

---

## 4. Constants / specialization inventory (TASK.md table)

| Item | Location | Category | Judgement |
|---|---|---|---|
| 32 channels, 4 bg x 4 banks, 16384 rows, 1024 cols, 16-bit, GB 64x256b, GPR 512 KiB, AF LUT | `targets.py:249-300` | Physical capacities / ISA facts | Sourced (GDDR6_AiM_org, Hot Chips 34); validated via `_target_geometry`. OK. |
| `mac_reg slots=32` | `targets.py:280` | Installed mapper/simulator contract | Commented as such; OK (validate `>=1`). |
| `LANES_PER_PU=16`, `BANKS_PER_CHANNEL=16` | `costs/aim.py:27-28` | Duplicated target facts in cost | Should derive from geometry (M-06). |
| `N_*` timings, `CMD_*`, `BATCH_MAPPING_ROW_ACTIVATE_CYCLES`, `REUSE_GROUP_SWITCH_CYCLES`, `HOST_*_BURST_CYCLES`, `HOST_BURST_BYTES=32` | `costs/aim.py:30-66` | Measured/analytical cost coefficients | Provenance given (sim commit, `GDDR6.cpp`); validity domain/error band missing (M-06); host terms slice-blind (M-05). |
| `_EWMUL_SLOTS`, `_BANK_PAIR_STRIDE=2`, `_BANK_PAIR_SLOTS` | `spmw_aim.py:378-384` | Target idiom (CENT mapper) | Named; not derived; document or derive from target (M-12). |
| `channel_mask` reversed encoding | `aim_lowering.py:495-500` | ISA encoding | OK; conflicts with dead `AimCtx.cmd` (M-11). |
| `bias_gpr=0`, `input_gpr=0`, `result_gpr=0` | `aim_lowering.py:103-105`, `build_contraction` | Derived addresses | Fixed; acceptable for timing-only but not representing semantics (M-09). |
| `replica_partition_domain` powers of two | `aim_lowering.py:1973-1980` | Finite candidate catalog | Explicit policy; coverage scope should be stated (only powers of two). |
| `_default_reuse_candidates` powers of two + limit | `aim_lowering.py:541-549` | Finite candidate catalog | Explicit; OK. |
| `batch_dim` rule | `spmw_match_engine.py:1604` | Structural recognition | **Wrong precondition** (M-01). |
| `outputs = outer * channels` fallback | `aim_lowering.py:1805-1809` | Derived dimension | Uses target count, not program (M-01). |
| `TARGET_CHANNELS=32 ...` | `benchmarks/cent_aim/workloads.py:546-551` | Benchmark-side target facts | Benchmark package; acceptable, duplicative. |
| `CENT_CASES`, `DEFAULT_DECODE_SPEC` (D=4096,H=32,Dh=128,F=11008,L,replicas=4,cpr=8) | `workloads.py:610-958` | Benchmark dimensions | Benchmark side; select stage builders only (read: no case-ID conditionals in `allo/`). OK. |
| `TYPED_TRACE_SHA256`, `TYPED_TRACE_LINES`, `VENDOR_TRACE_CONTRACTS` | `benchmarks/cent_aim/reference_traces.py`, `vendor_contract.py` | Expected results / archived totals | Test/provenance data; not read by production. OK. |
| Golden rows (11,750 ... 697,316) | `tests/pim/test_paper_golden.py:58-71` | Expected results | Test side; default set runs 3 rows (~30 s). OK. |
| `_AIM_COMPILE_BUDGET_S=120` | `test_paper_golden.py:120` | Test budget | OK. |
| `tempfile ... dir=root/test`, docker image name `aim-simulator-build`, `test/example.yaml` | `spmw_aim.py:1109-1128`, `spmw_simenv.py:278-284` | Environment facts | Configurable root only; document (M-13). |
| Caller-stack name collection | `allo/ir/utils.py:55-66` | Frontend policy | **Defect** (M-02). |

---

## 5. AttAcc path: location and release options

### 5.1 Where each piece lives (read)

| Piece | Location | In `experiments/allo`? |
|---|---|---|
| Target description (16 HBM3 channels x 64 PIM units, gemv/softmax buffers, LD/WR_GB/MV_SB/MAC_AB) | `tenon-artifacts/attacc-llm-e2e/tenon/infra/attacc_target.py` (110 lines, Tenon `@allo.target` DSL, stub-imports `allo` from `/work/shared/users/sg2682/tmp/allo-tenon-inspect`, which no longer exists) | No |
| Enumerator (every DCC tiling as a LinearLayout Placement) + `@cost attacc_cost` | `.../tenon/infra/tenon_attacc_backend.py` (281 lines; absolute paths to `/work/shared/users/sg2682/project/pim-compilation/repro/dcc-attacc/src/src`, scratch and output dirs; registers `_BACKEND_CTX["attacc"]`). As archived, the `@rule` uses `pred_cyc` from the command replay, not the closed-form cost. | No |
| Closed-form tiling selector ("M8" `max(compute, movement)` for GEMV; `serial_q` n_mac-quantized serial cost for attention) over byte-exact command counts | `.../tenon/FULL_PIPELINE_FINDINGS.md` (formulas), `.../tenon/infra/score_guarantee.py:21 serial_q`, `.../tenon/infra/attacc_counts.py` (`count_gemv`/`count_attn`); compile-time harness `.../compile-time/ten_compile_table.py` | No |
| Command-replay model (2.97%/6.30%) | `.../tenon/infra/timing_replay.py` (413 lines; HBM3-PIM.cpp constants; imports DCC `generate_configs`) | No |
| Ramulator2-AttAcc integration | DCC checkout `/work/shared/users/sg2682/project/pim-compilation/repro/dcc-attacc/src/src/Ramulator2/ramulator2-AttAcc` (binary present, owned by `sg2682`) driven via DCC's `gen_trace_AttAcc_GEMV.run_GEMV` | No |
| e2e composition (GPT-13B decode, GPU roofline) | DCC `inference/system.py` + `.../tenon/infra/run_tenon_e2e_faithful.py` ("column swap" of per-kernel times) | No |
| Paper's final numbers (398/500, 123/316/61, MAPE 30.78/14.19) | `reviews/experiments/attacc_audit.py` -> `reviews/experiments/results/2026-09-28-attacc-audit-v2/{kernel-summary.json,all-500.csv}` (fresh analysis of archived measurements; `tenon_status: serial_q_cf_rank1_sim`) | No |
| Snapshot adapter | `.worktrees/allo-snapshot/allo/pim/attacc_gemv_shared_schedule.py` (authenticates the archived provider by SHA; `_ALLOWED_SHAPES` = exactly the four paper GEMV shapes, line 54) and `dev/07292026-shared-loop-layout-autoscheduler/prototype/attacc_gemv_shared_provider.py` (expects `.tenon-artifact-upstream-68d1a6a`, absent) | No (worktree / ignored dev) |

Conclusion: the AttAcc path is a **virtual-model + external-simulator evaluation
harness**, not a hardware runner and not a release backend. It satisfies the
paper's "modeled decode latency" qualification only as an archived artifact.
The snapshot adapter is benchmark-identity gated (`_ALLOWED_SHAPES`) and must
not be transplanted.

### 5.2 Options

**Option A: documented evaluation-only artifact.** Record in `CLAIMS.md` that
AttAcc rows are reproduction-only, with the artifact path, the DCC dependency
(ISCA'26 DCC repo + `ramulator2-AttAcc` build), and the fact that selection used
closed-form costs while prediction used the replay. Required work: rewrite the
absolute paths in `tenon_attacc_backend.py`/`timing_replay.py`/`ten_compile_table.py`
to a documented `DCC_ROOT`; replace the stub-import with the installed `allo`;
license review of DCC code reuse. Does not satisfy G1 if AttAcc counts as a
"substantive evaluation path" of the frozen paper (section 5.3-5.5 of the paper
are substantive).

**Option B: expose the virtual-model path through the public API.** Feasible in
the existing DSL: (1) `build_attacc_target()` in `allo/pim/targets.py` (port of
`attacc_target.py`, structure only); (2) `attacc_cost` in `allo/pim/costs/attacc.py`
implementing M8/serial_q over command counts derived from the candidate's
LinearLayout tiling (port of `attacc_counts.py`, with provenance to
`HBM3-PIM.cpp`); (3) an enumerator registered via `register_enumerator("attacc")`
that exposes the per-unit tiling extent `s` as a **knob** (the archived plugin
had to search at the workload-schedule level because the placement enumerator
dedups candidates differing only in `s`: FULL_PIPELINE_FINDINGS.md v5); (4) a
workload family (GEMV + attention score) through `allo.compile(..., backend="virtual")`.
This yields target + cost + workload with no simulator and would make the
compile-time and selection claims reproducible. **Not** feasible without DCC:
trace emission (`gen_trace_AttAcc_*`) and `ramulator2-AttAcc` execution; these
stay an optional, documented external runner (`_run_attacc` behind
`ATTACC_DCC_ROOT`). The e2e decode composition (GPU roofline, `inference/system.py`)
should remain an evaluation artifact.

Effort estimate (read): target 1 day, cost + counts port 2-3 days (needs the
byte-exact count mirrors re-verified against DCC traces), knob/enumerator 1-2
days, tests with reviewer-chosen tilings 1 day.

---

## 6. Open questions

1. Is the `batch_dim` rule (M-01) also used by Samsung/UPMEM cost graphs (`spmw_plan.py:666` sets `batch` for every backend)? If so the over-issue affects their costing too (suspected; not probed here).
2. M-02 is upstream Allo behaviour (`_get_global_vars`); the fix touches shared frontend code. Who owns the blast-radius review (architect)?
3. Should the paper's "Token Partitioning" row be classified as a compiler mechanism (then M-03 must be ported) or as an evaluation study (then documented as reproduction-only)? The paper text says "Tenon expresses ... as channel placement with partial-result reduction" and "search over 1,2,4,8 partitions", which reads as a compiler mechanism.
4. The paper's virtual study reports predicted 1.30x/1.547x; the release test (`test_aim_virtual_study.py`) only checks opcode counts, and the release `aim_cost` is slice-blind for host terms (M-05). Which cost program produced the paper's "predicted" numbers (`studies.virtual_backend_architecture.aim_model.AimCostProgram` in `reviews/experiments`, suspected), and should it replace/inform `costs/aim.py`?
5. `benchmarks/` ships with `find_packages()`; is it intended to be part of the installed artifact (it is needed by `tests/pim/test_cent_aim_workloads.py` and `test_paper_golden.py`)?
6. The 11 `paper_full` AiM golden rows were not executed in this audit (marker-skipped; each needs one docker run of up to 237k-line traces). Status: UNVERIFIED at `8160d6d`.
7. Tie policy: should `placement_ranking` expose tied minima explicitly, and should the emitted program for `replica_partitions=16` vs `32` be considered the same plan?

## Appendix: executed probe summary

| Probe | What | Result |
|---|---|---|
| P1 `p1_route_gemv.py` | public route + docker | 216 cands, cost 255, sim 222, outputs untouched |
| P2 `p2_layout_perturb.py` | forced SBK vs ABK | traces differ; sim 1527 vs 222; estimates 765 vs 156 |
| P3 `p3_neighbors.py` | shapes/dtypes/ops | see section 2; sub op rejected; dtypes not enforced |
| P4 `p4_cent_attention.py` | attention_l128 | SHA pinned, 11,010 = golden, WR_ABK 1024 vs vendor 256 |
| P5 `p5_target_variants.py` | geometry variants | GB<row rejected; 24 ch accepted; slots=0 IndexError |
| P6 `p6_coverage.py` | issued vs logical MACs | ratios 8x, 341x, 13x, 2.3x, 9x, 8x, 16x, 2x |
| P7 `p7_syntax_ab.py` | equivalent GEMV spellings | 1 vs 8 MAC_ABK, sim 222 vs 858 |
| P8 `p8_state_and_cost.py` | virtual spans, 24-ch mask | host spans 2,097,186 each; mask 0xFFFF00 |
| P9/P11-P14 | history-dependence bisection | traced to caller-scope `M` (M-02) |
| P15 `p15_caller_scope.py` | minimal M-02 reproducer | M=7 fails, M=256 silently wrong, M=4096 ok |
| pytest | matcher, virtual study, golden -k aim, gemv -k aim | 5 + 3 (+11 skipped) + 2 passed |
