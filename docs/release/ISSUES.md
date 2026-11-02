# Tenon release issue ledger

Each entry: observed failure or source concern, severity, root cause, affected
public behaviour, planned repair, closure evidence. IDs come from the per-area
audits in `audit/` (S Samsung, U UPMEM, M AiM/AttAcc, V1 APU v1, G2 Gemini-II,
C core, P packaging) and the initial survey (`RELEASE_AUDIT.md` A01-A12 in the
containing repository). Severity: **blocker** (silent wrong code, missing-output
success, or a paper mechanism with no release route), **major** (gate
violation needing repair before release), **minor**.

Status values: `open`, `in-progress`, `fixed (commit)`, `boundary` (documented
non-core support boundary, with justification), `unverified`.

Frontend findings (F-xx) were discovered during repair work and concern
shared Allo infrastructure; they are listed here because they affect Tenon's
G2 guarantees.

## Core compiler (audit/core.md)

| ID | Sev | Finding | Gate | Status |
|---|---|---|---|---|
| C-01 | blocker | No access-scope / operand legality analysis: a hand-supplied placement violating `MAC.dst=grf_b` is accepted, emitted and priced cheaper than the legal one; `srf` declared legal but unemittable | G1 G2 G4 | open |
| C-02 | blocker | Unmatched stores silently dropped (`y = acc / x[0]` compiles, `y` never produced) | G2 | fixed: `match_workload` store-coverage analysis (`spmw_match_engine._FunctionCoverage`, `require_match_coverage`), `UnsupportedWorkloadError`; tests `tests/pim/test_match_coverage.py` |
| C-03 | major | Ranking precedes materialization; only `InfeasibleSchedule` counts as rejection; per-bucket fallback never materializes | G4 | open |
| C-04 | major | Work-grid vs target-grid mismatch is warn-only (`_check_work_grid`) | G2 | open |
| C-05 | major | Per-kernel ranking then `_reconcile_resident_pairs` without re-costing (A02); `grid_search` unused by matcher route | G4 | open |
| C-06 | major | Ties broken by knob registration position, undocumented | G4 | open |
| C-07 | major | `_eval_sym` defaults unbound `UnitId` to 0 at AiM concrete bindings (A05) | G2 | fixed: raises unless `representative=True`; AiM binds every level via `AimCtx._unit_env` |
| C-08 | major | `lanes=8` Samsung GRF width hard-coded in target-neutral `spmw_plan.py` | G3 G5 | open |
| C-09 | blocker | No floating-point reduction-order / reassociation permission anywhere; `dual_fiber`, UPMEM `nc/kc` split reductions unconditionally | G1 G2 | open |
| C-10 | minor | Host program: Python orchestration, no lowered host computation; `schedule_residency`/`schedule_cost` exported without production caller | G2 G7 | open |
| C-11 | major | Minimum-movement layout machinery (rank formula, chains, bound, ties, masked exact count) absent from the package (A01) | G1 G4 | open; salvage plan in audit/core.md §3.3 |
| C-12 | minor | `LinearLayout.identity` silently zero-maps unmatched dims | G2 | open |
| C-13 | minor | Broad exception swallowing in matcher/codegen/autoschedule | G2 | open |
| C-14 | major | Nonzero or non-dominating accumulator initializers silently dropped (S-08, M-09, V1-02 share the root cause) | G2 | fixed at the matcher: `extra["accumulator_init"]` in {zero, carried}; nonzero/guarded/inside-loop initializers raise; Samsung/AiM/APU-v1 matcher routes reject `carried` (they zero the accumulator) |

## Frontend (shared Allo infrastructure)

| ID | Sev | Finding | Gate | Status |
|---|---|---|---|---|
| F-01 (M-02) | blocker | Shape names in annotations resolved from the caller's call stack (`allo/ir/utils.py:_get_global_vars`): an unrelated `M = 256` in the calling script silently compiles `attention_l128` with a wrong K/V-cache geometry | G2 G6 G7 | open; shared-file change, needs upstream-compatibility review |
| F-02 | blocker | Integer literals typed as i32 and silently truncated (`-(1 << 40)` and `m: int64 = -1099511627776` become `arith.constant 0 : i32`); `1 << 40` on an i32 value lowers to an undefined C shift | G2 | open (found while fixing U-02) |

## Samsung HBM-PIM (audit/samsung.md)

| ID | Sev | Finding | Gate | Status |
|---|---|---|---|---|
| S-01 (A03) | blocker | Selected placement/layout never determines the executed REDUCE program: `_run_samsung` writes a fixed 5-line CRF and the conductor regenerates the vendor GEMV microcode; cycles/outputs identical across placements | G4 | open |
| S-02 (A04) | blocker | All-zero or truncated outputs returned as success without outputs | G2 | fixed: readback returns arrays unconditionally; absent/short `out.bin` raises; `CompiledCallable` raises on missing gathered outputs; test `tests/pim/samsung_hbm_pim/gemv/test_gemv_samsung_readback.py` |
| S-03 | blocker | Paper's source-order FP16 "ordered fallback" exists only in the uncommitted simulator overlay (`TEMPORAL_REDUCE`) and a snapshot-only driver; unreachable from `allo.compile` | G1 | open |
| S-04 | blocker | The `pim_driver` that runs is built from uncommitted `dev/` conductor sources; the committed overlay lacks `--op GENERIC`/REDUCE | G7 G8 | open |
| S-05 | major | Operand roles depend on source operand order (`x[k]*W[i,k]` fails at run); host-eligibility lever inverted vs executed program | G6 G3 G2 | open |
| S-06 | major | Reported cycles are those of the zero-padded canonical program (K=128/100, M=2048/128 all report 4435) without disclosure | G5 G8 | open |
| S-07 | major | Elementwise ops advertised but unreachable; GENERIC runtime mis-tiles | G1 G8 | open |
| S-08 | major | `acc: f16 = 0.5` silently dropped | G2 | fixed (C-14) |
| S-09 | minor | `_samsung_num_pim_blocks` broad-except -> 8 (reachable only on GENERIC route) | G2 | open |
| S-10 | minor | Work-grid/partition mismatches warn only | G2 | open (see C-04) |
| S-11 | minor | No Samsung docs page; stale skill doc; simulator discovery outside package | G7 | open |
| S-12 | minor | Cost constants lack provenance; ties by position; cost changes cannot reach the simulator (S-01) | G5 | open |
| S-13 | minor | XOR swizzle 148->116 reachable only via a test harness | G1 | open |

## UPMEM (audit/upmem.md)

| ID | Sev | Finding | Gate | Status |
|---|---|---|---|---|
| U-01 | blocker | int64/float32 operands silently computed as int32 (`_i32` truncates inputs and oracle alike) | G2 | fixed: `UPMEM_ROLE_ELEMENT_TYPES`, recognition-time and ABI-time type checks, exact packers/`_store_exact`; tests `tests/pim/test_upmem_dtypes.py` |
| U-02 (A09) | blocker | `backend/c.py` maps >64-bit integers to `uint64_t`; signed shift/compare/div/high-bit ops wrong (production host-placed groups) | G2 | fixed: `WideIntegerError` allow-list check in `MLIRToCModule`; tests in `tests/test_c_backend.py` |
| U-03 (A08) | major | MV/GEMM/reduction have no zero-init check; irregular families check existence only | G2 | partially covered by C-14 (matcher `accumulator_init` fact); UPMEM-side consumption of the fact still open |
| U-04 | major | Paper's 391x65 GEMV 1-32 DPU scaling with uneven partitions not reproducible (even division required) | G1 | open |
| U-05 | major | Legality comes from cost-model feature functions (divisibility), not the padding-capable templates; realizable shapes rejected | G3 G4 | open |
| U-06 | major | Shape-exact paths (`tasklet_private_wram` only at the paper MMTV shape; pointwise only ADD and literal `a*A + b*B`) | G3 | open |
| U-07 (A11) | major | Execution requires the archived uPIMulator binary, the `GENERIC` slot from the snapshot tar, and the pinned SDK manifest | G7 | open |
| U-08 | major | `SPMW_ARCHITECTURE.md`, `design/06`, `design/08`, skill doc describe the deleted `UPMEMProgram` route | G7 | open |
| U-09 | major | Virtual cycles outside the calibration stratum carry no validity status (1.39x-1.53x off) | G5 | open |
| U-10 | major | `kmeans_update` recognizer admits any op-superset kernel and runs it on the host at cost 0 | G3 G5 | open |
| U-11..U-15 | minor | Python-side selection compaction and cross-DPU merge; vestigial knobs; carried F2 layout unread by codegen; diagnostics; tests import `scripts/` | | open |

## SK-Hynix AiM and AttAcc (audit/aim-attacc.md)

| ID | Sev | Finding | Gate | Status |
|---|---|---|---|---|
| M-01 | blocker | Output axis mis-derived as batch; 8x to 341x MAC over-issue for syntactic variants; fallback used `target.channels` | G2 G4 G6 | fixed: structural `axis_roles` in the matcher, `contraction_shape` from roles and work grid, `AimCoverageError` invariant on issued vs program MACs; tests `tests/pim/test_aim_axis_roles.py` |
| M-02 | blocker | Shape names from caller frames | G2 | see F-01 |
| M-03 | blocker | Token partitioning (paper subsection) not in the release; `replica_partitions` splits outputs, not a reduction with host reduce | G1 G2 | open |
| M-04 | blocker | AttAcc compiler/virtual-model path entirely outside the release | G1 | open; options in audit §5 |
| M-05 | major | Host-transfer cost slice-blind (virtual estimate 380x simulated) | G5 | open |
| M-06 | major | Two inconsistent cost routes for the same trace; cost hard-codes geometry | G5 | open |
| M-07 | major | Min-cost ties resolved positionally and silently | G4 | open (see C-06) |
| M-08 | major | Timing-only not marked in results; nonzero exit code tolerated | G2 | partially fixed: `RunResult.kind="simulator-timing-only"`, outputs never written; exit-code tolerance still open |
| M-09 | major | Nonzero initial accumulator identical to zero-init trace | G2 | fixed (C-14) |
| M-10 | minor | 24-channel target emits masks for a 32-channel simulator; `slots=0` IndexError | G2 | fixed: `_check_simulator_geometry`, slots check |
| M-11..M-13 | minor | Dead `AimCtx.cmd` encodings; CENT slot conventions hard-coded; temp traces inside simulator checkout; no docs page | | open |

## GSI APU v1 (audit/apu-v1.md)

| ID | Sev | Finding | Gate | Status |
|---|---|---|---|---|
| V1-01 | blocker | Native recipes (`apu_v1_histogram/record_frequency/moments/distance_argmin`) are regex recognizers that fail open on semantic mutations (`counts += 2`, nonzero init, swapped columns emit the paper kernel) | G2 G3 G6 | open |
| V1-02 | blocker | Vector device route never transfers the carried output (`C += A@B` with nonzero C returns `A@B` on hardware; virtual/functional disagree) | G2 | open (matcher-level `carried` rejection covers only the matcher route) |
| V1-03 | major | FP16 GEMV default selects a reassociating tree plan; no reassociation permission in the API; paper's source-order run came from an external script | G2 G1 | open (see C-09) |
| V1-04 | major | `APUv1Program` ignored `backend`; virtual/functional launched hardware | G2 G8 | fixed: `CompiledAPUv1Program` honours `virtual` (cost graph) and `functional` (host execution of the retained portable C); tests `tests/perf/test_run_result_contract.py` |
| V1-05 | major | Paper's fitted count rule absent from the released cost program; native estimates off 19-56% without validity flag | G1 G5 | open |
| V1-06 | major | Tile choice is a fixed `min(choices)` heuristic; 12.71x row plans not the default winner; golden 7,145,109 vs paper 7,349,987 | G4 G1 | open |
| V1-07 | major | No board lock in production runtimes; profile parsed as last cumulative `total` with no freshness proof | G8 | open |
| V1-08 | major | Template default under `Path.home()/shared/...`; 12 GSI/Synopsys files copied verbatim without license headers | G7 | open |
| V1-09 | major | Legacy matcher runner guesses output shapes and skips missing output files | G2 | open |
| V1-10..V1-14 | minor | cost=None picks slowest plan; dead API surface; Newton sqrt error; misleading errors; short-readback reshape errors | | open |

## GSI Gemini-II / APU G2 (audit/apu-g2.md)

| ID | Sev | Finding | Gate | Status |
|---|---|---|---|---|
| G2-01 (A06) | blocker | Matcher route returned success for source without compiling or executing | G2 G8 | fixed: `spmw_apu_v2.py`, enumerator, knobs and `_BACKENDS` entry removed; `allo.compile` raises `NotImplementedError` for source on `apu_v2`; test `test_gemv_apu_v2_source_rejected` |
| G2-02 | blocker | No source route to Gemini-II; paper kernels are typed records built by a per-case switch in an artifact adapter; nothing searched or ranked | G1 G3 | open |
| G2-03 | major | Typed/composed cost "cycles" are an op count; no rule covers typed opcodes; real-card calibrations unreachable | G5 | open |
| G2-04 | major | No hardware tests; no hardware evidence at the candidate revision | G8 | open (remote validation pending) |
| G2-05 | major | Composed route single-carrier; `APUG2DotTiling` disconnected | G3 | open |
| G2-06 | major | Hard-coded `/opt/gsi/...` paths; `gsi_tool` on PATH; device literal `apu-00`; no `flock` | G7 G8 | open |
| G2-07 | major | No-op, virtual and hardware results indistinguishable | G2 | fixed by `RunResult.kind` |
| G2-08..G2-12 | minor | Fingerprint only echoed; mod-2^bits wrap in decode; padded-lane division; dead uint16 templates; docs never mention Gemini-II | | open |

## Packaging, installation, documentation (audit/packaging.md)

| ID | Sev | Finding | Gate | Status |
|---|---|---|---|---|
| P-01 | blocker | No `package_data`/`MANIFEST.in`: sdist/wheel ship zero non-`.py` assets (`allo/pim/templates/apu_g2/**`, `.mlir` templates) | G7 | open |
| P-02 | blocker | `allo/_mlir` is a symlink into `mlir/build`; wheel has no extension; sdist omits `mlir/`; only editable install works | G7 | open |
| P-03 | blocker | UPMEM SDK manifest hash hard-required; fresh correct build can never run (A11) | G7 | open |
| P-04 | blocker | Modified Samsung PIMSimulator sources with Samsung's non-redistribution notice committed under `dev/` in the containing repository | G7 | open (containing repo; not in the release payload) |
| P-05 | major | Committed `pim_driver.cc` overlay stale (323 vs 721 lines); installed PIMSimulator has uncommitted modifications; cannot be rebuilt from committed source | G7 G8 | open (same as S-04) |
| P-06 | major | uPIMulator checkout has 24 modified + 22 untracked files not captured by `setup/`; no simulator commit pins documented | G7 | open |
| P-07 | major | Release identity is upstream Allo (`allo 0.5`, upstream README, no Tenon mention) | G7 | open |
| P-08 | major | `SPMW_ARCHITECTURE.md` references nonexistent modules and a removed API; omits APU v2 | G7 | open |
| P-09 | major | No install guide, capability matrix, env-var documentation, or test-run instructions | G7 G8 | open |
| P-10..P-21 | minor/medium | `/opt/gsi` paths outside `spmw_simenv`; docker image literals; `examples`/`benchmarks` installed as packages; dev tooling in `install_requires`; private calibration script; home-share defaults; proprietary UPMEM SDK image naming; stray tracked files; stale APU v2 docs; `bdist_wheel` fails with cmake 3.16; AiM temp traces in checkout | G7 | open |
