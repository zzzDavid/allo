<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Tenon backend capability matrix

Scope: `experiments/allo`, branch `tenon`, at `ffb4a78` plus uncommitted
round-2 repairs (2026-11-02). Sources: `docs/release/audit/*.md`,
`docs/release/ISSUES.md`, `docs/release/VALIDATION.md`, and the runs noted in
the "Evidence" rows. Finding IDs (S-xx, M-xx, U-xx, V1-xx, G2-xx, C-xx, P-xx)
refer to `ISSUES.md`; check it for the current status of each.

Terms used below:

- **Input kind.** *Source*: an ordinary `@allo` workload (usually an
  `allo.dataflow.region` with `@allo.work(mapping=...)`) passed to
  `allo.compile(workload, target, cost)`. *Typed*: a typed program object
  (`APUv1Program`, `APUG2TypedProgram`, `APUG2ComposedContractionProgram`)
  passed to `allo.compile`; these bypass the matcher.
- **Modes**, matching `RunResult.kind`: *compile-only* (emit and inspect, no
  run), *analytical* (`backend="virtual"`, `kind="virtual"`; cost-program
  evaluation of the retained execution graph; nothing executes and caller
  arrays are not written), *simulator functional* (`kind="simulator"`;
  outputs read back from the simulator), *simulator timing-only*
  (`kind="simulator-timing-only"`; no outputs exist), *hardware*
  (`kind="device"`), *host functional* (`kind="functional"`; host execution
  of the lowered program, cycles are not a measurement).
- **Evidence status.** *Verified* means run at or after `ffb4a78` on the
  authors' host through the public API and recorded; *verified (earlier
  revision)* names that revision; **UNVERIFIED** means no run at the
  candidate revision, whatever older artifacts say. No simulator result is a
  hardware measurement.

## Summary

| Backend | Target builder / cost | Input | Compile-only | Analytical | Simulator functional | Simulator timing-only | Hardware |
|---|---|---|---|---|---|---|---|
| Samsung HBM-PIM | `build_samsung_target()` / `samsung_cost` | source | yes | yes | yes (PIMSimulator `pim_driver`) | (batched path only, untested) | no |
| SK hynix AiM | `build_aim_target()` / `aim_cost` | source | yes | yes | no | yes (Ramulator2 `aim_simulator`) | no |
| UPMEM | `build_upmem_target()` / `upmem_cost` | source | yes | yes | yes (uPIMulator + DPU SDK) | no | no |
| GSI APU v1, source contraction route | `build_apu_v1_target()` / `apu_v1_cost` | source (rank-2 contractions, 1-D GEMV) | yes | yes (see note) | no simulator exists | no | yes (Gemini-I board) |
| GSI APU v1, `APUv1Program` scalar route | same | typed (`APUv1Phase(vectorize=False)`) | yes | yes | no | no | yes; plus host functional |
| GSI APU v1, native recipes | same | typed (`APUv1Phase(vectorize="required")`) | yes | yes | no | no | yes |
| GSI APU v2 (Gemini-II), `APUG2TypedProgram` | `build_apu_g2_target()` (= `build_apu_v2_target()`) / `apu_g2_cost` | typed | yes | yes (op count) | no | no | yes (Gemini-II card) |
| GSI APU v2, `APUG2ComposedContractionProgram` | same | typed | yes | yes (op count) | no | no | yes |
| GSI APU v2, source | same | source | **rejected**: `NotImplementedError` | - | - | - | - |

Every PIM target compiles and runs analytically with only Allo installed.
Everything else needs the external pieces in `simulators.md`.

## Samsung HBM-PIM

| Aspect | Status |
|---|---|
| Input | Source GEMV/GEMM: one `MAC` per work-id stream with roles `x`, `y`, `acc` (`acc += W[i,k] * x[k]`), mapping `[P, U]` with `P <= 16`, `U <= 8`. ADD/MUL/RELU and `y += W*x` forms are declared on the target but rejected at compile time (`NotImplementedError ... missing one of x/y/acc roles`, S-07). |
| Compile-only | `compiled.compiled.cmds` is the emitted PIM command list (e.g. 6 commands for shared CRF), `compiled.placement_ranking` the ranked placements (24 per bucket, 3 bases). |
| Analytical | `samsung_cost` over the retained graph; 226 cycles for the 4096x256 GEMV at the selected `dual_fiber` placement. |
| Simulator | PIMSimulator `pim_driver --op GENERIC --op-kind REDUCE`; outputs read back from `out.bin` (absent/short output raises, S-02 fixed). |
| Numerical scope | FP16 only (inputs cast to `float16`); per-lane FP16 with 16-way regrouping, **not** source order (the paper's ordered fallback is outside the release, S-03; no reassociation permission exists, C-09). `x[k]*W[i,k]` operand order fails at run time (S-05). |
| What cycles mean | Simulated DRAM cycles of the **vendor canonical GEMV program on the zero-padded shape** (K padded to 256, M to 4096): K=100/128/256 and M=128/2048/4096 all report 4435 (S-06). The selected placement/layout does **not** determine the executed program; the conductor regenerates the vendor microcode (S-01). |
| Hardware | none (no HBM-PIM device). |
| Evidence | Verified 2026-11-02 at the candidate: `tests/pim/test_gemv_all_targets.py::test_gemv_samsung_hbm_pim` passed (functional, 4096x256), and the installed wheel ran the same GEMV on PIMSimulator from outside the checkout: 4435 cycles, max abs error 4.5e-4 vs float32 (`INSTALL.md` section 6). Paper rows E1 reproduce (`CLAIMS.md`). The `pim_driver` cannot be rebuilt from public sources (S-04, `simulators.md`). |

## SK hynix AiM

| Aspect | Status |
|---|---|
| Input | Source GEMV/attention-style MAC (`acc += W*x`), elementwise MUL/ADD and AF through target ops; mapping a prefix of `[32, 4, 4]`. Output/reduction axes are derived structurally (`axis_roles`, M-01 fixed); issued MACs are checked against the program (`AimCoverageError`). |
| Compile-only | `compiled.compiled.cmds` is the exact Ramulator2 trace text (`AiM WR_GB ...`, `MAC_ABK ...`, `EOC`). |
| Analytical | `aim_cost`; host-transfer cost is slice-blind (virtual estimate up to 380x the simulated cycles on small GEMVs, M-05). |
| Simulator | Ramulator2 timing simulator in Docker (`aim-simulator-build`); `kind="simulator-timing-only"`, `extra["outputs"]` absent, caller arrays untouched. Nonzero simulator exit codes are recorded in `extra["returncode"]` but not rejected (M-08, open). |
| Numerical scope | **None.** The simulator has no data path. dtype is not enforced (float32/int32 GEMVs emit the same trace as bf16); nonzero initial accumulators produce the zero-init trace (timing-equivalent only). |
| What cycles mean | Max over channels of Ramulator2 `memory_system_cycles`, per distinct trace segment, multiplied by the segment's compile-time repeat count and summed. Simulator geometry is fixed at 32 channels; other targets are rejected. |
| Hardware | none. |
| Evidence | Verified 2026-11-02: `test_gemv_aim` passed through the public API (trace now written to a private temp directory, not the simulator checkout). CENT reproduction: 14 traces byte-identical to pinned SHA-256, golden cycles 11,750 / 129,605 / 11,010 / 6,307 (`VALIDATION.md`, after repair increment 1). Token partitioning and the AttAcc path are **not** in the release (M-03, M-04). |

## UPMEM

| Aspect | Status |
|---|---|
| Input | Source kernels recognized as families: `pointwise` (ADD, `a*A + b*B` with literal coefficients), `reduction`, `matrix_vector`, `gemm`, `histogram`, `selection` (odd values), `kmeans_distances`, `kmeans_update` (host-placed), `feature_gradient`; mapping `[N]` over DPUs with even partitions. |
| Compile-only | `compiled.compiled.cmds` names the segments; `compiled.compiled.layout_ctx.segments` holds the DPU `task.c` source per segment. |
| Analytical | `upmem_cost`; calibrated factors are valid inside the calibration stratum only, outside it virtual cycles were 1.39x-1.53x off without a validity flag (U-09). |
| Simulator | DPU program built with the DPU SDK in Docker, run on uPIMulator with a fixture; uPIMulator byte-compares the observed MRAM window with the host oracle (`emit_c_from_mlir`) and the decoded outputs overwrite the caller's arrays (`kind="simulator"`). |
| Numerical scope | Element types enforced per role (`UPMEM_ROLE_ELEMENT_TYPES`): int32 operands, int64 accumulators for reduction/k-means distances/gradients (U-01 fixed); wide-integer host code rejected instead of truncated (U-02 fixed). Shape constraints come from cost-model features: MV `rows % 12 == 0`, `K % 16 == 0`, `rows > 12`; GEMM `rows % 12 == 0`, `N`, `K` divisibility; uneven DPU partitions rejected (U-04, U-05). |
| What cycles mean | Sum over DPU segments of the max per-DPU uPIMulator `logic_cycle` (350 MHz logic clock); host packing, host transfers and host-placed groups are excluded (`extra["timing_scope"]`). |
| Hardware | none (no UPMEM DIMMs). |
| Evidence | Verified 2026-11-02: `test_gemv_upmem` passed on the pinned-campaign route and again with `UPIMULATOR_ROOT`/`UPIMULATOR_BIN` set (user-build route, identity `user-build:d45e285c34b5`). 13 `paper_full` golden rows pass (`VALIDATION.md`). A user build from public sources is **not yet possible**: the Tenon uPIMulator patch is unpublished (P-06), and the run record does not yet carry `sdk_build` (P-03, pending call site in `spmw_upmem.py`). |

## GSI APU v1 (Gemini-I)

Three routes share `build_apu_v1_target()`.

| Aspect | Source contraction route | `APUv1Program` scalar route | Native recipes |
|---|---|---|---|
| Input | Source `@allo` function (not a dataflow region) whose stores are rank-2 contractions `C[i,j] += A[i,k]*B[k,j]` (mul, and/or/xor/xnor-popcount), 1-D GEMV lifted with a unit axis | `APUv1Program([APUv1Phase(fn, vectorize=False)])`: whole function lowered to C11 with `emit_c_from_mlir`, runs on APUC 0 | `APUv1Program([APUv1Phase(fn, vectorize="required")])` matched by one of four recognizers: dense 256-bin histogram (32,768 values), squared-L2 argmin (P=32,768, D<=48), bivariate moments, record frequency |
| Plan selection | Finite recipe catalog (`temporal_svp`, `temporal_dma_coalescing*`, `..._acc{2,4,8}`, `spatial_gemv_group_reduction`, ...); every candidate realized to GVML, infeasible ones dropped, ranked by `apu_v1_cost` on the materialized instructions; `layout=` forces a plan | none | the recognizer's single template |
| Analytical | yes; note: `backend="virtual"` on this route executes the same NumPy reference as `"functional"` and reports `kind="functional"` with the analytical cycles | yes (`kind="virtual"`, V1-04 fixed) | yes |
| Host functional | yes (NumPy reference of the source; `kind="functional"`) | yes (retained portable C on the host) | rejected at compile time |
| Hardware | yes: GVML project built with the ARC toolchain, launched on the board, outputs read back | yes | yes |
| Numerical scope | FP16, uint16, bitwise; the default FP16 GEMV plan **reassociates** the reduction (V1-03, no permission flag, C-09); nonzero initial `C` on plans without output ingress: see V1-02 | C11 semantics of the C backend; `sqrt` replaced by a Newton helper that is wrong above about 1e8 (V1-12) | fixed shapes/dtypes listed above; recognizers **fail open** on several semantic mutations (`+= 2`, nonzero init, swapped columns emit the paper kernel, V1-01) |
| What cycles mean | ARC core cycles (`crun`) of the profiled `total` section printed by the device build (`PROF_PRINT`), read from the board log | same | same |
| Evidence | Baseline `test_gemv_apu_v1` ran on the board at `8160d6d` (verified, earlier revision). **UNVERIFIED** at the candidate revision: no post-repair device run. Paper numbers (E9-E12) came from an external locked runner (`CLAIMS.md`). | **UNVERIFIED** on hardware at the candidate; virtual/functional covered by `tests/perf/test_run_result_contract.py` | **UNVERIFIED** on hardware at the candidate; four kernels compile-only in `test_paper_golden.py`, `histogram` has a board test |

A legacy matcher route (`spmw_apu_v1.py`, grouped FP16 MAC over four APUCs)
is still reachable for dataflow regions; it is mostly dead and guesses output
shapes (V1-09). Board launches take an exclusive `flock`
(`TENON_APU_V1_LOCK`, default `/tmp/tenon-apu-v1-<pci node>.lock`); the tests
use the same lock (V1-07 repair, in progress).

## GSI APU v2 (Gemini-II)

| Aspect | `APUG2TypedProgram` | `APUG2ComposedContractionProgram` |
|---|---|---|
| Input | Typed op record: `add`, `mul`, `div`, `block_sum`, `dot` over `input_types`/`output_type`, `shape`, `reduction_extent`, `repetitions` | Typed record of independent dot products `(dot_count, K)` with an epilogue (identity, two-coefficient pair-affine, accumulate); GEMM/GEMV/batched forms must be pre-packed by the caller with the host helpers |
| Legality | exact output width (`out_bits == lhs + rhs (+ log2 K)`), shared signedness, unsigned-only div/block_sum, <= 24-bit segments, K power of two in [2, 4096], up to 262,144 elements | one `(4, 65536)` carrier: `padded(dot_count) <= 4*65536/K_pad`, `K_pad <= 256`, K power of two; larger problems are rejected, not tiled (G2-05) |
| Plan selection | none: fixed layout, nothing ranked; `layout=`/`host_moves=` rejected | none: one deterministic `APUG2DotTilePlan` (`LinearLayout` over F2); nothing searched (G2-02) |
| Analytical | `kind="virtual"`; the "cycles" are an **operation count**, not calibrated cycles (G2-03) | same |
| Hardware | fixed project template (`allo/pim/templates/apu_g2`) built per call with CMake (ARC device library + host), run on card `apu-00` after `gsi_tool info` reports it Available and core 0 idle; outputs read back and compared with an int64 NumPy oracle, mismatch raises; device values are returned | same, plus layout fingerprint and epilogue attested |
| Numerical scope | exact integer arithmetic modulo the declared widths (1-24 bit, signed/unsigned) | same |
| What cycles mean | `round(device pipeline ticks / repetitions)`: ARC running-state ticks over `repetitions` bodies plus one barrier, host transfers excluded (`extra` keeps host microseconds separately); default `repetitions=256` | same |
| Evidence | Executed on the `apu-00` card of `zhang-x4` (FW V1.24.1) on 2026-11-02 at `ffb4a78` + the wave 2 tree: typed add/mul/div/block_sum/dot and the u16 add runner, all exact against the runtime oracle (`VALIDATION.md`, Round 2). No hardware test in `tests/` yet (G2-04); virtual route and a fake runtime are tested (`tests/perf/test_apu_g2_typed_program.py`). | Executed on the same card and date: the five paper programs (GEMM, batched GEMM, GESUMMV, BiCG, MVT) with fresh random inputs, exact, cycles within 1.3% of the paper medians (`VALIDATION.md`). No hardware test in `tests/` yet (G2-04). |

Source workloads on `apu_v2` raise `NotImplementedError` (the old matcher
route reported success without compiling or executing and was removed,
G2-01). The SDK root and device are routed through `spmw_simenv` but only the
defaults `/opt/gsi` and `apu-00` are accepted until the templates are
parameterized (`INSTALL.md` section 3).

## Cross-cutting limits

- No access-scope/operand legality pass: a hand-supplied `layout=` that
  violates operand placement can be accepted and priced (C-01).
- No floating-point reduction-order permission: reduction-splitting
  placements are chosen by cost alone (C-09).
- Ranking is per kernel group and precedes materialization (C-03, C-05); ties
  are broken by enumeration position (C-06).
- The minimum-movement layout search of the paper's Section IV is not in the
  package (C-11).
- Shared frontend: annotation shape names are resolved from the caller's
  frames (F-01) and wide integer literals are truncated to i32 (F-02); both
  affect every backend.
