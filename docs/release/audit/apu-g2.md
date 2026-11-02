# Release audit: GSI APU v2 / Gemini-II (G2) backend

- Candidate: `experiments/allo` branch `tenon` @ `8160d6d82c28211900e156fcff6e0b452e94131d`
- Audit date: 2026-11-01. Read-only; no tracked file edited, no commit, no ssh.
- Evidence labels: **executed (compile-only)** = run on this server with the
  documented environment, no hardware; **read** = source inspection;
  **suspected** = inferred, needs remote confirmation.
- Probes and raw outputs: `/tmp/tenon-release/probes/apug2/probe_0{1,1b,2,3,4,5}*.py|.out`.
- Local facts (executed): Docker image `gsi-g2-l1sim` is present
  (`sha256:74beb8db...`); `/opt/gsi` exists but is the Gemini-I install (none of
  the four G2 SDK paths exist); `gsi_tool` on PATH is a G2 binary that fails on
  this host's glibc. `tests/perf/test_apu_g2_*.py`: 13 passed in 3.41 s.

## 1. Routes and entry points

All three routes share `target.name == "apu_v2"` (`allo/pim/targets.py:1154`,
`:1167`; `build_apu_g2_target` is an alias, `:1408`) and the backend tag
`RunResult.backend == "apu_v2"`.

### Route (a): matcher route, `@allo` source -> `APUv2Ctx` -> `_run_apu_v2`

| Aspect | Finding |
|---|---|
| Input | Ordinary `@allo` source (`allo.compile(fn, build_apu_g2_target(), apu_g2_cost)` -> `compiler.py:710` `match_workload` -> `compile_for_target`). |
| Emitted | GTML C++ text lines (`allo/spmw_apu_v2.py` `APUv2Ctx.cmd/emit_mac_matmul`, wrapped by `get_program_src`). **Executed:** for a uint16 and a float16 GEMV source the trace has 0 matches, `cmds == []`, and `program_src` is an empty `main()` (`probe_01b.out`). Every op in `build_apu_v2_target` is `matchable=False` (35 occurrences, `targets.py:1154-1405`), so the matcher can never bind anything on this target. |
| Build/launch | None. `_run_apu_v2` (`spmw_apu_v2.py:112-132`) probes `docker image inspect gsi-g2-l1sim` (`:121`), then returns `RunResult(cycles=None, stdout="... functional-only ...", extra={"program_src": ...})` (`:127`). Its own comment: "we don't bother compiling here". |
| Readback | None. **Executed:** `y` passed as output stays at its initial value (`[7]`), `estimate().cycles == 0`, no exception (`probe_01b.out`). |
| Cycles | Never measured. |
| Failure mode | Only `SimulatorUnavailable` when the Docker image is absent. On this host the image exists, so `simulator_unavailable_reason("apu_v2") is None` (`spmw_codegen.py:829`) and the route reports apparent success. |
| A06 | **Confirmed and worse than recorded**: the runner never executes, and on the release candidate it also emits nothing for ordinary source, yet returns a success-shaped `RunResult`. See G2-01. |

### Route (b): `APUG2TypedProgram` typed frontend

| Aspect | Finding |
|---|---|
| Input | Typed op record: `APUG2TypedProgram(operation in {add,mul,div,block_sum,dot}, input_types, output_type, shape, reduction_extent, repetitions)` (`apu_g2_typed_program.py:202-292`); routed by `compiler.py:602-624` (`_PROGRAM_ROUTES`), `host_moves`/`layout` rejected. |
| Legality | `_validate_type_legality` (`:293-366`): exact output width rule (`out_bits == lhs+rhs(+log K)`), shared signedness, unsigned-only div/block_sum, 24-bit segment cap, K power of two in [2, 4096] (`:192`). |
| Layout | Fixed: operands flattened row-major into a `(4, 65536)` carrier (`apu_g2_typed_runtime.py:45`), fixed L1 slot rows 64/192/320/448 (`:40-48`), GTML temp rows 2800/2928 (`:50-53`). No `LinearLayout`, no candidates. |
| Emitted | Nothing generated per program. A fixed project template `allo/pim/templates/apu_g2/` (17 device `.cc`, 18 host `.cc`) is copied verbatim (`apu_g2_typed_runtime.py:226` `shutil.copytree`); the program is passed as 12 argv values to `tenon_apu_g2_typed` (`host_typed.cc:168`), which forwards a 60-byte `ApuG2TypedParams` to the ARC task `tenon_apu_g2_typed` (`device/apu_g2_typed.cc:197`). Device-side `operation_legal` re-checks widths. |
| Build/launch | Per call, in a fresh tempdir: `cmake -S project/device -B build/device` + `cmake --build` (ARC, `CMAKE_BUILD_TYPE=ArcDebug`, `-O3 -DNDEBUG`, `device/CMakeLists.txt:9,73`) -> `tenon_apu_g2_tasks.update.bin`; then host `cmake -DGSI_TARGET=DEVICE -DBOARD_DEVICE_LIB=... -DCMAKE_BUILD_TYPE=Release` + `cmake --build` **without `--target`** (`apu_g2_typed_runtime.py:266`, builds all 18 host executables); then runs the host binary. Precondition gate `_require_hardware_stack` (`apu_g2_runtime.py:182-213`): four absolute `/opt/gsi/...` files, `gsi_tool` on PATH, `gsi_tool info apu-00 -v` must print `Status : Available` and `Core 0 is idle`. |
| Readback | Host zeroes the output pack before upload (`host_typed.cc:220`), reads back the full `(4,65536)` carrier, canonicalizes each lane to `out_bits` (`apu_g2_typed_bits.h:13`), writes `out.bin`; Python checks exact element count, gathers heads (`:55`), computes an int64 NumPy oracle (`:81`), and raises on any mismatch. Output arrays are the device values, never the oracle. |
| Cycles | `gsi_perf_start/end` around `repetitions` bodies + one `seu_barrier`, then one independent final body (`device/apu_g2_typed.cc` `measure`). `cycles = round(pipeline_ticks / repetitions)`. Default `repetitions=256`; the paper's "batches of 20" is the caller's choice. Host `h2d/host_task/d2h/end_to_end_us` kept separate. |
| Failures | Explicit `RuntimeError`/`ValueError` for missing SDK, card not Available, build failure, missing attestation fields, wrong output size, mismatch. `backend="virtual"` returns `RunResult(backend="virtual", outputs={})`. |
| Tests | Virtual-only plus a monkeypatched fake runtime (`tests/perf/test_apu_g2_typed_program.py:100-140`). No hardware test exists in `tests/`. |

### Route (c): `APUG2ComposedContractionProgram` (LinearLayout + composed runtime)

| Aspect | Finding |
|---|---|
| Input | Typed record: two `APUG2ScalarType`s, `output_type`, `output_shape`, power-of-two `reduction_extent`, `APUG2DotEpilogue` (identity / two-coefficient pair-affine / same-width accumulator), `repetitions` (`apu_g2_composed_contraction.py:151-262`). Operands are already **independent dot rows** `(dot_count, K)`; the caller must pre-pack GEMM/GEMV/batched/transposed/interleaved streams with host helpers (`apu_g2_composed_layout.py:207-318`). |
| Layout | Deterministic `APUG2DotTilePlan(dot_count, K)` (`apu_g2_layout.py:322`): `set=row&3, group=(row>>2)&15, slot=row>>6, column=group*4096+slot*K_pad+k`, expressed as a `LinearLayout` over F2; capacity `4*65536/K_pad` rows (`:355`), `K_pad <= 256` (`:353`). No temporal tiling, no alternative candidate. `layout_manifest()` carries a SHA-256 fingerprint. |
| Emitted | Recipe only (`build_apu_g2_composed_recipe`, `apu_g2_composed_program.py:62`: 8/12/16 ops). Device/host code is again the fixed template; the program travels as 17 argv values incl. the layout fingerprint (`host_composed_dot.cc:172`), to task `tenon_apu_g2_composed_dot` (`device/apu_g2_composed_dot.cc:197`) with a device-side `legal()` (`:35-60`). The host only checks the fingerprint is 64 hex digits (`host_composed_dot.cc:66`) and echoes it; nothing physical depends on it. |
| Build/launch | Same as (b) but host build uses `--target tenon_apu_g2_composed_dot` (`apu_g2_composed_runtime.py:213`). |
| Readback | Full carrier -> `gather_apu_g2_composed_output` reads lane `(set, column)` of row `terms_per_output*i` (`apu_g2_composed_layout.py:156`), compares to int64 oracle (`:179`, runtime `:137,:268`), device values returned. |
| Cycles | Same timer protocol; `cycles = round(pipeline_ticks / repetitions)`. |
| Failures | As (b); also attests `layout_fingerprint`, epilogue mode, all widths, `repetitions` (`apu_g2_composed_runtime.py:56-120`). |
| Tests | Virtual-only; `_parse_metrics` tested on synthetic text (`tests/perf/test_apu_g2_composed_contraction.py:243-307`). No hardware test. |

### Historical uint16 add route

`run_apu_g2_u16_add` (`apu_g2_runtime.py:317`) is a hardware runner for one
fixed `(4,65536)` uint16 add; exported, not routed through `allo.compile`.

### Orphaned machinery (read)

`APUG2ReductionPlan`, `APUG2DotTiling` (`apu_g2_layout.py:61,447`), the 1,400
lines of uint16 recipes in `apu_g2_recipe.py:980-2368` (dot tile, select,
minmax, div, mul, fill, block_sum, shift, sub, sqrt, contraction, chain), the 20
`APUG2_DOT_TILE_MEASURED_POINTS` (`:553`), `calibrate_apu_g2_recipe_from_measured_tile`
(`:726`), the three normalized GEMV/GESUMMV/ATAX cost graphs (`costs/apu_g2.py:709-980`),
the wall model (`:275`) and the 14 uint16 host templates + 15 uint16 device
sources are not reachable from any `allo.compile` route or test except as
exports. `grep` finds no caller outside their defining modules. The design docs
cite `apu_g2_program.py` / `APUG2Program(operation="gesummv_u16")`, which does
not exist in the candidate.

## 2. Enforced vs advertised domain

| Claim source | Advertised | Enforced in candidate |
|---|---|---|
| Paper Table `eval-apuv2.tex`: five kernels, "median batch-amortized ARC running-state cycles, batches of 20, transfers excluded" | Tenon compiles the kernels and is 1.07-1.67x faster than GTML | Route (c) can express exactly these five programs (executed: all five constructors accept, `probe_02.out`). Timer protocol in the device template matches the description (repetitions bodies, one barrier, transfers outside). `repetitions` default is 256; 20 must be set by the caller. |
| Paper 5-evaluation.tex: "graph-level layout, fusion, and residency choices" for GESUMMV | choices made by the compiler | **Read:** no choice exists. GESUMMV = one `APUG2ComposedContractionProgram` with `APUG2DotEpilogue((3,2))` whose layout, interleaving (`interleave_apu_g2_dot_streams`), fusion (pair-affine epilogue) and residency are fixed by the single plan and by the caller's packing in the artifact adapter (`tenon-artifacts/apuv2-composed-v2/provenance/implementations/tenon/adapter.py:84-98`). Nothing is ranked or searched (`probe_05.out`: `layout_plan` is one `APUG2DotTilePlan`; `allo.compile(..., layout=...)` is rejected). |
| Paper "Tenon LoC 33-46" per kernel | standalone Tenon workloads of 33-46 lines | The repository contains no per-kernel Tenon workload sources for these five cases. The only Tenon-side source is the campaign adapter's per-case `if case == ...` branch (adapter.py:70-134, ~8-15 lines each) plus packing helpers. Counting basis not recoverable from the candidate (open question OQ-2). |
| `compile()` docstring (`compiler.py:647-660`), `SPMW_ARCHITECTURE.md` | "Typed program frontends ... compile through their own route"; a plain `@allo` workload goes to the matcher | `SPMW_ARCHITECTURE.md` and `README.md` never mention Gemini-II / `apu_v2` / typed G2 programs (`grep` empty). The matcher route for `apu_v2` is advertised by `_BACKENDS` (`spmw_codegen.py:1045`) and by `simulator_unavailable_reason` but cannot bind any op. |
| Design docs (`design_doc/targets/apu-g2-vl64-backend.md`, `apu-g2-polybench-completion.md`) | ordinary Allo GEMV/GESUMMV/ATAX/GEMM/... compile from retained MLIR to VL64; 15 PolyBench kernels on card | None of that source path exists in the candidate (no `APUG2Module`, no retained-MLIR analyzer for `apu_v2`). Only typed routes (b)/(c) remain. |
| Skill `pim-apu-v2/SKILL.md` | "L1 functional simulator only", `caps["perf_is_placeholder"]`, run path `spmw_codegen._run_apu_v2` | Hardware routes exist; `_run_apu_v2` moved to `spmw_apu_v2.py`; the skill is stale. |
| Typed route (b) domain | "generic direct-VL64 integer program" | Shapes up to 262,144 elements, 1-24 bit, K in [2,4096]; ADD/MUL/DIV/BLOCK_SUM/DOT only; exact output width required (I16 output for an I4xI4 K128 dot is rejected: "must be exactly 15 bits", `probe_02.out`). |
| Composed route (c) domain | "arbitrary independent dot products ... common carrier for GEMM, batched, Gram, triangular, GEMV" | One carrier only: `padded(dot_count) <= 4*65536/K_pad`. Executed: GEMM INT4 64x128x31 accepted, 64x128x33 rejected ("padded output extent 4096 exceeds capacity 2048"), 65x128x31 accepted, K=96 rejected (power of two), K=512 rejected, GESUMMV M=1025 rejected, BiCG N=513 rejected (`probe_02.out`). Larger problems are not tiled although `APUG2DotTiling` exists. |

## 3. Findings

Severity: **blocker** = violates a gate on a public path; **major** = gate
evidence missing or misleading; **minor** = hygiene/documentation.

### G2-01 (blocker) Matcher route returns success for ordinary source without compiling, emitting, or executing

- Files: `allo/spmw_apu_v2.py:112-132`; `allo/spmw_codegen.py:829,1045`;
  `allo/pim/targets.py:1154-1405` (all ops `matchable=False`); `allo/spmw_autoschedule.py:1101`.
- What the code does: `allo.compile(gemv_source, build_apu_g2_target(), apu_g2_cost)`
  matches nothing, emits an empty GTML `main()`, and `compiled(W, x, y)` returns
  `RunResult(cycles=None, backend="apu_v2", extra={"program_src": ""})` while
  the output array is untouched and `estimate().cycles == 0`.
- Gates: G2 (missing-output success; emitted-only indistinguishable from
  executed), G8 (advertised execution mode with no evidence), G1 (public API
  implies source compilation for G2 that does not exist).
- Reproduction (executed, compile-only): `probe_01_matcher_route.py`,
  `probe_01b_matcher_detail.py`. Observed: `run() returned: RunResult cycles=
  None backend= apu_v2`, `inner.cmds: 0 []`, `inner.trace matches: []`,
  `y after run ... [7]`, `program_src bytes: 0`.
- Repair: remove the `apu_v2` matcher runner, enumerator and `_BACKENDS` entry
  (there is no l1_sim compile path and no card on a Docker host), and make
  `allo.compile` raise a finite error for source workloads on `apu_v2` that
  names the supported typed routes; drop the Docker gate from
  `simulator_unavailable_reason("apu_v2")`. If an emitted-only artifact is
  ever wanted, it must be a distinct type (not `RunResult`) and must carry the
  actual commands. Update `tests/perf/test_simulator_unavailable.py:55` and
  `tests/pim/conftest.py:95` accordingly.

### G2-02 (blocker) No public source route to Gemini-II; paper kernels are hand-packed typed op records

- Files: `allo/compiler.py:602-624`; `tenon-artifacts/apuv2-composed-v2/provenance/implementations/tenon/adapter.py:70-134`.
- What the code does: the five paper kernels were produced by route (c) with
  the adapter constructing `APUG2ComposedContractionProgram(...)` per case
  name and pre-packing operands (`pack_apu_g2_batched_gemm_dots`,
  `pack_apu_g2_matrix_vector_dots`, `interleave_apu_g2_dot_streams`,
  `np.stack((x1,x2),axis=1)` accumulator, `transpose_output` on the host). The
  program/layout/recipe fingerprints are then asserted against archived
  manifests (adapter.py:179-191). Which stream goes to which MMB set, the
  coefficient pairing, and the transposed access are all decided by the
  adapter, not by the compiler.
- Gates: G1 (paper claims "the same workload ... retargeted", "graph-level
  layout, fusion, residency choices"), G3 (`if case == "gesummv_i2_m1024_k128"`
  dispatch in the only Tenon-side driver), G4 (no search, no candidates).
- Reproduction (read): adapter source above; (executed) `probe_05.out`
  `layout_plan is a single deterministic APUG2DotTilePlan`.
- Repair: either (i) implement an `apu_v2` contraction frontend over retained
  MLIR that lowers GEMM/GEMV/two-stream/accumulate graphs to
  `APUG2ComposedContractionProgram` plus the packing/gather helpers (the
  helpers already exist; the missing piece is the analyzer and a documented
  epilogue recognizer), and ship the five paper kernels as ordinary `@allo`
  examples; or (ii) document, in `SPMW_ARCHITECTURE.md`/`CLAIMS.md`, that
  Gemini-II is typed-only, and move the five programs into
  `examples/` as public, non-case-switched Python. (ii) alone does not satisfy
  the paper's "same workload" claim; (i) is required for G1.

### G2-03 (major) Typed/composed cost estimate is an op count presented as cycles; no coefficient reaches it

- Files: `allo/pim/apu_g2_recipe.py:842-930` (`cycles = 1 if latency is None`,
  `:888`); `allo/pim/apu_g2_composed_program.py:301-324` (`graph = build_apu_g2_recipe_graph(recipe, target)` with no calibration, `"analytical": True`);
  `allo/pim/apu_g2_typed_program.py:664-688`; virtual `RunResult(cycles=int(self.estimate_result.cycles))` at `apu_g2_composed_program.py:410-412` and `apu_g2_typed_program.py:796-806`.
- What the code does: every recipe call costs 1 "cycle"; estimates for the five
  paper programs are 8/8/16/8/12 and do not change with shape (GEMM (2048,1)
  and (1,) K=2 both 8) or with a different cost spec (perturbed fingerprint ->
  identical 16). None of the `apu_g2_cost` rules (`ADD_U16`, `MUL_U8_TO_U16`,
  `GROUP_REDUCE_ADD_U16_TO_U23`, `SHIFT_LEFT_U16`, `SEU_BARRIER`, squeeze,
  spread) covers the typed opcodes (`MUL_TYPED`, `ADD_TYPED`,
  `GROUP_REDUCE_ADD_TYPED`, `COPY_*`, `COPY_ODD_TO_EVEN_VECTORS`), so the
  calibrated constants in `costs/apu_g2.py` and `apu_g2_pipeline.py` are
  unreachable from both public G2 routes. The archived artifact confirms the
  paper campaign compiled with `backend='virtual'` and recorded
  `"cycles": 16` for GESUMMV (`provenance/.../cases/gesummv_i2_m1024_k128/operations.json`).
- Gates: G5 (model without validated domain advertises predictions;
  "changing a relevant cost changes estimates" fails), G4 (no ranking).
- Reproduction (executed): `probe_02.out` estimate rows; `probe_05.out`
  `base cycles: 16 | perturbed-fingerprint cost cycles: 16`, `latencies: [1]`,
  `recipe_calibration: None`.
- Repair: either attach a real calibration to the typed recipes (there are
  real-card medians for exactly these five programs in the artifact, and the
  `APUG2RecipeCalibration` plumbing already accepts one) with a declared
  validity domain and an "interpolated" flag, or stop reporting `cycles` for
  virtual typed runs (return `cycles=None` and a `latency_unit` field). Remove
  `"analytical": True` where the unit is a structural issue count.

### G2-04 (major) No hardware evidence in the candidate's test suite; hardware routes untested against a card since the artifact commit

- Files: `tests/perf/test_apu_g2_typed_program.py:100-140` (fake runtime),
  `tests/perf/test_apu_g2_composed_contraction.py` (virtual only), no test
  calls `run_apu_g2_typed`/`run_apu_g2_composed`/`run_apu_g2_u16_add`.
- Gate: G8 (hardware capability needs hardware evidence at the candidate
  revision). The artifact ran `allo_commit 29b69d96`, not `8160d6d`.
- Repair: add opt-in hardware tests (skip with a named reason when
  `_require_hardware_stack()` fails) for the add runner, one typed program per
  operation, and the five composed programs; run them on `zhang-x4` (Section 5)
  and record ticks and binary hashes in `VALIDATION.md`.

### G2-05 (major) Composed route supports one carrier only; tiling exists but is disconnected

- Files: `apu_g2_composed_contraction.py:234` (`APUG2DotTilePlan(self.dot_count, K)`),
  `apu_g2_layout.py:355-362`, `:447` (`APUG2DotTiling` unused).
- What the code does: GEMM INT4 64x128x33 (2112 dots) is rejected with a
  capacity error while 64x128x31 passes; GESUMMV M=1025, BiCG N=513 likewise.
- Gate: G3/G6 (nearby legal shapes in the advertised family fail; the
  rejection is explicit, so this is not silent wrong code, but the "GEMM,
  batched GEMM" family claim is one-tile only).
- Reproduction (executed): `probe_02.out` composed neighbors.
- Repair: drive `APUG2DotTiling` from the composed program (one task per
  output tile and per 256-wide reduction chunk with accumulate epilogue), or
  document the single-carrier domain in the public API and the claim matrix.

### G2-06 (major) Hardware discovery is hard-coded and there is no device lock

- Files: `allo/pim/apu_g2_runtime.py:36-41` (four `/opt/gsi/...` absolute
  paths), `:188-213` (`gsi_tool` must be on PATH; device literal `"apu-00"`;
  regexes `^Status\s*:\s*(.+?)\s*$`, `^Core 0 is (.+?)\s*$`); template
  CMakeLists have 116 `/opt/gsi` occurrences (`probe_04.out`); no `flock`
  anywhere in the G2 runtime; `tests/pim/.apu_g2.lock` referenced by
  `FRESH_REPRODUCTION.md` does not exist (only `.apu_v1.lock`).
- What the code does: the only mutual exclusion is a `gsi_tool` status
  read before building (minutes of cmake/ARC compile) and launching, which is
  a time-of-check/time-of-use race between two users of `apu-00`.
- Gates: G7 (configurable discovery), G8 ("exclusive device locks shared by
  processes on the device host").
- Repair: environment/config override for SDK root, `gsi_tool` path and
  device name (`TENON_APU_G2_SDK_ROOT`, `TENON_APU_G2_DEVICE`), pass them to
  CMake as cache variables, and take an `fcntl.flock` on a documented lock
  file (default `tests/pim/.apu_g2.lock` or `/var/lock/tenon-apu-g2.lock`)
  around build+launch.

### G2-07 (major) Backend tag and result type do not distinguish no-op, virtual, and hardware results

- Files: `allo/spmw_apu_v2.py:127-132` (`backend="apu_v2"`, `cycles=None`),
  `apu_g2_typed_runtime.py:343` and `apu_g2_composed_runtime.py:284`
  (`backend="apu_v2"`), `spmw_codegen.py:795-806` (docstring says `cycles is
  None` means "executed on a functional-only backend").
- Gate: G2 ("virtual results and emitted-only artifacts are distinguishable
  from executed results").
- Repair: part of G2-01; additionally add an explicit `execution` field
  (`"hardware" | "virtual" | "emitted_only"`) to `RunResult.extra` for all G2
  runners (the hardware ones already print `target=hardware`).

### G2-08 (minor) Public layout fingerprint is attested but not bound to anything physical

- Files: `host_composed_dot.cc:66-74,316`; `apu_g2_composed_runtime.py:73`.
- What the code does: the host validates that argv[17] is 64 hex digits and
  echoes it; the runtime compares the echo with the program's fingerprint. The
  device packing is done in Python; the host and device never derive anything
  from the fingerprint. It is provenance, not a check that the executed layout
  equals the selected layout.
- Gate: G4 ("the chosen plan survives through packing ... and actual
  dispatch") is satisfied only because there is exactly one plan; document
  this honestly in the manifest (`"attestation": "echo"`).

### G2-09 (minor) Fixture decoding and host readback wrap silently instead of rejecting out-of-width payloads

- Files: `apu_g2_composed_layout.py:28-62` (`canonicalize` = `mod 2^bits`,
  `decode_apu_g2_u16_bitpatterns`), `apu_g2_typed_bits.h:13-20`.
- Executed: `decode([0xFFFF,0x8003,0x0004], I2) -> [-1, -1, 0]`;
  `canonicalize([16384], I15) -> -16384` (`probe_03.out`). Range checks on
  packed operands and accumulators do reject out-of-range logical values, and
  oracle widths are exact by construction (max |dot| 8192 < 16384 for the
  paper GEMM), so no overflow can occur on legal inputs; the wrap only hides
  corrupt fixture bits.
- Repair: make `decode_apu_g2_u16_bitpatterns` reject payloads with bits above
  `scalar_type.bits` unless a `strict=False` flag is passed.

### G2-10 (minor, suspected) Partial-shape typed DIV divides zero by zero in padded lanes

- Files: `apu_g2_typed_runtime.py:45-52` (zero-fill), `:207` (only logical
  divisors checked), `host_typed.cc` (no padding fill).
- Typed ADD/MUL/DIV accept shapes smaller than the carrier (executed: shape
  `(3,100)` accepted). For DIV the remaining 261,844 lanes hold divisor 0 and
  the VL64 `div` runs on them. The lanes are discarded, but hardware behavior
  of `div` by zero (trap, hang, or garbage) is unverified in the candidate; all
  historical div gates used full packs.
- Repair: fill padded divisor lanes with 1 in `_pack_operand` for DIV, or
  restrict DIV to full carriers; confirm on card.

### G2-11 (minor) Dead templates and benchmark constants ship in the production payload

- Files: `allo/pim/templates/apu_g2/` (14 uint16 host programs, 15 uint16
  device sources, all compiled into every task library); `apu_g2_params.h:257-258`
  (`APUG2_U16_GEMM_CHUNK = 128`, `APUG2_U16_GEMM_BMAX = 31`); `costs/apu_g2.py:221-262`
  (wall model keyed to `gemm_m1000_k1200_n1100_b31`, `gemv_m1900_k2100_n1_b31`,
  `APUG2_GEMM_MAX_BATCH_COLUMNS = 31`); `apu_g2_recipe.py:553-630` (20
  measured points keyed by paper-era tile shapes); `apu_g2_pipeline.py:385-480`
  (GESUMMV N=90, ATAX 116x124 attributions).
- Impact: build time (the typed runtime compiles all 18 host executables),
  review burden, and benchmark identifiers in production modules. Note the
  coincidence `BMAX = 31` vs paper GEMM N=31: the paper shape sits exactly at
  the resident-accumulator limit of the removed persistent-GEMM path.
- Repair: delete or move to `examples/evaluation`; keep only
  `apu_g2_typed.cc`, `apu_g2_composed_dot.cc`, `apu_g2_u16_add.cc` (if the add
  runner stays) and their hosts; build only the needed host target.

### G2-12 (minor) Documentation drift

- `SPMW_ARCHITECTURE.md` and `README.md` do not mention Gemini-II at all;
  `design_doc/targets/apu-g2-*.md` describe source-driven and PolyBench
  machinery absent from the candidate; `.claude/skills/pim-apu-v2/SKILL.md`
  says l1_sim only and points at `spmw_codegen._run_apu_v2`;
  `apu_g2_typed_program.py:5` refers to a "legacy `APUG2Program`" that does
  not exist. `allo.pim` exports `APUG2ReductionPlan`, `APUG2DotTiling`,
  `build_apu_g2_u16_dot_tile_recipe` etc. without any consumer.
- Repair: a Gemini-II section in `SPMW_ARCHITECTURE.md` listing the typed
  routes, the supported domain table above, the hardware prerequisites, and
  the exact relation to the paper rows.

### Confirmed sound (read / executed)

- Typed and composed legality checks are structural (widths, signedness, K
  power of two, capacity) and reject neighbors with finite, actionable
  messages; names never affect fingerprints (`probe_02.out`: rename
  invariance True).
- Oracle is computed before launch from the caller's operands and compared
  after; device values are what is returned (`apu_g2_composed_runtime.py:137,268`;
  typed `:326`). No CPU substitution.
- Output carrier is zeroed before upload; accumulators are explicit inputs
  (ACCUMULATE epilogue) and nonzero accumulators are covered by the oracle
  (`probe_03.out`); truncated or misshapen readback raises.
- Pair-affine coefficient placement (coefficient 0 on even MMB sets,
  coefficient 1 on odd) agrees with the device `copy_from_odd_to_even_vectors`
  + `add` sequence and the gather of even rows (`probe_03.out`).
- Padded operand lanes are zero, so `sum` over `K_pad` cannot leak
  (`probe_03.out`: nonzero lanes == dot_count*K).
- Timer scope matches the paper's wording (repeated bodies + one barrier;
  transfers outside; device ticks separate from host microseconds).

## 4. Constants / specialization inventory

| Item | Location | Category (TASK.md) | Judgment |
|---|---|---|---|
| 16 groups, 4096 lanes/group, 65,536 columns, 4 MMB sets, 24-row segments, 3072 L1 rows | `apu_g2_layout.py:22-26`, `targets.py:1172-1216`, templates `kGroups/kVectors/...` | ISA / physical capacity | Acceptable; named and sourced. |
| L1 slot rows 64/192/320/448/576, GTML temp 2800+128, index 2928+16 | `apu_g2_typed_program.py:40-53`, `apu_g2_composed_program.py:32-38`, `host_*.cc` static_asserts | Allocation plan constants | Acceptable as a fixed allocator for one carrier; must be documented as the allocator policy; duplicated in Python and C++ (drift risk). |
| `K_pad <= 256`, capacity `4*65536/K_pad`, `log_reduction in [1,8]` | `apu_g2_layout.py:353-362`, device `legal()` | Derived from 24-bit `sum` envelope | Acceptable; derivation stated. |
| Typed `K in [2,4096]`, device `log_reduction <= 12` | `apu_g2_typed_program.py:192-199` | Derived (group width) | Acceptable. |
| `repetitions=256` default | typed/composed programs | Measurement policy | Acceptable; paper used 20; document. |
| Exact-output-width rule (`out_bits == lhs+rhs+logK[+coef+1]`) | `apu_g2_typed_program.py:293-366`, `apu_g2_composed_contraction.py:200-232` | Semantic contract | Acceptable but restrictive; document that wider containers are rejected by design. |
| `(3,2)` coefficients, `(64,31)`, `(4,32,16)`, `1024/128`, `256` | only in tests and the artifact adapter | Benchmark dimensions | Acceptable location (not production). |
| `APUG2_U16_GEMM_BMAX = 31`, `APUG2_U16_GEMM_CHUNK = 128` | `apu_g2_params.h:257-258`, `host_u16_gemm.cc` | Archived measured limit / benchmark-era constant | Not acceptable in production payload (G2-11). |
| `APUG2_GEMM_MAX_BATCH_COLUMNS = 31`, wall-model coefficients, anchors `gemm_m1000_k1200_n1100_b31` | `costs/apu_g2.py:221-262` | Measured model with provenance but no consumer | Move to evaluation or delete (G2-11). |
| 20 `APUG2_DOT_TILE_MEASURED_POINTS`, normalized 8,001/6,893/76,118-tick attributions | `apu_g2_recipe.py:553`, `apu_g2_pipeline.py:359-480` | Measured coefficients, provenance dated 2026-07-08, interpolation explicit | Acceptable provenance; unreachable from public routes (G2-03). |
| `"apu-00"`, `/opt/gsi/...`, `gsi_tool` on PATH | `apu_g2_runtime.py:36-41,188-213`, templates | Environment constants | Not acceptable as fixed (G2-06). |
| `matchable=False` on every target op | `targets.py:1154-1405` | Target legality | Honest (no matcher lowering exists) but makes route (a) a no-op (G2-01). |
| Pair-affine limited to 2 terms; accumulator exclusive with coefficients | `apu_g2_composed_contraction.py:82-110` | Finite epilogue catalog | Acceptable if labeled as the implemented dataflows (it is, in the docstring). |
| `if case == ...` program construction | artifact adapter only | Benchmark driver | Acceptable location, but it is the only Tenon-side "workload" (G2-02). |

## 5. Remote validation plan (zhang-x4; not executed here)

Preconditions to confirm on the remote host before anything else:

```sh
# identity
hostname; cat /etc/os-release | head -2; cmake --version | head -1; python3 --version
/opt/gsi/g2/bin/amd64/gsi_tool info apu-00 -v | sed -n '1,12p'   # expect "Status          : Available", "Core 0 is idle", FW GSI_G2_Infrastructure_V1.24.1
# SDK files the runtime requires verbatim (apu_g2_runtime.py:36-41) and CMake packages
ls -l /opt/gsi/include/64vl/g2_64vl/g2_64vl_types.h /opt/gsi/g2/vector_core/lib/libg2_64vl.a \
      /opt/gsi/g2/vector_core/bin/g2_md_update.py /opt/gsi/lib/libg2_device_core.a
ls -d /opt/gsi/lib/cmake/g2_device /opt/gsi/lib/cmake/g2_transport /opt/gsi/g2/Linux/lib/amd64
ls /opt/gsi/toolchain/vector_core/ 2>/dev/null | head
# checkout and revision
cd <remote>/spmw-for-pim/experiments/allo && git rev-parse HEAD && git status --short | head
# interpreter with numpy (artifact used /work/shared/users/phd/nz264/anaconda3/envs/allo-dev/bin/python, py3.12, numpy 2.4.3)
PY=<python>; $PY -c "import numpy, sys; print(sys.version, numpy.__version__)"
```

Environment expected by the runtime: `PATH` must include `/opt/gsi/g2/bin/amd64`
(for `shutil.which("gsi_tool")`), `PYTHONPATH` to the allo checkout (and MLIR
python package only if the source route is exercised), optional
`TENON_APU_G2_KEEP_TMP=1` (retain the generated project under
`/tmp/tenon-apu-g2-*`), `TENON_APU_G2_BUILD_JOBS=N`. Serialize with a shared
lock on the device host, e.g. `flock /tmp/tenon-apu-g2.lock <command>`
(create `tests/pim/.apu_g2.lock` if the historical path is preferred).

Step 1, software gates at the candidate revision:

```sh
$PY -m pytest -q tests/perf/test_apu_g2_typed_program.py tests/perf/test_apu_g2_composed_contraction.py tests/perf/test_apu_g2_target_layout.py
```

Step 2, hardware smoke (historical add runner, builds the full task library):

```sh
export PATH=/opt/gsi/g2/bin/amd64:$PATH TENON_APU_G2_KEEP_TMP=1
flock /tmp/tenon-apu-g2.lock $PY - <<'PYEOF'
import numpy as np
from allo.pim.apu_g2_runtime import run_apu_g2_u16_add
rng = np.random.default_rng(1)
lhs = rng.integers(0, 65536, size=(4, 65536), dtype=np.uint16)
rhs = rng.integers(0, 65536, size=(4, 65536), dtype=np.uint16)
r = run_apu_g2_u16_add(lhs, rhs, repetitions=20)
print(r.cycles, r.extra["raw_ticks"], r.extra["project"]["generated_project_path"])
PYEOF
```

Step 3, typed route, one program per operation through the public API
(expect exact NumPy equality enforced inside the runtime; record `cycles`,
`raw_ticks`, `host_timings_us`, the `.update.bin` and host binary SHA-256
from `extra["project"]`):

```sh
flock /tmp/tenon-apu-g2.lock $PY - <<'PYEOF'
import numpy as np, allo
from allo.pim.apu_g2_typed_program import APUG2ScalarType as T, APUG2TypedProgram
from allo.pim.targets import build_apu_g2_target; from allo.pim.costs.apu_g2 import apu_g2_cost
I4,I8,I15,I16,U8,U16 = T(4,True),T(8,True),T(15,True),T(16,True),T(8,False),T(16,False)
rng = np.random.default_rng(2); tgt = build_apu_g2_target()
cases = [
 (APUG2TypedProgram("add",(I16,I16),I16,(4,65536),repetitions=20), lambda: (rng.integers(-32768,32768,(4,65536),dtype=np.int16), rng.integers(-32768,32768,(4,65536),dtype=np.int16))),
 (APUG2TypedProgram("mul",(I8,I8),I16,(4,65536),repetitions=20),   lambda: (rng.integers(-128,128,(4,65536),dtype=np.int8), rng.integers(-128,128,(4,65536),dtype=np.int8))),
 (APUG2TypedProgram("div",(U16,U16),U16,(4,65536),repetitions=20), lambda: (rng.integers(0,65536,(4,65536),dtype=np.uint16), rng.integers(1,65536,(4,65536),dtype=np.uint16))),
 (APUG2TypedProgram("block_sum",(U8,),U16,(4,65536),256,repetitions=20), lambda: (rng.integers(0,256,(4,65536),dtype=np.uint8),)),
 (APUG2TypedProgram("dot",(I4,I4),I15,(1985,128),128,repetitions=20), lambda: (rng.integers(-8,8,(1985,128),dtype=np.int8), rng.integers(-8,8,(1985,128),dtype=np.int8))),
]
for prog, mk in cases:
    c = allo.compile(prog, tgt, apu_g2_cost)
    r = c(*mk())
    print(prog.name, "ticks/call", r.cycles, "final", r.extra["final_pipeline_ticks"], r.extra["project"]["source_sha256"].get("device/apu_g2_typed.cc"))
PYEOF
```

Step 4, composed route, the five paper programs with fresh random fixtures and
`repetitions=20` (bit-exact check is inside the runtime; compare `cycles`
with the paper medians 1,733 / 1,739 / 1,860 / 2,549 / 2,767 only as a sanity
range, not as a pass criterion):

```sh
flock /tmp/tenon-apu-g2.lock $PY - <<'PYEOF'
import numpy as np, allo
from allo.pim.apu_g2_typed_program import APUG2ScalarType as T
from allo.pim.apu_g2_composed_contraction import APUG2ComposedContractionProgram as C, APUG2DotEpilogue as E
from allo.pim.apu_g2_composed_layout import (pack_apu_g2_batched_gemm_dots as bg, pack_apu_g2_matrix_vector_dots as mv, interleave_apu_g2_dot_streams as il)
from allo.pim.targets import build_apu_g2_target; from allo.pim.costs.apu_g2 import apu_g2_cost
I2,I3,I4,I14,I15 = T(2,True),T(3,True),T(4,True),T(14,True),T(15,True)
rng = np.random.default_rng(3); tgt = build_apu_g2_target(); R = 20
a=rng.integers(-8,8,(64,128),np.int8); b=rng.integers(-8,8,(128,31),np.int8)
ab=rng.integers(-8,8,(4,32,128),np.int8); bb=rng.integers(-8,8,(4,128,16),np.int8)
ga=rng.integers(-2,2,(1024,128),np.int8); gb=rng.integers(-2,2,(1024,128),np.int8); gx=rng.integers(-2,2,(128,),np.int8)
m=rng.integers(-4,4,(256,256),np.int8); p=rng.integers(-4,4,(256,),np.int8); r=rng.integers(-4,4,(256,),np.int8)
x1=rng.integers(-8192,8192,(256,),np.int16); x2=rng.integers(-8192,8192,(256,),np.int16)
runs = [
 ("gemm",    C((I4,I4),I15,(64,31),128,repetitions=R),                       bg(a,b), None),
 ("bgemm",   C((I4,I4),I15,(4,32,16),128,repetitions=R),                     bg(ab,bb), None),
 ("gesummv", C((I2,I2),I15,(1024,),128,E((3,2)),repetitions=R),              il(mv(ga,gx),mv(gb,gx)), None),
 ("bicg",    C((I3,I3),I14,(256,2),256,repetitions=R),                       il(mv(m,p),mv(m,r,transpose=True)), None),
 ("mvt",     C((I3,I3),I15,(256,2),256,E((1,),I14),repetitions=R),           il(mv(m,p),mv(m,r,transpose=True)), np.stack((x1,x2),axis=1)),
]
for name, prog, (lhs, rhs), acc in runs:
    c = allo.compile(prog, tgt, apu_g2_cost)
    res = c(lhs, rhs, acc) if acc is not None else c(lhs, rhs)
    print(name, "ticks/call", res.cycles, "final", res.extra["final_pipeline_ticks"], res.extra["project"]["artifact_sha256"])
PYEOF
```

Step 5, negative controls on the card: (i) run step 4's GEMM with
`repetitions=1` and confirm `cycles == final_pipeline_ticks` order of
magnitude; (ii) feed a deliberately wrong oracle by calling
`run_apu_g2_composed` with operands whose packed file is patched (keep
`TENON_APU_G2_KEEP_TMP=1`, re-run the host binary by hand with a modified
`lhs.bin`) and confirm the Python gate raises; (iii) typed DIV with shape
`(3,100)` to settle G2-10; (iv) run the matcher route (`probe_01`) on the remote
to confirm it still returns the empty no-op (it must not be used as evidence).

Step 6, record for `VALIDATION.md`: host, commit, `gsi_tool` output, SDK file
hashes, generated project path, `.update.bin` and host SHA-256 (from
`extra["project"]`), inputs seed, `cycles`, `raw_ticks`, host microseconds,
and any failure text verbatim.

## 6. Open questions

- OQ-1: Is `apu_g2_u16_add` (route with 17-source task library) still wanted
  in the release, or should the task library be reduced to the typed and
  composed entry points?
- OQ-2: What exactly was counted for "Tenon LoC 33-46"? No per-kernel Tenon
  workload files exist in the candidate or the sealed artifact
  (`tenon/<case>/` holds only runs; `provenance/.../cases/<case>/` holds
  generated device/host source, layout and operations manifests).
- OQ-3: Can the five composed programs be derived from ordinary `@allo`
  GEMM/GEMV/GESUMMV/BiCG/MVT source at these shapes without benchmark-name
  dispatch? The packing helpers are name-free; the missing component is a
  retained-MLIR contraction analyzer for `apu_v2` plus an epilogue recognizer
  for `alpha*A@x + beta*B@x` and `x += A@y`.
- OQ-4: Does the installed firmware's `gsi_tool` still print `Status          :
  Available` and `Core 0 is idle` in the exact format the regexes expect
  (recorded post-campaign output does, `provenance/hardware-postflight.raw.txt`)?
- OQ-5: Is `div` by zero in padded lanes benign on silicon (G2-10)?
- OQ-6: Should a calibrated typed-recipe cost be added from the artifact's
  real-card medians (five exact points), or should the virtual G2 estimate be
  declared non-performance (cycles=None) for release?
