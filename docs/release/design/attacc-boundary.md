# AttAcc release boundary (recommendation)

Status: recommendation, not implemented (2026-11-02). Addresses release audit
`docs/release/audit/aim-attacc.md` M-04 and `CLAIMS.md` row E16. This file is a
copy of the recommendation sections of the containing repository's design note
`design_doc/compiler/attacc-release-boundary.md`, which also holds the claim
classification, the archive inventory and the probe table. Paths starting with
`allo/`, `tests/`, `benchmarks/`, `setup.py` are relative to this repository;
`TASK.md`, `paper/`, `tenon-artifacts/` and `reviews/` are in the containing
repository. Probes: `/tmp/tenon-release/probes/attacc/` (P1 to P4; no file in
this repository was edited).

## Probe summary

Release line numbers refer to the working tree on 2026-11-02 (HEAD `6efaca5` plus uncommitted edits by concurrent release tasks).

- P1: a DCC-free re-implementation of the tiling space, command counts and
  closed-form cost equals DCC's code on all 500 paper instances (668,575
  candidates, 0 count or cost mismatches) and selects the paper manifest tiling
  on 500/500 (first-minimum and lexicographic tie-break alike).
- P2: with three in-process seam changes (`_BACKENDS["attacc"]` with runner
  `None`, `"attacc"` in `_SINGLE_ITEM_LAUNCH_BACKENDS`, `"attacc_tiling"` in
  `_MATCHER_PHYSICAL_EXTRA_FIELDS`), public `allo.compile(..., backend="virtual")`
  on an `@allo` GEMV source selects the paper tiling for all four GEMV
  instances (7,040 / 25,600 / 2,432 / 8,192 cycles). Without the allowlist entry
  3,043 candidates collapse to 1 decision key; on a target outside `_BACKENDS`
  the virtual route never ranks (`spmw_codegen.py:1604-1608`).
- P3: a fused attention source (QK^T, softmax, PV) matches with complete
  coverage only without max-stabilization (`spmw_match_engine.py:2534` accepts
  only literal-0 accumulator initializers).
- P4: Ramulator2-AttAcc (sg2682's DCC build; needs GLIBC 2.34, runs here in the
  `aim-simulator-build` image) gives 11,454 cycles for the selected qkv nh8
  tiling, equal to the archived timing; the trace's opcode counts equal the
  symbolic counts of P2.

## Decision

| Criterion | A: evaluation-only | B: virtual backend, external sim/e2e | Recommended mix |
|---|---|---|---|
| G1: target, cost rules, decision domain, ranking reachable through a release API | fails (`TASK.md:161-166`: private harness does not satisfy) | passes | passes |
| "Removing a core paper capability ... not acceptable" (`TASK.md:137-138`) | removes the selector | keeps it | keeps it |
| Grounded scope limit (`TASK.md:136-137`) | none for the selector | simulator and e2e external: paper says decode latency "combines Ramulator2-AttAcc PIM timings with DCC's analytical GPU roofline" (`5-evaluation.tex:199-200`) and the replay "is not used for tiling selection" (`:188-189`); technical: DCC unlicensed, binary in another user's tree, GLIBC 2.34 | same, plus scripts consume the release program |
| G8: distinguish analytical, simulator timing, composition | partial | analytical only | analytical in package; simulator timing and composition via scripts, labelled |
| No DCC code in the package | yes | yes (re-implementation) | yes |

**Decision: the mix.** The AttAcc target, tiling domain, command counts,
closed-form cost and knob are compiler mechanisms and go into the package as an
analytical-only backend (no runner). Ramulator2-AttAcc timing, the reference-best
search, the replay, accuracy and the e2e composition are evaluation, depend on
unlicensed DCC code or a simulator binary, and stay as documented scripts outside
the installed package that read the release's selected program.

## Concrete step list (recommended option)

1. **Target.** `allo/pim/targets.py::build_attacc_target()` (about 90 lines),
   name `"attacc"`. Device `hbm3_pim`; `@allo.unit(mapping={"channel": 16})`
   with `banks` (64 banks/channel = 2 pch x 2 rank x 4 bg x 4 bank, 2^14 rows,
   32 columns of 32 B), `gemv_buffer`, `softmax_buffer`;
   `@allo.unit(mapping={"pim": 64})` with register `acc`. Ops: `MAC`
   (`a + x * y`, accumulates), `SFM_EXP` (`exp(x)`), `SFM_SUM` (accumulates),
   `SFM_DIV` (`x / s`); `SFM_MAX` once step 10 lands. Moves: `WR_GB`, `MV_SB`,
   `MV_GB`; host `H2D` (scatter), `D2H` (gather). No timing. Provenance comments:
   `HBM3_8Gb_2R` preset, `gen_trace_attacc_bank.py` (MIT), `eval-platforms.tex:53-56`.
2. **Tiling domain.** New `allo/pim/attacc_tiling.py` (about 260 lines):
   `AttAccKernel(kind: Literal["gemv", "attention"], rows, reduction, outputs)`;
   `attacc_kernel_from_matches(matches) -> AttAccKernel` deriving axes by operand
   role (reduction = loop the MAC accumulator is carried over; outputs = index of
   the result store; rows = the rest), never by loop position (audit M-01);
   `AttAccTiling` (frozen, `dims: tuple[tuple[int, int, int], ...]`,
   `manifest()`, `aligned()`); `attacc_tiling_space(kernel, *, channels,
   units) -> list[AttAccTiling]` re-implementing DCC's 3D space from target
   geometry (citing Yang et al. 2026, no DCC code); `AttAccCounts` and
   `attacc_command_counts(kernel, tiling, geometry)`; `AttAccProgram(kernel,
   tiling, counts)` with `dcc_tiling() -> list[list[int]]`; and
   `rank_attacc_tilings(kernel, geometry, cost_fn)` for the timing harness.
   P1's lib is the reference implementation.
3. **Cost.** New `allo/pim/costs/attacc.py` (about 150 lines): `N_CCDAB = 6`,
   `N_CCDL = 4`, `N_BL = 2` with `HBM3-PIM.cpp` (`HBM3_5.2Gbps`) provenance;
   `gemv_cycles(counts)` (M8, `max(compute, movement)` per channel);
   `attention_cycles(counts)` (`serial_q_cf`, n_mac-quantized MAC + data-bus
   moves per channel); `attacc_kernel_cycles(program)`; `@cost(target="attacc")
   def attacc_cost(target)` with `@rule(target.op("MAC"))` pricing the kernel
   record once per bucket and zero-latency rules for `SFM_*`, `WR_GB`, `MV_SB`,
   `MV_GB`, `H2D`, `D2H`, documented as "priced inside the kernel record".
   One cost route: the rule and `rank_attacc_tilings` call the same function.
   Export in `allo/pim/costs/__init__.py`. Docstring must state the validity
   domain: MAPE 30.78% / 14.19%, Spearman 0.94 / 0.98, ranking only.
4. **Knob.** `allo/spmw_knobs.py`: `_attacc_tiling_candidates(ctx)` (the
   space for `attacc_kernel_from_matches(ctx.matches)`),
   `_attacc_tiling_emit(value, base, ctx)` writing `extra["attacc_tiling"]`,
   `register_knob("attacc", Knob("attacc_tiling", ...))` inside
   `register_default_knobs`. Not a schedule-search knob, so
   `SPMW_DISABLE_SCHEDULE_SEARCH` leaves it on.
5. **Enumerator and allowlist.** `allo/spmw_autoschedule.py`:
   `@register_enumerator("attacc") def _attacc_enumerate(target, matches)`
   building one base placement (roles to `banks`, `gemv_buffer`, `acc`;
   `extra["attacc_kernel"]`) and returning `cross_with_knobs(...)`; add
   `"attacc_tiling"` to `_MATCHER_PHYSICAL_EXTRA_FIELDS`.
6. **Codegen context.** New `allo/spmw_attacc.py` (about 120 lines):
   `AttAccCtx(CodegenContext)` with `emit_program` producing one
   `AttAccProgram` per kernel into `self.programs` and symbolic `cmds` lines
   (`"PIM_MAC_AB count=15360"`), `resolve_moves -> (None, None)`,
   `runtime_route -> ("attacc-analytical", ())`, and a legality check for
   user `layout=` tilings (coverage `a*b*s >= extent`, channel product <= 16,
   unit product <= 64; raise `InfeasibleSchedule`). In
   `allo/spmw_codegen.py`: `_BACKENDS["attacc"] = ("allo.spmw_attacc",
   "AttAccCtx", None)`, add `"attacc"` to `_SINGLE_ITEM_LAUNCH_BACKENDS`, and
   make the runner-`None` error at `:1136` say "attacc is analytical-only in
   this release; use backend='virtual'; simulator timing:
   benchmarks/attacc/external/run_ramulator.py".
7. **Workloads.** `benchmarks/attacc/workloads.py` (not installed;
   `setup.py:204-205` packages only `allo`): `GPT13B_INSTANCES` (4 GEMV:
   (8, 5120, 384), (32, 5120, 384), (8, 128, 5120), (32, 128, 5120); attention:
   nhead 8 and 32, D = 128, L = 144..4096 step 16), `gemv_region(H, K, N)`,
   `attention_region(H, D, L)`.
8. **Evaluation scripts.** `benchmarks/attacc/external/` (not installed),
   rewritten from the archive with no absolute paths: `run_ramulator.py` (takes
   `AttAccProgram.dcc_tiling()` from `allo.compile`, imports DCC's
   `gen_trace_AttAcc_*` from `$ATTACC_DCC_ROOT` at run time, writes the trace
   and `temp/log` in a private temp dir, runs `$ATTACC_RAMULATOR` natively or in
   a container, reports `kind: "simulator-timing-only"`); `compose_e2e.py`
   (DCC `inference/system.py`, `kind: "analytical-composition"`);
   `selector_timing.py` (times `rank_attacc_tilings`, the quantity in
   `tab:attacc-compile`); `accuracy.py` (release costs vs the 517 archived
   samples). The archive and crosswalk stay as provenance for frozen numbers.
9. **Tests.** `tests/pim/test_attacc_virtual.py` (about 200 lines): space sizes
   1,391 / 2,273 / 3,043 / 4,478; counts vs a frozen JSON fixture of DCC counts
   for sampled tilings (no DCC import at test time); public virtual compile picks
   the four manifest GEMV tilings with 2,432 / 8,192 / 7,040 / 25,600 cycles;
   attention spot checks including the tie shape and the worst-loss shape
   (nh 8, L 480, manifest `[[8,1,1],[2,4,16],[1,16,30]]`); decision keys equal
   space size; non-paper shapes (GEMV 4 x 1000 x 200, attention 2 x 64 x 100)
   compile and select a covering tiling; `backend=None` raises the actionable
   error; an illegal `layout=` tiling is rejected; attention source coverage is
   complete. A `paper_full` test runs the 500-instance selector and expects
   500/500.
10. **Matcher identity (follow-up, shared file).** Accept an accumulator
    initialized to the op's declared identity, for example
    `allo.op("SFM_MAX", ..., identity=float("-inf"))`, in
    `_accumulator_init` (`spmw_match_engine.py:2504-2540`), so the
    max-stabilized softmax source matches. Until then the attention source uses
    the unstabilized form (mathematically the same softmax, no arithmetic is
    executed on this backend) and the docs say so.
11. **Docs.** Split CLAIMS E16 into E16a (target + selector, implemented,
    `allo.compile(..., backend="virtual")`) and E16b (Ramulator timing,
    reference-best, accuracy, e2e: external evaluation, scripts above). Add an
    AttAcc row to CAPABILITIES: source input yes; analytical yes; simulator
    timing external script (Ramulator2-AttAcc, timing-only, no numerical
    output); hardware none (academic architecture).

## Effort

| Area | Files | Lines (est.) |
|---|---|---|
| Package core (target, tiling, cost, ctx) | `targets.py`, `attacc_tiling.py`, `costs/attacc.py`, `costs/__init__.py`, `spmw_attacc.py` | about 620 |
| Shared-file deltas (architect review) | `spmw_knobs.py`, `spmw_autoschedule.py`, `spmw_codegen.py` | about 85 |
| Workloads + evaluation scripts | `benchmarks/attacc/workloads.py`, `benchmarks/attacc/external/*.py` (4) | about 490 |
| Tests + fixture | `tests/pim/test_attacc_virtual.py`, counts fixture | about 200 + data |
| Docs | CLAIMS, CAPABILITIES, simulators.md, release design copy | about 150 |
| **Total** | 13 files | about 1,550; about 5 engineer-days (matcher identity extension extra) |

## What stays evaluation-only, and why

| Item | Reason |
|---|---|
| Ramulator2-AttAcc timing (incl. 398/500, 123/316/61, loss replays) | external simulator; trace emission is DCC code with no license; binary lives in another user's tree and needs GLIBC 2.34 |
| Reference-best search | simulator sweeps with an assumed error-bound stopping rule (`5-evaluation.tex:182-183`, `6-discussion.tex:7-9`) |
| Command-replay predictor | the paper excludes it from selection (`5-evaluation.tex:188-189`); depends on DCC `match_alignment` |
| Accuracy (Spearman / MAPE) | needs simulator samples; computed from release costs and archived actuals |
| GPT-13B e2e decode | DCC `inference/system.py` and an analytical A100 roofline; not a PIM backend |
| DCC baselines (XGBoost, 6 h sweep) | competitor artifact |
| Compile-time table | a measurement of the selector; `allo.compile` adds per-candidate graph construction (about 0.66 ms per candidate in P2, about 7 to 8 min over 668,575 candidates) and must not be presented as the 24 s figure |

## Recommendation and open questions

**Recommendation.** Use the mix above. Land steps 1 to 7 and 9 as the
`attacc` analytical backend. The acceptance test is
`allo.compile(gemv_region(8, 5120, 384), build_attacc_target(), attacc_cost,
backend="virtual")`, which must select `[[1,8,1],[8,8,80],[2,1,192]]` at 7,040
cycles, together with the `paper_full` selector check at 500/500 and the
non-paper-shape tests. Land step 8 so the G8 evidence run takes the selected
program from the public API through Ramulator2-AttAcc. P4 shows this works here
via the `aim-simulator-build` image. Step 10 is a follow-up for the architect.

Open:
1. **DCC license.** DCC has no license. The port re-implements the tiling-space
   definition from the DCC paper and our geometry and vendors nothing. Should
   the user ask the DCC authors for permission, or for a license, before
   release? The external scripts import DCC only from a user-supplied checkout.
2. **Workload LoC 42/28.** No Tenon AttAcc workload source was found in the
   archive or the review tree. After step 7, count the release sources and
   record any difference from `tab:attacc-compile`.
3. **Attention tie.** P2 checked the release tie-break only on the GEMV ties.
   The one attention tie shape should be pinned in a step 9 test.
4. **Self-contained simulator.** A rebuild recipe from the MIT AttAcc simulator,
   or from DCC's `Ramulator2/build.sh` in a container, would remove the
   dependence on sg2682's binary.
5. **In-package runner.** Should `_run_attacc` become an in-package
   timing-only runner, as AiM has? It would import unlicensed DCC code at run
   time and mutate DCC module globals, so this note keeps it external.

