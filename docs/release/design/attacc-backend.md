<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# AttAcc analytical backend

Status: implemented (2026-11-02), following the recommendation in
`attacc-boundary.md`. Addresses audit finding M-04 and `CLAIMS.md` row E16
(compiler half). Paths are relative to this repository unless they start
with `<repo>/` (the containing Tenon repository).

## 1. What ships

| Piece | Where | Role |
|---|---|---|
| Target | `allo/pim/targets.py::build_attacc_target` | 5 HBM3-PIM devices (`stack`, mapping 5) x 16 channels x 64 PIM units (banks); `banks` 2^14 rows x 2^5 columns x 32 B, 2 pseudo-channels x 2 ranks x 4 bank groups x 4 banks; per-unit `gemv_buffer` and 16-lane FP16 `acc`; per-channel `softmax_buffer`. Ops `MAC` (unit), `SFM_EXP`, `SFM_SUM`, `SFM_DIV` (channel). Moves `WR_GB`, `MV_SB`, `MV_GB`; host `H2D` (scatter), `D2H` (gather). No timing. |
| Tiling domain, counts, program record | `allo/pim/attacc_tiling.py` | `AttAccKernel`, `attacc_kernel_from_matches`, `AttAccTiling`, `attacc_tiling_space`, `AttAccCounts`, `attacc_command_counts`, `AttAccProgram`, `rank_attacc_tilings` |
| Cost | `allo/pim/costs/attacc.py` (`attacc_cost`, `attacc_cost_program(timing)`) | paper's closed forms `m8` (GEMV) and `serial_q_cf` (attention) over the command inventory |
| Knob | `allo/spmw_knobs.py` `attacc_tiling` (registered for `"attacc"` only) | every legal tiling is a candidate |
| Enumerator, allowlist | `allo/spmw_autoschedule.py::_attacc_enumerate`; `"attacc_tiling"` in `_MATCHER_PHYSICAL_EXTRA_FIELDS` | one base placement crossed with the knob; the tiling is part of the decision key |
| Codegen context | `allo/spmw_attacc.py::AttAccCtx` | emits one `AttAccProgram` per kernel (`ctx.programs`), `cmds` lines `"PIM_MAC_AB count=15360"`; rejects illegal `layout=` tilings with `InfeasibleSchedule` |
| Route registration | `allo/spmw_codegen.py` `_BACKENDS["attacc"]`, `_SINGLE_ITEM_LAUNCH_BACKENDS` | runner `_run_attacc` raises `NotImplementedError` (analytical-only) |
| Workloads | `benchmarks/attacc/workloads.py`, `paper_selection.csv` | `gemv_region(H, K, N)`, `attention_region(H, D, L)`, the 500 instances, the paper's selected tilings (oracle) |
| Evaluation scripts | `benchmarks/attacc/scripts/` | `verify_paper_selection.py`, `selector_timing.py`, `run_ramulator.py` (external simulator), `compose_e2e.py` (external DCC composition) |

Public route:

```python
import allo
from allo.pim.targets import build_attacc_target
from allo.pim.costs import attacc_cost
from benchmarks.attacc.workloads import gemv_region

c = allo.compile(gemv_region(8, 5120, 384), build_attacc_target(), attacc_cost,
                 backend="virtual", reduction_order="reassociate")
r = c(W, x, y)              # kind="virtual", cycles=7040, arrays untouched
r.extra["attacc_tiling"]    # [[1, 8, 1], [8, 8, 80], [2, 1, 192]]
c.placement_ranking.kernels[0].scores   # 3,043 ranked tilings
```

`reduction_order="reassociate"` is required for FP16: every AttAcc MAC sums
16 lanes per column and the tiling splits the reduction over units, so no
program keeps the source order; under the default `"strict"` the route
raises `ReductionOrderError` before scheduling.

## 2. Target geometry and its sources

| Fact | Value | Source |
|---|---|---|
| devices, channels, banks per channel, dtype | 5, 16, 64, FP16 | paper `tables/eval-platforms.tex:53-56`; DCC `inference/config.py` `num_hbm=5` |
| channel organization | 2 pCH x 2 ranks x 4 BG x 4 banks, 2^14 rows, 2^5 columns | Ramulator2-AttAcc `src/dram/impl/HBM3-PIM.cpp:21` preset `HBM3_8Gb_2R` (MIT) |
| column, MAC lanes | 32 B, 16 FP16 lanes | AttAcc trace generator `prefetch_size = 32`, `n_mac = 16` (MIT AttAcc simulator) |
| command scopes | `WRGB`/`MVSB`/`MVGB` bank, `SFM` channel | `HBM3-PIM.cpp:86-88` |

The paper's "pCH" is Ramulator2's `channel` level; Ramulator2-AttAcc simulates
one device with `channel: 16`. One `@allo.work(mapping=[1])` kernel is tiled
onto one device; splitting heads over the five devices is the decode
composition's split (DCC divides heads by `NUM_HBM`) and is not a compiler
decision. Multi-item grids are rejected with that reason.

## 3. Tiling domain

Kernel axes come from `match.extra["axis_roles"]` (batch, reduction, output),
never loop positions: GEMV `(rows, reduction, outputs) = (H, K, N)`; attention
`(H, D, L)` from the score MAC, checked against the context MAC
`(H, L, D)` and the three softmax ops over `H x L`. FP16 operands, a
literal-zero accumulator and a single work item are required
(`AttAccDomainError` otherwise).

A tiling is `((a, b, s) for rows, reduction, outputs)`: channels, units,
serial extent. The space is DCC's 3-D space (Yang et al. 2025,
`generate_configs_3D`) re-implemented from target geometry: products of `a`
and `b` exhaust the device under integer division, the reduction keeps at
least one 16-lane column per unit, one tiling per distinct serial triple, plus
AttAcc's fixed forms (all channels on rows, one bank group's 4 banks on the
reduction, the 16 bank groups on the outputs; two extra fixed mappings when
there are fewer rows than channels). Legality from the target: factor
products within 16 channels and 64 units, `a*b*s >= extent`, and one row's
working set (vector + matrix tile + result, 32 B columns) within a 16 MiB bank.
Space sizes: 1,391 / 2,273 / 3,043 / 4,478 for the four paper GEMVs; 668,575
over the 500 instances (equal to DCC).

Known property of the reference model, kept: before counting, a non-fixed
tiling's factors are rounded up to powers of two (the trace address
mapping); for 197,321 of the 668,575 candidates the rounded fan-out exceeds
16 channels or 64 units, and those candidates are priced over the rounded
channel count. None is selected on the 500 instances.

## 4. Cost rules

`attacc_cost` prices the kernel record once per work item: on the `MAC`
event of a GEMV (`m8`: `max(MAC*nCCDAB + WR_GB*nCCDL, MV_SB*nCCDL +
ST*nBL) / channels`), on the `SFM_EXP` event of an attention kernel
(`serial_q_cf`: `(MAC_q*nCCDAB + WR_GB + MV_GB + MV_SB + SFM + ST) /
channels`, `MAC_q` with serial extents rounded to whole 16-lane columns).
Other record primitives are zero-latency (their commands are in the
counts); host transfers raise (no host-link model). Constants `nCCDAB = 6`,
`nCCDL = 4`, `nBL = 2` from `HBM3-PIM.cpp` preset `HBM3_5.2Gbps`. Cycles
are truncated to integers by the cost engine; the selection over the 500
instances is unchanged.

Provenance: formulas from `<repo>/reviews/experiments/results/2026-09-29-attacc-version-crosswalk-v3/sources/attacc_cost_cf.py`
(the paper's selector; the archived plugin's replay ranking is not used).
Counts validated against DCC commit `212f86ea7f08da059836e3a1311d3a11afae8dbe`
(checkout dated 2026-07-07): equal space, counts and cost on all 668,575
candidates (probe P1, 2026-11-02); `tests/pim/test_attacc_virtual.py` pins
24 sampled DCC counts including two multi-iteration tilings.

Validity: a ranking model for FP16 GEMV and attention scoring at the studied
GPT-13B per-device shapes (paper: Spearman 0.94 / 0.98, MAPE 30.78% / 14.19%
against Ramulator2-AttAcc on 337 / 180 samples); row-buffer state, refresh
and issue order are omitted; other shapes are extrapolated. The virtual
result carries this as `extra["cost_validity"]` and `timing_scope`.

## 5. Evidence (2026-11-02)

- Acceptance: qkv batch 1 selects `[[1,8,1],[8,8,80],[2,1,192]]` at 7,040
  cycles; the four GEMV instances 7,040 / 25,600 / 2,432 / 8,192 with the
  paper tilings; worst-loss attention shape (8, 128, 480) selects
  `[[8,1,1],[2,4,16],[1,16,30]]`; the attention tie shape (32, 128, 144)
  keeps a 2-way cost tie and selects the paper tiling.
- All 500 instances through `allo.compile(..., backend="virtual")`
  (`verify_paper_selection.py`, run once): 500/500 select the paper tiling,
  668,575 candidates ranked, 0 mismatches; the three instances with a 2-way
  cost tie (proj nh8, qkv nh32, attention nh32 L 144) resolve to the paper
  tiling under `TIE_BREAK_RULE`. Wall time 1,564.8 s (median 2.9 s per
  instance) on a shared host under load from concurrent jobs; this is
  frontend + matching + per-candidate graph construction, not the paper's
  24 s selector time.
- Selector alone (`selector_timing.py`): 668,575 candidates, first minimum
  equals the paper on 500/500, 40.0 s CPU on a loaded shared host
  (enumeration alone 33.9 s). The paper's 24 s was measured on another
  setup; this is not that figure.
- External timing smoke run (`run_ramulator.py`, `aim-simulator-build`
  image, DCC checkout read-only): qkv nh8 selected program 11,454 cycles,
  worst-loss attention 953 cycles, both equal to the archived/paper values;
  trace opcode counts equal the compiled `AttAccCounts`.
- Perturbation: `attacc_cost_program(AttAccTiming(n_bl=8))` selects
  `[[1,8,1],[4,8,160],[4,1,96]]` (8,576 cycles) on the same 3,043 legal
  candidates.
- Round 51 (2026-11-05), the paper's kernel rows through the shared
  pipeline with the native Ramulator2-AttAcc (`run_ramulator.py` rewritten:
  compiled artifact kept, selection provenance, trace structure check,
  simulator identity, retained artifacts; `benchmarks/attacc/trace_check.py`):
  qkv batch 1 `[[1,8,1],[8,8,80],[2,1,192]]` 11,454 cycles (0.00881 ms),
  qkv batch 4 `[[1,16,2],[16,4,80],[1,1,384]]` 43,750 (0.03364), projection
  batch 1 `[[1,8,1],[1,4,32],[16,2,160]]` 3,448 (0.00265), projection batch
  4 `[[1,32,1],[2,2,32],[8,1,640]]` 13,318 (0.01024): the paper's Tenon
  column; the fixed-form baselines re-run (minimum over the fixed-form
  candidates, DCC's rule) 36,504 / 224,170 / 3,448 / 13,398 cycles, the
  paper's Fixed column. The DCC generator's trace of each selected program
  equals the stream the program implies command by command (round 52
  check; round 51 compared phase order and fan-out cardinalities). Score rows, 248 shapes per batch: every selection equals
  `paper_selection.csv`; mean 0.00193 / 0.00678 ms; speedup over the fixed
  tiling 1.00x / 1.13x as the frozen figure plots (mean of per-shape
  ratios); against DCC's archived per-shape times 123 wins, 316 ties, 61
  losses over the 500 instances with a median loss of 7.07% (the frozen
  text's 123 / 316 / 61 and 7.1%). The superseded table
  `tables/eval-attacc-kernels.tex` (not input by the frozen paper) quotes
  0.0019 / 0.00677 ms and "Tenon <= DCC on every shape"; the frozen figure
  and text are what the measurements reproduce.

## 6. What stays external, and why

| Item | Why | Script |
|---|---|---|
| Ramulator2-AttAcc timing | simulator binary outside the package (a native build recipe ships: `build_ramulator_attacc.sh`); the trace emitter is DCC code (no license) | `run_ramulator.py` (`kind: "simulator-timing-only"`; selection labels `shared-pipeline` / `forced-tiling` / `fixed-form-baseline`; the trace checked against the selected program by `trace_check.py`) |
| GPT-13B decode composition | DCC `inference/system.py` + analytical A100 roofline (no license) | `compose_e2e.py` (`kind: "analytical-composition"`) |
| Reference-best search, replay predictor, accuracy, DCC baselines | evaluation of the cost model against simulator sweeps; the paper excludes the replay from selection | archived under `<repo>/tenon-artifacts/attacc-llm-e2e/` |

`compose_e2e.py` (rewritten in round 52) composes the six configurations
from the shared-pipeline kernel records only (conflicts rejected, forced
and fixed references ignored, unmeasured rows NaN) and re-runs DCC's
selection and the fixed baseline through the same model, the fixed variant
with DCC's own placement (attention on PIM, QKV and projection on the
GPU). With every kernel measured it reproduces the frozen figure, table
and text: Tenon full 6.50 / 6.95 / 7.40 / 3.09 / 3.48 / 3.87x, base 1.47 /
1.56 / 1.66 / 1.34 / 1.50 / 1.65x; DCC 6.50 / 6.95 / 7.40 / 3.05 / 3.44 /
3.82x; fixed 2.55 / 2.73 / 2.91 / 2.29 / 2.58 / 2.87x; batch 4 Tenon 1.27%
faster, QKV 1.2799 to 1.2332 ms of 3.9933 to 3.9431 ms. The earlier smoke
run's batch-4 numbers (1.28 / 1.45 / 1.61x) were the fixed-tiling times
substituted for the unmeasured kernels under the PIM placement; that
substitution no longer exists.

## 7. Open items

1. Max-stabilized softmax: needs an identity-initialized `SFM_MAX`
   accumulator in the matcher (`spmw_match_engine._accumulator_init`); the
   target and `attention_region` use the un-stabilized form until then.
2. `spmw_reduction_policy._REGROUPING_MATCHER_ROUTES` has no `attacc` entry,
   so the strict-order error names the generic reason.
3. `_runnable_backends()` lists `attacc` because its runner entry is not
   `None` (the runner only raises).
4. The movement stage has no `COMPUTE_LEVELS` entry for `attacc`; the
   movement record reads `not-analyzed` (the tiling is not a GF(2) layout).
5. The score row of an attention kernel stays unplaced (two homes, see
   `attacc_base_placement`).
6. DCC license (`attacc-boundary.md` open question 1) still applies to the
   external scripts.
7. `trace_check.py` (round 52) derives every command's opcode and address
   from the selected program and the target geometry (the generator's
   region placement, fan-out, serial walks and composition) and compares
   command by command; validated against DCC's own output on GEMV and
   attention shapes. The paper's 500 instances all fit one row block, so
   the paired-block attention branch is exercised only synthetically.
