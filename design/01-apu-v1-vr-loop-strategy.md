# Design 01 — APU v1 VR (vector-register) improvement strategy for the loop phase

Status: COMMITTED (ruling). Task `010-needs-arch-ruling-apu-v1`.
Owner: architect. Inputs: coder report 006
(`dev/06172026-beat-cinnamon-exo-cross-target/work/reports/006-coder.md`),
flo evidence (`experiments/baselines/cinnamon-exo/apu-v1/{gemv-cinm-4096,ffn-cinm-opt}.flo.log`),
TASK_DESCRIPTION Phase-1 mechanism budgets, MICRO '25 Opt2 (stage-axis lift).

This ruling tells the coder **which mechanism the APU v1 loop body
(task 015) may touch and which it must not**, derived from the Phase-0
profile and the operand shapes/`target.*` — not from any benchmark
literal. It does **not** authorise code; it scopes the spec the coder
will receive.

---

## 1. Problem statement

Phase-0 reproduced both APU v1 cells on the real Leda board:

| cell | shape | layout | crun (this session) | reference | delta | bottleneck class |
|---|---|---|---:|---:|---:|---|
| gemv-cinm-4096 | 4096×1024 gemv u16 | intra-VR | 3,597,670 | 3,603,732 | −0.17% | **compute-bound (at floor)** |
| ffn-cinm-opt | 64-256-64 FFN s16 | intra-VR | ~399k | 262,470 | +1.52x wall | **data-movement-bound** |

The loop phase asks: from the flo per-region crun (DMA% vs compute%,
inter-VR vs intra-VR), name the *single dominant* bottleneck per cell
and the Tenon mechanism that should match-or-beat Cinnamon.

The profile splits cleanly (ffn-cinm-opt, last run, region crun):

```
retile  132,890   seu:0      <- pure data movement (no compute issued)
dma     103,922   seu:348    <- L2DMA host<->L4<->VR
l1read   40,227   seu:2816   compute (GVML)
l1reduce 31,677   seu:719    compute
l2read   13,978   seu:704    compute
l2reduce  9,164   seu:821    compute
flush     4,892   seu:0
```

`retile + dma = 236,812` of ~399k total crun = **59% pure data
movement**. The four GVML compute regions sum to ~95k = ~24%. The
seu:0 on `retile` is decisive: zero scalar-engine-update events means
the region issues no compute, only moves. **For FFN the dominant
bottleneck is the data-movement / VR-tiling layer, not the MAC
expansion.** For GEMV the reverse holds: it is at the compute floor
(−0.17%), DMA is amortised across the 4096×1024 weight stream.

## 2. The mechanism gap this exposes

The current APU v1 decision path prices **only** compute:

- `spmw_autoschedule.py::_apu_v1_enumerate` returns exactly two
  candidates (`mode="sv"`, `mode="sv_lookup"`) whose `placements` dict
  is identical (`x,y,acc -> target.vrs`). It has, by its own docstring,
  "zero swizzle degrees of freedom"; the MICRO '25 Opt2 stage-axis
  lift and any VR-tile assignment are explicitly **deferred**.
- `spmw_cost_models.py::_apu_v1_kernel_cycles` sums `target.op(...).cycles`
  over matches with outer-loop multipliers. It models the MAC and the
  GVML ops. **It never reads `target.move(...)`.** Data movement is
  invisible to argmin.

So today argmin distinguishes SV from SV-lookup (a compute decision)
and is correct for GEMV. But it is **blind to the 59% of FFN's cost
that lives in retile + DMA**. inter-VR vs intra-VR — the very axis the
profile says dominates FFN — is not even an enumerated `Placement`
choice; it is a host-layout arg (`intra`) the coder passes on the
command line, and getting it wrong is the documented `FAIL 62/64`
gotcha. The decision is currently *outside Tenon*.

## 3. Considered options

**Option A — Recalibrate SV/SV-lookup cost buckets only.**
The 2026-04-21 note flags the cost gap (15,650× modelled vs ~19.8×
real). Tightening it is real work but orthogonal: it refines a
*compute* decision that is already correct for both cells (both pick
intra-VR/SV-lookup). It does nothing for FFN's movement bottleneck.
Rejected as the loop-phase mechanism: it does not touch the dominant
cost. (Kept as a separate, lower-priority cost-realism cleanup; see §7.)

**Option B — Codegen async-DMA / double-buffer the retile (mechanism in codegen).**
Hide retile latency behind compute by emitting async L2DMA. This is a
real win lever, but it is a **codegen materialisation**, and per the
blast-radius rule codegen "must not contain the decision." Whether to
double-buffer, and which VR tiling makes the retile cheap, is a
*placement* decision that must come from the enumerator+cost model
first. Rejected as the *entry* mechanism; admissible later as the
materialisation of a `Placement` field once B/C below exist.

**Option C — Make inter-VR vs intra-VR a real enumerated `Placement`,
and teach the cost model to price the move/DMA so argmin chooses it.**
This is the only option that (a) puts the dominant FFN cost inside
Tenon's decision loop, (b) is derived from operand shapes + `target.*`
(not a benchmark literal), and (c) leaves a `Placement` field codegen
can later materialise (Option B) without re-deriving the decision.

## 4. Chosen option: C, in two mechanism budgets, GEMV untouched

### Ruling, per cell

- **GEMV 4096×1024: NO mechanism change.** It reproduces at −0.17%,
  is compute-bound, and the existing SV-lookup pick is already the
  floor. The loop body must RUN→MEASURE→PROFILE it and record parity;
  it must **not** add VR machinery that cannot beat a floor. Any FFN
  mechanism change must re-prove GEMV stays at −0.17% (regression gate
  in §6).

- **FFN 64-256-64: the dominant bottleneck is data movement (retile +
  L2DMA, 59% of crun).** The permitted mechanism is **(2) the
  candidate enumerator + (3) the cost model**, in that order:

  1. **Enumerator (`spmw_autoschedule.py::_apu_v1_enumerate`)** gains
     the inter-VR vs intra-VR tiling as *two real `Placement`
     candidates*, distinguished by a `Placement.extra` field naming the
     VR-tile / DMA mode (e.g. `extra["vr_dma"] in {"intra","inter"}`).
     Each must be a materialisable placement (the `intra` one is what
     the board runs today; the `inter` one is the no-arg host layout
     that currently FAILs only because host and device disagree — under
     Tenon both sides are emitted from the same `Placement`, so it
     becomes a legitimate, correct candidate). The *number* of VR tiles
     and which operand axis folds into the 32K-lane width must be
     **computed from the operand shape and `target.vrs` / lane count**,
     never asserted. For 64-256-64 the K=256 contraction and the
     16-VR file are the inputs; the tile count is `ceil(K /
     lanes_per_fold)` style arithmetic over `target.*`, not `256`.

  2. **Cost model (`spmw_cost_models.py::_apu_v1_kernel_cycles`, or a
     sibling `move_cycles` component)** must price the retile + DMA so
     argmin can see it. The move cost is `n_moves * target.move(...).cycles`
     where `n_moves` is derived from the same shape/`target.vrs`
     arithmetic the enumerator used, and the per-move cost traces to a
     `target.move("LD_VR"/"ST_VR"/"L2DMA"…).cycles` field. **Add the
     declared move-op cycles to the target spec if a needed one is
     missing — that is a target-spec data change, not a fudge factor.**
     The intra/inter choice then falls out of argmin organically
     because inter-VR amortises the L4 round-trip across the contraction
     tile while intra-VR re-fetches per tile (or vice versa, whichever
     the shape arithmetic says) — the model ranks them; it does not
     recognise "FFN."

  3. **Codegen (`spmw_codegen.py` / `spmw_apu_v1_build.py`)** then
     *materialises* the chosen `extra["vr_dma"]` — emit the matching
     host L4 layout and device VR-tile DMA. It branches on the
     `Placement` field; it never re-derives intra-vs-inter from the
     workload. This is where Option B (async/double-buffer the retile)
     may later land, as the materialisation of an already-chosen
     placement, in a *follow-up* iteration once C lands and the profile
     re-measures.

### Why this is the right seam

inter-VR vs intra-VR is, in the layout algebra, a choice of **which
operand axis maps to the 32K-lane VR fold and which is streamed
through L4** — i.e. exactly a `LinearLayout` swizzle over the element
axis vs the contraction axis. The current enumerator collapsed this to
"identity, zero DOF" because at GEMV scale the streamed weight
dominates and the choice is forced. At FFN scale (small K, reused
activations) it is live. Surfacing it is mechanism-budget (1)
Linear-Layout reasoning feeding budget (2) the enumerator — the
canonical seam, not a special case.

## 5. File-level change list (for the coder's eventual spec — NOT authorised here)

All edits are **additive** (new candidate, new cost component, new
`Placement.extra` key with default), so FPGA/AIE and the other three
PIM backends are untouched by construction:

| file | change | additive? |
|---|---|---|
| `spmw_autoschedule.py::_apu_v1_enumerate` | emit 2× the candidates: existing {sv,sv_lookup} crossed with {intra,inter} vr_dma in `extra`; tile count computed from shape + `target.vrs` | additive (more candidates) |
| `spmw_cost_models.py::_apu_v1_kernel_cycles` (or new `_apu_v1_move_cycles`) | add move/DMA term: `n_moves(shape, target.vrs) * target.move(...).cycles` | additive (new term; absent → 0 keeps old ranking) |
| `spmw_target.py` (APU v1 spec data) | declare any missing `move(...)` cycles (L2DMA / retile) used above; pure data | additive (new spec field) |
| `spmw_codegen.py` / `spmw_apu_v1_build.py` | branch on `Placement.extra["vr_dma"]` to emit matching host+device layout (resolves the `FAIL 62/64` gotcha by construction) | additive (new branch on a field) |

No shared `allo/ir/builder.py`, `allo/ir/infer.py`, or `allo/dataflow.py`
edit is implied. The whole change lives in `spmw_*.py` + the APU v1
target data. This satisfies guardrail Q1 (cannot go *entirely* inside
one spmw file because enumerator + cost + codegen must agree on the
same `extra` key, but it stays entirely inside the `spmw_*` set and is
additive).

## 6. The test that proves it works

- **Correctness gate (board):** `ffn-cinm-opt` emitted from the chosen
  `Placement` PASSes all 64 outputs on the real board for *both*
  enumerated candidates (the inter-VR one no longer FAILs 62/64
  because host+device come from one placement). This is the proof that
  the inter-VR candidate is *materialisable*, not a label.
- **Win gate (board, profile-driven):** after argmin picks the
  movement-minimising candidate, re-run RUN→MEASURE→PROFILE on a
  **quiet board** (the +1.52x was board-load drift on a verified
  kernel; the kernel/layout/numerics were already reproduced). The
  retile+dma region crun must drop relative to the reproduced intra-VR
  baseline, and the drop must show in real crun, not just the model.
- **GEMV non-regression gate:** `gemv-cinm-4096 intra` must still
  reproduce at −0.17% (crun ≈ 3.60M); argmin must still pick SV-lookup
  intra-VR for it. Same enumerator, same cost model — proves the FFN
  change did not perturb the compute-bound cell.
- **SPMW pytest non-regression:** full `tests/spmw/` suite green +
  every other in-scope target's passing cells (blast-radius rule).
  Relevant existing guards: `tests/pim/test_autoselect_bmatmul_layout.py`
  (argmin still picks the cheaper APU candidate), `tests/pim/
  test_codegen_layout_dispatch.py` (codegen branches on `Placement`
  fields, never workload identity), `tests/pim/
  test_bmatmul_sv_lookup_low_mode.py` (SV-lookup reference parity).
- **Anti-hardcoding gate:** the audit must trace the FFN tile count to
  `ceil`-arithmetic over operand shape + `target.vrs`, and the move
  cost to a `target.move(...).cycles` field — no `256`, no `64`, no
  inter/intra literal keyed on FFN. (TASK_DESCRIPTION §forbidden list.)

## 7. Rollback story if APU/FPGA breaks

The change is additive: the new candidate and the new cost term carry
defaults that reproduce today's ranking (no `vr_dma` key → cost term
0 → argmin behaves exactly as now, picking intra-VR/SV-lookup). FPGA
CI is untouched (no shared-IR edit). Rollback = drop the `inter`
candidate from `_apu_v1_enumerate` and the move term from the cost
model; the codegen branch becomes dead. GEMV and all other backends
are unaffected at every step because they never see the new `extra`
key.

## 8a. Open question surfaced during implementation (task 015, coder)

Two mechanism facts collided with the existing autoscheduler structure;
flagging for the architect before the FFN board win can land end-to-end:

1. **The inter-stage retile is invisible to the per-bucket cost path.**
   `autoschedule` (`spmw_autoschedule.py:693`) groups matches by
   `func_name` and runs the enumerator + `cost_fn(sub_trace, layout)`
   **per `@allo.work` bucket**. The FFN's two layers are two distinct
   `@allo.work` funcs (`mlp_layer1`/`mlp_layer2`), and even a *fused*
   single-func kernel is rejected because its two MACs bind role `x` to
   different weights (W1 vs W2) and `_trace_memrefs_by_role`
   (`spmw_autoschedule.py:87`) forbids a non-uniform role->memref within
   one bucket. So `n_stage_boundaries` (= #MACs-1) is always 0 at the
   per-bucket level, and the move cost ties intra/inter for *every*
   single matmul. The cross-stage retile term only fires when
   `_apu_v1_move_cycles` is called on the **whole trace**. Landed
   mechanism is correct and verified at the whole-trace cost-call level
   (`tests/spmw/test_apu_v1_vr_dma_cost.py::test_argmin_flips_to_inter_for_fused_ffn`),
   but argmin via the production `autoschedule` path will *not* flip a
   two-bucket FFN to inter. **Decision needed:** should the APU v1 cost
   see the full trace (a whole-kernel move-cost pass, distinct from the
   per-bucket compute cost), or should the FFN be matched as one bucket
   (which needs the role-uniformity invariant relaxed for multi-weight
   fused kernels)? Either is an autoscheduler-structure change beyond the
   additive scope this task authorized.

2. **The `inter` board candidate is not materialisable on Shiran's
   harness.** The correctness gate (§6) asks the inter-VR candidate to
   PASS 64/64 once host+device come from one Placement. Tenon does not
   yet emit an FFN board kernel — the board harness is Shiran's
   hand-written `ffn-cinm-opt`, whose `inter`/no-arg path FAILs 62/64
   (measured this session: host doesn't emit the matching inter-VR L4
   layout for that device kernel). The codegen materialisation hook
   (`APUv1Ctx.vr_dma_mode`, branched in `_walk_and_emit`) is in place, but
   demonstrating the inter PASS on hardware requires a Tenon-emitted FFN
   board project (the SPEC for which does not exist). The board evidence
   in this cell is therefore: GEMV reproduced at floor (crun 3,610,756,
   +0.19%), FFN intra verified PASS (crun ~401,648, retile+dma = 59%
   confirming the bottleneck class), inter board-blocked on the missing
   Tenon FFN emitter.

## 8. Open / deferred (not blocking this ruling)

- **Cost-bucket realism** (Option A, the 15,650× vs ~19.8× note): a
  separate, lower-priority cost-realism cleanup. It does not gate the
  FFN movement win and is not part of task 015. Leave for a later
  iteration; surface in the paper's scoring-rubric section.
- **Async/double-buffered retile** (Option B): the *next* iteration
  after C lands — materialise the chosen `vr_dma` placement with async
  L2DMA to overlap retile with compute. Pure codegen mechanism; needs
  the C `Placement` field as its input. Tracked as a follow-up, not
  appended as a research task because it is engineering, not an open
  research question.
- **No research task appended.** Every sub-question here resolves from
  the profile + `target.*` + existing reports (12, 14); nothing is
  exploratory enough to route to the researcher.

---

This updates `SPMW_ARCHITECTURE.md` Open-tension **T17 (new)**: APU v1
inter-VR/intra-VR DMA tiling is enumerator-ephemeral and cost-invisible
— resolved-in-principle by this ruling, to be implemented by task 015.

---

## Implemented (task 015, coder)

- `spmw_autoschedule.py::_apu_v1_enumerate` — now emits 4 candidates
  ({sv,sv_lookup} x {intra,inter} vr_dma), each carrying shape-derived
  `extra` tile counts (n_out_tiles, n_weight_tiles, n_stage_boundaries).
- `spmw_cost_models.py` — new `_apu_v1_vr_tiling` (ceil-arithmetic over
  enclosing_loops + target.vrs.width) and `_apu_v1_move_cycles`
  (intra = weight-stream + per-boundary replication; inter = weight-stream
  only; per-move = target.move("DMA_L4_L1").cycles + target.move("LD_VR").cycles).
  Added to `_apu_v1_kernel_cycles` (absent vr_dma -> +0, ranking preserved).
- `spmw_codegen.py::APUv1Ctx` — new `vr_dma_mode` field, set from
  `Placement.extra["vr_dma"]` in `_walk_and_emit` (materialisation hook).
- Target spec (APU v1 fixture) — no new move-op needed; reused declared
  DMA_L4_L1 (140) + LD_VR (5).
- Tests: `tests/spmw/test_apu_v1_vr_dma_cost.py` (7 new) + updated 2 APU v1
  tests in `tests/spmw/test_autoschedule.py`.
- Board (real Leda, clean rebuild, health-checked FW=background Apuc_Mask=0xf):
  GEMV intra PASS crun 3,610,756 (+0.19% vs floor 3,603,732); FFN intra PASS
  crun ~401,648 / 803us (retile 133k seu:0 + dma 103k = 59%, confirms
  data-movement bottleneck class). FFN inter board-blocked (host/device
  mismatch FAIL 62/64 on Shiran's harness; needs Tenon FFN emitter — §8a).
- Two open seams flagged to architect in §8a (cross-stage cost visibility
  under per-bucket autoschedule; missing Tenon FFN board emitter).
