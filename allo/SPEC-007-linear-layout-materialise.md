# SPEC-007 — LinearLayout materialisation, per-backend resolution

**Scope.** Resolve the `_layout = LinearLayout.identity(...); del _layout`
anti-pattern in `spmw_autoschedule.py` for the four non-Samsung backends
(AiM, UPMEM, APU v1, APU v2). Samsung is the reference: it builds the
layout, calls `optimal_swizzle`, then drives `materialise_handle` to derive
the canonical bank handle. The four others build a layout, discard it, and
assign handles by hand. TASK_DESCRIPTION §3 requires that each dead
construction either be made live (option **a**) or be removed and replaced
by an honest comment that states why the algebra cannot constrain anything
(option **b**).

## TL;DR

| Backend | Choice | One-line reason |
|---|---|---|
| AiM     | **b** (remove) | `gb` is a distinct hardware unit from any bank, not an algebraic offset; the layout has no output dim that names the bank-vs-gb axis. |
| UPMEM   | **b** (remove) | `wram` and `gprs` are discrete named storage classes; the layout's `element`/`tasklet` axes do not project onto a `storage-class` output dim. |
| APU v1  | **b** (remove) | Single VR file — one Placement, zero degrees of freedom; the layout adds no information. |
| APU v2  | **b** (remove, for now) | `l1[0/1/2]` are arbitrary distinct symbolic slots, not derived from any algebraic out_dim today. Note for Task 009: if argmin enumeration introduces a real `l1_row` out_dim, revisit. |

All four are **option b**. No `materialise_handle` extension is required
for this spec; it remains live only for Samsung (and any future backend
that materialises a real swizzle). Coder Task 008 deletes the dead
constructions and inserts the docstring text below verbatim.

## Status of `materialise_handle`

`materialise_handle` is defined in
`experiments/allo/allo/spmw_linear_layout.py:569-714`. It is callable today
and supports:

* `out_dim` resolution via direct attribute (`target.<out_dim>`),
  pluralised attribute (`target.<out_dim>s`), or explicit override
  (`handle_table={...}`).
* Scaled-identity columns: `masks = (s, 2s, 4s, ...)` → contribution
  `scale * sym`.
* Single-bit input dims: contribution `m * sym`.
* Fixed-int inputs: XOR-of-masks (equals addition on disjoint bits).
* Short-circuit on `Register` stores: returns the register directly
  (`spmw_linear_layout.py:710-711`).

What it does **not** support:

* Discrete storage-class selection that is not an output coordinate
  of the layout (e.g. "which of `wram` or `gprs` does `acc` live in").
  Adding this would mean inventing a new out_dim per storage-class and
  registering the per-class handle in `handle_table`. That is a real
  extension, not in scope for SPEC-007; the four backends below all
  choose **b** so the extension is unnecessary today.
* Multi-axis trees where two output dims (e.g. `bg`, `bank` in AiM)
  must be combined into a flat index. Today AiM does this by hand
  (`banks[8 * bg_id + bk_id]` at `spmw_autoschedule.py:208`); a real
  materialise call would need a multi-out-dim materialisation helper,
  also out of scope here.

## Per-backend rationale and replacement text

### AiM (`spmw_autoschedule.py:189-197`) — option **b**

**Why not (a).** The two candidates' algebraic layouts (`no_bank_layout`
with out_dims `("k",)`, `all_bank_layout` with `("k", "bank")`) describe
DRAM access patterns, not the choice of whether `y` is a per-bank operand
or the global-buffer broadcast operand. The hardware distinction —
`bank_handle = banks[8*bg + bk]` vs. `gb = target.gb` — is between two
*physically distinct units* on the AiM die (a per-bank SRAM-resident
operand vs. the chip-level global buffer with broadcast semantics for
`MAC_ABK`). `gb` is not an offset within `bank`; there is no
LinearLayout out_dim that names this axis, and `materialise_handle(out_dim="bank")`
on either layout would yield the same bank handle, losing the candidate
distinction.

**What the coder does.** Delete lines 189-197 (`no_bank_layout`,
`all_bank_layout`, `del`) and insert this comment immediately before
the `bg_id = UnitId(...)` line:

```python
# AiM topology note: the A/B choice here is per-bank MAC (MAC_SBK) vs.
# all-bank-broadcast MAC (MAC_ABK). Algebraically this is "does the
# layout factor `bank` into an input dim?" but the runtime distinction
# is which *physical unit* `y` lives on -- a per-bank operand
# (`banks[8*bg+bk]`) or the chip-level global buffer (`target.gb`).
# `gb` is not an algebraic offset from `bank`; it is a discrete
# hardware unit. LinearLayout cannot model the choice, so we enumerate
# the two Placements directly. `acc` always lives in the per-channel
# accumulator file (`target.gpr`) because AiM's MAC ISR writes there.
```

### UPMEM (`spmw_autoschedule.py:248-252`) — option **b**

**Why not (a).** The candidate layouts (`scalar_layout`,
`tasklet_layout`) have out_dims `("element",)` and
`("element", "tasklet")`. Neither names the storage class. The A/B
choice is whether `acc` rides `wram[2]` or `gprs` — and `wram`,
`gprs`, `mram` are discrete named slots on the DPU's storage
hierarchy, not coordinates on a linear address space. To make
`materialise_handle` drive this choice you would need to invent a
new out_dim (`storage_class` ∈ {`wram`, `gprs`}) and register
per-class handles in `handle_table`; that is a real algebra
extension with no other consumer.

**What the coder does.** Delete lines 248-252
(`scalar_layout`, `tasklet_layout`, `del`) and insert this comment
immediately before the `wram = target.wram` line:

```python
# UPMEM topology note: the A/B choice is whether `acc` lives in a
# WRAM cell or in the per-tasklet GPR file. `wram[0/1/2]` and `gprs`
# are discrete named storage classes on the DPU hierarchy
# (mram-vs-wram-vs-gprs), not coordinates on a linear address space,
# so LinearLayout has no out_dim that names this choice. The two
# Placements below encode the choice directly. Bulk MRAM loads are
# the C runtime's job; live operands stay in WRAM/GPR.
```

### APU v1 (`spmw_autoschedule.py:299-300`) — option **b**

**Why not (a).** The APU v1 enumerator returns exactly one
Placement; there is no A/B choice to express. All three operands
land in `target.vrs` (the single VR file). The "layout" is a
32768-bit identity on a single output dim `element` whose handle
materialises to the `Register` `target.vrs` — and
`materialise_handle` short-circuits any `Register` store to return
the register itself (`spmw_linear_layout.py:710-711`). The
algebra is degenerate: it is ignored. Carrying a dead constructor
to "look like Samsung" is worse than honest documentation here,
because future readers will look for a swizzle that does not
exist.

**What the coder does.** Delete lines 299-300 (`_layout = ...`,
`del _layout`) and insert this comment immediately before the
`vrs = target.vrs` line:

```python
# APU v1 topology note: 32K-lane bit-serial element axis with a
# single VR file (16 VRs per APUC; register pressure is deferred
# to the regalloc spec, Task 015). All compute operands live in
# `target.vrs`, so the enumerator has zero swizzle degrees of
# freedom -- one Placement, no LinearLayout algebra to exercise.
# MICRO '25 Opt2 (stage-axis lift, 1.25x) needs the move scheduler
# to see the full @allo.work chain and is also deferred.
```

### APU v2 (`spmw_autoschedule.py:331-334`) — option **b** (now)

**Why not (a) today.** The constructed layout has out_dims
`("element", "group")` — element-within-row and group-of-rows.
Neither projects onto the `l1[0/1/2]` slot index used in the
Placement dict. The three integer subscripts are arbitrary
distinct labels picked to keep the three operands distinguishable
in the placement dict (parity with UPMEM's `wram[0/1/2]` at
`spmw_autoschedule.py:260-262`); they are not L1 hardware
coordinates derived from the algebra. A real
`materialise_handle(out_dim="l1_row")` would need (i) an `l1_row`
input or output dim on the layout, (ii) a multi-out-dim
materialisation helper that knows how to combine `element` and
`group` into a single flat L1 address, and (iii) a meaningful A/B
choice over that axis.

**Interaction with Task 009.** This is the dependency the task
file flags. If Task 009 introduces a real argmin enumeration for
APU v2 (e.g. row-major vs. group-major L1 walk), the right shape
is to add an `l1_row` or `l1_walk` algebraic axis to the layout
*at that point* and revisit option (a). For SPEC-007 today,
option (b) is correct and Task 009 is free to either keep it
trivial or graduate to (a) when it has a concrete swizzle to
encode.

**What the coder does.** Delete lines 331-334
(`_layout = ...`, `del _layout`) and insert this comment
immediately before the `l1 = target.l1` line:

```python
# APU v2 topology note: 64K-lane element axis with a 16-row L1
# group. l1_sim treats all L1 addresses uniformly, so there is no
# swizzle algebra to exercise here yet -- the three operands ride
# distinct symbolic L1-row indices (`l1[0/1/2]`) chosen to keep
# the placement dict's operands distinguishable, not derived from
# any algebraic axis. If Task 009 introduces a real argmin
# enumeration over an L1-walk axis, add the axis as a LinearLayout
# out_dim at that point and revisit; today this is one Placement.
```

## What stays algebraic

* Samsung remains the live `materialise_handle` consumer
  (`_samsung_enumerate`, `spmw_autoschedule.py:105-161`). Do not
  touch it.
* The `LinearLayout` class and `materialise_handle` helper in
  `spmw_linear_layout.py` are unchanged.

## Test inventory

This spec is documentation-only at the algebra level, but the coder
implementation must not break:

* `experiments/allo/tests/` — any test that imports
  `_aim_enumerate`, `_upmem_enumerate`, `_apu_v1_enumerate`,
  `_apu_v2_enumerate` from `spmw_autoschedule` and inspects
  Placements. Replacement comments do not change Placement
  contents.
* Samsung is untouched, so `_samsung_enumerate` tests are gated
  by no-op.

The blast radius is confined to `spmw_autoschedule.py`. No
shared-file edits. No `materialise_handle` extension.

## Rollback

`git revert` of the SPEC-008 commit. The dead `del _layout` form is
restored; functional behaviour unchanged either way.

## Implemented (Task 008)

* `experiments/allo/allo/spmw_autoschedule.py` — option (b) applied to
  all four enumerators (`_aim_enumerate`, `_upmem_enumerate`,
  `_apu_v1_enumerate`, `_apu_v2_enumerate`). Each backend now carries
  the verbatim topology-note docstring from §"Per-backend rationale"
  above in place of the dead `LinearLayout.identity(...) + del`
  construction. `LinearLayout` import retained (Samsung still uses it).
* `grep -n 'del.*layout' experiments/allo/allo/spmw_autoschedule.py` →
  0 matches.
* `pytest tests/spmw/test_autoschedule.py tests/spmw/test_target_aim.py
  tests/spmw/test_target_apu_v1.py tests/spmw/test_target_apu_v2.py
  tests/spmw/test_target_upmem.py tests/spmw/test_linear_layout.py -q`
  → 34/34 passed. Full-suite failures (`test_e2e_mlp_apu_v1`,
  `test_run_returns_runresult_for_all_backends`) are pre-existing APU
  v1 ARC-toolchain/hardware failures confirmed against baseline; not
  caused by this change.
