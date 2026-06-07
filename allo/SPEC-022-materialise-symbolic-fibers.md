# SPEC-022 — `materialise_handle`: both bank fibers from symbolic layout algebra

**Status.** Architect spec for task 020 (`needs-arch-materialise-symbolic-fibers`).
Blocks coder task 031 (lever-1 even/odd dual-fiber placement). Independent of
task 010's cycle-model verdict (this is pure Tenon-side layout algebra; it does
not touch the run path or the simulator).

**Scope guardrail.** This spec changes exactly one shared file,
`experiments/allo/allo/spmw_linear_layout.py`, and only **additively** (a new
keyword argument with a default, plus one new helper). `spmw_codegen.py::_bank_parity`
is **unchanged** — the new path must keep emitting the SymExpr forms it already
classifies. `spmw_autoschedule.py::_samsung_enumerate` is updated by the coder
under task 031, not here; this spec only fixes its call into `materialise_handle`.

---

## 1. Verdict: the current path can emit both fibers, but NOT as a layout receipt

**The scalar-multiplier path already emits both fibers.** With the swizzled
layout, the existing helper produces `2*pid` for `fixed={"tile":0}` and
`2*pid+1` for `fixed={"tile":1}`. This is already exercised and green:

> `experiments/allo/tests/spmw/test_linear_layout.py:143-164`
> ```python
> h_even = materialise_handle(swizzled, target=target, out_dim="bank",
>     fixed={"grf": 0, "tile": 0}, symbol_table={"bank": 2 * pid})
> assert _bank_parity(h_even.idx) == "EVEN_BANK"
> h_odd = materialise_handle(swizzled, target=target, out_dim="bank",
>     fixed={"grf": 0, "tile": 1}, symbol_table={"bank": 2 * pid})
> assert _bank_parity(h_odd.idx) == "ODD_BANK"
> ```

So at the level of "does an even and an odd index come out," the answer is
**already-works**.

**But it fails the anti-hardcoding gate's positive evidence #4.** The gate
(TASK_DESCRIPTION §"Positive evidence" #4) requires the even/odd indices to
"come out of `materialise_handle` over the swizzled layout for *symbolic*
`pid`/`tile`, **not from a literal `2*pid`/`2*pid+1` pasted at the call site**."
The current call site fails this on two counts, both visible in the quoted
source:

1. **The `2` multiplier is pasted, not derived.** The bank coordinate is bound
   by hand via `symbol_table={"bank": 2 * pid}` in the enumerator
   (`spmw_autoschedule.py:160`). The factor `2` is the bank-per-unit ratio
   (`banks(16) / pim_units(8)`); today it is a literal the caller types, not a
   number the layout algebra produces. A reviewer greps `2 * pid` at the call
   site and cannot point to a layout output that yielded it.

2. **`tile` is bound by `fixed=`, so the fiber is selected by the caller, not
   spanned by the algebra.** Producing the odd fiber requires the caller to
   *know* to pass `fixed={"tile":1}`. The two fibers are two separate calls with
   two hand-chosen constants, not one symbolic evaluation over the `tile` axis.
   The swizzle column `bases["tile"]=[(0,1)]` (the `tile→bank-bit-0` XOR) is what
   *makes* tile=1 add `+1`, but that derivation is invisible: the receipt the
   verifier must reproduce ("show `2*pid` and `2*pid+1` emerge from the swizzle
   algebra for symbolic inputs") cannot be written against a `fixed`-int call.

**Decision: bounded extension.** `materialise_handle` needs a small additive
extension so that (a) the even-bank base `2*pid` is *derived* from the layout's
`bank` column scale times the symbolic pim unit (not pasted), and (b) the `tile`
axis can be carried **symbolically** so a single call emits a fiber expression
parameterised by `tile`, from which both concrete fibers fall out by
substitution. The XOR-of-masks F2 rule the helper already implements for *fixed*
single-bit inputs is generalised to a *symbolic* single-bit input. No full
arbitrary-multi-axis F2 expansion is required — the extension is bounded to the
single-bit-symbolic case, which is exactly the swizzle column shape.

---

## 2. The algebra (why `2*pid` and `+tile` are layout outputs, not literals)

### 2.1 The swizzled layout, concretely

`_samsung_enumerate` builds, then swizzles:

```
base     = identity({"grf":8,"bank":16,"tile":2}, out=("grf","bank"))
swizzled = optimal_swizzle(base, vec_dims=("grf",), bank_dims=("bank",),
                           segment_dims=("tile",))
```

For `out_dim="bank"` (out index `j=1`) the swizzled basis columns are:

| input dim | basis vectors (bank-column entry only) | kind |
|---|---|---|
| `grf`  | `(0,0,0)` | all-zero → no bank contribution |
| `bank` | `(1,2,4,8)` | scaled-identity, **scale `s=1`** |
| `tile` | `(1,)` | single-bit, **mask `m=1`** (the swizzle XOR) |

(Verified by `test_optimal_swizzle_samsung_gemv` at
`tests/spmw/test_linear_layout.py:123`: `swizzled.bases["tile"] == [(0,1)]`.)

### 2.2 The bank base `2*pid` is the scaled-identity column applied to the pim unit

The hardware ratio is a **target-geometry fact**, not a workload fact. The
Samsung target tree declares (in `tests/spmw/_fixtures.py:107-115`):

```python
@allo.unit(mapping=[8])      # 8 pim units
def pim():
    _, pid = allo.get_uid()
    even_bank = banks[2 * pid]      # banks declared with banks=16
    odd_bank  = banks[2 * pid + 1]
```

So `bank_index = (banks_per_pim) * pid + parity`, where
`banks_per_pim = target.banks.banks // pim_mapping = 16 // 8 = 2`. The `2` is
`out_size(bank) / n_units(pim)` — read off the target, valid for any geometry.

This `2*pid` is exactly the image of the `bank` **scaled-identity column** when
its input is bound to the symbolic pim unit *scaled by the per-unit stride*. The
extension makes the helper emit `stride * pid` from the column, where `stride`
is supplied by the caller as the **bank-per-unit ratio computed from
`target` geometry** — not a literal `2`. The column scale `s` (here 1) and the
caller stride compose: `contribution = (s * stride) * pid`.

### 2.3 The `+tile` term is the swizzle XOR column, carried symbolically

The `tile` column `(1,)` is the swizzle's `tile→bank-bit-0` XOR. For a fixed
input the helper already does (lines 681-687) "XOR the basis masks for set bits,"
which on bit 0 gives `+1` when `tile=1` and `+0` when `tile=0`. Generalised to a
**symbolic** single-bit input `tile_sym ∈ {0,1}`, the F2 contribution on bit 0
is `m * tile_sym = 1 * tile_sym = tile_sym` (mask `m=1`). Because the column
touches only bit 0 and the `bank` column's image `2*pid` is even (bit 0 always
clear), the XOR is disjoint from the bank base and equals addition:

```
idx(pid, tile_sym) = 2*pid  XOR  tile_sym  =  2*pid + tile_sym      (tile_sym ∈ {0,1})
```

This is the single symbolic expression both fibers fall out of:
`tile_sym = 0 → 2*pid` (EVEN_BANK), `tile_sym = 1 → 2*pid + 1` (ODD_BANK).

---

## 3. API change (additive, defaulted)

Add **one keyword argument** to `materialise_handle` and **one module-level
helper**. Nothing existing changes shape; all current callers and the four
non-Samsung backends are unaffected because the new kwarg defaults to today's
behaviour.

### 3.1 New keyword on `materialise_handle`

```python
def materialise_handle(
    layout,
    *,
    target,
    out_dim,
    fixed=None,
    symbol_table=None,
    handle_table=None,
    symbolic=None,          # NEW: dict[str, SymExpr]  (default None == {})
):
```

**Semantics of `symbolic`.** For each input dim `d`:

* If `d in fixed` → unchanged (concrete int, XOR-of-masks; existing path).
* Elif `d in symbolic` → bind `d` to the supplied `SymExpr` and evaluate its
  column **symbolically**. The column must be either scaled-identity
  (contribution `(s) * sym`, or `sym` when `s==1`) or single-bit
  (contribution `sym` when mask `m==1`, else `m * sym`). This is the **same
  rule** the `else:` branch (lines 688-703) already applies to `symbol_table`
  entries — `symbolic` just makes the intent explicit and lets a dim that the
  caller previously had to `fixed` be carried as a free `SymExpr`. The
  `NotImplementedError` for non-single-bit / non-scaled-identity multi-bit
  symbolic columns (line 698) is **retained verbatim** — we are not adding
  arbitrary multi-bit symbolic F2 decomposition; only the single-bit swizzle
  column and scaled-identity columns are in scope.
* Elif `d in symbol_table` → unchanged (back-compat).
* Else → auto `UnitId` default, unchanged.

`symbolic` and `symbol_table` are merged with `symbolic` taking precedence;
`fixed` keys must not appear in `symbolic` (raise `ValueError` if they do — a
dim cannot be both pinned and free).

**Disjointness/parity contract.** When the same `out_dim` receives a
scaled-identity column image (e.g. `2*pid`, even) and a single-bit swizzle
column on bit 0 (`tile_sym`), the helper emits `base + sym` (addition), which is
valid because the masks are bit-disjoint (the F2 XOR equals integer addition on
disjoint bits — the invariant already relied on at lines 660-661, 680). The
extension must assert this disjointness when it combines a symbolic single-bit
column with any other column on the same out_dim, and raise a clear error
otherwise rather than silently emitting a wrong `+`.

### 3.2 New helper: bank-per-unit stride from target geometry

```python
def bank_stride(target, *, out_dim: str, unit_level: int) -> int:
    """Banks-per-unit ratio = out_size(out_dim) // n_units(unit_level).

    Reads the geometry off `target` ONLY (e.g. target.banks.banks // 8 for
    Samsung's pim level). Carries no workload shape. Used by the enumerator to
    derive the scaled stride for the pim unit so the bank base `stride*pid`
    is layout/target-derived, not a pasted `2`.
    """
```

The coder implements `bank_stride` to read `out_size` from the layout's
`out_sizes[out_dim]` (or the target memory's declared axis size) and the unit
count from the target's `mapping` at `unit_level`. **It must contain no integer
literal `2`, `8`, or `16`.** It returns `2` for Samsung *because the geometry
says so*, and would return a different number for a target with a different
bank:unit ratio.

### 3.3 The new enumerator call site (for task 031; shown here as the contract)

The coder under **task 031** replaces the pasted binding
(`symbol_table={"bank": 2 * pid}`, `fixed={"grf":0,"tile":0}`) with:

```python
pid  = UnitId(level=1, unit=None)                      # the pim unit
tile = UnitId(level=<tile_level>, unit=None)           # symbolic tile/parity axis
stride = bank_stride(target, out_dim="bank", unit_level=1)   # == 2 for Samsung, derived

fiber = materialise_handle(
    swizzled, target=target, out_dim="bank",
    fixed={"grf": 0},
    symbolic={"bank": stride * pid, "tile": tile},     # tile carried SYMBOLIC
)
# fiber.idx is the single SymExpr  stride*pid + tile.
# Even fiber  = substitute tile->0  (or call with symbolic tile==0): _bank_parity == EVEN_BANK
# Odd  fiber  = substitute tile->1                                  : _bank_parity == ODD_BANK
```

Two acceptable shapes for "produce both concrete fibers," coder may pick either
(both satisfy the receipt because the `+tile` term is the swizzle output, and the
`stride*pid` base is `bank_stride`-derived):

* **(A) one symbolic call + substitution.** `materialise_handle` returns
  `idx = stride*pid + tile` once; codegen/enumerator substitutes `tile=0/1`. This
  requires a `SymExpr` substitution utility (a 1-arg walk replacing a `UnitId`
  with an int) — note this as the small dependency; `_bank_parity` already
  evaluates `2*X` and `2*X+1` so the substituted forms classify correctly.
* **(B) two calls, symbolic `tile` re-bound to a 1-bit literal each.**
  `symbolic={"bank": stride*pid, "tile": 0_or_1}` where the `0`/`1` is the
  parity index iterated by the codegen loop (`for parity in range(banks_per_pim)`),
  not a pasted constant. Each call emits `stride*pid + parity`.

Either way the receipt holds: `_bank_parity` must classify the `tile=0`
expression as `EVEN_BANK` and the `tile=1` expression as `ODD_BANK`, and the
only constants in the call are `target`-derived (`stride` from `bank_stride`) or
the parity loop index spanning `range(banks_per_pim)`.

---

## 4. Worked symbolic derivation (the receipt template)

This is the exact derivation the verifier reproduces for positive-evidence #4.
**No workload literal (4096/1024/etc.) appears; the only integers are the
swizzle column masks and the target-geometry stride.**

```
Given (all from layout + target, none from the workload):
  swizzled.bases["bank"][:, bank] = (1, 2, 4, 8)     # scaled-identity, s = 1
  swizzled.bases["tile"][:, bank] = (1,)             # swizzle XOR on bit 0, m = 1
  stride = bank_stride(target, out_dim="bank", unit_level=1)
         = out_size(bank) // n_units(pim) = 16 // 8 = 2     # target geometry

Let pid be the symbolic pim UnitId, tile the symbolic 1-bit segment axis.

Bank column (scaled-identity, s=1), input bound to stride*pid:
    contribution_bank = s * (stride * pid) = 1 * (2*pid) = 2*pid        ... (even base)

Tile column (single-bit, m=1), input bound symbolically to tile:
    contribution_tile = m * tile = 1 * tile = tile                      ... (parity)

Combine over F2 on out_dim "bank". The two columns are bit-disjoint
(2*pid has bit 0 == 0 for all pid; tile occupies bit 0), so XOR == addition:
    idx(pid, tile) = (2*pid) XOR (tile) = 2*pid + tile

Substitute the two values the segment axis ranges over (range(stride) parity):
    tile = 0  ->  idx = 2*pid       =>  _bank_parity == "EVEN_BANK"
    tile = 1  ->  idx = 2*pid + 1   =>  _bank_parity == "ODD_BANK"
```

Both fibers are now *provably layout-derived*: `2*pid` is the `bank`
scaled-identity column at the `bank_stride`-derived input, and `+1` is the
`tile→bank-bit-0` swizzle XOR column with the segment axis at parity 1. The
enumerator never types `2*pid` or `2*pid+1`.

---

## 5. SymExpr forms the two fibers MUST produce (so `_bank_parity` classifies them)

`_bank_parity` (`spmw_codegen.py:159-178`) matches exactly:

* `EVEN_BANK` ⟺ `SymExpr("mul", 2, UnitId)` or `SymExpr("mul", UnitId, 2)`.
* `ODD_BANK` ⟺ `SymExpr("add", <2*UnitId>, 1)` (either arg order), where
  `<2*UnitId>` is the mul form above.

Therefore the extension MUST emit, for Samsung's `stride==2`:

* even fiber `idx` = `2 * pid` as `SymExpr("mul", 2, pid)` (the `scale==1`
  branch returns the bound symbol unchanged; the bound symbol is `stride*pid`
  which is already `SymExpr("mul", 2, pid)` — so this is automatic).
* odd fiber `idx` = `2 * pid + 1` as `SymExpr("add", SymExpr("mul", 2, pid), 1)`
  (the disjoint XOR-as-addition of the bank base and the `tile=1` mask).

**Constraint on the coder:** do not "simplify" `stride * pid` to a form
`_bank_parity` won't recognise. If `stride != 2`, `_bank_parity` (which is
hardcoded to factor-2 even/odd) will *correctly* reject it — that is the honest
signal that a non-2 bank:unit ratio needs a generalised classifier, a real
finding, not something to paper over. For this task Samsung is `stride==2`, so
the emitted forms match `_bank_parity` as-is and **`_bank_parity` is not
touched**.

---

## 6. Shared-file boundary (for SPMW_ARCHITECTURE.md §2)

| File | Change | Justification | Guarding test |
|---|---|---|---|
| `allo/spmw_linear_layout.py` | **additive**: new `symbolic=` kwarg on `materialise_handle`; new `bank_stride(target, *, out_dim, unit_level)` helper | Lets the swizzle `tile` column be carried symbolically and the bank base be target-derived, so even/odd fibers are layout outputs (gate evidence #4), not pasted call-site literals | `tests/spmw/test_linear_layout.py::test_materialise_symbolic_fibers` (new, see §7) + existing `test_materialise_samsung_bank_handle` must stay green (back-compat) |
| `allo/spmw_codegen.py` | **none** | `_bank_parity` already classifies `2*pid` / `2*pid+1`; extension emits the same forms | `test_materialise_samsung_bank_handle` asserts `_bank_parity` on both fibers |
| `allo/spmw_autoschedule.py::_samsung_enumerate` | call-site only, **under task 031** | Replace `symbol_table={"bank":2*pid}`+`fixed={"tile":0}` with `symbolic={"bank":stride*pid,"tile":tile}` | `tests/spmw/test_autoschedule.py`, `test_linear_layout.py::test_autoschedule_picks_canonical_samsung_layout` |

**Blast radius.** `materialise_handle` is shared, but the change is a new kwarg
defaulting to `None`. The four non-Samsung enumerators choose option (b) in
SPEC-007 and **do not call `materialise_handle` with `symbolic=`** — they are
untouched. Upstream Allo (FPGA/AIE) does not import `spmw_linear_layout`, so
`tests/dataflow/` and `tests/customize/` are not gated by this change.

**Rollback.** The kwarg is additive; reverting the enumerator call site
(task 031) restores the `2*pid`/`fixed-tile` form and the helper extension is
dead but harmless. `git revert` of the SPEC-022 implementation commit suffices.

---

## 7. Acceptance criteria the coder must meet

1. **Verdict honoured.** `materialise_handle` gains a `symbolic=` kwarg; the
   single-bit swizzle column is evaluated symbolically. No new
   `NotImplementedError` regressions — the existing multi-bit-symbolic guard
   (line 698) stays.
2. **`bank_stride` derives the stride.** `bank_stride(target, out_dim="bank",
   unit_level=1)` returns `2` for Samsung **computed** from
   `out_size(bank) // n_units(pim)`. It contains no `2`/`8`/`16` literal. A
   grep for `2 * pid` in `spmw_autoschedule.py` after task 031 finds **zero**
   hardcoded occurrences (the binding is `stride * pid`).
3. **Both fibers from one symbolic layout call.** A new test
   `tests/spmw/test_linear_layout.py::test_materialise_symbolic_fibers`:
   * builds the swizzled layout (no shape constants),
   * computes `stride = bank_stride(...)`,
   * calls `materialise_handle(..., symbolic={"bank": stride*pid, "tile": tile})`
     (form A) **or** loops `parity in range(stride)` with
     `symbolic={"bank": stride*pid, "tile": parity}` (form B),
   * asserts `_bank_parity(even_idx) == "EVEN_BANK"` and
     `_bank_parity(odd_idx) == "ODD_BANK"`,
   * asserts the **only** integers feeding the call are `stride` (from
     `bank_stride`) and the parity loop bound `range(stride)` — i.e. the test
     itself contains no `2 * pid` / `2*pid+1` literal binding.
4. **Back-compat green.** `test_materialise_samsung_bank_handle`,
   `test_optimal_swizzle_samsung_gemv`,
   `test_autoschedule_picks_canonical_samsung_layout`, and the full
   `tests/spmw/` suite stay green (AiM/UPMEM/APU enumerators unchanged).
5. **Receipt reproducible.** The §4 derivation runs symbolically end-to-end:
   feeding symbolic `pid`/`tile` yields `idx = 2*pid + tile`, and substituting
   `tile∈{0,1}` yields the two classified fibers — with every integer in the
   derivation traceable to a swizzle column mask or `bank_stride`. This is the
   artifact the anti-hardcoding audit (task 100) cites for positive evidence #4.

---

## 8. Open tension (not blocking task 031)

`_bank_parity` is hardwired to factor-2 even/odd (`2*X`, `2*X+1`). It generalises
the fiber concept only for `stride==2`. If a future target has
`banks_per_pim > 2` (or a non-power-of-two unit:bank ratio), the classifier and
the codegen even/odd interleave both need a `range(stride)`-fiber generalisation.
That is out of scope for this Samsung task (stride is exactly 2) and is logged
here so it is not silently assumed away. See architect memory entry for the
considered options.
