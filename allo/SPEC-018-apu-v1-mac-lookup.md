# SPEC-018 — APU v1: fix `gvml_lookup_16` 4-arg signature and supply LUT

Status: draft for coder, design-only artifact (no source edits in this task).
Owner of implementation: coder agent (task 003).
Author: architect.
Date: 2026-05-19.
Gated by: SPEC-001 / task 001 PASS (already landed — canary header set
trimmed to just `gsi/libgvml_element_wise.h`, so the skip gate no
longer hides the compile failure this spec fixes).

---

## 1. TL;DR for the coder

`APUv1Ctx.emit_mac_lookup` at `spmw_codegen.py:883–898` emits

```c
gvml_lookup_16(mac_tmp_vr, vrs, vrs);     // 3 args — WRONG
gvml_add_s16(<acc>, <acc>, mac_tmp_vr);
```

but `gvml_lookup_16` is declared in `/usr/local/include/gsi/libgvml_element_wise.h:751` as

```c
void gvml_lookup_16(enum gvml_vr16 vdst, enum gvml_vr16 vsrc,
                    const uint16_t *lut_addr, unsigned int lut_size);
```

Fix:

1. **LUT lives in the `program_cmd` struct as an inline `uint16_t lut[256]` field**
   (canonical pattern, copied verbatim from
   `/usr/local/gsi-apu/13.7.1/gsitech-course101-3bb486a917be/programming_guide/app_development_on_apu_tutorial/lab_1b/`,
   which is the only `gvml_lookup_16` example shipped with the SDK).
   The cmd struct is allocated by `host.c` in `GDL_CONST_MAPPED_POOL` and
   DMA'd to the device via `gdl_mem_cpy_to_dev(dev_cmd_buf, &cmd, ...)`
   — see `spmw_apu_v1_build.py:477`. No new L4 buffer, no new memory
   handle, no DMA tier in the device prologue. 512 B against a 32 KB L1
   budget is unmeasurable; the LUT does not transit L1 at all because
   `gvml_lookup_16` takes a raw `const uint16_t *` pointer (not a
   `gvml_vm_reg`).
2. **Host writes the LUT once** in `host.c`'s setup block. For the
   binary-MAC pattern that `sv_lookup` selects, the contents are
   `cmd.data.mac_lut[k] = popcount16(k) for k in 0..255` (256 entries,
   one byte index → its bit count). The function is computed in plain
   C; no extra dependency.
3. **Device side picks the LUT pointer up from the cmd struct.**
   `_emit_device_c` declares one extra decl:
   ```c
   const uint16_t *mac_lut_ptr = (const uint16_t *)data->mac_lut;
   ```
   and `APUv1Ctx.emit_mac_lookup` emits the corrected 4-arg call.
4. **No `MatchedOp` role change, no `Placement` field change.** The
   LUT slot is implicit: every `placement.mode == "sv_lookup"` use of
   `MAC` shares the single `data->mac_lut` field, named once by the
   build harness. This is the same shape as `_L4_PTR_BY_ROLE` —
   canonical names baked into the ctx, not autoscheduler output.

Three files change. Zero shared-Allo edits. One test grows an assertion.

---

## 2. Background and evidence

### 2.1 The real signature

`libgvml_element_wise.h:751`:

```c
void gvml_lookup_16(enum gvml_vr16 vdst, enum gvml_vr16 vsrc,
                    const uint16_t *lut_addr, unsigned int lut_size);
```

Notes from the doc comment (lines 701–730):

- `vdst[i] = lut_addr[vsrc[i]]` — the keys in `vsrc` index entries in
  `lut_addr`.
- `lut_addr` for `gvml_lookup_16` has **no L3 alignment constraint**
  in the doc comment — the L3 requirement (`lut_addr must be in L3
  memory and align 4 bytes`) is documented only for `gvml_lookup_32`
  / `gvml_lookup_4x32` / the `_interval` family. The 16-bit variant
  takes any uint16_t-aligned pointer.
- `lut_size` is the number of entries (not bytes). For a binary MAC
  the natural domain is `0..255` (popcount of a u8 byte index),
  so `lut_size = 256`. We use `256` even though only 16-bit keys with
  values in `[0,255]` index it — the API requires keys < lut_size,
  not all entries to be exercised.

### 2.2 The canonical caller pattern

`/usr/local/gsi-apu/.../lab_1b/dev_src/gsi_device_lab_1.c`:

```c
/* Need extra (void *) casting to ignore alignment warning */
gvml_lookup_16(vr_input, vr_idx, (const uint16_t *)demo_data->input,
               demo_data->input_len);
```

with the corresponding host-side cmd-struct field at
`lab_1b/gsi_device_lab_1.h:24`:

```c
struct gd_lab_1_lookup_demo_data {
    uint16_t input[8];     /* These elements will be spread over 32k elements */
    uint16_t input_len;    /* <= 8 */
    uint64_t output;
} __attribute__((packed));
```

The LUT is **inline in the cmd struct**. The host populates it before
`gdl_mem_cpy_to_dev(dev_cmd_buf, &cmd, cmd_buf_size)` (which copies
the entire cmd including the inline LUT into device memory). The
device-side kernel reads `demo_data->input` as a normal pointer; GVML
internally handles whatever caching/L3-fetching is needed.

This is exactly the shape we want. Our LUT is 512 B versus lab_1b's
16 B — well within the 64-byte cmd buffer plus the program_data union
(which the linker grows as needed; the struct is `__attribute__((packed))`
so no implicit padding fights us).

### 2.3 Why not the alternatives

- **(a) L1 via `gvml_load_16`.** Rejected. `gvml_load_16` reads a
  full 32K-element VR slot from `gvml_vm_reg`; there is no API to
  populate a 256-entry uint16_t array in L1 from VRs. We would have to
  DMA the LUT from L4 → L1 anyway, and then we *still* need a L4
  pointer to pass to `gvml_lookup_16` — the GVML API takes a C
  pointer, not a `gvml_vm_reg`. So routing through L1 buys nothing.
- **(b) L4 via a fourth `mem_hndl_*` role.** Rejected. The LUT is
  constant per program (popcount table for binary MAC, fixed at
  compile time) and tiny. Adding a fourth `mem_hndl_mac_lut` field
  would (i) require host.c to allocate a separate 512 B device buffer,
  (ii) require chained `gdl_add_to_mem_handle` arithmetic, (iii)
  surface a synthetic "input" role to `_role_io_table`, polluting
  argv. None of that is justified for a constant table half the size
  of one L4 cache line.
- **(c) Static `const uint16_t mac_lut[256] = { ... }` in `device.c`.**
  Rejected as the first-choice, but **acceptable as a fallback** if
  cmd-struct embedding hits an ARC compiler alignment issue. The
  table is constant; the ARC linker handles `const` arrays. The
  reason this is not the primary choice: future LUTs (e.g. s16 MAC
  partial products) will be data-dependent on workload dtype and
  cannot be hard-coded. Cmd-struct embedding is the forward-compatible
  path. Document this fallback in `_emit_device_c`'s comment block but
  do not implement it.

### 2.4 Why `mode="sv_lookup"` always wants a popcount LUT today

`spmw_autoschedule.py:301–337` only emits two candidates: `mode="sv"`
and `mode="sv_lookup"`. Cost-model in `spmw_cost_models.py:73` and
`test_target_apu_v1.py:138` pin `sv_lookup` as the binary-MAC win
(`(lookup 6 + add 2) * 16 = 128` cyc) — i.e. the only workload shape
the cost model knows about is the binary MAC. The MICRO '25 reference
implementation that this matches uses popcount(XOR) for binary
matmul. For now there is exactly one LUT identity, and it lives in
the build harness as a hard-coded `popcount16(k)` table.

Future work (T8 in `SPMW_ARCHITECTURE.md`, when an s16 MAC variant
ships): the build harness will need to dispatch on
`compiled.dtype` (or a `Placement.extra["lut_kind"]` key). That
upgrade slot is documented in §6 below; it is NOT part of this spec.

---

## 3. The fix

### 3.1 Cmd struct: add an inline LUT field

**File:** `experiments/allo/allo/spmw_apu_v1_build.py`
**Function:** `_emit_struct_h` (lines 157–191).

Inject one field into `struct program_data` after the existing
`mem_hndl_*` fields. The field name is fixed (`mac_lut`); the size is
fixed (`256`). The struct body becomes:

```c
struct program_data
{
    uint64_t mem_hndl_local_W;
    uint64_t mem_hndl_local_x;
    uint64_t mem_hndl_acc;
    uint16_t mac_lut[256];
} __attribute__((packed));
```

Implementation: after the `f"{body}\n"` line in the emitted text,
append a single canonical line:

```python
"        uint16_t mac_lut[256];\n"
```

unconditionally. Always emit the field — adding it costs 512 B per
cmd and is independent of whether the program uses sv_lookup. This
keeps the struct shape the same regardless of placement choice, so
the cmd struct's binary layout is stable across `mode="sv"` and
`mode="sv_lookup"` runs.

The cmd's outer envelope (`char buffer[64]; union { struct program_data data; }`)
is unchanged. The union grows to `sizeof(struct program_data) =
3 * 8 + 512 = 536` bytes; `gdl_mem_cpy_to_dev(dev_cmd_buf, &cmd,
sizeof(cmd))` already copies the whole thing.

### 3.2 Host: populate the LUT before `gdl_mem_cpy_to_dev`

**File:** `experiments/allo/allo/spmw_apu_v1_build.py`
**Function:** `_emit_host_c` (lines 339–552).

Inject a popcount-table initialisation block **immediately before** the
existing `ret = gdl_mem_cpy_to_dev(dev_cmd_buf, &cmd, cmd_buf_size);`
call (line 477). The block is plain C, unconditional:

```c
    /* SPEC-018: populate the MAC popcount LUT used by gvml_lookup_16
     * for the sv_lookup mode of binary MAC. The LUT lives inline in
     * cmd.data.mac_lut and is copied to the device with the rest of
     * the cmd struct. */
    for (unsigned k = 0; k < 256; ++k) {
        unsigned c = 0;
        for (unsigned b = 0; b < 8; ++b) if ((k >> b) & 1u) ++c;
        cmd.data.mac_lut[k] = (uint16_t)c;
    }
```

Emit this **always**, even when the program does not use sv_lookup.
The cost (one 256-iter loop, ~µs on the host CPU) is irrelevant
compared to the simulator/HW run time, and unconditional emission
keeps the host.c shape stable. The host writes into `cmd` (which is
a stack-allocated `struct program_cmd` per line 371), then
`gdl_mem_cpy_to_dev` copies the whole struct including the LUT into
the device-mapped buffer.

### 3.3 Device: declare `mac_lut_ptr` and use it in the corrected call

**File:** `experiments/allo/allo/spmw_apu_v1_build.py`
**Function:** `_emit_device_c` (lines 209–325).

Add one new line to the per-role `l4_decls` block, immediately after
the loop that emits `inp_L4ptr` / `wgt_L4ptr` / `out_L4ptr` decls
(line ~256). The decl is **always** emitted (parallel to the
unconditional struct field):

```c
    const uint16_t *mac_lut_ptr = (const uint16_t *)data->mac_lut;
```

Emit this line **after** the L4 ptr decls and **before** the
`vr_block` join. Use the same indentation style (4-space).

The `(const uint16_t *)` cast mirrors lab_1b's pattern (it suppresses
an ARC GCC alignment warning; `data` is `__attribute__((packed))` so
the field is byte-addressed). The cast is mandatory; do not drop it.

Implementation note: `_emit_device_c` already has a pattern for
"emit decls only when the body references the symbol" (see line 251,
`if ptr in ptr_to_field and (ptr in ctx_l4_roles or ptr == "out_L4ptr")`).
**Do NOT** apply that pattern to `mac_lut_ptr`. The body always
references it iff the body contains a `gvml_lookup_16` call; but the
codegen ctx emits unsuppressable `mac_lut_ptr` text into `self.cmds`
(see §3.4) only when `emit_mac_lookup` runs. The cleanest invariant:
unconditionally emit the decl, accept an `__attribute__((unused))`
warning for `mode="sv"` device.c. (ARC GCC's `-Wunused-variable` is
the relevant warning; if `Common/common.mk` treats it as -Werror,
add a `(void)mac_lut_ptr;` line after the decl. Coder should verify
empirically — see §5.3.)

### 3.4 Ctx: emit the 4-arg call

**File:** `experiments/allo/allo/spmw_codegen.py`
**Function:** `APUv1Ctx.emit_mac_lookup` (lines 883–898).

Change the one line at 897 from:

```python
self.cmds.append(f"gvml_lookup_16({tmp_name}, {x_n}, {y_n});")
```

to:

```python
self.cmds.append(
    f"gvml_lookup_16({tmp_name}, {x_n}, mac_lut_ptr, 256);"
)
```

The literal `"mac_lut_ptr"` is the canonical C name from §3.3; the
literal `256` matches the LUT length from §3.1. Both are baked into
the ctx as canonical names, matching the same shape as
`_L4_PTR_BY_ROLE` (`"inp_L4ptr"` / `"wgt_L4ptr"` / `"out_L4ptr"` —
inline literals, not parameters).

Note that the second arg switches from `{y_n}` to `{x_n}`. In the
3-arg call both operands were passed; in the 4-arg call there is
exactly one input VR (the index source), and `y` does not appear.
This is the correct semantics: `gvml_lookup_16(vdst, vsrc, lut, n)`
computes `vdst[i] = lut[vsrc[i]]`. For binary MAC the index encodes
both operands (`(x_byte << 8) | y_byte`), so the autoscheduler must
ensure the matcher's `x` VR has been pre-packed with the byte pair
before `MAC` fires. **That packing is out of scope for SPEC-018**:
the existing fixture passes the same VR (`target.vrs`) for both
`x` and `y` (see `test_target_apu_v1.py:99`, where
`mac.emit(vrs, vrs, vrs, ctx)`), so dropping `y_n` from the emitted
line does not change observable behaviour today — both names resolve
to `vrs` and `_name(y, "y")` is never the unique source of meaning.
Document the "x carries the packed (x,y) byte index" semantics in
the docstring update so the future regalloc pass knows what `x_n`
must contain.

Do **not** touch `emit_mac_mul_add` (lines 900–916). That path
selects `mode="sv"` (raw MUL+ADD) which has no LUT.

### 3.5 Update the docstring

The new `emit_mac_lookup` docstring replaces the existing one:

```python
def emit_mac_lookup(self, acc, x, y) -> None:
    """Expand MAC into the GSI sv-lookup pattern:

        gvml_lookup_16(<tmp>, <x>, mac_lut_ptr, 256);
        gvml_add_s16(<acc>, <acc>, <tmp>);

    Semantics: `<tmp>[i] = mac_lut_ptr[<x>[i]]`. The autoscheduler
    must pre-pack the byte-pair index into `x` before this MAC fires
    (today's matcher passes the same VR for `x` and `y`, so binary
    MAC must encode both operands into the `x` VR upstream of this
    call — out of scope for SPEC-018; a TODO for SPEC-018b).

    `mac_lut_ptr` resolves to a 256-entry `uint16_t` popcount table
    declared in the build harness's emitted `device.c`. The LUT lives
    inline in the cmd struct (`data->mac_lut`); see SPEC-018 §3.1–§3.3.
    Length is fixed at 256.

    The autoscheduler may reserve a scratch VR alias via
    `bind_handle("mac_tmp", "<alias>")`; otherwise the canonical name
    `mac_tmp_vr` is used. `y` is accepted for signature symmetry with
    `emit_mac_mul_add` but is not referenced — the byte pair is packed
    into `x` upstream.
    """
```

The `y` parameter stays in the signature (the fixture's emit lambda
calls `emit_mac_lookup(acc, x, y)` so the arg arity must not change).
We just stop using it.

---

## 4. File-level change list

Three files. No shared-Allo edits.

### 4.1 `experiments/allo/allo/spmw_codegen.py`

- Line 897: change the single `self.cmds.append(...)` line as in §3.4.
- Lines 884–892 (the docstring): replace per §3.5.
- Lines 813–815 (class-level docstring of `APUv1Ctx`): one-sentence
  tweak. Change "expands it into `gvml_lookup_16` + `gvml_add_s16`"
  to "expands it into `gvml_lookup_16(..., mac_lut_ptr, 256)` +
  `gvml_add_s16`" so the canonical pointer name appears once in the
  class header.

No other edits in `spmw_codegen.py`. **In particular, do not touch
the `mode="sv"` dispatch at lines 1302–1316**; that path stays.

### 4.2 `experiments/allo/allo/spmw_apu_v1_build.py`

- `_emit_struct_h` (lines 157–191): inject one line as in §3.1.
- `_emit_host_c` (lines 339–552): inject the popcount init block
  immediately before `gdl_mem_cpy_to_dev(dev_cmd_buf, ...)`, as in
  §3.2.
- `_emit_device_c` (lines 209–325): add one decl line and (if
  needed) a `(void)mac_lut_ptr;` guard, as in §3.3.

The three injections all live inside `f""`-templates; no signature
changes to any of these emitters. `gen_apu_v1_low_mode_project`
(lines 560+) is untouched.

### 4.3 `experiments/allo/tests/spmw/test_apu_v1_build_harness.py`

Tighten `test_emits_compilable_project_layout` (around line 129) so
the now-correct emit is regression-pinned. The current assertion is:

```python
assert "gvml_lookup_16" in dev_text or "gvml_load_16" in dev_text
```

Replace with:

```python
assert "gvml_lookup_16(" in dev_text, dev_text
assert "mac_lut_ptr, 256" in dev_text, dev_text
assert "const uint16_t *mac_lut_ptr" in dev_text, dev_text
```

and in the same test, after the existing `struct.h` assertions
(around line 117), add:

```python
assert "uint16_t mac_lut[256]" in sh_text, sh_text
```

and after the existing `host.c` assertions (around line 139), add:

```python
assert "cmd.data.mac_lut[k]" in host_text, host_text
```

These pin every emission point so a future regression touching any
of the three files trips one of these four asserts.

### 4.4 `experiments/allo/tests/spmw/test_target_apu_v1.py`

The existing
`test_apu_v1_ctx_emit_mac_expands_to_lookup_plus_add`
(lines 89–108) asserts only `ctx.cmds[0].startswith("gvml_lookup_16(")`.
Strengthen it with one new assertion at the end of the test:

```python
    # SPEC-018: the lookup call must carry the canonical LUT pointer
    # and the 256-entry length. These names are emitted as literals
    # by APUv1Ctx and must match the build-harness LUT decl.
    assert "mac_lut_ptr" in ctx.cmds[0], ctx.cmds[0]
    assert ", 256)" in ctx.cmds[0], ctx.cmds[0]
```

`test_apu_v1_ctx_emit_mac_honors_bind_handle` (lines 111–120) needs
no change — it asserts the `mac_tmp` alias path, which is orthogonal.

The cost-model test
`test_apu_v1_sv_lookup_beats_sv_mode` (lines 138–158) needs no
change — `_apu_v1_kernel_cycles` does not branch on
`mac_lut_ptr` presence; its `mode`-based dispatch is unchanged.

---

## 5. Verification

### 5.1 Unit-level

After implementation, all four of these must PASS in one run:

- `pytest experiments/allo/tests/spmw/test_target_apu_v1.py::test_apu_v1_ctx_emit_mac_expands_to_lookup_plus_add`
  — strengthened per §4.4.
- `pytest experiments/allo/tests/spmw/test_target_apu_v1.py::test_apu_v1_ctx_emit_mac_honors_bind_handle`
  — unchanged behaviour.
- `pytest experiments/allo/tests/spmw/test_target_apu_v1.py::test_apu_v1_sv_lookup_beats_sv_mode`
  — unchanged behaviour.
- `pytest experiments/allo/tests/spmw/test_apu_v1_build_harness.py::test_emits_compilable_project_layout`
  — strengthened per §4.3 (four new asserts).

### 5.2 Build-level

Run `pytest experiments/allo/tests/spmw/test_apu_v1_run_hw.py` (or
its end-to-end MLP equivalent). The build harness will:

1. Emit `device.c`, `host.c`, `struct.h`, `Makefile` into a temp dir.
2. Invoke `make` against `Common/common.mk`. Pre-fix, this fails with
   the GCC error "too few arguments to function 'gvml_lookup_16'".
   Post-fix, it must succeed.

The pass criterion: `result.cycles is not None and result.cycles > 0`
(per SPEC-001 §1.4). The make-exit-with-error → `RuntimeError`
promotion already exists (SPEC-001 §1.3); SPEC-018 is the change that
flips that RuntimeError from "raised" to "not raised".

### 5.3 ARC compiler warning audit

The new `const uint16_t *mac_lut_ptr` decl is **always** emitted but
only **sometimes** referenced (only in `mode="sv_lookup"` builds; the
`mode="sv"` path emits `emit_mac_mul_add` which never names
`mac_lut_ptr`). If ARC GCC's default warning set includes
`-Wunused-variable` and `Common/common.mk` treats warnings as errors,
this trips. Coder must check empirically:

```bash
grep -n "Werror\|-W" /usr/local/gsi-apu/13.7.1/.../Common/common.mk
```

If `-Werror` is in effect, append `(void)mac_lut_ptr;` immediately
after the decl in `_emit_device_c`. If not, document via comment
that the decl is unused-in-sv-mode and is intentional. Default
recommendation: emit the `(void)mac_lut_ptr;` guard unconditionally
— a single C statement, no observable cost.

### 5.4 Regression: upstream Allo tests

This task touches `spmw_codegen.py` (the APU v1 ctx, lines 883–898)
and `spmw_apu_v1_build.py` (three emitters). Neither is reached by
any FPGA or AIE upstream test:

- `tests/test_vhls.py`, `tests/test_vitis.py`, `tests/test_pynq.py`,
  `tests/test_xls.py`, `tests/test_catapult_hls.py`, `tests/test_nn.py`
  — none import `APUv1Ctx` or `spmw_apu_v1_build`.
- `tests/customize/`, `tests/dataflow/` — none reach SPMW code.

The change is purely additive to APU v1 codegen and has no blast
radius outside `spmw_*` files.

---

## 6. Future-work slots (NOT in scope for SPEC-018)

- **SPEC-018b: byte-pair packing of `(x, y)` into the lookup index VR.**
  Today `emit_mac_lookup` drops `y`. The MICRO '25 reference
  implementation packs `(x_byte << 8) | y_byte` into a single u16
  before calling `gvml_lookup_16` (the LUT is then 65536 entries of
  `popcount16(k)` indexed by the packed pair). That packing is a
  GVML expression that should live one autoschedule level up (as a
  pre-MAC bit-pack op), not inside `emit_mac_lookup`. The current
  popcount(u8)-only LUT is correct only when the matcher upstream
  produces a u8 index in `x` — true for the existing fixture
  (`test_apu_v1_run_hw.py`) by construction. SPEC-018b is the task
  that wires the byte-pair pre-pack into the autoschedule + ctx.
- **SPEC-018c: variable-dtype LUT identity.** When an s16 MAC
  variant ships (the LUT is `i * j` for signed 16-bit, sized
  ~131K × 2 B = 256 KB), the inline cmd-struct path no longer
  fits. The build harness must dispatch on `compiled.dtype` (or
  `Placement.extra["lut_kind"]`) and route s16 LUTs through a
  separate L4 buffer with its own `mem_hndl_mac_lut` field. Trigger
  to start SPEC-018c: a workload appears whose `MatchedOp.dtype` is
  not `u8` / `u16` for the MAC.
- **SPMW_ARCHITECTURE.md §T11.** Add a new open tension after this
  spec lands: "LUT identity is implicit in the build harness today
  (always popcount(u8)). When more than one LUT kind exists,
  `Placement.extra` should carry the LUT identity, and the build
  harness should dispatch on it." This is the architectural seam
  SPEC-018c will cut.

---

## 7. Receipt for orchestrator

LUT lives **inline in the `program_cmd` struct** as a 256-entry
`uint16_t mac_lut[256]` field (no separate L4 buffer, no L1 DMA);
**host** populates it once with `popcount(u8)` before
`gdl_mem_cpy_to_dev`; the corrected emit is
`gvml_lookup_16(mac_tmp_vr, vrs, mac_lut_ptr, 256);` where
`mac_lut_ptr` is a `const uint16_t *` decl added unconditionally in
`_emit_device_c`. Three files change
(`spmw_codegen.py`, `spmw_apu_v1_build.py`, two test files); zero
shared-Allo edits.
