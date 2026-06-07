# SPEC-020 — Samsung HBM-PIM: measure the actual MLP, not a padded substitute

Status: draft (architect-issued 2026-05-19)
Owner of implementation: coder
Blast-radius class: backend-local; Samsung-only run path; zero shared-file changes; zero `KernelTestCases.cpp` semantic changes.

## 1. The bug, stated precisely

`tests/spmw/test_e2e_mlp.py::test_e2e_mlp_samsung` currently substitutes the
MLP inputs with a zero-filled `W[4096, 1024]` / `x[1024]` before calling
`compiled.run(...)`. The 223,048 cycle number that
`MLP_PERFORMANCE_REPORT.md` reports is for that 4.2M-MAC standin, not for
the 49,152-MAC MLP the autoscheduler compiled. The test comment claims
"pim_driver requires the minimum tile shape (M=4096, K=1024)". **That
claim is false.** There is no structural minimum in the driver or the
simulator (see §2 below).

The task is to make the cycle number reflect the workload `compile_for_target`
actually compiled.

## 2. Investigation — there is no driver-side shape floor

Files audited (all under
`/work/shared/users/phd/nz264/spmw-for-pim/experiments/simulators/PIMSimulator/`):

- `src/pim_driver.cc:178-323` (the `main()` flat-argparse + dispatch).
- `src/tests/PIMKernel.cpp:249-515`
  (`getResultColGemv`, `preloadGemv`, `executeGemv`, `executeGemvWithCmds`,
  `computeGemv`).
- `src/Burst.h:355-477` (`NumpyBurstType::loadFp16`, `bShape`,
  `loadTobShape`, `getTotalDim`).

### 2.1 The driver itself rejects nothing for GEMV

`pim_driver.cc:287-313` validates only that `--weight`, `--in`,
`--output-dim` and `--input-dim` are present; there is no
`output_dim >= MIN` or `input_dim >= MIN` guard. The eltwise path has
`n >= 131072` semantics but only as a divisor (`int nbursts = n / 16`);
the GEMV path does not even use that.

### 2.2 The simulator tile math degrades gracefully via `ceil()`

Quoting `PIMKernel.cpp:366-367`:

```cpp
int num_output_tiles = ceil(((double)w_data->bShape[0] / (num_total_pim_blocks_)) / num_grfB_);
int num_input_tiles  = ceil((double)w_data->bShape[1] / (double)num_grfA_);
```

With `num_total_pim_blocks_ = 64 ch × 16 banks / 2 = 512` and
`num_grfB_ = 8`, one output-tile spans `4096` rows; with `num_grfA_ = 8`
one input-tile spans `8 × 16 = 128` columns. For the MLP layer dims:

| Layer | M   | K   | bShape[0] | bShape[1]=K/16 | num_output_tiles | num_input_tiles | num_batch (x.bShape[0]) |
|-------|-----|-----|-----------|----------------|------------------|-----------------|-------------------------|
| L1    | 256 | 128 | 256       | 8              | 1                | 1               | 1 (if x is shape (1,128)) |
| L2    | 64  | 256 | 64        | 16             | 1                | 2               | 1 (if h is shape (1,256)) |

Every loop in `executeGemv` / `preloadGemv` / `computeGemv` runs at
least once; `readResult` iterates `for (int x = 0; x < output_dim; x += num_grf_=16)`
which is fine for `M=256` (16 iters) and `M=64` (4 iters). No assert,
no abort, no shape-dependent indexing fault.

### 2.3 One degenerate corner in CRF microcode, but not for our cmd stream

`PIMKernel.cpp:397-400` computes
`num_jump_of_odd_bank = num_grfB_ * floor(num_input_tiles/2) - 1`.
With `num_input_tiles = 1`, that gives `0*8 - 1 = -1`, which would be
wrong if the driver re-emitted CRF via `PIMCmdGen::getPIMCmds`. But
since SPEC-005 `_run_samsung` passes `--cmds`, this entire code path is
bypassed — the CRF is the SamsungCtx-emitted PIMCmd stream, which the
cost model already prices correctly via `_emit_inner_loop_jump`
(`spmw_codegen.py:1037-1063`) using the per-match inner-loop bound from
`MatchedOp.enclosing_loops`. Layer 1 emits `loop_counter = 128/8 - 1 = 15`,
layer 2 emits `loop_counter = 256/8 - 1 = 31`. Both are well-formed.

### 2.4 What the canonical PIMSimulator test uses, and why 4096×1024 looks special

`src/tests/KernelTestCases.cpp:30-31` and `data/gemv/gen_gemv.py:5-8`
hardwire `DIM_OUT=4096, DIM_IN=1024` because that's `num_total_pim_blocks_ × num_grfB_` exactly one output tile and `128` input tiles. It is the
**fixture-canonical shape**, not a hardware-imposed minimum. The
"structural floor" referenced in `test_e2e_mlp.py:107-108` does not
exist in the source.

### 2.5 But there *is* a real `_run_samsung` bug — the 1D-input shape

`spmw_codegen.py:1410-1415` calls `np.save(x_path, x)` where the test
passes `x` as a 1D `(K,)` array. `Burst.h:395-398` then runs
`loadFp16` on shape `[K]`, which produces `bShape = [ceil(K/16)]` —
**a single-dim shape**. `executeGemv` reads `num_batch = i_data->bShape[0]`,
so `num_batch = ceil(K/16)` instead of `1`. The result: the GEMV inner
loop iterates `num_batch` times.

For the existing padded case (`x.shape = (1024,)`): `num_batch =
ceil(1024/16) = 64`. The 223,048 cycle figure is therefore not even
honest for a single 4096×1024 GEMV — it is **64 GEMVs of 4096×1024**.
The MLP-performance-report theoretical baseline (8,192 cyc) is off by
roughly the same factor, which is why the headline efficiency of 3.67%
looks pessimistic. (Canonical files in `data/gemv/gen_gemv.py:34`
explicitly save inputs as `batch_in.T` → shape `(1, 1024)` to avoid this.)

This shape bug must be fixed regardless of option (a) or (b).

## 3. Option evaluation

### Option (a) — extend `pim_driver` / `_run_samsung` to handle sub-tile shapes

What it actually requires (in priority order):

1. **Fix the 1D-x save in `_run_samsung`** to write `(1, K)` not `(K,)`.
   One-line change at `spmw_codegen.py:1411`:
   `x = np.asarray(x, dtype=np.float16).reshape(1, -1)`.
2. **Multi-layer cmds split + per-layer driver invocation.** `pim_driver`
   today accepts **one** GEMV per invocation (`--weight`, `--in`,
   `--output-dim`, `--input-dim`). The MLP emits one flat
   `compiled.cmds` containing two MAC blocks. To measure the actual
   MLP we must split the emitted stream at the MAC-group boundary and
   invoke the driver twice (once per layer), summing the
   `PIM_CYCLES total=` numbers. No C++ change.
3. **Drop the 4096×1024 padding in the test.** Call
   `compiled.run(layers=[{"W": W1, "x": x}, {"W": W2, "x": h}])` (new
   kwargs shape — see §5).

Files touched (Python only):
- `experiments/allo/allo/spmw_codegen.py` (`_run_samsung`).
- `experiments/allo/tests/spmw/test_e2e_mlp.py` (`test_e2e_mlp_samsung`).
- `experiments/allo/tests/spmw/_fixtures.py` — unchanged (MLP shape stays
  256×128 + 64×256).
- `experiments/allo/tests/spmw/test_samsung_placement_changes_cycles.py`
  — left alone (it uses 4096×1024 explicitly as a single-GEMV regression;
  do not touch under this spec).
- `MLP_PERFORMANCE_REPORT.md` — caveat 1 rewritten (see §6).

No C++ files change. No shared `allo/` files change. The `Compiled.run`
signature stays backwards-compatible (the existing `W=, x=` form continues
to work for single-layer regressions; the new `layers=` form is opt-in).

### Option (b) — reshape the MLP fixture to clear the (nonexistent) floor

Considered and rejected. Rationale:

- The "floor" is fictional (§2). There is nothing to clear.
- Even if we wanted bigger MLP layers for orthogonal reasons (more MACs
  → more amortisation, larger theoretical baseline), changing
  `_fixtures.py` MLP dims to e.g. 4096×1024 + 1024×4096 would still
  require option (a) anyway because the driver cannot iterate two
  layers in one invocation. Option (b) does not subsume option (a).
- A bigger MLP would change which K-bounds the cost model sees but not
  which placement argmin picks: at K=128, K=256, K=1024 the
  bank_row vs grf_staged ratio is constant (`grf_staged / bank_row =
  K*4 / ((K/8)*4 + 1) → ~8×` for any K). Argmin stays on bank_row.
  So changing dims would not "defeat the purpose" of measuring the MLP
  — but it would *replace* the workload of interest. The user task
  describes the workload `_fixtures.py` already defines; changing the
  workload to fit the test harness is the wrong direction.

### Decision: Option (a). Fix the run path; keep the fixture.

## 4. Spec — what the coder implements

### 4.1 `_run_samsung` new contract

The current single-GEMV invocation path stays as a back-compat branch.
A new code path activates when **either**:
- the caller passes `layers=[…]` kwarg (a list of dicts, each
  `{"W": np.ndarray, "x": np.ndarray}`), OR
- the emitted `compiled.cmds` contains more than one MAC-bounded group
  (autodetected) AND the caller passed exactly one `W`/`x` pair, in
  which case raise `ValueError("Samsung: multi-MAC cmd stream needs
  layers=[...] kwarg; got single W/x")` so the test author cannot
  accidentally fall back to single-layer averaging.

#### Pseudocode (in `_run_samsung`, replacing lines ~1387-1545):

```python
# Split compiled.cmds into per-MAC-group sublists.
def _split_by_mac(cmds: list[PIMCmd]) -> list[list[PIMCmd]]:
    """Group cmds so each group ends at the JUMP that closes one
    MAC's inner-K fold. A group is `[setup ... MAC ... JUMP]`."""
    groups, cur = [], []
    for c in cmds:
        cur.append(c)
        if c.type_ == "JUMP":
            groups.append(cur)
            cur = []
    if cur:                       # trailing non-JUMP tail (no fold)
        groups.append(cur)
    return groups

groups = _split_by_mac(_crf_valid_cmds)
n_macs = sum(1 for g in groups for c in g if c.type_ == "MAC")
multi = n_macs > 1

if multi:
    layers = inputs.get("layers")
    if layers is None:
        return RunResult(
            cycles=None,
            stdout="Samsung: multi-MAC cmd stream needs layers=[...] kwarg",
            backend="samsung_hbm_pim",
        )
    if len(layers) != len(groups):
        raise ValueError(
            f"Samsung: {len(groups)} MAC groups in cmd stream but "
            f"{len(layers)} layers= entries"
        )
    total_cycles = 0
    combined_stdout = []
    for grp, layer_in in zip(groups, layers):
        cyc, out = _run_samsung_one(driver, root, grp, layer_in)
        total_cycles += cyc
        combined_stdout.append(out)
    return RunResult(
        cycles=total_cycles,
        stdout="\n--- next layer ---\n".join(combined_stdout),
        backend="samsung_hbm_pim",
    )
else:
    # Existing single-GEMV path, unchanged except for the 1D-x fix.
    ...
```

`_run_samsung_one(driver, root, cmd_subset, layer_input) -> (cycles, stdout)`
is a new private helper that:
- Writes the per-layer `W.npy`, `x.npy` (with the shape fix:
  `x.reshape(1, -1)` for 1D `x`).
- Writes the per-layer `cmds.txt`.
- Invokes `pim_driver --op GEMV --weight ... --in ... --out ... --output-dim M --input-dim K --cmds <path>`.
- Parses `PIM_CYCLES total=<N>` and returns it.

### 4.2 The 1D-x fix (single-layer back-compat branch)

Replace `spmw_codegen.py:1411`:
```python
x = np.asarray(x, dtype=np.float16)
```
with
```python
x = np.asarray(x, dtype=np.float16)
if x.ndim == 1:
    x = x.reshape(1, -1)
```

This change is also applied inside `_run_samsung_one`. The Samsung
canonical fixture (`data/gemv/gen_gemv.py:34`) stores `(1, K)` — we are
matching that contract.

### 4.3 Test rewrite

`tests/spmw/test_e2e_mlp.py::test_e2e_mlp_samsung` becomes:

```python
import numpy as np

target = build_samsung_target()
workload = build_mlp_workload()
trace, compiled = _pipeline(target, workload)

macs = trace.by_target_op("MAC")
assert len(macs) >= 2, f"Expected >=2 MAC matches, got {len(macs)}"

# Per-layer inputs — actual MLP shapes (no padding).
W1 = np.zeros((256, 128), dtype=np.float16)
x  = np.zeros(128,          dtype=np.float16)
W2 = np.zeros((64, 256),    dtype=np.float16)
h  = np.zeros(256,          dtype=np.float16)
result = compiled.run(layers=[
    {"W": W1, "x": x},
    {"W": W2, "x": h},
])
assert isinstance(result, RunResult)
assert result.backend == "samsung_hbm_pim"
if not _sim_unavailable(result):
    assert result.cycles is not None and result.cycles > 0
else:
    assert result.cycles is None
```

(Numerical correctness is not asserted; zero inputs are fine because
the spec is measuring cycles for the workload that
`compile_for_target` emitted.)

### 4.4 `test_run.py` back-compat check

The existing `test_run_returns_runresult_for_all_backends` calls
`compiled.run()` with no kwargs and expects `cycles=None` from the
multi-layer Samsung path. The new code path should keep that contract:
when `layers=None` AND `multi=True`, return
`RunResult(cycles=None, stdout="…needs layers= kwarg", ...)`. When
`layers=None` AND `multi=False` AND `W=None`, the existing
`cycles=None` escape at `spmw_codegen.py:1399-1409` continues to apply
unchanged.

## 5. Success criterion

The verifier runs `pytest tests/spmw/test_e2e_mlp.py::test_e2e_mlp_samsung`
and gets a `result.cycles` that:

1. Is positive (existing).
2. Is the **sum** of two `PIM_CYCLES total=` values printed by two
   `pim_driver` invocations (new).
3. Corresponds to GEMVs of shape `(256, 128)` and `(64, 256)` — the
   shapes `compile_for_target` actually compiled — visible in the
   driver stdout's `--output-dim` / `--input-dim` echo.
4. Is **substantially smaller** than the previous 223,048 figure
   (because we are no longer running a `64 × 4096 × 1024` GEMV).
   Plausible range: O(10²-10³) cycles per layer's setup overhead +
   the inner-K MAC counts (16 + 32 folded MACs at 4 cyc each = 192
   compute cycles plus preload + readback overhead).

`MLP_PERFORMANCE_REPORT.md` then reports the actual MLP measurement
instead of the standin.

## 6. `MLP_PERFORMANCE_REPORT.md` caveat resolution

After SPEC-020 lands, the Samsung section is rewritten:

- "Workload run by sim" column changes from
  `4096x1024 GEMV (4.2M MAC)` to
  `MLP layers L1[256x128] + L2[64x256] (49,152 MAC)`.
- "Theoretical cyc" becomes `49,152 / 512 = 96 cycles` (same baseline
  as the other backends).
- "Measured cyc" is the new value reported by the post-SPEC-020 test.
- The historical 3.67%-of-peak number for the 4M-MAC standin is moved
  to a "Historical (pre-SPEC-020) data point" footnote so it stays
  searchable.
- Caveat 1 ("The Samsung number is for a 4096x1024 GEMV, not the MLP")
  is **deleted** — it no longer applies. The two-instruction-trace
  caveats for AiM and UPMEM (caveats 2, 3) are unchanged; they belong
  to separate spec slots (T8a, T8b in `SPMW_ARCHITECTURE.md`).

## 7. Files modified by this spec

| File | Change | Diff size | Justification |
|---|---|---|---|
| `experiments/allo/allo/spmw_codegen.py` | `_run_samsung` grows multi-layer branch; 1D-x reshape fix | ~50 lines added, 0 lines semantically changed | Backend-local; Samsung-only. Single-layer call path identical. |
| `experiments/allo/tests/spmw/test_e2e_mlp.py` | Use real MLP shapes via `layers=[…]` kwarg | ~10 lines | Test correctness fix. |
| `MLP_PERFORMANCE_REPORT.md` | Replace Samsung row + caveat 1 | ~20 lines | Documentation. |

No C++ changes. No `KernelTestCases.cpp` change. No `allo/dataflow.py`
/ `allo/ir/*` / `allo/customize.py` change. Upstream FPGA / AIE tests
are unaffected.

## 8. Regression guard

Existing tests that must stay green:
- `tests/spmw/test_codegen_gemv.py` — single-GEMV cmd-emission;
  doesn't call `compiled.run`. Unaffected.
- `tests/spmw/test_samsung_placement_changes_cycles.py::test_samsung_layout_changes_cmd_stream_and_runs`
  — uses single-layer (`bmatmul`-shape) GEMV with `M_drv=4096, K_drv=1024`.
  After SPEC-020 it still has exactly 1 MAC in the emitted cmd stream
  (single `@allo.work`), so `multi=False`, the single-layer back-compat
  branch runs unchanged. **Must keep green.**
- `tests/spmw/test_samsung_placement_changes_cycles.py::test_samsung_pim_driver_accepts_cmds_flag`
  — calls `pim_driver` directly; doesn't go through `_run_samsung`.
  Unaffected.
- `tests/spmw/test_run.py::test_run_returns_runresult_for_all_backends`
  — calls `compiled.run()` with no kwargs; expects `cycles=None`. After
  SPEC-020 the MLP workload's multi-MAC stream + missing `layers=` kwarg
  produces `cycles=None` with stdout `"…needs layers= kwarg"`. Test
  contract is "cycles is None when sim path can't infer inputs", which
  this satisfies. **Must keep green.**

## 9. Rollback story

The change is a pure additive code path in `_run_samsung` plus a test
edit. Reverting is `git revert <single commit>`; nothing in the
shared-file boundary or the C++ simulator changed. The PIMSimulator
binary is unchanged. FPGA / AIE CI cannot regress because zero
upstream files were touched.

## 10. Open questions deferred

None. Spec is committed.

## 11. Implemented

- 2026-05-19 (coder, task 008): `experiments/allo/allo/spmw_codegen.py`
  grew `_write_samsung_cmds` + `_run_samsung_one` helpers and a
  multi-layer branch in `_run_samsung` that splits cmds at JUMP
  boundaries and invokes `pim_driver` once per MAC group; the 1D-x
  `(K,) -> (1, K)` reshape fix lands in both the single-layer
  back-compat path and `_run_samsung_one`. `tests/spmw/test_e2e_mlp.py`
  now passes the actual MLP layer shapes via `layers=[...]`. New
  static test file `tests/spmw/test_samsung_multi_layer_split.py`
  asserts the multi-MAC-without-layers / wrong-layer-count contracts
  and that single-layer traces still hit the legacy W/x path. No C++
  changes. `MLP_PERFORMANCE_REPORT.md` Samsung row marked _pending
  post-SPEC-020 verifier run_; pre-SPEC-020 223,048-cycle figure moved
  to a historical-data-point footnote.
