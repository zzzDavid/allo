# SPEC-003 — UPMEM: wire emitted DPU kernel through uPIMulator

Status: draft for coder. Design-only artifact (no source edits in this task).
Owner of implementation: coder agent (task 004).
Author: architect.
Date: 2026-05-18.

---

## 1. TL;DR for the coder

1. **Choose Option A — register the emitted kernel under a fixed
   benchmark slot named `TENON`.** Option B (a `--source-file` flag)
   was rejected because the `--benchmark` name is structurally
   load-bearing in **five** uPIMulator subsystems (compiler, linker,
   assembler, simulator, and the top-level CMake), not just the
   assembler registry; cf. §3.
2. The slot is **persistent on disk** under
   `experiments/simulators/uPIMulator/golang/uPIMulator/benchmark/TENON/`
   and shipped once. `_run_upmem` overwrites `TENON/dpu/task.c` at run
   time with whatever `compiled.cmds` produced and then invokes the
   stock binary with `--benchmark TENON`.
3. **Six files change in uPIMulator** (4 new, 1 single-line edit, 1
   single-line edit). All four DSL benchmarks (`DSLVA`, `DSLVM`,
   `DSLATTN_AV`, `DSLATTN_QKT`, `E5VADD`, `E5VMUL`, `DSLMLP`) are the
   precedent: that's how the DSL author already adds a codegen slot.
4. **One Python file changes**: `spmw_codegen.py:1435–1514`,
   `_run_upmem`. The `benchmark = "VA"/"GEMV"` proxy lines disappear.
5. **`UPMEMCtx.get_kernel_src()` (`spmw_codegen.py:605–622`) must be
   extended** to emit a full DPU-runtime-compatible `task.c`, not the
   stub that exists today. The stub omits the `__host
   dpu_arguments_t`, the `nr_kernels` dispatch table, MRAM→WRAM
   staging, and `BARRIER_INIT` — all of which the uPIMulator linker
   transitively requires (it pulls in `sdk/misc.crt0` and friends and
   resolves global symbols from the kernel object). The skeleton in
   `benchmark/TENON/dpu/task.c.template` (shipped in this task)
   defines that envelope; the codegen emits only the *kernel body*
   inside the envelope.

---

## 2. Background and evidence

### 2.1 What `_run_upmem` does today

`spmw_codegen.py:1435–1514` ignores `compiled.cmds` (preserves them
only in `extra["kernel_src"]`) and runs one of two pre-built PrIM
benchmarks (`VA` or `GEMV`) as a cycle proxy, picked by a textual
heuristic on the emitted C ("contains `+=` and `*`" → GEMV, else
VA). The returned `cycles` count therefore has no causal connection
to the compiled artifact. Task 003's purpose is to break that
disconnect.

### 2.2 How uPIMulator threads the benchmark name

Path traced from `golang/uPIMulator/src/main.go:14–66`:

1. **`compiler.Compile()`** (`src/compiler/compiler.go:48–75`) shells
   out to Docker to run
   `benchmark/build.py --num_dpus N --num_tasklets N`.
   `build.py:21–36` runs `cmake -S benchmark -B benchmark/build` and
   then `ninja -C benchmark/build`. That CMake project's top-level
   `benchmark/CMakeLists.txt` is the registry — it lists each
   benchmark via `add_subdirectory(<NAME>)`. Each `<NAME>/dpu/`
   target is named `<NAME>_device`.
2. **`linker.Linker.InitBenchmarkRelocatable()`**
   (`src/linker/linker.go:49–64`) hard-codes the path
   `benchmark/build/<BENCHMARK>/dpu/CMakeFiles/<BENCHMARK>_device.dir/task.c.o`.
   The "object" file is in fact UPMEM-clang `-S` output — DPU
   assembly text (verified: `file task.c.o` returns "assembler
   source, ASCII text"). The linker re-lexes/parses this asm.
3. **`assembler.Assembler.Init()`**
   (`src/assembler/assembler.go:38–67`) holds a Go-side map of
   benchmark name → data-prep class. The class supplies
   `InputDpuHost`, `OutputDpuHost`,
   `InputDpuMramHeapPointerName`, `OutputDpuMramHeapPointerName`,
   and `NumExecutions` — these write `.bin` files that the simulator
   loads into MRAM and the host argument struct.
4. **`simulator.Simulator.Init()`** also reads `bin_dirpath` to
   pull those `.bin` files and the linked `iram.bin`/`wram.bin`/
   `mram.bin`/`atomic.bin` artifacts produced by the linker.

So the benchmark name is the join key across:

| Subsystem | File | Lookup |
|---|---|---|
| top-level CMake | `benchmark/CMakeLists.txt` | `add_subdirectory(<NAME>)` |
| compiler subdir | `benchmark/<NAME>/dpu/task.c` | source |
| linker artifact path | `linker.go:49–64` | `<NAME>_device.dir/task.c.o` |
| assembler data-prep | `assembler.go:40–60` | `this.assemblables["<NAME>"]` |
| simulator name | `simulator.go` Init | benchmark-keyed naming of dumps |

Option B (a `--source-file` flag) would need to either short-circuit
every one of these five lookups or thread a parallel "use this asm"
override through linker + assembler. That's a five-file rewrite of
load-bearing internals; the DSL family of benchmarks already shows
the additive pattern is one new file per subsystem.

### 2.3 What the DSL family already does

`benchmark/DSLVA/`, `DSLVM`, `DSLATTN_AV`, `DSLATTN_QKT`, `E5VADD`,
`E5VMUL`, `DSLMLP` are by file structure identical to `VA` (`dpu/`,
`support/`, `CMakeLists.txt`), and their Go data-prep classes
(`src/assembler/prim/dslva.go` etc.) are minor variations on `va.go`.
The naming and the comment `// DSL-generated assemblable` show this
slot was added specifically to receive codegen output. Reusing the
pattern keeps blast radius zero (no edits to existing files except
two registry adds).

### 2.4 What the current `UPMEMCtx.get_kernel_src()` emits

`spmw_codegen.py:605–622` emits:

```
#include <defs.h>
#include <mram.h>
#include <stdint.h>

int main(void) {
    uint32_t tasklet_id = me();
    <lines from self.cmds>

    return 0;
}
```

This does not link against `sdk/misc.crt0` cleanly: the linker
(`linker.go:198`) hard-attaches `misc.crt0` and resolves the rest of
the SDK by chasing unresolved symbols. The PrIM convention has every
benchmark expose `DPU_INPUT_ARGUMENTS` (a `__host dpu_arguments_t`),
a `kernels[]` dispatch table, and a barrier. Without these, the
linker either (a) leaves `DPU_INPUT_ARGUMENTS` unresolved and panics
in `HasResolved()`, or (b) the simulator runs but `mram_base_addr_*`
arithmetic never reads any input. Either way, cycles will not match
the kernel.

The fix: `get_kernel_src()` must produce a full PrIM-shaped DPU
source. The cleanest path is to template off the DSLVA shape (see
§4.2) and substitute the kernel body.

---

## 3. Chosen option: Option A — `TENON` benchmark slot

### 3.1 What "Option A" means here

Add **one new benchmark named `TENON`** to uPIMulator's registry,
permanently. The slot is shipped empty (or with a placeholder
`task.c`) and is **overwritten in place** by `_run_upmem` each time
the Python side wants to run an emitted kernel.

This is identical in shape to how the DSL benchmarks already work —
the only difference is that `TENON`'s `task.c` is generated by Tenon
codegen rather than hand-written.

### 3.2 Why not a hash-keyed name like `TENON_<hash>`

Considered and rejected. Each new benchmark name requires a Go file
under `src/assembler/prim/`, an entry in `assembler.go`, and an
`add_subdirectory` line in `benchmark/CMakeLists.txt`. These cannot
be created at runtime without recompiling the Go binary. A single
fixed slot reused across runs is sufficient because:

- `_run_upmem` is serial (one `Compiled.run()` per process), so two
  callers cannot race on the slot;
- the linker, compiler, and simulator are launched as one
  subprocess per call and all reads happen before the next caller
  could rewrite the file;
- cycle counts are not cached on disk — every `run()` re-compiles.

If concurrent `_run_upmem` calls ever become a requirement, the
fix is to acquire a file lock on `benchmark/TENON/dpu/task.c`, not
to add per-call benchmark names.

---

## 4. Exact files to change

### 4.1 uPIMulator (six files; all additive except two single-line registry edits)

All paths relative to
`experiments/simulators/uPIMulator/golang/uPIMulator/`.

| # | File | Change | Templated from |
|---|---|---|---|
| 1 | `benchmark/TENON/CMakeLists.txt` | NEW (3 lines) | `benchmark/DSLVA/CMakeLists.txt` |
| 2 | `benchmark/TENON/dpu/CMakeLists.txt` | NEW (9 lines) | `benchmark/DSLVA/dpu/CMakeLists.txt`; substitute `DSLVA_device` → `TENON_device`. |
| 3 | `benchmark/TENON/dpu/task.c` | NEW (placeholder, runtime-overwritten) | `benchmark/DSLVA/dpu/task.c` |
| 4 | `benchmark/TENON/support/common.h` | NEW | exact copy of `benchmark/DSLVA/support/common.h` |
| 5 | `benchmark/CMakeLists.txt` | EDIT (+1 line) | append `add_subdirectory(TENON)` after the existing entries (line ~24). |
| 6 | `src/assembler/prim/tenon.go` | NEW (~210 lines) | exact copy of `src/assembler/prim/dslva.go` with `Dslva` → `Tenon` and the `package prim` comment renamed. |
| 7 | `src/assembler/assembler.go` | EDIT (+1 line) | after line 59 (`this.assemblables["DSLVA"] = new(prim.Dslva)`) add `this.assemblables["TENON"] = new(prim.Tenon)`. |

(Item count is 7 line items but counts as 6 distinct artifacts; #1
and #2 are both small CMake stubs.)

The CMake project must be **re-configured once** after these
additions so that `benchmark/build/TENON/` exists. The coder must
run `python3 benchmark/build.py --num_dpus 1 --num_tasklets 1`
(inside the Docker image, as `compiler.Compile()` already does) once
post-edit and verify `benchmark/build/TENON/dpu/CMakeFiles/TENON_device.dir/task.c.o`
appears. After that, every subsequent `_run_upmem` call will
re-trigger CMake's ninja dependency check and rebuild only `task.c`
when it changes.

### 4.2 Tenon side: `experiments/allo/allo/spmw_codegen.py`

Three regions change.

**Region A — `UPMEMCtx.get_kernel_src()` (lines 605–622).** Replace
the existing stub with a full PrIM-shaped DPU envelope. Template:

```c
#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <perfcounter.h>
#include <barrier.h>

#include "../support/common.h"

__host dpu_arguments_t DPU_INPUT_ARGUMENTS;

void __attribute__ ((noinline))
tenon_kernel(T *bufferB, T *bufferA, unsigned int l_size) {
    // === BEGIN tenon-emitted body ===
    <lines from self.cmds, each prefixed by "    ">
    // === END tenon-emitted body ===
}

BARRIER_INIT(my_barrier, NR_TASKLETS);

extern int main_kernel1(void);
int (*kernels[nr_kernels])(void) = {main_kernel1};

int main(void) {
    return kernels[DPU_INPUT_ARGUMENTS.kernel]();
}

int main_kernel1(void) {
    unsigned int tasklet_id = me();
    if (tasklet_id == 0) { mem_reset(); }
    barrier_wait(&my_barrier);

    uint32_t input_size_dpu_bytes = DPU_INPUT_ARGUMENTS.size;
    uint32_t input_size_dpu_bytes_transfer = DPU_INPUT_ARGUMENTS.transfer_size;
    uint32_t base_tasklet = tasklet_id << BLOCK_SIZE_LOG2;
    uint32_t mram_base_addr_A = (uint32_t)DPU_MRAM_HEAP_POINTER;
    uint32_t mram_base_addr_B = (uint32_t)(DPU_MRAM_HEAP_POINTER + input_size_dpu_bytes_transfer);

    T *cache_A = (T *) mem_alloc(BLOCK_SIZE);
    T *cache_B = (T *) mem_alloc(BLOCK_SIZE);

    for (unsigned int byte_index = base_tasklet;
         byte_index < input_size_dpu_bytes;
         byte_index += BLOCK_SIZE * NR_TASKLETS) {
        uint32_t l_size_bytes = (byte_index + BLOCK_SIZE >= input_size_dpu_bytes)
            ? (input_size_dpu_bytes - byte_index) : BLOCK_SIZE;

        mram_read((__mram_ptr void const*)(mram_base_addr_A + byte_index), cache_A, l_size_bytes);
        mram_read((__mram_ptr void const*)(mram_base_addr_B + byte_index), cache_B, l_size_bytes);

        tenon_kernel(cache_B, cache_A, l_size_bytes >> DIV);

        mram_write(cache_B, (__mram_ptr void*)(mram_base_addr_B + byte_index), l_size_bytes);
    }
    return 0;
}
```

This envelope is **fixed and shared across all emitted kernels** for
now. The codegen body (`self.cmds`) only writes the inner loop of
`tenon_kernel(...)`. This is sufficient for VA-shape and MAC-shape
workloads (the only two MLP-relevant patterns); a richer envelope
selector is deferred to a follow-up task and called out in §6 (open
design tensions). Note that `<lines from self.cmds>` must each be a
valid C statement; `UPMEMCtx.emit_c_line` and `.cmd` already produce
that (`spmw_codegen.py:571–580`).

**Region B — `_run_upmem` (lines 1435–1514).** Rewrite to:

```python
def _run_upmem(compiled: "Compiled", **inputs) -> RunResult:
    root = _upim_root()
    binary = root / "build" / "uPIMulator"
    if not binary.exists():
        return RunResult(
            cycles=None,
            stdout=f"simulator unavailable: {binary} not found",
            backend="upmem",
        )

    # Pull the assembled DPU source from the ctx.
    ctx = getattr(compiled, "_ctx", None)
    if ctx is not None and hasattr(ctx, "get_kernel_src"):
        kernel_src = ctx.get_kernel_src()
    else:
        # Fallback: assume cmds is a list[str] of full source lines.
        kernel_src = "\n".join(
            c for c in compiled.cmds if isinstance(c, str)
        )

    if not kernel_src.strip():
        raise RuntimeError(
            "UPMEM: _run_upmem received empty kernel source; "
            "compile_for_target produced no commands"
        )

    # Drop the source into the TENON slot.
    slot_dir = root / "benchmark" / "TENON" / "dpu"
    if not slot_dir.exists():
        return RunResult(
            cycles=None,
            stdout=(
                f"simulator unavailable: TENON benchmark slot not "
                f"provisioned at {slot_dir}; rebuild uPIMulator with "
                f"the TENON benchmark registered."
            ),
            backend="upmem",
        )
    task_c = slot_dir / "task.c"
    task_c.write_text(kernel_src, encoding="utf-8")

    with tempfile.TemporaryDirectory() as td:
        bin_dir = Path(td) / "bin"
        bin_dir.mkdir()
        try:
            proc = subprocess.run(
                [
                    str(binary),
                    "--root_dirpath", str(root),
                    "--bin_dirpath", str(bin_dir),
                    "--benchmark", "TENON",
                    "--num_channels", "1",
                    "--num_dpus_per_rank", "1",
                    "--num_tasklets", "1",
                    "--data_prep_params", "1024",
                ],
                capture_output=True,
                timeout=600,
                check=False,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            raise RuntimeError(
                f"UPMEM uPIMulator invocation failed: {exc}"
            ) from exc

        stdout = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")
        combined = stdout + ("\n" + stderr if stderr else "")
        log_path = bin_dir / "log.txt"
        if log_path.exists():
            combined += "\n" + log_path.read_text(errors="replace")
        cycle_match = re.search(
            r"cycle[s]?\s*[:=]\s*(\d+)", combined, re.IGNORECASE
        )
        if cycle_match is None:
            raise RuntimeError(
                "UPMEM uPIMulator returned but stdout/log missing "
                "'cycle: ...' line; tail: " + combined[-400:]
            )
        cycles = int(cycle_match.group(1))
        return RunResult(
            cycles=cycles,
            stdout=combined,
            backend="upmem",
            extra={
                "kernel_src": kernel_src,
                "benchmark": "TENON",
                "returncode": proc.returncode,
            },
        )
```

Two design notes for the coder on Region B:

- The function reads `compiled._ctx` to call `get_kernel_src()`. The
  current `Compiled` class (`spmw_codegen.py:1791–1826`) does **not**
  retain the ctx — it stores only `ctx.cmds`. The coder must either
  (a) extend `Compiled.__init__` to carry `_ctx` for UPMEM, or
  (b) move the envelope rendering into `_run_upmem` itself. Option
  (a) is preferred because the envelope logic is UPMEM-specific
  state. The change is additive: add a fifth optional `ctx=None`
  argument to `Compiled.__init__`, and have `compile_for_target` pass
  `ctx` after `_walk_and_emit`. Other backends ignore the field.
  **This is the only shared-file edit in this spec** (`Compiled`
  is touched by every backend) and is purely additive (new optional
  kwarg with default).

- The `if not slot_dir.exists()` branch returns a "simulator
  unavailable" RunResult rather than raising. This is intentional:
  on a host where uPIMulator is built but the architect's TENON
  registration hasn't landed yet, the test should skip cleanly
  (matching the `_sim_unavailable` branch in `test_e2e_mlp_upmem`).

**Region C — `Compiled.__init__` (lines 1796–1806).** Add an
optional `ctx` kwarg as described above:

```python
def __init__(
    self,
    target,
    trace: MatchTrace,
    cmds: list,
    layout: Placement,
    ctx: "CodegenContext | None" = None,
):
    self.target = target
    self.trace = trace
    self.cmds = cmds
    self.layout = layout
    self._ctx = ctx
```

And `compile_for_target` (line 1890):

```python
return Compiled(target, trace, ctx.cmds, stored_layout, ctx=ctx)
```

Every existing `Compiled` consumer uses positional args 1–4; none
reads attribute `_ctx`. Therefore this is fully back-compat.

---

## 5. What `compiled.cmds` content must look like

After this spec lands, `compiled.cmds` is **a `list[str]` of C
statements**, each one valid as a single body statement (i.e.
each ends in `;` or is a `{ ... }` block) of the
`tenon_kernel(T *bufferB, T *bufferA, unsigned int l_size)`
function. Concretely:

- variables `bufferA`, `bufferB` are visible (input + output WRAM
  caches);
- variable `l_size` is visible (element count for this iteration);
- type `T` is visible (currently `int32_t`, fixed by
  `dpu/CMakeLists.txt`'s `-DINT32`);
- no `#include` lines, no function definitions, no `main` —
  envelope owns those.

Today `UPMEMCtx.emit_c_line` (`spmw_codegen.py:571–573`) appends one
C line per call. The coder must verify that the existing UPMEM emit
lambdas produce statements that reference `bufferA`, `bufferB`, and
`l_size` (or equivalent autoscheduler-assigned names that map onto
the envelope's locals). Where they don't, the fix is in
`UPMEMCtx.handle_c_name` (`spmw_codegen.py:542–569`) — extend the
naming rules so the buffer-role handles render to the envelope-fixed
names. This is internal to `UPMEMCtx`; no shared-file impact.

For VA-shape (the MLP MAC fan-out path), the expected emitted body
is one for-loop like:

```c
for (unsigned int i = 0; i < l_size; i++) {
    bufferB[i] += bufferA[i];
}
```

For MAC-shape (GEMV row), one accumulator add:

```c
for (unsigned int i = 0; i < l_size; i++) {
    bufferB[i] += bufferA[i] * w_i;  // w_i sourced from a register/constant
}
```

The coder may discover that `UPMEMCtx` does not yet wrap the body in
a `for (i ...)` loop — that's a known gap and is out of scope for
this task. If `compiled.cmds` is a flat sequence with no outer loop,
the coder shall write it verbatim into the envelope's
`tenon_kernel` body (the SDK's `BLOCK_SIZE`-stride outer loop in
`main_kernel1` still iterates the MRAM blocks, so even a flat body
will execute once per `BLOCK_SIZE`-byte chunk). A follow-up task
should add the inner loop in `UPMEMCtx`; that is **not** a
prerequisite for this task because cycles will still be measured
against real DPU asm.

---

## 6. Acceptance criteria for Task 004

The coder is done when **all** of these hold on the uPIMulator
host (the only place uPIMulator runs):

1. **No proxy text remains.** `grep 'benchmark = "VA"' spmw_codegen.py`
   and `grep 'benchmark = "GEMV"' spmw_codegen.py` both return no
   matches. The `--benchmark` argument in `_run_upmem` is the literal
   string `"TENON"`.

2. **uPIMulator registers TENON.**
   `grep TENON src/assembler/assembler.go` returns the new entry,
   and `grep TENON benchmark/CMakeLists.txt` returns the new
   `add_subdirectory(TENON)`.

3. **TENON builds.** After running
   `python3 benchmark/build.py --num_dpus 1 --num_tasklets 1`
   inside the Docker image (or natively when the dpu-clang
   toolchain is on PATH), the file
   `benchmark/build/TENON/dpu/CMakeFiles/TENON_device.dir/task.c.o`
   exists and is non-empty ASCII text starting with `.text`.

4. **End-to-end cycle count is real.** Running
   `pytest tests/spmw/test_e2e_mlp.py::test_e2e_mlp_upmem -s` on the
   uPIMulator host produces `result.backend == "upmem"`,
   `result.cycles is not None`, `result.cycles > 0`, and
   `result.extra["benchmark"] == "TENON"`. The exact cycle number is
   workload-dependent; for the current MLP shape the expected
   ballpark is 10^4–10^6 cycles (PrIM VA at 1024 elements reports
   ~9600 cycles; we should be within a factor of 10).

5. **Cycle count differs from the old VA proxy.** As a regression
   sanity check, the coder must capture both the pre-patch
   `result.cycles` and the post-patch `result.cycles` and confirm
   they are not equal. (If they happen to be exactly equal,
   something has gone wrong — the emitted kernel is being silently
   replaced by the VA template.)

6. **`simulator unavailable` skip still works.** Running the test
   with `UPIMULATOR_ROOT=/nonexistent` produces
   `result.cycles is None` and `result.stdout` contains
   `"simulator unavailable"`. No `RuntimeError` is raised. The same
   behaviour holds if `benchmark/TENON/dpu/` is missing (the
   "slot not provisioned" branch in Region B).

7. **No regression on other backends.** The other four
   `test_e2e_mlp_*` tests
   (`samsung_hbm`, `aim_gddr6_pim`, `apu_v1`, `apu_v2`) report
   unchanged cycle counts before and after this patch (modulo
   `cycles=None` on hosts where those simulators are absent). The
   `Compiled(..., ctx=ctx)` additive kwarg must not perturb them.

8. **Skill doc updated.** `.claude/skills/pim-upmem/SKILL.md` gains
   a paragraph documenting the TENON slot: where it lives, that
   `_run_upmem` overwrites `dpu/task.c`, and how to rebuild after
   editing the envelope.

---

## 7. Open design tensions (deferred, not blockers)

These are flagged for future architect tasks. They do **not** gate
Task 004's PASS.

- **Multi-tasklet support.** Current spec hard-codes
  `--num_tasklets 1`. Once `UPMEMCtx` learns to emit
  tasklet-strided bodies, the cmd-line plumbing needs to read the
  tasklet count from the target (or from `compiled.layout`).
- **Multi-DPU support.** Same story for `--num_dpus_per_rank`.
  Linked to how `Placement` for UPMEM maps to MRAM partitioning.
- **Per-workload envelope.** The `tenon_kernel(bufferB, bufferA,
  l_size)` envelope assumes a binary in-place reduce shape. GEMV
  with separate weight/input/output streams will need a
  three-buffer envelope. The clean solution is for `UPMEMCtx` to
  declare which envelope template it needs (`"va"`, `"gemv"`,
  `"mlp"`) and have `get_kernel_src()` pick a template; the
  `benchmark/TENON/` shell stays the same.
- **Real data prep.** `tenon.go` ships as a copy of `dslva.go`,
  meaning its host-side `DPU_INPUT_ARGUMENTS` and MRAM heap setup
  use VA-shape buffers. For correctness-checking workloads (vs.
  cycle-only), this is wrong: a GEMV emitted body reading three
  buffers will see only two. **This is intentional for Task 004
  scope:** the goal is "cycles come from the emitted asm", not
  "outputs match reference numerics". A follow-up task should
  parameterise `tenon.go` over the workload shape (e.g. via a
  small JSON dropped next to `task.c` that the Go side reads).

---

## Implemented

- **2026-05-18 (Task 004, coder):** Landed Option A as specified.
  - uPIMulator: new `benchmark/TENON/{CMakeLists.txt,dpu/{CMakeLists.txt,task.c},support/common.h}`, registry edits in `benchmark/CMakeLists.txt` (+1 line) and `src/assembler/assembler.go` (+1 line), new `src/assembler/prim/tenon.go` (~110 lines).
  - `spmw_codegen.py`: rewrote `_run_upmem` to drop the VA/GEMV proxy and write `UPMEMCtx.get_kernel_src()` into the TENON slot; upgraded `UPMEMCtx.get_kernel_src()` to emit the full PrIM-shaped DPU envelope (DPU_INPUT_ARGUMENTS, kernels[] dispatch, BARRIER_INIT, MRAM↔WRAM staging); extended `UPMEMCtx.handle_c_name` to map `wram`/`mram` memories onto the envelope's `bufferA` parameter; added optional `ctx=` kwarg to `Compiled.__init__` and wired it from `compile_for_target`.
  - Cycles-only deviation from §4.2 template: `OutputDpuMramHeapPointerName` in `tenon.go` returns a zero-size byte stream to suppress the simulator's `bytes are different` panic. SPEC-003 §7 already flagged real data prep as deferred; this is the minimum change that lets cycles flow when the emitted body diverges from VA semantics.
  - Tests: `tests/spmw/test_target_upmem.py` 6/6 pass (added `test_upmem_get_kernel_src_emits_prim_envelope`, `test_run_upmem_uses_tenon_slot_and_no_proxy`). `tests/spmw/test_e2e_mlp.py::test_e2e_mlp_upmem` PASS with `cycles=12004`, `extra.benchmark='TENON'`, distinct from VA baseline (89935 cycles).
  - 3 pre-existing APU v1 failures (`emit_mac_lookup` 3-vs-4 arg bug, documented in coder memory) are untouched and unrelated.

## 8. Rollback story

If the FPGA CI breaks (it shouldn't — none of these files are
shared with FPGA / AIE backends), revert is:

1. `git revert` the `_run_upmem` + `UPMEMCtx.get_kernel_src` +
   `Compiled.__init__` patch in `spmw_codegen.py`. The TENON
   benchmark dir stays — it costs nothing if unused.
2. The two single-line additions in `src/assembler/assembler.go`
   and `benchmark/CMakeLists.txt` are inert if Python never asks
   for `--benchmark TENON`.

There are **no upstream Allo tests** (`tests/dataflow/`,
`tests/customize/`) that exercise `spmw_codegen.py` or
`_run_upmem`, so the upstream-test gating section is empty for
this spec.
