# SPEC-005 — Samsung: drive `pim_driver` from the emitted PIMCmd stream

Status: design / ready for coder
Owner: architect
Related: HANDOFF Blocker 5; `spmw_codegen.py:1302–1434`; report 16.

---

## 1. Problem

`_run_samsung` (in `experiments/allo/allo/spmw_codegen.py`, lines
1302–1434) currently throws away `compiled.cmds`. It scans the
emitted `PIMCmd` list only to *recognise* a kernel (`"MAC"` → GEMV,
`"MUL"` → MUL, `"ADD"` → ADD, else RELU), then invokes
`pim_driver --op <kernel>`. The C++ side regenerates the canonical
CRF microcode via `PIMCmdGen::getPIMCmds(ktype, ...)` inside
`PIMKernel::executeEltwise` /`executeGemv` and runs the kernel.

Because the actual emitted CRF program is discarded:

- Two `compile_for_target` invocations with different `layout=` arguments
  produce identical `pim_driver --op GEMV` invocations, so cycles do
  not move. The autoscheduler's `Placement` is observationally inert
  on Samsung.
- Any Tenon-side change to the CRF program (e.g. fewer MACs because
  the layout shrinks the inner loop, or extra MOVs to stage to a
  different GRF bank) cannot be measured.

The fix has two halves:

a. **Codegen carries placement info into the emitted PIMCmd stream.**
   Specifically: the *JUMP loopCounter* (= `num_tile − 1`) must be a
   function of how much work each compute kernel covers, which in
   turn depends on placement (bank tiling, GRF tiling). Today
   `SamsungCtx` emits no JUMP at all and emits a fixed body — placement
   already changes the dst/src parity slots but cannot yet change the
   trip count. **This spec scopes only the run-side plumbing**; the
   codegen-side change is followed up in `needs-arch-MMM` (see §8).

b. **`pim_driver` accepts the emitted PIMCmd stream and uses it as
   the actual CRF program**, so the cycle count it reports reflects
   what Tenon emitted rather than what `PIMCmdGen` hardcodes. This
   spec covers half (b).

## 2. Option chosen — A: extend `pim_driver` with `--cmds <file>`

Investigated both options. Recommendation: **Option A**, add a
`--cmds <file>` flag to `pim_driver` that reads a serialised list
of `PIMCmd` records and uses it as the CRF microcode passed to
`programCrf()` instead of calling `PIMCmdGen::getPIMCmds(...)`.

### Why not Option B (Python-accessible entry point)

PIMSimulator already builds a static + shared `libdramsim2`
library (`Sconstruct` lines 89–103). In principle Python could
ctypes into it. Reasons not to:

- The simulator's lifecycle (`MultiChannelMemorySystem` constructor
  needs an ini file, working directory, and a callback-driven event
  loop via `runPIM()`) is non-trivial to drive from ctypes without
  re-implementing the kernel scaffolding currently in
  `PIMKernel::executeEltwise` and `executeGemv`.
- The whole point of `pim_driver` (added for this project per the
  comment in `pim_driver.cc:18–20`) is to be the boundary between
  Python and the simulator. Re-crossing that boundary via ctypes
  duplicates effort.
- Build risk: switching the simulator's library to be loaded by
  Python pulls gtest, pthread, and DRAMSim2's globals into the
  Python process — past experience on this server (PIMSimulator
  needs gtest 1.14.0 from conda) suggests this is fragile.

The simpler change is to keep the subprocess boundary and broaden
the CLI. The existing `--op <kernel>` path stays as a back-compat
fallback for tests that don't have a PIMCmd stream (and for the
`cycles=None / no kwargs` SPEC-001 §6 escape on line 1349).

### Why JSON, not binary or PIMCmd::toInt() ints

`PIMCmd::toInt()` already exists (`PIMCmd.cpp:100`) and would give
us a single uint32 per command. Two reasons not to use it as the
wire format:

- The Python `PIMCmd` dataclass stores opcode/opd as **strings**
  (`type_: str`, `dst_: str`, etc — see `spmw_codegen.py:37–58`).
  Round-tripping through bit-packing requires the Python side to
  duplicate the enum-int mappings declared in `PIMCmd.h`. A JSON
  wire format keeps the strings as-is and asks the C++ side to
  parse them with the existing `cmdToStr` / `opdToStr` inverses.
- JSON is human-greppable. The trace is short (tens to low hundreds
  of cmds), so size is not a concern. A failing test on cycle
  parity will be debuggable by `cat`-ing the trace file.

## 3. Wire format

A single JSON object per `pim_driver --cmds` invocation:

```json
{
  "cmds": [
    {"type": "MAC",  "dst": "GRF_B", "src0": "GRF_A", "src1": "EVEN_BANK",
     "is_auto": 1, "dst_idx": 0, "src0_idx": 0, "src1_idx": 0},
    {"type": "JUMP", "loop_counter": 7, "loop_offset": 2},
    {"type": "NOP",  "loop_counter": 7},
    {"type": "EXIT"}
  ]
}
```

Field mapping (Python `PIMCmd` dataclass → JSON key → C++
`DRAMSim::PIMCmd` member):

| Python field   | JSON key       | C++ field          | Default if missing |
|----------------|----------------|--------------------|--------------------|
| `type_`        | `type`         | `type_`            | required            |
| `dst_`         | `dst`          | `dst_`             | `"A_OUT"`           |
| `src0_`        | `src0`         | `src0_`            | `"A_OUT"`           |
| `src1_`        | `src1`         | `src1_`            | `"A_OUT"`           |
| `src2_`        | `src2`         | `src2_`            | `"A_OUT"`           |
| `loopCounter_` | `loop_counter` | `loopCounter_`     | `0`                 |
| `loopOffset_`  | `loop_offset`  | `loopOffset_`      | `0`                 |
| `isAuto_`      | `is_auto`      | `isAuto_`          | `0`                 |
| `dstIdx_`      | `dst_idx`      | `dstIdx_`          | `0`                 |
| `src0Idx_`     | `src0_idx`     | `src0Idx_`         | `0`                 |
| `src1Idx_`     | `src1_idx`     | `src1Idx_`         | `0`                 |
| `isRelu_`      | `is_relu`      | `isRelu_`          | `0`                 |

JSON keys are snake_case without the trailing underscore — the
trailing-underscore convention is a C++ style choice that does not
need to leak through the wire.

Opcode strings recognised on the C++ side (mirror of `cmdToStr` in
`PIMCmd.h:208–233`): `EXIT`, `NOP`, `JUMP`, `FILL`, `MOV`, `ADD`,
`MUL`, `MAC`, `MAD`. Unknown opcodes → `die("unknown cmd type")`.

Operand strings (mirror of `opdToStr`): `A_OUT`, `M_OUT`,
`EVEN_BANK`, `ODD_BANK`, `GRF_A`, `GRF_B`, `SRF_M`, `SRF_A`.
Unknown operands → `die("unknown opd type")`.

Library choice on the C++ side: pull in a single-header JSON parser
**only if one is already available**. Check `experiments/simulators/PIMSimulator/lib/`
first; if `nlohmann/json.hpp` or similar is not present, **do not add
a third-party dependency**. Instead, use the minimal hand-rolled
parser approach: the JSON shape is fixed, so a flat scan with
`std::regex` (already a C++14 feature) or a 30-line tokenizer over
`std::ifstream` suffices. Concretely, a line-delimited variant works:

```
MAC dst=GRF_B src0=GRF_A src1=EVEN_BANK is_auto=1
JUMP loop_counter=7 loop_offset=2
NOP loop_counter=7
EXIT
```

This is easier to parse without a library. **If the coder elects to
go this route, the spec endorses it.** Either format is acceptable
provided field semantics match the table above; the line-delimited
form is the recommended default to keep the C++ change small.

The coder picks one of {JSON, line-delimited} and pins the choice
in `_run_samsung`'s docstring. The Python writer is ~15 lines either
way.

## 4. Files that change

### 4.1 PIMSimulator (C++)

**`experiments/simulators/PIMSimulator/src/pim_driver.cc`** — additive.
Lines to touch:

- `~77` (variable declarations): add `std::string cmds_path;`
- `~82–93` (argparse loop): add `else if (a == "--cmds") cmds_path = next();`
- New helper above `main`: `std::vector<PIMCmd> load_cmds(const std::string& path)`
  — parses the trace file into a `vector<PIMCmd>` using the
  string-form constructors of `DRAMSim::PIMCmd`. ~60 lines.
- New helper: `PIMCmdType parseCmdType(const std::string& s)` and
  `PIMOpdType parseOpdType(const std::string& s)` — inverse maps of
  `cmdToStr`/`opdToStr` from `PIMCmd.h`. ~25 lines each (case-by-case).
- In each kernel branch (ADD/MUL at `~102`, RELU at `~127`, GEMV at
  `~148`): if `cmds_path` is non-empty, **bypass `executeEltwise`/
  `executeGemv`** and inline an equivalent sequence that calls
  `programCrf(loaded_cmds)` instead of letting
  `PIMKernel::executeEltwise` regenerate them. The minimum-invasive
  pattern is to add a public `PIMKernel::executeEltwiseWithCmds(...)`
  / `executeGemvWithCmds(...)` that takes a `vector<PIMCmd>` and
  skips the `getPIMCmds` call. See §4.2.
- `num_tile` recovery: when a custom cmd stream is supplied, the
  driver must still know how many tile iterations to loop the data
  addresses over (the C++ side issues `addTransactionAll` per tile
  in `computeAddOrMul`). The JUMP `loop_counter` field carries
  `num_tile - 1` (see `PIMKernel.cpp:503`), so the driver extracts
  it: scan loaded cmds for the first `JUMP`, set
  `num_tile = jump.loopCounter_ + 1`. If no JUMP is present, fall
  back to deriving it from `--n` like today.

Total new C++ in `pim_driver.cc`: ~150 lines (parser + helpers +
branching).

**`experiments/simulators/PIMSimulator/src/tests/PIMKernel.h`** — additive.

- After line 96 (`executeEltwise` declaration), add the two new public
  methods:
  - `void executeEltwiseWithCmds(int dim, pimBankType, KernelType, int, int, int, std::vector<PIMCmd>& cmds);`
  - `void executeGemvWithCmds(NumpyBurstType* w, NumpyBurstType* i, bool is_tree, std::vector<PIMCmd>& cmds);`

**`experiments/simulators/PIMSimulator/src/tests/PIMKernel.cpp`** —
additive. Two new method bodies after `executeEltwise` (lines 499–526
today) and `executeGemv` (lines ~364):

- `executeEltwiseWithCmds` is a copy of `executeEltwise` (~28 lines)
  with the single difference at the current line 504: instead of
  `vector<PIMCmd> pim_cmds = PIMCmdGen::getPIMCmds(...)`, accept
  `cmds` as the parameter. The `num_tile` derivation stays the same;
  it must agree with what's encoded in the JUMP.
- `executeGemvWithCmds` is the symmetric copy of `executeGemv`.

Adding new methods (rather than mutating the existing `executeEltwise`)
preserves the FPGA/AIE/upstream gtest tests in `KernelTestCases.cpp`
that exercise the canonical PIMCmdGen path.

**`experiments/simulators/PIMSimulator/Sconstruct`** — no change. The
new `pim_driver.cc` lines and new `PIMKernel.{h,cpp}` methods are
picked up by the existing `getSources("driver")` glob (line 39).

### 4.2 spmw_codegen.py (Python)

**`experiments/allo/allo/spmw_codegen.py`** — additive in
`_run_samsung` (lines 1302–1434):

- After the `_pimsim_root() / "pim_driver"` existence check, keep
  the current `op_types` kernel inference (it still determines which
  numpy inputs to load and which `--op` to pass).
- After serialising `W`/`x`/`a`/`b` to .npy, write
  `compiled.cmds` (filtered to `PIMCmd` instances) to
  `td_path / "cmds.txt"` in the format chosen in §3. ~20 lines.
- Append `argv += ["--cmds", str(cmds_path)]` whenever
  `compiled.cmds` contains at least one `PIMCmd`. If the cmd list
  is empty (no `@allo.work` body emitted any PIMCmds — currently
  possible only in degenerate tests), skip the `--cmds` arg so the
  legacy `getPIMCmds` path is used and tests continue to pass.

### 4.3 No changes outside these files

- No edits to `experiments/allo/allo/` files other than
  `spmw_codegen.py`.
- No edits to shared Allo files (`allo/ir/`, `allo/dataflow.py`,
  `allo/customize.py`). The blast radius is confined to the
  Samsung backend and PIMSimulator. The architect's shared-file
  ledger (`SPMW_ARCHITECTURE.md`) is **not touched** by this spec.

## 5. Rebuild story

`pim_driver` must be rebuilt because `pim_driver.cc`, `PIMKernel.h`,
and `PIMKernel.cpp` change. Procedure:

```sh
cd experiments/simulators/PIMSimulator
scons -j32
```

The `Sconstruct` builds three targets in one pass: `sim` (gtest
binary), `pim_driver`, and `libdramsim2.{a,so}`. The
`getSources("driver")` glob (line 39) automatically picks up
`pim_driver.cc` and any new lines added to it; the new
`PIMKernel.{h,cpp}` methods are picked up via the
`Glob(tests/*.cpp)` on line 43. No `Sconstruct` edit needed.

Rollback: revert `pim_driver.cc`, `PIMKernel.h`, `PIMKernel.cpp` and
re-run `scons -j32`. The legacy `--op <kernel>` path still works
because the `--cmds` flag is additive.

Upstream gtest tests gated:
`experiments/simulators/PIMSimulator/src/tests/KernelTestCases.cpp` —
specifically the GEMV/ADD/MUL/RELU cases that call `executeEltwise`
and `executeGemv` directly. Because we add new `*WithCmds` methods
rather than mutating the existing ones, these tests must continue
to pass. CI command: `scons -j32 && ./sim --gtest_filter=*Kernel*`.

## 6. Concrete verification — does placement change cycles?

Test file: `experiments/allo/tests/spmw/test_samsung_placement_changes_cycles.py`
(new). Skeleton:

```python
import pytest
from allo import compile_for_target, match_workload
from allo.spmw_autoschedule import Placement
# ... import the Samsung target + a small GEMV workload ...

@pytest.mark.skipif(not _pim_driver_present(), reason="PIMSimulator not built")
def test_layout_changes_samsung_cycles():
    workload = _build_gemv_workload(M=4096, K=1024)
    target  = _samsung_target()
    trace   = match_workload(workload, target)

    # Two distinct placements: layout_a puts the y accumulator on
    # GRF_A; layout_b puts it on GRF_B. The PIMCmd stream's dst/src
    # parity changes accordingly, and JUMP loopCounter remains the
    # same so this is a microcode-pattern delta, not a tile-count delta.
    layout_a = _make_placement(target, acc_register="grf_a")
    layout_b = _make_placement(target, acc_register="grf_b")

    cycles_a = compile_for_target(target, trace, layout=layout_a).run(
        W=W_data, x=x_data
    ).cycles
    cycles_b = compile_for_target(target, trace, layout=layout_b).run(
        W=W_data, x=x_data
    ).cycles

    # Both placements run. Both report a positive cycle count. The
    # cmd streams differ (verifiable by also asserting compiled_a.cmds
    # != compiled_b.cmds), but cycles may or may not differ depending
    # on which register file the GRF read/write contention picks up.
    assert cycles_a is not None
    assert cycles_b is not None
    assert cycles_a > 0 and cycles_b > 0
```

The minimum assertion is the cmds-are-different test (this gates
that codegen carries placement into PIMCmds at all). A
stronger-but-flakier follow-up assertion is `cycles_a != cycles_b`;
**that assertion is only valid once §8 lands**, because today the
PIMCmd JUMP loop counter is constant — placement affects only the
EVEN/ODD bank parity slots, which `programCrf` issues at the same
DRAM cycle cost. We accept the weaker assertion for this task and
log the stronger one as a follow-up gated by §8.

Concrete weaker assertion the coder must land:

```python
assert compiled_a.cmds != compiled_b.cmds
assert cycles_a is not None and cycles_b is not None
```

This proves: (a) PIMCmd stream is now flowing into the simulator
(`cycles_a is not None` confirms the run path), and (b) the codegen
already distinguishes placements at the cmd-stream level.

## 7. Coder implementation summary (one paragraph)

In `_run_samsung` (`spmw_codegen.py:1302`), after the existing
numpy-input serialisation, write `[c for c in compiled.cmds if
isinstance(c, PIMCmd)]` to a `cmds.txt` file in the temp dir using
the line-delimited format from §3, and append `--cmds <path>` to
`argv` when the list is non-empty. In `pim_driver.cc`, add a
`--cmds` flag, a `load_cmds(path)` parser that returns
`std::vector<DRAMSim::PIMCmd>`, and inverse-string helpers
`parseCmdType` / `parseOpdType`. In `PIMKernel.{h,cpp}`, add public
`executeEltwiseWithCmds` and `executeGemvWithCmds` methods that
take a pre-built `vector<PIMCmd>` and call `programCrf` on it
instead of `PIMCmdGen::getPIMCmds`. In each kernel branch of
`pim_driver.cc`'s `main`, dispatch to `*WithCmds` when `cmds_path`
is non-empty; otherwise keep the current `executeEltwise` /
`executeGemv` call. Rebuild with `scons -j32` in
`experiments/simulators/PIMSimulator/`. Land the test from §6.

## 8. Followup: codegen-side change (out of scope here)

For placement to move cycles (not just dst/src parity), SamsungCtx
must emit a `JUMP` whose `loopCounter_ = num_tile − 1`, where
`num_tile` is derived from how much vector work the placement
assigns to each tile. The current SamsungCtx (`spmw_codegen.py:180`)
emits no JUMP at all. That codegen work is a separate task and
should be filed as `needs-arch-MMM-samsung-jump-from-placement.task`
**after** the present spec lands and the §6 weaker assertion is
green. The stronger assertion `cycles_a != cycles_b` becomes the
acceptance gate for the followup.

## 9. Memory entry

Architect should log to
`.claude/agent-memory/architect/2026-05-18-samsung-placement-driven.md`:

- Decision: Option A (extend CLI), reject Option B (ctypes/shared lib).
- Why: simulator lifecycle is non-trivial to drive from Python; CLI
  boundary is the project's stable interface.
- Constraint discovered: PIMCmd alone does not carry tile count — it
  carries the JUMP loop counter, and the C++ side derives `num_tile`
  from that. Future codegen must emit JUMP with the right
  `loopCounter_` for placement to affect cycles.
- Followup task ID: `needs-arch-MMM-samsung-jump-from-placement`.

## 10. Implemented (coder, 2026-05-18)

- C++: `pim_driver.cc` gained `--cmds <file>` (line-delimited parser
  per §3 alt format), `parseCmdType` / `parseOpdType` helpers, JUMP-
  based `num_tile` recovery, and `use_cmds`-gated dispatch through new
  `PIMKernel::executeEltwiseWithCmds` / `executeGemvWithCmds` methods
  (added in `PIMKernel.{h,cpp}` alongside the originals — the canonical
  paths are untouched, gtest `PIMKernelFixture.gemv` still passes).
- Python: `_run_samsung` now serialises every `PIMCmd` in
  `compiled.cmds` to `cmds.txt` in the temp dir and appends
  `--cmds <path>` to the driver argv. The `--op` flag is still chosen
  by an opcode scan but no longer drives the CRF microcode (comment
  pinned in source). A run-side filter drops `MOV` / `FILL` cmds with
  `dst in {EVEN_BANK, ODD_BANK}` and `src in {GRF_A, GRF_B}` to
  satisfy `PIMCmd::validationCheck`'s ISA-1.0 rules — flagged here as
  a SamsungCtx codegen leak (the ST_A/ST_B storeback moves currently
  live in `compiled.cmds` even though bank stores are issued at the
  DRAM-controller level, not via CRF MOV). The proper fix is in
  SamsungCtx and is part of §8's followup.
- Test: `experiments/allo/tests/spmw/test_samsung_placement_changes_cycles.py`
  with three cases: cmds-differ + both-run-positive for a bank-parity
  swap of the W memref (EVEN_BANK vs ODD_BANK), a direct CLI sanity
  check that `pim_driver --cmds` accepts the canonical GEMV trace,
  and a static grep on `spmw_codegen.py` proving the CRF-source
  comment is present.
- Verified: `test_e2e_mlp_samsung` (cycles ~91k for the canonical MLP
  GEMV trace via --cmds), `test_codegen_gemv`, and the new test file
  all green. Upstream `PIMKernelFixture.gemv` gtest still green.
- Coder ambiguity flagged: SamsungCtx emits `MOV ODD_BANK <- GRF_B`
  as the ST_B move; this is rejected by ISA validation. The spec
  authorized only run-side plumbing, so I added a CRF-validity filter
  in `_run_samsung` rather than touching SamsungCtx. Architect should
  decide whether the filter stays or migrates into SamsungCtx as part
  of the §8 followup.
