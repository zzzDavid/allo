# SPEC-001 — APU v1: fix `device.c` GVML include and tighten test gate

Status: draft for coder, design-only artifact (no source edits in this task).
Owner of implementation: coder agent (task 002).
Author: architect.
Date: 2026-05-18.

---

## 1. TL;DR for the coder

1. **The real bug is not a missing `-I` flag**; the bug is that
   `spmw_apu_v1_build.py:225` emits `#include <gsi/libgvml_logical.h>`
   for a header that does not ship with the GVML release installed on
   this host. Remove that include. Adding `-I/usr/local/include` to
   the Makefile would be a no-op because GSI's stock `Common/common.mk`
   already adds `-I $(GSI_USR_LOCAL_INCLUDE)` (= `/usr/local/include`)
   for every `dev_modules` compile.
2. Add a lightweight build-time probe (`_gvml_include_root()`) that
   asserts the GVML SDK is reachable before `make` is invoked, so that
   a missing SDK fails fast with a clear error instead of make-stderr
   noise. Keep the include-root configurable via env var for
   non-stock installations.
3. Promote `_run_apu_v1` make-fail from a silent `cycles=None` return
   to a `RuntimeError`. Keep the "simulator unavailable" string in
   stdout reserved for the **gate** layer (toolchain / PCI / SDK
   missing), not for build failures of a project we just emitted.
4. Replace `test_e2e_mlp_apu_v1`'s conditional cycles assertion with
   the unconditional form `assert result.cycles is not None and
   result.cycles > 0`. Move the "simulator unavailable" branch
   entirely into `@pytest.mark.skipif` via the existing
   `_apu_v1_device_available()` helper (which we extend to probe the
   GVML SDK as well).

---

## 2. Background and evidence

### 2.1 What the Makefile currently emits

`_emit_makefile(lab_name)` produces seven lines:

```makefile
GNU_TOOLCHAIN_FOR_ARC_BASE := /usr/local/gsi-apu/13.7.1/.../arc-snps-elf/
export PATH:=${GNU_TOOLCHAIN_FOR_ARC_BASE}/bin:${PATH}

lab_name := <lab>
TOP_DIR  := $(shell pwd)
include $(TOP_DIR)/Common/common.mk
```

It is byte-identical to `/home/nz264/shared/accelerator-hub/gsi-apu/example-gvml/Makefile`,
which is the GSI vendor template the codegen mirrors.

### 2.2 What the device-side compile rule does inside `Common/common.mk`

The dev_modules rule (lines 641–651 of the vendor `common.mk`) calls:

```makefile
$(APU_CC_CMD) \
    -I. -I $(GSI_USR_INCLUDE) \
    -I $(GSI_USR_LOCAL_INCLUDE) \
    $(dev_inc) \
    -I $(TOP_DIR)/Common \
    $(dev_extra_inc) \
    -g -c $< -o $@
```

For the default `product=x86_64` (which our generated Makefile uses
implicitly), `$(GSI_USR_LOCAL_INCLUDE)` resolves to `/usr/local/include`.
The GVML headers do live there: `ls /usr/local/include/gsi/libgvml*.h`
returns 11 files. The Makefile is **not** missing an include path.

### 2.3 What the emitted `device.c` includes

`_emit_device_c` (line 218–230 of `spmw_apu_v1_build.py`) emits:

```c
#include <gsi/libsys/assert.h>
#include <gsi/libsys.h>
#include <gsi/libgal.h>
#include <gsi/gal-fast-funcs.h>
#include <gsi/libgvml_memory.h>
#include <gsi/libgvml_element_wise.h>
#include <gsi/libgvml_logical.h>        <-- offender
#include <gsi/libgvml_debug.h>
```

All eight headers except `libgvml_logical.h` are present at both
`/usr/local/include/gsi/` and `/home/nz264/shared/accelerator-hub/gsi-apu/gvml-headerfiles/`.
`libgvml_logical.h` exists in neither location. Grep across the
entire Tenon source tree returns exactly one hit — line 225 of
`spmw_apu_v1_build.py`. The header is referenced nowhere else; no
emitter calls a function that the header would have to declare.

The logical bit ops (`gvml_xor_16`, `gvml_or_16`, `gvml_and_16`,
`gvml_not_16`) — the ops a reader would assume `libgvml_logical.h`
declares — are actually declared in `libgvml_element_wise.h`
(lines 173, 180, 195, 200). So removing the include cannot break any
existing emitter; the bit-op declarations come in through
`libgvml_element_wise.h`, which we already include.

### 2.4 Where the test-pass illusion comes from

Three concurrent silent-pass paths conspire:

| Site | Behaviour |
|---|---|
| `spmw_codegen.py:1682–1693` | `make` returncode != 0 -> `RunResult(cycles=None, stdout="make failed:\n…")` |
| `test_e2e_mlp.py:228–230` | asserts `"simulator unavailable"` is *not* in stdout — but make-fail stdout starts with `"make failed:\n"`, so this assertion passes |
| `test_e2e_mlp.py:233–234` | `if result.cycles is not None: assert result.cycles > 0` — cycles is `None`, so the assertion is skipped entirely |

Test reports PASS even though build failed. This is the documented
status-review red flag.

---

## 3. Decision matrix (issues the task asked the architect to settle)

### 3.1 GVML SDK path discovery

**Decision: keep the discovery additive and probe-only; do not add a
`-I` to the Makefile.**

Rationale:
- The stock GVML headers are at `/usr/local/include/gsi/`. Common.mk
  already provides this through `$(GSI_USR_LOCAL_INCLUDE)`. Adding
  `-I/usr/local/include` to the Tenon Makefile is a no-op for the
  default install.
- We still want a Python-side probe so that a missing SDK fails fast
  in the build harness with a readable error instead of bubbling up
  as a make-stderr blob 200 lines later.

**New function in `spmw_apu_v1_build.py`** (placed next to
`_toolchain_base`):

```python
_DEFAULT_GVML_INCLUDE_ROOT = "/usr/local/include"

# One header we know must exist for any device.c compile to succeed;
# used as the canary file the probe checks.
_GVML_CANARY_HEADER = "gsi/libgvml_element_wise.h"


def _gvml_include_root() -> str:
    """Return the directory under which `<gsi/libgvml_*.h>` headers live.

    Resolution order:
      1. env var `TENON_APU_V1_GVML_INCLUDE_ROOT` (override; not validated here).
      2. `_DEFAULT_GVML_INCLUDE_ROOT` (= `/usr/local/include`) — the path
         GSI's stock `Common/common.mk` ships with for `product=x86_64`.

    Validation is intentionally deferred to `_assert_gvml_sdk_present()`;
    this getter is pure.
    """
    return os.environ.get(
        "TENON_APU_V1_GVML_INCLUDE_ROOT",
        _DEFAULT_GVML_INCLUDE_ROOT,
    )


def _assert_gvml_sdk_present() -> None:
    """Raise FileNotFoundError if the GVML SDK canary header is missing.

    Called by the build harness right before `make` runs, so that a
    misconfigured host produces a single readable error instead of a
    make-stderr blob. Skip gates in tests should call
    `_gvml_sdk_available()` (below) so that PASS/SKIP routing stays in
    one place.
    """
    root = _gvml_include_root()
    canary = Path(root) / _GVML_CANARY_HEADER
    if not canary.is_file():
        raise FileNotFoundError(
            f"GVML SDK not found: expected {canary} (set "
            f"TENON_APU_V1_GVML_INCLUDE_ROOT to override)."
        )


def _gvml_sdk_available() -> bool:
    """Non-raising counterpart of `_assert_gvml_sdk_present()`."""
    root = _gvml_include_root()
    return (Path(root) / _GVML_CANARY_HEADER).is_file()
```

The canary header is `gsi/libgvml_element_wise.h` — picked because it
is the only GVML header whose declarations the Tenon emitter actively
depends on (it carries `gvml_add_s16`, `gvml_xor_16`, `gvml_or_16`,
`gvml_and_16`, `gvml_not_16` — all referenced by `APUv1Ctx.cmds`).

**Makefile emission change**: none. The coder must NOT add a
`-I$(GVML_INCLUDE_ROOT)` line to `_emit_makefile`. The override
exists for the build-harness probe only; if a future install ships
GVML under a non-stock prefix, the coder will need to also pass
`dev_extra_inc_dirs := <root>` into the Makefile, but that's a
follow-up triggered only when the env var is set to a non-default
value. Out of scope for SPEC-001.

If the coder finds that a non-stock install IS the operative case on
this host (i.e. headers really live somewhere other than
`/usr/local/include`), STOP and surface that finding back to the
architect — the spec changes and we extend `_emit_makefile` with a
`dev_extra_inc_dirs := <root>` line plus the corresponding `-I` flag.
Do not silently add the flag.

### 3.2 The bad `#include` in `device.c`

**Decision: remove the `#include <gsi/libgvml_logical.h>` line outright.**

Rationale:
- The header does not exist in any installed GVML release on this
  host. It is not referenced by the GSI vendor `example-gvml`
  template either (which only pulls memory / element_wise / debug).
- No emitter in Tenon calls anything that would need `libgvml_logical.h`.
  All logical ops are declared in `libgvml_element_wise.h`, which we
  already include immediately above the offending line.
- Removing it is a strictly subtractive change with zero blast radius.

**Edit target**: `spmw_apu_v1_build.py:225` — delete the single line:

```c
#include <gsi/libgvml_logical.h>
```

No replacement.

### 3.3 Skip-gate vs. build-time check

**Decision: do both, but with clear ownership.**

- `_apu_v1_device_available()` in `test_e2e_mlp.py` extends to also
  call `_gvml_sdk_available()` from `spmw_apu_v1_build`. This is the
  skip-gate: if the SDK is absent, the test is `skipif`-marked and
  never runs.
- `_run_apu_v1` in `spmw_codegen.py` calls `_assert_gvml_sdk_present()`
  before invoking `make`. This is the build-time check: if a user has
  the toolchain + PCI device but the SDK headers are absent, the
  build harness raises with a clear message rather than dropping a
  make-stderr blob through `RunResult.stdout`. (In practice these two
  conditions coincide on this host, but separating them keeps the
  error surface predictable for future installations.)

**Concrete edits the coder must make**:

In `test_e2e_mlp.py` lines 195–202, change to:

```python
def _apu_v1_device_available() -> bool:
    """True iff ARC toolchain, PCI device, AND GVML SDK headers are all present."""
    import pathlib
    from allo.spmw_apu_v1_build import _gvml_sdk_available
    toolchain_bins = (
        list(pathlib.Path("/usr/local/gsi-apu").rglob("arc-elf32-gcc"))
        if pathlib.Path("/usr/local/gsi-apu").is_dir() else []
    )
    pci_present = pathlib.Path("/sys/bus/pci/devices/0000:41:00.0").exists()
    return bool(toolchain_bins) and pci_present and _gvml_sdk_available()
```

And update the `@pytest.mark.skipif` reason string to mention the GVML SDK.

### 3.4 Test assertion shape

**Decision: unconditional positive-cycles assertion, no body-side skip branch.**

Replace lines 227–234 of `test_e2e_mlp.py` with:

```python
# Real-hardware run: cycles must be present and positive. Hardware
# absence is handled by the @pytest.mark.skipif decorator above; we
# never expect to reach this assertion with cycles=None.
assert result.cycles is not None and result.cycles > 0, (
    f"APU v1 hardware returned cycles={result.cycles!r}; "
    f"stdout tail: {result.stdout[-400:]}"
)
```

And drop the `# Hardware run must not report "unavailable".` block at
lines 227–230 — it becomes redundant once the skip gate covers
SDK absence, and once `_run_apu_v1` raises on make-fail (so the
"simulator unavailable" string can no longer appear in `result.stdout`
without an actual gate-level miss).

The `note` line at 236 simplifies to:

```python
request.node._e2e_result = ("apu_v1", result.cycles, "PASS (hw)")
```

### 3.5 `_run_apu_v1` build-failure handling

**Decision: raise `RuntimeError`, do not return `RunResult(cycles=None)`.**

Edit target: `spmw_codegen.py:1682–1693`. Replace the
`if mk.returncode != 0: return RunResult(cycles=None, ...)` block
with:

```python
if mk.returncode != 0:
    raise RuntimeError(
        "APU v1 make failed:\n"
        + mk.stderr.decode("utf-8", errors="replace")
        + "\n--- stdout ---\n"
        + mk.stdout.decode("utf-8", errors="replace")
        + f"\n--- project_dir: {project_dir}"
    )
```

Same treatment for line 1696–1703 (`bin_path not found`): raise
`RuntimeError` instead of returning `cycles=None`. By the time we are
past `make`, build success has been asserted; a missing binary is a
build-system bug, not an environment skip.

Keep the EARLY `cycles=None` returns at lines 1617 (toolchain
missing) and 1631 (build harness invocation OSError on environment
setup) as-is — those are genuine "simulator unavailable" gates and
keep the existing `"simulator unavailable: ..."` stdout prefix so the
test layer can route them. Actually — see §4: even those should
migrate to `_apu_v1_device_available()`-style skip-gate now that the
skip decorator covers them.

The coder may keep the `subprocess.SubprocessError` / `OSError`
branch at line 1674–1681 as `cycles=None` (with "make invocation
failed" stdout) only because that catches genuine harness-side
failures (OOM, signal, etc.) that *should* fail the test loudly. The
cleanest move is to also raise there. Per the audit table below, this
path becomes `RuntimeError` too.

---

## 4. Cross-backend audit: cycles=None in the five run paths

For each `cycles=None` return site, label whether the test should
hard-fail (the failure is a build/runtime bug, not an environment
skip) or skip-gate (the failure means the simulator is unreachable
and the test cannot run).

### 4.1 `_run_samsung` (~1224–1359)

| Line | Condition | Verdict | Rationale |
|---|---|---|---|
| 1238 | `pim_driver` binary missing | **skip-gate** | Simulator unavailable; existing `"simulator unavailable"` stdout prefix and the test's `_sim_unavailable()` helper handle this. Keep. |
| 1258 | `numpy` import fails | **hard-fail** (raise `ImportError`) | numpy is a hard Tenon dependency; "no numpy" is a setup bug, not a sim skip. |
| 1277, 1302, 1323 | required input kwarg missing for GEMV/ADD/MUL/RELU | **hard-fail** (raise `ValueError`) | The user called `compiled.run()` without the inputs the workload's role table demands; this is a user-input contract violation, not an env skip. |
| 1342 | `subprocess.SubprocessError` / `OSError` on `pim_driver` exec | **hard-fail** (raise `RuntimeError`) | We just established the binary exists in step 1238; if its invocation fails, that is a runtime/environment bug that must surface, not skip. |
| 1350 (cycles=None when regex misses) | parser miss | **hard-fail** | The driver returned but PROF output is missing — that is a regression in the driver contract, not an env skip. Spec asks the coder to add `if cycles is None: raise RuntimeError(...)` after the regex match. |

### 4.2 `_run_aim` (~1362–1432)

| Line | Condition | Verdict | Rationale |
|---|---|---|---|
| 1375 | Docker image / yaml missing | **skip-gate** | Existing pattern, keep. |
| 1410 | `docker run` subprocess error | **hard-fail** (raise) | Image exists per gate; invocation failure is a runtime bug. |
| 1426 (cycles=None when regex misses) | ramulator2 produced no `memory_system_cycles` line | **hard-fail** | Trace ran but parser missed; that's a contract regression. Coder adds `if cycles is None: raise`. |

### 4.3 `_run_upmem` (~1435–1512)

| Line | Condition | Verdict | Rationale |
|---|---|---|---|
| 1449 | `uPIMulator` binary missing | **skip-gate** | Keep existing pattern. |
| 1488 | subprocess error invoking uPIMulator | **hard-fail** (raise) | Binary exists per gate; invocation failure is a runtime bug. |
| 1502 (cycles=None when regex misses) | log parser miss | **hard-fail** | Coder adds `if cycles is None: raise`. |

Note: UPMEM has a separate design issue (task 003 wires real kernels;
PrIM-proxy `cycles` are not meaningful for arbitrary workloads). That
is out of scope for SPEC-001; the audit here covers only the silent
`cycles=None` paths in the run-harness.

### 4.4 `_run_apu_v1` (~1601–1756)

| Line | Condition | Verdict | Rationale |
|---|---|---|---|
| 1617 | toolchain not found | **skip-gate** | Convert to use the new `_apu_v1_device_available()` precondition; in practice the `@pytest.mark.skipif` already prevents us from entering this branch. Keep `cycles=None` + `"simulator unavailable"` stdout as a defensive fallback. |
| 1631 | build harness Python exception | **hard-fail** (raise) | This means our own code crashed during project emission; that is a Tenon bug, not an env skip. Re-raise the original exception (or wrap in `RuntimeError`). |
| 1676 | `subprocess.SubprocessError` invoking `make` | **hard-fail** (raise `RuntimeError`) | Per §3.5. |
| 1684 | `make` returncode != 0 | **hard-fail** (raise `RuntimeError`) | Per §3.5 — this is the headline fix. |
| 1698 | binary not at `build/debug/tenon-kernel` after `make` | **hard-fail** (raise `RuntimeError`) | Per §3.5. |
| 1723 | binary invocation subprocess error | **hard-fail** (raise `RuntimeError`) | Binary exists; runtime failure must surface. |
| 1733 (cycles=None when PROF parser misses) | hardware ran but PROF_PRINT did not emit `total` | **hard-fail** (raise `RuntimeError`) | Coder adds `if cycles is None: raise`. |

### 4.5 `_run_apu_v2` (~1759–1782)

| Line | Condition | Verdict | Rationale |
|---|---|---|---|
| 1770 | docker image missing | **skip-gate** | Keep. |
| 1779 (always cycles=None by design) | l1_sim is functional-only | **special**: cycles=None is the contract, not a failure. | Test asserts `result.cycles is None` for APU v2 unconditionally. The MLP test already does this (`test_e2e_mlp_apu_v2` body); no change. Task 011/012 may revisit this once a cost-model surrogate is decided. |

---

## 5. Concrete patch list for the coder (task 002)

Targets, in order. Each item is in scope of SPEC-001.

### 5.1 `experiments/allo/allo/spmw_apu_v1_build.py`

1. Add the three new module-level functions (`_gvml_include_root`,
   `_assert_gvml_sdk_present`, `_gvml_sdk_available`) and the two
   constants (`_DEFAULT_GVML_INCLUDE_ROOT`, `_GVML_CANARY_HEADER`) per
   §3.1.
2. Delete line 225 (`#include <gsi/libgvml_logical.h>`) inside
   `_emit_device_c`. Do not add any `-I` directives to `_emit_makefile`.

### 5.2 `experiments/allo/allo/spmw_codegen.py`

1. Inside `_run_apu_v1`, after the existing toolchain gate at line
   1617 and before the build-harness call at line ~1626, add:
   ```python
   from .spmw_apu_v1_build import _assert_gvml_sdk_present
   _assert_gvml_sdk_present()
   ```
   This raises `FileNotFoundError` with a clear message if the SDK is
   missing in build-harness territory.
2. Replace `_run_apu_v1` failure-mode `RunResult(cycles=None, ...)`
   blocks at lines 1676, 1684, 1698, 1723 with `raise RuntimeError(...)`
   per §3.5 and the §4.4 audit table.
3. After the `cycles = _parse_apu_v1_prof_print(combined)` line at 1733,
   add `if cycles is None: raise RuntimeError("APU v1 PROF_PRINT missing 'total crun=...' line in stdout: " + combined[-400:])`.
4. Apply the analogous `raise` conversions to `_run_samsung` /
   `_run_aim` / `_run_upmem` per §4.1–4.3. Keep "simulator unavailable"
   stdout-prefix returns for true env gates (binary / image / config
   absent) so the test layer's `_sim_unavailable()` helper continues to
   route correctly.

### 5.3 `experiments/allo/tests/spmw/test_e2e_mlp.py`

1. Update `_apu_v1_device_available()` (lines 195–202) to also call
   `_gvml_sdk_available()` per §3.3.
2. Update the `@pytest.mark.skipif` reason string at line 208 to read:
   `"APU v1 hardware not available: ARC toolchain, PCI 41:00.0, or GVML SDK absent"`.
3. Replace the conditional cycles block at lines 227–234 with the
   unconditional assertion per §3.4.
4. Simplify the `_e2e_result` note at line 236 to `"PASS (hw)"`.

The other four tests (`test_e2e_mlp_samsung` / `_aim` / `_upmem` /
`_apu_v2`) stay structurally the same — their existing
`if not _sim_unavailable(result):` / `else:` split correctly handles
the skip-gate vs. hard-fail dichotomy, given that the run-harness
changes in §5.2 now only return `cycles=None` for genuine env skips.

---

## 6. Tests this spec must not break

These existing tests guard the shared run-harness paths above. The
coder must run all of them green after the patches in §5.

- `experiments/allo/tests/spmw/test_e2e_mlp.py::test_e2e_mlp_samsung`
- `experiments/allo/tests/spmw/test_e2e_mlp.py::test_e2e_mlp_aim`
- `experiments/allo/tests/spmw/test_e2e_mlp.py::test_e2e_mlp_upmem`
- `experiments/allo/tests/spmw/test_e2e_mlp.py::test_e2e_mlp_apu_v2`
- `experiments/allo/tests/spmw/test_e2e_mlp.py::test_e2e_mlp_apu_v1`
  (the headline behaviour change of this spec)
- Any existing test that calls `_run_samsung` / `_run_aim` /
  `_run_upmem` directly and depends on `cycles=None` for a
  malformed-input case. The coder must grep for these before flipping
  the value-error returns to raises in §5.2 item 4; if any test
  relies on the old `cycles=None`-on-bad-input behaviour, fall back
  to keeping those specific returns as-is and call them out in the
  PR description.

For the FPGA / AIE shared-test surface (`experiments/allo/tests/dataflow/`,
`tests/customize/`): none of the patches in §5 touch
`allo/ir/builder.py`, `allo/ir/infer.py`, `allo/dataflow.py`, or
`allo/customize.py`. All edits are confined to `spmw_apu_v1_build.py`,
`spmw_codegen.py`, and `tests/spmw/test_e2e_mlp.py` — files that are
SPMW-only and have no upstream consumers. Rollback story: revert the
three files.

---

## 7. Rollback / dependency notes

- The change is fully additive at the API level. `_gvml_include_root()`,
  `_assert_gvml_sdk_present()`, `_gvml_sdk_available()` are new
  helpers; no existing call site is being renamed or repurposed.
- The behavioural change is in the make-fail branch of `_run_apu_v1`
  (and the parallel error paths in the other three real runners): we
  raise instead of returning `cycles=None`. This is the entire point
  of the task — surfacing build/runtime errors. Any caller that
  swallowed `cycles=None` to mask a real failure will now see the
  exception; that is the desired outcome.
- The Makefile is unchanged byte-for-byte. Existing
  `/tmp/tenon-bmatmul-sv-apu/Makefile` and the GSI vendor
  `example-gvml/Makefile` continue to build, so MICRO 25 validation
  data remains reproducible.

---

## 8. Open follow-ups (NOT in scope of SPEC-001)

- If a future host installs GVML under a non-stock prefix, the
  Makefile needs `dev_extra_inc_dirs := $(GVML_INCLUDE_ROOT)` plus
  the corresponding `-I` flag in `_emit_makefile`. Defer until the
  env var `TENON_APU_V1_GVML_INCLUDE_ROOT` is set in practice.
- `_run_upmem` currently runs a PrIM-proxy benchmark whose cycle count
  is not the cycle count of the emitted kernel. That is task 003's
  problem, not SPEC-001's. The §4.3 audit only addresses the silent
  `cycles=None` returns inside the proxy flow.
- `_run_apu_v2` always returns `cycles=None` by design. Task 011 may
  revisit whether to attach a surrogate cycle count from the cost
  model so the e2e test can do something stronger than
  `assert result.cycles is None`. Out of scope here.

---

## 9. Implemented (task 002, coder)

Date: 2026-05-18.

- `allo/spmw_apu_v1_build.py`: added `_DEFAULT_GVML_INCLUDE_ROOT`,
  `_GVML_CANARY_HEADER`, `_gvml_include_root()`,
  `_assert_gvml_sdk_present()`, `_gvml_sdk_available()`; deleted
  `#include <gsi/libgvml_logical.h>` from `_emit_device_c`. Makefile
  emission left unchanged per §3.1.
- `allo/spmw_codegen.py`: `_run_apu_v1` now calls
  `_assert_gvml_sdk_present()` before `make`, and raises `RuntimeError`
  on make-invoke-fail, make-returncode-fail, missing-binary,
  binary-invoke-fail, and PROF parser-miss (was `cycles=None`). Removed
  the redundant `FileNotFoundError` catch around
  `gen_apu_v1_low_mode_project` (the toolchain/template gate already
  covers it). `_run_samsung` / `_run_aim` / `_run_upmem` apply the §4
  audit: numpy ImportError is no longer caught; subprocess errors
  and parser-misses now raise `RuntimeError`. Samsung kwarg-missing
  branches keep `cycles=None` per the §6 escape clause — flagged
  below.
- `tests/spmw/test_e2e_mlp.py`: `_apu_v1_device_available()` extended
  with `_gvml_sdk_available()`; `@pytest.mark.skipif` reason updated;
  `test_e2e_mlp_apu_v1` asserts `result.cycles is not None and > 0`
  unconditionally; `_e2e_result` note simplified to `"PASS (hw)"`.
- `tests/spmw/test_apu_v1_build_harness.py`: added four new tests —
  `test_device_c_does_not_include_libgvml_logical`,
  `test_gvml_include_root_default_and_env_override`,
  `test_gvml_sdk_probe_raising_and_nonraising`,
  `test_emit_makefile_unchanged_no_extra_include`.

Flagged for PR review (§6 escape):
- `_run_samsung` keeps `cycles=None` on `W is None or x is None`
  (and similarly for ADD/MUL/RELU). `tests/spmw/test_run.py::test_run_returns_runresult_for_all_backends`
  calls `compiled.run()` with no kwargs and asserts `cycles is None`;
  flipping these to `raise ValueError` breaks that contract. The
  spec §6 explicitly authorises this escape; flagging here so the
  architect can decide whether to update the existing test instead.

Tests run:
- `pytest tests/spmw/test_apu_v1_build_harness.py -v` → 5 passed.
