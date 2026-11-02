<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# PIMSimulator overlay for Tenon (`pim_driver`)

The Tenon-authored part of the `pim_driver` conductor that Tenon's Samsung
HBM-PIM runtime (`allo/spmw_samsung.py`) invokes. It applies on top of an
unmodified upstream checkout together with five patches to Samsung-licensed
upstream files, which are **not** in this repository (next section).

## Samsung limitation: this route cannot be built from this repository alone

`pim_driver` needs five patches to Samsung-licensed upstream files:

| Patch | Upstream file | Change |
|---|---|---|
| `PIMKernel.h.patch` | `src/tests/PIMKernel.h` | includes `tests/TenonConductor.h`; declares `runTemporalReduce` and the `tenon*` conductor members |
| `PIMKernel.cpp.patch` | `src/tests/PIMKernel.cpp` | adds `<fstream>`/`<stdexcept>` and includes `tests/TemporalReduce.inc` and `tests/TenonConductor.inc`; no upstream function body changes |
| `PIMCmdGen.h.patch` | `src/tests/PIMCmdGen.h` | declares the `PIMCmdGen::getPIMCmds` invocation counter (`getInvocationCount`) |
| `PIMCmdGen.cpp.patch` | `src/tests/PIMCmdGen.cpp` | counts `getPIMCmds` invocations, so the conductor can prove the vendor microcode generator was not used (`pimcmdgen_calls=0`) |
| `Burst.h.patch` | `src/Burst.h` | `NumpyBurstType::loadFp16` no longer reads past the end of an array whose length is not a multiple of 16 |

These patches are **not included**. PIMSimulator is distributed under
Samsung's notice (`LICENSE-PIMSimulator`: the software is the property of
Samsung Electronics, no part of it may be copied or distributed without
Samsung's express written permission, and use is limited to non-commercial
academic research). The patches carry context lines of Samsung's files, so
they cannot be published until Samsung authorizes it (`docs/release/ISSUES.md`
P-04). For the authors they live in the containing Tenon research repository
at `experiments/simulators/setup/PIMSimulator/patches/` (not public).

**Until Samsung authorizes publication, the Samsung HBM-PIM simulator route
cannot be built from the release repository alone.** `apply.sh` stops with
exit status 2 and says so when the patches are missing. The Samsung compile
route and `allo.compile(..., backend="virtual")` need no simulator.

## Upstream

* Repository: <https://github.com/SAITPublic/PIMSimulator>
* Commit: `3703d1f19c8f027360cc33a3243eb271e3bb6898` (branch `dev`, "Update readme")

## Contents

| File | Purpose |
|---|---|
| `Sconstruct` | Replaces upstream's build script (same structure): gtest from `$GTEST_PREFIX`/`$CONDA_PREFIX` for the `sim` test binary, plus a `pim_driver` target that links only `pthread`. |
| `src/pim_driver.cc` | Command-line conductor: `--describe`, `--op GENERIC --op-kind REDUCE|ELTWISE` (Tenon programs), `--op GENERIC --op-kind TEMPORAL_REDUCE` (ordered FP16 contraction), `--op GEMV` (vendor reference baseline). |
| `src/tests/TenonConductor.h`, `src/tests/TenonConductor.inc` | Plan-driven conductor: uploads the compiled CRF verbatim, walks it with an exact model of `PIMRank::doPIM`, and issues the column commands each instruction consumes at the addresses the plan assigns. Never calls `PIMCmdGen`. |
| `src/tests/TemporalReduce.inc` | Ordered (source-order FP16) contraction conductor (release audit S-03; not yet reachable from `allo.compile`). |
| `apply.sh` | Checks for the five patches, applies them, installs the files above, builds `pim_driver` and `libdramsim/libdramsim2.a`. |

## Build (requires the patches)

```bash
git clone https://github.com/SAITPublic/PIMSimulator.git
git -C PIMSimulator checkout 3703d1f19c8f027360cc33a3243eb271e3bb6898
bash $ALLO/simulators/PIMSimulator/apply.sh PIMSimulator <patches-dir>
export PIMSIMULATOR_ROOT=$PWD/PIMSimulator
```

`apply.sh` looks for `PIMKernel.h.patch`, `PIMKernel.cpp.patch`,
`PIMCmdGen.h.patch`, `PIMCmdGen.cpp.patch` and `Burst.h.patch` in, in order,
its second argument, `$TENON_PIMSIM_PATCHES`, and a `patches/` directory next
to `apply.sh`. With all five present it applies them (`patch -p1`,
idempotent), installs the files above, runs
`scons pim_driver libdramsim/libdramsim2.a -j$(nproc)` in the checkout and
prints the geometry (`./pim_driver --describe`). The static library is linked
by the layout-swizzle trace probe (`scripts/run_pimsim_layout_trace.py`);
without it that test skips. `TENON_SKIP_BUILD=1` only installs the sources.
Toolchain: g++ with C++14 (recorded with system g++ 9.4.0) and SCons; the
driver needs no gtest. Never commit a `patches/` directory into this
repository (`.gitignore` excludes `simulators/PIMSimulator/patches/`).

## Verification

```bash
cd PIMSimulator && ./pim_driver --describe
# PIM_GEOMETRY channels=64 ranks=1 pim_blocks=8 banks=16 bank_groups=4 grf_a=8 grf_b=8 lanes=16 crf_words=32 cols_per_row=32 rows=16384
cd $ALLO && python -m pytest tests/pim/test_paper_golden.py -k samsung
```

The vendor-equivalent Samsung GEMV placements must report 4,435 / 8,041 /
8,247 cycles for M x K = 4096 x 256, 4096 x 512 and 8192 x 256 (cost-selected
default 4,389 / 7,947 / 8,153). Recorded runs: `docs/release/simulators.md`.
