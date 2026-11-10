<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Bundled Samsung PIMSimulator for Tenon (`pim_driver`)

Tenon's Samsung HBM-PIM runtime (`allo/spmw_samsung.py`) executes the
programs it compiles on `pim_driver`, a conductor built on Samsung's
PIMSimulator. This directory holds everything that build needs, so the
Samsung simulator route installs from the Tenon distribution alone:

| Content | Purpose |
|---|---|
| `upstream/` | Pristine copy of <https://github.com/SAITPublic/PIMSimulator> at commit `3703d1f19c8f027360cc33a3243eb271e3bb6898` (branch `dev`, "Update readme"): `src/`, `lib/`, `tools/`, `ini/`, `system_hbm*.ini`, upstream's `Sconstruct`, `README.md`, `LICENSE-PIMSimulator`, `LICENSE-DRAMSIM2`. Upstream's `data/` and `dump/` benchmark inputs are not included (about 25 MiB; `pim_driver` does not read them). |
| `UPSTREAM_MANIFEST.sha256` | SHA-256 of every file under `upstream/`; the build refuses a copy that does not match (`python -m allo.pim.samsung_simulator verify`). |
| `patches/` | Tenon's five patches to upstream files (table below). |
| `Sconstruct` | Replaces upstream's build script (same structure): gtest from `$GTEST_PREFIX`/`$CONDA_PREFIX` for the `sim` test binary only, a `pim_driver` target that links just `pthread`, and the caller's `TMPDIR` passed to the compiler. |
| `src/pim_driver.cc` | Command-line conductor: `--describe`, `--op GENERIC --op-kind REDUCE|ELTWISE` (Tenon programs), `--op GENERIC --op-kind TEMPORAL_REDUCE` (ordered FP16 contraction), `--op GEMV` (vendor reference baseline). |
| `src/tests/TenonConductor.h`, `src/tests/TenonConductor.inc` | Plan-driven conductor: uploads the compiled CRF verbatim, walks it with an exact model of `PIMRank::doPIM`, and issues the column commands each instruction consumes at the addresses the plan assigns. Never calls `PIMCmdGen`. |
| `src/tests/TemporalReduce.inc` | Ordered (source-order FP16) contraction conductor (release audit S-03). |
| `apply.sh` | Applies the patches, installs the Tenon files and builds `pim_driver` and `libdramsim/libdramsim2.a` in a PIMSimulator source tree. |
| `NOTICE.md` | The terms these files carry and the redistribution status. **Read it**: use of the Samsung sources is restricted to non-commercial research, and their notice requires Samsung's written permission for distribution, which is not recorded here. |

## The patches

| Patch | Upstream file | Change |
|---|---|---|
| `PIMKernel.h.patch` | `src/tests/PIMKernel.h` | includes `tests/TenonConductor.h`; declares `runTemporalReduce` and the `tenon*` conductor members |
| `PIMKernel.cpp.patch` | `src/tests/PIMKernel.cpp` | adds `<fstream>`/`<stdexcept>` and includes `tests/TemporalReduce.inc` and `tests/TenonConductor.inc`; no upstream function body changes |
| `PIMCmdGen.h.patch` | `src/tests/PIMCmdGen.h` | declares the `PIMCmdGen::getPIMCmds` invocation counter (`getInvocationCount`) |
| `PIMCmdGen.cpp.patch` | `src/tests/PIMCmdGen.cpp` | counts `getPIMCmds` invocations, so the conductor can prove the vendor microcode generator was not used (`pimcmdgen_calls=0`) |
| `Burst.h.patch` | `src/Burst.h` | `NumpyBurstType::loadFp16` no longer reads past the end of an array whose length is not a multiple of 16 |

## Build from the installed package (the supported route)

```bash
python -m allo.pim.samsung_simulator build            # ~15 s on 16 cores
#   copies upstream/ to $TENON_HOME/simulators/PIMSimulator (default
#   ~/.local/share/tenon/simulators/PIMSimulator), applies patches/, installs
#   the Tenon files, runs scons, prints PIM_GEOMETRY
python -m allo.pim.samsung_simulator locate           # what pimsim_root() finds
```

`allo.spmw_simenv.pimsim_root()` discovers that default root as soon as it
holds `pim_driver`; no environment variable is needed. `PIMSIMULATOR_ROOT`
and `TENON_SIMULATORS_DIR` keep precedence (`docs/release/INSTALL.md`
section 3). `--root DIR` builds elsewhere (then set `PIMSIMULATOR_ROOT`),
`--jobs N` sets the parallelism, `--skip-build` only assembles the patched
tree, `--scons EXE` names the SCons to use. Toolchain: g++ with C++14
(recorded with system g++ 9.4.0), SCons (`python -m pip install scons` or
the distribution package; 3.1.2 recorded) and `patch`; no gtest. The
compiler's temporaries follow `TMPDIR`.

The same command works from a source checkout (the recipe is found at
`simulators/PIMSimulator`); the wheel carries a copy under
`allo/pim/simulators/PIMSimulator` (`setup.py`'s `BuildPy`).

## Build into your own upstream clone

```bash
git clone https://github.com/SAITPublic/PIMSimulator.git
git -C PIMSimulator checkout 3703d1f19c8f027360cc33a3243eb271e3bb6898
bash $ALLO/simulators/PIMSimulator/apply.sh PIMSimulator
export PIMSIMULATOR_ROOT=$PWD/PIMSimulator
```

`apply.sh` takes the patches from `patches/` next to it (a second argument
or `TENON_PIMSIM_PATCHES` selects another set), applies them (`patch -p1`,
idempotent), installs the Tenon files, runs
`scons pim_driver libdramsim/libdramsim2.a -j$(nproc)` and prints the
geometry (`./pim_driver --describe`). `TENON_SKIP_BUILD=1` only installs the
sources; `TENON_SCONS` and `TENON_BUILD_JOBS` override the SCons command and
the parallelism. The static library is linked by the layout-swizzle trace
probe (`scripts/run_pimsim_layout_trace.py`); without it that test skips.

## Verification

```bash
cd $(python -m allo.pim.samsung_simulator locate | awk '/^pimsim_root/ {print $2}') && ./pim_driver --describe
# PIM_GEOMETRY channels=64 ranks=1 pim_blocks=8 banks=16 bank_groups=4 grf_a=8 grf_b=8 lanes=16 crf_words=32 cols_per_row=32 rows=16384
cd $ALLO && python -m pytest tests/pim/test_gemv_all_targets.py -k samsung
cd $ALLO && python -m pytest tests/pim/test_paper_golden.py -k samsung
```

The vendor-equivalent Samsung GEMV placements must report 4,435 / 8,041 /
8,247 cycles for M x K = 4096 x 256, 4096 x 512 and 8192 x 256 (cost-selected
default 4,389 / 7,947 / 8,153). A `pim_driver` built from this bundle is,
after `objcopy --strip-all --remove-section=.note.gnu.build-id`, identical to
the one the authors' evaluation ran (`docs/release/simulators.md`). Recorded
runs: `docs/release/simulators.md`, `docs/release/VALIDATION.md`.
