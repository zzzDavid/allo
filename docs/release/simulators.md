<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Simulators and vendor SDKs used by Tenon backends

Tenon does not bundle any simulator, vendor SDK or Docker image. This page
records, for each external dependency, where it comes from, the revision the
evaluation used, where the Tenon-side changes live, how to build it, its
license, and what a user has to obtain on their own. Discovery variables are
in `INSTALL.md` section 3.

The Tenon-side recipes are in this repository under `simulators/`
(`simulators/README.md`); `$ALLO` below is the root of a checkout of
`zzzDavid/allo`, branch `tenon`. The AiM and UPMEM routes build from that
checkout alone (UPMEM also needs the UPMEM DPU SDK, which you obtain
yourself). **Samsung is the exception: the five patches to Samsung-licensed
PIMSimulator files are not in this repository, so until Samsung authorizes
their publication the Samsung HBM-PIM simulator route cannot be built from the
release repository alone** (P-04; Samsung section below).

Revisions were read on the authors' host on 2026-11-02 with
`git -C <checkout> rev-parse HEAD` (PIMSimulator through the parent
repository's `.git/modules/simulators/PIMSimulator`, because the checkout's
own `.git` pointer is broken); on that date each pinned commit was also the
upstream `HEAD`.

The three simulator recipes were verified twice on 2026-11-02, each time by
building from fresh upstream clones in scratch directories and new Docker tags,
without touching the installed simulators, then running the Tenon tests
through `allo.compile` against the scratch builds: first from the recipes'
original home (below), then from the copies in `simulators/` of this
repository (section "Verification from the release copies"). Host: Linux
x86-64 (Ubuntu 20.04 userland), Docker 24.0.2, system g++ 9.4.0 and SCons
3.1.2, Go 1.23.12.

Historical note: until 2026-11-02 the recipes lived only in the containing
Tenon research repository (the repository that holds this Allo fork under
`experiments/allo`, written `<repo>` below) under
`<repo>/experiments/simulators/setup/`, which a checkout of this repository
does not have. `simulators/` holds byte-for-byte copies of every
redistributable file from there, except the `README.md` files and
`PIMSimulator/apply.sh`, which were adapted to the release paths and to the
missing Samsung patches. The Samsung-derived `PIMSimulator/patches/` stay only
in `<repo>`.

## Summary

| Dependency | Upstream | Pinned revision | Tenon changes (in `simulators/` of this repository) | License | Rebuildable from this repository and public sources? |
|---|---|---|---|---|---|
| Samsung PIMSimulator | <https://github.com/SAITPublic/PIMSimulator> | `3703d1f19c8f027360cc33a3243eb271e3bb6898` (2024-07-01) | `PIMSimulator/`: `pim_driver` conductor sources, `Sconstruct`, `apply.sh`; the 5 patches to Samsung files are **not included** | Samsung proprietary notice (no redistribution without written permission; non-commercial research only); DRAMSim2 part BSD | **No**: needs the 5 Samsung-derived patches, which await Samsung's permission (P-04); verified with the authors' copy of the patches |
| SK hynix AiM simulator (Ramulator 2.0 based) | <https://github.com/arkhadem/aim_simulator> | `0f28a07bdb83e42b9305ad3d45410ebd3aa2c091` (2025-07-22) | `aim_simulator/Dockerfile`, `README.md` | MIT (University of Michigan, 2025) | **Yes** (verified from the release copy) |
| Ramulator2-AttAcc (evaluation script only; not used by the package) | DCC artifact <https://github.com/SPIN-Research-Group/DCC> `src/Ramulator2/` (AttAcc model from <https://github.com/scale-snu/attacc_simulator>) | DCC `212f86ea7f08da059836e3a1311d3a11afae8dbe` | none (no Tenon recipe yet) | Ramulator2 and AttAcc simulator MIT; DCC (trace generator, `inference/`) **no license** | user builds it from the DCC checkout (`build.sh`); no Tenon recipe |
| uPIMulator (Go) | <https://github.com/VIA-Research/uPIMulator> | `870d916334e9ff0b190f555f951a9ec3c4257781` (2025-02-01) | `uPIMulator/tenon-upimulator.patch` (47 files), `apply.sh`, `Dockerfile`, README | MIT (VIA Research Group, KAIST, 2024) | **Yes** (verified from the release copy), given the user-obtained SDK |
| UPMEM DPU SDK 2023.1.0 | UPMEM (`sdk-releases.upmem.com`, returned HTTP 403 when fetched); mirror <https://github.com/CMU-SAFARI/UPMEM> | mirror `84a0dcd3f57ded98ea049f2be5d45c71eda96ace` | none | UPMEM proprietary; the mirror ships no license file | user must obtain it |
| GSI APU v1 (Gemini-I) SDK: ARC toolchain, GDL/GAL/GVML, `example-gvml` | GSI Technology | `/usr/local/gsi-apu/13.7.1` on the authors' host | none | GSI proprietary | user must obtain it |
| GSI APU v2 (Gemini-II) SDK under `/opt/gsi`, `gsi_tool`, card firmware | GSI Technology | firmware `GSI_G2_Infrastructure_V1.24.1` on `zhang-x4` | none | GSI proprietary | user must obtain it |

## Samsung HBM-PIM: PIMSimulator

- **Upstream:** <https://github.com/SAITPublic/PIMSimulator>, revision
  `3703d1f19c8f027360cc33a3243eb271e3bb6898` (branch `dev`).
- **What Tenon needs:** a `pim_driver` executable at `$PIMSIMULATOR_ROOT/pim_driver`
  (`allo/spmw_samsung.py` always passes `--op GENERIC`), the `ini/`
  configurations of the upstream tree, and, for the layout-swizzle probe
  only, `libdramsim/libdramsim2.a`.
- **Where the Tenon changes are:** `simulators/PIMSimulator/` in this
  repository (README with the file list): `Sconstruct`, `src/pim_driver.cc`,
  `src/tests/TenonConductor.{h,inc}`, `src/tests/TemporalReduce.inc`,
  `apply.sh`. Samsung files are changed only through
  `patches/{PIMKernel.h,PIMKernel.cpp,PIMCmdGen.h,PIMCmdGen.cpp,Burst.h}.patch`
  (against `src/tests/PIMKernel.h`, `src/tests/PIMKernel.cpp`,
  `src/tests/PIMCmdGen.h`, `src/tests/PIMCmdGen.cpp`, `src/Burst.h`), which
  are **not in this repository**; for the authors they live in
  `<repo>/experiments/simulators/setup/PIMSimulator/patches/`.
- **Samsung limitation:** until Samsung authorizes publication of those five
  patches, the Samsung HBM-PIM simulator route cannot be built from the
  release repository alone. `simulators/PIMSimulator/apply.sh` exits with
  status 2 and prints that message when the patches are missing; it accepts
  them as its second argument, through `TENON_PIMSIM_PATCHES`, or as a
  `patches/` directory next to it (ignored by git and pruned from the sdist).
- The earlier `--slice-plan` route (`SamsungSlicePlan.*`) and the stale
  323-line overlay are gone; every overlay file now equals the installed
  tree's, and the installed tree has no other modified tracked file.
- **Build recipe (verified; needs the patches):**

  ```bash
  git clone https://github.com/SAITPublic/PIMSimulator.git
  git -C PIMSimulator checkout 3703d1f19c8f027360cc33a3243eb271e3bb6898
  bash $ALLO/simulators/PIMSimulator/apply.sh PIMSimulator <patches-dir>
  #   without <patches-dir> it stops: "...cannot be built from the release
  #   repository alone..." (exit 2)
  #   applies the patches, installs the conductor sources, runs
  #   `scons pim_driver libdramsim/libdramsim2.a -j$(nproc)`, prints PIM_GEOMETRY
  export PIMSIMULATOR_ROOT=$PWD/PIMSimulator
  ```

  Needs g++ with C++14 and SCons; no gtest for `pim_driver`. No Docker image.
- **Verification (2026-11-02, first round, `<repo>` overlay):** clone + `apply.sh` took about 12 s (48
  cores). Scratch `pim_driver` sha256
  `75dd297c3c102b33f6a77080f82819ff067d6102c2387a186b7f061f62441734`;
  installed `0ae6a3d9bc0aa1ed06e52eb74d81d7c2b98c0fe344e1a5591add5b05b50c2df2`.
  They differ only in embedded debug paths: after
  `objcopy --strip-all --remove-section=.note.gnu.build-id` both (and a second
  fresh build) are
  `181fb3bba17aef77aa4330120ac667e7dcf12490c4675ce22cdf8975470e5668`. With
  `PIMSIMULATOR_ROOT` at the scratch build:
  `tests/pim/test_paper_golden.py -k samsung` 3 passed in 113 s, vendor
  layout (`samsung_vendor_gemv_layout`) **4,435** / 8,041 / 8,247 cycles and
  cost-selected default **4,389** / 7,947 / 8,153 for 4096x256 / 4096x512 /
  8192x256; `tests/pim/samsung_hbm_pim` + `test_gemv_all_targets.py -k
  samsung` 34 passed, 1 skipped in 714 s (the skip was the layout-swizzle
  probe, which needs `libdramsim2.a`; `apply.sh` now builds it, and the probe
  then passes with 148 / 116 cycles).
- **License and redistribution:** `LICENSE-PIMSimulator` states that the
  software is the property of Samsung Electronics, that no part may be copied
  or distributed without Samsung's express written permission, and that use
  is restricted to non-commercial academic research. The Allo-authored
  conductor files are ours, but the five patches carry context lines from
  Samsung files, so publishing them still needs Samsung's permission (open
  question in `audit/packaging.md` section 9). The containing repository also
  tracks modified Samsung files under `dev/` (P-04; untracking commands in
  `RELEASE_CHECKLIST.md`); they must not be part of any public tree.
- **What a user must do:** today a user outside the authors' group cannot
  build the Samsung simulator route from this repository: besides cloning
  PIMSimulator (accepting Samsung's notice), `apply.sh` needs the five
  patches, which are published only once Samsung authorizes it. The Samsung
  compile and `backend="virtual"` routes need no simulator.

## SK hynix AiM: aim_simulator (Ramulator 2.0)

- **Upstream:** <https://github.com/arkhadem/aim_simulator>, revision
  `0f28a07bdb83e42b9305ad3d45410ebd3aa2c091`; no tracked modifications on
  the authors' host.
- **Tenon changes:** `simulators/aim_simulator/Dockerfile` (+ `README.md`)
  (Ubuntu 24.04 + g++-12; the upstream ships no build environment, and the
  binary needs glibc 2.38).
- **Build recipe (verified):**

  ```bash
  git clone https://github.com/arkhadem/aim_simulator.git
  cd aim_simulator && git checkout 0f28a07bdb83e42b9305ad3d45410ebd3aa2c091
  cp $ALLO/simulators/aim_simulator/Dockerfile .
  docker build -t tenon-aim-simulator-build:0f28a07 .
  docker run --rm -u $(id -u):$(id -g) -v $PWD:/work tenon-aim-simulator-build:0f28a07 \
      bash -c "mkdir -p build && cd build && cmake .. && make -j8"
  # smoke test (689 memory-system cycles)
  docker run --rm -u $(id -u):$(id -g) -v $PWD:/work tenon-aim-simulator-build:0f28a07 \
      bash -c "cd /work && ./build/ramulator2 -f test/example.yaml -t test/all_isr.trace"
  export AIM_SIMULATOR_ROOT=$PWD TENON_AIM_SIM_IMAGE=tenon-aim-simulator-build:0f28a07
  ```

  `-u $(id -u):$(id -g)` keeps the build tree owned by you (without it Docker
  writes root-owned files). CMake downloads its third-party dependencies
  (`build/_deps`) during configure, so the build needs network access.
- **Verification (2026-11-02, first round, `<repo>` Dockerfile):** `docker build --no-cache` 27 s (image
  `tenon-aim-simulator-build:0f28a07`, id `sha256:fc69e92cf358...`, base
  image already cached); `cmake` + `make -j8` 120 s. With
  `AIM_SIMULATOR_ROOT` at the scratch clone and
  `TENON_AIM_SIM_IMAGE=tenon-aim-simulator-build:0f28a07`:
  `tests/pim/test_gemv_all_targets.py -k aim` 2 passed in 30 s; the AiM GEMV
  (256x256 BF16) reports 222 cycles, the same as on the installed checkout and
  the authors' image `aim-simulator-build` (id `sha256:215d37b48f08...`,
  built 2026-04-18 from the same Dockerfile); `tests/pim/test_paper_golden.py
  -k aim` 3 passed (default CENT rows), 11 `paper_full` rows skipped, 52 s.
- **Native build (verified 2026-11-04, when Docker cannot run):** the same
  tree builds out of tree with a C++20 toolchain (conda-forge `gxx_linux-64=12`
  into a private prefix); the dependency sources the Docker configure fetched
  into `ext/` are reused offline:

  ```bash
  mamba create -p $TOOLCHAIN -c conda-forge --override-channels gxx_linux-64=12 gcc_linux-64=12 cmake ninja
  cmake -S aim_simulator -B $BUILD -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DFETCHCONTENT_FULLY_DISCONNECTED=ON -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DCMAKE_C_COMPILER=$TOOLCHAIN/bin/x86_64-conda-linux-gnu-gcc \
      -DCMAKE_CXX_COMPILER=$TOOLCHAIN/bin/x86_64-conda-linux-gnu-g++
  cmake --build $BUILD -j 16
  export TENON_AIM_RAMULATOR2=$BUILD/ramulator2     # AIM_SIMULATOR_ROOT still names the tree
  ```

  The upstream `CMakeLists.txt` writes `libramulator.so` into the *source*
  tree (`LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}`), so a native build
  replaces the shared library the Docker-built `build/ramulator2` loads;
  both binaries come from one source revision and compiler major (g++ 12),
  and the Docker route must be re-smoked (689 cycles) after a native build.
  The native binary reproduced the smoke test (689 memory-system cycles),
  the AiM GEMV (222 cycles) and every exact CENT golden row
  (`tests/pim/test_paper_golden.py -k aim`, incl. the `paper_full` rows);
  each run records `{"kind": "native", "binary", "sha256", "config"}`.
- **Model scope:** timing only (no data path); `test/example.yaml` fixes 32
  channels, and Tenon rejects targets whose geometry differs from it.
- **License:** MIT. Redistribution is allowed with the notice; Tenon vendors
  no simulator code, only the Dockerfile.
- **What a user must do:** nothing beyond the recipe; this route is fully
  reproducible from this repository and the public upstream.

## AttAcc: Ramulator2-AttAcc (external evaluation only)

The `attacc` backend is analytical-only: `allo.compile(...,
backend="virtual")` needs nothing from this section, and no package code
calls the simulator (`backend=None` raises). This simulator is used only by
the evaluation script `benchmarks/attacc/scripts/run_ramulator.py`, which
times the program the compiler selected (`kind: "simulator-timing-only"`;
no data path).

- **What it is:** Ramulator 2.0 with the AttAcc HBM3-PIM DRAM and controller
  model (`src/dram/impl/HBM3-PIM.cpp`, presets `HBM3_8Gb_2R` and
  `HBM3_5.2Gbps`), as packaged by the DCC artifact
  (<https://github.com/SPIN-Research-Group/DCC>, directory
  `src/Ramulator2/`: `AttAcc/` sources, `build.sh`, `patches/`). The AttAcc
  trace format and timing model originate in the AttAcc simulator
  <https://github.com/scale-snu/attacc_simulator>.
- **Where the paper's build lived:** a DCC checkout at commit
  `212f86ea7f08da059836e3a1311d3a11afae8dbe` (checked out 2026-07-07) in
  another user's tree on the authors' host,
  `.../pim-compilation/repro/dcc-attacc/src/src/Ramulator2/ramulator2-AttAcc`
  (701,352 bytes) with `Ramulator2/AttAcc/libramulator.so`. It is not under
  `<repo>/experiments/simulators/` and Tenon has no recipe of its own yet.
- **GLIBC:** that binary needs GLIBC 2.32-2.34 and GLIBCXX 3.4.29-3.4.30; the
  authors' host (Ubuntu 20.04, GLIBC 2.31) cannot run it natively. The study
  ran it read-only inside the `aim-simulator-build` image (Ubuntu 24.04,
  GLIBC 2.39, see the AiM section) with the DCC checkout bind-mounted
  read-only; `run_ramulator.py` does the same when `ATTACC_RAMULATOR_IMAGE`
  is set.
- **Tenon-owned native build (round 51, 2026-11-05):** when no container
  is available (Docker's data root was full on the authors' host), build
  the binary from the MIT sources of the DCC tree with
  `benchmarks/attacc/scripts/build_ramulator_attacc.sh <DCC>/src/src/Ramulator2/AttAcc <out> [CXX [CC]]`
  (CMake >= 3.14, a C++20 compiler; the authors used the conda gcc 12
  toolchain of the native AiM build). It copies the sources once (the
  checkout may be read-only), configures with the vendored `ext/` dependencies
  and no network, and prints the `ATTACC_RAMULATOR` / `ATTACC_RAMULATOR_LIBDIR`
  exports. The native binary (sha256 `01e857a7...`) reproduces the archived
  Docker run exactly: 11,454 cycles for the qkv batch-1 selected program.

  ```bash
  export ATTACC_DCC_ROOT=<DCC checkout>/src/src      # attacc_trace_gen/, Ramulator2/
  export ATTACC_RAMULATOR=<out>/build/ramulator2     # or the DCC-built binary
  export ATTACC_RAMULATOR_LIBDIR=<out>/src           # libramulator.so
  # export ATTACC_RAMULATOR_IMAGE=aim-simulator-build  # container route instead
  python -m benchmarks.attacc.scripts.run_ramulator --row qkv_b1 --fixed-baseline \
      --artifacts out.artifacts --out out.json
  python -m benchmarks.attacc.scripts.run_ramulator --gemv 8 5120 384
  python -m benchmarks.attacc.scripts.run_ramulator --attention 8 128 480
  ```

  The script compiles the kernel through `allo.compile` and keeps the
  artifact; records the selection's provenance (compiler revision, the
  ranking's candidates, chosen position, ties and tie-break, the matched
  operations, the program manifest and estimate); writes the trace with
  DCC's generator into a private temporary directory; checks the trace
  against the selected program with `benchmarks/attacc/trace_check.py`
  (opcode totals, the phase ordering command by command, and the decoded
  address mapping: channels, banks, matrix columns per channel, vector
  positions; a mismatch fails the run) for both the GEMV and the attention
  traces (each re-derived from the program's counts record in the
  generator's emission order, validated against DCC's own generator output
  in `tests/pim/test_attacc_trace_check.py`); runs the simulator; and
  records the binary, library, trace
  and configuration digests, the command and the raw output, retaining
  trace, configuration and output per run under `--artifacts`. Labels:
  `shared-pipeline` (selection evidence), `forced-tiling` (`--tiling`, a
  controlled reference), `fixed-form-baseline` (`--fixed-baseline`: the
  paper's fixed AttAcc tiling as DCC's `predictor.py` defines it, the
  minimum simulated time over the fixed-form candidates, each simulated and
  recorded). Milliseconds use tCK 0.769 ns (`HBM3_5.2Gbps`), the paper's
  unit. Round 51 results (`tenon-release-evidence/r51/attacc_*.json`): qkv
  batch 1 / 4 and projection batch 1 / 4 selected programs 0.00881 /
  0.03364 / 0.00265 / 0.01024 ms, the paper's values; fixed baselines
  0.02807 / 0.17239 / 0.00265 / 0.01030 ms, the paper's; score batch 1 / 4
  over 248 shapes each: 1.00x / 1.13x over the fixed tiling (the figure's),
  123 wins / 316 ties / 61 losses against DCC's archived per-shape times on
  the 500 instances (the text's).
- **Licenses:** Ramulator2 (`Ramulator2/AttAcc/LICENSE`, MIT, SAFARI 2023)
  and the AttAcc simulator (MIT) permit redistribution with the notice.
  **DCC has no license file** (GitHub reports `license: null`), so DCC's
  trace generator (`attacc_trace_gen/gen_trace_AttAcc_*.py`), its
  `generate_configs.py` and its `inference/` decode model are neither
  vendored nor re-implemented here; the scripts import them at run time from
  the user's checkout. The compiler's tiling space, command counts and cost
  are an independent re-implementation (`allo/pim/attacc_tiling.py`).
- **Decode composition:** `benchmarks/attacc/scripts/compose_e2e.py` feeds
  per-kernel Ramulator2-AttAcc times to DCC's `inference/system.py`
  (analytical A100 roofline); `kind: "analytical-composition"`. Same DCC
  checkout, same license caveat.
- **What a user must do:** clone DCC, build `ramulator2-AttAcc` with its
  `build.sh` or with `build_ramulator_attacc.sh` (above), and set the
  variables above.

## UPMEM: uPIMulator and the UPMEM DPU SDK

- **Upstream:** <https://github.com/VIA-Research/uPIMulator>, revision
  `870d916334e9ff0b190f555f951a9ec3c4257781` (Go version under
  `golang/uPIMulator`).
- **What Tenon needs from uPIMulator:** a `benchmark/GENERIC` slot (Tenon
  writes `dpu/task.c` there), the `--fixture_dir` input path and observed-MRAM
  comparison, `--skip_compilation`, the linker and DPU-semantics fixes the
  campaign was measured with, and a `sdk/build` tree of DPU runtime objects
  compiled by the UPMEM SDK.
- **Where the Tenon changes are:** `simulators/uPIMulator/` in this
  repository: `tenon-upimulator.patch` (47 files, +1,930/-170, sha256
  `bf4887b825ab...`), `apply.sh`, the SDK image `Dockerfile`, and a README
  with the per-area change list and the upstream MIT license verbatim. The patch was derived on 2026-11-02 from the
  evaluation's provenance snapshot (`$TENON_ARTIFACTS/upmem/provenance/upimulator/`,
  `source-snapshot.tar` sha256 `13ed0d80...` and `tracked-changes.patch`
  sha256 `eba19cc5...`) by diffing it against a pristine clone of the pinned
  commit. Everything is text; no binary or fixture file is needed. Two
  deliberate differences from the snapshot, neither affecting the Go binary
  or the SDK objects: `benchmark/CMakeLists.txt` adds only `GENERIC` (the
  snapshot also listed two comparison benchmarks whose directories it does
  not contain), and the stale SimplePIM program left in the snapshot's
  `GENERIC/` slot is replaced by a placeholder `task.c` (Tenon clears the slot
  before every build). The lab checkout's older `TENON/`, `E5*`, `DSL*` slots
  are pre-release experiments and are not part of the patch.
- **Build recipe (verified; times from the verification run):**

  ```bash
  git clone https://github.com/VIA-Research/uPIMulator.git
  git -C uPIMulator checkout 870d916334e9ff0b190f555f951a9ec3c4257781
  bash $ALLO/simulators/uPIMulator/apply.sh uPIMulator   # patch + go build, ~7 s
  export UPIMULATOR_ROOT=$PWD/uPIMulator/golang/uPIMulator
  export UPIMULATOR_BIN=$UPIMULATOR_ROOT/build/uPIMulator

  # DPU SDK image under a new tag; <upmem-sdk> is your copy of the SDK (below)
  tar czf $UPIMULATOR_ROOT/docker/upmem-sdk.tar.gz -C <upmem-sdk> upmem-2023.1.0-Linux-x86_64   # ~32 s
  docker build -f $ALLO/simulators/uPIMulator/Dockerfile \
      -t tenon-upimulator-sdk:2023.1.0 $UPIMULATOR_ROOT/docker/                                # ~36 s
  export TENON_UPMEM_SDK_IMAGE=tenon-upimulator-sdk:2023.1.0

  # DPU runtime objects under $UPIMULATOR_ROOT/sdk/build                                       ~2 s
  docker run --rm -v $UPIMULATOR_ROOT:/root/uPIMulator $TENON_UPMEM_SDK_IMAGE \
      python3 /root/uPIMulator/sdk/build.py --num_tasklets 12
  ```

  `apply.sh` needs `go` (>= 1.21) on `PATH`. Docker runs as root in the SDK
  steps and in every Tenon run, so `sdk/build` and `benchmark/build` become
  root-owned. Setting `UPIMULATOR_ROOT` and `UPIMULATOR_BIN` selects the
  user-build route (`INSTALL.md` section 3): no snapshot staging, no SDK
  manifest pin, run records labelled `user-build:<manifest12>`. The
  pinned-campaign route (default) instead uses the archived binary and
  snapshot and the SDK object-manifest pin `d45e285c34b5...`; it exists for
  paper reproduction on the authors' archive.
- **Verification (2026-11-02, first round, `<repo>` patch):** fresh clone + `apply.sh` 7 s, `go test
  -vet=off ./...` 5 packages ok. The Go sources equal the snapshot's:
  `go build -trimpath -buildvcs=false` of the patched tree and of the snapshot
  give the same binary (sha256 `90022c862724...`); the default build
  (`90cc4ef291e6...`) differs from the archived campaign binary
  (`0ab6af912dc6...`) only through embedded paths (identical Go build info:
  go1.23.12, `vcs.revision=870d916`, `CGO_ENABLED=0`). SDK image
  `tenon-upimulator-sdk:2023.1.0` (id `sha256:db2d99e95fba...`, `docker build
  --no-cache`). The `sdk/build` object manifest is
  `d45e285c34b5ab53e02f9d8a7289725112a3a25c0c3ed8a8a15d47c41173963a`, equal to
  the pinned `UPMEM_SDK_BUILD_MANIFEST_SHA256` (and independent of
  `--num_tasklets`: 1, 12 and 16 give the same objects). With the three
  variables above set, all 16 UPMEM rows of `tests/pim/test_paper_golden.py`
  run through `allo.compile` reproduce the paper cycles exactly with
  NumPy-exact outputs, every DPU record labelled `sdk_build =
  "user-build:d45e285c34b5"`: VA 114,534; RED 83,117; MTV 359,135; **GEMV
  359,026**; GEVA 126,649; TTV 214,361; MMTV 211,371; HIST 183,782; SEL
  89,271; K-means 221,196; linear reg. 212,718; logistic reg. 208,442; 1mm
  3,135,600; 2mm 6,271,190; 3mm 9,406,780; conv 2,956,792 (501 s for all 16,
  GEMV 9 s). `tests/pim/test_gemv_all_targets.py -k upmem` (768x128 int32
  GEMV) 1 passed in 60 s on the same build.
- **Docker image naming:** the authors' local SDK image is tagged
  `bongjoonhyun/upimulator` (id `sha256:57dc7591...`, 2026-07-21), the same
  name as the upstream author's Docker Hub image, which does not contain the
  SDK the same way. `docker pull bongjoonhyun/upimulator` therefore does not
  give you the tested image. Build your own under a new tag as above and set
  `TENON_UPMEM_SDK_IMAGE` (the default name is kept only so the authors'
  host keeps working; P-17).
- **Licenses:** uPIMulator is MIT (`LICENSE`, "Copyright (c) 2024, VIA
  Research Group at KAIST", sha256 `73de95273ba8...` at `870d916`); the Tenon
  patch is redistributable with that notice, which
  `simulators/uPIMulator/README.md` reproduces verbatim. The patched `sdk/` runtime
  sources are UPMEM's BSD-style DPU runtime files that upstream uPIMulator
  already bundles. The UPMEM DPU SDK itself (compiler and libraries) is UPMEM
  proprietary; the CMU-SAFARI mirror carries no license file. Neither the
  SDK, the SDK tarball, nor any image containing it may be redistributed by
  this project.
- **What a user must do:** obtain the UPMEM SDK 2023.1.0 themselves (from
  UPMEM, or the mirror at their own discretion), then follow the recipe:
  clone uPIMulator, run `$ALLO/simulators/uPIMulator/apply.sh`, build the SDK
  image under their own tag and the `sdk/build` objects. The UPMEM compile and `backend="virtual"`
  routes need neither.

## Verification from the release copies (2026-11-02)

Every command below used only the files in `simulators/` of this repository
(plus, for Samsung, the authors' patches passed explicitly), fresh upstream
clones and new Docker tags. Python package: release `HEAD` `e41ebbd`
exported with `git archive` (MLIR bindings from the live build), so the run
does not depend on uncommitted work. Scratch: `/tmp/tenon-release/distbuild/`
(a symlink to local disk, because `/tmp` was full); `TMPDIR` and
`TENON_UPMEM_SCRATCH` pointed into it. `/docker` had about 1 GB free, so both
images were built with the layer cache: every layer resolved to the layers of
the first round's `docker build --no-cache` of byte-identical Dockerfiles, and
the image ids equal the first round's. Times are wall clock.

**SK hynix AiM** (all from the release copy):

```bash
git clone https://github.com/arkhadem/aim_simulator.git aim          # 14:18
git -C aim checkout 0f28a07bdb83e42b9305ad3d45410ebd3aa2c091
cp $ALLO/simulators/aim_simulator/Dockerfile aim/
docker build -t tenon-aim-simulator-build:dist aim                     # 0.6 s (cache)
docker run --rm -u $(id -u):$(id -g) -v $PWD/aim:/work tenon-aim-simulator-build:dist \
    bash -c "mkdir -p build && cd build && cmake .. && make -j8"       # 114.6 s
docker run --rm -u $(id -u):$(id -g) -v $PWD/aim:/work tenon-aim-simulator-build:dist \
    bash -c "cd /work && ./build/ramulator2 -f test/example.yaml -t test/all_isr.trace"
#   memory_system_cycles: 689
AIM_SIMULATOR_ROOT=$PWD/aim TENON_AIM_SIM_IMAGE=tenon-aim-simulator-build:dist \
    python -m pytest tests/pim/test_gemv_all_targets.py -k aim         # 2 passed in 27.6 s
```

Image `tenon-aim-simulator-build:dist` id
`sha256:fc69e92cf3583e8bd22498c5ad69508f073aa14618d31ae5b59a3e6675ecab38`
(666 MB). `build/ramulator2` sha256
`3a7d88a86a26b2af3a6251e47c16812efd8b2995c5f2c8f225be9f020c2777e7`,
`libramulator.so` `f7477027cd2a7e88cb77d648a33c83606d864271741d3dc068088fe0ba0c1f58`.
AiM GEMV of `test_gemv_all_targets.py` (256x256 BF16): **222 cycles**,
`kind="simulator-timing-only"`, same as the first round.

**UPMEM** (release copy of the patch, `apply.sh` and `Dockerfile`; user-obtained SDK):

```bash
git clone https://github.com/VIA-Research/uPIMulator.git upim        # 14:18
git -C upim checkout 870d916334e9ff0b190f555f951a9ec3c4257781
bash $ALLO/simulators/uPIMulator/apply.sh upim                        # 6.9 s, go1.23.12
export UPIMULATOR_ROOT=$PWD/upim/golang/uPIMulator UPIMULATOR_BIN=$PWD/upim/golang/uPIMulator/build/uPIMulator
cp <SDK tarball> $UPIMULATOR_ROOT/docker/upmem-sdk.tar.gz
docker build -f $ALLO/simulators/uPIMulator/Dockerfile \
    -t tenon-upimulator-sdk:dist $UPIMULATOR_ROOT/docker/              # 1.9 s (cache)
docker run --rm -v $UPIMULATOR_ROOT:/root/uPIMulator tenon-upimulator-sdk:dist \
    python3 /root/uPIMulator/sdk/build.py --num_tasklets 12            # 1.5 s, 148 steps
TENON_UPMEM_SDK_IMAGE=tenon-upimulator-sdk:dist \
    python -m pytest tests/pim/test_gemv_all_targets.py -k upmem       # 1 passed in 63.7 s
```

* SDK tarball: the authors' user-obtained copy, reused from the installed
  checkout's Docker context
  (`<repo>/experiments/simulators/uPIMulator/golang/uPIMulator/docker/upmem-sdk.tar.gz`,
  dated 2026-04-18, 204,556,701 bytes, sha256
  `9da2972ab7b8f39e87ae76c1fa789d1b6aff1f78741f2e6dbdacf3a558710240`). Its
  extracted `upmem-2023.1.0-Linux-x86_64/` equals (`diff -r`, no differences)
  the clean CMU-SAFARI mirror checkout at `84a0dcd` in
  `<repo>/experiments/simulators/upmem-sdk`. It is not, and must not be, in
  this repository.
* Go binary (`script/build.py`) sha256
  `1787f31228cd27e1e467c6840cb3258524255382c76292a3207872017d67b00b` (build
  info go1.23.12, `vcs.revision=870d916`, `CGO_ENABLED=0`); `go build
  -trimpath -buildvcs=false` gives
  `90022c862724528b6e5d790ca10b32493afd7840bb1e52ca9de0b4f466ccd2c6`, the
  provenance snapshot's binary; `go test -vet=off ./...` 5 packages ok.
* Image `tenon-upimulator-sdk:dist` id
  `sha256:db2d99e95fba4fcf12ba0873f7f5f42d7f026bb6bf2909384aad7299414fd287`.
  `sdk/build` object manifest
  `d45e285c34b5ab53e02f9d8a7289725112a3a25c0c3ed8a8a15d47c41173963a` = the
  pinned `UPMEM_SDK_BUILD_MANIFEST_SHA256`.
* `test_gemv_all_targets.py` GEMV (768x128 int32, 8 DPUs): NumPy-exact,
  **461,165 cycles**, DPU record `sdk_build = "user-build:d45e285c34b5"`,
  `simulator_sha256` `1787f312...`; the pinned-campaign route (archived binary
  `0ab6af912dc6...`) gives the same 461,165 cycles. Paper GEMV (96x128,
  `benchmarks.upmem.workloads` case `gemv`, canonical inputs) on the same
  build: **359,026 cycles** (the pinned paper value), NumPy-exact,
  `sdk_build = "user-build:d45e285c34b5"`, 9.2 s.

**Samsung HBM-PIM** (release copy; patches supplied from `<repo>`):

```bash
git clone https://github.com/SAITPublic/PIMSimulator.git pimsim       # 14:18
git -C pimsim checkout 3703d1f19c8f027360cc33a3243eb271e3bb6898
bash $ALLO/simulators/PIMSimulator/apply.sh pimsim
#   exit 2: "error: the Samsung-derived PIMSimulator patches are not available ...
#   The Samsung HBM-PIM simulator route cannot be built from the release
#   repository alone ..."; the checkout is left untouched
bash $ALLO/simulators/PIMSimulator/apply.sh pimsim <repo>/experiments/simulators/setup/PIMSimulator/patches
#   patched: 5 patches; scons pim_driver libdramsim/libdramsim2.a; 11.7 s
#   PIM_GEOMETRY channels=64 ranks=1 pim_blocks=8 banks=16 ... lanes=16 crf_words=32
PIMSIMULATOR_ROOT=$PWD/pimsim python -m pytest tests/pim/test_paper_golden.py -k samsung -s
#   3 passed in 130 s
```

* `pim_driver` sha256
  `210b7f542f5844d0320cae712a12355667dc3a733b8be5ea7568f1983bada948`,
  `libdramsim/libdramsim2.a`
  `08807bf9c0ad07a5811a69de58cf39fe1c80c9461d2f8320f64fb5a0ff7f2fc0`. After
  `objcopy --strip-all --remove-section=.note.gnu.build-id`:
  `9780b17c73965f7abc7060be7f434d119422d04198362c50caec3342e52b53e7`, equal to
  the installed lab binary stripped the same way. (The first round's
  `181fb3bb...` predates the `TenonConductor.inc` update of 2026-11-02 07:43;
  the lab binary was rebuilt at 07:54.) The five installed overlay files equal
  `<repo>`'s byte for byte.
* Vendor layout (`samsung_vendor_gemv_layout`) **4,435** / 8,041 / 8,247
  cycles and cost-selected default **4,389** / 7,947 / 8,153 for 4096x256 /
  4096x512 / 8192x256.
* Not verifiable from the release repository alone: without the patches the
  build stops by design.

**Recipe tests:** `tests/perf/test_simulator_recipes.py` (files present and
non-empty, pins and the uPIMulator license recorded, Samsung limitation in the
README, `apply.sh` refusal without or with an incomplete patch directory,
the three ways of supplying patches with synthetic non-Samsung patches, sdist
ships `simulators/**` but never `simulators/PIMSimulator/patches/`).

## GSI APU v1 (Gemini-I)

- **Needs:** the GSI APU SDK (ARC GNU toolchain
  `arc_gnu_2021.09-release_elf32_le_linux_no_sdata`, GDL/GAL host and device
  libraries, GVML headers under `<include>/gsi/libgvml_*.h`), the GSI
  `example-gvml` project (Tenon copies its `Makefile`/`Common/` harness), a
  Gemini-I PCI board (vendor `0x1e4c`), and `ledag-ssh`/`flo` for optional
  profile logs.
- **Discovery:** `TENON_APU_V1_TOOLCHAIN_BASE`, `TENON_APU_V1_GVML_INCLUDE_ROOT`,
  `TENON_APU_V1_TEMPLATE_DIR`, `TENON_APU_V1_PCI_NODE`.
- **License:** GSI proprietary; nothing from the SDK is bundled. The APU v1
  build harness copies files from the user's `example-gvml` at build time
  (V1-08 notes 12 GSI/Synopsys files without license headers in that
  project; they stay on the user's machine).
- **What a user must do:** obtain the SDK and board from GSI.

## GSI APU v2 (Gemini-II)

- **Needs:** the GSI Gemini-II SDK under `/opt/gsi` (headers
  `include/64vl`, `include/g2_common`, `include/gtml`, `include/gsi`;
  libraries `g2/vector_core/lib/libg2_64vl.a`, `lib/libg2_device_core.a`;
  `g2/vector_core/bin/g2_md_update.py`; CMake packages `lib/cmake/g2_device`,
  `lib/cmake/g2_transport`), `gsi_tool`, and a Gemini-II card (`apu-00`).
- **Discovery:** `TENON_APU_G2_SDK_ROOT`, `TENON_APU_G2_GSI_TOOL`,
  `TENON_APU_G2_DEVICE`; the shipped CMake templates still hard-code
  `/opt/gsi` and device 0, so only the defaults are accepted for the SDK root
  and device.
- **Templates:** `allo/pim/templates/apu_g2/**` are Allo-authored
  (Apache-2.0) and reference SDK headers by include path; no GSI header text
  is copied.
- **L1 functional simulator:** the `gsi-g2-l1sim` Docker image (built from a
  GSI-provided archive) is no longer used by the package: the source-route
  runner that probed it was removed (G2-01).
- **License:** GSI proprietary. **What a user must do:** obtain the SDK, tool
  and card from GSI.
