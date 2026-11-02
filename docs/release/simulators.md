<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Simulators and vendor SDKs used by Tenon backends

Tenon does not bundle any simulator, vendor SDK or Docker image. This page
records, for each external dependency, where it comes from, the revision the
evaluation used, where the Tenon-side changes live, how to build it, its
license, and what a user has to obtain on their own. Discovery variables are
in `INSTALL.md` section 3.

Paths below of the form `<repo>/...` are in the containing Tenon repository
(the repository that holds this Allo fork under `experiments/allo`), not in
this package. Revisions were read on the authors' host on 2026-11-02 with
`git -C <checkout> rev-parse HEAD` (PIMSimulator through the parent
repository's `.git/modules/simulators/PIMSimulator`, because the checkout's
own `.git` pointer is broken); on that date each pinned commit was also the
upstream `HEAD`.

The three simulator recipes below were verified on 2026-11-02 by building
from fresh upstream clones in scratch directories (`/tmp/tenon-release/simbuild/`)
and new Docker tags, without touching the installed simulators, then running
the Tenon tests through `allo.compile` against the scratch builds. Host: Linux
x86-64 (Ubuntu 20.04 userland), Docker 24.0.2, system g++ 9.4.0 and SCons
3.1.2, Go 1.23.12.

## Summary

| Dependency | Upstream | Pinned revision | Tenon changes (in `<repo>/experiments/simulators/setup/`) | License | Rebuildable from public sources? |
|---|---|---|---|---|---|
| Samsung PIMSimulator | <https://github.com/SAITPublic/PIMSimulator> | `3703d1f19c8f027360cc33a3243eb271e3bb6898` (2024-07-01) | `PIMSimulator/`: `pim_driver` conductor sources, `Sconstruct`, 5 patches, `apply.sh` | Samsung proprietary notice (no redistribution without written permission; non-commercial research only); DRAMSim2 part BSD | **Yes, with the user's own clone** (verified; publishing the patches is a licensing question, below) |
| SK hynix AiM simulator (Ramulator 2.0 based) | <https://github.com/arkhadem/aim_simulator> | `0f28a07bdb83e42b9305ad3d45410ebd3aa2c091` (2025-07-22) | `aim_simulator/Dockerfile` | MIT (University of Michigan, 2025) | **Yes** (verified) |
| Ramulator2-AttAcc (evaluation script only; not used by the package) | DCC artifact <https://github.com/SPIN-Research-Group/DCC> `src/Ramulator2/` (AttAcc model from <https://github.com/scale-snu/attacc_simulator>) | DCC `212f86ea7f08da059836e3a1311d3a11afae8dbe` | none (no Tenon recipe yet) | Ramulator2 and AttAcc simulator MIT; DCC (trace generator, `inference/`) **no license** | user builds it from the DCC checkout (`build.sh`); no Tenon recipe |
| uPIMulator (Go) | <https://github.com/VIA-Research/uPIMulator> | `870d916334e9ff0b190f555f951a9ec3c4257781` (2025-02-01) | `uPIMulator/tenon-upimulator.patch` (47 files), `apply.sh`, README | MIT-style (VIA Research Group, KAIST, 2024) | **Yes** (verified), given the user-obtained SDK |
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
- **Where the Tenon changes are:** `<repo>/experiments/simulators/setup/PIMSimulator/`
  (README with the file list). Allo-authored files copied in:
  `Sconstruct`, `src/pim_driver.cc`, `src/tests/TenonConductor.{h,inc}`,
  `src/tests/TemporalReduce.inc`. Samsung files changed only through
  `patches/{PIMKernel.h,PIMKernel.cpp,PIMCmdGen.h,PIMCmdGen.cpp,Burst.h}.patch`.
  The earlier `--slice-plan` route (`SamsungSlicePlan.*`) and the stale
  323-line overlay are gone; every overlay file now equals the installed
  tree's, and the installed tree has no other modified tracked file.
- **Build recipe (verified):**

  ```bash
  git clone https://github.com/SAITPublic/PIMSimulator.git
  git -C PIMSimulator checkout 3703d1f19c8f027360cc33a3243eb271e3bb6898
  bash <repo>/experiments/simulators/setup/PIMSimulator/apply.sh PIMSimulator
  #   applies the patches, installs the conductor sources, runs
  #   `scons pim_driver libdramsim/libdramsim2.a -j$(nproc)`, prints PIM_GEOMETRY
  export PIMSIMULATOR_ROOT=$PWD/PIMSimulator
  ```

  Needs g++ with C++14 and SCons; no gtest for `pim_driver`. No Docker image.
- **Verification (2026-11-02):** clone + `apply.sh` took about 12 s (48
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
- **What a user must do:** clone PIMSimulator themselves (accepting Samsung's
  notice) and run `apply.sh`. The Samsung compile and `backend="virtual"`
  routes need no simulator.

## SK hynix AiM: aim_simulator (Ramulator 2.0)

- **Upstream:** <https://github.com/arkhadem/aim_simulator>, revision
  `0f28a07bdb83e42b9305ad3d45410ebd3aa2c091`; no tracked modifications on
  the authors' host.
- **Tenon changes:** `<repo>/experiments/simulators/setup/aim_simulator/Dockerfile`
  (Ubuntu 24.04 + g++-12; the upstream ships no build environment, and the
  binary needs glibc 2.38).
- **Build recipe (verified):**

  ```bash
  git clone https://github.com/arkhadem/aim_simulator.git
  cd aim_simulator && git checkout 0f28a07bdb83e42b9305ad3d45410ebd3aa2c091
  cp <repo>/experiments/simulators/setup/aim_simulator/Dockerfile .
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
- **Verification (2026-11-02):** `docker build --no-cache` 27 s (image
  `tenon-aim-simulator-build:0f28a07`, id `sha256:fc69e92cf358...`, base
  image already cached); `cmake` + `make -j8` 120 s. With
  `AIM_SIMULATOR_ROOT` at the scratch clone and
  `TENON_AIM_SIM_IMAGE=tenon-aim-simulator-build:0f28a07`:
  `tests/pim/test_gemv_all_targets.py -k aim` 2 passed in 30 s; the AiM GEMV
  (256x256 BF16) reports 222 cycles, the same as on the installed checkout and
  the authors' image `aim-simulator-build` (id `sha256:215d37b48f08...`,
  built 2026-04-18 from the same Dockerfile); `tests/pim/test_paper_golden.py
  -k aim` 3 passed (default CENT rows), 11 `paper_full` rows skipped, 52 s.
- **Model scope:** timing only (no data path); `test/example.yaml` fixes 32
  channels, and Tenon rejects targets whose geometry differs from it.
- **License:** MIT. Redistribution is allowed with the notice; Tenon vendors
  no simulator code, only the Dockerfile.
- **What a user must do:** nothing beyond the recipe; this route is fully
  reproducible from public sources.

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
  read-only. `run_ramulator.py` does the same when
  `ATTACC_RAMULATOR_IMAGE` is set:

  ```bash
  export ATTACC_DCC_ROOT=<DCC checkout>/src/src      # attacc_trace_gen/, Ramulator2/
  export ATTACC_RAMULATOR_IMAGE=aim-simulator-build  # omit to run natively
  python -m benchmarks.attacc.scripts.run_ramulator --gemv 8 5120 384
  python -m benchmarks.attacc.scripts.run_ramulator --attention 8 128 480
  ```

  The script writes the trace, the YAML and `temp/log/` into a private
  temporary directory, mounts only that directory writable, and checks the
  trace's per-opcode counts against the compiled `AttAccCounts`. Smoke run
  2026-11-02: 11,454 cycles (qkv batch 1, selected tiling) and 953 cycles
  (attention 8 x 128 x 480, the paper's worst loss), equal to the archived
  values.
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
  `build.sh` (a container with GLIBC >= 2.34 if the host is older), and set
  the variables above. A Tenon-owned rebuild recipe from the MIT sources is
  open.

## UPMEM: uPIMulator and the UPMEM DPU SDK

- **Upstream:** <https://github.com/VIA-Research/uPIMulator>, revision
  `870d916334e9ff0b190f555f951a9ec3c4257781` (Go version under
  `golang/uPIMulator`).
- **What Tenon needs from uPIMulator:** a `benchmark/GENERIC` slot (Tenon
  writes `dpu/task.c` there), the `--fixture_dir` input path and observed-MRAM
  comparison, `--skip_compilation`, the linker and DPU-semantics fixes the
  campaign was measured with, and a `sdk/build` tree of DPU runtime objects
  compiled by the UPMEM SDK.
- **Where the Tenon changes are:**
  `<repo>/experiments/simulators/setup/uPIMulator/tenon-upimulator.patch`
  (47 files, +1,930/-170, sha256 `bf4887b825ab...`), `apply.sh` and a README
  with the per-area change list. The patch was derived on 2026-11-02 from the
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
  bash <repo>/experiments/simulators/setup/uPIMulator/apply.sh uPIMulator   # patch + go build, ~7 s
  export UPIMULATOR_ROOT=$PWD/uPIMulator/golang/uPIMulator
  export UPIMULATOR_BIN=$UPIMULATOR_ROOT/build/uPIMulator

  # DPU SDK image under a new tag; <upmem-sdk> is your copy of the SDK (below)
  tar czf $UPIMULATOR_ROOT/docker/upmem-sdk.tar.gz -C <upmem-sdk> upmem-2023.1.0-Linux-x86_64   # ~32 s
  docker build -t tenon-upimulator-sdk:2023.1.0 $UPIMULATOR_ROOT/docker/                       # ~36 s
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
- **Verification (2026-11-02):** fresh clone + `apply.sh` 7 s, `go test
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
- **Licenses:** uPIMulator is MIT-style (KAIST VIA Research Group); the Tenon
  patch is redistributable with that notice. The patched `sdk/` runtime
  sources are UPMEM's BSD-style DPU runtime files that upstream uPIMulator
  already bundles. The UPMEM DPU SDK itself (compiler and libraries) is UPMEM
  proprietary; the CMU-SAFARI mirror carries no license file. Neither the
  SDK, the SDK tarball, nor any image containing it may be redistributed by
  this project.
- **What a user must do:** obtain the UPMEM SDK 2023.1.0 themselves (from
  UPMEM, or the mirror at their own discretion), then follow the recipe:
  clone uPIMulator, run `apply.sh`, build the SDK image under their own tag
  and the `sdk/build` objects. The UPMEM compile and `backend="virtual"`
  routes need neither.

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
