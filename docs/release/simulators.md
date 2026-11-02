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
own `.git` pointer is broken).

## Summary

| Dependency | Upstream | Pinned revision | Tenon changes | License | Rebuildable from public sources today? |
|---|---|---|---|---|---|
| Samsung PIMSimulator | <https://github.com/SAITPublic/PIMSimulator> | `3703d1f19c8f027360cc33a3243eb271e3bb6898` (2024-07-01) | `pim_driver` CLI and REDUCE/GENERIC kernel paths; partly committed, mostly not (below) | Samsung proprietary notice (no redistribution without written permission; non-commercial research only); DRAMSim2 part BSD | **No** |
| SK hynix AiM simulator (Ramulator 2.0 based) | <https://github.com/arkhadem/aim_simulator> | `0f28a07bdb83e42b9305ad3d45410ebd3aa2c091` (2025-07-22) | Dockerfile only | MIT (University of Michigan, 2025) | **Yes** |
| uPIMulator (Go) | <https://github.com/VIA-Research/uPIMulator> | `870d916334e9ff0b190f555f951a9ec3c4257781` (2025-02-01) | GENERIC benchmark slot, fixture loader, assembler and SDK build fixes (patch in the unpublished artifact archive) | MIT-style (VIA Research Group, KAIST, 2024) | **No** (patch unpublished) |
| UPMEM DPU SDK 2023.1.0 | UPMEM (`sdk-releases.upmem.com`, returned HTTP 403 when fetched); mirror <https://github.com/CMU-SAFARI/UPMEM> | mirror `84a0dcd3f57ded98ea049f2be5d45c71eda96ace` | none | UPMEM proprietary; the mirror ships no license file | user must obtain it |
| GSI APU v1 (Gemini-I) SDK: ARC toolchain, GDL/GAL/GVML, `example-gvml` | GSI Technology | `/usr/local/gsi-apu/13.7.1` on the authors' host | none | GSI proprietary | user must obtain it |
| GSI APU v2 (Gemini-II) SDK under `/opt/gsi`, `gsi_tool`, card firmware | GSI Technology | firmware `GSI_G2_Infrastructure_V1.24.1` on `zhang-x4` | none | GSI proprietary | user must obtain it |

## Samsung HBM-PIM: PIMSimulator

- **Upstream:** <https://github.com/SAITPublic/PIMSimulator>, revision
  `3703d1f19c8f027360cc33a3243eb271e3bb6898`.
- **What Tenon needs:** a `pim_driver` executable at `$PIMSIMULATOR_ROOT/pim_driver`
  that accepts `--op GENERIC --op-kind REDUCE --roles ... --reduce-k ...`
  (`allo/spmw_samsung.py` always passes `--op GENERIC`), plus the
  `ini/` configurations of the upstream tree.
- **Where the Tenon changes are:**
  - `<repo>/experiments/simulators/setup/PIMSimulator/{Sconstruct,src/pim_driver.cc}`
    and `setup/apply.sh`. This overlay is **stale**: its `pim_driver.cc` has
    323 lines and accepts only `ADD|MUL|RELU|GEMV`; a fresh clone + `apply.sh`
    produces a driver that cannot serve the Samsung backend.
  - The driver that produced the evaluation (`pim_driver`, sha256
    `3c5039f19ca22a334bc95d3521f121871b42e7fdf7aecde435dce4a882cc188a`,
    built 2026-09-29) was built from the installed checkout, which carries
    7 modified upstream files (`Sconstruct`, `src/Burst.h`,
    `src/tests/{KernelAddrGen.h,PIMCmdGen.cpp,PIMCmdGen.h,PIMKernel.cpp,PIMKernel.h}`)
    and untracked `src/pim_driver.cc` (721 lines),
    `src/SamsungSlicePlan.{cpp,h}`, `src/tests/TemporalReduce.inc`. Those
    sources equal an uncommitted working copy under
    `<repo>/dev/06292026-samsung-general-instr-codegen/pimsim-conductor/`
    (audit S-04 / P-05); `Burst.h`, `PIMCmdGen.*`, `SamsungSlicePlan.*` and
    `TemporalReduce.inc` are committed nowhere.
- **Build recipe (authors' host):** conda environment with Python 3.12,
  SCons and gtest 1.14 (`$CONDA_PREFIX` or `$GTEST_PREFIX` is used by the
  overlay `Sconstruct`), then `scons -j32 pim_driver` in the checkout
  (target defined by the overlay `Sconstruct`). On the authors' tree the
  gtest `sim` target no longer links because the untracked
  `src/gemv_bench_main.cc` defines a second `main`; upstream's smoke test
  `./sim --gtest_filter=PIMKernelFixture.gemv_4096x1024` works on a clean
  clone. No Docker image.
- **License and redistribution:** `LICENSE-PIMSimulator` states that the
  software is the property of Samsung Electronics, that no part may be copied
  or distributed without Samsung's express written permission, and that use
  is restricted to non-commercial academic research. Tenon's changes modify
  Samsung files (`PIMKernel.cpp/.h`, `KernelAddrGen.h` carry the Samsung
  notice), so they cannot be published as a modified source tree, and a
  `.patch` that embeds Samsung code also needs permission (open question in
  `audit/packaging.md` section 9). The containing repository currently tracks
  such modified files under `dev/` (P-04); they must not be part of any public
  tree.
- **What a user must do:** clone PIMSimulator themselves and apply the Tenon
  driver changes. Until a permissible patch (or an out-of-tree driver that
  links the unmodified simulator) is published, **the Samsung simulator route
  is not reproducible from public sources**; the Samsung compile and
  `backend="virtual"` routes need no simulator.

## SK hynix AiM: aim_simulator (Ramulator 2.0)

- **Upstream:** <https://github.com/arkhadem/aim_simulator>, revision
  `0f28a07bdb83e42b9305ad3d45410ebd3aa2c091`; no tracked modifications on
  the authors' host.
- **Tenon changes:** `<repo>/experiments/simulators/setup/aim_simulator/Dockerfile`
  (Ubuntu 24.04 + g++-12; the upstream ships no build environment, and the
  binary needs glibc 2.38).
- **Build recipe:**

  ```bash
  git clone https://github.com/arkhadem/aim_simulator.git
  cd aim_simulator && git checkout 0f28a07bdb83e42b9305ad3d45410ebd3aa2c091
  cp <repo>/experiments/simulators/setup/aim_simulator/Dockerfile .
  docker build -t aim-simulator-build .
  docker run --rm -v $PWD:/work aim-simulator-build \
      bash -c "mkdir -p build && cd build && cmake .. && make -j8"
  # smoke test
  docker run --rm -v $PWD:/work aim-simulator-build \
      bash -c "cd /work && ./build/ramulator2 -f test/example.yaml -t test/all_isr.trace"
  export AIM_SIMULATOR_ROOT=$PWD          # TENON_AIM_SIM_IMAGE if you tag the image differently
  ```

  The authors' image `aim-simulator-build` has id `sha256:215d37b48f08...`
  (built 2026-04-18 from the committed Dockerfile).
- **Model scope:** timing only (no data path); `test/example.yaml` fixes 32
  channels, and Tenon rejects targets whose geometry differs from it.
- **License:** MIT. Redistribution is allowed with the notice; Tenon vendors
  no simulator code, only the Dockerfile.
- **What a user must do:** nothing beyond the recipe; this route is fully
  reproducible from public sources.

## UPMEM: uPIMulator and the UPMEM DPU SDK

- **Upstream:** <https://github.com/VIA-Research/uPIMulator>, revision
  `870d916334e9ff0b190f555f951a9ec3c4257781` (Go version under
  `golang/uPIMulator`).
- **What Tenon needs from uPIMulator:** a `benchmark/GENERIC` slot (Tenon
  writes `dpu/task.c` there), the `--fixture_dir` input path and observed-MRAM
  window comparison, and `--skip_compilation`. None of these exist upstream.
- **Where the Tenon changes are:**
  - Committed in the containing repository:
    `<repo>/experiments/simulators/setup/uPIMulator/Dockerfile` (installs the
    SDK from a local tarball) and `setup/patch-upmem-benchmarks.sh`
    (benchmark CMake rewrites). These are not sufficient: the live checkout
    `<repo>/experiments/simulators/uPIMulator` has neither the GENERIC slot
    nor `--fixture_dir`.
  - The complete change set used by the evaluation is in the authors'
    unpublished artifact archive `$TENON_ARTIFACTS/upmem/provenance/upimulator/`:
    `tracked-changes.patch` (sha256 `eba19cc5...`, 32 files: Go assembler,
    compiler, linker, simulator ALU/logic, `main.go`, `benchmark/build.py`,
    benchmark CMake files, the bundled `sdk/` stdlib/syslib) plus untracked
    files captured in `source-snapshot.tar` (sha256 `13ed0d80...`):
    `benchmark/GENERIC/`, `src/assembler/prim/generic.go` and tests, and
    others. Built binary `uPIMulator` sha256 `0ab6af91...`, Go 1.23.12.
- **Build recipe (with the patch applied):**

  ```bash
  cd uPIMulator/golang/uPIMulator
  python script/build.py                      # -> build/uPIMulator
  # SDK image: put the DPU SDK tarball next to the Dockerfile
  tar czf docker/upmem-sdk.tar.gz -C <upmem-sdk> upmem-2023.1.0-Linux-x86_64
  cp <repo>/experiments/simulators/setup/uPIMulator/Dockerfile docker/Dockerfile
  docker build -t tenon-upimulator-sdk ./docker/
  export UPIMULATOR_ROOT=$PWD UPIMULATOR_BIN=$PWD/build/uPIMulator
  export TENON_UPMEM_SDK_IMAGE=tenon-upimulator-sdk
  ```

  Setting both `UPIMULATOR_ROOT` and `UPIMULATOR_BIN` selects the user-build
  route (`INSTALL.md` section 3): no snapshot staging, no SDK manifest pin,
  run records labelled `user-build:<manifest12>`. The pinned-campaign route
  (default) instead uses the archived binary and snapshot and the SDK
  object-manifest pin `d45e285c34b5...`; that is for paper reproduction only.
- **Docker image naming:** the authors' local SDK image is tagged
  `bongjoonhyun/upimulator` (id `sha256:57dc7591...`, 2026-07-21), the same
  name as the upstream author's Docker Hub image, which does not contain the
  SDK the same way. `docker pull bongjoonhyun/upimulator` therefore does not
  give you the tested image. Tag your own build differently and set
  `TENON_UPMEM_SDK_IMAGE` (the default name is kept only so the authors'
  host keeps working; P-17).
- **Licenses:** uPIMulator is MIT-style (KAIST VIA Research Group); a Tenon
  patch can be redistributed with the notice. The UPMEM DPU SDK is UPMEM
  proprietary; the CMU-SAFARI mirror carries no license file. Neither the SDK,
  the SDK tarball, nor any image containing it may be redistributed by this
  project.
- **What a user must do:** obtain the UPMEM SDK 2023.1.0 themselves (UPMEM,
  or the mirror at their own discretion), build the SDK image, and apply the
  Tenon uPIMulator patch. **The patch is not yet published** (P-06); until it
  is, the UPMEM simulator route needs the patch from the authors. The UPMEM
  compile and `backend="virtual"` routes need neither.

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
