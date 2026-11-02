<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# uPIMulator patch for Tenon (UPMEM backend)

Everything needed to turn an unmodified upstream uPIMulator checkout into the
`UPIMULATOR_ROOT` / `UPIMULATOR_BIN` pair that Tenon's UPMEM runtime
(`allo/spmw_upmem.py`) runs on its *user-build* route. The UPMEM DPU SDK is
not included and must be obtained by you (below).

## Upstream

* Repository: <https://github.com/VIA-Research/uPIMulator> (Go version under
  `golang/uPIMulator`)
* Commit: `870d916334e9ff0b190f555f951a9ec3c4257781` (2025-02-01, upstream
  `HEAD` on 2026-11-02)
* License: MIT (`LICENSE` at the root of the checkout, sha256
  `73de95273ba877b7c8b1037d8c478179fd58ff97169da6683bc10d88b79fd16d` at the
  commit above).

`tenon-upimulator.patch` contains Tenon's own code and modified hunks of
upstream uPIMulator files (including the UPMEM BSD-style DPU runtime sources
that upstream already bundles under `sdk/`). It is distributed under the
upstream license, reproduced verbatim:

```text
Copyright (c) 2024, VIA Research Group at KAIST

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

## Contents

| File | Purpose |
|---|---|
| `tenon-upimulator.patch` | 47-file unified diff (`git apply` or `patch -p1` in the checkout root), sha256 `bf4887b825abfe01141ddf4ff04274f468aef9980a2e0763421f3ddce6f05ca0` |
| `apply.sh` | applies the patch (idempotent) and builds `golang/uPIMulator/build/uPIMulator` with `script/build.py` (needs `go` >= 1.21 on `PATH`; `TENON_SKIP_BUILD=1` skips the build) |
| `Dockerfile` | DPU SDK image recipe (Ubuntu 22.04 + SDK 2023.1.0 from a local `upmem-sdk.tar.gz`); the patch's `golang/uPIMulator/docker/Dockerfile` is the same recipe apart from one comment, so either can be used |

No binary or fixture file is needed: every file of the patch is text. The
DPU SDK objects under `sdk/build` are build outputs (step 4 below), not part
of the patch.

### What the patch changes

Derived on 2026-11-02 from the evaluation's provenance snapshot (sha256
`13ed0d80...`, plus its `tracked-changes.patch`, sha256 `eba19cc5...`) by
diffing against a pristine clone of the commit above.

| Area | Files | Change |
|---|---|---|
| GENERIC benchmark slot | `benchmark/GENERIC/**` (new), `benchmark/CMakeLists.txt` | Slot that receives Tenon's generated `dpu/task.c`; every `*.c` in `dpu/` is its own translation unit. `dpu/task.c` is a placeholder that Tenon overwrites before each build. |
| Fixture loader | `src/assembler/prim/generic.go` (+ test), `src/assembler/assembler.go`, `src/main.go` | `--benchmark GENERIC --fixture_dir <dir>`: loads a version-1 `manifest.json` with raw MRAM/host input blobs and complete expected outputs, compares every observed output byte. `src/main.go` also adds `--skip_compilation` |
| Compilation skip | `src/compiler/compiler.go` | `--skip_compilation true` reuses the benchmark and SDK assembly already built in Docker (Tenon builds the slot itself); the Docker calls drop `--privileged` |
| Benchmark build | `benchmark/build.py` | `--benchmark <name>` builds only `<name>_device`; propagates CMake/ninja failures |
| Linker | `src/linker/linker.go`, `kernel/executable.go`, `parser/parser.go`, `logic/linker_script.go` (+ tests) | multi-translation-unit staging; MRAM/WRAM section placement with gap preservation; SDK-aligned stack and cache symbols; unused stdio MRAM buffer not linked |
| DPU semantics | `src/simulator/dpu/logic/{alu,logic}.go` (+ tests) | signed/unsigned pair loads; chained-subtraction borrow and unsigned flags; UPMEM DMA address masking |
| Host read-back | `src/simulator/host/{host,channel_transfer_read_job}.go` (+ test) | every observed output byte stream persisted for hashing |
| DPU runtime sources | `sdk/stdlib/stdio.c`, `sdk/syslib/dpuruntime.h`, `sdk/misc/stdoutBufferState.c` (new) | the bootstrap's stdout buffer state moves to its own object, so a program that does not use stdio no longer links `stdio.c` and its 1 MiB MRAM buffer |
| SDK image and CMake | `docker/Dockerfile`, `sdk/*/CMakeLists.txt`, `benchmark/*/dpu/CMakeLists.txt` | image installs SDK 2023.1.0 from a local tarball (upstream downloads 2023.2.0 from a host that returns HTTP 403); compiler is `dpu-upmem-dpurte-clang` from `PATH` |
| Unused by Tenon | `src/assembler/prim/{simplepim_va,atim_va}.go` (+ test) | assemblers for two comparison benchmarks of the evaluation; kept so the Go binary is built from exactly the snapshot's Go sources. Their benchmark directories are not included |

## The UPMEM DPU SDK (you must obtain it yourself)

The DPU programs are compiled with the UPMEM SDK 2023.1.0
(`dpu-upmem-dpurte-clang`). It is UPMEM proprietary software; Tenon does not
ship it, the SDK tarball, or any image containing it, and you must not
redistribute such an image (`docs/release/ISSUES.md` P-17). The official
download host (`sdk-releases.upmem.com`) returned HTTP 403 when we fetched
it; the evaluation used the CMU-SAFARI mirror at its own discretion:

```bash
git clone https://github.com/CMU-SAFARI/UPMEM.git upmem-sdk   # mirror; no license file
git -C upmem-sdk checkout 84a0dcd3f57ded98ea049f2be5d45c71eda96ace
# provides upmem-sdk/upmem-2023.1.0-Linux-x86_64/
```

Obtain it from UPMEM if you can; use the mirror only if its terms suit you.

## Build

```bash
# 1. upstream at the pinned commit
git clone https://github.com/VIA-Research/uPIMulator.git
git -C uPIMulator checkout 870d916334e9ff0b190f555f951a9ec3c4257781

# 2. patch + Go simulator (go >= 1.21 on PATH)
bash $ALLO/simulators/uPIMulator/apply.sh uPIMulator
export UPIMULATOR_ROOT=$PWD/uPIMulator/golang/uPIMulator
export UPIMULATOR_BIN=$UPIMULATOR_ROOT/build/uPIMulator

# 3. DPU SDK image under a NEW tag (never reuse bongjoonhyun/upimulator,
#    the upstream author's Docker Hub name); upmem-sdk/ is your SDK copy
tar czf $UPIMULATOR_ROOT/docker/upmem-sdk.tar.gz -C upmem-sdk upmem-2023.1.0-Linux-x86_64
docker build -f $ALLO/simulators/uPIMulator/Dockerfile -t tenon-upimulator-sdk:2023.1.0 $UPIMULATOR_ROOT/docker/
export TENON_UPMEM_SDK_IMAGE=tenon-upimulator-sdk:2023.1.0

# 4. DPU runtime objects (sdk/build) inside the image
docker run --rm -v $UPIMULATOR_ROOT:/root/uPIMulator $TENON_UPMEM_SDK_IMAGE \
    python3 /root/uPIMulator/sdk/build.py --num_tasklets 12

# optional checks
(cd $UPIMULATOR_ROOT/src && go test -vet=off ./...)   # 5 packages ok
```

Docker runs as root and leaves root-owned files under `$UPIMULATOR_ROOT`
(`sdk/build`, `benchmark/build`). Tenon takes `run.lock` in the directory
above `$UPIMULATOR_ROOT` and rewrites `benchmark/GENERIC/dpu/` on every run,
so give each concurrent user their own root.

With `UPIMULATOR_ROOT`, `UPIMULATOR_BIN` and `TENON_UPMEM_SDK_IMAGE` set,
`allo.compile(..., build_upmem_target(), upmem_cost)` runs on this build and
labels every DPU record `sdk_build = "user-build:<manifest12>"`.

## Verification

* Step 4 produces the SDK object manifest
  `d45e285c34b5ab53e02f9d8a7289725112a3a25c0c3ed8a8a15d47c41173963a`, equal to
  the pinned `UPMEM_SDK_BUILD_MANIFEST_SHA256` in `allo/spmw_simenv.py`.
* `go build -trimpath -buildvcs=false` in `golang/uPIMulator/src` gives sha256
  `90022c862724528b6e5d790ca10b32493afd7840bb1e52ca9de0b4f466ccd2c6`, the same
  binary as the evaluation's source snapshot.
* `python -m pytest tests/pim/test_gemv_all_targets.py -k upmem` passes; the
  paper GEMV (96x128) reports 359,026 cycles labelled
  `user-build:d45e285c34b5`. Recorded runs: `docs/release/simulators.md`.
