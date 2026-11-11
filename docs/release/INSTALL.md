<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Installing Tenon (the PIM extensions of Allo)

Tenon lives inside the Allo package (`allo.spmw_*`, `allo.pim`, `allo.perf`).
Installing Tenon is installing this Allo fork. The supported installation is
upstream Allo's: build LLVM/MLIR, then build Allo's MLIR Python bindings from
source with CMake (`docs/source/setup/index.rst`, "Install from Source").
There is no prebuilt binary wheel. Once the bindings are built, either an
editable install or a regular wheel works; the wheel route was validated on
this host from the source distribution, outside the checkout (transcript in
section 5).

What you need depends on what you want to run:

| Goal | Needs |
|---|---|
| Compile, inspect, and get analytical estimates (`backend="virtual"`) for every PIM target | Allo + MLIR bindings only |
| Samsung HBM-PIM simulation | + the bundled PIMSimulator (`pim_driver`), built with `python -m allo.pim.samsung_simulator build`; needs g++, SCons and `patch` (section 3a) |
| SK hynix AiM timing simulation | + aim_simulator at `0f28a07` + a Docker image built from Tenon's Dockerfile (section 3a) |
| UPMEM simulation | + uPIMulator at `870d916` with `tenon-upimulator.patch` + a UPMEM DPU SDK 2023.1.0 Docker image you build from your own copy of the SDK (section 3a) |
| GSI APU v1 (Gemini-I) hardware | + GSI APU SDK, ARC toolchain, GVML headers, `example-gvml` project, a Gemini-I board |
| GSI APU v2 (Gemini-II) hardware | + GSI Gemini-II SDK under `/opt/gsi`, `gsi_tool`, a Gemini-II card |

Of the simulators, only Samsung's is bundled: the package carries the
PIMSimulator sources and Tenon's conductor as a build recipe
(`python -m allo.pim.samsung_simulator build`, section 3; read its `NOTICE.md`
for the non-commercial research terms). The AiM Ramulator model, uPIMulator,
the UPMEM SDK and the GSI SDKs are external and are not bundled.
`simulators.md` lists where each comes from, its license, and which parts you
have to obtain yourself.
`CAPABILITIES.md` lists what each backend can do and how well that is
verified.

## 1. Prerequisites

- Linux x86-64, CPython 3.12 (`python_requires=">=3.12"`).
- CMake >= 3.20 and Ninja. `setup.py` calls the `cmake` on `PATH`. Many
  distributions ship an older one (Ubuntu 20.04: 3.16.3); then run
  `python3 -m pip install cmake ninja` in the build interpreter's
  environment and put its `bin` directory first on `PATH`. With an older
  `cmake` first on `PATH`, `build_ext` stops before configuring with
  `RuntimeError: CMake >= 3.20 is required to build the Allo MLIR bindings,
  but cmake on PATH is 3.16. Run python3 -m pip install cmake and put its bin
  directory first on PATH.` (checked against the system cmake 3.16.3 on
  2026-11-02; P-20).
- `nanobind>=2.9,<3.0` in the build interpreter.
- An LLVM/MLIR build with Python bindings
  (`-DLLVM_ENABLE_PROJECTS="clang;mlir;openmp" -DMLIR_ENABLE_BINDINGS_PYTHON=ON`).
  The pinned revision is the `externals/llvm-project` submodule
  (`git submodule status externals/llvm-project`). Build it exactly as in
  `docs/source/setup/index.rst`.
- The C++ standard library that LLVM was built against must be the one the
  bindings link with. If LLVM was built with a newer GCC than the system
  default (as on the authors' host, GCC 13), put that libstdc++ first, for
  example `LDFLAGS="-L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib"` for a
  conda environment that ships it. Otherwise the final link of
  `libAlloMLIRAggregateCAPI.so` fails with undefined
  `std::__glibcxx_assert_fail`.
- Python dependencies (P-13):
  - `requirements.txt` is the runtime list and the only file `setup.py`
    reads for `install_requires`: numpy, PyYAML (MLIR `linalg` opdsl), sympy,
    ml_dtypes, rich, tabulate, xmltodict, pandas, pyparsing, graphviz (the
    last four for upstream Allo's report and AIE modules). `setup.py` rejects
    pip options and direct URLs in that file.
  - `requirements-dev.txt` (`-r requirements.txt` plus build, test and lint
    tools: cmake, ninja, nanobind, pybind11, pytest, clang-format, black,
    pylint) is for developers and CI. It also lists upstream Allo's optional
    features, which are imported only when used and are not installed by
    `pip install`: `gurobipy` (commercial solver client; `allo.autoscheduler`)
    and the direct-URL `past` wheel (PAST verifier, cp312 Linux only;
    `allo/verify.py`).
  - `python3 -m pip install -r requirements-dev.txt` before running the test
    suite. A venv holding only `requirements.txt` imports `allo` and compiles
    the Samsung, UPMEM and AiM GEMVs with `backend="virtual"` (checked
    2026-11-02).

## 2. Source install

```bash
git clone --recursive <this repository> allo && cd allo
export ALLO_ROOT=$PWD
# Build LLVM/MLIR as in docs/source/setup/index.rst, then:
export LLVM_BUILD_DIR=$ALLO_ROOT/externals/llvm-project/build   # or your MLIR build
python3 -m pip install cmake ninja "nanobind>=2.9,<3.0"

# (a) editable, as upstream documents
python3 -m pip install -v -e .

# (b) regular install from the source distribution
python3 setup.py sdist                       # or: python3 -m build --sdist
python3 -m pip wheel --no-deps --no-build-isolation -w dist dist/allo-0.5.tar.gz
python3 -m pip install dist/allo-0.5-*.whl
```

How the bindings reach the package:

- `CMakeBuild` (`setup.py`) configures `mlir/` against
  `$LLVM_BUILD_DIR/lib/cmake/mlir` and builds into `mlir/build`. The
  bindings land in `mlir/build/tools/allo/_mlir`.
- Editable installs import them through the tracked symlink
  `allo/_mlir -> ../mlir/build/tools/allo/_mlir`.
- Every other build copies that tree into the wheel as `allo/_mlir`
  (symlinks into the LLVM source tree are copied as files; the unversioned
  `libAlloMLIRAggregateCAPI.so` alias is dropped). The extensions use
  `$ORIGIN`-relative RPATHs, so the installed copy no longer depends on the
  build tree (validated by hiding the build tree after installation).
- `setup.py` passes `-DPython_EXECUTABLE` as well as `-DPython3_EXECUTABLE`;
  without it nanobind can pick another interpreter on `PATH` and build
  `cpython-313` extensions for a 3.12 package (observed on this host).

Failure modes you may see:

| Situation | Message |
|---|---|
| `LLVM_BUILD_DIR` unset | `RuntimeError: LLVM_BUILD_DIR environment variable is not set. Allo's MLIR Python bindings are built from source ...` |
| `LLVM_BUILD_DIR` is not an MLIR build | `RuntimeError: LLVM_BUILD_DIR=... is not an MLIR build: .../MLIRConfig.cmake does not exist ...` |
| `cmake` older than 3.20 | `RuntimeError: CMake >= 3.20 is required ... Run python3 -m pip install cmake ...` |
| package installed without bindings (for example `pip install` of a wheel built with `--skip-build`) | `import allo` -> `ModuleNotFoundError: No module named 'allo._mlir'` raised from `allo/ir/types.py:9` |

What the source distribution contains (checked by
`tests/perf/test_packaging.py`): the `allo` package including every
non-Python runtime asset (`allo/pim/templates/apu_g2/**`,
`allo/ir/template/*.mlir`, `allo/autoscheduler/outline_loops.mlir`,
`allo/harness/**`, `allo/library/aie/{kernels,modules}`), the `mlir/` C++
sources needed by `build_ext`, `requirements.txt`, `README.md`, `LICENSE`.
It does not contain `allo/_mlir` (rebuilt), `mlir/build`, `tests/`,
`examples/`, `benchmarks/`, `tutorials/` or `playground/`. Only `allo` is
installed as a top-level package (`examples` and `benchmarks` used to be
installed as generic top-level packages, audit P-12). The tests run from a
git checkout.

Build release artifacts from a clean checkout (`git clone` or
`git worktree`): `MANIFEST.in` grafts `mlir/`, so untracked files under
`mlir/` in a working tree end up in the sdist.

## 3. PIM backend discovery

Every location outside the package is owned by `allo/spmw_simenv.py` and
has an environment override. When a backend cannot run, its runner raises
`SimulatorUnavailable(target, reason)`; the reason names the variable to set.
`allo.spmw_codegen.simulator_unavailable_reason(target)` returns the same
reason without running anything.

| Variable | Meaning | Default when unset |
|---|---|---|
| `TENON_SIMULATORS_DIR` | Directory holding `PIMSimulator/` and `aim_simulator/` | `<repo>/experiments/simulators` when the package runs from the Tenon repository checkout; otherwise unset (installed copies must set it or the per-simulator roots) |
| `PIMSIMULATOR_ROOT` | Samsung PIMSimulator tree containing `pim_driver` | `$TENON_SIMULATORS_DIR/PIMSimulator`; when neither is set, the bundled simulator's build root below, once it holds `pim_driver` |
| `TENON_HOME` | Where Tenon keeps what it builds for the user; the bundled Samsung simulator installs under `$TENON_HOME/simulators/PIMSimulator` (`allo.spmw_simenv.default_pimsim_root()`) | `$XDG_DATA_HOME/tenon`, else `~/.local/share/tenon` |
| `AIM_SIMULATOR_ROOT` | aim_simulator tree (`build/ramulator2`, `test/example.yaml`) | `$TENON_SIMULATORS_DIR/aim_simulator` |
| `TENON_AIM_SIM_IMAGE` | Docker image that runs `ramulator2` | `aim-simulator-build` |
| `TENON_AIM_RAMULATOR2` | a natively built `ramulator2` binary to run instead of the Docker image (its SHA-256 is recorded with every run) | unset (Docker) |
| `UPIMULATOR_ROOT` | uPIMulator Go project root (`benchmark/`, `sdk/`) | the archived source snapshot, staged under `$TENON_UPMEM_SCRATCH` |
| `UPIMULATOR_BIN` | uPIMulator binary | `$TENON_UPMEM_PROVENANCE_DIR/uPIMulator` |
| `TENON_UPMEM_SDK_IMAGE` | Docker image holding the UPMEM DPU SDK | `bongjoonhyun/upimulator` (a local build that reuses the upstream image name; see `simulators.md`) |
| `TENON_UPMEM_SDK_HOME` | A host-installed UPMEM DPU SDK (`bin/dpu-upmem-dpurte-clang`): the DPU build runs natively through a user-namespace bind at the image's path instead of the Docker image (needs cmake, ninja, unshare, user namespaces; `simulators.md`) | unset (the Docker image) |
| `TENON_UPMEM_BENCHMARK_SLOT` | `benchmark/<slot>` that receives the generated `task.c` | `GENERIC` |
| `TENON_UPMEM_SCRATCH` | staged roots, per-run directories, `root.lock`, `run.lock` | `<tmpdir>/tenon-upmem` |
| `TENON_UPMEM_PROVENANCE_DIR` | archived campaign uPIMulator binary and `source-snapshot.tar` | `$TENON_ARTIFACTS/upmem/provenance/upimulator` |
| `TENON_ARTIFACTS` | Tenon evaluation-artifact archive (not published) | `~/shared/tenon-artifacts` (authors' lab layout, fallback only) |
| `TENON_APU_V1_TOOLCHAIN_BASE` | ARC GNU toolchain root, without `arc-snps-elf/` | `/usr/local/gsi-apu/13.7.1/ubuntu_20_04/arc_gnu_2021.09-release_elf32_le_linux_no_sdata` |
| `TENON_APU_V1_GVML_INCLUDE_ROOT` | directory holding `gsi/libgvml_*.h` | `/usr/local/include` |
| `TENON_APU_V1_TEMPLATE_DIR` | GSI `example-gvml` project (ships with the GSI SDK) | `~/shared/accelerator-hub/gsi-apu/example-gvml` (authors' lab layout, fallback only) |
| `TENON_APU_V1_PCI_NODE` | sysfs node of the Gemini-I board | scan `/sys/bus/pci/devices` for vendor `0x1e4c` |
| `TENON_APU_V1_LOCK` | board lock file taken by every Gemini-I launch (read by `allo/pim/apu_v1_program.py`, not yet by `spmw_simenv`) | `/tmp/tenon-apu-v1-<pci node>.lock` |
| `TENON_APU_G2_SDK_ROOT` | GSI Gemini-II SDK root | `/opt/gsi` |
| `TENON_APU_G2_GSI_TOOL` | `gsi_tool` executable | `gsi_tool` on `PATH` |
| `TENON_APU_G2_DEVICE` | Gemini-II card probed with `gsi_tool info` | `apu-00` |

Other run-time knobs read by the runtimes: `TENON_APU_G2_KEEP_TMP=1` keeps
Gemini-II build directories, `TENON_APU_G2_BUILD_JOBS` sets their build
parallelism.

Limits of the overrides:

- The Gemini-II project templates (`allo/pim/templates/apu_g2/CMakeLists.txt`,
  `device/CMakeLists.txt`) hard-code `/opt/gsi`, and the host programs open
  `DeviceConfig{0, 0}`. `TENON_APU_G2_SDK_ROOT` and `TENON_APU_G2_DEVICE`
  values other than the defaults are therefore rejected with an explicit
  `RuntimeError` instead of checking one installation and building or running
  against another. Parameterizing the templates is open work.
- The defaults that point into `~/shared/...` describe the authors' lab
  machines. On any other machine set the variable.

### UPMEM: pinned-campaign build vs your own build

The UPMEM backend has two routes.

- **Pinned campaign** (default; reproduces the paper's UPMEM cycles). Uses
  the archived uPIMulator binary and source snapshot from
  `$TENON_UPMEM_PROVENANCE_DIR`, stages the snapshot under
  `$TENON_UPMEM_SCRATCH`, and requires the DPU SDK object manifest under
  `<root>/sdk/build` to equal `UPMEM_SDK_BUILD_MANIFEST_SHA256`
  (`d45e285c34b5...`). Another SDK build changes the cycles (VA 114567
  instead of 114534), so a mismatch is refused.
- **User build.** Set both `UPIMULATOR_ROOT` (your Tenon-patched uPIMulator
  project root, containing the `benchmark/GENERIC` slot) and `UPIMULATOR_BIN`
  (its binary). Nothing is staged and the manifest is not compared. This is
  the route for ordinary use. The Tenon uPIMulator patch (GENERIC slot,
  fixture loader, linker and SDK build fixes) is
  `simulators/uPIMulator/tenon-upimulator.patch` in this repository;
  section 3a has the build. A build made that way with the SDK 2023.1.0
  mirror produced the pinned object manifest and all 16 paper UPMEM cycle
  counts on 2026-11-02.

`allo.spmw_simenv.upmem_sdk_build_identity(manifest)` returns
`"pinned-campaign"` or `"user-build:<first 12 hex digits of the manifest>"`
(and raises on a drifted manifest on the pinned route). Each DPU record in
`RunResult.extra["upmem"]` carries it as `sdk_build`, next to the raw
manifest in `sdk_build_sha256` (for example `"user-build:d45e285c34b5"` on
the verified user build).

### 3a. Building the simulators (verified 2026-11-02)

The Tenon-side recipes ship in this repository under `simulators/` (one
README per simulator, `simulators/README.md` for the overview); `$ALLO` is
the root of your checkout of `zzzDavid/allo`, branch `tenon`. Each recipe was
run from a fresh upstream clone in a scratch directory with a new Docker tag,
from the copies in `simulators/`, then exercised through `allo.compile`; full
commands, hashes, times and licensing are in `simulators.md`. All three
build from this repository alone (UPMEM also needs the DPU SDK you obtain
yourself); the Samsung simulator is bundled, sources included, and builds
from the installed package without a checkout.

```bash
# SK hynix AiM
git clone https://github.com/arkhadem/aim_simulator.git
git -C aim_simulator checkout 0f28a07bdb83e42b9305ad3d45410ebd3aa2c091
cp $ALLO/simulators/aim_simulator/Dockerfile aim_simulator/
docker build -t tenon-aim-simulator-build:0f28a07 aim_simulator          # ~30 s
docker run --rm -u $(id -u):$(id -g) -v $PWD/aim_simulator:/work tenon-aim-simulator-build:0f28a07 \
    bash -c "mkdir -p build && cd build && cmake .. && make -j8"        # ~2 min, downloads deps
export AIM_SIMULATOR_ROOT=$PWD/aim_simulator TENON_AIM_SIM_IMAGE=tenon-aim-simulator-build:0f28a07
#   test_gemv_all_targets.py AiM GEMV (256x256 BF16): 222 cycles

# UPMEM (the DPU SDK 2023.1.0 is UPMEM proprietary: obtain it yourself;
# <upmem-sdk> holds upmem-2023.1.0-Linux-x86_64/)
git clone https://github.com/VIA-Research/uPIMulator.git
git -C uPIMulator checkout 870d916334e9ff0b190f555f951a9ec3c4257781
bash $ALLO/simulators/uPIMulator/apply.sh uPIMulator    # patch + go build (go >= 1.21), ~7 s
export UPIMULATOR_ROOT=$PWD/uPIMulator/golang/uPIMulator UPIMULATOR_BIN=$PWD/uPIMulator/golang/uPIMulator/build/uPIMulator
tar czf $UPIMULATOR_ROOT/docker/upmem-sdk.tar.gz -C <upmem-sdk> upmem-2023.1.0-Linux-x86_64
docker build -f $ALLO/simulators/uPIMulator/Dockerfile \
    -t tenon-upimulator-sdk:2023.1.0 $UPIMULATOR_ROOT/docker/            # ~36 s
export TENON_UPMEM_SDK_IMAGE=tenon-upimulator-sdk:2023.1.0
# or, without Docker: the SDK unpacked on the host (needs cmake, ninja, unshare, user namespaces)
# export TENON_UPMEM_SDK_HOME=<upmem-sdk>/upmem-2023.1.0-Linux-x86_64
docker run --rm -v $UPIMULATOR_ROOT:/root/uPIMulator $TENON_UPMEM_SDK_IMAGE \
    python3 /root/uPIMulator/sdk/build.py --num_tasklets 12
#   paper GEMV (96x128): 359,026 cycles, sdk_build "user-build:d45e285c34b5"

# Samsung HBM-PIM: bundled (sources, patches and conductor ship with Tenon)
python -m pip install scons                              # or the distribution's scons; g++ and patch needed too
python -m allo.pim.samsung_simulator build               # ~15 s on 16 cores
#   copies the bundled upstream sources to $TENON_HOME/simulators/PIMSimulator
#   (default ~/.local/share/tenon/simulators/PIMSimulator), applies the five
#   patches, installs the Tenon conductor, runs scons, prints PIM_GEOMETRY;
#   pimsim_root() finds that root without any variable. Compiler temporaries
#   follow TMPDIR. `--root DIR` builds elsewhere (then export PIMSIMULATOR_ROOT=DIR).
python -m allo.pim.samsung_simulator locate              # the recipe, the default root, what pimsim_root() finds
#   paper GEMV 4096x256: vendor layout 4,435 cycles, cost-selected default 4,389
# or into your own upstream clone:
#   git clone https://github.com/SAITPublic/PIMSimulator.git
#   git -C PIMSimulator checkout 3703d1f19c8f027360cc33a3243eb271e3bb6898
#   bash $ALLO/simulators/PIMSimulator/apply.sh PIMSimulator && export PIMSIMULATOR_ROOT=$PWD/PIMSimulator
```

**Samsung terms.** The bundled PIMSimulator sources
(`simulators/PIMSimulator/upstream/`, pristine at `3703d1f`, hashed in
`UPSTREAM_MANIFEST.sha256`) and the five Tenon patches to them
(`simulators/PIMSimulator/patches/`) carry Samsung's notice: use is
restricted to non-commercial, personal or academic research, and copying or
distribution requires Samsung's express written permission. The Tenon
maintainers decided on 2026-11-06 to distribute the simulator with the tool;
no written permission is recorded in this repository. Read
`simulators/PIMSimulator/NOTICE.md` (also `THIRD_PARTY_NOTICES.md`) before
redistributing Tenon; the release ledger tracks this as P-04. The Samsung
compile route and `backend="virtual"` need no simulator.

What you must obtain yourself: the UPMEM DPU SDK (proprietary; the official
host returned HTTP 403, the CMU-SAFARI mirror carries no license file), and,
for the GSI backends, the GSI SDKs and boards. Never publish an image or
tarball containing the UPMEM SDK, and do not reuse the tag
`bongjoonhyun/upimulator` (the upstream author's Docker Hub name; P-17).

Historical note: before 2026-11-02 these recipes lived only in the
containing Tenon research repository under
`<repo>/experiments/simulators/setup/`; `simulators/` holds the copies, and
since 2026-11-10 the Samsung patches and the pinned upstream sources as well.

## 4. Running the tests

From a git checkout, with the bindings built:

```bash
cd <checkout>
python -m pip install -r requirements-dev.txt   # pytest and the dev tools
export PYTHONPATH=$PWD:$PYTHONPATH
python -m pytest tests/pim tests/perf -q -p no:cacheprovider
```

- Rows that need a simulator, Docker image, board or artifact skip with
  `BLOCKED-SIM(<reason>)`, `BLOCKED-DEVICE(<reason>)` or
  `BLOCKED-ENV(<reason>)`; the reason is the `spmw_simenv` probe text.
- `-m paper_full` selects the long paper-golden rows in
  `tests/pim/test_paper_golden.py` (skipped otherwise). These need
  `$TENON_ARTIFACTS` and, for UPMEM, the pinned-campaign route.
- `apu_v1_device` marks tests that run on a Gemini-I board. They take the
  same board lock as production launches (`TENON_APU_V1_LOCK`) and skip when
  `apu_v1_unavailable_reason()` is not None. Deselect them with
  `-m "not apu_v1_device"`.
- Packaging and discovery contract: `tests/perf/test_packaging.py` (builds an
  sdist into a temp dir; skips if no build backend is available),
  `tests/perf/test_simulator_unavailable.py`,
  `tests/pim/test_import_surface.py`.
- One GEMV per backend through the public API:
  `tests/pim/test_gemv_all_targets.py`.

## 5. Inspecting what the compiler chose

```python
import allo
from allo.pim import build_samsung_target, samsung_cost

compiled = allo.compile(workload, build_samsung_target(), samsung_cost)  # or backend="virtual"
ranking = compiled.placement_ranking            # PlacementRanking: .kernels, .incomplete
k = ranking.kernels[0]                          # KernelRanking of the first @allo.work kernel
k.scores, k.chosen_position, k.cost_ties        # (position, cycles, mode) in rank order; tie set at the winner's cycles
k.rejections, k.movement_rejections, k.tie_break # cost-rule/codegen/legality rejections; unrealizable movement ties; TIE_BREAK_RULE
k.policy_pruned                                 # (mode, reason) pruned by the reduction-order policy
k.movement                                      # KernelMovement: status, termination, optimality_proven, min_cost, program_cost, chosen_label, ties
compiled.reduction_order, compiled.reduction_policy_pruned
compiled.compiled.cmds            # the emitted target command stream
compiled.compiled.layout          # the chosen Placement (placements, extra knob values, layout)
compiled.compiled.layout_ctx      # backend state (Samsung .programs, UPMEM .segments, AiM .host_steps)
compiled.execution_graph          # retained cost graph of the chosen candidate
compiled.estimate()               # Estimate(cycles=..., spans=...) from that graph, no backend
result = compiled(*arrays)        # RunResult
result.kind                       # "device" | "simulator" | "simulator-timing-only" | "functional" | "virtual"
result.cycles, result.backend, result.extra   # extra always carries reduction_order and reduction_policy
```

The per-backend `RunResult.extra` keys and the fields of the APU v1 and
Gemini-II compiled objects are listed in `SPMW_ARCHITECTURE.md` section 5.

`RunResult.kind` says what the number is evidence of: `device` is a hardware
run with outputs read back; `simulator` a functional simulator run with
outputs; `simulator-timing-only` a timing simulator with no outputs (AiM);
`functional` a host execution of the lowered program (cycles are not a
measurement); `virtual` an analytical evaluation, nothing executed. Only
`device`, `simulator` and `functional` results write the caller's output
arrays. `allo.compile(..., layout=...)` forces a placement instead of
ranking. The typed APU routes (`APUv1Program`, `APUG2TypedProgram`,
`APUG2ComposedContractionProgram`) return their own compiled objects; see
`CAPABILITIES.md`.

Failure behaviour, timeouts and temporary files:

| Backend | Timeouts | Temporary files and locks |
|---|---|---|
| Samsung | `pim_driver` 600 s per call | per-run temp directory |
| AiM | `docker run ... ramulator2` 600 s per segment | trace in a private `tenon-aim-*` temp directory mounted read-only at `/trace`; the simulator checkout is not written |
| UPMEM | SDK build 600 s, simulation 1800 s per segment | `$TENON_UPMEM_SCRATCH/root-<sha12>/` (staged snapshot, reused; Docker writes root-owned files there), `runs/<key>/`, `root.lock` (staging), `run.lock` next to the root (serializes the shared `benchmark/<slot>`) |
| APU v1 vector route | build 600 s, execution 900 s | `tenon-apu-*` temp directories; board `flock` on `$TENON_APU_V1_LOCK` (default `/tmp/tenon-apu-v1-<pci node>.lock`) around every launch (V1-07 repair, in progress) |
| APU v1 device-log reads (`ledag-ssh`, every Gemini-I launch) | 30 s per read | the vendor tool is a PyInstaller bundle: it extracts itself into `$TMPDIR` (else `/tmp`, about 20 MB) at every start, and writes its session log and an `online_log_<time>` copy of the device log (about 5 MB) into `/tmp` regardless of `TMPDIR`; it has no option to move them (`--help` lists only `--host`). Set `TMPDIR` to a directory with space for the extraction; the runtime removes the two `/tmp` files it created after each read, but each read still needs about 5 MB free in `/tmp`, else the tool exits 255 and the launch is recorded as lost (round 85) |
| APU v2 | configure 120 s, builds 600 s, execution 120-180 s, `gsi_tool info` 15 s | `tenon-apu-g2-*` temp directories (kept with `TENON_APU_G2_KEEP_TMP=1`); card ownership checked with `gsi_tool`, no `flock` yet (G2-06) |

## 6. Fresh-install transcript (this host, 2026-11-02)

Revision: `tenon` at `ffb4a78` plus the uncommitted packaging changes in
this round. Interpreter: `/work/zhang-capra/users/nz264/miniforge3/envs/spmw`
(CPython 3.12.13, setuptools 83.0.0, cmake 4.4.0 from the environment; no
`build` module). LLVM: `/work/shared/common/llvm-project-main/build-rhel8`
(LLVM 22.0git; the `externals/llvm-project` submodule is not checked out on
this host). All artifacts under `/tmp/tenon-release/pkg2/`.

```text
$ python setup.py egg_info --egg-base /tmp/tenon-release/pkg2 sdist --dist-dir /tmp/tenon-release/pkg2/dist
  -> allo-0.5.tar.gz (~1.0 MB): all tracked allo/ and mlir/ files present (39 Gemini-II template
     files, 4 ir/template .mlir, outline_loops.mlir, harness, AIE kernels/modules); no allo/_mlir,
     tests/, examples/, benchmarks/

$ tar xzf allo-0.5.tar.gz -C src
$ LLVM_BUILD_DIR=.../build-rhel8 CC=$LLVM_BUILD_DIR/bin/clang CXX=$LLVM_BUILD_DIR/bin/clang++ \
  LDFLAGS="-L$ENV/lib -Wl,-rpath,$ENV/lib -Wl,-rpath-link,$ENV/lib" NUM_THREADS=48 \
  python -m pip wheel --no-deps --no-build-isolation -w wheel src/allo-0.5
  -> 333 ninja steps, about 3 minutes; allo-0.5-cp312-cp312-linux_x86_64.whl (87 MB) with
     allo/_mlir/_mlir_libs/*.cpython-312-*.so and allo/_mlir/libAlloMLIRAggregateCAPI.so.22.0git
     (RUNPATH: $ENV/lib:$ORIGIN:$ORIGIN/../)
  [first attempts failed: without LDFLAGS the CAPI link misses std::__glibcxx_assert_fail;
   without -DPython_EXECUTABLE nanobind built cpython-313 extensions. Both handled above / in setup.py.]

$ python -m venv --system-site-packages venv      # numpy etc. from the env
$ venv/bin/python -m pip install --no-deps --no-index wheel/allo-0.5-cp312-cp312-linux_x86_64.whl
$ cd /tmp && env -u PYTHONPATH venv/bin/python fresh_install_check.py
allo from /tmp/tenon-release/pkg2/venv/lib/python3.12/site-packages/allo/__init__.py
compile 13.4 s -> CompiledCallable          # Samsung GEMV 4096x256 fp16, mapping [16, 8], backend="virtual"
estimate 226
placement_ranking.kernels 1 bucket_count 128
cmds 6 [FILL GRF_A<-EVEN_BANK, MAC GRF_B<-GRF_A*EVEN_BANK auto, ...]
RunResult kind virtual backend virtual cycles 226
apu_g2 template shipped True                # importlib.resources
ir template shipped True
simulators_dir None                         # installed copy: no checkout-relative default
samsung reason PIMSimulator location is not configured (set PIMSIMULATOR_ROOT or TENON_SIMULATORS_DIR)
modules loaded from the checkout: []

$ mv src/allo-0.5/mlir/build{,.hidden}; venv/bin/python -c "import allo, allo._mlir.ir"   # ok: relocatable

$ TENON_SIMULATORS_DIR=<repo>/experiments/simulators venv/bin/python fresh_sim_check.py
pimsim_root <repo>/experiments/simulators/PIMSimulator
RunResult kind simulator cycles 4435 max abs err 0.00045    # PIMSimulator, random fp16 GEMV

# Pure-Python variant (sdist tree on PYTHONPATH, bindings from the live build via a symlink
# allo/_mlir -> <checkout>/mlir/build/tools/allo/_mlir): same output as above (estimate 226).

# Bindings missing:
$ venv/bin/python -c "import allo"   # with site-packages/allo/_mlir removed
ModuleNotFoundError: No module named 'allo._mlir'   (from allo/ir/types.py:9)
```

Caveat: the wheel's RUNPATH also carries the absolute library directory given
in `LDFLAGS` (here the conda environment that supplies libstdc++), so it is
a per-host artifact, not a portable `manylinux` wheel.

Round 8 (2026-11-02, commit `9cd8fb3`, wheel sha256 `788e5e13...2e69`): the
same build was installed into a **clean** venv (`python -m venv venv2`,
`pip install -r requirements.txt psutil`, then the wheel with
`--no-deps --force-reinstall`) and exercised from `/tmp` with `PYTHONPATH`
unset; `allo` imported from the venv only ("modules loaded from the
checkout: []"). Results are in `VALIDATION.md` round 8 (virtual, Samsung
simulator, AiM timing-only, UPMEM simulator, both reviewer probes rejected).
Two pitfalls (ISSUES P-22): a venv made with `--system-site-packages` over
an environment that has an editable `allo` install silently keeps the
editable copy (pip reports the wheel's requirement satisfied and the
editable finder shadows site-packages), and the wheel's metadata lists the
development requirements (`past@<github url>`, `pylint`, `pytest`), so
install the wheel with `--no-deps` after `requirements.txt`.
