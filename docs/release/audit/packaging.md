<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Release audit: packaging, installation, documentation, licensing, environment coupling

Gate: TASK.md G7 (plus the G8 capability-matrix requirement), RELEASE_AUDIT.md A11/A12.
Scope: `experiments/allo` at `tenon@8160d6d` (4 commits ahead of `origin/main` merge-base `ac21bae`). Read-only audit; the only artifacts produced are under `/tmp/tenon-release/` and this file.
Date: 2026-11-01. Auditor environment: `/work/zhang-capra/users/nz264/miniforge3/envs/spmw/bin/python` (CPython 3.12.13, setuptools 83.0.0, wheel 0.47.0, no `build` module).

## 1. Package build and fresh-install results

### 1.1 Metadata (setup.py / pyproject.toml)

| Field | Value | Note |
|---|---|---|
| name / version | `allo` / `0.5` | Still upstream Allo identity (`setup.py:109-111`). No Tenon name, no version bump, no PIM keywords/classifiers. |
| description / url | "Allo" / `github.com/cornell-zhang/allo` | Points at upstream, not the fork. |
| long_description | `README.md` | Upstream README; zero mention of Tenon or PIM. |
| packages | `find_packages()` (`setup.py:115`) | Also picks up top-level `examples` (94 modules) and `benchmarks` (8 modules) because both carry `__init__.py`; installing the wheel pollutes site-packages with generic `examples` and `benchmarks` top-level packages. |
| package_data / MANIFEST.in | none | No non-`.py` asset is shipped (see 1.3). |
| ext_modules | `CMakeExtension("mlir")` | Custom `build_ext` runs cmake+ninja but never copies outputs into `build_lib`; the wheel can never contain the compiled `_mlir` extension regardless of build success. |
| install_requires | `requirements.txt` | Includes dev-only pins (`black==24.8.0`, `pylint==3.0.2`, `clang-format`, `pytest`) and a `gurobipy` dependency; `past` is a direct-URL cp312 linux wheel (not PyPI-publishable). |
| build-system requires | `cmake>=3.20, ninja, nanobind` | OK, but system `cmake` on this host is 3.16.3; `setup.py` uses whatever `cmake` is on PATH. |
| python_requires | `>=3.12` | OK. |

### 1.2 Commands run

```
cd experiments/allo
PY=/work/zhang-capra/users/nz264/miniforge3/envs/spmw/bin/python
$PY setup.py --quiet sdist --dist-dir /tmp/tenon-release/pkg                      # OK, 4 s -> allo-0.5.tar.gz (874 KB, 329 entries)
$PY setup.py --quiet build --build-base /tmp/tenon-release/build bdist_wheel ...   # FAIL in build_ext:
#   CMake Error at CMakeLists.txt:4: CMake 3.20.0 or higher is required. You are running version 3.16.3
#   (also: LLVM_BUILD_DIR=$PWD/externals/llvm-project/build does not exist on this host; the live
#    mlir/build/CMakeCache.txt points MLIR_DIR at /work/shared/common/llvm-project-main/build-rhel8)
# Pure-Python wheel (what build_py actually collects), without touching the user's live mlir/build:
printf "[build]\nbuild_base = /tmp/tenon-release/build\n" > /tmp/tenon-release/extra.cfg
DIST_EXTRA_CONFIG=/tmp/tenon-release/extra.cfg $PY setup.py --quiet build_py
DIST_EXTRA_CONFIG=/tmp/tenon-release/extra.cfg $PY setup.py --quiet bdist_wheel --skip-build --dist-dir /tmp/tenon-release/pkg
#   -> allo-0.5-cp312-cp312-linux_x86_64.whl (935 KB, 247 files)
```

### 1.3 Payload inspection

| Asset read at runtime | Location | In sdist? | In wheel? | Reader |
|---|---|---|---|---|
| `allo/pim/templates/apu_g2/**` (46 files: `CMakeLists.txt`, `*.h`, 17 device `*.cc`, 16 host `*.cc`) | inside package, dir has no `__init__.py` | no (0) | no (0) | `allo/pim/apu_g2_runtime.py:32` (`_TEMPLATE = Path(__file__)...templates/apu_g2`), `apu_g2_composed_runtime.py:320-322` |
| `allo/autoscheduler/outline_loops.mlir` | inside package | no | no | `allo/autoscheduler/passes.py:504` |
| `allo/ir/template/*.mlir` (4) | inside package, no `__init__.py` | no | no | `allo/passes.py:495` |
| `allo/harness/{aie,catapult,pynq,tapa,vitis,vivado,xlscc}/*` (upstream FPGA harness files) | inside package, no `__init__.py` | no | no | `allo/backend/hls.py:279,300,409`, `allo/backend/xls.py:494`, `allo/backend/aie/__init__.py:914` |
| `allo/library/aie/kernels/*.cc` | inside package, no `__init__.py` | no | no | AIE backend |
| `allo/_mlir/` compiled bindings (`_mlir_libs/*.so`, `libAlloMLIRAggregateCAPI.so`, `dialects/`, `extras/`) | **symlink** `allo/_mlir -> ../mlir/build/tools/allo/_mlir` (tracked as mode 120000) | 13 `.py` files only; no `dialects/`, no `extras/`, no `.so` | same 10 `.py` + 1 `.pyi` | `import allo` (`allo/_mlir/_mlir_libs/__init__.py:94`) |
| `mlir/` C++ sources (103 tracked files) needed by `ext_modules` | repo root | **no (0 files)** | n/a | sdist cannot rebuild its own extension |
| `allo/pim/*.py` (40), `allo/spmw_*.py` (20), `allo/perf/*.py` (4) | package | yes (40/20) | yes (40/20) | - |
| `tests/` | 31 upstream `tests/test_*.py` only | partial (no `tests/pim`, no `tests/perf`) | no | sdist includes a non-runnable subset |

Stray files in the payload: none of `np_W*.txt`, `*.lp`, `*.prj`, `*.dot`, `tmp/` leak into the sdist or wheel (they are untracked and ignored).

### 1.4 Fresh-venv install (outside the checkout)

```
$PY -m venv --system-site-packages /tmp/tenon-release/venv      # numpy etc. from spmw env; venv site-packages shadows the editable allo
/tmp/tenon-release/venv/bin/python -m pip install --no-deps --no-index /tmp/tenon-release/pkg/allo-0.5-cp312-cp312-linux_x86_64.whl
cd /tmp; unset PYTHONPATH            # documented mlir_core path does not exist on this host, so there is nothing to put on PYTHONPATH
python -c "import allo"
#   ModuleNotFoundError: No module named 'allo._mlir._mlir_libs._mlir'   (allo/_mlir/_mlir_libs/__init__.py:94)
python -c "import allo.pim"          # same error
python -c "import allo.spmw_simenv"  # same error (package __init__ imports _mlir eagerly)
```

Repair probe (manual, to find the next failure): copied the live `mlir/build/tools/allo/_mlir/` (88 files incl. `.so`, `dialects/`, `extras/`) over the installed `allo/_mlir/`. After that, from `/tmp` with `PYTHONPATH` unset:

```
import allo; from allo.pim import build_samsung_target, samsung_cost; import allo.compiler     -> ok (all three resolve to the venv copy)
allo.compile(gemv_m4096_k256 [16x8 work mapping, fp16 4096x256], build_samsung_target(), samsung_cost, backend="virtual")
   -> CompiledCallable in 14.7 s; compiled.estimate() = Estimate(cycles=226, spans=...); placement_ranking.kernels == 1
allo.spmw_simenv._DEFAULT_PIMSIM_ROOT -> /tmp/tenon-release/venv/lib/python3.12/simulators/PIMSimulator (exists: False)
samsung_unavailable_reason() -> "pim_driver not found at .../simulators/PIMSimulator/pim_driver (set PIMSIMULATOR_ROOT)"
```

Conclusion: the virtual (simulator-free) Samsung path works from an installed copy once the MLIR extension is present; the shipped artifact does not contain that extension, any template, or `_mlir/dialects`. The APU v2 and APU v1 hardware paths would fail next on the missing `templates/apu_g2` tree (`apu_g2_runtime.py:32`) and on the missing `example-gvml` template dir.

How a user is expected to obtain the MLIR bindings: the docs (`docs/source/setup/index.rst`) describe building `externals/llvm-project` and then `pip install -e .` (editable). There is no documented non-editable route; `allo/_mlir` is a tracked symlink into `mlir/build`, so the package only works when `mlir/build` exists next to it. On this host the submodule `externals/llvm-project` is not even checked out (empty; gitlink `6b09f739`), and the live `_mlir` is built against `/work/shared/common/llvm-project-main/build-rhel8` (LLVM 22.0git), which the task's documented `PYTHONPATH`/`LLVM_BUILD_DIR` do not reference.

## 2. Environment coupling table

Grep over `allo/` for absolute paths, home-directory defaults, docker image names, hostnames, user ids. `allo/spmw_simenv.py` is the declared single owner of external locations; bypasses are marked.

| file:line | Value | Class | Env override | Note |
|---|---|---|---|---|
| `allo/spmw_simenv.py:22` | `Path(__file__).resolve().parents[2] / "simulators"` | checkout-relative default that escapes the package (`experiments/simulators`) | `PIMSIMULATOR_ROOT`, `AIM_SIMULATOR_ROOT` | Resolves to `<site-packages>/../simulators` when installed; harmless but misleading error text. Acceptable with override. |
| `allo/spmw_simenv.py:25` | docker image `bongjoonhyun/upimulator` | acceptable default | `TENON_UPMEM_SDK_IMAGE` | Image is locally built (see 3), not pullable under that name with the SDK baked in. |
| `allo/spmw_simenv.py:40-42` | `UPMEM_SDK_BUILD_MANIFEST_SHA256` pinned manifest | **unacceptable hard gate** | none | `upmem_unavailable_reason():314-320` and `spmw_upmem.py:2573-2575` refuse any SDK build whose `.o` manifest differs; `UPIMULATOR_ROOT` does not bypass it. A fresh legitimate build can never pass (A11). |
| `allo/spmw_simenv.py:44-47` | `/usr/local/gsi-apu/13.7.1/ubuntu_20_04/arc_gnu_2021.09-...` | system-install default | `TENON_APU_V1_TOOLCHAIN_BASE` | acceptable |
| `allo/spmw_simenv.py:48` | `/usr/local/include` | system default | `TENON_APU_V1_GVML_INCLUDE_ROOT` | acceptable |
| `allo/spmw_simenv.py:67` | `Path.home()/"shared"/"tenon-artifacts"` | **home-directory default to an unpublished archive** | `TENON_ARTIFACTS` | Default UPMEM binary (`:116`) and source snapshot (`:75-79`) live here; normal UPMEM use depends on it (A11). |
| `allo/spmw_simenv.py:200` | `Path.home()/"shared"/"accelerator-hub"/"gsi-apu"/"example-gvml"` | home-directory default | `TENON_APU_V1_TEMPLATE_DIR` | GSI example project (vendor SDK content) must be supplied by user; default is lab-specific. |
| `allo/spmw_simenv.py:185` | `tempfile.gettempdir()/"tenon-upmem"` | acceptable | `TENON_UPMEM_SCRATCH` | shared scratch; docker writes root-owned files here (documented in docstring). |
| `allo/pim/apu_g2_runtime.py:37-40` | `/opt/gsi/include/64vl/...`, `/opt/gsi/g2/vector_core/lib/libg2_64vl.a`, `/opt/gsi/g2/vector_core/bin/g2_md_update.py`, `/opt/gsi/lib/libg2_device_core.a` | **hard-coded vendor SDK paths, bypasses spmw_simenv** | none | APU v2 hardware route cannot be relocated. |
| `allo/spmw_aim.py:1143` | docker image literal `"aim-simulator-build"` (also `spmw_simenv.py:279`) | acceptable default, but **no env override** | none | Not pullable; must be built locally from `experiments/simulators/setup/aim_simulator/Dockerfile`. |
| `allo/spmw_aim.py:1128-1131` | writes temp `.trace` into `<AIM root>/test/` | writes into the simulator checkout | - | Requires a writable simulator tree; cleaned in `finally`. |
| `allo/spmw_codegen.py:830`, `allo/spmw_apu_v2.py:121` | docker image literal `"gsi-g2-l1sim"` | acceptable default, no env override | none | Built from GSI-provided zip (see 3); not redistributable by us. |
| `allo/pim/costs/upmem.py:97` | comment `# Generated by: python /work/shared/users/phd/nz264/tmp/coder014_factors.py` | **user path in production source** (comment) | - | Generator script is not in the repo; calibration table is unreproducible from the public tree. |
| `allo/spmw_apu_v1.py:523-527`, `allo/pim/apu_v1_program.py:380-384` | `ledag-ssh -o localhost` | site tool probed via `shutil.which` | - | acceptable (optional log source) |
| `allo/harness/makefile_gen/makegen.py:406`, `allo/backend/hls.py:818` | `/opt/xilinx/xrt/setup.sh`, `$MGC_HOME/shared/include` | upstream Allo FPGA defaults | - | out of scope (upstream). |
| `tests/pim/test_paper_golden.py:156` | `/home/nz264/shared/tenon-artifacts` | evaluation-only test default | `TENON_ARTIFACTS` | skips when absent; should default via `spmw_simenv.tenon_artifacts_root()`. |
| `design/10-apu-v1-reference-backend.md:13` | `/home/nz264/shared/accelerator-hub/gsi-apu` | doc | - | user path in a shipped design note. |

Nothing in `allo/` imports from `dev/`, another worktree, `tests/`, or `paper/`; `grep` for `spmw-for-pim`, `tenon-artifacts`, `dev/`, `nz264` in `allo/*.py` hits only the `spmw_simenv.py` defaults and the `costs/upmem.py:97` comment. Production simulator discovery is otherwise routed through `spmw_simenv` except the APU v2 `/opt/gsi` constants and the two docker image literals in `spmw_aim.py`/`spmw_codegen.py`.

## 3. Simulator reproducibility table

Committed overlays live in the containing repo at `experiments/simulators/setup/` (7 tracked files). The upstream checkouts are gitignored. No commit pin for any simulator is recorded in `experiments/simulators/README.md`, `design_doc/targets/available-targets-setup.md`, or `design_doc/targets/simulator-setup-history.md`.

| Simulator | Upstream | Installed HEAD | Local state of installed checkout | Committed patches vs installed | Build recipe (documented) | Docker image | Fresh-build discoverable by `spmw_simenv`? |
|---|---|---|---|---|---|---|---|
| Samsung PIMSimulator | `github.com/SAITPublic/PIMSimulator` | `3703d1f19c8f` (2024-07-01); git metadata only recoverable via the parent repo's `.git/modules/simulators/PIMSimulator` (the checkout's `.git` file points at a nonexistent path, so `git -C PIMSimulator` fails) | 7 tracked files modified (+1004/-13): `Sconstruct`, `src/Burst.h`, `src/tests/{KernelAddrGen.h,PIMCmdGen.cpp,PIMCmdGen.h,PIMKernel.cpp,PIMKernel.h}`; untracked `src/pim_driver.cc` (721 lines), `src/SamsungSlicePlan.{cpp,h}`, `src/tests/TemporalReduce.inc`, `src/gemv_bench_main.cc`, binaries `pim_driver` (Sep 29), `gemv_bench` | `setup/PIMSimulator/src/pim_driver.cc` is 323 lines, installed is 721 (+484/-86): **committed overlay is stale**. Installed driver equals `dev/06292026-samsung-general-instr-codegen/pimsim-conductor/src/pim_driver.cc` *working copy*, which itself has 189 uncommitted lines vs its committed version. `Burst.h`, `PIMCmdGen.*`, `SamsungSlicePlan.*`, `TemporalReduce.inc` are committed **nowhere**. `Sconstruct` overlay differs (gtest link line). | `scons -j32` (targets doc); conda env `pim-dev` sourced from `/home/nz264/anaconda3` | none (native) | Yes via `PIMSIMULATOR_ROOT` pointing at a tree containing `pim_driver`; but a user cannot rebuild an equivalent `pim_driver` from committed sources. |
| SK hynix AiM (`aim_simulator`, Ramulator 2.0) | `github.com/arkhadem/aim_simulator` | `0f28a07bdb83` (2025-07-22) | 0 tracked modifications; 56 untracked (`Dockerfile`, `test/*.trace`, `build/`, `libramulator.so`) | `setup/aim_simulator/Dockerfile` identical to installed | `docker build -t aim-simulator-build .` then `cmake .. && make -j8` inside image | `aim-simulator-build:latest` (id `215d37b4`, built 2026-04-18 from Ubuntu 24.04 + g++-12; history matches committed Dockerfile) | Yes: `AIM_SIMULATOR_ROOT` + locally built image named exactly `aim-simulator-build` (name not overridable). Reproducible. |
| UPMEM uPIMulator (Go) | `github.com/VIA-Research/uPIMulator` | `870d916334e9` (2025-02-01) | 24 tracked files modified (+295/-301: benchmark `CMakeLists.txt` rewrites per `setup/patch-upmem-benchmarks.sh`, plus `GEMV/BS/TS task.c` edits), 22 untracked (new benchmark slots `TENON/`, `E5*`, `DSL*`, `CINM_FFN`, new Go assembler files `src/assembler/prim/*.go`, `docker/upmem-sdk.tar.gz`) | Dockerfile overlay identical. The `task.c` edits and the untracked Go assembler/benchmark additions are **not committed anywhere**. | `python script/build.py`; `docker build -t bongjoonhyun/upimulator ./docker/` after re-tarring the SDK | `bongjoonhyun/upimulator:latest` (id `57dc7591`, 2026-07-21; `COPY upmem-sdk.tar.gz` + tar -zxf layered on a 6-month-old base) -- a local rebuild under the upstream author's image name | **No.** Default binary/snapshot come from `~/shared/tenon-artifacts/upmem/provenance/upimulator/` (A11). With `UPIMULATOR_ROOT`/`UPIMULATOR_BIN` set, `upmem_unavailable_reason()` and `_run_upmem` still require the SDK object manifest to equal `UPMEM_SDK_BUILD_MANIFEST_SHA256`; no override exists. |
| UPMEM DPU SDK 2023.1.0 | `github.com/CMU-SAFARI/UPMEM` mirror (`upmem-sdk/`) | `84a0dcd3f57d` (2026-02-10) | clean | n/a | tarred into the docker context | inside `bongjoonhyun/upimulator` | SDK is UPMEM proprietary binaries redistributed via a third-party mirror; license file not present in the mirror checkout. |
| GSI APU v2 L1 simulator | GSI-provided `g2_gtml_v2.zip` + `example-gtml/` (not in any repo) | n/a | n/a | n/a | `docker build -f simulator/Dockerfile -t gsi-g2-l1sim .` from `~/shared/accelerator-hub/gsi-apu-g2/` | `gsi-g2-l1sim:latest` (id `74beb8db`, 2026-04-23) | Image name hard-coded; contents are vendor-proprietary. |

## 4. Documentation gaps

What exists:
- `README.md`: upstream Allo README (FPGA/AIE), zero Tenon/PIM mention.
- `SPMW_ARCHITECTURE.md` (229 lines): abstractions, module map, backend paths. **Stale**: module map lists `allo/pim/upmem_program.py`, `allo/pim/upmem_abi.py`, `allo/pim/apu_v1_hybrid_runtime.py` (none exist; 3 of 26 referenced modules missing); describes UPMEM as `allo.compile(allo.UPMEMProgram(...))` and says "the uPIMulator matcher path has been removed", whereas the tree has no `UPMEMProgram` and `allo/spmw_upmem.py` (2.5k lines) runs uPIMulator in Docker through `allo.compile(@allo.work region, build_upmem_target(), upmem_cost)` (`tests/pim/test_gemv_all_targets.py:217-254`). APU v2 / Gemini-II (`allo/pim/apu_g2_*`, `build_apu_v2_target`, `build_apu_g2_target`, `apu_g2_cost`) is absent entirely.
- `design/03..10-*.md`: design-era rationale; `design/10` cites a user home path.
- `docs/` (Sphinx): upstream only; no PIM page; `setup/index.rst` documents only the editable install.
- `AGENTS.md` (23 lines): upstream agent notes, no PIM.
- `tutorials/`, `examples/`: no PIM content (polybench examples gained a `uint16` type parameter only).
- `tests/pim/conftest.py`: registers `apu_v1_device`, `paper_full`; the suite's existence and run instructions are documented nowhere in the package.
- Backend knowledge is in the containing repo (`design_doc/`, `.claude/skills/pim-*`), not in the package. `design_doc/targets/available-targets-setup.md:36-56` bootstraps via `/home/nz264/anaconda3` and still references pre-Tenon module names (`pimdsl/runtime/aim_shadow.py`); `.claude/skills/pim-apu-v2/SKILL.md:9` and `available-targets-setup.md:106` still say APU v2 is "L1 simulator only", contradicting `design_doc/targets/apu-g2-vl64-backend.md` and the hardware runtime in `allo/pim/apu_g2_runtime.py`.

Missing for a compiler expert:
1. Install guide for the release artifact (non-editable), including how to provide the MLIR extension and the five optional backends' external dependencies with their env variables (`PIMSIMULATOR_ROOT`, `AIM_SIMULATOR_ROOT`, `UPIMULATOR_ROOT/BIN`, `TENON_UPMEM_*`, `TENON_APU_V1_*`, `TENON_ARTIFACTS`); none of these is documented outside docstrings.
2. Supported-targets / capability matrix (G8): source vs typed input, compile-only, analytical (`backend="virtual"`), simulator, hardware, numeric vs timing scope, per backend.
3. How to add a primitive (target op pattern + cost rule) and how to add a backend (target spec, cost spec, codegen runner, `spmw_simenv` probe).
4. How to inspect the chosen program: `CompiledCallable.estimate()`, `.execution_graph()`, `.placement_ranking`, retained manifests; nothing documents these.
5. Failure behaviour: `SimulatorUnavailable` reasons, timeouts (600 s Samsung/AiM/APU builds, 1800 s UPMEM simulate), temp dirs (`tenon-apu-*`, `tenon-upmem`), locks (`tenon-upmem/root.lock`, `run.lock`), docker write-ownership caveat.
6. How to run the PIM suite (`pytest tests/pim tests/perf`, 253 tests collected in 1.7 s; `-m paper_full` for 5 golden rows; 8 `apu_v1_device`-gated tests).

## 5. Licensing / redistribution table

| Item | Location | License | Redistributable under Apache-2.0? | Action |
|---|---|---|---|---|
| Allo core + Tenon Python/C++ | `experiments/allo` | Apache-2.0 (`LICENSE`, SPDX headers) | yes | keep attribution ("Copyright Allo authors"); release identity still says upstream Allo. |
| Xilinx harness files, AMD AIE CMake | `allo/harness/{tapa,vitis}/xcl2.*`, `allo/harness/aie/CMakeLists.txt`, `makefile_gen/*` | Xilinx/AMD Apache-2.0 headers | yes (upstream Allo already ships them) | none |
| APU v2 templates | `allo/pim/templates/apu_g2/**` | Allo-authored, Apache-2.0 headers; `#include <device/vector_core/gsi_libgal.h>`, `<g2_64vl_types.h>`, `<gsi_perf.h>`, `<transport_gsi_library.h>` etc. from the GSI SDK | yes: no GSI header text is copied; SDK is referenced by include path | document GSI SDK install (`/opt/gsi`) as a prerequisite; move paths to `spmw_simenv`. |
| GSI GVML (APU v1) | referenced via `/usr/local/include/gsi` and `~/shared/accelerator-hub/gsi-apu/example-gvml` | proprietary GSI SDK | not bundled (good) | document as external prerequisite. |
| GSI G2 L1 simulator image `gsi-g2-l1sim` | docker only, built from `g2_gtml_v2.zip` | proprietary | must not be published | documented as user-supplied. |
| Samsung PIMSimulator | `experiments/simulators/PIMSimulator` (ignored) | `LICENSE-PIMSimulator`: "property of Samsung Electronics ... no part may be copied or distributed ... without express written permission; non-commercial research only"; `LICENSE-DRAMSIM2` BSD | **no** | Modified Samsung files **are committed** in the containing repo: `dev/06292026-samsung-general-instr-codegen/pimsim-conductor/src/tests/{KernelAddrGen.h,PIMKernel.cpp,PIMKernel.h}` (carry the Samsung notice). Must be removed from any public tree or replaced by a patch distributed only with permission. `setup/PIMSimulator/src/pim_driver.cc` and `Sconstruct` are team-authored but build against Samsung code; ship as a patch, not as a derivative source copy. |
| SK hynix AiM simulator | `aim_simulator/` | MIT (University of Michigan, 2025) | yes with notice | add NOTICE entry if any part is vendored (currently only a Dockerfile). |
| uPIMulator | `uPIMulator/` | MIT-style (VIA Research Group, KAIST 2024) | yes with notice | local Go/benchmark edits are uncommitted; publish as a patch with the notice. |
| UPMEM DPU SDK 2023.1.0 | `upmem-sdk/` (CMU-SAFARI mirror), baked into `bongjoonhyun/upimulator` | UPMEM proprietary; no license file in mirror | **no** | document download; never ship the image or `upmem-sdk.tar.gz`. |
| CENT-derived data | `benchmarks/cent_aim/{vendor_contract,reference_traces}.py` | opcode inventories + SHA-256 of traces; `CENT_SOURCE_COMMIT = 3b0f874a...` | derived facts, but CENT repository URL/license not cited | add upstream URL + license note. |
| PolyBench workloads | `examples/polybench/*.py` | Allo re-implementations under Apache-2.0 (upstream Allo) | yes | none |
| Paper golden values | `tests/pim/test_paper_golden.py` | project data | yes | depends on `tenon-artifacts` (unpublished). |

## 6. Accidental payload list

Tracked in `experiments/allo` (`git ls-files`):
- `playground/int8_gemm.py`, `playground/mat_vec_test.py` (upstream scratch; carried over).
- `experiments/scripts/verifier_probe_residency_panic.py` -- agent-authored debugging probe ("VERIFIER-OWNED probe script"), not product code.
- `scripts/prepare_upmem_tenon_campaign.py`, `scripts/run_pimsim_layout_trace.py` -- campaign scripts; the latter is required by `tests/pim/samsung_hbm_pim/layout_swizzle/test_layout_swizzle_probe.py:49`.
- `tests/pim/.apu_v1.lock` -- empty board lock file committed.
- `allo/_mlir` -- symlink into an untracked build directory (mode 120000).
- `allo/pim/costs/upmem.py:97` -- comment with a private `/work/shared/users/phd/nz264/tmp/...` path.
- `tests/pim/test_paper_golden.py:10,156`, `design/10-apu-v1-reference-backend.md:13` -- `/home/nz264/...` paths.
- `tutorials/*.py` author emails (`hzchen@cs.cornell.edu`) and `.github/workflows/fpga_weekly.yml:28` license-server host `2100@en-license-05.coecis.cornell.edu` -- upstream Allo, pre-existing.
- `.gitignore` contains `*.txt`, which also hides legitimate text assets; no `requirements*.txt` problem today but a trap.

Untracked in the working tree (would leak into a `git add -A` or a tarball of the directory, not into `git archive`): `np_W0.txt`, `np_W1.txt`, `np_W2.txt`, `simple.lp`, `simple2.lp`, `simple-node.lp`, `simple2.dot`, `df-gemm-daisy.prj`, `top.prj`, `tmp/`, `allo/pim/calibration/` (empty dir), six `scripts/verifier_check_*.py`, `mlir/build/`.

Secrets: `git grep -i "token|password|secret|BEGIN RSA|nz264@"` finds only identifier names (`HandleToken`, `BufferToken`, DMA `token`). No credentials. Largest tracked files: `docs/source/_static/scalar-vector-product.png` (396 KB), `tutorials/allo-icon.png` (180 KB), `allo/ir/builder.py` (184 KB); nothing binary beyond images.

## 7. Upstream delta summary

`git merge-base origin/main HEAD` = `ac21bae` ("[Example] Add two Flashattention accelerator example (#615)", 2026-10-11); `origin/main` has no commits beyond the merge-base. HEAD is 4 squashed commits ahead:

```
89402b2 [IR][Backend] Fixed-width bitwise ops, integer sqrt bridge, static-length slices, NumPy scalar args, portable C backend
cc23672 [Dataflow][IR] @allo.work frontend: scalar mappings, get_wid, retained group scope and call-ABI identity
4e56984 [PIM] Tenon: SPMW target description, matcher, unified autoscheduler, executable cost specs and five PIM backends
8160d6d [Docs] Tenon architecture overview and design notes
```

219 files changed, +79,024 / -131: 174 added, 45 modified, 0 deleted. Modified upstream (shared-core) files: `allo/__init__.py` (+110, lazy PIM exports), `allo/ir/builder.py` (+494/-12), `allo/ir/infer.py` (+88/-7), `allo/dsl.py` (+38), `allo/backend/llvm.py` (26), `allo/ir/typing_rule.py` (19), `allo/dataflow.py` (8), `allo/ir/visitor.py` (4), `allo/backend/__init__.py` (2), `mlir/lib/Translation/EmitBaseHLS.cpp` (+20), `EmitIntelHLS.cpp`/`EmitTapaHLS.cpp` (6 each), `EmitVivadoHLS.cpp` (4), `mlir/include/allo/Dialect/Visitor.h` (5), `mlir/include/allo/Translation/EmitBaseHLS.h` (3), and 30 `examples/polybench/*.py` (add `uint16` to type parameters). Added: `allo/pim` (79), `allo/spmw_*.py` (20), `allo/perf` (4), `tests/pim` (21), `tests/perf` (21), `benchmarks/` (8), 5 new `tests/test_*.py`, `design/06-10`, `SPMW_ARCHITECTURE.md`, 2 scripts.

Shared-core smoke check: `pytest tests/test_builder.py tests/test_types.py tests/test_linalg.py` -> **124 passed, 2 skipped in 38 s** (no regression observed in the sampled upstream surface). `tests/pim/test_import_surface.py` asserts `import allo` does not load the PIM stack.

## 8. Findings

| ID | Severity | Where | Finding | Repair |
|---|---|---|---|---|
| P-01 | **Critical** | `setup.py:115` (`find_packages()`), no `package_data`/`MANIFEST.in`; `allo/pim/apu_g2_runtime.py:32` | Wheel and sdist ship zero non-`.py` assets: all 46 `allo/pim/templates/apu_g2/**` files, `allo/autoscheduler/outline_loops.mlir`, `allo/ir/template/*.mlir`, `allo/harness/**`, `allo/library/aie/kernels/*.cc` are absent. APU v2 routes and upstream HLS/AIE routes cannot run from an installed artifact. | Add `package_data`/`include_package_data=True` with a `MANIFEST.in` (`recursive-include allo/pim/templates *`, `*.mlir`, harness files) and a test that opens each runtime-read asset via `importlib.resources` from an installed copy. |
| P-02 | **Critical** | `allo/_mlir` symlink (mode 120000) -> `mlir/build/tools/allo/_mlir`; `setup.py:33-90` `build_extension` never copies outputs | The compiled `_mlir` extension (`_mlir_libs/*.so`, `libAlloMLIRAggregateCAPI.so`, `dialects/`, `extras/`) is not in the wheel; `import allo` fails from a fresh venv (`ModuleNotFoundError: allo._mlir._mlir_libs._mlir`). sdist also omits the `mlir/` C++ sources (0 files), so it cannot rebuild the extension either. Only the editable install works, which G7 explicitly rules insufficient. | Make `build_ext` install the CMake output into `build_lib/allo/_mlir` (or use scikit-build-core), add `dialects/`+`extras/` as package data, and include `mlir/` in the sdist. |
| P-03 | **Critical** | `allo/spmw_simenv.py:40-42,314-320`; `allo/spmw_upmem.py:2573-2575` | UPMEM availability and every UPMEM run hard-require the SDK object manifest to equal the archived campaign hash; `UPIMULATOR_ROOT` does not bypass it. Default binary/snapshot come from `~/shared/tenon-artifacts` (`:63-79,116`). A user with a fresh, correct uPIMulator+SDK build cannot use the backend (confirms A11). | Make the manifest check advisory (record `sdk_build_sha256` in the run record; warn on mismatch; fail only when `TENON_UPMEM_REQUIRE_CAMPAIGN_SDK=1` or inside `test_paper_golden`). Default `upimulator_bin()` to `UPIMULATOR_ROOT/build/uPIMulator`. |
| P-04 | **Critical** | `dev/06292026-samsung-general-instr-codegen/pimsim-conductor/src/tests/{KernelAddrGen.h,PIMKernel.cpp,PIMKernel.h}` (containing repo); `experiments/simulators/PIMSimulator/LICENSE-PIMSimulator` | Modified Samsung PIMSimulator sources carrying Samsung's "no copying or distribution without written permission, non-commercial research only" notice are committed. Redistribution under Apache-2.0 is not permitted. | Remove from any public history; distribute the Samsung changes as a `.patch` with an explicit notice, or obtain permission. Keep PIMSimulator itself user-cloned. |
| P-05 | **High** | `experiments/simulators/setup/PIMSimulator/src/pim_driver.cc` (323 lines) vs installed `PIMSimulator/src/pim_driver.cc` (721 lines); installed checkout has 7 modified upstream files (+1004/-13) and untracked `SamsungSlicePlan.{cpp,h}`, `TemporalReduce.inc`, `Burst.h`/`PIMCmdGen.*` edits committed nowhere; the installed driver equals an **uncommitted** working copy in `dev/` | The `pim_driver` binary the Samsung backend and paper numbers depend on cannot be rebuilt from any committed source. PIMSimulator's own `.git` pointer is broken (recoverable only via the parent `.git/modules`). | Commit the complete PIMSimulator patch set (as a patch series against `3703d1f`), regenerate `pim_driver`, record its sha256 and the upstream commit in `experiments/simulators/README.md`. |
| P-06 | **High** | `experiments/simulators/uPIMulator` working tree: 24 modified tracked files incl. `benchmark/{GEMV,BS,TS}/dpu/task.c`, 22 untracked (`src/assembler/prim/*.go`, `benchmark/TENON/`, `E5*`, `DSL*`) not in `setup/` | uPIMulator Go assembler and benchmark additions used by the UPMEM flow are unreproducible from the repo; only the Dockerfile and CMake-rewrite script are committed. No upstream commit pin documented (installed: `870d9163`). | Commit a patch series against `870d9163`; pin AiM `0f28a07b`, PIMSimulator `3703d1f1`, SDK mirror `84a0dcd3` in the README. |
| P-07 | **High** | `setup.py:109-113`, `README.md` | Release identity is upstream Allo (`allo 0.5`, "Allo Community", upstream URL, upstream README); the long description never mentions Tenon or PIM. Installing the wheel would shadow/conflict with upstream `allo`. | Decide the release identity (e.g. `allo-tenon` or an upstream PR); bump version; write a Tenon README section and Sphinx page. |
| P-08 | **High** | `SPMW_ARCHITECTURE.md:46-47,58,71,86` | Architecture doc references three nonexistent modules (`upmem_program.py`, `upmem_abi.py`, `apu_v1_hybrid_runtime.py`) and a nonexistent `UPMEMProgram` API, states the uPIMulator path was removed (it is the live path in `spmw_upmem.py`), and omits APU v2 entirely. Fails G7 "architecture docs match the implementation". | Regenerate the module map from the tree; describe the five routes as implemented; add sections on adding a primitive/backend and inspecting `CompiledCallable.estimate()/execution_graph()/placement_ranking`. |
| P-09 | **High** | no file: `docs/`, `README.md`, `SPMW_ARCHITECTURE.md` | No install guide for the artifact, no supported-targets/capability matrix (G8), no documented env variables for backend discovery, no documented way to run `tests/pim`. The only bootstrap text (`design_doc/targets/available-targets-setup.md:40-43`) sources `/home/nz264/anaconda3`. | Add `docs/source/backends/pim/*.rst` (install, env vars, capability matrix, failure/timeout/tempdir behaviour, extension guide). |
| P-10 | **Medium** | `allo/pim/apu_g2_runtime.py:37-40` | Four `/opt/gsi/...` SDK paths hard-coded outside `spmw_simenv`, no override. | Add `TENON_APU_G2_SDK_ROOT` (default `/opt/gsi`) in `spmw_simenv` and derive the four paths. |
| P-11 | **Medium** | `allo/spmw_aim.py:1143`, `allo/spmw_simenv.py:279`, `allo/spmw_codegen.py:830`, `allo/spmw_apu_v2.py:121` | Docker image names `aim-simulator-build` and `gsi-g2-l1sim` are literals with no env override (UPMEM has `TENON_UPMEM_SDK_IMAGE`). | Add `TENON_AIM_IMAGE`, `TENON_APU_G2_SIM_IMAGE` accessors in `spmw_simenv`. |
| P-12 | **Medium** | `setup.py:115` `find_packages()` | Top-level `examples` (94 modules) and `benchmarks` (8) are installed as generic top-level packages; `tests/` partially included in the sdist (31 upstream files, none of `tests/pim`). | `find_packages(include=["allo", "allo.*"])`; ship `benchmarks/cent_aim` under `allo.benchmarks` if it is product code. |
| P-13 | **Medium** | `requirements.txt` -> `install_requires` | Runtime dependencies include dev tooling (`black==24.8.0`, `pylint==3.0.2`, `clang-format`, `pytest`), `gurobipy` (commercial solver client), and a direct-URL `past` wheel (cp312 linux only) which PyPI rejects. | Split runtime vs dev extras; make `gurobipy`/`past` optional extras. |
| P-14 | **Medium** | `allo/pim/costs/upmem.py:97` | Calibration table generated by a private script at `/work/shared/users/phd/nz264/tmp/coder014_factors.py` not in the repo. | Commit the generator (or the evidence + a `scripts/` regenerator) and reference it relatively. |
| P-15 | **Medium** | `allo/spmw_simenv.py:22` | Default simulator root is `Path(__file__).parents[2]/"simulators"`, i.e. the containing repo's `experiments/simulators`; from an installed package this is `<site-packages>/../simulators`. Works only in-checkout. | Default to `None` with a clear "set PIMSIMULATOR_ROOT" message, or to `~/.tenon/simulators`. |
| P-16 | **Medium** | `tests/pim/test_paper_golden.py:156`; `allo/spmw_simenv.py:67,200` | Home-directory defaults to lab shares (`~/shared/tenon-artifacts`, `~/shared/accelerator-hub/...`) and a literal `/home/nz264/...` in a test. | Route the test through `spmw_simenv.tenon_artifacts_root()`; document the two shares as user-provided. |
| P-17 | **Medium** | `experiments/simulators/upmem-sdk`, `bongjoonhyun/upimulator` image | UPMEM SDK is proprietary and obtained from a third-party mirror; the locally built image reuses the upstream author's image name, so `docker pull` would fetch a different image than the one tested. | Rename the local image (e.g. `tenon-upimulator-sdk`) via `TENON_UPMEM_SDK_IMAGE` default; document the SDK download and license. |
| P-18 | **Low** | `experiments/scripts/verifier_probe_residency_panic.py`, `tests/pim/.apu_v1.lock`, `playground/` | Agent debugging probe, a committed lock file, and upstream scratch files in the tracked tree. | Remove; create the lock file at runtime. |
| P-19 | **Low** | `.claude/skills/pim-apu-v2/SKILL.md:9`, `design_doc/targets/available-targets-setup.md:106` | Still say APU v2 is L1-simulator only; contradicts `apu-g2-vl64-backend.md` and `allo/pim/apu_g2_runtime.py` hardware route (A12). | Update both. |
| P-20 | **Low** | `pyproject.toml` build-system `cmake>=3.20` vs `setup.py` calling `cmake` from PATH | Isolated PEP 517 builds get a new cmake, but `setup.py bdist_wheel` picks up system cmake 3.16.3 and fails with an unhelpful traceback; `LLVM_BUILD_DIR` documented as `externals/llvm-project/build` does not exist here (submodule not checked out; live build uses `/work/shared/common/llvm-project-main/build-rhel8`). | Use `cmake.CMAKE_BIN_DIR`/`shutil.which` fallback to the pip `cmake`; document the LLVM pin (`externals/llvm-project` gitlink `6b09f739`) and verify it matches the LLVM used to build. |
| P-21 | **Low** | `allo/spmw_aim.py:1128-1131` | Temp trace files are written inside the user's simulator checkout (`<AIM root>/test/`) because the docker mount is the root; needs a writable simulator tree. | Mount a scratch dir as a second volume. |

## 9. Open questions

1. Release identity: upstream PR into `cornell-zhang/allo` (then `allo` metadata is right and the PIM backends must pass upstream CI) versus a separately named distribution? The payload decisions in P-01/P-02/P-12 depend on this.
2. Does Samsung permit distributing a patch against PIMSimulator that embeds fragments of `PIMKernel.cpp`? If not, the Samsung backend's `pim_driver` must be re-expressed as an out-of-tree driver linking the unmodified simulator, or the required simulator changes must be upstreamed to `SAITPublic/PIMSimulator`.
3. Which uPIMulator revision and which exact SDK build produced the paper cycles? `UPMEM_SDK_BUILD_MANIFEST_SHA256` pins a build, but the Go-side edits (`src/assembler/prim/*.go`) are uncommitted; can the campaign numbers be reproduced from a clean clone + committed patches at all?
4. Should `benchmarks/cent_aim` (CENT comparison data) and `scripts/prepare_upmem_tenon_campaign.py` be part of the compiler package or of the artifact bundle?
5. The live `_mlir` extension is built against a shared LLVM 22.0git tree that is not the `externals/llvm-project` gitlink. Which LLVM commit is the supported one for the release, and does the gitlink `6b09f739` build the same `_mlir`?
6. `gurobipy` is imported only by upstream `allo/autoscheduler/dfg.py` (no PIM module uses it); should it become an optional extra so the PIM release installs without a solver license?
