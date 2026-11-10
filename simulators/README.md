<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Simulator recipes for the Tenon PIM backends

This directory holds the Tenon-side recipes that turn a pinned upstream
simulator into the simulator a backend runs. Two recipes build from a
checkout of the upstream (AiM, uPIMulator); the Samsung PIMSimulator recipe
is complete in itself: it bundles the pinned upstream sources, Tenon's
patches and conductor, and a build command, so that route installs from the
Tenon distribution alone (`PIMSimulator/NOTICE.md` states the terms the
Samsung sources carry). No vendor SDK or Docker image is bundled. Full
verification records (commands, times, hashes, image ids) are in
`docs/release/simulators.md`; discovery variables are in
`docs/release/INSTALL.md` section 3.

| Directory | Upstream (pinned commit) | Contents | Builds from this repository alone? |
|---|---|---|---|
| `aim_simulator/` | arkhadem/aim_simulator `0f28a07` (MIT) | `Dockerfile`, `README.md` | **Yes** |
| `uPIMulator/` | VIA-Research/uPIMulator `870d916` (MIT) | `tenon-upimulator.patch`, `apply.sh`, `Dockerfile`, `README.md` | **Yes**, plus the UPMEM DPU SDK 2023.1.0 you obtain yourself (proprietary) |
| `PIMSimulator/` | SAITPublic/PIMSimulator `3703d1f` (Samsung notice; DRAMSim2 parts BSD) | `upstream/` (pristine sources, hashed in `UPSTREAM_MANIFEST.sha256`), `patches/`, `Sconstruct`, `src/pim_driver.cc`, `src/tests/TenonConductor.{h,inc}`, `src/tests/TemporalReduce.inc`, `apply.sh`, `README.md`, `NOTICE.md` | **Yes**: `python -m allo.pim.samsung_simulator build` (needs g++, SCons and `patch`); the wheel carries this recipe under `allo/pim/simulators/PIMSimulator` |

`ALLO` below and in each README is the root of your checkout of this
repository (`zzzDavid/allo`, branch `tenon`).
