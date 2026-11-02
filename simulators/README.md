<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Simulator recipes for the Tenon PIM backends

Tenon bundles no simulator, vendor SDK or Docker image. This directory holds
the Tenon-side build recipes that turn a pinned upstream checkout into the
simulator a backend runs: every file here is either Allo-authored or a patch
that is redistributable under the upstream license. Full verification records
(commands, times, hashes, image ids) are in `docs/release/simulators.md`;
discovery variables are in `docs/release/INSTALL.md` section 3.

| Directory | Upstream (pinned commit) | Contents | Builds from this repository alone? |
|---|---|---|---|
| `aim_simulator/` | arkhadem/aim_simulator `0f28a07` (MIT) | `Dockerfile`, `README.md` | **Yes** |
| `uPIMulator/` | VIA-Research/uPIMulator `870d916` (MIT) | `tenon-upimulator.patch`, `apply.sh`, `Dockerfile`, `README.md` | **Yes**, plus the UPMEM DPU SDK 2023.1.0 you obtain yourself (proprietary) |
| `PIMSimulator/` | SAITPublic/PIMSimulator `3703d1f` (Samsung notice) | `Sconstruct`, `src/pim_driver.cc`, `src/tests/TenonConductor.{h,inc}`, `src/tests/TemporalReduce.inc`, `apply.sh`, `README.md` | **No.** The five patches to Samsung-licensed files are not included until Samsung authorizes publication (`PIMSimulator/README.md`) |

`ALLO` below and in each README is the root of your checkout of this
repository (`zzzDavid/allo`, branch `tenon`).
