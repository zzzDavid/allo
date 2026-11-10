<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Redistribution notice for the bundled PIMSimulator sources

This directory redistributes the parts of Samsung's PIMSimulator that Tenon's
Samsung HBM-PIM simulator route builds, together with Tenon's changes to them:

| Content | Origin | Terms |
|---|---|---|
| `upstream/` | pristine copy of <https://github.com/SAITPublic/PIMSimulator> at commit `3703d1f19c8f027360cc33a3243eb271e3bb6898` (branch `dev`), restricted to what `pim_driver` builds and reads: `src/`, `lib/`, `tools/`, `ini/`, the `system_hbm*.ini` files, upstream's `Sconstruct`, `README.md` and both license files. Upstream's `data/` and `dump/` benchmark inputs (about 25 MiB, unused by `pim_driver`) and its `.gitignore` are not included. `UPSTREAM_MANIFEST.sha256` lists the SHA-256 of every file; `python -m allo.pim.samsung_simulator verify` checks a copy against it | Samsung's notice (`upstream/LICENSE-PIMSimulator`) for Samsung's files; the DRAMSim2 parts under `upstream/LICENSE-DRAMSIM2` (BSD 3-Clause, University of Maryland) |
| `patches/` | Tenon's five patches to upstream files (`src/tests/PIMKernel.h`, `src/tests/PIMKernel.cpp`, `src/tests/PIMCmdGen.h`, `src/tests/PIMCmdGen.cpp`, `src/Burst.h`); each carries context lines of the Samsung file it changes | the same notice as the files they modify |
| `Sconstruct`, `src/pim_driver.cc`, `src/tests/TenonConductor.{h,inc}`, `src/tests/TemporalReduce.inc`, `apply.sh` | written for Tenon; `Sconstruct` keeps the structure of upstream's build script | Apache-2.0 (Allo), except that `Sconstruct` is also under Samsung's notice as a derivative of upstream's |

Samsung's notice, reproduced verbatim (also installed with the wheel as
`THIRD_PARTY_LICENSES/PIMSIMULATOR-LICENSE.txt`):

```
Copyright (C) 2021 Samsung Electronics Co. LTD

This software is a property of Samsung Electronics.
No part of this software, either material or conceptual may be copied or distributed,
transmitted, transcribed, stored in a retrieval system, or translated into any human
or computer language in any form by any means,electronic, mechanical, manual or otherwise,
or disclosed to third parties without the express written permission of Samsung Electronics.
(Use of the Software is restricted to non-commercial, personal or academic, research purpose
only)
```

## What this means

- **Use** of the bundled sources and of the `pim_driver` built from them is
  restricted to non-commercial, personal or academic research, as the notice
  says. Tenon's Apache-2.0 license does not extend to these files.
- **Distribution.** The notice requires Samsung's express written permission
  for copying or distribution. The Tenon maintainers decided on 2026-11-06 to
  distribute the simulator with the tool so that the Samsung route installs
  from the release artifacts alone; no written permission from Samsung is
  recorded in this repository. Anyone who redistributes Tenon's sdist or wheel
  redistributes these files under the same notice and should consider that
  requirement. The release ledger records this as packaging item P-04
  (`docs/release/ISSUES.md`).
- **Provenance.** Upstream's copy of the notice is at
  `upstream/LICENSE-PIMSimulator`; the upstream repository is public on GitHub
  under the same terms. Every file under `upstream/` is byte-identical to the
  pinned commit (`UPSTREAM_MANIFEST.sha256`).

## How the bundle is used

`python -m allo.pim.samsung_simulator build` copies `upstream/` into a build
root (default `$TENON_HOME/simulators/PIMSimulator`, see
`docs/release/INSTALL.md` section 3), applies `patches/`, installs the Tenon
files and builds `pim_driver` with SCons; `allo.spmw_simenv.pimsim_root()`
then discovers that root without any environment variable. `apply.sh` does
the same for a checkout of upstream you cloned yourself (`README.md`).
