#!/usr/bin/env bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Install the Tenon PIMSimulator overlay into a PIMSimulator source tree and
# build pim_driver.
#
#   Usage: apply.sh <path/to/PIMSimulator> [<patches-dir>]
#
# The tree is either the copy of the bundled upstream sources that
# `python -m allo.pim.samsung_simulator build` assembles (upstream/ next to
# this script, pinned commit below) or your own clone of
# SAITPublic/PIMSimulator at that commit.  The five patches to upstream files
# ship in patches/ next to this script (NOTICE.md states the terms they and
# the upstream sources carry); the second argument or $TENON_PIMSIM_PATCHES
# selects another set.
#
# Environment: TENON_SKIP_BUILD=1 installs the sources without building;
# TENON_SCONS names the SCons executable (default: `scons` on PATH);
# TENON_BUILD_JOBS sets the parallelism (default: nproc).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dest="${1:?usage: apply.sh <PIMSimulator tree> [<patches-dir>]}"
upstream_commit=3703d1f19c8f027360cc33a3243eb271e3bb6898
required_patches=(PIMKernel.h.patch PIMKernel.cpp.patch PIMCmdGen.h.patch
                  PIMCmdGen.cpp.patch Burst.h.patch)

patches="${2:-${TENON_PIMSIM_PATCHES:-$here/patches}}"
missing=()
for p in "${required_patches[@]}"; do
  [[ -f "$patches/$p" ]] || missing+=("$p")
done
if (( ${#missing[@]} )); then
  cat >&2 <<MSG
error: the PIMSimulator patches are not available
  (looked in: $patches; missing: ${missing[*]}).
pim_driver needs five patches to upstream files (src/tests/PIMKernel.h,
src/tests/PIMKernel.cpp, src/tests/PIMCmdGen.h, src/tests/PIMCmdGen.cpp,
src/Burst.h). They ship in the patches/ directory next to apply.sh
(simulators/PIMSimulator/patches/ of the Tenon repository, allo/pim/simulators/
PIMSimulator/patches/ of an installed wheel); a copy without them is
incomplete. To use another set, rerun with
  apply.sh <PIMSimulator tree> <patches-dir>
or set TENON_PIMSIM_PATCHES=<patches-dir>.
MSG
  exit 2
fi
patches="$(cd "$patches" && pwd)"
[[ -d "$dest/src/tests" ]] || {
  echo "error: $dest is not a PIMSimulator source tree (no src/tests/)" >&2
  exit 2
}
dest="$(cd "$dest" && pwd)"

if [[ -d "$dest/.git" ]] && command -v git >/dev/null; then
  head="$(git -C "$dest" rev-parse HEAD 2>/dev/null || true)"
  if [[ -n "$head" && "$head" != "$upstream_commit" ]]; then
    echo "warning: $dest is at $head, the overlay is pinned to $upstream_commit" >&2
  fi
fi

for name in "${required_patches[@]}"; do
  p="$patches/$name"
  if patch -d "$dest" -p1 --dry-run --forward --silent < "$p" >/dev/null; then
    patch -d "$dest" -p1 --forward --silent < "$p"
    echo "patched: $name"
  elif patch -d "$dest" -p1 --dry-run --reverse --silent < "$p" >/dev/null; then
    echo "already applied: $name"
  else
    echo "error: $name does not apply to $dest" >&2
    exit 1
  fi
done

install -m 0644 "$here/Sconstruct" "$dest/Sconstruct"
install -m 0644 "$here/src/pim_driver.cc" "$dest/src/pim_driver.cc"
for f in TenonConductor.h TenonConductor.inc TemporalReduce.inc; do
  install -m 0644 "$here/src/tests/$f" "$dest/src/tests/$f"
done

if [[ "${TENON_SKIP_BUILD:-0}" != "1" ]]; then
  scons_cmd="${TENON_SCONS:-scons}"
  if ! command -v "${scons_cmd%% *}" >/dev/null; then
    echo "error: SCons is not available ('$scons_cmd'); install it (python -m pip install scons, or the distribution package) or set TENON_SCONS" >&2
    exit 3
  fi
  jobs="${TENON_BUILD_JOBS:-$(nproc)}"
  # libdramsim2.a: linked by the layout-swizzle trace probe
  # (scripts/run_pimsim_layout_trace.py).
  (cd "$dest" && $scons_cmd pim_driver libdramsim/libdramsim2.a -j"$jobs")
  (cd "$dest" && ./pim_driver --describe)
fi
