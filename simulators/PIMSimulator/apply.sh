#!/usr/bin/env bash
# Install the Tenon PIMSimulator overlay into an upstream SAITPublic/PIMSimulator
# checkout and build pim_driver.
#
#   Usage: apply.sh <path/to/PIMSimulator> [<patches-dir>]
#
# The five patches to Samsung-licensed upstream files are NOT part of this
# repository (README.md, "Samsung limitation").  They are read from, in order:
# the second argument, $TENON_PIMSIM_PATCHES, or a sibling patches/ directory.
# See README.md in this directory for the pinned upstream commit.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dest="${1:?usage: apply.sh <PIMSimulator checkout> [<patches-dir>]}"
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
error: the Samsung-derived PIMSimulator patches are not available
  (looked in: $patches; missing: ${missing[*]}).
The Samsung HBM-PIM simulator route cannot be built from the release
repository alone. pim_driver needs five patches to Samsung-licensed upstream
files (src/tests/PIMKernel.h, src/tests/PIMKernel.cpp, src/tests/PIMCmdGen.h,
src/tests/PIMCmdGen.cpp, src/Burst.h). PIMSimulator is distributed under
Samsung's notice (no copying or distribution without Samsung's express written
permission), so these patches are not published here until Samsung authorizes
it (docs/release/ISSUES.md P-04). If you have them, rerun with
  apply.sh <PIMSimulator checkout> <patches-dir>
or set TENON_PIMSIM_PATCHES=<patches-dir>, or place them in $here/patches/.
MSG
  exit 2
fi
patches="$(cd "$patches" && pwd)"
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
  # libdramsim2.a: linked by the layout-swizzle trace probe
  # (scripts/run_pimsim_layout_trace.py).
  (cd "$dest" && scons pim_driver libdramsim/libdramsim2.a -j"$(nproc)")
  (cd "$dest" && ./pim_driver --describe)
fi
