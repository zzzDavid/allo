#!/usr/bin/env bash
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Apply the Tenon uPIMulator patch to an upstream VIA-Research/uPIMulator
# checkout and build the Go simulator.  Usage: apply.sh <path/to/uPIMulator>
# See README.md in this directory for the pinned upstream commit and the
# DPU SDK image / SDK object build that complete a UPIMULATOR_ROOT.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dest="${1:?usage: apply.sh <uPIMulator checkout>}"
dest="$(cd "$dest" && pwd)"
upstream_commit=870d916334e9ff0b190f555f951a9ec3c4257781
patch_file="$here/tenon-upimulator.patch"

if command -v git >/dev/null && git -C "$dest" rev-parse HEAD >/dev/null 2>&1; then
  head="$(git -C "$dest" rev-parse HEAD)"
  if [[ "$head" != "$upstream_commit" ]]; then
    echo "warning: $dest is at $head, the patch is pinned to $upstream_commit" >&2
  fi
fi

if patch -d "$dest" -p1 --dry-run --forward --silent < "$patch_file" >/dev/null; then
  patch -d "$dest" -p1 --forward --silent < "$patch_file"
  echo "patched: $(basename "$patch_file")"
elif patch -d "$dest" -p1 --dry-run --reverse --silent < "$patch_file" >/dev/null; then
  echo "already applied: $(basename "$patch_file")"
else
  echo "error: $(basename "$patch_file") does not apply to $dest" >&2
  exit 1
fi

project="$dest/golang/uPIMulator"
if [[ "${TENON_SKIP_BUILD:-0}" != "1" ]]; then
  (cd "$project" && python3 script/build.py)
  echo "built: $project/build/uPIMulator"
fi
echo "next: build the DPU SDK image and the SDK objects (README.md, steps 3-4)"
