# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench-on-PIM suite: target constructors, re-exported (Phase-0 Answer 1).

This module RE-EXPORTS the four per-target constructors from
`tests/spmw/_fixtures.py`. No constructor body is duplicated or edited -- the
suite reuses the validated target trees verbatim (spec Answer 6: re-export-only,
zero behavior change to `_fixtures.py`).

`_fixtures.py` is a top-level test module (pytest prepends `tests/spmw/` to
`sys.path` for the spmw suite); it is not importable as a package path, so we
add `tests/spmw/` to `sys.path` here rather than touch `tests/` packaging. This
is additive and changes no existing import behavior.
"""

from __future__ import annotations

import sys
import pathlib

# tests/pim/lib/targets.py -> parents[1] == tests/pim, parents[2] == tests
_SPMW_DIR = pathlib.Path(__file__).resolve().parents[2] / "spmw"
if str(_SPMW_DIR) not in sys.path:
    sys.path.insert(0, str(_SPMW_DIR))

from _fixtures import (  # noqa: E402  -- re-export ONLY, no behavior change
    build_samsung_target,
    build_aim_target,
    build_upmem_target,
    build_apu_v1_target,
)

# The four suite backends keyed by target.name (matches each constructor's
# `@allo.target("<name>")` so a kernel folder names a backend, never a tree).
TARGETS = {
    "samsung_hbm_pim": build_samsung_target,
    "aim": build_aim_target,
    "upmem": build_upmem_target,
    "apu_v1": build_apu_v1_target,
}


def build_target(name: str):
    """Construct the target tree for a suite backend by name."""
    if name not in TARGETS:
        raise KeyError(
            f"unknown suite target {name!r}; known: {sorted(TARGETS)}"
        )
    return TARGETS[name]()
