# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench-on-PIM suite: target constructors, re-exported (Phase-0 Answer 1).

This module RE-EXPORTS the four per-target constructors from the
:mod:`allo.pim.targets` device library. No constructor body is duplicated or
edited -- the suite reuses the validated target trees verbatim (spec Answer 6:
re-export-only). The trees now live in the package (not the test folder), so
this module imports them directly with no ``sys.path`` manipulation.
"""

from __future__ import annotations

from allo.pim.targets import (
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
