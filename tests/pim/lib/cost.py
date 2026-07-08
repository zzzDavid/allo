# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench-on-PIM suite: executable cost-program binding surface."""

from __future__ import annotations

from allo.pim.costs import aim_cost, apu_v1_cost, samsung_cost


def bind_cost(target):
    """Return the standalone cost source for a ported target."""
    if getattr(target, "name", target) == "samsung_hbm_pim":
        return samsung_cost
    if getattr(target, "name", target) == "aim":
        return aim_cost
    if getattr(target, "name", target) == "apu_v1":
        return apu_v1_cost
    return None
