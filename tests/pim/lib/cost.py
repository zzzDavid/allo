# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench-on-PIM suite: cost-model binding surface (Phase-0 Answer 1).

`cost.py` BINDS the existing design-04 CostModels (`spmw_cost_tables.py`,
self-registered into `spmw_cost_model._model_registry`) to a target by name. It
defines NO new cost numbers -- it is a lookup surface so a kernel test can name
the faithful CostModel the pipeline scores with. `compile_for_target` already
scores with `cost_flavor="faithful"` by default; this module just makes that
binding queryable without re-implementing it (spec Answer 6: bind, never edit).
"""

from __future__ import annotations

from allo.spmw_cost_model import get_cost_model, CostModel

# The faithful flavor is today's pipeline default (compile_for_target's
# `cost_flavor="faithful"`); the suite scores every cell with it.
FAITHFUL = "faithful"


def bind_cost(target, flavor: str = FAITHFUL, concern: str = "kernel_cycles") -> CostModel:
    """Look up the CostModel the pipeline scores `target` with.

    `target` may be the target tree (with `.name`) or a bare target-name str.
    Returns the registered `CostModel` for `(name, flavor, concern)`; raises
    `KeyError` if none is registered (an unbound backend, not a silent default).
    """
    name = getattr(target, "name", target)
    return get_cost_model(name, flavor, concern)
