# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resource budget: a search policy that caps the device units a program may
occupy (``allo.compile(..., resources={"channel": 8})``).

A budget names structural units of the target (``target.unit(name)``) and,
for each, the largest number of them one compiled program may use.  It is
stamped on every match of the trace before enumeration, so the backend
enumerators bound their partition domains by it (the AiM replica and token
partition counts, ``spmw_knobs``), the shared ranking selects among the
candidates that fit, and the realization check compares the lowered
program's unit use with it.  A supplied placement whose decision does not
fit the budget designates no candidate of the budgeted enumeration and is
refused by the ordinary rebinding.  The budget is recorded on every result
(``RunResult.extra["resource_budget"]``).  It is a cap on the search, never a
device description: the target's geometry (and the simulator's) is
unchanged, and a cap above the device's count is refused as a mistake.
"""

from __future__ import annotations

from collections.abc import Mapping
import math

BUDGET_KEY = "resource_budget"


def normalize_resource_budget(value, target=None) -> dict | None:
    """``{unit name: cap}`` from a user-supplied budget, or None.

    Every unit must be a structural unit of ``target`` (when given) and
    every cap a positive integer no larger than the device's count of that
    unit (the product of the unit's axes)."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(
            f"resources must be a mapping of unit name to cap, got {type(value).__name__}"
        )
    budget = {}
    for unit, cap in value.items():
        name = str(unit)
        if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
            raise ValueError(f"resources[{name!r}] must be a positive integer, got {cap!r}")
        if target is not None:
            try:
                declared = target.unit(name)
            except KeyError as error:
                raise ValueError(
                    f"resources names unit {name!r}, which target "
                    f"{getattr(target, 'name', None)!r} does not declare"
                ) from error
            count = math.prod(int(v) for v in getattr(declared, "axes", {}).values()) or 1
            if cap > count:
                raise ValueError(
                    f"resources[{name!r}] = {cap} exceeds the {count} {name} unit(s) "
                    f"of target {getattr(target, 'name', None)!r}"
                )
        budget[name] = int(cap)
    return budget or None


def stamp_resource_budget(trace_or_matches, budget) -> None:
    """Stamp the budget on every match (remove a previous one when None)."""
    matches = getattr(trace_or_matches, "matches", trace_or_matches)
    for match in matches:
        if budget:
            match.extra[BUDGET_KEY] = dict(budget)
        else:
            match.extra.pop(BUDGET_KEY, None)


def budget_of(matches) -> dict | None:
    """The budget stamped on a work item's matches (the first match's)."""
    for match in matches or ():
        budget = (getattr(match, "extra", None) or {}).get(BUDGET_KEY)
        if budget:
            return dict(budget)
        return None
    return None


def resource_cap(matches, unit: str) -> int | None:
    """The cap on ``unit`` for a work item, or None when unbudgeted."""
    budget = budget_of(matches)
    return None if budget is None else budget.get(unit)


__all__ = [
    "BUDGET_KEY",
    "budget_of",
    "normalize_resource_budget",
    "resource_cap",
    "stamp_resource_budget",
]
