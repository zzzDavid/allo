# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resource budget: a search policy that caps the device units a program may
occupy (``allo.compile(..., resources={"channel": 8})``).

A budget names structural units of the target (``target.unit(name)``) and,
for each, the largest number of them one compiled program may use.  It is
stamped on every match of the trace before enumeration, so the backend
enumerators bound their partition domains by it, the shared ranking selects
among the candidates that fit, and the realization check compares the
lowered program's unit use with it.  A supplied placement whose decision
does not fit the budget designates no candidate of the budgeted enumeration
and is refused by the ordinary rebinding.  The budget is recorded on every
result (``RunResult.extra["resource_budget"]``).  It is a cap on the
search, never a device description: the target's geometry (and the
simulator's) is unchanged, and a cap above the device's count is refused as
a mistake.

The contract is bounded to the (target, unit) pairs a backend enforces end
to end (:data:`BUDGET_UNITS`, registered by the backend module): a budget
on any other pair is refused before enumeration rather than accepted and
ignored.  Today: AiM ``channel`` (the replica and token partition domains in
``spmw_knobs``, the realization check in ``spmw_aim``).
"""

from __future__ import annotations

from collections.abc import Mapping
import math

BUDGET_KEY = "resource_budget"

# target name -> units whose budget that backend's enumeration and
# realization check enforce; the declared contract, checked before any
# backend module is imported.  AiM ``channel``: the replica and token
# partition domains (``spmw_knobs._budgeted_channels``) and the lowered
# program's channel use (``spmw_aim.AimCtx.check_realization``).  A backend
# adding an enforced pair extends this table (``register_budget_unit``).
BUDGET_UNITS: dict = {"aim": {"channel"}}


def register_budget_unit(target_name: str, unit: str) -> None:
    """Declare that ``target_name``'s enumerators bound their candidates by
    a cap on ``unit`` and its realization check compares the lowered
    program's use of ``unit`` with the cap."""
    BUDGET_UNITS.setdefault(str(target_name), set()).add(str(unit))


def supported_budget_units(target) -> set:
    """The units a budget may name for ``target`` (empty when none)."""
    return set(BUDGET_UNITS.get(getattr(target, "name", None), ()))


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
            raise ValueError(
                f"resources[{name!r}] must be a positive integer, got {cap!r}"
            )
        if target is not None:
            try:
                declared = target.unit(name)
            except KeyError as error:
                raise ValueError(
                    f"resources names unit {name!r}, which target "
                    f"{getattr(target, 'name', None)!r} does not declare"
                ) from error
            supported = supported_budget_units(target)
            if name not in supported:
                enforced = (
                    ", ".join(
                        f"{t}: {', '.join(sorted(units))}"
                        for t, units in sorted(BUDGET_UNITS.items())
                    )
                    or "none"
                )
                raise ValueError(
                    f"resources[{name!r}]: target {getattr(target, 'name', None)!r} "
                    f"enforces no budget on {name!r} (its enumeration and realization "
                    f"check do not read it); budgets are enforced for {enforced}"
                )
            count = (
                math.prod(int(v) for v in getattr(declared, "axes", {}).values()) or 1
            )
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
    "BUDGET_UNITS",
    "budget_of",
    "normalize_resource_budget",
    "register_budget_unit",
    "resource_cap",
    "stamp_resource_budget",
    "supported_budget_units",
]
