# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SPMW cost registry (Tenon `@tn.cost` / `tn.get_cost`).

Step D of report 16: cost modules are factories named by *what they
price*, not by which target. The factory takes a `target` argument and
returns a closure (the cost callback) specialized to that target.

Per the report, the factory runs once per target (mutating it to attach
e.g. per-move cycles); the returned closure is what an autoscheduler
calls repeatedly during search. `get_cost` caches by (name, target id)
so the factory does not re-run.
"""

_registry: dict[str, callable] = {}
_callback_cache: dict[tuple[str, int], callable] = {}


def cost(name):
    """Decorator: register a cost-factory under `name`."""

    def decorator(fn):
        if name in _registry:
            raise ValueError(f"duplicate cost name {name!r}")
        _registry[name] = fn
        return fn

    return decorator


def get_cost(name, target):
    """Build (or reuse) the cost callback specialized to `target`."""
    if name not in _registry:
        raise KeyError(f"no cost registered as {name!r}")
    key = (name, id(target))
    cb = _callback_cache.get(key)
    if cb is None:
        cb = _registry[name](target)
        _callback_cache[key] = cb
    return cb
