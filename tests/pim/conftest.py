# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test harness for ``allo.pim`` when the full ``import allo`` is
unavailable.

``import allo`` requires the MLIR C-extension ``allo._mlir`` to be
built, which is not always the case in CI sandboxes that only exercise
the pure-Python ``allo.pim`` subpackage. This conftest detects that
situation and installs a synthetic namespace package under the same
name ``allo`` (with the real ``allo/pim/`` directory on its
``__path__``), so that the canonical ``from allo.pim...`` import works
without executing ``allo/__init__.py``.

If a proper ``import allo`` succeeds (MLIR is built and the full Allo
package is importable), we leave sys.modules alone; tests then import
from the real Allo package.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types


def _install_synthetic_allo_pim():
    """Install a minimal synthetic ``allo`` package that exposes
    ``allo.pim`` (and its submodules), plus the pure-Python
    ``allo.unit`` / ``allo.work`` / ``allo.compile`` modules. Only
    touches sys.modules if ``import allo`` currently fails.

    The implementation trick: register the synthetic ``allo`` package
    with the real ``allo/`` source dir on its ``__path__``. Then every
    ``import allo.pim`` / ``from allo.pim.target import ...`` falls
    through Python's normal submodule finder, which discovers the real
    files on disk. No hand-loading of submodules is required — the
    standard import machinery does all of it, which also guarantees
    each ``allo.pim.X`` module is instantiated exactly once."""
    try:
        import allo  # noqa: F401 — probe real allo.
        if hasattr(allo, "pim"):
            return
        # ``import allo`` worked but didn't expose ``pim`` — rare, but
        # force it in so downstream ``from allo.pim import ...`` works.
        from allo import pim  # noqa: F401
        return
    except Exception:
        pass

    if "allo" in sys.modules and hasattr(sys.modules["allo"], "pim"):
        return

    here = os.path.dirname(os.path.abspath(__file__))
    allo_pkg_dir = os.path.normpath(os.path.join(here, "..", "..", "allo"))

    # Synthetic ``allo`` package pointing at the real ``allo/``
    # directory. We don't execute ``allo/__init__.py`` because it
    # eagerly imports the unbuilt MLIR C-extension; just build a bare
    # namespace package whose __path__ lets the normal submodule finder
    # locate the real ``allo/pim`` subpackage on disk.
    allo_pkg = types.ModuleType("allo")
    allo_pkg.__path__ = [allo_pkg_dir]
    sys.modules["allo"] = allo_pkg

    # Trigger the REAL import machinery for ``allo.pim`` — this
    # ensures each ``allo.pim.X`` module is loaded exactly once (which
    # is critical for ``isinstance(...)`` checks across modules).
    pim_mod = importlib.import_module("allo.pim")
    setattr(allo_pkg, "pim", pim_mod)

    # Pre-import ``allo.pim.backends`` so the backend submodules are
    # registered under their canonical names.
    backends_mod = importlib.import_module("allo.pim.backends")
    setattr(pim_mod, "backends", backends_mod)

    # ``allo.pim.runtime`` is a namespace with simulator-specific
    # submodules that some tests import on demand.
    runtime_mod = importlib.import_module("allo.pim.runtime")
    setattr(pim_mod, "runtime", runtime_mod)

    # allo.unit / allo.work / allo.compile — pure-Python. Loading them
    # via importlib (not spec_from_file_location) lets relative imports
    # like ``from .pim.target import ...`` inside ``unit.py`` resolve
    # against the synthetic ``allo`` package we just installed.
    for sub in ("unit", "work", "compile"):
        m = importlib.import_module(f"allo.{sub}")
        setattr(allo_pkg, sub, m)
        for attr in ("target", "unit", "memory", "op", "stream",
                     "cost", "work", "Work", "compile"):
            if hasattr(m, attr):
                setattr(allo_pkg, attr, getattr(m, attr))


_install_synthetic_allo_pim()
