# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every public export the PIM and SPMW modules declare exists (round 87).

``allo.pim.apu_v1_record_frequency`` exported a ``MAX_RESIDENT_CHUNKS`` it
no longer defined, so ``from allo.pim.apu_v1_record_frequency import *``
raised ``AttributeError`` (the reviewer's round 86 export probe, pylint
E0603).  Importing by star is part of the public interface, so every name
in an ``__all__`` of these modules must resolve.
"""

import importlib
import pkgutil

import pytest

import allo
import allo.pim


def _modules():
    names = [m.name for m in pkgutil.walk_packages(allo.pim.__path__, "allo.pim.")]
    names += [
        m.name
        for m in pkgutil.iter_modules(allo.__path__, "allo.")
        if m.name.startswith("allo.spmw")
    ]
    return sorted(names)


@pytest.mark.parametrize("module_name", _modules())
def test_every_declared_export_exists(module_name):
    module = importlib.import_module(module_name)
    declared = getattr(module, "__all__", None)
    if declared is None:
        pytest.skip("no __all__")
    missing = [name for name in declared if not hasattr(module, name)]
    assert missing == [], f"{module_name}.__all__ names {missing} it does not define"


def test_star_import_of_the_record_frequency_module_works():
    namespace = {}
    exec("from allo.pim.apu_v1_record_frequency import *", namespace)
    assert "analyze_apu_v1_record_frequency" in namespace
    assert "MAX_RESIDENT_CHUNKS" not in namespace
