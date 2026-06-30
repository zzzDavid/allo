# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PolyBench-on-PIM suite: resource-DAG calibration binding surface.

Samsung binds the packaged JSON-backed ``CalibrationProfile`` consumed by
``allo.compile``. A target not yet ported to the new model returns ``None`` so
the same public facade can compile the rest of the cross-target suite.
"""

from __future__ import annotations

from allo.pim.performance import default_profile


def bind_cost(target):
    """Return the target's default calibration profile, if it has one."""
    try:
        return default_profile(target)
    except KeyError:
        return None
