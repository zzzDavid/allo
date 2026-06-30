# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Default calibration profiles for importable PIM targets.

Only parameter values live here. Formulas and resource topology are declared by
the target; measured overlays can replace this profile without changing either.
"""

from __future__ import annotations

from importlib.resources import files

from ..perf import CalibrationProfile, VirtualTarget


SAMSUNG_BASE_PROFILE = CalibrationProfile.load(
    files(__package__).joinpath("calibration/samsung_hbm_pim-base.json")
)


_DEFAULT_PROFILES = {
    "samsung_hbm_pim": SAMSUNG_BASE_PROFILE,
}


def default_profile(target) -> CalibrationProfile:
    name = getattr(target, "name", target)
    try:
        return _DEFAULT_PROFILES[name]
    except KeyError as exc:
        raise KeyError(
            f"no resource-DAG calibration profile for target {name!r}"
        ) from exc


def virtual_target(target, profile: CalibrationProfile | None = None) -> VirtualTarget:
    """Bind a structural target to a calibration profile."""
    if not getattr(target, "has_performance_model", False):
        raise ValueError(
            f"target {getattr(target, 'name', None)!r} has no performance model"
        )
    return VirtualTarget(
        target=target,
        topology=target.resource_topology,
        timings=target.timing_library,
        calibration=profile or default_profile(target),
    )
