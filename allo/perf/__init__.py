# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cycle-based analytical performance modeling for Tenon targets.

The package deliberately separates three questions:

* :mod:`allo.perf.graph` records what executes and its dependencies.
* :mod:`allo.perf.resources` records what the target can execute in parallel.
* :mod:`allo.perf.timing` records how many cycles an activity takes.

``Evaluator`` combines the three without any target-specific composition code.
"""

from .calibration import (
    CalibrationProfile,
    MeasurementRecord,
    ParameterValue,
    ProbeSpec,
    fit_profile,
)
from .evaluator import ActivitySpan, Estimate, Evaluator
from .graph import Activity, ExecutionGraph, Invocation
from .resources import ResourceRequest, ResourceSpec, ResourceTopology
from .timing import CycleTiming, TimingLibrary, TimingModel
from .virtual_target import VirtualTarget

__all__ = [
    "Activity",
    "ActivitySpan",
    "CalibrationProfile",
    "CycleTiming",
    "Estimate",
    "Evaluator",
    "ExecutionGraph",
    "Invocation",
    "MeasurementRecord",
    "ParameterValue",
    "ProbeSpec",
    "ResourceRequest",
    "ResourceSpec",
    "ResourceTopology",
    "TimingLibrary",
    "TimingModel",
    "VirtualTarget",
    "fit_profile",
]
