# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A target topology plus timing profile, executable without hardware."""

from __future__ import annotations

from dataclasses import dataclass, field

from .calibration import CalibrationProfile
from .evaluator import Estimate, Evaluator
from .graph import ExecutionGraph
from .resources import ResourceTopology
from .timing import TimingLibrary


@dataclass(frozen=True)
class VirtualTarget:
    target: object
    topology: ResourceTopology
    timings: TimingLibrary
    calibration: CalibrationProfile
    evaluator: Evaluator = field(default_factory=Evaluator)

    def __post_init__(self):
        target_name = getattr(self.target, "name", None)
        if target_name is not None and target_name != self.calibration.target:
            raise ValueError(
                f"calibration targets {self.calibration.target!r}, not {target_name!r}"
            )

    def evaluate(self, graph: ExecutionGraph) -> Estimate:
        return self.evaluator.evaluate(
            graph, self.topology, self.timings, self.calibration
        )
