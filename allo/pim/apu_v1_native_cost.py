# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared execution-graph builder for complete APUv1 native recipes.

Native recipes bypass the ordinary retained-MLIR scalar estimator because
their device source executes a proven GVML plan instead of that scalar loop
nest.  This builder keeps those recipes on the normal target/cost path: every
step is a target primitive interpreted by the bound APUv1 cost specification,
and the exact lowering inventory remains attached to the graph for audit.
"""

from __future__ import annotations

from ..perf import CostEvent
from ..perf.graph import ExecutionGraph


class APUv1NativeCostGraphBuilder:
    """Emit one sequential, target-bound graph for a native GVML recipe."""

    def __init__(self, route, inventory, target, cost):
        self.route = route
        self.inventory = dict(inventory)
        self.target = target
        self.cost = cost
        self.graph = ExecutionGraph(
            route,
            metadata={
                "target": "apu_v1",
                "route": route,
                "operation_inventory": self.inventory,
                "analytical": True,
                "estimator": "native_gvml_structural_cost",
            },
        )
        self._dependencies = ()
        self._event_index = 0

    def _emit(self, handle, operation, *, count=1, **metrics):
        count = int(count)
        if count < 0:
            raise ValueError(
                f"native APUv1 operation count cannot be negative: {count}"
            )
        if count == 0:
            return
        event = CostEvent.create(
            f"native-vector:{self._event_index}:{operation}",
            handle,
            work_id=(0,),
            metrics={"count": count, **metrics},
            attributes={"route": self.route, "operation": operation},
        )
        self._dependencies = tuple(
            self.cost.emit(self.graph, event, self._dependencies)
        )
        self._event_index += 1

    def move(self, primitive, operation, *, count=1, **metrics):
        self._emit(
            self.target.move(primitive),
            operation,
            count=count,
            **metrics,
        )

    def op(self, primitive, operation, *, count=1, **metrics):
        self._emit(
            self.target.op(primitive),
            operation,
            count=count,
            **metrics,
        )

    def finish(self):
        if not self.graph.activities:
            raise ValueError("native APUv1 execution graph must contain an activity")
        return self.graph


__all__ = ["APUv1NativeCostGraphBuilder"]
