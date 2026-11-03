# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gemini-II (``apu_v2``) backend context and runner of the shared pipeline.

Ordinary integer contractions reach this backend through the common
candidate pipeline (release gate G9): the matcher runs against the target's
source-idiom view, the shared semantic facts feed
:func:`allo.pim.apu_g2_source_route.apu_g2_source_candidates`, which
enumerates one ``Placement`` per composed-dot launch plan, the shared driver
prices every candidate through :meth:`APUG2SourceCtx.candidate_execution_graph`
and ranks them, and this context materializes the selected plan as an
:class:`allo.pim.apu_g2_source_route.APUG2SourceProgram`.

``backend=None`` packs the host carriers, launches every program on the card
through :mod:`allo.pim.apu_g2_composed_runtime` and checks the gathered
result against the source oracle (``kind="device"``); ``backend="virtual"``
is the shared virtual runner over the selected plan's execution graph.
"""

from __future__ import annotations

from .spmw_codegen import CodegenContext, RunResult


class APUG2SourceCtx(CodegenContext):
    """Backend context: the selected launch plan is the program."""

    def __init__(self, target):
        super().__init__(target)
        self.cmds: list[str] = []
        self.program = None

    # -- shared-pipeline candidate hooks ---------------------------------------

    @staticmethod
    def _payload(layouts):
        from .pim.apu_g2_source_route import APUG2SourceCandidate

        layouts = layouts if isinstance(layouts, (list, tuple)) else [layouts]
        payloads = [
            (getattr(layout, "extra", None) or {}).get("payload") for layout in layouts
        ]
        if not any(isinstance(p, APUG2SourceCandidate) for p in payloads):
            return None
        if len(payloads) != 1 or not isinstance(payloads[0], APUG2SourceCandidate):
            raise NotImplementedError(
                "Gemini-II launch plans realize one plain-function program; a "
                "multi-kernel program mixes candidates"
            )
        return payloads[0]

    @classmethod
    def candidate_execution_graph(cls, target, _trace, layout, bound_cost, **_options):
        """Price a launch-plan candidate: every launch's issued recipe,
        back to back (`price_apu_g2_launch_plan`); the plan keeps the
        graph and its estimate for inspection."""

        from .pim.apu_g2_source_route import price_apu_g2_launch_plan

        payload = cls._payload(layout)
        if payload is None:
            return None
        plan = payload.plan
        fingerprint = getattr(bound_cost, "fingerprint", None)
        graph = plan.execution_graph
        if graph is None or graph.metadata.get("cost_fingerprint") != fingerprint:
            graph = price_apu_g2_launch_plan(plan, target, bound_cost)
            plan.execution_graph = graph
            plan.estimate_cycles = int(bound_cost.evaluate(graph).cycles)
            plan.estimate_validity = graph.metadata["estimate_validity"]
        return graph

    def emit_program(
        self, trace, layouts, *, schedule=None, host_moves=(), module=None
    ) -> None:
        """Materialize the selected launch plan as the realized program."""

        from .pim.apu_g2_source_route import APUG2SourceProgram

        payload = self._payload(layouts)
        if payload is None:
            raise NotImplementedError(
                "Gemini-II executes composed-dot launch plans only; the selected "
                "placement carries no launch plan"
            )
        (placement,) = layouts if isinstance(layouts, (list, tuple)) else [layouts]
        selection = (
            "forced"
            if placement.extra.get("candidate_selection") == "forced"
            else "cost"
        )
        self.program = APUG2SourceProgram(
            payload.contract,
            self.target,
            payload.plan,
            candidates=payload.catalog,
            rejected=payload.rejected,
            selection=selection,
        )
        self.cmds = [
            f"launch {index}: streams={list(launch.streams)} "
            f"coordinates={list(launch.coordinates)} "
            f"reduction={list(launch.reduction)} accumulator={launch.accumulator} "
            f"program={launch.program.structural_fingerprint}"
            for index, launch in enumerate(payload.plan.launches)
        ]

    def runtime_route(self, commands) -> tuple[str, tuple]:
        if self.program is None:
            return super().runtime_route(commands)
        return "apu-g2-composed-dot-v1", tuple(
            launch.layout_fingerprint for launch in self.program.plan.launches
        )

    def virtual_run_extra(self) -> dict:
        return {} if self.program is None else self.program.virtual_extra()

    def cmd(self, name: str, **fields):
        raise NotImplementedError(
            "the Gemini-II backend emits launch plans, not per-op commands"
        )

    def resolve_moves(self, role, src_handle=None, dst_handle=None):
        # The launch plan owns its transfers (host packing per launch).
        return (None, None)

    def resolve_spill_moves(self, tier, home_handle, *, n_entries=1):
        raise NotImplementedError("Gemini-II launch plans do not spill")


def _run_apu_g2(compiled, **inputs) -> RunResult:
    """Run the selected launch plan on the Gemini-II card."""

    program = getattr(getattr(compiled, "layout_ctx", None), "program", None)
    if program is None:
        raise RuntimeError("the compiled Gemini-II artifact has no launch plan")
    return program.run(compiled, inputs)


__all__ = ["APUG2SourceCtx", "_run_apu_g2"]
