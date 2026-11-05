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


from .pim.apu_g2_source_route import check_launch_recipes  # noqa: E402

# Kept under its historical name for probes that import it from here.
_check_launch_recipes = check_launch_recipes


class APUG2SourceCtx(CodegenContext):
    """Backend context: the selected launch plan is the program."""

    def __init__(self, target):
        super().__init__(target)
        self.cmds: list[str] = []
        self.program = None

    # -- shared-pipeline candidate hooks ---------------------------------------

    @staticmethod
    def _payload(layouts):
        from .pim.apu_g2_record_frontend import APUG2RecordCandidate
        from .pim.apu_g2_source_route import APUG2SourceCandidate

        kinds = (APUG2SourceCandidate, APUG2RecordCandidate)
        layouts = layouts if isinstance(layouts, (list, tuple)) else [layouts]
        payloads = [
            (getattr(layout, "extra", None) or {}).get("payload") for layout in layouts
        ]
        if not any(isinstance(p, kinds) for p in payloads):
            return None
        if len(payloads) != 1 or not isinstance(payloads[0], kinds):
            raise NotImplementedError(
                "Gemini-II launch plans realize one plain-function program; a "
                "multi-kernel program mixes candidates"
            )
        return payloads[0]

    @classmethod
    def validate_orchestration(cls, trace, launch_schedule, host_moves) -> None:
        """A launch plan is the program's orchestration: one plan per kernel
        group, its launches back to back, host packing per launch."""

        from .spmw_codegen import _single_launch_orchestration

        _single_launch_orchestration(
            "apu_v2", trace, launch_schedule, host_moves, "Gemini-II launch plans"
        )

    @classmethod
    def candidate_execution_graph(cls, target, trace, layout, bound_cost, **options):
        """Price a launch-plan candidate: every launch's issued recipe,
        back to back (`price_apu_g2_launch_plan`); the plan keeps the
        graph and its estimate for inspection."""

        from .pim.apu_g2_source_route import price_apu_g2_launch_plan

        payload = cls._payload(layout)
        if payload is None:
            return None
        cls.validate_orchestration(
            trace, options.get("launch_schedule"), options.get("host_moves") or ()
        )
        from .pim.apu_g2_record_frontend import APUG2RecordCandidate

        if isinstance(payload, APUG2RecordCandidate):
            from .pim.apu_g2_record_frontend import record_execution_graph

            return record_execution_graph(payload.record, target, bound_cost)
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
        self.validate_orchestration(trace, schedule, host_moves)
        (placement,) = layouts if isinstance(layouts, (list, tuple)) else [layouts]
        selection = (
            "forced"
            if placement.extra.get("candidate_selection") == "forced"
            else "cost"
        )
        from .pim.apu_g2_record_frontend import (
            APUG2RecordCandidate,
            APUG2RecordProgram,
            record_fingerprint,
        )

        if isinstance(payload, APUG2RecordCandidate):
            self.program = APUG2RecordProgram(payload.record, payload.facts, selection)
            self.cmds = [
                f"record {payload.record.name}: {payload.kind} "
                f"{record_fingerprint(payload.record)}"
            ]
            return
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

    def check_realization(self, layouts) -> None:
        """The realized launch program against the candidate's contract:
        function, outputs, inputs and every launch (instances, reduction
        range, accumulator mode, streams), in order."""

        from .spmw_candidate import candidate_effects

        if self.program is None:
            return
        (placement,) = layouts if isinstance(layouts, (list, tuple)) else [layouts]
        effects = candidate_effects(placement)
        if effects is None:
            raise RuntimeError("Gemini-II candidate declares no effects")
        from .pim.apu_g2_record_frontend import APUG2RecordProgram

        if isinstance(self.program, APUG2RecordProgram):
            # The realization is the record itself: its effects, derived
            # again from the record and the facts (coverage, layouts,
            # transfers, the one launch's instances, reduction and
            # accumulator, ABI), must be the declared ones.
            from .pim.apu_g2_record_frontend import APUG2RecordCandidate

            record = self.program.record
            actual = APUG2RecordCandidate(record, self.program.facts).effects()
            if actual != effects:
                raise RuntimeError(
                    f"Gemini-II record realization {record.name!r} differs from the "
                    f"candidate contract: realized {actual.manifest()}, declared "
                    f"{effects.manifest()}"
                )
            return
        contract = self.program.contract
        plan = self.program.plan
        if set(contract.outputs) != set(effects.abi_outputs):
            raise RuntimeError(
                "Gemini-II realization differs from the candidate contract: "
                f"outputs {sorted(contract.outputs)} vs {sorted(effects.abi_outputs)}"
            )
        if contract.function != effects.coverage.function:
            raise RuntimeError(
                f"Gemini-II realization covers {contract.function!r}, the candidate "
                f"declared {effects.coverage.function!r}"
            )
        realized = [
            (
                tuple(int(v) for v in launch.coordinates),
                tuple(int(v) for v in launch.reduction),
                launch.accumulator,
                tuple(launch.streams),
            )
            for launch in plan.launches
        ]
        declared = [
            (
                tuple(launch.instances),
                tuple(launch.reduction or ()),
                launch.accumulator,
                tuple(launch.covers[1]) if len(launch.covers) > 1 else (),
            )
            for launch in effects.launches
        ]
        if realized != declared or any(
            launch.kind != "composed_dot" or launch.covers[:1] != (plan.name,)
            for launch in effects.launches
        ):
            raise RuntimeError(
                "Gemini-II realization launches differ from the candidate contract: "
                f"realized {realized}, declared {declared}"
            )
        check_launch_recipes(contract, plan)
        inputs = {
            access.parameter
            for stream in contract.streams
            for access in (stream.lhs, stream.rhs)
        } | {
            # The runtime packs a carried accumulator's buffer as the first
            # launch's accumulator input.
            stream.output
            for stream in contract.streams
            if stream.accumulator == "carried" and stream.output is not None
        }
        if inputs != set(effects.abi_inputs):
            raise RuntimeError(
                f"Gemini-II realization reads {sorted(inputs)}, the candidate "
                f"declared {sorted(effects.abi_inputs)}"
            )

    def runtime_route(self, commands) -> tuple[str, tuple]:
        if self.program is None:
            return super().runtime_route(commands)
        from .pim.apu_g2_record_frontend import APUG2RecordProgram, record_fingerprint

        if isinstance(self.program, APUG2RecordProgram):
            record = self.program.record
            return "apu-g2-record-v1", (record.name, record_fingerprint(record))
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
