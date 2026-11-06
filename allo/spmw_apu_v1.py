# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""GSI APU v1 codegen context and device runtime (moved from spmw_codegen)."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .spmw_target import MemoryRef, Register
from .spmw_codegen import (
    CodegenContext,
    RunResult,
    SimulatorUnavailable,
    Compiled,
)
from .spmw_simenv import apu_v1_unavailable_reason as _apu_v1_unavailable_reason


class APUv1Ctx(CodegenContext):
    """Codegen ctx for GSI APU v1.

    Output is a list of C-source lines containing real GVML calls. Logical
    reductions use an F2 layout-derived group size and lower to native FP16
    multiply plus ``gvml_add_subgrps_f16_grp``.
    """

    _L4_PTR_BY_ROLE = {
        "x": "inp_L4ptr",
        "y": "wgt_L4ptr",
        "acc": "out_L4ptr",
        "dst": "out_L4ptr",
    }

    def __init__(self, target):
        super().__init__(target)
        # Override the parent's PIMCmd list -- APU v1 emits plain C text.
        self.cmds: list[str] = []
        # The realized vector program when the selected candidate is a GVML
        # vector plan of a plain-function contraction
        # (`allo.pim.apu_v1_vector_program.APUv1VectorProgram`); None on the
        # grouped matcher program of a dataflow region.
        self.program = None
        # Reverse-handle map populated by the autoscheduler before
        # walk_and_emit runs. Maps id(handle) -> C symbol. Also accepts
        # string keys (e.g. "mac_tmp") for scratch slots that aren't
        # backed by a Tenon handle.
        self._handle_names: dict = {}
        self._used_vr_names: list[str] = []
        self.group_size: int = 32768
        self.vector_batches: int = 1
        # What the grouped emission did, as the emitters did it (work id,
        # event): the loads and stores by role and register, every grouped
        # MAC with its group size, every other operation.
        # `check_realization` compares these with the candidate's declared
        # effects (ISSUES X-28); the per-bucket layouts and role maps the
        # walk received are kept beside them.
        self.realized: list[tuple] = []
        self._grouped_buckets: list[tuple] | None = None

    # -- shared-pipeline candidate hooks (gate G9) ---------------------------

    @staticmethod
    def _vector_payloads(layouts):
        """The one plan-carrying payload of ``layouts`` (a vector plan of a
        plain-function contraction or the lowering of an ``APUv1Program``
        record), None for handle-only grouped placements."""
        from .pim.apu_v1_record_frontend import APUv1RecordCandidate
        from .pim.apu_v1_vector_program import APUv1VectorCandidate

        kinds = (APUv1VectorCandidate, APUv1RecordCandidate)
        layouts = layouts if isinstance(layouts, (list, tuple)) else [layouts]
        payloads = [
            (getattr(layout, "extra", None) or {}).get("payload") for layout in layouts
        ]
        if not any(isinstance(p, kinds) for p in payloads):
            return None
        if len(payloads) != 1 or not isinstance(payloads[0], kinds):
            raise NotImplementedError(
                "APU v1 vector plans and program records realize one "
                "plain-function program; a multi-kernel program mixes candidates"
            )
        return payloads[0]

    @classmethod
    def validate_orchestration(cls, trace, launch_schedule, host_moves) -> None:
        """Vector plans and the grouped program execute one launch of the
        whole problem per kernel group and own their transfers."""

        from .spmw_codegen import _single_launch_orchestration

        _single_launch_orchestration(
            "apu_v1", trace, launch_schedule, host_moves, "APU v1 programs"
        )

    @classmethod
    def candidate_execution_graph(cls, target, trace, layout, bound_cost, **options):
        """Price a vector-plan candidate from its materialized GVML inventory
        (``apu_v1_cost`` over the realized operations); None for the grouped
        matcher placements, which take the generic lowering."""

        from .pim.apu_v1_vector_cost import estimate_apu_v1_realization
        from .pim.schedule_search import InfeasibleSchedule

        payload = cls._vector_payloads(layout)
        if payload is None:
            return None
        cls.validate_orchestration(
            trace, options.get("launch_schedule"), options.get("host_moves") or ()
        )
        from .pim.apu_v1_record_frontend import APUv1RecordCandidate

        if isinstance(payload, APUv1RecordCandidate):
            # The record's retained graph: the native recipe's structural
            # graph or the scalar C instruction count of the retained MLIR.
            return payload.lowered.execution_graph_for(bound_cost)
        if payload.realization is None:
            raise InfeasibleSchedule(
                payload.realization_error
                or "APU v1 plan has no faithful GVML " "realization"
            )
        estimate = payload.estimate
        if estimate is None or estimate.graph.metadata.get(
            "cost_fingerprint"
        ) != getattr(bound_cost, "fingerprint", None):
            estimate = estimate_apu_v1_realization(
                payload.realization, target, bound_cost
            )
            payload.estimate = estimate
        return estimate.graph

    def emit_program(
        self, trace, layouts, *, schedule=None, host_moves=(), module=None
    ) -> None:
        """Materialize the selected candidate: a vector plan instantiates
        the realized program (its GVML device source is the command stream);
        a grouped placement of a dataflow region takes the per-bucket walk."""

        from .pim.apu_v1_vector_program import APUv1VectorProgram
        from .pim.schedule_search import InfeasibleSchedule

        payload = self._vector_payloads(layouts)
        if payload is None:
            from .spmw_autoschedule import (
                _bucket_for_autoschedule,
                _matcher_work_scope,
                _trace_memrefs_by_role,
            )

            layout_list = layouts if isinstance(layouts, (list, tuple)) else [layouts]
            self._grouped_buckets = []
            for (_scope, matches), layout in zip(
                _bucket_for_autoschedule(trace), layout_list
            ):
                try:
                    roles = _trace_memrefs_by_role(matches)
                except NotImplementedError:
                    roles = {}
                self._grouped_buckets.append(
                    (
                        _matcher_work_scope(matches[0]).work_id,
                        list(matches),
                        layout,
                        roles,
                    )
                )
            self.realized = []
            super().emit_program(
                trace, layouts, schedule=schedule, host_moves=host_moves, module=module
            )
            return
        from .pim.apu_v1_record_frontend import APUv1RecordCandidate, APUv1RecordProgram

        if isinstance(payload, APUv1RecordCandidate):
            self.validate_orchestration(trace, schedule, host_moves)
            # A native recipe has no host execution path: refuse
            # backend="functional" here, before an artifact exists.
            payload.lowered.check_backend(getattr(self, "backend", None))
            (placement,) = layouts if isinstance(layouts, (list, tuple)) else [layouts]
            selection = (
                "forced"
                if placement.extra.get("candidate_selection") == "forced"
                else "cost"
            )
            self.program = APUv1RecordProgram(payload, selection)
            self.cmds = list(payload.lowered.device_source.splitlines())
            return
        if payload.realization is None:
            raise InfeasibleSchedule(
                payload.realization_error
                or "APU v1 plan has no faithful GVML " "realization"
            )
        self.validate_orchestration(trace, schedule, host_moves)
        (placement,) = layouts if isinstance(layouts, (list, tuple)) else [layouts]
        self.program = APUv1VectorProgram(self.target, payload, trace, placement)
        self.cmds = list(payload.realization.device_source().splitlines())

    def check_realization(self, layouts) -> None:
        """The realized GVML program against the candidate's declared
        contract: ABI, value layouts, transfers and the single launch."""

        from .spmw_candidate import candidate_effects

        if self.program is None:
            self._check_grouped_realization(layouts)
            return
        (placement,) = layouts if isinstance(layouts, (list, tuple)) else [layouts]
        effects = candidate_effects(placement)
        if effects is None:
            raise RuntimeError("APU v1 vector candidate declares no effects")
        from .pim.apu_v1_record_frontend import APUv1RecordProgram

        if isinstance(self.program, APUv1RecordProgram):
            # The realized runtime moves every source argument to L4, reads
            # the outputs back and allocates the scalar program's scratch:
            # the declaration's transfers and allocations must be exactly
            # that movement, and the declared launch must be this lowering.
            transfers, allocations = self.program.candidate.realized_transfers()
            declared = {
                (t.value, t.direction, tuple(t.route), t.physical)
                for t in effects.transfers
            }
            realized = {
                (t.value, t.direction, tuple(t.route), t.physical) for t in transfers
            }
            if declared != realized:
                raise RuntimeError(
                    "APU v1 program realization moves "
                    f"{sorted(realized)}, the candidate declared {sorted(declared)}"
                )
            if tuple(effects.allocations) != tuple(allocations):
                raise RuntimeError(
                    "APU v1 program realization allocates "
                    f"{list(allocations)}, the candidate declared "
                    f"{list(effects.allocations)}"
                )
            lowered = self.program.lowered
            abi_outputs = {argument.name for argument in lowered.outputs}
            if not set(effects.abi_outputs) <= abi_outputs:
                raise RuntimeError(
                    "APU v1 program realization writes "
                    f"{sorted(abi_outputs)}, the candidate declared outputs "
                    f"{sorted(effects.abi_outputs)}"
                )
            (launch,) = effects.launches
            from .pim.apu_v1_record_frontend import program_fingerprint

            if not launch.whole or launch.covers != (
                lowered.route,
                program_fingerprint(lowered),
            ):
                raise RuntimeError(
                    "APU v1 program realization is the whole-function lowering "
                    f"{lowered.route!r}; the candidate declares launch "
                    f"{(launch.kind, launch.whole, launch.covers)}"
                )
            return
        realization = self.program.realization
        values = {value.name: value.intent for value in realization.values}
        realized_outputs = {
            n for n, intent in values.items() if intent in {"out", "inout"}
        }
        realized_inputs = {n for n, intent in values.items() if intent == "in"}
        analysis = self.program.candidate.analysis
        if analysis.output_initializer == "carried":
            # The program reads the output's incoming contents as the
            # accumulator seed: a realized input as well as an output.
            realized_inputs.add(analysis.output.value)
        if realized_outputs != set(effects.abi_outputs) or realized_inputs != set(
            effects.abi_inputs
        ):
            raise RuntimeError(
                "APU v1 realization ABI differs from the candidate contract: "
                f"realized inputs {sorted(realized_inputs)} / outputs "
                f"{sorted(realized_outputs)}, declared {sorted(effects.abi_inputs)} / "
                f"{sorted(effects.abi_outputs)}"
            )
        plan = realization.plan
        declared_layouts = {(v.value, tuple(v.axes)) for v in effects.value_layouts}
        realized_layouts = {(v.value, tuple(v.axes)) for v in plan.value_layouts}
        if declared_layouts != realized_layouts:
            raise RuntimeError(
                "APU v1 realization value layouts differ from the candidate "
                f"contract: realized {sorted(realized_layouts)}, declared "
                f"{sorted(declared_layouts)}"
            )
        declared_transfers = {(t.value, tuple(t.route)) for t in effects.transfers}
        realized_transfers = {
            (t.value, tuple(step.kind for step in t.route)) for t in plan.transfers
        }
        if declared_transfers != realized_transfers:
            raise RuntimeError(
                "APU v1 realization transfers differ from the candidate contract: "
                f"realized {sorted(realized_transfers)}, declared "
                f"{sorted(declared_transfers)}"
            )
        launches = effects.launches
        if (
            len(launches) != 1
            or launches[0].kind != "gvml_program"
            or not launches[0].covers
            or launches[0].covers[0] != plan.name
        ):
            raise RuntimeError(
                "APU v1 realization is one GVML program of plan "
                f"{plan.name!r}; the candidate declares launches "
                f"{[(l.kind, l.covers) for l in launches]}"
            )
        analysis = self.program.candidate.analysis
        extents = analysis.axis_extents
        instances = 1
        for axis in analysis.output_axes:
            instances *= int(extents[axis])
        realized = (
            (0, instances),
            (0, int(extents[analysis.reduction_axis])),
            analysis.output_initializer,
        )
        declared = (
            tuple(launches[0].instances),
            tuple(launches[0].reduction or ()),
            launches[0].accumulator,
        )
        if realized != declared:
            raise RuntimeError(
                "APU v1 realization launch differs from the candidate contract: "
                f"realized (instances, reduction, accumulator) {realized}, "
                f"declared {declared}"
            )
        self._check_vector_compute(realization, analysis, plan)

    def _check_vector_compute(self, realization, analysis, plan) -> None:
        """The realized computation against the contraction (round 79,
        ISSUES X-35).

        Three bindings, each re-derived structurally from the shared
        contraction analysis and the selected plan, none from an approved
        fingerprint: (1) the realization's neutral operations must be the
        operations the contraction and the plan's reduction strategy derive
        (`_realization_operations`: opcode, output, inputs in order, attrs);
        (2) the realization's compute instructions must be the lowering of
        those operations (`_op_instructions`: API, every argument including
        the operand registers in order, definitions, uses, predicate); (3)
        every such instruction, rendered with the realization's register
        bindings, must appear in the emitted device source.  A realization
        whose multiply reads the left operand twice, or whose emitted call
        does, is refused here as a feasibility rejection of the candidate
        (the reviewer's round 78 injections: damaged neutral operations
        behind a coherent realization, and a damaged instruction behind
        intact operations).
        """
        from .pim.apu_v1_vector_codegen import (
            VRRef,
            _normalize_operations,
            _op_instructions,
        )
        from .pim.apu_v1_vector_program import _realization_operations

        derived = _realization_operations(analysis, plan)
        if derived is None:
            raise RuntimeError(
                "APU v1 realization has no derivable computation for "
                f"{analysis.multiply_operation!r} under {plan.name!r}"
            )
        expected_ops = tuple(_normalize_operations(plan, derived))
        realized_ops = tuple(realization.operations)

        def op_key(op):
            return (op.opcode, op.output, tuple(op.inputs), dict(op.attrs))

        if [op_key(o) for o in realized_ops] != [op_key(o) for o in expected_ops]:
            raise RuntimeError(
                "APU v1 realization operations differ from the contraction: "
                f"realized {[op_key(o) for o in realized_ops]}, derived from the "
                f"shared contraction and plan {plan.name!r} "
                f"{[op_key(o) for o in expected_ops]}"
            )

        def arg_key(value):
            if isinstance(value, VRRef):
                return ("vr", value.name)
            return ("literal", repr(value))

        def instr_key(instr):
            return (
                instr.opcode,
                instr.api,
                tuple(arg_key(a) for a in instr.args),
                tuple(instr.definitions),
                tuple(instr.uses),
                instr.predicate,
            )

        expected_instr = [
            instr
            for index, op in enumerate(expected_ops)
            for instr in _op_instructions(op, index)
        ]
        realized_instr = [i for i in realization.instructions if i.phase == "compute"]
        if [instr_key(i) for i in realized_instr] != [
            instr_key(i) for i in expected_instr
        ]:
            raise RuntimeError(
                "APU v1 realization instructions differ from the contraction's "
                f"operations: realized {[instr_key(i) for i in realized_instr]}, "
                f"derived {[instr_key(i) for i in expected_instr]}"
            )
        bindings = {b.logical_name: b for b in realization.vr_bindings}
        emitted = {line.strip() for line in self.cmds}
        for instr in expected_instr:
            text = instr.render(bindings).strip()
            if text not in emitted:
                raise RuntimeError(
                    "APU v1 emitted device source lacks the contraction's "
                    f"instruction {text!r}"
                )

    def _check_grouped_realization(self, layouts) -> None:
        """The grouped program of a dataflow region against every work
        item's declared effects (ISSUES X-28).

        The emission is one SPMD body per kernel group (the walk emits the
        body of the group's work item 0; every APUC runs it).  Per declared
        work item: the launch is one `LAUNCH_KIND` launch whose group
        geometry partitions the VR, whose reduction fits the group and
        whose accumulator starts at zero (the group sums overwrite the
        accumulator register, so a carried destination is unrealizable);
        the body loads each declared input, by its role, into the register
        the declaration places it in and that its transfer ends in; the
        body's grouped MACs are one per declared operation, over exactly
        the declared group size, on those registers; the register whose
        group heads are stored back is the declared output's; the batch
        loop count is the declared one; the L4 images the body reads are
        exactly the declared ABI inputs.  A trace without shared facts (no
        declaration anywhere) is the legacy handle-only route and is not
        checked."""
        from .pim.apu_v1_grouped_route import APUCS, LAUNCH_KIND, declared_launch
        from .spmw_candidate import candidate_effects

        buckets = self._grouped_buckets
        if buckets is None:
            layout_list = layouts if isinstance(layouts, (list, tuple)) else [layouts]
            buckets = [((), [], layout, {}) for layout in layout_list]
        declared = [
            (work_id, layout, roles, candidate_effects(layout), matches)
            for work_id, matches, layout, roles in buckets
        ]
        if all(effects is None for _w, _l, _r, effects, _m in declared):
            return
        lanes = int(self.target.vr0.axes["lane"])
        bodies: dict = {}
        for work_id, *event in self.realized:
            bodies.setdefault(tuple(work_id), []).append(tuple(event))
        if not bodies:
            raise RuntimeError("APU v1 grouped realization emitted no program")

        def fail(what, work_id, layout):
            raise RuntimeError(
                f"APU v1 grouped realization of work item {tuple(work_id)} "
                f"(candidate {getattr(layout, 'mode', None)!r}) {what}"
            )

        for work_id, layout, roles, effects, matches in declared:
            if effects is None:
                fail("declares no effects", work_id, layout)
            launch = declared_launch(effects)
            if launch is None:
                fail(
                    "is one grouped MAC program; the candidate declares launches "
                    f"{[(l.kind, l.covers) for l in effects.launches]}, not one "
                    f"{LAUNCH_KIND}",
                    work_id,
                    layout,
                )
            group_size, groups, apucs, batches, _kind = launch
            # The placement's selected map must be the one the emitted group
            # tree and the host packing realize (contiguous groups, X-31).
            from .pim.apu_v1_grouped_route import grouped_geometry, grouped_map_problem

            try:
                map_problem = grouped_map_problem(
                    getattr(layout, "layout", None),
                    grouped_geometry(self.target, layout),
                )
            except ValueError as error:
                map_problem = str(error)
            if map_problem:
                fail(f"realizes contiguous groups; {map_problem}", work_id, layout)
            if group_size * groups != lanes or apucs != APUCS:
                fail(
                    f"partitions {lanes} lanes over {APUCS} APUCs; the candidate "
                    f"declares {groups} groups of {group_size} on {apucs} APUCs",
                    work_id,
                    layout,
                )
            (declared_launch_record,) = effects.launches
            reduction = declared_launch_record.reduction
            if reduction is None or int(reduction[1]) - int(reduction[0]) > group_size:
                fail(
                    f"reduces at most {group_size} lanes per group; the candidate "
                    f"declares the reduction range {reduction}",
                    work_id,
                    layout,
                )
            if declared_launch_record.accumulator != "zero":
                fail(
                    "overwrites its accumulator register with the group sums (a "
                    "zero-initialized accumulator); the candidate declares "
                    f"accumulator {declared_launch_record.accumulator!r}",
                    work_id,
                    layout,
                )
            body_id = tuple(0 for _ in work_id)
            events = bodies.get(body_id)
            if events is None:
                fail(
                    f"has no emitted SPMD body (work item {body_id} of its group "
                    f"emitted nothing; bodies: {sorted(bodies)})",
                    work_id,
                    layout,
                )
            # A grouped MAC is a full-VR multiply into the scratch register
            # followed by the group-sum tree of that scratch into the
            # accumulator; anything else in the body is uncovered.
            loads, stores, macs, others = {}, {}, [], []
            pending = None
            for event in events:
                if event[0] == "load":
                    loads[event[1]] = event[2]
                elif event[0] == "store":
                    stores[event[1]] = event[2]
                elif event[0] == "mul" and pending is None:
                    pending = event
                elif (
                    event[0] == "group_reduce"
                    and pending is not None
                    and event[2] == self._name("mac_tmp")
                ):
                    macs.append(("mac", event[1], pending[1], pending[2], event[3]))
                    pending = None
                else:
                    others.append(event)
            if pending is not None:
                others.append(pending)
            if others:
                fail(
                    f"emits {sorted({e[1] for e in others})}, which the grouped MAC "
                    "declaration does not cover",
                    work_id,
                    layout,
                )
            if len(macs) != len(declared_launch_record.operations):
                fail(
                    f"emits {len(macs)} grouped MAC(s); the candidate declares "
                    f"{len(declared_launch_record.operations)} operation(s)",
                    work_id,
                    layout,
                )
            realized_groups = sorted({int(mac[4]) for mac in macs})
            if realized_groups != [group_size]:
                fail(
                    f"reduces groups of {realized_groups} lanes; the candidate "
                    f"declares {group_size}",
                    work_id,
                    layout,
                )
            if int(self.vector_batches) != batches:
                fail(
                    f"loops over {self.vector_batches} batch(es); the candidate "
                    f"declares {batches}",
                    work_id,
                    layout,
                )
            # Independently of the declaration: the packing the runtime will
            # perform for this kernel group, re-derived from the facts, must
            # need exactly the batches the body loops over (X-30).
            if _kind == "gemm" and matches:
                from .pim.apu_v1_grouped_route import (
                    group_contractions,
                    grouped_device_packing,
                    grouped_geometry,
                )

                packing = grouped_device_packing(
                    group_contractions(matches), grouped_geometry(self.target, layout)
                )
                if packing.kind != "gemm" or int(packing.batches) != int(
                    self.vector_batches
                ):
                    fail(
                        f"loops over {self.vector_batches} batch(es); the packing the "
                        f"runtime performs for this kernel group needs "
                        f"{packing.batches if packing.kind == 'gemm' else 'no device packing'}"
                        f" ({packing.reason or 'dense GEMM'})",
                        work_id,
                        layout,
                    )
            memref_role = {memref: role for role, memref in roles.items()}
            by_value = {v.value: v for v in effects.value_layouts}
            transfers = {(t.value, t.direction): t for t in effects.transfers}
            for memref in effects.abi_inputs:
                role = memref_role.get(memref)
                register = loads.get(role)
                decl = by_value.get(memref)
                if register is None or decl is None or decl.storage != register:
                    fail(
                        f"loads {memref!r} (role {role!r}) into {register!r}; the "
                        f"candidate declares it in {decl and decl.storage!r}",
                        work_id,
                        layout,
                    )
                transfer = transfers.get((memref, "in")) or transfers.get(
                    (memref, "inout")
                )
                if transfer is None or register not in transfer.route:
                    fail(
                        f"brings {memref!r} into {register!r}; the candidate's "
                        f"transfer is {transfer and tuple(transfer.route)}",
                        work_id,
                        layout,
                    )
            accumulator = stores.get("acc")
            if accumulator is None:
                fail("stores no accumulator register back to L4", work_id, layout)
            for memref in effects.abi_outputs:
                decl = by_value.get(memref)
                if decl is None or decl.storage != accumulator:
                    fail(
                        f"writes the group heads of {accumulator!r} to L4 as the "
                        f"output; the candidate declares {memref!r} in "
                        f"{decl and decl.storage!r}",
                        work_id,
                        layout,
                    )
                transfer = transfers.get((memref, "out")) or transfers.get(
                    (memref, "inout")
                )
                if (
                    transfer is None
                    or accumulator not in transfer.route
                    or transfer.route[-1] != "host_gather"
                ):
                    fail(
                        f"returns {memref!r} from {accumulator!r} through L4 to the "
                        f"host; the candidate's transfer is "
                        f"{transfer and tuple(transfer.route)}",
                        work_id,
                        layout,
                    )
            operand_registers = {loads.get("x"), loads.get("y")}
            for _kind, acc_name, x_name, y_name, _group in macs:
                if acc_name != accumulator or {x_name, y_name} != operand_registers:
                    fail(
                        f"multiplies {x_name!r} by {y_name!r} into {acc_name!r}; the "
                        f"loads bind {sorted(loads.items())} and the store "
                        f"{accumulator!r}",
                        work_id,
                        layout,
                    )
            # The L4 images the body reads: the x and y operands (the load
            # of the accumulator's L4 image is dead under zero
            # initialization); the declared ABI inputs are exactly those.
            realized_inputs = {
                roles[r] for r in ("x", "y") if r in roles and r in loads
            }
            if set(effects.abi_inputs) != realized_inputs:
                fail(
                    f"reads {sorted(realized_inputs)} from L4; the candidate "
                    f"declares the ABI inputs {sorted(effects.abi_inputs)}",
                    work_id,
                    layout,
                )

    def runtime_route(self, commands) -> tuple[str, tuple]:
        if self.program is None:
            return super().runtime_route(commands)
        from .pim.apu_v1_record_frontend import APUv1RecordProgram

        if isinstance(self.program, APUv1RecordProgram):
            decision = self.program.candidate.decision()
            kind = (
                "apu-v1-scalar-arc-v1"
                if decision[0] == "scalar_arc_c"
                else "apu-v1-native-recipe-v1"
            )
            return kind, decision[1:]
        decision = self.program.candidate.decision()
        return "apu-v1-vector-gvml-v1", (self.program.selected_plan.name, decision[-1])

    def virtual_run_extra(self) -> dict:
        return {} if self.program is None else self.program.virtual_extra()

    def virtual_run_check(self, inputs) -> None:
        if self.program is not None:
            self.program.check_virtual_inputs(inputs)

    def _enabled(self) -> bool:
        """Emit one SPMD body; the host launches it on every APUC."""
        work_id = tuple(getattr(self, "_active_work_id", ()))
        return not work_id or all(int(value) == 0 for value in work_id)

    def _append(self, line: str) -> None:
        if self._enabled():
            self.cmds.append(line)

    def _record(self, *event) -> None:
        """Record one emitted event of the SPMD body (see ``realized``)."""
        if self._enabled():
            work_id = tuple(getattr(self, "_active_work_id", ()) or ())
            self.realized.append((work_id, *event))

    def bind_handle(self, handle, c_name: str) -> None:
        """Teach the ctx that `handle` lowers to the C identifier
        `c_name`. Called by the autoscheduler (or the test harness) once
        per placement -- e.g. `bind_handle(target.vrs, "inp_vr0")`.

        `handle` may be a Tenon handle (`Register`/`MemoryRef`/`Memory`)
        or a string label used to name scratch slots like `"mac_tmp"`.
        """
        key = handle if isinstance(handle, str) else id(handle)
        self._handle_names[key] = c_name

    def _name(self, handle, role: str = "") -> str:
        from .spmw_target import Memory

        if isinstance(handle, str):
            return self._handle_names.get(handle, f"{handle}_vr")
        cached = self._handle_names.get(id(handle))
        if cached is not None:
            return cached
        if isinstance(handle, Register):
            name = handle.name or "vr_unknown"
            if name.startswith("vr") and name[2:].isdigit():
                if name not in self._used_vr_names:
                    self._used_vr_names.append(name)
            return name
        if isinstance(handle, MemoryRef):
            mem = handle.memory
            if mem.name == "l4":
                return self._L4_PTR_BY_ROLE.get(role, "inp_L4ptr")
            if mem.name == "l1":
                return f"GVML_VM_{handle.idx!r}"
            return f"{mem.name}_ref"
        if isinstance(handle, Memory):
            if handle.name == "l4":
                return self._L4_PTR_BY_ROLE.get(role, "inp_L4ptr")
            if handle.name == "l1":
                return "GVML_VM_0"
            return f"{handle.name}_ref"
        raise NotImplementedError(
            f"APUv1Ctx: unknown handle type {type(handle).__name__}."
        )

    def cmd(self, name: str, dst=None, src0=None, src1=None, **kwargs):
        """Emit one GVML call as a C line."""
        operands = []
        if dst is not None:
            operands.append(self._name(dst, "dst"))
        if src0 is not None:
            operands.append(self._name(src0, "src0"))
        if src1 is not None:
            operands.append(self._name(src1, "src1"))
        for k, v in kwargs.items():
            operands.append(f"/* {k}= */ {v!r}")
        self._record("op", name, tuple(operands))
        self._append(name + "(" + ", ".join(operands) + ");")

    @staticmethod
    def _group_enum(group_size: int) -> str:
        if group_size <= 0 or group_size > 32768 or group_size & (group_size - 1):
            raise ValueError(f"invalid GVML group size {group_size}")
        suffix = f"{group_size // 1024}K" if group_size >= 1024 else str(group_size)
        return f"GVML_P2_{suffix}"

    def emit_l4_to_vr(self, role: str, vr, vm_index: int) -> None:
        pointer = self._L4_PTR_BY_ROLE[role]
        if self.vector_batches > 1:
            pointer = f"({pointer} + batch * 32768)"
        vr_name = self._name(vr, role)
        self._record("load", role, vr_name, int(vm_index))
        self._append(f"direct_dma_l4_to_l1_32k(GVML_VM_{vm_index}, {pointer});")
        self._append(f"gvml_load_16({vr_name}, GVML_VM_{vm_index});")

    def emit_vr_to_l4(self, role: str, vr, vm_index: int) -> None:
        pointer = self._L4_PTR_BY_ROLE[role]
        if self.vector_batches > 1:
            pointer = f"({pointer} + batch * 32768)"
        vr_name = self._name(vr, role)
        self._record("store", role, vr_name, int(vm_index))
        self._append(f"gvml_store_16(GVML_VM_{vm_index}, {vr_name});")
        self._append(f"direct_dma_l1_to_l4_32k({pointer}, GVML_VM_{vm_index});")

    def emit_group_reduce_f16(self, dst, src) -> None:
        dst_name = self._name(dst, "acc")
        src_name = self._name(src, "x")
        group = self._group_enum(self.group_size)
        # Recorded here, from the group enum actually emitted.
        self._record("group_reduce", dst_name, src_name, int(self.group_size))
        self._append(
            "gvml_add_subgrps_f16_grp("
            f"{dst_name}, {src_name}, {group}, GVML_P2_1, 0, "
            "GVML_VM_3, reduce_tmp_vr);"
        )

    def emit_grouped_f16_mac(self, acc, x, y) -> None:
        x_name, y_name = self._name(x, "x"), self._name(y, "y")
        self._record("mul", x_name, y_name)
        self._append(f"gvml_mul_f16(mac_tmp_vr, {x_name}, {y_name});")
        self.emit_group_reduce_f16(acc, "mac_tmp")

    def append(self, line: str) -> None:
        """Low-level escape hatch -- append a raw C source line."""
        self._append(line)

    def iter_vr_aliases(self) -> list[tuple[str, str]]:
        """Return [(c_name, gvml_vr_enum), ...] in bind order.

        Each entry becomes one line `enum gvml_vr16 <c_name> = <enum>;`
        in the emitted device.c. Until the regalloc spec rebinds VRs
        per live-range, the autoscheduler's bind list may be empty; in
        that case the build harness substitutes a canonical default
        (vrs + mac_tmp_vr) so the emitted body still compiles.
        """
        out: list[tuple[str, str]] = []
        used_indices = set()
        for name in self._used_vr_names:
            index = int(name[2:])
            used_indices.add(index)
            out.append((name, f"GVML_VR16_{index}"))
        for temporary in ("mac_tmp_vr", "reduce_tmp_vr"):
            index = next(i for i in range(15) if i not in used_indices)
            used_indices.add(index)
            out.append((temporary, f"GVML_VR16_{index}"))
        return out

    def iter_l4_roles(self) -> list[str]:
        """Return the L4 pointer C names referenced by `self.cmds`, in
        canonical role-table order (inp, wgt, out). Used by the build
        harness to emit one `gal_mem_handle_to_apu_ptr` decl per
        actual reference (avoiding unused-variable warnings)."""
        names: set[str] = set()
        for line in self.cmds:
            for ptr in self._L4_PTR_BY_ROLE.values():
                if ptr in line:
                    names.add(ptr)
        role_order = ["inp_L4ptr", "wgt_L4ptr", "out_L4ptr"]
        return [n for n in role_order if n in names]

    def resolve_moves(self, role, src_handle=None, dst_handle=None):
        if dst_handle is None:
            return (None, None)
        if isinstance(dst_handle, Register):
            role_name = {"x": "X", "y": "Y", "acc": "ACC", "dst": "ACC"}.get(role)
            if role_name is None:
                return (None, None)
            store = "ST_ACC_VR_TO_L4" if role_name == "ACC" else None
            return (f"LD_{role_name}_L4_TO_VR", store)
        return (None, None)

    def resolve_spill_moves(self, tier, home_handle, *, n_entries=1):
        return super().resolve_spill_moves(tier, home_handle, n_entries=n_entries)


def _apu_v1_kernel_src(compiled: "Compiled") -> str:
    return "\n".join(str(c) for c in compiled.cmds if isinstance(c, str))


_APU_V1_MATCHER_APUCS = 4


def _parse_apu_v1_prof_print(text: str) -> int:
    """Return the four-APUC parallel makespan from the ``total`` records in
    ``text``, which must hold exactly the records this launch appended (one
    per APUC; see :func:`allo.pim.apu_v1_program.fresh_profile_records`).
    The APUCs run concurrently, so the cost is their maximum CRUN.
    """
    from .pim.apu_v1_program import parse_fresh_profile

    return max(parse_fresh_profile("", text, _APU_V1_MATCHER_APUCS))


_FLOAT_MEMREF_DTYPES = {"f16": "float16", "f32": "float32", "f64": "float64"}


def _memref_shape_dtype(memref_type):
    match = re.fullmatch(r"memref<((?:\d+x)*)([a-z]\w*)>", str(memref_type or ""))
    if match is None:
        return None
    dims = tuple(int(item) for item in match.group(1).split("x") if item)
    return dims, match.group(2)


def _apu_v1_output_specs(compiled: "Compiled", inputs: dict) -> dict:
    """Derive `output_specs: {role: (shape, dtype)}` from the compiled ABI.

    This is the fallback for non-grouped operations. Grouped contractions use
    `_apu_v1_prepare_io`, which knows the logical output shape and FP16 ABI.
    Each output role is a matched result memref. Its shape and dtype come
    from the caller's in-place array of that name or, for a float memref,
    from the retained memref type; anything else raises rather than guessing.
    """
    import numpy as np

    out: dict = {}
    trace = getattr(compiled, "trace", None)
    if trace is None:
        raise NotImplementedError(
            "APU v1 matcher route: compiled artifact has no match trace, so its "
            "output ABI cannot be derived"
        )

    def _strip(name):
        return name[len("local_") :] if name and name.startswith("local_") else name

    memref_types = {}
    for match in trace.matches:
        for operand in match.operands:
            name = _strip(operand.memref_name)
            if name and operand.memref_type:
                memref_types.setdefault(name, operand.memref_type)

    for m in trace.matches:
        name = _strip(m.result_memref_name)
        if not name or name in out:
            continue
        if name in inputs and hasattr(inputs[name], "shape"):
            out[name] = (tuple(inputs[name].shape), inputs[name].dtype)
            continue
        parsed = _memref_shape_dtype(memref_types.get(name))
        if parsed is not None and parsed[1] in _FLOAT_MEMREF_DTYPES:
            out[name] = (parsed[0], np.dtype(_FLOAT_MEMREF_DTYPES[parsed[1]]))
            continue
        raise NotImplementedError(
            f"APU v1 matcher route cannot derive the shape/dtype of output "
            f"{name!r} from the compiled ABI (memref type "
            f"{memref_types.get(name)!r}); pass it as an in-place array argument"
        )
    if not out:
        raise NotImplementedError(
            "APU v1 matcher route: the match trace names no output memref"
        )
    return out


def _apu_v1_prepare_io(compiled: "Compiled", inputs: dict):
    """Normalize inputs and realize the grouped-contraction callable ABI.

    Dense contractions are packed one output dot product per GVML group, split
    over four APUCs, and streamed in 32K-lane batches. The returned metadata
    drives group-head gathering after execution.
    """
    import numpy as np

    normalized: dict = {}
    for name, arr in inputs.items():
        if not isinstance(arr, np.ndarray):
            raise ValueError(
                f"APU v1 run: input {name!r} must be a numpy.ndarray; "
                f"got {type(arr).__name__}"
            )
        # APU v1 is uint16-native today; we view int16/float16 buffers
        # as their raw bytes (host.c writes them straight into L4) so
        # we don't impose a dtype cast here.
        normalized[name] = arr

    layout = getattr(compiled, "layout", None)
    if isinstance(layout, list):
        layout = layout[0] if layout else None
    from .spmw_candidate import candidate_effects

    effects = candidate_effects(layout)
    if effects is not None:
        # A candidate of the shared contract: the packing and the ABI
        # binding come from the retained facts and the declaration.
        return _apu_v1_prepare_grouped_io(compiled, layout, effects, normalized)
    extra = getattr(layout, "extra", {}) or {}
    group_size = int(extra.get("group_size", 0) or 0)
    batches = max(1, int(extra.get("n_out_tiles", 1)))
    if group_size and len(normalized) >= 2:
        trace_output_names = []
        operand_names = []
        for match in compiled.trace.matches:
            name = match.result_memref_name
            if name:
                name = name[len("local_") :] if name.startswith("local_") else name
                if name not in trace_output_names:
                    trace_output_names.append(name)
            for operand in match.operands:
                if operand.role not in ("x", "y") or not operand.memref_name:
                    continue
                operand_name = operand.memref_name
                if operand_name.startswith("local_"):
                    operand_name = operand_name[len("local_") :]
                if operand_name not in operand_names:
                    operand_names.append(operand_name)
        # Reduction matchers name the loop-carried scalar ``acc`` as the
        # result.  At the callable boundary, the real output is the source
        # argument not bound to semantic x/y (C in GEMM, y in GEMV).
        abi_output_names = [name for name in normalized if name not in operand_names]
        operands = [
            (name, arr)
            for name, arr in normalized.items()
            if name in operand_names and arr.ndim > 0
        ]
        matrices = [(name, arr) for name, arr in operands if arr.ndim == 2]
        if len(matrices) >= 2:
            # Grouped GEMM: one group is one output dot product.  Rows are
            # partitioned over four APUCs; each core may stream several VRs.
            (lhs_name, lhs), (rhs_name, rhs) = matrices[:2]
            if lhs.shape[1] == rhs.shape[0]:
                if lhs.dtype != np.float16 or rhs.dtype != np.float16:
                    raise ValueError(
                        "APU v1 grouped contractions require float16 inputs; "
                        f"got {lhs.dtype} and {rhs.dtype}"
                    )
                rows, reduction = lhs.shape
                cols = rhs.shape[1]
                if reduction > group_size:
                    raise ValueError("APU v1 group is smaller than GEMM reduction")
                groups_per_vr = 32768 // group_size
                rows_per_core = (rows + 3) // 4
                required = (rows_per_core * cols + groups_per_vr - 1) // groups_per_vr
                batches = max(batches, required)
                packed_lhs = np.zeros((4, batches, 32768), dtype=lhs.dtype)
                packed_rhs = np.zeros((4, batches, 32768), dtype=rhs.dtype)
                for core in range(4):
                    row0 = core * rows_per_core
                    pairs = [
                        (row, col)
                        for row in range(row0, min(rows, row0 + rows_per_core))
                        for col in range(cols)
                    ]
                    for output_index, (row, col) in enumerate(pairs):
                        batch, group = divmod(output_index, groups_per_vr)
                        base = group * group_size
                        packed_lhs[core, batch, base : base + reduction] = lhs[row, :]
                        packed_rhs[core, batch, base : base + reduction] = rhs[:, col]
                normalized = {
                    lhs_name: packed_lhs.reshape(-1),
                    rhs_name: packed_rhs.reshape(-1),
                }
                output_name = (
                    abi_output_names[0]
                    if abi_output_names
                    else trace_output_names[0] if trace_output_names else "output"
                )
                output_specs = {output_name: ((4, batches, 32768), np.dtype(lhs.dtype))}
                return (
                    normalized,
                    output_specs,
                    {
                        "kind": "gemm",
                        "output": output_name,
                        "logical_shape": (rows, cols),
                        "rows_per_core": rows_per_core,
                        "group_size": group_size,
                        "groups_per_vr": groups_per_vr,
                    },
                )

        raise NotImplementedError(
            "APU v1 physical execution currently supports a dense FP16 matrix "
            "contraction with lhs.shape[1] == rhs.shape[0]"
        )

    output_specs = _apu_v1_output_specs(compiled, normalized)
    return normalized, output_specs, None


def _apu_v1_prepare_grouped_io(compiled: "Compiled", layout, effects, inputs: dict):
    """Pack the caller's arrays for the declared grouped program.

    The packing (`grouped_device_packing`) and the binding of the work
    body's buffers to the caller's parameters (`grouped_runtime_binding`,
    by retained identity) are re-derived from the retained facts and
    compared with the declaration the artifact carries: the group geometry
    and the batch count the device program loops over must be the
    declared ones, and a program without a device packing is refused with
    the reason (``backend="virtual"`` gives its estimate).
    """
    import numpy as np

    from .pim.apu_v1_grouped_route import (
        APUCS,
        contraction_facts,
        declared_launch,
        grouped_device_packing,
        grouped_geometry,
        grouped_runtime_binding,
    )

    launch = declared_launch(effects)
    if launch is None:
        raise RuntimeError(
            "APU v1 grouped runtime: the selected candidate declares no grouped launch"
        )
    group_size, groups_per_vr, apucs, batches, declared_kind = launch
    facts = contraction_facts(getattr(compiled.trace, "matches", ()))
    if facts is None:
        raise RuntimeError(
            "APU v1 grouped runtime: the artifact declares effects but its trace "
            "carries no shared facts"
        )
    geometry = grouped_geometry(compiled.target, layout)
    if (geometry.group_size, geometry.groups_per_vr) != (group_size, groups_per_vr):
        raise RuntimeError(
            f"APU v1 grouped runtime: the placement partitions the VR into "
            f"{geometry.groups_per_vr} groups of {geometry.group_size}; the candidate "
            f"declares {groups_per_vr} of {group_size}"
        )
    from .pim.apu_v1_grouped_route import group_contractions

    packing = grouped_device_packing(
        group_contractions(compiled.trace.matches), geometry
    )
    if packing.kind != "gemm" or declared_kind != "gemm":
        raise NotImplementedError(
            "APU v1 grouped matcher program: device execution is not implemented "
            f"for this program: {packing.reason or declared_kind}; backend='virtual' "
            "gives its estimate"
        )
    if int(packing.batches) != int(batches):
        raise RuntimeError(
            f"APU v1 grouped runtime: the device program loops over {batches} "
            f"batch(es) but the packing needs {packing.batches}"
        )
    parameters = getattr(compiled, "source_parameters", None)
    binding = grouped_runtime_binding(compiled.trace, parameters) if parameters else {}

    def caller_name(memref):
        name = binding.get(memref, memref)
        if name not in inputs:
            raise ValueError(
                f"APU v1 run: no argument for the work body's {memref!r} (caller "
                f"parameter {name!r}); got {sorted(inputs)}"
            )
        return name

    lhs_name = caller_name(packing.lhs)
    rhs_name = caller_name(packing.rhs)
    output_name = caller_name(packing.output)
    lhs, rhs = inputs[lhs_name], inputs[rhs_name]
    rows, columns, reduction = packing.rows, packing.columns, packing.reduction
    if tuple(lhs.shape) != (rows, reduction) or tuple(rhs.shape) != (
        reduction,
        columns,
    ):
        raise ValueError(
            f"APU v1 run: {lhs_name!r} is {tuple(lhs.shape)} and {rhs_name!r} is "
            f"{tuple(rhs.shape)}; the program declares [{rows}, {reduction}] x "
            f"[{reduction}, {columns}]"
        )
    if lhs.dtype != np.float16 or rhs.dtype != np.float16:
        raise ValueError(
            "APU v1 grouped contractions require float16 inputs; "
            f"got {lhs.dtype} and {rhs.dtype}"
        )
    lanes = geometry.lanes
    rows_per_core = int(packing.rows_per_apuc)
    packed_lhs = np.zeros((APUCS, batches, lanes), dtype=lhs.dtype)
    packed_rhs = np.zeros((APUCS, batches, lanes), dtype=rhs.dtype)
    for core in range(APUCS):
        row0 = core * rows_per_core
        pairs = [
            (row, col)
            for row in range(row0, min(rows, row0 + rows_per_core))
            for col in range(columns)
        ]
        for output_index, (row, col) in enumerate(pairs):
            batch, group = divmod(output_index, groups_per_vr)
            base = group * group_size
            packed_lhs[core, batch, base : base + reduction] = lhs[row, :]
            packed_rhs[core, batch, base : base + reduction] = rhs[:, col]
    normalized = {lhs_name: packed_lhs.reshape(-1), rhs_name: packed_rhs.reshape(-1)}
    output_specs = {output_name: ((APUCS, batches, lanes), np.dtype(lhs.dtype))}
    return (
        normalized,
        output_specs,
        {
            "kind": "gemm",
            "output": output_name,
            "logical_shape": (rows, columns),
            "rows_per_core": rows_per_core,
            "group_size": group_size,
            "groups_per_vr": groups_per_vr,
        },
    )


def _run_apu_v1(compiled: "Compiled", **inputs) -> RunResult:
    """Run a compiled APU v1 artifact.

    A vector-plan program (`APUv1Ctx.program`) runs through its own
    realization: the GVML project on the board (``backend=None``) or the
    host-side functional model (``backend="functional"``).  The grouped
    matcher program of a dataflow region builds and launches the low-mode
    project below.
    """
    program = getattr(getattr(compiled, "layout_ctx", None), "program", None)
    if program is not None:
        return program.run(compiled, inputs)
    if compiled.backend not in (None, "device"):
        raise ValueError(
            "the APU v1 grouped matcher program supports backend=None (board) or "
            f"'virtual'; got {compiled.backend!r}"
        )
    return _run_apu_v1_grouped(compiled, **inputs)


def _run_apu_v1_grouped(compiled: "Compiled", **inputs) -> RunResult:
    """Run a compiled grouped APU v1 artifact against real GSI hardware.

    Materializes a build directory via `gen_apu_v1_low_mode_project`,
    runs `make` with the ARC GNU toolchain, executes the resulting
    binary against the Gemini PCI device, and parses cycle counts from
    its PROF_PRINT stdout. Raises `SimulatorUnavailable` when an
    environment gate (toolchain, template dir, PCI sysfs node, GVML SDK
    headers) is missing. Build failures past the gate raise
    `RuntimeError` so tests cannot silently PASS. The launch holds the
    APU v1 board lock and drains the device log before and after it; exactly
    one fresh `total` record per APUC is required, and a missing, stale, or
    ambiguous profile raises. Missing or short output files raise with the
    role name.
    """
    kernel_src = _apu_v1_kernel_src(compiled)

    from .pim.apu_v1_program import _ledag_available_reason

    skip_reason = _apu_v1_unavailable_reason() or _ledag_available_reason()
    if skip_reason:
        raise SimulatorUnavailable("apu_v1", skip_reason)

    from .spmw_apu_v1_build import (
        _apu_v1_build_config,
        _assert_gvml_sdk_present,
        gen_apu_v1_low_mode_project,
    )

    # Build-harness probe: a missing SDK here means the toolchain gate
    # passed but headers are absent; fail fast with a readable error
    # rather than letting `make` emit a 200-line stderr blob below.
    _assert_gvml_sdk_present()
    build_config = _apu_v1_build_config()

    tmpdir = tempfile.mkdtemp(prefix="tenon-apu-v1-")
    try:
        # prepare_io is build-harness Python; ValueError here is a Tenon
        # bug or user-input contract violation, not an env skip.
        inputs_np, output_specs, packing = _apu_v1_prepare_io(compiled, inputs)

        # Write each input to <tmpdir>/in_<role>.bin so host.c can fread it.
        input_bin_paths: dict[str, str] = {}
        for role, arr in inputs_np.items():
            p = Path(tmpdir) / f"in_{role}.bin"
            arr.tofile(str(p))
            input_bin_paths[role] = str(p)

        output_bin_paths: dict[str, str] = {
            role: str(Path(tmpdir) / f"out_{role}.bin") for role in output_specs
        }

        # Project emission. The toolchain/template gate above already
        # ensured the template dir exists; if gen_apu_v1_low_mode_project
        # still raises FileNotFoundError, that's a Tenon bug, not an
        # env skip -- let it propagate.
        project_dir = gen_apu_v1_low_mode_project(
            dst_dir=Path(tmpdir) / "project",
            compiled=compiled,
            inputs=inputs_np,
            output_specs=output_specs,
            lab_name="tenon-kernel",
        )

        # Build. Past the toolchain/SDK gate, every failure below is a
        # build- or runtime-bug, not an environment skip -- raise so
        # the test does not silently PASS on a cycles=None RunResult.
        try:
            mk = subprocess.run(
                build_config.make_command,
                cwd=str(project_dir),
                capture_output=True,
                timeout=600,
                check=False,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            raise RuntimeError(
                f"APU v1 make invocation failed: {exc}\n"
                f"--- project_dir: {project_dir}"
            ) from exc
        if mk.returncode != 0:
            raise RuntimeError(
                "APU v1 make failed:\n"
                + mk.stderr.decode("utf-8", errors="replace")
                + "\n--- stdout ---\n"
                + mk.stdout.decode("utf-8", errors="replace")
                + f"\n--- project_dir: {project_dir}"
            )

        bin_path = build_config.binary_path(project_dir, "tenon-kernel")
        if not bin_path.exists():
            raise RuntimeError(
                f"APU v1 build succeeded but binary not found at {bin_path} "
                f"(project_dir: {project_dir})"
            )

        # Build argv: inputs first then outputs, in the same role order
        # the build harness wrote into struct.h (sorted alpha).
        sorted_in = sorted(input_bin_paths.keys())
        sorted_out = sorted(output_bin_paths.keys())
        argv = [str(bin_path)]
        argv += [input_bin_paths[r] for r in sorted_in]
        argv += [output_bin_paths[r] for r in sorted_out]

        import numpy as np

        import hashlib

        from .pim.apu_v1_program import (
            apu_v1_lock_path,
            launch_on_board,
            profile_counters,
            read_output_image,
        )

        binary_sha256 = hashlib.sha256(Path(bin_path).read_bytes()).hexdigest()
        # The board lock is held across launch and the pre/post device-log
        # drains, so the parsed totals belong to this launch only.
        binary_output, ledag_text, apuc_cycles = launch_on_board(
            argv,
            str(project_dir),
            expected_records=_APU_V1_MATCHER_APUCS,
            timeout=300,
            attempts=1,
        )
        combined = binary_output + ("\n" + ledag_text if ledag_text else "")
        cycles = max(apuc_cycles)

        outputs: dict = {}
        for role, (shape, dtype) in output_specs.items():
            outputs[role] = read_output_image(
                output_bin_paths[role], role, dtype, shape
            )

        if packing and packing["kind"] == "gemm":
            packed = outputs[packing["output"]]
            rows, cols = packing["logical_shape"]
            rows_per_core = packing["rows_per_core"]
            group_size = packing["group_size"]
            groups_per_vr = packing["groups_per_vr"]
            logical = np.zeros((rows, cols), dtype=packed.dtype)
            for core in range(4):
                row0 = core * rows_per_core
                pairs = [
                    (row, col)
                    for row in range(row0, min(rows, row0 + rows_per_core))
                    for col in range(cols)
                ]
                for output_index, (row, col) in enumerate(pairs):
                    batch, group = divmod(output_index, groups_per_vr)
                    logical[row, col] = packed[core, batch, group * group_size]
            outputs[packing["output"]] = logical

        return RunResult(
            kind="device",
            cycles=cycles,
            stdout=combined,
            backend="apu_v1",
            extra={
                "outputs": outputs,
                "kernel_src": kernel_src,
                "build_mode": build_config.mode,
                "project_dir": str(project_dir),
                "returncode": 0,
                "board_lock": str(apu_v1_lock_path()),
                "profile_records": _APU_V1_MATCHER_APUCS,
                "binary_sha256": binary_sha256,
                "binary_bytes": Path(bin_path).stat().st_size,
                "profile_counters": profile_counters(ledag_text),
                "profile_text": ledag_text,
                "apuc_cycles": list(apuc_cycles),
            },
        )
    finally:
        if os.environ.get("TENON_APU_V1_KEEP_TMP") != "1":
            shutil.rmtree(tmpdir, ignore_errors=True)
