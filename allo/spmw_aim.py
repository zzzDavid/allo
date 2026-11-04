# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SK hynix AiM codegen context and ramulator2 runtime (moved from spmw_codegen)."""

from __future__ import annotations

import functools

import math
import re
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

from .spmw_autoschedule import _matcher_work_scope
from .spmw_plan import LaunchStep
from .spmw_match import MatchedOp
from .spmw_codegen import (
    CodegenContext,
    _parse_loop_bound,
    RunResult,
    SimulatorUnavailable,
    Compiled,
)
from .spmw_simenv import aim_root as _aim_root
from .spmw_simenv import aim_sim_image as _aim_sim_image
from .spmw_simenv import aim_unavailable_reason as _aim_unavailable_reason
from .pim.aim_lowering import (
    AimAllBankWrite,
    AimBankCopy,
    AimContraction,
    AimDistributedHostTransfer,
    AimElementwise,
    AimHostReduction,
    AimHostTransfer,
    AimSync,
    _AimLowerer,
    _memref_shape,
    _target_geometry,
    _work_items,
    build_contraction,
    check_contraction_coverage,
    contraction_shape,
    group_replicated,
    issued_scalar_macs,
    layout_reduction_partitions,
    mac_operands,
    matrix_axes,
    resolve_batch_mapping,
)


def _aim_storage_name(handle) -> str | None:
    home = getattr(handle, "home_handle", handle)
    memory = getattr(home, "memory", None)
    if memory is not None:
        return getattr(memory, "name", None)
    return getattr(home, "name", None)


def _layout_image(images, value, side):
    """The layout of ``value``'s ``side`` (``in``/``out``) image from a
    value -> image -> layout map: the side's own image when declared, else
    the single image."""
    candidates = images.get(value, {})
    return candidates.get(side) or candidates.get("value")


def _aim_axes(access, labels) -> tuple:
    axes = []
    for coord in getattr(access, "coords", None) or ():
        if coord is None:
            continue
        for atom, _coefficient in coord[0]:
            label = labels.get(atom)
            if label is not None and label not in axes:
                axes.append(label)
    return tuple(axes)


def aim_candidate_effects(target, matches, placement):
    """`CandidateEffects` of one AiM placement for one work item, derived from
    the shared semantic facts of its matches (`Contraction`/`Epilogue`: loop
    labels, trips, access coordinates, matrix and vector roles, accumulator
    initialization, sinks) and from the placement (handles and knobs).

    Backing storage: the matrix in the ``banks``; the vector in the global
    buffer ``gb`` (broadcast) or, when it shares the matrix's buffer, in the
    banks; results in the host-visible ``gpr`` the MAC results are read back
    into; elementwise MUL operands in the banks and ADD operands in the GPR.
    The accumulator register file ``mac_reg`` is a transient allocation.
    The one launch of a contraction advances the work item's result
    instances over the whole reduction; its ``covers`` name the placement
    and the knobs the lowering realizes (bank scope, reduction partitions,
    batch mapping), which `AimCtx.check_realization` compares with the
    lowered records.  None when the matches carry no facts."""
    from .spmw_candidate import (
        CandidateCoverage,
        CandidateEffects,
        CandidateLaunch,
        CandidateTransfer,
        CandidateValueLayout,
    )

    del target
    facts = [m.extra.get("semantic_facts") for m in matches]
    if not matches or any(f is None for f in facts):
        return None
    extra = placement.extra or {}
    ops = {m.target_op_name for m in matches}
    labels = {}
    for fact in facts:
        labels.update(fact.loop_labels)
    inputs, outputs, layouts, transfers = [], [], {}, {}
    instances = reduction = None
    init = None
    contraction = "MAC" in ops
    for match, fact in zip(matches, facts):
        if hasattr(fact, "reduction_vars"):
            trips = fact.trips
            extent = 1
            for var in fact.result_vars:
                extent *= int(trips.get(var) or 0)
            red = 1
            for var in fact.reduction_vars:
                red *= int(trips.get(var) or 0)
            instances = extent if instances is None else instances
            reduction = red if reduction is None else reduction
            init = fact.accumulator_init
            matrix = fact.matrix.memref_name if fact.matrix is not None else None
            vector = fact.vector.memref_name if fact.vector is not None else None
            for access in fact.inputs:
                memref = access.memref_name
                if memref is None or memref in inputs:
                    continue
                inputs.append(memref)
                handle = placement.placements.get(memref)
                storage = _aim_storage_name(handle) or (
                    "banks" if memref == matrix else "gb"
                )
                if memref == vector and vector != matrix and storage != "gb":
                    storage = "banks"
                layouts[memref] = CandidateValueLayout(
                    memref, _aim_axes(access, labels), storage
                )
                transfers[memref] = CandidateTransfer(
                    memref,
                    "in",
                    (
                        ("host_broadcast", "gb")
                        if storage == "gb"
                        else ("host_scatter", "banks")
                    ),
                )
            result_accesses = []
            if not fact.accumulator.local:
                result_accesses.append((fact.accumulator.memref_name, fact.accumulator))
            for sink in fact.external_sinks:
                result_accesses.append((sink.memref_name, sink))
        else:
            extent = 1
            for loop in fact.loops:
                extent *= int(loop.trip or 0)
            instances = extent if instances is None else instances
            home = "gpr" if match.target_op_name == "ADD" else "banks"
            for access in fact.operands:
                memref = access.memref_name
                if memref is None or access.local or memref in inputs:
                    continue
                inputs.append(memref)
                storage = _aim_storage_name(placement.placements.get(memref)) or home
                layouts[memref] = CandidateValueLayout(
                    memref, _aim_axes(access, labels), storage
                )
                transfers[memref] = CandidateTransfer(
                    memref, "in", ("host_scatter", storage)
                )
            result_accesses = (
                [] if fact.result.local else [(fact.result.memref_name, fact.result)]
            )
            for memref, _text in match.extra.get("output_sinks", ()):
                if memref != fact.result.memref_name:
                    result_accesses.append((memref, fact.result))
        for memref, access in result_accesses:
            if memref is None:
                continue
            if memref not in outputs:
                outputs.append(memref)
            carried = contraction and init == "carried"
            if carried and memref not in inputs:
                inputs.append(memref)
            storage = "gpr" if contraction or match.target_op_name == "ADD" else "banks"
            layouts.setdefault(
                (memref, "out") if memref in inputs else memref,
                CandidateValueLayout(
                    memref,
                    _aim_axes(access, labels),
                    storage,
                    image="out" if memref in inputs else "value",
                ),
            )
            transfers[memref] = CandidateTransfer(
                memref,
                "inout" if memref in inputs else "out",
                (
                    ("host_scatter", storage, "host_gather")
                    if memref in inputs
                    else (storage, "host_gather")
                ),
            )
    if contraction:
        kind = str(extra.get("operation_name") or "MAC")
        covers = (
            placement.mode,
            str(extra.get("bank_scope") or ("abk" if kind == "MAC_ABK" else "sbk")),
            int(extra.get("reduction_partitions", 1) or 1),
            str(extra.get("batch_mapping") or "none"),
        )
        allocations = (("accumulator", 0, "mac_reg"),)
    else:
        kind = "MUL" if "MUL" in ops else "ADD"
        covers = (placement.mode,)
        allocations = ()
    launch = CandidateLaunch(
        index=0,
        kind=kind,
        operations=tuple(range(len(matches))),
        instances=(0, int(instances or 0)),
        reduction=(0, int(reduction or 0)) if contraction else None,
        accumulator=init if contraction else None,
        covers=covers,
    )
    return CandidateEffects(
        coverage=CandidateCoverage(
            matches[0].func_name, tuple(sorted(ops)), tuple(inputs), tuple(outputs)
        ),
        value_layouts=tuple(layouts.values()),
        transfers=tuple(transfers.values()),
        launches=(launch,),
        abi_inputs=tuple(inputs),
        abi_outputs=tuple(outputs),
        device_scope="one channel-masked command stream, every channel of the device",
        allocations=allocations,
    )


def _derive_aim_effects(view, matches, placement):
    """Late-binding registry entry (see `aim_candidate_effects`)."""
    return aim_candidate_effects(view, matches, placement)


def _register_effects_deriver():
    from .spmw_candidate import EFFECTS_DERIVERS

    EFFECTS_DERIVERS["aim"] = _derive_aim_effects


_register_effects_deriver()


_HOST_TRANSFER_RECORDS = (AimDistributedHostTransfer, AimHostTransfer, AimAllBankWrite)


def _check_host_transfer_extent(op, group, geometry, shape, value, index, expected):
    """One distributed transfer or all-bank broadcast against the element
    extent of the slice it was lowered for."""
    if isinstance(op, AimDistributedHostTransfer):
        replicas = op.replicas if group.split_replicas else 1
        realized = op.elements * replicas
        if not expected <= realized < expected + replicas:
            raise RuntimeError(
                f"AiM realization moves {realized} element(s) of {value!r}; the "
                f"slice {index} holds {expected}"
            )
        return
    if shape is None:
        raise RuntimeError(
            f"AiM realization broadcasts {value!r} into a group without a "
            "contraction"
        )
    parts = group.reduction_partitions
    accumulators = shape["outputs"] * shape["batches"]
    if (
        expected != accumulators
        or op.rows != _ceil(shape["outputs"], geometry.banks)
        or op.copies != _ceil(shape["batches"] * parts, group.partitions)
    ):
        raise RuntimeError(
            f"AiM realization broadcasts {value!r} to {op.rows} row(s) x "
            f"{op.copies} copies; the slice {index} feeds {accumulators} "
            f"accumulator(s) ({expected} elements)"
        )


def _check_host_transfers(program, group, effects, ops, storage_of) -> None:
    """The host transfers lowered for ``group`` against its candidate's
    declared transfers, bound by retained buffer identity.

    Every host record the derivation lowered for the group carries the
    buffer keys it moves (`buffer_refs`), their slices and the group; each is
    matched to the declared transfer of the work-body value with the same
    retained identity (the body's operands and results, never a name), and
    its direction (host write: ``in``/``inout``; host read: ``out``/``inout``),
    the extent of the slice it moves and its device endpoint (the banks the
    group's layout holds the buffer in) are compared with the declaration.
    Under a host program every declared input must reach the device: a host
    write of it, residency, or a store by another group of the program.
    """
    records = [
        op
        for op in ops
        if isinstance(op, _HOST_TRANSFER_RECORDS) and op.group_id == group.group_id
    ]
    if not records and program.schedule is None:
        return
    ref_of_name: dict = {}
    for match in group.matches:
        for operand in match.operands:
            if operand.memref_name is not None:
                ref_of_name.setdefault(
                    operand.memref_name,
                    program._key(operand.value_ref, operand.memref_name),
                )
        if match.result_memref_name is not None:
            ref_of_name.setdefault(
                match.result_memref_name,
                program._key(match.result_value_ref, match.result_memref_name),
            )
    declared_by_ref: dict = {}
    for transfer in effects.transfers:
        ref = ref_of_name.get(transfer.value)
        if ref is not None:
            declared_by_ref.setdefault(ref, transfer)
    label = program.name_by_key.get
    geometry = program.geometry
    shape = group.contractions[0]["shape"] if group.contractions else None
    written: set = set()
    bursts: dict = {}
    for op in records:
        if not op.buffer_refs or len(op.buffer_refs) != len(op.buffer_slices):
            raise RuntimeError(
                f"AiM realization lowered the host transfer {op.name!r} without a "
                "retained buffer identity"
            )
        direction = "write" if isinstance(op, AimAllBankWrite) else op.direction
        for ref, index in zip(op.buffer_refs, op.buffer_slices):
            transfer = declared_by_ref.get(ref)
            if transfer is None:
                raise RuntimeError(
                    f"AiM realization moves {label(ref, ref)!r} between host and "
                    f"the device for group {group.name!r}; the candidate declares "
                    f"transfers of {sorted(t.value for t in effects.transfers)}"
                )
            allowed = {"in", "inout"} if direction == "write" else {"out", "inout"}
            if transfer.direction not in allowed:
                raise RuntimeError(
                    f"AiM realization {direction}s {label(ref, ref)!r} "
                    f"({transfer.value!r}); the candidate declares it {transfer.direction!r}"
                )
            side = "in" if direction == "write" else "out"
            landing = storage_of(transfer.value, side)
            if (
                landing is None
                or landing.storage != "banks"
                or "banks" not in tuple(transfer.route)
            ):
                raise RuntimeError(
                    f"AiM realization moves {transfer.value!r} through the banks; "
                    f"the candidate declared {landing} with route {transfer.route}"
                )
            expected = program._token_extent(ref, index)
            if isinstance(op, AimHostTransfer):
                bursts.setdefault((ref, index, transfer.value), [0, expected])
                bursts[(ref, index, transfer.value)][0] += op.bursts
            else:
                _check_host_transfer_extent(
                    op, group, geometry, shape, transfer.value, index, expected
                )
            if direction == "write":
                written.add(ref)
    for (ref, index, value), (total, expected) in bursts.items():
        per_replica = total * geometry.lanes // max(group.replicas, 1)
        batches = shape["batches"] if shape else 1
        if not expected <= per_replica <= expected + batches * (geometry.lanes - 1):
            raise RuntimeError(
                f"AiM realization writes {per_replica} element(s) of {value!r} "
                f"per replica; the slice {index} holds {expected}"
            )
    if program.schedule is None:
        return
    # Inputs the group reads from the banks reach them by a host write (for
    # any group's layout: one scatter serves every reader of the buffer), by
    # residency, or as the result of another group of the program.  GPR
    # operands (EWADD) have no lowered ingress: the AiM trace starts with the
    # GPR host-written, as the lowering's manifest states, so they are not
    # checked here.
    host_written = {
        ref
        for op in ops
        if isinstance(op, _HOST_TRANSFER_RECORDS)
        and (isinstance(op, AimAllBankWrite) or op.direction == "write")
        for ref in op.buffer_refs
    }
    produced = {
        program._key(match.result_value_ref, match.result_memref_name)
        for other in program.groups
        if other is not group
        for match in other.matches
    }
    for transfer in effects.transfers:
        if transfer.direction not in {"in", "inout"} or transfer.physical:
            continue
        landing = storage_of(transfer.value, "in")
        if landing is None or landing.storage != "banks":
            continue
        ref = ref_of_name.get(transfer.value)
        if ref is None or ref in written or ref in host_written:
            continue
        if ref in program.resident or ref in produced:
            continue
        raise RuntimeError(
            f"AiM realization never brings {transfer.value!r} to the device: the "
            "host program neither writes it nor keeps it resident, and no other "
            "group of the program produces it"
        )


class AimCtx(CodegenContext):
    """Codegen ctx for AiM: emits the ramulator2 text trace.

    ``self.cmds`` is a list of trace lines (not ``PIMCmd`` records).  Every
    line comes from :meth:`emit_program`: matched kernels become typed
    ``AimOp`` records that ``_AimLowerer`` expands into simulator commands, so
    there is exactly one ISR encoder (``allo.pim.aim_lowering``).  The target's
    op/move declarations carry no ``emit`` callbacks for AiM; only the host
    moves keep one, which reports the transfer direction (``host_scatter``
    and friends below).
    """

    def __init__(self, target):
        super().__init__(target)
        # AiM's ramulator2 frontend consumes a text trace.
        self.cmds: list[str] = []
        # Host steps of the retained program that lower to no ISR line
        # (``AimHostReduction``), in program order.
        self.host_steps: list = []

    def append(self, line: str):
        """Low-level escape hatch — append a raw trace line."""
        self.cmds.append(line)

    def emit_eoc(self):
        """End-of-command marker — every AiM trace ends with EOC."""
        self.cmds.append("AiM EOC")

    def _sbk_banks(self, placement) -> tuple[int, ...]:
        """Physical banks one single-bank launch touches, from the layout."""
        layout = getattr(placement, "layout", None)
        if layout is None or not {"work_bank", "lane_bank"}.issubset(layout.bases):
            raise NotImplementedError(
                "AiM SBK expansion needs a LinearLayout with work_bank and "
                "lane_bank inputs"
            )
        out_index = layout.out_dims.index("bank")
        fixed = {
            name: 0 for name in layout.bases if name not in ("work_bank", "lane_bank")
        }
        banks: list[int] = []
        for work_bank in range(layout.size_of("work_bank")):
            for lane_bank in range(layout.size_of("lane_bank")):
                bank = layout.apply(**fixed, work_bank=work_bank, lane_bank=lane_bank)[
                    out_index
                ]
                if bank not in banks:
                    banks.append(bank)
        return tuple(banks)

    def emit_program(
        self, trace, layouts, *, schedule=None, host_moves=(), module=None
    ) -> None:
        """Derive AimOp records from matches, placements and the host program.

        Each kernel group lowers once per launch from its representative
        bucket: AiM commands are channel-masked, so replicas are carried by the
        records, never by repeated emission. Bank rows come from the A4
        allocator; host steps lower per A5 (spec 002).
        """
        program = _AimProgramDerivation(self.target, trace, layouts, schedule)
        records = program.records(host_moves, self)
        # id(layout) -> kernel group, for `check_realization`.
        from .spmw_autoschedule import _bucket_for_autoschedule

        self._group_of_layout = {}
        for (_scope, bucket), layout in zip(_bucket_for_autoschedule(trace), layouts):
            self._group_of_layout[id(layout)] = _matcher_work_scope(bucket[0]).group_id
        self._program = program
        # Groups the host program launches (every group without a schedule):
        # a group the given host program never launches lowers no record,
        # so its candidate's declaration is not realized by this program.
        self._every_group_launched = schedule is None
        self._launched_groups = set(
            step.group_id
            for step in (schedule.steps if schedule is not None else ())
            if isinstance(step, LaunchStep)
        )

        lowerer = _AimLowerer(self.target)
        lines: list[str] = []
        operations = []
        for operation, sbk_banks in records:
            begin = len(lines)
            start = len(lowerer.commands)
            try:
                lowerer.lower(operation, len(operations))
            except ValueError as error:
                from .pim.schedule_search import InfeasibleSchedule

                raise InfeasibleSchedule(
                    f"AiM {operation.name} is not realizable: {error}"
                ) from error
            operations.append(operation)
            for line in lowerer.commands[start:]:
                if sbk_banks is not None and line.startswith("AiM MAC_ABK "):
                    _opcode, _name, op_size, mask, row = line.split(" ")
                    lines.extend(
                        f"AiM MAC_SBK {op_size} {mask} {bank} {row}"
                        for bank in sbk_banks
                    )
                else:
                    lines.append(line)
            entry = lowerer.operations[-1]
            entry["command_span"] = [begin, len(lines)]
            entry["command_count"] = len(lines) - begin
            if isinstance(operation, AimHostReduction):
                # Runs on the host after the device commands before it.
                entry["after_command"] = len(lines)
                self.host_steps.append(operation)
            if entry["kind"] == "contraction":
                row_map = program.row_maps.get(id(operation))
                if row_map is not None:
                    entry["row_map"] = _row_map_entry(entry, *row_map)
                entry["bank_scope"] = "abk" if sbk_banks is None else "sbk"
                if sbk_banks is not None:
                    entry["sbk_banks"] = list(sbk_banks)
                program_macs = program.program_macs.get(id(operation))
                if program_macs is not None:
                    entry.update(
                        check_contraction_coverage(
                            entry,
                            issued_scalar_macs(lines[begin:], lowerer.geometry),
                            program_macs,
                        )
                    )
        self.lowered_operations = tuple(operations)
        self.lowering_manifest = lowerer.operations
        self.row_regions = program.regions
        self.cmds.extend(lines)
        if program.groups:
            self.emit_eoc()

    def check_realization(self, layouts) -> None:
        """The lowered AiM records against the selected candidates' declared
        effects, per kernel group: the contraction's result instances and
        reduction extent (per work item, or aggregated over a partitioned
        group), its bank scope, reduction partitions (with the host reduction
        that merges them), the vector's residency (GB broadcast or bank
        pair), the matrix's bank residency, the batch mapping it requests
        and the one the lowering realizes, and the GPR readout of its results;
        an elementwise record's element count and home; and every host
        transfer the host program lowered for the group, bound to the
        candidate's declared transfers through the retained buffer identities
        the records carry (direction, slice extent and device endpoint), with
        every declared input written by the host, resident or produced on the
        device."""
        from .spmw_candidate import candidate_effects

        layouts = list(layouts) if isinstance(layouts, (list, tuple)) else [layouts]
        program = getattr(self, "_program", None)
        if program is None:
            return
        groups = {group.group_id: group for group in program.groups}
        ops = list(getattr(self, "lowered_operations", ()))
        manifest = list(getattr(self, "lowering_manifest", ()))
        if len(manifest) != len(ops):
            raise RuntimeError(
                f"AiM realization lowered {len(ops)} records but {len(manifest)} "
                "manifest entries"
            )
        manifest_of = {id(op): entry for op, entry in zip(ops, manifest)}
        for layout in layouts:
            effects = candidate_effects(layout)
            if effects is None:
                continue
            group = groups.get(getattr(self, "_group_of_layout", {}).get(id(layout)))
            if group is None:
                raise RuntimeError(
                    f"AiM realization lowered no group for the candidate "
                    f"{layout.mode!r} that declares effects"
                )
            if not getattr(self, "_every_group_launched", True) and (
                group.group_id not in getattr(self, "_launched_groups", set())
            ):
                continue
            names = set(group.functions) | {group.name}
            (launch,) = effects.launches if len(effects.launches) == 1 else (None,)
            if launch is None:
                raise RuntimeError("AiM realization is one record per kernel group")
            images = {}
            for v in effects.value_layouts:
                images.setdefault(v.value, {})[v.image] = v
            storage_of = functools.partial(_layout_image, images)
            items = len(group.functions)
            if group.kind == "contraction":
                records = [
                    op
                    for op in ops
                    if isinstance(op, AimContraction) and op.name in names
                ]
                if not records:
                    raise RuntimeError(
                        f"AiM realization lowered no contraction record for group "
                        f"{group.name!r}"
                    )
                # A host program may launch the group more than once; every
                # launch lowers the same contraction, checked below.
                record = records[0]
                if any(
                    (r.outputs, r.batches, r.reduction, r.reduction_partitions)
                    != (
                        record.outputs,
                        record.batches,
                        record.reduction,
                        record.reduction_partitions,
                    )
                    or r.input_source != record.input_source
                    for r in records[1:]
                ):
                    raise RuntimeError(
                        f"AiM realization lowered differing contraction records for "
                        f"the launches of group {group.name!r}"
                    )
                declared = (tuple(launch.instances), tuple(launch.reduction or ()))
                per_item = (0, record.outputs * record.batches)
                aggregate = (0, record.outputs * record.batches // max(items, 1))
                if declared[1] != (0, record.reduction) or declared[0] not in (
                    per_item,
                    aggregate,
                ):
                    raise RuntimeError(
                        "AiM realization launch differs from the candidate contract: "
                        f"realized outputs {record.outputs} x batches {record.batches} "
                        f"over {items} work item(s), reduction {record.reduction}; "
                        f"declared {declared}"
                    )
                scope = "abk" if group.sbk_banks is None else "sbk"
                kind = "MAC_ABK" if scope == "abk" else "MAC"
                partitions = int(record.reduction_partitions)
                merges = [
                    op
                    for op in ops
                    if isinstance(op, AimHostReduction) and op.source == record.name
                ]
                if (
                    launch.kind != kind
                    or len(launch.covers) < 3
                    or launch.covers[1] != scope
                    or int(launch.covers[2]) != partitions
                    or (partitions > 1) != bool(merges)
                ):
                    raise RuntimeError(
                        f"AiM realization issues {kind} ({scope}, {partitions} reduction "
                        f"partition(s){', host-merged' if merges else ''}); the "
                        f"candidate declares {launch.kind} {launch.covers}"
                    )
                (contraction_entry,) = [
                    c for c in group.contractions if c["match"].func_name == record.name
                ] or [None]
                shape = contraction_entry["shape"] if contraction_entry else {}
                matrix = storage_of(shape.get("matrix"), "in")
                vector = storage_of(shape.get("vector"), "in")
                vector_storage = "banks" if record.input_source == "banks" else "gb"
                if matrix is None or matrix.storage != "banks":
                    raise RuntimeError(
                        f"AiM realization keeps the matrix {shape.get('matrix')!r} in "
                        f"the banks; the candidate declared {matrix}"
                    )
                if vector is None or vector.storage != vector_storage:
                    raise RuntimeError(
                        f"AiM realization reads the vector {shape.get('vector')!r} from "
                        f"{vector_storage}; the candidate declared {vector}"
                    )
                vector_transfer = {t.value: t for t in effects.transfers}.get(
                    shape.get("vector")
                )
                if vector_transfer is None or (
                    tuple(vector_transfer.route)[-1] != vector_storage
                ):
                    raise RuntimeError(
                        f"AiM realization moves the vector into {vector_storage}; the "
                        f"candidate's transfer ends at "
                        f"{vector_transfer and vector_transfer.route}"
                    )
                if {s for _n, _b, s in effects.allocations} != {"mac_reg"}:
                    raise RuntimeError(
                        "AiM realization accumulates in mac_reg; the candidate "
                        f"declares {list(effects.allocations)}"
                    )
                # Batch mapping: the declared request is the record's request
                # (an unspecified one is the lowering's ``auto`` rule) and an
                # explicit one is the mapping realized; the lowering's choice
                # is the one the derivation laid the host transfers out for.
                requested = str(launch.covers[3]) if len(launch.covers) > 3 else "auto"
                requested = "auto" if requested in {"", "none"} else requested
                realized_mappings = {
                    manifest_of[id(r)].get("batch_mapping") for r in records
                }
                expected_realized = (
                    contraction_entry["mapping"] if contraction_entry else None
                )
                if (
                    requested != record.batch_mapping
                    or (requested != "auto" and realized_mappings != {requested})
                    or (
                        expected_realized is not None
                        and realized_mappings != {expected_realized}
                    )
                ):
                    raise RuntimeError(
                        f"AiM realization maps batches {sorted(realized_mappings)} "
                        f"(request {record.batch_mapping!r}); the candidate declares "
                        f"the batch mapping {requested!r}"
                    )
                # Results: the MAC accumulators are read back into the GPR
                # (RD_MAC) and gathered from there; a declared result that
                # lives elsewhere, or leaves by another route, is not this
                # program.
                for name in effects.coverage.outputs:
                    result = storage_of(name, "out")
                    transfer = {t.value: t for t in effects.transfers}.get(name)
                    route = tuple(transfer.route) if transfer is not None else ()
                    if (
                        result is None
                        or result.storage != "gpr"
                        or "gpr" not in route
                        or route[-1] != "host_gather"
                    ):
                        raise RuntimeError(
                            f"AiM realization reads the results of {name!r} back "
                            "from gpr to the host; the candidate declared "
                            f"{result} with transfer {transfer}"
                        )
            else:
                records = [
                    op
                    for op in ops
                    if isinstance(op, AimElementwise) and op.name in names
                ]
                if not records:
                    raise RuntimeError(
                        f"AiM realization lowered no elementwise record for group "
                        f"{group.name!r}"
                    )
                record = records[0]
                if any(
                    (r.operation, r.elements) != (record.operation, record.elements)
                    for r in records[1:]
                ):
                    raise RuntimeError(
                        f"AiM realization lowered differing elementwise records for "
                        f"the launches of group {group.name!r}"
                    )
                per_item = (0, record.elements)
                aggregate = (0, record.elements // max(items, 1))
                if launch.kind != record.operation.upper() or tuple(
                    launch.instances
                ) not in (per_item, aggregate):
                    raise RuntimeError(
                        f"AiM realization is {record.operation.upper()} over "
                        f"{record.elements} elements ({items} work item(s)); the "
                        f"candidate declares {launch.kind} {launch.instances}"
                    )
                home = "banks" if record.operation == "mul" else "gpr"
                for v in effects.value_layouts:
                    if v.storage != home:
                        raise RuntimeError(
                            f"AiM realization keeps {v.value!r} in {home}; the "
                            f"candidate declared {v}"
                        )
            _check_host_transfers(program, group, effects, ops, storage_of)

    # Host-move emit lambdas reach these; the derivation consumes the
    # recorded direction instead, so they only validate the device endpoint.
    def host_scatter(self, handle):
        return ("scatter", handle)

    def host_gather(self, handle):
        return ("gather", handle)

    def host_broadcast(self, handle):
        return ("broadcast", handle)

    def runtime_route(self, commands) -> tuple[str, tuple]:
        try:
            runtime_segments = _aim_runtime_segments(commands)
        except ValueError:
            runtime_segments = ()
        return "ramulator2-preterminated-segments", runtime_segments


# --------------------------------------------------------------------- #
# Spec 002 A3-A5: AimOp derivation, row allocation, host-step lowering.
# --------------------------------------------------------------------- #


def _bank_slot_conventions(target):
    """EWMUL and bank-pair MAC bank slots declared by the target's ``banks``
    memory (CENT mapper conventions; see ``build_aim_target``)."""
    geometry = target.banks.geometry
    try:
        x, y, dst = (int(slot) for slot in geometry["ewmul_slots"])
        stride = int(geometry["bank_pair_stride"])
        pair = tuple(int(slot) for slot in geometry["bank_pair_slots"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "aim target banks must declare ewmul_slots, bank_pair_stride and "
            "bank_pair_slots"
        ) from error
    return {"x": x, "y": y, "dst": dst}, stride, pair


def _ceil(value, divisor):
    return -(-int(value) // int(divisor))


def _reduction_partitions(group, placement) -> int:
    """P of a group: the placement's LinearLayout carries it (reduction bits
    mapped onto ``token_partition``); the knob value must agree."""
    carried = layout_reduction_partitions(getattr(placement, "layout", None))
    declared = group.properties.get("reduction_partitions")
    declared = 1 if declared in (None, "none") else int(declared)
    if declared != carried:
        raise ValueError(
            f"AiM {group.name}: placement declares reduction_partitions="
            f"{declared} but its LinearLayout splits the reduction {carried} ways"
        )
    return carried


def _row_map_digest(values) -> str:
    """FNV-1a 64 over the row -> slot table."""
    digest = 0xCBF29CE484222325
    for value in values:
        digest ^= int(value)
        digest = (digest * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{digest:016x}"


def _contraction_row_map(target, group, matches, layouts):
    """``(row slots or None, record)`` of the owner map a contraction group's
    placements carry, or None for the work grid's own map.

    The work grid partitions the output frontier in work-id order, so work
    item ``w`` owns frontier positions ``[w R, (w + 1) R)``.  A realizable
    map (``spmw_layout_problem.row_map_unrealizable_reason``) gives flat
    owner unit ``u`` the same ``R`` rows; they occupy that unit's positions
    in ascending order: ``slot[row] = u R + rank``.  The trace's MAC, GB and
    readout commands address positions, not rows, so only per-row host
    transfers and the host-side result mapping change.  The AiM
    ``reduction_partitions`` split is realized by the placements that carry
    the same ``P`` (rows stay in place).  Raises InfeasibleSchedule with the
    reason otherwise.
    """
    from .pim.layout_problem import TargetHierarchyError
    from .pim.schedule_search import InfeasibleSchedule
    from .spmw_layout_problem import (
        LayoutDerivationError,
        derive_layout_problem,
        owner_rows,
        reduction_partition_split,
        row_map_unrealizable_reason,
        split_movement_layout,
    )

    movements = [split_movement_layout(getattr(p, "layout", None))[1] for p in layouts]
    if all(m is None for m in movements):
        return None
    if any(m is None or m != movements[0] for m in movements):
        raise InfeasibleSchedule(
            f"AiM {group.name}: work items carry different owner maps"
        )
    try:
        derived = derive_layout_problem(target, matches)
        matrix = derived.movement_matrix(movements[0])
    except (LayoutDerivationError, TargetHierarchyError) as error:
        raise InfeasibleSchedule(
            f"AiM {group.name}: the carried movement layout cannot be placed: {error}"
        ) from error
    program = derived.program_layout()
    if matrix == program:
        return None
    partitions = reduction_partition_split(derived, matrix)
    if partitions is not None:
        if group.reduction_partitions != partitions:
            raise InfeasibleSchedule(
                f"AiM {group.name}: the owner map is the reduction_partitions="
                f"{partitions} split; this placement lowers reduction_partitions="
                f"{group.reduction_partitions}"
            )
        return None, {"map": "reduction_partitions", "partitions": partitions}
    if derived.owner_flag(matrix) != derived.owner_flag(program):
        reason = row_map_unrealizable_reason(derived, matrix)
        if reason is not None:
            raise InfeasibleSchedule(
                f"AiM {group.name}: owner map not realizable: {reason}"
            )
    shapes = [contraction["shape"] for contraction in group.contractions]
    if (
        len(shapes) != 1
        or shapes[0]["batches"] != 1
        or shapes[0]["replicas"] != 1
        or group.replicas != 1
        or group.reduction_partitions != 1
    ):
        raise InfeasibleSchedule(
            f"AiM {group.name}: owner map not realizable: rows are placed by the "
            "owner map only for one unreplicated single-batch contraction with an "
            "unsplit reduction"
        )
    outputs = shapes[0]["outputs"]
    try:
        units, own = owner_rows(derived, matrix), owner_rows(derived, program)
    except LayoutDerivationError as error:
        raise InfeasibleSchedule(f"AiM {group.name}: {error}") from error
    count = len(own)
    if count == 0 or outputs % count:
        raise InfeasibleSchedule(
            f"AiM {group.name}: {outputs} outputs over {count} work items"
        )
    per_unit = outputs // count
    frontier = {u: list(range(u * per_unit, (u + 1) * per_unit)) for u in range(count)}
    if (
        own != frontier
        or sorted(units) != list(range(count))
        or any(len(rows) != per_unit for rows in units.values())
    ):
        raise InfeasibleSchedule(
            f"AiM {group.name}: owner map not realizable: the work items do not "
            f"partition the {outputs}-output frontier (rows 0..{outputs - 1}) in "
            f"work-id order, or the map does not give each of the {count} units "
            f"{per_unit} rows"
        )
    slots = [0] * outputs
    for unit, rows in units.items():
        for rank, row in enumerate(rows):
            slots[row] = unit * per_unit + rank
    if slots == list(range(outputs)):
        return None
    moved = sum(1 for row, slot in enumerate(slots) if row != slot)
    return tuple(slots), {
        "map": "permuted",
        "rows_moved": moved,
        "rows_per_unit": per_unit,
        "digest": _row_map_digest(slots),
    }


def _frontier_position(entry, slot) -> dict:
    """Launch or tile, physical channel and bank of output-frontier position
    ``slot`` under the entry's matrix storage contract (one batch, one
    replica)."""
    banks = int(entry["active_banks_per_channel"])
    mapping = entry.get("batch_mapping")
    if mapping == "flattened":
        capacity = int(entry["output_capacity_per_launch"])
        channels = entry["active_channels"]
        return {
            "output_launch": slot // capacity,
            "channel": channels[(slot % capacity) // banks],
            "bank": slot % banks,
        }
    if mapping == "row_packed":
        capacity = int(entry["output_capacity_per_tile_per_replica"])
        channels = entry["replica_channel_groups"][0]
        return {
            "output_tile": slot // capacity,
            "channel": channels[(slot % capacity) // banks],
            "bank": slot % banks,
        }
    return {
        "output_tile": slot // banks,
        "channel": entry["replica_channel_groups"][0][0],
        "bank": slot % banks,
    }


def _row_map_entry(entry, slots, record) -> dict:
    """The lowering-manifest ``row_map`` block: the record, the row -> slot
    table and the physical position of sample rows."""
    out = dict(record)
    if slots is None:
        return out
    rows = len(slots)
    per_unit = record["rows_per_unit"]
    samples = sorted({0, 1, per_unit - 1, per_unit, rows - 1} & set(range(rows)))
    out["row_slots"] = list(slots)
    out["placement"] = {
        row: {"slot": slots[row], **_frontier_position(entry, slots[row])}
        for row in samples
    }
    return out


class _Usage:
    """How one kernel group holds one buffer in banks."""

    __slots__ = ("kind", "stride", "slot", "copies", "store", "contraction")

    def __init__(
        self, kind, *, stride=0, slot=0, copies=1, store=False, contraction=None
    ):
        self.kind = kind  # "slot" or "matrix"
        self.stride = stride
        self.slot = slot
        self.copies = copies
        self.store = store
        self.contraction = contraction


class _Group:
    __slots__ = (
        "group_id",
        "name",
        "functions",
        "matches",
        "properties",
        "kind",
        "replicas",
        "partitions",
        "elements",
        "usages",
        "contractions",
        "sbk_banks",
        "row",
        "reduction_partitions",
        "row_map",
        "split_replicas",
    )

    def __init__(self, group_id, name):
        self.group_id = group_id
        self.name = name
        self.usages = {}
        self.contractions = []
        self.sbk_banks = None
        self.elements = 0
        self.row = 0
        self.reduction_partitions = 1
        # True when the replicas are a partitioned grid's work items, each
        # owning 1/replicas of every operand (EWMUL), not full copies.
        self.split_replicas = False
        # (row -> frontier slot table or None, record) of a carried owner map
        # other than the work grid's; None for the work grid's own map.
        self.row_map = None


class _AimProgramDerivation:
    """Derive the ordered AimOp records one compiled AiM program lowers."""

    def __init__(self, target, trace, layouts, schedule):
        from .spmw_autoschedule import _bucket_for_autoschedule

        self.target = target
        self.trace = trace
        self.schedule = schedule
        self.geometry = _target_geometry(target)
        self.group_width = self.geometry.banks // self.geometry.bank_groups
        (
            self.ewmul_slots,
            self.bank_pair_stride,
            self.bank_pair_slots,
        ) = _bank_slot_conventions(target)
        self.layout_by_scope = {
            scope: layout
            for (scope, _matches), layout in zip(
                _bucket_for_autoschedule(trace), layouts
            )
        }
        self._index_buffers()
        # id(AimContraction record) -> scalar MACs of its matched program.
        self.program_macs: dict[int, int] = {}
        # id(AimContraction record) -> (row slots or None, row-map record).
        self.row_maps: dict[int, tuple] = {}
        self.groups = self._derive_groups()
        self.by_id = {group.group_id: group for group in self.groups}
        self.regions = self._allocate_rows()

    # ----------------------------- buffers ----------------------------- #

    @staticmethod
    def _key(ref, name):
        return ref if ref is not None else ("memref", name)

    def _index_buffers(self):
        refs = getattr(self.trace, "source_value_refs", None) or {}
        parameters = tuple(getattr(self.schedule, "parameters", ()) or ())
        self.key_by_name = {}
        self.name_by_key = {}
        self.shape_by_key = {}
        self.declaration = {}
        for ordinal, (name, shape) in enumerate(parameters):
            ref = refs.get(ordinal)
            if ref is None:
                continue
            self.key_by_name[name] = ref
            self.name_by_key[ref] = name
            self.declaration[ref] = ordinal
            if shape is not None:
                self.shape_by_key[ref] = tuple(shape)
        for match in self.trace.matches:
            for operand in match.operands:
                key = self._key(operand.value_ref, operand.memref_name)
                shape = _memref_shape(getattr(operand, "memref_type", None))
                if shape and key not in self.shape_by_key:
                    self.shape_by_key[key] = shape
                self.declaration.setdefault(
                    key, len(parameters) + len(self.declaration)
                )
            key = self._key(match.result_value_ref, match.result_memref_name)
            self.declaration.setdefault(key, len(parameters) + len(self.declaration))
        self.resident = {
            self.key_by_name[name]
            for name in (getattr(self.schedule, "resident", ()) or ())
            if name in self.key_by_name
        }

    def _elements(self, key, fallback):
        shape = self.shape_by_key.get(key)
        return math.prod(shape) if shape else fallback

    # ----------------------------- groups ------------------------------ #

    def _derive_groups(self):
        from .spmw_autoschedule import _matcher_search_scope, derive_layout_properties
        from .spmw_match_engine import _parse_work_id

        members: dict[int, list] = {}
        order: list[int] = []
        for match in self.trace.matches:
            group_id = _matcher_work_scope(match).group_id
            if group_id not in members:
                members[group_id] = []
                order.append(group_id)
            members[group_id].append(match)

        groups = []
        for group_id in order:
            matches = members[group_id]
            functions = list(dict.fromkeys(match.func_name for match in matches))
            reps = [match for match in matches if match.func_name == functions[0]]
            group = _Group(group_id, _parse_work_id(functions[0])[0])
            group.functions = functions
            group.matches = reps
            placement = self.layout_by_scope[_matcher_search_scope(reps[0])]
            group.properties = derive_layout_properties(self.target, placement)
            replicated = group_replicated(matches)
            group.replicas = len(functions) if replicated else 1
            partitions = group.properties.get("replica_partitions")
            group.partitions = (
                int(partitions)
                if partitions not in (None, "none")
                else self.geometry.channels // group.replicas
            )
            names = {match.target_op_name for match in reps}
            group.reduction_partitions = _reduction_partitions(group, placement)
            if "MAC" in names:
                group.kind = "contraction"
                if group.properties.get("bank_fanout") == 1:
                    group.sbk_banks = AimCtx._sbk_banks(None, placement)
                for match in reps:
                    if match.target_op_name != "MAC":
                        continue
                    shape = contraction_shape(
                        match,
                        replicated=replicated,
                        replicas=group.replicas,
                    )
                    matrix, vector = mac_operands(match)
                    acc = next(op for op in match.operands if op.role == "acc")
                    activation = any(
                        other.target_op_name == "AF"
                        and any(
                            op.role == "x" and op.memref_name == acc.memref_name
                            for op in other.operands
                        )
                        for other in reps
                    )
                    mapping = resolve_batch_mapping(
                        shape, group.properties.get("batch_mapping"), self.geometry
                    )
                    contraction = {
                        "match": match,
                        "shape": shape,
                        "activation": activation,
                        "mapping": mapping,
                        "axes": matrix_axes(match),
                        "matrix": self._key(matrix.value_ref, matrix.memref_name),
                    }
                    group.contractions.append(contraction)
                    key = contraction["matrix"]
                    if shape["input_source"] == "banks":
                        group.usages[key] = _Usage(
                            "slot",
                            stride=self.bank_pair_stride,
                            slot=self.bank_pair_slots[0],
                            copies=len(self.bank_pair_slots),
                        )
                    else:
                        group.usages[key] = _Usage("matrix", contraction=contraction)
                group.row_map = _contraction_row_map(
                    self.target,
                    group,
                    matches,
                    [
                        self.layout_by_scope[_matcher_search_scope(match)]
                        for match in matches
                        if _matcher_search_scope(match) in self.layout_by_scope
                    ],
                )
                if group.row_map is not None and group.row_map[0] is not None:
                    for contraction in group.contractions:
                        contraction["row_slots"] = group.row_map[0]
            elif "MUL" in names or "ADD" in names:
                kind = "MUL" if "MUL" in names else "ADD"
                matched = [match for match in reps if match.target_op_name == kind]
                if len(matched) != 1:
                    raise NotImplementedError(
                        f"AiM {group.name}: expected one {kind} per kernel, "
                        f"got {len(matched)}"
                    )
                match = matched[0]
                from .spmw_match_engine import output_map_problem

                problem = output_map_problem(match)
                if problem is not None:
                    # The EWMUL/EWADD stream writes element i of the result
                    # from element i of the operands (C-24).
                    raise NotImplementedError(
                        f"AiM {group.name}: the output map is not the identity the "
                        f"elementwise lowering realizes: {problem}"
                    )
                group.kind = kind.lower()
                group.elements = math.prod(
                    _parse_loop_bound(ub)
                    for _var, _lb, ub, _step in match.enclosing_loops
                )
                self._spread_elementwise_grid(group, match, replicated)
                if kind == "MUL":
                    roles = {op.role: op for op in match.operands}
                    for role in ("x", "y"):
                        operand = roles[role]
                        key = self._key(operand.value_ref, operand.memref_name)
                        group.usages[key] = _Usage(
                            "slot", stride=self.group_width, slot=self.ewmul_slots[role]
                        )
                    key = self._key(match.result_value_ref, match.result_memref_name)
                    group.usages[key] = _Usage(
                        "slot",
                        stride=self.group_width,
                        slot=self.ewmul_slots["dst"],
                        store=True,
                    )
            else:
                raise NotImplementedError(
                    f"AiM {group.name} has no MAC, MUL or ADD match; got {sorted(names)}"
                )
            groups.append(group)
        return groups

    def _spread_elementwise_grid(self, group, match, replicated):
        """Cover every work item of a partitioned elementwise grid (M-14).

        ``group.elements`` is one work item's trip count. A replicated grid
        repeats one body, so one copy of it is the program (MUL keeps one
        replica per work item, as before). A partitioned grid's work items
        each own a disjoint slice of the operands:

        * MUL lowers one replica per work item, each on its own group of
          ``replica_partitions`` channels holding that item's slice, so one
          EWMUL whose mask spans every replica's channels covers all
          ``work_items * elements``. This is the layout the knob enumerator
          (replicas = work items) and the cost program (one EWMUL per work
          item on its own channel) already assume.
        * ADD runs on the single device GPR, so one EWADD covers the
          aggregate ``work_items * elements``.

        The slices must tile the operands exactly; anything else is refused
        rather than lowered with dropped or duplicated work.
        """
        items = _work_items(match)
        if replicated or items == 1:
            return
        per_item = group.elements
        total = per_item * items
        roles = {op.role: op for op in match.operands}
        operands = [roles.get(role) for role in ("x", "y")]
        extents = {
            self._elements(self._key(op.value_ref, op.memref_name), None)
            for op in operands
            if op is not None
        }
        extents.add(
            self._elements(
                self._key(match.result_value_ref, match.result_memref_name), None
            )
        )
        extents.discard(None)
        if extents != {total}:
            raise NotImplementedError(
                f"AiM {group.name}: {items} work items x {per_item} elements "
                f"do not tile the operands (extents {sorted(extents)})"
            )
        if group.kind == "add":
            group.elements = total
            return
        group.replicas = items
        group.split_replicas = True
        if group.properties.get("replica_partitions") in (None, "none"):
            group.partitions = max(1, self.geometry.channels // items)

    # --------------------------- row allocator ------------------------- #

    def _relocated(self, group, key):
        usage = group.usages[key]
        if usage.store:
            return False
        return any(
            other is not group
            and key in other.usages
            and other.usages[key].store
            and other.usages[key].slot != usage.slot
            for other in self.groups
        )

    def _footprint(self, group, key, usage):
        geometry = self.geometry
        cols = geometry.row_elements
        if usage.kind == "slot":
            elements = self._elements(key, group.elements)
            if group.split_replicas:
                elements = _ceil(elements, group.replicas)
            partitions = group.partitions * (geometry.banks // usage.stride)
            return _ceil(_ceil(elements, partitions), cols)
        contraction = usage.contraction
        shape = self.shape_by_key.get(key) or ()
        extents = {"batch": 1, "output": 1, "reduction": 1}
        if len(shape) == len(contraction["axes"]):
            for extent, role in zip(shape, contraction["axes"]):
                extents[role] *= extent
        else:
            extents = {
                "batch": contraction["shape"]["batches"],
                "output": contraction["shape"]["outputs"],
                "reduction": contraction["shape"]["reduction"],
            }
        batch, outputs, reduction = (
            extents["batch"],
            extents["output"],
            extents["reduction"],
        )
        cpr = group.partitions
        channels = group.replicas * cpr
        mapping = contraction["mapping"]
        if mapping == "row_packed":
            aligned = _ceil(reduction, geometry.lanes) * geometry.lanes
            pack = max(1, min(batch, cols // aligned))
            return _ceil(batch, pack) * _ceil(outputs, cpr * geometry.banks)
        if mapping == "channels":
            # Reduction partitions are extra channel-batched accumulators,
            # each storing ceil(reduction / P) local elements.
            parts = group.reduction_partitions
            return (
                _ceil(batch * parts, cpr)
                * _ceil(outputs, geometry.banks)
                * _ceil(_ceil(reduction, parts), cols)
            )
        return _ceil(
            outputs * batch * group.replicas, channels * geometry.banks
        ) * _ceil(reduction, cols)

    def _allocate_rows(self):
        from .pim.schedule_search import InfeasibleSchedule

        parent: dict = {}

        def find(key):
            parent.setdefault(key, key)
            while parent[key] != key:
                parent[key] = parent[parent[key]]
                key = parent[key]
            return key

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        anchors = {}
        for group in self.groups:
            keys = list(group.usages)
            if not keys:
                continue
            for key in keys:
                find(key)
            relocated = {key for key in keys if self._relocated(group, key)}
            if group.kind == "mul" and any(key in self.resident for key in keys):
                kept = [key for key in keys if key not in relocated]
            else:
                kept = keys
            for key in kept[1:]:
                union(kept[0], key)
            anchors[group.group_id] = kept[0]

        footprints: dict = {}
        for group in self.groups:
            for key, usage in group.usages.items():
                if (
                    group.kind == "mul"
                    and self._relocated(group, key)
                    and any(k in self.resident for k in group.usages)
                ):
                    continue
                root = find(key)
                footprints[root] = max(
                    footprints.get(root, 1), self._footprint(group, key, usage)
                )
        components: dict = {}
        for key in parent:
            components.setdefault(find(key), []).append(key)
        ordered = sorted(
            components,
            key=lambda root: min(
                self.declaration.get(k, 1 << 30) for k in components[root]
            ),
        )
        rows = {}
        regions = []
        next_row = 0
        for root in ordered:
            count = max(1, footprints.get(root, 1))
            rows[root] = next_row
            regions.append(
                {
                    "buffers": sorted(
                        self.name_by_key.get(k, str(k)) for k in components[root]
                    ),
                    "row": next_row,
                    "rows": count,
                }
            )
            next_row += count
        if next_row > self.geometry.rows:
            raise InfeasibleSchedule(
                f"AiM bank rows: program needs {next_row}, target has "
                f"{self.geometry.rows}"
            )
        self.row_of_key = {key: rows[find(key)] for key in parent}
        for group in self.groups:
            if group.group_id in anchors:
                group.row = self.row_of_key[anchors[group.group_id]]
        return regions

    # ------------------------------ records ---------------------------- #

    def _launch_records(self, group):
        records = []
        if group.kind == "contraction":
            knobs = {
                name: group.properties[name]
                for name in ("batch_mapping", "reuse_group", "replica_partitions")
                if group.properties.get(name) not in (None, "none")
            }
            knobs["reduction_partitions"] = group.reduction_partitions
            for contraction in group.contractions:
                shape = contraction["shape"]
                if shape["replicas"] * group.partitions > self.geometry.channels and (
                    "replica_partitions" in knobs
                ):
                    from .pim.schedule_search import InfeasibleSchedule

                    raise InfeasibleSchedule(
                        f"AiM {group.name}: {shape['replicas']} replicas x "
                        f"{group.partitions} partitions exceed "
                        f"{self.geometry.channels} channels"
                    )
                if shape["input_source"] == "banks":
                    self._check_bank_pair_span(group, contraction)
                operation = build_contraction(
                    shape,
                    knobs,
                    activation=contraction["activation"],
                    name=contraction["match"].func_name,
                )
                operation = replace(
                    operation, row=self.row_of_key[contraction["matrix"]]
                )
                self.program_macs[id(operation)] = shape["program_scalar_macs"]
                if group.row_map is not None:
                    self.row_maps[id(operation)] = group.row_map
                records.append((operation, group.sbk_banks))
                if operation.reduction_partitions > 1:
                    records.append(
                        (
                            AimHostReduction(
                                partitions=operation.reduction_partitions,
                                outputs=operation.outputs,
                                batches=operation.batches,
                                replicas=operation.replicas,
                                source=operation.name,
                                name=f"{operation.name}.host_reduce",
                            ),
                            None,
                        )
                    )
        elif group.kind == "mul":
            records.append(
                (
                    AimElementwise(
                        "mul",
                        elements=group.elements,
                        row=group.row,
                        replicas=group.replicas,
                        channels_per_replica=group.partitions,
                        name=group.name,
                    ),
                    None,
                )
            )
        else:
            records.append(
                (
                    AimElementwise(
                        "add",
                        elements=group.elements,
                        gpr_addr_0=0,
                        gpr_addr_1=_ceil(group.elements, self.geometry.lanes),
                        name=group.name,
                    ),
                    None,
                )
            )
        return records

    def _check_bank_pair_span(self, group, contraction):
        """A bank-pair MAC reduces each pair's whole slice into one output.

        The host duplicates the buffer into both banks of every pair, one
        contiguous span per pair, so the span must equal the reduction.
        """
        from .pim.schedule_search import InfeasibleSchedule

        key = contraction["matrix"]
        elements = self._elements(key, 0)
        pairs = group.partitions * (self.geometry.banks // self.bank_pair_stride)
        span = _ceil(elements, pairs) if elements else None
        reduction = contraction["shape"]["reduction"]
        if span != reduction:
            raise InfeasibleSchedule(
                f"AiM {group.name}: {group.partitions} partitions give a "
                f"{span}-element bank-pair span, the reduction is {reduction}"
            )

    def _relocations(self, group, position, steps):
        """EWMUL results whose next reader takes them from another slot."""
        records = []
        for key, usage in group.usages.items():
            if not usage.store:
                continue
            reader = None
            for step in steps[position + 1 :]:
                if isinstance(step, LaunchStep):
                    candidate = self.by_id[step.group_id]
                    if key in candidate.usages and not candidate.usages[key].store:
                        reader = candidate
                        break
            if reader is None or reader.usages[key].kind != "slot":
                continue
            target_slot = reader.usages[key].slot
            if target_slot == usage.slot:
                continue
            for bank_group in range(self.geometry.bank_groups):
                base = bank_group * self.group_width
                common = dict(
                    elements=group.elements,
                    replicas=group.replicas,
                    channels_per_replica=group.partitions,
                    partitions_per_replica=group.partitions * self.geometry.bank_groups,
                )
                records.append(
                    (
                        AimBankCopy(
                            "bank_to_gb",
                            bank=base + usage.slot,
                            row=group.row,
                            name=f"{group.name}.relocate.bank_{base + usage.slot}",
                            **common,
                        ),
                        None,
                    )
                )
                records.append(
                    (
                        AimBankCopy(
                            "gb_to_bank",
                            bank=base + target_slot,
                            row=reader.row,
                            name=f"{group.name}.relocate.to_bank_{base + target_slot}",
                            **common,
                        ),
                        None,
                    )
                )
        return records

    def _host_usage(self, key, position, steps, direction):
        """(group, usage) whose layout a host step at ``position`` follows."""

        def launches(indices):
            for index in indices:
                step = steps[index]
                if isinstance(step, LaunchStep):
                    yield self.by_id[step.group_id]

        if direction in ("scatter", "broadcast"):
            for group in launches(range(position + 1, len(steps))):
                usage = group.usages.get(key)
                if usage is not None and not usage.store:
                    return group, usage
            for group in launches(range(position - 1, -1, -1)):
                usage = group.usages.get(key)
                if usage is not None and usage.store:
                    return group, usage
        else:
            for group in launches(range(position - 1, -1, -1)):
                usage = group.usages.get(key)
                if usage is not None and usage.store:
                    return group, usage
        raise NotImplementedError(
            f"AiM host {direction} of {self.name_by_key.get(key, key)} has no "
            "kernel layout to follow"
        )

    def _token_extent(self, key, index):
        shape = self.shape_by_key.get(key)
        if shape is None:
            raise NotImplementedError(
                f"AiM host transfer of unknown-shape buffer {key}"
            )
        if index is None:
            return math.prod(shape)
        if len(index) != len(shape):
            if len(index) == 1 and isinstance(index[0], tuple) and len(shape) >= 1:
                start, stop = index[0]
                return stop - start
            raise NotImplementedError(f"AiM host slice {index} does not fit {shape}")
        extent = 1
        for item, dim in zip(index, shape):
            if item is None:
                extent *= dim
            elif isinstance(item, tuple):
                extent *= item[1] - item[0]
        return extent

    def _matrix_records(self, direction, group, usage, index, name, key=None):
        """Cache-append transfers into a contraction matrix layout (A5)."""
        identity = {
            "buffer_refs": (key,),
            "buffer_slices": (index,),
            "group_id": group.group_id,
        }
        geometry = self.geometry
        contraction = usage.contraction
        axes = contraction["axes"]
        shape = contraction["shape"]
        if index is None or len(index) != len(axes):
            raise NotImplementedError(
                f"AiM {direction} {name}: expected a full-rank slice"
            )
        fixed = [role for item, role in zip(index, axes) if isinstance(item, int)]
        position = next((item for item in index if isinstance(item, int)), None)
        if any(isinstance(item, tuple) for item in index):
            raise NotImplementedError(f"AiM {direction} {name}: bounded matrix slices")
        cpr = group.partitions
        base = self.row_of_key[contraction["matrix"]]
        mapping = contraction["mapping"]
        if direction == "scatter" and mapping == "row_packed" and fixed == ["output"]:
            aligned = _ceil(shape["reduction"], geometry.lanes) * geometry.lanes
            pack = min(shape["batches"], geometry.row_elements // aligned)
            batch_groups = _ceil(shape["batches"], pack)
            tile = cpr * geometry.banks
            # A carried owner map places output row `position` at its slot.
            slot = contraction.get("row_slots", range(position + 1))[position]
            output_tile, in_tile = divmod(slot, tile)
            local_channel, bank = divmod(in_tile, geometry.banks)
            records = []
            for batch_group in range(batch_groups):
                packed = min(pack, shape["batches"] - batch_group * pack)
                row = base + output_tile * batch_groups + batch_group
                for replica in range(group.replicas):
                    records.append(
                        (
                            AimHostTransfer(
                                "write",
                                channel=replica * cpr + local_channel,
                                bank=bank,
                                row=row,
                                bursts=packed * aligned // geometry.lanes,
                                name=f"{name}.group{batch_group}.r{replica}",
                                **identity,
                            ),
                            None,
                        )
                    )
            return records
        if (
            direction == "broadcast"
            and mapping == "channels"
            and fixed == ["reduction"]
        ):
            # Element `position` of a reduction split P ways lives in partition
            # position % P at local index position // P; only the channels of
            # that partition's accumulators (local channel c with c % P == p)
            # receive it.  P == 1 is the unpartitioned layout.
            parts = group.reduction_partitions
            storage = shape["storage_extent"] or shape["reduction"]
            storage_rows = _ceil(_ceil(storage, parts), geometry.row_elements)
            output_tiles = _ceil(shape["outputs"], geometry.banks)
            partition, local = divmod(position, parts)[::-1]
            if parts == 1:
                channels = tuple(range(group.replicas * cpr))
            else:
                if cpr % parts:
                    raise NotImplementedError(
                        f"AiM {direction} {name}: {parts} reduction partitions "
                        f"do not divide {cpr} channels per replica"
                    )
                channels = tuple(
                    replica * cpr + channel
                    for replica in range(group.replicas)
                    for channel in range(
                        partition, min(cpr, shape["batches"] * parts), parts
                    )
                )
            return [
                (
                    AimAllBankWrite(
                        row=base + local // geometry.row_elements,
                        rows=output_tiles,
                        row_stride=storage_rows,
                        copies=_ceil(shape["batches"] * parts, cpr),
                        copy_row_stride=output_tiles * storage_rows,
                        channels=channels,
                        name=name,
                        **identity,
                    ),
                    None,
                )
            ]
        raise NotImplementedError(
            f"AiM host {direction} of {name} into a {mapping} layout with fixed "
            f"{fixed} axes"
        )

    def records(self, host_moves, ctx):
        """Ordered (AimOp, sbk_banks) records for the whole program."""
        if self.schedule is None:
            records = []
            for group in self.groups:
                records.extend(self._launch_records(group))
            return records

        steps = list(self.schedule.steps)
        host_moves = list(host_moves or ())
        records = []
        pending: list = []  # coalescible distributed transfers
        gather_run = False

        def flush():
            nonlocal gather_run
            for transfer in pending:
                records.append((AimDistributedHostTransfer(**transfer), None))
            pending.clear()
            if gather_run:
                records.append((AimSync(name="host_gather_split"), None))
            gather_run = False

        for position, step in enumerate(steps):
            if isinstance(step, LaunchStep):
                flush()
                group = self.by_id[step.group_id]
                records.extend(self._launch_records(group))
                records.extend(self._relocations(group, position, steps))
                continue
            resolved = host_moves[step.host_index]
            verb, _handle = resolved.move.emit(ctx)
            role = resolved.buffer_role
            base = getattr(role, "base", None) if not isinstance(role, str) else None
            index = None
            text = str(role)
            if base is None:
                base, index = _parse_buffer_role(text)
            if base not in self.key_by_name:
                raise NotImplementedError(f"AiM host {verb} of unknown buffer {text!r}")
            key = self.key_by_name[base]
            group, usage = self._host_usage(key, position, steps, verb)
            if verb == "gather" and not gather_run:
                flush()
            elif verb != "gather" and gather_run:
                flush()
            if usage.kind == "matrix":
                flush()
                records.extend(
                    self._matrix_records(verb, group, usage, index, text, key)
                )
                continue
            if verb == "broadcast":
                raise NotImplementedError(f"AiM broadcast of {text} into a slot layout")
            elements = self._token_extent(key, index)
            if group.split_replicas:
                if index is not None:
                    raise NotImplementedError(
                        f"AiM host {verb} of {text}: a sliced transfer into a "
                        "work-item-split elementwise layout"
                    )
                elements = _ceil(elements, group.replicas)
            transfer = dict(
                direction="read" if verb == "gather" else "write",
                elements=elements,
                replicas=group.replicas,
                channels_per_replica=group.partitions,
                row=group.row if group.kind != "contraction" else self.row_of_key[key],
                bank_stride=usage.stride,
                bank_offset=usage.slot,
                copies=usage.copies,
                name=text,
                buffer_refs=(key,),
                buffer_slices=(index,),
                group_id=group.group_id,
            )
            last = pending[-1] if pending else None
            if (
                last is not None
                and all(
                    last[field] == transfer[field]
                    for field in (
                        "direction",
                        "elements",
                        "replicas",
                        "channels_per_replica",
                        "row",
                        "bank_stride",
                        "group_id",
                    )
                )
                and last["bank_offset"] + last["copies"] == transfer["bank_offset"]
            ):
                merged = pending[-1]
                merged["copies"] += transfer["copies"]
                merged["name"] = f"{merged['name']}+{text}"
                merged["buffer_refs"] += (key,)
                merged["buffer_slices"] += (index,)
            else:
                for earlier in pending:
                    records.append((AimDistributedHostTransfer(**earlier), None))
                pending.clear()
                pending.append(transfer)
            gather_run = verb == "gather"
        flush()
        return records


def _parse_buffer_role(text):
    """``name`` or ``name[i, :, a:b]`` -> (name, normalized index or None)."""
    if "[" not in text:
        return text, None
    base, rest = text.split("[", 1)
    body = rest.rstrip("]")
    index = []
    for item in body.split(","):
        item = item.strip()
        if item == ":":
            index.append(None)
        elif ":" in item:
            start, stop = item.split(":")
            index.append((int(start or 0), int(stop)))
        else:
            try:
                index.append(int(item))
            except ValueError:
                return text, None
    return base, tuple(index)


def _aim_runtime_segments(commands) -> tuple[tuple[str, int, tuple[str, ...]], ...]:
    """The single pre-terminated AiM stream consumed by Ramulator2."""

    lines = tuple(map(str, commands))
    if not lines:
        raise ValueError("AiM runtime requires at least one terminated segment")
    eoc_positions = [
        index for index, line in enumerate(lines) if line.strip() == "AiM EOC"
    ]
    if eoc_positions != [len(lines) - 1]:
        raise ValueError("AiM runtime segment must contain exactly one trailing EOC")
    return (("program", 1, lines),)


# ``GDDR6_AiM_org`` in aim_simulator ``src/dram/impl/GDDR6.cpp`` (commit
# 0f28a07): {Ch 32, Bg 4, Ba 4, Ro 16K, Co 1K}.  The ramulator2 build fixes
# this organization; trace channel masks and bank/row fields assume it.
_SIMULATOR_ORGS = {
    "GDDR6_AiM_org": {
        "channels": 32,
        "bank_groups": 4,
        "banks": 16,
        "rows": 1 << 14,
        "row_elements": 1 << 10,
    },
}


def _check_simulator_geometry(target, yaml_cfg: Path) -> None:
    """Refuse a trace whose target geometry differs from the simulator's."""
    match = re.search(r"org:\s*\n\s*preset:\s*(\S+)", yaml_cfg.read_text())
    preset = match.group(1) if match else None
    org = _SIMULATOR_ORGS.get(preset)
    if org is None:
        raise ValueError(
            f"AiM simulator config {yaml_cfg} uses DRAM org preset {preset!r}; "
            f"only {sorted(_SIMULATOR_ORGS)} is described"
        )
    geometry = _target_geometry(target)
    mismatched = {
        field: (getattr(geometry, field), expected)
        for field, expected in org.items()
        if getattr(geometry, field) != expected
    }
    if mismatched:
        raise ValueError(
            f"AiM target geometry differs from simulator preset {preset} "
            f"(target, simulator): {mismatched}; its channel masks and bank "
            "addresses are not executable there (use backend='virtual')"
        )


def _run_aim(compiled: "Compiled", **inputs) -> RunResult:
    """Run a compiled AiM artifact through ramulator2 in Docker.

    `compiled.cmds` is a list of already-terminated text trace lines. We
    serialize those exact candidate-owned segments and invoke the ramulator2
    binary inside the `aim-simulator-build` Docker image (`TENON_AIM_SIM_IMAGE`).
    Each trace is written to a private temp directory mounted read-only at
    `/trace`, so the simulator checkout is never written. The simulator's
    YAML stdout carries per-channel `memory_system_cycles`; we parse
    out the max across channels as the headline cycle count.

    ramulator2 is a timing simulator: it replays the command trace and
    computes no data. The result is ``kind="simulator-timing-only"`` with no
    outputs, and the caller's output arrays are never written.
    """
    reason = _aim_unavailable_reason()
    if reason is not None:
        raise SimulatorUnavailable("aim", reason)
    root = _aim_root()
    yaml_cfg = root / "test" / "example.yaml"
    _check_simulator_geometry(compiled.target, yaml_cfg)

    cfg_rel = yaml_cfg.relative_to(root)

    image = _aim_sim_image()

    def run_segment(lines: list[str]) -> tuple[int, str, int]:
        with tempfile.TemporaryDirectory(prefix="tenon-aim-") as trace_dir:
            trace_path = Path(trace_dir) / "segment.trace"
            trace_path.write_text("".join(str(line) + "\n" for line in lines))
            try:
                proc = subprocess.run(
                    [
                        "docker",
                        "run",
                        "--rm",
                        "-v",
                        f"{root}:/work",
                        "-v",
                        f"{trace_dir}:/trace:ro",
                        image,
                        "bash",
                        "-c",
                        f"cd /work && ./build/ramulator2 -f {cfg_rel} "
                        f"-t /trace/{trace_path.name}",
                    ],
                    capture_output=True,
                    timeout=600,
                    check=False,
                )
            except (subprocess.SubprocessError, OSError) as exc:
                raise RuntimeError(f"AiM ramulator2 invocation failed: {exc}") from exc
        stdout = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")
        combined = stdout + ("\n" + stderr if stderr else "")
        if proc.returncode != 0:
            # A failed run may still print partial statistics; never report
            # cycles from it.
            raise RuntimeError(
                f"AiM ramulator2 exited with status {proc.returncode}; no cycles "
                "are reported from a failed run; tail: " + combined[-400:]
            )
        values = [
            int(m) for m in re.findall(r"memory_system_cycles:\s*(\d+)", combined)
        ]
        if not values:
            raise RuntimeError(
                "AiM ramulator2 returned but stdout missing "
                "'memory_system_cycles: ...' line; tail: " + combined[-400:]
            )
        return max(values), combined, proc.returncode

    # Compact traces carry independently dispatched GEMV templates. Profile
    # every native segment once, then apply its compile-time repeat count. All
    # base values are direct Ramulator2 measurements from this invocation.
    segments = tuple(getattr(compiled, "runtime_segments", ()))
    if not segments:
        segments = _aim_runtime_segments(compiled.cmds)

    total_cycles = 0
    logs: list[str] = []
    returncodes: list[int] = []
    cache: dict[tuple[str, ...], tuple[int, str, int]] = {}
    for label, repeat, lines in segments:
        key = tuple(lines)
        if key not in cache:
            cache[key] = run_segment(lines)
        raw_cycles, log, returncode = cache[key]
        total_cycles += raw_cycles * repeat
        returncodes.append(returncode)
        logs.append(
            f"=== {label}; repeat={repeat}; raw_cycles={raw_cycles}; "
            f"composed_cycles={raw_cycles * repeat} ===\n{log}"
        )
    cycles = total_cycles
    combined = "\n".join(logs)
    return RunResult(
        kind="simulator-timing-only",
        cycles=cycles,
        stdout=(
            "AiM ramulator2 is timing-only: cycles are simulated, no numerical "
            "outputs exist and output arrays are not written\n" + combined
        ),
        backend="aim",
        extra={
            "returncode": max(returncodes, default=0),
            "segments": len(segments),
            "unique_segments": len(cache),
            # Host steps of the program (e.g. a reduction-partition merge) are
            # not simulated: they are listed, never added to `cycles`.
            "host_steps": [
                step.manifest()
                for step in getattr(
                    getattr(compiled, "layout_ctx", None), "host_steps", ()
                )
            ],
            **_row_map_extra(compiled),
        },
    )


def _row_map_extra(compiled) -> dict:
    """``{"row_map": [...]}`` when a contraction places rows by a carried
    owner map: the host-side result mapping (row -> frontier slot ->
    channel, bank) of every such contraction; empty otherwise."""
    manifest = getattr(getattr(compiled, "layout_ctx", None), "lowering_manifest", ())
    maps = [
        {"name": entry.get("name"), **entry["row_map"]}
        for entry in manifest or ()
        if isinstance(entry, dict) and "row_map" in entry
    ]
    return {"row_map": maps} if maps else {}
