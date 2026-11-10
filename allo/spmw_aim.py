# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SK hynix AiM codegen context and ramulator2 runtime (moved from spmw_codegen)."""

from __future__ import annotations

import functools

import math
import os
import re
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

from .spmw_autoschedule import _matcher_work_scope
from .spmw_plan import HostStep, LaunchStep
from .spmw_codegen import (
    CodegenContext,
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
    AimHostStage,
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
    """The shared `spmw_candidate.value_axes`."""
    from .spmw_candidate import value_axes

    return value_axes(access, labels)


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
            orientation = extra.get("aim_orientation") or {}
            matrix = orientation.get("matrix") or (
                fact.matrix.memref_name if fact.matrix is not None else None
            )
            vector = orientation.get("vector") or (
                fact.vector.memref_name if fact.vector is not None else None
            )
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
                # A GB-resident vector is supplied by the launch itself: the
                # lowering copies it into the GB (``WR_GB``) from a GPR the
                # host loads before the traced program (the manifest states
                # it), not by a ``host_xfer`` step of the host program.  A
                # bank-resident vector is scattered by the host program.
                transfers[memref] = CandidateTransfer(
                    memref,
                    "in",
                    (
                        ("host_gpr_preload", "gpr", "WR_GB", "gb")
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
                if (
                    memref == fact.result.memref_name
                    and match.extra.get("accumulator_init") == "zero"
                ):
                    # Reads the element it writes after a proven zero
                    # initializer: dead incoming contents, not an input.
                    continue
                inputs.append(memref)
                storage = _aim_storage_name(placement.placements.get(memref)) or home
                layouts[memref] = CandidateValueLayout(
                    memref, _aim_axes(access, labels), storage
                )
                # EWADD operands live in the GPR the host loads before the
                # traced program (no lowered ingress): declared as such.
                transfers[memref] = CandidateTransfer(
                    memref,
                    "in",
                    (
                        ("host_gpr_preload", storage)
                        if storage == "gpr"
                        else ("host_scatter", storage)
                    ),
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
    from .spmw_candidate import EFFECTS_DERIVERS, ROUTE_VOCABULARY

    EFFECTS_DERIVERS["aim"] = _derive_aim_effects
    # The channel budget: the replica and token partition domains
    # (`spmw_knobs._budgeted_channels`) and `AimCtx.check_realization`.
    from .spmw_resources import register_budget_unit

    register_budget_unit("aim", "channel")
    # The GPR preload the host performs before the traced program.
    ROUTE_VOCABULARY.setdefault("aim", set()).add("host_gpr_preload")


_register_effects_deriver()


_HOST_TRANSFER_RECORDS = (AimDistributedHostTransfer, AimHostTransfer, AimAllBankWrite)


def _kernel_record(ops, group):
    """The launch record of ``group`` (its EWMUL/EWADD or MAC record)."""
    names = set(group.functions) | {group.name}
    return next(
        (
            op
            for op in ops
            if isinstance(op, (AimElementwise, AimContraction)) and op.name in names
        ),
        None,
    )


def _check_host_record_physical(op, group, program, ref, index, kernel, value, slot):
    """One host record against the physical layout the launched kernel
    record addresses: its rows, channels, replicas, bank slot, stride and
    copies, and the slice frame of the logical vector it moves.  ``slot`` is
    the bank slot the record's next constituent occupies (a coalesced record
    moves consecutive slots); returns the slot after this constituent."""
    from .spmw_plan import flat_range, slice_region

    geometry = program.geometry
    usage = group.usages.get(ref)
    if usage is None or kernel is None:
        raise RuntimeError(
            f"AiM realization moves {value!r} for group {group.name!r}, whose "
            "launch holds it in no bank layout"
        )
    channels = group.replicas * group.partitions
    kernel_channels = (
        kernel.channels_per_replica
        if kernel.channels_per_replica is not None
        else group.partitions
    )
    if isinstance(op, AimDistributedHostTransfer):
        region_first, region_total = program._slot_frame(group, ref)
        if group.split_replicas:
            expected = (0, _ceil(region_total, group.replicas), None)
        else:
            shape = program.shape_by_key.get(ref)
            flat = (
                flat_range(shape, slice_region(shape, index))
                if shape is not None
                else None
            )
            if flat is None:
                raise RuntimeError(
                    f"AiM realization moves a slice {index} of {value!r} that is "
                    "not one contiguous range of the kernel's vector"
                )
            start, count = flat
            expected = (start - region_first, count, region_total)
        realized = (
            op.first_element,
            op.elements,
            op.vector_elements if op.vector_elements is not None else op.elements,
        )
        frame = (expected[0], expected[1], expected[2] or expected[1])
        if realized != frame:
            raise RuntimeError(
                f"AiM realization moves elements [{realized[0]}, "
                f"{realized[0] + realized[1]}) of a {realized[2]}-element vector "
                f"for {value!r}; the slice {index} is [{frame[0]}, "
                f"{frame[0] + frame[1]}) of the {frame[2]}-element region the "
                "kernel's layout holds"
            )
        physical = (
            op.row,
            op.replicas,
            op.channels_per_replica,
            op.bank_stride,
            slot,
        )
        layout = (
            kernel.row,
            kernel.replicas,
            kernel_channels,
            usage.stride,
            usage.slot,
        )
        if physical != layout or slot + usage.copies > op.bank_offset + op.copies:
            raise RuntimeError(
                f"AiM realization addresses {value!r} at (row, replicas, "
                f"channels per replica, bank stride, bank slot) {physical} with "
                f"{op.copies} slot(s) from {op.bank_offset}; the launched kernel "
                f"{kernel.name!r} holds it at {layout} in {usage.copies} slot(s)"
            )
        return slot + usage.copies
    # Matrix row writes and all-bank broadcasts land in the matrix's rows.
    base = program.row_of_key.get(ref)
    rows = program._footprint(group, ref, usage)
    if base is None or kernel.row != base:
        raise RuntimeError(
            f"AiM realization writes {value!r} into rows from {base}; the launched "
            f"kernel {kernel.name!r} reads its matrix from row {kernel.row}"
        )
    if isinstance(op, AimHostTransfer):
        if not base <= op.row < base + rows or op.channel >= channels:
            raise RuntimeError(
                f"AiM realization writes {value!r} at channel {op.channel} row "
                f"{op.row}; the kernel's matrix occupies rows [{base}, {base + rows}) "
                f"on {channels} channel(s)"
            )
        return slot
    touched = (
        op.rows * op.row_stride * op.copies
        if op.copy_row_stride is None
        else (op.row_stride * (op.rows - 1) + op.copy_row_stride * (op.copies - 1) + 1)
    )
    if not base <= op.row < base + rows or op.row + touched > base + rows:
        raise RuntimeError(
            f"AiM realization broadcasts {value!r} into rows [{op.row}, "
            f"{op.row + touched}); the kernel's matrix occupies rows "
            f"[{base}, {base + rows})"
        )
    if op.channels is not None and not set(op.channels) <= set(range(channels)):
        raise RuntimeError(
            f"AiM realization broadcasts {value!r} to channels {list(op.channels)}; "
            f"the kernel runs on {channels} channel(s)"
        )
    return slot


def _check_host_transfers(program, group, effects, ops, storage_of) -> None:
    """The host transfers lowered for ``group`` against its candidate's
    declared transfers and the physical layout its launched kernel addresses.

    Every host record the derivation lowered for the group carries the
    buffer keys it moves (`buffer_refs`), their slices and the group; each is
    matched to the declared transfer of the work-body value with the same
    retained identity (the body's operands and results, never a name): its
    direction (host write: ``in``/``inout``; host read: ``out``/``inout``)
    and declared landing (the banks) are compared with the declaration, and
    its rows, channels, replicas, bank slot, stride, copies and slice frame
    with the kernel record the launch emits and the target's slot
    conventions (`_check_host_record_physical`).  Availability of what each
    launch reads is the shared `spmw_plan.check_transfer_availability`.
    """
    records = [
        op
        for op in ops
        if isinstance(op, _HOST_TRANSFER_RECORDS) and op.group_id == group.group_id
    ]
    if not records:
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
    kernel = _kernel_record(ops, group)
    bursts: dict = {}
    for op in records:
        if not op.buffer_refs or len(op.buffer_refs) != len(op.buffer_slices):
            raise RuntimeError(
                f"AiM realization lowered the host transfer {op.name!r} without a "
                "retained buffer identity"
            )
        direction = "write" if isinstance(op, AimAllBankWrite) else op.direction
        slot = getattr(op, "bank_offset", 0)
        for ref, index in zip(op.buffer_refs, op.buffer_slices):
            transfer = declared_by_ref.get(ref)
            if transfer is None:
                raise RuntimeError(
                    f"AiM realization moves {label(ref, ref)!r} between host and "
                    f"the device for group {group.name!r}; the candidate declares "
                    f"transfers of {sorted(t.value for t in effects.transfers)}"
                )
            allowed = {"in", "inout"} if direction == "write" else {"out", "inout"}
            reupload = direction == "write" and getattr(op, "reupload", False)
            if reupload:
                # The host rewrites a produced value's device copy after the
                # launch: it binds to the declared output (its device home).
                allowed = {"out", "inout"}
            if transfer.direction not in allowed:
                raise RuntimeError(
                    f"AiM realization {direction}s {label(ref, ref)!r} "
                    f"({transfer.value!r}){' after its launch' if reupload else ''}; "
                    f"the candidate declares it {transfer.direction!r}"
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
            slot = _check_host_record_physical(
                op, group, program, ref, index, kernel, transfer.value, slot
            )
            if isinstance(op, AimHostTransfer):
                expected = program._token_extent(ref, index)
                bursts.setdefault((ref, index, transfer.value), [0, expected])
                bursts[(ref, index, transfer.value)][0] += op.bursts
            elif isinstance(op, AimAllBankWrite):
                parts = group.reduction_partitions
                accumulators = shape["outputs"] * shape["batches"] if shape else None
                expected = program._token_extent(ref, index)
                if shape is None or (
                    expected != accumulators
                    or op.rows != _ceil(shape["outputs"], geometry.banks)
                    or op.copies != _ceil(shape["batches"] * parts, group.partitions)
                ):
                    raise RuntimeError(
                        f"AiM realization broadcasts {transfer.value!r} to "
                        f"{op.rows} row(s) x {op.copies} copies; the slice {index} "
                        f"feeds {accumulators} accumulator(s) ({expected} elements)"
                    )
        if isinstance(op, AimDistributedHostTransfer) and (
            slot != op.bank_offset + op.copies
        ):
            raise RuntimeError(
                f"AiM realization moves {op.copies} bank slot(s) from "
                f"{op.bank_offset} in {op.name!r}; its buffers occupy slots up to "
                f"{slot}"
            )
    for (ref, index, value), (total, expected) in bursts.items():
        per_replica = total * geometry.lanes // max(group.replicas, 1)
        batches = shape["batches"] if shape else 1
        if not expected <= per_replica <= expected + batches * (geometry.lanes - 1):
            raise RuntimeError(
                f"AiM realization writes {per_replica} element(s) of {value!r} "
                f"per replica; the slice {index} holds {expected}"
            )


class AimCtx(CodegenContext):
    # The AiM program represents, prices and binds declared host stages
    # (``allo.host_compute``); ``compile_for_target`` refuses them elsewhere.
    supports_host_compute = True
    # The lowering stages a single work item's matrix rows and results from
    # row zero (`AimContraction.row` counts whole items); a constant result
    # offset of one item is not realized and is refused before selection
    # (X-42).  Multi-item kernels tile the result through the work grid.
    REALIZES_RESULT_OFFSETS = False
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
            if isinstance(operation, (AimHostReduction, AimHostStage)):
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
        # The complete emission input, retained for the realization check:
        # each record with the bank payload the emission expanded it with
        # (``None`` for an all-bank launch, the banks of a single-bank one).
        self._lowered_records = tuple((op, banks) for op, banks in records)
        self.lowering_manifest = lowerer.operations
        self.row_regions = program.regions
        self.cmds.extend(lines)
        if program.groups:
            self.emit_eoc()

    def _check_launch_inventory(self, program, layouts, ops) -> None:  # noqa: C901
        """The lowered records against the program's launch occurrences.

        The expected inventory comes from the retained facts and the selected
        candidates, never from the emitted records: every scheduled launch
        occurrence of a group (each ``LaunchStep``; without a schedule, one
        occurrence per group) lowers exactly one primary record per matched
        contraction of the group (or one elementwise record), and, for a
        contraction whose candidate declares P > 1 reduction partitions, one
        host reduction of that contraction with that P, before any later
        stage or launch begins.  The records of one occurrence are
        contiguous in the realized sequence (host transfers, bank copies and
        syncs are owned by other steps and not ordered here); a declared host
        stage sits between occurrences exactly where the schedule puts it.
        A duplicated, missing or displaced record is a realization failure.
        """
        from .spmw_candidate import candidate_effects
        from .spmw_plan import HostComputeStep, LaunchStep

        schedule = getattr(program, "schedule", None)
        steps = list(getattr(schedule, "steps", ()) or ())
        groups = {group.group_id: group for group in program.groups}
        emitted_payload = {
            id(op): banks for op, banks in getattr(self, "_lowered_records", ())
        }

        # Declared partitions per group: the selected candidate's launch
        # record (``covers[2]``), else the group's own derived knob.
        declared_partitions = {}
        for layout in layouts:
            gid = getattr(self, "_group_of_layout", {}).get(id(layout))
            effects = candidate_effects(layout)
            if gid is None or effects is None:
                continue
            launches = list(effects.launches)
            if len(launches) == 1 and len(launches[0].covers) >= 3:
                try:
                    declared_partitions[gid] = int(launches[0].covers[2])
                except (TypeError, ValueError):
                    pass

        def expectation(gid):
            group = groups[gid]
            if group.kind == "contraction":
                names = [c["match"].func_name for c in group.contractions]
                partitions = declared_partitions.get(
                    gid, int(group.reduction_partitions)
                )
                merges = {name: partitions for name in names} if partitions > 1 else {}
                return {"primaries": names, "merges": merges}
            return {"primaries": [group.name], "merges": {}}

        if steps:
            tokens = []
            occurrence = {}
            for step in steps:
                if isinstance(step, LaunchStep):
                    occurrence[step.group_id] = occurrence.get(step.group_id, 0) + 1
                    tokens.append(
                        ("launch", step.group_id, occurrence[step.group_id] - 1)
                    )
                elif isinstance(step, HostComputeStep):
                    tokens.append(("stage", step.name))
        else:
            tokens = [("launch", group.group_id, 0) for group in program.groups]

        primary_group = {}
        for group in program.groups:
            if group.kind == "contraction":
                for c in group.contractions:
                    primary_group[c["match"].func_name] = group.group_id
            else:
                primary_group[group.name] = group.group_id

        state = {
            "index": -1,
            "pending_primaries": [],
            "pending_merges": {},
            "done_primaries": set(),
            # name -> the primary record of this occurrence: a host reduction
            # is bound to its producing contraction's whole work domain.
            "primary_records": {},
            # The launch-owned records of the open occurrence, in order.
            "occurrence_records": [],
        }

        def differing_fields(realized, expected):
            from dataclasses import fields as dataclass_fields

            names = []
            for field in dataclass_fields(expected):
                if getattr(realized, field.name, None) != getattr(
                    expected, field.name, None
                ):
                    names.append(
                        f"{field.name}: realized {getattr(realized, field.name, None)!r}, "
                        f"expected {getattr(expected, field.name, None)!r}"
                    )
            return names

        def compare_with_derivation(token, realized_records):
            """The occurrence's launch-owned records (contractions, their host
            reductions, elementwise records) must equal, record by record,
            the records the retained program and the selected decision
            derive for one launch of the group (``_launch_records``, computed
            here afresh from the group's facts, never from the records under
            check): activation, reuse scheduling, GPR addressing, channels,
            rows, partitions and every other emitted field included."""
            group = groups[token[1]]
            expected = list(program._launch_records(group))
            if len(realized_records) != len(expected):
                raise RuntimeError(
                    f"AiM realization of {describe(token)} lowers {len(realized_records)} "
                    f"launch record(s) {[type(r).__name__ for r in realized_records]}; the "
                    f"retained program derives {len(expected)} "
                    f"{[type(e).__name__ for e, _b in expected]}"
                )
            for realized, (derived, derived_banks) in zip(realized_records, expected):
                if type(realized) is not type(derived):
                    raise RuntimeError(
                        f"AiM realization of {describe(token)} lowers a "
                        f"{type(realized).__name__} where the retained program derives a "
                        f"{type(derived).__name__}"
                    )
                if realized != derived:
                    raise RuntimeError(
                        f"AiM realization of {describe(token)} lowers "
                        f"{getattr(realized, 'name', type(realized).__name__)!r} with "
                        + "; ".join(differing_fields(realized, derived))
                        + "; the retained program under the selected decision derives otherwise"
                    )
                # The emission input is the record *and* its bank payload: the
                # banks the emission expanded MAC_ABK into MAC_SBK with.  The
                # payload the emission actually used is compared with the
                # derived one, and must itself be a set of distinct banks of
                # the device.
                if id(realized) not in emitted_payload:
                    raise RuntimeError(
                        f"AiM realization of {describe(token)}: the emission input of "
                        f"{getattr(realized, 'name', type(realized).__name__)!r} was not retained"
                    )
                realized_banks = emitted_payload[id(realized)]
                normalized = (
                    None
                    if realized_banks is None
                    else tuple(int(b) for b in realized_banks)
                )
                expected_banks = (
                    None
                    if derived_banks is None
                    else tuple(int(b) for b in derived_banks)
                )
                if normalized != expected_banks:
                    raise RuntimeError(
                        f"AiM realization of {describe(token)} expands "
                        f"{getattr(realized, 'name', type(realized).__name__)!r} over banks "
                        f"{normalized}; the retained program under the selected decision "
                        f"derives {expected_banks}"
                    )
                if normalized is not None and (
                    len(set(normalized)) != len(normalized)
                    or any(b < 0 or b >= program.geometry.banks for b in normalized)
                ):
                    raise RuntimeError(
                        f"AiM realization of {describe(token)} expands "
                        f"{getattr(realized, 'name', type(realized).__name__)!r} over banks "
                        f"{normalized}, not distinct banks of the device"
                    )

        def describe(token):
            return (
                f"host stage {token[1]!r}"
                if token[0] == "stage"
                else f"launch {token[2]} of group {groups[token[1]].name!r}"
            )

        def close_current():
            if state["index"] < 0:
                return
            token = tokens[state["index"]]
            if token[0] == "launch" and (
                state["pending_primaries"] or state["pending_merges"]
            ):
                raise RuntimeError(
                    f"AiM realization: {describe(token)} is incomplete before the next "
                    f"step: missing primary records {state['pending_primaries']} and host "
                    f"reductions {sorted(state['pending_merges'])}"
                )
            if token[0] == "launch":
                compare_with_derivation(token, list(state["occurrence_records"]))
            state["occurrence_records"] = []

        def open_next(kind, gid=None, what=""):
            close_current()
            index = state["index"] + 1
            if index >= len(tokens):
                raise RuntimeError(
                    f"AiM realization lowered {what} after the last scheduled step"
                )
            token = tokens[index]
            if token[0] != kind or (kind == "launch" and token[1] != gid):
                raise RuntimeError(
                    f"AiM realization lowered {what} where the schedule expects "
                    f"{describe(token)}"
                )
            state["index"] = index
            if kind == "launch":
                expected = expectation(gid)
                state["pending_primaries"] = list(expected["primaries"])
                state["pending_merges"] = dict(expected["merges"])
                state["done_primaries"] = set()
                state["primary_records"] = {}
            else:
                state["pending_primaries"], state["pending_merges"] = [], {}
                state["done_primaries"] = set()
                state["primary_records"] = {}

        def current_launch(gid):
            return (
                state["index"] >= 0
                and tokens[state["index"]][0] == "launch"
                and tokens[state["index"]][1] == gid
            )

        for op in ops:
            if isinstance(op, AimHostStage):
                open_next("stage", what=f"host stage {op.stage!r}")
                continue
            if isinstance(op, (AimContraction, AimElementwise)):
                name = str(op.name)
                gid = primary_group.get(name)
                if gid is None:
                    raise RuntimeError(
                        f"AiM realization lowered a primary record {name!r} that no "
                        "matched group of the program declares"
                    )
                if not (current_launch(gid) and name in state["pending_primaries"]):
                    open_next("launch", gid, what=f"primary record {name!r}")
                    if name not in state["pending_primaries"]:
                        raise RuntimeError(
                            f"AiM realization lowered primary record {name!r} twice for "
                            f"{describe(tokens[state['index']])}"
                        )
                state["pending_primaries"].remove(name)
                state["done_primaries"].add(name)
                state["primary_records"][name] = op
                state["occurrence_records"].append(op)
                continue
            if isinstance(op, AimHostReduction):
                source = str(op.source)
                gid = primary_group.get(source)
                if gid is None or not current_launch(gid):
                    raise RuntimeError(
                        f"AiM realization lowered the host reduction of {source!r} outside "
                        "the launch occurrence that produces its partials"
                    )
                if source not in state["pending_merges"]:
                    raise RuntimeError(
                        f"AiM realization lowered a host reduction of {source!r} that "
                        f"{describe(tokens[state['index']])} does not declare (or a second one)"
                    )
                if source not in state["done_primaries"]:
                    raise RuntimeError(
                        f"AiM realization lowered the host reduction of {source!r} before "
                        "its contraction"
                    )
                expected = state["pending_merges"].pop(source)
                if int(op.partitions) != int(expected):
                    raise RuntimeError(
                        f"AiM realization merges {op.partitions} partials of {source!r}; the "
                        f"candidate declares {expected}"
                    )
                # The reduction's whole work domain is its producing
                # contraction's, in this occurrence: outputs, batches and
                # replicas (the host reads P x batches x replicas partial
                # vectors of `outputs` elements) and the contraction's own
                # partition count.  The contraction record is itself checked
                # against the candidate's declared launch; the merge's price
                # (`host_reduction_cycles`) follows from these fields, so the
                # lowered host cost equals the priced graph's.
                producer = state["primary_records"].get(source)
                if not isinstance(producer, AimContraction):
                    raise RuntimeError(
                        f"AiM realization lowered the host reduction of {source!r} "
                        "without the contraction that produces its partials"
                    )
                realized = (
                    int(op.outputs),
                    int(op.batches),
                    int(op.replicas),
                    int(op.partitions),
                )
                produced = (
                    int(producer.outputs),
                    int(producer.batches),
                    int(producer.replicas),
                    int(producer.reduction_partitions),
                )
                if realized != produced:
                    raise RuntimeError(
                        f"AiM realization merges (outputs, batches, replicas, partitions) "
                        f"{realized} of {source!r}; its contraction produces {produced}"
                    )
                state["occurrence_records"].append(op)
                continue
            # Host transfers, bank copies and syncs are owned by other steps.
        close_current()
        if state["index"] != len(tokens) - 1:
            missing = tokens[state["index"] + 1 :]
            raise RuntimeError(
                "AiM realization lowered no records for "
                + ", ".join(describe(t) for t in missing)
            )

    def _check_host_stages(self, program, layouts, ops) -> None:  # noqa: C901
        """The lowered host stages against the schedule's declared stages:
        one `AimHostStage` per declared stage (their order among the launch
        occurrences is checked by `_check_launch_inventory`), each with the
        declared label, operation and buffers and the very computation the
        host program recorded (the same callable object); each stage input
        produced before the stage (by an earlier launch whose candidate
        declares it as an output, by an earlier stage, or a region input no
        launch of the program writes); each stage output reaching a later
        launch through that launch's declared host ingress.
        """
        from .spmw_autoschedule import _matcher_work_scope
        from .spmw_candidate import candidate_effects
        from .spmw_plan import HostComputeStep, LaunchStep

        schedule = getattr(program, "schedule", None)
        steps = list(getattr(schedule, "steps", ()) or ())
        declared = [s for s in steps if isinstance(s, HostComputeStep)]
        lowered = [op for op in ops if isinstance(op, AimHostStage)]
        if len(lowered) != len(declared):
            raise RuntimeError(
                f"AiM realization lowered {len(lowered)} host stage(s) for "
                f"{len(declared)} declared"
            )
        if not declared:
            return

        # --- per stage: identity, computation, inputs and outputs ---
        effects_of = {}
        for layout in layouts:
            group_id = getattr(self, "_group_of_layout", {}).get(id(layout))
            effects = candidate_effects(layout)
            if group_id is not None and effects is not None:
                effects_of[group_id] = effects

        def local_names(launch, region_name):
            # The candidate's transfers name the kernel's own buffers.  The
            # launch binds the region's buffers to the kernel's parameters
            # positionally (allo.launch(kernel, *operands) in the kernel's
            # signature order, checked by launch_schedule); the retained
            # value identities confirm it when present.
            names = set()
            matches = [
                m
                for m in program.trace.matches
                if _matcher_work_scope(m).group_id == launch.group_id
            ]
            for match in matches:
                facts = match.extra.get("program_facts")
                parameters = tuple(getattr(facts, "parameter_names", ()) or ())
                for operand, local in zip(launch.operands, parameters):
                    if operand == region_name:
                        names.add(local)
            key = program.key_by_name.get(region_name)
            if key is not None:
                for match in matches:
                    for operand in match.operands:
                        if operand.value_ref is not None and operand.value_ref == key:
                            names.add(operand.memref_name)
                    if (
                        match.result_value_ref is not None
                        and match.result_value_ref == key
                    ):
                        names.add(match.result_memref_name)
            return names

        def declares(launch_steps, region_name, directions, ingress=False):
            for launch in launch_steps:
                names = local_names(launch, region_name)
                for t in getattr(effects_of.get(launch.group_id), "transfers", ()):
                    if t.value not in names or t.direction not in directions:
                        continue
                    if ingress and not _host_ingress_route(t.route):
                        continue
                    return True
            return False

        def written_by_a_launch(region_name):
            key = program.key_by_name.get(region_name)
            if key is None:
                return False
            return any(
                key in group.usages and group.usages[key].store
                for group in program.groups
            )

        produced_by_stages = set()
        for step, record in zip(declared, lowered):
            if (record.stage, record.operation, record.inputs, record.outputs) != (
                step.name,
                step.operation,
                tuple(step.inputs),
                tuple(step.outputs),
            ):
                raise RuntimeError(
                    f"AiM realization of host stage {step.name!r} differs from the "
                    f"schedule: lowered {record.stage!r} ({record.operation}) "
                    f"{record.inputs} -> {record.outputs}, declared ({step.operation}) "
                    f"{step.inputs} -> {step.outputs}"
                )
            if record.function is not step.function:
                raise RuntimeError(
                    f"AiM realization of host stage {step.name!r} carries a computation "
                    "other than the one the host program recorded"
                )
            position = steps.index(step)
            before = [s for s in steps[:position] if isinstance(s, LaunchStep)]
            after = [s for s in steps[position + 1 :] if isinstance(s, LaunchStep)]
            for name, _shape, _bits in step.inputs:
                if name in produced_by_stages:
                    continue
                if written_by_a_launch(name):
                    if not declares(before, name, ("out", "inout")):
                        raise RuntimeError(
                            f"AiM realization: host stage {step.name!r} reads {name!r} "
                            "before the launch that produces it declares it as an output"
                        )
                # else: a region input no launch writes, host data from the start.
            for name, _shape, _bits in step.outputs:
                produced_by_stages.add(name)
                if not declares(after, name, ("in", "inout"), ingress=True):
                    raise RuntimeError(
                        f"AiM realization: host stage {step.name!r} produces {name!r}, "
                        "which no later launch receives through a declared host ingress"
                    )

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
        self._check_launch_inventory(program, layouts, ops)
        # The lowered program against the compile's resource budget: no
        # record may occupy more channels than the cap.
        cap = (getattr(self, "resource_budget", None) or {}).get("channel")
        if cap is not None:
            for entry in manifest:
                if not isinstance(entry, dict):
                    continue
                used = entry.get("active_channel_count")
                if used is None and entry.get("active_channels"):
                    used = len(entry["active_channels"])
                if used is not None and int(used) > int(cap):
                    raise RuntimeError(
                        f"AiM realization of {entry.get('name')!r} occupies {used} "
                        f"channels; the compile's resource budget allows {cap}"
                    )
        every_launched = getattr(self, "_every_group_launched", True)
        launched = getattr(self, "_launched_groups", set())
        self._check_host_stages(program, layouts, ops)
        for op in ops:
            if isinstance(op, _HOST_TRANSFER_RECORDS) and (
                op.group_id not in groups
                or (not every_launched and op.group_id not in launched)
            ):
                raise RuntimeError(
                    f"AiM realization lowered the host transfer {op.name!r} for no "
                    f"launched kernel group ({op.group_id!r})"
                )
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
                # Every launch's record realizes the retained semantic shape of
                # its contraction under the selected candidate, dimension by
                # dimension (outputs, batches, replicas, reduction and the
                # vector's source), never only the product of outputs and
                # batches: equal flattened work does not give equal batch and
                # output addressing.  A grouped (aggregate) realization is the
                # shape the derivation retained for the group, so it binds
                # the same way.
                for realized in records:
                    (entry,) = [
                        c
                        for c in group.contractions
                        if c["match"].func_name == realized.name
                    ] or [None]
                    if entry is None:
                        raise RuntimeError(
                            f"AiM realization lowered a contraction {realized.name!r} "
                            f"that group {group.name!r} does not derive"
                        )
                    shape = entry["shape"]
                    expected_fields = {
                        "outputs": int(shape["outputs"]),
                        "batches": int(shape["batches"]),
                        "replicas": int(shape["replicas"]),
                        "reduction": int(shape["reduction"]),
                        "input_source": shape["input_source"],
                    }
                    realized_fields = {
                        "outputs": int(realized.outputs),
                        "batches": int(realized.batches),
                        "replicas": int(realized.replicas),
                        "reduction": int(realized.reduction),
                        "input_source": realized.input_source,
                    }
                    if realized_fields != expected_fields:
                        raise RuntimeError(
                            f"AiM realization of {realized.name!r} realizes "
                            f"{realized_fields}; the retained program under the selected "
                            f"candidate gives {expected_fields}"
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
                # The scope the emission realized, from the payload it expanded
                # the record with (not the group's expectation).
                emitted = {
                    id(op): banks for op, banks in getattr(self, "_lowered_records", ())
                }
                if any(id(r) not in emitted for r in records):
                    raise RuntimeError(
                        f"AiM realization of group {group.name!r}: an emission input was "
                        "not retained"
                    )
                payloads = {emitted[id(r)] is None for r in records}
                if len(payloads) != 1:
                    raise RuntimeError(
                        f"AiM realization of group {group.name!r} expands its launches with "
                        "differing bank scopes"
                    )
                scope = "abk" if emitted[id(record)] is None else "sbk"
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
                if record.operation == "mul":
                    # The copies and channels the record realizes are the
                    # candidate's: ``replica_partitions`` channels per copy,
                    # and as many copies as work items unless the candidate
                    # asks for fewer (a replicated grid realized once over
                    # the device, X-19 round 71).
                    requested = (layout.extra or {}).get("replica_partitions")
                    if requested not in (None, "none"):
                        channels = int(requested)
                        total = _target_geometry(self.target).channels
                        expected_copies = max(items, 1)
                        if expected_copies * channels > total:
                            expected_copies = max(1, total // channels)
                        realized = (
                            int(getattr(record, "replicas", 0) or 0),
                            int(getattr(record, "channels_per_replica", 0) or 0),
                        )
                        if realized != (expected_copies, channels):
                            raise RuntimeError(
                                f"AiM realization lowers {group.name!r} as "
                                f"{realized[0]} copy(ies) over {realized[1]} channels each; "
                                f"the candidate declares replica_partitions {channels} for "
                                f"{items} work item(s) ({expected_copies} copy(ies))"
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
        "work_items",
    )

    def __init__(self, group_id, name):
        self.group_id = group_id
        self.name = name
        self.usages = {}
        # Work items of the kernel group; ``replicas`` is the number of copies
        # the lowering realizes (fewer than the work items for a replicated
        # elementwise grid realized once over the device).
        self.work_items = 1
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
        # id(AimBankCopy record) -> the identity of the value it moves.
        self._copy_identity: dict[int, tuple] = {}
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
            copies = len(functions) if replicated else 1
            partitions = group.properties.get("replica_partitions")
            elementwise = not any(match.target_op_name == "MAC" for match in reps)
            if partitions not in (None, "none"):
                partitions = int(partitions)
                if (
                    replicated
                    and elementwise
                    and copies * partitions > self.geometry.channels
                ):
                    # The candidate realizes a replicated elementwise grid with
                    # fewer copies than work items (one copy over the whole
                    # device for ``partitions == channels``): every work item
                    # computes the same values, so one copy is the program;
                    # `_relocations` refuses it when a device reader needs the
                    # per-work-item copies (X-19, round 71).
                    copies = max(1, self.geometry.channels // partitions)
            group.replicas = copies
            group.work_items = len(functions)
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
                    orientation = (getattr(placement, "extra", None) or {}).get(
                        "aim_orientation"
                    )
                    shape = contraction_shape(
                        match,
                        replicated=replicated,
                        replicas=group.replicas,
                        orientation=orientation,
                    )
                    matrix, vector = mac_operands(match, orientation)
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
                        shape, group.properties.get("batch_mapping")
                    )
                    contraction = {
                        "match": match,
                        "shape": shape,
                        "activation": activation,
                        "mapping": mapping,
                        "axes": matrix_axes(match, orientation),
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
                group.elements = self._dense_elementwise_domain(group, match)
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

    def _dense_elementwise_domain(self, group, match):
        """The element count of an elementwise kernel whose iteration domain
        and store image are provably dense, else NotImplementedError.

        The EWMUL / EWADD stream writes every element of a contiguous span
        of the result from the same elements of the operands, so the
        lowering is sound only when the kernel's loops step by one and the
        result's coordinates are, dimension by dimension, one loop variable
        each (coefficient one, any constant offset) with every loop used
        once: the store image is then the dense box of the trip counts, and
        `access_region`'s bounding box (an over-approximation in general) is
        exact.  A strided loop (``range(0, n, 2)``) or a strided coordinate
        (``z[2 * i]``) would make the stream write elements the source
        leaves unchanged (round 76, ISSUES X-33: the reviewer's stride-two
        witness); such kernels are refused before selection.
        """
        facts = match.extra.get("semantic_facts")
        axes = {loop.var: loop for loop in (getattr(facts, "loops", None) or ())}
        enclosing = [var for var, _lb, _ub, _step in match.enclosing_loops]
        for var, _lb, _ub, step in match.enclosing_loops:
            axis = axes.get(var)
            trip = getattr(axis, "trip", None)
            step_value = getattr(axis, "step", step)
            try:
                step_value = int(step_value)
            except (TypeError, ValueError):
                step_value = None
            if trip is None or step_value != 1:
                raise NotImplementedError(
                    f"AiM {group.name}: the elementwise lowering covers dense "
                    f"iteration domains only; loop {var} steps by {step_value!r} "
                    f"(trip {trip!r}), so the EWMUL / EWADD stream would write the "
                    "elements the source skips"
                )
        store = match.extra.get("store_coords")
        if not store:
            raise NotImplementedError(
                f"AiM {group.name}: the result's coordinates were not resolved; "
                "the elementwise lowering needs a dense store image"
            )
        used = []
        for coordinate in store:
            atoms, _const = coordinate
            atoms = tuple(atoms)
            if (
                len(atoms) != 1
                or atoms[0][0] not in enclosing
                or int(atoms[0][1]) != 1
                or atoms[0][0] in used
            ):
                raise NotImplementedError(
                    f"AiM {group.name}: the result coordinate {coordinate!r} is "
                    "not one loop variable with coefficient one; the elementwise "
                    "lowering writes a dense span of the result and is refused "
                    "for strided or repeated stores"
                )
            used.append(atoms[0][0])
        missing = [var for var in enclosing if var not in used]
        if missing:
            raise NotImplementedError(
                f"AiM {group.name}: loop(s) {missing} do not index the result; "
                "the elementwise lowering writes each result element once and is "
                "refused for repeated stores"
            )
        return math.prod(int(axes[var].trip) for var in enclosing)

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
        """Bank rows of every value the program keeps in the banks.

        A contraction's matrix has one region per buffer (its launches and
        the host's matrix writes share it).  An elementwise group has one
        region of its own: its operands sit in the EWMUL's x and y slots and
        its result in the dst slot of that region's rows, and a value another
        group reads is copied into that group's region (`_copy_records`).
        Groups never share a region, so a later scatter or launch of one group
        cannot overwrite a value another group still has to read (round 73,
        ISSUES X-32: the reviewer's slot-liveness witness); the program-order
        liveness of every slot is then checked on the emitted records
        (`_check_physical_liveness`).
        """
        from .pim.schedule_search import InfeasibleSchedule

        parent: dict = {}

        def find(node):
            parent.setdefault(node, node)
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        def node_of(group, key):
            # A contraction matrix is one region per buffer; an elementwise
            # group's buffers are nodes of that group alone.
            if group.kind == "contraction":
                return ("key", key)
            return ("group", group.group_id, key)

        def key_of(node):
            return node[-1]

        order_of_group = {group.group_id: n for n, group in enumerate(self.groups)}
        anchors = {}
        footprints: dict = {}
        for group in self.groups:
            keys = list(group.usages)
            if not keys:
                continue
            nodes = [node_of(group, key) for key in keys]
            for node in nodes:
                find(node)
            for node in nodes[1:]:
                union(nodes[0], node)
            anchors[group.group_id] = nodes[0]
            for key, usage in group.usages.items():
                root = find(node_of(group, key))
                footprints[root] = max(
                    footprints.get(root, 1), self._footprint(group, key, usage)
                )
        components: dict = {}
        for node in parent:
            components.setdefault(find(node), []).append(node)

        def rank(root):
            nodes = components[root]
            groups = [order_of_group[n[1]] for n in nodes if n[0] == "group"]
            return (
                min(self.declaration.get(key_of(n), 1 << 30) for n in nodes),
                min(groups) if groups else -1,
            )

        rows = {}
        regions = []
        next_row = 0
        for root in sorted(components, key=rank):
            count = max(1, footprints.get(root, 1))
            rows[root] = next_row
            regions.append(
                {
                    "buffers": sorted(
                        {
                            self.name_by_key.get(key_of(n), str(key_of(n)))
                            for n in components[root]
                        }
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
        # A buffer's row: its matrix region, else the first elementwise
        # region holding it (matrix records and the realization check
        # address matrices only; elementwise records use their group's row).
        self.row_of_key = {}
        for node in parent:
            if node[0] == "key":
                self.row_of_key[key_of(node)] = rows[find(node)]
        for group in self.groups:
            for key in group.usages:
                self.row_of_key.setdefault(key, rows[find(node_of(group, key))])
        for group in self.groups:
            if group.group_id in anchors:
                group.row = rows[find(anchors[group.group_id])]
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

    def _relocations(self, group, position, steps, host_keys=None):
        """Copies of the values a launch stores into the regions of the
        readers of their live range (`_copy_records`), emitted right after the
        launch while the producer's slot holds them."""
        records = []
        for key, usage in group.usages.items():
            if not usage.store:
                continue
            records.extend(
                self._copy_records(
                    group,
                    usage,
                    key,
                    self._live_readers(key, position, steps, host_keys),
                )
            )
        return records

    def _copy_records(self, source, source_usage, key, readers):
        """Bank-to-GB-to-bank copies of buffer ``key`` from the cell its
        writer uses (``source`` group's row, ``source_usage`` slot) into every
        distinct (row, slot) cell the ``readers`` (group, usage) take it from;
        a reader whose cell is the writer's needs none.  Readers share the
        writer's distribution (`_check_device_value_layouts`)."""
        if source_usage.kind != "slot":
            return []
        targets = []
        for reader, usage in readers:
            if usage.kind != "slot":
                continue
            target = (usage.slot, reader.row)
            if target == (source_usage.slot, source.row) or target in targets:
                continue
            targets.append(target)
        records = []
        identity = self._identity(source, source_usage, key)
        geometry = self.geometry
        partitions = source.partitions * geometry.bank_groups
        span = _ceil(source.elements, partitions)
        rows = max(1, _ceil(span, geometry.row_elements))
        for target_slot, target_row in targets:
            for bank_group in range(geometry.bank_groups):
                base = bank_group * self.group_width
                # One row of the vector at a time through the global buffer:
                # the GB holds one row, so a chunk is written to its
                # destination row before the next source row replaces it
                # (round 74, ISSUES X-32: the reviewer's multi-row witness).
                for offset in range(rows):
                    chunk = min(
                        geometry.row_elements, span - offset * geometry.row_elements
                    )
                    suffix = f".row{offset}" if rows > 1 else ""
                    common = dict(
                        elements=chunk * partitions,
                        replicas=source.replicas,
                        channels_per_replica=source.partitions,
                        partitions_per_replica=partitions,
                    )
                    out = AimBankCopy(
                        "bank_to_gb",
                        bank=base + source_usage.slot,
                        row=source.row + offset,
                        name=f"{source.name}.relocate.bank_{base + source_usage.slot}{suffix}",
                        **common,
                    )
                    back = AimBankCopy(
                        "gb_to_bank",
                        bank=base + target_slot,
                        row=target_row + offset,
                        name=f"{source.name}.relocate.to_bank_{base + target_slot}{suffix}",
                        **common,
                    )
                    self._copy_identity[id(out)] = (identity, offset)
                    self._copy_identity[id(back)] = (identity, offset)
                    records.append((out, None))
                    records.append((back, None))
        return records

    def _identity(self, group, usage, key):
        """What one bank cell holds: the buffer, its distribution, the bank
        family it is laid out in (the EWMUL slots or the MAC bank pairs) and
        the logical frame the partitions hold."""
        family = "pair" if usage.stride == self.bank_pair_stride else "slot"
        return (
            key,
            int(group.replicas),
            int(group.partitions),
            family,
            self._frame(group, usage, key),
        )

    def _describe_identity(self, identity):
        if identity is None:
            return "nothing"
        key, replicas, partitions, family, frame = identity
        text = (
            f"{self.name_by_key.get(key, key)} ({replicas} x {partitions} channels, "
            f"{family} layout"
        )
        if frame is not None:
            first, total, span = frame
            text += f", elements [{first}, {first + total}) at {span} per partition"
        return text + ")"

    def _check_physical_liveness(self, records):
        """Every bank-slot read of the emitted program finds the value the
        program order placed there.

        Walk the records with one cell per (row, bank within a bank group):
        a slot-layout host write places its buffers, an EWMUL reads its x and
        y slots and places its result in the dst slot, a bank copy moves a
        cell through the global buffer, a gather reads its cells.  A read of a
        cell holding another buffer, another distribution or nothing is a
        feasibility rejection of the candidate assignment (round 73, ISSUES
        X-32: the reviewer's slot-liveness witness, where a host scatter and a
        launch overwrote copies a later reader still needed).  Matrix regions
        and the bank-pair inputs of contractions are not modelled; they live
        in regions of their own.
        """
        from .pim.schedule_search import InfeasibleSchedule

        geometry = self.geometry
        cells: dict = {}
        gb = None
        by_name = {group.name: group for group in self.groups}

        def rows_of(elements, partitions):
            return max(1, _ceil(_ceil(elements, partitions), geometry.row_elements))

        # A cell holds one row chunk of a value: (identity, chunk), chunk
        # being the row offset within the value's rows; so does the GB.
        def place(row, rows, bank, identity, first_chunk=0):
            for offset in range(rows):
                cells[(row + offset, bank)] = (identity, first_chunk + offset)

        def describe(held):
            if held is None:
                return "nothing"
            identity, chunk = held
            return f"row chunk {chunk} of {self._describe_identity(identity)}"

        def expect(row, rows, bank, identity, reader, first_chunk=0):
            for offset in range(rows):
                held = cells.get((row + offset, bank))
                if held != (identity, first_chunk + offset):
                    raise InfeasibleSchedule(
                        f"AiM {reader}: reads row chunk {first_chunk + offset} of "
                        f"{self._describe_identity(identity)} at row {row + offset} "
                        f"bank slot {bank}, but that location holds "
                        f"{describe(held)} at that point of the program (physical "
                        "liveness)"
                    )

        # Buffers the schedule declares resident are on the device before
        # the program runs, in the layout of each kernel that reads them
        # (CENT's norm gains and RoPE tables; nothing uploads them).
        for group in self.groups:
            if group.kind != "mul":
                continue
            for key, usage in group.usages.items():
                if usage.store or key not in self.resident or usage.kind != "slot":
                    continue
                place(
                    group.row,
                    self._footprint(group, key, usage),
                    usage.slot,
                    self._identity(group, usage, key),
                )

        for op, _banks in records:
            if isinstance(op, AimDistributedHostTransfer):
                stride = int(op.bank_stride)
                if stride not in (self.group_width, self.bank_pair_stride):
                    continue
                family = "slot" if stride == self.group_width else "pair"
                partitions = op.channels_per_replica * (geometry.banks // stride)
                total = op.vector_elements or op.elements
                rows = rows_of(total, partitions)
                refs = tuple(op.buffer_refs)
                bound = self.by_id.get(op.group_id)
                for copy in range(op.copies):
                    key = refs[copy] if len(refs) == op.copies else refs[0]
                    usage = bound.usages.get(key) if bound is not None else None
                    if usage is None:
                        continue
                    identity = self._identity(bound, usage, key)
                    if family == "slot":
                        banks = [op.bank_offset + copy]
                    else:
                        banks = [
                            op.bank_offset + copy + pair * stride
                            for pair in range(self.group_width // stride)
                        ]
                    for bank in banks:
                        if op.direction == "write":
                            place(op.row, rows, bank, identity)
                        else:
                            expect(
                                op.row,
                                rows,
                                bank,
                                identity,
                                f"host gather of {op.name}",
                            )
            elif isinstance(op, AimElementwise):
                if op.operation != "mul":
                    continue
                group = by_name.get(op.name)
                if group is None or group.kind != "mul":
                    continue
                partitions = group.partitions * geometry.bank_groups
                rows = rows_of(op.elements, partitions)
                for key, usage in group.usages.items():
                    if usage.store:
                        continue
                    expect(
                        op.row,
                        rows,
                        usage.slot,
                        self._identity(group, usage, key),
                        group.name,
                    )
                for key, usage in group.usages.items():
                    if usage.store:
                        place(
                            op.row, rows, usage.slot, self._identity(group, usage, key)
                        )
            elif isinstance(op, AimBankCopy):
                moved = self._copy_identity.get(id(op))
                if moved is None:
                    continue
                identity, first_chunk = moved
                partitions = op.partitions_per_replica or (
                    op.channels_per_replica * geometry.bank_groups
                )
                rows = rows_of(op.elements, partitions)
                bank = op.bank % self.group_width
                # The record is lowered one row after another; the GB holds
                # the last row read, so a multi-row record cannot carry its
                # rows faithfully.
                for offset in range(rows):
                    chunk = first_chunk + offset
                    if op.direction == "bank_to_gb":
                        expect(
                            op.row + offset, 1, bank, identity, f"copy {op.name}", chunk
                        )
                        gb = (identity, chunk)
                    else:
                        if gb != (identity, chunk):
                            raise InfeasibleSchedule(
                                f"AiM copy {op.name}: the global buffer holds "
                                f"{describe(gb)}, not row chunk {chunk} of "
                                f"{self._describe_identity(identity)} (physical "
                                "liveness)"
                            )
                        place(op.row + offset, 1, bank, identity, chunk)

    def _live_readers(self, key, position, steps, host_keys):
        """(group, usage) of every launch after ``position`` that reads buffer
        ``key`` from the device before its next writer: a launch storing it, a
        host scatter / broadcast of it, or a host stage producing it."""
        from .spmw_plan import HostComputeStep

        host_keys = host_keys or {}
        readers = []
        for index in range(position + 1, len(steps)):
            step = steps[index]
            if isinstance(step, LaunchStep):
                candidate = self.by_id[step.group_id]
                usage = candidate.usages.get(key)
                if usage is None:
                    continue
                if usage.store:
                    break
                readers.append((candidate, usage))
            elif isinstance(step, HostComputeStep):
                if key in self._stage_keys(step.outputs):
                    break
            else:
                verb, host_key = host_keys.get(index, (None, None))
                if host_key == key and verb in ("scatter", "broadcast"):
                    break
        return readers

    def _stage_keys(self, buffers):
        keys = set()
        for entry in buffers or ():
            name = entry[0] if isinstance(entry, (tuple, list)) else entry
            key = self.key_by_name.get(name)
            if key is not None:
                keys.add(key)
        return keys

    def _host_step_keys(self, steps, host_moves):
        """position -> (verb, buffer key) of every host transfer step whose
        buffer is a region parameter (the main walk reports the others)."""
        host_keys = {}
        for position, step in enumerate(steps):
            if not isinstance(step, HostStep):
                continue
            resolved = host_moves[step.host_index]
            verb = getattr(getattr(resolved, "verb", None), "name", None)
            role = resolved.buffer_role
            base = getattr(role, "base", None) if not isinstance(role, str) else None
            if base is None:
                base, _index = _parse_buffer_role(str(role))
            key = self.key_by_name.get(base)
            if verb is not None and key is not None:
                host_keys[position] = (verb, key)
        return host_keys

    def _frame(self, group, usage, key):
        """(first element, element count, elements per bank partition) of the
        logical vector a slot layout holds: bank partition p of the group's
        channels holds elements [first + p * span, first + (p + 1) * span) of
        the buffer (`aim_lowering._distributed_host_transfer`); None for a
        matrix layout."""
        if usage.kind != "slot":
            return None
        first, total = self._slot_frame(group, key)
        if group.split_replicas:
            total = _ceil(total, group.replicas)
        partitions = group.partitions * (self.geometry.banks // usage.stride)
        return (int(first), int(total), _ceil(total, partitions))

    def _distribution(self, group, usage, key):
        """The physical distribution of one buffer held by one group: copies,
        channels per copy, whether the copies split the vector, the bank
        family (the slot stride, or the matrix layout), and the logical
        frame the partitions hold (round 75, ISSUES X-32: the reviewer's
        prefix composition, where producer and consumer agreed on every
        count but assigned different elements to the same bank partition)."""
        return (
            int(group.replicas),
            int(group.partitions),
            bool(group.split_replicas),
            ("slot", int(usage.stride)) if usage.kind == "slot" else ("matrix",),
            self._frame(group, usage, key),
        )

    @staticmethod
    def _describe_distribution(distribution):
        replicas, partitions, split, family, frame = distribution
        text = (
            f"{replicas} {'work-item part' if split else 'cop'}"
            f"{'s' if split and replicas != 1 else ('y' if replicas == 1 else 'ies')} "
            f"over {partitions} channels each"
        )
        if family[0] == "slot":
            first, total, span = frame
            return (
                f"{text}, bank stride {family[1]}, elements [{first}, "
                f"{first + total}) at {span} per bank partition"
            )
        return f"{text}, a contraction matrix layout"

    def _check_device_value_layouts(self, steps, host_keys):
        """Every device read of a buffer takes the distribution its last
        writer left on the device.

        Walk the schedule keeping, per buffer, the distribution of its last
        writer: a launch that stores it, or a host scatter / broadcast (which
        follows the layout `_host_usage` binds it to).  A host stage that
        produces the buffer ends its device residency.  A launch reading the
        buffer in another distribution is refused for this candidate
        assignment: no redistribution between slot layouts is lowered, so the
        reader would address data the writer never placed there (reviewer,
        round 71: a fan-out whose second reader differs, and a producer whose
        readers differ, both with ordinary source and enumerated candidates).
        """
        from .pim.schedule_search import InfeasibleSchedule
        from .spmw_plan import HostComputeStep

        resident: dict = {}
        for position, step in enumerate(steps):
            if isinstance(step, LaunchStep):
                group = self.by_id[step.group_id]
                for key, usage in group.usages.items():
                    if usage.store:
                        continue
                    held = resident.get(key)
                    if held is None:
                        continue
                    writer, distribution = held
                    reading = self._distribution(group, usage, key)
                    if reading != distribution:
                        name = self.name_by_key.get(key, key)
                        hint = (
                            "a device-resident value is read in the frame its "
                            "writer laid out (no redistribution between bank "
                            "partitions is lowered): make the kernels cover the "
                            "same elements of the buffer, or have the host "
                            "scatter the region the reader needs"
                            if reading[:4] == distribution[:4]
                            else "a device-resident value needs readers of the "
                            "same layout (no redistribution is lowered)"
                        )
                        raise InfeasibleSchedule(
                            f"AiM {group.name}: reads {name} from the device as "
                            f"{self._describe_distribution(reading)}, but {writer} "
                            f"left it as {self._describe_distribution(distribution)}; "
                            + hint
                        )
                for key, usage in group.usages.items():
                    if usage.store:
                        resident[key] = (
                            f"launch {group.name}",
                            self._distribution(group, usage, key),
                        )
            elif isinstance(step, HostComputeStep):
                for key in self._stage_keys(step.outputs):
                    resident.pop(key, None)
            else:
                verb, key = host_keys.get(position, (None, None))
                if key is None or verb not in ("scatter", "broadcast"):
                    continue
                bound, usage = self._host_usage(key, position, steps, verb)
                resident[key] = (
                    f"the host {verb} (laid out for {bound.name})",
                    self._distribution(bound, usage, key),
                )

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

    def _slot_frame(self, group, key):
        """(first flat element, element count) of the region the group's work
        bodies access of buffer ``key``: the logical vector its slot layout
        holds (the kernel's vector, which may be a prefix of the host buffer).
        A partitioned grid holds the whole buffer, 1/replicas per replica."""
        from .spmw_plan import _facts_accesses, access_region, flat_range

        shape = self.shape_by_key.get(key)
        if group.split_replicas:
            return 0, self._elements(key, group.elements * group.replicas)
        boxes = []
        for match in group.matches:
            facts = match.extra.get("semantic_facts")
            reads, writes = _facts_accesses(facts)
            for access in reads + writes:
                if self._key(access.value_ref, access.memref_name) != key:
                    continue
                box = access_region(access, getattr(facts, "loops", ()))
                if box is not None:
                    boxes.append(box)
        if not boxes:
            return 0, self._elements(key, group.elements)
        box = boxes[0]
        name = self.name_by_key.get(key, key)
        if any(other != box for other in boxes[1:]):
            raise NotImplementedError(
                f"AiM {group.name}: {name} is accessed over different regions "
                f"{boxes}"
            )
        frame = flat_range(shape or tuple(hi for _lo, hi in box), box)
        if frame is None:
            raise NotImplementedError(
                f"AiM {group.name}: the region {list(box)} of {name} is not one "
                "contiguous range of the buffer"
            )
        return frame

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
        host_keys = self._host_step_keys(steps, host_moves)
        self._check_device_value_layouts(steps, host_keys)
        records = []
        pending: list = []  # coalescible distributed transfers
        # Copies of host-written values into the regions of their other
        # readers, emitted after the transfers that place them (at the next
        # flush, before any launch).
        deferred: list = []
        gather_run = False

        def flush():
            nonlocal gather_run
            for transfer in pending:
                records.append((AimDistributedHostTransfer(**transfer), None))
            pending.clear()
            if gather_run:
                records.append((AimSync(name="host_gather_split"), None))
            gather_run = False
            records.extend(deferred)
            deferred.clear()

        from .spmw_plan import HostComputeStep

        for position, step in enumerate(steps):
            if isinstance(step, LaunchStep):
                flush()
                group = self.by_id[step.group_id]
                records.extend(self._launch_records(group))
                records.extend(self._relocations(group, position, steps, host_keys))
                continue
            if isinstance(step, HostComputeStep):
                # The declared host stage, in program order: no commands; its
                # volumes are the stage's declared buffers.
                flush()
                records.append(
                    (
                        AimHostStage(
                            stage=step.name,
                            inputs=step.inputs,
                            outputs=step.outputs,
                            name=f"host_compute.{step.name}",
                            function=step.function,
                            operation=step.operation,
                        ),
                        None,
                    )
                )
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
            region_first, region_total = self._slot_frame(group, key)
            if group.split_replicas:
                if index is not None:
                    raise NotImplementedError(
                        f"AiM host {verb} of {text}: a sliced transfer into a "
                        "work-item-split elementwise layout"
                    )
                elements, first, vector = _ceil(region_total, group.replicas), 0, None
            else:
                # The slice's coordinates in the kernel's vector: the layout
                # is the vector's, and only the bursts of the slice move.
                from .spmw_plan import flat_range, slice_region

                shape = self.shape_by_key.get(key)
                if shape is None:
                    raise NotImplementedError(
                        f"AiM host {verb} of unknown-shape buffer {text}"
                    )
                try:
                    flat = flat_range(shape, slice_region(shape, index))
                except ValueError as error:
                    raise NotImplementedError(
                        f"AiM host {verb} of {text}: {error}"
                    ) from error
                if flat is None:
                    raise NotImplementedError(
                        f"AiM host {verb} of {text}: the slice is not one contiguous "
                        "range of the buffer"
                    )
                start, elements = flat
                first, vector = start - region_first, region_total
                if first < 0 or first + elements > region_total:
                    raise NotImplementedError(
                        f"AiM host {verb} of {text}: elements [{start}, "
                        f"{start + elements}) lie outside the region [{region_first}, "
                        f"{region_first + region_total}) the kernel's layout holds"
                    )
            transfer = dict(
                direction="read" if verb == "gather" else "write",
                elements=elements,
                first_element=first,
                vector_elements=vector,
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
                # A write into a layout the group *stores* to: the host
                # rewrites a value the kernel produced, after its launch
                # (`_host_usage` binds a scatter with no later reader to the
                # producer's output slot).
                reupload=bool(usage.store and verb != "gather"),
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
                        "first_element",
                        "vector_elements",
                        "reupload",
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
            if verb != "gather" and not transfer["reupload"]:
                deferred.extend(
                    self._copy_records(
                        group,
                        usage,
                        key,
                        self._live_readers(key, position, steps, host_keys),
                    )
                )
        flush()
        self._check_physical_liveness(records)
        return records


def _parse_buffer_role(text):
    """Shared with the plan: `spmw_plan.parse_buffer_role`."""
    from .spmw_plan import parse_buffer_role

    return parse_buffer_role(text)


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


def _host_ingress_route(route) -> bool:
    """A declared transfer route that starts on the host: the consumer
    receives the value from the host (``host_scatter``/``host_broadcast``
    into the banks, or the ``host_gpr_preload`` of a GB-fed vector)."""
    route = tuple(route or ())
    return bool(route) and str(route[0]).startswith("host")


class AimSimulatorFailure(RuntimeError):
    """A ramulator2 invocation that produced no cycles.  ``evidence`` keeps
    what the run was: the executor (binary / library / config identity),
    the exact segment trace and its digest, the command and working
    directory, the return code, the complete stdout and stderr, and the
    segments completed before it; no timing is reported."""

    def __init__(self, message: str, evidence: dict):
        super().__init__(message)
        self.evidence = evidence


def _resolved_shared_library(binary: Path, stem: str, library_path: str) -> dict:
    """The shared library ``stem`` the dynamic loader resolves for
    ``binary`` (via ``ldd`` under the run's ``LD_LIBRARY_PATH``), with its
    content digest; the loader's answer when it cannot be resolved."""
    import hashlib

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = library_path
    try:
        out = subprocess.run(
            ["ldd", str(binary)], capture_output=True, timeout=30, env=env, check=False
        ).stdout.decode("utf-8", errors="replace")
    except (subprocess.SubprocessError, OSError) as error:
        return {"path": None, "sha256": None, "error": str(error)}
    for line in out.splitlines():
        if stem in line and "=>" in line:
            path = line.split("=>", 1)[1].split("(")[0].strip()
            if path and Path(path).is_file():
                resolved = Path(path).resolve()
                return {
                    "path": str(resolved),
                    "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
                }
            return {"path": path or None, "sha256": None, "ldd": line.strip()}
    return {"path": None, "sha256": None, "ldd": out[-300:]}


def _run_aim(compiled: "Compiled", **inputs) -> RunResult:
    """Run a compiled AiM artifact through ramulator2.

    `compiled.cmds` is a list of already-terminated text trace lines. We
    serialize those exact candidate-owned segments and invoke the ramulator2
    binary inside the `aim-simulator-build` Docker image (`TENON_AIM_SIM_IMAGE`)
    or, when `TENON_AIM_RAMULATOR2` names a natively built binary, that
    binary from the simulator tree (the executor and the binary's SHA-256
    are recorded in ``extra["simulator"]``).  Each trace is written to a
    private temp directory (mounted read-only at `/trace` under Docker), so
    the simulator checkout is never written. The simulator's YAML stdout
    carries per-channel `memory_system_cycles`; we parse out the max across
    channels as the headline cycle count.

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

    from .spmw_simenv import aim_ramulator2_binary

    import hashlib

    config = {
        "path": str(yaml_cfg),
        "sha256": hashlib.sha256(yaml_cfg.read_bytes()).hexdigest(),
    }
    native = aim_ramulator2_binary()
    if native is not None:
        native = native.resolve()
        library_path = os.pathsep.join(
            p for p in (str(native.parent), os.environ.get("LD_LIBRARY_PATH")) if p
        )
        executor = {
            "kind": "native",
            "binary": str(native),
            "sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
            # The shared library the executable resolves at run time (the
            # upstream build keeps libramulator.so in the source tree).
            "library": _resolved_shared_library(native, "libramulator", library_path),
            "config": config,
            "cwd": str(root),
            "ld_library_path": library_path,
        }
    else:
        executor = {
            "kind": "docker",
            "image": _aim_sim_image(),
            "config": {**config, "mounted_as": str(cfg_rel)},
            "mount": str(root),
        }

    invocations: list[dict] = []
    segment_records: list[dict] = []

    def failure(
        message,
        *,
        label,
        repeat,
        lines,
        command,
        cwd,
        returncode,
        stdout,
        stderr,
        error=None,
    ):
        text = "".join(str(line) + "\n" for line in lines)
        return AimSimulatorFailure(
            message,
            {
                "executor": executor,
                "segment": {
                    "label": label,
                    "repeat": int(repeat),
                    "lines": len(lines),
                    "trace_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "trace": list(map(str, lines)),
                },
                "command": list(command),
                "cwd": cwd,
                "returncode": returncode,
                "stdout": stdout,
                "stderr": stderr,
                "exception": (
                    None if error is None else f"{type(error).__name__}: {error}"
                ),
                "completed_segments": [
                    {k: v for k, v in record.items() if k != "trace"}
                    for record in segment_records
                ],
                "invocations": list(invocations),
            },
        )

    def run_segment(
        lines: list[str], label: str = "program", repeat: int = 1
    ) -> tuple[int, str, int]:
        with tempfile.TemporaryDirectory(prefix="tenon-aim-") as trace_dir:
            trace_path = Path(trace_dir) / "segment.trace"
            trace_path.write_text("".join(str(line) + "\n" for line in lines))
            if native is not None:
                env = dict(os.environ)
                env["LD_LIBRARY_PATH"] = executor["ld_library_path"]
                command = [str(native), "-f", str(cfg_rel), "-t", str(trace_path)]
                run = dict(cwd=str(root), env=env)
            else:
                command = [
                    "docker",
                    "run",
                    "--rm",
                    "-v",
                    f"{root}:/work",
                    "-v",
                    f"{trace_dir}:/trace:ro",
                    executor["image"],
                    "bash",
                    "-c",
                    f"cd /work && ./build/ramulator2 -f {cfg_rel} "
                    f"-t /trace/{trace_path.name}",
                ]
                run = {}
            cwd = run.get("cwd")
            invocations.append(
                {"command": list(command), **({"cwd": cwd} if cwd else {})}
            )
            try:
                proc = subprocess.run(
                    command, capture_output=True, timeout=600, check=False, **run
                )
            except (subprocess.SubprocessError, OSError) as exc:
                # A timeout (or another subprocess error) may carry the
                # partial streams the process wrote before it was killed:
                # keep them, decoded, beside the trace and executor identity.
                def _text(stream):
                    if stream is None:
                        return None
                    if isinstance(stream, bytes):
                        return stream.decode("utf-8", errors="replace")
                    return str(stream)

                raise failure(
                    f"AiM ramulator2 invocation failed: {exc}",
                    label=label,
                    repeat=repeat,
                    lines=lines,
                    command=command,
                    cwd=cwd,
                    returncode=None,
                    stdout=_text(
                        getattr(exc, "stdout", None) or getattr(exc, "output", None)
                    ),
                    stderr=_text(getattr(exc, "stderr", None)),
                    error=exc,
                ) from exc
        stdout = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")
        combined = stdout + ("\n" + stderr if stderr else "")
        if proc.returncode != 0:
            # A failed run may still print partial statistics; never report
            # cycles from it.
            raise failure(
                f"AiM ramulator2 exited with status {proc.returncode}; no cycles "
                "are reported from a failed run; tail: " + combined[-400:],
                label=label,
                repeat=repeat,
                lines=lines,
                command=command,
                cwd=cwd,
                returncode=proc.returncode,
                stdout=stdout,
                stderr=stderr,
            )
        values = [
            int(m) for m in re.findall(r"memory_system_cycles:\s*(\d+)", combined)
        ]
        if not values:
            raise failure(
                "AiM ramulator2 returned but stdout missing "
                "'memory_system_cycles: ...' line; tail: " + combined[-400:],
                label=label,
                repeat=repeat,
                lines=lines,
                command=command,
                cwd=cwd,
                returncode=proc.returncode,
                stdout=stdout,
                stderr=stderr,
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
    # One record per segment: the exact trace lines the simulator consumed,
    # their digest, the repeat count, the measured and composed cycles, the
    # simulator's own output and whether the run was shared with an
    # identical earlier segment.
    for label, repeat, lines in segments:
        key = tuple(lines)
        cached = key in cache
        if not cached:
            cache[key] = run_segment(lines, label, repeat)
        raw_cycles, log, returncode = cache[key]
        total_cycles += raw_cycles * repeat
        returncodes.append(returncode)
        logs.append(
            f"=== {label}; repeat={repeat}; raw_cycles={raw_cycles}; "
            f"composed_cycles={raw_cycles * repeat} ===\n{log}"
        )
        text = "".join(str(line) + "\n" for line in lines)
        segment_records.append(
            {
                "label": label,
                "repeat": int(repeat),
                "raw_cycles": int(raw_cycles),
                "composed_cycles": int(raw_cycles * repeat),
                "lines": len(lines),
                "trace_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "trace": list(map(str, lines)),
                "stdout": log,
                "returncode": int(returncode),
                "shared_run": cached,
            }
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
            "simulator": executor,
            "segment_records": segment_records,
            "invocations": invocations,
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
