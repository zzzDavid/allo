# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SK hynix AiM codegen context and ramulator2 runtime (moved from spmw_codegen)."""

from __future__ import annotations

import math
import re
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

from .spmw_autoschedule import _matcher_work_scope
from .spmw_plan import LaunchStep
from .spmw_match import MatchedOp
from .spmw_target import MemoryRef, Register
from .spmw_codegen import (
    CodegenContext,
    _eval_sym,
    _parse_loop_bound,
    RunResult,
    SimulatorUnavailable,
    Compiled,
)
from .spmw_simenv import aim_root as _aim_root
from .spmw_simenv import aim_unavailable_reason as _aim_unavailable_reason
from .pim.aim_lowering import (
    AimAllBankWrite,
    AimBankCopy,
    AimDistributedHostTransfer,
    AimElementwise,
    AimHostTransfer,
    AimSync,
    _AimLowerer,
    _memref_shape,
    _target_geometry,
    build_contraction,
    contraction_shape,
    group_replicated,
    mac_operands,
    matrix_axes,
    resolve_batch_mapping,
)


class AimCtx(CodegenContext):
    """Codegen ctx for AiM. Emits one text-format ISR line per `cmd(...)`.

    Output is a list of strings on `self.cmds` (NOT `PIMCmd` records —
    AiM's ramulator2 consumes a plain text trace). Each line is a
    space-separated, positional sequence of integer fields in the order
    ramulator2 expects per opcode (see `_ISR_FIELDS`). A side-channel
    `_human_lines` mirror keeps the older `key=value` annotation that
    `_operand_text` produces — useful for unit tests that inspect the
    rendered trace by substring.
    """

    # Per-opcode field ordering (positional, matches
    # ``Ramulator::AiMISRInfo::opcode_str_to_aim_ISR`` in request.h).
    # Each tuple is the legal field sequence; values are pulled from
    # the field map built by `_field_map(...)`.
    _ISR_FIELDS = {
        "WR_SBK": ("gpr_addr_0", "channel_mask", "bank_index", "row_addr"),
        "WR_ABK": ("gpr_addr_0", "channel_mask", "row_addr"),
        "WR_GB": ("opsize", "gpr_addr_0", "channel_mask"),
        "WR_BIAS": ("gpr_addr_0", "channel_mask"),
        "WR_AFLUT": ("opsize",),
        "RD_MAC": ("gpr_addr_0", "channel_mask"),
        "RD_AF": ("gpr_addr_0", "channel_mask"),
        "RD_SBK": ("gpr_addr_0", "channel_mask", "bank_index", "row_addr"),
        "COPY_BKGB": ("opsize", "channel_mask", "bank_index", "row_addr"),
        "COPY_GBBK": ("opsize", "channel_mask", "bank_index", "row_addr"),
        "MAC_SBK": ("opsize", "channel_mask", "bank_index", "row_addr"),
        "MAC_ABK": ("opsize", "channel_mask", "row_addr"),
        "AF": ("channel_mask",),
        "EWMUL": ("opsize", "channel_mask", "row_addr"),
        "EWADD": ("opsize", "gpr_addr_0", "gpr_addr_1"),
        "SYNC": (),
        "EOC": (),
    }

    def __init__(self, target):
        super().__init__(target)
        # Override the parent's PIMCmd list with a plain text-line list —
        # AiM's ramulator2 frontend consumes a text trace.
        self.cmds: list[str] = []
        # Parallel list of the human-readable key=value annotation per
        # emitted line; populated by `cmd()` so tests can introspect.
        self._human_lines: list[str] = []

    def _operand_text(self, handle, role: str) -> str:
        """Render a Tenon handle as the AiM trace field set it implies.

        This is the human-readable `key=value` annotation; positional
        emission flows through `_handle_to_fields`. Tests that previously
        asserted on this format inspect `self._human_lines`.
        """
        if handle is None:
            return ""
        if isinstance(handle, Register):
            if handle.name == "gpr":
                prefix = "gpr_in" if role.startswith("src") else "gpr_out"
                return f"{prefix}=0"
            if handle.name in ("mac_reg", "af_reg"):
                return f"{handle.name}=0"
            raise NotImplementedError(
                f"AimCtx: register {handle.name!r} has no operand mapping."
            )
        if isinstance(handle, MemoryRef):
            mem = handle.memory
            if mem.name == "banks":
                # `handle.idx` may be a symbolic SymExpr (e.g. autoscheduler
                # produces `8*bg + bank` with placeholder UnitIds); collapse
                # to a concrete int via `_eval_sym` so ramulator2 accepts it.
                env = {
                    level: value
                    for level, value in enumerate(getattr(self, "_active_work_id", ()))
                }
                bank_int = _eval_sym(handle.idx, env)
                if bank_int is None:
                    raise NotImplementedError(
                        f"AimCtx: bank index {handle.idx!r} cannot be reduced "
                        "to an integer; supply concrete UnitId values."
                    )
                return f"bank={bank_int} row=0"
            if mem.name == "gb":
                return "gb=1"
            raise NotImplementedError(
                f"AimCtx: memory {mem.name!r} has no operand mapping."
            )
        # Memory (whole-memory broadcast operand, e.g. WR_ABK src0=banks).
        from .spmw_target import Memory

        if isinstance(handle, Memory):
            if handle.name == "banks":
                return "bank=ALL row=0"
            if handle.name == "gb":
                return "gb=1"
            if handle.name == "gpr":
                prefix = "gpr_in" if role.startswith("src") else "gpr_out"
                return f"{prefix}=0"
            raise NotImplementedError(
                f"AimCtx: whole-memory {handle.name!r} has no operand mapping."
            )
        raise NotImplementedError(
            f"AimCtx: unknown handle type {type(handle).__name__}."
        )

    def _handle_to_fields(self, handle, role: str, fields: dict) -> None:
        """Update `fields` with the integer values implied by `handle`.

        Keys written: `gpr_addr_0`, `gpr_addr_1`, `bank_index`, `row_addr`,
        and `channel_mask` for whole-memory broadcasts. Unknown handles
        are tolerated — the caller's ISR-field selector picks only the
        fields the opcode declares as legal.
        """
        if handle is None:
            return
        from .spmw_target import Memory

        if isinstance(handle, Register):
            if handle.name in ("gpr", "mac_reg", "af_reg"):
                # gpr_addr_1 hosts the second GPR for ISRs that need two
                # (EWADD); the first src GPR routes to gpr_addr_0.
                if role == "src1" and "gpr_addr_0" in fields:
                    fields["gpr_addr_1"] = 0
                else:
                    fields.setdefault("gpr_addr_0", 0)
            return
        if isinstance(handle, MemoryRef):
            mem = handle.memory
            if mem.name == "banks":
                env = {
                    level: value
                    for level, value in enumerate(getattr(self, "_active_work_id", ()))
                }
                bank_int = _eval_sym(handle.idx, env)
                if bank_int is None:
                    raise NotImplementedError(
                        f"AimCtx: bank index {handle.idx!r} cannot be reduced "
                        "to an integer; supply concrete UnitId values."
                    )
                fields["bank_index"] = bank_int
                fields.setdefault("row_addr", 0)
            elif mem.name == "gb":
                fields.setdefault("channel_mask", 1)
            return
        if isinstance(handle, Memory):
            if handle.name == "banks":
                # WR_ABK / MAC_ABK target every bank in a channel; the
                # parser still wants a row_addr field.
                fields.setdefault("row_addr", 0)
            elif handle.name == "gb":
                fields.setdefault("channel_mask", 1)
            elif handle.name == "gpr":
                if role == "src1" and "gpr_addr_0" in fields:
                    fields["gpr_addr_1"] = 0
                else:
                    fields.setdefault("gpr_addr_0", 0)
            return

    def cmd(self, name: str, dst=None, src0=None, src1=None, **fields):
        """Emit one ISR line in ramulator2's positional format.

        `fields` may carry overrides (e.g. `row_addr=8`, `opsize=2`);
        unspecified fields default to 0 (or 1 for `channel_mask`).
        Unknown opcodes fall back to the legacy key=value form so
        callers using non-canonical opcode names still produce a
        parseable diagnostic line.
        """
        # Build the integer field map from operand handles + overrides.
        work_id = getattr(self, "_active_work_id", ())
        channel_id = int(work_id[0]) if work_id else 0
        field_map: dict = {"channel_mask": 1 << channel_id}
        self._handle_to_fields(src0, "src0", field_map)
        self._handle_to_fields(src1, "src1", field_map)
        self._handle_to_fields(dst, "dst", field_map)
        # Caller overrides (e.g. `row_addr=...`, `opsize=...`).
        for k, v in fields.items():
            field_map[k] = v

        # Human-readable mirror — keeps the older key=value annotation so
        # tests can introspect handle-derived field labels.
        parts = [f"AiM {name}", f"ch={channel_id}"]
        if src0 is not None:
            parts.append(self._operand_text(src0, "src0"))
        if src1 is not None:
            parts.append(self._operand_text(src1, "src1"))
        if dst is not None:
            parts.append(self._operand_text(dst, "dst"))
        for k, v in fields.items():
            parts.append(f"{k}={v}")
        human = "  ".join(p for p in parts if p)
        self._human_lines.append(human)

        # Positional output — ramulator2 expects space-separated ints in
        # the order declared by `_ISR_FIELDS[opcode]`. Defaults:
        # gpr/opsize/bank/row -> 0; channel_mask -> 1 (channel 0 enabled).
        order = self._ISR_FIELDS.get(name)
        if order is None:
            # Unknown opcode -- preserve diagnostic mode by emitting the
            # key=value annotation. ramulator2 will reject it, which is
            # the correct behaviour (caller asked for an opcode the spec
            # doesn't know about).
            self.cmds.append(human)
            return
        defaults = {
            "gpr_addr_0": 0,
            "gpr_addr_1": 0,
            "opsize": 1,
            "bank_index": 0,
            "row_addr": 0,
            "channel_mask": 1,
        }
        tokens = [f"AiM {name}"]
        for fld in order:
            tokens.append(str(field_map.get(fld, defaults[fld])))
        self.cmds.append(" ".join(tokens))

    def append(self, line: str):
        """Low-level escape hatch — append a raw trace line."""
        self.cmds.append(line)

    def emit_eoc(self):
        """End-of-command marker — every AiM trace ends with EOC."""
        self.cmds.append("AiM EOC")
        self._human_lines.append("AiM EOC")

    def resolve_moves(self, role, src_handle=None, dst_handle=None):
        from .spmw_target import Memory

        if dst_handle is None:
            return (None, None)
        if isinstance(dst_handle, MemoryRef):
            mem = dst_handle.memory
            if mem.name == "banks":
                # per-bank: WR_SBK in, RD_SBK out
                return ("WR_SBK", "RD_SBK")
            if mem.name == "gb":
                return ("WR_GB", None)
            return (None, None)
        if isinstance(dst_handle, Memory):
            if dst_handle.name == "banks":
                # all-bank broadcast write; no readback path
                return ("WR_ABK", None)
            if dst_handle.name == "gb":
                return ("WR_GB", None)
            return (None, None)
        if isinstance(dst_handle, Register):
            if dst_handle.name == "mac_reg":
                return (None, "RD_MAC")
            if dst_handle.name == "af_reg":
                return (None, "RD_AF")
            if dst_handle.name == "gpr":
                return (None, None)
            return (None, None)
        return (None, None)

    def _sbk_banks(self, placement) -> tuple[int, ...]:
        """Physical banks one single-bank launch touches, from the layout."""
        layout = getattr(placement, "layout", None)
        if layout is None or not {"work_bank", "lane_bank"}.issubset(layout.bases):
            raise NotImplementedError(
                "AiM SBK expansion needs a LinearLayout with work_bank and "
                "lane_bank inputs"
            )
        out_index = layout.out_dims.index("bank")
        fixed = {name: 0 for name in layout.bases if name not in ("work_bank", "lane_bank")}
        banks: list[int] = []
        for work_bank in range(layout.size_of("work_bank")):
            for lane_bank in range(layout.size_of("lane_bank")):
                bank = layout.apply(
                    **fixed, work_bank=work_bank, lane_bank=lane_bank
                )[out_index]
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
            if entry["kind"] == "contraction":
                entry["bank_scope"] = "abk" if sbk_banks is None else "sbk"
                if sbk_banks is not None:
                    entry["sbk_banks"] = list(sbk_banks)
        self.lowered_operations = tuple(operations)
        self.lowering_manifest = lowerer.operations
        self.row_regions = program.regions
        self.cmds.extend(lines)
        if program.groups:
            self.emit_eoc()

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

# CENT's mapper conventions for the bank slots inside one bank group: an EWMUL
# reads x and y from slots 0 and 1 and writes slot 2; a bank-pair MAC reads its
# shared buffer from slots 0 and 1 of each neighbour pair.
_EWMUL_SLOTS = {"x": 0, "y": 1, "dst": 2}
_BANK_PAIR_STRIDE = 2
_BANK_PAIR_SLOTS = (0, 1)


def _ceil(value, divisor):
    return -(-int(value) // int(divisor))


class _Usage:
    """How one kernel group holds one buffer in banks."""

    __slots__ = ("kind", "stride", "slot", "copies", "store", "contraction")

    def __init__(self, kind, *, stride=0, slot=0, copies=1, store=False, contraction=None):
        self.kind = kind  # "slot" or "matrix"
        self.stride = stride
        self.slot = slot
        self.copies = copies
        self.store = store
        self.contraction = contraction


class _Group:
    __slots__ = (
        "group_id", "name", "functions", "matches", "properties", "kind",
        "replicas", "partitions", "elements", "usages", "contractions",
        "sbk_banks", "row",
    )

    def __init__(self, group_id, name):
        self.group_id = group_id
        self.name = name
        self.usages = {}
        self.contractions = []
        self.sbk_banks = None
        self.elements = 0
        self.row = 0


class _AimProgramDerivation:
    """Derive the ordered AimOp records one compiled AiM program lowers."""

    def __init__(self, target, trace, layouts, schedule):
        from .spmw_autoschedule import _bucket_for_autoschedule

        self.target = target
        self.trace = trace
        self.schedule = schedule
        self.geometry = _target_geometry(target)
        self.group_width = self.geometry.banks // self.geometry.bank_groups
        self.layout_by_scope = {
            scope: layout
            for (scope, _matches), layout in zip(_bucket_for_autoschedule(trace), layouts)
        }
        self._index_buffers()
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
                self.declaration.setdefault(key, len(parameters) + len(self.declaration))
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
                        channels=self.geometry.channels,
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
                            stride=_BANK_PAIR_STRIDE,
                            slot=_BANK_PAIR_SLOTS[0],
                            copies=len(_BANK_PAIR_SLOTS),
                        )
                    else:
                        group.usages[key] = _Usage("matrix", contraction=contraction)
            elif "MUL" in names or "ADD" in names:
                kind = "MUL" if "MUL" in names else "ADD"
                matched = [match for match in reps if match.target_op_name == kind]
                if len(matched) != 1:
                    raise NotImplementedError(
                        f"AiM {group.name}: expected one {kind} per kernel, "
                        f"got {len(matched)}"
                    )
                match = matched[0]
                group.kind = kind.lower()
                group.elements = math.prod(
                    _parse_loop_bound(ub) for _var, _lb, ub, _step in match.enclosing_loops
                )
                if kind == "MUL":
                    roles = {op.role: op for op in match.operands}
                    for role in ("x", "y"):
                        operand = roles[role]
                        key = self._key(operand.value_ref, operand.memref_name)
                        group.usages[key] = _Usage(
                            "slot", stride=self.group_width, slot=_EWMUL_SLOTS[role]
                        )
                    key = self._key(match.result_value_ref, match.result_memref_name)
                    group.usages[key] = _Usage(
                        "slot",
                        stride=self.group_width,
                        slot=_EWMUL_SLOTS["dst"],
                        store=True,
                    )
            else:
                raise NotImplementedError(
                    f"AiM {group.name} has no MAC, MUL or ADD match; got {sorted(names)}"
                )
            groups.append(group)
        return groups

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
            extents["batch"], extents["output"], extents["reduction"]
        )
        cpr = group.partitions
        channels = group.replicas * cpr
        mapping = contraction["mapping"]
        if mapping == "row_packed":
            aligned = _ceil(reduction, geometry.lanes) * geometry.lanes
            pack = max(1, min(batch, cols // aligned))
            return _ceil(batch, pack) * _ceil(outputs, cpr * geometry.banks)
        if mapping == "channels":
            return _ceil(batch, cpr) * _ceil(outputs, geometry.banks) * _ceil(reduction, cols)
        return _ceil(outputs * batch * group.replicas, channels * geometry.banks) * _ceil(
            reduction, cols
        )

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
                if group.kind == "mul" and self._relocated(group, key) and any(
                    k in self.resident for k in group.usages
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
            key=lambda root: min(self.declaration.get(k, 1 << 30) for k in components[root]),
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
                operation = replace(operation, row=self.row_of_key[contraction["matrix"]])
                records.append((operation, group.sbk_banks))
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
        pairs = group.partitions * (self.geometry.banks // _BANK_PAIR_STRIDE)
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
            raise NotImplementedError(f"AiM host transfer of unknown-shape buffer {key}")
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

    def _matrix_records(self, direction, group, usage, index, name):
        """Cache-append transfers into a contraction matrix layout (A5)."""
        geometry = self.geometry
        contraction = usage.contraction
        axes = contraction["axes"]
        shape = contraction["shape"]
        if index is None or len(index) != len(axes):
            raise NotImplementedError(f"AiM {direction} {name}: expected a full-rank slice")
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
            output_tile, in_tile = divmod(position, tile)
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
                            ),
                            None,
                        )
                    )
            return records
        if direction == "broadcast" and mapping == "channels" and fixed == ["reduction"]:
            storage = shape["storage_extent"] or shape["reduction"]
            storage_rows = _ceil(storage, geometry.row_elements)
            output_tiles = _ceil(shape["outputs"], geometry.banks)
            return [
                (
                    AimAllBankWrite(
                        row=base + position // geometry.row_elements,
                        rows=output_tiles,
                        row_stride=storage_rows,
                        copies=_ceil(shape["batches"], cpr),
                        copy_row_stride=output_tiles * storage_rows,
                        channels=tuple(range(group.replicas * cpr)),
                        name=name,
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
                records.extend(self._matrix_records(verb, group, usage, index, text))
                continue
            if verb == "broadcast":
                raise NotImplementedError(f"AiM broadcast of {text} into a slot layout")
            transfer = dict(
                direction="read" if verb == "gather" else "write",
                elements=self._token_extent(key, index),
                replicas=group.replicas,
                channels_per_replica=group.partitions,
                row=group.row if group.kind != "contraction" else self.row_of_key[key],
                bank_stride=usage.stride,
                bank_offset=usage.slot,
                copies=usage.copies,
                name=text,
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
                    )
                )
                and last["bank_offset"] + last["copies"] == transfer["bank_offset"]
            ):
                last["copies"] += transfer["copies"]
                last["name"] = f"{last['name']}+{text}"
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


def _run_aim(compiled: "Compiled", **inputs) -> RunResult:
    """Run a compiled AiM artifact through ramulator2 in Docker.

    `compiled.cmds` is a list of already-terminated text trace lines. We
    serialize those exact candidate-owned segments and invoke the ramulator2
    binary inside the `aim-simulator-build` Docker image. The simulator's
    YAML stdout carries per-channel `memory_system_cycles`; we parse
    out the max across channels as the headline cycle count.
    """
    reason = _aim_unavailable_reason()
    if reason is not None:
        raise SimulatorUnavailable("aim", reason)
    root = _aim_root()
    yaml_cfg = root / "test" / "example.yaml"

    cfg_rel = yaml_cfg.relative_to(root)

    def run_segment(lines: list[str]) -> tuple[int, str, int]:
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".trace", dir=str(root / "test")
        ) as tf:
            for line in lines:
                tf.write(str(line) + "\n")
            trace_path = Path(tf.name)
        trace_rel = trace_path.relative_to(root)
        try:
            proc = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "-v",
                    f"{root}:/work",
                    "aim-simulator-build",
                    "bash",
                    "-c",
                    f"cd /work && ./build/ramulator2 -f {cfg_rel} -t {trace_rel}",
                ],
                capture_output=True,
                timeout=600,
                check=False,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            raise RuntimeError(f"AiM ramulator2 invocation failed: {exc}") from exc
        finally:
            trace_path.unlink(missing_ok=True)
        stdout = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")
        combined = stdout + ("\n" + stderr if stderr else "")
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
        cycles=cycles,
        stdout=combined,
        backend="aim",
        extra={
            "returncode": max(returncodes, default=0),
            "segments": len(segments),
            "unique_segments": len(cache),
        },
    )
