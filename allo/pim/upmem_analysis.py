# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conservative MLIR cost summarization for the general UPMEM C path.

The UPMEM runtime retains source/lowered MLIR even when execution ultimately
uses generated C.  :func:`analyze_upmem_mlir` accepts that module object (or
its textual form) and reports source-operation counts, WRAM access counts, and
logical bytes after statically scaling affine/SCF loops.

The summary intentionally reports operation *iterations*, not calibrated DPU
assembly counts.  ``InstructionSummary.cost_metrics`` is therefore suitable
for the standalone UPMEM cost rules: those rules remain the single owner of
native-versus-soft-float instruction expansion.  A later assembly profiler may
instead supply ``instruction_count`` directly to those rules.

Dynamic bounds are never silently treated as one trip.  By default an
:class:`UnsupportedDynamicBoundError` is raised.  Callers may provide an
explicit ``dynamic_trip_count`` upper bound; its use is retained in
``diagnostics`` so estimated artifacts cannot misrepresent it as exact.

The parser targets canonical textual MLIR emitted by Allo/MLIR.  It recognizes
``affine.for`` and ``scf.for`` with integer or constant-SSA bounds, common
arith/math/control operations, scalar/vector loads and stores, and static
``memref.copy``.  Affine maps with symbolic expressions, ``scf.while``, dynamic
memref dimensions, and indirect calls require an explicit dynamic fallback or
are listed as unsupported diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from types import MappingProxyType
from typing import Mapping


class UnsupportedDynamicBoundError(ValueError):
    """Raised when a loop trip count cannot be proven from retained MLIR."""


@dataclass(frozen=True)
class InstructionSummary:
    """One target primitive and numeric kind after loop-trip scaling."""

    primitive: str
    numeric_kind: str
    count: int

    def cost_metrics(self) -> dict[str, object]:
        """Metrics consumed by the standalone UPMEM cost rules."""

        return {
            "iterations": self.count,
            "numeric_kind": self.numeric_kind,
            "summary_source": "mlir",
        }


@dataclass(frozen=True)
class MemorySummary:
    """Logical WRAM accesses represented by the retained MLIR."""

    load_instructions: int = 0
    store_instructions: int = 0
    read_bytes: int = 0
    written_bytes: int = 0
    unknown_byte_accesses: int = 0

    def load_cost_metrics(self) -> dict[str, object]:
        return {
            "iterations": self.load_instructions,
            "bytes": self.read_bytes,
            "summary_source": "mlir",
        }

    def store_cost_metrics(self) -> dict[str, object]:
        return {
            "iterations": self.store_instructions,
            "bytes": self.written_bytes,
            "summary_source": "mlir",
        }


@dataclass(frozen=True)
class UPMEMMLIRSummary:
    """Immutable cost summary returned by :func:`analyze_upmem_mlir`."""

    instructions: tuple[InstructionSummary, ...]
    memory: MemorySummary
    diagnostics: tuple[str, ...] = ()
    unclassified_operations: Mapping[str, int] = MappingProxyType({})

    @property
    def instruction_count(self) -> int:
        """Compute/control primitive iterations, excluding memory accesses."""

        return sum(item.count for item in self.instructions)

    @property
    def total_instruction_count(self) -> int:
        """Conservative total including WRAM and unclassified operations."""

        return (
            self.instruction_count
            + self.memory.load_instructions
            + self.memory.store_instructions
            + sum(self.unclassified_operations.values())
        )

    @property
    def is_exact(self) -> bool:
        """Whether all bounds, byte footprints, and operations were classified."""

        return (
            not self.diagnostics
            and not self.unclassified_operations
            and self.memory.unknown_byte_accesses == 0
        )

    def count(self, primitive: str, numeric_kind: str | None = None) -> int:
        primitive = primitive.upper()
        return sum(
            item.count
            for item in self.instructions
            if item.primitive == primitive
            and (numeric_kind is None or item.numeric_kind == numeric_kind)
        )

    def cost_records(self) -> tuple[tuple[str, dict[str, object]], ...]:
        """Return ``(target primitive, metrics)`` records for a cost wrapper."""

        records = [(item.primitive, item.cost_metrics()) for item in self.instructions]
        if self.memory.load_instructions:
            records.append(("LD_WRAM", self.memory.load_cost_metrics()))
        if self.memory.store_instructions:
            records.append(("ST_WRAM", self.memory.store_cost_metrics()))
        return tuple(records)


_CONSTANT_RE = re.compile(
    r"(?P<name>%[-\w.$]+)\s*=\s*(?:\"arith\.constant\"|arith\.constant)\s+"
    r"(?P<value>-?\d+)\b"
)
_AFFINE_FOR_RE = re.compile(
    r"\baffine\.for\s+%[-\w.$]+\s*=\s*(?P<lb>\S+)\s+to\s+"
    r"(?P<ub>\S+)(?:\s+step\s+(?P<step>\S+))?"
)
_SCF_FOR_RE = re.compile(
    r"\bscf\.for\s+%[-\w.$]+\s*=\s*(?P<lb>\S+)\s+to\s+"
    r"(?P<ub>\S+)\s+step\s+(?P<step>\S+)"
)
_OP_RE = re.compile(
    r"^(?:%[-\w.$]+(?:\s*:\s*\d+)?\s*=\s*)?"
    r"(?:\"(?P<quoted>[A-Za-z_][\w.-]*)\"|(?P<plain>[A-Za-z_][\w.-]*))\b"
)
_LEADING_CLOSE_RE = re.compile(r"^(?:\s*}\s*)+")
_TYPE_RE = re.compile(r"(?P<type>bf16|f16|f32|f64|i\d+|index)(?![A-Za-z0-9_])")
_VECTOR_RE = re.compile(r"vector<(?P<shape>(?:\d+x)*)?(?P<type>bf16|f16|f32|f64|i\d+)>")
_MEMREF_RE = re.compile(
    r"memref<(?P<body>(?:\?|-?\d+|x)+x)?(?P<type>bf16|f16|f32|f64|i\d+)>"
)


_PRIMITIVE_BY_OP = {
    "arith.addi": "ADD",
    "arith.addf": "ADD",
    "arith.subi": "SUB",
    "arith.subf": "SUB",
    "arith.muli": "MUL",
    "arith.mulf": "MUL",
    "arith.divsi": "DIV",
    "arith.divui": "DIV",
    "arith.divf": "DIV",
    "math.sqrt": "SQRT",
    "math.sqrtf": "SQRT",
    "arith.cmpi": "CMP",
    "arith.cmpf": "CMP",
    "arith.select": "SELECT",
    "arith.minsi": "MIN",
    "arith.minui": "MIN",
    "arith.minimumf": "MIN",
    "arith.maxsi": "MAX",
    "arith.maxui": "MAX",
    "arith.maximumf": "MAX",
    "cf.br": "BRANCH",
    "cf.cond_br": "BRANCH",
}

_LOAD_OPS = {
    "affine.load",
    "memref.load",
    "vector.load",
    "vector.transfer_read",
}
_STORE_OPS = {
    "affine.store",
    "memref.store",
    "vector.store",
    "vector.transfer_write",
}
_STRUCTURAL_OPS = {
    "module",
    "builtin.module",
    "func.func",
    "func.return",
    "affine.for",
    "affine.yield",
    "scf.for",
    "scf.yield",
    "return",
    "yield",
    "else",
}


def _strip_comment(line: str) -> str:
    return line.split("//", 1)[0].strip()


def _resolve_bound(
    token: str, constants: Mapping[str, int], symbols: Mapping[str, int]
):
    token = token.rstrip(",{")
    try:
        return int(token)
    except ValueError:
        pass
    for key in (token, token.lstrip("%")):
        if key in symbols:
            return int(symbols[key])
    return constants.get(token)


def _trip_count(lb: int, ub: int, step: int) -> int:
    if step <= 0:
        raise ValueError(
            f"UPMEM cost analyzer requires a positive loop step, got {step}"
        )
    distance = ub - lb
    return 0 if distance <= 0 else (distance + step - 1) // step


def _numeric_kind(line: str, op_name: str) -> str:
    if op_name.endswith("f") or op_name in {
        "math.sqrt",
        "math.sqrtf",
        "arith.minimumf",
        "arith.maximumf",
        "arith.cmpf",
    }:
        return "float"
    types = [match.group("type") for match in _TYPE_RE.finditer(line)]
    if any(value.startswith(("f", "bf")) for value in types):
        return "float"
    return "integer"


def _element_footprint(line: str) -> tuple[int, int] | None:
    """Return ``(elements, bytes)`` for one load/store, if statically known."""

    vector = _VECTOR_RE.search(line)
    if vector is not None:
        dimensions = [int(v) for v in (vector.group("shape") or "").split("x") if v]
        elements = math.prod(dimensions) if dimensions else 1
        bits = _type_bits(vector.group("type"))
        return elements, elements * max(1, math.ceil(bits / 8))
    types = [match.group("type") for match in _TYPE_RE.finditer(line)]
    if not types:
        return None
    bits = _type_bits(types[-1])
    return 1, max(1, math.ceil(bits / 8))


def _type_bits(type_name: str) -> int:
    if type_name == "index":
        return 32
    if type_name == "bf16":
        return 16
    return int(re.search(r"\d+", type_name).group())


def _static_memref_footprint(line: str) -> tuple[int, int] | None:
    match = _MEMREF_RE.search(line)
    if match is None:
        return None
    body = match.group("body") or ""
    if "?" in body:
        return None
    dimensions = [int(v) for v in body.rstrip("x").split("x") if v]
    elements = math.prod(dimensions) if dimensions else 1
    bits = _type_bits(match.group("type"))
    return elements, elements * max(1, math.ceil(bits / 8))


def _operation_name(line: str) -> str | None:
    match = _OP_RE.match(line)
    if match is None:
        return None
    return match.group("quoted") or match.group("plain")


def analyze_upmem_mlir(
    module_or_text,
    *,
    symbol_values: Mapping[str, int] | None = None,
    dynamic_trip_count: int | None = None,
) -> UPMEMMLIRSummary:
    """Summarize retained MLIR for UPMEM analytical costing.

    Parameters
    ----------
    module_or_text:
        An MLIR module/operation with a stable ``str`` representation, or MLIR
        text directly.
    symbol_values:
        Optional values for dynamic SSA/symbol bounds, keyed with or without a
        leading ``%``.
    dynamic_trip_count:
        Explicit conservative fallback for every unresolved loop.  Omitting it
        rejects unsupported dynamic bounds instead of underestimating them.
    """

    text = module_or_text if isinstance(module_or_text, str) else str(module_or_text)
    symbols = dict(symbol_values or {})
    if dynamic_trip_count is not None and int(dynamic_trip_count) < 0:
        raise ValueError("dynamic_trip_count must be non-negative")

    constants: dict[str, int] = {}
    instruction_counts: dict[tuple[str, str], int] = {}
    unclassified: dict[str, int] = {}
    diagnostics: list[str] = []
    load_instructions = store_instructions = 0
    read_bytes = written_bytes = unknown_byte_accesses = 0

    # One multiplier for every open textual region.  Generic module/function/
    # conditional regions retain the current multiplier; loop bodies multiply
    # it by their resolved trip count.
    multipliers = [1]

    def add_instruction(primitive: str, kind: str, count: int) -> None:
        key = (primitive, kind)
        instruction_counts[key] = instruction_counts.get(key, 0) + int(count)

    def dynamic_fallback(line_number: int, line: str) -> int:
        if dynamic_trip_count is None:
            raise UnsupportedDynamicBoundError(
                f"cannot prove loop trip count at MLIR line {line_number}: {line}; "
                "provide symbol_values or an explicit conservative "
                "dynamic_trip_count"
            )
        fallback = int(dynamic_trip_count)
        diagnostics.append(
            f"line {line_number}: used dynamic_trip_count={fallback} for {line}"
        )
        return fallback

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = _strip_comment(raw_line)
        if not line:
            continue

        leading = _LEADING_CLOSE_RE.match(line)
        leading_closes = 0
        if leading is not None:
            leading_closes = leading.group().count("}")
            for _ in range(leading_closes):
                if len(multipliers) > 1:
                    multipliers.pop()
            line = line[leading.end() :].strip()
            if not line:
                continue

        constant = _CONSTANT_RE.search(line)
        if constant is not None:
            constants[constant.group("name")] = int(constant.group("value"))

        outer_multiplier = multipliers[-1]
        loop_match = _AFFINE_FOR_RE.search(line) or _SCF_FOR_RE.search(line)
        loop_opened = False
        if loop_match is not None:
            lb = _resolve_bound(loop_match.group("lb"), constants, symbols)
            ub = _resolve_bound(loop_match.group("ub"), constants, symbols)
            step_token = loop_match.groupdict().get("step") or "1"
            step = _resolve_bound(step_token, constants, symbols)
            if lb is None or ub is None or step is None:
                trips = dynamic_fallback(line_number, line)
            else:
                trips = _trip_count(lb, ub, step)
            # Canonical C loop lowering performs one induction update per trip
            # and one condition/branch per trip plus the terminating check.
            add_instruction("ADD", "integer", outer_multiplier * trips)
            add_instruction("BRANCH", "integer", outer_multiplier * (trips + 1))
            if "{" in line:
                multipliers.append(outer_multiplier * trips)
                loop_opened = True
        elif "scf.while" in line:
            trips = dynamic_fallback(line_number, line)
            add_instruction("BRANCH", "integer", outer_multiplier * (trips + 1))
            if "{" in line:
                multipliers.append(outer_multiplier * trips)
                loop_opened = True

        op_name = _operation_name(line)
        multiplier = outer_multiplier
        if op_name in _PRIMITIVE_BY_OP:
            add_instruction(
                _PRIMITIVE_BY_OP[op_name],
                _numeric_kind(line, op_name),
                multiplier,
            )
        elif op_name in _LOAD_OPS:
            footprint = _element_footprint(line)
            if footprint is None:
                load_instructions += multiplier
                unknown_byte_accesses += multiplier
            else:
                elements, nbytes = footprint
                load_instructions += multiplier * elements
                read_bytes += multiplier * nbytes
        elif op_name in _STORE_OPS:
            footprint = _element_footprint(line)
            if footprint is None:
                store_instructions += multiplier
                unknown_byte_accesses += multiplier
            else:
                elements, nbytes = footprint
                store_instructions += multiplier * elements
                written_bytes += multiplier * nbytes
        elif op_name == "memref.copy":
            footprint = _static_memref_footprint(line)
            if footprint is None:
                unknown_byte_accesses += multiplier
                diagnostics.append(
                    f"line {line_number}: dynamic memref.copy footprint is unknown"
                )
            else:
                elements, nbytes = footprint
                load_instructions += multiplier * elements
                store_instructions += multiplier * elements
                read_bytes += multiplier * nbytes
                written_bytes += multiplier * nbytes
        elif op_name in {"scf.if", "affine.if"}:
            # Both regions are walked and summed: a deliberate upper bound for
            # data-dependent control when branch probabilities are unavailable.
            add_instruction("BRANCH", "integer", multiplier)
            diagnostics.append(
                f"line {line_number}: summed mutually exclusive conditional regions"
            )
        elif (
            op_name is not None
            and op_name not in _STRUCTURAL_OPS
            and op_name != "arith.constant"
        ):
            unclassified[op_name] = unclassified.get(op_name, 0) + multiplier

        open_count = line.count("{")
        close_count = line.count("}")
        if loop_opened:
            open_count -= 1
        for _ in range(max(0, open_count)):
            multipliers.append(multipliers[-1])
        # ``line`` was sliced after any leading closes, so ``close_count`` now
        # contains only braces in the remaining suffix.  This matters for the
        # canonical affine spelling ``} {loop_name = ..., op_name = ...}``:
        # the first brace closes the loop region and the balanced second pair
        # is an attribute dictionary.  Subtracting ``leading_closes`` here used
        # to leave one fake region frame per sequential loop, exponentially
        # multiplying every later PolyBench phase (Deriche reached 1e17).
        for _ in range(close_count):
            if len(multipliers) > 1:
                multipliers.pop()

    instructions = tuple(
        InstructionSummary(primitive, kind, count)
        for (primitive, kind), count in sorted(instruction_counts.items())
        if count
    )
    memory = MemorySummary(
        load_instructions=load_instructions,
        store_instructions=store_instructions,
        read_bytes=read_bytes,
        written_bytes=written_bytes,
        unknown_byte_accesses=unknown_byte_accesses,
    )
    return UPMEMMLIRSummary(
        instructions=instructions,
        memory=memory,
        diagnostics=tuple(diagnostics),
        unclassified_operations=MappingProxyType(dict(sorted(unclassified.items()))),
    )


__all__ = [
    "InstructionSummary",
    "MemorySummary",
    "UPMEMMLIRSummary",
    "UnsupportedDynamicBoundError",
    "analyze_upmem_mlir",
]
