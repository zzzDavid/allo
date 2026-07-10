"""Immutable structural analysis for static affine/SCF MLIR programs."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import math
import re
from typing import TypeAlias


class StructuredProgramAnalysisError(ValueError):
    """Base class for fail-closed structured-program analysis errors."""


class UnsupportedStructuredProgramError(StructuredProgramAnalysisError):
    """Raised when retained MLIR uses semantics this analyzer cannot prove."""


class MalformedStructuredProgramError(StructuredProgramAnalysisError):
    """Raised when retained MLIR is malformed or violates lexical SSA scope."""


@dataclass(frozen=True)
class StaticType:
    """A scalar or statically shaped memref type."""

    dtype: str
    shape: tuple[int, ...] | None = None
    signedness: str | None = None

    @property
    def is_memref(self) -> bool:
        return self.shape is not None


@dataclass(frozen=True)
class ProgramArgument:
    """One function argument, identified semantically by ordinal position."""

    position: int
    name: str
    type: StaticType

    @property
    def shape(self) -> tuple[int, ...] | None:
        return self.type.shape

    @property
    def dtype(self) -> str:
        return self.type.dtype

    @property
    def signedness(self) -> str | None:
        return self.type.signedness


@dataclass(frozen=True)
class StaticLoopAxis:
    """One static induction axis in lexical discovery order."""

    position: int
    name: str
    lower_bound: int
    upper_bound: int
    step: int
    extent: int
    reduction: bool = False


@dataclass(frozen=True)
class MemoryReference:
    """A memref identity normalized away from textual SSA and source names."""

    source_kind: str
    source_position: int
    type: StaticType
    result_position: int = 0

    @property
    def shape(self) -> tuple[int, ...]:
        assert self.type.shape is not None
        return self.type.shape

    @property
    def dtype(self) -> str:
        return self.type.dtype

    @property
    def signedness(self) -> str | None:
        return self.type.signedness


@dataclass(frozen=True)
class ConstantExpression:
    value: int | float | bool | str
    dtype: str
    signedness: str | None = None


@dataclass(frozen=True)
class ArgumentExpression:
    position: int
    dtype: str
    signedness: str | None = None


@dataclass(frozen=True)
class LoopIndexExpression:
    axis_position: int
    dtype: str = "index"
    signedness: str | None = None


@dataclass(frozen=True)
class MemoryAccess:
    """One normalized read or write and its structural index expressions."""

    memory: MemoryReference
    indices: tuple[ScalarExpression, ...]
    mode: str
    signedness: str | None = None

    @property
    def source_kind(self) -> str:
        return self.memory.source_kind

    @property
    def source_position(self) -> int:
        return self.memory.source_position

    @property
    def shape(self) -> tuple[int, ...]:
        return self.memory.shape

    @property
    def dtype(self) -> str:
        return self.memory.dtype


@dataclass(frozen=True)
class LoadExpression:
    access: MemoryAccess
    dtype: str
    signedness: str | None = None


@dataclass(frozen=True)
class CastExpression:
    operation: str
    operand: ScalarExpression
    dtype: str
    signedness: str | None = None


@dataclass(frozen=True)
class UnaryExpression:
    operation: str
    operand: ScalarExpression
    dtype: str
    signedness: str | None = None


@dataclass(frozen=True)
class BinaryExpression:
    operation: str
    lhs: ScalarExpression
    rhs: ScalarExpression
    dtype: str
    signedness: str | None = None


@dataclass(frozen=True)
class ComparisonExpression:
    operation: str
    predicate: str
    lhs: ScalarExpression
    rhs: ScalarExpression
    dtype: str = "i1"
    signedness: str | None = None


@dataclass(frozen=True)
class SelectExpression:
    condition: ScalarExpression
    true_value: ScalarExpression
    false_value: ScalarExpression
    dtype: str
    signedness: str | None = None


@dataclass(frozen=True)
class CallResultExpression:
    call_order: int
    result_position: int
    dtype: str
    signedness: str | None = None


ScalarExpression: TypeAlias = (
    ConstantExpression
    | ArgumentExpression
    | LoopIndexExpression
    | LoadExpression
    | CastExpression
    | UnaryExpression
    | BinaryExpression
    | ComparisonExpression
    | SelectExpression
    | CallResultExpression
)
CallOperand: TypeAlias = ScalarExpression | MemoryReference


@dataclass(frozen=True)
class Predicate:
    """One active branch condition and its true/false polarity."""

    condition: ComparisonExpression
    polarity: bool

    @property
    def operation(self) -> str:
        return self.condition.predicate

    @property
    def lhs(self) -> ScalarExpression:
        return self.condition.lhs

    @property
    def rhs(self) -> ScalarExpression:
        return self.condition.rhs


@dataclass(frozen=True)
class MemoryAllocation:
    position: int
    kind: str
    type: StaticType


@dataclass(frozen=True)
class OrderedStore:
    """One store in source order with its exact loop and predicate context."""

    order: int
    target: MemoryAccess
    value: ScalarExpression
    axes: tuple[StaticLoopAxis, ...]
    predicates: tuple[Predicate, ...] = ()


@dataclass(frozen=True)
class MemoryFill:
    """One static whole-memref fill retained as an ordered memory effect."""

    order: int
    target: MemoryReference
    value: ScalarExpression
    axes: tuple[StaticLoopAxis, ...]
    predicates: tuple[Predicate, ...] = ()


@dataclass(frozen=True)
class FunctionCall:
    """A direct call retained as an ordered conservative semantic event."""

    order: int
    callee_position: int
    arguments: tuple[CallOperand, ...]
    result_types: tuple[StaticType, ...]
    axes: tuple[StaticLoopAxis, ...]
    predicates: tuple[Predicate, ...] = ()


RegionOperation: TypeAlias = OrderedStore | MemoryFill | FunctionCall


@dataclass(frozen=True)
class OrderedRegion:
    """One top-level lexical region and its side effects in source order."""

    order: int
    axes: tuple[StaticLoopAxis, ...]
    operations: tuple[RegionOperation, ...]

    @property
    def stores(self) -> tuple[OrderedStore, ...]:
        return tuple(item for item in self.operations if isinstance(item, OrderedStore))

    @property
    def calls(self) -> tuple[FunctionCall, ...]:
        return tuple(item for item in self.operations if isinstance(item, FunctionCall))

    @property
    def fills(self) -> tuple[MemoryFill, ...]:
        return tuple(item for item in self.operations if isinstance(item, MemoryFill))


@dataclass(frozen=True)
class FunctionReturn:
    order: int
    values: tuple[CallOperand, ...]


@dataclass(frozen=True)
class StructuredFunction:
    """One retained function with normalized arguments and ordered regions."""

    position: int
    name: str
    arguments: tuple[ProgramArgument, ...]
    result_types: tuple[StaticType, ...]
    axes: tuple[StaticLoopAxis, ...]
    allocations: tuple[MemoryAllocation, ...]
    regions: tuple[OrderedRegion, ...]
    returns: tuple[FunctionReturn, ...]
    opaque: bool = False

    @property
    def stores(self) -> tuple[OrderedStore, ...]:
        return tuple(store for region in self.regions for store in region.stores)

    @property
    def calls(self) -> tuple[FunctionCall, ...]:
        return tuple(call for region in self.regions for call in region.calls)

    @property
    def fills(self) -> tuple[MemoryFill, ...]:
        return tuple(fill for region in self.regions for fill in region.fills)


@dataclass(frozen=True)
class StructuredProgram:
    """Target-neutral immutable semantics recovered from retained MLIR."""

    functions: tuple[StructuredFunction, ...]

    def manifest(self) -> dict[str, object]:
        """Return a JSON-compatible manifest retaining diagnostic source names."""

        return _program_manifest(self, canonical=False)

    def canonical_manifest(self) -> dict[str, object]:
        """Return the alpha-renamed manifest used for structural identity."""

        return _program_manifest(self, canonical=True)

    @property
    def canonical_signature(self) -> str:
        payload = json.dumps(
            self.canonical_manifest(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass
class _RawAxis:
    position: int
    name: str
    lower_bound: int
    upper_bound: int
    step: int
    extent: int
    reduction: bool = False


@dataclass
class _RawStore:
    order: int
    target: MemoryAccess
    value: ScalarExpression
    axis_positions: tuple[int, ...]
    predicates: tuple[Predicate, ...]


@dataclass
class _RawCall:
    order: int
    callee_position: int
    arguments: tuple[CallOperand, ...]
    result_types: tuple[StaticType, ...]
    axis_positions: tuple[int, ...]
    predicates: tuple[Predicate, ...]


@dataclass
class _RawFill:
    order: int
    target: MemoryReference
    value: ScalarExpression
    axis_positions: tuple[int, ...]
    predicates: tuple[Predicate, ...]


@dataclass
class _RawRegion:
    order: int
    axis_positions: list[int]
    operations: list[_RawStore | _RawFill | _RawCall]


@dataclass
class _Frame:
    kind: str
    values: dict[str, ScalarExpression | MemoryReference]
    axis_position: int | None = None
    predicate: Predicate | None = None


@dataclass(frozen=True)
class _FunctionHeader:
    position: int
    name: str
    arguments: tuple[ProgramArgument, ...]
    result_types: tuple[StaticType, ...]
    has_body: bool


_SSA = r"%[-\w.$]+"
_SSA_RE = re.compile(rf"^{_SSA}$")
_FUNC_RE = re.compile(r"^\s*func\.func(?:\s+\w+)*\s+@(?P<name>[-\w.$]+)")
_LOOP_RE = re.compile(
    rf"^(?P<dialect>affine|scf)\.for\s+(?P<ssa>{_SSA})\s*=\s*"
    rf"(?P<lb>-?\d+|{_SSA})\s+to\s+(?P<ub>-?\d+|{_SSA})"
    rf"(?:\s+step\s+(?P<step>-?\d+|{_SSA}))?\s*\{{\s*$"
)
_IF_RE = re.compile(rf"^scf\.if\s+(?P<condition>{_SSA})\s*\{{\s*$")
_ELSE_RE = re.compile(r"^\}\s*else\s*\{\s*$")
_CLOSE_RE = re.compile(r"^\}\s*(?:\{(?P<attrs>[^{}]*)\})?\s*$")
_CONSTANT_RE = re.compile(
    rf"^(?P<result>{_SSA})\s*=\s*arith\.constant\s+"
    r"(?P<value>.+?)(?:\s+(?P<attrs>\{[^{}]*\}))?\s*:\s*"
    r"(?P<dtype>index|i\d+|[su]i\d+|bf16|f\d+)\s*$"
)
_LOAD_RE = re.compile(
    rf"^(?P<result>{_SSA})\s*=\s*(?P<op>affine|memref)\.load\s+"
    rf"(?P<memory>{_SSA})\[(?P<indices>.*)\]\s*"
    r"(?P<attrs>\{[^{}]*\})?\s*:\s*(?P<type>memref<.*>)\s*$"
)
_STORE_RE = re.compile(
    rf"^(?P<op>affine|memref)\.store\s+(?P<value>{_SSA})\s*,\s*"
    rf"(?P<memory>{_SSA})\[(?P<indices>.*)\]\s*"
    r"(?P<attrs>\{[^{}]*\})?\s*:\s*(?P<type>memref<.*>)\s*$"
)
_ALLOC_RE = re.compile(
    rf"^(?P<result>{_SSA})\s*=\s*memref\.(?P<kind>alloca?)"
    r"(?:\((?P<dynamic>[^()]*)\))?\s*(?P<attrs>\{[^{}]*\})?\s*:\s*"
    r"(?P<type>memref<.*>)\s*$"
)
_FILL_RE = re.compile(
    rf"^linalg\.fill\s+ins\((?P<value>{_SSA})\s*:\s*"
    r"(?P<value_type>index|i\d+|[su]i\d+|bf16|f\d+)\)\s+"
    rf"outs\((?P<memory>{_SSA})\s*:\s*(?P<memory_type>memref<.*>)\)\s*$"
)
_BINARY_RE = re.compile(
    rf"^(?P<result>{_SSA})\s*=\s*(?P<operation>arith\.\w+)\s+"
    rf"(?P<lhs>{_SSA})\s*,\s*(?P<rhs>{_SSA})(?P<tail>.*)$"
)
_COMPARE_RE = re.compile(
    rf"^(?P<result>{_SSA})\s*=\s*(?P<operation>arith\.cmp[if])\s+"
    rf"(?P<predicate>\w+)\s*,\s*(?P<lhs>{_SSA})\s*,\s*"
    rf"(?P<rhs>{_SSA})(?P<tail>.*)$"
)
_SELECT_RE = re.compile(
    rf"^(?P<result>{_SSA})\s*=\s*arith\.select\s+"
    rf"(?P<condition>{_SSA})\s*,\s*(?P<true>{_SSA})\s*,\s*"
    rf"(?P<false>{_SSA})(?P<tail>.*)$"
)
_CAST_RE = re.compile(
    rf"^(?P<result>{_SSA})\s*=\s*(?P<operation>"
    r"(?:arith\.(?:extf|truncf|extsi|extui|trunci|index_cast|index_castui|"
    r"sitofp|uitofp|fptosi|fptoui|bitcast)|builtin\.unrealized_conversion_cast))"
    rf"\s+(?P<operand>{_SSA})(?P<tail>.*)$"
)
_UNARY_RE = re.compile(
    rf"^(?P<result>{_SSA})\s*=\s*(?P<operation>math\.\w+)\s+"
    rf"(?P<operand>{_SSA})(?P<tail>.*)$"
)
_CALL_RE = re.compile(
    rf"^(?:(?P<results>{_SSA}(?:\s*,\s*{_SSA})*)\s*=\s*)?"
    r"(?:func\.)?call\s+@(?P<callee>[-\w.$]+)\((?P<arguments>.*)\)\s*:\s*"
    r"\((?P<argument_types>.*)\)\s*->\s*(?P<result_types>.+?)\s*$"
)

_BINARY_OPERATIONS = {
    "arith.addi",
    "arith.addf",
    "arith.subi",
    "arith.subf",
    "arith.muli",
    "arith.mulf",
    "arith.divsi",
    "arith.divui",
    "arith.divf",
    "arith.ceildivsi",
    "arith.ceildivui",
    "arith.floordivsi",
    "arith.minsi",
    "arith.minui",
    "arith.minimumf",
    "arith.minf",
    "arith.minnumf",
    "arith.maxsi",
    "arith.maxui",
    "arith.maximumf",
    "arith.maxf",
    "arith.maxnumf",
}
_UNARY_OPERATIONS = {
    "math.absf",
    "math.ceil",
    "math.cos",
    "math.erf",
    "math.exp",
    "math.exp2",
    "math.floor",
    "math.log",
    "math.log10",
    "math.log2",
    "math.round",
    "math.roundeven",
    "math.rsqrt",
    "math.sin",
    "math.sqrt",
    "math.sqrtf",
    "math.tanh",
}
_INTEGER_PREDICATES = {
    "eq",
    "ne",
    "slt",
    "sle",
    "sgt",
    "sge",
    "ult",
    "ule",
    "ugt",
    "uge",
}
_FLOAT_PREDICATES = {
    "false",
    "oeq",
    "ogt",
    "oge",
    "olt",
    "ole",
    "one",
    "ord",
    "ueq",
    "ugt",
    "uge",
    "ult",
    "ule",
    "une",
    "uno",
    "true",
}
_TYPE_TOKEN_RE = re.compile(r"^(?:index|i\d+|[su]i\d+|bf16|f\d+)$")
_INDEX_TOKEN_RE = re.compile(rf"{_SSA}|\d+|ceildiv|floordiv|mod|[()+*\-]")


def _strip_comment(line: str) -> str:
    quoted = False
    escaped = False
    for index, character in enumerate(line):
        if character == '"' and not escaped:
            quoted = not quoted
        if not quoted and line[index : index + 2] == "//":
            return line[:index].strip()
        escaped = character == "\\" and not escaped
        if character != "\\":
            escaped = False
    return line.strip()


def _brace_delta(line: str) -> int:
    quoted = False
    escaped = False
    delta = 0
    for character in line:
        if character == '"' and not escaped:
            quoted = not quoted
        elif not quoted and character == "{":
            delta += 1
        elif not quoted and character == "}":
            delta -= 1
        escaped = character == "\\" and not escaped
        if character != "\\":
            escaped = False
    return delta


def _split_top_level(text: str, delimiter: str = ",") -> tuple[str, ...]:
    if not text.strip():
        return ()
    result = []
    start = 0
    depth = 0
    quoted = False
    for index, character in enumerate(text):
        if character == '"':
            quoted = not quoted
        elif not quoted and character in "<([{":
            depth += 1
        elif not quoted and character in ">)]}":
            depth -= 1
        elif not quoted and character == delimiter and depth == 0:
            result.append(text[start:index].strip())
            start = index + 1
    if depth != 0 or quoted:
        raise MalformedStructuredProgramError(f"unbalanced syntax in {text!r}")
    result.append(text[start:].strip())
    return tuple(item for item in result if item)


def _matching(text: str, start: int, opening: str, closing: str) -> int:
    depth = 0
    quoted = False
    for index in range(start, len(text)):
        character = text[index]
        if character == '"':
            quoted = not quoted
        elif not quoted and character == opening:
            depth += 1
        elif not quoted and character == closing:
            depth -= 1
            if depth == 0:
                return index
    raise MalformedStructuredProgramError(f"unbalanced {opening}{closing} in {text!r}")


def _signed_type(dtype: str, hint: str | None = None) -> tuple[str, str | None]:
    if dtype.startswith("ui"):
        return "i" + dtype[2:], "unsigned"
    if dtype.startswith("si"):
        return "i" + dtype[2:], "signed"
    if dtype.startswith("i"):
        if hint == "u":
            return dtype, "unsigned"
        return dtype, "signed"
    return dtype, None


def _parse_static_type(text: str, hint: str | None = None) -> StaticType:
    text = text.strip()
    if text.startswith("memref<") and text.endswith(">"):
        body = text[7:-1]
        logical = _split_top_level(body)[0]
        parts = logical.split("x")
        dtype_text = parts[-1]
        dimensions = parts[:-1]
        if not _TYPE_TOKEN_RE.fullmatch(dtype_text):
            raise UnsupportedStructuredProgramError(
                f"unsupported memref element type {dtype_text!r}"
            )
        if any(not dimension.isdigit() for dimension in dimensions):
            raise UnsupportedStructuredProgramError(
                f"dynamic or unsupported memref shape {text!r}"
            )
        dtype, signedness = _signed_type(dtype_text, hint)
        return StaticType(dtype, tuple(int(item) for item in dimensions), signedness)
    if not _TYPE_TOKEN_RE.fullmatch(text):
        raise UnsupportedStructuredProgramError(f"unsupported value type {text!r}")
    dtype, signedness = _signed_type(text, hint)
    return StaticType(dtype, None, signedness)


def _parse_type_list(text: str, hints: str | None = None) -> tuple[StaticType, ...]:
    text = text.strip()
    if text in {"", "()"}:
        return ()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    pieces = _split_top_level(text)
    if hints is not None and len(hints) != len(pieces):
        raise MalformedStructuredProgramError(
            "function signedness attribute does not match its type arity"
        )
    return tuple(
        _parse_static_type(piece, hints[index] if hints is not None else None)
        for index, piece in enumerate(pieces)
    )


def _header_attributes(line: str) -> tuple[str | None, str | None]:
    itypes = re.search(r'\bitypes\s*=\s*"([^"]*)"', line)
    otypes = re.search(r'\botypes\s*=\s*"([^"]*)"', line)
    return (
        itypes.group(1) if itypes is not None else None,
        otypes.group(1) if otypes is not None else None,
    )


def _parse_header(line: str, position: int) -> _FunctionHeader:
    match = _FUNC_RE.match(line)
    if match is None:
        raise MalformedStructuredProgramError(f"malformed func.func header: {line}")
    open_paren = line.find("(", match.end())
    if open_paren < 0:
        raise MalformedStructuredProgramError(f"function has no argument list: {line}")
    close_paren = _matching(line, open_paren, "(", ")")
    itypes, otypes = _header_attributes(line)
    argument_parts = _split_top_level(line[open_paren + 1 : close_paren])
    if itypes is not None and len(itypes) != len(argument_parts):
        raise MalformedStructuredProgramError(
            f"function {match.group('name')!r} has mismatched itypes"
        )
    arguments = []
    for argument_position, part in enumerate(argument_parts):
        if ":" not in part:
            raise MalformedStructuredProgramError(
                f"malformed function argument {part!r}"
            )
        name, type_text = part.split(":", 1)
        name = name.strip()
        if _SSA_RE.fullmatch(name) is None:
            raise MalformedStructuredProgramError(f"invalid argument SSA name {name!r}")
        type_text = type_text.strip()
        argument_attrs = ()
        if type_text.endswith("}") and " {" in type_text:
            type_text, attrs_text = type_text.rsplit(" {", 1)
            argument_attrs = _attr_items("{" + attrs_text, {"unsigned", "signed"})
        hint = itypes[argument_position] if itypes is not None else None
        if "unsigned" in argument_attrs:
            if hint not in {None, "u"}:
                raise MalformedStructuredProgramError(
                    f"conflicting signedness for argument {name}"
                )
            hint = "u"
        elif "signed" in argument_attrs and hint == "u":
            raise MalformedStructuredProgramError(
                f"conflicting signedness for argument {name}"
            )
        arguments.append(
            ProgramArgument(
                argument_position,
                name,
                _parse_static_type(type_text, hint),
            )
        )
    remainder = line[close_paren + 1 :].strip()
    result_text = ""
    if remainder.startswith("->"):
        result_text = remainder[2:].strip()
        cut = len(result_text)
        for marker in (" attributes ", " {"):
            marker_index = result_text.find(marker)
            if marker_index >= 0:
                cut = min(cut, marker_index)
        if result_text.endswith("{}"):
            cut = min(cut, len(result_text) - 2)
        result_text = result_text[:cut].strip()
    result_types = _parse_type_list(result_text, otypes) if result_text else ()
    tail_without_attrs = re.sub(r"\battributes\s*\{[^{}]*\}", "", remainder)
    stripped_tail = tail_without_attrs.rstrip()
    has_body = stripped_tail.endswith("{") or stripped_tail.endswith("{}")
    if "{" in stripped_tail and not has_body:
        raise UnsupportedStructuredProgramError(
            "inline func.func bodies are unsupported; print one operation per line"
        )
    return _FunctionHeader(
        position,
        match.group("name"),
        tuple(arguments),
        result_types,
        has_body,
    )


def _extract_functions(lines: tuple[str, ...]):
    chunks = []
    consumed: set[int] = set()
    index = 0
    while index < len(lines):
        if _FUNC_RE.match(lines[index]) is None:
            index += 1
            continue
        start = index
        header = _parse_header(lines[index], len(chunks))
        if not header.has_body or lines[index].rstrip().endswith("{}"):
            chunks.append((header, (lines[index],)))
            consumed.add(index)
            index += 1
            continue
        depth = _brace_delta(lines[index])
        if depth != 1:
            raise MalformedStructuredProgramError(
                f"function body must open on its func.func line: {lines[index]}"
            )
        index += 1
        while index < len(lines) and depth > 0:
            depth += _brace_delta(lines[index])
            if depth < 0:
                raise MalformedStructuredProgramError(
                    "function has unmatched closing brace"
                )
            index += 1
        if depth != 0:
            raise MalformedStructuredProgramError(
                f"function {header.name!r} has an unclosed body"
            )
        chunk = lines[start:index]
        chunks.append((header, chunk))
        consumed.update(range(start, index))
    if not chunks:
        raise MalformedStructuredProgramError("retained MLIR contains no func.func")
    names = [header.name for header, _chunk in chunks]
    if len(names) != len(set(names)):
        raise MalformedStructuredProgramError(
            "retained MLIR has duplicate function names"
        )
    for line_index, line in enumerate(lines):
        if line_index in consumed or not line:
            continue
        if re.match(r"^(?:builtin\.)?module\b.*\{\s*$", line) or line == "}":
            continue
        raise UnsupportedStructuredProgramError(
            f"unsupported top-level MLIR construct on line {line_index + 1}: {line}"
        )
    return tuple(chunks)


def _attr_items(text: str | None, allowed_keys: set[str]) -> tuple[str, ...]:
    if not text:
        return ()
    body = text.strip()
    if body.startswith("{") and body.endswith("}"):
        body = body[1:-1]
    keys = []
    for item in _split_top_level(body):
        key = item.split("=", 1)[0].strip()
        if key not in allowed_keys:
            raise UnsupportedStructuredProgramError(
                f"unsupported semantic attribute {key!r}"
            )
        keys.append(key)
    return tuple(keys)


def _operation_type(tail: str) -> tuple[StaticType, tuple[str, ...]]:
    match = re.fullmatch(
        r"\s*(?P<attrs>\{[^{}]*\})?\s*:\s*"
        r"(?P<type>index|i\d+|[su]i\d+|bf16|f\d+)\s*",
        tail,
    )
    if match is None:
        raise UnsupportedStructuredProgramError(
            f"unsupported operation syntax {tail!r}"
        )
    attrs = _attr_items(match.group("attrs"), {"unsigned", "signed"})
    hint = "u" if "unsigned" in attrs else None
    return _parse_static_type(match.group("type"), hint), attrs


def _cast_type(tail: str) -> tuple[StaticType, StaticType, tuple[str, ...]]:
    match = re.fullmatch(
        r"\s*(?P<attrs>\{[^{}]*\})?\s*:\s*"
        r"(?P<source>index|i\d+|[su]i\d+|bf16|f\d+)\s+to\s+"
        r"(?P<type>index|i\d+|[su]i\d+|bf16|f\d+)\s*",
        tail,
    )
    if match is None:
        raise UnsupportedStructuredProgramError(f"unsupported cast syntax {tail!r}")
    attrs = _attr_items(match.group("attrs"), {"unsigned", "signed"})
    hint = "u" if "unsigned" in attrs else None
    return (
        _parse_static_type(match.group("source")),
        _parse_static_type(match.group("type"), hint),
        attrs,
    )


def _same_logical_type(lhs: StaticType, rhs: StaticType) -> bool:
    return lhs.dtype == rhs.dtype and lhs.shape == rhs.shape


def _value_type(value: ScalarExpression | MemoryReference) -> StaticType:
    if isinstance(value, MemoryReference):
        return value.type
    return StaticType(value.dtype, None, value.signedness)


class _IndexParser:
    def __init__(self, text: str, resolve):
        matches = tuple(_INDEX_TOKEN_RE.finditer(text))
        remainder = _INDEX_TOKEN_RE.sub("", text)
        if remainder.strip():
            raise UnsupportedStructuredProgramError(
                f"unsupported affine index expression {text!r}"
            )
        self.tokens = tuple(match.group() for match in matches)
        self.position = 0
        self.resolve = resolve

    def parse(self) -> ScalarExpression:
        expression = self._additive()
        if self.position != len(self.tokens):
            raise MalformedStructuredProgramError("trailing affine index tokens")
        return expression

    def _peek(self) -> str | None:
        return self.tokens[self.position] if self.position < len(self.tokens) else None

    def _take(self) -> str:
        token = self._peek()
        if token is None:
            raise MalformedStructuredProgramError("incomplete affine index expression")
        self.position += 1
        return token

    def _additive(self) -> ScalarExpression:
        value = self._multiplicative()
        while self._peek() in {"+", "-"}:
            operation = "affine.add" if self._take() == "+" else "affine.sub"
            value = BinaryExpression(operation, value, self._multiplicative(), "index")
        return value

    def _multiplicative(self) -> ScalarExpression:
        value = self._unary()
        operations = {
            "*": "affine.mul",
            "floordiv": "affine.floordiv",
            "ceildiv": "affine.ceildiv",
            "mod": "affine.mod",
        }
        while self._peek() in operations:
            operation = operations[self._take()]
            value = BinaryExpression(operation, value, self._unary(), "index")
        return value

    def _unary(self) -> ScalarExpression:
        if self._peek() == "-":
            self._take()
            return BinaryExpression(
                "affine.sub",
                ConstantExpression(0, "index"),
                self._unary(),
                "index",
            )
        return self._primary()

    def _primary(self) -> ScalarExpression:
        token = self._take()
        if token == "(":
            value = self._additive()
            if self._take() != ")":
                raise MalformedStructuredProgramError("unclosed affine index group")
            return value
        if token.isdigit():
            return ConstantExpression(int(token), "index")
        if _SSA_RE.fullmatch(token):
            return self.resolve(token)
        raise MalformedStructuredProgramError(
            f"unexpected affine index token {token!r}"
        )


class _FunctionParser:
    def __init__(
        self,
        header: _FunctionHeader,
        chunk: tuple[str, ...],
        headers: tuple[_FunctionHeader, ...],
    ):
        self.header = header
        self.chunk = chunk
        self.headers = headers
        self.name_to_position = {item.name: item.position for item in headers}
        base_values: dict[str, ScalarExpression | MemoryReference] = {}
        for argument in header.arguments:
            if argument.type.is_memref:
                base_values[argument.name] = MemoryReference(
                    "argument", argument.position, argument.type
                )
            else:
                base_values[argument.name] = ArgumentExpression(
                    argument.position,
                    argument.dtype,
                    argument.signedness,
                )
        self.frames = [_Frame("function", base_values)]
        self.raw_axes: list[_RawAxis] = []
        self.allocations: list[MemoryAllocation] = []
        self.raw_regions: list[_RawRegion] = []
        self.active_region: _RawRegion | None = None
        self.returns: list[FunctionReturn] = []
        self.event_order = 0
        self.terminated = False

    def parse(self) -> StructuredFunction:
        if not self.header.has_body:
            return StructuredFunction(
                self.header.position,
                self.header.name,
                self.header.arguments,
                self.header.result_types,
                (),
                (),
                (),
                (),
                opaque=True,
            )
        if len(self.chunk) == 1:
            body = ()
        else:
            if self.chunk[-1] != "}":
                raise MalformedStructuredProgramError(
                    f"function {self.header.name!r} has malformed closing syntax"
                )
            body = self.chunk[1:-1]
        for line_number, line in enumerate(body, 2):
            try:
                self._parse_line(line)
            except StructuredProgramAnalysisError as error:
                raise type(error)(
                    f"function {self.header.name!r}, line {line_number}: {error}"
                ) from error
        if len(self.frames) != 1:
            raise MalformedStructuredProgramError(
                f"function {self.header.name!r} has unclosed lexical regions"
            )
        if not self.returns:
            raise MalformedStructuredProgramError(
                f"function {self.header.name!r} has no func.return"
            )
        axes = tuple(
            StaticLoopAxis(
                item.position,
                item.name,
                item.lower_bound,
                item.upper_bound,
                item.step,
                item.extent,
                item.reduction,
            )
            for item in self.raw_axes
        )
        axis_by_position = {axis.position: axis for axis in axes}
        regions = []
        for raw_region in self.raw_regions:
            operations = []
            for operation in raw_region.operations:
                operation_axes = tuple(
                    axis_by_position[position] for position in operation.axis_positions
                )
                if isinstance(operation, _RawStore):
                    operations.append(
                        OrderedStore(
                            operation.order,
                            operation.target,
                            operation.value,
                            operation_axes,
                            operation.predicates,
                        )
                    )
                elif isinstance(operation, _RawFill):
                    operations.append(
                        MemoryFill(
                            operation.order,
                            operation.target,
                            operation.value,
                            operation_axes,
                            operation.predicates,
                        )
                    )
                else:
                    operations.append(
                        FunctionCall(
                            operation.order,
                            operation.callee_position,
                            operation.arguments,
                            operation.result_types,
                            operation_axes,
                            operation.predicates,
                        )
                    )
            regions.append(
                OrderedRegion(
                    raw_region.order,
                    tuple(
                        axis_by_position[position]
                        for position in raw_region.axis_positions
                    ),
                    tuple(operations),
                )
            )
        return StructuredFunction(
            self.header.position,
            self.header.name,
            self.header.arguments,
            self.header.result_types,
            axes,
            tuple(self.allocations),
            tuple(regions),
            tuple(self.returns),
        )

    def _lookup(self, name: str) -> ScalarExpression | MemoryReference:
        for frame in reversed(self.frames):
            if name in frame.values:
                return frame.values[name]
        raise MalformedStructuredProgramError(f"use of undefined SSA value {name}")

    def _scalar(self, name: str) -> ScalarExpression:
        value = self._lookup(name)
        if isinstance(value, MemoryReference):
            raise MalformedStructuredProgramError(f"memref {name} used as a scalar")
        return value

    def _memory(self, name: str) -> MemoryReference:
        value = self._lookup(name)
        if not isinstance(value, MemoryReference):
            raise MalformedStructuredProgramError(f"scalar {name} used as a memref")
        return value

    def _define(self, name: str, value: ScalarExpression | MemoryReference) -> None:
        if name in self.frames[-1].values:
            raise MalformedStructuredProgramError(
                f"duplicate SSA definition {name} in one lexical scope"
            )
        self.frames[-1].values[name] = value

    def _axis_positions(self) -> tuple[int, ...]:
        return tuple(
            frame.axis_position
            for frame in self.frames
            if frame.axis_position is not None
        )

    def _predicates(self) -> tuple[Predicate, ...]:
        return tuple(
            frame.predicate for frame in self.frames if frame.predicate is not None
        )

    def _ensure_region(self, *, structural_root: bool = False) -> _RawRegion:
        if structural_root and self.active_region is not None:
            self.active_region = None
        if self.active_region is None:
            self.active_region = _RawRegion(len(self.raw_regions), [], [])
            self.raw_regions.append(self.active_region)
        return self.active_region

    def _finish_root_region(self) -> None:
        if not self._axis_positions() and not self._predicates():
            self.active_region = None

    def _parse_line(self, line: str) -> None:
        if not line:
            return
        if self.terminated:
            raise MalformedStructuredProgramError("operation appears after func.return")
        if _ELSE_RE.fullmatch(line):
            self._parse_else()
            return
        close = _CLOSE_RE.fullmatch(line)
        if close is not None:
            self._parse_close(close.group("attrs"))
            return
        loop = _LOOP_RE.fullmatch(line)
        if loop is not None:
            self._parse_loop(loop)
            return
        condition = _IF_RE.fullmatch(line)
        if condition is not None:
            self._parse_if(condition.group("condition"))
            return
        allocation = _ALLOC_RE.fullmatch(line)
        if allocation is not None:
            self._parse_allocation(allocation)
            return
        fill = _FILL_RE.fullmatch(line)
        if fill is not None:
            self._parse_fill(fill)
            return
        constant = _CONSTANT_RE.fullmatch(line)
        if constant is not None:
            self._parse_constant(constant)
            return
        load = _LOAD_RE.fullmatch(line)
        if load is not None:
            self._parse_load(load)
            return
        store = _STORE_RE.fullmatch(line)
        if store is not None:
            self._parse_store(store)
            return
        compare = _COMPARE_RE.fullmatch(line)
        if compare is not None:
            self._parse_compare(compare)
            return
        cast = _CAST_RE.fullmatch(line)
        if cast is not None:
            self._parse_cast(cast)
            return
        unary = _UNARY_RE.fullmatch(line)
        if unary is not None:
            self._parse_unary(unary)
            return
        select = _SELECT_RE.fullmatch(line)
        if select is not None:
            self._parse_select(select)
            return
        binary = _BINARY_RE.fullmatch(line)
        if binary is not None:
            self._parse_binary(binary)
            return
        call = _CALL_RE.fullmatch(line)
        if call is not None:
            self._parse_call(call)
            return
        if (
            line.startswith("func.return")
            or line == "return"
            or line.startswith("return ")
        ):
            self._parse_return(line)
            return
        if line in {"affine.yield", "scf.yield"}:
            return
        if line.startswith(("affine.yield ", "scf.yield ")):
            raise UnsupportedStructuredProgramError(
                "loop/conditional results are not supported"
            )
        raise UnsupportedStructuredProgramError(f"unsupported MLIR operation: {line}")

    def _static_int(self, token: str) -> int:
        if re.fullmatch(r"-?\d+", token):
            return int(token)
        value = self._scalar(token)
        evaluated = _evaluate_static_int(value)
        if evaluated is None:
            raise UnsupportedStructuredProgramError(
                f"cannot prove static integer bound {token}"
            )
        return evaluated

    def _parse_loop(self, match: re.Match[str]) -> None:
        lower = self._static_int(match.group("lb"))
        upper = self._static_int(match.group("ub"))
        step_token = match.group("step")
        if match.group("dialect") == "scf" and step_token is None:
            raise UnsupportedStructuredProgramError("scf.for requires a static step")
        step = self._static_int(step_token) if step_token is not None else 1
        if step <= 0:
            raise UnsupportedStructuredProgramError(
                f"loop step must be positive, got {step}"
            )
        distance = upper - lower
        extent = 0 if distance <= 0 else (distance + step - 1) // step
        root = not self._axis_positions() and not self._predicates()
        region = self._ensure_region(structural_root=root)
        position = len(self.raw_axes)
        self.raw_axes.append(
            _RawAxis(
                position,
                match.group("ssa").lstrip("%"),
                lower,
                upper,
                step,
                extent,
            )
        )
        region.axis_positions.append(position)
        self.frames.append(
            _Frame(
                "loop",
                {match.group("ssa"): LoopIndexExpression(position)},
                axis_position=position,
            )
        )

    def _parse_if(self, condition_name: str) -> None:
        condition = self._scalar(condition_name)
        if not isinstance(condition, ComparisonExpression):
            raise UnsupportedStructuredProgramError(
                "scf.if condition must resolve to arith.cmpi or arith.cmpf"
            )
        root = not self._axis_positions() and not self._predicates()
        self._ensure_region(structural_root=root)
        self.frames.append(_Frame("if_then", {}, predicate=Predicate(condition, True)))

    def _parse_else(self) -> None:
        if self.frames[-1].kind != "if_then":
            raise MalformedStructuredProgramError(
                "else does not close an scf.if then-region"
            )
        condition = self.frames.pop().predicate
        assert condition is not None
        self.frames.append(
            _Frame("if_else", {}, predicate=Predicate(condition.condition, False))
        )

    def _parse_close(self, attrs: str | None) -> None:
        if len(self.frames) == 1:
            raise MalformedStructuredProgramError("unmatched lexical closing brace")
        frame = self.frames.pop()
        if frame.kind == "loop":
            items = _split_top_level(attrs or "")
            axis = self.raw_axes[frame.axis_position]
            for item in items:
                key, _, value = item.partition("=")
                key = key.strip()
                if key == "loop_name":
                    name = value.strip()
                    if not (name.startswith('"') and name.endswith('"')):
                        raise MalformedStructuredProgramError(
                            "loop_name must be a string"
                        )
                    axis.name = name[1:-1]
                elif key == "reduction":
                    if value.strip() not in {"", "true"}:
                        raise UnsupportedStructuredProgramError(
                            "reduction attribute must be a unit or true attribute"
                        )
                    axis.reduction = True
                elif key not in {
                    "op_name",
                    "pipeline_ii",
                    "rewind",
                    "unroll",
                    "parallel",
                }:
                    raise UnsupportedStructuredProgramError(
                        f"unsupported loop attribute {key!r}"
                    )
        elif attrs:
            raise MalformedStructuredProgramError(
                "only loop closures may carry loop attributes"
            )
        self._finish_root_region()

    def _parse_allocation(self, match: re.Match[str]) -> None:
        if (match.group("dynamic") or "").strip():
            raise UnsupportedStructuredProgramError(
                "dynamic memref.alloc is unsupported"
            )
        attrs = _attr_items(
            match.group("attrs"), {"name", "alignment", "unsigned", "signed"}
        )
        type_spec = _parse_static_type(
            match.group("type"), "u" if "unsigned" in attrs else None
        )
        if not type_spec.is_memref:
            raise MalformedStructuredProgramError("memref.alloc must produce a memref")
        position = len(self.allocations)
        self.allocations.append(
            MemoryAllocation(position, match.group("kind"), type_spec)
        )
        self._define(
            match.group("result"), MemoryReference("allocation", position, type_spec)
        )

    def _parse_constant(self, match: re.Match[str]) -> None:
        attrs = _attr_items(match.group("attrs"), {"unsigned", "signed"})
        hint = "u" if "unsigned" in attrs else None
        type_spec = _parse_static_type(match.group("dtype"), hint)
        value = _constant_value(match.group("value").strip(), type_spec.dtype)
        self._define(
            match.group("result"),
            ConstantExpression(value, type_spec.dtype, type_spec.signedness),
        )

    def _parse_fill(self, match: re.Match[str]) -> None:
        value = self._scalar(match.group("value"))
        value_type = _parse_static_type(match.group("value_type"))
        memory = self._memory(match.group("memory"))
        memory_type = _parse_static_type(match.group("memory_type"))
        if not _same_logical_type(_value_type(value), value_type):
            raise MalformedStructuredProgramError("linalg.fill scalar type mismatch")
        if not _same_logical_type(memory.type, memory_type):
            raise MalformedStructuredProgramError("linalg.fill memref type mismatch")
        if value.dtype != memory.dtype:
            raise MalformedStructuredProgramError("linalg.fill element type mismatch")
        region = self._ensure_region()
        region.operations.append(
            _RawFill(
                self.event_order,
                memory,
                value,
                self._axis_positions(),
                self._predicates(),
            )
        )
        self.event_order += 1

    def _indices(self, text: str) -> tuple[ScalarExpression, ...]:
        return tuple(
            _IndexParser(item, self._scalar).parse() for item in _split_top_level(text)
        )

    def _memory_access(
        self,
        memory_name: str,
        indices_text: str,
        type_text: str,
        attrs_text: str | None,
        mode: str,
    ) -> MemoryAccess:
        memory = self._memory(memory_name)
        declared = _parse_static_type(type_text)
        if not _same_logical_type(memory.type, declared):
            raise MalformedStructuredProgramError(
                f"memory operation type {declared} does not match {memory.type}"
            )
        allowed = {"from", "to", "unsigned", "signed"}
        attrs = _attr_items(attrs_text, allowed)
        signedness = memory.signedness
        if "unsigned" in attrs:
            signedness = "unsigned"
        elif "signed" in attrs:
            signedness = "signed"
        indices = self._indices(indices_text)
        if len(indices) != len(memory.shape):
            raise MalformedStructuredProgramError(
                f"rank-{len(memory.shape)} memref indexed with {len(indices)} values"
            )
        return MemoryAccess(memory, indices, mode, signedness)

    def _parse_load(self, match: re.Match[str]) -> None:
        access = self._memory_access(
            match.group("memory"),
            match.group("indices"),
            match.group("type"),
            match.group("attrs"),
            "read",
        )
        self._define(
            match.group("result"),
            LoadExpression(access, access.dtype, access.signedness),
        )

    def _parse_store(self, match: re.Match[str]) -> None:
        value = self._scalar(match.group("value"))
        access = self._memory_access(
            match.group("memory"),
            match.group("indices"),
            match.group("type"),
            match.group("attrs"),
            "write",
        )
        if value.dtype != access.dtype:
            raise MalformedStructuredProgramError("store value type mismatch")
        region = self._ensure_region()
        store = _RawStore(
            self.event_order,
            access,
            value,
            self._axis_positions(),
            self._predicates(),
        )
        self.event_order += 1
        region.operations.append(store)

    def _parse_binary(self, match: re.Match[str]) -> None:
        operation = match.group("operation")
        if operation not in _BINARY_OPERATIONS:
            raise UnsupportedStructuredProgramError(
                f"unsupported arithmetic operation {operation}"
            )
        type_spec, _attrs = _operation_type(match.group("tail"))
        lhs = self._scalar(match.group("lhs"))
        rhs = self._scalar(match.group("rhs"))
        if lhs.dtype != type_spec.dtype or rhs.dtype != type_spec.dtype:
            raise MalformedStructuredProgramError("binary operand type mismatch")
        float_operation = operation.endswith("f")
        if float_operation != type_spec.dtype.startswith(("f", "bf")):
            raise MalformedStructuredProgramError(
                f"operation {operation} does not match dtype {type_spec.dtype}"
            )
        signedness = _result_signedness(type_spec, operation, lhs, rhs)
        self._define(
            match.group("result"),
            BinaryExpression(operation, lhs, rhs, type_spec.dtype, signedness),
        )

    def _parse_compare(self, match: re.Match[str]) -> None:
        operand_type, _attrs = _operation_type(match.group("tail"))
        operation = match.group("operation")
        predicate = match.group("predicate")
        allowed = (
            _INTEGER_PREDICATES if operation == "arith.cmpi" else _FLOAT_PREDICATES
        )
        if predicate not in allowed:
            raise UnsupportedStructuredProgramError(
                f"unsupported {operation} predicate {predicate!r}"
            )
        lhs = self._scalar(match.group("lhs"))
        rhs = self._scalar(match.group("rhs"))
        if lhs.dtype != operand_type.dtype or rhs.dtype != operand_type.dtype:
            raise MalformedStructuredProgramError("comparison operand type mismatch")
        self._define(
            match.group("result"),
            ComparisonExpression(operation, predicate, lhs, rhs),
        )

    def _parse_cast(self, match: re.Match[str]) -> None:
        source_type, type_spec, _attrs = _cast_type(match.group("tail"))
        operand = self._scalar(match.group("operand"))
        if operand.dtype != source_type.dtype:
            raise MalformedStructuredProgramError("cast operand type mismatch")
        operation = match.group("operation")
        signedness = _result_signedness(type_spec, operation, operand)
        self._define(
            match.group("result"),
            CastExpression(operation, operand, type_spec.dtype, signedness),
        )

    def _parse_unary(self, match: re.Match[str]) -> None:
        operation = match.group("operation")
        if operation not in _UNARY_OPERATIONS:
            raise UnsupportedStructuredProgramError(
                f"unsupported math operation {operation}"
            )
        type_spec, _attrs = _operation_type(match.group("tail"))
        operand = self._scalar(match.group("operand"))
        if operand.dtype != type_spec.dtype:
            raise MalformedStructuredProgramError("math unary operand type mismatch")
        signedness = _result_signedness(type_spec, operation, operand)
        self._define(
            match.group("result"),
            UnaryExpression(operation, operand, type_spec.dtype, signedness),
        )

    def _parse_select(self, match: re.Match[str]) -> None:
        type_spec, _attrs = _operation_type(match.group("tail"))
        condition = self._scalar(match.group("condition"))
        true_value = self._scalar(match.group("true"))
        false_value = self._scalar(match.group("false"))
        if condition.dtype != "i1":
            raise MalformedStructuredProgramError("arith.select condition is not i1")
        if true_value.dtype != type_spec.dtype or false_value.dtype != type_spec.dtype:
            raise MalformedStructuredProgramError("arith.select value type mismatch")
        signedness = _result_signedness(
            type_spec, "arith.select", true_value, false_value
        )
        self._define(
            match.group("result"),
            SelectExpression(
                condition,
                true_value,
                false_value,
                type_spec.dtype,
                signedness,
            ),
        )

    def _parse_call(self, match: re.Match[str]) -> None:
        callee_name = match.group("callee")
        if callee_name not in self.name_to_position:
            raise UnsupportedStructuredProgramError(
                f"call to undeclared function {callee_name!r}"
            )
        callee_position = self.name_to_position[callee_name]
        callee = self.headers[callee_position]
        argument_names = _split_top_level(match.group("arguments"))
        arguments = tuple(self._lookup(name) for name in argument_names)
        call_argument_types = _parse_type_list(match.group("argument_types"))
        call_result_types = _parse_type_list(match.group("result_types"))
        if len(arguments) != len(call_argument_types):
            raise MalformedStructuredProgramError("func.call argument arity mismatch")
        if len(arguments) != len(callee.arguments):
            raise MalformedStructuredProgramError("func.call callee arity mismatch")
        for value, written, declared in zip(
            arguments, call_argument_types, callee.arguments
        ):
            if not _same_logical_type(
                _value_type(value), written
            ) or not _same_logical_type(written, declared.type):
                raise MalformedStructuredProgramError(
                    "func.call argument type mismatch"
                )
        if len(call_result_types) != len(callee.result_types) or any(
            not _same_logical_type(written, declared)
            for written, declared in zip(call_result_types, callee.result_types)
        ):
            raise MalformedStructuredProgramError("func.call result type mismatch")
        result_names = _split_top_level(match.group("results") or "")
        if len(result_names) != len(call_result_types):
            raise MalformedStructuredProgramError(
                "func.call result assignment mismatch"
            )
        region = self._ensure_region()
        order = self.event_order
        self.event_order += 1
        region.operations.append(
            _RawCall(
                order,
                callee_position,
                arguments,
                callee.result_types,
                self._axis_positions(),
                self._predicates(),
            )
        )
        for result_position, (name, type_spec) in enumerate(
            zip(result_names, callee.result_types)
        ):
            if type_spec.is_memref:
                value: ScalarExpression | MemoryReference = MemoryReference(
                    "call", order, type_spec, result_position
                )
            else:
                value = CallResultExpression(
                    order,
                    result_position,
                    type_spec.dtype,
                    type_spec.signedness,
                )
            self._define(name, value)

    def _parse_return(self, line: str) -> None:
        if self._axis_positions() or self._predicates():
            raise UnsupportedStructuredProgramError(
                "func.return inside a loop or conditional is unsupported"
            )
        remainder = line.removeprefix("func.return").removeprefix("return").strip()
        if not remainder:
            values: tuple[CallOperand, ...] = ()
            written_types: tuple[StaticType, ...] = ()
        else:
            pieces = re.split(r"\s+:\s+", remainder, maxsplit=1)
            if len(pieces) != 2:
                raise MalformedStructuredProgramError(
                    "non-empty func.return requires result types"
                )
            operands_text, types_text = pieces
            values = tuple(
                self._lookup(name) for name in _split_top_level(operands_text)
            )
            written_types = _parse_type_list(types_text)
        if len(values) != len(self.header.result_types):
            raise MalformedStructuredProgramError("func.return result arity mismatch")
        if len(written_types) != len(values):
            raise MalformedStructuredProgramError("func.return type arity mismatch")
        for value, written, expected in zip(
            values, written_types, self.header.result_types
        ):
            if not _same_logical_type(
                _value_type(value), written
            ) or not _same_logical_type(written, expected):
                raise MalformedStructuredProgramError(
                    "func.return result type mismatch"
                )
        self.returns.append(FunctionReturn(self.event_order, values))
        self.event_order += 1
        self.terminated = True


def _constant_value(text: str, dtype: str) -> int | float | bool | str:
    if dtype == "i1" and text in {"true", "false"}:
        return text == "true"
    if dtype == "index" or dtype.startswith("i"):
        if re.fullmatch(r"[+-]?\d+", text) is None:
            raise UnsupportedStructuredProgramError(
                f"unsupported integer constant {text!r}"
            )
        return int(text)
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", text):
        value = float(text)
        if not math.isfinite(value):
            raise UnsupportedStructuredProgramError(
                "non-finite float constants are unsupported"
            )
        return value
    if re.fullmatch(r"0x[0-9A-Fa-f]+", text):
        return text.lower()
    raise UnsupportedStructuredProgramError(f"unsupported floating constant {text!r}")


def _result_signedness(
    type_spec: StaticType,
    operation: str,
    *operands: ScalarExpression,
) -> str | None:
    if not type_spec.dtype.startswith("i"):
        return None
    if operation in {
        "arith.extui",
        "arith.uitofp",
        "arith.fptoui",
        "arith.divui",
        "arith.ceildivui",
        "arith.minui",
        "arith.maxui",
        "arith.index_castui",
    }:
        return "unsigned"
    if operation in {"arith.extsi", "arith.sitofp", "arith.fptosi", "arith.divsi"}:
        return "signed"
    signedness = {operand.signedness for operand in operands if operand.signedness}
    if len(signedness) == 1:
        return signedness.pop()
    return type_spec.signedness


def _evaluate_static_int(expression: ScalarExpression) -> int | None:
    if isinstance(expression, ConstantExpression):
        return expression.value if type(expression.value) is int else None
    if isinstance(expression, CastExpression):
        return _evaluate_static_int(expression.operand)
    if isinstance(expression, BinaryExpression):
        lhs = _evaluate_static_int(expression.lhs)
        rhs = _evaluate_static_int(expression.rhs)
        if lhs is None or rhs is None:
            return None
        if expression.operation in {"arith.addi", "affine.add"}:
            return lhs + rhs
        if expression.operation in {"arith.subi", "affine.sub"}:
            return lhs - rhs
        if expression.operation in {"arith.muli", "affine.mul"}:
            return lhs * rhs
        if expression.operation in {"arith.divsi", "arith.divui"} and rhs:
            return int(lhs / rhs)
    return None


_SOURCE_NAMED_TYPES = (ProgramArgument, StaticLoopAxis, StructuredFunction)


def _manifest_value(value, canonical: bool):
    if isinstance(value, float):
        return {"float_hex": value.hex()}
    if isinstance(value, tuple):
        return [_manifest_value(item, canonical) for item in value]
    if is_dataclass(value):
        result = {"kind": _class_kind(type(value).__name__)}
        for field in fields(value):
            if (
                canonical
                and field.name == "name"
                and isinstance(value, _SOURCE_NAMED_TYPES)
            ):
                continue
            result[field.name] = _manifest_value(getattr(value, field.name), canonical)
        return result
    if value is None or isinstance(value, (bool, int, str)):
        return value
    raise TypeError(f"unsupported manifest value {type(value).__name__}")


def _class_kind(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _program_manifest(
    program: StructuredProgram, *, canonical: bool
) -> dict[str, object]:
    data = _manifest_value(program, canonical)
    data["schema"] = "structured-program-v1"
    return data


def analyze_structured_program(module_or_text) -> StructuredProgram:
    """Analyze retained textual MLIR, rejecting every unproven construct."""

    try:
        text = (
            module_or_text if isinstance(module_or_text, str) else str(module_or_text)
        )
    except Exception as error:
        raise StructuredProgramAnalysisError(
            "retained MLIR object could not be stringified"
        ) from error
    if not text.strip():
        raise MalformedStructuredProgramError("retained MLIR is empty")
    lines = tuple(_strip_comment(line) for line in text.splitlines())
    depth = 0
    for line in lines:
        depth += _brace_delta(line)
        if depth < 0:
            raise MalformedStructuredProgramError("retained MLIR has unmatched braces")
    if depth != 0:
        raise MalformedStructuredProgramError("retained MLIR has unmatched braces")
    chunks = _extract_functions(lines)
    headers = tuple(header for header, _chunk in chunks)
    functions = tuple(
        _FunctionParser(header, chunk, headers).parse() for header, chunk in chunks
    )
    return StructuredProgram(functions)


def canonical_signature(module_or_program) -> str:
    """Return the alpha-renaming-invariant structural SHA-256 signature."""

    program = (
        module_or_program
        if isinstance(module_or_program, StructuredProgram)
        else analyze_structured_program(module_or_program)
    )
    return program.canonical_signature


def manifest(module_or_program, *, canonical: bool = False) -> dict[str, object]:
    """Return a regular or alpha-normalized structural program manifest."""

    program = (
        module_or_program
        if isinstance(module_or_program, StructuredProgram)
        else analyze_structured_program(module_or_program)
    )
    return program.canonical_manifest() if canonical else program.manifest()


__all__ = [
    "ArgumentExpression",
    "BinaryExpression",
    "CallResultExpression",
    "CastExpression",
    "ComparisonExpression",
    "ConstantExpression",
    "FunctionCall",
    "FunctionReturn",
    "LoadExpression",
    "LoopIndexExpression",
    "MalformedStructuredProgramError",
    "MemoryAccess",
    "MemoryAllocation",
    "MemoryFill",
    "MemoryReference",
    "OrderedRegion",
    "OrderedStore",
    "Predicate",
    "ProgramArgument",
    "SelectExpression",
    "ScalarExpression",
    "StaticLoopAxis",
    "StaticType",
    "StructuredFunction",
    "StructuredProgram",
    "StructuredProgramAnalysisError",
    "UnsupportedStructuredProgramError",
    "UnaryExpression",
    "analyze_structured_program",
    "canonical_signature",
    "manifest",
]
