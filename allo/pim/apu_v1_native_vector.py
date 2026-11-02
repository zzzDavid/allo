# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structural recognition core and registry for native APUv1 vector recipes.

Contractions continue through :mod:`apu_v1_vectorize`.  The recipes registered
here (squared-L2 argmin, dense histogram, record frequency, bivariate moments)
lower complete programs whose control flow is not a contraction region.  They
are considered only when a phase explicitly requires vectorization.

Recognition is semantic, not textual.  The retained MLIR of the phase is parsed
into an operation tree with scoped def-use (:func:`parse_native_function`); any
operation outside the supported subset is a recognition failure.  A symbolic
executor (:func:`summarize_native_function`) then

* promotes rank-0 scalar locals (``memref<i16>`` allocs) to SSA values,
* summarizes every loop by the update it applies to each loop-carried local,
* merges ``scf.if`` branches into ``select`` values and records guards,
* records every store to a non-local array as an :class:`NativeEffect` with
  its loop nest, guards, index terms, and value expression, and
* records each array's ``linalg.fill`` initializer.

Values are kept as typed expressions (:class:`E`).  :func:`ring16` folds a
value into its exact meaning modulo ``2**16``: Allo widens operands before
``addi``/``subi``/``muli`` and truncates the result, and those operations
commute with truncation, so the folded expression is exactly the 16-bit lane
value the GVML templates compute.  Widths narrower than 16 bits and other
operations stay opaque, so a recipe that needs them rejects.

Every recipe binds every semantic element its template implements (constants,
initial values, stored locals, comparison predicate and operand order,
reduction operator, operand roles, dtypes, shapes) and raises
:class:`APUv1NativeRecipeMismatch` naming the first difference.  When no recipe
proves a ``vectorize="required"`` phase, :class:`NoNativeVectorRouteError`
reports every recipe's reason, closest recipe first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import functools
import re


class APUv1NativeRecipeMismatch(NotImplementedError):
    """A native recipe cannot prove the phase; ``reason`` names the difference.

    ``stage`` orders how far recognition progressed: 0 = ABI arity/rank, 1 =
    program structure, 2 = a semantic element of a structurally matching
    program, 3 = a shape/capacity limit of the recipe's documented domain.
    """

    def __init__(self, recipe: str, reason: str, *, stage: int = 1):
        self.recipe = recipe
        self.reason = reason
        self.stage = int(stage)
        super().__init__(f"{recipe}: {reason}")


class NoNativeVectorRouteError(NotImplementedError):
    """No native recipe proves a ``vectorize='required'`` APUv1 phase."""

    def __init__(self, function: str, mismatches):
        self.function = function
        self.mismatches = tuple(
            sorted(mismatches, key=lambda item: (-item.stage, item.recipe))
        )
        lines = [
            f"APUv1 phase {function!r} requires vectorization, but no native "
            "vector recipe proves it (closest recipe first):"
        ]
        lines.extend(f"  - {item.recipe}: {item.reason}" for item in self.mismatches)
        lines.append(
            "Rank-2 contractions use the vector-program route: compile the plain "
            "callable with allo.compile instead of an APUv1Program phase."
        )
        super().__init__("\n".join(lines))


# --------------------------------------------------------------------------- #
# Target facts
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class APUv1NativeTargetFacts:
    """Capacities the native templates depend on, read from the target."""

    lanes: int
    lane_bits: int
    writable_vrs: int
    vmr_slots: int
    marker_slots: int
    l3_bytes: int

    @classmethod
    def from_target(cls, target=None) -> "APUv1NativeTargetFacts":
        if target is None:
            from .targets import build_apu_v1_target

            target = build_apu_v1_target()
        if getattr(target, "name", None) != "apu_v1":
            raise ValueError("native APUv1 recipes require the apu_v1 target")
        apuc = target.unit("apuc")
        registers = apuc.registers
        writable = [
            registers[name]
            for name in registers
            if re.fullmatch(r"vr\d+", str(name)) and registers[name].slots == 1
        ]
        if not writable:
            raise ValueError("apu_v1 target declares no writable vector registers")
        lanes = {int(register.lanes) for register in writable}
        widths = {int(register.width) for register in writable}
        if len(lanes) != 1 or len(widths) != 1:
            raise ValueError("apu_v1 writable VRs must share one lane count and width")
        return cls(
            lanes=lanes.pop(),
            lane_bits=widths.pop(),
            writable_vrs=len(writable),
            vmr_slots=int(registers["vmrs"].slots),
            marker_slots=int(registers["markers"].slots),
            l3_bytes=int(apuc.memories["l3"].geometry["size_bytes"]),
        )


# --------------------------------------------------------------------------- #
# Strict parser for the retained Allo MLIR subset
# --------------------------------------------------------------------------- #

_SSA = r"%[-\w.$]+"
_TYPE = r"(?:memref<[^>]*>|[a-z]\w*)"


@dataclass(eq=False)
class NativeValue:
    """One SSA value; identity is object identity (names repeat across scopes)."""

    name: str
    type: str
    kind: str  # "arg" | "iv" | "result"
    owner: "NativeOp | None" = None
    position: int = -1

    # The repr is the tie-break key that orders commutative operands.
    def __repr__(self):
        return f"<{self.name}:{self.type}>"


@dataclass(eq=False)
class NativeOp:
    name: str
    results: tuple = ()
    operands: tuple = ()
    info: dict = field(default_factory=dict)
    regions: list = field(default_factory=list)
    text: str = ""

    def __repr__(self):
        return f"<op {self.name}: {self.text.strip()[:60]}>"


@dataclass(eq=False)
class NativeFunction:
    name: str
    arguments: tuple
    result_types: tuple
    body: list


class _ParseError(ValueError):
    pass


def _int_width(type_text: str) -> int | None:
    match = re.fullmatch(r"i(\d+)", type_text)
    return int(match.group(1)) if match else None


def memref_shape(type_text: str) -> tuple[tuple[int, ...], str] | None:
    """``memref<8x5xi16>`` -> ((8, 5), "i16"); ``memref<i16>`` -> ((), "i16")."""

    match = re.fullmatch(r"memref<((?:\d+x)*)([a-z]\w*)>", type_text)
    if match is None:
        return None
    dims = tuple(int(item) for item in match.group(1).split("x") if item)
    return dims, match.group(2)


class _Scopes:
    def __init__(self):
        self.stack = [{}]

    def push(self):
        self.stack.append({})

    def pop(self):
        self.stack.pop()

    def define(self, value: NativeValue):
        self.stack[-1][value.name] = value

    def lookup(self, name: str) -> NativeValue:
        for scope in reversed(self.stack):
            if name in scope:
                return scope[name]
        raise _ParseError(f"use of undefined SSA value {name}")


_LINE_PATTERNS = {
    "for": re.compile(
        rf"^affine\.for\s+(?P<iv>{_SSA})\s*=\s*(?P<lb>-?\d+)\s+to\s+(?P<ub>-?\d+)"
        r"(?:\s+step\s+(?P<step>\d+))?\s*\{$"
    ),
    "if": re.compile(rf"^scf\.if\s+(?P<cond>{_SSA})\s*\{{$"),
    "constant": re.compile(
        rf"^(?P<res>{_SSA})\s*=\s*arith\.constant\s+(?P<value>-?\d+|true|false)"
        rf"\s*:\s*(?P<type>{_TYPE})$"
    ),
    "cast": re.compile(
        rf"^(?P<res>{_SSA})\s*=\s*arith\.(?P<op>extui|extsi|trunci|index_cast)\s+"
        rf"(?P<src>{_SSA})\s*(?:\{{[^}}]*\}})?\s*:\s*(?P<from>{_TYPE})\s+to\s+"
        rf"(?P<to>{_TYPE})$"
    ),
    "cmpi": re.compile(
        rf"^(?P<res>{_SSA})\s*=\s*arith\.cmpi\s+(?P<pred>\w+)\s*,\s*(?P<lhs>{_SSA})"
        rf"\s*,\s*(?P<rhs>{_SSA})\s*(?:\{{[^}}]*\}})?\s*:\s*(?P<type>{_TYPE})$"
    ),
    "select": re.compile(
        rf"^(?P<res>{_SSA})\s*=\s*arith\.select\s+(?P<cond>{_SSA})\s*,\s*"
        rf"(?P<lhs>{_SSA})\s*,\s*(?P<rhs>{_SSA})\s*(?:\{{[^}}]*\}})?\s*:\s*"
        rf"(?P<type>{_TYPE})$"
    ),
    "binary": re.compile(
        rf"^(?P<res>{_SSA})\s*=\s*arith\.(?P<op>addi|subi|muli|andi|ori|xori|"
        rf"divui|divsi|remui|remsi|shli|shrui|shrsi|minui|minsi|maxui|maxsi)\s+"
        rf"(?P<lhs>{_SSA})\s*,\s*(?P<rhs>{_SSA})\s*(?:overflow<[^>]*>)?\s*"
        rf"(?:\{{[^}}]*\}})?\s*:\s*(?P<type>{_TYPE})$"
    ),
    "load": re.compile(
        rf"^(?P<res>{_SSA})\s*=\s*(?:affine|memref)\.load\s+(?P<mem>{_SSA})"
        rf"\[(?P<idx>[^\]]*)\]\s*(?:\{{[^}}]*\}})?\s*:\s*(?P<type>memref<[^>]*>)$"
    ),
    "store": re.compile(
        rf"^(?:affine|memref)\.store\s+(?P<value>{_SSA})\s*,\s*(?P<mem>{_SSA})"
        rf"\[(?P<idx>[^\]]*)\]\s*(?:\{{[^}}]*\}})?\s*:\s*(?P<type>memref<[^>]*>)$"
    ),
    "alloc": re.compile(
        rf"^(?P<res>{_SSA})\s*=\s*memref\.alloc\(\)\s*(?P<attrs>\{{[^}}]*\}})?\s*:"
        rf"\s*(?P<type>memref<[^>]*>)$"
    ),
    "fill": re.compile(
        rf"^linalg\.fill\s+ins\((?P<value>{_SSA})\s*:\s*(?P<vtype>{_TYPE})\)\s+"
        rf"outs\((?P<mem>{_SSA})\s*:\s*(?P<type>memref<[^>]*>)\)$"
    ),
    "return": re.compile(
        rf"^(?:func\.)?return(?:\s+(?P<values>{_SSA}(?:\s*,\s*{_SSA})*)\s*:.*)?$"
    ),
}
_FUNC = re.compile(
    r"^func\.func\s+(?:private\s+)?@(?P<name>[-\w.$]+)\((?P<args>[^)]*)\)"
    r"(?:\s*->\s*(?P<results>\([^)]*\)|[^{]+?))?\s*(?:attributes\s*\{.*\})?\s*\{$"
)
_CLOSE = re.compile(r"^\}\s*(?:\{(?P<attrs>.*)\})?$")


def _index_terms(text: str, scopes: _Scopes):
    terms = []
    for raw in (part.strip() for part in text.split(",")):
        if not raw:
            continue
        if re.fullmatch(r"-?\d+", raw):
            terms.append(("const", int(raw)))
            continue
        if re.fullmatch(_SSA, raw):
            value = scopes.lookup(raw)
            if value.kind == "iv":
                terms.append(("iv", value.owner))
                continue
        terms.append(("expr", raw))
    return tuple(terms)


# The parser deliberately enumerates the supported operation subset.
# pylint: disable=too-many-branches,too-many-statements,too-many-locals
def _parse(text: str, function: str) -> NativeFunction:
    lines = [line.split("//", 1)[0].strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    index = 0
    while index < len(lines):
        match = _FUNC.match(lines[index])
        if match is not None and match.group("name") == function:
            break
        index += 1
    else:
        raise _ParseError(f"retained MLIR has no function @{function}")

    header = _FUNC.match(lines[index])
    scopes = _Scopes()
    arguments = []
    for position, item in enumerate(
        re.finditer(rf"({_SSA})\s*:\s*({_TYPE})", header.group("args"))
    ):
        value = NativeValue(item.group(1), item.group(2), "arg", position=position)
        scopes.define(value)
        arguments.append(value)
    results_text = (header.group("results") or "").strip().strip("()")
    result_types = tuple(
        item.strip() for item in re.findall(_TYPE, results_text) if item.strip()
    )

    body: list = []
    frames = [("func", None, body)]
    index += 1
    while index < len(lines):
        line = lines[index]
        index += 1
        block = frames[-1][2]
        if line == "} else {":
            kind, op, _ = frames[-1]
            if kind != "then":
                raise _ParseError("unexpected else region")
            scopes.pop()
            scopes.push()
            op.regions.append([])
            frames[-1] = ("else", op, op.regions[1])
            continue
        close = _CLOSE.match(line)
        if close is not None:
            kind, op, _ = frames.pop()
            if kind == "func":
                return NativeFunction(function, tuple(arguments), result_types, body)
            scopes.pop()
            if kind == "for":
                attrs = close.group("attrs") or ""
                loop_name = re.search(r'loop_name\s*=\s*"([^"]+)"', attrs)
                op.info["loop_name"] = loop_name.group(1) if loop_name else None
            continue

        matched = _LINE_PATTERNS["for"].match(line)
        if matched is not None:
            op = NativeOp(
                "affine.for",
                info={
                    "lb": int(matched.group("lb")),
                    "ub": int(matched.group("ub")),
                    "step": int(matched.group("step") or 1),
                },
                regions=[[]],
                text=line,
            )
            iv = NativeValue(matched.group("iv"), "index", "iv", owner=op)
            op.results = (iv,)
            block.append(op)
            scopes.push()
            scopes.define(iv)
            frames.append(("for", op, op.regions[0]))
            continue
        matched = _LINE_PATTERNS["if"].match(line)
        if matched is not None:
            op = NativeOp(
                "scf.if",
                operands=(scopes.lookup(matched.group("cond")),),
                regions=[[]],
                text=line,
            )
            block.append(op)
            scopes.push()
            frames.append(("then", op, op.regions[0]))
            continue

        op = None
        for kind in (
            "constant",
            "cast",
            "cmpi",
            "select",
            "binary",
            "load",
            "store",
            "alloc",
            "fill",
            "return",
        ):
            matched = _LINE_PATTERNS[kind].match(line)
            if matched is None:
                continue
            groups = matched.groupdict()
            if kind == "constant":
                raw = groups["value"]
                value = {"true": 1, "false": 0}.get(raw)
                op = NativeOp(
                    "arith.constant",
                    info={"value": int(raw) if value is None else value},
                )
                result_type = groups["type"]
            elif kind == "cast":
                op = NativeOp(
                    f"arith.{groups['op']}",
                    operands=(scopes.lookup(groups["src"]),),
                    info={"from": groups["from"], "to": groups["to"]},
                )
                result_type = groups["to"]
            elif kind == "cmpi":
                op = NativeOp(
                    "arith.cmpi",
                    operands=(
                        scopes.lookup(groups["lhs"]),
                        scopes.lookup(groups["rhs"]),
                    ),
                    info={"predicate": groups["pred"], "type": groups["type"]},
                )
                result_type = "i1"
            elif kind == "select":
                op = NativeOp(
                    "arith.select",
                    operands=(
                        scopes.lookup(groups["cond"]),
                        scopes.lookup(groups["lhs"]),
                        scopes.lookup(groups["rhs"]),
                    ),
                )
                result_type = groups["type"]
            elif kind == "binary":
                op = NativeOp(
                    f"arith.{groups['op']}",
                    operands=(
                        scopes.lookup(groups["lhs"]),
                        scopes.lookup(groups["rhs"]),
                    ),
                    info={"type": groups["type"]},
                )
                result_type = groups["type"]
            elif kind == "load":
                op = NativeOp(
                    "load",
                    operands=(scopes.lookup(groups["mem"]),),
                    info={
                        "indices": _index_terms(groups["idx"], scopes),
                        "type": groups["type"],
                    },
                )
                shape = memref_shape(groups["type"])
                if shape is None:
                    raise _ParseError(f"unsupported memref type {groups['type']}")
                result_type = shape[1]
            elif kind == "store":
                op = NativeOp(
                    "store",
                    operands=(
                        scopes.lookup(groups["value"]),
                        scopes.lookup(groups["mem"]),
                    ),
                    info={
                        "indices": _index_terms(groups["idx"], scopes),
                        "type": groups["type"],
                    },
                )
                result_type = None
            elif kind == "alloc":
                name = re.search(r'name\s*=\s*"([^"]+)"', groups["attrs"] or "")
                op = NativeOp(
                    "memref.alloc",
                    info={
                        "type": groups["type"],
                        "name": name.group(1) if name else None,
                    },
                )
                result_type = groups["type"]
            elif kind == "fill":
                op = NativeOp(
                    "linalg.fill",
                    operands=(
                        scopes.lookup(groups["value"]),
                        scopes.lookup(groups["mem"]),
                    ),
                )
                result_type = None
            else:
                values = groups.get("values")
                op = NativeOp(
                    "return",
                    operands=tuple(
                        scopes.lookup(item.strip())
                        for item in (values.split(",") if values else ())
                    ),
                )
                result_type = None
            if "res" in groups and groups["res"] is not None:
                result = NativeValue(groups["res"], result_type, "result", owner=op)
                op.results = (result,)
                scopes.define(result)
            break
        if op is None:
            name = re.search(r"([a-z_]+\.[a-z_.]+)", line)
            raise _ParseError(
                f"unsupported operation {name.group(1) if name else line!r} ({line})"
            )
        op.text = line
        block.append(op)
    raise _ParseError(f"function @{function} is not closed")


# --------------------------------------------------------------------------- #
# Symbolic execution
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class E:
    """A typed symbolic value; ``width`` is the integer bit width or None."""

    kind: str
    args: tuple
    width: int | None


@dataclass(eq=False)
class NativeLocal:
    """A rank-0 scalar local promoted to SSA by the executor."""

    alloc: NativeOp
    name: str | None
    width: int | None


@dataclass(frozen=True)
class NativeEffect:
    """One store to a non-local array (function argument or array alloc)."""

    base: NativeValue
    indices: tuple
    value: E
    guards: tuple
    loops: tuple
    op: NativeOp


@dataclass(eq=False)
class NativeLoopSummary:
    loop: NativeOp
    parent_loops: tuple
    init: dict  # NativeLocal -> E (value before the loop)
    updates: dict  # NativeLocal -> E (value after one iteration, carries symbolic)


@dataclass(eq=False)
class NativeSummary:
    function: NativeFunction
    effects: tuple
    fills: dict  # array alloc value -> E
    arrays: dict  # array alloc value -> NativeOp (top-level rank>=1 allocs)
    returned: tuple
    loops: tuple  # every affine.for in program order
    summaries: dict  # loop op -> NativeLoopSummary


class _Unsupported(ValueError):
    pass


def _bits(value: int, width: int | None) -> int:
    return value if width is None else value & ((1 << width) - 1)


class _Executor:
    def __init__(self, function: NativeFunction):
        self.function = function
        self.values: dict = {}
        self.locals: dict = {}  # alloc result value -> NativeLocal
        self.arrays: dict = {}
        self.fills: dict = {}
        self.effects: list = []
        self.loops: list = []
        self.summaries: dict = {}
        self.returned: tuple = ()
        for argument in function.arguments:
            self.values[argument] = E("arg", (argument,), None)

    def value(self, ssa: NativeValue) -> E:
        try:
            return self.values[ssa]
        except KeyError as error:
            raise _Unsupported(
                f"value {ssa.name} has no symbolic definition"
            ) from error

    # pylint: disable=too-many-branches,too-many-statements
    def run_block(self, ops, env, loops, guards, *, top_level=False):
        for op in ops:
            name = op.name
            if name == "arith.constant":
                width = _int_width(op.results[0].type)
                self.values[op.results[0]] = E(
                    "const", (_bits(op.info["value"], width),), width
                )
            elif name in {"arith.extui", "arith.extsi", "arith.trunci"}:
                source = self.value(op.operands[0])
                source_width = _int_width(op.info["from"])
                target_width = _int_width(op.info["to"])
                if source_width is None or target_width is None:
                    raise _Unsupported(f"non-integer cast {op.text}")
                kind = {"arith.extui": "u", "arith.extsi": "s"}.get(name)
                if kind is None:
                    self.values[op.results[0]] = E("trunc", (source,), target_width)
                else:
                    self.values[op.results[0]] = E(
                        "ext", (kind, source_width, source), target_width
                    )
            elif name == "arith.index_cast":
                source = self.value(op.operands[0])
                self.values[op.results[0]] = E(
                    "icast", (source,), _int_width(op.info["to"])
                )
            elif name == "arith.cmpi":
                lhs, rhs = (self.value(item) for item in op.operands)
                self.values[op.results[0]] = E(
                    "cmp",
                    (op.info["predicate"], lhs, rhs),
                    _int_width(op.info["type"]),
                )
            elif name == "arith.select":
                cond, lhs, rhs = (self.value(item) for item in op.operands)
                self.values[op.results[0]] = E("select", (cond, lhs, rhs), lhs.width)
            elif name.startswith("arith."):
                lhs, rhs = (self.value(item) for item in op.operands)
                self.values[op.results[0]] = E(
                    "bin",
                    (name[len("arith.") :], lhs, rhs),
                    _int_width(op.info["type"]),
                )
            elif name == "memref.alloc":
                shape = memref_shape(op.info["type"])
                if shape is None:
                    raise _Unsupported(f"unsupported allocation {op.text}")
                dims, element = shape
                if not dims:
                    local = NativeLocal(op, op.info.get("name"), _int_width(element))
                    self.locals[op.results[0]] = local
                    env[local] = E("undef", (local,), local.width)
                elif top_level:
                    self.arrays[op.results[0]] = op
                    self.values[op.results[0]] = E("array", (op.results[0],), None)
                else:
                    raise _Unsupported(
                        f"array allocation inside a loop or branch ({op.text})"
                    )
            elif name == "linalg.fill":
                fill_value, memory = op.operands
                if memory not in self.arrays or not top_level or loops:
                    raise _Unsupported(f"unsupported fill {op.text}")
                if memory in self.fills:
                    raise _Unsupported(f"array filled twice ({op.text})")
                self.fills[memory] = self.value(fill_value)
            elif name == "load":
                memory = op.operands[0]
                indices = op.info["indices"]
                self._check_indices(indices, op)
                width = _int_width(op.results[0].type)
                if memory in self.locals:
                    local = self.locals[memory]
                    if local not in env:
                        raise _Unsupported(
                            f"local {local.name!r} read outside its scope ({op.text})"
                        )
                    self.values[op.results[0]] = env[local]
                else:
                    self.values[op.results[0]] = E("load", (memory, indices), width)
            elif name == "store":
                value, memory = op.operands
                indices = op.info["indices"]
                self._check_indices(indices, op)
                if memory in self.locals:
                    local = self.locals[memory]
                    if local not in env:
                        raise _Unsupported(
                            f"local {local.name!r} written outside its scope "
                            f"({op.text})"
                        )
                    env[local] = self.value(value)
                else:
                    self.effects.append(
                        NativeEffect(
                            memory,
                            indices,
                            self.value(value),
                            tuple(guards),
                            tuple(loops),
                            op,
                        )
                    )
            elif name == "affine.for":
                self._run_loop(op, env, loops, guards)
            elif name == "scf.if":
                self._run_if(op, env, loops, guards)
            elif name == "return":
                if not top_level:
                    raise _Unsupported("return inside a region")
                self.returned = tuple(op.operands)
            else:
                raise _Unsupported(f"unsupported operation {name!r} ({op.text})")

    @staticmethod
    def _check_indices(indices, op):
        for term in indices:
            if term[0] == "expr":
                raise _Unsupported(
                    f"index {term[1]!r} is not a loop induction variable or "
                    f"integer literal ({op.text})"
                )

    def _run_loop(self, op, env, loops, guards):
        self.loops.append(op)
        if op.info["step"] != 1:
            raise _Unsupported(f"loop step {op.info['step']} ({op.text})")
        self.values[op.results[0]] = E("iv", (op,), None)
        carried = {local: E("carry", (local, op), local.width) for local in list(env)}
        body_env = dict(carried)
        self.run_block(op.regions[0], body_env, loops + (op,), guards)
        updates = {
            local: body_env[local]
            for local in carried
            if body_env[local] != carried[local]
        }
        self.summaries[op] = NativeLoopSummary(
            op, tuple(loops), {local: env[local] for local in updates}, updates
        )
        for local in updates:
            env[local] = E("loopout", (op, local), local.width)

    def _run_if(self, op, env, loops, guards):
        cond = self.value(op.operands[0])
        branches = []
        for region_index, negate in ((0, False), (1, True)):
            branch_env = dict(env)
            if region_index < len(op.regions):
                guard = E("not", (cond,), 1) if negate else cond
                self.run_block(
                    op.regions[region_index], branch_env, loops, guards + (guard,)
                )
            branches.append(branch_env)
        then_env, else_env = branches
        for local in env:
            if then_env[local] != env[local] or else_env[local] != env[local]:
                env[local] = E(
                    "select", (cond, then_env[local], else_env[local]), local.width
                )


@functools.lru_cache(maxsize=64)
def _summarize_cached(source_mlir: str, function: str):
    try:
        parsed = _parse(source_mlir, function)
        executor = _Executor(parsed)
        executor.run_block(parsed.body, {}, (), (), top_level=True)
    except (_ParseError, _Unsupported) as error:
        return None, str(error)
    return (
        NativeSummary(
            parsed,
            tuple(executor.effects),
            dict(executor.fills),
            dict(executor.arrays),
            executor.returned,
            tuple(executor.loops),
            dict(executor.summaries),
        ),
        None,
    )


def summarize_native_function(source_mlir: str, function: str, recipe: str):
    """Parse and symbolically execute ``@function``; raise a recipe mismatch."""

    summary, error = _summarize_cached(source_mlir, function)
    if summary is None:
        raise APUv1NativeRecipeMismatch(recipe, error, stage=1)
    return summary


# --------------------------------------------------------------------------- #
# Modulo-2^16 normalization and comparison views
# --------------------------------------------------------------------------- #

RING_BITS = 16
_COMMUTATIVE = {"addi": "add", "muli": "mul", "andi": "and", "ori": "or", "xori": "xor"}


def _canonical_pair(lhs, rhs):
    return (lhs, rhs) if repr(lhs) <= repr(rhs) else (rhs, lhs)


def ring_product(lhs: E, rhs: E) -> E:
    """The normalized 16-bit product ``lhs * rhs`` (operand order canonical)."""

    return E("mul", _canonical_pair(lhs, rhs), RING_BITS)


# pylint: disable=too-many-return-statements
def ring16(expr: E) -> E:
    """Return ``expr`` modulo 2**16 when that is exactly determined, else opaque."""

    kind = expr.kind
    width = expr.width
    if kind == "const":
        if width is not None and width < RING_BITS:
            return E("opaque", (expr,), None)
        return E("const", (_bits(expr.args[0], RING_BITS),), RING_BITS)
    if kind == "trunc":
        if width is None or width < RING_BITS:
            return E("opaque", (expr,), None)
        return ring16(expr.args[0])
    if kind == "ext":
        ext_kind, source_width, source = expr.args
        if source_width >= RING_BITS:
            return ring16(source)
        if source.kind == "const":
            value = source.args[0]
            if ext_kind == "s" and value >> (source_width - 1) & 1:
                value -= 1 << source_width
            return E("const", (_bits(value, RING_BITS),), RING_BITS)
        return E("opaque", (expr,), None)
    if kind == "icast":
        source = expr.args[0]
        if source.kind == "iv" and (width is None or width >= RING_BITS):
            return E("iv", source.args, RING_BITS)
        return E("opaque", (expr,), None)
    if kind == "bin":
        operation, lhs, rhs = expr.args
        if width is None or width < RING_BITS:
            return E("opaque", (expr,), None)
        lhs, rhs = ring16(lhs), ring16(rhs)
        if operation in _COMMUTATIVE:
            lhs, rhs = _canonical_pair(lhs, rhs)
            return E(_COMMUTATIVE[operation], (lhs, rhs), RING_BITS)
        if operation == "subi":
            return E("sub", (lhs, rhs), RING_BITS)
        return E("opaque", (expr,), None)
    if kind == "load":
        if width != RING_BITS:
            return E("opaque", (expr,), None)
        return E("load", expr.args, RING_BITS)
    if kind in {"carry", "loopout", "undef"}:
        if width != RING_BITS:
            return E("opaque", (expr,), None)
        return E(kind, expr.args, RING_BITS)
    if kind == "select":
        cond, lhs, rhs = expr.args
        return E("select", (cond, ring16(lhs), ring16(rhs)), RING_BITS)
    return E("opaque", (expr,), None)


def compare_view(expr: E):
    """Return ``(ring16 value, extension)`` of one comparison operand.

    ``extension`` is ``None`` for a 16-bit operand, ``"u"``/``"s"`` for a
    value zero/sign extended from 16 bits (outer extensions must preserve
    that value), ``"index"`` for a cast induction variable, and ``"opaque"``
    otherwise.
    """

    if expr.kind == "icast" and expr.args[0].kind == "iv":
        return E("iv", expr.args[0].args, RING_BITS), "index"
    if expr.width == RING_BITS:
        return ring16(expr), None
    outer = []
    current = expr
    while current.kind == "ext":
        ext_kind, source_width, source = current.args
        if source_width == RING_BITS:
            # After a zero extension the sign bit is clear, so any further
            # extension keeps the value; a sign extension needs sign extensions.
            if ext_kind == "s" and any(item != "s" for item in outer):
                break
            return ring16(source), ext_kind
        outer.append(ext_kind)
        current = source
    else:
        if current.kind == "icast" and current.args[0].kind == "iv":
            return E("iv", current.args[0].args, RING_BITS), "index"
    return E("opaque", (expr,), None), "opaque"


_PREDICATE_TEXT = {
    "eq": "==",
    "ne": "!=",
    "slt": "< (signed)",
    "sle": "<= (signed)",
    "sgt": "> (signed)",
    "sge": ">= (signed)",
    "ult": "< (unsigned)",
    "ule": "<= (unsigned)",
    "ugt": "> (unsigned)",
    "uge": ">= (unsigned)",
}


def predicate_text(predicate: str) -> str:
    return _PREDICATE_TEXT.get(predicate, predicate)


def effective_predicate(predicate: str, lhs_extension, rhs_extension) -> str | None:
    """The comparison's meaning on the two operands' 16-bit lane patterns."""

    if predicate in {"eq", "ne"}:
        return predicate
    extensions = {lhs_extension, rhs_extension}
    if extensions == {None}:
        return predicate
    if extensions <= {"u", "index"}:
        return "u" + predicate[1:]
    if extensions == {"s"}:
        return predicate
    return None


def iv_of(term):
    """The loop of an index term or ring value that is an induction variable."""

    if isinstance(term, tuple) and term and term[0] == "iv":
        return term[1]
    if isinstance(term, E) and term.kind == "iv":
        return term.args[0]
    return None


def loop_name(loop: NativeOp) -> str:
    return loop.info.get("loop_name") or loop.results[0].name


def describe(expr: E, names=None) -> str:
    """A short human-readable rendering of a ring expression for diagnostics."""

    names = names or {}

    kind = expr.kind
    if kind == "const":
        return str(expr.args[0])
    if kind == "iv":
        return loop_name(expr.args[0])
    if kind == "load":
        memory, indices = expr.args
        rendered = ", ".join(
            loop_name(term[1]) if term[0] == "iv" else str(term[1]) for term in indices
        )
        name = _memory_name(memory, names)
        return f"{name}[{rendered}]" if indices else name
    if kind in {"carry", "loopout", "undef"}:
        local = expr.args[0] if kind != "loopout" else expr.args[1]
        return local.name or "local"
    symbol = {"add": "+", "sub": "-", "mul": "*", "and": "&", "or": "|", "xor": "^"}
    if kind in symbol:
        lhs, rhs = (describe(item, names) for item in expr.args)
        return f"({lhs} {symbol[kind]} {rhs})"
    if kind == "select":
        lhs, rhs = (describe(item, names) for item in expr.args[1:])
        return f"select({lhs}, {rhs})"
    return "<unsupported expression>"


def _memory_name(memory: NativeValue, names) -> str:
    name = names.get(memory)
    if name is not None:
        return name
    if memory.owner is not None and memory.owner.name == "memref.alloc":
        return memory.owner.info.get("name") or memory.name
    return memory.name


@dataclass(frozen=True)
class NativeABI:
    """Top-level MLIR formals/results bound to the Tenon ABI argument records."""

    sources: tuple  # (NativeValue formal, argument record)
    results: tuple  # (returned NativeValue, argument record)

    def role(self, value: NativeValue):
        for formal, record in self.sources + self.results:
            if formal is value:
                return record
        return None

    @property
    def names(self) -> dict:
        return {formal: record.name for formal, record in self.sources + self.results}


def bind_abi(summary: NativeSummary, arguments, recipe: str) -> NativeABI:
    """Tie MLIR formals/returned allocs to ABI records (names, dtypes, modes)."""

    sources = [item for item in arguments if item.source == "argument"]
    results = [item for item in arguments if item.source == "result"]
    formals = summary.function.arguments
    if len(formals) != len(sources):
        raise APUv1NativeRecipeMismatch(
            recipe, "MLIR formals do not match the source ABI", stage=0
        )
    if len(summary.returned) != len(results):
        raise APUv1NativeRecipeMismatch(
            recipe, "returned values do not match the result ABI", stage=0
        )
    for formal, record in zip(formals, sources):
        shape = memref_shape(formal.type)
        if shape is None or tuple(shape[0]) != tuple(record.shape):
            raise APUv1NativeRecipeMismatch(
                recipe, f"argument {record.name!r} is not a ranked memref", stage=0
            )
    for value, record in zip(summary.returned, results):
        if value not in summary.arrays:
            raise APUv1NativeRecipeMismatch(
                recipe,
                f"result {record.name!r} is not a freshly allocated array",
                stage=0,
            )
    return NativeABI(
        tuple(zip(formals, sources)), tuple(zip(summary.returned, results))
    )


def require_fill_zero(summary, array, role: str, recipe: str):
    """Prove that a returned array starts as all zeros."""

    fill = summary.fills.get(array)
    if fill is None:
        raise APUv1NativeRecipeMismatch(
            recipe,
            f"output {role!r} has no whole-array initializer; the template "
            "requires it to start at zero",
            stage=2,
        )
    value = ring16(fill)
    if value.kind != "const":
        raise APUv1NativeRecipeMismatch(
            recipe,
            f"output {role!r} is initialized to a non-constant value",
            stage=2,
        )
    if value.args[0] != 0:
        raise APUv1NativeRecipeMismatch(
            recipe,
            f"output {role!r} is initialized to {value.args[0]}; the template "
            "implements a zero initial value only",
            stage=2,
        )


def require_no_carried_locals(summary, recipe: str):
    for loop_summary in summary.summaries.values():
        if loop_summary.updates:
            names = sorted(local.name or "local" for local in loop_summary.updates)
            raise APUv1NativeRecipeMismatch(
                recipe,
                f"loop {loop_name(loop_summary.loop)!r} carries scalar locals "
                f"{names} the template does not implement",
                stage=1,
            )


def require_loop(loop, *, lower: int, upper: int, what: str, recipe: str):
    if loop.info["lb"] != lower or loop.info["ub"] != upper or loop.info["step"] != 1:
        raise APUv1NativeRecipeMismatch(
            recipe,
            f"{what} loop {loop_name(loop)!r} runs "
            f"[{loop.info['lb']}, {loop.info['ub']}) step {loop.info['step']}; "
            f"expected [{lower}, {upper}) step 1",
            stage=2,
        )


def match_increment(value: E, counter: E):
    """Return ``k`` when ``value == counter + k`` (mod 2**16), else None."""

    if value.kind != "add":
        return None
    lhs, rhs = value.args
    if lhs == counter and rhs.kind == "const":
        return rhs.args[0]
    if rhs == counter and lhs.kind == "const":
        return lhs.args[0]
    return None


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


def _discover_squared_l2_argmin(source_mlir, arguments, *, function, phase, target):
    from .apu_v1_distance_argmin import (
        APUv1SquaredL2ArgminLowering,
        analyze_apu_v1_squared_l2_argmin,
    )

    return APUv1SquaredL2ArgminLowering(
        analyze_apu_v1_squared_l2_argmin(
            source_mlir,
            arguments,
            function=function,
            argument_bounds=getattr(phase, "argument_bounds", ()),
            target=target,
        )
    )


def _discover_dense_histogram(source_mlir, arguments, *, function, phase, target):
    del phase
    from .apu_v1_histogram import (
        APUv1DenseHistogramLowering,
        analyze_apu_v1_dense_histogram,
    )

    return APUv1DenseHistogramLowering(
        analyze_apu_v1_dense_histogram(
            source_mlir, arguments, function=function, target=target
        )
    )


def _discover_record_frequency(source_mlir, arguments, *, function, phase, target):
    del phase
    from .apu_v1_record_frequency import (
        APUv1RecordFrequencyLowering,
        analyze_apu_v1_record_frequency,
    )

    return APUv1RecordFrequencyLowering(
        analyze_apu_v1_record_frequency(
            source_mlir, arguments, function=function, target=target
        )
    )


def _discover_bivariate_moments(source_mlir, arguments, *, function, phase, target):
    del phase
    from .apu_v1_moments import (
        APUv1BivariateMomentsLowering,
        analyze_apu_v1_bivariate_moments,
    )

    return APUv1BivariateMomentsLowering(
        analyze_apu_v1_bivariate_moments(
            source_mlir, arguments, function=function, target=target
        )
    )


APU_V1_NATIVE_VECTOR_DISCOVERERS = (
    _discover_squared_l2_argmin,
    _discover_dense_histogram,
    _discover_record_frequency,
    _discover_bivariate_moments,
)


def discover_apu_v1_native_vector_lowering(
    phase, artifact, arguments, function, target=None
):
    """Return the unique native recipe proving a ``vectorize='required'`` phase.

    Returns None for phases that do not require vectorization.  Raises
    :class:`NoNativeVectorRouteError` (listing every recipe's mismatch) when
    vectorization is required but no recipe proves the phase.
    """

    if getattr(phase, "vectorize", False) != "required":
        return None
    matches = []
    mismatches = []
    for discover in APU_V1_NATIVE_VECTOR_DISCOVERERS:
        try:
            matches.append(
                discover(
                    artifact.source_mlir,
                    arguments,
                    function=function,
                    phase=phase,
                    target=target,
                )
            )
        except APUv1NativeRecipeMismatch as mismatch:
            mismatches.append(mismatch)
    if len(matches) > 1:
        routes = ", ".join(item.route for item in matches)
        raise ValueError(f"ambiguous native APUv1 vector lowering: {routes}")
    if not matches:
        raise NoNativeVectorRouteError(function, mismatches)
    return matches[0]


__all__ = [
    "APUv1NativeRecipeMismatch",
    "APUv1NativeTargetFacts",
    "APU_V1_NATIVE_VECTOR_DISCOVERERS",
    "NoNativeVectorRouteError",
    "discover_apu_v1_native_vector_lowering",
    "ring16",
    "ring_product",
    "summarize_native_function",
]
