# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared semantic program contract (release gate G9, stage 1).

One analysis over the matcher's :class:`~allo.spmw_match.MatchTrace` and the
retained MLIR module establishes the facts every backend planner consumes:

* the function's parameters (ordinal, shape, element type, signedness,
  retained identity);
* every matched contraction: its loop nest with static trip counts and
  source loop names, the structural axis roles the matcher proved
  (reduction, output and batch loops), the two inputs and the accumulator
  as :class:`BufferAccess` records with resolved coordinates, the
  accumulator initialization fact, the copies of the result into other
  buffers (:class:`OutputSink`), guards, the product and accumulate
  operations, the element types and the reduction-order policy;
* every matched epilogue (a non-accumulating match such as Gemini-II's
  ``AXPBY``) with its operand identities, literal constants and store
  coordinates.

The facts are *derived* from the matcher's structural analysis
(``axis_roles``, ``accumulator_init``, ``output_sinks``, ``store_coords``,
``guards``) and the retained IR types.  Nothing here re-parses MLIR text.
Backend planners (:mod:`allo.pim.apu_v1_vectorize`,
:mod:`allo.pim.apu_g2_source_route`) refine these facts with target
legality; they may reject a program, never reinterpret it.

A fact that cannot be established is recorded as such (``roles_unresolved``,
``dims`` entries of ``None``, element ``None``) so a consumer fails closed
instead of guessing.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from .spmw_match import IRValueRef, MatchTrace, MatchedOp, OperandBinding


class SemanticAnalysisError(ValueError):
    """The shared analysis cannot establish a fact a consumer requires."""


# --------------------------------------------------------------------- #
# Element and buffer types
# --------------------------------------------------------------------- #

_MEMREF_RE = re.compile(r"^memref<((?:\d+x)*)([A-Za-z_][A-Za-z0-9_]*)>$")
_INT_RE = re.compile(r"^(u?)i(\d+)$")
_FLOAT_BITS = {"f16": 16, "bf16": 16, "f32": 32, "f64": 64}


@dataclass(frozen=True)
class ElementType:
    """A scalar MLIR element type with the source signedness attached.

    ``text`` is the MLIR spelling (``f16``, ``i15``); MLIR integers are
    signless, so ``signed`` comes from the frontend's per-argument
    ``itypes`` codes or the ``unsigned`` attribute of a local allocation.
    ``signed`` is ``None`` for floating-point types and for integers whose
    signedness is not retained.
    """

    text: str
    bits: int
    kind: str  # "float" | "int" | "index"
    signed: bool | None = None

    @property
    def is_float(self) -> bool:
        return self.kind == "float"

    @property
    def is_integer(self) -> bool:
        return self.kind == "int"

    @property
    def name(self) -> str:
        """Allo-style name (``float16``, ``uint16``, ``int15``)."""

        if self.kind == "float":
            return {"f16": "float16", "bf16": "bfloat16", "f32": "float32"}.get(
                self.text, f"float{self.bits}"
            )
        if self.kind == "index":
            return "index"
        prefix = "uint" if self.signed is False else "int"
        return f"{prefix}{self.bits}"

    @property
    def source_text(self) -> str:
        """The spelling the APU v1 planner historically used (``ui16`` for
        an unsigned integer, the MLIR text otherwise)."""

        if self.kind == "int" and self.signed is False:
            return f"ui{self.bits}"
        return self.text

    @classmethod
    def parse(cls, text: str, signed: bool | None = None) -> "ElementType":
        text = str(text).strip()
        if text in _FLOAT_BITS:
            return cls(text, _FLOAT_BITS[text], "float", None)
        if text == "index":
            return cls(text, 64, "index", None)
        found = _INT_RE.match(text)
        if found is None:
            raise SemanticAnalysisError(f"unsupported element type {text!r}")
        bits = int(found.group(2))
        if found.group(1) == "u":
            signed = False
        return cls(f"i{bits}", bits, "int", signed)


def parse_memref_type(text: str | None) -> tuple[tuple[int, ...], str] | None:
    """``memref<4x8xf16>`` -> ``((4, 8), "f16")``; None when not retained."""

    if text is None:
        return None
    found = _MEMREF_RE.match(str(text).strip())
    if found is None:
        return None
    dims = tuple(int(item) for item in found.group(1).split("x") if item)
    return dims, found.group(2)


@dataclass(frozen=True)
class Parameter:
    """One function argument of the analyzed function."""

    name: str
    ordinal: int
    shape: tuple[int, ...]
    element: ElementType
    value_ref: IRValueRef | None


@dataclass(frozen=True)
class LoopAxis:
    """One enclosing loop of a match, outermost first in ``loops``."""

    var: str  # retained induction-variable name (unique in the function)
    name: str | None  # source loop name (``loop_name`` attribute) when kept
    lower: int | None
    upper: int | None
    step: int

    @property
    def trip(self) -> int | None:
        if self.lower is None or self.upper is None or self.step <= 0:
            return None
        return max(0, -(-(self.upper - self.lower) // self.step))

    @property
    def normalized(self) -> bool:
        """``range(N)`` with a static N: lower 0, step 1."""

        return self.lower == 0 and self.step == 1 and self.upper is not None

    @property
    def label(self) -> str:
        return self.name if self.name is not None else self.var


@dataclass(frozen=True)
class BufferAccess:
    """One buffer access of a match with its resolved coordinates.

    ``dims[d]`` is the loop variable that indexes dimension ``d`` when the
    coordinate is exactly that variable (coefficient 1, offset 0), else
    ``None`` (an affine expression, a constant or a data-dependent index).
    ``coords`` keeps the full resolved forms (`spmw_match_engine.affine_form`).
    """

    role: str
    memref_name: str | None
    parameter: str | None  # function-argument name, None for a local buffer
    local: bool
    value_ref: IRValueRef | None
    shape: tuple[int, ...] | None
    element: ElementType | None
    coords: tuple | None
    dims: tuple[str | None, ...]
    indices: tuple[str, ...]
    index_map: str | None

    @property
    def plain_dims(self) -> tuple[str, ...]:
        """Loop variables of every dimension; raises when one is not plain."""

        if any(dim is None for dim in self.dims):
            raise SemanticAnalysisError(
                f"access to {self.memref_name!r} is not indexed by plain loop "
                "variables in every dimension"
            )
        return tuple(self.dims)  # type: ignore[arg-type]


@dataclass(frozen=True)
class OutputSink:
    """A copy of a contraction's result into another buffer (the matcher's
    ``output_sinks``: ``C[i, j] = acc``)."""

    memref_name: str | None
    parameter: str | None
    local: bool | None
    value_ref: IRValueRef | None
    shape: tuple[int, ...] | None
    element: ElementType | None
    coords: tuple | None
    dims: tuple[str | None, ...]
    indices: tuple[str, ...]


@dataclass(frozen=True)
class Contraction:  # pylint: disable=too-many-instance-attributes
    """One matched accumulating operation and the facts the matcher proved."""

    match: MatchedOp
    function: str
    target_op_name: str
    loops: tuple[LoopAxis, ...]
    reduction_vars: tuple[str, ...]
    output_vars: tuple[str, ...]
    batch_vars: tuple[str, ...]
    roles_unresolved: str | None
    inputs: tuple[BufferAccess, ...]  # pattern order (x, y)
    matrix: BufferAccess | None
    vector: BufferAccess | None
    accumulator: BufferAccess
    accumulator_init: str | None  # "zero" | "carried" | "unsupported" | None
    sinks: tuple[OutputSink, ...]
    store_coords: tuple | None
    guards: tuple
    product_kind: str  # normalized: "mul" | "and" | "or" | "xor" | ...
    accumulate_kind: str  # normalized: "add"
    product_operation: str  # MLIR spelling, e.g. "arith.mulf"
    accumulate_operation: str  # MLIR spelling, e.g. "arith.addf"
    reduction_policy: Any  # spmw_reduction_policy.ReductionPolicy or None
    same_buffer: bool
    vector_width: int

    @property
    def loop_vars(self) -> tuple[str, ...]:
        return tuple(loop.var for loop in self.loops)

    @property
    def result_vars(self) -> tuple[str, ...]:
        """Output and batch loops in loop-nest order."""

        result = set(self.output_vars) | set(self.batch_vars)
        return tuple(var for var in self.loop_vars if var in result)

    @property
    def trips(self) -> dict[str, int | None]:
        return {loop.var: loop.trip for loop in self.loops}

    @property
    def loop_labels(self) -> dict[str, str]:
        return {loop.var: loop.label for loop in self.loops}

    def loop(self, var: str) -> LoopAxis:
        for loop in self.loops:
            if loop.var == var:
                return loop
        raise KeyError(var)

    @property
    def numeric(self) -> ElementType | None:
        """Element type of the accumulation (the accumulator's)."""

        return self.accumulator.element

    @property
    def input_aliases_accumulator(self) -> tuple[str, ...]:
        """Inputs that read the accumulator's buffer or one of its sinks."""

        refs = {self.accumulator.value_ref} | {sink.value_ref for sink in self.sinks}
        refs.discard(None)
        return tuple(
            access.role
            for access in self.inputs
            if access.value_ref is not None and access.value_ref in refs
        )

    @property
    def external_sinks(self) -> tuple[OutputSink, ...]:
        return tuple(sink for sink in self.sinks if not sink.local)


@dataclass(frozen=True)
class Epilogue:
    """A non-accumulating match (e.g. ``alpha * x + beta * y``) whose
    operands may be the results of contractions."""

    match: MatchedOp
    function: str
    target_op_name: str
    loops: tuple[LoopAxis, ...]
    operands: tuple[BufferAccess, ...]
    constants: dict[str, Any]
    result: BufferAccess
    store_coords: tuple | None
    guards: tuple

    @property
    def loop_vars(self) -> tuple[str, ...]:
        return tuple(loop.var for loop in self.loops)


@dataclass(frozen=True)
class ProgramFacts:
    """The semantic contract of one analyzed function."""

    function: str
    parameters: tuple[Parameter, ...]
    contractions: tuple[Contraction, ...]
    epilogues: tuple[Epilogue, ...]
    other_matches: tuple[MatchedOp, ...]
    foreign_functions: tuple[str, ...]  # functions with matches outside this one
    trace: MatchTrace = field(repr=False)
    # A typed program record these facts were normalized from (a frontend
    # without source); None for ordinary source.  The backend enumerator of
    # the record's target turns it into the record's candidate.
    frontend: Any = None

    def parameter(self, name: str) -> Parameter:
        for parameter in self.parameters:
            if parameter.name == name:
                return parameter
        raise KeyError(name)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(parameter.name for parameter in self.parameters)

    def parameter_by_ref(self, ref: IRValueRef | None) -> Parameter | None:
        if ref is None:
            return None
        for parameter in self.parameters:
            if parameter.value_ref == ref:
                return parameter
        return None

    def manifest(self) -> dict[str, Any]:
        """Name-free structural digest of the facts (renamed programs give
        equal manifests): loop labels are positional, buffers are ordinals."""

        ordinals = {p.value_ref: p.ordinal for p in self.parameters}

        def access(item):
            if item is None:
                return None
            return {
                "buffer": ordinals.get(item.value_ref, "local"),
                "shape": None if item.shape is None else list(item.shape),
                "element": None if item.element is None else item.element.name,
                "dims": list(item.dims),
                "coords": repr(item.coords),
            }

        out = {
            "parameters": [
                {"shape": list(p.shape), "element": p.element.name}
                for p in self.parameters
            ],
            "contractions": [],
            "epilogues": [],
        }
        for item in self.contractions:
            out["contractions"].append(
                {
                    "op": item.target_op_name,
                    "loops": [
                        (loop.var, loop.lower, loop.upper, loop.step)
                        for loop in item.loops
                    ],
                    "reduction": list(item.reduction_vars),
                    "output": list(item.output_vars),
                    "batch": list(item.batch_vars),
                    "inputs": [access(a) for a in item.inputs],
                    "accumulator": access(item.accumulator),
                    "accumulator_init": item.accumulator_init,
                    "sinks": [access(s) for s in item.sinks],
                    "guards": len(item.guards),
                    "product": item.product_operation,
                    "accumulate": item.accumulate_operation,
                    "policy": getattr(
                        item.reduction_policy, "order", item.reduction_policy
                    ),
                }
            )
        for item in self.epilogues:
            out["epilogues"].append(
                {
                    "op": item.target_op_name,
                    "operands": [access(a) for a in item.operands],
                    "constants": dict(item.constants),
                    "result": access(item.result),
                    "coords": repr(item.store_coords),
                }
            )
        return out


# --------------------------------------------------------------------- #
# Module facts: argument signedness, local allocation signedness, types
# --------------------------------------------------------------------- #


def _attr_text(operation, name: str) -> str | None:
    if name not in operation.attributes:
        return None
    text = str(operation.attributes[name])
    return text[1:-1] if text.startswith('"') and text.endswith('"') else text


def _walk(operation):
    for region in operation.regions:
        for block in region.blocks:
            for child in block.operations:
                yield child
                yield from _walk(child)


class _ModuleFacts:
    """Retained-type and signedness facts of one function, keyed by the
    matcher's canonical value references (the same canonicalization
    `match_workload` used, so identities agree)."""

    def __init__(self, module, function_name: str):
        from .spmw_match_engine import _build_ir_value_refs

        self.refs, _sources = _build_ir_value_refs(module)
        self.function = None
        for operation in module.body.operations:
            if operation.operation.name != "func.func":
                continue
            if _attr_text(operation, "sym_name") == function_name:
                self.function = operation
                break
        if self.function is None:
            raise SemanticAnalysisError(
                f"retained module has no function {function_name!r}"
            )
        codes = _attr_text(self.function, "itypes") or ""
        self.argument_types: list[tuple[str, bool | None]] = []
        memref_arguments = [
            argument
            for argument in self.function.arguments
            if str(argument.type).startswith("memref<")
        ]
        self.argument_refs = [self.refs.get(a) for a in memref_arguments]
        # ``itypes`` has one code per Python argument; memref arguments are
        # the ones the matcher binds.  Scalars are not part of the facts.
        if len(codes) != len(self.function.arguments):
            codes = ""
        for index, argument in enumerate(self.function.arguments):
            code = codes[index] if codes else None
            signed = {"s": True, "u": False}.get(code) if code else None
            self.argument_types.append((str(argument.type), signed))
        self.type_by_ref: dict = {}
        self.name_by_ref: dict = {}
        self.signed_by_ref: dict = {}
        for argument, (text, signed) in zip(
            self.function.arguments, self.argument_types
        ):
            ref = self.refs.get(argument)
            if ref is not None:
                self.type_by_ref[ref] = text
                self.signed_by_ref[ref] = signed
        for operation in _walk(self.function):
            if operation.operation.name in {"memref.alloc", "memref.alloca"}:
                if not operation.results:
                    continue
                ref = self.refs.get(operation.results[0])
                if ref is None:
                    continue
                self.type_by_ref[ref] = str(operation.results[0].type)
                name = _attr_text(operation, "name")
                if name is not None:
                    self.name_by_ref[ref] = name
                signed = "unsigned" not in operation.attributes
                previous = self.signed_by_ref.get(ref)
                if previous is not None and previous != signed:
                    # Two declarations of one buffer disagree: not retained.
                    self.signed_by_ref[ref] = None
                else:
                    self.signed_by_ref[ref] = signed

    def element(self, memref_type: str | None, ref) -> ElementType | None:
        parsed = parse_memref_type(memref_type)
        if parsed is None and ref is not None:
            parsed = parse_memref_type(self.type_by_ref.get(ref))
        if parsed is None:
            return None
        _shape, text = parsed
        try:
            return ElementType.parse(text, self.signed_by_ref.get(ref))
        except SemanticAnalysisError:
            return None

    def shape(self, memref_type: str | None, ref) -> tuple[int, ...] | None:
        parsed = parse_memref_type(memref_type)
        if parsed is None and ref is not None:
            parsed = parse_memref_type(self.type_by_ref.get(ref))
        return None if parsed is None else parsed[0]


# --------------------------------------------------------------------- #
# Deriving the facts
# --------------------------------------------------------------------- #


def _static_bound(text) -> int | None:
    from .spmw_match_engine import _parse_loop_upper

    return _parse_loop_upper(str(text))


def _loops_of(match: MatchedOp) -> tuple[LoopAxis, ...]:
    names = match.extra.get("loop_names") or {}
    loops = []
    for var, lower, upper, step in match.enclosing_loops:
        try:
            step_value = int(step)
        except (TypeError, ValueError):
            step_value = 0
        loops.append(
            LoopAxis(
                var=str(var),
                name=names.get(var),
                lower=_static_bound(lower),
                upper=_static_bound(upper),
                step=step_value,
            )
        )
    return tuple(loops)


def _plain_dims(coords, loop_vars: set[str]) -> tuple[str | None, ...]:
    if coords is None:
        return ()
    dims = []
    for coord in coords:
        if (
            coord is not None
            and len(coord[0]) == 1
            and coord[1] == 0
            and coord[0][0][1] == 1
            and coord[0][0][0] in loop_vars
        ):
            dims.append(coord[0][0][0])
        else:
            dims.append(None)
    return tuple(dims)


def _access(
    binding: OperandBinding,
    facts: _ModuleFacts,
    parameters_by_ref: dict,
    loop_vars: set[str],
) -> BufferAccess:
    ref = binding.value_ref
    parameter = parameters_by_ref.get(ref)
    coords = binding.coords
    dims = _plain_dims(coords, loop_vars)
    if coords is None:
        # No resolved coordinates (a match assembled outside a function
        # scope): fall back to the textual indices when each is a loop var.
        dims = tuple(index if index in loop_vars else None for index in binding.indices)
    return BufferAccess(
        role=binding.role,
        memref_name=binding.memref_name,
        parameter=None if parameter is None else parameter.name,
        local=parameter is None,
        value_ref=ref,
        shape=facts.shape(binding.memref_type, ref),
        element=facts.element(binding.memref_type, ref),
        coords=coords,
        dims=dims,
        indices=tuple(binding.indices),
        index_map=binding.index_map,
    )


def _sinks(
    match: MatchedOp,
    accumulator: BufferAccess,
    facts: _ModuleFacts,
    parameters_by_ref: dict,
    loop_vars: set[str],
) -> tuple[OutputSink, ...]:
    sinks = list(match.extra.get("output_sinks") or ())
    coords = list(match.extra.get("output_sink_coords") or ())
    refs = list(match.extra.get("output_sink_refs") or ())
    store_coords = match.extra.get("store_coords")
    out = []
    for position, (name, index_text) in enumerate(sinks):
        sink_coords = coords[position] if position < len(coords) else None
        ref, local = refs[position] if position < len(refs) else (None, None)
        if (
            name == accumulator.memref_name
            and ref in (None, accumulator.value_ref)
            and sink_coords == store_coords
        ):
            # The carried accumulator's own store (recorded by the matcher as
            # the result's sink) is not a copy.
            continue
        parameter = parameters_by_ref.get(ref)
        if local is None and ref is not None:
            local = parameter is None
        out.append(
            OutputSink(
                memref_name=name,
                parameter=None if parameter is None else parameter.name,
                local=local,
                value_ref=ref,
                shape=facts.shape(None, ref),
                element=facts.element(None, ref),
                coords=sink_coords,
                dims=_plain_dims(sink_coords, loop_vars),
                indices=tuple(item.strip() for item in str(index_text).split(",")),
            )
        )
    return tuple(out)


def _pattern_kinds(target, op_name: str, acc_role: str | None) -> tuple[str, str]:
    """``(product kind, accumulate kind)`` of an accumulating target op, read
    from its compiled pattern ``acc + f(x, y)``."""

    from .spmw_match_engine import PBinOp, PUnary, PVar, compile_op_pattern

    if target is None:
        return "mul", "add"
    try:
        op_obj = target.op(op_name)
    except KeyError:
        return "mul", "add"
    body = compile_op_pattern(op_obj).body
    if not isinstance(body, PBinOp):
        return "?", "?"
    for side, other in ((body.lhs, body.rhs), (body.rhs, body.lhs)):
        if isinstance(side, PVar) and (acc_role is None or side.name == acc_role):
            if isinstance(other, PBinOp):
                return other.op, body.op
            if isinstance(other, PUnary):
                return other.fn, body.op
            return "?", body.op
    return "?", body.op


_MLIR_INT = {
    "mul": "arith.muli",
    "add": "arith.addi",
    "and": "arith.andi",
    "or": "arith.ori",
    "xor": "arith.xori",
    "popcount": "allo.xnor_popcount",
}
_MLIR_FLOAT = {"mul": "arith.mulf", "add": "arith.addf"}


def _mlir_operation(kind: str, element: ElementType | None) -> str:
    if element is not None and element.is_float:
        return _MLIR_FLOAT.get(kind, f"arith.{kind}f")
    return _MLIR_INT.get(kind, f"arith.{kind}i")


def analyze_program(
    target,
    module,
    trace: MatchTrace,
    *,
    function: str | None = None,
    parameter_names=None,
) -> ProgramFacts:
    """Derive the shared semantic facts of one function from a match trace.

    ``target`` is the target (or pattern view) the trace was matched
    against; it supplies the operation kinds of each matched op.
    ``function`` selects the analyzed function (default: the only function
    with matches, else the first).  ``parameter_names`` are the Python
    argument names in signature order; when omitted the arguments are
    named ``arg<ordinal>``.  Matches in other functions are listed in
    ``foreign_functions``; the consumer decides whether that is supported.
    """

    matches = list(trace.matches)
    if function is None:
        names = list(dict.fromkeys(m.func_name for m in matches))
        if not names:
            body = getattr(module, "body", None)
            for operation in getattr(body, "operations", ()):
                if operation.operation.name == "func.func":
                    function = _attr_text(operation, "sym_name")
                    break
            if function is None:
                raise SemanticAnalysisError("the module has no function to analyze")
        else:
            function = names[0]
    facts = _ModuleFacts(module, function)

    parameters = []
    memref_ordinal = 0
    for ordinal, (text, signed) in enumerate(facts.argument_types):
        parsed = parse_memref_type(text)
        if parsed is None:
            continue
        shape, element_text = parsed
        ref = facts.argument_refs[memref_ordinal]
        memref_ordinal += 1
        if parameter_names is not None and ordinal < len(parameter_names):
            name = str(parameter_names[ordinal])
        else:
            name = f"arg{ordinal}"
        try:
            element = ElementType.parse(element_text, signed)
        except SemanticAnalysisError as error:
            raise SemanticAnalysisError(f"argument {name!r}: {error}") from error
        parameters.append(Parameter(name, ordinal, shape, element, ref))
    parameters_by_ref = {p.value_ref: p for p in parameters if p.value_ref is not None}

    contractions: list[Contraction] = []
    epilogues: list[Epilogue] = []
    others: list[MatchedOp] = []
    foreign = sorted({m.func_name for m in matches if m.func_name != function})
    for match in matches:
        if match.func_name != function:
            continue
        loops = _loops_of(match)
        loop_vars = {loop.var for loop in loops}
        accumulators = [op for op in match.operands if op.is_loop_carried]
        if len(accumulators) == 1:
            (acc_binding,) = accumulators
            roles = match.extra.get("axis_roles") or {}
            inputs = tuple(
                _access(op, facts, parameters_by_ref, loop_vars)
                for op in match.operands
                if not op.is_loop_carried and op.memref_name is not None
            )
            accumulator = _access(acc_binding, facts, parameters_by_ref, loop_vars)
            by_role = {access.role: access for access in inputs}
            product_kind, accumulate_kind = _pattern_kinds(
                target, match.target_op_name, acc_binding.role
            )
            contractions.append(
                Contraction(
                    match=match,
                    function=function,
                    target_op_name=match.target_op_name,
                    loops=loops,
                    reduction_vars=tuple(roles.get("reduction", ())),
                    output_vars=tuple(roles.get("output", ())),
                    batch_vars=tuple(roles.get("batch", ())),
                    roles_unresolved=(
                        roles.get("unresolved")
                        if roles
                        else "the match carries no axis roles"
                    ),
                    inputs=inputs,
                    matrix=by_role.get(roles.get("matrix")),
                    vector=by_role.get(roles.get("vector")),
                    accumulator=accumulator,
                    accumulator_init=match.extra.get("accumulator_init"),
                    sinks=_sinks(
                        match, accumulator, facts, parameters_by_ref, loop_vars
                    ),
                    store_coords=match.extra.get("store_coords"),
                    guards=tuple(match.extra.get("guards") or ()),
                    product_kind=product_kind,
                    accumulate_kind=accumulate_kind,
                    product_operation=_mlir_operation(
                        product_kind, accumulator.element
                    ),
                    accumulate_operation=_mlir_operation(
                        accumulate_kind, accumulator.element
                    ),
                    reduction_policy=match.extra.get("reduction_policy"),
                    same_buffer=bool(roles.get("same_buffer", False)),
                    vector_width=int(getattr(match, "vector_width", 1) or 1),
                )
            )
        elif not accumulators and match.result_value_ref is not None:
            operands = tuple(
                _access(op, facts, parameters_by_ref, loop_vars)
                for op in match.operands
                if op.memref_name is not None
            )
            result_parameter = parameters_by_ref.get(match.result_value_ref)
            result = BufferAccess(
                role="result",
                memref_name=match.result_memref_name,
                parameter=(None if result_parameter is None else result_parameter.name),
                local=result_parameter is None,
                value_ref=match.result_value_ref,
                shape=facts.shape(
                    match.extra.get("result_memref_type"), match.result_value_ref
                ),
                element=facts.element(
                    match.extra.get("result_memref_type"), match.result_value_ref
                ),
                coords=match.extra.get("store_coords"),
                dims=_plain_dims(match.extra.get("store_coords"), loop_vars),
                indices=tuple(match.extra.get("store_indices") or ()),
                index_map=None,
            )
            epilogues.append(
                Epilogue(
                    match=match,
                    function=function,
                    target_op_name=match.target_op_name,
                    loops=loops,
                    operands=operands,
                    constants=dict(match.extra.get("constants") or {}),
                    result=result,
                    store_coords=match.extra.get("store_coords"),
                    guards=tuple(match.extra.get("guards") or ()),
                )
            )
        else:
            others.append(match)

    return ProgramFacts(
        function=function,
        parameters=tuple(parameters),
        contractions=tuple(contractions),
        epilogues=tuple(epilogues),
        other_matches=tuple(others),
        foreign_functions=tuple(foreign),
        trace=trace,
    )


def product_extent(values) -> int:
    return math.prod(int(v) for v in values)


def stamp_program_facts(trace: MatchTrace, facts: ProgramFacts) -> None:
    """Attach the facts to the trace and to its matches so downstream stages
    (enumerators, backend contexts) read the semantic stage's result instead
    of re-deriving it: ``trace.semantics``, ``match.extra["program_facts"]``
    on every match of the analyzed function, and ``extra["semantic_facts"]``
    (the `Contraction` or `Epilogue`) on the matches that are one."""

    trace.semantics = facts
    for match in trace.matches:
        if match.func_name == facts.function:
            match.extra["program_facts"] = facts
    for item in (*facts.contractions, *facts.epilogues):
        item.match.extra["semantic_facts"] = item


__all__ = [
    "BufferAccess",
    "Contraction",
    "ElementType",
    "Epilogue",
    "LoopAxis",
    "OutputSink",
    "Parameter",
    "ProgramFacts",
    "SemanticAnalysisError",
    "analyze_program",
    "parse_memref_type",
    "stamp_program_facts",
]
