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
    # True for a buffer the function allocates and returns (a result), as
    # opposed to an argument the caller passes.
    returned: bool = False


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
        """Output and batch loops in loop-nest order; when the matcher left
        the roles unresolved, every loop that is not a reduction loop (each
        iteration of those writes one result instance)."""

        result = set(self.output_vars) | set(self.batch_vars)
        if not result:
            result = set(self.loop_vars) - set(self.reduction_vars)
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

    @property
    def trips(self) -> dict[str, int | None]:
        return {loop.var: loop.trip for loop in self.loops}

    @property
    def loop_labels(self) -> dict[str, str]:
        return {loop.var: loop.label for loop in self.loops}


@dataclass(frozen=True)
class StoreEffect:  # pylint: disable=too-many-instance-attributes
    """One externally visible store of the function, matched or not, with
    the facts the matcher retained from the IR: its destination, the loop
    nest and guards it executes under, the buffers its value and guards
    read, whether it reads the element it writes (a carried value), the
    conversions in its value.  The complete effect set of a function is the
    semantic contract of a program that executes the function verbatim
    (an ``APUv1Program`` phase)."""

    function: str
    store: str
    category: str
    destination: BufferAccess
    loops: tuple[LoopAxis, ...]
    guards: tuple[str, ...]
    reads: tuple[BufferAccess, ...]
    reads_destination: bool
    conversions: tuple[str, ...]
    term: str
    constant: bool
    # A whole-buffer write by construction (a ``linalg.fill`` initializer).
    whole: bool = False

    @property
    def loop_vars(self) -> tuple[str, ...]:
        return tuple(loop.var for loop in self.loops)

    @property
    def trips(self) -> dict[str, int | None]:
        return {loop.var: loop.trip for loop in self.loops}

    @property
    def loop_labels(self) -> dict[str, str]:
        return {loop.var: loop.label for loop in self.loops}

    @property
    def external(self) -> bool:
        """The destination is visible to the caller (an argument or a
        returned buffer)."""
        return not self.destination.local

    @property
    def dims(self) -> tuple[str | None, ...]:
        """The loop variable each destination coordinate is, or None."""
        return self.destination.dims

    @property
    def writes_whole_buffer(self) -> bool:
        """Every element of the destination is written unconditionally: no
        guard, every coordinate a distinct enclosing loop variable whose
        range is exactly the buffer's extent in that dimension."""
        shape = self.destination.shape
        dims = self.dims
        if self.whole:
            return not self.guards
        if self.guards or shape is None or len(dims) != len(shape):
            return False
        if any(d is None for d in dims) or len(set(dims)) != len(dims):
            return False
        by_var = {loop.var: loop for loop in self.loops}
        for var, extent in zip(dims, shape):
            loop = by_var.get(var)
            if (
                loop is None
                or loop.lower != 0
                or loop.upper != int(extent)
                or loop.step != 1
            ):
                return False
        return True

    @property
    def instances(self) -> int | None:
        """Result instances the store writes: the trips of the loops its
        coordinates range over (every enclosing loop when a coordinate is
        not a plain loop variable); every element for a whole-buffer fill."""
        if self.whole:
            shape = self.destination.shape
            return None if shape is None else _product_or_none(shape)
        dims = self.dims
        if dims and all(d is not None for d in dims):
            vars_ = set(dims)
        else:
            vars_ = set(self.loop_vars)
        return _product_or_none(loop.trip for loop in self.loops if loop.var in vars_)

    @property
    def reduction_extent(self) -> int | None:
        """Trips of the enclosing loops that do not move the destination
        element when the value is carried across them; None otherwise."""
        if not self.reads_destination:
            return None
        dims = self.dims
        if not dims or any(d is None for d in dims):
            return None
        vars_ = set(dims)
        return _product_or_none(
            loop.trip for loop in self.loops if loop.var not in vars_
        )


def _product_or_none(values) -> int | None:
    total = 1
    for value in values:
        if value is None:
            return None
        total *= int(value)
    return total


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
    # Every externally visible store of the function, in program order,
    # with its retained facts (`StoreEffect`); () for a trace without
    # coverage (a hand-assembled or synthetic trace).
    effects: tuple = ()
    # True when the program executes the whole function verbatim from its
    # retained IR (an ``APUv1Program`` phase): ``effects`` is then the
    # semantic contract the candidate and the realization are checked
    # against, matched or not.
    verbatim: bool = False

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
        # Buffers the function returns: allocated locally, visible to the
        # caller as results (an ``APUv1Program`` phase's ``result_names``).
        self.returned_refs: list = []
        for operation in _walk(self.function):
            if operation.operation.name == "func.return":
                for operand in operation.operands:
                    ref = self.refs.get(operand)
                    if ref is not None and str(operand.type).startswith("memref<"):
                        self.returned_refs.append(ref)
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
    result_names=None,
    verbatim: bool = False,
    parameter_names_by_ref=None,
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
        if parameter_names_by_ref is not None and ref in parameter_names_by_ref:
            # A dataflow work body: its arguments are named by the buffers
            # the matched operations bind them to (``local_W``).
            name = str(parameter_names_by_ref[ref])
        elif parameter_names is not None and ordinal < len(parameter_names):
            name = str(parameter_names[ordinal])
        else:
            name = f"arg{ordinal}"
        try:
            element = ElementType.parse(element_text, signed)
        except SemanticAnalysisError as error:
            raise SemanticAnalysisError(f"argument {name!r}: {error}") from error
        parameters.append(Parameter(name, ordinal, shape, element, ref))
    # Returned buffers are parameters too (named by ``result_names`` or
    # ``result<k>``), after the arguments.
    for position, ref in enumerate(facts.returned_refs):
        parsed = parse_memref_type(facts.type_by_ref.get(ref))
        if parsed is None:
            continue
        shape, element_text = parsed
        name = (
            str(result_names[position])
            if result_names is not None and position < len(result_names)
            else f"result{position}"
        )
        try:
            element = ElementType.parse(element_text, facts.signed_by_ref.get(ref))
        except SemanticAnalysisError as error:
            raise SemanticAnalysisError(f"result {name!r}: {error}") from error
        parameters.append(
            Parameter(
                name,
                len(facts.argument_types) + position,
                shape,
                element,
                ref,
                returned=True,
            )
        )
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
        effects=_store_effects(trace, function, facts, parameters_by_ref),
        verbatim=verbatim,
    )


def _retained_access(role, load, facts, parameters_by_ref, loop_vars) -> BufferAccess:
    parameter = parameters_by_ref.get(load.value_ref)
    return BufferAccess(
        role=role,
        memref_name=load.memref_name,
        parameter=None if parameter is None else parameter.name,
        local=parameter is None,
        value_ref=load.value_ref,
        shape=facts.shape(load.memref_type, load.value_ref),
        element=facts.element(load.memref_type, load.value_ref),
        coords=load.coords,
        dims=_plain_dims(load.coords, loop_vars),
        indices=tuple(load.indices),
        index_map=load.index_map,
    )


def _store_effects(trace, function, facts, parameters_by_ref) -> tuple:
    """`StoreEffect` of every store of ``function`` in the trace's coverage,
    in program order: stores to arguments and returned buffers are the
    function's external effects (``destination.local`` is False); stores to
    internal temporaries are kept too (``destination.local`` is True), since
    the buffers their values and guards read are reads of the function."""
    coverage = getattr(trace, "coverage", None)
    if coverage is None:
        return ()
    effects = []
    for store in coverage.stores:
        if store.func_name != function:
            continue
        names = dict(store.loop_names)
        loops = tuple(
            LoopAxis(
                var=str(var),
                name=names.get(str(var)),
                lower=_static_bound(lower),
                upper=_static_bound(upper),
                step=int(step) if str(step).lstrip("-").isdigit() else 0,
            )
            for var, lower, upper, step in store.enclosing_loops
        )
        loop_vars = {loop.var for loop in loops}
        parameter = parameters_by_ref.get(store.value_ref)
        destination = BufferAccess(
            role="store",
            memref_name=store.memref_name or facts.name_by_ref.get(store.value_ref),
            parameter=None if parameter is None else parameter.name,
            local=parameter is None,
            value_ref=store.value_ref,
            shape=facts.shape(store.memref_type, store.value_ref),
            element=facts.element(store.memref_type, store.value_ref),
            coords=store.coords,
            dims=_plain_dims(store.coords, loop_vars),
            indices=tuple(store.indices),
            index_map=None,
        )
        reads = (
            tuple(
                _retained_access("load", load, facts, parameters_by_ref, loop_vars)
                for load in store.loads
            )
            + tuple(
                _retained_access("guard", load, facts, parameters_by_ref, loop_vars)
                for load in store.guard_loads
            )
            + tuple(
                _retained_access("index", load, facts, parameters_by_ref, loop_vars)
                for load in store.index_loads
            )
        )
        effects.append(
            StoreEffect(
                function=function,
                store=store.store,
                category=store.category,
                destination=destination,
                loops=loops,
                guards=tuple(store.guards),
                reads=reads,
                reads_destination=bool(store.reads_destination),
                conversions=tuple(store.conversions),
                term=store.term,
                constant=bool(store.constant),
                whole=bool(store.whole),
            )
        )
    return tuple(effects)


def product_extent(values) -> int:
    return math.prod(int(v) for v in values)


def _effective_access_forms(fact):
    """Per input coordinate of a contraction, its effective form over the
    normalized loop counters (every loop variable substituted by
    ``lower + step * t``): ``(where, read, effective_terms, constant,
    reads_reduction)`` or a problem string for a loop without a static
    domain.  Coordinates with no retained form, with a non-loop atom or
    without loop terms are skipped."""
    loops = list(fact.loops)
    order = [loop.var for loop in loops]
    loop_vars = set(order)
    by_var = {loop.var: loop for loop in loops}
    reduction = set(fact.reduction_vars)
    labels = fact.loop_labels
    out = []
    for access in fact.inputs:
        coords = getattr(access, "coords", None) or ()
        for position, coord in enumerate(coords):
            if coord is None:
                continue
            terms, constant = coord
            if any(atom not in loop_vars for atom, _c in terms):
                continue
            loop_terms = sorted(
                ((atom, int(c)) for atom, c in terms if int(c) != 0),
                key=lambda item: order.index(item[0]),
            )
            if not loop_terms:
                continue
            where = (
                f"input {access.memref_name!r} (operand {access.role!r}) "
                f"coordinate {position}"
            )
            spelled = " + ".join(
                f"{c}*{labels.get(atom, atom)}" if c != 1 else labels.get(atom, atom)
                for atom, c in loop_terms
            )
            effective_constant = int(constant)
            effective = []
            problem = None
            for atom, c in loop_terms:
                loop = by_var[atom]
                if loop.lower is None or loop.upper is None or loop.step <= 0:
                    problem = (
                        f"{where} reads {spelled} over loop "
                        f"{labels.get(atom, atom)!r} without a static domain"
                    )
                    break
                effective.append((atom, c * loop.step, loop.trip or 0))
                effective_constant += c * loop.lower
            if problem is not None:
                out.append(problem)
                continue
            domains = ", ".join(
                f"{labels.get(atom, atom)} in [{by_var[atom].lower}, {by_var[atom].upper})"
                + (f" step {by_var[atom].step}" if by_var[atom].step != 1 else "")
                for atom, _c in loop_terms
                if by_var[atom].lower != 0 or by_var[atom].step != 1
            )
            read = spelled + (f" with {domains}" if domains else "")
            reads_reduction = any(atom in reduction for atom, _c in loop_terms)
            out.append(
                (
                    (access.role, access.memref_name, position),
                    where,
                    read,
                    tuple(effective),
                    effective_constant,
                    reads_reduction,
                )
            )
    return out


def _repeated_axis_problems(fact) -> list[str]:
    """An operand that reads one loop variable in two coordinates (``b[k, k]``,
    ``a[i, i, k]``) addresses a diagonal of its buffer: no PIM lowering
    stages such a gather (they stage each dimension along its own loops), so
    it is reported per operand, naming the coordinates (X-42, round 99)."""
    loop_vars = set(fact.loop_vars)
    labels = fact.loop_labels
    problems = []
    for access in fact.inputs:
        coords = getattr(access, "coords", None) or ()
        seen: dict = {}
        for position, coord in enumerate(coords):
            if coord is None:
                continue
            for atom, c in coord[0]:
                if atom in loop_vars and int(c) != 0:
                    seen.setdefault(atom, []).append(position)
        for atom, positions in seen.items():
            if len(positions) > 1:
                problems.append(
                    f"input {access.memref_name!r} (operand {access.role!r}) reads "
                    f"loop {labels.get(atom, atom)!r} in coordinates "
                    f"{positions}, a diagonal of the buffer; the PIM lowerings "
                    "stage each dimension of an operand along its own loops and "
                    "realize no gather"
                )
    return problems


def _bounds_problem(fact, key, effective, constant):
    """The staged extent of a dense coordinate must fit the buffer: with one
    coordinate per buffer dimension, ``offset + product of the trip counts
    <= shape[coordinate]``; with a single coordinate over a buffer of
    higher rank (a flat view of the whole buffer, the Gemini-II typed
    ``block_sum`` record's ``src[b*K + k]`` over its logical shape), against
    the buffer's element count.  Any other arity is not judged here."""
    role, memref, position = key
    access = next(
        (a for a in fact.inputs if a.role == role and a.memref_name == memref),
        None,
    )
    shape = getattr(access, "shape", None) if access is not None else None
    coords = getattr(access, "coords", None) if access is not None else None
    if not shape or any(dim is None for dim in shape) or coords is None:
        return None
    extent = 1
    for _atom, _c, trip in effective:
        extent *= int(trip)
    if len(coords) == len(shape):
        bound, what = int(shape[position]), f"a dimension of {int(shape[position])}"
    elif len(coords) == 1:
        bound = 1
        for dim in shape:
            bound *= int(dim)
        what = f"a flat view of {bound} elements ({'x'.join(str(d) for d in shape)})"
    else:
        return None
    if constant < 0 or constant + extent > bound:
        return (
            f"input {memref!r} (operand {role!r}) coordinate {position} stages "
            f"{extent} element(s) from offset {constant} of {what}; the access "
            "leaves the buffer"
        )
    return None


def _is_dense_flattening(effective) -> bool:
    """The innermost effective coefficient is one and every outer coefficient
    is the product of the trip counts of the terms inside it."""
    expected = 1
    for _atom, c, trip in reversed(effective):
        if c != expected:
            return False
        expected *= int(trip)
    return True


def contraction_access_problems(
    fact,
    *,
    reduction_offsets: bool = True,
    result_offsets: str = "any",
    realized=frozenset(),
) -> list[str]:
    """Input accesses of a contraction that no PIM lowering realizes as
    written (release ledger X-42), judged on the **effective** access: every
    loop variable is substituted by ``lower + step * t`` over its normalized
    counter ``t`` (`LoopAxis.lower`, `LoopAxis.step`), so a loop that starts
    at 128 or advances by two is judged like the coordinate ``k + 128`` or
    ``2*k`` it is.

    The lowerings stage a contraction operand contiguously along its loops
    from the operand's first element: the effective coordinate may be one
    normalized loop counter (``b[k]`` over ``range(K)``), a result-loop
    counter plus a constant where the backend realizes result offsets (the
    Samsung row offset ``row0 + i``) or where the work grid's items tile the
    result (`contraction_result_offsets`, checked per kernel group by
    `spmw_codegen.require_contraction_accesses_realizable`), or a dense
    row-major flattening of several counters (``xs[p * Q + k]`` with ``Q``
    the trip count of ``k``), which addresses the same contiguous elements a
    multi-dimensional access would.  Reported, one message per coordinate:
    an innermost effective coefficient other than one (``b[2*k]``,
    ``range(0, K, 2)``); an outer coefficient that is not the product of the
    inner trip counts (``xs[p * 2Q + k]``); unless ``reduction_offsets`` is
    allowed, an effective constant on a coordinate reading a reduction loop
    (``b[k + 1]``, ``range(128, K)``); and, under ``result_offsets="none"``,
    an effective constant on a result-loop coordinate (``a[16 + i]`` with
    ``c[16 + i]``, ``range(16, 32)``); per operand, a loop variable read in
    two coordinates (``b[k, k]``, a diagonal gather) and a dense coordinate
    whose staged extent leaves the buffer dimension.  A coordinate with no retained form
    (data dependent) or with an atom that is not a loop variable is not
    judged here; the backends' own checks keep handling those.  A loop
    without a static domain is reported.  ``realized`` names the operand
    coordinates ``(role, memref, position)`` whose access the program's own
    lowering realizes by construction (a program record's declaration,
    admitted by `spmw_codegen.require_contraction_accesses_realizable` for
    record matches only, G2-16); those coordinates are not judged."""
    if not hasattr(fact, "reduction_vars"):
        return []
    problems = list(_repeated_axis_problems(fact))
    for item in _effective_access_forms(fact):
        if isinstance(item, str):
            problems.append(item)
            continue
        key, where, read, effective, constant, reads_reduction = item
        if key in realized:
            continue
        if not _is_dense_flattening(effective):
            problems.append(
                f"{where} reads {read}, which is not one loop counter or a "
                "dense row-major flattening of its loops; the PIM lowerings "
                "stage a contraction operand contiguously, so a strided or "
                "padded access would be read as the dense one"
            )
            continue
        if constant != 0 and reads_reduction and not reduction_offsets:
            problems.append(
                f"{where} reads {read}, an effective shift of {constant} "
                "on a reduction loop; the PIM lowerings stage the reduction "
                "extent from the operand's first element"
            )
            continue
        if constant != 0 and not reads_reduction and result_offsets == "none":
            problems.append(
                f"{where} reads {read}, an effective offset of {constant} "
                "on a result loop; this backend stages a single work item's "
                "rows from the first row and realizes no result offset "
                "(partition the rows over a work grid, or index from zero)"
            )
            continue
        bounds = _bounds_problem(fact, key, effective, constant)
        if bounds is not None:
            problems.append(bounds)
    return problems


def contraction_result_offsets(fact) -> dict:
    """``{(operand role, memref, coordinate): (read, effective constant,
    extent)}`` for every dense input coordinate of the contraction that
    reads result loops only (keyed by the operand, so two operands reading
    one buffer, ``a[16 + i, k] * a[j, k]``, keep both accesses):
    the constant offset the coordinate adds over the loops' normalized
    counters and the extent those counters span (the product of their trip
    counts).  A work grid realizes such offsets only when its items tile the
    result: item ``w`` (in work-id order) must read offset ``w * extent``
    (`spmw_codegen.require_contraction_accesses_realizable`, X-42)."""
    out = {}
    if not hasattr(fact, "reduction_vars"):
        return out
    for item in _effective_access_forms(fact):
        if isinstance(item, str):
            continue
        key, _where, read, effective, constant, reads_reduction = item
        if reads_reduction or not _is_dense_flattening(effective):
            continue
        extent = 1
        for _atom, _c, trip in effective:
            extent *= int(trip)
        out[key] = (read, constant, extent)
    return out


def stamp_program_facts(
    trace: MatchTrace, facts: ProgramFacts, *, primary: bool = True
) -> None:
    """Attach the facts to the trace and to its matches so downstream stages
    (enumerators, backend contexts) read the semantic stage's result instead
    of re-deriving it: ``trace.semantics`` (the entry function's facts;
    ``primary=False`` for a dataflow work body's facts, which join
    ``trace.function_facts``), ``match.extra["program_facts"]`` on every
    match of the analyzed function, and ``extra["semantic_facts"]`` (the
    `Contraction` or `Epilogue`) on the matches that are one."""

    if primary:
        trace.semantics = facts
    function_facts = getattr(trace, "function_facts", None)
    if function_facts is None:
        function_facts = {}
        trace.function_facts = function_facts
    function_facts[facts.function] = facts
    for match in trace.matches:
        if match.func_name == facts.function:
            match.extra["program_facts"] = facts
    for item in (*facts.contractions, *facts.epilogues):
        item.match.extra["semantic_facts"] = item


__all__ = [
    "contraction_access_problems",
    "contraction_result_offsets",
    "BufferAccess",
    "Contraction",
    "ElementType",
    "Epilogue",
    "LoopAxis",
    "OutputSink",
    "Parameter",
    "ProgramFacts",
    "SemanticAnalysisError",
    "StoreEffect",
    "analyze_program",
    "parse_memref_type",
    "stamp_program_facts",
]
