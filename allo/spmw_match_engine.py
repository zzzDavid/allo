# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SPMW workload to target Op matcher (Step E of report 16).

Two public entry points:

  - ``compile_op_pattern(op)``: lift a target ``Op``'s ``fn`` lambda into a
    small expression tree (``OpPattern``) by parsing its Python AST. The
    pattern is also cached on the op as ``op.pattern``.

  - ``match_workload(target, mlir_module)``: walk the workload's MLIR,
    identify sites that implement one of the target's ops, and return a
    :class:`MatchTrace` (defined in :mod:`allo.spmw_match`).

The OpPattern tree uses these node kinds:

  - ``PVar(name)``           a free parameter (e.g. ``x``, ``y``, ``acc``)
  - ``PConst(value)``        a literal numeric constant
  - ``PBinOp(op, lhs, rhs)`` op in {"add","sub","mul","div","max","min"};
                             ``add``/``mul``/``max``/``min`` are commutative.
  - ``PUnary(fn, arg)``      fn in {"exp","log","sqrt","rsqrt","tanh",
                             "erf","abs"} (the ``allo.spmw_match`` calls),
                             or "neg" for Python unary minus.
  - ``PCmp(pred, lhs, rhs)`` pred in {"lt","le","gt","ge","eq","ne"}.
  - ``PSelect(cond, a, b)``  ``select(cond, a, b)``.

The matcher reflects each candidate workload site as a similar tree of
``WTerm`` nodes (``WLoad`` / ``WConst`` / ``WBinOp`` / ``WUnary`` /
``WCmp`` / ``WSelect``) by tracing the SSA
def-use chain backward from the value being stored. Unification is then
a structural walk that, for each commutative binop, tries both operand
orderings.
"""

from __future__ import annotations

import ast
import inspect
import dataclasses
import re
from dataclasses import dataclass, field
from typing import Any

from ._mlir.ir import AsmState
from .spmw_match import (
    STORE_ACCUMULATOR_INIT,
    STORE_COPY,
    STORE_FILL,
    STORE_LOCAL,
    STORE_MATCHED,
    STORE_UNMATCHED,
    IRValueRef,
    MatchCoverage,
    MatchedOp,
    MatchTrace,
    OperandBinding,
    RetainedLoad,
    StoreCoverage,
    UnsupportedWorkloadError,
)


# --------------------------------------------------------------------- #
# Pattern AST
# --------------------------------------------------------------------- #


@dataclass
class PVar:
    name: str


@dataclass
class PConst:
    value: Any


@dataclass
class PBinOp:
    op: str  # "add" | "sub" | "mul" | "div"
    lhs: Any
    rhs: Any


@dataclass
class PUnary:
    fn: str  # "exp" | "log" | "sqrt" | "rsqrt" | "tanh" | "erf" | "abs" | "neg"
    arg: Any


@dataclass
class PCmp:
    pred: str  # "lt" | "le" | "gt" | "ge" | "eq" | "ne"
    lhs: Any
    rhs: Any


@dataclass
class PSelect:
    cond: Any
    a: Any
    b: Any


@dataclass
class OpPattern:
    """Compiled pattern for one target Op.

    ``param_names`` is the lambda's argument list (in declaration order),
    ``body`` is the pattern tree.
    """

    param_names: list[str]
    body: Any  # PVar | PConst | PBinOp | PUnary | PCmp | PSelect


_AST_BINOP_TO_NAME = {
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mul",
    ast.Div: "div",
    ast.BitAnd: "and",
    ast.BitOr: "or",
    ast.BitXor: "xor",
    ast.LShift: "shl",
    ast.RShift: "shr",
}

# Callee name in a target lambda -> normalized unary function name.
_PATTERN_UNARY_CALLS = {
    "exp": "exp",
    "log": "log",
    "sqrt": "sqrt",
    "rsqrt": "rsqrt",
    "tanh": "tanh",
    "erf": "erf",
    "abs_": "abs",
    "popcount": "popcount",
}

_AST_CMPOP_TO_PRED = {
    ast.Lt: "lt",
    ast.LtE: "le",
    ast.Gt: "gt",
    ast.GtE: "ge",
    ast.Eq: "eq",
    ast.NotEq: "ne",
}


def _callee_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _compile_expr(node: ast.AST, params: set[str]) -> Any:
    if isinstance(node, ast.BinOp):
        kind = _AST_BINOP_TO_NAME.get(type(node.op))
        if kind is None:
            raise ValueError(f"unsupported binop in lambda: {ast.dump(node.op)}")
        return PBinOp(
            kind, _compile_expr(node.left, params), _compile_expr(node.right, params)
        )
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return PUnary("neg", _compile_expr(node.operand, params))
    callee = _callee_name(node) if isinstance(node, ast.Call) else None
    if callee in {"max", "min"}:
        # max(x, 0) / min(x, 0) -> binary PBinOp (allo lowers max -> arith.maximumf).
        if len(node.args) != 2:
            raise ValueError(f"{callee} in lambda body requires exactly 2 args")
        return PBinOp(
            callee,
            _compile_expr(node.args[0], params),
            _compile_expr(node.args[1], params),
        )
    if callee in _PATTERN_UNARY_CALLS:
        if len(node.args) != 1 or node.keywords:
            raise ValueError(f"{callee} in lambda body requires exactly 1 arg")
        return PUnary(_PATTERN_UNARY_CALLS[callee], _compile_expr(node.args[0], params))
    if callee == "select":
        if len(node.args) != 3 or node.keywords:
            raise ValueError("select in lambda body requires exactly 3 args")
        return PSelect(*(_compile_expr(arg, params) for arg in node.args))
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise ValueError("chained comparison in lambda body is unsupported")
        pred = _AST_CMPOP_TO_PRED.get(type(node.ops[0]))
        if pred is None:
            raise ValueError(
                f"unsupported comparison in lambda: {ast.dump(node.ops[0])}"
            )
        return PCmp(
            pred,
            _compile_expr(node.left, params),
            _compile_expr(node.comparators[0], params),
        )
    if isinstance(node, ast.Name):
        if node.id in params:
            return PVar(node.id)
        raise ValueError(f"unbound name in lambda body: {node.id!r}")
    if isinstance(node, ast.Constant):
        return PConst(node.value)
    if isinstance(node, ast.Num):  # pragma: no cover - py<3.8 compat
        return PConst(node.n)
    raise ValueError(f"unsupported AST node in lambda: {ast.dump(node)}")


def compile_op_pattern(op) -> OpPattern:
    """Parse ``op.fn``'s source and return its :class:`OpPattern`.

    Caches the result on ``op.pattern`` so subsequent calls are free.
    """
    if getattr(op, "pattern", None) is not None:
        return op.pattern
    fn = op.fn
    # Recover the lambda's source. ``inspect.getsource(lambda)`` returns the
    # single line the lambda was written on, which is unparseable when the
    # lambda is embedded mid-expression (e.g. ``foo(fn=lambda ...)`` where the
    # call spans multiple lines). Instead, parse the whole containing source
    # file and find the Lambda whose ``lineno`` matches ``co_firstlineno``.
    code = fn.__code__
    try:
        with open(code.co_filename, "r") as f:
            file_src = f.read()
        tree = ast.parse(file_src, filename=code.co_filename, mode="exec")
        lam = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Lambda) and node.lineno == code.co_firstlineno:
                lam = node
                break
    except (OSError, SyntaxError):
        lam = None
    if lam is None:
        # Fallback: try the single-line form for the simple case where the
        # lambda fits on its own line (e.g. ``mac = lambda x, y, acc: ...``).
        try:
            src = inspect.getsource(fn).strip()
            tree = ast.parse(src, mode="exec")
            for node in ast.walk(tree):
                if isinstance(node, ast.Lambda):
                    lam = node
                    break
        except (OSError, SyntaxError):
            pass
    if lam is None:
        raise ValueError(f"cannot recover lambda source for op {op.name!r}")

    params = [a.arg for a in lam.args.args]
    body = _compile_expr(lam.body, set(params))
    pat = OpPattern(param_names=params, body=body)
    op.pattern = pat
    return pat


def compile_target_patterns(target) -> None:
    """Eagerly compile every ``Op``'s pattern attached to ``target``.

    Idempotent; safe to call after :func:`target.target` decoration.
    """
    for u in target._walk():
        for op in u.ops.values():
            if not op.matchable:
                continue
            compile_op_pattern(op)


# --------------------------------------------------------------------- #
# Workload term tree
# --------------------------------------------------------------------- #


@dataclass
class WLoad:
    """A value produced by a (memref|affine).load. Leaf in the term tree."""

    memref_name: str | None
    ssa_name: str  # the load result's SSA name (e.g. "%21")
    indices: list[str]
    source_op_name: str  # MLIR op name, e.g. "memref.load"
    memref_type: str | None = None
    value_ref: IRValueRef | None = None
    # ``affine.load`` access-map text, e.g. ``affine_map<(d0) -> (d0 * 2)>``;
    # None for ``memref.load``.
    index_map: str | None = None
    # Memory state of the memref at this load (spec 004 M2): ``("nostore",)``
    # when the function never stores to it, else ``(block, stores before)``.
    # Two loads with equal versions see no intervening store. None outside a
    # function scope, which keeps SSA-name equality.
    version: tuple | None = None
    # Traced index operands (spec 004 M6), filled inside a function scope.
    index_terms: tuple | None = None
    # How the program reads the loaded integer: True after the frontend
    # widened it with ``arith.extsi``, False after ``arith.extui``, None when
    # the load is used at its own width (MLIR integers are signless; Allo's
    # ``uint`` types lower to the same ``iN`` as ``int``).  Only the range
    # analysis (`_interval`, `_float_leaf_fits`) reads it; None is treated
    # as "either", the union of both ranges.
    signed: bool | None = None


@dataclass
class WBlockArg:
    """An SSA value coming from a block argument we couldn't resolve to a load."""

    ssa_name: str


@dataclass
class WConst:
    value: Any
    ssa_name: str


@dataclass
class WBinOp:
    op: str  # "add" | "sub" | "mul" | "div"
    lhs: Any
    rhs: Any
    ssa_name: str
    # MLIR result type text (``i16``, ``f32``) when traced from the IR; None
    # for synthesized terms.  Lets the range analysis check that the wide
    # intermediate the frontend chose cannot wrap.
    result_type: str | None = None


@dataclass
class WUnary:
    fn: str  # normalized name, see _MATH_UNARY; "neg" for arith.negf
    arg: Any
    ssa_name: str
    # MLIR result type text when traced from the IR (the precision the
    # source evaluates the function in); None for synthesized terms.
    result_type: str | None = None


@dataclass
class WCmp:
    pred: str  # normalized: "lt" | "le" | "gt" | "ge" | "eq" | "ne"
    lhs: Any
    rhs: Any
    ssa_name: str
    raw: str  # original MLIR predicate, e.g. "ogt" / "slt", for emitters


@dataclass
class WSelect:
    cond: Any
    a: Any
    b: Any
    ssa_name: str


_FP_BINOPS = {
    "arith.mulf": "mul",
    "arith.addf": "add",
    "arith.subf": "sub",
    "arith.divf": "div",
    "arith.muli": "mul",
    "arith.addi": "add",
    "arith.subi": "sub",
    "arith.maximumf": "max",
    "arith.minimumf": "min",
    "arith.maxnumf": "max",
    "arith.minnumf": "min",
    "arith.maxsi": "max",
    "arith.maxui": "max",
    "arith.minsi": "min",
    "arith.minui": "min",
    "arith.andi": "and",
    "arith.ori": "or",
    "arith.xori": "xor",
    "arith.shli": "shl",
    "arith.shrsi": "shr",
    # No Python spelling, so no pattern ever matches it.
    "arith.shrui": "shru",
}

_MATH_UNARY = {
    "math.exp": "exp",
    "math.log": "log",
    "math.sqrt": "sqrt",
    "math.rsqrt": "rsqrt",
    "math.tanh": "tanh",
    "math.erf": "erf",
    "math.absf": "abs",
    # ``allo.popcount`` on a scalar lowers to ``math.ctpop``.
    "math.ctpop": "popcount",
}

# arith.CmpFPredicate / arith.CmpIPredicate enum order; the IR attribute is
# the integer case.
_CMPF_PREDICATES = (
    "false", "oeq", "ogt", "oge", "olt", "ole", "one", "ord",
    "ueq", "ugt", "uge", "ult", "ule", "une", "uno", "true",
)  # fmt: skip
_CMPI_PREDICATES = (
    "eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge",
)  # fmt: skip
_NORMALIZED_PREDICATES = {"lt", "le", "gt", "ge", "eq", "ne"}
_MIRRORED_PREDICATE = {
    "lt": "gt",
    "gt": "lt",
    "le": "ge",
    "ge": "le",
    "eq": "eq",
    "ne": "ne",
}

_CAST_OPS = {
    "arith.extsi",
    "arith.extui",
    "arith.trunci",
    "arith.extf",
    "arith.truncf",
    "arith.sitofp",
    "arith.uitofp",
    "arith.fptosi",
    "arith.fptoui",
    "arith.index_cast",
}

_LOAD_OPS = {"memref.load", "affine.load"}
_STORE_OPS = {"memref.store", "affine.store"}

# Significand bits (including the implicit one) of the MLIR float types: an
# integer of at most this many bits converts to the float exactly.
_FLOAT_SIGNIFICAND_BITS = {"f16": 11, "bf16": 8, "f32": 24, "f64": 53}
_INDEX_BITS = 64


def _scalar_type_text(mlir_type) -> str:
    text = str(mlir_type).strip()
    return text


def _int_bits(type_text: str) -> int | None:
    if type_text == "index":
        return _INDEX_BITS
    m = re.fullmatch(r"[su]?i(\d+)", type_text)
    return int(m.group(1)) if m else None


def cast_is_exact(op_name: str, source_type, result_type) -> bool:
    """True when the conversion preserves every source value exactly, so the
    matcher may look through it as if it were not there.

    Exact: widening of the same kind (``extf``: f16/bf16 -> f32 -> f64;
    ``extsi``/``extui``: a narrower integer to a wider one; ``index_cast``
    to ``index``), and an integer to float conversion whose integer width is
    at most the float's significand width (``sitofp``/``uitofp``: i8 -> f16,
    up to i24 -> f32, up to i53 -> f64).  Everything else (``trunci``,
    ``truncf``, ``fptosi``, ``fptoui``, ``index_cast`` to a narrower type, an
    int to float conversion that can round) changes values and is a
    computation of its own: the matcher keeps it as an opaque unary term, so
    it can only be lowered by a target op that declares it, and otherwise
    fails coverage.
    """
    src = _scalar_type_text(source_type)
    dst = _scalar_type_text(result_type)
    if op_name == "arith.extf":
        return (
            src in _FLOAT_SIGNIFICAND_BITS
            and dst in _FLOAT_SIGNIFICAND_BITS
            and _FLOAT_SIGNIFICAND_BITS[dst] >= _FLOAT_SIGNIFICAND_BITS[src]
            and dst != src
        ) or src == dst
    if op_name in ("arith.extsi", "arith.extui"):
        a, b = _int_bits(src), _int_bits(dst)
        return a is not None and b is not None and b >= a
    if op_name == "arith.index_cast":
        if dst == "index":
            # Address formation: the frontend computes an address in a wide
            # integer and casts it to `index`; an in-bounds address (the only
            # kind a well-defined program forms) is unchanged by the cast.
            return True
        a, b = _int_bits(src), _int_bits(dst)
        return a is not None and b is not None and b >= a
    if op_name in ("arith.sitofp", "arith.uitofp"):
        a = _int_bits(src)
        bits = _FLOAT_SIGNIFICAND_BITS.get(dst)
        if a is None or bits is None:
            return False
        # A signed source spends one bit on the sign; its magnitude is a-1 bits.
        magnitude = a - 1 if op_name == "arith.sitofp" else a
        return magnitude <= bits
    return False


def _cast_constant(value, op_name: str):
    """The destination-type value of a constant under ``op_name``."""
    if op_name in ("arith.sitofp", "arith.uitofp", "arith.extf", "arith.truncf"):
        return float(value)
    if op_name in ("arith.fptosi", "arith.fptoui"):
        return int(value)
    return int(value)


def _constant_survives_cast(value, op_name: str, result_type) -> bool:
    """True when casting the constant ``value`` with ``op_name`` into
    ``result_type`` is exact for that value (a proof for the constant, not
    for the operation)."""
    import math

    dst = _scalar_type_text(result_type)
    if isinstance(value, bool):
        return False
    if op_name in ("arith.sitofp", "arith.uitofp", "arith.extf", "arith.truncf"):
        bits = _FLOAT_SIGNIFICAND_BITS.get(dst)
        if bits is None:
            return False
        if op_name in ("arith.sitofp", "arith.uitofp"):
            if not isinstance(value, int):
                return False
            return abs(value) < (1 << bits)
        # float narrowing/widening of a constant: exact when the float
        # round-trips through the destination type
        import numpy as np

        if dst == "bf16":
            # bf16 is the upper half of an f32: the value is representable
            # when its f32 form is exact and has no low-half bits.
            as_f32 = np.float32(value)
            if not math.isfinite(float(as_f32)) or float(as_f32) != float(value):
                return False
            return int(as_f32.view(np.uint32)) & 0xFFFF == 0
        try:
            dtype = {"f16": np.float16, "f32": np.float32, "f64": np.float64}[dst]
        except KeyError:
            return False
        converted = dtype(value)
        return math.isfinite(float(converted)) and float(converted) == float(value)
    if op_name in ("arith.fptosi", "arith.fptoui"):
        if isinstance(value, float) and not value.is_integer():
            return False
        bits = _int_bits(dst)
        if bits is None:
            return False
        as_int = int(value)
        if op_name == "arith.fptoui":
            return 0 <= as_int < (1 << bits)
        return -(1 << (bits - 1)) <= as_int < (1 << (bits - 1))
    if op_name in ("arith.trunci", "arith.extsi", "arith.extui", "arith.index_cast"):
        bits = _int_bits(dst)
        if bits is None or not isinstance(value, int):
            return False
        return -(1 << (bits - 1)) <= value < (1 << bits)
    return False


def _cast_exact_op(src_op) -> bool:
    """`cast_is_exact` for an MLIR cast op with one operand and one result."""
    try:
        return cast_is_exact(
            src_op.operation.name, src_op.operands[0].type, src_op.results[0].type
        )
    except (AttributeError, IndexError):
        return False


_MEMREF_ELEM_RE = re.compile(r"memref<(?:[^x<>]*x)*([^x<>]+)>")


def _memref_element_type(memref_type: str | None) -> str | None:
    """Element type text of a memref type string (``memref<8x4xi15>`` -> ``i15``)."""
    if not memref_type:
        return None
    m = _MEMREF_ELEM_RE.match(memref_type.strip())
    return m.group(1).strip() if m else None


def _int_fits(value: int, bits: int) -> bool:
    return -(1 << (bits - 1)) <= value < (1 << (bits - 1))


# Binary ops that commute with truncation to N bits (ring homomorphism of
# Z -> Z/2^N, and bitwise ops act lane-wise): trunc(a op b) == trunc(a) op
# trunc(b) in N-bit two's complement.  Everything else (division, right
# shifts, min/max, comparisons) does not distribute, so the truncation is
# exact only when the whole value provably fits (`_fits_integer`).
_TRUNC_DISTRIBUTES = {"add", "sub", "mul", "and", "or", "xor"}


def _signed_range(bits: int) -> tuple[int, int]:
    return (-(1 << (bits - 1)), (1 << bits - 1) - 1)


def _load_range(bits: int, signed: bool | None) -> tuple[int, int]:
    """Every value a ``bits``-bit signless load may denote: the signed range
    when the program sign-extends it, the unsigned range when it zero-extends
    it, and the union of both when the width is used as is."""
    if signed is True:
        return _signed_range(bits)
    if signed is False:
        return (0, (1 << bits) - 1)
    return (-(1 << (bits - 1)), (1 << bits) - 1)


def _interval(term, loop_bounds) -> tuple[int, int] | None:
    """Sound bounds on the integer value of ``term`` as the source computes
    it, or None when no bound is known.

    Leaves: a constant, a load of an ``iK`` element (any ``K``-bit value), a
    loop index with constant bounds.  Interior ops use interval arithmetic;
    a result that would not fit the op's own IR type is discarded (the wide
    op could wrap, so the arithmetic bound would be unsound).
    """
    if isinstance(term, WConst):
        value = term.value
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        return (value, value)
    if isinstance(term, WLoad):
        width = _int_bits(_memref_element_type(term.memref_type) or "")
        return None if width is None else _load_range(width, term.signed)
    if isinstance(term, WBlockArg):
        bounds = loop_bounds.get(term.ssa_name)
        return (
            None
            if bounds is None or bounds[1] <= bounds[0]
            else (bounds[0], bounds[1] - 1)
        )
    if isinstance(term, WSelect):
        a = _interval(term.a, loop_bounds)
        b = _interval(term.b, loop_bounds)
        if a is None or b is None:
            return None
        return (min(a[0], b[0]), max(a[1], b[1]))
    if isinstance(term, WUnary):
        if term.fn != "neg":
            return None
        inner = _interval(term.arg, loop_bounds)
        return None if inner is None else (-inner[1], -inner[0])
    if not isinstance(term, WBinOp):
        return None
    lhs = _interval(term.lhs, loop_bounds)
    rhs = _interval(term.rhs, loop_bounds)
    if lhs is None or rhs is None:
        return None
    op = term.op
    if op == "add":
        result = (lhs[0] + rhs[0], lhs[1] + rhs[1])
    elif op == "sub":
        result = (lhs[0] - rhs[1], lhs[1] - rhs[0])
    elif op == "mul":
        products = [a * b for a in lhs for b in rhs]
        result = (min(products), max(products))
    elif op in ("min", "max"):
        pick = min if op == "min" else max
        result = (pick(lhs[0], rhs[0]), pick(lhs[1], rhs[1]))
    elif op in ("and", "or", "xor"):
        if lhs[0] >= 0 and rhs[0] >= 0:
            result = (0, (1 << max(lhs[1], rhs[1]).bit_length()) - 1)
        else:
            width = max(_bits_for(lhs), _bits_for(rhs))
            result = _signed_range(width)
    elif op in ("shr", "shl", "shru"):
        if rhs[0] != rhs[1] or rhs[0] < 0:
            return None
        shift = rhs[0]
        if op == "shl":
            result = (lhs[0] << shift, lhs[1] << shift)
        elif op == "shr":
            result = (lhs[0] >> shift, lhs[1] >> shift)
        elif lhs[0] >= 0:
            result = (lhs[0] >> shift, lhs[1] >> shift)
        else:
            return None
    elif op == "div":
        if rhs[0] != rhs[1] or rhs[0] == 0:
            return None
        divisor = rhs[0]
        quotients = [int(a / divisor) for a in lhs]
        result = (min(quotients), max(quotients))
    else:
        return None
    width = _int_bits(term.result_type or "")
    if width is not None and not (
        _int_fits(result[0], width) and _int_fits(result[1], width)
    ):
        # The IR evaluates the op in `width` bits; a value outside that range
        # would wrap there, so the arithmetic bound is not what the IR holds.
        return None
    return result


def _bits_for(interval: tuple[int, int]) -> int:
    """Smallest signed width holding both ends of ``interval``."""
    bits = 1
    while not (_int_fits(interval[0], bits) and _int_fits(interval[1], bits)):
        bits += 1
    return bits


def _fits_integer(term, bits: int, loop_bounds) -> bool:
    """True when ``term``'s value provably lies in the signed ``bits``-bit
    range, so a truncation to ``bits`` bits leaves it unchanged."""
    interval = _interval(term, loop_bounds)
    return (
        interval is not None
        and _int_fits(interval[0], bits)
        and _int_fits(interval[1], bits)
    )


def _narrow_integer(term, bits: int, loop_bounds):
    """The term ``term`` read in ``bits``-bit two's-complement arithmetic, or
    None when that is not provably the value ``trunci`` produces.

    The frontend widens integer arithmetic (``i4 * i4`` in ``i8``, the sum
    in ``i16``) and truncates back to the variable's width at the store.
    Truncation distributes over the ring ops (`_TRUNC_DISTRIBUTES`), so
    the stored value equals the same expression evaluated in ``bits``-bit
    wraparound arithmetic, provided every leaf is a ``bits``-bit value
    (a load of at most ``bits`` bits whatever its signedness, a constant or
    loop index with a ``bits``-bit two's-complement pattern).
    That evaluation is exactly what a target op declared on the
    destination's element type computes, so the returned tree is the one
    the patterns match against.  Any other shape (a wider leaf, a value
    that may not fit, a cast we cannot prove) returns None and the caller
    keeps the cast visible so matching fails closed.
    """
    if _fits_integer(term, bits, loop_bounds):
        # The value fits: evaluating in ``bits`` bits gives the same value.
        if isinstance(term, (WBinOp, WUnary)):
            return dataclasses.replace(term, result_type=f"i{bits}")
        return term
    # Leaves are judged by bit width, not by the signed range: truncating a
    # ``bits``-bit (or narrower) load to ``bits`` bits is the identity on its
    # bit pattern whatever its signedness, and a constant or loop index with
    # some ``bits``-bit two's-complement pattern denotes that pattern.  The
    # target op's declared element type decides how the device reads the
    # pattern; backends check that against the source (UPMEM's ABI check).
    if isinstance(term, WLoad):
        width = _int_bits(_memref_element_type(term.memref_type) or "")
        return term if width is not None and width <= bits else None
    if isinstance(term, WConst):
        value = term.value
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        return term if -(1 << (bits - 1)) <= value < (1 << bits) else None
    if isinstance(term, WBlockArg):
        bounds = loop_bounds.get(term.ssa_name)
        if bounds is None or bounds[1] <= bounds[0]:
            return None
        lo, hi = bounds[0], bounds[1] - 1
        return term if -(1 << (bits - 1)) <= lo and hi < (1 << bits) else None
    if isinstance(term, WBinOp) and term.op in _TRUNC_DISTRIBUTES:
        lhs = _narrow_integer(term.lhs, bits, loop_bounds)
        rhs = _narrow_integer(term.rhs, bits, loop_bounds)
        if lhs is None or rhs is None:
            return None
        return WBinOp(term.op, lhs, rhs, term.ssa_name, result_type=f"i{bits}")
    if isinstance(term, WBinOp) and term.op == "shl":
        # Low bits of a left shift depend only on the low bits of the value.
        lhs = _narrow_integer(term.lhs, bits, loop_bounds)
        if lhs is None or not _fits_integer(term.rhs, bits, loop_bounds):
            return None
        return WBinOp(term.op, lhs, term.rhs, term.ssa_name, result_type=f"i{bits}")
    if isinstance(term, WSelect):
        a = _narrow_integer(term.a, bits, loop_bounds)
        b = _narrow_integer(term.b, bits, loop_bounds)
        if a is None or b is None:
            return None
        return WSelect(term.cond, a, b, term.ssa_name)
    if isinstance(term, WUnary) and term.fn == "neg":
        arg = _narrow_integer(term.arg, bits, loop_bounds)
        return (
            None
            if arg is None
            else WUnary("neg", arg, term.ssa_name, result_type=f"i{bits}")
        )
    # Any other operation the IR itself evaluates in at most ``bits`` bits
    # is a ``bits``-bit value whatever its range: its result type bounds it,
    # and the exact widening that feeds it to the wider parent is undone by
    # the truncation.  The structural rules above come first so a product
    # the interval analysis can bound keeps its proven range for the
    # diagnostics; this covers the rest, e.g. a right shift of an int64
    # temporary under an i65 sum when the shifted product may have wrapped
    # in its own width (the UPMEM linear-regression gradient, C-31).
    if isinstance(term, (WBinOp, WUnary)):
        width = _int_bits(term.result_type or "")
        if width is not None and width <= bits:
            return term
    return None


def _float_leaf_fits(term, dst: str, loop_bounds) -> bool:
    """True when every leaf of the float expression ``term`` is a value of
    the destination float type ``dst`` (or exactly representable in it)."""
    bits = _FLOAT_SIGNIFICAND_BITS[dst]
    if isinstance(term, WConst):
        value = term.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        return _constant_survives_cast(float(value), "arith.truncf", dst)
    if isinstance(term, WLoad):
        elem = _memref_element_type(term.memref_type) or ""
        if elem in _FLOAT_SIGNIFICAND_BITS:
            return _FLOAT_SIGNIFICAND_BITS[elem] <= bits
        width = _int_bits(elem)
        if width is None:
            return False
        magnitude = width - 1 if term.signed is True else width
        return magnitude <= bits
    if isinstance(term, WBlockArg):
        bounds = loop_bounds.get(term.ssa_name)
        return bounds is not None and max(abs(bounds[0]), abs(bounds[1] - 1)) < (
            1 << bits
        )
    if isinstance(term, WBinOp):
        return _float_leaf_fits(term.lhs, dst, loop_bounds) and _float_leaf_fits(
            term.rhs, dst, loop_bounds
        )
    if isinstance(term, WUnary):
        # math.* functions and negation of a fitting argument; a cast term
        # (``arith.*<..>``) is a conversion we could not prove and stays out.
        return not term.fn.startswith("arith.") and _float_leaf_fits(
            term.arg, dst, loop_bounds
        )
    if isinstance(term, WSelect):
        return (
            _float_leaf_fits(term.cond, dst, loop_bounds)
            and _float_leaf_fits(term.a, dst, loop_bounds)
            and _float_leaf_fits(term.b, dst, loop_bounds)
        )
    if isinstance(term, WCmp):
        return _float_leaf_fits(term.lhs, dst, loop_bounds) and _float_leaf_fits(
            term.rhs, dst, loop_bounds
        )
    return False


def _is_float_leaf(term) -> bool:
    return isinstance(term, (WLoad, WConst, WBlockArg))


def _single_float_operation(term) -> bool:
    """True when ``term`` is one arithmetic operation on leaves (or a leaf):
    a leaf, ``-leaf``, ``leaf (+|-|*|/) leaf`` (one rounding, see
    `_narrow_float`) or ``min``/``max`` of leaves (no rounding at all: the
    result is one of the operands)."""
    if _is_float_leaf(term):
        return True
    if isinstance(term, WUnary):
        return term.fn == "neg" and _is_float_leaf(term.arg)
    if isinstance(term, WBinOp):
        return (
            term.op in ("add", "sub", "mul", "div", "min", "max")
            and _is_float_leaf(term.lhs)
            and _is_float_leaf(term.rhs)
        )
    return False


def _truncf_types(fn: str) -> tuple[str, str] | None:
    """``("f32", "bf16")`` for the unary term name ``arith.truncf<f32->bf16>``."""
    m = re.fullmatch(r"arith\.truncf<([^-]+)->([^>]+)>", fn)
    return (m.group(1), m.group(2)) if m else None


def _narrow_float(term, src: str, dst: str, state):
    """Accept ``truncf`` from ``src`` to ``dst`` at a store as exactly the
    ``dst``-precision operation on ``dst`` operands, or return None.

    The frontend evaluates ``float(a[i]) * float(b[i])`` (and arithmetic on
    ``bf16`` values mixed with ``f32`` literals) in ``src`` and rounds once at
    the store.  A device op declared on ``dst`` rounds after every operation.
    The two agree, bit for bit, only for a *single* operation on ``dst``
    values: rounding the exact result first to ``src`` and then to ``dst``
    equals rounding it once to ``dst`` whenever ``src`` carries at least
    ``2 p + 2`` significand bits for a ``p``-bit ``dst`` (Figueroa, "When is
    double rounding innocuous?", 1995; f16 and bf16 in f32, f32 in f64 all
    satisfy it; the device's single rounding is the correctly rounded
    operation).  A compound expression keeps ``src``-precision intermediates
    in the source (``256 * 256 + (-65504)`` is 32 in f32, overflow in f16),
    so the cast stays visible: only an op that declares ``intermediate``
    precision at least ``src`` may bind it (see ``_try_match_at_store``).
    Leaves must be ``dst`` values (loads of ``dst``, constants ``dst``
    represents exactly); an ``f32`` input means the source really computes
    in ``f32``.
    """
    if dst not in _FLOAT_SIGNIFICAND_BITS or src not in _FLOAT_SIGNIFICAND_BITS:
        return None
    if _FLOAT_SIGNIFICAND_BITS[src] < 2 * _FLOAT_SIGNIFICAND_BITS[dst] + 2:
        return None
    loop_bounds = getattr(state, "loop_bounds", {})
    if not _float_leaf_fits(term, dst, loop_bounds):
        return None
    if not _single_float_operation(term):
        return None
    # The accepted operation *is* the dst-precision operation: retag it so
    # the precision contract (`arithmetic_precision_problem`) sees dst.
    if isinstance(term, (WBinOp, WUnary)):
        return dataclasses.replace(term, result_type=dst)
    return term


def _float_leaf_types(term, out: list) -> list:
    """Element types of the float-typed loads under ``term``."""
    for load in _term_loads(term, []):
        elem = _memref_element_type(load.memref_type) or ""
        if elem in _FLOAT_SIGNIFICAND_BITS:
            out.append(elem)
    return out


def _int_leaf_widths(term, out: list) -> list:
    for load in _term_loads(term, []):
        width = _int_bits(_memref_element_type(load.memref_type) or "")
        if width is not None:
            out.append(width)
    return out


def arithmetic_precision_problem(term) -> str | None:
    """Why a target op declared on its operands' element type cannot compute
    ``term``: the source evaluates the root operation in a wider type than
    its operands (exact widening casts are transparent to the matcher, so
    ``float(a[i]) + float(b[i])`` on f16 arrays stored to f32 is ``add`` in
    f32 over f16 leaves; a device f16 add overflows where the source does
    not).  None when the root's result type is at most the operands' type,
    or when the store narrowed it back and the narrowing proof retagged the
    operation (`_narrow_float`, `_narrow_integer`).  A compound expression
    whose root is wide is caught here as well; its inner operations are
    covered by the store-side narrowing rules.
    """
    if not isinstance(term, (WBinOp, WUnary)):
        return None
    result = term.result_type
    if result is None:
        return None
    if result in _FLOAT_SIGNIFICAND_BITS:
        leaves = _float_leaf_types(term, [])
        if not leaves:
            return None
        narrowest = min(leaves, key=lambda t: _FLOAT_SIGNIFICAND_BITS[t])
        if _FLOAT_SIGNIFICAND_BITS[result] > _FLOAT_SIGNIFICAND_BITS[narrowest]:
            return (
                f"the source evaluates this operation in {result} on {narrowest} "
                f"operands (a widened intermediate stored without narrowing back); a "
                f"target op computes in its operands' type {narrowest}, which rounds "
                "and overflows differently"
            )
        return None
    width = _int_bits(result)
    if width is None:
        return None
    leaves = _int_leaf_widths(term, [])
    if leaves and width > max(leaves):
        return (
            f"the source evaluates this operation in {result} on operands of at most "
            f"i{max(leaves)} (a widened intermediate stored without truncating back); "
            f"a target op computes in its operands' width, which wraps where the "
            "source does not"
        )
    return None


def _op_computes_wide(op_obj, term) -> bool:
    """True when ``op_obj`` declares an intermediate precision that covers
    the type ``term`` is evaluated in: a float type of at least as many
    significand bits, or an integer width of at least as many bits."""
    declared = getattr(op_obj, "intermediate", None)
    result = getattr(term, "result_type", None)
    if declared is None or result is None:
        return False
    if result in _FLOAT_SIGNIFICAND_BITS:
        return (
            _FLOAT_SIGNIFICAND_BITS.get(declared, 0) >= _FLOAT_SIGNIFICAND_BITS[result]
        )
    width, declared_width = _int_bits(result), _int_bits(str(declared))
    return width is not None and declared_width is not None and declared_width >= width


def _attr_str(a) -> str | None:
    """Best-effort: strip surrounding quotes from a StringAttr-ish value."""
    if a is None:
        return None
    s = str(a)
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    return s


class ValueNames:
    """Function-unique SSA names.

    The printer restarts numbering in each region, so ``Value.get_name`` can
    give two values of one function the same name (``%0`` in two sibling
    loops). Name-keyed lookups (``defining_map``) would then resolve to the
    wrong op. The first value keeps its printed name; later ones with the same
    name get a ``#k`` suffix, assigned in program order.
    """

    def __init__(self, func, state):
        self.state = state
        self._names: dict = {}
        # affine.for induction variable name -> (lower, upper) when both
        # bounds are constants (the iv ranges over [lower, upper)); used to
        # prove that a narrowing cast of a loop index is exact.
        self.loop_bounds: dict[str, tuple[int, int]] = {}
        # affine.for induction variable name -> the frontend's ``loop_name``
        # attribute (the source loop label), when retained.  Diagnostics and
        # the shared semantic facts use it; matching never does.
        self.loop_names: dict[str, str] = {}
        taken: set = set()

        def assign(value):
            base = value.get_name(state)
            name, k = base, 1
            while name in taken:
                name = f"{base}#{k}"
                k += 1
            taken.add(name)
            self._names[value] = name

        def visit(block):
            for argument in block.arguments:
                assign(argument)
            for op in block.operations:
                for result in op.results:
                    assign(result)
                for region in op.regions:
                    for nested in region.blocks:
                        visit(nested)
                if op.operation.name == "affine.for":
                    self._record_loop_bounds(op)

        for argument in func.arguments:
            if argument not in self._names:
                assign(argument)
        visit(func.regions[0].blocks[0])

    def _record_loop_bounds(self, op) -> None:
        attrs = op.attributes
        try:
            iv = self._names[op.regions[0].blocks[0].arguments[0]]
        except (KeyError, IndexError):
            return
        if "loop_name" in attrs:
            self.loop_names[iv] = _attr_str(attrs["loop_name"])
        try:
            lo = _AFFINE_CONST_RE.match(str(attrs["lowerBoundMap"]).strip())
            hi = _AFFINE_CONST_RE.match(str(attrs["upperBoundMap"]).strip())
        except KeyError:
            return
        if lo and hi:
            self.loop_bounds[iv] = (int(lo.group(1)), int(hi.group(1)))

    def __call__(self, value) -> str:
        name = self._names.get(value)
        return name if name is not None else value.get_name(self.state)


def _value_name(value, state) -> str:
    return state(value) if isinstance(state, ValueNames) else value.get_name(state)


def _operand_names(op, *, state) -> list[str]:
    return [_value_name(o, state) for o in op.operands]


def _result_names(op, *, state) -> list[str]:
    return [_value_name(r, state) for r in op.results]


@dataclass
class _FunctionScope:
    """Per-function facts for the spec 004 matcher extensions.

    ``versions``: load result name -> memory-state version (M2).
    ``forward``: load result name -> stored SSA name of the single dominating
    store into a scalar ``memref.alloc`` (M3).
    """

    versions: dict
    forward: dict


def _scalar_alloc(value) -> bool:
    owner = getattr(value, "owner", None)
    if getattr(getattr(owner, "operation", owner), "name", None) != "memref.alloc":
        return False
    shape = getattr(value.type, "shape", None)
    try:
        shape = list(shape)
    except TypeError:
        return False
    return len(shape) == 0 or (len(shape) == 1 and shape[0] == 1)


def _build_function_scope(func, value_refs, *, state) -> _FunctionScope:
    """Walk ``func`` once, in program order, recording op paths."""
    loads, stores = [], []

    def visit(block, prefix):
        for index, op in enumerate(block.operations):
            path = prefix + (index,)
            name = op.operation.name
            if name in _LOAD_OPS and op.operands and op.results:
                loads.append((op, path))
            elif name in _STORE_OPS and len(op.operands) > 1:
                stores.append((op, path))
            for region_index, region in enumerate(op.regions):
                for block_index, nested in enumerate(region.blocks):
                    visit(nested, path + (region_index, block_index))

    visit(func.regions[0].blocks[0], ())

    def key(memref):
        return value_refs.get(memref, memref)

    stored: dict = {}
    for op, path in stores:
        stored.setdefault(key(op.operands[1]), []).append((op, path))

    versions, forward = {}, {}
    for op, path in loads:
        memref = op.operands[0]
        result = _value_name(op.results[0], state)
        writes = stored.get(key(memref), [])
        if not writes:
            versions[result] = ("nostore",)
        else:
            block = path[:-1]
            before = sum(
                1
                for _w, wpath in writes
                if wpath[:-1] == block and wpath[-1] < path[-1]
            )
            versions[result] = (block, before)
        # M3: one store into a scalar alloc, in this block or an enclosing one,
        # before the load in program order.
        if len(writes) == 1 and len(op.operands) - 1 <= 1 and _scalar_alloc(memref):
            store, spath = writes[0]
            depth = len(spath) - 1
            if (
                len(path) > depth
                and path[:depth] == spath[:depth]
                and spath[depth] < path[depth]
            ):
                forward[result] = _value_name(store.operands[0], state)
    return _FunctionScope(versions, forward)


def _build_load_term(
    load_op, value_refs, *, state, scope=None, defining_map=None
) -> WLoad:
    name = load_op.operation.name
    attrs = load_op.attributes
    memref_name = None
    if "from" in attrs:
        memref_name = _attr_str(attrs["from"])
    operands = _operand_names(load_op, state=state)
    # operands[0] is the memref; the rest are the indices.
    indices = operands[1:] if len(operands) > 1 else []
    ssa = _result_names(load_op, state=state)[0] if load_op.results else "<no-result>"
    memref_type = str(load_op.operands[0].type) if load_op.operands else None
    index_map = str(attrs["map"]) if "map" in attrs else None
    version = None
    index_terms = None
    if scope is not None:
        version = scope.versions.get(ssa)
        index_terms = tuple(
            _trace_value(index, defining_map, value_refs, state=state, scope=scope)
            for index in indices
        )
    return WLoad(
        memref_name=memref_name,
        ssa_name=ssa,
        indices=indices,
        source_op_name=name,
        memref_type=memref_type,
        value_ref=value_refs.get(load_op.operands[0]) if load_op.operands else None,
        index_map=index_map,
        version=version,
        index_terms=index_terms,
    )


def _compare_predicate(cmp_op) -> tuple[str, str] | None:
    """Return ``(normalized, raw)`` for an ``arith.cmpf``/``arith.cmpi``.

    None for predicates with no single-comparison form (``ord``, ``uno``,
    ``true``, ``false``) or an unreadable attribute.
    """
    table = (
        _CMPF_PREDICATES if cmp_op.operation.name == "arith.cmpf" else _CMPI_PREDICATES
    )
    try:
        case = int(str(cmp_op.attributes["predicate"]).split(":")[0].strip())
        raw = table[case]
    except (KeyError, ValueError, IndexError):
        return None
    if raw in _NORMALIZED_PREDICATES:
        return raw, raw
    normalized = raw[1:]
    if normalized not in _NORMALIZED_PREDICATES:
        return None
    return normalized, raw


def _trace_value(
    ssa_name: str, defining_map: dict[str, Any], value_refs, *, state, scope=None
) -> Any:
    """Build a WTerm for the SSA value named ``ssa_name``.

    ``defining_map`` maps SSA result names to the MLIR op that defines them.
    Block arguments (e.g. ``%arg3``) won't be in the map and become
    :class:`WBlockArg`.
    """
    src = defining_map.get(ssa_name)
    if src is None:
        return WBlockArg(ssa_name=ssa_name)

    op_name = src.operation.name
    if op_name in _LOAD_OPS:
        if scope is not None and ssa_name in scope.forward:
            return _trace_value(
                scope.forward[ssa_name],
                defining_map,
                value_refs,
                state=state,
                scope=scope,
            )
        return _build_load_term(
            src, value_refs, state=state, scope=scope, defining_map=defining_map
        )
    if op_name in _FP_BINOPS:
        kind = _FP_BINOPS[op_name]
        ops = _operand_names(src, state=state)
        lhs = _trace_value(ops[0], defining_map, value_refs, state=state, scope=scope)
        rhs = _trace_value(ops[1], defining_map, value_refs, state=state, scope=scope)
        return WBinOp(
            kind,
            lhs,
            rhs,
            _result_names(src, state=state)[0],
            result_type=_scalar_type_text(src.results[0].type) if src.results else None,
        )
    if op_name in _MATH_UNARY:
        arg = _trace_value(
            _operand_names(src, state=state)[0],
            defining_map,
            value_refs,
            state=state,
            scope=scope,
        )
        return WUnary(
            _MATH_UNARY[op_name],
            arg,
            _result_names(src, state=state)[0],
            result_type=_scalar_type_text(src.results[0].type) if src.results else None,
        )
    if op_name in {"arith.cmpf", "arith.cmpi"}:
        predicate = _compare_predicate(src)
        if predicate is not None:
            ops = _operand_names(src, state=state)
            lhs = _trace_value(
                ops[0], defining_map, value_refs, state=state, scope=scope
            )
            rhs = _trace_value(
                ops[1], defining_map, value_refs, state=state, scope=scope
            )
            return WCmp(
                predicate[0], lhs, rhs, _result_names(src, state=state)[0], predicate[1]
            )
        return WBlockArg(ssa_name=_result_names(src, state=state)[0])
    if op_name == "arith.select":
        ops = _operand_names(src, state=state)
        cond, a, b = (
            _trace_value(name, defining_map, value_refs, state=state, scope=scope)
            for name in ops
        )
        return WSelect(cond, a, b, _result_names(src, state=state)[0])
    if op_name == "arith.constant":
        # Try to extract the literal; not strictly needed for matching.
        try:
            val = src.attributes["value"]
            sval = str(val).split(":")[0].strip()
            try:
                v = int(sval)
            except ValueError:
                try:
                    v = float(sval)
                except ValueError:
                    v = sval
            return WConst(v, _result_names(src, state=state)[0])
        except KeyError:  # MLIR attribute map: no "value" attribute
            return WConst(None, _result_names(src, state=state)[0])
    # Any other math.* op computes a value; tracing through it as a cast
    # would let e.g. ``a + exp(b)`` match ``x + y`` with ``y = b``.
    if op_name.startswith("math."):
        return WBlockArg(
            ssa_name=_result_names(src, state=state)[0] if src.results else ssa_name
        )
    # `(-1.0) * x` lowers to `negf 1.0`: fold a negated constant, and keep
    # any other negation as a real term so `-x` never traces as `x`.
    if op_name == "arith.negf" and len(src.operands) == 1 and src.results:
        inner = _trace_value(
            _operand_names(src, state=state)[0],
            defining_map,
            value_refs,
            state=state,
            scope=scope,
        )
        value = _const_value(inner)
        result = _result_names(src, state=state)[0]
        if value is not None:
            return WConst(-value, result)
        return WUnary(
            "neg",
            inner,
            result,
            result_type=_scalar_type_text(src.results[0].type) if src.results else None,
        )
    # Only value-preserving casts are traced through (`cast_is_exact`), so
    # that constants on the other side still appear as constants when the
    # pattern needs them.  A conversion that can change a value (narrowing,
    # float to int, int to float beyond the significand) is a computation:
    # it stays visible as a unary term named after the op and the types, so
    # no pattern silently absorbs it and coverage reports it.
    if op_name in _CAST_OPS and len(src.operands) == 1 and src.results:
        inner = _trace_value(
            _operand_names(src, state=state)[0],
            defining_map,
            value_refs,
            state=state,
            scope=scope,
        )
        dst = _scalar_type_text(src.results[0].type)
        if _cast_exact_op(src):
            # Value-preserving: the term is unchanged (its ssa name stays the
            # source's so codegen can audit).  A sign or zero extension of a
            # load tells the range analysis how the program reads it.
            if op_name in ("arith.extsi", "arith.extui") and isinstance(inner, WLoad):
                return dataclasses.replace(inner, signed=op_name == "arith.extsi")
            return inner
        if _const_value(inner) is not None and _constant_survives_cast(
            _const_value(inner), op_name, src.results[0].type
        ):
            # A constant proven representable in the destination type (the
            # frontend's `acc: f16 = 0` arrives as sitofp(0)).
            return WConst(
                _cast_constant(_const_value(inner), op_name),
                _result_names(src, state=state)[0],
            )
        narrowed = None
        if (
            op_name in ("arith.trunci", "arith.index_cast")
            and _int_bits(dst) is not None
        ):
            narrowed = _narrow_integer(
                inner, _int_bits(dst), getattr(state, "loop_bounds", {})
            )
        elif op_name == "arith.truncf":
            narrowed = _narrow_float(
                inner, _scalar_type_text(src.operands[0].type), dst, state
            )
        elif op_name in ("arith.sitofp", "arith.uitofp") and isinstance(
            inner, WBlockArg
        ):
            bounds = getattr(state, "loop_bounds", {}).get(inner.ssa_name)
            bits = _FLOAT_SIGNIFICAND_BITS.get(dst)
            if (
                bounds is not None
                and bits is not None
                and max(abs(bounds[0]), abs(bounds[1] - 1)) < (1 << bits)
            ):
                narrowed = inner
        if narrowed is not None:
            return narrowed
        return WUnary(
            f"{op_name}<{src.operands[0].type}->{dst}>",
            inner,
            _result_names(src, state=state)[0],
        )
    return WBlockArg(
        ssa_name=_result_names(src, state=state)[0] if src.results else ssa_name
    )


# --------------------------------------------------------------------- #
# Pattern unification
# --------------------------------------------------------------------- #


_COMMUTATIVE = {"add", "mul", "max", "min", "and", "or", "xor"}

_CONST_FOLD = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "and": lambda a, b: a & b,
    "or": lambda a, b: a | b,
    "xor": lambda a, b: a ^ b,
    "shl": lambda a, b: a << b,
    "shr": lambda a, b: a >> b,
}


def _const_value(term):
    """Numeric value of a constant-only term, else None.

    Literals such as ``-1`` reach the IR as ``subi 0, 1`` rather than a
    single ``arith.constant``, so add/sub/mul over constants are evaluated.
    """
    if isinstance(term, WConst):
        value = term.value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
        return None
    if isinstance(term, WBinOp) and term.op in _CONST_FOLD:
        lhs = _const_value(term.lhs)
        rhs = _const_value(term.rhs)
        if lhs is None or rhs is None:
            return None
        try:
            return _CONST_FOLD[term.op](lhs, rhs)
        except TypeError:  # bitwise op on a float constant
            return None
    if isinstance(term, WUnary) and term.fn == "neg":
        value = _const_value(term.arg)
        return None if value is None else -value
    return None


def _acc_param(op_obj, pattern) -> str | None:
    """The accumulator: the last parameter that is not a ``const_params`` or
    ``iv_params`` entry (``GRAD_LINEAR(x, y, acc, alpha, beta)``)."""
    if not op_obj.accumulates:
        return None
    skip = set(getattr(op_obj, "const_params", ())) | set(
        getattr(op_obj, "iv_params", ())
    )
    names = [name for name in pattern.param_names if name not in skip]
    return names[-1] if names else None


def _param_kinds(op_obj, pattern) -> dict[str, str] | None:
    """Per-parameter binding kind for an op with ``const_params``.

    None for every other op, which keeps the legacy unify-then-filter path.
    """
    const_params = getattr(op_obj, "const_params", ())
    iv_params = getattr(op_obj, "iv_params", ())
    if not const_params and not iv_params:
        return None
    return {
        name: (
            "const" if name in const_params else "iv" if name in iv_params else "load"
        )
        for name in pattern.param_names
    }


def _unify(pattern, term, bindings: dict[str, Any], kinds=None) -> bool:
    """Try to unify a pattern node against a workload term.

    On success ``bindings`` is populated with ``{param_name: WTerm}`` and
    True is returned. On failure ``bindings`` is left in an undefined
    state — callers should pass a fresh dict each top-level call.

    ``kinds`` (from :func:`_param_kinds`) constrains each parameter to a
    load or a constant during the search, so a commutative operand order
    that binds a constant parameter to a load is rejected and the other
    order is tried.
    """
    if isinstance(pattern, PVar):
        if kinds is not None:
            kind = kinds.get(pattern.name)
            if kind == "const" and _const_value(term) is None:
                return False
            if kind == "load" and not isinstance(term, WLoad):
                return False
            if kind == "iv" and not isinstance(term, WBlockArg):
                return False
        existing = bindings.get(pattern.name)
        if existing is None:
            bindings[pattern.name] = term
            return True
        # Same param appearing twice must bind to the same value (by SSA
        # name when available). For loads, compare ssa_name.
        return _term_eq(existing, term)

    if isinstance(pattern, PConst):
        if isinstance(term, WConst):
            # Literal values are int/float/str/None; == never raises on them.
            return term.value == pattern.value
        return False

    if isinstance(pattern, PBinOp):
        if not isinstance(term, WBinOp):
            return False
        if pattern.op != term.op:
            return False
        # Try direct order
        b = dict(bindings)
        if _unify(pattern.lhs, term.lhs, b, kinds) and _unify(
            pattern.rhs, term.rhs, b, kinds
        ):
            bindings.clear()
            bindings.update(b)
            return True
        if pattern.op in _COMMUTATIVE:
            b = dict(bindings)
            if _unify(pattern.lhs, term.rhs, b, kinds) and _unify(
                pattern.rhs, term.lhs, b, kinds
            ):
                bindings.clear()
                bindings.update(b)
                return True
        return False

    if isinstance(pattern, PUnary):
        if pattern.fn == "neg":
            # Float negation is `negf x`; integer negation lowers to
            # `subi 0, x`; a negated literal is already folded to a constant.
            if isinstance(term, WBinOp) and term.op == "sub":
                if _const_value(term.lhs) == 0:
                    return _unify_in_order([(pattern.arg, term.rhs)], bindings, kinds)
            if isinstance(pattern.arg, PConst) and not isinstance(term, WUnary):
                value = _const_value(term)
                try:
                    return value is not None and value == -pattern.arg.value
                except TypeError:
                    return False
        if not isinstance(term, WUnary) or pattern.fn != term.fn:
            return False
        return _unify_in_order([(pattern.arg, term.arg)], bindings, kinds)

    if isinstance(pattern, PSelect):
        if not isinstance(term, WSelect):
            return False
        return _unify_in_order(
            [(pattern.cond, term.cond), (pattern.a, term.a), (pattern.b, term.b)],
            bindings,
            kinds,
        )

    if isinstance(pattern, PCmp):
        if not isinstance(term, WCmp):
            return False
        if pattern.pred == term.pred and _unify_in_order(
            [(pattern.lhs, term.lhs), (pattern.rhs, term.rhs)], bindings, kinds
        ):
            return True
        # `x > y` is `y < x`; eq/ne mirror onto themselves.
        if _MIRRORED_PREDICATE[pattern.pred] == term.pred and _unify_in_order(
            [(pattern.lhs, term.rhs), (pattern.rhs, term.lhs)], bindings, kinds
        ):
            return True
        return False

    return False


def _unify_in_order(pairs, bindings: dict[str, Any], kinds=None) -> bool:
    b = dict(bindings)
    if all(_unify(p, t, b, kinds) for p, t in pairs):
        bindings.clear()
        bindings.update(b)
        return True
    return False


def _term_eq(a, b) -> bool:
    """Structural term equality (spec 004 M2).

    Two loads are equal when they read the same memref at the same index
    operands and map, and no store to that memref separates them (equal
    ``version``). Loads traced outside a function scope compare by SSA name.
    """
    if type(a) is not type(b):
        return False
    if isinstance(a, WLoad):
        if a.ssa_name == b.ssa_name:
            return True
        if a.version is None or a.version != b.version:
            return False
        same_memref = (
            a.value_ref == b.value_ref
            if a.value_ref is not None and b.value_ref is not None
            else a.memref_name == b.memref_name
        )
        return (
            same_memref
            and list(a.indices) == list(b.indices)
            and a.index_map == b.index_map
        )
    if isinstance(a, WBlockArg):
        return a.ssa_name == b.ssa_name
    if isinstance(a, WConst):
        return a.value == b.value
    if isinstance(a, WBinOp):
        return a.op == b.op and _term_eq(a.lhs, b.lhs) and _term_eq(a.rhs, b.rhs)
    if isinstance(a, WUnary):
        return a.fn == b.fn and _term_eq(a.arg, b.arg)
    if isinstance(a, WCmp):
        return a.pred == b.pred and _term_eq(a.lhs, b.lhs) and _term_eq(a.rhs, b.rhs)
    if isinstance(a, WSelect):
        return _term_eq(a.cond, b.cond) and _term_eq(a.a, b.a) and _term_eq(a.b, b.b)
    return False


def _contains_load(term) -> bool:
    if isinstance(term, WLoad):
        return True
    if isinstance(term, WBinOp):
        return _contains_load(term.lhs) or _contains_load(term.rhs)
    if isinstance(term, WUnary):
        return _contains_load(term.arg)
    if isinstance(term, (WCmp,)):
        return _contains_load(term.lhs) or _contains_load(term.rhs)
    if isinstance(term, WSelect):
        return any(_contains_load(t) for t in (term.cond, term.a, term.b))
    return False


def _data_dependent(source_op_name, index_terms) -> bool:
    """An index that depends on loaded data. ``affine`` accesses never do."""
    if source_op_name.startswith("affine.") or not index_terms:
        return False
    return any(_contains_load(term) for term in index_terms)


def _terms_eq(a, b) -> bool:
    return len(a) == len(b) and all(_term_eq(x, y) for x, y in zip(a, b))


# --------------------------------------------------------------------- #
# IR walking
# --------------------------------------------------------------------- #


_FUNC_RE = re.compile(r"^(?P<work>.+?)((?:_\d+)+)$")


def _parse_work_id(func_name: str, work_root: str | None = None):
    """Return (work_root, (id, ...)) parsed from `<work>_<d>(_<d>)*`.

    If the name doesn't match, returns (func_name, ()).
    """
    m = _FUNC_RE.match(func_name)
    if not m:
        return func_name, ()
    root = m.group("work")
    tail = m.group(2)
    ids = tuple(int(p) for p in tail.split("_") if p)
    if work_root is not None and root != work_root:
        return func_name, ()
    return root, ids


def _affine_map_text(attr) -> str:
    return str(attr)


def _walk_loops(
    block, prefix: list[tuple[str, str, str, int]], collector, *, state, guards=()
):
    """Recurse into ``block``'s ops, collecting affine.for loops in ``prefix``
    and yielding (op, current_loop_stack, guard_frames) for every non-loop op
    via ``collector.append``.

    Guard frames (spec 004 M5): the then region of ``scf.if`` pushes
    ``("if", cond_ssa)``, the else region ``("not", cond_ssa)``; ``affine.if``
    pushes ``("affine_if", set_text, operand_names)`` (negated as
    ``("not_affine_if", ...)``). Frames are outermost first.
    """
    for op in block.operations:
        if op.operation.name == "affine.for":
            attrs = op.attributes
            iv_name = _value_name(op.regions[0].blocks[0].arguments[0], state)
            lb = (
                _affine_map_text(attrs["lowerBoundMap"])
                if "lowerBoundMap" in attrs
                else "?"
            )
            ub = (
                _affine_map_text(attrs["upperBoundMap"])
                if "upperBoundMap" in attrs
                else "?"
            )
            step_attr = attrs["step"] if "step" in attrs else None
            try:
                step = (
                    int(str(step_attr).split(":")[0].strip())
                    if step_attr is not None
                    else 1
                )
            except ValueError:
                step = 1
            loop = (iv_name, lb, ub, step)
            for inner_block in op.regions[0].blocks:
                _walk_loops(
                    inner_block, prefix + [loop], collector, state=state, guards=guards
                )
        else:
            collector.append((op, list(prefix), tuple(guards)))
            name = op.operation.name
            for region_index, r in enumerate(op.regions):
                frames = guards
                if name == "scf.if":
                    cond = _value_name(op.operands[0], state)
                    frames = guards + (
                        (("if", cond) if region_index == 0 else ("not", cond)),
                    )
                elif name == "affine.if":
                    condition = (
                        str(op.attributes["condition"])
                        if "condition" in op.attributes
                        else "?"
                    )
                    operands = tuple(_value_name(o, state) for o in op.operands)
                    tag = "affine_if" if region_index == 0 else "not_affine_if"
                    frames = guards + ((tag, condition, operands),)
                for blk in r.blocks:
                    _walk_loops(blk, prefix, collector, state=state, guards=frames)


def _build_defining_map(func, *, state) -> dict[str, Any]:
    m: dict[str, Any] = {}

    def visit(block):
        for op in block.operations:
            for r in op.results:
                m[_value_name(r, state)] = op
            for region in op.regions:
                for blk in region.blocks:
                    visit(blk)

    body = func.regions[0].blocks[0]
    visit(body)
    return m


def _is_memref_value(value) -> bool:
    return str(getattr(value, "type", "")).startswith("memref<")


def _walk_ir_values(block, prefix, collector):
    for operation_index, operation in enumerate(block.operations):
        path = prefix + (operation_index,)
        collector.append((operation, path))
        for region_index, region in enumerate(operation.regions):
            for block_index, nested in enumerate(region.blocks):
                _walk_ir_values(
                    nested,
                    path + (region_index, block_index),
                    collector,
                )


def _retained_abi_value_ids(function) -> tuple[int, ...] | None:
    attributes = function.attributes
    name = "spmw.abi_value_ids"
    if name not in attributes:
        return None
    source_ids = tuple(
        int(getattr(value, "value", value)) for value in attributes[name]
    )
    if len(source_ids) != len(function.arguments):
        raise ValueError("retained matcher ABI identity has the wrong arity")
    if any(value < 0 for value in source_ids):
        raise ValueError("retained matcher ABI identity must be non-negative")
    return source_ids


def _callee_symbol(operation) -> str | None:
    if operation.operation.name != "func.call" or "callee" not in operation.attributes:
        return None
    symbol = _attr_str(operation.attributes["callee"])
    return symbol[1:] if symbol and symbol.startswith("@") else symbol


def _build_ir_value_refs(
    mlir_module,
) -> tuple[dict[Any, IRValueRef], dict[int, IRValueRef]]:
    """Canonicalize exact MLIR def-use, call, and retained ABI edges.

    The second result retains the frontend source-ID to canonical-reference
    relation.  Public compilation uses it to bind annotation-derived geometry
    to matcher values without consulting diagnostic buffer names or shapes.
    """

    functions = [
        function
        for function in mlir_module.body.operations
        if function.operation.name == "func.func" and "sym_name" in function.attributes
    ]
    functions_by_symbol = {
        _attr_str(function.attributes["sym_name"]): function for function in functions
    }
    order = {}
    parents = {}
    calls = []
    retained_sources = {}

    def add(value, position):
        if not _is_memref_value(value):
            return
        if value not in parents:
            parents[value] = value
            order[value] = tuple(int(component) for component in position)

    def find(value):
        parent = parents[value]
        if parent != value:
            parents[value] = find(parent)
        return parents[value]

    def union(lhs, rhs):
        if lhs not in parents or rhs not in parents:
            return
        lhs_root = find(lhs)
        rhs_root = find(rhs)
        if lhs_root == rhs_root:
            return
        if order[rhs_root] < order[lhs_root]:
            lhs_root, rhs_root = rhs_root, lhs_root
        parents[rhs_root] = lhs_root

    for function_index, function in enumerate(functions):
        for argument_index, argument in enumerate(function.arguments):
            add(argument, (function_index, 0, argument_index))
        source_ids = _retained_abi_value_ids(function)
        if source_ids is not None:
            for argument, source_id in zip(function.arguments, source_ids):
                if _is_memref_value(argument):
                    retained_sources.setdefault(source_id, []).append(argument)

        operations = []
        _walk_ir_values(function.regions[0].blocks[0], (), operations)
        for operation, path in operations:
            for result_index, result in enumerate(operation.results):
                add(result, (function_index, 1, *path, result_index))
            for operand_index, operand in enumerate(operation.operands):
                add(operand, (function_index, 2, *path, operand_index))
            if operation.operation.name == "memref.cast":
                if len(operation.operands) == len(operation.results) == 1 and str(
                    operation.operands[0].type
                ) == str(operation.results[0].type):
                    union(operation.operands[0], operation.results[0])
            if operation.operation.name == "func.call":
                calls.append(operation)

    for source_id, values in retained_sources.items():
        types = {str(value.type) for value in values}
        if len(types) != 1:
            raise ValueError(
                f"retained matcher ABI source {source_id} has incompatible types"
            )
        for value in values[1:]:
            union(values[0], value)

    for call in calls:
        symbol = _callee_symbol(call)
        callee = functions_by_symbol.get(symbol)
        if callee is None:
            continue
        if len(call.operands) != len(callee.arguments):
            raise ValueError("matcher call edge has incompatible ABI arity")
        for operand, argument in zip(call.operands, callee.arguments):
            if _is_memref_value(operand) != _is_memref_value(argument):
                raise ValueError("matcher call edge has incompatible ABI value kinds")
            if _is_memref_value(operand):
                if str(operand.type) != str(argument.type):
                    raise ValueError("matcher call edge has incompatible memref types")
                union(operand, argument)

    component_sources = {}
    for source_id, values in retained_sources.items():
        for value in values:
            component_sources.setdefault(find(value), set()).add(source_id)
    if any(len(source_ids) != 1 for source_ids in component_sources.values()):
        raise ValueError("one matcher def-use component has conflicting ABI identities")

    components = {}
    for value in parents:
        components.setdefault(find(value), []).append(value)
    refs = {}
    for members in components.values():
        canonical_path = min(order[value] for value in members)
        value_ref = IRValueRef("mlir", canonical_path)
        for value in members:
            refs[value] = value_ref

    source_value_refs = {}
    for source_id, values in retained_sources.items():
        retained_refs = {refs[value] for value in values}
        if len(retained_refs) != 1:
            raise ValueError("one retained source has conflicting value identities")
        source_value_refs[int(source_id)] = next(iter(retained_refs))

    # An ordinary (non-dataflow) workload has no retained source-ID attribute.
    # Its sole function ABI is nevertheless an exact structural source.
    if not retained_sources and len(functions) == 1:
        for argument_index, argument in enumerate(functions[0].arguments):
            if _is_memref_value(argument):
                source_value_refs[argument_index] = refs[argument]
    return refs, source_value_refs


# --------------------------------------------------------------------- #
# Match driver
# --------------------------------------------------------------------- #


def _operand_bindings(
    pattern: OpPattern,
    bindings: dict[str, Any],
    acc_name: str | None,
) -> list[OperandBinding]:
    """Build OperandBindings (in fn signature order) from a successful unify.

    ``acc_name`` is the loop-carried accumulator parameter (see
    :func:`_acc_param`), or None for a non-accumulating op.
    """
    out: list[OperandBinding] = []
    for name in pattern.param_names:
        term = bindings.get(name)
        if isinstance(term, WLoad):
            out.append(
                OperandBinding(
                    role=name,
                    memref_name=term.memref_name,
                    indices=list(term.indices),
                    is_loop_carried=(name == acc_name),
                    memref_type=term.memref_type,
                    value_ref=term.value_ref,
                    index_map=term.index_map,
                    coords=_coords_of_terms(term.index_terms, term.index_map),
                )
            )
        elif isinstance(term, WBlockArg):
            out.append(
                OperandBinding(
                    role=name,
                    memref_name=None,
                    indices=[term.ssa_name],
                    is_loop_carried=(name == acc_name),
                )
            )
        elif isinstance(term, WConst):
            out.append(
                OperandBinding(
                    role=name,
                    memref_name=None,
                    indices=[str(term.value)],
                    is_loop_carried=(name == acc_name),
                )
            )
        else:  # WBinOp / unknown — leave a conservative entry
            out.append(
                OperandBinding(
                    role=name,
                    memref_name=None,
                    indices=[],
                    is_loop_carried=(name == acc_name),
                )
            )
    return out


def _flatten_add_chain(term, result_memref_name):
    """Flatten a left-nested associative `add`-chain into
    `(acc_load, [summand, ...])` for the multi-term reduction matcher (D3).

    The chain `add(add(add(load(A), t1), t2), t3)` accumulating onto memref `A`
    is collected as `acc_load = load(A)`, `summands = [t1, t2, t3]`. Returns
    `(None, [])` when `term` is not an add-chain whose innermost left leaf is a
    `WLoad` of `result_memref_name` (so the flatten is inert outside the
    accumulate shape it targets). Pure term-tree walk -- reads only the existing
    `WBinOp`/`WLoad` tree from `_trace_value`; no IR re-walk, no allo/ir edit.
    """
    if not isinstance(term, WBinOp) or term.op != "add":
        return None, []
    summands: list[Any] = []
    node = term
    # Walk down the left spine collecting right-hand summands.
    while isinstance(node, WBinOp) and node.op == "add":
        summands.append(node.rhs)
        node = node.lhs
    # `node` is now the innermost left leaf -- the accumulator load.
    if not isinstance(node, WLoad):
        return None, []
    if (
        result_memref_name is not None
        and node.memref_name is not None
        and node.memref_name != result_memref_name
    ):
        return None, []
    summands.reverse()  # restore source order
    return node, summands


_AFFINE_MAP_RE = re.compile(
    r"^affine_map<\((?P<dims>[^)]*)\)(?:\[[^\]]*\])?\s*->\s*\((?P<results>.*)\)>$"
)


def _affine_map_last_result(text: str) -> tuple[int, str] | None:
    """Return ``(dim_count, last_result)`` of an affine-map attribute text."""
    m = _AFFINE_MAP_RE.match(text.strip())
    if m is None:
        return None
    dims = [d for d in m.group("dims").split(",") if d.strip()]
    results, depth, start = [], 0, 0
    body = m.group("results")
    for i, ch in enumerate(body):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            results.append(body[start:i])
            start = i + 1
    results.append(body[start:])
    last = results[-1].strip()
    if not last:
        return None
    return len(dims), last


def _load_last_axis_is(load: WLoad, iv: str) -> bool:
    """True iff ``load``'s last access coordinate is exactly ``iv``."""
    if not load.indices:
        return False
    if load.source_op_name == "memref.load":
        return load.indices[-1] == iv
    if load.index_map is None:
        return False
    parsed = _affine_map_last_result(load.index_map)
    if parsed is None:
        return False
    dim_count, last = parsed
    return any(
        index == iv and position < dim_count and last == f"d{position}"
        for position, index in enumerate(load.indices)
    )


def _matched_vector_width(op_obj, pattern, bindings, enclosing_loops) -> int:
    width = getattr(op_obj, "vector_width", 1)
    if width == 1 or not enclosing_loops:
        return 1
    innermost_iv = enclosing_loops[-1][0]
    acc_name = _acc_param(op_obj, pattern)
    for name in pattern.param_names:
        if name == acc_name:
            continue
        term = bindings.get(name)
        if isinstance(term, WLoad) and _load_last_axis_is(term, innermost_iv):
            return width
    return 1


def _try_match_at_store(
    store_op,
    enclosing_loops: list[tuple[str, str, str, int]],
    target,
    func_name: str,
    work_id: tuple[int, ...],
    defining_map: dict[str, Any],
    value_refs: dict[Any, IRValueRef],
    *,
    state,
    scope=None,
    guards=(),
) -> list[MatchedOp]:
    """If ``store_op`` (a memref.store / affine.store) writes a value that
    matches one of the target's compiled op patterns, emit MatchedOps for it.
    """
    op_name = store_op.operation.name
    if op_name not in _STORE_OPS:
        return []
    operands = _operand_names(store_op, state=state)
    if not operands:
        return []
    stored_ssa = operands[0]
    # The memref being stored into
    memref_ssa = operands[1] if len(operands) > 1 else None
    # SPEC-026 §1.2: the store's index SSA names (everything after the
    # value + memref operands). Carried additively on MatchedOp.extra so
    # the batch-dim resolver can test the result's leading index without a
    # second IR walk. None of the existing match contract changes.
    store_indices = operands[2:] if len(operands) > 2 else []
    result_memref_name = None
    if "to" in store_op.attributes:
        result_memref_name = _attr_str(store_op.attributes["to"])
    result_value_ref = (
        value_refs.get(store_op.operands[1]) if len(store_op.operands) > 1 else None
    )

    term = _trace_value(stored_ssa, defining_map, value_refs, state=state, scope=scope)
    # Only a computed term is interesting; a bare load/const store is a copy,
    # except for ops that move data to a computed destination (M6).
    computed = isinstance(term, (WBinOp, WUnary, WSelect, WCmp))
    if not computed and not isinstance(term, WLoad):
        return []

    store_handle = f"{op_name}@{stored_ssa}"
    store_data_dependent = False
    store_map = (
        str(store_op.attributes["map"]) if "map" in store_op.attributes else None
    )
    index_terms = None
    guard_terms = ()
    if scope is not None:
        index_terms = tuple(
            _trace_value(index, defining_map, value_refs, state=state, scope=scope)
            for index in store_indices
        )
        store_data_dependent = _data_dependent(op_name, index_terms)
        guard_terms = tuple(
            (
                (
                    frame[0],
                    _trace_value(
                        frame[1], defining_map, value_refs, state=state, scope=scope
                    ),
                )
                if frame[0] in ("if", "not")
                else frame
            )
            for frame in guards
        )

    def _match_term(t) -> MatchedOp | None:
        """Unify one computed term against the target's op patterns; return
        the first matching MatchedOp (the existing single-term logic), or None."""
        for unit in target._walk():
            for op_obj in unit.ops.values():
                if not op_obj.matchable:
                    continue
                # M5: a guarded store matches only ops that opt in.
                if guards and not getattr(op_obj, "guarded", False):
                    continue
                any_index = getattr(op_obj, "dst_index", "affine") == "any"
                if not computed and not any_index:
                    continue
                # An ordinary-index op never writes a data-dependent element
                # (that is a scatter, spec 004 M6 and its 016 review).
                if not any_index and store_data_dependent:
                    continue
                candidate = t
                if isinstance(t, WUnary) and t.fn.startswith("arith.truncf<"):
                    # A store-side float narrowing the matcher kept visible
                    # (compound expression evaluated in the wider type): only
                    # an op that declares it evaluates its whole expression
                    # with at least that intermediate precision and rounds
                    # once may bind the expression under the cast.
                    types = _truncf_types(t.fn)
                    declared = getattr(op_obj, "intermediate", None)
                    if (
                        types is None
                        or declared is None
                        or _FLOAT_SIGNIFICAND_BITS.get(declared, 0)
                        < _FLOAT_SIGNIFICAND_BITS.get(types[0], 1 << 30)
                        or not _float_leaf_fits(
                            t.arg, types[1], getattr(state, "loop_bounds", {})
                        )
                    ):
                        continue
                    candidate = t.arg
                # Precision contract (C-26): an operation the source evaluates
                # in a wider type than its operands binds only to an op that
                # declares it computes in that precision.
                if arithmetic_precision_problem(
                    candidate
                ) is not None and not _op_computes_wide(op_obj, candidate):
                    continue
                pat = compile_op_pattern(op_obj)
                bindings: dict[str, Any] = {}
                kinds = _param_kinds(op_obj, pat)
                if not _unify(pat.body, candidate, bindings, kinds):
                    continue
                constants = None
                if kinds is None:
                    # Conservative filter: every parameter must bind to a memref
                    # load. Block-arg / constant / opaque-cast bindings are
                    # rejected so we don't match index-arithmetic chains (e.g.
                    # `pid * 8` computing `row0`) against data-plane ops.
                    if not all(
                        isinstance(bindings.get(n), WLoad) for n in pat.param_names
                    ):
                        continue
                else:
                    # `const_params` bind to compile-time constants and
                    # `iv_params` to enclosing loop variables; every other
                    # parameter keeps the memref-load rule.
                    if not all(n in bindings for n in pat.param_names):
                        continue
                    constants = {}
                    iv_positions = {}
                    loop_names = [loop[0] for loop in enclosing_loops]
                    for n in pat.param_names:
                        if kinds[n] == "const":
                            constants[n] = _const_value(bindings[n])
                            bindings[n] = WConst(constants[n], bindings[n].ssa_name)
                        elif kinds[n] == "iv" and bindings[n].ssa_name in loop_names:
                            iv_positions[n] = loop_names.index(bindings[n].ssa_name)
                    if any(
                        kinds[n] == "iv" and n not in iv_positions
                        for n in pat.param_names
                    ):
                        continue
                # If this op accumulates, validate the accumulator: the last
                # parameter must bind to a WLoad whose memref equals the
                # store's "to" memref (i.e. the same accumulator memref).
                acc_name = _acc_param(op_obj, pat)
                if acc_name is not None:
                    acc_term = bindings[acc_name]
                    if not isinstance(acc_term, WLoad):
                        continue
                    if (
                        result_memref_name is not None
                        and acc_term.memref_name is not None
                        and acc_term.memref_name != result_memref_name
                    ):
                        continue
                    if (
                        result_value_ref is not None
                        and acc_term.value_ref is not None
                        and acc_term.value_ref != result_value_ref
                    ):
                        continue
                    # M6: the accumulator is read at the stored element.
                    if any_index:
                        if (
                            index_terms is None
                            or acc_term.index_terms is None
                            or acc_term.index_map != store_map
                            or not _terms_eq(acc_term.index_terms, index_terms)
                        ):
                            continue
                    elif acc_term.index_map != store_map or _data_dependent(
                        acc_term.source_op_name, acc_term.index_terms
                    ):
                        continue
                    elif list(acc_term.indices) != list(store_indices) and not (
                        # The same element may be addressed through distinct
                        # SSA values (``y[row0 + r]`` recomputed for the load
                        # and the store): compare the traced index terms.
                        index_terms is not None
                        and acc_term.index_terms is not None
                        and _terms_eq(acc_term.index_terms, index_terms)
                    ):
                        continue
                extra = {"store_indices": list(store_indices)}
                if memref_ssa is not None and len(store_op.operands) > 1:
                    # Shaped type of the destination (backends check the
                    # output's extent and element type against the program).
                    extra["result_memref_type"] = str(store_op.operands[1].type)
                if index_terms is not None:
                    # Resolved store coordinates (`affine_form`); consumers
                    # compare them with the operands' ``coords`` instead of
                    # reading index text (`output_map_problem`).
                    extra["store_coords"] = _coords_of_terms(index_terms, store_map)
                if index_terms is not None and not any(
                    _term_loads(t, []) for t in index_terms
                ):
                    # Loop variables each store coordinate depends on (the
                    # coordinate may be an SSA value such as ``row0 + r``).
                    extra["store_index_vars"] = tuple(
                        tuple(sorted(_term_blockargs(t, set()))) for t in index_terms
                    )
                if constants is not None and getattr(op_obj, "const_params", ()):
                    extra["constants"] = constants
                if kinds is not None and any(k == "iv" for k in kinds.values()):
                    extra["iv_params"] = iv_positions
                if any_index:
                    extra["index_terms"] = index_terms
                if guards:
                    extra["guards"] = guard_terms
                retained_loop_names = getattr(state, "loop_names", None)
                if retained_loop_names:
                    # Source loop labels of the enclosing loops (diagnostics
                    # and the shared semantic facts; never used to match).
                    extra["loop_names"] = {
                        loop[0]: retained_loop_names[loop[0]]
                        for loop in enclosing_loops
                        if loop[0] in retained_loop_names
                    }
                # op_range — first contributing load through the store.
                return MatchedOp(
                    target_op_name=op_obj.name,
                    func_name=func_name,
                    work_id=work_id,
                    enclosing_loops=list(enclosing_loops),
                    operands=_operand_bindings(pat, bindings, _acc_param(op_obj, pat)),
                    result_memref_name=result_memref_name,
                    op_range=(t.ssa_name, store_handle),
                    extra=extra,
                    result_value_ref=result_value_ref,
                    vector_width=_matched_vector_width(
                        op_obj, pat, bindings, enclosing_loops
                    ),
                )
            # (no break needed — return above exits on first match)
        return None

    # 1) The canonical single-term path: try the stored term as-is. A
    #    single-`mul` GEMV/GEMM (`add(acc, mul(x,y))`) matches MAC here and the
    #    multi-term flatten below NEVER fires -> byte-identical.
    m = _match_term(term)
    if m is not None:
        return [m]

    # 2) SPEC-004 (multi-output sub-spec D3) additive multi-term reduction
    #    flatten. Reached ONLY when the single-term unify failed. A left-nested
    #    associative `add`-chain accumulating onto the store's `to` memref --
    #    e.g. gemver's `A[i,j] = A[i,j] + u1*v1 + u2*v2` -> two MAC terms, or
    #    `x[i] = x[i] + z[i]` -> one reduction-free ADD term -- is flattened
    #    into per-summand terms, each re-matched as `add(acc_load, summand)`.
    #    Rides the existing `_trace_value` WBinOp tree; no allo/ir edit.
    acc_load, summands = _flatten_add_chain(term, result_memref_name)
    if acc_load is not None and len(summands) >= 2:
        flat: list[MatchedOp] = []
        for s in summands:
            # Re-form `add(acc_load, summand)` and match it as a standalone
            # accumulate (MAC for a mul summand, ADD-family for a load summand).
            synth = WBinOp("add", acc_load, s, term.ssa_name)
            sm = _match_term(synth)
            if sm is None:
                # A summand that does not match any pattern aborts the flatten
                # (do not emit a partial / fabricated decomposition).
                return []
            flat.append(sm)
        return flat
    return []


def _parse_loop_upper(text: str) -> int | None:
    """Best-effort integer from an affine-map upper-bound string.

    Mirrors the cost model's `_parse_loop_bound`; kept local so the
    resolver remains independent of cost-program lowering (which imports this
    module). Returns None on failure.
    """
    try:
        return int(text.strip())
    except ValueError:
        pass
    nums = re.findall(r"\b(\d+)\b", text)
    if len(nums) == 1:
        return int(nums[0])
    return None


def _accumulator_scope_depth(acc: OperandBinding) -> int | None:
    """Number of enclosing region ops around the accumulator's definition.

    Reads the canonical ``IRValueRef`` path built by `_build_ir_value_refs`:
    ``(function, 0, argument)`` for a function argument (depth 0) and
    ``(function, 1, op, (region, block, op)*, result)`` for an op result,
    where each ``(region, block, op)`` triple is one nesting level.
    """
    ref = acc.value_ref
    if ref is None or ref.namespace != "mlir":
        return None
    path = ref.path
    if len(path) == 3 and path[1] == 0:
        return 0
    if len(path) < 4 or path[1] != 1 or (len(path) - 4) % 3:
        return None
    return (len(path) - 4) // 3


def _axis_roles(match: MatchedOp) -> dict:
    """Structural role of every enclosing loop of one accumulating match.

    The result coordinate of an accumulation is the accumulator element it
    stores to.  For an indexed accumulator (``y[i] += ...``) the result loops
    are the loop variables in the store index (``extra['store_indices']``).
    For an accumulator element whose index carries no loop variable
    (``acc: T = 0`` declared in a loop body, or ``y[0] += ...``) a fresh
    element exists per iteration of exactly the loops that enclose its
    definition, so those loops are the result loops.  Every other enclosing
    loop is a reduction loop.

    The two inputs split into the matrix and the vector: the matrix is the
    input whose index set strictly contains the other's (``K[h, s, d]`` over
    ``q[h, d]``), otherwise the first input in pattern order.  A result loop
    is then a BATCH axis iff its variable indexes the vector (``X[b, k]``,
    and also the per-head ``h`` of ``q[h, d]``/``K[h, s, d]``: each value
    selects its own vector), and an OUTPUT axis otherwise (``W[i, k]`` /
    ``W[row0 + i, k]`` with ``x[k]``: the vector is shared).  When both
    inputs bind one buffer there is no shared vector and every result loop
    is an output axis.

    Returns ``{"reduction", "output", "batch"}`` (loop-variable tuples in
    loop order), ``trips`` (variable -> static trip count), ``matrix`` /
    ``vector`` (input roles) and ``unresolved``: ``None`` or the reason the
    roles cannot be proven from the retained match.  Matches without a
    loop-carried accumulator carry no roles (``unresolved`` says so).
    """
    roles = {
        "reduction": (),
        "output": (),
        "batch": (),
        "trips": {},
        "matrix": None,
        "vector": None,
        "same_buffer": False,
        "unresolved": None,
    }

    def unresolved(reason):
        roles["unresolved"] = reason
        return roles

    loops = list(match.enclosing_loops)
    loop_vars = [loop[0] for loop in loops]
    accumulators = [opb for opb in match.operands if opb.is_loop_carried]
    inputs = [
        opb
        for opb in match.operands
        if not opb.is_loop_carried and opb.memref_name is not None
    ]
    if len(accumulators) != 1:
        return unresolved("match has no single loop-carried accumulator")
    if not loops:
        return unresolved("accumulation has no enclosing loop")
    if len(inputs) != 2:
        return unresolved(f"expected two memref inputs, got {len(inputs)}")

    trips = {}
    for var, lower, upper, step in loops:
        lo, hi = _parse_loop_upper(str(lower)), _parse_loop_upper(str(upper))
        if lo is None or hi is None or int(step) <= 0:
            return unresolved(f"loop {var} has no static trip count ({upper})")
        trips[var] = max(0, -(-(hi - lo) // int(step)))
    roles["trips"] = trips

    first, second = inputs
    same_buffer = first.memref_name == second.memref_name
    if same_buffer:
        matrix = vector = first
    elif set(second.indices) > set(first.indices):
        matrix, vector = second, first
    else:
        matrix, vector = first, second
    roles["matrix"], roles["vector"] = matrix.role, vector.role
    roles["same_buffer"] = same_buffer

    (acc,) = accumulators
    store = list(match.extra.get("store_indices", ()))
    if store:
        opaque = [index for index in store if index not in loop_vars]
        if opaque:
            # ``y[row0 + r]``: the coordinate is an SSA value; the matcher
            # recorded the loop variables it depends on (loads excluded).
            depends = match.extra.get("store_index_vars")
            if depends is None:
                return unresolved(
                    f"result index {opaque} is not an enclosing loop variable "
                    "and its dependence on the loops is not retained"
                )
            result_vars = [
                var for var in loop_vars if any(var in vars_ for vars_ in depends)
            ]
        else:
            result_vars = [var for var in loop_vars if var in store]
    else:
        if match.extra.get("guards"):
            return unresolved("guarded accumulator scope")
        depth = _accumulator_scope_depth(acc)
        if depth is None:
            return unresolved("accumulator definition scope is not retained")
        if depth > len(loops):
            return unresolved("accumulator is defined below the matched loops")
        result_vars = loop_vars[:depth]
    roles["reduction"] = tuple(var for var in loop_vars if var not in result_vars)

    vector_opaque = [index for index in vector.indices if index not in loop_vars]
    outputs, batches = [], []
    for var in result_vars:
        if same_buffer:
            outputs.append(var)
        elif var in vector.indices:
            batches.append(var)
        elif vector_opaque:
            return unresolved(
                f"vector index {vector_opaque} may depend on result loop {var}"
            )
        else:
            outputs.append(var)
    roles["output"], roles["batch"] = tuple(outputs), tuple(batches)
    return roles


def batch_dim(match: MatchedOp) -> tuple[str | None, int | None]:
    """Return ``(batch_loop_var, B)`` for a batched reduction match.

    The batch axes are the result loops whose variable indexes the vector
    input (see `_axis_roles`); ``B`` is the product of their trip counts and
    ``batch_loop_var`` the outermost one.  The output loop of ``W[f(i), k] *
    x[k]`` is never a batch: it does not index the vector.

    Returns ``(None, None)`` when the match has no batch axis or its roles
    are unresolved; downstream treats that as ``B = 1``.
    """
    roles = _axis_roles(match)
    if roles["unresolved"] is not None or not roles["batch"]:
        return (None, None)
    extent = 1
    for var in roles["batch"]:
        extent *= roles["trips"][var]
    return (roles["batch"][0], extent)


def _stamp_batch_dims(trace: MatchTrace) -> None:
    """Write `match.extra['batch_loop_var']` / `['batch_dim']` for every
    reduction match in ``trace`` (SPEC-026 §1.2).

    Single source of truth: cost (205) and codegen (206) read
    `extra['batch_dim']` and never re-derive it. A match with no batch
    axis gets ``batch_dim = 1`` (I4 parity).

    Also stamps ``extra['axis_roles']`` (see `_axis_roles`) and, per matched
    position of a work-item group (functions ``<root>_<id>...``), the whole
    group fact ``axis_roles['work_axis']``: which contraction axis the work id
    partitions.  It is ``"output"`` when only the matrix access differs
    between work items (or hides an index that is not a loop variable) while
    the vector access is identical and fully visible, ``"batch"`` in the
    mirrored case, and ``None`` otherwise, with the reason in
    ``axis_roles['work_axis_reason']``.
    """

    def access_status(column, role):
        accesses = set()
        opaque = False
        for match in column:
            operand = next(op for op in match.operands if op.role == role)
            loop_vars = {loop[0] for loop in match.enclosing_loops}
            accesses.add(
                (operand.memref_name, tuple(operand.indices), operand.index_map)
            )
            opaque = opaque or any(index not in loop_vars for index in operand.indices)
        if len(accesses) > 1:
            return "varies"
        return "opaque" if opaque else "invariant"

    def work_axis(column):
        roles = [match.extra["axis_roles"] for match in column]
        if any(r["unresolved"] is not None for r in roles):
            return None, "axis roles are unresolved"
        first = roles[0]
        if any(
            (r["matrix"], r["vector"], r["trips"])
            != (first["matrix"], first["vector"], first["trips"])
            for r in roles
        ):
            return None, "work items differ in operand roles or loop extents"
        if first["same_buffer"]:
            if access_status(column, first["matrix"]) == "invariant":
                return None, "no input access depends on the work id"
            return None, "the single shared input depends on the work id"
        matrix = access_status(column, first["matrix"])
        vector = access_status(column, first["vector"])
        if vector == "invariant" and matrix != "invariant":
            return "output", "only the matrix access depends on the work id"
        if matrix == "invariant" and vector != "invariant":
            return "batch", "only the vector access depends on the work id"
        if matrix == vector == "invariant":
            return None, "no input access depends on the work id"
        return None, (
            f"matrix access {matrix} and vector access {vector} across work "
            "items: the work id may split the reduction or a shared axis"
        )

    groups: dict[str, dict[str, list]] = {}
    for m in trace.matches:
        roles = _axis_roles(m)
        batch = roles["batch"] if roles["unresolved"] is None else ()
        extent = 1
        for var in batch:
            extent *= roles["trips"][var]
        roles["work_axis"] = None
        roles["work_axis_reason"] = "one work item"
        m.extra["axis_roles"] = roles
        m.extra["batch_loop_var"] = batch[0] if batch else None
        m.extra["batch_dim"] = extent
        root, work_id = _parse_work_id(m.func_name)
        if work_id:
            groups.setdefault(root, {}).setdefault(m.func_name, []).append(m)

    for bodies in groups.values():
        bodies = list(bodies.values())
        if len(bodies) < 2:
            continue
        uniform = all(len(body) == len(bodies[0]) for body in bodies)
        for position in range(max(len(body) for body in bodies)):
            column = [body[position] for body in bodies if position < len(body)]
            if uniform:
                axis, reason = work_axis(column)
            else:
                axis, reason = None, "work items differ in matched structure"
            for m in column:
                m.extra["axis_roles"]["work_axis"] = axis
                m.extra["axis_roles"]["work_axis_reason"] = reason


def match_workload(target, mlir_module, *, reduction_order=None) -> MatchTrace:
    """Walk ``mlir_module`` and emit a :class:`MatchTrace` for ``target``.

    Eagerly compiles target Op patterns the first time it sees them.
    Every accumulating match carries ``extra["reduction_policy"]`` for
    ``reduction_order`` (``"strict"`` when None; ``allo.spmw_reduction_policy``).
    """
    compile_target_patterns(target)
    value_refs, source_value_refs = _build_ir_value_refs(mlir_module)
    trace = MatchTrace(
        target_name=getattr(target, "name", "<unknown>"),
        module_name=str(getattr(mlir_module, "name", "<unnamed>")),
        source_value_refs=source_value_refs,
        coverage=MatchCoverage(),
    )
    coverage_problems: dict = {}

    # Iterate every func.func at the module top level.
    for func in mlir_module.body.operations:
        if func.operation.name != "func.func":
            continue
        if "sym_name" not in func.attributes:
            continue
        func_name = _attr_str(func.attributes["sym_name"])
        if func_name is None:
            continue

        _, work_id = _parse_work_id(func_name)
        # Be permissive: also accept funcs with no numeric tail.
        # Value.get_name() without an AsmState re-prints the enclosing
        # function on every call (quadratic in function size); one shared
        # state per function yields the same names in linear time.
        state = ValueNames(func, AsmState(func))
        defining_map = _build_defining_map(func, state=state)

        body = func.regions[0].blocks[0]
        scope = _build_function_scope(func, value_refs, state=state)
        sites: list = []
        _walk_loops(body, [], sites, state=state)

        site_matches: dict[int, list[MatchedOp]] = {}
        for site_index, (op, loops, guards) in enumerate(sites):
            if op.operation.name not in _STORE_OPS:
                continue
            ms = _try_match_at_store(
                op,
                loops,
                target,
                func_name,
                work_id,
                defining_map,
                value_refs,
                state=state,
                scope=scope,
                guards=guards,
            )
            if ms:
                site_matches[site_index] = ms
            trace.matches.extend(ms)

        _FunctionCoverage(
            target,
            func,
            func_name,
            sites,
            site_matches,
            defining_map,
            value_refs,
            state=state,
            scope=scope,
        ).run(trace.coverage, coverage_problems)

    trace.coverage.problems = [
        (tuple(functions), message) for message, functions in coverage_problems.values()
    ]
    trace.coverage.errors = _format_coverage_problems(trace.coverage.problems)

    # SPEC-026 §1.2: stamp the batch dim on each match (one source of
    # truth for cost/codegen). Additive; default batch_dim=1.
    _stamp_batch_dims(trace)
    from .spmw_reduction_policy import stamp_reduction_policy

    stamp_reduction_policy(trace, reduction_order)
    return trace


# --------------------------------------------------------------------- #
# Store coverage and accumulator initialization (release gate G2)
# --------------------------------------------------------------------- #
#
# After matching, every store of every function is put in exactly one
# category (``spmw_match.STORE_*``):
#
#   matched           the store produced a MatchedOp.
#   accumulator_init  a non-matched store into the result memref of an
#                     accumulating match in the same function.  Its legality
#                     is the accumulator-initialization fact below; anything
#                     but a supported zero initializer is an error.
#   copy              the stored value is a load (through value-preserving
#                     casts) of the result memref of a match that precedes the
#                     store, written into a different memref.  The sink is
#                     recorded on those matches as ``extra["output_sinks"]``.
#   local             a store into a function-local ``memref.alloc``/``alloca``
#                     whose every read is consumed only by index expressions,
#                     loop/branch control, other accepted local stores, or the
#                     traced term of a matched store.  A read that reaches a
#                     matched op as a load operand (not forwarded into its
#                     term) additionally requires every store into that local
#                     to be a constant or a plain copy, because the matched op
#                     only reads the local; it does not compute it.
#   unmatched         everything else: an error naming the function, the
#                     store, its loop nest, the term, and the ops considered.
#
# Accumulator initialization (``extra["accumulator_init"]`` on every match
# with a loop-carried operand):
#
#   "zero"     every other store into the accumulator is the constant 0, comes
#              before the match in program order, is under no guard the match
#              is not also under, and either (a) sits in a strict prefix of the
#              match's loop nest and writes the same element, or (b) is a
#              whole-buffer fill: below the loops it shares with the match,
#              each memref dimension is indexed by its own loop over [0, dim).
#   "carried"  no other store, and the accumulator is a function argument: the
#              initial value comes from the caller.
#   otherwise  "unsupported", with an error stating what was found.

# Value ops ``_trace_value`` traces through.  A value reaching a matched store
# only along these is part of that store's traced term.
_COVERAGE_TRACED_OPS = (
    set(_FP_BINOPS)
    | set(_MATH_UNARY)
    | _CAST_OPS
    | {"arith.cmpf", "arith.cmpi", "arith.select", "arith.negf"}
)
_COVERAGE_INDEX_OPS = {"affine.apply"}
_COVERAGE_CONTROL_OPS = {"scf.if", "affine.if", "scf.for", "affine.for"}
_LOCAL_ALLOC_OPS = {"memref.alloc", "memref.alloca"}
_AFFINE_CONST_RE = re.compile(r"^affine_map<\(\)(?:\[[^\]]*\])?\s*->\s*\((-?\d+)\)>$")
_COVERAGE_ACC_HINT = (
    "Supported: initialize the accumulator with literal 0 before the reduction "
    "loop, or accumulate directly into the output buffer."
)


def _render_term(term, depth: int = 0) -> str:
    """Compact source-like text of a traced term, for diagnostics."""
    if depth > 8:
        return "..."
    value = None if isinstance(term, WLoad) else _const_value(term)
    if value is not None:
        return repr(value)
    if isinstance(term, WLoad):
        if term.index_terms is not None:
            indices = [_render_term(t, depth + 1) for t in term.index_terms]
        else:
            indices = list(term.indices)
        return (
            f"{term.memref_name or '?'}"
            f"[{', '.join(_access_text(term.index_map, indices))}]"
        )
    if isinstance(term, WBinOp):
        return (
            f"{term.op}({_render_term(term.lhs, depth + 1)}, "
            f"{_render_term(term.rhs, depth + 1)})"
        )
    if isinstance(term, WUnary):
        return f"{term.fn}({_render_term(term.arg, depth + 1)})"
    if isinstance(term, WCmp):
        return (
            f"{term.pred}({_render_term(term.lhs, depth + 1)}, "
            f"{_render_term(term.rhs, depth + 1)})"
        )
    if isinstance(term, WSelect):
        return (
            f"select({_render_term(term.cond, depth + 1)}, "
            f"{_render_term(term.a, depth + 1)}, {_render_term(term.b, depth + 1)})"
        )
    if isinstance(term, WConst):
        return repr(term.value)
    if isinstance(term, WBlockArg):
        return term.ssa_name
    return "?"


def _access_text(index_map, rendered) -> list[str]:
    """Per-dimension access text: ``rendered`` operands substituted into the
    affine map results (``index_map`` None means the operands are the
    coordinates)."""
    if index_map is None:
        return list(rendered)
    results = _affine_map_results(index_map)
    if results is None:
        return [f"{index_map}({', '.join(rendered)})"]

    def substitute(match):
        position = int(match.group(1))
        if position >= len(rendered):
            return match.group(0)
        text = rendered[position]
        return text if re.fullmatch(r"[\w%#.-]+", text) else f"({text})"

    return [re.sub(r"\bd(\d+)\b", substitute, result) for result in results]


def _term_blockargs(term, out: set) -> set:
    if isinstance(term, WBlockArg):
        out.add(term.ssa_name)
    elif isinstance(term, WLoad):
        for t in term.index_terms or ():
            _term_blockargs(t, out)
    elif isinstance(term, (WBinOp, WCmp)):
        _term_blockargs(term.lhs, out)
        _term_blockargs(term.rhs, out)
    elif isinstance(term, WUnary):
        _term_blockargs(term.arg, out)
    elif isinstance(term, WSelect):
        for t in (term.cond, term.a, term.b):
            _term_blockargs(t, out)
    return out


def _term_loads(term, out: list) -> list:
    if isinstance(term, WLoad):
        out.append(term)
    elif isinstance(term, (WBinOp, WCmp)):
        _term_loads(term.lhs, out)
        _term_loads(term.rhs, out)
    elif isinstance(term, WUnary):
        _term_loads(term.arg, out)
    elif isinstance(term, WSelect):
        for t in (term.cond, term.a, term.b):
            _term_loads(t, out)
    return out


def _retained_loads(term, out=None) -> list:
    """`RetainedLoad` records of every load under ``term``, in term order,
    recursing through load addresses: ``src[index[i]]`` reads ``index`` as
    well as ``src`` (a dependency the semantic contract must keep)."""
    if out is None:
        out = []
    for load in _term_loads(term, []):
        index_terms = getattr(load, "index_terms", None)
        out.append(
            RetainedLoad(
                memref_name=load.memref_name,
                value_ref=load.value_ref,
                memref_type=load.memref_type,
                indices=tuple(str(i) for i in load.indices),
                index_map=load.index_map,
                coords=(
                    _coords_of_terms(index_terms, load.index_map)
                    if index_terms is not None
                    else None
                ),
            )
        )
        for index_term in index_terms or ():
            _retained_loads(index_term, out)
    return out


def _term_casts(term) -> list[str]:
    """Cast operations applied inside ``term`` (``arith.trunci``, ...), in
    term order, each once."""
    out: list[str] = []

    def visit(node):
        if isinstance(node, WUnary):
            name = node.fn.split("<", 1)[0]
            if name in _CAST_OPS and name not in out:
                out.append(name)
            visit(node.arg)
        elif isinstance(node, (WBinOp, WCmp)):
            visit(node.lhs)
            visit(node.rhs)
        elif isinstance(node, WSelect):
            for child in (node.cond, node.a, node.b):
                visit(child)

    visit(term)
    return out


def affine_form(term):
    """``term`` as an affine coordinate ``((atom, coefficient), ...), constant``
    over loop variables and load-free SSA atoms, or None when the coordinate
    depends on loaded data.

    Atoms are SSA names: loop induction variables and any load-free value
    the analysis does not expand (a product of two variables, a shift, a
    select).  Equal forms denote equal coordinates in every execution of the
    function; the converse is not claimed.
    """

    def norm(coefs, const):
        return (tuple(sorted((k, v) for k, v in coefs.items() if v)), int(const))

    def atom(t):
        name = getattr(t, "ssa_name", None)
        if name is None or _term_loads(t, []):
            return None
        return norm({name: 1}, 0)

    if isinstance(term, WConst):
        value = term.value
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        return norm({}, value)
    if isinstance(term, WBlockArg):
        return norm({term.ssa_name: 1}, 0)
    if isinstance(term, WLoad):
        return None
    if isinstance(term, WUnary):
        if term.fn == "neg":
            inner = affine_form(term.arg)
            if inner is None:
                return None
            return norm({k: -v for k, v in inner[0]}, -inner[1])
        return atom(term)
    if isinstance(term, WBinOp):
        lhs, rhs = affine_form(term.lhs), affine_form(term.rhs)
        if lhs is not None and rhs is not None:
            if term.op in ("add", "sub"):
                sign = 1 if term.op == "add" else -1
                coefs = dict(lhs[0])
                for k, v in rhs[0]:
                    coefs[k] = coefs.get(k, 0) + sign * v
                return norm(coefs, lhs[1] + sign * rhs[1])
            if term.op == "mul":
                if not lhs[0]:
                    return norm({k: v * lhs[1] for k, v in rhs[0]}, rhs[1] * lhs[1])
                if not rhs[0]:
                    return norm({k: v * rhs[1] for k, v in lhs[0]}, lhs[1] * rhs[1])
        return atom(term)
    return atom(term)


_ACCESS_MAP_RE = re.compile(r"affine_map<\(([^)]*)\)(\[[^\]]*\])?\s*->\s*\((.*)\)>")


def _form_scale(form, factor):
    return (tuple((k, v * factor) for k, v in form[0] if v * factor), form[1] * factor)


def _form_add(a, b, sign=1):
    coefs = dict(a[0])
    for k, v in b[0]:
        coefs[k] = coefs.get(k, 0) + sign * v
    return (tuple(sorted((k, v) for k, v in coefs.items() if v)), a[1] + sign * b[1])


def _eval_affine_expr(node, dims):
    """Affine-map result expression (Python AST over ``dN`` names) as a form."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return ((), int(node.value))
    if isinstance(node, ast.Name):
        return dims.get(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _eval_affine_expr(node.operand, dims)
        return None if inner is None else _form_scale(inner, -1)
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult)
    ):
        lhs = _eval_affine_expr(node.left, dims)
        rhs = _eval_affine_expr(node.right, dims)
        if lhs is None or rhs is None:
            return None
        if isinstance(node.op, ast.Mult):
            if not lhs[0]:
                return _form_scale(rhs, lhs[1])
            if not rhs[0]:
                return _form_scale(lhs, rhs[1])
            return None
        return _form_add(lhs, rhs, 1 if isinstance(node.op, ast.Add) else -1)
    return None  # floordiv, ceildiv, mod, symbols: not an affine coordinate


def coordinate_forms(*coords):
    """Resolved coordinate forms for a hand-assembled match: each entry is a
    variable name (``"i"``), an integer, or ``(name, offset)``.  Production
    matches get their forms from the IR (`affine_form`); harnesses and tests
    that build `MatchedOp`s by hand state them with this helper."""
    out = []
    for coord in coords:
        if isinstance(coord, bool):
            raise TypeError("coordinate must be a name, an int or (name, offset)")
        if isinstance(coord, int):
            out.append(((), coord))
        elif isinstance(coord, str):
            out.append((((coord, 1),), 0))
        else:
            name, offset = coord
            out.append((((name, 1),), int(offset)))
    return tuple(out)


def _coords_of_terms(index_terms, index_map=None):
    """Resolved coordinates of an access: the operands' forms, composed
    with the ``affine.load``/``affine.store`` map when there is one
    (``affine_map<(d0) -> (d0 + 48)>`` over ``[i]`` is the coordinate
    ``i + 48``).  None entries are coordinates that are not affine in
    load-free values."""
    if index_terms is None:
        return None
    operand_forms = tuple(affine_form(t) for t in index_terms)
    if index_map is None:
        return operand_forms
    found = _ACCESS_MAP_RE.fullmatch(str(index_map).strip())
    if not found or (found.group(2) and found.group(2).strip("[] ")):
        return None
    dim_names = [d.strip() for d in found.group(1).split(",") if d.strip()]
    if len(dim_names) != len(operand_forms):
        return None
    dims = dict(zip(dim_names, operand_forms))
    results = _split_affine_results(found.group(3))
    coords = []
    for text in results:
        if re.search(r"floordiv|ceildiv|\bmod\b", text):
            coords.append(None)
            continue
        try:
            tree = ast.parse(text.strip(), mode="eval")
        except SyntaxError:
            coords.append(None)
            continue
        coords.append(_eval_affine_expr(tree.body, dims))
    return tuple(coords)


def _split_affine_results(text: str) -> list[str]:
    parts, depth, current = [], 0, []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current or parts:
        parts.append("".join(current))
    return [p for p in parts if p.strip()]


def _form_mentions(form, names) -> bool:
    return any(atom in names for atom, _coef in form[0])


def _render_form(form) -> str:
    parts = [(f"{coef}*{atom}" if coef != 1 else atom) for atom, coef in form[0]]
    if form[1] or not parts:
        parts.append(str(form[1]))
    return " + ".join(parts).replace("+ -", "- ")


def output_map_problem(match) -> str | None:
    """Why the match's result coordinates are not the identity image of its
    operands' coordinates, or None when they are.

    A contraction backend writes the result of the row it reads: element
    ``y[f]`` receives the products of ``W[f, k]``.  That is sound only when
    every non-reduction coordinate of every input operand (``f``, resolved
    with `affine_form`) is literally one of the store's coordinates.  A
    store coordinate that no operand carries (a loop variable the operands
    do not read, a constant element) is permitted; a reversed, shifted or
    broadcast result (``out[M - 1 - row]``, ``out[row + 1]``, ``out[0]`` for
    every row) is not, and no backend may silently write the identity
    instead.  Unresolved or data-dependent coordinates are reported.
    """
    store = match.extra.get("store_coords")
    if store is None:
        return "the store's coordinates were not resolved by the matcher"
    # A scalar accumulator (``acc[] += ...``) reaches the output through its
    # copies: the result coordinates are the sinks' (`output_sink_coords`,
    # stamped by the coverage analysis); every sink must be the identity.
    sink_coords = match.extra.get("output_sink_coords")
    if not store and sink_coords:
        problems = [
            _output_map_problem_for(match, coords, f"sink {memref}")
            for (memref, _text), coords in zip(
                match.extra.get("output_sinks", ()), sink_coords
            )
        ]
        return next((p for p in problems if p is not None), None)
    if not store and match.extra.get("output_sinks"):
        return "the result sinks' coordinates were not resolved by the matcher"
    if store and sink_coords:
        for (memref, _text), coords in zip(
            match.extra.get("output_sinks", ()), sink_coords
        ):
            if coords is None or tuple(coords) != tuple(store):
                return (
                    f"the result is copied into {memref!r} at a different element "
                    "than it is produced at; copies are realized element for element"
                )
    return _output_map_problem_for(match, store, "the store")


def _output_map_problem_for(match, store, what) -> str | None:
    if store is None:
        return f"the coordinates of {what} were not resolved by the matcher"
    if any(c is None for c in store):
        return f"a coordinate of {what} depends on loaded data"
    roles = match.extra.get("axis_roles") or {}
    reduction = set(roles.get("reduction") or ())
    store_set = set(store)
    for operand in match.operands:
        if operand.is_loop_carried or operand.memref_name is None:
            continue
        if operand.coords is None:
            return (
                f"the coordinates of operand {operand.memref_name!r} were not resolved"
            )
        for coord in operand.coords:
            if coord is None:
                return (
                    f"operand {operand.memref_name!r} has a data-dependent coordinate"
                )
            if _form_mentions(coord, reduction):
                continue
            if coord not in store_set:
                return (
                    f"operand {operand.memref_name!r} is read at coordinate "
                    f"{_render_form(coord)} but the result is stored at "
                    f"[{', '.join(_render_form(c) for c in store)}]; the output "
                    "map is not the identity, and the target lowering writes each "
                    "result at the coordinate it reads"
                )
    return None


def _render_guards(guards) -> str:
    if not guards:
        return "none"
    return ", ".join(
        f"{frame[0]}({frame[1]})" if len(frame) > 1 else str(frame[0])
        for frame in guards
    )


def _render_loops(loops) -> str:
    parts = []
    for iv, lower, upper, step in loops:
        lo = _AFFINE_CONST_RE.match(str(lower).strip())
        hi = _AFFINE_CONST_RE.match(str(upper).strip())
        if lo and hi:
            text = f"{iv} in [{lo.group(1)}, {hi.group(1)})"
            parts.append(text if step == 1 else f"{text} step {step}")
        else:
            parts.append(f"{iv} in [{lower}, {upper})")
    return "[" + ", ".join(parts) + "]" if parts else "[] (no enclosing loop)"


def _affine_map_results(text: str) -> list[str] | None:
    m = _AFFINE_MAP_RE.match(text.strip())
    if m is None or "[" in text.split("->")[0]:
        return None
    results, depth, start = [], 0, 0
    body = m.group("results")
    for i, ch in enumerate(body):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            results.append(body[start:i].strip())
            start = i + 1
    results.append(body[start:].strip())
    return [r for r in results if r]


class _FunctionCoverage:
    """Classify every store of one function; see the comment block above."""

    def __init__(
        self,
        target,
        func,
        func_name,
        sites,
        site_matches,
        defining_map,
        value_refs,
        *,
        state,
        scope,
    ):
        self.target = target
        self.func_name = func_name
        self.work_root = _parse_work_id(func_name)[0]
        self.sites = sites
        self.site_matches = site_matches
        self.defining_map = defining_map
        self.value_refs = value_refs
        self.state = state
        self.scope = scope
        self.arg_refs = {
            self._ref(argument)
            for argument in func.arguments
            if _is_memref_value(argument)
        }
        self.stores = [
            index
            for index, (op, _loops, _guards) in enumerate(sites)
            if op.operation.name in _STORE_OPS and len(op.operands) > 1
        ]
        self.loads_by_ref: dict = {}
        self.uses: dict[str, list[tuple[int, int]]] = {}
        for index, (op, _loops, _guards) in enumerate(sites):
            if op.operation.name in _LOAD_OPS and op.operands and op.results:
                self.loads_by_ref.setdefault(self._ref(op.operands[0]), []).append(
                    index
                )
            for position, operand in enumerate(op.operands):
                self.uses.setdefault(_value_name(operand, state), []).append(
                    (index, position)
                )
        self._terms: dict[int, Any] = {}
        self._index_terms: dict[int, tuple] = {}
        # store index -> (source type, destination type) of the one integer
        # narrowing `_copy_source` looked through for that store
        self._copy_narrowing: dict[int, tuple[str, str]] = {}

    # -- small accessors ------------------------------------------------ #

    def _ref(self, value):
        return self.value_refs.get(value, value)

    def _op(self, index):
        return self.sites[index][0]

    def _dest(self, index):
        return self._ref(self._op(index).operands[1])

    def _dest_name(self, index):
        op = self._op(index)
        return _attr_str(op.attributes["to"]) if "to" in op.attributes else None

    def _handle(self, index):
        op = self._op(index)
        return f"{op.operation.name}@{_value_name(op.operands[0], self.state)}"

    def _term(self, index):
        if index not in self._terms:
            self._terms[index] = _trace_value(
                _value_name(self._op(index).operands[0], self.state),
                self.defining_map,
                self.value_refs,
                state=self.state,
                scope=self.scope,
            )
        return self._terms[index]

    def _store_index_terms(self, index):
        if index not in self._index_terms:
            self._index_terms[index] = tuple(
                _trace_value(
                    _value_name(operand, self.state),
                    self.defining_map,
                    self.value_refs,
                    state=self.state,
                    scope=self.scope,
                )
                for operand in list(self._op(index).operands)[2:]
            )
        return self._index_terms[index]

    def _store_map(self, index):
        op = self._op(index)
        return str(op.attributes["map"]) if "map" in op.attributes else None

    def _access_key(self, index) -> tuple:
        rendered = [_render_term(t) for t in self._store_index_terms(index)]
        return tuple(_access_text(self._store_map(index), rendered))

    def _index_text(self, index) -> str:
        return ", ".join(self._access_key(index))

    def _reads_own_element(self, index) -> bool:
        """The stored term reads the element being stored (``y[i] += ...``),
        and some enclosing loop does not move that element: the value is
        carried across that loop even when the matched op does not declare
        an accumulator."""
        loops = self.sites[index][1]
        index_terms = self._store_index_terms(index)
        used = set()
        for t in index_terms:
            _term_blockargs(t, used)
        if all(loop[0] in used for loop in loops):
            return False
        dest = self._dest(index)
        return any(
            load.value_ref == dest
            and self._same_access(
                load.index_map, load.index_terms, self._store_map(index), index_terms
            )
            for load in _term_loads(self._term(index), [])
            if load.index_terms is not None
        )

    @staticmethod
    def _same_access(lhs_map, lhs_terms, rhs_map, rhs_terms) -> bool:
        if lhs_map == rhs_map and _terms_eq(lhs_terms, rhs_terms):
            return True
        # Equal text is trusted only for load-free (pure index) coordinates.
        if any(_term_loads(t, []) for t in (*lhs_terms, *rhs_terms)):
            return False
        lhs = _access_text(lhs_map, [_render_term(t) for t in lhs_terms])
        rhs = _access_text(rhs_map, [_render_term(t) for t in rhs_terms])
        return lhs == rhs

    def _copy_source(self, index):
        """The load whose value the store writes, through casts; else None.

        Exact casts are transparent.  One integer narrowing (``trunci`` or a
        narrowing ``index_cast``) of the loaded value is peeled as well and
        remembered in ``_copy_narrowing``; `_copy_contract_problem` then has
        to prove from the producer's accumulated range that the truncation
        cannot change the value, otherwise the store is not a copy.
        """
        name = _value_name(self._op(index).operands[0], self.state)
        src = self.defining_map.get(name)
        narrowing = None
        while (
            src is not None
            and src.operation.name in _CAST_OPS
            and len(src.operands) == 1
            and src.results
        ):
            if _cast_exact_op(src):
                pass
            elif (
                narrowing is None
                and src.operation.name in ("arith.trunci", "arith.index_cast")
                and _int_bits(_scalar_type_text(src.results[0].type)) is not None
                and _int_bits(_scalar_type_text(src.operands[0].type)) is not None
            ):
                narrowing = (
                    _scalar_type_text(src.operands[0].type),
                    _scalar_type_text(src.results[0].type),
                )
            else:
                return None
            src = self.defining_map.get(_value_name(src.operands[0], self.state))
        if src is not None and src.operation.name in _LOAD_OPS and src.operands:
            if narrowing is not None:
                self._copy_narrowing[index] = narrowing
            return src
        return None

    def _accumulator_load(self, term, dest, dest_name):
        """The operand of ``term`` that reloads the accumulator ``dest``,
        and the remaining operand(s); None when ``term`` is not an
        accumulation of a recognized shape."""

        def is_acc(t):
            return isinstance(t, WLoad) and (
                (t.value_ref is not None and t.value_ref == dest)
                or (t.value_ref is None and t.memref_name == dest_name)
            )

        if isinstance(term, WBinOp) and term.op in ("add", "sub", "min", "max"):
            if is_acc(term.lhs):
                return term.op, term.rhs
            if is_acc(term.rhs) and term.op != "sub":
                return term.op, term.lhs
        if isinstance(term, WSelect):
            if is_acc(term.a):
                return "select", term.b
            if is_acc(term.b):
                return "select", term.a
        return None

    def _carried_interval(self, producer):
        """Sound bounds on the value held by the accumulator that the matched
        store ``producer`` updates, over the whole reduction; None when no
        bound can be established.

        ``init + trips * [increment range]`` for ``acc += x``/``acc -= x``;
        the hull of the initializer and the other operand for ``min``,
        ``max`` and ``select`` accumulators.  The bound must also fit the
        accumulator's own element type, otherwise the source wraps there
        and the bound says nothing about the stored value.
        """
        dest = self._dest(producer)
        dest_name = self._dest_name(producer)
        loop_bounds = getattr(self.state, "loop_bounds", {})
        shape = self._accumulator_load(self._term(producer), dest, dest_name)
        if shape is None:
            return None
        kind, other = shape
        other_interval = _interval(other, loop_bounds)
        if other_interval is None:
            return None
        init = None
        for index in self.stores:
            if (
                index == producer
                or self._dest(index) != dest
                or index in self.site_matches
            ):
                continue
            interval = _interval(self._term(index), loop_bounds)
            if interval is None:
                return None
            init = (
                interval
                if init is None
                else (min(init[0], interval[0]), max(init[1], interval[1]))
            )
        if init is None:
            return None
        if kind in ("add", "sub"):
            depth = None
            for match in self.site_matches[producer]:
                for operand in match.operands:
                    if operand.is_loop_carried:
                        depth = _accumulator_scope_depth(operand)
            loops = self.sites[producer][1]
            if depth is None:
                return None
            trips = 1
            for loop in loops[min(depth, len(loops)) :]:
                lo = _AFFINE_CONST_RE.match(str(loop[1]).strip())
                hi = _AFFINE_CONST_RE.match(str(loop[2]).strip())
                if not lo or not hi or not loop[3]:
                    return None
                trips *= max(0, -(-(int(hi.group(1)) - int(lo.group(1))) // loop[3]))
            lo, hi = other_interval
            if kind == "sub":
                lo, hi = -hi, -lo
            total = (init[0] + min(0, trips * lo), init[1] + max(0, trips * hi))
        else:
            total = (min(init[0], other_interval[0]), max(init[1], other_interval[1]))
        try:
            width = _int_bits(str(self._op(producer).operands[1].type.element_type))
        except AttributeError:
            return None
        if width is None or not (
            _int_fits(total[0], width) and _int_fits(total[1], width)
        ):
            return None
        return total

    def _copy_contract_problem(self, index, source, producers) -> str | None:
        """Why the store at ``index`` is not an output copy a backend may
        realize unconditionally, or None when it is.

        A backend writes the produced value to the sink once per produced
        element, unconditionally, in the result's element type.  The copy
        therefore has to (a) run under exactly the guards its producers run
        under, (b) sit in exactly the producers' result loops (the loops
        enclosing the accumulator's definition; for a non-accumulating
        producer, all of its loops), so it executes once per produced
        element, and (c) write a buffer of the same element type (a copy
        through an exact widening cast still changes the stored type).  A
        copy through an integer truncation is accepted only when the
        producer's accumulated range provably fits the destination type.
        """

        def source_name(load_op):
            return (
                _attr_str(load_op.attributes["from"])
                if "from" in load_op.attributes
                else "?"
            )

        copy_guards = tuple(self.sites[index][2])
        copy_loops = [loop[0] for loop in self.sites[index][1]]
        for j in producers:
            producer_guards = tuple(self.sites[j][2])
            if copy_guards != producer_guards:
                return (
                    "is a copy of a matched result under a different guard than "
                    f"its producer (copy guards {_render_guards(copy_guards)}, producer "
                    f"guards {_render_guards(producer_guards)}); a conditional output "
                    "copy would be executed unconditionally by the device"
                )
            producer_loops = [loop[0] for loop in self.sites[j][1]]
            result_loops = producer_loops
            for match in self.site_matches[j]:
                accumulators = [o for o in match.operands if o.is_loop_carried]
                if accumulators:
                    depth = _accumulator_scope_depth(accumulators[0])
                    if depth is None:
                        return (
                            "copies an accumulator whose definition scope is not "
                            "retained, so the copy cannot be proven to run once per "
                            "reduced element"
                        )
                    result_loops = producer_loops[: min(depth, len(producer_loops))]
            if copy_loops != result_loops:
                return (
                    f"is a copy of a matched result in loop nest "
                    f"{_render_loops(self.sites[index][1])}, but the result is "
                    f"produced once per iteration of {result_loops or '[]'}; the "
                    "copy must execute exactly once per produced element"
                )
        source_terms = tuple(
            _trace_value(
                _value_name(operand, self.state),
                self.defining_map,
                self.value_refs,
                state=self.state,
                scope=self.scope,
            )
            for operand in list(source.operands)[1:]
        )
        source_map = (
            str(source.attributes["map"]) if "map" in source.attributes else None
        )
        for j in producers:
            if source_map != self._store_map(j) or not _terms_eq(
                source_terms, self._store_index_terms(j)
            ):
                return (
                    "copies element "
                    f"[{', '.join(_render_term(t) for t in source_terms)}] of the "
                    f"result, but its producer stores element "
                    f"[{', '.join(_render_term(t) for t in self._store_index_terms(j))}]"
                )
            # Destination relation (C-24): a backend realizes a copy of an
            # array result as the same element of another buffer.  The copy's
            # own coordinates must therefore equal the producer's (a scalar
            # producer has none; its sinks are checked by `output_map_problem`).
            produced = _coords_of_terms(self._store_index_terms(j), self._store_map(j))
            written = _coords_of_terms(
                self._store_index_terms(index), self._store_map(index)
            )
            if produced and (
                written is None
                or any(c is None for c in written)
                or any(c is None for c in produced)
                or written != produced
            ):
                return (
                    "copies the result produced at element "
                    f"[{', '.join(_render_form(c) if c else '?' for c in produced)}] to "
                    f"element [{', '.join(_render_form(c) if c else '?' for c in (written or ()))}]"
                    "; a copy is realized as the same element of another buffer, "
                    "and no target lowering realizes a permuted or shifted copy"
                )
        try:
            source_type = str(source.operands[0].type.element_type)
            dest_type = str(self._op(index).operands[1].type.element_type)
        except AttributeError:
            return "copies a value whose buffer element types cannot be read"
        narrowing = self._copy_narrowing.get(index)
        if narrowing is not None:
            bits = _int_bits(narrowing[1])
            for j in producers:
                interval = self._carried_interval(j)
                if interval is None:
                    return (
                        f"copies the {source_type} result '{source_name(source)}' into a "
                        f"{narrowing[1]} buffer through a truncation, and the accumulated "
                        "value has no provable range, so the truncation may change it"
                    )
                if not (_int_fits(interval[0], bits) and _int_fits(interval[1], bits)):
                    return (
                        f"copies the {source_type} result '{source_name(source)}' into a "
                        f"{narrowing[1]} buffer through a truncation; the accumulated value "
                        f"ranges over [{interval[0]}, {interval[1]}], which {narrowing[1]} "
                        "cannot hold, so the truncation changes it"
                    )
            # The value provably fits the destination: the device writing the
            # result into the destination buffer stores exactly that value.
            source_type = dest_type
        if source_type != dest_type:
            return (
                f"copies a {source_type} result into a {dest_type} buffer; the "
                "device writes the result in its own element type, so the "
                "conversion is not performed"
            )
        return None

    def _record_fill(self, index, coverage, loop_names) -> None:
        """Retain a ``linalg.fill`` initializer as a whole-buffer effect:
        its destination, value, loop nest and guards.  The matcher does not
        classify fills (they initialize buffers a function declares); a
        program that executes the function verbatim states them in its
        contract, since a returned buffer's untouched elements hold them."""
        op = self._op(index)
        value_ssa = _value_name(op.operands[0], self.state)
        buffer_ssa = _value_name(op.operands[1], self.state)
        dest_ref = self._ref(op.operands[1])
        alloc = self.defining_map.get(buffer_ssa)
        name = None
        if alloc is not None and "name" in alloc.attributes:
            name = _attr_str(alloc.attributes["name"])
        term = _trace_value(
            value_ssa,
            self.defining_map,
            self.value_refs,
            state=self.state,
            scope=self.scope,
        )
        guard_terms = self._guard_terms(index)
        coverage.stores.append(
            StoreCoverage(
                func_name=self.func_name,
                store=f"linalg.fill@{buffer_ssa}",
                memref_name=name,
                enclosing_loops=tuple(self.sites[index][1]),
                category=STORE_FILL,
                detail=f"initializer {_render_term(term)}",
                local=dest_ref not in self.arg_refs,
                value_ref=dest_ref if isinstance(dest_ref, IRValueRef) else None,
                memref_type=str(op.operands[1].type),
                indices=(),
                coords=None,
                guards=tuple(
                    f"{kind}({_render_term(cond)})" if cond is not None else kind
                    for kind, cond in guard_terms
                ),
                loads=tuple(_retained_loads(term)),
                guard_loads=tuple(
                    load
                    for _kind, cond in guard_terms
                    if cond is not None
                    for load in _retained_loads(cond)
                ),
                reads_destination=False,
                conversions=tuple(_term_casts(term)),
                term=_render_term(term),
                constant=_const_value(term) is not None,
                loop_names=loop_names,
                whole=True,
            )
        )

    def _guard_terms(self, index) -> list:
        """``(kind, traced condition or None)`` for every guard frame
        enclosing the store, outermost first."""
        out = []
        for frame in self.sites[index][2]:
            kind = frame[0]
            if kind in ("if", "not") and len(frame) > 1:
                out.append(
                    (
                        kind,
                        _trace_value(
                            frame[1],
                            self.defining_map,
                            self.value_refs,
                            state=self.state,
                            scope=self.scope,
                        ),
                    )
                )
            else:
                out.append((str(kind), None))
        return out

    def _is_local(self, index) -> bool:
        owner = getattr(self._op(index).operands[1], "owner", None)
        name = getattr(getattr(owner, "operation", owner), "name", None)
        return name in _LOCAL_ALLOC_OPS

    def _target_ops(self) -> str:
        names = sorted(
            {
                op.name
                for unit in self.target._walk()
                for op in unit.ops.values()
                if op.matchable
            }
        )
        return ", ".join(names) if names else "(none)"

    # -- classification ------------------------------------------------- #

    def run(self, coverage: MatchCoverage, problems: dict) -> None:
        result_sites: dict = {}
        acc_sites: dict = {}
        for index, matches in self.site_matches.items():
            dest = self._dest(index)
            result_sites.setdefault(dest, []).append(index)
            self_carried = None
            for match in matches:
                carried = any(operand.is_loop_carried for operand in match.operands)
                if not carried:
                    if self_carried is None:
                        self_carried = self._reads_own_element(index)
                    carried = self_carried
                if carried:
                    acc_sites.setdefault(dest, []).append((index, match))

        category: dict[int, str] = {}
        detail: dict[int, str] = {}
        copy_producers: dict[int, list[int]] = {}
        for index in self.stores:
            dest = self._dest(index)
            if index in self.site_matches:
                category[index] = STORE_MATCHED
                detail[index] = ", ".join(
                    m.target_op_name for m in self.site_matches[index]
                )
                continue
            if dest in acc_sites:
                category[index] = STORE_ACCUMULATOR_INIT
                continue
            source = self._copy_source(index)
            if source is not None:
                source_ref = self._ref(source.operands[0])
                producers = [j for j in result_sites.get(source_ref, ()) if j < index]
                if source_ref != dest and producers:
                    problem = self._copy_contract_problem(index, source, producers)
                    if problem is not None:
                        category[index] = STORE_UNMATCHED
                        detail[index] = problem
                        continue
                    category[index] = STORE_COPY
                    copy_producers[index] = producers
                    sink = (self._dest_name(index), self._index_text(index))
                    sink_coords = _coords_of_terms(
                        self._store_index_terms(index), self._store_map(index)
                    )
                    sink_ref = self._ref(self._op(index).operands[1])
                    for j in producers:
                        for match in self.site_matches[j]:
                            sinks = match.extra.setdefault("output_sinks", [])
                            if sink not in sinks:
                                sinks.append(sink)
                                # Resolved sink coordinates and destination
                                # identities, one entry per sink
                                # (`output_map_problem`, `external_effects`).
                                match.extra.setdefault("output_sink_coords", []).append(
                                    sink_coords
                                )
                                match.extra.setdefault("output_sink_refs", []).append(
                                    (
                                        (
                                            sink_ref
                                            if isinstance(sink_ref, IRValueRef)
                                            else None
                                        ),
                                        self._is_local(index),
                                    )
                                )
                    detail[index] = f"copy of {_render_term(self._term(index))}"
                    continue
            if self._is_local(index):
                category[index] = STORE_LOCAL
                continue
            category[index] = STORE_UNMATCHED

        self._settle_locals(category, detail)
        self._check_result_aliases(copy_producers, problems)

        for dest, entries in acc_sites.items():
            for index, match in entries:
                init, found = self._accumulator_init(dest, index, category)
                match.extra["accumulator_init"] = init
                if init == "carried":
                    # ``y[i] += ...``: the matched store itself writes the
                    # output buffer, so it is the result's sink.
                    sink = (self._dest_name(index), self._index_text(index))
                    sinks = match.extra.setdefault("output_sinks", [])
                    if sink not in sinks:
                        sinks.append(sink)
                        match.extra.setdefault("output_sink_coords", []).append(
                            _coords_of_terms(
                                self._store_index_terms(index), self._store_map(index)
                            )
                        )
                if found:
                    loops = self.sites[index][1]
                    problems.setdefault(
                        (self.work_root, index, "accumulator"),
                        [
                            f"function '{self.func_name}': the accumulator "
                            f"'{match.result_memref_name}' of "
                            f"{match.target_op_name} (store {self._handle(index)}, "
                            f"loop nest {_render_loops(loops)}) is not supported: "
                            f"{'; '.join(found)}. {_COVERAGE_ACC_HINT}",
                            [],
                        ],
                    )[1].append(self.func_name)

        loop_names = tuple(
            sorted((getattr(self.state, "loop_names", {}) or {}).items())
        )
        fills = {
            index
            for index, (op, _loops, _guards) in enumerate(self.sites)
            if op.operation.name == "linalg.fill" and len(op.operands) >= 2
        }
        # One traversal in site (program) order: a fill that reads a value an
        # earlier store defined is retained after that store.
        for index in sorted(fills | set(self.stores)):
            if index in fills:
                self._record_fill(index, coverage, loop_names)
                continue
            cat = category[index]
            dest_ref = self._ref(self._op(index).operands[1])
            local = self._is_local(index)
            for match in self.site_matches.get(index, ()):
                match.extra["result_is_local"] = local
            op = self._op(index)
            term = self._term(index)
            guard_terms = self._guard_terms(index)
            coverage.stores.append(
                StoreCoverage(
                    func_name=self.func_name,
                    store=self._handle(index),
                    memref_name=self._dest_name(index),
                    enclosing_loops=tuple(self.sites[index][1]),
                    category=cat,
                    detail=detail.get(index, ""),
                    local=local,
                    value_ref=dest_ref if isinstance(dest_ref, IRValueRef) else None,
                    memref_type=str(op.operands[1].type),
                    indices=tuple(self._access_key(index)),
                    coords=_coords_of_terms(
                        self._store_index_terms(index), self._store_map(index)
                    ),
                    guards=tuple(
                        f"{kind}({_render_term(cond)})" if cond is not None else kind
                        for kind, cond in guard_terms
                    ),
                    loads=tuple(_retained_loads(term)),
                    guard_loads=tuple(
                        load
                        for _kind, cond in guard_terms
                        if cond is not None
                        for load in _retained_loads(cond)
                    ),
                    index_loads=tuple(
                        load
                        for index_term in self._store_index_terms(index)
                        for load in _retained_loads(index_term)
                    ),
                    reads_destination=self._reads_own_element(index)
                    or any(
                        load.value_ref == dest_ref
                        and self._same_access(
                            load.index_map,
                            load.index_terms,
                            self._store_map(index),
                            self._store_index_terms(index),
                        )
                        for load in _term_loads(term, [])
                        if load.index_terms is not None
                    ),
                    conversions=tuple(_term_casts(term)),
                    term=_render_term(term),
                    constant=_const_value(term) is not None,
                    loop_names=loop_names,
                )
            )
            if cat == STORE_UNMATCHED:
                problems.setdefault(
                    (self.work_root, index, "unmatched"),
                    [self._unmatched_message(index, detail.get(index)), []],
                )[1].append(self.func_name)

    def _check_result_aliases(self, copy_producers, problems) -> None:
        """Reject a match whose operand reads the buffer its own result is
        written to, at another element (``y[i] = sum_k W[i, k] * y[k]``).

        In source order a later iteration reads elements an earlier one
        already overwrote; every PIM program reads its operands before it
        writes its result, so lowering it would silently compute
        ``W @ y_initial`` instead.  Reading the element being written
        (``y[i] += ...``, ``a[i] = a[i] + b[i]``) is not an alias.
        """
        result_stores: dict[int, list[int]] = {
            index: [index] for index in self.site_matches
        }
        for copy, producers in copy_producers.items():
            for producer in producers:
                result_stores[producer].append(copy)
        for index, stores in result_stores.items():
            for load in _term_loads(self._term(index), []):
                if load.index_terms is None or load.value_ref is None:
                    continue
                for store in stores:
                    if load.value_ref != self._dest(store) or self._same_access(
                        load.index_map,
                        load.index_terms,
                        self._store_map(store),
                        self._store_index_terms(store),
                    ):
                        continue
                    names = ", ".join(
                        m.target_op_name for m in self.site_matches[index]
                    )
                    read = _render_term(load)
                    written = f"{self._dest_name(store)}[{self._index_text(store)}]"
                    problems.setdefault(
                        (self.work_root, index, "alias", store),
                        [
                            f"function '{self.func_name}': operand {read} of {names} "
                            f"(store {self._handle(index)}, loop nest "
                            f"{_render_loops(self.sites[index][1])}) aliases its "
                            f"result {written}: the source reads elements of "
                            f"'{load.memref_name}' that earlier iterations overwrite, "
                            "but the target program reads every operand before it "
                            "writes the result. Write the result to a separate buffer.",
                            [],
                        ],
                    )[1].append(self.func_name)

    def _unmatched_message(self, index, reason) -> str:
        op = self._op(index)
        term = self._term(index)
        stored = self.defining_map.get(_value_name(op.operands[0], self.state))
        root = stored.operation.name if stored is not None else "block argument"
        loops = _render_loops(self.sites[index][1])
        guards = self.sites[index][2]
        where = (
            f"function '{self.func_name}', store {self._handle(index)} to "
            f"'{self._dest_name(index)}[{self._index_text(index)}]', loop nest {loops}"
            + (" under a guard" if guards else "")
        )
        if reason and reason.startswith(("is a copy", "copies")):
            what = f"{reason} (stored value {_render_term(term)}, {root})"
        elif reason:
            what = f"stores {_render_term(term)} ({root}) into a local buffer whose value {reason}"
        elif isinstance(term, WLoad):
            what = (
                f"copies {_render_term(term)} ({root}), but "
                f"'{term.memref_name}' is not the result of a preceding matched op"
            )
        elif _const_value(term) is not None:
            what = (
                f"stores the constant {_render_term(term)}, which is not the "
                "initializer of any matched accumulator"
            )
        else:
            what = f"computes {_render_term(term)} (root op {root}), which matches no target op"
            precision = arithmetic_precision_problem(term)
            if precision is not None:
                what += f"; {precision}"
        return (
            f"{where}: {what}. Target ops considered: {self._target_ops()}. "
            "Compiling would silently drop this effect; express it with the "
            "target's ops or extend the target description."
        )

    def _settle_locals(self, category, detail) -> None:
        """Greatest fixpoint over local stores; failures become unmatched."""
        accepted = {i for i in self.stores if category[i] == STORE_LOCAL}
        reasons: dict[int, str] = {}
        changed = True
        while changed:
            changed = False
            for index in sorted(accepted):
                reason = self._local_problem(index, category, accepted)
                if reason is not None:
                    accepted.discard(index)
                    reasons[index] = reason
                    changed = True
        for index, reason in reasons.items():
            category[index] = STORE_UNMATCHED
            detail[index] = reason
        for index in accepted:
            detail[index] = "local value consumed by matched ops / index expressions"

    def _local_problem(self, index, category, accepted) -> str | None:
        dest = self._dest(index)
        memref_name = _value_name(self._op(index).operands[1], self.state)
        for user, position in self.uses.get(memref_name, ()):
            user_name = self._op(user).operation.name
            if user_name in _LOAD_OPS and position == 0:
                continue
            if user_name in _STORE_OPS and position == 1:
                continue
            if user_name == "memref.dealloc":
                continue
            return f"escapes to '{user_name}'"
        writers = [i for i in self.stores if self._dest(i) == dest]
        computed_writers = [
            i
            for i in writers
            if i not in self.site_matches
            and _const_value(self._term(i)) is None
            and self._copy_source(i) is None
        ]
        for load in self.loads_by_ref.get(dest, ()):
            load_op = self._op(load)
            result = _value_name(load_op.results[0], self.state)
            forwarded = result in self.scope.forward
            stack = [(result, False)]
            seen = set()
            while stack:
                value, index_only = stack.pop()
                if (value, index_only) in seen:
                    continue
                seen.add((value, index_only))
                for user, position in self.uses.get(value, ()):
                    user_op = self._op(user)
                    user_name = user_op.operation.name
                    if user_name in _LOAD_OPS and position >= 1:
                        continue
                    if user_name in _STORE_OPS and position >= 2:
                        continue
                    if user_name in _COVERAGE_CONTROL_OPS:
                        continue
                    if user_name in _STORE_OPS and position == 0:
                        if index_only:
                            return "reaches a stored value through index arithmetic"
                        if category[user] == STORE_MATCHED:
                            if not forwarded and computed_writers:
                                return (
                                    "is read as an operand by matched op(s) "
                                    + ", ".join(
                                        m.target_op_name
                                        for m in self.site_matches[user]
                                    )
                                    + ", but is computed by an expression no target "
                                    "op implements"
                                )
                            continue
                        if user in accepted:
                            continue
                        return (
                            f"flows into the {category[user]} store to "
                            f"'{self._dest_name(user)}'"
                        )
                    if user_name in _COVERAGE_TRACED_OPS:
                        for r in user_op.results:
                            stack.append((_value_name(r, self.state), index_only))
                        continue
                    if user_name in _COVERAGE_INDEX_OPS:
                        for r in user_op.results:
                            stack.append((_value_name(r, self.state), True))
                        continue
                    return f"is read by '{user_name}', which no target op implements"
        return None

    # -- accumulator initialization ------------------------------------ #

    def _accumulator_init(self, dest, index, category):
        others = [i for i in self.stores if i != index and self._dest(i) == dest]
        if not others:
            if dest in self.arg_refs:
                return "carried", []
            return "unsupported", ["it is a local buffer that is never initialized"]
        _op, match_loops, match_guards = self.sites[index]
        match_ivs = [loop[0] for loop in match_loops]
        found = []
        for other in others:
            other_loops, other_guards = self.sites[other][1], self.sites[other][2]
            other_ivs = [loop[0] for loop in other_loops]
            at = (
                f"store {self._handle(other)} at loop nest {_render_loops(other_loops)}"
            )
            if other in self.site_matches:
                found.append(
                    f"it is also written by matched op(s) "
                    f"{', '.join(m.target_op_name for m in self.site_matches[other])} "
                    f"({at}): multiple writers"
                )
                continue
            term = self._term(other)
            value = _const_value(term)
            if value is None:
                found.append(
                    f"its initializer {at} is the non-constant value {_render_term(term)}"
                )
                continue
            if value != 0:
                found.append(f"its initializer {at} is the nonzero constant {value!r}")
                continue
            if other > index:
                found.append(f"the zero store {at} comes after the reduction")
                continue
            if tuple(other_guards) != tuple(match_guards[: len(other_guards)]):
                found.append(
                    f"the zero store {at} is under a guard the reduction is not under"
                )
                continue
            prefix = 0
            while (
                prefix < min(len(other_ivs), len(match_ivs))
                and other_ivs[prefix] == match_ivs[prefix]
            ):
                prefix += 1
            if prefix == len(other_ivs) and prefix < len(match_ivs):
                if self._same_element(other, index):
                    continue
                found.append(
                    f"the zero store {at} writes "
                    f"'{self._dest_name(other)}[{self._index_text(other)}]', not "
                    f"the reduced element '[{self._index_text(index)}]'"
                )
                continue
            if prefix < len(other_ivs) and self._whole_buffer_fill(other, prefix):
                continue
            found.append(
                f"the zero store {at} is inside the reduction loop nest "
                f"{_render_loops(match_loops)} (not before it) and does not fill the "
                "whole buffer"
            )
        return ("zero", []) if not found else ("unsupported", found)

    def _same_element(self, lhs, rhs) -> bool:
        return self._same_access(
            self._store_map(lhs),
            self._store_index_terms(lhs),
            self._store_map(rhs),
            self._store_index_terms(rhs),
        )

    def _whole_buffer_fill(self, index, prefix) -> bool:
        op = self._op(index)
        try:
            shape = [int(d) for d in op.operands[1].type.shape]
        except (AttributeError, TypeError, ValueError):
            return False
        fill = self.sites[index][1][prefix:]
        if len(fill) != len(shape) or not shape:
            return False
        operands = [_value_name(o, self.state) for o in list(op.operands)[2:]]
        store_map = self._store_map(index)
        if store_map is None:
            dims = operands
        else:
            results = _affine_map_results(store_map)
            if results is None or len(results) != len(shape):
                return False
            dims = []
            for result in results:
                m = re.fullmatch(r"d(\d+)", result)
                if m is None or int(m.group(1)) >= len(operands):
                    return False
                dims.append(operands[int(m.group(1))])
        loops = {loop[0]: loop for loop in fill}
        if len(set(dims)) != len(dims):
            return False
        for iv, extent in zip(dims, shape):
            loop = loops.get(iv)
            if loop is None or loop[3] != 1:
                return False
            lo = _AFFINE_CONST_RE.match(str(loop[1]).strip())
            hi = _AFFINE_CONST_RE.match(str(loop[2]).strip())
            if not lo or not hi or int(lo.group(1)) != 0 or int(hi.group(1)) != extent:
                return False
        return True


def _format_coverage_problems(problems, exempt=frozenset()) -> list[str]:
    errors = []
    for functions, message in problems:
        count = sum(1 for name in functions if name not in exempt)
        if count == 0:
            continue
        if count > 1:
            message += (
                f" (the same store occurs in {count - 1} more work-item function(s))"
            )
        errors.append(message)
    return errors


def _host_executed_functions(target, trace: MatchTrace):
    """Functions the backend runs verbatim from their own MLIR on the host.

    Every store of such a function is implemented, matched or not.  Today
    only UPMEM has them: a kernel group whose family is placed on the host
    (``kmeans_update``, spec 004 D2) is compiled whole by the C backend.
    Group membership needs the retained work scopes stamped by
    ``compiler._stamp_matcher_work_scopes``; None means not yet known.
    """
    if getattr(target, "name", None) != "upmem":
        return frozenset()
    if any("spmw_work_scope" not in match.extra for match in trace.matches):
        return None
    from .spmw_autoschedule import _matcher_work_scope
    from .spmw_upmem import classify_upmem_kernel

    groups: dict = {}
    for match in trace.matches:
        groups.setdefault(_matcher_work_scope(match).group_id, []).append(match)
    host = set()
    for matches in groups.values():
        try:
            kernel = classify_upmem_kernel(target, matches)
        except (NotImplementedError, ValueError):
            continue
        if kernel.placement == "host":
            host.update(match.func_name for match in matches)
    return frozenset(host)


def external_effects(trace: MatchTrace, target=None) -> list[tuple]:
    """Externally observable writes a backend must realize: every matched or
    copied store whose destination is not a function-local buffer, as
    ``(function, memref name, destination ref, category)``.  Stores of
    functions the backend runs verbatim on the host are excluded (their
    effects are realized by that code).
    """
    coverage = getattr(trace, "coverage", None)
    if coverage is None:
        return []
    exempt = (
        _host_executed_functions(target, trace) if target is not None else frozenset()
    )
    effects = []
    for record in coverage.stores:
        if record.category not in (STORE_MATCHED, STORE_COPY) or record.local:
            continue
        if exempt and record.func_name in exempt:
            continue
        effects.append(
            (record.func_name, record.memref_name, record.value_ref, record.category)
        )
    return effects


def default_realized_effects(trace: MatchTrace) -> set:
    """Destination refs a matcher backend realizes unless it says otherwise:
    the matched stores' own destinations, and the sinks of a producer whose
    own destination is local (a scalar accumulator copied to the output).  A
    copy of an externally visible array result into another buffer is an
    additional effect; a backend that realizes it declares so through
    ``realized_effects``."""
    realized = set()
    for match in trace.matches:
        if match.result_value_ref is not None:
            realized.add(match.result_value_ref)
        if match.extra.get("result_is_local"):
            for ref, _local in match.extra.get("output_sink_refs", ()):
                if ref is not None:
                    realized.add(ref)
    return realized


def require_effects_realized(target, trace: MatchTrace, realized: set | None) -> None:
    """Raise unless every external effect of ``trace`` is realized.

    ``realized`` is the backend's declaration (destination refs it writes);
    None selects `default_realized_effects`.  Release gate G2: a compiled
    program must not report success while an observable write of the source
    never happens.
    """
    if realized is None:
        realized = default_realized_effects(trace)
    missing = []
    for func_name, memref, ref, category in external_effects(trace, target):
        if ref is None or ref not in realized:
            missing.append(f"'{memref}' in function '{func_name}' ({category} store)")
    if missing:
        name = getattr(target, "name", trace.target_name)
        raise NotImplementedError(
            f"target {name!r}: the lowering does not realize the write(s) to "
            + ", ".join(dict.fromkeys(missing))
            + "; the program would report success without performing them. "
            "Store the result only where the backend writes it, or extend the "
            "backend's realized effects."
        )


def require_match_coverage(
    trace: MatchTrace, *, target=None, module=None, defer_unknown=False
) -> MatchTrace:
    """Raise :class:`UnsupportedWorkloadError` if ``trace`` drops an effect.

    A trace from :func:`match_workload` carries its coverage.  A trace without
    one (assembled by hand or sliced) is re-matched from ``module`` when given;
    the returned trace is the one whose coverage was checked.  Problems in
    functions the backend executes verbatim on the host are exempt; when that
    set cannot be known yet (unstamped UPMEM trace), ``defer_unknown=True``
    skips the check, which ``compile_for_target`` then repeats, failing closed.
    """
    checked = trace
    coverage = getattr(trace, "coverage", None)
    if coverage is None and target is not None and module is not None:
        checked = match_workload(target, module)
        coverage = checked.coverage
    if coverage is None or not coverage.problems:
        return checked
    exempt = _host_executed_functions(target, checked)
    verbatim = frozenset(getattr(trace, "verbatim_functions", ()) or ())
    if exempt is None:
        if not verbatim and defer_unknown:
            return checked
        exempt = frozenset()
    # A function a program record executes verbatim from its own retained
    # IR (an ``APUv1Program`` phase) implements every store itself; its
    # store facts are retained in the coverage and stated as the record's
    # effects, so its coverage problems are not dropped effects.
    errors = _format_coverage_problems(coverage.problems, exempt | verbatim)
    if errors:
        raise UnsupportedWorkloadError(
            f"workload is not supported by target {trace.target_name!r}:\n  - "
            + "\n  - ".join(errors)
        )
    return checked


__all__ = [
    "OpPattern",
    "PVar",
    "PConst",
    "PBinOp",
    "PUnary",
    "PCmp",
    "PSelect",
    "WLoad",
    "WBlockArg",
    "WConst",
    "WBinOp",
    "WUnary",
    "WCmp",
    "WSelect",
    "compile_op_pattern",
    "compile_target_patterns",
    "match_workload",
    "require_match_coverage",
    "UnsupportedWorkloadError",
    "batch_dim",
]
