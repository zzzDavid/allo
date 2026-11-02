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
import re
from dataclasses import dataclass, field
from typing import Any

from ._mlir.ir import AsmState
from .spmw_match import (
    STORE_ACCUMULATOR_INIT,
    STORE_COPY,
    STORE_LOCAL,
    STORE_MATCHED,
    STORE_UNMATCHED,
    IRValueRef,
    MatchCoverage,
    MatchedOp,
    MatchTrace,
    OperandBinding,
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


@dataclass
class WUnary:
    fn: str  # normalized name, see _MATH_UNARY; "neg" for arith.negf
    arg: Any
    ssa_name: str


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

        for argument in func.arguments:
            if argument not in self._names:
                assign(argument)
        visit(func.regions[0].blocks[0])

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
                1 for _w, wpath in writes if wpath[:-1] == block and wpath[-1] < path[-1]
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


def _build_load_term(load_op, value_refs, *, state, scope=None, defining_map=None) -> WLoad:
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
                scope.forward[ssa_name], defining_map, value_refs, state=state, scope=scope
            )
        return _build_load_term(
            src, value_refs, state=state, scope=scope, defining_map=defining_map
        )
    if op_name in _FP_BINOPS:
        kind = _FP_BINOPS[op_name]
        ops = _operand_names(src, state=state)
        lhs = _trace_value(ops[0], defining_map, value_refs, state=state, scope=scope)
        rhs = _trace_value(ops[1], defining_map, value_refs, state=state, scope=scope)
        return WBinOp(kind, lhs, rhs, _result_names(src, state=state)[0])
    if op_name in _MATH_UNARY:
        arg = _trace_value(
            _operand_names(src, state=state)[0], defining_map, value_refs, state=state, scope=scope
        )
        return WUnary(_MATH_UNARY[op_name], arg, _result_names(src, state=state)[0])
    if op_name in {"arith.cmpf", "arith.cmpi"}:
        predicate = _compare_predicate(src)
        if predicate is not None:
            ops = _operand_names(src, state=state)
            lhs = _trace_value(ops[0], defining_map, value_refs, state=state, scope=scope)
            rhs = _trace_value(ops[1], defining_map, value_refs, state=state, scope=scope)
            return WCmp(
                predicate[0], lhs, rhs, _result_names(src, state=state)[0], predicate[1]
            )
        return WBlockArg(ssa_name=_result_names(src, state=state)[0])
    if op_name == "arith.select":
        ops = _operand_names(src, state=state)
        cond, a, b = (
            _trace_value(name, defining_map, value_refs, state=state, scope=scope) for name in ops
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
            _operand_names(src, state=state)[0], defining_map, value_refs, state=state, scope=scope
        )
        value = _const_value(inner)
        result = _result_names(src, state=state)[0]
        if value is not None:
            return WConst(-value, result)
        return WUnary("neg", inner, result)
    # Only value-preserving casts are traced through, so that constants on
    # the other side still appear as constants when the pattern needs them.
    # Every other single-operand op is opaque.
    if op_name in _CAST_OPS and len(src.operands) == 1 and src.results:
        inner = _trace_value(
            _operand_names(src, state=state)[0], defining_map, value_refs, state=state, scope=scope
        )
        # Wrap-through: keep the original ssa name so codegen can audit.
        if isinstance(inner, (WLoad, WBlockArg, WConst, WBinOp, WSelect)):
            return inner
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
            "const"
            if name in const_params
            else "iv"
            if name in iv_params
            else "load"
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
        return (
            _term_eq(a.cond, b.cond)
            and _term_eq(a.a, b.a)
            and _term_eq(a.b, b.b)
        )
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
                    frames = guards + ((("if", cond) if region_index == 0 else ("not", cond)),)
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
                (frame[0], _trace_value(
                    frame[1], defining_map, value_refs, state=state, scope=scope
                ))
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
                pat = compile_op_pattern(op_obj)
                bindings: dict[str, Any] = {}
                kinds = _param_kinds(op_obj, pat)
                if not _unify(pat.body, t, bindings, kinds):
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
            (r["matrix"], r["vector"], r["trips"]) != (first["matrix"], first["vector"], first["trips"])
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
        """The load whose value the store writes, through casts; else None."""
        name = _value_name(self._op(index).operands[0], self.state)
        src = self.defining_map.get(name)
        while (
            src is not None
            and src.operation.name in _CAST_OPS
            and len(src.operands) == 1
        ):
            src = self.defining_map.get(_value_name(src.operands[0], self.state))
        if src is not None and src.operation.name in _LOAD_OPS and src.operands:
            return src
        return None

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
                    category[index] = STORE_COPY
                    copy_producers[index] = producers
                    sink = (self._dest_name(index), self._index_text(index))
                    for j in producers:
                        for match in self.site_matches[j]:
                            sinks = match.extra.setdefault("output_sinks", [])
                            if sink not in sinks:
                                sinks.append(sink)
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

        for index in self.stores:
            cat = category[index]
            coverage.stores.append(
                StoreCoverage(
                    func_name=self.func_name,
                    store=self._handle(index),
                    memref_name=self._dest_name(index),
                    enclosing_loops=tuple(self.sites[index][1]),
                    category=cat,
                    detail=detail.get(index, ""),
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
                    names = ", ".join(m.target_op_name for m in self.site_matches[index])
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
        if reason:
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
            return "unsupported", [
                "it is a local buffer that is never initialized"
            ]
        _op, match_loops, match_guards = self.sites[index]
        match_ivs = [loop[0] for loop in match_loops]
        found = []
        for other in others:
            other_loops, other_guards = self.sites[other][1], self.sites[other][2]
            other_ivs = [loop[0] for loop in other_loops]
            at = f"store {self._handle(other)} at loop nest {_render_loops(other_loops)}"
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
            message += f" (the same store occurs in {count - 1} more work-item function(s))"
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
    if exempt is None:
        if defer_unknown:
            return checked
        exempt = frozenset()
    errors = _format_coverage_problems(coverage.problems, exempt)
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
