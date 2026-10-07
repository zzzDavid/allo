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
from .spmw_match import IRValueRef, MatchedOp, MatchTrace, OperandBinding


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
        except Exception:  # noqa: BLE001
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
            # Numeric equality where possible; otherwise structural.
            try:
                return term.value == pattern.value
            except Exception:  # noqa: BLE001
                return False
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
                    elif (
                        list(acc_term.indices) != list(store_indices)
                        or acc_term.index_map != store_map
                        or _data_dependent(acc_term.source_op_name, acc_term.index_terms)
                    ):
                        continue
                extra = {"store_indices": list(store_indices)}
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


def batch_dim(match: MatchedOp) -> tuple[str | None, int | None]:
    """Return ``(batch_loop_var, B)`` for a batched reduction match.

    SPEC-026 §1.2. A loop ``L`` in ``match.enclosing_loops`` is the BATCH
    loop iff its iter var:
      (a) indexes some non-accumulator input operand load (the per-batch input
          vector may be stored either X[b,k] or X[k,b]), AND
      (b) is ABSENT from some OTHER non-accumulator input operand's index
          list (the resident weight W[row, k], which the batch axis never
          indexes).

    This is the structural separation between the batch axis (leading
    index of one input, absent from the other) and the reduction axis
    (present in *every* input's index list). It is computed over loop
    vars + operand indices only -- never a positional ``enclosing_loops[0]``
    assumption, never a shape literal. ``B`` is `_parse_loop_upper(L.upper)`.

    Returns ``(None, None)`` when no loop satisfies (a)+(b): the canonical
    single-vector GEMV and the eltwise paths, which downstream treats as
    ``B = 1`` (SPEC-026 I4 parity).
    """
    loops = match.enclosing_loops
    if not loops:
        return (None, None)
    loop_vars = {L[0] for L in loops}

    # Non-accumulator input operands with at least one index.
    inputs = [opb for opb in match.operands if not opb.is_loop_carried and opb.indices]
    if len(inputs) < 2:
        # Need >=2 inputs to separate "leading index of one, absent from
        # another"; a single-input reduction has no batch axis.
        return (None, None)

    for L in loops:
        var = L[0]
        # (a) indexes some input operand.  Requiring a leading index loses the
        # canonical PolyBench A@B spelling, whose column batch is B[k,j].
        is_input_axis = any(var in opb.indices for opb in inputs)
        if not is_input_axis:
            continue
        # (b) absent from some OTHER input operand entirely.
        absent_elsewhere = any(var not in opb.indices for opb in inputs)
        if not absent_elsewhere:
            continue
        # Guard: a batch axis is an enclosing loop var (so B is bounded by
        # the trace, never a literal we invented).
        if var not in loop_vars:
            continue
        return (var, _parse_loop_upper(L[2]))
    return (None, None)


def _stamp_batch_dims(trace: MatchTrace) -> None:
    """Write `match.extra['batch_loop_var']` / `['batch_dim']` for every
    reduction match in ``trace`` (SPEC-026 §1.2).

    Single source of truth: cost (205) and codegen (206) read
    `extra['batch_dim']` and never re-derive it. A match with no batch
    axis gets ``batch_dim = 1`` (I4 parity).
    """
    for m in trace.matches:
        var, B = batch_dim(m)
        m.extra["batch_loop_var"] = var
        m.extra["batch_dim"] = B if B is not None else 1


def match_workload(target, mlir_module) -> MatchTrace:
    """Walk ``mlir_module`` and emit a :class:`MatchTrace` for ``target``.

    Eagerly compiles target Op patterns the first time it sees them.
    """
    compile_target_patterns(target)
    value_refs, source_value_refs = _build_ir_value_refs(mlir_module)
    trace = MatchTrace(
        target_name=getattr(target, "name", "<unknown>"),
        module_name=str(getattr(mlir_module, "name", "<unnamed>")),
        source_value_refs=source_value_refs,
    )

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

        for op, loops, guards in sites:
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
            trace.matches.extend(ms)

    # SPEC-026 §1.2: stamp the batch dim on each match (one source of
    # truth for cost/codegen). Additive; default batch_dim=1.
    _stamp_batch_dims(trace)
    return trace


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
    "batch_dim",
]
