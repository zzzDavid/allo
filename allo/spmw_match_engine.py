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
  - ``PBinOp(op, lhs, rhs)`` op in {"add","sub","mul","div"};
                             ``add``/``mul`` are commutative.

The matcher reflects each candidate workload site as a similar tree of
``WTerm`` nodes (``WLoad`` / ``WConst`` / ``WBinOp``) by tracing the SSA
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

from .spmw_match import MatchedOp, MatchTrace, OperandBinding


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
class OpPattern:
    """Compiled pattern for one target Op.

    ``param_names`` is the lambda's argument list (in declaration order),
    ``body`` is the pattern tree.
    """

    param_names: list[str]
    body: Any  # PVar | PConst | PBinOp


_AST_BINOP_TO_NAME = {
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mul",
    ast.Div: "div",
}


def _compile_expr(node: ast.AST, params: set[str]) -> Any:
    if isinstance(node, ast.BinOp):
        kind = _AST_BINOP_TO_NAME.get(type(node.op))
        if kind is None:
            raise ValueError(f"unsupported binop in lambda: {ast.dump(node.op)}")
        return PBinOp(kind, _compile_expr(node.left, params), _compile_expr(node.right, params))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        # -x  =>  0 - x   (keeps the AST simple)
        return PBinOp("sub", PConst(0), _compile_expr(node.operand, params))
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


_FP_BINOPS = {
    "arith.mulf": "mul",
    "arith.addf": "add",
    "arith.subf": "sub",
    "arith.divf": "div",
    "arith.muli": "mul",
    "arith.addi": "add",
    "arith.subi": "sub",
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


def _operand_names(op) -> list[str]:
    return [o.get_name() for o in op.operands]


def _result_names(op) -> list[str]:
    return [r.get_name() for r in op.results]


def _build_load_term(load_op) -> WLoad:
    name = load_op.operation.name
    attrs = load_op.attributes
    memref_name = None
    if "from" in attrs:
        memref_name = _attr_str(attrs["from"])
    operands = _operand_names(load_op)
    # operands[0] is the memref; the rest are the indices.
    indices = operands[1:] if len(operands) > 1 else []
    ssa = _result_names(load_op)[0] if load_op.results else "<no-result>"
    return WLoad(memref_name=memref_name, ssa_name=ssa, indices=indices, source_op_name=name)


def _trace_value(ssa_name: str, defining_map: dict[str, Any]) -> Any:
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
        return _build_load_term(src)
    if op_name in _FP_BINOPS:
        kind = _FP_BINOPS[op_name]
        ops = _operand_names(src)
        lhs = _trace_value(ops[0], defining_map)
        rhs = _trace_value(ops[1], defining_map)
        return WBinOp(kind, lhs, rhs, _result_names(src)[0])
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
            return WConst(v, _result_names(src)[0])
        except Exception:  # noqa: BLE001
            return WConst(None, _result_names(src)[0])
    # Anything else (extsi, index_cast, sitofp, ...) is treated as opaque.
    # We trace through single-input casts so that constants on the other
    # side still appear as constants when the pattern needs them.
    if len(src.operands) == 1 and src.results:
        inner = _trace_value(_operand_names(src)[0], defining_map)
        # Wrap-through: keep the original ssa name so codegen can audit.
        if isinstance(inner, (WLoad, WBlockArg, WConst, WBinOp)):
            return inner
    return WBlockArg(ssa_name=_result_names(src)[0] if src.results else ssa_name)


# --------------------------------------------------------------------- #
# Pattern unification
# --------------------------------------------------------------------- #


_COMMUTATIVE = {"add", "mul"}


def _unify(pattern, term, bindings: dict[str, Any]) -> bool:
    """Try to unify a pattern node against a workload term.

    On success ``bindings`` is populated with ``{param_name: WTerm}`` and
    True is returned. On failure ``bindings`` is left in an undefined
    state — callers should pass a fresh dict each top-level call.
    """
    if isinstance(pattern, PVar):
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
        if _unify(pattern.lhs, term.lhs, b) and _unify(pattern.rhs, term.rhs, b):
            bindings.clear()
            bindings.update(b)
            return True
        if pattern.op in _COMMUTATIVE:
            b = dict(bindings)
            if _unify(pattern.lhs, term.rhs, b) and _unify(pattern.rhs, term.lhs, b):
                bindings.clear()
                bindings.update(b)
                return True
        return False

    return False


def _term_eq(a, b) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, WLoad):
        return a.ssa_name == b.ssa_name
    if isinstance(a, WBlockArg):
        return a.ssa_name == b.ssa_name
    if isinstance(a, WConst):
        return a.value == b.value
    if isinstance(a, WBinOp):
        return a.ssa_name == b.ssa_name
    return False


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


def _walk_loops(block, prefix: list[tuple[str, str, str, int]], collector):
    """Recurse into ``block``'s ops, collecting affine.for loops in ``prefix``
    and yielding (op, current_loop_stack) for every non-loop op via
    ``collector.append``.
    """
    for op in block.operations:
        if op.operation.name == "affine.for":
            attrs = op.attributes
            iv_name = op.regions[0].blocks[0].arguments[0].get_name()
            lb = _affine_map_text(attrs["lowerBoundMap"]) if "lowerBoundMap" in attrs else "?"
            ub = _affine_map_text(attrs["upperBoundMap"]) if "upperBoundMap" in attrs else "?"
            step_attr = attrs["step"] if "step" in attrs else None
            try:
                step = int(str(step_attr).split(":")[0].strip()) if step_attr is not None else 1
            except ValueError:
                step = 1
            loop = (iv_name, lb, ub, step)
            for inner_block in op.regions[0].blocks:
                _walk_loops(inner_block, prefix + [loop], collector)
        else:
            collector.append((op, list(prefix)))
            for r in op.regions:
                for blk in r.blocks:
                    _walk_loops(blk, prefix, collector)


def _build_defining_map(func) -> dict[str, Any]:
    m: dict[str, Any] = {}

    def visit(block):
        for op in block.operations:
            for r in op.results:
                m[r.get_name()] = op
            for region in op.regions:
                for blk in region.blocks:
                    visit(blk)

    body = func.regions[0].blocks[0]
    visit(body)
    return m


# --------------------------------------------------------------------- #
# Match driver
# --------------------------------------------------------------------- #


def _operand_bindings(
    pattern: OpPattern,
    bindings: dict[str, Any],
    accumulates: bool,
) -> list[OperandBinding]:
    """Build OperandBindings (in fn signature order) from a successful unify."""
    out: list[OperandBinding] = []
    # The accumulator parameter — convention: if accumulates, the last
    # parameter is the loop-carried accumulator. Allo's frontend names it
    # "acc" via the {from = "acc"} attr; we mark it as such.
    acc_name = pattern.param_names[-1] if (accumulates and pattern.param_names) else None
    for name in pattern.param_names:
        term = bindings.get(name)
        if isinstance(term, WLoad):
            out.append(
                OperandBinding(
                    role=name,
                    memref_name=term.memref_name,
                    indices=list(term.indices),
                    is_loop_carried=(name == acc_name),
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


def _try_match_at_store(
    store_op,
    enclosing_loops: list[tuple[str, str, str, int]],
    target,
    func_name: str,
    work_id: tuple[int, ...],
    defining_map: dict[str, Any],
) -> list[MatchedOp]:
    """If ``store_op`` (a memref.store / affine.store) writes a value that
    matches one of the target's compiled op patterns, emit MatchedOps for it.
    """
    op_name = store_op.operation.name
    if op_name not in _STORE_OPS:
        return []
    operands = _operand_names(store_op)
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

    term = _trace_value(stored_ssa, defining_map)
    # Only a binop term is interesting for the patterns we care about.
    if not isinstance(term, WBinOp):
        return []

    store_handle = f"{op_name}@{stored_ssa}"

    def _match_term(t: "WBinOp") -> MatchedOp | None:
        """Unify one binop term against the target's op patterns; return the
        first matching MatchedOp (the existing single-term logic), or None."""
        for unit in target._walk():
            for op_obj in unit.ops.values():
                pat = compile_op_pattern(op_obj)
                bindings: dict[str, Any] = {}
                if not _unify(pat.body, t, bindings):
                    continue
                # Conservative filter: every parameter must bind to a memref
                # load. Block-arg / constant / opaque-cast bindings are
                # rejected so we don't match index-arithmetic chains (e.g.
                # `pid * 8` computing `row0`) against data-plane ops.
                if not all(isinstance(bindings.get(n), WLoad) for n in pat.param_names):
                    continue
                # If this op accumulates, validate the accumulator: the last
                # parameter must bind to a WLoad whose memref equals the
                # store's "to" memref (i.e. the same accumulator memref).
                if op_obj.accumulates and pat.param_names:
                    acc_term = bindings[pat.param_names[-1]]
                    if not isinstance(acc_term, WLoad):
                        continue
                    if (
                        result_memref_name is not None
                        and acc_term.memref_name is not None
                        and acc_term.memref_name != result_memref_name
                    ):
                        continue
                # op_range — first contributing load through the store.
                return MatchedOp(
                    target_op_name=op_obj.name,
                    func_name=func_name,
                    work_id=work_id,
                    enclosing_loops=list(enclosing_loops),
                    operands=_operand_bindings(pat, bindings, op_obj.accumulates),
                    result_memref_name=result_memref_name,
                    op_range=(t.ssa_name, store_handle),
                    extra={"store_indices": list(store_indices)},
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
    resolver does not import from `spmw_cost_models` (which imports this
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
      (a) is the LEADING index of some non-accumulator input operand load
          (the per-batch input vector X[b, k]), AND
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
    inputs = [
        opb
        for opb in match.operands
        if not opb.is_loop_carried and opb.indices
    ]
    if len(inputs) < 2:
        # Need >=2 inputs to separate "leading index of one, absent from
        # another"; a single-input reduction has no batch axis.
        return (None, None)

    for L in loops:
        var = L[0]
        # (a) leading index of some input operand.
        is_leading = any(opb.indices[0] == var for opb in inputs)
        if not is_leading:
            continue
        # (b) absent from some OTHER input operand entirely.
        absent_elsewhere = any(
            opb.indices[0] != var and var not in opb.indices
            for opb in inputs
        )
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
    trace = MatchTrace(
        target_name=getattr(target, "name", "<unknown>"),
        module_name=str(getattr(mlir_module, "name", "<unnamed>")),
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
        defining_map = _build_defining_map(func)

        body = func.regions[0].blocks[0]
        sites: list[tuple[Any, list[tuple[str, str, str, int]]]] = []
        _walk_loops(body, [], sites)

        for op, loops in sites:
            if op.operation.name not in _STORE_OPS:
                continue
            ms = _try_match_at_store(
                op,
                loops,
                target,
                func_name,
                work_id,
                defining_map,
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
    "WLoad",
    "WBlockArg",
    "WConst",
    "WBinOp",
    "compile_op_pattern",
    "compile_target_patterns",
    "match_workload",
    "batch_dim",
]
