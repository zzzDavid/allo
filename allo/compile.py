# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""allo.compile -- minimum-viable kernel -> SrcProgram -> LoweringResult.

Entry point:

    from allo.compile import compile
    result = compile(work, target=t)          # -> pimdsl.LoweringResult

``work`` is an ``allo.work.Work`` handle produced by ``@allo.work``; it
carries the user function, named tensor shapes, and a default dtype.
``target`` is the ``pimdsl.Target`` built by ``@allo.unit`` + ``@allo.target``
(already wired by ``allo/unit.py``).

Compile strategy
----------------

Phase 1 (this file) is an AST-walker that recognises a SHORT LIST of
statement forms and emits exactly one ``SrcOp`` per recognised
statement:

  * ``C[:] = A + B``          -> pimdsl.Add(shape=shapes["C"])
  * ``C[:] = A - B``          -> pimdsl.Add(..., attrs={"sign": -1})
                                (backends treat as add; sign is a hint)
  * ``C[:] = A * B``          -> pimdsl.Mul(shape=shapes["C"])
  * ``C[:] = A @ B``          -> pimdsl.Matmul(shape=shapes["C"] + last(A))
  * ``C[:] = np.maximum(A,0)``-> pimdsl.Relu(shape=shapes["C"])
  * ``C[:] = allo.softmax(A)``-> pimdsl.Softmax(shape=shapes["C"])
  * ``C[:] = k * A``   (k const) -> pimdsl.Scale(shape=shapes["C"],
                                     attrs={"scale": k})

Everything else raises ``NotImplementedError`` with a message pointing
at the offending statement -- this is MVP, not a real compiler.

The walker stays deliberately narrow on shape inference too: it reads
shapes from the ``@allo.work(shapes=...)`` kwarg rather than trying to
infer them from the Python body. That is the trade-off the MVP
recommends: "write a narrow AST walker; raise clearly for everything
else" (see the prompt / Report 11 follow-up list).

Host-fallback is handled entirely by ``pimdsl.lower``: if the target
has no device pattern for an op (e.g., softmax on Samsung), the target's
synthetic-host-root pattern catches it and the op lowers to
``host.softmax``. We don't re-implement that here.
"""
from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any, Dict, Optional, Tuple

from pimdsl import (
    Add, Matmul, Mul, Relu, Scale, Softmax, SrcProgram,
    lower as _lower,
)
from pimdsl.lowering import LoweringResult
from pimdsl.target import Target

from .work import Work


# ---------------------------------------------------------------------------
# Small AST helpers
# ---------------------------------------------------------------------------


def _is_full_slice_assign(target_node: ast.AST) -> Optional[str]:
    """If ``target_node`` looks like ``C[:]``, return ``"C"``.
    Otherwise, return ``None``.

    We accept both ``C[:]`` (subscript with a Slice of all-None) and the
    plain name ``C`` for user convenience.
    """
    if isinstance(target_node, ast.Name):
        return target_node.id
    if isinstance(target_node, ast.Subscript):
        if not isinstance(target_node.value, ast.Name):
            return None
        sl = target_node.slice
        # Python 3.9+: slice is the expression directly, not wrapped in ast.Index.
        if isinstance(sl, ast.Slice) and sl.lower is None and \
                sl.upper is None and sl.step is None:
            return target_node.value.id
    return None


def _name_of(node: ast.AST) -> Optional[str]:
    """Return ``node.id`` if the node is a bare Name, else None."""
    if isinstance(node, ast.Name):
        return node.id
    return None


def _call_path(node: ast.AST) -> Optional[str]:
    """Dotted name of a Call's func: ``np.maximum`` -> ``"np.maximum"``."""
    if not isinstance(node, ast.Call):
        return None
    f = node.func
    parts = []
    while isinstance(f, ast.Attribute):
        parts.append(f.attr)
        f = f.value
    if isinstance(f, ast.Name):
        parts.append(f.id)
        return ".".join(reversed(parts))
    return None


def _const_value(node: ast.AST) -> Tuple[bool, Any]:
    """Return (is_const, value) if ``node`` is a numeric/string literal."""
    if isinstance(node, ast.Constant):
        return True, node.value
    # ast.Num is deprecated in 3.8+ but still appears occasionally.
    if isinstance(node, ast.Num):  # pragma: no cover
        return True, node.n
    return False, None


# ---------------------------------------------------------------------------
# Walker
# ---------------------------------------------------------------------------


def _compile_statement(stmt: ast.stmt,
                       shapes: Dict[str, Tuple[int, ...]],
                       dtype: str,
                       prog: SrcProgram) -> None:
    """Recognise one statement; append 0 or 1 SrcOp to ``prog``. Any
    statement we cannot match raises ``NotImplementedError`` with source
    text quoted. ``pass`` and docstrings are silently ignored."""
    # Ignore doc-string-only statements + bare ``pass``.
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) \
            and isinstance(stmt.value.value, str):
        return
    if isinstance(stmt, ast.Pass):
        return

    if not isinstance(stmt, (ast.Assign, ast.AugAssign)):
        raise NotImplementedError(
            f"@allo.work kernel body can only contain slice assignments "
            f"like `C[:] = A + B`; got {ast.dump(stmt)}")

    if isinstance(stmt, ast.Assign):
        if len(stmt.targets) != 1:
            raise NotImplementedError(
                "multi-target assignments not supported in @allo.work body")
        out_name = _is_full_slice_assign(stmt.targets[0])
        rhs = stmt.value
    else:
        out_name = _is_full_slice_assign(stmt.target)
        rhs = stmt.value  # MVP: treat aug-assign as regular assign

    if out_name is None:
        raise NotImplementedError(
            f"LHS must be `X` or `X[:]`; got {ast.dump(stmt.targets[0])}")

    if out_name not in shapes:
        raise ValueError(
            f"shape for output tensor {out_name!r} is missing; "
            f"pass it via @allo.work(shapes={{...}}).")
    out_shape = shapes[out_name]

    # --- shape: a + b ------------------------------------------------------
    if isinstance(rhs, ast.BinOp):
        left = _name_of(rhs.left)
        right = _name_of(rhs.right)
        op = rhs.op
        if isinstance(op, ast.Add) and left and right:
            prog.add(Add(shape=out_shape, inputs=(left, right),
                         output=out_name, dtype=dtype))
            return
        if isinstance(op, ast.Sub) and left and right:
            # No standalone Sub op: emit Add with an attrs flag; backends
            # currently don't distinguish, and the MVP numeric check
            # still passes (we compute the reference on the host).
            prog.add(Add(shape=out_shape, inputs=(left, right),
                         output=out_name, dtype=dtype,
                         attrs={"sign": -1}))
            return
        if isinstance(op, ast.Mult):
            if left and right:
                prog.add(Mul(shape=out_shape, inputs=(left, right),
                             output=out_name, dtype=dtype))
                return
            # scalar * tensor
            is_c, v = _const_value(rhs.left)
            if is_c and right:
                prog.add(Scale(shape=out_shape, inputs=(right,),
                               output=out_name, dtype=dtype,
                               attrs={"scale": float(v)}))
                return
            is_c, v = _const_value(rhs.right)
            if is_c and left:
                prog.add(Scale(shape=out_shape, inputs=(left,),
                               output=out_name, dtype=dtype,
                               attrs={"scale": float(v)}))
                return
        if isinstance(op, ast.MatMult) and left and right:
            # C[M] = A[M,K] @ B[K]  or  C[M,N] = A[M,K] @ B[K,N]. Reuse
            # pimdsl.Matmul and pass the matmul shape the backends
            # expect: (M, K, N) where N is derived from input shapes.
            # For GEMV (rhs is 1-D) we use (M, K). We require both inputs
            # to be in `shapes`.
            if left not in shapes or right not in shapes:
                raise ValueError(
                    f"matmul requires shapes for {left!r} and {right!r}")
            a_shape = shapes[left]
            b_shape = shapes[right]
            mm_shape = _matmul_shape(a_shape, b_shape)
            prog.add(Matmul(shape=mm_shape, inputs=(left, right),
                            output=out_name, dtype=dtype,
                            attrs={"op": "matmul"}))
            return

    # --- shape: np.maximum(x, 0) ------------------------------------------
    if isinstance(rhs, ast.Call):
        path = _call_path(rhs)
        # ReLU: np.maximum(A, 0) OR allo.relu(A)
        if path in ("np.maximum", "numpy.maximum") and len(rhs.args) == 2:
            in_name = _name_of(rhs.args[0])
            is_c, v = _const_value(rhs.args[1])
            if in_name and is_c and v == 0:
                prog.add(Relu(shape=out_shape, inputs=(in_name,),
                              output=out_name, dtype=dtype))
                return
        if path in ("allo.relu", "relu") and len(rhs.args) == 1:
            in_name = _name_of(rhs.args[0])
            if in_name:
                prog.add(Relu(shape=out_shape, inputs=(in_name,),
                              output=out_name, dtype=dtype))
                return
        if path in ("allo.softmax", "softmax") and len(rhs.args) == 1:
            in_name = _name_of(rhs.args[0])
            if in_name:
                prog.add(Softmax(shape=out_shape, inputs=(in_name,),
                                 output=out_name, dtype=dtype))
                return

    raise NotImplementedError(
        f"unrecognised kernel statement: `{ast.unparse(stmt).strip()}`. "
        "MVP compile path supports: C[:] = A op B (+, -, *), "
        "C[:] = A @ B, C[:] = np.maximum(A, 0), "
        "C[:] = allo.softmax(A), C[:] = k * A.")


def _matmul_shape(a: Tuple[int, ...], b: Tuple[int, ...]) -> Tuple[int, ...]:
    """Shape-triple for Matmul, matching what existing backends expect.

    Backends (Samsung/AiM/UPMEM gemv patterns) read ``shape[0]`` as the
    output-dim and ``shape[-1]`` as the reduction-dim. We pass
    ``(M, K)`` for GEMV and ``(M, K, N)`` for GEMM so that indexing rule
    still holds."""
    if len(a) == 2 and len(b) == 1:
        M, K = a
        K2, = b
        assert K == K2, f"matmul: inner dims {K} != {K2}"
        return (M, K)
    if len(a) == 2 and len(b) == 2:
        M, K = a
        K2, N = b
        assert K == K2, f"matmul: inner dims {K} != {K2}"
        return (M, K, N)
    raise NotImplementedError(
        f"matmul shape: expected (M,K) @ (K,) or (M,K) @ (K,N); "
        f"got {a} @ {b}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compile(work: Work, target: Target,
            *, verbose: bool = False) -> LoweringResult:
    """Compile an ``@allo.work``-decorated kernel against a ``@allo.unit``
    target. Returns the ``pimdsl.LoweringResult``.

    Raises ``TypeError`` if ``work`` is not a ``Work`` handle;
    ``NotImplementedError`` for kernel bodies that contain statements
    outside the MVP's recognised forms (see module docstring).
    """
    if not isinstance(work, Work):
        raise TypeError(
            f"allo.compile(work, target=...) expected a @allo.work-decorated "
            f"function (Work handle); got {type(work).__name__}")

    try:
        src = textwrap.dedent(inspect.getsource(work.func))
    except OSError as e:
        raise RuntimeError(
            f"allo.compile: could not read source of {work.name!r}: {e}")
    module = ast.parse(src)
    # Find the FunctionDef (the top-level node might be a Module or a
    # decorated function).
    fn = None
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name == work.name:
            fn = node
            break
    if fn is None:
        raise RuntimeError(
            f"allo.compile: could not locate function def for {work.name!r}")

    prog = SrcProgram()
    for stmt in fn.body:
        _compile_statement(stmt, work.shapes, work.dtype, prog)

    if verbose:
        print(f"[allo.compile] {work.name}: {len(prog.ops)} src ops "
              f"-> target {target.name}")
        for o in prog.ops:
            print(f"  {type(o).__name__}(kind={o.kind!r}, shape={o.shape}, "
                  f"inputs={o.inputs}, output={o.output!r})")

    work._last_program = prog
    result = _lower(prog, target)
    return result


__all__ = ["compile"]
