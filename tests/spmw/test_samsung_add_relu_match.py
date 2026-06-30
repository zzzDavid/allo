# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Samsung general-instruction codegen extension:
the matcher unifies a ``max(x, 0)`` ReLU body (via ``arith.maximumf``) and a
pure ``x + y`` store (via ``arith.addf``), and the new fixture ADD/RELU ops
compile to the expected pattern shapes.

These are static matcher/codegen unit assertions (no simulator boot), per the
coder contract (simulator-running tests are the verifier's domain).
"""

from __future__ import annotations

from allo.spmw_match_engine import (
    PBinOp,
    PConst,
    PVar,
    WBinOp,
    WConst,
    WLoad,
    compile_op_pattern,
    _unify,
)

from _fixtures import build_samsung_target


class _Fake:
    pass


def _mk(fn):
    f = _Fake()
    f.fn = fn
    f.pattern = None
    f.name = "anon"
    return f


# --------------------------------------------------------------------- #
# max / ReLU
# --------------------------------------------------------------------- #


def test_relu_max_pattern_compiles():
    """``lambda x: max(x, 0)`` -> PBinOp("max", PVar(x), PConst(0))."""
    op = _mk(lambda x: max(x, 0))
    pat = compile_op_pattern(op)
    assert pat.param_names == ["x"]
    assert isinstance(pat.body, PBinOp)
    assert pat.body.op == "max"
    assert isinstance(pat.body.lhs, PVar) and pat.body.lhs.name == "x"
    assert isinstance(pat.body.rhs, PConst) and pat.body.rhs.value == 0


def test_relu_unifies_against_maximumf():
    """The ReLU body unifies against an ``arith.maximumf(%load, %const0)``
    term, in BOTH operand orders (max is commutative)."""
    op = _mk(lambda x: max(x, 0))
    pat = compile_op_pattern(op)

    in_load = WLoad("local_in", "%10", ["%arg0"], "affine.load")
    zero = WConst(0.0, "%c0")

    # canonical order: maximumf(load, 0.0)
    term = WBinOp("max", in_load, zero, "%11")
    bind: dict = {}
    assert _unify(pat.body, term, bind)
    assert isinstance(bind["x"], WLoad) and bind["x"].memref_name == "local_in"

    # reversed order: maximumf(0.0, load) -- commutativity must still unify
    term_rev = WBinOp("max", zero, in_load, "%11")
    bind_rev: dict = {}
    assert _unify(pat.body, term_rev, bind_rev)
    assert isinstance(bind_rev["x"], WLoad)


def test_relu_does_not_unify_against_add():
    """A ``max`` pattern must NOT spuriously match an ``add`` term."""
    op = _mk(lambda x: max(x, 0))
    pat = compile_op_pattern(op)
    in_load = WLoad("local_in", "%10", ["%arg0"], "affine.load")
    add_term = WBinOp("add", in_load, WConst(0.0, "%c0"), "%11")
    assert not _unify(pat.body, add_term, {})


# --------------------------------------------------------------------- #
# ADD
# --------------------------------------------------------------------- #


def test_add_pattern_compiles():
    """``lambda x, y: x + y`` -> PBinOp("add", PVar(x), PVar(y))."""
    op = _mk(lambda x, y: x + y)
    pat = compile_op_pattern(op)
    assert pat.param_names == ["x", "y"]
    assert isinstance(pat.body, PBinOp)
    assert pat.body.op == "add"


def test_add_unifies_pure_two_load_store():
    """``x + y`` unifies against ``addf(load_a, load_b)`` in both orders."""
    op = _mk(lambda x, y: x + y)
    pat = compile_op_pattern(op)

    a_load = WLoad("local_a", "%10", ["%arg0"], "affine.load")
    b_load = WLoad("local_b", "%11", ["%arg0"], "affine.load")
    term = WBinOp("add", a_load, b_load, "%12")

    bind: dict = {}
    assert _unify(pat.body, term, bind)
    assert isinstance(bind["x"], WLoad) and isinstance(bind["y"], WLoad)

    # reversed operand order: commutative add still unifies
    term_rev = WBinOp("add", b_load, a_load, "%12")
    bind_rev: dict = {}
    assert _unify(pat.body, term_rev, bind_rev)


def test_add_does_not_unify_against_mul():
    """ADD must NOT match a MUL term (distinct top-level op)."""
    op = _mk(lambda x, y: x + y)
    pat = compile_op_pattern(op)
    a_load = WLoad("local_a", "%10", ["%arg0"], "affine.load")
    b_load = WLoad("local_b", "%11", ["%arg0"], "affine.load")
    mul_term = WBinOp("mul", a_load, b_load, "%12")
    assert not _unify(pat.body, mul_term, {})


# --------------------------------------------------------------------- #
# Fixture op decls present + compile to expected shapes
# --------------------------------------------------------------------- #


def test_fixture_declares_add_and_relu():
    target = build_samsung_target()
    add = target.op("ADD")
    relu = target.op("RELU")

    add_pat = compile_op_pattern(add)
    assert add_pat.body.op == "add"
    assert add_pat.param_names == ["x", "y"]

    relu_pat = compile_op_pattern(relu)
    assert relu_pat.body.op == "max"
    assert relu_pat.param_names == ["x"]
