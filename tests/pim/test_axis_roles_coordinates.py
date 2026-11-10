# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The shared role analysis reads resolved coordinates (release ledger X-41,
first increment): an operand indexed by an SSA value that the matcher resolved
to a loop variable plus a constant (``lA[row0 + i, k]``) ranges over that loop
variable, so a contraction's roles resolve the same way whichever input is
written first, and the tie between two inputs that each read the reduction
loop and one result loop is stamped as the pattern-order convention it is."""

from __future__ import annotations

import pytest

import allo
from allo.dataflow import region as _df_region
from allo.ir.types import float16 as fp16
from allo.pim.costs import samsung_cost
from allo.pim.targets import build_samsung_target
from allo.spmw_match import OperandBinding
from allo.spmw_match_engine import operand_index_terms

M, K, ROWS, N = 4096, 256, 32, 4


def _binding(role, indices, coords):
    return OperandBinding(
        role=role, memref_name=role, indices=list(indices), coords=coords
    )


def test_operand_index_terms_read_resolved_coordinates():
    loops = {"%i", "%k", "%j"}
    # lA[row0 + i, k]: the row coordinate is the SSA value %21, resolved to
    # the form i + 992; the reduction coordinate is the loop variable.
    row = ((("%i", 1),), 992)
    red = ((("%k", 1),), 0)
    variables, opaque = operand_index_terms(
        _binding("y", ["%21", "%k"], (row, red)), loops
    )
    assert (variables, opaque) == ({"%i", "%k"}, [])
    # A data-dependent coordinate (no form) stays opaque under its SSA name.
    variables, opaque = operand_index_terms(
        _binding("x", ["%7", "%k"], (None, red)), loops
    )
    assert (variables, opaque) == ({"%k"}, ["%7"])
    # A load-free atom that is not a loop variable is opaque too.
    atom = ((("%base", 1), ("%i", 2)), 0)
    variables, opaque = operand_index_terms(
        _binding("x", ["%9", "%k"], (atom, red)), loops
    )
    assert (variables, opaque) == ({"%i", "%k"}, ["%base"])
    # Without coordinates the textual indices decide, as before.
    variables, opaque = operand_index_terms(_binding("x", ["%k", "%j"], None), loops)
    assert (variables, opaque) == ({"%k", "%j"}, [])
    variables, opaque = operand_index_terms(_binding("x", ["%21", "%k"], None), loops)
    assert (variables, opaque) == ({"%k"}, ["%21"])


@_df_region()
def gemm_vector_first(A: fp16[M, K], B: fp16[K, N], C: fp16[M, N]):
    @allo.work(mapping=[16, 8], args=[A, B, C])
    def kernel(lA: fp16[M, K], lB: fp16[K, N], lC: fp16[M, N]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * ROWS
        for i in range(ROWS):
            for j in range(N):
                acc: fp16 = 0
                for k in range(K):
                    acc += lB[k, j] * lA[row0 + i, k]
                lC[row0 + i, j] = acc


@_df_region()
def gemm_matrix_first(A: fp16[M, K], B: fp16[K, N], C: fp16[M, N]):
    @allo.work(mapping=[16, 8], args=[A, B, C])
    def kernel(lA: fp16[M, K], lB: fp16[K, N], lC: fp16[M, N]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * ROWS
        for i in range(ROWS):
            for j in range(N):
                acc: fp16 = 0
                for k in range(K):
                    acc += lA[row0 + i, k] * lB[k, j]
                lC[row0 + i, j] = acc


@pytest.fixture(scope="module")
def gemm_roles():
    out = {}
    for name, workload in (
        ("vector_first", gemm_vector_first),
        ("matrix_first", gemm_matrix_first),
    ):
        compiled = allo.compile(
            workload,
            build_samsung_target(),
            samsung_cost,
            reduction_order="reassociate",
        )
        match = compiled.trace.matches[0]
        out[name] = (match, dict(match.extra["axis_roles"]))
    return out


def test_gemm_roles_resolve_in_both_spellings(gemm_roles):
    """Before X-41 the vector-first spelling was ``unresolved: vector index
    ['%21'] may depend on result loop %arg3`` while the matrix-first one
    resolved; both now resolve, with the same loop classification up to
    the pattern-order naming of the two inputs."""
    for name, (match, roles) in gemm_roles.items():
        assert roles["unresolved"] is None, (name, roles)
        assert roles["matrix_by"] == "pattern_order", (name, roles)
        loops = [loop[0] for loop in match.enclosing_loops]
        i, j, k = loops
        assert roles["reduction"] == (k,), (name, roles)
        # The result loop indexing the input named ``vector`` is the batch
        # axis, the other the output axis.
        by_role = {op.role: op for op in match.operands}
        vector = by_role[roles["vector"]]
        if vector.memref_name == "lA":
            assert (roles["batch"], roles["output"]) == ((i,), (j,)), (name, roles)
        else:
            assert (roles["batch"], roles["output"]) == ((j,), (i,)), (name, roles)
    # The naming follows the pattern order, as stamped: the first input is
    # the ``matrix`` in both spellings.
    v_match, v_roles = gemm_roles["vector_first"]
    m_match, m_roles = gemm_roles["matrix_first"]
    first = {op.role: op.memref_name for op in v_match.operands}[v_roles["matrix"]]
    assert first == "lB"
    first = {op.role: op.memref_name for op in m_match.operands}[m_roles["matrix"]]
    assert first == "lA"
