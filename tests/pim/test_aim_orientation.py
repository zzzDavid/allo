# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Release ledger X-41 (third increment): AiM's storage orientation for a
two-sided tie is a cost-ranked candidate decision, not the shared roles'
pattern-order naming.

A batched GEMV ``Y[b, i] = sum_k W[i, k] X[b, k]`` reads the reduction loop
and one result loop in each input, so neither input's index set contains
the other's: either may be the bank-resident matrix with the other streamed
through the GB as the batch of vectors.  Both orientations are enumerated
(`spmw_match_engine.contraction_orientations`), each with its own shape,
layouts and knobs, and ranked by cost, so the selected program does not
depend on which input is written first; ``layout="aim_bank_operand=<memref>"``
forces one, and the realization check accepts both.  A GEMV (index
containment) keeps its single orientation."""

from __future__ import annotations

import pytest

import allo
from allo.dataflow import region
from allo.ir.types import bfloat16
from allo.pim.costs import aim_cost
from allo.pim.targets import build_aim_target
from allo.spmw_match_engine import contraction_orientations

M, K, B = 32, 256, 4


@region()
def w_first(W: bfloat16[M, K], X: bfloat16[B, K], Y: bfloat16[B, M]):
    @allo.work(mapping=[1], args=[W, X, Y])
    def dot(a: bfloat16[M, K], x: bfloat16[B, K], c: bfloat16[B, M]):
        for bb in range(B):
            for i in range(M):
                acc: bfloat16 = 0
                for k in range(K):
                    acc += a[i, k] * x[bb, k]
                c[bb, i] = acc


@region()
def x_first(W: bfloat16[M, K], X: bfloat16[B, K], Y: bfloat16[B, M]):
    @allo.work(mapping=[1], args=[W, X, Y])
    def dot(a: bfloat16[M, K], x: bfloat16[B, K], c: bfloat16[B, M]):
        for bb in range(B):
            for i in range(M):
                acc: bfloat16 = 0
                for k in range(K):
                    acc += x[bb, k] * a[i, k]
                c[bb, i] = acc


@region()
def gemv(W: bfloat16[M, K], x: bfloat16[K], y: bfloat16[M]):
    @allo.work(mapping=[1], args=[W, x, y])
    def dot(a: bfloat16[M, K], b: bfloat16[K], c: bfloat16[M]):
        for i in range(M):
            acc: bfloat16 = 0
            for k in range(K):
                acc += a[i, k] * b[k]
            c[i] = acc


def _compile(workload, **kwargs):
    return allo.compile(
        workload,
        build_aim_target(),
        aim_cost,
        backend="virtual",
        reduction_order="reassociate",
        **kwargs,
    )


def _selection(compiled):
    kernel = compiled.placement_ranking.kernels[0]
    chosen = [s for s in kernel.scores if s[0] == kernel.chosen_position][0]
    placement = next(
        pl for pos, pl in kernel.candidates if pos == kernel.chosen_position
    )
    (operation,) = [
        op
        for op in compiled.compiled.layout_ctx.lowered_operations
        if type(op).__name__ == "AimContraction"
    ]
    return {
        "cycles": chosen[1],
        "mode": chosen[2],
        "matrix": placement.extra["aim_orientation"]["matrix"],
        "outputs": operation.outputs,
        "batches": operation.batches,
        "cmds": [str(c) for c in compiled.compiled.cmds],
        "rejections": len(kernel.rejections),
    }


@pytest.fixture(scope="module")
def both():
    return {"w_first": _compile(w_first), "x_first": _compile(x_first)}


def test_batched_gemv_has_two_orientations_whichever_input_is_first(both):
    for compiled in both.values():
        match = compiled.trace.matches[0]
        orientations = contraction_orientations(match)
        assert [o["by"] for o in orientations] == ["roles", "alternative"]
        assert {(o["matrix"], o["vector"]) for o in orientations} == {
            ("a", "x"),
            ("x", "a"),
        }
        # Under either orientation the result loops split into the loop
        # indexing the vector (batch) and the loop indexing the matrix.
        by_matrix = {
            o["matrix"]: (len(o["output"]), len(o["batch"])) for o in orientations
        }
        assert by_matrix == {"a": (1, 1), "x": (1, 1)}
        kernel = compiled.placement_ranking.kernels[0]
        families = {
            f
            for _pos, pl in kernel.candidates
            for f in pl.extra.get("candidate_families", ())
            if f.startswith("aim_bank_operand=")
        }
        assert families == {"aim_bank_operand=a", "aim_bank_operand=x"}


def test_operand_order_does_not_change_the_selected_program(both):
    first, second = _selection(both["w_first"]), _selection(both["x_first"])
    assert (first["matrix"], first["outputs"], first["batches"], first["cycles"]) == (
        second["matrix"],
        second["outputs"],
        second["batches"],
        second["cycles"],
    )
    assert first["cmds"] == second["cmds"]
    assert first["rejections"] == second["rejections"] == 0
    # The cheaper orientation keeps the 4-row operand resident and streams
    # the 32 rows as the batch: (outputs, batches) = (4, 32) with x in the
    # banks, in both spellings.
    assert (first["matrix"], first["outputs"], first["batches"]) == ("x", 4, 32)


@pytest.mark.parametrize("bank_operand", ["a", "x"])
def test_forcing_an_orientation_realizes_it_through_the_realization_check(bank_operand):
    compiled = _compile(w_first, layout=f"aim_bank_operand={bank_operand}")
    selection = _selection(compiled)
    expected = {"a": (32, 4), "x": (4, 32)}[bank_operand]
    assert (selection["matrix"], selection["outputs"], selection["batches"]) == (
        bank_operand,
        *expected,
    )
    assert selection["rejections"] == 0
    # The forced alternative is the dearer one here: the default selects x.
    default = _selection(_compile(w_first))
    if bank_operand == "a":
        assert selection["cycles"] > default["cycles"]
        assert selection["cmds"] != default["cmds"]
    else:
        assert selection["cycles"] == default["cycles"]


def test_gemv_keeps_a_single_orientation():
    compiled = _compile(gemv)
    (orientation,) = contraction_orientations(compiled.trace.matches[0])
    assert (orientation["matrix"], orientation["vector"], orientation["by"]) == (
        "a",
        "b",
        "roles",
    )
    kernel = compiled.placement_ranking.kernels[0]
    families = {
        f
        for _pos, pl in kernel.candidates
        for f in pl.extra.get("candidate_families", ())
        if f.startswith("aim_bank_operand=")
    }
    assert families == {"aim_bank_operand=a"}
    assert not any("transposed" in pl.mode for _pos, pl in kernel.candidates)
