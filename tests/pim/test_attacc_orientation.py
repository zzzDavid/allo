# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Release ledger X-41 (second increment): AttAcc's storage orientation is an
explicit candidate decision, not the shared roles' pattern-order naming.

For a GEMM (each input reads the reduction loop and one result loop) both
orientations are legal realizations: either input may be the bank-resident
matrix with the other streamed through the GEMV buffer.  Both are
enumerated, priced with their tilings and ranked, so the selected program
does not depend on which input is written first; forcing the other
orientation realizes it, and the realization check accepts both.  A GEMV
(index containment) keeps its single orientation."""

from __future__ import annotations

import pytest

import allo
from allo.dataflow import region
from allo.ir.types import float16
from allo.pim.attacc_tiling import attacc_orientations
from allo.pim.costs import attacc_cost
from allo.pim.targets import build_attacc_target

P, R, Q = 32, 64, 8


@region()
def left_first(A: float16[P, R], B: float16[R, Q], C: float16[P, Q]):
    @allo.work(mapping=[1], args=[A, B, C])
    def body(a: float16[P, R], b: float16[R, Q], c: float16[P, Q]):
        for row in range(P):
            for col in range(Q):
                acc: float16 = 0
                for red in range(R):
                    acc += a[row, red] * b[red, col]
                c[row, col] = acc


@region()
def right_first(A: float16[P, R], B: float16[R, Q], C: float16[P, Q]):
    @allo.work(mapping=[1], args=[A, B, C])
    def body(a: float16[P, R], b: float16[R, Q], c: float16[P, Q]):
        for row in range(P):
            for col in range(Q):
                acc: float16 = 0
                for red in range(R):
                    acc += b[red, col] * a[row, red]
                c[row, col] = acc


@region()
def gemv(W: float16[P, R], x: float16[R], y: float16[P]):
    @allo.work(mapping=[1], args=[W, x, y])
    def body(w: float16[P, R], v: float16[R], out: float16[P]):
        for row in range(P):
            acc: float16 = 0
            for red in range(R):
                acc += w[row, red] * v[red]
            out[row] = acc


def _compile(workload, **kwargs):
    return allo.compile(
        workload,
        build_attacc_target(),
        attacc_cost,
        backend="virtual",
        reduction_order="reassociate",
        **kwargs,
    )


def _selected(compiled):
    kernel = compiled.compiled.layout.extra["attacc_kernel"]
    tiling = compiled.compiled.layout.extra["attacc_tiling"]
    return kernel.manifest(), tiling.dims, compiled.placement_ranking.program_cycles


@pytest.fixture(scope="module")
def both():
    return {"left": _compile(left_first), "right": _compile(right_first)}


def test_gemm_has_two_orientations_whichever_input_is_first(both):
    for compiled in both.values():
        match = compiled.trace.matches[0]
        orientations = attacc_orientations([match])
        assert [o["by"] for o in orientations] == ["roles", "alternative"]
        assert {(o["matrix"], o["vector"]) for o in orientations} == {
            ("a", "b"),
            ("b", "a"),
        }
        kernel = compiled.placement_ranking.kernels[0]
        modes = {pl.mode.split("+")[0] for _pos, pl in kernel.candidates}
        assert modes == {"attacc_gemv", "attacc_gemv_transposed"}
        kernels = {
            (pl.extra["attacc_kernel"].matrix, pl.extra["attacc_kernel"].dims)
            for _pos, pl in kernel.candidates
        }
        assert kernels == {("a", (8, 64, 32)), ("b", (32, 64, 8))}


def test_operand_order_does_not_change_the_selected_realization(both):
    left, right = _selected(both["left"]), _selected(both["right"])
    assert left == right
    kernel, _tiling, _cycles = left
    assert kernel["kind"] == "gemv" and kernel["matrix"] in ("a", "b")
    # The two orientations are priced differently; the cheaper one is chosen
    # in both spellings, and no candidate was rejected at realization.
    for compiled in both.values():
        ranking = compiled.placement_ranking.kernels[0]
        assert not [r for r in ranking.rejections if "realization" in r[1]]
        chosen = [s for s in ranking.scores if s[0] == ranking.chosen_position][0]
        assert chosen[1] == min(s[1] for s in ranking.scores)


@pytest.mark.parametrize("mode", ["attacc_gemv", "attacc_gemv_transposed"])
def test_forcing_an_orientation_realizes_it_and_passes_the_realization_check(mode):
    compiled = _compile(left_first, layout=mode)
    kernel = compiled.compiled.layout.extra["attacc_kernel"]
    expected = {
        "attacc_gemv": ("a", (8, 64, 32)),
        "attacc_gemv_transposed": ("b", (32, 64, 8)),
    }
    assert (kernel.matrix, kernel.dims) == expected[mode]
    programs = compiled.compiled.layout_ctx.programs
    assert len(programs) == 1 and programs[0].kernel == kernel
    # The declared storage follows the orientation: the matrix in the
    # banks, the other input through the GEMV buffer.
    from allo.spmw_candidate import candidate_effects

    ranking = compiled.placement_ranking.kernels[0]
    selected = next(
        pl for pos, pl in ranking.candidates if pos == ranking.chosen_position
    )
    storage = {v.value: v.storage for v in candidate_effects(selected).value_layouts}
    other = "b" if kernel.matrix == "a" else "a"
    assert storage[kernel.matrix] == "banks" and storage[other] == "gemv_buffer"


def test_gemv_keeps_a_single_orientation():
    compiled = _compile(gemv)
    match = compiled.trace.matches[0]
    (orientation,) = attacc_orientations([match])
    assert (orientation["matrix"], orientation["vector"], orientation["by"]) == (
        "w",
        "v",
        "roles",
    )
    kernel = compiled.compiled.layout.extra["attacc_kernel"]
    assert (kernel.matrix, kernel.dims) == ("w", (1, 64, 32))
    modes = {
        pl.mode.split("+")[0]
        for _pos, pl in compiled.placement_ranking.kernels[0].candidates
    }
    assert modes == {"attacc_gemv"}
