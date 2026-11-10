# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Release ledger V1-19: the APU v1 functional host reference runs the source
workload with scalar induction variables for one-dimensional Allo loops,
rebound locally to the call, whatever spelling the source uses for Allo's
``grid`` / ``reduction`` (module-qualified, imported, aliased), and without
touching process-wide bindings (the round 103 form patched module attributes,
so ``from allo import grid`` sources still failed under NumPy 2.4+ and
overlapping references left the module altered; the reviewer's
``reviewer-r103/probe_imports.py``).  The deprecation NumPy 2.4 turned into an
error is treated as an error here so the checks discriminate on every NumPy
version."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import allo
from allo import dsl as allo_dsl
from allo import grid, reduction
from allo import reduction as red
from allo.ir.types import uint16
from allo.pim.apu_v1_vector_program import _scalar_loop_view
from allo.pim.targets import build_apu_v1_target

M, K, N = 64, 32, 64


def qualified(left: uint16[M, K], right: uint16[K, N], result: uint16[M, N]):
    for row, column in allo.grid(M, N):
        for depth in allo.reduction(K):
            result[row, column] += left[row, depth] * right[depth, column]


def imported(left: uint16[M, K], right: uint16[K, N], result: uint16[M, N]):
    for row, column in grid(M, N):
        for depth in reduction(K):
            result[row, column] += left[row, depth] * right[depth, column]


def aliased(left: uint16[M, K], right: uint16[K, N], result: uint16[M, N]):
    for row, column in allo_dsl.grid(M, N):
        for depth in red(K):
            result[row, column] += left[row, depth] * right[depth, column]


ORIGINAL_GRID, ORIGINAL_REDUCTION = allo.grid, allo.reduction
observed_during_run = []


def record_bindings():
    """Called from inside a workload: the process-wide bindings as another
    module sees them while the reference runs."""
    observed_during_run.append(
        (allo.grid is ORIGINAL_GRID, allo_dsl.reduction is ORIGINAL_REDUCTION)
    )


def observing(left: uint16[M, K], right: uint16[K, N], result: uint16[M, N]):
    record_bindings()
    for row, column in allo.grid(M, N):
        for depth in allo.reduction(K):
            result[row, column] += left[row, depth] * right[depth, column]


def _inputs():
    left = np.ones((M, K), dtype=np.uint16)
    right = np.ones((K, N), dtype=np.uint16)
    return left, right, np.zeros((M, N), dtype=np.uint16)


def _run_functional(workload):
    compiled = allo.compile(
        workload,
        build_apu_v1_target(),
        None,
        backend="functional",
        layout="temporal_svp",
    )
    left, right, result = _inputs()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        run = compiled(left, right, result)
    assert run.kind == "functional"
    return result


@pytest.mark.parametrize(
    "workload", [qualified, imported, aliased], ids=lambda f: f.__name__
)
def test_every_spelling_of_the_loop_ranges_executes(workload):
    result = _run_functional(workload)
    np.testing.assert_array_equal(result, np.full((M, N), K, dtype=np.uint16))
    assert allo.grid is ORIGINAL_GRID and allo.reduction is ORIGINAL_REDUCTION
    assert allo_dsl.grid is ORIGINAL_GRID and allo_dsl.reduction is ORIGINAL_REDUCTION


def test_the_scalar_loops_are_local_to_the_workload_call():
    """While a rebound workload runs, another module still sees the original
    bindings (``observing`` calls a helper of this module from inside its
    loops, so it is run through the rebound view directly rather than
    compiled); the workload's own view yields scalars for one dimension and
    tuples for two, and the module it sees delegates everything else."""
    observed_during_run.clear()
    left, right, result = _inputs()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        _scalar_loop_view(observing)(left, right, result)
    np.testing.assert_array_equal(result, np.full((M, N), K, dtype=np.uint16))
    assert observed_during_run == [(True, True)]
    view = _scalar_loop_view(imported)
    assert view.__globals__ is not imported.__globals__
    assert imported.__globals__["grid"] is ORIGINAL_GRID
    assert list(view.__globals__["reduction"](3)) == [0, 1, 2]
    assert list(view.__globals__["grid"](2, 2)) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert view.__globals__["allo"].get_wid is allo.get_wid
    with pytest.raises(AttributeError, match="read-only"):
        view.__globals__["allo"].grid = None
    assert list(ORIGINAL_REDUCTION(2)) == [(0,), (1,)]


def test_overlapping_references_and_exceptions_leave_the_bindings_untouched():
    views = [_scalar_loop_view(qualified), _scalar_loop_view(imported)]
    left, right, _ = _inputs()
    results = [np.zeros((M, N), dtype=np.uint16) for _ in views]
    # Interleave the two rebound workloads (neither holds any global state).
    views[0](left, right, results[0])
    views[1](left, right, results[1])
    for result in results:
        np.testing.assert_array_equal(result, np.full((M, N), K, dtype=np.uint16))

    def failing(left, right, result):
        for row in allo.grid(M):
            raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        _scalar_loop_view(failing)(left, right, results[0])
    assert allo.grid is ORIGINAL_GRID and allo.reduction is ORIGINAL_REDUCTION
    assert allo_dsl.grid is ORIGINAL_GRID and allo_dsl.reduction is ORIGINAL_REDUCTION
    assert list(allo.grid(2)) == [(0,), (1,)]


def test_a_non_function_workload_is_refused_with_its_type():
    with pytest.raises(RuntimeError, match="Python function; got int"):
        _scalar_loop_view(3)


def make_direct():
    """A factory whose contraction captures locally imported ranges."""
    from allo import grid as local_grid
    from allo import reduction as local_reduction

    def contraction(left: uint16[M, K], right: uint16[K, N], result: uint16[M, N]):
        for row, column in local_grid(M, N):
            for depth in local_reduction(K):
                result[row, column] += left[row, depth] * right[depth, column]

    return contraction


def make_module():
    """A factory whose contraction captures a locally imported module."""
    import allo as local_allo

    def contraction(left: uint16[M, K], right: uint16[K, N], result: uint16[M, N]):
        for row, column in local_allo.grid(M, N):
            for depth in local_allo.reduction(K):
                result[row, column] += left[row, depth] * right[depth, column]

    return contraction


def nested_dsl(left: uint16[M, K], right: uint16[K, N], result: uint16[M, N]):
    for row, column in allo.dsl.grid(M, N):
        for depth in allo.dsl.reduction(K):
            result[row, column] += left[row, depth] * right[depth, column]


UNRELATED = object()


def make_with_unrelated_capture():
    from allo import reduction as local_reduction

    unrelated = UNRELATED

    def contraction(left: uint16[M, K], right: uint16[K, N], result: uint16[M, N]):
        for row, column in allo.grid(M, N):
            for depth in local_reduction(K):
                result[row, column] += left[row, depth] * right[depth, column]
        return unrelated

    return contraction


@pytest.mark.parametrize(
    "factory",
    [make_direct, make_module, lambda: nested_dsl],
    ids=["closure_imports", "closure_module", "allo_dsl_attribute"],
)
def test_closure_captures_and_the_dsl_submodule_follow_the_view(factory):
    """The reviewer's round 104 probes (`reviewer-r104/probe_closures.py`):
    a factory's locally imported ranges or module live in closure cells,
    which the round 104 form reused unchanged, and ``allo.dsl.grid``
    escaped through the module view's delegation.  Both now follow the
    view through ``allo.compile``'s functional route."""
    workload = factory()
    result = _run_functional(workload)
    np.testing.assert_array_equal(result, np.full((M, N), K, dtype=np.uint16))
    assert allo.grid is ORIGINAL_GRID and allo_dsl.reduction is ORIGINAL_REDUCTION


def test_closure_cells_are_fresh_and_unrelated_captures_are_kept():
    workload = make_with_unrelated_capture()
    original_cells = workload.__closure__
    original_contents = [cell.cell_contents for cell in original_cells]
    view = _scalar_loop_view(workload)
    assert view.__closure__ is not original_cells
    assert all(a is not b for a, b in zip(view.__closure__, original_cells))
    # The original cells still hold the original range and the unrelated value.
    assert [cell.cell_contents for cell in original_cells] == original_contents
    assert ORIGINAL_REDUCTION in original_contents and UNRELATED in original_contents
    rebound = {id(c.cell_contents) for c in view.__closure__}
    assert id(UNRELATED) in rebound and id(ORIGINAL_REDUCTION) not in rebound
    left, right, result = _inputs()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert view(left, right, result) is UNRELATED
    np.testing.assert_array_equal(result, np.full((M, N), K, dtype=np.uint16))
