# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Release gate G2: store coverage and accumulator initialization.

Every store of a matched kernel must be implemented (matched op, copy of a
match result, supported accumulator initializer, or a local consumed only by
matched ops); anything else is an UnsupportedWorkloadError.  Backends that zero
their accumulator reject a caller-carried accumulator.  Virtual backend only.
"""

import importlib.util
from collections import Counter
from pathlib import Path

import pytest

import allo
from allo.dataflow import region as _df_region
from allo.ir.types import float16 as fp16
from allo.pim.costs import samsung_cost
from allo.pim.targets import build_samsung_target
from allo.spmw_codegen import compile_for_target
from allo.spmw_match import UnsupportedWorkloadError
from allo.spmw_match_engine import match_workload

M, K = 128, 16


@_df_region()
def gemv_small(W: fp16[M, K], x: fp16[K], y: fp16[M]):
    @allo.work(mapping=[16, 8], args=[W, x, y])
    def gemv(local_W: fp16[M, K], local_x: fp16[K], local_y: fp16[M]):
        pid, uid = allo.get_wid()
        row0 = pid * 8 + uid
        acc: fp16 = 0
        for k in range(K):
            acc += local_W[row0, k] * local_x[k]
        local_y[row0] = acc


@_df_region()
def renamed_small(A: fp16[M, K], v: fp16[K], out: fp16[M]):
    @allo.work(mapping=[16, 8], args=[A, v, out])
    def matvec(mat: fp16[M, K], vec: fp16[K], res: fp16[M]):
        g, l = allo.get_wid()
        r = g * 8 + l
        s: fp16 = 0
        for j in range(K):
            s += mat[r, j] * vec[j]
        res[r] = s


@_df_region()
def gemv_half_init(W: fp16[M, K], x: fp16[K], y: fp16[M]):
    @allo.work(mapping=[16, 8], args=[W, x, y])
    def gemv(local_W: fp16[M, K], local_x: fp16[K], local_y: fp16[M]):
        pid, uid = allo.get_wid()
        row0 = pid * 8 + uid
        acc: fp16 = 0.5
        for k in range(K):
            acc += local_W[row0, k] * local_x[k]
        local_y[row0] = acc


@_df_region()
def gemv_init_in_loop(W: fp16[M, K], x: fp16[K], y: fp16[M]):
    @allo.work(mapping=[16, 8], args=[W, x, y])
    def gemv(local_W: fp16[M, K], local_x: fp16[K], local_y: fp16[M]):
        pid, uid = allo.get_wid()
        row0 = pid * 8 + uid
        acc: fp16 = 0
        for k in range(K):
            acc = 0
            acc += local_W[row0, k] * local_x[k]
        local_y[row0] = acc


@_df_region()
def gemv_div(W: fp16[M, K], x: fp16[K], y: fp16[M]):
    @allo.work(mapping=[16, 8], args=[W, x, y])
    def gemv(local_W: fp16[M, K], local_x: fp16[K], local_y: fp16[M]):
        pid, uid = allo.get_wid()
        row0 = pid * 8 + uid
        acc: fp16 = 0
        for k in range(K):
            acc += local_W[row0, k] * local_x[k]
        local_y[row0] = acc / local_x[0]


@_df_region()
def gemv_sub_after_copy(W: fp16[M, K], x: fp16[K], y: fp16[M]):
    @allo.work(mapping=[16, 8], args=[W, x, y])
    def gemv(local_W: fp16[M, K], local_x: fp16[K], local_y: fp16[M]):
        pid, uid = allo.get_wid()
        row0 = pid * 8 + uid
        acc: fp16 = 0
        for k in range(K):
            acc += local_W[row0, k] * local_x[k]
        local_y[row0] = acc
        local_y[row0] = local_y[row0] - local_x[0]


@_df_region()
def gemv_carried(W: fp16[M, K], x: fp16[K], y: fp16[M]):
    @allo.work(mapping=[16, 8], args=[W, x, y])
    def gemv(local_W: fp16[M, K], local_x: fp16[K], local_y: fp16[M]):
        pid, uid = allo.get_wid()
        row0 = pid * 8 + uid
        for k in range(K):
            local_y[row0] += local_W[row0, k] * local_x[k]


def _compile(workload):
    return allo.compile(workload, build_samsung_target(), samsung_cost, backend="virtual")


def _paper_gemv():
    path = Path(__file__).resolve().parent / "samsung_hbm_pim" / "gemv" / "workload.py"
    spec = importlib.util.spec_from_file_location("match_coverage_samsung_gemv", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CASES["gemv_m4096_k256"][0]


def _classification(trace):
    """Rename-invariant summary: categories per store and per-match facts."""
    stores = Counter(store.category for store in trace.coverage.stores)
    matches = Counter(
        (
            match.target_op_name,
            match.extra.get("accumulator_init"),
            tuple(index for _name, index in match.extra.get("output_sinks", ())),
        )
        for match in trace.matches
    )
    return stores, matches


def test_paper_gemv_zero_init_and_output_sink():
    compiled = _compile(_paper_gemv())
    trace = compiled.trace
    assert trace.coverage.errors == []
    assert {match.target_op_name for match in trace.matches} == {"MAC"}
    assert {match.extra["accumulator_init"] for match in trace.matches} == {"zero"}
    sinks = {
        sink for match in trace.matches for sink in match.extra["output_sinks"]
    }
    assert {name for name, _index in sinks} == {"local_y"}
    # One distinct output row base per work item.
    assert len(sinks) == len(trace.matches) == 128
    assert Counter(store.category for store in trace.coverage.stores) == {
        "local": 128,  # row0 index precomputation
        "accumulator_init": 128,  # acc = 0
        "matched": 128,  # acc += W * x
        "copy": 128,  # y[row0 + i] = acc
    }
    assert compiled.estimate().cycles > 0


def test_small_gemv_sinks_name_output_rows():
    trace = _compile(gemv_small).trace
    by_work = {match.func_name: match for match in trace.matches}
    assert by_work["gemv_0_0"].extra["output_sinks"] == [("local_y", "0")]
    assert by_work["gemv_1_3"].extra["output_sinks"] == [("local_y", "11")]


def test_rename_only_variant_has_same_classification():
    assert _classification(_compile(gemv_small).trace) == _classification(
        _compile(renamed_small).trace
    )
    sinks = {
        name
        for match in _compile(renamed_small).trace.matches
        for name, _index in match.extra["output_sinks"]
    }
    assert sinks == {"res"}


def test_nonzero_accumulator_initializer_is_rejected():
    with pytest.raises(UnsupportedWorkloadError) as info:
        _compile(gemv_half_init)
    message = str(info.value)
    assert "initializer" in message
    assert "nonzero constant 0.5" in message
    assert "accumulator 'acc' of MAC" in message
    assert "literal 0 before the reduction loop" in message


def test_initializer_inside_reduction_loop_is_rejected():
    with pytest.raises(UnsupportedWorkloadError, match="inside the reduction loop"):
        _compile(gemv_init_in_loop)


def test_unmatched_division_is_rejected():
    with pytest.raises(UnsupportedWorkloadError) as info:
        _compile(gemv_div)
    message = str(info.value)
    assert "div(acc[], local_x[0])" in message
    assert "arith.divf" in message
    assert "'local_y[0]'" in message
    assert "function 'gemv_0_0'" in message
    assert "loop nest [] (no enclosing loop)" in message
    assert "Target ops considered: ADD, MAC, MAD, MUL, RELU" in message


def test_unmatched_update_after_copy_is_rejected():
    with pytest.raises(UnsupportedWorkloadError) as info:
        _compile(gemv_sub_after_copy)
    assert "sub(local_y[0], local_x[0])" in str(info.value)
    assert "arith.subf" in str(info.value)


def test_carried_accumulator_rejected_on_samsung():
    target = build_samsung_target()
    trace = match_workload(target, allo.customize(gemv_carried, enable_tensor=False).module)
    assert trace.coverage.errors == []
    assert {match.extra["accumulator_init"] for match in trace.matches} == {"carried"}
    with pytest.raises(NotImplementedError) as info:
        _compile(gemv_carried)
    message = str(info.value)
    assert "accumulator_init='carried'" in message
    assert "zeroes its accumulator" in message
    assert "y[i] += ..." in message


def test_compile_for_target_rechecks_an_unchecked_trace():
    target = build_samsung_target()
    module = allo.customize(gemv_div, enable_tensor=False).module
    trace = match_workload(target, module)
    trace.coverage = None
    with pytest.raises(UnsupportedWorkloadError, match="arith.divf"):
        compile_for_target(
            target, trace, backend="virtual", module=module
        )
