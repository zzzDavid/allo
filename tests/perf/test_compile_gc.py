# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The compile heap scope preserves caller state and MLIR wrapper lifetimes."""
import gc
import threading
import weakref

import pytest
import allo
from allo.compiler import _compile_heap_scope
from allo._mlir.ir import Context
from allo.ir.types import float16
from allo.pim.costs import apu_v1_cost
from allo.pim.targets import build_apu_v1_target


class Cycle:
    def __init__(self):
        self.link = self


def contraction(W: float16[16, 16], x: float16[16], y: float16[16]):
    for i in range(16):
        acc: float16 = 0.0
        for k in allo.reduction(16):
            acc += W[i, k] * x[k]
        y[i] = acc


def test_nested_scope_sweeps_promoted_cycles_after_exception(monkeypatch):
    from allo.ir.builder import _loop_gc_generation

    real_collect = gc.collect
    calls = []

    def collect(generation=2):
        calls.append(generation)
        return real_collect(generation)

    monkeypatch.setattr(gc, "collect", collect)
    enabled, thresholds = gc.isenabled(), gc.get_threshold()
    with pytest.raises(RuntimeError, match="scope failure"):
        with _compile_heap_scope():
            assert _loop_gc_generation.get() == 0
            with _compile_heap_scope():
                wrapper = Cycle()
                wrapper.context = Context()
                ref = weakref.ref(wrapper.context)
                # Promote while reachable, then leave an unreachable old cycle.
                real_collect()
                del wrapper
            assert _loop_gc_generation.get() == 0
            assert calls == []
            assert ref() is not None
            raise RuntimeError("scope failure")
    assert calls == [2]
    assert ref() is None
    assert _loop_gc_generation.get() == 2
    assert (gc.isenabled(), gc.get_threshold()) == (enabled, thresholds)


def test_caller_frozen_heap_is_not_extended_or_thawed():
    old = Cycle()
    old_ref = weakref.ref(old)
    gc.freeze()
    try:
        newer = Cycle()
        newer_ref = weakref.ref(newer)
        del old, newer
        with _compile_heap_scope():
            with _compile_heap_scope():
                gc.collect()
                assert old_ref() is not None
                assert newer_ref() is None
        gc.collect()
        assert old_ref() is not None
        assert gc.get_freeze_count() > 0
    finally:
        gc.unfreeze()
    gc.collect()
    assert old_ref() is None


def test_scope_policy_is_thread_local():
    from allo.ir.builder import _loop_gc_generation

    observed = []

    def worker():
        observed.append(_loop_gc_generation.get())
        with _compile_heap_scope():
            observed.append(_loop_gc_generation.get())
        observed.append(_loop_gc_generation.get())

    with _compile_heap_scope():
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(10)
        assert not thread.is_alive()
        assert _loop_gc_generation.get() == 0
    assert observed == [2, 0, 2]
    assert _loop_gc_generation.get() == 2


def test_public_compile_error_restores_gc_state():
    from allo.ir.builder import _loop_gc_generation

    enabled = gc.isenabled()
    gc.disable()
    try:
        with pytest.raises(TypeError, match="workload must"):
            allo.compile(object(), build_apu_v1_target())
        assert not gc.isenabled()
        assert _loop_gc_generation.get() == 2
    finally:
        if enabled:
            gc.enable()


def test_new_mlir_context_cycles_collect_inside_scope():
    from allo.ir.builder import _loop_gc_generation

    gc.collect()
    count = Context._get_live_count()
    with _compile_heap_scope():
        context = Context()
        wrapper = Cycle()
        wrapper.context = context
        ref = weakref.ref(context)
        del wrapper, context
        gc.collect(_loop_gc_generation.get())
        assert ref() is None
        assert Context._get_live_count() == count


def test_compiled_artifact_survives_thaw_and_releases_context():
    gc.collect()
    count = Context._get_live_count()
    compiled = allo.compile(
        contraction, build_apu_v1_target(), apu_v1_cost, backend="functional"
    )
    module = compiled.schedule.module
    ref = weakref.ref(module.context)
    gc.collect()
    assert module.operation.verify()
    import numpy as np

    matrix = np.eye(16, dtype=np.float16)
    vector = np.arange(16, dtype=np.float16)
    output = np.full(16, 9, dtype=np.float16)
    result = compiled(matrix, vector, output)
    assert result.kind == "functional"
    np.testing.assert_array_equal(output.view(np.uint16), vector.view(np.uint16))
    del result, compiled, module
    gc.collect()
    assert ref() is None
    assert Context._get_live_count() == count
