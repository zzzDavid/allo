# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SPEC-004 multi-output / multi-term matcher guard tests (sub-spec D5).

Guards the additive add-chain flatten in `_try_match_at_store` and the
stage-decomposition convention for the multi-output Tier-1 family. The
load-bearing anchor is `test_single_output_gemv_byte_identical`: the flatten is
inert on the canonical single-`mul` GEMV path.

These are matcher-level tests (trace structure), no simulator. They build small
shapes so they are fast.
"""

from __future__ import annotations

import allo
from allo.ir.types import float32 as fp16
from allo.dataflow import region as _df_region

from _fixtures import build_samsung_target


def _trace(region):
    target = build_samsung_target()
    sch = allo.customize(region, enable_tensor=False)
    return allo.match_workload(target, sch.module)


def _funcs(trace):
    return sorted({m.func_name for m in trace.matches})


# --------------------------------------------------------------------- #
# D3: gemver multi-term reduction flatten
# --------------------------------------------------------------------- #


def test_gemver_multi_term_accumulate():
    """gemver's two-`mul` rank-1 store `A += u1*v1 + u2*v2` flattens into TWO
    MAC matches, both writing `A`, both with a loop-carried accumulator -- the
    additive add-chain flatten (the single-`mul` MAC pattern cannot match the
    two-`mul` term)."""
    N = 16

    @_df_region()
    def _top(A: fp16[N, N], u1: fp16[N], v1: fp16[N], u2: fp16[N], v2: fp16[N]):
        @allo.work(mapping=[1], args=[A, u1, v1, u2, v2])
        def rank1(lA: fp16[N, N], lu1: fp16[N], lv1: fp16[N], lu2: fp16[N], lv2: fp16[N]):
            for i in range(N):
                for j in range(N):
                    lA[i, j] = lA[i, j] + lu1[i] * lv1[j] + lu2[i] * lv2[j]

    trace = _trace(_top)
    macs = trace.by_target_op("MAC")
    assert len(macs) == 2, f"expected 2 MAC matches from the flatten, got {len(macs)}"
    # Both write the same result memref (the A accumulator).
    results = {m.result_memref_name for m in macs}
    assert results == {"lA"}, results
    # Both carry a loop-carried accumulator (the A += contract).
    for m in macs:
        assert any(o.is_loop_carried for o in m.operands), m


def test_gemver_lowers_via_two_func_decomposition():
    """The suite's gemver (rank-1 split into two single-`mul` funcs + two GEMVs)
    lowers and compiles -- 4 MAC across 4 buckets, no role-collision raise. (The
    single-statement two-`mul` form is recognized by the matcher above but cannot
    lower past `_trace_memrefs_by_role` in one bucket; the two-func authoring is
    the form that lowers today.)"""
    import sys, pathlib

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "pim"))
    from workloads import gemver  # noqa: E402

    target = build_samsung_target()
    sch = allo.customize(gemver.build(), enable_tensor=False)
    trace = allo.match_workload(target, sch.module)
    assert len(trace.by_target_op("MAC")) == 4, trace.by_target_op("MAC")
    assert len(_funcs(trace)) == 4, _funcs(trace)
    allo.compile_for_target(target, trace)  # must not raise


# --------------------------------------------------------------------- #
# D2: stage-decomposition convention (mvt / atax / bicg) -- no matcher change
# --------------------------------------------------------------------- #


def _check_two_bucket(modname):
    import sys, pathlib

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "pim"))
    mod = __import__(f"workloads.{modname}", fromlist=["build"])
    target = build_samsung_target()
    sch = allo.customize(mod.build(), enable_tensor=False)
    trace = allo.match_workload(target, sch.module)
    assert len(trace.by_target_op("MAC")) == 2, (modname, trace.by_target_op("MAC"))
    assert len(_funcs(trace)) == 2, (modname, _funcs(trace))
    allo.compile_for_target(target, trace)  # no role-collision raise


def test_mvt_two_single_output_buckets():
    _check_two_bucket("mvt")


def test_atax_two_pass_buckets():
    _check_two_bucket("atax")


def test_bicg_two_pass_buckets():
    _check_two_bucket("bicg")


# --------------------------------------------------------------------- #
# D4 / gate 4: the additive default is byte-identical on the single-output path
# --------------------------------------------------------------------- #


def test_single_output_gemv_byte_identical():
    """A canonical single-`mul` GEMV trace is unchanged by the flatten: exactly
    one MAC match, single bucket, the acc loop-carried -- the flatten never
    fires (the single-term path matches first). The additive-default anchor."""
    N = 16

    @_df_region()
    def _top(W: fp16[N, N], x: fp16[N], y: fp16[N]):
        @allo.work(mapping=[1], args=[W, x, y])
        def gv(lW: fp16[N, N], lx: fp16[N], ly: fp16[N]):
            for i in range(N):
                acc: fp16 = 0
                for k in range(N):
                    acc += lW[i, k] * lx[k]
                ly[i] = acc

    trace = _trace(_top)
    macs = trace.by_target_op("MAC")
    assert len(macs) == 1, f"single-output GEMV must emit exactly 1 MAC, got {len(macs)}"
    # The canonical GEMV result memref is the accumulator "acc" (matching the
    # landed test_match_gemv.py:144 contract) -- unchanged by the flatten.
    assert macs[0].result_memref_name == "acc", macs[0].result_memref_name
    assert any(o.is_loop_carried for o in macs[0].operands)


if __name__ == "__main__":
    test_gemver_multi_term_accumulate()
    test_gemver_lowers_via_two_func_decomposition()
    test_mvt_two_single_output_buckets()
    test_atax_two_pass_buckets()
    test_bicg_two_pass_buckets()
    test_single_output_gemv_byte_identical()
    print("ALL PASSED")
