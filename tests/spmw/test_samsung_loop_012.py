# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-012 loop-body cell tests: Samsung gemv 4096x1024 + FFN 256-1024-256.

Spmw-side static assertions that pin the loop-012 measurement behaviour
(the simulator-running cycle measurements live in
``test_samsung_batched_gemv.py`` / ``test_samsung_faithful_run_path.py``,
the verifier's domain). Per arch ruling 008:

  * Cell 1 -- gemv 4096x1024 single shape: MATCH at floor. The argmin
    picks the ``dual_fiber+crf_shared`` placement at B=1 (the same
    placement the faithful run path materialises and the simulator times
    at the 15,251-cyc floor). The cost model (a *separate* instrument
    from the faithful simulator) prices that placement's P/E/R closed
    form; this test asserts the argmin placement + the placement-invariant
    P=11368 / R=181 anchors, NOT the simulator's 15,251 (cost model and
    simulator are deliberately different instruments).

  * Cell 2 -- FFN 256-1024-256: MATCH at floor (per leg). Two GEMV legs
    (W1[1024,256]@x + host ReLU, W2[256,1024]@h); host ReLU = 0 PIM cyc.
    The FFN cannot be measured on the reference PIMSimulator (it cores at
    the non-4096 leg shapes -- coder-003 / MANIFEST samsung-ffn
    BLOCKED-SIM), so the cell's verdict is documented-floor, not a sim
    number. This test asserts the structural split into two single-shape
    GEMV legs (the unit the per-leg floor mechanism prices) and records a
    latent batch_dim heuristic false-positive on bare-index GEMV legs
    (flagged to architect; does not affect any committed/measured result
    because the FFN never runs on the sim).
"""

from __future__ import annotations

import allo
from allo.dataflow import region as _df_region
from allo.ir.types import float32 as fp16
from allo.spmw_autoschedule import _bucket_for_autoschedule, _samsung_enumerate
from allo.spmw_cost_models import (
    _parse_loop_bound,
    _samsung_mk,
    _samsung_preload_cycles,
    _samsung_readback_cycles,
    _trace_batch_dim,
)
from allo.spmw_match import MatchTrace

from _fixtures import build_samsung_target

# Samsung gemv floor cell.
GM, GK = 4096, 1024
GROWS = GM // (16 * 8)

# FFN 256-1024-256: in=256, hidden=1024, out=256.
FIN, FHID, FOUT = 256, 1024, 256


@_df_region()
def loop012_gemv_top(W: fp16[GM, GK], x: fp16[GK], y: fp16[GM]):
    @allo.work(mapping=[16, 8], args=[W, x, y])
    def gemv(local_W: fp16[GM, GK], local_x: fp16[GK], local_y: fp16[GM]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * GROWS
        for i in range(GROWS):
            acc: fp16 = 0
            for k in range(GK):
                acc += local_W[row0 + i, k] * local_x[k]
            local_y[row0 + i] = acc


@_df_region()
def ffn_256_1024_256_top(
    W1: fp16[FHID, FIN],
    x: fp16[FIN],
    W2: fp16[FOUT, FHID],
    h: fp16[FHID],
    y: fp16[FOUT],
):
    # Leg 1: h = W1 @ x  (W1[1024,256]). Host ReLU on h between legs is
    # 0 PIM cyc (runs on the host; ruling 008 Cell-2).
    @allo.work(mapping=[1], args=[W1, x, h])
    def ffn_layer1(local_W1: fp16[FHID, FIN], local_x: fp16[FIN], local_h: fp16[FHID]):
        for i1 in range(FHID):
            acc: fp16 = 0
            for k1 in range(FIN):
                acc += local_W1[i1, k1] * local_x[k1]
            local_h[i1] = acc

    # Leg 2: y = W2 @ h  (W2[256,1024]).
    @allo.work(mapping=[1], args=[W2, h, y])
    def ffn_layer2(local_W2: fp16[FOUT, FHID], local_h: fp16[FHID], local_y: fp16[FOUT]):
        for i2 in range(FOUT):
            acc: fp16 = 0
            for k2 in range(FHID):
                acc += local_W2[i2, k2] * local_h[k2]
            local_y[i2] = acc


def _sub_trace(trace: MatchTrace, idx: int) -> MatchTrace:
    _fn, matches = _bucket_for_autoschedule(trace)[idx]
    return MatchTrace(
        target_name=trace.target_name,
        module_name=trace.module_name,
        matches=matches,
    )


# --------------------------------------------------------------------- #
# Cell 1 -- gemv 4096x1024 single-shape: argmin picks the floor placement.
# --------------------------------------------------------------------- #


def test_gemv_single_shape_argmin_is_dual_fiber_at_b1():
    target = build_samsung_target()
    cost_fn = allo.get_cost("kernel_cycles", target)
    sch = allo.customize(loop012_gemv_top, enable_tensor=False)
    trace = allo.match_workload(target, sch.module)
    sub = _sub_trace(trace, 0)
    # The computed output index (row0 + i) is NOT a bare loop var, so the
    # batch heuristic does not fire: single-vector GEMV defaults to B=1.
    assert _trace_batch_dim(sub) == 1
    M_, K_ = _samsung_mk(target, sub)
    assert (M_, K_) == (GM, GK)
    # Placement-invariant phase anchors (calibrated to the faithful run at
    # 4096x1024: P=11368, R=181; report 18 §3 / ruling 008 §1.5).
    assert _samsung_preload_cycles(target, M_, K_) == 11368
    assert _samsung_readback_cycles(target, M_) == 181
    # argmin over all candidates lands on the dual-fiber/shared-CRF
    # placement -- the one the faithful run path materialises (and the sim
    # times at the 15,251 floor). At B=1 the resident flag ties (I4), so
    # the non-resident dual_fiber variant wins (or ties) the argmin.
    cands = _samsung_enumerate(target, sub.matches)
    scored = sorted((cost_fn(sub, c), c.mode) for c in cands)
    best_cost, best_mode = scored[0]
    assert best_mode.startswith("dual_fiber+crf_shared"), best_mode
    # The argmin cost decomposes into P + E + R with a positive exec body.
    E = best_cost - 11368 - 181
    assert E > 0, best_cost


# --------------------------------------------------------------------- #
# Cell 2 -- FFN 256-1024-256: structural two-leg split (per-leg floor).
# --------------------------------------------------------------------- #


def test_ffn_splits_into_two_single_shape_legs():
    target = build_samsung_target()
    sch = allo.customize(ffn_256_1024_256_top, enable_tensor=False)
    trace = allo.match_workload(target, sch.module)
    buckets = _bucket_for_autoschedule(trace)
    assert len(buckets) == 2, f"FFN must split into two GEMV legs, got {len(buckets)}"
    macs = trace.by_target_op("MAC")
    assert len(macs) == 2, f"one MAC per leg, got {len(macs)}"
    # Each leg's reduction (innermost) loop carries the leg's K; the two
    # legs have distinct K (256 vs 1024), confirming W1 != W2 -- so there
    # is no inter-leg weight reuse (ruling 008 Cell-2: no fair beat single
    # shape).
    leg1 = _sub_trace(trace, 0)
    leg2 = _sub_trace(trace, 1)
    k1 = _parse_loop_bound(leg1.matches[0].enclosing_loops[-1][2])
    k2 = _parse_loop_bound(leg2.matches[0].enclosing_loops[-1][2])
    assert {k1, k2} == {FIN, FHID}, (k1, k2)


def test_ffn_host_relu_is_zero_pim_cyc_contract():
    """The two-leg FFN total is leg1 + leg2 with NOTHING added for the
    inter-leg ReLU (ruling 008 Cell-2: host ReLU = 0 PIM cyc, the activation
    crosses the Python boundary at zero PIM cost). This is a contract
    assertion: the multi-layer run path sums per-leg cycles and adds no
    activation term."""
    from allo.spmw_codegen import _split_samsung_layers, PIMCmd

    target = build_samsung_target()
    sch = allo.customize(ffn_256_1024_256_top, enable_tensor=False)
    trace = allo.match_workload(target, sch.module)
    compiled = allo.compile_for_target(target, trace)
    pim_cmds = [c for c in compiled.cmds if isinstance(c, PIMCmd)]
    # The stream splits into exactly two layer groups (one per GEMV leg);
    # there is no third "activation" group that would carry PIM cycles.
    groups = _split_samsung_layers(pim_cmds)
    assert len(groups) == 2, f"two layer groups (no PIM activation group), got {len(groups)}"


def test_ffn_batch_dim_false_positive_is_documented():
    """KNOWN LATENT ISSUE (flagged to architect, ruling 008 Cell-2 frame):
    the structural `batch_dim` heuristic (SPEC-026 §1.2) misclassifies a
    plain GEMV leg's OUTPUT-ROW loop as a batch axis when the weight is
    indexed with a BARE loop var (`W[i,k]`), because `i` is the leading
    index of W and absent from x. The 4096x1024 gemv cell escapes this
    (its output index is the computed expression `row0 + i`, not a bare
    var), but the FFN legs written `W1[i1,k1]` trip it -> batch_dim=1024/256.

    This does NOT corrupt any committed or measured result: the FFN cell
    is BLOCKED-SIM (the reference PIMSimulator cores at the non-4096 leg
    shapes), so no FFN cost-model floor is recorded as a baseline of
    record, and the gemv cell (B=1) is unaffected. The FFN verdict is
    documented-floor (MATCH) per ruling 008 Cell-2, not a derived number.

    This test pins the current (mis)behaviour so a future fix to the
    heuristic is a deliberate, reviewed change rather than a silent drift.
    """
    target = build_samsung_target()
    sch = allo.customize(ffn_256_1024_256_top, enable_tensor=False)
    trace = allo.match_workload(target, sch.module)
    leg1 = _sub_trace(trace, 0)
    leg2 = _sub_trace(trace, 1)
    # Current behaviour: the bare-index output loop is read as a batch axis.
    assert _trace_batch_dim(leg1) == FHID  # 1024 -- the output-row loop
    assert _trace_batch_dim(leg2) == FOUT  # 256
    # Contrast: the 4096x1024 gemv (computed output index) is correctly B=1.
    gsch = allo.customize(loop012_gemv_top, enable_tensor=False)
    gtrace = allo.match_workload(target, gsch.module)
    assert _trace_batch_dim(_sub_trace(gtrace, 0)) == 1


if __name__ == "__main__":
    test_gemv_single_shape_argmin_is_dual_fiber_at_b1()
    test_ffn_splits_into_two_single_shape_legs()
    test_ffn_host_relu_is_zero_pim_cyc_contract()
    test_ffn_batch_dim_false_positive_is_documented()
    print("STATIC PASSED")
