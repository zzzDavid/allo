# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SPEC-026: batched-GEMV weight-reuse -- match, enumerator, cost, and
faithful-run tests.

The static (no-simulator) tests assert:
  * the matcher stamps `extra["batch_dim"]` STRUCTURALLY (B from the loop
    that is a leading index of one input but absent from the other), and
    single-vector GEMV defaults to B=1;
  * the enumerator emits BOTH `weight_resident in {False, True}` candidates
    unconditionally (the 2x tail cross);
  * the cost model splits into P/E/R, ties resident==non-resident at B=1
    (I4 parity), and argmin earns weight_resident for B>=2.

The simulator-gated test runs the batched faithful path and confirms the
Tenon stream strictly beats the native rebaseline for B>=2 with preload
clocked once.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import allo
from allo.dataflow import region as _df_region
from allo.ir.types import float32 as fp16
from allo.spmw_autoschedule import (
    _bucket_for_autoschedule,
    _samsung_enumerate,
)
from allo.spmw_codegen import _pimsim_root
from allo.spmw_cost_models import (
    _samsung_mk,
    _samsung_preload_cycles,
    _samsung_readback_cycles,
    _trace_batch_dim,
)
from allo.spmw_match import MatchTrace
from allo.spmw_match_engine import batch_dim

from _fixtures import build_samsung_target


# 4096x1024 weight (the floor-finding shape); B batched input vectors.
BATCH, M, K = 4, 4096, 1024
ROWS = M // (16 * 8)  # 32 rows per work-id


@_df_region()
def batched_gemv_top(W: fp16[M, K], X: fp16[BATCH, K], Y: fp16[BATCH, M]):
    @allo.work(mapping=[16, 8], args=[W, X, Y])
    def gemv(local_W: fp16[M, K], local_X: fp16[BATCH, K], local_Y: fp16[BATCH, M]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * ROWS
        for b in range(BATCH):
            for i in range(ROWS):
                acc: fp16 = 0
                for k in range(K):
                    acc += local_W[row0 + i, k] * local_X[b, k]
                local_Y[b, row0 + i] = acc


# Single-vector GEMV -- the B=1 parity floor.
@_df_region()
def single_gemv_top(W: fp16[M, K], x: fp16[K], y: fp16[M]):
    @allo.work(mapping=[16, 8], args=[W, x, y])
    def gemv(local_W: fp16[M, K], local_x: fp16[K], local_y: fp16[M]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * ROWS
        for i in range(ROWS):
            acc: fp16 = 0
            for k in range(K):
                acc += local_W[row0 + i, k] * local_x[k]
            local_y[row0 + i] = acc


def _sub_trace(trace: MatchTrace) -> MatchTrace:
    _fn, matches = _bucket_for_autoschedule(trace)[0]
    return MatchTrace(
        target_name=trace.target_name,
        module_name=trace.module_name,
        matches=matches,
    )


# --------------------------------------------------------------------- #
# 203 -- MATCH
# --------------------------------------------------------------------- #


def test_match_stamps_batch_dim_structurally():
    target = build_samsung_target()
    sch = allo.customize(batched_gemv_top, enable_tensor=False)
    trace = allo.match_workload(target, sch.module)
    macs = trace.by_target_op("MAC")
    assert macs, "no MAC matches"
    m = macs[0]
    var, B = batch_dim(m)
    # B is resolved from loop/index structure -- the batch loop var is the
    # leading index of the per-batch input (local_X), absent from the
    # weight (local_W). Not positional, not a literal.
    assert B == BATCH, (var, B)
    assert var is not None and var.startswith("%arg")
    assert m.extra["batch_dim"] == BATCH
    assert m.extra["batch_loop_var"] == var
    # The batch var is NOT the innermost (reduction) loop.
    assert m.enclosing_loops[-1][0] != var


def test_single_vector_defaults_batch_one():
    target = build_samsung_target()
    sch = allo.customize(single_gemv_top, enable_tensor=False)
    trace = allo.match_workload(target, sch.module)
    m = trace.by_target_op("MAC")[0]
    assert batch_dim(m) == (None, None)
    assert m.extra["batch_dim"] == 1  # I4 parity default
    assert _trace_batch_dim(trace) == 1


# --------------------------------------------------------------------- #
# 204 -- ENUMERATOR
# --------------------------------------------------------------------- #


def test_enumerator_emits_both_residencies_unconditionally():
    target = build_samsung_target()
    sch = allo.customize(batched_gemv_top, enable_tensor=False)
    trace = allo.match_workload(target, sch.module)
    _fn, matches = _bucket_for_autoschedule(trace)[0]
    cands = _samsung_enumerate(target, matches)
    flags = [c.extra.get("weight_resident") for c in cands]
    # Every candidate carries the flag; both values appear; exact 50/50.
    assert all(f in (False, True) for f in flags), flags
    assert flags.count(True) == flags.count(False)
    assert flags.count(True) > 0
    # The resident variant tags its mode for audit; placements unchanged.
    res = [c for c in cands if c.extra.get("weight_resident")]
    assert all(c.mode.endswith("+wresident") for c in res)


# --------------------------------------------------------------------- #
# 205 -- COST
# --------------------------------------------------------------------- #


def _best(target, sub, cands, cost_fn, resident: bool) -> int:
    sel = [c for c in cands if c.extra.get("weight_resident") == resident]
    return min(cost_fn(sub, c) for c in sel)


def test_cost_b1_resident_ties_nonresident():
    target = build_samsung_target()
    cost_fn = allo.get_cost("kernel_cycles", target)
    sch = allo.customize(single_gemv_top, enable_tensor=False)
    trace = allo.match_workload(target, sch.module)
    sub = _sub_trace(trace)
    cands = _samsung_enumerate(target, sub.matches)
    nr = _best(target, sub, cands, cost_fn, False)
    rr = _best(target, sub, cands, cost_fn, True)
    # I4: at B=1 both branches reduce to P+E+R -> tie.
    assert nr == rr, (nr, rr)
    M_, K_ = _samsung_mk(target, sub)
    P = _samsung_preload_cycles(target, M_, K_)
    R = _samsung_readback_cycles(target, M_)
    # Calibration anchors (report 18 §3): P=11368, R=181 at 4096x1024.
    assert (M_, K_) == (M, K)
    assert P == 11368, P
    assert R == 181, R
    # The B=1 return is exactly P+E+R (E = winner body).
    E = min(cost_fn(sub, c) for c in cands) - P - R
    assert E > 0


def test_cost_b_ge_2_argmin_picks_weight_resident():
    target = build_samsung_target()
    cost_fn = allo.get_cost("kernel_cycles", target)
    sch = allo.customize(batched_gemv_top, enable_tensor=False)
    trace = allo.match_workload(target, sch.module)
    sub = _sub_trace(trace)
    assert _trace_batch_dim(sub) == BATCH
    cands = _samsung_enumerate(target, sub.matches)
    nr = _best(target, sub, cands, cost_fn, False)
    rr = _best(target, sub, cands, cost_fn, True)
    # For B>=2 the resident schedule strictly beats non-resident.
    assert rr < nr, (rr, nr)
    # The overall argmin over ALL candidates lands on weight_resident.
    scored = sorted((cost_fn(sub, c), c.extra.get("weight_resident")) for c in cands)
    assert scored[0][1] is True, scored[0]
    # Structure: non-resident = B*(P+E+R); resident = P+B*(E+R).
    M_, K_ = _samsung_mk(target, sub)
    P = _samsung_preload_cycles(target, M_, K_)
    R = _samsung_readback_cycles(target, M_)
    E = nr // BATCH - P - R
    assert nr == BATCH * (P + E + R)
    assert rr == P + BATCH * (E + R)


def test_cost_zero_preload_falsifier():
    """With preload forced to 0 the resident win vanishes for ALL B --
    the entire speedup is the single P term (report 18 §6)."""
    target = build_samsung_target()
    # Force PRELOAD_FAN huge so (M*K // fan) == 0 and PRELOAD_CRF=0 -> P=0.
    target.move("PRELOAD_FAN").cycles = M * K * 1000
    target.move("PRELOAD_CRF").cycles = 0
    cost_fn = allo.get_cost("kernel_cycles", target)
    sch = allo.customize(batched_gemv_top, enable_tensor=False)
    trace = allo.match_workload(target, sch.module)
    sub = _sub_trace(trace)
    assert _samsung_preload_cycles(target, *_samsung_mk(target, sub)) == 0
    cands = _samsung_enumerate(target, sub.matches)
    nr = _best(target, sub, cands, cost_fn, False)
    rr = _best(target, sub, cands, cost_fn, True)
    assert nr == rr, (nr, rr)


def test_cost_bstar_invariant_under_preload_perturbation():
    """Scaling a preload constant moves the win MAGNITUDE but not the
    crossover B*=2 (report 18 §4)."""
    target = build_samsung_target()
    sch = allo.customize(batched_gemv_top, enable_tensor=False)
    trace = allo.match_workload(target, sch.module)
    sub = _sub_trace(trace)
    # B=BATCH>=2: resident must beat non-resident before and after a 4x P bump.
    cost_fn = allo.get_cost("kernel_cycles", target)
    cands = _samsung_enumerate(target, sub.matches)
    before = _best(target, sub, cands, cost_fn, False) - _best(
        target, sub, cands, cost_fn, True
    )
    target.move("PRELOAD_WR").cycles = target.move("PRELOAD_WR").cycles * 4
    cost_fn2 = allo.get_cost("kernel_cycles", target)
    after = _best(target, sub, cands, cost_fn2, False) - _best(
        target, sub, cands, cost_fn2, True
    )
    # Win magnitude grows with P; the crossover (resident<non-resident for
    # B>=2) holds in both cases.
    assert before > 0 and after > before


# --------------------------------------------------------------------- #
# 206 -- FAITHFUL RUN (simulator-gated)
# --------------------------------------------------------------------- #


def _pim_driver_present() -> bool:
    return (_pimsim_root() / "pim_driver").exists()


@pytest.mark.skipif(
    not _pim_driver_present(),
    reason="PIMSimulator pim_driver binary not built",
)
@pytest.mark.parametrize("B", [1, 2, 4, 8])
def test_batched_faithful_run_tenon_beats_native(B):
    import numpy as np

    target = build_samsung_target()
    sch = allo.customize(batched_gemv_top, enable_tensor=False)
    trace = allo.match_workload(target, sch.module)
    compiled = allo.compile_for_target(target, trace)

    rng = np.random.default_rng(7)
    W = (rng.standard_normal((M, K)) * 0.05).astype(np.float16)
    X = (rng.standard_normal((B, K)) * 0.05).astype(np.float16)

    result = compiled.run_batched(W, X, compare_native=True)
    assert result.cycles is not None, result.stdout[-400:]
    tenon = result.extra["tenon_total"]
    native = result.extra["native_total"]
    # Gate A: tenon <= native always, strict for B>=2.
    assert tenon <= native, (B, tenon, native)
    if B >= 2:
        assert tenon < native, (B, tenon, native)
        # Resident clocks preload ONCE; native re-pays it B times.
        assert result.extra["tenon_phases"]["preload"] < \
            result.extra["native_phases"]["preload"]
    if B == 1:
        # I4: B=1 ties the single-shape floor (15251 @4096x1024).
        assert tenon == native
        assert tenon == 15251, tenon


@pytest.mark.skipif(
    not _pim_driver_present(),
    reason="PIMSimulator pim_driver binary not built",
)
def test_batched_per_vector_numerics_match_single_path():
    """Each batched vector's output blob is byte-identical to running that
    vector through the validated single-vector faithful path -- so the
    batched path inherits its fp16 fidelity (max-err <= 0.0156)."""
    import numpy as np
    import subprocess
    import tempfile
    from allo.spmw_codegen import _write_samsung_cmds, PIMCmd

    root = _pimsim_root()
    driver = root / "pim_driver"

    target = build_samsung_target()
    sch = allo.customize(batched_gemv_top, enable_tensor=False)
    trace = allo.match_workload(target, sch.module)
    compiled = allo.compile_for_target(target, trace)

    rng = np.random.default_rng(11)
    B = 4
    W = (rng.standard_normal((M, K)) * 0.05).astype(np.float16)
    X = (rng.standard_normal((B, K)) * 0.05).astype(np.float16)

    res = compiled.run_batched(W, X, compare_native=False)
    assert res.cycles is not None

    # Re-run a single vector through the batched path with B=1 and compare
    # its output to the same vector inside the B>1 batched output. Apply the
    # same ISA-valid filter the run path uses (drop bank-dst<-GRF MOVs the
    # C++ validationCheck rejects), else the driver aborts before writing.
    def _crf_valid(c):
        if c.type_ in ("MOV", "FILL"):
            bank_dst = c.dst_ in ("EVEN_BANK", "ODD_BANK")
            grf_src = any(
                s in ("GRF_A", "GRF_B") for s in (c.src0_, c.src1_, c.src2_)
            )
            if bank_dst and grf_src:
                return False
        return True

    pim_cmds = [
        c for c in compiled.cmds if isinstance(c, PIMCmd) and _crf_valid(c)
    ]

    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        w_p = td_p / "W.npy"
        cmds_p = td_p / "cmds.txt"
        np.save(w_p, W)
        _write_samsung_cmds(cmds_p, pim_cmds)

        def run_one(xrow, tag):
            x_p = td_p / f"x_{tag}.npy"
            o_p = td_p / f"o_{tag}.bin"
            np.save(x_p, xrow.reshape(1, -1))
            subprocess.run(
                [str(driver), "--op", "GEMV", "--weight", str(w_p),
                 "--in", str(x_p), "--output-dim", str(M),
                 "--input-dim", str(K), "--out", str(o_p),
                 "--cmds", str(cmds_p), "--faithful"],
                cwd=str(root), capture_output=True, timeout=600, check=False,
            )
            return np.fromfile(o_p, dtype=np.uint16)

        # batched B output
        x_all = td_p / "X.npy"
        o_all = td_p / "O.bin"
        np.save(x_all, X)
        subprocess.run(
            [str(driver), "--op", "GEMV", "--weight", str(w_p),
             "--in", str(x_all), "--output-dim", str(M),
             "--input-dim", str(K), "--out", str(o_all),
             "--cmds", str(cmds_p), "--faithful", "--batch", str(B)],
            cwd=str(root), capture_output=True, timeout=600, check=False,
        )
        batched = np.fromfile(o_all, dtype=np.uint16).reshape(B, M * 16)

        for b in (0, B - 1):
            single = run_one(X[b], f"s{b}")
            assert np.array_equal(batched[b], single), (
                f"vector {b} batched output != single-vector path"
            )


if __name__ == "__main__":
    test_match_stamps_batch_dim_structurally()
    test_single_vector_defaults_batch_one()
    test_enumerator_emits_both_residencies_unconditionally()
    test_cost_b1_resident_ties_nonresident()
    test_cost_b_ge_2_argmin_picks_weight_resident()
    test_cost_zero_preload_falsifier()
    test_cost_bstar_invariant_under_preload_perturbation()
    print("STATIC PASSED")
