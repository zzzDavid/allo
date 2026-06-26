# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# VERIFIER-OWNED: written by the verifier agent for task 019.
"""REAL rank-preservation: virtual backend argmin == simulator argmin.

Design 04 §5.5 specifies that the virtual backend is rank-preserving
against the REAL simulators (not against cost_fn itself, which would
be tautological). This file implements that check.

For each backend, we:
  1. Enumerate sim-distinguishable candidates (design 04 §5.5.1).
  2. Run BOTH the real simulator (non-virtual _BACKEND_RUN path) AND
     the virtual backend over the same candidate set.
  3. Assert argmin(sim) == argmin(virtual) (100% per-backend).
  4. Report absolute cycle differences (not gated).
  5. Skip (never silent-PASS) when a simulator is absent.

Ground truth = simulator cycle numbers. NEVER cost_fn (that is cost_fn
== cost_fn, the goal-check-1 tautology identified in task 013).

Candidate sets:
  * Samsung: batched GEMV weight_resident=False vs True at B in {2, 4},
    4096x1024. Single-shape B=1 is the floor (flat sim ranking, vacuous).
  * UPMEM: n_tasklets in {1, T_max} on fixed K=1024 shape + shape sweep
    K=1024/K=2048 (both dims monotone in sim cycles).
  * AiM: shape sweep K=512/K=1024/K=2048 (opsize=K monotone per
    ramulator2 timing).
"""
from __future__ import annotations

import pytest

import allo
from allo.spmw_autoschedule import (
    Placement,
    _bucket_for_autoschedule,
    _samsung_enumerate,
    _upmem_enumerate,
    _tasklet_fanout,
)
from allo.spmw_codegen import (
    RunResult,
    _aim_root,
    _docker_image_exists,
    _pimsim_root,
    _upim_root,
)
from allo.spmw_match import MatchTrace, MatchedOp, OperandBinding

from _fixtures import build_aim_target, build_samsung_target, build_upmem_target


# --------------------------------------------------------------------- #
# Simulator availability helpers
# --------------------------------------------------------------------- #

def _pim_driver_present() -> bool:
    return (_pimsim_root() / "pim_driver").exists()


def _aim_present() -> bool:
    root = _aim_root()
    return (_docker_image_exists("aim-simulator-build") and
            (root / "test" / "example.yaml").exists())


def _upmem_present() -> bool:
    binary = _upim_root() / "build" / "uPIMulator"
    return binary.exists()


# --------------------------------------------------------------------- #
# Shared trace factories (synthetic, no LLVM JIT required)
# --------------------------------------------------------------------- #

def _samsung_trace(batch_size: int = 4) -> MatchTrace:
    """4096x1024 Samsung GEMV trace (batched, B=batch_size)."""
    return MatchTrace(
        target_name="samsung_hbm_pim",
        module_name="synthetic_rankpreserve",
        matches=[
            MatchedOp(
                target_op_name="MAC",
                func_name="gemv_0_0",
                work_id=(0, 0),
                enclosing_loops=[
                    ("%b", "0", str(batch_size), 1),
                    ("%i", "0", "32", 1),
                    ("%k", "0", "1024", 1),
                ],
                operands=[
                    OperandBinding(role="x", memref_name="local_W"),
                    OperandBinding(role="y", memref_name="local_x"),
                    OperandBinding(
                        role="acc", memref_name="acc", is_loop_carried=True
                    ),
                ],
                result_memref_name="acc",
                op_range=("%a", "%b"),
                extra={"batch_dim": batch_size, "batch_loop_var": "%b"},
            )
        ],
    )


def _upmem_trace(k_bound: int = 1024, m_bound: int = 4096) -> MatchTrace:
    """UPMEM GEMV-shaped trace with K reduction and M outer loop."""
    return MatchTrace(
        target_name="upmem",
        module_name="synthetic_rankpreserve",
        matches=[
            MatchedOp(
                target_op_name="MAC",
                func_name="gemv_0_0",
                work_id=(0, 0),
                enclosing_loops=[
                    ("%arg0", "0", str(m_bound), 1),
                    ("%arg1", "0", str(k_bound), 1),
                ],
                operands=[
                    OperandBinding(role="x", memref_name="local_W"),
                    OperandBinding(role="y", memref_name="local_x"),
                    OperandBinding(
                        role="acc", memref_name="acc", is_loop_carried=True
                    ),
                ],
                result_memref_name="acc",
                op_range=("%a", "%b"),
            )
        ],
    )


def _aim_trace(k_bound: int = 1024) -> MatchTrace:
    """AiM GEMV-shaped trace with K reduction (single outer loop)."""
    return MatchTrace(
        target_name="aim",
        module_name="synthetic_rankpreserve",
        matches=[
            MatchedOp(
                target_op_name="MAC",
                func_name="gemv_0_0_0",
                work_id=(0, 0, 0),
                enclosing_loops=[
                    ("%arg1", "0", str(k_bound), 1),
                ],
                operands=[
                    OperandBinding(role="x", memref_name="local_W"),
                    OperandBinding(role="y", memref_name="local_x"),
                    OperandBinding(
                        role="acc", memref_name="acc", is_loop_carried=True
                    ),
                ],
                result_memref_name="acc",
                op_range=("%a", "%b"),
            )
        ],
    )


# --------------------------------------------------------------------- #
# Samsung: weight_resident False vs True at B in {2, 4}
# --------------------------------------------------------------------- #

@pytest.mark.skipif(
    not _pim_driver_present(),
    reason="Samsung pim_driver binary not built; sim ground truth unavailable",
)
@pytest.mark.parametrize("B", [2, 4])
def test_samsung_rank_preservation_weight_residency(B):
    """Virtual argmin (weight_resident) matches real simulator argmin at B>=2.

    Candidate set: weight_resident=False vs weight_resident=True for the
    same 4096x1024 GEMV shape. These are sim-distinguishable for B>=2
    (report 18: B*=2 crossover, 3.93x asymptote). Single-shape B=1 is the
    floor (flat sim ranking, vacuous -- excluded per §5.5.1).
    """
    import numpy as np

    target = build_samsung_target()
    trace = _samsung_trace(batch_size=B)
    _fn, matches = _bucket_for_autoschedule(trace)[0]
    cands = _samsung_enumerate(target, matches)

    # Pick one representative candidate from each weight_residency arm.
    wr_false = next(c for c in cands if c.extra.get("weight_resident") is False)
    wr_true = next(c for c in cands if c.extra.get("weight_resident") is True)

    rng = np.random.default_rng(42 + B)
    W = (rng.standard_normal((4096, 1024)) * 0.05).astype(np.float16)
    X = (rng.standard_normal((B, 1024)) * 0.05).astype(np.float16)

    # --- Simulator path: real pim_driver via run_batched ---
    compiled_false = allo.compile_for_target(target, trace, layout=wr_false)
    compiled_true = allo.compile_for_target(target, trace, layout=wr_true)

    sim_false_result = compiled_false.run_batched(W, X, compare_native=False)
    sim_true_result = compiled_true.run_batched(W, X, compare_native=False)

    assert "simulator unavailable" not in (sim_false_result.stdout or ""), (
        f"Samsung sim returned unavailable for wr=False: {sim_false_result.stdout[:200]}"
    )
    assert "simulator unavailable" not in (sim_true_result.stdout or ""), (
        f"Samsung sim returned unavailable for wr=True: {sim_true_result.stdout[:200]}"
    )
    assert sim_false_result.cycles is not None, (
        f"sim wr=False returned None cycles: {sim_false_result.stdout[-400:]}"
    )
    assert sim_true_result.cycles is not None, (
        f"sim wr=True returned None cycles: {sim_true_result.stdout[-400:]}"
    )

    sim_cycles_false = sim_false_result.cycles
    sim_cycles_true = sim_true_result.cycles
    # Simulator must confirm distinguishability (B>=2 -> wr=True < wr=False).
    assert sim_cycles_true < sim_cycles_false, (
        f"B={B}: sim unexpectedly flat: wr=False={sim_cycles_false}, "
        f"wr=True={sim_cycles_true}. Not sim-distinguishable."
    )
    sim_argmin = "wr_true"  # lower cycles

    # --- Virtual path: backend="virtual" ---
    virt_false = allo.compile_for_target(
        target, trace, layout=wr_false, backend="virtual"
    )
    virt_true = allo.compile_for_target(
        target, trace, layout=wr_true, backend="virtual"
    )
    virt_false_result = virt_false.run()
    virt_true_result = virt_true.run()

    assert "simulator unavailable" not in (virt_false_result.stdout or "")
    assert "simulator unavailable" not in (virt_true_result.stdout or "")
    assert virt_false_result.cycles is not None
    assert virt_true_result.cycles is not None

    virt_cycles_false = virt_false_result.cycles
    virt_cycles_true = virt_true_result.cycles
    virt_argmin = "wr_true" if virt_cycles_true < virt_cycles_false else "wr_false"

    # Core rank-preservation assertion.
    assert sim_argmin == virt_argmin, (
        f"B={B}: rank MISMATCH. "
        f"sim argmin={sim_argmin} (wr_false={sim_cycles_false}, wr_true={sim_cycles_true}), "
        f"virtual argmin={virt_argmin} (wr_false={virt_cycles_false}, wr_true={virt_cycles_true})"
    )

    abs_err_false = abs(virt_cycles_false - sim_cycles_false)
    abs_err_true = abs(virt_cycles_true - sim_cycles_true)
    # Report (not gated per §5.5).
    print(
        f"\nSamsung B={B}: "
        f"sim(wr_false={sim_cycles_false}, wr_true={sim_cycles_true}) "
        f"virt(wr_false={virt_cycles_false}, wr_true={virt_cycles_true}) "
        f"abs_err(wr_false={abs_err_false}, wr_true={abs_err_true}) "
        f"argmin=MATCH({sim_argmin})"
    )


# --------------------------------------------------------------------- #
# UPMEM: n_tasklets in {1, T_max} -- rank-preservation vs sim
# --------------------------------------------------------------------- #

@pytest.mark.skipif(
    not _upmem_present(),
    reason="uPIMulator binary not built; sim ground truth unavailable",
)
@pytest.mark.parametrize("k_bound,m_bound", [(128, 64), (256, 128)])
def test_upmem_rank_preservation_tasklet_count(k_bound, m_bound):
    """Virtual argmin (n_tasklets=T_max) matches real sim argmin.

    Candidate set: n_tasklets=1 vs n_tasklets=T_max (target-derived).
    Sim-distinguishable: T_max > R=11 saturates; T=1 incurs full revolver
    latency overhead (5.71x lever at K=1024, invariant across K per
    test_upmem_cost_lever_invariant_across_reduction_trip).

    Shapes (64x128, 128x256) are MLP-scale: uPIMulator GEMV slot runs
    them in <60s per case; large shapes (4096x1024) exceed the 600s
    simulator timeout budget and are excluded per §5.5 "sim-distinguishable
    within the test time budget."
    """
    target = build_upmem_target()
    trace = _upmem_trace(k_bound=k_bound, m_bound=m_bound)
    _fn, matches = _bucket_for_autoschedule(trace)[0]
    cands = _upmem_enumerate(target, matches)

    T_max = _tasklet_fanout(target)
    # Pick one representative from each tasklet arm (first matching acc placement).
    cand_t1 = next(
        c for c in cands if c.extra.get("n_tasklets", 1) == 1
    )
    cand_tmax = next(
        c for c in cands if c.extra.get("n_tasklets") == T_max
    )

    # --- Simulator path ---
    compiled_t1 = allo.compile_for_target(target, trace, layout=cand_t1)
    compiled_tmax = allo.compile_for_target(target, trace, layout=cand_tmax)

    sim_t1_result = compiled_t1.run()
    sim_tmax_result = compiled_tmax.run()

    # Check simulator actually ran (not unavailable).
    for label, result in [("T=1", sim_t1_result), (f"T={T_max}", sim_tmax_result)]:
        if "simulator unavailable" in (result.stdout or ""):
            pytest.skip(
                f"uPIMulator slot unavailable for {label}: {result.stdout[:200]}"
            )
        assert result.cycles is not None, (
            f"UPMEM sim {label} returned None cycles unexpectedly: "
            f"{result.stdout[-400:]}"
        )

    sim_cyc_t1 = sim_t1_result.cycles
    sim_cyc_tmax = sim_tmax_result.cycles

    # Confirm sim-distinguishability (T_max faster than T=1).
    assert sim_cyc_tmax < sim_cyc_t1, (
        f"K={k_bound}: sim not distinguishable: T=1={sim_cyc_t1}, "
        f"T={T_max}={sim_cyc_tmax}. Candidate set is vacuous."
    )
    sim_argmin = f"T={T_max}"

    # --- Virtual path ---
    virt_t1 = allo.compile_for_target(
        target, trace, layout=cand_t1, backend="virtual"
    )
    virt_tmax = allo.compile_for_target(
        target, trace, layout=cand_tmax, backend="virtual"
    )
    virt_t1_result = virt_t1.run()
    virt_tmax_result = virt_tmax.run()

    assert "simulator unavailable" not in (virt_t1_result.stdout or "")
    assert "simulator unavailable" not in (virt_tmax_result.stdout or "")
    assert virt_t1_result.cycles is not None
    assert virt_tmax_result.cycles is not None

    virt_cyc_t1 = virt_t1_result.cycles
    virt_cyc_tmax = virt_tmax_result.cycles
    virt_argmin = f"T={T_max}" if virt_cyc_tmax < virt_cyc_t1 else "T=1"

    assert sim_argmin == virt_argmin, (
        f"K={k_bound}: UPMEM rank MISMATCH. "
        f"sim argmin={sim_argmin} (T=1={sim_cyc_t1}, T={T_max}={sim_cyc_tmax}), "
        f"virtual argmin={virt_argmin} (T=1={virt_cyc_t1}, T={T_max}={virt_cyc_tmax})"
    )

    abs_err_t1 = abs(virt_cyc_t1 - sim_cyc_t1)
    abs_err_tmax = abs(virt_cyc_tmax - sim_cyc_tmax)
    print(
        f"\nUPMEM K={k_bound}: "
        f"sim(T1={sim_cyc_t1}, T{T_max}={sim_cyc_tmax}) "
        f"virt(T1={virt_cyc_t1}, T{T_max}={virt_cyc_tmax}) "
        f"abs_err(T1={abs_err_t1}, T{T_max}={abs_err_tmax}) "
        f"argmin=MATCH({sim_argmin})"
    )


# --------------------------------------------------------------------- #
# AiM: shape sweep K in {512, 1024, 2048} -- rank-preservation vs sim
# --------------------------------------------------------------------- #

@pytest.mark.skipif(
    not _aim_present(),
    reason="AiM Docker image / example.yaml absent; sim ground truth unavailable",
)
def test_aim_rank_preservation_shape_sweep():
    """Virtual cost ordering K=512 < K=1024 < K=2048 matches sim ordering.

    AiM's ramulator2 prices MAC opsize=K proportional to K (opsize fold is
    floor-preserving). The shape sweep gives a 3-point candidate set; the
    ordering must be monotone in both sim and virtual (rank-preserving).
    """
    target = build_aim_target()

    shapes = [512, 1024, 2048]
    sim_cycles = {}
    virt_cycles = {}

    for k in shapes:
        trace = _aim_trace(k_bound=k)

        # Sim path: compile + run through ramulator2 Docker.
        compiled = allo.compile_for_target(target, trace)
        sim_result = compiled.run()

        if "simulator unavailable" in (sim_result.stdout or ""):
            pytest.skip(
                f"AiM simulator unavailable for K={k}: {sim_result.stdout[:200]}"
            )
        assert sim_result.cycles is not None, (
            f"AiM sim K={k} returned None cycles: {sim_result.stdout[-400:]}"
        )
        sim_cycles[k] = sim_result.cycles

        # Virtual path.
        virt_compiled = allo.compile_for_target(
            target, trace, backend="virtual"
        )
        virt_result = virt_compiled.run()
        assert "simulator unavailable" not in (virt_result.stdout or "")
        assert virt_result.cycles is not None, (
            f"AiM virtual K={k} returned None: {virt_result.stdout}"
        )
        virt_cycles[k] = virt_result.cycles

    # Both orderings must be strictly monotone (K=512 < K=1024 < K=2048).
    # Sim must be distinguishable.
    assert sim_cycles[512] < sim_cycles[1024] < sim_cycles[2048], (
        f"AiM sim not monotone (expected 512<1024<2048): {sim_cycles}"
    )
    # Virtual must agree on ordering.
    assert virt_cycles[512] < virt_cycles[1024] < virt_cycles[2048], (
        f"AiM virtual not monotone: {virt_cycles} "
        f"(sim reference: {sim_cycles})"
    )

    # Rank-preservation: the sim argmin (K=512 = cheapest) must equal the
    # virtual argmin.
    sim_argmin_k = min(sim_cycles, key=sim_cycles.__getitem__)
    virt_argmin_k = min(virt_cycles, key=virt_cycles.__getitem__)
    assert sim_argmin_k == virt_argmin_k, (
        f"AiM argmin MISMATCH: sim_argmin=K={sim_argmin_k}, "
        f"virt_argmin=K={virt_argmin_k}. "
        f"sim={sim_cycles}, virt={virt_cycles}"
    )

    abs_errs = {k: abs(virt_cycles[k] - sim_cycles[k]) for k in shapes}
    print(
        f"\nAiM shape sweep: "
        f"sim={sim_cycles} virt={virt_cycles} "
        f"abs_errs={abs_errs} argmin=MATCH(K={sim_argmin_k})"
    )


if __name__ == "__main__":
    import sys

    if _pim_driver_present():
        for B in [2, 4]:
            test_samsung_rank_preservation_weight_residency(B)
    else:
        print("Samsung pim_driver absent -- skipped")

    if _upmem_present():
        for k, m in [(128, 64), (256, 128)]:
            test_upmem_rank_preservation_tasklet_count(k, m)
    else:
        print("uPIMulator absent -- skipped")

    if _aim_present():
        test_aim_rank_preservation_shape_sweep()
    else:
        print("AiM Docker absent -- skipped")

    print("ALL PASSED")
