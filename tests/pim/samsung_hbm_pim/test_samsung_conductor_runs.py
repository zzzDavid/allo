# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Samsung programs executed by the Tenon conductor on PIMSimulator.

The selected placement determines the uploaded CRF, the data placement and
the cycles (release audit S-01).  Contract:
docs/release/design/samsung-execution.md.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import allo
from allo.dataflow import region as _df_region
from allo.ir.types import float16 as fp16
from allo.pim.costs import samsung_cost
from allo.pim.targets import build_samsung_target
from allo.spmw_autoschedule import _samsung_enumerate
from allo.spmw_codegen import PIMCmd
from allo import spmw_samsung
from allo.spmw_simenv import pimsim_root

M, K, ROWS, N = 4096, 256, 32, 4


def _workloads():
    path = Path(__file__).resolve().parent / "gemv" / "workload.py"
    spec = importlib.util.spec_from_file_location("samsung_conductor_workload", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CASES


@_df_region()
def gemm(A: fp16[M, K], B: fp16[K, N], C: fp16[M, N]):
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
def ragged_gemv(W: fp16[2048, 100], x: fp16[100], y: fp16[2048]):
    @allo.work(mapping=[16, 8], args=[W, x, y])
    def kernel(lW: fp16[2048, 100], lx: fp16[100], ly: fp16[2048]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * 16
        for i in range(16):
            acc: fp16 = 0
            for k in range(100):
                acc += lW[row0 + i, k] * lx[k]
            ly[row0 + i] = acc


ELT_N = 100000


@_df_region()
def flat_add(a: fp16[ELT_N], b: fp16[ELT_N], c: fp16[ELT_N]):
    @allo.work(mapping=[1], args=[a, b, c])
    def kernel(la: fp16[ELT_N], lb: fp16[ELT_N], lc: fp16[ELT_N]):
        for i in range(ELT_N):
            lc[i] = la[i] + lb[i]


@_df_region()
def flat_mul(a: fp16[ELT_N], b: fp16[ELT_N], c: fp16[ELT_N]):
    @allo.work(mapping=[1], args=[a, b, c])
    def kernel(la: fp16[ELT_N], lb: fp16[ELT_N], lc: fp16[ELT_N]):
        for i in range(ELT_N):
            lc[i] = la[i] * lb[i]


@_df_region()
def flat_relu(a: fp16[ELT_N], c: fp16[ELT_N]):
    @allo.work(mapping=[1], args=[a, c])
    def kernel(la: fp16[ELT_N], lc: fp16[ELT_N]):
        for i in range(ELT_N):
            lc[i] = max(la[i], 0.0)


@_df_region()
def flat_mad(a: fp16[ELT_N], b: fp16[ELT_N], d: fp16[ELT_N], c: fp16[ELT_N]):
    @allo.work(mapping=[1], args=[a, b, d, c])
    def kernel(la: fp16[ELT_N], lb: fp16[ELT_N], ld: fp16[ELT_N], lc: fp16[ELT_N]):
        for i in range(ELT_N):
            lc[i] = la[i] * lb[i] + ld[i]


def _f32(a):
    return a.astype(np.float32)


# The REDUCE programs regroup the FP16 reduction (16 lane partial sums summed
# in FP32); these tests run them under explicit permission.
REASSOCIATE = {"reduction_order": "reassociate"}


def _forced(workload, mode, residency, resident=False):
    target = build_samsung_target()
    default = allo.compile(workload, target, samsung_cost, **REASSOCIATE)
    for candidate in _samsung_enumerate(target, default.trace.matches[:1]):
        modes = set((candidate.extra.get("grf_residency") or {}).values()) or {"crf"}
        if (
            candidate.mode.split("+")[0] == mode
            and candidate.extra.get("crf_issue") == "shared"
            and modes == {residency}
            and bool(candidate.extra.get("stage_resident")) == resident
        ):
            n = default.placement_ranking.kernels[0].bucket_count
            return allo.compile(
                workload, target, samsung_cost, layout=[candidate] * n, **REASSOCIATE
            )
    raise KeyError((mode, residency, resident))


@pytest.fixture(scope="module")
def gemv_inputs():
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((M, K)) * 0.1).astype(np.float16)
    x = (rng.standard_normal(K) * 0.1).astype(np.float16)
    return W, x, _f32(W) @ _f32(x)


def test_canonical_run_reports_program_plan_and_padding(samsung_sim_gate, gemv_inputs):
    workload, _M, _K = _workloads()["gemv_m4096_k256"]
    target = build_samsung_target()
    default = allo.compile(workload, target, samsung_cost, **REASSOCIATE)
    compiled = allo.compile(
        workload,
        target,
        samsung_cost,
        layout=spmw_samsung.samsung_vendor_gemv_layout(target, default.trace),
        **REASSOCIATE,
    )
    W, x, ref = gemv_inputs
    y = np.zeros(M, dtype=np.float16)
    result = compiled(W, x, y)

    assert result.cycles == 4435
    np.testing.assert_allclose(_f32(y), ref, rtol=1e-2, atol=1e-2)
    extra = result.extra
    assert extra["crf"] == spmw_samsung._samsung_cmds_text(compiled.compiled.cmds)
    assert extra["conductor"]["pimcmdgen_calls"] == 0
    assert (
        extra["conductor"]["mac_groups_even"]
        == extra["conductor"]["mac_groups_odd"]
        == 8
    )
    assert extra["logical_shape"] == {"rows": 4096, "reduction": 256, "columns": 1}
    assert extra["padded_shape"] == {"rows": 4096, "reduction": 256, "columns": 1}
    assert extra["padding_fraction"] == 0.0
    assert "padded program" in extra["timing_scope"]
    assert sum(extra["cycle_phases"].values()) == result.cycles


def test_noncanonical_placements_execute_their_own_programs(
    samsung_sim_gate, gemv_inputs
):
    workload, _M, _K = _workloads()["gemv_m4096_k256"]
    W, x, ref = gemv_inputs
    cycles = {}
    for mode, residency in (("bank_row", "host"), ("dual_fiber", "crf")):
        compiled = _forced(workload, mode, residency)
        y = np.zeros(M, dtype=np.float16)
        result = compiled(W, x, y)
        np.testing.assert_allclose(_f32(y), ref, rtol=1e-2, atol=1e-2)
        receipt = result.extra["conductor"]
        if mode == "bank_row":
            assert (receipt["mac_groups_even"], receipt["mac_groups_odd"]) == (16, 0)
        else:
            assert receipt["fills"] == 2 and receipt["host_vector_writes"] == 0
        cycles[(mode, residency)] = result.cycles
    assert 4435 not in cycles.values()
    assert len(set(cycles.values())) == 2


def test_tampered_crf_is_rejected_not_canonicalized(samsung_sim_gate, gemv_inputs):
    """The conductor executes the uploaded CRF; it neither canonicalizes a
    tampered program nor reports success for it.  Two tampers of the selected
    GEMV program (MAC even, MAC odd, JUMP x7 over both, NOP drain, EXIT):

    * shortening the tile loop (JUMP x3) terminates with too few MAC groups
      per fiber, rejected against the plan's group count;
    * replacing the odd-bank MAC by a second JUMP makes the two JUMPs reset
      each other's counters (PIMRank::doPIM's ``lastJumpIdx_`` rule, which
      the conductor's walk mirrors), so the program never reaches EXIT; the
      walk's column-command budget, derived from the plan's tiles and
      groups, rejects it instead of looping."""
    workload, _M, _K = _workloads()["gemv_m4096_k256"]
    compiled = allo.compile(
        workload, build_samsung_target(), samsung_cost, **REASSOCIATE
    )
    (program,) = compiled.compiled.layout_ctx.programs
    words = list(compiled.compiled.cmds)
    assert words[2].type_ == "JUMP" and words[2].loopCounter_ == 7
    W, x, _ref = gemv_inputs
    root = pimsim_root()
    geometry = spmw_samsung.samsung_simulator_geometry(root)
    short_loop = list(words)
    short_loop[2] = PIMCmd(type_="JUMP", loopCounter_=3, loopOffset_=3)
    with pytest.raises(RuntimeError, match="MAC groups on fiber 0"):
        spmw_samsung._samsung_run_reduce(
            program, short_loop, root, geometry, W, x.reshape(1, K)
        )
    nonterminating = list(words)
    nonterminating[1] = PIMCmd(type_="JUMP", loopCounter_=3, loopOffset_=2)
    with pytest.raises(RuntimeError, match="column commands without reaching EXIT"):
        spmw_samsung._samsung_run_reduce(
            program, nonterminating, root, geometry, W, x.reshape(1, K)
        )


def test_ragged_shape_reports_padding(samsung_sim_gate):
    rng = np.random.default_rng(1)
    W = (rng.standard_normal((2048, 100)) * 0.1).astype(np.float16)
    x = (rng.standard_normal(100) * 0.1).astype(np.float16)
    y = np.zeros(2048, dtype=np.float16)
    compiled = allo.compile(
        ragged_gemv, build_samsung_target(), samsung_cost, **REASSOCIATE
    )
    result = compiled(W, x, y)
    np.testing.assert_allclose(_f32(y), _f32(W) @ _f32(x), rtol=1e-2, atol=1e-2)
    assert result.extra["logical_shape"] == {
        "rows": 2048,
        "reduction": 100,
        "columns": 1,
    }
    assert result.extra["padded_shape"] == {
        "rows": 4096,
        "reduction": 128,
        "columns": 1,
    }
    assert result.extra["padding_fraction"] == pytest.approx(
        1 - 2048 * 100 / (4096 * 128)
    )


def test_gemm_matrix_residency_lever_changes_staging(samsung_sim_gate):
    rng = np.random.default_rng(2)
    A = (rng.standard_normal((M, K)) * 0.1).astype(np.float16)
    B = (rng.standard_normal((K, N)) * 0.1).astype(np.float16)
    ref = _f32(A) @ _f32(B)
    runs = {}
    for resident in (True, False):
        compiled = _forced(gemm, "dual_fiber", "host", resident=resident)
        C = np.zeros((M, N), dtype=np.float16)
        result = compiled(A, B, C)
        np.testing.assert_allclose(_f32(C), ref, rtol=1e-2, atol=1e-2)
        runs[resident] = result
    assert runs[True].extra["conductor"]["matrix_preloads"] == 1
    assert runs[False].extra["conductor"]["matrix_preloads"] == N
    assert runs[True].cycles < runs[False].cycles
    default = allo.compile(gemm, build_samsung_target(), samsung_cost, **REASSOCIATE)
    assert default.compiled.layout_ctx.programs[0].weight_resident is True


@pytest.mark.parametrize(
    "workload, n_inputs, reference",
    [
        (flat_add, 2, lambda a, b: a + b),
        (flat_mul, 2, lambda a, b: a * b),
        (flat_relu, 1, lambda a: np.maximum(a, np.float16(0))),
        (flat_mad, 3, lambda a, b, d: a * b + d),
    ],
    ids=["ADD", "MUL", "RELU", "MAD"],
)
def test_elementwise_ops_run_exactly(samsung_sim_gate, workload, n_inputs, reference):
    rng = np.random.default_rng(3)
    inputs = [
        (rng.standard_normal(ELT_N) * 0.5).astype(np.float16) for _ in range(n_inputs)
    ]
    out = np.zeros(ELT_N, dtype=np.float16)
    compiled = allo.compile(workload, build_samsung_target(), samsung_cost)
    result = compiled(*inputs, out)
    assert result.kind == "simulator"
    np.testing.assert_array_equal(out, reference(*inputs).astype(np.float16))
    assert result.extra["padded_shape"] == {"elements": 131072}
    assert result.extra["conductor"]["pimcmdgen_calls"] == 0
