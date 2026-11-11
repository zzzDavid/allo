# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The vector staging a bank-filled GRF_A costs (S-12, round 132), on the
production path: an ordinary-source three-column contraction and a paper
GEMV compiled through ``allo.compile`` with forced placements, the emitted
``mac_vector_stage`` step checked against counts derived here from a
transcription of the conductor's staging loops (``TenonConductor.inc``
``stage_vector``, the resident and restaged phase structure), not against
the cost helper; and, when the simulator runs, against the conductor's own
address trace; the reversed multiplication operand order (the matcher's
batch_dim then names the row loop) prices and runs the same (round 134)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import allo
from allo.dataflow import region as _df_region
from allo.ir.types import float16 as fp16
from allo.pim.costs import samsung_cost
from allo.pim.costs.samsung import (
    ELEMENT_BITS,
    HOST_BURST_CYCLES,
    STAGE_ROW_VISIT_CYCLES,
    STAGE_WRITE_CYCLES,
    samsung_staging_share,
)
from allo.pim.targets import build_samsung_target
from allo.spmw_autoschedule import _bucket_for_autoschedule, _samsung_enumerate
from allo.spmw_knobs import reset_active_liveness, set_active_liveness
from allo.spmw_pipeline import normalize_source

M, K, N, ROWS = 1024, 384, 3, 8  # K = 384: three tiles, a column span crossing a row
COLS_PER_ROW = 32


@_df_region()
def three_columns(W: fp16[M, K], B: fp16[K, N], C: fp16[M, N]):
    @allo.work(mapping=[16, 8], args=[W, B, C])
    def body(a: fp16[M, K], b: fp16[K, N], c: fp16[M, N]):
        pid, uid = allo.get_wid()
        base = (pid * 8 + uid) * ROWS
        for row in range(ROWS):
            for col in range(N):
                acc: fp16 = 0
                for k in range(K):
                    acc += a[base + row, k] * b[k, col]
                c[base + row, col] = acc


@_df_region()
def three_columns_vector_first(W: fp16[M, K], B: fp16[K, N], C: fp16[M, N]):
    """The same contraction with the multiplication operands reversed: the
    matcher's batch_dim then names the row loop (8 > N = 3), while the
    validated Samsung contraction still has N = 3 columns (round 134)."""

    @allo.work(mapping=[16, 8], args=[W, B, C])
    def body(a: fp16[M, K], b: fp16[K, N], c: fp16[M, N]):
        pid, uid = allo.get_wid()
        base = (pid * 8 + uid) * ROWS
        for row in range(ROWS):
            for col in range(N):
                acc: fp16 = 0
                for k in range(K):
                    acc += b[k, col] * a[base + row, k]
                c[base + row, col] = acc


def _gemv_case():
    spec = importlib.util.spec_from_file_location(
        "samsung_vector_staging_workload",
        Path(__file__).with_name("gemv") / "workload.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CASES["gemv_m4096_k256"]


def conductor_even_bank_vector_staging(
    k_tiles, E, columns, resident, cols_per_row=COLS_PER_ROW
):
    """Transcription of the conductor: ``stage_vector(n)`` writes, for every
    tile ``t`` and entry ``a`` (blocks between them), column
    ``vector_base + (n * k_tiles + t) * E + a`` of each block's even bank,
    ``vector_base`` row aligned (``J * T0 * tile_cols``).  Resident matrix:
    ``stage_matrix`` then ``stage_vector(n)`` for every column, one phase.
    Restaged matrix: for each column, ``stage_matrix`` (the even bank at the
    matrix rows), ``stage_vector(n)``, then that column's passes, so every
    column's phase enters its vector row anew.  Returns the writes and row
    visits of one even bank."""
    phases = [range(columns)] if resident else [[n] for n in range(columns)]
    writes = visits = 0
    for phase in phases:
        last_row = None  # the matrix staging and the passes leave the vector rows
        for n in phase:
            for t in range(k_tiles):
                for a in range(E):
                    row = ((n * k_tiles + t) * E + a) // cols_per_row
                    writes += 1
                    if row != last_row:
                        visits += 1
                        last_row = row
    return writes, visits


def _forced(fn, target, family, residency, resident):
    front = normalize_source(fn, target, reduction_order="reassociate", analyze=True)
    buckets = _bucket_for_autoschedule(front.trace)
    token = set_active_liveness(None)
    try:
        candidates = _samsung_enumerate(target, buckets[0][1])
    finally:
        reset_active_liveness(token)
    for candidate in candidates:
        extra = candidate.extra
        if (
            candidate.mode.split("+")[0] == family
            and set((extra.get("grf_residency") or {}).values()) == {residency}
            and extra.get("crf_issue") == "shared"
            and bool(extra.get("stage_resident")) == resident
        ):
            compiled = allo.compile(
                fn,
                target,
                samsung_cost,
                reduction_order="reassociate",
                layout=[candidate] * len(buckets),
            )
            compiled.matcher_batch_dim = int(
                front.trace.matches[0].extra.get("batch_dim", 1)
            )
            return compiled
    raise KeyError((family, residency, resident))


@pytest.fixture(scope="module")
def target():
    return build_samsung_target()


@pytest.fixture(scope="module")
def forced(target):
    return {
        resident: _forced(three_columns, target, "bank_row", "crf", resident)
        for resident in (False, True)
    }


@pytest.fixture(scope="module")
def forced_vector_first(target):
    return {
        resident: _forced(
            three_columns_vector_first, target, "bank_row", "crf", resident
        )
        for resident in (False, True)
    }


def _restage_steps(compiled):
    return [
        a
        for a in compiled.compiled.execution_graph.activities
        if a.label == "mac_restage"
    ]


def _vector_steps(compiled):
    return [
        a
        for a in compiled.compiled.execution_graph.activities
        if a.label == "mac_vector_stage"
    ]


def test_emitted_vector_staging_equals_the_conductors_work(target, forced):
    E = int(target.grf_a.slots)
    burst = int(target.grf_a.width) // ELEMENT_BITS
    # The production event carries Register.lanes (8 entries); the conductor
    # sizes a tile by the burst (16 FP16), so K = 384 is 3 tiles, not 6.
    assert int(target.grf_b.lanes) == 8 and burst == 16
    k_tiles = -(-K // (E * burst))
    assert k_tiles == 3
    for resident, compiled in forced.items():
        program = compiled.compiled.layout_ctx.programs[0]
        assert program.vector_from_bank and program.weight_resident == resident
        assert program.k_tiles == k_tiles
        steps = _vector_steps(compiled)
        assert len(steps) == 128  # one per work item of the 16 x 8 grid
        writes, visits = conductor_even_bank_vector_staging(k_tiles, E, N, resident)
        assert writes == N * k_tiles * E == 72
        assert visits == (3 if resident else 5)  # columns 24-47 and 48-71 cross a row
        expected = max(
            writes * STAGE_WRITE_CYCLES,
            visits * STAGE_ROW_VISIT_CYCLES + writes * STAGE_WRITE_CYCLES,
        )
        assert {s.latency_cycles for s in steps} == {expected}, (resident, expected)
    assert (
        _vector_steps(forced[False])[0].latency_cycles
        > _vector_steps(forced[True])[0].latency_cycles
    )
    # The matrix staging of the same production events also counts bursts by
    # the register width: the share at 16 elements per burst, not at 8.
    stage = [
        a
        for a in forced[True].compiled.execution_graph.activities
        if a.label == "mac_stage"
    ]
    G = int(target.grf_b.slots)
    assert {a.latency_cycles for a in stage} == {
        samsung_staging_share(ROWS, K, G=G, E=E, lanes=burst, columns=1)["latency"]
    }
    assert (
        stage[0].latency_cycles
        != samsung_staging_share(ROWS, K, G=G, E=E, lanes=8)["latency"]
    )


def test_operand_order_does_not_change_the_staging_counts(
    target, forced, forced_vector_first
):
    """Round 134 (review): the matcher's batch_dim follows the source operand
    order (the row loop, 8, when the vector is written first; the column
    loop, 3, otherwise), the validated contraction has N = 3 either way,
    and the vector staging and the matrix restaging must price the
    contraction's columns, not the matcher's count."""
    E = int(target.grf_a.slots)
    burst = int(target.grf_a.width) // ELEMENT_BITS
    k_tiles = -(-K // (E * burst))
    assert forced[False].matcher_batch_dim == N == 3
    assert forced_vector_first[False].matcher_batch_dim == ROWS == 8 > N
    for resident in (False, True):
        normal, swapped = forced[resident], forced_vector_first[resident]
        for compiled in (normal, swapped):
            program = compiled.compiled.layout_ctx.programs[0]
            assert program.contraction.N == N
            assert program.weight_resident == resident
        writes, visits = conductor_even_bank_vector_staging(k_tiles, E, N, resident)
        expected = max(
            writes * STAGE_WRITE_CYCLES,
            visits * STAGE_ROW_VISIT_CYCLES + writes * STAGE_WRITE_CYCLES,
        )
        assert {s.latency_cycles for s in _vector_steps(swapped)} == {expected}
        assert {s.latency_cycles for s in _vector_steps(normal)} == {expected}
        # the matrix restaging prices N - 1 extra stagings of the item's rows
        restage_normal, restage_swapped = _restage_steps(normal), _restage_steps(
            swapped
        )
        if resident:
            assert restage_normal == restage_swapped == []
        else:
            expected_restage = (N - 1) * ROWS * -(-K // burst) * HOST_BURST_CYCLES
            assert len(restage_normal) == len(restage_swapped) == 128
            assert {s.latency_cycles for s in restage_normal} == {expected_restage}
            assert {s.latency_cycles for s in restage_swapped} == {expected_restage}
        # the whole graphs price alike: operand order is not a cost
        totals = [
            sorted(a.latency_cycles for a in c.compiled.execution_graph.activities)
            for c in (normal, swapped)
        ]
        assert totals[0] == totals[1]
        assert normal.estimate().cycles == swapped.estimate().cycles


def test_paper_gemv_vector_staging_is_two_tiles(target):
    """K = 256 with a bank fill: two tiles of eight bursts in one row, 72
    cycles per work item (the first round 132 version charged four)."""
    fn, _m, k = _gemv_case()
    assert k == 256
    compiled = _forced(fn, target, "bank_row", "crf", False)
    E = int(target.grf_a.slots)
    writes, visits = conductor_even_bank_vector_staging(2, E, 1, False)
    assert (writes, visits) == (16, 1)
    assert (
        {s.latency_cycles for s in _vector_steps(compiled)}
        == {
            max(
                writes * STAGE_WRITE_CYCLES,
                visits * STAGE_ROW_VISIT_CYCLES + writes * STAGE_WRITE_CYCLES,
            )
        }
        == {72}
    )


def _trace_even_bank_vector_counts(text):
    """Per even bank: ``Wstage_vector`` writes and row visits, a visit being a
    write whose row differs from the bank's previous staging write of any
    kind (the matrix staging between phases leaves the vector rows)."""
    last_row, writes, visits, first_columns = {}, {}, {}, []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != 5 or not fields[0].startswith("Wstage_"):
            continue
        tag, _ch, bank, row, col = fields
        bank, row, col = int(bank), int(row), int(col)
        if tag == "Wstage_vector":
            writes[bank] = writes.get(bank, 0) + 1
            if last_row.get(bank) != row:
                visits[bank] = visits.get(bank, 0) + 1
                if bank == 0:
                    first_columns.append(col)
        last_row[bank] = row
    return writes, visits, first_columns


def test_conductor_trace_matches_the_transcription(
    target, forced, forced_vector_first, samsung_sim_gate, tmp_path, monkeypatch
):
    E = int(target.grf_a.slots)
    rng = np.random.default_rng(132)
    W = (rng.standard_normal((M, K)) * 0.03).astype(np.float16)
    B = (rng.standard_normal((K, N)) * 0.03).astype(np.float16)
    reference = W.astype(np.float32) @ B.astype(np.float32)
    runs = [(resident, "normal", c) for resident, c in forced.items()] + [
        (resident, "vector_first", c) for resident, c in forced_vector_first.items()
    ]
    cycles = {}
    for resident, order, compiled in runs:
        dump = tmp_path / f"trace-{order}-{resident}.txt"
        monkeypatch.setenv("TENON_CONDUCTOR_ADDRESS_TRACE", str(dump))
        out = np.zeros((M, N), np.float16)
        result = compiled(W, B, out)
        assert result.kind == "simulator"
        cycles[(resident, order)] = int(result.cycles)
        assert np.all(np.isfinite(out.astype(np.float32)))
        np.testing.assert_allclose(out.astype(np.float32), reference, atol=1e-2)
        text = dump.read_text()
        writes, visits, first_columns = _trace_even_bank_vector_counts(text)
        expected_writes, expected_visits = conductor_even_bank_vector_staging(
            3, E, N, resident
        )
        assert set(writes) == set(range(0, 16, 2)), sorted(writes)  # every even bank
        assert set(writes.values()) == {expected_writes}, (resident, writes)
        assert set(visits.values()) == {expected_visits}, (resident, visits)
        assert first_columns[0] % COLS_PER_ROW == 0  # vector_base is row aligned
        # the emitted step priced exactly these counts
        assert {s.latency_cycles for s in _vector_steps(compiled)} == {
            max(
                expected_writes * STAGE_WRITE_CYCLES,
                expected_visits * STAGE_ROW_VISIT_CYCLES
                + expected_writes * STAGE_WRITE_CYCLES,
            )
        }
    # the reversed operand order runs the same program: same cycles
    for resident in (False, True):
        assert cycles[(resident, "normal")] == cycles[(resident, "vector_first")]
