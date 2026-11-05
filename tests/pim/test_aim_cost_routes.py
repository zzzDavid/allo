# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AiM cost-route, host-transfer, tie, runner-exit and dead-code repairs
(release audit aim-attacc.md M-05, M-06, M-07, M-08, M-11, M-12)."""

import hashlib
import inspect
from types import SimpleNamespace

import pytest

import allo
import allo.pim.costs.aim as aim_costs
from allo.dataflow import region
from allo.ir.types import bfloat16
from allo.perf import CostEvent, ExecutionGraph
from allo.pim.aim_lowering import _target_geometry
from allo.pim.costs import aim_cost
from allo.pim.targets import build_aim_target
from allo.spmw_autoschedule import (
    _aim_enumerate,
    _aim_layout_candidates,
    _bucket_for_autoschedule,
)

_POLICY = (
    {"reduction_order": "reassociate"}
    if "reduction_order" in inspect.signature(allo.compile).parameters
    else {}
)

N, K, C = 256, 256, 32
ROWS = N // C


@region()
def gemv(W: bfloat16[N, K], x: bfloat16[K], y: bfloat16[N]):
    @allo.work(mapping=[C], args=[W, x, y])
    def rows(lW: bfloat16[N, K], lx: bfloat16[K], ly: bfloat16[N]):
        ch = allo.get_wid()
        row0 = ch * ROWS
        for i in range(ROWS):
            acc: bfloat16 = 0
            for j in range(K):
                acc += lW[row0 + i, j] * lx[j]
            ly[row0 + i] = acc


def _virtual(target, layout=None):
    # Placements hold handles of one target instance: reuse it.
    kwargs = {} if layout is None else {"layout": layout}
    return allo.compile(gemv, target, aim_cost, backend="virtual", **kwargs, **_POLICY)


def _trace_digest(target, layout):
    built = allo.compile(gemv, target, aim_cost, layout=layout, **_POLICY)
    return hashlib.sha256("\n".join(map(str, built.compiled.cmds)).encode()).hexdigest()


# M-06 ------------------------------------------------------------------ #


def test_layout_and_knob_paths_price_one_route():
    """Same trace, same price, whichever way the placement was chosen.

    Calibration probe (aim_simulator 0f28a07, measured 2026-11-02): the ABK
    trace (WR_GB, WR_BIAS, MAC_ABK 16 columns, RD_MAC) simulates 222 cycles
    and the SBK expansion (16 MAC_SBK) 1527.  Before the repair the routes
    priced them 255 (knobs) / 156 (layout=) and 720 / 765.  The profile
    recalibrated per command class (release round 50, M-19) prices them 223
    and 1513: the entry mode change, one burst per WR_GB column, the
    activation and column pitch of each MAC line, the read latency of RD_MAC;
    the 14 cycles under on the SBK expansion are the one extra cycle each
    consecutive fresh MAC_SBK takes on a new bank.
    """
    target = build_aim_target()
    ranked = _virtual(target)
    sbk, abk = _aim_layout_candidates(target, ranked.compiled.trace.matches)
    (kernel,) = ranked.compiled.placement_ranking.kernels
    knob_cost = {}
    for _pos, cycles, mode in kernel.scores:
        scope = "sbk" if "bank_scope=sbk" in mode else "abk"
        if "batch_mapping=flattened" in mode and "reuse_group=1+" in mode:
            knob_cost.setdefault(scope, cycles)
    assert _virtual(target, abk).run_backend().cycles == knob_cost["abk"] == 223
    assert _virtual(target, sbk).run_backend().cycles == knob_cost["sbk"] == 1513


def test_cost_reads_geometry_from_target():
    assert not hasattr(aim_costs, "LANES_PER_PU")
    assert not hasattr(aim_costs, "BANKS_PER_CHANNEL")
    target = build_aim_target()
    geometry = _target_geometry(target)
    record = SimpleNamespace(outputs=128, batches=2, replicas=1, partitions=4)
    # 16 merged vectors: (P + 1) 32-byte entries each over 32 B/cycle, plus
    # P - 1 host adds per vector.
    assert aim_costs.host_reduction_cycles(record, geometry) == 16 * 5 + 16 * 3
    # A 512-bit vector target: same bytes, half the vectors to add.
    wide = SimpleNamespace(**{**geometry.__dict__, "lanes": 32, "vector_bits": 512})
    assert aim_costs.host_reduction_cycles(record, wide) == 8 * 5 * 2 + 8 * 3


# M-05 ------------------------------------------------------------------ #


def _host_event_cycles(target, name, metrics):
    bound = aim_cost.bind(target)
    graph = ExecutionGraph(name="probe")
    event = CostEvent.create("host:probe", target.move(name), metrics=metrics)
    bound.emit(graph, event, ())
    return bound.evaluate(graph).cycles


def test_host_transfer_prices_the_moved_slice():
    target = build_aim_target()
    whole = {"shape": (32, 4096, 128), "dtype_bits": 16, "bytes": 32 * 4096 * 128 * 2}
    full = _host_event_cycles(target, "SCATTER_BANKS", whole)
    assert full == aim_costs.HOST_WRITE_BURST_CYCLES + (2**20 - 1) * aim_costs.N_CCD
    # k_cache[:, 127, :]: 32 x 128 BF16 = 256 bursts of 32 bytes.
    sliced = _host_event_cycles(
        target, "SCATTER_BANKS", {**whole, "host_slice": (None, 127, None)}
    )
    assert sliced == aim_costs.HOST_WRITE_BURST_CYCLES + 255 * aim_costs.N_CCD
    # a[0:4096] of a flat buffer.
    flat = {"shape": (131072,), "dtype_bits": 16, "bytes": 262144}
    assert (
        _host_event_cycles(target, "GATHER_BANKS", {**flat, "host_slice": ((0, 4096),)})
        == aim_costs.HOST_READ_BURST_CYCLES + 255 * aim_costs.N_CCD
    )
    # A resident buffer is on the device before the program starts.
    assert (
        _host_event_cycles(target, "SCATTER_BANKS", {**whole, "host_resident": True})
        == 0
    )


# M-07 ------------------------------------------------------------------ #


def test_tied_minima_emit_one_program():
    """Equal-cost AiM candidates at the minimum are the same program.

    Channel groups wider than the work (replica_partitions 32 for 256
    outputs) only repeated the stream with padded channels; they are no
    longer enumerated.  The remaining minimum ties (reuse_group values above
    the single launch, flattened vs row_packed for one batch) lower to
    byte-identical traces, so the tie is genuine.
    """
    target = build_aim_target()
    ranked = _virtual(target)
    (kernel,) = ranked.compiled.placement_ranking.kernels
    # Only a 32-way reduction split can occupy all 32 channels.
    for _p, _c, mode in kernel.scores:
        if "replica_partitions=32" in mode:
            assert "reduction_partitions=32" in mode, mode
    best = min(cycles for _p, cycles, _m in kernel.scores)
    tied = [mode for _p, cycles, mode in kernel.scores if cycles == best]
    assert len(tied) > 1
    (_scope, matches) = _bucket_for_autoschedule(ranked.compiled.trace)[0]
    by_mode = {c.mode: c for c in _aim_enumerate(target, matches)}
    digests = {_trace_digest(target, by_mode[mode]) for mode in tied}
    assert len(digests) == 1


# M-08 ------------------------------------------------------------------ #


def test_failed_simulator_run_reports_no_cycles(monkeypatch, tmp_path):
    import allo.spmw_aim as spmw_aim

    (tmp_path / "test").mkdir()
    (tmp_path / "test" / "example.yaml").write_text("")
    monkeypatch.setattr(spmw_aim, "_aim_unavailable_reason", lambda: None)
    monkeypatch.setattr(spmw_aim, "_aim_root", lambda: tmp_path)
    monkeypatch.setattr(spmw_aim, "_check_simulator_geometry", lambda *a: None)

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(
            returncode=139, stdout=b"memory_system_cycles: 1234\n", stderr=b"segfault"
        )

    monkeypatch.setattr(spmw_aim.subprocess, "run", fake_run)
    compiled = SimpleNamespace(target=build_aim_target(), cmds=["AiM EOC"])
    with pytest.raises(RuntimeError, match="exited with status 139"):
        spmw_aim._run_aim(compiled)


# M-11 / M-12 ----------------------------------------------------------- #


def test_single_isr_encoder_and_target_slot_conventions():
    from allo.spmw_aim import AimCtx, _bank_slot_conventions

    for name in (
        "cmd",
        "_ISR_FIELDS",
        "_operand_text",
        "_handle_to_fields",
        "_unit_env",
    ):
        assert name not in AimCtx.__dict__
    target = build_aim_target()
    host = {"SCATTER_BANKS", "GATHER_BANKS", "BCAST_BANKS"}
    for unit in target._walk():
        for move in unit.moves.values():
            assert (move.emit is not None) == (move.name in host), move.name
        for op in getattr(unit, "ops", {}).values():
            assert op.emit is None, op.name
    assert _bank_slot_conventions(target) == ({"x": 0, "y": 1, "dst": 2}, 2, (0, 1))
    target.banks.geometry["ewmul_slots"] = (1, 2, 3)
    assert _bank_slot_conventions(target)[0] == {"x": 1, "y": 2, "dst": 3}
