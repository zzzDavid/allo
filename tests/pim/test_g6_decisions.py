# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""G6 decision probes (docs/release/EVIDENCE.md section 3, items 5 to 10).

Each probe compares *normalized decisions* (placement mode and knob values,
layout matrix text, emitted command inventory by opcode, program records with
buffer names replaced by their roles), never raw symbol names, and checks
executed results against an independent NumPy oracle on fresh random inputs
(the seed is drawn per run and reported in every assertion message).

* Cost perturbation reaching compilation and execution (m): Samsung, UPMEM,
  APU v1 (functional route).
* Target-parameter variation (d): Samsung with 8 pseudo-channels, UPMEM with
  32 DPUs. AiM global-buffer size: ``test_aim_virtual_study.py::
  test_aim_geometry_variant_opcode_counts`` (2/4/8 KiB GB with matching rows).
* Renames with normalized decisions (a): Samsung and UPMEM GEMVs loaded from a
  differently named file.
* Operation or constant change at a fixed shape (b): Samsung, UPMEM, AiM.
* Composed producer/consumer stages executed (k): Samsung two-kernel host
  program, UPMEM k-means distances plus host update.
* Archive removal (n): ``TENON_ARTIFACTS`` / ``TENON_UPMEM_PROVENANCE_DIR``
  pointed at missing paths; an sdist unpacked outside the checkout.

Simulator tests skip with BLOCKED-SIM when their simulator is absent.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import re
import subprocess
import sys
import tarfile
import textwrap
import types
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import allo
from allo import spmw_samsung, spmw_simenv
from allo.dataflow import region as _df_region
from allo.ir.types import bfloat16, float16, int32, int64
from allo.perf import cost as _cost_decorator
from allo.pim import targets as _targets
from allo.pim.costs import apu_v1 as apu_v1_costs
from allo.pim.costs import samsung as samsung_costs
from allo.pim.costs import upmem as upmem_costs
from allo.pim.costs import aim_cost, apu_v1_cost, samsung_cost, upmem_cost
from allo.spmw_match import UnsupportedWorkloadError
from allo.pim.targets import (
    build_aim_target,
    build_apu_v1_target,
    build_samsung_target,
    build_upmem_target,
)

REPO = Path(__file__).resolve().parents[2]
SAMSUNG_GEMV_FILE = (
    Path(__file__).resolve().parent / "samsung_hbm_pim" / "gemv" / "workload.py"
)
REASSOCIATE = {"reduction_order": "reassociate"}


def _seed():
    return int(np.random.SeedSequence().entropy % 2**32)


def _f32(a):
    return np.asarray(a, dtype=np.float32)


def _sim_gate(target_name):
    import conftest

    conftest._simulator_gate(target_name)


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _samsung_paper_gemv():
    module = _load_module(SAMSUNG_GEMV_FILE, "g6_samsung_paper_gemv")
    return module.CASES["gemv_m4096_k256"]


def _rebound_cost(spec, module, name, **constants):
    """``spec`` with module-level calibration constants replaced.

    The builder's code is re-bound to a copy of its module namespace, so every
    rule it defines reads the perturbed constants; the stock spec is untouched
    and both can be compiled side by side (distinct fingerprints).
    """
    namespace = dict(vars(module))
    missing = sorted(set(constants) - set(namespace))
    assert not missing, missing
    namespace.update(constants)
    builder = spec.builder
    rebound = types.FunctionType(
        builder.__code__,
        namespace,
        builder.__name__,
        builder.__defaults__,
        builder.__closure__,
    )
    return _cost_decorator(
        target=spec.target_name,
        name=name,
        fingerprint_data=spec.fingerprint_data,
        materialization_scorer=spec.materialization_scorer,
    )(rebound)


def _target_variant(tmp_path, builder, old, new, name):
    """Rebuild ``builder``'s target with one declaration changed.

    The variant is ``builder``'s own source with ``old`` replaced by ``new``,
    written to a file (the matcher reads op lambdas back from source) and
    built through the public ``@allo.target`` / ``@allo.unit`` decorators.
    """
    source = textwrap.dedent(inspect.getsource(builder))
    assert source.count(old) == 1, (old, source.count(old))
    source = source.replace(old, new).replace(
        f"def {builder.__name__}(", f"def {name}(", 1
    )
    path = tmp_path / f"{name}.py"
    path.write_text(
        "import math\n\nimport allo\nfrom allo.spmw_match import select\n\n\n" + source
    )
    return getattr(_load_module(path, name), name)()


# --------------------------------------------------------------------- #
# Normalized decisions
# --------------------------------------------------------------------- #


def _opcode_inventory(lines):
    return dict(Counter(line.split()[0] for line in lines if line.strip()))


def _samsung_decision(compiled):
    """Name-free record of a Samsung compile: what a reviewer compares."""
    (program,) = compiled.compiled.layout_ctx.programs
    contraction = program.contraction
    roles = {}
    if contraction is not None:
        roles = {
            contraction.matrix_memref: "matrix",
            contraction.vector_memref: "vector",
            contraction.output_memref: "output",
            contraction.acc_memref: "accumulator",
        }
    lever = dict(program.lever_record or {})
    lever["grf_residency"] = {
        roles.get(name, name): mode
        for name, mode in (lever.get("grf_residency") or {}).items()
    }
    shape = None
    if contraction is not None:
        shape = (
            contraction.K,
            contraction.rows_per_item,
            contraction.N,
            contraction.matrix_shape,
            contraction.vector_shape,
            contraction.matrix_k_axis,
            contraction.vector_k_axis,
            contraction.matrix_role,
            contraction.vector_role,
            contraction.accumulator_init,
        )
    crf = spmw_samsung._samsung_cmds_text(compiled.compiled.cmds)
    ranking = compiled.placement_ranking
    # A layout= compile is not ranked.
    (kernel,) = ranking.kernels if ranking is not None else (None,)
    layouts = sorted(
        {str(getattr(p, "layout", None)) for p in compiled.compiled.layout}
    )
    return {
        "kind": program.kind,
        "lever": lever,
        "program": (
            program.crf_issue,
            program.work_items,
            program.rows,
            program.k_tiles,
            program.tile_fiber,
            program.rows_per_block,
            program.vector_from_bank,
            program.weight_resident,
            program.binds,
            program.row_map,
            program.swap_parity,
        ),
        "contraction": shape,
        "crf": crf,
        "opcodes": _opcode_inventory(crf.splitlines()),
        "bucket_count": kernel and kernel.bucket_count,
        "scores": kernel and kernel.scores,
        "chosen_position": kernel and kernel.chosen_position,
        "rejections": kernel and kernel.rejections,
        "layouts": layouts,
    }


def _samsung_placement(compiled):
    lever = _samsung_decision(compiled)["lever"]
    return lever["mode"].split("+")[0], set(lever["grf_residency"].values())


_UPMEM_NAME_FREE_EXTRA = (
    "upmem_family",
    "upmem_geometry",
    "upmem_anchor_op",
    "upmem_constants",
    "upmem_placement",
    "upmem_accumulator_init",
    "data_layout",
    "tasklets",
    "dma_bytes",
    "chunk",
    "wram_residency",
    "nc",
    "kc",
    "predicate_lowering",
    "movement",
    "movement_cost",
    "tasklet_fanout",
    "upmem_features",
)


def _placements(layout):
    return list(layout) if isinstance(layout, list) else [layout]


def _upmem_dpu_segments(compiled):
    return [s for s in compiled.compiled.layout_ctx.segments if s.kind == "dpu"]


def _upmem_decision(compiled, names=None):
    """Name-free record of a UPMEM compile.

    ``names`` maps this compile's buffer names to canonical ones; the device
    source is compared after that substitution.
    """
    names = names or {}
    placements = _placements(compiled.compiled.layout)
    modes = sorted({p.mode for p in placements})
    extras = sorted(
        {
            tuple((key, repr(p.extra.get(key))) for key in _UPMEM_NAME_FREE_EXTRA)
            for p in placements
        }
    )
    layouts = sorted({str(p.layout) for p in placements})
    segments = []
    for segment in _upmem_dpu_segments(compiled):
        kernel = segment.kernel
        source = segment.device_source
        for old, new in sorted(names.items(), key=lambda item: -len(item[0])):
            source = re.sub(rf"\b{re.escape(old)}\b", new, source)
        segments.append(
            (
                kernel.family,
                kernel.geometry,
                kernel.anchor_op,
                kernel.constants,
                tuple(role for role, _name in kernel.roles),
                kernel.dpus,
                kernel.placement,
                kernel.accumulator_init,
                kernel.streams,
                segment.executions,
                repr(segment.decision),
                source,
            )
        )
    commands = [
        re.sub(r"source_sha256=\w+", "source_sha256=*", line)
        for line in compiled.compiled.cmds
    ]
    return {
        "modes": modes,
        "extras": extras,
        "layouts": layouts,
        "segments": segments,
        "commands": commands,
        "opcodes": _opcode_inventory(compiled.compiled.cmds),
    }


# ===================================================================== #
# (m) Cost perturbation reaching compilation and execution
# ===================================================================== #

# The host GRF_A broadcast priced at 8 bus cycles per burst instead of the
# calibrated 1: one host fill (2 + 8 entries x 8 = 66 cycles) is dearer than a
# CRF FILL from the bank (26), so the selection must move to a CRF-filled
# placement; the fiber choice is unaffected (both fills are per work item).
SAMSUNG_HOST_BUS_X8 = _rebound_cost(
    samsung_cost, samsung_costs, "samsung_cost_host_bus_x8", HOST_BURST_CYCLES=8
)
# The measured odd-fiber row-open term removed.
SAMSUNG_NO_FIBER_TERM = _rebound_cost(
    samsung_cost, samsung_costs, "samsung_cost_no_fiber_term", FIBER_TILE_CYCLES=0
)


def _samsung_residency(placement):
    return set((placement.extra.get("grf_residency") or {}).values()) or {"crf"}


def _samsung_scored(compiled):
    """``position -> (cycles, placement)`` of every priced candidate."""
    (kernel,) = compiled.placement_ranking.kernels
    by_position = dict(kernel.candidates)
    return {
        position: (cycles, by_position[position])
        for position, cycles, _mode in kernel.scores
        if position in by_position
    }


def _samsung_units_per_channel(compiled):
    """Work items sharing one pseudo-channel's column-command bus (the
    inner extent of the work grid: ``mapping=[16, 8]`` puts eight items,
    one per PIM unit, on a channel)."""
    scope = compiled.trace.matches[0].extra["spmw_work_scope"]
    return int(scope.group_shape[-1])


def _samsung_odd_fiber_tiles(placement, program):
    """Odd-fiber reduction tiles one work item pays `FIBER_TILE_CYCLES` for,
    read from the candidate and its program exactly as the cost rule does
    (`allo/pim/costs/samsung.py`, the MAC rule): one per tile fiber beyond the
    first, plus one per tile when the tile rows are split across the fibers
    (every tile's second row lives in the odd fiber)."""
    extra = placement.extra or {}
    later_fibers = max(0, int(extra.get("n_fibers", 1)) - 1)
    split = extra.get("tile_rows") == "split_fibers"
    return later_fibers + (int(program.k_tiles) if split else 0)


def _samsung_host_fill_flip_factor(compiled):
    """The smallest `HOST_BURST_CYCLES` at which the cheapest realizable
    CRF-filled candidate undercuts the cheapest host-filled one, derived from
    the stock ranking: a host fill broadcasts the eight GRF_A entries of a
    work item at one burst each on the critical path, so each extra cycle per
    burst adds eight cycles to a host-filled estimate while a CRF-filled
    estimate does not read the constant.  Candidates whose tile rows are
    split across the fibers have no CRF-filled variant (the realization
    refuses them), so they are not the cheapest realizable CRF fill."""
    scored = _samsung_scored(compiled)
    host = min(c for c, pl in scored.values() if _samsung_residency(pl) == {"host"})
    crf = min(
        c
        for c, pl in scored.values()
        if _samsung_residency(pl) == {"crf"}
        and (pl.extra or {}).get("tile_rows") != "split_fibers"
    )
    gap = crf - host
    assert gap > 0, (host, crf)
    return 1 + gap // 8 + 1, gap


def test_samsung_host_fill_cost_flip_reaches_execution(
    samsung_sim_gate, record_property
):
    """Stock and perturbed cost programs select different placements; both
    execute correctly; a fixed layout compiles to the same program under both.

    Derived from the current model before running (S-17 restatement,
    round 102): the stock selection is host-filled (since round 37
    ``split_rows`` with the host GRF_A fill, the estimate including the
    matrix staging share).  The cheapest CRF-filled candidates are the
    split-row ones, 16 cycles dearer, but a split-row tile has no CRF-filled
    variant (the realization refuses it), so the cheapest realizable CRF fill
    is the ``bank_row``/``dual_fiber`` family, about 3,500 cycles dearer.  A
    host fill costs eight bursts at ``HOST_BURST_CYCLES`` each on the
    critical path, so the selection flips to that family once the constant
    exceeds ``1 + gap // 8``; the round 36 ``x8`` perturbation (56 extra
    cycles) can no longer flip it and is not what this test applies.  The
    perturbation is a counterfactual, so the stock choice is predicted to be
    faster in the simulator than the perturbed choice (previously measured
    4,108 versus 4,589 cycles at 4096x256).
    """
    workload, M, K = _samsung_paper_gemv()
    target = build_samsung_target()
    stock = allo.compile(workload, target, samsung_cost, **REASSOCIATE)
    factor, gap = _samsung_host_fill_flip_factor(stock)
    flip_cost = _rebound_cost(
        samsung_cost,
        samsung_costs,
        f"samsung_cost_host_bus_x{factor}",
        HOST_BURST_CYCLES=factor,
    )
    assert flip_cost.bind(target).fingerprint != samsung_cost.bind(target).fingerprint
    moved = allo.compile(workload, target, flip_cost, **REASSOCIATE)

    assert _samsung_placement(stock)[1] == {"host"}
    assert _samsung_placement(moved)[1] == {"crf"}
    assert _samsung_placement(moved)[0] != "split_rows"
    stock_decision, moved_decision = _samsung_decision(stock), _samsung_decision(moved)
    assert stock_decision["crf"] != moved_decision["crf"]
    # A CRF fill adds one FILL per fiber the program reads (two here).
    assert (
        moved_decision["opcodes"]["FILL"]
        == stock_decision["opcodes"].get("FILL", 0) + 2
    )
    # Legality does not read the cost: the same candidates are scored, and a
    # candidate probed under both programs is rejected for the same reason
    # (code generation is lazy, in cost order, so the probed sets differ).
    scored = {
        name: {(position, mode) for position, _cycles, mode in d["scores"]}
        for name, d in (("stock", stock_decision), ("moved", moved_decision))
    }
    assert scored["stock"] == scored["moved"]
    stock_rejected = dict(stock_decision["rejections"])
    moved_rejected = dict(moved_decision["rejections"])
    assert stock_rejected and moved_rejected
    for position in stock_rejected.keys() & moved_rejected.keys():
        assert stock_rejected[position] == moved_rejected[position]
    assert stock_decision["chosen_position"] not in moved_rejected
    assert moved_decision["chosen_position"] not in stock_rejected
    # The split-row CRF fills were cheaper than the chosen program under the
    # perturbed model, were probed in cost order and were refused at
    # realization: the flip lands on the cheapest realizable CRF fill.
    moved_scored = _samsung_scored(moved)
    cheaper_split_crf = [
        position
        for position, (cycles, pl) in moved_scored.items()
        if _samsung_residency(pl) == {"crf"}
        and (pl.extra or {}).get("tile_rows") == "split_fibers"
        and cycles < moved.estimate().cycles
    ]
    assert cheaper_split_crf
    assert all(position in moved_rejected for position in cheaper_split_crf)

    # A fixed layout: identical program under both cost programs, estimate
    # moves by exactly the host fill's extra bursts (the vendor placement
    # fills GRF_A from the host).
    vendor = spmw_samsung.samsung_vendor_gemv_layout(target, stock.trace)
    pinned = {
        name: allo.compile(
            workload, target, spec, layout=vendor, backend="virtual", **REASSOCIATE
        )
        for name, spec in (("stock", samsung_cost), ("moved", flip_cost))
    }
    pinned_decisions = {name: _samsung_decision(c) for name, c in pinned.items()}
    for key in ("kind", "lever", "program", "contraction", "crf", "opcodes", "layouts"):
        assert pinned_decisions["stock"][key] == pinned_decisions["moved"][key], key
    assert (
        pinned["moved"].estimate().cycles - pinned["stock"].estimate().cycles
        == (factor - 1) * 8
    )

    # Each cost program prices its own pick below the other's.
    cross = {
        "stock_pick_under_moved": allo.compile(
            workload,
            target,
            flip_cost,
            layout=list(stock.compiled.layout),
            backend="virtual",
            **REASSOCIATE,
        )
        .estimate()
        .cycles,
        "moved_pick_under_stock": allo.compile(
            workload,
            target,
            samsung_cost,
            layout=list(moved.compiled.layout),
            backend="virtual",
            **REASSOCIATE,
        )
        .estimate()
        .cycles,
    }
    assert stock.estimate().cycles < cross["moved_pick_under_stock"]
    assert cross["moved_pick_under_stock"] - stock.estimate().cycles == gap
    # The CRF-filled program issues no host broadcast: its estimate does not
    # read the perturbed constant.
    assert cross["moved_pick_under_stock"] == moved.estimate().cycles
    assert moved.estimate().cycles < cross["stock_pick_under_moved"]
    assert cross["stock_pick_under_moved"] - stock.estimate().cycles == (factor - 1) * 8

    seed = _seed()
    rng = np.random.default_rng(seed)
    W = (rng.standard_normal((M, K)) * 0.1).astype(np.float16)
    x = (rng.standard_normal(K) * 0.1).astype(np.float16)
    reference = _f32(W) @ _f32(x)
    cycles = {}
    for name, compiled in (("stock", stock), ("moved", moved)):
        y = np.zeros(M, dtype=np.float16)
        result = compiled(W, x, y)
        assert result.kind == "simulator"
        assert result.extra["crf"] == _samsung_decision(compiled)["crf"]
        np.testing.assert_allclose(
            _f32(y), reference, rtol=1e-2, atol=1e-2, err_msg=f"{name} seed {seed}"
        )
        cycles[name] = result.cycles
    perturbed_matches_simulator = cycles["moved"] < cycles["stock"]
    record_property(
        "samsung_cost_flip",
        {
            "flip_factor": factor,
            "gap": gap,
            "stock": (
                "+".join(_samsung_placement(stock)[0:1]) + "+host",
                stock.estimate().cycles,
                cycles["stock"],
            ),
            "moved": (
                _samsung_placement(moved)[0] + "+crf",
                moved.estimate().cycles,
                cycles["moved"],
            ),
            "cross_estimates": cross,
            "perturbed_matches_simulator": perturbed_matches_simulator,
        },
    )
    assert cycles["stock"] < cycles["moved"], cycles
    assert not perturbed_matches_simulator


def test_samsung_fiber_term_removal_moves_every_estimate_by_its_odd_fiber_tiles():
    """Reasoned invariant (S-17 restatement, round 102): the odd-fiber
    row-open term enters the MAC rule once per tile fiber beyond the first
    and, for split tile rows, once per tile; a channel's units serialize on
    its bus, so an estimate carries the term ``units x tiles`` times.
    Setting the term to zero therefore lowers every candidate's estimate by
    exactly ``FIBER_TILE_CYCLES x units x odd-fiber tiles``, derived from
    the candidate (``n_fibers``, ``tile_rows``) and its program
    (``k_tiles``); ``bank_row`` (no odd-fiber tile) is unchanged, the vendor
    ``dual_fiber`` drops by one tile per unit, the default ``split_rows``
    by three (one later fiber and two split tiles).  The ranking under the
    zero-term program is the stock ranking shifted by those derived amounts,
    and the selection stays the stock one (the shift favours it)."""
    workload, _M, _K = _samsung_paper_gemv()
    target = build_samsung_target()
    stock = allo.compile(
        workload, target, samsung_cost, backend="virtual", **REASSOCIATE
    )
    moved = allo.compile(
        workload, target, SAMSUNG_NO_FIBER_TERM, backend="virtual", **REASSOCIATE
    )
    units = _samsung_units_per_channel(stock)
    term = samsung_costs.FIBER_TILE_CYCLES
    (program,) = stock.compiled.layout_ctx.programs

    assert _samsung_placement(stock) == _samsung_placement(moved)
    assert _samsung_placement(stock)[0] == "split_rows"
    stock_scored, moved_scored = _samsung_scored(stock), _samsung_scored(moved)
    assert stock_scored.keys() == moved_scored.keys()
    shifts = {}
    for position, (cycles, placement) in stock_scored.items():
        expected = cycles - term * units * _samsung_odd_fiber_tiles(placement, program)
        shifts[position] = expected
        assert moved_scored[position][0] == expected, (position, placement.mode)
    chosen = stock.placement_ranking.kernels[0].chosen_position
    assert moved.placement_ranking.kernels[0].chosen_position == chosen
    selected = stock_scored[chosen][1]
    assert _samsung_odd_fiber_tiles(selected, program) == 1 + program.k_tiles
    assert stock.estimate().cycles - moved.estimate().cycles == term * units * (
        1 + program.k_tiles
    )

    def pinned(layout):
        return {
            spec.name: allo.compile(
                workload, target, spec, layout=layout, backend="virtual", **REASSOCIATE
            )
            .estimate()
            .cycles
            for spec in (samsung_cost, SAMSUNG_NO_FIBER_TERM)
        }

    vendor = spmw_samsung.samsung_vendor_gemv_layout(target, stock.trace)
    estimates = pinned(vendor)
    assert estimates["samsung_cost"] - estimates["samsung_cost_no_fiber_term"] == (
        term * units
    )
    bank_row = next(
        pl
        for _cycles, pl in stock_scored.values()
        if pl.mode.startswith("bank_row")
        and _samsung_residency(pl) == {"host"}
        and (pl.extra or {}).get("crf_issue") == "shared"
        and not (pl.extra or {}).get("stage_resident")
    )
    n = stock.placement_ranking.kernels[0].bucket_count
    estimates = pinned([bank_row] * n)
    assert estimates["samsung_cost"] == estimates["samsung_cost_no_fiber_term"]
    assert estimates["samsung_cost"] > moved.estimate().cycles


def test_samsung_host_burst_cost_follows_the_documented_first_fit_schedule():
    """C-18 restated (S-17, round 102): the shared evaluator places each
    unit's LD_A fill slot first-fit (Evaluator docstring).  A channel's 8
    PIM units each take a 4-cycle channel slot for LD_A, LD_B and MAC; the
    host GRF_A fill lasts ``2 + 8*b`` cycles, so unit 0's LD_B slot opens at
    that cycle and units 0-5's fill slots end at 24.  At ``b=3`` the gap
    ``[24, 26)`` cannot hold unit 6's fill slot, which first-fits behind the
    units' LD_B and MAC slots; at ``b=4`` the gap ``[24, 34)`` holds units 6
    and 7 at 24 and 28.  Through round 36 that made the kernel estimate
    non-monotone (275 at ``b=3``, 257 at ``b=4``, the anomaly C-18 records).
    Since round 37 the estimate includes the matrix staging share, behind
    which the delayed fills complete, so the total is monotone: each extra
    burst cycle adds 8 (one per GRF_A entry of the last unit's fill).  Both
    statements are checked on the default placement and on ``bank_row``."""
    workload, _M, _K = _samsung_paper_gemv()
    target = build_samsung_target()
    stock = allo.compile(
        workload, target, samsung_cost, backend="virtual", **REASSOCIATE
    )
    stock_scored = _samsung_scored(stock)
    bank_row = next(
        pl
        for _cycles, pl in stock_scored.values()
        if pl.mode.startswith("bank_row")
        and _samsung_residency(pl) == {"host"}
        and (pl.extra or {}).get("crf_issue") == "shared"
        and not (pl.extra or {}).get("stage_resident")
    )
    n = stock.placement_ranking.kernels[0].bucket_count

    def fill_start(estimate, unit):
        (span,) = [
            span
            for name, span in estimate.spans.items()
            if f":0.{unit}:pre:0:LD_A:" in name
        ]
        return span.start_cycle

    for name, layout in (
        ("default", list(stock.compiled.layout)),
        ("bank_row", [bank_row] * n),
    ):
        estimates = {
            b: allo.compile(
                workload,
                target,
                _rebound_cost(
                    samsung_cost,
                    samsung_costs,
                    f"samsung_burst_{name}_{b}",
                    HOST_BURST_CYCLES=b,
                ),
                layout=layout,
                backend="virtual",
                **REASSOCIATE,
            ).estimate()
            for b in (3, 4, 5)
        }
        cycles = {b: e.cycles for b, e in estimates.items()}
        assert cycles[4] - cycles[3] == 8 and cycles[5] - cycles[4] == 8, (name, cycles)
        # b=4: units 6 and 7 fit the gap after the first six fills.
        assert [fill_start(estimates[4], u) for u in (5, 6, 7)] == [20, 24, 28], name
        # b=3: unit 5 still fills at 20; units 6 and 7 first-fit behind the
        # LD_B and MAC slots, well after the gap they no longer fit.
        starts = [fill_start(estimates[3], u) for u in (5, 6, 7)]
        assert starts[0] == 20 and starts[1] > 28 and starts[2] == starts[1] + 4, (
            name,
            starts,
        )


VA_N = 12288


@_df_region()
def upmem_va(A: int32[VA_N], B: int32[VA_N], C: int32[VA_N]):
    @allo.work(mapping=[1], args=[A, B, C])
    def add(lA: int32[VA_N], lB: int32[VA_N], lC: int32[VA_N]):
        for i in range(VA_N):
            lC[i] = lA[i] + lB[i]


def test_upmem_calibration_flip_reaches_execution(monkeypatch, record_property):
    """A 10x calibration factor on the selected stratum moves the VA decision;
    both decisions execute exactly; legality and a fixed layout's program do
    not depend on the factor.

    Stated before running: the factor is a counterfactual (the anchor measured
    the stock factor), so the simulator keeps preferring the stock pick and
    the perturbed preference disagrees with it."""
    _sim_gate("upmem")
    target = build_upmem_target()
    stock = allo.compile(upmem_va, target, upmem_cost)
    stock_fingerprint = upmem_cost.bind(target).fingerprint
    stock_pick = _placements(stock.compiled.layout)
    key = ("pointwise", "fused_replicated", 12, "contiguous", 256, 32)
    assert upmem_costs._calibration_key(stock_pick[0].extra) == key

    factors = dict(upmem_costs.PHYSICAL_CALIBRATION_FACTORS)
    factors[key] = 10 * factors[key]
    monkeypatch.setattr(upmem_costs, "PHYSICAL_CALIBRATION_FACTORS", factors)
    assert upmem_cost.bind(target).fingerprint != stock_fingerprint
    moved = allo.compile(upmem_va, target, upmem_cost)
    pinned = allo.compile(upmem_va, target, upmem_cost, layout=stock_pick)
    moved_pick = _placements(moved.compiled.layout)
    stock_pick_under_moved = pinned.estimate().cycles
    monkeypatch.undo()
    moved_pick_under_stock = (
        allo.compile(upmem_va, target, upmem_cost, layout=moved_pick, backend="virtual")
        .estimate()
        .cycles
    )

    stock_decision, moved_decision = _upmem_decision(stock), _upmem_decision(moved)
    assert stock_decision["modes"] != moved_decision["modes"]
    assert (moved_pick[0].extra["data_layout"], moved_pick[0].extra["dma_bytes"]) != (
        "fused_replicated",
        256,
    )
    # The fixed stock layout compiles to the stock program under the
    # perturbed factors: same knobs, layout, device source and commands.
    assert _upmem_decision(pinned) == stock_decision
    assert stock_pick_under_moved > moved.estimate().cycles
    assert moved_pick_under_stock > stock.estimate().cycles

    seed = _seed()
    rng = np.random.default_rng(seed)
    A = rng.integers(-(1 << 31), (1 << 31) - 1, VA_N, dtype=np.int32)
    B = rng.integers(-(1 << 31), (1 << 31) - 1, VA_N, dtype=np.int32)
    want = (A.astype(np.int64) + B.astype(np.int64)).astype(np.uint32).astype(np.int32)
    cycles = {}
    for name, compiled in (("stock", stock), ("moved", moved)):
        C = np.zeros(VA_N, dtype=np.int32)
        result = compiled(A, B, C)
        assert result.kind == "simulator"
        np.testing.assert_array_equal(C, want, err_msg=f"{name} seed {seed}")
        cycles[name] = result.cycles
    perturbed_matches_simulator = cycles["moved"] < cycles["stock"]
    record_property(
        "upmem_cost_flip",
        {
            "stock": (
                stock_decision["modes"],
                stock.estimate().cycles,
                cycles["stock"],
            ),
            "moved": (
                moved_decision["modes"],
                moved.estimate().cycles,
                cycles["moved"],
            ),
            "stock_pick_under_moved": stock_pick_under_moved,
            "moved_pick_under_stock": moved_pick_under_stock,
            "perturbed_matches_simulator": perturbed_matches_simulator,
        },
    )
    assert cycles["stock"] < cycles["moved"], cycles
    assert not perturbed_matches_simulator


def _apu_contraction(left: float16[4, 8], right: float16[8, 6], result: float16[4, 6]):
    for row, column in allo.grid(4, 6):
        for depth in allo.reduction(8):
            result[row, column] += left[row, depth] * right[depth, column]


def test_apu_v1_pio_cost_flip_reaches_functional_execution(
    monkeypatch, record_property
):
    """PIO operand loads priced 20x: the plans that load operands by PIO
    (spatial reduction, SVP) lose to the DMA-coalescing plans; the DMA plans'
    estimates do not move (they issue no PIO load). Both selected plans run
    on the functional route (NumPy model of the plan's order) and agree with
    FP32 at FP16 tolerance. There is no APU v1 simulator: the functional
    cycles are the plan's estimate, so no simulator preference exists."""
    target = build_apu_v1_target()

    def compile_(**kwargs):
        return allo.compile(
            _apu_contraction,
            target,
            apu_v1_cost,
            backend="functional",
            **REASSOCIATE,
            **kwargs,
        )

    stock = compile_()
    stock_estimates = {e.plan.name: e.cycles for e in stock.candidate_estimates}
    stock_fingerprint = apu_v1_cost.bind(target).fingerprint
    monkeypatch.setattr(
        apu_v1_costs,
        "MICRO_PIO_LOAD_PER_ELEMENT",
        20 * apu_v1_costs.MICRO_PIO_LOAD_PER_ELEMENT,
    )
    assert apu_v1_cost.bind(target).fingerprint != stock_fingerprint
    moved = compile_()
    pinned = compile_(layout=stock.selected_plan.name)
    moved_estimates = {e.plan.name: e.cycles for e in moved.candidate_estimates}

    assert stock.selected_plan.name == "baseline_spatial_reduction"
    assert moved.selected_plan.name.startswith("temporal_dma_coalescing")
    for name in (
        "temporal_dma_coalescing",
        "temporal_dma_coalescing_broadcast_friendly",
    ):
        assert moved_estimates[name] == stock_estimates[name]
    for name in ("baseline_spatial_reduction", "temporal_svp"):
        assert moved_estimates[name] > stock_estimates[name]
    # Legality: the same candidate set; a fixed plan emits the same source.
    assert [c.name for c in moved.candidates] == [c.name for c in stock.candidates]
    assert pinned.device_source() == stock.device_source()
    assert moved.device_source() != stock.device_source()

    seed = _seed()
    rng = np.random.default_rng(seed)
    left = rng.normal(size=(4, 8)).astype(np.float16)
    right = rng.normal(size=(8, 6)).astype(np.float16)
    reference = _f32(left) @ _f32(right)
    runs = {}
    for name, compiled in (("stock", stock), ("moved", moved)):
        result = np.zeros((4, 6), dtype=np.float16)
        run = compiled(left, right, result)
        assert (
            run.kind == "functional"
            and run.extra["plan"] == compiled.selected_plan.name
        )
        np.testing.assert_allclose(
            _f32(result), reference, rtol=1e-2, atol=1e-2, err_msg=f"{name} seed {seed}"
        )
        runs[name] = run.cycles
    record_property(
        "apu_v1_cost_flip",
        {
            "stock": (stock.selected_plan.name, runs["stock"]),
            "moved": (moved.selected_plan.name, runs["moved"]),
        },
    )


# ===================================================================== #
# (d) Target-parameter variation
# ===================================================================== #

S8_M, S8_K, S8_ROWS = 4096, 256, 64


@_df_region()
def samsung_gemv_8x8(W: float16[S8_M, S8_K], x: float16[S8_K], y: float16[S8_M]):
    @allo.work(mapping=[8, 8], args=[W, x, y])
    def gemv(lW: float16[S8_M, S8_K], lx: float16[S8_K], ly: float16[S8_M]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * S8_ROWS
        for i in range(S8_ROWS):
            acc: float16 = 0
            for k in range(S8_K):
                acc += lW[row0 + i, k] * lx[k]
            ly[row0 + i] = acc


def _graph_channels(compiled):
    pattern = re.compile(r"pseudo_channel\[channel=(\d+)\]$")
    return {
        int(match.group(1))
        for activity in compiled.execution_graph.activities
        for use in activity.occupancy
        if (match := pattern.search(use.handle.name))
    }


def _graph_streams(compiled):
    return {a.id.split(":")[1] for a in compiled.execution_graph.activities}


def test_samsung_eight_channel_target(tmp_path, samsung_sim_gate, record_property):
    """8 pseudo-channels declared: the work grid, the bucket and stream counts,
    the channels the cost graph occupies and the rows per work item follow the
    declared extent; a 16-channel work mapping is rejected.

    The simulator is fixed at 64 channels and the runtime checks only the
    per-channel facts against it (design/samsung-execution.md section 2), so
    the 8-channel program executes on the simulator's own channel count: the
    same padded program as the 16-channel compile, same cycles."""
    t8 = _target_variant(
        tmp_path,
        _targets.build_samsung_target,
        'mapping={"channel": 16}',
        'mapping={"channel": 8}',
        "build_samsung_target_8ch",
    )
    t16 = build_samsung_target()
    assert t8.work_grid() == ([8, 8], 64)
    assert t16.work_grid() == ([16, 8], 128)

    paper, M, K = _samsung_paper_gemv()
    with pytest.raises(ValueError, match="outside 'pseudo_channel' extent 8"):
        allo.compile(paper, t8, samsung_cost, backend="virtual", **REASSOCIATE)

    c8 = allo.compile(
        samsung_gemv_8x8, t8, samsung_cost, backend="virtual", **REASSOCIATE
    )
    c16 = allo.compile(paper, t16, samsung_cost, backend="virtual", **REASSOCIATE)
    d8, d16 = _samsung_decision(c8), _samsung_decision(c16)
    assert (d8["bucket_count"], d16["bucket_count"]) == (64, 128)
    assert (len(_graph_streams(c8)), len(_graph_streams(c16))) == (64, 128)
    assert (_graph_channels(c8), _graph_channels(c16)) == (
        set(range(8)),
        set(range(16)),
    )
    (p8,), (p16,) = c8.compiled.layout_ctx.programs, c16.compiled.layout_ctx.programs
    assert (p8.facts.channels, p16.facts.channels) == (8, 16)
    assert (p8.work_items, p16.work_items) == (64, 128)
    assert (p8.contraction.rows_per_item, p16.contraction.rows_per_item) == (64, 32)
    assert p8.rows == p16.rows == M
    # Same per-pass program: the CRF and fiber placement do not depend on the
    # channel count.
    assert d8["crf"] == d16["crf"] and d8["lever"] == d16["lever"]
    record_property(
        "samsung_8ch_estimates", (c8.estimate().cycles, c16.estimate().cycles)
    )

    seed = _seed()
    rng = np.random.default_rng(seed)
    W = (rng.standard_normal((M, K)) * 0.1).astype(np.float16)
    x = (rng.standard_normal(K) * 0.1).astype(np.float16)
    runs = {}
    for name, workload, target in (
        ("8ch", samsung_gemv_8x8, t8),
        ("16ch", paper, t16),
    ):
        y = np.zeros(M, dtype=np.float16)
        result = allo.compile(workload, target, samsung_cost, **REASSOCIATE)(W, x, y)
        np.testing.assert_allclose(
            _f32(y),
            _f32(W) @ _f32(x),
            rtol=1e-2,
            atol=1e-2,
            err_msg=f"{name} seed {seed}",
        )
        runs[name] = (result.cycles, result.extra["padded_shape"], result.extra["crf"])
    record_property("samsung_8ch_sim_cycles", {k: v[0] for k, v in runs.items()})
    assert runs["8ch"] == runs["16ch"]


def test_upmem_thirty_two_dpu_target(tmp_path):
    """32 DPUs declared: the work grid follows; a 64-work-item mapping is
    rejected naming the extent; a 32-work-item GEMV gets the same partition,
    padding and estimate as on the 64-DPU rank (idle declared DPUs are not
    priced)."""
    from test_upmem_generality import uneven_gemv_region
    from allo.spmw_upmem import padding_fraction

    t32 = _target_variant(
        tmp_path,
        _targets.build_upmem_target,
        'mapping={"dpu": 64}',
        'mapping={"dpu": 32}',
        "build_upmem_target_32dpu",
    )
    t64 = build_upmem_target()
    assert t32.work_grid() == ([32, 24], 768)
    assert t64.work_grid() == ([64, 24], 1536)

    with pytest.raises(ValueError, match="outside 'dpu' extent 32"):
        allo.compile(
            uneven_gemv_region(391, 65, 64), t32, upmem_cost, backend="virtual"
        )

    on32 = allo.compile(
        uneven_gemv_region(391, 65, 32), t32, upmem_cost, backend="virtual"
    )
    on64 = allo.compile(
        uneven_gemv_region(391, 65, 32), t64, upmem_cost, backend="virtual"
    )
    (segment,) = _upmem_dpu_segments(on32)
    kernel = segment.kernel
    last = 391 - 31 * (391 // 32)
    assert kernel.dpus == 32 and kernel.geometry_value("rows") == last == 19
    assert [dict(g)["rows"] for g in kernel.streams] == [12] * 31 + [19]
    assert padding_fraction(kernel) == pytest.approx(1 - 391 / (19 * 32))
    assert _upmem_decision(on32) == _upmem_decision(on64)
    assert on32.estimate().cycles == on64.estimate().cycles


# ===================================================================== #
# (a) Renames with normalized decisions
# ===================================================================== #

_RENAMED_SAMSUNG = """
import allo
from allo.dataflow import region as zone
from allo.ir.types import float16 as half

ROWS_TOTAL, DEPTH, SLICE = 4096, 256, 32


@zone()
def projection_unit(weights: half[ROWS_TOTAL, DEPTH], probe: half[DEPTH], answer: half[ROWS_TOTAL]):
    @allo.work(mapping=[16, 8], args=[weights, probe, answer])
    def dot_rows(wt: half[ROWS_TOTAL, DEPTH], pv: half[DEPTH], out: half[ROWS_TOTAL]):
        lane_group, unit = allo.get_wid()
        base = (lane_group * 8 + unit) * SLICE
        for r in range(SLICE):
            total: half = 0
            for t in range(DEPTH):
                total += wt[base + r, t] * pv[t]
            out[base + r] = total
"""


@pytest.mark.parametrize("order", ["strict", "reassociate"])
def test_samsung_renamed_gemv_same_decision_and_cycles(
    tmp_path, samsung_sim_gate, record_property, order
):
    """Function, arguments, loops, locals, the region decorator alias and the
    module file renamed: the normalized decision (placement mode, knobs,
    program record, CRF and opcode inventory, ranking, layout text) and the
    executed cycles and outputs are identical."""
    path = tmp_path / "renamed_projection_kernel.py"
    path.write_text(_RENAMED_SAMSUNG)
    renamed = _load_module(path, "g6_renamed_projection").projection_unit
    original, M, K = _samsung_paper_gemv()
    target = build_samsung_target()
    kw = {"reduction_order": order}
    compiled = {
        "original": allo.compile(original, target, samsung_cost, **kw),
        "renamed": allo.compile(renamed, target, samsung_cost, **kw),
    }
    decisions = {name: _samsung_decision(c) for name, c in compiled.items()}
    assert decisions["renamed"] == decisions["original"]
    assert (
        compiled["renamed"].estimate().cycles == compiled["original"].estimate().cycles
    )

    seed = _seed()
    rng = np.random.default_rng(seed)
    W = (rng.standard_normal((M, K)) * 0.1).astype(np.float16)
    x = (rng.standard_normal(K) * 0.1).astype(np.float16)
    outputs, cycles = {}, {}
    for name, c in compiled.items():
        y = np.zeros(M, dtype=np.float16)
        result = c(W, x, y)
        outputs[name], cycles[name] = y, result.cycles
    record_property(
        "samsung_rename_cycles", (order, decisions["original"]["kind"], cycles)
    )
    np.testing.assert_allclose(
        _f32(outputs["original"]),
        _f32(W) @ _f32(x),
        rtol=1e-2,
        atol=1e-2,
        err_msg=f"seed {seed}",
    )
    np.testing.assert_array_equal(outputs["renamed"], outputs["original"])
    assert cycles["renamed"] == cycles["original"]


_RENAMED_UPMEM = """
import allo
from allo.dataflow import region as zone
from allo.ir.types import int32 as word

HEIGHT, WIDTH, CORES = 768, 128, 8
SHARE = HEIGHT // CORES


@zone()
def banded_product(table: word[HEIGHT, WIDTH], weights: word[WIDTH], sums: word[HEIGHT]):
    @allo.work(mapping=[CORES], args=[table, weights, sums])
    def per_core(tb: word[HEIGHT, WIDTH], wv: word[WIDTH], sv: word[HEIGHT]):
        (core,) = allo.get_wid()
        for r in range(SHARE):
            sv[core * SHARE + r] = 0
            for c in range(WIDTH):
                sv[core * SHARE + r] += tb[core * SHARE + r, c] * wv[c]
"""


def test_upmem_renamed_gemv_same_decision_and_cycles(tmp_path, record_property):
    """The UPMEM paper GEMV (768x128 over 8 DPUs) renamed throughout and loaded
    from another file: same knobs, layout text, kernel record (roles, not
    names), device source after a role-name substitution, command lines and
    simulated cycles; bit-identical outputs."""
    from test_gemv_all_targets import upmem_gemv

    path = tmp_path / "renamed_banded_product.py"
    path.write_text(_RENAMED_UPMEM)
    renamed = _load_module(path, "g6_renamed_banded").banded_product
    target = build_upmem_target()
    original = allo.compile(upmem_gemv, target, upmem_cost)
    moved = allo.compile(renamed, target, upmem_cost)
    names = {"tb": "local_W", "wv": "local_x", "sv": "local_y", "per_core": "mv"}
    (segment,) = _upmem_dpu_segments(moved)
    assert (
        dict(segment.kernel.roles).keys()
        == dict(_upmem_dpu_segments(original)[0].kernel.roles).keys()
    )
    assert _upmem_decision(moved, names) == _upmem_decision(original)
    assert moved.estimate().cycles == original.estimate().cycles

    _sim_gate("upmem")
    seed = _seed()
    rng = np.random.default_rng(seed)
    W = rng.integers(-1000, 1000, size=(768, 128), dtype=np.int32)
    x = rng.integers(-1000, 1000, size=128, dtype=np.int32)
    outputs, cycles = {}, {}
    for name, compiled in (("original", original), ("renamed", moved)):
        y = np.zeros(768, dtype=np.int32)
        result = compiled(W, x, y)
        outputs[name], cycles[name] = y, result.cycles
    record_property("upmem_rename_cycles", cycles)
    want = (W.astype(np.int64) @ x.astype(np.int64)).astype(np.uint32).astype(np.int32)
    np.testing.assert_array_equal(outputs["original"], want, err_msg=f"seed {seed}")
    np.testing.assert_array_equal(outputs["renamed"], outputs["original"])
    assert cycles["renamed"] == cycles["original"]


# ===================================================================== #
# (b) Operation or constant change at a fixed shape
# ===================================================================== #

EW_N = 70001  # off the 131,072-element tile; padded


@_df_region()
def ew_mad(a: float16[EW_N], b: float16[EW_N], d: float16[EW_N], c: float16[EW_N]):
    @allo.work(mapping=[1], args=[a, b, d, c])
    def k(la: float16[EW_N], lb: float16[EW_N], ld: float16[EW_N], lc: float16[EW_N]):
        for i in range(EW_N):
            lc[i] = la[i] * lb[i] + ld[i]


@_df_region()
def ew_mad_swapped(
    a: float16[EW_N], b: float16[EW_N], d: float16[EW_N], c: float16[EW_N]
):
    @allo.work(mapping=[1], args=[a, b, d, c])
    def k(la: float16[EW_N], lb: float16[EW_N], ld: float16[EW_N], lc: float16[EW_N]):
        for i in range(EW_N):
            lc[i] = la[i] * ld[i] + lb[i]


@_df_region()
def ew_add(a: float16[EW_N], b: float16[EW_N], c: float16[EW_N]):
    @allo.work(mapping=[1], args=[a, b, c])
    def k(la: float16[EW_N], lb: float16[EW_N], lc: float16[EW_N]):
        for i in range(EW_N):
            lc[i] = la[i] + lb[i]


@_df_region()
def ew_mul(a: float16[EW_N], b: float16[EW_N], c: float16[EW_N]):
    @allo.work(mapping=[1], args=[a, b, c])
    def k(la: float16[EW_N], lb: float16[EW_N], lc: float16[EW_N]):
        for i in range(EW_N):
            lc[i] = la[i] * lb[i]


@_df_region()
def ew_scaled_3(a: float16[EW_N], b: float16[EW_N], c: float16[EW_N]):
    @allo.work(mapping=[1], args=[a, b, c])
    def k(la: float16[EW_N], lb: float16[EW_N], lc: float16[EW_N]):
        for i in range(EW_N):
            lc[i] = la[i] * 3.0 + lb[i]


@_df_region()
def ew_scaled_half(a: float16[EW_N], b: float16[EW_N], c: float16[EW_N]):
    @allo.work(mapping=[1], args=[a, b, c])
    def k(la: float16[EW_N], lb: float16[EW_N], lc: float16[EW_N]):
        for i in range(EW_N):
            lc[i] = la[i] * 0.5 + lb[i]


def test_samsung_operation_change_at_fixed_shape(samsung_sim_gate, record_property):
    """At one shape: ADD and MUL emit different CRFs (the opcode follows the
    operation); the two MAD kernels differ only in which input is the
    multiplicand, so they emit the same CRF with the operands bound to
    different MAD sources (input buffers in source-operand order).
    Every program reads back its own FP16 result exactly."""
    target = build_samsung_target()
    seed = _seed()
    rng = np.random.default_rng(seed)
    a, b, d = ((rng.standard_normal(EW_N) * 0.5).astype(np.float16) for _ in range(3))
    cases = {
        "add": (ew_add, (a, b), a + b),
        "mul": (ew_mul, (a, b), a * b),
        "mad": (ew_mad, (a, b, d), a * b + d),
        "mad_swapped": (ew_mad_swapped, (a, b, d), a * d + b),
    }
    crf, inputs, outputs = {}, {}, {}
    for name, (workload, operands, reference) in cases.items():
        compiled = allo.compile(workload, target, samsung_cost)
        (program,) = compiled.compiled.layout_ctx.programs
        crf[name] = spmw_samsung._samsung_cmds_text(compiled.compiled.cmds)
        inputs[name] = program.elementwise.input_memrefs
        out = np.zeros(EW_N, dtype=np.float16)
        result = compiled(*operands, out)
        assert result.kind == "simulator"
        np.testing.assert_array_equal(
            out, reference.astype(np.float16), err_msg=f"{name} seed {seed}"
        )
        outputs[name] = out
    assert crf["add"] != crf["mul"]
    assert _opcode_inventory(crf["add"].splitlines()).get("ADD", 0) > 0
    assert _opcode_inventory(crf["mul"].splitlines()).get("MUL", 0) > 0
    assert crf["mad"] == crf["mad_swapped"]
    assert inputs["mad"] != inputs["mad_swapped"]
    assert not np.array_equal(outputs["mad"], outputs["mad_swapped"])
    assert not np.array_equal(outputs["add"], outputs["mul"])
    record_property("samsung_elementwise_inputs", inputs)


@pytest.mark.parametrize(
    "workload, literal", [(ew_scaled_3, "3.0"), (ew_scaled_half, "0.5")]
)
def test_samsung_literal_coefficient_is_rejected_naming_it(workload, literal):
    """A literal coefficient needs an SRF operand, which the Samsung target
    does not expose (targets.py, release audit C-01): the matcher rejects the
    expression and names the literal, so no constant is dropped."""
    with pytest.raises(
        UnsupportedWorkloadError,
        match=rf"add\(mul\(la\[%arg\d+\], {re.escape(literal)}\), lb\[%arg\d+\]\)",
    ):
        allo.compile(workload, build_samsung_target(), samsung_cost, backend="virtual")


AXPBY_N = 777


@_df_region()
def axpby_3_m7(A: int32[AXPBY_N], B: int32[AXPBY_N], C: int32[AXPBY_N]):
    @allo.work(mapping=[1], args=[A, B, C])
    def f(lA: int32[AXPBY_N], lB: int32[AXPBY_N], lC: int32[AXPBY_N]):
        for i in range(AXPBY_N):
            lC[i] = 3 * lA[i] + (-7) * lB[i]


@_df_region()
def axpby_2_5(A: int32[AXPBY_N], B: int32[AXPBY_N], C: int32[AXPBY_N]):
    @allo.work(mapping=[1], args=[A, B, C])
    def f(lA: int32[AXPBY_N], lB: int32[AXPBY_N], lC: int32[AXPBY_N]):
        for i in range(AXPBY_N):
            lC[i] = 2 * lA[i] + 5 * lB[i]


def test_upmem_axpby_coefficients_at_fixed_shape(record_property):
    """AXPBY (3, -7) versus (2, 5) at one shape: same family and knobs, the
    coefficients reach the kernel record and the device source, and each
    program computes its own int32 (wrapping) result."""
    target = build_upmem_target()
    compiled = {
        (3, -7): allo.compile(axpby_3_m7, target, upmem_cost),
        (2, 5): allo.compile(axpby_2_5, target, upmem_cost),
    }
    decisions = {k: _upmem_decision(c) for k, c in compiled.items()}
    kernels = {k: _upmem_dpu_segments(c)[0].kernel for k, c in compiled.items()}
    for coefficients, kernel in kernels.items():
        assert kernel.family == "pointwise"
        assert set(coefficients) <= {
            int(v) for _n, v in kernel.constants
        }, kernel.constants
    first, second = decisions[(3, -7)], decisions[(2, 5)]
    assert first["modes"] == second["modes"]
    assert first["segments"][0][-1] != second["segments"][0][-1]  # device source
    record_property("upmem_axpby_modes", first["modes"])

    _sim_gate("upmem")
    seed = _seed()
    rng = np.random.default_rng(seed)
    A = rng.integers(-(1 << 31), (1 << 31) - 1, AXPBY_N, dtype=np.int32)
    B = rng.integers(-(1 << 31), (1 << 31) - 1, AXPBY_N, dtype=np.int32)
    A[:3], B[:3] = [-(1 << 31), (1 << 31) - 1, 0], [(1 << 31) - 1, -(1 << 31), -1]
    a, b = A.astype(np.int64), B.astype(np.int64)
    results = {}
    for (alpha, beta), c in compiled.items():
        C = np.zeros(AXPBY_N, dtype=np.int32)
        run = c(A, B, C)
        assert run.kind == "simulator"
        want = (alpha * a + beta * b).astype(np.uint32).astype(np.int32)
        np.testing.assert_array_equal(C, want, err_msg=f"({alpha}, {beta}) seed {seed}")
        results[(alpha, beta)] = C
    assert not np.array_equal(results[(3, -7)], results[(2, 5)])


AIM_N = 4096


@_df_region()
def aim_ew_mul(a: bfloat16[AIM_N], b: bfloat16[AIM_N], c: bfloat16[AIM_N]):
    @allo.work(mapping=[32], args=[a, b, c])
    def k(la: bfloat16[AIM_N], lb: bfloat16[AIM_N], lc: bfloat16[AIM_N]):
        (ch,) = allo.get_wid()
        for i in range(128):
            lc[ch * 128 + i] = la[ch * 128 + i] * lb[ch * 128 + i]


@_df_region()
def aim_ew_add(a: bfloat16[AIM_N], b: bfloat16[AIM_N], c: bfloat16[AIM_N]):
    @allo.work(mapping=[32], args=[a, b, c])
    def k(la: bfloat16[AIM_N], lb: bfloat16[AIM_N], lc: bfloat16[AIM_N]):
        (ch,) = allo.get_wid()
        for i in range(128):
            lc[ch * 128 + i] = la[ch * 128 + i] + lb[ch * 128 + i]


AIM_M = AIM_K = 256


@_df_region()
def aim_gemv_mac(W: bfloat16[AIM_M, AIM_K], x: bfloat16[AIM_K], y: bfloat16[AIM_M]):
    @allo.work(mapping=[32], args=[W, x, y])
    def mv(lW: bfloat16[AIM_M, AIM_K], lx: bfloat16[AIM_K], ly: bfloat16[AIM_M]):
        (c,) = allo.get_wid()
        for i in range(8):
            acc: bfloat16 = 0
            for j in range(AIM_K):
                acc += lW[c * 8 + i, j] * lx[j]
            ly[c * 8 + i] = acc


@_df_region()
def aim_gemv_sub(W: bfloat16[AIM_M, AIM_K], x: bfloat16[AIM_K], y: bfloat16[AIM_M]):
    @allo.work(mapping=[32], args=[W, x, y])
    def mv(lW: bfloat16[AIM_M, AIM_K], lx: bfloat16[AIM_K], ly: bfloat16[AIM_M]):
        (c,) = allo.get_wid()
        for i in range(8):
            acc: bfloat16 = 0
            for j in range(AIM_K):
                acc += lW[c * 8 + i, j] - lx[j]
            ly[c * 8 + i] = acc


def _aim_opcodes(compiled):
    return Counter(
        line.split()[1]
        for line in map(str, compiled.compiled.cmds)
        if len(line.split()) > 1
    )


def test_aim_operation_change_at_fixed_shape():
    """Compile-only (AiM replays a timing-only trace). At one shape, MUL and
    ADD lower to EWMUL and EWADD; the MAC GEMV issues MAC_ABK, and the same
    nest with a subtraction in place of the product is rejected rather than
    lowered to a MAC."""
    target = build_aim_target()
    mul = allo.compile(aim_ew_mul, target, aim_cost, backend="virtual")
    add = allo.compile(aim_ew_add, target, aim_cost, backend="virtual")
    assert _aim_opcodes(mul)["EWMUL"] >= 1 and "EWADD" not in _aim_opcodes(mul)
    assert _aim_opcodes(add)["EWADD"] >= 1 and "EWMUL" not in _aim_opcodes(add)
    assert list(map(str, mul.compiled.cmds)) != list(map(str, add.compiled.cmds))

    mac = allo.compile(aim_gemv_mac, target, aim_cost, backend="virtual", **REASSOCIATE)
    assert _aim_opcodes(mac)["MAC_ABK"] >= 1
    with pytest.raises(UnsupportedWorkloadError, match="'aim'"):
        allo.compile(aim_gemv_sub, target, aim_cost, backend="virtual", **REASSOCIATE)


# One EWMUL column is 16 BF16 lanes in each of a channel's 4 bank groups;
# one EWADD column is 16 lanes of the GPR (GDDR6_AiM_org, 256-bit columns).
AIM_LANES, AIM_BANK_GROUPS = 16, 4


def _aim_issued_elements(compiled):
    issued = Counter()
    for line in map(str, compiled.compiled.cmds):
        fields = line.split()
        if fields[:2] == ["AiM", "EWMUL"]:
            columns, mask = int(fields[2]), int(fields[3])
            issued["mul"] += (
                columns * AIM_LANES * AIM_BANK_GROUPS * bin(mask).count("1")
            )
        elif fields[:2] == ["AiM", "EWADD"]:
            issued["add"] += int(fields[2]) * AIM_LANES
    return issued


def test_aim_elementwise_grid_issues_every_work_item():
    """M-14: the 32-item elementwise grids issue the program's 32 x 128
    elements. MUL places one work item per channel (one EWMUL, every
    channel's mask bit, 2 columns per channel), as the cost program prices
    it; ADD covers the aggregate in one GPR EWADD. A grid whose items do not
    tile the operands is refused, not lowered with work dropped."""
    target = build_aim_target()
    program = 32 * 128
    assert program == AIM_N
    mul = allo.compile(aim_ew_mul, target, aim_cost, backend="virtual")
    add = allo.compile(aim_ew_add, target, aim_cost, backend="virtual")
    assert list(map(str, mul.compiled.cmds)) == [
        f"AiM EWMUL 2 {2**32 - 1} 0",
        "AiM EOC",
    ]
    assert _aim_issued_elements(mul) == {"mul": program}
    assert _aim_issued_elements(add) == {"add": program}
    # The ADD estimate prices the issued EWADD inventory: one command of
    # 4096 / 16 = 256 GPR columns at one cycle each (it was 8 when the cost
    # program priced 32 parallel 8-column EWADDs).
    assert list(map(str, add.compiled.cmds)) == [
        f"AiM EWADD {program // AIM_LANES} 0 256",
        "AiM EOC",
    ]
    assert add.estimate().cycles == program // AIM_LANES == 256

    with pytest.raises(NotImplementedError, match="do not tile the operands"):
        allo.compile(_aim_ew_mul_half_grid, target, aim_cost, backend="virtual")


@_df_region()
def _aim_ew_mul_half_grid(a: bfloat16[AIM_N], b: bfloat16[AIM_N], c: bfloat16[AIM_N]):
    @allo.work(mapping=[16], args=[a, b, c])
    def k(la: bfloat16[AIM_N], lb: bfloat16[AIM_N], lc: bfloat16[AIM_N]):
        (ch,) = allo.get_wid()
        for i in range(128):
            lc[ch * 128 + i] = la[ch * 128 + i] * lb[ch * 128 + i]


def test_aim_elementwise_grid_simulates_every_work_item(aim_sim_gate):
    """M-14 in ramulator2: every channel issues its 2 EWMUL16 device commands
    (64 lanes each over the 4 bank groups), 32 x 2 x 64 = the program's 4096
    elements, and the run takes one channel's 2-column time (the channels
    run in parallel), within a few cycles of the estimate."""
    compiled = allo.compile(aim_ew_mul, build_aim_target(), aim_cost)
    a = np.ones(AIM_N, dtype=np.float32)
    result = compiled(a, a, np.zeros_like(a))
    per_channel = {
        int(ch): int(count)
        for ch, count in re.findall(
            r"CH(\d+)_num_EWMUL16_commands:\s*(\d+)", result.stdout
        )
    }
    assert per_channel == {ch: 2 for ch in range(32)}
    assert sum(per_channel.values()) * AIM_LANES * AIM_BANK_GROUPS == AIM_N
    estimate = compiled.estimate().cycles
    assert estimate <= result.cycles <= estimate + 4, (estimate, result.cycles)


# ===================================================================== #
# (k) Composed producer/consumer stages executed
# ===================================================================== #

_SM, _SK, _SR = 64, 64, 4


@_df_region()
def _mv2_alone(V: float16[_SM, _SK], y: float16[_SK], z: float16[_SM]):
    @allo.work(mapping=[16], args=[V, y, z])
    def mv2(lV: float16[_SM, _SK], ly: float16[_SK], lz: float16[_SM]):
        (c,) = allo.get_wid()
        for i in range(_SR):
            acc: float16 = 0
            for k in range(_SK):
                acc += lV[c * _SR + i, k] * ly[k]
            lz[c * _SR + i] = acc


@pytest.mark.parametrize("order", ["strict", "reassociate"])
def test_samsung_two_kernel_program_composes_transparently(
    samsung_sim_gate, record_property, order
):
    """``z = V @ (W @ x)`` through the two-kernel host program (gather y,
    broadcast y between launches): the consumer launch equals the consumer
    kernel compiled alone and run on the gathered y (bit-identical z), and
    both stages match the FP32 chain. Complements
    ``test_host_program_analysis.py::test_samsung_runs_chained_launches``."""
    from test_host_program_analysis import _chained, _two_kernels

    target = build_samsung_target()
    kw = {"reduction_order": order}
    chained = allo.compile(
        _two_kernels, target, samsung_cost, host_moves=_chained, **kw
    )
    alone = allo.compile(_mv2_alone, target, samsung_cost, **kw)

    seed = _seed()
    rng = np.random.default_rng(seed)
    W = rng.uniform(-0.5, 0.5, (_SM, _SK)).astype(np.float16)
    V = rng.uniform(-0.5, 0.5, (_SM, _SK)).astype(np.float16)
    x = rng.uniform(-1.0, 1.0, _SK).astype(np.float16)
    y = np.zeros(_SM, dtype=np.float16)
    z = np.zeros(_SM, dtype=np.float16)
    result = chained(W, V, x, y, z)
    assert result.kind == "simulator"
    assert [launch["kernel"] for launch in result.extra["launches"]] == ["mv1", "mv2"]

    z_alone = np.zeros(_SM, dtype=np.float16)
    single = alone(V, y.copy(), z_alone)
    np.testing.assert_array_equal(z, z_alone, err_msg=f"seed {seed}")
    assert result.extra["launches"][1]["cycles"] == single.cycles
    y_ref = _f32(W) @ _f32(x)
    z_ref = _f32(V) @ y_ref
    assert np.abs(_f32(y) - y_ref).max() <= 1e-2 * np.abs(y_ref).max(), f"seed {seed}"
    assert np.abs(_f32(z) - z_ref).max() <= 1e-2 * np.abs(z_ref).max(), f"seed {seed}"
    record_property(
        "samsung_two_kernel", (order, [l["cycles"] for l in result.extra["launches"]])
    )


# 44 x 3 distance pairs split evenly over the 12 tasklets of the template.
KM_POINTS, KM_DIM, KM_K = 44, 5, 3


@_df_region()
def kmeans_step(
    P: int32[KM_POINTS, KM_DIM],
    C: int32[KM_K, KM_DIM],
    D: int64[KM_POINTS, KM_K],
    C_new: int32[KM_K, KM_DIM],
    counts: int32[KM_K],
):
    @allo.work(mapping=[1], args=[P, C, D])
    def distances(
        lP: int32[KM_POINTS, KM_DIM],
        lC: int32[KM_K, KM_DIM],
        lD: int64[KM_POINTS, KM_K],
    ):
        for p in range(KM_POINTS):
            for c in range(KM_K):
                lD[p, c] = 0
                for d in range(KM_DIM):
                    lD[p, c] += (lP[p, d] - lC[c, d]) * (lP[p, d] - lC[c, d])

    @allo.work(mapping=[1], args=[P, D, C_new, counts])
    def update(
        uP: int32[KM_POINTS, KM_DIM],
        uD: int64[KM_POINTS, KM_K],
        uC: int32[KM_K, KM_DIM],
        ucounts: int32[KM_K],
    ):
        sums: int32[KM_K, KM_DIM] = 0
        for c0 in range(KM_K):
            ucounts[c0] = 0
        for p in range(KM_POINTS):
            best: int64 = uD[p, 0]
            idx: int32 = 0
            for c in range(KM_K):
                idx = c if uD[p, c] < best else idx
                best = uD[p, c] if uD[p, c] < best else best
            ucounts[idx] += 1
            for d in range(KM_DIM):
                sums[idx, d] += uP[p, d]
        for c1 in range(KM_K):
            for d1 in range(KM_DIM):
                if ucounts[c1] == 0:
                    uC[c1, d1] = 0
                else:
                    num: int32 = sums[c1, d1]
                    den: int32 = ucounts[c1]
                    adj: int32 = num + den // 2
                    if num < 0:
                        adj = num - den // 2
                    mag: int32 = (adj if adj >= 0 else -adj) // den
                    uC[c1, d1] = -mag if adj < 0 else mag


@allo.host_program(kmeans_step)
def kmeans_step_host(P, C, D, C_new, counts):
    allo.launch("distances", P, C, D)
    allo.launch("update", P, D, C_new, counts)


def _kmeans_oracle(P, C):
    P64, C64 = P.astype(np.int64), C.astype(np.int64)
    D = ((P64[:, None, :] - C64[None, :, :]) ** 2).sum(axis=2)
    nearest = np.argmin(D, axis=1)  # first minimum, as the strict `<` scan
    counts = np.bincount(nearest, minlength=C.shape[0]).astype(np.int64)
    sums = np.zeros(C.shape, dtype=np.int64)
    np.add.at(sums, nearest, P64)
    centers = np.zeros(C.shape, dtype=np.int64)
    for c in range(C.shape[0]):
        if counts[c]:
            num, den = sums[c], counts[c]
            adj = np.where(num < 0, num - den // 2, num + den // 2)
            mag = np.abs(adj) // den
            centers[c] = np.where(adj < 0, -mag, mag)
    return D, centers.astype(np.int32), counts.astype(np.int32)


def test_upmem_kmeans_distances_and_update_execute(record_property):
    """Two groups through ``allo.compile`` at a non-paper shape (44 points,
    5 dimensions, 3 clusters; the paper row is 120 x 8 x 4) on fresh inputs: DPU distances, then the host
    update group reading the distances; compared with an independent NumPy
    k-means step (squared distances, first-minimum assignment, rounded mean)."""
    target = build_upmem_target()
    compiled = allo.compile(
        kmeans_step, target, upmem_cost, host_moves=kmeans_step_host
    )
    segments = compiled.compiled.layout_ctx.segments
    assert [s.kind for s in segments] == ["dpu", "host"]
    assert segments[1].kernel.family == "kmeans_update"

    # The neighbour whose 37 x 3 pairs do not split over 12 tasklets is
    # refused by the template, not run on a partial split.
    from test_upmem_generality import kmeans_distances_region

    with pytest.raises(NotImplementedError, match="divide evenly across 12 tasklets"):
        allo.compile(kmeans_distances_region(37, KM_DIM, KM_K), target, upmem_cost)

    _sim_gate("upmem")
    seed = _seed()
    rng = np.random.default_rng(seed)
    P = rng.integers(-500, 500, size=(KM_POINTS, KM_DIM), dtype=np.int32)
    C = P[rng.choice(KM_POINTS, KM_K, replace=False)] + rng.integers(
        -20, 20, size=(KM_K, KM_DIM), dtype=np.int32
    )
    D = np.zeros((KM_POINTS, KM_K), dtype=np.int64)
    C_new = np.full((KM_K, KM_DIM), -1, dtype=np.int32)
    counts = np.full(KM_K, -1, dtype=np.int32)
    result = compiled(P, C, D, C_new, counts)
    assert result.kind == "simulator"
    D_ref, centers_ref, counts_ref = _kmeans_oracle(P, C)
    np.testing.assert_array_equal(D, D_ref, err_msg=f"seed {seed}")
    np.testing.assert_array_equal(counts, counts_ref, err_msg=f"seed {seed}")
    np.testing.assert_array_equal(C_new, centers_ref, err_msg=f"seed {seed}")
    record_property("upmem_kmeans_cycles", result.cycles)


# ===================================================================== #
# (n) Archive removal
# ===================================================================== #


@pytest.fixture
def no_archives(monkeypatch, tmp_path):
    monkeypatch.setenv("TENON_ARTIFACTS", str(tmp_path / "no-tenon-artifacts"))
    monkeypatch.setenv(
        "TENON_UPMEM_PROVENANCE_DIR", str(tmp_path / "no-upmem-provenance")
    )
    assert not spmw_simenv.tenon_artifacts_root().exists()
    assert not spmw_simenv.upimulator_provenance_dir().exists()
    return tmp_path


def test_samsung_executes_without_archives(no_archives, samsung_sim_gate):
    workload, M, K = _samsung_paper_gemv()
    compiled = allo.compile(workload, build_samsung_target(), samsung_cost)
    seed = _seed()
    rng = np.random.default_rng(seed)
    W = (rng.standard_normal((M, K)) * 0.1).astype(np.float16)
    x = (rng.standard_normal(K) * 0.1).astype(np.float16)
    y = np.zeros(M, dtype=np.float16)
    result = compiled(W, x, y)
    assert result.kind == "simulator" and result.cycles > 0
    np.testing.assert_allclose(
        _f32(y), _f32(W) @ _f32(x), rtol=1e-2, atol=1e-2, err_msg=f"seed {seed}"
    )


def test_aim_executes_without_archives(no_archives, aim_sim_gate):
    compiled = allo.compile(aim_gemv_mac, build_aim_target(), aim_cost, **REASSOCIATE)
    rng = np.random.default_rng(_seed())
    W = (rng.standard_normal((AIM_M, AIM_K)) * 0.1).astype(np.float32)
    x = (rng.standard_normal(AIM_K) * 0.1).astype(np.float32)
    y = np.zeros(AIM_M, dtype=np.float32)
    result = compiled(W, x, y)
    # AiM is timing-only: the trace replays in ramulator2, no outputs.
    assert result.kind == "simulator-timing-only" and result.cycles > 0
    assert "[AiMTrace]" in result.stdout


def test_upmem_without_archives_names_the_variables_or_runs_a_user_build(no_archives):
    from allo.spmw_codegen import SimulatorUnavailable

    compiled = allo.compile(upmem_va, build_upmem_target(), upmem_cost)
    rng = np.random.default_rng(_seed())
    A = rng.integers(-1000, 1000, VA_N, dtype=np.int32)
    B = rng.integers(-1000, 1000, VA_N, dtype=np.int32)
    C = np.zeros(VA_N, dtype=np.int32)
    if not spmw_simenv.upmem_user_build():
        reason = spmw_simenv.upmem_unavailable_reason()
        for variable in (
            "UPIMULATOR_ROOT",
            "UPIMULATOR_BIN",
            "TENON_ARTIFACTS",
            "TENON_UPMEM_PROVENANCE_DIR",
        ):
            assert variable in reason, reason
        with pytest.raises(SimulatorUnavailable, match="UPIMULATOR_BIN"):
            compiled(A, B, C)
        assert not C.any()
        return
    _sim_gate("upmem")
    result = compiled(A, B, C)
    np.testing.assert_array_equal(C, A + B)
    assert result.extra["upmem"][0]["sdk_build"].startswith("user-build:")


_SDIST_SMOKE = """
import os
import sys

import allo
from allo.pim.costs import samsung_cost
from allo.pim.targets import build_samsung_target

# Paths are compared unresolved: allo/_mlir is a symlink to the built
# bindings, whose own files link into the LLVM source tree.
root = os.path.abspath(sys.argv[1]) + os.sep
scripts = os.path.abspath(sys.argv[2]) + os.sep
assert os.path.abspath(allo.__file__).startswith(root), allo.__file__
sys.path.insert(0, scripts)
from smoke_workload import gemv

compiled = allo.compile(
    gemv, build_samsung_target(), samsung_cost, backend="virtual",
    reduction_order="reassociate",
)
stdlib = os.path.dirname(os.__file__) + os.sep
leaked = sorted(
    name
    for name, module in list(sys.modules.items())
    if getattr(module, "__file__", None)
    and not os.path.abspath(module.__file__).startswith((root, scripts, stdlib))
    and "site-packages" not in module.__file__
)
print("MODE", compiled.compiled.layout_ctx.programs[0].lever_record["mode"])
print("CYCLES", compiled.estimate().cycles)
print("LEAKED", ",".join(leaked))
"""

_SDIST_WORKLOAD = """
import allo
from allo.dataflow import region
from allo.ir.types import float16

M, K, ROWS = 4096, 256, 32


@region()
def gemv(W: float16[M, K], x: float16[K], y: float16[M]):
    @allo.work(mapping=[16, 8], args=[W, x, y])
    def mv(lW: float16[M, K], lx: float16[K], ly: float16[M]):
        pid, uid = allo.get_wid()
        row0 = (pid * 8 + uid) * ROWS
        for i in range(ROWS):
            acc: float16 = 0
            for k in range(K):
                acc += lW[row0 + i, k] * lx[k]
            ly[row0 + i] = acc
"""


def test_sdist_tree_compiles_samsung_gemv_outside_the_checkout(tmp_path, no_archives):
    """Build the sdist, unpack it outside the checkout, add only the built
    MLIR bindings (``allo/_mlir`` is a build product, not an sdist member),
    and compile the Samsung GEMV on the virtual backend in a fresh
    interpreter whose ``allo`` and every imported non-stdlib module come from
    the unpacked tree."""
    live_mlir = REPO / "allo" / "_mlir"
    if not live_mlir.is_dir():
        pytest.skip("no built allo/_mlir bindings in this checkout")
    if importlib.util.find_spec("setuptools") is None:
        pytest.skip("setuptools is not installed")
    dist = tmp_path / "dist"
    if importlib.util.find_spec("build") is not None:
        command = [
            sys.executable,
            "-m",
            "build",
            "--sdist",
            "--no-isolation",
            "--outdir",
            str(dist),
            str(REPO),
        ]
    else:  # the same fallback as tests/perf/test_packaging.py::sdist_members
        command = [
            sys.executable,
            "setup.py",
            "--quiet",
            "egg_info",
            "--egg-base",
            str(dist),
            "sdist",
            "--dist-dir",
            str(dist),
        ]
    dist.mkdir()
    built = subprocess.run(
        command, cwd=REPO, capture_output=True, text=True, timeout=600, check=False
    )
    assert built.returncode == 0, built.stderr[-2000:]
    (archive,) = dist.glob("*.tar.gz")
    unpack = tmp_path / "unpacked"
    with tarfile.open(archive) as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(unpack, filter="data")
        else:
            tar.extractall(unpack)
    (tree,) = [p for p in unpack.iterdir() if p.is_dir()]
    assert not (tree / "allo" / "_mlir").exists()
    assert not (tree / "tests").exists()
    (tree / "allo" / "_mlir").symlink_to(live_mlir.resolve(), target_is_directory=True)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "smoke.py").write_text(_SDIST_SMOKE)
    (scripts / "smoke_workload.py").write_text(_SDIST_WORKLOAD)

    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")
    }
    env["PYTHONPATH"] = str(tree)
    env["HOME"] = str(tmp_path / "home")
    run = subprocess.run(
        [sys.executable, "-s", str(scripts / "smoke.py"), str(tree), str(scripts)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert run.returncode == 0, (run.stdout + run.stderr)[-3000:]
    lines = dict(line.split(" ", 1) for line in run.stdout.splitlines() if " " in line)
    assert lines["MODE"].startswith("split_rows"), run.stdout
    assert int(lines["CYCLES"]) > 0
    assert lines.get("LEAKED", "") == "", lines.get("LEAKED")
