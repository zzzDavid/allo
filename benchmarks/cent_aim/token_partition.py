# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The paper's token-partitioning study (5-evaluation.tex "Token Partitioning")
as a maintained reproduction driver (release gate G10, inventory row
``token_partitioning_nine_workloads``).

Nine resident-input decode-attention workloads (1/8/32 heads x 128/512/4096
tokens, head dim 128) compiled through ``allo.compile`` on the AiM target.
For each workload the shared ranking (the cost program over the enumerated
candidates) is read once; the paper's device had eight channels, so the
compared plans are the ranking's best SV placement per reduction split
P in {1, 2, 4, 8} among the candidates that use at most eight channel
replicas (``CHANNEL_BUDGET``), plus the head-only control (one head per
channel, no split).  The paper's selection is the cheapest of the four by
the cost program; the simulator decides which is fastest.  Every compared
plan is compiled as an explicit layout (the supplied placement is rebound
and its declaration validated by the shared boundary), its estimate read
from the virtual backend, its device cycles from the AiM simulator
(timing-only ramulator2; ``kind == "simulator-timing-only"`` is asserted)
and its host merge priced by the declared 32 GB/s link model; the merge's
numerics are checked on integer and BF16 operands.

Comparison boundaries (the inventory's): scoring + global softmax + value
sum + host stages/reductions with inputs resident; the cache append and the
projections are outside.  Reported: per workload the total (device + host)
cycles of every plan, the selected P, the simulated best P and the gap;
over the nine, the minima reached and the largest gap; for 1 head x 4096
tokens the head-only / selected ratio.  The unrestricted default selection
of the shared pipeline on the simulated 32-channel device is recorded for
transparency and is not the paper's metric.  A failed compile or run, an
inexact merge or an execution of the wrong kind is recorded and exits
nonzero; nothing here is read by the compiler.

Usage::

    python -m benchmarks.cent_aim.token_partition --out tp.json [--no-sim]
        [--shapes 1x128,8x512]

The workloads are defined here (eager region annotations); the tests in
``tests/pim/test_aim_token_partition.py`` import them.
"""

import argparse
import inspect
import json
import math
import platform
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np

import allo
from allo.dataflow import region
from allo.ir.types import bfloat16
from allo.pim.aim_lowering import (
    _target_geometry,
    bf16_round,
    reference_partial_sums,
)
from allo.pim.costs import aim_cost
from allo.pim.costs.aim import host_reduction_cycles
from allo.pim.targets import build_aim_target
from allo.spmw_autoschedule import (
    _aim_enumerate,
    _bucket_for_autoschedule,
    _matcher_work_scope,
)

# The token split reassociates a floating-point reduction: compile under the
# explicit permission when this build has the reduction-order policy.
POLICY = (
    {"reduction_order": "reassociate"}
    if "reduction_order" in inspect.signature(allo.compile).parameters
    else {}
)

HEAD_DIM = 128
MAX_TOKENS = 4096
CHANNEL_BUDGET = 8  # the paper's eight simulated channels
PARTITIONS = (1, 2, 4, 8)
SHAPES = tuple((heads, tokens) for heads in (1, 8, 32) for tokens in (128, 512, 4096))

# The paper's reported outcome of this study (sections/5-evaluation.tex).
PAPER = {
    "minima_reached": 7,
    "workloads": 9,
    "max_gap": 0.0263,
    "head_only_over_selected_1x4096": 1.071,
}


def build_attention(heads, tokens, head_dim=HEAD_DIM, max_tokens=MAX_TOKENS):
    """Resident-input decode attention: scores = q.K, heads_out = probs.V.

    The host softmax between the kernels is identical for every partition
    choice and is not a compiler-priced stage; ``probs`` is resident.
    """
    H, L, Dh, M = heads, tokens, head_dim, max_tokens

    @region()
    def attention(
        q: bfloat16[H, Dh],
        k_cache: bfloat16[H, M, Dh],
        scores: bfloat16[H, L],
        probs: bfloat16[H, L],
        v_cache: bfloat16[H, Dh, M],
        heads_out: bfloat16[H, Dh],
    ):
        @allo.work(mapping=[1], args=[q, k_cache, scores])
        def attention_qk(
            lq: bfloat16[H, Dh], kc: bfloat16[H, M, Dh], sc: bfloat16[H, L]
        ):
            for h in range(H):
                for s in range(L):
                    acc: bfloat16 = 0
                    for d in range(Dh):
                        acc += lq[h, d] * kc[h, s, d]
                    sc[h, s] = acc

        @allo.work(mapping=[1], args=[probs, v_cache, heads_out])
        def attention_sv(
            pr: bfloat16[H, L], vc: bfloat16[H, Dh, M], o: bfloat16[H, Dh]
        ):
            for h in range(H):
                for d in range(Dh):
                    acc: bfloat16 = 0
                    for s in range(L):
                        acc += pr[h, s] * vc[h, d, s]
                    o[h, d] = acc

    def host(q, k_cache, scores, probs, v_cache, heads_out):
        allo.launch("attention_qk", q, k_cache, scores)
        allo.launch("attention_sv", probs, v_cache, heads_out)

    program = allo.host_program(
        attention, resident=("q", "k_cache", "probs", "v_cache")
    )(host)
    return attention, program


def build_sv(heads, tokens, head_dim=HEAD_DIM, max_tokens=MAX_TOKENS, append=False):
    """SV alone; ``append`` adds the decode step's value-cache append (CENT's
    broadcast of token L-1) before the launch."""
    H, L, Dh, M = heads, tokens, head_dim, max_tokens

    @region()
    def attention(
        probs: bfloat16[H, L], v_cache: bfloat16[H, Dh, M], heads_out: bfloat16[H, Dh]
    ):
        @allo.work(mapping=[1], args=[probs, v_cache, heads_out])
        def attention_sv(
            pr: bfloat16[H, L], vc: bfloat16[H, Dh, M], o: bfloat16[H, Dh]
        ):
            for h in range(H):
                for d in range(Dh):
                    acc: bfloat16 = 0
                    for s in range(L):
                        acc += pr[h, s] * vc[h, d, s]
                    o[h, d] = acc

    def host(probs, v_cache, heads_out):
        if append:
            hx = allo.host_xfer
            hx.broadcast(v_cache[:, :, L - 1], hx.banks)
        allo.launch("attention_sv", probs, v_cache, heads_out)

    # The cache before this decode step is on the device in both forms: the
    # append writes token L-1 into a resident cache the kernel then reads
    # over [0, L) (availability is explicit, review round 22).
    resident = ("probs", "v_cache")
    return attention, allo.host_program(attention, resident=resident)(host)


# --------------------------------------------------------------------- #
# Selection helpers
# --------------------------------------------------------------------- #


def knob(mode, name, default=None):
    marker = f"+{name}="
    if marker not in mode:
        return default
    return mode.split(marker, 1)[1].split("+", 1)[0]


def partitions_of(mode):
    return int(knob(mode, "reduction_partitions", 1))


def within_budget(mode, budget=CHANNEL_BUDGET):
    return int(knob(mode, "replica_partitions", 1)) <= budget


def ranked(heads, tokens, *, qk=True):
    """The shared ranking of one workload: ``(target, region, host program,
    compiled)``, the compiled object carrying the placement ranking."""
    if qk:
        attention, program = build_attention(heads, tokens)
    else:
        attention, program = build_sv(heads, tokens)
    target = build_aim_target()
    compiled = allo.compile(
        attention, target, aim_cost, backend="virtual", host_moves=program, **POLICY
    )
    return target, attention, program, compiled


def kernel_choices(target, compiled):
    """``{kernel name: (bucket index, [(cycles, mode, placement)])}``."""
    trace = compiled.compiled.trace
    buckets = _bucket_for_autoschedule(trace)
    out = {}
    for kernel in compiled.compiled.placement_ranking.kernels:
        index, matches = next(
            (i, m)
            for i, (_scope, m) in enumerate(buckets)
            if _matcher_work_scope(m[0]).group_id == kernel.group_id
        )
        by_mode = {c.mode: c for c in _aim_enumerate(target, matches)}
        name = matches[0].func_name.rsplit("_", 1)[0]
        out[name] = (
            index,
            [(cycles, mode, by_mode[mode]) for _pos, cycles, mode in kernel.scores],
        )
    return out


def budget_variants(target, compiled, budget=CHANNEL_BUDGET):
    """The cost program's best placement per decision inside the channel
    budget: the QK choice, the SV choice for each P, and the two controls."""
    choices = kernel_choices(target, compiled)
    qk_index, qk_scores = choices.get("attention_qk", (None, []))
    sv_index, sv_scores = choices["attention_sv"]
    qk = min((s for s in qk_scores if within_budget(s[1], budget)), default=None)
    sv = [s for s in sv_scores if within_budget(s[1], budget)]
    per_partition = {}
    for cycles, mode, placement in sv:
        partitions = partitions_of(mode)
        if partitions in PARTITIONS and (
            partitions not in per_partition or cycles < per_partition[partitions][0]
        ):
            per_partition[partitions] = (cycles, mode, placement)
    head_only = min(
        (
            s
            for s in sv
            if partitions_of(s[1]) == 1 and knob(s[1], "batch_mapping") == "channels"
        ),
        key=lambda s: s[0],
    )
    selected = min(per_partition.values(), key=lambda s: s[0])
    return SimpleNamespace(
        qk=qk,
        qk_index=qk_index,
        sv_index=sv_index,
        per_partition=per_partition,
        head_only=head_only,
        fixed_eight=per_partition.get(8),
        selected_partitions=partitions_of(selected[1]),
    )


def compile_variant(attention, program, target, variants, sv_choice):
    layouts = [None, None]
    layouts[variants.sv_index] = sv_choice[2]
    if variants.qk is not None:
        layouts[variants.qk_index] = variants.qk[2]
    layouts = [layout for layout in layouts if layout is not None]
    return allo.compile(
        attention, target, aim_cost, host_moves=program, layout=layouts, **POLICY
    )


def sv_entry(compiled):
    manifest = compiled.compiled.layout_ctx.lowering_manifest
    return next(
        entry
        for entry in manifest
        if entry["kind"] == "contraction"
        and str(entry["name"]).startswith("attention_sv")
    )


def check_merge_numerics(compiled, heads, tokens, seed=0):
    """Check the lowered partition ownership and the host merge on integer
    (exact) and BF16 operands; returns the BF16 max relative error (None
    for an unsplit program).  Raises on an inexact integer merge or a BF16
    error beyond a few units of the BF16 epsilon."""
    entry = sv_entry(compiled)
    split = entry.get("reduction_partition")
    if split is None:
        return None
    host = next(
        step
        for step in compiled.compiled.layout_ctx.host_steps
        if step.source == entry["name"]
    )
    rng = np.random.default_rng(seed + heads + tokens)
    # Integer operands: exact partials and exact merge.
    values = rng.integers(-16, 17, (heads, HEAD_DIM, tokens))
    probs = rng.integers(0, 9, (heads, tokens))
    partials = reference_partial_sums(entry, values, probs, arithmetic="exact")
    merged = host.merge(partials)[0]
    np.testing.assert_array_equal(merged, np.einsum("hds,hs->hd", values, probs))
    # BF16 operands under the declared readout contract.
    values = bf16_round(
        rng.standard_normal((heads, HEAD_DIM, tokens)).astype(np.float32)
    )
    probs = bf16_round(rng.random((heads, tokens)).astype(np.float32))
    merged = host.merge(reference_partial_sums(entry, values, probs))[0]
    reference = np.einsum(
        "hds,hs->hd", values.astype(np.float64), probs.astype(np.float64)
    )
    scale = np.max(np.abs(reference))
    error = float(np.max(np.abs(merged - reference)) / scale)
    # BF16 readout per row chunk: a few units of the 2^-8 BF16 epsilon.
    if not error < 2.0**-6:
        raise AssertionError(f"BF16 merge error {error} exceeds 2^-6")
    return error


# --------------------------------------------------------------------- #
# The nine-workload experiment
# --------------------------------------------------------------------- #


def _json_ready(value, depth=0):
    if depth > 24:
        return repr(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_ready(v, depth + 1) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "manifest"):
        return _json_ready(value.manifest(), depth + 1)
    return repr(value)


def _placements(compiled):
    stored = compiled.compiled.layout
    return list(stored) if isinstance(stored, (list, tuple)) else [stored]


def _plan_provenance(compiled):
    """What was compiled: every placement's mode and declared effects, the
    SV lowering entry (partition ownership, channel groups) and the host
    steps."""
    ctx = compiled.compiled.layout_ctx
    return {
        "placements": [
            {
                "mode": p.mode,
                "effects": _json_ready(p.extra.get("effects")),
                "candidate_selection": p.extra.get("candidate_selection"),
            }
            for p in _placements(compiled)
        ],
        "sv_lowering": _json_ready(sv_entry(compiled)),
        "host_steps": [_json_ready(step) for step in getattr(ctx, "host_steps", ())],
    }


def run_workload(heads, tokens, *, simulate=True):
    """Compile and (optionally) simulate every compared plan of one workload.

    Returns a namespace with ``rows`` (label -> row), ``selected`` and
    ``best`` P, ``gap``, the unrestricted default selection and a JSON-ready
    ``record``.  A failed plan raises; the caller records it.
    """
    target, attention, program, ranking = ranked(heads, tokens)
    variants = budget_variants(target, ranking)
    geometry = _target_geometry(target)
    default = [p.mode for p in _placements(ranking)]
    rows = {}
    labelled = [
        (f"P{p}", choice) for p, choice in sorted(variants.per_partition.items())
    ]
    if variants.head_only[1] != variants.per_partition[1][1]:
        labelled.append(("head-only", variants.head_only))
    for label, choice in labelled:
        built = compile_variant(attention, program, target, variants, choice)
        host = sum(
            host_reduction_cycles(step, geometry)
            for step in built.compiled.layout_ctx.host_steps
        )
        estimate = (
            allo.compile(
                attention,
                target,
                aim_cost,
                backend="virtual",
                host_moves=program,
                layout=built.compiled.layout,
                **POLICY,
            )
            .run_backend()
            .cycles
        )
        execution = None
        device = None
        if simulate:
            result = built.run_backend()
            if result.kind != "simulator-timing-only":
                raise RuntimeError(
                    f"{label}: expected a simulator-timing-only run, got {result.kind!r}"
                )
            if not isinstance(result.cycles, int) or result.cycles <= 0:
                raise RuntimeError(f"{label}: simulator reported cycles {result.cycles!r}")
            device = int(result.cycles)
            execution = {
                "kind": result.kind,
                "simulator": result.extra.get("simulator"),
                "segments": result.extra.get("segments"),
                "unique_segments": result.extra.get("unique_segments"),
                "returncode": result.extra.get("returncode"),
            }
        rows[label] = SimpleNamespace(
            mode=choice[1],
            ranked_cycles=choice[0],
            partitions=partitions_of(choice[1]),
            estimate=estimate,
            device=device,
            host=host,
            total=None if device is None else device + host,
            bf16_error=check_merge_numerics(built, heads, tokens),
            execution=execution,
            provenance=_plan_provenance(built),
        )
    if "head-only" not in rows:
        rows["head-only"] = rows["P1"]
    rows["fixed-8"] = rows["P8"]
    best = min((rows[f"P{p}"] for p in PARTITIONS), key=lambda r: r.total or r.estimate)
    chosen = rows[f"P{variants.selected_partitions}"]
    gap = (chosen.total / best.total - 1.0) if simulate else None
    head_only_ratio = (
        rows["head-only"].total / chosen.total if simulate else None
    )
    record = {
        "heads": heads,
        "tokens": tokens,
        "channel_budget": CHANNEL_BUDGET,
        "default_selection_unrestricted": default,
        "default_within_budget": all(within_budget(m) for m in default),
        "selected_partitions": variants.selected_partitions,
        "best_partitions": best.partitions,
        "gap": gap,
        "head_only_over_selected": head_only_ratio,
        "plans": {
            label: {
                "mode": row.mode,
                "partitions": row.partitions,
                "ranked_cycles": row.ranked_cycles,
                "estimate_cycles": row.estimate,
                "device_cycles": row.device,
                "host_cycles": row.host,
                "total_cycles": row.total,
                "bf16_merge_error": row.bf16_error,
                "execution": row.execution,
                "provenance": row.provenance,
            }
            for label, row in rows.items()
        },
    }
    return SimpleNamespace(
        heads=heads,
        tokens=tokens,
        rows=rows,
        selected=variants.selected_partitions,
        best=best.partitions,
        gap=gap,
        default=default,
        record=record,
    )


def summarize(results):
    """The study's figures over the completed workloads."""
    complete = [r for r in results if r.get("gap") is not None]
    reached = sum(r["selected_partitions"] == r["best_partitions"] for r in complete)
    gaps = [r["gap"] for r in complete]
    one_4096 = next(
        (r for r in complete if (r["heads"], r["tokens"]) == (1, 4096)), None
    )
    return {
        "workloads_completed": len(complete),
        "minima_reached": reached,
        "max_gap": max(gaps) if gaps else None,
        "head_only_over_selected_1x4096": (
            one_4096["head_only_over_selected"] if one_4096 else None
        ),
        "paper": PAPER,
    }


def _parse_shapes(text):
    shapes = []
    for item in text.split(","):
        heads, tokens = item.lower().split("x")
        shapes.append((int(heads), int(tokens)))
    return tuple(shapes)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-sim", action="store_true", help="estimates only")
    parser.add_argument(
        "--shapes", default=None, help="subset, e.g. 1x128,8x512 (default: all nine)"
    )
    args = parser.parse_args(argv)
    shapes = _parse_shapes(args.shapes) if args.shapes else SHAPES
    simulate = not args.no_sim
    session = {
        "host": platform.node(),
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.executable,
        "allo": allo.__file__,
        "simulate": simulate,
        "shapes": [list(s) for s in shapes],
        "workloads": [],
    }
    failures = 0
    print(
        "heads tokens | P1 P2 P4 P8 total(estimate) | head-only | selected P | "
        "best P | gap | merge bf16 err | default (unrestricted)",
        flush=True,
    )
    for heads, tokens in shapes:
        started = time.perf_counter()
        try:
            result = run_workload(heads, tokens, simulate=simulate)
            record = result.record
            rows = result.rows
            totals = " ".join(
                f"{rows[f'P{p}'].total if simulate else rows[f'P{p}'].estimate}"
                f"({rows[f'P{p}'].estimate})"
                for p in PARTITIONS
            )
            errors = max((r.bf16_error or 0.0) for r in rows.values())
            print(
                f"{heads:5d} {tokens:6d} | {totals} | "
                f"{rows['head-only'].total if simulate else rows['head-only'].estimate} | "
                f"{result.selected} | {result.best} | "
                f"{(result.gap or 0.0) * 100:.2f}% | {errors:.2e} | {result.default}",
                flush=True,
            )
        except Exception as error:  # a failed workload is evidence too
            failures += 1
            record = {
                "heads": heads,
                "tokens": tokens,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
            print(f"{heads:5d} {tokens:6d} | FAILED {record['error']}", flush=True)
        record["seconds"] = time.perf_counter() - started
        session["workloads"].append(record)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(session, handle, indent=1, default=str)
    session["summary"] = summarize(session["workloads"])
    session["status"] = "pass" if not failures else f"fail ({failures} workload(s))"
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(session, handle, indent=1, default=str)
    summary = session["summary"]
    print(
        f"selection reaches {summary['minima_reached']} of "
        f"{summary['workloads_completed']} simulated minima"
        + (
            f"; max gap {summary['max_gap'] * 100:.2f}%"
            if summary["max_gap"] is not None
            else ""
        )
        + f" (paper: {PAPER['minima_reached']} of {PAPER['workloads']}, "
        f"{PAPER['max_gap'] * 100:.2f}%)",
        flush=True,
    )
    print(session["status"], flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
