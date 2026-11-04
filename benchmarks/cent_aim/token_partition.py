# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The paper's token-partitioning study (5-evaluation.tex "Token Partitioning")
as a maintained reproduction driver (release gate G10, inventory row
``token_partitioning_nine_workloads``).

Nine resident-input decode-attention workloads (1/8/32 heads x
128/512/4096 tokens, head dim 128) compiled through ``allo.compile`` on the
AiM target under the paper's eight-channel device budget expressed as the
compiler's resource policy (``resources={"channel": 8}``).  Since round 42
the program is the bridged attention: QK scores -> global softmax on the
host -> probabilities -> SV, with the softmax a declared host stage of the
host program (``allo.host_compute``) that the schedule orders, the
availability check binds (the scores produced by QK before it, the
probabilities reaching SV's GB preload), the AiM program lowers as a host
step and the cost program prices in the graph the normal selection ranks
(``host_stage:softmax``: the readout of the scores and the upload of the
probabilities over the declared 32 B/cycle link, plus the declared host
vector passes).  The partial-result readout of a split SV is priced by the
host reduction, so nothing is counted twice.

* The **normal selection** under the budget is the headline plan
  (``default``); its P is the study's "selected" partition count.  The
  forced alternatives are the ablation: the budgeted ranking's cheapest SV
  plan per P in {1, 2, 4, 8} and the head-only control (one head per
  channel, no split), compiled as explicit layouts under the same budget.
  Aliases are decided before anything runs and never replace a failure.
* Per plan: the estimate from the virtual backend, the device cycles from
  the AiM simulator (timing-only ramulator2; ``kind == "simulator-timing-only"``
  asserted; no values exist and none are attributed to it), the priced host
  cycles (stage + reduction, with the breakdown), **independent functional
  evidence** (the lowered QK program's oracle -> the compiled stage's own
  function -> the lowered SV program's oracle, against a float64 reference;
  ``functional_check``), the integer-exact merge check of a split plan,
  measured host-stage samples (``host_stage_measured``, recorded beside
  the modelled price and never substituted for it), and complete
  provenance and artifacts written before the next plan runs; a failed
  plan is retained with its evidence.
* Controls: the head-only plan; the CENT-derived attention program of the
  same token length (``benchmarks.cent_aim`` ``attention_l<L>``, CENT's
  layout and command stream, available for the 32-head workloads, its
  boundary stated); the unbudgeted default selection, recorded only.
* Totals: ``total_cycles`` = device + priced host; also
  ``total_cycles_measured_host_stage`` with the measured stage median in
  place of the modelled stage (1 cycle = 1 ns at the simulator's declared
  rate).  A minima comparison needs the normal selection and all four
  partition plans; otherwise it is reported incomplete.

The study stays DIAGNOSTIC in the inventory until its boundary is judged
comparable with the frozen paper's row (M-15): the host stage's price is a
declared model beside measured samples, and the CENT control's boundary is
CENT's own.

Usage::

    python -m benchmarks.cent_aim.token_partition --out tp.json [--no-sim]
        [--shapes 1x128,8x512] [--artifacts DIR] [--neighbor 2x256|none]

The workloads are defined here (eager region annotations); the tests in
``tests/pim/test_aim_token_partition.py`` and ``tests/pim/test_host_compute.py``
import them.
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
    AimHostReduction,
    AimHostStage,
    _target_geometry,
    bf16_round,
    reference_outputs,
    reference_partial_sums,
)
from allo.pim.costs import aim_cost
from allo.pim.costs.aim import host_reduction_cycles, host_stage_cycles
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
BUDGET = {"channel": CHANNEL_BUDGET}  # the compiler's resource policy for it
PARTITIONS = (1, 2, 4, 8)
SHAPES = tuple((heads, tokens) for heads in (1, 8, 32) for tokens in (128, 512, 4096))

# The paper's reported outcome of this study (sections/5-evaluation.tex).
PAPER = {
    "minima_reached": 7,
    "workloads": 9,
    "max_gap": 0.0263,
    "head_only_over_selected_1x4096": 1.071,
}


def global_softmax(scores):
    """The bridge's host computation: the global softmax over the token axis
    of the QK scores, in FP32 on the host (row max subtracted, exp, row sum,
    divide), rounded to BF16 for the upload to the SV kernel's GB vector.
    The declared host-time model prices it as four vector passes
    (``allo.pim.costs.aim.HOST_STAGE_VECTOR_PASSES``)."""
    x = np.asarray(scores, dtype=np.float32)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return bf16_round(e / e.sum(axis=-1, keepdims=True))


def build_attention(heads, tokens, head_dim=HEAD_DIM, max_tokens=MAX_TOKENS, *, bridge=True):
    """Resident-input decode attention: scores = q.K, probs = softmax(scores)
    on the host, heads_out = probs.V.

    With ``bridge`` (the default) the host program declares the global
    softmax between the two launches (``allo.host_compute``): the compiler
    orders it, prices its readout of the scores, its upload of the
    probabilities and its host time, checks that the scores are produced
    before it and that the probabilities reach SV's GB preload, and the
    functional path executes it.  ``bridge=False`` is the pre-round-42
    diagnostic program with ``probs`` resident (no bridge), kept for
    comparison only.
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

    if bridge:

        def host(q, k_cache, scores, probs, v_cache, heads_out):
            allo.launch("attention_qk", q, k_cache, scores)
            allo.host_compute("softmax", global_softmax, inputs=(scores,), outputs=(probs,))
            allo.launch("attention_sv", probs, v_cache, heads_out)

        program = allo.host_program(attention, resident=("q", "k_cache", "v_cache"))(host)
        return attention, program

    def host_resident(q, k_cache, scores, probs, v_cache, heads_out):
        allo.launch("attention_qk", q, k_cache, scores)
        allo.launch("attention_sv", probs, v_cache, heads_out)

    program = allo.host_program(
        attention, resident=("q", "k_cache", "probs", "v_cache")
    )(host_resident)
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


def compile_program(attention, program, target, *, layouts=None, budget=BUDGET, virtual=False):
    """One compile of a workload under the budget: the normal selection
    (``layouts`` None) or an explicit layout; ``virtual`` for the estimate."""
    kwargs = dict(host_moves=program, **POLICY)
    if budget is not None:
        kwargs["resources"] = budget
    if layouts is not None:
        kwargs["layout"] = layouts
    if virtual:
        kwargs["backend"] = "virtual"
    return allo.compile(attention, target, aim_cost, **kwargs)


def ranked(heads, tokens, *, qk=True, budget=BUDGET):
    """The shared ranking of one workload under the budget: ``(target,
    region, host program, compiled)``, the compiled object carrying the
    placement ranking."""
    if qk:
        attention, program = build_attention(heads, tokens)
    else:
        attention, program = build_sv(heads, tokens)
    target = build_aim_target()
    compiled = compile_program(attention, program, target, budget=budget, virtual=True)
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
    """The forced alternatives of the ablation from a ranking compiled under
    the budget: the cheapest QK plan, the cheapest SV plan per P and the
    head-only control.  Every scored plan is checked against the budget
    (the compiler's policy already bounded the domains)."""
    choices = kernel_choices(target, compiled)
    qk_index, qk_scores = choices.get("attention_qk", (None, []))
    sv_index, sv_scores = choices["attention_sv"]
    outside = [m for scores in (qk_scores, sv_scores) for _c, m, _p in scores if not within_budget(m, budget)]
    if outside:
        raise RuntimeError(f"ranking scored plans outside the budget: {outside[:3]}")
    qk = min(qk_scores, default=None)
    sv = list(sv_scores)
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


def compile_variant(attention, program, target, variants, sv_choice, budget=BUDGET):
    layouts = [None, None]
    layouts[variants.sv_index] = sv_choice[2]
    if variants.qk is not None:
        layouts[variants.qk_index] = variants.qk[2]
    layouts = [layout for layout in layouts if layout is not None]
    return compile_program(attention, program, target, layouts=layouts, budget=budget)


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
        if isinstance(step, AimHostReduction) and step.source == entry["name"]
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


def _entry(compiled, kernel):
    return next(
        e
        for e in compiled.compiled.layout_ctx.lowering_manifest
        if e.get("kind") == "contraction" and str(e["name"]).startswith(kernel)
    )


def _stage(compiled, name="softmax"):
    return next(
        s for s in compiled.compiled.layout_ctx.host_steps
        if isinstance(s, AimHostStage) and s.stage == name
    )


def functional_check(compiled, heads, tokens, *, seed=0, head_dim=HEAD_DIM):
    """Independent functional evidence for one compiled bridged program:
    random BF16 operands through the lowered QK program's oracle
    (``reference_outputs``: the lowering's ownership and coverage), the
    compiled host stage's own function, and the lowered SV program's oracle
    (``reference_partial_sums`` + the host merge for a split SV,
    ``reference_outputs`` otherwise), against a float64 reference of
    softmax(q.K).V.  No simulator is involved: the timing-only ramulator2
    produces no values and none are attributed to it."""
    qk, sv = _entry(compiled, "attention_qk"), _entry(compiled, "attention_sv")
    stage = _stage(compiled)
    rng = np.random.default_rng(seed + 31 * heads + tokens)
    scale = np.float32(0.1)
    q = bf16_round(rng.standard_normal((heads, head_dim)).astype(np.float32) * scale)
    K = bf16_round(rng.standard_normal((heads, tokens, head_dim)).astype(np.float32) * scale)
    V = bf16_round(rng.standard_normal((heads, head_dim, tokens)).astype(np.float32) * scale)
    replicas = int(qk["replicas"])
    scores = reference_outputs(qk, np.broadcast_to(K[None], (replicas, *K.shape)), np.broadcast_to(q[None], (replicas, *q.shape)))
    probs = stage.run(scores[0])[0]
    split = sv.get("reduction_partition") is not None
    out = reference_outputs(sv, np.broadcast_to(V[None], (int(sv["replicas"]), *V.shape)), np.broadcast_to(probs[None], (int(sv["replicas"]), *probs.shape)))
    reference_scores = np.einsum("hd,hld->hl", q.astype(np.float64), K.astype(np.float64))
    reference_probs = np.exp(reference_scores - reference_scores.max(-1, keepdims=True))
    reference_probs /= reference_probs.sum(-1, keepdims=True)
    reference = np.einsum("hl,hdl->hd", reference_probs, V.astype(np.float64))
    error = float(np.max(np.abs(out[0] - reference)) / np.max(np.abs(reference)))
    tolerance = 2.0**-6
    if not error < tolerance:
        raise AssertionError(f"functional pipeline error {error} exceeds {tolerance}")
    return {
        "evidence": "functional oracle, not the timing-only simulator",
        "pipeline": "reference_outputs(QK) -> stage.run -> " + ("reference_partial_sums(SV) + host merge" if split else "reference_outputs(SV)"),
        "stage": stage.stage,
        "split": split,
        "max_rel_error": error,
        "tolerance": tolerance,
        "seed": int(seed),
    }


def host_stage_samples(compiled, heads, tokens, *, repeats=20, seed=0):
    """Measured host time of the compiled program's stage on representative
    BF16 scores (nanoseconds per call, ``repeats`` samples after one warm-up),
    recorded beside the modelled price and never substituted for it."""
    import time

    stage = _stage(compiled)
    rng = np.random.default_rng(seed)
    scores = bf16_round(rng.standard_normal((heads, tokens)).astype(np.float32))
    stage.run(scores)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        stage.run(scores)
        samples.append(time.perf_counter_ns() - started)
    samples.sort()
    return {
        "unit": "ns per call",
        "repeats": repeats,
        "median_ns": samples[len(samples) // 2],
        "min_ns": samples[0],
        "samples_ns": samples,
        "note": "1 simulated cycle = 1 ns at the simulator's declared rate; not in the priced graph",
    }


def cent_control(heads, tokens, *, simulate, budget=None):
    """The CENT-derived attention program for the same token length, as the
    control the paper's study compared against: ``benchmarks.cent_aim``'s
    ``attention_l<L>`` case (CENT's layout and command stream, K/V cache
    append + QK + SV, softmax outside the device program) compiled through
    the shared pipeline and simulated.  CENT fixes 32 heads and 32 channels
    (4 replicas x 8), so the control exists for the 32-head workloads only
    and its boundary (cache append included, no host stage, whole device) is
    stated, not aligned, with the study's."""
    from benchmarks.cent_aim.workloads import DEFAULT_DECODE_SPEC, build_case

    record = {
        "program": f"cent_aim attention_l{tokens}",
        "boundary": "CENT K/V cache append + QK + SV on 32 channels (4 replicas x 8), softmax outside; not the study's boundary",
    }
    if heads != DEFAULT_DECODE_SPEC.H or tokens not in (128, 512, 4096):
        record["available"] = False
        record["reason"] = f"CENT fixes {DEFAULT_DECODE_SPEC.H} heads and L in (128, 512, 4096)"
        return record
    case = build_case(f"attention_l{tokens}")
    built = allo.compile(case.region, build_aim_target(), aim_cost, host_moves=case.host_program, **POLICY)
    record["available"] = True
    record["modes"] = [p.mode for p in _placements(built)]
    record["command_count"] = len(built.compiled.cmds)
    if simulate:
        result = built.run_backend()
        if result.kind != "simulator-timing-only":
            raise RuntimeError(f"CENT control: expected a simulator-timing-only run, got {result.kind!r}")
        record["device_cycles"] = int(result.cycles)
        record["execution"] = {
            "kind": result.kind,
            "simulator": result.extra.get("simulator"),
            "segments": [{k: v for k, v in seg.items() if k not in ("trace", "stdout")} for seg in result.extra.get("segment_records") or ()],
        }
    return record


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


def compiler_revision():
    """The compiler the plans were compiled with: the package path and the
    checkout's revision and cleanliness (None outside a checkout)."""
    import os
    import subprocess

    package = os.path.dirname(os.path.abspath(allo.__file__))
    record = {"allo": allo.__file__, "git_revision": None, "git_dirty": None}
    try:
        rev = subprocess.run(
            ["git", "-C", package, "rev-parse", "HEAD"], capture_output=True, timeout=20, check=False
        )
        if rev.returncode == 0:
            record["git_revision"] = rev.stdout.decode().strip()
            status = subprocess.run(
                ["git", "-C", package, "status", "--porcelain", "--untracked-files=no"],
                capture_output=True,
                timeout=60,
                check=False,
            )
            record["git_dirty"] = bool(status.stdout.strip())
    except (subprocess.SubprocessError, OSError):
        pass
    return record


def _graph_activities(compiled):
    """The priced graph in full: every activity's primitive, latency,
    occupancies (handle, cycles, amount), dependencies and metadata, with
    the graph's own name and metadata, for reproducible cost accounting."""
    graph = getattr(compiled.compiled, "execution_graph", None)
    if graph is None:
        return None
    activities = []
    for activity in getattr(graph, "activities", ()) or ():
        activities.append(
            {
                "id": getattr(activity, "id", None),
                "label": getattr(activity, "label", None),
                "primitive": getattr(activity, "primitive", None),
                "latency_cycles": getattr(activity, "latency_cycles", None),
                "occupancy": [
                    {
                        "handle": getattr(getattr(use, "handle", None), "name", repr(getattr(use, "handle", None))),
                        "cycles": getattr(use, "cycles", None),
                        "amount": getattr(use, "amount", None),
                    }
                    for use in getattr(activity, "occupancy", ()) or ()
                ],
                "depends_on": list(getattr(activity, "depends_on", ()) or ()),
                "metadata": _json_ready(dict(getattr(activity, "metadata", {}) or {})),
            }
        )
    return {
        "name": getattr(graph, "name", None),
        "activities": activities,
        "metadata": _json_ready(getattr(graph, "metadata", {}) or {}),
    }


def _plan_provenance(compiled):
    """What was compiled: every placement's mode, declared effects and
    rebinding note, the complete lowering manifest (every QK/SV record and
    host step the lowering produced), the host steps and the priced graph."""
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
        "lowering_manifest": _json_ready(list(getattr(ctx, "lowering_manifest", ()) or ())),
        "sv_lowering": _json_ready(sv_entry(compiled)),
        "host_steps": [_json_ready(step) for step in getattr(ctx, "host_steps", ())],
        "resource_budget": getattr(ctx, "resource_budget", None),
        "command_count": len(getattr(compiled.compiled, "cmds", ()) or ()),
        "graph": _graph_activities(compiled),
    }


def _write_artifacts(directory, label, record, result):
    """Write a plan's record, segment traces and simulator output under
    ``directory/label`` before the next plan runs."""
    import json
    import os

    if directory is None:
        return None
    folder = os.path.join(directory, label)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "plan.json"), "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1, default=str)
    if result is not None:
        with open(os.path.join(folder, "stdout.txt"), "w", encoding="utf-8") as handle:
            handle.write(result.stdout or "")
        for index, segment in enumerate(result.extra.get("segment_records") or ()):
            path = os.path.join(folder, f"segment-{index}-{segment['label']}.trace")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("".join(line + "\n" for line in segment["trace"]))
    return folder


class PlanFailure(Exception):
    """A plan that could not be measured; ``record`` is its retained record
    (modes, provenance, what was completed) with a ``failure`` entry."""

    def __init__(self, record):
        super().__init__(record["failure"]["error"])
        self.record = record


def _failure_entry(stage, error):
    import traceback

    entry = {
        "stage": stage,
        "error": f"{type(error).__name__}: {error}",
        "traceback": traceback.format_exc(),
    }
    evidence = getattr(error, "evidence", None)
    if evidence is not None:
        entry["evidence"] = _json_ready(evidence)
    return entry


def measure_plan(label, built, attention, program, target, heads, tokens, *, simulate, budget, artifacts, sv_index=None):
    """Estimate, simulate and check one compiled plan.

    The plan's record (its modes and complete provenance) is established
    before any fallible step; a failure during the estimate, the merge
    check or the simulation is appended to that record as ``failure``
    (with the simulator's structured evidence when it has it), the record
    is written to ``artifacts``, and :class:`PlanFailure` carrying it is
    raised.  A successful plan's record is written too.
    """
    import time

    geometry = _target_geometry(target)
    modes = [p.mode for p in _placements(built)]
    # The SV placement is the one compiled for the SV work item (the bucket
    # index the ranking reported); a single-kernel program has only it.
    sv_mode = modes[sv_index] if sv_index is not None and sv_index < len(modes) else modes[-1]
    record = {
        "label": label,
        "modes": modes,
        "sv_mode": sv_mode,
        "partitions": partitions_of(sv_mode),
        "estimate_cycles": None,
        "host_cycles": None,
        "device_cycles": None,
        "total_cycles": None,
        "merge_check": {"applicable": None, "bf16_error": None, "integer_exact": None},
        "functional_check": None,
        "host_price": None,
        "host_stage_measured": None,
        "total_cycles_measured_host_stage": None,
        "execution": None,
        "provenance": None,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "completed": [],
    }
    result = None
    stage = "provenance"
    try:
        record["provenance"] = _plan_provenance(built)
        record["completed"].append(stage)
        stage = "host_price"
        host_steps = list(built.compiled.layout_ctx.host_steps)
        reduction_cycles = sum(host_reduction_cycles(s, geometry) for s in host_steps if isinstance(s, AimHostReduction))
        stage_cycles = sum(host_stage_cycles(s, geometry) for s in host_steps if isinstance(s, AimHostStage))
        record["host_price"] = {
            "unit": "cycles (1 cycle = 1 ns at the simulator's declared rate)",
            "host_reduction_cycles": reduction_cycles,
            "host_stage_cycles": stage_cycles,
            "host_stages": [s.manifest() for s in host_steps if isinstance(s, AimHostStage)],
            "link_model": "32 B/cycle, zero fixed latency (allo.pim.costs.aim)",
            "no_double_counting": "readout = stage inputs, upload = stage outputs, partial readout = host reduction only",
        }
        record["host_cycles"] = reduction_cycles + stage_cycles
        record["completed"].append(stage)
        stage = "estimate"
        record["estimate_cycles"] = compile_program(
            attention, program, target, layouts=built.compiled.layout, budget=budget, virtual=True
        ).run_backend().cycles
        record["completed"].append(stage)
        stage = "merge_check"
        split = sv_entry(built).get("reduction_partition") is not None
        record["merge_check"]["applicable"] = split
        if split:
            record["merge_check"]["bf16_error"] = check_merge_numerics(built, heads, tokens)
            record["merge_check"]["integer_exact"] = True
        record["completed"].append(stage)
        stage = "functional_check"
        record["functional_check"] = functional_check(built, heads, tokens, seed=heads * 1000 + tokens)
        record["completed"].append(stage)
        stage = "host_samples"
        record["host_stage_measured"] = host_stage_samples(built, heads, tokens)
        record["completed"].append(stage)
        if simulate:
            stage = "simulate"
            result = built.run_backend()
            if result.kind != "simulator-timing-only":
                raise RuntimeError(f"{label}: expected a simulator-timing-only run, got {result.kind!r}")
            if not isinstance(result.cycles, int) or result.cycles <= 0:
                raise RuntimeError(f"{label}: simulator reported cycles {result.cycles!r}")
            record["device_cycles"] = int(result.cycles)
            record["total_cycles"] = int(result.cycles) + record["host_cycles"]
            record["total_cycles_measured_host_stage"] = (
                int(result.cycles)
                + record["host_price"]["host_reduction_cycles"]
                + int(record["host_stage_measured"]["median_ns"])
            )
            record["execution"] = {
                "kind": result.kind,
                "simulator": result.extra.get("simulator"),
                "segments": [
                    {k: v for k, v in segment.items() if k not in ("trace", "stdout")}
                    for segment in result.extra.get("segment_records") or ()
                ],
                "invocations": result.extra.get("invocations"),
                "returncode": result.extra.get("returncode"),
                "resource_budget": result.extra.get("resource_budget"),
            }
            record["completed"].append(stage)
    except Exception as error:
        record["failure"] = _failure_entry(stage, error)
        record["artifacts"] = _write_artifacts(artifacts, label, record, result)
        raise PlanFailure(record) from error
    record["artifacts"] = _write_artifacts(artifacts, label, record, result)
    return record


def run_workload(heads, tokens, *, simulate=True, artifacts=None, budget=BUDGET, neighbor=False):
    """Compile, measure and record every plan of one workload: the normal
    selection under the budget and the forced alternatives.

    Returns a namespace with ``rows`` (label -> row namespace of the plans
    that completed), ``selected`` (the normal selection's P), ``best`` and
    ``gap`` (only when every partition plan and the normal selection
    completed: a partial set is not a minima comparison), the failed plans
    (their records retained) and a JSON-ready ``record``.  Aliases are
    decided before anything runs (the head-only control that is the same
    placement as P1; the fixed eight-way control that is P8 by definition)
    and never stand in for a plan that failed.
    """
    import os

    target, attention, program, ranking = ranked(heads, tokens, budget=budget)
    variants = budget_variants(target, ranking)
    unrestricted = ranked(heads, tokens, budget=None)[3]
    folder = None if artifacts is None else os.path.join(artifacts, f"h{heads}_l{tokens}")
    plans = [("default", None)]
    plans += [(f"P{p}", choice) for p, choice in sorted(variants.per_partition.items())]
    # Established before execution: the head-only control is the P1 plan
    # itself when the ranking's cheapest unsplit plan is the one-head-per-
    # channel mapping; the fixed eight-way control is the P8 plan.
    aliases = {"fixed-8": "P8"}
    if variants.head_only[1] == variants.per_partition[1][1]:
        aliases["head-only"] = "P1"
    else:
        plans.append(("head-only", variants.head_only))
    expected = [label for label, _choice in plans]
    records, failures = {}, {}
    for label, choice in plans:
        try:
            built = (
                compile_program(attention, program, target, budget=budget)
                if choice is None
                else compile_variant(attention, program, target, variants, choice, budget=budget)
            )
        except Exception as error:  # the plan did not compile: no record beyond the failure
            record = {"label": label, "modes": None, "failure": _failure_entry("compile", error)}
            record["artifacts"] = _write_artifacts(folder, label, record, None)
            records[label] = record
            failures[label] = record["failure"]
            continue
        try:
            record = measure_plan(
                label, built, attention, program, target, heads, tokens,
                simulate=simulate, budget=budget, artifacts=folder, sv_index=variants.sv_index,
            )
        except PlanFailure as error:
            records[label] = error.record
            failures[label] = error.record["failure"]
            continue
        if choice is not None:
            record["ranked_cycles"] = choice[0]
        records[label] = record
    try:
        control = cent_control(heads, tokens, simulate=simulate)
    except Exception as error:  # the control's failure is evidence too
        control = {"available": False, "failure": _failure_entry("cent_control", error)}
    if control.get("failure"):
        failures["cent-attention"] = control["failure"]
    rows = {
        label: SimpleNamespace(
            mode=r["sv_mode"], modes=r["modes"], partitions=r["partitions"],
            ranked_cycles=r.get("ranked_cycles"), estimate=r["estimate_cycles"],
            device=r["device_cycles"], host=r["host_cycles"], total=r["total_cycles"],
            bf16_error=r["merge_check"]["bf16_error"], execution=r["execution"], provenance=r["provenance"],
        )
        for label, r in records.items()
        if "failure" not in r
    }
    for alias, source in aliases.items():
        if source in rows:
            rows[alias] = rows[source]
    default = rows.get("default")
    selected = default.partitions if default is not None else None
    required = ["default"] + [f"P{p}" for p in PARTITIONS]
    missing = [label for label in required if label not in rows or (simulate and rows[label].total is None)]
    comparison_complete = not missing
    best = gap = None
    if simulate and comparison_complete:
        measured = [rows[label] for label in required]
        best = min(measured, key=lambda r: r.total)
        gap = default.total / best.total - 1.0
    controls_complete = "head-only" in rows and (not simulate or rows["head-only"].total is not None)
    head_only_ratio = (
        rows["head-only"].total / default.total
        if simulate and controls_complete and default is not None and default.total
        else None
    )
    forced = variants.per_partition.get(selected) if selected is not None else None
    record = {
        "heads": heads,
        "tokens": tokens,
        "resource_budget": budget,
        "compiler": compiler_revision(),
        "default_selection_unrestricted": [p.mode for p in _placements(unrestricted)],
        "expected_plans": expected,
        "selected_partitions": selected,
        "default_equals_forced_choice": None if default is None or forced is None else default.mode == forced[1],
        "comparison": {
            "complete": comparison_complete,
            "required": required,
            "missing": missing,
            "controls_complete": controls_complete,
        },
        "best_partitions": None if best is None else best.partitions,
        "gap": gap,
        "head_only_over_selected": head_only_ratio,
        "plans": records,
        "controls": {"cent_attention": control},
        "neighbor": bool(neighbor),
        "boundary": {
            "device": "AiM ramulator2 memory-system cycles of the complete program (QK, SV)",
            "host": "priced host stage (readout + upload link bytes, declared vector passes) and host reduction (partial readout + adds)",
            "bridge": "scores -> global softmax (host) -> probabilities, declared in the schedule and priced in the selection graph",
            "measured_host": "host-stage samples recorded separately (total_cycles_measured_host_stage)",
        },
        "aliases": aliases,
        "failed_plans": failures,
        "distinct_plans": len({tuple(r["modes"]) for r in records.values() if r.get("modes")}),
        "split_plans_checked": sum(1 for r in records.values() if r.get("merge_check", {}).get("applicable")),
        "artifacts": folder,
    }
    return SimpleNamespace(
        heads=heads, tokens=tokens, rows=rows, selected=selected,
        best=None if best is None else best.partitions, gap=gap,
        comparison_complete=comparison_complete, controls_complete=controls_complete,
        expected=expected, default=record["default_selection_unrestricted"],
        failures=failures, record=record,
    )


def assert_complete(result) -> None:
    """Raise unless every expected plan of ``result`` completed with a
    positive simulated device cycle count, the minima comparison is
    complete and the controls completed; the paper_full test's gate."""
    problems = []
    for label, failure in sorted(result.failures.items()):
        problems.append(f"{label} failed at {failure.get('stage')}: {failure.get('error')}")
    for label in result.expected:
        row = result.rows.get(label)
        if row is None:
            if label not in result.failures:
                problems.append(f"{label} has no record")
        elif not (isinstance(row.device, int) and row.device > 0):
            problems.append(f"{label} has no simulated device cycles ({row.device!r})")
    if not result.comparison_complete:
        problems.append(f"the minima comparison is incomplete: missing {result.record['comparison']['missing']}")
    if not result.controls_complete:
        problems.append("the head-only control did not complete")
    if problems:
        raise AssertionError(
            f"{result.heads} heads x {result.tokens} tokens: " + "; ".join(problems)
        )


def summarize(results):
    """The study's figures over the completed workloads (diagnostic; see
    the module docstring for what the paper's boundary adds)."""
    results = [r for r in results if not r.get("neighbor")]
    complete = [
        r for r in results
        if r.get("gap") is not None and (r.get("comparison") or {}).get("complete", True)
    ]
    incomplete = [
        {
            "heads": r.get("heads"),
            "tokens": r.get("tokens"),
            "missing": (r.get("comparison") or {}).get("missing"),
            "failed_plans": sorted(r.get("failed_plans") or {}),
            "error": r.get("error"),
        }
        for r in results
        if r not in complete
    ]
    reached = sum(r["selected_partitions"] == r["best_partitions"] for r in complete)
    gaps = [r["gap"] for r in complete]
    one_4096 = next((r for r in complete if (r["heads"], r["tokens"]) == (1, 4096)), None)
    return {
        "workloads_completed": len(complete),
        "incomplete_workloads": incomplete,
        "minima_reached": reached,
        "max_gap": max(gaps) if gaps else None,
        "head_only_over_selected_1x4096": (
            one_4096["head_only_over_selected"]
            if one_4096 and (one_4096.get("comparison") or {}).get("controls_complete", True)
            else None
        ),
        "distinct_plans": sum(int(r.get("distinct_plans") or 0) for r in results if "error" not in r),
        "split_plans_checked": sum(int(r.get("split_plans_checked") or 0) for r in results if "error" not in r),
        "failed_plans": sum(len(r.get("failed_plans") or {}) for r in results if "error" not in r),
        "comparable_with_paper": False,
        "not_comparable_because": (
            "no scores-to-softmax-to-probs bridge, no host-stage time, no QK readout / "
            "score upload / partial readout link volumes; see the module docstring"
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
    parser.add_argument("--shapes", default=None, help="subset, e.g. 1x128,8x512 (default: all nine)")
    parser.add_argument("--artifacts", default=None, help="per-plan artifacts directory (default: <out>.artifacts)")
    parser.add_argument("--neighbor", default="2x256", help="a neighbouring supported workload run beside the nine (none to skip)")
    args = parser.parse_args(argv)
    shapes = _parse_shapes(args.shapes) if args.shapes else SHAPES
    neighbors = () if args.neighbor in ("", "none") else _parse_shapes(args.neighbor)
    simulate = not args.no_sim
    artifacts = args.artifacts or (args.out + ".artifacts")
    session = {
        "host": platform.node(),
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.executable,
        "compiler": compiler_revision(),
        "resource_budget": BUDGET,
        "simulate": simulate,
        "artifacts": artifacts,
        "shapes": [list(s) for s in shapes],
        "neighbors": [list(s) for s in neighbors],
        "workloads": [],
    }
    failures = 0
    print(
        "heads tokens | default total(estimate) P | P1 P2 P4 P8 total(estimate) | head-only | "
        "selected P | best P | gap | merge bf16 err | failed plans",
        flush=True,
    )
    for heads, tokens in tuple(shapes) + tuple(neighbors):
        is_neighbor = (heads, tokens) in neighbors and (heads, tokens) not in shapes
        started = time.perf_counter()
        try:
            result = run_workload(heads, tokens, simulate=simulate, artifacts=artifacts, neighbor=is_neighbor)
            record = result.record
            rows = result.rows
            failures += len(result.failures)

            def cell(row):
                if row is None:
                    return "-"
                return f"{row.total if simulate else row.estimate}({row.estimate})"

            totals = " ".join(cell(rows.get(f"P{p}")) for p in PARTITIONS)
            errors = max((r.bf16_error or 0.0) for r in rows.values()) if rows else 0.0
            default = rows.get("default")
            print(
                f"{heads:5d} {tokens:6d} | {cell(default)} P{result.selected} | {totals} | "
                f"{cell(rows.get('head-only'))} | {result.selected} | {result.best} | "
                f"{'-' if result.gap is None else f'{result.gap * 100:.2f}%'} | {errors:.2e} | "
                f"{sorted(result.failures) or '-'}",
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
    session["status"] = "pass" if not failures else f"fail ({failures} plan(s)/workload(s))"
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(session, handle, indent=1, default=str)
    summary = session["summary"]
    print(
        f"normal budgeted selection reaches {summary['minima_reached']} of "
        f"{summary['workloads_completed']} complete simulated comparisons"
        + (f" ({len(summary['incomplete_workloads'])} incomplete)" if summary["incomplete_workloads"] else "")
        + (f"; max gap {summary['max_gap'] * 100:.2f}%" if summary["max_gap"] is not None else "")
        + f"; {summary['distinct_plans']} distinct plans, {summary['split_plans_checked']} split plans "
        f"numerically checked (diagnostic; paper: {PAPER['minima_reached']} of {PAPER['workloads']}, "
        f"{PAPER['max_gap'] * 100:.2f}%, not comparable yet)",
        flush=True,
    )
    print(session["status"], flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
