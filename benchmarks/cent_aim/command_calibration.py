# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Command-level calibration of the AiM contraction profile against the
timing-only simulator, independent of any workload's selection outcome
(the token-partitioning study's one-head misses, design doc section 3a).

For a workload and a set of compared plans, the lowered command stream of
each plan is simulated as a sequence of prefixes (the first k commands
followed by ``EOC``), so that the k-th command's *incremental* memory-system
cycles are the simulator's own measurement of what that command adds under
the commands before it.  The increments are aggregated by command class
(WR_GB, WR_BIAS, MAC_ABK at a reuse-group start, consecutive MAC_ABK,
RD_MAC, AF, RD_AF) and set beside the cost model's charges for the same
classes (``allo.pim.costs.aim._contraction_profile_cycles`` and the profile
it prices from).  Nothing here reads the study's measured totals or the
archived winners; the calibration object is the per-class command behaviour
of the device model.  The output is evidence (JSON with every increment);
no constant is changed by this driver.

Usage::

    python -m benchmarks.cent_aim.command_calibration --heads 1 --tokens 4096
        --plans P1,P8 --out calibration.json
"""

import argparse
import json
import platform
import sys
import time
from types import SimpleNamespace

import allo
from allo import spmw_aim
from allo.pim import aim_lowering
from allo.pim.aim_lowering import _target_geometry
from allo.pim.costs import aim as aim_costs
from allo.pim.costs.aim import contraction_profile_charges
from allo.pim.targets import build_aim_target

from .token_partition import (
    BUDGET,
    PARTITIONS,
    _placements,
    budget_variants,
    compile_variant,
    partitions_of,
    ranked,
)


def _classify(commands):
    """Command classes in order: the opcode, with MAC_ABK split into the
    first MAC after a WR_BIAS run (a reuse-group start) and the rest."""
    classes = []
    previous = None
    for line in commands:
        words = line.split()
        opcode = words[1] if words and words[0] == "AiM" else words[0]
        if opcode == "MAC_ABK" or opcode == "MAC_SBK":
            opcode = f"{opcode}:first" if previous in ("WR_BIAS", "WR_GB") else f"{opcode}:next"
        classes.append(opcode)
        previous = words[1] if words and words[0] == "AiM" else words[0]
    return classes


def _simulate(target, lines):
    probe = SimpleNamespace(target=target, cmds=list(lines), runtime_segments=(), layout_ctx=None)
    result = spmw_aim._run_aim(probe)
    if result.kind != "simulator-timing-only":
        raise RuntimeError(f"expected a simulator-timing-only run, got {result.kind!r}")
    return int(result.cycles), result.extra.get("simulator")


def increments(target, commands):
    """Incremental simulator cycles of every command of ``commands`` (which
    must end with ``AiM EOC``): the k-th prefix's cycles minus the
    (k-1)-th's, the empty prefix (``EOC`` alone) measured explicitly as the
    baseline the first command's increment is taken from.  Prefix deltas are
    contextual completion-time contributions under the commands before
    them, not opcode latencies.  Returns the records, the baseline and the
    executor."""
    body = [line for line in commands if line.strip() != "AiM EOC"]
    records = []
    baseline, executor = _simulate(target, ["AiM EOC"])
    previous = baseline
    for k in range(1, len(body) + 1):
        cycles, executor = _simulate(target, body[:k] + ["AiM EOC"])
        records.append({"index": k - 1, "command": body[k - 1], "prefix_cycles": cycles, "increment": cycles - previous})
        previous = cycles
    return records, baseline, executor


def model_charges(candidate_extra, geometry):
    """The cost model's charges for one contraction candidate by class, from
    the profile it prices (``allo.pim.costs.aim.contraction_profile_charges``)."""
    operation = aim_costs._contraction_record(candidate_extra)
    profile = aim_lowering.aim_contraction_profile(operation, geometry, aim_costs._sbk_banks(candidate_extra, geometry))
    return {
        "profile": {k: v for k, v in profile.items() if k != "counts"} | {"counts": dict(counts := profile["counts"])},
        "charges": contraction_profile_charges(profile),
        "total": aim_costs._contraction_profile_cycles(candidate_extra, geometry),
    }


def calibrate_plan(target, attention, program, variants, choice, label):
    built = compile_variant(attention, program, target, variants, choice, budget=BUDGET)
    manifest = built.compiled.layout_ctx.lowering_manifest
    commands = [str(c) for c in built.compiled.cmds]
    records, baseline, executor = increments(target, commands)
    classes = _classify([r["command"] for r in records])
    by_class = {}
    for record, klass in zip(records, classes):
        record["class"] = klass
        entry = by_class.setdefault(klass, {"count": 0, "simulated_increment_cycles": 0})
        entry["count"] += 1
        entry["simulated_increment_cycles"] += record["increment"]
    kernels = {}
    for entry, placement in zip([e for e in manifest if e.get("kind") == "contraction"], _placements(built)):
        geometry = _target_geometry(target)
        span = entry["command_span"]
        kernel_records = records[span[0] : span[1]]
        kernel_classes = {}
        for r in kernel_records:
            k = kernel_classes.setdefault(r["class"], {"count": 0, "simulated_increment_cycles": 0})
            k["count"] += 1
            k["simulated_increment_cycles"] += r["increment"]
        kernels[entry["name"]] = {
            "mode": placement.mode,
            "command_span": span,
            "simulated_span_cycles": sum(r["increment"] for r in kernel_records),
            "by_class": kernel_classes,
            "model": model_charges(placement.extra, geometry),
        }
    total_cycles = records[-1]["prefix_cycles"] if records else 0
    return {
        "label": label,
        "modes": [p.mode for p in _placements(built)],
        "partitions": partitions_of(choice[1]),
        "commands": len(commands),
        "simulated_cycles": total_cycles,
        "eoc_baseline_cycles": baseline,
        "estimate_cycles": built.estimate().cycles if hasattr(built, "estimate") else None,
        "by_class": by_class,
        "kernels": kernels,
        "increments": records,
        "executor": executor,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--plans", default="P1,P8")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    target = build_aim_target()
    target_, attention, program, ranking = ranked(args.heads, args.tokens, budget=BUDGET)
    variants = budget_variants(target, ranking)
    session = {
        "host": platform.node(),
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.executable,
        "workload": {"heads": args.heads, "tokens": args.tokens},
        "method": "prefix simulation: the k-th command's increment is cycles(prefix k) - cycles(prefix k-1); classes aggregated; model charges from the contraction profile",
        "plans": [],
    }
    failures = 0
    for label in args.plans.split(","):
        p = int(label[1:])
        choice = variants.per_partition.get(p)
        if choice is None:
            session["plans"].append({"label": label, "error": "no budgeted plan with this P"}); failures += 1
            continue
        try:
            plan = calibrate_plan(target, attention, program, variants, choice, label)
        except Exception as error:
            session["plans"].append({"label": label, "error": f"{type(error).__name__}: {error}"}); failures += 1
            continue
        session["plans"].append(plan)
        print(f"== {label} ({plan['modes'][-1][:70]}): simulated {plan['simulated_cycles']} estimate {plan['estimate_cycles']} commands {plan['commands']}")
        for name, kernel in plan["kernels"].items():
            print(f"   {name}: span cycles (simulated increments) {kernel['simulated_span_cycles']} | model total {kernel['model']['total']}")
            for klass, entry in sorted(kernel["by_class"].items()):
                print(f"      {klass:16s} n={entry['count']:4d} simulated {entry['simulated_increment_cycles']:7d}")
            print("      model charges:", kernel["model"]["charges"])
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(session, handle, indent=1, default=str)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(session, handle, indent=1, default=str)
    return 0 if not failures else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
